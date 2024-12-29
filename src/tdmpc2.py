import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel

class TDMPC2:
    """
    TD-MPC2 agent for offline/latent usage + MPPI shape mismatch fixes.
    - Offline/latent: we assume obs -> latent outside, or do .encode(obs) if needed.
    - MPPI: carefully handle shape expansions & sums (keepdim=False) to avoid mismatch.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) Create world model (encoder frozen or partial, but we won't backward it)
        self.model = WorldModel(cfg).to(self.device)

        # 2) Optimizers for model/Q and policy (no encoder param if offline)
        self.optim = torch.optim.Adam([
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if cfg.multitask else []}
        ], lr=cfg.lr)
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=cfg.lr, eps=1e-5
        )

        self.model.eval()
        self.scale = RunningScale(cfg)

        # Heuristic more iterations for large action dim
        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)

        # discount factor
        if cfg.multitask:
            self.discount = torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
                device=self.device
            )
        else:
            self.discount = self._get_discount(cfg.episode_length)
        self.cfg.discount = self.discount

    def _get_discount(self, episode_length):
        """
        Heuristic: linearly scale discount w.r.t. episode_length.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1)/frac, self.cfg.discount_min),
            self.cfg.discount_max
        )

    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        state_dict = torch.load(fp) if isinstance(fp, str) else fp
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        If raw obs => do encode => latent => plan or policy.
        If offline-latent usage, you can skip encode here and pass latent directly from trainer.
        """
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)

        # 1) Encode obs => latent (if truly offline-latent, we might skip)
        z = self.model.encode(obs, task)  # shape => [num_envs, latent_dim]

        # 2) If MPC => plan, else => policy
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)]
        return a.cpu()

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        MPPI in latent space. 
        Carefully do shape expansions, using keepdim=False sums to avoid mismatch.
        """
        num_envs = self.cfg.num_eval_envs if eval_mode else self.cfg.num_envs
        horizon = self.cfg.horizon
        num_samples = self.cfg.num_samples
        num_elites = self.cfg.num_elites

        # If desired, sample pi trajectories as part of MPPI
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(num_envs, horizon, self.cfg.num_pi_trajs,
                                     self.cfg.action_dim, device=self.device)
            _z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(horizon - 1):
                pi_actions[:, t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:, t], task)
            pi_actions[:, -1] = self.model.pi(_z, task)[1]

        # expand z => [num_envs, num_samples, latent_dim]
        z = z.unsqueeze(1).repeat(1, num_samples, 1)

        # mean / std => [num_envs, horizon, action_dim]
        mean = torch.zeros(num_envs, horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(num_envs, horizon, self.cfg.action_dim, device=self.device)

        # warm start if not t0
        if not t0 and hasattr(self, '_prev_mean'):
            if eval_mode:
                mean[:, :-1] = self._prev_mean_eval[:, 1:]
            else:
                mean[:, :-1] = self._prev_mean[:, 1:]

        # actions => [num_envs, horizon, num_samples, action_dim]
        actions = torch.empty(num_envs, horizon, num_samples, self.cfg.action_dim,
                              device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :, :self.cfg.num_pi_trajs] = pi_actions

        # MPPI iter
        for _ in range(self.cfg.iterations):
            # sample
            actions[:, :, self.cfg.num_pi_trajs:] = (
                mean.unsqueeze(2)
                + std.unsqueeze(2) * torch.randn(
                    num_envs, horizon,
                    num_samples - self.cfg.num_pi_trajs,
                    self.cfg.action_dim, device=std.device
                )
            ).clamp(-1, 1)

            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # evaluate => shape [num_envs, num_samples, 1]
            value = self._estimate_value(z, actions, task).nan_to_num_(0)

            # shape fix => val_2d => [num_envs, num_samples]
            # remove trailing dims if present
            while value.dim() > 2 and value.shape[-1] == 1:
                value = value.squeeze(-1)

            val_2d = value if value.dim() == 2 else value.squeeze(-1)

            # top-k => [num_envs, num_elites]
            elite_idxs = torch.topk(val_2d, num_elites, dim=1).indices
            if elite_idxs.dim() == 3 and elite_idxs.shape[-1] == 1:
                elite_idxs = elite_idxs.squeeze(-1)

            # gather elite_value => shape [num_envs, num_elites, 1] or [num_envs,num_elites]
            row_idx = torch.arange(num_envs, device=self.device).unsqueeze(1)

            if value.dim() == 2:
                value_3d = value.unsqueeze(-1)  # => [num_envs, num_samples, 1]
            else:
                value_3d = value

            elite_value = value_3d[row_idx, elite_idxs]
            if elite_value.dim() == 2:
                elite_value = elite_value.unsqueeze(-1)

            # gather elite actions => [num_envs, horizon, num_elites, action_dim]
            idx_for_actions = elite_idxs.unsqueeze(1).unsqueeze(3).expand(
                -1, horizon, -1, self.cfg.action_dim
            )
            elite_actions = torch.gather(actions, 2, idx_for_actions)

            # compute score => shape [num_envs, num_elites, 1] => expand to [num_envs,horizon,num_elites,1]
            max_value = elite_value.max(dim=1, keepdim=True)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            # sum over dim=1 => keepdim=False => shape [num_envs, ...]
            sum_score_2d = score.sum(dim=1, keepdim=False)  # => [num_envs, 1] or [num_envs,]
            score = score / (sum_score_2d.unsqueeze(-1) + 1e-9)  # => [num_envs,num_elites,1]
            score_4d = score.unsqueeze(1).expand(-1, horizon, -1, 1)  # => [num_envs,horizon,num_elites,1]

            # weighted => [num_envs,horizon,num_elites,action_dim]
            weighted_actions = score_4d * elite_actions
            # sum => [num_envs,horizon,action_dim]
            sum_score_3d = score_4d.sum(dim=2, keepdim=False)  # => [num_envs,horizon,1]
            mean = weighted_actions.sum(dim=2) / (sum_score_3d + 1e-9)

            var = (score_4d * (elite_actions - mean.unsqueeze(2))**2).sum(dim=2) / (sum_score_3d + 1e-9)
            std = torch.sqrt(var).clamp_(self.cfg.min_std, self.cfg.max_std)

            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # pick final => [num_envs,horizon,action_dim]
        score_2d = score.squeeze(-1).cpu().numpy()  # => [num_envs, num_elites]
        out_actions = torch.zeros(num_envs, horizon, self.cfg.action_dim, device=self.device)
        for i in range(num_envs):
            idx = np.random.choice(np.arange(score_2d.shape[1]), p=score_2d[i])
            out_actions[i] = elite_actions[i, :, idx]

        if eval_mode:
            self._prev_mean_eval = mean
        else:
            self._prev_mean = mean

        a, std_0 = out_actions[:, 0], std[:, 0]
        if not eval_mode:
            a += std_0 * torch.randn(num_envs, self.cfg.action_dim, device=std_0.device)
        return a.clamp_(-1, 1)

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """
        Evaluate candidate action sequences in latent space.
         z => [num_envs, num_samples, latent_dim]
         actions => [num_envs, horizon, num_samples, action_dim]
        returns => [num_envs, num_samples, 1]
        """
        G, discount = 0, 1
        horizon = self.cfg.horizon
        for t in range(horizon):
            at = actions[:, t]  # => [num_envs, num_samples, action_dim]
            reward = math.two_hot_inv(self.model.reward(z, at, task), self.cfg)
            z = self.model.next(z, at, task)
            G += discount * reward
            if self.cfg.multitask:
                # discount must handle array-like tasks
                discount *= self.discount[torch.tensor(task, device=z.device)]
            else:
                discount *= self.discount

        # final Q => [num_envs,num_samples]
        pi_act = self.model.pi(z, task)[1]
        Qz = self.model.Q(z, pi_act, task, return_type='avg')
        return (G + discount * Qz).unsqueeze(-1)  # => [num_envs,num_samples,1]

    def update_pi(self, zs, task):
        """
        Separate policy backward => uses zs.detach() externally
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_latent, reward, task):
        """
        TD-target = reward + discount * Q(next_latent, pi(next_latent)).
        next_latent => [T,B,latent_dim]
        reward => [T,B,1]
        """
        pi_next = self.model.pi(next_latent, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_latent, pi_next, task, return_type='min', target=True)

    def update(self, buffer):
        """
        1) sample offline-latent => (latent, action, reward, task)
        2) model+Q => total_loss => backward
        3) policy => second backward => zs.detach()
        """
        latent, action, reward, task = buffer.sample()
        # shapes: 
        #  latent => [T+1,B,latent_dim]
        #  action => [T,B,action_dim]
        #  reward => [T,B,1]

        with torch.no_grad():
            next_latent = latent[1:]      # => [T,B,latent_dim]
            td_targets = self._td_target(next_latent, reward, task)

        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # latent rollout => consistency + reward + Q
        horizon = self.cfg.horizon
        zs = torch.empty(horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = latent[0]  # [B,latent_dim]
        zs[0] = z
        consistency_loss = 0.

        for t in range(horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_latent[t]) * (self.cfg.rho**t)
            zs[t+1] = z

        # Q + reward => [T,B]
        _zs = zs[:-1]  # [T,B,latent_dim]
        qs_all = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)

        reward_loss, value_loss = 0., 0.
        for t in range(horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * (self.cfg.rho**t)
            for q_i in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs_all[q_i][t], td_targets[t], self.cfg).mean() * (self.cfg.rho**t)

        consistency_loss *= (1/horizon)
        reward_loss *= (1/horizon)
        value_loss *= (1/(horizon*self.cfg.num_q))

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )

        # first backward => model/Q
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # second backward => policy => use zs.detach()
        pi_loss = self.update_pi(zs.detach(), task)

        # soft update target Q
        self.model.soft_update_target_Q()

        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
        }
