# online_trainer.py
from collections import defaultdict
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer

class OnlineTrainer(Trainer):
    """
    Trainer class for single-task online TD-MPC2 training,
    now adapted to offline (latent) usage.

    Key points:
      - We do 'agent.model.encode(obs)' each time we get new obs,
        store the resulting latent in _tds / buffer,
      - agent.act(...) also expects latent, not raw obs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def final_info_metrics(self, info):
        metrics = dict()
        if self.cfg.env_type == 'gpu':
            for k, v in info["final_info"]["episode"].items():
                metrics[k] = v.float().mean().item()
        else:
            temp = defaultdict(list)
            for final_info in info["final_info"]:
                for k, v in final_info["episode"].items():
                    temp[k].append(v)
            for k, v in temp.items():
                metrics[k] = np.mean(v)
        return metrics

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """
        Evaluate a TD-MPC2 agent, in offline-encoder scenario:
         - We still do agent.model.encode(obs) to get latent,
           then feed that latent to agent.act(...).
        """
        has_success, has_fail = False, False
        for i in range(self.cfg.eval_episodes_per_env):
            obs, _ = self.eval_env.reset()
            done = torch.full((self.cfg.num_eval_envs,), False, device=obs.device)
            t = 0
            while not done[0]:
                # 1) encode obs => latent
                latent_obs = self.agent.model.encode(obs).detach()
                action = self.agent.act(latent_obs, t0=(t == 0), eval_mode=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated | truncated
                t += 1

        eval_metrics = dict()
        eval_metrics.update(self.final_info_metrics(info))
        return eval_metrics

    def to_td(self, latent_obs, num_envs, action=None, reward=None):
        """
        Creates a TensorDict for one step, storing 'latent_obs' instead of raw obs.
        latent_obs shape: [num_envs, latent_dim].
        We'll store it under key 'obs' in tensordict,
        but note that it's actually latent, not raw obs.
        """
        # shape adjustments
        if latent_obs.ndim == 2:
            # [num_envs, latent_dim] => add time dim => [num_envs, 1, latent_dim]
            latent_obs = latent_obs.unsqueeze(1).cpu()
        elif latent_obs.ndim == 1:
            # fallback => [1,1,latent_dim]
            latent_obs = latent_obs.unsqueeze(0).unsqueeze(0).cpu()
        else:
            latent_obs = latent_obs.cpu()

        if action is None:
            action = torch.full((num_envs, self.cfg.action_dim), float('nan'))
        if reward is None:
            reward = torch.full((num_envs,), float('nan'))

        td = TensorDict(
            source={
                'obs': latent_obs,                 # store latent
                'action': action.unsqueeze(1).cpu(),
                'reward': reward.unsqueeze(1).cpu()
            },
            batch_size=(num_envs, 1)
        )
        return td

    def train(self):
        """
        Train a TD-MPC2 agent with offline/latent approach.
        We store & pass around latent, not raw obs, in the buffer.
        """
        train_metrics, time_metrics = {}, {}
        vec_done, eval_next = [True], True
        seed_finish = False
        rollout_times = []

        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq < self.cfg.num_envs:
                eval_next = True

            # Reset environment if done
            if vec_done[0]:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    self.logger.log_wandb_video(self._step)
                    eval_next = False

                if self._step > 0:
                    # Concat entire episode => [num_envs, ep_len+1, ...]
                    tds = torch.cat(self._tds, dim=1)
                    train_metrics.update(self.final_info_metrics(vec_info))

                    if seed_finish:
                        time_metrics.update(
                            rollout_time=np.mean(rollout_times),
                            rollout_fps=self.cfg.num_envs / np.mean(rollout_times),
                            update_time=update_time,
                        )
                        time_metrics.update(self.common_metrics())
                        self.logger.log(time_metrics, 'time')
                        rollout_times = []

                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')

                    assert len(self._tds) == self.env.max_episode_steps + 1, \
                        f"Episode length mismatch. got {len(self._tds)} in _tds, expected {self.env.max_episode_steps+1}."

                    # Add entire episode to replay buffer
                    self._ep_idx = self.buffer.add(tds)

                # Reset env
                obs, _ = self.env.reset()
                # 1) encode offline => get latent
                latent_obs = self.agent.model.encode(obs).detach()
                self._tds = [self.to_td(latent_obs, self.cfg.num_envs)]

            # Collect experience
            rollout_time = time()

            # decide action
            if self._step > self.cfg.seed_steps:
                # get last latent => shape [num_envs,1,latent_dim]
                # convert to => [num_envs, latent_dim]
                latent_obs = self._tds[-1]['obs'].squeeze(1).to(self.agent.device)
                action = self.agent.act(latent_obs, t0=(len(self._tds) == 1), eval_mode=False)
            else:
                action = torch.from_numpy(self.env.action_space.sample())

            # env step
            obs_next, reward, vec_terminated, vec_truncated, vec_info = self.env.step(action)
            vec_done = vec_terminated | vec_truncated

            if vec_done[0]:
                if self.cfg.obs == 'rgb':
                    obs_next[:, -3:, ...] = vec_info["final_observation"]['sensor_data']['base_camera']['rgb'].permute(0,3,1,2)
                else:
                    obs_next = vec_info["final_observation"]

            # 2) encode next obs => latent => store
            latent_next = self.agent.model.encode(obs_next).detach()
            self._tds.append(self.to_td(latent_next, self.cfg.num_envs, action, reward))

            rollout_time = time() - rollout_time
            rollout_times.append(rollout_time)

            # Update agent
            if self._step >= self.cfg.seed_steps:
                update_time = time()
                if not seed_finish:
                    seed_finish = True
                    num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))

                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)
                update_time = time() - update_time

            obs = obs_next
            self._step += self.cfg.num_envs

        self.logger.finish(self.agent)
        print('\nTraining completed successfully')
