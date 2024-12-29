from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn as nn

from vjepa_encoder.vision_encoder import JepaEncoder
from common import layers, math, init

class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Now adapted for offline usage (encoder can be frozen).
    If trainer passes raw obs => we do pixel->latent,
    If trainer passes latent => we skip the pixel pipeline.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # (Optional) multi-task
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.

        # -----------------------------
        # 1) V-JEPA Encoder + Frozen
        # -----------------------------
        #  - If offline usage, we often skip calling this if input is already latent.
        #  - If raw obs passed in, we do pixel->latent.

        self._encoder = JepaEncoder.load_model(
            "/home/tdmpc2-jepa/tdmpc2-jepa/src/jepa-encoder/params-encoder.yaml"
        )
        self._encoder.to("cuda")

        # Freeze encoder to avoid backprop overhead
        for param in self._encoder.parameters():
            param.requires_grad = False

        # V-JEPA 輸出: [N, 196, 1024]
        patch_dim = 1024
        reduced_patch_dim = 32
        flattened_dim = 196 * reduced_patch_dim

        # 先把 [196,1024] => [196,reduced_patch_dim], flatten => [196*reduced_patch_dim] => latent_dim
        self.patch_fc = nn.Sequential(
            nn.Linear(patch_dim, reduced_patch_dim),
            nn.ReLU()
        )
        self.global_fc = nn.Sequential(
            nn.Linear(flattened_dim, cfg.latent_dim),
            nn.ReLU()
        )

        # -----------------------------
        # 2) Dynamics / Reward / Policy / Q
        # -----------------------------
        self._dynamics = layers.mlp(
            cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0),
            2 * [cfg.mlp_dim],
            cfg.latent_dim,
            act=layers.SimNorm(cfg)
        )
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0),
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1)
        )
        self._pi = layers.mlp(
            cfg.latent_dim + (cfg.task_dim if cfg.multitask else 0),
            2 * [cfg.mlp_dim],
            2 * cfg.action_dim
        )
        self._Qs = layers.Ensemble([
            layers.mlp(
                cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0),
                2 * [cfg.mlp_dim],
                max(cfg.num_bins, 1),
                dropout=cfg.dropout
            )
            for _ in range(cfg.num_q)
        ])

        modules_to_initialize = [self._dynamics, self._reward, self._pi, self._Qs]
        for module in modules_to_initialize:
            module.apply(init.weight_init)

        # Initialize final layers to zero
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])

        # Target Q
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

        # Log std range for policy
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    @property
    def total_params(self):
        """
        Count total trainable parameters (excluding the frozen encoder).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Override `to` to also move extra buffers to device.
        """
        super().to(*args, **kwargs)
        if self.cfg.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Keep target Q in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enable/disable gradient for Q-nets (and optional task_emb).
        Useful in policy update step.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.cfg.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Polyak averaging for target Q-nets.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)

    def task_emb(self, x, task):
        """
        For multi-task usage: concat task embedding to x.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            # e.g. [T, B, latent_dim]
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task=None, obs_type='single'):
        """
        Offline usage scenario:
          - If obs is already latent shape => skip pixel pipeline
          - Else do pixel pipeline
        obs_type='single' or 'multi' controls frames reshape logic.
        """
        # 0) If obs already in latent shape => skip
        #   e.g. shape => [batch_size, latent_dim], and user might call encode() by mistake
        if obs.ndim == 2 and obs.shape[-1] == self.cfg.latent_dim:
            # Already latent, just return it
            return obs

        # 1) Multi-task embedding if needed (usually for state-based obs, not pixel),
        #    but let's keep it. If raw pixel => we typically not do this, but no harm:
        if self.cfg.multitask and task is not None:
            obs = self.task_emb(obs, task)

        # 2) shape check if obs is raw pixel => e.g. [batch_size, 9, H, W]
        if obs_type == 'single':
            # check if obs.shape[1] == 9 => (3 frames * 3 channels) or 3 => (single frame, 3 channel) ...
            # for simplicity, assume [batch_size, 9, H, W]
            batch_size = obs.shape[0]
            if obs.shape[1] != 9:
                raise RuntimeError(
                    f"encode(obs): expected obs.shape[1]==9 for 'single', got {obs.shape[1]}"
                )
            # reshape => [batch_size, 3, 3, H, W]
            obs = obs.reshape(batch_size, 3, 3, obs.shape[-2], obs.shape[-1])
            # flatten => [batch_size*3, 3, H, W]
            obs = obs.reshape(batch_size * 3, 3, obs.shape[-2], obs.shape[-1])
        else:
            # obs_type == 'multi'
            # obs shape: [T, batch_size, 9, H, W] => reorder => [batch_size, T, 9, H, W]
            T = obs.shape[0]
            batch_size = obs.shape[1]
            if obs.shape[2] != 9:
                raise RuntimeError(
                    f"encode(obs): expected obs.shape[2]==9 for 'multi', got {obs.shape[2]}"
                )
            obs = obs.permute(1, 0, 2, 3, 4)  # => [batch_size, T, 9, H, W]
            obs = obs.reshape(batch_size, T, 3, 3, obs.shape[-2], obs.shape[-1])
            obs = obs.reshape(batch_size * T * 3, 3, obs.shape[-2], obs.shape[-1])

        # 3) pixel normalize + resize
        obs = obs.float() / 255.0
        obs = torch.nn.functional.interpolate(obs, size=(224, 224), mode='bilinear', align_corners=False)

        # 4) pass V-JEPA encoder => shape [N, 196, 1024]
        encoder_output = self._encoder.embed_image(obs)  # => [N,196,1024], N depends on batch_size * frames

        # 5) flatten patch => patch_fc => global_fc
        # single vs multi
        if obs_type == 'single':
            # N = batch_size*3
            # => reshape => [batch_size, 3, 196, 1024] => average frames => [batch_size,196,1024]
            batch_size = obs.shape[0] // 3
            encoder_output = encoder_output.reshape(batch_size, 3, 196, 1024)
            encoder_output = encoder_output.mean(dim=1)  # => [batch_size,196,1024]
            # flatten => pass patch_fc => global_fc
            encoder_output = encoder_output.reshape(batch_size * 196, 1024)
            encoder_output = self.patch_fc(encoder_output)  # => [batch_size*196, reduced_patch_dim]
            encoder_output = encoder_output.reshape(batch_size, -1)
            latent = self.global_fc(encoder_output)  # => [batch_size, latent_dim]
        else:
            # multi => N= batch_size * T * 3
            # => reshape => [batch_size,T,3,196,1024], average frames => [batch_size,T,196,1024]
            # => permute => [T,batch_size,196,1024]
            T = obs.shape[0] // 3 // (obs.shape[1] if obs.ndim>1 else 1)  # or re-calc T carefully
            # safer approach => we stored T in local var above
            # let's recast:
            # for sure: encoder_output.shape[0] = batch_size*T*3
            # so batch_size was known
            # we do:
            bT3 = encoder_output.shape[0]
            # bT3 = batch_size*T*3
            batch_size_times_T = bT3 // 3
            encoder_output = encoder_output.reshape(batch_size_times_T, 3, 196, 1024)
            encoder_output = encoder_output.mean(dim=1)  # => [batch_size_times_T,196,1024]
            # now => [batch_size_times_T,196,1024]
            # if we want final shape => [T,batch_size, ...], we do:
            # but let's do patch_fc => global_fc first
            encoder_output = encoder_output.reshape(batch_size_times_T * 196, 1024)
            encoder_output = self.patch_fc(encoder_output)  # => [batch_size_times_T*196, reduced_patch_dim]
            encoder_output = encoder_output.reshape(batch_size_times_T, -1)
            latent = self.global_fc(encoder_output)  # => [batch_size_times_T, latent_dim]

            # if we want [T,batch_size,latent_dim], we do:
            # but let's see how your original code wanted it:
            # originally => latent = latent.reshape(T,batch_size,-1)
            # so we can do:
            latent = latent.reshape(T, batch_size, -1)

        return latent

    # ----------------------------------
    # Next/Rew/Policy/Q same as baseline
    # ----------------------------------
    def next(self, z, a, task):
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        za = torch.cat([z, a], dim=-1)
        return self._dynamics(za)

    def reward(self, z, a, task):
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        za = torch.cat([z, a], dim=-1)
        return self._reward(za)

    def pi(self, z, task):
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)
        if self.cfg.multitask:
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False):
        assert return_type in {'min', 'avg', 'all'}
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        za = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(za)
        if return_type == 'all':
            return out

        idxs = np.random.choice(self.cfg.num_q, 2, replace=False)
        Q1, Q2 = out[idxs[0]], out[idxs[1]]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        if return_type == 'min':
            return torch.min(Q1, Q2)
        else:
            return (Q1 + Q2) / 2
