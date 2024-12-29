import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

class Buffer():
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.

    Now adapted for offline/latent storage:
      - The 'obs' key in the tensordict is actually a latent vector,
        not raw pixels. This is handled in `online_trainer.py` where
        obs is encoded before being stored.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda')
        # Limit capacity by buffer_size or total steps
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage backend.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=int(self.cfg.num_envs / self.cfg.steps_per_update),
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """
        Initialize the replay buffer. Use the first episode (tensordict)
        to estimate storage requirements and decide where to store (CPU/GPU).
        """
        print(f'Buffer capacity: {self._capacity:,}')
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum([
            (v.numel() * v.element_size() if not isinstance(v, TensorDict)
             else sum(x.numel() * x.element_size() for x in v.values()))
            for v in tds.values()
        ]) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        print(f'Storage required: {total_bytes/1e9:.2f} GB')

        # Decide whether to store on CUDA or CPU
        storage_device = 'cuda' if 2.5 * total_bytes < mem_free else 'cpu'
        print(f'Using {storage_device.upper()} memory for storage.')
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        """
        Move tensors to the target device (default: self._device).
        """
        if device is None:
            device = self._device
        return tuple(
            arg.to(device, non_blocking=True) if arg is not None else None
            for arg in args
        )

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` (TensorDict) shape is [T+1, B] after we reshape/permute in `sample()`.

        In the offline/latent setting:
          - td['obs'] is actually latent, shape [T+1, B, latent_dim].
          - td['action'] has shape [T+1, B, action_dim], but we only need [T, B, action_dim].
          - td['reward'] has shape [T+1, B], but we only need [T, B].
        """
        latent = td['obs']                    # shape: [T+1, B, latent_dim]
        action = td['action'][1:]            # [T, B, action_dim]
        reward = td['reward'][1:].unsqueeze(-1)  # [T, B, 1]
        task = td['task'][0] if 'task' in td.keys() else None

        # Move to device
        return self._to_device(latent, action, reward, task)

    def add(self, td):
        """
        Add an episode (or multiple episodes) to the buffer.

        The 'obs' field in `td` is expected to be a latent vector
        (already encoded by the VJEPA encoder) rather than raw pixels.

        For vectorized environment, `td` might have shape [num_envs, ep_len+1, ...].
        """
        for _td in td:
            # Mark the episode index
            _td['episode'] = torch.ones_like(_td['reward'], dtype=torch.int64) * self._num_eps

            # If first episode, init the buffer
            if self._num_eps == 0:
                self._buffer = self._init(_td)

            self._buffer.extend(_td)
            self._num_eps += 1
        return self._num_eps

    def sample(self):
        """
        Sample a batch of subsequences from the buffer, each of length (horizon+1).
        After sampling, we reshape to [N, horizon+1] then permute to [horizon+1, N].
        => final shape: [T+1, B, ...], where T = horizon, B = batch_size.

        Returns:
            (latent, action, reward, task)
        """
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)
