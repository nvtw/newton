# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from .kernels import (
    compute_gae_kernel,
    compute_vtrace_returns_kernel,
    gather_trajectory_minibatch_kernel,
    gaussian_entropy_kernel,
    mirror_2d_kernel,
    mirrored_action_mse_grad_kernel,
    mirrored_action_mse_loss_kernel,
    normalize_kernel,
    ppo_actor_loss_backward_kernel,
    ppo_actor_loss_kernel,
    scatter_trajectory_ratios_kernel,
    scatter_trajectory_values_kernel,
    sum_and_sumsq_kernel,
    trajectory_priority_kernel,
    value_loss_kernel,
    value_symmetry_loss_kernel,
    weight_trajectory_advantages_kernel,
    zero_scalar_kernel,
)
from .networks import GaussianActor, WarpMLP
from .optim import Adam


@dataclass(frozen=True)
class MirrorMapPPO:
    """Observation/action reflection map for PPO symmetry regularization.

    Args:
        obs_src: Source observation index for each mirrored observation column.
        obs_sign: Sign multiplier for each mirrored observation column.
        action_src: Source action index for each mirrored action column.
        action_sign: Sign multiplier for each mirrored action column.
    """

    obs_src: tuple[int, ...]
    obs_sign: tuple[float, ...]
    action_src: tuple[int, ...]
    action_sign: tuple[float, ...]


@dataclass
class ConfigPPO:
    """Configuration for :class:`TrainerPPO`.

    Args:
        gamma: Discount factor.
        gae_lambda: Generalized advantage estimation trace decay.
        clip_ratio: PPO policy-ratio clipping threshold.
        entropy_coeff: Entropy bonus coefficient.
        actor_lr: Actor Adam learning rate.
        critic_lr: Critic Adam learning rate.
        train_epochs: Full-buffer optimization epochs per rollout when replay
            minibatches are disabled.
        minibatch_size: Optional trajectory-minibatch size. A value less than or
            equal to zero disables minibatch replay.
        replay_ratio: Number of sampled transition updates per collected
            transition when trajectory minibatches are enabled.
        priority_alpha: Trajectory priority exponent. A value less than or
            equal to zero uses uniform trajectory sampling.
        priority_beta: Importance-correction exponent for priority replay. A
            value less than or equal to zero disables the correction.
        manual_actor_backward: Use a hand-written Gaussian PPO loss backward
            kernel and seed the actor MLP gradients directly. This avoids Warp
            Tape through the log-probability, entropy, and PPO loss kernels.
        vtrace_rho_clip: V-trace policy-ratio clip for replayed trajectories.
            A value less than or equal to zero disables V-trace recomputation.
        vtrace_c_clip: V-trace trace-ratio clip for replayed trajectories.
            A value less than or equal to zero disables V-trace recomputation.
        normalize_advantages: Whether to normalize advantages in-place before
            updating. With minibatch replay this normalizes each sampled
            minibatch.
        reward_clip: Absolute reward clamp used before advantage/return
            computation. A value less than or equal to zero disables clipping.
        max_grad_norm: Global gradient-norm clipping threshold for actor and
            critic optimizers. A value less than or equal to zero disables clipping.
        mirror_loss_coeff: Coefficient for optional mirror-symmetry MSE on
            policy means and value predictions. Requires a mirror map.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 1.0e-3
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    train_epochs: int = 4
    minibatch_size: int = 0
    replay_ratio: float = 0.0
    priority_alpha: float = 0.0
    priority_beta: float = 0.0
    manual_actor_backward: bool = False
    vtrace_rho_clip: float = 0.0
    vtrace_c_clip: float = 0.0
    normalize_advantages: bool = True
    reward_clip: float = 0.0
    max_grad_norm: float = 0.0
    mirror_loss_coeff: float = 0.0


@dataclass
class StatsPPOUpdate:
    """Scalar diagnostics from a PPO update."""

    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float


class BatchPPO:
    """Sampled transition batch used by :class:`TrainerPPO`."""

    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: wp.context.Devicelike = None,
    ):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = wp.get_device(device)
        if self.num_steps <= 0 or self.num_envs <= 0:
            raise ValueError("num_steps and num_envs must be positive")

        count = self.num_steps * self.num_envs
        self.obs = wp.zeros((count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((count, self.action_dim), dtype=wp.float32, device=self.device)
        self.old_log_probs = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.advantages = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.returns = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.ratios = wp.ones(count, dtype=wp.float32, device=self.device)
        self.priority_weights = wp.ones(self.num_envs, dtype=wp.float32, device=self.device)

    @property
    def num_samples(self) -> int:
        """Number of transition rows in the batch."""

        return self.num_steps * self.num_envs

    def normalize_advantages(self, eps: float = 1.0e-8) -> None:
        """Normalize advantages in place."""

        _normalize_advantages(self.advantages, self.num_samples, self.device, eps=eps)


class BufferRollout:
    """On-policy storage for vectorized PPO rollouts.

    Args:
        num_steps: Number of policy steps per environment.
        num_envs: Number of parallel worlds/environments.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        device: Warp device.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: wp.context.Devicelike = None,
    ):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = wp.get_device(device)
        if self.num_steps <= 0 or self.num_envs <= 0:
            raise ValueError("num_steps and num_envs must be positive")

        count = self.num_steps * self.num_envs
        self.obs = wp.zeros((count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((count, self.action_dim), dtype=wp.float32, device=self.device)
        self.old_log_probs = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.successes = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.values = wp.zeros((self.num_steps + 1) * self.num_envs, dtype=wp.float32, device=self.device)
        self.advantages = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.returns = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.ratios = wp.ones(count, dtype=wp.float32, device=self.device)

    @property
    def num_samples(self) -> int:
        """Number of transition rows in the buffer."""

        return self.num_steps * self.num_envs

    def compute_returns(self, *, gamma: float, gae_lambda: float, reward_clip: float = 0.0) -> None:
        """Compute GAE advantages and returns in place."""

        self.ratios.fill_(1.0)
        wp.launch(
            compute_gae_kernel,
            dim=self.num_envs,
            inputs=[
                self.rewards,
                self.dones,
                self.values,
                self.num_steps,
                self.num_envs,
                float(gamma),
                float(gae_lambda),
                float(reward_clip),
            ],
            outputs=[self.advantages, self.returns],
            device=self.device,
        )

    def normalize_advantages(self, eps: float = 1.0e-8) -> None:
        """Normalize advantages in place."""

        _normalize_advantages(self.advantages, self.num_samples, self.device, eps=eps)

    def compute_vtrace_returns(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        rho_clip: float,
        c_clip: float,
        reward_clip: float = 0.0,
    ) -> None:
        """Compute V-trace advantages and returns in place."""

        wp.launch(
            compute_vtrace_returns_kernel,
            dim=self.num_envs,
            inputs=[
                self.rewards,
                self.dones,
                self.values,
                self.ratios,
                self.num_steps,
                self.num_envs,
                float(gamma),
                float(gae_lambda),
                float(rho_clip),
                float(c_clip),
                float(reward_clip),
            ],
            outputs=[self.advantages, self.returns],
            device=self.device,
        )


class TrainerPPO:
    """Pure-Warp PPO trainer for continuous actions.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_layers: Hidden layer widths for actor and critic.
        config: PPO hyperparameters.
        device: Warp device.
        seed: Initializer seed.
        squash_actions: Whether the actor uses tanh-squashed actions.
        activation: Hidden-layer activation.
        log_std_init: Initial actor log-standard-deviation.
        mirror_map: Optional observation/action reflection map used when
            ``config.mirror_loss_coeff`` is positive.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] = (64, 64),
        config: ConfigPPO | None = None,
        device: wp.context.Devicelike = None,
        seed: int = 0,
        squash_actions: bool = True,
        activation: str = "tanh",
        log_std_init: float = 0.0,
        mirror_map: MirrorMapPPO | None = None,
    ):
        self.config = config or ConfigPPO()
        self.seed = int(seed)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.activation = activation
        self.squash_actions = bool(squash_actions)
        self.log_std_init = float(log_std_init)
        self.device = wp.get_device(device)
        self.actor = GaussianActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=self.hidden_layers,
            activation=activation,
            state_dependent_std=False,
            log_std_init=log_std_init,
            squash=squash_actions,
            device=self.device,
            seed=seed,
        )
        self.critic = WarpMLP(
            (self.obs_dim, *self.hidden_layers, 1),
            activation=activation,
            output_activation="linear",
            device=self.device,
            seed=seed + 1,
        )
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.config.actor_lr, max_grad_norm=self.config.max_grad_norm
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.config.critic_lr, max_grad_norm=self.config.max_grad_norm
        )
        self.iteration = 0
        self.mirror_map: MirrorMapPPO | None = None
        self._mirror_obs_src: wp.array[wp.int32] | None = None
        self._mirror_obs_sign: wp.array[wp.float32] | None = None
        self._mirror_action_src: wp.array[wp.int32] | None = None
        self._mirror_action_sign: wp.array[wp.float32] | None = None
        self._mirror_obs: wp.array2d[wp.float32] | None = None
        self._minibatch: BatchPPO | None = None
        self._minibatch_env_ids: wp.array[wp.int32] | None = None
        self._trajectory_priorities: wp.array[wp.float32] | None = None
        self._actor_policy_out_grad: wp.array2d[wp.float32] | None = None
        self._actor_log_std_grad = wp.zeros_like(self.actor.log_std, requires_grad=False)
        if mirror_map is not None:
            self.set_mirror_map(mirror_map)

        self._policy_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._value_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._approx_kl = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._clip_fraction = wp.zeros(1, dtype=wp.float32, device=self.device)

    def set_mirror_map(self, mirror_map: MirrorMapPPO | None) -> None:
        """Set or clear the optional PPO symmetry map."""

        if mirror_map is None:
            self.mirror_map = None
            self._mirror_obs_src = None
            self._mirror_obs_sign = None
            self._mirror_action_src = None
            self._mirror_action_sign = None
            self._mirror_obs = None
            return
        _validate_mirror_map(mirror_map, self.obs_dim, self.action_dim)
        self.mirror_map = mirror_map
        self._mirror_obs_src = wp.array(
            np.asarray(mirror_map.obs_src, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._mirror_obs_sign = wp.array(
            np.asarray(mirror_map.obs_sign, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._mirror_action_src = wp.array(
            np.asarray(mirror_map.action_src, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._mirror_action_sign = wp.array(
            np.asarray(mirror_map.action_sign, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._mirror_obs = None

    def save_checkpoint(self, path: str | Path, *, iteration: int | None = None) -> None:
        """Save network parameters and optimizer state to a NumPy archive.

        Args:
            path: Output ``.npz`` path.
            iteration: Optional training iteration stored as metadata.
        """

        data: dict[str, np.ndarray] = {
            "obs_dim": np.asarray(self.obs_dim, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "hidden_layers": np.asarray(self.hidden_layers, dtype=np.int64),
            "activation": np.asarray(self.activation),
            "squash_actions": np.asarray(int(self.squash_actions), dtype=np.int64),
            "log_std_init": np.asarray(self.log_std_init, dtype=np.float32),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "iteration": np.asarray(self.iteration if iteration is None else int(iteration), dtype=np.int64),
        }
        for key, value in asdict(self.config).items():
            data[f"config_{key}"] = np.asarray(value)
        _pack_mlp(data, "actor", self.actor.net)
        data["actor_log_std"] = self.actor.log_std.numpy()
        _pack_mlp(data, "critic", self.critic)
        _pack_adam(data, "actor_optimizer", self.actor_optimizer)
        _pack_adam(data, "critic_optimizer", self.critic_optimizer)
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, **data)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        config: ConfigPPO | None = None,
        device: wp.context.Devicelike = None,
    ) -> TrainerPPO:
        """Load a PPO trainer from a checkpoint archive.

        Args:
            path: Input ``.npz`` path.
            config: Optional PPO config overriding the saved optimizer settings.
            device: Warp device for restored arrays.

        Returns:
            Restored trainer.
        """

        with np.load(Path(path), allow_pickle=False) as data:
            saved_config = config or _config_from_checkpoint(data)
            trainer = cls(
                obs_dim=int(data["obs_dim"]),
                action_dim=int(data["action_dim"]),
                hidden_layers=tuple(int(width) for width in data["hidden_layers"]),
                config=saved_config,
                device=device,
                seed=int(data["seed"]) if "seed" in data else 0,
                squash_actions=bool(int(data["squash_actions"])),
                activation=str(data["activation"].item()),
                log_std_init=float(data["log_std_init"]),
            )
            _unpack_mlp(data, "actor", trainer.actor.net)
            trainer.actor.log_std.assign(data["actor_log_std"])
            _unpack_mlp(data, "critic", trainer.critic)
            _unpack_adam(data, "actor_optimizer", trainer.actor_optimizer)
            _unpack_adam(data, "critic_optimizer", trainer.critic_optimizer)
            if "iteration" in data:
                trainer.iteration = max(int(data["iteration"]), 0)
        return trainer

    def act(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array2d[wp.float32]]:
        """Sample actions and evaluate values for rollout collection.

        Args:
            obs: Observation batch.
            seed: Sampling seed.
            deterministic: Use deterministic policy means.

        Returns:
            Tuple ``(actions, log_probs, values)``.
        """

        actions, log_probs, _policy_out = self.actor.sample(
            obs, seed=seed, deterministic=deterministic, requires_grad=False
        )
        values = self.critic.forward(obs, requires_grad=False)
        return actions, log_probs, values

    def update(self, buffer: BufferRollout) -> StatsPPOUpdate:
        """Update actor and critic from a finished rollout buffer."""

        if buffer.obs_dim != self.obs_dim or buffer.action_dim != self.action_dim:
            raise ValueError("BufferRollout dimensions do not match trainer dimensions")
        if self._uses_minibatch_replay(buffer):
            return self._update_minibatches(buffer)

        if self.config.normalize_advantages:
            buffer.normalize_advantages()

        policy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clip_fraction = 0.0
        train_epochs = int(self.config.train_epochs)
        for epoch in range(train_epochs):
            read_stats = epoch == train_epochs - 1
            policy_loss, approx_kl, clip_fraction = self._update_actor(buffer, read_stats=read_stats)
            value_loss = self._update_critic(buffer, read_stats=read_stats)
        return StatsPPOUpdate(
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    def _uses_minibatch_replay(self, buffer: BufferRollout) -> bool:
        minibatch_size = int(self.config.minibatch_size)
        return minibatch_size > 0 and float(self.config.replay_ratio) > 0.0 and minibatch_size < buffer.num_samples

    def _update_minibatches(self, buffer: BufferRollout) -> StatsPPOUpdate:
        minibatch_size = int(self.config.minibatch_size)
        if minibatch_size % buffer.num_steps != 0:
            raise ValueError("PPO trajectory minibatch_size must be a multiple of rollout steps")
        segment_count = minibatch_size // buffer.num_steps
        if segment_count <= 0:
            raise ValueError("PPO trajectory minibatch_size is smaller than one rollout trajectory")
        batch = self._ensure_minibatch(buffer, segment_count)
        num_minibatches = max(
            1, int(float(self.config.replay_ratio) * float(buffer.num_samples) / float(minibatch_size))
        )

        rng = np.random.default_rng(self.seed + 1000003 * self.iteration)
        use_vtrace = self._uses_vtrace_replay()
        probabilities = None if use_vtrace else self._trajectory_sampling_probabilities(buffer)
        max_cols = max(self.obs_dim, self.action_dim, 1)
        policy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clip_fraction = 0.0
        for minibatch_id in range(num_minibatches):
            read_stats = minibatch_id == num_minibatches - 1
            if use_vtrace:
                self._compute_vtrace_returns(buffer)
                probabilities = self._trajectory_sampling_probabilities(buffer)
            if probabilities is None:
                env_ids = rng.integers(0, buffer.num_envs, size=segment_count, dtype=np.int32)
            else:
                env_ids = rng.choice(buffer.num_envs, size=segment_count, replace=True, p=probabilities).astype(
                    np.int32, copy=False
                )
            self._minibatch_env_ids.assign(env_ids)
            wp.launch(
                gather_trajectory_minibatch_kernel,
                dim=(batch.num_samples, max_cols),
                inputs=[
                    self._minibatch_env_ids,
                    buffer.num_envs,
                    segment_count,
                    self.obs_dim,
                    self.action_dim,
                    buffer.obs,
                    buffer.actions,
                    buffer.old_log_probs,
                    buffer.advantages,
                    buffer.returns,
                ],
                outputs=[batch.obs, batch.actions, batch.old_log_probs, batch.advantages, batch.returns],
                device=self.device,
            )
            if self.config.normalize_advantages:
                batch.normalize_advantages()
            self._weight_minibatch_advantages(batch, env_ids, probabilities, buffer.num_envs)
            policy_loss, approx_kl, clip_fraction = self._update_actor(batch, read_stats=read_stats)
            if use_vtrace:
                self._scatter_minibatch_ratios(buffer, batch, segment_count)
            value_loss = self._update_critic(batch, read_stats=read_stats)
            if use_vtrace:
                self._scatter_minibatch_values(buffer, batch, segment_count)

        return StatsPPOUpdate(
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    def _uses_vtrace_replay(self) -> bool:
        return float(self.config.vtrace_rho_clip) > 0.0 and float(self.config.vtrace_c_clip) > 0.0

    def _compute_vtrace_returns(self, buffer: BufferRollout) -> None:
        buffer.compute_vtrace_returns(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            rho_clip=self.config.vtrace_rho_clip,
            c_clip=self.config.vtrace_c_clip,
            reward_clip=self.config.reward_clip,
        )

    def _weight_minibatch_advantages(
        self, batch: BatchPPO, env_ids: np.ndarray, probabilities: np.ndarray | None, total_envs: int
    ) -> None:
        priority_beta = float(self.config.priority_beta)
        if priority_beta <= 0.0 or probabilities is None:
            return
        weights = np.power(np.maximum(float(total_envs) * probabilities[env_ids], 1.0e-6), -priority_beta).astype(
            np.float32, copy=False
        )
        batch.priority_weights.assign(weights)
        wp.launch(
            weight_trajectory_advantages_kernel,
            dim=batch.num_samples,
            inputs=[batch.priority_weights, batch.num_envs],
            outputs=[batch.advantages],
            device=self.device,
        )

    def _trajectory_sampling_probabilities(self, buffer: BufferRollout) -> np.ndarray | None:
        priority_alpha = float(self.config.priority_alpha)
        if priority_alpha <= 0.0:
            return None
        if self._trajectory_priorities is None or int(self._trajectory_priorities.shape[0]) != buffer.num_envs:
            self._trajectory_priorities = wp.zeros(buffer.num_envs, dtype=wp.float32, device=self.device)
        wp.launch(
            trajectory_priority_kernel,
            dim=buffer.num_envs,
            inputs=[buffer.advantages, buffer.num_steps, buffer.num_envs],
            outputs=[self._trajectory_priorities],
            device=self.device,
        )
        priorities = self._trajectory_priorities.numpy().astype(np.float64, copy=False)
        weights = np.power(np.maximum(priorities, 0.0) + 1.0e-6, priority_alpha)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return None
        return weights / total

    def _scatter_minibatch_ratios(self, buffer: BufferRollout, batch: BatchPPO, segment_count: int) -> None:
        wp.launch(
            scatter_trajectory_ratios_kernel,
            dim=batch.num_samples,
            inputs=[self._minibatch_env_ids, buffer.num_envs, segment_count, batch.ratios],
            outputs=[buffer.ratios],
            device=self.device,
        )

    def _scatter_minibatch_values(self, buffer: BufferRollout, batch: BatchPPO, segment_count: int) -> None:
        values = self.critic.forward(batch.obs, requires_grad=False)
        wp.launch(
            scatter_trajectory_values_kernel,
            dim=batch.num_samples,
            inputs=[self._minibatch_env_ids, buffer.num_envs, segment_count, values],
            outputs=[buffer.values],
            device=self.device,
        )

    def _ensure_minibatch(self, buffer: BufferRollout, segment_count: int) -> BatchPPO:
        if (
            self._minibatch is None
            or self._minibatch.num_steps != buffer.num_steps
            or self._minibatch.num_envs != int(segment_count)
        ):
            self._minibatch = BatchPPO(
                num_steps=buffer.num_steps,
                num_envs=int(segment_count),
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )
            self._minibatch_env_ids = wp.zeros(int(segment_count), dtype=wp.int32, device=self.device)
        return self._minibatch

    def _update_actor(self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True) -> tuple[float, float, float]:
        if self.config.manual_actor_backward:
            return self._update_actor_manual(buffer, read_stats=read_stats)
        return self._update_actor_tape(buffer, read_stats=read_stats)

    def _ensure_actor_backward_buffers(self, rows: int, cols: int) -> wp.array2d[wp.float32]:
        if (
            self._actor_policy_out_grad is None
            or int(self._actor_policy_out_grad.shape[0]) != int(rows)
            or int(self._actor_policy_out_grad.shape[1]) != int(cols)
        ):
            self._actor_policy_out_grad = wp.zeros(
                (int(rows), int(cols)), dtype=wp.float32, device=self.device, requires_grad=False
            )
        return self._actor_policy_out_grad

    def _update_actor_manual(
        self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True
    ) -> tuple[float, float, float]:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_policy_out = None
        if mirror_obs is not None:
            mirror_policy_out = self.actor.forward(mirror_obs, requires_grad=False)

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._policy_loss], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._approx_kl], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._clip_fraction], device=self.device)
        self._actor_log_std_grad.zero_()
        with wp.Tape() as tape:
            policy_out = self.actor.forward(buffer.obs, requires_grad=True)
        policy_out_grad = self._ensure_actor_backward_buffers(buffer.num_samples, int(policy_out.shape[1]))
        wp.launch(
            ppo_actor_loss_backward_kernel,
            dim=buffer.num_samples,
            inputs=[
                policy_out,
                self.actor.log_std,
                buffer.actions,
                buffer.old_log_probs,
                buffer.advantages,
                self.config.clip_ratio,
                self.config.entropy_coeff,
                self.action_dim,
                int(self.actor.state_dependent_std),
                int(self.actor.squash),
                self.actor.log_std_min,
                self.actor.log_std_max,
                buffer.num_samples,
            ],
            outputs=[
                self._policy_loss,
                self._approx_kl,
                self._clip_fraction,
                buffer.ratios,
                policy_out_grad,
                self._actor_log_std_grad,
            ],
            device=self.device,
        )
        if mirror_policy_out is not None:
            wp.launch(
                mirrored_action_mse_grad_kernel,
                dim=(buffer.num_samples, self.action_dim),
                inputs=[
                    policy_out,
                    mirror_policy_out,
                    self._mirror_action_src,
                    self._mirror_action_sign,
                    self.action_dim,
                    self.config.mirror_loss_coeff,
                    buffer.num_samples,
                ],
                outputs=[policy_out_grad, self._policy_loss],
                device=self.device,
            )
        tape.backward(grads={policy_out: policy_out_grad, self.actor.log_std: self._actor_log_std_grad})
        if read_stats:
            loss = float(self._policy_loss.numpy()[0])
            kl = float(self._approx_kl.numpy()[0])
            clip_fraction = float(self._clip_fraction.numpy()[0])
        else:
            loss = 0.0
            kl = 0.0
            clip_fraction = 0.0
        self.actor_optimizer.step()
        tape.zero()
        return loss, kl, clip_fraction

    def _update_actor_tape(
        self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True
    ) -> tuple[float, float, float]:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_policy_out = None
        if mirror_obs is not None:
            mirror_policy_out = self.actor.forward(mirror_obs, requires_grad=False)

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._policy_loss], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._approx_kl], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._clip_fraction], device=self.device)
        with wp.Tape() as tape:
            policy_out, new_log_probs = self.actor.log_prob(buffer.obs, buffer.actions, requires_grad=True)
            entropy = wp.empty(buffer.num_samples, dtype=wp.float32, device=self.device, requires_grad=True)
            wp.launch(
                gaussian_entropy_kernel,
                dim=buffer.num_samples,
                inputs=[
                    policy_out,
                    self.actor.log_std,
                    self.action_dim,
                    int(self.actor.state_dependent_std),
                    self.actor.log_std_min,
                    self.actor.log_std_max,
                ],
                outputs=[entropy],
                device=self.device,
            )
            wp.launch(
                ppo_actor_loss_kernel,
                dim=buffer.num_samples,
                inputs=[
                    new_log_probs,
                    buffer.old_log_probs,
                    buffer.advantages,
                    entropy,
                    self.config.clip_ratio,
                    self.config.entropy_coeff,
                    buffer.num_samples,
                ],
                outputs=[self._policy_loss, self._approx_kl, self._clip_fraction, buffer.ratios],
                device=self.device,
            )
            if mirror_policy_out is not None:
                wp.launch(
                    mirrored_action_mse_loss_kernel,
                    dim=(buffer.num_samples, self.action_dim),
                    inputs=[
                        policy_out,
                        mirror_policy_out,
                        self._mirror_action_src,
                        self._mirror_action_sign,
                        self.action_dim,
                        self.config.mirror_loss_coeff,
                        buffer.num_samples,
                    ],
                    outputs=[self._policy_loss],
                    device=self.device,
                )
        tape.backward(self._policy_loss)
        if read_stats:
            loss = float(self._policy_loss.numpy()[0])
            kl = float(self._approx_kl.numpy()[0])
            clip_fraction = float(self._clip_fraction.numpy()[0])
        else:
            loss = 0.0
            kl = 0.0
            clip_fraction = 0.0
        self.actor_optimizer.step()
        tape.zero()
        return loss, kl, clip_fraction

    def _update_critic(self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True) -> float:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_values = None
        if mirror_obs is not None:
            mirror_values = self.critic.forward(mirror_obs, requires_grad=False)

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._value_loss], device=self.device)
        with wp.Tape() as tape:
            values = self.critic.forward(buffer.obs, requires_grad=True)
            wp.launch(
                value_loss_kernel,
                dim=buffer.num_samples,
                inputs=[values, buffer.returns, buffer.num_samples],
                outputs=[self._value_loss],
                device=self.device,
            )
            if mirror_values is not None:
                wp.launch(
                    value_symmetry_loss_kernel,
                    dim=buffer.num_samples,
                    inputs=[values, mirror_values, self.config.mirror_loss_coeff, buffer.num_samples],
                    outputs=[self._value_loss],
                    device=self.device,
                )
        tape.backward(self._value_loss)
        loss = float(self._value_loss.numpy()[0]) if read_stats else 0.0
        self.critic_optimizer.step()
        tape.zero()
        return loss

    def _mirrored_obs(self, buffer: BufferRollout | BatchPPO) -> wp.array2d[wp.float32] | None:
        if self.config.mirror_loss_coeff <= 0.0:
            return None
        if self.mirror_map is None:
            raise ValueError("PPO mirror_loss_coeff requires a mirror map")
        if self._mirror_obs is None or int(self._mirror_obs.shape[0]) != buffer.num_samples:
            self._mirror_obs = wp.empty(
                (buffer.num_samples, self.obs_dim), dtype=wp.float32, device=self.device, requires_grad=False
            )
        wp.launch(
            mirror_2d_kernel,
            dim=(buffer.num_samples, self.obs_dim),
            inputs=[buffer.obs, self._mirror_obs_src, self._mirror_obs_sign],
            outputs=[self._mirror_obs],
            device=self.device,
        )
        return self._mirror_obs


def _normalize_advantages(
    advantages: wp.array[wp.float32], count: int, device: wp.context.Device, *, eps: float
) -> None:
    stats = wp.zeros(2, dtype=wp.float32, device=device)
    wp.launch(
        sum_and_sumsq_kernel,
        dim=count,
        inputs=[advantages, count],
        outputs=[stats],
        device=device,
    )
    stats_np = stats.numpy()
    mean = float(stats_np[0]) / float(count)
    var = max(float(stats_np[1]) / float(count) - mean * mean, 0.0)
    inv_std = 1.0 / np.sqrt(var + float(eps))
    wp.launch(
        normalize_kernel,
        dim=count,
        inputs=[advantages, mean, float(inv_std), count],
        device=device,
    )


def save_ppo_checkpoint(trainer: TrainerPPO, path: str | Path, *, iteration: int | None = None) -> None:
    """Save a PPO trainer checkpoint.

    Args:
        trainer: Trainer to save.
        path: Output ``.npz`` path.
        iteration: Optional training iteration stored as metadata.
    """

    trainer.save_checkpoint(path, iteration=iteration)


def load_ppo_checkpoint(
    path: str | Path,
    *,
    config: ConfigPPO | None = None,
    device: wp.context.Devicelike = None,
) -> TrainerPPO:
    """Load a PPO trainer checkpoint.

    Args:
        path: Input ``.npz`` path.
        config: Optional PPO config overriding saved settings.
        device: Warp device for restored arrays.

    Returns:
        Restored trainer.
    """

    return TrainerPPO.load_checkpoint(path, config=config, device=device)


def _validate_mirror_map(mirror_map: MirrorMapPPO, obs_dim: int, action_dim: int) -> None:
    obs_src = np.asarray(mirror_map.obs_src, dtype=np.int64)
    obs_sign = np.asarray(mirror_map.obs_sign, dtype=np.float32)
    action_src = np.asarray(mirror_map.action_src, dtype=np.int64)
    action_sign = np.asarray(mirror_map.action_sign, dtype=np.float32)
    if obs_src.shape != (int(obs_dim),) or obs_sign.shape != (int(obs_dim),):
        raise ValueError(f"Observation mirror map must have length {obs_dim}")
    if action_src.shape != (int(action_dim),) or action_sign.shape != (int(action_dim),):
        raise ValueError(f"Action mirror map must have length {action_dim}")
    if np.any(obs_src < 0) or np.any(obs_src >= int(obs_dim)):
        raise ValueError("Observation mirror source indices are out of range")
    if np.any(action_src < 0) or np.any(action_src >= int(action_dim)):
        raise ValueError("Action mirror source indices are out of range")
    if not np.isfinite(obs_sign).all() or not np.isfinite(action_sign).all():
        raise ValueError("Mirror signs must be finite")
    if not _is_signed_involution(obs_src, obs_sign):
        raise ValueError("Observation mirror map must be a signed involution")
    if not _is_signed_involution(action_src, action_sign):
        raise ValueError("Action mirror map must be a signed involution")


def _is_signed_involution(src: np.ndarray, sign: np.ndarray) -> bool:
    for i, j in enumerate(src):
        if int(src[int(j)]) != int(i):
            return False
        if not np.isclose(float(sign[i]) * float(sign[int(j)]), 1.0, rtol=0.0, atol=1.0e-6):
            return False
    return True


def _pack_mlp(data: dict[str, np.ndarray], prefix: str, mlp: WarpMLP) -> None:
    data[f"{prefix}_layer_count"] = np.asarray(len(mlp.weights), dtype=np.int64)
    for index, (weight, bias) in enumerate(zip(mlp.weights, mlp.biases, strict=True)):
        data[f"{prefix}_weight_{index}"] = weight.numpy()
        data[f"{prefix}_bias_{index}"] = bias.numpy()


def _unpack_mlp(data: np.lib.npyio.NpzFile, prefix: str, mlp: WarpMLP) -> None:
    layer_count = int(data[f"{prefix}_layer_count"])
    if layer_count != len(mlp.weights):
        raise ValueError(f"Checkpoint {prefix} layer count does not match trainer")
    for index, (weight, bias) in enumerate(zip(mlp.weights, mlp.biases, strict=True)):
        weight.assign(data[f"{prefix}_weight_{index}"])
        bias.assign(data[f"{prefix}_bias_{index}"])


def _pack_adam(data: dict[str, np.ndarray], prefix: str, optimizer: Adam) -> None:
    data[f"{prefix}_step_count"] = np.asarray(optimizer.step_count, dtype=np.int64)
    data[f"{prefix}_state_count"] = np.asarray(len(optimizer.m), dtype=np.int64)
    for index, (m, v) in enumerate(zip(optimizer.m, optimizer.v, strict=True)):
        data[f"{prefix}_m_{index}"] = m.numpy()
        data[f"{prefix}_v_{index}"] = v.numpy()


def _unpack_adam(data: np.lib.npyio.NpzFile, prefix: str, optimizer: Adam) -> None:
    state_count = int(data[f"{prefix}_state_count"])
    if state_count != len(optimizer.m):
        raise ValueError(f"Checkpoint {prefix} state count does not match optimizer")
    optimizer.step_count = int(data[f"{prefix}_step_count"])
    for index, (m, v) in enumerate(zip(optimizer.m, optimizer.v, strict=True)):
        m.assign(data[f"{prefix}_m_{index}"])
        v.assign(data[f"{prefix}_v_{index}"])


def _config_from_checkpoint(data: np.lib.npyio.NpzFile) -> ConfigPPO:
    values = {}
    for field in ConfigPPO.__dataclass_fields__:
        key = f"config_{field}"
        if key in data:
            value = data[key].item()
            if field == "normalize_advantages":
                value = bool(value)
            values[field] = value
    return ConfigPPO(**values)
