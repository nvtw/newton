# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import (
    amp_finalize_step_kernel,
    amp_reset_found_inf_kernel,
    amp_update_scale_kernel,
    concat_obs_action_kernel,
    fill_eps_kernel,
    fill_eps_seed_counter_kernel,
    normalize_observations_kernel,
    observation_count_update_kernel,
    observation_moments_partial_kernel,
    observation_moments_update_kernel,
    pack_sac_update_stats_kernel,
    replay_sample_kernel,
    replay_store_kernel,
    sac_actor_policy_backward_device_alpha_kernel,
    sac_actor_q_backward_kernel,
    sac_alpha_loss_kernel,
    sac_critic_loss_backward_kernel,
    sac_critic_loss_kernel,
    sac_distributional_actor_q_backward_kernel,
    sac_distributional_critic_loss_backward_kernel,
    sac_distributional_min_projection_device_alpha_kernel,
    sac_distributional_projection_device_alpha_kernel,
    sac_distributional_q_value_kernel,
    sac_q_target_device_alpha_kernel,
    sac_refresh_alpha_kernel,
    sample_gaussian_actions_kernel,
    scale_2d_in_place_kernel,
    zero_scalar_kernel,
)
from .networks import GaussianActor, WarpMLP
from .optim import Adam


@wp.kernel
def _concat_rows_kernel(
    first: wp.array2d[wp.float32],
    second: wp.array2d[wp.float32],
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    if row < first.shape[0]:
        out[row, col] = first[row, col]
    else:
        out[row, col] = second[row - first.shape[0], col]


@wp.kernel
def _copy_rows_kernel(
    source: wp.array2d[wp.float32],
    source_offset: wp.int32,
    destination_offset: wp.int32,
    destination: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    destination[row + destination_offset, col] = source[row + source_offset, col]


@dataclass
class ConfigSAC:
    """Configuration for :class:`TrainerSAC`.

    Args:
        gamma: Discount factor.
        n_step: Number of transitions accumulated into replay returns.
        tau: Target critic Polyak update factor.
        actor_lr: Actor Adam learning rate.
        critic_lr: Critic Adam learning rate.
        alpha_lr: Entropy-temperature Adam learning rate.
        initial_alpha: Initial entropy-temperature value.
        auto_alpha: Whether to tune entropy temperature automatically.
        target_entropy: Target action entropy. ``None`` uses ``-action_dim``.
        update_steps: Critic gradient updates per call to :meth:`TrainerSAC.update`.
        policy_frequency: Number of critic updates between actor updates.
        average_critics: Average double-Q estimates instead of taking their minimum.
        distributional_critic: Learn categorical return distributions instead of scalar Q-values.
        distributional_atoms: Number of categorical return atoms.
        distributional_v_min: Minimum categorical return support.
        distributional_v_max: Maximum categorical return support.
        distributional_min_target: Select the lower-expected-value target critic distribution.
        normalize_observations: Normalize observations with automatic running statistics.
        observation_clip: Absolute clipping bound after observation normalization.
    """

    gamma: float = 0.99
    n_step: int = 1
    tau: float = 5.0e-3
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    initial_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None
    update_steps: int = 1
    policy_frequency: int = 1
    average_critics: bool = False
    distributional_critic: bool = False
    distributional_atoms: int = 101
    distributional_v_min: float = -20.0
    distributional_v_max: float = 20.0
    distributional_min_target: bool = False
    normalize_observations: bool = True
    observation_clip: float = 10.0


@dataclass
class StatsSACUpdate:
    """Scalar diagnostics from a SAC update."""

    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float


@dataclass
class BatchSAC:
    """Transition batch used by :class:`TrainerSAC`.

    Args:
        obs: Observation batch.
        actions: Action batch.
        rewards: Reward batch.
        dones: Terminal flags where ``1`` means terminal.
        next_obs: Next-observation batch.
    """

    obs: wp.array
    actions: wp.array
    rewards: wp.array
    dones: wp.array
    next_obs: wp.array

    @property
    def batch_size(self) -> int:
        """Number of transitions in the batch."""

        return int(self.rewards.shape[0])


class BufferReplaySAC:
    """Fixed-size Warp replay buffer for SAC.

    Args:
        capacity: Maximum number of transitions.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        batch_size: Default sampled batch size.
        device: Warp device.
    """

    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        batch_size: int,
        device: wp.context.Devicelike = None,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.device = wp.get_device(device)
        if self.capacity <= 0 or self.batch_size <= 0:
            raise ValueError("capacity and batch_size must be positive")
        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.obs = wp.zeros((self.capacity, self.obs_dim), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((self.capacity, self.action_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.next_obs = wp.zeros((self.capacity, self.obs_dim), dtype=wp.float32, device=self.device)
        self.size = 0
        self.position = 0

    def add_batch(
        self,
        obs: wp.array,
        actions: wp.array,
        rewards: wp.array,
        dones: wp.array,
        next_obs: wp.array,
    ) -> None:
        """Append a batch of transitions, overwriting old rows if needed."""

        row_count = int(rewards.shape[0])
        if row_count <= 0:
            return
        if row_count > self.capacity:
            raise ValueError("Cannot add more transitions than replay capacity in one call")
        if int(obs.shape[1]) != self.obs_dim or int(next_obs.shape[1]) != self.obs_dim:
            raise ValueError("Observation dimensions do not match replay buffer")
        if int(actions.shape[1]) != self.action_dim:
            raise ValueError("Action dimensions do not match replay buffer")

        max_cols = max(self.obs_dim, self.action_dim, 1)
        wp.launch(
            replay_store_kernel,
            dim=(row_count, max_cols),
            inputs=[
                obs,
                actions,
                rewards,
                dones,
                next_obs,
                self.position,
                self.capacity,
                row_count,
                self.obs_dim,
                self.action_dim,
            ],
            outputs=[self.obs, self.actions, self.rewards, self.dones, self.next_obs],
            device=self.device,
        )
        self.position = (self.position + row_count) % self.capacity
        self.size = min(self.capacity, self.size + row_count)

    def sample(self, *, seed: int, batch_size: int | None = None) -> BatchSAC:
        """Sample a random transition batch on the Warp device."""

        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        n = int(batch_size) if batch_size is not None else self.batch_size
        if n <= 0:
            raise ValueError("batch_size must be positive")

        obs = wp.empty((n, self.obs_dim), dtype=wp.float32, device=self.device)
        actions = wp.empty((n, self.action_dim), dtype=wp.float32, device=self.device)
        rewards = wp.empty(n, dtype=wp.float32, device=self.device)
        dones = wp.empty(n, dtype=wp.float32, device=self.device)
        next_obs = wp.empty((n, self.obs_dim), dtype=wp.float32, device=self.device)
        max_cols = max(self.obs_dim, self.action_dim, 1)
        wp.launch(
            replay_sample_kernel,
            dim=(n, max_cols),
            inputs=[
                self.obs,
                self.actions,
                self.rewards,
                self.dones,
                self.next_obs,
                self.size,
                int(seed),
                self.obs_dim,
                self.action_dim,
            ],
            outputs=[obs, actions, rewards, dones, next_obs],
            device=self.device,
        )
        return BatchSAC(obs=obs, actions=actions, rewards=rewards, dones=dones, next_obs=next_obs)


class TrainerSAC:
    """Pure-Warp soft actor-critic trainer for continuous actions.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_layers: Hidden layer widths for the actor and, by default, critics.
        critic_hidden_layers: Optional hidden layer widths for both critics.
        config: SAC hyperparameters.
        device: Warp device.
        seed: Initializer and update seed.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] = (256, 256),
        critic_hidden_layers: tuple[int, ...] | None = None,
        config: ConfigSAC | None = None,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ):
        self.config = config or ConfigSAC()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = wp.get_device(device)
        self.seed = int(seed)
        self._update_count = 0
        self._gradient_update_count = 0
        if self.config.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")
        if self.config.n_step < 1:
            raise ValueError("n_step must be positive")
        if self.config.update_steps < 1:
            raise ValueError("update_steps must be positive")
        if self.config.policy_frequency < 1:
            raise ValueError("policy_frequency must be positive")
        if self.config.distributional_atoms < 2:
            raise ValueError("distributional_atoms must be at least 2")
        if self.config.distributional_v_max <= self.config.distributional_v_min:
            raise ValueError("distributional_v_max must be greater than distributional_v_min")
        self.target_entropy = (
            -float(self.action_dim) if self.config.target_entropy is None else float(self.config.target_entropy)
        )

        self.actor = GaussianActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=hidden_layers,
            activation="relu",
            state_dependent_std=True,
            squash=True,
            device=self.device,
            seed=self.seed,
        )
        q_input_dim = self.obs_dim + self.action_dim
        q_output_dim = self.config.distributional_atoms if self.config.distributional_critic else 1
        critic_layers = hidden_layers if critic_hidden_layers is None else critic_hidden_layers
        self.critic1 = WarpMLP(
            (q_input_dim, *critic_layers, q_output_dim), activation="relu", device=self.device, seed=seed + 1
        )
        self.critic2 = WarpMLP(
            (q_input_dim, *critic_layers, q_output_dim), activation="relu", device=self.device, seed=seed + 2
        )
        self.target_critic1 = WarpMLP(
            (q_input_dim, *critic_layers, q_output_dim), activation="relu", device=self.device, seed=seed + 3
        )
        self.target_critic2 = WarpMLP(
            (q_input_dim, *critic_layers, q_output_dim), activation="relu", device=self.device, seed=seed + 4
        )
        self.target_critic1.copy_from(self.critic1)
        self.target_critic2.copy_from(self.critic2)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.config.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.config.critic_lr)
        self.log_alpha = wp.array(
            np.array([np.log(float(self.config.initial_alpha))], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)

        self._actor_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._critic_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._alpha_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._alpha = wp.array([float(self.config.initial_alpha)], dtype=wp.float32, device=self.device)
        self._loss_scale = wp.ones(1, dtype=wp.float32, device=self.device)
        self._update_stats = wp.empty(4, dtype=wp.float32, device=self.device)
        self._update_stats_host = wp.empty(4, dtype=wp.float32, device="cpu", pinned=self.device.is_cuda)
        self._obs_mean = wp.zeros(self.obs_dim, dtype=wp.float32, device=self.device)
        self._obs_m2 = wp.ones(self.obs_dim, dtype=wp.float32, device=self.device)
        self._obs_count = wp.array([2.0], dtype=wp.float32, device=self.device)
        self._obs_moment_partial_count = 128
        self._obs_sums = wp.empty((self._obs_moment_partial_count, self.obs_dim), dtype=wp.float32, device=self.device)
        self._obs_sums_sq = wp.empty_like(self._obs_sums)

    @property
    def alpha(self) -> float:
        """Current SAC entropy-temperature value."""

        return float(self._alpha.numpy()[0])

    def act(
        self,
        obs: wp.array,
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array, wp.array]:
        """Sample actions for environment interaction."""

        normalized_obs = self._normalize_observations(obs)
        actions, log_probs, _policy_out = self.actor.sample(
            normalized_obs, seed=seed, deterministic=deterministic, requires_grad=False
        )
        return actions, log_probs

    def copy_update_stats_to_host(self) -> wp.array[wp.float32]:
        """Copy compact SAC update statistics into pinned host storage."""

        wp.launch(
            pack_sac_update_stats_kernel,
            dim=1,
            inputs=[self._actor_loss, self._critic_loss, self._alpha_loss, self._alpha],
            outputs=[self._update_stats],
            device=self.device,
        )
        wp.copy(self._update_stats_host, self._update_stats, count=4)
        return self._update_stats_host

    def _read_update_stats(self) -> StatsSACUpdate:
        stats_host = self.copy_update_stats_to_host()
        if self.device.is_cuda:
            wp.synchronize_device(self.device)
        stats = stats_host.numpy()
        return StatsSACUpdate(
            actor_loss=float(stats[0]),
            critic_loss=float(stats[1]),
            alpha_loss=float(stats[2]),
            alpha=float(stats[3]),
        )

    def update(self, batch: BatchSAC, *, seed: int | None = None, read_stats: bool = True) -> StatsSACUpdate:
        """Update actor, critics, targets, and entropy temperature.

        Args:
            batch: Transition batch to train on.
            seed: Random seed. Uses a deterministic update counter when omitted.
            read_stats: Whether to synchronize and return scalar diagnostics.
        """

        if int(batch.obs.shape[1]) != self.obs_dim or int(batch.next_obs.shape[1]) != self.obs_dim:
            raise ValueError("Batch observation dimensions do not match trainer")
        if int(batch.actions.shape[1]) != self.action_dim:
            raise ValueError("Batch action dimensions do not match trainer")

        batch = self._normalize_batch(batch)
        base_seed = self.seed + self._update_count * 9973 if seed is None else int(seed)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._actor_loss], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._alpha_loss], device=self.device)
        for i in range(int(self.config.update_steps)):
            self._update_critics(batch, seed=base_seed + 3 * i)
            if self._gradient_update_count % int(self.config.policy_frequency) == 0:
                self._update_actor(batch, seed=base_seed + 3 * i + 1)
            if self.config.auto_alpha:
                self._update_alpha(batch, seed=base_seed + 3 * i + 2)
            self.target_critic1.soft_update_from(self.critic1, self.config.tau)
            self.target_critic2.soft_update_from(self.critic2, self.config.tau)
            self._gradient_update_count += 1
        self._update_count += 1
        if read_stats:
            return self._read_update_stats()
        return StatsSACUpdate(actor_loss=0.0, critic_loss=0.0, alpha_loss=0.0, alpha=0.0)

    def _normalize_observations(self, obs: wp.array) -> wp.array:
        if not self.config.normalize_observations:
            return obs
        normalized = wp.empty_like(obs)
        wp.launch(
            normalize_observations_kernel,
            dim=obs.shape,
            inputs=[obs, self._obs_mean, self._obs_m2, self._obs_count, float(self.config.observation_clip)],
            outputs=[normalized],
            device=self.device,
        )
        return normalized

    def _normalize_batch(self, batch: BatchSAC) -> BatchSAC:
        if not self.config.normalize_observations:
            return batch
        wp.launch(
            observation_moments_partial_kernel,
            dim=self._obs_sums.shape,
            inputs=[batch.obs, self._obs_moment_partial_count],
            outputs=[self._obs_sums, self._obs_sums_sq],
            device=self.device,
        )
        wp.launch(
            observation_moments_update_kernel,
            dim=self.obs_dim,
            inputs=[
                self._obs_sums,
                self._obs_sums_sq,
                self._obs_moment_partial_count,
                batch.batch_size,
                self._obs_count,
            ],
            outputs=[self._obs_mean, self._obs_m2],
            device=self.device,
        )
        wp.launch(
            observation_count_update_kernel,
            dim=1,
            inputs=[batch.batch_size],
            outputs=[self._obs_count],
            device=self.device,
        )
        return BatchSAC(
            obs=self._normalize_observations(batch.obs),
            actions=batch.actions,
            rewards=batch.rewards,
            dones=batch.dones,
            next_obs=self._normalize_observations(batch.next_obs),
        )

    def _concat(self, obs: wp.array, actions: wp.array, *, requires_grad: bool) -> wp.array:
        batch_size = int(obs.shape[0])
        out = wp.empty(
            (batch_size, self.obs_dim + self.action_dim),
            dtype=wp.float32,
            device=self.device,
            requires_grad=requires_grad,
        )
        wp.launch(
            concat_obs_action_kernel,
            dim=(batch_size, self.obs_dim + self.action_dim),
            inputs=[obs, actions, self.obs_dim, self.action_dim],
            outputs=[out],
            device=self.device,
        )
        return out

    def _concat_rows(self, first: wp.array2d[wp.float32], second: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        if first.shape != second.shape:
            raise ValueError("Row-concatenated arrays must have matching shapes")
        out = wp.empty((int(first.shape[0]) * 2, int(first.shape[1])), dtype=wp.float32, device=self.device)
        wp.launch(_concat_rows_kernel, dim=out.shape, inputs=[first, second], outputs=[out], device=self.device)
        return out

    def _slice_rows(self, source: wp.array2d[wp.float32], *, offset: int, row_count: int) -> wp.array2d[wp.float32]:
        out = wp.empty((int(row_count), int(source.shape[1])), dtype=wp.float32, device=self.device)
        wp.launch(
            _copy_rows_kernel,
            dim=out.shape,
            inputs=[source, int(offset), 0],
            outputs=[out],
            device=self.device,
        )
        return out

    def _expand_first_row_grad(self, source: wp.array2d[wp.float32], *, total_rows: int) -> wp.array2d[wp.float32]:
        out = wp.zeros((int(total_rows), int(source.shape[1])), dtype=wp.float32, device=self.device)
        wp.launch(
            _copy_rows_kernel,
            dim=source.shape,
            inputs=[source, 0, 0],
            outputs=[out],
            device=self.device,
        )
        return out

    def _update_critics(
        self,
        batch: BatchSAC,
        *,
        seed: int,
        seed_counter: wp.array[wp.int32] | None = None,
        seed_offset: int = 0,
    ) -> None:
        next_actions, next_log_probs, _policy_out = self.actor.sample(
            batch.next_obs,
            seed=seed,
            deterministic=False,
            requires_grad=False,
            seed_counter=seed_counter,
            seed_offset=seed_offset,
        )
        next_q_input = self._concat(batch.next_obs, next_actions, requires_grad=False)
        q_input = self._concat(batch.obs, batch.actions, requires_grad=False)
        reference_batch_norm = bool(getattr(self.critic1, "reference_batch_norm", False))
        if reference_batch_norm:
            combined_q_input = self._concat_rows(q_input, next_q_input)
            critic_ensemble = getattr(self, "_critic_ensemble", None)
            target_critic_ensemble = getattr(self, "_target_critic_ensemble", None)
            if critic_ensemble is not None and target_critic_ensemble is not None:
                target_q1_all, target_q2_all = target_critic_ensemble.forward(combined_q_input, training=True)
            else:
                target_q1_all = self.target_critic1.forward(combined_q_input, requires_grad=False)
                target_q2_all = self.target_critic2.forward(combined_q_input, requires_grad=False)
            target_q1 = self._slice_rows(target_q1_all, offset=batch.batch_size, row_count=batch.batch_size)
            target_q2 = self._slice_rows(target_q2_all, offset=batch.batch_size, row_count=batch.batch_size)
            if critic_ensemble is not None:
                q1_all, q2_all = critic_ensemble.forward_manual(combined_q_input, training=True)
            else:
                q1_all = self.critic1.forward_manual(combined_q_input, training=True)
                q2_all = self.critic2.forward_manual(combined_q_input, training=True)
            q1 = self._slice_rows(q1_all, offset=0, row_count=batch.batch_size)
            q2 = self._slice_rows(q2_all, offset=0, row_count=batch.batch_size)
        else:
            target_q1 = self.target_critic1.forward(next_q_input, requires_grad=False)
            target_q2 = self.target_critic2.forward(next_q_input, requires_grad=False)
            q1 = self.critic1.forward_manual(q_input)
            q2 = self.critic2.forward_manual(q_input)

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._critic_loss], device=self.device)
        q1_grad = wp.empty_like(q1)
        q2_grad = wp.empty_like(q2)
        if self.config.distributional_critic:
            atoms = int(self.config.distributional_atoms)
            target_distribution1 = wp.zeros((batch.batch_size, atoms), dtype=wp.float32, device=self.device)
            target_distribution2 = wp.zeros_like(target_distribution1)
            projection_kernel = (
                sac_distributional_min_projection_device_alpha_kernel
                if self.config.distributional_min_target
                else sac_distributional_projection_device_alpha_kernel
            )
            wp.launch(
                projection_kernel,
                dim=(batch.batch_size, atoms),
                inputs=[
                    batch.rewards,
                    batch.dones,
                    target_q1,
                    target_q2,
                    next_log_probs,
                    self.config.gamma**self.config.n_step,
                    self._alpha,
                    atoms,
                    self.config.distributional_v_min,
                    self.config.distributional_v_max,
                ],
                outputs=[target_distribution1, target_distribution2],
                device=self.device,
            )
            wp.launch(
                sac_distributional_critic_loss_backward_kernel,
                dim=batch.batch_size,
                inputs=[
                    q1,
                    q2,
                    target_distribution1,
                    target_distribution2,
                    batch.batch_size,
                    atoms,
                ],
                outputs=[self._critic_loss, q1_grad, q2_grad],
                device=self.device,
            )
        else:
            targets = wp.empty(batch.batch_size, dtype=wp.float32, device=self.device)
            wp.launch(
                sac_q_target_device_alpha_kernel,
                dim=batch.batch_size,
                inputs=[
                    batch.rewards,
                    batch.dones,
                    target_q1,
                    target_q2,
                    next_log_probs,
                    self.config.gamma**self.config.n_step,
                    self._alpha,
                    self.config.average_critics,
                ],
                outputs=[targets],
                device=self.device,
            )
            wp.launch(
                sac_critic_loss_kernel,
                dim=batch.batch_size,
                inputs=[q1, q2, targets, batch.batch_size],
                outputs=[self._critic_loss],
                device=self.device,
            )
            wp.launch(
                sac_critic_loss_backward_kernel,
                dim=batch.batch_size,
                inputs=[q1, q2, targets, batch.batch_size],
                outputs=[q1_grad, q2_grad],
                device=self.device,
            )
        amp_scale = getattr(self, "_amp_scale", None)
        found_inf = getattr(self, "_amp_found_inf", None)
        if amp_scale is not None:
            wp.launch(amp_reset_found_inf_kernel, dim=1, outputs=[found_inf], device=self.device)
            wp.launch(scale_2d_in_place_kernel, dim=q1_grad.shape, inputs=[q1_grad, amp_scale], device=self.device)
            wp.launch(scale_2d_in_place_kernel, dim=q2_grad.shape, inputs=[q2_grad, amp_scale], device=self.device)
        if reference_batch_norm:
            q1_grad = self._expand_first_row_grad(q1_grad, total_rows=batch.batch_size * 2)
            q2_grad = self._expand_first_row_grad(q2_grad, total_rows=batch.batch_size * 2)
            if critic_ensemble is not None:
                critic_ensemble.backward_manual(q1_grad, q2_grad, loss_scale=amp_scale, found_inf=found_inf)
            else:
                self.critic1.backward_manual(q1_grad, loss_scale=amp_scale, found_inf=found_inf)
                self.critic2.backward_manual(q2_grad, loss_scale=amp_scale, found_inf=found_inf)
        else:
            self.critic1.backward_manual(q1_grad)
            self.critic2.backward_manual(q2_grad)
        if amp_scale is not None:
            wp.launch(
                amp_finalize_step_kernel,
                dim=1,
                inputs=[found_inf],
                outputs=[self._amp_step_condition],
                device=self.device,
            )
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        if amp_scale is not None:
            wp.launch(
                amp_update_scale_kernel,
                dim=1,
                inputs=[found_inf, self._amp_scale, self._amp_growth_tracker, 2.0, 0.5, 2000],
                device=self.device,
            )

    def _update_actor(
        self,
        batch: BatchSAC,
        *,
        seed: int,
        seed_counter: wp.array[wp.int32] | None = None,
        seed_offset: int = 0,
    ) -> None:
        reference_batch_norm = bool(getattr(self.actor.net, "reference_batch_norm", False))
        if reference_batch_norm:
            actor_observations = self._concat_rows(batch.obs, batch.next_obs)
            policy_out_all = self.actor.net.forward_manual(actor_observations, training=True)
            policy_out = self._slice_rows(policy_out_all, offset=0, row_count=batch.batch_size)
        else:
            policy_out = self.actor.net.forward_manual(batch.obs)
        actions = wp.empty((batch.batch_size, self.action_dim), dtype=wp.float32, device=self.device)
        log_probs = wp.empty(batch.batch_size, dtype=wp.float32, device=self.device)
        eps = wp.empty((batch.batch_size, self.action_dim), dtype=wp.float32, device=self.device)
        if seed_counter is None:
            wp.launch(fill_eps_kernel, dim=eps.shape, inputs=[int(seed)], outputs=[eps], device=self.device)
        else:
            wp.launch(
                fill_eps_seed_counter_kernel,
                dim=eps.shape,
                inputs=[seed_counter, int(seed_offset)],
                outputs=[eps],
                device=self.device,
            )
        wp.launch(
            sample_gaussian_actions_kernel,
            dim=batch.batch_size,
            inputs=[
                policy_out,
                self.actor.log_std,
                eps,
                self.action_dim,
                int(self.actor.state_dependent_std),
                int(self.actor.squash),
                0,
                self.actor.log_std_min,
                self.actor.log_std_max,
            ],
            outputs=[actions, log_probs],
            device=self.device,
        )
        self._actor_update_log_probs = log_probs

        q_input = self._concat(batch.obs, actions, requires_grad=False)
        if reference_batch_norm:
            critic_ensemble = getattr(self, "_critic_ensemble", None)
            if critic_ensemble is not None:
                q1, q2 = critic_ensemble.forward_manual(q_input, training=False)
            else:
                q1 = self.critic1.forward_manual(q_input, training=False)
                q2 = self.critic2.forward_manual(q_input, training=False)
        else:
            q1 = self.critic1.forward_manual(q_input)
            q2 = self.critic2.forward_manual(q_input)
        q1_grad = wp.empty_like(q1)
        q2_grad = wp.empty_like(q2)
        if self.config.distributional_critic:
            atoms = int(self.config.distributional_atoms)
            q1_value = wp.empty((batch.batch_size, 1), dtype=wp.float32, device=self.device)
            q2_value = wp.empty_like(q1_value)
            for logits, values in ((q1, q1_value), (q2, q2_value)):
                wp.launch(
                    sac_distributional_q_value_kernel,
                    dim=batch.batch_size,
                    inputs=[
                        logits,
                        atoms,
                        self.config.distributional_v_min,
                        self.config.distributional_v_max,
                    ],
                    outputs=[values],
                    device=self.device,
                )
            wp.launch(
                sac_distributional_actor_q_backward_kernel,
                dim=(batch.batch_size, atoms),
                inputs=[
                    q1,
                    q2,
                    q1_value,
                    q2_value,
                    batch.batch_size,
                    atoms,
                    self.config.distributional_v_min,
                    self.config.distributional_v_max,
                    self.config.average_critics,
                ],
                outputs=[q1_grad, q2_grad],
                device=self.device,
            )
        else:
            q1_value = q1
            q2_value = q2
            wp.launch(
                sac_actor_q_backward_kernel,
                dim=batch.batch_size,
                inputs=[q1, q2, batch.batch_size, self.config.average_critics],
                outputs=[q1_grad, q2_grad],
                device=self.device,
            )
        amp_scale = getattr(self, "_amp_scale", None)
        found_inf = getattr(self, "_amp_found_inf", None)
        if amp_scale is not None:
            wp.launch(amp_reset_found_inf_kernel, dim=1, outputs=[found_inf], device=self.device)
            wp.launch(scale_2d_in_place_kernel, dim=q1_grad.shape, inputs=[q1_grad, amp_scale], device=self.device)
            wp.launch(scale_2d_in_place_kernel, dim=q2_grad.shape, inputs=[q2_grad, amp_scale], device=self.device)
        q_input_grad1 = wp.empty_like(q_input)
        q_input_grad2 = wp.empty_like(q_input)
        if reference_batch_norm and critic_ensemble is not None:
            critic_ensemble.backward_manual(
                q1_grad,
                q2_grad,
                first_input_grad=q_input_grad1,
                second_input_grad=q_input_grad2,
                loss_scale=amp_scale,
                found_inf=found_inf,
            )
        else:
            self.critic1.backward_manual(q1_grad, input_grad=q_input_grad1)
            self.critic2.backward_manual(q2_grad, input_grad=q_input_grad2)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._actor_loss], device=self.device)
        policy_out_grad = wp.empty_like(policy_out)
        wp.launch(
            sac_actor_policy_backward_device_alpha_kernel,
            dim=batch.batch_size,
            inputs=[
                policy_out,
                eps,
                q_input_grad1,
                q_input_grad2,
                q1_value,
                q2_value,
                log_probs,
                self.obs_dim,
                self.action_dim,
                batch.batch_size,
                self._alpha,
                self.actor.log_std_min,
                self.actor.log_std_max,
                self.config.average_critics,
                self._loss_scale,
            ],
            outputs=[self._actor_loss, policy_out_grad],
            device=self.device,
        )
        if reference_batch_norm:
            policy_out_grad = self._expand_first_row_grad(policy_out_grad, total_rows=batch.batch_size * 2)
        if reference_batch_norm:
            self.actor.net.backward_manual(policy_out_grad, loss_scale=amp_scale, found_inf=found_inf)
        else:
            self.actor.net.backward_manual(policy_out_grad)
        if amp_scale is not None:
            wp.launch(
                amp_finalize_step_kernel,
                dim=1,
                inputs=[found_inf],
                outputs=[self._amp_step_condition],
                device=self.device,
            )
        self.actor_optimizer.step()
        if amp_scale is not None:
            wp.launch(
                amp_update_scale_kernel,
                dim=1,
                inputs=[found_inf, self._amp_scale, self._amp_growth_tracker, 2.0, 0.5, 2000],
                device=self.device,
            )

    def _update_alpha(self, batch: BatchSAC, *, seed: int) -> None:
        _actions, log_probs, _policy_out = self.actor.sample(
            batch.obs, seed=seed, deterministic=False, requires_grad=False
        )
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._alpha_loss], device=self.device)
        with wp.Tape() as tape:
            wp.launch(
                sac_alpha_loss_kernel,
                dim=batch.batch_size,
                inputs=[log_probs, self.log_alpha, batch.batch_size, self.target_entropy],
                outputs=[self._alpha_loss],
                device=self.device,
            )
        tape.backward(self._alpha_loss)
        self.alpha_optimizer.step()
        wp.launch(
            sac_refresh_alpha_kernel,
            dim=1,
            inputs=[self.log_alpha],
            outputs=[self._alpha],
            device=self.device,
        )
        tape.zero()
