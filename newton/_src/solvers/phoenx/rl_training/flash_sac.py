# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Protocol

import numpy as np
import warp as wp

from .flash_sac_networks import EnsembleNetworkFlashSAC, NetworkFlashSAC
from .kernels import (
    flash_sac_graph_n_step_finalize_kernel,
    flash_sac_graph_n_step_store_kernel,
    flash_sac_graph_replay_sample_kernel,
    flash_sac_n_step_accumulate_kernel,
    flash_sac_normalize_rewards_kernel,
    flash_sac_return_stats_kernel,
    flash_sac_update_return_normalizer_kernel,
    sac_refresh_alpha_kernel,
    seed_counter_increment_kernel,
    zero_scalar_kernel,
)
from .optim import Adam
from .ppo import (
    _pack_optimizer,
    _pack_policy_network,
    _unpack_optimizer,
    _unpack_policy_network,
)
from .sac import BatchSAC, BufferReplaySAC, ConfigSAC, StatsSACUpdate, TrainerSAC


def _capture_flash_stream_graph(stream: wp.Stream, device: wp.context.Device, workload: Callable[[], None]) -> object:
    """Capture one workload on a dedicated stream after the main stream."""

    main_stream = wp.get_stream(device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        wp.wait_stream(main_stream)
        with wp.ScopedCapture(device=device, stream=stream) as capture:
            workload()
    wp.wait_stream(stream)
    wp.synchronize_device(device)
    return capture.graph


def _allocate_flash_sac_batch(trainer: TrainerFlashSAC) -> BatchSAC:
    rows = int(trainer.config.sample_batch_size)
    return BatchSAC(
        obs=wp.empty((rows, trainer.obs_dim), dtype=wp.float32, device=trainer.device),
        actions=wp.empty((rows, trainer.action_dim), dtype=wp.float32, device=trainer.device),
        rewards=wp.empty(rows, dtype=wp.float32, device=trainer.device),
        dones=wp.empty(rows, dtype=wp.float32, device=trainer.device),
        next_obs=wp.empty((rows, trainer.obs_dim), dtype=wp.float32, device=trainer.device),
    )


def _copy_flash_sac_batch(destination: BatchSAC, source: BatchSAC) -> None:
    wp.copy(destination.obs, source.obs)
    wp.copy(destination.actions, source.actions)
    wp.copy(destination.rewards, source.rewards)
    wp.copy(destination.dones, source.dones)
    wp.copy(destination.next_obs, source.next_obs)


@wp.kernel
def _flash_sac_alpha_loss_kernel(
    log_probs: wp.array2d[wp.float32],
    log_alpha: wp.array2d[wp.float32],
    batch_size: wp.int32,
    target_entropy: wp.float32,
    loss: wp.array[wp.float32],
    log_alpha_grad: wp.array2d[wp.float32],
):
    member, lane = wp.tid()
    entropy_sum = wp.float32(0.0)
    for row in range(lane, batch_size, wp.int32(32)):
        entropy_sum -= log_probs[member, row]
    entropy_total = wp.tile_sum(wp.tile(entropy_sum))[0]
    if lane == wp.int32(0):
        alpha = wp.exp(log_alpha[member, 0])
        value = alpha * (entropy_total / wp.float32(batch_size) - target_entropy)
        loss[member] = value
        log_alpha_grad[member, 0] = value


@wp.kernel
def _prepare_flash_sac_graph_update_kernel(
    gradient_update_count: wp.array[wp.int32],
    update_count: wp.array[wp.int32],
    seed_base: wp.int32,
    policy_frequency: wp.int32,
    warmup_steps: wp.int32,
    decay_steps: wp.int32,
    peak_lr: wp.float32,
    end_lr: wp.float32,
    actor_base_lr: wp.float32,
    critic_base_lr: wp.float32,
    alpha_base_lr: wp.float32,
    actor_condition: wp.array[wp.int32],
    actor_skip_condition: wp.array[wp.int32],
    update_seed: wp.array[wp.int32],
    actor_lr_scale: wp.array[wp.float32],
    critic1_lr_scale: wp.array[wp.float32],
    critic2_lr_scale: wp.array[wp.float32],
    alpha_lr_scale: wp.array[wp.float32],
):
    step = gradient_update_count[0]
    actor_condition[0] = wp.int32(step % policy_frequency == 0)
    actor_skip_condition[0] = wp.int32(step % policy_frequency != 0)
    seed64 = wp.int64(seed_base) + wp.int64(update_count[0]) * wp.int64(9973)
    update_seed[0] = wp.int32(seed64 % wp.int64(2147483647))
    lr = peak_lr
    if warmup_steps > 0 and step < warmup_steps:
        lr = peak_lr * wp.float32(step + 1) / wp.float32(warmup_steps)
    else:
        decay_step = wp.min(wp.max(step - warmup_steps, 0), decay_steps)
        progress = wp.float32(decay_step) / wp.float32(decay_steps)
        cosine = wp.float32(0.5) * (wp.float32(1.0) + wp.cos(wp.pi * progress))
        lr = end_lr + (peak_lr - end_lr) * cosine
    actor_lr_scale[0] = lr / actor_base_lr
    critic1_lr_scale[0] = lr / critic_base_lr
    critic2_lr_scale[0] = lr / critic_base_lr
    alpha_lr_scale[0] = lr / alpha_base_lr


@wp.kernel
def _prepare_flash_sac_exploration_kernel(
    seed_counter: wp.array[wp.int32],
    noise_cdf: wp.array[wp.float32],
    repeat_count: wp.array[wp.int32],
    repeat_steps: wp.array[wp.int32],
    exploration_seed: wp.array[wp.int32],
):
    if repeat_count[0] >= repeat_steps[0]:
        seed = seed_counter[0]
        rng = wp.rand_init(seed, 0)
        uniform = wp.randf(rng)
        duration = wp.int32(noise_cdf.shape[0])
        for index in range(noise_cdf.shape[0]):
            if uniform <= noise_cdf[index] and duration == noise_cdf.shape[0]:
                duration = index + 1
        exploration_seed[0] = seed
        repeat_steps[0] = duration
        repeat_count[0] = wp.int32(0)
    repeat_count[0] = repeat_count[0] + 1


class EnvFlashSAC(Protocol):
    """Vectorized environment interface consumed by :func:`train_flash_sac`."""

    world_count: int
    obs_dim: int
    action_dim: int
    device: wp.context.Device

    def reset(self) -> wp.array2d[wp.float32]: ...

    def observe(self) -> wp.array2d[wp.float32]: ...

    def step(self, actions: wp.array2d[wp.float32]) -> tuple[wp.array, wp.array, wp.array]: ...


@dataclass
class ConfigFlashSAC(ConfigSAC):
    """Configuration for :class:`TrainerFlashSAC`.

    These defaults follow the upstream FlashSAC configuration. With the default
    architecture, the Warp port uses upstream-equivalent batch-normalized
    residual actor and critic backbones.

    Args:
        target_sigma: Standard deviation used to derive the target entropy.
        noise_zeta_mu: Exponent of the repeated-noise zeta distribution.
        noise_zeta_max: Maximum number of steps for which exploration noise is reused.
        learning_rate_end: Final cosine-decay learning rate.
        learning_rate_warmup_steps: Number of linear learning-rate warmup updates.
        learning_rate_decay_steps: Number of updates over which to decay the learning rate.
        normalize_weights: Whether to apply FlashSAC unit incoming-weight constraints.
        n_step: Number of transitions in replay returns.
        normalize_rewards: Whether replay samples use discounted-return normalization.
        normalized_return_max: Maximum normalized discounted-return magnitude.
        buffer_max_length: Maximum number of replay transitions.
        buffer_min_length: Number of replay transitions required before training.
        sample_batch_size: Number of replay transitions sampled per update.
        actor_num_blocks: Number of expanded actor blocks.
        actor_hidden_dim: Actor feature width.
        critic_num_blocks: Number of expanded critic blocks.
        critic_hidden_dim: Critic feature width.
        use_amp: Whether reference-backbone dense contractions use FP16 inputs with FP32 accumulation.
    """

    tau: float = 0.01
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    initial_alpha: float = 0.01
    policy_frequency: int = 2
    distributional_critic: bool = True
    distributional_atoms: int = 101
    distributional_v_min: float = -5.0
    distributional_v_max: float = 5.0
    distributional_min_target: bool = True
    target_sigma: float = 0.15
    noise_zeta_mu: float = 2.0
    noise_zeta_max: int = 16
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_steps: int = 0
    learning_rate_decay_steps: int = 1_000_000
    normalize_weights: bool = True
    n_step: int = 1
    normalize_rewards: bool = True
    normalized_return_max: float = 5.0
    buffer_max_length: int = 1_000_000
    buffer_min_length: int = 10_000
    sample_batch_size: int = 2048
    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    use_amp: bool = False


class RewardNormalizerFlashSAC:
    """Normalize rewards by running discounted-return scale using Warp arrays."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        normalized_return_max: float = 5.0,
        device: wp.context.Devicelike = None,
    ):
        if normalized_return_max <= 0.0:
            raise ValueError("normalized_return_max must be positive")
        self.gamma = float(gamma)
        self.normalized_return_max = float(normalized_return_max)
        self.device = wp.get_device(device)
        self.returns: wp.array[wp.float32] | None = None
        self.running_mean = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.running_var = wp.ones(1, dtype=wp.float32, device=self.device)
        self.running_count = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.max_abs_return = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._metrics = wp.zeros(3, dtype=wp.float32, device=self.device)

    def reserve(self, world_count: int) -> None:
        """Reserve fixed discounted-return state for graph replay."""

        count = int(world_count)
        if count <= 0:
            raise ValueError("world_count must be positive")
        if self.returns is None:
            self.returns = wp.zeros(count, dtype=wp.float32, device=self.device)
        elif int(self.returns.shape[0]) != count:
            raise ValueError("Reward normalizer environment count cannot change")

    def normalize_into(self, rewards: wp.array[wp.float32], normalized_rewards: wp.array[wp.float32]) -> None:
        """Normalize rewards into a fixed-address output buffer."""

        wp.launch(
            flash_sac_normalize_rewards_kernel,
            dim=rewards.shape[0],
            inputs=[rewards, self.running_var, self.max_abs_return, self.normalized_return_max, 1.0e-8],
            outputs=[normalized_rewards],
            device=self.device,
        )

    def update(
        self,
        rewards: wp.array[wp.float32],
        terminateds: wp.array[wp.float32],
        truncateds: wp.array[wp.float32],
    ) -> None:
        """Update discounted returns and their running scale."""

        count = int(rewards.shape[0])
        self.reserve(count)
        self._metrics.zero_()
        wp.launch(
            flash_sac_return_stats_kernel,
            dim=count,
            inputs=[rewards, terminateds, truncateds, self.gamma, self.returns],
            outputs=[self._metrics],
            device=self.device,
        )
        wp.launch(
            flash_sac_update_return_normalizer_kernel,
            dim=1,
            inputs=[self._metrics, count],
            outputs=[self.running_mean, self.running_var, self.running_count, self.max_abs_return],
            device=self.device,
        )

    def normalize(self, rewards: wp.array[wp.float32]) -> wp.array[wp.float32]:
        """Scale rewards using running return variance and range bounds."""

        normalized = wp.empty_like(rewards)
        self.normalize_into(rewards, normalized)
        return normalized


class BufferReplayFlashSAC(BufferReplaySAC):
    """Uniform Warp replay with n-step returns and reward normalization.

    Args:
        minimum_size: Number of stored transitions required before updates begin.
        capacity: Maximum number of transitions.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        batch_size: Default sampled batch size.
        n_step: Number of transitions accumulated into each replay row.
        gamma: Discount factor used by n-step returns and reward normalization.
        normalize_rewards: Whether sampled rewards are normalized.
        normalized_return_max: Maximum normalized discounted-return magnitude.
        device: Warp device.
    """

    def __init__(
        self,
        *,
        minimum_size: int = 10_000,
        capacity: int = 1_000_000,
        obs_dim: int,
        action_dim: int,
        batch_size: int = 2048,
        n_step: int = 1,
        gamma: float = 0.99,
        normalize_rewards: bool = True,
        normalized_return_max: float = 5.0,
        device: wp.context.Devicelike = None,
    ):
        super().__init__(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            batch_size=batch_size,
            device=device,
        )
        self.minimum_size = int(minimum_size)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.normalize_rewards = bool(normalize_rewards)
        if self.minimum_size < 0 or self.minimum_size > self.capacity:
            raise ValueError("minimum_size must be between zero and capacity")
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        self.reward_normalizer = RewardNormalizerFlashSAC(
            gamma=self.gamma, normalized_return_max=normalized_return_max, device=self.device
        )
        self._n_step_transitions: deque[
            tuple[
                wp.array2d[wp.float32],
                wp.array2d[wp.float32],
                wp.array[wp.float32],
                wp.array[wp.float32],
                wp.array[wp.float32],
                wp.array2d[wp.float32],
            ]
        ] = deque(maxlen=self.n_step)
        self._graph_world_count = 0
        self._graph_pending_obs: wp.array3d[wp.float32] | None = None
        self._graph_pending_actions: wp.array3d[wp.float32] | None = None
        self._graph_pending_rewards: wp.array2d[wp.float32] | None = None
        self._graph_pending_terminateds: wp.array2d[wp.float32] | None = None
        self._graph_pending_truncateds: wp.array2d[wp.float32] | None = None
        self._graph_pending_next_obs: wp.array3d[wp.float32] | None = None
        self._graph_pending_cursor: wp.array[wp.int32] | None = None
        self._graph_pending_count: wp.array[wp.int32] | None = None
        self._graph_position: wp.array[wp.int32] | None = None
        self._graph_size: wp.array[wp.int32] | None = None
        self._graph_sample_raw_rewards: wp.array[wp.float32] | None = None
        self._graph_batch: BatchSAC | None = None
        self._graph_pending_count_host = 0

    def reserve_graph_buffers(self, world_count: int) -> BatchSAC:
        """Reserve fixed-address n-step and sampled-batch buffers for graph replay."""

        worlds = int(world_count)
        if worlds <= 0 or worlds > self.capacity:
            raise ValueError("world_count must be positive and no larger than replay capacity")
        if self._graph_world_count not in (0, worlds):
            raise ValueError("graph replay world_count cannot change after reservation")
        if self._graph_world_count == worlds and self._graph_batch is not None:
            return self._graph_batch
        if self._n_step_transitions:
            raise RuntimeError("reserve graph buffers before adding eager n-step transitions")
        self._graph_world_count = worlds
        self._graph_pending_obs = wp.empty((self.n_step, worlds, self.obs_dim), dtype=wp.float32, device=self.device)
        self._graph_pending_actions = wp.empty(
            (self.n_step, worlds, self.action_dim), dtype=wp.float32, device=self.device
        )
        self._graph_pending_rewards = wp.empty((self.n_step, worlds), dtype=wp.float32, device=self.device)
        self._graph_pending_terminateds = wp.empty_like(self._graph_pending_rewards)
        self._graph_pending_truncateds = wp.empty_like(self._graph_pending_rewards)
        self._graph_pending_next_obs = wp.empty_like(self._graph_pending_obs)
        self._graph_pending_cursor = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._graph_pending_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._graph_position = wp.array([self.position], dtype=wp.int32, device=self.device)
        self._graph_size = wp.array([self.size], dtype=wp.int32, device=self.device)
        self.reward_normalizer.reserve(worlds)
        sample_obs = wp.empty((self.batch_size, self.obs_dim), dtype=wp.float32, device=self.device)
        sample_actions = wp.empty((self.batch_size, self.action_dim), dtype=wp.float32, device=self.device)
        sample_rewards = wp.empty(self.batch_size, dtype=wp.float32, device=self.device)
        self._graph_sample_raw_rewards = wp.empty_like(sample_rewards)
        sample_dones = wp.empty(self.batch_size, dtype=wp.float32, device=self.device)
        sample_next_obs = wp.empty_like(sample_obs)
        self._graph_batch = BatchSAC(
            obs=sample_obs,
            actions=sample_actions,
            rewards=sample_rewards,
            dones=sample_dones,
            next_obs=sample_next_obs,
        )
        return self._graph_batch

    def advance_graph_host_state(self, step_count: int = 1) -> None:
        """Advance replay host mirrors after known graph launches without a readback."""

        for _ in range(int(step_count)):
            self._graph_pending_count_host = min(self._graph_pending_count_host + 1, self.n_step)
            if self._graph_pending_count_host >= self.n_step:
                self.position = (self.position + self._graph_world_count) % self.capacity
                self.size = min(self.size + self._graph_world_count, self.capacity)

    def sync_graph_host_state(self) -> None:
        """Synchronize replay counters for explicit persistence or diagnostics."""

        if self._graph_size is None or self._graph_position is None or self._graph_pending_count is None:
            return
        self.size = int(self._graph_size.numpy()[0])
        self.position = int(self._graph_position.numpy()[0])
        self._graph_pending_count_host = int(self._graph_pending_count.numpy()[0])

    def add_batch_graph(
        self,
        obs: wp.array2d[wp.float32],
        actions: wp.array2d[wp.float32],
        rewards: wp.array[wp.float32],
        terminateds: wp.array[wp.float32],
        next_obs: wp.array2d[wp.float32],
        *,
        truncateds: wp.array[wp.float32],
    ) -> None:
        """Insert one vectorized transition step without allocation or host state."""

        worlds = int(rewards.shape[0])
        self.reserve_graph_buffers(worlds)
        if self.normalize_rewards:
            self.reward_normalizer.update(rewards, terminateds, truncateds)
        max_cols = max(self.obs_dim, self.action_dim, 1)
        wp.launch(
            flash_sac_graph_n_step_store_kernel,
            dim=(worlds, max_cols),
            inputs=[
                obs,
                actions,
                rewards,
                terminateds,
                truncateds,
                next_obs,
                self._graph_pending_obs,
                self._graph_pending_actions,
                self._graph_pending_rewards,
                self._graph_pending_terminateds,
                self._graph_pending_truncateds,
                self._graph_pending_next_obs,
                self._graph_pending_cursor,
                self._graph_pending_count,
                self._graph_position,
                self.capacity,
                self.n_step,
                self.gamma,
                self.obs_dim,
                self.action_dim,
            ],
            outputs=[self.obs, self.actions, self.rewards, self.dones, self.next_obs],
            device=self.device,
        )
        wp.launch(
            flash_sac_graph_n_step_finalize_kernel,
            dim=1,
            inputs=[worlds, self.capacity, self.n_step],
            outputs=[
                self._graph_pending_cursor,
                self._graph_pending_count,
                self._graph_position,
                self._graph_size,
            ],
            device=self.device,
        )

        if not self.device.is_capturing:
            self.advance_graph_host_state()

    def sample_graph_seed_counter(
        self,
        seed_counter: wp.array[wp.int32],
        *,
        seed_offset: int = 0,
    ) -> BatchSAC:
        """Sample into fixed buffers using device replay size and RNG state."""

        if self._graph_batch is None or self._graph_sample_raw_rewards is None:
            raise RuntimeError("reserve_graph_buffers() must be called before graph sampling")
        batch = self._graph_batch
        max_cols = max(self.obs_dim, self.action_dim, 1)
        reward_output = self._graph_sample_raw_rewards if self.normalize_rewards else batch.rewards
        wp.launch(
            flash_sac_graph_replay_sample_kernel,
            dim=(self.batch_size, max_cols),
            inputs=[
                self.obs,
                self.actions,
                self.rewards,
                self.dones,
                self.next_obs,
                self._graph_size,
                seed_counter,
                int(seed_offset),
                self.obs_dim,
                self.action_dim,
            ],
            outputs=[batch.obs, batch.actions, reward_output, batch.dones, batch.next_obs],
            device=self.device,
        )
        if self.normalize_rewards:
            self.reward_normalizer.normalize_into(reward_output, batch.rewards)
        return batch

    def add_batch(
        self,
        obs: wp.array2d[wp.float32],
        actions: wp.array2d[wp.float32],
        rewards: wp.array[wp.float32],
        dones: wp.array[wp.float32],
        next_obs: wp.array2d[wp.float32],
        truncateds: wp.array[wp.float32] | None = None,
    ) -> None:
        """Append transitions using upstream terminated and truncated semantics."""

        if truncateds is None:
            truncateds = wp.zeros_like(dones)
        if self.normalize_rewards:
            self.reward_normalizer.update(rewards, dones, truncateds)
        transition = tuple(wp.clone(value) for value in (obs, actions, rewards, dones, truncateds, next_obs))
        self._n_step_transitions.append(transition)
        if len(self._n_step_transitions) < self.n_step:
            return

        first_obs, first_actions, _rewards, _dones, _truncateds, _next_obs = self._n_step_transitions[0]
        latest = self._n_step_transitions[-1]
        aggregate_rewards = wp.clone(latest[2])
        aggregate_dones = wp.clone(latest[3])
        aggregate_truncateds = wp.clone(latest[4])
        aggregate_next_obs = wp.clone(latest[5])
        for transition_row in reversed(tuple(self._n_step_transitions)[:-1]):
            wp.launch(
                flash_sac_n_step_accumulate_kernel,
                dim=(rewards.shape[0], max(self.obs_dim, 1)),
                inputs=[
                    transition_row[2],
                    transition_row[3],
                    transition_row[4],
                    transition_row[5],
                    self.gamma,
                    self.obs_dim,
                    aggregate_rewards,
                    aggregate_dones,
                    aggregate_truncateds,
                    aggregate_next_obs,
                ],
                device=self.device,
            )
        super().add_batch(first_obs, first_actions, aggregate_rewards, aggregate_dones, aggregate_next_obs)

    def sample(self, *, seed: int, batch_size: int | None = None) -> BatchSAC:
        """Sample replay rows and normalize their rewards at current scale."""

        batch = super().sample(seed=seed, batch_size=batch_size)
        if not self.normalize_rewards:
            return batch
        return BatchSAC(
            obs=batch.obs,
            actions=batch.actions,
            rewards=self.reward_normalizer.normalize(batch.rewards),
            dones=batch.dones,
            next_obs=batch.next_obs,
        )

    def can_sample(self) -> bool:
        """Return whether the upstream-style replay warmup is complete."""

        return self.size >= self.minimum_size

    def save(self, path: str | Path) -> None:
        """Save replay storage, normalization state, and pending n-step rows."""

        self.sync_graph_host_state()
        data: dict[str, np.ndarray] = {
            "capacity": np.asarray(self.capacity, dtype=np.int64),
            "minimum_size": np.asarray(self.minimum_size, dtype=np.int64),
            "obs_dim": np.asarray(self.obs_dim, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "batch_size": np.asarray(self.batch_size, dtype=np.int64),
            "n_step": np.asarray(self.n_step, dtype=np.int64),
            "gamma": np.asarray(self.gamma, dtype=np.float32),
            "normalize_rewards": np.asarray(self.normalize_rewards, dtype=np.bool_),
            "normalized_return_max": np.asarray(self.reward_normalizer.normalized_return_max, dtype=np.float32),
            "size": np.asarray(self.size, dtype=np.int64),
            "position": np.asarray(self.position, dtype=np.int64),
            "obs": self.obs.numpy(),
            "actions": self.actions.numpy(),
            "rewards": self.rewards.numpy(),
            "dones": self.dones.numpy(),
            "next_obs": self.next_obs.numpy(),
            "return_running_mean": self.reward_normalizer.running_mean.numpy(),
            "return_running_var": self.reward_normalizer.running_var.numpy(),
            "return_running_count": self.reward_normalizer.running_count.numpy(),
            "return_max_abs": self.reward_normalizer.max_abs_return.numpy(),
            "return_values_present": np.asarray(self.reward_normalizer.returns is not None, dtype=np.bool_),
            "pending_count": np.asarray(len(self._n_step_transitions), dtype=np.int64),
            "graph_world_count": np.asarray(self._graph_world_count, dtype=np.int64),
        }
        if self.reward_normalizer.returns is not None:
            data["return_values"] = self.reward_normalizer.returns.numpy()
        names = ("obs", "actions", "rewards", "dones", "truncateds", "next_obs")
        for index, transition in enumerate(self._n_step_transitions):
            for name, value in zip(names, transition, strict=True):
                data[f"pending_{index}_{name}"] = value.numpy()
        if self._graph_world_count:
            graph_arrays = {
                "pending_obs": self._graph_pending_obs,
                "pending_actions": self._graph_pending_actions,
                "pending_rewards": self._graph_pending_rewards,
                "pending_terminateds": self._graph_pending_terminateds,
                "pending_truncateds": self._graph_pending_truncateds,
                "pending_next_obs": self._graph_pending_next_obs,
                "pending_cursor": self._graph_pending_cursor,
                "pending_count": self._graph_pending_count,
                "position": self._graph_position,
                "size": self._graph_size,
            }
            for name, value in graph_arrays.items():
                data[f"graph_{name}"] = value.numpy()
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, **data)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: wp.context.Devicelike = None,
    ) -> BufferReplayFlashSAC:
        """Restore replay storage, normalization state, and pending n-step rows."""

        with np.load(Path(path), allow_pickle=False) as data:
            replay = cls(
                minimum_size=int(data["minimum_size"]),
                capacity=int(data["capacity"]),
                obs_dim=int(data["obs_dim"]),
                action_dim=int(data["action_dim"]),
                batch_size=int(data["batch_size"]),
                n_step=int(data["n_step"]),
                gamma=float(data["gamma"]),
                normalize_rewards=bool(data["normalize_rewards"]),
                normalized_return_max=float(data["normalized_return_max"]),
                device=device,
            )
            replay.obs.assign(data["obs"])
            replay.actions.assign(data["actions"])
            replay.rewards.assign(data["rewards"])
            replay.dones.assign(data["dones"])
            replay.next_obs.assign(data["next_obs"])
            replay.size = int(data["size"])
            replay.position = int(data["position"])
            replay.reward_normalizer.running_mean.assign(data["return_running_mean"])
            replay.reward_normalizer.running_var.assign(data["return_running_var"])
            replay.reward_normalizer.running_count.assign(data["return_running_count"])
            replay.reward_normalizer.max_abs_return.assign(data["return_max_abs"])
            if bool(data["return_values_present"]):
                replay.reward_normalizer.returns = wp.array(
                    data["return_values"], dtype=wp.float32, device=replay.device
                )
            names = ("obs", "actions", "rewards", "dones", "truncateds", "next_obs")
            for index in range(int(data["pending_count"])):
                transition = tuple(
                    wp.array(data[f"pending_{index}_{name}"], dtype=wp.float32, device=replay.device) for name in names
                )
                replay._n_step_transitions.append(transition)
            graph_world_count = int(data["graph_world_count"]) if "graph_world_count" in data else 0
            if graph_world_count:
                replay.reserve_graph_buffers(graph_world_count)
                graph_arrays = {
                    "pending_obs": replay._graph_pending_obs,
                    "pending_actions": replay._graph_pending_actions,
                    "pending_rewards": replay._graph_pending_rewards,
                    "pending_terminateds": replay._graph_pending_terminateds,
                    "pending_truncateds": replay._graph_pending_truncateds,
                    "pending_next_obs": replay._graph_pending_next_obs,
                    "pending_cursor": replay._graph_pending_cursor,
                    "pending_count": replay._graph_pending_count,
                    "position": replay._graph_position,
                    "size": replay._graph_size,
                }
                for name, value in graph_arrays.items():
                    value.assign(data[f"graph_{name}"])
                replay._graph_pending_count_host = int(data["graph_pending_count"][0])
            return replay


class TrainerFlashSAC(TrainerSAC):
    """FlashSAC training preset implemented entirely with Warp.

    The trainer preserves FlashSAC's categorical twin critic, delayed actor,
    learned temperature, unit weight constraints, target entropy rule,
    cosine learning-rate schedule, and zeta-distributed repeated exploration
    noise while reusing PhoenX's device-native SAC update kernels.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_layers: Optional shared hidden layers overriding upstream capacities.
        config: FlashSAC hyperparameters.
        device: Warp device.
        seed: Initializer and exploration seed.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] | None = None,
        config: ConfigFlashSAC | None = None,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ):
        flash_config = config or ConfigFlashSAC()
        if flash_config.target_sigma <= 0.0:
            raise ValueError("target_sigma must be positive")
        if flash_config.noise_zeta_max < 1:
            raise ValueError("noise_zeta_max must be positive")
        if flash_config.noise_zeta_mu <= 0.0:
            raise ValueError("noise_zeta_mu must be positive")
        if flash_config.learning_rate_warmup_steps < 0 or flash_config.learning_rate_decay_steps < 1:
            raise ValueError("learning-rate schedule steps are invalid")
        if (
            min(
                flash_config.actor_num_blocks,
                flash_config.actor_hidden_dim,
                flash_config.critic_num_blocks,
                flash_config.critic_hidden_dim,
            )
            < 1
        ):
            raise ValueError("FlashSAC network dimensions and block counts must be positive")
        reference_backbone = hidden_layers is None
        if flash_config.use_amp and not reference_backbone:
            raise ValueError("use_amp requires the FlashSAC reference backbone")
        contraction_dtype = "float16" if flash_config.use_amp else "float32"
        if hidden_layers is None:
            actor_layers = self._expanded_block_layers(flash_config.actor_hidden_dim, flash_config.actor_num_blocks)
            critic_layers = self._expanded_block_layers(flash_config.critic_hidden_dim, flash_config.critic_num_blocks)
        else:
            actor_layers = hidden_layers
            critic_layers = hidden_layers
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layers=actor_layers,
            critic_hidden_layers=critic_layers,
            config=flash_config,
            device=device,
            seed=seed,
        )
        if reference_backbone:
            flash_config.normalize_observations = False
            self.actor.net = NetworkFlashSAC(
                input_dim=obs_dim,
                hidden_dim=flash_config.actor_hidden_dim,
                num_blocks=flash_config.actor_num_blocks,
                output_dim=action_dim * 2,
                actor_heads=True,
                device=self.device,
                seed=seed,
                contraction_dtype=contraction_dtype,
            )
            critic_kwargs = {
                "input_dim": obs_dim + action_dim,
                "hidden_dim": flash_config.critic_hidden_dim,
                "num_blocks": flash_config.critic_num_blocks,
                "output_dim": flash_config.distributional_atoms if flash_config.distributional_critic else 1,
                "actor_heads": False,
                "device": self.device,
                "contraction_dtype": contraction_dtype,
            }
            self.critic1 = NetworkFlashSAC(**critic_kwargs, seed=seed + 1)
            self.critic2 = NetworkFlashSAC(**critic_kwargs, seed=seed + 2)
            self.target_critic1 = NetworkFlashSAC(**critic_kwargs, seed=seed + 3)
            self.target_critic2 = NetworkFlashSAC(**critic_kwargs, seed=seed + 4)
            self.target_critic1.default_training = True
            self.target_critic2.default_training = True
            self.target_critic1.copy_from(self.critic1)
            self.target_critic2.copy_from(self.critic2)
            self._critic_ensemble = EnsembleNetworkFlashSAC(self.critic1, self.critic2)
            self._target_critic_ensemble = EnsembleNetworkFlashSAC(self.target_critic1, self.target_critic2)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=flash_config.actor_lr)
            self.critic1_optimizer = Adam(self.critic1.parameters(), lr=flash_config.critic_lr)
            self.critic2_optimizer = Adam(self.critic2.parameters(), lr=flash_config.critic_lr)
        self.config = flash_config
        self._replay_buffer: BufferReplayFlashSAC | None = None
        if flash_config.target_entropy is None:
            sigma_sq = float(flash_config.target_sigma) ** 2
            self.target_entropy = 0.5 * self.action_dim * math.log(2.0 * math.pi * math.e * sigma_sq)
        self.actor.log_std_min = -10.0
        self._noise_rng = np.random.default_rng(seed)
        ranks = np.arange(1, flash_config.noise_zeta_max + 1, dtype=np.float64)
        probabilities = ranks ** (-float(flash_config.noise_zeta_mu))
        self._noise_cdf = np.cumsum(probabilities / probabilities.sum())
        self._noise_cdf_device = wp.array(self._noise_cdf.astype(np.float32), device=self.device)
        self._device_noise_repeat_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._device_noise_repeat_steps = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._device_exploration_seed = wp.array([int(seed)], dtype=wp.int32, device=self.device)
        self._device_interaction_seed = wp.array([int(seed)], dtype=wp.int32, device=self.device)
        self._noise_repeat_count = 0
        self._noise_repeat_steps = 0
        self._exploration_seed = int(seed)
        self._device_update_count = wp.array([0], dtype=wp.int32, device=self.device)
        self._device_gradient_update_count = wp.array([0], dtype=wp.int32, device=self.device)
        self._device_update_seed = wp.array([int(seed)], dtype=wp.int32, device=self.device)
        initial_amp_scale = 65536.0 if flash_config.use_amp else 1.0
        self._amp_scale = wp.array([initial_amp_scale], dtype=wp.float32, device=self.device)
        self._loss_scale = self._amp_scale
        self._amp_growth_tracker = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._amp_found_inf = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._amp_step_condition = wp.ones(1, dtype=wp.int32, device=self.device)
        if flash_config.use_amp:
            self.actor_optimizer.step_condition = self._amp_step_condition
            self.critic1_optimizer.step_condition = self._amp_step_condition
            self.critic2_optimizer.step_condition = self._amp_step_condition
        self._device_actor_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._device_actor_skip_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._deterministic_critic_stats = False
        if self.config.normalize_weights:
            self._normalize_online_weights()
            self.target_critic1.copy_from(self.critic1)
            self.target_critic2.copy_from(self.critic2)
            if self.config.use_amp:
                self.actor.net.refresh_contraction_weights()
                self._critic_ensemble.refresh_contraction_weights()
                self._target_critic_ensemble.refresh_contraction_weights()

    @property
    def replay_buffer(self) -> BufferReplayFlashSAC | None:
        """Replay buffer owned by the trainer, if transition collection has started."""

        return self._replay_buffer

    def set_pbt_learning_rates(self, actor_lr: float, critic_lr: float, alpha_lr: float) -> None:
        """Set graph-safe effective learning rates for online tuning.

        The optimizer base rates and captured learning-rate schedule remain
        unchanged. Device-resident multipliers apply these effective rates on
        subsequent eager or captured updates without rebuilding graphs.

        Args:
            actor_lr: Effective actor learning rate.
            critic_lr: Effective learning rate for both critics.
            alpha_lr: Effective entropy-temperature learning rate.
        """

        rates = (float(actor_lr), float(critic_lr), float(alpha_lr))
        if not all(math.isfinite(rate) and rate > 0.0 for rate in rates):
            raise ValueError("PBT learning rates must be finite and positive")
        self.actor_optimizer.set_pbt_lr(rates[0])
        self.critic1_optimizer.set_pbt_lr(rates[1])
        self.critic2_optimizer.set_pbt_lr(rates[1])
        self.alpha_optimizer.set_pbt_lr(rates[2])

    def copy_training_state_from(self, source: TrainerFlashSAC) -> None:
        """Copy complete learner state without replacing owned allocations.

        Replay storage is deliberately excluded so multiple compatible
        trainers can consume one shared replay. Existing CUDA graphs remain
        valid because every destination array keeps its address.

        Args:
            source: Compatible trainer donating networks, targets, optimizer
                state, normalization state, temperature, counters, and AMP
                scaler state.
        """

        if not isinstance(source, TrainerFlashSAC):
            raise TypeError("source must be a TrainerFlashSAC")
        if source is self:
            return
        if self.device != source.device or self.obs_dim != source.obs_dim or self.action_dim != source.action_dim:
            raise ValueError("FlashSAC trainer devices and dimensions must match")
        tunable = {"actor_lr", "critic_lr", "alpha_lr"}
        for config_field in fields(ConfigFlashSAC):
            if config_field.name not in tunable and getattr(self.config, config_field.name) != getattr(
                source.config, config_field.name
            ):
                raise ValueError(f"FlashSAC config field '{config_field.name}' is incompatible")
        networks = (
            (self.actor.net, source.actor.net),
            (self.critic1, source.critic1),
            (self.critic2, source.critic2),
            (self.target_critic1, source.target_critic1),
            (self.target_critic2, source.target_critic2),
        )
        for destination, donor in networks:
            if type(destination) is not type(donor) or destination.layer_sizes != donor.layer_sizes:
                raise ValueError("FlashSAC network structures do not match")
            destination.copy_from(donor)
        wp.copy(self.actor.log_std, source.actor.log_std)
        for destination, donor in (
            (self.log_alpha, source.log_alpha),
            (self._alpha, source._alpha),
            (self._obs_mean, source._obs_mean),
            (self._obs_m2, source._obs_m2),
            (self._obs_count, source._obs_count),
            (self._device_update_count, source._device_update_count),
            (self._device_gradient_update_count, source._device_gradient_update_count),
            (self._device_update_seed, source._device_update_seed),
            (self._device_noise_repeat_count, source._device_noise_repeat_count),
            (self._device_noise_repeat_steps, source._device_noise_repeat_steps),
            (self._device_exploration_seed, source._device_exploration_seed),
            (self._device_interaction_seed, source._device_interaction_seed),
            (self._amp_scale, source._amp_scale),
            (self._amp_growth_tracker, source._amp_growth_tracker),
            (self._amp_found_inf, source._amp_found_inf),
            (self._amp_step_condition, source._amp_step_condition),
            (self._device_actor_condition, source._device_actor_condition),
            (self._device_actor_skip_condition, source._device_actor_skip_condition),
        ):
            wp.copy(destination, donor)
        for destination, donor in (
            (self.actor_optimizer, source.actor_optimizer),
            (self.critic1_optimizer, source.critic1_optimizer),
            (self.critic2_optimizer, source.critic2_optimizer),
            (self.alpha_optimizer, source.alpha_optimizer),
        ):
            _copy_optimizer_state(destination, donor)
        self._update_count = source._update_count
        self._gradient_update_count = source._gradient_update_count
        self._noise_repeat_count = source._noise_repeat_count
        self._noise_repeat_steps = source._noise_repeat_steps
        self._exploration_seed = source._exploration_seed
        self._noise_rng.bit_generator.state = json.loads(json.dumps(source._noise_rng.bit_generator.state))
        if isinstance(self.actor.net, NetworkFlashSAC):
            self.actor.net.refresh_contraction_weights()
            self._critic_ensemble.refresh_contraction_weights()
            self._target_critic_ensemble.refresh_contraction_weights()

    def initialize_replay_buffer(self) -> BufferReplayFlashSAC:
        """Allocate the configured replay buffer and return it.

        Allocation is lazy because the upstream one-million-row default is
        substantial for G1-sized observations. Calling :meth:`process_transition`
        also initializes the buffer on first use.
        """

        if self._replay_buffer is None:
            self._replay_buffer = BufferReplayFlashSAC(
                minimum_size=self.config.buffer_min_length,
                capacity=self.config.buffer_max_length,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                batch_size=self.config.sample_batch_size,
                n_step=self.config.n_step,
                gamma=self.config.gamma,
                normalize_rewards=self.config.normalize_rewards,
                normalized_return_max=self.config.normalized_return_max,
                device=self.device,
            )
        return self._replay_buffer

    def process_transition(
        self,
        obs: wp.array2d[wp.float32],
        actions: wp.array2d[wp.float32],
        rewards: wp.array[wp.float32],
        terminateds: wp.array[wp.float32],
        next_obs: wp.array2d[wp.float32],
        truncateds: wp.array[wp.float32] | None = None,
    ) -> None:
        """Add one vectorized environment transition to the owned replay buffer."""

        self.initialize_replay_buffer().add_batch(
            obs,
            actions,
            rewards,
            terminateds,
            next_obs,
            truncateds=truncateds,
        )

    def can_start_training(self) -> bool:
        """Return whether the owned replay buffer has completed warmup."""

        return self._replay_buffer is not None and self._replay_buffer.can_sample()

    def prepare_training_graph(
        self,
        env: EnvFlashSAC,
        *,
        warmup_interaction_steps: int | None = None,
        updates_per_step: int = 2,
        interactions_per_graph: int = 2,
        seed: int | None = None,
        reset_at_start: bool = True,
        overlap: bool = False,
    ) -> GraphFlashSACTraining:
        """Warm graph-native replay and capture steady-state training.

        This is the high-level entry point for a fresh captured training run.
        Replay graph buffers are reserved before warmup, so n-step pending
        transitions remain device-resident and can continue unchanged in the
        captured steady-state cadence.

        Args:
            env: Vectorized environment matching the trainer dimensions.
            warmup_interaction_steps: Eager graph-native collection steps before
                capture. ``None`` collects the minimum needed for replay sampling.
            updates_per_step: Learner updates per environment interaction.
            interactions_per_graph: Environment interactions per graph launch.
            seed: Base exploration and graph seed. Uses the trainer seed by default.
            reset_at_start: Whether to reset the environment before warmup.
            overlap: Whether rollout and learner graphs may execute concurrently
                using a lagged FP32 policy snapshot and pre-sampled batches.

        Returns:
            Captured steady-state training cadence ready for repeated launches.
        """

        policy_action_dim = int(getattr(env, "policy_action_dim", env.action_dim))
        if self.obs_dim != env.obs_dim or self.action_dim != policy_action_dim:
            raise ValueError("FlashSAC trainer dimensions do not match environment policy interface")
        replay = self.initialize_replay_buffer()
        if replay._n_step_transitions:
            raise RuntimeError("prepare_training_graph requires fresh or graph-native replay state")
        replay.reserve_graph_buffers(env.world_count)
        if replay.size <= 0 or not replay.can_sample():
            pending = replay._graph_pending_count_host
            rows_needed = max(replay.minimum_size, 1) - replay.size
            row_steps = (rows_needed + env.world_count - 1) // env.world_count
            minimum_steps = row_steps + max(replay.n_step - pending - 1, 0)
            steps = minimum_steps if warmup_interaction_steps is None else int(warmup_interaction_steps)
            if steps < 0:
                raise ValueError("warmup_interaction_steps must be non-negative")
            obs = env.reset() if reset_at_start else env.observe()
            pre_step_obs = wp.empty_like(obs)
            zero_truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=self.device)
            seed_base = self.seed if seed is None else int(seed)
            for step in range(steps):
                actions, _log_probs = self.act(obs, seed=seed_base + step)
                wp.copy(pre_step_obs, obs)
                next_obs, rewards, dones = env.step(actions)
                replay.add_batch_graph(
                    pre_step_obs,
                    actions,
                    rewards,
                    getattr(env, "step_terminateds", dones),
                    getattr(env, "step_next_obs", next_obs),
                    truncateds=getattr(env, "step_truncateds", zero_truncateds),
                )
                obs = next_obs
            if not replay.can_sample():
                raise RuntimeError(f"warmup produced {replay.size} replay rows; {replay.minimum_size} are required")
        return self.capture_training_graph(
            env,
            replay,
            updates_per_step=updates_per_step,
            interactions_per_graph=interactions_per_graph,
            seed=seed,
            overlap=overlap,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save complete trainer state except the potentially large replay buffer.

        Replay serialization is intentionally separate from trainer checkpoints.
        """

        data: dict[str, np.ndarray] = {
            "obs_dim": np.asarray(self.obs_dim, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "reference_backbone": np.asarray(isinstance(self.actor.net, NetworkFlashSAC), dtype=np.bool_),
            "actor_hidden_layers": np.asarray(self.actor.net.layer_sizes[1:-1], dtype=np.int64),
            "critic_hidden_layers": np.asarray(self.critic1.layer_sizes[1:-1], dtype=np.int64),
            "actor_log_std": self.actor.log_std.numpy(),
            "log_alpha": self.log_alpha.numpy(),
            "alpha": self._alpha.numpy(),
            "update_count": np.asarray(self._update_count, dtype=np.int64),
            "gradient_update_count": np.asarray(self._gradient_update_count, dtype=np.int64),
            "noise_repeat_count": np.asarray(self._noise_repeat_count, dtype=np.int64),
            "noise_repeat_steps": np.asarray(self._noise_repeat_steps, dtype=np.int64),
            "exploration_seed": np.asarray(self._exploration_seed, dtype=np.int64),
            "device_noise_repeat_count": self._device_noise_repeat_count.numpy(),
            "device_noise_repeat_steps": self._device_noise_repeat_steps.numpy(),
            "device_exploration_seed": self._device_exploration_seed.numpy(),
            "device_interaction_seed": self._device_interaction_seed.numpy(),
            "noise_rng_state": np.asarray(json.dumps(self._noise_rng.bit_generator.state)),
            "obs_mean": self._obs_mean.numpy(),
            "obs_m2": self._obs_m2.numpy(),
            "obs_count": self._obs_count.numpy(),
            "amp_scale": self._amp_scale.numpy(),
            "amp_growth_tracker": self._amp_growth_tracker.numpy(),
        }
        for key, value in asdict(self.config).items():
            none_key = f"config_{key}_is_none"
            if value is None:
                data[none_key] = np.asarray(True, dtype=np.bool_)
            else:
                data[none_key] = np.asarray(False, dtype=np.bool_)
                data[f"config_{key}"] = np.asarray(value)
        for prefix, network in (
            ("actor", self.actor.net),
            ("critic1", self.critic1),
            ("critic2", self.critic2),
            ("target_critic1", self.target_critic1),
            ("target_critic2", self.target_critic2),
        ):
            _pack_flash_sac_network(data, prefix, network)
        for prefix, optimizer in (
            ("actor_optimizer", self.actor_optimizer),
            ("critic1_optimizer", self.critic1_optimizer),
            ("critic2_optimizer", self.critic2_optimizer),
            ("alpha_optimizer", self.alpha_optimizer),
        ):
            _pack_optimizer(data, prefix, optimizer)
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, **data)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        config: ConfigFlashSAC | None = None,
        device: wp.context.Devicelike = None,
    ) -> TrainerFlashSAC:
        """Restore networks, targets, optimizers, temperature, and counters.

        Args:
            path: Input ``.npz`` checkpoint path.
            config: Optional optimizer configuration override.
            device: Warp device for restored arrays.

        Returns:
            Restored FlashSAC trainer without replay contents.
        """

        with np.load(Path(path), allow_pickle=False) as data:
            saved_config = config or _config_from_flash_sac_checkpoint(data)
            actor_hidden = tuple(int(width) for width in data["actor_hidden_layers"])
            critic_hidden = tuple(int(width) for width in data["critic_hidden_layers"])
            reference_backbone = (
                bool(data["reference_backbone"])
                if "reference_backbone" in data
                else str(data["actor_network_type"].item()) == "flash_sac_reference"
            )
            hidden_layers = None if reference_backbone else actor_hidden
            trainer = cls(
                obs_dim=int(data["obs_dim"]),
                action_dim=int(data["action_dim"]),
                hidden_layers=hidden_layers,
                config=saved_config,
                device=device,
                seed=int(data["seed"]),
            )
            if (
                tuple(trainer.actor.net.layer_sizes[1:-1]) != actor_hidden
                or tuple(trainer.critic1.layer_sizes[1:-1]) != critic_hidden
            ):
                raise ValueError("Checkpoint network architecture does not match configuration")
            for prefix, network in (
                ("actor", trainer.actor.net),
                ("critic1", trainer.critic1),
                ("critic2", trainer.critic2),
                ("target_critic1", trainer.target_critic1),
                ("target_critic2", trainer.target_critic2),
            ):
                _unpack_flash_sac_network(data, prefix, network)
            trainer.actor.log_std.assign(data["actor_log_std"])
            trainer.log_alpha.assign(data["log_alpha"])
            trainer._alpha.assign(data["alpha"])
            trainer._obs_mean.assign(data["obs_mean"])
            trainer._obs_m2.assign(data["obs_m2"])
            trainer._obs_count.assign(data["obs_count"])
            trainer._update_count = int(data["update_count"])
            trainer._gradient_update_count = int(data["gradient_update_count"])
            trainer._device_update_count.assign(np.asarray([trainer._update_count], dtype=np.int32))
            trainer._device_gradient_update_count.assign(np.asarray([trainer._gradient_update_count], dtype=np.int32))
            if "amp_scale" in data:
                trainer._amp_scale.assign(data["amp_scale"])
                trainer._amp_growth_tracker.assign(data["amp_growth_tracker"])
            if "noise_rng_state" in data:
                trainer._noise_repeat_count = int(data["noise_repeat_count"])
                trainer._noise_repeat_steps = int(data["noise_repeat_steps"])
                trainer._exploration_seed = int(data["exploration_seed"])
                trainer._noise_rng.bit_generator.state = json.loads(str(data["noise_rng_state"].item()))
            if "device_interaction_seed" in data:
                trainer._device_noise_repeat_count.assign(data["device_noise_repeat_count"])
                trainer._device_noise_repeat_steps.assign(data["device_noise_repeat_steps"])
                trainer._device_exploration_seed.assign(data["device_exploration_seed"])
                trainer._device_interaction_seed.assign(data["device_interaction_seed"])
            for prefix, optimizer in (
                ("actor_optimizer", trainer.actor_optimizer),
                ("critic1_optimizer", trainer.critic1_optimizer),
                ("critic2_optimizer", trainer.critic2_optimizer),
                ("alpha_optimizer", trainer.alpha_optimizer),
            ):
                _unpack_optimizer(data, prefix, optimizer)
            if reference_backbone:
                trainer.actor.net.refresh_contraction_weights()
                trainer._critic_ensemble.refresh_contraction_weights()
                trainer._target_critic_ensemble.refresh_contraction_weights()
            return trainer

    def _graph_update_operations(
        self,
        batch: BatchSAC,
        *,
        include_actor: bool,
        seed_base: int,
    ) -> None:
        """Record one allocation-stable learner update into an active capture."""

        wp.launch(
            _prepare_flash_sac_graph_update_kernel,
            dim=1,
            inputs=[
                self._device_gradient_update_count,
                self._device_update_count,
                int(seed_base),
                int(self.config.policy_frequency),
                int(self.config.learning_rate_warmup_steps),
                int(self.config.learning_rate_decay_steps),
                float(self.config.actor_lr),
                float(self.config.learning_rate_end),
                float(self.config.actor_lr),
                float(self.config.critic_lr),
                float(self.config.alpha_lr),
            ],
            outputs=[
                self._device_actor_condition,
                self._device_actor_skip_condition,
                self._device_update_seed,
                self.actor_optimizer.lr_scale,
                self.critic1_optimizer.lr_scale,
                self.critic2_optimizer.lr_scale,
                self.alpha_optimizer.lr_scale,
            ],
            device=self.device,
        )
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._actor_loss], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._alpha_loss], device=self.device)
        if include_actor:
            self._update_actor(batch, seed=0, seed_counter=self._device_update_seed, seed_offset=0)
            if self.config.auto_alpha:
                self._update_alpha(batch, seed=0)
        self._update_critics(
            batch,
            seed=0,
            seed_counter=self._device_update_seed,
            seed_offset=2,
        )
        self.target_critic1.soft_update_from(self.critic1, self.config.tau)
        self.target_critic2.soft_update_from(self.critic2, self.config.tau)
        if self.config.use_amp:
            self._target_critic_ensemble.refresh_contraction_weights()
        wp.launch(
            seed_counter_increment_kernel,
            dim=1,
            inputs=[self._device_gradient_update_count, 1],
            device=self.device,
        )
        wp.launch(
            seed_counter_increment_kernel,
            dim=1,
            inputs=[self._device_update_count, 1],
            device=self.device,
        )

    def capture_update_graph(
        self,
        batch: BatchSAC,
        *,
        seed: int | None = None,
    ) -> GraphFlashSACUpdate:
        """Capture one algorithmically complete update for repeated CUDA replay.

        The leapfrog graphs advance learning-rate, optimizer, and stochastic-seed
        state on the device. A host-mirrored counter selects the delayed actor
        graph without a device synchronization. ``batch`` arrays must retain
        their addresses; callers may overwrite their contents between launches.
        """

        if not self.device.is_cuda or not self.device.is_mempool_enabled:
            raise RuntimeError("FlashSAC update graphs require CUDA with memory pools enabled")
        if int(self.config.update_steps) != 1:
            raise ValueError("FlashSAC update graph capture requires update_steps=1")
        if self.config.normalize_observations:
            raise ValueError("FlashSAC update graph capture requires reference internal observation normalization")
        if int(batch.obs.shape[1]) != self.obs_dim or int(batch.next_obs.shape[1]) != self.obs_dim:
            raise ValueError("Batch observation dimensions do not match trainer")
        if int(batch.actions.shape[1]) != self.action_dim:
            raise ValueError("Batch action dimensions do not match trainer")
        if min(self.config.actor_lr, self.config.critic_lr, self.config.alpha_lr) <= 0.0:
            raise ValueError("FlashSAC update graph capture requires positive optimizer learning rates")

        training_rows = int(batch.obs.shape[0]) * 2
        if isinstance(self.actor.net, NetworkFlashSAC):
            self.actor.net.reserve_training_buffers(training_rows)
        for network in (self.critic1, self.critic2, self.target_critic1, self.target_critic2):
            if isinstance(network, NetworkFlashSAC):
                network.reserve_training_buffers(training_rows)

        self._device_update_count.assign(np.asarray([self._update_count], dtype=np.int32))
        self._device_gradient_update_count.assign(np.asarray([self._gradient_update_count], dtype=np.int32))
        seed_base = self.seed if seed is None else int(seed)
        self.actor_optimizer.lr = float(self.config.actor_lr)
        self.critic1_optimizer.lr = float(self.config.critic_lr)
        self.critic2_optimizer.lr = float(self.config.critic_lr)
        self.alpha_optimizer.lr = float(self.config.alpha_lr)

        def capture_one_update(*, include_actor: bool) -> object:
            with wp.ScopedCapture(device=self.device) as capture:
                self._graph_update_operations(batch, include_actor=include_actor, seed_base=seed_base)
            return capture.graph

        actor_graph = capture_one_update(include_actor=True)
        critic_graph = capture_one_update(include_actor=False)
        return GraphFlashSACUpdate(
            trainer=self,
            actor_graph=actor_graph,
            critic_graph=critic_graph,
            batch=batch,
            policy_frequency=int(self.config.policy_frequency),
        )

    def capture_training_graph(
        self,
        env: EnvFlashSAC,
        replay: BufferReplayFlashSAC,
        *,
        updates_per_step: int = 2,
        interactions_per_graph: int = 2,
        seed: int | None = None,
        overlap: bool = False,
    ) -> GraphFlashSACTraining:
        """Capture steady-state interaction, replay, and learner updates.

        ``overlap=True`` runs rollout and fixed-batch learner graphs concurrently.
        Transitions collected by one launch become eligible for learner sampling
        during preparation of the next launch. The collector uses an FP32 policy
        snapshot copied after the preceding rollout and learner streams join.

        Call :meth:`GraphFlashSACTraining.synchronize` before directly inspecting
        device state or checkpointing. :meth:`GraphFlashSACTraining.close` drains
        all streams and releases every captured phase graph.

        Args:
            env: Vectorized environment matching the trainer dimensions.
            replay: Warm graph-native replay buffer owned by this cadence.
            updates_per_step: Learner updates per environment interaction.
            interactions_per_graph: Environment interactions per graph launch.
            seed: Base exploration and update seed. Uses the trainer seed by default.
            overlap: Whether to overlap rollout and learner work with phase buffers.

        Returns:
            Captured training cadence ready for repeated launches.
        """

        updates = int(updates_per_step)
        interactions = int(interactions_per_graph)
        if not self.device.is_cuda or not self.device.is_mempool_enabled:
            raise RuntimeError("FlashSAC training graphs require CUDA with memory pools enabled")
        if env.device != self.device or replay.device != self.device:
            raise ValueError("environment, replay, and trainer must use the same device")
        if self.obs_dim != env.obs_dim or self.action_dim != int(getattr(env, "policy_action_dim", env.action_dim)):
            raise ValueError("FlashSAC trainer dimensions do not match environment policy interface")
        if updates <= 0 or interactions <= 0:
            raise ValueError("updates_per_step and interactions_per_graph must be positive")
        total_updates = updates * interactions
        if total_updates % int(self.config.policy_frequency) != 0:
            raise ValueError("captured updates must span a complete policy-frequency cadence")
        if interactions % 2 != 0:
            raise ValueError("interactions_per_graph must be even for environment state-buffer parity")
        if int(self.config.update_steps) != 1 or self.config.normalize_observations:
            raise ValueError("training graph requires update_steps=1 and reference internal normalization")
        if not replay.can_sample():
            raise RuntimeError("warm replay eagerly before capturing steady-state training")

        replay.reserve_graph_buffers(env.world_count)
        self.reserve_buffers(env.world_count)
        training_rows = int(replay.batch_size) * 2
        if isinstance(self.actor.net, NetworkFlashSAC):
            self.actor.net.reserve_training_buffers(training_rows)
        for network in (self.critic1, self.critic2, self.target_critic1, self.target_critic2):
            if isinstance(network, NetworkFlashSAC):
                network.reserve_training_buffers(training_rows)
        pre_step_obs = wp.empty((env.world_count, self.obs_dim), dtype=wp.float32, device=self.device)
        zero_truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=self.device)
        env_seed_counter = wp.array([int(self.seed if seed is None else seed)], dtype=wp.int32, device=self.device)
        if hasattr(env, "use_reset_seed_counter"):
            env.use_reset_seed_counter(env_seed_counter)
        if hasattr(env, "use_command_seed_counter"):
            env.use_command_seed_counter(env_seed_counter)
        self._device_update_count.assign(np.asarray([self._update_count], dtype=np.int32))
        self._device_gradient_update_count.assign(np.asarray([self._gradient_update_count], dtype=np.int32))
        if seed is not None:
            self._device_interaction_seed.assign(np.asarray([int(seed)], dtype=np.int32))
        self.actor_optimizer.lr = float(self.config.actor_lr)
        self.critic1_optimizer.lr = float(self.config.critic_lr)
        self.critic2_optimizer.lr = float(self.config.critic_lr)
        self.alpha_optimizer.lr = float(self.config.alpha_lr)
        seed_base = self.seed if seed is None else int(seed)
        start_gradient_update = self._gradient_update_count
        sim_time_before = getattr(env, "sim_time", None)
        if overlap:
            return self._capture_overlapped_training_graph(
                env=env,
                replay=replay,
                interactions=interactions,
                total_updates=total_updates,
                seed_base=seed_base,
                start_gradient_update=start_gradient_update,
                pre_step_obs=pre_step_obs,
                zero_truncateds=zero_truncateds,
                env_seed_counter=env_seed_counter,
                sim_time_before=sim_time_before,
            )

        with wp.ScopedCapture(device=self.device) as capture:
            local_update = 0
            for interaction in range(interactions):
                wp.copy(pre_step_obs, env.obs)
                exploration_seed = self.prepare_graph_exploration_seed()
                actions, _log_probs = self.act_reuse_seed_counter(pre_step_obs, seed_counter=exploration_seed)
                next_obs, rewards, dones = env.step(actions)
                replay_next_obs = getattr(env, "step_next_obs", next_obs)
                truncateds = getattr(env, "step_truncateds", zero_truncateds)
                terminateds = getattr(env, "step_terminateds", dones)
                replay.add_batch_graph(
                    pre_step_obs,
                    actions,
                    rewards,
                    terminateds,
                    replay_next_obs,
                    truncateds=truncateds,
                )
                for update_index in range(updates):
                    batch = replay.sample_graph_seed_counter(
                        self._device_update_count,
                        seed_offset=interaction * updates + update_index + 101,
                    )
                    include_actor = (start_gradient_update + local_update) % int(self.config.policy_frequency) == 0
                    self._graph_update_operations(
                        batch,
                        include_actor=include_actor,
                        seed_base=seed_base,
                    )
                    local_update += 1
        if sim_time_before is not None:
            env.sim_time = sim_time_before
        return GraphFlashSACTraining(
            trainer=self,
            replay=replay,
            env=env,
            graph=capture.graph,
            interactions_per_launch=interactions,
            updates_per_launch=total_updates,
            retained_arrays=(pre_step_obs, zero_truncateds, env_seed_counter),
        )

    def _capture_overlapped_training_graph(
        self,
        *,
        env: EnvFlashSAC,
        replay: BufferReplayFlashSAC,
        interactions: int,
        total_updates: int,
        seed_base: int,
        start_gradient_update: int,
        pre_step_obs: wp.array2d[wp.float32],
        zero_truncateds: wp.array[wp.float32],
        env_seed_counter: wp.array[wp.int32],
        sim_time_before: float | None,
    ) -> GraphFlashSACTraining:
        """Capture a two-phase rollout/learner leapfrog without replay races."""

        if not isinstance(self.actor.net, NetworkFlashSAC):
            raise ValueError("overlapped training requires the FlashSAC reference backbone")
        source_net = self.actor.net
        rollout_actor = copy.copy(self.actor)
        rollout_actor.net = NetworkFlashSAC(
            input_dim=source_net.input_dim,
            hidden_dim=source_net.hidden_dim,
            num_blocks=source_net.num_blocks,
            output_dim=source_net.output_dim,
            actor_heads=True,
            device=self.device,
            seed=self.seed,
            contraction_dtype="float32",
        )
        rollout_actor.log_std = wp.clone(self.actor.log_std)
        rollout_actor._sample_reuse_capacity = 0
        rollout_actor._sample_reuse_actions = None
        rollout_actor._sample_reuse_log_probs = None
        rollout_actor._sample_reuse_eps = None
        rollout_actor.copy_from(self.actor)
        rollout_actor.reserve_reuse_buffers(env.world_count)

        phase_batches = tuple(
            tuple(_allocate_flash_sac_batch(self) for _ in range(total_updates)) for _phase in range(2)
        )

        def collect() -> None:
            for _interaction in range(interactions):
                wp.copy(pre_step_obs, env.obs)
                exploration_seed = self.prepare_graph_exploration_seed()
                actions, _log_probs, _policy_out = rollout_actor.sample_reuse_seed_counter(
                    pre_step_obs,
                    seed_counter=exploration_seed,
                )
                next_obs, rewards, dones = env.step(actions)
                replay.add_batch_graph(
                    pre_step_obs,
                    actions,
                    rewards,
                    getattr(env, "step_terminateds", dones),
                    getattr(env, "step_next_obs", next_obs),
                    truncateds=getattr(env, "step_truncateds", zero_truncateds),
                )

        def prepare(phase: int) -> None:
            rollout_actor.copy_from(self.actor)
            for update_index, batch in enumerate(phase_batches[phase]):
                sampled = replay.sample_graph_seed_counter(
                    self._device_update_count,
                    seed_offset=update_index + 101,
                )
                _copy_flash_sac_batch(batch, sampled)

        def update(phase: int) -> None:
            for update_index, batch in enumerate(phase_batches[phase]):
                include_actor = (start_gradient_update + update_index) % int(self.config.policy_frequency) == 0
                self._graph_update_operations(
                    batch,
                    include_actor=include_actor,
                    seed_base=seed_base,
                )

        rollout_stream = wp.Stream(self.device, priority=-1)
        update_stream = wp.Stream(self.device)
        prepare_stream = wp.Stream(self.device)
        rollout_graph = _capture_flash_stream_graph(rollout_stream, self.device, collect)
        update_graphs = tuple(
            _capture_flash_stream_graph(update_stream, self.device, lambda phase=phase: update(phase))
            for phase in range(2)
        )
        prepare_graphs = tuple(
            _capture_flash_stream_graph(prepare_stream, self.device, lambda phase=phase: prepare(phase))
            for phase in range(2)
        )
        wp.capture_launch(prepare_graphs[0], stream=prepare_stream)
        wp.synchronize_device(self.device)
        if sim_time_before is not None:
            env.sim_time = sim_time_before
        return GraphFlashSACTraining(
            trainer=self,
            replay=replay,
            env=env,
            graph=None,
            interactions_per_launch=interactions,
            updates_per_launch=total_updates,
            retained_arrays=(pre_step_obs, zero_truncateds, env_seed_counter),
            rollout_graph=rollout_graph,
            update_graphs=update_graphs,
            prepare_graphs=prepare_graphs,
            rollout_stream=rollout_stream,
            update_stream=update_stream,
            prepare_stream=prepare_stream,
            rollout_actor=rollout_actor,
            phase_batches=phase_batches,
        )

    def prepare_graph_exploration_seed(self) -> wp.array[wp.int32]:
        """Advance device-resident zeta exploration duration state."""

        wp.launch(
            _prepare_flash_sac_exploration_kernel,
            dim=1,
            inputs=[
                self._device_interaction_seed,
                self._noise_cdf_device,
                self._device_noise_repeat_count,
                self._device_noise_repeat_steps,
            ],
            outputs=[self._device_exploration_seed],
            device=self.device,
        )
        wp.launch(
            seed_counter_increment_kernel,
            dim=1,
            inputs=[self._device_interaction_seed, 1],
            device=self.device,
        )
        return self._device_exploration_seed

    def reserve_buffers(self, batch_size: int) -> None:
        """Reserve graph-replay-safe interaction buffers for a fixed batch size."""

        rows = int(batch_size)
        if rows <= 0:
            raise ValueError("batch_size must be positive")
        self.actor.reserve_reuse_buffers(rows)

    def act_reuse_seed_counter(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed_counter: wp.array[wp.int32],
        seed_offset: int = 0,
        deterministic: bool = False,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32]]:
        """Sample into persistent buffers using a device-resident seed.

        This path is suitable for CUDA graph replay. The caller controls
        temporally correlated exploration by retaining a seed-counter value
        for the desired number of environment steps.
        """

        normalized_obs = self._normalize_observations(obs)
        actions, log_probs, _policy_out = self.actor.sample_reuse_seed_counter(
            normalized_obs,
            seed_counter=seed_counter,
            seed_offset=int(seed_offset),
            deterministic=deterministic,
        )
        return actions, log_probs

    def act(
        self,
        obs: wp.array,
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array, wp.array]:
        """Sample an action, reusing exploration noise for zeta-distributed durations."""

        if deterministic:
            return super().act(obs, seed=seed, deterministic=True)
        if self._noise_repeat_count >= self._noise_repeat_steps:
            self._exploration_seed = int(seed)
            uniform = float(self._noise_rng.random())
            self._noise_repeat_steps = int(np.searchsorted(self._noise_cdf, uniform, side="right")) + 1
            self._noise_repeat_count = 0
        self._noise_repeat_count += 1
        return super().act(obs, seed=self._exploration_seed, deterministic=False)

    def _update_alpha(self, batch: BatchSAC, *, seed: int) -> None:
        """Update temperature from the actor update's detached entropy sample."""

        del seed
        log_probs = getattr(self, "_actor_update_log_probs", None)
        if log_probs is None or int(log_probs.shape[0]) != batch.batch_size:
            raise RuntimeError("FlashSAC temperature update requires a preceding actor update")
        log_probs_2d = log_probs.reshape((1, batch.batch_size))
        log_alpha_2d = self.log_alpha.reshape((1, 1))
        log_alpha_grad_2d = self.log_alpha.grad.reshape((1, 1))
        wp.launch(
            _flash_sac_alpha_loss_kernel,
            dim=(1, 32),
            inputs=[log_probs_2d, log_alpha_2d, batch.batch_size, self.target_entropy],
            outputs=[self._alpha_loss, log_alpha_grad_2d],
            block_dim=32,
            device=self.device,
        )
        self.alpha_optimizer.step()
        wp.launch(
            sac_refresh_alpha_kernel,
            dim=1,
            inputs=[self.log_alpha],
            outputs=[self._alpha],
            device=self.device,
        )

    def update(
        self,
        batch: BatchSAC | None = None,
        *,
        seed: int | None = None,
        read_stats: bool = True,
    ) -> StatsSACUpdate:
        """Sample owned replay when needed and update networks in upstream order."""

        if batch is None:
            if not self.can_start_training() or self._replay_buffer is None:
                raise RuntimeError("FlashSAC replay buffer has not completed warmup")
            sample_seed = self.seed + self._update_count * 9973 if seed is None else int(seed)
            batch = self._replay_buffer.sample(seed=sample_seed)
        if int(batch.obs.shape[1]) != self.obs_dim or int(batch.next_obs.shape[1]) != self.obs_dim:
            raise ValueError("Batch observation dimensions do not match trainer")
        if int(batch.actions.shape[1]) != self.action_dim:
            raise ValueError("Batch action dimensions do not match trainer")

        batch = self._normalize_batch(batch)
        base_seed = self.seed + self._update_count * 9973 if seed is None else int(seed)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._actor_loss], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._alpha_loss], device=self.device)
        for i in range(int(self.config.update_steps)):
            learning_rate = self._scheduled_learning_rate(self._gradient_update_count)
            self.actor_optimizer.lr = learning_rate
            self.critic1_optimizer.lr = learning_rate
            self.critic2_optimizer.lr = learning_rate
            self.alpha_optimizer.lr = learning_rate
            update_seed = base_seed + 3 * i
            if self._gradient_update_count % int(self.config.policy_frequency) == 0:
                self._update_actor(batch, seed=update_seed)
                if self.config.auto_alpha:
                    self._update_alpha(batch, seed=update_seed + 1)
            self._deterministic_critic_stats = bool(read_stats and i + 1 == int(self.config.update_steps))
            self._update_critics(batch, seed=update_seed + 2)
            self.target_critic1.soft_update_from(self.critic1, self.config.tau)
            self.target_critic2.soft_update_from(self.critic2, self.config.tau)
            if self.config.use_amp:
                if self._target_critic_ensemble is not None:
                    self._target_critic_ensemble.refresh_contraction_weights()
                else:
                    self.target_critic1.refresh_contraction_weights()
                    self.target_critic2.refresh_contraction_weights()
            self._gradient_update_count += 1
        self._update_count += 1
        wp.launch(
            seed_counter_increment_kernel,
            dim=1,
            inputs=[self._device_gradient_update_count, int(self.config.update_steps)],
            device=self.device,
        )
        wp.launch(
            seed_counter_increment_kernel,
            dim=1,
            inputs=[self._device_update_count, 1],
            device=self.device,
        )
        self._deterministic_critic_stats = False
        if read_stats:
            return self._read_update_stats()
        return StatsSACUpdate(actor_loss=0.0, critic_loss=0.0, alpha_loss=0.0, alpha=0.0)

    @staticmethod
    def _expanded_block_layers(hidden_dim: int, num_blocks: int) -> tuple[int, ...]:
        """Return widths matching upstream embedder and block expansions."""

        layers = [int(hidden_dim)]
        for _ in range(int(num_blocks)):
            layers.extend((int(hidden_dim) * 4, int(hidden_dim)))
        return tuple(layers)

    def _scheduled_learning_rate(self, step: int) -> float:
        warmup_steps = int(self.config.learning_rate_warmup_steps)
        peak = float(self.config.actor_lr)
        if warmup_steps > 0 and step < warmup_steps:
            return peak * float(step + 1) / float(warmup_steps)
        decay_step = min(max(step - warmup_steps, 0), int(self.config.learning_rate_decay_steps))
        progress = float(decay_step) / float(self.config.learning_rate_decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        end = float(self.config.learning_rate_end)
        return end + (peak - end) * cosine

    def _normalize_online_weights(self) -> None:
        self.actor.normalize_weights()
        self.critic1.normalize_weights()
        self.critic2.normalize_weights()
        if self.config.use_amp:
            self.actor.net.refresh_contraction_weights()
            self._critic_ensemble.refresh_contraction_weights()

    def _update_actor(
        self,
        batch: BatchSAC,
        *,
        seed: int,
        seed_counter: wp.array[wp.int32] | None = None,
        seed_offset: int = 0,
    ) -> None:
        super()._update_actor(batch, seed=seed, seed_counter=seed_counter, seed_offset=seed_offset)
        if self.config.normalize_weights:
            self.actor.normalize_weights()
        if self.config.use_amp:
            self.actor.net.refresh_contraction_weights()

    def _update_critics(
        self,
        batch: BatchSAC,
        *,
        seed: int,
        seed_counter: wp.array[wp.int32] | None = None,
        seed_offset: int = 0,
    ) -> None:
        super()._update_critics(batch, seed=seed, seed_counter=seed_counter, seed_offset=seed_offset)
        if self.config.normalize_weights:
            self.critic1.normalize_weights()
            self.critic2.normalize_weights()
        if self.config.use_amp:
            if self._critic_ensemble is not None:
                self._critic_ensemble.refresh_contraction_weights()
            else:
                self.critic1.refresh_contraction_weights()
                self.critic2.refresh_contraction_weights()


@dataclass
class GraphFlashSACTraining:
    """Captured steady-state environment, replay, and learner cadence."""

    trainer: TrainerFlashSAC
    replay: BufferReplayFlashSAC
    env: EnvFlashSAC
    graph: object | None
    interactions_per_launch: int
    updates_per_launch: int
    retained_arrays: tuple[wp.array, ...]
    rollout_graph: object | None = None
    update_graphs: tuple[object, object] | None = None
    prepare_graphs: tuple[object, object] | None = None
    rollout_stream: wp.Stream | None = None
    update_stream: wp.Stream | None = None
    prepare_stream: wp.Stream | None = None
    rollout_actor: object | None = None
    phase_batches: tuple[tuple[BatchSAC, ...], tuple[BatchSAC, ...]] | None = None
    phase: int = 0

    @property
    def overlaps_rollout_and_updates(self) -> bool:
        """Return whether this handle uses the multi-stream leapfrog cadence."""

        return self.update_graphs is not None

    def synchronize(self) -> None:
        """Wait for all cadence streams before checkpointing or reading state."""

        wp.synchronize_device(self.trainer.device)

    def close(self) -> None:
        """Drain cadence streams and destroy graphs before releasing arrays."""

        if self.graph is None and self.rollout_graph is None:
            return
        self.synchronize()
        self.graph = None
        self.rollout_graph = None
        self.update_graphs = None
        self.prepare_graphs = None
        self.rollout_stream = None
        self.update_stream = None
        self.prepare_stream = None
        self.rollout_actor = None
        self.phase_batches = None
        self.retained_arrays = ()

    def __del__(self) -> None:
        self.close()

    def launch(self, *, read_stats: bool = False) -> StatsSACUpdate:
        """Replay one full cadence and update host mirrors without synchronization."""

        if self.update_graphs is None:
            if self.graph is None:
                raise RuntimeError("training graph is closed")
            wp.capture_launch(self.graph)
        else:
            if (
                self.rollout_graph is None
                or self.prepare_graphs is None
                or self.rollout_stream is None
                or self.update_stream is None
                or self.prepare_stream is None
            ):
                raise RuntimeError("overlapped training graph is closed")
            phase = self.phase
            with wp.ScopedStream(self.update_stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(self.prepare_stream)
            with wp.ScopedStream(self.rollout_stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(self.prepare_stream)
            wp.capture_launch(self.update_graphs[phase], stream=self.update_stream)
            wp.capture_launch(self.rollout_graph, stream=self.rollout_stream)
            next_phase = 1 - phase
            with wp.ScopedStream(self.prepare_stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(self.update_stream)
                wp.wait_stream(self.rollout_stream)
            wp.capture_launch(self.prepare_graphs[next_phase], stream=self.prepare_stream)
            self.phase = next_phase
        self.replay.advance_graph_host_state(self.interactions_per_launch)
        self.trainer._gradient_update_count += self.updates_per_launch
        self.trainer._update_count += self.updates_per_launch
        if hasattr(self.env, "sim_time") and hasattr(self.env, "config"):
            self.env.sim_time += self.interactions_per_launch * float(self.env.config.frame_dt)
        if read_stats:
            return self.trainer._read_update_stats()
        return StatsSACUpdate(actor_loss=0.0, critic_loss=0.0, alpha_loss=0.0, alpha=0.0)

    def run(self, launch_count: int, *, stats_interval: int = 0) -> list[StatsSACUpdate]:
        """Replay multiple cadences with diagnostics only at the requested interval."""

        count = int(launch_count)
        interval = int(stats_interval)
        if count < 0 or interval < 0:
            raise ValueError("launch_count and stats_interval must be non-negative")
        stats: list[StatsSACUpdate] = []
        for index in range(count):
            read_stats = interval > 0 and (index + 1) % interval == 0
            result = self.launch(read_stats=read_stats)
            if read_stats:
                stats.append(result)
        return stats


@dataclass
class GraphFlashSACUpdate:
    """Captured one-step FlashSAC learner update bound to fixed batch arrays."""

    trainer: TrainerFlashSAC
    actor_graph: object
    critic_graph: object
    batch: BatchSAC
    policy_frequency: int

    def close(self) -> None:
        """Destroy both CUDA graphs before releasing trainer state."""

        self.actor_graph = None
        self.critic_graph = None

    def __del__(self) -> None:
        self.close()

    def launch(self, *, read_stats: bool = False) -> StatsSACUpdate:
        """Replay one update and advance host-visible checkpoint counters."""

        graph = (
            self.actor_graph if self.trainer._gradient_update_count % self.policy_frequency == 0 else self.critic_graph
        )
        wp.capture_launch(graph)
        self.trainer._gradient_update_count += 1
        self.trainer._update_count += 1
        if read_stats:
            return self.trainer._read_update_stats()
        return StatsSACUpdate(actor_loss=0.0, critic_loss=0.0, alpha_loss=0.0, alpha=0.0)


def train_flash_sac(
    env: EnvFlashSAC,
    trainer: TrainerFlashSAC,
    *,
    interaction_steps: int,
    updates_per_step: int = 1,
    stats_interval: int = 1,
    seed: int = 0,
    reset_at_start: bool = True,
) -> list[StatsSACUpdate]:
    """Collect vectorized transitions and perform replay-driven FlashSAC updates.

    Args:
        env: Vectorized PhoenX environment.
        trainer: FlashSAC trainer matching the environment dimensions.
        interaction_steps: Number of environment steps to collect.
        updates_per_step: Maximum replay updates after each environment step.
        stats_interval: Number of updates between synchronized diagnostic reads.
            ``0`` disables diagnostic reads.
        seed: Base stochastic action and update seed.
        reset_at_start: Whether to reset all environments before collection.

    Returns:
        Update statistics produced after replay warmup.
    """

    if interaction_steps <= 0:
        raise ValueError("interaction_steps must be positive")
    if updates_per_step < 0:
        raise ValueError("updates_per_step must be non-negative")
    if stats_interval < 0:
        raise ValueError("stats_interval must be non-negative")
    policy_action_dim = int(getattr(env, "policy_action_dim", env.action_dim))
    if trainer.obs_dim != env.obs_dim or trainer.action_dim != policy_action_dim:
        raise ValueError("FlashSAC trainer dimensions do not match environment policy interface")

    obs = env.reset() if reset_at_start else env.observe()
    stats: list[StatsSACUpdate] = []
    zero_truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    for step in range(int(interaction_steps)):
        actions, _log_probs = trainer.act(obs, seed=int(seed) + step)
        next_obs, rewards, dones = env.step(actions)
        replay_next_obs = getattr(env, "step_next_obs", next_obs)
        truncateds = getattr(env, "step_truncateds", zero_truncateds)
        terminateds = getattr(env, "step_terminateds", dones)
        trainer.process_transition(
            obs,
            actions,
            rewards,
            terminateds,
            replay_next_obs,
            truncateds=truncateds,
        )
        for update_index in range(int(updates_per_step)):
            if not trainer.can_start_training():
                break
            read_stats = stats_interval > 0 and (trainer._update_count + 1) % stats_interval == 0
            update_stats = trainer.update(
                seed=int(seed) + step * max(updates_per_step, 1) + update_index,
                read_stats=read_stats,
            )
            if read_stats:
                stats.append(update_stats)
        obs = next_obs
    return stats


def _config_from_flash_sac_checkpoint(data: np.lib.npyio.NpzFile) -> ConfigFlashSAC:
    """Restore scalar FlashSAC configuration fields from an archive."""

    kwargs: dict[str, object] = {}
    for config_field in fields(ConfigFlashSAC):
        none_key = f"config_{config_field.name}_is_none"
        if none_key in data and bool(data[none_key]):
            kwargs[config_field.name] = None
        elif f"config_{config_field.name}" in data:
            kwargs[config_field.name] = data[f"config_{config_field.name}"].item()
    return ConfigFlashSAC(**kwargs)


def _copy_optimizer_state(destination: Adam, source: Adam) -> None:
    """Copy Adam state into compatible setup-owned arrays."""

    if type(destination) is not type(source) or len(destination.params) != len(source.params):
        raise ValueError("FlashSAC optimizer structures do not match")
    for dst_param, src_param, dst_m, src_m, dst_v, src_v in zip(
        destination.params, source.params, destination.m, source.m, destination.v, source.v, strict=True
    ):
        if dst_param.shape != src_param.shape or dst_param.dtype != src_param.dtype:
            raise ValueError("FlashSAC optimizer parameter structures do not match")
        wp.copy(dst_m, src_m)
        wp.copy(dst_v, src_v)
    for dst, src in (
        (destination._step_count, source._step_count),
        (destination._step_corrections, source._step_corrections),
        (destination._grad_sumsq, source._grad_sumsq),
        (destination.lr_scale, source.lr_scale),
        (destination.step_condition, source.step_condition),
    ):
        wp.copy(dst, src)
    source_pbt_scale = float(source.pbt_lr_scale.numpy()[0])
    destination.set_pbt_lr(source.lr * source_pbt_scale)
    destination._step_count_host = source._step_count_host


def _pack_flash_sac_network(
    data: dict[str, np.ndarray],
    prefix: str,
    network: object,
) -> None:
    if not isinstance(network, NetworkFlashSAC):
        _pack_policy_network(data, prefix, network)
        return
    arrays = network.state_arrays()
    data[f"{prefix}_network_type"] = np.asarray("flash_sac_reference")
    data[f"{prefix}_state_count"] = np.asarray(len(arrays), dtype=np.int64)
    for index, array in enumerate(arrays):
        data[f"{prefix}_state_{index}"] = array.numpy()


def _unpack_flash_sac_network(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    network: object,
) -> None:
    if not isinstance(network, NetworkFlashSAC):
        _unpack_policy_network(data, prefix, network)
        return
    if str(data[f"{prefix}_network_type"].item()) != "flash_sac_reference":
        raise ValueError(f"Checkpoint {prefix} network type does not match FlashSAC reference backbone")
    arrays = network.state_arrays()
    if int(data[f"{prefix}_state_count"]) != len(arrays):
        raise ValueError(f"Checkpoint {prefix} reference state count does not match trainer")
    for index, array in enumerate(arrays):
        array.assign(data[f"{prefix}_state_{index}"])


def save_flash_sac_checkpoint(trainer: TrainerFlashSAC, path: str | Path) -> None:
    """Save a FlashSAC trainer checkpoint without replay contents."""

    trainer.save_checkpoint(path)


def load_flash_sac_checkpoint(
    path: str | Path,
    *,
    config: ConfigFlashSAC | None = None,
    device: wp.context.Devicelike = None,
) -> TrainerFlashSAC:
    """Load a FlashSAC trainer checkpoint without replay contents."""

    return TrainerFlashSAC.load_checkpoint(
        path,
        config=config,
        device=device,
    )
