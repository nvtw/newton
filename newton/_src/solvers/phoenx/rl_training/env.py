# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

import warp as wp

from .kernels import seed_counter_increment_kernel
from .ppo import BufferRollout, TrainerPPO


class EnvPPO(Protocol):
    """Minimal vectorized environment interface needed by PPO rollout collection."""

    world_count: int
    obs_dim: int
    action_dim: int
    device: wp.context.Device
    obs: wp.array
    step_successes: wp.array

    def observe(self) -> wp.array:
        """Return the current batched observation array."""

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Advance the environment with batched actions."""


@wp.kernel
def rollout_store_pre_step_kernel(
    step: wp.int32,
    num_envs: wp.int32,
    obs_dim: wp.int32,
    action_dim: wp.int32,
    obs_src: wp.array2d[wp.float32],
    actions_src: wp.array2d[wp.float32],
    log_probs_src: wp.array[wp.float32],
    values_src: wp.array2d[wp.float32],
    value_col: wp.int32,
    obs_dst: wp.array2d[wp.float32],
    actions_dst: wp.array2d[wp.float32],
    log_probs_dst: wp.array[wp.float32],
    values_dst: wp.array[wp.float32],
):
    env, col = wp.tid()
    row = step * num_envs + env
    if col < obs_dim:
        obs_dst[row, col] = obs_src[env, col]
    if col < action_dim:
        actions_dst[row, col] = actions_src[env, col]
    if col == 0:
        log_probs_dst[row] = log_probs_src[env]
        values_dst[row] = values_src[env, value_col]


@wp.kernel
def rollout_store_post_step_kernel(
    step: wp.int32,
    num_envs: wp.int32,
    rewards_src: wp.array[wp.float32],
    dones_src: wp.array[wp.float32],
    successes_src: wp.array[wp.float32],
    rewards_dst: wp.array[wp.float32],
    dones_dst: wp.array[wp.float32],
    successes_dst: wp.array[wp.float32],
):
    env = wp.tid()
    row = step * num_envs + env
    rewards_dst[row] = rewards_src[env]
    dones_dst[row] = dones_src[env]
    successes_dst[row] = successes_src[env]


@wp.kernel
def rollout_store_bootstrap_values_kernel(
    values_src: wp.array2d[wp.float32],
    value_col: wp.int32,
    num_steps: wp.int32,
    num_envs: wp.int32,
    values_dst: wp.array[wp.float32],
):
    env = wp.tid()
    values_dst[num_steps * num_envs + env] = values_src[env, value_col]


def capture_env_steps(
    env: EnvPPO,
    actions: wp.array,
    *,
    steps_per_graph: int = 1,
    warmup_steps: int = 1,
):
    """Capture repeated environment steps into one CUDA graph.

    Args:
        env: Vectorized environment implementing :class:`EnvPPO`.
        actions: Persistent action array read by each captured step.
        steps_per_graph: Number of policy steps recorded in the graph.
        warmup_steps: Eager policy steps run before capture to compile kernels
            and allocate lazy scratch buffers.

    Returns:
        Captured CUDA graph ready for :func:`wp.capture_launch`.
    """

    steps = int(steps_per_graph)
    warmup = int(warmup_steps)
    if steps <= 0:
        raise ValueError("steps_per_graph must be positive")
    if warmup < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not env.device.is_cuda or not wp.is_mempool_enabled(env.device):
        raise RuntimeError("environment graph capture requires a CUDA device with Warp mempool enabled")

    for _ in range(warmup):
        env.step(actions)
    with wp.ScopedCapture(device=env.device) as capture:
        for _ in range(steps):
            env.step(actions)
    return capture.graph


def make_seed_counter(seed: int, *, device: wp.context.Devicelike = None) -> wp.array[wp.int32]:
    """Allocate a one-element device seed counter for graph replay."""

    return wp.array([int(seed) & 0x7FFFFFFF], dtype=wp.int32, device=device)


def advance_seed_counter(
    seed_counter: wp.array[wp.int32], delta: int = 1, *, device: wp.context.Devicelike = None
) -> None:
    """Advance a device seed counter in a graph-capturable kernel."""

    counter_device = wp.get_device(device) if device is not None else seed_counter.device
    wp.launch(seed_counter_increment_kernel, dim=1, inputs=[seed_counter, int(delta)], device=counter_device)


def collect_ppo_rollout(env: EnvPPO, trainer: TrainerPPO, buffer: BufferRollout, *, seed: int) -> None:
    """Collect one PPO rollout from any vectorized Warp environment.

    Args:
        env: Vectorized environment implementing :class:`EnvPPO`.
        trainer: PPO trainer used for action sampling and value estimates.
        buffer: Rollout buffer to fill.
        seed: Base stochastic action seed.
    """

    _collect_ppo_rollout_impl(env, trainer, buffer, seed=int(seed), seed_counter=None)


def collect_ppo_rollout_seed_counter(
    env: EnvPPO, trainer: TrainerPPO, buffer: BufferRollout, *, seed_counter: wp.array[wp.int32]
) -> None:
    """Collect one PPO rollout using a graph-replay-safe device seed counter."""

    _collect_ppo_rollout_impl(env, trainer, buffer, seed=0, seed_counter=seed_counter)
    advance_seed_counter(seed_counter, buffer.num_steps, device=env.device)


def _collect_ppo_rollout_impl(
    env: EnvPPO,
    trainer: TrainerPPO,
    buffer: BufferRollout,
    *,
    seed: int,
    seed_counter: wp.array[wp.int32] | None,
) -> None:
    if buffer.num_envs != env.world_count or buffer.obs_dim != env.obs_dim or buffer.action_dim != env.action_dim:
        raise ValueError("PPO buffer dimensions do not match environment")

    trainer.reset_rollout_state()
    obs = env.observe()
    max_cols = max(env.obs_dim, env.action_dim, 1)
    value_col = trainer.value_column
    for step in range(buffer.num_steps):
        if seed_counter is None:
            actions, log_probs, values = trainer.act_reuse(obs, seed=int(seed) + step)
        else:
            actions, log_probs, values = trainer.act_reuse_seed_counter(
                obs, seed_counter=seed_counter, seed_offset=step
            )
        wp.launch(
            rollout_store_pre_step_kernel,
            dim=(env.world_count, max_cols),
            inputs=[
                step,
                env.world_count,
                env.obs_dim,
                env.action_dim,
                obs,
                actions,
                log_probs,
                values,
                value_col,
            ],
            outputs=[buffer.obs, buffer.actions, buffer.old_log_probs, buffer.values],
            device=env.device,
        )
        next_obs, rewards, dones = env.step(actions)
        wp.launch(
            rollout_store_post_step_kernel,
            dim=env.world_count,
            inputs=[step, env.world_count, rewards, dones, env.step_successes],
            outputs=[buffer.rewards, buffer.dones, buffer.successes],
            device=env.device,
        )
        trainer.reset_rollout_state(dones)
        obs = next_obs

    final_values = trainer.value_reuse(obs)
    wp.launch(
        rollout_store_bootstrap_values_kernel,
        dim=env.world_count,
        inputs=[final_values, value_col, buffer.num_steps, env.world_count],
        outputs=[buffer.values],
        device=env.device,
    )
    buffer.compute_returns(
        gamma=trainer.config.gamma,
        gae_lambda=trainer.config.gae_lambda,
        reward_clip=trainer.config.reward_clip,
    )
