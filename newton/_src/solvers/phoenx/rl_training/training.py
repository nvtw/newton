# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

from .anymal import ConfigEnvAnymalPhoenX, EnvAnymalPhoenX, anymal_mirror_map_ppo
from .env import collect_ppo_rollout_seed_counter, make_seed_counter
from .g1 import ConfigEnvG1PhoenX, EnvG1PhoenX, g1_mirror_map_ppo
from .g1_recipe import (
    ACTIVATION,
    COMMAND_CURRICULUM_SAMPLES,
    COMMAND_CURRICULUM_START,
    COMMAND_RESAMPLE_STEPS,
    COMMAND_SAMPLING,
    COMMAND_X_RANGE,
    COMMAND_Y_RANGE,
    COMMAND_YAW_RANGE,
    COMMAND_ZERO_PROBABILITY,
    HIDDEN_LAYERS,
    LOG_STD_INIT,
    RANDOMIZE_COMMANDS,
    RESET_RECURRENT_STATE_ON_ROLLOUT_START,
    ROLLOUT_STEPS,
    SEED,
    SPARSE_TARGET_ANGLE_MAX,
    SPARSE_TARGET_ANGLE_MIN,
    SPARSE_TARGET_CURRICULUM_END,
    SPARSE_TARGET_CURRICULUM_SAMPLES,
    SPARSE_TARGET_CURRICULUM_START,
    SPARSE_TARGET_RANDOMIZE,
    SQUASH_ACTIONS,
    TRAIN_ITERATIONS,
    default_g1_ppo_config,
)
from .ppo import BufferRollout, ConfigPPO, StatsPPOUpdate, TrainerPPO, load_ppo_checkpoint

_G1_TRAIN_STAT_COUNT = 15
_ANYMAL_TRAIN_STAT_COUNT = 9


@wp.kernel
def _g1_train_stat_sums_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
    actions: wp.array2d[wp.float32],
    log_std: wp.array[wp.float32],
    command: wp.array2d[wp.float32],
    num_samples: wp.int32,
    num_envs: wp.int32,
    action_dim: wp.int32,
    policy_loss: wp.array[wp.float32],
    value_loss: wp.array[wp.float32],
    approx_kl: wp.array[wp.float32],
    clip_fraction: wp.array[wp.float32],
    stats: wp.array[wp.float32],
):
    i = wp.tid()
    if i < num_samples:
        wp.atomic_add(stats, 0, rewards[i])
        wp.atomic_add(stats, 1, dones[i])
        wp.atomic_add(stats, 2, successes[i])
    action_total = num_samples * action_dim
    if i < action_total:
        row = i // action_dim
        action = i - row * action_dim
        a = actions[row, action]
        abs_a = wp.abs(a)
        clipped = wp.min(abs_a, wp.float32(1.0))
        wp.atomic_add(stats, 10, a * a)
        wp.atomic_add(stats, 11, clipped * clipped)
        clip_hit = wp.float32(0.0)
        if abs_a > wp.float32(1.0):
            clip_hit = wp.float32(1.0)
        wp.atomic_add(stats, 12, clip_hit)
        wp.atomic_add(stats, 13, clipped)
    if i < action_dim:
        wp.atomic_add(stats, 14, log_std[i])
    if i < num_envs:
        wp.atomic_add(stats, 3, command[i, 0])
        wp.atomic_add(stats, 4, command[i, 1])
        wp.atomic_add(stats, 5, command[i, 2])
    if i == 0:
        stats[6] = policy_loss[0]
        stats[7] = value_loss[0]
        stats[8] = approx_kl[0]
        stats[9] = clip_fraction[0]


class _G1TrainDiagnosticsReadback:
    """Preallocated compact host readback for monitored G1 PPO training."""

    def __init__(self, device: wp.context.Devicelike):
        self.device = wp.get_device(device)
        self._stats = wp.zeros(_G1_TRAIN_STAT_COUNT, dtype=wp.float32, device=self.device)
        self._stats_host = wp.empty(_G1_TRAIN_STAT_COUNT, dtype=wp.float32, device="cpu", pinned=self.device.is_cuda)

    def read(
        self, buffer: BufferRollout, env: EnvG1PhoenX, trainer: TrainerPPO
    ) -> tuple[
        tuple[float, float, float, float, float, float], StatsPPOUpdate, tuple[float, float, float, float, float]
    ]:
        self._stats.zero_()
        wp.launch(
            _g1_train_stat_sums_kernel,
            dim=max(buffer.num_samples * env.action_dim, env.world_count, env.action_dim),
            inputs=[
                buffer.rewards,
                buffer.dones,
                buffer.successes,
                buffer.actions,
                trainer.actor.log_std,
                env.command,
                buffer.num_samples,
                env.world_count,
                env.action_dim,
                trainer._policy_loss,
                trainer._value_loss,
                trainer._approx_kl,
                trainer._clip_fraction,
            ],
            outputs=[self._stats],
            device=self.device,
        )
        wp.copy(self._stats_host, self._stats, count=_G1_TRAIN_STAT_COUNT)
        if self.device.is_cuda:
            # Pinned host arrays expose memory directly; wait for the async graph copy.
            wp.synchronize_device(self.device)
        stats = self._stats_host.numpy()
        inv_samples = 1.0 / float(buffer.num_samples)
        inv_envs = 1.0 / float(env.world_count)
        inv_actions = 1.0 / float(buffer.num_samples * env.action_dim)
        inv_action_dim = 1.0 / float(env.action_dim)
        rollout_metrics = (
            float(stats[0]) * inv_samples,
            float(stats[1]) * inv_samples,
            float(stats[2]) * inv_samples,
            float(stats[3]) * inv_envs,
            float(stats[4]) * inv_envs,
            float(stats[5]) * inv_envs,
        )
        update_stats = StatsPPOUpdate(
            policy_loss=float(stats[6]),
            value_loss=float(stats[7]),
            approx_kl=float(stats[8]),
            clip_fraction=float(stats[9]),
        )
        action_metrics = (
            float(np.sqrt(max(float(stats[10]) * inv_actions, 0.0))),
            float(np.sqrt(max(float(stats[11]) * inv_actions, 0.0))),
            float(stats[12]) * inv_actions,
            float(stats[13]) * inv_actions,
            float(stats[14]) * inv_action_dim,
        )
        return rollout_metrics, update_stats, action_metrics


def _g1_disabled_rollout_metrics(env: EnvG1PhoenX) -> tuple[float, float, float, float, float, float]:
    command = env.config.command
    return (0.0, 0.0, 0.0, float(command[0]), float(command[1]), float(command[2]))


@wp.kernel
def _anymal_train_stat_partials_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
    obs: wp.array2d[wp.float32],
    num_samples: wp.int32,
    command_x: wp.float32,
    use_observed_command: wp.int32,
    partials: wp.array2d[wp.float32],
):
    tile, lane = wp.tid()
    i = tile * wp.int32(32) + lane
    reward_sum = wp.float32(0.0)
    done_sum = wp.float32(0.0)
    success_sum = wp.float32(0.0)
    vx_sum = wp.float32(0.0)
    vx_error_sum = wp.float32(0.0)
    if i < num_samples:
        vx = obs[i, 0]
        target_vx = command_x
        if use_observed_command != wp.int32(0):
            target_vx = obs[i, 9]
        reward_sum = rewards[i]
        done_sum = dones[i]
        success_sum = successes[i]
        vx_sum = vx
        vx_error_sum = wp.abs(vx - target_vx)

    reward_tile = wp.tile_sum(wp.tile(reward_sum))
    done_tile = wp.tile_sum(wp.tile(done_sum))
    success_tile = wp.tile_sum(wp.tile(success_sum))
    vx_tile = wp.tile_sum(wp.tile(vx_sum))
    vx_error_tile = wp.tile_sum(wp.tile(vx_error_sum))
    if lane == wp.int32(0):
        partials[tile, 0] = reward_tile[0]
        partials[tile, 1] = done_tile[0]
        partials[tile, 2] = success_tile[0]
        partials[tile, 3] = vx_tile[0]
        partials[tile, 4] = vx_error_tile[0]


@wp.kernel
def _anymal_train_stat_finalize_kernel(
    partials: wp.array2d[wp.float32],
    partial_count: wp.int32,
    policy_loss: wp.array[wp.float32],
    value_loss: wp.array[wp.float32],
    approx_kl: wp.array[wp.float32],
    clip_fraction: wp.array[wp.float32],
    stats: wp.array[wp.float32],
):
    lane = wp.tid()
    reward_sum = wp.float32(0.0)
    done_sum = wp.float32(0.0)
    success_sum = wp.float32(0.0)
    vx_sum = wp.float32(0.0)
    vx_error_sum = wp.float32(0.0)
    p = lane
    while p < partial_count:
        reward_sum = reward_sum + partials[p, 0]
        done_sum = done_sum + partials[p, 1]
        success_sum = success_sum + partials[p, 2]
        vx_sum = vx_sum + partials[p, 3]
        vx_error_sum = vx_error_sum + partials[p, 4]
        p = p + wp.int32(32)

    reward_tile = wp.tile_sum(wp.tile(reward_sum))
    done_tile = wp.tile_sum(wp.tile(done_sum))
    success_tile = wp.tile_sum(wp.tile(success_sum))
    vx_tile = wp.tile_sum(wp.tile(vx_sum))
    vx_error_tile = wp.tile_sum(wp.tile(vx_error_sum))
    if lane == wp.int32(0):
        stats[0] = reward_tile[0]
        stats[1] = done_tile[0]
        stats[2] = success_tile[0]
        stats[3] = vx_tile[0]
        stats[4] = vx_error_tile[0]
        stats[5] = policy_loss[0]
        stats[6] = value_loss[0]
        stats[7] = approx_kl[0]
        stats[8] = clip_fraction[0]


class _AnymalTrainDiagnosticsReadback:
    """Preallocated compact host readback for monitored Anymal PPO training."""

    def __init__(self, device: wp.context.Devicelike, max_samples: int):
        self.device = wp.get_device(device)
        self._partial_count = max(1, (int(max_samples) + 31) // 32)
        self._partials = wp.empty((self._partial_count, 5), dtype=wp.float32, device=self.device)
        self._stats = wp.empty(_ANYMAL_TRAIN_STAT_COUNT, dtype=wp.float32, device=self.device)
        self._stats_host = wp.empty(
            _ANYMAL_TRAIN_STAT_COUNT, dtype=wp.float32, device="cpu", pinned=self.device.is_cuda
        )

    def read(
        self,
        buffer: BufferRollout,
        trainer: TrainerPPO,
        command_x: float,
        target_distance: float,
        *,
        use_observed_command: bool,
    ) -> tuple[tuple[float, float, float, float, float, float], StatsPPOUpdate]:
        partial_count = max(1, (int(buffer.num_samples) + 31) // 32)
        if partial_count > self._partial_count:
            raise RuntimeError("Anymal diagnostics partial buffer is too small")
        wp.launch(
            _anymal_train_stat_partials_kernel,
            dim=(partial_count, 32),
            inputs=[
                buffer.rewards,
                buffer.dones,
                buffer.successes,
                buffer.obs,
                buffer.num_samples,
                float(command_x),
                int(bool(use_observed_command)),
            ],
            outputs=[self._partials],
            device=self.device,
        )
        wp.launch(
            _anymal_train_stat_finalize_kernel,
            dim=32,
            inputs=[
                self._partials,
                partial_count,
                trainer._policy_loss,
                trainer._value_loss,
                trainer._approx_kl,
                trainer._clip_fraction,
            ],
            outputs=[self._stats],
            device=self.device,
        )
        wp.copy(self._stats_host, self._stats, count=_ANYMAL_TRAIN_STAT_COUNT)
        if self.device.is_cuda:
            wp.synchronize_device(self.device)
        stats = self._stats_host.numpy()
        inv_samples = 1.0 / float(buffer.num_samples)
        rollout_metrics = (
            float(stats[0]) * inv_samples,
            float(stats[1]) * inv_samples,
            float(stats[2]) * inv_samples,
            float(target_distance),
            float(stats[3]) * inv_samples,
            float(stats[4]) * inv_samples,
        )
        update_stats = StatsPPOUpdate(
            policy_loss=float(stats[5]),
            value_loss=float(stats[6]),
            approx_kl=float(stats[7]),
            clip_fraction=float(stats[8]),
        )
        return rollout_metrics, update_stats


@dataclass
class ConfigTrainAnymalPPO:
    """Configuration for :func:`train_anymal_ppo`.

    Args:
        iterations: Number of collect-update iterations.
        rollout_steps: Policy steps per environment and iteration.
        hidden_layers: Actor and critic hidden-layer widths.
        activation: Actor and critic hidden-layer activation.
        log_std_init: Initial actor log-standard-deviation.
        env_config: PhoenX Anymal environment configuration.
        ppo_config: PPO optimizer configuration.
        device: Warp device.
        seed: Base RNG seed.
        log_interval: Print every ``log_interval`` iterations when positive.
        use_target_curriculum: Increase sparse target distance after enough successes.
        target_distance_start: Initial sparse target distance [m].
        target_distance_end: Maximum sparse target distance [m].
        target_distance_step: Curriculum distance increment [m].
        target_success_threshold: Success rate required to advance the curriculum.
        randomize_target_positions: Sample target directions per world before each rollout.
        target_angle_min: Minimum sampled target angle relative to initial body-forward [rad].
        target_angle_max: Maximum sampled target angle relative to initial body-forward [rad].
        randomize_commands: Sample dense velocity commands per world before each rollout.
        command_x_range: Sampled forward command range [m/s].
        command_y_range: Sampled lateral command range [m/s].
        command_yaw_range: Sampled yaw-rate command range [rad/s].
        command_height_range: Sampled base-height offset range [m].
        command_yaw_min_abs: Minimum non-zero sampled yaw-rate magnitude [rad/s].
        command_zero_probability: Probability of replacing a sampled command with zero velocity and height offset.
        resume_checkpoint: Optional PPO checkpoint to resume from.
        checkpoint_path: Optional path for writing PPO checkpoints.
        checkpoint_interval: Save a checkpoint every N iterations when positive.
        execution_mode: ``"eager"`` for serial collect-update or
            ``"graph_leapfrog"`` to overlap rollout and previous update graphs
            on separate CUDA streams.
    """

    iterations: int = 100
    rollout_steps: int = 64
    hidden_layers: tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    log_std_init: float = -0.3
    env_config: ConfigEnvAnymalPhoenX | None = None
    ppo_config: ConfigPPO | None = None
    device: wp.context.Devicelike = None
    seed: int = 0
    log_interval: int = 1
    use_target_curriculum: bool = True
    target_distance_start: float = 0.45
    target_distance_end: float = 1.0
    target_distance_step: float = 0.1
    target_success_threshold: float = 0.025
    randomize_target_positions: bool = True
    target_angle_min: float = -float(np.pi)
    target_angle_max: float = float(np.pi)
    randomize_commands: bool = False
    command_x_range: tuple[float, float] = (0.0, 0.0)
    command_y_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_range: tuple[float, float] = (0.0, 0.0)
    command_height_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_min_abs: float = 0.0
    command_zero_probability: float = 0.0
    resume_checkpoint: str | None = None
    checkpoint_path: str | None = None
    checkpoint_interval: int = 0
    execution_mode: str = "eager"


@dataclass
class StatsTrainAnymalPPO:
    """Per-iteration Anymal PPO training diagnostics."""

    iteration: int
    mean_reward: float
    mean_done: float
    mean_success: float
    target_distance: float
    mean_forward_velocity: float
    mean_abs_forward_velocity_error: float
    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float


@dataclass
class ResultTrainAnymalPPO:
    """Result returned by train_anymal_ppo."""

    trainer: TrainerPPO
    env: EnvAnymalPhoenX
    buffer: BufferRollout
    history: list[StatsTrainAnymalPPO]


@dataclass
class ConfigTrainG1PPO:
    """Configuration for train_g1_ppo.

    Sparse-target training keeps the reward sparse; the optional target
    curriculum only changes task distance on the device between rollouts.
    """

    iterations: int = TRAIN_ITERATIONS
    rollout_steps: int = ROLLOUT_STEPS
    hidden_layers: tuple[int, ...] = HIDDEN_LAYERS
    activation: str = ACTIVATION
    log_std_init: float = LOG_STD_INIT
    squash_actions: bool = SQUASH_ACTIONS
    env_config: ConfigEnvG1PhoenX | None = None
    ppo_config: ConfigPPO | None = None
    device: wp.context.Devicelike = None
    seed: int = SEED
    log_interval: int = 1
    randomize_commands: bool = RANDOMIZE_COMMANDS
    command_sampling: str = COMMAND_SAMPLING
    command_x_range: tuple[float, float] = COMMAND_X_RANGE
    command_y_range: tuple[float, float] = COMMAND_Y_RANGE
    command_yaw_range: tuple[float, float] = COMMAND_YAW_RANGE
    command_zero_probability: float = COMMAND_ZERO_PROBABILITY
    command_resample_steps: int = COMMAND_RESAMPLE_STEPS
    command_curriculum_start: float = COMMAND_CURRICULUM_START
    command_curriculum_samples: int = COMMAND_CURRICULUM_SAMPLES
    use_target_curriculum: bool = True
    target_distance_start: float = SPARSE_TARGET_CURRICULUM_START
    target_distance_end: float = SPARSE_TARGET_CURRICULUM_END
    target_curriculum_samples: int = SPARSE_TARGET_CURRICULUM_SAMPLES
    target_curriculum_start_samples: int | None = None
    randomize_target_positions: bool = SPARSE_TARGET_RANDOMIZE
    randomize_target_distance_range: bool = False
    target_angle_min: float = SPARSE_TARGET_ANGLE_MIN
    target_angle_max: float = SPARSE_TARGET_ANGLE_MAX
    reset_recurrent_state_on_rollout_start: bool = RESET_RECURRENT_STATE_ON_ROLLOUT_START
    resume_checkpoint: str | None = None
    checkpoint_path: str | None = None
    checkpoint_interval: int = 0
    readback_diagnostics: bool = True
    execution_mode: str = "eager"


@dataclass
class StatsTrainG1PPO:
    """Per-iteration G1 PPO training diagnostics."""

    iteration: int
    mean_reward: float
    mean_done: float
    mean_tracking_perf: float
    mean_command_x: float
    mean_command_y: float
    mean_command_yaw: float
    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float
    raw_action_rms: float
    clipped_action_rms: float
    action_clip_fraction: float
    clipped_action_abs_mean: float
    mean_log_std: float
    rollout_seconds: float
    update_seconds: float
    samples_per_second: float


@dataclass
class ResultTrainG1PPO:
    """Result returned by train_g1_ppo."""

    trainer: TrainerPPO
    env: EnvG1PhoenX
    buffer: BufferRollout
    history: list[StatsTrainG1PPO]


@dataclass
class ConfigEvaluateG1PPO:
    """Configuration for :func:`evaluate_g1_ppo`."""

    env_config: ConfigEnvG1PhoenX | None = None
    steps: int = 200
    device: wp.context.Devicelike = None
    deterministic: bool = True
    seed: int = 1000


@dataclass
class StatsEvaluateG1PPO:
    """Deterministic G1 rollout diagnostics for a trained PPO policy."""

    steps: int
    mean_reward: float
    mean_done: float
    mean_tracking_perf: float
    mean_command_x: float
    mean_command_y: float
    mean_command_yaw: float
    fall_fraction: float
    mean_survival_steps: float
    mean_command_aligned_displacement: float
    mean_command_aligned_velocity: float
    mean_lateral_displacement_abs: float
    mean_path_length: float
    samples_per_second: float


@dataclass
class ResultEvaluateG1PPO:
    """Result returned by :func:`evaluate_g1_ppo`."""

    stats: StatsEvaluateG1PPO


@dataclass
class ConfigEvaluateG1TargetPPO:
    """Configuration for :func:`evaluate_g1_target_ppo`.

    Args:
        env_config: Base PhoenX G1 environment configuration.
        target_positions: Target XY positions to evaluate [m].
        steps: Maximum policy steps per target.
        device: Warp device. Uses the trainer device when ``None``.
        deterministic: Use deterministic policy means.
        seed: Base action seed for stochastic evaluation.
        max_tilt_degrees: Maximum allowed base tilt from upright [deg].
        min_valid_base_height: Minimum allowed base height for strict walking [m].
        max_valid_base_height: Maximum allowed base height for strict walking [m].
    """

    env_config: ConfigEnvG1PhoenX | None = None
    target_positions: tuple[tuple[float, float], ...] = ((0.6, 0.0), (1.0, 0.0), (1.4, 0.0))
    steps: int = 300
    device: wp.context.Devicelike = None
    deterministic: bool = True
    seed: int = 1000
    max_tilt_degrees: float = 30.0
    min_valid_base_height: float = 0.35
    max_valid_base_height: float = 1.10


@dataclass
class StatsEvaluateG1TargetPPO:
    """Target-following diagnostics for one evaluated G1 target."""

    target_position: tuple[float, float]
    success_fraction: float
    strict_success_fraction: float
    fall_fraction: float
    tilt_violation_fraction: float
    height_violation_fraction: float
    mean_first_success_step: float
    mean_initial_distance: float
    mean_final_distance: float
    mean_min_distance: float
    mean_target_aligned_displacement: float
    mean_path_length: float
    mean_speed: float
    mean_forward_velocity: float
    max_tilt_degrees: float
    mean_max_tilt_degrees: float
    min_base_height: float
    max_base_height: float
    mean_base_height: float


@dataclass
class ResultEvaluateG1TargetPPO:
    """Result returned by :func:`evaluate_g1_target_ppo`."""

    stats: list[StatsEvaluateG1TargetPPO]


@dataclass
class ConfigEvaluateG1GatePPO:
    """Configuration for :func:`evaluate_g1_gate_ppo`."""

    env_config: ConfigEnvG1PhoenX | None = None
    battery_commands: tuple[tuple[float, float, float], ...] = (
        (0.8, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.3, 0.0, 1.0),
        (0.3, 0.0, -1.0),
        (0.0, 0.4, 0.0),
        (0.0, 0.0, 0.0),
    )
    seeds_per_command: int = 4
    battery_steps: int = 1000
    diagnostic_command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    diagnostic_world_count: int = 1
    diagnostic_steps: int = 2000
    device: wp.context.Devicelike = None
    deterministic: bool = True
    seed: int = 1000
    max_battery_falls: int = 1
    min_battery_perf: float = 0.90
    max_action_jerk_rms: float = 0.21
    max_ang_vel_xy_rms: float = 0.21
    max_yaw_rate_rms: float = 0.20
    max_leg_qvel_rms: float = 1.22


@dataclass
class StatsEvaluateG1GateCommandPPO:
    """Per-command statistics for the G1 quality gate."""

    command: tuple[float, float, float]
    falls: int
    mean_tracking_perf: float
    mean_linear_velocity_error: float
    mean_yaw_rate_error: float
    samples: int


@dataclass
class StatsEvaluateG1GatePPO:
    """nanoG1-style quality-gate diagnostics for a G1 PPO policy."""

    battery_falls: int
    battery_perf: float
    action_jerk_rms: float
    ang_vel_xy_rms: float
    yaw_rate_rms: float
    leg_qvel_rms: float
    diagnostic_falls: int
    battery_samples: int
    diagnostic_samples: int
    samples_per_second: float
    pass_gate: bool
    per_command: tuple[StatsEvaluateG1GateCommandPPO, ...]


@dataclass
class ResultEvaluateG1GatePPO:
    """Result returned by :func:`evaluate_g1_gate_ppo`."""

    stats: StatsEvaluateG1GatePPO


@dataclass
class ConfigEvaluateAnymalPPO:
    """Configuration for :func:`evaluate_anymal_ppo`.

    Args:
        env_config: Base PhoenX Anymal environment configuration.
        target_positions: Target XY positions to evaluate [m].
        steps: Maximum policy steps per target.
        device: Warp device. Uses the trainer device when ``None``.
        deterministic: Use deterministic policy means.
        seed: Base action seed for stochastic evaluation.
        max_tilt_degrees: Maximum allowed base tilt from upright [deg].
        min_valid_base_height: Minimum allowed base height for strict walking [m].
        max_valid_base_height: Maximum allowed base height for strict walking [m].
    """

    env_config: ConfigEnvAnymalPhoenX | None = None
    target_positions: tuple[tuple[float, float], ...] = (
        (0.0, 0.75),
        (0.75, 0.0),
        (-0.75, 0.0),
        (0.0, -0.75),
        (0.53, 0.53),
    )
    steps: int = 120
    device: wp.context.Devicelike = None
    deterministic: bool = True
    seed: int = 1000
    max_tilt_degrees: float = 20.0
    min_valid_base_height: float = 0.40
    max_valid_base_height: float = 0.85


@dataclass
class StatsEvaluateAnymalTargetPPO:
    """Target-following diagnostics for one evaluated target."""

    target_position: tuple[float, float]
    success_fraction: float
    strict_success_fraction: float
    fall_fraction: float
    tilt_violation_fraction: float
    height_violation_fraction: float
    mean_first_success_step: float
    mean_initial_distance: float
    mean_final_distance: float
    mean_min_distance: float
    mean_target_aligned_displacement: float
    mean_path_length: float
    mean_speed: float
    mean_forward_velocity: float
    max_tilt_degrees: float
    mean_max_tilt_degrees: float
    min_base_height: float
    max_base_height: float
    mean_base_height: float


@dataclass
class ResultEvaluateAnymalPPO:
    """Result returned by :func:`evaluate_anymal_ppo`."""

    stats: list[StatsEvaluateAnymalTargetPPO]


def _default_ppo_config() -> ConfigPPO:
    return ConfigPPO(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=2.0e-3,
        actor_lr=7.0e-4,
        critic_lr=1.0e-3,
        train_epochs=5,
        normalize_advantages=True,
    )


def _default_g1_ppo_config() -> ConfigPPO:
    return default_g1_ppo_config()


def _copy_trainer_policy(dst: TrainerPPO, src: TrainerPPO) -> None:
    dst.actor.copy_from(src.actor)
    if dst.critic is not None or src.critic is not None:
        if dst.critic is None or src.critic is None:
            raise RuntimeError("trainer critic layouts do not match")
        dst.critic.copy_from(src.critic)


def _copy_g1_trainer_policy(dst: TrainerPPO, src: TrainerPPO) -> None:
    _copy_trainer_policy(dst, src)


def _make_g1_rollout_trainer(env: EnvG1PhoenX, trainer: TrainerPPO) -> TrainerPPO:
    rollout = TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=trainer.hidden_layers,
        config=trainer.config,
        device=env.device,
        seed=trainer.seed + 17,
        squash_actions=trainer.squash_actions,
        activation=trainer.activation,
        log_std_init=trainer.log_std_init,
        mirror_map=g1_mirror_map_ppo() if trainer.config.mirror_loss_coeff > 0.0 else None,
    )
    _copy_trainer_policy(rollout, trainer)
    return rollout


def _make_anymal_rollout_trainer(env: EnvAnymalPhoenX, trainer: TrainerPPO) -> TrainerPPO:
    rollout = TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=trainer.hidden_layers,
        config=trainer.config,
        device=env.device,
        seed=trainer.seed + 17,
        squash_actions=trainer.squash_actions,
        activation=trainer.activation,
        log_std_init=trainer.log_std_init,
        mirror_map=anymal_mirror_map_ppo() if trainer.config.mirror_loss_coeff > 0.0 else None,
    )
    _copy_trainer_policy(rollout, trainer)
    return rollout


def _make_g1_buffer(env: EnvG1PhoenX, steps: int) -> BufferRollout:
    return BufferRollout(
        num_steps=int(steps),
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=env.device,
    )


def _make_anymal_buffer(env: EnvAnymalPhoenX, steps: int) -> BufferRollout:
    return BufferRollout(
        num_steps=int(steps),
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=env.device,
    )


def _capture_stream_graph(stream: wp.Stream, device: wp.context.Device, workload) -> object:
    main_stream = wp.get_stream(device)
    with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
        wp.wait_stream(main_stream)
        with wp.ScopedCapture(device=device, stream=stream) as capture:
            workload()
    wp.wait_stream(stream)
    wp.synchronize_device(device)
    return capture.graph


class _G1GraphTrainPhase:
    def __init__(self, rollout_graph, update_graph):
        self.rollout_graph = rollout_graph
        self.update_graph = update_graph


def _g1_command_metric_override(cfg: ConfigTrainG1PPO, env: EnvG1PhoenX) -> tuple[float, float, float]:
    if not cfg.randomize_commands:
        return tuple(float(v) for v in env.config.command)
    return (
        0.5 * (float(cfg.command_x_range[0]) + float(cfg.command_x_range[1])),
        0.5 * (float(cfg.command_y_range[0]) + float(cfg.command_y_range[1])),
        0.5 * (float(cfg.command_yaw_range[0]) + float(cfg.command_yaw_range[1])),
    )


def _make_g1_command_curriculum_counter(
    cfg: ConfigTrainG1PPO, env: EnvG1PhoenX, start_iteration: int
) -> wp.array[wp.int32]:
    start_samples = int(start_iteration) * int(cfg.rollout_steps) * int(env.world_count)
    start_samples = min(start_samples, np.iinfo(np.int32).max)
    return wp.array(np.asarray([start_samples], dtype=np.int32), dtype=wp.int32, device=env.device)


def _advance_g1_command_curriculum(
    cfg: ConfigTrainG1PPO, env: EnvG1PhoenX, sample_counter: wp.array[wp.int32], sample_delta: int
) -> None:
    if not cfg.randomize_commands:
        return
    env.update_command_curriculum(
        sample_counter,
        sample_delta=int(sample_delta),
        start_scale=float(cfg.command_curriculum_start),
        ramp_samples=float(cfg.command_curriculum_samples),
    )


def _make_g1_target_curriculum_counter(
    cfg: ConfigTrainG1PPO, env: EnvG1PhoenX, start_iteration: int
) -> wp.array[wp.int32]:
    if cfg.target_curriculum_start_samples is None:
        start_samples = int(start_iteration) * int(cfg.rollout_steps) * int(env.world_count)
    else:
        start_samples = int(cfg.target_curriculum_start_samples)
    start_samples = min(start_samples, np.iinfo(np.int32).max)
    return wp.array(np.asarray([start_samples], dtype=np.int32), dtype=wp.int32, device=env.device)


def _configure_g1_sparse_targets(
    cfg: ConfigTrainG1PPO,
    env: EnvG1PhoenX,
    sample_counter: wp.array[wp.int32],
    seed_counter: wp.array[wp.int32],
    sample_delta: int,
) -> None:
    if env.config.reward_mode not in ("sparse_target", "dense_target"):
        return
    if cfg.use_target_curriculum:
        env.update_sparse_target_distance_curriculum(
            sample_counter,
            sample_delta=int(sample_delta),
            start_distance=float(cfg.target_distance_start),
            end_distance=float(cfg.target_distance_end),
            ramp_samples=float(cfg.target_curriculum_samples),
        )
    if cfg.randomize_target_positions:
        env.randomize_target_positions_seed_counter(
            seed_counter=seed_counter,
            target_angle_range=(float(cfg.target_angle_min), float(cfg.target_angle_max)),
            target_distance_min=float(cfg.target_distance_start) if cfg.randomize_target_distance_range else None,
        )
    else:
        env.set_sparse_targets_from_distance()


def _g1_graph_rollout_metrics(
    diagnostics: _G1TrainDiagnosticsReadback | None,
    buffer: BufferRollout,
    env: EnvG1PhoenX,
    trainer: TrainerPPO,
    cfg: ConfigTrainG1PPO,
) -> tuple[tuple[float, float, float, float, float, float], StatsPPOUpdate, tuple[float, float, float, float, float]]:
    if diagnostics is None:
        action_metrics = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        return _g1_disabled_rollout_metrics(env), StatsPPOUpdate(0.0, 0.0, 0.0, 0.0), action_metrics
    rollout_metrics, update_stats, action_metrics = diagnostics.read(buffer, env, trainer)
    command = _g1_command_metric_override(cfg, env)
    rollout_metrics = (rollout_metrics[0], rollout_metrics[1], rollout_metrics[2], *command)
    return rollout_metrics, update_stats, action_metrics


def _maybe_log_g1_train_stats(cfg: ConfigTrainG1PPO, stats: StatsTrainG1PPO, local_iteration: int) -> None:
    if cfg.log_interval > 0 and (stats.iteration % cfg.log_interval == 0 or local_iteration == cfg.iterations - 1):
        print(
            f"iter={stats.iteration:04d} "
            f"reward={stats.mean_reward:.4f} "
            f"perf={stats.mean_tracking_perf:.3f} "
            f"done={stats.mean_done:.4f} "
            f"act_rms={stats.clipped_action_rms:.3f} "
            f"act_clip={stats.action_clip_fraction:.3f} "
            f"logstd={stats.mean_log_std:.3f} "
            f"sps={stats.samples_per_second:.1f} "
            f"pi_loss={stats.policy_loss:.4f} "
            f"v_loss={stats.value_loss:.4f}"
        )


def _maybe_checkpoint_g1(cfg: ConfigTrainG1PPO, trainer: TrainerPPO) -> None:
    if cfg.checkpoint_path is not None and cfg.checkpoint_interval > 0:
        if trainer.iteration % int(cfg.checkpoint_interval) == 0:
            checkpoint_path = _format_checkpoint_path(cfg.checkpoint_path, trainer.iteration)
            trainer.save_checkpoint(checkpoint_path, iteration=trainer.iteration)


def _train_g1_ppo_graph_leapfrog(
    cfg: ConfigTrainG1PPO,
    env: EnvG1PhoenX,
    trainer: TrainerPPO,
    first_buffer: BufferRollout,
    command_curriculum_counter: wp.array[wp.int32],
    *,
    start_iteration: int,
    diagnostics: _G1TrainDiagnosticsReadback | None,
) -> ResultTrainG1PPO:
    device = env.device
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("G1 graph_leapfrog training requires CUDA with Warp mempool enabled")

    rollout_trainer = _make_g1_rollout_trainer(env, trainer)
    buffers = (first_buffer, _make_g1_buffer(env, int(cfg.rollout_steps)))
    for graph_trainer in (trainer, rollout_trainer):
        for buffer in buffers:
            graph_trainer.reserve_update_buffers(buffer)

    command_seed_counter = make_seed_counter(int(cfg.seed) + 53_321 + int(start_iteration), device=device)
    target_seed_counter = make_seed_counter(int(cfg.seed) + 71_129 + int(start_iteration), device=device)
    target_curriculum_counter = _make_g1_target_curriculum_counter(cfg, env, int(start_iteration))
    rollout_seed_counter = make_seed_counter(
        int(cfg.seed) + int(start_iteration) * int(cfg.rollout_steps), device=device
    )
    update_seed_counter = make_seed_counter(int(trainer.seed) + 1_000_003 * int(start_iteration), device=device)
    reset_seed_counter = make_seed_counter(
        int(cfg.seed) + 91_337 + int(start_iteration) * int(cfg.rollout_steps), device=device
    )
    env.use_reset_seed_counter(reset_seed_counter)

    env.use_command_seed_counter(command_seed_counter)

    def collect(buffer: BufferRollout) -> None:
        _advance_g1_command_curriculum(cfg, env, command_curriculum_counter, buffer.num_samples)
        if cfg.randomize_commands and cfg.command_sampling == "rollout":
            env.randomize_commands_seed_counter(
                seed_counter=command_seed_counter,
                command_x_range=cfg.command_x_range,
                command_y_range=cfg.command_y_range,
                command_yaw_range=cfg.command_yaw_range,
                zero_probability=cfg.command_zero_probability,
            )
        _configure_g1_sparse_targets(cfg, env, target_curriculum_counter, target_seed_counter, buffer.num_samples)
        env.collect_ppo_rollout_seed_counter(
            rollout_trainer,
            buffer,
            seed_counter=rollout_seed_counter,
            reset_state_at_start=bool(cfg.reset_recurrent_state_on_rollout_start),
        )

    def update(buffer: BufferRollout) -> None:
        trainer.update_seed_counter(buffer, seed_counter=update_seed_counter, read_stats=False)

    collect(buffers[0])
    wp.synchronize_device(device)

    rollout_stream = wp.Stream(device)
    update_stream = wp.Stream(device)
    copy_stream = wp.Stream(device)
    phases = (
        _G1GraphTrainPhase(
            _capture_stream_graph(rollout_stream, device, lambda: collect(buffers[1])),
            _capture_stream_graph(update_stream, device, lambda: update(buffers[0])),
        ),
        _G1GraphTrainPhase(
            _capture_stream_graph(rollout_stream, device, lambda: collect(buffers[0])),
            _capture_stream_graph(update_stream, device, lambda: update(buffers[1])),
        ),
    )
    copy_graph = _capture_stream_graph(copy_stream, device, lambda: _copy_g1_trainer_policy(rollout_trainer, trainer))

    history: list[StatsTrainG1PPO] = []
    prev = 0
    for local_iteration in range(int(cfg.iterations)):
        iteration = int(start_iteration) + local_iteration
        t0 = time.perf_counter()
        if local_iteration < int(cfg.iterations) - 1:
            phase = phases[prev]
            wp.capture_launch(phase.rollout_graph, stream=rollout_stream)
            wp.capture_launch(phase.update_graph, stream=update_stream)
            with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(rollout_stream)
                wp.wait_stream(update_stream)
            wp.capture_launch(copy_graph, stream=copy_stream)
            wp.synchronize_device(device)
        else:
            wp.capture_launch(phases[prev].update_graph, stream=update_stream)
            with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(update_stream)
            wp.capture_launch(copy_graph, stream=copy_stream)
            wp.synchronize_device(device)
        t1 = time.perf_counter()

        trainer.iteration = iteration + 1
        rollout_metrics, update_stats, action_metrics = _g1_graph_rollout_metrics(
            diagnostics, buffers[prev], env, trainer, cfg
        )
        stats = _merge_g1_stats(
            iteration, rollout_metrics, update_stats, action_metrics, t1 - t0, 0.0, buffers[prev].num_samples
        )
        history.append(stats)
        _maybe_log_g1_train_stats(cfg, stats, local_iteration)
        _maybe_checkpoint_g1(cfg, trainer)

        if local_iteration < int(cfg.iterations) - 1:
            prev = 1 - prev

    if cfg.checkpoint_path is not None:
        final_iteration = int(start_iteration) + int(cfg.iterations)
        trainer.iteration = final_iteration
        trainer.save_checkpoint(
            _format_checkpoint_path(cfg.checkpoint_path, final_iteration), iteration=final_iteration
        )

    return ResultTrainG1PPO(trainer=trainer, env=env, buffer=buffers[prev], history=history)


def _check_anymal_graph_leapfrog_config(cfg: ConfigTrainAnymalPPO, env: EnvAnymalPhoenX) -> None:
    if cfg.randomize_commands and env.config.reward_mode != "dense_command":
        raise RuntimeError("Anymal graph_leapfrog command randomization requires dense_command reward mode")
    if env.config.reward_mode == "sparse_target" and (cfg.randomize_target_positions or cfg.use_target_curriculum):
        raise RuntimeError("Anymal graph_leapfrog currently requires a fixed sparse target")


def _train_anymal_ppo_graph_leapfrog(
    cfg: ConfigTrainAnymalPPO,
    env: EnvAnymalPhoenX,
    trainer: TrainerPPO,
    first_buffer: BufferRollout,
    *,
    start_iteration: int,
    command_x: float,
    target_distance: float,
    diagnostics: _AnymalTrainDiagnosticsReadback,
) -> ResultTrainAnymalPPO:
    _check_anymal_graph_leapfrog_config(cfg, env)
    device = env.device
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("Anymal graph_leapfrog training requires CUDA with Warp mempool enabled")

    rollout_trainer = _make_anymal_rollout_trainer(env, trainer)
    buffers = (first_buffer, _make_anymal_buffer(env, int(cfg.rollout_steps)))
    for graph_trainer in (trainer, rollout_trainer):
        for buffer in buffers:
            graph_trainer.reserve_update_buffers(buffer)

    rollout_seed_counter = make_seed_counter(
        int(cfg.seed) + int(start_iteration) * int(cfg.rollout_steps), device=device
    )
    command_seed_counter = make_seed_counter(int(cfg.seed) + 29_131 + int(start_iteration), device=device)
    update_seed_counter = make_seed_counter(int(trainer.seed) + 1_000_003 * int(start_iteration), device=device)

    def collect(buffer: BufferRollout) -> None:
        if cfg.randomize_commands:
            env.randomize_commands_seed_counter(
                seed_counter=command_seed_counter,
                command_x_range=cfg.command_x_range,
                command_y_range=cfg.command_y_range,
                command_yaw_range=cfg.command_yaw_range,
                command_height_range=cfg.command_height_range,
                command_yaw_min_abs=cfg.command_yaw_min_abs,
                zero_probability=cfg.command_zero_probability,
            )
        collect_ppo_rollout_seed_counter(env, rollout_trainer, buffer, seed_counter=rollout_seed_counter)

    def update(buffer: BufferRollout) -> None:
        trainer.update_seed_counter(buffer, seed_counter=update_seed_counter, read_stats=False)

    collect(buffers[0])
    wp.synchronize_device(device)

    rollout_stream = wp.Stream(device)
    update_stream = wp.Stream(device)
    copy_stream = wp.Stream(device)
    phases = (
        _G1GraphTrainPhase(
            _capture_stream_graph(rollout_stream, device, lambda: collect(buffers[1])),
            _capture_stream_graph(update_stream, device, lambda: update(buffers[0])),
        ),
        _G1GraphTrainPhase(
            _capture_stream_graph(rollout_stream, device, lambda: collect(buffers[0])),
            _capture_stream_graph(update_stream, device, lambda: update(buffers[1])),
        ),
    )
    copy_graph = _capture_stream_graph(copy_stream, device, lambda: _copy_trainer_policy(rollout_trainer, trainer))

    history: list[StatsTrainAnymalPPO] = []
    prev = 0
    for local_iteration in range(int(cfg.iterations)):
        iteration = int(start_iteration) + local_iteration
        if local_iteration < int(cfg.iterations) - 1:
            phase = phases[prev]
            wp.capture_launch(phase.rollout_graph, stream=rollout_stream)
            wp.capture_launch(phase.update_graph, stream=update_stream)
            with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(rollout_stream)
                wp.wait_stream(update_stream)
            wp.capture_launch(copy_graph, stream=copy_stream)
            wp.synchronize_device(device)
        else:
            wp.capture_launch(phases[prev].update_graph, stream=update_stream)
            with wp.ScopedStream(copy_stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(update_stream)
            wp.capture_launch(copy_graph, stream=copy_stream)
            wp.synchronize_device(device)

        rollout_metrics, update_stats = diagnostics.read(
            buffers[prev],
            trainer,
            command_x,
            target_distance,
            use_observed_command=cfg.randomize_commands,
        )
        stats = _merge_stats(iteration, rollout_metrics, update_stats)
        history.append(stats)
        if cfg.log_interval > 0 and (iteration % cfg.log_interval == 0 or local_iteration == cfg.iterations - 1):
            print(
                f"iter={iteration:04d} "
                f"reward={stats.mean_reward:.4f} "
                f"success={stats.mean_success:.3f} "
                f"target={stats.target_distance:.2f} "
                f"vx={stats.mean_forward_velocity:.4f} "
                f"|vx-cmd|={stats.mean_abs_forward_velocity_error:.4f} "
                f"done={stats.mean_done:.4f} "
                f"pi_loss={stats.policy_loss:.4f} "
                f"v_loss={stats.value_loss:.4f}"
            )
        trainer.iteration = iteration + 1
        if cfg.checkpoint_path is not None and cfg.checkpoint_interval > 0:
            if trainer.iteration % int(cfg.checkpoint_interval) == 0:
                checkpoint_path = _format_checkpoint_path(cfg.checkpoint_path, trainer.iteration)
                trainer.save_checkpoint(checkpoint_path, iteration=trainer.iteration)
        if local_iteration < int(cfg.iterations) - 1:
            prev = 1 - prev

    if cfg.checkpoint_path is not None:
        final_iteration = int(start_iteration) + int(cfg.iterations)
        trainer.iteration = final_iteration
        trainer.save_checkpoint(
            _format_checkpoint_path(cfg.checkpoint_path, final_iteration), iteration=final_iteration
        )

    return ResultTrainAnymalPPO(trainer=trainer, env=env, buffer=buffers[prev], history=history)


def train_g1_ppo(config: ConfigTrainG1PPO | None = None) -> ResultTrainG1PPO:
    """Train a Unitree G1 walking policy with PhoenX and Warp-only PPO."""

    cfg = config or ConfigTrainG1PPO()
    if cfg.iterations <= 0:
        raise ValueError("iterations must be positive")
    if cfg.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if cfg.command_x_range[1] < cfg.command_x_range[0]:
        raise ValueError("command_x_range must be ordered")
    if cfg.command_y_range[1] < cfg.command_y_range[0]:
        raise ValueError("command_y_range must be ordered")
    if cfg.command_yaw_range[1] < cfg.command_yaw_range[0]:
        raise ValueError("command_yaw_range must be ordered")
    if not 0.0 <= float(cfg.command_zero_probability) <= 1.0:
        raise ValueError("command_zero_probability must be in [0, 1]")
    if cfg.command_resample_steps < 0:
        raise ValueError("command_resample_steps must be non-negative")
    if not 0.0 <= float(cfg.command_curriculum_start) <= 1.0:
        raise ValueError("command_curriculum_start must be in [0, 1]")
    if int(cfg.command_curriculum_samples) < 0:
        raise ValueError("command_curriculum_samples must be non-negative")
    if float(cfg.target_distance_start) < 0.0 or float(cfg.target_distance_end) < 0.0:
        raise ValueError("target curriculum distances must be non-negative")
    if int(cfg.target_curriculum_samples) < 0:
        raise ValueError("target_curriculum_samples must be non-negative")
    if cfg.target_curriculum_start_samples is not None and int(cfg.target_curriculum_start_samples) < 0:
        raise ValueError("target_curriculum_start_samples must be non-negative when provided")
    if cfg.target_angle_max < cfg.target_angle_min:
        raise ValueError("target_angle_max must be greater than or equal to target_angle_min")
    if cfg.command_sampling not in ("episode", "rollout"):
        raise ValueError("command_sampling must be 'episode' or 'rollout'")
    if cfg.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if cfg.execution_mode not in ("eager", "graph_leapfrog"):
        raise ValueError("execution_mode must be 'eager' or 'graph_leapfrog'")

    device = wp.get_device(cfg.device)
    env_config = cfg.env_config or ConfigEnvG1PhoenX()
    if cfg.randomize_commands and cfg.command_sampling == "episode":
        env_config = replace(
            env_config,
            command_x_range=cfg.command_x_range,
            command_y_range=cfg.command_y_range,
            command_yaw_range=cfg.command_yaw_range,
            randomize_commands_on_reset=True,
            command_zero_probability=cfg.command_zero_probability,
            command_resample_steps=cfg.command_resample_steps,
        )
    elif not cfg.randomize_commands or cfg.command_sampling == "rollout":
        env_config = replace(env_config, randomize_commands_on_reset=False, command_resample_steps=0)
    ppo_config = cfg.ppo_config or _default_g1_ppo_config()
    env = EnvG1PhoenX(env_config, device=device)
    if cfg.resume_checkpoint is not None:
        trainer = load_ppo_checkpoint(cfg.resume_checkpoint, config=cfg.ppo_config, device=device)
        ppo_config = trainer.config
        if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
            raise ValueError("Checkpoint dimensions do not match the G1 environment")
        if trainer.config.mirror_loss_coeff > 0.0:
            trainer.set_mirror_map(g1_mirror_map_ppo())
    else:
        trainer = TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=cfg.hidden_layers,
            config=ppo_config,
            device=device,
            seed=cfg.seed,
            squash_actions=cfg.squash_actions,
            activation=cfg.activation,
            log_std_init=cfg.log_std_init,
            mirror_map=g1_mirror_map_ppo() if ppo_config.mirror_loss_coeff > 0.0 else None,
        )
    buffer = _make_g1_buffer(env, int(cfg.rollout_steps))
    trainer.reserve_update_buffers(buffer)

    history: list[StatsTrainG1PPO] = []
    start_iteration = int(getattr(trainer, "iteration", 0))
    diagnostics = _G1TrainDiagnosticsReadback(device) if cfg.readback_diagnostics else None
    command_curriculum_counter = _make_g1_command_curriculum_counter(cfg, env, start_iteration)
    target_curriculum_counter = _make_g1_target_curriculum_counter(cfg, env, start_iteration)
    target_seed_counter = make_seed_counter(int(cfg.seed) + 71_129 + int(start_iteration), device=device)
    _advance_g1_command_curriculum(cfg, env, command_curriculum_counter, 0)
    _configure_g1_sparse_targets(cfg, env, target_curriculum_counter, target_seed_counter, 0)
    if cfg.randomize_commands and cfg.command_sampling == "episode":
        env.randomize_commands(
            seed=int(cfg.seed) + 53_321 + start_iteration,
            command_x_range=cfg.command_x_range,
            command_y_range=cfg.command_y_range,
            command_yaw_range=cfg.command_yaw_range,
            zero_probability=cfg.command_zero_probability,
        )

    if cfg.execution_mode == "graph_leapfrog":
        return _train_g1_ppo_graph_leapfrog(
            cfg,
            env,
            trainer,
            buffer,
            command_curriculum_counter,
            start_iteration=start_iteration,
            diagnostics=diagnostics,
        )

    for local_iteration in range(cfg.iterations):
        iteration = start_iteration + local_iteration
        _advance_g1_command_curriculum(cfg, env, command_curriculum_counter, buffer.num_samples)
        if cfg.randomize_commands and cfg.command_sampling == "rollout":
            env.randomize_commands(
                seed=int(cfg.seed) + 53_321 + iteration,
                command_x_range=cfg.command_x_range,
                command_y_range=cfg.command_y_range,
                command_yaw_range=cfg.command_yaw_range,
                zero_probability=cfg.command_zero_probability,
            )
        _configure_g1_sparse_targets(cfg, env, target_curriculum_counter, target_seed_counter, buffer.num_samples)

        t0 = time.perf_counter()
        env.collect_ppo_rollout(
            trainer,
            buffer,
            seed=cfg.seed + iteration * cfg.rollout_steps,
            reset_state_at_start=bool(cfg.reset_recurrent_state_on_rollout_start),
        )
        t1 = time.perf_counter()
        update_stats = trainer.update(buffer, read_stats=False)
        if diagnostics is not None:
            rollout_metrics, update_stats, action_metrics = diagnostics.read(buffer, env, trainer)
        else:
            rollout_metrics = _g1_disabled_rollout_metrics(env)
            action_metrics = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        t2 = time.perf_counter()
        stats = _merge_g1_stats(
            iteration, rollout_metrics, update_stats, action_metrics, t1 - t0, t2 - t1, buffer.num_samples
        )
        history.append(stats)

        _maybe_log_g1_train_stats(cfg, stats, local_iteration)
        trainer.iteration = iteration + 1
        _maybe_checkpoint_g1(cfg, trainer)

    if cfg.checkpoint_path is not None:
        final_iteration = start_iteration + int(cfg.iterations)
        trainer.iteration = final_iteration
        trainer.save_checkpoint(
            _format_checkpoint_path(cfg.checkpoint_path, final_iteration), iteration=final_iteration
        )

    return ResultTrainG1PPO(trainer=trainer, env=env, buffer=buffer, history=history)


def evaluate_g1_ppo(trainer: TrainerPPO, config: ConfigEvaluateG1PPO | None = None) -> ResultEvaluateG1PPO:
    """Evaluate a saved or in-memory G1 PPO policy with deterministic rollouts."""

    cfg = config or ConfigEvaluateG1PPO()
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    device = wp.get_device(cfg.device if cfg.device is not None else trainer.device)
    base_env_config = cfg.env_config or ConfigEnvG1PhoenX(world_count=64)
    eval_config = replace(base_env_config, auto_reset=False, max_episode_steps=0)
    env = EnvG1PhoenX(eval_config, device=device)
    if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
        raise ValueError("Trainer dimensions do not match the G1 environment")

    obs = env.reset()
    trainer.reset_rollout_state()
    start_q = _joint_q_matrix_g1(env)
    start_xy = start_q[:, 0:2].copy()
    previous_xy = start_xy.copy()
    last_alive_xy = start_xy.copy()
    path_length = np.zeros(env.world_count, dtype=np.float64)
    first_done_step = np.full(env.world_count, -1, dtype=np.int32)
    reward_sum = 0.0
    done_sum = 0.0
    tracking_perf_sum = 0.0
    t0 = time.perf_counter()
    for step in range(int(cfg.steps)):
        alive_before = first_done_step < 0
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + step, deterministic=bool(cfg.deterministic)
        )
        obs, rewards, dones = env.step(actions)
        reward_sum += float(np.mean(rewards.numpy()))
        done_np = dones.numpy() > 0.5
        done_sum += float(np.mean(done_np))
        tracking_perf_sum += float(np.mean(env.step_successes.numpy()))

        q = _joint_q_matrix_g1(env)
        xy = q[:, 0:2].copy()
        path_length[alive_before] += np.linalg.norm(xy[alive_before] - previous_xy[alive_before], axis=1)
        last_alive_xy[alive_before] = xy[alive_before]
        previous_xy = xy
        first_done_step[(first_done_step < 0) & done_np] = step + 1
    elapsed = max(time.perf_counter() - t0, 1.0e-12)
    command_np = env.command.numpy()
    steps = int(cfg.steps)
    command_xy = command_np[:, 0:2].astype(np.float64, copy=False)
    command_norm = np.linalg.norm(command_xy, axis=1)
    command_dir = np.zeros_like(command_xy, dtype=np.float64)
    moving = command_norm > 1.0e-6
    command_dir[moving] = command_xy[moving] / command_norm[moving, None]
    command_dir[~moving, 0] = 1.0
    lateral_dir = np.stack((-command_dir[:, 1], command_dir[:, 0]), axis=1)
    displacement = last_alive_xy - start_xy
    aligned_displacement = np.sum(displacement * command_dir, axis=1)
    lateral_displacement = np.sum(displacement * lateral_dir, axis=1)
    survival_steps = np.where(first_done_step >= 0, first_done_step, steps)
    eval_seconds = max(float(steps) * float(env.config.frame_dt), 1.0e-12)
    stats = StatsEvaluateG1PPO(
        steps=steps,
        mean_reward=reward_sum / float(steps),
        mean_done=done_sum / float(steps),
        mean_tracking_perf=tracking_perf_sum / float(steps),
        mean_command_x=float(np.mean(command_np[:, 0])),
        mean_command_y=float(np.mean(command_np[:, 1])),
        mean_command_yaw=float(np.mean(command_np[:, 2])),
        fall_fraction=float(np.mean(first_done_step >= 0)),
        mean_survival_steps=float(np.mean(survival_steps)),
        mean_command_aligned_displacement=float(np.mean(aligned_displacement)),
        mean_command_aligned_velocity=float(np.mean(aligned_displacement) / eval_seconds),
        mean_lateral_displacement_abs=float(np.mean(np.abs(lateral_displacement))),
        mean_path_length=float(np.mean(path_length)),
        samples_per_second=float(env.world_count * steps) / elapsed,
    )
    return ResultEvaluateG1PPO(stats=stats)


def evaluate_g1_target_ppo(
    trainer: TrainerPPO, config: ConfigEvaluateG1TargetPPO | None = None
) -> ResultEvaluateG1TargetPPO:
    """Evaluate whether a trained G1 PPO policy reaches sparse targets.

    The evaluator disables auto-reset and records first terminal events, so a
    target hit is not conflated with post-reset motion.
    """

    cfg = config or ConfigEvaluateG1TargetPPO()
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    device = wp.get_device(cfg.device if cfg.device is not None else trainer.device)
    base_env_config = cfg.env_config or ConfigEnvG1PhoenX(world_count=64, reward_mode="sparse_target")
    stats = []
    for target in cfg.target_positions:
        eval_config = replace(
            base_env_config,
            reward_mode="sparse_target",
            sparse_target_position=(float(target[0]), float(target[1])),
            auto_reset=False,
            max_episode_steps=0,
        )
        env = EnvG1PhoenX(eval_config, device=device)
        _check_g1_trainer_dimensions(trainer, env)
        stats.append(_evaluate_g1_target(trainer, env, cfg, (float(target[0]), float(target[1]))))
    return ResultEvaluateG1TargetPPO(stats=stats)


def evaluate_g1_gate_ppo(trainer: TrainerPPO, config: ConfigEvaluateG1GatePPO | None = None) -> ResultEvaluateG1GatePPO:
    """Evaluate a G1 PPO policy against nanoG1-style quality-gate metrics."""

    cfg = config or ConfigEvaluateG1GatePPO()
    _validate_g1_gate_config(cfg)
    device = wp.get_device(cfg.device if cfg.device is not None else trainer.device)
    base_env_config = cfg.env_config or ConfigEnvG1PhoenX()
    commands = _g1_gate_commands_array(cfg.battery_commands)

    battery_config = replace(
        base_env_config,
        world_count=int(commands.shape[0]) * int(cfg.seeds_per_command),
        auto_reset=False,
        max_episode_steps=0,
    )
    battery_env = EnvG1PhoenX(battery_config, device=device)
    _check_g1_trainer_dimensions(trainer, battery_env)

    diagnostic_config = replace(
        base_env_config,
        world_count=int(cfg.diagnostic_world_count),
        command=tuple(float(x) for x in cfg.diagnostic_command),
        auto_reset=False,
        max_episode_steps=0,
    )
    diagnostic_env = EnvG1PhoenX(diagnostic_config, device=device)
    _check_g1_trainer_dimensions(trainer, diagnostic_env)

    t0 = time.perf_counter()
    per_command, battery_falls, battery_perf, battery_samples = _evaluate_g1_gate_battery(
        trainer, battery_env, cfg, commands
    )
    diagnostic = _evaluate_g1_gate_diagnostics(trainer, diagnostic_env, cfg)
    elapsed = max(time.perf_counter() - t0, 1.0e-12)

    diagnostic_falls, diagnostic_samples, action_jerk, ang_vel_xy, yaw_rate, leg_qvel = diagnostic
    pass_gate = _g1_gate_passes(
        cfg,
        battery_falls=battery_falls,
        battery_perf=battery_perf,
        action_jerk_rms=action_jerk,
        ang_vel_xy_rms=ang_vel_xy,
        yaw_rate_rms=yaw_rate,
        leg_qvel_rms=leg_qvel,
    )
    total_samples = int(battery_samples) + int(diagnostic_samples)
    stats = StatsEvaluateG1GatePPO(
        battery_falls=int(battery_falls),
        battery_perf=float(battery_perf),
        action_jerk_rms=float(action_jerk),
        ang_vel_xy_rms=float(ang_vel_xy),
        yaw_rate_rms=float(yaw_rate),
        leg_qvel_rms=float(leg_qvel),
        diagnostic_falls=int(diagnostic_falls),
        battery_samples=int(battery_samples),
        diagnostic_samples=int(diagnostic_samples),
        samples_per_second=float(total_samples) / elapsed,
        pass_gate=bool(pass_gate),
        per_command=per_command,
    )
    return ResultEvaluateG1GatePPO(stats=stats)


def train_anymal_ppo(config: ConfigTrainAnymalPPO | None = None) -> ResultTrainAnymalPPO:
    """Train an Anymal C walking policy from scratch with Warp-only PPO.

    Args:
        config: Training configuration.

    Returns:
        The trained PPO trainer, environment, rollout buffer, and metric history.
    """

    cfg = config or ConfigTrainAnymalPPO()
    if cfg.iterations <= 0:
        raise ValueError("iterations must be positive")
    if cfg.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if cfg.target_angle_max <= cfg.target_angle_min:
        raise ValueError("target_angle_max must be greater than target_angle_min")
    if cfg.command_x_range[1] < cfg.command_x_range[0]:
        raise ValueError("command_x_range must be ordered")
    if cfg.command_y_range[1] < cfg.command_y_range[0]:
        raise ValueError("command_y_range must be ordered")
    if cfg.command_yaw_range[1] < cfg.command_yaw_range[0]:
        raise ValueError("command_yaw_range must be ordered")
    if cfg.command_height_range[1] < cfg.command_height_range[0]:
        raise ValueError("command_height_range must be ordered")
    if float(cfg.command_yaw_min_abs) < 0.0:
        raise ValueError("command_yaw_min_abs must be non-negative")
    if not 0.0 <= float(cfg.command_zero_probability) <= 1.0:
        raise ValueError("command_zero_probability must be in [0, 1]")
    if cfg.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if cfg.execution_mode not in ("eager", "graph_leapfrog"):
        raise ValueError("execution_mode must be 'eager' or 'graph_leapfrog'")

    device = wp.get_device(cfg.device)
    env_config = cfg.env_config or ConfigEnvAnymalPhoenX()
    ppo_config = cfg.ppo_config or _default_ppo_config()
    env = EnvAnymalPhoenX(env_config, device=device)
    if cfg.resume_checkpoint is not None:
        trainer = load_ppo_checkpoint(cfg.resume_checkpoint, config=cfg.ppo_config, device=device)
        ppo_config = trainer.config
        if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
            raise ValueError("Checkpoint dimensions do not match the Anymal environment")
        if trainer.config.mirror_loss_coeff > 0.0:
            trainer.set_mirror_map(anymal_mirror_map_ppo())
    else:
        trainer = TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=cfg.hidden_layers,
            config=ppo_config,
            device=device,
            seed=cfg.seed,
            squash_actions=True,
            activation=cfg.activation,
            log_std_init=cfg.log_std_init,
            mirror_map=anymal_mirror_map_ppo() if ppo_config.mirror_loss_coeff > 0.0 else None,
        )
    buffer = _make_anymal_buffer(env, int(cfg.rollout_steps))
    trainer.reserve_update_buffers(buffer)
    diagnostics = _AnymalTrainDiagnosticsReadback(device, buffer.num_samples)

    history: list[StatsTrainAnymalPPO] = []
    start_iteration = int(getattr(trainer, "iteration", 0))
    command_x = float(env.config.command[0])
    target_xy = np.asarray(env.config.target_position, dtype=np.float32)
    target_distance = float(np.linalg.norm(target_xy))
    target_direction = _target_direction(target_xy)
    if cfg.use_target_curriculum and env.config.reward_mode == "sparse_target":
        target_distance = float(cfg.target_distance_start)

    target_rng = np.random.default_rng(int(cfg.seed) + 17_917 + start_iteration)
    _configure_rollout_targets(env, cfg, target_rng, target_direction, target_distance)
    _configure_rollout_commands(env, cfg, target_rng)

    if cfg.execution_mode == "graph_leapfrog":
        return _train_anymal_ppo_graph_leapfrog(
            cfg,
            env,
            trainer,
            buffer,
            start_iteration=start_iteration,
            command_x=command_x,
            target_distance=target_distance,
            diagnostics=diagnostics,
        )

    for local_iteration in range(cfg.iterations):
        iteration = start_iteration + local_iteration
        target_rng = np.random.default_rng(int(cfg.seed) + 17_917 + iteration)
        _configure_rollout_targets(env, cfg, target_rng, target_direction, target_distance)
        _configure_rollout_commands(env, cfg, target_rng)
        env.collect_ppo_rollout(trainer, buffer, seed=cfg.seed + iteration * cfg.rollout_steps)
        trainer.update(buffer, read_stats=False)
        rollout_metrics, update_stats = diagnostics.read(
            buffer,
            trainer,
            command_x,
            target_distance,
            use_observed_command=cfg.randomize_commands,
        )
        stats = _merge_stats(iteration, rollout_metrics, update_stats)
        history.append(stats)
        if cfg.log_interval > 0 and (iteration % cfg.log_interval == 0 or local_iteration == cfg.iterations - 1):
            print(
                f"iter={iteration:04d} "
                f"reward={stats.mean_reward:.4f} "
                f"success={stats.mean_success:.3f} "
                f"target={stats.target_distance:.2f} "
                f"vx={stats.mean_forward_velocity:.4f} "
                f"|vx-cmd|={stats.mean_abs_forward_velocity_error:.4f} "
                f"done={stats.mean_done:.4f} "
                f"pi_loss={stats.policy_loss:.4f} "
                f"v_loss={stats.value_loss:.4f}"
            )
        if (
            cfg.use_target_curriculum
            and env.config.reward_mode == "sparse_target"
            and stats.mean_success >= cfg.target_success_threshold
            and target_distance < cfg.target_distance_end
        ):
            target_distance = min(float(cfg.target_distance_end), target_distance + float(cfg.target_distance_step))
        trainer.iteration = iteration + 1
        if cfg.checkpoint_path is not None and cfg.checkpoint_interval > 0:
            if trainer.iteration % int(cfg.checkpoint_interval) == 0:
                checkpoint_path = _format_checkpoint_path(cfg.checkpoint_path, trainer.iteration)
                trainer.save_checkpoint(checkpoint_path, iteration=trainer.iteration)

    if cfg.checkpoint_path is not None:
        final_iteration = start_iteration + int(cfg.iterations)
        trainer.iteration = final_iteration
        trainer.save_checkpoint(
            _format_checkpoint_path(cfg.checkpoint_path, final_iteration), iteration=final_iteration
        )

    return ResultTrainAnymalPPO(trainer=trainer, env=env, buffer=buffer, history=history)


def evaluate_anymal_ppo(trainer: TrainerPPO, config: ConfigEvaluateAnymalPPO | None = None) -> ResultEvaluateAnymalPPO:
    """Evaluate whether a trained Anymal PPO policy reaches target points.

    The evaluator disables auto-reset and records first terminal events, so a
    target hit is not conflated with post-reset motion.

    Args:
        trainer: Trained PPO trainer.
        config: Evaluation configuration.

    Returns:
        Target-wise deterministic or stochastic rollout metrics.
    """

    cfg = config or ConfigEvaluateAnymalPPO()
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    device = wp.get_device(cfg.device if cfg.device is not None else trainer.device)
    base_env_config = cfg.env_config or ConfigEnvAnymalPhoenX(world_count=64)
    stats = []
    for target in cfg.target_positions:
        eval_config = replace(
            base_env_config,
            target_position=(float(target[0]), float(target[1])),
            auto_reset=False,
            max_episode_steps=0,
        )
        env = EnvAnymalPhoenX(eval_config, device=device)
        stats.append(_evaluate_target(trainer, env, cfg, (float(target[0]), float(target[1]))))
    return ResultEvaluateAnymalPPO(stats=stats)


def _format_checkpoint_path(path: str, iteration: int) -> Path:
    if "{" in path:
        return Path(path.format(iteration=int(iteration)))
    return Path(path)


def _configure_rollout_targets(
    env: EnvAnymalPhoenX,
    cfg: ConfigTrainAnymalPPO,
    rng: np.random.Generator,
    target_direction: np.ndarray,
    target_distance: float,
) -> None:
    if env.config.reward_mode != "sparse_target":
        return
    if cfg.randomize_target_positions:
        positions = _sample_target_positions(
            rng=rng,
            world_count=env.world_count,
            target_direction=target_direction,
            target_distance=target_distance,
            angle_min=cfg.target_angle_min,
            angle_max=cfg.target_angle_max,
        )
        env.set_target_positions(positions)
    else:
        target_xy = target_direction * np.float32(target_distance)
        env.set_target_position((float(target_xy[0]), float(target_xy[1])))


def _configure_rollout_commands(env: EnvAnymalPhoenX, cfg: ConfigTrainAnymalPPO, rng: np.random.Generator) -> None:
    if env.config.reward_mode != "dense_command" or not cfg.randomize_commands:
        return
    commands = _sample_velocity_commands(
        rng=rng,
        world_count=env.world_count,
        x_range=cfg.command_x_range,
        y_range=cfg.command_y_range,
        yaw_range=cfg.command_yaw_range,
        height_range=cfg.command_height_range,
        yaw_min_abs=cfg.command_yaw_min_abs,
        zero_probability=cfg.command_zero_probability,
    )
    env.set_commands(commands)


def _sample_velocity_commands(
    *,
    rng: np.random.Generator,
    world_count: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range: tuple[float, float],
    height_range: tuple[float, float],
    yaw_min_abs: float,
    zero_probability: float,
) -> np.ndarray:
    commands = np.empty((int(world_count), 4), dtype=np.float32)
    commands[:, 0] = rng.uniform(float(x_range[0]), float(x_range[1]), size=int(world_count)).astype(np.float32)
    commands[:, 1] = rng.uniform(float(y_range[0]), float(y_range[1]), size=int(world_count)).astype(np.float32)
    commands[:, 2] = rng.uniform(float(yaw_range[0]), float(yaw_range[1]), size=int(world_count)).astype(np.float32)
    commands[:, 3] = rng.uniform(float(height_range[0]), float(height_range[1]), size=int(world_count)).astype(
        np.float32
    )
    min_abs_yaw = float(max(yaw_min_abs, 0.0))
    if min_abs_yaw > 0.0:
        yaw = commands[:, 2]
        signs = np.where(yaw < 0.0, -1.0, 1.0).astype(np.float32)
        commands[:, 2] = np.clip(signs * np.maximum(np.abs(yaw), min_abs_yaw), yaw_range[0], yaw_range[1])
    zero_p = float(np.clip(zero_probability, 0.0, 1.0))
    if zero_p > 0.0:
        zero_mask = rng.random(int(world_count)) < zero_p
        commands[zero_mask, :] = 0.0
    return commands


def _target_direction(target_xy: np.ndarray) -> np.ndarray:
    distance = float(np.linalg.norm(target_xy))
    if distance > 1.0e-6:
        return (target_xy / np.float32(distance)).astype(np.float32)
    return np.asarray((0.0, 1.0), dtype=np.float32)


def _sample_target_positions(
    *,
    rng: np.random.Generator,
    world_count: int,
    target_direction: np.ndarray,
    target_distance: float,
    angle_min: float,
    angle_max: float,
) -> np.ndarray:
    angles = rng.uniform(float(angle_min), float(angle_max), int(world_count)).astype(np.float32)
    forward = target_direction.astype(np.float32)
    right = np.asarray((forward[1], -forward[0]), dtype=np.float32)
    directions = np.cos(angles)[:, None] * forward[None, :] + np.sin(angles)[:, None] * right[None, :]
    return (directions * np.float32(target_distance)).astype(np.float32)


def _evaluate_target(
    trainer: TrainerPPO,
    env: EnvAnymalPhoenX,
    cfg: ConfigEvaluateAnymalPPO,
    target: tuple[float, float],
) -> StatsEvaluateAnymalTargetPPO:
    obs = env.reset()
    q0 = _joint_q_matrix(env)
    start_xy = q0[:, 0:2].copy()
    target_xy = np.asarray(target, dtype=np.float32)
    initial_distance = np.linalg.norm(target_xy[None, :] - start_xy, axis=1)
    min_distance = initial_distance.copy()
    last_distance = initial_distance.copy()
    previous_xy = start_xy.copy()
    path_length = np.zeros(env.world_count, dtype=np.float32)
    first_success_step = np.full(env.world_count, -1, dtype=np.int32)
    first_done_step = np.full(env.world_count, -1, dtype=np.int32)
    tilt_violation = np.zeros(env.world_count, dtype=bool)
    height_violation = np.zeros(env.world_count, dtype=bool)
    max_tilt_per_env = np.zeros(env.world_count, dtype=np.float32)
    min_height_per_env = np.full(env.world_count, np.inf, dtype=np.float32)
    max_height_per_env = np.full(env.world_count, -np.inf, dtype=np.float32)
    height_sum = 0.0
    height_count = 0
    forward_velocity_sum = 0.0
    forward_velocity_count = 0

    max_upright_cos = float(np.cos(np.deg2rad(float(cfg.max_tilt_degrees))))

    for step in range(int(cfg.steps)):
        alive_before = first_done_step < 0
        if not np.any(alive_before):
            break

        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + step, deterministic=bool(cfg.deterministic)
        )
        obs, _rewards, dones = env.step(actions)
        q = _joint_q_matrix(env)
        xy = q[:, 0:2].copy()
        distance = np.linalg.norm(target_xy[None, :] - xy, axis=1)
        min_distance = np.minimum(min_distance, distance)
        path_length[alive_before] += np.linalg.norm(xy[alive_before] - previous_xy[alive_before], axis=1)
        previous_xy = xy
        last_distance[alive_before] = distance[alive_before]

        obs_np = obs.numpy()
        alive_heights = q[alive_before, 2]
        alive_gravity = obs_np[alive_before, 6:9]
        upright_cos = np.clip(-alive_gravity[:, 2], -1.0, 1.0)
        tilt_degrees = np.rad2deg(np.arccos(upright_cos)).astype(np.float32)
        alive_indices = np.nonzero(alive_before)[0]
        max_tilt_per_env[alive_indices] = np.maximum(max_tilt_per_env[alive_indices], tilt_degrees)
        min_height_per_env[alive_indices] = np.minimum(min_height_per_env[alive_indices], alive_heights)
        max_height_per_env[alive_indices] = np.maximum(max_height_per_env[alive_indices], alive_heights)
        tilt_violation[alive_indices] |= upright_cos < max_upright_cos
        height_violation[alive_indices] |= (alive_heights < cfg.min_valid_base_height) | (
            alive_heights > cfg.max_valid_base_height
        )
        height_sum += float(np.sum(alive_heights))
        height_count += int(np.sum(alive_before))
        forward_velocity_sum += float(np.sum(obs_np[alive_before, 0]))
        forward_velocity_count += int(np.sum(alive_before))

        successes = env.step_successes.numpy() > 0.5
        done = dones.numpy() > 0.5
        first_success_step[(first_success_step < 0) & successes] = step + 1
        first_done_step[(first_done_step < 0) & done] = step + 1

    success = first_success_step >= 0
    strict_success = success & ~tilt_violation & ~height_violation
    done = first_done_step >= 0
    fall = done & ~success
    target_vector = target_xy[None, :] - start_xy
    target_norm = np.maximum(np.linalg.norm(target_vector, axis=1), 1.0e-6)
    displacement = previous_xy - start_xy
    aligned_displacement = np.sum(displacement * target_vector, axis=1) / target_norm
    mean_first_success_step = float(np.mean(first_success_step[success])) if np.any(success) else float("nan")
    elapsed = max(1, int(cfg.steps)) * float(env.config.frame_dt)
    mean_forward_velocity = forward_velocity_sum / float(max(forward_velocity_count, 1))

    return StatsEvaluateAnymalTargetPPO(
        target_position=target,
        success_fraction=float(np.mean(success)),
        strict_success_fraction=float(np.mean(strict_success)),
        fall_fraction=float(np.mean(fall)),
        tilt_violation_fraction=float(np.mean(tilt_violation)),
        height_violation_fraction=float(np.mean(height_violation)),
        mean_first_success_step=mean_first_success_step,
        mean_initial_distance=float(np.mean(initial_distance)),
        mean_final_distance=float(np.mean(last_distance)),
        mean_min_distance=float(np.mean(min_distance)),
        mean_target_aligned_displacement=float(np.mean(aligned_displacement)),
        mean_path_length=float(np.mean(path_length)),
        mean_speed=float(np.mean(path_length) / elapsed),
        mean_forward_velocity=float(mean_forward_velocity),
        max_tilt_degrees=float(np.max(max_tilt_per_env)),
        mean_max_tilt_degrees=float(np.mean(max_tilt_per_env)),
        min_base_height=float(np.min(min_height_per_env)),
        max_base_height=float(np.max(max_height_per_env)),
        mean_base_height=float(height_sum / float(max(height_count, 1))),
    )


def _joint_q_matrix(env: EnvAnymalPhoenX) -> np.ndarray:
    return env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)


def _evaluate_g1_target(
    trainer: TrainerPPO,
    env: EnvG1PhoenX,
    cfg: ConfigEvaluateG1TargetPPO,
    target: tuple[float, float],
) -> StatsEvaluateG1TargetPPO:
    obs = env.reset()
    trainer.reset_rollout_state()
    q0 = _joint_q_matrix_g1(env)
    start_xy = q0[:, 0:2].copy()
    target_xy = np.asarray(target, dtype=np.float32)
    initial_distance = np.linalg.norm(target_xy[None, :] - start_xy, axis=1)
    min_distance = initial_distance.copy()
    last_distance = initial_distance.copy()
    previous_xy = start_xy.copy()
    path_length = np.zeros(env.world_count, dtype=np.float32)
    first_success_step = np.full(env.world_count, -1, dtype=np.int32)
    first_done_step = np.full(env.world_count, -1, dtype=np.int32)
    tilt_violation = np.zeros(env.world_count, dtype=bool)
    height_violation = np.zeros(env.world_count, dtype=bool)
    max_tilt_per_env = np.zeros(env.world_count, dtype=np.float32)
    min_height_per_env = np.full(env.world_count, np.inf, dtype=np.float32)
    max_height_per_env = np.full(env.world_count, -np.inf, dtype=np.float32)
    height_sum = 0.0
    height_count = 0
    forward_velocity_sum = 0.0
    forward_velocity_count = 0
    max_upright_cos = float(np.cos(np.deg2rad(float(cfg.max_tilt_degrees))))

    for step in range(int(cfg.steps)):
        alive_before = first_done_step < 0
        if not np.any(alive_before):
            break
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + step, deterministic=bool(cfg.deterministic)
        )
        obs, _rewards, dones = env.step(actions)
        q = _joint_q_matrix_g1(env)
        xy = q[:, 0:2].copy()
        distance = np.linalg.norm(target_xy[None, :] - xy, axis=1)
        min_distance = np.minimum(min_distance, distance)
        path_length[alive_before] += np.linalg.norm(xy[alive_before] - previous_xy[alive_before], axis=1)
        previous_xy = xy
        last_distance[alive_before] = distance[alive_before]

        obs_np = obs.numpy()
        alive_heights = q[alive_before, 2]
        alive_gravity = obs_np[alive_before, 3:6]
        upright_cos = np.clip(-alive_gravity[:, 2], -1.0, 1.0)
        tilt_degrees = np.rad2deg(np.arccos(upright_cos)).astype(np.float32)
        alive_indices = np.nonzero(alive_before)[0]
        max_tilt_per_env[alive_indices] = np.maximum(max_tilt_per_env[alive_indices], tilt_degrees)
        min_height_per_env[alive_indices] = np.minimum(min_height_per_env[alive_indices], alive_heights)
        max_height_per_env[alive_indices] = np.maximum(max_height_per_env[alive_indices], alive_heights)
        tilt_violation[alive_indices] |= upright_cos < max_upright_cos
        height_violation[alive_indices] |= (alive_heights < cfg.min_valid_base_height) | (
            alive_heights > cfg.max_valid_base_height
        )
        height_sum += float(np.sum(alive_heights))
        height_count += int(np.sum(alive_before))
        forward_velocity_sum += float(np.sum(obs_np[alive_before, 0]))
        forward_velocity_count += int(np.sum(alive_before))

        successes = env.step_successes.numpy() > 0.5
        done = dones.numpy() > 0.5
        first_success_step[(first_success_step < 0) & successes] = step + 1
        first_done_step[(first_done_step < 0) & done] = step + 1

    success = first_success_step >= 0
    done = first_done_step >= 0
    fall = done & ~success
    strict_success = success & ~tilt_violation & ~height_violation
    target_vector = target_xy[None, :] - start_xy
    target_norm = np.maximum(np.linalg.norm(target_vector, axis=1), 1.0e-6)
    displacement = previous_xy - start_xy
    aligned_displacement = np.sum(displacement * target_vector, axis=1) / target_norm
    mean_first_success_step = float(np.mean(first_success_step[success])) if np.any(success) else float("nan")
    elapsed = max(1, int(cfg.steps)) * float(env.config.frame_dt)
    mean_forward_velocity = forward_velocity_sum / float(max(forward_velocity_count, 1))

    return StatsEvaluateG1TargetPPO(
        target_position=target,
        success_fraction=float(np.mean(success)),
        strict_success_fraction=float(np.mean(strict_success)),
        fall_fraction=float(np.mean(fall)),
        tilt_violation_fraction=float(np.mean(tilt_violation)),
        height_violation_fraction=float(np.mean(height_violation)),
        mean_first_success_step=mean_first_success_step,
        mean_initial_distance=float(np.mean(initial_distance)),
        mean_final_distance=float(np.mean(last_distance)),
        mean_min_distance=float(np.mean(min_distance)),
        mean_target_aligned_displacement=float(np.mean(aligned_displacement)),
        mean_path_length=float(np.mean(path_length)),
        mean_speed=float(np.mean(path_length) / elapsed),
        mean_forward_velocity=float(mean_forward_velocity),
        max_tilt_degrees=float(np.max(max_tilt_per_env)),
        mean_max_tilt_degrees=float(np.mean(max_tilt_per_env)),
        min_base_height=float(np.min(min_height_per_env)),
        max_base_height=float(np.max(max_height_per_env)),
        mean_base_height=float(height_sum / float(max(height_count, 1))),
    )


def _validate_g1_gate_config(cfg: ConfigEvaluateG1GatePPO) -> None:
    if cfg.seeds_per_command <= 0:
        raise ValueError("seeds_per_command must be positive")
    if cfg.battery_steps <= 0:
        raise ValueError("battery_steps must be positive")
    if cfg.diagnostic_steps <= 0:
        raise ValueError("diagnostic_steps must be positive")
    if cfg.diagnostic_world_count <= 0:
        raise ValueError("diagnostic_world_count must be positive")
    if cfg.max_battery_falls < 0:
        raise ValueError("max_battery_falls must be non-negative")


def _g1_gate_commands_array(commands: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    command_np = np.asarray(commands, dtype=np.float32)
    if command_np.ndim != 2 or command_np.shape[0] == 0 or command_np.shape[1] != 3:
        raise ValueError("battery_commands must have shape [command_count, 3]")
    if not np.isfinite(command_np).all():
        raise ValueError("battery_commands must be finite")
    return command_np


def _check_g1_trainer_dimensions(trainer: TrainerPPO, env: EnvG1PhoenX) -> None:
    if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
        raise ValueError("Trainer dimensions do not match the G1 environment")


def _evaluate_g1_gate_battery(
    trainer: TrainerPPO,
    env: EnvG1PhoenX,
    cfg: ConfigEvaluateG1GatePPO,
    commands: np.ndarray,
) -> tuple[tuple[StatsEvaluateG1GateCommandPPO, ...], int, float, int]:
    command_ids = np.repeat(np.arange(commands.shape[0], dtype=np.int32), int(cfg.seeds_per_command))
    command_np = commands[command_ids].astype(np.float32, copy=False)
    samples_per_step = np.bincount(command_ids, minlength=commands.shape[0]).astype(np.int64)
    falls = np.zeros(commands.shape[0], dtype=np.int64)
    perf_sum = np.zeros(commands.shape[0], dtype=np.float64)
    lin_err_sum = np.zeros(commands.shape[0], dtype=np.float64)
    yaw_err_sum = np.zeros(commands.shape[0], dtype=np.float64)
    sample_count = np.zeros(commands.shape[0], dtype=np.int64)

    root_com_local = env.model.body_com.numpy().reshape(env.world_count, env.body_stride, 3)[:, 0, :].copy()
    obs = env.reset_noisy(seed=int(cfg.seed))
    env.set_commands(command_np)
    obs = env.observe()
    trainer.reset_rollout_state()
    for step in range(int(cfg.battery_steps)):
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + step, deterministic=bool(cfg.deterministic)
        )
        obs, _rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        perf_np = env.step_successes.numpy().astype(np.float64)
        q = _joint_q_matrix_g1(env)
        qd = _joint_qd_matrix_g1(env)
        lin_b = _g1_root_origin_linear_velocity_body_np(q, qd, root_com_local)
        ang_b = _quat_rotate_inverse_xyzw_np(q[:, 3:7], qd[:, 3:6])
        lin_err = np.linalg.norm(command_np[:, 0:2] - lin_b[:, 0:2], axis=1)
        yaw_err = np.abs(command_np[:, 2] - ang_b[:, 2])

        falls += np.bincount(command_ids, weights=done_np.astype(np.float64), minlength=commands.shape[0]).astype(
            np.int64
        )
        perf_sum += np.bincount(command_ids, weights=perf_np, minlength=commands.shape[0])
        lin_err_sum += np.bincount(command_ids, weights=lin_err, minlength=commands.shape[0])
        yaw_err_sum += np.bincount(command_ids, weights=yaw_err, minlength=commands.shape[0])
        sample_count += samples_per_step
        if np.any(done_np):
            trainer.reset_rollout_state(dones)
            obs = _reset_g1_done_worlds(env)

    stats = []
    for command_index, command in enumerate(commands):
        denom = float(max(int(sample_count[command_index]), 1))
        stats.append(
            StatsEvaluateG1GateCommandPPO(
                command=(float(command[0]), float(command[1]), float(command[2])),
                falls=int(falls[command_index]),
                mean_tracking_perf=float(perf_sum[command_index] / denom),
                mean_linear_velocity_error=float(lin_err_sum[command_index] / denom),
                mean_yaw_rate_error=float(yaw_err_sum[command_index] / denom),
                samples=int(sample_count[command_index]),
            )
        )
    total_samples = int(np.sum(sample_count))
    battery_perf = float(np.sum(perf_sum) / float(max(total_samples, 1)))
    return tuple(stats), int(np.sum(falls)), battery_perf, total_samples


def _evaluate_g1_gate_diagnostics(
    trainer: TrainerPPO, env: EnvG1PhoenX, cfg: ConfigEvaluateG1GatePPO
) -> tuple[int, int, float, float, float, float]:
    obs = env.reset()
    trainer.reset_rollout_state()
    falls = 0
    valid_samples = 0
    jerk_sum = 0.0
    jerk_count = 0
    ang_vel_xy_sum = 0.0
    yaw_rate_sum = 0.0
    leg_qvel_sum = 0.0
    leg_qvel_count = 0
    previous_actions = np.zeros((env.world_count, 12), dtype=np.float32)
    has_previous = np.zeros(env.world_count, dtype=bool)

    for step in range(int(cfg.diagnostic_steps)):
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + 1_000_003 + step, deterministic=bool(cfg.deterministic)
        )
        obs, _rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        valid = ~done_np
        falls += int(np.sum(done_np))

        actions_np = env.current_actions.numpy()[:, :12]
        jerk_worlds = valid & has_previous
        if np.any(jerk_worlds):
            action_delta = actions_np[jerk_worlds] - previous_actions[jerk_worlds]
            jerk_sum += float(np.sum(action_delta * action_delta))
            jerk_count += int(action_delta.size)
        if np.any(valid):
            previous_actions[valid] = actions_np[valid]
            has_previous[valid] = True
        has_previous[done_np] = False

        q = _joint_q_matrix_g1(env)
        qd = _joint_qd_matrix_g1(env)
        if np.any(valid):
            qd_valid = qd[valid]
            ang_b_valid = _quat_rotate_inverse_xyzw_np(q[valid, 3:7], qd_valid[:, 3:6])
            ang_vel_xy_sum += float(
                np.sum(ang_b_valid[:, 0] * ang_b_valid[:, 0] + ang_b_valid[:, 1] * ang_b_valid[:, 1])
            )
            yaw_rate_sum += float(np.sum(ang_b_valid[:, 2] * ang_b_valid[:, 2]))
            leg_qvel = qd_valid[:, 6:18]
            leg_qvel_sum += float(np.sum(leg_qvel * leg_qvel))
            leg_qvel_count += int(leg_qvel.size)
            valid_samples += int(qd_valid.shape[0])
        if np.any(done_np):
            trainer.reset_rollout_state(dones)
            obs = _reset_g1_done_worlds(env)

    action_jerk = float(np.sqrt(jerk_sum / float(max(jerk_count, 1))))
    ang_vel_xy = float(np.sqrt(ang_vel_xy_sum / float(max(valid_samples, 1))))
    yaw_rate = float(np.sqrt(yaw_rate_sum / float(max(valid_samples, 1))))
    leg_qvel = float(np.sqrt(leg_qvel_sum / float(max(leg_qvel_count, 1))))
    diagnostic_samples = int(env.world_count) * int(cfg.diagnostic_steps)
    return falls, diagnostic_samples, action_jerk, ang_vel_xy, yaw_rate, leg_qvel


def _reset_g1_done_worlds(env: EnvG1PhoenX) -> wp.array:
    env.reset_done()
    env.dones.zero_()
    return env.observe()


def _g1_gate_passes(
    cfg: ConfigEvaluateG1GatePPO,
    *,
    battery_falls: int,
    battery_perf: float,
    action_jerk_rms: float,
    ang_vel_xy_rms: float,
    yaw_rate_rms: float,
    leg_qvel_rms: float,
) -> bool:
    return (
        int(battery_falls) <= int(cfg.max_battery_falls)
        and np.isfinite(battery_perf)
        and float(battery_perf) >= float(cfg.min_battery_perf)
        and np.isfinite(action_jerk_rms)
        and float(action_jerk_rms) <= float(cfg.max_action_jerk_rms)
        and np.isfinite(ang_vel_xy_rms)
        and float(ang_vel_xy_rms) <= float(cfg.max_ang_vel_xy_rms)
        and np.isfinite(yaw_rate_rms)
        and float(yaw_rate_rms) <= float(cfg.max_yaw_rate_rms)
        and np.isfinite(leg_qvel_rms)
        and float(leg_qvel_rms) <= float(cfg.max_leg_qvel_rms)
    )


def _joint_q_matrix_g1(env: EnvG1PhoenX) -> np.ndarray:
    return env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)


def _joint_qd_matrix_g1(env: EnvG1PhoenX) -> np.ndarray:
    return env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)


def _g1_root_origin_linear_velocity_body_np(q: np.ndarray, qd: np.ndarray, root_com_local: np.ndarray) -> np.ndarray:
    root_com_w = _quat_rotate_xyzw_np(q[:, 3:7], root_com_local)
    lin_origin_w = qd[:, 0:3] - np.cross(qd[:, 3:6], root_com_w)
    return _quat_rotate_inverse_xyzw_np(q[:, 3:7], lin_origin_w)


def _quat_rotate_inverse_xyzw_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw = q[:, 3:4]
    qv = q[:, 0:3]
    return v * (2.0 * qw * qw - 1.0) - np.cross(qv, v) * (2.0 * qw) + qv * (2.0 * np.sum(qv * v, axis=1, keepdims=True))


def _quat_rotate_xyzw_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw = q[:, 3:4]
    qv = q[:, 0:3]
    return v * (2.0 * qw * qw - 1.0) + np.cross(qv, v) * (2.0 * qw) + qv * (2.0 * np.sum(qv * v, axis=1, keepdims=True))


def _merge_g1_stats(
    iteration: int,
    rollout_metrics: tuple[float, float, float, float, float, float],
    update_stats: StatsPPOUpdate,
    action_metrics: tuple[float, float, float, float, float],
    rollout_seconds: float,
    update_seconds: float,
    num_samples: int,
) -> StatsTrainG1PPO:
    mean_reward, mean_done, mean_tracking_perf, mean_command_x, mean_command_y, mean_command_yaw = rollout_metrics
    raw_action_rms, clipped_action_rms, action_clip_fraction, clipped_action_abs_mean, mean_log_std = action_metrics
    elapsed = max(float(rollout_seconds) + float(update_seconds), 1.0e-12)
    return StatsTrainG1PPO(
        iteration=iteration,
        mean_reward=mean_reward,
        mean_done=mean_done,
        mean_tracking_perf=mean_tracking_perf,
        mean_command_x=mean_command_x,
        mean_command_y=mean_command_y,
        mean_command_yaw=mean_command_yaw,
        policy_loss=update_stats.policy_loss,
        value_loss=update_stats.value_loss,
        approx_kl=update_stats.approx_kl,
        clip_fraction=update_stats.clip_fraction,
        raw_action_rms=float(raw_action_rms),
        clipped_action_rms=float(clipped_action_rms),
        action_clip_fraction=float(action_clip_fraction),
        clipped_action_abs_mean=float(clipped_action_abs_mean),
        mean_log_std=float(mean_log_std),
        rollout_seconds=float(rollout_seconds),
        update_seconds=float(update_seconds),
        samples_per_second=float(num_samples) / elapsed,
    )


def _rollout_metrics(
    buffer: BufferRollout, command_x: float, target_distance: float, *, use_observed_command: bool = False
) -> tuple[float, float, float, float, float, float]:
    rewards = buffer.rewards.numpy()
    dones = buffer.dones.numpy()
    successes = buffer.successes.numpy()
    obs = buffer.obs.numpy()
    forward_velocity = obs[:, 0]
    target_forward_velocity = obs[:, 9] if use_observed_command else command_x
    return (
        float(np.mean(rewards)),
        float(np.mean(dones)),
        float(np.mean(successes)),
        float(target_distance),
        float(np.mean(forward_velocity)),
        float(np.mean(np.abs(forward_velocity - target_forward_velocity))),
    )


def _merge_stats(
    iteration: int,
    rollout_metrics: tuple[float, float, float, float, float, float],
    update_stats: StatsPPOUpdate,
) -> StatsTrainAnymalPPO:
    (
        mean_reward,
        mean_done,
        mean_success,
        target_distance,
        mean_forward_velocity,
        mean_abs_forward_velocity_error,
    ) = rollout_metrics
    return StatsTrainAnymalPPO(
        iteration=iteration,
        mean_reward=mean_reward,
        mean_done=mean_done,
        mean_success=mean_success,
        target_distance=target_distance,
        mean_forward_velocity=mean_forward_velocity,
        mean_abs_forward_velocity_error=mean_abs_forward_velocity_error,
        policy_loss=update_stats.policy_loss,
        value_loss=update_stats.value_loss,
        approx_kl=update_stats.approx_kl,
        clip_fraction=update_stats.clip_fraction,
    )
