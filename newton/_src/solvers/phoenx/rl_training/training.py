# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

from .anymal import ConfigEnvAnymalPhoenX, EnvAnymalPhoenX
from .g1 import ConfigEnvG1PhoenX, EnvG1PhoenX, g1_mirror_map_ppo
from .ppo import BufferRollout, ConfigPPO, StatsPPOUpdate, TrainerPPO, load_ppo_checkpoint


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
        resume_checkpoint: Optional PPO checkpoint to resume from.
        checkpoint_path: Optional path for writing PPO checkpoints.
        checkpoint_interval: Save a checkpoint every N iterations when positive.
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
    resume_checkpoint: str | None = None
    checkpoint_path: str | None = None
    checkpoint_interval: int = 0


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
    """Configuration for train_g1_ppo."""

    iterations: int = 100
    rollout_steps: int = 64
    hidden_layers: tuple[int, ...] = (128, 128, 128)
    activation: str = "relu"
    log_std_init: float = -0.5
    env_config: ConfigEnvG1PhoenX | None = None
    ppo_config: ConfigPPO | None = None
    device: wp.context.Devicelike = None
    seed: int = 42
    log_interval: int = 1
    randomize_commands: bool = True
    command_x_range: tuple[float, float] = (-0.5, 0.8)
    command_y_range: tuple[float, float] = (-0.4, 0.4)
    command_yaw_range: tuple[float, float] = (-1.0, 1.0)
    resume_checkpoint: str | None = None
    checkpoint_path: str | None = None
    checkpoint_interval: int = 0


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
    samples_per_second: float


@dataclass
class ResultEvaluateG1PPO:
    """Result returned by :func:`evaluate_g1_ppo`."""

    stats: StatsEvaluateG1PPO


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
    return ConfigPPO(
        gamma=0.97,
        gae_lambda=0.9,
        clip_ratio=0.2,
        entropy_coeff=1.0e-5,
        actor_lr=2.0e-3,
        critic_lr=2.0e-3,
        train_epochs=3,
        minibatch_size=32768,
        replay_ratio=3.0,
        priority_alpha=0.4,
        priority_beta=1.0,
        manual_actor_backward=True,
        manual_critic_backward=True,
        vtrace_rho_clip=3.0,
        vtrace_c_clip=3.0,
        normalize_advantages=True,
        reward_clip=1.0,
        max_grad_norm=0.3,
        mirror_loss_coeff=0.25,
    )


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
    if cfg.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")

    device = wp.get_device(cfg.device)
    env_config = cfg.env_config or ConfigEnvG1PhoenX()
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
            squash_actions=True,
            activation=cfg.activation,
            log_std_init=cfg.log_std_init,
            mirror_map=g1_mirror_map_ppo() if ppo_config.mirror_loss_coeff > 0.0 else None,
        )
    buffer = BufferRollout(
        num_steps=cfg.rollout_steps,
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )

    history: list[StatsTrainG1PPO] = []
    start_iteration = int(getattr(trainer, "iteration", 0))
    command_np = np.tile(np.asarray(env.config.command, dtype=np.float32), (env.world_count, 1))

    for local_iteration in range(cfg.iterations):
        iteration = start_iteration + local_iteration
        if cfg.randomize_commands:
            command_np = _sample_g1_commands(
                rng=np.random.default_rng(int(cfg.seed) + 53_321 + iteration),
                world_count=env.world_count,
                command_x_range=cfg.command_x_range,
                command_y_range=cfg.command_y_range,
                command_yaw_range=cfg.command_yaw_range,
            )
            env.set_commands(command_np)

        t0 = time.perf_counter()
        env.collect_ppo_rollout(trainer, buffer, seed=cfg.seed + iteration * cfg.rollout_steps)
        t1 = time.perf_counter()
        rollout_metrics = _g1_rollout_metrics(buffer, command_np)
        update_stats = trainer.update(buffer)
        t2 = time.perf_counter()
        stats = _merge_g1_stats(iteration, rollout_metrics, update_stats, t1 - t0, t2 - t1, buffer.num_samples)
        history.append(stats)

        if cfg.log_interval > 0 and (iteration % cfg.log_interval == 0 or local_iteration == cfg.iterations - 1):
            print(
                f"iter={iteration:04d} "
                f"reward={stats.mean_reward:.4f} "
                f"perf={stats.mean_tracking_perf:.3f} "
                f"done={stats.mean_done:.4f} "
                f"sps={stats.samples_per_second:.1f} "
                f"pi_loss={stats.policy_loss:.4f} "
                f"v_loss={stats.value_loss:.4f}"
            )
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
    reward_sum = 0.0
    done_sum = 0.0
    tracking_perf_sum = 0.0
    t0 = time.perf_counter()
    for step in range(int(cfg.steps)):
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + step, deterministic=bool(cfg.deterministic)
        )
        obs, rewards, dones = env.step(actions)
        reward_sum += float(np.mean(rewards.numpy()))
        done_sum += float(np.mean(dones.numpy()))
        tracking_perf_sum += float(np.mean(env.step_successes.numpy()))
    elapsed = max(time.perf_counter() - t0, 1.0e-12)
    command_np = env.command.numpy()
    steps = int(cfg.steps)
    stats = StatsEvaluateG1PPO(
        steps=steps,
        mean_reward=reward_sum / float(steps),
        mean_done=done_sum / float(steps),
        mean_tracking_perf=tracking_perf_sum / float(steps),
        mean_command_x=float(np.mean(command_np[:, 0])),
        mean_command_y=float(np.mean(command_np[:, 1])),
        mean_command_yaw=float(np.mean(command_np[:, 2])),
        samples_per_second=float(env.world_count * steps) / elapsed,
    )
    return ResultEvaluateG1PPO(stats=stats)


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
    if cfg.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")

    device = wp.get_device(cfg.device)
    env_config = cfg.env_config or ConfigEnvAnymalPhoenX()
    ppo_config = cfg.ppo_config or _default_ppo_config()
    env = EnvAnymalPhoenX(env_config, device=device)
    if cfg.resume_checkpoint is not None:
        trainer = load_ppo_checkpoint(cfg.resume_checkpoint, config=cfg.ppo_config, device=device)
        ppo_config = trainer.config
        if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
            raise ValueError("Checkpoint dimensions do not match the Anymal environment")
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
        )
    buffer = BufferRollout(
        num_steps=cfg.rollout_steps,
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )

    history: list[StatsTrainAnymalPPO] = []
    start_iteration = int(getattr(trainer, "iteration", 0))
    command_x = float(env.config.command[0])
    target_xy = np.asarray(env.config.target_position, dtype=np.float32)
    target_distance = float(np.linalg.norm(target_xy))
    target_direction = _target_direction(target_xy)
    if cfg.use_target_curriculum and env.config.reward_mode == "sparse_target":
        target_distance = float(cfg.target_distance_start)

    for local_iteration in range(cfg.iterations):
        iteration = start_iteration + local_iteration
        target_rng = np.random.default_rng(int(cfg.seed) + 17_917 + iteration)
        _configure_rollout_targets(env, cfg, target_rng, target_direction, target_distance)
        env.collect_ppo_rollout(trainer, buffer, seed=cfg.seed + iteration * cfg.rollout_steps)
        rollout_metrics = _rollout_metrics(buffer, command_x, target_distance)
        update_stats = trainer.update(buffer)
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

    env.set_commands(command_np)
    obs = env.reset_noisy(seed=int(cfg.seed))
    for step in range(int(cfg.battery_steps)):
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + step, deterministic=bool(cfg.deterministic)
        )
        obs, _rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        perf_np = env.step_successes.numpy().astype(np.float64)
        q = _joint_q_matrix_g1(env)
        qd = _joint_qd_matrix_g1(env)
        lin_b = _quat_rotate_inverse_wxyz_np(q[:, 3:7], qd[:, 0:3])
        lin_err = np.linalg.norm(command_np[:, 0:2] - lin_b[:, 0:2], axis=1)
        yaw_err = np.abs(command_np[:, 2] - qd[:, 5])

        falls += np.bincount(command_ids, weights=done_np.astype(np.float64), minlength=commands.shape[0]).astype(
            np.int64
        )
        perf_sum += np.bincount(command_ids, weights=perf_np, minlength=commands.shape[0])
        lin_err_sum += np.bincount(command_ids, weights=lin_err, minlength=commands.shape[0])
        yaw_err_sum += np.bincount(command_ids, weights=yaw_err, minlength=commands.shape[0])
        sample_count += samples_per_step
        if np.any(done_np):
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

        qd = _joint_qd_matrix_g1(env)
        if np.any(valid):
            qd_valid = qd[valid]
            ang_vel_xy_sum += float(np.sum(qd_valid[:, 3] * qd_valid[:, 3] + qd_valid[:, 4] * qd_valid[:, 4]))
            yaw_rate_sum += float(np.sum(qd_valid[:, 5] * qd_valid[:, 5]))
            leg_qvel = qd_valid[:, 6:18]
            leg_qvel_sum += float(np.sum(leg_qvel * leg_qvel))
            leg_qvel_count += int(leg_qvel.size)
            valid_samples += int(qd_valid.shape[0])
        if np.any(done_np):
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


def _quat_rotate_inverse_wxyz_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw = q[:, 0:1]
    qv = q[:, 1:4]
    return v * (2.0 * qw * qw - 1.0) - np.cross(qv, v) * (2.0 * qw) + qv * (2.0 * np.sum(qv * v, axis=1, keepdims=True))


def _sample_g1_commands(
    *,
    rng: np.random.Generator,
    world_count: int,
    command_x_range: tuple[float, float],
    command_y_range: tuple[float, float],
    command_yaw_range: tuple[float, float],
) -> np.ndarray:
    commands = np.empty((int(world_count), 3), dtype=np.float32)
    commands[:, 0] = rng.uniform(float(command_x_range[0]), float(command_x_range[1]), int(world_count))
    commands[:, 1] = rng.uniform(float(command_y_range[0]), float(command_y_range[1]), int(world_count))
    commands[:, 2] = rng.uniform(float(command_yaw_range[0]), float(command_yaw_range[1]), int(world_count))
    return commands


def _g1_rollout_metrics(buffer: BufferRollout, commands: np.ndarray) -> tuple[float, float, float, float, float, float]:
    rewards = buffer.rewards.numpy()
    dones = buffer.dones.numpy()
    tracking_perf = buffer.successes.numpy()
    return (
        float(np.mean(rewards)),
        float(np.mean(dones)),
        float(np.mean(tracking_perf)),
        float(np.mean(commands[:, 0])),
        float(np.mean(commands[:, 1])),
        float(np.mean(commands[:, 2])),
    )


def _merge_g1_stats(
    iteration: int,
    rollout_metrics: tuple[float, float, float, float, float, float],
    update_stats: StatsPPOUpdate,
    rollout_seconds: float,
    update_seconds: float,
    num_samples: int,
) -> StatsTrainG1PPO:
    mean_reward, mean_done, mean_tracking_perf, mean_command_x, mean_command_y, mean_command_yaw = rollout_metrics
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
        rollout_seconds=float(rollout_seconds),
        update_seconds=float(update_seconds),
        samples_per_second=float(num_samples) / elapsed,
    )


def _rollout_metrics(
    buffer: BufferRollout, command_x: float, target_distance: float
) -> tuple[float, float, float, float, float, float]:
    rewards = buffer.rewards.numpy()
    dones = buffer.dones.numpy()
    successes = buffer.successes.numpy()
    obs = buffer.obs.numpy()
    forward_velocity = obs[:, 0]
    return (
        float(np.mean(rewards)),
        float(np.mean(dones)),
        float(np.mean(successes)),
        float(target_distance),
        float(np.mean(forward_velocity)),
        float(np.mean(np.abs(forward_velocity - command_x))),
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
