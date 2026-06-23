# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train Anymal C command walking with PhoenX and Warp-only PPO.

This experimental runner validates PhoenX RL on a quadruped that is simpler
than G1 while still using the production Anymal PhoenX environment. It supports
both a single fixed velocity command and phased curricula for straight walking
and ASDWQE-style steering. Curriculum phases train from scratch, checkpoint
between phases, and evaluate without auto-reset so falls cannot be hidden by
rollout resets.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import newton.rl as rl

_BASE_ENV = {
    "reward_mode": "dense_command",
    "command": (0.6, 0.0, 0.0),
    "max_episode_steps": 500,
    "lin_vel_reward_scale": 1.0,
    "yaw_rate_reward_scale": 0.5,
    "z_vel_reward_scale": -2.0,
    "ang_vel_reward_scale": -0.05,
    "action_rate_reward_scale": -0.01,
    "joint_speed_reward_scale": -1.0e-4,
    "flat_orientation_reward_scale": -5.0,
    "forward_progress_reward_scale": 0.0,
    "fall_reward_scale": -2.0,
    "energy_reward_scale": -2.5e-5,
    "min_base_height": 0.30,
    "min_upright_cos": 0.35,
    "disturbance_warmup_steps": 0,
    "disturbance_noise_velocity_xy": 0.0,
    "disturbance_noise_yaw_velocity": 0.0,
    "disturbance_kick_probability": 0.0,
    "disturbance_kick_velocity_xy": 0.0,
    "disturbance_kick_yaw_velocity": 0.0,
    "disturbance_seed": 0,
}


_STEERING_EVAL_COMMANDS = (
    (0.0, 0.0, 0.0),
    (0.75, 0.0, 0.0),
    (-0.35, 0.0, 0.0),
    (0.0, 0.35, 0.0),
    (0.0, -0.35, 0.0),
    (0.0, 0.0, 0.65),
    (0.0, 0.0, -0.65),
    (0.55, 0.0, 0.65),
    (0.55, 0.0, -0.65),
)


@dataclass(frozen=True)
class PhaseAnymalWalk:
    """One Anymal command-walking training phase."""

    name: str
    command: tuple[float, float, float]
    iterations: int
    env_overrides: tuple[tuple[str, object], ...] = ()
    randomize_commands: bool = False
    command_x_range: tuple[float, float] = (0.0, 0.0)
    command_y_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_min_abs: float = 0.0
    command_zero_probability: float = 0.0
    eval_commands: tuple[tuple[float, float, float], ...] = ()
    gate_min_tracking_perf: float = 0.45
    gate_max_fall_fraction: float = 0.35
    gate_min_survival_fraction: float = 0.70
    gate_min_forward_velocity_fraction: float = 0.35
    gate_max_abs_forward_velocity_error: float = 0.45
    gate_max_abs_lateral_velocity_error: float = 0.45
    gate_max_abs_yaw_rate_error: float = 0.65


@dataclass(frozen=True)
class StatsEvaluateAnymalWalk:
    """No-reset command-walking evaluation metrics."""

    steps: int
    command: tuple[float, float, float]
    mean_reward: float
    mean_done: float
    fall_fraction: float
    mean_survival_steps: float
    survival_fraction: float
    mean_tracking_perf: float
    mean_velocity_tracking_perf: float
    mean_yaw_tracking_perf: float
    mean_forward_velocity: float
    mean_abs_forward_velocity_error: float
    mean_abs_lateral_velocity: float
    mean_abs_yaw_rate_error: float
    mean_displacement_forward: float
    mean_displacement_x: float
    mean_displacement_y: float
    mean_path_length: float
    mean_action_rms: float
    samples_per_second: float


def anymal_recipe(name: str) -> tuple[PhaseAnymalWalk, ...]:
    """Return a named Anymal walking curriculum."""

    if name == "single":
        return ()

    forward = (
        PhaseAnymalWalk(
            name="warmup_forward",
            command=(0.35, 0.0, 0.0),
            iterations=120,
            env_overrides=(("action_scale", 0.45), ("forward_progress_reward_scale", 1.0)),
            gate_min_tracking_perf=0.30,
            gate_max_fall_fraction=0.55,
            gate_min_survival_fraction=0.45,
            gate_min_forward_velocity_fraction=0.20,
            gate_max_abs_forward_velocity_error=0.50,
        ),
        PhaseAnymalWalk(
            name="walk_forward",
            command=(0.65, 0.0, 0.0),
            iterations=220,
            env_overrides=(("action_scale", 0.50), ("forward_progress_reward_scale", 0.75)),
            gate_min_tracking_perf=0.48,
            gate_max_fall_fraction=0.30,
            gate_min_survival_fraction=0.70,
            gate_min_forward_velocity_fraction=0.45,
            gate_max_abs_forward_velocity_error=0.40,
        ),
        PhaseAnymalWalk(
            name="fast_efficient_forward",
            command=(0.90, 0.0, 0.0),
            iterations=260,
            env_overrides=(
                ("action_scale", 0.50),
                ("energy_reward_scale", -3.0e-5),
                ("action_rate_reward_scale", -0.015),
                ("forward_progress_reward_scale", 0.35),
            ),
            gate_min_tracking_perf=0.58,
            gate_max_fall_fraction=0.18,
            gate_min_survival_fraction=0.82,
            gate_min_forward_velocity_fraction=0.55,
            gate_max_abs_forward_velocity_error=0.35,
        ),
        PhaseAnymalWalk(
            name="disturbed_forward",
            command=(0.90, 0.0, 0.0),
            iterations=180,
            env_overrides=(
                ("action_scale", 0.50),
                ("energy_reward_scale", -3.0e-5),
                ("action_rate_reward_scale", -0.015),
                ("forward_progress_reward_scale", 0.25),
                ("disturbance_warmup_steps", 50),
                ("disturbance_noise_velocity_xy", 0.025),
                ("disturbance_noise_yaw_velocity", 0.015),
                ("disturbance_kick_probability", 0.003),
                ("disturbance_kick_velocity_xy", 0.45),
                ("disturbance_kick_yaw_velocity", 0.35),
                ("disturbance_seed", 41_337),
            ),
            gate_min_tracking_perf=0.50,
            gate_max_fall_fraction=0.08,
            gate_min_survival_fraction=0.92,
            gate_min_forward_velocity_fraction=0.50,
            gate_max_abs_forward_velocity_error=0.45,
        ),
    )
    if name == "forward":
        return forward
    if name == "steering":
        return (
            *forward,
            PhaseAnymalWalk(
                name="turn_stand",
                command=(0.0, 0.0, 0.65),
                iterations=220,
                env_overrides=(
                    ("action_scale", 0.55),
                    ("lin_vel_reward_scale", 0.35),
                    ("yaw_rate_reward_scale", 2.50),
                    ("forward_progress_reward_scale", 0.0),
                    ("energy_reward_scale", -3.0e-5),
                    ("action_rate_reward_scale", -0.015),
                ),
                randomize_commands=True,
                command_x_range=(0.0, 0.0),
                command_y_range=(0.0, 0.0),
                command_yaw_range=(-0.85, 0.85),
                command_yaw_min_abs=0.45,
                command_zero_probability=0.10,
                eval_commands=((0.0, 0.0, 0.0), (0.0, 0.0, 0.65), (0.0, 0.0, -0.65)),
                gate_min_tracking_perf=0.35,
                gate_max_fall_fraction=0.12,
                gate_min_survival_fraction=0.88,
                gate_max_abs_forward_velocity_error=0.30,
                gate_max_abs_lateral_velocity_error=0.30,
                gate_max_abs_yaw_rate_error=0.48,
            ),
            PhaseAnymalWalk(
                name="omni_steering",
                command=(0.55, 0.0, 0.0),
                iterations=320,
                env_overrides=(
                    ("action_scale", 0.50),
                    ("lin_vel_reward_scale", 2.50),
                    ("yaw_rate_reward_scale", 1.25),
                    ("forward_progress_reward_scale", 0.20),
                    ("energy_reward_scale", -3.0e-5),
                    ("action_rate_reward_scale", -0.015),
                ),
                randomize_commands=True,
                command_x_range=(-0.40, 0.95),
                command_y_range=(-0.45, 0.45),
                command_yaw_range=(-0.90, 0.90),
                command_yaw_min_abs=0.0,
                command_zero_probability=0.05,
                eval_commands=_STEERING_EVAL_COMMANDS,
                gate_min_tracking_perf=0.42,
                gate_max_fall_fraction=0.15,
                gate_min_survival_fraction=0.85,
                gate_min_forward_velocity_fraction=0.35,
                gate_max_abs_forward_velocity_error=0.40,
                gate_max_abs_lateral_velocity_error=0.32,
                gate_max_abs_yaw_rate_error=0.55,
            ),
            PhaseAnymalWalk(
                name="disturbed_steering",
                command=(0.55, 0.0, 0.0),
                iterations=260,
                env_overrides=(
                    ("action_scale", 0.50),
                    ("lin_vel_reward_scale", 2.50),
                    ("yaw_rate_reward_scale", 1.25),
                    ("forward_progress_reward_scale", 0.20),
                    ("energy_reward_scale", -3.0e-5),
                    ("action_rate_reward_scale", -0.015),
                    ("disturbance_warmup_steps", 50),
                    ("disturbance_noise_velocity_xy", 0.025),
                    ("disturbance_noise_yaw_velocity", 0.015),
                    ("disturbance_kick_probability", 0.003),
                    ("disturbance_kick_velocity_xy", 0.45),
                    ("disturbance_kick_yaw_velocity", 0.35),
                    ("disturbance_seed", 52_091),
                ),
                randomize_commands=True,
                command_x_range=(-0.40, 0.95),
                command_y_range=(-0.45, 0.45),
                command_yaw_range=(-0.90, 0.90),
                command_yaw_min_abs=0.0,
                command_zero_probability=0.10,
                eval_commands=_STEERING_EVAL_COMMANDS,
                gate_min_tracking_perf=0.38,
                gate_max_fall_fraction=0.12,
                gate_min_survival_fraction=0.88,
                gate_min_forward_velocity_fraction=0.30,
                gate_max_abs_forward_velocity_error=0.45,
                gate_max_abs_lateral_velocity_error=0.36,
                gate_max_abs_yaw_rate_error=0.60,
            ),
        )
    raise ValueError(f"Unknown Anymal recipe {name!r}")


def build_env_config(
    args: argparse.Namespace,
    *,
    world_count: int | None = None,
    auto_reset: bool = True,
    command: tuple[float, float, float] | None = None,
    env_overrides: dict[str, object] | None = None,
) -> rl.ConfigEnvAnymalPhoenX:
    values = dict(_BASE_ENV)
    values.update(
        world_count=int(args.world_count if world_count is None else world_count),
        frame_dt=float(args.frame_dt),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        action_scale=float(args.action_scale),
        command=tuple(float(v) for v in (command or (args.command_x, args.command_y, args.command_yaw))),
        max_episode_steps=int(args.max_episode_steps),
        target_base_height=float(args.target_base_height),
        actuator_ke=float(args.actuator_ke),
        actuator_kd=float(args.actuator_kd),
        disturbance_warmup_steps=int(args.disturbance_warmup_steps),
        disturbance_noise_velocity_xy=float(args.disturbance_noise_velocity_xy),
        disturbance_noise_yaw_velocity=float(args.disturbance_noise_yaw_velocity),
        disturbance_kick_probability=float(args.disturbance_kick_probability),
        disturbance_kick_velocity_xy=float(args.disturbance_kick_velocity_xy),
        disturbance_kick_yaw_velocity=float(args.disturbance_kick_yaw_velocity),
        disturbance_seed=int(args.disturbance_seed),
        auto_reset=bool(auto_reset),
    )
    if env_overrides:
        values.update(env_overrides)
    return rl.ConfigEnvAnymalPhoenX(**values)


def build_ppo_config(args: argparse.Namespace) -> rl.ConfigPPO:
    return rl.ConfigPPO(
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_ratio=float(args.clip_ratio),
        entropy_coeff=float(args.entropy_coeff),
        value_loss_coeff=float(args.value_loss_coeff),
        value_clip_range=float(args.value_clip_range),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        train_epochs=int(args.train_epochs),
        minibatch_size=int(args.minibatch_size),
        replay_ratio=float(args.replay_ratio),
        normalize_advantages=True,
        reward_clip=float(args.reward_clip),
        max_grad_norm=float(args.max_grad_norm),
        mirror_loss_coeff=float(args.mirror_loss_coeff),
        manual_actor_backward=not bool(args.disable_manual_backward),
        manual_critic_backward=not bool(args.disable_manual_backward),
    )


def _phase_checkpoint_pattern(pattern: str, phase: PhaseAnymalWalk | None, phase_index: int | None) -> str:
    text = str(pattern)
    if phase is None or phase_index is None:
        return text
    text = text.replace("{phase_index:02d}", f"{int(phase_index):02d}")
    text = text.replace("{phase_index}", str(int(phase_index)))
    return text.replace("{phase}", phase.name)


def _format_checkpoint_path(
    pattern: str | None,
    iteration: int,
    *,
    phase: PhaseAnymalWalk | None = None,
    phase_index: int | None = None,
) -> str | None:
    if pattern is None:
        return None
    return _phase_checkpoint_pattern(str(pattern), phase, phase_index).format(iteration=int(iteration))


def _default_checkpoint_pattern(args: argparse.Namespace, *, curriculum: bool) -> str | None:
    if args.checkpoint_path is not None:
        return str(args.checkpoint_path)
    if args.output_dir is None:
        return None
    name = "checkpoint_{phase_index:02d}_{phase}_{iteration}.npz" if curriculum else "checkpoint_{iteration}.npz"
    return str(Path(args.output_dir) / name)


def _summary_path(args: argparse.Namespace) -> Path | None:
    if args.summary_path is not None:
        return Path(args.summary_path)
    if args.output_dir is not None:
        return Path(args.output_dir) / "summary.json"
    return None


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_vec = q[..., :3]
    qw = q[..., 3:4]
    return (
        v * (2.0 * qw * qw - 1.0)
        + np.cross(q_vec, v) * (2.0 * qw)
        + q_vec * (2.0 * np.sum(q_vec * v, axis=-1, keepdims=True))
    )


def evaluate_checkpoint(
    trainer: rl.TrainerPPO,
    args: argparse.Namespace,
    *,
    command: tuple[float, float, float] | None = None,
    env_overrides: dict[str, object] | None = None,
) -> StatsEvaluateAnymalWalk:
    eval_command = tuple(float(v) for v in (command or (args.command_x, args.command_y, args.command_yaw)))
    eval_overrides = {"max_episode_steps": 0}
    if env_overrides:
        eval_overrides.update(env_overrides)
    env = rl.EnvAnymalPhoenX(
        build_env_config(
            args,
            world_count=int(args.eval_world_count),
            auto_reset=False,
            command=eval_command,
            env_overrides=eval_overrides,
        ),
        device=args.device,
    )
    obs = env.reset()
    trainer.reset_rollout_state()
    q0 = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
    start_xy = q0[:, 0:2].copy()
    previous_xy = start_xy.copy()
    last_alive_xy = start_xy.copy()
    forward_w = _quat_rotate_xyzw(
        q0[:, 3:7], np.tile(np.asarray((1.0, 0.0, 0.0), dtype=np.float32), (env.world_count, 1))
    )
    first_done_step = np.full(env.world_count, -1, dtype=np.int32)
    reward_sum = 0.0
    done_sum = 0.0
    tracking_sum = 0.0
    velocity_tracking_sum = 0.0
    yaw_tracking_sum = 0.0
    forward_error_sum = 0.0
    lateral_abs_sum = 0.0
    yaw_error_sum = 0.0
    forward_sum = 0.0
    alive_count = 0
    action_sq_sum = 0.0
    action_count = 0
    path_length = np.zeros(env.world_count, dtype=np.float64)
    command_np = np.asarray(eval_command, dtype=np.float32)
    t0 = time.perf_counter()
    for step in range(int(args.eval_steps)):
        alive_before = first_done_step < 0
        actions, _log_probs, _values = trainer.act(obs, seed=int(args.seed) + 90_000 + step, deterministic=True)
        obs, rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        reward_sum += float(np.mean(rewards.numpy()))
        done_sum += float(np.mean(done_np))
        obs_np = obs.numpy()
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        xy = q[:, 0:2].copy()
        if np.any(alive_before):
            alive_idx = alive_before
            path_length[alive_idx] += np.linalg.norm(xy[alive_idx] - previous_xy[alive_idx], axis=1)
            last_alive_xy[alive_idx] = xy[alive_idx]
            lin_alive = obs_np[alive_idx, 0:2]
            yaw_alive = obs_np[alive_idx, 5]
            vel_err = lin_alive - command_np[None, 0:2]
            yaw_err = yaw_alive - float(command_np[2])
            vel_perf = np.exp(-np.sum(vel_err * vel_err, axis=1) / 0.25)
            yaw_perf = np.exp(-(yaw_err * yaw_err) / 0.25)
            command_speed_sq = float(command_np[0] * command_np[0] + command_np[1] * command_np[1])
            if command_speed_sq > 1.0e-6:
                speed_quality = np.clip(np.sum(lin_alive * command_np[None, 0:2], axis=1) / command_speed_sq, 0.0, 1.0)
            else:
                speed_quality = np.ones(lin_alive.shape[0], dtype=np.float32)
            tracking_sum += float(np.sum(vel_perf * yaw_perf * speed_quality))
            velocity_tracking_sum += float(np.sum(vel_perf))
            yaw_tracking_sum += float(np.sum(yaw_perf))
            forward_sum += float(np.sum(lin_alive[:, 0]))
            forward_error_sum += float(np.sum(np.abs(lin_alive[:, 0] - float(command_np[0]))))
            lateral_abs_sum += float(np.sum(np.abs(lin_alive[:, 1] - float(command_np[1]))))
            yaw_error_sum += float(np.sum(np.abs(yaw_err)))
            action_alive = obs_np[alive_idx, 36:48]
            action_sq_sum += float(np.sum(action_alive * action_alive))
            action_count += int(action_alive.size)
            alive_count += int(np.sum(alive_idx))
        previous_xy = xy
        first_done_step[(first_done_step < 0) & done_np] = step + 1
    elapsed = max(time.perf_counter() - t0, 1.0e-12)
    survival_steps = np.where(first_done_step >= 0, first_done_step, int(args.eval_steps))
    alive_den = float(max(alive_count, 1))
    displacement = last_alive_xy - start_xy
    displacement_forward = np.sum(displacement * forward_w[:, :2], axis=1)
    return StatsEvaluateAnymalWalk(
        steps=int(args.eval_steps),
        command=eval_command,
        mean_reward=reward_sum / float(args.eval_steps),
        mean_done=done_sum / float(args.eval_steps),
        fall_fraction=float(np.mean(first_done_step >= 0)),
        mean_survival_steps=float(np.mean(survival_steps)),
        survival_fraction=float(np.mean(survival_steps)) / float(max(int(args.eval_steps), 1)),
        mean_tracking_perf=tracking_sum / alive_den,
        mean_velocity_tracking_perf=velocity_tracking_sum / alive_den,
        mean_yaw_tracking_perf=yaw_tracking_sum / alive_den,
        mean_forward_velocity=forward_sum / alive_den,
        mean_abs_forward_velocity_error=forward_error_sum / alive_den,
        mean_abs_lateral_velocity=lateral_abs_sum / alive_den,
        mean_abs_yaw_rate_error=yaw_error_sum / alive_den,
        mean_displacement_forward=float(np.mean(displacement_forward)),
        mean_displacement_x=float(np.mean(displacement[:, 0])),
        mean_displacement_y=float(np.mean(displacement[:, 1])),
        mean_path_length=float(np.mean(path_length)),
        mean_action_rms=float(math.sqrt(action_sq_sum / float(max(action_count, 1)))),
        samples_per_second=float(env.world_count * int(args.eval_steps)) / elapsed,
    )


def check_phase_gate(stats: StatsEvaluateAnymalWalk, phase: PhaseAnymalWalk) -> list[str]:
    failures: list[str] = []
    if stats.mean_tracking_perf < phase.gate_min_tracking_perf:
        failures.append(f"tracking={stats.mean_tracking_perf:.3f} < {phase.gate_min_tracking_perf:.3f}")
    if stats.fall_fraction > phase.gate_max_fall_fraction:
        failures.append(f"fall_fraction={stats.fall_fraction:.3f} > {phase.gate_max_fall_fraction:.3f}")
    if stats.survival_fraction < phase.gate_min_survival_fraction:
        failures.append(f"survival={stats.survival_fraction:.3f} < {phase.gate_min_survival_fraction:.3f}")
    command_x = float(stats.command[0])
    min_vx = abs(command_x) * float(phase.gate_min_forward_velocity_fraction)
    if abs(command_x) > 1.0e-6:
        if command_x > 0.0 and stats.mean_forward_velocity < min_vx:
            failures.append(f"vx={stats.mean_forward_velocity:.3f} < {min_vx:.3f}")
        if command_x < 0.0 and stats.mean_forward_velocity > -min_vx:
            failures.append(f"vx={stats.mean_forward_velocity:.3f} > {-min_vx:.3f}")
    if stats.mean_abs_forward_velocity_error > phase.gate_max_abs_forward_velocity_error:
        failures.append(
            f"|vx-cmd|={stats.mean_abs_forward_velocity_error:.3f} > {phase.gate_max_abs_forward_velocity_error:.3f}"
        )
    if stats.mean_abs_lateral_velocity > phase.gate_max_abs_lateral_velocity_error:
        failures.append(
            f"|vy-cmd|={stats.mean_abs_lateral_velocity:.3f} > {phase.gate_max_abs_lateral_velocity_error:.3f}"
        )
    if stats.mean_abs_yaw_rate_error > phase.gate_max_abs_yaw_rate_error:
        failures.append(f"|yaw-cmd|={stats.mean_abs_yaw_rate_error:.3f} > {phase.gate_max_abs_yaw_rate_error:.3f}")
    return failures


def _phase_eval_commands(phase: PhaseAnymalWalk) -> tuple[tuple[float, float, float], ...]:
    if phase.eval_commands:
        return phase.eval_commands
    return (phase.command,)


def evaluate_phase_commands(
    trainer: rl.TrainerPPO,
    args: argparse.Namespace,
    phase: PhaseAnymalWalk,
    env_overrides: dict[str, object],
) -> list[StatsEvaluateAnymalWalk]:
    return [
        evaluate_checkpoint(trainer, args, command=command, env_overrides=env_overrides)
        for command in _phase_eval_commands(phase)
    ]


def check_phase_gates(stats: list[StatsEvaluateAnymalWalk], phase: PhaseAnymalWalk) -> list[str]:
    failures: list[str] = []
    for item in stats:
        prefix = f"cmd=({item.command[0]:.2f},{item.command[1]:.2f},{item.command[2]:.2f})"
        failures.extend(f"{prefix} {failure}" for failure in check_phase_gate(item, phase))
    return failures


def _write_summary(args: argparse.Namespace, payload: dict[str, object]) -> None:
    path = _summary_path(args)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_single(args: argparse.Namespace) -> dict[str, object]:
    env_config = build_env_config(args)
    ppo_config = build_ppo_config(args)
    if bool(args.eval_only):
        if args.resume_checkpoint is None:
            raise ValueError("--eval-only requires --resume-checkpoint")
        trainer = rl.load_ppo_checkpoint(args.resume_checkpoint, config=ppo_config, device=args.device)
        eval_stats = evaluate_checkpoint(trainer, args)
        result = {"final_checkpoint": args.resume_checkpoint, "final_train_stats": {}, "eval_stats": asdict(eval_stats)}
        _write_summary(args, result)
        print(json.dumps(result, sort_keys=True))
        return result

    checkpoint_path = _default_checkpoint_pattern(args, curriculum=False)
    result = rl.train_anymal_ppo(
        rl.ConfigTrainAnymalPPO(
            iterations=int(args.iterations),
            rollout_steps=int(args.rollout_steps),
            hidden_layers=tuple(int(v) for v in args.hidden_layers),
            activation=str(args.activation),
            log_std_init=float(args.log_std_init),
            env_config=env_config,
            ppo_config=ppo_config,
            device=args.device,
            seed=int(args.seed),
            log_interval=int(args.log_interval),
            use_target_curriculum=False,
            randomize_target_positions=False,
            resume_checkpoint=args.resume_checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=int(args.checkpoint_interval),
        )
    )
    final_iteration = int(result.trainer.iteration)
    final_checkpoint = _format_checkpoint_path(checkpoint_path, final_iteration)
    eval_stats = None if bool(args.no_eval) else evaluate_checkpoint(result.trainer, args)
    final_stats = asdict(result.history[-1]) if result.history else {}
    payload = {
        "recipe": "single",
        "final_checkpoint": final_checkpoint,
        "final_train_stats": final_stats,
        "eval_stats": asdict(eval_stats) if eval_stats is not None else None,
    }
    _write_summary(args, payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def train_curriculum(args: argparse.Namespace) -> dict[str, object]:
    phases = anymal_recipe(str(args.recipe))
    selected = phases[int(args.start_phase) :]
    if args.phase_count is not None:
        selected = selected[: int(args.phase_count)]
    if not selected:
        raise ValueError("Selected curriculum is empty")
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_pattern = _default_checkpoint_pattern(args, curriculum=True)
    if checkpoint_pattern is None:
        raise ValueError("Curriculum recipes require --output-dir or --checkpoint-path so phases can resume")
    resume_checkpoint = args.resume_checkpoint
    phase_payloads: list[dict[str, object]] = []
    final_checkpoint = resume_checkpoint
    for local_index, phase in enumerate(selected, start=int(args.start_phase)):
        phase_iterations = max(1, int(round(float(phase.iterations) * float(args.iteration_scale))))
        env_overrides = dict(phase.env_overrides)
        phase_checkpoint_pattern = None
        if checkpoint_pattern is not None:
            phase_checkpoint_pattern = _phase_checkpoint_pattern(checkpoint_pattern, phase, local_index)
        print(
            f"phase={local_index}:{phase.name} command={phase.command} iterations={phase_iterations} "
            f"resume={resume_checkpoint or '-'}"
        )
        result = rl.train_anymal_ppo(
            rl.ConfigTrainAnymalPPO(
                iterations=phase_iterations,
                rollout_steps=int(args.rollout_steps),
                hidden_layers=tuple(int(v) for v in args.hidden_layers),
                activation=str(args.activation),
                log_std_init=float(args.log_std_init),
                env_config=build_env_config(args, command=phase.command, env_overrides=env_overrides),
                ppo_config=build_ppo_config(args),
                device=args.device,
                seed=int(args.seed) + local_index * 10_003,
                log_interval=int(args.log_interval),
                use_target_curriculum=False,
                randomize_target_positions=False,
                randomize_commands=bool(phase.randomize_commands),
                command_x_range=tuple(float(v) for v in phase.command_x_range),
                command_y_range=tuple(float(v) for v in phase.command_y_range),
                command_yaw_range=tuple(float(v) for v in phase.command_yaw_range),
                command_yaw_min_abs=float(phase.command_yaw_min_abs),
                command_zero_probability=float(phase.command_zero_probability),
                resume_checkpoint=resume_checkpoint,
                checkpoint_path=phase_checkpoint_pattern,
                checkpoint_interval=int(args.checkpoint_interval),
            )
        )
        final_iteration = int(result.trainer.iteration)
        final_checkpoint = _format_checkpoint_path(phase_checkpoint_pattern, final_iteration)
        eval_stats = None if bool(args.no_eval) else evaluate_phase_commands(result.trainer, args, phase, env_overrides)
        gate_failures = [] if eval_stats is None else check_phase_gates(eval_stats, phase)
        pass_gate = not gate_failures
        phase_payload = {
            "phase_index": local_index,
            "phase": asdict(phase),
            "iterations": phase_iterations,
            "checkpoint": final_checkpoint,
            "final_train_stats": asdict(result.history[-1]) if result.history else {},
            "eval_stats": asdict(eval_stats[0]) if eval_stats else None,
            "eval_command_stats": [asdict(item) for item in eval_stats] if eval_stats else None,
            "pass_gate": pass_gate,
            "gate_failures": gate_failures,
        }
        phase_payloads.append(phase_payload)
        payload = {"recipe": args.recipe, "final_checkpoint": final_checkpoint, "phases": phase_payloads}
        _write_summary(args, payload)
        print(json.dumps(phase_payload, sort_keys=True))
        if gate_failures and not bool(args.allow_gate_failure):
            raise RuntimeError(f"Anymal phase {phase.name!r} failed gate: {', '.join(gate_failures)}")
        resume_checkpoint = final_checkpoint
    payload = {"recipe": args.recipe, "final_checkpoint": final_checkpoint, "phases": phase_payloads}
    _write_summary(args, payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def train(args: argparse.Namespace) -> dict[str, object]:
    if str(args.recipe) == "single":
        return train_single(args)
    if bool(args.eval_only):
        return train_single(args)
    return train_curriculum(args)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=12_321)
    parser.add_argument("--world-count", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--recipe", choices=("single", "forward", "steering"), default="single")
    parser.add_argument("--iteration-scale", type=float, default=1.0)
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument("--allow-gate-failure", action="store_true")

    parser.add_argument("--command-x", type=float, default=0.6)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--frame-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--target-base-height", type=float, default=0.62)
    parser.add_argument("--actuator-ke", type=float, default=150.0)
    parser.add_argument("--actuator-kd", type=float, default=5.0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--disturbance-warmup-steps", type=int, default=0)
    parser.add_argument("--disturbance-noise-velocity-xy", type=float, default=0.0)
    parser.add_argument("--disturbance-noise-yaw-velocity", type=float, default=0.0)
    parser.add_argument("--disturbance-kick-probability", type=float, default=0.0)
    parser.add_argument("--disturbance-kick-velocity-xy", type=float, default=0.0)
    parser.add_argument("--disturbance-kick-yaw-velocity", type=float, default=0.0)
    parser.add_argument("--disturbance-seed", type=int, default=0)

    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--activation", choices=("relu", "elu", "tanh"), default="elu")
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--actor-lr", type=float, default=1.0e-3)
    parser.add_argument("--critic-lr", type=float, default=1.0e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=5.0e-3)
    parser.add_argument("--value-loss-coeff", type=float, default=1.0)
    parser.add_argument("--value-clip-range", type=float, default=0.2)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mirror-loss-coeff", type=float, default=0.02)
    parser.add_argument("--disable-manual-backward", action="store_true")

    parser.add_argument("--eval-world-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    train(_make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
