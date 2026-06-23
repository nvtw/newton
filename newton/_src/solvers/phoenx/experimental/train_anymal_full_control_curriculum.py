# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train a steerable Anymal walking policy with a readable curriculum.

This script is intentionally organized like an advanced example. The phase
factory functions below are the curriculum: each one names the behavior being
taught, explains why it exists, and narrows the command distribution before the
next phase broadens it. The low-level Warp-only PPO and PhoenX environment code
is reused from :mod:`train_anymal_walk_phoenx_ppo`, so this file stays focused
on the training plan.

Run from scratch with, for example:

.. code-block:: bash

    uv run --extra dev -m newton._src.solvers.phoenx.experimental.train_anymal_full_control_curriculum \
        --device cuda:0 --output-dir /tmp/phoenx_anymal_full_control

The resulting checkpoint can be replayed with:

.. code-block:: bash

    uv run --extra examples -m newton.examples robot_anymal_rl_phoenx \
        --device cuda:0 --checkpoint /tmp/phoenx_anymal_full_control/checkpoint_10_robust_full_control_*.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import newton.rl as rl
from newton._src.solvers.phoenx.experimental import train_anymal_walk_phoenx_ppo as base

Command = tuple[float, float, float, float]
Override = tuple[str, object]

IDLE: Command = (0.0, 0.0, 0.0, 0.0)
IDLE_LOW: Command = (0.0, 0.0, 0.0, -0.07)
IDLE_HIGH: Command = (0.0, 0.0, 0.0, 0.07)
FORWARD_SLOW: Command = (0.35, 0.0, 0.0, 0.0)
FORWARD: Command = (0.70, 0.0, 0.0, 0.0)
FORWARD_FAST: Command = (0.75, 0.0, 0.0, 0.0)
FORWARD_RUN: Command = (1.15, 0.0, 0.0, 0.0)
FORWARD_LOW: Command = (0.45, 0.0, 0.0, -0.07)
FORWARD_HIGH: Command = (0.45, 0.0, 0.0, 0.07)
BACKWARD: Command = (-0.20, 0.0, 0.0, 0.0)
LEFT: Command = (0.0, 0.35, 0.0, 0.0)
RIGHT: Command = (0.0, -0.35, 0.0, 0.0)
TURN_LEFT: Command = (0.0, 0.0, 0.65, 0.0)
TURN_RIGHT: Command = (0.0, 0.0, -0.65, 0.0)
CURVE_LEFT: Command = (0.55, 0.0, 0.65, 0.0)
CURVE_RIGHT: Command = (0.55, 0.0, -0.65, 0.0)

FULL_CONTROL_EVAL_COMMANDS: tuple[Command, ...] = (
    IDLE,
    FORWARD,
    FORWARD_FAST,
    FORWARD_RUN,
    BACKWARD,
    LEFT,
    RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    CURVE_LEFT,
    CURVE_RIGHT,
    IDLE_LOW,
    IDLE_HIGH,
    FORWARD_LOW,
    FORWARD_HIGH,
)


@dataclass(frozen=True)
class CurriculumPhase:
    """One named phase in the full-control Anymal curriculum."""

    name: str
    title: str
    purpose: str
    command: Command
    iterations: int
    env_overrides: tuple[Override, ...] = ()
    randomize_commands: bool = False
    command_x_range: tuple[float, float] = (0.0, 0.0)
    command_y_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_range: tuple[float, float] = (0.0, 0.0)
    command_height_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_min_abs: float = 0.0
    command_zero_probability: float = 0.0
    eval_commands: tuple[Command, ...] = ()
    gate_min_tracking_perf: float = 0.40
    gate_max_fall_fraction: float = 0.30
    gate_min_survival_fraction: float = 0.70
    gate_min_forward_velocity_fraction: float = 0.25
    gate_max_abs_forward_velocity_error: float = 0.45
    gate_max_abs_lateral_velocity_error: float = 0.45
    gate_max_abs_yaw_rate_error: float = 0.65
    gate_max_abs_base_height_error: float = 1.0

    def as_training_phase(self) -> base.PhaseAnymalWalk:
        """Convert to the compact phase type used by the PPO runner."""

        return base.PhaseAnymalWalk(
            name=self.name,
            command=self.command,
            iterations=self.iterations,
            env_overrides=self.env_overrides,
            randomize_commands=self.randomize_commands,
            command_x_range=self.command_x_range,
            command_y_range=self.command_y_range,
            command_yaw_range=self.command_yaw_range,
            command_height_range=self.command_height_range,
            command_yaw_min_abs=self.command_yaw_min_abs,
            command_zero_probability=self.command_zero_probability,
            eval_commands=self.eval_commands,
            gate_min_tracking_perf=self.gate_min_tracking_perf,
            gate_max_fall_fraction=self.gate_max_fall_fraction,
            gate_min_survival_fraction=self.gate_min_survival_fraction,
            gate_min_forward_velocity_fraction=self.gate_min_forward_velocity_fraction,
            gate_max_abs_forward_velocity_error=self.gate_max_abs_forward_velocity_error,
            gate_max_abs_lateral_velocity_error=self.gate_max_abs_lateral_velocity_error,
            gate_max_abs_yaw_rate_error=self.gate_max_abs_yaw_rate_error,
            gate_max_abs_base_height_error=self.gate_max_abs_base_height_error,
        )


# -----------------------------------------------------------------------------
# Curriculum definition
# -----------------------------------------------------------------------------


def phase_balance_and_step_forward() -> CurriculumPhase:
    """Teach the policy to stay upright while discovering a forward gait."""

    return CurriculumPhase(
        name="balance_and_step_forward",
        title="Balance and Step Forward",
        purpose="Start with easy sampled forward commands, including zero, so command conditioning starts immediately.",
        command=FORWARD_SLOW,
        iterations=140,
        env_overrides=(
            ("action_scale", 0.45),
            ("forward_progress_reward_scale", 0.0),
            ("lin_vel_reward_scale", 1.50),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.35),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.25,
        eval_commands=(IDLE, FORWARD_SLOW),
        gate_min_tracking_perf=0.25,
        gate_max_fall_fraction=0.55,
        gate_min_survival_fraction=0.45,
        gate_min_forward_velocity_fraction=0.15,
        gate_max_abs_forward_velocity_error=0.55,
    )


def phase_walk_forward() -> CurriculumPhase:
    """Turn the early stepping behavior into normal forward walking."""

    return CurriculumPhase(
        name="walk_forward",
        title="Walk Forward",
        purpose="Increase the sampled forward speed range while preserving zero-command standing.",
        command=FORWARD,
        iterations=240,
        env_overrides=(
            ("action_scale", 0.50),
            ("forward_progress_reward_scale", 0.0),
            ("lin_vel_reward_scale", 2.00),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.75),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.20,
        eval_commands=(IDLE, FORWARD_SLOW, FORWARD),
        gate_min_tracking_perf=0.45,
        gate_max_fall_fraction=0.30,
        gate_min_survival_fraction=0.70,
        gate_min_forward_velocity_fraction=0.40,
        gate_max_abs_forward_velocity_error=0.42,
    )


def phase_fast_efficient_forward() -> CurriculumPhase:
    """Make the forward gait faster and less wasteful before adding controls."""

    return CurriculumPhase(
        name="fast_efficient_forward",
        title="Fast Efficient Forward",
        purpose="Push the commanded-speed range to the viewer forward speed while penalizing excess action and energy.",
        command=FORWARD_FAST,
        iterations=280,
        env_overrides=(
            ("action_scale", 0.50),
            ("forward_progress_reward_scale", 0.0),
            ("lin_vel_reward_scale", 2.25),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.75),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.15,
        eval_commands=(IDLE, FORWARD, FORWARD_FAST),
        gate_min_tracking_perf=0.55,
        gate_max_fall_fraction=0.18,
        gate_min_survival_fraction=0.82,
        gate_min_forward_velocity_fraction=0.52,
        gate_max_abs_forward_velocity_error=0.36,
    )


def phase_run_forward() -> CurriculumPhase:
    """Teach a higher-speed forward gait without giving up posture quality."""

    return CurriculumPhase(
        name="run_forward",
        title="Run Forward",
        purpose="Extend the command-conditioned forward gait to running speeds while regularizing posture and lateral hips.",
        command=FORWARD_RUN,
        iterations=260,
        env_overrides=(
            ("action_scale", 0.58),
            ("lin_vel_reward_scale", 3.25),
            ("yaw_rate_reward_scale", 0.75),
            ("lin_vel_tracking_sigma", 0.32),
            ("yaw_rate_tracking_sigma", 0.45),
            ("base_height_reward_scale", 0.80),
            ("base_height_tracking_sigma", 0.075),
            ("hip_abduction_reward_scale", -0.35),
            ("joint_position_reward_scale", -0.025),
            ("forward_progress_reward_scale", 0.25),
            ("energy_reward_scale", -4.0e-5),
            ("action_rate_reward_scale", -0.020),
        ),
        randomize_commands=True,
        command_x_range=(0.45, 1.20),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.05,
        eval_commands=(FORWARD, FORWARD_FAST, FORWARD_RUN),
        gate_min_tracking_perf=0.42,
        gate_max_fall_fraction=0.20,
        gate_min_survival_fraction=0.80,
        gate_min_forward_velocity_fraction=0.45,
        gate_max_abs_forward_velocity_error=0.60,
        gate_max_abs_lateral_velocity_error=0.35,
        gate_max_abs_yaw_rate_error=0.50,
        gate_max_abs_base_height_error=0.12,
    )


def phase_robust_forward() -> CurriculumPhase:
    """Add small velocity noise and rare kicks without changing the task."""

    return CurriculumPhase(
        name="robust_forward",
        title="Robust Forward",
        purpose="Keep command-conditioned forward walking stable under realistic noise and occasional kicks.",
        command=FORWARD_FAST,
        iterations=200,
        env_overrides=(
            ("action_scale", 0.50),
            ("forward_progress_reward_scale", 0.0),
            ("lin_vel_reward_scale", 2.25),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
            ("disturbance_warmup_steps", 50),
            ("disturbance_noise_velocity_xy", 0.025),
            ("disturbance_noise_yaw_velocity", 0.015),
            ("disturbance_kick_probability", 0.003),
            ("disturbance_kick_velocity_xy", 0.45),
            ("disturbance_kick_yaw_velocity", 0.35),
            ("disturbance_seed", 41_337),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.75),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.15,
        eval_commands=(IDLE, FORWARD, FORWARD_FAST),
        gate_min_tracking_perf=0.50,
        gate_max_fall_fraction=0.10,
        gate_min_survival_fraction=0.90,
        gate_min_forward_velocity_fraction=0.48,
        gate_max_abs_forward_velocity_error=0.45,
    )


def phase_base_height_control() -> CurriculumPhase:
    """Teach commanded lower and higher body heights on stable walking."""

    return CurriculumPhase(
        name="base_height_control",
        title="Base Height Control",
        purpose="Condition the policy on a commanded body-height offset before adding yaw and lateral commands.",
        command=FORWARD_HIGH,
        iterations=220,
        env_overrides=(
            ("action_scale", 0.50),
            ("lin_vel_reward_scale", 2.25),
            ("yaw_rate_reward_scale", 1.00),
            ("base_height_reward_scale", 1.40),
            ("base_height_tracking_sigma", 0.055),
            ("hip_abduction_reward_scale", -0.25),
            ("forward_progress_reward_scale", 0.10),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.65),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_height_range=(-0.08, 0.08),
        command_zero_probability=0.15,
        eval_commands=(IDLE, IDLE_LOW, IDLE_HIGH, FORWARD_LOW, FORWARD_HIGH),
        gate_min_tracking_perf=0.35,
        gate_max_fall_fraction=0.18,
        gate_min_survival_fraction=0.82,
        gate_min_forward_velocity_fraction=0.20,
        gate_max_abs_forward_velocity_error=0.50,
        gate_max_abs_base_height_error=0.085,
    )


def phase_turn_in_place() -> CurriculumPhase:
    """Teach yaw control in isolation while preserving idle balance."""

    return CurriculumPhase(
        name="turn_in_place",
        title="Turn In Place",
        purpose="Learn left and right yaw commands before combining yaw with translation.",
        command=TURN_LEFT,
        iterations=220,
        env_overrides=(
            ("action_scale", 0.55),
            ("lin_vel_reward_scale", 0.35),
            ("yaw_rate_reward_scale", 2.50),
            ("yaw_rate_tracking_sigma", 0.35),
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
        eval_commands=(IDLE, TURN_LEFT, TURN_RIGHT),
        gate_min_tracking_perf=0.35,
        gate_max_fall_fraction=0.12,
        gate_min_survival_fraction=0.88,
        gate_max_abs_forward_velocity_error=0.32,
        gate_max_abs_lateral_velocity_error=0.32,
        gate_max_abs_yaw_rate_error=0.50,
    )


def phase_recover_forward_after_turning() -> CurriculumPhase:
    """Re-anchor forward walking after the turning specialization."""

    return CurriculumPhase(
        name="recover_forward_after_turning",
        title="Recover Forward After Turning",
        purpose="Prevent the yaw phase from becoming a stand-only local optimum by revisiting forward walking.",
        command=FORWARD,
        iterations=120,
        env_overrides=(
            ("action_scale", 0.50),
            ("lin_vel_reward_scale", 2.00),
            ("yaw_rate_reward_scale", 1.00),
            ("forward_progress_reward_scale", 0.0),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.75),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.20,
        eval_commands=(IDLE, FORWARD, TURN_LEFT, TURN_RIGHT),
        gate_min_tracking_perf=0.42,
        gate_max_fall_fraction=0.18,
        gate_min_survival_fraction=0.82,
        gate_min_forward_velocity_fraction=0.35,
        gate_max_abs_forward_velocity_error=0.46,
        gate_max_abs_yaw_rate_error=0.58,
    )


def phase_curved_forward() -> CurriculumPhase:
    """Combine forward motion and yaw without lateral or reverse commands yet."""

    return CurriculumPhase(
        name="curved_forward",
        title="Curved Forward",
        purpose="Teach ASDWQE-style Q/E steering: move forward while turning left or right.",
        command=CURVE_LEFT,
        iterations=220,
        env_overrides=(
            ("action_scale", 0.50),
            ("lin_vel_reward_scale", 2.25),
            ("yaw_rate_reward_scale", 1.75),
            ("lin_vel_tracking_sigma", 0.35),
            ("yaw_rate_tracking_sigma", 0.35),
            ("forward_progress_reward_scale", 0.0),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.75),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(-0.85, 0.85),
        command_zero_probability=0.10,
        eval_commands=(IDLE, FORWARD, CURVE_LEFT, CURVE_RIGHT),
        gate_min_tracking_perf=0.40,
        gate_max_fall_fraction=0.15,
        gate_min_survival_fraction=0.85,
        gate_min_forward_velocity_fraction=0.30,
        gate_max_abs_forward_velocity_error=0.45,
        gate_max_abs_yaw_rate_error=0.60,
    )


def phase_reverse_walk() -> CurriculumPhase:
    """Add gentle backwards control after the forward and yaw gait is stable."""

    return CurriculumPhase(
        name="reverse_walk",
        title="Reverse Walk",
        purpose="Add S-key style reverse walking as a late refinement, at a speed the no-reset eval can hold.",
        command=BACKWARD,
        iterations=320,
        env_overrides=(
            ("action_scale", 0.50),
            ("lin_vel_reward_scale", 4.00),
            ("yaw_rate_reward_scale", 0.35),
            ("lin_vel_tracking_sigma", 0.16),
            ("yaw_rate_tracking_sigma", 0.40),
            ("forward_progress_reward_scale", 3.00),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
        ),
        randomize_commands=True,
        command_x_range=(-0.25, -0.04),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.10,
        eval_commands=(IDLE, BACKWARD),
        gate_min_tracking_perf=0.30,
        gate_max_fall_fraction=0.18,
        gate_min_survival_fraction=0.82,
        gate_min_forward_velocity_fraction=0.25,
        gate_max_abs_forward_velocity_error=0.35,
    )


def phase_side_step() -> CurriculumPhase:
    """Teach lateral body-frame velocity commands separately."""

    return CurriculumPhase(
        name="side_step",
        title="Side Step",
        purpose="Add A/D lateral walking only after forward, reverse, and yaw are stable.",
        command=LEFT,
        iterations=280,
        env_overrides=(
            ("action_scale", 0.50),
            ("lin_vel_reward_scale", 3.25),
            ("yaw_rate_reward_scale", 1.00),
            ("lin_vel_tracking_sigma", 0.24),
            ("yaw_rate_tracking_sigma", 0.40),
            ("forward_progress_reward_scale", 1.50),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
        ),
        randomize_commands=True,
        command_x_range=(0.0, 0.0),
        command_y_range=(-0.45, 0.45),
        command_yaw_range=(0.0, 0.0),
        command_zero_probability=0.05,
        eval_commands=(IDLE, LEFT, RIGHT),
        gate_min_tracking_perf=0.32,
        gate_max_fall_fraction=0.20,
        gate_min_survival_fraction=0.80,
        gate_min_forward_velocity_fraction=0.00,
        gate_max_abs_forward_velocity_error=0.42,
        gate_max_abs_lateral_velocity_error=0.35,
    )


def phase_full_control_mix() -> CurriculumPhase:
    """Mix all user controls after each component behavior has a foothold."""

    return CurriculumPhase(
        name="full_control_mix",
        title="Full Control Mix",
        purpose="Train the complete W/S, A/D, and Q/E command space without disturbances.",
        command=FORWARD,
        iterations=340,
        env_overrides=(
            ("action_scale", 0.50),
            ("lin_vel_reward_scale", 3.00),
            ("yaw_rate_reward_scale", 1.60),
            ("lin_vel_tracking_sigma", 0.28),
            ("yaw_rate_tracking_sigma", 0.35),
            ("forward_progress_reward_scale", 1.00),
            ("hip_abduction_reward_scale", -0.20),
            ("joint_position_reward_scale", -0.015),
            ("energy_reward_scale", -3.0e-5),
            ("action_rate_reward_scale", -0.015),
        ),
        randomize_commands=True,
        command_x_range=(-0.25, 1.15),
        command_y_range=(-0.45, 0.45),
        command_yaw_range=(-0.90, 0.90),
        command_height_range=(-0.07, 0.07),
        command_zero_probability=0.08,
        eval_commands=FULL_CONTROL_EVAL_COMMANDS,
        gate_min_tracking_perf=0.38,
        gate_max_fall_fraction=0.16,
        gate_min_survival_fraction=0.84,
        gate_min_forward_velocity_fraction=0.25,
        gate_max_abs_forward_velocity_error=0.46,
        gate_max_abs_lateral_velocity_error=0.40,
        gate_max_abs_yaw_rate_error=0.62,
    )


def phase_robust_full_control() -> CurriculumPhase:
    """Finish by making the full command policy robust to noise and kicks."""

    return CurriculumPhase(
        name="robust_full_control",
        title="Robust Full Control",
        purpose="Keep the final steerable gait alive under small continuous noise and rare finite kicks.",
        command=FORWARD,
        iterations=280,
        env_overrides=(
            ("action_scale", 0.50),
            ("lin_vel_reward_scale", 3.00),
            ("yaw_rate_reward_scale", 1.60),
            ("lin_vel_tracking_sigma", 0.28),
            ("yaw_rate_tracking_sigma", 0.35),
            ("forward_progress_reward_scale", 1.00),
            ("hip_abduction_reward_scale", -0.20),
            ("joint_position_reward_scale", -0.015),
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
        command_x_range=(-0.25, 1.15),
        command_y_range=(-0.45, 0.45),
        command_yaw_range=(-0.90, 0.90),
        command_height_range=(-0.07, 0.07),
        command_zero_probability=0.10,
        eval_commands=FULL_CONTROL_EVAL_COMMANDS,
        gate_min_tracking_perf=0.35,
        gate_max_fall_fraction=0.12,
        gate_min_survival_fraction=0.88,
        gate_min_forward_velocity_fraction=0.20,
        gate_max_abs_forward_velocity_error=0.50,
        gate_max_abs_lateral_velocity_error=0.44,
        gate_max_abs_yaw_rate_error=0.66,
    )


def build_full_control_curriculum() -> tuple[CurriculumPhase, ...]:
    """Return the phase sequence, written in the order it is trained."""

    return (
        phase_balance_and_step_forward(),
        phase_walk_forward(),
        phase_fast_efficient_forward(),
        phase_robust_forward(),
        phase_run_forward(),
        phase_base_height_control(),
        phase_turn_in_place(),
        phase_recover_forward_after_turning(),
        phase_curved_forward(),
        phase_reverse_walk(),
        phase_side_step(),
        phase_full_control_mix(),
        phase_robust_full_control(),
    )


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def build_phase_env_config(
    args: argparse.Namespace,
    phase: CurriculumPhase,
    *,
    world_count: int | None = None,
    auto_reset: bool = True,
) -> rl.ConfigEnvAnymalPhoenX:
    """Build the PhoenX Anymal environment config for one phase."""

    env_overrides = _phase_env_overrides(args, phase)
    return base.build_env_config(
        args,
        world_count=world_count,
        auto_reset=auto_reset,
        command=phase.command,
        env_overrides=env_overrides,
    )


def _phase_env_overrides(args: argparse.Namespace, phase: CurriculumPhase) -> dict[str, object]:
    env_overrides = dict(phase.env_overrides)
    if args.phase_energy_reward_scale is not None:
        env_overrides["energy_reward_scale"] = float(args.phase_energy_reward_scale)
    if args.phase_action_rate_reward_scale is not None:
        env_overrides["action_rate_reward_scale"] = float(args.phase_action_rate_reward_scale)
    if args.phase_joint_speed_reward_scale is not None:
        env_overrides["joint_speed_reward_scale"] = float(args.phase_joint_speed_reward_scale)
    if args.phase_base_height_reward_scale is not None:
        env_overrides["base_height_reward_scale"] = float(args.phase_base_height_reward_scale)
    if args.phase_hip_abduction_reward_scale is not None:
        env_overrides["hip_abduction_reward_scale"] = float(args.phase_hip_abduction_reward_scale)
    return env_overrides


def _selected_phases(args: argparse.Namespace) -> tuple[CurriculumPhase, ...]:
    phases = build_full_control_curriculum()[int(args.start_phase) :]
    if args.phase_count is not None:
        phases = phases[: int(args.phase_count)]
    if not phases:
        raise ValueError("Selected curriculum is empty")
    return phases


def _checkpoint_pattern(output_dir: Path, phase_index: int, phase: CurriculumPhase) -> str:
    return str(output_dir / f"checkpoint_{phase_index:02d}_{phase.name}_{{iteration}}.npz")


def _format_checkpoint(pattern: str, iteration: int) -> str:
    return str(Path(pattern.format(iteration=int(iteration))))


def _print_phase_header(
    phase_number: int, phase_count: int, phase: CurriculumPhase, resume_checkpoint: str | None
) -> None:
    print("")
    print(f"=== Phase {phase_number}/{phase_count}: {phase.title} ===")
    print(f"name={phase.name}")
    print(f"purpose={phase.purpose}")
    print(f"command={phase.command} resume={resume_checkpoint or '-'}")
    if phase.randomize_commands:
        print(
            "command_ranges="
            f"x{phase.command_x_range} y{phase.command_y_range} yaw{phase.command_yaw_range} "
            f"height{phase.command_height_range} zero_probability={phase.command_zero_probability:.2f}"
        )


def _phase_payload(
    *,
    phase_index: int,
    phase: CurriculumPhase,
    iterations: int,
    checkpoint: str | None,
    result: rl.ResultTrainAnymalPPO,
    eval_stats: list[base.StatsEvaluateAnymalWalk] | None,
    gate_failures: list[str],
) -> dict[str, object]:
    return {
        "phase_index": phase_index,
        "phase": asdict(phase),
        "iterations": iterations,
        "checkpoint": checkpoint,
        "final_train_stats": asdict(result.history[-1]) if result.history else {},
        "eval_command_stats": [asdict(item) for item in eval_stats] if eval_stats else None,
        "pass_gate": not gate_failures,
        "gate_failures": gate_failures,
    }


def _write_summary(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_one_phase(
    args: argparse.Namespace,
    *,
    phase_index: int,
    phase_number: int,
    phase_count: int,
    phase: CurriculumPhase,
    output_dir: Path,
    resume_checkpoint: str | None,
) -> tuple[dict[str, object], str | None]:
    training_phase = phase.as_training_phase()
    phase_iterations = max(1, int(round(float(training_phase.iterations) * float(args.iteration_scale))))
    checkpoint_pattern = _checkpoint_pattern(output_dir, phase_index, phase)
    _print_phase_header(phase_number, phase_count, phase, resume_checkpoint)
    result = rl.train_anymal_ppo(
        rl.ConfigTrainAnymalPPO(
            iterations=phase_iterations,
            rollout_steps=int(args.rollout_steps),
            hidden_layers=tuple(int(v) for v in args.hidden_layers),
            activation=str(args.activation),
            log_std_init=float(args.log_std_init),
            env_config=build_phase_env_config(args, phase),
            ppo_config=base.build_ppo_config(args),
            device=args.device,
            seed=int(args.seed) + phase_index * 10_003,
            log_interval=int(args.log_interval),
            use_target_curriculum=False,
            randomize_target_positions=False,
            randomize_commands=bool(training_phase.randomize_commands),
            command_x_range=tuple(float(v) for v in training_phase.command_x_range),
            command_y_range=tuple(float(v) for v in training_phase.command_y_range),
            command_yaw_range=tuple(float(v) for v in training_phase.command_yaw_range),
            command_height_range=tuple(float(v) for v in training_phase.command_height_range),
            command_yaw_min_abs=float(training_phase.command_yaw_min_abs),
            command_zero_probability=float(training_phase.command_zero_probability),
            resume_checkpoint=resume_checkpoint,
            checkpoint_path=checkpoint_pattern,
            checkpoint_interval=int(args.checkpoint_interval),
        )
    )
    final_checkpoint = _format_checkpoint(checkpoint_pattern, int(result.trainer.iteration))
    eval_stats = None
    gate_failures: list[str] = []
    if not bool(args.no_eval):
        eval_stats = base.evaluate_phase_commands(
            result.trainer, args, training_phase, _phase_env_overrides(args, phase)
        )
        gate_failures = base.check_phase_gates(eval_stats, training_phase)
    payload = _phase_payload(
        phase_index=phase_index,
        phase=phase,
        iterations=phase_iterations,
        checkpoint=final_checkpoint,
        result=result,
        eval_stats=eval_stats,
        gate_failures=gate_failures,
    )
    print(json.dumps(payload, sort_keys=True))
    if gate_failures and not bool(args.allow_gate_failure):
        raise RuntimeError(f"Anymal phase {phase.name!r} failed gate: {', '.join(gate_failures)}")
    return payload, final_checkpoint


def run_curriculum(args: argparse.Namespace) -> dict[str, object]:
    """Train each phase from scratch or from the requested checkpoint."""

    phases = _selected_phases(args)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is None:
        raise ValueError("--output-dir is required for curriculum training")
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.dry_run):
        payload = {"phases": [asdict(phase) for phase in phases]}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    phase_payloads: list[dict[str, object]] = []
    resume_checkpoint = args.resume_checkpoint
    final_checkpoint = resume_checkpoint
    for phase_number, phase in enumerate(phases, start=1):
        phase_index = int(args.start_phase) + phase_number - 1
        phase_payload, final_checkpoint = _run_one_phase(
            args,
            phase_index=phase_index,
            phase_number=phase_number,
            phase_count=len(phases),
            phase=phase,
            output_dir=output_dir,
            resume_checkpoint=resume_checkpoint,
        )
        phase_payloads.append(phase_payload)
        payload = {
            "recipe": "full_control",
            "final_checkpoint": final_checkpoint,
            "phases": phase_payloads,
        }
        _write_summary(output_dir, payload)
        resume_checkpoint = final_checkpoint
    payload = {"recipe": "full_control", "final_checkpoint": final_checkpoint, "phases": phase_payloads}
    _write_summary(output_dir, payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=12_321)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument("--iteration-scale", type=float, default=1.0)
    parser.add_argument("--phase-energy-reward-scale", type=float, default=None)
    parser.add_argument("--phase-action-rate-reward-scale", type=float, default=None)
    parser.add_argument("--phase-joint-speed-reward-scale", type=float, default=None)
    parser.add_argument("--phase-base-height-reward-scale", type=float, default=None)
    parser.add_argument("--phase-hip-abduction-reward-scale", type=float, default=None)
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--world-count", type=int, default=1024)
    parser.add_argument("--rollout-steps", type=int, default=32)
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
    parser.add_argument("--command-x", type=float, default=0.6)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-height", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_curriculum(_make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
