# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Repeatable PhoenX G1 velocity-command walking curricula.

This runner trains sustained command following first, which is closer to the
nanoG1 task shape than one-shot target reaching. Each phase writes a normal PPO
checkpoint, evaluates it without auto-reset on fixed command gates, and stops
before chaining to the next phase when the current checkpoint is not stable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe

_BASE_COMMAND_ENV_OVERRIDES: tuple[tuple[str, Any], ...] = (
    ("reward_mode", "nanog1_dense"),
    ("w_alive", 1.0),
    ("w_termination", -4.0),
    ("w_track_lin", 3.0),
    ("w_track_ang", 1.25),
    ("w_lin_vel_z", -2.0),
    ("w_ang_vel_xy", -1.3),
    ("w_orientation", -10.0),
    ("w_torque", -2.0e-5),
    ("w_action_rate", -0.01),
    ("w_mechanical_power", -5.0e-5),
    ("w_feet_air_time", 0.5),
    ("feet_air_time_threshold", 0.4),
    ("w_feet_slide", -0.1),
    ("w_joint_deviation_hip", -0.05),
    ("w_joint_deviation_waist", -0.02),
    ("w_joint_deviation_upper", -0.01),
    ("w_joint_acc_legs", -1.0e-5),
    ("w_joint_pos_limit_ankle", -0.5),
)

_BASE_COMMAND_PPO_OVERRIDES: tuple[tuple[str, Any], ...] = (
    ("actor_lr", 2.0e-4),
    ("critic_lr", 2.0e-4),
    ("entropy_coeff", 1.0e-5),
    ("max_grad_norm", 0.3),
    ("reward_clip", 1.0),
    ("mirror_loss_coeff", 0.25),
)


@dataclass(frozen=True)
class PhaseG1Command:
    """One stage of sustained command-following training."""

    name: str
    iterations: int
    command_x_range: tuple[float, float]
    command_y_range: tuple[float, float] = (0.0, 0.0)
    command_yaw_range: tuple[float, float] = (0.0, 0.0)
    command_zero_probability: float = 0.0
    command_curriculum_start: float = 0.35
    command_curriculum_samples: int = 80_000_000
    log_std_init: float = -0.7
    env_overrides: tuple[tuple[str, Any], ...] = ()
    ppo_overrides: tuple[tuple[str, Any], ...] = ()
    gate_commands: tuple[tuple[float, float, float], ...] = ()
    gate_min_tracking_perf: float = 0.30
    gate_max_fall_fraction: float = 0.05
    gate_min_survival_fraction: float = 0.95


@dataclass(frozen=True)
class PhaseG1CommandResult:
    """Summary for one completed command curriculum stage."""

    phase: PhaseG1Command
    checkpoint: str
    elapsed_seconds: float
    final_train_stats: dict[str, Any]
    command_eval_stats: list[dict[str, Any]]
    phase_gate_passed: bool
    phase_gate_failures: list[str]


def build_command_curriculum(recipe: str, *, iteration_scale: float = 1.0) -> list[PhaseG1Command]:
    """Build a named command-following curriculum."""

    if iteration_scale <= 0.0:
        raise ValueError("iteration_scale must be positive")
    if recipe == "simple-forward":
        phases = [
            PhaseG1Command(
                name="slow_forward",
                iterations=220,
                command_x_range=(0.15, 0.45),
                gate_commands=((0.3, 0.0, 0.0),),
            )
        ]
    elif recipe == "advanced-command":
        phases = [
            PhaseG1Command(
                name="slow_forward",
                iterations=220,
                command_x_range=(0.15, 0.45),
                gate_commands=((0.3, 0.0, 0.0),),
            ),
            PhaseG1Command(
                name="forward_walk",
                iterations=260,
                command_x_range=(0.25, 0.80),
                command_curriculum_start=0.50,
                ppo_overrides=(("actor_lr", 1.5e-4), ("critic_lr", 1.5e-4)),
                gate_commands=((0.3, 0.0, 0.0), (0.6, 0.0, 0.0)),
            ),
            PhaseG1Command(
                name="omnidirectional_walk",
                iterations=260,
                command_x_range=(-0.4, 0.9),
                command_y_range=(-0.35, 0.35),
                command_yaw_range=(-0.8, 0.8),
                command_zero_probability=0.05,
                command_curriculum_start=0.55,
                ppo_overrides=(("actor_lr", 1.0e-4), ("critic_lr", 1.0e-4)),
                gate_commands=((0.6, 0.0, 0.0), (0.3, 0.0, 0.6), (0.0, 0.3, 0.0)),
            ),
        ]
    else:
        raise ValueError("recipe must be 'simple-forward' or 'advanced-command'")
    return [replace(phase, iterations=max(1, int(round(phase.iterations * iteration_scale)))) for phase in phases]


def select_phases(
    phases: list[PhaseG1Command],
    *,
    start_phase: int,
    phase_count: int | None,
) -> list[tuple[int, PhaseG1Command]]:
    """Select indexed phases for a full or resumed run."""

    if start_phase < 0:
        raise ValueError("start_phase must be non-negative")
    if start_phase >= len(phases):
        raise ValueError("start_phase must select an existing phase")
    if phase_count is not None and phase_count <= 0:
        raise ValueError("phase_count must be positive when provided")
    selected = list(enumerate(phases[start_phase:], start=start_phase))
    if phase_count is not None:
        selected = selected[: int(phase_count)]
    return selected


def _merged_overrides(*items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in items:
        values.update(dict(group))
    return values


def build_env_config(phase: PhaseG1Command, args: argparse.Namespace, *, command: tuple[float, float, float]):
    """Build the environment configuration for one command phase."""

    values = _merged_overrides(_BASE_COMMAND_ENV_OVERRIDES, phase.env_overrides)
    values.update(
        {
            "world_count": int(args.world_count),
            "command": tuple(float(v) for v in command),
            "command_x_range": tuple(float(v) for v in phase.command_x_range),
            "command_y_range": tuple(float(v) for v in phase.command_y_range),
            "command_yaw_range": tuple(float(v) for v in phase.command_yaw_range),
            "command_zero_probability": float(phase.command_zero_probability),
            "sim_substeps": int(args.sim_substeps),
            "solver_iterations": int(args.solver_iterations),
            "velocity_iterations": int(args.velocity_iterations),
            "joint_friction_model": str(args.joint_friction_model),
            "joint_friction_scale": float(args.joint_friction_scale),
            "actuation_model": str(args.actuation_model),
            "action_scale": float(args.action_scale),
            "controlled_action_count": int(args.controlled_action_count),
            "observation_mode": str(args.observation_mode),
            "parse_meshes": bool(args.parse_meshes),
            "contact_geometry": str(args.contact_geometry),
            "ground_friction": float(args.ground_friction),
            "foot_box_xy_scale": float(args.foot_box_xy_scale),
            "rigid_contact_max_per_world": int(args.rigid_contact_max_per_world),
        }
    )
    return g1_recipe.default_g1_env_config(**values)


def build_ppo_config(phase: PhaseG1Command, args: argparse.Namespace):
    """Build the PPO configuration for one command phase."""

    values = _merged_overrides(_BASE_COMMAND_PPO_OVERRIDES, phase.ppo_overrides)
    if args.actor_lr is not None:
        values["actor_lr"] = float(args.actor_lr)
    if args.critic_lr is not None:
        values["critic_lr"] = float(args.critic_lr)
    return g1_recipe.default_g1_ppo_config(**values)


def build_train_config(
    phase: PhaseG1Command,
    args: argparse.Namespace,
    *,
    seed: int,
    resume_checkpoint: str | None,
    checkpoint_path: Path,
) -> rl.ConfigTrainG1PPO:
    """Build the train configuration for one command phase."""

    return rl.ConfigTrainG1PPO(
        iterations=int(phase.iterations),
        rollout_steps=int(args.rollout_steps),
        env_config=build_env_config(phase, args, command=(float(phase.command_x_range[1]), 0.0, 0.0)),
        ppo_config=build_ppo_config(phase, args),
        device=args.device,
        seed=int(seed),
        log_interval=int(args.log_interval),
        randomize_commands=True,
        command_sampling="rollout",
        command_x_range=tuple(float(v) for v in phase.command_x_range),
        command_y_range=tuple(float(v) for v in phase.command_y_range),
        command_yaw_range=tuple(float(v) for v in phase.command_yaw_range),
        command_zero_probability=float(phase.command_zero_probability),
        command_curriculum_start=float(phase.command_curriculum_start),
        command_curriculum_samples=int(phase.command_curriculum_samples),
        reset_recurrent_state_on_rollout_start=True,
        activation=str(args.activation),
        log_std_init=float(args.log_std_init if args.log_std_init is not None else phase.log_std_init),
        resume_checkpoint=resume_checkpoint,
        checkpoint_path=str(checkpoint_path),
        checkpoint_interval=int(args.checkpoint_interval),
        readback_diagnostics=not bool(args.no_readback_diagnostics),
        execution_mode=str(args.execution_mode),
    )


def _format_checkpoint_path(pattern: Path, iteration: int) -> Path:
    return Path(str(pattern).format(iteration=int(iteration)))


def evaluate_checkpoint(checkpoint: Path, phase: PhaseG1Command, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Evaluate a checkpoint on fixed command gates."""

    trainer = rl.load_ppo_checkpoint(str(checkpoint), device=args.device)
    eval_stats = []
    eval_args = argparse.Namespace(**vars(args))
    eval_args.world_count = int(args.eval_world_count)
    for command in phase.gate_commands:
        env_config = build_env_config(phase, eval_args, command=command)
        result = rl.evaluate_g1_ppo(
            trainer,
            rl.ConfigEvaluateG1PPO(
                env_config=env_config,
                steps=int(args.eval_steps),
                device=args.device,
                deterministic=not bool(args.stochastic_eval),
                seed=int(args.seed) + 911,
            ),
        )
        stat = asdict(result.stats)
        stat["command"] = tuple(float(v) for v in command)
        eval_stats.append(stat)
    return eval_stats


def check_phase_gate(phase: PhaseG1Command, command_eval_stats: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Check whether a command phase is good enough to feed the next phase."""

    failures: list[str] = []
    if not phase.gate_commands:
        return True, failures
    stats_by_command = {tuple(float(v) for v in stat["command"]): stat for stat in command_eval_stats}
    for command in phase.gate_commands:
        stat = stats_by_command.get(tuple(float(v) for v in command))
        if stat is None:
            failures.append(f"missing eval command={command}")
            continue
        fall_fraction = float(stat["fall_fraction"])
        survival_fraction = float(stat["mean_survival_steps"]) / max(float(stat["steps"]), 1.0)
        tracking_perf = float(stat["mean_tracking_perf"])
        if fall_fraction > float(phase.gate_max_fall_fraction):
            failures.append(f"command={command} fall_fraction={fall_fraction:.3f} > {phase.gate_max_fall_fraction:.3f}")
        if survival_fraction < float(phase.gate_min_survival_fraction):
            failures.append(
                f"command={command} survival_fraction={survival_fraction:.3f} < {phase.gate_min_survival_fraction:.3f}"
            )
        if tracking_perf < float(phase.gate_min_tracking_perf):
            failures.append(f"command={command} tracking_perf={tracking_perf:.3f} < {phase.gate_min_tracking_perf:.3f}")
    return not failures, failures


def run_curriculum(args: argparse.Namespace) -> list[PhaseG1CommandResult]:
    """Train all selected command phases."""

    phases = build_command_curriculum(str(args.recipe), iteration_scale=float(args.iteration_scale))
    selected = select_phases(phases, start_phase=int(args.start_phase), phase_count=args.phase_count)
    if bool(args.dry_run):
        print(json.dumps({"recipe": args.recipe, "phases": [{"index": i, **asdict(p)} for i, p in selected]}, indent=2))
        return []
    if int(args.start_phase) > 0 and args.resume_checkpoint is None:
        raise ValueError("--resume-checkpoint is required when --start-phase is greater than zero")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    results: list[PhaseG1CommandResult] = []
    resume_checkpoint = args.resume_checkpoint
    for phase_index, phase in selected:
        checkpoint_pattern = output_dir / f"{phase_index:02d}_{phase.name}_{{iteration}}.npz"
        t0 = time.perf_counter()
        train_result = rl.train_g1_ppo(
            build_train_config(
                phase,
                args,
                seed=int(args.seed) + phase_index * 10_003,
                resume_checkpoint=resume_checkpoint,
                checkpoint_path=checkpoint_pattern,
            )
        )
        checkpoint = _format_checkpoint_path(checkpoint_pattern, int(train_result.trainer.iteration))
        final_stats = asdict(train_result.history[-1]) if train_result.history else {}
        eval_stats = [] if bool(args.no_eval) else evaluate_checkpoint(checkpoint, phase, args)
        elapsed = time.perf_counter() - t0
        gate_passed, gate_failures = (True, [])
        if not bool(args.no_phase_gates):
            if bool(args.no_eval):
                gate_passed = False
                gate_failures = ["phase gates require evaluation; remove --no-eval or pass --no-phase-gates"]
            else:
                gate_passed, gate_failures = check_phase_gate(phase, eval_stats)
        result = PhaseG1CommandResult(
            phase=phase,
            checkpoint=str(checkpoint),
            elapsed_seconds=float(elapsed),
            final_train_stats=final_stats,
            command_eval_stats=eval_stats,
            phase_gate_passed=bool(gate_passed),
            phase_gate_failures=list(gate_failures),
        )
        results.append(result)
        resume_checkpoint = str(checkpoint)
        summary_path.write_text(
            json.dumps(
                {"recipe": args.recipe, "output_dir": str(output_dir), "results": [asdict(r) for r in results]},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if not gate_passed and not bool(args.allow_gate_failure):
            raise RuntimeError(f"phase {phase_index} {phase.name!r} failed quality gate: {gate_failures}")
    print(json.dumps({"summary": str(summary_path), "final_checkpoint": resume_checkpoint}, sort_keys=True))
    return results


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("simple-forward", "advanced-command"), default="simple-forward")
    parser.add_argument("--output-dir", default="/tmp/phoenx_g1_command_curriculum")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=41_001)
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--iteration-scale", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument("--joint-friction-model", choices=("hard", "mujoco"), default=g1_recipe.JOINT_FRICTION_MODEL)
    parser.add_argument("--joint-friction-scale", type=float, default=g1_recipe.JOINT_FRICTION_SCALE)
    parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
    )
    parser.add_argument("--action-scale", type=float, default=g1_recipe.ACTION_SCALE)
    parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    parser.add_argument("--observation-mode", choices=("nanog1", "isaaclab_flat"), default=g1_recipe.OBSERVATION_MODE)
    parser.add_argument("--activation", choices=("relu", "elu", "tanh"), default=g1_recipe.ACTIVATION)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY)
    parser.add_argument("--ground-friction", type=float, default=0.4)
    parser.add_argument("--foot-box-xy-scale", type=float, default=g1_recipe.FOOT_BOX_XY_SCALE)
    parser.add_argument("--rigid-contact-max-per-world", type=int, default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
    parser.add_argument("--execution-mode", choices=("eager", "graph_leapfrog"), default="eager")
    parser.add_argument("--no-readback-diagnostics", action="store_true")

    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)

    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-phase-gates", action="store_true")
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--eval-world-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=700)
    parser.add_argument("--stochastic-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    run_curriculum(_make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
