# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Repeatable PhoenX G1 target-walking training curricula.

This script is intentionally experimental, but it uses the normal public
``newton.rl`` training API and writes normal PPO checkpoints. The simple recipe
is a short dense-target walking probe. The advanced recipe stages the same task
from short, nearly-forward targets to longer target-conditioned walking.

Example:

```
uv run --extra dev -m newton._src.solvers.phoenx.rl_training.examples.train_g1_curriculum \
    --recipe advanced-target --output-dir /tmp/phoenx_g1_advanced --device cuda:0
```
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe

_BASE_TARGET_ENV_OVERRIDES: tuple[tuple[str, Any], ...] = (
    ("reward_mode", "dense_target"),
    ("command", (0.0, 0.0, 0.0)),
    ("w_sparse_command_success", 4.0),
    ("w_target_progress", 3.0),
    ("w_alive", 1.0),
    ("w_termination", -2.0),
    ("w_track_ang", 1.0),
    ("w_lin_vel_z", -2.0),
    ("w_ang_vel_xy", -1.0),
    ("w_orientation", -8.0),
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

_BASE_TARGET_PPO_OVERRIDES: tuple[tuple[str, Any], ...] = (
    ("actor_lr", 2.0e-4),
    ("critic_lr", 2.0e-4),
    ("entropy_coeff", 1.0e-5),
    ("max_grad_norm", 0.3),
    ("reward_clip", 1.0),
    ("mirror_loss_coeff", 0.25),
)


@dataclass(frozen=True)
class PhaseG1Curriculum:
    """One stage of target-conditioned G1 PPO training."""

    name: str
    iterations: int
    target_distance_start: float
    target_distance_end: float
    target_angle_min: float
    target_angle_max: float
    sparse_target_radius: float = 0.30
    target_curriculum_samples: int = 60_000_000
    log_std_init: float = -0.7
    env_overrides: tuple[tuple[str, Any], ...] = ()
    ppo_overrides: tuple[tuple[str, Any], ...] = ()
    gate_targets: tuple[float, ...] = ()
    gate_min_strict_success_fraction: float = 0.90
    gate_max_fall_fraction: float = 0.05
    gate_max_tilt_violation_fraction: float = 0.05


@dataclass(frozen=True)
class PhaseG1Result:
    """Summary for one completed curriculum stage."""

    phase: PhaseG1Curriculum
    checkpoint: str
    elapsed_seconds: float
    final_train_stats: dict[str, Any]
    target_eval_stats: list[dict[str, Any]]
    phase_gate_passed: bool
    phase_gate_failures: list[str]


def build_curriculum(
    recipe: str,
    *,
    iteration_scale: float = 1.0,
    min_target_margin: float = 0.10,
) -> list[PhaseG1Curriculum]:
    """Build a named G1 training curriculum.

    Args:
        recipe: Either ``"simple-target"`` or ``"advanced-target"``.
        iteration_scale: Multiplier for every phase iteration count.
        min_target_margin: Minimum start-distance margin beyond the success
            radius [m]. This prevents degenerate curricula that succeed at
            reset.

    Returns:
        Sanitized curriculum phases.
    """

    if iteration_scale <= 0.0:
        raise ValueError("iteration_scale must be positive")
    if min_target_margin <= 0.0:
        raise ValueError("min_target_margin must be positive")

    if recipe == "simple-target":
        phases = [
            PhaseG1Curriculum(
                name="simple_dense_target",
                iterations=160,
                target_distance_start=0.45,
                target_distance_end=1.0,
                target_angle_min=-0.20,
                target_angle_max=0.20,
                sparse_target_radius=0.30,
                target_curriculum_samples=45_000_000,
                gate_targets=(0.6,),
            )
        ]
    elif recipe == "advanced-target":
        phases = [
            PhaseG1Curriculum(
                name="short_forward_targets",
                iterations=140,
                target_distance_start=0.45,
                target_distance_end=0.95,
                target_angle_min=-0.15,
                target_angle_max=0.15,
                sparse_target_radius=0.30,
                target_curriculum_samples=45_000_000,
                gate_targets=(0.6,),
            ),
            PhaseG1Curriculum(
                name="one_meter_forward",
                iterations=200,
                target_distance_start=0.45,
                target_distance_end=1.00,
                target_angle_min=-0.15,
                target_angle_max=0.15,
                sparse_target_radius=0.30,
                target_curriculum_samples=70_000_000,
                env_overrides=(("w_sparse_command_success", 6.0), ("w_target_progress", 2.0), ("w_termination", -4.0)),
                ppo_overrides=(("actor_lr", 1.5e-4), ("critic_lr", 1.5e-4)),
                gate_targets=(0.6, 1.0),
            ),
            PhaseG1Curriculum(
                name="medium_forward_cone",
                iterations=180,
                target_distance_start=0.75,
                target_distance_end=1.40,
                target_angle_min=-0.30,
                target_angle_max=0.30,
                sparse_target_radius=0.30,
                target_curriculum_samples=80_000_000,
                env_overrides=(("w_sparse_command_success", 6.0), ("w_target_progress", 2.0), ("w_termination", -4.0)),
                ppo_overrides=(("actor_lr", 1.0e-4), ("critic_lr", 1.0e-4)),
                gate_targets=(0.6, 1.0, 1.4),
            ),
            PhaseG1Curriculum(
                name="long_forward_cone",
                iterations=180,
                target_distance_start=1.00,
                target_distance_end=2.00,
                target_angle_min=-0.45,
                target_angle_max=0.45,
                sparse_target_radius=0.30,
                target_curriculum_samples=90_000_000,
                env_overrides=(("w_sparse_command_success", 6.0), ("w_target_progress", 2.0), ("w_termination", -4.0)),
                ppo_overrides=(("actor_lr", 7.5e-5), ("critic_lr", 7.5e-5)),
                gate_targets=(0.6, 1.0, 1.4),
            ),
            PhaseG1Curriculum(
                name="strict_finish",
                iterations=120,
                target_distance_start=1.20,
                target_distance_end=2.40,
                target_angle_min=-0.50,
                target_angle_max=0.50,
                sparse_target_radius=0.25,
                target_curriculum_samples=80_000_000,
                env_overrides=(
                    ("w_sparse_command_success", 6.0),
                    ("w_target_progress", 2.0),
                    ("w_termination", -4.0),
                    ("w_mechanical_power", -1.0e-4),
                ),
                ppo_overrides=(("actor_lr", 5.0e-5), ("critic_lr", 5.0e-5)),
                gate_targets=(0.6, 1.0, 1.4, 2.0),
            ),
        ]
    else:
        raise ValueError("recipe must be 'simple-target' or 'advanced-target'")

    scaled = []
    for phase in phases:
        scaled_iterations = max(1, int(math.ceil(float(phase.iterations) * float(iteration_scale))))
        scaled.append(_sanitize_phase(replace(phase, iterations=scaled_iterations), min_target_margin))
    return scaled


def _sanitize_phase(phase: PhaseG1Curriculum, min_target_margin: float) -> PhaseG1Curriculum:
    radius = max(float(phase.sparse_target_radius), 1.0e-6)
    start = max(float(phase.target_distance_start), radius + float(min_target_margin))
    end = max(float(phase.target_distance_end), start)
    if phase.target_angle_max < phase.target_angle_min:
        raise ValueError(f"{phase.name}: target_angle_max must be >= target_angle_min")
    return replace(
        phase,
        target_distance_start=start,
        target_distance_end=end,
        sparse_target_radius=radius,
    )


def _merged_overrides(*items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in items:
        values.update(dict(group))
    return values


def build_env_config(phase: PhaseG1Curriculum, args: argparse.Namespace, *, reward_mode: str | None = None):
    """Build the environment configuration for one phase."""

    values = _merged_overrides(_BASE_TARGET_ENV_OVERRIDES, phase.env_overrides)
    values.update(
        {
            "world_count": int(args.world_count),
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
            "sparse_target_radius": float(phase.sparse_target_radius),
            "sparse_target_position": (float(phase.target_distance_start), 0.0),
        }
    )
    if reward_mode is not None:
        values["reward_mode"] = str(reward_mode)
    return g1_recipe.default_g1_env_config(**values)


def build_ppo_config(phase: PhaseG1Curriculum, args: argparse.Namespace):
    """Build the PPO configuration for one phase."""

    values = _merged_overrides(_BASE_TARGET_PPO_OVERRIDES, phase.ppo_overrides)
    if args.actor_lr is not None:
        values["actor_lr"] = float(args.actor_lr)
    if args.critic_lr is not None:
        values["critic_lr"] = float(args.critic_lr)
    if args.entropy_coeff is not None:
        values["entropy_coeff"] = float(args.entropy_coeff)
    if args.max_grad_norm is not None:
        values["max_grad_norm"] = float(args.max_grad_norm)
    return g1_recipe.default_g1_ppo_config(**values)


def build_train_config(
    phase: PhaseG1Curriculum,
    args: argparse.Namespace,
    *,
    seed: int,
    resume_checkpoint: str | None,
    checkpoint_path: Path,
) -> rl.ConfigTrainG1PPO:
    """Build the train configuration for one phase."""

    return rl.ConfigTrainG1PPO(
        iterations=int(phase.iterations),
        rollout_steps=int(args.rollout_steps),
        env_config=build_env_config(phase, args),
        ppo_config=build_ppo_config(phase, args),
        device=args.device,
        seed=int(seed),
        log_interval=int(args.log_interval),
        randomize_commands=False,
        use_target_curriculum=True,
        target_distance_start=float(phase.target_distance_start),
        target_distance_end=float(phase.target_distance_end),
        target_curriculum_samples=int(phase.target_curriculum_samples),
        target_curriculum_start_samples=0,
        randomize_target_positions=True,
        randomize_target_distance_range=True,
        target_angle_min=float(phase.target_angle_min),
        target_angle_max=float(phase.target_angle_max),
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


def _parse_eval_targets(raw: str) -> tuple[tuple[float, float], ...]:
    targets = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        if ":" in text:
            x_text, y_text = text.split(":", 1)
            targets.append((float(x_text), float(y_text)))
        else:
            targets.append((float(text), 0.0))
    if not targets:
        raise ValueError("at least one evaluation target is required")
    return tuple(targets)


def select_curriculum_phases(
    phases: list[PhaseG1Curriculum],
    *,
    start_phase: int,
    phase_count: int | None,
) -> list[tuple[int, PhaseG1Curriculum]]:
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


def check_phase_gate(phase: PhaseG1Curriculum, target_eval_stats: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Check whether a phase is good enough to feed the next phase."""

    failures: list[str] = []
    if not phase.gate_targets:
        return True, failures
    stats_by_x = {round(float(stat["target_position"][0]), 6): stat for stat in target_eval_stats}
    for target in phase.gate_targets:
        key = round(float(target), 6)
        stat = stats_by_x.get(key)
        if stat is None:
            failures.append(f"missing eval target x={target:g}")
            continue
        strict_success = float(stat["strict_success_fraction"])
        fall_fraction = float(stat["fall_fraction"])
        tilt_fraction = float(stat["tilt_violation_fraction"])
        if strict_success < float(phase.gate_min_strict_success_fraction):
            failures.append(
                f"x={target:g} strict_success={strict_success:.3f} < {phase.gate_min_strict_success_fraction:.3f}"
            )
        if fall_fraction > float(phase.gate_max_fall_fraction):
            failures.append(f"x={target:g} fall_fraction={fall_fraction:.3f} > {phase.gate_max_fall_fraction:.3f}")
        if tilt_fraction > float(phase.gate_max_tilt_violation_fraction):
            failures.append(
                f"x={target:g} tilt_violation_fraction={tilt_fraction:.3f} "
                f"> {phase.gate_max_tilt_violation_fraction:.3f}"
            )
    return not failures, failures


def evaluate_checkpoint(
    checkpoint: Path,
    phase: PhaseG1Curriculum,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Evaluate a checkpoint on sparse target metrics."""

    trainer = rl.load_ppo_checkpoint(str(checkpoint), device=args.device)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.world_count = int(args.eval_world_count)
    env_config = build_env_config(phase, eval_args, reward_mode="sparse_target")
    result = rl.evaluate_g1_target_ppo(
        trainer,
        rl.ConfigEvaluateG1TargetPPO(
            env_config=env_config,
            target_positions=_parse_eval_targets(str(args.eval_targets)),
            steps=int(args.eval_steps),
            device=args.device,
            deterministic=not bool(args.stochastic_eval),
            seed=int(args.seed) + 911,
            max_tilt_degrees=float(args.max_tilt_degrees),
        ),
    )
    return [asdict(stat) for stat in result.stats]


def run_curriculum(args: argparse.Namespace) -> list[PhaseG1Result]:
    """Train all phases in the selected curriculum."""

    phases = build_curriculum(
        str(args.recipe),
        iteration_scale=float(args.iteration_scale),
        min_target_margin=float(args.min_target_margin),
    )
    selected_phases = select_curriculum_phases(
        phases,
        start_phase=int(args.start_phase),
        phase_count=args.phase_count,
    )
    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "recipe": args.recipe,
                    "phases": [{"index": index, **asdict(phase)} for index, phase in selected_phases],
                },
                indent=2,
            )
        )
        return []
    if int(args.start_phase) > 0 and args.resume_checkpoint is None:
        raise ValueError("--resume-checkpoint is required when --start-phase is greater than zero")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[PhaseG1Result] = []
    resume_checkpoint = args.resume_checkpoint
    summary_path = output_dir / "summary.json"
    for phase_index, phase in selected_phases:
        checkpoint_pattern = output_dir / f"{phase_index:02d}_{phase.name}_{{iteration}}.npz"
        train_seed = int(args.seed) + phase_index * 10_003
        train_config = build_train_config(
            phase,
            args,
            seed=train_seed,
            resume_checkpoint=resume_checkpoint,
            checkpoint_path=checkpoint_pattern,
        )
        t0 = time.perf_counter()
        train_result = rl.train_g1_ppo(train_config)
        elapsed = time.perf_counter() - t0
        checkpoint = _format_checkpoint_path(checkpoint_pattern, int(train_result.trainer.iteration))
        final_stats = asdict(train_result.history[-1]) if train_result.history else {}
        eval_stats = [] if bool(args.no_eval) else evaluate_checkpoint(checkpoint, phase, args)
        gate_passed, gate_failures = (True, [])
        if not bool(args.no_phase_gates):
            if bool(args.no_eval):
                gate_passed = False
                gate_failures = ["phase gates require evaluation; remove --no-eval or pass --no-phase-gates"]
            else:
                gate_passed, gate_failures = check_phase_gate(phase, eval_stats)
        phase_result = PhaseG1Result(
            phase=phase,
            checkpoint=str(checkpoint),
            elapsed_seconds=float(elapsed),
            final_train_stats=final_stats,
            target_eval_stats=eval_stats,
            phase_gate_passed=bool(gate_passed),
            phase_gate_failures=list(gate_failures),
        )
        results.append(phase_result)
        resume_checkpoint = str(checkpoint)
        summary_path.write_text(
            json.dumps(
                {
                    "recipe": args.recipe,
                    "output_dir": str(output_dir),
                    "results": [asdict(result) for result in results],
                },
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
    parser.add_argument("--recipe", choices=("simple-target", "advanced-target"), default="simple-target")
    parser.add_argument("--output-dir", default="/tmp/phoenx_g1_curriculum")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=34_001)
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--iteration-scale", type=float, default=1.0)
    parser.add_argument("--min-target-margin", type=float, default=0.10)
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
    parser.add_argument(
        "--observation-mode",
        choices=("nanog1", "isaaclab_flat"),
        default=g1_recipe.OBSERVATION_MODE,
    )
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
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-phase-gates", action="store_true")
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--eval-targets", default="0.6,1.0,1.4,2.0")
    parser.add_argument("--eval-world-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=350)
    parser.add_argument("--max-tilt-degrees", type=float, default=30.0)
    parser.add_argument("--stochastic-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    run_curriculum(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
