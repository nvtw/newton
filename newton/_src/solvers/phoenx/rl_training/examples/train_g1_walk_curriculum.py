# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-phase G1 walk curriculum for PhoenX — inspired by both nanoG1 and the Anymal curriculum.

Why a curriculum?
-----------------
Training directly at the nanoG1 speed (0.8 m/s) with pure velocity-tracking
rewards causes the policy to oscillate between good and bad gaits, because
the policy entropy collapses before the value function is accurate enough to
guide it past local reward optima (standing still also gets the alive bonus).

The Anymal curriculum addresses this with a progressive speed increase.  For
G1 we use a similar two-phase strategy:

Phase 1 — gait_shape
    Fixed slow command (--phase1-command-x, default 0.4 m/s) plus explicit
    gait-shaping rewards (base height, gait contact, gait swing, joint hip
    penalty).  These shape the body into an alternating walking gait with
    correct hip height BEFORE the pure nanoG1 reward takes over.
    Higher entropy (entropy_coeff=5e-5), Adam optimizer at lr=2e-3 for
    stable early exploration.

Phase 2 — nanog1_speed
    Pure nanoG1 reward (tracking only, no shaping) at the target speed
    (--phase2-command-x, default 0.8 m/s), resumed from Phase 1.  Muon
    optimizer at lr=2e-2 with mirror loss 0.25, matching nanoG1.

Both phases write checkpoints to --output-dir and print per-iteration stats
including the samples-to-walk estimate.

Quickstart:

    uv run --extra dev -m newton._src.solvers.phoenx.rl_training.examples.train_g1_walk_curriculum \\
        --device cuda:0 --output-dir /tmp/g1_curriculum

Resume from a Phase-1 checkpoint:

    uv run --extra dev -m newton._src.solvers.phoenx.rl_training.examples.train_g1_walk_curriculum \\
        --device cuda:0 --output-dir /tmp/g1_curriculum --start-phase 1 \\
        --resume-checkpoint /tmp/g1_curriculum/phase0_gait_shape_000200.npz

Run only Phase 1 (e.g. to tune gait shaping before committing to Phase 2):

    ... --phase-count 1

Run only Phase 2 with an existing Phase-1 checkpoint:

    ... --start-phase 1 --resume-checkpoint <path>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe

_WALK_GATE_TRACKING_PERF = 0.30
_WALK_GATE_SAMPLES = 75_000_000

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseG1Walk:
    """One stage of G1 walk curriculum training."""

    name: str
    iterations: int
    command_x: float
    env_overrides: tuple[tuple[str, Any], ...]
    ppo_overrides: tuple[tuple[str, Any], ...]
    gate_min_tracking_perf: float = 0.20
    gate_max_fall_fraction: float = 0.10


# Phase 1 env: nanoG1 base reward plus gait/height shaping to lock in a stable gait.
# Reduced w_alive so standing still is less rewarding than actually walking.
_GAIT_SHAPE_ENV: tuple[tuple[str, Any], ...] = (
    ("reward_mode", "nanog1_dense"),
    ("w_alive", 1.0),  # reduced: don't over-reward standing
    ("w_termination", -2.0),
    ("w_track_lin", 3.0),  # slightly stronger tracking signal
    ("w_track_ang", 1.0),
    ("w_lin_vel_z", -2.0),
    ("w_ang_vel_xy", -1.3),
    ("w_orientation", -10.0),
    ("w_torque", -2.0e-5),
    ("w_action_rate", -0.01),
    ("w_command_progress", 0.0),
    # Gait shaping
    ("w_gait_contact", 0.4),  # reward stance/swing timing
    ("w_gait_swing", -10.0),  # penalise foot in air during stance
    ("w_gait_swing_contact", 0.0),
    ("w_gait_hip", -2.0),  # keep hips near neutral
    ("gait_foot_height", 0.06),
    ("w_base_height", -5.0),  # stay near target height
    ("base_height_target", 0.78),
    # Joint regularisation
    ("w_joint_deviation_hip", -0.05),
    ("w_joint_deviation_waist", -0.01),
    ("w_joint_deviation_upper", -0.005),
    ("w_joint_acc_legs", 0.0),
    ("w_joint_pos_limit_ankle", -0.2),
    # Feet quality
    ("w_feet_air_time", 0.3),
    ("feet_air_time_threshold", 0.3),
    ("w_feet_slide", -0.05),
    ("w_mechanical_power", -5.0e-5),
)

# Phase 1 PPO: Adam with moderate LR, higher entropy for better exploration.
_GAIT_SHAPE_PPO: tuple[tuple[str, Any], ...] = (
    ("optimizer", "adam"),
    ("actor_lr", 2.0e-3),
    ("critic_lr", 2.0e-3),
    ("entropy_coeff", 5.0e-5),
    ("max_grad_norm", 0.5),
    ("reward_clip", 4.0),
    ("mirror_loss_coeff", 0.25),
    ("anneal_lr", False),
    ("replay_ratio", 3.0),
    ("priority_alpha", 0.4),
    ("priority_beta", 1.0),
    ("vtrace_rho_clip", 3.0),
    ("vtrace_c_clip", 3.0),
    ("minibatch_size", g1_recipe.MINIBATCH_SIZE),
    ("value_loss_coeff", 0.5),
    ("value_clip_range", 20.0),
    ("manual_actor_backward", g1_recipe.MANUAL_ACTOR_BACKWARD),
    ("manual_critic_backward", g1_recipe.MANUAL_CRITIC_BACKWARD),
    ("manual_mlp_weight_grad_dtype", g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE),
    ("manual_mlp_forward_dtype", g1_recipe.MANUAL_MLP_FORWARD_DTYPE),
    ("policy_network", g1_recipe.POLICY_NETWORK),
)

# Phase 2 env: pure nanoG1 reward, no shaping — match nanoG1's exact recipe.
_NANOG1_SPEED_ENV: tuple[tuple[str, Any], ...] = (
    ("reward_mode", "nanog1_dense"),
    ("w_alive", g1_recipe.W_ALIVE),
    ("w_termination", g1_recipe.W_TERMINATION),
    ("w_track_lin", g1_recipe.W_TRACK_LIN),
    ("w_track_ang", g1_recipe.W_TRACK_ANG),
    ("w_lin_vel_z", g1_recipe.W_LIN_VEL_Z),
    ("w_ang_vel_xy", g1_recipe.W_ANG_VEL_XY),
    ("w_orientation", g1_recipe.W_ORIENTATION),
    ("w_torque", g1_recipe.W_TORQUE),
    ("w_action_rate", g1_recipe.W_ACTION_RATE),
    ("w_command_progress", 0.0),
    ("w_gait_contact", 0.0),
    ("w_gait_swing", 0.0),
    ("w_gait_swing_contact", 0.0),
    ("w_gait_hip", 0.0),
    ("w_base_height", 0.0),
    ("w_feet_air_time", 0.0),
    ("w_feet_slide", 0.0),
    ("w_joint_deviation_hip", 0.0),
    ("w_joint_deviation_waist", 0.0),
    ("w_joint_deviation_upper", 0.0),
    ("w_joint_acc_legs", 0.0),
    ("w_joint_pos_limit_ankle", 0.0),
    ("w_mechanical_power", 0.0),
)

# Phase 2 PPO: Adam fine-tune at nanoG1 speed with fresh optimizer state, mirror loss.
# We use resume_policy_only=True (always reset optimizer on phase transition) so that
# Phase 1's momentum — tuned for stability at 0.4 m/s — does not oppose the
# "go faster" gradients in Phase 2.  lr=5e-3 (2.5x Phase 1) gives enough signal
# to adapt to 0.8 m/s while staying stable.
# Muon at lr=2e-2 causes policy divergence via recurrent-state amplification;
# Adam at lr=1e-3 with stale Phase-1 momentum is too conservative (clip_frac < 1%).
_NANOG1_SPEED_PPO: tuple[tuple[str, Any], ...] = (
    ("optimizer", "adam"),
    ("actor_lr", 5.0e-3),
    ("critic_lr", 5.0e-3),
    ("anneal_lr", True),
    ("lr_anneal_timesteps", 97_779_712),  # 373 iters x 262144 samples/iter
    ("min_lr_ratio", 0.0),
    ("entropy_coeff", g1_recipe.ENTROPY_COEFF),
    ("max_grad_norm", g1_recipe.MAX_GRAD_NORM),
    ("reward_clip", g1_recipe.REWARD_CLIP),
    ("mirror_loss_coeff", g1_recipe.MIRROR_LOSS_COEFF),
    ("replay_ratio", g1_recipe.REPLAY_RATIO),
    ("priority_alpha", g1_recipe.PRIORITY_ALPHA),
    ("priority_beta", g1_recipe.PRIORITY_BETA),
    ("vtrace_rho_clip", g1_recipe.VTRACE_RHO_CLIP),
    ("vtrace_c_clip", g1_recipe.VTRACE_C_CLIP),
    ("minibatch_size", g1_recipe.MINIBATCH_SIZE),
    ("value_loss_coeff", g1_recipe.VALUE_LOSS_COEFF),
    ("value_clip_range", g1_recipe.VALUE_CLIP_RANGE),
    ("manual_actor_backward", g1_recipe.MANUAL_ACTOR_BACKWARD),
    ("manual_critic_backward", g1_recipe.MANUAL_CRITIC_BACKWARD),
    ("manual_mlp_weight_grad_dtype", g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE),
    ("manual_mlp_forward_dtype", g1_recipe.MANUAL_MLP_FORWARD_DTYPE),
    ("policy_network", g1_recipe.POLICY_NETWORK),
)


def _default_phases(args: argparse.Namespace) -> list[PhaseG1Walk]:
    return [
        PhaseG1Walk(
            name="gait_shape",
            iterations=int(args.phase1_iterations),
            command_x=float(args.phase1_command_x),
            env_overrides=_GAIT_SHAPE_ENV,
            ppo_overrides=_GAIT_SHAPE_PPO,
            gate_min_tracking_perf=0.20,
            gate_max_fall_fraction=0.12,
        ),
        PhaseG1Walk(
            name="nanog1_speed",
            iterations=int(args.phase2_iterations),
            command_x=float(args.phase2_command_x),
            env_overrides=_NANOG1_SPEED_ENV,
            ppo_overrides=_NANOG1_SPEED_PPO,
            gate_min_tracking_perf=0.30,
            gate_max_fall_fraction=0.06,
        ),
    ]


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _build_env_config(phase: PhaseG1Walk, args: argparse.Namespace) -> rl.ConfigEnvG1PhoenX:
    overrides = dict(phase.env_overrides)
    overrides.update(
        world_count=int(args.world_count),
        command=(float(phase.command_x), 0.0, 0.0),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        ground_friction=float(args.ground_friction),
        actuation_model=g1_recipe.ACTUATION_MODEL,
        action_scale=g1_recipe.ACTION_SCALE,
        controlled_action_count=g1_recipe.CONTROLLED_ACTION_COUNT,
        contact_geometry=g1_recipe.CONTACT_GEOMETRY,
        joint_friction_model=g1_recipe.JOINT_FRICTION_MODEL,
    )
    return g1_recipe.default_g1_env_config(**overrides)


def _build_ppo_config(phase: PhaseG1Walk) -> rl.ConfigPPO:
    overrides = dict(phase.ppo_overrides)
    return g1_recipe.default_g1_ppo_config(**overrides)


def _build_train_config(
    phase: PhaseG1Walk,
    args: argparse.Namespace,
    *,
    seed: int,
    resume_checkpoint: str | None,
    checkpoint_path: str,
    resume_policy_only: bool = False,
) -> rl.ConfigTrainG1PPO:
    return rl.ConfigTrainG1PPO(
        iterations=int(phase.iterations),
        rollout_steps=int(args.rollout_steps),
        env_config=_build_env_config(phase, args),
        ppo_config=_build_ppo_config(phase),
        device=args.device,
        seed=int(seed),
        log_interval=int(args.log_interval),
        randomize_commands=False,
        reset_recurrent_state_on_rollout_start=g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
        resume_checkpoint=resume_checkpoint,
        resume_policy_only=resume_policy_only,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=int(args.checkpoint_interval),
        readback_diagnostics=not bool(args.no_readback_diagnostics),
        execution_mode=str(args.execution_mode),
    )


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _check_gate(
    stats_history: list[Any],
    phase: PhaseG1Walk,
) -> tuple[bool, list[str]]:
    """Check the last few iterations' mean tracking perf and fall rate."""

    if not stats_history:
        return False, ["no training history"]

    window = stats_history[-min(10, len(stats_history)) :]
    mean_perf = sum(s.mean_tracking_perf for s in window) / len(window)
    mean_done = sum(s.mean_done for s in window) / len(window)

    failures = []
    if mean_perf < float(phase.gate_min_tracking_perf):
        failures.append(
            f"mean_tracking_perf={mean_perf:.3f} < gate={phase.gate_min_tracking_perf:.3f} (last {len(window)} iters)"
        )
    if mean_done > float(phase.gate_max_fall_fraction):
        failures.append(
            f"mean_done={mean_done:.4f} > gate={phase.gate_max_fall_fraction:.4f} (last {len(window)} iters)"
        )
    return not failures, failures


# ---------------------------------------------------------------------------
# Curriculum runner
# ---------------------------------------------------------------------------


def run_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    phases = _default_phases(args)
    selected = phases[int(args.start_phase) :]
    if args.phase_count is not None:
        selected = selected[: int(args.phase_count)]
    if not selected:
        raise ValueError("No phases selected")
    if int(args.start_phase) > 0 and args.resume_checkpoint is None:
        raise ValueError("--resume-checkpoint is required when --start-phase > 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    samples_per_iter = int(args.rollout_steps) * int(args.world_count)
    total_iters = sum(p.iterations for p in selected)
    total_samples = total_iters * samples_per_iter
    walk_gate_iter = max(1, _WALK_GATE_SAMPLES // samples_per_iter)

    print(
        f"G1 walk curriculum\n"
        f"  phases={[p.name for p in selected]}\n"
        f"  worlds={args.world_count}  rollout={args.rollout_steps}  "
        f"samples_per_iter={samples_per_iter:,}\n"
        f"  total_iterations={total_iters}  total_samples={total_samples / 1e6:.1f}M\n"
        f"  walk_gate_at_iter≈{walk_gate_iter} ({_WALK_GATE_SAMPLES / 1e6:.0f}M samples cumulative)\n"
        f"  execution={args.execution_mode}"
    )

    resume_checkpoint = args.resume_checkpoint
    prev_phase: PhaseG1Walk | None = None if int(args.start_phase) == 0 else phases[int(args.start_phase) - 1]
    phase_results = []
    t_global_start = time.perf_counter()
    cumulative_samples = (
        int(args.start_phase) * sum(p.iterations for p in phases[: int(args.start_phase)]) * samples_per_iter
    )

    for phase_local_idx, phase in enumerate(selected, start=int(args.start_phase)):
        checkpoint_pattern = str(output_dir / f"phase{phase_local_idx}_{phase.name}_{{iteration:06d}}.npz")
        t0 = time.perf_counter()

        # Always load policy weights only (fresh optimizer) on phase transitions.
        # Carrying over Phase N's optimizer momentum into Phase N+1 causes the
        # accumulated gradient direction from the previous task to oppose the new
        # reward signal, producing either too-conservative updates (Adam) or
        # divergence (Muon).  A fresh optimizer starts with zero momentum, letting
        # Phase 2's reward gradients drive updates from the first iteration.
        policy_only = resume_checkpoint is not None and prev_phase is not None

        print(
            f"\n--- Phase {phase_local_idx}: {phase.name} "
            f"command_x={phase.command_x:.2f}  iterations={phase.iterations}  "
            f"resume={resume_checkpoint or '-'}" + ("  (policy-only, optimizer reset)" if policy_only else "") + " ---"
        )

        train_cfg = _build_train_config(
            phase,
            args,
            seed=int(args.seed) + phase_local_idx * 10_003,
            resume_checkpoint=resume_checkpoint,
            checkpoint_path=checkpoint_pattern,
            resume_policy_only=policy_only,
        )
        prev_phase = phase
        result = rl.train_g1_ppo(train_cfg)

        elapsed = time.perf_counter() - t0
        final_iter = int(result.trainer.iteration)
        phase_samples = phase.iterations * samples_per_iter
        cumulative_samples += phase_samples
        avg_sps = phase_samples / max(elapsed, 1.0e-9)
        t_walk_est = _WALK_GATE_SAMPLES / avg_sps if avg_sps > 0 else float("inf")

        final_stats = result.history[-1] if result.history else None
        gate_passed, gate_failures = _check_gate(result.history, phase)

        # Save final checkpoint for this phase
        final_checkpoint = str(output_dir / f"phase{phase_local_idx}_{phase.name}_{final_iter:06d}.npz")
        result.trainer.save_checkpoint(final_checkpoint, iteration=final_iter)

        phase_payload = {
            "phase_index": phase_local_idx,
            "phase_name": phase.name,
            "command_x": float(phase.command_x),
            "iterations": phase.iterations,
            "elapsed_seconds": float(elapsed),
            "samples_per_second": float(avg_sps),
            "estimated_t_walk_seconds": float(t_walk_est),
            "cumulative_samples": int(cumulative_samples),
            "final_checkpoint": final_checkpoint,
            "gate_passed": bool(gate_passed),
            "gate_failures": list(gate_failures),
            "final_stats": asdict(final_stats) if final_stats is not None else {},
        }
        phase_results.append(phase_payload)
        summary_path.write_text(json.dumps({"phases": phase_results}, indent=2, sort_keys=True) + "\n")

        walk_status = ""
        if final_stats is not None:
            walk_ok = final_stats.mean_tracking_perf >= _WALK_GATE_TRACKING_PERF
            walk_status = (
                "WALKED"
                if walk_ok
                else f"perf={final_stats.mean_tracking_perf:.3f} (gate={_WALK_GATE_TRACKING_PERF:.2f})"
            )

        print(
            f"Phase {phase_local_idx} done: elapsed={elapsed:.1f}s  "
            f"avg_sps={avg_sps:.0f}  est_T_walk={t_walk_est:.1f}s\n"
            f"  gate={'PASS' if gate_passed else 'FAIL'}  {', '.join(gate_failures) or 'ok'}\n"
            f"  walk_status={walk_status}\n"
            f"  checkpoint={final_checkpoint}"
        )

        resume_checkpoint = final_checkpoint

        if not gate_passed and not bool(args.allow_gate_failure):
            raise RuntimeError(f"Phase {phase_local_idx} '{phase.name}' failed quality gate: {gate_failures}")

    total_elapsed = time.perf_counter() - t_global_start
    total_avg_sps = cumulative_samples / max(total_elapsed, 1.0e-9)
    t_walk_overall = _WALK_GATE_SAMPLES / total_avg_sps if total_avg_sps > 0 else float("inf")

    final_phase_stats = phase_results[-1].get("final_stats", {})
    final_perf = float(final_phase_stats.get("mean_tracking_perf", 0.0))

    print(
        f"\n=== G1 CURRICULUM RESULT ===\n"
        f"  total_elapsed={total_elapsed:.1f}s  total_samples={cumulative_samples / 1e6:.1f}M\n"
        f"  overall_avg_sps={total_avg_sps:.0f}  est_T_walk={t_walk_overall:.1f}s\n"
        f"  final_perf={final_perf:.3f}  "
        f"walk_gate={'PASSED' if final_perf >= _WALK_GATE_TRACKING_PERF else 'NOT YET'}\n"
        f"  final_checkpoint={resume_checkpoint}\n"
        f"  summary={summary_path}\n"
        "=== END ==="
    )

    payload = {
        "total_elapsed_seconds": float(total_elapsed),
        "total_samples": int(cumulative_samples),
        "overall_avg_sps": float(total_avg_sps),
        "estimated_t_walk_seconds": float(t_walk_overall),
        "final_perf": float(final_perf),
        "walk_gate_passed": bool(final_perf >= _WALK_GATE_TRACKING_PERF),
        "final_checkpoint": resume_checkpoint,
        "summary": str(summary_path),
        "phases": phase_results,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "phases"}, sort_keys=True))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="/tmp/g1_walk_curriculum")
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "graph_leapfrog"),
        default="graph_leapfrog",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=0,
        help="0 = start from Phase 1 (gait_shape); 1 = start from Phase 2 (nanog1_speed)",
    )
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument(
        "--phase1-iterations",
        type=int,
        default=200,
        help="Phase 1 (gait_shape) iteration count (default: %(default)s)",
    )
    parser.add_argument(
        "--phase1-command-x",
        type=float,
        default=0.4,
        help="Phase 1 forward speed command [m/s] (default: %(default)s)",
    )
    parser.add_argument(
        "--phase2-iterations",
        type=int,
        default=373,
        help=("Phase 2 (nanog1_speed) iteration count (default: %(default)s ≈ 75M samples at 4096x64 after Phase 1)"),
    )
    parser.add_argument(
        "--phase2-command-x",
        type=float,
        default=0.8,
        help="Phase 2 forward speed command [m/s] (default: %(default)s, same as nanoG1)",
    )
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--ground-friction", type=float, default=g1_recipe.GROUND_FRICTION)
    parser.add_argument("--no-readback-diagnostics", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    run_curriculum(_make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
