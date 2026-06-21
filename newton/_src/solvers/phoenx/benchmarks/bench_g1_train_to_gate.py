# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX G1 PPO train-to-gate progress.

This benchmark mirrors nanoG1's user-visible lifecycle: train from scratch,
write a checkpoint, reload it, and run the frozen G1 walking quality gate. It can
run a short smoke or continue chunks until the policy passes the gate.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate \
        --max-iterations 2 --chunk-iterations 1 --execution-mode graph_leapfrog \
        --battery-steps 1 --diagnostic-steps 1 --diagnostic-world-count 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.benchmarks.bench_g1_train import (
    _g1_ppo_config,
    _parse_hidden_layers,
    _parse_int_or_auto,
)
from newton._src.solvers.phoenx.rl_training import g1_recipe

_NANOG1_WALK_SAMPLES = 75_000_000
_NANOG1_WALK_SECONDS = 58.9
_NANOG1_WALK_SPS = 1_276_000.0
_NANOG1_REFERENCE = "nanoG1 RESULTS.md time-to-walk table"


def _format_checkpoint_template(path: str, iteration: int) -> Path:
    if "{iteration}" in path:
        return Path(path.format(iteration=int(iteration)))
    return Path(path)


def _make_env_config(args: argparse.Namespace, *, world_count: int | None = None) -> rl.ConfigEnvG1PhoenX:
    return rl.ConfigEnvG1PhoenX(
        world_count=int(args.world_count if world_count is None else world_count),
        command=(float(args.command_x), float(args.command_y), float(args.command_yaw)),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        controlled_action_count=int(args.controlled_action_count),
        parse_meshes=bool(args.parse_meshes),
        contact_geometry=str(getattr(args, "contact_geometry", g1_recipe.CONTACT_GEOMETRY)),
        rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
        threads_per_world=args.threads_per_world,
        multi_world_scheduler=str(args.multi_world_scheduler),
        prepare_refresh_stride=args.prepare_refresh_stride,
    )


def _make_ppo_config(args: argparse.Namespace) -> rl.ConfigPPO:
    return _g1_ppo_config(
        args.train_epochs,
        args.mirror_loss_coeff,
        args.minibatch_size,
        args.replay_ratio,
        args.priority_alpha,
        args.priority_beta,
        not args.no_manual_actor_backward,
        not args.no_manual_critic_backward,
        args.manual_mlp_weight_grad_dtype,
        args.manual_mlp_forward_dtype,
        args.vtrace_rho_clip,
        args.vtrace_c_clip,
        args.reward_clip,
        args.max_grad_norm,
        args.value_loss_coeff,
        args.value_clip_range,
    )


def _samples_for_iteration(args: argparse.Namespace, iteration: int) -> int:
    return int(iteration) * int(args.world_count) * int(args.rollout_steps)


def benchmark_train_to_gate(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("PhoenX G1 train-to-gate benchmark requires CUDA with Warp mempool enabled")
    if args.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if args.chunk_iterations <= 0:
        raise ValueError("chunk_iterations must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if args.target_samples <= 0:
        raise ValueError("target_samples must be positive")

    checkpoint_template = args.checkpoint_path or "/tmp/phoenx_g1_gate_{iteration}.npz"
    env_config = _make_env_config(args)
    ppo_config = _make_ppo_config(args)

    train_seconds = 0.0
    gate_seconds = 0.0
    completed_iterations = 0
    resume_checkpoint: str | None = None
    gate_history: list[dict[str, Any]] = []
    train_history: list[dict[str, Any]] = []
    first_pass: dict[str, Any] | None = None
    total_t0 = time.perf_counter()

    while completed_iterations < int(args.max_iterations):
        chunk_iterations = min(int(args.chunk_iterations), int(args.max_iterations) - completed_iterations)
        chunk_t0 = time.perf_counter()
        result = rl.train_g1_ppo(
            rl.ConfigTrainG1PPO(
                iterations=chunk_iterations,
                rollout_steps=int(args.rollout_steps),
                hidden_layers=tuple(args.hidden_layers),
                env_config=env_config,
                ppo_config=ppo_config,
                device=device,
                seed=int(args.seed),
                log_interval=0,
                randomize_commands=not bool(args.no_command_randomization),
                command_x_range=tuple(float(v) for v in args.command_x_range),
                command_y_range=tuple(float(v) for v in args.command_y_range),
                command_yaw_range=tuple(float(v) for v in args.command_yaw_range),
                squash_actions=bool(args.squash_actions),
                resume_checkpoint=resume_checkpoint,
                checkpoint_path=checkpoint_template,
                checkpoint_interval=0,
                readback_diagnostics=bool(args.readback_diagnostics),
                execution_mode=str(args.execution_mode),
            )
        )
        train_seconds += time.perf_counter() - chunk_t0
        completed_iterations = int(result.trainer.iteration)
        checkpoint_path = _format_checkpoint_template(checkpoint_template, completed_iterations)
        resume_checkpoint = str(checkpoint_path)
        train_history.extend(asdict(item) for item in result.history)

        gate_t0 = time.perf_counter()
        reloaded = rl.load_ppo_checkpoint(checkpoint_path, device=device)
        gate_result = rl.evaluate_g1_gate_ppo(
            reloaded,
            rl.ConfigEvaluateG1GatePPO(
                env_config=_make_env_config(args),
                battery_steps=int(args.battery_steps),
                seeds_per_command=int(args.seeds_per_command),
                diagnostic_steps=int(args.diagnostic_steps),
                diagnostic_world_count=int(args.diagnostic_world_count),
                device=device,
                deterministic=not bool(args.stochastic_gate),
                seed=int(args.gate_seed),
                max_battery_falls=int(args.max_battery_falls),
                min_battery_perf=float(args.min_battery_perf),
                max_action_jerk_rms=float(args.max_action_jerk_rms),
                max_ang_vel_xy_rms=float(args.max_ang_vel_xy_rms),
                max_yaw_rate_rms=float(args.max_yaw_rate_rms),
                max_leg_qvel_rms=float(args.max_leg_qvel_rms),
            ),
        )
        gate_seconds += time.perf_counter() - gate_t0
        samples = _samples_for_iteration(args, completed_iterations)
        gate_entry = {
            "iteration": int(completed_iterations),
            "samples": int(samples),
            "checkpoint": str(checkpoint_path),
            "stats": asdict(gate_result.stats),
        }
        gate_history.append(gate_entry)
        if gate_result.stats.pass_gate and first_pass is None:
            first_pass = gate_entry
            if not bool(args.keep_going_after_pass):
                break

    total_seconds = max(time.perf_counter() - total_t0, 1.0e-12)
    trained_samples = _samples_for_iteration(args, completed_iterations)
    train_sps = float(trained_samples) / max(train_seconds, 1.0e-12)
    total_sps = float(trained_samples) / total_seconds
    estimated_target_train_seconds = float(args.target_samples) / max(train_sps, 1.0e-12)
    estimated_target_total_seconds = float(args.target_samples) / max(total_sps, 1.0e-12)
    pass_gate = first_pass is not None

    result = {
        "engine": "phoenx_g1_warp_ppo_train_to_gate",
        "metric": "train, save, reload, and evaluate G1 quality gate",
        "device": device.name,
        "execution_mode": str(args.execution_mode),
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "squash_actions": bool(args.squash_actions),
        "value_loss_coeff": float(args.value_loss_coeff),
        "value_clip_range": float(args.value_clip_range),
        "max_iterations": int(args.max_iterations),
        "chunk_iterations": int(args.chunk_iterations),
        "completed_iterations": int(completed_iterations),
        "trained_samples": int(trained_samples),
        "target_samples": int(args.target_samples),
        "train_seconds": float(train_seconds),
        "gate_seconds": float(gate_seconds),
        "total_wall_seconds": float(total_seconds),
        "train_env_samples_per_s": float(train_sps),
        "total_env_samples_per_s": float(total_sps),
        "estimated_target_train_seconds": float(estimated_target_train_seconds),
        "estimated_target_total_seconds": float(estimated_target_total_seconds),
        "checkpoint_path": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "pass_gate": bool(pass_gate),
        "first_pass": first_pass,
        "gate_history": gate_history,
        "train_history": train_history if bool(args.include_train_history) else [],
        "nanog1_reference_source": _NANOG1_REFERENCE,
        "nanog1_walk_samples": _NANOG1_WALK_SAMPLES,
        "nanog1_walk_seconds": _NANOG1_WALK_SECONDS,
        "nanog1_walk_env_samples_per_s": _NANOG1_WALK_SPS,
        "phoenx_estimated_target_slowdown_vs_nanog1": estimated_target_train_seconds / _NANOG1_WALK_SECONDS,
        "phoenx_train_sps_over_nanog1": train_sps / _NANOG1_WALK_SPS,
    }
    if args.fail_on_miss and not pass_gate:
        result["error"] = "quality gate was not reached within max_iterations"
    return result


def _default_max_iterations(world_count: int, rollout_steps: int, target_samples: int) -> int:
    samples_per_iteration = int(world_count) * int(rollout_steps)
    return max(1, int(math.ceil(float(target_samples) / float(samples_per_iteration))))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--target-samples", type=int, default=_NANOG1_WALK_SAMPLES)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--chunk-iterations", type=int, default=25)
    parser.add_argument("--hidden-layers", type=_parse_hidden_layers, default=g1_recipe.HIDDEN_LAYERS)
    parser.add_argument("--train-epochs", type=int, default=g1_recipe.TRAIN_EPOCHS)
    parser.add_argument("--mirror-loss-coeff", type=float, default=g1_recipe.MIRROR_LOSS_COEFF)
    parser.add_argument("--minibatch-size", type=int, default=g1_recipe.MINIBATCH_SIZE)
    parser.add_argument("--replay-ratio", type=float, default=g1_recipe.REPLAY_RATIO)
    parser.add_argument("--priority-alpha", type=float, default=g1_recipe.PRIORITY_ALPHA)
    parser.add_argument("--priority-beta", type=float, default=g1_recipe.PRIORITY_BETA)
    parser.add_argument("--no-manual-actor-backward", action="store_true")
    parser.add_argument("--no-manual-critic-backward", action="store_true")
    parser.add_argument(
        "--manual-mlp-weight-grad-dtype",
        choices=("float32", "bfloat16"),
        default=g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE,
    )
    parser.add_argument(
        "--manual-mlp-forward-dtype",
        choices=("float32", "bfloat16"),
        default=g1_recipe.MANUAL_MLP_FORWARD_DTYPE,
    )
    parser.add_argument("--vtrace-rho-clip", type=float, default=g1_recipe.VTRACE_RHO_CLIP)
    parser.add_argument("--vtrace-c-clip", type=float, default=g1_recipe.VTRACE_C_CLIP)
    parser.add_argument("--reward-clip", type=float, default=g1_recipe.REWARD_CLIP)
    parser.add_argument("--max-grad-norm", type=float, default=g1_recipe.MAX_GRAD_NORM)
    parser.add_argument("--value-loss-coeff", type=float, default=g1_recipe.VALUE_LOSS_COEFF)
    parser.add_argument("--value-clip-range", type=float, default=g1_recipe.VALUE_CLIP_RANGE)
    parser.add_argument(
        "--squash-actions",
        action=argparse.BooleanOptionalAction,
        default=g1_recipe.SQUASH_ACTIONS,
        help="Use tanh-squashed PPO actions instead of the nanoG1-compatible raw Gaussian policy.",
    )
    parser.add_argument("--command-x", type=float, default=g1_recipe.COMMAND[0])
    parser.add_argument("--command-y", type=float, default=g1_recipe.COMMAND[1])
    parser.add_argument("--command-yaw", type=float, default=g1_recipe.COMMAND[2])
    parser.add_argument("--command-x-range", type=float, nargs=2, default=g1_recipe.COMMAND_X_RANGE)
    parser.add_argument("--command-y-range", type=float, nargs=2, default=g1_recipe.COMMAND_Y_RANGE)
    parser.add_argument("--command-yaw-range", type=float, nargs=2, default=g1_recipe.COMMAND_YAW_RANGE)
    parser.add_argument("--no-command-randomization", action="store_true")
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY)
    parser.add_argument("--rigid-contact-max-per-world", type=int, default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
    parser.add_argument("--threads-per-world", type=_parse_int_or_auto, default=g1_recipe.THREADS_PER_WORLD)
    parser.add_argument("--multi-world-scheduler", default=g1_recipe.MULTI_WORLD_SCHEDULER)
    parser.add_argument("--prepare-refresh-stride", type=_parse_int_or_auto, default=g1_recipe.PREPARE_REFRESH_STRIDE)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "graph_leapfrog"),
        default="graph_leapfrog",
        help="Use eager PPO or the separate-graph rollout/update schedule.",
    )
    parser.add_argument("--readback-diagnostics", action="store_true")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--battery-steps", type=int, default=1000)
    parser.add_argument("--seeds-per-command", type=int, default=4)
    parser.add_argument("--diagnostic-steps", type=int, default=2000)
    parser.add_argument("--diagnostic-world-count", type=int, default=1)
    parser.add_argument("--stochastic-gate", action="store_true")
    parser.add_argument("--gate-seed", type=int, default=1000)
    parser.add_argument("--max-battery-falls", type=int, default=1)
    parser.add_argument("--min-battery-perf", type=float, default=0.90)
    parser.add_argument("--max-action-jerk-rms", type=float, default=0.21)
    parser.add_argument("--max-ang-vel-xy-rms", type=float, default=0.21)
    parser.add_argument("--max-yaw-rate-rms", type=float, default=0.20)
    parser.add_argument("--max-leg-qvel-rms", type=float, default=1.22)
    parser.add_argument("--keep-going-after-pass", action="store_true")
    parser.add_argument("--fail-on-miss", action="store_true")
    parser.add_argument("--include-train-history", action="store_true")
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args()
    if args.max_iterations is None:
        args.max_iterations = _default_max_iterations(args.world_count, args.rollout_steps, args.target_samples)
    return args


def main() -> int:
    args = _parse_args()
    result = benchmark_train_to_gate(args)
    print(json.dumps(result, indent=args.json_indent, sort_keys=True))
    return 1 if args.fail_on_miss and not result["pass_gate"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
