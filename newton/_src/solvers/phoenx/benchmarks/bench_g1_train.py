# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX G1 pure-Warp PPO training throughput benchmark.

Measures the full ``train_g1_ppo`` collect-update loop rather than only
graph-captured environment stepping. This benchmark is intentionally small by
default and reports warm iterations separately from first-iteration setup and
JIT costs.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train --iterations 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe

_NANOG1_TRAIN_ENV_SPS = 1_280_000.0
_NANOG1_TRAIN_PHYSICS_SPS = 6_400_000.0
_NANOG1_REFERENCE_SOURCE = "nanoG1 RESULTS.md rounded steady training SPS"
_RESULT_END_MARKER = "=== END RESULT ==="
_NANOG1_TRAIN_MARKERS = ("=== nanoG1 RESULT ===",)


def _parse_hidden_layers(text: str) -> tuple[int, ...]:
    widths = tuple(int(item) for item in text.split(",") if item)
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("hidden layers must be a comma-separated list of positive widths")
    return widths


def _parse_int_or_auto(text: str) -> int | str:
    value = text.strip().lower()
    return "auto" if value == "auto" else int(value)


def _g1_ppo_config(
    train_epochs: int,
    mirror_loss_coeff: float,
    minibatch_size: int,
    replay_ratio: float,
    priority_alpha: float,
    priority_beta: float,
    manual_actor_backward: bool,
    manual_critic_backward: bool,
    manual_mlp_weight_grad_dtype: str,
    manual_mlp_forward_dtype: str,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
    reward_clip: float,
    max_grad_norm: float,
) -> rl.ConfigPPO:
    return g1_recipe.default_g1_ppo_config(
        train_epochs=int(train_epochs),
        mirror_loss_coeff=float(mirror_loss_coeff),
        minibatch_size=int(minibatch_size),
        replay_ratio=float(replay_ratio),
        priority_alpha=float(priority_alpha),
        priority_beta=float(priority_beta),
        manual_actor_backward=bool(manual_actor_backward),
        manual_critic_backward=bool(manual_critic_backward),
        manual_mlp_weight_grad_dtype=str(manual_mlp_weight_grad_dtype),
        manual_mlp_forward_dtype=str(manual_mlp_forward_dtype),
        vtrace_rho_clip=float(vtrace_rho_clip),
        vtrace_c_clip=float(vtrace_c_clip),
        reward_clip=float(reward_clip),
        max_grad_norm=float(max_grad_norm),
    )


def _json_from_result_text(text: str, *, markers: tuple[str, ...], path: Path) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{path} is empty")
    try:
        blob = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        if not isinstance(blob, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return blob

    for marker in markers:
        start = stripped.find(marker)
        if start < 0:
            continue
        start += len(marker)
        end = stripped.find(_RESULT_END_MARKER, start)
        blob_text = stripped[start : end if end >= 0 else None].strip()
        try:
            blob = json.loads(blob_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} has an invalid JSON blob after {marker!r}") from exc
        if not isinstance(blob, dict):
            raise ValueError(f"{path} must contain a JSON object after {marker!r}")
        return blob
    raise ValueError(f"{path} does not contain JSON or a nanoG1 result marker")


def _load_nanog1_train_result(path: Path) -> dict[str, Any]:
    return _json_from_result_text(path.read_text(encoding="utf-8"), markers=_NANOG1_TRAIN_MARKERS, path=path)


def _nanog1_training_reference(args: argparse.Namespace) -> tuple[float, float, str]:
    env_sps = _NANOG1_TRAIN_ENV_SPS
    physics_sps = _NANOG1_TRAIN_PHYSICS_SPS
    source = _NANOG1_REFERENCE_SOURCE

    if args.nanog1_train_result is not None:
        result_path = Path(args.nanog1_train_result)
        blob = _load_nanog1_train_result(result_path)
        steady_sps = blob.get("steady_sps")
        if steady_sps is None:
            raise ValueError(f"{result_path} is missing nanoG1 train.py 'steady_sps'")
        env_sps = float(steady_sps)
        physics_value = blob.get("physics_steps_per_s")
        physics_sps = float(physics_value) if physics_value is not None else env_sps * float(args.sim_substeps)
        source = str(result_path)
        gpu = blob.get("gpu")
        if gpu:
            source = f"{source} ({gpu})"

    if args.nanog1_reference_env_sps is not None:
        env_sps = float(args.nanog1_reference_env_sps)
        if env_sps <= 0.0:
            raise ValueError("--nanog1-reference-env-sps must be positive")
        source = args.nanog1_reference_source
    if args.nanog1_reference_physics_sps is not None:
        physics_sps = float(args.nanog1_reference_physics_sps)
        if physics_sps <= 0.0:
            raise ValueError("--nanog1-reference-physics-sps must be positive")
        source = args.nanog1_reference_source
    elif args.nanog1_reference_env_sps is not None:
        physics_sps = env_sps * float(args.sim_substeps)

    if env_sps <= 0.0 or physics_sps <= 0.0:
        raise ValueError("nanoG1 reference throughput must be positive")
    return env_sps, physics_sps, source


def benchmark_train(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("PhoenX G1 training benchmark requires CUDA with Warp mempool enabled")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if args.warmup_iterations >= args.iterations:
        raise ValueError("warmup_iterations must be less than iterations")

    env_config = rl.ConfigEnvG1PhoenX(
        world_count=int(args.world_count),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        controlled_action_count=int(args.controlled_action_count),
        parse_meshes=bool(args.parse_meshes),
        rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
        threads_per_world=args.threads_per_world,
        multi_world_scheduler=str(args.multi_world_scheduler),
        prepare_refresh_stride=args.prepare_refresh_stride,
    )
    config = rl.ConfigTrainG1PPO(
        iterations=int(args.iterations),
        rollout_steps=int(args.rollout_steps),
        hidden_layers=tuple(args.hidden_layers),
        env_config=env_config,
        ppo_config=_g1_ppo_config(
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
        ),
        device=device,
        seed=int(args.seed),
        log_interval=0,
        randomize_commands=not bool(args.no_command_randomization),
        readback_diagnostics=not bool(args.no_readback_diagnostics),
        execution_mode=str(args.execution_mode),
    )
    result = rl.train_g1_ppo(config)
    world = result.env.solver.world
    effective_tpw = int(world._tpw_choice.numpy()[0])
    history = result.history
    _validate_finite_history(history)
    warm = history[int(args.warmup_iterations) :]
    excluded_final_graph_drain = False
    if args.execution_mode == "graph_leapfrog" and len(warm) > 1:
        # The last graph-leapfrog interval only drains the pending PPO update; the
        # matching rollout was already timed in the previous overlapped interval.
        warm = warm[:-1]
        excluded_final_graph_drain = True
    warm_sps = np.asarray([item.samples_per_second for item in warm], dtype=np.float64)
    warm_rollout = np.asarray([item.rollout_seconds for item in warm], dtype=np.float64)
    warm_update = np.asarray([item.update_seconds for item in warm], dtype=np.float64)
    env_sps = float(np.mean(warm_sps))
    physics_sps = env_sps * float(args.sim_substeps)
    nanog1_env_sps, nanog1_physics_sps, nanog1_source = _nanog1_training_reference(args)
    return {
        "engine": "phoenx_g1_warp_ppo",
        "metric": "full collect-update training throughput",
        "device": device.name,
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "iterations": int(args.iterations),
        "warmup_iterations": int(args.warmup_iterations),
        "hidden_layers": list(args.hidden_layers),
        "train_epochs": int(args.train_epochs),
        "mirror_loss_coeff": float(args.mirror_loss_coeff),
        "minibatch_size": int(args.minibatch_size),
        "replay_ratio": float(args.replay_ratio),
        "priority_alpha": float(args.priority_alpha),
        "priority_beta": float(args.priority_beta),
        "manual_actor_backward": not bool(args.no_manual_actor_backward),
        "manual_critic_backward": not bool(args.no_manual_critic_backward),
        "manual_mlp_weight_grad_dtype": str(args.manual_mlp_weight_grad_dtype),
        "manual_mlp_forward_dtype": str(args.manual_mlp_forward_dtype),
        "vtrace_rho_clip": float(args.vtrace_rho_clip),
        "vtrace_c_clip": float(args.vtrace_c_clip),
        "reward_clip": float(args.reward_clip),
        "max_grad_norm": float(args.max_grad_norm),
        "readback_diagnostics": not bool(args.no_readback_diagnostics),
        "execution_mode": str(args.execution_mode),
        "sim_substeps": int(args.sim_substeps),
        "solver_internal_substeps": int(world.substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "parse_meshes": bool(args.parse_meshes),
        "rigid_contact_max_per_world": int(args.rigid_contact_max_per_world),
        "threads_per_world": args.threads_per_world,
        "effective_threads_per_world": effective_tpw,
        "threads_per_world_auto_active": bool(world._tpw_auto),
        "multi_world_scheduler": str(args.multi_world_scheduler),
        "prepare_refresh_stride": args.prepare_refresh_stride,
        "env_samples_per_s": env_sps,
        "physics_steps_per_s": physics_sps,
        "mean_rollout_seconds": float(np.mean(warm_rollout)),
        "mean_update_seconds": float(np.mean(warm_update)),
        "mean_samples_iterations": [int(item.iteration) for item in warm],
        "excluded_final_graph_drain_from_mean": excluded_final_graph_drain,
        "nanog1_reference_source": nanog1_source,
        "nanog1_reference_env_samples_per_s": nanog1_env_sps,
        "nanog1_reference_physics_steps_per_s": nanog1_physics_sps,
        "phoenx_over_nanog1_train_ratio": env_sps / nanog1_env_sps,
        "phoenx_train_slowdown_vs_nanog1": nanog1_env_sps / env_sps,
        "history": [asdict(item) for item in history],
    }


def _validate_finite_history(history: list[rl.StatsTrainG1PPO]) -> None:
    fields = (
        "mean_reward",
        "mean_done",
        "mean_tracking_perf",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "clip_fraction",
        "samples_per_second",
    )
    for stats in history:
        for field in fields:
            value = float(getattr(stats, field))
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite {field} at iteration {stats.iteration}: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
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
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument(
        "--rigid-contact-max-per-world",
        type=int,
        default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
        help="Per-world G1 rigid-contact capacity; 0 keeps SolverPhoenX auto-sizing.",
    )
    parser.add_argument("--threads-per-world", type=_parse_int_or_auto, default=g1_recipe.THREADS_PER_WORLD)
    parser.add_argument("--multi-world-scheduler", default=g1_recipe.MULTI_WORLD_SCHEDULER)
    parser.add_argument("--prepare-refresh-stride", type=_parse_int_or_auto, default=g1_recipe.PREPARE_REFRESH_STRIDE)
    parser.add_argument("--no-command-randomization", action="store_true")
    parser.add_argument(
        "--no-readback-diagnostics",
        action="store_true",
        help="Skip host diagnostic readbacks during the measured train loop.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "graph_leapfrog"),
        default="eager",
        help="Use eager PPO or the experimental separate-graph rollout/update schedule.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument(
        "--nanog1-train-result",
        type=Path,
        default=None,
        help="Path to a nanoG1 train.py JSON result or log containing the '=== nanoG1 RESULT ===' blob.",
    )
    parser.add_argument(
        "--nanog1-reference-env-sps",
        type=float,
        default=None,
        help="Override the nanoG1 training reference env samples/s.",
    )
    parser.add_argument(
        "--nanog1-reference-physics-sps",
        type=float,
        default=None,
        help="Override the nanoG1 training reference physics steps/s.",
    )
    parser.add_argument(
        "--nanog1-reference-source",
        default="manual nanoG1 training reference",
        help="Source label used when --nanog1-reference-*-sps overrides are supplied.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(benchmark_train(args), indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
