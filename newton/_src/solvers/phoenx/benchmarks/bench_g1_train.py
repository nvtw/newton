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
from typing import Any

import numpy as np
import warp as wp

import newton.rl as rl

_NANOG1_TRAIN_ENV_SPS = 1_280_000.0
_NANOG1_TRAIN_PHYSICS_SPS = 6_400_000.0


def _parse_hidden_layers(text: str) -> tuple[int, ...]:
    widths = tuple(int(item) for item in text.split(",") if item)
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("hidden layers must be a comma-separated list of positive widths")
    return widths


def _g1_ppo_config(
    train_epochs: int,
    mirror_loss_coeff: float,
    minibatch_size: int,
    replay_ratio: float,
    priority_alpha: float,
    priority_beta: float,
    manual_actor_backward: bool,
    manual_critic_backward: bool,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
    reward_clip: float,
    max_grad_norm: float,
) -> rl.ConfigPPO:
    return rl.ConfigPPO(
        gamma=0.97,
        gae_lambda=0.9,
        clip_ratio=0.2,
        entropy_coeff=1.0e-5,
        actor_lr=2.0e-3,
        critic_lr=2.0e-3,
        train_epochs=int(train_epochs),
        minibatch_size=int(minibatch_size),
        replay_ratio=float(replay_ratio),
        priority_alpha=float(priority_alpha),
        priority_beta=float(priority_beta),
        manual_actor_backward=bool(manual_actor_backward),
        manual_critic_backward=bool(manual_critic_backward),
        vtrace_rho_clip=float(vtrace_rho_clip),
        vtrace_c_clip=float(vtrace_c_clip),
        normalize_advantages=True,
        reward_clip=float(reward_clip),
        max_grad_norm=float(max_grad_norm),
        mirror_loss_coeff=float(mirror_loss_coeff),
    )


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
            args.vtrace_rho_clip,
            args.vtrace_c_clip,
            args.reward_clip,
            args.max_grad_norm,
        ),
        device=device,
        seed=int(args.seed),
        log_interval=0,
        randomize_commands=not bool(args.no_command_randomization),
    )
    result = rl.train_g1_ppo(config)
    history = result.history
    _validate_finite_history(history)
    warm = history[int(args.warmup_iterations) :]
    warm_sps = np.asarray([item.samples_per_second for item in warm], dtype=np.float64)
    warm_rollout = np.asarray([item.rollout_seconds for item in warm], dtype=np.float64)
    warm_update = np.asarray([item.update_seconds for item in warm], dtype=np.float64)
    env_sps = float(np.mean(warm_sps))
    physics_sps = env_sps * float(args.sim_substeps)
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
        "vtrace_rho_clip": float(args.vtrace_rho_clip),
        "vtrace_c_clip": float(args.vtrace_c_clip),
        "reward_clip": float(args.reward_clip),
        "max_grad_norm": float(args.max_grad_norm),
        "sim_substeps": int(args.sim_substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "parse_meshes": bool(args.parse_meshes),
        "env_samples_per_s": env_sps,
        "physics_steps_per_s": physics_sps,
        "mean_rollout_seconds": float(np.mean(warm_rollout)),
        "mean_update_seconds": float(np.mean(warm_update)),
        "nanog1_reference_env_samples_per_s": _NANOG1_TRAIN_ENV_SPS,
        "nanog1_reference_physics_steps_per_s": _NANOG1_TRAIN_PHYSICS_SPS,
        "phoenx_over_nanog1_train_ratio": env_sps / _NANOG1_TRAIN_ENV_SPS,
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
    parser.add_argument("--world-count", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--hidden-layers", type=_parse_hidden_layers, default=(128, 128, 128))
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--mirror-loss-coeff", type=float, default=0.25)
    parser.add_argument("--minibatch-size", type=int, default=32768)
    parser.add_argument("--replay-ratio", type=float, default=3.0)
    parser.add_argument("--priority-alpha", type=float, default=0.4)
    parser.add_argument("--priority-beta", type=float, default=1.0)
    parser.add_argument("--no-manual-actor-backward", action="store_true")
    parser.add_argument("--no-manual-critic-backward", action="store_true")
    parser.add_argument("--vtrace-rho-clip", type=float, default=3.0)
    parser.add_argument("--vtrace-c-clip", type=float, default=3.0)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--sim-substeps", type=int, default=5)
    parser.add_argument("--solver-iterations", type=int, default=2)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--controlled-action-count", type=int, default=12)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--no-command-randomization", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(benchmark_train(args), indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
