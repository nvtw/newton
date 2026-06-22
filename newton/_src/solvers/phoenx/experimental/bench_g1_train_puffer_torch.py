# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train PhoenX G1 with a PufferLib-style Torch learner.

This benchmark is an isolation tool: physics, observations, rewards, resets, and
actuation come from PhoenX, while the RL update follows PufferLib/nanoG1
conventions as closely as possible from Python/Torch.

Example:
    uv run --extra dev -m newton._src.solvers.phoenx.experimental.bench_g1_train_puffer_torch \
        --iterations 2 --world-count 64 --rollout-steps 8 \
        --pufferlib-root ../PufferLib
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import warp as wp

from newton._src.solvers.phoenx.experimental.puffer_torch import (
    ConfigTrainG1PufferTorch,
    evaluate_g1_gate_puffer_torch,
    train_g1_puffer_torch,
)
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.training import ConfigEvaluateG1GatePPO


def _default_pufferlib_root() -> str | None:
    env_root = os.environ.get("PUFFERLIB_ROOT")
    if env_root:
        return env_root
    candidate = Path.cwd().resolve().parent / "PufferLib"
    return str(candidate) if candidate.exists() else None


def benchmark_train_puffer_torch(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("PhoenX G1 Puffer-style benchmark requires CUDA")
    env_config = g1_recipe.default_g1_env_config(
        world_count=int(args.world_count),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        actuation_model=str(args.actuation_model),
    )
    cfg = ConfigTrainG1PufferTorch(
        iterations=int(args.iterations),
        rollout_steps=int(args.rollout_steps),
        hidden_layers=tuple(int(v) for v in args.hidden_layers),
        env_config=env_config,
        device=device,
        seed=int(args.seed),
        total_timesteps=int(args.total_timesteps),
        minibatch_size=int(args.minibatch_size),
        replay_ratio=float(args.replay_ratio),
        mirror_loss_coeff=0.0 if args.no_mirror_loss else float(args.mirror_loss_coeff),
        mirror_value_loss=not bool(args.no_mirror_value_loss),
        command_curriculum_start=float(args.command_curriculum_start),
        command_curriculum_samples=int(args.command_curriculum_samples),
        reset_recurrent_state_on_rollout_start=not bool(args.keep_recurrent_state),
        pufferlib_root=args.pufferlib_root,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=int(args.checkpoint_interval),
        log_interval=int(args.log_interval),
    )
    result = train_g1_puffer_torch(cfg)
    final = result.history[-1]
    total_samples = int(args.iterations) * int(args.world_count) * int(args.rollout_steps)
    train_seconds = sum(s.rollout_seconds + s.update_seconds for s in result.history)
    output: dict[str, Any] = {
        "trainer": "puffer_torch",
        "pufferlib_root": args.pufferlib_root,
        "iterations": int(args.iterations),
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "total_samples": total_samples,
        "train_seconds": train_seconds,
        "env_samples_per_s": total_samples / max(train_seconds, 1.0e-12),
        "actuation_model": str(args.actuation_model),
        "sim_substeps": int(args.sim_substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "command_curriculum_start": float(args.command_curriculum_start),
        "command_curriculum_samples": int(args.command_curriculum_samples),
        "final": asdict(final),
        "history": [asdict(s) for s in result.history],
    }
    if args.gate:
        gate_cfg = ConfigEvaluateG1GatePPO(
            env_config=replace(env_config, world_count=6 * int(args.gate_seeds_per_command)),
            seeds_per_command=int(args.gate_seeds_per_command),
            battery_steps=int(args.gate_steps),
            diagnostic_steps=int(args.gate_diagnostic_steps),
            diagnostic_world_count=int(args.gate_diagnostic_worlds),
            device=device,
            deterministic=not bool(args.stochastic_gate),
            seed=int(args.gate_seed),
        )
        gate = evaluate_g1_gate_puffer_torch(result.trainer, gate_cfg).stats
        output["gate"] = {
            "pass_gate": bool(gate.pass_gate),
            "battery_perf": float(gate.battery_perf),
            "battery_falls": int(gate.battery_falls),
            "action_jerk_rms": float(gate.action_jerk_rms),
            "ang_vel_xy_rms": float(gate.ang_vel_xy_rms),
            "yaw_rate_rms": float(gate.yaw_rate_rms),
            "leg_qvel_rms": float(gate.leg_qvel_rms),
            "samples_per_second": float(gate.samples_per_second),
            "per_command": [asdict(item) for item in gate.per_command],
        }
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pufferlib-root", default=_default_pufferlib_root())
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=list(g1_recipe.HIDDEN_LAYERS))
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--total-timesteps", type=int, default=g1_recipe.LR_ANNEAL_TIMESTEPS)
    parser.add_argument("--minibatch-size", type=int, default=g1_recipe.MINIBATCH_SIZE)
    parser.add_argument("--replay-ratio", type=float, default=g1_recipe.REPLAY_RATIO)
    parser.add_argument("--mirror-loss-coeff", type=float, default=g1_recipe.MIRROR_LOSS_COEFF)
    parser.add_argument("--no-mirror-loss", action="store_true")
    parser.add_argument("--no-mirror-value-loss", action="store_true")
    parser.add_argument("--keep-recurrent-state", action="store_true")
    parser.add_argument(
        "--command-curriculum-start",
        type=float,
        default=g1_recipe.COMMAND_CURRICULUM_START,
        help="Initial nanoG1 command-range scale for randomized G1 commands.",
    )
    parser.add_argument(
        "--command-curriculum-samples",
        type=int,
        default=g1_recipe.COMMAND_CURRICULUM_SAMPLES,
        help="Samples used to ramp randomized G1 commands to full range; 0 disables the ramp.",
    )
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--gate-steps", type=int, default=1000)
    parser.add_argument("--gate-seeds-per-command", type=int, default=4)
    parser.add_argument("--gate-diagnostic-steps", type=int, default=2000)
    parser.add_argument("--gate-diagnostic-worlds", type=int, default=1)
    parser.add_argument("--gate-seed", type=int, default=1000)
    parser.add_argument("--stochastic-gate", action="store_true")
    parser.add_argument("--json-indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = benchmark_train_puffer_torch(args)
    print(json.dumps(result, indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
