# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train Unitree Go2 on flat terrain with PhoenX and Warp-only PPO.

This executable has no IsaacLab runtime dependency. Its robot asset is loaded
through Newton and its rollout, policy, optimizer, and physics all stay in the
pure Newton/Warp stack.

Example::

    python newton/_src/solvers/phoenx/rl_training/examples/train_go2_phoenx_ppo.py \
        --device cuda:0 --execution-mode graph_leapfrog
"""

from __future__ import annotations

import argparse
import time

import newton.rl as rl


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-count", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--execution-mode", choices=("eager", "graph_leapfrog"), default="graph_leapfrog")
    parser.add_argument("--command-x-min", type=float, default=-1.0)
    parser.add_argument("--command-x-max", type=float, default=1.0)
    parser.add_argument("--command-y-min", type=float, default=-1.0)
    parser.add_argument("--command-y-max", type=float, default=1.0)
    parser.add_argument("--command-yaw-min", type=float, default=-1.0)
    parser.add_argument("--command-yaw-max", type=float, default=1.0)
    parser.add_argument("--command-zero-probability", type=float, default=0.1)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    env_config = rl.ConfigEnvGo2PhoenX(
        world_count=int(args.world_count),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        reward_mode="dense_command",
        command=(0.0, 0.0, 0.0, 0.0),
    )
    train_config = rl.ConfigTrainAnymalPPO(
        iterations=int(args.iterations),
        rollout_steps=int(args.rollout_steps),
        env_config=env_config,
        device=args.device,
        seed=int(args.seed),
        log_interval=int(args.log_interval),
        use_target_curriculum=False,
        randomize_commands=True,
        command_x_range=(float(args.command_x_min), float(args.command_x_max)),
        command_y_range=(float(args.command_y_min), float(args.command_y_max)),
        command_yaw_range=(float(args.command_yaw_min), float(args.command_yaw_max)),
        command_height_range=(0.0, 0.0),
        command_zero_probability=float(args.command_zero_probability),
        resume_checkpoint=args.resume_checkpoint,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=int(args.checkpoint_interval),
        execution_mode=str(args.execution_mode),
    )

    start = time.perf_counter()
    result = rl.train_anymal_ppo(train_config)
    elapsed = max(time.perf_counter() - start, 1.0e-12)
    samples = int(args.iterations) * int(args.rollout_steps) * int(args.world_count)
    print(f"Go2 training complete: {samples / elapsed:,.0f} samples/s over {samples:,} samples")
    if result.history:
        print(result.history[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
