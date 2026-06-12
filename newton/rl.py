# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-only reinforcement learning utilities for Newton."""

from ._src.solvers.phoenx.rl_training import (
    ACTION_DIM_ANYMAL,
    OBS_DIM_ANYMAL,
    Adam,
    BatchSAC,
    BufferReplaySAC,
    BufferRollout,
    ConfigEnvAnymalPhoenX,
    ConfigPPO,
    ConfigSAC,
    ConfigTrainAnymalPPO,
    EnvAnymalPhoenX,
    GaussianActor,
    ResultTrainAnymalPPO,
    StatsPPOUpdate,
    StatsSACUpdate,
    StatsTrainAnymalPPO,
    TrainerPPO,
    TrainerSAC,
    WarpMLP,
    train_anymal_ppo,
)

__all__ = [
    "ACTION_DIM_ANYMAL",
    "OBS_DIM_ANYMAL",
    "Adam",
    "BatchSAC",
    "BufferReplaySAC",
    "BufferRollout",
    "ConfigEnvAnymalPhoenX",
    "ConfigPPO",
    "ConfigSAC",
    "ConfigTrainAnymalPPO",
    "EnvAnymalPhoenX",
    "GaussianActor",
    "ResultTrainAnymalPPO",
    "StatsPPOUpdate",
    "StatsSACUpdate",
    "StatsTrainAnymalPPO",
    "TrainerPPO",
    "TrainerSAC",
    "WarpMLP",
    "train_anymal_ppo",
]


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Warp-only Newton RL utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train-anymal-ppo", help="Train Anymal C with PhoenX and Warp-only PPO")
    train_parser.add_argument("--iterations", type=int, default=100)
    train_parser.add_argument("--rollout-steps", type=int, default=64)
    train_parser.add_argument("--world-count", type=int, default=1024)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--command-x", type=float, default=1.0)
    train_parser.add_argument("--command-y", type=float, default=0.0)
    train_parser.add_argument("--command-yaw", type=float, default=0.0)
    train_parser.add_argument("--reward-mode", choices=("sparse_target", "dense_command"), default="sparse_target")
    train_parser.add_argument("--target-distance", type=float, default=0.45)
    train_parser.add_argument("--target-distance-end", type=float, default=1.0)
    train_parser.add_argument("--target-distance-step", type=float, default=0.1)
    train_parser.add_argument("--target-success-threshold", type=float, default=0.045)
    train_parser.add_argument("--no-target-curriculum", action="store_true")
    train_parser.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()
    if args.command == "train-anymal-ppo":
        env_config = ConfigEnvAnymalPhoenX(
            world_count=args.world_count,
            reward_mode=args.reward_mode,
            command=(args.command_x, args.command_y, args.command_yaw),
            target_position=(0.0, args.target_distance),
        )
        train_anymal_ppo(
            ConfigTrainAnymalPPO(
                iterations=args.iterations,
                rollout_steps=args.rollout_steps,
                env_config=env_config,
                device=args.device,
                seed=args.seed,
                log_interval=args.log_interval,
                use_target_curriculum=not args.no_target_curriculum,
                target_distance_start=args.target_distance,
                target_distance_end=args.target_distance_end,
                target_distance_step=args.target_distance_step,
                target_success_threshold=args.target_success_threshold,
            )
        )
        return 0
    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
