# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-only reinforcement learning utilities for Newton."""

import argparse
import json
from dataclasses import asdict

from ._src.solvers.phoenx.rl_training import (
    ACTION_DIM_ANYMAL,
    ACTION_DIM_G1,
    OBS_DIM_ANYMAL,
    OBS_DIM_G1,
    Adam,
    BatchSAC,
    BufferReplaySAC,
    BufferRollout,
    ConfigEnvAnymalPhoenX,
    ConfigEnvG1PhoenX,
    ConfigEvaluateAnymalPPO,
    ConfigEvaluateG1PPO,
    ConfigPPO,
    ConfigSAC,
    ConfigTrainAnymalPPO,
    ConfigTrainG1PPO,
    EnvAnymalPhoenX,
    EnvG1PhoenX,
    EnvPPO,
    GaussianActor,
    ResultEvaluateAnymalPPO,
    ResultEvaluateG1PPO,
    ResultTrainAnymalPPO,
    ResultTrainG1PPO,
    StatsEvaluateAnymalTargetPPO,
    StatsEvaluateG1PPO,
    StatsPPOUpdate,
    StatsSACUpdate,
    StatsTrainAnymalPPO,
    StatsTrainG1PPO,
    TrainerPPO,
    TrainerSAC,
    WarpMLP,
    capture_env_steps,
    collect_ppo_rollout,
    evaluate_anymal_ppo,
    evaluate_g1_ppo,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
    train_anymal_ppo,
    train_g1_ppo,
)

__all__ = [
    "ACTION_DIM_ANYMAL",
    "ACTION_DIM_G1",
    "OBS_DIM_ANYMAL",
    "OBS_DIM_G1",
    "Adam",
    "BatchSAC",
    "BufferReplaySAC",
    "BufferRollout",
    "ConfigEnvAnymalPhoenX",
    "ConfigEnvG1PhoenX",
    "ConfigEvaluateAnymalPPO",
    "ConfigEvaluateG1PPO",
    "ConfigPPO",
    "ConfigSAC",
    "ConfigTrainAnymalPPO",
    "ConfigTrainG1PPO",
    "EnvAnymalPhoenX",
    "EnvG1PhoenX",
    "EnvPPO",
    "GaussianActor",
    "ResultEvaluateAnymalPPO",
    "ResultEvaluateG1PPO",
    "ResultTrainAnymalPPO",
    "ResultTrainG1PPO",
    "StatsEvaluateAnymalTargetPPO",
    "StatsEvaluateG1PPO",
    "StatsPPOUpdate",
    "StatsSACUpdate",
    "StatsTrainAnymalPPO",
    "StatsTrainG1PPO",
    "TrainerPPO",
    "TrainerSAC",
    "WarpMLP",
    "capture_env_steps",
    "collect_ppo_rollout",
    "evaluate_anymal_ppo",
    "evaluate_g1_ppo",
    "load_ppo_checkpoint",
    "save_ppo_checkpoint",
    "train_anymal_ppo",
    "train_g1_ppo",
]


def _main() -> int:
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
    train_parser.add_argument("--target-success-threshold", type=float, default=0.025)
    train_parser.add_argument("--target-angle-min", type=float, default=-3.141592653589793)
    train_parser.add_argument("--target-angle-max", type=float, default=3.141592653589793)
    train_parser.add_argument("--no-target-randomization", action="store_true")
    train_parser.add_argument("--no-target-curriculum", action="store_true")
    train_parser.add_argument("--resume-checkpoint", default=None)
    train_parser.add_argument("--checkpoint-path", default=None)
    train_parser.add_argument("--checkpoint-interval", type=int, default=0)
    train_parser.add_argument("--log-interval", type=int, default=1)

    g1_parser = subparsers.add_parser("train-g1-ppo", help="Train Unitree G1 with PhoenX and Warp-only PPO")
    g1_parser.add_argument("--iterations", type=int, default=100)
    g1_parser.add_argument("--rollout-steps", type=int, default=64)
    g1_parser.add_argument("--world-count", type=int, default=4096)
    g1_parser.add_argument("--device", default=None)
    g1_parser.add_argument("--seed", type=int, default=42)
    g1_parser.add_argument("--command-x", type=float, default=0.8)
    g1_parser.add_argument("--command-y", type=float, default=0.0)
    g1_parser.add_argument("--command-yaw", type=float, default=0.0)
    g1_parser.add_argument("--command-x-min", type=float, default=-0.5)
    g1_parser.add_argument("--command-x-max", type=float, default=0.8)
    g1_parser.add_argument("--command-y-min", type=float, default=-0.4)
    g1_parser.add_argument("--command-y-max", type=float, default=0.4)
    g1_parser.add_argument("--command-yaw-min", type=float, default=-1.0)
    g1_parser.add_argument("--command-yaw-max", type=float, default=1.0)
    g1_parser.add_argument("--no-command-randomization", action="store_true")
    g1_parser.add_argument("--sim-substeps", type=int, default=5)
    g1_parser.add_argument("--solver-iterations", type=int, default=2)
    g1_parser.add_argument("--velocity-iterations", type=int, default=1)
    g1_parser.add_argument("--parse-meshes", action="store_true")
    g1_parser.add_argument("--controlled-action-count", type=int, default=12)
    g1_parser.add_argument("--resume-checkpoint", default=None)
    g1_parser.add_argument("--checkpoint-path", default=None)
    g1_parser.add_argument("--checkpoint-interval", type=int, default=0)
    g1_parser.add_argument("--log-interval", type=int, default=1)

    g1_eval_parser = subparsers.add_parser("eval-g1-ppo", help="Evaluate a saved Unitree G1 PPO checkpoint")
    g1_eval_parser.add_argument("--checkpoint", required=True)
    g1_eval_parser.add_argument("--steps", type=int, default=200)
    g1_eval_parser.add_argument("--world-count", type=int, default=64)
    g1_eval_parser.add_argument("--device", default=None)
    g1_eval_parser.add_argument("--seed", type=int, default=1000)
    g1_eval_parser.add_argument("--command-x", type=float, default=0.8)
    g1_eval_parser.add_argument("--command-y", type=float, default=0.0)
    g1_eval_parser.add_argument("--command-yaw", type=float, default=0.0)
    g1_eval_parser.add_argument("--sim-substeps", type=int, default=5)
    g1_eval_parser.add_argument("--solver-iterations", type=int, default=2)
    g1_eval_parser.add_argument("--velocity-iterations", type=int, default=1)
    g1_eval_parser.add_argument("--parse-meshes", action="store_true")
    g1_eval_parser.add_argument("--controlled-action-count", type=int, default=12)
    g1_eval_parser.add_argument("--stochastic", action="store_true")

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
                randomize_target_positions=not args.no_target_randomization,
                target_angle_min=args.target_angle_min,
                target_angle_max=args.target_angle_max,
                resume_checkpoint=args.resume_checkpoint,
                checkpoint_path=args.checkpoint_path,
                checkpoint_interval=args.checkpoint_interval,
            )
        )
        return 0
    if args.command == "train-g1-ppo":
        env_config = ConfigEnvG1PhoenX(
            world_count=args.world_count,
            command=(args.command_x, args.command_y, args.command_yaw),
            sim_substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            controlled_action_count=args.controlled_action_count,
            parse_meshes=args.parse_meshes,
        )
        train_g1_ppo(
            ConfigTrainG1PPO(
                iterations=args.iterations,
                rollout_steps=args.rollout_steps,
                env_config=env_config,
                device=args.device,
                seed=args.seed,
                log_interval=args.log_interval,
                randomize_commands=not args.no_command_randomization,
                command_x_range=(args.command_x_min, args.command_x_max),
                command_y_range=(args.command_y_min, args.command_y_max),
                command_yaw_range=(args.command_yaw_min, args.command_yaw_max),
                resume_checkpoint=args.resume_checkpoint,
                checkpoint_path=args.checkpoint_path,
                checkpoint_interval=args.checkpoint_interval,
            )
        )
        return 0
    if args.command == "eval-g1-ppo":
        env_config = ConfigEnvG1PhoenX(
            world_count=args.world_count,
            command=(args.command_x, args.command_y, args.command_yaw),
            sim_substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            controlled_action_count=args.controlled_action_count,
            parse_meshes=args.parse_meshes,
        )
        trainer = load_ppo_checkpoint(args.checkpoint, device=args.device)
        result = evaluate_g1_ppo(
            trainer,
            ConfigEvaluateG1PPO(
                env_config=env_config,
                steps=args.steps,
                device=args.device,
                deterministic=not args.stochastic,
                seed=args.seed,
            ),
        )
        print(json.dumps(asdict(result.stats), sort_keys=True))
        return 0
    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
