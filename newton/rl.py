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
    ConfigEvaluateG1GatePPO,
    ConfigEvaluateG1PPO,
    ConfigPPO,
    ConfigSAC,
    ConfigTrainAnymalPPO,
    ConfigTrainG1PPO,
    EnvAnymalPhoenX,
    EnvG1PhoenX,
    EnvPPO,
    GaussianActor,
    MirrorMapPPO,
    Muon,
    PufferMinGRUNet,
    ResultEvaluateAnymalPPO,
    ResultEvaluateG1GatePPO,
    ResultEvaluateG1PPO,
    ResultTrainAnymalPPO,
    ResultTrainG1PPO,
    StatsEvaluateAnymalTargetPPO,
    StatsEvaluateG1GateCommandPPO,
    StatsEvaluateG1GatePPO,
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
    evaluate_g1_gate_ppo,
    evaluate_g1_ppo,
    g1_mirror_map_ppo,
    g1_recipe,
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
    "ConfigEvaluateG1GatePPO",
    "ConfigEvaluateG1PPO",
    "ConfigPPO",
    "ConfigSAC",
    "ConfigTrainAnymalPPO",
    "ConfigTrainG1PPO",
    "EnvAnymalPhoenX",
    "EnvG1PhoenX",
    "EnvPPO",
    "GaussianActor",
    "MirrorMapPPO",
    "Muon",
    "PufferMinGRUNet",
    "ResultEvaluateAnymalPPO",
    "ResultEvaluateG1GatePPO",
    "ResultEvaluateG1PPO",
    "ResultTrainAnymalPPO",
    "ResultTrainG1PPO",
    "StatsEvaluateAnymalTargetPPO",
    "StatsEvaluateG1GateCommandPPO",
    "StatsEvaluateG1GatePPO",
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
    "evaluate_g1_gate_ppo",
    "evaluate_g1_ppo",
    "g1_mirror_map_ppo",
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
    g1_parser.add_argument("--iterations", type=int, default=g1_recipe.TRAIN_ITERATIONS)
    g1_parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    g1_parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    g1_parser.add_argument("--device", default=None)
    g1_parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    g1_parser.add_argument("--command-x", type=float, default=g1_recipe.COMMAND[0])
    g1_parser.add_argument("--command-y", type=float, default=g1_recipe.COMMAND[1])
    g1_parser.add_argument("--command-yaw", type=float, default=g1_recipe.COMMAND[2])
    g1_parser.add_argument("--command-x-min", type=float, default=g1_recipe.COMMAND_X_RANGE[0])
    g1_parser.add_argument("--command-x-max", type=float, default=g1_recipe.COMMAND_X_RANGE[1])
    g1_parser.add_argument("--command-y-min", type=float, default=g1_recipe.COMMAND_Y_RANGE[0])
    g1_parser.add_argument("--command-y-max", type=float, default=g1_recipe.COMMAND_Y_RANGE[1])
    g1_parser.add_argument("--command-yaw-min", type=float, default=g1_recipe.COMMAND_YAW_RANGE[0])
    g1_parser.add_argument("--command-yaw-max", type=float, default=g1_recipe.COMMAND_YAW_RANGE[1])
    g1_parser.add_argument("--no-command-randomization", action="store_true")
    g1_parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    g1_parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    g1_parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    g1_parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path: nanoG1-style explicit clamped PD torques or legacy PhoenX drive rows.",
    )
    g1_parser.add_argument("--parse-meshes", action="store_true")
    g1_parser.add_argument(
        "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY
    )
    g1_parser.add_argument(
        "--rigid-contact-max-per-world",
        type=int,
        default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
        help="Per-world G1 rigid-contact capacity; 0 keeps SolverPhoenX auto-sizing.",
    )
    g1_parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    g1_parser.add_argument("--mirror-loss-coeff", type=float, default=g1_recipe.MIRROR_LOSS_COEFF)
    g1_parser.add_argument(
        "--policy-network",
        choices=("mlp", "puffer_mingru"),
        default=g1_recipe.POLICY_NETWORK,
        help="PPO policy backbone. puffer_mingru matches nanoG1/PufferLib's recurrent default.",
    )
    g1_parser.add_argument("--minibatch-size", type=int, default=g1_recipe.MINIBATCH_SIZE)
    g1_parser.add_argument("--replay-ratio", type=float, default=g1_recipe.REPLAY_RATIO)
    g1_parser.add_argument("--priority-alpha", type=float, default=g1_recipe.PRIORITY_ALPHA)
    g1_parser.add_argument("--priority-beta", type=float, default=g1_recipe.PRIORITY_BETA)
    g1_parser.add_argument("--no-manual-actor-backward", action="store_true")
    g1_parser.add_argument("--no-manual-critic-backward", action="store_true")
    g1_parser.add_argument(
        "--manual-mlp-weight-grad-dtype",
        choices=("float32", "bfloat16"),
        default=g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE,
    )
    g1_parser.add_argument(
        "--manual-mlp-forward-dtype",
        choices=("float32", "bfloat16"),
        default=g1_recipe.MANUAL_MLP_FORWARD_DTYPE,
    )
    g1_parser.add_argument("--vtrace-rho-clip", type=float, default=g1_recipe.VTRACE_RHO_CLIP)
    g1_parser.add_argument("--vtrace-c-clip", type=float, default=g1_recipe.VTRACE_C_CLIP)
    g1_parser.add_argument("--reward-clip", type=float, default=g1_recipe.REWARD_CLIP)
    g1_parser.add_argument("--max-grad-norm", type=float, default=g1_recipe.MAX_GRAD_NORM)
    g1_parser.add_argument("--value-loss-coeff", type=float, default=g1_recipe.VALUE_LOSS_COEFF)
    g1_parser.add_argument("--value-clip-range", type=float, default=g1_recipe.VALUE_CLIP_RANGE)
    g1_parser.add_argument("--optimizer", choices=("adam", "muon"), default=g1_recipe.OPTIMIZER)
    g1_parser.add_argument("--optimizer-eps", type=float, default=g1_recipe.OPTIMIZER_EPS)
    g1_parser.add_argument("--optimizer-weight-decay", type=float, default=g1_recipe.OPTIMIZER_WEIGHT_DECAY)
    g1_parser.add_argument("--muon-momentum", type=float, default=g1_recipe.MUON_MOMENTUM)
    g1_parser.add_argument(
        "--squash-actions",
        action=argparse.BooleanOptionalAction,
        default=g1_recipe.SQUASH_ACTIONS,
        help="Use tanh-squashed PPO actions instead of the nanoG1-compatible raw Gaussian policy.",
    )
    g1_parser.add_argument(
        "--reset-recurrent-state-on-rollout-start",
        action=argparse.BooleanOptionalAction,
        default=g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
        help="Clear recurrent policy state at each PPO rollout boundary.",
    )
    g1_parser.add_argument("--resume-checkpoint", default=None)
    g1_parser.add_argument("--checkpoint-path", default=None)
    g1_parser.add_argument("--checkpoint-interval", type=int, default=0)
    g1_parser.add_argument("--log-interval", type=int, default=1)
    g1_parser.add_argument(
        "--execution-mode",
        choices=("eager", "graph_leapfrog"),
        default="eager",
        help="Use eager PPO or the experimental separate-graph rollout/update schedule.",
    )
    g1_parser.add_argument(
        "--no-readback-diagnostics",
        action="store_true",
        help="Do not copy training diagnostics to the host during the train loop.",
    )

    g1_eval_parser = subparsers.add_parser("eval-g1-ppo", help="Evaluate a saved Unitree G1 PPO checkpoint")
    g1_eval_parser.add_argument("--checkpoint", required=True)
    g1_eval_parser.add_argument("--steps", type=int, default=200)
    g1_eval_parser.add_argument("--world-count", type=int, default=64)
    g1_eval_parser.add_argument("--device", default=None)
    g1_eval_parser.add_argument("--seed", type=int, default=1000)
    g1_eval_parser.add_argument("--command-x", type=float, default=g1_recipe.COMMAND[0])
    g1_eval_parser.add_argument("--command-y", type=float, default=g1_recipe.COMMAND[1])
    g1_eval_parser.add_argument("--command-yaw", type=float, default=g1_recipe.COMMAND[2])
    g1_eval_parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    g1_eval_parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    g1_eval_parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    g1_eval_parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path used during evaluation.",
    )
    g1_eval_parser.add_argument("--parse-meshes", action="store_true")
    g1_eval_parser.add_argument(
        "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY
    )
    g1_eval_parser.add_argument(
        "--rigid-contact-max-per-world",
        type=int,
        default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
        help="Per-world G1 rigid-contact capacity; 0 keeps SolverPhoenX auto-sizing.",
    )
    g1_eval_parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    g1_eval_parser.add_argument("--stochastic", action="store_true")

    g1_gate_parser = subparsers.add_parser(
        "gate-g1-ppo", help="Run the nanoG1-style quality gate on a saved Unitree G1 PPO checkpoint"
    )
    g1_gate_parser.add_argument("--checkpoint", required=True)
    g1_gate_parser.add_argument("--battery-steps", type=int, default=1000)
    g1_gate_parser.add_argument("--seeds-per-command", type=int, default=4)
    g1_gate_parser.add_argument("--diagnostic-steps", type=int, default=2000)
    g1_gate_parser.add_argument("--diagnostic-world-count", type=int, default=1)
    g1_gate_parser.add_argument("--device", default=None)
    g1_gate_parser.add_argument("--seed", type=int, default=1000)
    g1_gate_parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    g1_gate_parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    g1_gate_parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    g1_gate_parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path used during the quality gate.",
    )
    g1_gate_parser.add_argument("--parse-meshes", action="store_true")
    g1_gate_parser.add_argument(
        "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY
    )
    g1_gate_parser.add_argument(
        "--rigid-contact-max-per-world",
        type=int,
        default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
        help="Per-world G1 rigid-contact capacity; 0 keeps SolverPhoenX auto-sizing.",
    )
    g1_gate_parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    g1_gate_parser.add_argument("--stochastic", action="store_true")
    g1_gate_parser.add_argument("--no-fail-on-gate", action="store_true")

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
            actuation_model=args.actuation_model,
            controlled_action_count=args.controlled_action_count,
            parse_meshes=args.parse_meshes,
            contact_geometry=args.contact_geometry,
            rigid_contact_max_per_world=args.rigid_contact_max_per_world,
        )
        ppo_config = g1_recipe.default_g1_ppo_config(
            minibatch_size=args.minibatch_size,
            replay_ratio=args.replay_ratio,
            priority_alpha=args.priority_alpha,
            priority_beta=args.priority_beta,
            manual_actor_backward=not args.no_manual_actor_backward,
            manual_critic_backward=not args.no_manual_critic_backward,
            manual_mlp_weight_grad_dtype=args.manual_mlp_weight_grad_dtype,
            manual_mlp_forward_dtype=args.manual_mlp_forward_dtype,
            vtrace_rho_clip=args.vtrace_rho_clip,
            vtrace_c_clip=args.vtrace_c_clip,
            reward_clip=args.reward_clip,
            max_grad_norm=args.max_grad_norm,
            value_loss_coeff=args.value_loss_coeff,
            value_clip_range=args.value_clip_range,
            optimizer=args.optimizer,
            optimizer_eps=args.optimizer_eps,
            optimizer_weight_decay=args.optimizer_weight_decay,
            muon_momentum=args.muon_momentum,
            mirror_loss_coeff=args.mirror_loss_coeff,
            policy_network=args.policy_network,
        )
        train_g1_ppo(
            ConfigTrainG1PPO(
                iterations=args.iterations,
                rollout_steps=args.rollout_steps,
                env_config=env_config,
                ppo_config=ppo_config,
                device=args.device,
                seed=args.seed,
                log_interval=args.log_interval,
                randomize_commands=not args.no_command_randomization,
                command_x_range=(args.command_x_min, args.command_x_max),
                command_y_range=(args.command_y_min, args.command_y_max),
                command_yaw_range=(args.command_yaw_min, args.command_yaw_max),
                reset_recurrent_state_on_rollout_start=bool(args.reset_recurrent_state_on_rollout_start),
                squash_actions=bool(args.squash_actions),
                resume_checkpoint=args.resume_checkpoint,
                checkpoint_path=args.checkpoint_path,
                checkpoint_interval=args.checkpoint_interval,
                readback_diagnostics=not args.no_readback_diagnostics,
                execution_mode=args.execution_mode,
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
            actuation_model=args.actuation_model,
            controlled_action_count=args.controlled_action_count,
            parse_meshes=args.parse_meshes,
            contact_geometry=args.contact_geometry,
            rigid_contact_max_per_world=args.rigid_contact_max_per_world,
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
    if args.command == "gate-g1-ppo":
        env_config = ConfigEnvG1PhoenX(
            sim_substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            actuation_model=args.actuation_model,
            controlled_action_count=args.controlled_action_count,
            parse_meshes=args.parse_meshes,
            contact_geometry=args.contact_geometry,
            rigid_contact_max_per_world=args.rigid_contact_max_per_world,
        )
        trainer = load_ppo_checkpoint(args.checkpoint, device=args.device)
        result = evaluate_g1_gate_ppo(
            trainer,
            ConfigEvaluateG1GatePPO(
                env_config=env_config,
                battery_steps=args.battery_steps,
                seeds_per_command=args.seeds_per_command,
                diagnostic_steps=args.diagnostic_steps,
                diagnostic_world_count=args.diagnostic_world_count,
                device=args.device,
                deterministic=not args.stochastic,
                seed=args.seed,
            ),
        )
        print(json.dumps(asdict(result.stats), sort_keys=True))
        return 0 if result.stats.pass_gate or args.no_fail_on_gate else 1
    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
