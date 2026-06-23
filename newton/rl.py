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
    OBS_DIM_G1_ISAACLAB_FLAT,
    OBS_DIM_G1_NANOG1,
    Adam,
    BatchSAC,
    BufferReplaySAC,
    BufferRollout,
    ConfigEnvAnymalPhoenX,
    ConfigEnvG1PhoenX,
    ConfigEvaluateAnymalPPO,
    ConfigEvaluateG1GatePPO,
    ConfigEvaluateG1PPO,
    ConfigEvaluateG1TargetPPO,
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
    ResultEvaluateG1TargetPPO,
    ResultTrainAnymalPPO,
    ResultTrainG1PPO,
    StatsEvaluateAnymalTargetPPO,
    StatsEvaluateG1GateCommandPPO,
    StatsEvaluateG1GatePPO,
    StatsEvaluateG1PPO,
    StatsEvaluateG1TargetPPO,
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
    evaluate_g1_target_ppo,
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
    "OBS_DIM_G1_ISAACLAB_FLAT",
    "OBS_DIM_G1_NANOG1",
    "Adam",
    "BatchSAC",
    "BufferReplaySAC",
    "BufferRollout",
    "ConfigEnvAnymalPhoenX",
    "ConfigEnvG1PhoenX",
    "ConfigEvaluateAnymalPPO",
    "ConfigEvaluateG1GatePPO",
    "ConfigEvaluateG1PPO",
    "ConfigEvaluateG1TargetPPO",
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
    "ResultEvaluateG1TargetPPO",
    "ResultTrainAnymalPPO",
    "ResultTrainG1PPO",
    "StatsEvaluateAnymalTargetPPO",
    "StatsEvaluateG1GateCommandPPO",
    "StatsEvaluateG1GatePPO",
    "StatsEvaluateG1PPO",
    "StatsEvaluateG1TargetPPO",
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
    "evaluate_g1_target_ppo",
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
    g1_parser.add_argument("--command-zero-probability", type=float, default=g1_recipe.COMMAND_ZERO_PROBABILITY)
    g1_parser.add_argument("--command-curriculum-start", type=float, default=g1_recipe.COMMAND_CURRICULUM_START)
    g1_parser.add_argument("--command-curriculum-samples", type=int, default=g1_recipe.COMMAND_CURRICULUM_SAMPLES)
    g1_parser.add_argument("--no-command-randomization", action="store_true")
    g1_parser.add_argument(
        "--reward-mode",
        choices=("nanog1_dense", "sparse_command", "sparse_target", "dense_sparse_command"),
        default=g1_recipe.REWARD_MODE,
    )
    g1_parser.add_argument("--w-track-lin", type=float, default=g1_recipe.W_TRACK_LIN)
    g1_parser.add_argument("--w-track-ang", type=float, default=g1_recipe.W_TRACK_ANG)
    g1_parser.add_argument("--w-command-progress", type=float, default=g1_recipe.W_COMMAND_PROGRESS)
    g1_parser.add_argument("--w-lin-vel-z", type=float, default=g1_recipe.W_LIN_VEL_Z)
    g1_parser.add_argument("--w-ang-vel-xy", type=float, default=g1_recipe.W_ANG_VEL_XY)
    g1_parser.add_argument("--w-orientation", type=float, default=g1_recipe.W_ORIENTATION)
    g1_parser.add_argument("--w-torque", type=float, default=g1_recipe.W_TORQUE)
    g1_parser.add_argument("--w-action-rate", type=float, default=g1_recipe.W_ACTION_RATE)
    g1_parser.add_argument("--w-alive", type=float, default=g1_recipe.W_ALIVE)
    g1_parser.add_argument("--w-termination", type=float, default=g1_recipe.W_TERMINATION)
    g1_parser.add_argument("--w-sparse-command-success", type=float, default=g1_recipe.W_SPARSE_COMMAND_SUCCESS)
    g1_parser.add_argument("--w-target-progress", type=float, default=g1_recipe.W_TARGET_PROGRESS)
    g1_parser.add_argument(
        "--sparse-command-velocity-tolerance",
        type=float,
        default=g1_recipe.SPARSE_COMMAND_VELOCITY_TOLERANCE,
    )
    g1_parser.add_argument("--sparse-command-yaw-tolerance", type=float, default=g1_recipe.SPARSE_COMMAND_YAW_TOLERANCE)
    g1_parser.add_argument("--w-base-height", type=float, default=g1_recipe.W_BASE_HEIGHT)
    g1_parser.add_argument("--w-mechanical-power", type=float, default=g1_recipe.W_MECHANICAL_POWER)
    g1_parser.add_argument("--w-gait-contact", type=float, default=g1_recipe.W_GAIT_CONTACT)
    g1_parser.add_argument("--w-gait-swing", type=float, default=g1_recipe.W_GAIT_SWING)
    g1_parser.add_argument("--w-gait-swing-contact", type=float, default=g1_recipe.W_GAIT_SWING_CONTACT)
    g1_parser.add_argument("--w-gait-hip", type=float, default=g1_recipe.W_GAIT_HIP)
    g1_parser.add_argument("--w-feet-air-time", type=float, default=g1_recipe.W_FEET_AIR_TIME)
    g1_parser.add_argument("--feet-air-time-threshold", type=float, default=g1_recipe.FEET_AIR_TIME_THRESHOLD)
    g1_parser.add_argument("--w-feet-slide", type=float, default=g1_recipe.W_FEET_SLIDE)
    g1_parser.add_argument("--w-joint-deviation-hip", type=float, default=g1_recipe.W_JOINT_DEVIATION_HIP)
    g1_parser.add_argument("--w-joint-deviation-waist", type=float, default=g1_recipe.W_JOINT_DEVIATION_WAIST)
    g1_parser.add_argument("--w-joint-deviation-upper", type=float, default=g1_recipe.W_JOINT_DEVIATION_UPPER)
    g1_parser.add_argument("--w-joint-acc-legs", type=float, default=g1_recipe.W_JOINT_ACC_LEGS)
    g1_parser.add_argument("--w-joint-pos-limit-ankle", type=float, default=g1_recipe.W_JOINT_POS_LIMIT_ANKLE)
    g1_parser.add_argument("--target-x", type=float, default=g1_recipe.SPARSE_TARGET_POSITION[0])
    g1_parser.add_argument("--target-y", type=float, default=g1_recipe.SPARSE_TARGET_POSITION[1])
    g1_parser.add_argument("--sparse-target-radius", type=float, default=g1_recipe.SPARSE_TARGET_RADIUS)
    g1_parser.add_argument(
        "--sparse-target-success-upright-cos", type=float, default=g1_recipe.SPARSE_TARGET_SUCCESS_UPRIGHT_COS
    )
    g1_parser.add_argument(
        "--sparse-target-success-min-base-height",
        type=float,
        default=g1_recipe.SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT,
    )
    g1_parser.add_argument(
        "--sparse-target-success-max-base-height",
        type=float,
        default=g1_recipe.SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT,
    )
    g1_parser.add_argument("--target-distance-start", type=float, default=g1_recipe.SPARSE_TARGET_CURRICULUM_START)
    g1_parser.add_argument("--target-distance-end", type=float, default=g1_recipe.SPARSE_TARGET_CURRICULUM_END)
    g1_parser.add_argument("--target-curriculum-samples", type=int, default=g1_recipe.SPARSE_TARGET_CURRICULUM_SAMPLES)
    g1_parser.add_argument("--no-target-curriculum", action="store_true")
    g1_parser.add_argument(
        "--randomize-target-positions",
        action=argparse.BooleanOptionalAction,
        default=g1_recipe.SPARSE_TARGET_RANDOMIZE,
    )
    g1_parser.add_argument("--target-angle-min", type=float, default=g1_recipe.SPARSE_TARGET_ANGLE_MIN)
    g1_parser.add_argument("--target-angle-max", type=float, default=g1_recipe.SPARSE_TARGET_ANGLE_MAX)
    g1_parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    g1_parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    g1_parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    g1_parser.add_argument("--joint-friction-model", choices=("hard", "mujoco"), default=g1_recipe.JOINT_FRICTION_MODEL)
    g1_parser.add_argument("--joint-friction-scale", type=float, default=g1_recipe.JOINT_FRICTION_SCALE)
    g1_parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path: nanoG1-style explicit clamped PD torques or legacy PhoenX drive rows.",
    )
    g1_parser.add_argument("--action-scale", type=float, default=g1_recipe.ACTION_SCALE)
    g1_parser.add_argument(
        "--observation-mode",
        choices=("nanog1", "isaaclab_flat"),
        default=g1_recipe.OBSERVATION_MODE,
        help="G1 policy observation layout. Use isaaclab_flat only for checkpoints trained with that mode.",
    )
    g1_parser.add_argument("--parse-meshes", action="store_true")
    g1_parser.add_argument(
        "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY
    )
    g1_parser.add_argument("--ground-friction", type=float, default=g1_recipe.GROUND_FRICTION)
    g1_parser.add_argument("--foot-box-xy-scale", type=float, default=g1_recipe.FOOT_BOX_XY_SCALE)
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
    g1_parser.add_argument(
        "--activation",
        choices=("relu", "elu"),
        default=g1_recipe.ACTIVATION,
        help="Hidden activation for the MLP policy backbone. Ignored by puffer_mingru.",
    )
    g1_parser.add_argument("--log-std-init", type=float, default=g1_recipe.LOG_STD_INIT)
    g1_parser.add_argument("--minibatch-size", type=int, default=g1_recipe.MINIBATCH_SIZE)
    g1_parser.add_argument("--train-epochs", type=int, default=g1_recipe.TRAIN_EPOCHS)
    g1_parser.add_argument("--actor-lr", type=float, default=g1_recipe.ACTOR_LR)
    g1_parser.add_argument("--critic-lr", type=float, default=g1_recipe.CRITIC_LR)
    g1_parser.add_argument("--gamma", type=float, default=g1_recipe.GAMMA)
    g1_parser.add_argument("--gae-lambda", type=float, default=g1_recipe.GAE_LAMBDA)
    g1_parser.add_argument("--entropy-coeff", type=float, default=g1_recipe.ENTROPY_COEFF)
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
        "--joint-friction-model", choices=("hard", "mujoco"), default=g1_recipe.JOINT_FRICTION_MODEL
    )
    g1_eval_parser.add_argument("--joint-friction-scale", type=float, default=g1_recipe.JOINT_FRICTION_SCALE)
    g1_eval_parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path used during evaluation.",
    )
    g1_eval_parser.add_argument("--action-scale", type=float, default=g1_recipe.ACTION_SCALE)
    g1_eval_parser.add_argument(
        "--observation-mode",
        choices=("nanog1", "isaaclab_flat"),
        default=g1_recipe.OBSERVATION_MODE,
        help="G1 policy observation layout used by the checkpoint.",
    )
    g1_eval_parser.add_argument("--parse-meshes", action="store_true")
    g1_eval_parser.add_argument(
        "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY
    )
    g1_eval_parser.add_argument("--ground-friction", type=float, default=g1_recipe.GROUND_FRICTION)
    g1_eval_parser.add_argument("--foot-box-xy-scale", type=float, default=g1_recipe.FOOT_BOX_XY_SCALE)
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
        "--joint-friction-model", choices=("hard", "mujoco"), default=g1_recipe.JOINT_FRICTION_MODEL
    )
    g1_gate_parser.add_argument("--joint-friction-scale", type=float, default=g1_recipe.JOINT_FRICTION_SCALE)
    g1_gate_parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path used during the quality gate.",
    )
    g1_gate_parser.add_argument("--action-scale", type=float, default=g1_recipe.ACTION_SCALE)
    g1_gate_parser.add_argument(
        "--observation-mode",
        choices=("nanog1", "isaaclab_flat"),
        default=g1_recipe.OBSERVATION_MODE,
        help="G1 policy observation layout used by the checkpoint.",
    )
    g1_gate_parser.add_argument("--parse-meshes", action="store_true")
    g1_gate_parser.add_argument(
        "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY
    )
    g1_gate_parser.add_argument("--ground-friction", type=float, default=g1_recipe.GROUND_FRICTION)
    g1_gate_parser.add_argument("--foot-box-xy-scale", type=float, default=g1_recipe.FOOT_BOX_XY_SCALE)
    g1_gate_parser.add_argument(
        "--rigid-contact-max-per-world",
        type=int,
        default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
        help="Per-world G1 rigid-contact capacity; 0 keeps SolverPhoenX auto-sizing.",
    )
    g1_gate_parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    g1_gate_parser.add_argument("--stochastic", action="store_true")
    g1_gate_parser.add_argument("--no-fail-on-gate", action="store_true")

    g1_target_parser = subparsers.add_parser(
        "target-g1-ppo", help="Evaluate a saved Unitree G1 PPO checkpoint on sparse target-reaching metrics"
    )
    g1_target_parser.add_argument("--checkpoint", required=True)
    g1_target_parser.add_argument("--steps", type=int, default=300)
    g1_target_parser.add_argument("--world-count", type=int, default=64)
    g1_target_parser.add_argument("--device", default=None)
    g1_target_parser.add_argument("--seed", type=int, default=1000)
    g1_target_parser.add_argument("--target-x", type=float, action="append", default=None)
    g1_target_parser.add_argument("--target-y", type=float, action="append", default=None)
    g1_target_parser.add_argument("--sparse-target-radius", type=float, default=g1_recipe.SPARSE_TARGET_RADIUS)
    g1_target_parser.add_argument(
        "--sparse-target-success-upright-cos", type=float, default=g1_recipe.SPARSE_TARGET_SUCCESS_UPRIGHT_COS
    )
    g1_target_parser.add_argument(
        "--sparse-target-success-min-base-height",
        type=float,
        default=g1_recipe.SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT,
    )
    g1_target_parser.add_argument(
        "--sparse-target-success-max-base-height",
        type=float,
        default=g1_recipe.SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT,
    )
    g1_target_parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    g1_target_parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    g1_target_parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    g1_target_parser.add_argument(
        "--joint-friction-model", choices=("hard", "mujoco"), default=g1_recipe.JOINT_FRICTION_MODEL
    )
    g1_target_parser.add_argument("--joint-friction-scale", type=float, default=g1_recipe.JOINT_FRICTION_SCALE)
    g1_target_parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path used during target evaluation.",
    )
    g1_target_parser.add_argument("--action-scale", type=float, default=g1_recipe.ACTION_SCALE)
    g1_target_parser.add_argument(
        "--observation-mode",
        choices=("nanog1", "isaaclab_flat"),
        default=g1_recipe.OBSERVATION_MODE,
        help="G1 policy observation layout used by the checkpoint.",
    )
    g1_target_parser.add_argument("--parse-meshes", action="store_true")
    g1_target_parser.add_argument(
        "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY
    )
    g1_target_parser.add_argument("--ground-friction", type=float, default=g1_recipe.GROUND_FRICTION)
    g1_target_parser.add_argument("--foot-box-xy-scale", type=float, default=g1_recipe.FOOT_BOX_XY_SCALE)
    g1_target_parser.add_argument(
        "--rigid-contact-max-per-world",
        type=int,
        default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
        help="Per-world G1 rigid-contact capacity; 0 keeps SolverPhoenX auto-sizing.",
    )
    g1_target_parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    g1_target_parser.add_argument("--max-tilt-degrees", type=float, default=30.0)
    g1_target_parser.add_argument("--min-valid-base-height", type=float, default=0.35)
    g1_target_parser.add_argument("--max-valid-base-height", type=float, default=1.10)
    g1_target_parser.add_argument("--stochastic", action="store_true")

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
            joint_friction_model=args.joint_friction_model,
            joint_friction_scale=args.joint_friction_scale,
            actuation_model=args.actuation_model,
            action_scale=args.action_scale,
            controlled_action_count=args.controlled_action_count,
            observation_mode=args.observation_mode,
            reward_mode=args.reward_mode,
            w_track_lin=args.w_track_lin,
            w_track_ang=args.w_track_ang,
            w_command_progress=args.w_command_progress,
            w_lin_vel_z=args.w_lin_vel_z,
            w_ang_vel_xy=args.w_ang_vel_xy,
            w_orientation=args.w_orientation,
            w_torque=args.w_torque,
            w_action_rate=args.w_action_rate,
            w_alive=args.w_alive,
            w_termination=args.w_termination,
            w_sparse_command_success=args.w_sparse_command_success,
            w_target_progress=args.w_target_progress,
            w_mechanical_power=args.w_mechanical_power,
            sparse_command_velocity_tolerance=args.sparse_command_velocity_tolerance,
            sparse_command_yaw_tolerance=args.sparse_command_yaw_tolerance,
            w_gait_contact=args.w_gait_contact,
            w_gait_swing=args.w_gait_swing,
            w_gait_swing_contact=args.w_gait_swing_contact,
            w_gait_hip=args.w_gait_hip,
            w_base_height=args.w_base_height,
            w_feet_air_time=args.w_feet_air_time,
            feet_air_time_threshold=args.feet_air_time_threshold,
            w_feet_slide=args.w_feet_slide,
            w_joint_deviation_hip=args.w_joint_deviation_hip,
            w_joint_deviation_waist=args.w_joint_deviation_waist,
            w_joint_deviation_upper=args.w_joint_deviation_upper,
            w_joint_acc_legs=args.w_joint_acc_legs,
            w_joint_pos_limit_ankle=args.w_joint_pos_limit_ankle,
            sparse_target_position=(args.target_x, args.target_y),
            sparse_target_radius=args.sparse_target_radius,
            sparse_target_success_upright_cos=args.sparse_target_success_upright_cos,
            sparse_target_success_min_base_height=args.sparse_target_success_min_base_height,
            sparse_target_success_max_base_height=args.sparse_target_success_max_base_height,
            parse_meshes=args.parse_meshes,
            contact_geometry=args.contact_geometry,
            ground_friction=args.ground_friction,
            foot_box_xy_scale=args.foot_box_xy_scale,
            rigid_contact_max_per_world=args.rigid_contact_max_per_world,
        )
        ppo_config = g1_recipe.default_g1_ppo_config(
            minibatch_size=args.minibatch_size,
            train_epochs=args.train_epochs,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coeff=args.entropy_coeff,
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
                command_zero_probability=args.command_zero_probability,
                command_curriculum_start=args.command_curriculum_start,
                command_curriculum_samples=args.command_curriculum_samples,
                use_target_curriculum=not args.no_target_curriculum,
                target_distance_start=args.target_distance_start,
                target_distance_end=args.target_distance_end,
                target_curriculum_samples=args.target_curriculum_samples,
                randomize_target_positions=bool(args.randomize_target_positions),
                target_angle_min=args.target_angle_min,
                target_angle_max=args.target_angle_max,
                reset_recurrent_state_on_rollout_start=bool(args.reset_recurrent_state_on_rollout_start),
                squash_actions=bool(args.squash_actions),
                activation=args.activation,
                log_std_init=args.log_std_init,
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
            joint_friction_model=args.joint_friction_model,
            joint_friction_scale=args.joint_friction_scale,
            actuation_model=args.actuation_model,
            action_scale=args.action_scale,
            controlled_action_count=args.controlled_action_count,
            observation_mode=args.observation_mode,
            parse_meshes=args.parse_meshes,
            contact_geometry=args.contact_geometry,
            ground_friction=args.ground_friction,
            foot_box_xy_scale=args.foot_box_xy_scale,
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
            joint_friction_model=args.joint_friction_model,
            joint_friction_scale=args.joint_friction_scale,
            actuation_model=args.actuation_model,
            action_scale=args.action_scale,
            controlled_action_count=args.controlled_action_count,
            observation_mode=args.observation_mode,
            parse_meshes=args.parse_meshes,
            contact_geometry=args.contact_geometry,
            ground_friction=args.ground_friction,
            foot_box_xy_scale=args.foot_box_xy_scale,
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
    if args.command == "target-g1-ppo":
        if (args.target_x is None) != (args.target_y is None):
            parser.error("--target-x and --target-y must be provided together")
        if args.target_x is None:
            target_positions = ConfigEvaluateG1TargetPPO().target_positions
        else:
            if len(args.target_x) != len(args.target_y):
                parser.error("--target-x and --target-y must have the same count")
            target_positions = tuple((float(x), float(y)) for x, y in zip(args.target_x, args.target_y, strict=True))
        env_config = ConfigEnvG1PhoenX(
            world_count=args.world_count,
            reward_mode="sparse_target",
            sparse_target_radius=args.sparse_target_radius,
            sparse_target_success_upright_cos=args.sparse_target_success_upright_cos,
            sparse_target_success_min_base_height=args.sparse_target_success_min_base_height,
            sparse_target_success_max_base_height=args.sparse_target_success_max_base_height,
            sim_substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            joint_friction_model=args.joint_friction_model,
            joint_friction_scale=args.joint_friction_scale,
            actuation_model=args.actuation_model,
            action_scale=args.action_scale,
            controlled_action_count=args.controlled_action_count,
            observation_mode=args.observation_mode,
            parse_meshes=args.parse_meshes,
            contact_geometry=args.contact_geometry,
            ground_friction=args.ground_friction,
            foot_box_xy_scale=args.foot_box_xy_scale,
            rigid_contact_max_per_world=args.rigid_contact_max_per_world,
        )
        trainer = load_ppo_checkpoint(args.checkpoint, device=args.device)
        result = evaluate_g1_target_ppo(
            trainer,
            ConfigEvaluateG1TargetPPO(
                env_config=env_config,
                target_positions=target_positions,
                steps=args.steps,
                device=args.device,
                deterministic=not args.stochastic,
                seed=args.seed,
                max_tilt_degrees=args.max_tilt_degrees,
                min_valid_base_height=args.min_valid_base_height,
                max_valid_base_height=args.max_valid_base_height,
            ),
        )
        print(json.dumps({"stats": [asdict(stat) for stat in result.stats]}, sort_keys=True))
        return 0
    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
