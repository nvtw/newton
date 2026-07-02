# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default PhoenX G1 PPO recipe.

This file is the single tuning surface for the experimental pure-Warp G1
training run: environment timing, solver decimation, reward weights, PPO
hyperparameters, and the left/right symmetry regularizer.
"""

from __future__ import annotations

from typing import Any

WORLD_COUNT = 8192
FRAME_DT = 1.0 / 50.0
SIM_SUBSTEPS = 3
SOLVER_ITERATIONS = 2
VELOCITY_ITERATIONS = 1
JOINT_FRICTION_MODEL = "mujoco"
JOINT_FRICTION_SCALE = 1.0
ACTUATION_MODEL = "explicit_torque"
ACTION_SCALE = 0.25
CONTROLLED_ACTION_COUNT = 12
COMMAND = (0.8, 0.0, 0.0)
MAX_EPISODE_STEPS = 1000
RESET_NOISE = 0.05
MIN_BASE_HEIGHT = 0.35
MIN_UPRIGHT_COS = 0.6
MAX_ABS_ROOT_POSITION = 100.0
MAX_ABS_ROOT_LINEAR_VELOCITY = 50.0
MAX_ABS_ROOT_ANGULAR_VELOCITY = 100.0
MAX_ABS_JOINT_POSITION = 20.0
MAX_ABS_JOINT_VELOCITY = 200.0
PHASE_PERIOD = 40
OBSERVATION_MODE = "nanog1"

W_TRACK_LIN = 2.5
W_TRACK_ANG = 2.5
W_COMMAND_PROGRESS = 0.0
W_LIN_VEL_Z = -2.0
W_ANG_VEL_XY = -1.7
W_ORIENTATION = -10.0
W_TORQUE = -2.0e-5
W_ACTION_RATE = -0.3
ANGULAR_FINE_TUNE_START_SAMPLES = 137_363_456
ANGULAR_FINE_TUNE_W_ANG_VEL_XY = -4.0
ANGULAR_FINE_TUNE_LR_ANNEAL_TIMESTEPS = 175_000_000
W_ALIVE = 3.0
W_TERMINATION = -1.0
REWARD_MODE = "nanog1_dense"
W_SPARSE_COMMAND_SUCCESS = 5.0
W_TARGET_PROGRESS = 0.0
SPARSE_COMMAND_VELOCITY_TOLERANCE = 0.35
SPARSE_COMMAND_YAW_TOLERANCE = 0.4
SPARSE_TARGET_POSITION = (0.6, 0.0)
SPARSE_TARGET_RADIUS = 0.4
SPARSE_TARGET_SUCCESS_UPRIGHT_COS = 0.8660254037844386
SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT = 0.35
SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT = 1.10
SPARSE_TARGET_CURRICULUM_START = 0.6
SPARSE_TARGET_CURRICULUM_END = 1.4
SPARSE_TARGET_CURRICULUM_SAMPLES = 80_000_000
SPARSE_TARGET_RANDOMIZE = False
SPARSE_TARGET_ANGLE_MIN = -0.5
SPARSE_TARGET_ANGLE_MAX = 0.5
W_MECHANICAL_POWER = -1.0e-4
GAIT_STANCE_FRACTION = 0.55
W_GAIT_CONTACT = 0.5
W_GAIT_SWING = -20.0
W_GAIT_SWING_CONTACT = 0.0
W_GAIT_HIP = -4.0
GAIT_FOOT_HEIGHT = 0.08
W_BASE_HEIGHT = -10.0
BASE_HEIGHT_TARGET = 0.78
W_FEET_AIR_TIME = 0.0
FEET_AIR_TIME_THRESHOLD = 0.4
W_FEET_SLIDE = 0.0
FOOT_CONTACT_NORMAL_IMPULSE_THRESHOLD = 1.0e-6
W_JOINT_DEVIATION_HIP = 0.0
W_JOINT_DEVIATION_WAIST = 0.0
W_JOINT_DEVIATION_UPPER = 0.0
W_JOINT_ACC_LEGS = 0.0
W_JOINT_POS_LIMIT_ANKLE = 0.0

PARSE_MESHES = False
PARSE_VISUALS = False
CONTACT_GEOMETRY = "nanog1_foot_boxes"
GROUND_FRICTION = 0.6
FOOT_BOX_XY_SCALE = 1.0
AUTO_RESET = True
RIGID_CONTACT_MAX_PER_WORLD = 32
THREADS_PER_WORLD: int | str = "auto"
MULTI_WORLD_SCHEDULER = "auto"
PREPARE_REFRESH_STRIDE: int | str = "auto"

TRAIN_ITERATIONS = 100
ROLLOUT_STEPS = 64
HIDDEN_LAYERS = (128, 128, 128)
ACTIVATION = "relu"
POLICY_NETWORK = "puffer_mingru"
LOG_STD_INIT = 0.0
SQUASH_ACTIONS = False
SEED = 42
RANDOMIZE_COMMANDS = True
COMMAND_SAMPLING = "episode"
COMMAND_X_RANGE = (-1.0, 1.0)
COMMAND_Y_RANGE = (-0.6, 0.6)
COMMAND_YAW_RANGE = (-1.0, 1.0)
COMMAND_ZERO_PROBABILITY = 0.1
COMMAND_RESAMPLE_STEPS = 500
COMMAND_CURRICULUM_START = 1.0
COMMAND_CURRICULUM_SAMPLES = 0
RESET_RECURRENT_STATE_ON_ROLLOUT_START = True

GAMMA = 0.97
GAE_LAMBDA = 0.9
CLIP_RATIO = 0.2
ENTROPY_COEFF = 1.0e-5
VALUE_LOSS_COEFF = 0.5
VALUE_CLIP_RANGE = 20.0
ACTOR_LR = 2.0e-2
CRITIC_LR = 2.0e-2
ANNEAL_LR = True
LR_ANNEAL_TIMESTEPS = 150_000_000
MIN_LR_RATIO = 0.0
OPTIMIZER = "muon"
OPTIMIZER_EPS = 1.0e-12
OPTIMIZER_WEIGHT_DECAY = 0.0
MUON_MOMENTUM = 0.9
TRAIN_EPOCHS = 3
MINIBATCH_SIZE = 32768
REPLAY_RATIO = 3.0
PRIORITY_ALPHA = 0.4
PRIORITY_BETA = 1.0
MANUAL_ACTOR_BACKWARD = True
MANUAL_CRITIC_BACKWARD = True
SHARED_VALUE_NETWORK = True
MANUAL_MLP_WEIGHT_GRAD_DTYPE = "bfloat16"
MANUAL_MLP_FORWARD_DTYPE = "bfloat16"
VTRACE_RHO_CLIP = 3.0
VTRACE_C_CLIP = 3.0
NORMALIZE_ADVANTAGES = True
REWARD_CLIP = 1.0
PUFFER_VTRACE_ADVANTAGE = True
MAX_GRAD_NORM = 0.3
MIRROR_LOSS_COEFF = 0.25


def default_g1_env_config(**overrides: Any):
    """Return the default PhoenX G1 environment config."""

    from .g1 import ConfigEnvG1PhoenX  # noqa: PLC0415

    values = {
        "world_count": WORLD_COUNT,
        "frame_dt": FRAME_DT,
        "sim_substeps": SIM_SUBSTEPS,
        "solver_iterations": SOLVER_ITERATIONS,
        "velocity_iterations": VELOCITY_ITERATIONS,
        "joint_friction_model": JOINT_FRICTION_MODEL,
        "joint_friction_scale": JOINT_FRICTION_SCALE,
        "actuation_model": ACTUATION_MODEL,
        "action_scale": ACTION_SCALE,
        "controlled_action_count": CONTROLLED_ACTION_COUNT,
        "command": COMMAND,
        "command_x_range": COMMAND_X_RANGE,
        "command_y_range": COMMAND_Y_RANGE,
        "command_yaw_range": COMMAND_YAW_RANGE,
        "randomize_commands_on_reset": False,
        "command_zero_probability": COMMAND_ZERO_PROBABILITY,
        "command_resample_steps": COMMAND_RESAMPLE_STEPS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "reset_noise": RESET_NOISE,
        "min_base_height": MIN_BASE_HEIGHT,
        "min_upright_cos": MIN_UPRIGHT_COS,
        "max_abs_root_position": MAX_ABS_ROOT_POSITION,
        "max_abs_root_linear_velocity": MAX_ABS_ROOT_LINEAR_VELOCITY,
        "max_abs_root_angular_velocity": MAX_ABS_ROOT_ANGULAR_VELOCITY,
        "max_abs_joint_position": MAX_ABS_JOINT_POSITION,
        "max_abs_joint_velocity": MAX_ABS_JOINT_VELOCITY,
        "phase_period": PHASE_PERIOD,
        "observation_mode": OBSERVATION_MODE,
        "w_track_lin": W_TRACK_LIN,
        "w_track_ang": W_TRACK_ANG,
        "w_command_progress": W_COMMAND_PROGRESS,
        "w_lin_vel_z": W_LIN_VEL_Z,
        "w_ang_vel_xy": W_ANG_VEL_XY,
        "w_orientation": W_ORIENTATION,
        "w_torque": W_TORQUE,
        "w_action_rate": W_ACTION_RATE,
        "w_alive": W_ALIVE,
        "w_termination": W_TERMINATION,
        "reward_mode": REWARD_MODE,
        "w_sparse_command_success": W_SPARSE_COMMAND_SUCCESS,
        "w_target_progress": W_TARGET_PROGRESS,
        "sparse_command_velocity_tolerance": SPARSE_COMMAND_VELOCITY_TOLERANCE,
        "sparse_command_yaw_tolerance": SPARSE_COMMAND_YAW_TOLERANCE,
        "sparse_target_position": SPARSE_TARGET_POSITION,
        "sparse_target_radius": SPARSE_TARGET_RADIUS,
        "sparse_target_success_upright_cos": SPARSE_TARGET_SUCCESS_UPRIGHT_COS,
        "sparse_target_success_min_base_height": SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT,
        "sparse_target_success_max_base_height": SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT,
        "w_mechanical_power": W_MECHANICAL_POWER,
        "gait_stance_fraction": GAIT_STANCE_FRACTION,
        "w_gait_contact": W_GAIT_CONTACT,
        "w_gait_swing": W_GAIT_SWING,
        "w_gait_swing_contact": W_GAIT_SWING_CONTACT,
        "w_gait_hip": W_GAIT_HIP,
        "gait_foot_height": GAIT_FOOT_HEIGHT,
        "w_base_height": W_BASE_HEIGHT,
        "base_height_target": BASE_HEIGHT_TARGET,
        "w_feet_air_time": W_FEET_AIR_TIME,
        "feet_air_time_threshold": FEET_AIR_TIME_THRESHOLD,
        "w_feet_slide": W_FEET_SLIDE,
        "foot_contact_normal_impulse_threshold": FOOT_CONTACT_NORMAL_IMPULSE_THRESHOLD,
        "w_joint_deviation_hip": W_JOINT_DEVIATION_HIP,
        "w_joint_deviation_waist": W_JOINT_DEVIATION_WAIST,
        "w_joint_deviation_upper": W_JOINT_DEVIATION_UPPER,
        "w_joint_acc_legs": W_JOINT_ACC_LEGS,
        "w_joint_pos_limit_ankle": W_JOINT_POS_LIMIT_ANKLE,
        "parse_meshes": PARSE_MESHES,
        "parse_visuals": PARSE_VISUALS,
        "contact_geometry": CONTACT_GEOMETRY,
        "ground_friction": GROUND_FRICTION,
        "foot_box_xy_scale": FOOT_BOX_XY_SCALE,
        "auto_reset": AUTO_RESET,
        "rigid_contact_max_per_world": RIGID_CONTACT_MAX_PER_WORLD,
        "threads_per_world": THREADS_PER_WORLD,
        "multi_world_scheduler": MULTI_WORLD_SCHEDULER,
        "prepare_refresh_stride": PREPARE_REFRESH_STRIDE,
    }
    values.update(overrides)
    return ConfigEnvG1PhoenX(**values)


def default_g1_ppo_config(**overrides: Any):
    """Return the default PhoenX G1 PPO config."""

    from .ppo import ConfigPPO  # noqa: PLC0415

    values = {
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_ratio": CLIP_RATIO,
        "entropy_coeff": ENTROPY_COEFF,
        "value_loss_coeff": VALUE_LOSS_COEFF,
        "value_clip_range": VALUE_CLIP_RANGE,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "anneal_lr": ANNEAL_LR,
        "lr_anneal_timesteps": LR_ANNEAL_TIMESTEPS,
        "min_lr_ratio": MIN_LR_RATIO,
        "optimizer": OPTIMIZER,
        "optimizer_eps": OPTIMIZER_EPS,
        "optimizer_weight_decay": OPTIMIZER_WEIGHT_DECAY,
        "muon_momentum": MUON_MOMENTUM,
        "train_epochs": TRAIN_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "replay_ratio": REPLAY_RATIO,
        "priority_alpha": PRIORITY_ALPHA,
        "priority_beta": PRIORITY_BETA,
        "manual_actor_backward": MANUAL_ACTOR_BACKWARD,
        "manual_critic_backward": MANUAL_CRITIC_BACKWARD,
        "shared_value_network": SHARED_VALUE_NETWORK,
        "policy_network": POLICY_NETWORK,
        "manual_mlp_weight_grad_dtype": MANUAL_MLP_WEIGHT_GRAD_DTYPE,
        "manual_mlp_forward_dtype": MANUAL_MLP_FORWARD_DTYPE,
        "vtrace_rho_clip": VTRACE_RHO_CLIP,
        "vtrace_c_clip": VTRACE_C_CLIP,
        "normalize_advantages": NORMALIZE_ADVANTAGES,
        "reward_clip": REWARD_CLIP,
        "puffer_vtrace_advantage": PUFFER_VTRACE_ADVANTAGE,
        "max_grad_norm": MAX_GRAD_NORM,
        "mirror_loss_coeff": MIRROR_LOSS_COEFF,
    }
    values.update(overrides)
    return ConfigPPO(**values)
