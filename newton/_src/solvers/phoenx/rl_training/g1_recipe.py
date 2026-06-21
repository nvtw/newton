# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default PhoenX G1 PPO recipe.

This file is the single tuning surface for the experimental pure-Warp G1
training run: environment timing, solver decimation, reward weights, PPO
hyperparameters, and the left/right symmetry regularizer.
"""

from __future__ import annotations

from typing import Any

WORLD_COUNT = 4096
FRAME_DT = 1.0 / 50.0
SIM_SUBSTEPS = 10
SOLVER_ITERATIONS = 4
VELOCITY_ITERATIONS = 1
ACTION_SCALE = 0.25
CONTROLLED_ACTION_COUNT = 12
COMMAND = (0.8, 0.0, 0.0)
MAX_EPISODE_STEPS = 1000
RESET_NOISE = 0.05
MIN_BASE_HEIGHT = 0.35
MIN_UPRIGHT_COS = 0.6
PHASE_PERIOD = 40

W_TRACK_LIN = 2.5
W_TRACK_ANG = 1.25
W_LIN_VEL_Z = -2.0
W_ANG_VEL_XY = -1.3
W_ORIENTATION = -10.0
W_TORQUE = -2.0e-5
W_ACTION_RATE = -0.01
W_ALIVE = 3.0
W_TERMINATION = -1.0
GAIT_STANCE_FRACTION = 0.55
W_GAIT_CONTACT = 0.5
W_GAIT_SWING = -20.0
W_GAIT_HIP = -4.0
GAIT_FOOT_HEIGHT = 0.08
W_BASE_HEIGHT = -10.0
BASE_HEIGHT_TARGET = 0.78

PARSE_MESHES = False
CONTACT_GEOMETRY = "nanog1_foot_boxes"
AUTO_RESET = True
RIGID_CONTACT_MAX_PER_WORLD = 32
THREADS_PER_WORLD: int | str = "auto"
MULTI_WORLD_SCHEDULER = "auto"
PREPARE_REFRESH_STRIDE: int | str = "auto"

TRAIN_ITERATIONS = 100
ROLLOUT_STEPS = 64
HIDDEN_LAYERS = (128, 128, 128)
ACTIVATION = "relu"
LOG_STD_INIT = -0.5
SQUASH_ACTIONS = False
SEED = 42
RANDOMIZE_COMMANDS = True
COMMAND_SAMPLING = "episode"
COMMAND_X_RANGE = (-1.0, 1.0)
COMMAND_Y_RANGE = (-0.6, 0.6)
COMMAND_YAW_RANGE = (-1.0, 1.0)
COMMAND_ZERO_PROBABILITY = 0.1
COMMAND_RESAMPLE_STEPS = 500

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
        "phase_period": PHASE_PERIOD,
        "w_track_lin": W_TRACK_LIN,
        "w_track_ang": W_TRACK_ANG,
        "w_lin_vel_z": W_LIN_VEL_Z,
        "w_ang_vel_xy": W_ANG_VEL_XY,
        "w_orientation": W_ORIENTATION,
        "w_torque": W_TORQUE,
        "w_action_rate": W_ACTION_RATE,
        "w_alive": W_ALIVE,
        "w_termination": W_TERMINATION,
        "gait_stance_fraction": GAIT_STANCE_FRACTION,
        "w_gait_contact": W_GAIT_CONTACT,
        "w_gait_swing": W_GAIT_SWING,
        "w_gait_hip": W_GAIT_HIP,
        "gait_foot_height": GAIT_FOOT_HEIGHT,
        "w_base_height": W_BASE_HEIGHT,
        "base_height_target": BASE_HEIGHT_TARGET,
        "parse_meshes": PARSE_MESHES,
        "contact_geometry": CONTACT_GEOMETRY,
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
        "manual_mlp_weight_grad_dtype": MANUAL_MLP_WEIGHT_GRAD_DTYPE,
        "manual_mlp_forward_dtype": MANUAL_MLP_FORWARD_DTYPE,
        "vtrace_rho_clip": VTRACE_RHO_CLIP,
        "vtrace_c_clip": VTRACE_C_CLIP,
        "normalize_advantages": NORMALIZE_ADVANTAGES,
        "reward_clip": REWARD_CLIP,
        "max_grad_norm": MAX_GRAD_NORM,
        "mirror_loss_coeff": MIRROR_LOSS_COEFF,
    }
    values.update(overrides)
    return ConfigPPO(**values)
