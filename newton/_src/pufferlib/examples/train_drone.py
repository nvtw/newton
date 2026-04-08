# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Drone Hover with PufferLib PPO on Warp.

A Crazyflie quadrotor must hover near a randomly placed target.
4 continuous actions (motor commands), 23 observations.

Usage::

    uv run python warp/pufferlib/examples/train_drone.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.drone import DroneEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

config = PPOConfig(
    num_envs=2048,
    horizon=32,
    obs_dim=23,
    num_actions=4,
    hidden=128,
    total_timesteps=40_000_000,
    lr=0.00975,
    anneal_lr=True,
    gamma=0.981,
    gae_lambda=0.884,
    clip_coef=0.068,
    vf_coef=4.42,
    vf_clip_coef=3.42,
    ent_coef=9.63e-5,
    max_grad_norm=0.692,
    momentum=0.914,
    replay_ratio=2.25,
    minibatch_size=8192,
    rho_clip=3.33,
    c_clip=4.37,
    prio_alpha=0.427,
    prio_beta0=1.0,
    continuous=True,
    env_name="Drone Hover",
    best_return_init=-100.0,
    return_format="8.2f",
    step_width=11,
    log_interval=10,
)


def make_env(device):
    return DroneEnv(
        num_envs=config.num_envs,
        device=device,
        seed=config.seed,
        hover_target_dist=5.0,
        hover_dist=0.1,
        hover_omega=0.1,
        hover_vel=0.1,
        alpha_dist=0.782192,
        alpha_hover=0.071445,
        alpha_shaping=3.9754,
        alpha_omega=0.00135588,
    )


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
