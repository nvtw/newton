# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train CartPole with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_cartpole.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.cartpole import CartPoleEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

config = PPOConfig(
    num_envs=4096,
    horizon=32,
    obs_dim=4,
    num_actions=2,
    hidden=32,
    total_timesteps=5_642_560,
    lr=0.1,
    gamma=0.8,
    gae_lambda=0.922,
    clip_coef=0.144,
    vf_coef=1.78,
    vf_clip_coef=3.91,
    ent_coef=0.037,
    max_grad_norm=0.33,
    momentum=0.943,
    replay_ratio=0.381,
    minibatch_size=16384,
    rho_clip=2.91,
    c_clip=1.66,
    env_name="CartPole",
    best_return_init=0.0,
    return_format="6.1f",
)


def make_env(device):
    return CartPoleEnv(num_envs=config.num_envs, device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
