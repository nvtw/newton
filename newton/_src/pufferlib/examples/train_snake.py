# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Snake with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_snake.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.snake import SnakeEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

config = PPOConfig(
    num_envs=4096,
    horizon=64,
    obs_dim=121,
    num_actions=4,
    hidden=128,
    total_timesteps=30_000_000,
    lr=0.015,
    gamma=0.99,
    gae_lambda=0.90,
    clip_coef=0.2,
    vf_coef=2.0,
    vf_clip_coef=0.2,
    ent_coef=0.001,
    max_grad_norm=1.5,
    momentum=0.95,
    replay_ratio=1.0,
    minibatch_size=32768,
    rho_clip=1.0,
    c_clip=1.0,
    env_name="Snake",
)


def make_env(device):
    return SnakeEnv(num_envs=config.num_envs, device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
