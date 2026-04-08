# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Pong with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_pong.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.pong import PongEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

FRAMESKIP = 8

config = PPOConfig(
    num_envs=1024,
    horizon=32,
    obs_dim=8,
    num_actions=3,
    hidden=32,
    total_timesteps=5_000_000,
    lr=0.1,
    gamma=0.935,
    gae_lambda=0.992,
    clip_coef=0.823,
    vf_coef=5.0,
    vf_clip_coef=4.96,
    ent_coef=0.0004,
    max_grad_norm=0.753,
    momentum=0.5,
    replay_ratio=3.05,
    minibatch_size=32768,
    rho_clip=4.88,
    c_clip=1.49,
    env_name="Pong",
    banner_extras={"Frameskip": FRAMESKIP},
)


def make_env(device):
    return PongEnv(num_envs=config.num_envs, frameskip=FRAMESKIP,
                   device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
