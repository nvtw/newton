# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Asteroids with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_asteroids.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.asteroids import AsteroidsEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

FRAMESKIP = 1

config = PPOConfig(
    num_envs=4096,
    horizon=64,
    obs_dim=104,
    num_actions=4,
    hidden=128,
    total_timesteps=30_000_000,
    lr=0.005,
    gamma=0.9999,
    gae_lambda=0.84,
    clip_coef=0.186,
    vf_coef=2.96,
    vf_clip_coef=0.1,
    ent_coef=0.00166,
    max_grad_norm=0.731,
    momentum=0.9755,
    replay_ratio=1.0,
    minibatch_size=8192,
    rho_clip=4.133,
    c_clip=1.077,
    env_name="Asteroids",
    banner_extras={"Frameskip": FRAMESKIP},
)


def make_env(device):
    return AsteroidsEnv(num_envs=config.num_envs, frameskip=FRAMESKIP,
                        device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
