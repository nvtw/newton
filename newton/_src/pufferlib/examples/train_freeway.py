# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Freeway with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_freeway.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.freeway import FreewayEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

FRAMESKIP = 4

config = PPOConfig(
    num_envs=16384,
    horizon=64,
    obs_dim=34,
    num_actions=3,
    hidden=128,
    total_timesteps=100_000_000,
    lr=0.00357,
    gamma=0.9887,
    gae_lambda=0.759,
    clip_coef=0.168,
    vf_coef=3.512,
    vf_clip_coef=0.1796,
    ent_coef=0.000235,
    max_grad_norm=5.0,
    momentum=0.9737,
    replay_ratio=2.08,
    minibatch_size=32768,
    rho_clip=2.368,
    c_clip=1.292,
    env_name="Freeway",
    banner_extras={"Frameskip": FRAMESKIP},
)


def make_env(device):
    return FreewayEnv(num_envs=config.num_envs, frameskip=FRAMESKIP,
                      device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
