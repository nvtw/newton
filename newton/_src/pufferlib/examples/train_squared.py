# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Squared with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_squared.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.squared import SquaredEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

GRID_SIZE = 11

config = PPOConfig(
    num_envs=4096,
    horizon=64,
    obs_dim=GRID_SIZE * GRID_SIZE,
    num_actions=5,
    hidden=128,
    total_timesteps=100_000_000,
    lr=0.005,
    gamma=0.99,
    gae_lambda=0.90,
    clip_coef=0.2,
    vf_coef=2.0,
    vf_clip_coef=0.2,
    ent_coef=0.01,
    max_grad_norm=1.5,
    momentum=0.95,
    replay_ratio=1.0,
    minibatch_size=32768,
    rho_clip=1.0,
    c_clip=1.0,
    env_name="Squared",
    best_return_init=-10.0,
    banner_extras={"Grid size": f"{GRID_SIZE}x{GRID_SIZE}"},
    return_format="7.2f",
    step_width=10,
    log_condition_fn=lambda it, _: it <= 20 or it % 10 == 0,
)


def make_env(device):
    return SquaredEnv(num_envs=config.num_envs, size=GRID_SIZE,
                      device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
