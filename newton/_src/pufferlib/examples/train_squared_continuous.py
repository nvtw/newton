# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Squared Continuous with PufferLib PPO on Warp.

Same grid world as Squared, but with 2 continuous action dimensions.

Usage::

    uv run python warp/pufferlib/examples/train_squared_continuous.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.squared_continuous import SquaredContinuousEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

GRID_SIZE = 11

config = PPOConfig(
    num_envs=64,
    horizon=256,
    obs_dim=GRID_SIZE * GRID_SIZE,
    num_actions=2,
    hidden=128,
    total_timesteps=5_000_000,
    lr=0.015,
    anneal_lr=False,
    gamma=0.8,
    gae_lambda=0.90,
    clip_coef=0.2,
    vf_coef=2.0,
    vf_clip_coef=0.2,
    ent_coef=0.01,
    max_grad_norm=1.5,
    momentum=0.95,
    replay_ratio=1.0,
    minibatch_size=4096,
    rho_clip=1.0,
    c_clip=1.0,
    continuous=True,
    env_name="Squared Continuous",
    best_return_init=-10.0,
    banner_extras={"Grid size": f"{GRID_SIZE}x{GRID_SIZE}"},
    return_format="7.2f",
    step_width=10,
    log_condition_fn=lambda it, _: it <= 20 or it % 10 == 0,
)


def make_env(device):
    return SquaredContinuousEnv(num_envs=config.num_envs, size=GRID_SIZE,
                                device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
