# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Breakout with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_breakout.py
"""

from __future__ import annotations

import warp as wp

wp.init()

from newton._src.pufferlib.envs.breakout import BreakoutEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

FRAMESKIP = 4

config = PPOConfig(
    num_envs=4096,
    horizon=64,
    obs_dim=BreakoutEnv.OBS_SIZE,
    num_actions=BreakoutEnv.NUM_ACTIONS,
    hidden=64,
    total_timesteps=30_000_000,
    lr=0.1,
    gamma=0.9721,
    gae_lambda=0.9487,
    clip_coef=0.6746,
    vf_coef=1.2196,
    vf_clip_coef=1.2292,
    ent_coef=0.00332,
    max_grad_norm=1.811,
    momentum=0.728,
    replay_ratio=1.424,
    minibatch_size=65536,
    rho_clip=2.102,
    c_clip=1.083,
    env_name="Breakout",
    banner_extras={"Frameskip": FRAMESKIP, "Obs dim": BreakoutEnv.OBS_SIZE},
)


def make_env(device):
    return BreakoutEnv(num_envs=config.num_envs, frameskip=FRAMESKIP,
                       device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
