# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Memory with PufferLib PPO on Warp.

Usage::

    uv run python warp/pufferlib/examples/train_memory.py
"""

from __future__ import annotations

import numpy as np

import warp as wp

wp.init()

from newton._src.pufferlib.envs.memory import MemoryEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

MEMORY_LENGTH = 16


def _format_return(avg_ret, best_return, loss_np, sps):
    accuracy = avg_ret * 100.0
    return (
        f"Step {int(sps * 0 + 0):>10,} | "  # placeholder, overridden below
        f"Return {avg_ret:6.3f} | "
        f"Acc {accuracy:5.1f}% | "
        f"Best {best_return:6.3f} | "
        f"PG {loss_np[1]:8.4f} | "
        f"Ent {loss_np[3]:7.4f} | "
        f"SPS {sps:,.0f}"
    )


def _format_summary(total_steps, best_return, avg_ret, elapsed):
    lines = [
        "",
        "=" * 60,
        f"Training complete in {elapsed:.1f}s",
        f"  Total steps:  {total_steps:,}",
        f"  Best return:  {best_return:.3f} ({best_return * 100:.1f}% accuracy)",
        f"  Final return: {avg_ret:.3f} ({avg_ret * 100:.1f}% accuracy)",
        f"  Throughput:   {total_steps / elapsed:,.0f} steps/sec",
        f"  NOTE: 50% = random, >50% = learning signal present",
        "=" * 60,
    ]
    return "\n".join(lines)


config = PPOConfig(
    num_envs=1024,
    horizon=64,
    obs_dim=1,
    num_actions=2,
    hidden=128,
    total_timesteps=50_000_000,
    lr=0.015,
    gamma=0.995,
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
    env_name="Memory",
    best_return_init=-1.0,
    banner_extras={"Memory length": MEMORY_LENGTH,
                   "NOTE": "Feed-forward net — random baseline is ~50% accuracy"},
    step_width=10,
    log_interval=20,
    format_summary_fn=_format_summary,
)


# Memory needs a custom log line (accuracy %) — use format_return_fn
# but we also need the step count which the callback doesn't receive.
# Instead, override the full line via format_return_fn and let the trainer
# inject the step count.  We'll use a simpler approach: just override
# format_return_fn to None and use a custom log_condition + the default
# format with a custom return_format.
# Actually, the cleanest way: use format_return_fn that the trainer calls
# with (avg_ret, best_return, loss_np, sps).  The step count is not passed,
# but we can compute it from sps * elapsed... that's fragile.
# Better: just accept the default format for now and override format_summary.
# The Memory-specific accuracy line is nice-to-have but not critical.
# Let's keep it simple: use default logging with 6.3f return format.

config.return_format = "6.3f"


def make_env(device):
    return MemoryEnv(num_envs=config.num_envs, length=MEMORY_LENGTH,
                     device=device, seed=config.seed)


if __name__ == "__main__":
    PPOTrainer(config, make_env).train()
