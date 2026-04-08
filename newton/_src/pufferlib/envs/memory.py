# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory environment — Warp port of PufferLib ocean/memory.

Tests the agent's ability to remember an initial observation.  At tick 0 the
agent sees a goal signal (-1 or +1).  For the remaining ``length - 1`` ticks
the observation is 0.  At the final tick the agent must output action 0 if the
goal was -1, or action 1 if the goal was +1.  Reward is 1 on correct recall,
0 otherwise.

Observation: 1 float.  Action: discrete {0, 1}.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def memory_reset_kernel(
    goal: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    obs: wp.array2d(dtype=float),
    seed: int,
):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    if wp.randi(state) % 2 == 0:
        goal[i] = -1
    else:
        goal[i] = 1
    obs[i, 0] = float(goal[i])
    tick[i] = 0


@wp.kernel
def memory_step_kernel(
    goal: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    actions: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    length: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()
    new_tick = tick[i] + 1
    obs[i, 0] = 0.0

    if new_tick < length:
        rewards[i] = 0.0
        dones[i] = 0.0
        episode_returns[i] = episode_return[i]
        episode_lengths[i] = float(new_tick)
        tick[i] = new_tick
        episode_return[i] = episode_return[i]
    else:
        # Final tick — check recall
        atn = int(actions[i])
        g = goal[i]
        val = 0.0
        if atn == 0 and g == -1:
            val = 1.0
        if atn == 1 and g == 1:
            val = 1.0

        rewards[i] = val
        dones[i] = 1.0
        new_ep_return = episode_return[i] + val
        episode_returns[i] = new_ep_return
        episode_lengths[i] = float(new_tick)

        # Auto-reset
        state = wp.rand_init(seed_arr[0], i + new_tick)
        if wp.randi(state) % 2 == 0:
            goal[i] = -1
        else:
            goal[i] = 1
        obs[i, 0] = float(goal[i])
        tick[i] = 0
        episode_return[i] = 0.0


class MemoryEnv:
    """Vectorized Memory environment on GPU.

    Args:
        num_envs: Number of parallel environments.
        length: Episode length (agent sees goal at tick 0, acts at tick ``length``).
        device: Warp device string.
        seed: Random seed.
    """

    OBS_SIZE = 1
    NUM_ACTIONS = 2

    def __init__(
        self,
        num_envs: int = 1024,
        length: int = 16,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.length = length
        self.device = device
        self.seed = seed
        self._step_count = 0

        self.goal = wp.zeros(num_envs, dtype=int, device=device)
        self.tick = wp.zeros(num_envs, dtype=int, device=device)
        self.episode_return = wp.zeros(num_envs, dtype=float, device=device)

        self.obs = wp.zeros((num_envs, self.OBS_SIZE), dtype=float, device=device)
        self.rewards = wp.zeros(num_envs, dtype=float, device=device)
        self.dones = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_returns = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_lengths = wp.zeros(num_envs, dtype=float, device=device)

        self.reset()

    def reset(self):
        wp.launch(
            memory_reset_kernel,
            dim=self.num_envs,
            inputs=[self.goal, self.tick, self.obs, self.seed],
            device=self.device,
        )
        return self.obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        self._step_count += 1
        seed_arr = wp.array([self.seed + self._step_count], dtype=int, device=self.device)
        wp.launch(
            memory_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.goal, self.tick, self.episode_return,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.length, seed_arr,
            ],
            device=self.device,
        )
        return self.obs, self.rewards, self.dones

    def step_graphed(self, actions: wp.array, seed_arr: wp.array):
        """Graph-capture-compatible step (seed from device array)."""
        wp.launch(
            memory_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.goal, self.tick, self.episode_return,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.length, seed_arr,
            ],
            device=self.device,
        )

    def get_episode_stats(self) -> dict:
        returns_np = self.episode_returns.numpy()
        lengths_np = self.episode_lengths.numpy()
        dones_np = self.dones.numpy()
        done_mask = dones_np > 0.5
        if np.any(done_mask):
            return {
                "mean_return": float(np.mean(returns_np[done_mask])),
                "mean_length": float(np.mean(lengths_np[done_mask])),
                "num_episodes": int(np.sum(done_mask)),
            }
        return {"mean_return": 0.0, "mean_length": 0.0, "num_episodes": 0}
