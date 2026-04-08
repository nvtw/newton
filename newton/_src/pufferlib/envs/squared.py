# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Squared grid-navigation environment — Warp port of PufferLib ocean/squared.

The agent spawns in the center of an NxN grid and must navigate to a randomly
placed target.  Observation is the full NxN grid (flattened) with cell values:
0 = empty, 1 = agent, 2 = target.

Discrete action space: 0=noop, 1=down, 2=up, 3=left, 4=right.
Reward: +1 for reaching the target, -1 for going out of bounds or timeout.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

EMPTY = wp.constant(0)
AGENT = wp.constant(1)
TARGET = wp.constant(2)

ATN_NOOP = wp.constant(0)
ATN_DOWN = wp.constant(1)
ATN_UP = wp.constant(2)
ATN_LEFT = wp.constant(3)
ATN_RIGHT = wp.constant(4)


@wp.kernel
def squared_reset_kernel(
    grid: wp.array2d(dtype=int),
    agent_r: wp.array(dtype=int, ndim=1),
    agent_c: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    size: int,
    seed: int,
):
    i = wp.tid()
    tiles = size * size
    center = size / 2

    state = wp.rand_init(seed, i)

    for t in range(tiles):
        grid[i, t] = EMPTY

    grid[i, center * size + center] = AGENT
    agent_r[i] = center
    agent_c[i] = center
    tick[i] = 0
    episode_return[i] = 0.0

    target_idx = wp.randi(state) % tiles
    while target_idx == center * size + center:
        target_idx = wp.randi(state) % tiles
    grid[i, target_idx] = TARGET

    for t in range(tiles):
        obs[i, t] = float(grid[i, t])


@wp.kernel
def squared_step_kernel(
    grid: wp.array2d(dtype=int),
    agent_r: wp.array(dtype=int, ndim=1),
    agent_c: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    actions: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    size: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()
    tiles = size * size

    new_tick = tick[i] + 1
    atn = int(actions[i])

    r = agent_r[i]
    c = agent_c[i]

    # Clear old agent position
    grid[i, r * size + c] = EMPTY

    if atn == ATN_DOWN:
        r = r + 1
    elif atn == ATN_UP:
        r = r - 1
    elif atn == ATN_LEFT:
        c = c - 1
    elif atn == ATN_RIGHT:
        c = c + 1

    # Check out of bounds or timeout
    timeout = 3 * size
    if new_tick > timeout or r < 0 or c < 0 or r >= size or c >= size:
        reward = -1.0
        done = 1
    else:
        pos = r * size + c
        cell = grid[i, pos]
        if cell == TARGET:
            reward = 1.0
            done = 1
        else:
            reward = 0.0
            done = 0
            grid[i, pos] = AGENT
            agent_r[i] = r
            agent_c[i] = c

    new_ep_return = episode_return[i] + reward
    rewards[i] = reward
    dones[i] = float(done)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(new_tick)

    if done == 1:
        # Auto-reset
        state = wp.rand_init(seed_arr[0], i + new_tick)
        center = size / 2
        for t in range(tiles):
            grid[i, t] = EMPTY
        grid[i, center * size + center] = AGENT
        agent_r[i] = center
        agent_c[i] = center

        target_idx = wp.randi(state) % tiles
        while target_idx == center * size + center:
            target_idx = wp.randi(state) % tiles
        grid[i, target_idx] = TARGET

        tick[i] = 0
        episode_return[i] = 0.0
    else:
        tick[i] = new_tick
        episode_return[i] = new_ep_return

    for t in range(tiles):
        obs[i, t] = float(grid[i, t])


class SquaredEnv:
    """Vectorized Squared grid-navigation environment on GPU.

    Args:
        num_envs: Number of parallel environments.
        size: Grid side length (size x size).
        device: Warp device string.
        seed: Random seed.
    """

    NUM_ACTIONS = 5

    def __init__(
        self,
        num_envs: int = 4096,
        size: int = 11,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.size = size
        self.obs_size = size * size
        self.device = device
        self.seed = seed
        self._step_count = 0

        tiles = size * size
        self.grid = wp.zeros((num_envs, tiles), dtype=int, device=device)
        self.agent_r = wp.zeros(num_envs, dtype=int, device=device)
        self.agent_c = wp.zeros(num_envs, dtype=int, device=device)
        self.tick = wp.zeros(num_envs, dtype=int, device=device)
        self.episode_return = wp.zeros(num_envs, dtype=float, device=device)

        self.obs = wp.zeros((num_envs, self.obs_size), dtype=float, device=device)
        self.rewards = wp.zeros(num_envs, dtype=float, device=device)
        self.dones = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_returns = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_lengths = wp.zeros(num_envs, dtype=float, device=device)

        self.reset()

    def reset(self):
        wp.launch(
            squared_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.grid, self.agent_r, self.agent_c,
                self.tick, self.episode_return, self.obs,
                self.size, self.seed,
            ],
            device=self.device,
        )
        return self.obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        self._step_count += 1
        seed_arr = wp.array([self.seed + self._step_count], dtype=int, device=self.device)
        wp.launch(
            squared_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.grid, self.agent_r, self.agent_c,
                self.tick, self.episode_return,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.size, seed_arr,
            ],
            device=self.device,
        )
        return self.obs, self.rewards, self.dones

    def step_graphed(self, actions: wp.array, seed_arr: wp.array):
        """Graph-capture-compatible step (seed from device array)."""
        wp.launch(
            squared_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.grid, self.agent_r, self.agent_c,
                self.tick, self.episode_return,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.size, seed_arr,
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
