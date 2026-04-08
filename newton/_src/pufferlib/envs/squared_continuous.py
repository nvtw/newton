# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Squared Continuous grid-navigation environment — Warp port of PufferLib ocean/squared_continuous.

Same grid world as Squared, but with 2 continuous action dimensions instead of
5 discrete actions.  Actions are clamped to [-1, 1] and thresholded at |0.25|
to determine movement direction.

- ``action[0]``: vertical (positive = down, negative = up)
- ``action[1]``: horizontal (positive = right, negative = left)

Observation is the full NxN grid (flattened) with cell values:
0 = empty, 1 = agent, 2 = target.

Reward: +1 for reaching the target, -1 for going out of bounds or timeout.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

EMPTY = wp.constant(0)
AGENT = wp.constant(1)
TARGET = wp.constant(2)


@wp.kernel
def squared_cont_reset_kernel(
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
def squared_cont_step_kernel(
    grid: wp.array2d(dtype=int),
    agent_r: wp.array(dtype=int, ndim=1),
    agent_c: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    actions: wp.array2d(dtype=float),
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

    vert = wp.clamp(actions[i, 0], -1.0, 1.0)
    horiz = wp.clamp(actions[i, 1], -1.0, 1.0)

    r = agent_r[i]
    c = agent_c[i]

    grid[i, r * size + c] = EMPTY

    if vert > 0.25:
        r = r + 1
    elif vert < -0.25:
        r = r - 1

    if horiz > 0.25:
        c = c + 1
    elif horiz < -0.25:
        c = c - 1

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


class SquaredContinuousEnv:
    """Vectorized Squared Continuous grid-navigation environment on GPU.

    Uses 2 continuous action dimensions instead of 5 discrete actions.

    Args:
        num_envs: Number of parallel environments.
        size: Grid side length (size x size).
        device: Warp device string.
        seed: Random seed.
    """

    NUM_ACTIONS = 2
    OBS_SIZE = 121

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

        # For continuous actions, the env receives (N, 2) actions directly
        self._actions_2d = wp.zeros((num_envs, 2), dtype=float, device=device)

        self.reset()

    def reset(self):
        wp.launch(
            squared_cont_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.grid, self.agent_r, self.agent_c,
                self.tick, self.episode_return, self.obs,
                self.size, self.seed,
            ],
            device=self.device,
        )
        return self.obs

    def step_graphed(self, actions_2d: wp.array, seed_arr: wp.array):
        """Graph-capture-compatible step.

        Args:
            actions_2d: (N, 2) continuous actions.
            seed_arr: Device-side 1-element seed array.
        """
        wp.launch(
            squared_cont_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.grid, self.agent_r, self.agent_c,
                self.tick, self.episode_return,
                actions_2d, self.obs, self.rewards, self.dones,
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
