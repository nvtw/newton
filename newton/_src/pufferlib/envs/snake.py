# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snake environment — Warp port of PufferLib ocean/snake.

Single-agent snake on a grid with walls, food, and corpses.
Observation: (2*vision+1)^2 = 121 normalized floats (local grid view around head).
Discrete action space: 0=up, 1=down, 2=left, 3=right.
Reward: +0.1 food, +0.1 corpse, -1.0 death (wall/body collision).
Episode auto-resets on death.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

EMPTY = wp.constant(0)
FOOD = wp.constant(1)
CORPSE = wp.constant(2)
WALL = wp.constant(3)
SNAKE_BODY = wp.constant(4)


@wp.kernel
def snake_reset_kernel(
    grid: wp.array(dtype=int, ndim=1),
    snake_body: wp.array(dtype=int, ndim=1),
    snake_length: wp.array(dtype=int, ndim=1),
    snake_ptr: wp.array(dtype=int, ndim=1),
    snake_lifetime: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    grid_w: int,
    grid_h: int,
    vision: int,
    max_snake_length: int,
    num_food: int,
    seed: int,
):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    grid_size = grid_w * grid_h
    grid_base = i * grid_size
    body_base = i * max_snake_length * 2

    # Clear grid
    for idx in range(grid_size):
        grid[grid_base + idx] = EMPTY

    # Place walls (vision-thick border)
    for r in range(grid_h):
        for c in range(grid_w):
            if r < vision or r >= grid_h - vision or c < vision or c >= grid_w - vision:
                grid[grid_base + r * grid_w + c] = WALL

    # Clear snake body buffer
    for idx in range(max_snake_length * 2):
        snake_body[body_base + idx] = -1

    # Spawn snake at random empty cell (10000 attempts to match C++ unbounded loop)
    head_r = int(0)
    head_c = int(0)
    found = int(0)
    for _attempt in range(10000):
        if found == 0:
            head_r = wp.randi(state) % grid_h
            head_c = wp.randi(state) % grid_w
            tile = grid[grid_base + head_r * grid_w + head_c]
            if tile == EMPTY or tile == CORPSE:
                found = 1

    if found == 1:
        grid[grid_base + head_r * grid_w + head_c] = SNAKE_BODY
    snake_body[body_base] = head_r
    snake_body[body_base + 1] = head_c
    snake_length[i] = 1
    snake_ptr[i] = 0
    snake_lifetime[i] = 0
    tick[i] = 0
    episode_return[i] = 0.0

    # Spawn food
    for _f in range(num_food):
        fr = int(0)
        fc = int(0)
        food_found = int(0)
        for _attempt2 in range(10000):
            if food_found == 0:
                fr = wp.randi(state) % grid_h
                fc = wp.randi(state) % grid_w
                ft = grid[grid_base + fr * grid_w + fc]
                if ft == EMPTY or ft == CORPSE:
                    food_found = 1
        if food_found == 1:
            grid[grid_base + fr * grid_w + fc] = FOOD

    # Compute initial observations
    window = 2 * vision + 1
    obs_size = window * window
    r_off = head_r - vision
    c_off = head_c - vision
    for r in range(window):
        for c in range(window):
            obs[i, r * window + c] = float(grid[grid_base + (r_off + r) * grid_w + c_off + c]) / 3.0

    # Zero out remaining obs slots (shouldn't exist if sized correctly, but safety)
    for idx in range(obs_size, obs_size):
        obs[i, idx] = 0.0


@wp.kernel
def snake_step_kernel(
    grid: wp.array(dtype=int, ndim=1),
    snake_body: wp.array(dtype=int, ndim=1),
    snake_length: wp.array(dtype=int, ndim=1),
    snake_ptr: wp.array(dtype=int, ndim=1),
    snake_lifetime: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    actions: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    grid_w: int,
    grid_h: int,
    vision: int,
    max_snake_length: int,
    num_food: int,
    reward_food: float,
    reward_corpse: float,
    reward_death: float,
    leave_corpse: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()
    grid_size = grid_w * grid_h
    grid_base = i * grid_size
    body_base = i * max_snake_length * 2
    window = 2 * vision + 1

    new_tick = tick[i] + 1
    reward = float(0.0)
    done = int(0)

    atn = int(actions[i])
    dr = int(0)
    dc = int(0)
    if atn == 0:
        dr = -1
    elif atn == 1:
        dr = 1
    elif atn == 2:
        dc = -1
    elif atn == 3:
        dc = 1

    head_p = snake_ptr[i]
    head_off = body_base + 2 * head_p
    cur_r = snake_body[head_off]
    cur_c = snake_body[head_off + 1]

    next_r = cur_r + dr
    next_c = cur_c + dc

    # Disallow reversing into own neck (length >= 2)
    s_len = snake_length[i]
    if s_len >= 2:
        prev_p = head_p - 1
        if prev_p < 0:
            prev_p = max_snake_length - 1
        prev_off = body_base + 2 * prev_p
        prev_r = snake_body[prev_off]
        prev_c = snake_body[prev_off + 1]
        if prev_r == next_r and prev_c == next_c:
            next_r = cur_r - dr
            next_c = cur_c - dc

    tile = grid[grid_base + next_r * grid_w + next_c]

    # Death check: wall or snake body
    died = int(0)
    if tile >= WALL:
        died = 1

    state = wp.rand_init(seed_arr[0], i + new_tick)

    if died == 1:
        reward = reward_death
        done = 1

        # Delete snake, optionally leaving corpse
        del_ptr = head_p
        del_remaining = s_len
        for _d in range(max_snake_length):
            if del_remaining > 0:
                d_off = body_base + 2 * del_ptr
                d_r = snake_body[d_off]
                d_c = snake_body[d_off + 1]
                if leave_corpse == 1 and del_remaining % 2 == 0:
                    grid[grid_base + d_r * grid_w + d_c] = CORPSE
                else:
                    grid[grid_base + d_r * grid_w + d_c] = EMPTY
                snake_body[d_off] = -1
                snake_body[d_off + 1] = -1
                del_ptr = del_ptr - 1
                if del_ptr < 0:
                    del_ptr = max_snake_length - 1
                del_remaining = del_remaining - 1

        # Respawn snake
        sp_r = int(0)
        sp_c = int(0)
        sp_found = int(0)
        for _sp in range(10000):
            if sp_found == 0:
                sp_r = wp.randi(state) % grid_h
                sp_c = wp.randi(state) % grid_w
                sp_tile = grid[grid_base + sp_r * grid_w + sp_c]
                if sp_tile == EMPTY or sp_tile == CORPSE:
                    sp_found = 1

        if sp_found == 1:
            grid[grid_base + sp_r * grid_w + sp_c] = SNAKE_BODY
        snake_body[body_base] = sp_r
        snake_body[body_base + 1] = sp_c
        snake_length[i] = 1
        snake_ptr[i] = 0
        snake_lifetime[i] = 0

        # Write obs for respawned snake
        r_off = sp_r - vision
        c_off = sp_c - vision
        for r in range(window):
            for c in range(window):
                obs[i, r * window + c] = float(grid[grid_base + (r_off + r) * grid_w + c_off + c]) / 3.0

    else:
        # Advance head pointer (circular buffer)
        new_head_p = head_p + 1
        if new_head_p >= max_snake_length:
            new_head_p = 0
        new_head_off = body_base + 2 * new_head_p
        snake_body[new_head_off] = next_r
        snake_body[new_head_off + 1] = next_c
        snake_ptr[i] = new_head_p

        grow = int(0)
        if tile == FOOD:
            reward = reward_food
            grow = 1
            # Respawn food at random empty/corpse cell
            fr = int(0)
            fc = int(0)
            ff = int(0)
            for _fa in range(10000):
                if ff == 0:
                    fr = wp.randi(state) % grid_h
                    fc = wp.randi(state) % grid_w
                    ft = grid[grid_base + fr * grid_w + fc]
                    if ft == EMPTY or ft == CORPSE:
                        ff = 1
            if ff == 1:
                grid[grid_base + fr * grid_w + fc] = FOOD
        elif tile == CORPSE:
            reward = reward_corpse
            grow = 1

        if grow == 1 and s_len < max_snake_length - 1:
            snake_length[i] = s_len + 1
        else:
            # Remove tail
            tail_p = new_head_p - s_len
            if tail_p < 0:
                tail_p = tail_p + max_snake_length
            tail_off = body_base + 2 * tail_p
            tail_r = snake_body[tail_off]
            tail_c = snake_body[tail_off + 1]
            grid[grid_base + tail_r * grid_w + tail_c] = EMPTY
            snake_body[tail_off] = -1
            snake_body[tail_off + 1] = -1

        # Place head on grid
        grid[grid_base + next_r * grid_w + next_c] = SNAKE_BODY
        snake_lifetime[i] = snake_lifetime[i] + 1

        # Write obs
        r_off = next_r - vision
        c_off = next_c - vision
        for r in range(window):
            for c in range(window):
                obs[i, r * window + c] = float(grid[grid_base + (r_off + r) * grid_w + c_off + c]) / 3.0

    new_ep_return = episode_return[i] + reward
    rewards[i] = reward
    dones[i] = float(done)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(new_tick)

    if done == 1:
        tick[i] = 0
        episode_return[i] = 0.0
    else:
        tick[i] = new_tick
        episode_return[i] = new_ep_return


class SnakeEnv:
    """Vectorized Snake environment on GPU.

    Single-agent snake per parallel env instance. The snake moves on a
    grid with vision-thick walls around the border, eats food/corpses to
    grow, and dies on collision with walls or its own body.

    Args:
        num_envs: Number of parallel environments.
        grid_w: Grid width (including wall border).
        grid_h: Grid height (including wall border).
        vision: Observation radius around the snake head.
        max_snake_length: Maximum snake body length (circular buffer size).
        num_food: Number of food items on the grid.
        reward_food: Reward for eating food.
        reward_corpse: Reward for eating a corpse segment.
        reward_death: Reward (penalty) for dying.
        leave_corpse: Whether dead snakes leave corpse segments.
        device: Warp device string.
        seed: Random seed.
    """

    OBS_SIZE = 121  # (2*5+1)^2
    NUM_ACTIONS = 4

    def __init__(
        self,
        num_envs: int = 1024,
        grid_w: int = 64,
        grid_h: int = 64,
        vision: int = 5,
        max_snake_length: int = 256,
        num_food: int = 32,
        reward_food: float = 0.1,
        reward_corpse: float = 0.1,
        reward_death: float = -1.0,
        leave_corpse: bool = True,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.vision = vision
        self.max_snake_length = max_snake_length
        self.num_food = num_food
        self.reward_food = reward_food
        self.reward_corpse = reward_corpse
        self.reward_death = reward_death
        self.leave_corpse_int = 1 if leave_corpse else 0
        self.device = device
        self.seed = seed
        self._step_count = 0

        grid_size = grid_w * grid_h
        body_size = max_snake_length * 2

        self.grid = wp.zeros(num_envs * grid_size, dtype=int, device=device)
        self.snake_body = wp.zeros(num_envs * body_size, dtype=int, device=device)
        self.snake_length = wp.zeros(num_envs, dtype=int, device=device)
        self.snake_ptr = wp.zeros(num_envs, dtype=int, device=device)
        self.snake_lifetime = wp.zeros(num_envs, dtype=int, device=device)
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
            snake_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.grid, self.snake_body,
                self.snake_length, self.snake_ptr, self.snake_lifetime,
                self.tick, self.episode_return, self.obs,
                self.grid_w, self.grid_h,
                self.vision, self.max_snake_length,
                self.num_food, self.seed,
            ],
            device=self.device,
        )
        return self.obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        self._step_count += 1
        seed_arr = wp.array([self.seed + self._step_count], dtype=int, device=self.device)
        self._step_impl(actions, seed_arr)
        return self.obs, self.rewards, self.dones

    def step_graphed(self, actions: wp.array, seed_arr: wp.array):
        """Graph-capture-compatible step (seed from device array)."""
        self._step_impl(actions, seed_arr)

    def _step_impl(self, actions, seed_arr):
        wp.launch(
            snake_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.grid, self.snake_body,
                self.snake_length, self.snake_ptr, self.snake_lifetime,
                self.tick, self.episode_return,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.grid_w, self.grid_h,
                self.vision, self.max_snake_length,
                self.num_food,
                self.reward_food, self.reward_corpse, self.reward_death,
                self.leave_corpse_int, seed_arr,
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
