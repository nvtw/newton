# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeway environment — Warp port of PufferLib ocean/freeway.

Agent controls a chicken crossing 10 lanes of traffic.
Observation: 34 normalized floats (player state + enemy positions).
Discrete action space: 0=noop, 1=up, 2=down.
Dense reward: +1/NUM_LANES per new lane crossed, -0.01 per hit.
Episode ends when time runs out (136 seconds at 60 fps).
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LANES = 10
MAX_ENEMIES_PER_LANE = 3
NUM_LEVELS = 8
NUM_ENEMY_SLOTS = NUM_LANES * MAX_ENEMIES_PER_LANE  # 30

TICK_RATE = 1.0 / 60.0
GAME_LENGTH = 136.0
RANDOMIZE_SPEED_FREQ = 360
TICKS_STUNT = 40
PENALTY_HIT = -0.01
BASE_PLAYER_SPEED = 1.0 / 3.5
BASE_ROAD_SPEED = 1.0 / 13.0
MULT_ROAD_SPEED = 1.35

OBS_SIZE = 4 + NUM_ENEMY_SLOTS  # 34

# Pre-compute speed values on the host then upload as wp.array
_SPEED_VALUES = [BASE_ROAD_SPEED * MULT_ROAD_SPEED**k for k in range(6)]

_HUMAN_HIGH_SCORE = [25, 20, 18, 20, 25, 18, 15, 18]

# fmt: off
_ENEMIES_PER_LANE = [
    1,1,1,1,1,1,1,1,1,1,  # level 0
    1,2,2,3,2,1,2,3,2,2,  # level 1
    3,3,1,3,1,1,3,1,3,1,  # level 2
    1,1,1,1,1,1,1,1,1,1,  # level 3
    1,1,1,1,1,1,1,1,1,1,  # level 4
    1,2,2,3,2,1,2,3,2,2,  # level 5
    3,3,1,3,1,1,3,1,3,1,  # level 6
    1,1,1,1,1,1,1,1,1,1,  # level 7
]

_ENEMIES_TYPES = [
    0,0,0,0,0,0,0,0,0,0,  # level 0
    0,0,0,0,0,1,0,0,0,0,  # level 1
    0,0,0,0,1,0,0,0,0,0,  # level 2
    1,1,1,1,1,1,1,1,1,1,  # level 3
    0,0,0,0,0,0,0,0,0,0,  # level 4
    0,0,0,0,0,1,0,0,0,0,  # level 5
    0,0,0,0,1,0,0,0,0,0,  # level 6
    1,1,1,1,1,1,1,1,1,1,  # level 7
]

_SPEED_RANDOMIZATION = [0,0,0,0,1,1,1,1]

# Flat [level * NUM_LANES * MAX_ENEMIES_PER_LANE + lane * MAX_ENEMIES_PER_LANE + slot]
_ENEMIES_INITIAL_X = [
    # level 0
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
    # level 1
    0.0,0.0,0.0, 0.0,0.1,0.0, 0.0,0.2,0.0, 0.0,0.1,0.2, 0.0,0.4,0.0,
    0.0,0.0,0.0, 0.0,0.4,0.0, 0.0,0.1,0.2, 0.0,0.2,0.0, 0.0,0.1,0.0,
    # level 2
    0.0,0.2,0.4, 0.0,0.2,0.4, 0.0,0.0,0.0, 0.0,0.2,0.4, 0.0,0.0,0.0,
    0.0,0.0,0.0, 0.0,0.2,0.4, 0.0,0.0,0.0, 0.0,0.2,0.4, 0.0,0.2,0.4,
    # level 3
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
    # level 4
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
    # level 5
    0.0,0.0,0.0, 0.0,0.1,0.0, 0.0,0.2,0.0, 0.0,0.1,0.2, 0.0,0.4,0.0,
    0.0,0.0,0.0, 0.0,0.4,0.0, 0.0,0.1,0.2, 0.0,0.2,0.0, 0.0,0.1,0.0,
    # level 6
    0.0,0.2,0.4, 0.0,0.2,0.4, 0.0,0.0,0.0, 0.0,0.2,0.4, 0.0,0.0,0.0,
    0.0,0.0,0.0, 0.0,0.2,0.4, 0.0,0.0,0.0, 0.0,0.2,0.4, 0.0,0.2,0.4,
    # level 7
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
    0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
]

# Flat [level * NUM_LANES + lane]
_ENEMIES_INITIAL_SPEED_IDX = [
    0,1,2,3,4,4,3,2,1,0,  # level 0
    0,1,3,4,5,5,4,3,1,0,  # level 1
    0,1,3,4,5,5,4,3,1,0,  # level 2
    5,4,2,4,5,5,4,2,4,5,  # level 3
    0,1,2,3,4,4,3,2,1,0,  # level 4
    0,1,3,4,5,5,4,3,1,0,  # level 5
    0,1,3,4,5,5,4,3,1,0,  # level 6
    5,4,2,4,5,5,4,2,4,5,  # level 7
]
# fmt: on


# ---------------------------------------------------------------------------
# Reset kernel
# ---------------------------------------------------------------------------
@wp.kernel
def freeway_reset_kernel(
    # Per-env player state
    player_y: wp.array(dtype=float, ndim=1),
    best_lane_idx: wp.array(dtype=int, ndim=1),
    ticks_stunts_left: wp.array(dtype=int, ndim=1),
    score: wp.array(dtype=int, ndim=1),
    ep_return: wp.array(dtype=float, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    level: wp.array(dtype=int, ndim=1),
    # Enemy flat arrays (num_envs * NUM_ENEMY_SLOTS)
    enemy_x: wp.array(dtype=float, ndim=1),
    enemy_vx: wp.array(dtype=float, ndim=1),
    enemy_enabled: wp.array(dtype=int, ndim=1),
    enemy_speed_idx: wp.array(dtype=int, ndim=1),
    # Level data tables (read-only)
    tbl_enemies_per_lane: wp.array(dtype=int, ndim=1),
    tbl_enemies_types: wp.array(dtype=int, ndim=1),
    tbl_enemies_initial_x: wp.array(dtype=float, ndim=1),
    tbl_enemies_initial_speed_idx: wp.array(dtype=int, ndim=1),
    tbl_speed_randomization: wp.array(dtype=int, ndim=1),
    tbl_speed_values: wp.array(dtype=float, ndim=1),
    # Observations
    obs: wp.array2d(dtype=float),
    # Scalars
    width: float,
    height: float,
    player_height: float,
    car_width: int,
    car_height: int,
    lane_size: int,
    env_randomization: int,
    requested_level: int,
    seed: int,
):
    i = wp.tid()
    state = wp.rand_init(seed, i)

    # Pick level
    lvl = int(0)
    if requested_level < 0 or requested_level >= 8:
        lvl = wp.randi(state) % 8
        if lvl < 0:
            lvl = -lvl
    else:
        lvl = requested_level
    level[i] = lvl

    road_start = height / 2.0 + float(10 * lane_size) / 2.0
    road_end = road_start - float(10 * lane_size)

    # Reset player to bottom
    py = height - float(player_height) / 2.0
    player_y[i] = py
    best_lane_idx[i] = 0
    ticks_stunts_left[i] = 0
    score[i] = 0
    ep_return[i] = 0.0
    tick[i] = 0

    truck_width = 2 * car_width
    f_width = float(width)

    # Spawn enemies
    for lane in range(10):
        lane_offset_x = f_width * wp.randf(state)
        for slot in range(3):
            flat_idx = i * 30 + lane * 3 + slot
            enabled = int(0)
            if slot < tbl_enemies_per_lane[lvl * 10 + lane]:
                enabled = 1
            enemy_enabled[flat_idx] = enabled

            init_spd_idx = tbl_enemies_initial_speed_idx[lvl * 10 + lane]
            enemy_speed_idx[flat_idx] = init_spd_idx

            etype = tbl_enemies_types[lvl * 10 + lane]
            e_width = int(0)
            if etype == 0:
                e_width = car_width
            else:
                e_width = truck_width

            init_x_frac = tbl_enemies_initial_x[lvl * 30 + lane * 3 + slot]
            ex = init_x_frac * f_width
            if lane >= 5:
                ex = f_width - ex

            if env_randomization == 1 and enabled == 1:
                ex = ex + lane_offset_x

            enemy_x[flat_idx] = ex

            spd = tbl_speed_values[init_spd_idx] * float(1) / 60.0 * f_width
            if lane >= 5:
                spd = -spd
            enemy_vx[flat_idx] = spd

    # Write observations
    obs[i, 0] = py / height
    obs[i, 1] = 0.0
    obs[i, 2] = 0.0
    obs[i, 3] = 0.0
    for lane in range(10):
        for slot in range(3):
            flat_idx = i * 30 + lane * 3 + slot
            obs_idx = 4 + lane * 3 + slot
            if enemy_enabled[flat_idx] == 1:
                etype = tbl_enemies_types[lvl * 10 + lane]
                e_height = int(0)
                if etype == 0:
                    e_height = car_height
                else:
                    e_height = car_height
                offset = float(0.0)
                if lane < 5:
                    offset = float(e_height) / (2.0 * f_width)
                else:
                    offset = -float(e_height) / (2.0 * f_width)
                obs[i, obs_idx] = enemy_x[flat_idx] / f_width + offset
            else:
                obs[i, obs_idx] = 0.0


# ---------------------------------------------------------------------------
# Step kernel
# ---------------------------------------------------------------------------
@wp.kernel
def freeway_step_kernel(
    # Per-env player state
    player_y: wp.array(dtype=float, ndim=1),
    best_lane_idx: wp.array(dtype=int, ndim=1),
    ticks_stunts_left: wp.array(dtype=int, ndim=1),
    score: wp.array(dtype=int, ndim=1),
    ep_return: wp.array(dtype=float, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    level: wp.array(dtype=int, ndim=1),
    # Enemy flat arrays
    enemy_x: wp.array(dtype=float, ndim=1),
    enemy_vx: wp.array(dtype=float, ndim=1),
    enemy_enabled: wp.array(dtype=int, ndim=1),
    enemy_speed_idx: wp.array(dtype=int, ndim=1),
    # Level data tables
    tbl_enemies_per_lane: wp.array(dtype=int, ndim=1),
    tbl_enemies_types: wp.array(dtype=int, ndim=1),
    tbl_enemies_initial_x: wp.array(dtype=float, ndim=1),
    tbl_enemies_initial_speed_idx: wp.array(dtype=int, ndim=1),
    tbl_speed_randomization: wp.array(dtype=int, ndim=1),
    tbl_speed_values: wp.array(dtype=float, ndim=1),
    tbl_human_high_score: wp.array(dtype=int, ndim=1),
    # Actions / outputs
    actions: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    # Scalars
    width: float,
    height: float,
    player_width: float,
    player_height: float,
    car_width: int,
    car_height: int,
    lane_size: int,
    difficulty: int,
    use_dense_rewards: int,
    env_randomization: int,
    frameskip: int,
    requested_level: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()

    lvl = level[i]
    f_width = float(width)
    f_height = float(height)
    f_lane_size = float(lane_size)
    f_player_height = float(player_height)
    f_player_width = float(player_width)
    truck_width = 2 * car_width

    road_start = f_height / 2.0 + float(10 * lane_size) / 2.0
    road_end = road_start - float(10 * lane_size)

    new_tick = tick[i]
    py = player_y[i]
    best_li = best_lane_idx[i]
    stunts_left = ticks_stunts_left[i]
    sc = score[i]
    ep_ret = ep_return[i]
    reward = float(0.0)
    done = int(0)

    atn = int(actions[i])

    state = wp.rand_init(seed_arr[0], i + new_tick)

    for _f in range(frameskip):
        if done == 0:
            new_tick = new_tick + 1

            # Player movement
            player_dy = float(0.0)
            if atn == 2:
                player_dy = -1.0
            elif atn == 1:
                player_dy = 1.0

            if stunts_left == 0:
                py = py - player_dy * (1.0 / 3.5) * f_height * (1.0 / 60.0)
            else:
                stunts_left = stunts_left - 1
                if difficulty == 0:
                    py = py + 1.5 * f_lane_size / float(40)

            # Clip player
            min_py = f_player_height / 2.0
            max_py = f_height - f_player_height / 2.0
            if py < min_py:
                py = min_py
            if py > max_py:
                py = max_py

            # Check collisions (only when not stunned)
            if stunts_left == 0:
                hit = int(0)
                for lane in range(10):
                    for slot in range(3):
                        if hit == 0:
                            flat_idx = i * 30 + lane * 3 + slot
                            if enemy_enabled[flat_idx] == 1:
                                etype = tbl_enemies_types[lvl * 10 + lane]
                                e_w = int(0)
                                if etype == 0:
                                    e_w = car_width
                                else:
                                    e_w = truck_width
                                e_h = car_height

                                ey = road_start + (road_end - road_start) * float(lane) / 10.0 - f_lane_size / 2.0

                                p_minx = f_width / 4.0 - f_player_width / 2.0
                                p_maxx = f_width / 4.0 + f_player_width / 2.0
                                p_miny = py - f_player_height / 2.0
                                p_maxy = py + f_player_height / 2.0

                                e_minx = enemy_x[flat_idx] - float(e_w) / 2.0
                                e_maxx = enemy_x[flat_idx] + float(e_w) / 2.0
                                e_miny = ey - float(e_h) / 2.0
                                e_maxy = ey + float(e_h) / 2.0

                                if p_minx < e_maxx and p_maxx > e_minx and p_miny < e_maxy and p_maxy > e_miny:
                                    hit = 1

                if hit == 1:
                    stunts_left = 40
                    if use_dense_rewards == 1:
                        reward = reward + (-0.01)
                    if difficulty == 1:
                        # C++ reset_player only resets position and stunt timer,
                        # NOT best_lane_idx
                        py = f_height - f_player_height / 2.0
                        stunts_left = 0

            # Check lane progress
            threshold = road_start - float(best_li + 1) * f_lane_size
            if py <= threshold:
                best_li = best_li + 1
                if use_dense_rewards == 1:
                    reward = reward + 1.0 / 10.0

            # Check if crossed all lanes
            if best_li >= 10:
                py = f_height - f_player_height / 2.0
                stunts_left = 0
                best_li = 0
                sc = sc + 1
                reward = reward + 1.0

            # Move enemies
            for lane in range(10):
                for slot in range(3):
                    flat_idx = i * 30 + lane * 3 + slot
                    if enemy_enabled[flat_idx] == 1:
                        ex = enemy_x[flat_idx] + enemy_vx[flat_idx]
                        etype = tbl_enemies_types[lvl * 10 + lane]
                        e_w = int(0)
                        if etype == 0:
                            e_w = car_width
                        else:
                            e_w = truck_width
                        half_ew = float(e_w) / 2.0
                        if ex > f_width + half_ew:
                            ex = ex - f_width
                        elif ex < -half_ew:
                            ex = ex + f_width
                        enemy_x[flat_idx] = ex

            # Check time
            if float(new_tick) * (1.0 / 60.0) >= 136.0:
                done = 1

            # Speed randomization
            if new_tick % 360 == 0:
                if tbl_speed_randomization[lvl] == 1:
                    for lane in range(10):
                        delta_spd = (wp.randi(state) % 3) - 1
                        if delta_spd < -1:
                            delta_spd = delta_spd + 3
                        init_spd = tbl_enemies_initial_speed_idx[lvl * 10 + lane]
                        for slot in range(3):
                            flat_idx = i * 30 + lane * 3 + slot
                            cur_spd = enemy_speed_idx[flat_idx]
                            # C++ order: first clamp current to [init-2, init+2],
                            # then add delta and clamp to [0, 5]
                            clamped = cur_spd
                            lo1 = init_spd - 2
                            hi1 = init_spd + 2
                            if clamped < lo1:
                                clamped = lo1
                            if clamped > hi1:
                                clamped = hi1
                            new_spd = clamped + delta_spd
                            if new_spd < 0:
                                new_spd = 0
                            if new_spd > 5:
                                new_spd = 5
                            enemy_speed_idx[flat_idx] = new_spd
                            spd = tbl_speed_values[new_spd] * (1.0 / 60.0) * f_width
                            if lane >= 5:
                                spd = -spd
                            enemy_vx[flat_idx] = spd

    # Store outputs
    ep_ret = ep_ret + reward
    new_ep_return = ep_ret
    rewards[i] = reward
    dones[i] = float(done)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(new_tick)

    # Auto-reset on done
    if done == 1:
        reset_state = wp.rand_init(seed_arr[0], i + new_tick + 7777)

        new_lvl = int(0)
        if requested_level < 0 or requested_level >= 8:
            new_lvl = wp.randi(reset_state) % 8
            if new_lvl < 0:
                new_lvl = -new_lvl
        else:
            new_lvl = requested_level
        lvl = new_lvl
        level[i] = lvl

        py = f_height - f_player_height / 2.0
        best_li = 0
        stunts_left = 0
        sc = 0
        new_ep_return = 0.0
        new_tick = 0

        # Re-spawn enemies
        for lane in range(10):
            lane_offset_x = f_width * wp.randf(reset_state)
            for slot in range(3):
                flat_idx = i * 30 + lane * 3 + slot
                enabled = int(0)
                if slot < tbl_enemies_per_lane[lvl * 10 + lane]:
                    enabled = 1
                enemy_enabled[flat_idx] = enabled

                init_spd_idx = tbl_enemies_initial_speed_idx[lvl * 10 + lane]
                enemy_speed_idx[flat_idx] = init_spd_idx

                etype = tbl_enemies_types[lvl * 10 + lane]
                e_w = int(0)
                if etype == 0:
                    e_w = car_width
                else:
                    e_w = truck_width

                init_x_frac = tbl_enemies_initial_x[lvl * 30 + lane * 3 + slot]
                ex = init_x_frac * f_width
                if lane >= 5:
                    ex = f_width - ex
                if env_randomization == 1 and enabled == 1:
                    ex = ex + lane_offset_x
                enemy_x[flat_idx] = ex

                spd = tbl_speed_values[init_spd_idx] * (1.0 / 60.0) * f_width
                if lane >= 5:
                    spd = -spd
                enemy_vx[flat_idx] = spd

    player_y[i] = py
    best_lane_idx[i] = best_li
    ticks_stunts_left[i] = stunts_left
    score[i] = sc
    ep_return[i] = new_ep_return
    tick[i] = new_tick

    # Write observations
    hhs = tbl_human_high_score[lvl]
    obs[i, 0] = py / f_height
    obs[i, 1] = float(best_li) / 10.0
    if hhs > 0:
        obs[i, 2] = float(sc) / float(hhs)
    else:
        obs[i, 2] = 0.0
    if stunts_left > 0:
        obs[i, 3] = 1.0
    else:
        obs[i, 3] = 0.0

    for lane in range(10):
        for slot in range(3):
            flat_idx = i * 30 + lane * 3 + slot
            obs_idx = 4 + lane * 3 + slot
            if enemy_enabled[flat_idx] == 1:
                etype = tbl_enemies_types[lvl * 10 + lane]
                e_h = car_height
                offset = float(0.0)
                if lane < 5:
                    offset = float(e_h) / (2.0 * f_width)
                else:
                    offset = -float(e_h) / (2.0 * f_width)
                obs[i, obs_idx] = enemy_x[flat_idx] / f_width + offset
            else:
                obs[i, obs_idx] = 0.0


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------
class FreewayEnv:
    """Vectorized Freeway environment on GPU.

    Args:
        num_envs: Number of parallel environments.
        width: Screen width in pixels.
        height: Screen height in pixels.
        player_width: Player sprite width.
        player_height: Player sprite height.
        car_width: Car sprite width.
        car_height: Car sprite height.
        lane_size: Vertical size of each lane in pixels.
        difficulty: 0 = push-back on hit, 1 = reset to bottom on hit.
        level: Level index (0-7), or -1 for random.
        use_dense_rewards: Whether to give per-lane and hit rewards.
        env_randomization: Whether to randomize enemy spawn offsets.
        frameskip: Physics sub-steps per action.
        device: Warp device string.
        seed: Random seed.
    """

    OBS_SIZE = OBS_SIZE  # 34
    NUM_ACTIONS = 3

    def __init__(
        self,
        num_envs: int = 1024,
        width: int = 1216,
        height: int = 720,
        player_width: int = 64,
        player_height: int = 64,
        car_width: int = 64,
        car_height: int = 40,
        lane_size: int = 64,
        difficulty: int = 0,
        level: int = -1,
        use_dense_rewards: int = 1,
        env_randomization: int = 1,
        frameskip: int = 4,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.width = float(width)
        self.height = float(height)
        self.player_width = float(player_width)
        self.player_height = float(player_height)
        self.car_width = car_width
        self.car_height = car_height
        self.lane_size = lane_size
        self.difficulty = difficulty
        self.level = level
        self.use_dense_rewards = use_dense_rewards
        self.env_randomization = env_randomization
        self.frameskip = frameskip
        self.device = device
        self.seed = seed
        self._step_count = 0

        n = num_envs
        ns = num_envs * NUM_ENEMY_SLOTS

        # Per-env player state
        self.player_y = wp.zeros(n, dtype=float, device=device)
        self.best_lane_idx = wp.zeros(n, dtype=int, device=device)
        self.ticks_stunts_left = wp.zeros(n, dtype=int, device=device)
        self.score = wp.zeros(n, dtype=int, device=device)
        self.ep_return = wp.zeros(n, dtype=float, device=device)
        self.tick = wp.zeros(n, dtype=int, device=device)
        self.level_arr = wp.zeros(n, dtype=int, device=device)

        # Enemy flat arrays
        self.enemy_x = wp.zeros(ns, dtype=float, device=device)
        self.enemy_vx = wp.zeros(ns, dtype=float, device=device)
        self.enemy_enabled = wp.zeros(ns, dtype=int, device=device)
        self.enemy_speed_idx = wp.zeros(ns, dtype=int, device=device)

        # Outputs
        self.obs = wp.zeros((n, self.OBS_SIZE), dtype=float, device=device)
        self.rewards = wp.zeros(n, dtype=float, device=device)
        self.dones = wp.zeros(n, dtype=float, device=device)
        self.episode_returns = wp.zeros(n, dtype=float, device=device)
        self.episode_lengths = wp.zeros(n, dtype=float, device=device)

        # Level data tables (uploaded once)
        self.tbl_enemies_per_lane = wp.array(_ENEMIES_PER_LANE, dtype=int, device=device)
        self.tbl_enemies_types = wp.array(_ENEMIES_TYPES, dtype=int, device=device)
        self.tbl_enemies_initial_x = wp.array(_ENEMIES_INITIAL_X, dtype=float, device=device)
        self.tbl_enemies_initial_speed_idx = wp.array(
            _ENEMIES_INITIAL_SPEED_IDX, dtype=int, device=device
        )
        self.tbl_speed_randomization = wp.array(_SPEED_RANDOMIZATION, dtype=int, device=device)
        self.tbl_speed_values = wp.array(_SPEED_VALUES, dtype=float, device=device)
        self.tbl_human_high_score = wp.array(_HUMAN_HIGH_SCORE, dtype=int, device=device)

        self.reset()

    def reset(self):
        wp.launch(
            freeway_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.player_y,
                self.best_lane_idx,
                self.ticks_stunts_left,
                self.score,
                self.ep_return,
                self.tick,
                self.level_arr,
                self.enemy_x,
                self.enemy_vx,
                self.enemy_enabled,
                self.enemy_speed_idx,
                self.tbl_enemies_per_lane,
                self.tbl_enemies_types,
                self.tbl_enemies_initial_x,
                self.tbl_enemies_initial_speed_idx,
                self.tbl_speed_randomization,
                self.tbl_speed_values,
                self.obs,
                self.width,
                self.height,
                self.player_height,
                self.car_width,
                self.car_height,
                self.lane_size,
                self.env_randomization,
                self.level,
                self.seed,
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
            freeway_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.player_y,
                self.best_lane_idx,
                self.ticks_stunts_left,
                self.score,
                self.ep_return,
                self.tick,
                self.level_arr,
                self.enemy_x,
                self.enemy_vx,
                self.enemy_enabled,
                self.enemy_speed_idx,
                self.tbl_enemies_per_lane,
                self.tbl_enemies_types,
                self.tbl_enemies_initial_x,
                self.tbl_enemies_initial_speed_idx,
                self.tbl_speed_randomization,
                self.tbl_speed_values,
                self.tbl_human_high_score,
                actions,
                self.obs,
                self.rewards,
                self.dones,
                self.episode_returns,
                self.episode_lengths,
                self.width,
                self.height,
                self.player_width,
                self.player_height,
                self.car_width,
                self.car_height,
                self.lane_size,
                self.difficulty,
                self.use_dense_rewards,
                self.env_randomization,
                self.frameskip,
                self.level,
                seed_arr,
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
