# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Breakout environment — Warp port of PufferLib ocean/breakout.

Agent controls a paddle to bounce a ball and destroy bricks.
Observation: 10 normalized floats + 108 brick states = 118 total.
Discrete action space: 0=NOOP, 1=LEFT, 2=RIGHT.
Reward: points per brick destroyed (top rows worth more).
Episode ends when all balls lost or max score reached.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

BRICK_ROWS = 6
BRICK_COLS = 18
NUM_BRICKS = BRICK_ROWS * BRICK_COLS
Y_OFFSET = 50.0
HALF_PADDLE_WIDTH = 31.0
TICK_RATE = 1.0 / 60.0
PI = 3.14159265358979323846


@wp.func
def calc_vline_collision(
    xw: float, yw: float, hw: float,
    x: float, y: float, vx: float, vy: float, h: float,
    col_t: float, col_overlap: float,
):
    """Collision of a moving horizontal segment against a stationary vertical line.

    Returns (hit, new_t, new_overlap, new_x, new_y, new_vx, new_vy).
    """
    t_new = (xw - x) / vx
    topmost = wp.min(yw + hw, y + h + vy * t_new)
    botmost = wp.max(yw, y + vy * t_new)
    overlap_new = topmost - botmost

    hit = int(0)
    out_t = col_t
    out_overlap = col_overlap
    out_x = float(0.0)
    out_y = float(0.0)
    out_vx = vx
    out_vy = vy

    if overlap_new > 0.0 and t_new > 0.0 and t_new <= 1.0:
        if t_new < col_t or (t_new == col_t and overlap_new > col_overlap):
            hit = 1
            out_t = t_new
            out_overlap = overlap_new
            out_x = xw
            out_y = y + vy * t_new
            out_vx = -vx
            out_vy = vy

    return hit, out_t, out_overlap, out_x, out_y, out_vx, out_vy


@wp.func
def calc_hline_collision(
    xw: float, yw: float, ww: float,
    x: float, y: float, vx: float, vy: float, w: float,
    col_t: float, col_overlap: float,
):
    """Collision of a moving vertical segment against a stationary horizontal line.

    Returns (hit, new_t, new_overlap, new_x, new_y, new_vx, new_vy).
    """
    t_new = (yw - y) / vy
    rightmost = wp.min(xw + ww, x + w + vx * t_new)
    leftmost = wp.max(xw, x + vx * t_new)
    overlap_new = rightmost - leftmost

    hit = int(0)
    out_t = col_t
    out_overlap = col_overlap
    out_x = float(0.0)
    out_y = float(0.0)
    out_vx = vx
    out_vy = vy

    if overlap_new > 0.0 and t_new > 0.0 and t_new <= 1.0:
        if t_new < col_t or (t_new == col_t and overlap_new > col_overlap):
            hit = 1
            out_t = t_new
            out_overlap = overlap_new
            out_x = x + vx * t_new
            out_y = yw
            out_vx = vx
            out_vy = -vy

    return hit, out_t, out_overlap, out_x, out_y, out_vx, out_vy


@wp.kernel
def breakout_reset_kernel(
    paddle_x: wp.array(dtype=float, ndim=1),
    paddle_y: wp.array(dtype=float, ndim=1),
    paddle_w: wp.array(dtype=float, ndim=1),
    ball_x: wp.array(dtype=float, ndim=1),
    ball_y: wp.array(dtype=float, ndim=1),
    ball_vx: wp.array(dtype=float, ndim=1),
    ball_vy: wp.array(dtype=float, ndim=1),
    ball_speed: wp.array(dtype=float, ndim=1),
    score: wp.array(dtype=int, ndim=1),
    num_balls: wp.array(dtype=int, ndim=1),
    balls_fired: wp.array(dtype=int, ndim=1),
    hits: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    brick_states: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    width: float,
    height: float,
    paddle_height: float,
    initial_paddle_width: float,
    ball_width: float,
    ball_height: float,
    initial_ball_speed: float,
    brick_rows: int,
    brick_cols: int,
    brick_width: float,
    brick_height: float,
):
    i = wp.tid()
    num_bricks = brick_rows * brick_cols

    pw = initial_paddle_width
    px = width / 2.0 - pw / 2.0
    py = height - paddle_height - 10.0
    bx = px + (pw / 2.0 - ball_width / 2.0)
    by = height / 2.0 - 30.0

    paddle_x[i] = px
    paddle_y[i] = py
    paddle_w[i] = pw
    ball_x[i] = bx
    ball_y[i] = by
    ball_vx[i] = 0.0
    ball_vy[i] = 0.0
    ball_speed[i] = initial_ball_speed
    score[i] = 0
    num_balls[i] = 5
    balls_fired[i] = 0
    hits[i] = 0
    tick[i] = 0
    episode_return[i] = 0.0

    for b in range(num_bricks):
        brick_states[i * num_bricks + b] = 0.0

    obs[i, 0] = px / width
    obs[i, 1] = py / height
    obs[i, 2] = bx / width
    obs[i, 3] = by / height
    obs[i, 4] = 0.0
    obs[i, 5] = 0.0
    obs[i, 6] = 0.0
    obs[i, 7] = 0.0
    obs[i, 8] = 5.0 / 5.0
    obs[i, 9] = pw / (2.0 * 31.0)
    for b in range(num_bricks):
        obs[i, 10 + b] = 0.0


@wp.kernel
def breakout_step_kernel(
    paddle_x: wp.array(dtype=float, ndim=1),
    paddle_y: wp.array(dtype=float, ndim=1),
    paddle_w: wp.array(dtype=float, ndim=1),
    ball_x: wp.array(dtype=float, ndim=1),
    ball_y: wp.array(dtype=float, ndim=1),
    ball_vx: wp.array(dtype=float, ndim=1),
    ball_vy: wp.array(dtype=float, ndim=1),
    ball_speed_arr: wp.array(dtype=float, ndim=1),
    score_arr: wp.array(dtype=int, ndim=1),
    num_balls_arr: wp.array(dtype=int, ndim=1),
    balls_fired_arr: wp.array(dtype=int, ndim=1),
    hits_arr: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    brick_states: wp.array(dtype=float, ndim=1),
    brick_x_arr: wp.array(dtype=float, ndim=1),
    brick_y_arr: wp.array(dtype=float, ndim=1),
    actions: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    width: float,
    height: float,
    paddle_height: float,
    initial_paddle_width: float,
    ball_width_f: float,
    ball_height_f: float,
    paddle_speed: float,
    initial_ball_speed: float,
    max_ball_speed: float,
    brick_rows: int,
    brick_cols: int,
    brick_width_f: float,
    brick_height_f: float,
    half_max_score: int,
    max_score: int,
    frameskip: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()
    num_bricks = brick_rows * brick_cols
    tick_rate = 1.0 / 60.0
    half_paddle_w = 31.0
    y_offset = 50.0
    base_angle = 3.14159265358979323846 / 4.0
    fire_angle = 3.14159265358979323846 / 3.25

    new_tick = tick[i]
    reward = float(0.0)
    done = int(0)

    atn = int(actions[i])

    px = paddle_x[i]
    py = paddle_y[i]
    pw = paddle_w[i]
    bx = ball_x[i]
    by = ball_y[i]
    bvx = ball_vx[i]
    bvy = ball_vy[i]
    bspeed = ball_speed_arr[i]
    sc = score_arr[i]
    nb = num_balls_arr[i]
    bf = balls_fired_arr[i]
    ht = hits_arr[i]

    for _f in range(frameskip):
        if done == 0:
            new_tick = new_tick + 1

            # Fire ball on first action
            if bf == 0:
                bf = 1
                bvy = wp.cos(fire_angle) * bspeed * tick_rate
                bvx = wp.sin(fire_angle) * bspeed * tick_rate
                state = wp.rand_init(seed_arr[0], i + new_tick)
                if wp.randi(state) % 2 == 0:
                    bvx = -bvx
            else:
                act = float(0.0)
                if atn == 1:
                    act = -1.0
                elif atn == 2:
                    act = 1.0
                px = px + act * paddle_speed * tick_rate
                if px <= 0.0:
                    px = 0.0
                if px > width - pw:
                    px = width - pw

            # --- Wall bounds safety ---
            offset = max_ball_speed * 1.1 * tick_rate
            if bx < 0.0:
                bx = bx + offset
            if bx > width:
                bx = bx - offset
            if by < 0.0:
                by = by + offset

            # --- Collision detection ---
            col_t = float(2.0)
            col_overlap = float(-1.0)
            col_x = float(0.0)
            col_y = float(0.0)
            col_vx = float(0.0)
            col_vy = float(0.0)
            col_brick = int(-4)  # NO_COLLISION

            # Brick collisions — scan bricks in the ball's trajectory bounding box
            ball_x_dst = bx + bvx
            ball_y_dst = by + bvy

            min_bx = bx
            if ball_x_dst < min_bx:
                min_bx = ball_x_dst
            min_by = by
            if ball_y_dst < min_by:
                min_by = ball_y_dst

            max_bx_end = bx + ball_width_f
            if ball_x_dst + ball_width_f > max_bx_end:
                max_bx_end = ball_x_dst + ball_width_f
            max_by_end = by + ball_height_f
            if ball_y_dst + ball_height_f > max_by_end:
                max_by_end = ball_y_dst + ball_height_f

            row_from = int((min_by - y_offset) / brick_height_f)
            if row_from < 0:
                row_from = 0
            col_from = int(min_bx / brick_width_f)
            if col_from < 0:
                col_from = 0
            row_to = int((max_by_end - y_offset) / brick_height_f)
            if row_to >= brick_rows:
                row_to = brick_rows - 1
            col_to = int(max_bx_end / brick_width_f)
            if col_to >= brick_cols:
                col_to = brick_cols - 1

            skip_bricks = int(0)
            if row_from > brick_rows:
                skip_bricks = 1

            if skip_bricks == 0:
                for row in range(brick_rows):
                    if row >= row_from and row <= row_to:
                        for col in range(brick_cols):
                            if col >= col_from and col <= col_to:
                                bidx = row * brick_cols + col
                                if brick_states[i * num_bricks + bidx] == 0.0:
                                    brk_x = brick_x_arr[bidx]
                                    brk_y = brick_y_arr[bidx]

                                    # Brick left wall vs ball right side
                                    if bvx > 0.0:
                                        h1, t1, o1, x1, y1, vx1, vy1 = calc_vline_collision(
                                            brk_x, brk_y, brick_height_f,
                                            bx + ball_width_f, by, bvx, bvy, ball_height_f,
                                            col_t, col_overlap,
                                        )
                                        if h1 == 1:
                                            col_t = t1
                                            col_overlap = o1
                                            col_x = x1 - ball_width_f
                                            col_y = y1
                                            col_vx = vx1
                                            col_vy = vy1
                                            col_brick = bidx

                                    # Brick right wall vs ball left side
                                    if bvx < 0.0:
                                        h2, t2, o2, x2, y2, vx2, vy2 = calc_vline_collision(
                                            brk_x + brick_width_f, brk_y, brick_height_f,
                                            bx, by, bvx, bvy, ball_height_f,
                                            col_t, col_overlap,
                                        )
                                        if h2 == 1:
                                            col_t = t2
                                            col_overlap = o2
                                            col_x = x2
                                            col_y = y2
                                            col_vx = vx2
                                            col_vy = vy2
                                            col_brick = bidx

                                    # Brick top wall vs ball bottom side
                                    if bvy > 0.0:
                                        h3, t3, o3, x3, y3, vx3, vy3 = calc_hline_collision(
                                            brk_x, brk_y, brick_width_f,
                                            bx, by + ball_height_f, bvx, bvy, ball_width_f,
                                            col_t, col_overlap,
                                        )
                                        if h3 == 1:
                                            col_t = t3
                                            col_overlap = o3
                                            col_x = x3
                                            col_y = y3 - ball_height_f
                                            col_vx = vx3
                                            col_vy = vy3
                                            col_brick = bidx

                                    # Brick bottom wall vs ball top side
                                    if bvy < 0.0:
                                        h4, t4, o4, x4, y4, vx4, vy4 = calc_hline_collision(
                                            brk_x, brk_y + brick_height_f, brick_width_f,
                                            bx, by, bvx, bvy, ball_width_f,
                                            col_t, col_overlap,
                                        )
                                        if h4 == 1:
                                            col_t = t4
                                            col_overlap = o4
                                            col_x = x4
                                            col_y = y4
                                            col_vx = vx4
                                            col_vy = vy4
                                            col_brick = bidx

            # Wall collisions
            # Left wall
            if bvx < 0.0:
                hw, tw, ow, xw, yw, vxw, vyw = calc_vline_collision(
                    0.0, 0.0, height,
                    bx, by, bvx, bvy, ball_height_f,
                    col_t, col_overlap,
                )
                if hw == 1:
                    col_t = tw
                    col_overlap = ow
                    col_x = xw
                    col_y = yw
                    col_vx = vxw
                    col_vy = vyw
                    col_brick = -3  # SIDEWALL

            # Right wall
            if bvx > 0.0:
                hw2, tw2, ow2, xw2, yw2, vxw2, vyw2 = calc_vline_collision(
                    width, 0.0, height,
                    bx + ball_width_f, by, bvx, bvy, ball_height_f,
                    col_t, col_overlap,
                )
                if hw2 == 1:
                    col_t = tw2
                    col_overlap = ow2
                    col_x = xw2 - ball_width_f
                    col_y = yw2
                    col_vx = vxw2
                    col_vy = vyw2
                    col_brick = -3  # SIDEWALL

            # Top wall (back wall)
            if bvy < 0.0:
                hw3, tw3, ow3, xw3, yw3, vxw3, vyw3 = calc_hline_collision(
                    0.0, 0.0, width,
                    bx, by, bvx, bvy, ball_width_f,
                    col_t, col_overlap,
                )
                if hw3 == 1:
                    col_t = tw3
                    col_overlap = ow3
                    col_x = xw3
                    col_y = yw3
                    col_vx = vxw3
                    col_vy = vyw3
                    col_brick = -2  # BACKWALL

            # Paddle collision
            paddle_hit = int(0)
            if by + ball_height_f + bvy >= py:
                hp, tp, op, xp, yp, vxp, vyp = calc_hline_collision(
                    px, py, pw,
                    bx, by + ball_height_f, bvx, bvy, ball_width_f,
                    col_t, col_overlap,
                )
                if hp == 1 and tp <= 1.0:
                    col_t = tp
                    col_overlap = op
                    col_x = xp
                    col_y = yp - ball_height_f
                    col_brick = -1  # PADDLE
                    paddle_hit = 1

                    relative_intersection = ((bx + ball_width_f / 2.0) - px) / pw
                    angle = -base_angle + relative_intersection * 2.0 * base_angle
                    col_vx = wp.sin(angle) * bspeed * tick_rate
                    col_vy = -wp.cos(angle) * bspeed * tick_rate

                    ht = ht + 1
                    if ht % 4 == 0 and bspeed < max_ball_speed:
                        bspeed = bspeed + 64.0

                    # Second stage: reset bricks at half_max_score
                    if sc == half_max_score:
                        for rb in range(num_bricks):
                            brick_states[i * num_bricks + rb] = 0.0

            # Apply collision results
            # C++ handle_collisions: paddle hit only updates velocity (no position snap);
            # non-paddle collisions update both position and velocity.
            # In both cases the regular integration (bx += bvx) is skipped.
            if paddle_hit == 1:
                bvx = col_vx
                bvy = col_vy
            elif col_t <= 1.0:
                bx = col_x
                by = col_y
                bvx = col_vx
                bvy = col_vy
                if col_brick >= 0:
                    # Destroy brick — use integer division for row/score
                    brick_states[i * num_bricks + col_brick] = 1.0
                    brick_row = int(col_brick) // int(brick_cols)
                    gained = 7 - 3 * (brick_row // 2)
                    sc = sc + gained
                    reward = reward + float(gained)
                    if brick_row < 3:
                        bspeed = max_ball_speed
                if col_brick == -2:  # BACKWALL
                    pw = half_paddle_w
            else:
                # No collision: regular timestep
                bx = bx + bvx
                by = by + bvy

            # Ball fell below paddle
            if by >= py + paddle_height:
                nb = nb - 1
                # Reset round
                bf = 0
                ht = 0
                bspeed = initial_ball_speed
                pw = initial_paddle_width
                px = width / 2.0 - pw / 2.0
                py = height - paddle_height - 10.0
                bx = px + (pw / 2.0 - ball_width_f / 2.0)
                by = height / 2.0 - 30.0
                bvx = 0.0
                bvy = 0.0

            # Check terminal
            if nb < 0 or sc >= max_score:
                done = 1

    new_ep_return = episode_return[i] + reward
    rewards[i] = reward
    dones[i] = float(done)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(new_tick)

    if done == 1:
        # Full episode reset
        sc = 0
        nb = 5
        bf = 0
        ht = 0
        bspeed = initial_ball_speed
        pw = initial_paddle_width
        px = width / 2.0 - pw / 2.0
        py = height - paddle_height - 10.0
        bx = px + (pw / 2.0 - ball_width_f / 2.0)
        by = height / 2.0 - 30.0
        bvx = 0.0
        bvy = 0.0
        new_tick = 0
        new_ep_return = 0.0
        for rb in range(num_bricks):
            brick_states[i * num_bricks + rb] = 0.0

    paddle_x[i] = px
    paddle_y[i] = py
    paddle_w[i] = pw
    ball_x[i] = bx
    ball_y[i] = by
    ball_vx[i] = bvx
    ball_vy[i] = bvy
    ball_speed_arr[i] = bspeed
    score_arr[i] = sc
    num_balls_arr[i] = nb
    balls_fired_arr[i] = bf
    hits_arr[i] = ht
    tick[i] = new_tick
    episode_return[i] = new_ep_return

    # Write observations
    obs[i, 0] = px / width
    obs[i, 1] = py / height
    obs[i, 2] = bx / width
    obs[i, 3] = by / height
    obs[i, 4] = bvx / 512.0
    obs[i, 5] = bvy / 512.0
    obs[i, 6] = float(bf) / 5.0
    obs[i, 7] = float(sc) / 864.0
    obs[i, 8] = float(nb) / 5.0
    obs[i, 9] = pw / (2.0 * 31.0)
    for b in range(num_bricks):
        obs[i, 10 + b] = brick_states[i * num_bricks + b]


class BreakoutEnv:
    """Vectorized Breakout environment on GPU.

    Args:
        num_envs: Number of parallel environments.
        width: Court width.
        height: Court height.
        initial_paddle_width: Starting paddle width.
        paddle_height: Paddle height.
        ball_width: Ball width.
        ball_height: Ball height.
        brick_width: Brick width.
        brick_height: Brick height.
        paddle_speed: Paddle movement speed.
        initial_ball_speed: Starting ball speed.
        max_ball_speed: Maximum ball speed.
        brick_rows: Number of brick rows.
        brick_cols: Number of brick columns.
        frameskip: Physics sub-steps per action.
        device: Warp device string.
        seed: Random seed.
    """

    OBS_SIZE = 10 + BRICK_ROWS * BRICK_COLS
    NUM_ACTIONS = 3

    def __init__(
        self,
        num_envs: int = 1024,
        width: float = 576.0,
        height: float = 330.0,
        initial_paddle_width: float = 62.0,
        paddle_height: float = 8.0,
        ball_width: float = 32.0,
        ball_height: float = 32.0,
        brick_width: float = 32.0,
        brick_height: float = 12.0,
        paddle_speed: float = 620.0,
        initial_ball_speed: float = 256.0,
        max_ball_speed: float = 448.0,
        brick_rows: int = BRICK_ROWS,
        brick_cols: int = BRICK_COLS,
        frameskip: int = 4,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.initial_paddle_width = initial_paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.paddle_speed = paddle_speed
        self.initial_ball_speed = initial_ball_speed
        self.max_ball_speed = max_ball_speed
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols
        self.num_bricks = brick_rows * brick_cols
        self.frameskip = frameskip
        self.device = device
        self.seed = seed
        self._step_count = 0

        # Compute half_max_score and max_score matching the C version
        half_max_score = 0
        for row in range(brick_rows):
            for col in range(brick_cols):
                half_max_score += 7 - 3 * (row // 2)
        self.half_max_score = half_max_score
        self.max_score = 2 * half_max_score

        # Precompute brick positions (shared across all envs)
        brick_x_np = np.zeros(self.num_bricks, dtype=np.float32)
        brick_y_np = np.zeros(self.num_bricks, dtype=np.float32)
        for row in range(brick_rows):
            for col in range(brick_cols):
                idx = row * brick_cols + col
                brick_x_np[idx] = col * brick_width
                brick_y_np[idx] = row * brick_height + Y_OFFSET
        self.brick_x_arr = wp.array(brick_x_np, dtype=float, device=device)
        self.brick_y_arr = wp.array(brick_y_np, dtype=float, device=device)

        # Per-env state
        self.paddle_x = wp.zeros(num_envs, dtype=float, device=device)
        self.paddle_y = wp.zeros(num_envs, dtype=float, device=device)
        self.paddle_w = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_x = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_y = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_vx = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_vy = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_speed = wp.zeros(num_envs, dtype=float, device=device)
        self.score = wp.zeros(num_envs, dtype=int, device=device)
        self.num_balls = wp.zeros(num_envs, dtype=int, device=device)
        self.balls_fired = wp.zeros(num_envs, dtype=int, device=device)
        self.hits = wp.zeros(num_envs, dtype=int, device=device)
        self.tick = wp.zeros(num_envs, dtype=int, device=device)
        self.episode_return = wp.zeros(num_envs, dtype=float, device=device)

        # Brick states: flat array (num_envs * num_bricks)
        self.brick_states = wp.zeros(num_envs * self.num_bricks, dtype=float, device=device)

        self.obs = wp.zeros((num_envs, self.OBS_SIZE), dtype=float, device=device)
        self.rewards = wp.zeros(num_envs, dtype=float, device=device)
        self.dones = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_returns = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_lengths = wp.zeros(num_envs, dtype=float, device=device)

        self.reset()

    def reset(self):
        wp.launch(
            breakout_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.paddle_x, self.paddle_y, self.paddle_w,
                self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
                self.ball_speed, self.score, self.num_balls,
                self.balls_fired, self.hits,
                self.tick, self.episode_return,
                self.brick_states, self.obs,
                self.width, self.height,
                self.paddle_height, self.initial_paddle_width,
                self.ball_width, self.ball_height,
                self.initial_ball_speed,
                self.brick_rows, self.brick_cols,
                self.brick_width, self.brick_height,
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
            breakout_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.paddle_x, self.paddle_y, self.paddle_w,
                self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
                self.ball_speed, self.score, self.num_balls,
                self.balls_fired, self.hits,
                self.tick, self.episode_return,
                self.brick_states,
                self.brick_x_arr, self.brick_y_arr,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.width, self.height,
                self.paddle_height, self.initial_paddle_width,
                self.ball_width, self.ball_height,
                self.paddle_speed,
                self.initial_ball_speed, self.max_ball_speed,
                self.brick_rows, self.brick_cols,
                self.brick_width, self.brick_height,
                self.half_max_score, self.max_score,
                self.frameskip, seed_arr,
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
