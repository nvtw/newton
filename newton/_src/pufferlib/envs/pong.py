# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pong environment — Warp port of PufferLib ocean/pong.

Agent controls the right paddle against a tracking-AI left paddle.
Observation: 8 normalized floats (paddle positions, ball pos/vel, scores).
Discrete action space: 0=still, 1=up, 2=down.
Reward: +1 per point scored, -1 per point conceded.
Episode ends when either side reaches ``max_score``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def pong_reset_kernel(
    paddle_yl: wp.array(dtype=float, ndim=1),
    paddle_yr: wp.array(dtype=float, ndim=1),
    ball_x: wp.array(dtype=float, ndim=1),
    ball_y: wp.array(dtype=float, ndim=1),
    ball_vx: wp.array(dtype=float, ndim=1),
    ball_vy: wp.array(dtype=float, ndim=1),
    score_l: wp.array(dtype=int, ndim=1),
    score_r: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    width: float,
    height: float,
    paddle_height: float,
    ball_height: float,
    ball_initial_speed_x: float,
    ball_initial_speed_y: float,
    ball_max_speed_y: float,
    max_score: int,
    seed: int,
):
    i = wp.tid()
    min_py = -paddle_height / 2.0
    max_py = height - paddle_height / 2.0

    paddle_yl[i] = height / 2.0 - paddle_height / 2.0
    paddle_yr[i] = height / 2.0 - paddle_height / 2.0
    ball_x[i] = width / 5.0
    ball_y[i] = height / 2.0 - ball_height / 2.0

    state = wp.rand_init(seed, i)
    ball_vx[i] = ball_initial_speed_x
    if wp.randi(state) % 2 == 0:
        ball_vy[i] = -ball_initial_speed_y
    else:
        ball_vy[i] = ball_initial_speed_y

    score_l[i] = 0
    score_r[i] = 0
    tick[i] = 0
    episode_return[i] = 0.0

    # Write observations
    obs[i, 0] = (paddle_yl[i] - min_py) / (max_py - min_py)
    obs[i, 1] = (paddle_yr[i] - min_py) / (max_py - min_py)
    obs[i, 2] = ball_x[i] / width
    obs[i, 3] = ball_y[i] / height
    obs[i, 4] = (ball_vx[i] + ball_initial_speed_x) / (2.0 * ball_initial_speed_x)
    obs[i, 5] = (ball_vy[i] + ball_max_speed_y) / (2.0 * ball_max_speed_y)
    obs[i, 6] = 0.0
    obs[i, 7] = 0.0


@wp.kernel
def pong_step_kernel(
    paddle_yl: wp.array(dtype=float, ndim=1),
    paddle_yr: wp.array(dtype=float, ndim=1),
    ball_x: wp.array(dtype=float, ndim=1),
    ball_y: wp.array(dtype=float, ndim=1),
    ball_vx: wp.array(dtype=float, ndim=1),
    ball_vy: wp.array(dtype=float, ndim=1),
    score_l: wp.array(dtype=int, ndim=1),
    score_r: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    actions: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    width: float,
    height: float,
    paddle_width: float,
    paddle_height: float,
    ball_width: float,
    ball_height: float,
    paddle_speed: float,
    ball_initial_speed_x: float,
    ball_initial_speed_y: float,
    ball_speed_y_increment: float,
    ball_max_speed_y: float,
    max_score: int,
    frameskip: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()
    min_py = -paddle_height / 2.0
    max_py = height - paddle_height / 2.0

    new_tick = tick[i] + 1
    reward = float(0.0)
    done = int(0)

    atn = int(actions[i])
    paddle_dir = 0.0
    if atn == 1:
        paddle_dir = 1.0
    elif atn == 2:
        paddle_dir = -1.0

    bx = ball_x[i]
    by = ball_y[i]
    bvx = ball_vx[i]
    bvy = ball_vy[i]
    pyl = paddle_yl[i]
    pyr = paddle_yr[i]
    sl = score_l[i]
    sr = score_r[i]

    round_over = int(0)

    for _f in range(frameskip):
        if round_over == 0:
            pyr = pyr + paddle_speed * paddle_dir

            # Opponent AI: track ball
            opp_delta = by - (pyl + paddle_height / 2.0)
            if opp_delta > paddle_speed:
                opp_delta = paddle_speed
            if opp_delta < -paddle_speed:
                opp_delta = -paddle_speed
            pyl = pyl + opp_delta

            # Clip paddles
            if pyr < min_py:
                pyr = min_py
            if pyr > max_py:
                pyr = max_py
            if pyl < min_py:
                pyl = min_py
            if pyl > max_py:
                pyl = max_py

            bx = bx + bvx
            by = by + bvy

            # Wall bounce (top/bottom)
            if by < 0.0 or by + ball_height > height:
                bvy = -bvy

            # Left wall collision (opponent misses → agent scores)
            if bx < 0.0:
                if by + ball_height > pyl and by < pyl + paddle_height:
                    bvx = -bvx
                else:
                    sr = sr + 1
                    reward = 1.0
                    if sr >= max_score:
                        done = 1
                    round_over = 1

            # Right wall collision (agent misses → opponent scores)
            if round_over == 0 and bx + ball_width > width:
                if by + ball_height > pyr and by < pyr + paddle_height:
                    bvx = -bvx
                    bvy = bvy + ball_speed_y_increment * paddle_dir
                    if bvy > ball_max_speed_y:
                        bvy = ball_max_speed_y
                    if bvy < -ball_max_speed_y:
                        bvy = -ball_max_speed_y
                    if bvy > -0.01 and bvy < 0.01:
                        bvy = ball_speed_y_increment
                else:
                    sl = sl + 1
                    reward = -1.0
                    if sl >= max_score:
                        done = 1
                    round_over = 1

            # Clip ball
            if bx < 0.0:
                bx = 0.0
            if bx > width - ball_width:
                bx = width - ball_width
            if by < 0.0:
                by = 0.0
            if by > height - ball_height:
                by = height - ball_height

    new_ep_return = episode_return[i] + reward
    rewards[i] = reward
    dones[i] = float(done)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(new_tick)

    if done == 1 or round_over == 1:
        if done == 1:
            # Full episode reset
            state = wp.rand_init(seed_arr[0], i + new_tick)
            pyl = height / 2.0 - paddle_height / 2.0
            pyr = height / 2.0 - paddle_height / 2.0
            bx = width / 5.0
            by = height / 2.0 - ball_height / 2.0
            bvx = ball_initial_speed_x
            if wp.randi(state) % 2 == 0:
                bvy = -ball_initial_speed_y
            else:
                bvy = ball_initial_speed_y
            sl = 0
            sr = 0
            new_tick = 0
            new_ep_return = 0.0
        else:
            # Round reset (not full episode)
            state = wp.rand_init(seed_arr[0], i + new_tick + 1000)
            pyl = height / 2.0 - paddle_height / 2.0
            pyr = height / 2.0 - paddle_height / 2.0
            bx = width / 5.0
            by = height / 2.0 - ball_height / 2.0
            bvx = ball_initial_speed_x
            if wp.randi(state) % 2 == 0:
                bvy = -ball_initial_speed_y
            else:
                bvy = ball_initial_speed_y

    paddle_yl[i] = pyl
    paddle_yr[i] = pyr
    ball_x[i] = bx
    ball_y[i] = by
    ball_vx[i] = bvx
    ball_vy[i] = bvy
    score_l[i] = sl
    score_r[i] = sr
    tick[i] = new_tick
    episode_return[i] = new_ep_return

    # Write observations
    obs[i, 0] = (pyl - min_py) / (max_py - min_py)
    obs[i, 1] = (pyr - min_py) / (max_py - min_py)
    obs[i, 2] = bx / width
    obs[i, 3] = by / height
    obs[i, 4] = (bvx + ball_initial_speed_x) / (2.0 * ball_initial_speed_x)
    obs[i, 5] = (bvy + ball_max_speed_y) / (2.0 * ball_max_speed_y)
    obs[i, 6] = float(sl) / float(max_score)
    obs[i, 7] = float(sr) / float(max_score)


class PongEnv:
    """Vectorized Pong environment on GPU.

    Args:
        num_envs: Number of parallel environments.
        width: Court width.
        height: Court height.
        paddle_width: Paddle width.
        paddle_height: Paddle height.
        ball_width: Ball width.
        ball_height: Ball height.
        paddle_speed: Paddle movement speed per frame.
        ball_initial_speed_x: Initial horizontal ball speed.
        ball_initial_speed_y: Initial vertical ball speed.
        ball_speed_y_increment: Vertical speed change on paddle hit.
        ball_max_speed_y: Maximum vertical ball speed.
        max_score: Points to win.
        frameskip: Physics sub-steps per action.
        device: Warp device string.
        seed: Random seed.
    """

    OBS_SIZE = 8
    NUM_ACTIONS = 3

    def __init__(
        self,
        num_envs: int = 1024,
        width: float = 500.0,
        height: float = 640.0,
        paddle_width: float = 20.0,
        paddle_height: float = 70.0,
        ball_width: float = 32.0,
        ball_height: float = 32.0,
        paddle_speed: float = 8.0,
        ball_initial_speed_x: float = 10.0,
        ball_initial_speed_y: float = 1.0,
        ball_speed_y_increment: float = 3.0,
        ball_max_speed_y: float = 13.0,
        max_score: int = 21,
        frameskip: int = 8,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.paddle_speed = paddle_speed
        self.ball_initial_speed_x = ball_initial_speed_x
        self.ball_initial_speed_y = ball_initial_speed_y
        self.ball_speed_y_increment = ball_speed_y_increment
        self.ball_max_speed_y = ball_max_speed_y
        self.max_score = max_score
        self.frameskip = frameskip
        self.device = device
        self.seed = seed
        self._step_count = 0

        self.paddle_yl = wp.zeros(num_envs, dtype=float, device=device)
        self.paddle_yr = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_x = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_y = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_vx = wp.zeros(num_envs, dtype=float, device=device)
        self.ball_vy = wp.zeros(num_envs, dtype=float, device=device)
        self.score_l = wp.zeros(num_envs, dtype=int, device=device)
        self.score_r = wp.zeros(num_envs, dtype=int, device=device)
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
            pong_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.paddle_yl, self.paddle_yr,
                self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
                self.score_l, self.score_r,
                self.tick, self.episode_return, self.obs,
                self.width, self.height,
                self.paddle_height, self.ball_height,
                self.ball_initial_speed_x, self.ball_initial_speed_y,
                self.ball_max_speed_y, self.max_score, self.seed,
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
            pong_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.paddle_yl, self.paddle_yr,
                self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
                self.score_l, self.score_r,
                self.tick, self.episode_return,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.width, self.height,
                self.paddle_width, self.paddle_height,
                self.ball_width, self.ball_height,
                self.paddle_speed,
                self.ball_initial_speed_x, self.ball_initial_speed_y,
                self.ball_speed_y_increment, self.ball_max_speed_y,
                self.max_score, self.frameskip, seed_arr,
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
