# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asteroids environment — Warp port of PufferLib ocean/asteroids.

Player controls a ship in a wrap-around arena, shooting asteroids for points.
Observation: 104 normalized floats (player state + up to 20 asteroids sorted
by distance, each with 5 features).
Discrete action space: 0=forward, 1=turn_left, 2=turn_right, 3=shoot.
Reward: +1 per asteroid destroyed, -1 on player-asteroid collision (terminal).
Episode ends on collision or after ``max_tick`` frames.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

MAX_PARTICLES = 10
MAX_ASTEROIDS = 20
OBS_SIZE = 4 + MAX_ASTEROIDS * 5  # 104


@wp.kernel
def asteroids_reset_kernel(
    player_x: wp.array(dtype=float, ndim=1),
    player_y: wp.array(dtype=float, ndim=1),
    player_vx: wp.array(dtype=float, ndim=1),
    player_vy: wp.array(dtype=float, ndim=1),
    player_angle: wp.array(dtype=float, ndim=1),
    particle_x: wp.array(dtype=float, ndim=1),
    particle_y: wp.array(dtype=float, ndim=1),
    particle_vx: wp.array(dtype=float, ndim=1),
    particle_vy: wp.array(dtype=float, ndim=1),
    asteroid_x: wp.array(dtype=float, ndim=1),
    asteroid_y: wp.array(dtype=float, ndim=1),
    asteroid_vx: wp.array(dtype=float, ndim=1),
    asteroid_vy: wp.array(dtype=float, ndim=1),
    asteroid_radius: wp.array(dtype=float, ndim=1),
    particle_index: wp.array(dtype=int, ndim=1),
    asteroid_index: wp.array(dtype=int, ndim=1),
    last_shot: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    score: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    arena_size: float,
    seed: int,
):
    i = wp.tid()

    player_x[i] = arena_size / 2.0
    player_y[i] = arena_size / 2.0
    player_vx[i] = 0.0
    player_vy[i] = 0.0
    player_angle[i] = 0.0
    particle_index[i] = 0
    asteroid_index[i] = 0
    last_shot[i] = 0
    tick[i] = 0
    score[i] = 0
    episode_return[i] = 0.0

    for p in range(MAX_PARTICLES):
        idx = i * MAX_PARTICLES + p
        particle_x[idx] = 0.0
        particle_y[idx] = 0.0
        particle_vx[idx] = 0.0
        particle_vy[idx] = 0.0

    for a in range(MAX_ASTEROIDS):
        idx = i * MAX_ASTEROIDS + a
        asteroid_x[idx] = 0.0
        asteroid_y[idx] = 0.0
        asteroid_vx[idx] = 0.0
        asteroid_vy[idx] = 0.0
        asteroid_radius[idx] = 0.0

    obs[i, 0] = 0.5
    obs[i, 1] = 0.5
    obs[i, 2] = 0.0
    obs[i, 3] = 0.0
    for k in range(MAX_ASTEROIDS * 5):
        obs[i, 4 + k] = 0.0


@wp.kernel
def asteroids_step_kernel(
    player_x: wp.array(dtype=float, ndim=1),
    player_y: wp.array(dtype=float, ndim=1),
    player_vx: wp.array(dtype=float, ndim=1),
    player_vy: wp.array(dtype=float, ndim=1),
    player_angle: wp.array(dtype=float, ndim=1),
    particle_x: wp.array(dtype=float, ndim=1),
    particle_y: wp.array(dtype=float, ndim=1),
    particle_vx: wp.array(dtype=float, ndim=1),
    particle_vy: wp.array(dtype=float, ndim=1),
    asteroid_x: wp.array(dtype=float, ndim=1),
    asteroid_y: wp.array(dtype=float, ndim=1),
    asteroid_vx: wp.array(dtype=float, ndim=1),
    asteroid_vy: wp.array(dtype=float, ndim=1),
    asteroid_radius: wp.array(dtype=float, ndim=1),
    particle_index: wp.array(dtype=int, ndim=1),
    asteroid_index: wp.array(dtype=int, ndim=1),
    last_shot: wp.array(dtype=int, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    score: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    actions: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    arena_size: float,
    player_radius: float,
    friction: float,
    speed: float,
    particle_speed: float,
    rotation_speed: float,
    asteroid_speed: float,
    shoot_delay: int,
    max_tick: int,
    frameskip: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()

    px = player_x[i]
    py = player_y[i]
    pvx = player_vx[i]
    pvy = player_vy[i]
    pa = player_angle[i]
    p_idx = particle_index[i]
    a_idx = asteroid_index[i]
    ls = last_shot[i]
    t = tick[i]
    sc = score[i]

    atn = int(actions[i])

    reward = float(0.0)
    done = int(0)

    for _f in range(frameskip):
        if done == 0:
            t = t + 1

            # Apply friction
            pvx = pvx * friction
            pvy = pvy * friction

            # Direction vector: rotate (0, -1) by player_angle → (sin(a), -cos(a))
            dir_x = wp.sin(pa)
            dir_y = -wp.cos(pa)

            # Handle actions
            if atn == 1:
                pa = pa - rotation_speed
            if atn == 2:
                pa = pa + rotation_speed
            if atn == 0:
                pvx = pvx + dir_x * speed
                pvy = pvy + dir_y * speed

            elapsed = t - ls
            if atn == 3 and elapsed >= shoot_delay:
                ls = t
                p_idx = (p_idx + 1) % MAX_PARTICLES
                pidx = i * MAX_PARTICLES + p_idx
                particle_x[pidx] = px + 20.0 * dir_x
                particle_y[pidx] = py + 20.0 * dir_y
                particle_vx[pidx] = dir_x
                particle_vy[pidx] = dir_y

            # Move player
            px = px + pvx
            py = py + pvy

            # Move particles
            for p in range(MAX_PARTICLES):
                pidx = i * MAX_PARTICLES + p
                particle_x[pidx] = particle_x[pidx] + particle_vx[pidx] * particle_speed
                particle_y[pidx] = particle_y[pidx] + particle_vy[pidx] * particle_speed

            # Spawn asteroids (10% chance per frame)
            state = wp.rand_init(seed_arr[0], i * max_tick + t)
            spawn_roll = wp.randi(state) % 10
            if spawn_roll == 0:
                edge = wp.randi(state) % 4
                sp_x = float(0.0)
                sp_y = float(0.0)
                sp_angle = float(0.0)

                if edge == 0:
                    sp_x = 0.0
                    sp_y = wp.randf(state) * arena_size
                    sp_angle = wp.randf(state) * 3.14159265 - 1.57079633
                if edge == 1:
                    sp_x = arena_size
                    sp_y = wp.randf(state) * arena_size
                    sp_angle = wp.randf(state) * 3.14159265 + 1.57079633
                if edge == 2:
                    sp_x = wp.randf(state) * arena_size
                    sp_y = 0.0
                    sp_angle = wp.randf(state) * 3.14159265 + 3.14159265
                if edge == 3:
                    sp_x = wp.randf(state) * arena_size
                    sp_y = arena_size
                    sp_angle = wp.randf(state) * 3.14159265

                adx = wp.cos(sp_angle)
                ady = wp.sin(sp_angle)

                size_roll = wp.randi(state) % 3
                rad = float(0.0)
                if size_roll == 0:
                    rad = 10.0
                if size_roll == 1:
                    rad = 20.0
                if size_roll == 2:
                    rad = 40.0

                a_idx = (a_idx + 1) % MAX_ASTEROIDS
                aidx = i * MAX_ASTEROIDS + a_idx
                asteroid_x[aidx] = sp_x
                asteroid_y[aidx] = sp_y
                asteroid_vx[aidx] = adx
                asteroid_vy[aidx] = ady
                asteroid_radius[aidx] = rad

            # Move asteroids
            for a in range(MAX_ASTEROIDS):
                aidx = i * MAX_ASTEROIDS + a
                if asteroid_radius[aidx] > 0.0:
                    asteroid_x[aidx] = asteroid_x[aidx] + asteroid_vx[aidx] * asteroid_speed
                    asteroid_y[aidx] = asteroid_y[aidx] + asteroid_vy[aidx] * asteroid_speed

            # Particle-asteroid collisions
            for p in range(MAX_PARTICLES):
                pidx = i * MAX_PARTICLES + p
                ppx = particle_x[pidx]
                ppy = particle_y[pidx]
                if ppx == 0.0 and ppy == 0.0:
                    continue

                hit_found = int(0)
                for a in range(MAX_ASTEROIDS):
                    if hit_found == 0:
                        aidx = i * MAX_ASTEROIDS + a
                        ar = asteroid_radius[aidx]
                        if ar > 0.0:
                            ddx = ppx - asteroid_x[aidx]
                            ddy = ppy - asteroid_y[aidx]
                            dist_sq = ddx * ddx + ddy * ddy
                            if dist_sq < ar * ar:
                                # Hit! Clear particle
                                particle_x[pidx] = 0.0
                                particle_y[pidx] = 0.0
                                particle_vx[pidx] = 0.0
                                particle_vy[pidx] = 0.0
                                sc = sc + 1
                                reward = reward + 1.0

                                if ar < 15.0:
                                    # Small asteroid — destroy
                                    asteroid_radius[aidx] = 0.0
                                    asteroid_x[aidx] = 0.0
                                    asteroid_y[aidx] = 0.0
                                    asteroid_vx[aidx] = 0.0
                                    asteroid_vy[aidx] = 0.0
                                else:
                                    # Split: modify existing + create new
                                    new_rad = float(0.0)
                                    if ar > 30.0:
                                        new_rad = 20.0
                                    else:
                                        new_rad = 10.0

                                    orig_angle = wp.atan2(asteroid_vy[aidx], asteroid_vx[aidx])
                                    split_state = wp.rand_init(seed_arr[0], i * max_tick * 2 + t * MAX_ASTEROIDS + a)
                                    off1 = (wp.randf(split_state) - 0.5) * 1.57079633
                                    off2 = (wp.randf(split_state) - 0.5) * 1.57079633

                                    a1 = orig_angle + off1
                                    a2 = orig_angle + off2

                                    save_ax = asteroid_x[aidx]
                                    save_ay = asteroid_y[aidx]

                                    asteroid_vx[aidx] = wp.cos(a1)
                                    asteroid_vy[aidx] = wp.sin(a1)
                                    asteroid_radius[aidx] = new_rad

                                    a_idx = (a_idx + 1) % MAX_ASTEROIDS
                                    nidx = i * MAX_ASTEROIDS + a_idx
                                    asteroid_x[nidx] = save_ax
                                    asteroid_y[nidx] = save_ay
                                    asteroid_vx[nidx] = wp.cos(a2)
                                    asteroid_vy[nidx] = wp.sin(a2)
                                    asteroid_radius[nidx] = new_rad

                                hit_found = 1

            # Player-asteroid collision
            if done == 0:
                for a in range(MAX_ASTEROIDS):
                    if done == 0:
                        aidx = i * MAX_ASTEROIDS + a
                        ar = asteroid_radius[aidx]
                        if ar > 0.0:
                            min_dist = player_radius + ar
                            ddx = px - asteroid_x[aidx]
                            ddy = py - asteroid_y[aidx]
                            if min_dist * min_dist > ddx * ddx + ddy * ddy:
                                reward = -1.0
                                done = 1

            # Wrap player position
            if px < 0.0:
                px = arena_size
            if py < 0.0:
                py = arena_size
            if px > arena_size:
                px = 0.0
            if py > arena_size:
                py = 0.0

            # Max tick check
            if t >= max_tick:
                done = 1

    new_ep_return = episode_return[i] + reward
    rewards[i] = reward
    dones[i] = float(done)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(t)

    if done == 1:
        # Full reset
        px = arena_size / 2.0
        py = arena_size / 2.0
        pvx = 0.0
        pvy = 0.0
        pa = 0.0
        p_idx = 0
        a_idx = 0
        ls = 0
        t = 0
        sc = 0
        new_ep_return = 0.0

        for p in range(MAX_PARTICLES):
            pidx = i * MAX_PARTICLES + p
            particle_x[pidx] = 0.0
            particle_y[pidx] = 0.0
            particle_vx[pidx] = 0.0
            particle_vy[pidx] = 0.0

        for a in range(MAX_ASTEROIDS):
            aidx = i * MAX_ASTEROIDS + a
            asteroid_x[aidx] = 0.0
            asteroid_y[aidx] = 0.0
            asteroid_vx[aidx] = 0.0
            asteroid_vy[aidx] = 0.0
            asteroid_radius[aidx] = 0.0

    # Write back state
    player_x[i] = px
    player_y[i] = py
    player_vx[i] = pvx
    player_vy[i] = pvy
    player_angle[i] = pa
    particle_index[i] = p_idx
    asteroid_index[i] = a_idx
    last_shot[i] = ls
    tick[i] = t
    score[i] = sc
    episode_return[i] = new_ep_return

    # Compute observations with distance-sorted asteroids
    obs[i, 0] = px / arena_size
    obs[i, 1] = py / arena_size
    obs[i, 2] = pvx
    obs[i, 3] = pvy

    # Gather asteroid distances for sorting (bubble sort on local arrays)
    # We store indices and distances in parallel local arrays
    sort_dist = wp.vector(dtype=float, length=MAX_ASTEROIDS)
    sort_idx = wp.vector(dtype=float, length=MAX_ASTEROIDS)
    num_active = int(0)

    for a in range(MAX_ASTEROIDS):
        aidx = i * MAX_ASTEROIDS + a
        ar = asteroid_radius[aidx]
        if ar > 0.0:
            ddx = asteroid_x[aidx] - px
            ddy = asteroid_y[aidx] - py
            sort_dist[num_active] = ddx * ddx + ddy * ddy
            sort_idx[num_active] = float(a)
            num_active = num_active + 1

    # Bubble sort by distance
    for outer in range(MAX_ASTEROIDS):
        for inner in range(MAX_ASTEROIDS - 1):
            if inner < num_active - 1 and outer < num_active - 1:
                if sort_dist[inner] > sort_dist[inner + 1]:
                    tmp_d = sort_dist[inner]
                    sort_dist[inner] = sort_dist[inner + 1]
                    sort_dist[inner + 1] = tmp_d
                    tmp_i = sort_idx[inner]
                    sort_idx[inner] = sort_idx[inner + 1]
                    sort_idx[inner + 1] = tmp_i

    # Write sorted asteroid observations
    for k in range(MAX_ASTEROIDS):
        base = 4 + k * 5
        if k < num_active:
            a = int(sort_idx[k])
            aidx = i * MAX_ASTEROIDS + a
            obs[i, base + 0] = (asteroid_x[aidx] - px) / arena_size
            obs[i, base + 1] = (asteroid_y[aidx] - py) / arena_size
            obs[i, base + 2] = asteroid_vx[aidx]
            obs[i, base + 3] = asteroid_vy[aidx]
            obs[i, base + 4] = asteroid_radius[aidx] / 40.0
        else:
            obs[i, base + 0] = 0.0
            obs[i, base + 1] = 0.0
            obs[i, base + 2] = 0.0
            obs[i, base + 3] = 0.0
            obs[i, base + 4] = 0.0


class AsteroidsEnv:
    """Vectorized Asteroids environment on GPU.

    Args:
        num_envs: Number of parallel environments.
        arena_size: Width and height of the square arena.
        player_radius: Collision radius of the player ship.
        friction: Velocity damping per frame.
        speed: Thrust acceleration magnitude.
        particle_speed: Bullet speed multiplier.
        rotation_speed: Turning rate in radians per frame.
        asteroid_speed: Asteroid movement speed multiplier.
        shoot_delay: Minimum frames between shots.
        max_tick: Maximum frames before episode ends.
        frameskip: Physics sub-steps per action.
        device: Warp device string.
        seed: Random seed.
    """

    OBS_SIZE = OBS_SIZE
    NUM_ACTIONS = 4

    def __init__(
        self,
        num_envs: int = 1024,
        arena_size: float = 500.0,
        player_radius: float = 12.0,
        friction: float = 0.95,
        speed: float = 0.6,
        particle_speed: float = 7.0,
        rotation_speed: float = 0.1,
        asteroid_speed: float = 3.0,
        shoot_delay: int = 18,
        max_tick: int = 3600,
        frameskip: int = 1,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.arena_size = arena_size
        self.player_radius = player_radius
        self.friction = friction
        self.speed = speed
        self.particle_speed = particle_speed
        self.rotation_speed = rotation_speed
        self.asteroid_speed = asteroid_speed
        self.shoot_delay = shoot_delay
        self.max_tick = max_tick
        self.frameskip = frameskip
        self.device = device
        self.seed = seed
        self._step_count = 0

        n = num_envs

        self.player_x = wp.zeros(n, dtype=float, device=device)
        self.player_y = wp.zeros(n, dtype=float, device=device)
        self.player_vx = wp.zeros(n, dtype=float, device=device)
        self.player_vy = wp.zeros(n, dtype=float, device=device)
        self.player_angle = wp.zeros(n, dtype=float, device=device)

        self.particle_x = wp.zeros(n * MAX_PARTICLES, dtype=float, device=device)
        self.particle_y = wp.zeros(n * MAX_PARTICLES, dtype=float, device=device)
        self.particle_vx = wp.zeros(n * MAX_PARTICLES, dtype=float, device=device)
        self.particle_vy = wp.zeros(n * MAX_PARTICLES, dtype=float, device=device)

        self.asteroid_x = wp.zeros(n * MAX_ASTEROIDS, dtype=float, device=device)
        self.asteroid_y = wp.zeros(n * MAX_ASTEROIDS, dtype=float, device=device)
        self.asteroid_vx = wp.zeros(n * MAX_ASTEROIDS, dtype=float, device=device)
        self.asteroid_vy = wp.zeros(n * MAX_ASTEROIDS, dtype=float, device=device)
        self.asteroid_radius = wp.zeros(n * MAX_ASTEROIDS, dtype=float, device=device)

        self.particle_index = wp.zeros(n, dtype=int, device=device)
        self.asteroid_index = wp.zeros(n, dtype=int, device=device)
        self.last_shot = wp.zeros(n, dtype=int, device=device)
        self.tick = wp.zeros(n, dtype=int, device=device)
        self.score = wp.zeros(n, dtype=int, device=device)
        self.episode_return = wp.zeros(n, dtype=float, device=device)

        self.obs = wp.zeros((n, self.OBS_SIZE), dtype=float, device=device)
        self.rewards = wp.zeros(n, dtype=float, device=device)
        self.dones = wp.zeros(n, dtype=float, device=device)
        self.episode_returns = wp.zeros(n, dtype=float, device=device)
        self.episode_lengths = wp.zeros(n, dtype=float, device=device)

        self.reset()

    def reset(self):
        wp.launch(
            asteroids_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.player_x, self.player_y,
                self.player_vx, self.player_vy,
                self.player_angle,
                self.particle_x, self.particle_y,
                self.particle_vx, self.particle_vy,
                self.asteroid_x, self.asteroid_y,
                self.asteroid_vx, self.asteroid_vy,
                self.asteroid_radius,
                self.particle_index, self.asteroid_index,
                self.last_shot, self.tick, self.score,
                self.episode_return, self.obs,
                self.arena_size, self.seed,
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
            asteroids_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.player_x, self.player_y,
                self.player_vx, self.player_vy,
                self.player_angle,
                self.particle_x, self.particle_y,
                self.particle_vx, self.particle_vy,
                self.asteroid_x, self.asteroid_y,
                self.asteroid_vx, self.asteroid_vy,
                self.asteroid_radius,
                self.particle_index, self.asteroid_index,
                self.last_shot, self.tick, self.score,
                self.episode_return,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.arena_size, self.player_radius,
                self.friction, self.speed,
                self.particle_speed, self.rotation_speed,
                self.asteroid_speed, self.shoot_delay,
                self.max_tick, self.frameskip, seed_arr,
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
