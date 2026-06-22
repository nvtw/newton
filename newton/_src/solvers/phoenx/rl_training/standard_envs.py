# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class ConfigEnvPendulumV1Warp:
    """Configuration for the Warp Pendulum-v1 benchmark environment."""

    world_count: int = 256
    max_episode_steps: int = 200
    dt: float = 0.05
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
    max_speed: float = 8.0
    max_torque: float = 2.0
    seed: int = 0


@wp.func
def _angle_normalize(x: wp.float32) -> wp.float32:
    two_pi = wp.float32(2.0) * wp.pi
    return wp.mod(x + wp.pi, two_pi) - wp.pi


@wp.kernel(enable_backward=False)
def pendulum_reset_kernel(
    theta: wp.array[wp.float32],
    theta_dot: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    obs: wp.array2d[wp.float32],
    seed: wp.int32,
):
    env = wp.tid()
    rng_theta = wp.rand_init(seed, env * wp.int32(2))
    rng_theta_dot = wp.rand_init(seed, env * wp.int32(2) + wp.int32(1))
    th = wp.randf(rng_theta, -wp.pi, wp.pi)
    thdot = wp.randf(rng_theta_dot, wp.float32(-1.0), wp.float32(1.0))
    theta[env] = th
    theta_dot[env] = thdot
    episode_steps[env] = wp.int32(0)
    obs[env, 0] = wp.cos(th)
    obs[env, 1] = wp.sin(th)
    obs[env, 2] = thdot


@wp.kernel(enable_backward=False)
def pendulum_observe_kernel(
    theta: wp.array[wp.float32],
    theta_dot: wp.array[wp.float32],
    obs: wp.array2d[wp.float32],
):
    env = wp.tid()
    th = theta[env]
    obs[env, 0] = wp.cos(th)
    obs[env, 1] = wp.sin(th)
    obs[env, 2] = theta_dot[env]


@wp.kernel(enable_backward=False)
def pendulum_step_kernel(
    actions: wp.array2d[wp.float32],
    theta: wp.array[wp.float32],
    theta_dot: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    dt: wp.float32,
    gravity: wp.float32,
    mass: wp.float32,
    length: wp.float32,
    max_speed: wp.float32,
    max_torque: wp.float32,
    reset_seed: wp.int32,
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
):
    env = wp.tid()
    th = theta[env]
    thdot = theta_dot[env]
    torque = wp.clamp(actions[env, 0] * max_torque, -max_torque, max_torque)
    normalized = _angle_normalize(th)
    cost = normalized * normalized + wp.float32(0.1) * thdot * thdot + wp.float32(0.001) * torque * torque

    new_thdot = (
        thdot
        + (
            wp.float32(3.0) * gravity / (wp.float32(2.0) * length) * wp.sin(th)
            + wp.float32(3.0) / (mass * length * length) * torque
        )
        * dt
    )
    new_thdot = wp.clamp(new_thdot, -max_speed, max_speed)
    new_th = th + new_thdot * dt

    step = episode_steps[env] + wp.int32(1)
    done = wp.float32(0.0)
    if max_episode_steps > wp.int32(0) and step >= max_episode_steps:
        done = wp.float32(1.0)
        rng_theta = wp.rand_init(reset_seed + step, env * wp.int32(2))
        rng_theta_dot = wp.rand_init(reset_seed + step, env * wp.int32(2) + wp.int32(1))
        new_th = wp.randf(rng_theta, -wp.pi, wp.pi)
        new_thdot = wp.randf(rng_theta_dot, wp.float32(-1.0), wp.float32(1.0))
        step = wp.int32(0)

    theta[env] = new_th
    theta_dot[env] = new_thdot
    episode_steps[env] = step
    obs[env, 0] = wp.cos(new_th)
    obs[env, 1] = wp.sin(new_th)
    obs[env, 2] = new_thdot
    rewards[env] = -cost
    dones[env] = done
    successes[env] = wp.where(cost < wp.float32(1.0), wp.float32(1.0), wp.float32(0.0))


class EnvPendulumV1Warp:
    """Vectorized Warp implementation of Gymnasium's Pendulum-v1 equations."""

    obs_dim = 3
    action_dim = 1

    def __init__(self, config: ConfigEnvPendulumV1Warp | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvPendulumV1Warp()
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if int(self.config.max_episode_steps) < 0:
            raise ValueError("max_episode_steps must be non-negative")

        self.theta = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.theta_dot = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.episode_steps = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros((self.world_count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._reset_seed = int(self.config.seed)
        self.reset(seed=self._reset_seed)

    def reset(self, *, seed: int | None = None) -> wp.array2d[wp.float32]:
        if seed is not None:
            self._reset_seed = int(seed)
        wp.launch(
            pendulum_reset_kernel,
            dim=self.world_count,
            inputs=[self.theta, self.theta_dot, self.episode_steps, self.obs, int(self._reset_seed)],
            device=self.device,
        )
        self.rewards.zero_()
        self.dones.zero_()
        self.step_successes.zero_()
        return self.obs

    def observe(self) -> wp.array2d[wp.float32]:
        wp.launch(
            pendulum_observe_kernel,
            dim=self.world_count,
            inputs=[self.theta, self.theta_dot],
            outputs=[self.obs],
            device=self.device,
        )
        return self.obs

    def step(
        self, actions: wp.array2d[wp.float32]
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array[wp.float32]]:
        wp.launch(
            pendulum_step_kernel,
            dim=self.world_count,
            inputs=[
                actions,
                self.theta,
                self.theta_dot,
                self.episode_steps,
                int(self.config.max_episode_steps),
                float(self.config.dt),
                float(self.config.gravity),
                float(self.config.mass),
                float(self.config.length),
                float(self.config.max_speed),
                float(self.config.max_torque),
                int(self._reset_seed),
            ],
            outputs=[self.obs, self.rewards, self.dones, self.step_successes],
            device=self.device,
        )
        return self.obs, self.rewards, self.dones


def pendulum_v1_numpy_step(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    actions: np.ndarray,
    config: ConfigEnvPendulumV1Warp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference Pendulum-v1 transition matching :class:`EnvPendulumV1Warp`."""

    th = np.asarray(theta, dtype=np.float32)
    thdot = np.asarray(theta_dot, dtype=np.float32)
    u = np.clip(
        np.asarray(actions, dtype=np.float32).reshape(th.shape) * config.max_torque,
        -config.max_torque,
        config.max_torque,
    )
    normalized = ((th + np.pi) % (2.0 * np.pi)) - np.pi
    costs = normalized * normalized + 0.1 * thdot * thdot + 0.001 * u * u
    new_thdot = (
        thdot
        + (
            3.0 * config.gravity / (2.0 * config.length) * np.sin(th)
            + 3.0 / (config.mass * config.length * config.length) * u
        )
        * config.dt
    )
    new_thdot = np.clip(new_thdot, -config.max_speed, config.max_speed)
    new_th = th + new_thdot * config.dt
    obs = np.stack((np.cos(new_th), np.sin(new_th), new_thdot), axis=-1).astype(np.float32)
    return new_th.astype(np.float32), new_thdot.astype(np.float32), (-costs).astype(np.float32), obs
