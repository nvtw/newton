# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vectorized CartPole environment running entirely on GPU via Warp kernels.

Port of PufferLib's ``ocean/cartpole/cartpole.h`` — identical physics,
reward, and termination conditions.  All N environments step in parallel
inside a single ``wp.launch``.
"""

from __future__ import annotations

import warp as wp
import numpy as np

wp.set_module_options({"enable_backward": False})

# Physics constants matching PufferLib's cartpole.h
X_THRESHOLD = wp.constant(2.4)
THETA_THRESHOLD = wp.constant(0.20943951)  # 12 degrees in radians
MAX_STEPS = wp.constant(200)


@wp.kernel
def cartpole_reset_kernel(
    x: wp.array(dtype=float, ndim=1),
    x_dot: wp.array(dtype=float, ndim=1),
    theta: wp.array(dtype=float, ndim=1),
    theta_dot: wp.array(dtype=float, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    seed: int,
):
    """Reset all environments to random initial states."""
    i = wp.tid()
    state = wp.rand_init(seed, i)
    x[i] = wp.randf(state, -0.04, 0.04)
    x_dot[i] = wp.randf(state, -0.04, 0.04)
    theta[i] = wp.randf(state, -0.04, 0.04)
    theta_dot[i] = wp.randf(state, -0.04, 0.04)
    tick[i] = 0
    episode_return[i] = 0.0


@wp.kernel
def cartpole_step_kernel(
    # State (read/write)
    x: wp.array(dtype=float, ndim=1),
    x_dot: wp.array(dtype=float, ndim=1),
    theta: wp.array(dtype=float, ndim=1),
    theta_dot: wp.array(dtype=float, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    # Inputs
    actions: wp.array(dtype=float, ndim=1),
    # Physics params
    cart_mass: float,
    pole_mass: float,
    pole_length: float,
    gravity: float,
    force_mag: float,
    tau: float,
    # Outputs
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    # Reset RNG
    reset_seed: int,
):
    """Step all CartPole environments in parallel.

    Discrete action space: action > 0.5 -> push right, else push left.
    """
    i = wp.tid()

    # Read action (discrete: 0 = left, 1 = right)
    a = actions[i]
    if a > 0.5:
        force = force_mag
    else:
        force = -force_mag

    # Physics (Euler integration, matching PufferLib's cartpole.h exactly)
    cos_th = wp.cos(theta[i])
    sin_th = wp.sin(theta[i])

    total_mass = cart_mass + pole_mass
    polemass_length = total_mass + pole_mass
    temp = (force + polemass_length * theta_dot[i] * theta_dot[i] * sin_th) / total_mass
    theta_acc = (gravity * sin_th - cos_th * temp) / (
        pole_length * (4.0 / 3.0 - total_mass * cos_th * cos_th / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos_th / total_mass

    new_x = x[i] + tau * x_dot[i]
    new_x_dot = x_dot[i] + tau * x_acc
    new_theta = theta[i] + tau * theta_dot[i]
    new_theta_dot = theta_dot[i] + tau * theta_acc

    new_tick = tick[i] + 1

    # Termination
    terminated = 0
    if new_x < -X_THRESHOLD or new_x > X_THRESHOLD:
        terminated = 1
    if new_theta < -THETA_THRESHOLD or new_theta > THETA_THRESHOLD:
        terminated = 1

    truncated = 0
    if new_tick >= MAX_STEPS:
        truncated = 1

    done = 0
    if terminated == 1 or truncated == 1:
        done = 1

    # Reward: +1 per step while alive, 0 on done
    reward = 1.0
    if done == 1:
        reward = 0.0

    new_ep_return = episode_return[i] + reward

    # Write outputs before potential reset
    rewards[i] = reward
    dones[i] = float(terminated)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(new_tick)

    # Auto-reset on done
    if done == 1:
        state = wp.rand_init(reset_seed, i + new_tick)
        new_x = wp.randf(state, -0.04, 0.04)
        new_x_dot = wp.randf(state, -0.04, 0.04)
        new_theta = wp.randf(state, -0.04, 0.04)
        new_theta_dot = wp.randf(state, -0.04, 0.04)
        new_tick = 0
        new_ep_return = 0.0

    # Update state
    x[i] = new_x
    x_dot[i] = new_x_dot
    theta[i] = new_theta
    theta_dot[i] = new_theta_dot
    tick[i] = new_tick
    episode_return[i] = new_ep_return

    # Write observations (post-reset if done)
    obs[i, 0] = new_x
    obs[i, 1] = new_x_dot
    obs[i, 2] = new_theta
    obs[i, 3] = new_theta_dot


@wp.kernel
def cartpole_step_dev_kernel(
    x: wp.array(dtype=float, ndim=1),
    x_dot: wp.array(dtype=float, ndim=1),
    theta: wp.array(dtype=float, ndim=1),
    theta_dot: wp.array(dtype=float, ndim=1),
    tick: wp.array(dtype=int, ndim=1),
    episode_return: wp.array(dtype=float, ndim=1),
    actions: wp.array(dtype=float, ndim=1),
    cart_mass: float,
    pole_mass: float,
    pole_length: float,
    gravity: float,
    force_mag: float,
    tau: float,
    obs: wp.array2d(dtype=float),
    rewards: wp.array(dtype=float, ndim=1),
    dones: wp.array(dtype=float, ndim=1),
    episode_returns: wp.array(dtype=float, ndim=1),
    episode_lengths: wp.array(dtype=float, ndim=1),
    reset_seed_arr: wp.array(dtype=int, ndim=1),
):
    """Like :func:`cartpole_step_kernel` but reads ``reset_seed`` from a device array.

    This makes the kernel graph-capture compatible — the seed array is
    updated between graph replays by a separate increment kernel.
    """
    i = wp.tid()

    a = actions[i]
    if a > 0.5:
        force = force_mag
    else:
        force = -force_mag

    cos_th = wp.cos(theta[i])
    sin_th = wp.sin(theta[i])

    total_mass = cart_mass + pole_mass
    polemass_length = total_mass + pole_mass
    temp = (force + polemass_length * theta_dot[i] * theta_dot[i] * sin_th) / total_mass
    theta_acc = (gravity * sin_th - cos_th * temp) / (
        pole_length * (4.0 / 3.0 - total_mass * cos_th * cos_th / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos_th / total_mass

    new_x = x[i] + tau * x_dot[i]
    new_x_dot = x_dot[i] + tau * x_acc
    new_theta = theta[i] + tau * theta_dot[i]
    new_theta_dot = theta_dot[i] + tau * theta_acc

    new_tick = tick[i] + 1

    terminated = 0
    if new_x < -X_THRESHOLD or new_x > X_THRESHOLD:
        terminated = 1
    if new_theta < -THETA_THRESHOLD or new_theta > THETA_THRESHOLD:
        terminated = 1

    truncated = 0
    if new_tick >= MAX_STEPS:
        truncated = 1

    done = 0
    if terminated == 1 or truncated == 1:
        done = 1

    reward = 1.0
    if done == 1:
        reward = 0.0

    new_ep_return = episode_return[i] + reward

    rewards[i] = reward
    dones[i] = float(terminated)
    episode_returns[i] = new_ep_return
    episode_lengths[i] = float(new_tick)

    if done == 1:
        state = wp.rand_init(reset_seed_arr[0], i + new_tick)
        new_x = wp.randf(state, -0.04, 0.04)
        new_x_dot = wp.randf(state, -0.04, 0.04)
        new_theta = wp.randf(state, -0.04, 0.04)
        new_theta_dot = wp.randf(state, -0.04, 0.04)
        new_tick = 0
        new_ep_return = 0.0

    x[i] = new_x
    x_dot[i] = new_x_dot
    theta[i] = new_theta
    theta_dot[i] = new_theta_dot
    tick[i] = new_tick
    episode_return[i] = new_ep_return

    obs[i, 0] = new_x
    obs[i, 1] = new_x_dot
    obs[i, 2] = new_theta
    obs[i, 3] = new_theta_dot


class CartPoleEnv:
    """Vectorized CartPole environment on GPU.

    All ``num_envs`` environments run in parallel inside a single Warp kernel.

    Args:
        num_envs: Number of parallel environments.
        cart_mass: Mass of the cart (default 1.0).
        pole_mass: Mass of the pole (default 0.1).
        pole_length: Half-length of the pole (default 0.5).
        gravity: Gravitational acceleration (default 9.8).
        force_mag: Magnitude of the applied force (default 10.0).
        tau: Time step for Euler integration (default 0.02).
        device: Warp device string.
        seed: Random seed.
    """

    OBS_SIZE = 4
    NUM_ACTIONS = 2  # discrete: left or right
    MAX_EPISODE_STEPS = 200

    def __init__(
        self,
        num_envs: int = 4096,
        cart_mass: float = 1.0,
        pole_mass: float = 0.1,
        pole_length: float = 0.5,
        gravity: float = 9.8,
        force_mag: float = 10.0,
        tau: float = 0.02,
        device: str = "cuda:0",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.gravity = gravity
        self.force_mag = force_mag
        self.tau = tau
        self.device = device
        self.seed = seed
        self._step_count = 0

        # State arrays
        self.x = wp.zeros(num_envs, dtype=float, device=device)
        self.x_dot = wp.zeros(num_envs, dtype=float, device=device)
        self.theta = wp.zeros(num_envs, dtype=float, device=device)
        self.theta_dot = wp.zeros(num_envs, dtype=float, device=device)
        self.tick = wp.zeros(num_envs, dtype=int, device=device)
        self.episode_return = wp.zeros(num_envs, dtype=float, device=device)

        # Output arrays
        self.obs = wp.zeros((num_envs, self.OBS_SIZE), dtype=float, device=device)
        self.rewards = wp.zeros(num_envs, dtype=float, device=device)
        self.dones = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_returns = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_lengths = wp.zeros(num_envs, dtype=float, device=device)

        self.reset()

    def reset(self):
        """Reset all environments."""
        wp.launch(
            cartpole_reset_kernel,
            dim=self.num_envs,
            inputs=[self.x, self.x_dot, self.theta, self.theta_dot,
                    self.tick, self.episode_return, self.seed],
            device=self.device,
        )
        wp.launch(
            _write_obs_kernel,
            dim=self.num_envs,
            inputs=[self.x, self.x_dot, self.theta, self.theta_dot, self.obs],
            device=self.device,
        )
        return self.obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Step all environments.

        Args:
            actions: (num_envs,) float array. For discrete: 0.0 = left, 1.0 = right.

        Returns:
            (obs, rewards, dones) — all Warp arrays on the same device.
        """
        self._step_count += 1
        wp.launch(
            cartpole_step_kernel,
            dim=self.num_envs,
            inputs=[
                self.x, self.x_dot, self.theta, self.theta_dot,
                self.tick, self.episode_return,
                actions,
                self.cart_mass, self.pole_mass, self.pole_length,
                self.gravity, self.force_mag, self.tau,
                self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                self.seed + self._step_count,
            ],
            device=self.device,
        )
        return self.obs, self.rewards, self.dones

    def step_graphed(self, actions: wp.array, seed_arr: wp.array):
        """Step using a device-side seed array (graph-capture compatible).

        Unlike :meth:`step`, this does **not** touch ``self._step_count``
        and reads the reset seed from ``seed_arr[0]`` on the device.
        """
        wp.launch(
            cartpole_step_dev_kernel,
            dim=self.num_envs,
            inputs=[
                self.x, self.x_dot, self.theta, self.theta_dot,
                self.tick, self.episode_return,
                actions,
                self.cart_mass, self.pole_mass, self.pole_length,
                self.gravity, self.force_mag, self.tau,
                self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                seed_arr,
            ],
            device=self.device,
        )

    def get_episode_stats(self) -> dict:
        """Get mean episode return and length from recently completed episodes."""
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


@wp.kernel
def _write_obs_kernel(
    x: wp.array(dtype=float, ndim=1),
    x_dot: wp.array(dtype=float, ndim=1),
    theta: wp.array(dtype=float, ndim=1),
    theta_dot: wp.array(dtype=float, ndim=1),
    obs: wp.array2d(dtype=float),
):
    i = wp.tid()
    obs[i, 0] = x[i]
    obs[i, 1] = x_dot[i]
    obs[i, 2] = theta[i]
    obs[i, 3] = theta_dot[i]
