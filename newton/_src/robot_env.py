# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Base class for vectorized robot training environments.

Subclass :class:`RobotEnv` and override the abstract methods to train
any articulated robot with :class:`~newton._src.ppo.PPOTrainer`.

Minimal example::

    class MyRobotEnv(RobotEnv):
        obs_dim = 6
        act_dim = 2

        def build_robot(self, builder):
            builder.add_urdf("robot.urdf", floating=True)
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 200
                builder.joint_target_kd[i] = 10

        def compute_obs(self):
            wp.launch(my_obs_kernel, dim=self.num_envs, inputs=[self.state.joint_q, ...], device=self.device)

        def compute_reward(self):
            wp.launch(
                my_reward_kernel,
                dim=self.num_envs,
                inputs=[self.state.joint_q, ..., self.rewards, self.dones],
                device=self.device,
            )

        def apply_actions(self, actions):
            wp.launch(
                my_action_kernel, dim=self.num_envs, inputs=[actions, self.control.joint_target_pos], device=self.device
            )


    env = MyRobotEnv(num_envs=1024)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import warp as wp

import newton
from newton._src.ppo import _increment_counter_kernel


class RobotEnv(ABC):
    """Vectorized robot environment base class.

    Handles model replication, solver setup, CUDA graph capture for the
    physics substep loop, episode tracking, and reset.  Subclasses provide
    the robot definition, observations, rewards, and action mapping.

    After construction the following attributes are available for use
    inside the overridden methods:

    * ``self.state`` -- current :class:`~newton.State` (aliased to ``state_0``)
    * ``self.model`` -- the finalized :class:`~newton.Model`
    * ``self.control`` -- :class:`~newton.Control` (write ``joint_target_pos``)
    * ``self.obs`` -- ``(num_envs, obs_dim)`` observation buffer
    * ``self.rewards`` -- ``(num_envs,)`` reward buffer
    * ``self.dones`` -- ``(num_envs,)`` termination buffer
    * ``self.episode_lengths`` -- ``(num_envs,)`` int counter
    * ``self.rng_counter`` -- single-element ``wp.array[int]`` for RNG
    * ``self.initial_joint_q`` -- ``(q_stride,)`` single-env initial state
    * ``self.initial_joint_qd`` -- ``(qd_stride,)`` single-env initial state
    * ``self.q_stride``, ``self.qd_stride`` -- joint DOF counts per world

    Args:
        num_envs: Number of parallel environments.
        device: Warp device string.
        seed: RNG seed.
    """

    # --- Subclass must set these ---
    obs_dim: int
    act_dim: int

    # --- Subclass may override these ---
    sim_substeps: int = 4
    sim_dt: float = 0.005
    max_episode_length: int = 1000
    use_collisions: bool = True

    def __init__(self, num_envs: int, device: str | None = None, seed: int = 123):
        self.num_envs = num_envs
        self.device = wp.get_device(device)
        self.sim_time = 0.0
        self.frame_dt = self.sim_dt * self.sim_substeps

        # -- Build single robot --
        art = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(art)
        self.build_robot(art)

        # -- Derive strides from the single-robot builder --
        self.q_stride = len(art.joint_q)
        self.qd_stride = len(art.joint_qd)

        # -- Replicate --
        builder = newton.ModelBuilder()
        builder.replicate(art, num_envs)
        if self.use_collisions:
            builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        # -- Solver --
        solver_kwargs: dict[str, Any] = {}
        if self.use_collisions:
            solver_kwargs.update(
                use_mujoco_contacts=False,
                solver="newton",
                ls_parallel=False,
                ls_iterations=50,
                njmax=50 * num_envs,
                nconmax=100 * num_envs,
            )
        self.solver = newton.solvers.SolverMuJoCo(self.model, **solver_kwargs)

        # -- State / control --
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state = self.state_0  # alias for convenience
        self.control = self.model.control()
        self.contacts = self.model.contacts() if self.use_collisions else None

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # -- Initial state (single world, for resets) --
        full_q = self.state_0.joint_q.numpy()
        full_qd = self.state_0.joint_qd.numpy()
        self.initial_joint_q = wp.array(
            full_q[: self.q_stride].astype(np.float32), dtype=wp.float32, device=self.device
        )
        self.initial_joint_qd = wp.array(
            full_qd[: self.qd_stride].astype(np.float32), dtype=wp.float32, device=self.device
        )

        # -- Pre-allocated buffers --
        d = self.device
        self.obs = wp.zeros((num_envs, self.obs_dim), dtype=wp.float32, device=d)
        self.rewards = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.dones = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.episode_lengths = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self.rng_counter = wp.array([seed], dtype=wp.int32, device=d)

        # -- CUDA graph capture for physics --
        self._graph = None
        if self.device.is_cuda:
            self.control.joint_target_pos = wp.zeros(num_envs * self.qd_stride, dtype=wp.float32, device=d)
            with wp.ScopedCapture() as capture:
                self._simulate()
            self._graph = capture.graph

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.state = self.state_0

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> wp.array:
        q_np = self.initial_joint_q.numpy()
        qd_np = self.initial_joint_qd.numpy()
        wp.copy(
            self.state_0.joint_q,
            wp.array(np.tile(q_np, self.num_envs).astype(np.float32), dtype=wp.float32, device=self.device),
        )
        wp.copy(
            self.state_0.joint_qd,
            wp.array(np.tile(qd_np, self.num_envs).astype(np.float32), dtype=wp.float32, device=self.device),
        )
        self.state = self.state_0
        self.episode_lengths.zero_()
        self.sim_time = 0.0

        # Mark all envs as done so reset_done_envs applies perturbation
        wp.copy(self.dones, wp.ones(self.num_envs, dtype=wp.float32, device=self.device))
        self.on_reset()
        self.reset_done_envs()
        self.dones.zero_()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.compute_obs()
        return self.obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        self.apply_actions(actions)

        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        wp.launch(_increment_episode_kernel, dim=self.num_envs, inputs=[self.episode_lengths], device=self.device)

        self.compute_reward()

        self.reset_done_envs()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state = self.state_0

        wp.launch(_increment_counter_kernel, dim=1, inputs=[self.rng_counter], device=self.device)

        self.compute_obs()

        self.sim_time += self.frame_dt
        return self.obs, self.rewards, self.dones

    # ------------------------------------------------------------------
    # Abstract methods -- subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def build_robot(self, builder: newton.ModelBuilder) -> None:
        """Configure a single-robot articulation on *builder*.

        Set joint positions, PD gains, shape properties, etc.
        """

    @abstractmethod
    def compute_obs(self) -> None:
        """Write observations into ``self.obs``.

        Typically a ``wp.launch`` reading from ``self.state.joint_q``,
        ``self.state.joint_qd``, ``self.state.body_q``, etc.
        """

    @abstractmethod
    def compute_reward(self) -> None:
        """Write rewards into ``self.rewards`` and termination into ``self.dones``."""

    @abstractmethod
    def apply_actions(self, actions: wp.array) -> None:
        """Map policy actions to ``self.control.joint_target_pos`` (or forces)."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def reset_done_envs(self) -> None:
        """Reset environments where ``self.dones > 0.5``.

        Default: copy ``initial_joint_q/qd`` for done envs, zero episode counter.
        Override for custom reset logic (e.g. domain randomization).
        """
        wp.launch(
            _default_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self.dones,
                self.initial_joint_q,
                self.initial_joint_qd,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_lengths,
                self.q_stride,
                self.qd_stride,
            ],
            device=self.device,
        )

    def on_reset(self) -> None:  # noqa: B027
        """Called once during :meth:`reset` after state is initialized.

        Override to randomize commands, set initial perturbations, etc.
        """


# ---------------------------------------------------------------------------
# Default kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _increment_episode_kernel(episode_lengths: wp.array[int]):
    env = wp.tid()
    episode_lengths[env] = episode_lengths[env] + 1


@wp.kernel
def _default_reset_kernel(
    dones: wp.array[float],
    initial_q: wp.array[float],
    initial_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    episode_lengths: wp.array[int],
    q_stride: int,
    qd_stride: int,
):
    env = wp.tid()
    if dones[env] > 0.5:
        q_off = env * q_stride
        qd_off = env * qd_stride
        for i in range(q_stride):
            joint_q[q_off + i] = initial_q[i]
        for i in range(qd_stride):
            joint_qd[qd_off + i] = initial_qd[i]
        episode_lengths[env] = 0
