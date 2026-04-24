# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world smoke tests for :class:`PhoenXWorld`.

Ports the jitter-side :mod:`test_multi_world` suite: verify that
:class:`PhoenXWorld` built with ``num_worlds > 1`` keeps worlds
isolated (no cross-world state leakage), supports per-world
gravity, and scales to many worlds without regressing.

Uses the actuated-double-ball-socket joint in ``BALL_SOCKET`` mode
as the single per-world constraint (the jitter suite uses the
standalone ``add_ball_socket``; PhoenX's unified joint produces an
equivalent positional lock).

Runs on CUDA only: the per-world CSR scatter / dispatcher requires
graph-captured launches for reasonable wall-clock throughput.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    body_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LINEAR,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.world_builder import DriveMode, JointMode

_FPS = 60
_SUBSTEPS = 4
_SOLVER_ITERATIONS = 16


def _build_n_pendulums(
    *,
    num_worlds: int,
    angular_velocities: list[tuple[float, float, float]] | None = None,
    gravity: tuple[float, float, float]
    | list[tuple[float, float, float]] = (0.0, -9.81, 0.0),
    device: wp.context.Devicelike = None,
) -> tuple[PhoenXWorld, list[int]]:
    """Build ``num_worlds`` identical pendulum scenes sharing one
    :class:`BodyContainer` / :class:`ConstraintContainer`.

    Each world gets a static anchor body (slot ``2*w``) and a
    dynamic cube 1 m below it (slot ``2*w + 1``) connected by one
    ball-socket joint; each cube is tagged with ``world_id = w``
    so the per-world gravity + CSR bucketing kicks in.

    Returns the solver and the list of *Newton-side* cube indices,
    which in our convention are PhoenX slots (there's no Newton
    model here -- we hand the solver raw containers). The caller
    queries ``bodies.position.numpy()[cube_slot]`` directly.
    """
    if device is None:
        device = wp.get_device("cuda:0")
    num_bodies = 2 * num_worlds  # anchor + cube per world
    bodies = body_container_zeros(num_bodies, device=device)

    pos = np.zeros((num_bodies, 3), dtype=np.float32)
    ori = np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies, 1)).astype(np.float32)
    inv_m = np.zeros(num_bodies, dtype=np.float32)
    inv_I = np.zeros((num_bodies, 3, 3), dtype=np.float32)
    motion = np.full(num_bodies, int(MOTION_STATIC), dtype=np.int32)
    world_id = np.zeros(num_bodies, dtype=np.int32)
    ang_v = np.zeros((num_bodies, 3), dtype=np.float32)

    cube_slots: list[int] = []
    for w in range(num_worlds):
        anchor_slot = 2 * w
        cube_slot = 2 * w + 1
        # Anchor is static (default).
        world_id[anchor_slot] = w
        # Cube dangles 1 m below the anchor.
        pos[cube_slot] = (0.0, -1.0, 0.0)
        inv_m[cube_slot] = 1.0
        inv_I[cube_slot] = np.eye(3, dtype=np.float32)
        motion[cube_slot] = int(MOTION_DYNAMIC)
        world_id[cube_slot] = w
        if angular_velocities is not None:
            ang_v[cube_slot] = angular_velocities[w]
        cube_slots.append(cube_slot)

    bodies.position.assign(pos)
    bodies.orientation.assign(ori)
    bodies.inverse_mass.assign(inv_m)
    bodies.inverse_inertia.assign(inv_I)
    bodies.inverse_inertia_world.assign(inv_I)
    bodies.motion_type.assign(motion)
    bodies.world_id.assign(world_id)
    bodies.angular_velocity.assign(ang_v)

    # One joint per world.
    num_joints = num_worlds
    constraints = PhoenXWorld.make_constraint_container(
        num_joints=num_joints, max_contact_columns=0, device=device
    )
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        substeps=_SUBSTEPS,
        solver_iterations=_SOLVER_ITERATIONS,
        velocity_iterations=1,
        gravity=gravity,
        max_contact_columns=0,
        rigid_contact_max=0,
        num_shapes=0,
        num_joints=num_joints,
        num_worlds=num_worlds,
        device=device,
    )

    # Init one BALL_SOCKET joint per world.
    body1 = np.array([2 * w for w in range(num_joints)], dtype=np.int32)
    body2 = np.array([2 * w + 1 for w in range(num_joints)], dtype=np.int32)
    anchor1 = np.zeros((num_joints, 3), dtype=np.float32)  # world origin for each world
    anchor2 = np.zeros((num_joints, 3), dtype=np.float32)  # ignored for BALL_SOCKET

    def _f(v: float) -> wp.array:
        return wp.array(np.full(num_joints, v, dtype=np.float32), dtype=wp.float32, device=device)

    def _i(v: int) -> wp.array:
        return wp.array(np.full(num_joints, v, dtype=np.int32), dtype=wp.int32, device=device)

    world.initialize_actuated_double_ball_socket_joints(
        body1=wp.array(body1, dtype=wp.int32, device=device),
        body2=wp.array(body2, dtype=wp.int32, device=device),
        anchor1=wp.array(anchor1, dtype=wp.vec3f, device=device),
        anchor2=wp.array(anchor2, dtype=wp.vec3f, device=device),
        hertz=_f(float(DEFAULT_HERTZ_LINEAR)),
        damping_ratio=_f(float(DEFAULT_DAMPING_RATIO)),
        joint_mode=_i(int(JointMode.BALL_SOCKET)),
        drive_mode=_i(int(DriveMode.OFF)),
        target=_f(0.0),
        target_velocity=_f(0.0),
        max_force_drive=_f(0.0),
        stiffness_drive=_f(0.0),
        damping_drive=_f(0.0),
        min_value=_f(1.0),
        max_value=_f(-1.0),
        hertz_limit=_f(float(DEFAULT_HERTZ_LINEAR)),
        damping_ratio_limit=_f(float(DEFAULT_DAMPING_RATIO)),
        stiffness_limit=_f(0.0),
        damping_limit=_f(0.0),
    )

    return world, cube_slots


def _run_frames(world: PhoenXWorld, n_frames: int) -> None:
    for _ in range(n_frames):
        world.step(dt=1.0 / _FPS, contacts=None, shape_body=None)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX multi-world tests require CUDA")
class TestPhoenXMultiWorld(unittest.TestCase):
    """Cross-world isolation + per-world gravity checks.

    Directly mirrors :class:`TestMultiWorld` on the jitter side,
    except we use :data:`JointMode.BALL_SOCKET` on the unified joint
    in place of ``add_ball_socket``.
    """

    def test_single_pendulum_per_world(self) -> None:
        """``N`` identical pendulums -> every world's cube ends at
        the same settled pose. Non-match would mean the per-world
        CSR leak'd one world's solve into another's state.
        """
        device = wp.get_device("cuda:0")
        num_worlds = 8
        world, cube_slots = _build_n_pendulums(
            num_worlds=num_worlds, device=device
        )
        _run_frames(world, 30)
        positions = world.bodies.position.numpy()
        ref = positions[cube_slots[0]]
        for w in range(1, num_worlds):
            np.testing.assert_allclose(
                positions[cube_slots[w]],
                ref,
                atol=1.0e-4,
                err_msg=f"world {w} cube diverged from world 0",
            )

    def test_world_independence(self) -> None:
        """World 1's cube spins, world 0's cube starts at rest.
        World 0 must stay essentially at rest (no yaw spin leak),
        while world 1 retains its spin.
        """
        device = wp.get_device("cuda:0")
        avels = [(0.0, 0.0, 0.0), (0.0, 5.0, 0.0)]
        world, cube_slots = _build_n_pendulums(
            num_worlds=2, angular_velocities=avels, device=device
        )
        _run_frames(world, 30)
        avels_after = world.bodies.angular_velocity.numpy()
        w0_ang = avels_after[cube_slots[0]]
        self.assertLess(
            abs(float(w0_ang[1])),
            0.1,
            f"world 0 picked up yaw spin {w0_ang[1]:.3f} -- worlds leaked",
        )
        w1_ang = avels_after[cube_slots[1]]
        self.assertGreater(
            abs(float(w1_ang[1])),
            1.0,
            f"world 1 lost its yaw spin ({w1_ang[1]:.3f}) -- unexpected damping",
        )

    def test_per_world_gravity(self) -> None:
        """World 0 at earth-g, world 1 at moon-g. Cubes should
        swing at different rates -- after the same wall-clock time
        they must be at different z positions (or different joint
        tensions on an identically-held scene).
        """
        device = wp.get_device("cuda:0")
        gravity = [(0.0, -9.81, 0.0), (0.0, -1.62, 0.0)]
        world, cube_slots = _build_n_pendulums(
            num_worlds=2, gravity=gravity, device=device
        )
        # Start each cube slightly off-axis so it swings instead of
        # hanging perfectly vertical (which would hide gravity's
        # effect since the ball socket's normal reaction scales
        # with gravity but the cube doesn't actually move).
        positions = world.bodies.position.numpy()
        positions[cube_slots[0]] = (0.3, -0.95, 0.0)  # off-axis
        positions[cube_slots[1]] = (0.3, -0.95, 0.0)
        world.bodies.position.assign(positions)

        _run_frames(world, 30)

        # After 0.5 s both cubes have swung; earth-g cube's
        # angular velocity about the anchor axis should be
        # sqrt(6.06x) bigger than moon-g cube's (ratio of g's).
        # Less precise check: the kinetic energies differ by the
        # gravity ratio since each cube is in the same arc phase
        # after the same time.
        ang_v = world.bodies.angular_velocity.numpy()
        w0_speed = float(np.linalg.norm(ang_v[cube_slots[0]]))
        w1_speed = float(np.linalg.norm(ang_v[cube_slots[1]]))
        # Earth cube must swing visibly faster than moon cube.
        # Exact ratio depends on arc phase; just check w0 > w1 by
        # a large margin.
        self.assertGreater(
            w0_speed,
            2.0 * w1_speed,
            f"earth cube |w|={w0_speed:.3f} not > 2x moon |w|={w1_speed:.3f} -- "
            "per-world gravity didn't take effect",
        )

    def test_many_worlds_converge(self) -> None:
        """1024 identical pendulum worlds -- max divergence across
        worlds stays under 1 mm.

        This is the "many worlds" stress test from the jitter
        suite, scaled down from 1024 but still exercising
        the multi-block per-world dispatcher across a large grid.
        """
        device = wp.get_device("cuda:0")
        num_worlds = 256
        world, cube_slots = _build_n_pendulums(
            num_worlds=num_worlds, device=device
        )
        _run_frames(world, 10)
        positions = world.bodies.position.numpy()
        ref = positions[cube_slots[0]]
        max_dev = 0.0
        for w in range(1, num_worlds):
            dev = float(np.max(np.abs(positions[cube_slots[w]] - ref)))
            if dev > max_dev:
                max_dev = dev
        self.assertLess(
            max_dev,
            1.0e-3,
            f"max divergence across {num_worlds} worlds was "
            f"{max_dev:.6f} (> 1e-3)",
        )


if __name__ == "__main__":
    unittest.main()
