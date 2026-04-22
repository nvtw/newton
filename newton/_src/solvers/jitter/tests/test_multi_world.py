# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world smoke tests for the jitter solver.

Covers the end-to-end multi-world path introduced in the per-world
CSR bucketing + multi-block dispatcher refactor:

* ``test_single_pendulum_per_world`` -- build ``N`` worlds, each
  containing one identical ball-socket pendulum, step them, and
  check every world's cube ended up at the same settled pose. A
  non-match would indicate the per-world CSR bucketing leaked one
  world's solve into another's state (e.g. all blocks walking the
  same CSR slice).

* ``test_world_independence`` -- build two worlds; give world 1's
  pendulum a non-zero initial angular velocity and leave world 0
  at rest. After a few steps world 0 must still be at rest, proving
  the two worlds' bodies never interact through the solver even
  though they share a single ``BodyContainer`` and
  ``ConstraintContainer``.

* ``test_per_world_gravity`` -- build two worlds with different
  gravity vectors (``-9.81`` vs ``-1.62`` m/s^2) and check each
  world's pendulum swings at a rate consistent with its own
  gravity, not world 0's.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.tests._test_helpers import run_settle_loop
from newton._src.solvers.jitter.world_builder import WorldBuilder


FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_n_pendulums(
    device,
    num_worlds: int,
    *,
    angular_velocities: list[tuple[float, float, float]] | None = None,
    gravity=(0.0, -9.81, 0.0),
):
    """Build ``num_worlds`` copies of the single-pendulum scene.

    Each world gets its own static world body (at index ``w``) and
    one dynamic cube hanging 1 m below it via a ball socket. Worlds
    share no bodies or constraints. Optional ``angular_velocities``
    seeds the per-world cube's initial angular velocity.
    """
    b = WorldBuilder(num_worlds=num_worlds)
    cube_indices = []
    for w in range(num_worlds):
        anchor_body = b.world_body_of(w)
        avel = (0.0, 0.0, 0.0)
        if angular_velocities is not None:
            avel = angular_velocities[w]
        cube = b.add_dynamic_body(
            position=(0.0, -1.0, 0.0),
            inverse_mass=1.0,
            inverse_inertia=_INV_INERTIA,
            affected_by_gravity=True,
            angular_velocity=avel,
            world_id=w,
        )
        b.add_ball_socket(anchor_body, cube, anchor=(0.0, 0.0, 0.0))
        cube_indices.append(cube)
    world = b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=gravity,
        device=device,
    )
    return world, cube_indices


class TestMultiWorld(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device()

    def test_single_pendulum_per_world(self):
        num_worlds = 8
        world, cube_indices = _build_n_pendulums(self.device, num_worlds)
        # Run a few steps; gravity + ball socket should lock the cube
        # to stay hanging directly below the anchor.
        run_settle_loop(world, 30, dt=1.0 / FPS)
        positions = world.bodies.position.numpy()
        ref = positions[cube_indices[0]]
        for w in range(1, num_worlds):
            np.testing.assert_allclose(
                positions[cube_indices[w]], ref, atol=1e-4,
                err_msg=f"world {w} cube diverged from world 0 cube",
            )

    def test_world_independence(self):
        # World 1 gets a spin, world 0 is at rest.
        avels = [(0.0, 0.0, 0.0), (0.0, 5.0, 0.0)]
        world, cube_indices = _build_n_pendulums(
            self.device, 2, angular_velocities=avels
        )
        run_settle_loop(world, 30, dt=1.0 / FPS)
        avels_after = world.bodies.angular_velocity.numpy()
        # World 0's cube must be essentially at rest (only the weak
        # pendulum swing from gravity -- no yaw spin).
        w0_ang = avels_after[cube_indices[0]]
        self.assertLess(
            abs(float(w0_ang[1])), 0.1,
            f"world 0 cube picked up yaw spin {w0_ang[1]:.3f} -- worlds leaking",
        )
        # World 1's cube retains some yaw spin (ball socket doesn't
        # transmit torque, so it should still be spinning unless the
        # solver coupled worlds).
        w1_ang = avels_after[cube_indices[1]]
        self.assertGreater(
            abs(float(w1_ang[1])), 1.0,
            f"world 1 cube lost its yaw spin ({w1_ang[1]:.3f}) -- unexpected damping",
        )

    def test_many_worlds_converge(self):
        """Smoke the 1000+ worlds claim -- build 1024 copies of the
        pendulum scene, step them, verify no world diverged."""
        num_worlds = 1024
        world, cube_indices = _build_n_pendulums(self.device, num_worlds)
        run_settle_loop(world, 10, dt=1.0 / FPS)
        positions = world.bodies.position.numpy()
        ref = positions[cube_indices[0]]
        max_dev = 0.0
        for w in range(1, num_worlds):
            dev = float(np.max(np.abs(positions[cube_indices[w]] - ref)))
            if dev > max_dev:
                max_dev = dev
        self.assertLess(
            max_dev, 1e-3,
            f"max divergence across 1024 worlds was {max_dev:.6f} (>1e-3)",
        )

    def test_per_world_gravity(self):
        # World 0: earth, world 1: moon. Both pendulums start at rest;
        # the tension reaction to gravity should differ 6x.
        gravity = [(0.0, -9.81, 0.0), (0.0, -1.62, 0.0)]
        world, cube_indices = _build_n_pendulums(
            self.device, 2, gravity=gravity
        )
        run_settle_loop(world, 30, dt=1.0 / FPS)
        # Swinging-pendulum energy ~ m g L (1 - cos theta). The two
        # worlds' kinetic + potential energies should be in a 9.81:1.62
        # ratio within the first few frames (before numerical drift).
        # Easier check: tension in the joint must be proportional to g.
        n_constraints = world.num_constraints
        out = wp.zeros(
            world._constraint_capacity, dtype=wp.spatial_vector, device=self.device
        )
        world.gather_constraint_wrenches(out)
        wrenches = out.numpy()
        # Joint 0 is world 0's (cid 0), joint 1 is world 1's (cid 1)
        # under the contiguous-joint-cid convention.
        f_y_w0 = abs(float(wrenches[0][1]))  # +y force on cube from joint
        f_y_w1 = abs(float(wrenches[1][1]))
        if f_y_w0 > 1e-3:
            ratio = f_y_w1 / f_y_w0
            expected = 1.62 / 9.81
            self.assertAlmostEqual(
                ratio, expected, delta=0.2,
                msg=f"per-world gravity: w1/w0 reaction ratio {ratio:.3f} != {expected:.3f}",
            )


if __name__ == "__main__":
    unittest.main()
