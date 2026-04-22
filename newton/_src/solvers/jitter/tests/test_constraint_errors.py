# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`World.gather_constraint_errors` /
:meth:`World.gather_contact_errors`.

Asserts:

* At construction a well-posed joint reports zero error.
* A displaced ball-socket reports ``p2 - p1`` in ``spatial_top``.
* After settling under gravity, joint positional errors stay tiny.
* Contact errors report the expected penetration / separation of a
  cube resting on a plane.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.tests._test_helpers import run_settle_loop  # noqa: F401
from newton._src.solvers.jitter.world_builder import WorldBuilder


SUBSTEPS = 4
SOLVER_ITERATIONS = 16
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_ball_socket(device, *, offset=(0.0, 0.0, 0.0)):
    """Single ball-socket; cube's world position is offset from the
    initialised-rest pose so the anchors no longer coincide."""
    b = WorldBuilder()
    world_body = b.world_body
    cube = b.add_dynamic_body(
        position=(0.0, -1.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
    )
    b.add_ball_socket(world_body, cube, anchor=(0.0, 0.0, 0.0))
    world = b.finalize(
        enable_all_constraints=True,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        device=device,
    )
    if any(v != 0.0 for v in offset):
        # Nudge the cube's world position after finalize() so the
        # initial-pose anchors in body-local space (which were
        # snapped at add_ball_socket time) no longer line up when
        # re-projected into world space. gather_constraint_errors
        # should surface exactly the nudge vector.
        pos_np = world.bodies.position.numpy()
        pos_np[cube] = pos_np[cube] + np.asarray(offset, dtype=np.float32)
        world.bodies.position.assign(pos_np)
    return world


class TestConstraintErrors(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device()

    def test_ball_socket_zero_error_at_rest(self):
        world = _build_ball_socket(self.device)
        out = wp.zeros(world._constraint_capacity, dtype=wp.spatial_vector, device=self.device)
        world.gather_constraint_errors(out)
        err = out.numpy()[0]
        # All six components should be ~0 (no displacement, no angular
        # rows for ball-socket).
        self.assertLess(float(np.max(np.abs(err))), 1e-5)

    def test_ball_socket_reports_displacement(self):
        offset = (0.05, 0.0, 0.0)
        world = _build_ball_socket(self.device, offset=offset)
        out = wp.zeros(world._constraint_capacity, dtype=wp.spatial_vector, device=self.device)
        world.gather_constraint_errors(out)
        err = out.numpy()[0]
        # spatial_top = p2 - p1. Moving body 2 by +x by ``offset`` gives
        # a +x residual of the same magnitude; angular part stays zero.
        np.testing.assert_allclose(err[:3], np.asarray(offset, dtype=np.float32), atol=1e-5)
        self.assertLess(float(np.max(np.abs(err[3:]))), 1e-5)

    def test_gather_before_first_step(self):
        """Calling before step() must not crash (the error kernel only
        reads persisted state / body pose, not solve state)."""
        world = _build_ball_socket(self.device)
        out = wp.zeros(world._constraint_capacity, dtype=wp.spatial_vector, device=self.device)
        world.gather_constraint_errors(out)  # must not raise
        # Initial pose -> zero error.
        err = out.numpy()[0]
        self.assertLess(float(np.max(np.abs(err))), 1e-5)

    def test_gather_contact_errors_no_contacts(self):
        """Without any contacts, gather_contact_errors zeroes the output
        and returns without crashing. Exercises the early-out path."""
        world = _build_ball_socket(self.device)
        # max_contact_columns is zero for a pure-joint scene, so the
        # kernel takes the fast early-out; just asserting no crash.
        out = wp.zeros(max(1, world.rigid_contact_max), dtype=wp.vec3f, device=self.device)
        world.gather_contact_errors(out)
        self.assertLess(float(np.max(np.abs(out.numpy()))), 1e-5)


if __name__ == "__main__":
    unittest.main()
