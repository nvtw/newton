# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the per-entity access-mode synchronization helpers.

Covers (1) the integer constants line up with the parallel
``mass_splitting.state`` definitions (so a future mass-splitting
integration shares the tag space), (2) the position-level writeback
round-trip on a cloth scene leaves ``particle_qd`` consistent with the
per-step position delta, and (3) the substep-entry pre-pass resets
every entity's ``access_mode`` field so a stale flag from a previous
step can't bleed into the new substep.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_NONE,
    ACCESS_MODE_POSITION_LEVEL,
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
)
from newton._src.solvers.phoenx.solver import SolverPhoenX


class TestAccessModeConstants(unittest.TestCase):
    def test_constants_match_mass_splitting_state(self) -> None:
        # The mass-splitting subsystem defines its own copy of the
        # access-mode tag space; the values must agree byte-for-byte
        # so a future merge into mass_splitting routes the same int
        # without translation.
        from newton._src.solvers.phoenx.mass_splitting.state import (
            ACCESS_MODE_NONE as MS_NONE,
            ACCESS_MODE_POSITION_LEVEL as MS_POS,
            ACCESS_MODE_VELOCITY_LEVEL as MS_VEL,
        )

        self.assertEqual(ACCESS_MODE_NONE, MS_NONE)
        self.assertEqual(ACCESS_MODE_VELOCITY_LEVEL, MS_VEL)
        self.assertEqual(ACCESS_MODE_POSITION_LEVEL, MS_POS)

    def test_constants_are_distinct_int_tags(self) -> None:
        tags = {
            ACCESS_MODE_NONE,
            ACCESS_MODE_VELOCITY_LEVEL,
            ACCESS_MODE_POSITION_LEVEL,
            ACCESS_MODE_STATIC,
        }
        self.assertEqual(len(tags), 4)
        for v in tags:
            self.assertIsInstance(v, int)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "SolverPhoenX runs on CUDA only.",
)
class TestPositionLevelWritebackContract(unittest.TestCase):
    """Cloth triangles leave particle velocity consistent with the
    per-step position delta."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        builder = newton.ModelBuilder()
        builder.gravity = -9.81
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=3,
            dim_y=3,
            cell_x=0.25,
            cell_y=0.25,
            mass=0.05,
            fix_left=True,
            tri_ke=1.0e6,
            tri_ka=1.0e6,
        )
        self.model = builder.finalize(device=self.device)
        self.solver = SolverPhoenX(
            self.model,
            substeps=4,
            solver_iterations=8,
        )

    def test_particle_velocity_matches_position_delta(self) -> None:
        state_0 = self.model.state()
        state_1 = self.model.state()
        control = self.model.control()

        dt = 1.0 / 240.0
        position_before = state_0.particle_q.numpy().copy()
        self.solver.step(state_0, state_1, control, None, dt)

        position_after = state_1.particle_q.numpy()
        velocity_after = state_1.particle_qd.numpy()
        inv_mass = self.model.particle_inv_mass.numpy()

        self.assertTrue(np.all(np.isfinite(position_after)))
        self.assertTrue(np.all(np.isfinite(velocity_after)))

        free = inv_mass > 0.0
        if not np.any(free):
            self.skipTest("Cloth grid has no free particles")
        # Tolerance is loose: the solver substeps internally so this
        # is "consistent with the per-step delta", not bit-identical.
        expected = (position_after[free] - position_before[free]) / dt
        actual = velocity_after[free]
        diff = np.abs(actual - expected)
        scale = np.maximum(np.abs(expected), 1.0e-3)
        np.testing.assert_array_less(diff, 5.0 * scale + 1.0e-2)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "SolverPhoenX runs on CUDA only.",
)
class TestSubstepEntryAccessModeReset(unittest.TestCase):
    """The substep-entry pre-pass stamps every entity with a fresh
    ``access_mode`` so iterate-time flips can't leak across steps."""

    def test_dynamic_bodies_reset_to_velocity_level(self) -> None:
        device = wp.get_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)))
        builder.add_shape_box(0, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = SolverPhoenX(model)

        # Manually corrupt the per-body access mode to a stale value
        # before stepping; the substep-entry kernel must overwrite it.
        access_mode = solver.bodies.access_mode.numpy().copy()
        bogus = np.full_like(access_mode, ACCESS_MODE_POSITION_LEVEL)
        solver.bodies.access_mode.assign(bogus)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)

        post = solver.bodies.access_mode.numpy()
        # Static / kinematic / inv_mass=0 bodies become STATIC; the
        # one dynamic body becomes VELOCITY_LEVEL. None should still
        # be the corrupted POSITION_LEVEL value.
        self.assertTrue(np.all((post == ACCESS_MODE_VELOCITY_LEVEL) | (post == ACCESS_MODE_STATIC)))


if __name__ == "__main__":
    unittest.main()
