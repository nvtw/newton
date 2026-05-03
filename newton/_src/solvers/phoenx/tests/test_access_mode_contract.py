# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the :class:`ConstraintAccessMode` contract.

Covers (1) metadata presence on every registered constraint module,
(2) position-level writeback round-trip on a cloth scene, and
(3) the host-side guard rejecting POSITION_LEVEL rigid constraints
that lack body-side snapshot fields.
"""

from __future__ import annotations

import types
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints import (
    constraint_actuated_double_ball_socket,
    constraint_cloth_triangle,
    constraint_contact,
)
from newton._src.solvers.phoenx.constraints.constraint_access_mode import ConstraintAccessMode
from newton._src.solvers.phoenx.solver import SolverPhoenX


class TestConstraintAccessModeMetadata(unittest.TestCase):
    def test_every_module_exports_access_mode(self) -> None:
        for module in SolverPhoenX._REGISTERED_CONSTRAINT_MODULES:
            with self.subTest(module=module.__name__):
                self.assertTrue(hasattr(module, "ACCESS_MODE"))
                self.assertIsInstance(module.ACCESS_MODE, ConstraintAccessMode)

    def test_known_assignments(self) -> None:
        self.assertIs(constraint_contact.ACCESS_MODE, ConstraintAccessMode.VELOCITY_LEVEL)
        self.assertIs(
            constraint_actuated_double_ball_socket.ACCESS_MODE,
            ConstraintAccessMode.VELOCITY_LEVEL,
        )
        self.assertIs(
            constraint_cloth_triangle.ACCESS_MODE,
            ConstraintAccessMode.POSITION_LEVEL,
        )

    def test_enum_values_match_mass_splitting_constants(self) -> None:
        from newton._src.solvers.phoenx.mass_splitting.state import (
            ACCESS_MODE_POSITION_LEVEL,
            ACCESS_MODE_VELOCITY_LEVEL,
        )

        self.assertEqual(int(ConstraintAccessMode.VELOCITY_LEVEL), ACCESS_MODE_VELOCITY_LEVEL)
        self.assertEqual(int(ConstraintAccessMode.POSITION_LEVEL), ACCESS_MODE_POSITION_LEVEL)


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
class TestPositionLevelRigidGuard(unittest.TestCase):
    def test_fake_position_level_rigid_constraint_raises(self) -> None:
        device = wp.get_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)))
        builder.add_shape_box(0, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        fake_module = types.ModuleType("newton._test_fake_position_level_rigid")
        fake_module.ACCESS_MODE = ConstraintAccessMode.POSITION_LEVEL

        original_registered = SolverPhoenX._REGISTERED_CONSTRAINT_MODULES
        original_rigid = SolverPhoenX._RIGID_TOUCHING_CONSTRAINT_MODULES
        SolverPhoenX._REGISTERED_CONSTRAINT_MODULES = original_registered + (fake_module,)
        SolverPhoenX._RIGID_TOUCHING_CONSTRAINT_MODULES = original_rigid | {fake_module}
        try:
            with self.assertRaises(NotImplementedError) as ctx:
                SolverPhoenX(model)
            self.assertIn("position_substep_start", str(ctx.exception))
        finally:
            SolverPhoenX._REGISTERED_CONSTRAINT_MODULES = original_registered
            SolverPhoenX._RIGID_TOUCHING_CONSTRAINT_MODULES = original_rigid

    def test_fake_module_without_access_mode_raises_typeerror(self) -> None:
        device = wp.get_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        model = builder.finalize(device=device)

        fake_module = types.ModuleType("newton._test_fake_no_access_mode")

        original_registered = SolverPhoenX._REGISTERED_CONSTRAINT_MODULES
        SolverPhoenX._REGISTERED_CONSTRAINT_MODULES = original_registered + (fake_module,)
        try:
            with self.assertRaises(TypeError) as ctx:
                SolverPhoenX(model)
            self.assertIn("ACCESS_MODE", str(ctx.exception))
        finally:
            SolverPhoenX._REGISTERED_CONSTRAINT_MODULES = original_registered


if __name__ == "__main__":
    unittest.main()
