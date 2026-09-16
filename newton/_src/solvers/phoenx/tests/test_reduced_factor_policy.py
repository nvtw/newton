# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Deep small reduced trees use the existing cooperative mass factor."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.reduced import ReducedArticulationSystem
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


def _chain_builder(levels):
    """Build a floating chain with alternating hinge axes and finite inertia."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    root = builder.add_link(mass=2.0)
    builder.add_shape_box(root, hx=0.1, hy=0.08, hz=0.06)
    joints = [builder.add_joint_free(parent=-1, child=root)]
    parent = root
    for level in range(1, levels):
        body = builder.add_link(mass=1.0 + 0.1 * level)
        builder.add_shape_box(body, hx=0.1, hy=0.08, hz=0.06)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=newton.Axis.Y if level % 2 else newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
            )
        )
        parent = body
    builder.add_articulation(joints)
    return builder


@unittest.skipUnless(_cuda_with_graph_capture(), "Cooperative factors require CUDA")
class TestReducedFactorPolicy(unittest.TestCase):
    def test_deep_single_tree_avoids_per_depth_launches(self):
        """Deep trees use cooperative work while shallow small batches keep their path."""
        for levels, count, expected in ((2, 1, False), (7, 1, False), (8, 1, True), (14, 1, True), (2, 32, True)):
            with self.subTest(levels=levels, count=count):
                blueprint = _chain_builder(levels)
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                for _ in range(count):
                    builder.add_builder(blueprint)
                system = ReducedArticulationSystem(builder.finalize(device="cuda:0"))
                self.assertEqual(system.use_warp_factor, expected)
                self.assertEqual(system._use_factor_dof_split, expected)

    def test_captured_factor_keeps_velocity_dependent_limit_refresh(self):
        """Changing predicted limit activation rebuilds the same current-pose operator."""
        model = _chain_builder(14).finalize(device="cuda:0")
        state, control = model.state(), model.control()
        q = state.joint_q.numpy()
        q[7:] = 0.49
        state.joint_q.assign(q)
        for name, value in (
            ("joint_limit_upper", 0.5),
            ("joint_limit_lower", -0.5),
            ("joint_limit_ke", 100.0),
            ("joint_limit_kd", 3.0),
            ("joint_damping", 0.2),
            ("joint_armature", 0.01),
        ):
            values = getattr(model, name).numpy()
            values[6:] = value
            getattr(model, name).assign(values)
        system = ReducedArticulationSystem(model)
        self.assertTrue(system.use_warp_factor)
        fields = (
            "joint_factor_diagonal",
            "joint_implicit_force",
            "joint_qd_internal",
            "body_i_s",
            "joint_s",
            "reduced_inertia",
            "joint_u_matrix",
            "joint_d_inv",
        )
        diagonals = []
        for speed in (0.0, 2.0, -2.0):
            qd = np.zeros(model.joint_dof_count, dtype=np.float32)
            qd[6:] = speed
            state.joint_qd.assign(qd)
            outputs = []
            for cooperative in (False, True):
                system.use_warp_factor = cooperative
                system._use_factor_dof_split = cooperative
                system.factor(state, control, 0.01)
                with wp.ScopedCapture(device=model.device) as capture:
                    system.factor(state, control, 0.01)
                wp.capture_launch(capture.graph)
                outputs.append({field: getattr(system, field).numpy().copy() for field in fields})
            for field in fields:
                np.testing.assert_array_equal(outputs[0][field], outputs[1][field], err_msg=f"speed={speed}, {field}")
                self.assertTrue(np.isfinite(outputs[1][field]).all())
            diagonals.append(outputs[1]["joint_factor_diagonal"])
        self.assertFalse(np.array_equal(diagonals[0], diagonals[1]))
        np.testing.assert_array_equal(diagonals[0], diagonals[2])


if __name__ == "__main__":
    unittest.main()
