# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Legacy D6 reductions for the restored PhoenX ADBS joint path."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)


def _make_body(builder: newton.ModelBuilder) -> int:
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    return body


def _mode_for(model: newton.Model) -> int:
    solver = newton.solvers.SolverPhoenX(model, substeps=1)
    return int(solver._adbs.joint_mode.numpy()[0])


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 legacy dispatch tests run on CUDA only")
class TestD6LegacyDispatch(unittest.TestCase):
    def test_angular_three_axis_d6_reduces_to_ball_socket_with_limits(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-1.0, limit_upper=1.0),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])

        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)

        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_BALL_SOCKET))
        self.assertEqual(int(solver._adbs.d6_limit_count.numpy()[0]), 3)
        np.testing.assert_allclose(solver._adbs.d6_limit_lower.numpy()[0], [-1.0, -1.0, -1.0], atol=1.0e-6)
        np.testing.assert_allclose(solver._adbs.d6_limit_upper.numpy()[0], [1.0, 1.0, 1.0], atol=1.0e-6)

    def test_angular_two_axis_mjcf_style_d6_reduces_to_universal_with_limits(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-0.8, limit_upper=0.8),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.3, limit_upper=0.5),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])

        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)

        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_UNIVERSAL))
        self.assertEqual(int(solver._adbs.d6_limit_count.numpy()[0]), 2)
        np.testing.assert_allclose(solver._adbs.d6_limit_lower.numpy()[0], [-0.8, -1.3, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(solver._adbs.d6_limit_upper.numpy()[0], [0.8, 0.5, 0.0], atol=1.0e-6)

    def test_angular_two_axis_d6_limit_row_reacts_to_violation(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-0.25, limit_upper=0.25),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        model.set_gravity((0.0, 0.0, 0.0))
        solver = newton.solvers.SolverPhoenX(model, substeps=2, solver_iterations=8)

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        body_q = state_0.body_q.numpy()
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.65)
        body_q[0, 3:7] = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
        state_0.body_q.assign(body_q)
        state_0.body_qd.assign(np.zeros((1, 6), dtype=np.float32))

        state_0.clear_forces()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        body_qd = state_1.body_qd.numpy()[0]
        self.assertLess(
            float(body_qd[3]),
            -1.0e-4,
            msg=f"D6 angular limit did not push back on a positive X violation: omega_x={body_qd[3]:.6f}",
        )

    def test_angular_one_axis_d6_reduces_to_revolute(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(
                axis=(0.0, 1.0, 0.0),
                limit_lower=-0.25,
                limit_upper=0.5,
                target_pos=0.1,
                target_ke=10.0,
                target_kd=1.0,
            )
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)

        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_REVOLUTE))
        self.assertEqual(int(solver._adbs.joint_idx_to_dof_start.numpy()[0]), 0)
        self.assertAlmostEqual(float(solver._adbs.target.numpy()[0]), 0.1, places=6)

    def test_linear_one_axis_d6_reduces_to_prismatic(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-0.2, limit_upper=0.3)]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])

        self.assertEqual(_mode_for(builder.finalize()), int(JOINT_MODE_PRISMATIC))

    def test_generic_d6_still_raises(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])

        with self.assertRaisesRegex(NotImplementedError, "cannot be reduced"):
            newton.solvers.SolverPhoenX(builder.finalize(), substeps=1)


if __name__ == "__main__":
    unittest.main()
