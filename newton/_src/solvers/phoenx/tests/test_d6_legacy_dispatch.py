# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Legacy D6 reductions for the restored PhoenX ADBS joint path."""

from __future__ import annotations

import unittest

import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
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
    def test_angular_three_axis_d6_reduces_to_ball_socket(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-1.0, limit_upper=1.0),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])

        self.assertEqual(_mode_for(builder.finalize()), int(JOINT_MODE_BALL_SOCKET))

    def test_angular_two_axis_mjcf_style_d6_reduces_to_ball_socket(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-0.8, limit_upper=0.8),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.3, limit_upper=0.5),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])

        self.assertEqual(_mode_for(builder.finalize()), int(JOINT_MODE_BALL_SOCKET))

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
