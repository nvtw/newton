# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for PhoenX joint inequality feature gaps."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _run(model: newton.Model, frames: int = 30) -> tuple[newton.State, newton.solvers.SolverPhoenX]:
    """Run a short five-substep maximal-coordinate CUDA-graph rollout."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=5,
        solver_iterations=2,
        velocity_iterations=1,
        articulation_mode="maximal",
    )
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / 60.0)
    for _ in range(frames):
        wp.capture_launch(capture.graph)
    return state, solver


def _distance_model(*, position: float, lower: float, upper: float) -> newton.Model:
    """Build one free body constrained to a radial distance interval."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform(wp.vec3(position, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_INERTIA,
    )
    builder.add_joint_distance(
        parent=-1,
        child=body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        min_distance=lower,
        max_distance=upper,
    )
    return builder.finalize(device=wp.get_preferred_device())


def _velocity_limited_model(joint_type: newton.JointType, velocity_limit: float = 1.0) -> newton.Model:
    """Build one axial joint starting substantially above its speed limit."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=_INERTIA,
    )
    if joint_type == newton.JointType.REVOLUTE:
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            limit_lower=-np.inf,
            limit_upper=np.inf,
            velocity_limit=velocity_limit,
        )
    else:
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=newton.Axis.X,
            limit_lower=-np.inf,
            limit_upper=np.inf,
            velocity_limit=velocity_limit,
        )
    builder.add_articulation([joint])
    model = builder.finalize(device=wp.get_preferred_device())
    model.joint_qd.assign(np.asarray((10.0,), dtype=np.float32))
    return model


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX gap regressions require CUDA graphs.")
class TestJointGapSupport(unittest.TestCase):
    """Verify distance and scalar velocity inequalities in production PhoenX."""

    def test_distance_maximum(self) -> None:
        """Correct an initially excessive distance through the maximum-bound row."""
        model = _distance_model(position=2.0, lower=1.0, upper=1.0)
        state, solver = _run(model)
        distance = float(np.linalg.norm(state.body_q.numpy()[0, :3]))
        self.assertAlmostEqual(distance, 1.0, delta=0.025)
        self.assertFalse(solver._direct_equality_system.enabled)
        np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [1])

    def test_distance_minimum(self) -> None:
        """Correct an initially deficient distance through the minimum-bound row."""
        model = _distance_model(position=0.25, lower=1.0, upper=1.0)
        state, solver = _run(model)
        distance = float(np.linalg.norm(state.body_q.numpy()[0, :3]))
        self.assertAlmostEqual(distance, 1.0, delta=0.025)
        self.assertFalse(solver._direct_equality_system.enabled)

    def test_revolute_velocity_limit(self) -> None:
        """Clamp a free hinge to its authored maximum angular speed."""
        model = _velocity_limited_model(newton.JointType.REVOLUTE)
        state, solver = _run(model, frames=2)
        joint_q = wp.zeros_like(model.joint_q)
        joint_qd = wp.zeros_like(model.joint_qd)
        newton.eval_ik(model, state, joint_q, joint_qd)
        self.assertLessEqual(abs(float(joint_qd.numpy()[0])), 1.01)
        np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [1])
        self.assertFalse(solver.world._combine_direct_prepare_projection)

    def test_prismatic_velocity_limit(self) -> None:
        """Clamp a free slider to its authored maximum linear speed."""
        model = _velocity_limited_model(newton.JointType.PRISMATIC)
        state, solver = _run(model, frames=2)
        joint_q = wp.zeros_like(model.joint_q)
        joint_qd = wp.zeros_like(model.joint_qd)
        newton.eval_ik(model, state, joint_q, joint_qd)
        self.assertLessEqual(abs(float(joint_qd.numpy()[0])), 1.01)
        np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [1])
        self.assertFalse(solver.world._combine_direct_prepare_projection)

    def test_projection_combination_tracks_velocity_limit_updates(self) -> None:
        """Refresh projection scheduling when an axial velocity limit changes."""
        model = _velocity_limited_model(newton.JointType.REVOLUTE, velocity_limit=1.0e6)
        _, solver = _run(model, frames=0)
        self.assertTrue(solver.world._combine_direct_prepare_projection)

        model.joint_velocity_limit.assign(np.asarray((1.0,), dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertFalse(solver.world._combine_direct_prepare_projection)

        model.joint_velocity_limit.assign(np.asarray((1.0e6,), dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertTrue(solver.world._combine_direct_prepare_projection)


if __name__ == "__main__":
    unittest.main()
