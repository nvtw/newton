# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-solver behavioral tests for maximal-coordinate fixed joints."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

FPS = 60
SUBSTEPS = 5
FRAMES = 30
_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_fixed_model() -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_INERTIA,
    )
    joint = builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device()), body


def _rollout(
    model: newton.Model, body_velocity: np.ndarray | None = None
) -> tuple[newton.State, newton.solvers.SolverPhoenX]:
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=2,
        velocity_iterations=1,
        articulation_mode="maximal",
    )
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    if body_velocity is not None:
        state.body_qd.assign(body_velocity)
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(FRAMES):
        wp.capture_launch(capture.graph)
    return state, solver


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX direct fixed-joint tests require CUDA graphs.")
class TestFixedJoint(unittest.TestCase):
    """Validate fixed joints through six direct equality rows."""

    def test_welded_cube_holds_position_under_gravity(self) -> None:
        """Hold the welded body at its authored world position under gravity."""
        model, body = _build_fixed_model()
        state, solver = _rollout(model)
        self.assertEqual(solver._direct_equality_system.topology.dimensions, (6,))
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
        np.testing.assert_allclose(state.body_q.numpy()[body, :3], (0.5, 0.0, 0.0), rtol=0.0, atol=2.0e-3)

    def test_welded_cube_does_not_rotate(self) -> None:
        """Reject initial angular velocity on every locked axis."""
        model, body = _build_fixed_model()
        velocity = np.zeros((1, 6), dtype=np.float32)
        velocity[body, 3:] = (1.0, -0.5, 1.0)
        state, _solver = _rollout(model, velocity)
        self.assertLess(float(np.linalg.norm(state.body_qd.numpy()[body, 3:])), 2.0e-3)

    def test_welded_orientation_preserved(self) -> None:
        """Preserve the authored identity orientation under external load."""
        model, body = _build_fixed_model()
        state, _solver = _rollout(model)
        quaternion = state.body_q.numpy()[body, 3:]
        self.assertLess(float(2.0 * np.arccos(np.clip(abs(quaternion[3]), 0.0, 1.0))), 2.0e-3)


if __name__ == "__main__":
    unittest.main()
