# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-solver behavioral tests for maximal-coordinate prismatic joints."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

FPS = 60
SUBSTEPS = 5
STEP_LAYOUTS = ("multi_world", "single_world")
_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_slider(*, kp: float = 0.0, kd: float = 0.0, gravity: bool = False) -> tuple[newton.Model, int]:
    gravity_vector = (0.0, -9.81, 0.0) if gravity else (0.0, 0.0, 0.0)
    builder = newton.ModelBuilder(gravity=gravity_vector, up_axis=newton.Axis.Y)
    body = builder.add_link(xform=wp.transform_identity(), mass=1.0, inertia=_INERTIA)
    target_mode = (
        newton.JointTargetMode.VELOCITY if kp == 0.0 and kd > 0.0 else newton.JointTargetMode.POSITION_VELOCITY
    )
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        target_ke=kp,
        target_kd=kd,
        effort_limit=0.0,
        actuator_mode=target_mode,
        limit_lower=-np.inf,
        limit_upper=np.inf,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device()), body


def _rollout(
    model: newton.Model,
    *,
    layout: str,
    frames: int,
    target_q: float = 0.0,
    target_qd: float = 0.0,
) -> tuple[newton.State, newton.solvers.SolverPhoenX]:
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    control.joint_target_q.assign(np.asarray((target_q,), dtype=np.float32))
    control.joint_target_qd.assign(np.asarray((target_qd,), dtype=np.float32))
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=2,
        velocity_iterations=1,
        articulation_mode="maximal",
        step_layout=layout,
    )
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(frames):
        wp.capture_launch(capture.graph)
    return state, solver


def _joint_state(model: newton.Model, state: newton.State) -> tuple[float, float]:
    q = wp.zeros_like(model.joint_q)
    qd = wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, state, q, qd)
    return float(q.numpy()[0]), float(qd.numpy()[0])


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX direct slider tests require CUDA graphs.")
class TestPrismatic(unittest.TestCase):
    """Validate prismatic structure and drives through direct rows."""

    def test_gravity_slide_is_free(self) -> None:
        """Accelerate freely along the authored axis while locking all other motion."""
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, body = _build_slider(gravity=True)
                state, solver = _rollout(model, layout=layout, frames=60)
                coordinate, speed = _joint_state(model, state)
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (5,))
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                self.assertAlmostEqual(speed, -9.81, delta=0.01)
                self.assertAlmostEqual(coordinate, -0.5 * 9.81, delta=0.1)
                velocity = state.body_qd.numpy()[body]
                self.assertLess(float(np.linalg.norm(velocity[[0, 2, 3, 4, 5]])), 2.0e-3)

    def test_velocity_drive_tracks_target(self) -> None:
        """Track a constant slider speed with the direct implicit velocity row."""
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, _body = _build_slider(kd=10.0)
                state, solver = _rollout(model, layout=layout, frames=120, target_qd=1.0)
                _coordinate, speed = _joint_state(model, state)
                self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
                self.assertAlmostEqual(speed, 1.0, delta=2.0e-3)

    def test_position_drive_tracks_target(self) -> None:
        """Track a slider position target with the direct implicit PD row."""
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, _body = _build_slider(kp=80.0, kd=10.0)
                state, solver = _rollout(model, layout=layout, frames=120, target_q=0.5)
                coordinate, speed = _joint_state(model, state)
                self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
                self.assertAlmostEqual(coordinate, 0.5, delta=2.0e-3)
                self.assertLess(abs(speed), 2.0e-3)


if __name__ == "__main__":
    unittest.main()
