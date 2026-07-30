# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-solver behavioral tests for maximal-coordinate revolute joints."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton

FPS = 60
SUBSTEPS = 5
STEP_LAYOUTS = ("multi_world", "single_world")
_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_hinge(
    *,
    kp: float = 0.0,
    kd: float = 0.0,
    damping: float = 0.0,
    effort: float = 0.0,
    gravity: bool = True,
) -> tuple[newton.Model, int]:
    gravity_vector = (0.0, -9.81, 0.0) if gravity else (0.0, 0.0, 0.0)
    builder = newton.ModelBuilder(gravity=gravity_vector, up_axis=newton.Axis.Y)
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, -1.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_INERTIA,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity()),
        target_ke=kp,
        target_kd=kd,
        damping=damping,
        effort_limit=effort,
        actuator_mode=(
            newton.JointTargetMode.VELOCITY if kp == 0.0 and kd > 0.0 else newton.JointTargetMode.POSITION_VELOCITY
        ),
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
    q: float = 0.0,
    qd: float = 0.0,
    target_q: float = 0.0,
    target_qd: float = 0.0,
) -> tuple[newton.State, newton.solvers.SolverPhoenX]:
    state = model.state()
    state.joint_q.assign(np.asarray((q,), dtype=np.float32))
    state.joint_qd.assign(np.asarray((qd,), dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
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


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX direct hinge tests require CUDA graphs.")
class TestHingeJoint(unittest.TestCase):
    """Validate revolute structure and drives through direct rows."""

    def test_pendulum_settles(self) -> None:
        """Settle a displaced damped pendulum while rejecting locked-axis motion."""
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, body = _build_hinge(damping=5.0)
                state, solver = _rollout(model, layout=layout, frames=240, q=math.pi / 2.0)
                coordinate, speed = _joint_state(model, state)
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (6,))
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                self.assertAlmostEqual(coordinate, 0.0, delta=0.08)
                self.assertAlmostEqual(speed, 0.0, delta=0.08)
                self.assertLess(float(np.linalg.norm(state.body_qd.numpy()[body, 3:5])), 2.0e-3)

    def test_pd_brake_kills_axial_spin(self) -> None:
        """Brake initial hinge spin with a direct implicit velocity drive."""
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, _body = _build_hinge(kd=5.0, effort=20.0, gravity=False)
                state, solver = _rollout(model, layout=layout, frames=180, qd=3.0)
                _coordinate, speed = _joint_state(model, state)
                self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
                self.assertLess(abs(speed), 0.02)

    def test_position_drive_tracks_target(self) -> None:
        """Track a revolute position target with the direct implicit PD row."""
        target = math.pi / 3.0
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, _body = _build_hinge(kp=40.0, kd=5.0, effort=50.0, gravity=False)
                state, solver = _rollout(model, layout=layout, frames=180, target_q=target)
                coordinate, speed = _joint_state(model, state)
                self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
                self.assertAlmostEqual(coordinate, target, delta=0.03)
                self.assertLess(abs(speed), 0.03)


if __name__ == "__main__":
    unittest.main()
