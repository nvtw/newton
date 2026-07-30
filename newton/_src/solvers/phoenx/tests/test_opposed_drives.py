# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical equilibrium tests for PhoenX direct revolute drives."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81
FPS = 240
SUBSTEPS = 5
SOLVER_ITERATIONS = 1
SETTLE_FRAMES = 480
LEVER = 1.0
MASS = 1.0
INERTIA = 0.1
STEP_LAYOUTS = ("multi_world", "single_world")


def _make_drive_config(*, ke: float, kd: float) -> dict[str, float | newton.JointTargetMode]:
    """Build an unlimited implicit position-velocity actuator configuration."""
    return {
        "target_ke": ke,
        "target_kd": kd,
        "effort_limit": 0.0,
        "actuator_mode": newton.JointTargetMode.POSITION_VELOCITY,
    }


def _build_pd_vs_gravity(*, stiffness: float, damping: float) -> newton.Model:
    """Build a one-metre pendulum driven about its world-Z hinge."""
    builder = newton.ModelBuilder(gravity=(0.0, -GRAVITY, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform((LEVER, 0.0, 0.0), wp.quat_identity()),
        mass=MASS,
        inertia=((INERTIA, 0.0, 0.0), (0.0, INERTIA, 0.0), (0.0, 0.0, INERTIA)),
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform((-LEVER, 0.0, 0.0), wp.quat_identity()),
        **_make_drive_config(ke=stiffness, kd=damping),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device())


def _build_opposed_drives(*, target_velocity: float, ke: float, kd_position: float, kd_velocity: float) -> newton.Model:
    """Build two redundant coaxial hinges that drive one dynamic body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=MASS,
        inertia=((INERTIA, 0.0, 0.0), (0.0, INERTIA, 0.0), (0.0, 0.0, INERTIA)),
    )
    position_joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        **_make_drive_config(ke=ke, kd=kd_position),
    )
    velocity_joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        target_ke=0.0,
        target_kd=kd_velocity,
        effort_limit=0.0,
        actuator_mode=newton.JointTargetMode.VELOCITY,
    )
    builder.add_articulation([position_joint, velocity_joint])
    model = builder.finalize(device=wp.get_preferred_device())
    target_qd = model.joint_target_qd.numpy()
    target_qd[1] = target_velocity
    model.joint_target_qd.assign(target_qd)
    return model


def _rollout(
    model: newton.Model,
    *,
    layout: str,
    target_q: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, newton.solvers.SolverPhoenX]:
    """Settle a maximal-coordinate model through a five-substep CUDA graph."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    if target_q is not None:
        control.joint_target_q.assign(target_q)
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        velocity_iterations=0,
        articulation_mode="maximal",
        step_layout=layout,
    )
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(SETTLE_FRAMES):
        wp.capture_launch(capture.graph)
    joint_q = wp.zeros_like(model.joint_q)
    joint_qd = wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, state, joint_q, joint_qd)
    return joint_q.numpy(), joint_qd.numpy(), solver


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX drive tests require CUDA graphs.")
class TestDirectDriveEquilibrium(unittest.TestCase):
    """Compare direct revolute drives with static analytical solutions."""

    def test_gravity_deflection_matches_rotary_hookes_law(self) -> None:
        """Balance pendulum gravity with an implicit direct PD drive."""
        ke = 200.0
        expected = 0.0
        for _ in range(40):
            expected = -(MASS * GRAVITY * LEVER * math.cos(expected)) / ke

        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model = _build_pd_vs_gravity(stiffness=ke, damping=20.0)
                q, qd, solver = _rollout(model, layout=layout)
                direct = solver._direct_equality_system
                self.assertEqual(direct.topology.dimensions, (6,))
                self.assertTrue(direct.direct_drive_joint_mask[0])
                np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [0])
                self.assertAlmostEqual(float(q[0]), expected, delta=2.0e-3)
                self.assertLess(abs(float(qd[0])), 2.0e-3)

    def test_stiffness_scales_gravity_deflection(self) -> None:
        """Scale the small-angle deflection inversely with drive stiffness."""
        for ke in (500.0, 1000.0, 2000.0, 4000.0):
            with self.subTest(stiffness=ke):
                model = _build_pd_vs_gravity(stiffness=ke, damping=30.0)
                q, qd, solver = _rollout(model, layout="multi_world")
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                self.assertAlmostEqual(ke * abs(float(q[0])), MASS * GRAVITY * LEVER, delta=0.1)
                self.assertLess(abs(float(qd[0])), 2.0e-3)

    def test_redundant_opposed_drives_balance(self) -> None:
        """Balance coaxial position and velocity drives in one detected mechanism."""
        target_velocity = 0.5
        ke = 200.0
        kd_velocity = 10.0
        expected_angle = kd_velocity * target_velocity / ke
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model = _build_opposed_drives(
                    target_velocity=target_velocity,
                    ke=ke,
                    kd_position=20.0,
                    kd_velocity=kd_velocity,
                )
                q, qd, solver = _rollout(model, layout=layout, target_q=np.zeros(2, dtype=np.float32))
                direct = solver._direct_equality_system
                self.assertEqual(direct.topology.dimensions, (12,))
                np.testing.assert_array_equal(direct.direct_drive_joint_mask, [True, True])
                np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [0, 0])
                np.testing.assert_allclose(q, expected_angle, rtol=0.0, atol=1.0e-3)
                np.testing.assert_allclose(qd, 0.0, rtol=0.0, atol=2.0e-3)


if __name__ == "__main__":
    unittest.main()
