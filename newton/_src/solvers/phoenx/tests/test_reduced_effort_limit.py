# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Effort limits of joint drives in reduced articulations.

A position drive far from its target asks for far more effort than its limit, so a saturated
drive must turn its joint like a constant torque of the limit; the reduced path once ignored
:attr:`newton.Model.joint_effort_limit` and let the drive apply any effort.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

_INERTIA = 0.5  # [kg*m^2] about the joint axis
_STIFFNESS = 1.0e4  # [N*m/rad]
_DAMPING = 10.0  # [N*m*s/rad]
_TARGET = 1.0  # [rad]
_DT = 1.0 / 240.0  # [s]
_FRAMES = 48


def _cuda_with_graph_capture() -> bool:
    device = wp.get_preferred_device()
    return bool(device.is_cuda and wp.is_mempool_enabled(device))


def _turn(effort_limit: float) -> np.ndarray:
    """Joint angle [rad] after every frame of a driven revolute link without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    link = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((_INERTIA, 0.0, 0.0), (0.0, _INERTIA, 0.0), (0.0, 0.0, _INERTIA)),
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 0.0, 1.0),
        target_ke=_STIFFNESS,
        target_kd=_DAMPING,
        actuator_mode=newton.JointTargetMode.POSITION,
        effort_limit=effort_limit,
    )
    builder.add_articulation([joint])
    model = builder.finalize()
    solver = newton.solvers.SolverPhoenX(model, substeps=4, joint_mode="reduced", reduced_articulations=[0])
    state = model.state()
    control = model.control()
    control.joint_target_q.fill_(_TARGET)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)

    def step() -> None:
        solver.step(state, state, control, None, _DT)
        newton.eval_ik(model, state, joint_q, joint_qd)

    step()  # compile outside the capture
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    with wp.ScopedCapture(device=model.device) as capture:
        step()
    angles = []
    for _ in range(_FRAMES):
        wp.capture_launch(capture.graph)
        angles.append(float(joint_q.numpy()[0]))
    return np.asarray(angles)


@unittest.skipUnless(_cuda_with_graph_capture(), "PhoenX tests need CUDA graph capture")
class TestReducedEffortLimit(unittest.TestCase):
    def test_saturated_drive_turns_like_a_constant_torque(self) -> None:
        limit = 1.0  # [N*m], far below the drive's demand of about _STIFFNESS * _TARGET
        angles = _turn(limit)
        time = _DT * np.arange(1, _FRAMES + 1)
        expected = 0.5 * limit / _INERTIA * time**2
        # Implicit Euler lags the exact parabola by at most one step's worth of velocity.
        np.testing.assert_allclose(angles, expected, atol=limit / _INERTIA * _DT * time[-1] + 1.0e-5)
        self.assertLess(angles[-1], 0.2 * _TARGET)

    def test_zero_limit_leaves_the_drive_unlimited(self) -> None:
        angles = _turn(0.0)
        self.assertAlmostEqual(angles[-1], _TARGET, delta=0.02)

    def test_unreached_limit_changes_nothing(self) -> None:
        np.testing.assert_allclose(_turn(1.0e5), _turn(0.0), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
