# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Production PhoenX tests for direct joint drives coupled to PGS limits."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

FPS = 60
SUBSTEPS = 5
SOLVER_ITERATIONS = 2
FRAMES = 120
_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_STEP_LAYOUTS = ("multi_world", "single_world")


def _build_axial_limit(
    joint_type: newton.JointType,
    *,
    lower: float,
    upper: float,
) -> newton.Model:
    """Build one directly driven axial joint with a hard inequality stop."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    child = builder.add_link(xform=wp.transform_identity(), mass=1.0, inertia=_INERTIA)
    joint_kwargs = {
        "parent": -1,
        "child": child,
        "limit_lower": lower,
        "limit_upper": upper,
        "target_ke": 100.0,
        "target_kd": 10.0,
        "effort_limit": 100.0,
        "actuator_mode": newton.JointTargetMode.POSITION_VELOCITY,
    }
    if joint_type == newton.JointType.REVOLUTE:
        joint = builder.add_joint_revolute(axis=(0.0, 0.0, 1.0), **joint_kwargs)
    else:
        joint = builder.add_joint_prismatic(axis=(1.0, 0.0, 0.0), **joint_kwargs)
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device())


def _rollout(
    model: newton.Model,
    *,
    layout: str,
    target: float,
) -> tuple[float, float, newton.solvers.SolverPhoenX]:
    """Run the direct-drive/PGS-limit split with a five-substep CUDA graph."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    control.joint_target_q.assign(np.asarray((target,), dtype=np.float32))
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        velocity_iterations=1,
        articulation_mode="maximal",
        step_layout=layout,
    )
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(FRAMES):
        wp.capture_launch(capture.graph)
    joint_q = wp.zeros_like(model.joint_q)
    joint_qd = wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, state, joint_q, joint_qd)
    return float(joint_q.numpy()[0]), float(joint_qd.numpy()[0]), solver


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX limit tests require CUDA graphs.")
class TestJointModes(unittest.TestCase):
    """Verify bilateral rows and drives are direct while limits remain inequalities."""

    def _check_limit(
        self,
        joint_type: newton.JointType,
        *,
        lower: float,
        upper: float,
        target: float,
        expected: float,
    ) -> None:
        """Push one free coordinate into a stop and verify ownership and response."""
        for layout in _STEP_LAYOUTS:
            with self.subTest(joint_type=joint_type.name, step_layout=layout, target=target):
                model = _build_axial_limit(joint_type, lower=lower, upper=upper)
                coordinate, speed, solver = _rollout(model, layout=layout, target=target)
                direct = solver._direct_equality_system
                self.assertEqual(direct.topology.dimensions, (6,))
                self.assertTrue(direct.direct_drive_joint_mask[0])
                np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), [1])
                self.assertAlmostEqual(coordinate, expected, delta=0.025)
                self.assertLess(abs(speed), 0.03)

    def test_revolute_upper_limit(self) -> None:
        """Hold a driven hinge at its upper angular stop."""
        self._check_limit(newton.JointType.REVOLUTE, lower=-2.0, upper=0.35, target=1.0, expected=0.35)

    def test_revolute_lower_limit(self) -> None:
        """Hold a driven hinge at its lower angular stop."""
        self._check_limit(newton.JointType.REVOLUTE, lower=-0.35, upper=2.0, target=-1.0, expected=-0.35)

    def test_prismatic_upper_limit(self) -> None:
        """Hold a driven slider at its upper linear stop."""
        self._check_limit(newton.JointType.PRISMATIC, lower=-2.0, upper=0.35, target=1.0, expected=0.35)

    def test_prismatic_lower_limit(self) -> None:
        """Hold a driven slider at its lower linear stop."""
        self._check_limit(newton.JointType.PRISMATIC, lower=-0.35, upper=2.0, target=-1.0, expected=-0.35)


if __name__ == "__main__":
    unittest.main()
