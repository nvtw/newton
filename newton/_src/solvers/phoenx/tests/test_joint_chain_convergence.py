# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-solver convergence regressions for long maximal-coordinate joint chains."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81
FPS = 120
SUBSTEPS = 5
SOLVER_ITERATIONS = 2
SETTLE_FRAMES = 120
NUM_LINKS = 16
HALF_EXTENT = 0.5
PITCH = 2.0 * HALF_EXTENT
_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_MAX_TIP_SAG = 0.02 * (NUM_LINKS * PITCH)
_MAX_RIGID_CABLE_TIP_SAG = 1.0e-3
_STEP_LAYOUTS = ("multi_world", "single_world")


def _build_cantilever(device: wp.context.Device, joint_type: newton.JointType) -> newton.Model:
    """Build a horizontal chain whose connectivity PhoenX detects independently."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY), up_axis=newton.Axis.Z)
    bodies = [
        builder.add_link(
            xform=wp.transform(wp.vec3(0.0, -(index + 0.5) * PITCH, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=_INERTIA,
        )
        for index in range(NUM_LINKS)
    ]
    joints: list[int] = []
    for index, child in enumerate(bodies):
        parent = -1 if index == 0 else bodies[index - 1]
        parent_xform = (
            wp.transform_identity() if index == 0 else wp.transform(wp.vec3(0.0, -HALF_EXTENT, 0.0), wp.quat_identity())
        )
        child_xform = wp.transform(wp.vec3(0.0, HALF_EXTENT, 0.0), wp.quat_identity())
        if joint_type == newton.JointType.REVOLUTE:
            joint = builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(0.0, 0.0, 1.0),
                parent_xform=parent_xform,
                child_xform=child_xform,
            )
        elif joint_type == newton.JointType.PRISMATIC:
            joint = builder.add_joint_prismatic(
                parent=parent,
                child=child,
                axis=(1.0, 0.0, 0.0),
                parent_xform=parent_xform,
                child_xform=child_xform,
            )
        else:
            joint = builder.add_joint_cable(
                parent=parent,
                child=child,
                parent_xform=parent_xform,
                child_xform=child_xform,
                stretch_stiffness=1.0e9,
                bend_stiffness=1.0e9,
                twist_stiffness=0.0,
            )
        joints.append(joint)
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _rollout(model: newton.Model, *, layout: str) -> tuple[newton.State, newton.solvers.SolverPhoenX]:
    """Run a five-substep CUDA-graph rollout through production PhoenX."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
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
    for _ in range(SETTLE_FRAMES):
        wp.capture_launch(capture.graph)
    return state, solver


def _tip_sag(state: newton.State) -> float:
    """Return downward free-end displacement in metres."""
    return -float(state.body_q.numpy()[-1, 2])


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX chain tests require CUDA graph capture.",
)
class TestJointChainConvergence(unittest.TestCase):
    """Validate direct mechanism solves on ill-conditioned cantilever chains."""

    def test_revolute_cantilever_holds(self) -> None:
        """Hold a revolute cantilever using only direct equality rows."""
        for layout in _STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model = _build_cantilever(wp.get_preferred_device(), newton.JointType.REVOLUTE)
                state, solver = _rollout(model, layout=layout)
                positions = state.body_q.numpy()[:, :3]
                self.assertTrue(np.isfinite(positions).all(), "non-finite body position")
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (5 * NUM_LINKS,))
                self.assertFalse(np.any(solver.world._joint_pgs_enabled.numpy()))
                sag = _tip_sag(state)
                self.assertLess(
                    sag,
                    _MAX_TIP_SAG,
                    msg=f"revolute cantilever drooped {sag * 1e3:.1f} mm",
                )

    def test_rigid_bend_cable_stays_straight(self) -> None:
        """Keep a rigid-bend cable within one millimetre of straight."""
        for layout in _STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                cable_model = _build_cantilever(wp.get_preferred_device(), newton.JointType.CABLE)
                cable, cable_solver = _rollout(cable_model, layout=layout)
                self.assertTrue(np.isfinite(cable.body_q.numpy()).all(), "non-finite cable pose")
                self.assertEqual(cable_solver._direct_equality_system.topology.dimensions, (6 * NUM_LINKS,))
                self.assertFalse(np.any(cable_solver.world._joint_pgs_enabled.numpy()))
                self.assertLess(
                    _tip_sag(cable),
                    _MAX_RIGID_CABLE_TIP_SAG,
                    "rigid cable bend exceeded the one-millimetre tip-sag budget",
                )

    def test_prismatic_cantilever_holds(self) -> None:
        """Hold a prismatic cantilever using only direct equality rows."""
        for layout in _STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model = _build_cantilever(wp.get_preferred_device(), newton.JointType.PRISMATIC)
                state, solver = _rollout(model, layout=layout)
                positions = state.body_q.numpy()[:, :3]
                self.assertTrue(np.isfinite(positions).all(), "non-finite body position")
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (5 * NUM_LINKS,))
                self.assertFalse(np.any(solver.world._joint_pgs_enabled.numpy()))
                sag = _tip_sag(state)
                self.assertLess(
                    sag,
                    _MAX_TIP_SAG,
                    msg=f"prismatic cantilever drooped {sag * 1e3:.1f} mm",
                )


if __name__ == "__main__":
    unittest.main()
