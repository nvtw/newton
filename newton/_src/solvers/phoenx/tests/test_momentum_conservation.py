# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical reaction-force tests for direct PhoenX joint mechanisms."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

NUM_BODIES = 10
HALF_LENGTH = 0.5
GRAVITY = 9.81
FPS = 60
SUBSTEPS = 5
SETTLE_FRAMES = 120
STEP_LAYOUTS = ("multi_world", "single_world")
INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_equilibrium_chain() -> newton.Model:
    """Build a vertical chain whose ball-socket reactions are known exactly."""
    builder = newton.ModelBuilder(gravity=(0.0, -GRAVITY, 0.0), up_axis=newton.Axis.Y)
    bodies = [
        builder.add_link(
            xform=wp.transform((0.0, -(index + HALF_LENGTH), 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=INERTIA,
        )
        for index in range(NUM_BODIES)
    ]
    joints: list[int] = []
    for index, child in enumerate(bodies):
        parent = -1 if index == 0 else bodies[index - 1]
        parent_xform = (
            wp.transform_identity() if index == 0 else wp.transform((0.0, -HALF_LENGTH, 0.0), wp.quat_identity())
        )
        joints.append(
            builder.add_joint_ball(
                parent=parent,
                child=child,
                parent_xform=parent_xform,
                child_xform=wp.transform((0.0, HALF_LENGTH, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=wp.get_preferred_device())


def _settle(layout: str) -> newton.solvers.SolverPhoenX:
    """Settle the chain with a five-substep CUDA graph."""
    model = _build_equilibrium_chain()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=1,
        velocity_iterations=0,
        articulation_mode="maximal",
        step_layout=layout,
    )
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(SETTLE_FRAMES):
        wp.capture_launch(capture.graph)
    return solver


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX reaction tests require CUDA graphs.")
class TestMomentumConservation(unittest.TestCase):
    """Validate direct ball-socket reactions in static chain equilibrium."""

    def test_hanging_chain_reaction_forces(self) -> None:
        """Support every downstream body with the analytical joint force."""
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                solver = _settle(layout)
                direct = solver._direct_equality_system
                self.assertEqual(direct.topology.dimensions, (3 * NUM_BODIES,))
                np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), np.zeros(NUM_BODIES))

                forces = direct.accumulated_impulse.numpy().reshape(NUM_BODIES, 3) * (FPS * SUBSTEPS)
                self.assertTrue(np.isfinite(forces).all())
                expected_y = np.arange(NUM_BODIES, 0, -1, dtype=np.float32) * GRAVITY
                np.testing.assert_allclose(np.abs(forces[:, 1]), expected_y, rtol=0.0, atol=5.0e-3)
                np.testing.assert_allclose(forces[:, (0, 2)], 0.0, rtol=0.0, atol=5.0e-3)


if __name__ == "__main__":
    unittest.main()
