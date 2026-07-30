# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-solver behavioral tests for maximal-coordinate ball joints."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 5
SOLVER_ITERATIONS = 2
SETTLE_FRAMES = 30
_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_ball_model(
    *,
    center: tuple[float, float, float],
    child_anchor: tuple[float, float, float],
    gravity: tuple[float, float, float],
) -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder(gravity=gravity, up_axis=newton.Axis.Y)
    body = builder.add_link(
        xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
        mass=1.0,
        inertia=_INERTIA,
    )
    joint = builder.add_joint_ball(
        parent=-1,
        child=body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(*child_anchor), wp.quat_identity()),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device()), body


def _make_solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        velocity_iterations=1,
        articulation_mode="maximal",
    )


def _rollout(model: newton.Model, solver: newton.solvers.SolverPhoenX, state: newton.State, frames: int) -> None:
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(frames):
        wp.capture_launch(capture.graph)


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    x, y, z, w = transform[3:]
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return transform[:3] + rotation @ point


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX direct ball-joint tests require CUDA graph capture.",
)
class TestBallSocket(unittest.TestCase):
    """Validate maximal-coordinate ball joints through direct equality rows."""

    def test_anchor_coincidence_under_gravity(self) -> None:
        """Keep the child and world anchors coincident under gravity."""
        model, body = _build_ball_model(
            center=(0.0, -1.0, 0.0),
            child_anchor=(0.0, 1.0, 0.0),
            gravity=(0.0, -GRAVITY, 0.0),
        )
        solver = _make_solver(model)
        self.assertEqual(solver._direct_equality_system.topology.dimensions, (3,))
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        _rollout(model, solver, state, SETTLE_FRAMES)

        anchor = _transform_point(state.body_q.numpy()[body].astype(np.float64), np.asarray((0.0, 1.0, 0.0)))
        self.assertLess(float(np.linalg.norm(anchor)), 2.0e-3)

    def test_reaction_force_matches_weight(self) -> None:
        """Balance unit-body weight with the direct vertical row impulse."""
        model, _body = _build_ball_model(
            center=(0.0, -1.0, 0.0),
            child_anchor=(0.0, 1.0, 0.0),
            gravity=(0.0, -GRAVITY, 0.0),
        )
        solver = _make_solver(model)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        _rollout(model, solver, state, SETTLE_FRAMES)

        direct = solver._direct_equality_system
        impulse = direct.accumulated_impulse.numpy()
        force = impulse * (FPS * SUBSTEPS)
        self.assertTrue(np.isfinite(force).all())
        self.assertAlmostEqual(abs(float(force[1])), GRAVITY, delta=5.0e-3)
        self.assertLess(float(np.max(np.abs(force[[0, 2]]))), 5.0e-3)

    def test_does_not_resist_rotation(self) -> None:
        """Preserve arbitrary spin when the constrained anchor is at the COM."""
        model, body = _build_ball_model(
            center=(0.0, 0.0, 0.0),
            child_anchor=(0.0, 0.0, 0.0),
            gravity=(0.0, 0.0, 0.0),
        )
        solver = _make_solver(model)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        omega = np.asarray((0.7, -0.3, 0.5), dtype=np.float32)
        velocity = state.body_qd.numpy()
        velocity[body, 3:] = omega
        state.body_qd.assign(velocity)
        _rollout(model, solver, state, SETTLE_FRAMES)

        np.testing.assert_allclose(state.body_qd.numpy()[body, 3:], omega, rtol=2.0e-4, atol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
