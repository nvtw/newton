# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world validation for PhoenX direct mechanisms."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

FPS = 60
SUBSTEPS = 5


def _build_direct_pendulums(
    *,
    world_count: int,
    mechanisms_per_world: int = 1,
    initial_angle: float = 0.0,
    gravity: list[tuple[float, float, float]] | None = None,
) -> tuple[newton.Model, list[int]]:
    """Build independent ball-socket mechanisms in public Newton worlds."""
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0), up_axis=newton.Axis.Y)
    bodies: list[int] = []
    initial_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), initial_angle)
    for world in range(world_count):
        builder.begin_world(label=f"world_{world}")
        for mechanism in range(mechanisms_per_world):
            x = 0.1 * mechanism
            body = builder.add_link(
                xform=wp.transform_identity(),
                mass=1.0,
                inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            )
            joint = builder.add_joint_ball(
                parent=-1,
                child=body,
                parent_xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()),
            )
            builder.add_articulation([joint])
            builder.joint_q[-4:] = initial_rotation
            bodies.append(body)
        builder.end_world()
    model = builder.finalize(device=wp.get_preferred_device())
    if gravity is not None:
        for world, value in enumerate(gravity):
            model.set_gravity(value, world=world)
    return model, bodies


def _make_direct_solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    """Create the five-substep production maximal-coordinate solver."""
    return newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=1,
        velocity_iterations=0,
        articulation_mode="maximal",
        step_layout="multi_world",
    )


def _rollout(
    model: newton.Model,
    solver: newton.solvers.SolverPhoenX,
    *,
    frames: int,
    initial_angular_velocity: np.ndarray | None = None,
) -> newton.State:
    """Advance all worlds with one captured frame graph."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    if initial_angular_velocity is not None:
        body_qd = state.body_qd.numpy()
        body_qd[:, 3:] = initial_angular_velocity
        state.body_qd.assign(body_qd)
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(frames):
        wp.capture_launch(capture.graph)
    return state


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX multi-world tests require CUDA graphs.")
class TestPhoenXMultiWorld(unittest.TestCase):
    """Check direct-mechanism isolation and per-world behavior."""

    def test_multiple_mechanisms_per_world_are_direct(self) -> None:
        """Detect every independent mechanism in every public world."""
        model, _bodies = _build_direct_pendulums(world_count=4, mechanisms_per_world=3)
        solver = _make_direct_solver(model)
        self.assertEqual(solver._direct_equality_system.topology.dimensions, (3,) * 12)
        np.testing.assert_array_equal(solver.world._joint_pgs_enabled.numpy(), np.zeros(12))
        state = _rollout(model, solver, frames=20)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

    def test_per_world_initial_state_does_not_leak(self) -> None:
        """Preserve distinct free spins across eight direct mechanisms."""
        model, bodies = _build_direct_pendulums(
            world_count=8,
            gravity=[(0.0, 0.0, 0.0)] * 8,
        )
        solver = _make_direct_solver(model)
        initial = np.zeros((len(bodies), 3), dtype=np.float32)
        initial[:, 1] = np.arange(1, len(bodies) + 1, dtype=np.float32)
        state = _rollout(model, solver, frames=30, initial_angular_velocity=initial)
        measured = state.body_qd.numpy()[bodies, 4]
        np.testing.assert_allclose(measured, initial[:, 1], rtol=2.0e-3, atol=2.0e-3)

    def test_per_world_gravity_changes_swing_rate(self) -> None:
        """Produce earth, moon, and zero-gravity pendulum responses independently."""
        gravity = [(0.0, -9.81, 0.0), (0.0, -1.62, 0.0), (0.0, 0.0, 0.0)]
        model, bodies = _build_direct_pendulums(world_count=3, initial_angle=0.3, gravity=gravity)
        state = _rollout(model, _make_direct_solver(model), frames=30)
        speeds = np.linalg.norm(state.body_qd.numpy()[bodies, 3:], axis=1)
        self.assertGreater(float(speeds[0]), 1.5 * float(speeds[1]))
        self.assertGreater(float(speeds[1]), 10.0 * float(speeds[2]) + 1.0e-5)

    def test_many_worlds_remain_identical(self) -> None:
        """Keep 256 independently solved worlds bitwise-close and finite."""
        model, bodies = _build_direct_pendulums(world_count=256, initial_angle=0.2)
        solver = _make_direct_solver(model)
        self.assertEqual(len(solver._direct_equality_system.topology.dimensions), 256)
        state = _rollout(model, solver, frames=8)
        positions = state.body_q.numpy()[bodies, :3]
        self.assertTrue(np.isfinite(positions).all())
        np.testing.assert_allclose(positions, np.broadcast_to(positions[0], positions.shape), rtol=0.0, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
