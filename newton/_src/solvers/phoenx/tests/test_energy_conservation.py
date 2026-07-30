# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Energy and momentum invariants for the production PhoenX path."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81
LEVER = 1.0
MASS = 1.0
INERTIA = 1.0e-3
INITIAL_ANGLE = 0.3
FPS = 240
SUBSTEPS = 5
STEP_LAYOUTS = ("multi_world", "single_world")


def _make_solver(model: newton.Model, layout: str) -> newton.solvers.SolverPhoenX:
    """Create a maximal-coordinate solver with only direct joint rows."""
    return newton.solvers.SolverPhoenX(
        model,
        substeps=SUBSTEPS,
        solver_iterations=1,
        velocity_iterations=0,
        articulation_mode="maximal",
        step_layout=layout,
    )


def _capture_rollout(
    model: newton.Model,
    state: newton.State,
    solver: newton.solvers.SolverPhoenX,
    frames: int,
) -> None:
    """Advance one state through a captured five-substep frame."""
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        state.clear_forces()
        solver.step(state, state, control, None, 1.0 / FPS)
    for _ in range(frames):
        wp.capture_launch(capture.graph)


def _build_pendulum() -> tuple[newton.Model, int]:
    """Build an undamped one-metre revolute pendulum."""
    builder = newton.ModelBuilder(gravity=(0.0, -GRAVITY, 0.0), up_axis=newton.Axis.Y)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=MASS,
        inertia=((INERTIA, 0.0, 0.0), (0.0, INERTIA, 0.0), (0.0, 0.0, INERTIA)),
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform((0.0, LEVER, 0.0), wp.quat_identity()),
        damping=0.0,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=wp.get_preferred_device())
    model.joint_q.assign(np.asarray((INITIAL_ANGLE,), dtype=np.float32))
    return model, body


def _pendulum_energy(state: newton.State, body: int) -> float:
    """Evaluate translational, rotational, and gravitational energy."""
    position = state.body_q.numpy()[body, :3].astype(np.float64)
    velocity = state.body_qd.numpy()[body].astype(np.float64)
    kinetic = 0.5 * MASS * float(velocity[:3] @ velocity[:3])
    kinetic += 0.5 * INERTIA * float(velocity[3:] @ velocity[3:])
    potential = MASS * GRAVITY * (float(position[1]) + LEVER)
    return kinetic + potential


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX invariant tests require CUDA graphs.")
class TestEnergyConservation(unittest.TestCase):
    """Check long-horizon invariants through the public production solver."""

    def test_direct_pendulum_energy_drifts_under_five_percent(self) -> None:
        """Preserve undamped pendulum energy over several periods."""
        for layout in STEP_LAYOUTS:
            with self.subTest(step_layout=layout):
                model, body = _build_pendulum()
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                initial_energy = _pendulum_energy(state, body)
                solver = _make_solver(model, layout)
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (5,))
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                _capture_rollout(model, state, solver, 6 * FPS)
                final_energy = _pendulum_energy(state, body)
                self.assertTrue(math.isfinite(final_energy))
                self.assertLess(abs(final_energy - initial_energy) / initial_energy, 0.05)

    def test_torque_free_body_conserves_momentum_and_energy(self) -> None:
        """Preserve torque-free asymmetric-body invariants with five substeps."""
        inertia_body = np.diag((1.0, 2.0, 2.5))
        omega_initial = np.asarray((1.1, -0.7, 0.9), dtype=np.float64)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = builder.add_link(
            xform=wp.transform_identity(),
            mass=1.0,
            inertia=((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.5)),
        )
        model = builder.finalize(device=wp.get_preferred_device())
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_qd = state.body_qd.numpy()
        body_qd[body, 3:] = omega_initial
        state.body_qd.assign(body_qd)
        solver = _make_solver(model, "multi_world")
        self.assertFalse(solver._direct_equality_system.enabled)

        momentum_initial = inertia_body @ omega_initial
        energy_initial = 0.5 * float(omega_initial @ momentum_initial)
        _capture_rollout(model, state, solver, 4 * FPS)

        quaternion = state.body_q.numpy()[body, 3:].astype(np.float64)
        x, y, z, w = quaternion
        rotation = np.asarray(
            (
                (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )
        omega_final = state.body_qd.numpy()[body, 3:].astype(np.float64)
        inertia_world = rotation @ inertia_body @ rotation.T
        momentum_final = inertia_world @ omega_final
        self.assertLess(np.linalg.norm(momentum_final - momentum_initial) / np.linalg.norm(momentum_initial), 3.0e-4)
        energy_final = 0.5 * float(omega_final @ momentum_final)
        self.assertLess(abs(energy_final - energy_initial) / energy_initial, 3.0e-4)


if __name__ == "__main__":
    unittest.main()
