# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Reported contact forces must balance the momentum a step changes."""

import unittest

import numpy as np
import warp as wp

import newton

_GRAVITY = 9.81  # [m/s^2]
_HALF = 0.1  # [m]


def _drop_box(substeps: int, frames: int):
    """Drop a box onto the ground; return its mass, step ``dt``, the reported ground force and the box velocity."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()))
    builder.add_shape_box(box, hx=_HALF, hy=_HALF, hz=_HALF, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))
    model = builder.finalize(device="cuda:0")
    model.request_contact_attributes("force")
    state_0, state_1, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, substeps=substeps, solver_iterations=6)
    dt = 1.0 / 60.0
    with wp.ScopedCapture() as capture:
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)

    shape_body = model.shape_body.numpy()
    forces, velocities = [], [float(state_0.body_qd.numpy()[box, 2])]
    for _ in range(frames):
        wp.capture_launch(capture.graph)
        solver.update_contacts(contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
        force_z = contacts.force.numpy()[:count, 2]  # on shape0's body
        on_box = np.where(body0 == box, force_z, -force_z)
        forces.append(float(on_box.sum()))
        velocities.append(float(state_0.body_qd.numpy()[box, 2]))
    mass = float(model.body_mass.numpy()[box])
    return mass, dt, np.asarray(forces), np.asarray(velocities)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX tests run on CUDA with graph capture")
class TestContactForceStepAverage(unittest.TestCase):
    def test_impact_force_balances_momentum(self):
        """Each step's reported ground force carries the box's momentum change beyond gravity.

        The box lands within the first frames and comes to rest; the contact
        impulse varies across the substeps of the landing steps, so the force of
        the last substep alone does not balance them.
        """
        mass, dt, forces, velocities = _drop_box(substeps=4, frames=40)
        expected = mass * (np.diff(velocities) / dt + _GRAVITY)
        self.assertGreater(float(expected.max()), 2.0 * mass * _GRAVITY, "the box never landed")
        error = np.abs(forces - expected)
        self.assertLess(float(error.max()), 1.0e-3 * float(expected.max()), f"worst step error {error.max():.4f} N")
        # At rest the ground carries the weight.
        self.assertAlmostEqual(forces[-1] / (mass * _GRAVITY), 1.0, delta=1.0e-4)


if __name__ == "__main__":
    unittest.main()
