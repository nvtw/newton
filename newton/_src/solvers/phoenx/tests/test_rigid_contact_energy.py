# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Energy and momentum checks for ordinary rigid Coulomb contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum


def _make_ordinary_contact(*, friction=10.0):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    inertia = wp.mat33(0.5005, 0.4995, 0.0, 0.4995, 0.5005, 0.0, 0.0, 0.0, 1.0)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=friction, gap=0.001)
    for center in (0.5, 1.5000001):
        body = builder.add_body(mass=1.0, inertia=inertia)
        builder.add_shape_sphere(
            body, radius=0.5, xform=wp.transform(wp.vec3(center, 1.0, -2.0), wp.quat_identity()), cfg=cfg
        )
    model = builder.finalize(device="cuda:0")
    state = model.state()
    speed = state.body_qd.numpy().reshape(-1)
    speed[:3] = 0.5
    speed[6:9] = -0.5
    state.body_qd.assign(speed.reshape(2, 6))
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8, contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        joint_mode="maximal_direct",
        step_layout="single_world",
        substeps=1,
        solver_iterations=1,
        velocity_iterations=0,
        sor_boost=1.0,
    )
    return model, state, pipeline, contacts, solver, inertia


class TestRigidContactEnergy(unittest.TestCase):
    def test_ordinary_rigid_friction_does_not_add_energy(self):
        """Dissipate contact energy while conserving both momenta."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        model, state, pipeline, contacts, solver, inertia = _make_ordinary_contact()

        def energy():
            q = state.body_q.numpy()
            qd = state.body_qd.numpy()
            total = 0.0
            for index in range(2):
                omega = np.asarray(wp.quat_rotate_inv(wp.quat(*q[index, 3:]), wp.vec3(*qd[index, 3:])))
                total += 0.5 * (qd[index, :3] @ qd[index, :3] + omega @ np.asarray(inertia).reshape(3, 3) @ omega)
            return total

        before = energy()
        momentum_before = _total_momentum(model, state)
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.step(state, state, model.control(), contacts, 1.0e-4)
        self.assertLessEqual(energy(), before + 1.0e-7)
        np.testing.assert_allclose(_total_momentum(model, state), momentum_before, rtol=0.0, atol=2.0e-6)
        self._last_velocity = state.body_qd.numpy()

    def test_sustained_rigid_friction_conserves_momentum(self):
        """Keep total linear and angular momentum through repeated contact updates."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        model, state, pipeline, contacts, solver, _ = _make_ordinary_contact()
        before = _total_momentum(model, state)
        control = model.control()
        with wp.ScopedCapture(device=model.device) as capture:
            state.clear_forces()
            pipeline.collide(state, contacts)
            solver.step(state, state, control, contacts, 1.0e-4)
        for _ in range(100):
            wp.capture_launch(capture.graph)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0.0, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
