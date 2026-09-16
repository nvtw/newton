# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Conservation with velocity-derived search gaps and ordinary Phoenx contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum


class TestRigidSpeculativeMomentum(unittest.TestCase):
    def test_predictive_impact_conserves_momentum_and_dissipates_energy(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        inertia = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.0)
        for center, direction in ((-0.50025, 1.0), (0.50025, -1.0)):
            body = builder.add_body(xform=wp.transform(wp.vec3(center, 0.0, 0.0)), mass=1.0, inertia=inertia)
            builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
            builder.body_qd[body] = (0.5 * direction, 0.5 * direction, 0.0, 0.0, 0.0, 0.0)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model, rigid_contact_max=8, contact_matching="sticky", speculative_contact_gap_max=0.005
        )
        contacts = pipeline.contacts()
        dt = 1.0 / 120.0
        original_velocity = state.body_qd.numpy()
        state.body_qd.zero_()
        pipeline.collide(state, contacts, dt=dt)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        np.testing.assert_array_equal(pipeline._shape_search_gap.numpy(), model.shape_gap.numpy())
        state.body_qd.assign(original_velocity)
        pipeline.collide(state, contacts, dt=dt)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            step_layout="single_world",
            substeps=30,
            solver_iterations=1,
            velocity_iterations=0,
            sor_boost=1.0,
        )

        def energy():
            total = 0.0
            q, qd = state.body_q.numpy(), state.body_qd.numpy()
            for i in range(2):
                omega = np.asarray(wp.quat_rotate_inv(wp.quat(*q[i, 3:]), wp.vec3(*qd[i, 3:])))
                total += 0.5 * (qd[i, :3] @ qd[i, :3] + omega @ np.asarray(inertia).reshape(3, 3) @ omega)
            return total

        initial_energy = energy()
        initial_momentum = _total_momentum(model, state)
        for _ in range(10):
            state.clear_forces()
            pipeline.collide(state, contacts, dt=dt)
            solver.step(state, state, model.control(), contacts, dt)
            np.testing.assert_allclose(_total_momentum(model, state), initial_momentum, rtol=0.0, atol=2.0e-5)
            self.assertLessEqual(energy(), initial_energy + 2.0e-5)
            np.testing.assert_array_equal(model.shape_gap.numpy(), np.zeros(model.shape_count))
        self.assertLess(energy(), initial_energy - 1.0e-4)


if __name__ == "__main__":
    unittest.main()
