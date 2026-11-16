# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check temporal integration with bounded colors and overflow mass splitting."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverPhoenX


@unittest.skipUnless(wp.is_cuda_available(), "Overflow-only dispatch requires CUDA")
class TestTemporalOverflow(unittest.TestCase):
    @staticmethod
    def _solver(model, pipeline=None, colors=0):
        return SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            substeps=1,
            solver_iterations=24,
            velocity_iterations=1,
            step_layout="single_world",
            joint_mode="maximal_pgs",
            temporal_iterations=True,
            contact_hertz=80,
            sleeping_velocity_threshold=0,
            mass_splitting=True,
            mass_splitting_overflow_only=True,
            mass_splitting_unrolled=True,
            mass_splitting_batch_size=1,
            max_colored_partitions=colors,
            partitioner_algorithm="greedy" if colors == 0 else "endpoint_owner",
        )

    def test_free_fall_and_external_force(self):
        """Apply every fractional gravity and external-force share without contacts."""
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0, 0, 10.0)), mass=2.0, inertia=wp.mat33(np.eye(3)))
            builder.add_shape_sphere(body, radius=0.1, cfg=builder.ShapeConfig(density=0))
            model = builder.finalize()
            state, control = model.state(), model.control()
            solver = self._solver(model)
            state.body_f.assign(np.array([[4, 0, 0, 0, 0, 0]], dtype=np.float32))
            with wp.ScopedCapture() as capture:
                solver.step(state, state, control, None, 1 / 120)
            for _ in range(60):
                wp.capture_launch(capture.graph)
            qd = state.body_qd.numpy()[0]
            np.testing.assert_allclose(qd[:3], [1.0, 0, -4.905], rtol=0, atol=2e-4)
            # No contact sweeps means the existing outer-step integration policy applies.
            self.assertTrue(np.isfinite(state.body_q.numpy()).all())

    def test_contact_force_matches_momentum_with_overflow(self):
        """Conserve total momentum with both all-overflow and mixed partitions."""
        for colors in (0, 1):
            with self.subTest(colors=colors):
                self._check_contact_force(colors)

    def test_sliding_distance_matches_coulomb_friction(self):
        """Stop a sliding block within five percent of its analytic distance."""
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(density=0, mu=0.3, mu_static=0.3, gap=0.002, margin=0)
            body = builder.add_body(
                xform=wp.transform((0, 0, 0.05)),
                mass=1.0,
                inertia=wp.mat33(np.diag([0.17 / 12, 0.17 / 12, 0.32 / 12])),
            )
            builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.05, cfg=cfg)
            builder.add_ground_plane(cfg=cfg)
            model = builder.finalize()
            state, control = model.state(), model.control()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=64)
            contacts = pipeline.contacts()
            solver = self._solver(model, pipeline)
            with wp.ScopedCapture() as capture:
                pipeline.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            for _ in range(120):
                wp.capture_launch(capture.graph)
            start = float(state.body_q.numpy()[0, 0])
            state.body_qd.assign(np.array([[2, 0, 0, 0, 0, 0]], dtype=np.float32))
            for _ in range(120):
                wp.capture_launch(capture.graph)
            distance = float(state.body_q.numpy()[0, 0]) - start
            expected = 2**2 / (2 * 0.3 * 9.81)
            self.assertAlmostEqual(distance / expected, 1.0, delta=0.05)
            self.assertLess(float(np.linalg.norm(state.body_qd.numpy()[0, :3])), 0.005)

    def test_joint_overflow_preserves_spatial_momentum(self):
        """Preserve linear and angular momentum through temporal joint-copy solves."""
        # The legacy fixture package requires CUDA at import time.
        from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model  # noqa: PLC0415
        from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum  # noqa: PLC0415

        with wp.ScopedDevice("cuda:0"):
            model = make_model(0.0)
            solver = self._solver(model)
            state, control = model.state(), model.control()
            velocity = state.body_qd.numpy()
            velocity[0] = (0.2, -0.1, 0.3, 0.1, -0.2, 0.4)
            velocity[1] = (-0.3, 0.4, -0.2, -0.3, 0.1, -0.5)
            state.body_qd.assign(velocity)
            initial = _total_momentum(model, state)
            with wp.ScopedCapture() as capture:
                solver.step(state, state, control, None, 0.005)
            for _ in range(4):
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(_total_momentum(model, state), initial, rtol=0, atol=3e-6)
            self.assertGreater(solver.step_report().overflow_size, 0)

    def _check_contact_force(self, colors):
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(density=0, mu=0.4, mu_static=0.4, gap=0.002, margin=0)
            plate = builder.add_body(
                xform=wp.transform((0, 0, 0.02)), mass=0.2, inertia=wp.mat33(np.diag([0.005, 0.015, 0.02]))
            )
            builder.add_shape_box(plate, hx=0.4, hy=0.2, hz=0.02, cfg=cfg)
            for x in (-0.24, 0, 0.24):
                body = builder.add_body(
                    xform=wp.transform((x, 0, 0.14)), mass=1.0, inertia=wp.mat33(np.eye(3) * (0.2**2 / 6))
                )
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            ground = builder.add_ground_plane(cfg=cfg)
            model = builder.finalize()
            state, control = model.state(), model.control()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=256)
            contacts = pipeline.contacts()
            solver = self._solver(model, pipeline, colors=colors)
            velocities = state.body_qd.numpy()
            velocities[1:, 0] = 0.2
            state.body_qd.assign(velocities)
            observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            mass = model.body_mass.numpy().astype(np.float64)
            weight = mass.sum() * 9.81
            with wp.ScopedCapture() as capture:
                pipeline.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120, observables=observables)
            residuals, forces = [], []
            for frame in range(240):
                before = mass @ state.body_qd.numpy()[:, :3]
                wp.capture_launch(capture.graph)
                after = mass @ state.body_qd.numpy()[:, :3]
                count = int(contacts.rigid_contact_count.numpy()[0])
                s0 = contacts.rigid_contact_shape0.numpy()[:count]
                s1 = contacts.rigid_contact_shape1.numpy()[:count]
                force_rows = observables.contact_f.numpy()[:count, :3]
                force = force_rows[s1 == ground].sum(axis=0) - force_rows[s0 == ground].sum(axis=0)
                self.assertGreaterEqual(float(force[2]), -1e-6)
                residuals.append(force - np.array([0, 0, weight]) - (after - before) * 120)
                if frame >= 180:
                    forces.append(float(force[2]))
            self.assertGreater(solver.step_report().overflow_size, 0)
            if colors:
                starts = solver.world._partitioner.color_starts.numpy()
                self.assertGreater(int(starts[colors] - starts[0]), 0)
            self.assertLess(float(np.max(np.abs(residuals))), 0.002)
            self.assertAlmostEqual(float(np.mean(forces)), weight, delta=0.01 * weight)
            self.assertGreater(float(state.body_q.numpy()[plate, 2]), 0.01)


if __name__ == "__main__":
    unittest.main()
