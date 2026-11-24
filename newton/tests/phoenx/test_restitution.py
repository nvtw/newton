# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check rigid impact and resting contacts in the same PhoenX solve."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverPhoenX


class TestRestitution(unittest.TestCase):
    def test_fast_separated_contact_preserves_approach_velocity(self):
        """Allow fast motion when the body cannot reach the surface this step."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        cfg = builder.ShapeConfig(gap=0.05, mu=0.0, mu_static=0.0)
        builder.add_ground_plane(cfg=cfg)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.14)))
        builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model, rigid_contact_max=8, contact_matching="sticky", speculative_contact_gap_max=0.055
        )
        solver = SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
            step_layout="single_world",
        )
        state, result = model.state(), model.state()
        velocity = state.body_qd.numpy()
        velocity[body, 2] = -20.0
        state.body_qd.assign(velocity)
        contacts = pipeline.contacts()
        dt = 0.001

        pipeline.collide(state, contacts, dt=dt)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.step(state, result, model.control(), contacts, dt)
        self.assertAlmostEqual(float(result.body_qd.numpy()[body, 2]), -20.0, places=5)
        self.assertAlmostEqual(float(result.body_q.numpy()[body, 2]), 0.12, places=6)

    def test_mixed_restitution_rebound_and_rest(self):
        """Recover the pair's restitution on impact while a resting bouncy sphere stays supported."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            builder.add_ground_plane(cfg=builder.ShapeConfig(restitution=0.0))
            radius = 0.1
            height = 0.5
            for x, z, coefficient in ((-1, height, 1.0), (0, height, 0.0), (1, 0, 1.0)):
                body = builder.add_body(xform=wp.transform(wp.vec3(x, 0, z + radius), wp.quat_identity()))
                builder.add_shape_sphere(body, radius=radius, cfg=builder.ShapeConfig(restitution=coefficient))
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = SolverPhoenX(model, collision_pipeline=pipeline)
            state, next_state = model.state(), model.state()
            contacts = pipeline.contacts()
            control = model.control()
            velocities = []
            for _ in range(120):
                solver.collide(state, contacts)
                solver.step(state, next_state, control, contacts, 1.0 / 240.0)
                state, next_state = next_state, state
                velocities.append(state.body_qd.numpy()[:, 2].copy())
            final_heights = state.body_q.numpy()[:, 2].copy()

        vz = np.array(velocities)
        # Shape 0 has e=1 and the floor has e=0, so their average gives e=0.5.
        np.testing.assert_allclose(vz[:, 0].max(), 0.5 * np.sqrt(2 * 9.81 * height), atol=0.12)
        self.assertLess(vz[:, 1].max(), 0.12)
        self.assertLess(np.abs(vz[:, 2]).max(), 0.12)
        np.testing.assert_allclose(final_heights[1:], radius, atol=0.005)

    def test_loaded_stack_sleeps(self):
        """A stack of touching boxes with restitution settles under its own weight."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            material = builder.ShapeConfig(mu=0.5, restitution=0.4, density=500.0, gap=0.003)
            builder.add_ground_plane(cfg=material)
            for layer in range(4):
                pose = wp.transform(wp.vec3(0.0, 0.0, 0.1 + layer * 0.2), wp.quat_identity())
                body = builder.add_body(xform=pose)
                builder.add_shape_box(body, hx=0.18, hy=0.14, hz=0.1, cfg=material)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=64)
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                temporal_iterations=True,
                solver_iterations=12,
                sleeping_velocity_threshold=0.01,
                sleeping_frames_required=60,
            )
            state, next_state = model.state(), model.state()
            initial = state.body_q.numpy()[:, :3].copy()
            contacts = pipeline.contacts()
            for _ in range(120):
                solver.collide(state, contacts)
                solver.step(state, next_state, None, contacts, 1.0 / 120.0)
                state, next_state = next_state, state
            asleep = solver.bodies.island_root.numpy()[1 : model.body_count + 1] >= 0
            self.assertTrue(np.all(asleep), f"Sleeping stack bodies: {asleep.tolist()}")
            np.testing.assert_allclose(state.body_q.numpy()[:, :3], initial, atol=0.001)

    def test_swept_eccentric_impact(self):
        """An impulse changes terminal velocity immediately and transports the new motion only after impact."""
        with wp.ScopedDevice("cpu"):
            for substeps, relax, sweeps, material_arrays, acceleration in (
                (1, 1, 0, False, 0.0),
                (4, 0, 0, False, 0.0),
                (1, 1, 4, True, -8.0),
                (3, 1, 2, False, 0.0),
            ):
                with self.subTest(substeps=substeps, relax=relax, sweeps=sweeps, material_arrays=material_arrays):
                    dt, fraction, speed, coefficient = 1.0 / 240.0, 0.65, 4.0, 0.8
                    radius, offset, inertia = 0.08, 0.2, 0.125
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    # The analytic bounce is frictionless: no sliding, torsional, or rolling friction.
                    material = builder.ShapeConfig(
                        density=0.0,
                        mu=0.0,
                        mu_torsional=0.0,
                        mu_rolling=0.0,
                        restitution=coefficient,
                        gap=0.05,
                        margin=0.0,
                    )
                    builder.add_ground_plane(cfg=material)
                    body = builder.add_body(
                        mass=1.0,
                        inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
                        lock_inertia=True,
                        xform=wp.transform(
                            wp.vec3(
                                0.0, 0.0, radius + fraction * speed * dt - 0.5 * acceleration * (fraction * dt) ** 2
                            ),
                            wp.quat_identity(),
                        ),
                    )
                    builder.add_shape_sphere(
                        body,
                        xform=wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity()),
                        radius=radius,
                        cfg=material,
                    )
                    model = builder.finalize()
                    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
                    temporal = (
                        {
                            "temporal_iterations": True,
                            "solver_iterations": sweeps,
                            "mass_splitting": False,
                            "step_layout": "single_world",
                        }
                        if sweeps
                        else {}
                    )
                    solver = SolverPhoenX(
                        model, collision_pipeline=pipeline, substeps=substeps, velocity_iterations=relax, **temporal
                    )
                    initial, final = model.state(), model.state()
                    initial.body_qd.assign(np.array([[0.0, 0.0, -speed, 0.0, 0.0, 0.0]], dtype=np.float32))
                    initial.body_f.assign(np.array([[0.0, 0.0, acceleration, 0.0, 0.0, 0.0]], dtype=np.float32))
                    contacts = (
                        newton.Contacts(
                            pipeline.rigid_contact_max,
                            pipeline.soft_contact_max,
                            per_contact_shape_properties=True,
                            contact_matching=True,
                        )
                        if material_arrays
                        else pipeline.contacts()
                    )
                    solver.collide(initial, contacts)
                    solver.step(initial, final, None, contacts, dt)

                    impulse = (
                        (1.0 + coefficient) * (speed - acceleration * fraction * dt) / (1.0 + offset * offset / inertia)
                    )
                    vz, wy = impulse - speed + acceleration * dt, -offset * impulse / inertia
                    remaining = (1.0 - fraction) * dt
                    expected_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wy * remaining)
                    np.testing.assert_allclose(
                        final.body_q.numpy()[0, 2],
                        radius + vz * remaining - 0.5 * acceleration * remaining**2,
                        atol=2e-6 + abs(acceleration) * dt**2 / (2 * max(1, sweeps) * substeps),
                    )
                    np.testing.assert_allclose(final.body_q.numpy()[0, 3:], np.asarray(expected_q), atol=2e-5)
                    np.testing.assert_allclose(final.body_qd.numpy()[0], (0.0, 0.0, vz, 0.0, wy, 0.0), atol=2e-4)
