# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: native preparation erases history at99% of the actual static cone. This asserts the observed premature reset, not acceptance of correct sticking."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    contact_prepare_for_iteration_lean_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@wp.kernel(enable_backward=False)
def _prepare(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    body_count: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    copies: CopyStateContainer,
):
    contact_prepare_for_iteration_lean_no_soft_pd(
        columns, 0, bodies, particles, body_count, wp.float32(1000.0), cc, contacts, copies, 0
    )


class TestRigidFrictionAnchors(unittest.TestCase):
    def test_native_prepare_erases_strictly_inside_static_cone(self):
        """Reset a broken material reference without changing collision witnesses."""
        biases = []
        for mass in (1.0,):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.9999), wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(mass * 0.4, 0.0, 0.0, 0.0, mass * 0.4, 0.0, 0.0, 0.0, mass * 0.4),
            )
            builder.add_shape_sphere(body, radius=1.0, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
            builder.add_ground_plane()
            model = builder.finalize(device="cpu")
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=8, contact_matching="sticky")
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                substeps=1,
                solver_iterations=1,
                velocity_iterations=0,
                sor_boost=1.0,
            )
            state = model.state()
            pipeline.collide(state, contacts)
            solver.step(state, state, model.control(), contacts, 1.0e-6)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
            # Same stored surface drift and proportionally scaled normal load.
            point = contacts.rigid_contact_point0.numpy()
            point[0, 0] += 1.0e-5
            contacts.rigid_contact_point0.assign(point)
            world = solver.world
            anchor_rows = world._contact_container.lambdas.numpy()
            anchor_rows[6:9, 0] = contacts.rigid_contact_point0.numpy()[0]
            anchor_rows[9:12, 0] = contacts.rigid_contact_point1.numpy()[0]
            world._contact_container.lambdas.assign(anchor_rows)
            impulses = world._contact_container.impulses.numpy()
            impulses[:, 0] = [mass * 1.0e-5, mass * 1.0e-5 * 0.99 * world._contact_cols.data.numpy()[3, 0], 0.0]
            world._contact_container.impulses.assign(impulses)
            wp.launch(
                _prepare,
                dim=1,
                inputs=[
                    world._contact_cols,
                    world.bodies,
                    world.particles or ParticleContainer(),
                    world.num_bodies,
                    world._contact_container,
                    world._active_contact_views(),
                    world._copy_state or CopyStateContainer(),
                ],
                device=model.device,
            )
            biases.append(world._contact_container.derived.numpy()[4:6, 0].copy())
        np.testing.assert_allclose(world._contact_container.impulses.numpy()[1:3, 0], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(biases[0], 0.0, atol=1.0e-8)
        np.testing.assert_array_equal(contacts.rigid_contact_point0.numpy(), point)
        # A second prepare at the same pose must not restore the broken drift.
        wp.launch(
            _prepare,
            dim=1,
            inputs=[
                world._contact_cols,
                world.bodies,
                world.particles or ParticleContainer(),
                world.num_bodies,
                world._contact_container,
                world._active_contact_views(),
                world._copy_state or CopyStateContainer(),
            ],
            device=model.device,
        )
        np.testing.assert_allclose(world._contact_container.derived.numpy()[4:6, 0], 0.0, atol=1.0e-8)

        # Slide the dynamic body after the break. The new material reference
        # must recover only this displacement, not the discarded old spring.
        positions = world.bodies.position.numpy()
        shape0 = int(contacts.rigid_contact_shape0.numpy()[0])
        dynamic_body = int(world._active_contact_views().shape_body.numpy()[shape0])
        positions[dynamic_body, 0] += 1.0e-5
        world.bodies.position.assign(positions)
        wp.launch(
            _prepare,
            dim=1,
            inputs=[
                world._contact_cols,
                world.bodies,
                world.particles or ParticleContainer(),
                world.num_bodies,
                world._contact_container,
                world._active_contact_views(),
                world._copy_state or CopyStateContainer(),
            ],
            device=model.device,
        )
        bias = world._contact_container.derived.numpy()[4:6, 0]
        self.assertAlmostEqual(float(np.linalg.norm(bias)), 0.0008, delta=1.0e-7)

        # Re-express the same physical configuration in a rotated, translated
        # world frame. Body-local material references must remain unchanged.
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.7)
        rotated_positions = world.bodies.position.numpy()
        orientations = world.bodies.orientation.numpy()
        for index in range(world.num_bodies):
            rotated_positions[index] = np.asarray(
                wp.quat_rotate(rotation, wp.vec3(*rotated_positions[index]))
            ) + np.array([0.1, -0.2, 0.3], dtype=np.float32)
            orientations[index] = np.asarray(rotation * wp.quat(*orientations[index]))
        world.bodies.position.assign(rotated_positions)
        world.bodies.orientation.assign(orientations)
        rows = world._contact_container.lambdas.numpy()
        rows[:3, 0] = np.asarray(wp.quat_rotate(rotation, wp.vec3(*rows[:3, 0])))
        rows[3:6, 0] = np.asarray(wp.quat_rotate(rotation, wp.vec3(*rows[3:6, 0])))
        world._contact_container.lambdas.assign(rows)
        normals = contacts.rigid_contact_normal.numpy()
        normals[0] = np.asarray(wp.quat_rotate(rotation, wp.vec3(*normals[0])))
        contacts.rigid_contact_normal.assign(normals)
        wp.launch(
            _prepare,
            dim=1,
            inputs=[
                world._contact_cols,
                world.bodies,
                world.particles or ParticleContainer(),
                world.num_bodies,
                world._contact_container,
                world._active_contact_views(),
                world._copy_state or CopyStateContainer(),
            ],
            device=model.device,
        )
        np.testing.assert_allclose(world._contact_container.derived.numpy()[4:6, 0], bias, rtol=0.0, atol=5.0e-6)
