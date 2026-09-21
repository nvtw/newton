# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX integration tests for Newton's standard deformable contacts."""

from __future__ import annotations

import unittest

import warp as wp

import newton


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX deformable contacts require CUDA.")
class TestOfficialDeformableContacts(unittest.TestCase):
    def test_solver_preserves_standard_full_surface_pipeline(self):
        device = wp.get_preferred_device()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.05)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.1, -0.1, 0.055),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=1,
            dim_y=1,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.1,
            particle_radius=0.01,
        )
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="sticky",
            enable_rigid_soft_full_surface_contact=True,
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            step_layout="single_world",
            substeps=2,
            solver_iterations=4,
        )

        state = model.state()
        solver.collide(state, contacts)

        self.assertIs(solver._collision_pipeline, pipeline)
        self.assertEqual(pipeline.extra_shape_count, 0)
        self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)

        state_out = model.state()
        solver.step(state, state_out, None, contacts, 1.0 / 60.0)
        self.assertGreater(int(solver.world._normalized_contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertGreater(int(solver.world._num_active_constraints.numpy()[0]), solver.world._contact_offset)

    def test_shared_soft_surface_tree_feeds_pgs_self_contacts(self):
        device = wp.get_preferred_device()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        vertices = [
            wp.vec3(-0.1, -0.1, 0.0),
            wp.vec3(0.1, -0.1, 0.0),
            wp.vec3(0.0, 0.1, 0.0),
        ]
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices,
            indices=[0, 1, 2],
            density=1.0,
            particle_radius=0.0025,
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=[
                wp.vec3(0.0, 0.0, 0.002),
                wp.vec3(0.15, 0.0, 0.1),
                wp.vec3(-0.15, 0.0, 0.1),
            ],
            indices=[0, 1, 2],
            density=1.0,
            particle_radius=0.0025,
        )
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverPhoenX(
            model,
            step_layout="single_world",
            substeps=1,
            solver_iterations=2,
        )
        pipeline = solver._collision_pipeline
        self.assertTrue(solver._deformable_self_contact_enabled)
        contacts = pipeline.contacts()
        state = model.state()
        solver.collide(state, contacts)

        collision_info = contacts.soft_self_contact_data
        self.assertIsNotNone(collision_info)
        detected = int(collision_info.vertex_colliding_triangles_count.numpy().sum())
        detected += int(collision_info.edge_colliding_edges_count.numpy().sum())
        self.assertGreater(detected, 0, f"q={state.particle_q.numpy()} worlds={model.particle_world.numpy()}")

        state_out = model.state()
        solver.step(state, state_out, None, contacts, 1.0 / 60.0)
        self.assertGreater(int(solver.world._normalized_contacts.rigid_contact_count.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
