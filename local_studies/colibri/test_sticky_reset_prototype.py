# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Physical invariance of rigid contact positional recovery."""

import unittest

import numpy as np
import warp as wp

import newton
from local_studies.colibri.prototype_sticky_prepare import (
    contact_prepare_for_iteration_lean_no_soft_pd,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer, ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
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


class TestStickyResetPrototype(unittest.TestCase):
    def test_broken_friction_anchor_starts_without_positional_error(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
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
            model = builder.finalize(device="cuda:0")
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
            impulses[:, 0] = [mass * 1.0e-5, mass * 1.0e-5, 0.0]
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


if __name__ == "__main__":
    unittest.main()
