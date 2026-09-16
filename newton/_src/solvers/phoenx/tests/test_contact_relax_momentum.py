# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Post-integration contact relaxation must conserve internal momentum."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import inertia_sym6_pack_np, inertia_sym6_unpack_np
from newton._src.solvers.phoenx.tests.test_rigid_normal_first import CopyStateContainer, ParticleContainer, _make_sweep
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


class TestContactRelaxMomentum(unittest.TestCase):
    def test_packed_relaxation_preserves_current_solve_history(self):
        """Rebase live colored rows without gathering stale canonical impulses."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        scene = _PhoenXScene(
            substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
            step_layout="single_world",
            mass_splitting=True,
            colored_contact_headers=True,
            colored_contact_rows=True,
        )
        scene.add_ground_plane()
        scene.add_box(position=(0.0, 0.0, 0.49), half_extents=(0.5, 0.5, 0.5), mass=1.0)
        scene.finalize()
        scene.step()
        world = scene.world
        current = world._contact_container_solve
        values = current.impulses.numpy()
        values[:, :4] = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.01
        current.impulses.assign(values)
        references = current.lambdas.numpy().copy()
        self.assertFalse(np.array_equal(values, world._contact_container.impulses.numpy()))
        world._refresh_owned_relax_geometry(wp.float32(60.0))
        np.testing.assert_array_equal(current.impulses.numpy(), values)
        np.testing.assert_array_equal(current.lambdas.numpy(), references)

    def test_free_pair_relaxation_uses_one_world_point(self):
        """Move COMs after prepare without moving the common impulse point."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        for split in (False, True):
            with self.subTest(mass_splitting=split):
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                for x in (-0.999, 0.999):
                    body = builder.add_body(
                        xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                        mass=1.0,
                        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
                    )
                    builder.add_shape_sphere(body, radius=1.0, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
                model = builder.finalize(device="cuda:0")
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=8, contact_matching="sticky")
                contacts = pipeline.contacts()
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    step_layout="single_world",
                    substeps=1,
                    solver_iterations=1,
                    velocity_iterations=1,
                    sor_boost=1.0,
                    mass_splitting=split,
                )
                state = model.state()
                pipeline.collide(state, contacts)
                solver.step(state, state, model.control(), contacts, 1.0e-6)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
                world = solver.world
                cc = world._contact_container
                positions = world.bodies.position.numpy()
                world.bodies.position_prev_substep.assign(positions)
                positions[1, 1] += 0.025
                positions[2, 1] -= 0.025
                world.bodies.position.assign(positions)
                # Rotate anisotropic bodies too. The prepared impulse point
                # remains in world space; only its COM-relative levers change.
                rotations = world.bodies.orientation.numpy()
                inverse_inertia = inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy())
                for body, angle in ((1, 0.3), (2, -0.2)):
                    q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), angle)
                    rotations[body] = np.asarray(q)
                    rotation = np.asarray(wp.quat_to_matrix(q)).reshape(3, 3)
                    inverse_inertia[body] = rotation @ np.diag([1.0, 0.5, 1.0 / 3.0]) @ rotation.T
                world.bodies.orientation.assign(rotations)
                world.bodies.inverse_inertia_world.assign(inertia_sym6_pack_np(inverse_inertia))
                velocities = np.zeros((world.num_bodies, 3), dtype=np.float32)
                velocities[1] = [1.0, 0.3, 0.0]
                velocities[2] = [-1.0, -0.3, 0.0]
                world.bodies.velocity.assign(velocities)
                world.bodies.angular_velocity.zero_()
                cc.impulses.zero_()
                references = cc.lambdas.numpy()[6:12].copy()
                points0 = contacts.rigid_contact_point0.numpy().copy()
                points1 = contacts.rigid_contact_point1.numpy().copy()
                derived = cc.derived.numpy()
                derived[3, :] = 0.0
                cc.derived.assign(derived)
                masses = 1.0 / world.bodies.inverse_mass.numpy()[1:]
                inertia = np.linalg.inv(
                    inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()[1:]).astype(np.float64)
                )

                def momentum(world=world, masses=masses, positions=positions, inertia=inertia):
                    v = world.bodies.velocity.numpy()[1:].astype(np.float64)
                    w = world.bodies.angular_velocity.numpy()[1:].astype(np.float64)
                    linear = masses[:, None] * v
                    return linear.sum(axis=0), (
                        np.cross(positions[1:], linear) + np.einsum("bij,bj->bi", inertia, w)
                    ).sum(axis=0)

                before = momentum()
                world._refresh_owned_relax_geometry(wp.float32(1000.0))
                prepared = cc.derived.numpy()
                mapping = world._active_contact_views().shape_body.numpy()
                b0 = int(mapping[int(contacts.rigid_contact_shape0.numpy()[0])])
                b1 = int(mapping[int(contacts.rigid_contact_shape1.numpy()[0])])
                r0, r1 = prepared[9:12, 0], prepared[12:15, 0]
                np.testing.assert_allclose(positions[b0] + r0, positions[b1] + r1, rtol=0.0, atol=1.0e-6)
                normal = cc.lambdas.numpy()[:3, 0]
                tangent = cc.lambdas.numpy()[3:6, 0]
                for row, direction in enumerate((normal, tangent, np.cross(normal, tangent))):
                    a0, a1 = np.cross(r0, direction), np.cross(r1, direction)
                    mobility = 2.0 + a0 @ inverse_inertia[b0] @ a0 + a1 @ inverse_inertia[b1] @ a1
                    self.assertAlmostEqual(float(prepared[row, 0]), 1.0 / mobility, delta=1.0e-6)
                # This single pair has no overflow copies; exercise the same
                # ordinary projection after each world's central refresh.
                wp.launch(
                    _make_sweep(fast=True, bias=False, pd=False),
                    dim=1,
                    inputs=[
                        world._contact_cols,
                        world.bodies,
                        world.particles or ParticleContainer(),
                        world.num_bodies,
                        cc,
                        world._active_contact_views(),
                        world._copy_state or CopyStateContainer(),
                    ],
                    device=model.device,
                )
                after = momentum()
                self.assertGreater(float(np.max(np.abs(cc.impulses.numpy()))), 0.1)
                np.testing.assert_allclose(after[0], before[0], rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(after[1], before[1], rtol=0.0, atol=1.0e-6)
                np.testing.assert_array_equal(cc.lambdas.numpy()[6:12], references)
                np.testing.assert_array_equal(contacts.rigid_contact_point0.numpy(), points0)
                np.testing.assert_array_equal(contacts.rigid_contact_point1.numpy(), points1)


if __name__ == "__main__":
    unittest.main()
