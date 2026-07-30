# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph-capture tests for the simple scalar-row PhoenX flavor."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.simple.contacts import CONTACT_ROW_STRIDE
from newton._src.solvers.phoenx.simple.rows import (
    apply_body_velocity_deltas_kernel,
    clear_body_split_state_kernel,
    count_body_split_incidence_kernel,
    scalar_row_container_zeros,
    snapshot_body_velocities_kernel,
    snapshot_row_multipliers_kernel,
    solve_scalar_rows_jacobi_kernel,
)
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene
from newton._src.solvers.phoenx.world_builder import JointMode, WorldBuilder


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "simple PhoenX tests require CUDA graph capture")
class TestSimplePhoenX(unittest.TestCase):
    def test_jacobi_color_estimate_scales_substeps_in_capture(self) -> None:
        """Scale contact-only Jacobi substeps inside a captured graph."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=wp.get_preferred_device())
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=1,
            velocity_iterations=0,
            solver_flavor="simple",
            jacobi_max_colors=3,
        )
        self.assertEqual(solver.world.base_substeps, 5)
        self.assertEqual(solver.world.substeps, 15)

        default_solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=1,
            velocity_iterations=0,
            solver_flavor="simple",
        )
        self.assertEqual(default_solver.world.jacobi_max_colors, 10)
        self.assertEqual(default_solver.world.substeps, 50)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        solver.step(state_in, state_out, control, None, 1.0 / 60.0)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_in, state_out, control, None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)

    def test_rejects_jointed_models_in_the_contact_only_flavor(self) -> None:
        """Reject joints instead of silently assembling iterative rows."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])
        model = builder.finalize(device=wp.get_preferred_device())
        with self.assertRaisesRegex(NotImplementedError, "contact-only"):
            newton.solvers.SolverPhoenX(
                model,
                substeps=5,
                solver_iterations=1,
                velocity_iterations=0,
                solver_flavor="simple",
            )

        raw_builder = WorldBuilder()
        raw_body = raw_builder.add_dynamic_body(
            inverse_mass=1.0,
            inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        raw_builder.add_joint(
            body1=raw_builder.world_body,
            body2=raw_body,
            anchor1=(0.0, 0.0, 0.0),
            mode=JointMode.FIXED,
            anchor2=(1.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(NotImplementedError, "contact-only"):
            raw_builder.finalize(solver_flavor="simple", device=wp.get_preferred_device())

    def test_contact_rows_settle_in_captured_pipeline(self) -> None:
        """Settle contact rows in both captured scheduling layouts."""
        for step_layout in ("single_world", "multi_world"):
            with self.subTest(step_layout=step_layout):
                scene = _PhoenXScene(
                    fps=60,
                    substeps=16,
                    solver_iterations=2,
                    velocity_iterations=0,
                    step_layout=step_layout,
                    solver_flavor="simple",
                    jacobi_max_colors=1,
                )
                scene.add_ground_plane()
                body = scene.add_box((0.0, 0.0, 0.6), (0.5, 0.5, 0.5))
                scene.finalize()
                for _ in range(90):
                    scene.step()
                self.assertAlmostEqual(float(scene.body_position(body)[2]), 0.5, delta=0.08)
                self.assertLess(abs(float(scene.body_velocity(body)[2])), 0.15)

    def test_contact_cache_seeds_and_scatters_rows_in_capture(self) -> None:
        """Seed active contact rows and clear stale inactive rows."""
        scene = _PhoenXScene(
            substeps=5,
            solver_iterations=1,
            velocity_iterations=0,
            solver_flavor="simple",
            jacobi_max_colors=1,
        )
        scene.add_ground_plane()
        body = scene.add_box((0.0, 0.0, 0.45), (0.5, 0.5, 0.5))
        scene.finalize()
        scene.step()

        world = scene.world
        dispatcher = world._dispatcher
        contact_count = int(scene.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0)
        impulses = np.zeros(world._contact_container.impulses.shape, dtype=np.float32)
        cached_lambdas = np.asarray((0.25, 0.1, -0.05), dtype=np.float32)
        impulses[:, :contact_count] = cached_lambdas[:, None]
        world._contact_container.impulses.assign(impulses)
        dispatcher.rows.multiplier.zero_()
        world.bodies.velocity.zero_()
        world.bodies.angular_velocity.zero_()
        world.solver_iterations = 0
        world._current_substep_index = 0
        idt = wp.float32(1.0 / world.substep_dt)

        with wp.ScopedCapture(device=world.device) as capture:
            dispatcher.solve(idt)
        wp.capture_launch(capture.graph)

        contact_rows = dispatcher._contact_row_offset + np.arange(contact_count)[:, None] * CONTACT_ROW_STRIDE
        contact_rows = contact_rows + np.arange(CONTACT_ROW_STRIDE)[None, :]
        row_lambdas = dispatcher.rows.multiplier.numpy()[contact_rows]
        expected_lambdas = np.broadcast_to(cached_lambdas, (contact_count, CONTACT_ROW_STRIDE))
        np.testing.assert_allclose(row_lambdas, expected_lambdas, atol=1.0e-7)
        np.testing.assert_allclose(
            world._contact_container.impulses.numpy()[:, :contact_count], expected_lambdas.T, atol=1.0e-7
        )
        self.assertGreater(float(np.linalg.norm(scene.body_velocity(body))), 1.0e-4)

        positions = world.bodies.position.numpy()
        positions[body + 1, 2] += 1.0
        world.bodies.position.assign(positions)
        world.bodies.velocity.zero_()
        dispatcher.rows.multiplier.zero_()
        world._contact_container.impulses.assign(impulses)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(dispatcher.rows.multiplier.numpy()[contact_rows], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(scene.body_velocity(body), np.zeros(3), atol=1.0e-7)

        world._contact_views = None
        dispatcher.rows.active.fill_(1)
        dispatcher.rows.multiplier.fill_(1.0)
        with wp.ScopedCapture(device=world.device) as empty_capture:
            dispatcher.solve(idt)
        wp.capture_launch(empty_capture.graph)

        contact_slice = slice(dispatcher._contact_row_offset, dispatcher._row_count)
        np.testing.assert_array_equal(dispatcher.rows.active.numpy()[contact_slice], 0)
        np.testing.assert_allclose(scene.body_velocity(body), np.zeros(3), atol=1.0e-7)

    def test_duplicate_rows_have_contact_count_independent_response(self) -> None:
        """Keep the Jacobi response independent of duplicate contact count."""
        device = wp.get_preferred_device()
        for row_count in (1, 8, 128):
            with self.subTest(row_count=row_count):
                bodies = body_container_zeros(2, device=device)
                bodies.inverse_mass.assign(np.ones(2, dtype=np.float32))
                rows = scalar_row_container_zeros(row_count, device=device)
                rows.active.assign(np.ones(row_count, dtype=np.int32))
                rows.split_anchor.assign(np.ones(row_count, dtype=np.int32))
                rows.body_a.assign(np.zeros(row_count, dtype=np.int32))
                rows.body_b.assign(np.ones(row_count, dtype=np.int32))
                rows.jacobian_linear_a.assign(np.tile(np.asarray((-1.0, 0.0, 0.0), dtype=np.float32), (row_count, 1)))
                rows.jacobian_linear_b.assign(np.tile(np.asarray((1.0, 0.0, 0.0), dtype=np.float32), (row_count, 1)))
                rows.bound_row.assign(np.arange(row_count, dtype=np.int32))

                velocity_snapshot = wp.zeros(2, dtype=wp.vec3f, device=device)
                angular_velocity_snapshot = wp.zeros(2, dtype=wp.vec3f, device=device)
                delta_velocity = wp.zeros(2, dtype=wp.vec3f, device=device)
                delta_angular_velocity = wp.zeros(2, dtype=wp.vec3f, device=device)
                multiplier_snapshot = wp.zeros(row_count, dtype=wp.float32, device=device)
                body_split_count = wp.zeros(2, dtype=wp.int32, device=device)

                sweep_state = (
                    row_count,
                    bodies,
                    rows,
                    velocity_snapshot,
                    angular_velocity_snapshot,
                    delta_velocity,
                    delta_angular_velocity,
                    multiplier_snapshot,
                    body_split_count,
                )

                def sweep(sweep_state=sweep_state):
                    (
                        row_count,
                        bodies,
                        rows,
                        velocity_snapshot,
                        angular_velocity_snapshot,
                        delta_velocity,
                        delta_angular_velocity,
                        multiplier_snapshot,
                        body_split_count,
                    ) = sweep_state
                    wp.launch(
                        clear_body_split_state_kernel,
                        dim=2,
                        outputs=[body_split_count, delta_velocity, delta_angular_velocity],
                        device=device,
                    )
                    wp.launch(
                        count_body_split_incidence_kernel,
                        dim=row_count,
                        inputs=[rows],
                        outputs=[body_split_count],
                        device=device,
                    )
                    wp.launch(
                        snapshot_body_velocities_kernel,
                        dim=2,
                        inputs=[bodies],
                        outputs=[
                            velocity_snapshot,
                            angular_velocity_snapshot,
                            delta_velocity,
                            delta_angular_velocity,
                        ],
                        device=device,
                    )
                    wp.launch(
                        snapshot_row_multipliers_kernel,
                        dim=row_count,
                        inputs=[rows],
                        outputs=[multiplier_snapshot],
                        device=device,
                    )
                    wp.launch(
                        solve_scalar_rows_jacobi_kernel,
                        dim=row_count,
                        inputs=[
                            rows,
                            bodies,
                            velocity_snapshot,
                            angular_velocity_snapshot,
                            multiplier_snapshot,
                            body_split_count,
                            wp.float32(1.0),
                            wp.float32(0.0),
                        ],
                        outputs=[delta_velocity, delta_angular_velocity],
                        device=device,
                    )
                    wp.launch(
                        apply_body_velocity_deltas_kernel,
                        dim=2,
                        inputs=[bodies, body_split_count, delta_velocity, delta_angular_velocity],
                        device=device,
                    )

                bodies.velocity.assign(np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=np.float32))
                sweep()
                bodies.velocity.assign(np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=np.float32))
                rows.multiplier.zero_()
                with wp.ScopedCapture(device=device) as capture:
                    sweep()
                wp.capture_launch(capture.graph)

                np.testing.assert_array_equal(body_split_count.numpy(), row_count)
                velocities = bodies.velocity.numpy()
                np.testing.assert_allclose(velocities.sum(axis=0), np.zeros(3), atol=2.0e-6)
                np.testing.assert_allclose(velocities, np.zeros((2, 3)), atol=2.0e-6)

    def test_contact_warmstart_conserves_pair_momentum(self) -> None:
        """Conserve pair momentum when contact warm-starting is enabled."""
        scene = _PhoenXScene(
            fps=240,
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            friction=0.0,
            solver_flavor="simple",
            jacobi_max_colors=2,
        )
        body_a = scene.add_box((-0.48, 0.0, 1.0), (0.5, 0.5, 0.5), mass=1.0)
        body_b = scene.add_box((0.48, 0.0, 1.0), (0.5, 0.5, 0.5), mass=1.0)
        scene.finalize()
        scene.set_body_velocity(body_a, (1.0, 0.0, 0.0))
        scene.set_body_velocity(body_b, (-1.0, 0.0, 0.0))

        for _ in range(20):
            scene.step()

        velocity_a = scene.body_velocity(body_a)
        velocity_b = scene.body_velocity(body_b)
        self.assertTrue(np.isfinite(velocity_a).all())
        self.assertTrue(np.isfinite(velocity_b).all())
        self.assertAlmostEqual(float(velocity_a[0] + velocity_b[0]), 0.0, delta=2.0e-5)


if __name__ == "__main__":
    unittest.main()
