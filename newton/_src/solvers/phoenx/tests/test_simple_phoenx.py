# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph-capture tests for the simple scalar-row PhoenX flavor."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.simple.contacts import CONTACT_ROW_STRIDE
from newton._src.solvers.phoenx.simple.joints import JOINT_ROW_STRIDE
from newton._src.solvers.phoenx.simple.rows import (
    apply_body_velocity_deltas_kernel,
    clear_body_split_counts_kernel,
    count_body_split_incidence_kernel,
    scalar_row_container_zeros,
    snapshot_body_velocities_kernel,
    snapshot_row_multipliers_kernel,
    solve_scalar_rows_jacobi_kernel,
)
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene
from newton._src.solvers.phoenx.world_builder import DriveMode, JointMode, WorldBuilder


def _make_welded_world(step_layout: str, solver_flavor: str):
    builder = WorldBuilder()
    body = builder.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        affected_by_gravity=True,
    )
    builder.add_joint(
        body1=builder.world_body,
        body2=body,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        mode=JointMode.FIXED,
    )
    return builder.finalize(
        substeps=8,
        solver_iterations=2,
        velocity_iterations=0,
        gravity=(0.0, 0.0, -9.81),
        step_layout=step_layout,
        solver_flavor=solver_flavor,
        jacobi_max_colors=1,
        device=wp.get_preferred_device(),
    )


def _make_dynamic_chain(step_layout: str):
    builder = WorldBuilder()
    positions = (-1.0, 0.0, 1.0)
    velocities = ((0.3, 1.0, 0.2), (-0.5, 0.2, -0.1), (0.2, -0.8, 0.4))
    angular_velocities = ((0.1, -0.2, 0.3), (-0.3, 0.4, 0.2), (0.2, -0.1, -0.4))
    bodies = []
    for x, velocity, angular_velocity in zip(positions, velocities, angular_velocities, strict=True):
        bodies.append(
            builder.add_dynamic_body(
                position=(x, 0.0, 0.0),
                velocity=velocity,
                angular_velocity=angular_velocity,
                inverse_mass=1.0,
                inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                affected_by_gravity=False,
            )
        )
    for body_a, body_b, anchor in ((bodies[0], bodies[1], -0.5), (bodies[1], bodies[2], 0.5)):
        builder.add_joint(
            body1=body_a,
            body2=body_b,
            anchor1=(anchor, 0.0, 0.0),
            mode=JointMode.BALL_SOCKET,
        )
    return builder.finalize(
        substeps=8,
        solver_iterations=2,
        velocity_iterations=0,
        gravity=(0.0, 0.0, 0.0),
        step_layout=step_layout,
        solver_flavor="simple",
        jacobi_max_colors=1,
        device=wp.get_preferred_device(),
    )


def _momentum(world) -> tuple[np.ndarray, np.ndarray]:
    positions = world.bodies.position.numpy()[1:].astype(np.float64)
    velocities = world.bodies.velocity.numpy()[1:].astype(np.float64)
    angular_velocities = world.bodies.angular_velocity.numpy()[1:].astype(np.float64)
    linear = velocities.sum(axis=0)
    angular = angular_velocities.sum(axis=0) + np.cross(positions, velocities).sum(axis=0)
    return linear, angular


def _make_driven_world():
    builder = WorldBuilder()
    body = builder.add_dynamic_body(
        position=(0.0, 1.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        affected_by_gravity=False,
    )
    builder.add_joint(
        body1=builder.world_body,
        body2=body,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        mode=JointMode.REVOLUTE,
        drive_mode=DriveMode.POSITION,
        target=0.5,
        max_force_drive=1000.0,
        stiffness_drive=800.0,
        damping_drive=40.0,
    )
    return builder.finalize(
        substeps=16,
        solver_iterations=2,
        velocity_iterations=0,
        gravity=(0.0, 0.0, 0.0),
        solver_flavor="simple",
        jacobi_max_colors=1,
        device=wp.get_preferred_device(),
    )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "simple PhoenX tests require CUDA graph capture")
class TestSimplePhoenX(unittest.TestCase):
    def test_jacobi_color_estimate_scales_substeps_in_capture(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.gravity = 0.0
        model = builder.finalize(device=wp.get_preferred_device())
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=2,
            solver_iterations=1,
            velocity_iterations=0,
            solver_flavor="simple",
            jacobi_max_colors=3,
        )
        self.assertEqual(solver.world.base_substeps, 2)
        self.assertEqual(solver.world.substeps, 6)

        default_solver = newton.solvers.SolverPhoenX(
            model,
            substeps=2,
            solver_iterations=1,
            velocity_iterations=0,
            solver_flavor="simple",
        )
        self.assertEqual(default_solver.world.jacobi_max_colors, 10)
        self.assertEqual(default_solver.world.substeps, 20)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        solver.step(state_in, state_out, control, None, 1.0 / 60.0)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_in, state_out, control, None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)

    def test_full_step_capture_replay_for_both_flavors_and_layouts(self):
        for solver_flavor in ("standard", "simple"):
            for step_layout in ("single_world", "multi_world"):
                with self.subTest(solver_flavor=solver_flavor, step_layout=step_layout):
                    world = _make_welded_world(step_layout, solver_flavor)
                    if solver_flavor == "simple":
                        rows = world._dispatcher.rows
                        self.assertEqual(rows.active.shape[0], JOINT_ROW_STRIDE)
                        self.assertIsNone(world._partitioner)
                        self.assertFalse(hasattr(rows, "color"))
                        self.assertFalse(hasattr(rows, "block"))
                    world.step(1.0 / 60.0)  # compile before capture
                    with wp.ScopedCapture(device=world.device) as capture:
                        world.step(1.0 / 60.0)
                    for _ in range(30):
                        wp.capture_launch(capture.graph)

                    position = world.bodies.position.numpy()[1]
                    velocity = world.bodies.velocity.numpy()[1]
                    np.testing.assert_allclose(position, np.zeros(3), atol=2.0e-2)
                    np.testing.assert_allclose(velocity, np.zeros(3), atol=2.0e-2)

    def test_contact_rows_settle_in_captured_pipeline(self):
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

    def test_joint_and_contact_rows_share_one_fixed_launch_domain(self):
        scene = _PhoenXScene(
            substeps=2,
            solver_iterations=1,
            velocity_iterations=0,
            solver_flavor="simple",
            jacobi_max_colors=1,
        )
        scene.add_ground_plane()
        body = scene.mb.add_link(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.18), q=wp.quat_identity()))
        scene.mb.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
        joint = scene.mb.add_joint_revolute(
            parent=-1,
            child=body,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.18), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
        )
        scene.mb.add_articulation([joint])
        scene.finalize()
        scene.step()

        dispatcher = scene.world._dispatcher
        rows = dispatcher.rows
        contact_count = int(scene.contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0)
        self.assertEqual(dispatcher.block_dim, 256)
        self.assertEqual(dispatcher._contact_row_offset, scene.world.num_joints * JOINT_ROW_STRIDE)
        self.assertEqual(
            dispatcher._row_count,
            scene.world.num_joints * JOINT_ROW_STRIDE + scene.world.rigid_contact_max * CONTACT_ROW_STRIDE,
        )

        active = rows.active.numpy()
        self.assertTrue(np.any(active[: dispatcher._contact_row_offset]))
        contact_row = dispatcher._contact_row_offset
        np.testing.assert_array_equal(active[contact_row : contact_row + CONTACT_ROW_STRIDE], np.ones(3))
        split_anchors = rows.split_anchor.numpy()
        np.testing.assert_array_equal(
            split_anchors[contact_row : contact_row + CONTACT_ROW_STRIDE], np.asarray((1, 0, 0))
        )
        bound_rows = rows.bound_row.numpy()
        self.assertEqual(int(bound_rows[contact_row]), contact_row)
        self.assertEqual(int(bound_rows[contact_row + 1]), contact_row)
        self.assertEqual(int(bound_rows[contact_row + 2]), contact_row)

    def test_duplicate_rows_have_contact_count_independent_response(self):
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
                    wp.launch(clear_body_split_counts_kernel, dim=2, inputs=[body_split_count], device=device)
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

    def test_internal_atomic_fanin_conserves_momentum(self):
        for step_layout in ("single_world", "multi_world"):
            with self.subTest(step_layout=step_layout):
                world = _make_dynamic_chain(step_layout)
                linear_before, angular_before = _momentum(world)
                world.step(1.0 / 240.0)
                with wp.ScopedCapture(device=world.device) as capture:
                    world.step(1.0 / 240.0)
                for _ in range(120):
                    wp.capture_launch(capture.graph)
                row_counts = world._dispatcher._body_split_count.numpy()[1:]
                self.assertGreater(int(row_counts[1]), int(row_counts[0]))
                linear_after, angular_after = _momentum(world)
                np.testing.assert_allclose(linear_after, linear_before, rtol=2.0e-5, atol=2.0e-5)
                np.testing.assert_allclose(angular_after, angular_before, rtol=2.0e-4, atol=2.0e-4)

    def test_axial_drive_is_an_independent_captured_row(self):
        world = _make_driven_world()
        world.step(1.0 / 120.0)
        with wp.ScopedCapture(device=world.device) as capture:
            world.step(1.0 / 120.0)
        for _ in range(120):
            wp.capture_launch(capture.graph)
        rows = world._dispatcher.rows
        self.assertEqual(int(rows.active.numpy()[6]), 1)
        self.assertGreater(abs(float(rows.multiplier.numpy()[6])), 1.0e-6)
        self.assertGreater(abs(float(world.bodies.orientation.numpy()[1, 0])), 0.05)


if __name__ == "__main__":
    unittest.main()
