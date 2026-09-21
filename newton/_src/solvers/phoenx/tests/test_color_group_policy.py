# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Public color-group scheduling keeps finite internal drives physical."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver
from newton._src.solvers.phoenx.tests.test_contact_chunks import _read_ranges, energy
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


@unittest.skipUnless(_cuda_with_graph_capture(), "Color groups require CUDA")
class TestColorGroupPolicy(unittest.TestCase):
    def test_captured_internal_drive_bound_and_momentum(self):
        """The public constructor dispatches captured group copies with bounded torques."""
        for width in (1, 2, 8):
            with self.subTest(width=width):
                model = make_model(100.0)
                model.joint_effort_limit.assign(np.asarray([0.2], dtype=np.float32))
                solver = make_solver(model, mass_splitting=True, mass_splitting_color_group_size=width)
                state = model.state()
                control = model.control()
                control.joint_target_q.assign(np.asarray([1.0], dtype=np.float32))
                initial_momentum = _total_momentum(model, state)
                with wp.ScopedCapture(device=model.device) as capture:
                    state.clear_forces()
                    solver.step(state, state, control, None, 0.01)
                previous = state.body_qd.numpy()
                for _ in range(8):
                    wp.capture_launch(capture.graph)
                    current = state.body_qd.numpy()
                    self.assertAlmostEqual(float(0.8 * (current[1, 5] - previous[1, 5])), 0.002, delta=2e-7)
                    np.testing.assert_allclose(_total_momentum(model, state), initial_momentum, atol=2e-6, rtol=0)
                    previous = current
                self.assertIsNotNone(solver.world._color_group_data)
                self.assertNotIn("_singleworld_head_plus_tail_sweep", solver.world.__dict__)
                self.assertNotIn("_rebuild_mass_splitting_graph", solver.world.__dict__)

    def test_grouped_face_contacts_preserve_momentum_and_energy(self):
        """Split every face point into a copy-safe column without losing rows."""
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001)
                positions = (1.0000001, 0.0) if reverse else (0.0, 1.0000001)
                for x in positions:
                    body = builder.add_body(
                        mass=1.0,
                        inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
                        xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                    )
                    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
                    # Force compound-body ingestion, whose warm-start gather
                    # rewrites the original pair-to-column map.
                    builder.add_shape_sphere(
                        body,
                        radius=0.01,
                        xform=wp.transform(wp.vec3(0.0, 10.0, 0.0), wp.quat_identity()),
                        cfg=cfg,
                    )
                model = builder.finalize(device="cuda:0")
                state = model.state()
                velocities = state.body_qd.numpy()
                for i, x in enumerate(positions):
                    velocities[i, :3] = [0.5 if x == 0.0 else -0.5, 0.15 if x == 0.0 else -0.15, 0.0]
                state.body_qd.assign(velocities)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
                contacts = pipeline.contacts()
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    step_layout="single_world",
                    mass_splitting=True,
                    max_colored_partitions=0,
                    mass_splitting_batch_size=1,
                    mass_splitting_color_group_size=2,
                    joint_mode="maximal_pgs",
                    parallel_contact_prepare=True,
                    substeps=1,
                    solver_iterations=1,
                    velocity_iterations=0,
                    sor_boost=1.0,
                    contact_chunk_size=1,
                )
                before = _total_momentum(model, state)
                before_energy = energy(model, state)
                pipeline.collide(state, contacts)
                point_count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreaterEqual(point_count, 4)
                solver.step(state, state, model.control(), contacts, 1.0e-4)
                world = solver.world
                count = int(world._ingest_scratch.num_contact_columns.numpy()[0])
                self.assertEqual(count, point_count)
                self.assertTrue(world._enable_body_pair_grouping)
                np.testing.assert_array_equal(
                    world._cid_of_contact_cur.numpy()[:point_count],
                    np.arange(point_count) + world._contact_offset,
                )
                ranges = wp.zeros((world.max_contact_columns, 2), dtype=wp.int32, device=model.device)
                wp.launch(
                    _read_ranges,
                    world.max_contact_columns,
                    [world._contact_cols, world._ingest_scratch.num_contact_columns, ranges],
                    device=model.device,
                )
                actual = ranges.numpy()[:count]
                np.testing.assert_array_equal(actual[:, 0], np.arange(point_count))
                np.testing.assert_array_equal(actual[:, 1], np.ones(point_count, dtype=np.int32))
                self.assertGreaterEqual(int(world._copy_state.count_per_node.numpy().max()), 2)
                np.testing.assert_allclose(_total_momentum(model, state), before, atol=2e-6, rtol=0)
                self.assertLessEqual(energy(model, state), before_energy + 1e-6)

    def test_grouped_direct_joints_use_ordinary_contacts_and_conserve_momentum(self):
        """Alternate exact joint projection with grouped contact PGS under capture."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        inertia = wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2)
        body0 = builder.add_body(mass=1.0, inertia=inertia)
        body1 = builder.add_body(
            mass=1.0,
            inertia=inertia,
            xform=wp.transform(wp.vec3(0.9, 0.0, 0.0), wp.quat_identity()),
        )
        follower = builder.add_body(
            mass=1.0,
            inertia=inertia,
            xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity()),
        )
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, gap=0.001)
        builder.add_shape_sphere(body0, radius=0.5, cfg=cfg)
        builder.add_shape_sphere(body1, radius=0.5, cfg=cfg)
        builder.add_joint_fixed(
            body0,
            follower,
            parent_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        )
        model = builder.finalize(device="cuda:0")
        state = model.state()
        velocity = state.body_qd.numpy()
        velocity[body0, :3] = (0.5, 0.1, 0.0)
        velocity[follower, :3] = velocity[body0, :3]
        velocity[body1, :3] = (-1.0, -0.2, 0.0)
        state.body_qd.assign(velocity)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16, contact_matching="sticky")
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="maximal_direct",
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=0,
            mass_splitting_batch_size=1,
            mass_splitting_color_group_size=2,
            parallel_contact_prepare=True,
            contact_chunk_size=1,
            substeps=2,
            solver_iterations=2,
            velocity_iterations=1,
            sor_boost=1.0,
        )
        self.assertIsNone(solver._direct_contact_response)
        self.assertIsNotNone(solver._direct_equality_system)
        before = _total_momentum(model, state)
        with wp.ScopedCapture(device=model.device) as capture:
            state.clear_forces()
            solver.step(state, state, model.control(), contacts, 1.0e-4)
        for _ in range(8):
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(_total_momentum(model, state), before, atol=3e-6, rtol=0)
        self.assertIsNotNone(solver.world._color_group_data)
        positions = state.body_q.numpy()[:, :3]
        self.assertAlmostEqual(float(np.linalg.norm(positions[body0] - positions[follower])), 1.0, delta=2e-5)

    def test_invalid_group_options_are_rejected(self):
        """Unsupported modes must fail explicitly instead of changing callbacks silently."""
        model = make_model(0.0)
        for options in (
            {"mass_splitting_color_group_size": -1, "mass_splitting": True},
            {"mass_splitting_color_group_size": 1.5, "mass_splitting": True},
            {"mass_splitting_color_group_size": True, "mass_splitting": True},
            {"mass_splitting_color_group_size": 8},
            {"mass_splitting_color_group_size": 8, "mass_splitting": True, "mass_splitting_unrolled": True},
            {"mass_splitting_color_group_size": 8, "mass_splitting": True, "sleeping_velocity_threshold": 0.1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_solver(model, **options)
        with self.assertRaisesRegex(ValueError, "sor_boost must be 1.0"):
            newton.solvers.SolverPhoenX(
                model,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                mass_splitting=True,
                mass_splitting_color_group_size=8,
                sor_boost=1.1,
            )
        self.assertIsNone(make_solver(model).world._color_group_data)


if __name__ == "__main__":
    unittest.main()
