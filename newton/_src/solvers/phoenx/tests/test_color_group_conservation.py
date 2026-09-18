# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Conserve momentum across real grouped contact solve and relaxation phases."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


def _physical_totals(world):
    """Sum physical body momentum and energy, independently of copy storage."""
    inverse_mass = world.bodies.inverse_mass.numpy().astype(np.float64)
    active = inverse_mass > 0.0
    mass = 1.0 / inverse_mass[active]
    position = world.bodies.position.numpy()[active].astype(np.float64)
    velocity = world.bodies.velocity.numpy()[active].astype(np.float64)
    spin = world.bodies.angular_velocity.numpy()[active].astype(np.float64)
    inertia = np.linalg.inv(
        inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()[active]).astype(np.float64)
    )
    linear = mass[:, None] * velocity
    angular = np.cross(position, linear) + np.einsum("bij,bj->bi", inertia, spin)
    energy = 0.5 * (np.sum(linear * velocity) + np.sum(spin * np.einsum("bij,bj->bi", inertia, spin)))
    return np.r_[linear.sum(axis=0), angular.sum(axis=0)], float(energy)


@unittest.skipUnless(_cuda_with_graph_capture(), "Color groups require CUDA")
class TestColorGroupConservation(unittest.TestCase):
    def test_unequal_copies_conserve_through_moving_com_relaxation(self):
        """Two frictional manifolds share a moving, rotating body across unequal copies."""
        for reverse in (False, True):
            for width in (1, 4):
                with self.subTest(reverse=reverse, width=width):
                    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                    positions = (-1.0000001, 0.0, 1.0000001)
                    if reverse:
                        positions = positions[::-1]
                    for x in positions:
                        mass = 2.0 + x * 0.5
                        body = builder.add_body(
                            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                            mass=mass,
                            inertia=wp.mat33(0.2 * mass, 0.0, 0.0, 0.0, 0.3 * mass, 0.0, 0.0, 0.0, 0.4 * mass),
                        )
                        builder.add_shape_box(
                            body,
                            hx=0.5,
                            hy=0.5,
                            hz=0.5,
                            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001),
                        )
                    model = builder.finalize(device="cuda:0")
                    state = model.state()
                    qd = state.body_qd.numpy()
                    for index, x in enumerate(positions):
                        qd[index, :3] = [-0.5 * x + 0.2, 0.2 * x + 0.3, 0.05]
                        qd[index, 3:] = [0.25, -0.3, 0.4]
                    state.body_qd.assign(qd)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
                    contacts = pipeline.contacts()
                    solver = newton.solvers.SolverPhoenX(
                        model,
                        collision_pipeline=pipeline,
                        articulation_mode="maximal",
                        joint_solver="block_pgs",
                        step_layout="single_world",
                        mass_splitting=True,
                        mass_splitting_color_group_size=width,
                        max_colored_partitions=1,
                        contact_chunk_size=1,
                        parallel_contact_prepare=True,
                        substeps=1,
                        solver_iterations=2,
                        velocity_iterations=1,
                        sor_boost=1.0,
                    )
                    world = solver.world
                    dispatcher_type = type(world._dispatcher)
                    original_solve, original_relax = dispatcher_type.solve, dispatcher_type.relax
                    records = []
                    poses = []
                    activity = []

                    def observed(original, phase, world=world, records=records, poses=poses, activity=activity):
                        def call(dispatcher, idt):
                            before, energy_before = _physical_totals(world)
                            velocity_before = world.bodies.velocity.numpy().copy()
                            original(dispatcher, idt)
                            after, energy_after = _physical_totals(world)
                            activity.append(
                                (phase, float(np.max(np.abs(world.bodies.velocity.numpy() - velocity_before))))
                            )
                            records.append((phase, before, after, energy_before, energy_after))
                            poses.append(
                                (world.bodies.position.numpy().copy(), world.bodies.orientation.numpy().copy())
                            )

                        return call

                    with (
                        patch.object(dispatcher_type, "solve", observed(original_solve, "biased")),
                        patch.object(dispatcher_type, "relax", observed(original_relax, "relax")),
                        patch.object(
                            type(world._partitioner),
                            "begin_sweep",
                            side_effect=AssertionError("Grouped sweeps must not launch ordinary cursor bookkeeping"),
                        ),
                    ):
                        for _ in range(4):
                            state.clear_forces()
                            pipeline.collide(state, contacts)
                            solver.step(state, state, model.control(), contacts, 0.001)
                    report = solver.step_report()
                    topology = world._color_group_data
                    color_count = int(topology["num_colors"].numpy()[0])
                    color_sizes = np.diff(topology["starts"].numpy()[: color_count + 1]).tolist()
                    self.assertEqual(report.num_colors, color_count)
                    self.assertEqual(report.color_sizes, color_sizes)
                    self.assertEqual(world.num_colors_used(), color_count)
                    self.assertEqual(report.overflow_size, 0)
                    self.assertEqual(
                        report.color_group_sizes,
                        [sum(color_sizes[i : i + width]) for i in range(0, color_count, width)],
                    )
                    self.assertEqual([r[0] for r in records], ["biased", "relax"] * 4)
                    counts = world._copy_state.count_per_node.numpy()
                    dynamic_counts = counts[world.bodies.inverse_mass.numpy() > 0.0]
                    self.assertGreater(int(dynamic_counts.max()), int(dynamic_counts.min()))
                    self.assertGreaterEqual(int(dynamic_counts.max()), 2)
                    self.assertGreater(float(np.max(np.abs(poses[1][0] - poses[0][0]))), 1e-5)
                    self.assertGreater(float(np.max(np.abs(poses[1][1] - poses[0][1]))), 1e-5)
                    for phase in ("biased", "relax"):
                        self.assertGreater(max(change for name, change in activity if name == phase), 1e-5)
                    for phase, before, after, _, _ in records:
                        np.testing.assert_allclose(after, before, atol=3e-6, rtol=0.0, err_msg=phase)
                    np.testing.assert_allclose(records[-1][2], records[0][1], atol=5e-6, rtol=0.0)
                    self.assertLessEqual(records[-1][4], records[0][3] + 5e-6)

    def test_ordinary_head_and_overflow_warm_contact_momentum(self):
        """Ordinary head and unequal overflow copies preserve paired internal and support impulses."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        links = []
        for x, mass in ((-1.0, 1.0), (0.0, 3.0), (1.0, 2.0)):
            body = builder.add_link(
                xform=wp.transform(wp.vec3(x, 0, 0.5), wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(mass * 0.2, 0, 0, 0, mass * 0.3, 0, 0, 0, mass * 0.4),
            )
            builder.add_shape_box(
                body, hx=0.5, hy=0.5, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, gap=0.001)
            )
            links.append(body)
        hinge = builder.add_joint_revolute(
            parent=links[0],
            child=links[1],
            axis=(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(0.5, 0.1, 0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5, 0.1, 0), wp.quat_identity()),
        )
        builder.add_articulation([hinge])
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            joint_solver="block_pgs",
            step_layout="single_world",
            mass_splitting=True,
            mass_splitting_color_group_size=0,
            max_colored_partitions=1,
            mass_splitting_batch_size=1,
            contact_chunk_size=1,
            parallel_contact_prepare=True,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
            sor_boost=1.0,
        )
        world = solver.world
        state = model.state()
        qd = state.body_qd.numpy()
        qd[:, :3] = [[0.2, 0.1, -0.1], [0.25, -0.1, -0.1], [-0.2, 0.05, -0.1]]
        qd[:, 3:] = [[0.05, 0.1, 0.2], [-0.05, 0.1, -0.1], [0.1, -0.1, 0.15]]
        state.body_qd.assign(qd)
        records = []
        warm_history = []
        unequal_counts = []
        count_history = []
        internal_activity = []
        joint_activity = []
        moment_arms = []
        head_activity = []
        overflow_activity = []
        dispatcher = type(world._dispatcher)
        solve, relax = dispatcher.solve, dispatcher.relax

        def observed(original, phase):
            def call(dispatcher, idt):
                before, energy_before = _physical_totals(world)
                old = world._contact_container.impulses.numpy().copy()
                joint_before = world.constraints.bilateral.accumulated.numpy().copy()
                if phase == "biased":
                    warm_history.append(bool(np.any(old != 0)))
                    old.fill(0.0)
                original(dispatcher, idt)
                after, energy_after = _physical_totals(world)
                new = world._contact_container.impulses.numpy()
                cc = world._contact_container.lambdas.numpy().astype(np.float64)
                derived = world._contact_container.derived.numpy().astype(np.float64)
                h = world._contact_cols.data.numpy().view(np.int32)
                positions = world.bodies.position.numpy().astype(np.float64)
                inverse_mass = world.bodies.inverse_mass.numpy()
                external = np.zeros(6)
                for column in range(int(world._ingest_scratch.num_contact_columns.numpy()[0])):
                    a, b = h[1:3, column]
                    if inverse_mass[a] > 0 and inverse_mass[b] > 0:
                        first, count = h[5:7, column]
                        internal_activity.append(
                            float(np.max(np.abs(new[:, first : first + count] - old[:, first : first + count])))
                        )
                        continue
                    first, count = h[5:7, column]
                    for k in range(first, first + count):
                        n, t = cc[:3, k], cc[3:6, k]
                        impulse = (
                            (new[0, k] - old[0, k]) * n
                            + (new[1, k] - old[1, k]) * t
                            + (new[2, k] - old[2, k]) * np.cross(n, t)
                        )
                        if inverse_mass[a] > 0:
                            impulse = -impulse
                            point = positions[a] + derived[9:12, k]
                        else:
                            point = positions[b] + derived[12:15, k]
                        moment_arms.append(float(np.linalg.norm(point - positions[a if inverse_mass[a] > 0 else b])))
                        external[:3] += impulse
                        external[3:] += np.cross(point, impulse)
                np.testing.assert_allclose(after - before, external, atol=2e-6, rtol=0)
                if phase == "relax":
                    self.assertLessEqual(energy_after, energy_before + 2e-6)
                counts = world._copy_state.count_per_node.numpy()
                unequal_counts.append(len(set(counts[1:4].tolist())) > 1 and max(counts[1:4]) > 1)
                count_history.append(tuple(counts[1:4].tolist()))
                starts = world._partitioner.color_starts.numpy()
                head_activity.append(int(starts[1] - starts[0]))
                overflow_activity.append(int(starts[2] - starts[1]))
                joint_activity.append(
                    float(np.max(np.abs(world.constraints.bilateral.accumulated.numpy() - joint_before)))
                )
                records.append(phase)

            return call

        with (
            patch.object(dispatcher, "solve", observed(solve, "biased")),
            patch.object(dispatcher, "relax", observed(relax, "relax")),
        ):
            for _ in range(4):
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, state, model.control(), contacts, 0.001)
        report = solver.step_report()
        self.assertEqual(report.overflow_size, overflow_activity[-1])
        self.assertIsNone(report.color_group_sizes)
        self.assertIn("biased", records)
        self.assertIn("relax", records)
        self.assertTrue(any(warm_history), "No persistent warm impulse exercised")
        self.assertTrue(any(unequal_counts), "Overflow fixture needs unequal endpoint copies")
        self.assertGreater(len(set(count_history)), 1, "Fixture must exercise changing copy counts")
        self.assertGreater(max(internal_activity), 1e-6, "No internal contact impulse exercised")
        self.assertGreater(max(joint_activity), 1e-6, "No joint impulse exercised")
        self.assertGreater(max(moment_arms), 0.1, "Support reactions must act off center")
        self.assertGreater(max(head_activity), 0, "Fixture never exercised ordinary rows")
        self.assertGreater(max(overflow_activity), 0, "Fixture never exercised overflow rows")


if __name__ == "__main__":
    unittest.main()
