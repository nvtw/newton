# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for mechanism-wide PhoenX equality solves."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.direct_equality import build_direct_equality_topology
from newton._src.solvers.phoenx.examples.example_motorized_hinge_chain import _build_model
from newton._src.solvers.phoenx.tests.test_drive_stability import _pendulum


def _add_link(builder: newton.ModelBuilder, position: wp.vec3, mass: float) -> int:
    inertia = max(mass / 12.0, 1.0e-8)
    return builder.add_link(
        xform=wp.transform(position, wp.quat_identity()),
        mass=mass,
        inertia=((inertia, 0.0, 0.0), (0.0, inertia, 0.0), (0.0, 0.0, inertia)),
    )


def _build_two_mechanisms() -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for x, link_count in zip((-1.0, 1.0), (2, 3), strict=True):
        parent = -1
        joints = []
        for index in range(link_count):
            child = _add_link(builder, wp.vec3(x, 0.0, -float(index)), 1.0)
            parent_xform = wp.transform(
                wp.vec3(x, 0.0, 0.0) if parent < 0 else wp.vec3(0.0, 0.0, -1.0),
                wp.quat_identity(),
            )
            joints.append(
                builder.add_joint_fixed(
                    parent=parent,
                    child=child,
                    parent_xform=parent_xform,
                )
            )
            parent = child
        builder.add_articulation(joints)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_badly_conditioned_chain(length: int = 12, mass_ratio: float = 1.0e4, offset_x: float = 0.0) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    parent = -1
    joints = []
    for index in range(length):
        mass = 1.0 if index % 2 == 0 else 1.0 / mass_ratio
        child = _add_link(builder, wp.vec3(offset_x, 0.0, -float(index) - 0.5), mass)
        if parent < 0:
            parent_xform = wp.transform(wp.vec3(offset_x, 0.0, 0.0), wp.quat_identity())
        else:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity())
        joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=parent_xform,
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            limit_lower=-np.inf,
            limit_upper=np.inf,
        )
        joints.append(joint)
        parent = child
    builder.add_articulation(joints)
    model = builder.finalize()
    model.set_gravity((9.81, 0.0, 0.0))
    return model


def _build_fixed_pair_with_shapes() -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    bodies = []
    for z in (0.0, -0.5):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
        )
        builder.add_shape_box(
            body,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
        )
        bodies.append(body)
    root_joint = builder.add_joint_fixed(parent=-1, child=bodies[0])
    child_joint = builder.add_joint_fixed(
        parent=bodies[0],
        child=bodies[1],
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
    )
    builder.add_articulation([root_joint, child_joint])
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _maximum_anchor_error(model: newton.Model, state: newton.State) -> float:
    body_q = state.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()

    def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        xyz = transform[3:6]
        w = transform[6]
        rotated = point + 2.0 * np.cross(xyz, np.cross(xyz, point) + w * point)
        return transform[:3] + rotated

    maximum = 0.0
    for joint in range(int(model.joint_count)):
        parent = int(joint_parent[joint])
        child = int(joint_child[joint])
        point_parent = joint_x_p[joint, :3]
        if parent >= 0:
            point_parent = transform_point(body_q[parent], point_parent)
        point_child = transform_point(body_q[child], joint_x_c[joint, :3])
        maximum = max(maximum, float(np.linalg.norm(point_child - point_parent)))
    return maximum


class TestDirectEquality(unittest.TestCase):
    def test_world_anchor_does_not_merge_mechanisms(self):
        model = _build_two_mechanisms()
        topology = build_direct_equality_topology(model)
        self.assertEqual(topology.dimensions, (12, 18))

    def test_varying_mechanism_blocks_solve_together(self):
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_two_mechanisms()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
            articulation_mode="maximal",
        )
        self.assertEqual(solver._direct_equality_system.topology.dimensions, (12, 18))
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_declared_articulation_remains_reduced_with_mass_splitting(self):
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_badly_conditioned_chain(length=4)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
            mass_splitting=True,
            articulation_mode="auto",
            step_layout="single_world",
        )
        self.assertEqual(solver.articulation_mode, "reduced")
        self.assertIsNotNone(solver._reduced_articulation)
        self.assertFalse(solver._direct_equality_system.enabled)

        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_direct_only_mechanism_with_mass_splitting(self):
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_badly_conditioned_chain(length=4)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
            articulation_mode="maximal",
            mass_splitting=True,
            step_layout="single_world",
        )
        self.assertTrue(solver._direct_equality_system.enabled)
        self.assertTrue(solver.world._skip_all_joint_pgs())

        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_direct_joint_edges_survive_sleeping_partition_reuse(self):
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_fixed_pair_with_shapes()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            articulation_mode="maximal",
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=1,
        )
        self.assertTrue(solver.world._skip_all_joint_pgs())

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = model._collision_pipeline
        contacts = collision_pipeline.contacts()
        for reuse_partition in (False, True):
            collision_pipeline.collide(state_in, contacts)
            solver.reuse_partition = reuse_partition
            solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)
            state_in, state_out = state_out, state_in

        roots = solver.world.bodies.island_root.numpy()[1:3]
        self.assertGreaterEqual(int(roots[0]), 0)
        self.assertEqual(int(roots[0]), int(roots[1]))

    def test_large_mechanism_uses_panel_parallel_factorization(self):
        """Solve a large mechanism with compact panel-parallel tasks."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_badly_conditioned_chain(length=26)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            articulation_mode="maximal",
        )
        direct = solver._direct_equality_system
        self.assertEqual(direct.topology.dimensions, (130,))
        self.assertEqual(direct.solver.symbolic.tile_counts, (9,))
        self.assertGreater(direct.solver.symbolic.panel_count, 9)
        self.assertLess(direct.matrix.size, 130 * 130)
        schedule = direct.solver._persistent_schedule
        self.assertTrue(np.any(schedule.task_tile_i.numpy() != schedule.task_tile_k.numpy()))

        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_direct_driven_hinge_leaves_pgs_when_no_inequality_remains(self):
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _pendulum(target_pos=0.7, target_ke=100.0, target_kd=10.0)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=2,
            solver_iterations=4,
            velocity_iterations=1,
            articulation_mode="maximal",
        )
        self.assertTrue(solver._direct_equality_system.enabled)
        self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
        self.assertTrue(solver.world._skip_all_joint_pgs())
        np.testing.assert_array_equal(
            solver.world._joint_pgs_enabled.numpy()[: solver.world.num_joints],
            [0],
        )

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        for _ in range(30):
            solver.step(state_in, state_out, control, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        orientation = state_in.body_q.numpy()[0, 3:7]
        self.assertGreater(abs(float(orientation[1])), 0.05)
        self.assertTrue(np.isfinite(state_in.body_qd.numpy()).all())

    def test_direct_driven_hinge_keeps_joint_limit_in_pgs(self):
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _pendulum(
            target_pos=0.7,
            target_ke=100.0,
            target_kd=10.0,
            limit_lower=-0.2,
            limit_upper=0.2,
        )
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=2,
            solver_iterations=4,
            velocity_iterations=1,
            articulation_mode="maximal",
        )
        self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
        self.assertTrue(solver.world._joint_pgs_inequality_only)
        np.testing.assert_array_equal(
            solver.world._joint_pgs_enabled.numpy()[: solver.world.num_joints],
            [1],
        )

    def test_ill_conditioned_chain_remains_accurate_at_one_iteration(self):
        """Keep a badly conditioned chain accurate with one direct solve."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_badly_conditioned_chain()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=1,
            velocity_iterations=0,
            articulation_mode="maximal",
        )
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        for _ in range(20):
            solver.step(state_in, state_out, control, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in

        self.assertLess(_maximum_anchor_error(model, state_in), 1.0e-3)

    def test_equilibration_keeps_extreme_mass_ratio_finite(self):
        """Keep an extreme-mass-ratio chain finite with bounded anchor error."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_badly_conditioned_chain(mass_ratio=1.0e6)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=1,
            velocity_iterations=0,
            articulation_mode="maximal",
        )
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        for _ in range(20):
            solver.step(state_in, state_out, control, None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in

        self.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_in.body_qd.numpy()).all())
        self.assertLess(_maximum_anchor_error(model, state_in), 0.05)
        self.assertLess(
            float(np.min(solver._direct_equality_system.row_scale.numpy())),
            0.01,
        )

    def test_com_centered_rows_are_translation_invariant(self):
        """Preserve direct-chain motion when the mechanism is far from the origin."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        results = {}
        for offset_x in (0.0, 1000.0):
            model = _build_badly_conditioned_chain(length=8, offset_x=offset_x)
            solver = newton.solvers.SolverPhoenX(
                model,
                substeps=5,
                solver_iterations=1,
                velocity_iterations=0,
                articulation_mode="maximal",
            )
            state_in = model.state()
            state_out = model.state()
            control = model.control()
            for _ in range(20):
                solver.step(state_in, state_out, control, None, 1.0 / 60.0)
                state_in, state_out = state_out, state_in
            body_q = state_in.body_q.numpy()
            body_q[:, 0] -= offset_x
            results[offset_x] = (body_q, state_in.body_qd.numpy())

        self.assertTrue(np.isfinite(results[1000.0][0]).all())
        self.assertTrue(np.isfinite(results[1000.0][1]).all())
        np.testing.assert_allclose(results[1000.0][0], results[0.0][0], rtol=2.0e-3, atol=2.0e-3)
        np.testing.assert_allclose(results[1000.0][1], results[0.0][1], rtol=5.0e-3, atol=5.0e-3)

    def test_heterogeneous_mechanisms_use_compact_tasks(self):
        """Solve heterogeneous mechanisms using only valid matrix and panel tasks."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for offset_x, length in zip((0.0, 2.0, 4.0), (2, 8, 26), strict=True):
            parent = -1
            joints = []
            for index in range(length):
                child = _add_link(builder, wp.vec3(offset_x, 0.0, -float(index) - 0.5), 1.0)
                parent_xform = wp.transform(
                    wp.vec3(offset_x, 0.0, 0.0) if parent < 0 else wp.vec3(0.0, 0.0, -0.5),
                    wp.quat_identity(),
                )
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=wp.vec3(0.0, 1.0, 0.0),
                        parent_xform=parent_xform,
                        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                        limit_lower=-np.inf,
                        limit_upper=np.inf,
                    )
                )
                parent = child
            builder.add_articulation(joints)
        model = builder.finalize()
        model.set_gravity((9.81, 0.0, 0.0))
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=2,
            velocity_iterations=0,
            articulation_mode="maximal",
        )
        direct = solver._direct_equality_system
        self.assertEqual(direct.topology.dimensions, (10, 40, 130))
        symbolic = direct.solver.symbolic
        self.assertEqual(direct.matrix.size, symbolic.panel_count * direct.solver.block_size**2)
        self.assertLess(direct.matrix.size, 10 * 10 + 40 * 40 + 130 * 130)
        persistent = direct.solver._persistent_schedule
        self.assertEqual(persistent.task_mechanism.size, symbolic.panel_count)
        self.assertEqual(persistent.initial_task.size, 3)
        task_mechanism = persistent.task_mechanism.numpy()
        task_tile_i = persistent.task_tile_i.numpy()
        task_tile_k = persistent.task_tile_k.numpy()
        self.assertEqual(
            int(np.count_nonzero(task_tile_i != task_tile_k)), symbolic.panel_count - sum(symbolic.tile_counts)
        )
        np.testing.assert_array_equal(np.unique(task_mechanism), np.arange(3, dtype=np.int32))
        for mechanism, tile_count in enumerate(symbolic.tile_counts):
            self.assertEqual(
                int(np.count_nonzero((task_mechanism == mechanism) & (task_tile_i == task_tile_k))),
                tile_count,
            )

        state_in = model.state()
        state_out = model.state()
        for _ in range(20):
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            state_in, state_out = state_out, state_in
        self.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_in.body_qd.numpy()).all())

    def test_hundred_link_driven_chain_recovers_from_pick_impulse(self):
        """Recover a long driven chain from a strong pick-like impulse."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        num_links = 100
        model = _build_model(num_links)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            articulation_mode="maximal",
        )
        state = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_qd = state.body_qd.numpy()
        body_qd[num_links // 2, 0] = 10.0
        body_qd[num_links // 2, 3] = 10.0
        state.body_qd.assign(body_qd)

        with wp.ScopedCapture(model.device) as capture:
            solver.step(state, state, control, None, 1.0 / 60.0)
        peak_velocity = 0.0
        peak_anchor_error = 0.0
        checkpoints = {0, 9, 29, 59, 119}
        for frame in range(120):
            wp.capture_launch(capture.graph)
            if frame in checkpoints:
                body_qd = state.body_qd.numpy()
                peak_velocity = max(peak_velocity, float(np.max(np.abs(body_qd))))
                peak_anchor_error = max(peak_anchor_error, _maximum_anchor_error(model, state))

        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state.body_qd.numpy()).all())
        self.assertLess(peak_velocity, 2.0)
        self.assertLess(peak_anchor_error, 1.0e-4)
        self.assertLess(_maximum_anchor_error(model, state), 5.0e-6)

    def test_multi_world_hybrid_reduced_and_direct_ownership(self):
        """Assign disjoint reduced and direct mechanisms in every world."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        template = newton.ModelBuilder(up_axis=newton.Axis.Z)
        parent = -1
        reduced_joints = []
        for index in range(2):
            child = _add_link(template, wp.vec3(0.0, 0.0, -float(index) - 0.5), 1.0)
            parent_xform = wp.transform(
                wp.vec3(0.0, 0.0, 0.0) if parent < 0 else wp.vec3(0.0, 0.0, -0.5),
                wp.quat_identity(),
            )
            reduced_joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=child,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=parent_xform,
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                    limit_lower=-np.inf,
                    limit_upper=np.inf,
                )
            )
            parent = child
        template.add_articulation(reduced_joints)

        direct_body = _add_link(template, wp.vec3(2.0, 0.0, -0.5), 1.0)
        template.add_joint_revolute(
            parent=-1,
            child=direct_body,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            limit_lower=-np.inf,
            limit_upper=np.inf,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, 3)
        model = builder.finalize()
        model.set_gravity((9.81, 0.0, 0.0))
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            articulation_mode="reduced",
            step_layout="multi_world",
        )

        direct = solver._direct_equality_system
        self.assertEqual(model.world_count, 3)
        self.assertEqual(direct.topology.dimensions, (5, 5, 5))
        self.assertEqual(int(np.count_nonzero(solver._reduced_articulation.owned_joint_mask_np)), 6)
        self.assertEqual(int(np.count_nonzero(direct.joint_mask)), 3)
        np.testing.assert_array_equal(direct.solver.permutation.numpy(), direct.topology.permutation)


if __name__ == "__main__":
    unittest.main()
