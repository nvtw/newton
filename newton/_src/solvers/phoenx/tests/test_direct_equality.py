# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for mechanism-wide PhoenX equality solves."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.direct_equality import build_direct_equality_topology
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


def _build_badly_conditioned_chain(length: int = 12) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    parent = -1
    joints = []
    for index in range(length):
        mass = 1.0 if index % 2 == 0 else 1.0e-4
        child = _add_link(builder, wp.vec3(0.0, 0.0, -float(index) - 0.5), mass)
        if parent < 0:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
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
        self.assertTrue(direct.solver._parallel_factorization)

        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_driven_hinge_keeps_only_inequality_row_in_pgs(self):
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
        np.testing.assert_array_equal(
            solver.world._joint_pgs_enabled.numpy()[: solver.world.num_joints],
            [1],
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

    def test_ill_conditioned_chain_converges_better_than_pgs(self):
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        errors = {}
        for equality_solver in ("pgs", "direct"):
            model = _build_badly_conditioned_chain()
            solver = newton.solvers.SolverPhoenX(
                model,
                substeps=1,
                solver_iterations=1,
                velocity_iterations=0,
                articulation_mode="maximal",
                joint_equality_solver=equality_solver,
            )
            state_in = model.state()
            state_out = model.state()
            control = model.control()
            for _ in range(20):
                solver.step(state_in, state_out, control, None, 1.0 / 60.0)
                state_in, state_out = state_out, state_in
            errors[equality_solver] = _maximum_anchor_error(model, state_in)

        self.assertLess(errors["direct"], 0.1 * errors["pgs"])


if __name__ == "__main__":
    unittest.main()
