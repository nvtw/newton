# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contacts cannot move an anchored hinge along its rotation axis."""

import unittest

import numpy as np
import warp as wp

import newton


class TestDirectContactNullspace(unittest.TestCase):
    def test_kinematic_contact_endpoint_has_no_mobility(self):
        """Use only the dynamic sphere mass against a kinematic sphere."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        kinematic = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
        dynamic = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.019), wp.quat_identity()))
        for body in (kinematic, dynamic):
            builder.add_shape_sphere(body, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0))
        builder.add_joint_prismatic(
            -1, dynamic, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.019), wp.quat_identity())
        )
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            step_layout="single_world",
            sor_boost=1.0,
            substeps=1,
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200.0)
        self.assertIsNotNone(solver._direct_contact_response)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_allclose(
            solver._direct_contact_response.data.mobility.numpy()[0, 0], model.body_mass.numpy()[dynamic], rtol=1.0e-5
        )
        np.testing.assert_allclose(state.body_qd.numpy()[kinematic], np.zeros(6), atol=1.0e-7)

    def test_ordinary_contact_preserves_kinematic_endpoint(self):
        """Use only the dynamic sphere mass against a kinematic sphere."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        kinematic = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
        dynamic = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.019), wp.quat_identity()))
        for body in (kinematic, dynamic):
            builder.add_shape_sphere(body, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0))
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            step_layout="single_world",
            sor_boost=1.0,
            substeps=1,
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200.0)
        self.assertIsNone(solver._direct_contact_response)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_allclose(
            solver.world._contact_container.derived.numpy()[0, 0], model.body_mass.numpy()[dynamic], rtol=1.0e-5
        )
        np.testing.assert_allclose(state.body_qd.numpy()[kinematic], np.zeros(6), atol=1.0e-7)

    def test_blocked_contact_has_zero_mobility(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        b = newton.ModelBuilder()
        pose = wp.transform(wp.vec3(0.031, 0.017, 0.009), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.37))
        body = b.add_link(xform=pose)
        b.add_shape_box(body, hx=0.01, hy=0.02, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000, mu=0.0))
        b.add_joint_revolute(-1, body, axis=newton.Axis.Z, parent_xform=pose)
        b.add_ground_plane()
        model = b.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, step_layout="single_world")
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        response = solver._direct_contact_response.data
        active = response.contact_mechanism.numpy() >= 0
        self.assertGreater(np.count_nonzero(active), 0)
        mobility = response.mobility.numpy()[0, active]
        np.testing.assert_array_equal(
            mobility, np.zeros_like(mobility), err_msg="A Z hinge cannot separate a Z-normal plane contact"
        )
        # A row that becomes blocked must also discard its cached normal load.
        history = np.zeros_like(solver.world._contact_container.impulses.numpy())
        history[0, active] = 1.0
        solver.world._contact_container.impulses.assign(history)
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        active = response.contact_mechanism.numpy() >= 0
        np.testing.assert_array_equal(
            solver.world._contact_container.impulses.numpy()[0, active],
            np.zeros(np.count_nonzero(active)),
            err_msg="Blocked contact rows must release their cached normal impulse",
        )

    def test_single_world_skips_schur_owned_contact_sweeps(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder()
        pose = wp.transform(wp.vec3(0.031, 0.017, 0.009), wp.quat_identity())
        body = builder.add_link(xform=pose)
        builder.add_shape_box(
            body, hx=0.01, hy=0.02, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000, mu=0.0)
        )
        builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, parent_xform=pose)
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model, collision_pipeline=pipeline, step_layout="single_world", sor_boost=1.0
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        snapshots = []

        def observe_owned_contacts(**kwargs):
            if kwargs["use_bias"]:
                owner = solver.world._contact_cols.articulation_owner.numpy()
                self.assertTrue(np.any(owner >= 0))
                snapshots.append(solver.world._contact_container.impulses.numpy().copy())

        # Stop at the handoff: ordinary PGS must leave these rows untouched.
        solver.world._solve_direct_contacts = observe_owned_contacts
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        self.assertTrue(snapshots)
        for snapshot in snapshots:
            np.testing.assert_array_equal(snapshot, np.zeros_like(snapshot))

    def test_near_blocked_penetration_preserves_hinge_pose(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_link()
        builder.add_shape_sphere(
            body,
            xform=wp.transform(wp.vec3(0.02, 0, 0.00999), wp.quat_identity()),
            radius=0.01,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000, mu=0.0),
        )
        joint = builder.add_joint_revolute(-1, body, axis=wp.normalize(wp.vec3(0, 0.01, 1)))
        builder.add_articulation([joint])
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="maximal",
            step_layout="single_world",
            sor_boost=1.0,
            substeps=1,
            solver_iterations=8,
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        self.assertGreater(contacts.rigid_contact_count.numpy()[0], 0)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        # Ten micrometers of penetration must not rotate a nearly vertical hinge
        # by degrees in one substep. Relaxation can hide that jump in velocity.
        self.assertLess(abs(float(state.joint_q.numpy()[0])), 0.001)

    def test_floating_tree_internal_contact_has_zero_mobility(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.37)
        parent = builder.add_link(xform=wp.transform(wp.vec3(0.031, 0.017, 0.03), rotation))
        child = builder.add_link(xform=wp.transform(wp.vec3(0.031, 0.017, 0.049), rotation))
        for body in (parent, child):
            builder.add_shape_box(
                body, hx=0.01, hy=0.02, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000, mu=0.0)
            )
        root = builder.add_joint_free(parent)
        hinge = builder.add_joint_revolute(
            parent,
            child,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0, 0, 0.019), wp.quat_identity()),
            collision_filter_parent=False,
        )
        builder.add_articulation([root, hinge])
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model, collision_pipeline=pipeline, articulation_mode="maximal", step_layout="single_world", sor_boost=1.0
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        schedule = solver.world._maximal_contact_schedule
        self.assertIsNotNone(schedule)
        self.assertGreater(contacts.rigid_contact_count.numpy()[0], 0)
        count = int(contacts.rigid_contact_count.numpy()[0])
        np.testing.assert_array_equal(schedule.mobility.numpy()[0, :count], np.zeros(count))
        history = np.zeros_like(solver.world._contact_container.impulses.numpy())
        history[0, :count] = 1.0
        solver.world._contact_container.impulses.assign(history)
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        np.testing.assert_array_equal(solver.world._contact_container.impulses.numpy()[0, :count], np.zeros(count))


if __name__ == "__main__":
    unittest.main()
