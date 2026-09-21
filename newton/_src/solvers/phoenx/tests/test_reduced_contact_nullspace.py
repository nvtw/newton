# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise contact stabilization near an articulation nullspace."""

import unittest

import numpy as np
import warp as wp

import newton


class TestReducedContactNullspace(unittest.TestCase):
    def test_nearly_blocked_penetration_does_not_rotate_hinge(self):
        """Bound correction when a contact is almost parallel to a hinge axis."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_link()
        builder.add_shape_sphere(
            body,
            xform=wp.transform(wp.vec3(0.02, 0.0, 0.00999), wp.quat_identity()),
            radius=0.01,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0),
        )
        joint = builder.add_joint_revolute(-1, body, axis=wp.normalize(wp.vec3(0.0, 0.01, 1.0)))
        builder.add_articulation([joint])
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="reduced",
            step_layout="single_world",
            sor_boost=1.0,
            substeps=1,
            solver_iterations=8,
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200.0)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertLess(abs(float(state.joint_q.numpy()[0])), 0.001)
        self.assertGreater(float(solver.world._contact_container.derived.numpy()[0, 0]), 0.0)

    def test_blocked_internal_contact_has_zero_mobility(self):
        """Reject roundoff mobility for contact along an internal hinge axis."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, 0.7, 0.5)), 0.73)
        pose = wp.transform(wp.vec3(0.031, 0.017, 0.009), rotation)
        offset = wp.transform(wp.vec3(0.0, 0.0, 0.019), wp.quat_identity())
        parent = builder.add_link(xform=pose)
        child = builder.add_link(xform=pose * offset)
        for body in (parent, child):
            builder.add_shape_box(
                body, hx=0.01, hy=0.02, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0)
            )
        root_joint = builder.add_joint_free(parent)
        hinge = builder.add_joint_revolute(
            parent, child, axis=newton.Axis.Z, parent_xform=offset, collision_filter_parent=False
        )
        builder.add_articulation([root_joint, hinge])
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="reduced",
            step_layout="single_world",
            sor_boost=1.0,
            substeps=1,
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200.0)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        np.testing.assert_array_equal(solver.world._contact_container.derived.numpy()[0, :count], np.zeros(count))
        history = solver.world._contact_container.impulses.numpy()
        history[0, :count] = 1.0
        solver.world._contact_container.impulses.assign(history)
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200.0)
        count = int(contacts.rigid_contact_count.numpy()[0])
        np.testing.assert_array_equal(solver.world._contact_container.impulses.numpy()[0, :count], np.zeros(count))

    def test_kinematic_contact_endpoint_has_no_mobility(self):
        """Use only the dynamic sphere mass against a kinematic sphere."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        kinematic = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
        dynamic = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.019), wp.quat_identity()))
        for body in (kinematic, dynamic):
            builder.add_shape_sphere(body, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0))
            builder.add_articulation([builder.add_joint_free(body)])
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="reduced",
            step_layout="single_world",
            sor_boost=1.0,
            substeps=1,
        )
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200.0)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_allclose(
            solver.world._contact_container.derived.numpy()[0, 0], model.body_mass.numpy()[dynamic], rtol=1.0e-5
        )
        np.testing.assert_allclose(state.body_qd.numpy()[kinematic], np.zeros(6), atol=1.0e-7)

    def test_kinematic_free_joint_is_not_integrated_by_reduced_dynamics(self):
        """Preserve a kinematic body while a neighboring dynamic tree falls."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        for dynamic_neighbor in (False, True):
            with self.subTest(dynamic_neighbor=dynamic_neighbor):
                builder = newton.ModelBuilder()
                body = builder.add_link(
                    xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                    is_kinematic=True,
                )
                builder.add_shape_sphere(body, radius=0.01)
                builder.add_articulation([builder.add_joint_free(body)])
                if dynamic_neighbor:
                    dynamic = builder.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 1.0), wp.quat_identity()))
                    builder.add_shape_sphere(dynamic, radius=0.01)
                    builder.add_articulation([builder.add_joint_free(dynamic)])
                model = builder.finalize(device="cuda:0")
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    joint_mode="reduced",
                    step_layout="single_world",
                    sor_boost=1.0,
                    substeps=1,
                )
                state = model.state()
                contacts = pipeline.contacts()
                for _ in range(2):
                    pipeline.collide(state, contacts)
                    solver.step(state, state, model.control(), contacts, 1.0 / 60.0)
                np.testing.assert_allclose(state.body_q.numpy()[body, :3], [0.0, 0.0, 1.0], atol=1.0e-6)
                np.testing.assert_array_equal(state.body_qd.numpy()[body], np.zeros(6))
                self.assertEqual(model.articulation_count, 2 if dynamic_neighbor else 1)
                if dynamic_neighbor:
                    self.assertLess(float(state.body_q.numpy()[dynamic, 2]), 1.0)
                    # Export reduced coordinates without overwriting a prescribed tree.
                    prescribed_q = state.joint_q.numpy()
                    prescribed_q[0] = 0.25
                    state.joint_q.assign(prescribed_q)
                    prescribed_qd = state.joint_qd.numpy()
                    prescribed_qd[0] = 0.5
                    state.joint_qd.assign(prescribed_qd)
                    solver._reduced_articulation.export_step(state)
                    self.assertEqual(float(state.joint_q.numpy()[0]), 0.25)
                    self.assertEqual(float(state.joint_qd.numpy()[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
