# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal contacts must cancel common floating-root motion accurately."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.maximal_contact_gs import (
    _contact_row_velocity,
    _write_exact_contact_mobility,
)
from newton._src.solvers.phoenx.articulations.maximal_contact_response import (
    MaximalContactResponseData,
    maximal_contact_pair_inverse_mass,
)
from newton._src.solvers.phoenx.articulations.maximal_projector import MaximalTreeProjectorData
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_normal,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1,
)


@wp.kernel
def evaluate(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    point: wp.vec3,
    direction: wp.vec3,
    result: wp.array[float],
):
    result[0] = maximal_contact_pair_inverse_mass(tree, response, bodies, 0, point, -direction, 1, point, direction)


@wp.kernel
def evaluate_velocity(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    result: wp.array[float],
):
    result[0] = _contact_row_velocity(
        tree, response, bodies, 0, 1, wp.vec3(0.02, 0, 0.0199), wp.vec3(0.02, 0, 0), wp.vec3(0, 0, 1)
    )


@wp.kernel
def evaluate_offset_mobility(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    contacts: ContactContainer,
    result: wp.array2d[float],
):
    cc_set_normal(contacts, 0, wp.vec3(0.001, 0, 1))
    cc_set_tangent1(contacts, 0, wp.vec3(0, 1, 0))
    cc_set_r0(contacts, 0, wp.vec3(0.017, -0.023, 0.009))
    cc_set_r1(contacts, 0, wp.vec3(-0.014, -0.05, -0.031))
    _write_exact_contact_mobility(tree, response, bodies, contacts, 0, 1, 0, result)


class TestMaximalContactMobility(unittest.TestCase):
    def test_internal_axial_contact_cancels_floating_root(self):
        """Preserve small joint mobility while cancelling shared root motion."""
        device = "cpu"
        shift = np.array([-0.031, -0.027, -0.04], dtype=np.float32)
        mapping = np.eye(6, dtype=np.float32)
        x, y, z = shift
        mapping[:3, 3:] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        rng = np.random.default_rng(13)
        factor = (
            rng.normal(size=(6, 6)).astype(np.float32)
            * np.array([10, 10, 10, 100, 100, 100], dtype=np.float32)[:, None]
        )
        root = factor @ factor.T
        motion = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
        conditional = np.outer(motion, motion) * 17
        child = mapping @ root @ mapping.T + conditional
        tree = MaximalTreeProjectorData()
        tree.depth = wp.array([[0, 1]], dtype=wp.int32, device=device)
        tree.parent = wp.array([[-1, 0]], dtype=wp.int32, device=device)
        tree.motion = wp.array([[motion, motion]], dtype=wp.spatial_vectorf, device=device)
        tree.inverse_d = wp.array([[0, 17]], dtype=wp.float32, device=device)
        response = MaximalContactResponseData()
        response.body_articulation = wp.array([0, 0], dtype=wp.int32, device=device)
        response.body_lane = wp.array([0, 1], dtype=wp.int32, device=device)
        response.conditional_map = wp.array([[np.eye(6), mapping]], dtype=wp.spatial_matrixf, device=device)
        response.mobility = wp.array([[root, child]], dtype=wp.spatial_matrixf, device=device)
        bodies = BodyContainer()
        bodies.position = wp.array([[0, 0, 0], -shift], dtype=wp.vec3, device=device)
        result = wp.zeros(1, dtype=wp.float32, device=device)
        point = wp.vec3(0.017, -0.023, 0.009)
        for transverse in (0.0, 0.001, 1.0):
            direction = wp.vec3(transverse, 0, 1)
            wp.launch(evaluate, dim=1, inputs=[tree, response, bodies, point, direction, result], device=device)
            expected = 17 * ((0.023 + 0.027) * transverse) ** 2
            with self.subTest(transverse=transverse):
                self.assertAlmostEqual(float(result.numpy()[0]), expected, delta=1e-11 + expected * 1e-5)

        # Translating the mechanism must not round the prepared contact levers.
        contacts = ContactContainer()
        contacts.lambdas = wp.zeros((32, 1), dtype=float, device=device)
        contacts.derived = wp.zeros((32, 1), dtype=float, device=device)
        mobility = wp.zeros((6, 1), dtype=float, device=device)
        for translation in (0.0, 1000.0):
            bodies.position = wp.array(np.array([[0, 0, 0], -shift]) + translation, dtype=wp.vec3, device=device)
            wp.launch(
                evaluate_offset_mobility, dim=1, inputs=[tree, response, bodies, contacts, mobility], device=device
            )
            expected_inverse = 17 * (0.05 * 0.001) ** 2
            self.assertAlmostEqual(1.0 / float(mobility.numpy()[0, 0]), expected_inverse, delta=1e-11)

    def test_contact_velocity_retains_small_hinge_motion(self):
        """Resolve a small relative contact speed under a large shared body motion."""
        axis = np.asarray(wp.normalize(wp.vec3(0, 0.001, 1)), dtype=np.float32)
        shift = np.array([0, 0, -0.0199], dtype=np.float32)
        angular = np.array([[1000, 1000, 1000], np.array([1000, 1000, 1000]) + axis], dtype=np.float32)
        linear = np.array(
            [[1000, 1000, 1000], np.array([1000, 1000, 1000]) + np.cross(shift, angular[0])], dtype=np.float32
        )
        tree = MaximalTreeProjectorData()
        tree.parent = wp.array([[-1, 0]], dtype=wp.int32, device="cpu")
        tree.depth = wp.array([[0, 1]], dtype=wp.int32, device="cpu")
        tree.body_slot = wp.array([[0, 1]], dtype=wp.int32, device="cpu")
        tree.shift = wp.array([[np.zeros(3), shift]], dtype=wp.vec3, device="cpu")
        tree.motion = wp.array(
            [[np.zeros(6), np.concatenate((np.zeros(3), axis))]], dtype=wp.spatial_vectorf, device="cpu"
        )
        response = MaximalContactResponseData()
        response.body_lane = wp.array([0, 1], dtype=wp.int32, device="cpu")
        response.body_articulation = wp.array([0, 0], dtype=wp.int32, device="cpu")
        bodies = BodyContainer()
        bodies.velocity = wp.array(linear, dtype=wp.vec3, device="cpu")
        bodies.angular_velocity = wp.array(angular, dtype=wp.vec3, device="cpu")
        result = wp.zeros(1, device="cpu")
        wp.launch(evaluate_velocity, dim=1, inputs=[tree, response, bodies, result], device="cpu")
        joint_speed = np.dot(axis.astype(float), angular[1].astype(float) - angular[0].astype(float))
        expected = -float(np.float32(0.02)) * float(axis[1]) * joint_speed
        self.assertAlmostEqual(float(result.numpy()[0]), expected, delta=1.0e-10)

    def test_owned_warmstart_does_not_create_raw_body_velocity(self):
        """Apply a cached internal normal load through the constrained response."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        parent = builder.add_link()
        child = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 0.0199), wp.quat_identity()))
        for body in (parent, child):
            builder.add_shape_sphere(body, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        root = builder.add_joint_free(parent)
        hinge = builder.add_joint_revolute(
            parent,
            child,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0, 0, 0.0199), wp.quat_identity()),
            collision_filter_parent=False,
        )
        builder.add_articulation([root, hinge])
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
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        cached = np.zeros_like(solver.world._contact_container.impulses.numpy())
        cached[0, 0] = 1000.0
        solver.world._contact_container.impulses.assign(cached)
        direct = solver._direct_equality_system
        original = direct.solve
        observed = []

        def observe(**kwargs):
            observed.append(float(np.max(np.abs(solver.world.bodies.velocity.numpy()))))
            original(**kwargs)

        direct.solve = observe
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        self.assertTrue(observed)
        self.assertLess(max(observed), 1.0)
        np.testing.assert_allclose(state.body_qd.numpy(), 0.0, atol=1.0e-5)

    def test_joint_position_recovery_does_not_launch_internal_contact(self):
        """Keep positional hinge repair out of near-blocked physical contacts."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        parent = builder.add_link()
        child = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 0.01999), wp.quat_identity()))
        for body in (parent, child):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(wp.vec3(0.02, 0, 0), wp.quat_identity()),
                radius=0.01,
                cfg=newton.ModelBuilder.ShapeConfig(density=1000, mu=0.0),
            )
        root = builder.add_joint_free(parent)
        hinge = builder.add_joint_revolute(
            parent,
            child,
            axis=wp.normalize(wp.vec3(0, 0.01, 1)),
            parent_xform=wp.transform(wp.vec3(0, 0, 0.0198), wp.quat_identity()),
            collision_filter_parent=False,
        )
        builder.add_articulation([root, hinge])
        model = builder.finalize(device="cuda:0")
        model.articulation_count = 0
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
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        speeds = []
        integrate = solver.world._integrate_positions

        def observe():
            speeds.append(float(np.max(np.abs(solver.world.bodies.angular_velocity.numpy()))))
            integrate()

        solver.world._integrate_positions = observe
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        self.assertIsNotNone(solver._maximal_contact_response)
        self.assertLess(max(speeds), 1.0)
        pose = state.body_q.numpy()
        parent_pose = wp.transform(wp.vec3(*pose[parent, :3]), wp.quat(*pose[parent, 3:]))
        parent_anchor = np.asarray(wp.transform_point(parent_pose, wp.vec3(0, 0, 0.0198)))
        self.assertLess(float(np.linalg.norm(parent_anchor - pose[child, :3])), 0.000189)

    def test_misaligned_hinge_contact_response_preserves_joint_rows(self):
        """Keep contact impulses in the current hinge Jacobian nullspace."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        parent = builder.add_link()
        child = builder.add_link(
            xform=wp.transform(wp.vec3(0.03, 0, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.004))
        )
        for body in (parent, child):
            builder.add_shape_box(body, hx=0.01, hy=0.02, hz=0.01)
        root = builder.add_joint_free(parent)
        hinge = builder.add_joint_revolute(
            parent, child, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.03, 0, 0), wp.quat_identity())
        )
        builder.add_articulation([root, hinge])
        model = builder.finalize(device="cuda:0")
        model.articulation_count = 0
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model, collision_pipeline=pipeline, articulation_mode="maximal", step_layout="single_world", sor_boost=1.0
        )
        # Inspect the prepared response at the same poses as its Jacobian.
        solver.world._integrate_positions = lambda: None
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0 / 1200)
        response = solver._maximal_contact_response
        response.projector.factor_contact_response()
        response.compute_mobility()
        impulse = np.zeros_like(response.data.impulse.numpy())
        impulse[0, 1] = [0.01, 0.02, 0.03, 0.001, 0.002, 0.003]
        response.data.impulse.assign(impulse)
        response.solve_impulses()
        velocity = np.zeros((model.body_count + 1, 6))
        slots = response.projector.data.body_slot.numpy()[0, :2]
        velocity[slots] = response.data.velocity.numpy()[0, :2]
        direct = solver._direct_equality_system
        row_joint = np.asarray(direct.topology.row_joint)
        structural = direct.joint_to_structural.numpy()[row_joint]
        local = direct.row_local.numpy()
        wrench0 = direct.row_wrench0.numpy()[structural, local]
        wrench1 = direct.row_wrench1.numpy()[structural, local]
        parent_slots = model.joint_parent.numpy()[row_joint] + 1
        child_slots = model.joint_child.numpy()[row_joint] + 1
        residual = np.sum(wrench0 * velocity[parent_slots], axis=1) + np.sum(wrench1 * velocity[child_slots], axis=1)
        np.testing.assert_allclose(residual, 0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
