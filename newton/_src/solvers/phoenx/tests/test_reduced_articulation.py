# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph tests for PhoenX reduced-coordinate articulations."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.reduced import ReducedArticulationSystem


def _make_mixed_tree_builder():
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=2.0)
    slider = builder.add_link(mass=1.5)
    ball = builder.add_link(mass=1.0)
    builder.add_shape_box(root, hx=0.25, hy=0.15, hz=0.1)
    builder.add_shape_box(slider, hx=0.2, hy=0.12, hz=0.1)
    builder.add_shape_box(ball, hx=0.15, hy=0.1, hz=0.08)
    root_joint = builder.add_joint_revolute(
        parent=-1,
        child=root,
        axis=newton.Axis.Z,
        child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()),
    )
    slider_joint = builder.add_joint_prismatic(
        parent=root,
        child=slider,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    ball_joint = builder.add_joint_ball(
        parent=slider,
        child=ball,
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root_joint, slider_joint, ball_joint])
    return builder


def _make_mixed_tree(device):
    return _make_mixed_tree_builder().finalize(device=device)


def _make_floating_tree(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=2.0)
    child = builder.add_link(mass=1.0)
    builder.add_shape_box(root, hx=0.25, hy=0.15, hz=0.1)
    builder.add_shape_box(child, hx=0.2, hy=0.1, hz=0.08)
    free_joint = builder.add_joint_free(parent=-1, child=root)
    hinge_joint = builder.add_joint_revolute(
        parent=root,
        child=child,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, hinge_joint])
    return builder.finalize(device=device)


def _make_contact_momentum_model(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=1.0)
    child = builder.add_link(mass=0.5)
    collider = builder.add_link(mass=1.0)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    builder.add_shape_sphere(root, radius=0.2, cfg=shape_cfg)
    builder.add_shape_sphere(collider, radius=0.2, cfg=shape_cfg)
    free_joint = builder.add_joint_free(parent=-1, child=root)
    hinge_joint = builder.add_joint_revolute(
        parent=root,
        child=child,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, hinge_joint])
    return builder.finalize(device=device), collider


def _make_self_contact_model(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=2.0)
    middle = builder.add_link(mass=1.0)
    tip = builder.add_link(mass=1.5)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    builder.add_shape_sphere(root, radius=0.2, cfg=shape_cfg)
    builder.add_shape_sphere(tip, radius=0.2, cfg=shape_cfg)
    free_joint = builder.add_joint_free(parent=-1, child=root)
    middle_joint = builder.add_joint_revolute(
        parent=root,
        child=middle,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
    )
    tip_joint = builder.add_joint_revolute(
        parent=middle,
        child=tip,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, middle_joint, tip_joint])
    return builder.finalize(device=device)


@wp.kernel
def _compute_body_momentum(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    momentum: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    transform = body_q[body]
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(transform))
    position_com = wp.transform_point(transform, body_com[body])
    twist = body_qd[body]
    linear = body_mass[body] * wp.spatial_top(twist)
    inertia_world = rotation * body_inertia[body] * wp.transpose(rotation)
    angular = inertia_world * wp.spatial_bottom(twist) + wp.cross(position_com, linear)
    momentum[body] = wp.spatial_vector(linear, angular)


def _total_momentum(model, state):
    momentum = wp.empty(model.body_count, dtype=wp.spatial_vector, device=model.device)
    wp.launch(
        _compute_body_momentum,
        dim=model.body_count,
        inputs=[state.body_q, state.body_qd, model.body_com, model.body_mass, model.body_inertia],
        outputs=[momentum],
        device=model.device,
    )
    return np.sum(momentum.numpy(), axis=0)


class TestReducedArticulation(unittest.TestCase):
    def test_aba_matches_common_mass_matrix_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_mixed_tree(device)
        state = model.state()
        q = state.joint_q.numpy()
        q[0] = 0.35
        q[1] = 0.12
        q[2:6] = np.asarray(wp.quat_rpy(0.2, -0.15, 0.3), dtype=np.float32)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        system = ReducedArticulationSystem(model)
        tau_np = np.array([0.7, -0.4, 0.2, -0.3, 0.5], dtype=np.float32)
        tau = wp.array(tau_np, dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            system.factor(state)
            result = system.solve_generalized(tau)
        wp.capture_launch(capture.graph)

        h = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
        h += np.diag(model.joint_armature.numpy())
        expected = np.linalg.solve(h.astype(np.float64), tau_np.astype(np.float64))
        np.testing.assert_allclose(result.numpy(), expected, rtol=2.0e-4, atol=2.0e-5)

    def test_common_solver_api_matches_featherstone_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_mixed_tree(device)
        state_phoenx = model.state()
        state_featherstone = model.state()
        output_phoenx = model.state()
        output_featherstone = model.state()
        q = state_phoenx.joint_q.numpy()
        q[0] = 0.35
        q[1] = 0.12
        q[2:6] = np.asarray(wp.quat_rpy(0.2, -0.15, 0.3), dtype=np.float32)
        qd = np.array([0.3, -0.2, 0.1, -0.15, 0.25], dtype=np.float32)
        state_phoenx.joint_q.assign(q)
        state_phoenx.joint_qd.assign(qd)
        state_featherstone.joint_q.assign(q)
        state_featherstone.joint_qd.assign(qd)
        newton.eval_fk(model, state_phoenx.joint_q, state_phoenx.joint_qd, state_phoenx)
        newton.eval_fk(model, state_featherstone.joint_q, state_featherstone.joint_qd, state_featherstone)

        control = model.control()
        control.joint_f.assign(np.array([0.7, -0.4, 0.2, -0.3, 0.5], dtype=np.float32))
        phoenx = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
        )
        featherstone = newton.solvers.SolverFeatherstone(model)
        dt = 1.0 / 240.0

        with wp.ScopedCapture(device=device) as capture_phoenx:
            phoenx.step(state_phoenx, output_phoenx, control, None, dt)
        wp.capture_launch(capture_phoenx.graph)
        with wp.ScopedCapture(device=device) as capture_featherstone:
            featherstone.step(state_featherstone, output_featherstone, control, None, dt)
        wp.capture_launch(capture_featherstone.graph)

        np.testing.assert_allclose(
            output_phoenx.joint_qd.numpy(), output_featherstone.joint_qd.numpy(), rtol=3.0e-4, atol=3.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.joint_q.numpy(), output_featherstone.joint_q.numpy(), rtol=3.0e-4, atol=3.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.body_q.numpy(), output_featherstone.body_q.numpy(), rtol=3.0e-4, atol=3.0e-5
        )

    def test_floating_common_solver_api_matches_featherstone_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state_phoenx = model.state()
        state_featherstone = model.state()
        output_phoenx = model.state()
        output_featherstone = model.state()
        q = state_phoenx.joint_q.numpy()
        q[:3] = np.array([0.2, -0.1, 0.4], dtype=np.float32)
        q[3:7] = np.asarray(wp.quat_rpy(0.2, -0.1, 0.3), dtype=np.float32)
        q[7] = 0.25
        qd = np.array([0.4, -0.2, 0.15, 0.3, -0.1, 0.2, -0.35], dtype=np.float32)
        body_f = np.array(
            [
                [0.3, -0.2, 0.1, -0.15, 0.25, 0.05],
                [-0.1, 0.15, -0.05, 0.2, -0.1, 0.3],
            ],
            dtype=np.float32,
        )
        for state in (state_phoenx, state_featherstone):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            state.body_f.assign(body_f)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        control = model.control()
        control.joint_f.assign(np.array([0.2, -0.1, 0.15, -0.3, 0.25, 0.1, 0.6], dtype=np.float32))
        phoenx = newton.solvers.SolverPhoenX(
            model, articulation_mode="reduced", substeps=1, solver_iterations=1, velocity_iterations=1
        )
        featherstone = newton.solvers.SolverFeatherstone(model)
        dt = 1.0 / 240.0

        with wp.ScopedCapture(device=device) as capture_phoenx:
            phoenx.step(state_phoenx, output_phoenx, control, None, dt)
        wp.capture_launch(capture_phoenx.graph)
        with wp.ScopedCapture(device=device) as capture_featherstone:
            featherstone.step(state_featherstone, output_featherstone, control, None, dt)
        wp.capture_launch(capture_featherstone.graph)

        np.testing.assert_allclose(
            output_phoenx.joint_qd.numpy(), output_featherstone.joint_qd.numpy(), rtol=6.0e-4, atol=6.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.joint_q.numpy(), output_featherstone.joint_q.numpy(), rtol=6.0e-4, atol=6.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.body_q.numpy(), output_featherstone.body_q.numpy(), rtol=6.0e-4, atol=6.0e-5
        )

    def test_multi_world_common_api_matches_featherstone_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        world_count = 8
        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        builder.replicate(_make_mixed_tree_builder(), world_count=world_count)
        model = builder.finalize(device=device)
        state_phoenx = model.state()
        state_featherstone = model.state()
        output_phoenx = model.state()
        output_featherstone = model.state()

        q = state_phoenx.joint_q.numpy().reshape(world_count, -1)
        qd = state_phoenx.joint_qd.numpy().reshape(world_count, -1)
        for world in range(world_count):
            scale = float(world + 1) / float(world_count)
            q[world, 0] = 0.3 * scale
            q[world, 1] = -0.1 * scale
            q[world, 2:6] = np.asarray(wp.quat_rpy(0.1 * scale, -0.2 * scale, 0.15 * scale))
            qd[world] = np.array([0.2, -0.1, 0.05, -0.15, 0.25], dtype=np.float32) * scale
        for state in (state_phoenx, state_featherstone):
            state.joint_q.assign(q.reshape(-1))
            state.joint_qd.assign(qd.reshape(-1))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        control = model.control()
        joint_f = np.empty((world_count, 5), dtype=np.float32)
        for world in range(world_count):
            scale = float(world + 1) / float(world_count)
            joint_f[world] = np.array([0.6, -0.3, 0.2, -0.4, 0.1], dtype=np.float32) * scale
        control.joint_f.assign(joint_f.reshape(-1))
        phoenx = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
        )
        featherstone = newton.solvers.SolverFeatherstone(model)
        dt = 1.0 / 240.0

        with wp.ScopedCapture(device=device) as capture_phoenx:
            phoenx.step(state_phoenx, output_phoenx, control, None, dt)
        wp.capture_launch(capture_phoenx.graph)
        with wp.ScopedCapture(device=device) as capture_featherstone:
            featherstone.step(state_featherstone, output_featherstone, control, None, dt)
        wp.capture_launch(capture_featherstone.graph)

        np.testing.assert_allclose(
            output_phoenx.joint_qd.numpy(), output_featherstone.joint_qd.numpy(), rtol=4.0e-4, atol=4.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.joint_q.numpy(), output_featherstone.joint_q.numpy(), rtol=4.0e-4, atol=4.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.body_q.numpy(), output_featherstone.body_q.numpy(), rtol=4.0e-4, atol=4.0e-5
        )

    def test_self_contact_conserves_spatial_momentum_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_self_contact_model(device)
        state = model.state()
        output = model.state()
        qd = state.joint_qd.numpy()
        qd[0] = 0.4
        qd[1] = -0.15
        qd[5] = 0.2
        qd[6] = -0.3
        qd[7] = 0.25
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=0,
        )
        contacts = model.contacts()
        momentum_before = _total_momentum(model, state)
        dt = 1.0 / 2000.0

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, dt)
        wp.capture_launch(capture.graph)

        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        momentum_after = _total_momentum(model, output)
        np.testing.assert_allclose(momentum_after, momentum_before, rtol=0.0, atol=1.0e-5)

        fk_state = model.state()
        newton.eval_fk(model, output.joint_q, output.joint_qd, fk_state)
        np.testing.assert_allclose(output.body_qd.numpy(), fk_state.body_qd.numpy(), atol=2.0e-5)

    def test_contact_conserves_spatial_momentum_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model, collider = _make_contact_momentum_model(device)
        state = model.state()
        output = model.state()
        q = state.joint_q.numpy()
        q[0] = -0.19
        q[6] = 1.0
        qd = np.zeros(model.joint_dof_count, dtype=np.float32)
        qd[0] = 0.5
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        body_q = state.body_q.numpy()
        body_q[collider, :3] = np.array([0.19, 0.0, 0.0], dtype=np.float32)
        body_q[collider, 3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        state.body_q.assign(body_q)
        body_qd = state.body_qd.numpy()
        body_qd[collider, 0] = -0.5
        state.body_qd.assign(body_qd)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=0,
        )
        contacts = model.contacts()
        momentum_before = _total_momentum(model, state)
        dt = 1.0 / 2000.0

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, dt)
        wp.capture_launch(capture.graph)

        momentum_after = _total_momentum(model, output)
        np.testing.assert_allclose(momentum_after[:3], momentum_before[:3], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(momentum_after[3:], momentum_before[3:], rtol=0.0, atol=1.0e-5)

        fk_state = model.state()
        newton.eval_fk(model, output.joint_q, output.joint_qd, fk_state)
        np.testing.assert_allclose(output.body_qd.numpy()[:2], fk_state.body_qd.numpy()[:2], atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
