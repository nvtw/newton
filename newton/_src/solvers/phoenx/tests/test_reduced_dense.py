# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph tests for the dense backend of PhoenX reduced articulations."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.dense_substep import PAGE_ROWS
from newton._src.solvers.phoenx.tests.test_reduced_articulation import (
    _make_branched_tree,
    _make_floating_mixed_tree,
    _total_momentum,
)


def _require_cuda(test: unittest.TestCase) -> wp.Device:
    device = wp.get_preferred_device()
    if not device.is_cuda:
        test.skipTest("dense reduced articulations require CUDA graph capture")
    return device


def _make_legged_fleet(device, *, worlds: int, gravity: float = -9.81) -> newton.Model:
    """Floating trunk with two driven three-link legs standing on a ground plane, replicated."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, gravity), up_axis=newton.Axis.Z)
    trunk = template.add_link(mass=6.0)
    template.add_shape_box(trunk, hx=0.2, hy=0.12, hz=0.08)
    joints = [template.add_joint_free(parent=-1, child=trunk)]
    for side in (-1.0, 1.0):
        parent = trunk
        offset = wp.vec3(0.0, 0.1 * side, -0.08)
        for segment in range(3):
            link = template.add_link(mass=0.8)
            template.add_shape_capsule(link, radius=0.03, half_height=0.08)
            axis = newton.Axis.Y if segment != 1 else newton.Axis.X
            joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=axis,
                    parent_xform=wp.transform(offset, wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.11), wp.quat_identity()),
                    target_ke=400.0,
                    target_kd=8.0,
                    armature=0.01,
                )
            )
            parent = link
            offset = wp.vec3(0.0, 0.0, -0.11)
    template.add_articulation(joints)
    template.joint_q[2] = 0.75
    template.add_ground_plane()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity), up_axis=newton.Axis.Z)
    builder.replicate(template, world_count=worlds)
    return builder.finalize(device=device)


def _make_centipede_fleet(device, *, worlds: int) -> newton.Model:
    """Floating trunk on twelve driven legs with sphere feet, without self-collision, replicated.

    Its ground contacts span two contact pages of the fused substep.
    """
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    trunk = template.add_link(mass=4.0)
    shapes = [template.add_shape_box(trunk, hx=0.36, hy=0.1, hz=0.04)]
    joints = [template.add_joint_free(parent=-1, child=trunk)]
    for leg in range(12):
        x = -0.3 + 0.6 * (leg // 2) / 5.0
        y = 0.12 if leg % 2 else -0.12
        link = template.add_link(mass=0.3)
        shapes.append(template.add_shape_sphere(link, radius=0.03))
        joints.append(
            template.add_joint_revolute(
                parent=trunk,
                child=link,
                axis=newton.Axis.X,
                parent_xform=wp.transform(wp.vec3(x, y, -0.04), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity()),
                target_ke=200.0,
                target_kd=5.0,
                armature=0.01,
            )
        )
    template.add_articulation(joints)
    for index, shape_a in enumerate(shapes):
        for shape_b in shapes[index + 1 :]:
            template.add_shape_collision_filter_pair(shape_a, shape_b)
    template.joint_q[2] = 0.19
    template.add_ground_plane()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count=worlds)
    return builder.finalize(device=device)


def _randomize_state(model: newton.Model, seed: int) -> newton.State:
    rng = np.random.default_rng(seed)
    state = model.state()
    q = model.joint_q.numpy().copy()
    joint_type = model.joint_type.numpy()
    q_start = model.joint_q_start.numpy()
    for joint, kind in enumerate(joint_type):
        if kind in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
            q[q_start[joint]] += rng.uniform(-0.7, 0.7)
        elif kind in (int(newton.JointType.FREE), int(newton.JointType.BALL)):
            rotation = rng.normal(size=4)
            offset = 3 if kind == int(newton.JointType.FREE) else 0
            q[q_start[joint] + offset : q_start[joint] + offset + 4] = rotation / np.linalg.norm(rotation)
    state.joint_q.assign(q)
    state.joint_qd.assign(rng.uniform(-1.0, 1.0, size=model.joint_dof_count).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _make_chain_and_free_box(device, *, resting: bool) -> tuple[newton.Model, int]:
    """A two-link chain and a free box: the chain rests on the box, or flies into it without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if resting else 0.0), up_axis=newton.Axis.Z)
    builder.default_shape_cfg.gap = 0.005
    builder.default_shape_cfg.mu = 0.6
    height = 0.34 if resting else 0.0
    previous = -1
    for i in range(2):
        body = builder.add_link(xform=wp.transform(wp.vec3(0.3 * i, 0.0, height), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.14, hy=0.14, hz=0.14)
        if previous < 0:
            builder.add_joint_free(body)
        else:
            builder.add_joint_revolute(
                previous,
                body,
                parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y,
            )
        previous = body
    builder.add_articulation(list(range(builder.joint_count)))
    if resting:
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.6, hy=0.4, hz=0.1)
    else:
        box = builder.add_body(xform=wp.transform(wp.vec3(0.95, 0.1, 0.12), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.2, hy=0.2, hz=0.2)
        builder.joint_qd[0] = 1.0
        builder.joint_qd[4] = 0.5
        builder.joint_qd[6] = 2.0
    return builder.finalize(device=device), box


def _momentum(model: newton.Model, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
    """Linear and angular momentum of all bodies about the world origin."""
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()
    com = model.body_com.numpy()
    linear = np.zeros(3)
    angular = np.zeros(3)
    for body in range(model.body_count):
        rotation = np.array(wp.quat_to_matrix(wp.quat(*body_q[body, 3:]))).reshape(3, 3)
        position = body_q[body, :3] + rotation @ com[body]
        linear += mass[body] * body_qd[body, :3]
        angular += np.cross(position, mass[body] * body_qd[body, :3])
        angular += rotation @ inertia[body] @ rotation.T @ body_qd[body, 3:]
    return linear, angular


def _run_chain_and_free_box(model: newton.Model, friction: str, frames: int):
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        joint_mode="reduced",
        contact_friction_model=friction,
        substeps=4,
        solver_iterations=4,
    )
    state, output = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = pipeline.contacts()
    with wp.ScopedCapture() as capture:
        pipeline.collide(state, contacts)
        solver.step(state, output, None, contacts, 1.0 / 60.0)
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            wp.copy(getattr(state, name), getattr(output, name))
    start = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, start)
    for _ in range(frames):
        wp.capture_launch(capture.graph)
    return solver, start, state


class TestReducedDenseBackend(unittest.TestCase):
    def test_fused_substep_exchanges_momentum_with_a_free_body_under_graph_capture(self):
        """A tree hitting a free rigid body trades momentum with it inside the fused substep."""
        device = _require_cuda(self)
        for friction in ("patch", "point"):
            with self.subTest(friction=friction):
                model, box = _make_chain_and_free_box(device, resting=False)
                solver, start, state = _run_chain_and_free_box(model, friction, frames=60)
                articulation = solver._reduced_articulation
                self.assertIsNotNone(articulation.fused_substep)
                self.assertTrue(articulation.external_rigid_rows)
                self.assertFalse(articulation.contact_block_system.requires_impulse_response)
                linear_start, angular_start = _momentum(model, start)
                linear, angular = _momentum(model, state)
                # The chain carried about 44 kg m/s into the box.
                np.testing.assert_allclose(linear, linear_start, atol=2.0e-2)
                np.testing.assert_allclose(angular, angular_start, atol=5.0e-2)
                self.assertGreater(float(state.body_qd.numpy()[box, 0]), 0.2)

    def test_fused_substep_rests_a_link_on_another_link_under_graph_capture(self):
        """A chain folded onto its own base rests there as it does with the articulated-body path."""
        device = _require_cuda(self)
        rest = {}
        for backend in ("aba", "auto"):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
            builder.default_shape_cfg.gap = 0.005
            builder.default_shape_cfg.mu = 0.6
            builder.add_ground_plane()
            # A base on the ground, an upright link hinged at its end, and a link that folds back onto the base.
            base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(base, hx=0.5, hy=0.2, hz=0.1)
            builder.add_joint_free(base)
            upright = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_box(upright, hx=0.08, hy=0.15, hz=0.3)
            builder.add_joint_revolute(
                base,
                upright,
                parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.3), wp.quat_identity()),
                axis=newton.Axis.Y,
            )
            folded = builder.add_link(xform=wp.transform(wp.vec3(0.1, 0.0, 0.62), wp.quat_identity()))
            builder.add_shape_box(folded, hx=0.4, hy=0.15, hz=0.08)
            builder.add_joint_revolute(
                upright,
                folded,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y,
            )
            builder.add_articulation(list(range(builder.joint_count)))
            model = builder.finalize(device=device)
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                joint_mode="reduced",
                reduced_backend=backend,
                substeps=4,
                solver_iterations=4,
            )
            state, output = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            contacts = pipeline.contacts()
            with wp.ScopedCapture() as capture:
                pipeline.collide(state, contacts)
                solver.step(state, output, None, contacts, 1.0 / 60.0)
                for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                    wp.copy(getattr(state, name), getattr(output, name))
            for _ in range(180):
                wp.capture_launch(capture.graph)
            articulation = solver._reduced_articulation
            self.assertEqual(articulation.fused_substep is not None, backend == "auto")
            self.assertEqual(articulation.contact_block_system.internal_rows, backend == "auto")
            self.assertLess(float(np.abs(state.body_qd.numpy()).max()), 2.0e-2)
            rest[backend] = state.body_q.numpy()[:, :3].copy()
        # The folded link holds the upright one up.
        self.assertGreater(float(rest["auto"][folded, 2]), 0.5)
        np.testing.assert_allclose(rest["auto"], rest["aba"], atol=2.0e-2)

    def test_fused_substep_rests_a_tree_on_a_free_body_under_graph_capture(self):
        """A tree stands on a free rigid body whose ground contact the regular solver holds."""
        device = _require_cuda(self)
        for friction in ("patch", "point"):
            with self.subTest(friction=friction):
                model, box = _make_chain_and_free_box(device, resting=True)
                solver, _start, state = _run_chain_and_free_box(model, friction, frames=90)
                self.assertTrue(solver._reduced_articulation.external_rigid_rows)
                body_q = state.body_q.numpy()
                self.assertAlmostEqual(float(body_q[box, 2]), 0.1, delta=2.0e-3)
                np.testing.assert_allclose(body_q[:2, 2], 0.34, atol=4.0e-3)
                self.assertLess(float(np.abs(state.body_qd.numpy()).max()), 2.0e-2)

    def test_dense_mass_solve_matches_articulated_body_factor_under_graph_capture(self):
        """The tree-sparse factor applies the same inverse mass as the ABA factor on random poses."""
        device = _require_cuda(self)
        for make_model in (_make_floating_mixed_tree, _make_branched_tree, lambda d: _make_legged_fleet(d, worlds=2)):
            model = make_model(device)
            solver = newton.solvers.SolverPhoenX(model, joint_mode="reduced", reduced_backend="dense")
            articulation = solver._reduced_articulation
            system = articulation.system
            dense = system.dense
            dense.requires_aba = True
            for seed in range(3):
                state = _randomize_state(model, seed)
                articulation.import_step(state, model.control())
                rng = np.random.default_rng(100 + seed)
                tau = rng.uniform(-1.0, 1.0, size=model.joint_dof_count).astype(np.float32)
                rhs = wp.array(tau, dtype=wp.float32, device=device)
                generalized = wp.array(tau, dtype=wp.float32, device=device)
                with wp.ScopedCapture(device=device) as capture:
                    system.factor(articulation.state, articulation.control, 1.0 / 240.0)
                    aba = system.solve_generalized(generalized)
                    solution = dense.solve(rhs)
                wp.capture_launch(capture.graph)
                expected = aba.numpy().astype(np.float64)
                np.testing.assert_allclose(
                    solution.numpy(), expected, rtol=0.0, atol=2.0e-5 * max(1.0, np.abs(expected).max())
                )
                # M is SPD: every pivot of L is positive.
                factor = dense.data.factor.numpy()
                entries = dense.data.dof_entry.numpy()
                self.assertTrue(np.all(np.isfinite(factor)))
                self.assertTrue(np.all(factor[entries[1:] - 1] > 0.0))

    def test_dense_contact_rows_match_articulated_body_response_under_graph_capture(self):
        """Packed responses equal M^-1 J^T of the ABA factor at the same pose."""
        device = _require_cuda(self)
        model = _make_legged_fleet(device, worlds=3)
        # Patch solves reuse the row tables as caches once built, so only point rows can be rebuilt.
        solver = newton.solvers.SolverPhoenX(
            model,
            joint_mode="reduced",
            reduced_backend="dense",
            contact_friction_model="point",
            substeps=2,
            velocity_iterations=1,
        )
        state0, state1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
        contacts = model.contacts()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(2):
                model.collide(state0, contacts)
                solver.step(state0, state1, model.control(), contacts, 1.0 / 60.0)
                state0, state1 = state1, state0
        for _ in range(20):
            wp.capture_launch(capture.graph)

        articulation = solver._reduced_articulation
        system = articulation.system
        block = articulation.contact_block_system
        row_count = block.row_count.numpy().reshape(model.articulation_count, -1)[:, 0]
        self.assertTrue(np.all(row_count > 0), "every leg should touch the ground")
        # Rebuild the relax rows with both factors at the final pose.
        system.dense.requires_aba = True
        system.factor(articulation.state, articulation.control, 1.0 / 120.0)
        block.page_index.zero_()
        system.dense.build_contact_rows(block, solver.bodies, solver.world._contact_container, True)
        jacobian = block.packed_jacobian.numpy()
        response = block.packed_response.numpy()
        dof_start = model.joint_qd_start.numpy()[model.articulation_start.numpy()[: model.articulation_count]]
        dofs = int(model.joint_dof_count // model.articulation_count)
        for row in range(int(row_count.max())):
            rhs = np.zeros(model.joint_dof_count, dtype=np.float32)
            for articulation_index, start in enumerate(dof_start):
                if row < row_count[articulation_index]:
                    rhs[start : start + dofs] = jacobian[row, start : start + dofs]
            expected = system.solve_generalized(wp.array(rhs, dtype=wp.float32, device=device)).numpy()
            for articulation_index, start in enumerate(dof_start):
                if row < row_count[articulation_index]:
                    np.testing.assert_allclose(
                        response[row, start : start + dofs],
                        expected[start : start + dofs],
                        rtol=0.0,
                        atol=1.0e-5 * max(1.0, np.abs(expected[start : start + dofs]).max()),
                    )

    def test_dense_backend_tracks_aba_bitwise_deterministically_under_graph_capture(self):
        """Standing legged fleets match the ABA backend closely and replay bitwise."""
        device = _require_cuda(self)
        model = _make_legged_fleet(device, worlds=4)

        def simulate(backend: str) -> tuple[np.ndarray, np.ndarray]:
            solver = newton.solvers.SolverPhoenX(
                model,
                joint_mode="reduced",
                reduced_backend=backend,
                contact_friction_model="patch",
                substeps=2,
                solver_iterations=4,
                velocity_iterations=1,
            )
            state0, state1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
            control = model.control()
            control.joint_target_q.assign(model.joint_q)
            contacts = model.contacts()
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(2):
                    model.collide(state0, contacts)
                    solver.step(state0, state1, control, contacts, 1.0 / 60.0)
                    state0, state1 = state1, state0
            for _ in range(60):
                wp.capture_launch(capture.graph)
            return state0.joint_q.numpy(), state0.body_q.numpy()

        dense_q, dense_body = simulate("dense")
        repeat_q, repeat_body = simulate("dense")
        aba_q, _ = simulate("aba")
        np.testing.assert_array_equal(dense_q, repeat_q)
        np.testing.assert_array_equal(dense_body, repeat_body)
        self.assertTrue(np.all(np.isfinite(dense_q)))
        trunk_height = dense_body.reshape(model.articulation_count, -1, 7)[:, 0, 2]
        self.assertTrue(np.all(trunk_height > 0.3), trunk_height)
        np.testing.assert_allclose(dense_q, aba_q, rtol=0.0, atol=2.0e-3)

    def test_dense_backend_conserves_momentum_of_free_tree_under_graph_capture(self):
        """Without contacts or gravity, a spinning floating tree keeps its spatial momentum."""
        device = _require_cuda(self)
        model = _make_floating_mixed_tree(device)
        state = _randomize_state(model, 7)
        output = model.state()
        solver = newton.solvers.SolverPhoenX(
            model, joint_mode="reduced", reduced_backend="dense", substeps=4, velocity_iterations=0
        )
        momentum_before = _total_momentum(model, state)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state, output, None, None, 1.0 / 240.0)
            solver.step(output, state, None, None, 1.0 / 240.0)
        for _ in range(50):
            wp.capture_launch(capture.graph)
        momentum_after = _total_momentum(model, state)
        np.testing.assert_allclose(
            momentum_after, momentum_before, rtol=0.0, atol=2.0e-4 * np.abs(momentum_before).max()
        )

    def test_heterogeneous_fleet_mixes_buckets_and_articulated_body_fallback_under_graph_capture(self):
        """Trees of four DOF buckets and a 66-DOF chain share one model; the chain keeps the ABA path."""
        device = _require_cuda(self)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        sizes = (3, 20, 40, 60, 66)
        for index, link_count in enumerate(sizes):
            parent = -1
            joints = []
            # Chains lie along x on the ground and bend vertically and sideways.
            for link_index in range(link_count):
                link = builder.add_link(mass=0.2)
                builder.add_shape_sphere(link, radius=0.03)
                xform = wp.transform(wp.vec3(0.0, 0.5 * index, 0.031), wp.quat_identity())
                if link_index > 0:
                    xform = wp.transform(wp.vec3(0.07, 0.0, 0.0), wp.quat_identity())
                axis = (newton.Axis.Z, newton.Axis.Y)[link_index % 2]
                joints.append(
                    builder.add_joint_revolute(parent=parent, child=link, axis=axis, parent_xform=xform, armature=0.01)
                )
                parent = link
            builder.add_articulation(joints)
        model = builder.finalize(device=device)

        def simulate(backend: str) -> np.ndarray:
            solver = newton.solvers.SolverPhoenX(
                model, joint_mode="reduced", reduced_backend=backend, substeps=2, velocity_iterations=1
            )
            if backend == "dense":
                dense = solver._reduced_articulation.system.dense
                self.assertEqual(sorted(bucket.max_dofs for bucket in dense.buckets), [16, 24, 40, 64])
                self.assertTrue(dense.has_fallback and dense.requires_aba)
            state0, state1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
            contacts = model.contacts()
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(2):
                    model.collide(state0, contacts)
                    solver.step(state0, state1, None, contacts, 1.0 / 120.0)
                    state0, state1 = state1, state0
            for _ in range(30):
                wp.capture_launch(capture.graph)
            return state0.joint_q.numpy()

        dense_q = simulate("dense")
        aba_q = simulate("aba")
        self.assertTrue(np.all(np.isfinite(dense_q)))
        np.testing.assert_allclose(dense_q, aba_q, rtol=0.0, atol=5.0e-3)

    def test_fused_substep_tracks_dense_operator_bitwise_deterministically_under_graph_capture(self):
        """Fused point and patch substeps over two contact pages track the unfused dense backend."""
        device = _require_cuda(self)
        model = _make_centipede_fleet(device, worlds=3)

        def simulate(friction: str, fused: bool) -> tuple[np.ndarray, np.ndarray]:
            solver = newton.solvers.SolverPhoenX(
                model,
                joint_mode="reduced",
                reduced_backend="dense",
                contact_friction_model=friction,
                substeps=3,
                solver_iterations=2,
                velocity_iterations=1,
            )
            articulation = solver._reduced_articulation
            self.assertIsNotNone(articulation.fused_substep)
            if not fused:
                articulation.fused_substep = None
            state0, state1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
            control = model.control()
            control.joint_target_q.assign(model.joint_q)
            contacts = model.contacts()
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(2):
                    model.collide(state0, contacts)
                    solver.step(state0, state1, control, contacts, 1.0 / 60.0)
                    state0, state1 = state1, state0
            for _ in range(30):
                wp.capture_launch(capture.graph)
            # More than one fused contact page per tree.
            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), PAGE_ROWS // 3 * model.articulation_count)
            return state0.joint_q.numpy(), state0.joint_qd.numpy()

        for friction in ("point", "patch"):
            fused_q, fused_qd = simulate(friction, True)
            repeat_q, repeat_qd = simulate(friction, True)
            unfused_q, _ = simulate(friction, False)
            np.testing.assert_array_equal(fused_q, repeat_q)
            np.testing.assert_array_equal(fused_qd, repeat_qd)
            trunk_height = fused_q.reshape(model.articulation_count, -1)[:, 2]
            self.assertTrue(np.all(trunk_height > 0.15), trunk_height)
            np.testing.assert_allclose(fused_q, unfused_q, rtol=0.0, atol=1.0e-3)

    def test_fused_substep_solves_root_paths_longer_than_a_warp_under_graph_capture(self):
        """A ball-jointed chain's contact rows span 42 DOFs, more path positions than lanes."""
        device = _require_cuda(self)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.collision_group = -7  # The links collide with the ground only.
        parent = -1
        joints = []
        # The chain lies along x on the ground and is nudged sideways at its far end.
        for link_index in range(14):
            link = builder.add_link(mass=0.2)
            builder.add_shape_sphere(link, radius=0.03, cfg=cfg)
            xform = wp.transform(wp.vec3(0.0, 0.0, 0.031), wp.quat_identity())
            if link_index > 0:
                xform = wp.transform(wp.vec3(0.07, 0.0, 0.0), wp.quat_identity())
            joints.append(builder.add_joint_ball(parent=parent, child=link, parent_xform=xform, armature=0.01))
            parent = link
        builder.add_articulation(joints)
        builder.joint_qd[-1] = 2.0
        model = builder.finalize(device=device)

        def simulate(fused: bool) -> np.ndarray:
            solver = newton.solvers.SolverPhoenX(
                model, joint_mode="reduced", reduced_backend="dense", substeps=2, velocity_iterations=1
            )
            articulation = solver._reduced_articulation
            self.assertIsNotNone(articulation.fused_substep)
            if not fused:
                articulation.fused_substep = None
            state0, state1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
            contacts = model.contacts()
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(2):
                    model.collide(state0, contacts)
                    solver.step(state0, state1, None, contacts, 1.0 / 120.0)
                    state0, state1 = state1, state0
            for _ in range(30):
                wp.capture_launch(capture.graph)
            # Every link rests on the ground, the far ones through rows longer than a warp.
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 14)
            return state0.body_q.numpy()

        fused_q = simulate(True)
        self.assertTrue(np.all(np.isfinite(fused_q)))
        np.testing.assert_allclose(fused_q[:, 2], 0.03, rtol=0.0, atol=2.0e-3)
        np.testing.assert_allclose(fused_q, simulate(False), rtol=0.0, atol=1.0e-3)

    def test_auto_backend_selects_dense_only_where_the_fused_substep_runs(self):
        device = _require_cuda(self)
        fleet = _make_centipede_fleet(device, worlds=2)
        solver = newton.solvers.SolverPhoenX(fleet, joint_mode="reduced", reduced_backend="auto")
        self.assertEqual(solver.reduced_backend, "dense")
        self.assertIsNotNone(solver._reduced_articulation.fused_substep)

        def make(second_tree: bool) -> newton.Model:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            builder.add_ground_plane()
            link = builder.add_link(mass=0.2)
            builder.add_shape_sphere(link, radius=0.03)
            builder.add_articulation([builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Y)])
            if second_tree:
                other = builder.add_link(mass=0.2)
                builder.add_shape_sphere(other, radius=0.03)
                builder.add_articulation([builder.add_joint_revolute(parent=-1, child=other, axis=newton.Axis.Y)])
            else:
                builder.add_shape_sphere(builder.add_body(mass=0.2), radius=0.03)
            return builder.finalize(device=device)

        # A free body that may touch one tree joins that tree's fused substep.
        solver = newton.solvers.SolverPhoenX(make(False), joint_mode="reduced", reduced_backend="auto")
        self.assertEqual(solver.reduced_backend, "dense")
        self.assertTrue(solver._reduced_articulation.external_rigid_rows)

        # Trees that may touch each other need the coupled response of the articulated-body path.
        solver = newton.solvers.SolverPhoenX(make(True), joint_mode="reduced", reduced_backend="auto")
        self.assertEqual(solver.reduced_backend, "aba")
        self.assertIsNone(solver._reduced_articulation.fused_substep)

    def test_dense_backend_rejects_models_without_eligible_trees(self):
        device = _require_cuda(self)
        builder = newton.ModelBuilder()
        parent = -1
        joints = []
        for _ in range(66):
            link = builder.add_link(mass=0.1)
            builder.add_shape_sphere(link, radius=0.02)
            joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
            parent = link
        builder.add_articulation(joints)
        model = builder.finalize(device=device)
        with self.assertRaisesRegex(ValueError, "dense reduced backend"):
            newton.solvers.SolverPhoenX(model, joint_mode="reduced", reduced_backend="dense")


if __name__ == "__main__":
    unittest.main()
