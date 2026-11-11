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
        # A second box in the chain's way and a dozen out of reach: a tree may touch several
        # free bodies at once and pair with any number.
        other = builder.add_body(xform=wp.transform(wp.vec3(0.95, -0.35, -0.1), wp.quat_identity()))
        builder.add_shape_box(other, hx=0.2, hy=0.2, hz=0.2)
        for i in range(12):
            far = builder.add_body(xform=wp.transform(wp.vec3(4.0 + 0.5 * i, 3.0, 0.0), wp.quat_identity()))
            builder.add_shape_box(far, hx=0.1, hy=0.1, hz=0.1)
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

    def test_fused_substep_holds_a_link_on_another_link_by_friction_under_graph_capture(self):
        """Friction between two links holds a slider on its tilted base, with either friction model.

        The links can only move against each other along one tangent, so the patch
        tangent block is singular and must still carry friction along that tangent.
        """
        device = _require_cuda(self)
        tilt = 0.3
        scale = 9.81 / np.sqrt(1.0 + tilt * tilt)
        for friction in ("point", "patch"):
            with self.subTest(friction=friction):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
                builder.default_shape_cfg.gap = 0.005
                builder.default_shape_cfg.mu = 0.6
                builder.add_ground_plane()
                base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
                builder.add_shape_box(base, hx=1.0, hy=0.4, hz=0.1)
                builder.add_joint_free(base)
                # A shapeless carriage slides along the base and carries a block that presses on it.
                carriage = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()), mass=0.1)
                builder.add_joint_prismatic(
                    base,
                    carriage,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                    axis=newton.Axis.X,
                )
                block = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()))
                builder.add_shape_box(block, hx=0.2, hy=0.2, hz=0.1)
                builder.add_joint_prismatic(
                    carriage,
                    block,
                    parent_xform=wp.transform_identity(),
                    child_xform=wp.transform_identity(),
                    axis=newton.Axis.Z,
                )
                builder.add_articulation(list(range(builder.joint_count)))
                model = builder.finalize(device=device)
                model.set_gravity((scale * tilt, 0.0, -scale))
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
                for _ in range(120):
                    wp.capture_launch(capture.graph)
                articulation = solver._reduced_articulation
                self.assertIsNotNone(articulation.fused_substep)
                self.assertTrue(articulation.contact_block_system.internal_rows)
                joint_q = state.joint_q.numpy()
                # Without friction between the links the carriage runs 5.6 m in these two seconds.
                self.assertLess(abs(float(joint_q[7])), 5.0e-3)
                self.assertLess(abs(float(joint_q[8])), 2.0e-3)

    def test_fused_substep_holds_a_tree_on_a_second_tree_under_graph_capture(self):
        """A floating chain rests on a tilted plate that hangs on a sprung hinge, a second tree.

        The fused substep solves both trees in one block; the articulated-body path keeps
        them apart and couples them through its fallback. With only the chain selected as
        reduced, the plate is a rigid body on a regular joint that the chain's block pushes.
        """
        device = _require_cuda(self)
        tilt = 0.2
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)
        for friction in ("point", "patch"):
            results = {}
            for case, backend, reduced in (("aba", "aba", None), ("auto", "auto", None), ("chain", "auto", ["chain"])):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
                builder.default_shape_cfg.gap = 0.005
                builder.default_shape_cfg.mu = 0.6
                previous = -1
                for i in range(2):
                    local = wp.vec3(-0.15 + 0.3 * i, 0.0, 0.245)
                    body = builder.add_link(
                        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0) + wp.quat_rotate(rotation, local), rotation)
                    )
                    builder.add_shape_box(body, hx=0.14, hy=0.14, hz=0.14)
                    if previous < 0:
                        builder.add_joint_free(body)
                    else:
                        builder.add_joint_revolute(
                            previous,
                            body,
                            parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
                            child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
                            axis=newton.Axis.Z,
                        )
                    previous = body
                builder.add_articulation(list(range(builder.joint_count)), label="chain")
                plate = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), rotation))
                builder.add_shape_box(plate, hx=1.0, hy=0.6, hz=0.1)
                hinge = builder.add_joint_revolute(
                    -1,
                    plate,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                    axis=newton.Axis.Y,
                    target_ke=2000.0,
                    target_kd=50.0,
                    actuator_mode=newton.JointTargetMode.POSITION,
                )
                builder.add_articulation([hinge], label="plate")
                builder.joint_q[-1] = tilt
                builder.joint_target_q[-1] = tilt
                model = builder.finalize(device=device)
                pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    joint_mode="reduced",
                    reduced_backend=backend,
                    reduced_articulations=reduced,
                    contact_friction_model=friction,
                    substeps=4,
                    solver_iterations=4,
                )
                state, output = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                contacts = pipeline.contacts()
                start = state.body_q.numpy()[0, :3].copy()
                with wp.ScopedCapture() as capture:
                    pipeline.collide(state, contacts)
                    solver.step(state, output, control, contacts, 1.0 / 60.0)
                    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                        wp.copy(getattr(state, name), getattr(output, name))
                for _ in range(120):
                    wp.capture_launch(capture.graph)
                articulation = solver._reduced_articulation
                fused = backend == "auto"
                self.assertEqual(articulation.fused_substep is not None, fused)
                self.assertEqual(int(articulation.model.articulation_count), 1 if fused else 2)
                # One block holds both trees, or the chain alone beside the plate's regular joint.
                self.assertEqual(int(articulation.model.joint_count), model.joint_count)
                self.assertEqual(articulation.external_rigid_rows, case == "chain")
                slid = float(np.linalg.norm(state.body_q.numpy()[0, :3] - start))
                # The plate's angle from its pose: a regular joint does not write joint_q.
                results[case] = (slid, 2.0 * float(np.arcsin(state.body_q.numpy()[plate, 4])))
            for case in ("auto", "chain"):
                with self.subTest(friction=friction, case=case):
                    # Friction holds the chain on the slope, and its weight opens the hinge a little.
                    self.assertLess(results[case][0], 2.0e-2)
                    self.assertAlmostEqual(results[case][0], results["aba"][0], delta=5.0e-3)
                    self.assertAlmostEqual(results[case][1], results["aba"][1], delta=5.0e-3)
                    self.assertGreater(results[case][1], tilt + 1.0e-3)

    def test_fused_substep_follows_notified_joint_parameters_under_graph_capture(self):
        """The fused kernels' packed drive gains follow the model after a change notification."""
        device = _require_cuda(self)

        def simulate(soften_before: bool, soften_after: bool) -> np.ndarray:
            model = _make_legged_fleet(device, worlds=2)
            if soften_before:
                model.joint_target_ke.assign(0.25 * model.joint_target_ke.numpy())
            solver = newton.solvers.SolverPhoenX(model, joint_mode="reduced", substeps=2)
            self.assertIsNotNone(solver._reduced_articulation.fused_substep)
            if soften_after:
                model.joint_target_ke.assign(0.25 * model.joint_target_ke.numpy())
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            state0, state1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
            control = model.control()
            control.joint_target_q.assign(model.joint_q.numpy() + 0.3)
            contacts = model.contacts()
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(2):
                    model.collide(state0, contacts)
                    solver.step(state0, state1, control, contacts, 1.0 / 60.0)
                    state0, state1 = state1, state0
            for _ in range(10):
                wp.capture_launch(capture.graph)
            return state0.joint_q.numpy()

        stiff = simulate(False, False)
        soft = simulate(True, False)
        notified = simulate(False, True)
        self.assertGreater(float(np.abs(soft - stiff).max()), 1.0e-3)
        np.testing.assert_array_equal(notified, soft)

    def test_reduced_contacts_follow_the_requested_contact_frequency_under_graph_capture(self):
        """A requested contact frequency softens the ground contacts of reduced articulations."""
        device = _require_cuda(self)

        def rest_height(backend: str, contact_hertz: float | None) -> float:
            model = _make_legged_fleet(device, worlds=2)
            solver = newton.solvers.SolverPhoenX(
                model, joint_mode="reduced", reduced_backend=backend, substeps=4, contact_hertz=contact_hertz
            )
            state0, state1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
            control = model.control()
            contacts = model.contacts()
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(2):
                    model.collide(state0, contacts)
                    solver.step(state0, state1, control, contacts, 1.0 / 60.0)
                    state0, state1 = state1, state0
            for _ in range(60):
                wp.capture_launch(capture.graph)
            return float(state0.joint_q.numpy()[2])

        for backend in ("auto", "aba"):
            with self.subTest(backend=backend):
                stiff = rest_height(backend, None)
                soft = rest_height(backend, 15.0)
                # The spring of a foot row scales with the row's effective mass, which is
                # a fraction of the carried weight: the trunk settles millimeters lower.
                self.assertLess(soft, stiff - 1.0e-3)
                self.assertGreater(soft, stiff - 2.0e-2)

    def test_fused_substep_couples_trees_like_the_articulated_body_path_under_graph_capture(self):
        """Contacts between two trees, or through a body both carry, match the coupled reference."""
        device = _require_cuda(self)

        # Tilted gravity stays inside the friction cone: friction holds everything in place.
        gravity = ((0.0, 0.0, -9.81), (1.5, 0.8, -9.81))

        def make(scene: str) -> newton.Model:
            builder = newton.ModelBuilder(gravity=gravity[scene == "tilted"], up_axis=newton.Axis.Z)
            builder.default_shape_cfg.gap = 0.005
            builder.default_shape_cfg.mu = 0.6
            builder.add_ground_plane()

            def chain(x: float, y: float, z: float) -> None:
                first = builder.joint_count
                previous = -1
                for i in range(2):
                    body = builder.add_link(xform=wp.transform(wp.vec3(x + 0.3 * i, y, z), wp.quat_identity()))
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
                builder.add_articulation(list(range(first, builder.joint_count)))

            if scene in ("stacked", "tilted"):
                # One chain lies on the other.
                chain(0.0, 0.0, 0.145)
                chain(0.05, 0.02, 0.44)
            else:
                # A free box lies across two chains, or on one of them.
                chain(0.0, -0.3, 0.145)
                chain(0.0, 0.3, 0.145)
                half_width = 0.45 if scene == "carried" else 0.1
                center = 0.0 if scene == "carried" else 0.3
                box = builder.add_body(xform=wp.transform(wp.vec3(0.15, center, 0.4), wp.quat_identity()))
                builder.add_shape_box(box, hx=0.12, hy=half_width, hz=0.1)
            return builder.finalize(device=device)

        def settle(scene: str, backend: str) -> tuple[np.ndarray, np.ndarray]:
            model = make(scene)
            model.request_contact_attributes("force")
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                joint_mode="reduced",
                reduced_backend=backend,
                substeps=4,
                solver_iterations=4,
            )
            self.assertEqual(solver._reduced_articulation.fused_substep is not None, backend == "auto")
            state, output = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            contacts = pipeline.contacts()
            with wp.ScopedCapture() as capture:
                pipeline.collide(state, contacts)
                solver.step(state, output, None, contacts, 1.0 / 60.0)
                for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                    wp.copy(getattr(state, name), getattr(output, name))
            for _ in range(120):
                wp.capture_launch(capture.graph)
            owners = solver.world.bodies.reduced.external_owner.numpy()[1:]
            cross = int(solver._reduced_articulation.contact_block_system.cross_column_count.numpy()[0])
            # Reported force of the first chain on the second, summed over their contacts.
            solver.update_contacts(contacts, state)
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape_body = model.shape_body.numpy()
            body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
            body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
            force = contacts.force.numpy()[:count, :3]
            first, second = (body0 >= 0) & (body0 < 2), (body1 >= 2) & (body1 < 4)
            swapped = (body1 >= 0) & (body1 < 2) & (body0 >= 2) & (body0 < 4)
            self.pair_force = force[swapped].sum(axis=0) - force[first & second].sum(axis=0)
            return state.body_q.numpy(), state.body_qd.numpy(), owners, cross

        for scene in ("stacked", "tilted", "carried", "owned"):
            with self.subTest(scene=scene):
                reference_q, _, _, _ = settle(scene, "aba")
                body_q, body_qd, owners, cross = settle(scene, "auto")
                self.assertTrue(np.isfinite(body_q).all())
                # Nothing sinks into what carries it, and everything comes to rest.
                np.testing.assert_allclose(body_q[:, 2], reference_q[:, 2], atol=5.0e-3)
                self.assertLess(float(np.abs(body_qd).max()), 0.005 if scene in ("stacked", "tilted") else 0.06)
                if scene == "owned":
                    # The one tree under the box solves it in its own block: nothing crosses.
                    self.assertEqual(int(owners[-1]), 1)
                    self.assertEqual(cross, 0)
                elif scene == "carried":
                    # Both trees solve the box, each with half of its mass: nothing crosses.
                    self.assertEqual(int(owners[-1]), -2)
                    self.assertEqual(cross, 0)
                else:
                    # Two trees touch each other: their contacts are solved between the kernels.
                    self.assertTrue(np.all(owners == -1))
                    self.assertGreater(cross, 0)
                    # The contacts between the trees report the force that holds the upper chain:
                    # its weight, and under tilted gravity the friction that keeps it from sliding.
                    mass = 2.0 * 1000.0 * 0.28**3
                    held = -mass * np.asarray(gravity[scene == "tilted"])
                    np.testing.assert_allclose(self.pair_force, held, atol=0.002 * mass * 9.81)
                    # A second run of the same scene repeats every bit.
                    repeat_q, repeat_qd, _, _ = settle(scene, "auto")
                    np.testing.assert_array_equal(body_q, repeat_q)
                    np.testing.assert_array_equal(body_qd, repeat_qd)

    def test_fused_substep_conserves_momentum_when_trees_collide_under_graph_capture(self):
        """Two chains that meet in free flight, and then fold onto themselves, keep their momentum."""
        device = _require_cuda(self)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        builder.default_shape_cfg.gap = 0.005
        builder.default_shape_cfg.mu = 0.6
        for x, y, z, speed in ((0.0, 0.0, 0.0, 1.0), (1.2, 0.1, 0.05, -0.5)):
            first = builder.joint_count
            previous = -1
            for i in range(3):
                body = builder.add_link(xform=wp.transform(wp.vec3(x + 0.3 * i, y, z), wp.quat_identity()))
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
            builder.add_articulation(list(range(first, builder.joint_count)))
            builder.joint_qd[builder.joint_qd_start[first]] = speed
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model, collision_pipeline=pipeline, joint_mode="reduced", substeps=4, solver_iterations=4
        )
        self.assertIsNotNone(solver._reduced_articulation.fused_substep)
        state, output = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        linear_start, angular_start = _momentum(model, state)
        contacts = pipeline.contacts()
        with wp.ScopedCapture() as capture:
            pipeline.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 60.0)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                wp.copy(getattr(state, name), getattr(output, name))
        crossed = False
        touched_itself = False
        for _ in range(240):
            wp.capture_launch(capture.graph)
            cross = int(solver._reduced_articulation.contact_block_system.cross_column_count.numpy()[0])
            crossed = crossed or cross > 0
            touched_itself = touched_itself or (cross == 0 and int(contacts.rigid_contact_count.numpy()[0]) > 0)
        # The chains met each other, and one then touched itself.
        self.assertTrue(crossed)
        self.assertTrue(touched_itself)
        linear, angular = _momentum(model, state)
        # The chains carry about 100 kg m/s.
        np.testing.assert_allclose(linear, linear_start, atol=0.05)
        np.testing.assert_allclose(angular, angular_start, atol=0.03)

    def test_fused_trees_stacked_three_high_rest_under_graph_capture(self):
        """A tree between two others, with friction, comes to rest as stacked rigid bodies do."""
        device = _require_cuda(self)

        def worst_speed(links: int) -> tuple[float, np.ndarray]:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
            builder.default_shape_cfg.gap = 0.005
            builder.default_shape_cfg.mu = 0.6
            builder.add_ground_plane()
            for level, (x, y) in enumerate(((0.0, 0.0), (0.05, 0.02), (0.02, -0.03))):
                first = builder.joint_count
                previous = -1
                for i in range(links):
                    pose = wp.transform(wp.vec3(x + 0.3 * i, y, 0.145 + 0.295 * level), wp.quat_identity())
                    body = builder.add_link(xform=pose)
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
                builder.add_articulation(list(range(first, builder.joint_count)))
            model = builder.finalize(device=device)
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(
                model, collision_pipeline=pipeline, joint_mode="reduced", substeps=4, solver_iterations=4
            )
            self.assertIsNotNone(solver._reduced_articulation.fused_substep)
            state, output = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            contacts = pipeline.contacts()
            with wp.ScopedCapture() as capture:
                pipeline.collide(state, contacts)
                solver.step(state, output, None, contacts, 1.0 / 60.0)
                for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                    wp.copy(getattr(state, name), getattr(output, name))
            worst = 0.0
            for frame in range(360):
                wp.capture_launch(capture.graph)
                if frame >= 240:
                    worst = max(worst, float(np.abs(state.body_qd.numpy()).max()))
            return worst, state.body_q.numpy()[:, 2]

        # The contacts between trees push out a substep late: too stiff a spring kept single
        # boxes rocking at 0.2 m/s, and one stiff enough for them still rocked hinged chains.
        for links in (1, 2):
            with self.subTest(links=links):
                worst, height = worst_speed(links)
                self.assertLess(worst, 0.005)
                np.testing.assert_allclose(height, np.repeat([0.14, 0.42, 0.70], links), atol=3.0e-3)

    def test_fused_tree_leaves_and_rejoins_the_scene_by_its_state_under_graph_capture(self):
        """A tree resting on another is parked far away between steps and placed back: a pool of robots."""
        device = _require_cuda(self)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        builder.default_shape_cfg.gap = 0.005
        builder.default_shape_cfg.mu = 0.6
        builder.add_ground_plane()
        for x, y, z in ((0.0, 0.0, 0.145), (0.05, 0.02, 0.44)):
            first = builder.joint_count
            previous = -1
            for i in range(2):
                body = builder.add_link(xform=wp.transform(wp.vec3(x + 0.3 * i, y, z), wp.quat_identity()))
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
            builder.add_articulation(list(range(first, builder.joint_count)))
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model, collision_pipeline=pipeline, joint_mode="reduced", substeps=4, solver_iterations=4
        )
        self.assertIsNotNone(solver._reduced_articulation.fused_substep)
        state, output = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        contacts = pipeline.contacts()
        with wp.ScopedCapture() as capture:
            pipeline.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 60.0)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                wp.copy(getattr(state, name), getattr(output, name))
        coordinate = int(model.joint_q_start.numpy()[2])
        rate = int(model.joint_qd_start.numpy()[2])
        reduced = solver.world.bodies.reduced
        cross_count = solver._reduced_articulation.contact_block_system.cross_column_count

        def place(x: float, y: float, z: float) -> None:
            joint_q = state.joint_q.numpy()
            joint_qd = state.joint_qd.numpy()
            joint_q[coordinate : coordinate + 7] = (x, y, z, 0.0, 0.0, 0.0, 1.0)
            joint_q[coordinate + 7 :] = 0.0
            joint_qd[rate:] = 0.0
            state.joint_q.assign(joint_q)
            state.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        def run(frames: int) -> tuple[np.ndarray, float]:
            for _ in range(frames):
                wp.capture_launch(capture.graph)
            return state.body_q.numpy()[:, 2], float(np.abs(state.body_qd.numpy()).max())

        height, speed = run(120)
        np.testing.assert_allclose(height, [0.14, 0.14, 0.42, 0.42], atol=2.0e-3)
        self.assertLess(speed, 0.005)
        self.assertGreater(float(np.abs(reduced.carried_wrench.numpy()).max()), 0.5)

        # Parked: both lie on the ground, and the impulses the contacts held are gone.
        place(10.0, 0.0, 0.145)
        height, speed = run(60)
        np.testing.assert_allclose(height, 0.14, atol=2.0e-3)
        self.assertLess(speed, 0.005)
        self.assertEqual(int(cross_count.numpy()[0]), 0)
        self.assertEqual(float(np.abs(reduced.carried_wrench.numpy()).max()), 0.0)

        # Placed back: it rests on the other tree as before.
        place(0.05, 0.02, 0.44)
        height, speed = run(120)
        np.testing.assert_allclose(height, [0.14, 0.14, 0.42, 0.42], atol=2.0e-3)
        self.assertLess(speed, 0.005)
        self.assertGreater(int(cross_count.numpy()[0]), 0)

    def test_two_link_rows_follow_the_collision_filters_without_a_pair_list(self):
        """Whether links of one tree may touch is read from groups and filters when pairs are unknown."""
        from newton._src.solvers.phoenx.articulations.reduced_contact_block import (  # noqa: PLC0415
            _links_of_a_tree_may_touch,
        )

        device = _require_cuda(self)
        for separate_groups in (False, True):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            previous = -1
            for i in range(3):
                body = builder.add_link(xform=wp.transform(wp.vec3(0.3 * i, 0.0, 1.0), wp.quat_identity()))
                cfg = newton.ModelBuilder.ShapeConfig(collision_group=i + 1 if separate_groups else 1)
                builder.add_shape_box(body, hx=0.14, hy=0.14, hz=0.14, cfg=cfg)
                if previous < 0:
                    builder.add_joint_free(body)
                else:
                    builder.add_joint_revolute(previous, body, axis=newton.Axis.Y)
                previous = body
            builder.add_articulation(list(range(builder.joint_count)))
            model = builder.finalize(device=device)
            # The first and the last link share no joint, so only their groups keep them apart.
            self.assertEqual(_links_of_a_tree_may_touch(model), not separate_groups)

    def test_reduced_articulations_selects_by_index_or_label(self):
        device = _require_cuda(self)
        fleet = _make_centipede_fleet(device, worlds=2)
        labels = list(fleet.articulation_label)
        solver = newton.solvers.SolverPhoenX(fleet, joint_mode="reduced", reduced_articulations=[0])
        self.assertEqual(int(solver._reduced_articulation.model.articulation_count), 1)
        # Every world's copy of a tree carries the same label.
        solver = newton.solvers.SolverPhoenX(fleet, joint_mode="reduced", reduced_articulations=[labels[1]])
        self.assertEqual(int(solver._reduced_articulation.model.articulation_count), labels.count(labels[1]))
        with self.assertRaisesRegex(ValueError, "matches no articulation label"):
            newton.solvers.SolverPhoenX(fleet, joint_mode="reduced", reduced_articulations=["no such tree"])
        with self.assertRaisesRegex(ValueError, "outside the model"):
            newton.solvers.SolverPhoenX(fleet, joint_mode="reduced", reduced_articulations=[len(labels)])

    def test_fused_substep_wakes_sleeping_free_bodies_under_graph_capture(self):
        """Free bodies that fell asleep wake when a tree of the fused substep pushes them."""
        device = _require_cuda(self)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        builder.default_shape_cfg.gap = 0.005
        builder.default_shape_cfg.mu = 0.3
        builder.add_ground_plane()
        previous = -1
        for i in range(2):
            body = builder.add_link(xform=wp.transform(wp.vec3(-1.5 + 0.3 * i, 0.0, 0.145), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.14, hy=0.14, hz=0.14)
            if previous < 0:
                builder.add_joint_free(body)
            else:
                builder.add_joint_revolute(
                    previous,
                    body,
                    parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
                    axis=newton.Axis.Z,
                )
            previous = body
        builder.add_articulation(list(range(builder.joint_count)))
        boxes = []
        for i in range(3):
            box = builder.add_body(xform=wp.transform(wp.vec3(0.5 * i, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
            boxes.append(box)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="reduced",
            substeps=4,
            solver_iterations=4,
            sleeping_velocity_threshold=0.05,
            sleeping_frames_required=10,
        )
        self.assertIsNotNone(solver._reduced_articulation.fused_substep)
        state, output = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        contacts = pipeline.contacts()
        with wp.ScopedCapture() as capture:
            pipeline.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 60.0)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                wp.copy(getattr(state, name), getattr(output, name))
        for _ in range(60):
            wp.capture_launch(capture.graph)
        asleep = solver.world.bodies.island_root.numpy()[1:] >= 0
        self.assertTrue(np.all(asleep[boxes]))
        rest = state.body_q.numpy()[boxes, 0].copy()

        # Send the chain into the first box, which it pushes into the second.
        joint_qd = state.joint_qd.numpy()
        joint_qd[0] = 3.0
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for _ in range(120):
            wp.capture_launch(capture.graph)
        moved = state.body_q.numpy()[boxes, 0] - rest
        self.assertGreater(float(moved[0]), 0.2)
        self.assertGreater(float(moved[1]), 0.02)
        self.assertAlmostEqual(float(moved[2]), 0.0, delta=1.0e-4)
        # At rest again, they sleep again.
        self.assertTrue(np.all((solver.world.bodies.island_root.numpy()[1:] >= 0)[boxes]))

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

    def test_heterogeneous_fleet_mixes_sizes_and_articulated_body_fallback_under_graph_capture(self):
        """Trees of four DOF sizes and a 66-DOF chain share one model; the chain keeps the ABA path."""
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
                # All dense trees step in one launch, in the workspace of the largest.
                self.assertEqual([bucket.max_dofs for bucket in dense.buckets], [64])
                self.assertEqual(int(dense.buckets[0].articulations.shape[0]), int(model.articulation_count) - 1)
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

            def add_floating_tree():
                base = builder.add_link(mass=0.2)
                builder.add_shape_sphere(base, radius=0.03)
                tip = builder.add_link(mass=0.2)
                builder.add_shape_sphere(tip, radius=0.03)
                builder.add_articulation(
                    [
                        builder.add_joint_free(base),
                        builder.add_joint_revolute(parent=base, child=tip, axis=newton.Axis.Y),
                    ]
                )

            add_floating_tree()
            if second_tree:
                add_floating_tree()
            else:
                builder.add_shape_sphere(builder.add_body(mass=0.2), radius=0.03)
            return builder.finalize(device=device)

        # A free body that may touch one tree joins that tree's fused substep.
        solver = newton.solvers.SolverPhoenX(make(False), joint_mode="reduced", reduced_backend="auto")
        self.assertEqual(solver.reduced_backend, "dense")
        self.assertTrue(solver._reduced_articulation.external_rigid_rows)

        # A second floating tree cannot share a block. Both keep their fused substep, and the
        # contacts between them are solved between its kernels.
        solver = newton.solvers.SolverPhoenX(make(True), joint_mode="reduced", reduced_backend="auto")
        self.assertEqual(solver.reduced_backend, "dense")
        self.assertIsNotNone(solver._reduced_articulation.fused_substep)
        self.assertTrue(solver._reduced_articulation.contact_block_system.requires_impulse_response)

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
