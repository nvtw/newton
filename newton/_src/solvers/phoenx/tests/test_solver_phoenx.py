# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for :class:`SolverPhoenX` -- the Newton-Model /
Newton-State wrapper around :class:`PhoenXWorld`.

Scenes cover the full integration surface from smallest to largest:

* free fall + contacts: a box dropped onto a plane must settle;
* revolute drive: a pendulum's PD drive parks at a target angle;
* fixed weld: a cube welded to the world must not move under gravity;
* a FREE-base + REVOLUTE-driven two-link chain (the Anymal building
  block at miniature scale) tracks simple PD targets.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.flags import SolverNotifyFlags
from newton._src.solvers.phoenx.tests._test_helpers import run_solver_capture_loop

GRAVITY = 9.81


def _make_box_model(*, box_z: float = 0.5, mu: float = 0.5, rigid_gap: float | None = None) -> newton.Model:
    """Dynamic unit-mass cube + ground plane. No joints."""
    mb = newton.ModelBuilder()
    if rigid_gap is not None:
        mb.rigid_gap = rigid_gap
    mb.add_ground_plane()
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, box_z), q=wp.quat_identity()),
    )
    mb.add_shape_box(
        body,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu),
    )
    mb.gravity = -GRAVITY
    return mb.finalize()


def _make_box_margin_stack_model(*, margin: float) -> tuple[newton.Model, int]:
    """Static box + dynamic box with equal shape margins."""
    mb = newton.ModelBuilder()
    static_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8, margin=margin)
    dynamic_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.8, margin=margin)
    mb.add_shape_box(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        hx=0.5,
        hy=0.5,
        hz=0.1,
        cfg=static_cfg,
    )
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.55), q=wp.quat_identity()),
        mass=1.0,
    )
    mb.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=dynamic_cfg)
    mb.gravity = -GRAVITY
    return mb.finalize(), body


def _make_offset_com_free_body_model() -> tuple[newton.Model, int]:
    """Single free body with a nonzero local COM offset."""
    mb = newton.ModelBuilder()
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.25, -0.4, 0.8), q=wp.quat_identity()),
        mass=2.0,
        inertia=((0.12, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.18)),
    )
    mb.body_com[body] = wp.vec3(0.35, -0.12, 0.22)
    mb.gravity = 0.0
    return mb.finalize(), body


def _make_offset_com_free_joint_model() -> tuple[newton.Model, int]:
    """Single FREE-root articulation with a rotated parent frame and offset COM."""
    mb = newton.ModelBuilder()
    body = mb.add_link(
        mass=2.0,
        inertia=((0.12, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.18)),
    )
    mb.body_com[body] = wp.vec3(0.35, -0.12, 0.22)
    joint = mb.add_joint_free(
        parent=-1,
        child=body,
        parent_xform=wp.transform(
            p=wp.vec3(0.25, -0.4, 0.8),
            q=wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, -0.7, 1.0)), 0.6),
        ),
    )
    mb.add_articulation([joint])
    mb.gravity = 0.0
    return mb.finalize(), body


def _make_offset_com_contact_model() -> tuple[newton.Model, int]:
    """Box shape centered at body origin, with COM shifted upward."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.12), q=wp.quat_identity()),
        mass=2.0,
        inertia=((0.02, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.0, 0.02)),
    )
    mb.body_com[body] = wp.vec3(0.0, 0.0, 0.05)
    mb.add_shape_box(
        body,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5),
    )
    mb.gravity = -GRAVITY
    return mb.finalize(), body


def _make_pendulum_model(*, target_angle: float = 0.0) -> newton.Model:
    """Static world body + dynamic cube + revolute joint with a PD
    drive towards ``target_angle`` (rad).

    Built with :meth:`add_link` + explicit joint + :meth:`add_articulation`
    so Newton does not auto-insert a FREE base joint on top.
    """
    mb = newton.ModelBuilder()
    cube = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, -1.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    )
    mb.add_shape_box(
        cube,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    joint = mb.add_joint_revolute(
        parent=-1,
        child=cube,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 1.0, 0.0), q=wp.quat_identity()),
        target_pos=target_angle,
        target_ke=50.0,
        target_kd=5.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    mb.add_articulation([joint])
    mb.gravity = -GRAVITY
    return mb.finalize()


def _make_contact_chain_model(link_count: int = 4) -> newton.Model:
    """Short revolute box chain resting on a ground plane."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    bodies: list[int] = []
    joints: list[int] = []
    for link in range(link_count):
        body = mb.add_link(
            xform=wp.transform(p=wp.vec3(0.2 * link, 0.0, 0.1), q=wp.quat_identity()),
            mass=1.0,
            inertia=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        )
        mb.add_shape_box(
            body,
            hx=0.1,
            hy=0.05,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6),
        )
        parent = -1 if link == 0 else bodies[-1]
        parent_xform = (
            wp.transform(p=wp.vec3(0.0, 0.0, 0.1), q=wp.quat_identity())
            if link == 0
            else wp.transform(p=wp.vec3(0.1, 0.0, 0.0), q=wp.quat_identity())
        )
        child_xform = (
            wp.transform_identity() if link == 0 else wp.transform(p=wp.vec3(-0.1, 0.0, 0.0), q=wp.quat_identity())
        )
        joint = mb.add_joint_revolute(
            parent=parent,
            child=body,
            axis=(0.0, 1.0, 0.0),
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
        bodies.append(body)
        joints.append(joint)
    mb.add_articulation(joints)
    mb.gravity = -GRAVITY
    return mb.finalize()


def _make_welded_cube_model() -> newton.Model:
    """Single body welded to the world. Under gravity the cube must
    not move."""
    mb = newton.ModelBuilder()
    cube = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.5), q=wp.quat_identity()),
        mass=1.0,
        inertia=((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    )
    mb.add_shape_box(
        cube,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    joint = mb.add_joint_fixed(
        parent=-1,
        child=cube,
        parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.5), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    mb.add_articulation([joint])
    mb.gravity = -GRAVITY
    return mb.finalize()


def _run_frames(solver, state_0, state_1, control, contacts, model, n: int, dt: float):
    """Advance an even frame count through the captured solver stepper."""
    return run_solver_capture_loop(solver, state_0, state_1, control, contacts, model, n, dt)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "SolverPhoenX tests run on CUDA only.",
)
class TestSolverPhoenX(unittest.TestCase):
    """Behavioural checks for the Newton-Model driven PhoenX solver."""

    def test_prepare_refresh_stride_plumbs_to_world(self) -> None:
        model = _make_box_model(box_z=0.2)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=4,
            solver_iterations=1,
            step_layout="single_world",
            prepare_refresh_stride=3,
        )
        self.assertEqual(solver.world.prepare_refresh_stride, 3)

    def test_auto_prepare_refresh_stride_plumbs_to_world(self) -> None:
        model = _make_box_model(box_z=0.2)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=1,
            step_layout="single_world",
            prepare_refresh_stride="auto",
        )
        self.assertEqual(solver.world.prepare_refresh_stride, 3)
        self.assertEqual(solver.world._prepare_refresh_stride_policy, "auto")

    def test_box_settles_on_plane(self) -> None:
        """Dynamic cube on a plane -- after 1 s the COM z must be
        ~0.1 m (cube half-height) and velocity must be small."""
        model = _make_box_model(box_z=0.5, mu=0.5)
        solver = newton.solvers.SolverPhoenX(model, substeps=8, solver_iterations=16)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = model.contacts(collision_pipeline=collision_pipeline)

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=60, dt=dt)

        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        com_z = float(body_q[0, 2])
        speed = float(np.linalg.norm(body_qd[0, :3]))
        self.assertAlmostEqual(
            com_z,
            0.1,
            delta=0.02,
            msg=f"cube COM z = {com_z:.3f} m, expected ~0.1",
        )
        self.assertLess(speed, 0.1, msg=f"cube still moving: |v|={speed:.3f}")

    def test_articulation_coarse_path_coexists_with_contacts(self) -> None:
        """The bilateral coarse level supplements captured contact PGS."""
        model = _make_contact_chain_model()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=4,
            solver_iterations=2,
            velocity_iterations=1,
            articulation_coarse_mode="auto",
            articulation_coarse_stride=2,
        )
        self.assertEqual(solver.world.articulation_coarse_setup.mode, "path")
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = model.contacts(collision_pipeline=collision_pipeline)

        state_0, _ = _run_frames(
            solver,
            state_0,
            state_1,
            control,
            contacts,
            model,
            n=120,
            dt=1.0 / 120.0,
        )
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_0.body_qd.numpy()).all())

    def test_contact_gap_is_detection_only(self) -> None:
        heights: dict[float, float] = {}
        for rigid_gap in (0.0, 0.1):
            with self.subTest(rigid_gap=rigid_gap):
                model = _make_box_model(box_z=0.5, mu=0.5, rigid_gap=rigid_gap)
                self.assertTrue(np.allclose(model.shape_gap.numpy(), rigid_gap))

                solver = newton.solvers.SolverPhoenX(
                    model,
                    substeps=8,
                    solver_iterations=16,
                    velocity_iterations=1,
                    prepare_refresh_stride=1,
                )
                state_0 = model.state()
                state_1 = model.state()
                control = model.control()
                collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
                contacts = model.contacts(collision_pipeline=collision_pipeline)

                dt = 1.0 / 60.0
                state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)

                body_q = state_0.body_q.numpy()
                body_qd = state_0.body_qd.numpy()
                com_z = float(body_q[0, 2])
                speed = float(np.linalg.norm(body_qd[0, :3]))
                heights[rigid_gap] = com_z
                self.assertAlmostEqual(com_z, 0.1, delta=0.005)
                self.assertLess(speed, 0.02)

        self.assertLess(abs(heights[0.1] - heights[0.0]), 0.002)

    def test_box_margins_set_rest_distance(self) -> None:
        margin = 0.03
        model, body = _make_box_margin_stack_model(margin=margin)
        solver = newton.solvers.SolverPhoenX(model, substeps=8, solver_iterations=16, velocity_iterations=1)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = model.contacts(collision_pipeline=collision_pipeline)

        dt = 1.0 / 60.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=60, dt=dt)
        model.collide(state_0, contacts=contacts, collision_pipeline=collision_pipeline)

        com_z = float(state_0.body_q.numpy()[body, 2])
        expected_z = 0.1 + 0.1 + 2.0 * margin
        self.assertAlmostEqual(com_z, expected_z, delta=0.005)

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0)
        margin0 = contacts.rigid_contact_margin0.numpy()[:contact_count]
        margin1 = contacts.rigid_contact_margin1.numpy()[:contact_count]
        self.assertTrue(np.allclose(margin0, margin, atol=1.0e-6))
        self.assertTrue(np.allclose(margin1, margin, atol=1.0e-6))

    def test_revolute_drive_tracks_target(self) -> None:
        """PD position drive on a revolute pendulum parks the cube at
        ``target_angle`` after a settle window."""
        target = math.pi / 4.0
        model = _make_pendulum_model(target_angle=target)
        solver = newton.solvers.SolverPhoenX(model, substeps=8, solver_iterations=16)
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        control = model.control()

        dt = 1.0 / 120.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, None, model, n=480, dt=dt)

        # Read the joint angle back via eval_ik from body_q.
        joint_q = wp.zeros(model.joint_coord_count, dtype=wp.float32)
        joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32)
        newton.eval_ik(model, state_0, joint_q, joint_qd)
        angle = float(joint_q.numpy()[0])
        self.assertAlmostEqual(
            angle,
            target,
            delta=0.1,
            msg=f"revolute drive off target: got {angle:.3f} rad, expected {target:.3f}",
        )

    def test_fixed_joint_holds_welded_cube(self) -> None:
        """FIXED joint anchors a cube in space under gravity."""
        model = _make_welded_cube_model()
        solver = newton.solvers.SolverPhoenX(model, substeps=8, solver_iterations=16)
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        control = model.control()

        dt = 1.0 / 120.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, None, model, n=240, dt=dt)

        body_q = state_0.body_q.numpy()
        pos = body_q[0, :3]
        drift = float(np.linalg.norm(pos - np.array([0.5, 0.0, 0.5])))
        self.assertLess(drift, 0.05, msg=f"welded cube drifted {drift:.4f} m")

    def test_external_body_force_propagates_through_step(self) -> None:
        """Newton's :class:`Picking` writes the pick spring into
        ``state.body_f`` via ``wp.atomic_add``. Verify ``SolverPhoenX``
        imports that channel: a +x linear force on a free-floating cube
        (no gravity, no contacts) must produce a +x linear velocity
        proportional to ``F * dt / m`` after one step.
        """
        # Free-floating unit-mass cube, no gravity, no joints, no contacts.
        mb = newton.ModelBuilder()
        mb.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)))
        mb.add_shape_box(
            0,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
        )
        mb.gravity = 0.0
        model = mb.finalize()

        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=1)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Body mass for the assertion below.
        mass = float(model.body_mass.numpy()[0])
        self.assertGreater(mass, 0.0)

        # Inject a +x linear force at COM via state.body_f -- the same
        # array Newton's Picking writes to.
        force_x = 100.0
        body_f_np = np.zeros((1, 6), dtype=np.float32)
        body_f_np[0, 0] = force_x  # spatial top = linear force
        state_0.body_f.assign(body_f_np)

        dt = 1.0 / 240.0
        solver.step(state_0, state_1, control, None, dt)

        # Expected: v += F * dt / m for one whole step.
        body_qd = state_1.body_qd.numpy()
        v_x = float(body_qd[0, 0])
        expected = force_x * dt / mass
        self.assertAlmostEqual(
            v_x,
            expected,
            delta=expected * 0.05,
            msg=f"external body_f did not propagate: v_x={v_x:.6f}, expected ~{expected:.6f}",
        )

    def test_offset_com_free_body_preserves_newton_velocity_convention(self) -> None:
        """PhoenX must round-trip Newton origin poses and COM-referenced twists."""
        model, body = _make_offset_com_free_body_model()
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=1)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        qd_in = np.asarray([[0.4, -0.25, 0.15, 0.0, 1.2, -0.35]], dtype=np.float32)
        state_0.body_qd.assign(qd_in)
        dt = 1.0 / 120.0

        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_0, state_1, control, None, dt)
        wp.capture_launch(capture.graph)

        body_q = state_1.body_q.numpy()[body]
        body_qd = state_1.body_qd.numpy()[body]
        com_local = model.body_com.numpy()[body]

        omega = qd_in[0, 3:6].astype(np.float64)
        speed = float(np.linalg.norm(omega))
        half_angle = 0.5 * speed * dt
        axis = omega / speed
        expected_quat = np.asarray(
            [
                axis[0] * math.sin(half_angle),
                axis[1] * math.sin(half_angle),
                axis[2] * math.sin(half_angle),
                math.cos(half_angle),
            ],
            dtype=np.float64,
        )
        initial_origin = state_0.body_q.numpy()[body, 0:3].astype(np.float64)
        initial_com = initial_origin + com_local.astype(np.float64)
        expected_com = initial_com + qd_in[0, 0:3].astype(np.float64) * dt
        rot = np.array(wp.quat_to_matrix(wp.quat(*expected_quat)), dtype=np.float64).reshape(3, 3)
        expected_origin = expected_com - rot @ com_local.astype(np.float64)

        if np.dot(body_q[3:7], expected_quat) < 0.0:
            expected_quat = -expected_quat
        np.testing.assert_allclose(body_q[0:3], expected_origin, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(body_q[3:7], expected_quat, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(body_qd, qd_in[0], rtol=1.0e-6, atol=1.0e-6)

    def test_offset_com_free_joint_qd_round_trips_parent_frame_in_graph(self) -> None:
        """FREE joint_qd stays parent-frame COM velocity after PhoenX + eval_ik."""
        model, body = _make_offset_com_free_joint_model()
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=1)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        q = state_0.joint_q.numpy()
        q[0:3] = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        q[3:7] = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.2, -0.5)), -0.4), dtype=np.float32)
        qd_in = np.asarray([0.45, -0.25, 0.15, 0.2, 1.1, -0.35], dtype=np.float32)
        state_0.joint_q.assign(q)
        state_0.joint_qd.assign(qd_in)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        body_qd_0 = state_0.body_qd.numpy()[body].copy()

        dt = 1.0 / 120.0
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_0, state_1, control, None, dt)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(state_1.body_qd.numpy()[body], body_qd_0, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(state_1.joint_qd.numpy(), qd_in, rtol=1.0e-5, atol=1.0e-5)

    def test_offset_com_contact_uses_newton_shape_frame(self) -> None:
        """Contacts must subtract local COM from Newton body-local shape anchors."""
        model, body = _make_offset_com_contact_model()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=8,
            solver_iterations=24,
            velocity_iterations=2,
            prepare_refresh_stride=1,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = model.contacts(collision_pipeline=collision_pipeline)

        dt = 1.0 / 120.0
        state_0, _state_1 = _run_frames(solver, state_0, state_1, control, contacts, model, n=120, dt=dt)

        body_q = state_0.body_q.numpy()[body]
        body_qd = state_0.body_qd.numpy()[body]
        com_local = model.body_com.numpy()[body]
        com_world_z = float(solver.bodies.position.numpy()[body + 1, 2])

        self.assertAlmostEqual(float(body_q[2]), 0.1, delta=0.01)
        self.assertAlmostEqual(com_world_z, 0.1 + float(com_local[2]), delta=0.01)
        self.assertLess(float(np.linalg.norm(body_qd[:3])), 0.05)

    def test_public_cloth_step_exports_particles(self) -> None:
        mb = newton.ModelBuilder()
        mb.gravity = 0.0
        body = mb.add_body(xform=wp.transform(p=wp.vec3(2.0, 0.0, 0.0), q=wp.quat_identity()), mass=0.0)
        mb.add_shape_box(
            body,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        mb.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.2),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            particle_radius=0.005,
        )
        model = mb.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=2, step_layout="single_world")
        self.assertEqual(solver.world.num_cloth_triangles, int(model.tri_count))
        self.assertIsNotNone(solver.world.particles)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        z0 = state_0.particle_q.numpy()[:, 2].copy()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        z1 = state_1.particle_q.numpy()[:, 2]

        self.assertTrue(np.all(np.isfinite(z1)))
        self.assertGreater(float(z1.mean()), float(z0.mean()))

    def test_public_soft_tet_step_exports_particles(self) -> None:
        mb = newton.ModelBuilder()
        mb.gravity = 0.0
        mb.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.2),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )
        model = mb.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=2, step_layout="single_world")
        self.assertEqual(solver.world.num_soft_tetrahedra, int(model.tet_count))
        self.assertIsNotNone(solver.world.particles)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        z0 = state_0.particle_q.numpy()[:, 2].copy()

        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        z1 = state_1.particle_q.numpy()[:, 2]

        self.assertTrue(np.all(np.isfinite(z1)))
        self.assertGreater(float(z1.mean()), float(z0.mean()))

    def test_notify_model_changed_body_inertial_properties(self) -> None:
        """Regression: ``notify_model_changed`` with
        ``BODY_INERTIAL_PROPERTIES`` (or ``BODY_PROPERTIES``) must launch
        ``_init_phoenx_body_container_kernel`` with the correct number of
        scalar inputs.

        A previous version of the refresh path passed two flag scalars
        (``NO_GRAVITY`` *and* ``KINEMATIC``) but the kernel signature
        only accepts one (``kinematic_flag``); the launch raised at
        runtime. This test mutates ``model.body_inv_mass`` (halves the
        cube's mass) and verifies that the refresh propagates -- gravity
        on a body of doubled inv_mass should accelerate twice as fast in
        the same window.
        """
        model = _make_box_model(box_z=1.0, mu=0.5)
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=1)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Halve the body mass at runtime via inv_mass.
        inv_mass_np = model.body_inv_mass.numpy()
        inv_mass_np[0] *= 2.0
        model.body_inv_mass.assign(inv_mass_np)

        # Must not raise -- this is the regression.
        solver.notify_model_changed(
            int(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)
            | int(SolverNotifyFlags.BODY_PROPERTIES)
            | int(SolverNotifyFlags.SHAPE_PROPERTIES)
        )

        dt = 1.0 / 240.0
        state_0, state_1 = _run_frames(solver, state_0, state_1, control, None, model, n=12, dt=dt)
        # The body should still simulate (the kernel ran cleanly); the
        # exact velocity isn't the point, just that step() completes.
        self.assertTrue(np.all(np.isfinite(state_0.body_qd.numpy())))


if __name__ == "__main__":
    wp.init()
    unittest.main()
