# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check sleeping through the public PhoenX solver adapter."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverPhoenX, phoenx


@wp.kernel
def _attach_box(dynamics: phoenx.DynamicsData, ids: wp.array[int], height: float, cancel: bool):
    pose = wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity())
    desc = phoenx.joint_desc(-1, 0, pose, wp.transform_identity())
    ids[0] = phoenx.add_joint(dynamics, desc)
    if cancel:
        phoenx.remove_joint(dynamics, ids[0])


@wp.kernel
def _detach_box(dynamics: phoenx.DynamicsData, ids: wp.array[int]):
    phoenx.remove_joint(dynamics, ids[0])


class TestSleepingRuntimeJoint(unittest.TestCase):
    def test_runtime_joint_changes_cpu(self):
        self._check_runtime_joint_changes("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_runtime_joint_changes_cuda(self):
        self._check_runtime_joint_changes("cuda:0")

    def _check_runtime_joint_changes(self, device):
        """Wake a sleeping box on attachment and release, preserving unrelated sleepers and canceled edits."""
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            for x in (0.0, 2.0):
                box = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
                builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
            builder.add_ground_plane()
            builder.reserve_dynamic_capacity(joints=2)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                direct_joint_mask=np.zeros(model.joint_count, dtype=bool),
                substeps=3,
                solver_iterations=4,
                temporal_iterations=True,
                sleeping_motion_threshold=0.0003,
                sleeping_velocity_threshold=0.02,
                sleeping_frames_required=2,
            )
            state, control, contacts = model.state(), model.control(), pipeline.contacts()
            ids = wp.zeros(1, dtype=int)

            def simulate():
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)

            graph = None
            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as capture:
                    simulate()
                graph = capture.graph

            def advance(frames=1):
                for _ in range(frames):
                    if graph is None:
                        simulate()
                    else:
                        wp.capture_launch(graph)

            def attach(height, *, cancel=False):
                wp.launch(_attach_box, dim=1, inputs=[solver.dynamics.data, ids, height, cancel])

            def detach():
                wp.launch(_detach_box, dim=1, inputs=[solver.dynamics.data, ids])

            advance(60)
            roots = solver.bodies.island_root
            self.assertTrue(np.all(roots.numpy()[1:] >= 0))
            attach(0.4, cancel=True)
            advance()
            self.assertTrue(np.all(roots.numpy()[1:] >= 0))
            canceled = int(ids.numpy()[0])
            self.assertGreaterEqual(canceled, 0)

            # The slot is available again after its canceled edit is committed.
            advance()
            attach(0.4)
            advance()
            self.assertEqual(int(ids.numpy()[0]), canceled)
            self.assertLess(int(roots.numpy()[1]), 0)
            self.assertGreaterEqual(int(roots.numpy()[2]), 0)
            self.assertGreater(float(state.body_q.numpy()[0, 2]), 0.11)
            advance(120)
            self.assertTrue(np.all(roots.numpy()[1:] >= 0))
            height = float(state.body_q.numpy()[0, 2])
            self.assertAlmostEqual(height, 0.4, delta=0.005)

            # An edit after collision must still wake the island during the step.
            solver.collide(state, contacts, dt=1 / 120)
            detach()
            solver.step(state, state, control, contacts, 1 / 120)
            self.assertLess(int(roots.numpy()[1]), 0)
            self.assertLess(float(state.body_qd.numpy()[0, 2]), -0.01)
            self.assertGreaterEqual(int(roots.numpy()[2]), 0)
            # Releasing a world joint also restores contacts excluded by the joint.
            advance(120)
            attach(0.1)
            advance(60)
            self.assertTrue(np.all(roots.numpy()[1:] >= 0))
            detach()
            solver.collide(state, contacts, dt=1 / 120)
            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            self.assertGreaterEqual(int(roots.numpy()[2]), 0)
            # Replace a later slot with an earlier one in the same commit.
            advance(2)
            attach(0.1, cancel=True)
            attach(0.1)
            later_slot = int(ids.numpy()[0])
            advance(2)
            detach()
            attach(0.1)
            advance()
            self.assertLess(int(ids.numpy()[0]), later_slot)
            self.assertLess(int(roots.numpy()[1]), 0)
            solver.collide(state, contacts, dt=1 / 120)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
            self.assertEqual(int(solver.dynamics.error_count.numpy()[0]), 0)


class TestSleepingKinematicContact(unittest.TestCase):
    def test_velocity_edit_wakes_before_collision(self):
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            builder.add_ground_plane()
            model = builder.finalize()
            solver = SolverPhoenX(model, sleeping_velocity_threshold=0.02, sleeping_frames_required=3)
            state, contacts, control = model.state(), model.contacts(), model.control()
            for _ in range(60):
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            self.assertGreaterEqual(solver.bodies.island_root.numpy()[1], 0)
            solver.collide(state, contacts, dt=1 / 120)
            self.assertEqual(contacts.rigid_contact_count.numpy()[0], 0)
            velocity = state.body_qd.numpy().copy()
            velocity[body, 0] = 0.5
            state.body_qd.assign(velocity)
            solver.wake_on_external_input(state)
            self.assertLess(solver.bodies.island_root.numpy()[1], 0)
            solver.collide(state, contacts, dt=1 / 120)
            self.assertGreater(contacts.rigid_contact_count.numpy()[0], 0)

    def test_sleeping_keeps_preconfigured_pipeline(self):
        with wp.ScopedDevice("cpu"):
            for broad_phase in ("explicit", "nxn", "sap"):
                with self.subTest(broad_phase=broad_phase):
                    builder = newton.ModelBuilder()
                    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.095)))
                    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                    builder.add_ground_plane()
                    model = builder.finalize()
                    pipeline = newton.CollisionPipeline(
                        model,
                        broad_phase=broad_phase,
                        rigid_contact_max=4096,
                        publish_capacity=False,
                        contact_matching="sticky",
                        speculative_contact_gap_max=0.005,
                    )
                    contacts = pipeline.contacts()
                    solver = SolverPhoenX(model, collision_pipeline=pipeline, sleeping_velocity_threshold=0.01)
                    self.assertIs(model._collision_pipeline, pipeline)
                    self.assertEqual(solver.world.rigid_contact_max, contacts.rigid_contact_max)
                    state = model.state()
                    pipeline.collide(state, contacts, dt=1 / 60)
                    self.assertGreater(contacts.rigid_contact_count.numpy()[0], 0)
                    solver.step(state, state, None, contacts, 1 / 60)
                    self.assertTrue(np.isfinite(state.body_q.numpy()).all())
                    solver.bodies.island_root.fill_(1)
                    pipeline.collide(state, contacts, dt=1 / 60)
                    self.assertEqual(contacts.rigid_contact_count.numpy()[0], 0)

    def test_read_only_support_does_not_serialize_contacts(self):
        """Keep independent awake contacts parallel when they share a kinematic support."""
        results = []
        for sleeping in (False, True):
            with wp.ScopedDevice("cpu"):
                builder = newton.ModelBuilder()
                support = builder.add_body(is_kinematic=True)
                builder.add_shape_box(support, hx=3, hy=1, hz=0.1)
                for x in (-1, 0, 1):
                    body = builder.add_body(xform=wp.transform(wp.vec3(x, 0, 0.3)))
                    builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
                model = builder.finalize()
                state = model.state()
                velocity = state.body_qd.numpy()
                velocity[1:, 0] = 0.1
                state.body_qd.assign(velocity)
                solver = SolverPhoenX(
                    model,
                    step_layout="single_world",
                    substeps=1,
                    solver_iterations=4,
                    contact_hertz=80,
                    sleeping_velocity_threshold=0.02 if sleeping else 0.0,
                )
                contacts = model._collision_pipeline.contacts()
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, model.control(), contacts, 1 / 120)
                world = solver.world
                count = int(world._num_active_constraints.numpy()[0])
                elements = world._elements.numpy()["bodies"][:count]
                self.assertTrue(np.all(world.bodies.island_root.numpy()[1:] < 0))
                results.append((int(world._partitioner.num_colors.numpy()[0]), elements, state.body_q.numpy()))
        self.assertEqual(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])
        np.testing.assert_array_equal(results[0][2], results[1][2])

    def test_shared_stationary_support_allows_independent_sleep_cpu(self):
        self._check_shared_stationary_support("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_shared_stationary_support_allows_independent_sleep_cuda(self):
        self._check_shared_stationary_support("cuda:0")

    def _check_shared_stationary_support(self, device):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            frictionless = builder.ShapeConfig(mu=0, mu_static=0)
            support = builder.add_body(is_kinematic=True)
            builder.add_shape_box(support, hx=4, hy=1, hz=0.1, cfg=frictionless)
            boxes = []
            for x in (-1, 1):
                box = builder.add_body(xform=wp.transform(wp.vec3(x, 0, 0.2)))
                builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=frictionless)
                boxes.append(box)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                substeps=1,
                solver_iterations=4,
                contact_hertz=80,
                sleeping_velocity_threshold=0.02,
                sleeping_frames_required=20,
            )
            state, contacts, control = model.state(), pipeline.contacts(), model.control()
            velocity = state.body_qd.numpy().copy()
            velocity[boxes[1], 0] = 0.12
            state.body_qd.assign(velocity)
            dt = 1 / 120
            for _ in range(80):
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=dt)
                solver.step(state, state, control, contacts, dt)
            slots = np.asarray(boxes) + 1
            np.testing.assert_array_equal(solver.bodies.island_root.numpy()[slots] >= 0, [True, False])
            self.assertAlmostEqual(state.body_qd.numpy()[boxes[1], 0], 0.12, places=4)

            poses = state.body_q.numpy().copy()
            poses[support, 2] = -2
            state.body_q.assign(poses)
            solver.wake_on_external_input(state)
            self.assertTrue(np.all(solver.bodies.island_root.numpy()[slots] < 0))
            solver.collide(state, contacts, dt=dt)
            solver.step(state, state, control, contacts, dt)
            np.testing.assert_allclose(state.body_qd.numpy()[boxes, 2], -9.81 * dt, atol=0.003)

    def test_moving_support_wakes_stack(self):
        """Wake the whole sleeping stack when its kinematic support rises."""
        self._check_moving_support(False)

    def test_temporal_moving_support_wakes_stack(self):
        """Preserve support-motion and removal waking with temporal contact sweeps."""
        self._check_moving_support(True)

    def _check_moving_support(self, temporal):
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            support = builder.add_body(is_kinematic=True)
            builder.add_shape_box(support, hx=1.0, hy=1.0, hz=0.1)
            cargo = []
            for z in (0.3, 0.7):
                body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, z)))
                builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
                cargo.append(body)
            model = builder.finalize()
            solver = SolverPhoenX(
                model,
                step_layout="single_world",
                substeps=1 if temporal else 4,
                solver_iterations=24 if temporal else 4,
                temporal_iterations=temporal,
                contact_hertz=80,
                sleeping_velocity_threshold=0.02,
                sleeping_frames_required=30,
            )
            state, control = model.state(), model.control()
            contacts = model._collision_pipeline.contacts()
            slots = np.asarray(cargo) + 1
            for _ in range(120):
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            self.assertTrue(np.all(solver.world.bodies.island_root.numpy()[slots] >= 0))
            start = state.body_q.numpy()[cargo, 2].copy()
            for step in range(1, 61):
                poses = state.body_q.numpy()
                poses[support, 2] = step * 0.002
                state.body_q.assign(poses)
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
                self.assertTrue(
                    np.all(solver.world.bodies.island_root.numpy()[slots] < 0),
                    msg=f"moving support failed to wake the stack on step {step}",
                )
            np.testing.assert_allclose(state.body_q.numpy()[cargo, 2] - start, 0.12, atol=0.01)
            for _ in range(120):
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            self.assertTrue(np.all(solver.world.bodies.island_root.numpy()[slots] >= 0))
            start = state.body_q.numpy()[cargo, 2].copy()
            initial_velocity = state.body_qd.numpy()[cargo, 2].copy()
            poses = state.body_q.numpy()
            poses[support, 3:] *= -1.0
            state.body_q.assign(poses)
            solver.wake_on_external_input(state)
            self.assertTrue(np.all(solver.world.bodies.island_root.numpy()[slots] >= 0))
            poses[support, 2] = -1.0
            state.body_q.assign(poses)
            solver.wake_on_external_input(state)
            self.assertTrue(
                np.all(solver.world.bodies.island_root.numpy()[slots] < 0),
                msg="removing the support must wake its stack before collision",
            )
            for _ in range(24):
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            duration = 24 / 120
            expected_fall = initial_velocity * duration - 0.5 * 9.81 * duration**2
            np.testing.assert_allclose(state.body_q.numpy()[cargo, 2] - start, expected_fall, atol=0.01)


class TestTemporalSleeping(unittest.TestCase):
    def test_passive_support_and_active_servo_cpu(self):
        """Sleep a supported passive mechanism while other bodies remain active."""
        self._check_passive_support("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_passive_support_and_active_servo_cuda(self):
        """Preserve support forces, sleep, and external-force waking during graph replay."""
        self._check_passive_support("cuda:0")

    def _check_passive_support(self, device):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            passive = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 0.1)))
            builder.add_shape_box(passive, hx=0.1, hy=0.1, hz=0.1)
            joint = builder.add_joint_prismatic(
                -1,
                passive,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0, 0, 0.1)),
                target_ke=0,
                target_kd=0,
                limit_lower=-1e6,
                limit_upper=1e6,
            )
            builder.add_articulation([joint])
            servo = builder.add_link(xform=wp.transform(wp.vec3(3, 0, 0.1)))
            builder.add_shape_box(servo, hx=0.1, hy=0.1, hz=0.1)
            joint = builder.add_joint_prismatic(
                -1,
                servo,
                axis=newton.Axis.X,
                parent_xform=wp.transform(wp.vec3(3, 0, 0.1)),
                target_ke=1000,
                target_kd=100,
            )
            builder.add_articulation([joint])
            falling = builder.add_body(xform=wp.transform(wp.vec3(5, 0, 100)))
            builder.add_shape_sphere(falling, radius=0.1)
            builder.add_ground_plane()
            model = builder.finalize()
            solver = SolverPhoenX(
                model,
                step_layout="single_world",
                joint_mode="maximal_direct",
                substeps=1,
                solver_iterations=24,
                temporal_iterations=True,
                contact_hertz=80,
                sleeping_velocity_threshold=0.02,
                sleeping_frames_required=60,
            )
            state, control = model.state(), model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            contacts = model._collision_pipeline.contacts()
            observables = solver.observables({solver.ObservableKind.CONTACT_F}, contacts=contacts)
            dt = 1 / 120
            mass = float(model.body_mass.numpy()[passive])
            shape_body = model.shape_body.numpy()

            def step():
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=dt)
                solver.step(state, state, control, contacts, dt, observables=observables)

            step()
            graph = None
            if model.device.is_cuda:
                with wp.ScopedCapture() as capture:
                    step()
                graph = capture.graph

            def advance():
                if graph is None:
                    step()
                else:
                    wp.capture_launch(graph)

            def support_force():
                count = int(contacts.rigid_contact_count.numpy()[0])
                a = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
                b = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
                forces = observables.contact_f.numpy()[:count, 2]
                return float(forces[a == passive].sum() - forces[b == passive].sum())

            sleeping_pose = None
            supports = []
            for frame in range(100):
                advance()
                roots = solver.bodies.island_root.numpy()
                # The idle servo may sleep; its new target below must wake it.
                self.assertLess(roots[falling + 1], 0)
                if 30 <= frame < 46:
                    supports.append(support_force())
                if frame == 75:
                    self.assertGreaterEqual(roots[passive + 1], 0)
                    sleeping_pose = state.body_q.numpy()[passive].copy()
                if frame > 75:
                    np.testing.assert_array_equal(state.body_q.numpy()[passive], sleeping_pose)
            # The jointed body receives the whole substep's gravity at its start while its
            # owned floor contact answers in every temporal sweep: the support alternates
            # around the weight from step to step (a 27 um bounce) and balances it on average.
            self.assertAlmostEqual(float(np.mean(supports)), mass * 9.81, delta=0.01)
            self.assertAlmostEqual(float(sleeping_pose[2]), 0.1, delta=0.001)
            targets = control.joint_target_q.numpy()
            targets[1] = 0.3
            control.joint_target_q.assign(targets)
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[passive, 2] = 2 * mass * 9.81
            state.body_f.assign(force)
            previous_z = float(state.body_q.numpy()[passive, 2])
            for _ in range(60):
                advance()
                self.assertLess(solver.bodies.island_root.numpy()[passive + 1], 0)
            self.assertGreater(float(state.body_q.numpy()[passive, 2]), previous_z + 1.0)
            self.assertAlmostEqual(support_force(), 0.0, delta=0.01)
            self.assertAlmostEqual(float(state.body_q.numpy()[servo, 0]), 3.3, delta=0.02)


if __name__ == "__main__":
    unittest.main()
