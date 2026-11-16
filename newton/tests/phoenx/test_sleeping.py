# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check sleeping through the public PhoenX solver adapter."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverPhoenX


class TestSleepingKinematicContact(unittest.TestCase):
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
            for frame in range(100):
                advance()
                roots = solver.bodies.island_root.numpy()
                self.assertLess(roots[servo + 1], 0)
                self.assertLess(roots[falling + 1], 0)
                if 30 <= frame < 45:
                    self.assertAlmostEqual(support_force(), mass * 9.81, delta=0.01)
                if frame == 75:
                    self.assertGreaterEqual(roots[passive + 1], 0)
                    sleeping_pose = state.body_q.numpy()[passive].copy()
                if frame > 75:
                    np.testing.assert_array_equal(state.body_q.numpy()[passive], sleeping_pose)
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
