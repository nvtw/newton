# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise live local-joint release and reactivation with exact neighbors."""

import unittest

import numpy as np
import warp as wp

import newton


class TestJointActivation(unittest.TestCase):
    def _check_lifecycle(self, device, capture):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            for offset in (0.0, 1.0):
                body = builder.add_link(
                    xform=wp.transform((0, offset, 0), wp.quat_identity()),
                    mass=1.0,
                    inertia=wp.mat33(np.eye(3)),
                )
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=builder.ShapeConfig(density=0))
                joint = builder.add_joint_revolute(
                    -1,
                    body,
                    parent_xform=wp.transform((0, offset, 0), wp.quat_identity()),
                    axis=newton.Axis.Z,
                )
                builder.add_articulation([joint])
            model = builder.finalize()
            solver = newton.solvers.SolverPhoenX(
                model,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                substeps=1,
                solver_iterations=4,
                direct_joint_mask=np.array([False, True]),
                sleeping_velocity_threshold=0.01,
                sleeping_frames_required=4,
            )
            state, control = model.state(), model.control()
            force = np.zeros((2, 6), dtype=np.float32)
            force[:, 0] = 1.0
            state.body_f.assign(force)
            dt = 1 / 240
            solver.step(state, state, control, None, dt)
            graph = None
            if capture:
                with wp.ScopedCapture() as captured:
                    solver.step(state, state, control, None, dt)
                graph = captured.graph

            trace = []

            def snapshot():
                trace.append(
                    (
                        state.body_q.numpy().copy(),
                        state.body_qd.numpy().copy(),
                        solver.bodies.island_root.numpy().copy(),
                        solver.world._joint_pgs_enabled.numpy().copy(),
                    )
                )

            def advance(count):
                for _ in range(count):
                    if graph is None:
                        solver.step(state, state, control, None, dt)
                    else:
                        wp.capture_launch(graph)
                snapshot()

            for _ in range(2):
                model.joint_enabled.assign(np.array([False, True]))
                start_velocity = state.body_qd.numpy()[0, 0]
                start_position = state.body_q.numpy()[0, 0]
                advance(24)
                # A released 1 kg body gains exactly F dt momentum; its exact
                # neighbor stays anchored despite the same external force.
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 0] - start_velocity), 0.1, delta=2e-6)
                self.assertGreater(float(state.body_q.numpy()[0, 0] - start_position), 0.004)
                np.testing.assert_allclose(state.body_q.numpy()[1, :3], [0, 1, 0], atol=1e-6)
                model.joint_enabled.assign(np.array([True, True]))
                advance(72)
                self.assertLess(abs(float(state.body_q.numpy()[0, 0])), 1e-5)
                self.assertLess(abs(float(state.body_qd.numpy()[0, 0])), 1e-6)
            state.clear_forces()
            advance(24)
            self.assertGreaterEqual(int(solver.bodies.island_root.numpy()[1]), 0)
            model.joint_enabled.assign(np.array([False, True]))
            # Activation is processed before collision, so sleeping contacts
            # cannot discard the released body's first awake collision pass.
            contacts = solver._collision_pipeline.contacts()
            solver.wake_on_external_input(state)
            solver._collision_pipeline.collide(state, contacts, dt=dt)
            self.assertEqual(int(solver.bodies.island_root.numpy()[1]), -1)
            self.assertTrue(np.isfinite(state.body_q.numpy()).all())
            snapshot()
            return trace

    def _check_repeatable_lifecycle(self, device, capture):
        first = self._check_lifecycle(device, capture)
        second = self._check_lifecycle(device, capture)
        for phase, (left, right) in enumerate(zip(first, second, strict=True)):
            for actual, expected in zip(left, right, strict=True):
                np.testing.assert_array_equal(actual, expected, err_msg=f"Lifecycle phase {phase} is not repeatable")

    def _check_disabled_drive(self, device, overflow):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            for _ in range(2):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=newton.Axis.X,
                    limit_lower=0,
                    limit_upper=0.01,
                    target_kd=100,
                    effort_limit=0.1,
                    actuator_mode=newton.JointTargetMode.VELOCITY,
                )
                builder.add_articulation([joint])
            model = builder.finalize()
            solver = newton.solvers.SolverPhoenX(
                model,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                substeps=1,
                solver_iterations=4,
                mass_splitting=overflow,
                max_colored_partitions=0 if overflow else None,
            )
            state, control = model.state(), model.control()
            control.joint_target_qd.fill_(100)
            model.joint_enabled.assign(np.array([False, True]))
            force = np.zeros((2, 6), dtype=np.float32)
            force[0, :2] = [4, 1]
            state.body_f.assign(force)
            for _ in range(24):
                solver.step(state, state, control, None, 1 / 240)
            if overflow:
                self.assertGreater(int(solver.world._copy_state.highest_index_in_use.numpy()[0]), 0)
            np.testing.assert_allclose(state.body_qd.numpy()[0, :2], [0.4, 0.1], atol=2e-6)
            self.assertGreater(float(state.body_q.numpy()[0, 0]), 0.02)
            self.assertAlmostEqual(float(state.body_qd.numpy()[1, 0]), 0.01, delta=1e-6)

    def test_disabled_drive_has_no_motor_or_stop_impulse(self):
        """A released slider stays free while a neighboring finite drive saturates."""
        self._check_disabled_drive("cpu", False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_disabled_drive_with_mass_splitting(self):
        """Correct copied-body drive bounds without restoring a disabled joint."""
        self._check_disabled_drive(wp.get_cuda_device(), True)

    def test_local_joint_lifecycle_cpu(self):
        """Repeat CPU release, restoration, and sleeping wakeup with identical state."""
        self._check_repeatable_lifecycle("cpu", False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_local_joint_lifecycle_cuda_graph(self):
        """Repeat captured toggles with identical poses, velocities, sleep, and ownership."""
        self._check_repeatable_lifecycle(wp.get_cuda_device(), True)


if __name__ == "__main__":
    unittest.main()
