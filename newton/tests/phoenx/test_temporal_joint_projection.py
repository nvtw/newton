# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical momentum and repeatability of temporal structural projection."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverPhoenX


class TestTemporalJointProjection(unittest.TestCase):
    def _run_impulse(self, device, overflow):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.02))
            joints = [builder.add_joint_free(child=root)]
            for position in ((0.2, 0, 0), (-0.2, 0, 0), (0, 0.2, 0), (0, -0.2, 0)):
                frame = wp.transform(position, wp.quat_identity())
                body = builder.add_link(xform=frame, mass=0.1, inertia=wp.mat33(np.eye(3) * 0.0005))
                joints.append(builder.add_joint_fixed(root, body, parent_xform=frame))
            builder.add_articulation(joints)
            model = builder.finalize(device=device)
            options = {}
            if overflow:
                options = {
                    "mass_splitting": True,
                    "mass_splitting_overflow_only": True,
                    "mass_splitting_unrolled": True,
                    "mass_splitting_batch_size": 1,
                    "max_colored_partitions": 1,
                    "partitioner_algorithm": "endpoint_owner",
                }
            solver = SolverPhoenX(
                model,
                substeps=1,
                solver_iterations=24,
                temporal_iterations=True,
                contact_hertz=80,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                direct_joint_mask=np.zeros(model.joint_count, dtype=bool),
                sleeping_velocity_threshold=0,
                **options,
            )
            state, control = model.state(), model.control()
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[root, 0] = 1.0
            force[root, 5] = 0.1
            dt = 0.001
            solver.step(state, state, control, None, dt)
            state.body_f.assign(force)
            if model.device.is_cuda:
                with wp.ScopedCapture(device=model.device) as capture:
                    solver.step(state, state, control, None, dt)
                wp.capture_launch(capture.graph)
            else:
                solver.step(state, state, control, None, dt)
            poses, velocities = state.body_q.numpy(), state.body_qd.numpy()
            mass = model.body_mass.numpy()
            momentum = (mass[:, None] * velocities[:, :3]).sum(axis=0)
            angular = np.cross(poses[:, :3], mass[:, None] * velocities[:, :3]).sum(axis=0)
            inertia = model.body_inertia.numpy()
            for body in range(model.body_count):
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(poses[body, 3:]))).reshape(3, 3)
                angular += rotation @ inertia[body] @ rotation.T @ velocities[body, 3:]
            np.testing.assert_allclose(momentum, [dt, 0, 0], atol=2.0e-7, rtol=2.0e-5)
            np.testing.assert_allclose(angular, [0, 0, 0.1 * dt], atol=2.0e-7, rtol=2.0e-5)
            if overflow:
                self.assertGreater(solver.step_report().overflow_size, 0)
            impulse = solver.world.constraints.bilateral.accumulated.numpy().copy()
            return poses.copy(), velocities.copy(), impulse

    def test_projection_preserves_external_impulse_cpu(self):
        """Preserve total momentum under an external force and torque."""
        first = self._run_impulse("cpu", False)
        second = self._run_impulse("cpu", False)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_welded_support_preserves_pose_and_weight(self):
        """Keep two light welded feet rigid while supporting a heavier body."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(density=0, mu=0.4, mu_static=0.45, gap=0.002, restitution=0)
            root = builder.add_link(
                xform=wp.transform((0, 0, 0.251), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.01),
            )
            joints = [builder.add_joint_free(child=root)]
            feet = []
            offsets = []
            for x in (-0.1, 0.1):
                offset = wp.vec3(x, 0, -0.24)
                foot = builder.add_link(
                    xform=wp.transform((x, 0, 0.011), wp.quat_identity()),
                    mass=0.02,
                    inertia=wp.mat33(np.diag([0.00007, 0.00004, 0.0001])),
                )
                builder.add_shape_box(foot, hx=0.08, hy=0.1, hz=0.01, cfg=cfg)
                joints.append(
                    builder.add_joint_fixed(root, foot, parent_xform=wp.transform(offset, wp.quat_identity()))
                )
                feet.append(foot)
                offsets.append(offset)
            builder.add_articulation(joints)
            ground = builder.add_ground_plane(cfg=cfg)
            model = builder.finalize(device="cpu")
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=64)
            solver = SolverPhoenX(
                model,
                substeps=1,
                solver_iterations=24,
                temporal_iterations=True,
                contact_hertz=80,
                velocity_iterations=1,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                sleeping_velocity_threshold=0,
                collision_pipeline=pipeline,
            )
            state, control, contacts = model.state(), model.control(), pipeline.contacts()
            observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            mass = model.body_mass.numpy()
            gravity = np.array([0, 0, -9.81 * mass.sum()])
            forces = []
            for step in range(240):
                before = (mass[:, None] * state.body_qd.numpy()[:, :3]).sum(axis=0).copy()
                state.clear_forces()
                pipeline.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120, observables=observables)
                poses = state.body_q.numpy()
                for foot, offset in zip(feet, offsets, strict=True):
                    expected = poses[root, :3] + np.asarray(wp.quat_rotate(wp.quat(poses[root, 3:]), offset))
                    self.assertLess(np.linalg.norm(poses[foot, :3] - expected), 5.0e-5)
                count = int(contacts.rigid_contact_count.numpy()[0])
                shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                shape1 = contacts.rigid_contact_shape1.numpy()[:count]
                force = observables.contact_f.numpy()[:count, :3]
                support = force[shape1 == ground].sum(axis=0) - force[shape0 == ground].sum(axis=0)
                normal = np.concatenate((force[shape1 == ground, 2], -force[shape0 == ground, 2]))
                self.assertGreaterEqual(normal.min(initial=0), -1.0e-7, "normal contact must not pull")
                after = (mass[:, None] * state.body_qd.numpy()[:, :3]).sum(axis=0)
                np.testing.assert_allclose(after - before, (support + gravity) / 120, atol=3.0e-7, rtol=0)
                if step >= 120:
                    forces.append(support[2])
            self.assertAlmostEqual(float(np.mean(forces)), float(-gravity[2]), delta=2.0e-5)

    def test_projection_does_not_repeat_motor_spring(self):
        """Leave the free hinge's spring increment unchanged during projection."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.8))
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.8))
            free = builder.add_joint_free(child=root)
            hinge = builder.add_joint_revolute(root, child, axis=newton.Axis.Z, target_ke=40.0)
            builder.add_articulation([free, hinge])
            model = builder.finalize(device="cpu")
            solver = SolverPhoenX(
                model,
                substeps=1,
                solver_iterations=24,
                temporal_iterations=True,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                sleeping_velocity_threshold=0,
            )
            state, control = model.state(), model.control()
            target = control.joint_target_q.numpy().copy()
            target[model.joint_target_q_start.numpy()[hinge]] = 1.0
            control.joint_target_q.assign(target)
            solver.step(state, state, control, None, 0.01)
            rows = solver.world.constraints.bilateral
            dynamic = np.flatnonzero(rows.row_dynamic.numpy())
            self.assertEqual(len(dynamic), 1)
            before = rows.accumulated.numpy().copy()
            self.assertGreater(abs(before[dynamic[0]]), 0)
            angular = solver.bodies.angular_velocity.numpy().copy()
            solver.world._project_temporal_joints()
            np.testing.assert_array_equal(rows.accumulated.numpy(), before)
            np.testing.assert_array_equal(solver.bodies.angular_velocity.numpy()[:, 2], angular[:, 2])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA projection coverage")
    def test_projection_preserves_external_impulse_cuda(self):
        for overflow in (False, True):
            with self.subTest(overflow=overflow):
                first = self._run_impulse("cuda:0", overflow)
                second = self._run_impulse("cuda:0", overflow)
                for a, b in zip(first, second, strict=True):
                    np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
