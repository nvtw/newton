# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check sleeping stacks on stopped and restarting conveyor surfaces."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.sleeping_filter import PhoenXSleepingFilterData
from newton.solvers import SolverPhoenX


@wp.func
def _retain_pair(pair: wp.vec2i, data: PhoenXSleepingFilterData) -> wp.int32:
    return wp.int32(1)


class TestSleepingConveyor(unittest.TestCase):
    def test_stopped_stack_cpu(self):
        """Sleep stopped stacks and restore complete support when the belt moves."""
        self._check_stopped_stack("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_stopped_stack_cuda(self):
        """Preserve conveyor sleep and wake behavior during CUDA graph replay."""
        self._check_stopped_stack("cuda:0")

    def _check_stopped_stack(self, device):
        reference = None
        for mode in ("awake", "canonical", "custom"):
            with self.subTest(mode=mode), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                mesh = newton.Mesh(
                    np.array([[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]], dtype=np.float32),
                    np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
                    compute_inertia=False,
                    enable_surface_velocity=True,
                )
                cfg = builder.ShapeConfig(mu=0.8)
                belt = builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg)
                for height in (0.1, 0.3, 0.5):
                    body = builder.add_body(xform=wp.transform((0, 0, height), wp.quat_identity()))
                    builder.add_shape_box(body, hx=0.18, hy=0.18, hz=0.1, cfg=cfg)
                model = builder.finalize()
                pipeline = None
                if mode == "custom":
                    pipeline = newton.CollisionPipeline(
                        model,
                        contact_matching="sticky",
                        broad_phase_filter=(_retain_pair, PhoenXSleepingFilterData),
                    )
                solver = SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    step_layout="single_world",
                    temporal_iterations=True,
                    solver_iterations=24,
                    substeps=1,
                    contact_hertz=80,
                    sleeping_velocity_threshold=0.01 if mode != "awake" else 0.0,
                    sleeping_frames_required=30,
                )
                state = model.state()
                contacts = model._collision_pipeline.contacts()
                observables = solver.observables({solver.ObservableKind.CONTACT_F}, contacts=contacts)
                masses = model.body_mass.numpy()
                weight = float(masses.sum()) * 9.81
                dt = 1 / 120

                def step(solver=solver, state=state, contacts=contacts, dt=dt, observables=observables):
                    solver.collide(state, contacts, dt=dt)
                    solver.step(state, state, None, contacts, dt, observables=observables)

                step()
                graph = None
                if model.device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        step()
                    graph = capture.graph

                def advance(graph=graph, step=step):
                    if graph is None:
                        step()
                    else:
                        wp.capture_launch(graph)

                samples = []
                previous_speed = 0.0
                # Stay below the static-friction and tipping acceleration limits.
                # The first moving command is also below the sleeping threshold,
                # so surface activity, rather than body speed, must wake the stack.
                ramp_steps = 20
                for speed, frames in ((0.0, 180), (0.15, 120), (0.0, 180), (-0.15, 120), (0.0, 180)):
                    mesh.mesh.velocities.fill_(wp.vec3(previous_speed + (speed - previous_speed) / ramp_steps, 0, 0))
                    previous_velocity = state.body_qd.numpy()[:, :3].copy()
                    advance()
                    if speed != 0:
                        # The first moving step must keep every stack contact, not
                        # only the belt pair, and wake before gravity is integrated.
                        self.assertTrue(np.all(solver.bodies.island_root.numpy()[1:] < 0))
                        count = int(contacts.rigid_contact_count.numpy()[0])
                        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
                        force = observables.contact_f.numpy()[:count, :3]
                        external_force = force[shape1 == belt].sum(axis=0) - force[shape0 == belt].sum(axis=0)
                        momentum_rate = ((state.body_qd.numpy()[:, :3] - previous_velocity) * masses[:, None] / dt).sum(
                            axis=0
                        )
                        momentum_rate[2] += weight
                        np.testing.assert_allclose(external_force, momentum_rate, atol=0.002)
                        self.assertGreater(float(external_force[2]), 0.9 * weight)
                        self.assertLess(float(external_force[2]), 1.15 * weight)
                    velocity_window = []
                    position_window = []
                    for frame in range(frames - 1):
                        if frame < ramp_steps - 1:
                            command = previous_speed + (speed - previous_speed) * (frame + 2) / ramp_steps
                            mesh.mesh.velocities.fill_(wp.vec3(command, 0, 0))
                        advance()
                        if frame >= frames - 31:
                            velocity_window.append(state.body_qd.numpy()[:, :3].copy())
                            position_window.append(state.body_q.numpy()[:, :3].copy())
                    previous_speed = speed
                    pose = state.body_q.numpy().copy()
                    np.testing.assert_allclose(pose[:, 2], (0.1, 0.3, 0.5), atol=0.001)
                    if speed == 0 and mode != "awake":
                        self.assertTrue(np.all(solver.bodies.island_root.numpy()[1:] >= 0))
                    if speed == 0:
                        advance()
                        if mode != "awake":
                            np.testing.assert_array_equal(state.body_q.numpy(), pose)
                    if speed != 0:
                        expected = np.tile((speed, 0, 0), (3, 1))
                        np.testing.assert_allclose(state.body_qd.numpy()[:, :3], expected, atol=0.001)
                        velocity_window = np.asarray(velocity_window)
                        np.testing.assert_allclose(velocity_window.mean(axis=0), expected, atol=0.001)
                        self.assertLess(float(np.sqrt(np.mean((velocity_window - expected) ** 2))), 0.001)
                        times = np.arange(len(position_window)) * dt
                        relative = np.asarray(position_window) - times[:, None, None] * expected[None, :, :]
                        self.assertLess(float(np.ptp(relative, axis=0).max()), 0.001)
                    samples.append(pose[:, :3])
                if reference is None:
                    reference = np.array(samples)
                else:
                    np.testing.assert_allclose(samples, reference, atol=0.001)

    def test_plain_scene_still_prunes_sleeping_pairs(self):
        """Retain the existing broad-phase optimization without conveyor surfaces."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0, 0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            builder.add_ground_plane()
            model = builder.finalize()
            solver = SolverPhoenX(model, sleeping_velocity_threshold=0.01, sleeping_frames_required=30)
            state = model.state()
            contacts = model._collision_pipeline.contacts()
            for _ in range(120):
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, None, contacts, 1 / 120)
            self.assertGreaterEqual(solver.bodies.island_root.numpy()[body + 1], 0)
            solver.collide(state, contacts, dt=1 / 120)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
