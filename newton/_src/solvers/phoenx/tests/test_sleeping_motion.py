# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise pose-based sleeping and the existing external wake path."""

import unittest

import numpy as np
import warp as wp

import newton


class TestMotionSleeping(unittest.TestCase):
    def test_settle_and_wake_cpu(self):
        self._settle_and_wake("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_settle_and_wake_cuda(self):
        self._settle_and_wake("cuda:0")

    def _settle_and_wake(self, device):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                sleeping_velocity_threshold=0.01,
                sleeping_frames_required=1,
                sleeping_motion_threshold=0.0002,
            )
            state, control, contacts = model.state(), model.control(), pipeline.contacts()

            def step():
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)

            if wp.get_device().is_cuda:
                with wp.ScopedCapture() as capture:
                    step()

                def advance():
                    wp.capture_launch(capture.graph)
            else:
                advance = step

            for _ in range(20):
                advance()
            self.assertEqual(solver.world.bodies.island_root.numpy()[body + 1], -1)
            for _ in range(100):
                advance()
            self.assertGreaterEqual(solver.world.bodies.island_root.numpy()[body + 1], 0)
            sleeping_pose = state.body_q.numpy()[body].copy()
            for _ in range(30):
                advance()
            np.testing.assert_array_equal(state.body_q.numpy()[body], sleeping_pose)

            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[body, 2] = 5000.0
            state.body_f.assign(force)
            advance()
            self.assertEqual(solver.world.bodies.island_root.numpy()[body + 1], -1)
            self.assertGreater(state.body_qd.numpy()[body, 2], 0.0)
