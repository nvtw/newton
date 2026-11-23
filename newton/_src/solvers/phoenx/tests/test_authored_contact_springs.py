# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Authored shape contact springs must let a resting stack come to rest."""

import math
import unittest

import numpy as np
import warp as wp

import newton


@unittest.skipUnless(wp.is_cuda_available(), "requires CUDA graph capture")
class TestAuthoredContactSprings(unittest.TestCase):
    def test_stiff_authored_springs_settle_a_stack(self):
        """Settle a stack whose authored springs are stiffer than the sweeps keep stable.

        Benchmarks size ``ke`` to hold a load within a small overlap and set
        ``kd = ke * dt``. On 20 kg blocks at 480 Hz that spring has
        ``omega * dt`` of about 4.6 and damping ratio 2.4, which the eight
        warm-started sweeps cannot hold still: the stack jitters at several
        cm/s instead of resting.
        """
        h, mass, dt = 0.1, 20.0, 1.0 / 480.0
        ke = 1.0e8
        kd = max(ke * dt, 2.0 * math.sqrt(ke * mass))
        cfg = newton.ModelBuilder.ShapeConfig(density=mass / (8 * h**3), ke=ke, kd=kd, mu=0.7, restitution=0.0)
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            builder.rigid_gap = 0.005
            builder.add_ground_plane(cfg=cfg)
            for i in range(6):
                body = builder.add_body(xform=wp.transform((0.0, 0.0, h + 2 * h * i), wp.quat_identity()))
                builder.add_shape_box(body, hx=h, hy=h, hz=h, cfg=cfg)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(
                model, collision_pipeline=pipeline, substeps=1, solver_iterations=8, velocity_iterations=1
            )
            state, control, contacts = model.state(), model.control(), pipeline.contacts()

            def step():
                pipeline.collide(state, contacts)
                solver.step(state, state, control, contacts, dt)

            step()
            with wp.ScopedCapture() as capture:
                step()
            for _ in range(480):
                wp.capture_launch(capture.graph)
            speed = np.abs(state.body_qd.numpy()).max()
            heights = state.body_q.numpy()[:, 2]
        self.assertLess(speed, 2.0e-3)
        # The stack still carries its weight: no block sinks into the one below.
        np.testing.assert_allclose(heights, h + 2 * h * np.arange(6), atol=1.0e-3)


if __name__ == "__main__":
    unittest.main()
