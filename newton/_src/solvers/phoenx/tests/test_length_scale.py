# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact rows behave the same in metres and in centimetres."""

import unittest

import numpy as np
import warp as wp

import newton


def _overlapping_box_heights(scale: float) -> np.ndarray:
    """Heights of a 0.1 m box that starts 5 mm into the ground, in a scene authored at ``scale`` units per metre."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 * scale))
    builder.rigid_gap = 0.005 * scale
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0 / scale**3, mu=0.5)
    builder.add_ground_plane(cfg=cfg)
    half = 0.05 * scale
    body = builder.add_body(xform=wp.transform((0.0, 0.0, half - 0.005 * scale), wp.quat_identity()))
    builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, substeps=1, solver_iterations=8)
    state, control, contacts = model.state(), model.control(), pipeline.contacts()

    def step():
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, 1.0 / 240.0)

    step()
    with wp.ScopedCapture() as capture:
        step()
    heights = []
    for _ in range(24):
        wp.capture_launch(capture.graph)
        heights.append(float(state.body_q.numpy()[body, 2]) / scale)
    return np.asarray(heights)


@unittest.skipUnless(wp.is_cuda_available(), "requires CUDA graph capture")
class TestLengthScale(unittest.TestCase):
    def test_penetration_recovery_matches_across_units(self):
        """Push an overlapping box out at the same rate whether the scene is in metres or centimetres.

        The push-out speed cap is a speed in scene units: unscaled, it lets a centimetre
        scene recover at 2 cm/s instead of 2 m/s.
        """
        with wp.ScopedDevice("cuda:0"):
            metres = _overlapping_box_heights(1.0)
            centimetres = _overlapping_box_heights(100.0)
        self.assertGreater(metres[-1], 0.0495)
        np.testing.assert_allclose(centimetres, metres, atol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
