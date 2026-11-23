# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Runtime gravity edits must reach a captured PhoenX step."""

import unittest

import numpy as np
import warp as wp

import newton


class TestLiveGravity(unittest.TestCase):
    def test_set_gravity_reaches_captured_graph(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cuda:0")
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=4)
        dt = 0.01
        with wp.ScopedCapture(device=model.device) as capture:
            for _ in range(2):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0

        # Model.set_gravity writes model.gravity in place, as a tilting
        # table does each frame; the captured step must use the new value.
        model.set_gravity((2.0, 0.0, -9.81))
        for _ in range(5):
            wp.capture_launch(capture.graph)
        velocity = state_0.body_qd.numpy()[body, :3]
        np.testing.assert_allclose(velocity, [2.0 * 10 * dt, 0.0, -9.81 * 10 * dt], rtol=1.0e-4, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
