# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


class TestSolverSTFLIP(unittest.TestCase):
    def _run_step(self, device, capture=False):
        """Run one solver step on the requested device."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        newton.solvers.SolverSTFLIP.register_custom_attributes(builder)
        spacing = 0.04
        density = 1000.0
        builder.add_particle_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=3,
            dim_y=3,
            dim_z=3,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=density * spacing**3,
            jitter=0.0,
            radius_mean=0.015,
        )
        model = builder.finalize(device=device)
        state_in = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverSTFLIP(
            model,
            newton.solvers.SolverSTFLIP.Config(
                cell_size=0.08,
                tile_size=4,
                max_active_tile_count=125,
                padding_tiles=1,
                pressure_iterations=4,
            ),
        )
        solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        if capture:
            with wp.ScopedCapture(device=device) as captured:
                solver.step(state_out, state_in, None, None, 1.0 / 120.0)
            wp.capture_launch(captured.graph)
            state_out = state_in
        solver.check_status()
        return solver, state_out

    def test_step_preserves_finite_particle_state(self):
        """Advance a fluid block while preserving finite particle state."""
        solver, state_out = self._run_step("cpu")

        positions = state_out.particle_q.numpy()
        velocities = state_out.particle_qd.numpy()
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(velocities)))
        self.assertAlmostEqual(float(np.sum(solver.cell_mass.numpy())), 27.0 * 1000.0 * 0.04**3, places=5)
        ages = state_out.stflip.particle_age.numpy()
        residuals = state_out.stflip.particle_time_residual.numpy()
        self.assertTrue(np.allclose(ages, 1.0 / 120.0))
        self.assertLessEqual(float(np.max(np.abs(residuals))), 0.5 / 120.0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_step_runs_on_cuda(self):
        """Advance a fluid block on the production CUDA path."""
        solver, state_out = self._run_step("cuda:0")

        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertGreater(float(np.sum(solver.cell_mass.numpy())), 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_step_captures_on_cuda(self):
        """Capture and replay a complete sparse-fluid step on CUDA."""
        _solver, state_out = self._run_step("cuda:0", capture=True)

        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))


if __name__ == "__main__":
    unittest.main()
