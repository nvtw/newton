# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


class TestSolverSTFLIP(unittest.TestCase):
    def _create_fluid(self, device, transfer_scheme="apic", gravity=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)):
        """Create a compact fluid model and solver."""
        builder = newton.ModelBuilder(gravity=gravity)
        newton.solvers.SolverSTFLIP.register_custom_attributes(builder)
        spacing = 0.04
        builder.add_particle_grid(
            pos=wp.vec3(0.2, 0.2, 0.2),
            rot=wp.quat_identity(),
            vel=wp.vec3(velocity),
            dim_x=3,
            dim_y=3,
            dim_z=3,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=1000.0 * spacing**3,
            jitter=0.0,
            radius_mean=0.015,
        )
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSTFLIP(
            model,
            newton.solvers.SolverSTFLIP.Config(
                cell_size=0.08,
                tile_size=4,
                max_active_tile_count=125,
                padding_tiles=1,
                pressure_iterations=80,
                transfer_scheme=transfer_scheme,
            ),
        )
        return model, solver

    def _advance(self, solver, state_in, state_out, steps, dt):
        """Advance and swap two state buffers."""
        for _ in range(steps):
            state_in.clear_forces()
            solver.step(state_in, state_out, None, None, dt)
            state_in, state_out = state_out, state_in
        solver.check_status()
        return state_in, state_out

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

    def test_transfer_schemes_preserve_uniform_translation(self):
        """Preserve uniform translation over repeated complete solver steps."""
        velocity = np.array([0.35, -0.2, 0.15], dtype=np.float32)
        dt = 1.0 / 120.0
        steps = 20
        for transfer_scheme in ("pic", "flip", "apic"):
            with self.subTest(transfer_scheme=transfer_scheme):
                model, solver = self._create_fluid("cpu", transfer_scheme, velocity=velocity)
                state_in = model.state()
                state_out = model.state()
                positions_initial = state_in.particle_q.numpy().copy()
                state_in, _state_out = self._advance(solver, state_in, state_out, steps, dt)

                np.testing.assert_allclose(
                    state_in.particle_q.numpy(),
                    positions_initial + velocity * (steps * dt),
                    rtol=5.0e-5,
                    atol=5.0e-5,
                )
                np.testing.assert_allclose(
                    state_in.particle_qd.numpy(),
                    np.broadcast_to(velocity, positions_initial.shape),
                    rtol=5.0e-5,
                    atol=5.0e-5,
                )
                self.assertAlmostEqual(
                    float(np.sum(solver.cell_mass.numpy())),
                    float(np.sum(model.particle_mass.numpy())),
                    places=5,
                )

    def test_uniform_gravity_matches_semi_implicit_motion(self):
        """Match semi-implicit free fall for a uniform fluid velocity."""
        gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        dt = 1.0 / 240.0
        steps = 16
        model, solver = self._create_fluid("cpu", "pic", gravity=gravity)
        state_in = model.state()
        state_out = model.state()
        positions_initial = state_in.particle_q.numpy().copy()
        state_in, _state_out = self._advance(solver, state_in, state_out, steps, dt)

        expected_velocity = gravity * (steps * dt)
        expected_displacement = gravity * (dt * dt * steps * (steps + 1) * 0.5)
        np.testing.assert_allclose(
            state_in.particle_qd.numpy(),
            np.broadcast_to(expected_velocity, positions_initial.shape),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            state_in.particle_q.numpy(),
            positions_initial + expected_displacement,
            rtol=2.0e-5,
            atol=2.0e-5,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_cpu_cuda_multistep_agreement(self):
        """Keep CPU and CUDA particle states close over repeated steps."""
        rng = np.random.default_rng(712)
        initial_velocity = rng.uniform(-0.3, 0.3, size=(27, 3)).astype(np.float32)
        results = {}
        for device in ("cpu", "cuda:0"):
            model, solver = self._create_fluid(device, "apic", gravity=(0.0, 0.0, -1.0))
            state_in = model.state()
            state_out = model.state()
            wp.copy(state_in.particle_qd, wp.array(initial_velocity, dtype=wp.vec3, device=device))
            state_in, _state_out = self._advance(solver, state_in, state_out, 8, 1.0 / 240.0)
            results[device] = (state_in.particle_q.numpy(), state_in.particle_qd.numpy())

        np.testing.assert_allclose(results["cuda:0"][0], results["cpu"][0], rtol=2.0e-4, atol=2.0e-5)
        np.testing.assert_allclose(results["cuda:0"][1], results["cpu"][1], rtol=2.0e-4, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
