# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.stflip.kernels import (
    finalize_grid_to_particles,
    initialize_temporal_offsets,
    normalize_grid,
    particle_faces_to_grid,
    particles_to_grid,
    sample_transfer_components,
    update_particle_clocks,
)
from newton._src.solvers.stflip.sparse_grid import SparseGrid


class TestSTFLIPTransfers(unittest.TestCase):
    def _devices(self):
        """Return every device used for transfer validation."""
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices

    def _transfer(self, device, positions, velocities, masses, active, affine=None, offsets=None, dt=0.0):
        """Transfer particle data to a normalized sparse MAC grid."""
        count = len(positions)
        flags_np = np.where(active, int(newton.ParticleFlags.ACTIVE), 0).astype(np.int32)
        positions_wp = wp.array(positions, dtype=wp.vec3, device=device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=device)
        masses_wp = wp.array(masses, dtype=wp.float32, device=device)
        inverse_masses_wp = wp.array(1.0 / masses, dtype=wp.float32, device=device)
        flags_wp = wp.array(flags_np, dtype=wp.int32, device=device)
        active_wp = wp.array(active.astype(np.int32), dtype=wp.int32, device=device)
        if affine is None:
            affine = np.zeros((count, 3, 3), dtype=np.float32)
        if offsets is None:
            offsets = np.zeros(count, dtype=np.float32)
        affine_wp = wp.array(affine, dtype=wp.mat33, device=device)
        offsets_wp = wp.array(offsets, dtype=wp.float32, device=device)
        forces_wp = wp.zeros(count, dtype=wp.vec3, device=device)

        grid = SparseGrid(
            point_capacity=count,
            tile_capacity=512,
            tile_size=4,
            cell_size=1.0,
            padding_tiles=1,
            device=device,
        )
        grid.build(positions_wp, active_wp)
        grid.check_status()
        cell_mass = wp.zeros(grid.cell_capacity, dtype=wp.float32, device=device)
        face_mass = wp.zeros(3 * grid.cell_capacity, dtype=wp.float32, device=device)
        face_momentum = wp.zeros(3 * grid.cell_capacity, dtype=wp.float32, device=device)
        wp.launch(
            particles_to_grid,
            dim=count,
            inputs=[
                grid.data,
                positions_wp,
                velocities_wp,
                forces_wp,
                masses_wp,
                inverse_masses_wp,
                flags_wp,
                affine_wp,
                offsets_wp,
                1.0,
                dt,
                cell_mass,
            ],
            device=device,
        )
        wp.launch(
            particle_faces_to_grid,
            dim=3 * count,
            inputs=[
                grid.data,
                positions_wp,
                velocities_wp,
                forces_wp,
                masses_wp,
                inverse_masses_wp,
                flags_wp,
                affine_wp,
                offsets_wp,
                1.0,
                dt,
                face_mass,
                face_momentum,
            ],
            device=device,
        )
        face_velocity = wp.zeros_like(face_momentum)
        face_velocity_old = wp.zeros_like(face_momentum)
        gravity = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(
            normalize_grid,
            dim=grid.cell_capacity,
            inputs=[face_mass, face_momentum, gravity, 0.0, face_velocity, face_velocity_old],
            device=device,
        )
        return {
            "grid": grid,
            "positions": positions_wp,
            "velocities": velocities_wp,
            "flags": flags_wp,
            "cell_mass": cell_mass,
            "face_mass": face_mass,
            "face_momentum": face_momentum,
            "face_velocity": face_velocity,
            "face_velocity_old": face_velocity_old,
        }

    def test_p2g_conserves_mass_and_linear_momentum(self):
        """Conserve active mass and linear momentum during P2G."""
        rng = np.random.default_rng(1729)
        count = 47
        positions = rng.uniform(-2.75, 2.75, size=(count, 3)).astype(np.float32)
        velocities = rng.uniform(-3.0, 3.0, size=(count, 3)).astype(np.float32)
        masses = rng.uniform(0.2, 2.0, size=count).astype(np.float32)
        active = rng.random(count) > 0.2
        expected_mass = np.sum(masses[active], dtype=np.float64)
        expected_momentum = np.sum(masses[active, None] * velocities[active], axis=0, dtype=np.float64)

        for device in self._devices():
            with self.subTest(device=device):
                transfer = self._transfer(device, positions, velocities, masses, active)
                cell_mass = transfer["cell_mass"].numpy()
                face_mass = transfer["face_mass"].numpy().reshape(-1, 3)
                face_momentum = transfer["face_momentum"].numpy().reshape(-1, 3)

                np.testing.assert_allclose(np.sum(cell_mass), expected_mass, rtol=3.0e-6, atol=3.0e-6)
                np.testing.assert_allclose(
                    np.sum(face_mass, axis=0), np.full(3, expected_mass), rtol=3.0e-6, atol=3.0e-6
                )
                np.testing.assert_allclose(np.sum(face_momentum, axis=0), expected_momentum, rtol=5.0e-6, atol=5.0e-6)

    def test_pic_reproduces_constant_velocity(self):
        """Reproduce a constant velocity field through PIC transfer."""
        rng = np.random.default_rng(234)
        positions = rng.uniform(-1.8, 1.8, size=(31, 3)).astype(np.float32)
        velocity = np.array([1.25, -0.75, 2.5], dtype=np.float32)
        velocities = np.broadcast_to(velocity, positions.shape).copy()
        masses = rng.uniform(0.5, 1.5, size=len(positions)).astype(np.float32)
        active = np.ones(len(positions), dtype=bool)

        for device in self._devices():
            with self.subTest(device=device):
                transfer = self._transfer(device, positions, velocities, masses, active)
                samples = wp.zeros(3 * len(positions), dtype=wp.vec4, device=device)
                gradient_z = wp.zeros(3 * len(positions), dtype=wp.float32, device=device)
                positions_out = wp.zeros_like(transfer["positions"])
                velocities_out = wp.zeros_like(transfer["velocities"])
                affine_out = wp.zeros(len(positions), dtype=wp.mat33, device=device)
                wp.launch(
                    sample_transfer_components,
                    dim=3 * len(positions),
                    inputs=[
                        transfer["grid"].data,
                        transfer["positions"],
                        transfer["flags"],
                        1.0,
                        transfer["face_velocity"],
                        transfer["face_velocity"],
                        False,
                        samples,
                        gradient_z,
                    ],
                    device=device,
                )
                wp.launch(
                    finalize_grid_to_particles,
                    dim=len(positions),
                    inputs=[
                        transfer["positions"],
                        transfer["velocities"],
                        transfer["flags"],
                        0.0,
                        False,
                        samples,
                        gradient_z,
                        positions_out,
                        velocities_out,
                        affine_out,
                    ],
                    device=device,
                )

                np.testing.assert_allclose(velocities_out.numpy(), velocities, rtol=2.0e-6, atol=2.0e-6)

    def test_apic_reproduces_affine_velocity(self):
        """Reproduce particle velocity and gradient through APIC transfer."""
        positions = np.array([[0.37, -0.21, 0.63]], dtype=np.float32)
        velocities = np.array([[1.2, -0.4, 0.8]], dtype=np.float32)
        affine = np.array(
            [[[0.3, -0.2, 0.1], [0.05, -0.4, 0.25], [-0.15, 0.35, 0.2]]],
            dtype=np.float32,
        )
        masses = np.ones(1, dtype=np.float32)
        active = np.ones(1, dtype=bool)

        for device in self._devices():
            with self.subTest(device=device):
                transfer = self._transfer(device, positions, velocities, masses, active, affine=affine)
                samples = wp.zeros(3, dtype=wp.vec4, device=device)
                gradient_z = wp.zeros(3, dtype=wp.float32, device=device)
                positions_out = wp.zeros_like(transfer["positions"])
                velocities_out = wp.zeros_like(transfer["velocities"])
                affine_out = wp.zeros(1, dtype=wp.mat33, device=device)
                wp.launch(
                    sample_transfer_components,
                    dim=3,
                    inputs=[
                        transfer["grid"].data,
                        transfer["positions"],
                        transfer["flags"],
                        1.0,
                        transfer["face_velocity"],
                        transfer["face_velocity"],
                        True,
                        samples,
                        gradient_z,
                    ],
                    device=device,
                )
                wp.launch(
                    finalize_grid_to_particles,
                    dim=1,
                    inputs=[
                        transfer["positions"],
                        transfer["velocities"],
                        transfer["flags"],
                        0.0,
                        True,
                        samples,
                        gradient_z,
                        positions_out,
                        velocities_out,
                        affine_out,
                    ],
                    device=device,
                )

                np.testing.assert_allclose(velocities_out.numpy(), velocities, rtol=2.0e-6, atol=2.0e-6)
                np.testing.assert_allclose(affine_out.numpy(), affine, rtol=3.0e-6, atol=3.0e-6)

    def test_flip_preserves_velocity_without_grid_delta(self):
        """Preserve particle velocity when the FLIP grid delta is zero."""
        rng = np.random.default_rng(991)
        positions = rng.uniform(-1.5, 1.5, size=(23, 3)).astype(np.float32)
        velocities = rng.uniform(-2.0, 2.0, size=(23, 3)).astype(np.float32)
        masses = np.ones(len(positions), dtype=np.float32)
        active = np.ones(len(positions), dtype=bool)

        for device in self._devices():
            with self.subTest(device=device):
                transfer = self._transfer(device, positions, velocities, masses, active)
                samples = wp.zeros(3 * len(positions), dtype=wp.vec4, device=device)
                gradient_z = wp.zeros(3 * len(positions), dtype=wp.float32, device=device)
                positions_out = wp.zeros_like(transfer["positions"])
                velocities_out = wp.zeros_like(transfer["velocities"])
                affine_out = wp.zeros(len(positions), dtype=wp.mat33, device=device)
                wp.launch(
                    sample_transfer_components,
                    dim=3 * len(positions),
                    inputs=[
                        transfer["grid"].data,
                        transfer["positions"],
                        transfer["flags"],
                        1.0,
                        transfer["face_velocity"],
                        transfer["face_velocity"],
                        False,
                        samples,
                        gradient_z,
                    ],
                    device=device,
                )
                wp.launch(
                    finalize_grid_to_particles,
                    dim=len(positions),
                    inputs=[
                        transfer["positions"],
                        transfer["velocities"],
                        transfer["flags"],
                        1.0,
                        False,
                        samples,
                        gradient_z,
                        positions_out,
                        velocities_out,
                        affine_out,
                    ],
                    device=device,
                )

                np.testing.assert_allclose(velocities_out.numpy(), velocities, rtol=2.0e-6, atol=2.0e-6)

    def test_temporal_offsets_and_clocks_are_deterministic(self):
        """Generate deterministic bounded offsets and advance only active clocks."""
        count = 64
        flags_np = np.full(count, int(newton.ParticleFlags.ACTIVE), dtype=np.int32)
        flags_np[::7] = 0
        reference = None

        for device in self._devices():
            with self.subTest(device=device):
                offsets = wp.zeros(count, dtype=wp.float32, device=device)
                repeated = wp.zeros(count, dtype=wp.float32, device=device)
                different = wp.zeros(count, dtype=wp.float32, device=device)
                wp.launch(initialize_temporal_offsets, dim=count, inputs=[42, offsets], device=device)
                wp.launch(initialize_temporal_offsets, dim=count, inputs=[42, repeated], device=device)
                wp.launch(initialize_temporal_offsets, dim=count, inputs=[43, different], device=device)
                values = offsets.numpy()
                np.testing.assert_array_equal(values, repeated.numpy())
                self.assertFalse(np.array_equal(values, different.numpy()))
                self.assertGreaterEqual(float(np.min(values)), -0.5)
                self.assertLess(float(np.max(values)), 0.5)
                if reference is None:
                    reference = values
                else:
                    np.testing.assert_array_equal(values, reference)

                flags = wp.array(flags_np, dtype=wp.int32, device=device)
                age_in = wp.array(np.linspace(0.0, 1.0, count, dtype=np.float32), device=device)
                residual = wp.zeros(count, dtype=wp.float32, device=device)
                age_out = wp.zeros(count, dtype=wp.float32, device=device)
                wp.launch(
                    update_particle_clocks,
                    dim=count,
                    inputs=[flags, offsets, 0.125, age_in, residual, age_out],
                    device=device,
                )
                expected_residual = np.where(flags_np != 0, values * 0.125, 0.0)
                expected_age = age_in.numpy() + np.where(flags_np != 0, 0.125, 0.0)
                np.testing.assert_allclose(residual.numpy(), expected_residual, rtol=0.0, atol=1.0e-8)
                np.testing.assert_allclose(age_out.numpy(), expected_age, rtol=0.0, atol=1.0e-7)

    def test_temporal_p2g_matches_explicit_position_shift(self):
        """Match staggered P2G to transfer from explicitly shifted particles."""
        rng = np.random.default_rng(867)
        count = 19
        positions = rng.uniform(1.0, 2.0, size=(count, 3)).astype(np.float32)
        velocities = rng.uniform(-0.7, 0.7, size=(count, 3)).astype(np.float32)
        masses = rng.uniform(0.5, 1.5, size=count).astype(np.float32)
        offsets = rng.uniform(-0.5, 0.5, size=count).astype(np.float32)
        active = np.ones(count, dtype=bool)
        dt = 0.2
        shifted_positions = positions + velocities * offsets[:, None] * dt

        for device in self._devices():
            with self.subTest(device=device):
                staggered = self._transfer(
                    device,
                    positions,
                    velocities,
                    masses,
                    active,
                    offsets=offsets,
                    dt=dt,
                )
                explicit = self._transfer(device, shifted_positions, velocities, masses, active)

                np.testing.assert_allclose(
                    staggered["cell_mass"].numpy(), explicit["cell_mass"].numpy(), rtol=2.0e-6, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    staggered["face_mass"].numpy(), explicit["face_mass"].numpy(), rtol=2.0e-6, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    staggered["face_momentum"].numpy(),
                    explicit["face_momentum"].numpy(),
                    rtol=2.0e-6,
                    atol=2.0e-6,
                )


if __name__ == "__main__":
    unittest.main()
