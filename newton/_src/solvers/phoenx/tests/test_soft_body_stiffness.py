# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Soft-body stiffness response checks for PhoenX."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tet_neohookean import (
    SoftBodyConstraintType,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

_BLOCK_SIZE = 0.30
_GRID_DIM = 2
_CELL_SIZE = _BLOCK_SIZE / _GRID_DIM
_POISSON = 0.25
_COMPRESSION = 0.005
_DENSITY = 200.0
_BETA = 8.0
_SUBSTEPS = 10
_SOLVER_ITERATIONS = 64
_FRAME_DT = 1.0 / 120.0


def _pin_bottom_layer(builder: newton.ModelBuilder, start: int) -> None:
    for i, p in enumerate(builder.particle_q[start:], start):
        if abs(float(p[2])) < 1.0e-7:
            builder.particle_mass[i] = 0.0


class _CompressedSoftBlock:
    def __init__(self, *, device: wp.Device):
        k_lambda, k_mu = soft_tet_lame_from_youngs_poisson(1.0e5, _POISSON)

        builder = newton.ModelBuilder()
        start = len(builder.particle_q)
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * _BLOCK_SIZE, -0.5 * _BLOCK_SIZE, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=_GRID_DIM,
            dim_y=_GRID_DIM,
            dim_z=_GRID_DIM,
            cell_x=_CELL_SIZE,
            cell_y=_CELL_SIZE,
            cell_z=_CELL_SIZE,
            density=_DENSITY,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            add_surface_mesh_edges=False,
            particle_radius=0.003,
        )
        _pin_bottom_layer(builder, start)

        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.rest_positions = self.model.particle_q.numpy().copy()

        bodies = body_container_zeros(1, device=device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            device=device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            num_worlds=1,
            substeps=_SUBSTEPS,
            solver_iterations=_SOLVER_ITERATIONS,
            velocity_iterations=1,
            rigid_contact_max=4096,
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=12,
            device=device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(
            self.model,
            constraint_type=SoftBodyConstraintType.BLOCK_NEOHOOKEAN,
            beta_lambda=_BETA,
            beta_mu=_BETA,
        )

    def measure_recovery_speed(self, youngs_modulus: float) -> float:
        """Impose compression and return the top layer's recovery speed."""
        k_lambda, k_mu = soft_tet_lame_from_youngs_poisson(youngs_modulus, _POISSON)
        materials = np.tile([k_mu, k_lambda, 0.0], (int(self.model.tet_count), 1)).astype(np.float32)
        self.model.tet_materials.assign(materials)
        self.world.populate_soft_tetrahedra_from_model(
            self.model,
            constraint_type=SoftBodyConstraintType.BLOCK_NEOHOOKEAN,
            beta_lambda=_BETA,
            beta_mu=_BETA,
        )
        top = np.isclose(self.rest_positions[:, 2], _BLOCK_SIZE)
        positions = self.rest_positions.copy()
        positions[top, 2] -= _COMPRESSION
        self.world.particles.position.assign(positions)
        self.world.particles.velocity.zero_()

        self.world.step(_FRAME_DT, contacts=None)
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)
        return float(np.mean(self.state.particle_qd.numpy()[top, 2]))


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-body stiffness tests are CUDA-only.",
)
class TestSoftBodyStiffness(unittest.TestCase):
    def test_stiffer_block_recovers_faster_from_compression(self) -> None:
        """Verify that elastic recovery increases with Young's modulus."""
        device = wp.get_preferred_device()
        block = _CompressedSoftBlock(device=device)
        speed_low = block.measure_recovery_speed(1.0e5)
        speed_high = block.measure_recovery_speed(1.0e6)

        self.assertGreater(speed_low, 0.0)
        self.assertGreater(speed_high, 1.5 * speed_low)

        positions = block.state.particle_q.numpy()
        velocities = block.state.particle_qd.numpy()
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(velocities)))

        pinned = np.isclose(block.rest_positions[:, 2], 0.0)
        pinned_drift = np.linalg.norm(positions[pinned] - block.rest_positions[pinned], axis=1)
        self.assertLess(float(pinned_drift.max()), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
