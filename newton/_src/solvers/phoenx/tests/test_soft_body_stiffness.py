# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Soft-body stiffness calibration checks for PhoenX.

The fixture compresses a pinned soft cube with a static rigid platen and
compares the measured vertical reaction to the small-strain prediction

    F = E A delta / H

where ``E`` is Young's modulus, ``A`` is the loaded area, ``H`` is the
block height, and ``delta`` is the imposed compression. This is the
inverse of the "known weight causes known sink" check, but it avoids a
flaky guided rigid-body setup and validates the same stiffness law.
"""

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
_PLATEN_HALF_Z = 0.02
_DENSITY = 200.0
_BETA = 8.0
_SUBSTEPS = 10
_SOLVER_ITERATIONS = 64
_FRAME_DT = 1.0 / 120.0
_SETTLE_FRAMES = 120
_MEASURE_FRAMES = 8


def _pin_bottom_layer(builder: newton.ModelBuilder, start: int) -> None:
    for i, p in enumerate(builder.particle_q[start:], start):
        if abs(float(p[2])) < 1.0e-7:
            builder.particle_mass[i] = 0.0


class _CompressedSoftBlock:
    def __init__(self, *, youngs_modulus: float, device: wp.Device):
        k_lambda, k_mu = soft_tet_lame_from_youngs_poisson(youngs_modulus, _POISSON)

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

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, _BLOCK_SIZE + _PLATEN_HALF_Z - _COMPRESSION),
                q=wp.quat_identity(),
            ),
            hx=0.55 * _BLOCK_SIZE,
            hy=0.55 * _BLOCK_SIZE,
            hz=_PLATEN_HALF_Z,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8),
        )

        self.device = device
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
        self.pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            soft_body_thickness=0.001,
            soft_body_gap=0.002,
            rigid_contact_max=4096,
        )
        self.contacts = self.pipeline.contacts()

    def _step(self) -> None:
        self.world.collide(self.state, self.contacts)
        self.world.step(_FRAME_DT, contacts=self.contacts)
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def run_captured(self) -> None:
        self._step()
        with wp.ScopedCapture(device=self.device) as capture:
            self._step()
        self.graph = capture.graph
        for _ in range(_SETTLE_FRAMES):
            wp.capture_launch(self.graph)

    def measure_vertical_reaction(self) -> float:
        substep_dt = _FRAME_DT / _SUBSTEPS
        readings = []
        for _ in range(_MEASURE_FRAMES):
            wp.capture_launch(self.graph)
            wp.synchronize_device(self.device)
            contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
            normals = self.contacts.rigid_contact_normal.numpy()[:contact_count]
            lambdas = self.world._contact_container.impulses.numpy()[0, :contact_count]
            vertical_impulse = float(np.sum(np.abs(normals[:, 2]) * lambdas))
            readings.append(vertical_impulse / substep_dt)
        return float(np.mean(readings))


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-body stiffness tests are CUDA-only.",
)
class TestSoftBodyStiffness(unittest.TestCase):
    def test_platen_reaction_matches_hooke_law_and_scales(self) -> None:
        device = wp.get_preferred_device()
        low = _CompressedSoftBlock(youngs_modulus=1.0e5, device=device)
        high = _CompressedSoftBlock(youngs_modulus=1.0e6, device=device)
        low.run_captured()
        high.run_captured()

        force_low = low.measure_vertical_reaction()
        force_high = high.measure_vertical_reaction()

        expected_force = 1.0e6 * (_BLOCK_SIZE * _BLOCK_SIZE) * _COMPRESSION / _BLOCK_SIZE
        measured_force = force_high
        rel_err = abs(measured_force - expected_force) / expected_force
        self.assertLess(
            rel_err,
            0.10,
            f"vertical reaction {measured_force:.2f} N vs Hooke prediction "
            f"{expected_force:.2f} N (rel err {rel_err:.1%})",
        )

        ratio = force_high / max(force_low, 1.0e-6)
        self.assertGreater(ratio, 8.0, f"reaction ratio too small: {ratio:.2f}")
        self.assertLess(ratio, 12.0, f"reaction ratio too large: {ratio:.2f}")

        positions = high.state.particle_q.numpy()
        velocities = high.state.particle_qd.numpy()
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(velocities)))

        pinned = np.isclose(high.rest_positions[:, 2], 0.0)
        pinned_drift = np.linalg.norm(positions[pinned] - high.rest_positions[pinned], axis=1)
        self.assertLess(float(pinned_drift.max()), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
