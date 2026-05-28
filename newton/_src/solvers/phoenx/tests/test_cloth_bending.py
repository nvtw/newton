# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for PhoenX cloth bending constraints.

The tests use a small pinned cloth grid with interior bending edges and
check both graph-captured execution and high-vs-low stiffness behavior.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cloth tests are CUDA-only.",
)
class TestClothBending(unittest.TestCase):
    def _build_cloth(self, *, bending_stiffness: float, dim_x: int = 4, dim_y: int = 4):
        device = wp.get_preferred_device()
        builder = newton.ModelBuilder()
        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            fix_left=True,  # pin one edge so we don't free-fall
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            edge_ke=bending_stiffness,
            edge_kd=0.0,
            particle_radius=0.04,
        )
        model = builder.finalize(device=device)

        bodies = body_container_zeros(max(1, int(model.body_count)), device=device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(model.tri_count),
            num_cloth_bending=int(model.edge_count),
            device=device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(model.particle_count),
            num_cloth_triangles=int(model.tri_count),
            num_cloth_bending=int(model.edge_count),
            rigid_contact_max=1,  # no contacts in this test
            num_worlds=1,
            substeps=1,
            solver_iterations=8,
            step_layout="single_world",
            device=device,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        world.populate_cloth_triangles_from_model(model)
        world.populate_cloth_bending_from_model(model)
        return world, model, device

    def test_high_stiffness_resists_deformation(self):
        # With very high bending stiffness, the cloth should stay
        # near-flat (no folding) even under gravity. The dihedral-angle
        # constraint pulls every hinge back toward its rest angle (0
        # for the flat cloth grid here).
        world_soft, _, _ = self._build_cloth(bending_stiffness=0.01)
        world_stiff, _, device = self._build_cloth(bending_stiffness=1.0e6)
        for _ in range(30):
            world_soft.step(1.0 / 60.0)
            world_stiff.step(1.0 / 60.0)
        p_soft = world_soft.particles.position.numpy()
        p_stiff = world_stiff.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(p_soft)))
        self.assertTrue(np.all(np.isfinite(p_stiff)))

        # The "stiff" cloth should have less spread in z than the soft
        # cloth (resists bending → stays closer to its rest plane).
        # Pinned vertices are at z=0.5; bending stiffness keeps the
        # rest near that plane.
        z_range_soft = float(p_soft[:, 2].max() - p_soft[:, 2].min())
        z_range_stiff = float(p_stiff[:, 2].max() - p_stiff[:, 2].min())
        self.assertLess(
            z_range_stiff,
            z_range_soft,
            f"stiff cloth z-range ({z_range_stiff:.3f}) should be < soft ({z_range_soft:.3f})",
        )

        with wp.ScopedCapture(device=device) as capture:
            world_stiff.step(1.0 / 60.0)
        for _ in range(5):
            wp.capture_launch(capture.graph)
        self.assertTrue(np.all(np.isfinite(world_stiff.particles.position.numpy())))


if __name__ == "__main__":
    unittest.main()
