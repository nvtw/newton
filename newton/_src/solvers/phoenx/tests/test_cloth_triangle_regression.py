# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-triangle cloth regression: scene-build + iterate stability.

Commit 6 of :file:`PLAN_CLOTH_TRIANGLE.md`. The earlier
:mod:`test_cloth_triangle_iterate` covered the math on a single
triangle (rest-pose convergence, area decay, shear decay).
:mod:`test_cloth_triangle_populate` covered the data-flow at the
scene-build boundary.

This file covers the multi-triangle iterate path on a scene built
through the full Newton mesh API:

* A cloth grid built via :meth:`~newton.ModelBuilder.add_cloth_grid`
  with two pinned corners.
* The scene goes through
  :meth:`PhoenXWorld.populate_cloth_triangles_from_model` so the
  cloth rows carry real ``inv_rest`` / ``rest_area`` / compliance
  values from the Newton mesh API (vs. the unit-rest-triangle
  fixtures the iterate-only tests use).
* A single particle is perturbed off rest; the iterate is run for
  many sweeps; we assert:
  * No NaN or Inf in any particle position.
  * The pinned corners haven't moved.
  * The perturbed particle has been pulled back toward (but not
    necessarily all the way to) the rest pose.

The end-to-end ``world.step(dt)`` driver path that integrates
particle gravity and velocity recovery is a follow-up; once that
lands, this regression will be extended to drive 60 frames of
gravity + iterate and assert the bottom edge has fallen to the
expected XPBD steady-state distance.

CUDA-only by Newton convention.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.body_or_particle import BodyOrParticleStore
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_triangle_iterate_at,
    cloth_triangle_prepare_for_iteration_at,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@wp.kernel(enable_backward=False)
def _prepare_all_cloth_kernel(
    c: ConstraintContainer,
    cid_offset: wp.int32,
    store: BodyOrParticleStore,
    idt: wp.float32,
):
    t = wp.tid()
    cid = cid_offset + t
    cloth_triangle_prepare_for_iteration_at(c, cid, wp.int32(0), store, idt)


@wp.kernel(enable_backward=False)
def _iterate_all_cloth_kernel(
    c: ConstraintContainer,
    cid_offset: wp.int32,
    store: BodyOrParticleStore,
):
    t = wp.tid()
    cid = cid_offset + t
    cloth_triangle_iterate_at(c, cid, wp.int32(0), store)


@unittest.skipUnless(wp.is_cuda_available(), "Cloth-regression test requires CUDA")
class TestClothTriangleRegression(unittest.TestCase):
    """Drive a 4x4-cell pinned cloth grid through the cloth iterate
    kernels for many sweeps; assert stability."""

    def setUp(self) -> None:
        self.device = wp.get_device()
        self.dim_x = 4
        self.dim_y = 4
        self.cell = 0.25
        self.mass = 0.05
        # Stiff cloth -- lower compliance => the XPBD iterate
        # makes a meaningful position correction per sweep. Newton's
        # builder default is 100.0 which is essentially "rubber
        # band" and won't recover noticeable perturbation in 64
        # iterations.
        self.tri_ke = 1.0e6  # k_mu (shear)
        self.tri_ka = 1.0e6  # k_lambda (area)
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=self.mass,
            # Pin the two top corners. Builder honours these via
            # ``inv_mass = 0`` -- the iterate's mass-weighted update
            # leaves zero-inv-mass particles in place.
            fix_top=False,
            fix_bottom=False,
            fix_left=True,  # pins entire left edge
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
        )
        self.model = builder.finalize(device=self.device)
        self.particle_count = (self.dim_x + 1) * (self.dim_y + 1)
        self.tri_count = self.dim_x * self.dim_y * 2
        self.assertEqual(self.model.particle_count, self.particle_count)
        self.assertEqual(self.model.tri_count, self.tri_count)

    def _make_world(self) -> PhoenXWorld:
        bodies = body_container_zeros(1, device=self.device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=self.tri_count,
            device=self.device,
        )
        return PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=self.particle_count,
            num_cloth_triangles=self.tri_count,
            num_worlds=1,
            step_layout="single_world",
            device=self.device,
        )

    def test_iterate_is_stable_under_perturbation(self) -> None:
        """Build a 4x4 cloth grid, populate, perturb one interior
        particle, run the cloth iterate for many sweeps. No NaN /
        no Inf, pinned corners stay put, perturbed particle gets
        pulled back."""
        world = self._make_world()
        world.populate_cloth_triangles_from_model(self.model)

        # Identify a free interior particle at grid coordinate (2, 2)
        # in the (dim_x+1) x (dim_y+1) lattice. Pinned particles
        # (left edge, x == 0) all have ``inv_mass == 0``.
        gx, gy = 2, 2
        free_index = gy * (self.dim_x + 1) + gx

        # Pinned indices (the entire left edge, x == 0).
        pinned_indices = [j * (self.dim_x + 1) for j in range(self.dim_y + 1)]

        positions = world.particles.position.numpy()
        rest_position = positions[free_index].copy()
        # Push the free particle 0.5 m straight down (away from
        # cloth plane). This drives the cloth iterate hard.
        positions[free_index] = rest_position + np.array([0.0, 0.0, -0.5], dtype=np.float32)
        world.particles.position.assign(positions)

        # Capture pinned positions for the post-iterate compare.
        pinned_before = positions[pinned_indices].copy()

        # Run prepare + 64 iterate sweeps. 64 is enough to converge
        # the perturbation back toward rest under the chosen
        # stiffness; we don't assert exact convergence (the iterate
        # is XPBD, not a static solver) -- just that the magnitude
        # of the residual drops materially and no NaN appears.
        idt = wp.float32(1.0 / 0.01)  # 100 Hz substep
        wp.launch(
            kernel=_prepare_all_cloth_kernel,
            dim=self.tri_count,
            inputs=[world.constraints, wp.int32(0), world.body_or_particle, idt],
            device=self.device,
        )
        for _ in range(64):
            wp.launch(
                kernel=_iterate_all_cloth_kernel,
                dim=self.tri_count,
                inputs=[world.constraints, wp.int32(0), world.body_or_particle],
                device=self.device,
            )

        positions_after = world.particles.position.numpy()

        # No NaN / no Inf in any particle position.
        self.assertTrue(np.all(np.isfinite(positions_after)), "non-finite particle position after iterate")

        # Pinned corners haven't moved.
        np.testing.assert_allclose(
            positions_after[pinned_indices],
            pinned_before,
            atol=1e-6,
            err_msg="pinned (inv_mass=0) particle moved during iterate",
        )

        # The free particle was perturbed by 0.5 m; assert that the
        # iterate has restored at least 30% of the perturbation
        # (i.e. the residual displacement is < 0.35 m). Generous
        # bound -- pure XPBD iterate at finite stiffness leaves a
        # residual; we just want to see the iterate is doing real
        # work, not silently no-oping.
        residual = np.linalg.norm(positions_after[free_index] - rest_position)
        self.assertLess(residual, 0.35, f"iterate didn't pull free particle back; residual={residual:.4f}")
        # And the free particle is *somewhere* (not at NaN-collapsed
        # origin). It should be between rest and the perturbed pose.
        self.assertGreater(np.linalg.norm(positions_after[free_index]), 0.0)


if __name__ == "__main__":
    unittest.main()
