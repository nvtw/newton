# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cloth-triangle AABB build kernel.

Verifies that
:func:`~newton._src.solvers.phoenx.cloth_collision.triangle_aabb.launch_cloth_triangle_aabbs`
fills the right slice of the concatenated AABB array, applies the
correct margin (max-of-three vertex radii + extra), and leaves the
rigid prefix untouched.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.cloth_collision.triangle_aabb import launch_cloth_triangle_aabbs


@unittest.skipUnless(wp.is_cuda_available(), "Cloth triangle AABB tests require CUDA")
class TestClothTriangleAABBs(unittest.TestCase):
    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_axis_aligned_triangle(self) -> None:
        """A triangle with vertices at (0,0,0), (1,0,0), (0,1,0).
        With per-vertex radius 0.05 and extra_margin 0.01, the AABB
        should be (-0.06, -0.06, -0.06) to (1.06, 1.06, 0.06)."""
        particle_q = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        particle_radius = wp.array(np.full(3, 0.05, dtype=np.float32), dtype=wp.float32, device=self.device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)

        S = 0
        T = 1
        aabb_lower = wp.zeros(S + T, dtype=wp.vec3, device=self.device)
        aabb_upper = wp.zeros(S + T, dtype=wp.vec3, device=self.device)

        launch_cloth_triangle_aabbs(
            particle_q=particle_q,
            particle_radius=particle_radius,
            tri_indices=tri_indices,
            base_offset=S,
            extra_margin=0.01,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            device=self.device,
        )

        lower = aabb_lower.numpy()[0]
        upper = aabb_upper.numpy()[0]
        np.testing.assert_allclose(lower, [-0.06, -0.06, -0.06], atol=1e-6)
        np.testing.assert_allclose(upper, [1.06, 1.06, 0.06], atol=1e-6)

    def test_max_of_three_radii(self) -> None:
        """When the three vertex radii differ, the per-axis pad uses
        the max."""
        particle_q = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        # Vertex 1 has the largest radius -> pad = 0.2
        particle_radius = wp.array(np.array([0.05, 0.20, 0.10], dtype=np.float32), dtype=wp.float32, device=self.device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)

        aabb_lower = wp.zeros(1, dtype=wp.vec3, device=self.device)
        aabb_upper = wp.zeros(1, dtype=wp.vec3, device=self.device)

        launch_cloth_triangle_aabbs(
            particle_q=particle_q,
            particle_radius=particle_radius,
            tri_indices=tri_indices,
            base_offset=0,
            extra_margin=0.0,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            device=self.device,
        )
        lower = aabb_lower.numpy()[0]
        upper = aabb_upper.numpy()[0]
        np.testing.assert_allclose(lower, [-0.20, -0.20, -0.20], atol=1e-6)
        np.testing.assert_allclose(upper, [1.20, 1.20, 0.20], atol=1e-6)

    def test_rigid_prefix_untouched(self) -> None:
        """When ``base_offset > 0``, the kernel must not overwrite the
        rigid AABB slots."""
        particle_q = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        particle_radius = wp.array(np.full(3, 0.05, dtype=np.float32), dtype=wp.float32, device=self.device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)

        S = 4
        T = 1
        # Pre-fill rigid prefix with a sentinel pattern.
        rigid_lo = np.full((S, 3), -123.0, dtype=np.float32)
        rigid_hi = np.full((S, 3), 456.0, dtype=np.float32)
        aabb_lower_np = np.concatenate([rigid_lo, np.zeros((T, 3), dtype=np.float32)], axis=0)
        aabb_upper_np = np.concatenate([rigid_hi, np.zeros((T, 3), dtype=np.float32)], axis=0)
        aabb_lower = wp.array(aabb_lower_np, dtype=wp.vec3, device=self.device)
        aabb_upper = wp.array(aabb_upper_np, dtype=wp.vec3, device=self.device)

        launch_cloth_triangle_aabbs(
            particle_q=particle_q,
            particle_radius=particle_radius,
            tri_indices=tri_indices,
            base_offset=S,
            extra_margin=0.0,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            device=self.device,
        )

        out_lower = aabb_lower.numpy()
        out_upper = aabb_upper.numpy()
        # Rigid slots untouched.
        np.testing.assert_allclose(out_lower[:S], rigid_lo)
        np.testing.assert_allclose(out_upper[:S], rigid_hi)
        # Triangle slot populated.
        np.testing.assert_allclose(out_lower[S], [-0.05, -0.05, -0.05], atol=1e-6)
        np.testing.assert_allclose(out_upper[S], [1.05, 1.05, 0.05], atol=1e-6)

    def test_zero_triangles_is_noop(self) -> None:
        """Empty triangle table -> kernel never launches and rigid prefix
        is unchanged."""
        particle_q = wp.zeros(0, dtype=wp.vec3, device=self.device)
        particle_radius = wp.zeros(0, dtype=wp.float32, device=self.device)
        tri_indices = wp.zeros(0, dtype=wp.vec3i, device=self.device)
        aabb_lower = wp.zeros(2, dtype=wp.vec3, device=self.device)
        aabb_upper = wp.zeros(2, dtype=wp.vec3, device=self.device)
        launch_cloth_triangle_aabbs(
            particle_q=particle_q,
            particle_radius=particle_radius,
            tri_indices=tri_indices,
            base_offset=0,
            extra_margin=0.0,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            device=self.device,
        )
        # Should remain zeroed -- kernel was a no-op.
        self.assertTrue(np.all(aabb_lower.numpy() == 0.0))
        self.assertTrue(np.all(aabb_upper.numpy() == 0.0))


if __name__ == "__main__":
    unittest.main()
