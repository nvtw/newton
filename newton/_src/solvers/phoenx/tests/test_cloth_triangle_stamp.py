# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused cloth-triangle stamp kernel.

Validates that
:func:`~newton._src.solvers.phoenx.cloth_collision.triangle_stamp.launch_cloth_triangle_stamp`
fills both the shape-descriptor slots (``shape_type`` /
``shape_transform`` / ``shape_data`` / ``shape_auxiliary``) and the
broadphase AABB slots in a single per-cloth-triangle pass, and
leaves the rigid prefix untouched.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.support_function import GeoTypeEx
from newton._src.solvers.phoenx.cloth_collision.triangle_stamp import launch_cloth_triangle_stamp


@unittest.skipUnless(wp.is_cuda_available(), "Cloth triangle stamp tests require CUDA")
class TestClothTriangleStamp(unittest.TestCase):
    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_axis_aligned_triangle(self) -> None:
        """Axis-aligned triangle: shape descriptors hold (B-A) / (C-A)
        offsets and the AABB matches the per-vertex bounding box
        expanded by ``max(radius) + extra_margin``."""
        pa = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pc = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        particle_q = wp.array(np.array([pa, pb, pc]), dtype=wp.vec3, device=self.device)
        particle_radius = wp.array(np.full(3, 0.05, dtype=np.float32), dtype=wp.float32, device=self.device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)

        n = 1
        shape_type = wp.zeros(n, dtype=wp.int32, device=self.device)
        shape_transform = wp.zeros(n, dtype=wp.transform, device=self.device)
        shape_data = wp.zeros(n, dtype=wp.vec4, device=self.device)
        shape_aux = wp.zeros(n, dtype=wp.vec3, device=self.device)
        aabb_lower = wp.zeros(n, dtype=wp.vec3, device=self.device)
        aabb_upper = wp.zeros(n, dtype=wp.vec3, device=self.device)

        launch_cloth_triangle_stamp(
            particle_q=particle_q,
            particle_radius=particle_radius,
            tri_indices=tri_indices,
            base_offset=0,
            aabb_extra_margin=0.01,
            shape_data_margin=0.005,
            shape_type=shape_type,
            shape_transform=shape_transform,
            shape_data=shape_data,
            shape_auxiliary=shape_aux,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            device=self.device,
        )

        self.assertEqual(int(shape_type.numpy()[0]), int(GeoTypeEx.TRIANGLE))
        xf = shape_transform.numpy()[0]
        np.testing.assert_allclose(xf[:3], pa, atol=1e-6)
        np.testing.assert_allclose(xf[3:], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        sd = shape_data.numpy()[0]
        np.testing.assert_allclose(sd[:3], pb - pa, atol=1e-6)
        self.assertAlmostEqual(float(sd[3]), 0.005, places=6)
        np.testing.assert_allclose(shape_aux.numpy()[0], pc - pa, atol=1e-6)
        # AABB pad = max(radius) + extra_margin = 0.05 + 0.01 = 0.06.
        np.testing.assert_allclose(aabb_lower.numpy()[0], [-0.06, -0.06, -0.06], atol=1e-6)
        np.testing.assert_allclose(aabb_upper.numpy()[0], [1.06, 1.06, 0.06], atol=1e-6)

    def test_rigid_prefix_untouched(self) -> None:
        """Rigid prefix slots (`< base_offset`) are unchanged on all six
        output arrays."""
        pa = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pc = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        particle_q = wp.array(np.array([pa, pb, pc]), dtype=wp.vec3, device=self.device)
        particle_radius = wp.array(np.full(3, 0.05, dtype=np.float32), dtype=wp.float32, device=self.device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)

        S = 3
        sentinel_type = np.array([99, 99, 99, 0], dtype=np.int32)
        sentinel_data = np.array([[1.0, 2.0, 3.0, 4.0]] * S + [[0.0] * 4], dtype=np.float32)
        sentinel_aux = np.array([[5.0, 6.0, 7.0]] * S + [[0.0] * 3], dtype=np.float32)
        sentinel_lo = np.array([[-1.0, -2.0, -3.0]] * S + [[0.0] * 3], dtype=np.float32)
        sentinel_hi = np.array([[1.0, 2.0, 3.0]] * S + [[0.0] * 3], dtype=np.float32)
        sentinel_xf = np.array([[1.0] * 7] * S + [[0.0] * 7], dtype=np.float32)

        shape_type = wp.array(sentinel_type, dtype=wp.int32, device=self.device)
        shape_transform = wp.array(sentinel_xf, dtype=wp.transform, device=self.device)
        shape_data = wp.array(sentinel_data, dtype=wp.vec4, device=self.device)
        shape_aux = wp.array(sentinel_aux, dtype=wp.vec3, device=self.device)
        aabb_lower = wp.array(sentinel_lo, dtype=wp.vec3, device=self.device)
        aabb_upper = wp.array(sentinel_hi, dtype=wp.vec3, device=self.device)

        launch_cloth_triangle_stamp(
            particle_q=particle_q,
            particle_radius=particle_radius,
            tri_indices=tri_indices,
            base_offset=S,
            aabb_extra_margin=0.0,
            shape_data_margin=0.0,
            shape_type=shape_type,
            shape_transform=shape_transform,
            shape_data=shape_data,
            shape_auxiliary=shape_aux,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            device=self.device,
        )

        # Rigid prefix unchanged.
        np.testing.assert_array_equal(shape_type.numpy()[:S], sentinel_type[:S])
        np.testing.assert_allclose(shape_data.numpy()[:S], sentinel_data[:S])
        np.testing.assert_allclose(shape_aux.numpy()[:S], sentinel_aux[:S])
        np.testing.assert_allclose(aabb_lower.numpy()[:S], sentinel_lo[:S])
        np.testing.assert_allclose(aabb_upper.numpy()[:S], sentinel_hi[:S])
        np.testing.assert_allclose(shape_transform.numpy()[:S], sentinel_xf[:S])
        # Triangle slot populated with the right type tag.
        self.assertEqual(int(shape_type.numpy()[S]), int(GeoTypeEx.TRIANGLE))

    def test_zero_triangles_is_noop(self) -> None:
        particle_q = wp.zeros(0, dtype=wp.vec3, device=self.device)
        particle_radius = wp.zeros(0, dtype=wp.float32, device=self.device)
        tri_indices = wp.zeros(0, dtype=wp.vec3i, device=self.device)
        n = 2
        shape_type = wp.zeros(n, dtype=wp.int32, device=self.device)
        shape_transform = wp.zeros(n, dtype=wp.transform, device=self.device)
        shape_data = wp.zeros(n, dtype=wp.vec4, device=self.device)
        shape_aux = wp.zeros(n, dtype=wp.vec3, device=self.device)
        aabb_lower = wp.zeros(n, dtype=wp.vec3, device=self.device)
        aabb_upper = wp.zeros(n, dtype=wp.vec3, device=self.device)
        launch_cloth_triangle_stamp(
            particle_q=particle_q,
            particle_radius=particle_radius,
            tri_indices=tri_indices,
            base_offset=0,
            aabb_extra_margin=0.0,
            shape_data_margin=0.0,
            shape_type=shape_type,
            shape_transform=shape_transform,
            shape_data=shape_data,
            shape_auxiliary=shape_aux,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            device=self.device,
        )
        self.assertTrue(np.all(shape_type.numpy() == 0))
        self.assertTrue(np.all(aabb_lower.numpy() == 0))


if __name__ == "__main__":
    unittest.main()
