# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cloth-triangle shape-data stamping kernel.

Verifies that
:func:`~newton._src.solvers.phoenx.cloth_collision.triangle_shape_data.launch_cloth_triangle_shape_data`
fills the right slice of the concatenated shape arrays with
``GeoTypeEx.TRIANGLE`` data sourced from particle positions, and
leaves the rigid prefix untouched.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.support_function import GeoTypeEx
from newton._src.solvers.phoenx.cloth_collision.triangle_shape_data import (
    launch_cloth_triangle_shape_data,
)


@unittest.skipUnless(wp.is_cuda_available(), "Cloth triangle shape-data tests require CUDA")
class TestClothTriangleShapeData(unittest.TestCase):
    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_axis_aligned_triangle(self) -> None:
        """An axis-aligned triangle gets TRIANGLE shape type, vertex A
        as the world transform origin, and (B-A, C-A) in shape_data /
        shape_auxiliary."""
        pa = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pc = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        particle_q = wp.array(np.array([pa, pb, pc]), dtype=wp.vec3, device=self.device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)

        S = 0
        n = S + 1  # one cloth triangle
        shape_type = wp.zeros(n, dtype=wp.int32, device=self.device)
        shape_transform = wp.zeros(n, dtype=wp.transform, device=self.device)
        shape_data = wp.zeros(n, dtype=wp.vec4, device=self.device)
        shape_aux = wp.zeros(n, dtype=wp.vec3, device=self.device)

        launch_cloth_triangle_shape_data(
            particle_q=particle_q,
            tri_indices=tri_indices,
            base_offset=S,
            margin=0.01,
            shape_type=shape_type,
            shape_transform=shape_transform,
            shape_data=shape_data,
            shape_auxiliary=shape_aux,
            device=self.device,
        )
        st = shape_type.numpy()
        xf = shape_transform.numpy()
        sd = shape_data.numpy()
        aux = shape_aux.numpy()

        self.assertEqual(int(st[0]), int(GeoTypeEx.TRIANGLE))
        # transform = (translation xyz, quat xyzw)
        np.testing.assert_allclose(xf[0][:3], pa, atol=1e-6)
        np.testing.assert_allclose(xf[0][3:], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(sd[0][:3], pb - pa, atol=1e-6)
        self.assertAlmostEqual(float(sd[0][3]), 0.01, places=6)
        np.testing.assert_allclose(aux[0], pc - pa, atol=1e-6)

    def test_rigid_prefix_untouched(self) -> None:
        """Rigid shape slots (`< base_offset`) must be unchanged."""
        pa = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pc = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        particle_q = wp.array(np.array([pa, pb, pc]), dtype=wp.vec3, device=self.device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.vec3i, device=self.device)

        S = 3
        # Pre-fill rigid prefix with sentinel values.
        rigid_type = np.array([99, 99, 99, 0], dtype=np.int32)
        rigid_data = np.array([[1.0, 2.0, 3.0, 4.0]] * S + [[0.0] * 4], dtype=np.float32)
        rigid_aux = np.array([[5.0, 6.0, 7.0]] * S + [[0.0] * 3], dtype=np.float32)
        rigid_xf = np.array([[1.0] * 7] * S + [[0.0] * 7], dtype=np.float32)

        shape_type = wp.array(rigid_type, dtype=wp.int32, device=self.device)
        shape_transform = wp.array(rigid_xf, dtype=wp.transform, device=self.device)
        shape_data = wp.array(rigid_data, dtype=wp.vec4, device=self.device)
        shape_aux = wp.array(rigid_aux, dtype=wp.vec3, device=self.device)

        launch_cloth_triangle_shape_data(
            particle_q=particle_q,
            tri_indices=tri_indices,
            base_offset=S,
            margin=0.0,
            shape_type=shape_type,
            shape_transform=shape_transform,
            shape_data=shape_data,
            shape_auxiliary=shape_aux,
            device=self.device,
        )

        # Rigid prefix unchanged.
        np.testing.assert_array_equal(shape_type.numpy()[:S], rigid_type[:S])
        np.testing.assert_allclose(shape_data.numpy()[:S], rigid_data[:S])
        np.testing.assert_allclose(shape_aux.numpy()[:S], rigid_aux[:S])
        np.testing.assert_allclose(shape_transform.numpy()[:S], rigid_xf[:S])
        # Triangle slot populated.
        self.assertEqual(int(shape_type.numpy()[S]), int(GeoTypeEx.TRIANGLE))

    def test_zero_triangles_is_noop(self) -> None:
        particle_q = wp.zeros(0, dtype=wp.vec3, device=self.device)
        tri_indices = wp.zeros(0, dtype=wp.vec3i, device=self.device)
        shape_type = wp.zeros(2, dtype=wp.int32, device=self.device)
        shape_transform = wp.zeros(2, dtype=wp.transform, device=self.device)
        shape_data = wp.zeros(2, dtype=wp.vec4, device=self.device)
        shape_aux = wp.zeros(2, dtype=wp.vec3, device=self.device)
        launch_cloth_triangle_shape_data(
            particle_q=particle_q,
            tri_indices=tri_indices,
            base_offset=0,
            margin=0.0,
            shape_type=shape_type,
            shape_transform=shape_transform,
            shape_data=shape_data,
            shape_auxiliary=shape_aux,
            device=self.device,
        )
        self.assertTrue(np.all(shape_type.numpy() == 0))


if __name__ == "__main__":
    unittest.main()
