# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused cloth-triangle stamp kernel.

Validates that
:func:`~newton._src.solvers.phoenx.cloth_collision.triangle_stamp.launch_cloth_triangle_stamp`
fills both the shape-descriptor slots (``shape_type`` /
``shape_transform`` / ``shape_data`` / ``shape_auxiliary``) and the
broadphase AABB slots in a single per-cloth-triangle pass, reading
triangle topology from the cloth-triangle rows of a phoenx
:class:`ConstraintContainer`.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.support_function import GeoTypeEx
from newton._src.solvers.phoenx.cloth_collision.triangle_stamp import launch_cloth_triangle_stamp
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_triangle_set_body1,
    cloth_triangle_set_body2,
    cloth_triangle_set_body3,
    cloth_triangle_set_type,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@wp.kernel(enable_backward=False)
def _stamp_cloth_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    indices: wp.array[wp.int32],
):
    """Test-only kernel that writes one cloth-triangle row per thread,
    stamping the type tag + the three particle-index slots."""
    t = wp.tid()
    cid = cid_offset + t
    cloth_triangle_set_type(constraints, cid)
    cloth_triangle_set_body1(constraints, cid, indices[t * 3 + 0])
    cloth_triangle_set_body2(constraints, cid, indices[t * 3 + 1])
    cloth_triangle_set_body3(constraints, cid, indices[t * 3 + 2])


def _make_cloth_constraints(rows_unified: list[tuple[int, int, int]], device):
    """Build a phoenx ConstraintContainer with the given cloth-triangle
    rows starting at cid 0. ``rows_unified`` carries the *unified*
    body-or-particle indices (i.e. ``num_bodies + particle_idx``)."""
    T = len(rows_unified)
    num_dwords = PhoenXWorld.required_constraint_dwords(num_joints=0, num_cloth_triangles=T)
    constraints = constraint_container_zeros(num_constraints=max(1, T), num_dwords=num_dwords, device=device)
    flat = np.array(rows_unified, dtype=np.int32).reshape(-1)
    indices = wp.array(flat, dtype=wp.int32, device=device)
    wp.launch(
        _stamp_cloth_rows_kernel,
        dim=T,
        inputs=[constraints, wp.int32(0), indices],
        device=device,
    )
    return constraints


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
        # Single cloth triangle whose three particles are at unified
        # indices num_bodies + (0, 1, 2). Use num_bodies = 0 for
        # the test so unified indices == raw particle indices.
        constraints = _make_cloth_constraints([(0, 1, 2)], self.device)

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
            constraints=constraints,
            cloth_cid_offset=0,
            num_bodies=0,
            num_cloth_triangles=1,
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
        np.testing.assert_allclose(aabb_lower.numpy()[0], [-0.06, -0.06, -0.06], atol=1e-6)
        np.testing.assert_allclose(aabb_upper.numpy()[0], [1.06, 1.06, 0.06], atol=1e-6)

    def test_zero_triangles_is_noop(self) -> None:
        """Empty triangle table: stamp launcher returns without launching."""
        particle_q = wp.zeros(0, dtype=wp.vec3, device=self.device)
        particle_radius = wp.zeros(0, dtype=wp.float32, device=self.device)
        constraints = constraint_container_zeros(num_constraints=1, num_dwords=1, device=self.device)
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
            constraints=constraints,
            cloth_cid_offset=0,
            num_bodies=0,
            num_cloth_triangles=0,
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
