# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct unit tests for triangle preconditioning (condition_triangle_for_collision_detection).

Tests call the conditioning function via a Warp kernel and verify that the
output triangle:
1. Lies in the same plane as the input triangle.
2. Has the same face normal direction (same winding).
3. Contains the bounding circle (every point on the circle is inside the output triangle).
4. Is smaller than the input (or unchanged when not needed).
5. Has reasonable aspect ratio (no degenerate triangles).
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.collision_core import condition_triangle_for_collision_detection
from newton._src.geometry.support_function import GenericShapeData, GeoTypeEx

_cuda_available = wp.is_cuda_available()


@wp.kernel(enable_backward=False)
def _call_condition_triangle(
    v0: wp.array[wp.vec3],
    edge_ab: wp.array[wp.vec3],
    edge_ac: wp.array[wp.vec3],
    aabb_lo: wp.array[wp.vec3],
    aabb_hi: wp.array[wp.vec3],
    convex_pos: wp.array[wp.vec3],
    convex_quat: wp.array[wp.quat],
    inflate: wp.array[float],
    out_v0: wp.array[wp.vec3],
    out_edge_ab: wp.array[wp.vec3],
    out_edge_ac: wp.array[wp.vec3],
):
    i = wp.tid()
    tri = GenericShapeData()
    tri.shape_type = int(GeoTypeEx.TRIANGLE)
    tri.scale = edge_ab[i]
    tri.auxiliary = edge_ac[i]

    new_tri, new_v0 = condition_triangle_for_collision_detection(
        tri, v0[i], aabb_lo[i], aabb_hi[i], convex_pos[i], convex_quat[i], inflate[i]
    )

    out_v0[i] = new_v0
    out_edge_ab[i] = new_tri.scale
    out_edge_ac[i] = new_tri.auxiliary


def _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate, convex_quat=None):
    """Run the conditioning function and return (new_va, new_vb, new_vc) as numpy arrays."""
    va, vb, vc = np.array(va, dtype=np.float32), np.array(vb, dtype=np.float32), np.array(vc, dtype=np.float32)
    convex_pos = np.array(convex_pos, dtype=np.float32)
    convex_half = np.array(convex_half, dtype=np.float32)
    if convex_quat is None:
        convex_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        convex_quat = np.array(convex_quat, dtype=np.float32)

    edge_ab = vb - va
    edge_ac = vc - va
    aabb_lo = -convex_half
    aabb_hi = convex_half

    wp_v0 = wp.array([va], dtype=wp.vec3)
    wp_eab = wp.array([edge_ab], dtype=wp.vec3)
    wp_eac = wp.array([edge_ac], dtype=wp.vec3)
    wp_lo = wp.array([aabb_lo], dtype=wp.vec3)
    wp_hi = wp.array([aabb_hi], dtype=wp.vec3)
    wp_pos = wp.array([convex_pos], dtype=wp.vec3)
    wp_quat = wp.array([convex_quat], dtype=wp.quat)
    wp_inf = wp.array([inflate], dtype=float)
    out_v0 = wp.zeros(1, dtype=wp.vec3)
    out_eab = wp.zeros(1, dtype=wp.vec3)
    out_eac = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
        _call_condition_triangle,
        dim=1,
        inputs=[wp_v0, wp_eab, wp_eac, wp_lo, wp_hi, wp_pos, wp_quat, wp_inf, out_v0, out_eab, out_eac],
    )

    new_va = out_v0.numpy()[0]
    new_vb = new_va + out_eab.numpy()[0]
    new_vc = new_va + out_eac.numpy()[0]
    return new_va, new_vb, new_vc


def _triangle_area(va, vb, vc):
    return 0.5 * np.linalg.norm(np.cross(vb - va, vc - va))


def _triangle_normal(va, vb, vc):
    n = np.cross(vb - va, vc - va)
    return n / np.linalg.norm(n)


def _min_angle_deg(va, vb, vc):
    """Return the minimum interior angle of the triangle in degrees."""
    edges = [vb - va, vc - vb, va - vc]
    angles = []
    for i in range(3):
        e1 = -edges[(i - 1) % 3]
        e2 = edges[i]
        cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    return min(angles)


def _point_in_triangle_2d(p, va, vb, vc, normal):
    """Check if point p is inside triangle (va, vb, vc) on the plane with given normal.

    Uses signed edge distances; returns True if all are >= -tolerance.
    """
    tol = -1e-4
    for e0, e1 in [(va, vb), (vb, vc), (vc, va)]:
        edge = e1 - e0
        inward = np.cross(normal, edge)
        inward /= np.linalg.norm(inward)
        if np.dot(inward, p - e0) < tol:
            return False
    return True


def _circle_contained_in_triangle(center, radius, va, vb, vc, normal, n_samples=36):
    """Check that all sampled points on the circle boundary lie inside the triangle."""
    # Build two orthonormal tangent vectors in the plane
    t1 = vb - va
    t1 = t1 - np.dot(t1, normal) * normal
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    for i in range(n_samples):
        angle = 2.0 * np.pi * i / n_samples
        p = center + radius * (np.cos(angle) * t1 + np.sin(angle) * t2)
        if not _point_in_triangle_2d(p, va, vb, vc, normal):
            return False
    return True


class TestConditionTriangleDirect(unittest.TestCase):
    """Direct tests of the condition_triangle_for_collision_detection function."""

    def test_small_triangle_unchanged(self):
        """A triangle already smaller than 2x the bounding sphere is returned unchanged."""
        va = np.array([0.0, 0.0, 0.0])
        vb = np.array([1.0, 0.0, 0.0])
        vc = np.array([0.5, 0.8, 0.0])
        convex_pos = np.array([0.5, 0.3, 0.1])
        convex_half = np.array([0.3, 0.3, 0.3])
        inflate = 0.01

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        np.testing.assert_allclose(new_va, va, atol=1e-5)
        np.testing.assert_allclose(new_vb, vb, atol=1e-5)
        np.testing.assert_allclose(new_vc, vc, atol=1e-5)

    def test_large_triangle_is_shrunk(self):
        """A 1000m triangle with a small shape must produce a much smaller output."""
        s = 500.0
        va = np.array([-s, -s, 0.0])
        vb = np.array([s, -s, 0.0])
        vc = np.array([0.0, s, 0.0])
        convex_pos = np.array([0.0, 0.0, 0.1])
        convex_half = np.array([0.1, 0.1, 0.1])
        inflate = 0.01

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        old_area = _triangle_area(va, vb, vc)
        new_area = _triangle_area(new_va, new_vb, new_vc)

        self.assertGreater(new_area, 0.0, "Output triangle must have positive area")
        self.assertLess(new_area, old_area * 0.01, "Output triangle must be much smaller than input")

    def test_preserves_plane_and_normal(self):
        """Output triangle must lie in the same plane with the same face normal."""
        s = 500.0
        va = np.array([-s, -s, 0.0])
        vb = np.array([s, -s, 0.0])
        vc = np.array([0.0, s, 0.0])
        convex_pos = np.array([10.0, 10.0, 0.05])
        convex_half = np.array([0.2, 0.2, 0.2])
        inflate = 0.01

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        orig_normal = _triangle_normal(va, vb, vc)
        new_normal = _triangle_normal(new_va, new_vb, new_vc)

        # Same normal direction
        self.assertGreater(np.dot(orig_normal, new_normal), 0.99, "Face normal must be preserved")

        # All new vertices on the same plane (z ≈ 0)
        self.assertAlmostEqual(float(new_va[2]), 0.0, delta=1e-4, msg="new_va not on original plane")
        self.assertAlmostEqual(float(new_vb[2]), 0.0, delta=1e-4, msg="new_vb not on original plane")
        self.assertAlmostEqual(float(new_vc[2]), 0.0, delta=1e-4, msg="new_vc not on original plane")

    def test_contains_bounding_circle(self):
        """Output triangle must contain the projected bounding circle."""
        s = 500.0
        va = np.array([-s, -s, 0.0])
        vb = np.array([s, -s, 0.0])
        vc = np.array([0.0, s, 0.0])
        convex_pos = np.array([0.0, 0.0, 0.05])
        convex_half = np.array([0.15, 0.15, 0.15])
        inflate = 0.01

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        normal = _triangle_normal(new_va, new_vb, new_vc)
        sphere_radius = np.linalg.norm(convex_half) + inflate
        # Project sphere center onto plane
        plane_center = convex_pos - np.dot(convex_pos - new_va, normal) * normal
        dist_to_plane = abs(np.dot(convex_pos - new_va, normal))
        circle_radius = np.sqrt(max(sphere_radius**2 - dist_to_plane**2, 0.0))

        self.assertTrue(
            _circle_contained_in_triangle(plane_center, circle_radius, new_va, new_vb, new_vc, normal),
            "Bounding circle must be contained in the conditioned triangle",
        )

    def test_equilateral_for_interior_shape(self):
        """Shape in the interior of a large triangle (0 edges hit) should produce good angles."""
        s = 500.0
        va = np.array([-s, -s, 0.0])
        vb = np.array([s, -s, 0.0])
        vc = np.array([0.0, s, 0.0])
        convex_pos = np.array([0.0, 0.0, 0.05])
        convex_half = np.array([0.1, 0.1, 0.1])
        inflate = 0.01

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        min_angle = _min_angle_deg(new_va, new_vb, new_vc)
        # Equilateral has 60° min angle; allow some tolerance
        self.assertGreater(min_angle, 45.0, f"Interior case should produce good angles, got min={min_angle:.1f}°")

    def test_skinny_triangle_off_center(self):
        """Shape at the center of a long narrow strip must still get valid output.

        The strip is wide enough (2m) to contain the bounding circle (~0.36m diameter).
        """
        va = np.array([-1.0, -500.0, 0.0])
        vb = np.array([1.0, -500.0, 0.0])
        vc = np.array([0.0, 500.0, 0.0])
        convex_pos = np.array([0.0, 0.0, 0.05])
        convex_half = np.array([0.1, 0.1, 0.1])
        inflate = 0.01

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        new_area = _triangle_area(new_va, new_vb, new_vc)
        self.assertGreater(new_area, 0.0, "Skinny triangle output must have positive area")

        # Must contain the bounding circle
        normal = _triangle_normal(new_va, new_vb, new_vc)
        sphere_radius = np.linalg.norm(convex_half) + inflate
        plane_center = convex_pos - np.dot(convex_pos - new_va, normal) * normal
        dist_to_plane = abs(np.dot(convex_pos - new_va, normal))
        circle_radius = np.sqrt(max(sphere_radius**2 - dist_to_plane**2, 0.0))

        self.assertTrue(
            _circle_contained_in_triangle(plane_center, circle_radius, new_va, new_vb, new_vc, normal),
            "Bounding circle must be contained in the conditioned skinny triangle",
        )

    def test_off_center_shape_on_large_triangle(self):
        """Shape far from origin on a large triangle must produce valid conditioned output."""
        s = 500.0
        va = np.array([-s, -s, 0.0])
        vb = np.array([s, -s, 0.0])
        vc = np.array([0.0, s, 0.0])
        convex_pos = np.array([200.0, -100.0, 0.05])
        convex_half = np.array([0.15, 0.15, 0.15])
        inflate = 0.01

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        old_area = _triangle_area(va, vb, vc)
        new_area = _triangle_area(new_va, new_vb, new_vc)
        self.assertGreater(new_area, 0.0)
        self.assertLess(new_area, old_area * 0.01, "Off-center output must be much smaller")

        # Check bounding circle containment
        normal = _triangle_normal(new_va, new_vb, new_vc)
        sphere_radius = np.linalg.norm(convex_half) + inflate
        plane_center = convex_pos - np.dot(convex_pos - new_va, normal) * normal
        dist_to_plane = abs(np.dot(convex_pos - new_va, normal))
        circle_radius = np.sqrt(max(sphere_radius**2 - dist_to_plane**2, 0.0))

        self.assertTrue(
            _circle_contained_in_triangle(plane_center, circle_radius, new_va, new_vb, new_vc, normal),
            "Bounding circle must be inside off-center conditioned triangle",
        )

    def test_three_edges_hit_returns_original(self):
        """When all 3 edges intersect the bounding circle, return the original triangle."""
        # Small triangle with a relatively large bounding sphere
        va = np.array([0.0, 0.0, 0.0])
        vb = np.array([2.0, 0.0, 0.0])
        vc = np.array([1.0, 1.5, 0.0])
        convex_pos = np.array([1.0, 0.5, 0.05])
        convex_half = np.array([0.6, 0.6, 0.6])
        inflate = 0.1

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate)

        np.testing.assert_allclose(new_va, va, atol=1e-5, err_msg="3-edge hit should return original va")
        np.testing.assert_allclose(new_vb, vb, atol=1e-5, err_msg="3-edge hit should return original vb")
        np.testing.assert_allclose(new_vc, vc, atol=1e-5, err_msg="3-edge hit should return original vc")

    def test_rotated_convex(self):
        """Conditioning with a rotated convex AABB must still produce valid output."""
        s = 100.0
        va = np.array([-s, -s, 0.0])
        vb = np.array([s, -s, 0.0])
        vc = np.array([0.0, s, 0.0])
        convex_pos = np.array([0.0, 0.0, 0.1])
        convex_half = np.array([0.3, 0.1, 0.2])  # non-uniform half-extents
        inflate = 0.01

        # 45 degree rotation around Z
        angle = np.pi / 4.0
        c, sn = np.cos(angle / 2), np.sin(angle / 2)
        convex_quat = np.array([0.0, 0.0, sn, c], dtype=np.float32)

        new_va, new_vb, new_vc = _run_conditioning(va, vb, vc, convex_pos, convex_half, inflate, convex_quat)

        new_area = _triangle_area(new_va, new_vb, new_vc)
        self.assertGreater(new_area, 0.0, "Rotated convex output must have positive area")

        normal = _triangle_normal(new_va, new_vb, new_vc)
        # Bounding sphere radius from local half-diagonal
        sphere_radius = np.linalg.norm(convex_half) + inflate
        plane_center = convex_pos - np.dot(convex_pos - new_va, normal) * normal
        dist_to_plane = abs(np.dot(convex_pos - new_va, normal))
        circle_radius = np.sqrt(max(sphere_radius**2 - dist_to_plane**2, 0.0))

        self.assertTrue(
            _circle_contained_in_triangle(plane_center, circle_radius, new_va, new_vb, new_vc, normal),
            "Bounding circle must be inside triangle for rotated convex",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
