# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the first-class :data:`newton.GeoType.TRIANGLE` primitive.

The triangle primitive is a single-triangle collider built from three
explicit vertices. Internally the points are rebased to a canonical
local frame:

    A_local = (0, 0, 0)
    B_local = (0, 0, edge_ab)
    C_local = (0, c_y, c_z)

with the three free parameters packed into ``shape_scale = (edge_ab, c_y, c_z)``
and the rigid offset folded into the shape's transform. The face normal
follows the right-hand rule on ``(B - A) x (C - A)``.

These tests cover:

1. ``ModelBuilder.add_shape_triangle`` parameter validation and the
   point-based -> canonical-frame mapping (round-trip through finalize
   reconstructs the requested vertices).
2. End-to-end narrow-phase collision via the shared GJK/MPR path
   (sphere vs. triangle, triangle vs. triangle).
3. Inertia / hydroelastic exclusions.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType

_cuda_available = wp.is_cuda_available()


def _build_sphere_vs_triangle(
    sphere_pos,
    sphere_radius=0.2,
    point_a=(0.0, 0.0, 0.0),
    point_b=(0.0, 0.0, 1.0),
    point_c=(0.0, 1.0, 0.0),
):
    """Build a scene with a static triangle (body=-1) and a sphere body."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    builder.add_shape_triangle(
        body=-1,
        point_a=wp.vec3(*point_a),
        point_b=wp.vec3(*point_b),
        point_c=wp.vec3(*point_c),
    )

    body = builder.add_body(xform=wp.transform(wp.vec3(*sphere_pos), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=sphere_radius)

    model = builder.finalize()
    cp = newton.CollisionPipeline(model, broad_phase="explicit")
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, cp, state


def _collide(model, cp, state):
    return model.collide(state, collision_pipeline=cp)


def _world_vertices_from_shape(builder: newton.ModelBuilder, idx: int) -> np.ndarray:
    """Reconstruct the three world-space vertices stored for shape ``idx``."""
    scale = np.array(tuple(builder.shape_scale[idx]), dtype=np.float64)
    edge_ab = float(scale[0])
    c_y = float(scale[1])
    c_z = float(scale[2])

    xform = builder.shape_transform[idx]
    p = np.array((float(xform.p[0]), float(xform.p[1]), float(xform.p[2])), dtype=np.float64)
    q = np.array(
        (float(xform.q[0]), float(xform.q[1]), float(xform.q[2]), float(xform.q[3])),
        dtype=np.float64,
    )

    def rotate(v: np.ndarray) -> np.ndarray:
        # Right-handed quaternion rotation: v' = q * v * q^-1.
        x, y, z, w = q
        t = 2.0 * np.cross(np.array((x, y, z)), v)
        return v + w * t + np.cross(np.array((x, y, z)), t)

    a_local = np.array((0.0, 0.0, 0.0))
    b_local = np.array((0.0, 0.0, edge_ab))
    c_local = np.array((0.0, c_y, c_z))
    return np.stack((p + rotate(a_local), p + rotate(b_local), p + rotate(c_local)))


class TestTriangleBuilder(unittest.TestCase):
    """``add_shape_triangle`` parameter handling and storage."""

    def test_axis_aligned_triangle_round_trips_through_finalize(self):
        builder = newton.ModelBuilder()
        idx = builder.add_shape_triangle(
            body=-1,
            point_a=wp.vec3(0.0, 0.0, 0.0),
            point_b=wp.vec3(2.5, 0.0, 0.0),
            point_c=wp.vec3(0.0, 1.5, 0.0),
        )
        self.assertEqual(idx, 0)
        self.assertEqual(builder.shape_type[idx], GeoType.TRIANGLE)

        # ``edge_ab`` = |B - A| = 2.5; the local frame rebases AB onto
        # +Z so |AC| projects entirely onto local +Y -> c_y = 1.5,
        # c_z = 0.
        scale = builder.shape_scale[idx]
        self.assertAlmostEqual(scale[0], 2.5, places=5)
        self.assertAlmostEqual(scale[1], 1.5, places=5)
        self.assertAlmostEqual(scale[2], 0.0, places=5)

        # Round-trip: reconstructed world vertices must match the
        # requested ones (modulo float rounding through the canonical
        # frame change).
        verts = _world_vertices_from_shape(builder, idx)
        np.testing.assert_allclose(verts[0], (0.0, 0.0, 0.0), atol=1e-5)
        np.testing.assert_allclose(verts[1], (2.5, 0.0, 0.0), atol=1e-5)
        np.testing.assert_allclose(verts[2], (0.0, 1.5, 0.0), atol=1e-5)

    def test_offset_and_oriented_triangle_round_trips(self):
        # Triangle in a generic pose: not at the origin, not axis-aligned.
        a = (1.0, -2.0, 3.0)
        b = (1.5, -1.0, 3.5)
        c = (0.5, -1.5, 2.5)
        builder = newton.ModelBuilder()
        idx = builder.add_shape_triangle(
            body=-1,
            point_a=wp.vec3(*a),
            point_b=wp.vec3(*b),
            point_c=wp.vec3(*c),
        )
        model = builder.finalize()
        scale = model.shape_scale.numpy()[idx]
        # |AB| = sqrt(0.5^2 + 1.0^2 + 0.5^2) = sqrt(1.5)
        self.assertAlmostEqual(float(scale[0]), float(np.sqrt(1.5)), places=4)

        # World-space round trip survives the canonicalisation.
        verts = _world_vertices_from_shape(builder, idx)
        np.testing.assert_allclose(verts[0], a, atol=1e-4)
        np.testing.assert_allclose(verts[1], b, atol=1e-4)
        np.testing.assert_allclose(verts[2], c, atol=1e-4)

    def test_degenerate_a_equals_b_raises(self):
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_triangle(
                body=-1,
                point_a=wp.vec3(0.0, 0.0, 0.0),
                point_b=wp.vec3(0.0, 0.0, 0.0),
                point_c=wp.vec3(0.0, 1.0, 0.0),
            )

    def test_degenerate_collinear_triangle_raises(self):
        # A, B, C all on the +X axis.
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_triangle(
                body=-1,
                point_a=wp.vec3(0.0, 0.0, 0.0),
                point_b=wp.vec3(1.0, 0.0, 0.0),
                point_c=wp.vec3(2.0, 0.0, 0.0),
            )

    def test_geo_type_is_classified_as_primitive(self):
        self.assertTrue(GeoType.TRIANGLE.is_primitive)
        self.assertFalse(GeoType.TRIANGLE.is_explicit)


class TestTriangleInertia(unittest.TestCase):
    """Triangles are modelled as thin prisms of thickness 2 * margin."""

    def test_zero_margin_yields_zero_mass(self):
        from newton._src.geometry.inertia import compute_inertia_shape

        # Without thickness the prism collapses to a zero-volume shell:
        # zero mass and zero rotational inertia, but the COM is still the
        # triangle centroid so external mass / inertia overrides resolve
        # at the geometric center, not at vertex A.
        scale = wp.vec3(1.0, 1.0, 0.0)
        m, com, inertia = compute_inertia_shape(GeoType.TRIANGLE, scale, None, density=1000.0, thickness=0.0)
        self.assertEqual(m, 0.0)
        # Centroid of A=(0,0,0), B=(0,0,1), C=(0,1,0) is (0, 1/3, 1/3).
        np.testing.assert_allclose(np.array(com), (0.0, 1.0 / 3.0, 1.0 / 3.0), atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(np.array(inertia).flatten()), 0.0)

    def test_prism_mass_matches_volume_times_density(self):
        from newton._src.geometry.inertia import compute_inertia_shape

        # Right triangle with legs L and c_y has area 0.5 * L * c_y.
        # With margin=t/2, prism volume = area * t = area * 2 * margin.
        L = 2.0
        c_y = 3.0
        margin = 0.05
        density = 1000.0
        scale = wp.vec3(L, c_y, 0.0)
        m, _com, _I = compute_inertia_shape(GeoType.TRIANGLE, scale, None, density=density, thickness=margin)
        expected_mass = density * (0.5 * L * c_y) * (2.0 * margin)
        self.assertAlmostEqual(m, expected_mass, places=4)

    def test_prism_inertia_recovers_known_thin_plate_limits(self):
        from newton._src.geometry.inertia import compute_inertia_shape

        # In the thin-plate limit (t² << triangle extent²) the in-plane
        # inertia I_xx must dominate I_yy and I_zz, because the prism
        # extent along x (the normal direction) shrinks to zero.
        L = 1.0
        c_y = 1.0
        margin = 1e-4  # very thin
        density = 1000.0
        scale = wp.vec3(L, c_y, 0.0)
        m, _com, inertia = compute_inertia_shape(GeoType.TRIANGLE, scale, None, density=density, thickness=margin)
        I = np.array(inertia).reshape(3, 3)
        # Symmetry: tensor must be symmetric.
        np.testing.assert_allclose(I, I.T, atol=1e-12)
        # Off-diagonals: the triangle is in the YZ plane and is symmetric
        # in x, so I_xy = I_xz = 0; I_yz can be non-zero (general
        # triangle), but for this right triangle with C on the +Y axis
        # and centroid at (0, 1/3, 1/3) it should still be non-trivial,
        # we just check I_xy = I_xz = 0 here.
        self.assertAlmostEqual(I[0, 1], 0.0)
        self.assertAlmostEqual(I[0, 2], 0.0)
        # Thin-plate dominant axis: I_xx > I_yy and I_xx > I_zz when
        # thickness is much smaller than the planar extent.
        self.assertGreater(I[0, 0], I[1, 1])
        self.assertGreater(I[0, 0], I[2, 2])
        # Non-degenerate mass.
        self.assertGreater(m, 0.0)

    def test_compute_shape_radius(self):
        from newton._src.geometry.utils import compute_shape_radius

        # Bounding sphere should enclose B and C around the origin (vertex A).
        r = compute_shape_radius(GeoType.TRIANGLE, (3.0, 4.0, 0.0), None)
        # |AB| = 3, |AC| = sqrt(4^2 + 0^2) = 4 -> max = 4.
        self.assertAlmostEqual(r, 4.0)


class TestTriangleHydroelasticValidation(unittest.TestCase):
    """Hydroelastic contact is intentionally not supported for triangles."""

    def test_hydroelastic_triangle_raises_or_silently_ignored(self):
        # Mirrors the PLANE/HFIELD policy: ShapeConfig.validate skips the
        # "hydroelastic requires SDF" branch for shape types that don't
        # support hydroelastic contact, so building a hydroelastic triangle
        # should not crash and should not request an SDF.
        cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True)
        # Should not raise even though no SDF resolution is configured.
        cfg.validate(shape_type=GeoType.TRIANGLE)

        builder = newton.ModelBuilder()
        idx = builder.add_shape_triangle(
            body=-1,
            point_a=wp.vec3(0.0, 0.0, 0.0),
            point_b=wp.vec3(1.0, 0.0, 0.0),
            point_c=wp.vec3(0.0, 1.0, 0.0),
            cfg=cfg,
        )
        self.assertEqual(builder.shape_type[idx], GeoType.TRIANGLE)


@unittest.skipUnless(_cuda_available, "Triangle collision pipeline requires a CUDA device")
class TestTriangleNarrowPhase(unittest.TestCase):
    """End-to-end collision through the shared GJK/MPR primitive path.

    These tests exercise the same code path used by the PhoenX rigid-body
    solver (which delegates to ``newton.CollisionPipeline``), so passing
    here also validates PhoenX integration.
    """

    def test_sphere_above_triangle_face_generates_contact(self):
        # Triangle in the world XZ plane (face normal +Y). Sphere placed
        # just above it on the +Y side overlaps the face.
        model, cp, state = _build_sphere_vs_triangle(
            sphere_pos=(0.3, 0.05, 0.3),
            sphere_radius=0.2,
            point_a=(0.0, 0.0, 0.0),
            point_b=(0.0, 0.0, 1.0),
            point_c=(1.0, 0.0, 0.0),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Sphere overlapping triangle face should produce a contact")

    def test_sphere_far_from_triangle_no_contact(self):
        model, cp, state = _build_sphere_vs_triangle(
            sphere_pos=(5.0, 5.0, 5.0),
            sphere_radius=0.1,
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 0, "Sphere far from triangle should produce no contacts")

    def test_triangle_is_double_sided(self):
        # Sphere on either side of the triangle should collide; the
        # canonical triangle below has its normal along world +Y, so we
        # test one sphere at +Y and one at -Y.
        model_pos, cp_pos, state_pos = _build_sphere_vs_triangle(
            sphere_pos=(0.3, 0.05, 0.3),
            sphere_radius=0.2,
            point_a=(0.0, 0.0, 0.0),
            point_b=(0.0, 0.0, 1.0),
            point_c=(1.0, 0.0, 0.0),
        )
        model_neg, cp_neg, state_neg = _build_sphere_vs_triangle(
            sphere_pos=(0.3, -0.05, 0.3),
            sphere_radius=0.2,
            point_a=(0.0, 0.0, 0.0),
            point_b=(0.0, 0.0, 1.0),
            point_c=(1.0, 0.0, 0.0),
        )
        n_pos = int(_collide(model_pos, cp_pos, state_pos).rigid_contact_count.numpy()[0])
        n_neg = int(_collide(model_neg, cp_neg, state_neg).rigid_contact_count.numpy()[0])
        self.assertGreater(n_pos, 0)
        self.assertGreater(n_neg, 0, "Triangle should be double-sided (both sides collide)")

    def test_world_offset_triangle_collides(self):
        # Translate the triangle far from the origin and put the sphere next to it.
        offset = (10.0, 20.0, 30.0)
        model, cp, state = _build_sphere_vs_triangle(
            sphere_pos=(offset[0] + 0.3, offset[1] + 0.05, offset[2] + 0.3),
            sphere_radius=0.2,
            point_a=(offset[0] + 0.0, offset[1] + 0.0, offset[2] + 0.0),
            point_b=(offset[0] + 0.0, offset[1] + 0.0, offset[2] + 1.0),
            point_c=(offset[0] + 1.0, offset[1] + 0.0, offset[2] + 0.0),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Translated triangle should still collide with overlapping sphere")


@unittest.skipUnless(_cuda_available, "Triangle collision pipeline requires a CUDA device")
class TestTriangleVsTriangle(unittest.TestCase):
    """Two TRIANGLE primitives should also route through GJK/MPR."""

    def test_intersecting_triangles_collide(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        # First triangle in the YZ plane (normal along +X).
        builder.add_shape_triangle(
            body=-1,
            point_a=wp.vec3(0.0, 0.0, 0.0),
            point_b=wp.vec3(0.0, 0.0, 1.0),
            point_c=wp.vec3(0.0, 1.0, 0.0),
        )

        # Second triangle in the XZ plane (normal along +Y), offset by
        # 0.3 m in Y so the two triangles intersect along Z.
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.3, 0.0), wp.quat_identity()))
        builder.add_shape_triangle(
            body=body_b,
            point_a=wp.vec3(0.0, 0.0, 0.0),
            point_b=wp.vec3(0.0, 0.0, 1.0),
            point_c=wp.vec3(1.0, 0.0, 0.0),
        )

        model = builder.finalize()
        cp = newton.CollisionPipeline(model, broad_phase="explicit")
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        contacts = model.collide(state, collision_pipeline=cp)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Two intersecting triangle primitives should produce a contact")


if __name__ == "__main__":
    unittest.main()
