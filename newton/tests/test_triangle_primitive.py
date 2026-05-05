# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the first-class :data:`newton.GeoType.TRIANGLE` primitive.

The triangle primitive is a single-triangle collider with the canonical
local frame:

    A = (0, 0, 0)
    B = (0, 0, edge_ab)
    C = (0, c_y, c_z)

with the three free parameters packed into ``shape_scale = (edge_ab, c_y, c_z)``.

These tests cover:

1. ``ModelBuilder.add_shape_triangle`` parameter validation.
2. ``shape_scale`` encoding and bookkeeping after ``finalize``.
3. End-to-end narrow-phase collision via the shared GJK/MPR path
   (sphere vs. triangle, capsule vs. triangle).
4. Inertia / hydroelastic exclusions.
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
    edge_ab=1.0,
    point_c=(1.0, 0.0),
    triangle_xform=None,
):
    """Build a scene with a static triangle (body=-1) and a sphere body."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    if triangle_xform is None:
        triangle_xform = wp.transform_identity()

    builder.add_shape_triangle(
        body=-1,
        xform=triangle_xform,
        edge_ab=edge_ab,
        point_c=point_c,
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


class TestTriangleBuilder(unittest.TestCase):
    """``add_shape_triangle`` parameter handling and storage."""

    def test_default_parameters_create_unit_right_triangle(self):
        builder = newton.ModelBuilder()
        idx = builder.add_shape_triangle(body=-1)
        self.assertEqual(builder.shape_type[idx], GeoType.TRIANGLE)
        scale = builder.shape_scale[idx]
        self.assertAlmostEqual(scale[0], 1.0)
        self.assertAlmostEqual(scale[1], 1.0)
        self.assertAlmostEqual(scale[2], 0.0)

    def test_custom_parameters_round_trip_through_finalize(self):
        builder = newton.ModelBuilder()
        idx = builder.add_shape_triangle(
            body=-1,
            edge_ab=2.5,
            point_c=(1.5, -0.75),
        )
        self.assertEqual(idx, 0)
        model = builder.finalize()
        scale = model.shape_scale.numpy()[idx]
        np.testing.assert_allclose(scale, (2.5, 1.5, -0.75), atol=1e-6)
        shape_type = int(model.shape_type.numpy()[idx])
        self.assertEqual(shape_type, int(GeoType.TRIANGLE))

    def test_negative_edge_ab_raises(self):
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_triangle(body=-1, edge_ab=-1.0)

    def test_zero_edge_ab_raises(self):
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_triangle(body=-1, edge_ab=0.0)

    def test_degenerate_collinear_triangle_raises(self):
        # c_y == 0 makes A, B, C colinear along the local Z axis -> zero area.
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_triangle(body=-1, edge_ab=1.0, point_c=(0.0, 0.5))

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
        m, com, inertia = compute_inertia_shape(
            GeoType.TRIANGLE, scale, None, density=1000.0, thickness=0.0
        )
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
        m, _com, _I = compute_inertia_shape(
            GeoType.TRIANGLE, scale, None, density=density, thickness=margin
        )
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
        m, _com, inertia = compute_inertia_shape(
            GeoType.TRIANGLE, scale, None, density=density, thickness=margin
        )
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
        idx = builder.add_shape_triangle(body=-1, cfg=cfg)
        self.assertEqual(builder.shape_type[idx], GeoType.TRIANGLE)


@unittest.skipUnless(_cuda_available, "Triangle collision pipeline requires a CUDA device")
class TestTriangleNarrowPhase(unittest.TestCase):
    """End-to-end collision through the shared GJK/MPR primitive path.

    These tests exercise the same code path used by the PhoenX rigid-body
    solver (which delegates to ``newton.CollisionPipeline``), so passing
    here also validates PhoenX integration.
    """

    def test_sphere_above_triangle_face_generates_contact(self):
        # Triangle in the YZ plane (canonical), normal along local +X.
        # Place sphere just inside the triangle on the +X side.
        model, cp, state = _build_sphere_vs_triangle(
            sphere_pos=(0.05, 0.3, 0.3),
            sphere_radius=0.2,
            edge_ab=1.0,
            point_c=(1.0, 0.0),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Sphere overlapping triangle face should produce a contact")

    def test_sphere_far_from_triangle_no_contact(self):
        model, cp, state = _build_sphere_vs_triangle(
            sphere_pos=(5.0, 5.0, 5.0),
            sphere_radius=0.1,
            edge_ab=1.0,
            point_c=(1.0, 0.0),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 0, "Sphere far from triangle should produce no contacts")

    def test_triangle_is_double_sided(self):
        # Sphere on the -X side should also collide (double-sided shell).
        model_pos, cp_pos, state_pos = _build_sphere_vs_triangle(
            sphere_pos=(0.05, 0.3, 0.3),
            sphere_radius=0.2,
        )
        model_neg, cp_neg, state_neg = _build_sphere_vs_triangle(
            sphere_pos=(-0.05, 0.3, 0.3),
            sphere_radius=0.2,
        )
        n_pos = int(_collide(model_pos, cp_pos, state_pos).rigid_contact_count.numpy()[0])
        n_neg = int(_collide(model_neg, cp_neg, state_neg).rigid_contact_count.numpy()[0])
        self.assertGreater(n_pos, 0)
        self.assertGreater(n_neg, 0, "Triangle should be double-sided (both -X and +X sides collide)")

    def test_world_transform_places_triangle(self):
        # Translate the triangle far from the origin and put the sphere next to it.
        triangle_origin = wp.vec3(10.0, 20.0, 30.0)
        model, cp, state = _build_sphere_vs_triangle(
            sphere_pos=(10.05, 20.3, 30.3),
            sphere_radius=0.2,
            edge_ab=1.0,
            point_c=(1.0, 0.0),
            triangle_xform=wp.transform(triangle_origin, wp.quat_identity()),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Translated triangle should still collide with overlapping sphere")

    def test_rotated_triangle_collides_along_world_up(self):
        # Rotate +π/2 about world +Y so local +Z (edge AB) maps to world +X
        # and local +Y (vertex C direction) is unchanged. After rotation the
        # canonical triangle vertices end up at world {(0,0,0), (1,0,0),
        # (0,1,0)} — a flat triangle in the world XY plane. A sphere just
        # above the triangle's interior should collide.
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
        model, cp, state = _build_sphere_vs_triangle(
            sphere_pos=(0.25, 0.25, 0.05),
            sphere_radius=0.2,
            edge_ab=1.0,
            point_c=(1.0, 0.0),
            triangle_xform=wp.transform(wp.vec3(), rot),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Rotated triangle should still collide with overlapping sphere")


@unittest.skipUnless(_cuda_available, "Triangle collision pipeline requires a CUDA device")
class TestTriangleVsTriangle(unittest.TestCase):
    """Two TRIANGLE primitives should also route through GJK/MPR."""

    def test_intersecting_triangles_collide(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        # First triangle in the YZ plane (canonical).
        builder.add_shape_triangle(body=-1, edge_ab=1.0, point_c=(1.0, 0.0))

        # Second triangle, rotated 90 deg around Y so it lies in the XZ plane,
        # offset slightly so the two triangles intersect along Z.
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.3, 0.0), rot))
        builder.add_shape_triangle(body=body_b, edge_ab=1.0, point_c=(1.0, 0.0))

        model = builder.finalize()
        cp = newton.CollisionPipeline(model, broad_phase="explicit")
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        contacts = model.collide(state, collision_pipeline=cp)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Two intersecting triangle primitives should produce a contact")


if __name__ == "__main__":
    unittest.main()
