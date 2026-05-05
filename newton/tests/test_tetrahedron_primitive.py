# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the first-class :data:`newton.GeoType.TETRAHEDRON` primitive.

The tetrahedron primitive is a single solid-tet collider with the
canonical local frame:

    A = (0, 0, 0)
    B = (0, 0, edge_ab)
    C = (0, c_y, c_z)
    D = (d_x, d_y, d_z)

The first three free parameters ``(edge_ab, c_y, c_z)`` pack into
``shape_scale`` exactly as for :data:`newton.GeoType.TRIANGLE`. The 4th
vertex ``D`` is quantised into the per-shape ``shape_source_ptr``
``uint64`` slot via the
:func:`encode_vec3 <newton._src.geometry.support_function.encode_vec3>`
codec (shared 4-bit exponent + three 20-bit signed mantissas), and the
narrow phase decodes it back on the fly.

These tests cover:

1. ``ModelBuilder.add_shape_tetrahedron`` parameter validation.
2. ``encode_vec3`` / ``decode_vec3`` Python-vs-Warp parity (the
   builder encodes on the host, the kernel decodes on the device --
   if these drift the support function reads garbage).
3. Solid-tet inertia: closed-form mass, COM at the four-vertex
   centroid, and the Tonon (2005) inertia tensor.
4. End-to-end narrow-phase collision via the shared GJK/MPR path
   (sphere vs. tetrahedron).
5. Bounding-sphere radius and hydroelastic exclusion.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType

_cuda_available = wp.is_cuda_available()


def _build_sphere_vs_tetrahedron(
    sphere_pos,
    sphere_radius=0.2,
    edge_ab=1.0,
    point_c=(1.0, 0.0),
    point_d=(0.5, 0.5, 0.5),
    tet_xform=None,
):
    """Build a scene with a static tetrahedron (body=-1) and a sphere body."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    if tet_xform is None:
        tet_xform = wp.transform_identity()

    builder.add_shape_tetrahedron(
        body=-1,
        xform=tet_xform,
        edge_ab=edge_ab,
        point_c=point_c,
        point_d=point_d,
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


class TestTetrahedronBuilder(unittest.TestCase):
    """``add_shape_tetrahedron`` parameter handling and storage."""

    def test_default_parameters_create_unit_tet(self):
        builder = newton.ModelBuilder()
        idx = builder.add_shape_tetrahedron(body=-1)
        self.assertEqual(builder.shape_type[idx], GeoType.TETRAHEDRON)
        scale = builder.shape_scale[idx]
        self.assertAlmostEqual(scale[0], 1.0)
        self.assertAlmostEqual(scale[1], 1.0)
        self.assertAlmostEqual(scale[2], 0.0)
        # ``shape_source[idx]`` is the ``_TetrahedronVertexD`` carrier.
        d = builder.shape_source[idx]
        self.assertAlmostEqual(d.d_x, 1.0)
        self.assertAlmostEqual(d.d_y, 0.5)
        self.assertAlmostEqual(d.d_z, 0.5)

    def test_custom_parameters_stored_in_scale(self):
        builder = newton.ModelBuilder()
        idx = builder.add_shape_tetrahedron(
            body=-1,
            edge_ab=2.5,
            point_c=(1.5, -0.75),
            point_d=(0.4, -0.2, 1.1),
        )
        self.assertEqual(idx, 0)
        np.testing.assert_allclose(
            tuple(builder.shape_scale[idx]),
            (2.5, 1.5, -0.75),
            atol=1e-6,
        )

    def test_finalize_round_trips_d_through_encode_decode(self):
        # The builder host-encodes D into ``shape_source_ptr``; the test
        # decodes on the device using the same codec the narrow phase
        # uses. Precision target: micrometer-level for vertex
        # magnitudes ~1m (the codec uses a shared 4-bit exponent +
        # three 20-bit mantissas, which gives ~scale * 2e-6 absolute
        # error per component).
        if not _cuda_available:
            self.skipTest("encode/decode round-trip kernel needs a CUDA device")

        builder = newton.ModelBuilder()
        edge_ab = 1.7
        c_y = 1.3
        c_z = 0.4
        d_x = 0.55
        d_y = -0.32
        d_z = 1.18
        builder.add_shape_tetrahedron(
            body=-1,
            edge_ab=edge_ab,
            point_c=(c_y, c_z),
            point_d=(d_x, d_y, d_z),
        )
        model = builder.finalize()
        encoded = model.shape_source_ptr.numpy()[0]

        from newton._src.geometry.support_function import decode_vec3

        @wp.kernel
        def decode_kernel(p: wp.array(dtype=wp.uint64), out: wp.array(dtype=wp.vec3)):
            out[0] = decode_vec3(p[0])

        device = "cuda:0"
        p_arr = wp.array([encoded], dtype=wp.uint64, device=device)
        out_arr = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(decode_kernel, dim=1, inputs=[p_arr, out_arr], device=device)
        decoded = out_arr.numpy()[0]
        np.testing.assert_allclose(decoded, (d_x, d_y, d_z), atol=5e-6)

    def test_negative_edge_ab_raises(self):
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_tetrahedron(body=-1, edge_ab=-1.0)

    def test_zero_edge_ab_raises(self):
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_tetrahedron(body=-1, edge_ab=0.0)

    def test_degenerate_zero_c_y_raises(self):
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_tetrahedron(body=-1, point_c=(0.0, 0.5))

    def test_degenerate_d_in_yz_plane_raises(self):
        # d_x = 0 leaves D in the YZ plane (coplanar with A/B/C) -> zero volume.
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_shape_tetrahedron(body=-1, point_d=(0.0, 0.5, 0.5))

    def test_geo_type_is_classified_as_primitive(self):
        self.assertTrue(GeoType.TETRAHEDRON.is_primitive)
        self.assertFalse(GeoType.TETRAHEDRON.is_explicit)


class TestTetrahedronInertia(unittest.TestCase):
    """Solid-tet inertia (Tonon 2005)."""

    def _make_carrier(self, d_x: float, d_y: float, d_z: float):
        from newton._src.sim.builder import _TetrahedronVertexD

        return _TetrahedronVertexD(d_x, d_y, d_z)

    def test_volume_matches_six_volume_formula(self):
        from newton._src.geometry.inertia import compute_inertia_shape

        # Volume = | edge_ab * c_y * d_x | / 6.
        edge_ab, c_y, c_z = 2.0, 3.0, 0.5
        d_x, d_y, d_z = 1.5, 0.7, -0.4
        density = 1000.0
        scale = wp.vec3(edge_ab, c_y, c_z)
        m, com, _I = compute_inertia_shape(
            int(GeoType.TETRAHEDRON),
            scale,
            self._make_carrier(d_x, d_y, d_z),
            density=density,
        )
        expected_vol = abs(edge_ab * c_y * d_x) / 6.0
        self.assertAlmostEqual(m, density * expected_vol, places=4)
        # Centroid is the four-vertex mean.
        np.testing.assert_allclose(
            np.array(com),
            np.array([d_x, c_y + d_y, edge_ab + c_z + d_z]) * 0.25,
            atol=1e-6,
        )

    def test_zero_density_returns_zero_mass(self):
        from newton._src.geometry.inertia import compute_inertia_shape

        scale = wp.vec3(1.0, 1.0, 0.0)
        m, _, _ = compute_inertia_shape(
            int(GeoType.TETRAHEDRON),
            scale,
            self._make_carrier(0.5, 0.5, 0.5),
            density=0.0,
        )
        self.assertEqual(m, 0.0)

    def test_inertia_tensor_is_symmetric_and_positive_definite(self):
        from newton._src.geometry.inertia import compute_inertia_shape

        scale = wp.vec3(1.0, 1.0, 0.0)
        _m, _com, inertia = compute_inertia_shape(
            int(GeoType.TETRAHEDRON),
            scale,
            self._make_carrier(0.6, 0.4, 0.7),
            density=1000.0,
        )
        I = np.array(inertia).reshape(3, 3)
        # Symmetry.
        np.testing.assert_allclose(I, I.T, atol=1e-9)
        # Positive-definite: all eigenvalues > 0.
        eigvals = np.linalg.eigvalsh(I)
        self.assertTrue(np.all(eigvals > 0.0), f"eigvals must be positive, got {eigvals}")

    def test_regular_tet_recovers_known_inertia(self):
        # A regular tetrahedron with edge length L and uniform density
        # has centroidal inertia ``I = (m * L^2 / 20) * I_3`` (spherical:
        # all three principal moments equal because the tet's symmetry
        # group is transitive on the principal axes). Trace is rotation-
        # invariant, so independent of how the canonical local frame
        # aligns to the principal axes:
        #     trace(I) = 3 * m * L^2 / 20.
        # Reference: any rigid-body dynamics textbook; trivially
        # derivable from the Tonon formula by symmetry.
        from newton._src.geometry.inertia import compute_inertia_shape

        L = 1.0
        # Regular tet with edge L, vertex A at origin and B along +Z:
        #   A = (0, 0, 0)
        #   B = (0, 0, L)
        #   C = (0, L * sqrt(3)/2, L/2)            -> in YZ plane
        #   D = (L * sqrt(2/3), L * sqrt(3)/6, L/2) -> off the YZ plane
        # All edge lengths equal L (verified analytically).
        c_y = L * math.sqrt(3.0) / 2.0
        c_z = L / 2.0
        d_x = L * math.sqrt(2.0 / 3.0)
        d_y = L * math.sqrt(3.0) / 6.0
        d_z = L / 2.0
        density = 1000.0
        scale = wp.vec3(L, c_y, c_z)
        m, _com, inertia = compute_inertia_shape(
            int(GeoType.TETRAHEDRON),
            scale,
            self._make_carrier(d_x, d_y, d_z),
            density=density,
        )
        I = np.array(inertia).reshape(3, 3)
        expected_trace = (3.0 / 20.0) * m * L * L
        self.assertAlmostEqual(float(np.trace(I)), expected_trace, places=5)
        # Eigenvalues should be (very nearly) equal -- spherical inertia.
        eigvals = np.linalg.eigvalsh(I)
        self.assertAlmostEqual(eigvals[0], eigvals[1], places=5)
        self.assertAlmostEqual(eigvals[1], eigvals[2], places=5)

    def test_compute_shape_radius(self):
        from newton._src.geometry.utils import compute_shape_radius

        # A=origin, B=(0,0,3), C=(0,4,0), D=(5,0,0). Bounding sphere
        # centered at A must enclose D at distance 5.
        r = compute_shape_radius(
            int(GeoType.TETRAHEDRON),
            (3.0, 4.0, 0.0),
            self._make_carrier(5.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(r, 5.0)


class TestTetrahedronHydroelasticValidation(unittest.TestCase):
    """Hydroelastic contact is intentionally not supported for tets."""

    def test_hydroelastic_tetrahedron_passes_validation_without_sdf(self):
        cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True)
        cfg.validate(shape_type=GeoType.TETRAHEDRON)

        builder = newton.ModelBuilder()
        idx = builder.add_shape_tetrahedron(body=-1, cfg=cfg)
        self.assertEqual(builder.shape_type[idx], GeoType.TETRAHEDRON)


@unittest.skipUnless(_cuda_available, "Tetrahedron collision pipeline requires a CUDA device")
class TestTetrahedronNarrowPhase(unittest.TestCase):
    """End-to-end collision through the shared GJK/MPR primitive path."""

    def test_sphere_inside_tet_generates_contact(self):
        # Place the sphere very close to the tet's centroid -- it must
        # be inside (or at least overlapping) so the narrow phase picks
        # up at least one contact.
        edge_ab, c_y, c_z = 1.0, 1.0, 0.0
        d_x, d_y, d_z = 1.0, 0.5, 0.5
        centroid = (
            d_x * 0.25,
            (c_y + d_y) * 0.25,
            (edge_ab + c_z + d_z) * 0.25,
        )
        model, cp, state = _build_sphere_vs_tetrahedron(
            sphere_pos=centroid,
            sphere_radius=0.2,
            edge_ab=edge_ab,
            point_c=(c_y, c_z),
            point_d=(d_x, d_y, d_z),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Sphere at tet centroid must overlap and produce a contact")

    def test_sphere_far_from_tet_no_contact(self):
        model, cp, state = _build_sphere_vs_tetrahedron(
            sphere_pos=(50.0, 50.0, 50.0),
            sphere_radius=0.1,
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 0)

    def test_world_transform_places_tetrahedron(self):
        tet_origin = wp.vec3(10.0, 20.0, 30.0)
        # Centroid for default tet (edge_ab=1, point_c=(1,0), point_d=(0.5,0.5,0.5)):
        # ((0+0+0+0.5)/4, (0+0+1+0.5)/4, (0+1+0+0.5)/4) = (0.125, 0.375, 0.375)
        sphere_local = wp.vec3(0.125, 0.375, 0.375)
        sphere_world = (
            float(tet_origin[0] + sphere_local[0]),
            float(tet_origin[1] + sphere_local[1]),
            float(tet_origin[2] + sphere_local[2]),
        )
        model, cp, state = _build_sphere_vs_tetrahedron(
            sphere_pos=sphere_world,
            sphere_radius=0.2,
            tet_xform=wp.transform(tet_origin, wp.quat_identity()),
        )
        contacts = _collide(model, cp, state)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0, "Tet should be translated in world space and collide")


if __name__ == "__main__":
    unittest.main()
