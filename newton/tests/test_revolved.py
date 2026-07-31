# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for convex cubic-Bézier solids of revolution."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import compute_inertia_shape
from newton.viewer import ViewerNull
from newton._src.geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    revolved_power_coefficients,
    support_map,
    unpack_revolved_control_radii,
)
from newton._src.geometry.types import RevolvedData
from newton._src.viewer.viewer import _create_revolved_display_mesh


@wp.kernel
def _support_kernel(
    scale: wp.vec3,
    packed_controls: wp.uint64,
    directions: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
):
    """Evaluate revolved support points."""
    controls = unpack_revolved_control_radii(packed_controls)
    shape = GenericShapeData()
    shape.shape_type = int(newton.GeoType.REVOLVED)
    shape.scale = scale
    shape.auxiliary = revolved_power_coefficients(scale, controls)
    provider = SupportMapDataProvider()
    points[wp.tid()] = support_map(shape, directions[wp.tid()], provider)


def _sample_profile(parameters, sample_count=20001):
    r0, r1, c0, c1, half_height = parameters
    t = np.linspace(0.0, 1.0, sample_count)
    omt = 1.0 - t
    radius = omt**3 * r0 + 3.0 * omt**2 * t * c0 + 3.0 * omt * t**2 * c1 + t**3 * r1
    z = -half_height + half_height * t**2 * (6.0 - 4.0 * t)
    return radius, z


class TestRevolved(unittest.TestCase):
    """Test the revolved primitive's public and analytic behavior."""

    def test_builder_storage_and_defaults(self):
        """Store controls in the existing source field and default to a frustum."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_revolved(body, radius_bottom=0.9, radius_top=0.3, half_height=0.7)
        source = builder.shape_source[shape]
        self.assertAlmostEqual(source.radius_control_bottom, 0.9)
        self.assertAlmostEqual(source.radius_control_top, 0.3)

        model = builder.finalize(device="cpu")
        self.assertEqual(int(model.shape_type.numpy()[shape]), int(newton.GeoType.REVOLVED))
        self.assertEqual(
            int(model.shape_source_ptr.numpy()[shape]),
            RevolvedData(0.9, 0.3).finalize(),
        )
        np.testing.assert_allclose(model.shape_scale.numpy()[shape], (0.9, 0.3, 0.7))

    def test_builder_rejects_nonconvex_profiles(self):
        """Reject cubic controls whose radial profile is not concave."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        with self.assertRaisesRegex(ValueError, "radius_control_bottom"):
            builder.add_shape_revolved(
                body,
                radius_bottom=1.0,
                radius_top=1.0,
                radius_control_bottom=0.75,
                radius_control_top=1.0,
            )
        with self.assertRaisesRegex(ValueError, "radius_control_top"):
            builder.add_shape_revolved(
                body,
                radius_bottom=1.0,
                radius_top=1.0,
                radius_control_bottom=1.0,
                radius_control_top=0.75,
            )

    def test_builder_rejects_invalid_dimensions(self):
        """Reject non-finite, negative, and degenerate dimensions."""
        invalid_arguments = (
            {"half_height": 0.0},
            {"radius_bottom": -0.1},
            {"radius_top": float("nan")},
            {
                "radius_bottom": 0.0,
                "radius_top": 0.0,
                "radius_control_bottom": 0.0,
                "radius_control_top": 0.0,
            },
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                builder = newton.ModelBuilder()
                body = builder.add_body()
                with self.assertRaises(ValueError):
                    builder.add_shape_revolved(body, **arguments)

    def test_support_matches_dense_sampling(self):
        """Match analytic support values against a densely sampled profile."""
        profiles = (
            (0.8, 0.6, 1.2, 1.0, 0.7),
            (0.8, 0.2, 0.8, 0.2, 0.7),
            (0.8, 0.6, 0.8, 1.0, 0.7),
            (0.8, 0.6, 1.2, 0.6, 0.7),
        )
        rng = np.random.default_rng(1234)
        directions_np = rng.normal(size=(128, 3)).astype(np.float32)
        directions_np = np.vstack(
            (directions_np, np.eye(3, dtype=np.float32), -np.eye(3, dtype=np.float32))
        )
        devices = ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]
        for parameters in profiles:
            source = RevolvedData(parameters[2], parameters[3])
            radius, z = _sample_profile(parameters)
            for device in devices:
                with self.subTest(parameters=parameters, device=device):
                    directions = wp.array(directions_np, dtype=wp.vec3, device=device)
                    points = wp.empty(len(directions_np), dtype=wp.vec3, device=device)
                    wp.launch(
                        _support_kernel,
                        dim=len(directions_np),
                        inputs=[
                            wp.vec3(parameters[0], parameters[1], parameters[4]),
                            wp.uint64(source.finalize()),
                            directions,
                            points,
                        ],
                        device=device,
                    )

                    support_points = points.numpy()
                    for direction, point in zip(directions_np, support_points, strict=True):
                        radial_direction = np.linalg.norm(direction[:2])
                        expected = np.max(radial_direction * radius + direction[2] * z)
                        self.assertAlmostEqual(float(np.dot(direction, point)), float(expected), delta=2.0e-5)

    def test_inertia_reduces_to_cylinder(self):
        """Recover exact cylinder mass properties from constant control radii."""
        radius = 0.6
        half_height = 0.8
        density = 1200.0
        source = RevolvedData(radius, radius)
        revolved = compute_inertia_shape(
            newton.GeoType.REVOLVED,
            wp.vec3(radius, radius, half_height),
            source,
            density,
        )
        cylinder = compute_inertia_shape(
            newton.GeoType.CYLINDER,
            wp.vec3(radius, half_height, 0.0),
            None,
            density,
        )
        self.assertAlmostEqual(revolved[0], cylinder[0], places=5)
        np.testing.assert_allclose(np.asarray(revolved[1]), np.asarray(cylinder[1]), atol=1.0e-6)
        np.testing.assert_allclose(np.asarray(revolved[2]), np.asarray(cylinder[2]), rtol=1.0e-6, atol=1.0e-6)

    def test_inertia_reduces_to_cone(self):
        """Recover exact cone mass properties from collinear control radii."""
        radius = 0.6
        half_height = 0.8
        density = 1200.0
        source = RevolvedData(radius, 0.0)
        revolved = compute_inertia_shape(
            newton.GeoType.REVOLVED,
            wp.vec3(radius, 0.0, half_height),
            source,
            density,
        )
        cone = compute_inertia_shape(
            newton.GeoType.CONE,
            wp.vec3(radius, half_height, 0.0),
            None,
            density,
        )
        self.assertAlmostEqual(revolved[0], cone[0], delta=1.0e-6 * cone[0])
        np.testing.assert_allclose(np.asarray(revolved[1]), np.asarray(cone[1]), atol=1.0e-6)
        np.testing.assert_allclose(np.asarray(revolved[2]), np.asarray(cone[2]), rtol=1.0e-6, atol=1.0e-6)

    def test_curved_inertia_matches_numerical_integration(self):
        """Match exact curved-profile inertia against numerical integration."""
        parameters = (0.8, 0.6, 1.2, 1.0, 0.7)
        density = 1200.0
        source = RevolvedData(parameters[2], parameters[3])
        mass, center, inertia = compute_inertia_shape(
            newton.GeoType.REVOLVED,
            wp.vec3(parameters[0], parameters[1], parameters[4]),
            source,
            density,
        )

        radius, z = _sample_profile(parameters, sample_count=100001)
        t = np.linspace(0.0, 1.0, len(radius))
        dz_dt = 12.0 * parameters[4] * t * (1.0 - t)
        mass_density = density * np.pi * radius**2 * dz_dt
        expected_mass = np.trapezoid(mass_density, t)
        expected_center_z = np.trapezoid(z * mass_density, t) / expected_mass
        expected_inertia_z = density * 0.5 * np.pi * np.trapezoid(radius**4 * dz_dt, t)
        expected_inertia_x = density * np.pi * np.trapezoid(
            (0.25 * radius**4 + z**2 * radius**2) * dz_dt, t
        ) - expected_mass * expected_center_z**2

        self.assertAlmostEqual(mass, expected_mass, delta=1.0e-7 * expected_mass)
        self.assertAlmostEqual(float(center[2]), expected_center_z, delta=1.0e-7)
        self.assertAlmostEqual(float(inertia[0, 0]), expected_inertia_x, delta=1.0e-6 * expected_inertia_x)
        self.assertAlmostEqual(float(inertia[2, 2]), expected_inertia_z, delta=1.0e-6 * expected_inertia_z)

    def test_rigid_collision_pipeline(self):
        """Generate a rigid contact between a revolved shape and a sphere."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        barrel = builder.add_body()
        builder.add_shape_revolved(
            barrel,
            radius_bottom=0.8,
            radius_top=0.8,
            radius_control_bottom=1.0,
            radius_control_top=1.0,
            half_height=0.5,
        )
        sphere = builder.add_body(xform=wp.transform(wp.vec3(1.2, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(sphere, radius=0.4)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def test_display_mesh_covers_profile(self):
        """Tessellate the complete side profile and both end caps."""
        source = RevolvedData(1.2, 1.0)
        mesh = _create_revolved_display_mesh((0.8, 0.6, 0.7), source)
        vertices = mesh.vertices
        self.assertEqual(vertices.shape[0], 17 * 32 + 2 * 33)
        self.assertAlmostEqual(float(vertices[:, 2].min()), -0.7, places=6)
        self.assertAlmostEqual(float(vertices[:, 2].max()), 0.7, places=6)
        self.assertGreater(float(np.linalg.norm(vertices[:, :2], axis=1).max()), 1.0)
        self.assertEqual(len(mesh.indices) % 3, 0)

        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_revolved(
            body,
            radius_bottom=0.8,
            radius_top=0.6,
            radius_control_bottom=1.2,
            radius_control_top=1.0,
            half_height=0.7,
        )
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(builder.finalize(device="cpu"))
        self.assertEqual(len(viewer._shape_instances), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
