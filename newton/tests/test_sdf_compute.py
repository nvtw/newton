# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test compute_sdf function for SDF generation.

This test suite validates:
1. SDF values inside the extent are smaller than the background value
2. Sparse and coarse SDFs have consistent values
3. SDF gradients point away from the surface
4. Points inside the mesh have negative SDF values
5. Points outside the mesh have positive SDF values
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.sdf_utils import compute_sdf
from newton._src.geometry.types import Mesh


def create_box_mesh(half_extents: tuple[float, float, float]) -> Mesh:
    """Create a simple box mesh for testing."""
    hx, hy, hz = half_extents
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            # Bottom face (z = -hz)
            0,
            2,
            1,
            0,
            3,
            2,
            # Top face (z = hz)
            4,
            5,
            6,
            4,
            6,
            7,
            # Front face (y = -hy)
            0,
            1,
            5,
            0,
            5,
            4,
            # Back face (y = hy)
            2,
            3,
            7,
            2,
            7,
            6,
            # Left face (x = -hx)
            0,
            4,
            7,
            0,
            7,
            3,
            # Right face (x = hx)
            1,
            2,
            6,
            1,
            6,
            5,
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, indices)


# Warp kernel for sampling SDF values
@wp.kernel
def sample_sdf_kernel(
    volume_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    point = points[tid]
    index_pos = wp.volume_world_to_index(volume_id, point)
    values[tid] = wp.volume_sample_f(volume_id, index_pos, wp.Volume.LINEAR)


# Warp kernel for sampling SDF gradients
@wp.kernel
def sample_sdf_gradient_kernel(
    volume_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
    gradients: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    point = points[tid]
    index_pos = wp.volume_world_to_index(volume_id, point)
    grad = wp.vec3(0.0, 0.0, 0.0)
    values[tid] = wp.volume_sample_grad_f(volume_id, index_pos, wp.Volume.LINEAR, grad)
    gradients[tid] = grad


def sample_sdf_at_points(volume, points_np: np.ndarray) -> np.ndarray:
    """Sample SDF values at given points using a Warp kernel."""
    n_points = len(points_np)
    points = wp.array(points_np, dtype=wp.vec3)
    values = wp.zeros(n_points, dtype=wp.float32)

    wp.launch(
        sample_sdf_kernel,
        dim=n_points,
        inputs=[volume.id, points, values],
    )
    wp.synchronize()

    return values.numpy()


def sample_sdf_with_gradient(volume, points_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample SDF values and gradients at given points using a Warp kernel."""
    n_points = len(points_np)
    points = wp.array(points_np, dtype=wp.vec3)
    values = wp.zeros(n_points, dtype=wp.float32)
    gradients = wp.zeros(n_points, dtype=wp.vec3)

    wp.launch(
        sample_sdf_gradient_kernel,
        dim=n_points,
        inputs=[volume.id, points, values, gradients],
    )
    wp.synchronize()

    return values.numpy(), gradients.numpy()


class TestComputeSDF(unittest.TestCase):
    """Test the compute_sdf function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def setUp(self):
        """Set up test fixtures."""
        self.half_extents = (0.5, 0.5, 0.5)
        self.mesh = create_box_mesh(self.half_extents)

    def test_sdf_returns_valid_data(self):
        """Test that compute_sdf returns valid data."""
        sdf_data, sparse_volume, coarse_volume = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        self.assertIsNotNone(sparse_volume)
        self.assertIsNotNone(coarse_volume)
        self.assertNotEqual(sdf_data.sparse_sdf_ptr, 0)
        self.assertNotEqual(sdf_data.coarse_sdf_ptr, 0)

    def test_sdf_extents_are_valid(self):
        """Test that SDF extents match the mesh bounds."""
        sdf_data, _, _ = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
            margin=0.05,
        )

        # Half extents should be at least as large as mesh half extents + margin
        min_half_extent = min(self.half_extents) + 0.05
        self.assertGreaterEqual(sdf_data.half_extents[0], min_half_extent - 0.01)
        self.assertGreaterEqual(sdf_data.half_extents[1], min_half_extent - 0.01)
        self.assertGreaterEqual(sdf_data.half_extents[2], min_half_extent - 0.01)

    def test_sparse_sdf_values_near_surface(self):
        """Test that sparse SDF values near the surface are smaller than background.

        Note: The sparse SDF is a narrow-band SDF, so only values near the surface
        (within narrow_band_distance) will have valid values. Points far from the
        surface will return the background value.
        """
        sdf_data, sparse_volume, _ = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
            narrow_band_distance=(-0.1, 0.1),
        )

        # Test points near the surface (within narrow band)
        # These are just inside and just outside each face of the box
        test_points = np.array(
            [
                [0.45, 0.0, 0.0],  # Near +X face (inside)
                [0.55, 0.0, 0.0],  # Near +X face (outside)
                [0.0, 0.45, 0.0],  # Near +Y face (inside)
                [0.0, 0.0, 0.45],  # Near +Z face (inside)
                [-0.45, 0.0, 0.0],  # Near -X face (inside)
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(sparse_volume, test_points)

        for _i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"SDF value {value} at {point} (near surface) should be less than background {sdf_data.background_value}",
            )

    def test_coarse_sdf_values_inside_extent(self):
        """Test that coarse SDF values inside the extent are smaller than background."""
        sdf_data, _, coarse_volume = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        # Sample points inside the SDF extent
        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Generate test points inside the extent
        test_points = np.array(
            [
                center,  # Center
                center + half_ext * 0.5,  # Offset from center
                center - half_ext * 0.5,  # Other offset
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(coarse_volume, test_points)

        for _i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Coarse SDF value {value} at {point} should be less than background {sdf_data.background_value}",
            )

    def test_coarse_sdf_values_at_extent_boundary(self):
        """Test that coarse SDF values at the extent boundary are valid.

        The extent boundary is at center ± half_extents. With margin=0.05 and
        mesh half_extents of 0.5, the boundary is at approximately ±0.55.
        Points at or near this boundary should still have valid SDF values.
        """
        margin = 0.05
        sdf_data, _, coarse_volume = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
            margin=margin,
        )

        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Verify the extent includes the margin
        expected_half_ext = self.half_extents[0] + margin  # 0.5 + 0.05 = 0.55
        self.assertAlmostEqual(
            half_ext[0],
            expected_half_ext,
            places=2,
            msg=f"Expected half_extent ~{expected_half_ext}, got {half_ext[0]}",
        )

        # Test points at extent boundary corners (slightly inside to ensure we're in the volume)
        boundary_factor = 0.99  # Just inside the boundary
        test_points = np.array(
            [
                # Corners of the extent (outside the mesh, inside the extent)
                center + half_ext * np.array([boundary_factor, boundary_factor, boundary_factor]),
                center + half_ext * np.array([boundary_factor, boundary_factor, -boundary_factor]),
                center + half_ext * np.array([boundary_factor, -boundary_factor, boundary_factor]),
                center + half_ext * np.array([boundary_factor, -boundary_factor, -boundary_factor]),
                center + half_ext * np.array([-boundary_factor, boundary_factor, boundary_factor]),
                center + half_ext * np.array([-boundary_factor, boundary_factor, -boundary_factor]),
                center + half_ext * np.array([-boundary_factor, -boundary_factor, boundary_factor]),
                center + half_ext * np.array([-boundary_factor, -boundary_factor, -boundary_factor]),
                # Face centers at the boundary
                center + half_ext * np.array([boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([-boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([0.0, boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, -boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, 0.0, boundary_factor]),
                center + half_ext * np.array([0.0, 0.0, -boundary_factor]),
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(coarse_volume, test_points)

        for i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Coarse SDF at extent boundary point {i} = {point} should be < {sdf_data.background_value}, got {value}",
            )
            # Corners are outside the mesh (which is at ±0.5), so SDF should be positive
            # Face center points at ±0.55 on one axis and 0 on others are also outside mesh
            self.assertGreater(
                value,
                0.0,
                f"Coarse SDF at extent boundary (outside mesh at ±0.5) should be positive, got {value} at {point}",
            )

    def test_sparse_sdf_values_at_extent_boundary(self):
        """Test that sparse SDF values at the actual extent boundary are valid.

        The extent boundary is at center ± half_extents. With margin=0.05 and
        mesh half_extents of 0.5, the extent boundary is at approximately ±0.55.

        The narrow band extends ±0.1 from the surface (at ±0.5), so the narrow
        band covers [0.4, 0.6] for each face. The extent boundary at 0.55 is
        within this narrow band, so we should get valid values there.
        """
        margin = 0.05
        sdf_data, sparse_volume, _ = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
            margin=margin,
        )

        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Verify the extent is what we expect (mesh half_extents + margin)
        expected_half_ext = self.half_extents[0] + margin  # 0.5 + 0.05 = 0.55
        self.assertAlmostEqual(
            half_ext[0],
            expected_half_ext,
            places=2,
            msg=f"Expected half_extent ~{expected_half_ext}, got {half_ext[0]}",
        )

        # Test points AT the extent boundary (0.99 * half_ext to stay just inside)
        # These should be within the narrow band since:
        # - Surface is at 0.5
        # - Narrow band extends to 0.5 + 0.1 = 0.6
        # - Extent boundary is at ~0.55, which is < 0.6
        boundary_factor = 0.99
        boundary_points = np.array(
            [
                # Face centers at extent boundary
                center + half_ext * np.array([boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([-boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([0.0, boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, -boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, 0.0, boundary_factor]),
                center + half_ext * np.array([0.0, 0.0, -boundary_factor]),
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(sparse_volume, boundary_points)

        for i, (point, value) in enumerate(zip(boundary_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Sparse SDF at extent boundary point {i} = {point} should be < {sdf_data.background_value}, got {value}",
            )
            # These points are outside the mesh surface, so SDF should be positive
            self.assertGreater(
                value,
                0.0,
                f"Sparse SDF at extent boundary (outside mesh) should be positive, got {value} at {point}",
            )

    def test_sdf_negative_inside_mesh(self):
        """Test that SDF values are negative inside the mesh.

        For the sparse SDF, we test a point just inside the surface (within the narrow band).
        For the coarse SDF, we can test the center since it covers the entire volume.
        """
        _sdf_data, sparse_volume, coarse_volume = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        # For sparse SDF: test point just inside a face (within narrow band)
        near_surface_inside = np.array([[0.45, 0.0, 0.0]], dtype=np.float32)
        sparse_values = sample_sdf_at_points(sparse_volume, near_surface_inside)
        self.assertLess(
            sparse_values[0], 0.0, f"Sparse SDF just inside surface should be negative, got {sparse_values[0]}"
        )

        # For coarse SDF: test at center (coarse SDF covers entire volume)
        center_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        coarse_values = sample_sdf_at_points(coarse_volume, center_point)
        self.assertLess(coarse_values[0], 0.0, f"Coarse SDF at center should be negative, got {coarse_values[0]}")

    def test_sdf_positive_outside_mesh(self):
        """Test that SDF values are positive outside the mesh."""
        _sdf_data, sparse_volume, coarse_volume = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        # Point well outside the box
        outside_point = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)

        # Test sparse SDF (may hit background value if outside narrow band)
        sparse_values = sample_sdf_at_points(sparse_volume, outside_point)
        self.assertGreater(sparse_values[0], 0.0, f"Sparse SDF outside should be positive, got {sparse_values[0]}")

        # Test coarse SDF
        coarse_values = sample_sdf_at_points(coarse_volume, outside_point)
        self.assertGreater(coarse_values[0], 0.0, f"Coarse SDF outside should be positive, got {coarse_values[0]}")

    def test_sdf_gradient_points_outward(self):
        """Test that SDF gradient points away from the surface (outward)."""
        _sdf_data, sparse_volume, _ = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        # Test gradient at a point slightly inside the +X face
        test_points = np.array([[0.4, 0.0, 0.0]], dtype=np.float32)  # Inside the box, close to +X face

        _values, gradients = sample_sdf_with_gradient(sparse_volume, test_points)

        gradient = gradients[0]
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm > 1e-6:
            gradient_normalized = gradient / gradient_norm
            # X component should be positive (pointing outward toward +X face)
            self.assertGreater(
                gradient_normalized[0],
                0.5,
                f"Gradient should point toward +X, got {gradient_normalized}",
            )

    def test_sparse_and_coarse_consistency(self):
        """Test that sparse and coarse SDFs have consistent signs near the surface.

        We test at a point near the surface (within the narrow band) where both
        SDFs should have valid values.
        """
        _sdf_data, sparse_volume, coarse_volume = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        # Sample at a point near the surface (within narrow band)
        near_surface = np.array([[0.45, 0.0, 0.0]], dtype=np.float32)

        sparse_values = sample_sdf_at_points(sparse_volume, near_surface)
        coarse_values = sample_sdf_at_points(coarse_volume, near_surface)

        # Both should have the same sign (both negative inside)
        self.assertEqual(
            np.sign(sparse_values[0]),
            np.sign(coarse_values[0]),
            f"Sparse ({sparse_values[0]}) and coarse ({coarse_values[0]}) should have same sign near surface",
        )

    def test_non_colliding_shape_returns_empty_sdf(self):
        """Test that non-colliding shapes return empty SDFData."""
        sdf_data, sparse_volume, coarse_volume = compute_sdf(
            shape_flags=0,  # No collision flags
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        self.assertIsNone(sparse_volume)
        self.assertIsNone(coarse_volume)
        self.assertEqual(sdf_data.sparse_sdf_ptr, 0)
        self.assertEqual(sdf_data.coarse_sdf_ptr, 0)

    def test_thickness_offset(self):
        """Test that thickness offsets the SDF values.

        We test near the surface where the sparse SDF has valid values.
        """
        thickness = 0.1

        _, sparse_no_thickness, _ = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        _, sparse_with_thickness, _ = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=thickness,
            mesh_src=self.mesh,
        )

        # Sample near the surface (within narrow band)
        near_surface = np.array([[0.45, 0.0, 0.0]], dtype=np.float32)

        values_no_thick = sample_sdf_at_points(sparse_no_thickness, near_surface)
        values_with_thick = sample_sdf_at_points(sparse_with_thickness, near_surface)

        # With thickness, SDF should be offset (more negative = thicker shell)
        self.assertAlmostEqual(
            values_with_thick[0],
            values_no_thick[0] - thickness,
            places=2,
            msg=f"Thickness should offset SDF by -{thickness}",
        )


class TestComputeSDFGridSampling(unittest.TestCase):
    """Test SDF by sampling on a grid of points."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def setUp(self):
        """Set up test fixtures."""
        self.half_extents = (0.5, 0.5, 0.5)
        self.mesh = create_box_mesh(self.half_extents)

    def test_grid_sampling_sparse_sdf_near_surface(self):
        """Sample sparse SDF on a grid near the surface and verify values are valid.

        Since the sparse SDF is a narrow-band SDF, we sample points near the surface
        (on a shell around the box) where the SDF should have valid values.
        """
        sdf_data, sparse_volume, _ = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        # Sample points on a grid near the +X face of the box (within narrow band)
        test_points = []
        for j in range(5):
            for k in range(5):
                # Grid on the YZ plane, at x = 0.45 (just inside the surface)
                y = (j / 4.0 - 0.5) * 0.8  # Range [-0.4, 0.4]
                z = (k / 4.0 - 0.5) * 0.8
                test_points.append([0.45, y, z])
                # Also test just outside
                test_points.append([0.55, y, z])

        test_points = np.array(test_points, dtype=np.float32)
        values = sample_sdf_at_points(sparse_volume, test_points)

        for i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"SDF at point {i} = {point} (near surface) should be < {sdf_data.background_value}, got {value}",
            )

    def test_grid_sampling_coarse_sdf(self):
        """Sample coarse SDF on a grid and verify all values are less than background."""
        sdf_data, _, coarse_volume = compute_sdf(
            shape_flags=ShapeFlags.COLLIDE_SHAPES,
            shape_thickness=0.0,
            mesh_src=self.mesh,
        )

        # Create a grid of test points inside the extent
        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Sample on a 5x5x5 grid inside the extent
        test_points = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    # Normalized coordinates [-0.8, 0.8] to stay inside extent
                    u = (i / 4.0 - 0.5) * 1.6
                    v = (j / 4.0 - 0.5) * 1.6
                    w = (k / 4.0 - 0.5) * 1.6
                    point = center + half_ext * np.array([u, v, w])
                    test_points.append(point)

        test_points = np.array(test_points, dtype=np.float32)
        values = sample_sdf_at_points(coarse_volume, test_points)

        for i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Coarse SDF at grid point {i} = {point} should be < {sdf_data.background_value}, got {value}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
