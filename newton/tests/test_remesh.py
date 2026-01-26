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

"""Test the remeshing functionality (PointCloudExtractor and SurfaceReconstructor).

This test suite validates:
1. Point cloud extraction from a simple cube mesh
2. Surface reconstruction produces a mesh close to the original geometry
3. Input validation works correctly

Note: SurfaceReconstructor requires Open3D which is an optional dependency.
"""

import importlib.util
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.remesh import compute_bounding_sphere, compute_camera_basis
from newton.geometry import PointCloudExtractor, SurfaceReconstructor

# Check if Open3D is available for reconstruction tests
OPEN3D_AVAILABLE = importlib.util.find_spec("open3d") is not None

# Check if CUDA is available (required for Warp mesh raycasting)
_cuda_available = wp.is_cuda_available()


def create_unit_cube_mesh(center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Create a unit cube mesh.

    Args:
        center: Optional center point. If None, cube is centered at origin.

    Returns:
        Tuple of (vertices, indices) where vertices is (8, 3) and indices is (36,).
    """
    # Half extents of 0.5 gives a unit cube
    hx, hy, hz = 0.5, 0.5, 0.5
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
    if center is not None:
        vertices = vertices + np.array(center, dtype=np.float32)

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
    return vertices, indices


def compute_distance_to_cube(
    points: np.ndarray, half_extent: float = 0.5, center: np.ndarray | None = None
) -> np.ndarray:
    """Compute the unsigned distance from points to a cube surface.

    Args:
        points: (N, 3) array of points.
        half_extent: Half the side length of the cube.
        center: Center of the cube. If None, assumes origin.

    Returns:
        (N,) array of unsigned distances to the cube surface.
    """
    if center is not None:
        points = points - np.array(center, dtype=np.float32)

    h = half_extent
    abs_coords = np.abs(points)

    # Distance outside each axis bound (clamped to 0 for inside)
    dx = np.maximum(abs_coords[:, 0] - h, 0)
    dy = np.maximum(abs_coords[:, 1] - h, 0)
    dz = np.maximum(abs_coords[:, 2] - h, 0)

    # For points outside the cube, distance is Euclidean distance to nearest corner/edge/face
    outside_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # For points inside, distance to nearest face
    inside_dist = h - np.max(abs_coords, axis=1)

    # Points are outside if any coordinate exceeds h
    is_outside = np.any(abs_coords > h, axis=1)

    distances = np.where(is_outside, outside_dist, np.abs(inside_dist))
    return distances


def classify_points_by_face(points: np.ndarray, half_extent: float = 0.5, tolerance: float = 0.01) -> dict[str, int]:
    """Classify points by which cube face they are closest to.

    Args:
        points: (N, 3) array of points on/near cube surface.
        half_extent: Half extent of the cube.
        tolerance: Distance tolerance for considering a point on a face.

    Returns:
        Dictionary mapping face names (+X, -X, +Y, -Y, +Z, -Z) to point counts.
    """
    h = half_extent
    counts = {"+X": 0, "-X": 0, "+Y": 0, "-Y": 0, "+Z": 0, "-Z": 0}

    for point in points:
        x, y, z = point
        # Find which face this point is closest to
        dists = {
            "+X": abs(x - h),
            "-X": abs(x + h),
            "+Y": abs(y - h),
            "-Y": abs(y + h),
            "+Z": abs(z - h),
            "-Z": abs(z + h),
        }
        closest_face = min(dists, key=dists.get)
        if dists[closest_face] < tolerance:
            counts[closest_face] += 1

    return counts


@unittest.skipUnless(_cuda_available, "Warp mesh raycasting requires CUDA")
class TestPointCloudExtractor(unittest.TestCase):
    """Test the PointCloudExtractor class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def test_extract_cube_produces_sufficient_points(self):
        """Test that extracting a point cloud from a cube produces many points."""
        vertices, indices = create_unit_cube_mesh()

        # Use fast settings: low resolution, low subdivision
        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)
        result = extractor.extract(vertices, indices)

        # With 80 views at 100x100 = 800,000 potential rays
        # A unit cube should intercept a significant fraction of rays
        # Expect at least 1000 points (very conservative minimum)
        self.assertGreater(
            result.num_points, 1000, f"Should extract many points from a cube, got only {result.num_points}"
        )

        # Points should have correct shape
        self.assertEqual(result.points.shape[1], 3, "Points should be 3D")
        self.assertEqual(result.normals.shape[1], 3, "Normals should be 3D")
        self.assertEqual(len(result.points), len(result.normals), "Should have same number of points and normals")

    def test_extract_cube_points_on_surface(self):
        """Test that extracted points are precisely on the cube surface."""
        vertices, indices = create_unit_cube_mesh()

        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)
        result = extractor.extract(vertices, indices)

        # Compute distance of each extracted point to the cube surface
        distances = compute_distance_to_cube(result.points)

        # Ray intersection points should be ON the surface, not near it
        # Allow only floating-point precision errors
        max_distance = np.max(distances)
        mean_distance = np.mean(distances)

        self.assertLess(
            max_distance,
            1e-4,  # Tight tolerance - points should be ON the surface
            f"Points should be on the cube surface, max distance: {max_distance:.6f}",
        )
        self.assertLess(
            mean_distance,
            1e-5,
            f"Mean distance should be near zero, got: {mean_distance:.6f}",
        )

    def test_extract_cube_covers_all_faces(self):
        """Test that point cloud extraction covers all 6 faces of the cube."""
        vertices, indices = create_unit_cube_mesh()

        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)
        result = extractor.extract(vertices, indices)

        # Classify points by face
        face_counts = classify_points_by_face(result.points, tolerance=0.01)

        # Each face should have a significant number of points
        min_points_per_face = 100  # Conservative minimum
        for face_name, count in face_counts.items():
            self.assertGreater(
                count,
                min_points_per_face,
                f"Face {face_name} should have at least {min_points_per_face} points, got {count}",
            )

    def test_extract_cube_normals_unit_length(self):
        """Test that extracted normals are unit length."""
        vertices, indices = create_unit_cube_mesh()

        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)
        result = extractor.extract(vertices, indices)

        # Compute normal lengths
        normal_lengths = np.linalg.norm(result.normals, axis=1)

        # Normals should be exactly unit length (within floating point precision)
        self.assertTrue(
            np.allclose(normal_lengths, 1.0, atol=1e-5),
            f"Normals should be unit length, got range [{normal_lengths.min():.6f}, {normal_lengths.max():.6f}]",
        )

    def test_extract_cube_normals_point_outward(self):
        """Test that normals point outward from the cube surface."""
        vertices, indices = create_unit_cube_mesh()

        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)
        result = extractor.extract(vertices, indices)

        # For a cube centered at origin, outward normals should point away from center
        # For each point on the surface, the normal should point in the same general
        # direction as the vector from center to point
        # More specifically, for cube faces, the normal should be parallel to one axis

        # Check that most normals point outward (dot product with position > 0)
        # Note: For points exactly at face centers, position and normal are parallel
        # For points near edges/corners, this is less precise
        dots = np.sum(result.points * result.normals, axis=1)

        # Almost all should be positive (pointing outward)
        fraction_outward = np.mean(dots > -0.01)  # Small negative tolerance for edge cases
        self.assertGreater(
            fraction_outward,
            0.99,
            f"At least 99% of normals should point outward, got {fraction_outward * 100:.1f}%",
        )

    def test_extract_translated_cube(self):
        """Test extraction works for mesh not centered at origin."""
        center = np.array([10.0, -5.0, 3.0])
        vertices, indices = create_unit_cube_mesh(center=center)

        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)
        result = extractor.extract(vertices, indices)

        # Points should still be on the (translated) cube surface
        distances = compute_distance_to_cube(result.points, half_extent=0.5, center=center)
        max_distance = np.max(distances)

        self.assertLess(
            max_distance,
            1e-4,
            f"Points should be on translated cube surface, max distance: {max_distance:.6f}",
        )

        # Should still cover all faces
        # Translate points back to origin for face classification
        centered_points = result.points - center
        face_counts = classify_points_by_face(centered_points, tolerance=0.01)
        for face_name, count in face_counts.items():
            self.assertGreater(count, 50, f"Face {face_name} should have points even for translated cube")

    def test_parameter_validation_subdivision_level(self):
        """Test that invalid subdivision_level raises ValueError."""
        with self.assertRaises(ValueError):
            PointCloudExtractor(subdivision_level=-1)

        with self.assertRaises(ValueError):
            PointCloudExtractor(subdivision_level=6)

    def test_parameter_validation_resolution(self):
        """Test that invalid resolution raises ValueError."""
        with self.assertRaises(ValueError):
            PointCloudExtractor(resolution=0)

        with self.assertRaises(ValueError):
            PointCloudExtractor(resolution=10001)

    def test_extract_empty_mesh_raises(self):
        """Test that extracting from empty mesh raises ValueError."""
        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)

        # Empty vertices
        with self.assertRaises(ValueError):
            extractor.extract(np.array([], dtype=np.float32).reshape(0, 3), np.array([0, 1, 2], dtype=np.int32))

        # Empty indices
        vertices, _ = create_unit_cube_mesh()
        with self.assertRaises(ValueError):
            extractor.extract(vertices, np.array([], dtype=np.int32))

    def test_extract_invalid_indices_raises(self):
        """Test that invalid indices raise ValueError."""
        vertices, indices = create_unit_cube_mesh()
        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)

        # Indices not multiple of 3
        with self.assertRaises(ValueError):
            extractor.extract(vertices, indices[:5])

        # Out of bounds indices
        bad_indices = indices.copy()
        bad_indices[0] = 100  # Out of bounds
        with self.assertRaises(ValueError):
            extractor.extract(vertices, bad_indices)


@unittest.skipUnless(_cuda_available and OPEN3D_AVAILABLE, "Requires CUDA and Open3D")
class TestSurfaceReconstructor(unittest.TestCase):
    """Test the SurfaceReconstructor class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def test_reconstruct_cube_mesh(self):
        """Test full remeshing pipeline: extract point cloud and reconstruct mesh."""
        vertices, indices = create_unit_cube_mesh()

        # Step 1: Extract point cloud
        # Use slightly higher resolution for better reconstruction quality
        extractor = PointCloudExtractor(subdivision_level=1, resolution=150)
        pointcloud = extractor.extract(vertices, indices)

        self.assertGreater(pointcloud.num_points, 500, "Should extract sufficient points for reconstruction")

        # Step 2: Reconstruct mesh
        reconstructor = SurfaceReconstructor(
            depth=7,  # Reasonable depth for a cube
            downsample_voxel_size="auto",
            simplify_tolerance=None,
            simplify_ratio=None,
            target_triangles=None,
        )
        recon_mesh = reconstructor.reconstruct_from_result(pointcloud, verbose=False)

        # Should produce a valid mesh
        self.assertGreater(recon_mesh.num_vertices, 0, "Should produce vertices")
        self.assertGreater(recon_mesh.num_triangles, 0, "Should produce triangles")

        # Step 3: Validate reconstructed mesh is close to original cube
        distances = compute_distance_to_cube(recon_mesh.vertices, half_extent=0.5)
        max_distance = np.max(distances)
        mean_distance = np.mean(distances)

        # Poisson reconstruction should produce a mesh very close to the original
        # Using 0.03 (3% of cube size) as threshold - still meaningful but achievable
        threshold = 0.03
        self.assertLess(
            max_distance,
            threshold,
            f"Reconstructed mesh vertices should be within {threshold} of original cube surface, "
            f"max distance: {max_distance:.4f}",
        )

        # Mean distance should be much smaller
        self.assertLess(
            mean_distance,
            0.015,
            f"Mean distance should be small, got: {mean_distance:.4f}",
        )

    def test_reconstruct_produces_reasonable_triangle_count(self):
        """Test that reconstruction produces a reasonable number of triangles."""
        vertices, indices = create_unit_cube_mesh()

        extractor = PointCloudExtractor(subdivision_level=1, resolution=100)
        pointcloud = extractor.extract(vertices, indices)

        reconstructor = SurfaceReconstructor(depth=6, downsample_voxel_size="auto")
        recon_mesh = reconstructor.reconstruct_from_result(pointcloud, verbose=False)

        # A cube needs at minimum 12 triangles (2 per face)
        # Poisson reconstruction typically produces more (smoother surface)
        # But it shouldn't be absurdly high for a simple cube
        self.assertGreater(recon_mesh.num_triangles, 12, "Should have at least 12 triangles")
        self.assertLess(recon_mesh.num_triangles, 50000, "Should not have excessive triangles for a simple cube")

    def test_parameter_validation(self):
        """Test that invalid parameters raise ValueError."""
        # Invalid depth
        with self.assertRaises(ValueError):
            SurfaceReconstructor(depth=0)

        # Invalid scale
        with self.assertRaises(ValueError):
            SurfaceReconstructor(scale=-1.0)

        # Invalid density_threshold_quantile
        with self.assertRaises(ValueError):
            SurfaceReconstructor(density_threshold_quantile=1.5)

        # Invalid simplify_ratio
        with self.assertRaises(ValueError):
            SurfaceReconstructor(simplify_ratio=0.0)
        with self.assertRaises(ValueError):
            SurfaceReconstructor(simplify_ratio=1.5)

        # Invalid target_triangles
        with self.assertRaises(ValueError):
            SurfaceReconstructor(target_triangles=0)

        # Invalid simplify_tolerance
        with self.assertRaises(ValueError):
            SurfaceReconstructor(simplify_tolerance=-0.1)

    def test_reconstruct_empty_pointcloud_raises(self):
        """Test that reconstructing from empty point cloud raises ValueError."""
        reconstructor = SurfaceReconstructor(depth=6)

        empty_points = np.array([], dtype=np.float32).reshape(0, 3)
        empty_normals = np.array([], dtype=np.float32).reshape(0, 3)

        with self.assertRaises(ValueError):
            reconstructor.reconstruct(empty_points, empty_normals, verbose=False)


class TestRemeshHelperFunctions(unittest.TestCase):
    """Test helper functions in the remesh module."""

    def test_compute_bounding_sphere_empty_raises(self):
        """Test that empty vertices raise ValueError."""
        with self.assertRaises(ValueError):
            compute_bounding_sphere(np.array([], dtype=np.float32).reshape(0, 3))

    def test_compute_bounding_sphere_single_vertex(self):
        """Test bounding sphere for single vertex."""
        vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        center, radius = compute_bounding_sphere(vertices)

        np.testing.assert_array_almost_equal(center, [1.0, 2.0, 3.0])
        self.assertGreater(radius, 0, "Single vertex should have small positive radius")

    def test_compute_bounding_sphere_cube(self):
        """Test bounding sphere for cube vertices."""
        vertices, _ = create_unit_cube_mesh()
        center, radius = compute_bounding_sphere(vertices)

        # Center should be at origin
        np.testing.assert_array_almost_equal(center, [0.0, 0.0, 0.0], decimal=5)

        # Radius should be distance from origin to corner: sqrt(0.5^2 + 0.5^2 + 0.5^2) = sqrt(0.75)
        expected_radius = np.sqrt(0.75)
        self.assertAlmostEqual(radius, expected_radius, places=5)

    def test_compute_camera_basis_zero_direction_raises(self):
        """Test that zero direction raises ValueError."""
        with self.assertRaises(ValueError):
            compute_camera_basis(np.array([0.0, 0.0, 0.0]))

    def test_compute_camera_basis_produces_orthonormal(self):
        """Test that camera basis produces orthonormal vectors."""
        direction = np.array([1.0, 0.5, 0.3], dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        right, up = compute_camera_basis(direction)

        # Check orthonormality
        self.assertAlmostEqual(np.dot(right, up), 0.0, places=5)
        self.assertAlmostEqual(np.dot(right, direction), 0.0, places=5)
        self.assertAlmostEqual(np.dot(up, direction), 0.0, places=5)

        # Check unit length
        self.assertAlmostEqual(np.linalg.norm(right), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(up), 1.0, places=5)

    def test_compute_camera_basis_multiple_directions(self):
        """Test camera basis for multiple different directions."""
        # Test various directions including edge cases
        directions = [
            [1.0, 0.0, 0.0],  # Along X
            [0.0, 1.0, 0.0],  # Along Y (triggers different world_up)
            [0.0, 0.0, 1.0],  # Along Z
            [1.0, 1.0, 1.0],  # Diagonal
            [-0.5, 0.8, 0.3],  # Arbitrary
        ]

        for dir_vec in directions:
            direction = np.array(dir_vec, dtype=np.float32)
            direction = direction / np.linalg.norm(direction)

            right, up = compute_camera_basis(direction)

            # All should produce orthonormal bases
            self.assertAlmostEqual(
                np.dot(right, up), 0.0, places=4, msg=f"right·up should be 0 for direction {dir_vec}"
            )
            self.assertAlmostEqual(
                np.dot(right, direction), 0.0, places=4, msg=f"right·dir should be 0 for direction {dir_vec}"
            )
            self.assertAlmostEqual(
                np.dot(up, direction), 0.0, places=4, msg=f"up·dir should be 0 for direction {dir_vec}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
