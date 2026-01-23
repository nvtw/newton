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

"""Point cloud extraction and surface reconstruction for mesh repair.

This module provides utilities to extract dense point clouds with reliable surface
normals from triangle meshes by shooting parallel rays from multiple viewpoints
arranged on an icosphere. The point cloud can then be used to reconstruct a clean,
watertight mesh using Poisson surface reconstruction.

Requirements:
    - Point cloud extraction (PointCloudExtractor): Only requires Warp (included with Newton)
    - Surface reconstruction (SurfaceReconstructor): Requires Open3D (`pip install open3d`)

This is useful for repairing meshes with:
- Inconsistent or flipped triangle winding
- Missing or incorrect vertex normals
- Non-manifold geometry
- Holes or self-intersections

Example:
    Remesh a problematic mesh to get a clean, watertight version::

        import numpy as np
        from newton._src.geometry.remesh import (
            PointCloudExtractor,
            SurfaceReconstructor,
        )

        # Load your mesh (vertices: Nx3, indices: Mx3 or flattened)
        vertices = np.array(...)  # your mesh vertices
        indices = np.array(...)  # your mesh triangle indices

        # Step 1: Extract point cloud with reliable normals
        # More subdivision = more views = better coverage (but slower)
        extractor = PointCloudExtractor(subdivision_level=2, resolution=1000)
        pointcloud = extractor.extract(vertices, indices)
        print(f"Extracted {pointcloud.num_points} points")

        # Step 2: Reconstruct clean mesh using Poisson reconstruction
        # Higher depth = more detail, simplify_tolerance controls decimation
        reconstructor = SurfaceReconstructor(
            depth=10,
            simplify_tolerance=1e-7,  # fraction of mesh diagonal
        )
        clean_mesh = reconstructor.reconstruct_from_result(pointcloud)
        print(f"Reconstructed {clean_mesh.num_triangles} triangles")

        # Use the clean mesh
        new_vertices = clean_mesh.vertices  # (N, 3) float32
        new_indices = clean_mesh.indices  # (M,) int32, flattened
"""

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class PointCloudResult:
    """Result of point cloud extraction.

    Attributes:
        points: World-space intersection points (N, 3).
        normals: World-space surface normals at each point (N, 3).
            Normals are guaranteed to point toward the camera that captured them.
        num_points: Total number of valid points extracted.
    """

    points: np.ndarray
    normals: np.ndarray
    num_points: int


def compute_bounding_sphere(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute a bounding sphere for a set of vertices.

    Uses Ritter's algorithm for a reasonable approximation.

    Args:
        vertices: (N, 3) array of vertex positions.

    Returns:
        Tuple of (center, radius) where center is (3,) array.
    """
    # Start with axis-aligned bounding box center
    min_pt = np.min(vertices, axis=0)
    max_pt = np.max(vertices, axis=0)
    center = (min_pt + max_pt) / 2.0

    # Compute radius as max distance from center
    distances = np.linalg.norm(vertices - center, axis=1)
    radius = np.max(distances)

    return center, radius


def create_icosahedron_directions(subdivision_level: int = 2) -> np.ndarray:
    """Create camera directions from subdivided icosahedron face centers.

    An icosahedron has 20 faces. Each subdivision level multiplies the face
    count by 4. The camera directions are the normalized vectors from origin
    to each face center.

    Args:
        subdivision_level: Number of subdivision iterations (0 = 20 faces,
            1 = 80 faces, 2 = 320 faces, etc.).

    Returns:
        (N, 3) array of unit direction vectors, one per face.
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedron vertices (normalized)
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # Icosahedron faces (20 triangles)
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # Subdivide faces
    verts_list = verts.tolist()

    for _ in range(subdivision_level):
        new_faces = []
        edge_midpoints = {}

        for face in faces:
            v0, v1, v2 = face

            # Get or create midpoints for each edge
            midpoint_indices = []
            for i0, i1 in [(v0, v1), (v1, v2), (v2, v0)]:
                edge = (min(i0, i1), max(i0, i1))
                if edge in edge_midpoints:
                    midpoint_indices.append(edge_midpoints[edge])
                else:
                    # Create new vertex at midpoint, projected to unit sphere
                    p0 = np.array(verts_list[i0])
                    p1 = np.array(verts_list[i1])
                    midpoint = (p0 + p1) / 2.0
                    midpoint = midpoint / np.linalg.norm(midpoint)

                    new_idx = len(verts_list)
                    verts_list.append(midpoint.tolist())
                    edge_midpoints[edge] = new_idx
                    midpoint_indices.append(new_idx)

            m01, m12, m20 = midpoint_indices

            # Create 4 new faces
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])

        faces = np.array(new_faces, dtype=np.int32)

    verts = np.array(verts_list, dtype=np.float64)

    # Compute face centers as camera directions
    face_centers = np.zeros((len(faces), 3), dtype=np.float64)
    for i, face in enumerate(faces):
        center = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3.0
        face_centers[i] = center / np.linalg.norm(center)

    return face_centers.astype(np.float32)


def compute_camera_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute orthonormal camera basis vectors from a view direction.

    Args:
        direction: Unit direction vector the camera is looking along.

    Returns:
        Tuple of (right, up) unit vectors forming an orthonormal basis with direction.
    """
    direction = direction / np.linalg.norm(direction)

    # Choose an arbitrary up vector that's not parallel to direction
    if abs(direction[1]) < 0.9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = np.cross(world_up, direction)
    right = right / np.linalg.norm(right)

    up = np.cross(direction, right)
    up = up / np.linalg.norm(up)

    return right, up


@wp.kernel
def raycast_orthographic_kernel(
    # Mesh
    mesh_id: wp.uint64,
    # Camera parameters
    cam_origin: wp.vec3,
    cam_dir: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    pixel_size: wp.float32,
    resolution: wp.int32,
    max_ray_dist: wp.float32,
    # Output buffers
    out_points: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
    out_count: wp.array(dtype=wp.int32),
    max_points: wp.int32,
):
    """Raycast kernel for orthographic projection from a single camera view.

    Each thread handles one pixel. Rays are shot parallel to cam_dir from a grid
    defined by cam_right and cam_up.
    """
    px, py = wp.tid()

    if px >= resolution or py >= resolution:
        return

    # Compute ray origin on the image plane
    # Center the grid around the camera origin
    half_res = wp.float32(resolution) * 0.5
    offset_x = (wp.float32(px) - half_res + 0.5) * pixel_size
    offset_y = (wp.float32(py) - half_res + 0.5) * pixel_size

    ray_origin = cam_origin + cam_right * offset_x + cam_up * offset_y
    ray_direction = cam_dir

    # Query mesh intersection
    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_ray_dist)

    if query.result:
        # Compute hit point
        hit_point = ray_origin + ray_direction * query.t

        # Get surface normal - ensure it points toward camera (opposite to ray direction)
        normal = query.normal
        if wp.dot(normal, ray_direction) > 0.0:
            normal = -normal
        normal = wp.normalize(normal)

        # Atomically append to output buffers
        idx = wp.atomic_add(out_count, 0, 1)
        if idx < max_points:
            out_points[idx] = hit_point
            out_normals[idx] = normal


class PointCloudExtractor:
    """Extract dense point clouds with normals from triangle meshes.

    Uses multi-view orthographic raycasting from directions distributed on
    an icosphere to capture the complete surface of a mesh. Normals are
    guaranteed to be consistent (always pointing outward toward the camera).

    Args:
        subdivision_level: Icosahedron subdivision level. Level 0 gives 20 views,
            level 1 gives 80 views, level 2 gives 320 views, etc.
        resolution: Pixel resolution of the orthographic camera (resolution x resolution).
        device: Warp device to use for computation.

    Example:
        >>> extractor = PointCloudExtractor(subdivision_level=2, resolution=1000)
        >>> result = extractor.extract(vertices, indices)
        >>> print(f"Extracted {result.num_points} points with normals")
    """

    def __init__(
        self,
        subdivision_level: int = 1,
        resolution: int = 1000,
        device: str | None = None,
    ):
        self.subdivision_level = subdivision_level
        self.resolution = resolution
        self.device = device if device is not None else wp.get_device()

        # Pre-compute camera directions
        self.directions = create_icosahedron_directions(subdivision_level)
        self.num_views = len(self.directions)

    def extract(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        padding_factor: float = 1.1,
    ) -> PointCloudResult:
        """Extract point cloud from a triangle mesh.

        Args:
            vertices: (N, 3) array of vertex positions.
            indices: (M,) or (M/3, 3) array of triangle indices.
            padding_factor: Multiplier for bounding sphere radius to ensure
                rays start outside the mesh.

        Returns:
            PointCloudResult containing extracted points and normals.
        """
        # Ensure correct shapes
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32).flatten()

        # Compute bounding sphere
        center, radius = compute_bounding_sphere(vertices)
        padded_radius = radius * padding_factor

        # Compute pixel size to cover the bounding sphere diameter
        pixel_size = (2.0 * padded_radius) / self.resolution

        # Maximum ray distance (diameter of bounding sphere with padding)
        max_ray_dist = 2.0 * padded_radius * 1.5

        # Create Warp mesh
        wp_vertices = wp.array(vertices, dtype=wp.vec3, device=self.device)
        wp_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)

        # Estimate max points (all pixels from all views could potentially hit)
        max_points_per_view = self.resolution * self.resolution
        max_total_points = max_points_per_view * self.num_views

        # Allocate output buffers
        out_points = wp.zeros(max_total_points, dtype=wp.vec3, device=self.device)
        out_normals = wp.zeros(max_total_points, dtype=wp.vec3, device=self.device)
        out_count = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Cast rays from each camera direction
        for i in range(self.num_views):
            direction = self.directions[i]
            right, up = compute_camera_basis(direction)

            # Camera origin is at bounding sphere center, offset back along view direction
            cam_origin = center - direction * padded_radius

            # Launch kernel
            wp.launch(
                kernel=raycast_orthographic_kernel,
                dim=(self.resolution, self.resolution),
                inputs=[
                    mesh.id,
                    wp.vec3(cam_origin[0], cam_origin[1], cam_origin[2]),
                    wp.vec3(direction[0], direction[1], direction[2]),
                    wp.vec3(right[0], right[1], right[2]),
                    wp.vec3(up[0], up[1], up[2]),
                    float(pixel_size),
                    self.resolution,
                    float(max_ray_dist),
                    out_points,
                    out_normals,
                    out_count,
                    max_total_points,
                ],
                device=self.device,
            )

        # Synchronize and get results
        wp.synchronize()

        num_points = int(out_count.numpy()[0])
        num_points = min(num_points, max_total_points)

        # Copy results to numpy
        points_np = out_points.numpy()[:num_points]
        normals_np = out_normals.numpy()[:num_points]

        return PointCloudResult(
            points=points_np,
            normals=normals_np,
            num_points=num_points,
        )


@dataclass
class ReconstructedMesh:
    """Result of surface reconstruction.

    Attributes:
        vertices: (N, 3) array of vertex positions.
        indices: (M,) array of triangle indices (flattened).
        num_vertices: Number of vertices.
        num_triangles: Number of triangles.
    """

    vertices: np.ndarray
    indices: np.ndarray
    num_vertices: int
    num_triangles: int


class SurfaceReconstructor:
    """Reconstruct triangle meshes from point clouds using Poisson reconstruction.

    Uses Open3D's implementation of Screened Poisson Surface Reconstruction.
    Includes optional voxel downsampling to remove duplicates and ensure
    uniform point density before reconstruction.

    Args:
        depth: Octree depth for Poisson reconstruction (higher = more detail, slower).
            Default is 10, which provides good detail.
        scale: Scale factor for the reconstruction bounding box. Default 1.1.
        linear_fit: Use linear interpolation for iso-surface extraction. Default False.
        density_threshold_quantile: Quantile for removing low-density vertices
            (boundary artifacts). Default 0.01 removes bottom 1%.
        downsample_voxel_size: Voxel size for downsampling before reconstruction.
            If None, no downsampling is performed. If "auto", computes based on
            point cloud extent (~0.1% of largest dimension).
        simplify_ratio: Target ratio to reduce triangle count (e.g., 0.1 = keep 10%).
            If None, no simplification is performed. Uses quadric decimation which
            preserves shape well and removes unnecessary triangles in flat areas.
        target_triangles: Target number of triangles after simplification.
            Overrides simplify_ratio if both are set.
        simplify_tolerance: Maximum geometric error allowed during simplification,
            as a fraction of the mesh bounding box diagonal (e.g., 0.0000001 = 0.00001% of diagonal).
            Only coplanar/nearly-coplanar triangles within this tolerance are merged.
            The mesh keeps all triangles it needs to stay within tolerance.
            This is the recommended option for quality-preserving simplification.
            Overrides simplify_ratio and target_triangles if set.

    Example:
        >>> extractor = PointCloudExtractor(subdivision_level=2, resolution=1000)
        >>> pointcloud = extractor.extract(vertices, indices)
        >>> reconstructor = SurfaceReconstructor(depth=10, simplify_tolerance=1e-7)
        >>> mesh = reconstructor.reconstruct_from_result(pointcloud)
        >>> print(f"Reconstructed {mesh.num_triangles} triangles")
    """

    def __init__(
        self,
        depth: int = 10,
        scale: float = 1.1,
        linear_fit: bool = False,
        density_threshold_quantile: float = 0.01,
        downsample_voxel_size: float | str | None = "auto",
        simplify_ratio: float | None = None,
        target_triangles: int | None = None,
        simplify_tolerance: float | None = None,
    ):
        self.depth = depth
        self.scale = scale
        self.linear_fit = linear_fit
        self.density_threshold_quantile = density_threshold_quantile
        self.downsample_voxel_size = downsample_voxel_size
        self.simplify_ratio = simplify_ratio
        self.target_triangles = target_triangles
        self.simplify_tolerance = simplify_tolerance

    def _downsample(self, points: np.ndarray, normals: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Downsample point cloud using voxel grid filtering."""
        import open3d as o3d  # noqa: PLC0415

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        pcd_down = pcd.voxel_down_sample(voxel_size)

        down_points = np.asarray(pcd_down.points, dtype=np.float32)
        down_normals = np.asarray(pcd_down.normals, dtype=np.float32)

        # Re-normalize normals (voxel downsampling averages them)
        norms = np.linalg.norm(down_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        down_normals = down_normals / norms

        return down_points, down_normals

    def reconstruct(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        verbose: bool = True,
    ) -> ReconstructedMesh:
        """Reconstruct a triangle mesh from a point cloud.

        Args:
            points: (N, 3) array of point positions.
            normals: (N, 3) array of surface normals (should be unit length).
            verbose: Print progress information.

        Returns:
            ReconstructedMesh containing vertices and triangle indices.
        """
        import open3d as o3d  # noqa: PLC0415

        points = np.asarray(points, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        original_count = len(points)

        # Downsample if requested
        if self.downsample_voxel_size is not None:
            if self.downsample_voxel_size == "auto":
                # Auto-compute: ~0.1% of largest dimension (keeps more detail)
                extent = np.max(points, axis=0) - np.min(points, axis=0)
                voxel_size = float(np.max(extent)) * 0.001
            else:
                voxel_size = float(self.downsample_voxel_size)

            if verbose:
                print(f"Downsampling point cloud (voxel_size={voxel_size:.6f})...")

            points, normals = self._downsample(points, normals, voxel_size)

            if verbose:
                ratio = 100 * len(points) / original_count
                print(f"Downsampled: {original_count} -> {len(points)} points ({ratio:.1f}%)")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        # Run Poisson reconstruction
        if verbose:
            print(f"Running Poisson reconstruction (depth={self.depth})...")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.depth,
            scale=self.scale,
            linear_fit=self.linear_fit,
        )

        # Remove low-density vertices (boundary artifacts)
        if self.density_threshold_quantile > 0:
            densities = np.asarray(densities)
            threshold = np.quantile(densities, self.density_threshold_quantile)
            vertices_to_remove = densities < threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        num_triangles_before = len(mesh.triangles)

        if verbose:
            print(f"Reconstructed mesh: {len(mesh.vertices)} vertices, {num_triangles_before} triangles")

        # Simplify mesh using quadric decimation (preserves shape, reduces flat areas)
        if self.simplify_tolerance is not None:
            # Error-based: aggressively target 1 triangle, but stop when error exceeds tolerance
            # This only removes triangles that are truly redundant (coplanar within tolerance)
            # Scale tolerance by mesh bounding box diagonal to make it scale-independent
            bbox = mesh.get_axis_aligned_bounding_box()
            diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            # QEM uses squared distances, so square the tolerance
            absolute_tolerance = (self.simplify_tolerance * diagonal) ** 2
            if verbose:
                print(
                    f"Simplifying mesh (tolerance={self.simplify_tolerance} = {self.simplify_tolerance * diagonal:.6f} absolute, diagonal={diagonal:.4f})..."
                )
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=1,
                maximum_error=absolute_tolerance,
            )
        elif self.target_triangles is not None:
            target = self.target_triangles
            if verbose:
                print(f"Simplifying mesh to {target} triangles...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        elif self.simplify_ratio is not None:
            target = int(num_triangles_before * self.simplify_ratio)
            if verbose:
                print(f"Simplifying mesh to {self.simplify_ratio:.1%} ({target} triangles)...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)

        # Extract results
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        indices = np.asarray(mesh.triangles, dtype=np.int32).flatten()

        if verbose and (
            self.simplify_tolerance is not None or self.target_triangles is not None or self.simplify_ratio is not None
        ):
            reduction = 100 * (1 - len(indices) // 3 / num_triangles_before)
            print(
                f"Simplified mesh: {len(vertices)} vertices, {len(indices) // 3} triangles ({reduction:.1f}% reduction)"
            )

        return ReconstructedMesh(
            vertices=vertices,
            indices=indices,
            num_vertices=len(vertices),
            num_triangles=len(indices) // 3,
        )

    def reconstruct_from_result(
        self,
        result: PointCloudResult,
        verbose: bool = True,
    ) -> ReconstructedMesh:
        """Reconstruct a triangle mesh from a PointCloudResult.

        Convenience method that extracts points and normals from the result.

        Args:
            result: PointCloudResult from extract_pointcloud or PointCloudExtractor.
            verbose: Print progress information.

        Returns:
            ReconstructedMesh containing vertices and triangle indices.
        """
        return self.reconstruct(
            result.points[: result.num_points],
            result.normals[: result.num_points],
            verbose=verbose,
        )
