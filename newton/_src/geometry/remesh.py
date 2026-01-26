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
        from newton.geometry import PointCloudExtractor, SurfaceReconstructor

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

from newton._src.geometry.hashtable import HashTable, hashtable_find_or_insert

# -----------------------------------------------------------------------------
# Morton encoding for sparse voxel grid (21 bits per axis = 63 bits total)
# -----------------------------------------------------------------------------

# Offset to handle negative coordinates: shift by 2^20 so range [-2^20, 2^20) maps to [0, 2^21)
VOXEL_COORD_OFFSET = wp.constant(wp.int32(1 << 20))  # 1,048,576
VOXEL_COORD_MASK = wp.constant(wp.uint64(0x1FFFFF))  # 21 bits = 2,097,151


@wp.func
def _split_by_3(x: wp.uint64) -> wp.uint64:
    """Spread 21-bit integer into 63 bits with 2 zeros between each bit (for Morton encoding)."""
    # x = ---- ---- ---- ---- ---- ---- ---x xxxx xxxx xxxx xxxx xxxx (21 bits)
    x = x & wp.uint64(0x1FFFFF)  # Mask to 21 bits
    # Spread bits apart using magic numbers (interleave with zeros)
    x = (x | (x << wp.uint64(32))) & wp.uint64(0x1F00000000FFFF)
    x = (x | (x << wp.uint64(16))) & wp.uint64(0x1F0000FF0000FF)
    x = (x | (x << wp.uint64(8))) & wp.uint64(0x100F00F00F00F00F)
    x = (x | (x << wp.uint64(4))) & wp.uint64(0x10C30C30C30C30C3)
    x = (x | (x << wp.uint64(2))) & wp.uint64(0x1249249249249249)
    return x


@wp.func
def morton_encode_3d(ix: wp.int32, iy: wp.int32, iz: wp.int32) -> wp.uint64:
    """Encode 3 signed integers into a 63-bit Morton code.

    Each coordinate is shifted by VOXEL_COORD_OFFSET to handle negatives,
    then the 21-bit values are interleaved: z takes bits 2,5,8,..., y takes 1,4,7,..., x takes 0,3,6,...
    """
    # Shift to unsigned range
    ux = wp.uint64(ix + VOXEL_COORD_OFFSET) & VOXEL_COORD_MASK
    uy = wp.uint64(iy + VOXEL_COORD_OFFSET) & VOXEL_COORD_MASK
    uz = wp.uint64(iz + VOXEL_COORD_OFFSET) & VOXEL_COORD_MASK
    # Interleave bits
    return _split_by_3(ux) | (_split_by_3(uy) << wp.uint64(1)) | (_split_by_3(uz) << wp.uint64(2))


@wp.func
def compute_voxel_key(
    point: wp.vec3,
    inv_voxel_size: wp.float32,
) -> wp.uint64:
    """Compute Morton-encoded voxel key for a point."""
    # Quantize to integer voxel coordinates
    ix = wp.int32(wp.floor(point[0] * inv_voxel_size))
    iy = wp.int32(wp.floor(point[1] * inv_voxel_size))
    iz = wp.int32(wp.floor(point[2] * inv_voxel_size))
    return morton_encode_3d(ix, iy, iz)


# -----------------------------------------------------------------------------
# VoxelHashGrid - sparse voxel grid with online accumulation
# -----------------------------------------------------------------------------


@wp.kernel
def _accumulate_point_kernel(
    point: wp.vec3,
    normal: wp.vec3,
    inv_voxel_size: wp.float32,
    # Hash table arrays
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    # Accumulator arrays
    sum_positions_x: wp.array(dtype=wp.float32),
    sum_positions_y: wp.array(dtype=wp.float32),
    sum_positions_z: wp.array(dtype=wp.float32),
    sum_normals_x: wp.array(dtype=wp.float32),
    sum_normals_y: wp.array(dtype=wp.float32),
    sum_normals_z: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
):
    """Accumulate a single point into the voxel grid (for testing)."""
    key = compute_voxel_key(point, inv_voxel_size)
    idx = hashtable_find_or_insert(key, keys, active_slots)
    if idx >= 0:
        wp.atomic_add(sum_positions_x, idx, point[0])
        wp.atomic_add(sum_positions_y, idx, point[1])
        wp.atomic_add(sum_positions_z, idx, point[2])
        wp.atomic_add(sum_normals_x, idx, normal[0])
        wp.atomic_add(sum_normals_y, idx, normal[1])
        wp.atomic_add(sum_normals_z, idx, normal[2])
        wp.atomic_add(counts, idx, 1)


@wp.kernel
def _finalize_voxels_kernel(
    active_slots: wp.array(dtype=wp.int32),
    num_active: wp.int32,
    # Accumulator arrays (input)
    sum_positions_x: wp.array(dtype=wp.float32),
    sum_positions_y: wp.array(dtype=wp.float32),
    sum_positions_z: wp.array(dtype=wp.float32),
    sum_normals_x: wp.array(dtype=wp.float32),
    sum_normals_y: wp.array(dtype=wp.float32),
    sum_normals_z: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    # Output arrays
    out_points: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Finalize voxel averages and write to output arrays."""
    tid = wp.tid()
    if tid >= num_active:
        return

    idx = active_slots[tid]
    count = counts[idx]
    if count <= 0:
        return

    inv_count = 1.0 / wp.float32(count)

    # Average position
    avg_pos = wp.vec3(
        sum_positions_x[idx] * inv_count,
        sum_positions_y[idx] * inv_count,
        sum_positions_z[idx] * inv_count,
    )

    # Average and normalize normal
    avg_normal = wp.vec3(
        sum_normals_x[idx],
        sum_normals_y[idx],
        sum_normals_z[idx],
    )
    normal_len = wp.length(avg_normal)
    if normal_len > 1e-8:
        avg_normal = avg_normal / normal_len
    else:
        avg_normal = wp.vec3(0.0, 1.0, 0.0)  # Fallback

    out_points[tid] = avg_pos
    out_normals[tid] = avg_normal


class VoxelHashGrid:
    """Sparse voxel grid with online accumulation of positions and normals.

    Uses a GPU hash table to map voxel coordinates (Morton-encoded) to
    accumulator slots. Points and normals are accumulated using atomic
    operations, allowing fully parallel insertion from multiple threads.

    This is useful for voxel-based downsampling of point clouds directly
    on the GPU without intermediate storage.

    Args:
        capacity: Maximum number of unique voxels. Rounded up to power of two.
        voxel_size: Size of each cubic voxel.
        device: Warp device for computation.

    Example:
        >>> grid = VoxelHashGrid(capacity=1_000_000, voxel_size=0.01)
        >>> # Accumulate points (typically done in a kernel)
        >>> # ...
        >>> points, normals, count = grid.finalize()
    """

    def __init__(
        self,
        capacity: int,
        voxel_size: float,
        device: str | None = None,
    ):
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")

        self.voxel_size = voxel_size
        self.inv_voxel_size = 1.0 / voxel_size
        self.device = device

        # Hash table for voxel keys
        self._hashtable = HashTable(capacity, device=device)
        self.capacity = self._hashtable.capacity

        # Accumulator arrays (separate x/y/z for atomic_add compatibility)
        self.sum_positions_x = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_positions_y = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_positions_z = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_normals_x = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_normals_y = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_normals_z = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.counts = wp.zeros(self.capacity, dtype=wp.int32, device=device)

    @property
    def keys(self) -> wp.array:
        """Hash table keys array (for use in kernels)."""
        return self._hashtable.keys

    @property
    def active_slots(self) -> wp.array:
        """Active slots tracking array (for use in kernels)."""
        return self._hashtable.active_slots

    def clear(self):
        """Clear all voxels and reset accumulators."""
        self._hashtable.clear()
        self.sum_positions_x.zero_()
        self.sum_positions_y.zero_()
        self.sum_positions_z.zero_()
        self.sum_normals_x.zero_()
        self.sum_normals_y.zero_()
        self.sum_normals_z.zero_()
        self.counts.zero_()

    def get_num_voxels(self) -> int:
        """Get the current number of occupied voxels."""
        return int(self._hashtable.active_slots.numpy()[self.capacity])

    def finalize(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Finalize accumulation and return averaged points and normals.

        Computes the average position and normalized normal for each occupied
        voxel and returns the results as numpy arrays.

        Returns:
            Tuple of (points, normals, num_points) where:
            - points: (N, 3) float32 array of averaged positions
            - normals: (N, 3) float32 array of normalized normals
            - num_points: number of occupied voxels
        """
        num_active = self.get_num_voxels()
        if num_active == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                0,
            )

        # Allocate output buffers
        out_points = wp.zeros(num_active, dtype=wp.vec3, device=self.device)
        out_normals = wp.zeros(num_active, dtype=wp.vec3, device=self.device)

        # Launch finalization kernel
        wp.launch(
            _finalize_voxels_kernel,
            dim=num_active,
            inputs=[
                self.active_slots,
                num_active,
                self.sum_positions_x,
                self.sum_positions_y,
                self.sum_positions_z,
                self.sum_normals_x,
                self.sum_normals_y,
                self.sum_normals_z,
                self.counts,
                out_points,
                out_normals,
            ],
            device=self.device,
        )

        wp.synchronize()

        return (
            out_points.numpy(),
            out_normals.numpy(),
            num_active,
        )


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

    Raises:
        ValueError: If vertices array is empty.
    """
    if len(vertices) == 0:
        raise ValueError("Cannot compute bounding sphere for empty vertex array")

    # Start with axis-aligned bounding box center
    min_pt = np.min(vertices, axis=0)
    max_pt = np.max(vertices, axis=0)
    center = (min_pt + max_pt) / 2.0

    # Compute radius as max distance from center
    distances = np.linalg.norm(vertices - center, axis=1)
    radius = float(np.max(distances))

    # Handle single-vertex case: use small positive radius
    if radius == 0.0:
        radius = 1e-6

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

    Raises:
        ValueError: If direction vector has zero or near-zero length.
    """
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        raise ValueError("Direction vector has zero or near-zero length")
    direction = direction / norm

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
    # Voxel hash grid parameters
    inv_voxel_size: wp.float32,
    # Hash table arrays
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    # Accumulator arrays
    sum_positions_x: wp.array(dtype=wp.float32),
    sum_positions_y: wp.array(dtype=wp.float32),
    sum_positions_z: wp.array(dtype=wp.float32),
    sum_normals_x: wp.array(dtype=wp.float32),
    sum_normals_y: wp.array(dtype=wp.float32),
    sum_normals_z: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
):
    """Raycast kernel for orthographic projection with direct voxel accumulation.

    Each thread handles one pixel. Rays are shot parallel to cam_dir from a grid
    defined by cam_right and cam_up. Hits are accumulated directly into a sparse
    voxel hash grid for memory-efficient point cloud extraction.
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

        # Accumulate into voxel hash grid
        key = compute_voxel_key(hit_point, inv_voxel_size)
        idx = hashtable_find_or_insert(key, keys, active_slots)
        if idx >= 0:
            wp.atomic_add(sum_positions_x, idx, hit_point[0])
            wp.atomic_add(sum_positions_y, idx, hit_point[1])
            wp.atomic_add(sum_positions_z, idx, hit_point[2])
            wp.atomic_add(sum_normals_x, idx, normal[0])
            wp.atomic_add(sum_normals_y, idx, normal[1])
            wp.atomic_add(sum_normals_z, idx, normal[2])
            wp.atomic_add(counts, idx, 1)


class PointCloudExtractor:
    """Extract dense point clouds with normals from triangle meshes.

    Uses multi-view orthographic raycasting from directions distributed on
    an icosphere to capture the complete surface of a mesh. Normals are
    guaranteed to be consistent (always pointing outward toward the camera).

    Points are accumulated directly into a sparse voxel hash grid during
    raycasting, providing built-in downsampling and dramatically reducing
    memory usage compared to storing all ray hits.

    Args:
        subdivision_level: Icosahedron subdivision level (0-5). Level 0 gives 20 views,
            level 1 gives 80 views, level 2 gives 320 views, etc. Higher levels provide
            better coverage.
        resolution: Pixel resolution of the orthographic camera (resolution x resolution).
            Must be between 1 and 10000.
        voxel_size: Size of voxels for point accumulation. If None (default), automatically
            computed as 0.1% of the mesh bounding sphere diameter. Smaller values give
            denser point clouds but require more memory.
        max_voxels: Maximum number of unique voxels (hash table capacity). Default 2^20
            (~1 million). Increase if you expect very dense sampling.
        device: Warp device to use for computation.
        seed: Random seed for camera roll angles. Each camera is rotated by a random
            angle around its view direction to reduce sample correlation and improve
            coverage uniformity. Set to None for non-deterministic behavior.

    Note:
        Memory usage is dominated by the voxel hash grid, which scales with
        ``max_voxels`` (~28 bytes per voxel slot), not with ``resolution^2 * num_views``.
        This makes high-resolution extraction practical even on memory-constrained systems.

    Example:
        >>> extractor = PointCloudExtractor(subdivision_level=2, resolution=1000)
        >>> result = extractor.extract(vertices, indices)
        >>> print(f"Extracted {result.num_points} points with normals")
    """

    def __init__(
        self,
        subdivision_level: int = 1,
        resolution: int = 1000,
        voxel_size: float | None = None,
        max_voxels: int = 1 << 20,
        device: str | None = None,
        seed: int | None = 42,
    ):
        # Validate parameters
        if subdivision_level < 0 or subdivision_level > 5:
            raise ValueError(f"subdivision_level must be between 0 and 5 (inclusive), got {subdivision_level}")
        if resolution < 1 or resolution > 10000:
            raise ValueError(f"resolution must be between 1 and 10000 (inclusive), got {resolution}")
        if voxel_size is not None and voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if max_voxels < 1:
            raise ValueError(f"max_voxels must be >= 1, got {max_voxels}")

        self.subdivision_level = subdivision_level
        self.resolution = resolution
        self.voxel_size = voxel_size  # None means auto-compute
        self.max_voxels = max_voxels
        self.device = device if device is not None else wp.get_device()
        self.seed = seed

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

        Raises:
            ValueError: If vertices or indices are empty, or indices are invalid.
        """
        # Ensure correct shapes
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32).flatten()

        # Validate inputs
        if len(vertices) == 0:
            raise ValueError("Vertices array cannot be empty")
        if len(indices) == 0:
            raise ValueError("Indices array cannot be empty")
        if len(indices) % 3 != 0:
            raise ValueError(f"Indices length must be a multiple of 3, got {len(indices)}")
        if np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError(
                f"Indices must be in range [0, {len(vertices)}), got range [{indices.min()}, {indices.max()}]"
            )

        # Compute bounding sphere
        center, radius = compute_bounding_sphere(vertices)
        padded_radius = radius * padding_factor

        # Compute pixel size to cover the bounding sphere diameter
        pixel_size = (2.0 * padded_radius) / self.resolution

        # Maximum ray distance (diameter of bounding sphere with padding)
        max_ray_dist = 2.0 * padded_radius * 1.5

        # Compute voxel size (auto or user-specified)
        if self.voxel_size is None:
            # Auto: 0.1% of bounding sphere diameter
            voxel_size = (2.0 * radius) * 0.001
            # Guard against zero (single point)
            if voxel_size == 0.0:
                voxel_size = 1e-6
        else:
            voxel_size = self.voxel_size

        # Create sparse voxel hash grid for accumulation
        voxel_grid = VoxelHashGrid(
            capacity=self.max_voxels,
            voxel_size=voxel_size,
            device=self.device,
        )

        # Create Warp mesh
        wp_vertices = wp.array(vertices, dtype=wp.vec3, device=self.device)
        wp_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)

        # Create random generator for camera roll angles
        rng = np.random.default_rng(self.seed)

        # Cast rays from each camera direction, accumulating into voxel grid
        for i in range(self.num_views):
            direction = self.directions[i]
            right, up = compute_camera_basis(direction)

            # Apply random rotation around view direction (camera roll)
            # This reduces sample correlation and improves coverage uniformity
            theta = rng.uniform(0, 2 * np.pi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            right_rot = cos_theta * right + sin_theta * up
            up_rot = cos_theta * up - sin_theta * right
            right, up = right_rot, up_rot

            # Camera origin is at bounding sphere center, offset back along view direction
            cam_origin = center - direction * padded_radius

            # Launch kernel - accumulates directly into voxel grid
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
                    float(voxel_grid.inv_voxel_size),
                    voxel_grid.keys,
                    voxel_grid.active_slots,
                    voxel_grid.sum_positions_x,
                    voxel_grid.sum_positions_y,
                    voxel_grid.sum_positions_z,
                    voxel_grid.sum_normals_x,
                    voxel_grid.sum_normals_y,
                    voxel_grid.sum_normals_z,
                    voxel_grid.counts,
                ],
                device=self.device,
            )

        # Finalize voxel grid to get averaged points and normals
        points_np, normals_np, num_points = voxel_grid.finalize()

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

    Note:
        When used with PointCloudExtractor, the point cloud is already downsampled
        via the built-in voxel hash grid accumulation. No additional downsampling
        is needed.

    Args:
        depth: Octree depth for Poisson reconstruction (higher = more detail, slower).
            Default is 10, which provides good detail.
        scale: Scale factor for the reconstruction bounding box. Default 1.1.
        linear_fit: Use linear interpolation for iso-surface extraction. Default False.
        density_threshold_quantile: Quantile for removing low-density vertices
            (boundary artifacts). Default 0.01 removes bottom 1%.
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
        simplify_ratio: float | None = None,
        target_triangles: int | None = None,
        simplify_tolerance: float | None = None,
    ):
        # Validate parameters
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if not (0.0 <= density_threshold_quantile <= 1.0):
            raise ValueError(f"density_threshold_quantile must be in [0, 1], got {density_threshold_quantile}")
        if simplify_ratio is not None and (simplify_ratio <= 0 or simplify_ratio > 1):
            raise ValueError(f"simplify_ratio must be in (0, 1], got {simplify_ratio}")
        if target_triangles is not None and target_triangles < 1:
            raise ValueError(f"target_triangles must be >= 1, got {target_triangles}")
        if simplify_tolerance is not None and simplify_tolerance < 0:
            raise ValueError(f"simplify_tolerance must be >= 0, got {simplify_tolerance}")

        self.depth = depth
        self.scale = scale
        self.linear_fit = linear_fit
        self.density_threshold_quantile = density_threshold_quantile
        self.simplify_ratio = simplify_ratio
        self.target_triangles = target_triangles
        self.simplify_tolerance = simplify_tolerance

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

        # Validate inputs
        if len(points) == 0:
            raise ValueError("Cannot reconstruct from empty point cloud")

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
            if num_triangles_before > 0:
                reduction = 100 * (1 - len(indices) // 3 / num_triangles_before)
                print(
                    f"Simplified mesh: {len(vertices)} vertices, {len(indices) // 3} triangles ({reduction:.1f}% reduction)"
                )
            else:
                print(f"Simplified mesh: {len(vertices)} vertices, {len(indices) // 3} triangles")

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
