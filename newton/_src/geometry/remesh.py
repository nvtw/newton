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

This module provides GPU-accelerated utilities to extract dense point clouds with
reliable surface normals from triangle meshes. The extraction uses multi-view
orthographic raycasting from camera directions distributed on an icosphere, with
online voxel-based downsampling for memory efficiency. Optional secondary "cavity
cameras" improve coverage of deep cavities and occluded regions.

The point cloud can then be used to reconstruct a clean, watertight mesh using
Poisson surface reconstruction.

Key features:
    - GPU-accelerated raycasting using Warp
    - Online downsampling via sparse voxel hash grid (Morton-encoded keys)
    - Random camera roll and randomized processing order to reduce sampling bias
    - Optional cavity cameras for improved coverage of occluded regions
    - Probability-scaled selection of cavity candidates (favors deeper hits)
    - Consistent outward-facing normals

Requirements:
    - Point cloud extraction (PointCloudExtractor): Only requires Warp (included with Newton)
    - Surface reconstruction (SurfaceReconstructor): Requires Open3D (`pip install open3d`)

This is useful for repairing meshes with:
    - Inconsistent or flipped triangle winding
    - Missing or incorrect vertex normals
    - Non-manifold geometry
    - Holes or self-intersections
    - Deep cavities that are hard to capture from external viewpoints

Example:
    Remesh a problematic mesh to get a clean, watertight version::

        import numpy as np
        from newton.geometry import PointCloudExtractor, SurfaceReconstructor

        # Load your mesh (vertices: Nx3, indices: Mx3 or flattened)
        vertices = np.array(...)  # your mesh vertices
        indices = np.array(...)  # your mesh triangle indices

        # Step 1: Extract point cloud with reliable normals
        # edge_segments controls view count: views = 20 * n^2
        # cavity_cameras adds secondary cameras for deep cavities
        extractor = PointCloudExtractor(
            edge_segments=4,  # 320 views
            resolution=1000,  # 1000x1000 rays per view
            cavity_cameras=100,  # 100 secondary hemisphere cameras
        )
        pointcloud = extractor.extract(vertices, indices)
        print(f"Extracted {pointcloud.num_points} points")

        # Step 2: Reconstruct clean mesh using Poisson reconstruction
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
# Random number generation (LCG-based, suitable for GPU)
# -----------------------------------------------------------------------------

# LCG constants (same as glibc)
_LCG_A = wp.constant(wp.uint32(1103515245))
_LCG_C = wp.constant(wp.uint32(12345))


@wp.func
def rand_init(seed: wp.uint32, thread_id: wp.uint32) -> wp.uint32:
    """Initialize random state from a seed and thread ID.

    Combines seed with thread_id using XOR and applies one LCG step
    to ensure different threads have different starting states.
    """
    state = seed ^ thread_id
    # Apply one LCG step to mix the bits
    return state * _LCG_A + _LCG_C


@wp.func
def rand_next(state: wp.uint32) -> wp.uint32:
    """Advance the random state and return the new state."""
    return state * _LCG_A + _LCG_C


@wp.func
def rand_float(state: wp.uint32) -> float:
    """Convert random state to a float in [0, 1)."""
    # Use upper bits (better quality in LCG)
    return wp.float32(state & wp.uint32(0x7FFFFFFF)) / wp.float32(0x7FFFFFFF)


@wp.func
def rand_next_float(state: wp.uint32) -> tuple[wp.uint32, float]:
    """Advance state and return (new_state, random_float).

    Use this when you need multiple random numbers in sequence.
    """
    new_state = state * _LCG_A + _LCG_C
    rand_val = wp.float32(new_state & wp.uint32(0x7FFFFFFF)) / wp.float32(0x7FFFFFFF)
    return new_state, rand_val


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
    sum_hit_distances: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    # Output arrays
    out_points: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
    out_hit_distances: wp.array(dtype=wp.float32),
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

    # Average hit distance
    avg_hit_dist = sum_hit_distances[idx] * inv_count

    out_points[tid] = avg_pos
    out_normals[tid] = avg_normal
    out_hit_distances[tid] = avg_hit_dist


class VoxelHashGrid:
    """Sparse voxel grid with online accumulation of positions, normals, and hit distances.

    Uses a GPU hash table to map voxel coordinates (Morton-encoded) to
    accumulator slots. Points, normals, and hit distances are accumulated using
    atomic operations, allowing fully parallel insertion from multiple threads.

    This is useful for voxel-based downsampling of point clouds directly
    on the GPU without intermediate storage. Hit distances are tracked to enable
    weighted sampling for cavity camera placement.

    Args:
        capacity: Maximum number of unique voxels. Rounded up to power of two.
        voxel_size: Size of each cubic voxel.
        device: Warp device for computation.

    Example:
        >>> grid = VoxelHashGrid(capacity=1_000_000, voxel_size=0.01)
        >>> # Accumulate points (typically done in a kernel)
        >>> # ...
        >>> points, normals, hit_distances, count = grid.finalize()
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
        self.sum_hit_distances = wp.zeros(self.capacity, dtype=wp.float32, device=device)
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
        self.sum_hit_distances.zero_()
        self.counts.zero_()

    def get_num_voxels(self) -> int:
        """Get the current number of occupied voxels."""
        return int(self._hashtable.active_slots.numpy()[self.capacity])

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Finalize accumulation and return averaged points, normals, and hit distances.

        Computes the average position, normalized normal, and average hit distance
        for each occupied voxel and returns the results as numpy arrays.

        Returns:
            Tuple of (points, normals, hit_distances, num_points) where:
            - points: (N, 3) float32 array of averaged positions
            - normals: (N, 3) float32 array of normalized normals
            - hit_distances: (N,) float32 array of averaged hit distances
            - num_points: number of occupied voxels
        """
        num_active = self.get_num_voxels()
        if num_active == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                0,
            )

        # Allocate output buffers
        out_points = wp.zeros(num_active, dtype=wp.vec3, device=self.device)
        out_normals = wp.zeros(num_active, dtype=wp.vec3, device=self.device)
        out_hit_distances = wp.zeros(num_active, dtype=wp.float32, device=self.device)

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
                self.sum_hit_distances,
                self.counts,
                out_points,
                out_normals,
                out_hit_distances,
            ],
            device=self.device,
        )

        wp.synchronize()

        return (
            out_points.numpy(),
            out_normals.numpy(),
            out_hit_distances.numpy(),
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


def create_icosahedron_directions(edge_segments: int = 2) -> np.ndarray:
    """Create camera directions from subdivided icosahedron face centers.

    An icosahedron has 20 faces. Each face is subdivided into n^2 smaller
    triangles where n is the number of segments per edge. This gives
    fine-grained control over the number of directions.

    Args:
        edge_segments: Number of segments per triangle edge (n >= 1).
            Total faces = 20 * n^2. Examples:
            - n=1: 20 faces (original icosahedron)
            - n=2: 80 faces
            - n=3: 180 faces
            - n=4: 320 faces
            - n=5: 500 faces

    Returns:
        (N, 3) array of unit direction vectors, one per face.
    """
    if edge_segments < 1:
        raise ValueError(f"edge_segments must be >= 1, got {edge_segments}")

    n = edge_segments

    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedron vertices (normalized)
    ico_verts = np.array(
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
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    # Icosahedron faces (20 triangles)
    ico_faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    if n == 1:
        # No subdivision needed - just return face centers
        face_centers = np.zeros((20, 3), dtype=np.float64)
        for i, (i0, i1, i2) in enumerate(ico_faces):
            center = (ico_verts[i0] + ico_verts[i1] + ico_verts[i2]) / 3.0
            face_centers[i] = center / np.linalg.norm(center)
        return face_centers.astype(np.float32)

    # Subdivide each face into n^2 triangles using barycentric coordinates
    # For n segments per edge, we create a grid of vertices in barycentric space
    # and then map to 3D, projecting onto the unit sphere.

    all_face_centers = []

    for i0, i1, i2 in ico_faces:
        v0 = ico_verts[i0]
        v1 = ico_verts[i1]
        v2 = ico_verts[i2]

        # Create vertex grid using barycentric coordinates
        # For n segments, we have (n+1) points along each edge
        # Barycentric coords (i, j, k) where i + j + k = n, and each >= 0
        # Map to 3D: p = (i*v0 + j*v1 + k*v2) / n, then normalize to sphere

        # Build vertex lookup: key = (i, j) -> vertex position on sphere
        vertex_grid = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                # Barycentric interpolation
                p = (i * v0 + j * v1 + k * v2) / n
                # Project to unit sphere
                p = p / np.linalg.norm(p)
                vertex_grid[(i, j)] = p

        # Create n^2 small triangles
        # For each row i (0 to n-1), we have triangles pointing up and down
        for i in range(n):
            for j in range(n - i):
                # Upward-pointing triangle: (i,j), (i+1,j), (i,j+1)
                p0 = vertex_grid[(i, j)]
                p1 = vertex_grid[(i + 1, j)]
                p2 = vertex_grid[(i, j + 1)]
                center = (p0 + p1 + p2) / 3.0
                center = center / np.linalg.norm(center)
                all_face_centers.append(center)

                # Downward-pointing triangle (if not on the edge): (i+1,j), (i+1,j+1), (i,j+1)
                if j < n - i - 1:
                    p0 = vertex_grid[(i + 1, j)]
                    p1 = vertex_grid[(i + 1, j + 1)]
                    p2 = vertex_grid[(i, j + 1)]
                    center = (p0 + p1 + p2) / 3.0
                    center = center / np.linalg.norm(center)
                    all_face_centers.append(center)

    face_centers = np.array(all_face_centers, dtype=np.float64)
    return face_centers.astype(np.float32)


def compute_hemisphere_edge_segments(target_rays: int) -> int:
    """Compute the icosahedron edge segments to get approximately target_rays hemisphere directions.

    The number of hemisphere directions is approximately half of the full sphere directions:
    - n edge segments gives 20 * n^2 full sphere directions
    - Hemisphere has ~10 * n^2 directions

    We solve: 10 * n^2 >= target_rays => n >= sqrt(target_rays / 10)

    Args:
        target_rays: Target number of hemisphere directions.

    Returns:
        Edge segments value that gives at least target_rays hemisphere directions.
    """
    import math

    # Direct formula: n = ceil(sqrt(target_rays / 10))
    n = max(1, math.ceil(math.sqrt(target_rays / 10.0)))
    return n


def create_hemisphere_directions(target_rays: int) -> np.ndarray:
    """Create hemisphere directions from a subdivided icosahedron.

    Generates approximately target_rays directions distributed over a hemisphere
    (local Z > 0). These can be rotated to align with any surface normal.

    Args:
        target_rays: Target number of hemisphere directions. The actual count
            will be the smallest icosahedron subdivision that meets or exceeds this.

    Returns:
        (N, 3) array of unit direction vectors in the upper hemisphere (z > 0).
    """
    # Find edge segments that gives enough rays
    edge_segments = compute_hemisphere_edge_segments(target_rays)

    # Generate full sphere directions
    all_directions = create_icosahedron_directions(edge_segments)

    # Filter to upper hemisphere (z > 0)
    hemisphere_directions = all_directions[all_directions[:, 2] > 0]

    return hemisphere_directions


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
    sum_hit_distances: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    # Cavity camera candidate buffers (optional - pass empty arrays to disable)
    cavity_origins: wp.array(dtype=wp.vec3),
    cavity_directions: wp.array(dtype=wp.vec3),
    cavity_hit_distances: wp.array(dtype=wp.float32),
    cavity_count: wp.array(dtype=wp.int32),  # Single-element array for atomic counter
    max_cavity_candidates: wp.int32,
    camera_offset: wp.float32,
    cavity_prob_scale: wp.float32,  # Scale factor to control acceptance rate
    random_seed: wp.uint32,
):
    """Raycast kernel for orthographic projection with direct voxel accumulation.

    Each thread handles one pixel (resolution x resolution grid). Rays are shot
    parallel to cam_dir from positions on the image plane. Hit points, normals,
    and ray distances are accumulated directly into a sparse voxel hash grid.

    Normals are flipped to always point toward the camera (outward from surface).

    Additionally, hits with larger distances (deeper in cavities) are probabilistically
    selected as cavity camera candidates. The acceptance probability is scaled by
    cavity_prob_scale to control the expected number of candidates.
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
            wp.atomic_add(sum_hit_distances, idx, query.t)
            wp.atomic_add(counts, idx, 1)

        # Probabilistically select this hit as a cavity camera candidate
        # Higher hit distance = higher probability of selection
        if max_cavity_candidates > 0:
            # Generate random number
            thread_id = wp.uint32(px * resolution + py)
            rand_state = rand_init(random_seed, thread_id)
            rand_val = rand_float(rand_state)

            # Acceptance probability: scaled by hit distance to favor deeper hits
            # cavity_prob_scale controls overall rate to match expected ray count
            accept_prob = (query.t / max_ray_dist) * cavity_prob_scale
            if rand_val < accept_prob:
                # Atomically claim a slot
                slot = wp.atomic_add(cavity_count, 0, 1)
                if slot < max_cavity_candidates:
                    # Origin: offset back along ray direction (guaranteed outside mesh)
                    cavity_origin = hit_point - ray_direction * camera_offset
                    # Direction: negative ray direction (points into the mesh)
                    cavity_dir = -ray_direction
                    cavity_origins[slot] = cavity_origin
                    cavity_directions[slot] = cavity_dir
                    cavity_hit_distances[slot] = query.t


@wp.kernel
def raycast_hemisphere_kernel(
    # Mesh
    mesh_id: wp.uint64,
    # Camera parameters
    cam_origin: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    cam_forward: wp.vec3,
    min_ray_dist: wp.float32,
    max_ray_dist: wp.float32,
    origin_hit_dist: wp.float32,  # Hit distance at the camera origin point
    # Hemisphere directions (local frame, z > 0)
    hemisphere_dirs: wp.array(dtype=wp.vec3),
    num_directions: wp.int32,
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
    sum_hit_distances: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
):
    """Raycast kernel for hemisphere projection from a cavity camera.

    Each thread handles one ray direction. Rays are shot from cam_origin in directions
    transformed from local hemisphere frame to world frame using the camera basis.

    The camera origin and forward direction come from cavity candidates collected
    during primary raycasting - the origin is offset back along the original ray
    direction to guarantee it's outside the mesh.

    Hit distances are accumulated as the sum of origin_hit_dist (how far the primary
    ray traveled) plus the secondary ray's travel distance, giving total cavity depth.

    Hits are accumulated directly into the sparse voxel hash grid.
    """
    tid = wp.tid()

    if tid >= num_directions:
        return

    # Get local hemisphere direction (z > 0 in local frame)
    local_dir = hemisphere_dirs[tid]

    # Transform to world space: local (x, y, z) -> world (right, up, forward)
    # local z (forward) maps to cam_forward (the surface normal)
    # local x maps to cam_right
    # local y maps to cam_up
    world_dir = cam_right * local_dir[0] + cam_up * local_dir[1] + cam_forward * local_dir[2]
    world_dir = wp.normalize(world_dir)

    # Query mesh intersection
    query = wp.mesh_query_ray(mesh_id, cam_origin, world_dir, max_ray_dist)

    if query.result and query.t > min_ray_dist:
        # Compute hit point
        hit_point = cam_origin + world_dir * query.t

        # Get surface normal - ensure it points toward camera (opposite to ray direction)
        normal = query.normal
        if wp.dot(normal, world_dir) > 0.0:
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
            # Total hit distance = distance to reach camera origin + this ray's distance
            wp.atomic_add(sum_hit_distances, idx, origin_hit_dist + query.t)
            wp.atomic_add(counts, idx, 1)


class PointCloudExtractor:
    """Extract dense point clouds with normals from triangle meshes.

    Uses multi-view orthographic raycasting from directions distributed on
    an icosphere to capture the complete surface of a mesh. Normals are
    guaranteed to be consistent (always pointing outward toward the camera).

    Points are accumulated directly into a sparse voxel hash grid during
    raycasting, providing built-in downsampling and dramatically reducing
    memory usage compared to storing all ray hits.

    Optionally, secondary "cavity cameras" can shoot hemisphere rays to improve
    coverage of deep cavities and occluded regions. During primary raycasting,
    hits with large ray distances are probabilistically collected as cavity camera
    candidates (acceptance probability scales with hit distance and is auto-tuned
    to match buffer capacity). The camera origin is offset back along the ray
    direction to guarantee it's outside the mesh. Primary cameras are processed
    in randomized order to ensure even distribution of cavity candidates.

    Args:
        edge_segments: Number of segments per icosahedron edge for camera directions.
            Total views = 20 * n^2. Examples:
            - n=1: 20 views
            - n=2: 80 views
            - n=3: 180 views
            - n=4: 320 views
            - n=5: 500 views
            Higher values provide better coverage with finer control than recursive
            subdivision.
        resolution: Pixel resolution of the orthographic camera (resolution x resolution).
            Must be between 1 and 10000. Also determines the number of rays per cavity
            camera (~resolution^2 hemisphere directions).
        voxel_size: Size of voxels for point accumulation. If None (default), automatically
            computed as 0.1% of the mesh bounding sphere radius. Smaller values give
            denser point clouds but require more memory.
        max_voxels: Maximum number of unique voxels (hash table capacity). If None (default),
            automatically estimated based on voxel_size and mesh extent to keep hash table
            load factor around 50%. Set explicitly if you know your requirements.
        device: Warp device to use for computation.
        seed: Random seed for reproducibility. Controls camera roll angles, camera
            processing order, and cavity candidate selection. Set to None for
            non-deterministic behavior.
        cavity_cameras: Number of secondary hemisphere cameras for improved cavity
            coverage. Set to 0 (default) to disable. Camera positions are collected
            during primary raycasting from hits with large ray distances (deep in
            cavities). Each camera shoots ~resolution^2 rays in a hemisphere pattern,
            with position and direction guaranteed to be outside the mesh surface.

    Note:
        Memory usage is dominated by the voxel hash grid, which scales with
        ``max_voxels`` (~32 bytes per voxel slot), not with ``resolution^2 * num_views``.
        This makes high-resolution extraction practical even on memory-constrained systems.

    Example:
        >>> extractor = PointCloudExtractor(edge_segments=4, resolution=1000)
        >>> result = extractor.extract(vertices, indices)
        >>> print(f"Extracted {result.num_points} points with normals")

        >>> # With cavity cameras for better coverage of occluded regions
        >>> extractor = PointCloudExtractor(edge_segments=4, resolution=500, cavity_cameras=100)
        >>> result = extractor.extract(vertices, indices)
    """

    def __init__(
        self,
        edge_segments: int = 2,
        resolution: int = 1000,
        voxel_size: float | None = None,
        max_voxels: int | None = None,
        device: str | None = None,
        seed: int | None = 42,
        cavity_cameras: int = 0,
    ):
        # Validate parameters
        if edge_segments < 1:
            raise ValueError(f"edge_segments must be >= 1, got {edge_segments}")
        if resolution < 1 or resolution > 10000:
            raise ValueError(f"resolution must be between 1 and 10000 (inclusive), got {resolution}")
        if voxel_size is not None and voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if max_voxels is not None and max_voxels < 1:
            raise ValueError(f"max_voxels must be >= 1, got {max_voxels}")
        if cavity_cameras < 0:
            raise ValueError(f"cavity_cameras must be >= 0, got {cavity_cameras}")

        self.edge_segments = edge_segments
        self.resolution = resolution
        self.voxel_size = voxel_size  # None means auto-compute
        self.max_voxels = max_voxels  # None means auto-compute
        self.device = device if device is not None else wp.get_device()
        self.seed = seed
        self.cavity_cameras = cavity_cameras

        # Pre-compute camera directions for primary pass
        self.directions = create_icosahedron_directions(edge_segments)
        self.num_views = len(self.directions)

        # Pre-compute hemisphere directions for cavity cameras
        if cavity_cameras > 0:
            target_rays = resolution * resolution
            self.hemisphere_directions = create_hemisphere_directions(target_rays)
            self.num_hemisphere_dirs = len(self.hemisphere_directions)
        else:
            self.hemisphere_directions = None
            self.num_hemisphere_dirs = 0

    def extract(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        padding_factor: float = 1.1,
    ) -> PointCloudResult:
        """Extract point cloud from a triangle mesh.

        Performs multi-view orthographic raycasting with online voxel-based accumulation:

        1. Primary pass: Rays from icosphere-distributed cameras (processed in random
           order) with random roll per camera. Hits with large ray distances are
           probabilistically collected as cavity camera candidates, with acceptance
           probability auto-scaled based on expected hit count and buffer capacity.
        2. Secondary pass (if cavity_cameras > 0): Hemisphere rays from sampled cavity
           camera candidates (weighted by hit distance to favor deeper cavities).
           Camera positions are offset back along the original ray direction to
           guarantee they're outside the mesh.

        Points are accumulated into a sparse voxel hash grid, automatically averaging
        multiple hits per voxel. This provides built-in downsampling with minimal memory.

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
            # Auto: 0.1% of bounding sphere radius (small for high quality)
            voxel_size = radius * 0.001
            # Guard against zero (single point)
            if voxel_size == 0.0:
                voxel_size = 1e-6
        else:
            voxel_size = self.voxel_size

        # Compute max_voxels (auto or user-specified)
        if self.max_voxels is None:
            # Estimate based on surface area: voxels ≈ 4πr² / voxel_size²
            # Use 2x safety factor for non-spherical shapes
            # Estimate surface voxels assuming sphere surface area (upper bound)
            # Use 4x for hash table load factor (~25%)
            # Cap at 16M to avoid excessive memory for small voxels
            estimated_surface_voxels = 4.0 * np.pi * (radius / voxel_size) ** 2
            max_voxels = min(1 << 24, max(1 << 20, int(estimated_surface_voxels * 4)))
        else:
            max_voxels = self.max_voxels

        # Create sparse voxel hash grid for accumulation
        voxel_grid = VoxelHashGrid(
            capacity=max_voxels,
            voxel_size=voxel_size,
            device=self.device,
        )

        # Create Warp mesh
        wp_vertices = wp.array(vertices, dtype=wp.vec3, device=self.device)
        wp_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)

        # Create random generator for camera roll angles
        rng = np.random.default_rng(self.seed)

        # Pre-compute all camera bases and random rotations (vectorized)
        # directions is (num_views, 3)
        directions = self.directions

        # Compute camera bases for all directions at once
        # Choose world_up based on direction[1] magnitude
        world_ups = np.where(
            np.abs(directions[:, 1:2]) < 0.9,
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        )  # (num_views, 3)

        # right = cross(world_up, direction), then normalize
        rights = np.cross(world_ups, directions)
        rights /= np.linalg.norm(rights, axis=1, keepdims=True)

        # up = cross(direction, right)
        ups = np.cross(directions, rights)

        # Pre-generate all random roll angles and apply rotation
        thetas = rng.uniform(0, 2 * np.pi, size=self.num_views)
        cos_thetas = np.cos(thetas)[:, np.newaxis]  # (num_views, 1)
        sin_thetas = np.sin(thetas)[:, np.newaxis]

        # Rotated: right' = cos*right + sin*up, up' = cos*up - sin*right
        rights_rot = cos_thetas * rights + sin_thetas * ups
        ups_rot = cos_thetas * ups - sin_thetas * rights

        # Camera origins: center - direction * padded_radius
        cam_origins = center - directions * padded_radius

        # Camera offset for cavity candidates (0.1% of bounding sphere radius)
        camera_offset = radius * 0.001

        # Allocate cavity camera candidate buffers if needed
        if self.cavity_cameras > 0:
            # Large buffer to collect candidates - at least 100K or 100x requested cameras
            max_cavity_candidates = max(100_000, self.cavity_cameras * 100)
            cavity_origins = wp.zeros(max_cavity_candidates, dtype=wp.vec3, device=self.device)
            cavity_directions = wp.zeros(max_cavity_candidates, dtype=wp.vec3, device=self.device)
            cavity_hit_distances = wp.zeros(max_cavity_candidates, dtype=wp.float32, device=self.device)
            cavity_count = wp.zeros(1, dtype=wp.int32, device=self.device)

            # Calculate probability scale to target ~2x buffer size candidates
            # Total rays = num_views * resolution^2, assume ~50% hit rate
            total_expected_hits = self.num_views * self.resolution * self.resolution * 0.5
            # Average hit distance ratio is ~0.5, so base acceptance would be 0.5
            # We want: 0.5 * prob_scale * total_hits ≈ 2 * max_cavity_candidates
            # prob_scale = 4 * max_cavity_candidates / total_hits
            cavity_prob_scale = float(4.0 * max_cavity_candidates / max(total_expected_hits, 1.0))
            # Clamp to reasonable range
            cavity_prob_scale = min(1.0, max(1e-6, cavity_prob_scale))
        else:
            # Empty arrays when cavity cameras disabled
            max_cavity_candidates = 0
            cavity_origins = wp.empty(0, dtype=wp.vec3, device=self.device)
            cavity_directions = wp.empty(0, dtype=wp.vec3, device=self.device)
            cavity_hit_distances = wp.empty(0, dtype=wp.float32, device=self.device)
            cavity_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            cavity_prob_scale = 0.0

        # Randomize camera order to get even distribution of cavity candidates
        # (prevents later cameras from always overflowing the buffer)
        camera_order = rng.permutation(self.num_views)

        # Cast rays from each camera direction, accumulating into voxel grid
        for i in camera_order:
            direction = directions[i]
            right = rights_rot[i]
            up = ups_rot[i]
            cam_origin = cam_origins[i]

            # Different random seed per view for cavity candidate selection
            random_seed = rng.integers(0, 2**31, dtype=np.uint32)

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
                    voxel_grid.sum_hit_distances,
                    voxel_grid.counts,
                    cavity_origins,
                    cavity_directions,
                    cavity_hit_distances,
                    cavity_count,
                    max_cavity_candidates,
                    float(camera_offset),
                    float(cavity_prob_scale),
                    int(random_seed),
                ],
                device=self.device,
            )

        # Check hash table load factor and warn if too high
        num_voxels_after_primary = voxel_grid.get_num_voxels()
        load_factor = num_voxels_after_primary / voxel_grid.capacity
        if load_factor > 0.7:
            import warnings

            warnings.warn(
                f"Voxel hash table is {load_factor:.0%} full ({num_voxels_after_primary}/{voxel_grid.capacity}). "
                f"This may cause slowdowns. Consider increasing max_voxels or using a larger voxel_size.",
                stacklevel=2,
            )

        # Secondary pass: cavity cameras for improved coverage of occluded regions
        if self.cavity_cameras > 0:
            wp.synchronize()

            # Get the number of cavity candidates that were attempted to write
            total_attempts = int(cavity_count.numpy()[0])
            # Clamp to buffer size (some may have been dropped due to overflow)
            num_candidates = min(total_attempts, max_cavity_candidates)

            # Report buffer status
            if total_attempts > max_cavity_candidates:
                overflow_count = total_attempts - max_cavity_candidates
                print(
                    f"Cavity candidates: {num_candidates:,} collected, "
                    f"{overflow_count:,} dropped (buffer overflow, not critical)"
                )
            elif num_candidates > 0:
                print(f"Cavity candidates: {num_candidates:,} collected")

            if num_candidates > 0:
                # Prepare hemisphere directions on GPU
                wp_hemisphere_dirs = wp.array(self.hemisphere_directions, dtype=wp.vec3, device=self.device)

                # Minimum ray distance to avoid self-occlusion
                min_ray_dist = camera_offset * 2.0

                # Get cavity candidate data from GPU
                origins_np = cavity_origins.numpy()[:num_candidates]
                directions_np = cavity_directions.numpy()[:num_candidates]
                hit_dists_np = cavity_hit_distances.numpy()[:num_candidates]

                # Sample cavity cameras with weighted random choice
                # Weight by hit distance to favor deeper cavities
                weights = hit_dists_np.copy()
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights /= weights_sum
                else:
                    weights = np.ones(num_candidates) / num_candidates

                # Sample up to cavity_cameras, but no more than available candidates
                num_to_sample = min(self.cavity_cameras, num_candidates)
                sample_indices = rng.choice(num_candidates, size=num_to_sample, p=weights, replace=True)

                # Pre-generate all random roll angles
                thetas = rng.uniform(0, 2 * np.pi, size=num_to_sample)

                # Launch kernel for each cavity camera
                for i in range(num_to_sample):
                    sample_idx = sample_indices[i]
                    cam_origin = origins_np[sample_idx]
                    cam_forward = directions_np[sample_idx]  # Already points into mesh
                    sample_hit_dist = hit_dists_np[sample_idx]

                    # Compute camera basis (cam_forward is the forward direction)
                    right, up = compute_camera_basis(cam_forward)

                    # Apply random roll around forward direction
                    theta = thetas[i]
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    right_rot = cos_theta * right + sin_theta * up
                    up_rot = cos_theta * up - sin_theta * right
                    right, up = right_rot, up_rot

                    # Launch hemisphere raycast kernel
                    wp.launch(
                        kernel=raycast_hemisphere_kernel,
                        dim=self.num_hemisphere_dirs,
                        inputs=[
                            mesh.id,
                            wp.vec3(cam_origin[0], cam_origin[1], cam_origin[2]),
                            wp.vec3(right[0], right[1], right[2]),
                            wp.vec3(up[0], up[1], up[2]),
                            wp.vec3(cam_forward[0], cam_forward[1], cam_forward[2]),
                            float(min_ray_dist),
                            float(max_ray_dist),
                            float(sample_hit_dist),  # Hit distance at camera origin
                            wp_hemisphere_dirs,
                            self.num_hemisphere_dirs,
                            float(voxel_grid.inv_voxel_size),
                            voxel_grid.keys,
                            voxel_grid.active_slots,
                            voxel_grid.sum_positions_x,
                            voxel_grid.sum_positions_y,
                            voxel_grid.sum_positions_z,
                            voxel_grid.sum_normals_x,
                            voxel_grid.sum_normals_y,
                            voxel_grid.sum_normals_z,
                            voxel_grid.sum_hit_distances,
                            voxel_grid.counts,
                        ],
                        device=self.device,
                    )

        # Finalize voxel grid to get averaged points and normals
        wp.synchronize()
        final_num_voxels = voxel_grid.get_num_voxels()
        final_load_factor = final_num_voxels / voxel_grid.capacity
        print(
            f"Voxel grid: {final_num_voxels:,} voxels, "
            f"{final_load_factor:.1%} load factor "
            f"({final_num_voxels:,}/{voxel_grid.capacity:,})"
        )

        points_np, normals_np, _hit_distances_np, num_points = voxel_grid.finalize()

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
        >>> extractor = PointCloudExtractor(edge_segments=4, resolution=1000)
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
