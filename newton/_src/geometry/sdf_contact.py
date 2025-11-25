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

from typing import Any

import warp as wp

from ..geometry.collision_core import (
    build_pair_key2,
)
from ..geometry.contact_data import ContactData

# Handle both direct execution and module import
try:
    from .contact_reduction import (
        ContactStruct,
        get_shared_memory_pointer_140_contacts,
        get_shared_memory_pointer_141_ints,
        get_shared_memory_pointer_block_dim_plus_2_ints,
        store_reduced_contact,
        synchronize,
    )
except ImportError:
    from contact_reduction import (
        ContactStruct,
        get_shared_memory_pointer_140_contacts,
        get_shared_memory_pointer_141_ints,
        get_shared_memory_pointer_block_dim_plus_2_ints,
        store_reduced_contact,
        synchronize,
    )


@wp.struct
class VolumeData:
    """Encapsulates all data needed to query an SDF volume."""

    volume_id: wp.uint64
    aabb_lower: wp.vec3
    aabb_upper: wp.vec3
    inv_sdf_dx: wp.vec3
    sdf_dx: float
    sdf_dims: wp.vec3


@wp.func
def Sample(texture: wp.uint64, localPos: wp.vec3) -> float:
    """
    Sample the SDF texture at the given local position using trilinear interpolation.

    Args:
        texture: The SDF volume texture to sample from (GPU texture handle)
        localPos: The local position in texture space to sample at

    Returns:
        The signed distance field value at the given position
    """
    # Sample the volume using trilinear interpolation
    return wp.volume_sample_f(texture, localPos, wp.Volume.LINEAR)


@wp.func
def PxSdfDistance(
    texture: wp.uint64,
    sdfBoxLower: wp.vec3,
    sdfBoxHigher: wp.vec3,
    invSdfDx: wp.vec3,
    localPos: wp.vec3,
) -> float:
    """
    Compute the signed distance from a point to the SDF surface.

    This function handles points both inside and outside the SDF volume bounds.
    For points outside the SDF volume, it adds the distance to the nearest point
    on the volume boundary to the SDF value at that boundary point.

    Args:
        texture: The SDF volume texture to sample from
        sdfBoxLower: Lower bounds of the SDF volume in world space
        sdfBoxHigher: Upper bounds of the SDF volume in world space
        invSdfDx: Inverse of the SDF voxel spacing (1.0 / voxel_size)
        localPos: The query position in world space

    Returns:
        The signed distance to the SDF surface. Negative values indicate
        penetration into the object.
    """
    # clamp to SDF support
    clampedGridPt = wp.max(localPos, sdfBoxLower)
    clampedGridPt = wp.min(clampedGridPt, sdfBoxHigher)

    diffMag = wp.length(localPos - clampedGridPt)
    f = (clampedGridPt - sdfBoxLower) * invSdfDx

    return Sample(texture, f) + diffMag


@wp.func
def PxVolumeGrad(
    texture: wp.uint64,
    sdfBoxLower: wp.vec3,
    sdfBoxHigher: wp.vec3,
    invSdfDx: wp.vec3,
    sdfDims: wp.vec3,
    sdfDx: float,
    localPos: wp.vec3,
) -> wp.vec3:
    """
    Compute the gradient of the SDF volume at the given position.

    The gradient computation uses central differences. For points well inside
    the volume (at least 1 voxel from boundaries), it samples directly from the
    texture. For points near boundaries or outside, it uses PxSdfDistance which
    handles extrapolation.

    Args:
        texture: The SDF volume texture
        sdfBoxLower: Lower bounds of the SDF volume in world space
        sdfBoxHigher: Upper bounds of the SDF volume in world space
        invSdfDx: Inverse of the SDF voxel spacing (1.0 / voxel_size)
        sdfDims: Dimensions of the SDF volume in voxels
        sdfDx: The SDF voxel spacing
        localPos: The query position in world space

    Returns:
        The gradient vector (unnormalized). The direction points toward increasing
        SDF values (i.e., away from the object surface for points outside).
    """
    grad = wp.vec3(0.0, 0.0, 0.0)

    clampedGridPt = wp.max(localPos, sdfBoxLower)
    clampedGridPt = wp.min(clampedGridPt, sdfBoxHigher)

    f = (clampedGridPt - sdfBoxLower) * invSdfDx

    if (
        f[0] >= 1.0
        and f[0] <= sdfDims[0] - 2.0
        and f[1] >= 1.0
        and f[1] <= sdfDims[1] - 2.0
        and f[2] >= 1.0
        and f[2] <= sdfDims[2] - 2.0
    ):
        grad = wp.vec3(
            Sample(texture, wp.vec3(f[0] + 1.0, f[1], f[2])) - Sample(texture, wp.vec3(f[0] - 1.0, f[1], f[2])),
            Sample(texture, wp.vec3(f[0], f[1] + 1.0, f[2])) - Sample(texture, wp.vec3(f[0], f[1] - 1.0, f[2])),
            Sample(texture, wp.vec3(f[0], f[1], f[2] + 1.0)) - Sample(texture, wp.vec3(f[0], f[1], f[2] - 1.0)),
        )
        return grad

    grad = wp.vec3(
        PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, localPos + wp.vec3(sdfDx, 0.0, 0.0))
        - PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, localPos - wp.vec3(sdfDx, 0.0, 0.0)),
        PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, localPos + wp.vec3(0.0, sdfDx, 0.0))
        - PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, localPos - wp.vec3(0.0, sdfDx, 0.0)),
        PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, localPos + wp.vec3(0.0, 0.0, sdfDx))
        - PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, localPos - wp.vec3(0.0, 0.0, sdfDx)),
    )

    return grad


@wp.func
def closestPtPointBaryTriangle(c: wp.vec3) -> wp.vec3:
    """
    Find the closest point to `c` on the standard barycentric triangle.

    This function projects a barycentric coordinate point onto the valid barycentric
    triangle defined by vertices (1,0,0), (0,1,0), (0,0,1) in barycentric space.
    The valid region is where all coordinates are non-negative and sum to 1.

    This is a specialized version of the general closest-point-on-triangle algorithm
    optimized for the barycentric simplex.

    Args:
        c: Input barycentric coordinates (may be outside valid triangle region)

    Returns:
        The closest valid barycentric coordinates. All components will be >= 0
        and sum to 1.0.

    Note:
        This is used in optimization algorithms that work in barycentric space,
        where gradient descent may produce invalid coordinates that need projection.
    """
    third = 1.0 / 3.0  # constexpr
    c = c - wp.vec3(third * (c[0] + c[1] + c[2] - 1.0))

    # two negative: return positive vertex
    if c[1] < 0.0 and c[2] < 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    if c[0] < 0.0 and c[2] < 0.0:
        return wp.vec3(0.0, 1.0, 0.0)

    if c[0] < 0.0 and c[1] < 0.0:
        return wp.vec3(0.0, 0.0, 1.0)

    # one negative: return projection onto line if it is on the edge, or the largest vertex otherwise
    if c[0] < 0.0:
        d = c[0] * 0.5
        y = c[1] + d
        z = c[2] + d
        if y > 1.0:
            return wp.vec3(0.0, 1.0, 0.0)
        if z > 1.0:
            return wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(0.0, y, z)
    if c[1] < 0.0:
        d = c[1] * 0.5
        x = c[0] + d
        z = c[2] + d
        if x > 1.0:
            return wp.vec3(1.0, 0.0, 0.0)
        if z > 1.0:
            return wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(x, 0.0, z)
    if c[2] < 0.0:
        d = c[2] * 0.5
        x = c[0] + d
        y = c[1] + d
        if x > 1.0:
            return wp.vec3(1.0, 0.0, 0.0)
        if y > 1.0:
            return wp.vec3(0.0, 1.0, 0.0)
        return wp.vec3(x, y, 0.0)
    return c


@wp.func
def PxSdfSampleWithGrad(
    texture: wp.uint64,
    sdfBoxLower: wp.vec3,
    sdfBoxHigher: wp.vec3,
    invSdfDx: wp.vec3,
    localPos: wp.vec3,
    sdfGradient: wp.vec3,
    tolerance: float,
) -> tuple[float, wp.vec3]:
    """
    Sample SDF with gradient information and compute contact distance and direction.

    This function evaluates the SDF at a point and, if the distance is within tolerance,
    computes the contact direction considering both the SDF gradient and any offset
    from the SDF volume boundaries.

    Args:
        texture: The SDF volume texture
        sdfBoxLower: Lower bounds of the SDF volume in world space
        sdfBoxHigher: Upper bounds of the SDF volume in world space
        invSdfDx: Inverse of the SDF voxel spacing
        localPos: The query position in world space
        sdfGradient: Pre-computed gradient at or near localPos (for efficiency)
        tolerance: Distance threshold for considering contacts

    Returns:
        A tuple of (distance, direction) where:
        - distance: Signed distance to surface (negative = penetration), or PX_MAX_F32 if > tolerance
        - direction: Normalized contact direction (from surface toward query point)
    """
    PX_MAX_F32 = 3.4028235e38

    clampedGridPt = wp.max(localPos, sdfBoxLower)
    clampedGridPt = wp.min(clampedGridPt, sdfBoxHigher)

    diff = localPos - clampedGridPt
    # const PxReal diffMag = diff.magnitudeSquared();
    # if (diffMag > tolerance*tolerance)
    #     return PX_MAX_F32;

    f = (clampedGridPt - sdfBoxLower) * invSdfDx

    dist = Sample(texture, f)

    dir = wp.vec3(0.0, 0.0, 0.0)

    if dist < tolerance:
        # estimate the contact direction based on the SDF
        # if we are outside the SDF support, add in the distance to the point
        grad_len = wp.length(sdfGradient)
        if grad_len > 0.0:
            sdfGradient_normalized = sdfGradient / grad_len
        else:
            sdfGradient_normalized = wp.vec3(0.0, 0.0, 1.0)

        dir = sdfGradient_normalized * wp.abs(dist) + diff
        dir_len = wp.length(dir)
        if dir_len > 0.0:
            dir = dir / dir_len
        else:
            dir = wp.vec3(0.0, 0.0, 1.0)

        return dist + wp.dot(dir, diff), dir

    return PX_MAX_F32, dir


@wp.func
def doTriangleSDFCollision(
    sdf: wp.uint64,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
) -> tuple[float, wp.vec3, wp.vec3]:
    """
    Compute the deepest contact between a triangle and an SDF volume.

    This function uses gradient descent in barycentric coordinates to find the point
    on the triangle that has the minimum (most negative) signed distance to the SDF.
    The optimization starts from either the triangle centroid or one of its vertices
    (whichever has the smallest initial distance).

    Algorithm:
    1. Evaluate SDF distance at triangle vertices and centroid
    2. Start from the point with minimum distance
    3. Iterate up to 16 times:
       - Compute SDF gradient at current point
       - Project gradient onto triangle edges (in barycentric space)
       - Take a gradient descent step with decreasing step size
       - Project result back onto valid barycentric triangle
    4. Return final distance, contact point, and contact direction

    Args:
        sdf: The SDF volume texture
        v0, v1, v2: Triangle vertices in world space
        tolerance: Contact distance threshold

    Returns:
        Tuple of (distance, contact_point, contact_direction) where:
        - distance: Signed distance to SDF surface (negative = penetration)
        - contact_point: The point on the triangle closest to the SDF surface
        - contact_direction: Normalized direction from surface to contact point
    """
    third = 1.0 / 3.0
    center = (v0 + v1 + v2) * third
    p = center

    dist = wp.volume_sample_f(sdf, wp.volume_world_to_index(sdf, p), wp.Volume.LINEAR)

    d0 = wp.volume_sample_f(sdf, wp.volume_world_to_index(sdf, v0), wp.Volume.LINEAR)
    d1 = wp.volume_sample_f(sdf, wp.volume_world_to_index(sdf, v1), wp.Volume.LINEAR)
    d2 = wp.volume_sample_f(sdf, wp.volume_world_to_index(sdf, v2), wp.Volume.LINEAR)

    uvw = wp.vec3(0.0, 0.0, 0.0)

    # choose starting iterate among centroid and triangle vertices
    if d0 < d1 and d0 < d2 and d0 < dist:
        p = v0
        uvw = wp.vec3(1.0, 0.0, 0.0)
    elif d1 < d2 and d1 < dist:
        p = v1
        uvw = wp.vec3(0.0, 1.0, 0.0)
    elif d2 < dist:
        p = v2
        uvw = wp.vec3(0.0, 0.0, 1.0)
    else:
        uvw = wp.vec3(third, third, third)

    difference = wp.sqrt(
        wp.max(
            wp.length_sq(v0 - p),
            wp.max(wp.length_sq(v1 - p), wp.length_sq(v2 - p)),
        )
    )

    toleranceSq = 1e-3 * 1e-3

    sdfGradient = wp.vec3(0.0, 0.0, 0.0)
    step = 1.0 / (2.0 * difference)

    for _iter in range(16):
        wp.volume_sample_grad_f(sdf, wp.volume_world_to_index(sdf, p), wp.Volume.LINEAR, sdfGradient)

        grad_len = wp.length(sdfGradient)
        if grad_len == 0.0:
            # We ran into a discontinuity e.g. the exact center of a cube
            # Just pick an arbitrary gradient of unit length to move out of the discontinuity
            sdfGradient = wp.vec3(0.571846586, 0.705545099, 0.418566116)
            grad_len = 1.0

        sdfGradient = sdfGradient / grad_len

        dfdu = wp.dot(sdfGradient, v0 - p)
        dfdv = wp.dot(sdfGradient, v1 - p)
        dfdw = wp.dot(sdfGradient, v2 - p)

        newUVW = uvw

        newUVW = wp.vec3(newUVW[0] - step * dfdu, newUVW[1] - step * dfdv, newUVW[2] - step * dfdw)

        step = step * 0.8

        newUVW = closestPtPointBaryTriangle(newUVW)

        p = v0 * newUVW[0] + v1 * newUVW[1] + v2 * newUVW[2]

        if wp.length_sq(uvw - newUVW) < toleranceSq:
            # if(iter != 0)
            #     printf("Iter = %i\n", iter);
            break

        uvw = newUVW

    dist = wp.volume_sample_grad_f(sdf, wp.volume_world_to_index(sdf, p), wp.Volume.LINEAR, sdfGradient)

    point = p
    dir = sdfGradient

    return dist, point, dir


@wp.func
def get_triangle_from_mesh(
    mesh_id: wp.uint64,
    mesh_scale: wp.vec3,
    X_mesh_ws: wp.transform,
    tri_idx: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """
    Extract a triangle from a mesh and transform it to world space.

    This function retrieves a specific triangle from a mesh by its index,
    applies scaling and transformation, and returns the three vertices
    in world space coordinates.

    Args:
        mesh_id: The mesh ID (use wp.mesh_get to retrieve the mesh object)
        mesh_scale: Scale to apply to mesh vertices (component-wise)
        X_mesh_ws: Mesh world-space transform (position and rotation)
        tri_idx: Triangle index in the mesh (0-based)

    Returns:
        Tuple of (v0_world, v1_world, v2_world) - the three triangle vertices
        in world space after applying scale and transform.

    Note:
        The mesh indices array stores triangle vertex indices as a flat array:
        [tri0_v0, tri0_v1, tri0_v2, tri1_v0, tri1_v1, tri1_v2, ...]
    """

    mesh = wp.mesh_get(mesh_id)

    # Extract triangle vertices from mesh (indices are stored as flat array: i0, i1, i2, i0, i1, i2, ...)
    idx0 = mesh.indices[tri_idx * 3 + 0]
    idx1 = mesh.indices[tri_idx * 3 + 1]
    idx2 = mesh.indices[tri_idx * 3 + 2]

    # Get vertex positions in mesh local space (with scale applied)
    v0_local = wp.cw_mul(mesh.points[idx0], mesh_scale)
    v1_local = wp.cw_mul(mesh.points[idx1], mesh_scale)
    v2_local = wp.cw_mul(mesh.points[idx2], mesh_scale)

    # Transform vertices to world space
    v0_world = wp.transform_point(X_mesh_ws, v0_local)
    v1_world = wp.transform_point(X_mesh_ws, v1_local)
    v2_world = wp.transform_point(X_mesh_ws, v2_local)

    return v0_world, v1_world, v2_world


@wp.func
def get_bounding_sphere(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> tuple[wp.vec3, float]:
    """
    Compute a conservative bounding sphere for a triangle.

    This uses the triangle centroid as the sphere center and the maximum
    distance from the centroid to any vertex as the radius. This is a
    conservative (potentially larger than optimal) but fast bounding sphere.

    Args:
        v0, v1, v2: Triangle vertices in world space

    Returns:
        Tuple of (center, radius) where:
        - center: The centroid of the triangle
        - radius: The maximum distance from centroid to any vertex

    Note:
        This is not the minimal bounding sphere, but it's fast to compute
        and adequate for broad-phase culling.
    """
    center = (v0 + v1 + v2) * (1.0 / 3.0)
    radius = wp.max(wp.max(wp.length_sq(v0 - center), wp.length_sq(v1 - center)), wp.length_sq(v2 - center))
    return center, wp.sqrt(radius)


@wp.func
def add_to_shared_buffer_atomic(
    thread_id: int,
    add_triangle: bool,
    tri_idx: int,
    buffer: wp.array(dtype=wp.int32),
):
    """
    Add a triangle index to a shared memory buffer using atomic operations.

    This is a faster alternative to scan-based compaction, reducing from 6 syncs to 2.

    Buffer layout:
    - [0 .. block_dim-1]: Triangle indices
    - [block_dim]: Current count of triangles in buffer
    - [block_dim+1]: Progress counter (triangles processed so far)

    Args:
        thread_id: The calling thread's index within the thread block
        add_triangle: Whether this thread wants to add a triangle
        tri_idx: The triangle index to add (only used if add_triangle is True)
        buffer: Shared memory buffer for triangle indices
    """
    capacity = wp.block_dim()
    idx = -1

    # Atomic add to get write position
    if add_triangle:
        idx = wp.atomic_add(buffer, capacity, 1)
        if idx < capacity:
            buffer[idx] = tri_idx

    # Thread 0 optimistically advances progress by block_dim
    if thread_id == 0:
        buffer[capacity + 1] += capacity

    synchronize()  # SYNC 1: All atomic writes and progress update complete

    # Cap count at capacity (in case of overflow)
    if thread_id == 0 and buffer[capacity] > capacity:
        buffer[capacity] = capacity

    # Overflow threads correct progress to their tri_idx (minimum wins)
    if add_triangle and idx >= capacity:
        wp.atomic_min(buffer, capacity + 1, tri_idx)

    synchronize()  # SYNC 2: All corrections complete, buffer consistent


@wp.func
def findInterestingTriangles(
    thread_id: int,
    mesh_scale: wp.vec3,
    mesh_to_sdf_transform: wp.transform,
    mesh_id: wp.uint64,
    sdf: wp.uint64,
    buffer: wp.array(dtype=wp.int32),
    contact_distance: float,
):
    """
    Broad-phase culling: fill shared buffer with triangle indices that may collide with SDF.

    Tests each triangle's bounding sphere against the SDF. Triangles are selected if
    SDF_distance(sphere_center) <= sphere_radius + contact_distance. Results are stored
    in shared memory buffer for narrow-phase processing.

    Uses atomic compaction (2 syncs per iteration) instead of scan-based (6 syncs).

    Buffer layout: [0..block_dim-1] = indices, [block_dim] = count, [block_dim+1] = progress
    """
    num_tris = wp.mesh_get(mesh_id).indices.shape[0] // 3
    capacity = wp.block_dim()

    synchronize()  # Ensure buffer state is consistent before starting

    while buffer[capacity + 1] < num_tris and buffer[capacity] < capacity:
        # All threads read the same base index (buffer consistent from previous sync)
        base_tri_idx = buffer[capacity + 1]
        tri_idx = base_tri_idx + thread_id
        add_triangle = False

        if tri_idx < num_tris:
            v0, v1, v2 = get_triangle_from_mesh(mesh_id, mesh_scale, mesh_to_sdf_transform, tri_idx)
            bounding_sphere_center, bounding_sphere_radius = get_bounding_sphere(v0, v1, v2)

            # Use SDF distance query for culling
            query_local = wp.volume_world_to_index(sdf, bounding_sphere_center)
            sdf_dist = wp.volume_sample_f(sdf, query_local, wp.Volume.LINEAR)
            add_triangle = sdf_dist <= (bounding_sphere_radius + contact_distance)

        synchronize()  # Ensure all threads have read base_tri_idx before any writes
        add_to_shared_buffer_atomic(thread_id, add_triangle, tri_idx, buffer)
        # add_to_shared_buffer_atomic ends with sync, buffer is consistent for next while check

    synchronize()  # Final sync before returning

def create_SdfMeshCollision(tile_size: int):
    """
    Create a warp kernel for triangle-SDF collision detection between mesh pairs.

    Args:
        tile_size: Tile size for contact reduction (must match contact reduction kernel)

    Returns:
        A warp kernel that processes mesh-SDF collision pairs in parallel.
    """

    @wp.kernel
    def do_sdf_mesh_collision(
        mesh_source: wp.array(dtype=wp.uint64),
        mesh_scales: wp.array(dtype=wp.vec3),
        mesh_transforms: wp.array(dtype=wp.transform),
        sdf_source: wp.array(dtype=wp.uint64),
        pairs: wp.array(dtype=wp.vec2i),
        pair_count: wp.array(dtype=int),
        contact_margin: wp.array(dtype=float),
        # Outputs
        out_contacts: wp.array(dtype=ContactStruct),
        out_count: wp.array(dtype=int),
    ):
        """
        Detect contacts between triangle meshes and SDF volumes for collision pairs.

        Each thread block processes one collision pair. For each pair, triangles from
        mesh0 are tested against volume1, then triangles from mesh1 are tested against
        volume0. Contacts are reduced and written to the output array.

        Args:
            mesh_source: Array of mesh IDs
            mesh_scales: Per-mesh scale factors
            mesh_transforms: Per-mesh world transforms
            volume_source: Array of SDF volume IDs
            pairs: Array of (mesh0_idx, mesh1_idx) collision pairs
            pair_count: Number of pairs to process
            contact_distance: Maximum distance threshold for contacts
            out_contacts: Output array for detected contacts (pre-allocated)
            out_count: Single-element array to store total number of contacts found
        """
        block_id, t = wp.tid()
        num_pairs = pair_count[0]
        if block_id >= num_pairs:
            return

        pair = pairs[block_id]
        mesh0 = mesh_source[pair[0]]
        mesh1 = mesh_source[pair[1]]
        sdf0 = sdf_source[pair[0]]
        sdf1 = sdf_source[pair[1]]

        # Extract mesh parameters
        mesh0_scale = mesh_scales[pair[0]]
        mesh1_scale = mesh_scales[pair[1]]
        mesh0_transform = mesh_transforms[pair[0]]
        mesh1_transform = mesh_transforms[pair[1]]

        margin = wp.max(contact_margin[pair[0]], contact_margin[pair[1]])

        # Initialize (shared memory) buffers for contact reduction
        empty_marker = -1000000000.0
        has_contact = True

        active_contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_141_ints(), shape=(141,), dtype=wp.int32)
        contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_140_contacts(), shape=(140,), dtype=ContactStruct)

        for i in range(t, 140, wp.block_dim()):
            contacts_shared_mem[i].projection = empty_marker

        if t == 0:
            active_contacts_shared_mem[140] = 0

        # Initialize (shared memory) buffers for triangle selection
        tri_capacity = wp.block_dim()
        selected_triangles = wp.array(
            ptr=get_shared_memory_pointer_block_dim_plus_2_ints(), shape=(wp.block_dim() + 2,), dtype=wp.int32
        )

        if t == 0:
            selected_triangles[tri_capacity] = 0  # Stores the number of indices inside selected_triangles
            selected_triangles[tri_capacity + 1] = (
                0  # Stores the number of triangles from the mesh that were already investigated
            )

        for mode in range(2):
            synchronize()
            if mode == 0:
                mesh = mesh0
                mesh_scale = mesh0_scale
                mesh_sdf_transform = wp.transform_compose(wp.transform_inverse(mesh1_transform), mesh0_transform)
                sdf_current = sdf1
            else:
                mesh = mesh1
                mesh_scale = mesh1_scale
                mesh_sdf_transform = wp.transform_compose(wp.transform_inverse(mesh0_transform), mesh1_transform)
                sdf_current = sdf0

            # Reset progress counter for this mesh
            if t == 0:
                selected_triangles[tri_capacity + 1] = 0
            synchronize()

            while selected_triangles[tri_capacity + 1] < wp.mesh_get(mesh).num_tris:
                findInterestingTriangles(
                    t,
                    mesh_scale,
                    mesh_sdf_transform,
                    mesh,
                    sdf_current,
                    selected_triangles,
                    margin,
                )

                has_contact = t < selected_triangles[tri_capacity]
                c = ContactStruct()

                if has_contact:
                    v0, v1, v2 = get_triangle_from_mesh(mesh, mesh_scale, mesh_sdf_transform, selected_triangles[t])
                    dist, point, dir = doTriangleSDFCollision(
                        sdf_current,
                        v0,
                        v1,
                        v2,
                    )

                    c.position = point
                    c.normal = dir
                    c.depth = dist
                    c.feature = 0
                    c.projection = empty_marker

                    has_contact = dist < margin

                synchronize()
                store_reduced_contact(
                    t, has_contact, c, contacts_shared_mem, active_contacts_shared_mem, 140, empty_marker
                )

                # Reset buffer for next batch
                synchronize()
                if t == 0:
                    selected_triangles[tri_capacity] = 0
                synchronize()

        # Now write the reduced contacts to the output array
        num_contacts_to_keep = wp.min(active_contacts_shared_mem[140], 140)

        # Thread 0 writes the count
        if t == 0:
            out_count[0] = num_contacts_to_keep  # TODO: Atomic?

        # All threads cooperatively write the contacts
        for i in range(t, num_contacts_to_keep, wp.block_dim()):
            contact_id = active_contacts_shared_mem[i]
            out_contacts[i] = contacts_shared_mem[contact_id]

    return do_sdf_mesh_collision


def create_narrow_phase_process_mesh_mesh_contacts_kernel(
    writer_func: Any, tile_size: int, reduce_contacts: bool = False
):
    @wp.kernel(enable_backward=False)
    def mesh_sdf_collision_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_sdf: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        writer_data: Any,
        total_num_blocks: int,
    ):
        """
        Process mesh-mesh collisions using SDF-mesh collision detection.

        Uses a strided loop to process mesh-mesh pairs, with threads within each block
        parallelizing over triangles. This follows the pattern from do_sdf_mesh_collision.

        Args:
            geom_types: Array of geometry types for all shapes
            geom_data: Array of vec4 containing scale (xyz) and thickness (w) for each shape
            geom_transform: Array of world-space transforms for each shape
            geom_source: Array of source pointers (mesh IDs) for each shape
            geom_sdf: Array of SDF volume IDs for mesh shapes
            geom_cutoff: Array of cutoff distances for each shape
            shape_pairs_mesh_mesh: Array of mesh-mesh pairs to process
            shape_pairs_mesh_mesh_count: Number of mesh-mesh pairs
            writer_data: Contact writer data structure
            total_num_blocks: Total number of blocks launched for strided loop
        """
        block_id, t = wp.tid()

        num_pairs = shape_pairs_mesh_mesh_count[0]

        # Strided loop over pairs
        for pair_idx in range(block_id, num_pairs, total_num_blocks):
            pair = shape_pairs_mesh_mesh[pair_idx]
            mesh_shape_a = pair[0]
            mesh_shape_b = pair[1]

            # Get mesh and SDF IDs
            mesh_id_a = shape_source[mesh_shape_a]
            mesh_id_b = shape_source[mesh_shape_b]
            sdf_ptr_a = shape_sdf[mesh_shape_a]
            sdf_ptr_b = shape_sdf[mesh_shape_b]

            # Skip if either mesh is invalid
            if mesh_id_a == wp.uint64(0) or mesh_id_b == wp.uint64(0):
                continue

            # Get mesh objects
            mesh_a = wp.mesh_get(mesh_id_a)
            mesh_b = wp.mesh_get(mesh_id_b)

            # Get mesh scales and transforms
            scale_data_a = shape_data[mesh_shape_a]
            scale_data_b = shape_data[mesh_shape_b]
            mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
            mesh_scale_b = wp.vec3(scale_data_b[0], scale_data_b[1], scale_data_b[2])

            X_mesh_a_ws = shape_transform[mesh_shape_a]
            X_mesh_b_ws = shape_transform[mesh_shape_b]

            # Get thickness values
            thickness_a = shape_data[mesh_shape_a][3]
            thickness_b = shape_data[mesh_shape_b][3]

            # Use per-geometry cutoff for contact detection
            cutoff_a = shape_contact_margin[mesh_shape_a]
            cutoff_b = shape_contact_margin[mesh_shape_b]
            margin = wp.max(cutoff_a, cutoff_b)

            # Build pair key for this mesh-mesh pair
            pair_key = build_pair_key2(wp.uint32(mesh_shape_a), wp.uint32(mesh_shape_b))

            # Test both directions: mesh A against SDF B, and mesh B against SDF A
            for mode in range(2):
                if mode == 0:
                    # Process mesh A triangles against SDF B (if SDF B exists)
                    if sdf_ptr_b == wp.uint64(0):
                        continue

                    mesh_id = mesh_id_a
                    mesh_scale = mesh_scale_a
                    sdf_ptr = sdf_ptr_b
                    # Transform from mesh A space to mesh B space
                    X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_mesh_b_ws), X_mesh_a_ws)
                    X_sdf_ws = X_mesh_b_ws
                    mesh = mesh_a
                    shape_a = mesh_shape_a
                    shape_b = mesh_shape_b
                    thickness_mesh = thickness_a
                    thickness_sdf = thickness_b
                else:
                    # Process mesh B triangles against SDF A (if SDF A exists)
                    if sdf_ptr_a == wp.uint64(0):
                        continue

                    mesh_id = mesh_id_b
                    mesh_scale = mesh_scale_b
                    sdf_ptr = sdf_ptr_a
                    # Transform from mesh B space to mesh A space
                    X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_mesh_a_ws), X_mesh_b_ws)
                    X_sdf_ws = X_mesh_a_ws
                    mesh = mesh_b
                    shape_a = mesh_shape_b
                    shape_b = mesh_shape_a
                    thickness_mesh = thickness_b
                    thickness_sdf = thickness_a

                num_tris = mesh.indices.shape[0] // 3
                # strided loop over triangles
                for tri_idx in range(t, num_tris, wp.block_dim()):
                    # Get triangle vertices in SDF's local space
                    v0, v1, v2 = get_triangle_from_mesh(mesh_id, mesh_scale, X_mesh_to_sdf, tri_idx)

                    # Early out: check bounding sphere distance to SDF surface
                    bounding_sphere_center, bounding_sphere_radius = get_bounding_sphere(v0, v1, v2)
                    query_local = wp.volume_world_to_index(sdf_ptr, bounding_sphere_center)
                    sdf_dist = wp.volume_sample_f(sdf_ptr, query_local, wp.Volume.LINEAR)

                    # Skip triangles that are too far from the SDF surface
                    if sdf_dist > (bounding_sphere_radius + margin):
                        continue

                    dist, point, direction = doTriangleSDFCollision(sdf_ptr, v0, v1, v2)

                    if dist < margin:
                        point_world = wp.transform_point(X_sdf_ws, point)

                        direction_world = wp.transform_vector(X_sdf_ws, direction)
                        direction_len = wp.length(direction_world)
                        if direction_len > 0.0:
                            direction_world = direction_world / direction_len

                        # Create contact data
                        contact_data = ContactData()
                        contact_data.contact_point_center = point_world
                        contact_data.contact_normal_a_to_b = (
                            -direction_world
                        )  # Negate: gradient points B->A, we need A->B
                        contact_data.contact_distance = dist
                        contact_data.radius_eff_a = 0.0
                        contact_data.radius_eff_b = 0.0
                        contact_data.thickness_a = thickness_mesh
                        contact_data.thickness_b = thickness_sdf
                        contact_data.shape_a = shape_a
                        contact_data.shape_b = shape_b
                        contact_data.margin = margin
                        contact_data.feature = wp.uint32(tri_idx + 1)
                        contact_data.feature_pair_key = pair_key

                        writer_func(contact_data, writer_data)

    @wp.kernel
    def mesh_sdf_collision_reduce_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_sdf: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        writer_data: Any,
        total_num_blocks: int,
    ):
        block_id, t = wp.tid()
        num_pairs = shape_pairs_mesh_mesh_count[0]
        if block_id >= num_pairs:
            return

        pair = shape_pairs_mesh_mesh[block_id]
        mesh0 = shape_source[pair[0]]
        mesh1 = shape_source[pair[1]]
        sdf0 = shape_sdf[pair[0]]
        sdf1 = shape_sdf[pair[1]]

        # Extract mesh parameters
        mesh0_data = shape_data[pair[0]]
        mesh1_data = shape_data[pair[1]]
        mesh0_scale = wp.vec3(mesh0_data[0], mesh0_data[1], mesh0_data[2])
        mesh1_scale = wp.vec3(mesh1_data[0], mesh1_data[1], mesh1_data[2])
        mesh0_transform = shape_transform[pair[0]]
        mesh1_transform = shape_transform[pair[1]]

        thickness0 = mesh0_data[3]
        thickness1 = mesh1_data[3]

        margin = wp.max(shape_contact_margin[pair[0]], shape_contact_margin[pair[1]])

        # Initialize (shared memory) buffers for contact reduction
        empty_marker = -1000000000.0
        has_contact = True

        active_contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_141_ints(), shape=(141,), dtype=wp.int32)
        contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_140_contacts(), shape=(140,), dtype=ContactStruct)

        for i in range(t, 140, wp.block_dim()):
            contacts_shared_mem[i].projection = empty_marker

        if t == 0:
            active_contacts_shared_mem[140] = 0

        # Initialize (shared memory) buffers for triangle selection
        tri_capacity = wp.block_dim()
        selected_triangles = wp.array(
            ptr=get_shared_memory_pointer_block_dim_plus_2_ints(), shape=(wp.block_dim() + 2,), dtype=wp.int32
        )

        if t == 0:
            selected_triangles[tri_capacity] = 0  # Stores the number of indices inside selected_triangles
            selected_triangles[tri_capacity + 1] = (
                0  # Stores the number of triangles from the mesh that were already investigated
            )


        for mode in range(2):
            synchronize()
            if mode == 0:
                mesh = mesh0
                mesh_scale = mesh0_scale
                mesh_sdf_transform = wp.transform_multiply(wp.transform_inverse(mesh1_transform), mesh0_transform)
                sdf_current = sdf1
                sdf_transform = mesh1_transform
            else:
                mesh = mesh1
                mesh_scale = mesh1_scale
                mesh_sdf_transform = wp.transform_multiply(wp.transform_inverse(mesh0_transform), mesh1_transform)
                sdf_current = sdf0
                sdf_transform = mesh0_transform

            # Reset progress counter for this mesh
            if t == 0:
                selected_triangles[tri_capacity + 1] = 0
            synchronize()

            num_tris = wp.mesh_get(mesh).indices.shape[0] // 3

            has_contact = wp.bool(False)

            while selected_triangles[tri_capacity + 1] < num_tris:
                findInterestingTriangles(
                    t,
                    mesh_scale,
                    mesh_sdf_transform,
                    mesh,
                    sdf_current,
                    selected_triangles,
                    margin,
                )

                has_contact = t < selected_triangles[tri_capacity]
                synchronize()
                c = ContactStruct()

                if has_contact:
                    v0, v1, v2 = get_triangle_from_mesh(mesh, mesh_scale, mesh_sdf_transform, selected_triangles[t])
                    dist, point, dir = doTriangleSDFCollision(
                        sdf_current,
                        v0,
                        v1,
                        v2,
                    )

                    has_contact = dist < margin

                    if has_contact:
                        # Transform to world space BEFORE storing in reduction buffer
                        # This is critical because contacts from mode 0 and mode 1 are in
                        # different local coordinate systems (SDF B space vs SDF A space)
                        point_world = wp.transform_point(sdf_transform, point)
                        dir_world = wp.transform_vector(sdf_transform, dir)
                        dir_len = wp.length(dir_world)
                        if dir_len > 0.0:
                            dir_world = dir_world / dir_len

                        # Normalize normal direction so it always points from pair[0] to pair[1]
                        # Mode 0: gradient points B->A (pair[1]->pair[0]), negate to get pair[0]->pair[1]
                        # Mode 1: gradient points A->B (pair[0]->pair[1]), already correct
                        if mode == 0:
                            dir_world = -dir_world

                        c.position = point_world
                        c.normal = dir_world  # Store normalized world-space normal pointing pair[0]->pair[1]
                        c.depth = dist
                        c.feature = selected_triangles[t]
                        c.projection = empty_marker

                store_reduced_contact(
                    t, has_contact, c, contacts_shared_mem, active_contacts_shared_mem, 140, empty_marker
                )

                # Reset buffer for next batch
                synchronize()
                if t == 0:
                    selected_triangles[tri_capacity] = 0
                synchronize()

        # Now write the reduced contacts to the output array
        # Contacts are already in world space (transformed before storing in reduction buffer)
        # All contacts use consistent convention: shape_a = pair[0], shape_b = pair[1],
        # normal points from pair[0] to pair[1]
        synchronize()
        num_contacts_to_keep = wp.min(active_contacts_shared_mem[140], 140)
        synchronize()

        for i in range(t, num_contacts_to_keep, wp.block_dim()):
            contact_id = active_contacts_shared_mem[i]
            contact = contacts_shared_mem[contact_id]

            # Create contact data - contacts are already in world space
            contact_data = ContactData()
            contact_data.contact_point_center = contact.position
            contact_data.contact_normal_a_to_b = contact.normal  # Already normalized and pointing pair[0]->pair[1]
            contact_data.contact_distance = contact.depth
            contact_data.radius_eff_a = 0.0
            contact_data.radius_eff_b = 0.0
            # Use consistent shape IDs for all contacts (both modes contribute to same buffer)
            contact_data.thickness_a = thickness0
            contact_data.thickness_b = thickness1
            contact_data.shape_a = pair[0]
            contact_data.shape_b = pair[1]
            contact_data.margin = margin
            contact_data.feature = wp.uint32(contact.feature + 1)
            contact_data.feature_pair_key = build_pair_key2(wp.uint32(pair[0]), wp.uint32(pair[1]))

            writer_func(contact_data, writer_data)

    if reduce_contacts:
        return mesh_sdf_collision_reduce_kernel
    else:
        return mesh_sdf_collision_kernel
