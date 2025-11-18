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

import warp as wp

# Handle both direct execution and module import
try:
    from .contact_reduction import (
        ContactStruct,
        create_contact_reduction_func,
        get_shared_memory_pointer_120_contacts,
        get_shared_memory_pointer_121_ints,
        get_shared_memory_pointer_block_dim_plus_2_ints,
        synchronize,
    )
except ImportError:
    from contact_reduction import (
        ContactStruct,
        create_contact_reduction_func,
        get_shared_memory_pointer_120_contacts,
        get_shared_memory_pointer_121_ints,
        get_shared_memory_pointer_block_dim_plus_2_ints,
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
    texture: wp.uint64,
    sdfBoxLower: wp.vec3,
    sdfBoxHigher: wp.vec3,
    invSdfDx: wp.vec3,
    sdfDims: wp.vec3,
    sdfDx: float,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    tolerance: float,
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
        texture: The SDF volume texture
        sdfBoxLower: Lower bounds of SDF volume in world space
        sdfBoxHigher: Upper bounds of SDF volume in world space
        invSdfDx: Inverse voxel spacing
        sdfDims: SDF volume dimensions in voxels
        sdfDx: SDF voxel spacing
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

    dist = PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, p)

    d0 = PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, v0)
    d1 = PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, v1)
    d2 = PxSdfDistance(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, v2)

    # PxVec3 nor = (v1 - v0).cross(v2 - v0);

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
        sdfGradient = PxVolumeGrad(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, sdfDims, sdfDx, p)

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

    sdfGradient = PxVolumeGrad(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, sdfDims, sdfDx, p)

    point = p

    dist, dir = PxSdfSampleWithGrad(texture, sdfBoxLower, sdfBoxHigher, invSdfDx, p, sdfGradient, tolerance)

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
    # Get the mesh object from the ID
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
    radius = wp.max(wp.length(v0 - center), wp.length(v1 - center), wp.length(v2 - center))
    return center, radius


@wp.func
def add_to_shared_buffer(
    idx_in_thread_block: int,
    add_triangle: bool,
    tri_idx: int,
    selected_triangle_index_buffer_shared_mem: wp.array(dtype=wp.int32),
):
    """
    Add a triangle index to a shared memory buffer using parallel scan.

    This function uses warp-level primitives (tile operations) to efficiently
    add triangle indices to a shared buffer in parallel. Multiple threads can
    call this simultaneously, and the scan operation ensures each triangle is
    added to a unique position without conflicts.

    The buffer layout is:
    - [0 .. block_dim-1]: Triangle indices
    - [block_dim]: Current count of triangles in buffer

    Args:
        idx_in_thread_block: The calling thread's index within the thread block
        add_triangle: Whether this thread wants to add a triangle
        tri_idx: The triangle index to add (only used if add_triangle is True)
        selected_triangle_index_buffer_shared_mem: Shared memory buffer for triangle indices

    Note:
        - Uses inclusive scan to compute write positions
        - Automatically handles buffer overflow by capping at block_dim
        - The last thread (block_dim-1) updates the global count
        - All threads must call this together (implicit synchronization via tile ops)
    """
    val = 1 if add_triangle else 0
    add_tile = wp.tile(val)
    inclusive_scan = wp.tile_scan_inclusive(add_tile)
    offset = 0
    if idx_in_thread_block == wp.block_dim() - 1:
        offset = selected_triangle_index_buffer_shared_mem[wp.block_dim()]
        num_added = inclusive_scan[wp.block_dim() - 1]
        selected_triangle_index_buffer_shared_mem[wp.block_dim()] += num_added
        overflow = selected_triangle_index_buffer_shared_mem[wp.block_dim()] - wp.block_dim()
        if overflow > 0:
            num_added -= overflow
            selected_triangle_index_buffer_shared_mem[wp.block_dim()] = wp.block_dim()

    offset_broadcast_tile = wp.tile(offset)
    offset_broadcast = offset_broadcast_tile[wp.block_dim() - 1]

    write_index = offset_broadcast + inclusive_scan[idx_in_thread_block] - val
    if add_triangle and write_index < wp.block_dim():
        selected_triangle_index_buffer_shared_mem[write_index] = tri_idx

    synchronize()


@wp.func
def findInterestingTriangles(
    thread_id: int,
    mesh_scale: wp.vec3,
    mesh_to_sdf_transform: wp.transform,
    volume_aabb_lower: wp.vec3,
    volume_aabb_upper: wp.vec3,
    mesh0: wp.uint64,
    sdf_volume: wp.uint64,
    sdf_box_lower: wp.vec3,
    sdf_box_upper: wp.vec3,
    inv_sdf_dx: wp.vec3,
    selected_triangle_index_buffer_shared_mem: wp.array(dtype=wp.int32),
    contact_distance: float,
):
    """
    Broad-phase culling: fill shared buffer with triangle indices that may collide with SDF.

    Tests each triangle's bounding sphere against the SDF. Triangles are selected if
    SDF_distance(sphere_center) <= sphere_radius + contact_distance. Results are stored
    in shared memory buffer for narrow-phase processing.

    Buffer layout: [0..block_dim-1] = indices, [block_dim] = count, [block_dim+1] = progress
    """
    # Get the mesh object from the mesh ID
    mesh = wp.mesh_get(mesh0)

    base_tri_idx = selected_triangle_index_buffer_shared_mem[wp.block_dim() + 1]

    while (
        base_tri_idx < mesh.num_tris
        and selected_triangle_index_buffer_shared_mem[wp.block_dim()] < wp.block_dim()
    ):
        tri_idx = base_tri_idx + thread_id
        add_triangle = False
        if tri_idx < mesh.num_tris:
            v0, v1, v2 = get_triangle_from_mesh(mesh0, mesh_scale, mesh_to_sdf_transform, tri_idx)
            bounding_sphere_center, bounding_sphere_radius = get_bounding_sphere(v0, v1, v2)

            # Use SDF distance query instead of AABB overlap for more accurate culling
            # If the distance from sphere center to SDF surface is greater than sphere radius + contact distance,
            # the triangle cannot generate contacts and can be safely culled
            sdf_dist = PxSdfDistance(sdf_volume, sdf_box_lower, sdf_box_upper, inv_sdf_dx, bounding_sphere_center)
            add_triangle = sdf_dist <= (bounding_sphere_radius + contact_distance)

        add_to_shared_buffer(thread_id, add_triangle, tri_idx, selected_triangle_index_buffer_shared_mem)

        base_tri_idx += wp.block_dim()

    if thread_id == 0:
        selected_triangle_index_buffer_shared_mem[wp.block_dim() + 1] = wp.min(base_tri_idx, mesh.num_tris)


def create_doTriangleTriangleCollision(tile_size: int):
    """
    Create a warp kernel for triangle-SDF collision detection between mesh pairs.

    Args:
        tile_size: Tile size for contact reduction (must match contact reduction kernel)

    Returns:
        A warp kernel that processes mesh-SDF collision pairs in parallel.
    """
    @wp.kernel
    def doTriangleTriangleCollision(
        mesh_source: wp.array(dtype=wp.uint64),
        mesh_scales: wp.array(dtype=wp.vec3),
        mesh_transforms: wp.array(dtype=wp.transform),
        volume_data: wp.array(dtype=VolumeData),
        pairs: wp.array(dtype=wp.vec2i),
        pair_count: wp.array(dtype=int),
        contact_distance: float,
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
            volume_data: Array of VolumeData structs with SDF information
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
        volume0_data = volume_data[pair[0]]
        volume1_data = volume_data[pair[1]]

        # Extract mesh parameters
        mesh0_scale = mesh_scales[pair[0]]
        mesh1_scale = mesh_scales[pair[1]]
        mesh0_transform = mesh_transforms[pair[0]]
        mesh1_transform = mesh_transforms[pair[1]]

        # Initialize (shared memory) buffers for contact reduction
        empty_marker = -1000000000.0
        has_contact = True

        active_contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_121_ints(), shape=(121,), dtype=wp.int32)
        contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_120_contacts(), shape=(120,), dtype=ContactStruct)

        for i in range(t, 120, wp.block_dim()):
            contacts_shared_mem[i].projection = empty_marker

        if t == 0:
            active_contacts_shared_mem[120] = 0

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

        synchronize()

        for mode in range(2):
            if mode == 0:
                mesh = mesh0
                mesh_scale = mesh0_scale
                mesh_transform = mesh0_transform
                volume_data_current = volume1_data
            else:
                mesh = mesh1
                mesh_scale = mesh1_scale
                mesh_transform = mesh1_transform
                volume_data_current = volume0_data

            # Reset progress counter for this mesh
            if t == 0:
                selected_triangles[tri_capacity + 1] = 0
            synchronize()

            while selected_triangles[tri_capacity + 1] < wp.mesh_get(mesh).num_tris:
                findInterestingTriangles(
                    t,
                    mesh_scale,
                    mesh_transform,
                    volume_data_current.aabb_lower,
                    volume_data_current.aabb_upper,
                    mesh,
                    volume_data_current.volume_id,
                    volume_data_current.aabb_lower,
                    volume_data_current.aabb_upper,
                    volume_data_current.inv_sdf_dx,
                    selected_triangles,
                    contact_distance,
                )

                has_contact = t < selected_triangles[tri_capacity]
                c = ContactStruct()

                if has_contact:
                    v0, v1, v2 = get_triangle_from_mesh(mesh, mesh_scale, mesh_transform, selected_triangles[t])
                    dist, point, dir = doTriangleSDFCollision(
                        volume_data_current.volume_id,
                        volume_data_current.aabb_lower,
                        volume_data_current.aabb_upper,
                        volume_data_current.inv_sdf_dx,
                        volume_data_current.sdf_dims,
                        volume_data_current.sdf_dx,
                        v0,
                        v1,
                        v2,
                        contact_distance,
                    )

                    c.position = point
                    c.normal = dir
                    c.depth = dist
                    c.feature = 0
                    c.projection = empty_marker

                    has_contact = dist < contact_distance

                wp.static(create_contact_reduction_func(tile_size))(
                    t, has_contact, c, contacts_shared_mem, active_contacts_shared_mem, 120, empty_marker
                )

                # Reset buffer for next batch
                synchronize()
                if t == 0:
                    selected_triangles[tri_capacity] = 0
                synchronize()

        # Now write the reduced contacts to the output array
        num_contacts_to_keep = active_contacts_shared_mem[120]

        # Thread 0 writes the count
        if t == 0:
            out_count[0] = num_contacts_to_keep

        # All threads cooperatively write the contacts
        for i in range(t, num_contacts_to_keep, wp.block_dim()):
            contact_id = active_contacts_shared_mem[i]
            out_contacts[i] = contacts_shared_mem[contact_id]

    return doTriangleTriangleCollision
