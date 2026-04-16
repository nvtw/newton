# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""3D signed Jump Flooding Algorithm (JFA) for watertight mesh SDF construction.

Produces a dense signed distance field from a watertight triangle mesh without
expensive per-voxel winding-number or parity queries.  The algorithm:

1. **Scanline sign fill** -- for each (y, z) column, march a ray along +x
   through the mesh BVH counting surface crossings.  The sign flips at each
   crossing (parity rule for watertight meshes).  This is O(Y*Z) parallel
   rays with a few BVH queries each, instead of O(X*Y*Z) point queries.

2. **Seed boundary** -- voxels adjacent to a sign change become JFA seeds
   and record their own grid coordinates as the nearest known boundary point.

3. **JFA passes** -- ``ceil(log2(max_dim))`` passes with geometrically
   shrinking step sizes propagate the closest boundary point to every voxel.

4. **Extract** -- signed distance = ``sign * ||pos - closest_boundary||``.

The dense grid is a transient intermediate; callers extract the values they need
(coarse corners, subgrid samples) and let it be freed.

Algorithm reference: Rong & Tan, *Jump Flooding in GPU with Applications to
Voronoi Diagram and Distance Transform*, 2006.  Signed variant inspired by
Inigo Quilez's Shadertoy implementation (MIT license).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

UNRESOLVED = wp.constant(wp.int32(-1))

# Sign values stored in the grid.
SIGN_OUTSIDE = wp.constant(wp.int32(1))
SIGN_INSIDE = wp.constant(wp.int32(-1))


@wp.func
def _jfa_idx(x: int, y: int, z: int, sx: int, sy: int) -> int:
    return z * sx * sy + y * sx + x


# ---- Phase 1: scanline sign fill ----


@wp.kernel
def _scanline_sign_kernel(
    mesh: wp.uint64,
    sign_grid: wp.array[wp.int32],
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    size_x: int,
    size_y: int,
    size_z: int,
):
    """Fill sign for one (y, z) column by marching a ray along +x.

    For watertight meshes the sign flips at each ray-mesh intersection.
    """
    tid = wp.tid()
    total_cols = size_y * size_z
    if tid >= total_cols:
        return

    iz = tid // size_y
    iy = tid - iz * size_y

    ray_o = wp.vec3(
        min_corner[0] - cell_size[0],
        min_corner[1] + float(iy) * cell_size[1],
        min_corner[2] + float(iz) * cell_size[2],
    )
    ray_d = wp.vec3(1.0, 0.0, 0.0)
    max_t = float(size_x + 2) * cell_size[0]

    # Collect intersection t-values by marching along the ray.
    # We store up to 512 crossings; meshes with more crossings per column
    # are extremely unlikely at typical SDF resolutions.
    crossing_ts = wp.vector(dtype=float, length=512)
    num_crossings = int(0)
    t_start = float(0.0)
    eps = cell_size[0] * 0.01

    for _iter in range(512):
        query = wp.mesh_query_ray(mesh, ray_o + ray_d * t_start, ray_d, max_t - t_start)
        if query.result:
            t_hit = t_start + query.t
            if num_crossings < 512:
                crossing_ts[num_crossings] = t_hit
                num_crossings = num_crossings + 1
            t_start = t_hit + eps
        else:
            break

    # Walk along x and assign sign.  The ray starts outside the mesh
    # (origin is one cell before min_corner).
    crossing_idx = int(0)
    sign = SIGN_OUTSIDE

    for ix in range(size_x):
        voxel_t = (float(ix) * cell_size[0]) + cell_size[0]

        # Advance past all crossings up to this voxel center.
        for _c in range(512):
            if crossing_idx >= num_crossings:
                break
            if crossing_ts[crossing_idx] > voxel_t:
                break
            if sign == SIGN_OUTSIDE:
                sign = SIGN_INSIDE
            else:
                sign = SIGN_OUTSIDE
            crossing_idx = crossing_idx + 1

        idx = _jfa_idx(ix, iy, iz, size_x, size_y)
        sign_grid[idx] = sign


# ---- Phase 2: seed boundary voxels ----


@wp.kernel
def _seed_boundary_kernel(
    sign_grid: wp.array[wp.int32],
    closest_x: wp.array[wp.int32],
    closest_y: wp.array[wp.int32],
    closest_z: wp.array[wp.int32],
    size_x: int,
    size_y: int,
    size_z: int,
):
    """Seed voxels adjacent to a sign change with their own coordinates."""
    tid = wp.tid()
    total = size_x * size_y * size_z
    if tid >= total:
        return

    z = tid // (size_x * size_y)
    rem = tid - z * size_x * size_y
    y = rem // size_x
    x = rem - y * size_x

    my_sign = sign_grid[tid]
    is_boundary = False

    if x > 0 and sign_grid[_jfa_idx(x - 1, y, z, size_x, size_y)] != my_sign:
        is_boundary = True
    if x < size_x - 1 and sign_grid[_jfa_idx(x + 1, y, z, size_x, size_y)] != my_sign:
        is_boundary = True
    if y > 0 and sign_grid[_jfa_idx(x, y - 1, z, size_x, size_y)] != my_sign:
        is_boundary = True
    if y < size_y - 1 and sign_grid[_jfa_idx(x, y + 1, z, size_x, size_y)] != my_sign:
        is_boundary = True
    if z > 0 and sign_grid[_jfa_idx(x, y, z - 1, size_x, size_y)] != my_sign:
        is_boundary = True
    if z < size_z - 1 and sign_grid[_jfa_idx(x, y, z + 1, size_x, size_y)] != my_sign:
        is_boundary = True

    if is_boundary:
        closest_x[tid] = x
        closest_y[tid] = y
        closest_z[tid] = z


# ---- Phase 3: JFA passes ----


@wp.kernel
def _jfa_pass_kernel(
    cx_in: wp.array[wp.int32],
    cy_in: wp.array[wp.int32],
    cz_in: wp.array[wp.int32],
    cx_out: wp.array[wp.int32],
    cy_out: wp.array[wp.int32],
    cz_out: wp.array[wp.int32],
    step: int,
    size_x: int,
    size_y: int,
    size_z: int,
):
    """One JFA pass: check 27 neighbors at ``step`` distance."""
    tid = wp.tid()
    total = size_x * size_y * size_z
    if tid >= total:
        return

    pz = tid // (size_x * size_y)
    pr = tid - pz * size_x * size_y
    py = pr // size_x
    px = pr - py * size_x

    bx = cx_in[tid]
    by = cy_in[tid]
    bz = cz_in[tid]

    if bx != UNRESOLVED:
        best2 = float((px - bx) * (px - bx) + (py - by) * (py - by) + (pz - bz) * (pz - bz))
    else:
        best2 = 1.0e20

    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                qx = px + dx * step
                qy = py + dy * step
                qz = pz + dz * step
                if qx < 0 or qx >= size_x or qy < 0 or qy >= size_y or qz < 0 or qz >= size_z:
                    continue
                qi = _jfa_idx(qx, qy, qz, size_x, size_y)
                nx = cx_in[qi]
                if nx == UNRESOLVED:
                    continue
                ny = cy_in[qi]
                nz = cz_in[qi]
                d2 = float((px - nx) * (px - nx) + (py - ny) * (py - ny) + (pz - nz) * (pz - nz))
                if d2 < best2:
                    best2 = d2
                    bx = nx
                    by = ny
                    bz = nz

    cx_out[tid] = bx
    cy_out[tid] = by
    cz_out[tid] = bz


# ---- Phase 4: extract signed distance ----


@wp.kernel
def _extract_sdf_kernel(
    sign_grid: wp.array[wp.int32],
    closest_x: wp.array[wp.int32],
    closest_y: wp.array[wp.int32],
    closest_z: wp.array[wp.int32],
    sdf_out: wp.array[float],
    cell_size: wp.vec3,
    size_x: int,
    size_y: int,
    size_z: int,
):
    """Compute ``sign * distance_to_closest_boundary`` per voxel."""
    tid = wp.tid()
    total = size_x * size_y * size_z
    if tid >= total:
        return

    z = tid // (size_x * size_y)
    rem = tid - z * size_x * size_y
    y = rem // size_x
    x = rem - y * size_x

    cx = closest_x[tid]
    cy = closest_y[tid]
    cz = closest_z[tid]

    if cx == UNRESOLVED:
        sdf_out[tid] = float(sign_grid[tid]) * 10000.0
        return

    dx = float(x - cx) * cell_size[0]
    dy = float(y - cy) * cell_size[1]
    dz = float(z - cz) * cell_size[2]
    dist = wp.sqrt(dx * dx + dy * dy + dz * dz)

    sdf_out[tid] = float(sign_grid[tid]) * dist


# ---- Host-side orchestration ----


def compute_dense_sdf_jfa(
    mesh: wp.Mesh,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    cell_size: np.ndarray,
    min_corner: np.ndarray,
    device: str = "cuda",
) -> wp.array:
    """Compute a dense signed distance field using scanline sign + 3D JFA.

    Only requires a standard ``wp.Mesh`` — no ``support_winding_number``.

    The cost is dominated by O(Y*Z) ray-march queries for sign (one per
    column) plus O(log(max_dim)) JFA passes over the dense grid.  This is
    much cheaper than O(X*Y*Z) per-voxel winding-number or parity queries.

    Args:
        mesh: Warp mesh (``support_winding_number`` not required).
        grid_size_x: Grid X dimension [voxels].
        grid_size_y: Grid Y dimension [voxels].
        grid_size_z: Grid Z dimension [voxels].
        cell_size: Voxel size per axis [m], shape ``(3,)``.
        min_corner: Lower corner of the domain [m], shape ``(3,)``.
        device: Warp device string.

    Returns:
        ``wp.array[float]`` of length ``grid_size_x * grid_size_y * grid_size_z``
        containing the signed distance at each voxel (row-major XYZ order).
    """
    total = grid_size_x * grid_size_y * grid_size_z
    mc = wp.vec3(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
    cs = wp.vec3(float(cell_size[0]), float(cell_size[1]), float(cell_size[2]))

    sign_grid = wp.zeros(total, dtype=wp.int32, device=device)

    # Phase 1: scanline sign fill (one ray per Y*Z column)
    num_cols = grid_size_y * grid_size_z
    wp.launch(_scanline_sign_kernel, dim=num_cols, inputs=[mesh.id, sign_grid, mc, cs, grid_size_x, grid_size_y, grid_size_z], device=device)

    # Phase 2: seed boundary voxels
    cx_a = wp.full(total, UNRESOLVED, dtype=wp.int32, device=device)
    cy_a = wp.full(total, UNRESOLVED, dtype=wp.int32, device=device)
    cz_a = wp.full(total, UNRESOLVED, dtype=wp.int32, device=device)
    wp.launch(_seed_boundary_kernel, dim=total, inputs=[sign_grid, cx_a, cy_a, cz_a, grid_size_x, grid_size_y, grid_size_z], device=device)

    # Phase 3: JFA passes with ping-pong buffers
    cx_b = wp.zeros_like(cx_a)
    cy_b = wp.zeros_like(cy_a)
    cz_b = wp.zeros_like(cz_a)

    max_dim = max(grid_size_x, grid_size_y, grid_size_z)
    num_passes = max(1, int(math.ceil(math.log2(max_dim))))

    a_src = True
    for i in range(num_passes):
        step = 1 << (num_passes - 1 - i)
        src = (cx_a, cy_a, cz_a) if a_src else (cx_b, cy_b, cz_b)
        dst = (cx_b, cy_b, cz_b) if a_src else (cx_a, cy_a, cz_a)
        wp.launch(_jfa_pass_kernel, dim=total, inputs=[*src, *dst, step, grid_size_x, grid_size_y, grid_size_z], device=device)
        a_src = not a_src

    # JFA+1 and JFA+2 refinement passes
    for extra_step in (2, 1):
        src = (cx_a, cy_a, cz_a) if a_src else (cx_b, cy_b, cz_b)
        dst = (cx_b, cy_b, cz_b) if a_src else (cx_a, cy_a, cz_a)
        wp.launch(_jfa_pass_kernel, dim=total, inputs=[*src, *dst, extra_step, grid_size_x, grid_size_y, grid_size_z], device=device)
        a_src = not a_src

    # Phase 4: extract signed distance
    sdf_out = wp.zeros(total, dtype=float, device=device)
    final = (cx_a, cy_a, cz_a) if a_src else (cx_b, cy_b, cz_b)
    wp.launch(_extract_sdf_kernel, dim=total, inputs=[sign_grid, *final, sdf_out, cs, grid_size_x, grid_size_y, grid_size_z], device=device)

    return sdf_out
