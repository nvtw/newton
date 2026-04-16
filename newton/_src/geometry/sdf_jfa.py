# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scanline ray-march sign grid for watertight mesh SDF construction.

For each (y, z) column, marches a ray along +x through the mesh BVH counting
surface crossings.  The sign flips at each crossing (parity rule for watertight
meshes).  This is O(Y*Z) parallel rays with a few BVH queries each, instead of
O(X*Y*Z) point queries.

The resulting per-voxel sign grid is combined with unsigned BVH closest-point
queries in :mod:`sdf_texture` to produce exact signed distances.

Algorithm reference: signed variant inspired by Inigo Quilez's Shadertoy
implementation (MIT license).
"""

from __future__ import annotations

import warp as wp

SIGN_OUTSIDE = wp.constant(wp.int32(1))
SIGN_INSIDE = wp.constant(wp.int32(-1))


@wp.func
def _sign_idx(x: int, y: int, z: int, sx: int, sy: int) -> int:
    return z * sx * sy + y * sx + x


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
    """Fill sign for one (y, z) column by marching a ray along +x."""
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

    crossing_idx = int(0)
    sign = SIGN_OUTSIDE

    for ix in range(size_x):
        voxel_t = (float(ix) * cell_size[0]) + cell_size[0]

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

        idx = _sign_idx(ix, iy, iz, size_x, size_y)
        sign_grid[idx] = sign


def build_sign_grid(
    mesh: wp.Mesh,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    cell_size,
    min_corner,
    device: str = "cuda",
) -> wp.array:
    """Build a per-voxel inside/outside sign grid via scanline ray-march.

    Only requires a standard ``wp.Mesh`` -- no ``support_winding_number``.

    Args:
        mesh: Warp mesh.
        grid_size_x: Grid X dimension [voxels].
        grid_size_y: Grid Y dimension [voxels].
        grid_size_z: Grid Z dimension [voxels].
        cell_size: Voxel size per axis [m], shape ``(3,)``.
        min_corner: Lower corner of the domain [m], shape ``(3,)``.
        device: Warp device string.

    Returns:
        ``wp.array[wp.int32]`` of length ``grid_size_x * grid_size_y * grid_size_z``
        with ``+1`` (outside) or ``-1`` (inside) per voxel.
    """
    mc = wp.vec3(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
    cs = wp.vec3(float(cell_size[0]), float(cell_size[1]), float(cell_size[2]))

    total = grid_size_x * grid_size_y * grid_size_z
    sign_grid = wp.zeros(total, dtype=wp.int32, device=device)

    num_cols = grid_size_y * grid_size_z
    wp.launch(
        _scanline_sign_kernel,
        dim=num_cols,
        inputs=[mesh.id, sign_grid, mc, cs, grid_size_x, grid_size_y, grid_size_z],
        device=device,
    )
    return sign_grid
