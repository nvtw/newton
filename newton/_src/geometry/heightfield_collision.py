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

"""
Heightfield collision detection module.

This module implements collision detection between heightfields and convex shapes.
Heightfields are stored internally as meshes, but we exploit the regular grid
structure for O(1) cell lookup during the midphase.

Collision flow:
1. In narrow phase kernel, heightfield-convex pairs are detected and triangle counts
   are computed using O(1) grid lookup. A cumulative sum is built atomically.
2. A midphase kernel uses binary search on the cumsum to find which pair each
   thread should process, emitting triangle pairs to the SAME buffer as mesh collision.
3. The shared triangle narrow phase kernel processes all triangle-convex pairs.

Each heightfield cell is split into two triangles for collision:
- Triangle 0 (per cell): (v00, v10, v01)
- Triangle 1 (per cell): (v01, v10, v11)

Where v_ij = mesh vertex at grid position (i, j).
Triangle index in mesh = 2 * (cell_j * (cols-1) + cell_i) + [0 or 1]

Grid info is stored in shape_data:
- shape_data.x = half_extent_x
- shape_data.y = half_extent_y
- shape_data.z = cols (number of columns in grid)
- shape_data.w = thickness
"""

from __future__ import annotations

import warp as wp

from .collision_core import compute_tight_aabb_from_support, find_pair_from_cumulative_index
from .support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from .types import GeoType


@wp.func
def build_generic_convex_data(
    convex_shape: int,
    shape_types: wp.array(dtype=int),
    shape_data: wp.array(dtype=wp.vec4),
    shape_source: wp.array(dtype=wp.uint64),
) -> GenericShapeData:
    """Build GenericShapeData for a convex shape from arrays.

    Args:
        convex_shape: Index of the convex shape
        shape_types: Array of shape types
        shape_data: Array of shape data (scale xyz, thickness w)
        shape_source: Array of source pointers (mesh IDs, etc.)

    Returns:
        GenericShapeData struct for the convex shape
    """
    convex_type = shape_types[convex_shape]
    convex_data_vec4 = shape_data[convex_shape]
    convex_scale = wp.vec3(convex_data_vec4[0], convex_data_vec4[1], convex_data_vec4[2])

    generic_convex = GenericShapeData()
    generic_convex.shape_type = convex_type
    generic_convex.scale = convex_scale
    generic_convex.auxiliary = wp.vec3(0.0, 0.0, 0.0)

    if convex_type == int(GeoType.CONVEX_MESH):
        generic_convex.auxiliary = pack_mesh_ptr(shape_source[convex_shape])

    return generic_convex


@wp.func
def compute_hfield_cell_range(
    hfield_shape: int,
    convex_shape: int,
    shape_types: wp.array(dtype=int),
    shape_data: wp.array(dtype=wp.vec4),
    shape_transform: wp.array(dtype=wp.transform),
    shape_source: wp.array(dtype=wp.uint64),
    shape_contact_margin: wp.array(dtype=float),
) -> tuple[int, int, int, int, int, GenericShapeData, float]:
    """Compute the cell range for a heightfield-convex pair.

    Args:
        hfield_shape: Heightfield shape index
        convex_shape: Convex shape index
        shape_types, shape_data, shape_transform, shape_source: Shape arrays
        shape_contact_margin: Contact margin array

    Returns:
        Tuple of (min_i, min_j, num_cells_i, num_cells_j, cols, generic_convex, convex_aabb_lower_z)
        where convex_aabb_lower_z is in heightfield local space (for early-out Z culling)
    """
    # Get heightfield mesh and grid info from shape_data
    hfield_mesh_id = shape_source[hfield_shape]
    hfield_mesh = wp.mesh_get(hfield_mesh_id)
    hfield_data = shape_data[hfield_shape]

    half_extent_x = hfield_data[0]
    half_extent_y = hfield_data[1]
    cols = int(hfield_data[2])
    num_vertices = hfield_mesh.points.shape[0]
    rows = num_vertices // cols

    # Compute cell sizes
    cell_size_x = (2.0 * half_extent_x) / float(cols - 1)
    cell_size_y = (2.0 * half_extent_y) / float(rows - 1)

    # Get transforms
    X_hfield_ws = shape_transform[hfield_shape]
    X_convex_ws = shape_transform[convex_shape]

    # Transform convex to heightfield local space
    X_hfield_sw = wp.transform_inverse(X_hfield_ws)
    X_hfield_convex = wp.transform_multiply(X_hfield_sw, X_convex_ws)

    pos_in_hfield = wp.transform_get_translation(X_hfield_convex)
    quat_in_hfield = wp.transform_get_rotation(X_hfield_convex)

    # Build GenericShapeData for the convex shape
    generic_convex = build_generic_convex_data(convex_shape, shape_types, shape_data, shape_source)

    data_provider = SupportMapDataProvider()

    # Compute tight AABB of convex in heightfield local space
    aabb_lower, aabb_upper = compute_tight_aabb_from_support(
        generic_convex, quat_in_hfield, pos_in_hfield, data_provider
    )

    # Expand by contact margin
    margin = shape_contact_margin[convex_shape]
    margin_vec = wp.vec3(margin, margin, margin)
    aabb_lower = aabb_lower - margin_vec
    aabb_upper = aabb_upper + margin_vec

    # Compute cell range using O(1) grid lookup
    min_i = int(wp.floor((aabb_lower[0] + half_extent_x) / cell_size_x))
    max_i = int(wp.floor((aabb_upper[0] + half_extent_x) / cell_size_x))
    min_j = int(wp.floor((aabb_lower[1] + half_extent_y) / cell_size_y))
    max_j = int(wp.floor((aabb_upper[1] + half_extent_y) / cell_size_y))

    # Clamp to valid cell range
    num_cells_x = cols - 1
    num_cells_y = rows - 1

    min_i = wp.max(0, min_i)
    max_i = wp.min(num_cells_x - 1, max_i)
    min_j = wp.max(0, min_j)
    max_j = wp.min(num_cells_y - 1, max_j)

    num_cells_i = max_i - min_i + 1
    num_cells_j = max_j - min_j + 1

    # Return convex_aabb_lower_z for early-out Z culling (already margin-expanded)
    return min_i, min_j, num_cells_i, num_cells_j, cols, generic_convex, aabb_lower[2]


@wp.kernel(enable_backward=False)
def heightfield_midphase_kernel(
    # Shape data
    shape_types: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    shape_data: wp.array(dtype=wp.vec4),
    shape_source: wp.array(dtype=wp.uint64),
    shape_contact_margin: wp.array(dtype=float),
    # Heightfield pairs from narrow phase
    shape_pairs_hfield: wp.array(dtype=wp.vec2i),
    shape_pairs_hfield_count: wp.array(dtype=int),
    shape_pairs_hfield_cumsum: wp.array(dtype=int),
    hfield_triangle_total_count: wp.array(dtype=int),
    # Precomputed per-pair data (avoids expensive recomputation per triangle)
    shape_pairs_hfield_cell_range: wp.array(dtype=wp.vec4i),  # (min_i, min_j, num_cells_j, cols)
    shape_pairs_hfield_aabb_z: wp.array(dtype=float),  # convex AABB lower Z for early-out
    # Output: triangle pairs (shared with mesh collision)
    triangle_pairs: wp.array(dtype=wp.vec3i),  # (hfield_shape, convex_shape, triangle_idx)
    triangle_pairs_count: wp.array(dtype=int),
    # Thread configuration
    total_num_threads: int,
):
    """Heightfield midphase kernel - emits triangle pairs for narrow phase processing.

    Uses binary search on cumsum for perfect load balancing across all triangles.
    Each cell has 2 triangles, so cumsum tracks num_cells * 2 per pair.

    Emits to the SAME triangle_pairs buffer as mesh collision, enabling a unified
    triangle narrow phase for both mesh and heightfield collisions.
    """
    tid = wp.tid()

    total_triangles = hfield_triangle_total_count[0]
    num_pairs = shape_pairs_hfield_count[0]

    if num_pairs == 0:
        return

    # Grid stride loop over all triangles
    for global_tri_idx in range(tid, total_triangles, total_num_threads):
        # Binary search to find which pair this triangle belongs to
        pair_idx, local_tri_idx = find_pair_from_cumulative_index(
            global_tri_idx, shape_pairs_hfield_cumsum, num_pairs
        )

        if pair_idx >= num_pairs:
            continue

        # Get the pair
        pair = shape_pairs_hfield[pair_idx]
        hfield_shape = pair[0]
        convex_shape = pair[1]

        # Read precomputed cell range data (computed once per pair in narrow phase)
        cell_range = shape_pairs_hfield_cell_range[pair_idx]
        min_i = cell_range[0]
        min_j = cell_range[1]
        num_cells_j = cell_range[2]
        cols = cell_range[3]
        convex_aabb_lower_z = shape_pairs_hfield_aabb_z[pair_idx]

        # Convert local triangle index to cell and triangle-within-cell
        local_cell_idx = local_tri_idx // 2
        tri_in_cell = local_tri_idx % 2  # 0 or 1

        # Convert local cell index to (cell_i, cell_j)
        local_i = local_cell_idx // num_cells_j
        local_j = local_cell_idx % num_cells_j
        cell_i = min_i + local_i
        cell_j = min_j + local_j

        # Early-out Z culling: check if convex AABB is above cell max Z
        # Only check once per cell (for the first triangle)
        if tri_in_cell == 0:
            # Get heightfield mesh
            hfield_mesh_id = shape_source[hfield_shape]
            hfield_mesh = wp.mesh_get(hfield_mesh_id)

            # Get cell vertex indices (vertices stored in row-major order: vertex[j * cols + i])
            v00_idx = cell_j * cols + cell_i
            v10_idx = cell_j * cols + (cell_i + 1)
            v01_idx = (cell_j + 1) * cols + cell_i
            v11_idx = (cell_j + 1) * cols + (cell_i + 1)

            # Get local vertices Z values
            v00_z = hfield_mesh.points[v00_idx][2]
            v10_z = hfield_mesh.points[v10_idx][2]
            v01_z = hfield_mesh.points[v01_idx][2]
            v11_z = hfield_mesh.points[v11_idx][2]

            cell_max_z = wp.max(wp.max(v00_z, v10_z), wp.max(v01_z, v11_z))

            # Skip this cell (both triangles) if convex is above
            if convex_aabb_lower_z > cell_max_z:
                continue
        else:
            # For triangle 1, we need to check if the previous triangle was culled
            # Since we can't easily share state, we do the check again
            hfield_mesh_id = shape_source[hfield_shape]
            hfield_mesh = wp.mesh_get(hfield_mesh_id)

            v00_idx = cell_j * cols + cell_i
            v10_idx = cell_j * cols + (cell_i + 1)
            v01_idx = (cell_j + 1) * cols + cell_i
            v11_idx = (cell_j + 1) * cols + (cell_i + 1)

            v00_z = hfield_mesh.points[v00_idx][2]
            v10_z = hfield_mesh.points[v10_idx][2]
            v01_z = hfield_mesh.points[v01_idx][2]
            v11_z = hfield_mesh.points[v11_idx][2]

            cell_max_z = wp.max(wp.max(v00_z, v10_z), wp.max(v01_z, v11_z))

            if convex_aabb_lower_z > cell_max_z:
                continue

        # Compute the actual mesh triangle index
        # Mesh triangles are stored in order: for j in range(rows-1): for i in range(cols-1): 2 triangles
        # So triangle index = 2 * (cell_j * (cols - 1) + cell_i) + tri_in_cell
        num_cells_x = cols - 1
        mesh_tri_idx = 2 * (cell_j * num_cells_x + cell_i) + tri_in_cell

        # Emit triangle pair to the shared buffer
        out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)
        if out_idx < triangle_pairs.shape[0]:
            triangle_pairs[out_idx] = wp.vec3i(hfield_shape, convex_shape, mesh_tri_idx)
