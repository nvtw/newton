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
1. In narrow phase kernel, heightfield-convex pairs are detected and cell counts
   are computed using O(1) grid lookup. A cumulative sum is built atomically.
2. A processing kernel uses binary search on the cumsum to find which pair each
   thread should process, achieving perfect load balancing across all cells.

Each heightfield cell is split into two triangles for collision:
- Triangle 1: (v00, v10, v01)
- Triangle 2: (v01, v10, v11)

Where v_ij = mesh vertex at grid position (i, j).

Grid info is stored in shape_data:
- shape_data.x = half_extent_x
- shape_data.y = half_extent_y
- shape_data.z = cols (number of columns in grid)
- shape_data.w = thickness
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .collision_core import compute_tight_aabb_from_support, create_find_contacts
from .support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from .types import GeoType


@wp.func
def find_hfield_pair_from_cumsum(
    global_cell_idx: int,
    shape_pairs_hfield_cumsum: wp.array(dtype=wp.int32),
    num_pairs: int,
) -> tuple[int, int]:
    """Binary search to find which heightfield pair a global cell index belongs to.

    Args:
        global_cell_idx: Global cell index (0-based)
        shape_pairs_hfield_cumsum: Cumulative sum of cell counts (inclusive)
        num_pairs: Number of heightfield pairs

    Returns:
        Tuple of (pair_idx, local_cell_idx) where:
        - pair_idx: Index of the pair in shape_pairs_hfield
        - local_cell_idx: Cell index within that pair
    """
    # Binary search for the pair that contains this global cell index
    # cumsum[i] contains the inclusive sum, so we search for the first
    # pair where cumsum[i] > global_cell_idx

    lo = int(0)
    hi = int(num_pairs)

    while lo < hi:
        mid = (lo + hi) // 2
        if shape_pairs_hfield_cumsum[mid] <= global_cell_idx:
            lo = mid + 1
        else:
            hi = mid

    pair_idx = lo

    # Compute local cell index within this pair
    if pair_idx > 0:
        local_cell_idx = global_cell_idx - shape_pairs_hfield_cumsum[pair_idx - 1]
    else:
        local_cell_idx = global_cell_idx

    return pair_idx, local_cell_idx


@wp.func
def compute_hfield_cell_range(
    hfield_shape: int,
    convex_shape: int,
    shape_types: wp.array(dtype=wp.int32),
    shape_data: wp.array(dtype=wp.vec4),
    shape_transform: wp.array(dtype=wp.transform),
    shape_source: wp.array(dtype=wp.uint64),
    shape_contact_margin: wp.array(dtype=wp.float32),
) -> tuple[int, int, int, int, int, int, float, float]:
    """Compute the cell range for a heightfield-convex pair.

    Args:
        hfield_shape: Heightfield shape index
        convex_shape: Convex shape index
        shape_types, shape_data, shape_transform, shape_source: Shape arrays
        shape_contact_margin: Contact margin array

    Returns:
        Tuple of (min_i, min_j, num_cells_i, num_cells_j, cols, rows, cell_size_x, cell_size_y)
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
    convex_type = shape_types[convex_shape]
    convex_data_vec4 = shape_data[convex_shape]
    convex_scale = wp.vec3(convex_data_vec4[0], convex_data_vec4[1], convex_data_vec4[2])

    generic_convex = GenericShapeData()
    generic_convex.shape_type = convex_type
    generic_convex.scale = convex_scale
    generic_convex.auxiliary = wp.vec3(0.0, 0.0, 0.0)

    if convex_type == int(GeoType.CONVEX_MESH):
        generic_convex.auxiliary = pack_mesh_ptr(shape_source[convex_shape])

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

    return min_i, min_j, num_cells_i, num_cells_j, cols, rows, cell_size_x, cell_size_y


@wp.func
def build_triangle_shape_data(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
) -> tuple[GenericShapeData, wp.vec3, wp.quat]:
    """Build GenericShapeData for a triangle.

    The triangle is encoded with v0 at the origin, and edges to v1 and v2
    stored in scale and auxiliary respectively.

    Args:
        v0, v1, v2: Triangle vertices in world space

    Returns:
        Tuple of (shape_data, position, orientation) where:
        - shape_data: GenericShapeData for the triangle
        - position: v0 (the reference vertex position)
        - orientation: identity quaternion (triangle is in world space)
    """
    shape_data = GenericShapeData()
    shape_data.shape_type = int(GeoTypeEx.TRIANGLE)
    shape_data.scale = v1 - v0  # Edge from v0 to v1
    shape_data.auxiliary = v2 - v0  # Edge from v0 to v2

    return shape_data, v0, wp.quat_identity()


def create_heightfield_process_cell_contacts_kernel(writer_func: Any):
    """Create a kernel for processing heightfield cell contacts.

    Uses binary search on the cumulative sum to find which pair each thread
    should process, achieving perfect load balancing.

    Each cell is split into two triangles, and collision is tested using
    the existing GJK/MPR infrastructure with GeoTypeEx.TRIANGLE.

    Args:
        writer_func: Contact writer function

    Returns:
        The kernel function
    """
    # Create the find_contacts function using the existing infrastructure
    find_contacts_func = create_find_contacts(writer_func)

    @wp.kernel(enable_backward=False)
    def heightfield_process_cell_contacts_kernel(
        # Shape data
        shape_types: wp.array(dtype=wp.int32),
        shape_transform: wp.array(dtype=wp.transform),
        shape_data: wp.array(dtype=wp.vec4),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=wp.float32),
        # Heightfield pairs from narrow phase
        shape_pairs_hfield: wp.array(dtype=wp.vec2i),
        shape_pairs_hfield_count: wp.array(dtype=wp.int32),
        shape_pairs_hfield_cumsum: wp.array(dtype=wp.int32),
        hfield_cell_total_count: wp.array(dtype=wp.int32),
        # Contact writer data
        writer_data: Any,
        # Thread configuration
        total_num_threads: int,
    ):
        """Process heightfield cell contacts with convex shapes.

        Uses binary search on cumsum for perfect load balancing.
        Each cell is split into two triangles for collision testing.
        """
        tid = wp.tid()

        total_cells = hfield_cell_total_count[0]
        num_pairs = shape_pairs_hfield_count[0]

        if num_pairs == 0:
            return

        # Grid stride loop over all cells
        for global_cell_idx in range(tid, total_cells, total_num_threads):
            # Binary search to find which pair this cell belongs to
            pair_idx, local_cell_idx = find_hfield_pair_from_cumsum(
                global_cell_idx, shape_pairs_hfield_cumsum, num_pairs
            )

            if pair_idx >= num_pairs:
                continue

            # Get the pair
            pair = shape_pairs_hfield[pair_idx]
            hfield_shape = pair[0]
            convex_shape = pair[1]

            # Recompute cell range for this pair
            min_i, min_j, num_cells_i, num_cells_j, cols, rows, cell_size_x, cell_size_y = compute_hfield_cell_range(
                hfield_shape,
                convex_shape,
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_contact_margin,
            )

            # Convert local cell index to (cell_i, cell_j)
            local_i = local_cell_idx // num_cells_j
            local_j = local_cell_idx % num_cells_j
            cell_i = min_i + local_i
            cell_j = min_j + local_j

            # Get heightfield mesh and transform
            hfield_mesh_id = shape_source[hfield_shape]
            hfield_mesh = wp.mesh_get(hfield_mesh_id)
            X_hfield_ws = shape_transform[hfield_shape]

            # Get cell vertices from mesh (vertices stored in row-major order)
            v00_idx = cell_j * cols + cell_i
            v10_idx = cell_j * cols + (cell_i + 1)
            v01_idx = (cell_j + 1) * cols + cell_i
            v11_idx = (cell_j + 1) * cols + (cell_i + 1)

            # Get local vertices and transform to world space
            v00 = wp.transform_point(X_hfield_ws, hfield_mesh.points[v00_idx])
            v10 = wp.transform_point(X_hfield_ws, hfield_mesh.points[v10_idx])
            v01 = wp.transform_point(X_hfield_ws, hfield_mesh.points[v01_idx])
            v11 = wp.transform_point(X_hfield_ws, hfield_mesh.points[v11_idx])

            # Build convex shape data
            X_convex_ws = shape_transform[convex_shape]
            convex_type = shape_types[convex_shape]
            convex_data_vec4 = shape_data[convex_shape]
            convex_scale = wp.vec3(convex_data_vec4[0], convex_data_vec4[1], convex_data_vec4[2])
            convex_thickness = convex_data_vec4[3]

            convex_pos = wp.transform_get_translation(X_convex_ws)
            convex_quat = wp.transform_get_rotation(X_convex_ws)

            generic_convex = GenericShapeData()
            generic_convex.shape_type = convex_type
            generic_convex.scale = convex_scale
            generic_convex.auxiliary = wp.vec3(0.0, 0.0, 0.0)

            if convex_type == int(GeoType.CONVEX_MESH):
                generic_convex.auxiliary = pack_mesh_ptr(shape_source[convex_shape])

            margin = shape_contact_margin[convex_shape]
            hfield_thickness = shape_data[hfield_shape][3]

            # Triangle 1: (v00, v10, v01)
            tri1_data, tri1_pos, tri1_quat = build_triangle_shape_data(v00, v10, v01)

            wp.static(find_contacts_func)(
                tri1_pos,
                convex_pos,
                tri1_quat,
                convex_quat,
                tri1_data,
                generic_convex,
                False,  # is_infinite_plane_a
                False,  # is_infinite_plane_b
                0.0,  # bsphere_radius_a (not used for triangles)
                0.0,  # bsphere_radius_b (not used)
                margin,
                hfield_shape,
                convex_shape,
                hfield_thickness,
                convex_thickness,
                writer_data,
            )

            # Triangle 2: (v01, v10, v11)
            tri2_data, tri2_pos, tri2_quat = build_triangle_shape_data(v01, v10, v11)

            wp.static(find_contacts_func)(
                tri2_pos,
                convex_pos,
                tri2_quat,
                convex_quat,
                tri2_data,
                generic_convex,
                False,  # is_infinite_plane_a
                False,  # is_infinite_plane_b
                0.0,  # bsphere_radius_a
                0.0,  # bsphere_radius_b
                margin,
                hfield_shape,
                convex_shape,
                hfield_thickness,
                convex_thickness,
                writer_data,
            )

    return heightfield_process_cell_contacts_kernel
