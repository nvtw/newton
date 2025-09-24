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
Multi-contact manifold generation for collision detection.

This module implements contact manifold generation algorithms for computing
multiple contact points between colliding shapes. It includes polygon clipping,
feature tracking, and contact point selection algorithms.
"""

from typing import Any

import warp as wp

from .kernels import build_orthonormal_basis

# Constants
EPS = 0.00001
ROT_DELTA_ANGLE = 60.0 * wp.pi / 180.0
SIN_OFFSET = 0.017452406437283512  # sin(1 degree)
COS_OFFSET = 0.9998476951563912391  # cos(1 degree)


@wp.struct
class Fvec3:
    """
    A 3D vector with an associated feature ID.

    This struct stores a 3D point along with feature information
    used for contact manifold generation and feature tracking.
    """

    x: float
    y: float
    z: float
    feature: wp.uint32


@wp.func
def fvec3_get_xyz(fv: Fvec3) -> wp.vec3:
    """Get the XYZ components as a vec3."""
    return wp.vec3(fv.x, fv.y, fv.z)


@wp.func
def fvec3_set_xyz(fv: Fvec3, v: wp.vec3) -> Fvec3:
    """Set the XYZ components from a vec3."""
    result = fv
    result.x = v[0]
    result.y = v[1]
    result.z = v[2]
    return result


@wp.func
def signed_area(a: wp.vec2, b: wp.vec2, query_point: wp.vec2) -> float:
    """
    Calculates twice the signed area for the triangle (a, b, query_point).

    The result's sign indicates the triangle's orientation and is a robust way
    to check which side of a line a point is on.

    Args:
        a: The first vertex of the triangle and the start of the line segment.
        b: The second vertex of the triangle and the end of the line segment.
        query_point: The third vertex of the triangle, the point to test against the line a-b.

    Returns:
        The result's sign determines the orientation of the points:
        - Positive (> 0): The points are in a counter-clockwise (CCW) order.
          This means query_point is to the "left" of the directed line from a to b.
        - Negative (< 0): The points are in a clockwise (CW) order.
          This means query_point is to the "right" of the directed line from a to b.
        - Zero (== 0): The points are collinear; query_point lies on the infinite line defined by a and b.
    """
    # It returns twice the signed area of the triangle
    return (b[0] - a[0]) * (query_point[1] - a[1]) - (b[1] - a[1]) * (query_point[0] - a[0])


@wp.func
def signed_distance_to_line(a: wp.vec2, b: wp.vec2, query_point: wp.vec2) -> float:
    """
    Calculate the signed distance from a point to a line defined by two points.

    Args:
        a: First point defining the line.
        b: Second point defining the line.
        query_point: Point to calculate distance to.

    Returns:
        Signed distance from query_point to the line a-b.
    """
    area = signed_area(a, b, query_point)
    length = wp.sqrt(wp.max(1e-6, (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])))
    return area / length


@wp.func
def intersection_point(trim_seg_start: wp.vec3, trim_seg_end: wp.vec3, a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """
    Calculate the intersection point between a line segment and a polygon edge.

    It is known that a and b lie on different sides of the trim segment.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        a: First point of the polygon edge.
        b: Second point of the polygon edge.

    Returns:
        The intersection point as a vec4.
    """
    # Get 2D projections
    trim_start_xy = wp.vec2(trim_seg_start[0], trim_seg_start[1])
    trim_end_xy = wp.vec2(trim_seg_end[0], trim_seg_end[1])
    a_xy = wp.vec2(a[0], a[1])
    b_xy = wp.vec2(b[0], b[1])

    dist_a = wp.abs(signed_area(trim_start_xy, trim_end_xy, a_xy))
    dist_b = wp.abs(signed_area(trim_start_xy, trim_end_xy, b_xy))
    interp_ab = dist_a / (dist_a + dist_b)

    # Interpolate between a and b
    a_xyz = wp.vec3(a[0], a[1], a[2])
    b_xyz = wp.vec3(b[0], b[1], b[2])
    interpolated_ab = (1.0 - interp_ab) * a_xyz + interp_ab * b_xyz

    # Calculate projection along trim segment
    delta = trim_end_xy - trim_start_xy
    delta = wp.normalize(delta)
    interpolated_xy = wp.vec2(interpolated_ab[0], interpolated_ab[1])
    trim_start_xy_offset = interpolated_xy - trim_start_xy
    interp_start_end = wp.dot(delta, trim_start_xy_offset)  # This is not necessarily in the range 0...1

    w = (1.0 - interp_start_end) * trim_seg_start[2] + interp_start_end * trim_seg_end[2]

    return wp.vec4(interpolated_ab[0], interpolated_ab[1], interpolated_ab[2], w)


@wp.func
def trim_by_line_segment_in_place(
    trim_seg_start: wp.vec3,
    trim_seg_end: wp.vec3,
    loop: wp.array(dtype=wp.vec4),
    loop_segments: wp.array(dtype=wp.uint8),
    loop_count: int,
    max_loop_capacity: int,
) -> int:
    """
    Intersect the input loop with a single finite line segment in the contact plane.

    Stores up to two intersection points (degenerate segment manifold).
    Segment IDs are set to 0 (first trim edge) for generated points.
    """
    if loop_count < 1:
        return 0

    trim_start_xy = wp.vec2(trim_seg_start[0], trim_seg_start[1])
    trim_end_xy = wp.vec2(trim_seg_end[0], trim_seg_end[1])
    seg_dir = trim_end_xy - trim_start_xy
    seg_len = wp.sqrt(wp.max(1e-12, seg_dir[0] * seg_dir[0] + seg_dir[1] * seg_dir[1]))
    seg_dir = seg_dir / seg_len

    write_idx = int(0)
    num_found = int(0)

    for i in range(loop_count):
        if num_found >= 2:
            break
        j = (i + 1) % loop_count

        ai_xy = wp.vec2(loop[i][0], loop[i][1])
        aj_xy = wp.vec2(loop[j][0], loop[j][1])

        side_i = signed_area(trim_start_xy, trim_end_xy, ai_xy)
        side_j = signed_area(trim_start_xy, trim_end_xy, aj_xy)

        crosses = (side_i > 0.0 and side_j < 0.0) or (side_i < 0.0 and side_j > 0.0)
        touches = (side_i == 0.0) or (side_j == 0.0)

        if crosses or touches:
            inter = intersection_point(trim_seg_start, trim_seg_end, loop[i], loop[j])
            inter_xy = wp.vec2(inter[0], inter[1])
            t_along = wp.dot(seg_dir, inter_xy - trim_start_xy)
            if t_along < -1e-6 or t_along > seg_len + 1e-6:
                continue

            loop[write_idx] = inter
            loop_segments[write_idx] = wp.uint8(0)
            write_idx += 1
            num_found += 1

    return num_found


@wp.func
def insert_vec4(arr: wp.array(dtype=wp.vec4), arr_count: int, index: int, element: wp.vec4):
    """
    Insert an element into an array at the specified index, shifting elements to the right.

    Args:
        arr: Array to insert into.
        arr_count: Current number of elements in the array.
        index: Index at which to insert the element.
        element: Element to insert.
    """
    i = arr_count
    while i > index:
        arr[i] = arr[i - 1]
        i -= 1
    arr[index] = element


@wp.func
def insert_byte(arr: wp.array(dtype=wp.uint8), arr_count: int, index: int, element: wp.uint8):
    """
    Insert a byte element into an array at the specified index, shifting elements to the right.

    Args:
        arr: Array to insert into.
        arr_count: Current number of elements in the array.
        index: Index at which to insert the element.
        element: Element to insert.
    """
    i = arr_count
    while i > index:
        arr[i] = arr[i - 1]
        i -= 1
    arr[index] = element


@wp.func
def size_check(loop_indexer: int, max_loop_capacity: int):
    """
    Check if we have enough capacity in the loop array.

    Args:
        loop_indexer: Current loop index.
        max_loop_capacity: Maximum capacity of the loop array.
    """
    if loop_indexer + 1 >= max_loop_capacity:
        # In Warp, we can't use assertions, so we'll just return
        # The calling code should handle this gracefully
        pass


@wp.func
def trim_in_place(
    trim_seg_start: wp.vec3,
    trim_seg_end: wp.vec3,
    trim_seg_id: wp.uint8,
    loop: wp.array(dtype=wp.vec4),
    loop_seg_ids: wp.array(dtype=wp.uint8),
    loop_count: int,
    max_loop_capacity: int,
) -> int:
    """
    Trim a polygon in place using a line segment.

    The vec4 format is as follows:
    - X, Y: 2D coordinates projected onto the contact plane
    - Z: The offset out of the plane for the polygon called loop
    - W: The offset out of the plane for the trim segment

    The trim segment format is as follows:
    - X, Y: 2D coordinates projected onto the contact plane
    - Z: The offset out of the plane for the trim segment

    loopSegIds[0] refers to the segment from loop[0] to loop[1], etc.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        trim_seg_id: ID of the trimming segment.
        loop: Array of loop vertices.
        loop_seg_ids: Array of segment IDs for the loop.
        loop_count: Number of vertices in the loop.
        max_loop_capacity: Maximum capacity of the loop arrays.

    Returns:
        New number of vertices in the trimmed loop.
    """
    if loop_count < 3:
        return loop_count

    intersection_a = wp.vec4(0.0, 0.0, 0.0, 0.0)
    change_a = int(-1)
    change_a_seg_id = wp.uint8(255)
    intersection_b = wp.vec4(0.0, 0.0, 0.0, 0.0)
    change_b = int(-1)
    change_b_seg_id = wp.uint8(255)

    keep = bool(False)

    # Get 2D projections for the trim segment
    trim_start_xy = wp.vec2(trim_seg_start[0], trim_seg_start[1])
    trim_end_xy = wp.vec2(trim_seg_end[0], trim_seg_end[1])

    # Check first vertex
    loop0_xy = wp.vec2(loop[0][0], loop[0][1])
    prev_outside = bool(signed_area(trim_start_xy, trim_end_xy, loop0_xy) <= 0.0)

    for i in range(loop_count):
        next_idx = (i + 1) % loop_count
        loop_next_xy = wp.vec2(loop[next_idx][0], loop[next_idx][1])
        outside = signed_area(trim_start_xy, trim_end_xy, loop_next_xy) <= 0.0

        if outside != prev_outside:
            intersection = intersection_point(trim_seg_start, trim_seg_end, loop[i], loop[next_idx])
            if change_a < 0:
                change_a = i
                change_a_seg_id = loop_seg_ids[i]
                keep = not prev_outside
                intersection_a = intersection
            else:
                change_b = i
                change_b_seg_id = loop_seg_ids[i]
                intersection_b = intersection

        prev_outside = outside

    if change_a >= 0 and change_b >= 0:
        loop_indexer = int(-1)
        new_loop_count = int(loop_count)

        i = int(0)
        while i < loop_count:
            # If the current vertex is on the side to be kept, copy it and its segment ID.
            if keep:
                size_check(loop_indexer, max_loop_capacity)
                loop_indexer += 1
                loop[loop_indexer] = loop[i]
                loop_seg_ids[loop_indexer] = loop_seg_ids[i]

            # If the current edge is one of the two that intersects the trim line,
            # add the intersection point to the new polygon.
            if i == change_a or i == change_b:
                pt = intersection_a if i == change_a else intersection_b
                original_seg_id = change_a_seg_id if i == change_a else change_b_seg_id

                # Determine the correct ID for the segment starting at the new intersection point.
                # If we are currently keeping vertices (`keep` is true), it means we're transitioning
                # to a discarded section. The new segment connects the two intersection points,
                # so its ID is `trim_seg_id`.
                # If we are currently discarding vertices (`keep` is false), it means we're
                # transitioning to a kept section. The new segment is a continuation of the
                # original edge that was cut, so it keeps its `original_seg_id`.
                new_seg_id = trim_seg_id if keep else original_seg_id

                # This block handles a special case for inserting the new point.
                if loop_indexer == i and not keep:
                    size_check(loop_indexer, max_loop_capacity)
                    loop_indexer += 1
                    insert_vec4(loop, new_loop_count, loop_indexer, pt)
                    insert_byte(loop_seg_ids, new_loop_count, loop_indexer, new_seg_id)

                    new_loop_count += 1
                    # Advance i and adjust change_b to account for insertion
                    i += 1
                    change_b += 1
                    # Keep iteration bound consistent with source mutation
                    loop_count += 1
                else:
                    size_check(loop_indexer, max_loop_capacity)
                    loop_indexer += 1
                    loop[loop_indexer] = pt
                    loop_seg_ids[loop_indexer] = new_seg_id

                # Flip the keep flag after processing an intersection.
                keep = not keep

            i += 1

        new_loop_count = loop_indexer + 1
    elif prev_outside:
        # If there was no intersection, all points are on the same side.
        # If all are outside, clear the loop.
        new_loop_count = 0
    else:
        new_loop_count = loop_count

    return new_loop_count


@wp.func
def signed_distance(plane: wp.vec4, point: wp.vec3) -> float:
    """
    Calculate the signed distance from a point to a plane.

    Args:
        plane: Plane equation coefficients (a, b, c, d) where ax + by + cz + d = 0.
        point: Point to calculate distance to.

    Returns:
        Signed distance from point to plane.
    """
    return point[0] * plane[0] + point[1] * plane[1] + point[2] * plane[2] + plane[3]


@wp.func
def trim_all_in_place(
    trim_poly: wp.array(dtype=wp.vec3),
    trim_poly_count: int,
    loop: wp.array(dtype=wp.vec4),
    loop_segments: wp.array(dtype=wp.uint8),
    loop_count: int,
    max_loop_capacity: int,
) -> int:
    """
    Trim a polygon using all edges of another polygon.

    Both polygons (trim_poly and loop) are in the contact frame space and they are both convex.

    Args:
        trim_poly: Array of vertices defining the trimming polygon.
        trim_poly_count: Number of vertices in the trimming polygon.
        loop: Array of vertices in the loop to be trimmed.
        loop_segments: Array of segment IDs for the loop.
        loop_count: Number of vertices in the loop.
        max_loop_capacity: Maximum capacity of the loop arrays.

    Returns:
        New number of vertices in the trimmed loop.
    """

    if trim_poly_count <= 1:
        return loop_count  # There is no trim polygon

    if trim_poly_count == 2:
        return trim_by_line_segment_in_place(
            trim_poly[0], trim_poly[1], loop, loop_segments, loop_count, max_loop_capacity
        )

    current_loop_count = loop_count

    for i in range(trim_poly_count):
        # For each trim segment, we will call the efficient trim function.
        trim_seg_start = trim_poly[i]
        trim_seg_end = trim_poly[(i + 1) % trim_poly_count]
        # Perform the in-place trimming for this segment.
        current_loop_count = trim_in_place(
            trim_seg_start, trim_seg_end, wp.uint8(i), loop, loop_segments, current_loop_count, max_loop_capacity
        )

    return current_loop_count


@wp.func
def approx_max_quadrilateral_area_with_calipers(hull: wp.array(dtype=wp.vec4), hull_count: int) -> wp.vec4i:
    """
    Finds an approximate maximum area quadrilateral inside a convex hull in O(n) time
    using the Rotating Calipers algorithm to find the hull's diameter.

    Args:
        hull: Array of hull vertices.
        hull_count: Number of vertices in the hull.

    Returns:
        vec4i containing (p1, p2, p3, p4) where p1, p2, p3, p4 are the indices
        of the quadrilateral vertices that form the maximum area quadrilateral.
    """
    n = hull_count

    # --- Step 1: Find the hull's diameter using Rotating Calipers in O(n) ---
    p1 = int(0)
    p3 = int(1)
    # Use vec4 distance
    hp1 = hull[p1]
    hp3 = hull[p3]
    diff = wp.vec4(hp1[0] - hp3[0], hp1[1] - hp3[1], hp1[2] - hp3[2], hp1[3] - hp3[3])
    max_dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + diff[3] * diff[3]

    # Start with point j opposite point i=0
    j = int(1)
    for i in range(n):
        # For the current point i, find its antipodal point j by advancing j
        # while the area of the triangle formed by the edge (i, i+1) and point j increases.
        # This is equivalent to finding the point j furthest from the edge (i, i+1).
        hull_i_xy = wp.vec2(hull[i][0], hull[i][1])
        hull_i_plus_1_xy = wp.vec2(hull[(i + 1) % n][0], hull[(i + 1) % n][1])

        while True:
            hull_j_xy = wp.vec2(hull[j][0], hull[j][1])
            hull_j_plus_1_xy = wp.vec2(hull[(j + 1) % n][0], hull[(j + 1) % n][1])

            area_j_plus_1 = signed_area(hull_i_xy, hull_i_plus_1_xy, hull_j_plus_1_xy)
            area_j = signed_area(hull_i_xy, hull_i_plus_1_xy, hull_j_xy)

            if area_j_plus_1 > area_j:
                j = (j + 1) % n
            else:
                break

        # Now, (i, j) is an antipodal pair. Check its distance (vec4)
        hi = hull[i]
        hj = hull[j]
        d1 = wp.vec4(hi[0] - hj[0], hi[1] - hj[1], hi[2] - hj[2], hi[3] - hj[3])
        dist_sq_1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2] + d1[3] * d1[3]
        if dist_sq_1 > max_dist_sq:
            max_dist_sq = dist_sq_1
            p1 = i
            p3 = j

        # The next point, (i+1, j), is also an antipodal pair. Check its distance too (vec4)
        hip1 = hull[(i + 1) % n]
        d2 = wp.vec4(hip1[0] - hj[0], hip1[1] - hj[1], hip1[2] - hj[2], hip1[3] - hj[3])
        dist_sq_2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2] + d2[3] * d2[3]
        if dist_sq_2 > max_dist_sq:
            max_dist_sq = dist_sq_2
            p1 = (i + 1) % n
            p3 = j

    # --- Step 2: Find points p2 and p4 furthest from the diameter (p1, p3) ---
    p2 = int(0)
    p4 = int(0)
    max_area_1 = float(0.0)
    max_area_2 = float(0.0)

    hull_p1_xy = wp.vec2(hull[p1][0], hull[p1][1])
    hull_p3_xy = wp.vec2(hull[p3][0], hull[p3][1])

    for i in range(n):
        # Use the signed area to determine which side of the line the point is on.
        hull_i_xy = wp.vec2(hull[i][0], hull[i][1])
        area = signed_area(hull_p1_xy, hull_p3_xy, hull_i_xy)

        if area > max_area_1:
            max_area_1 = area
            p2 = i
        elif -area > max_area_2:  # Check the other side
            max_area_2 = -area
            p4 = i

    return wp.vec4i(p1, p2, p3, p4)


@wp.func
def remove_zero_length_edges(
    loop: wp.array(dtype=wp.vec4), loop_seg_ids: wp.array(dtype=wp.uint8), loop_count: int, eps: float
) -> int:
    """
    Remove zero-length edges from a polygon loop.

    Args:
        loop: Array of loop vertices.
        loop_seg_ids: Array of segment IDs for the loop.
        loop_count: Number of vertices in the loop.
        eps: Epsilon threshold for considering edges as zero-length.

    Returns:
        New number of vertices in the cleaned loop.
    """
    # A loop must have at least 2 points to be valid per your requirement.
    if loop_count < 2:
        return 0

    # 'write_idx' is the index for the new, compacted loop.
    # It always points to the last valid point found so far.
    write_idx = int(0)

    # Iterate through the original loop, starting from the second point.
    # 'read_idx' is the index of the point we are currently considering.
    for read_idx in range(1, loop_count):
        # Check if the current point is distinct from the last point we kept.
        loop_read_xy = wp.vec2(loop[read_idx][0], loop[read_idx][1])
        loop_write_xy = wp.vec2(loop[write_idx][0], loop[write_idx][1])
        diff = loop_read_xy - loop_write_xy

        if wp.length_sq(diff) > eps:
            # It's a distinct point, so we advance the write index and keep it.
            write_idx += 1
            loop[write_idx] = loop[read_idx]
            loop_seg_ids[write_idx - 1] = loop_seg_ids[read_idx - 1]

    loop_seg_ids[write_idx] = loop_seg_ids[loop_count - 1]

    # At this point, the loop is clean but might not be closed properly.
    # The number of points in our cleaned chain is 'write_idx + 1'.

    # Handle the loop closure by checking if the last point is the same as the first.
    if write_idx > 0:
        loop_write_xy = wp.vec2(loop[write_idx][0], loop[write_idx][1])
        loop_0_xy = wp.vec2(loop[0][0], loop[0][1])
        diff = loop_write_xy - loop_0_xy

        if wp.length_sq(diff) < eps:
            # The last point is a duplicate of the first; we need to remove it.
            new_loop_count = write_idx
        else:
            # The last point is not a duplicate, so we keep all 'write_idx + 1' points.
            new_loop_count = write_idx + 1
    else:
        new_loop_count = write_idx + 1

    # Final check based on your requirement.
    # If simplification resulted in fewer than 2 points, it's a degenerate point.
    if new_loop_count < 2:
        new_loop_count = 0

    return new_loop_count


@wp.func
def edge_key(per_vertex_features: wp.types.vector(6, wp.uint8), count: int, edge_id: int) -> wp.uint32:
    """
    Creates a packed edge key from two consecutive feature IDs.
    Used to create compact identifiers for edges defined by vertex pairs.

    Args:
        per_vertex_features: Array of feature IDs.
        count: Number of features in the array.
        edge_id: Index of the first vertex of the edge.

    Returns:
        16-bit packed edge key: (first_feature << 8) | second_feature.
    """
    # An edge always goes from one vertex to the next, wrapping around at the end.
    first = per_vertex_features[edge_id]
    second = per_vertex_features[(edge_id + 1) % count]
    return wp.uint32(wp.uint32(first) << wp.uint32(8)) | wp.uint32(second)


@wp.func
def feature_id(
    loop_seg_ids: wp.array(dtype=wp.uint8),
    loop_id: int,
    loop_count: int,
    features_a: wp.types.vector(6, wp.uint8),
    features_b: wp.types.vector(6, wp.uint8),
    m_a_count: int,
    m_b_count: int,
) -> wp.uint32:
    """
    Determines the feature identifier for a vertex in the clipped contact polygon.
    This function assigns feature IDs that encode which geometric features from the original
    collision shapes (vertices, edges, or edge-edge intersections) each contact point represents.

    ENCODING SCHEME:
    - Original trim poly vertex: 8-bit feature ID from features_a
    - Original loop poly vertex: 16-bit (features_b[vertex] << 8)
    - Edge intersections: 32-bit ((edge1_key << 16) | edge2_key)
    - Shape intersections: 32-bit ((shapeA_edge << 16) | shapeB_edge)

    SEGMENT ID CONVENTION:
    - IDs 0-5: segments from trim polygon (shape A)
    - IDs 6+: segments from loop polygon (shape B, with offset)

    Args:
        loop_seg_ids: Array of segment IDs for the current clipped polygon.
        loop_id: Index of the vertex to compute feature ID for.
        loop_count: Total number of vertices in the polygon.
        features_a: Original feature IDs from trim polygon (shape A).
        features_b: Original feature IDs from loop polygon (shape B).
        m_a_count: Number of vertices in original trim polygon.
        m_b_count: Number of vertices in original loop polygon.

    Returns:
        A feature ID encoding the geometric origin of this contact point.
    """
    feature = wp.uint32(0)

    incoming = loop_seg_ids[(loop_id - 1 + loop_count) % loop_count]
    outgoing = loop_seg_ids[loop_id]
    incoming_belongs_to_trim_poly = incoming < 6
    outgoing_belongs_to_trim_poly = outgoing < 6

    if incoming_belongs_to_trim_poly != outgoing_belongs_to_trim_poly:
        # This must be an intersection point
        if incoming_belongs_to_trim_poly:
            x = edge_key(features_a, m_a_count, int(incoming))
        else:
            x = edge_key(features_b, m_b_count, int(incoming) - 6)

        if outgoing_belongs_to_trim_poly:
            y = edge_key(features_a, m_a_count, int(outgoing))
        else:
            y = edge_key(features_b, m_b_count, int(outgoing) - 6)

        feature = (x << wp.uint32(16)) | y
    else:
        if incoming_belongs_to_trim_poly:
            next_seg = (int(incoming) + 1) % m_a_count
            is_original_poly_point = next_seg == int(outgoing)
            if is_original_poly_point:
                feature = wp.uint32(features_a[int(outgoing)])
            else:
                # Should not happen because input poly A would have self intersections
                x = edge_key(features_a, m_a_count, int(incoming))
                y = edge_key(features_a, m_a_count, int(outgoing))
                feature = (x << wp.uint32(16)) | y
        else:
            next_seg = (int(incoming) - 6 + 1) % m_b_count + 6
            is_original_poly_point = next_seg == int(outgoing)
            if is_original_poly_point:
                # Shifted such that not the same id can get generated as produced by features_a
                feature = wp.uint32(features_b[int(outgoing) - 6]) << wp.uint32(8)
            else:
                # Should not happen because input poly B would have self intersections
                x = edge_key(features_b, m_b_count, int(incoming) - 6)
                y = edge_key(features_b, m_b_count, int(outgoing) - 6)
                feature = (x << wp.uint32(16)) | y

    return feature


@wp.func
def add_avoid_duplicates_vec3(arr: wp.array(dtype=wp.vec3), arr_count: int, vec: wp.vec3, eps: float) -> int:
    """
    Add a vector to an array, avoiding duplicates.

    Args:
        arr: Array to add to.
        arr_count: Current number of elements in the array.
        vec: Vector to add.
        eps: Epsilon threshold for duplicate detection.

    Returns:
        New array count.
    """
    # Check for duplicates. If the new vertex 'vec' is too close to the first or last existing vertex, ignore it.
    # This is a simple reduction step to avoid redundant points.
    if arr_count > 0:
        if wp.length_sq(arr[0] - vec) < eps:
            return arr_count

    if arr_count > 1:
        if wp.length_sq(arr[arr_count - 1] - vec) < eps:
            return arr_count

    arr[arr_count] = vec
    return arr_count + 1


@wp.func
def add_avoid_duplicates_vec4(arr: wp.array(dtype=wp.vec4), arr_count: int, vec: wp.vec3, eps: float) -> int:
    """
    Add a vec3 to a vec4 array (storing in XYZ components), avoiding duplicates.

    Args:
        arr: Array to add to.
        arr_count: Current number of elements in the array.
        vec: Vector to add.
        eps: Epsilon threshold for duplicate detection.

    Returns:
        New array count.
    """
    # Check for duplicates. If the new vertex 'vec' is too close to the first or last existing vertex, ignore it.
    if arr_count > 0:
        arr_0_xyz = wp.vec3(arr[0][0], arr[0][1], arr[0][2])
        if wp.length_sq(arr_0_xyz - vec) < eps:
            return arr_count

    if arr_count > 1:
        arr_last_xyz = wp.vec3(arr[arr_count - 1][0], arr[arr_count - 1][1], arr[arr_count - 1][2])
        if wp.length_sq(arr_last_xyz - vec) < eps:
            return arr_count

    arr[arr_count] = wp.vec4(vec[0], vec[1], vec[2], arr[arr_count][3])  # w component does not matter
    return arr_count + 1


# Support mapping interface - now imported from support_function module

get_array_ptr_cpp = """
    return (uint64_t)arr.data;
"""


@wp.func_native(get_array_ptr_cpp)
def get_Fvec3_array_ptr(arr: wp.array(dtype=Fvec3)) -> wp.uint64: ...


@wp.func
def extract_4_point_contact_manifolds(
    m_a: wp.array(dtype=wp.vec3),
    features_a: wp.types.vector(6, wp.uint8),
    m_a_count: int,
    m_b: wp.array(dtype=Fvec3),
    features_b: wp.types.vector(6, wp.uint8),
    m_b_count: int,
    normal: wp.vec3,
    cross_vector_1: wp.vec3,
    cross_vector_2: wp.vec3,
    anchor_point_a: wp.vec3,
    anchor_point_b: wp.vec3,
    feature_anchor_a: wp.int32,
    feature_anchor_b: wp.int32,
) -> int:
    """
    Extract 4-point contact manifolds from two convex polygons.

    m_A and m_B can have up to 6 points but m_B must provide space for 12 points.
    """
    # The trim poly (poly A) should be the polygon with the most points
    # This should ensure that zero area loops with only two points get trimmed correctly (they are considered valid)
    center = 0.5 * (anchor_point_a + anchor_point_b)

    # Transform into contact plane space
    for i in range(m_a_count):
        projected = m_a[i] - center
        m_a[i] = wp.vec3(
            wp.dot(cross_vector_1, projected),
            wp.dot(cross_vector_2, projected),
            wp.dot(normal, projected),
        )

    depth_of_plane_a = wp.dot(normal, anchor_point_a - center)

    max_points = 12
    loop = wp.array(ptr=get_Fvec3_array_ptr(m_b), shape=(12,), dtype=wp.vec4)  # stackalloc vec4[maxPoints];
    f_loop = m_b  # (Fvec3*)loop;
    loop_seg_ids = wp.zeros(shape=(12,), dtype=wp.uint8)  # stackalloc byte[maxPoints];

    for i in range(m_b_count):
        bb_xyz = fvec3_get_xyz(m_b[i])
        projected = bb_xyz - center
        loop[i] = wp.vec4(
            wp.dot(cross_vector_1, projected),
            wp.dot(cross_vector_2, projected),
            wp.dot(normal, projected),
            depth_of_plane_a,
        )
        loop_seg_ids[i] = wp.uint8(i + 6)

    loop_count = trim_all_in_place(m_a, m_a_count, loop, loop_seg_ids, m_b_count, max_points)

    loop_count = remove_zero_length_edges(loop, loop_seg_ids, loop_count, EPS)

    if loop_count > 4:
        result = approx_max_quadrilateral_area_with_calipers(loop, loop_count)
        ia = int(result[0])
        ib = int(result[1])
        ic = int(result[2])
        id = int(result[3])

        a = loop[ia]
        feature_a = feature_id(loop_seg_ids, ia, loop_count, features_a, features_b, m_a_count, m_b_count)
        f_loop[ia].feature = feature_a
        b = loop[ib]
        feature_b = feature_id(loop_seg_ids, ib, loop_count, features_a, features_b, m_a_count, m_b_count)
        f_loop[ib].feature = feature_b
        c = loop[ic]
        feature_c = feature_id(loop_seg_ids, ic, loop_count, features_a, features_b, m_a_count, m_b_count)
        f_loop[ic].feature = feature_c
        d = loop[id]
        feature_d = feature_id(loop_seg_ids, id, loop_count, features_a, features_b, m_a_count, m_b_count)
        f_loop[id].feature = feature_d

        # Transform back to world space
        m_a[0] = a[0] * cross_vector_1 + a[1] * cross_vector_2 + a[3] * normal + center
        m_a[1] = b[0] * cross_vector_1 + b[1] * cross_vector_2 + b[3] * normal + center
        m_a[2] = c[0] * cross_vector_1 + c[1] * cross_vector_2 + c[3] * normal + center
        m_a[3] = d[0] * cross_vector_1 + d[1] * cross_vector_2 + d[3] * normal + center

        m_b[0] = fvec3_set_xyz(m_b[0], a[0] * cross_vector_1 + a[1] * cross_vector_2 + a[2] * normal + center)
        m_b[1] = fvec3_set_xyz(m_b[1], b[0] * cross_vector_1 + b[1] * cross_vector_2 + b[2] * normal + center)
        m_b[2] = fvec3_set_xyz(m_b[2], c[0] * cross_vector_1 + c[1] * cross_vector_2 + c[2] * normal + center)
        m_b[3] = fvec3_set_xyz(m_b[3], d[0] * cross_vector_1 + d[1] * cross_vector_2 + d[2] * normal + center)

        # Ensure features are propagated to the first four outputs to match positions 0..3
        f_loop[0].feature = feature_a
        f_loop[1].feature = feature_b
        f_loop[2].feature = feature_c
        f_loop[3].feature = feature_d

        loop_count = 4
    else:
        # Transform back to world space
        for i in range(loop_count):
            l = loop[i]
            feature = feature_id(loop_seg_ids, i, loop_count, features_a, features_b, m_a_count, m_b_count)
            m_a[i] = l[0] * cross_vector_1 + l[1] * cross_vector_2 + l[3] * normal + center
            m_b[i] = fvec3_set_xyz(m_b[i], l[0] * cross_vector_1 + l[1] * cross_vector_2 + l[2] * normal + center)
            f_loop[i].feature = feature
        if loop_count == 0:
            feature = wp.uint32(0)  # (feature_anchor_a << 16) | feature_anchor_b;
            m_a[loop_count] = anchor_point_a
            m_b[loop_count] = fvec3_set_xyz(m_b[loop_count], anchor_point_b)
            loop_count += 1
            f_loop[loop_count - 1].feature = feature

    return loop_count


vec6_uint8 = wp.types.vector(6, wp.uint8)


def create_build_manifold(support_func: Any):
    # Main contact manifold generation function
    @wp.func
    def build_manifold_core(
        geom_a: Any,
        geom_b: Any,
        quaternion_a: wp.quat,
        quaternion_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        p_a: wp.vec3,
        p_b: wp.vec3,
        normal: wp.vec3,
        a_buffer: wp.array(dtype=wp.vec3),
        b_buffer: wp.array(dtype=Fvec3),
        feature_anchor_a: wp.int32,
        feature_anchor_b: wp.int32,
        data_provider: Any,
    ) -> int:
        """
        The result will be stored in a_buffer and b_buffer. They also serve as scratch memory during the calculation, therefore the exotic typing.
        a_buffer must have space for 6 elements, b_buffer space for 12 elements
        The return value of the methods tells the user how many elements in the buffers are valid. Both buffers have the same number of entries.
        The two shapes must always be queried in the same order to get stable feature ids.
        """

        normal = -normal  # The code below uses a different normal convention

        # Reset all counters for a new calculation.
        a_count = 0
        b_count = 0

        # Create an orthonormal basis from the collision normal.
        tangent_a, tangent_b = build_orthonormal_basis(normal)

        features_a = vec6_uint8(wp.uint8(0))
        features_b = vec6_uint8(wp.uint8(0))

        bb_buffer = wp.array(ptr=get_Fvec3_array_ptr(b_buffer), shape=(12,), dtype=wp.vec4)

        # --- Step 1: Find Contact Polygons using Perturbed Support Mapping ---
        # Loop 6 times to find up to 6 vertices for each shape's contact polygon.
        for e in range(6):
            # Create a perturbed normal direction. This is the main collision normal slightly
            # altered by a vector on the contact plane, defined by the hexagonal vertices.
            angle = float(e) * ROT_DELTA_ANGLE
            s = wp.sin(angle)
            c = wp.cos(angle)
            offset_normal = normal * COS_OFFSET + (c * SIN_OFFSET) * tangent_a + (s * SIN_OFFSET) * tangent_b

            # Find the support point on shape A in the perturbed direction.
            # 1. Transform the world-space direction into shape A's local space.
            tmp = wp.quat_rotate_inv(quaternion_a, offset_normal)
            # 2. Find the furthest point on shape A in that local direction.
            (pt_a, feature_a) = support_func(geom_a, tmp, data_provider)
            features_a[e] = wp.uint8(int(feature_a) + 1)
            # 3. Transform the local-space support point back to world space.
            pt_a = wp.quat_rotate(quaternion_a, pt_a) + position_a
            # 4. Add the world-space point to the 'left' polygon, checking for duplicates.
            a_count = add_avoid_duplicates_vec3(a_buffer, a_count, pt_a, EPS)

            # Invert the direction for the other shape.
            offset_normal = -offset_normal

            # Find the support point on shape B in the opposite perturbed direction.
            # (Process is identical to the one for shape A).
            tmp = wp.quat_rotate_inv(quaternion_b, offset_normal)
            (pt_b, feature_b) = support_func(geom_b, tmp, data_provider)
            features_b[e] = wp.uint8(int(feature_b) + 1)
            pt_b = wp.quat_rotate(quaternion_b, pt_b) + position_b
            b_count = add_avoid_duplicates_vec4(bb_buffer, b_count, pt_b, EPS)

        # All feature ids are one based such that it is clearly visible in a uint which of the 4 slots (8 bits each) are in use
        return extract_4_point_contact_manifolds(
            a_buffer,
            features_a,
            a_count,
            b_buffer,
            features_b,
            b_count,
            normal,
            tangent_a,
            tangent_b,
            p_a,
            p_b,
            feature_anchor_a + 1,
            feature_anchor_b + 1,
        )

    @wp.func
    def build_manifold(
        geom_a: Any,
        geom_b: Any,
        quaternion_a: wp.quat,
        quaternion_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        p_a: wp.vec3,
        p_b: wp.vec3,
        normal: wp.vec3,
        feature_anchor_a: wp.int32,
        feature_anchor_b: wp.int32,
        data_provider: Any,
    ) -> tuple[
        int,
        wp.vec4,
        wp.types.matrix((4, 3), wp.float32),
        wp.types.matrix((4, 3), wp.float32),
        wp.vec4i,
    ]:
        """
        Build a contact manifold between two convex shapes using perturbed support mapping and polygon clipping.

        This function generates up to 4 contact points between two colliding convex shapes by:
        1. Finding contact polygons using perturbed support mapping in 6 directions
        2. Clipping the polygons against each other in contact plane space
        3. Selecting the best 4 points using rotating calipers algorithm if more than 4 exist
        4. Transforming results back to world space with feature tracking

        The contact normal is the same for all contact points in the manifold. The two shapes
        must always be queried in the same order to get stable feature IDs for contact tracking.

        Args:
            geom_a: Geometry data for the first shape.
            geom_b: Geometry data for the second shape.
            quaternion_a: Orientation quaternion of the first shape.
            quaternion_b: Orientation quaternion of the second shape.
            position_a: World position of the first shape.
            position_b: World position of the second shape.
            p_a: Anchor contact point on the first shape (from GJK/EPA).
            p_b: Anchor contact point on the second shape (from GJK/EPA).
            normal: Contact normal vector pointing from shape A to shape B.
            feature_anchor_a: Feature ID of the anchor point on shape A. Can pass in 0 if anchor tracking is not needed.
            feature_anchor_b: Feature ID of the anchor point on shape B. Can pass in 0 if anchor tracking is not needed.
            data_provider: Support mapping data provider for shape queries.

        Returns:
            A tuple containing:
            - int: Number of valid contact points in the manifold (0-4).
            - wp.vec4: Penetration depths for each contact point (negative when shapes overlap).
            - wp.types.matrix((4, 3), wp.float32): Contact points on shape A in world space.
            - wp.types.matrix((4, 3), wp.float32): Contact points on shape B in world space.
            - wp.vec4i: Feature IDs for each contact point, enabling contact tracking across
              multiple frames for warm starting and contact persistence.

        Note:
            The feature IDs encode geometric information about which features (vertices, edges,
            or edge-edge intersections) each contact point represents, allowing the physics
            solver to maintain contact consistency over time.
        """
        left = wp.zeros(shape=(6,), dtype=wp.vec3)  # Array for shape A contact points
        right = wp.zeros(
            shape=(12,), dtype=Fvec3
        )  # Array for shape B contact points - also provides storage for intermediate results

        num_manifold_points = build_manifold_core(
            geom_a,
            geom_b,
            quaternion_a,
            quaternion_b,
            position_a,
            position_b,
            p_a,
            p_b,
            normal,
            left,
            right,
            feature_anchor_a,
            feature_anchor_b,
            data_provider,
        )

        # Extract results into fixed-size matrices
        contact_points_a = wp.types.matrix(shape=(4, 3), dtype=wp.float32)
        contact_points_b = wp.types.matrix(shape=(4, 3), dtype=wp.float32)
        feature_ids = wp.vec4i(0, 0, 0, 0)
        penetrations = wp.vec4(0.0, 0.0, 0.0, 0.0)

        # Copy contact points and extract feature IDs
        count_out = min(num_manifold_points, 4)
        for i in range(count_out):
            contact_points_a[i] = left[i]
            contact_points_b[i] = fvec3_get_xyz(right[i])
            feature_ids[i] = int(right[i].feature)
            # Newton convention: penetration is negative on overlap
            penetrations[i] = wp.dot(contact_points_a[i] - contact_points_b[i], normal)

        return num_manifold_points, penetrations, contact_points_a, contact_points_b, feature_ids

    return build_manifold
