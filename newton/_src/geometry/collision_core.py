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

from __future__ import annotations

import warp as wp

from .collision_convex import create_solve_convex_multi_contact, create_solve_convex_single_contact
from .support_function import GenericShapeData, GeoTypeEx, SupportMapDataProvider, support_map
from .types import GeoType

# Configuration flag for multi-contact generation
ENABLE_MULTI_CONTACT = True

# Pre-create the convex contact solvers (usable inside kernels)
solve_convex_multi_contact = create_solve_convex_multi_contact(support_map)
solve_convex_single_contact = create_solve_convex_single_contact(support_map)

# Type definitions for multi-contact manifolds
_mat53f = wp.types.matrix((5, 3), wp.float32)
_vec5 = wp.types.vector(5, wp.float32)


@wp.func
def is_discrete_shape(shape_type: int) -> bool:
    """A discrete shape can be represented with a finite amount of flat polygon faces."""
    return (
        shape_type == int(GeoType.BOX)
        or shape_type == int(GeoType.CONVEX_MESH)
        or shape_type == int(GeoTypeEx.TRIANGLE)
        or shape_type == int(GeoType.PLANE)
    )


@wp.func
def project_point_onto_plane(point: wp.vec3, plane_point: wp.vec3, plane_normal: wp.vec3) -> wp.vec3:
    """
    Project a point onto a plane defined by a point and normal.

    Args:
        point: The point to project
        plane_point: A point on the plane
        plane_normal: Normal vector of the plane (should be normalized)

    Returns:
        The projected point on the plane
    """
    to_point = point - plane_point
    distance_to_plane = wp.dot(to_point, plane_normal)
    projected_point = point - plane_normal * distance_to_plane
    return projected_point


@wp.func
def compute_plane_normal_from_contacts(
    points: _mat53f,
    normal: wp.vec3,
    signed_distances: _vec5,
    count: int,
) -> wp.vec3:
    """
    Compute plane normal from reconstructed plane points.

    Reconstructs the plane points from contact data and computes the plane normal
    using fan triangulation to find the largest area triangle for numerical stability.

    Args:
        points: Contact points matrix (5x3)
        normal: Initial contact normal (used for reconstruction)
        signed_distances: Signed distances vector (5 elements)
        count: Number of contact points

    Returns:
        Normalized plane normal from the contact points
    """
    if count < 3:
        # Not enough points to form a triangle, return original normal
        return normal

    # Reconstruct plane points from contact data
    # Use first point as anchor for fan triangulation
    # Contact points are at midpoint, move to discrete surface (plane)
    p0 = points[0] + normal * (signed_distances[0] * 0.5)

    # Find the triangle with the largest area for numerical stability
    # This avoids issues with nearly collinear points
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    max_area_sq = float(0.0)

    for i in range(1, count - 1):
        # Reconstruct plane points for this triangle
        pi = points[i] + normal * (signed_distances[i] * 0.5)
        pi_next = points[i + 1] + normal * (signed_distances[i + 1] * 0.5)

        # Compute cross product for triangle (p0, pi, pi_next)
        edge1 = pi - p0
        edge2 = pi_next - p0
        cross = wp.cross(edge1, edge2)
        area_sq = wp.dot(cross, cross)

        if area_sq > max_area_sq:
            max_area_sq = area_sq
            best_normal = cross

    # Normalize, avoid zero
    len_n = wp.sqrt(wp.max(1.0e-12, max_area_sq))
    plane_normal = best_normal / len_n

    # Ensure normal points in same direction as original normal
    if wp.dot(plane_normal, normal) < 0.0:
        plane_normal = -plane_normal

    return plane_normal


@wp.func
def postprocess_axial_shape_discrete_contacts(
    points: _mat53f,
    normal: wp.vec3,
    signed_distances: _vec5,
    count: int,
    shape_rot: wp.quat,
    shape_radius: float,
    shape_half_height: float,
    shape_pos: wp.vec3,
    is_cone: bool,
) -> tuple[int, _vec5, _mat53f]:
    """
    Post-process contact points for axial shape (cylinder/cone) vs discrete surface collisions.

    When an axial shape is rolling on a discrete surface (plane, box, convex hull),
    we project contact points onto a plane perpendicular to both the shape axis and
    contact normal to stabilize rolling contacts.

    Works for:
    - Cylinders: Axis perpendicular to surface normal when rolling
    - Cones: Axis at an angle = cone half-angle when rolling on base

    Args:
        points: Contact points matrix (5x3)
        normal: Contact normal (from discrete to shape)
        signed_distances: Signed distances vector (5 elements)
        count: Number of input contact points
        shape_rot: Shape orientation
        shape_radius: Shape radius (constant for cylinder, base radius for cone)
        shape_half_height: Shape half height
        shape_pos: Shape position
        is_cone: True if shape is a cone, False if cylinder

    Returns:
        Tuple of (new_count, new_signed_distances, new_points)
    """
    # Get shape axis in world space (Z-axis for both cylinders and cones)
    shape_axis = wp.quat_rotate(shape_rot, wp.vec3(0.0, 0.0, 1.0))

    # Check if shape is in rolling configuration
    axis_normal_dot = wp.abs(wp.dot(shape_axis, normal))

    # Compute threshold based on shape type
    if is_cone:
        # For a cone rolling on its base, the axis makes an angle with the normal
        # equal to the cone's half-angle: angle = atan(radius / (2 * half_height))
        # When rolling: dot(axis, normal) = cos(90 - angle) = sin(angle)
        # Add tolerance of +/-2 degrees
        cone_half_angle = wp.atan2(shape_radius, 2.0 * shape_half_height)
        tolerance_angle = wp.static(2.0 * wp.pi / 180.0)  # 2 degrees
        lower_threshold = wp.sin(cone_half_angle - tolerance_angle)
        upper_threshold = wp.sin(cone_half_angle + tolerance_angle)

        # Check if axis_normal_dot is in the expected range for rolling
        if axis_normal_dot < lower_threshold or axis_normal_dot > upper_threshold:
            # Not in rolling configuration
            return count, signed_distances, points
    else:
        # For cylinder: axis should be perpendicular to normal (dot product ≈ 0)
        perpendicular_threshold = wp.static(wp.sin(2.0 * wp.pi / 180.0))
        if axis_normal_dot > perpendicular_threshold:
            # Not rolling, return original contacts
            return count, signed_distances, points

    # Estimate plane from contact points using the contact normal
    # Use first contact point as plane reference
    if count == 0:
        return 0, signed_distances, points

    # Compute plane normal from the largest area triangle formed by contact points
    # shape_plane_normal = compute_plane_normal_from_contacts(points, normal, signed_distances, count)
    # projection_plane_normal = wp.normalize(wp.cross(shape_axis, shape_plane_normal))

    projection_plane_normal = wp.normalize(wp.cross(shape_axis, normal))
    point_on_projection_plane = shape_pos

    # Project points onto the projection plane and remove duplicates in one pass
    # This avoids creating intermediate arrays and saves registers
    tolerance = shape_radius * 0.01  # 1% of radius for duplicate detection
    output_count = int(0)
    first_point = wp.vec3(0.0, 0.0, 0.0)

    for i in range(count):
        # Project contact point onto projection plane
        projected_point = project_point_onto_plane(points[i], point_on_projection_plane, projection_plane_normal)
        is_duplicate = False

        if output_count > 0:
            # Check against previous output point
            if wp.length(projected_point - points[output_count - 1]) < tolerance:
                is_duplicate = True

        if not is_duplicate and i > 0 and i == count - 1 and output_count > 0:
            # Last point: check against first point (cyclic)
            if wp.length(projected_point - first_point) < tolerance:
                is_duplicate = True

        if not is_duplicate:
            points[output_count] = projected_point
            signed_distances[output_count] = signed_distances[i]
            if output_count == 0:
                first_point = projected_point
            output_count += 1

    return output_count, signed_distances, points


@wp.func
def compute_gjk_mpr_contacts(
    geom_a: GenericShapeData,
    geom_b: GenericShapeData,
    type_a: int,
    type_b: int,
    rot_a: wp.quat,
    rot_b: wp.quat,
    pos_a_adjusted: wp.vec3,
    pos_b_adjusted: wp.vec3,
    rigid_contact_margin: float,
):
    """
    Compute contacts between two shapes using GJK/MPR algorithm.

    Returns:
        Tuple of (count, normal, signed_distances, points, radius_eff_a, radius_eff_b)
    """
    data_provider = SupportMapDataProvider()

    radius_eff_a = float(0.0)
    radius_eff_b = float(0.0)

    small_radius = 0.0001

    # Special treatment for minkowski objects
    if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
        radius_eff_a = geom_a.scale[0]
        geom_a.scale[0] = small_radius

    if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
        radius_eff_b = geom_b.scale[0]
        geom_b.scale[0] = small_radius

    if wp.static(ENABLE_MULTI_CONTACT):
        count, normal, signed_distances, points, _features = wp.static(solve_convex_multi_contact)(
            geom_a,
            geom_b,
            rot_a,
            rot_b,
            pos_a_adjusted,
            pos_b_adjusted,
            0.0,  # sum_of_contact_offsets - gap
            data_provider,
            rigid_contact_margin + radius_eff_a + radius_eff_b,
            type_a == int(GeoType.SPHERE) or type_b == int(GeoType.SPHERE),
        )
    else:
        count, normal, signed_distances, points, _features = wp.static(solve_convex_single_contact)(
            geom_a,
            geom_b,
            rot_a,
            rot_b,
            pos_a_adjusted,
            pos_b_adjusted,
            0.0,  # sum_of_contact_offsets - gap
            data_provider,
            rigid_contact_margin + radius_eff_a + radius_eff_b,
        )

    # Special post processing for minkowski objects
    if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
        for i in range(count):
            points[i] = points[i] + normal * (radius_eff_a * 0.5)
            signed_distances[i] -= radius_eff_a - small_radius
    if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
        for i in range(count):
            points[i] = points[i] - normal * (radius_eff_b * 0.5)
            signed_distances[i] -= radius_eff_b - small_radius

    if wp.static(ENABLE_MULTI_CONTACT):
        # Post-process for axial shapes (cylinder/cone) rolling on discrete surfaces
        is_discrete_a = is_discrete_shape(geom_a.shape_type)
        is_discrete_b = is_discrete_shape(geom_b.shape_type)
        is_axial_a = type_a == int(GeoType.CYLINDER) or type_a == int(GeoType.CONE)
        is_axial_b = type_b == int(GeoType.CYLINDER) or type_b == int(GeoType.CONE)

        if is_discrete_a and is_axial_b and count >= 3:
            # Post-process axial shape (B) rolling on discrete surface (A)
            shape_radius = geom_b.scale[0]  # radius for cylinder, base radius for cone
            shape_half_height = geom_b.scale[1]
            is_cone_b = type_b == int(GeoType.CONE)
            count, signed_distances, points = postprocess_axial_shape_discrete_contacts(
                points,
                normal,
                signed_distances,
                count,
                rot_b,
                shape_radius,
                shape_half_height,
                pos_b_adjusted,
                is_cone_b,
            )

        if is_discrete_b and is_axial_a and count >= 3:
            # Post-process axial shape (A) rolling on discrete surface (B)
            # Note: normal points from A to B, so we need to negate it for the shape processing
            shape_radius = geom_a.scale[0]  # radius for cylinder, base radius for cone
            shape_half_height = geom_a.scale[1]
            is_cone_a = type_a == int(GeoType.CONE)
            count, signed_distances, points = postprocess_axial_shape_discrete_contacts(
                points,
                -normal,
                signed_distances,
                count,
                rot_a,
                shape_radius,
                shape_half_height,
                pos_a_adjusted,
                is_cone_a,
            )

    return count, normal, signed_distances, points, radius_eff_a, radius_eff_b


@wp.func
def compute_tight_aabb_from_support(
    shape_data: GenericShapeData,
    orientation: wp.quat,
    center_pos: wp.vec3,
    data_provider: SupportMapDataProvider,
) -> tuple[wp.vec3, wp.vec3]:
    """
    Compute tight AABB for a shape using support function.

    Args:
        shape_data: Generic shape data
        orientation: Shape orientation (quaternion)
        center_pos: Center position of the shape
        data_provider: Support map data provider

    Returns:
        Tuple of (aabb_min, aabb_max) in world space
    """
    # Transpose orientation matrix to transform world axes to local space
    # Convert quaternion to 3x3 rotation matrix and transpose (inverse rotation)
    rot_mat = wp.quat_to_matrix(orientation)
    rot_mat_t = wp.transpose(rot_mat)

    # Transform world axes to local space (multiply by transposed rotation = inverse rotation)
    local_x = wp.vec3(rot_mat_t[0, 0], rot_mat_t[1, 0], rot_mat_t[2, 0])
    local_y = wp.vec3(rot_mat_t[0, 1], rot_mat_t[1, 1], rot_mat_t[2, 1])
    local_z = wp.vec3(rot_mat_t[0, 2], rot_mat_t[1, 2], rot_mat_t[2, 2])

    # Compute AABB extents by evaluating support function in local space
    # Dot products are done in local space to avoid expensive rotations
    support_point = wp.vec3()

    # Max X: support along +local_x, dot in local space
    support_point, _feature_id = support_map(shape_data, local_x, data_provider)
    max_x = wp.dot(local_x, support_point)

    # Max Y: support along +local_y, dot in local space
    support_point, _feature_id = support_map(shape_data, local_y, data_provider)
    max_y = wp.dot(local_y, support_point)

    # Max Z: support along +local_z, dot in local space
    support_point, _feature_id = support_map(shape_data, local_z, data_provider)
    max_z = wp.dot(local_z, support_point)

    # Min X: support along -local_x, dot in local space
    support_point, _feature_id = support_map(shape_data, -local_x, data_provider)
    min_x = wp.dot(local_x, support_point)

    # Min Y: support along -local_y, dot in local space
    support_point, _feature_id = support_map(shape_data, -local_y, data_provider)
    min_y = wp.dot(local_y, support_point)

    # Min Z: support along -local_z, dot in local space
    support_point, _feature_id = support_map(shape_data, -local_z, data_provider)
    min_z = wp.dot(local_z, support_point)

    # AABB in world space (add world position to extents)
    aabb_min = wp.vec3(min_x, min_y, min_z) + center_pos
    aabb_max = wp.vec3(max_x, max_y, max_z) + center_pos

    return aabb_min, aabb_max


@wp.func
def compute_bounding_sphere_from_aabb(aabb_lower: wp.vec3, aabb_upper: wp.vec3) -> tuple[wp.vec3, float]:
    """
    Compute a bounding sphere from an AABB.

    Returns:
        Tuple of (center, radius) where center is the AABB center and radius is half the diagonal.
    """
    center = 0.5 * (aabb_lower + aabb_upper)
    half_extents = 0.5 * (aabb_upper - aabb_lower)
    radius = wp.length(half_extents)
    return center, radius


@wp.func
def convert_infinite_plane_to_cube(
    shape_data: GenericShapeData,
    plane_rotation: wp.quat,
    plane_position: wp.vec3,
    other_position: wp.vec3,
    other_radius: float,
) -> tuple[GenericShapeData, wp.vec3]:
    """
    Convert an infinite plane into a cube proxy for GJK/MPR collision detection.

    Since GJK/MPR cannot handle infinite planes, we create a finite cube where:
    - The cube is positioned with its top face at the plane surface
    - The cube's lateral dimensions are sized based on the other object's bounding sphere
    - The cube extends only 'downward' from the plane (half-space in -Z direction in plane's local frame)

    Args:
        shape_data: The plane's shape data (should have shape_type == GeoType.PLANE)
        plane_rotation: The plane's orientation (plane normal is along local +Z)
        plane_position: The plane's position in world space
        other_position: The other object's position in world space
        other_radius: Bounding sphere radius of the colliding object

    Returns:
        Tuple of (modified_shape_data, adjusted_position):
        - modified_shape_data: GenericShapeData configured as a BOX
        - adjusted_position: The cube's center position (centered on other object projected to plane)
    """
    result = GenericShapeData()
    result.shape_type = int(GeoType.BOX)

    # Size the cube based on the other object's bounding sphere radius
    # Make it large enough to always contain potential contact points
    # The lateral dimensions (x, y) should be at least 2x the radius to ensure coverage
    lateral_size = other_radius * 10.0

    # The depth (z) should be large enough to encompass the potential collision region
    # Half-space behavior: cube extends only below the plane surface (negative Z)
    depth = other_radius * 10.0

    # Set the box half-extents
    # x, y: lateral coverage (parallel to plane)
    # z: depth perpendicular to plane
    result.scale = wp.vec3(lateral_size, lateral_size, depth)

    # Position the cube center at the plane surface, directly under/over the other object
    # Project the other object's position onto the plane
    plane_normal = wp.quat_rotate(plane_rotation, wp.vec3(0.0, 0.0, 1.0))
    to_other = other_position - plane_position
    distance_along_normal = wp.dot(to_other, plane_normal)

    # Point on plane surface closest to the other object
    plane_surface_point = other_position - plane_normal * distance_along_normal

    # Position cube center slightly below the plane surface so the top face is at the surface
    # Since the cube has half-extent 'depth', its top face is at center + depth*normal
    # We want: center + depth*normal = plane_surface, so center = plane_surface - depth*normal
    adjusted_position = plane_surface_point - plane_normal * depth

    return result, adjusted_position


@wp.func
def check_infinite_plane_bsphere_overlap(
    shape_data_a: GenericShapeData,
    shape_data_b: GenericShapeData,
    pos_a: wp.vec3,
    pos_b: wp.vec3,
    quat_a: wp.quat,
    quat_b: wp.quat,
    bsphere_center_a: wp.vec3,
    bsphere_center_b: wp.vec3,
    bsphere_radius_a: float,
    bsphere_radius_b: float,
) -> bool:
    """
    Check if an infinite plane overlaps with another shape's bounding sphere.
    Treats the plane as a half-space: objects on or below the plane (negative side of the normal)
    are considered to overlap and will generate contacts.
    Returns True if they overlap, False otherwise.
    Uses data already extracted by extract_shape_data.
    """
    type_a = shape_data_a.shape_type
    type_b = shape_data_b.shape_type
    scale_a = shape_data_a.scale
    scale_b = shape_data_b.scale

    # Check if either shape is an infinite plane
    is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
    is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

    # If neither is an infinite plane, return True (no culling)
    if not (is_infinite_plane_a or is_infinite_plane_b):
        return True

    # Determine which is the plane and which is the other shape
    if is_infinite_plane_a:
        plane_pos = pos_a
        plane_quat = quat_a
        other_center = bsphere_center_b
        other_radius = bsphere_radius_b
    else:
        plane_pos = pos_b
        plane_quat = quat_b
        other_center = bsphere_center_a
        other_radius = bsphere_radius_a

    # Compute plane normal (plane's local +Z axis in world space)
    plane_normal = wp.quat_rotate(plane_quat, wp.vec3(0.0, 0.0, 1.0))

    # Distance from sphere center to plane (positive = above plane, negative = below plane)
    center_dist = wp.dot(other_center - plane_pos, plane_normal)

    # Treat plane as a half-space: objects on or below the plane (negative side) generate contacts
    # Remove absolute value to only check penetration side
    return center_dist <= other_radius
