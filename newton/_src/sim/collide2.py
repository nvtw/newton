from __future__ import annotations

from enum import IntEnum

import warp as wp

from ..core.types import Devicelike
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_convex import create_solve_convex_multi_contact
from ..geometry.collision_primitive import collide_plane_box, collide_plane_sphere, collide_sphere_box, collide_box_box
from ..geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    support_map as support_map_func,
)
from ..geometry.types import GeoType
from .contacts import Contacts
from .model import Model
from .state import State


class BroadPhaseMode(IntEnum):
    """Broad phase collision detection mode.

    Attributes:
        NONE: No broad phase, use explicitly provided shape pairs (fastest if pairs are known)
        NXN: All-pairs broad phase with AABB checks (simple, O(N²) but good for small scenes)
        SAP: Sweep and Prune broad phase with AABB sorting (faster for larger scenes, O(N log N))
    """

    NONE = 0
    NXN = 1
    SAP = 2


# Pre-create the convex multi-contact solver (usable inside kernels)
solve_convex_multi_contact = create_solve_convex_multi_contact(support_map_func)


@wp.func
def write_contact(
    contact_point_center: wp.vec3,
    contact_normal_a_to_b: wp.vec3,
    contact_distance: float,
    radius_eff_a: float,
    radius_eff_b: float,
    thickness_a: float,
    thickness_b: float,
    shape_a: int,
    shape_b: int,
    X_bw_a: wp.transform,
    X_bw_b: wp.transform,
    tid: int,
    rigid_contact_margin: float,
    contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    """
    Write a contact to the output arrays.

    Args:
        contact_point_center: Center point of contact in world space
        contact_normal_a_to_b: Contact normal pointing from shape A to B
        contact_distance: Distance between contact points
        radius_eff_a: Effective radius of shape A (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        radius_eff_b: Effective radius of shape B (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        thickness_a: Contact thickness for shape A (similar to contact offset)
        thickness_b: Contact thickness for shape B (similar to contact offset)
        shape_a: Shape A index
        shape_b: Shape B index
        X_bw_a: Transform from world to body A
        X_bw_b: Transform from world to body B
        tid: Thread ID
        rigid_contact_margin: Contact margin for rigid bodies
        contact_max: Maximum number of contacts
        contact_count: Array to track contact count
        out_shape0: Output array for shape A indices
        out_shape1: Output array for shape B indices
        out_point0: Output array for contact points on shape A
        out_point1: Output array for contact points on shape B
        out_offset0: Output array for offsets on shape A
        out_offset1: Output array for offsets on shape B
        out_normal: Output array for contact normals
        out_thickness0: Output array for thickness values for shape A
        out_thickness1: Output array for thickness values for shape B
        out_tids: Output array for thread IDs
    """

    total_separation_needed = radius_eff_a + radius_eff_b + thickness_a + thickness_b

    offset_mag_a = radius_eff_a + thickness_a
    offset_mag_b = radius_eff_b + thickness_b

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_normal_a_to_b)

    a_contact_world = contact_point_center - contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_a)
    b_contact_world = contact_point_center + contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_b)

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed
    if d < rigid_contact_margin:
        index = wp.atomic_add(contact_count, 0, 1)
        if index >= contact_max:
            # Reached buffer limit
            return

        out_shape0[index] = shape_a
        out_shape1[index] = shape_b

        # Contact points are stored in body frames
        out_point0[index] = wp.transform_point(X_bw_a, a_contact_world)
        out_point1[index] = wp.transform_point(X_bw_b, b_contact_world)

        # Match kernels.py convention: normal should point from box to plane (downward)
        contact_normal = -contact_normal_a_to_b

        # Offsets in body frames
        out_offset0[index] = wp.transform_vector(X_bw_a, -offset_mag_a * contact_normal)
        out_offset1[index] = wp.transform_vector(X_bw_b, offset_mag_b * contact_normal)

        out_normal[index] = contact_normal
        out_thickness0[index] = offset_mag_a
        out_thickness1[index] = offset_mag_b
        out_tids[index] = tid


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
    - The cube is positioned at the plane surface, directly under/over the other object
    - The cube's lateral dimensions are sized based on the other object's bounding sphere
    - The cube extends 'downward' from the plane (in -Z direction in plane's local frame)

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
    lateral_size = other_radius * 2.0

    # The depth (z) should be large enough to encompass the potential collision region
    # Make it extend from above the plane surface to well below
    depth = other_radius * 2.0

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


@wp.kernel(enable_backward=False)
def build_contacts_kernel_gjk_mpr(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    shape_collision_radius: wp.array(dtype=float),
    shape_pairs: wp.array(dtype=wp.vec2i),
    sum_of_contact_offsets: float,
    rigid_contact_margin: float,
    contact_max: int,
    num_pairs: int,
    sap_shape_pair_count: wp.array(dtype=int),
    # outputs
    contact_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    tid = wp.tid()
    # if tid == 0:
    #     wp.printf("build_contacts_kernel_gjk_mpr\n")

    # Early bounds check - following convert_newton_contacts_to_mjwarp_kernel pattern
    if tid >= shape_pairs.shape[0]:
        return

    if num_pairs == -1 and tid >= sap_shape_pair_count[0]:
        return

    pair = shape_pairs[tid]
    shape_a = pair[0]
    shape_b = pair[1]

    # Safety: ignore self-collision pairs
    if shape_a == shape_b:
        return

    # Validate shape indices - following convert_newton_contacts_to_mjwarp_kernel pattern
    if shape_a < 0 or shape_b < 0:
        return

    # Get shape types early for validation
    type_a = shape_type[shape_a]
    type_b = shape_type[shape_b]

    # Sort shapes by type to ensure consistent collision handling order
    if type_a > type_b:
        # Swap shapes to maintain consistent ordering
        shape_a, shape_b = shape_b, shape_a
        type_a, type_b = type_b, type_a

    # Get body indices - following convert_newton_contacts_to_mjwarp_kernel pattern
    rigid_a = shape_body[shape_a]
    rigid_b = shape_body[shape_b]

    # Compute world transforms
    X_ws_a = (
        shape_transform[shape_a] if rigid_a == -1 else wp.transform_multiply(body_q[rigid_a], shape_transform[shape_a])
    )
    X_ws_b = (
        shape_transform[shape_b] if rigid_b == -1 else wp.transform_multiply(body_q[rigid_b], shape_transform[shape_b])
    )

    pos_a = wp.transform_get_translation(X_ws_a)
    pos_b = wp.transform_get_translation(X_ws_b)

    # Check if shapes are infinite planes (scale.x == 0 and scale.y == 0)
    scale_a = shape_scale[shape_a]
    scale_b = shape_scale[shape_b]
    is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
    is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

    # Early bounding sphere check using precomputed collision radii - inflate by rigid_contact_margin
    radius_a = shape_collision_radius[shape_a] + rigid_contact_margin
    radius_b = shape_collision_radius[shape_b] + rigid_contact_margin
    center_distance = wp.length(pos_b - pos_a)

    # Early out if bounding spheres don't overlap
    # Skip this check if either shape is an infinite plane
    if not (is_infinite_plane_a or is_infinite_plane_b):
        if center_distance > (radius_a + radius_b):
            return
    elif is_infinite_plane_a and is_infinite_plane_b:
        # Plane-plane collisions are not supported
        return
    elif is_infinite_plane_a:
        # Check if shape B is close enough to the infinite plane A
        rot_a_temp = wp.transform_get_rotation(X_ws_a)
        plane_normal = wp.quat_rotate(rot_a_temp, wp.vec3(0.0, 0.0, 1.0))
        # Distance from shape B center to plane A
        dist_to_plane = wp.abs(wp.dot(pos_b - pos_a, plane_normal))
        if dist_to_plane > radius_b:
            return
    elif is_infinite_plane_b:
        # Check if shape A is close enough to the infinite plane B
        rot_b_temp = wp.transform_get_rotation(X_ws_b)
        plane_normal = wp.quat_rotate(rot_b_temp, wp.vec3(0.0, 0.0, 1.0))
        # Distance from shape A center to plane B
        dist_to_plane = wp.abs(wp.dot(pos_a - pos_b, plane_normal))
        if dist_to_plane > radius_a:
            return

    rot_a = wp.transform_get_rotation(X_ws_a)
    rot_b = wp.transform_get_rotation(X_ws_b)

    # World->body transforms for writing contact points
    X_bw_a = wp.transform_identity() if rigid_a == -1 else wp.transform_inverse(body_q[rigid_a])
    X_bw_b = wp.transform_identity() if rigid_b == -1 else wp.transform_inverse(body_q[rigid_b])

    # Create geometry data structures for convex collision detection
    geom_a = GenericShapeData()
    geom_a.shape_type = type_a
    geom_a.scale = scale_a

    # Convert infinite planes to cube proxies for GJK/MPR compatibility
    # Use the OTHER object's radius to properly size the cube
    # Only convert if it's an infinite plane (finite planes can be handled normally)
    pos_a_adjusted = pos_a
    if is_infinite_plane_a:
        # Position the cube based on the OTHER object's position (pos_b)
        geom_a, pos_a_adjusted = convert_infinite_plane_to_cube(geom_a, rot_a, pos_a, pos_b, radius_b)

    geom_b = GenericShapeData()
    geom_b.shape_type = type_b
    geom_b.scale = scale_b

    pos_b_adjusted = pos_b
    if is_infinite_plane_b:
        # Position the cube based on the OTHER object's position (pos_a)
        geom_b, pos_b_adjusted = convert_infinite_plane_to_cube(geom_b, rot_b, pos_b, pos_a, radius_a)

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

    count, normal, penetrations, points, features = wp.static(solve_convex_multi_contact)(
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

    # Special post processing for minkowski objects
    if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
        for i in range(count):
            points[i] = points[i] + normal * (radius_eff_a * 0.5)
            penetrations[i] -= radius_eff_a - small_radius
    if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
        for i in range(count):
            points[i] = points[i] - normal * (radius_eff_b * 0.5)
            penetrations[i] -= radius_eff_b - small_radius

    for id in range(count):
        write_contact(
            points[id],
            normal,
            penetrations[id],
            radius_eff_a,
            radius_eff_b,
            shape_thickness[shape_a],
            shape_thickness[shape_b],
            shape_a,
            shape_b,
            X_bw_a,
            X_bw_b,
            tid,
            rigid_contact_margin,
            contact_max,
            contact_count,
            out_shape0,
            out_shape1,
            out_point0,
            out_point1,
            out_offset0,
            out_offset1,
            out_normal,
            out_thickness0,
            out_thickness1,
            out_tids,
        )

    return

    if type_a == int(GeoType.PLANE) and type_b == int(GeoType.BOX):
        plane_normal_world = wp.transform_vector(X_ws_a, wp.vec3(0.0, 0.0, 1.0))
        box_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_b))

        # Call collide_plane_box to get contact information
        contact_distances, contact_positions, contact_normal = collide_plane_box(
            plane_normal_world,  # plane_normal
            pos_a,  # plane_pos (plane position in world space)
            pos_b,  # box_pos (box position in world space)
            box_rot_mat,  # box_rot (box rotation matrix)
            shape_scale[shape_b],  # box_size (box half-extents)
        )

        for id in range(4):
            dist = contact_distances[id]
            if dist < wp.inf:
                write_contact(
                    contact_positions[id],
                    contact_normal,
                    contact_distances[id],
                    0.0,
                    0.0,
                    shape_thickness[shape_a],
                    shape_thickness[shape_b],
                    shape_a,
                    shape_b,
                    X_bw_a,
                    X_bw_b,
                    tid,
                    rigid_contact_margin,
                    contact_max,
                    contact_count,
                    out_shape0,
                    out_shape1,
                    out_point0,
                    out_point1,
                    out_offset0,
                    out_offset1,
                    out_normal,
                    out_thickness0,
                    out_thickness1,
                    out_tids,
                )

    # Implement Plane vs Sphere contacts (type order enforced above)
    elif type_a == int(GeoType.PLANE) and type_b == int(GeoType.SPHERE):
        # Plane frame helpers
        plane_normal_world = wp.transform_vector(X_ws_a, wp.vec3(0.0, 0.0, 1.0))

        # Sphere radius
        sphere_radius = shape_scale[shape_b][0]  # Sphere radius is stored in x component

        # Call collide_plane_sphere to get contact information
        contact_distance, contact_position = collide_plane_sphere(
            plane_normal_world,  # plane_normal
            pos_a,  # plane_pos (plane position in world space)
            pos_b,  # sphere_pos (sphere position in world space)
            sphere_radius,  # sphere_radius
        )

        # Use the write_contact function to write the contact
        write_contact(
            contact_position,
            plane_normal_world,  # contact normal from plane to sphere
            contact_distance,
            0.0,
            sphere_radius,
            shape_thickness[shape_a],
            shape_thickness[shape_b],
            shape_a,
            shape_b,
            X_bw_a,
            X_bw_b,
            tid,
            rigid_contact_margin,
            contact_max,
            contact_count,
            out_shape0,
            out_shape1,
            out_point0,
            out_point1,
            out_offset0,
            out_offset1,
            out_normal,
            out_thickness0,
            out_thickness1,
            out_tids,
        )

    # Implement Sphere vs Box contacts (type order enforced above)
    elif type_a == int(GeoType.SPHERE) and type_b == int(GeoType.BOX):
        # Sphere radius
        sphere_radius = shape_scale[shape_a][0]  # Sphere radius is stored in x component

        # Box half extents
        box_half_extents = shape_scale[shape_b]  # Box half-extents (hx, hy, hz)

        # Get box rotation matrix from quaternion
        box_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_b))

        # Call collide_sphere_box to get contact information
        contact_distance, contact_position, contact_normal = collide_sphere_box(
            pos_a,  # sphere_pos (sphere position in world space)
            sphere_radius,  # sphere_radius
            pos_b,  # box_pos (box position in world space)
            box_rot_mat,  # box_rot (box rotation matrix)
            box_half_extents,  # box_size (box half-extents)
        )

        # Print contact information in the requested format
        wp.printf(
            "point_a: (%f,%f,%f), point_b: (%f,%f,%f), normal: (%f,%f,%f), dist: %f\n",
            contact_position[0] + contact_normal[0] * sphere_radius,  # point_a
            contact_position[1] + contact_normal[1] * sphere_radius,
            contact_position[2] + contact_normal[2] * sphere_radius,
            contact_position[0],  # point_b
            contact_position[1],
            contact_position[2],
            contact_normal[0],  # normal
            contact_normal[1],
            contact_normal[2],
            contact_distance,  # distance
        )

        # Use the write_contact function to write the contact
        write_contact(
            contact_position,
            contact_normal,  # contact normal from sphere to box
            contact_distance,
            sphere_radius,  # sphere's effective radius
            0.0,  # box has no effective radius
            shape_thickness[shape_a],
            shape_thickness[shape_b],
            shape_a,
            shape_b,
            X_bw_a,
            X_bw_b,
            tid,
            rigid_contact_margin,
            contact_max,
            contact_count,
            out_shape0,
            out_shape1,
            out_point0,
            out_point1,
            out_offset0,
            out_offset1,
            out_normal,
            out_thickness0,
            out_thickness1,
            out_tids,
        )

    # Implement Box vs Box contacts (type order enforced above)
    elif type_a == int(GeoType.BOX) and type_b == int(GeoType.BOX):
        # Box half extents for both boxes
        box1_half_extents = shape_scale[shape_a]  # Box A half-extents (hx, hy, hz)
        box2_half_extents = shape_scale[shape_b]  # Box B half-extents (hx, hy, hz)

        # Get box rotation matrices from quaternions
        box1_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_a))
        box2_rot_mat = wp.quat_to_matrix(wp.transform_get_rotation(X_ws_b))

        # Call collide_box_box to get contact information
        contact_distances8, contact_positions8, contact_normals8 = collide_box_box(
            pos_a,  # box1_pos (first box position in world space)
            box1_rot_mat,  # box1_rot (first box rotation matrix)
            box1_half_extents,  # box1_size (first box half-extents)
            pos_b,  # box2_pos (second box position in world space)
            box2_rot_mat,  # box2_rot (second box rotation matrix)
            box2_half_extents,  # box2_size (second box half-extents)
        )

        # Process up to 8 contact points (box-box can generate multiple contacts)
        for id in range(8):
            dist = contact_distances8[id]
            if dist < wp.inf:
                # Use the write_contact function to write the contact
                write_contact(
                    contact_positions8[id],
                    contact_normals8[id],  # contact normal for this specific contact
                    dist,
                    0.0,  # box A has no effective radius
                    0.0,  # box B has no effective radius
                    shape_thickness[shape_a],
                    shape_thickness[shape_b],
                    shape_a,
                    shape_b,
                    X_bw_a,
                    X_bw_b,
                    tid,
                    rigid_contact_margin,
                    contact_max,
                    contact_count,
                    out_shape0,
                    out_shape1,
                    out_point0,
                    out_point1,
                    out_offset0,
                    out_offset1,
                    out_normal,
                    out_thickness0,
                    out_thickness1,
                    out_tids,
                )


@wp.kernel
def compute_shape_aabbs(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    rigid_contact_margin: float,
    # outputs
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
):
    """Compute axis-aligned bounding boxes for each shape in world space.

    AABBs are enlarged by rigid_contact_margin to account for contact detection margin.
    """
    shape_id = wp.tid()

    rigid_id = shape_body[shape_id]

    # Compute world transform
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    rot = wp.transform_get_rotation(X_ws)
    scale = shape_scale[shape_id]
    geo_type = shape_type[shape_id]

    # Compute conservative AABB based on shape type
    # For simplicity, we use bounding sphere approach for now
    if geo_type == int(GeoType.SPHERE):
        radius = scale[0]
        half_extents = wp.vec3(radius, radius, radius)
    elif geo_type == int(GeoType.BOX):
        # Transform box corners to get tight AABB
        rot_mat = wp.quat_to_matrix(rot)
        abs_rot = wp.mat33(
            wp.abs(rot_mat[0, 0]),
            wp.abs(rot_mat[0, 1]),
            wp.abs(rot_mat[0, 2]),
            wp.abs(rot_mat[1, 0]),
            wp.abs(rot_mat[1, 1]),
            wp.abs(rot_mat[1, 2]),
            wp.abs(rot_mat[2, 0]),
            wp.abs(rot_mat[2, 1]),
            wp.abs(rot_mat[2, 2]),
        )
        half_extents = abs_rot * scale
    elif geo_type == int(GeoType.CAPSULE):
        radius = scale[0]
        half_height = scale[1]
        # Conservative: sphere of radius (radius + half_height)
        r = radius + half_height
        half_extents = wp.vec3(r, r, r)
    elif geo_type == int(GeoType.PLANE):
        # Infinite plane gets huge AABB
        if scale[0] > 0.0 and scale[1] > 0.0:
            # Finite plane
            half_extents = wp.vec3(scale[0], scale[1], 0.1)
        else:
            # Infinite plane
            half_extents = wp.vec3(1.0e6, 1.0e6, 1.0e6)
    else:
        # Default: use scale as conservative estimate
        half_extents = scale

    # Enlarge AABB by rigid_contact_margin for contact detection
    margin_vec = wp.vec3(rigid_contact_margin, rigid_contact_margin, rigid_contact_margin)
    aabb_lower[shape_id] = pos - half_extents - margin_vec
    aabb_upper[shape_id] = pos + half_extents + margin_vec


class CollisionPipeline2:
    """
    CollisionPipeline manages collision detection and contact generation for a simulation.

    This class is responsible for allocating and managing buffers for collision detection,
    generating rigid and soft contacts between shapes and particles, and providing an interface
    for running the collision pipeline on a given simulation state.
    """

    def __init__(
        self,
        shape_count: int,
        particle_count: int,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        rigid_contact_max: int | None = None,
        rigid_contact_max_per_pair: int = 10,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool = False,
        device: Devicelike = None,
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.NXN,
    ):
        """
        Initialize the CollisionPipeline.

        Args:
            shape_count (int): Number of shapes in the simulation.
            particle_count (int): Number of particles in the simulation.
            shape_pairs_filtered (wp.array | None, optional): Array of filtered shape pairs to consider for collision.
                Only used when broad_phase_mode is BroadPhaseMode.NONE.
            rigid_contact_max (int | None, optional): Maximum number of rigid contacts to allocate.
                If None, computed as shape_pairs_max * rigid_contact_max_per_pair.
            rigid_contact_max_per_pair (int, optional): Maximum number of contact points per shape pair. Defaults to 10.
            rigid_contact_margin (float, optional): Margin for rigid contact generation. Defaults to 0.01.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            iterate_mesh_vertices (bool, optional): Whether to iterate mesh vertices for collision. Defaults to True.
            requires_grad (bool, optional): Whether to enable gradient computation. Defaults to False.
            device (Devicelike, optional): The device on which to allocate arrays and perform computation.
            broad_phase_mode (BroadPhaseMode, optional): Broad phase mode for collision detection.
                - BroadPhaseMode.NONE: Use explicitly provided shape_pairs_filtered
                - BroadPhaseMode.NXN: Use all-pairs AABB broad phase (O(N²), good for small scenes)
                - BroadPhaseMode.SAP: Use sweep-and-prune AABB broad phase (O(N log N), better for larger scenes)
                Defaults to BroadPhaseMode.NXN.
        """
        # will be allocated during collide
        self.contacts = None

        self.shape_count = shape_count
        self.shape_pairs_filtered = shape_pairs_filtered
        self.broad_phase_mode = broad_phase_mode

        if self.shape_pairs_filtered is not None:
            self.shape_pairs_max = len(self.shape_pairs_filtered)
        else:
            # When using SAP, estimate based on worst case (all pairs)
            self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2

        self.rigid_contact_margin = rigid_contact_margin
        if rigid_contact_max is not None:
            self.rigid_contact_max = rigid_contact_max
        else:
            self.rigid_contact_max = self.shape_pairs_max * rigid_contact_max_per_pair

        # Initialize broad phase based on mode
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            self.nxn_broadphase = BroadPhaseAllPairs()
            self.sap_broadphase = None
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            # Estimate max groups for SAP - use reasonable defaults
            max_num_negative_group_members = max(int(shape_count**0.5), 10)
            max_num_distinct_positive_groups = max(int(shape_count**0.5), 10)
            self.sap_broadphase = BroadPhaseSAP(
                max_broad_phase_elements=shape_count,
                max_num_distinct_positive_groups=max_num_distinct_positive_groups,
                max_num_negative_group_members=max_num_negative_group_members,
            )
            self.nxn_broadphase = None
        else:  # BroadPhaseMode.NONE
            self.nxn_broadphase = None
            self.sap_broadphase = None

        # Allocate buffers for broadphase collision handling
        with wp.ScopedDevice(device):
            self.rigid_pair_shape0 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
            self.rigid_pair_shape1 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
            self.rigid_pair_point_limit = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
            self.rigid_pair_point_count = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
            self.rigid_pair_point_id = wp.empty(self.rigid_contact_max, dtype=wp.int32)

            # Allocate buffers for dynamically computed pairs and AABBs if using broad phase
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            if self.broad_phase_mode != BroadPhaseMode.NONE:
                self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
                self.shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
                self.shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)
                # Allocate dummy collision/shape group arrays once (reused every frame)
                # Collision group: 1 (positive) = all shapes in same group, collide with each other
                # Environment group: -1 (global) = collides with all environments
                self.dummy_collision_group = wp.full(shape_count, 1, dtype=wp.int32, device=device)
                self.dummy_shape_group = wp.full(shape_count, -1, dtype=wp.int32, device=device)

        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max

        self.iterate_mesh_vertices = iterate_mesh_vertices
        self.requires_grad = requires_grad
        self.edge_sdf_iter = edge_sdf_iter

    @classmethod
    def from_model(
        cls,
        model: Model,
        rigid_contact_max_per_pair: int | None = None,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool | None = None,
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.NXN,
    ) -> CollisionPipeline:
        """
        Create a CollisionPipeline instance from a Model.

        Args:
            model (Model): The simulation model.
            rigid_contact_max_per_pair (int | None, optional): Maximum number of contact points per shape pair.
                If None, uses model.rigid_contact_max and sets per-pair to 0.
            rigid_contact_margin (float, optional): Margin for rigid contact generation. Defaults to 0.01.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            iterate_mesh_vertices (bool, optional): Whether to iterate mesh vertices for collision. Defaults to True.
            requires_grad (bool | None, optional): Whether to enable gradient computation. If None, uses model.requires_grad.
            broad_phase_mode (BroadPhaseMode, optional): Broad phase collision detection mode. Defaults to BroadPhaseMode.NXN.

        Returns:
            CollisionPipeline: The constructed collision pipeline.
        """
        rigid_contact_max = None
        if rigid_contact_max_per_pair is None:
            rigid_contact_max = model.rigid_contact_max
            rigid_contact_max_per_pair = 0
        if requires_grad is None:
            requires_grad = model.requires_grad
        # Use model.shape_contact_pairs if BroadPhaseMode.NONE, otherwise None (dynamic broad phase)
        shape_pairs_filtered = model.shape_contact_pairs if broad_phase_mode == BroadPhaseMode.NONE else None

        return CollisionPipeline2(
            model.shape_count,
            model.particle_count,
            shape_pairs_filtered,
            rigid_contact_max,
            rigid_contact_max_per_pair,
            rigid_contact_margin,
            soft_contact_max,
            soft_contact_margin,
            edge_sdf_iter,
            iterate_mesh_vertices,
            requires_grad,
            model.device,
            broad_phase_mode,
        )

    def collide(self, model: Model, state: State) -> Contacts:
        """
        Run the collision pipeline for the given model and state, generating contacts.

        This method allocates or clears the contact buffer as needed, then generates
        soft and rigid contacts using the current simulation state. If using SAP broad phase,
        potentially colliding pairs are computed dynamically based on bounding sphere overlaps.

        Args:
            model (Model): The simulation model.
            state (State): The current simulation state.

        Returns:
            Contacts: The generated contacts for the current state.
        """
        # Allocate new contact memory for contacts if needed (e.g., for gradients)
        if self.contacts is None or self.requires_grad:
            self.contacts = Contacts(
                self.rigid_contact_max,
                self.soft_contact_max,
                requires_grad=self.requires_grad,
                device=model.device,
            )
        else:
            self.contacts.clear()

        # output contacts buffer
        contacts = self.contacts

        # Determine which shape pairs to check based on broad phase mode
        if self.broad_phase_mode == BroadPhaseMode.NONE:
            # Use explicitly provided shape pairs
            shape_pairs = self.shape_pairs_filtered
            num_pairs = len(self.shape_pairs_filtered) if self.shape_pairs_filtered is not None else 0
        else:
            # Compute AABBs for all shapes in world space (needed for both NXN and SAP)
            # AABBs are enlarged by rigid_contact_margin to ensure contact detection
            wp.launch(
                kernel=compute_shape_aabbs,
                dim=model.shape_count,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    self.rigid_contact_margin,
                ],
                outputs=[
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                ],
                device=model.device,
            )

            # Run appropriate broad phase
            self.broad_phase_pair_count.zero_()

            if self.broad_phase_mode == BroadPhaseMode.NXN:
                # Use NxN all-pairs broad phase with AABB overlaps
                self.nxn_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    model.shape_thickness,  # Use thickness as cutoff
                    self.dummy_collision_group,  # Preallocated, all 1 (same group = collide)
                    self.dummy_shape_group,  # Preallocated, all -1 (global entities)
                    model.shape_count,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                )
            else:  # BroadPhaseMode.SAP
                # Use sweep-and-prune broad phase with AABB sorting
                self.sap_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    model.shape_thickness,  # Use thickness as cutoff
                    self.dummy_collision_group,  # Preallocated, all 1 (same group = collide)
                    self.dummy_shape_group,  # Preallocated, all -1 (global entities)
                    model.shape_count,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                )

            shape_pairs = self.broad_phase_shape_pairs
            num_pairs = -1  # Use dynamic count from kernel

        # Launch kernel across all shape pairs
        block_dim = 128
        if num_pairs > 0 or num_pairs == -1:
            wp.launch(
                kernel=build_contacts_kernel_gjk_mpr,
                dim=num_pairs if num_pairs != -1 else block_dim * 256,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_thickness,
                    model.shape_collision_radius,
                    shape_pairs,
                    0.0,  # sum_of_contact_offsets
                    self.rigid_contact_margin,
                    contacts.rigid_contact_max,
                    num_pairs,
                    self.broad_phase_pair_count,
                ],
                outputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_thickness0,
                    contacts.rigid_contact_thickness1,
                    contacts.rigid_contact_tids,
                ],
                device=contacts.device,
                block_dim=block_dim,
            )

        return contacts
