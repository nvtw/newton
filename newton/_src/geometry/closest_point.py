# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Closest-point queries on convex primitives in their local frame.

These helpers provide exact closest-point queries for simple primitives.
They are intentionally separate from GJK/MPR seeding: MPR's ``v0`` is not
just a closest surface point, and feeding approximate closest points (for
example closest vertices on convex hulls) can destabilize resting contacts.

Convention matches ``GenericShapeData`` in :mod:`support_function`:

- BOX: half-extents in ``scale``
- SPHERE: radius in ``scale.x``
- CAPSULE: radius in ``scale.x``, half-height in ``scale.y``, axis +Z
- CYLINDER: radius in ``scale.x``, half-height in ``scale.y``, axis +Z
  (only handled when uniform; tapered cylinders fall back)
- PLANE: half-width in ``scale.x``, half-length in ``scale.y``; the dispatcher
  only handles the infinite-plane case (``width <= 0`` or ``length <= 0``).
- CONVEX_MESH: scale on ``scale``, packed mesh pointer in ``auxiliary``.
- TRIANGLE / TRIANGLE_PRISM: vertex A at origin, B-A in ``scale``, C-A in
  ``auxiliary``.

For shapes not covered (ellipsoid, cone, tapered cylinder, finite plane
quad) ``closest_point_on_shape`` returns ``handled=False`` so callers can
fall back to a safer method.
"""

from typing import Any

import warp as wp

from .support_function import GeoTypeEx, closest_point_on_triangle, unpack_mesh_ptr
from .types import GeoType


@wp.func
def closest_point_sphere(query: wp.vec3, radius: float) -> wp.vec3:
    """Closest point on a sphere centered at the local origin."""
    eps = 1.0e-12
    q_len_sq = wp.length_sq(query)
    if q_len_sq <= eps:
        return wp.vec3(radius, 0.0, 0.0)
    return query * (radius / wp.sqrt(q_len_sq))


@wp.func
def closest_point_box_local(half_extents: wp.vec3, query: wp.vec3) -> wp.vec3:
    """Closest point on the *surface* of an axis-aligned box centered at origin.

    Exterior queries clamp to the box. Interior queries snap to the nearest
    face — this is critical for GJK/MPR seeding under interpenetration: a
    surface seed produces a meaningful ``BtoA`` direction along the contact
    normal, whereas returning the query unchanged would collapse the seed
    when the shapes overlap (e.g. resting box pyramids).
    """
    inside_x = wp.abs(query[0]) <= half_extents[0]
    inside_y = wp.abs(query[1]) <= half_extents[1]
    inside_z = wp.abs(query[2]) <= half_extents[2]
    if inside_x and inside_y and inside_z:
        # Interior: project to the closest face (Chebyshev nearest face).
        sx = half_extents[0] - wp.abs(query[0])
        sy = half_extents[1] - wp.abs(query[1])
        sz = half_extents[2] - wp.abs(query[2])
        sign_x = 1.0 if query[0] >= 0.0 else -1.0
        sign_y = 1.0 if query[1] >= 0.0 else -1.0
        sign_z = 1.0 if query[2] >= 0.0 else -1.0
        if sx <= sy and sx <= sz:
            return wp.vec3(sign_x * half_extents[0], query[1], query[2])
        if sy <= sz:
            return wp.vec3(query[0], sign_y * half_extents[1], query[2])
        return wp.vec3(query[0], query[1], sign_z * half_extents[2])
    x = wp.clamp(query[0], -half_extents[0], half_extents[0])
    y = wp.clamp(query[1], -half_extents[1], half_extents[1])
    z = wp.clamp(query[2], -half_extents[2], half_extents[2])
    return wp.vec3(x, y, z)


@wp.func
def closest_point_capsule_z(query: wp.vec3, radius: float, half_height: float) -> wp.vec3:
    """Closest point on the *surface* of a capsule (axis +Z, centered at origin).

    Both exterior and interior queries return a surface point — interior
    queries get pushed radially out to the surface so the seed always
    yields a meaningful direction (see :func:`closest_point_box_local`).
    """
    eps = 1.0e-12
    z_clamped = wp.clamp(query[2], -half_height, half_height)
    axis_point = wp.vec3(0.0, 0.0, z_clamped)
    diff = query - axis_point
    dist_sq = wp.length_sq(diff)
    if dist_sq <= eps:
        return axis_point + wp.vec3(radius, 0.0, 0.0)
    return axis_point + diff * (radius / wp.sqrt(dist_sq))


@wp.func
def closest_point_cylinder_z(query: wp.vec3, radius: float, half_height: float) -> wp.vec3:
    """Closest point on the *surface* of a uniform cylinder (axis +Z, centered at origin).

    Both exterior and interior queries return a surface point — interior
    queries snap to the closest of the two caps or the lateral wall.
    """
    eps = 1.0e-12
    r_xy_sq = query[0] * query[0] + query[1] * query[1]
    abs_z = wp.abs(query[2])
    inside_radial = r_xy_sq <= radius * radius
    inside_axial = abs_z <= half_height
    if inside_radial and inside_axial:
        # Interior: pick the nearest of {top cap, bottom cap, lateral surface}.
        r_xy = wp.sqrt(r_xy_sq)
        radial_gap = radius - r_xy
        axial_gap = half_height - abs_z
        if radial_gap <= axial_gap:
            if r_xy <= eps:
                return wp.vec3(radius, 0.0, query[2])
            inv_r = radius / r_xy
            return wp.vec3(query[0] * inv_r, query[1] * inv_r, query[2])
        sign_z = 1.0 if query[2] >= 0.0 else -1.0
        return wp.vec3(query[0], query[1], sign_z * half_height)
    z_clamped = wp.clamp(query[2], -half_height, half_height)
    if r_xy_sq <= eps:
        return wp.vec3(0.0, 0.0, z_clamped)
    if r_xy_sq <= radius * radius:
        return wp.vec3(query[0], query[1], z_clamped)
    inv_r = radius / wp.sqrt(r_xy_sq)
    return wp.vec3(query[0] * inv_r, query[1] * inv_r, z_clamped)


@wp.func
def closest_point_plane_infinite(query: wp.vec3) -> wp.vec3:
    """Project onto the infinite plane z=0."""
    return wp.vec3(query[0], query[1], 0.0)


@wp.func
def closest_point_convex_mesh_face(
    geom: Any,
    query: wp.vec3,
    data_provider: Any,
) -> wp.vec3:
    """Closest point on a convex mesh's triangle surface.

    This is an O(F) exact closest-surface query over the hull's triangles.
    It deliberately uses triangle projection, not closest vertex, so flat
    faces return their face interior instead of an arbitrary corner.
    """
    _ = data_provider
    mesh_ptr = unpack_mesh_ptr(geom.auxiliary)
    mesh = wp.mesh_get(mesh_ptr)
    mesh_scale = geom.scale

    best_point = wp.vec3(0.0, 0.0, 0.0)
    best_dist_sq = float(1.0e30)
    tri_count = mesh.indices.shape[0] // 3
    for tri_idx in range(tri_count):
        idx0 = mesh.indices[tri_idx * 3 + 0]
        idx1 = mesh.indices[tri_idx * 3 + 1]
        idx2 = mesh.indices[tri_idx * 3 + 2]
        v0 = wp.cw_mul(mesh.points[idx0], mesh_scale)
        v1 = wp.cw_mul(mesh.points[idx1], mesh_scale)
        v2 = wp.cw_mul(mesh.points[idx2], mesh_scale)
        cp = closest_point_on_triangle(query, v0, v1, v2)
        d = cp - query
        dist_sq = wp.dot(d, d)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = cp

    return best_point


@wp.func
def closest_point_on_shape(
    geom: Any,
    query: wp.vec3,
    data_provider: Any,
) -> tuple[bool, wp.vec3]:
    """Dispatch to the per-primitive closest-point helper.

    The dispatcher covers only exact helpers. Convex hulls use an exact
    closest point over their triangle surface; this is O(F), so callers
    should use it only where that cost is justified.

    Returns:
        Tuple ``(handled, point)``. ``handled`` is ``True`` when a closest
        point was computed; ``False`` means the caller should fall back to
        its previous seeding behavior. ``point`` is in the shape's local
        frame.
    """
    handled = False
    point = wp.vec3(0.0, 0.0, 0.0)

    _ = data_provider

    if geom.shape_type == int(GeoType.SPHERE):
        point = closest_point_sphere(query, geom.scale[0])
        handled = True
    elif geom.shape_type == int(GeoType.BOX):
        point = closest_point_box_local(geom.scale, query)
        handled = True
    elif geom.shape_type == int(GeoType.CAPSULE):
        point = closest_point_capsule_z(query, geom.scale[0], geom.scale[1])
        handled = True
    elif geom.shape_type == int(GeoType.CYLINDER):
        point = closest_point_cylinder_z(query, geom.scale[0], geom.scale[1])
        handled = True
    elif geom.shape_type == int(GeoType.PLANE):
        if geom.scale[0] <= 0.0 or geom.scale[1] <= 0.0:
            point = closest_point_plane_infinite(query)
            handled = True
    elif geom.shape_type == int(GeoType.CONVEX_MESH):
        point = closest_point_convex_mesh_face(geom, query, data_provider)
        handled = True
    elif geom.shape_type == int(GeoTypeEx.TRIANGLE) or geom.shape_type == int(GeoTypeEx.TRIANGLE_PRISM):
        tri_a = wp.vec3(0.0, 0.0, 0.0)
        tri_b = geom.scale
        tri_c = geom.auxiliary
        point = closest_point_on_triangle(query, tri_a, tri_b, tri_c)
        handled = True

    return handled, point
