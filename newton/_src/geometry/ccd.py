# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Swept proximity queries for continuous collision detection (CCD) of rigid bodies.

Solvers that enforce speculative contacts use :func:`shapes_meet_within` to confirm, before a
contact activates, that its two shapes actually touch within the step. The query uses
conservative advancement (Mirtich 1996) on GJK distances over the bodies' current motion, against
convex shapes, planes, and the triangles of static meshes and heightfields.
"""

from __future__ import annotations

import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_core import aabb_to_unscaled, get_triangle_shape_from_mesh
from .simplex_solver import create_solve_closest_distance
from .support_function import GenericShapeData, SupportMapDataProvider, pack_mesh_ptr, support_map
from .types import GeoType

CCD_TOLERANCE_FRACTION = wp.constant(0.05)
"""Shapes closer than this fraction of the moving shape's smallest half-extent count as touching."""

CCD_MAX_ITERATIONS = wp.constant(20)

_solve_gjk = create_solve_closest_distance(support_map)

# Sphere and capsule cores are swept as points and segments; their radius is added to the gap.
_CORE_RADIUS = wp.constant(1.0e-4)


@wp.func
def _is_convex(shape_type: int) -> bool:
    return (
        shape_type == GeoType.SPHERE
        or shape_type == GeoType.CAPSULE
        or shape_type == GeoType.ELLIPSOID
        or shape_type == GeoType.CYLINDER
        or shape_type == GeoType.BOX
        or shape_type == GeoType.CONE
        or shape_type == GeoType.CONVEX_MESH
    )


@wp.func
def _pose_at(q0: wp.transform, q1: wp.transform, com: wp.vec3, t: float) -> wp.transform:
    """Interpolate a body pose about its center of mass (lerp center, slerp rotation)."""
    r0 = wp.transform_get_rotation(q0)
    r1 = wp.transform_get_rotation(q1)
    if wp.dot(r0, r1) < 0.0:
        r1 = -r1
    r = wp.quat_slerp(r0, r1, t)
    c = wp.lerp(wp.transform_point(q0, com), wp.transform_point(q1, com), t)
    return wp.transform(c - wp.quat_rotate(r, com), r)


@wp.func
def _rotation_angle(q0: wp.transform, q1: wp.transform) -> float:
    d = wp.abs(wp.dot(wp.transform_get_rotation(q0), wp.transform_get_rotation(q1)))
    return 2.0 * wp.acos(wp.min(d, 1.0))


@wp.func
def _gap(
    geom_a: GenericShapeData,
    xform_a: wp.transform,
    geom_b: GenericShapeData,
    xform_b: wp.transform,
    infinite_plane_b: bool,
    offset: float,
) -> tuple[float, wp.vec3]:
    """Return the surface gap [m] between two shapes and the unit normal pointing from A to B."""
    provider = SupportMapDataProvider()
    rot_a = wp.transform_get_rotation(xform_a)
    if infinite_plane_b:
        n = wp.quat_rotate(wp.transform_get_rotation(xform_b), wp.vec3(0.0, 0.0, 1.0))
        p = wp.transform_point(xform_a, support_map(geom_a, wp.quat_rotate_inv(rot_a, -n), provider))
        return wp.dot(n, p - wp.transform_get_translation(xform_b)) - offset, -n
    rel_rot = wp.quat_inverse(rot_a) * wp.transform_get_rotation(xform_b)
    rel_pos = wp.quat_rotate_inv(rot_a, wp.transform_get_translation(xform_b) - wp.transform_get_translation(xform_a))
    _separated, _pa, _pb, n, distance = wp.static(_solve_gjk.core)(geom_a, geom_b, rel_rot, rel_pos, 0.0, provider)
    return distance - offset, wp.quat_rotate(rot_a, n)


@wp.func
def _predicted_pose(q: wp.transform, qd: wp.spatial_vector, com: wp.vec3, dt: float) -> wp.transform:
    """Pose after moving for ``dt`` with constant linear velocity at the COM and angular velocity."""
    rot = wp.transform_get_rotation(q)
    w = wp.spatial_bottom(qd)
    angle = wp.length(w) * dt
    if angle > 0.0:
        rot = wp.quat_from_axis_angle(wp.normalize(w), angle) * rot
    c = wp.transform_point(q, com) + wp.spatial_top(qd) * dt
    return wp.transform(c - wp.quat_rotate(rot, com), rot)


@wp.func
def _model_shape_geometry(
    shape: int,
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    shape_source: wp.array[wp.uint64],
    center: wp.vec3,
) -> tuple[GenericShapeData, wp.transform, float]:
    """Return support data, body-relative transform and surface offset [m] of a shape from model arrays."""
    geom = GenericShapeData()
    geom.shape_type = shape_type[shape]
    geom.scale = shape_scale[shape]
    geom.auxiliary = wp.vec3(0.0)
    geom.center = wp.vec3(0.0)
    geom.shape_index = shape
    if geom.shape_type == GeoType.PLANE:
        geom.scale = wp.vec3(0.5 * geom.scale[0], 0.5 * geom.scale[1], 0.0)
    elif geom.shape_type == GeoType.CONVEX_MESH:
        geom.auxiliary = pack_mesh_ptr(shape_source[shape])
        geom.center = center
    offset = shape_margin[shape]
    if geom.shape_type == GeoType.SPHERE or geom.shape_type == GeoType.CAPSULE:
        offset += geom.scale[0]
        geom.scale[0] = _CORE_RADIUS
    return geom, shape_transform[shape], offset


@wp.func
def _sweep_meets(
    geom_a: GenericShapeData,
    xform_a: wp.transform,
    qa0: wp.transform,
    qa1: wp.transform,
    com_a: wp.vec3,
    radius_a: float,
    geom_b: GenericShapeData,
    xform_b: wp.transform,
    qb0: wp.transform,
    qb1: wp.transform,
    com_b: wp.vec3,
    radius_b: float,
    infinite_plane_b: bool,
    offset: float,
    target: float,
) -> bool:
    """Whether two shapes on moving bodies come within ``target`` [m] of each other.

    Conservative advancement on the relative motion: the gap closes at most at the relative
    translation along the normal plus each body's rotation angle times its bounding radius.
    Undecided searches count as meeting.
    """
    translation = (wp.transform_point(qa1, com_a) - wp.transform_point(qa0, com_a)) - (
        wp.transform_point(qb1, com_b) - wp.transform_point(qb0, com_b)
    )
    rotation = _rotation_angle(qa0, qa1) * radius_a + _rotation_angle(qb0, qb1) * radius_b
    t = float(0.0)
    for _ in range(CCD_MAX_ITERATIONS):
        gap, normal = _gap(
            geom_a,
            _pose_at(qa0, qa1, com_a, t) * xform_a,
            geom_b,
            _pose_at(qb0, qb1, com_b, t) * xform_b,
            infinite_plane_b,
            offset,
        )
        if gap <= target:
            return True
        approach = wp.max(wp.dot(normal, translation), 0.0) + rotation
        if gap - target >= (1.0 - t) * approach:
            return False
        t += (gap - target) / approach
    return True


@wp.func
def shapes_meet_within(
    shape_a: int,
    shape_b: int,
    dt: float,
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    shape_source: wp.array[wp.uint64],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    shape_heightfield_index: wp.array[int],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
) -> bool:
    """Whether two shapes touch within ``dt`` [s] when their bodies keep their current velocities.

    A convex shape is swept against another convex shape, a plane, or the triangles of a static
    mesh or heightfield under its swept bounds. Other pairs and undecided searches count as
    touching, so a contact is never dropped wrongly.
    """
    type_a = shape_type[shape_a]
    if type_a == GeoType.PLANE or type_a == GeoType.MESH or type_a == GeoType.HFIELD:
        tmp = shape_a
        shape_a = shape_b
        shape_b = tmp
        type_a = shape_type[shape_a]
    type_b = shape_type[shape_b]
    if not _is_convex(type_a):
        return True
    surface = type_b == GeoType.MESH or type_b == GeoType.HFIELD
    if not (_is_convex(type_b) or type_b == GeoType.PLANE or (surface and shape_body[shape_b] < 0)):
        return True

    center_a = 0.5 * (shape_aabb_lower[shape_a] + shape_aabb_upper[shape_a])
    half_a = 0.5 * (shape_aabb_upper[shape_a] - shape_aabb_lower[shape_a])
    geom_a, xform_a, offset_a = _model_shape_geometry(
        shape_a, shape_type, shape_transform, shape_scale, shape_margin, shape_source, center_a
    )
    body_a = shape_body[shape_a]
    qa0 = wp.transform_identity()
    com_a = wp.vec3(0.0)
    if body_a >= 0:
        qa0 = body_q[body_a]
        com_a = body_com[body_a]
    qa1 = qa0
    if body_a >= 0:
        qa1 = _predicted_pose(qa0, body_qd[body_a], com_a, dt)
    radius_a = wp.length(wp.transform_point(xform_a, center_a) - com_a) + wp.length(half_a) + shape_margin[shape_a]
    target = CCD_TOLERANCE_FRACTION * wp.min(half_a[0], wp.min(half_a[1], half_a[2]))
    identity = wp.transform_identity()

    if surface:
        # Triangles of a static surface under the bounds of the whole sweep, in its frame.
        xform_b = shape_transform[shape_b]
        inv_b = wp.transform_inverse(xform_b)
        c0 = wp.transform_point(inv_b, wp.transform_point(qa0, com_a))
        c1 = wp.transform_point(inv_b, wp.transform_point(qa1, com_a))
        reach = wp.vec3(radius_a + shape_margin[shape_b] + target)
        lower = wp.min(c0, c1) - reach
        upper = wp.max(c0, c1) + reach
        offset = offset_a + shape_margin[shape_b]
        if type_b == GeoType.MESH:
            mesh_scale = shape_scale[shape_b]
            lower, upper, _inv_scale = aabb_to_unscaled(lower, upper, mesh_scale)
            query = wp.mesh_query_aabb(shape_source[shape_b], lower, upper)
            tri = int(0)
            while wp.mesh_query_aabb_next(query, tri):
                geom_tri, v0 = get_triangle_shape_from_mesh(shape_source[shape_b], mesh_scale, xform_b, tri)
                if _sweep_meets(
                    geom_a,
                    xform_a,
                    qa0,
                    qa1,
                    com_a,
                    radius_a,
                    geom_tri,
                    wp.transform(v0, wp.quat_identity()),
                    identity,
                    identity,
                    wp.vec3(0.0),
                    0.0,
                    False,
                    offset,
                    target,
                ):
                    return True
            return False
        hfd = heightfield_data[shape_heightfield_index[shape_b]]
        rot_b = wp.transform_get_rotation(xform_b)
        dx = 2.0 * hfd.hx / float(hfd.ncol - 1)
        dy = 2.0 * hfd.hy / float(hfd.nrow - 1)
        col_min = wp.max(int(wp.floor((lower[0] + hfd.hx) / dx)), 0)
        col_max = wp.min(int(wp.floor((upper[0] + hfd.hx) / dx)), hfd.ncol - 2)
        row_min = wp.max(int(wp.floor((lower[1] + hfd.hy) / dy)), 0)
        row_max = wp.min(int(wp.floor((upper[1] + hfd.hy) / dy)), hfd.nrow - 2)
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                for sub in range(2):
                    tri = (row * (hfd.ncol - 1) + col) * 2 + sub
                    geom_tri, v0 = get_triangle_shape_from_heightfield(hfd, heightfield_elevations, xform_b, tri)
                    if _sweep_meets(
                        geom_a,
                        xform_a,
                        qa0,
                        qa1,
                        com_a,
                        radius_a,
                        geom_tri,
                        wp.transform(v0, rot_b),
                        identity,
                        identity,
                        wp.vec3(0.0),
                        0.0,
                        False,
                        offset,
                        target,
                    ):
                        return True
        return False

    center_b = 0.5 * (shape_aabb_lower[shape_b] + shape_aabb_upper[shape_b])
    half_b = 0.5 * (shape_aabb_upper[shape_b] - shape_aabb_lower[shape_b])
    geom_b, xform_b, offset_b = _model_shape_geometry(
        shape_b, shape_type, shape_transform, shape_scale, shape_margin, shape_source, center_b
    )
    infinite_plane_b = type_b == GeoType.PLANE and geom_b.scale[0] == 0.0 and geom_b.scale[1] == 0.0
    body_b = shape_body[shape_b]
    qb0 = wp.transform_identity()
    com_b = wp.vec3(0.0)
    if body_b >= 0:
        qb0 = body_q[body_b]
        com_b = body_com[body_b]
    qb1 = qb0
    if body_b >= 0:
        qb1 = _predicted_pose(qb0, body_qd[body_b], com_b, dt)
    radius_b = wp.length(wp.transform_point(xform_b, center_b) - com_b) + wp.length(half_b) + shape_margin[shape_b]
    return _sweep_meets(
        geom_a,
        xform_a,
        qa0,
        qa1,
        com_a,
        radius_a,
        geom_b,
        xform_b,
        qb0,
        qb1,
        com_b,
        radius_b,
        infinite_plane_b,
        offset_a + offset_b,
        target,
    )
