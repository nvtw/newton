# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Continuous collision detection (CCD) for fast rigid bodies.

After the solver step, every fast dynamic body is swept from its start pose to its solved pose
against static shapes, meshes, and heightfields, and moved back to its earliest time of impact.
Velocities are kept except for the normal velocity towards the hit surface, so solvers without
speculative contacts do not drive the body back into it.

The time of impact uses conservative advancement (Mirtich 1996) on GJK distances. A shape that
touches an obstacle at the start of the step is swept with a small core sphere instead, so
resting and sliding contacts never stall the body.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..sim.enums import BodyFlags, JointType
from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_core import aabb_to_unscaled, get_triangle_shape_from_mesh
from .simplex_solver import create_solve_closest_distance
from .support_function import GenericShapeData, SupportMapDataProvider, extract_shape_data, support_map
from .types import GeoType

CCD_SAFETY_FACTOR = wp.constant(0.5)
"""A shape is fast when it moves more than this fraction of its smallest half-extent in a step."""

CCD_CORE_FRACTION = wp.constant(0.5)
"""Radius of the fallback core sphere relative to the shape's smallest half-extent."""

CCD_TOLERANCE_FRACTION = wp.constant(0.05)
"""The time-of-impact search stops below this gap, as a fraction of the shape's smallest half-extent."""

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
def _shape_geometry(
    shape: int,
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    geom_data: wp.array[wp.vec4],
    shape_source: wp.array[wp.uint64],
    shape_center: wp.vec3,
) -> tuple[GenericShapeData, wp.transform, float]:
    """Return support data, body-relative transform and surface offset [m] of a shape."""
    pos, rot, geom, _scale, margin = extract_shape_data(shape, shape_transform, shape_type, geom_data, shape_source)
    if geom.shape_type == GeoType.CONVEX_MESH:
        geom.center = shape_center
    offset = margin
    if geom.shape_type == GeoType.SPHERE or geom.shape_type == GeoType.CAPSULE:
        offset += geom.scale[0]
        geom.scale[0] = _CORE_RADIUS
    return geom, wp.transform(pos, rot), offset


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
def _time_of_impact(
    geom_a: GenericShapeData,
    shape_xform_a: wp.transform,
    body_q0: wp.transform,
    body_q1: wp.transform,
    com: wp.vec3,
    radius: float,
    geom_b: GenericShapeData,
    xform_b: wp.transform,
    infinite_plane_b: bool,
    offset: float,
    tolerance: float,
) -> tuple[float, wp.vec3]:
    """Conservatively advance moving shape A towards static shape B over t in [0, 1].

    Returns the impact time and the A-to-B normal there; ``t = 1`` means no impact and ``t = 0``
    means the shapes are already within ``tolerance`` at the start.
    """
    translation = wp.transform_point(body_q1, com) - wp.transform_point(body_q0, com)
    angular_bound = _rotation_angle(body_q0, body_q1) * radius
    t = float(0.0)
    normal = wp.vec3(0.0)
    for _ in range(CCD_MAX_ITERATIONS):
        xform_a = _pose_at(body_q0, body_q1, com, t) * shape_xform_a
        gap, normal = _gap(geom_a, xform_a, geom_b, xform_b, infinite_plane_b, offset)
        if gap < tolerance:
            return t, normal
        # Upper bound on the approach speed of any point of A along the normal, per unit t.
        approach = wp.max(wp.dot(normal, translation), 0.0) + angular_bound
        if gap - tolerance >= (1.0 - t) * approach:
            return 1.0, normal
        # Stop half a tolerance short so the final query is still separated and has a normal.
        t += (gap - 0.5 * tolerance) / approach
    return t, normal


@wp.func
def _sweep(
    geom_a: GenericShapeData,
    shape_xform_a: wp.transform,
    body_q0: wp.transform,
    body_q1: wp.transform,
    com: wp.vec3,
    radius: float,
    offset_a: float,
    center_body: wp.vec3,
    core_radius: float,
    geom_b: GenericShapeData,
    xform_b: wp.transform,
    infinite_plane_b: bool,
    offset_b: float,
    tolerance: float,
) -> tuple[float, wp.vec3]:
    """Sweep shape A against static shape B; return impact time (1 = none) and A-to-B normal."""
    t, normal = _time_of_impact(
        geom_a,
        shape_xform_a,
        body_q0,
        body_q1,
        com,
        radius,
        geom_b,
        xform_b,
        infinite_plane_b,
        offset_a + offset_b,
        tolerance,
    )
    if t > 0.0:
        return t, normal
    # Already touching: only keep the core of the shape from tunneling.
    core = GenericShapeData()
    core.shape_type = int(GeoType.SPHERE)
    core.scale = wp.vec3(_CORE_RADIUS)
    t, normal = _time_of_impact(
        core,
        wp.transform(center_body, wp.quat_identity()),
        body_q0,
        body_q1,
        com,
        wp.length(center_body - com) + core_radius,
        geom_b,
        xform_b,
        infinite_plane_b,
        core_radius + offset_b,
        tolerance,
    )
    if t == 0.0:
        return 1.0, normal
    return t, normal


@wp.func
def _sweep_triangle(
    geom_a: GenericShapeData,
    shape_xform_a: wp.transform,
    body_q0: wp.transform,
    body_q1: wp.transform,
    com: wp.vec3,
    radius: float,
    offset_a: float,
    center_body: wp.vec3,
    core_radius: float,
    geom_tri: GenericShapeData,
    xform_tri: wp.transform,
    offset_b: float,
    tolerance: float,
) -> tuple[float, wp.vec3]:
    """Sweep shape A against a one-sided static triangle.

    The triangle is skipped if the shape center starts behind it, or approaches it by less than
    the core radius and ends in front of it, so faces the shape slides along never stop it.
    """
    n = wp.normalize(wp.quat_rotate(wp.transform_get_rotation(xform_tri), wp.cross(geom_tri.scale, geom_tri.auxiliary)))
    v0 = wp.transform_get_translation(xform_tri)
    offset0 = wp.dot(n, wp.transform_point(body_q0, center_body) - v0)
    offset1 = wp.dot(n, wp.transform_point(body_q1, center_body) - v0)
    if offset0 < 0.0 or (offset0 - offset1 < core_radius and offset1 > core_radius):
        return 1.0, n
    return _sweep(
        geom_a,
        shape_xform_a,
        body_q0,
        body_q1,
        com,
        radius,
        offset_a,
        center_body,
        core_radius,
        geom_tri,
        xform_tri,
        False,
        offset_b,
        tolerance,
    )


@wp.kernel(enable_backward=False)
def ccd_pair_impact_kernel(
    pairs: wp.array[wp.vec2i],
    pair_count: wp.array[int],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    geom_data: wp.array[wp.vec4],
    shape_source: wp.array[wp.uint64],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    shape_heightfield_index: wp.array[int],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_ccd_articulation: wp.array[int],
    body_q_start: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    # outputs
    pair_impact_time: wp.array[float],
    pair_normal: wp.array[wp.vec3],
    body_impact_time: wp.array[float],
):
    tid = wp.tid()
    pair_impact_time[tid] = 1.0
    if tid >= pair_count[0]:
        return

    # Sweep eligible dynamic shapes against static shapes only.
    shape_a = pairs[tid][0]
    shape_b = pairs[tid][1]
    if shape_body[shape_a] < 0:
        shape_a = pairs[tid][1]
        shape_b = pairs[tid][0]
    body = shape_body[shape_a]
    if body < 0 or shape_body[shape_b] >= 0 or body_ccd_articulation[body] < 0:
        return
    type_b = shape_type[shape_b]
    if not _is_convex(shape_type[shape_a]) or not (
        _is_convex(type_b) or type_b == GeoType.PLANE or type_b == GeoType.MESH or type_b == GeoType.HFIELD
    ):
        return

    q0 = body_q_start[body]
    q1 = body_q[body]
    com = body_com[body]
    center_a = 0.5 * (shape_aabb_lower[shape_a] + shape_aabb_upper[shape_a])
    half_extent = 0.5 * (shape_aabb_upper[shape_a] - shape_aabb_lower[shape_a])
    min_extent = wp.min(half_extent[0], wp.min(half_extent[1], half_extent[2]))

    geom_a, shape_xform_a, offset_a = _shape_geometry(
        shape_a, shape_type, shape_transform, geom_data, shape_source, center_a
    )
    center_body = wp.transform_point(shape_xform_a, center_a)
    radius = wp.length(center_body - com) + wp.length(half_extent) + geom_data[shape_a][3]

    # Skip shapes that cannot tunnel this step.
    motion = wp.length(wp.transform_point(q1, center_body) - wp.transform_point(q0, center_body))
    if motion + _rotation_angle(q0, q1) * radius <= CCD_SAFETY_FACTOR * min_extent:
        return

    core_radius = CCD_CORE_FRACTION * min_extent
    tolerance = CCD_TOLERANCE_FRACTION * min_extent
    center_b = 0.5 * (shape_aabb_lower[shape_b] + shape_aabb_upper[shape_b])
    geom_b, xform_b, offset_b = _shape_geometry(shape_b, shape_type, shape_transform, geom_data, shape_source, center_b)

    # Bounds of the whole sweep in the static shape's frame, for the mesh and heightfield queries.
    inv_b = wp.transform_inverse(xform_b)
    com0 = wp.transform_point(inv_b, wp.transform_point(q0, com))
    com1 = wp.transform_point(inv_b, wp.transform_point(q1, com))
    reach = wp.vec3(radius + offset_b + tolerance)
    lower = wp.min(com0, com1) - reach
    upper = wp.max(com0, com1) + reach

    t = float(1.0)
    normal = wp.vec3(0.0)
    if type_b == GeoType.MESH:
        mesh_scale = geom_b.scale
        lower, upper, _inv_scale = aabb_to_unscaled(lower, upper, mesh_scale)
        query = wp.mesh_query_aabb(shape_source[shape_b], lower, upper)
        tri = int(0)
        while wp.mesh_query_aabb_next(query, tri):
            geom_tri, v0 = get_triangle_shape_from_mesh(shape_source[shape_b], mesh_scale, xform_b, tri)
            t_tri, n_tri = _sweep_triangle(
                geom_a,
                shape_xform_a,
                q0,
                q1,
                com,
                radius,
                offset_a,
                center_body,
                core_radius,
                geom_tri,
                wp.transform(v0, wp.quat_identity()),
                offset_b,
                tolerance,
            )
            if t_tri < t:
                t = t_tri
                normal = n_tri
    elif type_b == GeoType.HFIELD:
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
                    t_tri, n_tri = _sweep_triangle(
                        geom_a,
                        shape_xform_a,
                        q0,
                        q1,
                        com,
                        radius,
                        offset_a,
                        center_body,
                        core_radius,
                        geom_tri,
                        wp.transform(v0, rot_b),
                        offset_b,
                        tolerance,
                    )
                    if t_tri < t:
                        t = t_tri
                        normal = n_tri
    else:
        infinite_plane_b = type_b == GeoType.PLANE and geom_b.scale[0] == 0.0 and geom_b.scale[1] == 0.0
        t, normal = _sweep(
            geom_a,
            shape_xform_a,
            q0,
            q1,
            com,
            radius,
            offset_a,
            center_body,
            core_radius,
            geom_b,
            xform_b,
            infinite_plane_b,
            offset_b,
            tolerance,
        )
    if t < 1.0:
        pair_impact_time[tid] = t
        pair_normal[tid] = normal
        wp.atomic_min(body_impact_time, body, t)


@wp.kernel(enable_backward=False)
def ccd_pick_hit_kernel(
    pairs: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    pair_impact_time: wp.array[float],
    body_impact_time: wp.array[float],
    # outputs
    body_hit_pair: wp.array[int],
):
    tid = wp.tid()
    t = pair_impact_time[tid]
    if t >= 1.0:
        return
    body = wp.max(shape_body[pairs[tid][0]], shape_body[pairs[tid][1]])
    # Lowest pair index among equal impact times keeps the result deterministic.
    if t == body_impact_time[body]:
        wp.atomic_min(body_hit_pair, body, tid)


@wp.kernel(enable_backward=False)
def ccd_apply_kernel(
    body_com: wp.array[wp.vec3],
    body_ccd_articulation: wp.array[int],
    body_impact_time: wp.array[float],
    body_hit_pair: wp.array[int],
    pair_normal: wp.array[wp.vec3],
    body_q_start: wp.array[wp.transform],
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    articulation_mask: wp.array[bool],
):
    body = wp.tid()
    t = body_impact_time[body]
    if t >= 1.0:
        return
    body_q[body] = _pose_at(body_q_start[body], body_q[body], body_com[body], t)

    # Remove the velocity towards the hit surface; tangential motion and spin are kept.
    normal = pair_normal[body_hit_pair[body]]
    v = wp.spatial_top(body_qd[body])
    v -= wp.max(wp.dot(v, normal), 0.0) * normal
    body_qd[body] = wp.spatial_vector(v, wp.spatial_bottom(body_qd[body]))
    articulation_mask[body_ccd_articulation[body]] = True


def ccd_body_articulations(model) -> np.ndarray:
    """Return each body's articulation if it is a free-floating dynamic body, else -1.

    Only single-body articulations with a free joint to the world are swept: moving one link of a
    multi-body articulation would break its joints.
    """
    result = np.full(model.body_count, -1, dtype=np.int32)
    if model.joint_count == 0:
        return result
    joint_type = model.joint_type.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_articulation = model.joint_articulation.numpy()
    body_flags = model.body_flags.numpy()
    free = (joint_type == int(JointType.FREE)) & (joint_parent == -1) & (joint_articulation >= 0)
    result[joint_child[free]] = joint_articulation[free]
    result[joint_parent[joint_parent >= 0]] = -1
    result[(body_flags & int(BodyFlags.DYNAMIC)) == 0] = -1
    return result
