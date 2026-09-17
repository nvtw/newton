# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common unilateral rows for maximal-coordinate D6 axes."""

import warp as wp

from newton._src.sim.articulation import (
    invert_2d_rotational_dofs,
    invert_3d_rotational_dofs,
    transform_2d_rotational_axes,
    transform_3d_rotational_axes,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.d6_joint_data import D6_AXIS_COUNT, D6JointData
from newton._src.solvers.phoenx.helpers.math_helpers import extract_rotation_angle
from newton._src.solvers.phoenx.solver_config import PHOENX_FRICTION_SLIP_VELOCITY


@wp.func
def _d6_body_origin_transform(bodies: BodyContainer, body: wp.int32) -> wp.transform:
    if body <= wp.int32(0):
        return wp.transform_identity()
    orientation = bodies.orientation[body]
    origin = bodies.position[body] - wp.quat_rotate(orientation, bodies.body_com[body])
    return wp.transform(origin, orientation)


@wp.func
def _d6_row_velocity(
    wrench0: wp.spatial_vector, wrench1: wp.spatial_vector, v0: wp.vec3f, w0: wp.vec3f, v1: wp.vec3f, w1: wp.vec3f
) -> wp.float32:
    return (
        wp.dot(wp.spatial_top(wrench0), v0)
        + wp.dot(wp.spatial_bottom(wrench0), w0)
        + wp.dot(wp.spatial_top(wrench1), v1)
        + wp.dot(wp.spatial_bottom(wrench1), w1)
    )


@wp.func
def _d6_apply_row_impulse(
    wrench0: wp.spatial_vector,
    wrench1: wp.spatial_vector,
    impulse: wp.float32,
    inverse_mass0: wp.float32,
    inverse_mass1: wp.float32,
    inverse_inertia0: wp.mat33f,
    inverse_inertia1: wp.mat33f,
    v0: wp.vec3f,
    w0: wp.vec3f,
    v1: wp.vec3f,
    w1: wp.vec3f,
):
    v0 += inverse_mass0 * impulse * wp.spatial_top(wrench0)
    w0 += inverse_inertia0 @ (impulse * wp.spatial_bottom(wrench0))
    v1 += inverse_mass1 * impulse * wp.spatial_top(wrench1)
    w1 += inverse_inertia1 @ (impulse * wp.spatial_bottom(wrench1))
    return v0, w0, v1, w1


@wp.func
def _d6_reciprocal_axes_2(axis0: wp.vec3f, axis1: wp.vec3f):
    """Return coordinate gradients dual to two transported motion axes."""
    coupling = wp.dot(axis0, axis1)
    determinant = wp.float32(1.0) - coupling * coupling
    reciprocal0 = axis0
    reciprocal1 = axis1
    if determinant > wp.float32(1.0e-4):
        reciprocal0 = (axis0 - coupling * axis1) / determinant
        reciprocal1 = (axis1 - coupling * axis0) / determinant
    return reciprocal0, reciprocal1


@wp.func
def _d6_reciprocal_axes_3(axis0: wp.vec3f, axis1: wp.vec3f, axis2: wp.vec3f):
    """Return coordinate gradients dual to three transported motion axes."""
    cross12 = wp.cross(axis1, axis2)
    determinant = wp.dot(axis0, cross12)
    reciprocal0 = axis0
    reciprocal1 = axis1
    reciprocal2 = axis2
    if wp.abs(determinant) > wp.float32(1.0e-4):
        reciprocal0 = cross12 / determinant
        reciprocal1 = wp.cross(axis2, axis0) / determinant
        reciprocal2 = wp.cross(axis0, axis1) / determinant
    return reciprocal0, reciprocal1, reciprocal2


@wp.func
def prepare_d6_inequalities(
    data: D6JointData,
    cid: wp.int32,
    bodies: BodyContainer,
    body0: wp.int32,
    body1: wp.int32,
    inverse_mass0: wp.float32,
    inverse_mass1: wp.float32,
    inverse_inertia0: wp.mat33f,
    inverse_inertia1: wp.mat33f,
    v0: wp.vec3f,
    w0: wp.vec3f,
    v1: wp.vec3f,
    w1: wp.vec3f,
):
    """Refresh D6 row geometry and apply its saved warm start."""
    count = data.row_count[cid]
    x_w0 = _d6_body_origin_transform(bodies, body0) * data.joint_x_p[cid]
    x_w1 = _d6_body_origin_transform(bodies, body1) * data.joint_x_c[cid]
    point0 = wp.transform_get_translation(x_w0)
    point1 = wp.transform_get_translation(x_w1)
    orientation0 = wp.transform_get_rotation(x_w0)
    orientation1 = wp.transform_get_rotation(x_w1)
    separation = point1 - point0
    linear_count = data.linear_count[cid]
    angular_count = data.angular_count[cid]
    angular_coordinates = wp.vec3f(0.0, 0.0, 0.0)
    angular_direction0 = wp.vec3f(1.0, 0.0, 0.0)
    angular_direction1 = wp.vec3f(0.0, 1.0, 0.0)
    angular_direction2 = wp.vec3f(0.0, 0.0, 1.0)
    if angular_count == wp.int32(2):
        angular_start = linear_count
        coordinates_two, _rates_two = invert_2d_rotational_dofs(
            data.axis[cid, angular_start],
            data.axis[cid, angular_start + wp.int32(1)],
            orientation0,
            orientation1,
            wp.vec3(),
        )
        direction0_two, direction1_two = transform_2d_rotational_axes(
            data.axis[cid, angular_start],
            data.axis[cid, angular_start + wp.int32(1)],
            coordinates_two[0],
        )
        angular_coordinates = wp.vec3f(coordinates_two[0], coordinates_two[1], 0.0)
        motion0 = wp.vec3f(direction0_two[0], direction0_two[1], direction0_two[2])
        motion1 = wp.vec3f(direction1_two[0], direction1_two[1], direction1_two[2])
        angular_direction0, angular_direction1 = _d6_reciprocal_axes_2(motion0, motion1)
    elif angular_count == wp.int32(3):
        angular_start = linear_count
        coordinates_three, _rates_three = invert_3d_rotational_dofs(
            data.axis[cid, angular_start],
            data.axis[cid, angular_start + wp.int32(1)],
            data.axis[cid, angular_start + wp.int32(2)],
            orientation0,
            orientation1,
            wp.vec3(),
        )
        direction0_three, direction1_three, direction2_three = transform_3d_rotational_axes(
            data.axis[cid, angular_start],
            data.axis[cid, angular_start + wp.int32(1)],
            data.axis[cid, angular_start + wp.int32(2)],
            coordinates_three[0],
            coordinates_three[1],
        )
        angular_coordinates = wp.vec3f(coordinates_three[0], coordinates_three[1], coordinates_three[2])
        motion0 = wp.vec3f(direction0_three[0], direction0_three[1], direction0_three[2])
        motion1 = wp.vec3f(direction1_three[0], direction1_three[1], direction1_three[2])
        motion2 = wp.vec3f(direction2_three[0], direction2_three[1], direction2_three[2])
        angular_direction0, angular_direction1, angular_direction2 = _d6_reciprocal_axes_3(motion0, motion1, motion2)

    parent_lever = wp.vec3f(0.0, 0.0, 0.0)
    child_lever = wp.vec3f(0.0, 0.0, 0.0)
    if body0 > wp.int32(0):
        parent_lever = point1 - bodies.position[body0]
    if body1 > wp.int32(0):
        child_lever = point1 - bodies.position[body1]

    for row in range(D6_AXIS_COUNT):
        if wp.int32(row) < count:
            axis_index = data.row_axis[cid, row]
            wrench0 = wp.spatial_vector()
            wrench1 = wp.spatial_vector()
            coordinate = wp.float32(0.0)
            if axis_index < linear_count:
                direction = wp.normalize(wp.quat_rotate(orientation0, data.axis[cid, axis_index]))
                force0 = -direction
                force1 = direction
                wrench0 = wp.spatial_vector(force0, wp.cross(parent_lever, force0))
                wrench1 = wp.spatial_vector(force1, wp.cross(child_lever, force1))
                coordinate = wp.dot(data.axis[cid, axis_index], wp.quat_rotate_inv(orientation0, separation))
            else:
                angular_axis = axis_index - linear_count
                direction_local = angular_direction0
                if angular_axis == wp.int32(1):
                    direction_local = angular_direction1
                elif angular_axis == wp.int32(2):
                    direction_local = angular_direction2
                direction = wp.quat_rotate(orientation0, direction_local)
                if angular_count == wp.int32(1):
                    coordinate = extract_rotation_angle(orientation1 * wp.quat_inverse(orientation0), direction)
                else:
                    coordinate = angular_coordinates[angular_axis]
                wrench0 = wp.spatial_vector(wp.vec3f(0.0), -direction)
                wrench1 = wp.spatial_vector(wp.vec3f(0.0), direction)
            data.wrench0[cid, row] = wrench0
            data.wrench1[cid, row] = wrench1
            data.coordinate[cid, row] = coordinate
            response0 = wp.spatial_vector(
                inverse_mass0 * wp.spatial_top(wrench0),
                inverse_inertia0 @ wp.spatial_bottom(wrench0),
            )
            response1 = wp.spatial_vector(
                inverse_mass1 * wp.spatial_top(wrench1),
                inverse_inertia1 @ wp.spatial_bottom(wrench1),
            )
            effective_mass_inverse = (
                wp.dot(wp.spatial_top(wrench0), wp.spatial_top(response0))
                + wp.dot(wp.spatial_bottom(wrench0), wp.spatial_bottom(response0))
                + wp.dot(wp.spatial_top(wrench1), wp.spatial_top(response1))
                + wp.dot(wp.spatial_bottom(wrench1), wp.spatial_bottom(response1))
            )
            data.effective_mass_inverse[cid, row] = effective_mass_inverse
            warm_start = data.lower_impulse[cid, row] + data.upper_impulse[cid, row] + data.friction_impulse[cid, row]
            v0, w0, v1, w1 = _d6_apply_row_impulse(
                wrench0,
                wrench1,
                warm_start,
                inverse_mass0,
                inverse_mass1,
                inverse_inertia0,
                inverse_inertia1,
                v0,
                w0,
                v1,
                w1,
            )
    return v0, w0, v1, w1


@wp.func
def iterate_d6_inequalities(
    data: D6JointData,
    cid: wp.int32,
    inverse_mass0: wp.float32,
    inverse_mass1: wp.float32,
    inverse_inertia0: wp.mat33f,
    inverse_inertia1: wp.mat33f,
    v0: wp.vec3f,
    w0: wp.vec3f,
    v1: wp.vec3f,
    w1: wp.vec3f,
    idt: wp.float32,
    sor_boost: wp.float32,
):
    """Solve predictive hard bounds, speed caps, and Coulomb friction."""
    count = data.row_count[cid]
    for row in range(D6_AXIS_COUNT):
        if wp.int32(row) < count:
            wrench0 = data.wrench0[cid, row]
            wrench1 = data.wrench1[cid, row]
            effective_mass_inverse = data.effective_mass_inverse[cid, row]
            if effective_mass_inverse > wp.float32(0.0):
                coordinate = data.coordinate[cid, row]
                lower = data.lower[cid, row]
                upper = data.upper[cid, row]
                speed_limit = data.velocity_limit[cid, row]

                upper_enabled = upper < wp.float32(5.0e9)
                if upper_enabled or speed_limit > wp.float32(0.0):
                    upper_velocity = wp.float32(1.0e30)
                    if upper_enabled:
                        upper_velocity = (upper - coordinate) * idt
                    if speed_limit > wp.float32(0.0):
                        upper_velocity = wp.min(upper_velocity, speed_limit)
                    relative_velocity = _d6_row_velocity(wrench0, wrench1, v0, w0, v1, w1)
                    old_impulse = data.upper_impulse[cid, row]
                    new_impulse = wp.min(
                        wp.float32(0.0),
                        old_impulse - sor_boost * (relative_velocity - upper_velocity) / effective_mass_inverse,
                    )
                    delta = new_impulse - old_impulse
                    data.upper_impulse[cid, row] = new_impulse
                    v0, w0, v1, w1 = _d6_apply_row_impulse(
                        wrench0,
                        wrench1,
                        delta,
                        inverse_mass0,
                        inverse_mass1,
                        inverse_inertia0,
                        inverse_inertia1,
                        v0,
                        w0,
                        v1,
                        w1,
                    )

                lower_enabled = lower > wp.float32(-5.0e9)
                if lower_enabled or speed_limit > wp.float32(0.0):
                    lower_velocity = wp.float32(-1.0e30)
                    if lower_enabled:
                        lower_velocity = (lower - coordinate) * idt
                    if speed_limit > wp.float32(0.0):
                        lower_velocity = wp.max(lower_velocity, -speed_limit)
                    relative_velocity = _d6_row_velocity(wrench0, wrench1, v0, w0, v1, w1)
                    old_impulse = data.lower_impulse[cid, row]
                    new_impulse = wp.max(
                        wp.float32(0.0),
                        old_impulse - sor_boost * (relative_velocity - lower_velocity) / effective_mass_inverse,
                    )
                    delta = new_impulse - old_impulse
                    data.lower_impulse[cid, row] = new_impulse
                    v0, w0, v1, w1 = _d6_apply_row_impulse(
                        wrench0,
                        wrench1,
                        delta,
                        inverse_mass0,
                        inverse_mass1,
                        inverse_inertia0,
                        inverse_inertia1,
                        v0,
                        w0,
                        v1,
                        w1,
                    )

                friction = data.friction[cid, row]
                if friction > wp.float32(0.0):
                    relative_velocity = _d6_row_velocity(wrench0, wrench1, v0, w0, v1, w1)
                    old_impulse = data.friction_impulse[cid, row]
                    impulse_limit = friction / idt
                    gamma = PHOENX_FRICTION_SLIP_VELOCITY / impulse_limit
                    effective_mass = wp.float32(1.0) / (effective_mass_inverse + gamma)
                    new_impulse = wp.clamp(
                        old_impulse - sor_boost * effective_mass * (relative_velocity + gamma * old_impulse),
                        -impulse_limit,
                        impulse_limit,
                    )
                    delta = new_impulse - old_impulse
                    data.friction_impulse[cid, row] = new_impulse
                    v0, w0, v1, w1 = _d6_apply_row_impulse(
                        wrench0,
                        wrench1,
                        delta,
                        inverse_mass0,
                        inverse_mass1,
                        inverse_inertia0,
                        inverse_inertia1,
                        v0,
                        w0,
                        v1,
                        w1,
                    )
                else:
                    data.friction_impulse[cid, row] = wp.float32(0.0)
    return v0, w0, v1, w1
