# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar-row assembly for PhoenX rigid joints."""

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    pd_coefficients,
    read_float,
    read_int,
    read_quat,
    read_vec3,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_BODY1,
    _OFF_BODY2,
    _OFF_DAMPING_DRIVE,
    _OFF_DRIVE_MODE,
    _OFF_FRICTION_COEFFICIENT,
    _OFF_INV_INITIAL_ORIENTATION,
    _OFF_JOINT_MODE,
    _OFF_LA1_B1,
    _OFF_LA1_B2,
    _OFF_LA2_B1,
    _OFF_LA2_B2,
    _OFF_LA3_B1,
    _OFF_LA3_B2,
    _OFF_MAX_FORCE_DRIVE,
    _OFF_MAX_VALUE,
    _OFF_MIN_VALUE,
    _OFF_STIFFNESS_DRIVE,
    _OFF_TARGET,
    _OFF_TARGET_VELOCITY,
    DRIVE_MODE_OFF,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)
from newton._src.solvers.phoenx.helpers.math_helpers import create_orthonormal, extract_rotation_angle
from newton._src.solvers.phoenx.simple.rows import ScalarRowContainer

__all__ = ["JOINT_ROW_STRIDE", "assemble_joint_scalar_rows_kernel"]


JOINT_ROW_STRIDE = 9


@wp.func
def _set_row(
    rows: ScalarRowContainer,
    row: wp.int32,
    body_a: wp.int32,
    body_b: wp.int32,
    direction: wp.vec3f,
    arm_a: wp.vec3f,
    arm_b: wp.vec3f,
    error: wp.float32,
    idt: wp.float32,
):
    rows.active[row] = wp.int32(1)
    rows.body_a[row] = body_a
    rows.body_b[row] = body_b
    rows.jacobian_linear_a[row] = direction
    rows.jacobian_angular_a[row] = wp.cross(arm_a, direction)
    rows.jacobian_linear_b[row] = -direction
    rows.jacobian_angular_b[row] = -wp.cross(arm_b, direction)
    rows.bias[row] = wp.float32(0.2) * idt * error
    rows.softness[row] = wp.float32(0.0)
    rows.lower[row] = wp.float32(-1.0e30)
    rows.upper[row] = wp.float32(1.0e30)
    rows.bound_row[row] = row
    rows.bound_scale[row] = wp.float32(0.0)


@wp.func
def _set_angular_row(
    rows: ScalarRowContainer,
    row: wp.int32,
    body_a: wp.int32,
    body_b: wp.int32,
    axis: wp.vec3f,
    error: wp.float32,
    idt: wp.float32,
):
    rows.active[row] = wp.int32(1)
    rows.body_a[row] = body_a
    rows.body_b[row] = body_b
    rows.jacobian_linear_a[row] = wp.vec3f(0.0)
    rows.jacobian_angular_a[row] = axis
    rows.jacobian_linear_b[row] = wp.vec3f(0.0)
    rows.jacobian_angular_b[row] = -axis
    rows.bias[row] = wp.float32(0.2) * idt * error
    rows.softness[row] = wp.float32(0.0)
    rows.lower[row] = wp.float32(-1.0e30)
    rows.upper[row] = wp.float32(1.0e30)
    rows.bound_row[row] = row
    rows.bound_scale[row] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def assemble_joint_scalar_rows_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    joint_count: wp.int32,
    idt: wp.float32,
    rows: ScalarRowContainer,
    body_split_count: wp.array[wp.int32],
):
    """Assemble one independent equation per thread from packed joint data."""
    row = wp.tid()
    cid = row // wp.int32(JOINT_ROW_STRIDE)
    local_row = row - cid * wp.int32(JOINT_ROW_STRIDE)
    if cid >= joint_count:
        return

    was_active = rows.active[row]
    rows.active[row] = wp.int32(0)
    if was_active == wp.int32(0):
        rows.multiplier[row] = wp.float32(0.0)
    rows.relax_bias[row] = wp.float32(0.0)
    rows.split_anchor[row] = wp.int32(0)
    if local_row == wp.int32(0):
        rows.split_anchor[row] = wp.int32(1)
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    if local_row == wp.int32(0):
        wp.atomic_add(body_split_count, body_a, wp.int32(1))
        if body_b != body_a:
            wp.atomic_add(body_split_count, body_b, wp.int32(1))
    mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    if (
        local_row >= wp.int32(3)
        or mode == JOINT_MODE_PRISMATIC
        or mode == JOINT_MODE_CYLINDRICAL
        or mode == JOINT_MODE_PLANAR
    ):
        rows.multiplier[row] = wp.float32(0.0)
    orientation_a = bodies.orientation[body_a]
    orientation_b = bodies.orientation[body_b]

    arm1_a = wp.quat_rotate(orientation_a, read_vec3(constraints, _OFF_LA1_B1, cid))
    arm1_b = wp.quat_rotate(orientation_b, read_vec3(constraints, _OFF_LA1_B2, cid))
    point1_a = bodies.position[body_a] + arm1_a
    point1_b = bodies.position[body_b] + arm1_b
    error1 = point1_a - point1_b

    arm2_a = wp.quat_rotate(orientation_a, read_vec3(constraints, _OFF_LA2_B1, cid))
    arm2_b = wp.quat_rotate(orientation_b, read_vec3(constraints, _OFF_LA2_B2, cid))
    point2_a = bodies.position[body_a] + arm2_a
    point2_b = bodies.position[body_b] + arm2_b
    axis_a_raw = point2_a - point1_a
    axis_b_raw = point2_b - point1_b
    axis_a = wp.vec3f(1.0, 0.0, 0.0)
    axis_b = axis_a
    if wp.dot(axis_a_raw, axis_a_raw) > wp.float32(1.0e-12):
        axis_a = wp.normalize(axis_a_raw)
    if wp.dot(axis_b_raw, axis_b_raw) > wp.float32(1.0e-12):
        axis_b = wp.normalize(axis_b_raw)
    tangent1 = create_orthonormal(axis_a)
    tangent2 = wp.cross(axis_a, tangent1)

    if (
        mode == JOINT_MODE_BALL_SOCKET
        or mode == JOINT_MODE_REVOLUTE
        or mode == JOINT_MODE_FIXED
        or mode == JOINT_MODE_CABLE
        or mode == JOINT_MODE_UNIVERSAL
    ):
        if local_row < wp.int32(3):
            direction = wp.vec3f(0.0)
            direction[local_row] = wp.float32(1.0)
            _set_row(rows, row, body_a, body_b, direction, arm1_a, arm1_b, error1[local_row], idt)
            return

    alignment_error = wp.cross(axis_a, axis_b)
    if mode == JOINT_MODE_REVOLUTE or mode == JOINT_MODE_FIXED or mode == JOINT_MODE_CABLE:
        if local_row == wp.int32(3):
            _set_angular_row(rows, row, body_a, body_b, tangent1, wp.dot(alignment_error, tangent1), idt)
        elif local_row == wp.int32(4):
            _set_angular_row(rows, row, body_a, body_b, tangent2, wp.dot(alignment_error, tangent2), idt)
        elif local_row == wp.int32(5) and (mode == JOINT_MODE_FIXED or mode == JOINT_MODE_CABLE):
            arm3_a = wp.quat_rotate(orientation_a, read_vec3(constraints, _OFF_LA3_B1, cid))
            arm3_b = wp.quat_rotate(orientation_b, read_vec3(constraints, _OFF_LA3_B2, cid))
            error3 = wp.dot((bodies.position[body_a] + arm3_a) - (bodies.position[body_b] + arm3_b), tangent2)
            _set_row(rows, row, body_a, body_b, tangent2, arm3_a, arm3_b, error3, idt)
    elif mode == JOINT_MODE_PRISMATIC or mode == JOINT_MODE_CYLINDRICAL:
        if local_row == wp.int32(0):
            _set_row(rows, row, body_a, body_b, tangent1, arm1_a, arm1_b, wp.dot(error1, tangent1), idt)
        elif local_row == wp.int32(1):
            _set_row(rows, row, body_a, body_b, tangent2, arm1_a, arm1_b, wp.dot(error1, tangent2), idt)
        elif local_row == wp.int32(2):
            _set_angular_row(rows, row, body_a, body_b, tangent1, wp.dot(alignment_error, tangent1), idt)
        elif local_row == wp.int32(3):
            _set_angular_row(rows, row, body_a, body_b, tangent2, wp.dot(alignment_error, tangent2), idt)
        elif local_row == wp.int32(4) and mode == JOINT_MODE_PRISMATIC:
            arm3_a = wp.quat_rotate(orientation_a, read_vec3(constraints, _OFF_LA3_B1, cid))
            arm3_b = wp.quat_rotate(orientation_b, read_vec3(constraints, _OFF_LA3_B2, cid))
            error3 = wp.dot((bodies.position[body_a] + arm3_a) - (bodies.position[body_b] + arm3_b), tangent2)
            _set_row(rows, row, body_a, body_b, tangent2, arm3_a, arm3_b, error3, idt)
    elif mode == JOINT_MODE_PLANAR:
        if local_row == wp.int32(0):
            _set_row(rows, row, body_a, body_b, axis_a, arm1_a, arm1_b, wp.dot(error1, axis_a), idt)
        elif local_row == wp.int32(1):
            _set_angular_row(rows, row, body_a, body_b, tangent1, wp.dot(alignment_error, tangent1), idt)
        elif local_row == wp.int32(2):
            _set_angular_row(rows, row, body_a, body_b, tangent2, wp.dot(alignment_error, tangent2), idt)
    elif mode == JOINT_MODE_UNIVERSAL and local_row == wp.int32(3):
        _set_angular_row(rows, row, body_a, body_b, axis_a, wp.dot(alignment_error, axis_a), idt)

    # Axial drive, limit, and friction deliberately occupy separate scalar
    # rows. They share a Jacobian but never a local equation-system solve.
    if local_row < wp.int32(6) or local_row > wp.int32(8):
        return
    if mode != JOINT_MODE_REVOLUTE and mode != JOINT_MODE_PRISMATIC:
        return

    axis = axis_a
    linear_a = wp.vec3f(0.0)
    angular_a = -axis
    linear_b = wp.vec3f(0.0)
    angular_b = axis
    coordinate = wp.float32(0.0)
    if mode == JOINT_MODE_PRISMATIC:
        linear_a = -axis
        angular_a = -wp.cross(arm1_a, axis)
        linear_b = axis
        angular_b = wp.cross(arm1_b, axis)
        coordinate = wp.dot(point1_b - point1_a, axis)
    else:
        initial_orientation_inv = read_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid)
        relative = orientation_b * initial_orientation_inv * wp.quat_inverse(orientation_a)
        coordinate = extract_rotation_angle(relative, axis)

    rows.body_a[row] = body_a
    rows.body_b[row] = body_b
    rows.jacobian_linear_a[row] = linear_a
    rows.jacobian_angular_a[row] = angular_a
    rows.jacobian_linear_b[row] = linear_b
    rows.jacobian_angular_b[row] = angular_b
    rows.bound_row[row] = row
    rows.bound_scale[row] = wp.float32(0.0)

    if local_row == wp.int32(6):
        drive_mode = read_int(constraints, _OFF_DRIVE_MODE, cid)
        stiffness = read_float(constraints, _OFF_STIFFNESS_DRIVE, cid)
        damping = read_float(constraints, _OFF_DAMPING_DRIVE, cid)
        if drive_mode == DRIVE_MODE_OFF or (stiffness <= wp.float32(0.0) and damping <= wp.float32(0.0)):
            return
        inv_inertia_a = mat33_from_sym6(bodies.inverse_inertia_world[body_a])
        inv_inertia_b = mat33_from_sym6(bodies.inverse_inertia_world[body_b])
        eff_inv = (
            bodies.inverse_mass[body_a] * wp.dot(linear_a, linear_a)
            + wp.dot(angular_a, inv_inertia_a @ angular_a)
            + bodies.inverse_mass[body_b] * wp.dot(linear_b, linear_b)
            + wp.dot(angular_b, inv_inertia_b @ angular_b)
        )
        target = read_float(constraints, _OFF_TARGET, cid)
        target_velocity = read_float(constraints, _OFF_TARGET_VELOCITY, cid)
        gamma, spring_bias, _effective_mass = pd_coefficients(
            stiffness,
            damping,
            coordinate - target,
            eff_inv,
            wp.float32(1.0) / idt,
            wp.float32(1.0),
        )
        max_force = read_float(constraints, _OFF_MAX_FORCE_DRIVE, cid)
        max_impulse = wp.float32(1.0e30)
        if max_force > wp.float32(0.0):
            max_impulse = max_force / idt
        rows.active[row] = wp.int32(1)
        rows.bias[row] = -spring_bias + target_velocity
        rows.relax_bias[row] = rows.bias[row]
        rows.softness[row] = gamma
        rows.lower[row] = -max_impulse
        rows.upper[row] = max_impulse
    elif local_row == wp.int32(7):
        minimum = read_float(constraints, _OFF_MIN_VALUE, cid)
        maximum = read_float(constraints, _OFF_MAX_VALUE, cid)
        if minimum > maximum or (coordinate >= minimum and coordinate <= maximum):
            return
        error = coordinate - wp.clamp(coordinate, minimum, maximum)
        rows.active[row] = wp.int32(1)
        rows.bias[row] = wp.float32(0.2) * idt * error
        rows.relax_bias[row] = rows.bias[row]
        rows.softness[row] = wp.float32(0.0)
        if coordinate > maximum:
            rows.lower[row] = wp.float32(-1.0e30)
            rows.upper[row] = wp.float32(0.0)
        else:
            rows.lower[row] = wp.float32(0.0)
            rows.upper[row] = wp.float32(1.0e30)
    else:
        friction = read_float(constraints, _OFF_FRICTION_COEFFICIENT, cid)
        if friction <= wp.float32(0.0):
            return
        max_impulse = friction / idt
        rows.active[row] = wp.int32(1)
        rows.bias[row] = wp.float32(0.0)
        rows.softness[row] = wp.float32(0.0)
        rows.lower[row] = -max_impulse
        rows.upper[row] = max_impulse
