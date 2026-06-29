# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense loop-closure blocks backed by reduced articulation inverse mass."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import JointType, Model
from newton._src.solvers.phoenx.body import MOTION_ARTICULATED, BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LINEAR,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import _articulation_pair_wrench_response
from newton._src.solvers.phoenx.helpers.math_helpers import apply_body_spatial_impulse, create_orthonormal

_MAX_ROWS = 6
_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_ivec6 = wp.types.vector(length=6, dtype=wp.int32)
_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)


@wp.func
def _pseudoinverse_symmetric6(matrix: _mat66, size: wp.int32) -> _mat66:
    """Invert an independent row basis of a positive semidefinite block."""
    factor = _mat66(0.0)
    residual = _vec6(0.0)
    permutation = _ivec6(0)
    largest = wp.float32(0.0)
    for row in range(_MAX_ROWS):
        permutation[row] = wp.int32(row)
        if wp.int32(row) < size:
            residual[row] = wp.max(matrix[row, row], wp.float32(0.0))
            largest = wp.max(largest, residual[row])
    threshold = wp.max(wp.float32(1.0e-7) * largest, wp.float32(1.0e-10))

    rank = wp.int32(0)
    for column in range(_MAX_ROWS):
        if wp.int32(column) < size:
            pivot = wp.int32(column)
            pivot_value = residual[column]
            for candidate in range(_MAX_ROWS):
                if wp.int32(candidate) > wp.int32(column) and wp.int32(candidate) < size:
                    if residual[candidate] > pivot_value:
                        pivot = wp.int32(candidate)
                        pivot_value = residual[candidate]
            if pivot_value > threshold:
                if pivot != wp.int32(column):
                    temporary_index = permutation[column]
                    permutation[column] = permutation[pivot]
                    permutation[pivot] = temporary_index
                    temporary_residual = residual[column]
                    residual[column] = residual[pivot]
                    residual[pivot] = temporary_residual
                    for previous in range(_MAX_ROWS):
                        if wp.int32(previous) < wp.int32(column):
                            temporary_factor = factor[column, previous]
                            factor[column, previous] = factor[pivot, previous]
                            factor[pivot, previous] = temporary_factor

                factor[column, column] = wp.sqrt(residual[column])
                for row in range(_MAX_ROWS):
                    if wp.int32(row) > wp.int32(column) and wp.int32(row) < size:
                        value = matrix[permutation[row], permutation[column]]
                        for previous in range(_MAX_ROWS):
                            if wp.int32(previous) < wp.int32(column):
                                value -= factor[row, previous] * factor[column, previous]
                        factor[row, column] = value / factor[column, column]
                        residual[row] = wp.max(
                            residual[row] - factor[row, column] * factor[row, column],
                            wp.float32(0.0),
                        )
                rank += wp.int32(1)

    inverse = _mat66(0.0)
    for basis_column in range(_MAX_ROWS):
        if wp.int32(basis_column) < rank:
            forward = _vec6(0.0)
            solution = _vec6(0.0)
            for row in range(_MAX_ROWS):
                if wp.int32(row) < rank:
                    value = wp.float32(1.0) if row == basis_column else wp.float32(0.0)
                    for previous in range(_MAX_ROWS):
                        if wp.int32(previous) < wp.int32(row):
                            value -= factor[row, previous] * forward[previous]
                    forward[row] = value / factor[row, row]
            for reverse in range(_MAX_ROWS):
                row = _MAX_ROWS - 1 - reverse
                if wp.int32(row) < rank:
                    value = forward[row]
                    for following in range(_MAX_ROWS):
                        if wp.int32(following) > wp.int32(row) and wp.int32(following) < rank:
                            value -= factor[following, row] * solution[following]
                    solution[row] = value / factor[row, row]
            for row in range(_MAX_ROWS):
                if wp.int32(row) < rank:
                    inverse[permutation[row], permutation[basis_column]] = solution[row]
    return inverse


@wp.func
def _body_origin_transform(bodies: BodyContainer, body: wp.int32) -> wp.transform:
    if body <= wp.int32(0):
        return wp.transform_identity()
    rotation = bodies.orientation[body]
    origin = bodies.position[body] - wp.quat_rotate(rotation, bodies.body_com[body])
    return wp.transform(origin, rotation)


@wp.func
def _body_origin_twist(bodies: BodyContainer, body: wp.int32) -> wp.spatial_vector:
    if body <= wp.int32(0):
        return wp.spatial_vector()
    omega = bodies.angular_velocity[body]
    velocity_origin = bodies.velocity[body] - wp.cross(omega, bodies.position[body])
    return wp.spatial_vector(velocity_origin, omega)


@wp.func
def _rigid_wrench_response(
    bodies: BodyContainer,
    body: wp.int32,
    wrench: wp.spatial_vector,
) -> wp.spatial_vector:
    if body <= wp.int32(0) or bodies.inverse_mass[body] == wp.float32(0.0):
        return wp.spatial_vector()
    force = wp.spatial_top(wrench)
    torque_origin = wp.spatial_bottom(wrench)
    position_com = bodies.position[body]
    torque_com = torque_origin - wp.cross(position_com, force)
    velocity_com = bodies.inverse_mass[body] * force
    omega = mat33_from_sym6(bodies.inverse_inertia_world[body]) * torque_com
    return wp.spatial_vector(velocity_com - wp.cross(omega, position_com), omega)


@wp.func
def _pair_wrench_response(
    bodies: BodyContainer,
    body0: wp.int32,
    wrench0: wp.spatial_vector,
    body1: wp.int32,
    wrench1: wp.spatial_vector,
):
    articulation0 = bodies.reduced.body_articulation[body0]
    articulation1 = bodies.reduced.body_articulation[body1]
    response0 = wp.spatial_vector()
    response1 = wp.spatial_vector()

    if bodies.motion_type[body0] != MOTION_ARTICULATED:
        response0 = _rigid_wrench_response(bodies, body0, wrench0)
    if bodies.motion_type[body1] != MOTION_ARTICULATED:
        response1 = _rigid_wrench_response(bodies, body1, wrench1)

    for group in range(2):
        articulation = articulation0
        if group == 1:
            articulation = articulation1
        if articulation < wp.int32(0) or (group == 1 and articulation == articulation0):
            continue
        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        group_wrench0 = wp.spatial_vector()
        group_wrench1 = wp.spatial_vector()
        if articulation0 == articulation:
            slot0 = body0
            group_wrench0 = wrench0
        if articulation1 == articulation:
            if slot0 < wp.int32(0):
                slot0 = body1
                group_wrench0 = wrench1
            else:
                slot1 = body1
                group_wrench1 = wrench1
        _articulation_pair_wrench_response(
            bodies,
            slot0,
            group_wrench0,
            slot1,
            group_wrench1,
            wp.bool(False),
        )

    if bodies.motion_type[body0] == MOTION_ARTICULATED:
        response0 = bodies.reduced.body_acceleration[body0 - wp.int32(1)]
    if bodies.motion_type[body1] == MOTION_ARTICULATED:
        response1 = bodies.reduced.body_acceleration[body1 - wp.int32(1)]
    return response0, response1


@wp.func
def _apply_rigid_wrench(bodies: BodyContainer, body: wp.int32, wrench: wp.spatial_vector):
    if body <= wp.int32(0) or bodies.inverse_mass[body] == wp.float32(0.0):
        return
    force = wp.spatial_top(wrench)
    torque_com = wp.spatial_bottom(wrench) - wp.cross(bodies.position[body], force)
    velocity, omega = apply_body_spatial_impulse(
        bodies.velocity[body],
        bodies.angular_velocity[body],
        bodies.inverse_mass[body],
        mat33_from_sym6(bodies.inverse_inertia_world[body]),
        force,
        torque_com,
    )
    bodies.velocity[body] = velocity
    bodies.angular_velocity[body] = omega


@wp.func
def _apply_pair_wrench(
    bodies: BodyContainer,
    body0: wp.int32,
    wrench0: wp.spatial_vector,
    body1: wp.int32,
    wrench1: wp.spatial_vector,
):
    articulation0 = bodies.reduced.body_articulation[body0]
    articulation1 = bodies.reduced.body_articulation[body1]
    if bodies.motion_type[body0] != MOTION_ARTICULATED:
        _apply_rigid_wrench(bodies, body0, wrench0)
    if bodies.motion_type[body1] != MOTION_ARTICULATED:
        _apply_rigid_wrench(bodies, body1, wrench1)

    for group in range(2):
        articulation = articulation0
        if group == 1:
            articulation = articulation1
        if articulation < wp.int32(0) or (group == 1 and articulation == articulation0):
            continue
        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        group_wrench0 = wp.spatial_vector()
        group_wrench1 = wp.spatial_vector()
        if articulation0 == articulation:
            slot0 = body0
            group_wrench0 = wrench0
        if articulation1 == articulation:
            if slot0 < wp.int32(0):
                slot0 = body1
                group_wrench0 = wrench1
            else:
                slot1 = body1
                group_wrench1 = wrench1
        _articulation_pair_wrench_response(
            bodies,
            slot0,
            group_wrench0,
            slot1,
            group_wrench1,
            wp.bool(True),
        )


@wp.func
def _quat_log(q: wp.quat) -> wp.vec3:
    if q[3] < wp.float32(0.0):
        q = wp.quat(-q[0], -q[1], -q[2], -q[3])
    xyz = wp.vec3(q[0], q[1], q[2])
    length = wp.length(xyz)
    if length < wp.float32(1.0e-8):
        return wp.float32(2.0) * xyz
    return xyz * (wp.float32(2.0) * wp.atan2(length, q[3]) / length)


@wp.func
def _set_point_row(
    loop: wp.int32,
    row: wp.int32,
    point0: wp.vec3,
    point1: wp.vec3,
    direction: wp.vec3,
    error: wp.float32,
    bias_rate: wp.float32,
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
):
    force0 = -direction
    force1 = direction
    row_wrench0[loop, row] = wp.spatial_vector(force0, wp.cross(point0, force0))
    row_wrench1[loop, row] = wp.spatial_vector(force1, wp.cross(point1, force1))
    row_bias[loop, row] = bias_rate * error


@wp.func
def _set_angular_row(
    loop: wp.int32,
    row: wp.int32,
    direction: wp.vec3,
    error: wp.float32,
    bias_rate: wp.float32,
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
):
    row_wrench0[loop, row] = wp.spatial_vector(wp.vec3(0.0), -direction)
    row_wrench1[loop, row] = wp.spatial_vector(wp.vec3(0.0), direction)
    row_bias[loop, row] = bias_rate * error


@wp.func
def _prepare_loop_rows(
    loop: wp.int32,
    joint: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    bodies: BodyContainer,
    bias_rate: wp.float32,
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
) -> wp.int32:
    parent = joint_parent[joint] + wp.int32(1)
    child = joint_child[joint] + wp.int32(1)
    x_wpj = _body_origin_transform(bodies, parent) * joint_x_p[joint]
    x_wcj = _body_origin_transform(bodies, child) * joint_x_c[joint]
    point0 = wp.transform_get_translation(x_wpj)
    point1 = wp.transform_get_translation(x_wcj)
    q0 = wp.transform_get_rotation(x_wpj)
    q1 = wp.transform_get_rotation(x_wcj)
    kind = joint_type[joint]

    for row in range(_MAX_ROWS):
        row_wrench0[loop, row] = wp.spatial_vector()
        row_wrench1[loop, row] = wp.spatial_vector()
        row_bias[loop, row] = wp.float32(0.0)

    if kind == JointType.BALL or kind == JointType.REVOLUTE or kind == JointType.FIXED:
        error = point1 - point0
        for row in range(3):
            direction = wp.vec3(0.0)
            direction[row] = wp.float32(1.0)
            _set_point_row(
                loop,
                wp.int32(row),
                point0,
                point1,
                direction,
                error[row],
                bias_rate,
                row_wrench0,
                row_wrench1,
                row_bias,
            )
        if kind == JointType.BALL:
            return wp.int32(3)

    if kind == JointType.REVOLUTE or kind == JointType.PRISMATIC:
        local_axis = joint_axis[joint_qd_start[joint]]
        axis0 = wp.normalize(wp.quat_rotate(q0, local_axis))
        axis1 = wp.normalize(wp.quat_rotate(q1, local_axis))
        tangent0 = create_orthonormal(axis0)
        tangent1 = wp.cross(axis0, tangent0)
        alignment_error = wp.cross(axis0, axis1)
        if kind == JointType.REVOLUTE:
            _set_angular_row(
                loop,
                wp.int32(3),
                tangent0,
                wp.dot(alignment_error, tangent0),
                bias_rate,
                row_wrench0,
                row_wrench1,
                row_bias,
            )
            _set_angular_row(
                loop,
                wp.int32(4),
                tangent1,
                wp.dot(alignment_error, tangent1),
                bias_rate,
                row_wrench0,
                row_wrench1,
                row_bias,
            )
            return wp.int32(5)

        error = point1 - point0
        _set_point_row(
            loop,
            wp.int32(0),
            point0,
            point1,
            tangent0,
            wp.dot(error, tangent0),
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        _set_point_row(
            loop,
            wp.int32(1),
            point0,
            point1,
            tangent1,
            wp.dot(error, tangent1),
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        rotation_error = _quat_log(q1 * wp.quat_inverse(q0))
        axis_x = wp.quat_rotate(q0, wp.vec3(1.0, 0.0, 0.0))
        axis_y = wp.quat_rotate(q0, wp.vec3(0.0, 1.0, 0.0))
        axis_z = wp.quat_rotate(q0, wp.vec3(0.0, 0.0, 1.0))
        _set_angular_row(
            loop,
            wp.int32(2),
            axis_x,
            wp.dot(rotation_error, axis_x),
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        _set_angular_row(
            loop,
            wp.int32(3),
            axis_y,
            wp.dot(rotation_error, axis_y),
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        _set_angular_row(
            loop,
            wp.int32(4),
            axis_z,
            wp.dot(rotation_error, axis_z),
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        return wp.int32(5)

    rotation_error = _quat_log(q1 * wp.quat_inverse(q0))
    for row in range(3):
        direction = wp.quat_rotate(
            q0,
            wp.vec3(
                wp.float32(1.0) if row == 0 else wp.float32(0.0),
                wp.float32(1.0) if row == 1 else wp.float32(0.0),
                wp.float32(1.0) if row == 2 else wp.float32(0.0),
            ),
        )
        _set_angular_row(
            loop,
            wp.int32(row + 3),
            direction,
            wp.dot(rotation_error, direction),
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
    return wp.int32(6)


@wp.func
def _apply_loop_delta(
    loop: wp.int32,
    count: wp.int32,
    body0: wp.int32,
    body1: wp.int32,
    delta: _vec6,
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    bodies: BodyContainer,
):
    wrench0 = wp.spatial_vector()
    wrench1 = wp.spatial_vector()
    for row in range(_MAX_ROWS):
        if wp.int32(row) < count:
            wrench0 += delta[row] * row_wrench0[loop, row]
            wrench1 += delta[row] * row_wrench1[loop, row]
    _apply_pair_wrench(bodies, body0, wrench0, body1, wrench1)


@wp.kernel(enable_backward=False)
def _solve_reduced_loops_kernel(
    world_loop_start: wp.array[wp.int32],
    loop_joint: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    bodies: BodyContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    iterations: wp.int32,
    use_bias: wp.bool,
    warmstart: wp.bool,
    row_count: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
    inverse_mass: wp.array3d[wp.float32],
    multiplier: wp.array2d[wp.float32],
):
    world = wp.tid()
    start = world_loop_start[world]
    end = world_loop_start[world + wp.int32(1)]
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_LINEAR,
        DEFAULT_DAMPING_RATIO,
        wp.float32(1.0) / idt,
    )

    for loop in range(start, end):
        joint = loop_joint[loop]
        body0 = joint_parent[joint] + wp.int32(1)
        body1 = joint_child[joint] + wp.int32(1)
        count = _prepare_loop_rows(
            loop,
            joint,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_x_p,
            joint_x_c,
            joint_axis,
            bodies,
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        row_count[loop] = count
        matrix = _mat66(0.0)
        for column in range(_MAX_ROWS):
            if wp.int32(column) < count:
                response0, response1 = _pair_wrench_response(
                    bodies,
                    body0,
                    row_wrench0[loop, column],
                    body1,
                    row_wrench1[loop, column],
                )
                for row in range(_MAX_ROWS):
                    if wp.int32(row) < count:
                        matrix[row, column] = wp.dot(row_wrench0[loop, row], response0) + wp.dot(
                            row_wrench1[loop, row], response1
                        )
        matrix_inverse = _pseudoinverse_symmetric6(matrix, count)
        for row in range(_MAX_ROWS):
            for column in range(_MAX_ROWS):
                inverse_mass[loop, row, column] = matrix_inverse[row, column]
            if wp.int32(row) < count and matrix_inverse[row, row] == wp.float32(0.0):
                multiplier[loop, row] = wp.float32(0.0)

        if warmstart:
            accumulated = _vec6(0.0)
            for row in range(_MAX_ROWS):
                if wp.int32(row) < count:
                    accumulated[row] = multiplier[loop, row]
            _apply_loop_delta(loop, count, body0, body1, accumulated, row_wrench0, row_wrench1, bodies)

    for _iteration in range(iterations):
        for loop in range(start, end):
            joint = loop_joint[loop]
            body0 = joint_parent[joint] + wp.int32(1)
            body1 = joint_child[joint] + wp.int32(1)
            count = row_count[loop]
            twist0 = _body_origin_twist(bodies, body0)
            twist1 = _body_origin_twist(bodies, body1)
            rhs = _vec6(0.0)
            old_multiplier = _vec6(0.0)
            for row in range(_MAX_ROWS):
                if wp.int32(row) < count:
                    rhs[row] = wp.dot(row_wrench0[loop, row], twist0) + wp.dot(row_wrench1[loop, row], twist1)
                    if use_bias:
                        rhs[row] += row_bias[loop, row]
                    old_multiplier[row] = multiplier[loop, row]

            delta_unsoftened = _vec6(0.0)
            for row in range(_MAX_ROWS):
                if wp.int32(row) < count:
                    for column in range(_MAX_ROWS):
                        if wp.int32(column) < count:
                            delta_unsoftened[row] -= inverse_mass[loop, row, column] * rhs[column]
            delta = _vec6(0.0)
            for row in range(_MAX_ROWS):
                if wp.int32(row) < count:
                    delta[row] = sor_boost * (mass_coeff * delta_unsoftened[row] - impulse_coeff * old_multiplier[row])
                    multiplier[loop, row] = old_multiplier[row] + delta[row]
            _apply_loop_delta(loop, count, body0, body1, delta, row_wrench0, row_wrench1, bodies)


class ReducedLoopSystem:
    """Graph-stable dense block solver for loop joints touching reduced trees."""

    _SUPPORTED_TYPES = (
        int(JointType.BALL),
        int(JointType.REVOLUTE),
        int(JointType.PRISMATIC),
        int(JointType.FIXED),
    )

    def __init__(self, model: Model, body_is_reduced: np.ndarray):
        joint_articulation = model.joint_articulation.numpy()
        parent = model.joint_parent.numpy()
        child = model.joint_child.numpy()
        body_world = model.body_world.numpy()
        joint_type = model.joint_type.numpy()
        loops: list[tuple[int, int]] = []
        for joint in range(int(model.joint_count)):
            if int(joint_articulation[joint]) >= 0:
                continue
            parent_reduced = int(parent[joint]) >= 0 and bool(body_is_reduced[int(parent[joint])])
            child_reduced = int(child[joint]) >= 0 and bool(body_is_reduced[int(child[joint])])
            if not (parent_reduced or child_reduced):
                continue
            kind = int(joint_type[joint])
            if kind not in self._SUPPORTED_TYPES:
                raise NotImplementedError(f"reduced PhoenX loop joint {joint} has unsupported JointType value {kind}")
            endpoint = int(child[joint]) if int(child[joint]) >= 0 else int(parent[joint])
            world = max(0, int(body_world[endpoint]))
            loops.append((world, joint))

        loops.sort()
        self.count = len(loops)
        self.device = model.device
        self.world_count = int(model.world_count)
        self.joint_indices_np = np.asarray([joint for _world, joint in loops], dtype=np.int32)
        offsets = np.zeros(self.world_count + 1, dtype=np.int32)
        for world, _joint in loops:
            offsets[world + 1] += 1
        np.cumsum(offsets, out=offsets)
        self.world_loop_start = wp.array(offsets, dtype=wp.int32, device=self.device)
        self.loop_joint = wp.array(self.joint_indices_np, dtype=wp.int32, device=self.device)
        capacity = max(1, self.count)
        self.row_count = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.row_wrench0 = wp.zeros((capacity, _MAX_ROWS), dtype=wp.spatial_vector, device=self.device)
        self.row_wrench1 = wp.zeros((capacity, _MAX_ROWS), dtype=wp.spatial_vector, device=self.device)
        self.row_bias = wp.zeros((capacity, _MAX_ROWS), dtype=wp.float32, device=self.device)
        self.inverse_mass = wp.zeros((capacity, _MAX_ROWS, _MAX_ROWS), dtype=wp.float32, device=self.device)
        self.multiplier = wp.zeros((capacity, _MAX_ROWS), dtype=wp.float32, device=self.device)

    def solve(
        self,
        model: Model,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: float,
        iterations: int,
        *,
        use_bias: bool,
        warmstart: bool,
    ) -> None:
        if self.count == 0 or iterations <= 0:
            return
        wp.launch(
            _solve_reduced_loops_kernel,
            dim=self.world_count,
            inputs=[
                self.world_loop_start,
                self.loop_joint,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                bodies,
                idt,
                wp.float32(sor_boost),
                wp.int32(iterations),
                wp.bool(use_bias),
                wp.bool(warmstart),
            ],
            outputs=[
                self.row_count,
                self.row_wrench0,
                self.row_wrench1,
                self.row_bias,
                self.inverse_mass,
                self.multiplier,
            ],
            device=self.device,
        )


__all__ = ["ReducedLoopSystem"]
