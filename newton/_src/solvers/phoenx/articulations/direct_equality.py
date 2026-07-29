# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mechanism-wide maximal-coordinate equality solves for PhoenX."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.sim import JointType, Model
from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from newton._src.solvers.phoenx.articulations.reduced_loop import (
    _body_origin_transform,
    _quat_log,
    _set_angular_row,
)
from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LINEAR,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_FIXED,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)
from newton._src.solvers.phoenx.helpers.math_helpers import create_orthonormal

_MAX_ROWS = 6


@dataclass(frozen=True)
class DirectEqualityTopology:
    """Immutable host topology for the mechanism equality systems."""

    joints: np.ndarray
    row_joint: np.ndarray
    row_local: np.ndarray
    dimensions: tuple[int, ...]
    permutation: np.ndarray
    mechanism_row_start: np.ndarray
    body_row_start: np.ndarray
    body_rows: np.ndarray


def _default_joint_modes(joint_types: np.ndarray) -> np.ndarray:
    modes = np.full(len(joint_types), -1, dtype=np.int32)
    modes[joint_types == int(JointType.BALL)] = int(JOINT_MODE_BALL_SOCKET)
    modes[joint_types == int(JointType.REVOLUTE)] = int(JOINT_MODE_REVOLUTE)
    modes[joint_types == int(JointType.PRISMATIC)] = int(JOINT_MODE_PRISMATIC)
    modes[joint_types == int(JointType.FIXED)] = int(JOINT_MODE_FIXED)
    modes[joint_types == int(JointType.CABLE)] = int(JOINT_MODE_CABLE)
    return modes


def _structural_row_count(mode: int) -> int:
    if mode == int(JOINT_MODE_BALL_SOCKET):
        return 3
    if mode in (int(JOINT_MODE_REVOLUTE), int(JOINT_MODE_PRISMATIC)):
        return 5
    if mode in (int(JOINT_MODE_FIXED), int(JOINT_MODE_CABLE)):
        return 6
    if mode == int(JOINT_MODE_UNIVERSAL):
        return 4
    return 0


def build_direct_equality_topology(
    model: Model,
    *,
    excluded_joint_mask: np.ndarray | None = None,
    effective_joint_mode: np.ndarray | None = None,
) -> DirectEqualityTopology:
    """Find disconnected maximal-coordinate mechanisms and their equality rows."""
    body_count = int(model.body_count)
    joint_count = int(model.joint_count)
    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int32)
    joint_mode = (
        _default_joint_modes(joint_type)
        if effective_joint_mode is None
        else np.asarray(effective_joint_mode, dtype=np.int32)
    )
    if joint_mode.shape != (joint_count,):
        raise ValueError(f"effective_joint_mode must have shape ({joint_count},), got {joint_mode.shape}")
    joint_parent = np.asarray(model.joint_parent.numpy(), dtype=np.int32)
    joint_child = np.asarray(model.joint_child.numpy(), dtype=np.int32)
    enabled = (
        np.asarray(model.joint_enabled.numpy(), dtype=bool)
        if model.joint_enabled is not None
        else np.ones(joint_count, dtype=bool)
    )
    excluded = (
        np.asarray(excluded_joint_mask, dtype=bool)
        if excluded_joint_mask is not None
        else np.zeros(joint_count, dtype=bool)
    )
    if excluded.shape != (joint_count,):
        raise ValueError(f"excluded_joint_mask must have shape ({joint_count},), got {excluded.shape}")

    inverse_mass = np.asarray(model.body_inv_mass.numpy(), dtype=np.float32)
    dynamic = inverse_mass > 0.0
    parent = np.arange(body_count, dtype=np.int32)

    def find(body: int) -> int:
        root = body
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[body]) != body:
            next_body = int(parent[body])
            parent[body] = root
            body = next_body
        return root

    def union(body0: int, body1: int) -> None:
        root0 = find(body0)
        root1 = find(body1)
        if root0 == root1:
            return
        if root0 < root1:
            parent[root1] = root0
        else:
            parent[root0] = root1

    structural: list[int] = []
    for joint in range(joint_count):
        if excluded[joint] or not enabled[joint] or _structural_row_count(int(joint_mode[joint])) == 0:
            continue
        body0 = int(joint_parent[joint])
        body1 = int(joint_child[joint])
        if body0 >= 0 and body1 >= 0 and dynamic[body0] and dynamic[body1]:
            union(body0, body1)
        structural.append(joint)

    mechanisms: dict[int, list[int]] = {}
    for joint in structural:
        endpoints = (int(joint_parent[joint]), int(joint_child[joint]))
        dynamic_endpoints = [body for body in endpoints if body >= 0 and dynamic[body]]
        if not dynamic_endpoints:
            continue
        root = find(dynamic_endpoints[0])
        mechanisms.setdefault(root, []).append(joint)

    ordered_mechanisms = sorted(mechanisms.values(), key=min)
    row_joint: list[int] = []
    row_local: list[int] = []
    mechanism_row_start = [0]
    ordered_joints: list[int] = []
    for joints in ordered_mechanisms:
        for joint in sorted(joints):
            ordered_joints.append(joint)
            for local_row in range(_structural_row_count(int(joint_mode[joint]))):
                row_joint.append(joint)
                row_local.append(local_row)
        mechanism_row_start.append(len(row_joint))

    body_rows_lists: list[list[int]] = [[] for _ in range(body_count + 1)]
    for row, joint in enumerate(row_joint):
        body0 = int(joint_parent[joint]) + 1
        body1 = int(joint_child[joint]) + 1
        if body0 > 0:
            body_rows_lists[body0].append(row)
        if body1 > 0 and body1 != body0:
            body_rows_lists[body1].append(row)
    body_row_start = [0]
    body_rows: list[int] = []
    for rows in body_rows_lists:
        body_rows.extend(rows)
        body_row_start.append(len(body_rows))

    permutation: list[int] = []
    for mechanism in range(len(ordered_mechanisms)):
        start = mechanism_row_start[mechanism]
        end = mechanism_row_start[mechanism + 1]
        dimension = end - start
        adjacency = [set() for _ in range(dimension)]
        row_bodies: list[set[int]] = []
        for row in range(start, end):
            joint = row_joint[row]
            row_bodies.append(
                {body for body in (int(joint_parent[joint]), int(joint_child[joint])) if body >= 0 and dynamic[body]}
            )
        for row in range(dimension):
            for column in range(row):
                if row_bodies[row].intersection(row_bodies[column]):
                    adjacency[row].add(column)
                    adjacency[column].add(row)

        unvisited = set(range(dimension))
        mechanism_order: list[int] = []
        while unvisited:
            seed = min(unvisited, key=lambda row: (len(adjacency[row]), row))
            queue = deque([seed])
            unvisited.remove(seed)
            component: list[int] = []
            while queue:
                row = queue.popleft()
                component.append(row)
                neighbors = sorted(
                    adjacency[row].intersection(unvisited),
                    key=lambda neighbor: (len(adjacency[neighbor]), neighbor),
                )
                for neighbor in neighbors:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
            mechanism_order.extend(reversed(component))
        permutation.extend(mechanism_order)

    dimensions = tuple(mechanism_row_start[i + 1] - mechanism_row_start[i] for i in range(len(ordered_mechanisms)))
    return DirectEqualityTopology(
        joints=np.asarray(ordered_joints, dtype=np.int32),
        row_joint=np.asarray(row_joint, dtype=np.int32),
        row_local=np.asarray(row_local, dtype=np.int32),
        dimensions=dimensions,
        permutation=np.asarray(permutation, dtype=np.int32),
        mechanism_row_start=np.asarray(mechanism_row_start, dtype=np.int32),
        body_row_start=np.asarray(body_row_start, dtype=np.int32),
        body_rows=np.asarray(body_rows, dtype=np.int32),
    )


@wp.func
def _set_direct_point_row(
    structural_index: wp.int32,
    row: wp.int32,
    point0_com: wp.vec3,
    point1_com: wp.vec3,
    direction: wp.vec3,
    error: wp.float32,
    bias_rate: wp.float32,
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
):
    force0 = -direction
    force1 = direction
    row_wrench0[structural_index, row] = wp.spatial_vector(force0, wp.cross(point0_com, force0))
    row_wrench1[structural_index, row] = wp.spatial_vector(force1, wp.cross(point1_com, force1))
    row_bias[structural_index, row] = bias_rate * error


@wp.func
def _prepare_direct_rows(
    structural_index: wp.int32,
    joint: wp.int32,
    effective_joint_mode: wp.array[wp.int32],
    effective_joint_axis: wp.array[wp.vec3],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    bodies: BodyContainer,
    bias_rate: wp.float32,
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
    row_error: wp.array2d[wp.float32],
    row_stiffness: wp.array2d[wp.float32],
    row_damping: wp.array2d[wp.float32],
) -> wp.int32:
    parent = joint_parent[joint] + wp.int32(1)
    child = joint_child[joint] + wp.int32(1)
    x_wpj = _body_origin_transform(bodies, parent) * joint_x_p[joint]
    x_wcj = _body_origin_transform(bodies, child) * joint_x_c[joint]
    point0 = wp.transform_get_translation(x_wpj)
    point1 = wp.transform_get_translation(x_wcj)
    q0 = wp.transform_get_rotation(x_wpj)
    q1 = wp.transform_get_rotation(x_wcj)
    mode = effective_joint_mode[joint]
    point0_com = point0
    point1_com = point1
    if parent > wp.int32(0):
        point0_com -= bodies.position[parent]
    if child > wp.int32(0):
        point1_com -= bodies.position[child]

    for row in range(_MAX_ROWS):
        row_wrench0[structural_index, row] = wp.spatial_vector()
        row_wrench1[structural_index, row] = wp.spatial_vector()
        row_bias[structural_index, row] = wp.float32(0.0)
        row_error[structural_index, row] = wp.float32(0.0)
        row_stiffness[structural_index, row] = wp.float32(0.0)
        row_damping[structural_index, row] = wp.float32(0.0)

    has_point_lock = (
        mode == JOINT_MODE_BALL_SOCKET
        or mode == JOINT_MODE_REVOLUTE
        or mode == JOINT_MODE_FIXED
        or mode == JOINT_MODE_CABLE
        or mode == JOINT_MODE_UNIVERSAL
    )
    if has_point_lock:
        point_error = point1 - point0
        for row in range(3):
            direction = wp.vec3(0.0)
            direction[row] = wp.float32(1.0)
            _set_direct_point_row(
                structural_index,
                wp.int32(row),
                point0_com,
                point1_com,
                direction,
                point_error[row],
                bias_rate,
                row_wrench0,
                row_wrench1,
                row_bias,
            )
            row_error[structural_index, row] = point_error[row]
        if mode == JOINT_MODE_BALL_SOCKET:
            return wp.int32(3)

    local_axis = effective_joint_axis[joint]
    axis0 = wp.normalize(wp.quat_rotate(q0, local_axis))
    if mode == JOINT_MODE_REVOLUTE:
        axis1 = wp.normalize(wp.quat_rotate(q1, local_axis))
        tangent0 = create_orthonormal(axis0)
        tangent1 = wp.cross(axis0, tangent0)
        alignment_error = wp.cross(axis0, axis1)
        error0 = wp.dot(alignment_error, tangent0)
        error1 = wp.dot(alignment_error, tangent1)
        _set_angular_row(
            structural_index,
            wp.int32(3),
            tangent0,
            error0,
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        _set_angular_row(
            structural_index,
            wp.int32(4),
            tangent1,
            error1,
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        row_error[structural_index, 3] = error0
        row_error[structural_index, 4] = error1
        return wp.int32(5)

    rotation_error = _quat_log(q1 * wp.quat_inverse(q0))
    if mode == JOINT_MODE_UNIVERSAL:
        error = wp.dot(rotation_error, axis0)
        _set_angular_row(
            structural_index,
            wp.int32(3),
            axis0,
            error,
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        row_error[structural_index, 3] = error
        return wp.int32(4)

    if mode == JOINT_MODE_PRISMATIC:
        tangent0 = create_orthonormal(axis0)
        tangent1 = wp.cross(axis0, tangent0)
        point_error = point1 - point0
        error0 = wp.dot(point_error, tangent0)
        error1 = wp.dot(point_error, tangent1)
        _set_direct_point_row(
            structural_index,
            wp.int32(0),
            point0_com,
            point1_com,
            tangent0,
            error0,
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        _set_direct_point_row(
            structural_index,
            wp.int32(1),
            point0_com,
            point1_com,
            tangent1,
            error1,
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        row_error[structural_index, 0] = error0
        row_error[structural_index, 1] = error1

    for angular_row in range(3):
        direction = wp.quat_rotate(
            q0,
            wp.vec3(
                wp.float32(1.0) if angular_row == 0 else wp.float32(0.0),
                wp.float32(1.0) if angular_row == 1 else wp.float32(0.0),
                wp.float32(1.0) if angular_row == 2 else wp.float32(0.0),
            ),
        )
        row = wp.int32(angular_row + 2) if mode == JOINT_MODE_PRISMATIC else wp.int32(angular_row + 3)
        error = wp.dot(rotation_error, direction)
        _set_angular_row(
            structural_index,
            row,
            direction,
            error,
            bias_rate,
            row_wrench0,
            row_wrench1,
            row_bias,
        )
        row_error[structural_index, row] = error
        if mode == JOINT_MODE_CABLE:
            bend_dof = joint_qd_start[joint] + wp.int32(1)
            row_stiffness[structural_index, row] = joint_target_ke[bend_dof]
            row_damping[structural_index, row] = joint_target_kd[bend_dof]
    return wp.int32(5) if mode == JOINT_MODE_PRISMATIC else wp.int32(6)


@wp.kernel(enable_backward=False)
def _prepare_direct_equality_rows_kernel(
    structural_joints: wp.array[wp.int32],
    effective_joint_mode: wp.array[wp.int32],
    effective_joint_axis: wp.array[wp.vec3],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    bodies: BodyContainer,
    idt: wp.float32,
    row_count: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
    row_error: wp.array2d[wp.float32],
    row_stiffness: wp.array2d[wp.float32],
    row_damping: wp.array2d[wp.float32],
):
    structural_index = wp.tid()
    bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_LINEAR,
        DEFAULT_DAMPING_RATIO,
        wp.float32(1.0) / idt,
    )
    joint = structural_joints[structural_index]
    count = _prepare_direct_rows(
        structural_index,
        joint,
        effective_joint_mode,
        effective_joint_axis,
        joint_parent,
        joint_child,
        joint_qd_start,
        joint_x_p,
        joint_x_c,
        joint_target_ke,
        joint_target_kd,
        bodies,
        bias_rate,
        row_wrench0,
        row_wrench1,
        row_bias,
        row_error,
        row_stiffness,
        row_damping,
    )
    row_count[structural_index] = count


@wp.func
def _row_wrench_for_body(
    body: wp.int32,
    joint: wp.int32,
    structural_index: wp.int32,
    local_row: wp.int32,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
) -> wp.spatial_vector:
    if joint_parent[joint] + wp.int32(1) == body:
        return row_wrench0[structural_index, local_row]
    if joint_child[joint] + wp.int32(1) == body:
        return row_wrench1[structural_index, local_row]
    return wp.spatial_vector()


@wp.func
def _direct_wrench_response(
    wrench: wp.spatial_vector,
    inverse_mass: wp.float32,
    inverse_inertia: wp.mat33,
) -> wp.spatial_vector:
    return wp.spatial_vector(
        inverse_mass * wp.spatial_top(wrench),
        inverse_inertia * wp.spatial_bottom(wrench),
    )


@wp.func
def _body_com_twist(bodies: BodyContainer, body: wp.int32) -> wp.spatial_vector:
    return wp.spatial_vector(bodies.velocity[body], bodies.angular_velocity[body])


@wp.kernel(enable_backward=False)
def _assemble_direct_equality_matrix_kernel(
    dimensions: wp.array[wp.int32],
    num_mechanisms: wp.int32,
    matrix_mechanism: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    joint_to_structural: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    bodies: BodyContainer,
    regularization: wp.float32,
    idt: wp.float32,
    row_error: wp.array2d[wp.float32],
    row_stiffness: wp.array2d[wp.float32],
    row_damping: wp.array2d[wp.float32],
    row_bias: wp.array2d[wp.float32],
    matrix: wp.array[wp.float32],
):
    matrix_index = wp.tid()
    mechanism = wp.int32(0)
    if num_mechanisms > wp.int32(1):
        mechanism = matrix_mechanism[matrix_index]
    dimension = dimensions[mechanism]
    local_index = matrix_index - matrix_offsets[mechanism]
    local_row = local_index // dimension
    local_column = local_index - local_row * dimension
    row = vector_offsets[mechanism] + local_row
    column = vector_offsets[mechanism] + local_column
    row_joint_index = row_joint[row]
    column_joint_index = row_joint[column]
    row_structural = joint_to_structural[row_joint_index]
    column_structural = joint_to_structural[column_joint_index]
    row_local_index = row_local[row]
    column_local_index = row_local[column]
    value = wp.float32(0.0)

    row_body0 = joint_parent[row_joint_index] + wp.int32(1)
    row_body1 = joint_child[row_joint_index] + wp.int32(1)
    for endpoint in range(2):
        body = row_body0 if endpoint == 0 else row_body1
        if body <= wp.int32(0) or bodies.inverse_mass[body] <= wp.float32(0.0):
            continue
        column_wrench = _row_wrench_for_body(
            body,
            column_joint_index,
            column_structural,
            column_local_index,
            joint_parent,
            joint_child,
            row_wrench0,
            row_wrench1,
        )
        row_wrench = _row_wrench_for_body(
            body,
            row_joint_index,
            row_structural,
            row_local_index,
            joint_parent,
            joint_child,
            row_wrench0,
            row_wrench1,
        )
        response = _direct_wrench_response(
            column_wrench,
            bodies.inverse_mass[body],
            mat33_from_sym6(bodies.inverse_inertia_world[body]),
        )
        value += wp.dot(row_wrench, response)

    if local_row == local_column:
        inverse_effective_mass = value
        stiffness = row_stiffness[row_structural, row_local_index]
        damping = row_damping[row_structural, row_local_index]
        if (stiffness > wp.float32(0.0) or damping > wp.float32(0.0)) and not wp.isinf(stiffness):
            dt = wp.float32(1.0) / idt
            denominator = damping + dt * stiffness
            if denominator > wp.float32(0.0) and not wp.isinf(damping):
                softness = wp.float32(1.0) / denominator
                value += softness * idt
                bias_factor = dt * stiffness * softness
                row_bias[row_structural, row_local_index] = (
                    row_error[row_structural, row_local_index] * bias_factor * idt
                )
        value += wp.max(
            regularization * wp.max(inverse_effective_mass, wp.float32(1.0)),
            wp.float32(1.0e-10),
        )
    matrix[matrix_index] = value


@wp.kernel(enable_backward=False)
def _compute_direct_equality_row_scale_kernel(
    diagonal_index: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    row_scale: wp.array[wp.float32],
):
    row = wp.tid()
    diagonal = matrix[diagonal_index[row]]
    row_scale[row] = wp.float32(1.0) / wp.sqrt(wp.max(diagonal, wp.float32(1.0e-20)))


@wp.kernel(enable_backward=False)
def _equilibrate_direct_equality_matrix_kernel(
    dimensions: wp.array[wp.int32],
    num_mechanisms: wp.int32,
    matrix_mechanism: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    row_scale: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
):
    matrix_index = wp.tid()
    mechanism = wp.int32(0)
    if num_mechanisms > wp.int32(1):
        mechanism = matrix_mechanism[matrix_index]
    dimension = dimensions[mechanism]
    local_index = matrix_index - matrix_offsets[mechanism]
    local_row = local_index // dimension
    local_column = local_index - local_row * dimension
    vector_offset = vector_offsets[mechanism]
    matrix[matrix_index] *= row_scale[vector_offset + local_row] * row_scale[vector_offset + local_column]


@wp.kernel(enable_backward=False)
def _build_direct_equality_rhs_kernel(
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    joint_to_structural: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
    bodies: BodyContainer,
    use_bias: wp.bool,
    row_scale: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
):
    row = wp.tid()
    joint = row_joint[row]
    structural_index = joint_to_structural[joint]
    local_row = row_local[row]
    body0 = joint_parent[joint] + wp.int32(1)
    body1 = joint_child[joint] + wp.int32(1)
    relative_velocity = wp.float32(0.0)
    if body0 > wp.int32(0):
        relative_velocity += wp.dot(row_wrench0[structural_index, local_row], _body_com_twist(bodies, body0))
    if body1 > wp.int32(0):
        relative_velocity += wp.dot(row_wrench1[structural_index, local_row], _body_com_twist(bodies, body1))
    bias = row_bias[structural_index, local_row] if use_bias else wp.float32(0.0)
    rhs[row] = -row_scale[row] * (relative_velocity + bias)


@wp.kernel(enable_backward=False)
def _apply_direct_equality_delta_kernel(
    body_row_start: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    joint_to_structural: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    delta: wp.array[wp.float32],
    row_scale: wp.array[wp.float32],
    bodies: BodyContainer,
):
    body = wp.tid()
    if body <= wp.int32(0) or bodies.inverse_mass[body] <= wp.float32(0.0):
        return
    wrench = wp.spatial_vector()
    for incidence in range(body_row_start[body], body_row_start[body + wp.int32(1)]):
        row = body_rows[incidence]
        joint = row_joint[row]
        structural_index = joint_to_structural[joint]
        local_row = row_local[row]
        wrench += (
            row_scale[row]
            * delta[row]
            * _row_wrench_for_body(
                body,
                joint,
                structural_index,
                local_row,
                joint_parent,
                joint_child,
                row_wrench0,
                row_wrench1,
            )
        )
    force = wp.spatial_top(wrench)
    bodies.velocity[body] += bodies.inverse_mass[body] * force
    bodies.angular_velocity[body] += mat33_from_sym6(bodies.inverse_inertia_world[body]) * wp.spatial_bottom(wrench)


def _effective_joint_axes(
    model: Model,
    joint_mode: np.ndarray,
    joint_dof_start: np.ndarray,
) -> np.ndarray:
    axes = np.zeros((int(model.joint_count), 3), dtype=np.float32)
    axes[:, 0] = 1.0
    model_axes = (
        np.asarray(model.joint_axis.numpy(), dtype=np.float32)
        if model.joint_axis is not None
        else np.empty((0, 3), dtype=np.float32)
    )
    qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int32)
    dof_dim = np.asarray(model.joint_dof_dim.numpy(), dtype=np.int32)
    lower = model.joint_limit_lower.numpy() if model.joint_limit_lower is not None else None
    upper = model.joint_limit_upper.numpy() if model.joint_limit_upper is not None else None

    for joint, mode in enumerate(joint_mode):
        if mode in (int(JOINT_MODE_REVOLUTE), int(JOINT_MODE_PRISMATIC)):
            dof = int(joint_dof_start[joint])
            if 0 <= dof < len(model_axes):
                axes[joint] = model_axes[dof]
        elif mode == int(JOINT_MODE_UNIVERSAL):
            start = int(qd_start[joint])
            linear_count = int(dof_dim[joint, 0])
            angular_count = int(dof_dim[joint, 1])
            locked_axis = -1
            if lower is not None and upper is not None:
                for axis in range(angular_count):
                    dof = start + linear_count + axis
                    if float(lower[dof]) > float(upper[dof]):
                        locked_axis = dof
                        break
            if locked_axis >= 0:
                axes[joint] = model_axes[locked_axis]
            elif angular_count == 2:
                axes[joint] = np.cross(model_axes[start], model_axes[start + 1])
        length = float(np.linalg.norm(axes[joint]))
        if length > 1.0e-12:
            axes[joint] /= length
        else:
            axes[joint] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return axes


class DirectEqualitySystem:
    """Batched direct equality systems, one dense block per mechanism."""

    def __init__(
        self,
        model: Model,
        bodies: BodyContainer,
        *,
        excluded_joint_mask: np.ndarray | None = None,
        effective_joint_mode: np.ndarray | None = None,
        effective_joint_dof_start: np.ndarray | None = None,
        regularization: float = 3.0e-6,
    ):
        self.model = model
        self.bodies = bodies
        joint_types = np.asarray(model.joint_type.numpy(), dtype=np.int32)
        joint_mode = (
            _default_joint_modes(joint_types)
            if effective_joint_mode is None
            else np.asarray(effective_joint_mode, dtype=np.int32)
        )
        joint_dof_start = (
            np.asarray(model.joint_qd_start.numpy(), dtype=np.int32)
            if effective_joint_dof_start is None
            else np.asarray(effective_joint_dof_start, dtype=np.int32)
        )
        joint_count = int(model.joint_count)
        if joint_mode.shape != (joint_count,) or joint_dof_start.shape != (joint_count,):
            raise ValueError("effective joint mode and DoF arrays must contain one entry per Newton joint")
        self.topology = build_direct_equality_topology(
            model,
            excluded_joint_mask=excluded_joint_mask,
            effective_joint_mode=joint_mode,
        )
        self.regularization = float(regularization)
        self.enabled = bool(self.topology.dimensions)
        if not self.enabled:
            return

        device = model.device
        structural_count = len(self.topology.joints)
        row_count = len(self.topology.row_joint)
        joint_to_structural = np.full(joint_count, -1, dtype=np.int32)
        joint_to_structural[self.topology.joints] = np.arange(structural_count, dtype=np.int32)

        self.structural_joints = wp.array(self.topology.joints, dtype=wp.int32, device=device)
        self.effective_joint_mode = wp.array(joint_mode, dtype=wp.int32, device=device)
        self.effective_joint_axis = wp.array(
            _effective_joint_axes(model, joint_mode, joint_dof_start),
            dtype=wp.vec3,
            device=device,
        )
        self.effective_joint_dof_start = wp.array(joint_dof_start, dtype=wp.int32, device=device)
        self.row_joint = wp.array(self.topology.row_joint, dtype=wp.int32, device=device)
        self.row_local = wp.array(self.topology.row_local, dtype=wp.int32, device=device)
        self.joint_to_structural = wp.array(joint_to_structural, dtype=wp.int32, device=device)
        self.body_row_start = wp.array(self.topology.body_row_start, dtype=wp.int32, device=device)
        self.body_rows = wp.array(self.topology.body_rows, dtype=wp.int32, device=device)
        self.row_count = wp.zeros(structural_count, dtype=wp.int32, device=device)
        self.row_wrench0 = wp.zeros((structural_count, _MAX_ROWS), dtype=wp.spatial_vector, device=device)
        self.row_wrench1 = wp.zeros((structural_count, _MAX_ROWS), dtype=wp.spatial_vector, device=device)
        self.row_bias = wp.zeros((structural_count, _MAX_ROWS), dtype=wp.float32, device=device)
        self.row_error = wp.zeros((structural_count, _MAX_ROWS), dtype=wp.float32, device=device)
        self.row_stiffness = wp.zeros((structural_count, _MAX_ROWS), dtype=wp.float32, device=device)
        self.row_damping = wp.zeros((structural_count, _MAX_ROWS), dtype=wp.float32, device=device)
        dof_count = max(1, int(model.joint_dof_count))
        self.joint_target_ke = (
            model.joint_target_ke
            if model.joint_target_ke is not None
            else wp.zeros(dof_count, dtype=wp.float32, device=device)
        )
        self.joint_target_kd = (
            model.joint_target_kd
            if model.joint_target_kd is not None
            else wp.zeros(dof_count, dtype=wp.float32, device=device)
        )

        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=list(self.topology.dimensions), dtype=wp.float32, device=device)
        matrix = wp.zeros(info.total_mat_size, dtype=wp.float32, device=device)
        self.operator = DenseLinearOperatorData(info=info, mat=matrix)
        self.rhs = wp.zeros(row_count, dtype=wp.float32, device=device)
        self.delta = wp.zeros(row_count, dtype=wp.float32, device=device)
        matrix_mechanism = np.zeros(1, dtype=np.int32)
        if len(self.topology.dimensions) > 1:
            matrix_mechanism = np.repeat(
                np.arange(len(self.topology.dimensions), dtype=np.int32),
                np.square(self.topology.dimensions),
            )
        self.matrix_mechanism = wp.array(
            matrix_mechanism,
            dtype=wp.int32,
            device=device,
        )
        matrix_offset = 0
        diagonal_index = []
        for dimension in self.topology.dimensions:
            diagonal_index.extend(matrix_offset + row * dimension + row for row in range(dimension))
            matrix_offset += dimension * dimension
        self.diagonal_index = wp.array(
            np.asarray(diagonal_index, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        self.row_scale = wp.ones(row_count, dtype=wp.float32, device=device)
        self.solver = LLTBlockedRCMSolver(
            operator=self.operator,
            block_size=16,
            reorder_tol=1.0e-12,
            parallel_factorization=max(self.topology.dimensions) >= 128,
            device=device,
        )
        permutation_wp = wp.array(self.topology.permutation, dtype=wp.int32, device=device)
        self.solver.set_permutation(permutation_wp)
        self.max_dimension = max(self.topology.dimensions)

    @property
    def joint_mask(self) -> np.ndarray:
        mask = np.zeros(int(self.model.joint_count), dtype=bool)
        if self.enabled:
            mask[self.topology.joints] = True
        return mask

    def prepare_and_factor(self, idt: wp.float32) -> None:
        if not self.enabled:
            return
        wp.launch(
            _prepare_direct_equality_rows_kernel,
            dim=len(self.topology.joints),
            inputs=[
                self.structural_joints,
                self.effective_joint_mode,
                self.effective_joint_axis,
                self.model.joint_parent,
                self.model.joint_child,
                self.effective_joint_dof_start,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.joint_target_ke,
                self.joint_target_kd,
                self.bodies,
                idt,
                self.row_count,
                self.row_wrench0,
                self.row_wrench1,
                self.row_bias,
                self.row_error,
                self.row_stiffness,
                self.row_damping,
            ],
            device=self.model.device,
        )
        wp.launch(
            _assemble_direct_equality_matrix_kernel,
            dim=self.operator.mat.size,
            inputs=[
                self.operator.info.dim,
                wp.int32(len(self.topology.dimensions)),
                self.matrix_mechanism,
                self.operator.info.mio,
                self.operator.info.vio,
                self.row_joint,
                self.row_local,
                self.joint_to_structural,
                self.model.joint_parent,
                self.model.joint_child,
                self.row_wrench0,
                self.row_wrench1,
                self.bodies,
                wp.float32(self.regularization),
                idt,
                self.row_error,
                self.row_stiffness,
                self.row_damping,
                self.row_bias,
                self.operator.mat,
            ],
            device=self.model.device,
        )
        wp.launch(
            _compute_direct_equality_row_scale_kernel,
            dim=len(self.topology.row_joint),
            inputs=[
                self.diagonal_index,
                self.operator.mat,
                self.row_scale,
            ],
            device=self.model.device,
        )
        wp.launch(
            _equilibrate_direct_equality_matrix_kernel,
            dim=self.operator.mat.size,
            inputs=[
                self.operator.info.dim,
                wp.int32(len(self.topology.dimensions)),
                self.matrix_mechanism,
                self.operator.info.mio,
                self.operator.info.vio,
                self.row_scale,
                self.operator.mat,
            ],
            device=self.model.device,
        )
        self.solver.compute(self.operator.mat)

    def solve(self, *, use_bias: bool) -> None:
        if not self.enabled:
            return
        wp.launch(
            _build_direct_equality_rhs_kernel,
            dim=len(self.topology.row_joint),
            inputs=[
                self.row_joint,
                self.row_local,
                self.joint_to_structural,
                self.model.joint_parent,
                self.model.joint_child,
                self.row_wrench0,
                self.row_wrench1,
                self.row_bias,
                self.bodies,
                wp.bool(use_bias),
                self.row_scale,
                self.rhs,
            ],
            device=self.model.device,
        )
        self.solver.solve(self.rhs, self.delta)
        wp.launch(
            _apply_direct_equality_delta_kernel,
            dim=int(self.model.body_count) + 1,
            inputs=[
                self.body_row_start,
                self.body_rows,
                self.row_joint,
                self.row_local,
                self.joint_to_structural,
                self.model.joint_parent,
                self.model.joint_child,
                self.row_wrench0,
                self.row_wrench1,
                self.delta,
                self.row_scale,
                self.bodies,
            ],
            device=self.model.device,
        )


__all__ = ["DirectEqualitySystem", "DirectEqualityTopology", "build_direct_equality_topology"]
