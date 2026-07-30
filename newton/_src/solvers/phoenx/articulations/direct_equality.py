# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mechanism-wide maximal-coordinate equality solves for PhoenX."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.sim import JointTargetMode, JointType, Model
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import FixedPatternPanelLLT
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
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
)
from newton._src.solvers.phoenx.helpers.math_helpers import create_orthonormal

_MAX_ROWS = 6

# Independent rows retain the high-accuracy floor. Symbolically redundant
# mechanisms use a sqrt(epsilon)-scale constraint-force-mixing floor so their
# equilibrated systems remain inside the reliable FP32 condition range.
_FP32_BASE_REGULARIZATION = 3.0e-6
_FP32_RANK_REGULARIZATION = float(np.sqrt(np.finfo(np.float32).eps))
_DIRECT_BAUMGARTE = 0.2


@dataclass(frozen=True)
class DirectEqualityTopology:
    """Immutable host topology for the mechanism equality systems."""

    joints: np.ndarray
    row_joint: np.ndarray
    row_local: np.ndarray
    row_dynamic: np.ndarray
    dimensions: tuple[int, ...]
    permutation: np.ndarray
    mechanism_row_start: np.ndarray
    mechanism_requires_rank_floor: tuple[bool, ...]
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


def _dynamic_joint_masks(
    model: Model,
    joint_mode: np.ndarray,
    joint_dof_start: np.ndarray,
    excluded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scalar dynamics, direct-drive, and bounded-drive joint masks."""
    joint_count = int(model.joint_count)
    armature = np.asarray(model.joint_armature.numpy(), dtype=np.float32)
    damping = np.asarray(model.joint_damping.numpy(), dtype=np.float32)
    target_mode = np.asarray(model.joint_target_mode.numpy(), dtype=np.int32)
    target_ke = np.asarray(model.joint_target_ke.numpy(), dtype=np.float32)
    target_kd = np.asarray(model.joint_target_kd.numpy(), dtype=np.float32)
    effort = np.asarray(model.joint_effort_limit.numpy(), dtype=np.float32)
    gear = np.asarray(model.joint_gear.numpy(), dtype=np.float32)

    dynamic = np.zeros(joint_count, dtype=bool)
    direct_drive = np.zeros(joint_count, dtype=bool)
    bounded_drive = np.zeros(joint_count, dtype=bool)
    for joint, mode in enumerate(joint_mode):
        dof = int(joint_dof_start[joint])
        if (
            excluded[joint]
            or mode not in (int(JOINT_MODE_REVOLUTE), int(JOINT_MODE_PRISMATIC))
            or not 0 <= dof < len(armature)
        ):
            continue
        tm = int(target_mode[dof])
        drive_active = (
            tm in (int(JointTargetMode.POSITION), int(JointTargetMode.POSITION_VELOCITY))
            and float(target_ke[dof]) > 0.0
        ) or (tm == int(JointTargetMode.VELOCITY) and float(target_kd[dof]) > 0.0)
        reflected_limit = float(gear[dof]) * float(effort[dof])
        unlimited = reflected_limit == 0.0 or not np.isfinite(reflected_limit) or abs(reflected_limit) > 1.0e18
        direct_drive[joint] = drive_active
        bounded_drive[joint] = drive_active and not unlimited
        dynamic[joint] = float(armature[dof]) > 0.0 or float(damping[dof]) > 0.0 or direct_drive[joint]
    return dynamic, direct_drive, bounded_drive


def build_direct_equality_topology(
    model: Model,
    *,
    excluded_joint_mask: np.ndarray | None = None,
    effective_joint_mode: np.ndarray | None = None,
    dynamic_joint_mask: np.ndarray | None = None,
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
    dynamic_rows = (
        np.asarray(dynamic_joint_mask, dtype=bool)
        if dynamic_joint_mask is not None
        else np.zeros(joint_count, dtype=bool)
    )
    if dynamic_rows.shape != (joint_count,):
        raise ValueError(f"dynamic_joint_mask must have shape ({joint_count},), got {dynamic_rows.shape}")

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
    mechanism_requires_rank_floor: list[bool] = []
    for joints in ordered_mechanisms:
        dynamic_bodies: set[int] = set()
        endpoint_pairs: set[tuple[int, int]] = set()
        repeated_pair = False
        anchored = False
        structural_rows = 0
        for joint in joints:
            body0 = int(joint_parent[joint])
            body1 = int(joint_child[joint])
            dynamic_bodies.update(body for body in (body0, body1) if body >= 0 and dynamic[body])
            anchored = anchored or body0 < 0 or body1 < 0
            pair = (min(body0, body1), max(body0, body1))
            repeated_pair = repeated_pair or pair in endpoint_pairs
            endpoint_pairs.add(pair)
            structural_rows += _structural_row_count(int(joint_mode[joint]))
        # An anchored component has 6n generalized rigid-body directions; a
        # floating one has six free rigid modes. Repeated body pairs also imply
        # dependent rows for every supported structural joint combination.
        maximum_rank = max(0, 6 * len(dynamic_bodies) - (0 if anchored else 6))
        mechanism_requires_rank_floor.append(repeated_pair or structural_rows > maximum_rank)

    row_joint: list[int] = []
    row_local: list[int] = []
    row_dynamic: list[bool] = []
    mechanism_row_start = [0]
    ordered_joints: list[int] = []
    for joints in ordered_mechanisms:
        for joint in sorted(joints):
            ordered_joints.append(joint)
            for local_row in range(_structural_row_count(int(joint_mode[joint]))):
                row_joint.append(joint)
                row_local.append(local_row)
                row_dynamic.append(False)
            if dynamic_rows[joint]:
                row_joint.append(joint)
                row_local.append(5)
                row_dynamic.append(True)
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
        row_dynamic=np.asarray(row_dynamic, dtype=bool),
        dimensions=dimensions,
        permutation=np.asarray(permutation, dtype=np.int32),
        mechanism_row_start=np.asarray(mechanism_row_start, dtype=np.int32),
        mechanism_requires_rank_floor=tuple(mechanism_requires_rank_floor),
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
        point0_com = wp.quat_rotate(
            bodies.orientation[parent],
            wp.transform_get_translation(joint_x_p[joint]) - bodies.body_com[parent],
        )
    if child > wp.int32(0):
        point1_com = wp.quat_rotate(
            bodies.orientation[child],
            wp.transform_get_translation(joint_x_c[joint]) - bodies.body_com[child],
        )

    point_error = point1 - point0
    if parent > wp.int32(0) and child > wp.int32(0):
        point_error = bodies.position[child] - bodies.position[parent] + point1_com - point0_com
    elif child > wp.int32(0):
        point_error = bodies.position[child] - point0 + point1_com
    elif parent > wp.int32(0):
        point_error = point1 - bodies.position[parent] - point0_com

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
        _set_angular_row(
            structural_index,
            wp.int32(5),
            axis0,
            wp.float32(0.0),
            wp.float32(0.0),
            row_wrench0,
            row_wrench1,
            row_bias,
        )
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
        _set_direct_point_row(
            structural_index,
            wp.int32(5),
            point0_com,
            point1_com,
            axis0,
            wp.float32(0.0),
            wp.float32(0.0),
            row_wrench0,
            row_wrench1,
            row_bias,
        )

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
    matrix_row: wp.array[wp.int32],
    matrix_column: wp.array[wp.int32],
    matrix_storage: wp.array[wp.int32],
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    row_dynamic: wp.array[wp.bool],
    joint_to_structural: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    bodies: BodyContainer,
    row_regularization: wp.array[wp.float32],
    idt: wp.float32,
    row_error: wp.array2d[wp.float32],
    row_stiffness: wp.array2d[wp.float32],
    row_damping: wp.array2d[wp.float32],
    dynamic_mass: wp.array[wp.float32],
    row_bias: wp.array2d[wp.float32],
    matrix: wp.array[wp.float32],
):
    entry = wp.tid()
    row = matrix_row[entry]
    column = matrix_column[entry]
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

    if row == column:
        inverse_effective_mass = value
        if row_dynamic[row]:
            value += wp.float32(1.0) / wp.max(dynamic_mass[row], wp.float32(1.0e-10))
        else:
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
            row_regularization[row] * wp.max(inverse_effective_mass, wp.float32(1.0)),
            wp.float32(1.0e-10),
        )
    matrix[matrix_storage[entry]] = value


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
    matrix_row: wp.array[wp.int32],
    matrix_column: wp.array[wp.int32],
    matrix_storage: wp.array[wp.int32],
    row_scale: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
):
    entry = wp.tid()
    row = matrix_row[entry]
    column = matrix_column[entry]
    matrix[matrix_storage[entry]] *= row_scale[row] * row_scale[column]


@wp.kernel(enable_backward=False)
def _build_direct_equality_rhs_kernel(
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    row_dynamic: wp.array[wp.bool],
    joint_to_structural: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    row_bias: wp.array2d[wp.float32],
    velocity_reference: wp.array[wp.float32],
    accumulated_impulse: wp.array[wp.float32],
    dynamic_mass: wp.array[wp.float32],
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
    if row_dynamic[row]:
        rhs[row] = row_scale[row] * (
            velocity_reference[row]
            - relative_velocity
            - accumulated_impulse[row] / wp.max(dynamic_mass[row], wp.float32(1.0e-10))
        )
    else:
        bias = row_bias[structural_index, local_row] if use_bias else wp.float32(0.0)
        rhs[row] = -row_scale[row] * (relative_velocity + bias)


@wp.kernel(enable_backward=False)
def _snapshot_direct_dynamic_velocity_kernel(
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    row_dynamic: wp.array[wp.bool],
    row_direct_drive: wp.array[wp.bool],
    joint_to_structural: wp.array[wp.int32],
    effective_joint_mode: wp.array[wp.int32],
    effective_joint_axis: wp.array[wp.vec3],
    effective_joint_dof_start: wp.array[wp.int32],
    effective_joint_target_start: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    joint_armature: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    joint_gear: wp.array[wp.float32],
    joint_target_mode: wp.array[wp.int32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    control_target_q: wp.array[wp.float32],
    control_target_qd: wp.array[wp.float32],
    dt: wp.float32,
    bodies: BodyContainer,
    previous_coordinate: wp.array[wp.float32],
    coordinate_revolutions: wp.array[wp.int32],
    dynamic_mass: wp.array[wp.float32],
    dynamic_old_velocity: wp.array[wp.float32],
    dynamic_coordinate: wp.array[wp.float32],
    velocity_reference: wp.array[wp.float32],
    accumulated_impulse: wp.array[wp.float32],
    drive_saturated: wp.array[wp.bool],
):
    row = wp.tid()
    accumulated_impulse[row] = wp.float32(0.0)
    drive_saturated[row] = False
    if not row_dynamic[row]:
        dynamic_mass[row] = wp.float32(0.0)
        dynamic_old_velocity[row] = wp.float32(0.0)
        dynamic_coordinate[row] = wp.float32(0.0)
        velocity_reference[row] = wp.float32(0.0)
        return
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
    dynamic_old_velocity[row] = relative_velocity
    dynamic_coordinate[row] = wp.float32(0.0)

    dof = effective_joint_dof_start[joint]
    gear = joint_gear[dof]
    armature = joint_armature[dof] * gear * gear
    passive_damping = joint_damping[dof]
    mass = armature + dt * passive_damping
    momentum = armature * relative_velocity

    if row_direct_drive[row]:
        target_mode = joint_target_mode[dof]
        stiffness = joint_target_ke[dof]
        drive_damping = joint_target_kd[dof]
        target_position = control_target_q[effective_joint_target_start[joint]]
        target_velocity = control_target_qd[dof]
        if target_mode == JointTargetMode.POSITION:
            target_velocity = wp.float32(0.0)
        elif target_mode == JointTargetMode.VELOCITY:
            stiffness = wp.float32(0.0)
        elif target_mode == JointTargetMode.NONE or target_mode == JointTargetMode.EFFORT:
            stiffness = wp.float32(0.0)
            drive_damping = wp.float32(0.0)
            target_velocity = wp.float32(0.0)

        mode = effective_joint_mode[joint]
        x_wpj = _body_origin_transform(bodies, body0) * joint_x_p[joint]
        x_wcj = _body_origin_transform(bodies, body1) * joint_x_c[joint]
        point0 = wp.transform_get_translation(x_wpj)
        point1 = wp.transform_get_translation(x_wcj)
        q0 = wp.transform_get_rotation(x_wpj)
        q1 = wp.transform_get_rotation(x_wcj)
        axis = wp.normalize(wp.quat_rotate(q0, effective_joint_axis[joint]))
        coordinate = wp.float32(0.0)
        if mode == JOINT_MODE_REVOLUTE:
            wrapped = extract_rotation_angle(q1 * wp.quat_inverse(q0), axis)
            counter, previous = revolution_tracker_update(
                wrapped,
                coordinate_revolutions[row],
                previous_coordinate[row],
            )
            coordinate_revolutions[row] = counter
            previous_coordinate[row] = previous
            coordinate = revolution_tracker_angle(counter, previous)
        elif mode == JOINT_MODE_PRISMATIC:
            coordinate = wp.dot(axis, point1 - point0)
        dynamic_coordinate[row] = coordinate

        mass += dt * drive_damping + dt * dt * stiffness
        momentum += dt * (stiffness * (target_position - coordinate) + drive_damping * target_velocity)

    dynamic_mass[row] = wp.max(mass, wp.float32(1.0e-10))
    velocity_reference[row] = momentum / dynamic_mass[row]


@wp.kernel(enable_backward=False)
def _activate_bounded_direct_drives_kernel(
    row_joint: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    row_bounded_drive: wp.array[wp.bool],
    drive_saturated: wp.array[wp.bool],
    joint_to_structural: wp.array[wp.int32],
    effective_joint_dof_start: wp.array[wp.int32],
    effective_joint_target_start: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    row_wrench0: wp.array2d[wp.spatial_vector],
    row_wrench1: wp.array2d[wp.spatial_vector],
    joint_armature: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    joint_gear: wp.array[wp.float32],
    joint_target_mode: wp.array[wp.int32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_effort_limit: wp.array[wp.float32],
    control_target_q: wp.array[wp.float32],
    control_target_qd: wp.array[wp.float32],
    dt: wp.float32,
    bodies: BodyContainer,
    dynamic_old_velocity: wp.array[wp.float32],
    dynamic_coordinate: wp.array[wp.float32],
    dynamic_mass: wp.array[wp.float32],
    velocity_reference: wp.array[wp.float32],
    active_set_flag: wp.array[wp.int32],
):
    row = wp.tid()
    if not row_bounded_drive[row] or drive_saturated[row]:
        return

    joint = row_joint[row]
    structural_index = joint_to_structural[joint]
    local_row = row_local[row]
    body0 = joint_parent[joint] + wp.int32(1)
    body1 = joint_child[joint] + wp.int32(1)
    velocity = wp.float32(0.0)
    if body0 > wp.int32(0):
        velocity += wp.dot(row_wrench0[structural_index, local_row], _body_com_twist(bodies, body0))
    if body1 > wp.int32(0):
        velocity += wp.dot(row_wrench1[structural_index, local_row], _body_com_twist(bodies, body1))

    dof = effective_joint_dof_start[joint]
    target_mode = joint_target_mode[dof]
    stiffness = joint_target_ke[dof]
    drive_damping = joint_target_kd[dof]
    target_position = control_target_q[effective_joint_target_start[joint]]
    target_velocity = control_target_qd[dof]
    if target_mode == JointTargetMode.POSITION:
        target_velocity = wp.float32(0.0)
    elif target_mode == JointTargetMode.VELOCITY:
        stiffness = wp.float32(0.0)

    effort = stiffness * (target_position - (dynamic_coordinate[row] + dt * velocity)) + drive_damping * (
        target_velocity - velocity
    )
    gear = joint_gear[dof]
    effort_limit = wp.abs(gear * joint_effort_limit[dof])
    if wp.abs(effort) <= effort_limit:
        return

    saturated_effort = wp.clamp(effort, -effort_limit, effort_limit)
    armature = joint_armature[dof] * gear * gear
    mass = wp.max(
        armature + dt * joint_damping[dof],
        wp.float32(1.0e-10),
    )
    momentum = armature * dynamic_old_velocity[row] + dt * saturated_effort
    dynamic_mass[row] = mass
    velocity_reference[row] = momentum / mass
    drive_saturated[row] = True
    wp.atomic_max(active_set_flag, wp.int32(0), wp.int32(1))


@wp.kernel(enable_backward=False)
def _accumulate_direct_dynamic_impulse_kernel(
    row_dynamic: wp.array[wp.bool],
    delta: wp.array[wp.float32],
    row_scale: wp.array[wp.float32],
    accumulated_impulse: wp.array[wp.float32],
):
    row = wp.tid()
    if row_dynamic[row]:
        accumulated_impulse[row] += row_scale[row] * delta[row]


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
    """Batched fixed-pattern direct equality systems, one per mechanism."""

    def __init__(
        self,
        model: Model,
        bodies: BodyContainer,
        *,
        excluded_joint_mask: np.ndarray | None = None,
        effective_joint_mode: np.ndarray | None = None,
        effective_joint_dof_start: np.ndarray | None = None,
        effective_joint_target_start: np.ndarray | None = None,
        regularization: float = _FP32_BASE_REGULARIZATION,
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
        joint_target_start = (
            joint_dof_start
            if effective_joint_target_start is None
            else np.asarray(effective_joint_target_start, dtype=np.int32)
        )
        if joint_target_start.shape != (joint_count,):
            raise ValueError("effective joint target indices must contain one entry per Newton joint")
        excluded = (
            np.asarray(excluded_joint_mask, dtype=bool)
            if excluded_joint_mask is not None
            else np.zeros(joint_count, dtype=bool)
        )
        dynamic_joint_mask, direct_drive_joint_mask, bounded_drive_joint_mask = _dynamic_joint_masks(
            model,
            joint_mode,
            joint_dof_start,
            excluded,
        )
        self._excluded_joint_mask = excluded
        self._joint_mode_np = joint_mode.copy()
        self._joint_dof_start_np = joint_dof_start.copy()
        self._joint_target_start_np = joint_target_start.copy()
        self.dynamic_joint_mask = dynamic_joint_mask
        self.direct_drive_joint_mask = direct_drive_joint_mask
        self.bounded_drive_joint_mask = bounded_drive_joint_mask
        self.has_dynamic_rows = bool(np.any(dynamic_joint_mask))
        self.has_bounded_drives = bool(np.any(bounded_drive_joint_mask))
        self.topology = build_direct_equality_topology(
            model,
            excluded_joint_mask=excluded,
            effective_joint_mode=joint_mode,
            dynamic_joint_mask=dynamic_joint_mask,
        )
        self.regularization = float(regularization)
        row_regularization = np.full(len(self.topology.row_joint), self.regularization, dtype=np.float32)
        for mechanism, needs_rank_floor in enumerate(self.topology.mechanism_requires_rank_floor):
            if not needs_rank_floor:
                continue
            start = int(self.topology.mechanism_row_start[mechanism])
            end = int(self.topology.mechanism_row_start[mechanism + 1])
            mechanism_regularization = row_regularization[start:end]
            structural_rows = ~self.topology.row_dynamic[start:end]
            mechanism_regularization[structural_rows] = max(
                self.regularization,
                _FP32_RANK_REGULARIZATION,
            )
        self.row_regularization = wp.array(row_regularization, dtype=wp.float32, device=model.device)
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
        self.effective_joint_target_start = wp.array(joint_target_start, dtype=wp.int32, device=device)
        self.row_joint = wp.array(self.topology.row_joint, dtype=wp.int32, device=device)
        self.row_local = wp.array(self.topology.row_local, dtype=wp.int32, device=device)
        self.row_dynamic = wp.array(self.topology.row_dynamic, dtype=wp.bool, device=device)
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
        self.row_direct_drive = wp.array(
            direct_drive_joint_mask[self.topology.row_joint] & self.topology.row_dynamic,
            dtype=wp.bool,
            device=device,
        )
        self.row_bounded_drive = wp.array(
            bounded_drive_joint_mask[self.topology.row_joint] & self.topology.row_dynamic,
            dtype=wp.bool,
            device=device,
        )
        self.dynamic_mass = wp.zeros(row_count, dtype=wp.float32, device=device)
        self.dynamic_old_velocity = wp.zeros(row_count, dtype=wp.float32, device=device)
        self.dynamic_coordinate = wp.zeros(row_count, dtype=wp.float32, device=device)
        self.velocity_reference = wp.zeros(row_count, dtype=wp.float32, device=device)
        self.drive_active_set_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self.drive_saturated = wp.zeros(row_count, dtype=wp.bool, device=device)
        self.accumulated_impulse = wp.zeros(row_count, dtype=wp.float32, device=device)
        joint_q = np.asarray(model.joint_q.numpy(), dtype=np.float32)
        joint_q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int32)
        previous_coordinate = np.zeros(row_count, dtype=np.float32)
        coordinate_revolutions = np.zeros(row_count, dtype=np.int32)
        for row, joint in enumerate(self.topology.row_joint):
            if not self.topology.row_dynamic[row] or joint_mode[joint] != int(JOINT_MODE_REVOLUTE):
                continue
            q = float(joint_q[int(joint_q_start[joint])])
            turns = int(np.floor((q + np.pi) / (2.0 * np.pi)))
            previous_coordinate[row] = q - turns * (2.0 * np.pi)
            coordinate_revolutions[row] = turns
        self.previous_coordinate = wp.array(previous_coordinate, dtype=wp.float32, device=device)
        self.coordinate_revolutions = wp.array(coordinate_revolutions, dtype=wp.int32, device=device)
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
        self.joint_damping = (
            model.joint_damping
            if model.joint_damping is not None
            else wp.zeros(dof_count, dtype=wp.float32, device=device)
        )
        self.control_target_q = model.joint_target_q
        self.control_target_qd = model.joint_target_qd

        self.rhs = wp.zeros(row_count, dtype=wp.float32, device=device)
        self.delta = wp.zeros(row_count, dtype=wp.float32, device=device)
        inverse_mass = np.asarray(model.body_inv_mass.numpy(), dtype=np.float32)
        joint_parent = np.asarray(model.joint_parent.numpy(), dtype=np.int32)
        joint_child = np.asarray(model.joint_child.numpy(), dtype=np.int32)
        row_bodies = tuple(
            frozenset(
                body
                for body in (int(joint_parent[joint]), int(joint_child[joint]))
                if body >= 0 and inverse_mass[body] > 0.0
            )
            for joint in self.topology.row_joint
        )
        self.solver = FixedPatternPanelLLT(
            self.topology.dimensions,
            self.topology.mechanism_row_start,
            self.topology.permutation,
            row_bodies,
            block_size=16,
            device=device,
        )
        symbolic = self.solver.symbolic
        self.matrix_row = wp.array(symbolic.matrix_row, dtype=wp.int32, device=device)
        self.matrix_column = wp.array(symbolic.matrix_column, dtype=wp.int32, device=device)
        self.matrix_storage = wp.array(symbolic.matrix_storage, dtype=wp.int32, device=device)
        self.diagonal_index = wp.array(symbolic.diagonal_storage, dtype=wp.int32, device=device)
        self.matrix = self.solver.matrix
        self.row_scale = wp.ones(row_count, dtype=wp.float32, device=device)
        self.max_dimension = max(self.topology.dimensions)

    @property
    def joint_mask(self) -> np.ndarray:
        mask = np.zeros(int(self.model.joint_count), dtype=bool)
        if self.enabled:
            mask[self.topology.joints] = True
        return mask

    def refresh_joint_properties(self) -> None:
        """Rebuild topology only when scalar dynamics rows change ownership."""
        dynamic_joint_mask, direct_drive_joint_mask, bounded_drive_joint_mask = _dynamic_joint_masks(
            self.model,
            self._joint_mode_np,
            self._joint_dof_start_np,
            self._excluded_joint_mask,
        )
        if (
            np.array_equal(dynamic_joint_mask, self.dynamic_joint_mask)
            and np.array_equal(
                direct_drive_joint_mask,
                self.direct_drive_joint_mask,
            )
            and np.array_equal(bounded_drive_joint_mask, self.bounded_drive_joint_mask)
        ):
            return
        self.__init__(
            self.model,
            self.bodies,
            excluded_joint_mask=self._excluded_joint_mask,
            effective_joint_mode=self._joint_mode_np,
            effective_joint_dof_start=self._joint_dof_start_np,
            effective_joint_target_start=self._joint_target_start_np,
            regularization=self.regularization,
        )

    def set_control_targets(
        self,
        target_q: wp.array[wp.float32],
        target_qd: wp.array[wp.float32],
    ) -> None:
        """Select the control arrays read by subsequent direct-drive solves."""
        self.control_target_q = target_q
        self.control_target_qd = target_qd

    def begin_substep(self, idt: wp.float32) -> None:
        """Prepare body-space rows and snapshot pre-force joint velocities."""
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
                wp.float32(_DIRECT_BAUMGARTE) * idt,
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
        if self.has_dynamic_rows:
            wp.launch(
                _snapshot_direct_dynamic_velocity_kernel,
                dim=len(self.topology.row_joint),
                inputs=[
                    self.row_joint,
                    self.row_local,
                    self.row_dynamic,
                    self.row_direct_drive,
                    self.joint_to_structural,
                    self.effective_joint_mode,
                    self.effective_joint_axis,
                    self.effective_joint_dof_start,
                    self.effective_joint_target_start,
                    self.model.joint_parent,
                    self.model.joint_child,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    self.row_wrench0,
                    self.row_wrench1,
                    self.model.joint_armature,
                    self.joint_damping,
                    self.model.joint_gear,
                    self.model.joint_target_mode,
                    self.joint_target_ke,
                    self.joint_target_kd,
                    self.control_target_q,
                    self.control_target_qd,
                    wp.float32(1.0) / idt,
                    self.bodies,
                    self.previous_coordinate,
                    self.coordinate_revolutions,
                    self.dynamic_mass,
                    self.dynamic_old_velocity,
                    self.dynamic_coordinate,
                    self.velocity_reference,
                    self.accumulated_impulse,
                    self.drive_saturated,
                ],
                device=self.model.device,
            )

    def resolve_bounded_drives(self, idt: wp.float32, *, use_bias: bool) -> None:
        """Activate finite effort bounds and correct the direct solution."""
        if not self.enabled or not self.has_bounded_drives:
            return
        dt = wp.float32(1.0) / idt

        def find_new_bounds() -> None:
            self.drive_active_set_flag.zero_()
            wp.launch(
                _activate_bounded_direct_drives_kernel,
                dim=len(self.topology.row_joint),
                inputs=[
                    self.row_joint,
                    self.row_local,
                    self.row_bounded_drive,
                    self.drive_saturated,
                    self.joint_to_structural,
                    self.effective_joint_dof_start,
                    self.effective_joint_target_start,
                    self.model.joint_parent,
                    self.model.joint_child,
                    self.row_wrench0,
                    self.row_wrench1,
                    self.model.joint_armature,
                    self.joint_damping,
                    self.model.joint_gear,
                    self.model.joint_target_mode,
                    self.joint_target_ke,
                    self.joint_target_kd,
                    self.model.joint_effort_limit,
                    self.control_target_q,
                    self.control_target_qd,
                    dt,
                    self.bodies,
                    self.dynamic_old_velocity,
                    self.dynamic_coordinate,
                    self.dynamic_mass,
                    self.velocity_reference,
                    self.drive_active_set_flag,
                ],
                device=self.model.device,
            )

        def activate_and_correct(remaining_passes: int) -> None:
            self.prepare_and_factor(idt)
            self.solve(use_bias=use_bias)
            if remaining_passes > 0:
                find_new_bounds()
                wp.capture_if(
                    self.drive_active_set_flag,
                    on_true=lambda: activate_and_correct(remaining_passes - 1),
                    on_false=None,
                )

        # A newly clamped drive can expose another bound in a coupled
        # mechanism. Nested conditionals keep the common unsaturated path to
        # one row scan and only perform extra scans after an actual refactor.
        find_new_bounds()
        wp.capture_if(
            self.drive_active_set_flag,
            on_true=lambda: activate_and_correct(3),
            on_false=None,
        )

    def prepare_and_factor(self, idt: wp.float32) -> None:
        if not self.enabled:
            return
        wp.launch(
            _assemble_direct_equality_matrix_kernel,
            dim=self.matrix_storage.size,
            inputs=[
                self.matrix_row,
                self.matrix_column,
                self.matrix_storage,
                self.row_joint,
                self.row_local,
                self.row_dynamic,
                self.joint_to_structural,
                self.model.joint_parent,
                self.model.joint_child,
                self.row_wrench0,
                self.row_wrench1,
                self.bodies,
                self.row_regularization,
                idt,
                self.row_error,
                self.row_stiffness,
                self.row_damping,
                self.dynamic_mass,
                self.row_bias,
                self.matrix,
            ],
            device=self.model.device,
        )
        wp.launch(
            _compute_direct_equality_row_scale_kernel,
            dim=len(self.topology.row_joint),
            inputs=[
                self.diagonal_index,
                self.matrix,
                self.row_scale,
            ],
            device=self.model.device,
        )
        wp.launch(
            _equilibrate_direct_equality_matrix_kernel,
            dim=self.matrix_storage.size,
            inputs=[
                self.matrix_row,
                self.matrix_column,
                self.matrix_storage,
                self.row_scale,
                self.matrix,
            ],
            device=self.model.device,
        )
        self.solver.compute()

    def solve(self, *, use_bias: bool) -> None:
        if not self.enabled:
            return
        wp.launch(
            _build_direct_equality_rhs_kernel,
            dim=len(self.topology.row_joint),
            inputs=[
                self.row_joint,
                self.row_local,
                self.row_dynamic,
                self.joint_to_structural,
                self.model.joint_parent,
                self.model.joint_child,
                self.row_wrench0,
                self.row_wrench1,
                self.row_bias,
                self.velocity_reference,
                self.accumulated_impulse,
                self.dynamic_mass,
                self.bodies,
                wp.bool(use_bias),
                self.row_scale,
                self.rhs,
            ],
            device=self.model.device,
        )
        self.solver.solve(self.rhs, self.delta)
        if self.has_dynamic_rows:
            wp.launch(
                _accumulate_direct_dynamic_impulse_kernel,
                dim=len(self.topology.row_joint),
                inputs=[
                    self.row_dynamic,
                    self.delta,
                    self.row_scale,
                    self.accumulated_impulse,
                ],
                device=self.model.device,
            )
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
