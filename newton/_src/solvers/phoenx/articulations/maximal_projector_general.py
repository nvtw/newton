# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""General mass-metric projection for maximal-coordinate articulation trees."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import warp as wp

from newton._src.sim import JointType, Model
from newton._src.solvers.phoenx.articulations.maximal_projector import _solve_spd6, _sync_warp
from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    read_float,
    read_int,
    read_vec3,
    write_float,
    write_vec3,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_ACC_IMP1,
    _OFF_ACC_IMP2,
    _OFF_ACC_IMP3,
    _OFF_ACC_LIMIT,
    _OFF_AXIS_WORLD,
    _OFF_BIAS1,
    _OFF_BIAS2,
    _OFF_BIAS3,
    _OFF_BIAS_LIMIT_BOX2D,
    _OFF_JOINT_MODE,
    _OFF_R1_B1,
    _OFF_R1_B2,
    _OFF_R2_B2,
    _OFF_R3_B2,
    _OFF_T1,
    _OFF_T2,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_FIXED,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)
from newton._src.solvers.phoenx.model_adapter import _classify_d6_legacy_mode, _is_locked_dof

_WARP_SIZE = 32


@wp.func
def _motion_column(motion: wp.spatial_matrixf, column: wp.int32):
    result = wp.spatial_vectorf(0.0)
    for row in range(6):
        result[row] = motion[row, column]
    return result


@wp.func
def _set_motion_column(
    motion: wp.spatial_matrixf,
    column: wp.int32,
    linear: wp.vec3f,
    angular: wp.vec3f,
):
    motion[0, column] = linear[0]
    motion[1, column] = linear[1]
    motion[2, column] = linear[2]
    motion[3, column] = angular[0]
    motion[4, column] = angular[1]
    motion[5, column] = angular[2]
    return motion


@wp.func
def _orthonormal_tangents(normal: wp.vec3f):
    seed = wp.vec3f(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > wp.float32(0.577350269):
        seed = wp.vec3f(0.0, 1.0, 0.0)
    tangent1 = wp.normalize(wp.cross(normal, seed))
    return tangent1, wp.cross(normal, tangent1)


@wp.func
def _reaction_map(
    lever2: wp.vec3f,
    lever3: wp.vec3f,
    tangent1: wp.vec3f,
    tangent2: wp.vec3f,
):
    result = wp.mat33f(0.0)
    column0 = wp.cross(lever2, tangent1)
    column1 = wp.cross(lever2, tangent2)
    column2 = wp.cross(lever3, tangent2)
    for row in range(3):
        result[row, 0] = column0[row]
        result[row, 1] = column1[row]
        result[row, 2] = column2[row]
    return result


@wp.func
def _locked_offset_three_anchor(
    target1: wp.vec3f,
    target2: wp.vec3f,
    target3: wp.float32,
    r1_child: wp.vec3f,
    r2_child: wp.vec3f,
    r3_child: wp.vec3f,
    tangent1: wp.vec3f,
    tangent2: wp.vec3f,
):
    mapping = _reaction_map(r2_child - r1_child, r3_child - r1_child, tangent1, tangent2)
    rhs = wp.vec3f(
        wp.dot(tangent1, target2 - target1),
        wp.dot(tangent2, target2 - target1),
        target3 - wp.dot(tangent2, target1),
    )
    angular = wp.inverse(wp.transpose(mapping)) @ rhs
    linear = target1 + wp.cross(r1_child, angular)
    return wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.struct
class GeneralMaximalTreeProjectorData:
    body_count: wp.array[wp.int32]
    max_depth: wp.array[wp.int32]
    floating_root: wp.array[wp.int32]
    joint_index: wp.array2d[wp.int32]
    body_slot: wp.array2d[wp.int32]
    depth: wp.array2d[wp.int32]
    parent: wp.array2d[wp.int32]
    child_start: wp.array2d[wp.int32]
    child_index: wp.array2d[wp.int32]
    dof_count: wp.array2d[wp.int32]
    transform: wp.array2d[wp.spatial_matrixf]
    motion: wp.array2d[wp.spatial_matrixf]
    affine_offset: wp.array2d[wp.spatial_vectorf]
    mass: wp.array2d[wp.spatial_matrixf]
    velocity_in: wp.array2d[wp.spatial_vectorf]
    articulated: wp.array2d[wp.spatial_matrixf]
    bias: wp.array2d[wp.spatial_vectorf]
    inverse_d: wp.array2d[wp.mat33f]
    parent_articulated: wp.array2d[wp.spatial_matrixf]
    parent_bias: wp.array2d[wp.spatial_vectorf]
    velocity_out: wp.array2d[wp.spatial_vectorf]
    reaction: wp.array2d[wp.spatial_vectorf]


@wp.func
def _gather_general_maximal_tree_thread(
    tid: wp.int32,
    use_bias: wp.bool,
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: GeneralMaximalTreeProjectorData,
):
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    if lane >= data.body_count[articulation]:
        return

    joint = data.joint_index[articulation, lane]
    body = data.body_slot[articulation, lane]
    linear = bodies.velocity[body]
    angular = bodies.angular_velocity[body]
    data.velocity_in[articulation, lane] = wp.spatial_vectorf(
        linear[0], linear[1], linear[2], angular[0], angular[1], angular[2]
    )
    if not use_bias:
        # PhoenX freezes inertia and prepared joint geometry until relax ends.
        data.affine_offset[articulation, lane] = wp.spatial_vectorf(0.0)
        return

    body_mass = wp.float32(1.0) / bodies.inverse_mass[body]
    inertia = wp.inverse(mat33_from_sym6(bodies.inverse_inertia_world[body]))
    spatial_mass = wp.spatial_matrixf(0.0)
    for row in range(3):
        spatial_mass[row, row] = body_mass
        for column in range(3):
            spatial_mass[row + wp.int32(3), column + wp.int32(3)] = inertia[row, column]
    data.mass[articulation, lane] = spatial_mass

    joint_transform = wp.spatial_matrixf(0.0)
    for diagonal in range(6):
        joint_transform[diagonal, diagonal] = wp.float32(1.0)
    joint_motion = wp.spatial_matrixf(0.0)
    affine_offset = wp.spatial_vectorf(0.0)
    dof_count = wp.int32(0)

    if data.floating_root[articulation] == wp.int32(0) or lane > wp.int32(0):
        cid = joint_to_cid[joint]
        mode = read_int(constraints, _OFF_JOINT_MODE, cid)
        r_parent = read_vec3(constraints, _OFF_R1_B1, cid)
        r_child = read_vec3(constraints, _OFF_R1_B2, cid)
        shift = wp.skew(r_child - r_parent)
        for row in range(3):
            for column in range(3):
                joint_transform[row, column + wp.int32(3)] = shift[row, column]

        axis = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
        if mode == JOINT_MODE_REVOLUTE:
            dof_count = wp.int32(1)
            joint_motion = _set_motion_column(joint_motion, wp.int32(0), wp.cross(r_child, axis), axis)
            if use_bias:
                bias1 = read_vec3(constraints, _OFF_BIAS1, cid)
                bias2 = read_vec3(constraints, _OFF_BIAS2, cid)
                tangent1 = read_vec3(constraints, _OFF_T1, cid)
                tangent2 = read_vec3(constraints, _OFF_T2, cid)
                target1 = -bias1
                target2 = -bias2[0] * tangent1 - bias2[1] * tangent2
                target1_tangent = wp.dot(target1, tangent1) * tangent1 + wp.dot(target1, tangent2) * tangent2
                tangent_delta = target2 - target1_tangent
                r2_child = read_vec3(constraints, _OFF_R2_B2, cid)
                lever_length = wp.dot(r2_child - r_child, axis)
                locked_angular = wp.vec3f(0.0, 0.0, 0.0)
                if wp.abs(lever_length) > wp.float32(1.0e-8):
                    locked_angular = wp.cross(axis, tangent_delta) / lever_length
                locked_linear = target1 + wp.cross(r_child, locked_angular)
                affine_offset = wp.spatial_vectorf(
                    locked_linear[0],
                    locked_linear[1],
                    locked_linear[2],
                    locked_angular[0],
                    locked_angular[1],
                    locked_angular[2],
                )
        elif mode == JOINT_MODE_PRISMATIC:
            dof_count = wp.int32(1)
            joint_motion = _set_motion_column(joint_motion, wp.int32(0), axis, wp.vec3f(0.0, 0.0, 0.0))
            if use_bias:
                tangent1 = read_vec3(constraints, _OFF_T1, cid)
                tangent2 = read_vec3(constraints, _OFF_T2, cid)
                bias1 = read_vec3(constraints, _OFF_BIAS1, cid)
                bias2 = read_vec3(constraints, _OFF_BIAS2, cid)
                target1 = -bias1[0] * tangent1 - bias1[1] * tangent2
                target2 = -bias2[0] * tangent1 - bias2[1] * tangent2
                affine_offset = _locked_offset_three_anchor(
                    target1,
                    target2,
                    -read_float(constraints, _OFF_BIAS3, cid),
                    r_child,
                    read_vec3(constraints, _OFF_R2_B2, cid),
                    read_vec3(constraints, _OFF_R3_B2, cid),
                    tangent1,
                    tangent2,
                )
        elif mode == JOINT_MODE_UNIVERSAL:
            dof_count = wp.int32(2)
            tangent1, tangent2 = _orthonormal_tangents(axis)
            joint_motion = _set_motion_column(joint_motion, wp.int32(0), wp.cross(r_child, tangent1), tangent1)
            joint_motion = _set_motion_column(joint_motion, wp.int32(1), wp.cross(r_child, tangent2), tangent2)
            if use_bias:
                locked_angular = read_float(constraints, _OFF_BIAS_LIMIT_BOX2D, cid) * axis
                target1 = -read_vec3(constraints, _OFF_BIAS1, cid)
                locked_linear = target1 + wp.cross(r_child, locked_angular)
                affine_offset = wp.spatial_vectorf(
                    locked_linear[0],
                    locked_linear[1],
                    locked_linear[2],
                    locked_angular[0],
                    locked_angular[1],
                    locked_angular[2],
                )
        elif mode == JOINT_MODE_BALL_SOCKET:
            dof_count = wp.int32(3)
            axis0 = wp.vec3f(1.0, 0.0, 0.0)
            axis1 = wp.vec3f(0.0, 1.0, 0.0)
            axis2 = wp.vec3f(0.0, 0.0, 1.0)
            joint_motion = _set_motion_column(joint_motion, wp.int32(0), wp.cross(r_child, axis0), axis0)
            joint_motion = _set_motion_column(joint_motion, wp.int32(1), wp.cross(r_child, axis1), axis1)
            joint_motion = _set_motion_column(joint_motion, wp.int32(2), wp.cross(r_child, axis2), axis2)
            if use_bias:
                target1 = -read_vec3(constraints, _OFF_BIAS1, cid)
                affine_offset = wp.spatial_vectorf(target1[0], target1[1], target1[2], 0.0, 0.0, 0.0)
        elif mode == JOINT_MODE_FIXED:
            if use_bias:
                tangent1 = read_vec3(constraints, _OFF_T1, cid)
                tangent2 = read_vec3(constraints, _OFF_T2, cid)
                bias2 = read_vec3(constraints, _OFF_BIAS2, cid)
                affine_offset = _locked_offset_three_anchor(
                    -read_vec3(constraints, _OFF_BIAS1, cid),
                    -bias2[0] * tangent1 - bias2[1] * tangent2,
                    -read_float(constraints, _OFF_BIAS3, cid),
                    r_child,
                    read_vec3(constraints, _OFF_R2_B2, cid),
                    read_vec3(constraints, _OFF_R3_B2, cid),
                    tangent1,
                    tangent2,
                )

    data.dof_count[articulation, lane] = dof_count
    data.transform[articulation, lane] = joint_transform
    data.motion[articulation, lane] = joint_motion
    data.affine_offset[articulation, lane] = affine_offset


@wp.func
def _project_general_maximal_tree_thread(tid: wp.int32, data: GeneralMaximalTreeProjectorData):
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    body_count = data.body_count[articulation]
    max_depth = data.max_depth[articulation]
    floating_root = data.floating_root[articulation] != wp.int32(0)

    if lane < body_count:
        body_mass = data.mass[articulation, lane]
        data.articulated[articulation, lane] = body_mass
        data.bias[articulation, lane] = body_mass @ data.velocity_in[articulation, lane]
        data.parent_articulated[articulation, lane] = wp.spatial_matrixf(0.0)
        data.parent_bias[articulation, lane] = wp.spatial_vectorf(0.0)
        data.inverse_d[articulation, lane] = wp.mat33f(0.0)
    _sync_warp()

    current_depth = max_depth
    while current_depth >= wp.int32(0):
        if lane < body_count and data.depth[articulation, lane] == current_depth:
            body_articulated = data.articulated[articulation, lane]
            body_bias = data.bias[articulation, lane]
            begin = data.child_start[articulation, lane]
            end = data.child_start[articulation, lane + wp.int32(1)]
            for cursor in range(begin, end):
                child = data.child_index[articulation, cursor]
                body_articulated += data.parent_articulated[articulation, child]
                body_bias += data.parent_bias[articulation, child]
            data.articulated[articulation, lane] = body_articulated
            data.bias[articulation, lane] = body_bias

            constrained = (not floating_root) or lane != wp.int32(0)
            if constrained:
                dof_count = data.dof_count[articulation, lane]
                motion = data.motion[articulation, lane]
                inverse_d = wp.mat33f(0.0)
                u_columns = wp.spatial_matrixf(0.0)
                d_matrix = wp.mat33f(0.0)
                for i in range(dof_count):
                    s_i = _motion_column(motion, i)
                    u_i = body_articulated @ s_i
                    for row in range(6):
                        u_columns[row, i] = u_i[row]
                    for j in range(i + 1):
                        value = wp.dot(_motion_column(motion, j), u_i)
                        d_matrix[j, i] = value
                        d_matrix[i, j] = value

                if dof_count == wp.int32(1):
                    inverse_d[0, 0] = wp.float32(1.0) / d_matrix[0, 0]
                elif dof_count == wp.int32(2):
                    determinant = d_matrix[0, 0] * d_matrix[1, 1] - d_matrix[0, 1] * d_matrix[0, 1]
                    inverse_d[0, 0] = d_matrix[1, 1] / determinant
                    inverse_d[0, 1] = -d_matrix[0, 1] / determinant
                    inverse_d[1, 0] = -d_matrix[0, 1] / determinant
                    inverse_d[1, 1] = d_matrix[0, 0] / determinant
                elif dof_count == wp.int32(3):
                    inverse_d = wp.inverse(d_matrix)
                data.inverse_d[articulation, lane] = inverse_d

                projected = body_articulated
                projected_bias = body_bias
                for i in range(dof_count):
                    u_i = _motion_column(u_columns, i)
                    bias_coefficient = wp.float32(0.0)
                    for j in range(dof_count):
                        coefficient = inverse_d[i, j]
                        bias_coefficient += coefficient * wp.dot(_motion_column(motion, j), body_bias)
                        projected -= coefficient * wp.outer(u_i, _motion_column(u_columns, j))
                    projected_bias -= bias_coefficient * u_i

                parent = data.parent[articulation, lane]
                if parent >= wp.int32(0):
                    joint_transform = data.transform[articulation, lane]
                    offset = data.affine_offset[articulation, lane]
                    data.parent_articulated[articulation, lane] = (
                        wp.transpose(joint_transform) @ projected @ joint_transform
                    )
                    data.parent_bias[articulation, lane] = wp.transpose(joint_transform) @ (
                        projected_bias - projected @ offset
                    )
        _sync_warp()
        current_depth -= wp.int32(1)

    if lane == wp.int32(0) and floating_root:
        data.velocity_out[articulation, lane] = _solve_spd6(
            data.articulated[articulation, lane], data.bias[articulation, lane]
        )
    _sync_warp()

    current_depth = wp.int32(0)
    while current_depth <= max_depth:
        if lane < body_count and data.depth[articulation, lane] == current_depth:
            if not (floating_root and lane == wp.int32(0)):
                parent = data.parent[articulation, lane]
                base = data.affine_offset[articulation, lane]
                if parent >= wp.int32(0):
                    base += data.transform[articulation, lane] @ data.velocity_out[articulation, parent]
                dof_count = data.dof_count[articulation, lane]
                motion = data.motion[articulation, lane]
                residual = data.bias[articulation, lane] - data.articulated[articulation, lane] @ base
                rhs = wp.vec3f(0.0, 0.0, 0.0)
                for i in range(dof_count):
                    rhs[i] = wp.dot(_motion_column(motion, i), residual)
                generalized_velocity = data.inverse_d[articulation, lane] @ rhs
                velocity = base
                for i in range(dof_count):
                    velocity += generalized_velocity[i] * _motion_column(motion, i)
                data.velocity_out[articulation, lane] = velocity
        _sync_warp()
        current_depth += wp.int32(1)

    current_depth = max_depth
    while current_depth >= wp.int32(0):
        if lane < body_count and data.depth[articulation, lane] == current_depth:
            impulse = data.mass[articulation, lane] @ (
                data.velocity_out[articulation, lane] - data.velocity_in[articulation, lane]
            )
            begin = data.child_start[articulation, lane]
            end = data.child_start[articulation, lane + wp.int32(1)]
            for cursor in range(begin, end):
                child = data.child_index[articulation, cursor]
                impulse += wp.transpose(data.transform[articulation, child]) @ data.reaction[articulation, child]
            data.reaction[articulation, lane] = impulse
        _sync_warp()
        current_depth -= wp.int32(1)


@wp.func
def _publish_general_maximal_tree_thread(
    tid: wp.int32,
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: GeneralMaximalTreeProjectorData,
):
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    if lane >= data.body_count[articulation]:
        return

    body = data.body_slot[articulation, lane]
    velocity = data.velocity_out[articulation, lane]
    bodies.velocity[body] = wp.vec3f(velocity[0], velocity[1], velocity[2])
    bodies.angular_velocity[body] = wp.vec3f(velocity[3], velocity[4], velocity[5])
    if data.floating_root[articulation] != wp.int32(0) and lane == wp.int32(0):
        return

    joint = data.joint_index[articulation, lane]
    cid = joint_to_cid[joint]
    mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    impulse = data.reaction[articulation, lane]
    linear = wp.vec3f(impulse[0], impulse[1], impulse[2])
    angular = wp.vec3f(impulse[3], impulse[4], impulse[5])
    r1 = read_vec3(constraints, _OFF_R1_B2, cid)

    lambda1 = linear
    lambda2 = wp.vec3f(0.0, 0.0, 0.0)
    lambda3 = wp.vec3f(0.0, 0.0, 0.0)
    if mode == JOINT_MODE_UNIVERSAL:
        axis = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
        locked_torque = angular - wp.cross(r1, linear)
        write_float(
            constraints,
            _OFF_ACC_LIMIT,
            cid,
            read_float(constraints, _OFF_ACC_LIMIT, cid) - wp.dot(axis, locked_torque),
        )
    elif mode == JOINT_MODE_REVOLUTE:
        r2 = read_vec3(constraints, _OFF_R2_B2, cid)
        axis = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
        torque_after_r1 = angular - wp.cross(r1, linear)
        lever_length = wp.dot(r2 - r1, axis)
        if wp.abs(lever_length) > wp.float32(1.0e-8):
            lambda2 = wp.cross(torque_after_r1, axis) / lever_length
        lambda1 = linear - lambda2
    elif mode == JOINT_MODE_FIXED or mode == JOINT_MODE_PRISMATIC:
        r2 = read_vec3(constraints, _OFF_R2_B2, cid)
        r3 = read_vec3(constraints, _OFF_R3_B2, cid)
        tangent1 = read_vec3(constraints, _OFF_T1, cid)
        tangent2 = read_vec3(constraints, _OFF_T2, cid)
        mapping = _reaction_map(r2 - r1, r3 - r1, tangent1, tangent2)
        coefficients = wp.inverse(mapping) @ (angular - wp.cross(r1, linear))
        lambda2 = coefficients[0] * tangent1 + coefficients[1] * tangent2
        lambda3 = coefficients[2] * tangent2
        lambda1 = linear - lambda2 - lambda3

    write_vec3(constraints, _OFF_ACC_IMP1, cid, read_vec3(constraints, _OFF_ACC_IMP1, cid) + lambda1)
    if mode != JOINT_MODE_BALL_SOCKET and mode != JOINT_MODE_UNIVERSAL:
        write_vec3(constraints, _OFF_ACC_IMP2, cid, read_vec3(constraints, _OFF_ACC_IMP2, cid) + lambda2)
    if mode == JOINT_MODE_FIXED or mode == JOINT_MODE_PRISMATIC:
        write_vec3(constraints, _OFF_ACC_IMP3, cid, read_vec3(constraints, _OFF_ACC_IMP3, cid) + lambda3)


@wp.kernel(enable_backward=False)
def _project_general_maximal_tree_fused_kernel(
    use_bias: wp.bool,
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: GeneralMaximalTreeProjectorData,
):
    tid = wp.tid()
    _gather_general_maximal_tree_thread(tid, use_bias, joint_to_cid, constraints, bodies, data)
    _sync_warp()
    _project_general_maximal_tree_thread(tid, data)
    _sync_warp()
    _publish_general_maximal_tree_thread(tid, joint_to_cid, constraints, bodies, data)


class GeneralMaximalTreeProjector:
    """Mass-metric projector for mixed rigid-joint trees."""

    _SUPPORTED_TYPES: ClassVar[set[int]] = {
        int(JointType.FIXED),
        int(JointType.REVOLUTE),
        int(JointType.PRISMATIC),
        int(JointType.BALL),
        int(JointType.D6),
    }

    @classmethod
    def supports_model(cls, model: Model) -> bool:
        """Return whether every articulation can use the general projector."""
        articulation_count = int(model.articulation_count)
        if articulation_count <= 0 or not model.device.is_cuda:
            return False
        starts = model.articulation_start.numpy()
        joint_articulation = model.joint_articulation.numpy()
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_enabled = model.joint_enabled.numpy() if model.joint_enabled is not None else None
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        limit_lower = model.joint_limit_lower.numpy()
        limit_upper = model.joint_limit_upper.numpy()
        armature = model.joint_armature.numpy()
        body_inv_mass = model.body_inv_mass.numpy()
        body_world = model.body_world.numpy()
        claimed_bodies: set[int] = set()

        for articulation in range(articulation_count):
            start = int(starts[articulation])
            end = int(starts[articulation + 1])
            owned_joints = [joint for joint in range(start, end) if int(joint_articulation[joint]) == articulation]
            if len(owned_joints) < 1 or len(owned_joints) > _WARP_SIZE:
                return False
            root_joint = owned_joints[0]
            root_type = int(joint_type[root_joint])
            floating_root = root_type == int(JointType.FREE)
            if not floating_root and root_type not in cls._SUPPORTED_TYPES:
                return False
            if int(joint_parent[root_joint]) >= 0:
                return False
            root = int(joint_child[root_joint])
            if (
                root < 0
                or root in claimed_bodies
                or not np.isfinite(body_inv_mass[root])
                or body_inv_mass[root] <= 0.0
                or (joint_enabled is not None and not bool(joint_enabled[root_joint]))
            ):
                return False

            bodies = {root}
            world = int(body_world[root])
            for joint in owned_joints:
                kind = int(joint_type[joint])
                if joint != root_joint and kind not in cls._SUPPORTED_TYPES:
                    return False
                qd_start = int(joint_qd_start[joint])
                dof_count = int(np.sum(joint_dof_dim[joint]))
                if kind == int(JointType.D6):
                    linear_count = int(joint_dof_dim[joint, 0])
                    angular_count = int(joint_dof_dim[joint, 1])
                    locked_linear = [
                        _is_locked_dof(limit_lower, limit_upper, qd_start + offset) for offset in range(linear_count)
                    ]
                    locked_angular = [
                        _is_locked_dof(limit_lower, limit_upper, qd_start + linear_count + offset)
                        for offset in range(angular_count)
                    ]
                    reduced_mode, _ = _classify_d6_legacy_mode(
                        linear_count,
                        angular_count,
                        locked_linear,
                        locked_angular,
                    )
                    if reduced_mode is None:
                        return False
                if kind != int(JointType.REVOLUTE) and np.any(armature[qd_start : qd_start + dof_count] > 0.0):
                    return False
                if joint == root_joint:
                    continue
                parent = int(joint_parent[joint])
                child = int(joint_child[joint])
                if (
                    parent not in bodies
                    or child in bodies
                    or child in claimed_bodies
                    or int(body_world[child]) != world
                    or not np.isfinite(body_inv_mass[child])
                    or body_inv_mass[child] <= 0.0
                    or (joint_enabled is not None and not bool(joint_enabled[joint]))
                ):
                    return False
                bodies.add(child)
            claimed_bodies.update(bodies)
        return True

    def __init__(
        self,
        model: Model,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        joint_to_cid: wp.array[wp.int32],
    ):
        if not self.supports_model(model):
            raise ValueError("general maximal tree projector received an unsupported model")
        self.model = model
        self.constraints = constraints
        self.bodies = bodies
        self.joint_to_cid = joint_to_cid
        self.articulation_count = int(model.articulation_count)
        self.launch_dim = self.articulation_count * _WARP_SIZE

        starts = model.articulation_start.numpy()
        joint_articulation = model.joint_articulation.numpy()
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_to_cid_np = joint_to_cid.numpy()
        shape = (self.articulation_count, _WARP_SIZE)
        body_count = np.zeros(self.articulation_count, dtype=np.int32)
        max_depth = np.zeros(self.articulation_count, dtype=np.int32)
        floating_root = np.zeros(self.articulation_count, dtype=np.int32)
        joint_index = np.full(shape, -1, dtype=np.int32)
        body_slot = np.full(shape, -1, dtype=np.int32)
        depth = np.full(shape, -1, dtype=np.int32)
        parent = np.full(shape, -1, dtype=np.int32)
        child_start = np.zeros((self.articulation_count, _WARP_SIZE + 1), dtype=np.int32)
        child_index = np.full(shape, -1, dtype=np.int32)

        for articulation in range(self.articulation_count):
            start = int(starts[articulation])
            end = int(starts[articulation + 1])
            owned_joints = [joint for joint in range(start, end) if int(joint_articulation[joint]) == articulation]
            count = len(owned_joints)
            body_count[articulation] = count
            root_joint = owned_joints[0]
            floating_root[articulation] = int(int(joint_type[root_joint]) == int(JointType.FREE))
            body_to_lane: dict[int, int] = {}
            children: list[list[int]] = [[] for _ in range(count)]
            for lane, joint in enumerate(owned_joints):
                child = int(joint_child[joint])
                joint_index[articulation, lane] = joint
                body_slot[articulation, lane] = child + 1
                body_to_lane[child] = lane
                if lane == 0:
                    depth[articulation, lane] = 0
                else:
                    cid = int(joint_to_cid_np[joint])
                    if cid < 0:
                        raise ValueError(f"projected joint {joint} has no maximal ADBS column")
                    parent_lane = body_to_lane[int(joint_parent[joint])]
                    parent[articulation, lane] = parent_lane
                    depth[articulation, lane] = depth[articulation, parent_lane] + 1
                    children[parent_lane].append(lane)
            if floating_root[articulation] == 0 and int(joint_to_cid_np[root_joint]) < 0:
                raise ValueError(f"projected root joint {root_joint} has no maximal ADBS column")
            max_depth[articulation] = int(np.max(depth[articulation, :count]))
            flat_children: list[int] = []
            for lane, lane_children in enumerate(children):
                flat_children.extend(lane_children)
                child_start[articulation, lane + 1] = len(flat_children)
            child_index[articulation, : len(flat_children)] = flat_children
            child_start[articulation, count + 1 :] = len(flat_children)

        device = model.device
        data = GeneralMaximalTreeProjectorData()
        data.body_count = wp.array(body_count, device=device)
        data.max_depth = wp.array(max_depth, device=device)
        data.floating_root = wp.array(floating_root, device=device)
        data.joint_index = wp.array(joint_index, device=device)
        data.body_slot = wp.array(body_slot, device=device)
        data.depth = wp.array(depth, device=device)
        data.parent = wp.array(parent, device=device)
        data.child_start = wp.array(child_start, device=device)
        data.child_index = wp.array(child_index, device=device)
        data.dof_count = wp.empty(shape, dtype=wp.int32, device=device)
        data.transform = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.motion = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.affine_offset = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.mass = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.velocity_in = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.inverse_d = wp.empty(shape, dtype=wp.mat33f, device=device)
        data.parent_articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.parent_bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.velocity_out = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.reaction = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        self.data = data

    def project(self, *, use_bias: bool) -> None:
        """Project body twists and publish recovered native-joint reactions."""
        wp.launch(
            _project_general_maximal_tree_fused_kernel,
            dim=self.launch_dim,
            block_dim=_WARP_SIZE,
            inputs=[use_bias, self.joint_to_cid, self.constraints, self.bodies, self.data],
            device=self.model.device,
        )


__all__ = ["GeneralMaximalTreeProjector", "GeneralMaximalTreeProjectorData"]
