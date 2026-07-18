# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact joint-tree projection for maximal-coordinate robot bodies."""

from __future__ import annotations

import os

import numpy as np
import warp as wp

from newton._src.sim import JointType, Model
from newton._src.solvers.phoenx.body import (
    BodyContainer,
    inertia_sym6,
    mat33_from_sym6,
    sym6_from_mat33,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_read_multiplier_vec3,
    constraint_write_multiplier_vec3,
    read_float,
    read_int,
    read_vec3,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _MUL_ACC_IMP1,
    _MUL_ACC_IMP2,
    _OFF_AXIS_WORLD,
    _OFF_BIAS1,
    _OFF_BIAS2,
    _OFF_DAMPING_DRIVE,
    _OFF_DRIVE_MODE,
    _OFF_LA1_B1,
    _OFF_LA1_B2,
    _OFF_LA2_B1,
    _OFF_LA2_B2,
    _OFF_PREVIOUS_QUATERNION_ANGLE,
    _OFF_R1_B1,
    _OFF_R1_B2,
    _OFF_R2_B2,
    _OFF_REVOLUTION_COUNTER,
    _OFF_STIFFNESS_DRIVE,
    _OFF_T1,
    _OFF_T2,
    _OFF_TARGET,
    _OFF_TARGET_VELOCITY,
    DRIVE_MODE_POSITION,
)
from newton._src.solvers.phoenx.helpers.math_helpers import revolution_tracker_angle
from newton._src.solvers.phoenx.solver_phoenx_kernels import _rotation_quaternion

_WARP_SIZE = 32

# See maximal_projector_general.py: opt-in exact implicit-PD drive.
_PHOENX_MAXIMAL_IMPLICIT_DRIVE = os.environ.get("PHOENX_MAXIMAL_IMPLICIT_DRIVE", "0").lower() not in (
    "0",
    "",
    "false",
    "off",
)
_SYNC_WARP_CUDA = """__syncwarp();"""


@wp.func_native(_SYNC_WARP_CUDA)
def _sync_warp(): ...


@wp.func
def _solve_spd6(a: wp.spatial_matrixf, b: wp.spatial_vectorf):
    lower = wp.spatial_matrixf(0.0)
    diagonal = wp.spatial_vectorf(0.0)
    for i in range(6):
        value = a[i, i]
        for k in range(i):
            value -= lower[i, k] * lower[i, k] * diagonal[k]
        diagonal[i] = value
        lower[i, i] = wp.float32(1.0)
        for j in range(i + 1, 6):
            value = a[j, i]
            for k in range(i):
                value -= lower[j, k] * lower[i, k] * diagonal[k]
            lower[j, i] = value / diagonal[i]

    forward = wp.spatial_vectorf(0.0)
    for i in range(6):
        value = b[i]
        for k in range(i):
            value -= lower[i, k] * forward[k]
        forward[i] = value

    scaled = wp.spatial_vectorf(0.0)
    for i in range(6):
        scaled[i] = forward[i] / diagonal[i]

    result = wp.spatial_vectorf(0.0)
    for reverse_i in range(6):
        i = wp.int32(5) - reverse_i
        value = scaled[i]
        for k in range(i + 1, 6):
            value -= lower[k, i] * result[k]
        result[i] = value
    return result


@wp.func
def _make_spatial_mass(body_mass: wp.float32, body_inertia: inertia_sym6):
    result = wp.spatial_matrixf(0.0)
    inertia = mat33_from_sym6(body_inertia)
    for row in range(3):
        result[row, row] = body_mass
        for column in range(3):
            result[row + wp.int32(3), column + wp.int32(3)] = inertia[row, column]
    return result


@wp.func
def _make_spatial_shift_transform(shift: wp.vec3f):
    result = wp.spatial_matrixf(0.0)
    shift_matrix = wp.skew(shift)
    for row in range(6):
        result[row, row] = wp.float32(1.0)
    for row in range(3):
        for column in range(3):
            result[row, column + wp.int32(3)] = shift_matrix[row, column]
    return result


@wp.struct
class MaximalTreeProjectorData:
    body_count: wp.array[wp.int32]
    max_depth: wp.array[wp.int32]
    joint_index: wp.array2d[wp.int32]
    body_slot: wp.array2d[wp.int32]
    depth: wp.array2d[wp.int32]
    parent: wp.array2d[wp.int32]
    child_start: wp.array2d[wp.int32]
    child_index: wp.array2d[wp.int32]
    shift: wp.array2d[wp.vec3f]
    motion: wp.array2d[wp.spatial_vectorf]
    affine_offset: wp.array2d[wp.spatial_vectorf]
    body_mass: wp.array2d[wp.float32]
    body_inertia: wp.array2d[inertia_sym6]
    velocity_in: wp.array2d[wp.spatial_vectorf]
    articulated: wp.array2d[wp.spatial_matrixf]
    bias: wp.array2d[wp.spatial_vectorf]
    inverse_d: wp.array2d[wp.float32]
    parent_articulated: wp.array2d[wp.spatial_matrixf]
    parent_bias: wp.array2d[wp.spatial_vectorf]
    velocity_out: wp.array2d[wp.spatial_vectorf]
    reaction: wp.array2d[wp.spatial_vectorf]
    drive_diag: wp.array2d[wp.float32]
    drive_bias: wp.array2d[wp.float32]


@wp.kernel(enable_backward=False)
def _gather_maximal_tree_kernel(
    use_bias: wp.bool,
    implicit_drive: wp.bool,
    dt: wp.float32,
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: MaximalTreeProjectorData,
):
    tid = wp.tid()
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
    data.drive_diag[articulation, lane] = wp.float32(0.0)
    data.drive_bias[articulation, lane] = wp.float32(0.0)
    if not use_bias:
        # PhoenX freezes inertia and prepared joint geometry until relax ends.
        data.affine_offset[articulation, lane] = wp.spatial_vectorf(0.0)
        return

    inverse_mass = bodies.inverse_mass[body]
    body_mass = wp.float32(1.0) / inverse_mass
    inertia = wp.inverse(mat33_from_sym6(bodies.inverse_inertia_world[body]))
    data.body_mass[articulation, lane] = body_mass
    data.body_inertia[articulation, lane] = sym6_from_mat33(inertia)

    joint_shift = wp.vec3f(0.0)
    joint_motion = wp.spatial_vectorf(0.0)
    affine_offset = wp.spatial_vectorf(0.0)
    if lane > wp.int32(0):
        cid = joint_to_cid[joint]
        r_parent = read_vec3(constraints, _OFF_R1_B1, cid)
        r_child = read_vec3(constraints, _OFF_R1_B2, cid)
        joint_shift = r_child - r_parent
        axis = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
        linear_motion = wp.cross(r_child, axis)
        joint_motion = wp.spatial_vectorf(
            linear_motion[0], linear_motion[1], linear_motion[2], axis[0], axis[1], axis[2]
        )

        if use_bias:
            bias1 = read_vec3(constraints, _OFF_BIAS1, cid)
            bias2 = read_vec3(constraints, _OFF_BIAS2, cid)
            tangent1 = read_vec3(constraints, _OFF_T1, cid)
            tangent2 = read_vec3(constraints, _OFF_T2, cid)
            target1 = -bias1
            target2_tangent = -bias2[0] * tangent1 - bias2[1] * tangent2
            target1_tangent = wp.dot(target1, tangent1) * tangent1 + wp.dot(target1, tangent2) * tangent2
            tangent_delta = target2_tangent - target1_tangent
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

        if implicit_drive and read_int(constraints, _OFF_DRIVE_MODE, cid) == DRIVE_MODE_POSITION:
            # Exact implicit-PD drive; see maximal_projector_general.py.
            ke = read_float(constraints, _OFF_STIFFNESS_DRIVE, cid)
            kd = read_float(constraints, _OFF_DAMPING_DRIVE, cid)
            target_q = read_float(constraints, _OFF_TARGET, cid)
            target_qd = read_float(constraints, _OFF_TARGET_VELOCITY, cid)
            counter = read_int(constraints, _OFF_REVOLUTION_COUNTER, cid)
            prev = read_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
            cumulative_angle = revolution_tracker_angle(counter, prev)
            data.drive_diag[articulation, lane] = dt * kd + dt * dt * ke
            data.drive_bias[articulation, lane] = dt * (ke * (target_q - cumulative_angle) + kd * target_qd)

    data.shift[articulation, lane] = joint_shift
    data.motion[articulation, lane] = joint_motion
    data.affine_offset[articulation, lane] = affine_offset


@wp.func
def _gather_position_lane(
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: MaximalTreeProjectorData,
    articulation: wp.int32,
    lane: wp.int32,
):
    """Gather pass for the position-level projection (one lane, ``lane < body_count``).

    Like :func:`_gather_maximal_tree_kernel` with ``use_bias=True`` except the
    input twist is zero and the affine targets are the FULL position errors
    recomputed at the CURRENT (post-integrate) pose from body-local anchors —
    the prepared ``r1_b1`` / ``bias1`` fields are stale after integration.
    """
    joint = data.joint_index[articulation, lane]
    body = data.body_slot[articulation, lane]
    data.velocity_in[articulation, lane] = wp.spatial_vectorf(0.0)

    inverse_mass = bodies.inverse_mass[body]
    body_mass = wp.float32(1.0) / inverse_mass
    inertia = wp.inverse(mat33_from_sym6(bodies.inverse_inertia_world[body]))
    data.body_mass[articulation, lane] = body_mass
    data.body_inertia[articulation, lane] = sym6_from_mat33(inertia)

    joint_shift = wp.vec3f(0.0)
    joint_motion = wp.spatial_vectorf(0.0)
    affine_offset = wp.spatial_vectorf(0.0)
    if lane > wp.int32(0):
        cid = joint_to_cid[joint]
        parent_body = data.body_slot[articulation, data.parent[articulation, lane]]
        orientation1 = bodies.orientation[parent_body]
        orientation2 = bodies.orientation[body]
        position1 = bodies.position[parent_body]
        position2 = bodies.position[body]
        r1_b1 = wp.quat_rotate(orientation1, read_vec3(constraints, _OFF_LA1_B1, cid))
        r1_b2 = wp.quat_rotate(orientation2, read_vec3(constraints, _OFF_LA1_B2, cid))
        r2_b1 = wp.quat_rotate(orientation1, read_vec3(constraints, _OFF_LA2_B1, cid))
        r2_b2 = wp.quat_rotate(orientation2, read_vec3(constraints, _OFF_LA2_B2, cid))
        p1_b1 = position1 + r1_b1
        p1_b2 = position2 + r1_b2
        p2_b1 = position1 + r2_b1
        p2_b2 = position2 + r2_b2

        # Current-pose hinge axis: same anchor-pair construction the joint
        # prepare pass uses for ``axis_world``.
        hinge_vec = p2_b2 - p1_b2
        hinge_len2 = wp.dot(hinge_vec, hinge_vec)
        axis = wp.vec3f(1.0, 0.0, 0.0)
        if hinge_len2 > wp.float32(1.0e-20):
            axis = hinge_vec / wp.sqrt(hinge_len2)

        joint_shift = r1_b2 - r1_b1
        linear_motion = wp.cross(r1_b2, axis)
        joint_motion = wp.spatial_vectorf(
            linear_motion[0], linear_motion[1], linear_motion[2], axis[0], axis[1], axis[2]
        )

        # Full anchor errors — no Baumgarte scaling / cap.
        target1 = p1_b1 - p1_b2
        drift2 = p2_b2 - p2_b1
        target2_tangent = wp.dot(drift2, axis) * axis - drift2
        target1_tangent = target1 - wp.dot(target1, axis) * axis
        tangent_delta = target2_tangent - target1_tangent
        lever_length = wp.dot(r2_b2 - r1_b2, axis)
        locked_angular = wp.vec3f(0.0, 0.0, 0.0)
        if wp.abs(lever_length) > wp.float32(1.0e-8):
            locked_angular = wp.cross(axis, tangent_delta) / lever_length
        locked_linear = target1 + wp.cross(r1_b2, locked_angular)
        affine_offset = wp.spatial_vectorf(
            locked_linear[0],
            locked_linear[1],
            locked_linear[2],
            locked_angular[0],
            locked_angular[1],
            locked_angular[2],
        )

    data.shift[articulation, lane] = joint_shift
    data.motion[articulation, lane] = joint_motion
    data.affine_offset[articulation, lane] = affine_offset


@wp.func
def _project_tree_velocities(
    data: MaximalTreeProjectorData,
    articulation: wp.int32,
    lane: wp.int32,
    body_count: wp.int32,
    max_depth: wp.int32,
    implicit_drive: wp.bool,
):
    """Warp-cooperative tree solve: articulated inertia recursion to ``velocity_out``.

    All 32 lanes of the articulation's warp must call this together — the
    ``lane < body_count`` guards sit INSIDE the depth loops with the warp syncs
    outside so every lane reaches every :func:`_sync_warp`.
    """
    if lane < body_count:
        spatial_mass = _make_spatial_mass(data.body_mass[articulation, lane], data.body_inertia[articulation, lane])
        data.articulated[articulation, lane] = spatial_mass
        data.bias[articulation, lane] = spatial_mass @ data.velocity_in[articulation, lane]
        data.parent_articulated[articulation, lane] = wp.spatial_matrixf(0.0)
        data.parent_bias[articulation, lane] = wp.spatial_vectorf(0.0)
        data.inverse_d[articulation, lane] = wp.float32(0.0)
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
            if lane != wp.int32(0):
                joint_motion = data.motion[articulation, lane]
                drive_diag = wp.float32(0.0)
                drive_bias = wp.float32(0.0)
                if implicit_drive:
                    drive_diag = data.drive_diag[articulation, lane]
                    drive_bias = data.drive_bias[articulation, lane]
                u = body_articulated @ joint_motion
                reciprocal_d = wp.float32(1.0) / (wp.dot(joint_motion, u) + drive_diag)
                data.inverse_d[articulation, lane] = reciprocal_d
                projected = body_articulated - reciprocal_d * wp.outer(u, u)
                projected_bias = body_bias - reciprocal_d * (wp.dot(joint_motion, body_bias) + drive_bias) * u
                joint_transform = _make_spatial_shift_transform(data.shift[articulation, lane])
                offset = data.affine_offset[articulation, lane]
                data.parent_articulated[articulation, lane] = (
                    wp.transpose(joint_transform) @ projected @ joint_transform
                )
                data.parent_bias[articulation, lane] = wp.transpose(joint_transform) @ (
                    projected_bias - projected @ offset
                )
        _sync_warp()
        current_depth -= wp.int32(1)

    if lane == wp.int32(0):
        data.velocity_out[articulation, lane] = _solve_spd6(
            data.articulated[articulation, lane], data.bias[articulation, lane]
        )
    _sync_warp()

    current_depth = wp.int32(1)
    while current_depth <= max_depth:
        if lane < body_count and data.depth[articulation, lane] == current_depth:
            parent = data.parent[articulation, lane]
            joint_transform = _make_spatial_shift_transform(data.shift[articulation, lane])
            base = joint_transform @ data.velocity_out[articulation, parent] + data.affine_offset[articulation, lane]
            joint_motion = data.motion[articulation, lane]
            drive_bias = wp.float32(0.0)
            if implicit_drive:
                drive_bias = data.drive_bias[articulation, lane]
            joint_velocity = data.inverse_d[articulation, lane] * (
                wp.dot(
                    joint_motion,
                    data.bias[articulation, lane] - data.articulated[articulation, lane] @ base,
                )
                + drive_bias
            )
            data.velocity_out[articulation, lane] = base + joint_velocity * joint_motion
        _sync_warp()
        current_depth += wp.int32(1)


@wp.func
def _project_tree_reactions(
    data: MaximalTreeProjectorData,
    articulation: wp.int32,
    lane: wp.int32,
    body_count: wp.int32,
    max_depth: wp.int32,
):
    """Recover per-joint reaction impulses from ``velocity_out`` (velocity path only)."""
    current_depth = max_depth
    while current_depth >= wp.int32(0):
        if lane < body_count and data.depth[articulation, lane] == current_depth:
            spatial_mass = _make_spatial_mass(data.body_mass[articulation, lane], data.body_inertia[articulation, lane])
            impulse = spatial_mass @ (data.velocity_out[articulation, lane] - data.velocity_in[articulation, lane])
            begin = data.child_start[articulation, lane]
            end = data.child_start[articulation, lane + wp.int32(1)]
            for cursor in range(begin, end):
                child = data.child_index[articulation, cursor]
                child_transform = _make_spatial_shift_transform(data.shift[articulation, child])
                impulse += wp.transpose(child_transform) @ data.reaction[articulation, child]
            data.reaction[articulation, lane] = impulse
        _sync_warp()
        current_depth -= wp.int32(1)


@wp.kernel(enable_backward=False)
def _project_maximal_tree_kernel(implicit_drive: wp.bool, data: MaximalTreeProjectorData):
    tid = wp.tid()
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    body_count = data.body_count[articulation]
    max_depth = data.max_depth[articulation]
    _project_tree_velocities(data, articulation, lane, body_count, max_depth, implicit_drive)
    _project_tree_reactions(data, articulation, lane, body_count, max_depth)


@wp.kernel(enable_backward=False)
def _publish_maximal_tree_kernel(
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: MaximalTreeProjectorData,
):
    tid = wp.tid()
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    if lane >= data.body_count[articulation]:
        return

    body = data.body_slot[articulation, lane]
    velocity = data.velocity_out[articulation, lane]
    bodies.velocity[body] = wp.vec3f(velocity[0], velocity[1], velocity[2])
    bodies.angular_velocity[body] = wp.vec3f(velocity[3], velocity[4], velocity[5])
    if lane == wp.int32(0):
        return

    joint = data.joint_index[articulation, lane]
    cid = joint_to_cid[joint]
    impulse = data.reaction[articulation, lane]
    linear = wp.vec3f(impulse[0], impulse[1], impulse[2])
    angular = wp.vec3f(impulse[3], impulse[4], impulse[5])
    r1 = read_vec3(constraints, _OFF_R1_B2, cid)
    r2 = read_vec3(constraints, _OFF_R2_B2, cid)
    axis = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
    torque_after_r1 = angular - wp.cross(r1, linear)
    lever_length = wp.dot(r2 - r1, axis)
    lambda2 = wp.vec3f(0.0, 0.0, 0.0)
    if wp.abs(lever_length) > wp.float32(1.0e-8):
        lambda2 = wp.cross(torque_after_r1, axis) / lever_length
    lambda1 = linear - lambda2
    constraint_write_multiplier_vec3(
        constraints,
        _MUL_ACC_IMP1,
        cid,
        constraint_read_multiplier_vec3(constraints, _MUL_ACC_IMP1, cid) + lambda1,
    )
    constraint_write_multiplier_vec3(
        constraints,
        _MUL_ACC_IMP2,
        cid,
        constraint_read_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid) + lambda2,
    )


@wp.func
def _apply_position_lane(
    bodies: BodyContainer,
    data: MaximalTreeProjectorData,
    articulation: wp.int32,
    lane: wp.int32,
):
    """Apply ``velocity_out`` as a position displacement twist (one lane).

    Mirrors ``_integrate_velocities_kernel``'s pose update with ``dt = 1``.
    Velocities and accumulated joint impulses stay untouched;
    ``inverse_inertia_world`` refreshes later via ``_refresh_world_inertia``.
    """
    body = data.body_slot[articulation, lane]
    displacement = data.velocity_out[articulation, lane]
    linear = wp.vec3f(displacement[0], displacement[1], displacement[2])
    angular = wp.vec3f(displacement[3], displacement[4], displacement[5])
    bodies.position[body] = bodies.position[body] + linear
    q_rot = _rotation_quaternion(angular, wp.float32(1.0))
    bodies.orientation[body] = wp.normalize(q_rot * bodies.orientation[body])


@wp.kernel(enable_backward=False)
def _project_maximal_tree_positions_kernel(
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: MaximalTreeProjectorData,
):
    """Fused position-level projection: 3 Newton iterations in one launch.

    Each iteration runs gather -> tree solve -> pose apply entirely
    warp-locally (every body belongs to exactly one free-root tree), with
    :func:`_sync_warp` between phases so iteration ``k+1`` reads the poses
    written by iteration ``k``. The reaction recursion is skipped — the
    position path never publishes joint impulses.
    """
    tid = wp.tid()
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    body_count = data.body_count[articulation]
    max_depth = data.max_depth[articulation]

    for _newton_iter in range(3):
        if lane < body_count:
            _gather_position_lane(joint_to_cid, constraints, bodies, data, articulation, lane)
        _sync_warp()
        _project_tree_velocities(data, articulation, lane, body_count, max_depth, wp.bool(False))
        if lane < body_count:
            _apply_position_lane(bodies, data, articulation, lane)
        _sync_warp()


class MaximalTreeProjector:
    """Mass-metric projector for free-root revolute articulation trees."""

    @staticmethod
    def supports_model(model: Model) -> bool:
        """Return whether every model articulation has a supported topology."""
        articulation_count = int(model.articulation_count)
        if articulation_count <= 0 or not model.device.is_cuda:
            return False
        starts = model.articulation_start.numpy()
        joint_articulation = model.joint_articulation.numpy()
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_enabled = model.joint_enabled.numpy() if model.joint_enabled is not None else None
        body_inv_mass = model.body_inv_mass.numpy()
        body_world = model.body_world.numpy()
        claimed_bodies: set[int] = set()
        for articulation in range(articulation_count):
            start = int(starts[articulation])
            end = int(starts[articulation + 1])
            owned_joints = [joint for joint in range(start, end) if int(joint_articulation[joint]) == articulation]
            if len(owned_joints) < 2 or len(owned_joints) > _WARP_SIZE:
                return False
            root_joint = owned_joints[0]
            root = int(joint_child[root_joint])
            if (
                int(joint_type[root_joint]) != int(JointType.FREE)
                or int(joint_parent[root_joint]) >= 0
                or root < 0
                or root in claimed_bodies
                or not np.isfinite(body_inv_mass[root])
                or body_inv_mass[root] <= 0.0
                or (joint_enabled is not None and not bool(joint_enabled[root_joint]))
            ):
                return False
            world = int(body_world[root])
            bodies = {root}
            for joint in owned_joints[1:]:
                if int(joint_type[joint]) != int(JointType.REVOLUTE):
                    return False
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
            raise ValueError("maximal tree projector requires CUDA free-root revolute trees with 2-32 bodies")
        self.model = model
        self.constraints = constraints
        self.bodies = bodies
        self.joint_to_cid = joint_to_cid
        self.articulation_count = int(model.articulation_count)
        self.launch_dim = self.articulation_count * _WARP_SIZE
        self.block_dim = _WARP_SIZE

        starts = model.articulation_start.numpy()
        joint_articulation = model.joint_articulation.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_to_cid_np = joint_to_cid.numpy()
        shape = (self.articulation_count, _WARP_SIZE)
        body_count = np.zeros(self.articulation_count, dtype=np.int32)
        max_depth = np.zeros(self.articulation_count, dtype=np.int32)
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
                        raise ValueError(f"projected revolute joint {joint} has no maximal ADBS column")
                    parent_lane = body_to_lane[int(joint_parent[joint])]
                    parent[articulation, lane] = parent_lane
                    depth[articulation, lane] = depth[articulation, parent_lane] + 1
                    children[parent_lane].append(lane)
            max_depth[articulation] = int(np.max(depth[articulation, :count]))
            flat_children: list[int] = []
            for lane, lane_children in enumerate(children):
                flat_children.extend(lane_children)
                child_start[articulation, lane + 1] = len(flat_children)
            child_index[articulation, : len(flat_children)] = flat_children
            child_start[articulation, count + 1 :] = len(flat_children)

        self.block_dim = 128 if self.articulation_count >= 512 else _WARP_SIZE

        device = model.device
        data = MaximalTreeProjectorData()
        data.body_count = wp.array(body_count, device=device)
        data.max_depth = wp.array(max_depth, device=device)
        data.joint_index = wp.array(joint_index, device=device)
        data.body_slot = wp.array(body_slot, device=device)
        data.depth = wp.array(depth, device=device)
        data.parent = wp.array(parent, device=device)
        data.child_start = wp.array(child_start, device=device)
        data.child_index = wp.array(child_index, device=device)
        data.shift = wp.empty(shape, dtype=wp.vec3f, device=device)
        data.motion = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.affine_offset = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.body_mass = wp.empty(shape, dtype=wp.float32, device=device)
        data.body_inertia = wp.empty(shape, dtype=inertia_sym6, device=device)
        data.velocity_in = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.inverse_d = wp.empty(shape, dtype=wp.float32, device=device)
        data.parent_articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.parent_bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.velocity_out = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.reaction = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.drive_diag = wp.zeros(shape, dtype=wp.float32, device=device)
        data.drive_bias = wp.zeros(shape, dtype=wp.float32, device=device)
        self.data = data
        self.implicit_drive = _PHOENX_MAXIMAL_IMPLICIT_DRIVE

    def project(self, *, use_bias: bool, dt: float = 0.0) -> None:
        """Project body twists and publish the recovered joint reactions.

        See :class:`GeneralMaximalTreeProjector` for the opt-in implicit-PD
        drive folded into the ``use_bias`` recursion.
        """
        implicit_drive = bool(self.implicit_drive and use_bias)
        wp.launch(
            _gather_maximal_tree_kernel,
            dim=self.launch_dim,
            block_dim=self.block_dim,
            inputs=[use_bias, implicit_drive, float(dt), self.joint_to_cid, self.constraints, self.bodies, self.data],
            device=self.model.device,
        )
        wp.launch(
            _project_maximal_tree_kernel,
            dim=self.launch_dim,
            block_dim=self.block_dim,
            inputs=[implicit_drive, self.data],
            device=self.model.device,
        )
        wp.launch(
            _publish_maximal_tree_kernel,
            dim=self.launch_dim,
            block_dim=self.block_dim,
            inputs=[self.joint_to_cid, self.constraints, self.bodies, self.data],
            device=self.model.device,
        )

    def project_positions(self) -> None:
        """Project post-integrate body poses back onto the joint manifold.

        Runs the mass-metric tree recursion on a zero input twist with the
        full current-pose anchor errors as affine targets, then applies the
        resulting twist as a rigid displacement. Momentum-free for free-root
        trees (zero net internal reaction); velocities are untouched. Three
        Newton iterations: each pass solves the linearized manifold
        projection, and the linearization residual dominates at large
        per-substep drift (few substeps) — 3 iterations bring the G1 3-substep
        anchor RMS from 5e-2 (no pass) to 2e-7.
        """
        wp.launch(
            _project_maximal_tree_positions_kernel,
            dim=self.launch_dim,
            block_dim=self.block_dim,
            inputs=[self.joint_to_cid, self.constraints, self.bodies, self.data],
            device=self.model.device,
        )


__all__ = ["MaximalTreeProjector", "MaximalTreeProjectorData"]
