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
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.model_adapter import _is_locked_dof

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
    joint_parent: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
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
    dof_count = wp.int32(0)

    if data.floating_root[articulation] == wp.int32(0) or lane > wp.int32(0):
        parent = joint_parent[joint] + wp.int32(1)
        parent_orientation = bodies.orientation[parent]
        child_orientation = bodies.orientation[body]
        r_parent = wp.quat_rotate(
            parent_orientation, wp.transform_get_translation(joint_x_p[joint]) - bodies.body_com[parent]
        )
        r_child = wp.quat_rotate(
            child_orientation, wp.transform_get_translation(joint_x_c[joint]) - bodies.body_com[body]
        )
        shift = wp.skew(r_child - r_parent)
        for row in range(3):
            for column in range(3):
                joint_transform[row, column + wp.int32(3)] = shift[row, column]

        axis_rotation = parent_orientation * wp.transform_get_rotation(joint_x_p[joint])
        qd_start = joint_qd_start[joint]
        linear_count = joint_dof_dim[joint, 0]
        total_count = linear_count + joint_dof_dim[joint, 1]
        for local in range(6):
            if wp.int32(local) < total_count:
                dof = qd_start + wp.int32(local)
                if joint_limit_lower[dof] <= joint_limit_upper[dof] and dof_count < wp.int32(3):
                    axis = wp.normalize(wp.quat_rotate(axis_rotation, joint_axis[dof]))
                    linear_axis = axis
                    angular_axis = wp.vec3f(0.0)
                    if wp.int32(local) >= linear_count:
                        linear_axis = wp.cross(r_child, axis)
                        angular_axis = axis
                    joint_motion = _set_motion_column(joint_motion, dof_count, linear_axis, angular_axis)
                    dof_count += wp.int32(1)

    data.dof_count[articulation, lane] = dof_count
    data.transform[articulation, lane] = joint_transform
    data.motion[articulation, lane] = joint_motion
    data.affine_offset[articulation, lane] = wp.spatial_vectorf(0.0)


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

    body = data.body_slot[articulation, lane]
    velocity = data.velocity_out[articulation, lane]
    bodies.velocity[body] = wp.vec3f(velocity[0], velocity[1], velocity[2])
    bodies.angular_velocity[body] = wp.vec3f(velocity[3], velocity[4], velocity[5])
    if data.floating_root[articulation] != wp.int32(0) and lane == wp.int32(0):
        return

    joint = data.joint_index[articulation, lane]
    cid = joint_to_cid[joint]
    impulse = data.reaction[articulation, lane]
    if use_bias:
        constraints.d6.reaction_wrench[cid] = impulse
    else:
        constraints.d6.reaction_wrench[cid] += impulse


@wp.kernel(enable_backward=False)
def _project_general_maximal_tree_fused_kernel(
    use_bias: wp.bool,
    joint_parent: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_to_cid: wp.array[wp.int32],
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    data: GeneralMaximalTreeProjectorData,
):
    tid = wp.tid()
    _gather_general_maximal_tree_thread(
        tid,
        use_bias,
        joint_parent,
        joint_x_p,
        joint_x_c,
        joint_axis,
        joint_qd_start,
        joint_dof_dim,
        joint_limit_lower,
        joint_limit_upper,
        bodies,
        data,
    )
    _sync_warp()
    _project_general_maximal_tree_thread(tid, data)
    _sync_warp()
    _publish_general_maximal_tree_thread(tid, use_bias, joint_to_cid, constraints, bodies, data)


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
                    free_count = locked_linear.count(False) + locked_angular.count(False)
                    if free_count > 3:
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
        for joint in range(int(model.joint_count)):
            if joint_enabled is not None and not bool(joint_enabled[joint]):
                continue
            if int(joint_articulation[joint]) >= 0:
                continue
            if int(joint_parent[joint]) in claimed_bodies or int(joint_child[joint]) in claimed_bodies:
                return False
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
                        raise ValueError(f"projected joint {joint} has no maximal joint constraint column")
                    parent_lane = body_to_lane[int(joint_parent[joint])]
                    parent[articulation, lane] = parent_lane
                    depth[articulation, lane] = depth[articulation, parent_lane] + 1
                    children[parent_lane].append(lane)
            if floating_root[articulation] == 0 and int(joint_to_cid_np[root_joint]) < 0:
                raise ValueError(f"projected root joint {root_joint} has no maximal joint constraint column")
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
            inputs=[
                use_bias,
                self.model.joint_parent,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.joint_to_cid,
                self.constraints,
                self.bodies,
                self.data,
            ],
            device=self.model.device,
        )

    def project_positions(self) -> None:
        """No-op: general (mixed-mode) trees keep prepare-time Baumgarte only.

        The revolute-tree :class:`MaximalTreeProjector` runs a position-level
        projection after integration; mirroring it here needs per-mode
        current-pose locked-error reconstruction (ball / universal /
        prismatic / fixed anchors), which is not yet ported.
        """


__all__ = ["GeneralMaximalTreeProjector", "GeneralMaximalTreeProjectorData"]
