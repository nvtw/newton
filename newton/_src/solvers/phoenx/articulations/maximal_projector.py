# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact joint-tree projection for maximal-coordinate robot bodies."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import JointType, Model
from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    read_vec3,
    write_vec3,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_ACC_IMP1,
    _OFF_ACC_IMP2,
    _OFF_AXIS_WORLD,
    _OFF_BIAS1,
    _OFF_BIAS2,
    _OFF_R1_B1,
    _OFF_R1_B2,
    _OFF_R2_B2,
    _OFF_T1,
    _OFF_T2,
)

_WARP_SIZE = 32
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
    transform: wp.array2d[wp.spatial_matrixf]
    motion: wp.array2d[wp.spatial_vectorf]
    affine_offset: wp.array2d[wp.spatial_vectorf]
    mass: wp.array2d[wp.spatial_matrixf]
    velocity_in: wp.array2d[wp.spatial_vectorf]
    articulated: wp.array2d[wp.spatial_matrixf]
    bias: wp.array2d[wp.spatial_vectorf]
    inverse_d: wp.array2d[wp.float32]
    parent_articulated: wp.array2d[wp.spatial_matrixf]
    parent_bias: wp.array2d[wp.spatial_vectorf]
    velocity_out: wp.array2d[wp.spatial_vectorf]
    reaction: wp.array2d[wp.spatial_vectorf]


@wp.kernel(enable_backward=False)
def _gather_maximal_tree_kernel(
    use_bias: wp.bool,
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
    inverse_mass = bodies.inverse_mass[body]
    body_mass = wp.float32(1.0) / inverse_mass
    inertia = wp.inverse(mat33_from_sym6(bodies.inverse_inertia_world[body]))
    spatial_mass = wp.spatial_matrixf(0.0)
    for row in range(3):
        spatial_mass[row, row] = body_mass
        for column in range(3):
            spatial_mass[row + wp.int32(3), column + wp.int32(3)] = inertia[row, column]
    data.mass[articulation, lane] = spatial_mass

    linear = bodies.velocity[body]
    angular = bodies.angular_velocity[body]
    data.velocity_in[articulation, lane] = wp.spatial_vectorf(
        linear[0], linear[1], linear[2], angular[0], angular[1], angular[2]
    )

    joint_transform = wp.spatial_matrixf(0.0)
    for diagonal in range(6):
        joint_transform[diagonal, diagonal] = wp.float32(1.0)
    joint_motion = wp.spatial_vectorf(0.0)
    affine_offset = wp.spatial_vectorf(0.0)
    if lane > wp.int32(0):
        cid = joint_to_cid[joint]
        r_parent = read_vec3(constraints, _OFF_R1_B1, cid)
        r_child = read_vec3(constraints, _OFF_R1_B2, cid)
        shift = wp.skew(r_child - r_parent)
        for row in range(3):
            for column in range(3):
                joint_transform[row, column + wp.int32(3)] = shift[row, column]
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

    data.transform[articulation, lane] = joint_transform
    data.motion[articulation, lane] = joint_motion
    data.affine_offset[articulation, lane] = affine_offset


@wp.kernel(enable_backward=False)
def _project_maximal_tree_kernel(data: MaximalTreeProjectorData):
    tid = wp.tid()
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    body_count = data.body_count[articulation]
    max_depth = data.max_depth[articulation]

    if lane < body_count:
        body_mass = data.mass[articulation, lane]
        data.articulated[articulation, lane] = body_mass
        data.bias[articulation, lane] = body_mass @ data.velocity_in[articulation, lane]
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
                u = body_articulated @ joint_motion
                reciprocal_d = wp.float32(1.0) / wp.dot(joint_motion, u)
                data.inverse_d[articulation, lane] = reciprocal_d
                projected = body_articulated - reciprocal_d * wp.outer(u, u)
                projected_bias = body_bias - reciprocal_d * wp.dot(joint_motion, body_bias) * u
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

    if lane == wp.int32(0):
        data.velocity_out[articulation, lane] = _solve_spd6(
            data.articulated[articulation, lane], data.bias[articulation, lane]
        )
    _sync_warp()

    current_depth = wp.int32(1)
    while current_depth <= max_depth:
        if lane < body_count and data.depth[articulation, lane] == current_depth:
            parent = data.parent[articulation, lane]
            base = (
                data.transform[articulation, lane] @ data.velocity_out[articulation, parent]
                + data.affine_offset[articulation, lane]
            )
            joint_motion = data.motion[articulation, lane]
            joint_velocity = data.inverse_d[articulation, lane] * wp.dot(
                joint_motion,
                data.bias[articulation, lane] - data.articulated[articulation, lane] @ base,
            )
            data.velocity_out[articulation, lane] = base + joint_velocity * joint_motion
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
    write_vec3(constraints, _OFF_ACC_IMP1, cid, read_vec3(constraints, _OFF_ACC_IMP1, cid) + lambda1)
    write_vec3(constraints, _OFF_ACC_IMP2, cid, read_vec3(constraints, _OFF_ACC_IMP2, cid) + lambda2)


class MaximalTreeProjector:
    """Mass-metric projector for free-root revolute articulation trees."""

    @staticmethod
    def supports_model(model: Model) -> bool:
        """Return whether every model articulation has a supported topology."""
        articulation_count = int(model.articulation_count)
        if articulation_count <= 0 or not model.device.is_cuda:
            return False
        starts = model.articulation_start.numpy()
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
            if end - start < 2 or end - start > _WARP_SIZE:
                return False
            root = int(joint_child[start])
            if (
                int(joint_type[start]) != int(JointType.FREE)
                or int(joint_parent[start]) >= 0
                or root < 0
                or root in claimed_bodies
                or not np.isfinite(body_inv_mass[root])
                or body_inv_mass[root] <= 0.0
                or (joint_enabled is not None and not bool(joint_enabled[start]))
            ):
                return False
            world = int(body_world[root])
            bodies = {root}
            for joint in range(start + 1, end):
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

        starts = model.articulation_start.numpy()
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
            count = end - start
            body_count[articulation] = count
            body_to_lane: dict[int, int] = {}
            children: list[list[int]] = [[] for _ in range(count)]
            for lane, joint in enumerate(range(start, end)):
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
        data.transform = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.motion = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.affine_offset = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.mass = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.velocity_in = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.inverse_d = wp.empty(shape, dtype=wp.float32, device=device)
        data.parent_articulated = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.parent_bias = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.velocity_out = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.reaction = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        self.data = data

    def project(self, *, use_bias: bool) -> None:
        """Project body twists and publish the recovered joint reactions."""
        wp.launch(
            _gather_maximal_tree_kernel,
            dim=self.launch_dim,
            block_dim=_WARP_SIZE,
            inputs=[use_bias, self.joint_to_cid, self.constraints, self.bodies, self.data],
            device=self.model.device,
        )
        wp.launch(
            _project_maximal_tree_kernel,
            dim=self.launch_dim,
            block_dim=_WARP_SIZE,
            inputs=[self.data],
            device=self.model.device,
        )
        wp.launch(
            _publish_maximal_tree_kernel,
            dim=self.launch_dim,
            block_dim=_WARP_SIZE,
            inputs=[self.joint_to_cid, self.constraints, self.bodies, self.data],
            device=self.model.device,
        )


__all__ = ["MaximalTreeProjector", "MaximalTreeProjectorData"]
