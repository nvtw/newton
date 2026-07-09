# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Factored contact mobility and impulse response for maximal joint trees."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.maximal_projector import (
    MaximalTreeProjector,
    MaximalTreeProjectorData,
    _make_spatial_shift_transform,
    _solve_spd6,
    _sync_warp,
)
from newton._src.solvers.phoenx.body import BodyContainer

_WARP_SIZE = 32


@wp.func
def _inverse_spd6(matrix: wp.spatial_matrixf):
    result = wp.spatial_matrixf(0.0)
    for column in range(6):
        basis = wp.spatial_vectorf(0.0)
        basis[column] = wp.float32(1.0)
        solution = _solve_spd6(matrix, basis)
        for row in range(6):
            result[row, column] = solution[row]
    return result


@wp.struct
class MaximalContactResponseData:
    body_articulation: wp.array[wp.int32]
    body_lane: wp.array[wp.int32]
    conditional_map: wp.array2d[wp.spatial_matrixf]
    mobility: wp.array2d[wp.spatial_matrixf]
    impulse: wp.array2d[wp.spatial_vectorf]
    rhs: wp.array2d[wp.spatial_vectorf]
    parent_rhs: wp.array2d[wp.spatial_vectorf]
    velocity: wp.array2d[wp.spatial_vectorf]


@wp.func
def _point_wrench(
    bodies: BodyContainer,
    body: wp.int32,
    point: wp.vec3f,
    direction: wp.vec3f,
):
    lever = point - bodies.position[body]
    torque = wp.cross(lever, direction)
    return wp.spatial_vectorf(
        direction[0],
        direction[1],
        direction[2],
        torque[0],
        torque[1],
        torque[2],
    )


@wp.func
def maximal_contact_pair_inverse_mass(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    body0: wp.int32,
    point0: wp.vec3f,
    direction0: wp.vec3f,
    body1: wp.int32,
    point1: wp.vec3f,
    direction1: wp.vec3f,
):
    # Exact tree-constrained response of one two-body row.
    articulation0 = wp.int32(-1)
    articulation1 = wp.int32(-1)
    lane0 = wp.int32(-1)
    lane1 = wp.int32(-1)
    wrench0 = wp.spatial_vectorf(0.0)
    wrench1 = wp.spatial_vectorf(0.0)
    result = wp.float32(0.0)

    if body0 >= wp.int32(0):
        articulation0 = response.body_articulation[body0]
        lane0 = response.body_lane[body0]
        if articulation0 >= wp.int32(0):
            wrench0 = _point_wrench(bodies, body0, point0, direction0)
            result += wp.dot(wrench0, response.mobility[articulation0, lane0] @ wrench0)
    if body1 >= wp.int32(0):
        articulation1 = response.body_articulation[body1]
        lane1 = response.body_lane[body1]
        if articulation1 >= wp.int32(0):
            wrench1 = _point_wrench(bodies, body1, point1, direction1)
            result += wp.dot(wrench1, response.mobility[articulation1, lane1] @ wrench1)

    if articulation0 >= wp.int32(0) and articulation0 == articulation1:
        node0 = lane0
        node1 = lane1
        depth0 = tree.depth[articulation0, node0]
        depth1 = tree.depth[articulation0, node1]
        while depth0 > depth1:
            wrench0 = wp.transpose(response.conditional_map[articulation0, node0]) @ wrench0
            node0 = tree.parent[articulation0, node0]
            depth0 -= wp.int32(1)
        while depth1 > depth0:
            wrench1 = wp.transpose(response.conditional_map[articulation0, node1]) @ wrench1
            node1 = tree.parent[articulation0, node1]
            depth1 -= wp.int32(1)
        while node0 != node1:
            wrench0 = wp.transpose(response.conditional_map[articulation0, node0]) @ wrench0
            wrench1 = wp.transpose(response.conditional_map[articulation0, node1]) @ wrench1
            node0 = tree.parent[articulation0, node0]
            node1 = tree.parent[articulation0, node1]
        result += wp.float32(2.0) * wp.dot(
            wrench0,
            response.mobility[articulation0, node0] @ wrench1,
        )
    return result


@wp.kernel(enable_backward=False)
def _compute_maximal_contact_mobility_kernel(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
):
    tid = wp.tid()
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    body_count = tree.body_count[articulation]
    max_depth = tree.max_depth[articulation]

    identity = wp.spatial_matrixf(0.0)
    for diagonal in range(6):
        identity[diagonal, diagonal] = wp.float32(1.0)
    if lane == wp.int32(0):
        response.mobility[articulation, lane] = _inverse_spd6(tree.articulated[articulation, lane])
        response.conditional_map[articulation, lane] = identity
    _sync_warp()

    current_depth = wp.int32(1)
    while current_depth <= max_depth:
        if lane < body_count and tree.depth[articulation, lane] == current_depth:
            motion = tree.motion[articulation, lane]
            conditional_mobility = tree.inverse_d[articulation, lane] * wp.outer(motion, motion)
            transform = _make_spatial_shift_transform(tree.shift[articulation, lane])
            mapping = (identity - conditional_mobility @ tree.articulated[articulation, lane]) @ transform
            response.conditional_map[articulation, lane] = mapping
            parent_mobility = response.mobility[articulation, tree.parent[articulation, lane]]
            response.mobility[articulation, lane] = (
                mapping @ parent_mobility @ wp.transpose(mapping) + conditional_mobility
            )
        _sync_warp()
        current_depth += wp.int32(1)


@wp.kernel(enable_backward=False)
def _apply_maximal_contact_impulse_kernel(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
):
    tid = wp.tid()
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    body_count = tree.body_count[articulation]
    max_depth = tree.max_depth[articulation]

    if lane < body_count:
        response.rhs[articulation, lane] = wp.spatial_vectorf(0.0)
        response.parent_rhs[articulation, lane] = wp.spatial_vectorf(0.0)
        response.velocity[articulation, lane] = wp.spatial_vectorf(0.0)
    _sync_warp()

    current_depth = max_depth
    while current_depth >= wp.int32(0):
        if lane < body_count and tree.depth[articulation, lane] == current_depth:
            rhs = response.impulse[articulation, lane]
            begin = tree.child_start[articulation, lane]
            end = tree.child_start[articulation, lane + wp.int32(1)]
            for cursor in range(begin, end):
                rhs += response.parent_rhs[articulation, tree.child_index[articulation, cursor]]
            response.rhs[articulation, lane] = rhs
            if lane != wp.int32(0):
                motion = tree.motion[articulation, lane]
                u = tree.articulated[articulation, lane] @ motion
                projected_rhs = rhs - tree.inverse_d[articulation, lane] * wp.dot(motion, rhs) * u
                transform = _make_spatial_shift_transform(tree.shift[articulation, lane])
                response.parent_rhs[articulation, lane] = wp.transpose(transform) @ projected_rhs
        _sync_warp()
        current_depth -= wp.int32(1)

    if lane == wp.int32(0):
        response.velocity[articulation, lane] = response.mobility[articulation, lane] @ response.rhs[articulation, lane]
    _sync_warp()

    current_depth = wp.int32(1)
    while current_depth <= max_depth:
        if lane < body_count and tree.depth[articulation, lane] == current_depth:
            transform = _make_spatial_shift_transform(tree.shift[articulation, lane])
            base = transform @ response.velocity[articulation, tree.parent[articulation, lane]]
            motion = tree.motion[articulation, lane]
            joint_velocity = tree.inverse_d[articulation, lane] * wp.dot(
                motion,
                response.rhs[articulation, lane] - tree.articulated[articulation, lane] @ base,
            )
            response.velocity[articulation, lane] = base + joint_velocity * motion
        _sync_warp()
        current_depth += wp.int32(1)


class MaximalContactResponse:
    """Exact reusable mobility for a factored free-root revolute forest."""

    def __init__(self, projector: MaximalTreeProjector):
        self.projector = projector
        shape = projector.data.body_slot.shape
        device = projector.model.device
        data = MaximalContactResponseData()
        body_articulation = np.full(int(projector.bodies.position.shape[0]), -1, dtype=np.int32)
        body_lane = np.full_like(body_articulation, -1)
        body_count = projector.data.body_count.numpy()
        body_slot = projector.data.body_slot.numpy()
        for articulation, count in enumerate(body_count):
            for lane, slot in enumerate(body_slot[articulation, : int(count)]):
                body_articulation[int(slot)] = articulation
                body_lane[int(slot)] = lane
        data.body_articulation = wp.array(body_articulation, device=device)
        data.body_lane = wp.array(body_lane, device=device)
        data.conditional_map = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.mobility = wp.empty(shape, dtype=wp.spatial_matrixf, device=device)
        data.impulse = wp.zeros(shape, dtype=wp.spatial_vectorf, device=device)
        data.rhs = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.parent_rhs = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        data.velocity = wp.empty(shape, dtype=wp.spatial_vectorf, device=device)
        self.data = data

    def compute_mobility(self) -> None:
        """Compute exact diagonal and path-factor mobility from tree factors."""
        wp.launch(
            _compute_maximal_contact_mobility_kernel,
            dim=self.projector.launch_dim,
            block_dim=self.projector.block_dim,
            inputs=[self.projector.data, self.data],
            device=self.projector.model.device,
        )

    def solve_impulses(self) -> None:
        """Apply the factored constrained response to :attr:`data.impulse`."""
        wp.launch(
            _apply_maximal_contact_impulse_kernel,
            dim=self.projector.launch_dim,
            block_dim=self.projector.block_dim,
            inputs=[self.projector.data, self.data],
            device=self.projector.model.device,
        )


__all__ = ["MaximalContactResponse", "maximal_contact_pair_inverse_mass"]
