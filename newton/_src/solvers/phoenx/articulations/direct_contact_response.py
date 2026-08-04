# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-space mobility constrained by PhoenX direct equalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.direct_equality import _row_wrench_for_body
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import GROUPED_RHS_ITEMS_PER_TASK
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import _block_sync
from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal,
    cc_get_r0,
    cc_get_r1,
    cc_get_tangent1,
)

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.articulations.direct_equality import DirectEqualitySystem


@wp.struct
class DirectContactResponseData:
    """Device view of packed contact-space Schur-complement work."""

    body_mechanism: wp.array[wp.int32]
    body_constraint_mechanism: wp.array[wp.int32]
    body_lane: wp.array[wp.int32]
    mechanism_body_start: wp.array[wp.int32]
    mechanism_body: wp.array[wp.int32]
    mechanism_row_start: wp.array[wp.int32]
    contact_mechanism: wp.array[wp.int32]
    contact_body0: wp.array[wp.int32]
    contact_body1: wp.array[wp.int32]
    workspace_stride: wp.int32
    rhs: wp.array[wp.float32]
    solution: wp.array[wp.float32]
    accumulated_solution: wp.array[wp.float32]
    mobility: wp.array2d[wp.float32]
    delta_coordinate: wp.array[wp.vec3]
    body_row_start: wp.array[wp.int32]
    body_rows: wp.array[wp.int32]
    row_joint: wp.array[wp.int32]
    row_local: wp.array[wp.int32]
    joint_to_structural: wp.array[wp.int32]
    joint_parent: wp.array[wp.int32]
    joint_child: wp.array[wp.int32]
    row_wrench0: wp.array2d[wp.spatial_vector]
    row_wrench1: wp.array2d[wp.spatial_vector]
    row_scale: wp.array[wp.float32]
    accumulated_impulse: wp.array[wp.float32]


@wp.func
def _contact_wrench_response_dot(
    row_wrench: wp.spatial_vector,
    force: wp.vec3,
    torque: wp.vec3,
    inverse_mass: wp.float32,
    inverse_inertia: wp.mat33,
) -> wp.float32:
    return wp.dot(wp.spatial_top(row_wrench), inverse_mass * force) + wp.dot(
        wp.spatial_bottom(row_wrench), inverse_inertia * torque
    )


@wp.func
def _unconstrained_wrench_cross_mobility(
    bodies: BodyContainer,
    body: wp.int32,
    r: wp.vec3,
    direction0: wp.vec3,
    direction1: wp.vec3,
) -> wp.float32:
    if body <= wp.int32(0) or bodies.inverse_mass[body] <= wp.float32(0.0):
        return wp.float32(0.0)
    torque0 = wp.cross(r, direction0)
    torque1 = wp.cross(r, direction1)
    return bodies.inverse_mass[body] * wp.dot(direction0, direction1) + wp.dot(
        torque0,
        mat33_from_sym6(bodies.inverse_inertia_world[body]) * torque1,
    )


@wp.func
def _unconstrained_pair_cross_mobility(
    bodies: BodyContainer,
    body0: wp.int32,
    r0: wp.vec3,
    direction00: wp.vec3,
    direction01: wp.vec3,
    body1: wp.int32,
    r1: wp.vec3,
    direction10: wp.vec3,
    direction11: wp.vec3,
) -> wp.float32:
    return _unconstrained_wrench_cross_mobility(bodies, body0, r0, direction00, direction01) + (
        _unconstrained_wrench_cross_mobility(bodies, body1, r1, direction10, direction11)
    )


@wp.kernel(enable_backward=False)
def _build_contact_equality_rhs_kernel(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    contacts: ContactContainer,
):
    tid = wp.tid()
    contact = tid // wp.int32(128)
    lane = tid - contact * wp.int32(128)
    mechanism = response.contact_mechanism[contact]
    if mechanism < wp.int32(0):
        return
    body0 = response.contact_body0[contact]
    body1 = response.contact_body1[contact]
    normal = cc_get_normal(contacts, contact)
    tangent0 = cc_get_tangent1(contacts, contact)
    tangent1 = wp.cross(normal, tangent0)
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    row_begin = response.mechanism_row_start[mechanism]
    row_end = response.mechanism_row_start[mechanism + wp.int32(1)]
    task_offset = contact * response.workspace_stride
    for local_row in range(lane, row_end - row_begin, wp.int32(128)):
        offset = task_offset + local_row * wp.int32(4)
        response.rhs[offset] = wp.float32(0.0)
        response.rhs[offset + wp.int32(1)] = wp.float32(0.0)
        response.rhs[offset + wp.int32(2)] = wp.float32(0.0)
    _block_sync()

    inverse_inertia0 = mat33_from_sym6(bodies.inverse_inertia_world[body0])
    inverse_inertia1 = mat33_from_sym6(bodies.inverse_inertia_world[body1])
    for incidence in range(
        response.body_row_start[body0] + lane,
        response.body_row_start[body0 + wp.int32(1)],
        wp.int32(128),
    ):
        row = response.body_rows[incidence]
        joint = response.row_joint[row]
        row_body = _row_wrench_for_body(
            body0,
            joint,
            response.joint_to_structural[joint],
            response.row_local[row],
            response.joint_parent,
            response.joint_child,
            response.row_wrench0,
            response.row_wrench1,
        )
        for axis in range(3):
            direction = normal
            if axis == wp.int32(1):
                direction = tangent0
            elif axis == wp.int32(2):
                direction = tangent1
            value = _contact_wrench_response_dot(
                row_body,
                -direction,
                -wp.cross(r0, direction),
                bodies.inverse_mass[body0],
                inverse_inertia0,
            )
            response.rhs[task_offset + (row - row_begin) * wp.int32(4) + axis] = response.row_scale[row] * value
    _block_sync()
    for incidence in range(
        response.body_row_start[body1] + lane,
        response.body_row_start[body1 + wp.int32(1)],
        wp.int32(128),
    ):
        row = response.body_rows[incidence]
        joint = response.row_joint[row]
        row_body = _row_wrench_for_body(
            body1,
            joint,
            response.joint_to_structural[joint],
            response.row_local[row],
            response.joint_parent,
            response.joint_child,
            response.row_wrench0,
            response.row_wrench1,
        )
        for axis in range(3):
            direction = normal
            if axis == wp.int32(1):
                direction = tangent0
            elif axis == wp.int32(2):
                direction = tangent1
            value = _contact_wrench_response_dot(
                row_body,
                direction,
                wp.cross(r1, direction),
                bodies.inverse_mass[body1],
                inverse_inertia1,
            )
            response.rhs[task_offset + (row - row_begin) * wp.int32(4) + axis] += response.row_scale[row] * value


@wp.kernel(enable_backward=False)
def _compute_contact_schur_diagonal_kernel(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    contacts: ContactContainer,
):
    contact = wp.tid()
    mechanism = response.contact_mechanism[contact]
    if mechanism < wp.int32(0):
        return
    body0 = response.contact_body0[contact]
    body1 = response.contact_body1[contact]
    normal = cc_get_normal(contacts, contact)
    tangent0 = cc_get_tangent1(contacts, contact)
    tangent1 = wp.cross(normal, tangent0)
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    row_begin = response.mechanism_row_start[mechanism]
    row_end = response.mechanism_row_start[mechanism + wp.int32(1)]
    task_offset = contact * response.workspace_stride
    inverse00 = _unconstrained_pair_cross_mobility(bodies, body0, r0, -normal, -normal, body1, r1, normal, normal)
    inverse01 = _unconstrained_pair_cross_mobility(bodies, body0, r0, -normal, -tangent0, body1, r1, normal, tangent0)
    inverse02 = _unconstrained_pair_cross_mobility(bodies, body0, r0, -normal, -tangent1, body1, r1, normal, tangent1)
    inverse11 = _unconstrained_pair_cross_mobility(
        bodies, body0, r0, -tangent0, -tangent0, body1, r1, tangent0, tangent0
    )
    inverse12 = _unconstrained_pair_cross_mobility(
        bodies, body0, r0, -tangent0, -tangent1, body1, r1, tangent0, tangent1
    )
    inverse22 = _unconstrained_pair_cross_mobility(
        bodies, body0, r0, -tangent1, -tangent1, body1, r1, tangent1, tangent1
    )
    for local_row in range(row_end - row_begin):
        offset = task_offset + local_row * wp.int32(4)
        rhs0 = response.rhs[offset]
        rhs1 = response.rhs[offset + wp.int32(1)]
        rhs2 = response.rhs[offset + wp.int32(2)]
        solution0 = response.solution[offset]
        solution1 = response.solution[offset + wp.int32(1)]
        solution2 = response.solution[offset + wp.int32(2)]
        inverse00 -= rhs0 * solution0
        inverse01 -= rhs0 * solution1
        inverse02 -= rhs0 * solution2
        inverse11 -= rhs1 * solution1
        inverse12 -= rhs1 * solution2
        inverse22 -= rhs2 * solution2
    response.mobility[0, contact] = wp.float32(0.0)
    response.mobility[1, contact] = wp.float32(0.0)
    response.mobility[2, contact] = wp.float32(0.0)
    if inverse00 > wp.float32(1.0e-12):
        response.mobility[0, contact] = wp.float32(1.0) / inverse00
    if inverse11 > wp.float32(1.0e-12):
        response.mobility[1, contact] = wp.float32(1.0) / inverse11
    if inverse22 > wp.float32(1.0e-12):
        response.mobility[2, contact] = wp.float32(1.0) / inverse22
    response.mobility[3, contact] = inverse01
    response.mobility[4, contact] = inverse02
    response.mobility[5, contact] = inverse12


class DirectContactResponse:
    """Packed contact-space response for arbitrary direct mechanisms."""

    def __init__(
        self,
        direct: DirectEqualitySystem,
        contact_capacity: int,
        *,
        active_mechanisms: tuple[bool, ...] | None = None,
    ):
        if not direct.enabled:
            raise ValueError("direct contact response requires an enabled equality system")
        self.direct = direct
        topology = direct.topology
        mechanism_count = len(topology.dimensions)
        if active_mechanisms is None:
            active_mechanisms = (True,) * mechanism_count
        if len(active_mechanisms) != mechanism_count:
            raise ValueError("active mechanism mask must contain one entry per direct mechanism")

        inverse_mass = np.asarray(direct.model.body_inv_mass.numpy(), dtype=np.float32)
        joint_parent = np.asarray(direct.model.joint_parent.numpy(), dtype=np.int32)
        joint_child = np.asarray(direct.model.joint_child.numpy(), dtype=np.int32)
        all_mechanism_bodies = []
        for mechanism in range(mechanism_count):
            row_begin = int(topology.mechanism_row_start[mechanism])
            row_end = int(topology.mechanism_row_start[mechanism + 1])
            joints = {int(joint) for joint in topology.row_joint[row_begin:row_end]}
            bodies = sorted(
                {
                    body
                    for joint in joints
                    for body in (int(joint_parent[joint]), int(joint_child[joint]))
                    if body >= 0 and inverse_mass[body] > 0.0
                }
            )
            all_mechanism_bodies.append(tuple(body + 1 for body in bodies))
        mechanism_bodies = [
            bodies if active_mechanisms[mechanism] else () for mechanism, bodies in enumerate(all_mechanism_bodies)
        ]

        body_starts = np.zeros(mechanism_count + 1, dtype=np.int32)
        if mechanism_count:
            body_starts[1:] = np.cumsum([len(bodies) for bodies in mechanism_bodies])
        flat_bodies = np.asarray([body for bodies in mechanism_bodies for body in bodies], dtype=np.int32)
        body_mechanism = np.full(int(direct.bodies.position.shape[0]), -1, dtype=np.int32)
        body_constraint_mechanism = np.full_like(body_mechanism, -1)
        body_lane = np.full_like(body_mechanism, -1)
        for mechanism, bodies in enumerate(all_mechanism_bodies):
            for body in bodies:
                body_constraint_mechanism[body] = mechanism
        for mechanism, bodies in enumerate(mechanism_bodies):
            for lane, body in enumerate(bodies):
                body_mechanism[body] = mechanism
                body_lane[body] = lane

        device = direct.model.device
        capacity = max(1, int(contact_capacity))
        self.active_mechanisms = tuple(active_mechanisms)
        self.active_mechanism = wp.array(
            np.flatnonzero(np.asarray(active_mechanisms, dtype=bool)).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )
        active_mechanism_count = sum(active_mechanisms)
        group_size = GROUPED_RHS_ITEMS_PER_TASK
        task_capacity = min(
            capacity,
            (capacity + (group_size - 1) * active_mechanism_count + group_size - 1) // group_size,
        )
        self.contact_batch = direct.solver.create_grouped_rhs_batch(capacity, task_capacity)
        self.contact_mechanism = wp.full(capacity, -1, dtype=wp.int32, device=device)
        self.contact_body0 = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.contact_body1 = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.data = DirectContactResponseData()
        self.data.body_mechanism = wp.array(body_mechanism, dtype=wp.int32, device=device)
        self.data.body_constraint_mechanism = wp.array(
            body_constraint_mechanism,
            dtype=wp.int32,
            device=device,
        )
        self.data.body_lane = wp.array(body_lane, dtype=wp.int32, device=device)
        self.data.mechanism_body_start = wp.array(body_starts, dtype=wp.int32, device=device)
        self.data.mechanism_body = wp.array(flat_bodies, dtype=wp.int32, device=device)
        self.data.mechanism_row_start = wp.array(topology.mechanism_row_start, dtype=wp.int32, device=device)
        self.data.contact_mechanism = self.contact_mechanism
        self.data.contact_body0 = self.contact_body0
        self.data.contact_body1 = self.contact_body1
        self.data.workspace_stride = wp.int32(self.contact_batch.item_workspace_stride)
        self.data.rhs = self.contact_batch.rhs
        self.data.solution = self.contact_batch.solution
        self.data.accumulated_solution = wp.zeros(len(topology.row_joint), dtype=wp.float32, device=device)
        self.data.mobility = wp.zeros((6, capacity), dtype=wp.float32, device=device)
        self.data.delta_coordinate = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.data.body_row_start = direct.body_row_start
        self.data.body_rows = direct.body_rows
        self.data.row_joint = direct.row_joint
        self.data.row_local = direct.row_local
        self.data.joint_to_structural = direct.joint_to_structural
        self.data.joint_parent = direct.model.joint_parent
        self.data.joint_child = direct.model.joint_child
        self.data.row_wrench0 = direct.row_wrench0
        self.data.row_wrench1 = direct.row_wrench1
        self.data.row_scale = direct.row_scale
        self.data.accumulated_impulse = direct.accumulated_impulse

    def compute(self, contacts: ContactContainer) -> None:
        """Solve equality responses and form each active contact's local block."""
        capacity = self.contact_batch.item_capacity
        wp.launch(
            _build_contact_equality_rhs_kernel,
            dim=capacity * 128,
            block_dim=128,
            inputs=[self.data, self.direct.bodies, contacts],
            device=self.direct.model.device,
        )
        self.contact_batch.solve()
        wp.launch(
            _compute_contact_schur_diagonal_kernel,
            dim=capacity,
            inputs=[self.data, self.direct.bodies, contacts],
            device=self.direct.model.device,
        )


__all__ = ["DirectContactResponse", "DirectContactResponseData"]
