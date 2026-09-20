# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-space mobility constrained by PhoenX direct equalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.direct_equality import _row_wrench_for_body
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import (
    GROUPED_RHS_ITEM_WIDTH,
    GROUPED_RHS_ITEMS_PER_TASK,
)
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import _block_sync
from newton._src.solvers.phoenx.body import MOTION_KINEMATIC, BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal,
    cc_get_r0,
    cc_get_r1,
    cc_get_tangent1,
)

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.articulations.direct_equality import DirectEqualitySystem


_CONTACT_RHS_BLOCK_DIM = 64


@wp.struct
class DirectContactResponseData:
    """Device view of packed contact-space Schur-complement work."""

    body_mechanism: wp.array[wp.int32]
    body_constraint_mechanism: wp.array[wp.int32]
    body_lane: wp.array[wp.int32]
    endpoint_response: wp.array3d[wp.spatial_vector]
    contact_response: wp.array3d[wp.spatial_vector]
    mechanism_body_start: wp.array[wp.int32]
    mechanism_body: wp.array[wp.int32]
    mechanism_row_start: wp.array[wp.int32]
    contact_mechanism: wp.array[wp.int32]
    contact_column: wp.array[wp.int32]
    contact_body0: wp.array[wp.int32]
    contact_body1: wp.array[wp.int32]
    column_mechanism0: wp.array[wp.int32]
    column_mechanism1: wp.array[wp.int32]
    column_body0: wp.array[wp.int32]
    column_body1: wp.array[wp.int32]
    workspace_stride: wp.int32
    rhs: wp.array[wp.float32]
    solution: wp.array[wp.float32]
    gram: wp.array2d[wp.float32]
    accumulated_solution: wp.array[wp.float32]
    mobility: wp.array2d[wp.float32]
    delta_coordinate: wp.array[wp.vec3]
    delta_wrench: wp.array2d[wp.spatial_vector]
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
def _unit_axis(axis: wp.int32) -> wp.vec3:
    value = wp.vec3(0.0)
    if axis == wp.int32(0):
        value[0] = wp.float32(1.0)
    elif axis == wp.int32(1):
        value[1] = wp.float32(1.0)
    else:
        value[2] = wp.float32(1.0)
    return value


@wp.func
def _spatial_dot_mass_inverse(
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
    if (
        body <= wp.int32(0)
        or bodies.inverse_mass[body] <= wp.float32(0.0)
        or bodies.motion_type[body] == MOTION_KINEMATIC
    ):
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
    direction0: wp.vec3,
    direction1: wp.vec3,
    body1: wp.int32,
    r1: wp.vec3,
) -> wp.float32:
    return _unconstrained_wrench_cross_mobility(bodies, body0, r0, direction0, direction1) + (
        _unconstrained_wrench_cross_mobility(bodies, body1, r1, direction0, direction1)
    )


@wp.func
def _build_endpoint_equality_rhs(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    item: wp.int32,
    lane: wp.int32,
):
    column = item // wp.int32(4)
    slot = item - column * wp.int32(4)
    endpoint = slot // wp.int32(2)
    angular_basis = slot - endpoint * wp.int32(2)
    mechanism = response.column_mechanism0[column]
    body = response.column_body0[column]
    if endpoint == wp.int32(1):
        mechanism = response.column_mechanism1[column]
        body = response.column_body1[column]

    row_begin = response.mechanism_row_start[mechanism]
    row_end = response.mechanism_row_start[mechanism + wp.int32(1)]
    task_offset = item * response.workspace_stride
    for local_row in range(lane, row_end - row_begin, wp.int32(_CONTACT_RHS_BLOCK_DIM)):
        offset = task_offset + local_row * wp.int32(GROUPED_RHS_ITEM_WIDTH)
        response.rhs[offset] = wp.float32(0.0)
        response.rhs[offset + wp.int32(1)] = wp.float32(0.0)
        response.rhs[offset + wp.int32(2)] = wp.float32(0.0)
    _block_sync()

    if response.body_mechanism[body] != mechanism:
        return
    inverse_inertia = mat33_from_sym6(bodies.inverse_inertia_world[body])
    for incidence in range(
        response.body_row_start[body] + lane,
        response.body_row_start[body + wp.int32(1)],
        wp.int32(_CONTACT_RHS_BLOCK_DIM),
    ):
        row = response.body_rows[incidence]
        joint = response.row_joint[row]
        row_body = _row_wrench_for_body(
            body,
            joint,
            response.joint_to_structural[joint],
            response.row_local[row],
            response.joint_parent,
            response.joint_child,
            response.row_wrench0,
            response.row_wrench1,
        )
        for axis in range(3):
            basis = _unit_axis(axis)
            force = basis
            torque = wp.vec3(0.0)
            if angular_basis != wp.int32(0):
                force = wp.vec3(0.0)
                torque = basis
            value = _spatial_dot_mass_inverse(
                row_body,
                force,
                torque,
                bodies.inverse_mass[body],
                inverse_inertia,
            )
            response.rhs[task_offset + (row - row_begin) * wp.int32(GROUPED_RHS_ITEM_WIDTH) + axis] = (
                response.row_scale[row] * value
            )


@wp.kernel(enable_backward=False)
def _build_grouped_endpoint_equality_rhs_kernel(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    task_item: wp.array[wp.int32],
):
    task, lane = wp.tid()
    for item_slot in range(GROUPED_RHS_ITEMS_PER_TASK):
        item = task_item[task * wp.int32(GROUPED_RHS_ITEMS_PER_TASK) + item_slot]
        if item >= wp.int32(0):
            _build_endpoint_equality_rhs(response, bodies, item, lane)


@wp.func
def _endpoint_twist_delta(
    response: DirectContactResponseData,
    column: wp.int32,
    target_endpoint: wp.int32,
    wrench0: wp.spatial_vector,
    wrench1: wp.spatial_vector,
) -> wp.spatial_vector:
    result = wp.spatial_vectorf(0.0)
    for source_endpoint in range(2):
        wrench = wrench0
        if source_endpoint == wp.int32(1):
            wrench = wrench1
        source = source_endpoint * wp.int32(6)
        force = wp.spatial_top(wrench)
        torque = wp.spatial_bottom(wrench)
        result += force[0] * response.endpoint_response[column, target_endpoint, source]
        result += force[1] * response.endpoint_response[column, target_endpoint, source + wp.int32(1)]
        result += force[2] * response.endpoint_response[column, target_endpoint, source + wp.int32(2)]
        result += torque[0] * response.endpoint_response[column, target_endpoint, source + wp.int32(3)]
        result += torque[1] * response.endpoint_response[column, target_endpoint, source + wp.int32(4)]
        result += torque[2] * response.endpoint_response[column, target_endpoint, source + wp.int32(5)]
    return result


@wp.func
def _contact_twist_delta(
    response: DirectContactResponseData,
    contact: wp.int32,
    endpoint: wp.int32,
    delta_coordinate: wp.vec3,
) -> wp.spatial_vector:
    return (
        delta_coordinate[0] * response.contact_response[contact, endpoint, 0]
        + delta_coordinate[1] * response.contact_response[contact, endpoint, 1]
        + delta_coordinate[2] * response.contact_response[contact, endpoint, 2]
    )


@wp.kernel(enable_backward=False)
def _compute_contact_endpoint_response_kernel(
    response: DirectContactResponseData,
    contacts: ContactContainer,
):
    contact, endpoint, axis = wp.tid()
    if response.contact_mechanism[contact] < wp.int32(0):
        return
    normal = cc_get_normal(contacts, contact)
    tangent0 = cc_get_tangent1(contacts, contact)
    direction = normal
    if axis == wp.int32(1):
        direction = tangent0
    elif axis == wp.int32(2):
        direction = wp.cross(normal, tangent0)
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    force0 = -direction
    force1 = direction
    torque0 = wp.cross(r0, force0)
    torque1 = wp.cross(r1, force1)
    wrench0 = wp.spatial_vectorf(force0[0], force0[1], force0[2], torque0[0], torque0[1], torque0[2])
    wrench1 = wp.spatial_vectorf(force1[0], force1[1], force1[2], torque1[0], torque1[1], torque1[2])
    response.contact_response[contact, endpoint, axis] = _endpoint_twist_delta(
        response,
        response.contact_column[contact],
        endpoint,
        wrench0,
        wrench1,
    )


@wp.func
def _contact_pair_inverse_mobility(
    response: DirectContactResponseData,
    contacts: ContactContainer,
    contact: wp.int32,
    direction0: wp.vec3,
    source_axis: wp.int32,
) -> wp.float32:
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    velocity0 = response.contact_response[contact, 0, source_axis]
    velocity1 = response.contact_response[contact, 1, source_axis]
    relative = (
        wp.spatial_top(velocity1)
        + wp.cross(wp.spatial_bottom(velocity1), r1)
        - wp.spatial_top(velocity0)
        - wp.cross(wp.spatial_bottom(velocity0), r0)
    )
    return wp.dot(relative, direction0)


@wp.kernel(enable_backward=False)
def _compute_contact_mobility_kernel(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    contacts: ContactContainer,
):
    contact = wp.tid()
    if response.contact_mechanism[contact] < wp.int32(0):
        return
    normal = cc_get_normal(contacts, contact)
    tangent0 = cc_get_tangent1(contacts, contact)
    tangent1 = wp.cross(normal, tangent0)
    inverse00 = _contact_pair_inverse_mobility(response, contacts, contact, normal, wp.int32(0))
    inverse01 = _contact_pair_inverse_mobility(response, contacts, contact, normal, wp.int32(1))
    inverse02 = _contact_pair_inverse_mobility(response, contacts, contact, normal, wp.int32(2))
    inverse11 = _contact_pair_inverse_mobility(response, contacts, contact, tangent0, wp.int32(1))
    inverse12 = _contact_pair_inverse_mobility(response, contacts, contact, tangent0, wp.int32(2))
    inverse22 = _contact_pair_inverse_mobility(response, contacts, contact, tangent1, wp.int32(2))

    body0 = response.contact_body0[contact]
    body1 = response.contact_body1[contact]
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    unconstrained00 = _unconstrained_pair_cross_mobility(bodies, body0, r0, normal, normal, body1, r1)
    unconstrained11 = _unconstrained_pair_cross_mobility(bodies, body0, r0, tangent0, tangent0, body1, r1)
    unconstrained22 = _unconstrained_pair_cross_mobility(bodies, body0, r0, tangent1, tangent1, body1, r1)
    tolerance0 = wp.float32(64.0 * 1.1920928955078125e-7) * wp.max(wp.abs(inverse00), wp.abs(unconstrained00))
    tolerance1 = wp.float32(64.0 * 1.1920928955078125e-7) * wp.max(wp.abs(inverse11), wp.abs(unconstrained11))
    tolerance2 = wp.float32(64.0 * 1.1920928955078125e-7) * wp.max(wp.abs(inverse22), wp.abs(unconstrained22))
    response.mobility[0, contact] = wp.float32(0.0)
    response.mobility[1, contact] = wp.float32(0.0)
    response.mobility[2, contact] = wp.float32(0.0)
    if inverse00 > wp.max(tolerance0, wp.float32(1.0e-12)):
        response.mobility[0, contact] = wp.float32(1.0) / inverse00
    if inverse11 > wp.max(tolerance1, wp.float32(1.0e-12)):
        response.mobility[1, contact] = wp.float32(1.0) / inverse11
    if inverse22 > wp.max(tolerance2, wp.float32(1.0e-12)):
        response.mobility[2, contact] = wp.float32(1.0) / inverse22
    if response.mobility[0, contact] == wp.float32(0.0):
        inverse01 = wp.float32(0.0)
        inverse02 = wp.float32(0.0)
    if response.mobility[1, contact] == wp.float32(0.0):
        inverse01 = wp.float32(0.0)
        inverse12 = wp.float32(0.0)
    if response.mobility[2, contact] == wp.float32(0.0):
        inverse02 = wp.float32(0.0)
        inverse12 = wp.float32(0.0)
    response.mobility[3, contact] = inverse01
    response.mobility[4, contact] = inverse02
    response.mobility[5, contact] = inverse12


@wp.kernel(enable_backward=False)
def _compute_column_endpoint_response_kernel(
    response: DirectContactResponseData,
    bodies: BodyContainer,
):
    column, target_endpoint, source = wp.tid()
    source_endpoint = source // wp.int32(6)
    source_mechanism = response.column_mechanism0[column]
    target_mechanism = response.column_mechanism0[column]
    if source_endpoint == wp.int32(1):
        source_mechanism = response.column_mechanism1[column]
    if target_endpoint == wp.int32(1):
        target_mechanism = response.column_mechanism1[column]
    mechanism = source_mechanism
    component = source - source_endpoint * wp.int32(6)
    source_body = response.column_body0[column]
    target_body = response.column_body0[column]
    if source_endpoint == wp.int32(1):
        source_body = response.column_body1[column]
    if target_endpoint == wp.int32(1):
        target_body = response.column_body1[column]

    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    axis = component
    if component < wp.int32(3):
        force = _unit_axis(component)
    else:
        axis = component - wp.int32(3)
        torque = _unit_axis(axis)

    corrected_wrench = wp.spatial_vectorf(0.0)
    if source_mechanism >= wp.int32(0) and target_mechanism == source_mechanism:
        item = column * wp.int32(4) + source_endpoint * wp.int32(2)
        item_column = component
        if component >= wp.int32(3):
            item += wp.int32(1)
            item_column -= wp.int32(3)
        for incidence in range(
            response.body_row_start[target_body],
            response.body_row_start[target_body + wp.int32(1)],
        ):
            row = response.body_rows[incidence]
            joint = response.row_joint[row]
            local_row = row - response.mechanism_row_start[mechanism]
            offset = item * response.workspace_stride + local_row * wp.int32(GROUPED_RHS_ITEM_WIDTH) + item_column
            corrected_wrench -= (
                response.row_scale[row]
                * response.solution[offset]
                * _row_wrench_for_body(
                    target_body,
                    joint,
                    response.joint_to_structural[joint],
                    response.row_local[row],
                    response.joint_parent,
                    response.joint_child,
                    response.row_wrench0,
                    response.row_wrench1,
                )
            )
    if target_body == source_body:
        corrected_wrench += wp.spatial_vectorf(force[0], force[1], force[2], torque[0], torque[1], torque[2])
    linear = bodies.inverse_mass[target_body] * wp.spatial_top(corrected_wrench)
    angular = mat33_from_sym6(bodies.inverse_inertia_world[target_body]) * wp.spatial_bottom(corrected_wrench)
    response.endpoint_response[column, target_endpoint, source] = wp.spatial_vectorf(
        linear[0], linear[1], linear[2], angular[0], angular[1], angular[2]
    )


class DirectContactResponse:
    """Packed contact-space response for arbitrary direct mechanisms."""

    def __init__(
        self,
        direct: DirectEqualitySystem,
        contact_capacity: int,
        column_capacity: int,
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
        column_capacity = max(1, int(column_capacity))
        self.active_mechanisms = tuple(active_mechanisms)
        item_capacity = 4 * column_capacity
        self.contact_batch = direct.solver.create_grouped_rhs_batch(item_capacity, 2 * column_capacity)
        self.contact_mechanism = wp.full(capacity, -1, dtype=wp.int32, device=device)
        self.contact_column = wp.full(capacity, -1, dtype=wp.int32, device=device)
        self.contact_body0 = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.contact_body1 = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.column_mechanism0 = wp.full(column_capacity, -1, dtype=wp.int32, device=device)
        self.column_mechanism1 = wp.full(column_capacity, -1, dtype=wp.int32, device=device)
        self.column_body0 = wp.zeros(column_capacity, dtype=wp.int32, device=device)
        self.column_body1 = wp.zeros(column_capacity, dtype=wp.int32, device=device)
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
        self.data.contact_column = self.contact_column
        self.data.contact_body0 = self.contact_body0
        self.data.contact_body1 = self.contact_body1
        self.data.column_mechanism0 = self.column_mechanism0
        self.data.column_mechanism1 = self.column_mechanism1
        self.data.column_body0 = self.column_body0
        self.data.column_body1 = self.column_body1
        self.data.workspace_stride = wp.int32(self.contact_batch.item_workspace_stride)
        self.data.rhs = self.contact_batch.rhs
        self.data.solution = self.contact_batch.solution
        self.data.gram = self.contact_batch.gram
        # A manifold's response is fully described by the two endpoint
        # twists under twelve endpoint spatial-wrench basis vectors.
        self.endpoint_response_dim = (column_capacity, 2, 12)
        self.data.endpoint_response = wp.zeros(self.endpoint_response_dim, dtype=wp.spatial_vector, device=device)
        self.contact_response_dim = (capacity, 2, 3)
        self.data.contact_response = wp.zeros(self.contact_response_dim, dtype=wp.spatial_vector, device=device)
        self.data.accumulated_solution = wp.zeros(len(topology.row_joint), dtype=wp.float32, device=device)
        self.data.mobility = wp.zeros((6, capacity), dtype=wp.float32, device=device)
        self.data.delta_coordinate = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.data.delta_wrench = wp.zeros((column_capacity, 2), dtype=wp.spatial_vector, device=device)
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
        """Solve endpoint spatial responses and form active contact blocks."""
        wp.launch_tiled(
            _build_grouped_endpoint_equality_rhs_kernel,
            dim=self.contact_batch.task_capacity,
            block_dim=_CONTACT_RHS_BLOCK_DIM,
            inputs=[self.data, self.direct.bodies, self.contact_batch.task_item],
            device=self.direct.model.device,
        )
        self.contact_batch.solve()
        wp.launch(
            _compute_column_endpoint_response_kernel,
            dim=self.endpoint_response_dim,
            inputs=[self.data, self.direct.bodies],
            device=self.direct.model.device,
        )
        wp.launch(
            _compute_contact_endpoint_response_kernel,
            dim=self.contact_response_dim,
            inputs=[self.data, contacts],
            device=self.direct.model.device,
        )
        wp.launch(
            _compute_contact_mobility_kernel,
            dim=self.contact_mechanism.shape[0],
            inputs=[self.data, self.direct.bodies, contacts],
            device=self.direct.model.device,
        )


__all__ = ["DirectContactResponse", "DirectContactResponseData"]
