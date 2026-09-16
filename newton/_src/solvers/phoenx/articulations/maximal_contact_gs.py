# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation-run contact GS for factored maximal joint trees."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.articulations.maximal_contact_response import (
    MaximalContactResponse,
    MaximalContactResponseData,
    apply_maximal_contact_impulse_thread,
    maximal_contact_pair_cross_inverse_mass_offsets,
)
from newton._src.solvers.phoenx.articulations.maximal_projector import (
    _TREE_WIDTH,
    MaximalTreeProjectorData,
    _sync_tree,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_friction,
    contact_get_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_pd_bias,
    cc_get_pd_eff_soft,
    cc_get_pd_gamma,
    cc_get_r0,
    cc_get_r1,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_projection import (
    contact_project_friction_metric,
    contact_project_normal_velocity_update,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64

_INT64_MAX = 0x7FFFFFFFFFFFFFFF


@wp.kernel(enable_backward=False)
def rebase_owned_contact_levers_kernel(
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    bodies: BodyContainer,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
):
    scheduled = wp.tid()
    if scheduled >= section_end[section_end.shape[0] - 1]:
        return
    column = scheduled_column[scheduled]
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    shift0 = bodies.position_prev_substep[body0] - bodies.position[body0]
    shift1 = bodies.position_prev_substep[body1] - bodies.position[body1]
    first = contact_get_contact_first(columns, column)
    count = contact_get_contact_count(columns, column)
    for contact in range(first, first + count):
        # Retain the prepared world-space impulse point while COMs advance.
        # Otherwise opposing impulses act at different world points.
        cc_set_r0(contacts, contact, cc_get_r0(contacts, contact) + shift0)
        cc_set_r1(contacts, contact, cc_get_r1(contacts, contact) + shift1)


@wp.kernel(enable_backward=False)
def _build_maximal_contact_schedule_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    response: MaximalContactResponseData,
    num_columns: wp.array[wp.int32],
    key_stride: wp.int64,
    keys: wp.array[wp.int64],
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
    reset_owner: wp.bool,
):
    column = wp.tid()
    scheduled_column[column] = column
    if reset_owner:
        columns.articulation_owner[column] = wp.int32(-1)
    if column >= num_columns[0]:
        keys[column] = wp.int64(_INT64_MAX)
        return
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    articulation0 = response.body_articulation[body0]
    articulation1 = response.body_articulation[body1]
    owner = wp.int32(-1)
    if articulation0 >= wp.int32(0):
        if articulation1 == articulation0 or (
            articulation1 < wp.int32(0) and bodies.inverse_mass[body1] == wp.float32(0.0)
        ):
            owner = articulation0
    if articulation1 >= wp.int32(0):
        if articulation0 == articulation1 or (
            articulation0 < wp.int32(0) and bodies.inverse_mass[body0] == wp.float32(0.0)
        ):
            owner = articulation1
    if owner < wp.int32(0):
        keys[column] = wp.int64(_INT64_MAX)
        return
    columns.articulation_owner[column] = owner
    keys[column] = wp.int64(owner) * key_stride + wp.int64(column)
    wp.atomic_add(section_end, owner, wp.int32(1))


@wp.func
def _set_spatial_impulse(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    articulation: wp.int32,
    body: wp.int32,
    force: wp.vec3f,
    torque: wp.vec3f,
):
    if body < wp.int32(0):
        return
    lane = response.body_lane[body]
    if lane >= wp.int32(0) and response.body_articulation[body] == articulation:
        value = response.impulse[articulation, lane]
        response.impulse[articulation, lane] = value + wp.spatial_vectorf(
            force[0],
            force[1],
            force[2],
            torque[0],
            torque[1],
            torque[2],
        )


@wp.func
def _apply_accumulated_impulse(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    dynamic_accumulated_impulse: wp.array[wp.float32],
    articulation: wp.int32,
    lane: wp.int32,
):
    body_count = tree.body_count[articulation]
    apply_maximal_contact_impulse_thread(articulation, lane, tree, response)
    if lane < body_count:
        row = tree.dynamic_row[articulation, lane]
        if row >= wp.int32(0):
            # Preserve the virtual generalized momentum used by the direct
            # equality row when a constrained contact changes joint velocity.
            dynamic_accumulated_impulse[row] -= (
                tree.generalized_mass[articulation, lane] * response.joint_velocity[articulation, lane]
            )
        body = tree.body_slot[articulation, lane]
        delta = response.velocity[articulation, lane]
        bodies.velocity[body] += wp.spatial_top(delta)
        bodies.angular_velocity[body] += wp.spatial_bottom(delta)
    _sync_tree()


@wp.func
def _vec3_double(value: wp.vec3f):
    return wp.vec3d(wp.float64(value[0]), wp.float64(value[1]), wp.float64(value[2]))


@wp.func
def _contact_velocity_branch(
    tree: MaximalTreeProjectorData,
    bodies: BodyContainer,
    articulation: wp.int32,
    lane: wp.int32,
    force: wp.vec3d,
    torque: wp.vec3d,
):
    parent = tree.parent[articulation, lane]
    body = tree.body_slot[articulation, lane]
    parent_body = tree.body_slot[articulation, parent]
    motion = tree.motion[articulation, lane]
    axis = wp.vec3d(wp.float64(motion[3]), wp.float64(motion[4]), wp.float64(motion[5]))
    linear = wp.vec3d(wp.float64(motion[0]), wp.float64(motion[1]), wp.float64(motion[2]))
    joint_speed = wp.dot(
        axis, _vec3_double(bodies.angular_velocity[body]) - _vec3_double(bodies.angular_velocity[parent_body])
    )
    velocity = (wp.dot(linear, force) + wp.dot(axis, torque)) * joint_speed
    return torque - wp.cross(_vec3_double(tree.shift[articulation, lane]), force), velocity


@wp.func
def _contact_row_velocity(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    body0: wp.int32,
    body1: wp.int32,
    r0: wp.vec3f,
    r1: wp.vec3f,
    direction: wp.vec3f,
):
    """Evaluate a contact row in joint coordinates without large velocity cancellation."""
    force0 = -_vec3_double(direction)
    force1 = _vec3_double(direction)
    torque0 = wp.cross(_vec3_double(r0), force0)
    torque1 = wp.cross(_vec3_double(r1), force1)
    lane0 = response.body_lane[body0]
    lane1 = response.body_lane[body1]
    articulation0 = response.body_articulation[body0]
    articulation1 = response.body_articulation[body1]
    velocity = wp.float64(0.0)
    if lane0 >= wp.int32(0) and lane1 >= wp.int32(0) and articulation0 == articulation1:
        while lane0 != lane1:
            if tree.depth[articulation0, lane0] >= tree.depth[articulation0, lane1]:
                torque0, increment = _contact_velocity_branch(tree, bodies, articulation0, lane0, force0, torque0)
                velocity += increment
                lane0 = tree.parent[articulation0, lane0]
            else:
                torque1, increment = _contact_velocity_branch(tree, bodies, articulation0, lane1, force1, torque1)
                velocity += increment
                lane1 = tree.parent[articulation0, lane1]
        common_body = tree.body_slot[articulation0, lane0]
        velocity += wp.dot(force0 + force1, _vec3_double(bodies.velocity[common_body]))
        velocity += wp.dot(torque0 + torque1, _vec3_double(bodies.angular_velocity[common_body]))
    else:
        while lane0 > wp.int32(0):
            torque0, increment = _contact_velocity_branch(tree, bodies, articulation0, lane0, force0, torque0)
            velocity += increment
            lane0 = tree.parent[articulation0, lane0]
        while lane1 > wp.int32(0):
            torque1, increment = _contact_velocity_branch(tree, bodies, articulation1, lane1, force1, torque1)
            velocity += increment
            lane1 = tree.parent[articulation1, lane1]
        source0 = body0
        source1 = body1
        if lane0 >= wp.int32(0):
            source0 = tree.body_slot[articulation0, lane0]
        if lane1 >= wp.int32(0):
            source1 = tree.body_slot[articulation1, lane1]
        velocity += wp.dot(force0, _vec3_double(bodies.velocity[source0])) + wp.dot(
            torque0, _vec3_double(bodies.angular_velocity[source0])
        )
        velocity += wp.dot(force1, _vec3_double(bodies.velocity[source1])) + wp.dot(
            torque1, _vec3_double(bodies.angular_velocity[source1])
        )
    return wp.float32(velocity)


@wp.func
def _write_exact_contact_mobility(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    contacts: ContactContainer,
    body0: wp.int32,
    body1: wp.int32,
    contact: wp.int32,
    mobility: wp.array2d[wp.float32],
):
    normal = cc_get_normal(contacts, contact)
    tangent0 = cc_get_tangent1(contacts, contact)
    tangent1 = wp.cross(normal, tangent0)
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    # Reconstructing world points here rounds the levers differently from
    # the torque applied below, particularly for nearly blocked contacts.
    inverse_n = maximal_contact_pair_cross_inverse_mass_offsets(
        tree, response, bodies, body0, r0, -normal, -normal, body1, r1, normal, normal
    )
    inverse_t0 = maximal_contact_pair_cross_inverse_mass_offsets(
        tree, response, bodies, body0, r0, -tangent0, -tangent0, body1, r1, tangent0, tangent0
    )
    inverse_t1 = maximal_contact_pair_cross_inverse_mass_offsets(
        tree, response, bodies, body0, r0, -tangent1, -tangent1, body1, r1, tangent1, tangent1
    )
    inverse_nt0 = maximal_contact_pair_cross_inverse_mass_offsets(
        tree,
        response,
        bodies,
        body0,
        r0,
        -normal,
        -tangent0,
        body1,
        r1,
        normal,
        tangent0,
    )
    inverse_nt1 = maximal_contact_pair_cross_inverse_mass_offsets(
        tree,
        response,
        bodies,
        body0,
        r0,
        -normal,
        -tangent1,
        body1,
        r1,
        normal,
        tangent1,
    )
    inverse_t01 = maximal_contact_pair_cross_inverse_mass_offsets(
        tree,
        response,
        bodies,
        body0,
        r0,
        -tangent0,
        -tangent1,
        body1,
        r1,
        tangent0,
        tangent1,
    )
    effective_n = wp.float32(0.0)
    effective_t0 = wp.float32(0.0)
    effective_t1 = wp.float32(0.0)
    if inverse_n > wp.float32(1.0e-12):
        effective_n = wp.float32(1.0) / inverse_n
    if inverse_t0 > wp.float32(1.0e-12):
        effective_t0 = wp.float32(1.0) / inverse_t0
    if inverse_t1 > wp.float32(1.0e-12):
        effective_t1 = wp.float32(1.0) / inverse_t1
    mobility[0, contact] = effective_n
    mobility[1, contact] = effective_t0
    mobility[2, contact] = effective_t1
    mobility[3, contact] = inverse_nt0
    mobility[4, contact] = inverse_nt1
    mobility[5, contact] = inverse_t01


@wp.kernel(enable_backward=False)
def refresh_maximal_contact_mobility_kernel(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
    mobility: wp.array2d[wp.float32],
):
    tid = wp.tid()
    articulation = tid // wp.int32(_TREE_WIDTH)
    lane = tid - articulation * wp.int32(_TREE_WIDTH)
    begin = wp.int32(0)
    if articulation > wp.int32(0):
        begin = section_end[articulation - wp.int32(1)]
    end = section_end[articulation]
    for scheduled in range(begin, end):
        column = scheduled_column[scheduled]
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        first = contact_get_contact_first(columns, column)
        count = contact_get_contact_count(columns, column)
        # Contact slots are disjoint; factor and geometry inputs are read-only.
        for offset in range(lane, count, wp.int32(_TREE_WIDTH)):
            _write_exact_contact_mobility(
                tree,
                response,
                bodies,
                contacts,
                body0,
                body1,
                first + offset,
                mobility,
            )


@wp.kernel(enable_backward=False)
def iterate_maximal_contact_runs_kernel(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    dynamic_accumulated_impulse: wp.array[wp.float32],
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    inverse_dt: wp.float32,
    sor_boost: wp.float32,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
    mobility: wp.array2d[wp.float32],
    use_bias: wp.bool,
):
    tid = wp.tid()
    articulation = tid // wp.int32(_TREE_WIDTH)
    lane = tid - articulation * wp.int32(_TREE_WIDTH)
    begin = wp.int32(0)
    if articulation > wp.int32(0):
        begin = section_end[articulation - wp.int32(1)]
    end = section_end[articulation]
    dt = wp.float32(1.0) / inverse_dt
    _, mass_coeff, impulse_coeff = soft_constraint_coefficients(DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt)

    for scheduled in range(begin, end):
        column = scheduled_column[scheduled]
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        friction_static = contact_get_friction(columns, column)
        friction_dynamic = contact_get_friction_dynamic(columns, column)
        first = contact_get_contact_first(columns, column)
        count = contact_get_contact_count(columns, column)
        # Apply each correction before visiting the next row. Reconstructing
        # every previous point response costs quadratic work per manifold and
        # subtracts large common-root terms for internal contacts.
        for offset in range(count):
            if lane < tree.body_count[articulation]:
                response.impulse[articulation, lane] = wp.spatial_vectorf(0.0)
            _sync_tree()
            contact = first + offset
            impulse = wp.vec3f(0.0)
            r0 = wp.vec3f(0.0)
            r1 = wp.vec3f(0.0)
            if lane == wp.int32(0):
                normal = cc_get_normal(contacts, contact)
                tangent0 = cc_get_tangent1(contacts, contact)
                tangent1 = wp.cross(normal, tangent0)
                r0 = cc_get_r0(contacts, contact)
                r1 = cc_get_r1(contacts, contact)
                normal_velocity = _contact_row_velocity(tree, response, bodies, body0, body1, r0, r1, normal)
                tangent_velocity0 = _contact_row_velocity(tree, response, bodies, body0, body1, r0, r1, tangent0)
                tangent_velocity1 = _contact_row_velocity(tree, response, bodies, body0, body1, r0, r1, tangent1)
                bias = cc_get_bias(contacts, contact)
                speculative = bias > wp.float32(0.0)
                if not use_bias:
                    bias = wp.float32(0.0)
                elif bias < wp.float32(0.0) and mobility[0, contact] > wp.float32(1.0e-12):
                    # Project the unconstrained recovery impulse through the joints.
                    # Enforcing the full recovery speed along a nearly blocked normal
                    # turns small overlaps into large joint motions. Physical velocity
                    # impulses below still use the constrained effective mass.
                    bias *= wp.min(wp.float32(1.0), cc_get_eff_n(contacts, contact) / mobility[0, contact])
                if use_bias or not speculative:
                    row_mass_coeff = mass_coeff
                    row_impulse_coeff = impulse_coeff
                    if speculative or not use_bias:
                        row_mass_coeff = wp.float32(1.0)
                        row_impulse_coeff = wp.float32(0.0)

                    pd_eff = cc_get_pd_eff_soft(contacts, contact)
                    pd_gamma = cc_get_pd_gamma(contacts, contact)
                    pd_bias = cc_get_pd_bias(contacts, contact)
                    if pd_eff > wp.float32(0.0):
                        pd_eff = wp.float32(0.0)
                        if mobility[0, contact] > wp.float32(1.0e-12):
                            projected_inverse_mass = wp.float32(1.0) / mobility[0, contact]
                            pd_eff = wp.float32(1.0) / (projected_inverse_mass + pd_gamma)
                    normal_impulse = -cc_get_normal_lambda(contacts, contact) * normal
                    if mobility[0, contact] > wp.float32(1.0e-12):
                        normal_impulse = contact_project_normal_velocity_update(
                            contacts,
                            contact,
                            normal,
                            normal_velocity,
                            mobility[0, contact],
                            bias,
                            row_mass_coeff,
                            row_impulse_coeff,
                            sor_boost,
                            pd_eff,
                            pd_gamma,
                            pd_bias,
                        )
                    else:
                        cc_set_normal_lambda(contacts, contact, wp.float32(0.0))
                    normal_delta = wp.dot(normal_impulse, normal)
                    normal_lambda = cc_get_normal_lambda(contacts, contact)
                    normal_load = normal_lambda
                    if pd_eff <= wp.float32(0.0):
                        normal_load += row_mass_coeff * mobility[0, contact] * bias * sor_boost
                        normal_load = wp.clamp(normal_load, wp.float32(0.0), normal_lambda)

                    rhs0 = tangent_velocity0 + mobility[3, contact] * normal_delta
                    rhs1 = tangent_velocity1 + mobility[4, contact] * normal_delta
                    if use_bias:
                        rhs0 += cc_get_bias_t1(contacts, contact)
                        rhs1 += cc_get_bias_t2(contacts, contact)

                    effective0 = mobility[1, contact]
                    effective1 = mobility[2, contact]
                    inverse00 = wp.float32(0.0)
                    inverse11 = wp.float32(0.0)
                    if effective0 > wp.float32(1.0e-12):
                        inverse00 = wp.float32(1.0) / effective0
                    if effective1 > wp.float32(1.0e-12):
                        inverse11 = wp.float32(1.0) / effective1
                    inverse01 = mobility[5, contact]
                    lambda0 = cc_get_tangent1_lambda(contacts, contact)
                    lambda1 = cc_get_tangent2_lambda(contacts, contact)
                    tangents = contact_project_friction_metric(
                        inverse00,
                        inverse01,
                        inverse11,
                        sor_boost * rhs0,
                        sor_boost * rhs1,
                        lambda0,
                        lambda1,
                        friction_static * normal_load,
                        friction_dynamic * normal_load,
                    )
                    cc_set_tangent1_lambda(contacts, contact, tangents[0])
                    cc_set_tangent2_lambda(contacts, contact, tangents[1])
                    impulse = normal_impulse + (tangents[0] - lambda0) * tangent0 + (tangents[1] - lambda1) * tangent1
                response.contact_active[articulation] = wp.int32(
                    impulse[0] != wp.float32(0.0) or impulse[1] != wp.float32(0.0) or impulse[2] != wp.float32(0.0)
                )
                _set_spatial_impulse(tree, response, articulation, body0, -impulse, -wp.cross(r0, impulse))
                _set_spatial_impulse(tree, response, articulation, body1, impulse, wp.cross(r1, impulse))
            _sync_tree()
            if response.contact_active[articulation] != wp.int32(0):
                _apply_accumulated_impulse(
                    tree,
                    response,
                    bodies,
                    dynamic_accumulated_impulse,
                    articulation,
                    lane,
                )


@wp.kernel(enable_backward=False)
def warm_start_maximal_contact_runs_kernel(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    dynamic_accumulated_impulse: wp.array[wp.float32],
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
):
    tid = wp.tid()
    articulation = tid // wp.int32(_TREE_WIDTH)
    lane = tid - articulation * wp.int32(_TREE_WIDTH)
    begin = wp.int32(0)
    if articulation > wp.int32(0):
        begin = section_end[articulation - wp.int32(1)]
    end = section_end[articulation]
    for scheduled in range(begin, end):
        column = scheduled_column[scheduled]
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        first = contact_get_contact_first(columns, column)
        count = contact_get_contact_count(columns, column)
        for offset in range(count):
            if lane < tree.body_count[articulation]:
                response.impulse[articulation, lane] = wp.spatial_vectorf(0.0)
            _sync_tree()
            if lane == wp.int32(0):
                contact = first + offset
                normal = cc_get_normal(contacts, contact)
                tangent = cc_get_tangent1(contacts, contact)
                impulse = (
                    cc_get_normal_lambda(contacts, contact) * normal
                    + cc_get_tangent1_lambda(contacts, contact) * tangent
                    + cc_get_tangent2_lambda(contacts, contact) * wp.cross(normal, tangent)
                )
                r0 = cc_get_r0(contacts, contact)
                r1 = cc_get_r1(contacts, contact)
                response.contact_active[articulation] = wp.int32(
                    impulse[0] != wp.float32(0.0) or impulse[1] != wp.float32(0.0) or impulse[2] != wp.float32(0.0)
                )
                _set_spatial_impulse(tree, response, articulation, body0, -impulse, -wp.cross(r0, impulse))
                _set_spatial_impulse(tree, response, articulation, body1, impulse, wp.cross(r1, impulse))
            _sync_tree()
            if response.contact_active[articulation] != wp.int32(0):
                _apply_accumulated_impulse(tree, response, bodies, dynamic_accumulated_impulse, articulation, lane)


class MaximalContactRunSchedule:
    """Deterministic articulation runs for rigid contact columns."""

    def __init__(self, response: MaximalContactResponse, column_capacity: int, contact_capacity: int):
        self.response = response
        self.projector = response.projector
        self.capacity = max(1, int(column_capacity))
        device = self.projector.model.device
        self.keys = wp.empty(2 * self.capacity, dtype=wp.int64, device=device)
        self.columns = wp.empty(2 * self.capacity, dtype=wp.int32, device=device)
        self.section_end = wp.zeros(
            self.projector.articulation_count,
            dtype=wp.int32,
            device=device,
        )
        self.mobility = wp.zeros(
            (6, max(1, int(contact_capacity))),
            dtype=wp.float32,
            device=device,
        )

    def build(
        self,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        num_columns: wp.array[wp.int32],
        *,
        reset_owner: bool = True,
    ) -> None:
        """Group immutable rigid contacts by articulation."""
        self.section_end.zero_()
        wp.launch(
            _build_maximal_contact_schedule_kernel,
            dim=self.capacity,
            inputs=[
                columns,
                bodies,
                self.response.data,
                num_columns,
                wp.int64(self.capacity + 1),
                self.keys,
                self.columns,
                self.section_end,
                wp.bool(reset_owner),
            ],
            device=self.projector.model.device,
        )
        sort_variable_length_int64(self.keys, self.columns, num_columns)
        wp.utils.array_scan(self.section_end, self.section_end, inclusive=True)


__all__ = [
    "MaximalContactRunSchedule",
    "iterate_maximal_contact_runs_kernel",
    "refresh_maximal_contact_mobility_kernel",
    "warm_start_maximal_contact_runs_kernel",
]
