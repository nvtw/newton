# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic contact runs using direct equality Schur complements."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.articulations.direct_contact_response import (
    DirectContactResponse,
    DirectContactResponseData,
    _contact_twist_delta,
)
from newton._src.solvers.phoenx.articulations.direct_equality import _row_wrench_for_body
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import (
    GROUPED_RHS_ITEM_WIDTH,
)
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import _block_sync
from newton._src.solvers.phoenx.body import MOTION_KINEMATIC, BodyContainer, mat33_from_sym6
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
def _build_direct_contact_schedule_kernel(
    columns: ContactColumnContainer,
    response: DirectContactResponseData,
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
    if column >= num_columns[0] or columns.articulation_owner[column] >= wp.int32(0):
        keys[column] = wp.int64(_INT64_MAX)
        return
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    mechanism0 = response.body_mechanism[body0]
    mechanism1 = response.body_mechanism[body1]
    constraint_mechanism0 = response.body_constraint_mechanism[body0]
    constraint_mechanism1 = response.body_constraint_mechanism[body1]
    owner = wp.int32(-1)
    if mechanism0 >= wp.int32(0):
        if mechanism1 == mechanism0 or constraint_mechanism1 < wp.int32(0):
            owner = mechanism0
    if mechanism1 >= wp.int32(0):
        if mechanism0 == mechanism1 or constraint_mechanism0 < wp.int32(0):
            owner = mechanism1
    if owner < wp.int32(0):
        keys[column] = wp.int64(_INT64_MAX)
        return
    columns.articulation_owner[column] = owner
    response.column_mechanism[column] = owner
    response.column_body0[column] = body0
    response.column_body1[column] = body1
    keys[column] = wp.int64(owner) * key_stride + wp.int64(column)
    wp.atomic_add(section_end, owner, wp.int32(1))
    first = contact_get_contact_first(columns, column)
    count = contact_get_contact_count(columns, column)
    for offset in range(count):
        contact = first + offset
        response.contact_mechanism[contact] = owner
        response.contact_column[contact] = column
        response.contact_body0[contact] = body0
        response.contact_body1[contact] = body1


@wp.kernel(enable_backward=False)
def _count_direct_contact_rhs_tasks_kernel(
    section_end: wp.array[wp.int32],
    rhs_task_section_end: wp.array[wp.int32],
):
    mechanism = wp.tid()
    begin = wp.int32(0)
    if mechanism > wp.int32(0):
        begin = section_end[mechanism - wp.int32(1)]
    rhs_task_section_end[mechanism] = section_end[mechanism] - begin


@wp.kernel(enable_backward=False)
def _fill_direct_contact_rhs_tasks_kernel(
    response: DirectContactResponseData,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
    rhs_task_section_end: wp.array[wp.int32],
    task_mechanism: wp.array[wp.int32],
    task_item: wp.array[wp.int32],
):
    mechanism = wp.tid()
    column_begin = wp.int32(0)
    task_begin = wp.int32(0)
    if mechanism > wp.int32(0):
        column_begin = section_end[mechanism - wp.int32(1)]
        task_begin = rhs_task_section_end[mechanism - wp.int32(1)]
    column_end = section_end[mechanism]
    for scheduled in range(column_begin, column_end):
        column = scheduled_column[scheduled]
        task = task_begin + scheduled - column_begin
        task_mechanism[task] = mechanism
        body0 = response.column_body0[column]
        body1 = response.column_body1[column]
        for slot in range(4):
            endpoint = wp.int32(slot // 2)
            body = body0
            if endpoint == wp.int32(1):
                body = body1
            if response.body_mechanism[body] == mechanism:
                task_item[task * wp.int32(4) + wp.int32(slot)] = column * wp.int32(4) + wp.int32(slot)


@wp.func
def _apply_raw_contact_impulse(
    bodies: BodyContainer,
    body: wp.int32,
    r: wp.vec3,
    impulse: wp.vec3,
):
    if (
        body <= wp.int32(0)
        or bodies.inverse_mass[body] <= wp.float32(0.0)
        or bodies.motion_type[body] == MOTION_KINEMATIC
    ):
        return
    bodies.velocity[body] += bodies.inverse_mass[body] * impulse
    bodies.angular_velocity[body] += mat33_from_sym6(bodies.inverse_inertia_world[body]) * wp.cross(r, impulse)


@wp.func
def _body_velocity_with_deferred_equality(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    mechanism: wp.int32,
    body: wp.int32,
):
    linear = bodies.velocity[body]
    angular = bodies.angular_velocity[body]
    if response.body_mechanism[body] != mechanism:
        return wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])

    wrench = wp.spatial_vectorf(0.0)
    for incidence in range(response.body_row_start[body], response.body_row_start[body + wp.int32(1)]):
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
        wrench -= response.row_scale[row] * response.accumulated_solution[row] * row_body
    linear += bodies.inverse_mass[body] * wp.spatial_top(wrench)
    angular += mat33_from_sym6(bodies.inverse_inertia_world[body]) * wp.spatial_bottom(wrench)
    return wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.func
def _apply_deferred_equality_thread(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    mechanism: wp.int32,
    lane: wp.int32,
):
    begin = response.mechanism_body_start[mechanism]
    end = response.mechanism_body_start[mechanism + wp.int32(1)]
    for local_body in range(lane, end - begin, wp.block_dim()):
        body = response.mechanism_body[begin + local_body]
        corrected = _body_velocity_with_deferred_equality(response, bodies, mechanism, body)
        bodies.velocity[body] = wp.spatial_top(corrected)
        bodies.angular_velocity[body] = wp.spatial_bottom(corrected)


@wp.func
def _make_spatial_wrench(force: wp.vec3, torque: wp.vec3) -> wp.spatial_vector:
    return wp.spatial_vectorf(force[0], force[1], force[2], torque[0], torque[1], torque[2])


@wp.func
def _row_correction_from_wrenches(
    response: DirectContactResponseData,
    mechanism: wp.int32,
    column: wp.int32,
    local_row: wp.int32,
    wrench0: wp.spatial_vector,
    wrench1: wp.spatial_vector,
) -> wp.float32:
    correction = wp.float32(0.0)
    body0 = response.column_body0[column]
    body1 = response.column_body1[column]
    for endpoint in range(2):
        body = body0
        wrench = wrench0
        if endpoint == wp.int32(1):
            body = body1
            wrench = wrench1
        if response.body_mechanism[body] == mechanism:
            item = column * wp.int32(4) + endpoint * wp.int32(2)
            force = wp.spatial_top(wrench)
            torque = wp.spatial_bottom(wrench)
            base = item * response.workspace_stride + local_row * wp.int32(GROUPED_RHS_ITEM_WIDTH)
            correction += response.solution[base] * force[0]
            correction += response.solution[base + wp.int32(1)] * force[1]
            correction += response.solution[base + wp.int32(2)] * force[2]
            base += response.workspace_stride
            correction += response.solution[base] * torque[0]
            correction += response.solution[base + wp.int32(1)] * torque[1]
            correction += response.solution[base + wp.int32(2)] * torque[2]
    return correction


@wp.kernel(enable_backward=False)
def warm_start_direct_contact_runs_kernel(
    active_mechanism: wp.array[wp.int32],
    response: DirectContactResponseData,
    bodies: BodyContainer,
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
):
    task, lane = wp.tid()
    mechanism = active_mechanism[task]
    row_begin = response.mechanism_row_start[mechanism]
    row_end = response.mechanism_row_start[mechanism + wp.int32(1)]
    for local_row in range(lane, row_end - row_begin, wp.block_dim()):
        response.accumulated_solution[row_begin + local_row] = wp.float32(0.0)
    _block_sync()

    begin = wp.int32(0)
    if mechanism > wp.int32(0):
        begin = section_end[mechanism - wp.int32(1)]
    end = section_end[mechanism]
    for scheduled in range(begin, end):
        column = scheduled_column[scheduled]
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        first = contact_get_contact_first(columns, column)
        count = contact_get_contact_count(columns, column)
        if lane == wp.int32(0):
            wrench0 = wp.spatial_vectorf(0.0)
            wrench1 = wp.spatial_vectorf(0.0)
            for offset in range(count):
                contact = first + offset
                normal = cc_get_normal(contacts, contact)
                tangent = cc_get_tangent1(contacts, contact)
                normal_lambda = cc_get_normal_lambda(contacts, contact)
                if response.mobility[0, contact] <= wp.float32(1.0e-12):
                    normal_lambda = wp.float32(0.0)
                    cc_set_normal_lambda(contacts, contact, normal_lambda)
                impulse = (
                    normal_lambda * normal
                    + cc_get_tangent1_lambda(contacts, contact) * tangent
                    + cc_get_tangent2_lambda(contacts, contact) * wp.cross(normal, tangent)
                )
                r0 = cc_get_r0(contacts, contact)
                r1 = cc_get_r1(contacts, contact)
                _apply_raw_contact_impulse(bodies, body0, r0, -impulse)
                _apply_raw_contact_impulse(bodies, body1, r1, impulse)
                wrench0 += _make_spatial_wrench(-impulse, -wp.cross(r0, impulse))
                wrench1 += _make_spatial_wrench(impulse, wp.cross(r1, impulse))
            response.delta_wrench[column, 0] = wrench0
            response.delta_wrench[column, 1] = wrench1
        _block_sync()
        wrench0 = response.delta_wrench[column, 0]
        wrench1 = response.delta_wrench[column, 1]
        for local_row in range(lane, row_end - row_begin, wp.block_dim()):
            row = row_begin + local_row
            correction = _row_correction_from_wrenches(
                response,
                mechanism,
                column,
                local_row,
                wrench0,
                wrench1,
            )
            response.accumulated_solution[row] += correction
            response.accumulated_impulse[row] -= response.row_scale[row] * correction
        _block_sync()
    _apply_deferred_equality_thread(response, bodies, mechanism, lane)


@wp.kernel(enable_backward=False)
def iterate_direct_contact_runs_kernel(
    active_mechanism: wp.array[wp.int32],
    response: DirectContactResponseData,
    bodies: BodyContainer,
    columns: ContactColumnContainer,
    contacts: ContactContainer,
    inverse_dt: wp.float32,
    sor_boost: wp.float32,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
    iteration_count: wp.int32,
    use_bias: wp.bool,
):
    task, lane = wp.tid()
    mechanism = active_mechanism[task]
    row_begin = response.mechanism_row_start[mechanism]
    row_end = response.mechanism_row_start[mechanism + wp.int32(1)]
    for _iteration in range(iteration_count):
        for local_row in range(lane, row_end - row_begin, wp.block_dim()):
            response.accumulated_solution[row_begin + local_row] = wp.float32(0.0)
        _block_sync()

        begin = wp.int32(0)
        if mechanism > wp.int32(0):
            begin = section_end[mechanism - wp.int32(1)]
        end = section_end[mechanism]
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
            velocity_body0 = wp.spatial_vectorf(0.0)
            velocity_body1 = wp.spatial_vectorf(0.0)
            if lane == wp.int32(0):
                velocity_body0 = _body_velocity_with_deferred_equality(response, bodies, mechanism, body0)
                velocity_body1 = _body_velocity_with_deferred_equality(response, bodies, mechanism, body1)
            if lane == wp.int32(0):
                wrench0 = wp.spatial_vectorf(0.0)
                wrench1 = wp.spatial_vectorf(0.0)
                for offset in range(count):
                    contact = first + offset
                    normal = cc_get_normal(contacts, contact)
                    tangent0 = cc_get_tangent1(contacts, contact)
                    tangent1 = wp.cross(normal, tangent0)
                    r0 = cc_get_r0(contacts, contact)
                    r1 = cc_get_r1(contacts, contact)
                    relative_velocity = (
                        wp.spatial_top(velocity_body1)
                        + wp.cross(wp.spatial_bottom(velocity_body1), r1)
                        - wp.spatial_top(velocity_body0)
                        - wp.cross(wp.spatial_bottom(velocity_body0), r0)
                    )
                    velocity0 = wp.dot(relative_velocity, normal)
                    velocity1 = wp.dot(relative_velocity, tangent0)
                    velocity2 = wp.dot(relative_velocity, tangent1)
                    bias = cc_get_bias(contacts, contact)
                    speculative = bias > wp.float32(0.0)
                    if not use_bias:
                        bias = wp.float32(0.0)
                    elif bias < wp.float32(0.0) and response.mobility[0, contact] > wp.float32(1.0e-12):
                        # Project the unconstrained recovery impulse through the joints.
                        # Enforcing the full recovery speed along a nearly blocked normal
                        # turns small overlaps into large joint motions. Physical velocity
                        # impulses below still use the constrained effective mass.
                        bias *= wp.min(wp.float32(1.0), cc_get_eff_n(contacts, contact) / response.mobility[0, contact])
                    delta_coordinate = wp.vec3(0.0)
                    impulse = wp.vec3(0.0)
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
                            if response.mobility[0, contact] > wp.float32(1.0e-12):
                                pd_eff = wp.float32(1.0) / (wp.float32(1.0) / response.mobility[0, contact] + pd_gamma)
                        normal_impulse = -cc_get_normal_lambda(contacts, contact) * normal
                        if response.mobility[0, contact] > wp.float32(1.0e-12):
                            normal_impulse = contact_project_normal_velocity_update(
                                contacts,
                                contact,
                                normal,
                                velocity0,
                                response.mobility[0, contact],
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
                            normal_load += row_mass_coeff * response.mobility[0, contact] * bias * sor_boost
                            normal_load = wp.clamp(normal_load, wp.float32(0.0), normal_lambda)
                        rhs0 = velocity1 + response.mobility[3, contact] * normal_delta
                        rhs1 = velocity2 + response.mobility[4, contact] * normal_delta
                        if use_bias:
                            rhs0 += cc_get_bias_t1(contacts, contact)
                            rhs1 += cc_get_bias_t2(contacts, contact)
                        inverse00 = wp.float32(0.0)
                        inverse11 = wp.float32(0.0)
                        if response.mobility[1, contact] > wp.float32(1.0e-12):
                            inverse00 = wp.float32(1.0) / response.mobility[1, contact]
                        if response.mobility[2, contact] > wp.float32(1.0e-12):
                            inverse11 = wp.float32(1.0) / response.mobility[2, contact]
                        inverse01 = response.mobility[5, contact]
                        tangent_old = wp.vec2f(
                            cc_get_tangent1_lambda(contacts, contact),
                            cc_get_tangent2_lambda(contacts, contact),
                        )
                        tangent_new = contact_project_friction_metric(
                            inverse00,
                            inverse01,
                            inverse11,
                            rhs0 * sor_boost,
                            rhs1 * sor_boost,
                            tangent_old[0],
                            tangent_old[1],
                            friction_static * normal_load,
                            friction_dynamic * normal_load,
                        )
                        tangent_delta = tangent_new - tangent_old
                        cc_set_tangent1_lambda(contacts, contact, tangent_new[0])
                        cc_set_tangent2_lambda(contacts, contact, tangent_new[1])
                        delta_coordinate = wp.vec3(normal_delta, tangent_delta[0], tangent_delta[1])
                        impulse = normal_impulse + tangent_delta[0] * tangent0 + tangent_delta[1] * tangent1
                    response.delta_coordinate[contact] = delta_coordinate
                    _apply_raw_contact_impulse(bodies, body0, r0, -impulse)
                    _apply_raw_contact_impulse(bodies, body1, r1, impulse)
                    delta_wrench0 = _make_spatial_wrench(-impulse, -wp.cross(r0, impulse))
                    delta_wrench1 = _make_spatial_wrench(impulse, wp.cross(r1, impulse))
                    wrench0 += delta_wrench0
                    wrench1 += delta_wrench1
                    velocity_body0 += _contact_twist_delta(
                        response,
                        contact,
                        wp.int32(0),
                        delta_coordinate,
                    )
                    velocity_body1 += _contact_twist_delta(
                        response,
                        contact,
                        wp.int32(1),
                        delta_coordinate,
                    )
                response.delta_wrench[column, 0] = wrench0
                response.delta_wrench[column, 1] = wrench1
            # One column is a body-pair manifold. Accumulate its equality
            # correction once after the ordered point solve; linearity makes
            # this identical to updating every joint row after every point,
            # while reducing block barriers from O(points) to O(1).
            _block_sync()
            wrench0 = response.delta_wrench[column, 0]
            wrench1 = response.delta_wrench[column, 1]
            for local_row in range(lane, row_end - row_begin, wp.block_dim()):
                row = row_begin + local_row
                correction = _row_correction_from_wrenches(
                    response,
                    mechanism,
                    column,
                    local_row,
                    wrench0,
                    wrench1,
                )
                response.accumulated_solution[row] += correction
                response.accumulated_impulse[row] -= response.row_scale[row] * correction
            _block_sync()
        _apply_deferred_equality_thread(response, bodies, mechanism, lane)
        _block_sync()


class DirectContactRunSchedule:
    """Group rigid contact columns by direct equality mechanism."""

    def __init__(self, response: DirectContactResponse, column_capacity: int):
        self.response = response
        self.capacity = max(1, int(column_capacity))
        device = response.direct.model.device
        self.keys = wp.empty(2 * self.capacity, dtype=wp.int64, device=device)
        self.columns = wp.empty(2 * self.capacity, dtype=wp.int32, device=device)
        mechanism_count = len(response.active_mechanisms)
        self.section_end = wp.zeros(mechanism_count, dtype=wp.int32, device=device)
        self.rhs_task_section_end = wp.zeros(mechanism_count, dtype=wp.int32, device=device)

    def build(
        self,
        columns: ContactColumnContainer,
        num_columns: wp.array[wp.int32],
        *,
        reset_owner: bool,
    ) -> None:
        """Group immutable contacts without changing another response owner's columns."""
        self.section_end.zero_()
        self.rhs_task_section_end.zero_()
        self.response.contact_mechanism.fill_(-1)
        self.response.contact_column.fill_(-1)
        self.response.column_mechanism.fill_(-1)
        self.response.contact_batch.task_mechanism.fill_(-1)
        self.response.contact_batch.task_item.fill_(-1)
        wp.launch(
            _build_direct_contact_schedule_kernel,
            dim=self.capacity,
            inputs=[
                columns,
                self.response.data,
                num_columns,
                wp.int64(self.capacity + 1),
                self.keys,
                self.columns,
                self.section_end,
                wp.bool(reset_owner),
            ],
            device=self.response.direct.model.device,
        )
        sort_variable_length_int64(self.keys, self.columns, num_columns)
        wp.utils.array_scan(self.section_end, self.section_end, inclusive=True)
        wp.launch(
            _count_direct_contact_rhs_tasks_kernel,
            dim=self.section_end.size,
            inputs=[
                self.section_end,
                self.rhs_task_section_end,
            ],
            device=self.response.direct.model.device,
        )
        wp.utils.array_scan(
            self.rhs_task_section_end,
            self.rhs_task_section_end,
            inclusive=True,
        )
        wp.launch(
            _fill_direct_contact_rhs_tasks_kernel,
            dim=self.section_end.size,
            inputs=[
                self.response.data,
                self.columns,
                self.section_end,
                self.rhs_task_section_end,
                self.response.contact_batch.task_mechanism,
                self.response.contact_batch.task_item,
            ],
            device=self.response.direct.model.device,
        )


__all__ = ["DirectContactRunSchedule", "iterate_direct_contact_runs_kernel"]
