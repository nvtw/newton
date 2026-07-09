# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation-run contact GS for factored maximal joint trees."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.articulations.maximal_contact_response import (
    MaximalContactResponse,
    MaximalContactResponseData,
    apply_maximal_contact_impulse_thread,
    maximal_contact_pair_cross_inverse_mass,
    maximal_contact_pair_inverse_mass,
)
from newton._src.solvers.phoenx.articulations.maximal_projector import (
    MaximalTreeProjectorData,
    _sync_warp,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_block import block_project_friction_delta_sor_2
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
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_pd_eff_soft,
    cc_get_r0,
    cc_get_r1,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_projection import (
    contact_project_normal_velocity_update_no_soft_pd,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64

_WARP_SIZE = 32
_INT64_MAX = 0x7FFFFFFFFFFFFFFF


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
):
    column = wp.tid()
    scheduled_column[column] = column
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
def _apply_point_impulse(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    bodies: BodyContainer,
    articulation: wp.int32,
    lane: wp.int32,
    body0: wp.int32,
    body1: wp.int32,
    r0: wp.vec3f,
    r1: wp.vec3f,
    impulse_on_body1: wp.vec3f,
):
    body_count = tree.body_count[articulation]
    if lane < body_count:
        response.impulse[articulation, lane] = wp.spatial_vectorf(0.0)
    _sync_warp()
    if lane == wp.int32(0):
        _set_spatial_impulse(
            tree,
            response,
            articulation,
            body0,
            -impulse_on_body1,
            -wp.cross(r0, impulse_on_body1),
        )
        _set_spatial_impulse(
            tree,
            response,
            articulation,
            body1,
            impulse_on_body1,
            wp.cross(r1, impulse_on_body1),
        )
    _sync_warp()
    apply_maximal_contact_impulse_thread(articulation, lane, tree, response)
    if lane < body_count:
        body = tree.body_slot[articulation, lane]
        delta = response.velocity[articulation, lane]
        bodies.velocity[body] += wp.spatial_top(delta)
        bodies.angular_velocity[body] += wp.spatial_bottom(delta)
    _sync_warp()


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
    point0 = bodies.position[body0] + r0
    point1 = bodies.position[body1] + r1
    inverse_n = maximal_contact_pair_inverse_mass(tree, response, bodies, body0, point0, -normal, body1, point1, normal)
    inverse_t0 = maximal_contact_pair_inverse_mass(
        tree, response, bodies, body0, point0, -tangent0, body1, point1, tangent0
    )
    inverse_t1 = maximal_contact_pair_inverse_mass(
        tree, response, bodies, body0, point0, -tangent1, body1, point1, tangent1
    )
    inverse_nt0 = maximal_contact_pair_cross_inverse_mass(
        tree,
        response,
        bodies,
        body0,
        point0,
        -normal,
        -tangent0,
        body1,
        point1,
        normal,
        tangent0,
    )
    inverse_nt1 = maximal_contact_pair_cross_inverse_mass(
        tree,
        response,
        bodies,
        body0,
        point0,
        -normal,
        -tangent1,
        body1,
        point1,
        normal,
        tangent1,
    )
    inverse_t01 = maximal_contact_pair_cross_inverse_mass(
        tree,
        response,
        bodies,
        body0,
        point0,
        -tangent0,
        -tangent1,
        body1,
        point1,
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
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
    if lane != wp.int32(0):
        return
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
    articulation = tid // wp.int32(_WARP_SIZE)
    lane = tid - articulation * wp.int32(_WARP_SIZE)
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
        for offset in range(count):
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
                relative_velocity = (
                    bodies.velocity[body1]
                    + wp.cross(bodies.angular_velocity[body1], r1)
                    - bodies.velocity[body0]
                    - wp.cross(bodies.angular_velocity[body0], r0)
                )
                bias = cc_get_bias(contacts, contact)
                speculative = bias > wp.float32(0.0)
                if not use_bias:
                    bias = wp.float32(0.0)
                if (use_bias or not speculative) and cc_get_pd_eff_soft(contacts, contact) <= wp.float32(0.0):
                    row_mass_coeff = mass_coeff
                    row_impulse_coeff = impulse_coeff
                    if speculative or not use_bias:
                        row_mass_coeff = wp.float32(1.0)
                        row_impulse_coeff = wp.float32(0.0)

                    normal_impulse = contact_project_normal_velocity_update_no_soft_pd(
                        contacts,
                        contact,
                        normal,
                        wp.dot(relative_velocity, normal),
                        mobility[0, contact],
                        bias,
                        row_mass_coeff,
                        row_impulse_coeff,
                        sor_boost,
                        wp.float32(0.0),
                        wp.float32(0.0),
                        wp.float32(0.0),
                    )
                    normal_delta = wp.dot(normal_impulse, normal)
                    normal_lambda = cc_get_normal_lambda(contacts, contact)
                    normal_load = normal_lambda + (row_mass_coeff * mobility[0, contact] * bias * sor_boost)
                    normal_load = wp.clamp(normal_load, wp.float32(0.0), normal_lambda)

                    rhs0 = wp.dot(relative_velocity, tangent0) + mobility[3, contact] * normal_delta
                    rhs1 = wp.dot(relative_velocity, tangent1) + mobility[4, contact] * normal_delta
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
                    determinant = inverse00 * inverse11 - inverse01 * inverse01
                    delta0 = wp.float32(0.0)
                    delta1 = wp.float32(0.0)
                    if determinant > wp.float32(1.0e-12) * wp.max(wp.float32(1.0), inverse00 * inverse11):
                        delta0 = -(inverse11 * rhs0 - inverse01 * rhs1) / determinant
                        delta1 = -(-inverse01 * rhs0 + inverse00 * rhs1) / determinant

                    lambda0 = cc_get_tangent1_lambda(contacts, contact)
                    lambda1 = cc_get_tangent2_lambda(contacts, contact)
                    tangents = block_project_friction_delta_sor_2(
                        lambda0,
                        lambda1,
                        delta0,
                        delta1,
                        sor_boost,
                        friction_static * normal_load,
                        friction_dynamic * normal_load,
                    )
                    cc_set_tangent1_lambda(contacts, contact, tangents.lambda_new[0])
                    cc_set_tangent2_lambda(contacts, contact, tangents.lambda_new[1])
                    impulse = normal_impulse + tangents.delta[0] * tangent0 + tangents.delta[1] * tangent1
            _apply_point_impulse(
                tree,
                response,
                bodies,
                articulation,
                lane,
                body0,
                body1,
                r0,
                r1,
                impulse,
            )


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
            ],
            device=self.projector.model.device,
        )
        sort_variable_length_int64(self.keys, self.columns, num_columns)
        wp.utils.array_scan(self.section_end, self.section_end, inclusive=True)


__all__ = [
    "MaximalContactRunSchedule",
    "iterate_maximal_contact_runs_kernel",
    "refresh_maximal_contact_mobility_kernel",
]
