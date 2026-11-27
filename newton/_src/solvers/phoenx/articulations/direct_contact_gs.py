# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic manifold sweeps through exact direct-mechanism responses."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.articulations.direct_contact_response import (
    DirectContactResponse,
    DirectContactResponseData,
    _basis_slot_body,
    _endpoint_basis_item,
    _push_response,
)
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import (
    GROUPED_RHS_ITEM_WIDTH,
    GROUPED_RHS_ITEMS_PER_TASK,
)
from newton._src.solvers.phoenx.body import MOTION_KINEMATIC, BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    block_solve_accumulated_inverse_bounded_1,
)
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
    cc_get_surface_target,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_projection import _contact_project_friction_metric_float32
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64
from newton._src.solvers.phoenx.islands.island_builder import UnionFindIslandBuilder

_INT64_MAX = 0x7FFFFFFFFFFFFFFF


@wp.kernel(enable_backward=False)
def _build_direct_contact_candidates_kernel(
    columns: ContactColumnContainer,
    response: DirectContactResponseData,
    num_columns: wp.array[wp.int32],
    interactions: wp.array2d[wp.int32],
    reset_owner: wp.bool,
):
    column = wp.tid()
    if reset_owner:
        columns.articulation_owner[column] = wp.int32(-1)
    if column >= num_columns[0] or columns.articulation_owner[column] >= wp.int32(0):
        return
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    mechanism0 = response.body_mechanism[body0]
    mechanism1 = response.body_mechanism[body1]
    constraint_mechanism0 = response.body_constraint_mechanism[body0]
    constraint_mechanism1 = response.body_constraint_mechanism[body1]
    if (
        (mechanism0 < wp.int32(0) and constraint_mechanism0 >= wp.int32(0))
        or (mechanism1 < wp.int32(0) and constraint_mechanism1 >= wp.int32(0))
        or (mechanism0 < wp.int32(0) and mechanism1 < wp.int32(0))
    ):
        return
    response.column_mechanism0[column] = mechanism0
    response.column_mechanism1[column] = mechanism1
    response.column_body0[column] = body0
    response.column_body1[column] = body1
    first = mechanism0
    second = mechanism1
    if first < wp.int32(0):
        first = second
        second = wp.int32(-1)
    elif second == first:
        second = wp.int32(-1)
    interactions[column, 0] = first
    interactions[column, 1] = second


@wp.kernel(enable_backward=False)
def _finalize_direct_contact_schedule_kernel(
    columns: ContactColumnContainer,
    response: DirectContactResponseData,
    num_columns: wp.array[wp.int32],
    mechanism_component: wp.array[wp.int32],
    body_bits: wp.int32,
    keys: wp.array[wp.int64],
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
):
    column = wp.tid()
    scheduled_column[column] = column
    mechanism0 = response.column_mechanism0[column]
    mechanism1 = response.column_mechanism1[column]
    if column >= num_columns[0] or (mechanism0 < wp.int32(0) and mechanism1 < wp.int32(0)):
        keys[column] = wp.int64(_INT64_MAX)
        return
    mechanism = mechanism0
    if mechanism < wp.int32(0):
        mechanism = mechanism1
    component = mechanism_component[mechanism]
    columns.articulation_owner[column] = component
    # Manifolds of one body pair are adjacent so a sweep relaxes them as one run.
    # Radix sorting is stable, so equal pairs retain ascending column order.
    key = wp.int64(component)
    if body_bits > wp.int32(0):
        key = (key << wp.int64(body_bits)) | wp.int64(response.column_body0[column])
        key = (key << wp.int64(body_bits)) | wp.int64(response.column_body1[column])
    keys[column] = key
    wp.atomic_add(section_end, component, wp.int32(1))
    first = contact_get_contact_first(columns, column)
    count = contact_get_contact_count(columns, column)
    for offset in range(count):
        contact = first + offset
        response.contact_mechanism[contact] = component
        response.contact_column[contact] = column
        response.contact_body0[contact] = response.column_body0[column]
        response.contact_body1[contact] = response.column_body1[column]


@wp.kernel(enable_backward=False)
def _fill_direct_contact_rhs_tasks_kernel(
    response: DirectContactResponseData,
    task_mechanism: wp.array[wp.int32],
    task_item: wp.array[wp.int32],
):
    column = wp.tid()
    for endpoint in range(2):
        task = column * wp.int32(2) + wp.int32(endpoint)
        mechanism = response.column_mechanism0[column]
        if endpoint == wp.int32(1):
            mechanism = response.column_mechanism1[column]
        if mechanism < wp.int32(0):
            continue
        if response.body_basis_item.shape[0] > wp.int32(0):
            body = response.column_body0[column]
            if endpoint == wp.int32(1):
                body = response.column_body1[column]
            body_task = response.body_basis_task[body]
            if body_task >= wp.int32(0):
                # All writers name the same mechanism; integer atomics keep
                # activation race-free without making contact order relevant.
                wp.atomic_max(task_mechanism, body_task, mechanism)
                wp.atomic_max(response.body_basis_active, body, wp.int32(1))
            continue
        item = column * wp.int32(4) + wp.int32(endpoint) * wp.int32(2)
        task_mechanism[task] = mechanism
        task_item[task * wp.int32(GROUPED_RHS_ITEMS_PER_TASK)] = item
        task_item[task * wp.int32(GROUPED_RHS_ITEMS_PER_TASK) + wp.int32(1)] = item + wp.int32(1)


@wp.func
def _make_spatial_wrench(force: wp.vec3, torque: wp.vec3) -> wp.spatial_vector:
    return wp.spatial_vectorf(force[0], force[1], force[2], torque[0], torque[1], torque[2])


@wp.func
def _raw_contact_twist(bodies: BodyContainer, body: wp.int32, wrench: wp.spatial_vector) -> wp.spatial_vector:
    """Twist change of a body outside every mechanism under a contact wrench."""
    if (
        body <= wp.int32(0)
        or bodies.inverse_mass[body] <= wp.float32(0.0)
        or bodies.motion_type[body] == MOTION_KINEMATIC
    ):
        return wp.spatial_vectorf(0.0)
    linear = bodies.inverse_mass[body] * wp.spatial_top(wrench)
    angular = mat33_from_sym6(bodies.inverse_inertia_world[body]) * wp.spatial_bottom(wrench)
    return wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.func
def _apply_raw_contact_wrench(bodies: BodyContainer, body: wp.int32, wrench: wp.spatial_vector):
    twist = _raw_contact_twist(bodies, body, wrench)
    if body > wp.int32(0) and wp.length_sq(twist) > wp.float32(0.0):
        bodies.velocity[body] += wp.spatial_top(twist)
        bodies.angular_velocity[body] += wp.spatial_bottom(twist)


@wp.func
def _pair_twist_change(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    body: wp.int32,
    slot: wp.int32,
    local_body: wp.int32,
    wrench: wp.spatial_vector,
    other_slot: wp.int32,
    other_wrench: wp.spatial_vector,
    same_mechanism: wp.bool,
) -> wp.spatial_vector:
    """Twist change of one manifold endpoint under both endpoint wrenches."""
    if slot < wp.int32(0):
        return _raw_contact_twist(bodies, body, wrench)
    twist = _push_response(response, slot, local_body, wrench)
    if same_mechanism:
        twist += _push_response(response, other_slot, local_body, other_wrench)
    return twist


@wp.func
def _relax_direct_contact_point(
    response: DirectContactResponseData,
    contacts: ContactContainer,
    contact: wp.int32,
    velocity_body0: wp.spatial_vector,
    velocity_body1: wp.spatial_vector,
    use_bias: wp.bool,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    friction_static: wp.float32,
    friction_dynamic: wp.float32,
    split: wp.float32,
) -> wp.vec3:
    """Relax one point against ``split`` times its exact constrained mobility.

    The impulse and soft-constraint terms shrink together, so the split update
    vanishes exactly when the unsplit Gauss-Seidel update does. Returns the
    impulse change applied to side 1.
    """
    # Load every row of the point before the first store: the stores alias the
    # contact arrays, so loads after them would wait for each round trip in turn.
    normal = cc_get_normal(contacts, contact)
    tangent0 = cc_get_tangent1(contacts, contact)
    r0 = cc_get_r0(contacts, contact)
    r1 = cc_get_r1(contacts, contact)
    normal_mobility = response.mobility[0, contact]
    bias = cc_get_bias(contacts, contact)
    eff_n = cc_get_eff_n(contacts, contact)
    pd_eff = cc_get_pd_eff_soft(contacts, contact)
    pd_gamma = cc_get_pd_gamma(contacts, contact)
    pd_bias = cc_get_pd_bias(contacts, contact)
    surface_target = cc_get_surface_target(contacts, contact)
    lambda_n_old = cc_get_normal_lambda(contacts, contact)
    coupling0 = response.mobility[3, contact]
    coupling1 = response.mobility[4, contact]
    bias_t1 = cc_get_bias_t1(contacts, contact)
    bias_t2 = cc_get_bias_t2(contacts, contact)
    cosine = response.mobility[6, contact]
    sine = response.mobility[7, contact]
    mobility0 = response.mobility[8, contact]
    mobility1 = response.mobility[9, contact]
    tangent_old = wp.vec2f(cc_get_tangent1_lambda(contacts, contact), cc_get_tangent2_lambda(contacts, contact))

    tangent1 = wp.cross(normal, tangent0)
    relative_velocity = (
        wp.spatial_top(velocity_body1)
        + wp.cross(wp.spatial_bottom(velocity_body1), r1)
        - wp.spatial_top(velocity_body0)
        - wp.cross(wp.spatial_bottom(velocity_body0), r0)
    )
    speculative = bias > wp.float32(0.0)
    if not use_bias:
        bias = wp.float32(0.0)
    elif bias < wp.float32(0.0) and normal_mobility > wp.float32(1.0e-12):
        # Project the unconstrained recovery impulse through the joints.
        # Enforcing the full recovery speed along a nearly blocked normal
        # turns small overlaps into large joint motions. Physical velocity
        # impulses below still use the constrained effective mass.
        bias *= wp.min(wp.float32(1.0), eff_n / normal_mobility)
    if not use_bias and speculative:
        return wp.vec3(0.0)
    row_mass_coeff = mass_coeff
    row_impulse_coeff = impulse_coeff / split
    if speculative or not use_bias:
        row_mass_coeff = wp.float32(1.0)
        row_impulse_coeff = wp.float32(0.0)
    if pd_eff > wp.float32(0.0):
        pd_eff = wp.float32(0.0)
        if normal_mobility > wp.float32(1.0e-12):
            pd_eff = wp.float32(1.0) / ((wp.float32(1.0) / normal_mobility + pd_gamma) * split)
    normal_impulse = -lambda_n_old * normal
    normal_load = wp.float32(0.0)
    if normal_mobility > wp.float32(1.0e-12):
        # The normal row of :func:`contact_project_normal_velocity_update`.
        inverse_response = normal_mobility / split
        rhs = wp.dot(relative_velocity, normal) + surface_target[2] + bias
        normal_mass_coeff = row_mass_coeff
        normal_impulse_coeff = row_impulse_coeff
        if pd_eff > wp.float32(0.0):
            inverse_response = pd_eff
            rhs = wp.dot(relative_velocity, normal) + surface_target[2] - pd_bias + pd_gamma * lambda_n_old
            normal_mass_coeff = wp.float32(1.0)
            normal_impulse_coeff = wp.float32(0.0)
        update = block_solve_accumulated_inverse_bounded_1(
            inverse_response,
            rhs,
            lambda_n_old,
            normal_mass_coeff,
            normal_impulse_coeff,
            sor_boost,
            wp.float32(0.0),
            BLOCK_LAMBDA_INF,
        )
        normal_load = update.lambda_new
        normal_impulse = update.delta * normal
    normal_delta = wp.dot(normal_impulse, normal)
    rhs0 = wp.dot(relative_velocity, tangent0) + surface_target[0] + split * coupling0 * normal_delta
    rhs1 = wp.dot(relative_velocity, tangent1) + surface_target[1] + split * coupling1 * normal_delta
    if use_bias:
        rhs0 += bias_t1
        rhs1 += bias_t2
    # Solve the Coulomb disk in the precomputed eigenframe of the tangent
    # inverse mass. The disk is rotation invariant and the diagonal metric
    # needs no float64 fallback for nearly rank-one blocks.
    tangent_rotated = _contact_project_friction_metric_float32(
        split * mobility0,
        wp.float32(0.0),
        split * mobility1,
        (cosine * rhs0 + sine * rhs1) * sor_boost,
        (cosine * rhs1 - sine * rhs0) * sor_boost,
        cosine * tangent_old[0] + sine * tangent_old[1],
        cosine * tangent_old[1] - sine * tangent_old[0],
        friction_static * normal_load,
        friction_dynamic * normal_load,
    )
    tangent_new = wp.vec2f(
        cosine * tangent_rotated[0] - sine * tangent_rotated[1],
        sine * tangent_rotated[0] + cosine * tangent_rotated[1],
    )
    cc_set_normal_lambda(contacts, contact, normal_load)
    cc_set_tangent1_lambda(contacts, contact, tangent_new[0])
    cc_set_tangent2_lambda(contacts, contact, tangent_new[1])
    tangent_delta = tangent_new - tangent_old
    return normal_impulse + tangent_delta[0] * tangent0 + tangent_delta[1] * tangent1


#: Most threads one contact-connected component uses: one per mechanism body pushes a
#: manifold's response in one round.
_COLUMN_LANES = 128

# Column record layout, one row per scheduled manifold.
_RECORD_BODY0 = 0
_RECORD_BODY1 = 1
_RECORD_TWIST0 = 2
_RECORD_TWIST1 = 3
_RECORD_SLOT0 = 4
_RECORD_SLOT1 = 5
_RECORD_FIRST = 6
_RECORD_COUNT = 7
_RECORD_BODY_BEGIN0 = 8
_RECORD_BODY_COUNT0 = 9
_RECORD_BODY_BEGIN1 = 10
_RECORD_BODY_COUNT1 = 11
# One past the last manifold of the record's body-pair run.
_RECORD_RUN_END = 12
_RECORD_WIDTH = 13
_record_t = wp.types.vector(length=_RECORD_WIDTH, dtype=wp.int32)


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    wp::spatial_vector result = value;
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int component = 0; component < 6; ++component) {
            result[component] += __shfl_xor_sync(0xffffffffu, result[component], offset);
        }
    }
    return result;
#else
    return value;
#endif
"""
)
def _warp_sum_spatial(value: wp.spatial_vector) -> wp.spatial_vector: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __shared__ float shared_wrench[12];
    float* entry = shared_wrench + 6 * index;
    if (mode == 1) {
        for (int component = 0; component < 6; ++component) entry[component] = value[component];
        return value;
    }
    wp::spatial_vector result;
    for (int component = 0; component < 6; ++component) result[component] = entry[component];
    return result;
#else
    return value;
#endif
"""
)
def _shared_wrench(index: wp.int32, value: wp.spatial_vector, mode: wp.int32) -> wp.spatial_vector:
    """Write (mode 1) or read one of a block's two run wrenches in shared memory."""
    ...


@wp.func
def _body_twist(bodies: BodyContainer, body: wp.int32) -> wp.spatial_vector:
    linear = bodies.velocity[body]
    angular = bodies.angular_velocity[body]
    return wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.func
def _same_run(
    columns: ContactColumnContainer, response: DirectContactResponseData, column: wp.int32, other: wp.int32
) -> wp.bool:
    """Whether two scheduled manifolds share their component and ordered body pair."""
    return (
        columns.articulation_owner[other] == columns.articulation_owner[column]
        and response.column_body0[other] == response.column_body0[column]
        and response.column_body1[other] == response.column_body1[column]
    )


@wp.kernel(enable_backward=False)
def _build_direct_contact_records_kernel(
    columns: ContactColumnContainer,
    response: DirectContactResponseData,
    scheduled_column: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
    records: wp.array[_record_t],
    record_friction: wp.array[wp.vec2],
):
    scheduled = wp.tid()
    total = section_end[section_end.shape[0] - 1]
    if scheduled >= total:
        return
    column = scheduled_column[scheduled]
    record = _record_t()
    run_end = scheduled + wp.int32(1)
    while run_end < total and _same_run(columns, response, column, scheduled_column[run_end]):
        run_end += wp.int32(1)
    record[_RECORD_RUN_END] = run_end
    record[_RECORD_FIRST] = contact_get_contact_first(columns, column)
    record[_RECORD_COUNT] = contact_get_contact_count(columns, column)
    record_friction[scheduled] = wp.vec2(
        contact_get_friction(columns, column), contact_get_friction_dynamic(columns, column)
    )
    for endpoint in range(2):
        body = response.column_body0[column]
        mechanism = response.column_mechanism0[column]
        if endpoint == 1:
            body = response.column_body1[column]
            mechanism = response.column_mechanism1[column]
        twist_index = wp.int32(-1)
        slot = wp.int32(-1)
        body_begin = wp.int32(0)
        body_count = wp.int32(0)
        if mechanism >= wp.int32(0):
            body_begin = response.mechanism_body_start[mechanism]
            body_count = response.mechanism_body_start[mechanism + wp.int32(1)] - body_begin
            twist_index = body_begin + response.body_lane[body]
            slot = _endpoint_basis_item(response, column, wp.int32(endpoint), body) // wp.int32(2)
        record[_RECORD_BODY0 + endpoint] = body
        record[_RECORD_TWIST0 + endpoint] = twist_index
        record[_RECORD_SLOT0 + endpoint] = slot
        record[_RECORD_BODY_BEGIN0 + 2 * endpoint] = body_begin
        record[_RECORD_BODY_COUNT0 + 2 * endpoint] = body_count
    records[scheduled] = record


@wp.kernel(enable_backward=False)
def _accumulate_direct_contact_row_impulses_kernel(
    response: DirectContactResponseData,
    basis_wrench: wp.array[wp.spatial_vector],
    slot_count: wp.int32,
):
    """Add the joint reactions of all pushed manifold wrenches to the equality impulses."""
    row = wp.tid()
    if row >= response.accumulated_impulse.shape[0]:
        return
    mechanism = response.row_mechanism[row]
    if mechanism < wp.int32(0):
        return
    local_row = row - response.mechanism_row_start[mechanism]
    slot_begin = wp.int32(0)
    slot_end = slot_count
    if response.body_basis_item.shape[0] > wp.int32(0):
        slot_begin = response.mechanism_body_start[mechanism]
        slot_end = response.mechanism_body_start[mechanism + wp.int32(1)]
    correction = wp.float32(0.0)
    for slot in range(slot_begin, slot_end):
        wrench = basis_wrench[slot]
        if wp.length_sq(wrench) > wp.float32(0.0):
            _, source_mechanism = _basis_slot_body(response, slot)
            if source_mechanism == mechanism:
                base = wp.int32(2) * slot * response.workspace_stride + local_row * wp.int32(GROUPED_RHS_ITEM_WIDTH)
                correction += response.solution[base] * wrench[0]
                correction += response.solution[base + wp.int32(1)] * wrench[1]
                correction += response.solution[base + wp.int32(2)] * wrench[2]
                base += response.workspace_stride
                correction += response.solution[base] * wrench[3]
                correction += response.solution[base + wp.int32(1)] * wrench[4]
                correction += response.solution[base + wp.int32(2)] * wrench[5]
    response.accumulated_impulse[row] -= response.row_scale[row] * correction


@wp.kernel(enable_backward=False)
def _build_direct_contact_endpoint_keys_kernel(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    section_end: wp.array[wp.int32],
    records: wp.array[_record_t],
    slot_capacity: wp.int32,
    keys: wp.array[wp.int64],
    entries: wp.array[wp.int32],
    entry_count: wp.array[wp.int32],
):
    """Key every manifold endpoint by the basis slot or free body it loads."""
    scheduled = wp.tid()
    total = section_end[section_end.shape[0] - 1]
    if scheduled == 0:
        entry_count[0] = wp.int32(2) * total
    if scheduled >= total:
        return
    record = records[scheduled]
    for endpoint in range(2):
        entry = wp.int32(2) * scheduled + endpoint
        entries[entry] = entry
        target = wp.int64(-1)
        if record[_RECORD_SLOT0 + endpoint] >= wp.int32(0):
            target = wp.int64(record[_RECORD_SLOT0 + endpoint])
        else:
            body = record[_RECORD_BODY0 + endpoint]
            if (
                body > wp.int32(0)
                and bodies.inverse_mass[body] > wp.float32(0.0)
                and bodies.motion_type[body] != MOTION_KINEMATIC
            ):
                target = wp.int64(slot_capacity) + wp.int64(body)
        if target < wp.int64(0):
            keys[entry] = wp.int64(_INT64_MAX)
        else:
            keys[entry] = target


@wp.kernel(enable_backward=False)
def _warm_start_direct_contact_wrench_kernel(
    response: DirectContactResponseData,
    contacts: ContactContainer,
    section_end: wp.array[wp.int32],
    records: wp.array[_record_t],
    endpoint_wrench: wp.array[wp.spatial_vector],
    threads: wp.int32,
):
    """Cached manifold impulses as endpoint wrenches, ``threads`` lanes striding over the schedule."""
    for scheduled in range(wp.tid(), section_end[section_end.shape[0] - 1], threads):
        record = records[scheduled]
        first = record[_RECORD_FIRST]
        wrench0 = wp.spatial_vectorf(0.0)
        wrench1 = wp.spatial_vectorf(0.0)
        for offset in range(record[_RECORD_COUNT]):
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
            wrench0 += _make_spatial_wrench(-impulse, -wp.cross(r0, impulse))
            wrench1 += _make_spatial_wrench(impulse, wp.cross(r1, impulse))
        endpoint_wrench[wp.int32(2) * scheduled] = wrench0
        endpoint_wrench[wp.int32(2) * scheduled + wp.int32(1)] = wrench1


@wp.kernel(enable_backward=False)
def _warm_start_direct_contact_targets_kernel(
    bodies: BodyContainer,
    slot_capacity: wp.int32,
    keys: wp.array[wp.int64],
    entries: wp.array[wp.int32],
    entry_count: wp.array[wp.int32],
    endpoint_wrench: wp.array[wp.spatial_vector],
    basis_wrench: wp.array[wp.spatial_vector],
    threads: wp.int32,
):
    """Sum each slot's or free body's endpoint wrenches in schedule order."""
    count = entry_count[0]
    for index in range(wp.tid(), count, threads):
        target = keys[index]
        if target != wp.int64(_INT64_MAX) and (index == 0 or keys[index - 1] != target):
            wrench = wp.spatial_vectorf(0.0)
            cursor = index
            while cursor < count and keys[cursor] == target:
                wrench += endpoint_wrench[entries[cursor]]
                cursor += 1
            if target < wp.int64(slot_capacity):
                basis_wrench[wp.int32(target)] = wrench
            else:
                _apply_raw_contact_wrench(bodies, wp.int32(target - wp.int64(slot_capacity)), wrench)


@wp.kernel(enable_backward=False)
def _warm_start_direct_contact_push_kernel(
    response: DirectContactResponseData,
    bodies: BodyContainer,
    mechanism_component: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
    records: wp.array[_record_t],
    basis_wrench: wp.array[wp.spatial_vector],
):
    """Apply all warm-start wrenches through the exact mechanism responses."""
    flat = wp.tid()
    if flat >= response.mechanism_body.shape[0]:
        return
    body = response.mechanism_body[flat]
    mechanism = response.body_mechanism[body]
    body_begin = response.mechanism_body_start[mechanism]
    local_body = flat - body_begin
    twist = wp.spatial_vectorf(0.0)
    if response.body_basis_item.shape[0] > wp.int32(0):
        for slot in range(body_begin, response.mechanism_body_start[mechanism + wp.int32(1)]):
            wrench = basis_wrench[slot]
            if wp.length_sq(wrench) > wp.float32(0.0):
                twist += _push_response(response, slot, local_body, wrench)
    else:
        component = mechanism_component[mechanism]
        begin = wp.int32(0)
        if component > wp.int32(0):
            begin = section_end[component - wp.int32(1)]
        for scheduled in range(begin, section_end[component]):
            record = records[scheduled]
            for endpoint in range(2):
                slot = record[_RECORD_SLOT0 + endpoint]
                if slot >= wp.int32(0) and record[_RECORD_BODY_BEGIN0 + 2 * endpoint] == body_begin:
                    twist += _push_response(response, slot, local_body, basis_wrench[slot])
    if wp.length_sq(twist) > wp.float32(0.0):
        bodies.velocity[body] += wp.spatial_top(twist)
        bodies.angular_velocity[body] += wp.spatial_bottom(twist)


# Shared twist storage per sweep block; larger mechanism sets update the body
# velocities in place instead.
_SHARED_TWIST_BYTES = 40 * 1024


def _make_twist_state_access(length: int):
    """Return an accessor for one block's mechanism twists in shared memory.

    Warp tiles synchronize the block after every element write. Each twist of
    the ordered sweep is written by exactly one lane and the phases are
    ordered with ``__syncthreads``, so raw shared storage suffices.
    """

    snippet = f"""
#if defined(__CUDA_ARCH__)
    __shared__ float twist_state[{6 * int(length)}];
    float* entry = twist_state + 6 * index;
    if (mode == 1) {{
        for (int component = 0; component < 6; ++component) entry[component] = value[component];
    }} else if (mode == 2) {{
        for (int component = 0; component < 6; ++component) entry[component] += value[component];
    }}
    wp::spatial_vector result;
    for (int component = 0; component < 6; ++component) result[component] = entry[component];
    return result;
#else
    return value;
#endif
"""

    @wp.func_native(snippet)
    def twist_state(index: wp.int32, value: wp.spatial_vector, mode: wp.int32) -> wp.spatial_vector: ...

    return twist_state


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
"""
)
def _sync_block(): ...


_TWIST_READ = 0
_TWIST_WRITE = 1
_TWIST_ADD = 2


def _component_lanes(max_mechanism_bodies: int, is_cuda: bool) -> int:
    """Threads per component: whole warps, one per mechanism body up to :data:`_COLUMN_LANES`."""
    if not is_cuda:
        return 1
    return min(_COLUMN_LANES, 32 * max(1, (int(max_mechanism_bodies) + 31) // 32))


def make_solve_direct_contact_columns_kernel(body_capacity: int, max_mechanism_bodies: int, *, is_cuda: bool = True):
    """Build the ordered manifold sweep for a response's body capacity."""
    capacity = max(1, int(body_capacity))
    # CPU native warp operations are scalar: visit every point and endpoint
    # serially, using body arrays instead of CUDA-only shared twist storage.
    lanes = _component_lanes(max_mechanism_bodies, is_cuda)
    relax_lanes = min(lanes, 32)
    rounds = max(1, (int(max_mechanism_bodies) + lanes - 1) // lanes)
    shared = is_cuda and 24 * capacity <= _SHARED_TWIST_BYTES
    twist_state = _make_twist_state_access(capacity if shared else 1)

    @wp.func
    def read_twist(bodies: BodyContainer, body: wp.int32, index: wp.int32) -> wp.spatial_vector:
        if wp.static(shared):
            if index >= wp.int32(0):
                return twist_state(index, wp.spatial_vectorf(0.0), _TWIST_READ)
        return _body_twist(bodies, body)

    @wp.func
    def write_twist(
        response: DirectContactResponseData, bodies: BodyContainer, index: wp.int32, twist: wp.spatial_vector
    ):
        if wp.static(shared):
            twist_state(index, twist, _TWIST_WRITE)
        else:
            body = response.mechanism_body[index]
            bodies.velocity[body] = wp.spatial_top(twist)
            bodies.angular_velocity[body] = wp.spatial_bottom(twist)

    @wp.func
    def add_twist(
        response: DirectContactResponseData, bodies: BodyContainer, index: wp.int32, twist: wp.spatial_vector
    ):
        if wp.static(shared):
            twist_state(index, twist, _TWIST_ADD)
        else:
            body = response.mechanism_body[index]
            bodies.velocity[body] += wp.spatial_top(twist)
            bodies.angular_velocity[body] += wp.spatial_bottom(twist)

    @wp.kernel(enable_backward=False, module="unique")
    def solve_direct_contact_columns_kernel(
        component_mechanism: wp.array[wp.int32],
        component_end: wp.array[wp.int32],
        num_components: wp.array[wp.int32],
        response: DirectContactResponseData,
        bodies: BodyContainer,
        contacts: ContactContainer,
        inverse_dt: wp.float32,
        sor_boost: wp.float32,
        section_end: wp.array[wp.int32],
        records: wp.array[_record_t],
        record_friction: wp.array[wp.vec2],
        basis_wrench: wp.array[wp.spatial_vector],
        iteration_count: wp.int32,
        use_bias: wp.bool,
    ):
        """Sweep manifolds in order through the exact mechanism response.

        One block owns a contact-connected component and visits its manifolds
        in schedule order, so later manifolds see every earlier update. The
        points of one manifold act on the same two bodies; each relaxes
        against its share of the manifold's exact mobility, which keeps the
        Gauss-Seidel fixed point.

        Manifolds of one body pair form a run. The first warp relaxes a run's
        manifolds in turn and updates the pair's twists through the pair's own
        response blocks. Only then does the whole block push the run's
        wrenches to the other mechanism bodies, which no manifold of the run
        reads.
        """
        component, lane = wp.tid()
        if component >= num_components[0]:
            return
        mechanism_begin = wp.int32(0)
        section_begin = wp.int32(0)
        if component > wp.int32(0):
            mechanism_begin = component_end[component - wp.int32(1)]
            section_begin = section_end[component - wp.int32(1)]
        mechanism_end = component_end[component]
        section_stop = section_end[component]

        if wp.static(shared):
            for item in range(mechanism_begin, mechanism_end):
                mechanism = component_mechanism[item]
                body_end = response.mechanism_body_start[mechanism + wp.int32(1)]
                for index in range(response.mechanism_body_start[mechanism] + lane, body_end, wp.int32(lanes)):
                    twist_state(index, _body_twist(bodies, response.mechanism_body[index]), _TWIST_WRITE)
            _sync_block()

        dt = wp.float32(1.0) / inverse_dt
        _, mass_coeff, impulse_coeff = soft_constraint_coefficients(DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt)
        for _iteration in range(iteration_count):
            run_begin = section_begin
            while run_begin < section_stop:
                record = records[run_begin]
                run_end = record[_RECORD_RUN_END]
                body0 = record[_RECORD_BODY0]
                body1 = record[_RECORD_BODY1]
                twist0 = record[_RECORD_TWIST0]
                twist1 = record[_RECORD_TWIST1]
                slot0 = record[_RECORD_SLOT0]
                slot1 = record[_RECORD_SLOT1]
                body_begin0 = record[_RECORD_BODY_BEGIN0]
                body_count0 = record[_RECORD_BODY_COUNT0]
                body_begin1 = record[_RECORD_BODY_BEGIN1]
                body_count1 = record[_RECORD_BODY_COUNT1]
                same_mechanism = body_begin1 == body_begin0 and slot0 >= wp.int32(0) and slot1 >= wp.int32(0)
                total0 = wp.spatial_vectorf(0.0)
                total1 = wp.spatial_vectorf(0.0)
                if lane < wp.int32(relax_lanes):
                    velocity_body0 = read_twist(bodies, body0, twist0)
                    velocity_body1 = read_twist(bodies, body1, twist1)
                    next_record = record
                    for scheduled in range(run_begin, run_end):
                        # Load the following manifold's record while this one runs.
                        manifold = next_record
                        if scheduled + wp.int32(1) < run_end:
                            next_record = records[scheduled + wp.int32(1)]
                        first = manifold[_RECORD_FIRST]
                        count = manifold[_RECORD_COUNT]
                        friction = record_friction[scheduled]
                        # Tonge mass splitting inside the manifold: every point sees the
                        # pair with ``count`` times its exact mobility.
                        split = wp.float32(wp.max(count, wp.int32(1)))
                        wrench0 = wp.spatial_vectorf(0.0)
                        wrench1 = wp.spatial_vectorf(0.0)
                        for offset in range(lane, count, wp.int32(relax_lanes)):
                            contact = first + offset
                            impulse = _relax_direct_contact_point(
                                response,
                                contacts,
                                contact,
                                velocity_body0,
                                velocity_body1,
                                use_bias,
                                mass_coeff,
                                impulse_coeff,
                                sor_boost,
                                friction[0],
                                friction[1],
                                split,
                            )
                            r0 = cc_get_r0(contacts, contact)
                            r1 = cc_get_r1(contacts, contact)
                            wrench0 += _make_spatial_wrench(-impulse, -wp.cross(r0, impulse))
                            wrench1 += _make_spatial_wrench(impulse, wp.cross(r1, impulse))
                        wrench0 = _warp_sum_spatial(wrench0)
                        wrench1 = _warp_sum_spatial(wrench1)
                        total0 += wrench0
                        total1 += wrench1
                        velocity_body0 += _pair_twist_change(
                            response,
                            bodies,
                            body0,
                            slot0,
                            twist0 - body_begin0,
                            wrench0,
                            slot1,
                            wrench1,
                            same_mechanism,
                        )
                        velocity_body1 += _pair_twist_change(
                            response,
                            bodies,
                            body1,
                            slot1,
                            twist1 - body_begin1,
                            wrench1,
                            slot0,
                            wrench0,
                            same_mechanism,
                        )
                    if lane == wp.int32(0):
                        _shared_wrench(0, total0, _TWIST_WRITE)
                        if slot0 >= wp.int32(0):
                            basis_wrench[slot0] += total0
                            write_twist(response, bodies, twist0, velocity_body0)
                        else:
                            _apply_raw_contact_wrench(bodies, body0, total0)
                    if lane == wp.int32(1) or wp.static(not is_cuda):
                        _shared_wrench(1, total1, _TWIST_WRITE)
                        if slot1 >= wp.int32(0):
                            basis_wrench[slot1] += total1
                            write_twist(response, bodies, twist1, velocity_body1)
                        else:
                            _apply_raw_contact_wrench(bodies, body1, total1)
                _sync_block()
                total0 = _shared_wrench(0, total0, _TWIST_READ)
                total1 = _shared_wrench(1, total1, _TWIST_READ)
                # Push the run's wrenches to the mechanisms' other bodies. Every
                # lane owns distinct twists, so all rounds' response loads can
                # be in flight before the twist updates.
                for round_index in range(wp.static(rounds)):
                    local_body = round_index * lanes + lane
                    target0 = body_begin0 + local_body
                    target1 = body_begin1 + local_body
                    value0 = wp.spatial_vectorf(0.0)
                    if slot0 >= wp.int32(0) and local_body < body_count0:
                        value0 = _push_response(response, slot0, local_body, total0)
                    if slot1 >= wp.int32(0) and local_body < body_count1:
                        value1 = _push_response(response, slot1, local_body, total1)
                        if same_mechanism:
                            value0 += value1
                        elif target1 != twist1:
                            add_twist(response, bodies, target1, value1)
                    if slot0 >= wp.int32(0) and local_body < body_count0 and target0 != twist0 and target0 != twist1:
                        add_twist(response, bodies, target0, value0)
                _sync_block()
                run_begin = run_end

        if wp.static(shared):
            for item in range(mechanism_begin, mechanism_end):
                mechanism = component_mechanism[item]
                body_end = response.mechanism_body_start[mechanism + wp.int32(1)]
                for index in range(response.mechanism_body_start[mechanism] + lane, body_end, wp.int32(lanes)):
                    body = response.mechanism_body[index]
                    twist = twist_state(index, wp.spatial_vectorf(0.0), _TWIST_READ)
                    bodies.velocity[body] = wp.spatial_top(twist)
                    bodies.angular_velocity[body] = wp.spatial_bottom(twist)

    return solve_direct_contact_columns_kernel


class DirectContactRunSchedule:
    """Group direct mechanisms into contact-connected solve components."""

    def __init__(self, response: DirectContactResponse, column_capacity: int):
        self.response = response
        self.capacity = max(1, int(column_capacity))
        device = response.direct.model.device
        self.keys = wp.empty(2 * self.capacity, dtype=wp.int64, device=device)
        self.columns = wp.empty(2 * self.capacity, dtype=wp.int32, device=device)
        mechanism_count = len(response.active_mechanisms)
        self.section_end = wp.zeros(mechanism_count, dtype=wp.int32, device=device)
        self.interactions = wp.full((self.capacity, 8), -1, dtype=wp.int32, device=device)
        self.num_mechanisms = wp.array([mechanism_count], dtype=wp.int32, device=device)
        self.islands = UnionFindIslandBuilder(mechanism_count, device=device)
        self.basis_wrench = wp.zeros(max(1, response.basis_slot_capacity), dtype=wp.spatial_vector, device=device)
        self.records = wp.zeros(self.capacity, dtype=_record_t, device=device)
        self.endpoint_keys = wp.empty(4 * self.capacity, dtype=wp.int64, device=device)
        self.endpoint_entries = wp.empty(4 * self.capacity, dtype=wp.int32, device=device)
        self.endpoint_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.endpoint_wrench = wp.zeros(2 * self.capacity, dtype=wp.spatial_vector, device=device)
        self.record_friction = wp.zeros(self.capacity, dtype=wp.vec2, device=device)
        self.solve_kernel = make_solve_direct_contact_columns_kernel(
            response.body_capacity, response.max_mechanism_bodies, is_cuda=device.is_cuda
        )
        self._lanes = _component_lanes(response.max_mechanism_bodies, device.is_cuda)
        #: Threads of the warm-start passes, which stride over the scheduled manifolds.
        #: Few are in use; a launch over the whole capacity costs more than their work.
        self.warm_start_threads = min(self.capacity, 1 << 15)
        # Sort keys hold the component and, when they fit, the ordered body pair.
        self._component_bits = self.section_end.size.bit_length()
        self._body_bits = max(1, (response.direct.bodies.position.shape[0] - 1).bit_length())
        if self._component_bits + 2 * self._body_bits > 63:
            self._body_bits = 0
        # The schedule after the candidate pass runs on a private stream.
        self._stream = wp.Stream(device) if device.is_cuda else None
        self._fork_event = wp.Event(device) if device.is_cuda else None
        self._join_event = wp.Event(device) if device.is_cuda else None

    def warm_start(self, bodies: BodyContainer, contacts: ContactContainer) -> None:
        """Apply cached contact impulses through the exact mechanism responses.

        The warm start is linear, so manifolds need no ordering: wrenches are
        formed in parallel and summed per basis slot in schedule order.
        """
        device = self.response.direct.model.device
        response = self.response
        self.basis_wrench.zero_()
        wp.launch(
            _warm_start_direct_contact_wrench_kernel,
            dim=self.warm_start_threads,
            inputs=[
                response.data,
                contacts,
                self.section_end,
                self.records,
                self.endpoint_wrench,
                wp.int32(self.warm_start_threads),
            ],
            device=device,
        )
        wp.launch(
            _warm_start_direct_contact_targets_kernel,
            dim=self.warm_start_threads,
            inputs=[
                bodies,
                wp.int32(self.basis_wrench.shape[0]),
                self.endpoint_keys,
                self.endpoint_entries,
                self.endpoint_count,
                self.endpoint_wrench,
                self.basis_wrench,
                wp.int32(self.warm_start_threads),
            ],
            device=device,
        )
        wp.launch(
            _warm_start_direct_contact_push_kernel,
            dim=max(1, response.body_capacity),
            inputs=[response.data, bodies, self.islands.set_nr, self.section_end, self.records, self.basis_wrench],
            device=device,
        )
        self._accumulate_row_impulses()

    def _accumulate_row_impulses(self) -> None:
        wp.launch(
            _accumulate_direct_contact_row_impulses_kernel,
            dim=self.response.data.accumulated_impulse.shape[0],
            inputs=[self.response.data, self.basis_wrench, wp.int32(self.basis_wrench.shape[0])],
            device=self.response.direct.model.device,
        )

    def solve(
        self,
        bodies: BodyContainer,
        contacts: ContactContainer,
        inverse_dt: float,
        sor_boost: float,
        *,
        iterations: int,
        use_bias: bool,
    ) -> None:
        """Run ordered manifold sweeps for every contact-connected component."""
        device = self.response.direct.model.device
        self.basis_wrench.zero_()
        wp.launch_tiled(
            self.solve_kernel,
            dim=self.section_end.size,
            block_dim=self._lanes,
            inputs=[
                self.islands.set_sizes,
                self.islands.set_sizes_compact,
                self.islands.num_sets,
                self.response.data,
                bodies,
                contacts,
                wp.float32(inverse_dt),
                wp.float32(sor_boost),
                self.section_end,
                self.records,
                self.record_friction,
                self.basis_wrench,
                wp.int32(iterations),
                wp.bool(use_bias),
            ],
            device=device,
        )
        self._accumulate_row_impulses()

    def build(
        self,
        columns: ContactColumnContainer,
        num_columns: wp.array[wp.int32],
        *,
        reset_owner: bool,
    ) -> None:
        """Group immutable contacts without changing another response owner's columns.

        On CUDA everything after the candidate pass runs on a private stream.
        The caller must call :meth:`join` before the schedule is read, before a
        graph conditional, and before any change to the contact columns, the
        response topology or the bodies' inverse mass or motion type.
        """
        self.section_end.zero_()
        self.response.contact_mechanism.fill_(-1)
        self.response.contact_column.fill_(-1)
        self.response.column_mechanism0.fill_(-1)
        self.response.column_mechanism1.fill_(-1)
        self.response.reset_rhs_tasks()
        self.interactions.fill_(-1)
        wp.launch(
            _build_direct_contact_candidates_kernel,
            dim=self.capacity,
            inputs=[
                columns,
                self.response.data,
                num_columns,
                self.interactions,
                wp.bool(reset_owner),
            ],
            device=self.response.direct.model.device,
        )
        if self._stream is None:
            self._build_schedule(columns, num_columns)
            return
        wp.get_stream(self._stream.device).record_event(self._fork_event)
        self._stream.wait_event(self._fork_event)
        with wp.ScopedStream(self._stream, sync_enter=False, sync_exit=False):
            self._build_schedule(columns, num_columns)
        self._stream.record_event(self._join_event)

    def join(self) -> None:
        """Make the current stream wait for the schedule issued by :meth:`build`."""
        if self._stream is not None:
            wp.get_stream(self._stream.device).wait_event(self._join_event)

    def _build_schedule(self, columns: ContactColumnContainer, num_columns: wp.array[wp.int32]) -> None:
        self.islands.build_islands(
            self.interactions,
            num_columns,
            self.num_mechanisms,
        )
        wp.launch(
            _finalize_direct_contact_schedule_kernel,
            dim=self.capacity,
            inputs=[
                columns,
                self.response.data,
                num_columns,
                self.islands.set_nr,
                wp.int32(self._body_bits),
                self.keys,
                self.columns,
                self.section_end,
            ],
            device=self.response.direct.model.device,
        )
        sort_variable_length_int64(
            self.keys,
            self.columns,
            num_columns,
            end_bit=self._component_bits + 2 * self._body_bits,
            tail_masked=True,
        )
        wp.utils.array_scan(self.section_end, self.section_end, inclusive=True)
        wp.launch(
            _build_direct_contact_records_kernel,
            dim=self.capacity,
            inputs=[columns, self.response.data, self.columns, self.section_end, self.records, self.record_friction],
            device=self.response.direct.model.device,
        )
        wp.launch(
            _build_direct_contact_endpoint_keys_kernel,
            dim=self.capacity,
            inputs=[
                self.response.data,
                self.response.direct.bodies,
                self.section_end,
                self.records,
                wp.int32(self.basis_wrench.shape[0]),
                self.endpoint_keys,
                self.endpoint_entries,
                self.endpoint_count,
            ],
            device=self.response.direct.model.device,
        )
        sort_variable_length_int64(
            self.endpoint_keys,
            self.endpoint_entries,
            self.endpoint_count,
            end_bit=(self.basis_wrench.shape[0] + self.response.direct.bodies.position.shape[0]).bit_length(),
        )
        wp.launch(
            _fill_direct_contact_rhs_tasks_kernel,
            dim=self.capacity,
            inputs=[
                self.response.data,
                self.response.contact_batch.task_mechanism,
                self.response.contact_batch.task_item,
            ],
            device=self.response.direct.model.device,
        )


__all__ = ["DirectContactRunSchedule", "make_solve_direct_contact_columns_kernel"]
