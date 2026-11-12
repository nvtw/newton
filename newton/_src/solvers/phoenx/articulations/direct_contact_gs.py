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
)
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import (
    GROUPED_RHS_ITEM_WIDTH,
    GROUPED_RHS_ITEMS_PER_TASK,
)
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
    cc_get_surface_target,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_projection import (
    _contact_project_friction_metric_float32,
    contact_project_normal_velocity_update,
)
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
    key_stride: wp.int64,
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
    keys[column] = wp.int64(component) * key_stride + wp.int64(column)
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
def _apply_raw_contact_wrench(bodies: BodyContainer, body: wp.int32, wrench: wp.spatial_vector):
    if (
        body <= wp.int32(0)
        or bodies.inverse_mass[body] <= wp.float32(0.0)
        or bodies.motion_type[body] == MOTION_KINEMATIC
    ):
        return
    bodies.velocity[body] += bodies.inverse_mass[body] * wp.spatial_top(wrench)
    bodies.angular_velocity[body] += mat33_from_sym6(bodies.inverse_inertia_world[body]) * wp.spatial_bottom(wrench)


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
    normal_mobility = response.mobility[0, contact]
    bias = cc_get_bias(contacts, contact)
    speculative = bias > wp.float32(0.0)
    if not use_bias:
        bias = wp.float32(0.0)
    elif bias < wp.float32(0.0) and normal_mobility > wp.float32(1.0e-12):
        # Project the unconstrained recovery impulse through the joints.
        # Enforcing the full recovery speed along a nearly blocked normal
        # turns small overlaps into large joint motions. Physical velocity
        # impulses below still use the constrained effective mass.
        bias *= wp.min(wp.float32(1.0), cc_get_eff_n(contacts, contact) / normal_mobility)
    if not use_bias and speculative:
        return wp.vec3(0.0)
    row_mass_coeff = mass_coeff
    row_impulse_coeff = impulse_coeff / split
    if speculative or not use_bias:
        row_mass_coeff = wp.float32(1.0)
        row_impulse_coeff = wp.float32(0.0)
    pd_eff = cc_get_pd_eff_soft(contacts, contact)
    pd_gamma = cc_get_pd_gamma(contacts, contact)
    pd_bias = cc_get_pd_bias(contacts, contact)
    if pd_eff > wp.float32(0.0):
        pd_eff = wp.float32(0.0)
        if normal_mobility > wp.float32(1.0e-12):
            pd_eff = wp.float32(1.0) / ((wp.float32(1.0) / normal_mobility + pd_gamma) * split)
    normal_impulse = -cc_get_normal_lambda(contacts, contact) * normal
    if normal_mobility > wp.float32(1.0e-12):
        normal_impulse = contact_project_normal_velocity_update(
            contacts,
            contact,
            normal,
            wp.dot(relative_velocity, normal),
            normal_mobility / split,
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
    normal_load = cc_get_normal_lambda(contacts, contact)
    surface_target = cc_get_surface_target(contacts, contact)
    rhs0 = (
        wp.dot(relative_velocity, tangent0) + surface_target[0] + split * response.mobility[3, contact] * normal_delta
    )
    rhs1 = (
        wp.dot(relative_velocity, tangent1) + surface_target[1] + split * response.mobility[4, contact] * normal_delta
    )
    if use_bias:
        rhs0 += cc_get_bias_t1(contacts, contact)
        rhs1 += cc_get_bias_t2(contacts, contact)
    # Solve the Coulomb disk in the precomputed eigenframe of the tangent
    # inverse mass. The disk is rotation invariant and the diagonal metric
    # needs no float64 fallback for nearly rank-one blocks.
    cosine = response.mobility[6, contact]
    sine = response.mobility[7, contact]
    tangent_old = wp.vec2f(cc_get_tangent1_lambda(contacts, contact), cc_get_tangent2_lambda(contacts, contact))
    tangent_rotated = _contact_project_friction_metric_float32(
        split * response.mobility[8, contact],
        wp.float32(0.0),
        split * response.mobility[9, contact],
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
    cc_set_tangent1_lambda(contacts, contact, tangent_new[0])
    cc_set_tangent2_lambda(contacts, contact, tangent_new[1])
    tangent_delta = tangent_new - tangent_old
    return normal_impulse + tangent_delta[0] * tangent0 + tangent_delta[1] * tangent1


_COLUMN_LANES = 32

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
_RECORD_WIDTH = 12
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


@wp.func
def _body_twist(bodies: BodyContainer, body: wp.int32) -> wp.spatial_vector:
    linear = bodies.velocity[body]
    angular = bodies.angular_velocity[body]
    return wp.spatial_vectorf(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.func
def _push_response(
    response: DirectContactResponseData,
    slot: wp.int32,
    local_body: wp.int32,
    wrench: wp.spatial_vector,
) -> wp.spatial_vector:
    twist = wp.spatial_vectorf(0.0)
    for component in range(6):
        twist += wrench[component] * response.body_response[slot, local_body, component]
    return twist


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
    if scheduled >= section_end[section_end.shape[0] - 1]:
        return
    column = scheduled_column[scheduled]
    record = _record_t()
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
    key_stride: wp.int64,
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
            keys[entry] = target * key_stride + wp.int64(entry)


@wp.kernel(enable_backward=False)
def _warm_start_direct_contact_wrench_kernel(
    response: DirectContactResponseData,
    contacts: ContactContainer,
    section_end: wp.array[wp.int32],
    records: wp.array[_record_t],
    endpoint_wrench: wp.array[wp.spatial_vector],
):
    """Cached manifold impulses as endpoint wrenches."""
    scheduled = wp.tid()
    if scheduled >= section_end[section_end.shape[0] - 1]:
        return
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
    key_stride: wp.int64,
    keys: wp.array[wp.int64],
    entries: wp.array[wp.int32],
    entry_count: wp.array[wp.int32],
    endpoint_wrench: wp.array[wp.spatial_vector],
    basis_wrench: wp.array[wp.spatial_vector],
):
    """Sum each slot's or free body's endpoint wrenches in schedule order."""
    index = wp.tid()
    count = entry_count[0]
    if index >= count or keys[index] == wp.int64(_INT64_MAX):
        return
    target = keys[index] // key_stride
    if index > 0 and keys[index - 1] // key_stride == target:
        return
    wrench = wp.spatial_vectorf(0.0)
    cursor = index
    while cursor < count and keys[cursor] != wp.int64(_INT64_MAX) and keys[cursor] // key_stride == target:
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
    """Return an accessor for one warp's mechanism twists in shared memory.

    Warp tiles synchronize the block after every element write. Each twist of
    the ordered sweep is written by exactly one lane and the phases are
    ordered with ``__syncwarp``, so raw shared storage suffices.
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
    __syncwarp();
#endif
"""
)
def _sync_warp(): ...


_TWIST_READ = 0
_TWIST_WRITE = 1
_TWIST_ADD = 2


def make_solve_direct_contact_columns_kernel(body_capacity: int, max_mechanism_bodies: int):
    """Build the ordered manifold sweep for a response's body capacity."""
    capacity = max(1, int(body_capacity))
    rounds = max(1, (int(max_mechanism_bodies) + _COLUMN_LANES - 1) // _COLUMN_LANES)
    shared = 24 * capacity <= _SHARED_TWIST_BYTES
    twist_state = _make_twist_state_access(capacity if shared else 1)

    @wp.func
    def read_twist(bodies: BodyContainer, body: wp.int32, index: wp.int32) -> wp.spatial_vector:
        if wp.static(shared):
            if index >= wp.int32(0):
                return twist_state(index, wp.spatial_vectorf(0.0), _TWIST_READ)
        return _body_twist(bodies, body)

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

        One warp owns a contact-connected component. Manifolds are visited in
        schedule order and each manifold's wrench is pushed through the exact
        constrained body responses, so later manifolds see every earlier
        update. The points of one manifold act on the same two bodies; each
        relaxes against its share of the manifold's exact mobility, which
        keeps the Gauss-Seidel fixed point.
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
                for index in range(response.mechanism_body_start[mechanism] + lane, body_end, wp.int32(_COLUMN_LANES)):
                    twist_state(index, _body_twist(bodies, response.mechanism_body[index]), _TWIST_WRITE)
            _sync_warp()

        dt = wp.float32(1.0) / inverse_dt
        _, mass_coeff, impulse_coeff = soft_constraint_coefficients(DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt)
        for _iteration in range(iteration_count):
            next_record = _record_t()
            if section_begin < section_stop:
                next_record = records[section_begin]
            for scheduled in range(section_begin, section_stop):
                # Load the following manifold's record while this one runs.
                record = next_record
                if scheduled + wp.int32(1) < section_stop:
                    next_record = records[scheduled + wp.int32(1)]
                body0 = record[_RECORD_BODY0]
                body1 = record[_RECORD_BODY1]
                slot0 = record[_RECORD_SLOT0]
                slot1 = record[_RECORD_SLOT1]
                first = record[_RECORD_FIRST]
                count = record[_RECORD_COUNT]
                body_begin0 = record[_RECORD_BODY_BEGIN0]
                body_count0 = record[_RECORD_BODY_COUNT0]
                body_begin1 = record[_RECORD_BODY_BEGIN1]
                body_count1 = record[_RECORD_BODY_COUNT1]
                friction = record_friction[scheduled]
                velocity_body0 = read_twist(bodies, body0, record[_RECORD_TWIST0])
                velocity_body1 = read_twist(bodies, body1, record[_RECORD_TWIST1])
                # Tonge mass splitting inside the manifold: every point sees the
                # pair with ``count`` times its exact mobility.
                split = wp.float32(wp.max(count, wp.int32(1)))
                wrench0 = wp.spatial_vectorf(0.0)
                wrench1 = wp.spatial_vectorf(0.0)
                for offset in range(lane, count, wp.int32(_COLUMN_LANES)):
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
                if lane == wp.int32(0):
                    if slot0 >= wp.int32(0):
                        basis_wrench[slot0] += wrench0
                    else:
                        _apply_raw_contact_wrench(bodies, body0, wrench0)
                elif lane == wp.int32(1):
                    if slot1 >= wp.int32(0):
                        basis_wrench[slot1] += wrench1
                    else:
                        _apply_raw_contact_wrench(bodies, body1, wrench1)
                # Push the manifold wrench through each touched mechanism. Every
                # lane owns distinct twists, so all rounds' response loads can
                # be in flight before the twist updates.
                same_mechanism = body_begin1 == body_begin0 and slot0 >= wp.int32(0) and slot1 >= wp.int32(0)
                for round_index in range(wp.static(rounds)):
                    local_body = round_index * _COLUMN_LANES + lane
                    value0 = wp.spatial_vectorf(0.0)
                    if slot0 >= wp.int32(0) and local_body < body_count0:
                        value0 = _push_response(response, slot0, local_body, wrench0)
                    if slot1 >= wp.int32(0) and local_body < body_count1:
                        value1 = _push_response(response, slot1, local_body, wrench1)
                        if same_mechanism:
                            value0 += value1
                        else:
                            add_twist(response, bodies, body_begin1 + local_body, value1)
                    if slot0 >= wp.int32(0) and local_body < body_count0:
                        add_twist(response, bodies, body_begin0 + local_body, value0)
                _sync_warp()

        if wp.static(shared):
            for item in range(mechanism_begin, mechanism_end):
                mechanism = component_mechanism[item]
                body_end = response.mechanism_body_start[mechanism + wp.int32(1)]
                for index in range(response.mechanism_body_start[mechanism] + lane, body_end, wp.int32(_COLUMN_LANES)):
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
        self.endpoint_key_stride = 2 * self.capacity + 1
        self.record_friction = wp.zeros(self.capacity, dtype=wp.vec2, device=device)
        self.solve_kernel = make_solve_direct_contact_columns_kernel(
            response.body_capacity, response.max_mechanism_bodies
        )

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
            dim=self.capacity,
            inputs=[response.data, contacts, self.section_end, self.records, self.endpoint_wrench],
            device=device,
        )
        wp.launch(
            _warm_start_direct_contact_targets_kernel,
            dim=2 * self.capacity,
            inputs=[
                bodies,
                wp.int32(self.basis_wrench.shape[0]),
                wp.int64(self.endpoint_key_stride),
                self.endpoint_keys,
                self.endpoint_entries,
                self.endpoint_count,
                self.endpoint_wrench,
                self.basis_wrench,
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
            block_dim=_COLUMN_LANES,
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
        """Group immutable contacts without changing another response owner's columns."""
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
                wp.int64(self.capacity + 1),
                self.keys,
                self.columns,
                self.section_end,
            ],
            device=self.response.direct.model.device,
        )
        sort_variable_length_int64(self.keys, self.columns, num_columns)
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
                wp.int64(self.endpoint_key_stride),
                self.endpoint_keys,
                self.endpoint_entries,
                self.endpoint_count,
            ],
            device=self.response.direct.model.device,
        )
        sort_variable_length_int64(self.endpoint_keys, self.endpoint_entries, self.endpoint_count)
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
