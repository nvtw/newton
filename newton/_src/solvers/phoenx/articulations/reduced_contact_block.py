# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-cooperative contact-space blocks for reduced articulations."""

from __future__ import annotations

import warp as wp

from newton._src.sim import Model
from newton._src.solvers.phoenx.articulations.reduced_contact import (
    _deferred_point_velocity,
    reduced_contact_deferred_owner,
    reduced_contact_iterate,
    reduced_contact_prepare,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
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
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_tangent1,
)
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_velocity_update_no_soft_pd
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
    element_interaction_data_make,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import IncrementalContactPartitioner
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64

_MAX_POINTS = 16
_MAX_ROWS = 3 * _MAX_POINTS
_BLOCK_DIM = 32
_MAX_DOFS = 64
_vec6 = wp.types.vector(length=6, dtype=wp.float32)

_INT64_MAX = 9223372036854775807


@wp.kernel(enable_backward=False)
def _clear_contact_schedule_counts_kernel(section_end: wp.array[wp.int32]):
    section_end[wp.tid()] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _classify_reduced_contact_columns_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    num_columns: wp.array[wp.int32],
    articulation_count: wp.int32,
    key_stride: wp.int64,
    keys: wp.array[wp.int64],
    values: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
):
    """Assign each active column to one deterministic articulation/world run."""
    column = wp.tid()
    values[column] = column
    if column >= num_columns[0]:
        keys[column] = wp.int64(_INT64_MAX)
        return

    owner = reduced_contact_deferred_owner(columns, column, bodies)
    group = owner
    if owner < wp.int32(0):
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        group = articulation_count + wp.max(bodies.world_id[body0], bodies.world_id[body1])
    keys[column] = wp.int64(group) * key_stride + wp.int64(column)
    wp.atomic_add(section_end, group, wp.int32(1))


@wp.kernel(enable_backward=False)
def _compact_fallback_contact_elements_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    num_columns: wp.array[wp.int32],
    articulation_count: wp.int32,
    schedule_section_end: wp.array[wp.int32],
    scheduled_column: wp.array[wp.int32],
    fallback_count: wp.array[wp.int32],
    fallback_column: wp.array[wp.int32],
    fallback_element: wp.array[ElementInteractionData],
):
    fallback = wp.tid()
    start = schedule_section_end[articulation_count - wp.int32(1)]
    count = num_columns[0] - start
    if fallback == wp.int32(0):
        fallback_count[0] = count
    if fallback >= count:
        return
    column = scheduled_column[start + fallback]
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    node0 = wp.int32(-1)
    node1 = wp.int32(-1)
    if bodies.inverse_mass[body0] != wp.float32(0.0):
        node0 = bodies.constraint_node[body0]
    if bodies.inverse_mass[body1] != wp.float32(0.0):
        node1 = bodies.constraint_node[body1]
    if node1 == node0:
        node1 = wp.int32(-1)
    if node0 < wp.int32(0):
        node0 = node1
        node1 = wp.int32(-1)
    fallback_column[fallback] = column
    fallback_element[fallback] = element_interaction_data_make(
        node0, node1, wp.int32(-1), wp.int32(-1), wp.int32(-1), wp.int32(-1), wp.int32(-1), wp.int32(-1)
    )


@wp.kernel(enable_backward=False)
def _solve_fallback_contact_color_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    fallback_column: wp.array[wp.int32],
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    color_cursor: wp.array[wp.int32],
    sweep_direction: wp.array[wp.int32],
    phase: wp.int32,
    use_bias: wp.bool,
):
    tid = wp.tid()
    cursor = color_cursor[0]
    if cursor <= wp.int32(0):
        return
    color_count = num_colors[0]
    step = color_count - cursor
    color = step
    if sweep_direction[0] != wp.int32(0):
        color = color_count - wp.int32(1) - step
    start = color_starts[color]
    end = color_starts[color + wp.int32(1)]
    if tid < end - start:
        fallback = element_ids_by_color[start + tid]
        column = fallback_column[fallback]
        if phase == wp.int32(0):
            reduced_contact_prepare(columns, column, bodies, idt, cc, contacts, wp.bool(False))
        else:
            reduced_contact_iterate(columns, column, bodies, idt, sor_boost, cc, contacts, use_bias, wp.bool(False))
    if tid == wp.int32(0):
        color_cursor[0] = cursor - wp.int32(1)


@wp.kernel(enable_backward=False)
def _gather_reduced_contact_blocks_kernel(
    schedule_section_end: wp.array[wp.int32],
    scheduled_column: wp.array[wp.int32],
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    prepare: wp.bool,
    cc: ContactContainer,
    contacts: ContactViews,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    point_contact: wp.array2d[wp.int32],
    point_column: wp.array2d[wp.int32],
    point0: wp.array2d[wp.vec3],
    point1: wp.array2d[wp.vec3],
    normal: wp.array2d[wp.vec3],
    tangent0: wp.array2d[wp.vec3],
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    row_velocity: wp.array2d[wp.float32],
    delta_lambda: wp.array2d[wp.float32],
):
    articulation = wp.tid()
    start = wp.int32(0)
    if articulation > wp.int32(0):
        start = schedule_section_end[articulation - wp.int32(1)]
    end = schedule_section_end[articulation]

    count = wp.int32(0)
    for index in range(start, end):
        column = scheduled_column[index]
        count += contact_get_contact_count(columns, column)

    articulation_start = bodies.reduced.articulation_start[articulation]
    articulation_end = bodies.reduced.articulation_end[articulation]
    articulation_dofs = (
        bodies.reduced.joint_qd_start[articulation_end] - bodies.reduced.joint_qd_start[articulation_start]
    )
    use_block = count > wp.int32(0) and count <= wp.int32(_MAX_POINTS) and articulation_dofs <= wp.int32(_MAX_DOFS)
    enabled[articulation] = wp.int32(1) if use_block else wp.int32(0)
    point_count[articulation] = count if use_block else wp.int32(0)
    for row in range(_MAX_ROWS):
        row_velocity[articulation, row] = wp.float32(0.0)
        delta_lambda[articulation, row] = wp.float32(0.0)
    if not use_block:
        return

    # The warm start is accumulated first; row velocities are evaluated in a
    # second kernel from one clean, common articulation state.
    point_offset = wp.int32(0)
    for index in range(start, end):
        column = scheduled_column[index]
        if prepare:
            reduced_contact_prepare(columns, column, bodies, idt, cc, contacts, wp.bool(True))
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        first = contact_get_contact_first(columns, column)
        column_count = contact_get_contact_count(columns, column)
        articulation_body = body0
        sign = wp.float32(-1.0)
        if bodies.reduced.body_articulation[body0] < wp.int32(0):
            articulation_body = body1
            sign = wp.float32(1.0)
        for offset in range(column_count):
            point = point_offset + offset
            contact = first + offset
            n = cc_get_normal(cc, contact)
            t0 = cc_get_tangent1(cc, contact)
            t1 = wp.cross(n, t0)
            local0 = cc_get_local_p0(cc, contact)
            local1 = cc_get_local_p1(cc, contact)
            p0 = (
                bodies.position[body0]
                + wp.quat_rotate(bodies.orientation[body0], local0 - bodies.body_com[body0])
                + contacts.rigid_contact_margin0[contact] * n
            )
            p1 = (
                bodies.position[body1]
                + wp.quat_rotate(bodies.orientation[body1], local1 - bodies.body_com[body1])
                - contacts.rigid_contact_margin1[contact] * n
            )
            point_contact[articulation, point] = contact
            point_column[articulation, point] = column
            point0[articulation, point] = p0
            point1[articulation, point] = p1
            normal[articulation, point] = n
            tangent0[articulation, point] = t0
            for axis in range(3):
                direction = n
                if axis == 1:
                    direction = t0
                elif axis == 2:
                    direction = t1
                row = wp.int32(3) * point + wp.int32(axis)
                impulse = sign * direction
                row_body[articulation, row] = articulation_body - wp.int32(1)
                row_wrench[articulation, row] = wp.spatial_vector(impulse, wp.cross(p0 if sign < 0.0 else p1, impulse))
        point_offset += column_count


@wp.kernel(enable_backward=False)
def _reset_reduced_contact_block_delta_kernel(
    enabled: wp.array[wp.int32],
    delta_lambda: wp.array2d[wp.float32],
):
    articulation, row = wp.tid()
    if enabled[articulation] != wp.int32(0):
        delta_lambda[articulation, row] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _finalize_reduced_contact_rows_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    point_column: wp.array2d[wp.int32],
    point0: wp.array2d[wp.vec3],
    point1: wp.array2d[wp.vec3],
    normal: wp.array2d[wp.vec3],
    tangent0: wp.array2d[wp.vec3],
    row_velocity: wp.array2d[wp.float32],
):
    articulation = wp.tid()
    if enabled[articulation] == wp.int32(0):
        return
    for point in range(point_count[articulation]):
        column = point_column[articulation, point]
        body0 = contact_get_body1(columns, column)
        body1 = contact_get_body2(columns, column)
        p0 = point0[articulation, point]
        p1 = point1[articulation, point]
        relative = _deferred_point_velocity(bodies, body1, p1) - _deferred_point_velocity(bodies, body0, p0)
        n = normal[articulation, point]
        t0 = tangent0[articulation, point]
        row = wp.int32(3) * point
        row_velocity[articulation, row] = wp.dot(relative, n)
        row_velocity[articulation, row + wp.int32(1)] = wp.dot(relative, t0)
        row_velocity[articulation, row + wp.int32(2)] = wp.dot(relative, wp.cross(n, t0))


@wp.kernel(enable_backward=False)
def _build_generalized_contact_rows_kernel(
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    jacobian: wp.array3d[wp.float32],
    response: wp.array3d[wp.float32],
    body_work: wp.array3d[wp.spatial_vector],
    joint_work: wp.array3d[wp.float32],
    body_response: wp.array3d[wp.spatial_vector],
):
    articulation, row = wp.tid()
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    dof_start_articulation = data.joint_qd_start[start]
    dof_end_articulation = data.joint_qd_start[end]
    dof_count_articulation = dof_end_articulation - dof_start_articulation
    body_count = end - start
    row_count = wp.int32(3) * point_count[articulation]
    if enabled[articulation] == wp.int32(0) or row >= row_count or dof_count_articulation > wp.int32(_MAX_DOFS):
        return

    for local_body in range(body_count):
        body_work[articulation, row, local_body] = wp.spatial_vector()
        body_response[articulation, row, local_body] = wp.spatial_vector()
    for local_dof in range(dof_count_articulation):
        jacobian[articulation, row, local_dof] = wp.float32(0.0)
        response[articulation, row, local_dof] = wp.float32(0.0)
        joint_work[articulation, row, local_dof] = wp.float32(0.0)

    source_body = row_body[articulation, row]
    wrench = row_wrench[articulation, row]
    source_joint = data.body_joint[source_body]
    body_work[articulation, row, source_joint - start] = -wrench
    path_start = data.body_path_start[source_body]
    path_end = data.body_path_start[source_body + wp.int32(1)]
    for path_index in range(path_start, path_end):
        joint = data.body_path_joint[path_index]
        for dof in range(data.joint_qd_start[joint], data.joint_qd_start[joint + wp.int32(1)]):
            jacobian[articulation, row, dof - dof_start_articulation] = wp.dot(data.joint_s[dof], wrench)

    for reverse in range(body_count):
        joint = end - wp.int32(1) - reverse
        local_child = joint - start
        parent = data.joint_parent[joint]
        p = body_work[articulation, row, local_child]
        propagated = p
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        reduced_force = _vec6(0.0)
        d_inv_u = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                reduced_force[dof_row] = -wp.dot(data.joint_s[dof], p)
                joint_work[articulation, row, dof - dof_start_articulation] = reduced_force[dof_row]
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        d_inv_u[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * reduced_force[dof_column]
                propagated += data.joint_u[dof_start + wp.int32(dof_row)] * d_inv_u[dof_row]
        if parent >= wp.int32(0):
            parent_joint = data.body_joint[parent]
            body_work[articulation, row, parent_joint - start] += propagated

    for local_joint in range(body_count):
        joint = start + local_joint
        parent = data.joint_parent[joint]
        parent_delta = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_delta = body_response[articulation, row, data.body_joint[parent] - start]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        rhs = _vec6(0.0)
        generalized_delta = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                rhs[dof_row] = joint_work[articulation, row, dof - dof_start_articulation] - wp.dot(
                    data.joint_u[dof], parent_delta
                )
        child_delta = parent_delta
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        generalized_delta[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * rhs[dof_column]
                dof = dof_start + wp.int32(dof_row)
                response[articulation, row, dof - dof_start_articulation] = generalized_delta[dof_row]
                child_delta += data.joint_s[dof] * generalized_delta[dof_row]
        body_response[articulation, row, local_joint] = child_delta


@wp.kernel(enable_backward=False, module="reduced_contact_generalized_solve")
def _solve_generalized_contact_tile_kernel(
    columns: ContactColumnContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    cc: ContactContainer,
    iterations: wp.int32,
    use_bias: wp.bool,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    point_contact: wp.array2d[wp.int32],
    point_column: wp.array2d[wp.int32],
    normal: wp.array2d[wp.vec3],
    tangent0: wp.array2d[wp.vec3],
    row_velocity: wp.array2d[wp.float32],
    jacobian: wp.array3d[wp.float32],
    response: wp.array3d[wp.float32],
    generalized_delta_out: wp.array2d[wp.float32],
):
    articulation, lane = wp.tid()
    if enabled[articulation] == wp.int32(0):
        return
    generalized_delta = wp.tile_zeros(shape=_MAX_DOFS, dtype=wp.float32, storage="shared")
    mass_coeff = wp.float32(1.0)
    impulse_coeff = wp.float32(0.0)
    if use_bias:
        _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
            DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
        )

    for iteration in range(iterations):
        for point_offset in range(_MAX_POINTS):
            point = wp.int32(point_offset)
            if (iteration & wp.int32(1)) != wp.int32(0):
                point = point_count[articulation] - wp.int32(1) - wp.int32(point_offset)
            active = point >= wp.int32(0) and point < point_count[articulation]
            delta0 = wp.float32(0.0)
            delta1 = wp.float32(0.0)
            delta2 = wp.float32(0.0)
            row = wp.int32(3) * wp.max(point, wp.int32(0))
            jv0 = wp.float32(0.0)
            jv1 = wp.float32(0.0)
            jv2 = wp.float32(0.0)
            if active:
                jacobian0 = wp.tile_load(jacobian[articulation, row], shape=_MAX_DOFS, storage="register")
                jacobian1 = wp.tile_load(jacobian[articulation, row + wp.int32(1)], shape=_MAX_DOFS, storage="register")
                jacobian2 = wp.tile_load(jacobian[articulation, row + wp.int32(2)], shape=_MAX_DOFS, storage="register")
                jv0 = row_velocity[articulation, row] + wp.tile_extract(wp.tile_sum(jacobian0 * generalized_delta), 0)
                jv1 = row_velocity[articulation, row + wp.int32(1)] + wp.tile_extract(
                    wp.tile_sum(jacobian1 * generalized_delta), 0
                )
                jv2 = row_velocity[articulation, row + wp.int32(2)] + wp.tile_extract(
                    wp.tile_sum(jacobian2 * generalized_delta), 0
                )
            if lane == wp.int32(0) and active:
                contact = point_contact[articulation, point]
                column = point_column[articulation, point]
                n = normal[articulation, point]
                t0 = tangent0[articulation, point]
                t1 = wp.cross(n, t0)
                bias = cc_get_bias(cc, contact)
                speculative = bias > wp.float32(0.0)
                if not (speculative and not use_bias):
                    bias_t0 = cc_get_bias_t1(cc, contact) if use_bias else wp.float32(0.0)
                    bias_t1 = cc_get_bias_t2(cc, contact) if use_bias else wp.float32(0.0)
                    if not use_bias:
                        bias = wp.float32(0.0)
                    row_mass_coeff = mass_coeff
                    row_impulse_coeff = impulse_coeff
                    mu_static = contact_get_friction(columns, column)
                    mu_dynamic = contact_get_friction_dynamic(columns, column)
                    if speculative:
                        row_mass_coeff = wp.float32(1.0)
                        row_impulse_coeff = wp.float32(0.0)
                        if bias > idt * wp.float32(0.002):
                            mu_static = wp.float32(0.0)
                            mu_dynamic = wp.float32(0.0)
                    impulse = contact_project_velocity_update_no_soft_pd(
                        cc,
                        contact,
                        n,
                        t0,
                        t1,
                        jv0,
                        jv1,
                        jv2,
                        cc_get_eff_n(cc, contact),
                        cc_get_eff_t1(cc, contact),
                        cc_get_eff_t2(cc, contact),
                        bias,
                        bias_t0,
                        bias_t1,
                        mu_static,
                        mu_dynamic,
                        row_mass_coeff,
                        row_impulse_coeff,
                        sor_boost,
                        wp.float32(0.0),
                        wp.float32(0.0),
                        wp.float32(0.0),
                    )
                    delta0 = wp.dot(impulse, n)
                    delta1 = wp.dot(impulse, t0)
                    delta2 = wp.dot(impulse, t1)

            broadcast0 = wp.tile_from_thread(shape=_MAX_DOFS, value=delta0, thread_idx=0, storage="shared")
            broadcast1 = wp.tile_from_thread(shape=_MAX_DOFS, value=delta1, thread_idx=0, storage="shared")
            broadcast2 = wp.tile_from_thread(shape=_MAX_DOFS, value=delta2, thread_idx=0, storage="shared")
            if active:
                response0 = wp.tile_load(response[articulation, row], shape=_MAX_DOFS, storage="register")
                response1 = wp.tile_load(response[articulation, row + wp.int32(1)], shape=_MAX_DOFS, storage="register")
                response2 = wp.tile_load(response[articulation, row + wp.int32(2)], shape=_MAX_DOFS, storage="register")
                generalized_delta += broadcast0 * response0 + broadcast1 * response1 + broadcast2 * response2

    wp.tile_store(generalized_delta_out[articulation], generalized_delta)


@wp.kernel(enable_backward=False)
def _apply_generalized_contact_delta_kernel(
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    generalized_delta: wp.array2d[wp.float32],
    body_delta: wp.array2d[wp.spatial_vector],
):
    articulation = wp.tid()
    if enabled[articulation] == wp.int32(0):
        return
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    dof_start_articulation = data.joint_qd_start[start]
    for joint in range(start, end):
        local_joint = joint - start
        parent = data.joint_parent[joint]
        delta = wp.spatial_vector()
        if parent >= wp.int32(0):
            delta = body_delta[articulation, data.body_joint[parent] - start]
        for dof in range(data.joint_qd_start[joint], data.joint_qd_start[joint + wp.int32(1)]):
            dof_delta = generalized_delta[articulation, dof - dof_start_articulation]
            data.joint_qd[dof] += dof_delta
            delta += data.joint_s[dof] * dof_delta
        body_delta[articulation, local_joint] = delta
        child = data.joint_child[joint]
        slot = child + wp.int32(1)
        delta_omega = wp.spatial_bottom(delta)
        bodies.angular_velocity[slot] += delta_omega
        bodies.velocity[slot] += wp.spatial_top(delta) + wp.cross(delta_omega, bodies.position[slot])


class ReducedContactBlockSystem:
    """Graph-stable generalized contact blocks with selectively colored fallback."""

    def __init__(self, model: Model):
        self.device = model.device
        articulation_count = max(1, int(model.articulation_count))
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        max_body_count = max(
            1,
            *(
                int(articulation_end[articulation] - articulation_start[articulation])
                for articulation in range(int(model.articulation_count))
            ),
        )
        self.articulation_count = int(model.articulation_count)
        self.body_count = int(model.body_count) + 1
        self.schedule_capacity = 0
        self.schedule_world_count = 0
        self.schedule_keys: wp.array[wp.int64] | None = None
        self.schedule_columns: wp.array[wp.int32] | None = None
        self.schedule_section_end: wp.array[wp.int32] | None = None
        self.fallback_count: wp.array[wp.int32] | None = None
        self.fallback_column: wp.array[wp.int32] | None = None
        self.fallback_element: wp.array[ElementInteractionData] | None = None
        self.fallback_partitioner: IncrementalContactPartitioner | None = None
        self.enabled = wp.zeros(articulation_count, dtype=wp.int32, device=self.device)
        self.point_count = wp.zeros(articulation_count, dtype=wp.int32, device=self.device)
        self.point_contact = wp.zeros((articulation_count, _MAX_POINTS), dtype=wp.int32, device=self.device)
        self.point_column = wp.zeros_like(self.point_contact)
        self.point0 = wp.zeros((articulation_count, _MAX_POINTS), dtype=wp.vec3, device=self.device)
        self.point1 = wp.zeros_like(self.point0)
        self.normal = wp.zeros_like(self.point0)
        self.tangent0 = wp.zeros_like(self.point0)
        self.row_body = wp.zeros((articulation_count, _MAX_ROWS), dtype=wp.int32, device=self.device)
        self.row_wrench = wp.zeros((articulation_count, _MAX_ROWS), dtype=wp.spatial_vector, device=self.device)
        self.row_velocity = wp.zeros((articulation_count, _MAX_ROWS), dtype=wp.float32, device=self.device)
        self.delta_lambda = wp.zeros_like(self.row_velocity)
        self.jacobian = wp.zeros((articulation_count, _MAX_ROWS, _MAX_DOFS), dtype=wp.float32, device=self.device)
        self.generalized_response = wp.zeros_like(self.jacobian)
        self.generalized_delta = wp.zeros((articulation_count, _MAX_DOFS), dtype=wp.float32, device=self.device)
        self.aba_body_work = wp.zeros(
            (articulation_count, _MAX_ROWS, max_body_count),
            dtype=wp.spatial_vector,
            device=self.device,
        )
        self.aba_joint_work = wp.zeros((articulation_count, _MAX_ROWS, _MAX_DOFS), dtype=wp.float32, device=self.device)
        self.aba_body_response = wp.zeros_like(self.aba_body_work)
        self.generalized_body_delta = wp.zeros(
            (articulation_count, max_body_count), dtype=wp.spatial_vector, device=self.device
        )

    def configure_schedule(self, capacity: int, world_count: int) -> None:
        """Allocate the fixed contact schedule before graph capture."""
        capacity = max(1, int(capacity))
        world_count = max(1, int(world_count))
        if self.schedule_keys is not None:
            if capacity != self.schedule_capacity or world_count != self.schedule_world_count:
                raise RuntimeError("Reduced contact schedule cannot be resized after binding")
            return
        self.schedule_capacity = capacity
        self.schedule_world_count = world_count
        self.schedule_keys = wp.empty(2 * capacity, dtype=wp.int64, device=self.device)
        self.schedule_columns = wp.empty(2 * capacity, dtype=wp.int32, device=self.device)
        self.schedule_section_end = wp.zeros(self.articulation_count + world_count, dtype=wp.int32, device=self.device)
        self.fallback_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.fallback_column = wp.empty(capacity, dtype=wp.int32, device=self.device)
        self.fallback_element = wp.empty(capacity, dtype=ElementInteractionData, device=self.device)
        self.fallback_partitioner = IncrementalContactPartitioner(
            max_num_interactions=capacity,
            max_num_nodes=self.body_count,
            device=self.device,
            seed=0,
            enable_warm_start=True,
        )
        self.fallback_partitioner.set_symmetric_sweep(True)

    def build_schedule(
        self,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        num_columns: wp.array[wp.int32],
    ) -> None:
        """Build stable articulation runs and color only conflicting fallback columns."""
        if self.schedule_keys is None or self.schedule_columns is None or self.schedule_section_end is None:
            raise RuntimeError("Reduced contact schedule has not been configured")
        wp.launch(
            _clear_contact_schedule_counts_kernel,
            dim=self.schedule_section_end.shape[0],
            inputs=[self.schedule_section_end],
            device=self.device,
        )
        wp.launch(
            _classify_reduced_contact_columns_kernel,
            dim=self.schedule_capacity,
            inputs=[
                columns,
                bodies,
                num_columns,
                wp.int32(self.articulation_count),
                wp.int64(self.schedule_capacity + 1),
            ],
            outputs=[self.schedule_keys, self.schedule_columns, self.schedule_section_end],
            device=self.device,
        )
        sort_variable_length_int64(self.schedule_keys, self.schedule_columns, num_columns)
        wp.utils.array_scan(self.schedule_section_end, self.schedule_section_end, inclusive=True)
        assert self.fallback_count is not None
        assert self.fallback_column is not None
        assert self.fallback_element is not None
        wp.launch(
            _compact_fallback_contact_elements_kernel,
            dim=self.schedule_capacity,
            inputs=[
                columns,
                bodies,
                num_columns,
                wp.int32(self.articulation_count),
                self.schedule_section_end,
                self.schedule_columns,
            ],
            outputs=[self.fallback_count, self.fallback_column, self.fallback_element],
            device=self.device,
        )
        self._build_fallback_coloring()

    def _build_fallback_coloring(self) -> None:
        assert self.fallback_partitioner is not None
        assert self.fallback_element is not None
        assert self.fallback_count is not None
        self.fallback_partitioner.reset(self.fallback_element, self.fallback_count)
        self.fallback_partitioner.build_csr()

    def solve_fallback(
        self,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: float,
        cc: ContactContainer,
        contacts: ContactViews,
        iterations: int,
        *,
        use_bias: bool,
        prepare: bool,
    ) -> None:
        """Solve only conflicting fallback contacts with deterministic coloring."""
        assert self.fallback_count is not None
        self._solve_fallback_coloring(
            columns=columns,
            bodies=bodies,
            idt=idt,
            sor_boost=sor_boost,
            cc=cc,
            contacts=contacts,
            iterations=iterations,
            use_bias=use_bias,
            prepare=prepare,
        )

    def _solve_fallback_coloring(
        self,
        *,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: float,
        cc: ContactContainer,
        contacts: ContactViews,
        iterations: int,
        use_bias: bool,
        prepare: bool,
    ) -> None:
        assert self.fallback_partitioner is not None
        assert self.fallback_column is not None
        partitioner = self.fallback_partitioner

        def sweep(phase: int) -> None:
            partitioner.begin_sweep()

            def solve_color() -> None:
                wp.launch(
                    _solve_fallback_contact_color_kernel,
                    dim=self.schedule_capacity,
                    inputs=[
                        columns,
                        bodies,
                        idt,
                        wp.float32(sor_boost),
                        cc,
                        contacts,
                        self.fallback_column,
                        partitioner.element_ids_by_color,
                        partitioner.color_starts,
                        partitioner.num_colors,
                        partitioner.color_cursor,
                        partitioner.sweep_direction,
                        wp.int32(phase),
                        wp.bool(use_bias),
                    ],
                    device=self.device,
                )

            wp.capture_while(partitioner.color_cursor, solve_color)

        if prepare:
            sweep(0)
        for _ in range(iterations):
            sweep(1)

    def solve(
        self,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: float,
        cc: ContactContainer,
        contacts: ContactViews,
        iterations: int,
        *,
        use_bias: bool,
        prepare: bool,
    ) -> None:
        if prepare:
            wp.launch(
                _gather_reduced_contact_blocks_kernel,
                dim=self.articulation_count,
                inputs=[
                    self.schedule_section_end,
                    self.schedule_columns,
                    columns,
                    bodies,
                    idt,
                    wp.bool(True),
                    cc,
                    contacts,
                ],
                outputs=[
                    self.enabled,
                    self.point_count,
                    self.point_contact,
                    self.point_column,
                    self.point0,
                    self.point1,
                    self.normal,
                    self.tangent0,
                    self.row_body,
                    self.row_wrench,
                    self.row_velocity,
                    self.delta_lambda,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _reset_reduced_contact_block_delta_kernel,
                dim=(self.articulation_count, _MAX_ROWS),
                inputs=[self.enabled],
                outputs=[self.delta_lambda],
                device=self.device,
            )
        wp.launch(
            _finalize_reduced_contact_rows_kernel,
            dim=self.articulation_count,
            inputs=[
                columns,
                bodies,
                self.enabled,
                self.point_count,
                self.point_column,
                self.point0,
                self.point1,
                self.normal,
                self.tangent0,
            ],
            outputs=[self.row_velocity],
            device=self.device,
        )
        if prepare:
            wp.launch(
                _build_generalized_contact_rows_kernel,
                dim=(self.articulation_count, _MAX_ROWS),
                inputs=[
                    bodies,
                    self.enabled,
                    self.point_count,
                    self.row_body,
                    self.row_wrench,
                ],
                outputs=[
                    self.jacobian,
                    self.generalized_response,
                    self.aba_body_work,
                    self.aba_joint_work,
                    self.aba_body_response,
                ],
                device=self.device,
            )
        wp.launch_tiled(
            _solve_generalized_contact_tile_kernel,
            dim=[self.articulation_count],
            inputs=[
                columns,
                idt,
                wp.float32(sor_boost),
                cc,
                wp.int32(iterations),
                wp.bool(use_bias),
                self.enabled,
                self.point_count,
                self.point_contact,
                self.point_column,
                self.normal,
                self.tangent0,
                self.row_velocity,
                self.jacobian,
                self.generalized_response,
            ],
            outputs=[self.generalized_delta],
            block_dim=_BLOCK_DIM,
            device=self.device,
        )
        wp.launch(
            _apply_generalized_contact_delta_kernel,
            dim=self.articulation_count,
            inputs=[
                bodies,
                self.enabled,
                self.generalized_delta,
            ],
            outputs=[self.generalized_body_delta],
            device=self.device,
        )


__all__ = ["ReducedContactBlockSystem"]
