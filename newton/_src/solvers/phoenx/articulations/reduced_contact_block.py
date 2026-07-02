# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-cooperative contact-space blocks for reduced articulations."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import Model
from newton._src.solvers.phoenx.articulations.reduced_contact import (
    _deferred_point_velocity,
    reduced_contact_deferred_owner,
    reduced_contact_iterate,
    reduced_contact_prepare,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.cloth_collision import SHAPE_ENDPOINT_KIND_RIGID
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    CONTACT_DWORDS,
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_friction,
    contact_get_friction_dynamic,
    contact_get_side0_kind,
    contact_get_side1_kind,
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
    cc_get_normal_lambda,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
)
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_velocity_update_no_soft_pd
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
    element_interaction_data_make,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import IncrementalContactPartitioner
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64

_POINTS_PER_PAGE = 32
_MAX_ROWS = 3 * _POINTS_PER_PAGE
_CACHED_PAGE_COUNT = 2
_BLOCK_DIM = 32
_RESPONSE_TILE = 32
_RESPONSE_ROW_TILES = _MAX_ROWS // _RESPONSE_TILE
_RESPONSE_DOF_TILES = 2
_RESPONSE_TILES_PER_ARTICULATION = _RESPONSE_ROW_TILES * _RESPONSE_DOF_TILES
_MAX_DOFS = 64
_FALLBACK_MAX_COLORED_PARTITIONS = 63
_FALLBACK_BLOCK_DIM = 256
_vec6 = wp.types.vector(length=6, dtype=wp.float32)

_INT64_MAX = 9223372036854775807


def _fallback_impossible_from_model_pairs(model: Model) -> tuple[bool, bool, bool, bool]:
    """Prove whether model-generated rigid pairs can require fallback."""
    pairs_array = getattr(model, "shape_contact_pairs", None)
    if pairs_array is None or model.shape_body is None:
        return False, False, False, False
    pairs = pairs_array.numpy()
    if pairs.size == 0:
        return True, True, True, True

    body_articulation = np.full(int(model.body_count), -1, dtype=np.int32)
    articulation_start = model.articulation_start.numpy()
    articulation_end = model.articulation_end.numpy()
    joint_child = model.joint_child.numpy()
    for articulation in range(int(model.articulation_count)):
        start = int(articulation_start[articulation])
        end = int(articulation_end[articulation])
        body_articulation[joint_child[start:end]] = articulation

    shape_body = model.shape_body.numpy()
    body0 = shape_body[pairs[:, 0]]
    body1 = shape_body[pairs[:, 1]]
    articulation0 = np.where(body0 >= 0, body_articulation[np.maximum(body0, 0)], -1)
    articulation1 = np.where(body1 >= 0, body_articulation[np.maximum(body1, 0)], -1)
    reduced0 = articulation0 >= 0
    reduced1 = articulation1 >= 0
    inverse_mass = model.body_inv_mass.numpy()
    static0 = (body0 < 0) | (inverse_mass[np.maximum(body0, 0)] == 0.0)
    static1 = (body1 < 0) | (inverse_mass[np.maximum(body1, 0)] == 0.0)
    block_owned = (reduced0 & ~reduced1 & static1) | (reduced1 & ~reduced0 & static0)
    safe = not bool(np.any(~block_owned))
    owner = np.where(reduced0, articulation0, articulation1)
    shape_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    shape_owner = owner[shape_order]
    shape_grouped = bool(np.all(shape_owner[1:] >= shape_owner[:-1]))
    body_lo = np.minimum(body0, body1).astype(np.int64)
    body_hi = np.maximum(body0, body1).astype(np.int64)
    body_key = body_lo * np.int64(max(1, int(model.body_count))) + body_hi
    body_order = np.argsort(body_key, kind="stable")
    body_owner = owner[body_order]
    body_grouped = bool(np.all(body_owner[1:] >= body_owner[:-1]))
    return safe, safe, safe and shape_grouped, safe and body_grouped


@wp.kernel(enable_backward=False)
def _clear_contact_schedule_counts_kernel(section_end: wp.array[wp.int32]):
    section_end[wp.tid()] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _clear_contact_ownership_counts_kernel(
    classic_count: wp.array[wp.int32],
    reduced_count: wp.array[wp.int32],
):
    classic_count[0] = wp.int32(0)
    reduced_count[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _classify_contact_ownership_kernel(
    columns: ContactColumnContainer,
    num_columns: wp.array[wp.int32],
    key_stride: wp.int64,
    keys: wp.array[wp.int64],
    values: wp.array[wp.int32],
    classic_count: wp.array[wp.int32],
    reduced_count: wp.array[wp.int32],
):
    column = wp.tid()
    values[column] = column
    if column >= num_columns[0]:
        keys[column] = wp.int64(_INT64_MAX)
        return
    is_rigid = contact_get_side0_kind(columns, column) == wp.int32(
        SHAPE_ENDPOINT_KIND_RIGID
    ) and contact_get_side1_kind(columns, column) == wp.int32(SHAPE_ENDPOINT_KIND_RIGID)
    ownership = wp.int32(1) if is_rigid else wp.int32(0)
    keys[column] = wp.int64(ownership) * key_stride + wp.int64(column)
    if is_rigid:
        wp.atomic_add(reduced_count, 0, wp.int32(1))
    else:
        wp.atomic_add(classic_count, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def _reorder_contact_columns_kernel(
    source_data: wp.array2d[wp.float32],
    source_pair: wp.array[wp.int32],
    num_columns: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    destination_data: wp.array2d[wp.float32],
    destination_pair: wp.array[wp.int32],
):
    column = wp.tid()
    if column >= num_columns[0]:
        return
    source = permutation[column]
    for row in range(source_data.shape[0]):
        destination_data[row, column] = source_data[row, source]
    destination_pair[column] = source_pair[source]


@wp.kernel(enable_backward=False)
def _set_rigid_only_ownership_kernel(
    num_columns: wp.array[wp.int32],
    contact_offset: wp.int32,
    classic_count: wp.array[wp.int32],
    reduced_count: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
):
    classic_count[0] = wp.int32(0)
    reduced_count[0] = num_columns[0]
    num_active_constraints[0] = contact_offset


@wp.kernel(enable_backward=False)
def _set_regular_constraint_count_kernel(
    contact_offset: wp.int32,
    classic_contact_count: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
):
    num_active_constraints[0] = contact_offset + classic_contact_count[0]


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

    if contact_get_side0_kind(columns, column) != wp.int32(SHAPE_ENDPOINT_KIND_RIGID) or contact_get_side1_kind(
        columns, column
    ) != wp.int32(SHAPE_ENDPOINT_KIND_RIGID):
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
def _classify_grouped_reduced_contact_columns_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    num_columns: wp.array[wp.int32],
    classic_column_count: wp.array[wp.int32],
    values: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
):
    column = wp.tid()
    if column >= num_columns[0]:
        return
    if contact_get_side0_kind(columns, column) != wp.int32(SHAPE_ENDPOINT_KIND_RIGID) or contact_get_side1_kind(
        columns, column
    ) != wp.int32(SHAPE_ENDPOINT_KIND_RIGID):
        return
    owner = reduced_contact_deferred_owner(columns, column, bodies)
    if owner < wp.int32(0):
        return
    values[column - classic_column_count[0]] = column
    wp.atomic_add(section_end, owner, wp.int32(1))


@wp.kernel(enable_backward=False)
def _compact_fallback_contact_elements_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    reduced_column_count: wp.array[wp.int32],
    articulation_count: wp.int32,
    schedule_section_end: wp.array[wp.int32],
    scheduled_column: wp.array[wp.int32],
    fallback_count: wp.array[wp.int32],
    fallback_column: wp.array[wp.int32],
    fallback_element: wp.array[ElementInteractionData],
):
    fallback = wp.tid()
    start = schedule_section_end[articulation_count - wp.int32(1)]
    count = reduced_column_count[0] - start
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


@wp.func
def _solve_fallback_contact_column(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    column: wp.int32,
    phase: wp.int32,
    use_bias: wp.bool,
):
    if phase == wp.int32(0):
        reduced_contact_prepare(columns, column, bodies, idt, cc, contacts, wp.bool(False), wp.bool(True))
    else:
        reduced_contact_iterate(columns, column, bodies, idt, sor_boost, cc, contacts, use_bias, wp.bool(False))


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
    max_colored_partitions: wp.int32,
    worker_count: wp.int32,
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
    has_overflow = color_count > max_colored_partitions
    if sweep_direction[0] != wp.int32(0):
        if has_overflow and step == max_colored_partitions:
            color = max_colored_partitions
        else:
            regular_count = wp.min(color_count, max_colored_partitions)
            color = regular_count - wp.int32(1) - step
    start = color_starts[color]
    end = color_starts[color + wp.int32(1)]
    if has_overflow and color == max_colored_partitions:
        if tid == wp.int32(0):
            for index in range(start, end):
                fallback = element_ids_by_color[index]
                _solve_fallback_contact_column(
                    columns, bodies, idt, sor_boost, cc, contacts, fallback_column[fallback], phase, use_bias
                )
    else:
        index = start + tid
        while index < end:
            fallback = element_ids_by_color[index]
            _solve_fallback_contact_column(
                columns, bodies, idt, sor_boost, cc, contacts, fallback_column[fallback], phase, use_bias
            )
            index += worker_count
    if tid == wp.int32(0):
        color_cursor[0] = cursor - wp.int32(1)


@wp.kernel(enable_backward=False)
def _count_reduced_contact_pages_kernel(
    schedule_section_end: wp.array[wp.int32],
    scheduled_column: wp.array[wp.int32],
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    total_point_count: wp.array[wp.int32],
    max_page_count: wp.array[wp.int32],
    multi_page_active: wp.array[wp.int32],
    overflow_page_active: wp.array[wp.int32],
    transpose_active: wp.array[wp.int32],
    deferred_active: wp.array[wp.int32],
):
    articulation = wp.tid()
    start = wp.int32(0)
    if articulation > wp.int32(0):
        start = schedule_section_end[articulation - wp.int32(1)]
    end = schedule_section_end[articulation]

    count = wp.int32(0)
    for index in range(start, end):
        count += contact_get_contact_count(columns, scheduled_column[index])

    articulation_start = bodies.reduced.articulation_start[articulation]
    articulation_end = bodies.reduced.articulation_end[articulation]
    articulation_dofs = (
        bodies.reduced.joint_qd_start[articulation_end] - bodies.reduced.joint_qd_start[articulation_start]
    )
    use_block = count > wp.int32(0) and articulation_dofs <= wp.int32(_MAX_DOFS)
    enabled[articulation] = wp.int32(1) if use_block else wp.int32(0)
    total_point_count[articulation] = count
    if use_block:
        pages = (count + wp.int32(_POINTS_PER_PAGE - 1)) // wp.int32(_POINTS_PER_PAGE)
        wp.atomic_max(max_page_count, wp.int32(0), pages)
        if pages > wp.int32(1):
            wp.atomic_max(multi_page_active, wp.int32(0), wp.int32(1))
        if pages > wp.int32(_CACHED_PAGE_COUNT):
            wp.atomic_max(overflow_page_active, wp.int32(0), wp.int32(1))
        if count * wp.int32(3) > wp.int32(_RESPONSE_TILE):
            wp.atomic_max(transpose_active, wp.int32(0), wp.int32(1))
    elif count > wp.int32(0):
        wp.atomic_max(deferred_active, wp.int32(0), wp.int32(1))


@wp.kernel(enable_backward=False)
def _reset_reduced_contact_page_cursor_kernel(
    max_page_count: wp.array[wp.int32],
    page_cursor: wp.array[wp.int32],
    page_index: wp.array[wp.int32],
):
    page_cursor[0] = max_page_count[0]
    page_index[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _advance_reduced_contact_page_cursor_kernel(
    page_cursor: wp.array[wp.int32],
    page_index: wp.array[wp.int32],
):
    page_cursor[0] -= wp.int32(1)
    page_index[0] += wp.int32(1)


@wp.kernel(enable_backward=False, module="reduced_contact_gather")
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
    total_point_count: wp.array[wp.int32],
    page_index: wp.array[wp.int32],
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
):
    articulation, lane = wp.tid()
    start = wp.int32(0)
    if articulation > wp.int32(0):
        start = schedule_section_end[articulation - wp.int32(1)]
    end = schedule_section_end[articulation]

    page = page_index[0]
    page_start = page * wp.int32(_POINTS_PER_PAGE)
    storage_page = wp.min(page, wp.int32(_CACHED_PAGE_COUNT - 1))
    packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + storage_page
    remaining = total_point_count[articulation] - page_start
    count = wp.min(wp.max(remaining, wp.int32(0)), wp.int32(_POINTS_PER_PAGE))
    if lane == wp.int32(0):
        point_count[packed_articulation] = count if enabled[articulation] != wp.int32(0) else wp.int32(0)
    if enabled[articulation] == wp.int32(0) or count <= wp.int32(0):
        return

    page_end = page_start + count
    global_point = page_start + lane
    point_offset = wp.int32(0)
    for index in range(start, end):
        column = scheduled_column[index]
        column_count = contact_get_contact_count(columns, column)
        column_end = point_offset + column_count
        overlaps_page = point_offset < page_end and column_end > page_start
        if overlaps_page and prepare and (index - start) % wp.int32(_BLOCK_DIM) == lane:
            reduced_contact_prepare(columns, column, bodies, idt, cc, contacts, wp.bool(True), wp.bool(False))

        if global_point >= point_offset and global_point < column_end and global_point < page_end:
            point = lane
            offset = global_point - point_offset
            contact = contact_get_contact_first(columns, column) + offset
            body0 = contact_get_body1(columns, column)
            body1 = contact_get_body2(columns, column)
            articulation_body = body0
            sign = wp.float32(-1.0)
            if bodies.reduced.body_articulation[body0] < wp.int32(0):
                articulation_body = body1
                sign = wp.float32(1.0)

            n = cc_get_normal(cc, contact)
            t0 = cc_get_tangent1(cc, contact)
            t1 = wp.cross(n, t0)
            local0 = cc_get_local_p0(cc, contact)
            local1 = cc_get_local_p1(cc, contact)
            offset0 = (
                wp.quat_rotate(bodies.orientation[body0], local0 - bodies.body_com[body0])
                + contacts.rigid_contact_margin0[contact] * n
            )
            offset1 = (
                wp.quat_rotate(bodies.orientation[body1], local1 - bodies.body_com[body1])
                - contacts.rigid_contact_margin1[contact] * n
            )
            p0 = bodies.position[body0] + offset0
            p1 = bodies.position[body1] + offset1
            separation = p1 - p0
            contact_point = p0 + wp.float32(0.5) * separation
            point_contact[packed_articulation, point] = contact
            point_column[packed_articulation, point] = column
            point0[packed_articulation, point] = contact_point
            point1[packed_articulation, point] = contact_point
            normal[packed_articulation, point] = n
            tangent0[packed_articulation, point] = t0

            relative = _deferred_point_velocity(bodies, body1, contact_point) - _deferred_point_velocity(
                bodies, body0, contact_point
            )
            row = wp.int32(3) * point
            row_velocity[packed_articulation, row] = wp.dot(relative, n)
            row_velocity[packed_articulation, row + wp.int32(1)] = wp.dot(relative, t0)
            row_velocity[packed_articulation, row + wp.int32(2)] = wp.dot(relative, t1)

            local_com_position = wp.transform_get_translation(
                bodies.reduced.body_q_com[articulation_body - wp.int32(1)]
            )
            point_local = local_com_position + offset0 + wp.float32(0.5) * separation
            if sign > wp.float32(0.0):
                point_local = local_com_position + offset1 - wp.float32(0.5) * separation
            for axis in range(3):
                direction = n
                if axis == 1:
                    direction = t0
                elif axis == 2:
                    direction = t1
                impulse = sign * direction
                row = wp.int32(3) * point + wp.int32(axis)
                row_body[packed_articulation, row] = articulation_body - wp.int32(1)
                row_wrench[packed_articulation, row] = wp.spatial_vector(impulse, wp.cross(point_local, impulse))
        point_offset = column_end


@wp.kernel(enable_backward=False)
def _finalize_reduced_contact_rows_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    page_index: wp.array[wp.int32],
    point_column: wp.array2d[wp.int32],
    point0: wp.array2d[wp.vec3],
    point1: wp.array2d[wp.vec3],
    normal: wp.array2d[wp.vec3],
    tangent0: wp.array2d[wp.vec3],
    row_velocity: wp.array2d[wp.float32],
):
    articulation, point = wp.tid()
    storage_page = wp.min(page_index[0], wp.int32(_CACHED_PAGE_COUNT - 1))
    packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + storage_page
    if enabled[articulation] == wp.int32(0) or point >= point_count[packed_articulation]:
        return
    column = point_column[packed_articulation, point]
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    p0 = point0[packed_articulation, point]
    p1 = point1[packed_articulation, point]
    relative = _deferred_point_velocity(bodies, body1, p1) - _deferred_point_velocity(bodies, body0, p0)
    n = normal[packed_articulation, point]
    t0 = tangent0[packed_articulation, point]
    row = wp.int32(3) * point
    row_velocity[packed_articulation, row] = wp.dot(relative, n)
    row_velocity[packed_articulation, row + wp.int32(1)] = wp.dot(relative, t0)
    row_velocity[packed_articulation, row + wp.int32(2)] = wp.dot(relative, wp.cross(n, t0))


@wp.kernel(enable_backward=False, module="reduced_contact_rows")
def _build_generalized_contact_rows_kernel(
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    jacobian: wp.array3d[wp.float32],
    response: wp.array3d[wp.float32],
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
    row_count = wp.int32(3) * point_count[articulation]
    if enabled[articulation] == wp.int32(0) or row >= row_count or dof_count_articulation > wp.int32(jacobian.shape[2]):
        return

    # Response overwrites every topology-owned DOF below; its zero-allocated tile tail is immutable.
    for local_dof in range(dof_count_articulation):
        jacobian[articulation, row, local_dof] = wp.float32(0.0)
        joint_work[articulation, local_dof, row] = wp.float32(0.0)

    source_body = row_body[articulation, row]
    source_wrench = row_wrench[articulation, row]
    path_start = data.body_path_start[source_body]
    path_end = data.body_path_start[source_body + wp.int32(1)]

    propagated_wrench = source_wrench
    for reverse in range(path_end - path_start):
        path_index = path_end - wp.int32(1) - reverse
        joint = data.body_path_joint[path_index]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        projected = _vec6(0.0)
        reduced = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                projected[dof_row] = wp.dot(data.joint_s[dof], propagated_wrench)
                joint_work[articulation, dof - dof_start_articulation, row] = projected[dof_row]
                jacobian[articulation, row, dof - dof_start_articulation] = wp.dot(data.joint_s[dof], source_wrench)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        reduced[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * projected[dof_column]
                propagated_wrench -= data.joint_u[dof_start + wp.int32(dof_row)] * reduced[dof_row]

    for joint in range(start, end):
        local_joint = joint - start
        parent = data.joint_parent[joint]
        parent_delta = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_delta = body_response[articulation, data.body_joint[parent] - start, row]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        rhs = _vec6(0.0)
        generalized_delta = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                rhs[dof_row] = joint_work[articulation, dof - dof_start_articulation, row] - wp.dot(
                    data.joint_u[dof], parent_delta
                )
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        generalized_delta[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * rhs[dof_column]
                dof = dof_start + wp.int32(dof_row)
                response[articulation, row, dof - dof_start_articulation] = generalized_delta[dof_row]
                parent_delta += data.joint_s[dof] * generalized_delta[dof_row]
        body_response[articulation, local_joint, row] = parent_delta


@wp.kernel(enable_backward=False, module="reduced_contact_rows_packed")
def _build_packed_generalized_contact_rows_kernel(
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    point_contact: wp.array2d[wp.int32],
    cc: ContactContainer,
    max_page_count: wp.array[wp.int32],
    page_index: wp.array[wp.int32],
    prepare: wp.bool,
    packed_jacobian: wp.array2d[wp.float32],
    packed_response: wp.array2d[wp.float32],
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
    page = page_index[0]
    storage_page = wp.min(page, wp.int32(_CACHED_PAGE_COUNT - 1))
    packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + storage_page
    row_count = wp.int32(3) * point_count[packed_articulation]
    if (
        enabled[articulation] == wp.int32(0)
        or row >= row_count
        or dof_count_articulation > wp.int32(packed_jacobian.shape[1])
    ):
        return

    if not prepare and (
        page == wp.int32(0) or (page == wp.int32(1) and max_page_count[0] <= wp.int32(_CACHED_PAGE_COUNT))
    ):
        return
    packed_row = packed_articulation * wp.int32(_MAX_ROWS) + row

    # The dense Jacobian needs explicit zeros; projected scratch is read only on the source path below.
    for local_dof in range(dof_count_articulation):
        packed_jacobian[packed_row, local_dof] = wp.float32(0.0)

    source_body = row_body[packed_articulation, row]
    source_wrench = row_wrench[packed_articulation, row]
    path_start = data.body_path_start[source_body]
    path_end = data.body_path_start[source_body + wp.int32(1)]

    propagated_wrench = source_wrench
    for reverse in range(path_end - path_start):
        path_index = path_end - wp.int32(1) - reverse
        joint = data.body_path_joint[path_index]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        projected = _vec6(0.0)
        reduced = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                projected[dof_row] = wp.dot(data.joint_s[dof], propagated_wrench)
                joint_work[articulation, dof - dof_start_articulation, row] = projected[dof_row]
                packed_jacobian[packed_row, dof - dof_start_articulation] = wp.dot(data.joint_s[dof], source_wrench)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        reduced[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * projected[dof_column]
                propagated_wrench -= data.joint_u[dof_start + wp.int32(dof_row)] * reduced[dof_row]

    inverse_mass = wp.float32(0.0)
    # Parent-first joint ordering makes the root-to-leaf path a monotone cursor through this forward pass.
    path_cursor = path_start
    next_path_joint = wp.int32(-1)
    if path_cursor < path_end:
        next_path_joint = data.body_path_joint[path_cursor]
    for joint in range(start, end):
        local_joint = joint - start
        on_source_path = joint == next_path_joint
        parent = data.joint_parent[joint]
        parent_delta = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_delta = body_response[articulation, data.body_joint[parent] - start, row]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        rhs = _vec6(0.0)
        generalized_delta = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                rhs[dof_row] = -wp.dot(data.joint_u[dof], parent_delta)
                if on_source_path:
                    rhs[dof_row] += joint_work[articulation, dof - dof_start_articulation, row]
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        generalized_delta[dof_row] += data.joint_d_inv[joint, dof_row, dof_column] * rhs[dof_column]
                dof = dof_start + wp.int32(dof_row)
                response_value = generalized_delta[dof_row]
                if row_count > wp.int32(_RESPONSE_TILE):
                    joint_work[articulation, dof - dof_start_articulation, row] = response_value
                else:
                    packed_response[packed_row, dof - dof_start_articulation] = response_value
                inverse_mass += packed_jacobian[packed_row, dof - dof_start_articulation] * response_value
                parent_delta += data.joint_s[dof] * response_value
        body_response[articulation, local_joint, row] = parent_delta
        if on_source_path:
            path_cursor += wp.int32(1)
            next_path_joint = wp.int32(-1)
            if path_cursor < path_end:
                next_path_joint = data.body_path_joint[path_cursor]

    if inverse_mass > wp.float32(1.0e-12):
        effective_mass = wp.float32(1.0) / inverse_mass
        point = row // wp.int32(3)
        axis = row - wp.int32(3) * point
        contact = point_contact[packed_articulation, point]
        if axis == wp.int32(0):
            cc_set_eff_n(cc, contact, effective_mass)
        elif axis == wp.int32(1):
            cc_set_eff_t1(cc, contact, effective_mass)
        else:
            cc_set_eff_t2(cc, contact, effective_mass)


@wp.kernel(enable_backward=False, module="reduced_contact_response_transpose")
def _transpose_generalized_contact_response_kernel(
    bodies: BodyContainer,
    enabled: wp.array[wp.int32],
    point_count: wp.array[wp.int32],
    page_index: wp.array[wp.int32],
    max_page_count: wp.array[wp.int32],
    prepare: wp.bool,
    joint_work: wp.array3d[wp.float32],
    packed_response: wp.array2d[wp.float32],
):
    tile, _lane = wp.tid()
    page = page_index[0]
    if not prepare and (
        page == wp.int32(0) or (page == wp.int32(1) and max_page_count[0] <= wp.int32(_CACHED_PAGE_COUNT))
    ):
        return
    articulation = tile // wp.int32(_RESPONSE_TILES_PER_ARTICULATION)
    local_tile = tile - articulation * wp.int32(_RESPONSE_TILES_PER_ARTICULATION)
    dof_tile = local_tile // wp.int32(_RESPONSE_ROW_TILES)
    row_tile = local_tile - dof_tile * wp.int32(_RESPONSE_ROW_TILES)
    storage_page = wp.min(page, wp.int32(_CACHED_PAGE_COUNT - 1))
    packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + storage_page
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    dof_count = data.joint_qd_start[end] - data.joint_qd_start[start]
    row_count = wp.int32(3) * point_count[packed_articulation]
    if (
        enabled[articulation] == wp.int32(0)
        or row_count <= wp.int32(_RESPONSE_TILE)
        or dof_tile * wp.int32(_RESPONSE_TILE) >= dof_count
        or row_tile * wp.int32(_RESPONSE_TILE) >= row_count
    ):
        return
    source = wp.tile_load(
        joint_work[articulation],
        shape=(_RESPONSE_TILE, _RESPONSE_TILE),
        offset=(dof_tile * wp.int32(_RESPONSE_TILE), row_tile * wp.int32(_RESPONSE_TILE)),
        storage="shared",
    )
    wp.tile_store(
        packed_response,
        wp.tile_transpose(source),
        offset=(
            packed_articulation * wp.int32(_MAX_ROWS) + row_tile * wp.int32(_RESPONSE_TILE),
            dof_tile * wp.int32(_RESPONSE_TILE),
        ),
    )


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
"""
)
def _sync_contact_warp(): ...


def _make_solve_generalized_contact_tile_kernel(max_dofs: int):
    module = wp.get_module(f"reduced_contact_generalized_solve_{max_dofs}")

    @wp.kernel(enable_backward=False, module=module)
    def _solve_generalized_contact_tile_kernel(
        columns: ContactColumnContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        cc: ContactContainer,
        iterations: wp.int32,
        use_bias: wp.bool,
        warmstart: wp.bool,
        enabled: wp.array[wp.int32],
        point_count: wp.array[wp.int32],
        point_contact: wp.array2d[wp.int32],
        point_column: wp.array2d[wp.int32],
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_velocity: wp.array2d[wp.float32],
        page_index: wp.array[wp.int32],
        packed_jacobian: wp.array2d[wp.float32],
        packed_response: wp.array2d[wp.float32],
        bodies: BodyContainer,
        fuse_apply: wp.bool,
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        generalized_delta_out: wp.array2d[wp.float32],
        body_delta: wp.array2d[wp.spatial_vector],
    ):
        articulation, lane = wp.tid()
        if enabled[articulation] == wp.int32(0):
            return
        storage_page = wp.min(page_index[0], wp.int32(_CACHED_PAGE_COUNT - 1))
        packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + storage_page
        generalized_delta = wp.tile_zeros(shape=max_dofs, dtype=wp.float32, storage="shared")
        active_point_count = point_count[packed_articulation]
        if warmstart:
            for point_offset in range(active_point_count):
                point = wp.int32(point_offset)
                lambda0 = wp.float32(0.0)
                lambda1 = wp.float32(0.0)
                lambda2 = wp.float32(0.0)
                if lane == wp.int32(0):
                    contact = point_contact[packed_articulation, point]
                    lambda0 = cc_get_normal_lambda(cc, contact)
                    lambda1 = cc_get_tangent1_lambda(cc, contact)
                    lambda2 = cc_get_tangent2_lambda(cc, contact)
                broadcast0 = wp.tile_from_thread(shape=max_dofs, value=lambda0, thread_idx=0, storage="shared")
                broadcast1 = wp.tile_from_thread(shape=max_dofs, value=lambda1, thread_idx=0, storage="shared")
                broadcast2 = wp.tile_from_thread(shape=max_dofs, value=lambda2, thread_idx=0, storage="shared")
                row = wp.int32(3) * point
                packed_row = packed_articulation * wp.int32(_MAX_ROWS) + row
                response0 = wp.tile_load(packed_response[packed_row], shape=max_dofs, storage="register")
                response1 = wp.tile_load(packed_response[packed_row + wp.int32(1)], shape=max_dofs, storage="register")
                response2 = wp.tile_load(packed_response[packed_row + wp.int32(2)], shape=max_dofs, storage="register")
                generalized_delta += broadcast0 * response0 + broadcast1 * response1 + broadcast2 * response2
        mass_coeff = wp.float32(1.0)
        impulse_coeff = wp.float32(0.0)
        if use_bias:
            _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
                DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
            )

        for iteration in range(iterations):
            for point_offset in range(active_point_count):
                point = wp.int32(point_offset)
                if (iteration & wp.int32(1)) != wp.int32(0):
                    point = active_point_count - wp.int32(1) - wp.int32(point_offset)
                delta0 = wp.float32(0.0)
                delta1 = wp.float32(0.0)
                delta2 = wp.float32(0.0)
                row = wp.int32(3) * point
                packed_row = packed_articulation * wp.int32(_MAX_ROWS) + row
                jacobian0 = wp.tile_load(packed_jacobian[packed_row], shape=max_dofs, storage="register")
                jacobian1 = wp.tile_load(packed_jacobian[packed_row + wp.int32(1)], shape=max_dofs, storage="register")
                jacobian2 = wp.tile_load(packed_jacobian[packed_row + wp.int32(2)], shape=max_dofs, storage="register")
                jv0 = row_velocity[packed_articulation, row] + wp.tile_extract(
                    wp.tile_sum(jacobian0 * generalized_delta), 0
                )
                jv1 = row_velocity[packed_articulation, row + wp.int32(1)] + wp.tile_extract(
                    wp.tile_sum(jacobian1 * generalized_delta), 0
                )
                jv2 = row_velocity[packed_articulation, row + wp.int32(2)] + wp.tile_extract(
                    wp.tile_sum(jacobian2 * generalized_delta), 0
                )
                if lane == wp.int32(0):
                    contact = point_contact[packed_articulation, point]
                    column = point_column[packed_articulation, point]
                    n = normal[packed_articulation, point]
                    t0 = tangent0[packed_articulation, point]
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

                broadcast0 = wp.tile_from_thread(shape=max_dofs, value=delta0, thread_idx=0, storage="shared")
                broadcast1 = wp.tile_from_thread(shape=max_dofs, value=delta1, thread_idx=0, storage="shared")
                broadcast2 = wp.tile_from_thread(shape=max_dofs, value=delta2, thread_idx=0, storage="shared")
                response0 = wp.tile_load(packed_response[packed_row], shape=max_dofs, storage="register")
                response1 = wp.tile_load(packed_response[packed_row + wp.int32(1)], shape=max_dofs, storage="register")
                response2 = wp.tile_load(packed_response[packed_row + wp.int32(2)], shape=max_dofs, storage="register")
                generalized_delta += broadcast0 * response0 + broadcast1 * response1 + broadcast2 * response2

        wp.tile_store(generalized_delta_out[articulation], generalized_delta)
        if not fuse_apply:
            return
        _sync_contact_warp()

        data = bodies.reduced
        start = data.articulation_start[articulation]
        dof_start_articulation = data.joint_qd_start[start]
        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                local_joint = joint - start
                parent = data.joint_parent[joint]
                delta = wp.spatial_vector()
                if parent >= wp.int32(0):
                    delta = body_delta[articulation, data.body_joint[parent] - start]
                for dof in range(data.joint_qd_start[joint], data.joint_qd_start[joint + wp.int32(1)]):
                    dof_delta = generalized_delta_out[articulation, dof - dof_start_articulation]
                    data.joint_qd[dof] += dof_delta
                    delta += data.joint_s[dof] * dof_delta
                body_delta[articulation, local_joint] = delta
                child = data.joint_child[joint]
                slot = child + wp.int32(1)
                delta_omega = wp.spatial_bottom(delta)
                bodies.angular_velocity[slot] += delta_omega
                local_com_position = wp.transform_get_translation(data.body_q_com[child])
                bodies.velocity[slot] += wp.spatial_top(delta) + wp.cross(delta_omega, local_com_position)
                index += wp.int32(_BLOCK_DIM)
            _sync_contact_warp()

    return _solve_generalized_contact_tile_kernel


_SOLVE_GENERALIZED_CONTACT_TILE_KERNELS = {
    32: _make_solve_generalized_contact_tile_kernel(32),
    40: _make_solve_generalized_contact_tile_kernel(40),
    48: _make_solve_generalized_contact_tile_kernel(48),
    64: _make_solve_generalized_contact_tile_kernel(64),
}


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
        local_com_position = wp.transform_get_translation(data.body_q_com[child])
        bodies.velocity[slot] += wp.spatial_top(delta) + wp.cross(delta_omega, local_com_position)


class ReducedContactBlockSystem:
    """Graph-stable generalized contact blocks with selectively colored fallback."""

    def __init__(
        self,
        model: Model,
        *,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        max_depth: int,
    ):
        self.device = model.device
        self.articulation_depth_start = articulation_depth_start
        self.articulation_depth_joint = articulation_depth_joint
        self.max_depth = int(max_depth)
        self.fuse_apply = bool(self.device.is_cuda)
        (
            self._fallback_impossible_partitioned,
            self._fallback_impossible_rigid_only,
            self._shape_schedule_already_grouped,
            self._body_schedule_already_grouped,
        ) = _fallback_impossible_from_model_pairs(model)
        self._skip_fallback_coloring = False
        self._schedule_already_grouped = False
        self._body_pair_grouping = False
        articulation_count = max(1, int(model.articulation_count))
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        max_dof_count = max(
            1,
            *(
                int(joint_qd_start[int(articulation_end[articulation])])
                - int(joint_qd_start[int(articulation_start[articulation])])
                for articulation in range(int(model.articulation_count))
            ),
        )
        if max_dof_count <= 32:
            self.contact_dof_width = 32
        elif max_dof_count <= 40:
            self.contact_dof_width = 40
        elif max_dof_count <= 48:
            self.contact_dof_width = 48
        else:
            self.contact_dof_width = 64
        self.solve_kernel = _SOLVE_GENERALIZED_CONTACT_TILE_KERNELS[self.contact_dof_width]
        self.requires_impulse_response = not (
            self._fallback_impossible_partitioned
            and self._fallback_impossible_rigid_only
            and max_dof_count <= _MAX_DOFS
        )
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
        self.fallback_worker_count = 1
        self.schedule_keys: wp.array[wp.int64] | None = None
        self.schedule_columns: wp.array[wp.int32] | None = None
        self.schedule_section_end: wp.array[wp.int32] | None = None
        self.classic_column_count: wp.array[wp.int32] | None = None
        self.reduced_column_count: wp.array[wp.int32] | None = None
        self.reorder_data: wp.array2d[wp.float32] | None = None
        self.reorder_pair_source: wp.array[wp.int32] | None = None
        self.fallback_count: wp.array[wp.int32] | None = None
        self.fallback_column: wp.array[wp.int32] | None = None
        self.fallback_element: wp.array[ElementInteractionData] | None = None
        self.fallback_partitioner: IncrementalContactPartitioner | None = None
        self.packed_jacobian: wp.array2d[wp.float32] | None = None
        self.packed_response: wp.array2d[wp.float32] | None = None
        self.enabled = wp.zeros(articulation_count, dtype=wp.int32, device=self.device)
        self.total_point_count = wp.zeros(articulation_count, dtype=wp.int32, device=self.device)
        self.max_page_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.multi_page_active = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow_page_active = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.transpose_active = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.page_cursor = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.page_index = wp.zeros(1, dtype=wp.int32, device=self.device)
        packed_articulation_count = articulation_count * _CACHED_PAGE_COUNT
        self.point_count = wp.zeros(packed_articulation_count, dtype=wp.int32, device=self.device)
        self.point_contact = wp.zeros((packed_articulation_count, _POINTS_PER_PAGE), dtype=wp.int32, device=self.device)
        self.point_column = wp.zeros_like(self.point_contact)
        self.point0 = wp.zeros((packed_articulation_count, _POINTS_PER_PAGE), dtype=wp.vec3, device=self.device)
        self.point1 = wp.zeros_like(self.point0)
        self.normal = wp.zeros_like(self.point0)
        self.tangent0 = wp.zeros_like(self.point0)
        self.row_body = wp.zeros((packed_articulation_count, _MAX_ROWS), dtype=wp.int32, device=self.device)
        self.row_wrench = wp.zeros((packed_articulation_count, _MAX_ROWS), dtype=wp.spatial_vector, device=self.device)
        self.row_velocity = wp.zeros((packed_articulation_count, _MAX_ROWS), dtype=wp.float32, device=self.device)
        self.deferred_active = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.generalized_delta = wp.zeros(
            (articulation_count, self.contact_dof_width), dtype=wp.float32, device=self.device
        )
        self.aba_joint_work = wp.zeros(
            (articulation_count, self.contact_dof_width, _MAX_ROWS), dtype=wp.float32, device=self.device
        )
        self.aba_body_response = wp.zeros(
            (articulation_count, max_body_count, _MAX_ROWS),
            dtype=wp.spatial_vector,
            device=self.device,
        )
        self.generalized_body_delta = wp.zeros(
            (articulation_count, max_body_count), dtype=wp.spatial_vector, device=self.device
        )

    def configure_schedule(self, capacity: int, world_count: int, *, body_pair_grouping: bool) -> None:
        """Allocate the fixed schedule and bounded two-page contact cache."""
        capacity = max(1, int(capacity))
        world_count = max(1, int(world_count))
        body_pair_grouping = bool(body_pair_grouping)
        if self.schedule_keys is not None:
            if capacity != self.schedule_capacity or world_count != self.schedule_world_count:
                raise RuntimeError("Reduced contact schedule cannot be resized after binding")
            if body_pair_grouping != self._body_pair_grouping:
                raise RuntimeError("Reduced contact ordering cannot be changed after binding")
            return
        self.schedule_capacity = capacity
        self.schedule_world_count = world_count
        self._body_pair_grouping = body_pair_grouping
        packed_row_capacity = self.articulation_count * _CACHED_PAGE_COUNT * _MAX_ROWS
        self.packed_jacobian = wp.zeros(
            (packed_row_capacity, self.contact_dof_width),
            dtype=wp.float32,
            device=self.device,
        )
        self.packed_response = wp.zeros_like(self.packed_jacobian)
        worker_blocks = max(1, int(getattr(self.device, "sm_count", 1)))
        self.fallback_worker_count = min(capacity, worker_blocks * _FALLBACK_BLOCK_DIM)
        self.schedule_keys = wp.empty(2 * capacity, dtype=wp.int64, device=self.device)
        self.schedule_columns = wp.empty(2 * capacity, dtype=wp.int32, device=self.device)
        self.schedule_section_end = wp.zeros(self.articulation_count + world_count, dtype=wp.int32, device=self.device)
        self.classic_column_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.reduced_column_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.reorder_data = wp.empty((CONTACT_DWORDS, capacity), dtype=wp.float32, device=self.device)
        self.reorder_pair_source = wp.empty(capacity, dtype=wp.int32, device=self.device)
        self.fallback_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.fallback_column = wp.empty(capacity, dtype=wp.int32, device=self.device)
        self.fallback_element = wp.empty(capacity, dtype=ElementInteractionData, device=self.device)
        self.fallback_partitioner = IncrementalContactPartitioner(
            max_num_interactions=capacity,
            max_num_nodes=self.body_count,
            device=self.device,
            seed=0,
            max_colored_partitions=_FALLBACK_MAX_COLORED_PARTITIONS,
            enable_warm_start=True,
        )
        self.fallback_partitioner.set_symmetric_sweep(True)

    def build_schedule(
        self,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        pair_source: wp.array[wp.int32],
        num_columns: wp.array[wp.int32],
        num_active_constraints: wp.array[wp.int32],
        contact_offset: int,
        *,
        partition_ownership: bool,
    ) -> None:
        """Partition classic/reduced ownership, then build reduced contact runs."""
        if self.schedule_keys is None or self.schedule_columns is None or self.schedule_section_end is None:
            raise RuntimeError("Reduced contact schedule has not been configured")
        assert self.classic_column_count is not None
        assert self.reduced_column_count is not None
        assert self.reorder_data is not None
        assert self.reorder_pair_source is not None

        if partition_ownership:
            wp.copy(self.reorder_data, columns.data)
            wp.launch(
                _clear_contact_ownership_counts_kernel,
                dim=1,
                outputs=[self.classic_column_count, self.reduced_column_count],
                device=self.device,
            )
            wp.launch(
                _classify_contact_ownership_kernel,
                dim=self.schedule_capacity,
                inputs=[columns, num_columns, wp.int64(self.schedule_capacity + 1)],
                outputs=[
                    self.schedule_keys,
                    self.schedule_columns,
                    self.classic_column_count,
                    self.reduced_column_count,
                ],
                device=self.device,
            )
            sort_variable_length_int64(self.schedule_keys, self.schedule_columns, num_columns)
            wp.launch(
                _reorder_contact_columns_kernel,
                dim=self.schedule_capacity,
                inputs=[self.reorder_data, pair_source, num_columns, self.schedule_columns],
                outputs=[columns.data, self.reorder_pair_source],
                device=self.device,
            )
            wp.copy(pair_source, self.reorder_pair_source, count=self.schedule_capacity)
            wp.launch(
                _set_regular_constraint_count_kernel,
                dim=1,
                inputs=[wp.int32(contact_offset), self.classic_column_count],
                outputs=[num_active_constraints],
                device=self.device,
            )
        else:
            wp.launch(
                _set_rigid_only_ownership_kernel,
                dim=1,
                inputs=[num_columns, wp.int32(contact_offset)],
                outputs=[self.classic_column_count, self.reduced_column_count, num_active_constraints],
                device=self.device,
            )

        wp.launch(
            _clear_contact_schedule_counts_kernel,
            dim=self.schedule_section_end.shape[0],
            inputs=[self.schedule_section_end],
            device=self.device,
        )
        self._skip_fallback_coloring = (
            self._fallback_impossible_partitioned if partition_ownership else self._fallback_impossible_rigid_only
        )
        self._schedule_already_grouped = (
            self._body_schedule_already_grouped if self._body_pair_grouping else self._shape_schedule_already_grouped
        )
        if self._schedule_already_grouped:
            wp.launch(
                _classify_grouped_reduced_contact_columns_kernel,
                dim=self.schedule_capacity,
                inputs=[
                    columns,
                    bodies,
                    num_columns,
                    self.classic_column_count,
                ],
                outputs=[self.schedule_columns, self.schedule_section_end],
                device=self.device,
            )
        else:
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
        if self._skip_fallback_coloring:
            self.fallback_count.zero_()
            return
        wp.launch(
            _compact_fallback_contact_elements_kernel,
            dim=self.schedule_capacity,
            inputs=[
                columns,
                bodies,
                self.reduced_column_count,
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
        if self._skip_fallback_coloring:
            return
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
                    dim=self.fallback_worker_count,
                    block_dim=_FALLBACK_BLOCK_DIM,
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
                        wp.int32(_FALLBACK_MAX_COLORED_PARTITIONS),
                        wp.int32(self.fallback_worker_count),
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
            self.deferred_active.zero_()
            self.max_page_count.zero_()
            self.multi_page_active.zero_()
            self.overflow_page_active.zero_()
            self.transpose_active.zero_()
            wp.launch(
                _count_reduced_contact_pages_kernel,
                dim=self.articulation_count,
                inputs=[
                    self.schedule_section_end,
                    self.schedule_columns,
                    columns,
                    bodies,
                ],
                outputs=[
                    self.enabled,
                    self.total_point_count,
                    self.max_page_count,
                    self.multi_page_active,
                    self.overflow_page_active,
                    self.transpose_active,
                    self.deferred_active,
                ],
                device=self.device,
            )

        def solve_resident_page(*, gathered: bool = False, build_rows: bool = True) -> None:
            assert self.packed_jacobian is not None
            assert self.packed_response is not None
            if not gathered:
                wp.launch(
                    _finalize_reduced_contact_rows_kernel,
                    dim=(self.articulation_count, _POINTS_PER_PAGE),
                    block_dim=_BLOCK_DIM,
                    inputs=[
                        columns,
                        bodies,
                        self.enabled,
                        self.point_count,
                        self.page_index,
                        self.point_column,
                        self.point0,
                        self.point1,
                        self.normal,
                        self.tangent0,
                    ],
                    outputs=[self.row_velocity],
                    device=self.device,
                )
            if build_rows:
                wp.launch(
                    _build_packed_generalized_contact_rows_kernel,
                    dim=(self.articulation_count, _MAX_ROWS),
                    block_dim=_MAX_ROWS,
                    inputs=[
                        bodies,
                        self.enabled,
                        self.point_count,
                        self.row_body,
                        self.row_wrench,
                        self.point_contact,
                        cc,
                        self.max_page_count,
                        self.page_index,
                        wp.bool(prepare),
                    ],
                    outputs=[
                        self.packed_jacobian,
                        self.packed_response,
                        self.aba_joint_work,
                        self.aba_body_response,
                    ],
                    device=self.device,
                )

                def transpose_response() -> None:
                    wp.launch_tiled(
                        _transpose_generalized_contact_response_kernel,
                        dim=[self.articulation_count * _RESPONSE_TILES_PER_ARTICULATION],
                        block_dim=_RESPONSE_TILE,
                        inputs=[
                            bodies,
                            self.enabled,
                            self.point_count,
                            self.page_index,
                            self.max_page_count,
                            wp.bool(prepare),
                            self.aba_joint_work,
                        ],
                        outputs=[self.packed_response],
                        device=self.device,
                    )

                wp.capture_if(self.transpose_active, on_true=transpose_response)
            wp.launch_tiled(
                self.solve_kernel,
                dim=[self.articulation_count],
                inputs=[
                    columns,
                    idt,
                    wp.float32(sor_boost),
                    cc,
                    wp.int32(iterations),
                    wp.bool(use_bias),
                    wp.bool(prepare),
                    self.enabled,
                    self.point_count,
                    self.point_contact,
                    self.point_column,
                    self.normal,
                    self.tangent0,
                    self.row_velocity,
                    self.page_index,
                    self.packed_jacobian,
                    self.packed_response,
                    bodies,
                    wp.bool(self.fuse_apply),
                    wp.int32(self.max_depth),
                    self.articulation_depth_start,
                    self.articulation_depth_joint,
                ],
                outputs=[self.generalized_delta, self.generalized_body_delta],
                block_dim=_BLOCK_DIM,
                device=self.device,
            )
            if not self.fuse_apply:
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

        def run_page_loop(*, gather: bool, build_rows: bool) -> None:
            wp.launch(
                _reset_reduced_contact_page_cursor_kernel,
                dim=1,
                inputs=[self.max_page_count],
                outputs=[self.page_cursor, self.page_index],
                device=self.device,
            )

            def solve_page() -> None:
                if gather:
                    wp.launch(
                        _gather_reduced_contact_blocks_kernel,
                        dim=(self.articulation_count, _POINTS_PER_PAGE),
                        block_dim=_BLOCK_DIM,
                        inputs=[
                            self.schedule_section_end,
                            self.schedule_columns,
                            columns,
                            bodies,
                            idt,
                            wp.bool(prepare),
                            cc,
                            contacts,
                            self.enabled,
                            self.total_point_count,
                            self.page_index,
                        ],
                        outputs=[
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
                        ],
                        device=self.device,
                    )
                solve_resident_page(gathered=gather, build_rows=build_rows)
                wp.launch(
                    _advance_reduced_contact_page_cursor_kernel,
                    dim=1,
                    inputs=[self.page_cursor, self.page_index],
                    device=self.device,
                )

            wp.capture_while(self.page_cursor, solve_page)

        if prepare:
            run_page_loop(gather=True, build_rows=True)
        else:

            def solve_single_cached_page() -> None:
                # Prepare leaves the cursor one past the last page.
                wp.launch(
                    _reset_reduced_contact_page_cursor_kernel,
                    dim=1,
                    inputs=[self.max_page_count],
                    outputs=[self.page_cursor, self.page_index],
                    device=self.device,
                )
                solve_resident_page(gathered=False, build_rows=False)

            def solve_cached_pages() -> None:
                wp.capture_if(
                    self.multi_page_active,
                    on_true=lambda: run_page_loop(gather=False, build_rows=False),
                    on_false=solve_single_cached_page,
                )

            wp.capture_if(
                self.overflow_page_active,
                on_true=lambda: run_page_loop(gather=True, build_rows=True),
                on_false=solve_cached_pages,
            )


__all__ = ["ReducedContactBlockSystem"]
