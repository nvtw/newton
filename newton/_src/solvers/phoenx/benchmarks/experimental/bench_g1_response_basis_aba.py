# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure exact unit-wrench ABA bases on the live reduced G1 contact state."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ContactColumnContainer,
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_normal_lambda,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    contact_get_friction,
    contact_get_friction_dynamic,
    contact_project_velocity_update_no_soft_pd,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_response_basis_synthesis import (
    _BASIS,
    _COLUMN_TILE,
    _DOFS,
    _PACKED_COLUMNS,
    _ROWS,
    _synthesize_tile_kernel,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.rl_training import g1_recipe

_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_BASIS_GROUP = 16


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_classify")
def _classify_basis_kernel(
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    basis_body: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
):
    articulation = wp.tid()
    packed_articulation = articulation * wp.int32(2)
    body0 = row_body[packed_articulation, 0]
    body1 = wp.int32(-1)
    for row_offset in range(_ROWS):
        body = row_body[packed_articulation, wp.int32(row_offset)]
        if body != body0:
            body1 = body
    basis_body[articulation, 0] = body0
    basis_body[articulation, 1] = body1
    coefficient_start = articulation * wp.int32(_ROWS)
    for row_offset in range(_ROWS):
        row = wp.int32(row_offset)
        for basis_offset in range(_BASIS):
            coefficients[coefficient_start + row, wp.int32(basis_offset)] = wp.float32(0.0)
        body = row_body[packed_articulation, row]
        slot = wp.int32(0) if body == body0 else wp.int32(1)
        wrench = row_wrench[packed_articulation, row]
        for component_offset in range(6):
            component = wp.int32(component_offset)
            coefficients[coefficient_start + row, slot * wp.int32(6) + component] = wrench[component]


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_classify_parallel")
def _classify_basis_bodies_kernel(
    row_body: wp.array2d[wp.int32],
    basis_body: wp.array2d[wp.int32],
):
    articulation = wp.tid()
    packed_articulation = articulation * wp.int32(2)
    body0 = row_body[packed_articulation, 0]
    body1 = wp.int32(-1)
    for row_offset in range(_ROWS):
        body = row_body[packed_articulation, wp.int32(row_offset)]
        if body != body0:
            body1 = body
    basis_body[articulation, 0] = body0
    basis_body[articulation, 1] = body1


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_coefficients")
def _build_basis_coefficients_kernel(
    row_body: wp.array2d[wp.int32],
    row_wrench: wp.array2d[wp.spatial_vector],
    basis_body: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
):
    articulation, row = wp.tid()
    packed_articulation = articulation * wp.int32(2)
    coefficient_row = articulation * wp.int32(_ROWS) + row
    for basis_offset in range(_BASIS):
        coefficients[coefficient_row, wp.int32(basis_offset)] = wp.float32(0.0)
    body = row_body[packed_articulation, row]
    slot = wp.int32(0) if body == basis_body[articulation, 0] else wp.int32(1)
    wrench = row_wrench[packed_articulation, row]
    for component_offset in range(6):
        component = wp.int32(component_offset)
        coefficients[coefficient_row, slot * wp.int32(6) + component] = wrench[component]


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_synthesis_split")
def _synthesize_split_tile_kernel(
    coefficients: wp.array2d[wp.float32],
    basis: wp.array2d[wp.float32],
    jacobian: wp.array2d[wp.float32],
    response: wp.array2d[wp.float32],
):
    articulation, output_kind = wp.tid()
    coefficients_tile = wp.tile_load(
        coefficients,
        shape=(_ROWS, _BASIS),
        offset=(articulation * _ROWS, 0),
        storage="shared",
    )
    basis_tile = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, output_kind * _DOFS),
        storage="shared",
    )
    result = wp.tile_matmul(coefficients_tile, basis_tile)
    packed_row = articulation * wp.int32(2 * 96)
    if output_kind == wp.int32(0):
        wp.tile_store(jacobian, result, offset=(packed_row, 0))
    else:
        wp.tile_store(response, result, offset=(packed_row, 0))


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_synthesis_split_fp16")
def _synthesize_split_tile_fp16_kernel(
    coefficients: wp.array2d[wp.float32],
    basis: wp.array2d[wp.float32],
    jacobian: wp.array2d[wp.float16],
    response: wp.array2d[wp.float16],
):
    articulation, output_kind = wp.tid()
    coefficients_tile = wp.tile_load(
        coefficients,
        shape=(_ROWS, _BASIS),
        offset=(articulation * _ROWS, 0),
        storage="shared",
    )
    basis_tile = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, output_kind * _DOFS),
        storage="shared",
    )
    result = wp.tile_astype(wp.tile_matmul(coefficients_tile, basis_tile), dtype=wp.float16)
    packed_row = articulation * wp.int32(2 * 96)
    if output_kind == wp.int32(0):
        wp.tile_store(jacobian, result, offset=(packed_row, 0))
    else:
        wp.tile_store(response, result, offset=(packed_row, 0))


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_effective")
def _build_basis_effective_matrix_kernel(
    basis: wp.array2d[wp.float32],
    effective: wp.array2d[wp.float32],
):
    articulation = wp.tid()
    jacobian = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, 0),
        storage="shared",
    )
    response = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, _DOFS),
        storage="shared",
    )
    wp.tile_store(
        effective,
        wp.tile_matmul(jacobian, wp.tile_transpose(response)),
        offset=(articulation * _BASIS, 0),
    )


@wp.kernel(enable_backward=False, module="experimental_g1_compact_patch_rows")
def _compact_first_point_patch_rows_kernel(
    point_count: wp.array[wp.int32],
    page_index: wp.array[wp.int32],
    point_column: wp.array2d[wp.int32],
    row_body_in: wp.array2d[wp.int32],
    row_wrench_in: wp.array2d[wp.spatial_vector],
    row_velocity_in: wp.array2d[wp.float32],
    row_count_out: wp.array[wp.int32],
    row_body_out: wp.array2d[wp.int32],
    row_wrench_out: wp.array2d[wp.spatial_vector],
    row_velocity_out: wp.array2d[wp.float32],
):
    articulation, lane = wp.tid()
    storage_page = wp.min(page_index[0], wp.int32(1))
    packed_articulation = articulation * wp.int32(2) + storage_page
    active_count = point_count[packed_articulation]

    column_count = wp.int32(0)
    if lane == wp.int32(0):
        for point_offset in range(active_count):
            point = wp.int32(point_offset)
            first_in_column = point == wp.int32(0)
            if point > wp.int32(0):
                first_in_column = (
                    point_column[packed_articulation, point] != point_column[packed_articulation, point - wp.int32(1)]
                )
            if first_in_column:
                column_count += wp.int32(1)
        row_count_out[packed_articulation] = active_count + wp.int32(2) * column_count

    if lane >= active_count:
        return

    source_row = wp.int32(3) * lane
    row_body_out[packed_articulation, lane] = row_body_in[packed_articulation, source_row]
    row_wrench_out[packed_articulation, lane] = row_wrench_in[packed_articulation, source_row]
    row_velocity_out[packed_articulation, lane] = row_velocity_in[packed_articulation, source_row]

    first_in_column = lane == wp.int32(0)
    if lane > wp.int32(0):
        first_in_column = (
            point_column[packed_articulation, lane] != point_column[packed_articulation, lane - wp.int32(1)]
        )
    if not first_in_column:
        return

    column_ordinal = wp.int32(0)
    for point_offset in range(lane):
        point = wp.int32(point_offset)
        previous_first = point == wp.int32(0)
        if point > wp.int32(0):
            previous_first = (
                point_column[packed_articulation, point] != point_column[packed_articulation, point - wp.int32(1)]
            )
        if previous_first:
            column_ordinal += wp.int32(1)
    target_row = active_count + wp.int32(2) * column_ordinal
    row_body_out[packed_articulation, target_row] = row_body_in[packed_articulation, source_row + wp.int32(1)]
    row_wrench_out[packed_articulation, target_row] = row_wrench_in[packed_articulation, source_row + wp.int32(1)]
    row_velocity_out[packed_articulation, target_row] = row_velocity_in[packed_articulation, source_row + wp.int32(1)]
    row_body_out[packed_articulation, target_row + wp.int32(1)] = row_body_in[
        packed_articulation, source_row + wp.int32(2)
    ]
    row_wrench_out[packed_articulation, target_row + wp.int32(1)] = row_wrench_in[
        packed_articulation, source_row + wp.int32(2)
    ]
    row_velocity_out[packed_articulation, target_row + wp.int32(1)] = row_velocity_in[
        packed_articulation, source_row + wp.int32(2)
    ]


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_solve")
def _solve_compressed_basis_contact_kernel(
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
    coefficients: wp.array2d[wp.float32],
    basis_effective: wp.array2d[wp.float32],
    basis_packed: wp.array2d[wp.float32],
    generalized_delta_out: wp.array2d[wp.float32],
):
    articulation, lane = wp.tid()
    if enabled[articulation] == wp.int32(0):
        return
    packed_articulation = articulation * wp.int32(2)
    active_point_count = point_count[packed_articulation]
    alpha = wp.tile_zeros(shape=_BASIS, dtype=wp.float32, storage="shared")
    effective = wp.tile_load(
        basis_effective,
        shape=(_BASIS, _BASIS),
        offset=(articulation * _BASIS, 0),
        storage="shared",
    )

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
            row = wp.int32(3) * point
            coefficient_row = articulation * wp.int32(_ROWS) + row
            coefficient0 = wp.tile_load(coefficients[coefficient_row], shape=_BASIS, storage="register")
            coefficient1 = wp.tile_load(coefficients[coefficient_row + wp.int32(1)], shape=_BASIS, storage="register")
            coefficient2 = wp.tile_load(coefficients[coefficient_row + wp.int32(2)], shape=_BASIS, storage="register")
            alpha += (
                wp.tile_from_thread(shape=_BASIS, value=lambda0, thread_idx=0, storage="shared") * coefficient0
                + wp.tile_from_thread(shape=_BASIS, value=lambda1, thread_idx=0, storage="shared") * coefficient1
                + wp.tile_from_thread(shape=_BASIS, value=lambda2, thread_idx=0, storage="shared") * coefficient2
            )

    basis_velocity = wp.tile_reshape(
        wp.tile_matmul(effective, wp.tile_reshape(alpha, shape=(_BASIS, 1))),
        shape=(_BASIS,),
    )
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
            row = wp.int32(3) * point
            coefficient_row = articulation * wp.int32(_ROWS) + row
            coefficient0 = wp.tile_load(coefficients[coefficient_row], shape=_BASIS, storage="register")
            coefficient1 = wp.tile_load(coefficients[coefficient_row + wp.int32(1)], shape=_BASIS, storage="register")
            coefficient2 = wp.tile_load(coefficients[coefficient_row + wp.int32(2)], shape=_BASIS, storage="register")
            jv0 = row_velocity[packed_articulation, row] + wp.tile_extract(
                wp.tile_sum(coefficient0 * basis_velocity), 0
            )
            jv1 = row_velocity[packed_articulation, row + wp.int32(1)] + wp.tile_extract(
                wp.tile_sum(coefficient1 * basis_velocity), 0
            )
            jv2 = row_velocity[packed_articulation, row + wp.int32(2)] + wp.tile_extract(
                wp.tile_sum(coefficient2 * basis_velocity), 0
            )
            delta0 = wp.float32(0.0)
            delta1 = wp.float32(0.0)
            delta2 = wp.float32(0.0)
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
            delta_alpha = (
                wp.tile_from_thread(shape=_BASIS, value=delta0, thread_idx=0, storage="shared") * coefficient0
                + wp.tile_from_thread(shape=_BASIS, value=delta1, thread_idx=0, storage="shared") * coefficient1
                + wp.tile_from_thread(shape=_BASIS, value=delta2, thread_idx=0, storage="shared") * coefficient2
            )
            alpha += delta_alpha
            basis_velocity += wp.tile_reshape(
                wp.tile_matmul(effective, wp.tile_reshape(delta_alpha, shape=(_BASIS, 1))),
                shape=(_BASIS,),
            )

    response = wp.tile_load(
        basis_packed,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, _DOFS),
        storage="shared",
    )
    generalized_delta = wp.tile_matmul(
        wp.tile_transpose(response),
        wp.tile_reshape(alpha, shape=(_BASIS, 1)),
    )
    wp.tile_store(
        generalized_delta_out,
        wp.tile_transpose(generalized_delta),
        offset=(articulation, 0),
    )


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_synthesis_split_96")
def _synthesize_split_tile_96_kernel(
    coefficients: wp.array2d[wp.float32],
    basis: wp.array2d[wp.float32],
    jacobian: wp.array2d[wp.float32],
    response: wp.array2d[wp.float32],
):
    articulation, output_kind = wp.tid()
    coefficients_tile = wp.tile_load(
        coefficients,
        shape=(96, _BASIS),
        offset=(articulation * wp.int32(96), 0),
        storage="shared",
    )
    basis_tile = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, output_kind * _DOFS),
        storage="shared",
    )
    result = wp.tile_matmul(coefficients_tile, basis_tile)
    packed_row = articulation * wp.int32(2 * 96)
    if output_kind == wp.int32(0):
        wp.tile_store(jacobian, result, offset=(packed_row, 0))
    else:
        wp.tile_store(response, result, offset=(packed_row, 0))


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_synthesis_split_96_fp16")
def _synthesize_split_tile_96_fp16_kernel(
    coefficients: wp.array2d[wp.float32],
    basis: wp.array2d[wp.float32],
    jacobian: wp.array2d[wp.float16],
    response: wp.array2d[wp.float16],
):
    articulation, output_kind = wp.tid()
    coefficients_tile = wp.tile_load(
        coefficients,
        shape=(96, _BASIS),
        offset=(articulation * wp.int32(96), 0),
        storage="shared",
    )
    basis_tile = wp.tile_load(
        basis,
        shape=(_BASIS, _DOFS),
        offset=(articulation * _BASIS, output_kind * _DOFS),
        storage="shared",
    )
    result = wp.tile_astype(wp.tile_matmul(coefficients_tile, basis_tile), dtype=wp.float16)
    packed_row = articulation * wp.int32(2 * 96)
    if output_kind == wp.int32(0):
        wp.tile_store(jacobian, result, offset=(packed_row, 0))
    else:
        wp.tile_store(response, result, offset=(packed_row, 0))


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis_scatter")
def _scatter_basis_kernel(
    synthesized: wp.array2d[wp.float32],
    jacobian: wp.array3d[wp.float32],
    response: wp.array3d[wp.float32],
):
    articulation, row, dof = wp.tid()
    synthesized_row = articulation * wp.int32(_ROWS) + row
    jacobian[articulation, row, dof] = synthesized[synthesized_row, dof]
    response[articulation, row, dof] = synthesized[synthesized_row, wp.int32(_DOFS) + dof]


@wp.kernel(enable_backward=False, module="experimental_g1_contact_basis")
def _build_unit_wrench_basis_kernel(
    bodies: BodyContainer,
    basis_body: wp.array2d[wp.int32],
    basis_packed: wp.array2d[wp.float32],
    joint_work: wp.array3d[wp.float32],
    body_response: wp.array3d[wp.spatial_vector],
):
    index = wp.tid()
    articulation = index // wp.int32(_BASIS_GROUP)
    basis_row = index - articulation * wp.int32(_BASIS_GROUP)
    if basis_row >= wp.int32(_BASIS):
        return
    body_slot = basis_row // wp.int32(6)
    component = basis_row - body_slot * wp.int32(6)
    source_body = basis_body[articulation, body_slot]
    if source_body < wp.int32(0):
        return
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    dof_start_articulation = data.joint_qd_start[start]
    dof_end_articulation = data.joint_qd_start[end]
    dof_count_articulation = dof_end_articulation - dof_start_articulation
    packed_basis_row = articulation * wp.int32(_BASIS) + basis_row
    for local_dof in range(dof_count_articulation):
        basis_packed[packed_basis_row, local_dof] = wp.float32(0.0)
        basis_packed[packed_basis_row, wp.int32(_DOFS) + local_dof] = wp.float32(0.0)
        joint_work[articulation, local_dof, basis_row] = wp.float32(0.0)

    source_wrench = wp.spatial_vector()
    source_wrench[component] = wp.float32(1.0)
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
                joint_work[articulation, dof - dof_start_articulation, basis_row] = projected[dof_row]
                basis_packed[packed_basis_row, dof - dof_start_articulation] = wp.dot(data.joint_s[dof], source_wrench)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        reduced[dof_row] += (
                            data.joint_d_inv[dof_start + wp.int32(dof_row), dof_column] * projected[dof_column]
                        )
                propagated_wrench -= data.joint_u[dof_start + wp.int32(dof_row)] * reduced[dof_row]

    for joint in range(start, end):
        local_joint = joint - start
        parent = data.joint_parent[joint]
        parent_delta = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_delta = body_response[articulation, data.body_joint[parent] - start, basis_row]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        rhs = _vec6(0.0)
        generalized_delta = _vec6(0.0)
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                dof = dof_start + wp.int32(dof_row)
                rhs[dof_row] = joint_work[articulation, dof - dof_start_articulation, basis_row] - wp.dot(
                    data.joint_u[dof], parent_delta
                )
        for dof_row in range(6):
            if wp.int32(dof_row) < dof_count:
                for dof_column in range(6):
                    if wp.int32(dof_column) < dof_count:
                        generalized_delta[dof_row] += (
                            data.joint_d_inv[dof_start + wp.int32(dof_row), dof_column] * rhs[dof_column]
                        )
                dof = dof_start + wp.int32(dof_row)
                basis_packed[packed_basis_row, wp.int32(_DOFS) + dof - dof_start_articulation] = generalized_delta[
                    dof_row
                ]
                parent_delta += data.joint_s[dof] * generalized_delta[dof_row]
        body_response[articulation, local_joint, basis_row] = parent_delta


def _time_graph(device: wp.context.Device, launch: Callable[[], None], replays: int) -> float:
    with wp.ScopedCapture(device=device) as capture:
        launch()
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(replays):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    return 1.0e6 * (time.perf_counter() - start) / float(replays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--basis-block-dim", type=int, default=96)
    parser.add_argument("--basis-launch-block-dim", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("live G1 response-basis benchmark requires CUDA")

    env = rl.EnvG1PhoenX(
        rl.ConfigEnvG1PhoenX(
            world_count=int(args.world_count),
            articulation_mode="reduced",
            sim_substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            contact_geometry=g1_recipe.CONTACT_GEOMETRY,
        ),
        device=device,
    )
    actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
    env.step(actions)
    wp.synchronize_device(device)
    reduced = env.solver._reduced_articulation
    if reduced is None:
        raise RuntimeError("reduced articulation was not initialized")
    block = reduced.contact_block_system
    if block.packed_jacobian is None or block.packed_response is None:
        raise RuntimeError("reduced contact buffers were not initialized")
    block.page_index.assign(np.asarray([0], dtype=np.int32))

    point_count = block.point_count.numpy()[::2]
    row_body = block.row_body.numpy()[::2]
    row_wrench = block.row_wrench.numpy()[::2]
    if np.any(point_count != 8):
        raise RuntimeError(f"expected the common eight-point G1 page, got {np.unique(point_count).tolist()}")
    basis_body_np = np.full((env.world_count, 2), -1, dtype=np.int32)
    coefficients_np = np.zeros((env.world_count, _ROWS, _BASIS), dtype=np.float32)
    for articulation in range(env.world_count):
        bodies = np.unique(row_body[articulation, :_ROWS])
        if bodies.size != 2:
            raise RuntimeError(f"articulation {articulation} has {bodies.size} contacted bodies")
        basis_body_np[articulation] = bodies
        for row in range(_ROWS):
            slot = int(np.nonzero(bodies == row_body[articulation, row])[0][0])
            coefficients_np[articulation, row, slot * 6 : slot * 6 + 6] = row_wrench[articulation, row]

    basis_body = wp.empty((env.world_count, 2), dtype=wp.int32, device=device)
    coefficients = wp.empty((env.world_count * _ROWS, _BASIS), dtype=wp.float32, device=device)
    coefficients_96 = wp.zeros((env.world_count * 96, _BASIS), dtype=wp.float32, device=device)
    compact_row_count = wp.zeros_like(block.row_count)
    compact_row_body = wp.zeros_like(block.row_body)
    compact_row_wrench = wp.zeros_like(block.row_wrench)
    compact_row_velocity = wp.zeros_like(block.row_velocity)
    compact_previous_row_body = wp.full(block.packed_previous_row_body.shape, value=-1, dtype=wp.int32, device=device)
    basis_packed = wp.zeros((env.world_count * _BASIS, _PACKED_COLUMNS), dtype=wp.float32, device=device)
    basis_effective = wp.zeros((env.world_count * _BASIS, _BASIS), dtype=wp.float32, device=device)
    compressed_generalized_delta = wp.zeros((env.world_count, _DOFS), dtype=wp.float32, device=device)
    source_cc = env.solver.world._contact_container
    direct_cc = ContactContainer()
    compressed_cc = ContactContainer()
    for target in (direct_cc, compressed_cc):
        target.impulses = wp.clone(source_cc.impulses)
        target.prev_impulses = wp.clone(source_cc.prev_impulses)
        target.lambdas = wp.clone(source_cc.lambdas)
        target.prev_lambdas = wp.clone(source_cc.prev_lambdas)
        target.derived = wp.clone(source_cc.derived)
    coefficients_96_np = np.zeros((env.world_count, 96, _BASIS), dtype=np.float32)
    coefficients_96_np[:, :_ROWS] = coefficients_np
    coefficients_96.assign(coefficients_96_np.reshape(-1, _BASIS))
    synthesized = wp.zeros((env.world_count * _ROWS, _PACKED_COLUMNS), dtype=wp.float32, device=device)
    scattered_jacobian = wp.empty((env.world_count, _ROWS, _DOFS), dtype=wp.float32, device=device)
    scattered_response = wp.empty_like(scattered_jacobian)

    def launch_classify() -> None:
        wp.launch(
            _classify_basis_kernel,
            dim=env.world_count,
            inputs=[block.row_body, block.row_wrench],
            outputs=[basis_body, coefficients],
            device=device,
        )

    def launch_classify_parallel() -> None:
        wp.launch(
            _classify_basis_bodies_kernel,
            dim=env.world_count,
            inputs=[block.row_body],
            outputs=[basis_body],
            device=device,
        )
        wp.launch(
            _build_basis_coefficients_kernel,
            dim=(env.world_count, _ROWS),
            inputs=[block.row_body, block.row_wrench, basis_body],
            outputs=[coefficients],
            device=device,
        )

    def launch_compact_patch_rows() -> None:
        wp.launch(
            _compact_first_point_patch_rows_kernel,
            dim=(env.world_count, 32),
            block_dim=32,
            inputs=[
                block.point_count,
                block.page_index,
                block.point_column,
                block.row_body,
                block.row_wrench,
                block.row_velocity,
            ],
            outputs=[compact_row_count, compact_row_body, compact_row_wrench, compact_row_velocity],
            device=device,
        )

    def launch_compact_direct() -> None:
        launch_compact_patch_rows()
        wp.launch(
            block.build_rows_kernel,
            dim=(env.world_count, 96),
            block_dim=int(args.basis_block_dim),
            inputs=[
                env.solver.world.bodies,
                block.enabled,
                block.point_count,
                compact_row_count,
                compact_row_body,
                compact_row_wrench,
                block.point_contact,
                env.solver.world._contact_container,
                block.max_page_count,
                block.page_index,
                wp.bool(True),
                compact_previous_row_body,
            ],
            outputs=[
                block.packed_jacobian,
                block.packed_response,
                block.aba_joint_work,
                block.aba_body_response,
            ],
            device=device,
        )

    def launch_direct() -> None:
        wp.launch(
            block.build_rows_kernel,
            dim=(env.world_count, 96),
            block_dim=int(args.basis_block_dim),
            inputs=[
                env.solver.world.bodies,
                block.enabled,
                block.point_count,
                block.row_count,
                block.row_body,
                block.row_wrench,
                block.point_contact,
                env.solver.world._contact_container,
                block.max_page_count,
                block.page_index,
                wp.bool(True),
                block.packed_previous_row_body,
            ],
            outputs=[
                block.packed_jacobian,
                block.packed_response,
                block.aba_joint_work,
                block.aba_body_response,
            ],
            device=device,
        )

    def launch_basis() -> None:
        wp.launch(
            _build_unit_wrench_basis_kernel,
            dim=env.world_count * _BASIS_GROUP,
            block_dim=int(args.basis_launch_block_dim),
            inputs=[env.solver.world.bodies, basis_body],
            outputs=[basis_packed, block.aba_joint_work, block.aba_body_response],
            device=device,
        )

    def launch_synthesis() -> None:
        wp.launch_tiled(
            _synthesize_tile_kernel,
            dim=[env.world_count, _PACKED_COLUMNS // _COLUMN_TILE],
            block_dim=64,
            inputs=[coefficients, basis_packed],
            outputs=[synthesized],
            device=device,
        )

    def launch_basis_effective() -> None:
        wp.launch_tiled(
            _build_basis_effective_matrix_kernel,
            dim=[env.world_count],
            block_dim=64,
            inputs=[basis_packed],
            outputs=[basis_effective],
            device=device,
        )

    def launch_compressed_solve() -> None:
        wp.launch_tiled(
            _solve_compressed_basis_contact_kernel,
            dim=[env.world_count],
            block_dim=32,
            inputs=[
                env.solver.world._contact_cols,
                wp.float32(250.0),
                wp.float32(env.solver.world.sor_boost),
                compressed_cc,
                wp.int32(2),
                wp.bool(True),
                wp.bool(True),
                block.enabled,
                block.point_count,
                block.point_contact,
                block.point_column,
                block.normal,
                block.tangent0,
                block.row_velocity,
                coefficients,
                basis_effective,
                basis_packed,
            ],
            outputs=[compressed_generalized_delta],
            device=device,
        )

    def launch_direct_solve() -> None:
        wp.launch_tiled(
            block.solve_kernel,
            dim=[env.world_count],
            block_dim=32,
            inputs=[
                env.solver.world._contact_cols,
                wp.float32(250.0),
                wp.float32(env.solver.world.sor_boost),
                direct_cc,
                wp.int32(2),
                wp.bool(True),
                wp.bool(True),
                block.enabled,
                block.point_count,
                block.point_contact,
                block.point_column,
                block.normal,
                block.tangent0,
                block.row_velocity,
                block.page_index,
                block.packed_jacobian_solve,
                block.packed_response_solve,
                env.solver.world.bodies,
                wp.bool(False),
                wp.int32(block.max_depth),
                block.articulation_depth_start,
                block.articulation_depth_joint,
            ],
            outputs=[block.generalized_delta_solve, block.generalized_body_delta],
            device=device,
        )

    def launch_synthesis_split_96() -> None:
        kernel = _synthesize_split_tile_96_fp16_kernel if block.packed_rows_fp16 else _synthesize_split_tile_96_kernel
        wp.launch_tiled(
            kernel,
            dim=[env.world_count, 2],
            block_dim=128,
            inputs=[coefficients_96, basis_packed],
            outputs=[block.packed_jacobian, block.packed_response],
            device=device,
        )

    def launch_scatter() -> None:
        wp.launch(
            _scatter_basis_kernel,
            dim=(env.world_count, _ROWS, _DOFS),
            inputs=[synthesized],
            outputs=[scattered_jacobian, scattered_response],
            device=device,
        )

    def launch_synthesis_split() -> None:
        kernel = _synthesize_split_tile_fp16_kernel if block.packed_rows_fp16 else _synthesize_split_tile_kernel
        wp.launch_tiled(
            kernel,
            dim=[env.world_count, 2],
            block_dim=64,
            inputs=[coefficients, basis_packed],
            outputs=[block.packed_jacobian, block.packed_response],
            device=device,
        )

    launch_classify()
    wp.synchronize_device(device)
    np.testing.assert_array_equal(basis_body.numpy(), basis_body_np)
    np.testing.assert_allclose(coefficients.numpy(), coefficients_np.reshape(-1, _BASIS), rtol=0.0, atol=0.0)
    launch_compact_patch_rows()
    wp.synchronize_device(device)
    compact_row_count_np = compact_row_count.numpy()[::2]
    compact_rows_mean = float(np.mean(compact_row_count_np))
    compact_rows_max = int(np.max(compact_row_count_np))
    launch_direct()
    wp.synchronize_device(device)
    direct_jacobian = block.packed_jacobian.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    direct_response = block.packed_response.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    launch_basis()
    launch_basis_effective()
    wp.synchronize_device(device)
    basis_effective_np = basis_effective.numpy().reshape(env.world_count, _BASIS, _BASIS)
    direct_effective = direct_jacobian @ np.swapaxes(direct_response, 1, 2)
    compressed_effective = coefficients_np @ basis_effective_np @ np.swapaxes(coefficients_np, 1, 2)
    effective_error = float(np.max(np.abs(compressed_effective - direct_effective)))
    launch_synthesis()
    launch_scatter()
    wp.synchronize_device(device)
    synthesized_np = synthesized.numpy().reshape(env.world_count, _ROWS, _PACKED_COLUMNS)
    jacobian_error = float(np.max(np.abs(synthesized_np[:, :, :_DOFS] - direct_jacobian)))
    response_error = float(np.max(np.abs(synthesized_np[:, :, _DOFS:] - direct_response)))
    launch_synthesis_split()
    wp.synchronize_device(device)
    split_jacobian = block.packed_jacobian.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    split_response = block.packed_response.numpy().reshape(env.world_count * 2, 96, _DOFS)[::2, :_ROWS]
    split_jacobian_error = float(np.max(np.abs(split_jacobian - direct_jacobian)))
    split_response_error = float(np.max(np.abs(split_response - direct_response)))
    launch_direct()
    launch_direct_solve()
    launch_compressed_solve()
    wp.synchronize_device(device)
    direct_generalized_delta = block.generalized_delta.numpy()[:, :_DOFS]
    compressed_generalized_delta_np = compressed_generalized_delta.numpy()
    generalized_delta_error = float(np.max(np.abs(compressed_generalized_delta_np - direct_generalized_delta)))
    impulse_error = float(np.max(np.abs(compressed_cc.impulses.numpy() - direct_cc.impulses.numpy())))
    point_contact_np = block.point_contact.numpy()[::2, : _ROWS // 3]
    direct_impulses_np = direct_cc.impulses.numpy()
    row_impulses = np.empty((env.world_count, _ROWS), dtype=np.float32)
    for axis in range(3):
        row_impulses[:, axis :: _ROWS // (_ROWS // 3)] = direct_impulses_np[axis, point_contact_np]
    cpu_direct_generalized_delta = np.einsum("wr,wrd->wd", row_impulses, direct_response)
    cpu_alpha = np.einsum("wr,wrb->wb", row_impulses, coefficients_np)
    basis_response_np = basis_packed.numpy().reshape(env.world_count, _BASIS, _PACKED_COLUMNS)[:, :, _DOFS:]
    cpu_compressed_generalized_delta = np.einsum("wb,wbd->wd", cpu_alpha, basis_response_np)
    direct_kernel_cpu_error = float(np.max(np.abs(direct_generalized_delta - cpu_direct_generalized_delta)))
    compressed_kernel_cpu_error = float(
        np.max(np.abs(compressed_generalized_delta_np - cpu_compressed_generalized_delta))
    )
    cpu_expansion_error = float(np.max(np.abs(cpu_compressed_generalized_delta - cpu_direct_generalized_delta)))
    direct_us = _time_graph(device, launch_direct, int(args.replays))
    compact_us = _time_graph(device, launch_compact_patch_rows, int(args.replays))
    compact_direct_us = _time_graph(device, launch_compact_direct, int(args.replays))
    direct_solve_us = _time_graph(device, launch_direct_solve, int(args.replays))
    classify_us = _time_graph(device, launch_classify, int(args.replays))
    classify_parallel_us = _time_graph(device, launch_classify_parallel, int(args.replays))
    basis_us = _time_graph(device, launch_basis, int(args.replays))
    basis_effective_us = _time_graph(device, launch_basis_effective, int(args.replays))
    compressed_solve_us = _time_graph(device, launch_compressed_solve, int(args.replays))
    synthesis_us = _time_graph(device, launch_synthesis, int(args.replays))
    synthesis_split_us = _time_graph(device, launch_synthesis_split, int(args.replays))
    synthesis_split_96_us = _time_graph(device, launch_synthesis_split_96, int(args.replays))
    scatter_us = _time_graph(device, launch_scatter, int(args.replays))
    adaptive_total_us = classify_us + basis_us + synthesis_us + scatter_us
    adaptive_parallel_total_us = classify_parallel_us + basis_us + synthesis_split_us
    compressed_prepare_total_us = classify_parallel_us + basis_us + basis_effective_us
    compressed_total_us = compressed_prepare_total_us + compressed_solve_us
    direct_total_us = direct_us + direct_solve_us
    print(
        json.dumps(
            {
                "basis_block_dim": int(args.basis_block_dim),
                "basis_launch_block_dim": int(args.basis_launch_block_dim),
                "basis_build_us": basis_us,
                "basis_effective_us": basis_effective_us,
                "adaptive_parallel_projected_speedup": direct_us / adaptive_parallel_total_us,
                "adaptive_parallel_total_us": adaptive_parallel_total_us,
                "adaptive_projected_speedup": direct_us / adaptive_total_us,
                "adaptive_total_us": adaptive_total_us,
                "basis_plus_synthesis_us": basis_us + synthesis_us,
                "classify_parallel_us": classify_parallel_us,
                "classify_us": classify_us,
                "compressed_effective_max_abs_error": effective_error,
                "compressed_generalized_delta_max_abs_error": generalized_delta_error,
                "compressed_impulse_max_abs_error": impulse_error,
                "compressed_kernel_cpu_expansion_max_abs_error": compressed_kernel_cpu_error,
                "compressed_prepare_projected_speedup": direct_us / compressed_prepare_total_us,
                "compressed_prepare_total_us": compressed_prepare_total_us,
                "compressed_solve_us": compressed_solve_us,
                "compressed_total_us": compressed_total_us,
                "direct_us": direct_us,
                "compact_first_point_patch_rows_mean": compact_rows_mean,
                "compact_first_point_patch_rows_max": compact_rows_max,
                "compact_first_point_patch_compact_us": compact_us,
                "compact_first_point_patch_direct_us": compact_direct_us,
                "compact_first_point_patch_projected_build_speedup": direct_us / compact_direct_us,
                "direct_solve_us": direct_solve_us,
                "direct_kernel_cpu_expansion_max_abs_error": direct_kernel_cpu_error,
                "direct_total_us": direct_total_us,
                "compressed_total_projected_speedup": direct_total_us / compressed_total_us,
                "jacobian_max_abs_error": jacobian_error,
                "projected_speedup": direct_us / (basis_us + synthesis_us),
                "response_max_abs_error": response_error,
                "cpu_expansion_max_abs_error": cpu_expansion_error,
                "scatter_us": scatter_us,
                "split_jacobian_max_abs_error": split_jacobian_error,
                "split_response_max_abs_error": split_response_error,
                "synthesis_split_96_us": synthesis_split_96_us,
                "synthesis_split_us": synthesis_split_us,
                "synthesis_us": synthesis_us,
                "world_count": int(args.world_count),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
