# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact contacted-link response bases for reduced articulation contacts."""

from __future__ import annotations

import functools

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
)

BASIS_BODY_CAPACITY = 2
BASIS_SIZE = 6 * BASIS_BODY_CAPACITY
BASIS_LAUNCH_WIDTH = 16
MAX_ROWS = 96
CACHED_PAGE_COUNT = 2

_vec6 = wp.types.vector(length=6, dtype=wp.float32)


@functools.cache
def make_reduced_contact_basis_build_ops(max_dofs: int):
    """Create exact basis build/effective-matrix kernels for one DOF width."""
    module = wp.get_module(f"reduced_contact_basis_build_{max_dofs}")

    @wp.func
    def build_basis_device(
        articulation: wp.int32,
        basis_row: wp.int32,
        bodies: BodyContainer,
        basis_enabled: wp.array[wp.int32],
        basis_body: wp.array2d[wp.int32],
        basis_packed: wp.array2d[wp.float32],
        joint_work: wp.array3d[wp.float32],
        body_response: wp.array3d[wp.spatial_vector],
    ):
        if basis_enabled[articulation] == wp.int32(0) or basis_row >= wp.int32(BASIS_SIZE):
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
        dof_count_articulation = data.joint_qd_start[end] - dof_start_articulation
        packed_basis_row = articulation * wp.int32(BASIS_SIZE) + basis_row
        for local_dof in range(dof_count_articulation):
            basis_packed[packed_basis_row, local_dof] = wp.float32(0.0)
            basis_packed[packed_basis_row, wp.int32(max_dofs) + local_dof] = wp.float32(0.0)
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
                    basis_packed[packed_basis_row, dof - dof_start_articulation] = wp.dot(
                        data.joint_s[dof], source_wrench
                    )
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
                    basis_packed[packed_basis_row, wp.int32(max_dofs) + dof - dof_start_articulation] = (
                        generalized_delta[dof_row]
                    )
                    parent_delta += data.joint_s[dof] * generalized_delta[dof_row]
            body_response[articulation, local_joint, basis_row] = parent_delta

    @wp.kernel(enable_backward=False, module=module)
    def build_basis_kernel(
        bodies: BodyContainer,
        basis_enabled: wp.array[wp.int32],
        basis_body: wp.array2d[wp.int32],
        basis_packed: wp.array2d[wp.float32],
        joint_work: wp.array3d[wp.float32],
        body_response: wp.array3d[wp.spatial_vector],
    ):
        index = wp.tid()
        articulation = index // wp.int32(BASIS_LAUNCH_WIDTH)
        basis_row = index - articulation * wp.int32(BASIS_LAUNCH_WIDTH)
        build_basis_device(
            articulation, basis_row, bodies, basis_enabled, basis_body, basis_packed, joint_work, body_response
        )

    @wp.kernel(enable_backward=False, module=module)
    def build_effective_kernel(
        basis_enabled: wp.array[wp.int32],
        basis_packed: wp.array2d[wp.float32],
        basis_effective: wp.array2d[wp.float32],
    ):
        articulation, element = wp.tid()
        if basis_enabled[articulation] == wp.int32(0):
            return
        row = element / wp.int32(BASIS_SIZE)
        column = element - row * wp.int32(BASIS_SIZE)
        value = wp.float32(0.0)
        jacobian_row = articulation * wp.int32(BASIS_SIZE) + row
        response_row = articulation * wp.int32(BASIS_SIZE) + column
        for dof in range(max_dofs):
            value += basis_packed[jacobian_row, dof] * basis_packed[response_row, wp.int32(max_dofs) + dof]
        basis_effective[jacobian_row, column] = value

    @wp.kernel(enable_backward=False, module=module)
    def build_coefficients_kernel(
        basis_enabled: wp.array[wp.int32],
        basis_body: wp.array2d[wp.int32],
        point_count: wp.array[wp.int32],
        page_index: wp.array[wp.int32],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        coefficients: wp.array2d[wp.float32],
    ):
        storage_page = wp.min(page_index[0], wp.int32(CACHED_PAGE_COUNT - 1))
        packed_articulation = articulation * wp.int32(CACHED_PAGE_COUNT) + storage_page
        if basis_enabled[articulation] == wp.int32(0) or row >= wp.int32(3) * point_count[packed_articulation]:
            return
        coefficient_row = packed_articulation * wp.int32(MAX_ROWS) + row
        for basis_column in range(BASIS_SIZE):
            coefficients[coefficient_row, basis_column] = wp.float32(0.0)
        body = row_body[packed_articulation, row]
        slot = wp.int32(0) if body == basis_body[articulation, 0] else wp.int32(1)
        wrench = row_wrench[packed_articulation, row]
        for component_offset in range(6):
            component = wp.int32(component_offset)
            coefficients[coefficient_row, slot * wp.int32(6) + component] = wrench[component]

    @wp.func
    def synthesize_rows_device(
        articulation: wp.int32,
        row: wp.int32,
        basis_enabled: wp.array[wp.int32],
        basis_body: wp.array2d[wp.int32],
        point_count: wp.array[wp.int32],
        page_index: wp.array[wp.int32],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        point_contact: wp.array2d[wp.int32],
        cc: ContactContainer,
        basis_packed: wp.array2d[wp.float32],
        packed_jacobian: wp.array2d[wp.float32],
        packed_response: wp.array2d[wp.float32],
    ):
        storage_page = wp.min(page_index[0], wp.int32(CACHED_PAGE_COUNT - 1))
        packed_articulation = articulation * wp.int32(CACHED_PAGE_COUNT) + storage_page
        row_count = wp.int32(3) * point_count[packed_articulation]
        if basis_enabled[articulation] == wp.int32(0) or row >= row_count:
            return
        body = row_body[packed_articulation, row]
        slot = wp.int32(0) if body == basis_body[articulation, 0] else wp.int32(1)
        wrench = row_wrench[packed_articulation, row]
        packed_row = packed_articulation * wp.int32(MAX_ROWS) + row
        basis_start = articulation * wp.int32(BASIS_SIZE) + slot * wp.int32(6)
        inverse_mass = wp.float32(0.0)
        for dof in range(max_dofs):
            jacobian_value = wp.float32(0.0)
            response_value = wp.float32(0.0)
            for component_offset in range(6):
                component = wp.int32(component_offset)
                coefficient = wrench[component]
                basis_row = basis_start + component
                jacobian_value += coefficient * basis_packed[basis_row, dof]
                response_value += coefficient * basis_packed[basis_row, wp.int32(max_dofs) + dof]
            packed_jacobian[packed_row, dof] = jacobian_value
            packed_response[packed_row, dof] = response_value
            inverse_mass += jacobian_value * response_value
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

    @wp.kernel(enable_backward=False, module=module)
    def synthesize_rows_kernel(
        basis_enabled: wp.array[wp.int32],
        basis_body: wp.array2d[wp.int32],
        point_count: wp.array[wp.int32],
        page_index: wp.array[wp.int32],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        point_contact: wp.array2d[wp.int32],
        cc: ContactContainer,
        basis_packed: wp.array2d[wp.float32],
        packed_jacobian: wp.array2d[wp.float32],
        packed_response: wp.array2d[wp.float32],
    ):
        articulation, row = wp.tid()
        synthesize_rows_device(
            articulation,
            row,
            basis_enabled,
            basis_body,
            point_count,
            page_index,
            row_body,
            row_wrench,
            point_contact,
            cc,
            basis_packed,
            packed_jacobian,
            packed_response,
        )

    return (
        build_basis_device,
        build_basis_kernel,
        build_effective_kernel,
        build_coefficients_kernel,
        synthesize_rows_device,
        synthesize_rows_kernel,
    )


__all__ = [
    "BASIS_BODY_CAPACITY",
    "BASIS_LAUNCH_WIDTH",
    "BASIS_SIZE",
    "make_reduced_contact_basis_build_ops",
]
