# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent scalar-row assembly for rigid contacts."""

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_friction,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.simple.rows import ScalarRowContainer

__all__ = [
    "CONTACT_ROW_STRIDE",
    "assemble_contact_scalar_rows_kernel",
    "writeback_contact_lambdas_kernel",
]


CONTACT_ROW_STRIDE = 3


@wp.kernel(enable_backward=False)
def assemble_contact_scalar_rows_kernel(
    columns: ContactColumnContainer,
    contacts: ContactViews,
    cc: ContactContainer,
    cid_of_contact: wp.array[wp.int32],
    contact_cid_offset: wp.int32,
    bodies: BodyContainer,
    row_offset: wp.int32,
    idt: wp.float32,
    rows: ScalarRowContainer,
    body_split_count: wp.array[wp.int32],
):
    """Assemble one normal or tangent equation per thread."""
    local_row = wp.tid()
    contact = local_row // wp.int32(CONTACT_ROW_STRIDE)
    axis_index = local_row - contact * wp.int32(CONTACT_ROW_STRIDE)
    row = row_offset + local_row
    rows.active[row] = wp.int32(0)
    rows.split_anchor[row] = wp.int32(0)
    if axis_index == wp.int32(0):
        rows.split_anchor[row] = wp.int32(1)
    if contact >= contacts.rigid_contact_count[0]:
        return

    column = cid_of_contact[contact] - contact_cid_offset
    if column < wp.int32(0):
        return
    body_a = contact_get_body1(columns, column)
    body_b = contact_get_body2(columns, column)
    if axis_index == wp.int32(0):
        wp.atomic_add(body_split_count, body_a, wp.int32(1))
        if body_b != body_a:
            wp.atomic_add(body_split_count, body_b, wp.int32(1))
    normal = cc_get_normal(cc, contact)
    tangent1 = cc_get_tangent1(cc, contact)
    direction = normal
    if axis_index == wp.int32(1):
        direction = tangent1
    elif axis_index == wp.int32(2):
        direction = wp.cross(normal, tangent1)

    local_p0 = cc_get_local_p0(cc, contact)
    local_p1 = cc_get_local_p1(cc, contact)
    arm_a = wp.quat_rotate(bodies.orientation[body_a], local_p0 - bodies.body_com[body_a])
    arm_b = wp.quat_rotate(bodies.orientation[body_b], local_p1 - bodies.body_com[body_b])
    point_a = bodies.position[body_a] + arm_a + contacts.rigid_contact_margin0[contact] * normal
    point_b = bodies.position[body_b] + arm_b - contacts.rigid_contact_margin1[contact] * normal

    rows.active[row] = wp.int32(1)
    rows.body_a[row] = body_a
    rows.body_b[row] = body_b
    rows.jacobian_linear_a[row] = -direction
    rows.jacobian_angular_a[row] = -wp.cross(arm_a, direction)
    rows.jacobian_linear_b[row] = direction
    rows.jacobian_angular_b[row] = wp.cross(arm_b, direction)
    rows.softness[row] = wp.float32(0.0)
    rows.multiplier[row] = cc_get_normal_lambda(cc, contact)
    rows.bound_row[row] = row
    rows.bound_scale[row] = wp.float32(0.0)

    if axis_index == wp.int32(0):
        gap = wp.dot(point_b - point_a, normal)
        rows.bias[row] = wp.float32(0.2) * idt * gap
        rows.lower[row] = wp.float32(0.0)
        rows.upper[row] = wp.float32(1.0e30)
    else:
        if axis_index == wp.int32(1):
            rows.multiplier[row] = cc_get_tangent1_lambda(cc, contact)
        else:
            rows.multiplier[row] = cc_get_tangent2_lambda(cc, contact)
        rows.bias[row] = wp.float32(0.0)
        rows.lower[row] = wp.float32(0.0)
        rows.upper[row] = wp.float32(0.0)
        rows.bound_row[row] = row_offset + contact * wp.int32(CONTACT_ROW_STRIDE)
        rows.bound_scale[row] = contact_get_friction(columns, column)


@wp.kernel(enable_backward=False)
def writeback_contact_lambdas_kernel(
    contact_count: wp.array[wp.int32],
    row_offset: wp.int32,
    rows: ScalarRowContainer,
    cc: ContactContainer,
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    row = row_offset + contact * wp.int32(CONTACT_ROW_STRIDE)
    cc_set_normal_lambda(cc, contact, rows.multiplier[row])
    cc_set_tangent1_lambda(cc, contact, rows.multiplier[row + wp.int32(1)])
    cc_set_tangent2_lambda(cc, contact, rows.multiplier[row + wp.int32(2)])
