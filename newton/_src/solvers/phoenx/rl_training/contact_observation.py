# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph-capturable rigid-foot contact observations shared by PhoenX RL tasks."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer, cc_get_normal_lambda


@wp.func
def mark_two_body_contact(
    shape_id: wp.int32,
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    body_stride: wp.int32,
    body0: wp.int32,
    body1: wp.int32,
    contacts: wp.array2d[wp.float32],
):
    if shape_id < wp.int32(0):
        return
    body = shape_body[shape_id]
    if body < wp.int32(0):
        return
    world = shape_world[shape_id]
    if world < wp.int32(0):
        world = body / body_stride
    if world < wp.int32(0) or world >= contacts.shape[0]:
        return
    local_body = body - world * body_stride
    if local_body == body0:
        wp.atomic_add(contacts, world, wp.int32(0), wp.float32(1.0))
    elif local_body == body1:
        wp.atomic_add(contacts, world, wp.int32(1), wp.float32(1.0))


@wp.kernel(enable_backward=False)
def scan_two_body_contacts_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    contact_container: ContactContainer,
    body_stride: wp.int32,
    body0: wp.int32,
    body1: wp.int32,
    normal_impulse_threshold: wp.float32,
    contacts: wp.array2d[wp.float32],
):
    contact = wp.tid()
    count = wp.min(rigid_contact_count[0], rigid_contact_shape0.shape[0])
    if contact >= count:
        return
    if wp.max(cc_get_normal_lambda(contact_container, contact), wp.float32(0.0)) <= normal_impulse_threshold:
        return
    mark_two_body_contact(rigid_contact_shape0[contact], shape_body, shape_world, body_stride, body0, body1, contacts)
    mark_two_body_contact(rigid_contact_shape1[contact], shape_body, shape_world, body_stride, body0, body1, contacts)
