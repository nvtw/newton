# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared kernels for the PhoenX example programs.

Every PhoenX example needs the same three small kernels to bridge
Newton's :class:`newton.State` and :class:`PhoenXWorld`'s
:class:`BodyContainer`:

* :func:`newton_to_phoenx_kernel` -- push Newton's body-origin
  transform + spatial velocity into PhoenX's COM-centred layout.
* :func:`phoenx_to_newton_kernel` -- the reverse, used after each
  :meth:`PhoenXWorld.step` so the viewer / downstream state
  consumers see the updated pose / velocity.
* :func:`init_phoenx_bodies_kernel` -- one-shot seed from a
  finalised Newton model into a fresh :class:`BodyContainer`. Slot
  0 is left alone (static world anchor); Newton body ``i`` lands
  at slot ``i + 1``.

These are pure state-mirroring kernels with no PhoenX-specific
logic, so they're identical across every example. The examples in
this package (:mod:`example_phoenx_tower`,
:mod:`example_phoenx_motorized_hinge_chain`,
:mod:`example_phoenx_nut_bolt`, ...) import them from here to avoid
duplicating the three launch sites.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    BodyContainer,
)

__all__ = [
    "init_phoenx_bodies_kernel",
    "newton_to_phoenx_kernel",
    "phoenx_to_newton_kernel",
]


@wp.kernel(enable_backward=False)
def newton_to_phoenx_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
):
    """Push Newton body state into PhoenX's :class:`BodyContainer`.

    Newton's ``body_q`` is the body-origin transform and ``body_com``
    is a body-local COM offset; PhoenX's ``position`` is the COM in
    world space, so we rotate ``body_com`` and add it.
    """
    i = wp.tid()
    q = body_q[i]
    pos_body = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    position[i] = pos_body + wp.quat_rotate(rot, body_com[i])
    orientation[i] = rot
    qd = body_qd[i]
    velocity[i] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[i] = wp.vec3f(qd[3], qd[4], qd[5])


@wp.kernel(enable_backward=False)
def phoenx_to_newton_kernel(
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Reverse of :func:`newton_to_phoenx_kernel`: strip out the
    rotated COM offset so Newton's ``body_q`` holds the body-origin
    transform that its viewer / FK helpers expect.
    """
    i = wp.tid()
    rot = orientation[i]
    com_world = wp.quat_rotate(rot, body_com[i])
    pos_body = position[i] - com_world
    body_q[i] = wp.transform(pos_body, rot)
    body_qd[i] = wp.spatial_vector(velocity[i], angular_velocity[i])


@wp.kernel(enable_backward=False)
def init_phoenx_bodies_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33f],
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia: wp.array[wp.mat33f],
    inverse_inertia_world: wp.array[wp.mat33f],
    motion_type: wp.array[wp.int32],
):
    """Copy a finalised Newton model's body state into PhoenX layout.

    Slot 0 is left alone (it's the static world anchor, pre-zero'd
    by :func:`body_container_zeros`). Newton body ``i`` lands at
    slot ``i + 1``. Dynamic bodies get :data:`MOTION_DYNAMIC`;
    zero-inverse-mass bodies stay :data:`MOTION_STATIC`.
    ``inverse_inertia_world`` is seeded with ``R * I * R^T`` so the
    first substep's effective-mass computation starts correct.
    """
    i = wp.tid()
    dst = i + 1
    q = body_q[i]
    pos_body = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    position[dst] = pos_body + wp.quat_rotate(rot, body_com[i])
    orientation[dst] = rot
    qd = body_qd[i]
    velocity[dst] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[dst] = wp.vec3f(qd[3], qd[4], qd[5])

    inv_m = body_inv_mass[i]
    inv_I = body_inv_inertia[i]
    inverse_mass[dst] = inv_m
    inverse_inertia[dst] = inv_I
    r = wp.quat_to_matrix(rot)
    inverse_inertia_world[dst] = r * inv_I * wp.transpose(r)
    if inv_m > 0.0:
        motion_type[dst] = wp.int32(MOTION_DYNAMIC)
    else:
        motion_type[dst] = wp.int32(MOTION_STATIC)
