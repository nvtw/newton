# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-contact persistent + per-substep state, keyed by sorted-buffer index k.

* ``impulses`` -- mutable accumulated normal/tangent impulse rows.
* ``lambdas`` -- read-mostly persistent contact manifold data (normal, tangent,
  anchors, barycentrics).
* ``derived`` -- per-substep scratch (eff masses, biases); rebuilt every prepare.

All buffers are ``[dword, k]`` with k inner for coalesced loads.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.helpers.array_access import read2d_f32, write2d_f32

__all__ = [
    "CC_DERIVED_DWORDS_PER_CONTACT",
    "CC_DWORDS_PER_CONTACT",
    "CC_IMPULSE_DWORDS_PER_CONTACT",
    "ContactContainer",
    "cc_get_bias",
    "cc_get_bias_t1",
    "cc_get_bias_t2",
    "cc_get_eff_n",
    "cc_get_eff_t1",
    "cc_get_eff_t2",
    "cc_get_local_p0",
    "cc_get_local_p1",
    "cc_get_normal",
    "cc_get_normal_lambda",
    "cc_get_pd_bias",
    "cc_get_pd_eff_soft",
    "cc_get_pd_gamma",
    "cc_get_prev_local_p0",
    "cc_get_prev_local_p1",
    "cc_get_prev_normal",
    "cc_get_prev_normal_lambda",
    "cc_get_prev_tangent1",
    "cc_get_prev_tangent1_lambda",
    "cc_get_prev_tangent2_lambda",
    "cc_get_r0",
    "cc_get_r1",
    "cc_get_side0_bary",
    "cc_get_side1_bary",
    "cc_get_start_gap",
    "cc_get_tangent1",
    "cc_get_tangent1_lambda",
    "cc_get_tangent2_lambda",
    "cc_set_bias",
    "cc_set_bias_t1",
    "cc_set_bias_t2",
    "cc_set_eff_n",
    "cc_set_eff_t1",
    "cc_set_eff_t2",
    "cc_set_local_p0",
    "cc_set_local_p1",
    "cc_set_normal",
    "cc_set_normal_lambda",
    "cc_set_pd_bias",
    "cc_set_pd_eff_soft",
    "cc_set_pd_gamma",
    "cc_set_r0",
    "cc_set_r1",
    "cc_set_side0_bary",
    "cc_set_side1_bary",
    "cc_set_start_gap",
    "cc_set_tangent1",
    "cc_set_tangent1_lambda",
    "cc_set_tangent2_lambda",
    "contact_container_clear_reset_worlds",
    "contact_container_copy_current_to_prev",
    "contact_container_zeros",
]


#: Dwords of mutable persistent impulse per contact: normal + two tangents.
CC_IMPULSE_DWORDS_PER_CONTACT: int = 3

#: 18 = normal(3) + tangent1(3) + local_p0(3) + local_p1(3) + side0_bary(3)
#: + side1_bary(3). The two ``bary`` slots are populated by contact ingest when
#: a side is a cloth triangle; rigid sides leave them at zero.
CC_DWORDS_PER_CONTACT: int = 18

#: 16 = eff_n + eff_t1 + eff_t2 + bias + bias_t1 + bias_t2 + pd_gamma + pd_bias +
#: pd_eff_soft + r0(3) + r1(3). pd_* are non-zero only for soft contacts (user
#: K/D); pd_eff_soft > 0 switches the normal row to absolute PD spring-damper.
#: Rigid-contact prepare caches lever arms for the velocity sweeps. The final
#: slot stores the generation-time gap for current contacts only; it is written
#: by ingest and consumed by prepare before the velocity sweeps.
CC_DERIVED_DWORDS_PER_CONTACT: int = 16


# Compile-time dword offsets.
_CC_OFF_NORMAL_LAMBDA = wp.constant(0)
_CC_OFF_TANGENT1_LAMBDA = wp.constant(1)
_CC_OFF_TANGENT2_LAMBDA = wp.constant(2)
_CC_OFF_NORMAL_X = wp.constant(0)
_CC_OFF_NORMAL_Y = wp.constant(1)
_CC_OFF_NORMAL_Z = wp.constant(2)
_CC_OFF_TANGENT1_X = wp.constant(3)
_CC_OFF_TANGENT1_Y = wp.constant(4)
_CC_OFF_TANGENT1_Z = wp.constant(5)
_CC_OFF_LOCAL_P0_X = wp.constant(6)
_CC_OFF_LOCAL_P0_Y = wp.constant(7)
_CC_OFF_LOCAL_P0_Z = wp.constant(8)
_CC_OFF_LOCAL_P1_X = wp.constant(9)
_CC_OFF_LOCAL_P1_Y = wp.constant(10)
_CC_OFF_LOCAL_P1_Z = wp.constant(11)
_CC_OFF_SIDE0_BARY_X = wp.constant(12)
_CC_OFF_SIDE0_BARY_Y = wp.constant(13)
_CC_OFF_SIDE0_BARY_Z = wp.constant(14)
_CC_OFF_SIDE1_BARY_X = wp.constant(15)
_CC_OFF_SIDE1_BARY_Y = wp.constant(16)
_CC_OFF_SIDE1_BARY_Z = wp.constant(17)

_CC_OFF_EFF_N = wp.constant(0)
_CC_OFF_EFF_T1 = wp.constant(1)
_CC_OFF_EFF_T2 = wp.constant(2)
_CC_OFF_BIAS = wp.constant(3)
_CC_OFF_BIAS_T1 = wp.constant(4)
_CC_OFF_BIAS_T2 = wp.constant(5)
# Soft-contact PD plumbing. pd_eff_soft == 0 = opt-out (Box2D path).
_CC_OFF_PD_GAMMA = wp.constant(6)
_CC_OFF_PD_BIAS = wp.constant(7)
_CC_OFF_PD_EFF_SOFT = wp.constant(8)
_CC_OFF_R0_X = wp.constant(9)
_CC_OFF_R0_Y = wp.constant(10)
_CC_OFF_R0_Z = wp.constant(11)
_CC_OFF_R1_X = wp.constant(12)
_CC_OFF_R1_Y = wp.constant(13)
_CC_OFF_R1_Z = wp.constant(14)
_CC_OFF_START_GAP = wp.constant(15)


@wp.struct
class ContactContainer:
    """Per-contact warm-start + derived state. All buffers are
    ``(dwords, rigid_contact_max)`` with k inner. k matches Newton's per-contact
    arrays."""

    impulses: wp.array2d[wp.float32]
    prev_impulses: wp.array2d[wp.float32]
    lambdas: wp.array2d[wp.float32]
    prev_lambdas: wp.array2d[wp.float32]
    derived: wp.array2d[wp.float32]


@wp.kernel(enable_backward=False)
def _contact_container_copy_current_to_prev_kernel(cc: ContactContainer, valid_count: wp.array[wp.int32]):
    k = wp.tid()
    # Newton packs live contacts into [0, count); slots at or beyond the prior
    # solve's count hold no warm-start state, so skip them. The launch still
    # spans the full capacity (graph-stable dim); the early-out elides the copy
    # traffic for the typically large inactive tail.
    if k >= valid_count[0]:
        return
    for row in range(CC_IMPULSE_DWORDS_PER_CONTACT):
        cc.prev_impulses[row, k] = cc.impulses[row, k]
    for row in range(CC_DWORDS_PER_CONTACT):
        cc.prev_lambdas[row, k] = cc.lambdas[row, k]


@wp.kernel(enable_backward=False)
def _contact_container_clear_reset_worlds_kernel(
    dones: wp.array[wp.float32],
    shape_world: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    cc: ContactContainer,
    cid_of_contact_cur: wp.array[wp.int32],
    cid_of_contact_prev: wp.array[wp.int32],
):
    k = wp.tid()

    world = wp.int32(-1)
    shape0 = rigid_contact_shape0[k]
    if shape0 >= wp.int32(0) and shape0 < shape_world.shape[0]:
        world0 = shape_world[shape0]
        if world0 >= wp.int32(0):
            world = world0

    shape1 = rigid_contact_shape1[k]
    if world < wp.int32(0) and shape1 >= wp.int32(0) and shape1 < shape_world.shape[0]:
        world1 = shape_world[shape1]
        if world1 >= wp.int32(0):
            world = world1

    if world < wp.int32(0) or world >= dones.shape[0] or dones[world] <= wp.float32(0.5):
        return

    for row in range(CC_IMPULSE_DWORDS_PER_CONTACT):
        cc.impulses[row, k] = wp.float32(0.0)
        cc.prev_impulses[row, k] = wp.float32(0.0)
    for row in range(CC_DWORDS_PER_CONTACT):
        cc.lambdas[row, k] = wp.float32(0.0)
        cc.prev_lambdas[row, k] = wp.float32(0.0)
    for row in range(CC_DERIVED_DWORDS_PER_CONTACT):
        cc.derived[row, k] = wp.float32(0.0)

    cid_of_contact_cur[k] = wp.int32(-1)
    cid_of_contact_prev[k] = wp.int32(-1)


# Mutable impulse accessors keyed by contact index k.


@wp.func
def cc_get_normal_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.impulses, _CC_OFF_NORMAL_LAMBDA, k)


@wp.func
def cc_set_normal_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.impulses, _CC_OFF_NORMAL_LAMBDA, k, v)


@wp.func
def cc_get_tangent1_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.impulses, _CC_OFF_TANGENT1_LAMBDA, k)


@wp.func
def cc_set_tangent1_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.impulses, _CC_OFF_TANGENT1_LAMBDA, k, v)


@wp.func
def cc_get_tangent2_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.impulses, _CC_OFF_TANGENT2_LAMBDA, k)


@wp.func
def cc_set_tangent2_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.impulses, _CC_OFF_TANGENT2_LAMBDA, k, v)


@wp.func
def cc_get_normal(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_NORMAL_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_NORMAL_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_NORMAL_Z, k),
    )


@wp.func
def cc_set_normal(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_NORMAL_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_NORMAL_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_NORMAL_Z, k, v[2])


@wp.func
def cc_get_tangent1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_TANGENT1_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Z, k),
    )


@wp.func
def cc_set_tangent1(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_TANGENT1_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Z, k, v[2])


@wp.func
def cc_get_local_p0(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_LOCAL_P0_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_LOCAL_P0_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_LOCAL_P0_Z, k),
    )


@wp.func
def cc_set_local_p0(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_LOCAL_P0_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_LOCAL_P0_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_LOCAL_P0_Z, k, v[2])


@wp.func
def cc_get_local_p1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_LOCAL_P1_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_LOCAL_P1_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_LOCAL_P1_Z, k),
    )


@wp.func
def cc_set_local_p1(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_LOCAL_P1_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_LOCAL_P1_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_LOCAL_P1_Z, k, v[2])


# Per-side barycentric weights for cloth-aware endpoints. Populated by
# the contact ingest only when the corresponding ``side*_kind`` is
# ``CLOTH``; rigid sides leave them at zero (the iterate's endpoint
# helper just consumes them once it knows the kind).


@wp.func
def cc_get_side0_bary(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Z, k),
    )


@wp.func
def cc_set_side0_bary(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Z, k, v[2])


@wp.func
def cc_get_side1_bary(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Z, k),
    )


@wp.func
def cc_set_side1_bary(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Z, k, v[2])


@wp.func
def cc_get_start_gap(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_START_GAP, k)


@wp.func
def cc_set_start_gap(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_START_GAP, k, v)


# ---- prev-step views ------------------------------------------------


@wp.func
def cc_get_prev_normal_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.prev_impulses, _CC_OFF_NORMAL_LAMBDA, k)


@wp.func
def cc_get_prev_tangent1_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.prev_impulses, _CC_OFF_TANGENT1_LAMBDA, k)


@wp.func
def cc_get_prev_tangent2_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.prev_impulses, _CC_OFF_TANGENT2_LAMBDA, k)


@wp.func
def cc_get_prev_normal(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.prev_lambdas, _CC_OFF_NORMAL_X, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_NORMAL_Y, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_NORMAL_Z, k),
    )


@wp.func
def cc_get_prev_tangent1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.prev_lambdas, _CC_OFF_TANGENT1_X, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_TANGENT1_Y, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_TANGENT1_Z, k),
    )


@wp.func
def cc_get_prev_local_p0(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.prev_lambdas, _CC_OFF_LOCAL_P0_X, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_LOCAL_P0_Y, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_LOCAL_P0_Z, k),
    )


@wp.func
def cc_get_prev_local_p1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.prev_lambdas, _CC_OFF_LOCAL_P1_X, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_LOCAL_P1_Y, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_LOCAL_P1_Z, k),
    )


# ---------------------------------------------------------------------------
# Derived (per-substep scratch) accessors -- keyed by contact index k.
# ---------------------------------------------------------------------------


@wp.func
def cc_get_eff_n(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_EFF_N, k)


@wp.func
def cc_set_eff_n(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_EFF_N, k, v)


@wp.func
def cc_get_eff_t1(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_EFF_T1, k)


@wp.func
def cc_set_eff_t1(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_EFF_T1, k, v)


@wp.func
def cc_get_eff_t2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_EFF_T2, k)


@wp.func
def cc_set_eff_t2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_EFF_T2, k, v)


@wp.func
def cc_get_bias(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_BIAS, k)


@wp.func
def cc_set_bias(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_BIAS, k, v)


@wp.func
def cc_get_bias_t1(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_BIAS_T1, k)


@wp.func
def cc_set_bias_t1(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_BIAS_T1, k, v)


@wp.func
def cc_get_bias_t2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_BIAS_T2, k)


@wp.func
def cc_set_bias_t2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_BIAS_T2, k, v)


@wp.func
def cc_get_pd_gamma(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_PD_GAMMA, k)


@wp.func
def cc_set_pd_gamma(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_PD_GAMMA, k, v)


@wp.func
def cc_get_pd_bias(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_PD_BIAS, k)


@wp.func
def cc_set_pd_bias(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_PD_BIAS, k, v)


@wp.func
def cc_get_pd_eff_soft(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_PD_EFF_SOFT, k)


@wp.func
def cc_set_pd_eff_soft(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_PD_EFF_SOFT, k, v)


@wp.func
def cc_get_r0(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.derived, _CC_OFF_R0_X, k),
        read2d_f32(cc.derived, _CC_OFF_R0_Y, k),
        read2d_f32(cc.derived, _CC_OFF_R0_Z, k),
    )


@wp.func
def cc_set_r0(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.derived, _CC_OFF_R0_X, k, v[0])
    write2d_f32(cc.derived, _CC_OFF_R0_Y, k, v[1])
    write2d_f32(cc.derived, _CC_OFF_R0_Z, k, v[2])


@wp.func
def cc_get_r1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.derived, _CC_OFF_R1_X, k),
        read2d_f32(cc.derived, _CC_OFF_R1_Y, k),
        read2d_f32(cc.derived, _CC_OFF_R1_Z, k),
    )


@wp.func
def cc_set_r1(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.derived, _CC_OFF_R1_X, k, v[0])
    write2d_f32(cc.derived, _CC_OFF_R1_Y, k, v[1])
    write2d_f32(cc.derived, _CC_OFF_R1_Z, k, v[2])


# ---------------------------------------------------------------------------
# Host-side factories.
# ---------------------------------------------------------------------------


def contact_container_zeros(
    rigid_contact_max: int,
    device: wp.DeviceLike = None,
) -> ContactContainer:
    """Allocate a zero-initialised :class:`ContactContainer`. ``rigid_contact_max``
    must match Newton's Contacts buffer."""
    n = max(1, int(rigid_contact_max))
    cc = ContactContainer()
    cc.impulses = wp.zeros((CC_IMPULSE_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.prev_impulses = wp.zeros((CC_IMPULSE_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.lambdas = wp.zeros((CC_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.prev_lambdas = wp.zeros((CC_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.derived = wp.zeros((CC_DERIVED_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    return cc


def contact_container_copy_current_to_prev(
    cc: ContactContainer, valid_count: wp.array[wp.int32], device: wp.DeviceLike = None
) -> None:
    """Copy current persistent contact state into prev buffers.

    Args:
        valid_count: Single-element device array with the number of live
            contact slots (the prior solve's contact count). Slots beyond it
            are skipped, since Newton packs contacts into ``[0, count)``.
    """
    wp.launch(
        _contact_container_copy_current_to_prev_kernel,
        dim=int(cc.impulses.shape[1]),
        inputs=[cc, valid_count],
        device=device,
    )


def contact_container_clear_reset_worlds(
    cc: ContactContainer,
    cid_of_contact_cur: wp.array[wp.int32],
    cid_of_contact_prev: wp.array[wp.int32],
    contacts,
    shape_world: wp.array[wp.int32],
    dones: wp.array[wp.float32],
    device: wp.DeviceLike = None,
) -> None:
    """Clear contact warm-start state for worlds whose reset flag is set."""
    wp.launch(
        _contact_container_clear_reset_worlds_kernel,
        dim=int(cc.impulses.shape[1]),
        inputs=[
            dones,
            shape_world,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
        ],
        outputs=[cc, cid_of_contact_cur, cid_of_contact_prev],
        device=device,
    )
