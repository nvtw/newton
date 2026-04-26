# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Persistent + per-substep state for contact constraints, keyed by the
contact's sorted-buffer index ``k``.

* :attr:`ContactContainer.lambdas` -- persistent warm-start
  (accumulated normal + tangent impulses, frozen ``(normal, tangent1)``
  frame, body-local anchors). Double-buffered against
  :attr:`prev_lambdas` via pointer swap so the warm-start gather can
  read last step's finished state while the new step scribbles.
* :attr:`ContactContainer.derived` -- per-substep scratch (lever arms
  ``r1`` / ``r2``, scalar effective masses, velocity biases). Rebuilt
  every prepare; not double-buffered.

Both buffers are ``[dword, k]`` with ``k`` inner so a warp walking
adjacent contacts issues one coalesced transaction per field load.

Per-contact dword budgets:

* :data:`CC_DWORDS_PER_CONTACT` = 15 -- ``lam_n, lam_t1, lam_t2,
  normal(3), tangent1(3), local_p0(3), local_p1(3)``.
* :data:`CC_DERIVED_DWORDS_PER_CONTACT` = 12 -- ``r1(3), r2(3),
  eff_n, eff_t1, eff_t2, bias, bias_t1, bias_t2``.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "CC_DERIVED_DWORDS_PER_CONTACT",
    "CC_DWORDS_PER_CONTACT",
    "CC_LAMBDA_DWORDS_PER_CONTACT",
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
    "cc_set_tangent1",
    "cc_set_tangent1_lambda",
    "cc_set_tangent2_lambda",
    "contact_container_swap_prev_current",
    "contact_container_zeros",
]


#: Dwords of persistent impulse per contact: normal + two tangent lambdas.
CC_LAMBDA_DWORDS_PER_CONTACT: int = 3

#: Total persistent dwords per contact: ``lam_n, lam_t1, lam_t2`` +
#: ``normal(3)`` + ``tangent1(3)`` + ``local_p0(3)`` + ``local_p1(3)``.
#: Mirrors PhoenX's rigid-rigid contact per-point footprint.
CC_DWORDS_PER_CONTACT: int = 15

#: Per-contact derived dwords filled by ``prepare_for_iteration``:
#: ``eff_n, eff_t1, eff_t2, bias, bias_t1, bias_t2,
#: pd_gamma, pd_bias, pd_eff_soft``. The last three are non-zero only
#: for PhysX-style soft contacts (user K/D on Newton :class:`Contacts`);
#: ``pd_eff_soft > 0`` switches the normal row from Box2D hertz-based
#: to absolute PD spring-damper.
#:
#: ``r1`` / ``r2`` (lever arms) are recomputed in iterate from
#: ``local_p0`` / ``local_p1`` + body pose (saves 6 dwords/contact;
#: one extra ``quat_rotate`` per body-contact, typically hoisted).
CC_DERIVED_DWORDS_PER_CONTACT: int = 9


# Module-level constants for Warp kernels. ``wp.constant`` so they embed
# as compile-time literals in the emitted PTX.
_CC_OFF_NORMAL_LAMBDA = wp.constant(0)
_CC_OFF_TANGENT1_LAMBDA = wp.constant(1)
_CC_OFF_TANGENT2_LAMBDA = wp.constant(2)
_CC_OFF_NORMAL_X = wp.constant(3)
_CC_OFF_NORMAL_Y = wp.constant(4)
_CC_OFF_NORMAL_Z = wp.constant(5)
_CC_OFF_TANGENT1_X = wp.constant(6)
_CC_OFF_TANGENT1_Y = wp.constant(7)
_CC_OFF_TANGENT1_Z = wp.constant(8)
_CC_OFF_LOCAL_P0_X = wp.constant(9)
_CC_OFF_LOCAL_P0_Y = wp.constant(10)
_CC_OFF_LOCAL_P0_Z = wp.constant(11)
_CC_OFF_LOCAL_P1_X = wp.constant(12)
_CC_OFF_LOCAL_P1_Y = wp.constant(13)
_CC_OFF_LOCAL_P1_Z = wp.constant(14)

_CC_OFF_EFF_N = wp.constant(0)
_CC_OFF_EFF_T1 = wp.constant(1)
_CC_OFF_EFF_T2 = wp.constant(2)
_CC_OFF_BIAS = wp.constant(3)
_CC_OFF_BIAS_T1 = wp.constant(4)
_CC_OFF_BIAS_T2 = wp.constant(5)
# Soft-contact PD plumbing (pairs with ``pd_coefficients`` in
# ``constraint_container.py``). Written by
# :func:`contact_prepare_for_iteration_at` when the Newton
# ``rigid_contact_stiffness`` / ``rigid_contact_damping`` arrays are
# populated (per-contact, absolute units: N/m and N*s/m).
# ``pd_eff_soft == 0`` is the per-contact opt-out -- iterate then
# falls through to the legacy Box2D path unchanged.
_CC_OFF_PD_GAMMA = wp.constant(6)
_CC_OFF_PD_BIAS = wp.constant(7)
_CC_OFF_PD_EFF_SOFT = wp.constant(8)


@wp.struct
class ContactContainer:
    """Per-contact warm-start + derived state.

    All three buffers have shape ``(dwords, rigid_contact_max)`` with
    the contact index ``k`` on the inner (contiguous) axis. ``k`` is the
    same index Newton's :class:`Contacts` buffer uses for its per-contact
    arrays (``rigid_contact_point0[k]`` etc.) so the solver kernels and
    the narrow phase speak the same coordinates.
    """

    #: Persistent dword-packed current-step state. Fields packed in the
    #: order ``lam_n, lam_t1, lam_t2, normal.xyz, tangent1.xyz,
    #: local_p0.xyz, local_p1.xyz``. Shape
    #: ``(CC_DWORDS_PER_CONTACT, rigid_contact_max)``.
    lambdas: wp.array2d[wp.float32]
    #: Persistent dword-packed previous-step state. Same layout as
    #: :attr:`lambdas`; swapped once per :meth:`step` so the warm-start
    #: gather can read last frame's impulses/frame while ingest
    #: scribbles the new step's state.
    prev_lambdas: wp.array2d[wp.float32]
    #: Per-substep derived scratch. Fields packed in the order
    #: ``r1.xyz, r2.xyz, eff_n, eff_t1, eff_t2, bias, bias_t1,
    #: bias_t2``. Not double-buffered -- ``prepare_for_iteration``
    #: fills every field before ``iterate`` reads it.
    derived: wp.array2d[wp.float32]


# ---------------------------------------------------------------------------
# Persistent (lambdas) accessors -- keyed by contact index k.
# ---------------------------------------------------------------------------


@wp.func
def cc_get_normal_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.lambdas[_CC_OFF_NORMAL_LAMBDA, k]


@wp.func
def cc_set_normal_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.lambdas[_CC_OFF_NORMAL_LAMBDA, k] = v


@wp.func
def cc_get_tangent1_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.lambdas[_CC_OFF_TANGENT1_LAMBDA, k]


@wp.func
def cc_set_tangent1_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.lambdas[_CC_OFF_TANGENT1_LAMBDA, k] = v


@wp.func
def cc_get_tangent2_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.lambdas[_CC_OFF_TANGENT2_LAMBDA, k]


@wp.func
def cc_set_tangent2_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.lambdas[_CC_OFF_TANGENT2_LAMBDA, k] = v


@wp.func
def cc_get_normal(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.lambdas[_CC_OFF_NORMAL_X, k],
        cc.lambdas[_CC_OFF_NORMAL_Y, k],
        cc.lambdas[_CC_OFF_NORMAL_Z, k],
    )


@wp.func
def cc_set_normal(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    cc.lambdas[_CC_OFF_NORMAL_X, k] = v[0]
    cc.lambdas[_CC_OFF_NORMAL_Y, k] = v[1]
    cc.lambdas[_CC_OFF_NORMAL_Z, k] = v[2]


@wp.func
def cc_get_tangent1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.lambdas[_CC_OFF_TANGENT1_X, k],
        cc.lambdas[_CC_OFF_TANGENT1_Y, k],
        cc.lambdas[_CC_OFF_TANGENT1_Z, k],
    )


@wp.func
def cc_set_tangent1(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    cc.lambdas[_CC_OFF_TANGENT1_X, k] = v[0]
    cc.lambdas[_CC_OFF_TANGENT1_Y, k] = v[1]
    cc.lambdas[_CC_OFF_TANGENT1_Z, k] = v[2]


@wp.func
def cc_get_local_p0(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.lambdas[_CC_OFF_LOCAL_P0_X, k],
        cc.lambdas[_CC_OFF_LOCAL_P0_Y, k],
        cc.lambdas[_CC_OFF_LOCAL_P0_Z, k],
    )


@wp.func
def cc_set_local_p0(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    cc.lambdas[_CC_OFF_LOCAL_P0_X, k] = v[0]
    cc.lambdas[_CC_OFF_LOCAL_P0_Y, k] = v[1]
    cc.lambdas[_CC_OFF_LOCAL_P0_Z, k] = v[2]


@wp.func
def cc_get_local_p1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.lambdas[_CC_OFF_LOCAL_P1_X, k],
        cc.lambdas[_CC_OFF_LOCAL_P1_Y, k],
        cc.lambdas[_CC_OFF_LOCAL_P1_Z, k],
    )


@wp.func
def cc_set_local_p1(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    cc.lambdas[_CC_OFF_LOCAL_P1_X, k] = v[0]
    cc.lambdas[_CC_OFF_LOCAL_P1_Y, k] = v[1]
    cc.lambdas[_CC_OFF_LOCAL_P1_Z, k] = v[2]


# ---- prev-step views ------------------------------------------------


@wp.func
def cc_get_prev_normal_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.prev_lambdas[_CC_OFF_NORMAL_LAMBDA, k]


@wp.func
def cc_get_prev_tangent1_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.prev_lambdas[_CC_OFF_TANGENT1_LAMBDA, k]


@wp.func
def cc_get_prev_tangent2_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.prev_lambdas[_CC_OFF_TANGENT2_LAMBDA, k]


@wp.func
def cc_get_prev_normal(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.prev_lambdas[_CC_OFF_NORMAL_X, k],
        cc.prev_lambdas[_CC_OFF_NORMAL_Y, k],
        cc.prev_lambdas[_CC_OFF_NORMAL_Z, k],
    )


@wp.func
def cc_get_prev_tangent1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.prev_lambdas[_CC_OFF_TANGENT1_X, k],
        cc.prev_lambdas[_CC_OFF_TANGENT1_Y, k],
        cc.prev_lambdas[_CC_OFF_TANGENT1_Z, k],
    )


@wp.func
def cc_get_prev_local_p0(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.prev_lambdas[_CC_OFF_LOCAL_P0_X, k],
        cc.prev_lambdas[_CC_OFF_LOCAL_P0_Y, k],
        cc.prev_lambdas[_CC_OFF_LOCAL_P0_Z, k],
    )


@wp.func
def cc_get_prev_local_p1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        cc.prev_lambdas[_CC_OFF_LOCAL_P1_X, k],
        cc.prev_lambdas[_CC_OFF_LOCAL_P1_Y, k],
        cc.prev_lambdas[_CC_OFF_LOCAL_P1_Z, k],
    )


# ---------------------------------------------------------------------------
# Derived (per-substep scratch) accessors -- keyed by contact index k.
# ---------------------------------------------------------------------------


@wp.func
def cc_get_eff_n(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_EFF_N, k]


@wp.func
def cc_set_eff_n(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_EFF_N, k] = v


@wp.func
def cc_get_eff_t1(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_EFF_T1, k]


@wp.func
def cc_set_eff_t1(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_EFF_T1, k] = v


@wp.func
def cc_get_eff_t2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_EFF_T2, k]


@wp.func
def cc_set_eff_t2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_EFF_T2, k] = v


@wp.func
def cc_get_bias(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_BIAS, k]


@wp.func
def cc_set_bias(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_BIAS, k] = v


@wp.func
def cc_get_bias_t1(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_BIAS_T1, k]


@wp.func
def cc_set_bias_t1(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_BIAS_T1, k] = v


@wp.func
def cc_get_bias_t2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_BIAS_T2, k]


@wp.func
def cc_set_bias_t2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_BIAS_T2, k] = v


@wp.func
def cc_get_pd_gamma(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_PD_GAMMA, k]


@wp.func
def cc_set_pd_gamma(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_PD_GAMMA, k] = v


@wp.func
def cc_get_pd_bias(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_PD_BIAS, k]


@wp.func
def cc_set_pd_bias(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_PD_BIAS, k] = v


@wp.func
def cc_get_pd_eff_soft(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return cc.derived[_CC_OFF_PD_EFF_SOFT, k]


@wp.func
def cc_set_pd_eff_soft(cc: ContactContainer, k: wp.int32, v: wp.float32):
    cc.derived[_CC_OFF_PD_EFF_SOFT, k] = v


# ---------------------------------------------------------------------------
# Host-side factories.
# ---------------------------------------------------------------------------


def contact_container_zeros(
    rigid_contact_max: int,
    device: wp.DeviceLike = None,
) -> ContactContainer:
    """Allocate a zero-initialised :class:`ContactContainer`.

    Args:
        rigid_contact_max: Upper bound on the number of individual
            contacts in the upstream Newton ``Contacts`` buffer; sizes
            the inner (``k``) axis of all three storage buffers. Must
            match ``Contacts.rigid_contact_max`` of the buffer the
            solver will ingest every step.
        device: Warp device.
    """
    # Always allocate at least 1 slot so the ``wp.array2d`` shape is
    # non-degenerate; kernels gate on the per-step active-count counter
    # anyway.
    n = max(1, int(rigid_contact_max))
    cc = ContactContainer()
    cc.lambdas = wp.zeros((CC_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.prev_lambdas = wp.zeros((CC_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.derived = wp.zeros((CC_DERIVED_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    return cc


def contact_container_swap_prev_current(cc: ContactContainer) -> None:
    """Pointer-swap the prev/current persistent lambda buffers in place.

    Called once at the top of :meth:`World.step`. After the swap,
    :attr:`ContactContainer.lambdas` is the scratch buffer that the
    warm-start gather seeds from :attr:`prev_lambdas`; the iterate
    kernels then read + write ``lambdas`` in place. The derived buffer
    is not swapped -- it's per-substep scratch rebuilt by prepare every
    substep.
    """
    cc.lambdas, cc.prev_lambdas = cc.prev_lambdas, cc.lambdas
