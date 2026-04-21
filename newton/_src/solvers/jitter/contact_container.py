# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Persistent warm-start state for contact constraints.

The Jitter solver packs contact geometry into the shared
:class:`newton._src.solvers.jitter.constraint_container.ConstraintContainer`
as :data:`CONSTRAINT_TYPE_CONTACT` columns. Each column represents one
``(shape_a, shape_b)`` shape-pair slice of up to six contacts from the
upstream Newton :class:`newton._src.sim.contacts.Contacts` buffer. The
column stores only *per-substep derived* quantities (lever arms,
effective masses, bias); everything persistent across frames lives in
this module's :class:`ContactContainer`.

Persistent state (PhoenX's rigid-rigid contact model):

* ``lam_n`` / ``lam_t1`` / ``lam_t2`` -- the accumulated normal and
  tangential impulses. Warm-started into the next step's PGS so the
  solver starts from a converged guess.
* ``normal`` -- the world-frame contact normal. Fixed for the lifetime
  of the contact; Newton's narrow phase recomputes a fresh normal every
  frame but we only honour the fresh value for *new* contacts. Matched
  contacts keep their original normal so the ``(n, t1, t2)`` frame
  they accumulated their impulses against stays meaningful.
* ``tangent1`` -- world-frame tangent, computed from the tangential
  relative velocity at contact creation (PhoenX's
  ``RRContactManifoldFunctions::Initialize``). ``tangent2`` is derived
  at use as ``cross(tangent1, normal)``.
* ``local_p0`` / ``local_p1`` -- body-1 / body-2 frame anchor points.
  Rotated by the current body orientation every substep to produce the
  world-frame lever arms ``r1`` / ``r2``.

Two reasons this state is split out of the :class:`ConstraintContainer`
column instead of packed inline:

1. Double-buffering. Warm-start needs both the previous step's finished
   state and a place to write this step's new state. Pointer-swapping
   the two dword-packed ``wp.array2d`` references in
   :func:`contact_container_swap_prev_current` is far cheaper than
   shuffling dwords inside a packed column.
2. Column stability. Ingest fully rewrites every contact column each
   step (cids may even change when pairs appear / disappear). If the
   persistent state lived inside the column it'd be clobbered before
   the warm-start gather could read it; a parallel buffer lets the
   gather read the (now swapped-in) *prev* buffer freely while ingest
   scribbles the new geometry.

Storage mirrors :class:`ConstraintContainer`'s dword-packed column-major
layout: a single ``wp.array2d[wp.float32]`` of shape
``(CC_DWORDS, num_cols)`` with ``cid`` on the inner (contiguous) axis.
Per slot we pack 15 dwords (``3 lambdas + normal(3) + tangent1(3) +
local_p0(3) + local_p1(3)``), tiled 6 times for the 6 slots → 90 dwords
per column. A warp full of threads walking the same slot field across
consecutive cids issues one 128-byte transaction per load.
``MAX_SLOTS = 6`` is the "up to 6 contacts per convex-convex pair"
limit from Newton's narrow phase; pairs that report more contacts are
split across multiple columns at ingest time (see
:mod:`constraint_contact`).
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "CC_DWORDS",
    "CC_DWORDS_PER_SLOT",
    "CC_LAMBDA_DWORDS_PER_SLOT",
    "MAX_SLOTS",
    "ContactContainer",
    "cc_get_local_p0",
    "cc_get_local_p1",
    "cc_get_normal",
    "cc_get_normal_lambda",
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
    "cc_set_local_p0",
    "cc_set_local_p1",
    "cc_set_normal",
    "cc_set_normal_lambda",
    "cc_set_tangent1",
    "cc_set_tangent1_lambda",
    "cc_set_tangent2_lambda",
    "contact_container_swap_prev_current",
    "contact_container_zeros",
]


#: Maximum number of contact slots per contact column. Matches Newton's
#: narrow-phase convex-convex upper bound; also the width of
#: ``active_mask`` in :class:`ContactConstraintData`. Keep in sync with
#: the schema in :mod:`constraint_contact`.
MAX_SLOTS: int = 6

#: Dwords of persistent impulse per contact slot: ``(normal_lambda,
#: tangent1_lambda, tangent2_lambda)``. Kept separate from the full
#: per-slot width so ingest zero-fill loops can mention "the impulse
#: triple" without magic numbers.
CC_LAMBDA_DWORDS_PER_SLOT: int = 3

#: Total persistent dwords per contact slot:
#: ``lam_n, lam_t1, lam_t2`` + ``normal(3)`` + ``tangent1(3)``
#: + ``local_p0(3)`` + ``local_p1(3)`` = 15. Matches PhoenX's
#: rigid-rigid contact per-point footprint.
CC_DWORDS_PER_SLOT: int = 15

#: Total dwords of persistent state per contact column.
#: ``CC_DWORDS_PER_SLOT * MAX_SLOTS = 90`` with the current fixed-width
#: 6-slot layout.
CC_DWORDS: int = CC_DWORDS_PER_SLOT * MAX_SLOTS


# Module-level constants for warp kernels. ``wp.constant`` so they
# embed as compile-time literals in the emitted PTX.
_CC_DWORDS_PER_SLOT = wp.constant(CC_DWORDS_PER_SLOT)
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


@wp.struct
class ContactContainer:
    """Persistent per-contact warm-start state.

    Stores two dword-packed buffers, :attr:`lambdas` (the "current"
    step's accumulating impulses) and :attr:`prev_lambdas` (the
    previous step's finished impulses read by the warm-start gather).
    Both have shape ``(CC_DWORDS, num_contact_columns)`` with ``cid``
    on the inner (contiguous) axis. Access via the
    ``cc_{get,set}_{normal,tangent1,tangent2}_lambda`` helpers -- the
    dword offset for slot ``s`` is ``s * CC_LAMBDA_DWORDS_PER_SLOT +
    {0, 1, 2}``.

    Inactive slots keep their zero-init values so any stray read from
    an unused slot degrades into a no-op warm-start impulse of zero.

    Double-buffering: :func:`contact_container_swap_prev_current` is
    called once at the very top of :meth:`World.step`, after which
    :attr:`prev_lambdas` holds last step's finished impulses (fed into
    the warm-start gather) and :attr:`lambdas` is the scratch buffer
    the gather then seeds and the iterate kernel fills in.
    """

    #: Dword-packed current-step impulses. Shape ``(CC_DWORDS,
    #: num_cols)``; dword ``s * 3 + {0, 1, 2}`` of column ``cid`` holds
    #: the normal / tangent1 / tangent2 lambda for slot ``s``.
    lambdas: wp.array2d[wp.float32]
    #: Dword-packed previous-step impulses, same layout as
    #: :attr:`lambdas`. Read by the warm-start gather kernel; swapped
    #: with :attr:`lambdas` each step.
    prev_lambdas: wp.array2d[wp.float32]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
#
# Slot ``s`` occupies dwords ``[s * 15, s * 15 + 15)`` of a column, laid
# out as ``[lam_n, lam_t1, lam_t2, n.xyz, t1.xyz, lp0.xyz, lp1.xyz]``.
# ``cid`` on the inner axis for coalesced access across a warp.


# ---- Impulses (current) ----


@wp.func
def cc_get_normal_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_NORMAL_LAMBDA, cid]


@wp.func
def cc_set_normal_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.float32):
    cc.lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_NORMAL_LAMBDA, cid] = v


@wp.func
def cc_get_tangent1_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_TANGENT1_LAMBDA, cid]


@wp.func
def cc_set_tangent1_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.float32):
    cc.lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_TANGENT1_LAMBDA, cid] = v


@wp.func
def cc_get_tangent2_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_TANGENT2_LAMBDA, cid]


@wp.func
def cc_set_tangent2_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.float32):
    cc.lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_TANGENT2_LAMBDA, cid] = v


# ---- Impulses (prev) ----


@wp.func
def cc_get_prev_normal_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.prev_lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_NORMAL_LAMBDA, cid]


@wp.func
def cc_get_prev_tangent1_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.prev_lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_TANGENT1_LAMBDA, cid]


@wp.func
def cc_get_prev_tangent2_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.prev_lambdas[slot * _CC_DWORDS_PER_SLOT + _CC_OFF_TANGENT2_LAMBDA, cid]


# ---- Contact frame + body-local anchors (current) ----


@wp.func
def cc_get_normal(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.lambdas[base + _CC_OFF_NORMAL_X, cid],
        cc.lambdas[base + _CC_OFF_NORMAL_Y, cid],
        cc.lambdas[base + _CC_OFF_NORMAL_Z, cid],
    )


@wp.func
def cc_set_normal(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.vec3f):
    base = slot * _CC_DWORDS_PER_SLOT
    cc.lambdas[base + _CC_OFF_NORMAL_X, cid] = v[0]
    cc.lambdas[base + _CC_OFF_NORMAL_Y, cid] = v[1]
    cc.lambdas[base + _CC_OFF_NORMAL_Z, cid] = v[2]


@wp.func
def cc_get_tangent1(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.lambdas[base + _CC_OFF_TANGENT1_X, cid],
        cc.lambdas[base + _CC_OFF_TANGENT1_Y, cid],
        cc.lambdas[base + _CC_OFF_TANGENT1_Z, cid],
    )


@wp.func
def cc_set_tangent1(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.vec3f):
    base = slot * _CC_DWORDS_PER_SLOT
    cc.lambdas[base + _CC_OFF_TANGENT1_X, cid] = v[0]
    cc.lambdas[base + _CC_OFF_TANGENT1_Y, cid] = v[1]
    cc.lambdas[base + _CC_OFF_TANGENT1_Z, cid] = v[2]


@wp.func
def cc_get_local_p0(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.lambdas[base + _CC_OFF_LOCAL_P0_X, cid],
        cc.lambdas[base + _CC_OFF_LOCAL_P0_Y, cid],
        cc.lambdas[base + _CC_OFF_LOCAL_P0_Z, cid],
    )


@wp.func
def cc_set_local_p0(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.vec3f):
    base = slot * _CC_DWORDS_PER_SLOT
    cc.lambdas[base + _CC_OFF_LOCAL_P0_X, cid] = v[0]
    cc.lambdas[base + _CC_OFF_LOCAL_P0_Y, cid] = v[1]
    cc.lambdas[base + _CC_OFF_LOCAL_P0_Z, cid] = v[2]


@wp.func
def cc_get_local_p1(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.lambdas[base + _CC_OFF_LOCAL_P1_X, cid],
        cc.lambdas[base + _CC_OFF_LOCAL_P1_Y, cid],
        cc.lambdas[base + _CC_OFF_LOCAL_P1_Z, cid],
    )


@wp.func
def cc_set_local_p1(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.vec3f):
    base = slot * _CC_DWORDS_PER_SLOT
    cc.lambdas[base + _CC_OFF_LOCAL_P1_X, cid] = v[0]
    cc.lambdas[base + _CC_OFF_LOCAL_P1_Y, cid] = v[1]
    cc.lambdas[base + _CC_OFF_LOCAL_P1_Z, cid] = v[2]


# ---- Contact frame + body-local anchors (prev) ----


@wp.func
def cc_get_prev_normal(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.prev_lambdas[base + _CC_OFF_NORMAL_X, cid],
        cc.prev_lambdas[base + _CC_OFF_NORMAL_Y, cid],
        cc.prev_lambdas[base + _CC_OFF_NORMAL_Z, cid],
    )


@wp.func
def cc_get_prev_tangent1(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.prev_lambdas[base + _CC_OFF_TANGENT1_X, cid],
        cc.prev_lambdas[base + _CC_OFF_TANGENT1_Y, cid],
        cc.prev_lambdas[base + _CC_OFF_TANGENT1_Z, cid],
    )


@wp.func
def cc_get_prev_local_p0(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.prev_lambdas[base + _CC_OFF_LOCAL_P0_X, cid],
        cc.prev_lambdas[base + _CC_OFF_LOCAL_P0_Y, cid],
        cc.prev_lambdas[base + _CC_OFF_LOCAL_P0_Z, cid],
    )


@wp.func
def cc_get_prev_local_p1(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.vec3f:
    base = slot * _CC_DWORDS_PER_SLOT
    return wp.vec3f(
        cc.prev_lambdas[base + _CC_OFF_LOCAL_P1_X, cid],
        cc.prev_lambdas[base + _CC_OFF_LOCAL_P1_Y, cid],
        cc.prev_lambdas[base + _CC_OFF_LOCAL_P1_Z, cid],
    )


def contact_container_zeros(
    max_contact_columns: int,
    device: wp.DeviceLike = None,
) -> ContactContainer:
    """Allocate a zero-initialised :class:`ContactContainer`.

    Args:
        max_contact_columns: Upper bound on the number of contact
            columns (``CONSTRAINT_TYPE_CONTACT`` constraints) the
            solver will ever pack per step. This is the user-supplied
            cap passed down through :meth:`World.__init__`; the
            container never resizes at run time.
        device: Warp device.
    """
    # Always allocate at least 1 column so the wp.array2d shape is
    # non-degenerate; the ingest/gather kernels gate on bounds anyway.
    cols = max(1, int(max_contact_columns))
    cc = ContactContainer()
    cc.lambdas = wp.zeros((CC_DWORDS, cols), dtype=wp.float32, device=device)
    cc.prev_lambdas = wp.zeros((CC_DWORDS, cols), dtype=wp.float32, device=device)
    return cc


def contact_container_swap_prev_current(cc: ContactContainer) -> None:
    """Pointer-swap the prev/current lambda buffers in place.

    Called once at the very top of :meth:`World.step`. After the swap,
    :attr:`ContactContainer.lambdas` is the scratch buffer that the
    warm-start gather will seed from :attr:`prev_lambdas` and the
    iterate kernel will then fill in; :attr:`prev_lambdas` holds last
    step's finished impulses ready to be read by gather. No device-side
    copy.
    """
    cc.lambdas, cc.prev_lambdas = cc.prev_lambdas, cc.lambdas
