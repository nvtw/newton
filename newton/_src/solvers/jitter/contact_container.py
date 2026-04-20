# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Persistent warm-start state for contact constraints.

The Jitter solver packs contact geometry into the shared
:class:`newton._src.solvers.jitter.constraint_container.ConstraintContainer`
as :data:`CONSTRAINT_TYPE_CONTACT` columns. Each column represents one
``(shape_a, shape_b)`` shape-pair slice of up to six contacts from the
upstream Newton :class:`newton._src.sim.contacts.Contacts` buffer. The
column stores only geometry + a ``[contact_first, contact_count]``
range pointing back into the sorted contact buffer -- everything is
fully replaced every step.

Persistent state (the accumulated normal + tangent impulses used to
warm-start the next step's PGS) lives *separately* in this module's
:class:`ContactContainer`. Two reasons to split it out of the main
:class:`ConstraintContainer` column:

1. Double-buffering. Warm-start needs both the previous step's finished
   lambdas and a place to write this step's new ones. Pointer-swapping
   the two dword-packed ``wp.array2d`` references in
   :func:`contact_container_swap_prev_current` is far cheaper than
   shuffling dwords inside a packed column.
2. Column stability. Ingest fully rewrites every contact column each
   step (cids may even change when pairs appear / disappear). If the
   lambdas lived inside the column they'd be clobbered before the warm-
   start gather could read them; keeping them in a parallel buffer
   means the gather can read the (now swapped-in) *prev* buffer freely
   while ingest scribbles the new geometry.

Storage mirrors :class:`ConstraintContainer`'s dword-packed column-major
layout: a single ``wp.array2d[wp.float32]`` of shape
``(CC_DWORDS, num_cols)`` with ``cid`` on the inner (contiguous) axis.
Per column the 18 dwords hold ``(normal, tangent1, tangent2)_lambda``
for each of the 6 slots. A warp full of threads walking the same slot
across consecutive cids issues one 128-byte transaction per load.
``MAX_SLOTS = 6`` is the "up to 6 contacts per convex-convex pair"
limit from Newton's narrow phase; pairs that report more contacts are
split across multiple columns at ingest time (see
:mod:`constraint_contact`).
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "CC_DWORDS",
    "CC_LAMBDA_DWORDS_PER_SLOT",
    "MAX_SLOTS",
    "ContactContainer",
    "cc_get_normal_lambda",
    "cc_get_prev_normal_lambda",
    "cc_get_prev_tangent1_lambda",
    "cc_get_prev_tangent2_lambda",
    "cc_get_tangent1_lambda",
    "cc_get_tangent2_lambda",
    "cc_set_normal_lambda",
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
#: tangent1_lambda, tangent2_lambda)``.
CC_LAMBDA_DWORDS_PER_SLOT: int = 3

#: Total dwords of persistent lambda state per contact column.
#: ``CC_LAMBDA_DWORDS_PER_SLOT * MAX_SLOTS = 18`` with the current
#: fixed-width 6-slot layout.
CC_DWORDS: int = CC_LAMBDA_DWORDS_PER_SLOT * MAX_SLOTS


# Module-level constants for warp kernels. ``wp.constant`` so they
# embed as compile-time literals in the emitted PTX.
_CC_LAMBDA_DWORDS_PER_SLOT = wp.constant(CC_LAMBDA_DWORDS_PER_SLOT)
_CC_OFF_NORMAL = wp.constant(0)
_CC_OFF_TANGENT1 = wp.constant(1)
_CC_OFF_TANGENT2 = wp.constant(2)


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
# Slot ``s`` occupies dwords ``[s * 3, s * 3 + 3)`` of a column, laid out
# as ``[normal, tangent1, tangent2]``. ``cid`` on the inner axis for
# coalesced access across a warp.


@wp.func
def cc_get_normal_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_NORMAL, cid]


@wp.func
def cc_set_normal_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.float32):
    cc.lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_NORMAL, cid] = v


@wp.func
def cc_get_tangent1_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_TANGENT1, cid]


@wp.func
def cc_set_tangent1_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.float32):
    cc.lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_TANGENT1, cid] = v


@wp.func
def cc_get_tangent2_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_TANGENT2, cid]


@wp.func
def cc_set_tangent2_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32, v: wp.float32):
    cc.lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_TANGENT2, cid] = v


@wp.func
def cc_get_prev_normal_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.prev_lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_NORMAL, cid]


@wp.func
def cc_get_prev_tangent1_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.prev_lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_TANGENT1, cid]


@wp.func
def cc_get_prev_tangent2_lambda(cc: ContactContainer, slot: wp.int32, cid: wp.int32) -> wp.float32:
    return cc.prev_lambdas[slot * _CC_LAMBDA_DWORDS_PER_SLOT + _CC_OFF_TANGENT2, cid]


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
