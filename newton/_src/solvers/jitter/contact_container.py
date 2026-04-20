# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel-SoA persistent state for contact constraints.

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
   a handful of ``wp.array2d`` references in :func:`contact_container_swap_prev_current`
   is far cheaper than shuffling dwords inside a packed column.
2. Column stability. Ingest fully rewrites every contact column each
   step (cids may even change when pairs appear / disappear). If the
   lambdas lived inside the column they'd be clobbered before the warm-
   start gather could read them; keeping them in a parallel SoA buffer
   means the gather can read the (now swapped-in) *prev* buffer freely
   while ingest scribbles the new geometry.

Layout is column-major-by-cid to match the rest of the solver's
coalesced access pattern: ``normal_lambda[slot, cid]`` with ``cid`` on
the inner axis. Slot is the per-contact index within the 6-slot
column; a warp full of threads walking the same slot across consecutive
cids issues one 128-byte transaction per load. ``MAX_SLOTS = 6`` is the
"up to 6 contacts per convex-convex pair" limit from Newton's narrow
phase; pairs that report more contacts are split across multiple
columns at ingest time (see :mod:`constraint_contact`).
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "MAX_SLOTS",
    "ContactContainer",
    "contact_container_swap_prev_current",
    "contact_container_zeros",
]


#: Maximum number of contact slots per contact column. Matches Newton's
#: narrow-phase convex-convex upper bound; also the width of
#: ``active_mask`` in :class:`ContactConstraintData`. Keep in sync with
#: the schema in :mod:`constraint_contact`.
MAX_SLOTS: int = 6


@wp.struct
class ContactContainer:
    """Persistent per-contact state for :data:`CONSTRAINT_TYPE_CONTACT`.

    All arrays are ``(MAX_SLOTS, num_contact_columns)`` with ``cid`` on
    the inner (contiguous) axis. Index as ``array[slot, cid]`` from
    kernels -- ``slot`` is the bit index inside the column's
    ``active_mask`` (0..5). Inactive slots keep their zero-init values
    so any stray read from an unused slot degrades into a no-op
    warm-start impulse of zero.

    The container is *double buffered*: the "current" triple
    (:attr:`normal_lambda` / :attr:`tangent1_lambda` / :attr:`tangent2_lambda`)
    is what the iterate kernel writes to; the "prev" triple is what the
    warm-start gather kernel reads from at the start of the next step.
    :func:`contact_container_swap_prev_current` swaps the six handles
    in place at the very top of every :meth:`World.step`, after which
    "prev" holds last step's finished impulses and "current" is a
    (logically) cleared scratch buffer that the gather kernel is free
    to seed from the match index.

    :attr:`prev_contact_index` is the remap the gather kernel consults
    to translate a current-frame ``rigid_contact_match_index`` (which
    is a sorted-buffer index into the *previous* frame's contact array)
    into the ``(slot, cid)`` coordinate where that prev contact's
    lambdas were stored. Written during ingest for this frame, read
    next frame during gather; see :mod:`constraint_contact` for the
    exact encoding.
    """

    #: Accumulated normal impulse [N·s] per contact slot, written by the
    #: iterate kernel every PGS iteration. Shape (MAX_SLOTS, num_cols).
    normal_lambda: wp.array2d[wp.float32]
    #: Accumulated tangent-1 friction impulse [N·s]. Shape
    #: (MAX_SLOTS, num_cols).
    tangent1_lambda: wp.array2d[wp.float32]
    #: Accumulated tangent-2 friction impulse [N·s] (the second axis of
    #: the pyramidal friction model). Shape (MAX_SLOTS, num_cols).
    tangent2_lambda: wp.array2d[wp.float32]

    #: Previous step's :attr:`normal_lambda` after the pointer swap at
    #: the top of :meth:`World.step`. The warm-start gather kernel
    #: reads this via ``prev_contact_index`` to seed the new step's
    #: impulses. Shape (MAX_SLOTS, num_cols).
    prev_normal_lambda: wp.array2d[wp.float32]
    #: Previous step's :attr:`tangent1_lambda`. Shape (MAX_SLOTS, num_cols).
    prev_tangent1_lambda: wp.array2d[wp.float32]
    #: Previous step's :attr:`tangent2_lambda`. Shape (MAX_SLOTS, num_cols).
    prev_tangent2_lambda: wp.array2d[wp.float32]

    #: Encodes, for each slot of each column in the *current* frame,
    #: the sorted-buffer index of the prev-frame contact whose state
    #: the iterate kernel is about to write. Populated during ingest
    #: from ``contacts.rigid_contact_match_index`` of the *previous*
    #: frame (but stored at the current frame's (slot, cid) because
    #: that's where the iterate kernel will look next step). Kernels
    #: that only care about the *current* step's warm-start use the
    #: mirrored reverse lookup built from this frame's match index --
    #: see :func:`_contact_warmstart_gather_kernel`. ``-1`` means the
    #: slot holds a newly-created contact with no prior history.
    #: Shape (MAX_SLOTS, num_cols).
    prev_contact_index: wp.array2d[wp.int32]


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
    cc.normal_lambda = wp.zeros((MAX_SLOTS, cols), dtype=wp.float32, device=device)
    cc.tangent1_lambda = wp.zeros((MAX_SLOTS, cols), dtype=wp.float32, device=device)
    cc.tangent2_lambda = wp.zeros((MAX_SLOTS, cols), dtype=wp.float32, device=device)
    cc.prev_normal_lambda = wp.zeros((MAX_SLOTS, cols), dtype=wp.float32, device=device)
    cc.prev_tangent1_lambda = wp.zeros((MAX_SLOTS, cols), dtype=wp.float32, device=device)
    cc.prev_tangent2_lambda = wp.zeros((MAX_SLOTS, cols), dtype=wp.float32, device=device)
    # -1 = new contact / no warm-start history.
    cc.prev_contact_index = wp.full((MAX_SLOTS, cols), -1, dtype=wp.int32, device=device)
    return cc


def contact_container_swap_prev_current(cc: ContactContainer) -> None:
    """Pointer-swap the prev/current lambda triples in place.

    Called once at the very top of :meth:`World.step`. After the swap,
    the *current* triple holds the zero-initialised scratch storage
    that the warm-start gather will seed from ``prev_*`` and the
    iterate kernel will then fill in; ``prev_*`` holds last step's
    finished lambdas ready to be read by gather. No device-side copy.
    """
    cc.normal_lambda, cc.prev_normal_lambda = cc.prev_normal_lambda, cc.normal_lambda
    cc.tangent1_lambda, cc.prev_tangent1_lambda = cc.prev_tangent1_lambda, cc.tangent1_lambda
    cc.tangent2_lambda, cc.prev_tangent2_lambda = cc.prev_tangent2_lambda, cc.tangent2_lambda
