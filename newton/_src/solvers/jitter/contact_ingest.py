# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Convert Newton's sorted ``Contacts`` buffer into Jitter contact columns.

Called once per :meth:`World.step` to materialise that step's
:data:`CONSTRAINT_TYPE_CONTACT` columns inside the shared
:class:`ConstraintContainer`. The upstream Newton
:class:`newton._src.sim.contacts.Contacts` buffer is already sorted by
``(shape_a, shape_b)`` when ``contact_matching`` is on (the
:func:`newton._src.geometry.contact_data.make_contact_sort_key`
invariant), so all contacts belonging to one shape-pair are
contiguous. Ingest therefore reduces to:

1. Segment the sorted active prefix of ``Contacts`` into shape-pair
   runs (``pair_first[p]``, ``pair_count[p]``, ``pair_shape_a/b[p]``).
2. For each pair ``p``, emit ``ceil(pair_count[p] / 6)`` adjacent
   contact columns with the appropriate ``[contact_first,
   contact_count]`` range and ``active_mask``.
3. Populate the :class:`ConstraintContainer` header
   (``constraint_type``, ``body1``, ``body2``) so the generic
   dispatcher routes each column to
   :func:`contact_prepare_for_iteration_at`.

The number of output columns is produced as a *device-side* scalar
(``IngestScratch.num_contact_columns``) and never read back to the
host during the step, which keeps the whole ingest sequence
graph-capture compatible. All kernels launch at fixed sizes
(``rigid_contact_max`` or ``max_contact_columns``) and gate
internally on the device-held counters. The only host reads happen
outside the captured region, in :meth:`World.step`, to decide how
many constraints to feed to the graph-coloring partitioner and the
iterate kernels.

Segmenting uses an ``(shape_a * num_shapes + shape_b)`` int32 key +
the graph-capture-safe
:func:`scan_and_sort.runlength_encode_variable_length` wrapper. The
int32 fits safely because Newton scenes with
``num_shapes >= 46340`` would overflow the key; the caller should
bound ``num_shapes`` accordingly at world construction time.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.constraint_contact import (
    CONTACT_MAX_SLOTS,
    contact_set_active_mask,
    contact_set_contact_first,
    contact_set_friction,
)
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_BODY1_OFFSET,
    CONSTRAINT_BODY2_OFFSET,
    CONSTRAINT_TYPE_CONTACT,
    CONSTRAINT_TYPE_OFFSET,
    ConstraintContainer,
    write_int,
)
from newton._src.solvers.jitter.contact_container import (
    ContactContainer,
    cc_get_prev_normal_lambda,
    cc_get_prev_tangent1_lambda,
    cc_get_prev_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.jitter.scan_and_sort import (
    RLE_SENTINEL_INT32,
    runlength_encode_variable_length,
)

__all__ = [
    "IngestScratch",
    "allocate_ingest_scratch",
    "gather_contact_warmstart",
    "ingest_contacts",
    "stamp_forward_contact_map",
]


# ---------------------------------------------------------------------------
# Host-side scratch -- reused across steps to avoid per-step allocs.
# ---------------------------------------------------------------------------


class IngestScratch:
    """Pre-allocated device buffers the ingest pipeline reuses each step.

    Sized to the upstream ``Contacts.rigid_contact_max`` (for per-
    contact arrays) and to ``max_contact_columns`` (for per-column
    arrays). Per-pair arrays are sized to ``rigid_contact_max`` too
    because in the worst case (every contact is its own shape pair)
    ``num_pairs == rigid_contact_max``.

    All scratch lives on ``device`` and is never resized after
    construction -- the shapes drive the launch sizes of every
    ingest kernel, which is what keeps the sequence graph-capture
    safe.
    """

    __slots__ = (
        "_device",
        "keys",
        "max_contact_columns",
        "num_contact_columns",
        "num_pairs",
        "pair_col_offset",
        "pair_columns",
        "pair_count",
        "pair_first",
        "pair_shape_a",
        "pair_shape_b",
        "pair_source_idx",
        "rigid_contact_max",
        "run_values",
    )

    def __init__(
        self,
        rigid_contact_max: int,
        max_contact_columns: int,
        device: wp.DeviceLike = None,
    ) -> None:
        self._device = device
        self.rigid_contact_max = int(rigid_contact_max)
        self.max_contact_columns = int(max_contact_columns)
        n_pairs_max = max(1, self.rigid_contact_max)
        n_cols_max = max(1, self.max_contact_columns)

        # Per-contact pair key. Bounded by ``rigid_contact_max``.
        self.keys = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=device)

        # Output slot for RLE's unique-values array (keeps ``keys``
        # intact so callers can re-RLE / re-inspect if needed).
        self.run_values = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)

        # Per-pair arrays.
        self.pair_first = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_count = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_a = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_shape_b = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_columns = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)
        self.pair_col_offset = wp.zeros(n_pairs_max, dtype=wp.int32, device=device)

        # Per-output-column arrays.
        self.pair_source_idx = wp.zeros(n_cols_max, dtype=wp.int32, device=device)

        # Device-held scalars. ``num_pairs`` is filled by the RLE pass;
        # ``num_contact_columns`` is filled by the total-columns
        # reduction kernel. Both are consumed by subsequent kernels
        # via a ``tid >= counter[0]`` gate, never read back host-side
        # inside the captured step.
        self.num_pairs = wp.zeros(1, dtype=wp.int32, device=device)
        self.num_contact_columns = wp.zeros(1, dtype=wp.int32, device=device)


def allocate_ingest_scratch(
    rigid_contact_max: int,
    max_contact_columns: int,
    device: wp.DeviceLike = None,
) -> IngestScratch:
    """Factory alias matching the ``*_zeros`` style elsewhere in the solver."""
    return IngestScratch(rigid_contact_max, max_contact_columns, device=device)


# ---------------------------------------------------------------------------
# Step 1: build the int32 (shape_a, shape_b) key per active contact.
# ---------------------------------------------------------------------------


@wp.kernel
def _contact_key_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    num_shapes: wp.int32,
    # out
    keys: wp.array[wp.int32],
):
    """Fill the per-contact key array for the RLE pass.

    The active prefix (``tid < rigid_contact_count[0]``) gets
    packed keys ``shape_a * num_shapes + shape_b``; the tail is
    zeroed and will be re-stamped with the sentinel by the RLE
    wrapper (:func:`runlength_encode_variable_length`) -- we don't
    stamp the sentinel here because the RLE wrapper needs to re-
    stamp it anyway to keep the mask/RLE pair self-contained.
    """
    tid = wp.tid()
    count = rigid_contact_count[0]
    if tid < count:
        sa = rigid_contact_shape0[tid]
        sb = rigid_contact_shape1[tid]
        keys[tid] = sa * num_shapes + sb
    else:
        # Stamp zero so the subsequent RLE-wrapper sentinel overwrite
        # can't accidentally leave stale data from a previous step
        # masquerading as a real key.
        keys[tid] = wp.int32(0)


# ---------------------------------------------------------------------------
# Step 2: turn run_lengths into [pair_count, shape_a, shape_b, pair_columns].
# ---------------------------------------------------------------------------


@wp.kernel
def _pair_metadata_kernel(
    run_values: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    num_shapes: wp.int32,
    # out
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
):
    """Unpack ``run_values`` into per-pair shapes + column counts.

    ``pair_count`` is already the RLE ``run_lengths`` output; the RLE
    wrapper wrote it directly into ``scratch.pair_count`` so there's
    nothing to do for it here.

    Launches at ``dim == rigid_contact_max`` (the max possible pair
    count); threads past ``num_pairs[0]`` zero out the corresponding
    ``pair_columns`` entry so the subsequent exclusive scan produces
    a stable total (equal to the inclusive tail of the active
    prefix) regardless of leftover data from a previous step.
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        # Clear tail so the downstream scan is deterministic.
        pair_columns[tid] = wp.int32(0)
        return
    key = run_values[tid]
    sa = key // num_shapes
    sb = key - sa * num_shapes
    pair_shape_a[tid] = sa
    pair_shape_b[tid] = sb
    # Number of 6-slot contact columns needed for this pair.
    # ``pair_count[tid]`` already holds the run length written by
    # the RLE wrapper; read it directly.
    # (Kernel can't both read and write the same array at the same
    # index, so we don't take pair_count as an output.)


@wp.kernel
def _pair_columns_from_count_kernel(
    pair_count: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    # out
    pair_columns: wp.array[wp.int32],
):
    """Compute ``pair_columns[p] = ceil(pair_count[p] / 6)`` for active pairs.

    Split out from :func:`_pair_metadata_kernel` because that kernel
    already has ``pair_columns`` bound as an output for the tail
    clear; keeping this in a separate launch makes both kernels'
    access patterns read-only except for one output array.
    """
    tid = wp.tid()
    n = num_pairs[0]
    if tid >= n:
        return
    length = pair_count[tid]
    pair_columns[tid] = (length + CONTACT_MAX_SLOTS - 1) // CONTACT_MAX_SLOTS


@wp.kernel
def _pair_first_kernel(
    run_lengths: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    # out
    pair_first: wp.array[wp.int32],
):
    """Exclusive prefix sum of ``run_lengths`` -> ``pair_first``.

    Single-thread O(num_pairs) scan. ``num_pairs`` is small in
    practice (100s not millions) so the serial cost is negligible
    compared to launching a proper device scan, and this keeps the
    ingest sequence free of extra scratch allocations.
    """
    tid = wp.tid()
    if tid != 0:
        return
    n = num_pairs[0]
    acc = wp.int32(0)
    for i in range(n):
        pair_first[i] = acc
        acc += run_lengths[i]


# ---------------------------------------------------------------------------
# Step 3: total column count & per-output-column pair map.
# ---------------------------------------------------------------------------


@wp.kernel
def _total_contact_columns_kernel(
    pair_col_offset: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    max_columns: wp.int32,
    # out
    num_contact_columns: wp.array[wp.int32],
):
    """Derive ``num_contact_columns = sum(pair_columns)`` on-device.

    The exclusive scan of ``pair_columns`` has already been written
    to ``pair_col_offset`` by :func:`wp.utils.array_scan` before
    this kernel is launched, so we just read the last element of
    the scan + ``pair_columns[num_pairs - 1]`` and clamp against
    the caller's hard cap. Single-thread kernel; output is a
    device-held 1-element array consumed by all downstream
    ingest / gather kernels as their active-length gate.
    """
    tid = wp.tid()
    if tid != 0:
        return
    n = num_pairs[0]
    if n <= 0:
        num_contact_columns[0] = wp.int32(0)
        return
    total = pair_col_offset[n - 1] + pair_columns[n - 1]
    if total > max_columns:
        total = max_columns
    if total < wp.int32(0):
        total = wp.int32(0)
    num_contact_columns[0] = total


@wp.kernel
def _pair_source_idx_kernel(
    pair_col_offset: wp.array[wp.int32],
    pair_columns: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
    max_columns: wp.int32,
    # out
    pair_source_idx: wp.array[wp.int32],
):
    """For each output column ``o``, write which pair ``p`` it comes from.

    Serial on one thread because ``num_pairs`` is small and the
    expansion is trivially bounded by ``max_columns``. No race, no
    atomics; clamps every write to ``[0, max_columns)``.
    """
    tid = wp.tid()
    if tid != 0:
        return
    n = num_pairs[0]
    for p in range(n):
        off = pair_col_offset[p]
        cols = pair_columns[p]
        for k in range(cols):
            o = off + k
            if o < max_columns:
                pair_source_idx[o] = p


# ---------------------------------------------------------------------------
# Step 4: per-column pack kernel -- writes the contact constraint header
# and the [contact_first, contact_count] range into the ConstraintContainer.
# ---------------------------------------------------------------------------


@wp.kernel
def _contact_pack_columns_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_col_offset: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    cid_base: wp.int32,
    default_friction: wp.float32,
    # out
    constraints: ConstraintContainer,
):
    """Materialise one contact column.

    Launched at ``dim == max_contact_columns`` (host-known) and
    gates internally on ``tid >= num_contact_columns[0]`` so the
    launch size is captureable in a graph independently of the
    per-step column count.

    Thread ``tid`` owns output column ``tid``. Looks up which pair
    ``p`` it belongs to via ``pair_source_idx``, then its sub-
    column index within that pair as ``tid - pair_col_offset[p]``.
    The contact range is ``[pair_first[p] + k*6, pair_first[p] +
    k*6 + slot_count)`` and the active mask is ``(1 << slot_count)
    - 1``. Contacts live at ``cid in [cid_base, cid_base +
    num_contact_columns)``.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    cid = cid_base + tid

    p = pair_source_idx[tid]
    col_in_pair = tid - pair_col_offset[p]
    total_in_pair = pair_count[p]
    start_contact = pair_first[p] + col_in_pair * CONTACT_MAX_SLOTS

    # Number of active slots in *this* column. Full 6 for interior
    # columns; the tail column gets what's left.
    remaining = total_in_pair - col_in_pair * CONTACT_MAX_SLOTS
    slot_count = remaining
    if slot_count > CONTACT_MAX_SLOTS:
        slot_count = CONTACT_MAX_SLOTS
    active_mask = (wp.int32(1) << slot_count) - wp.int32(1)

    # Resolve shape -> body once; both shapes within a pair map to
    # the same two bodies for every contact in the pair by
    # construction.
    sa = pair_shape_a[p]
    sb = pair_shape_b[p]
    b1 = shape_body[sa]
    b2 = shape_body[sb]

    # Header.
    write_int(constraints, CONSTRAINT_TYPE_OFFSET, cid, CONSTRAINT_TYPE_CONTACT)
    write_int(constraints, CONSTRAINT_BODY1_OFFSET, cid, b1)
    write_int(constraints, CONSTRAINT_BODY2_OFFSET, cid, b2)

    # Contact-specific header bits.
    contact_set_friction(constraints, cid, default_friction)
    contact_set_active_mask(constraints, cid, active_mask)
    contact_set_contact_first(constraints, cid, start_contact)


# ---------------------------------------------------------------------------
# Warm-start gather kernel.
# ---------------------------------------------------------------------------


@wp.kernel
def _contact_warmstart_gather_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_col_offset: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    rigid_contact_match_index: wp.array[wp.int32],
    prev_slot_of_contact: wp.array[wp.int32],
    prev_cid_of_contact: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    cid_base: wp.int32,
    cc: ContactContainer,
):
    """Seed this frame's ``cc.lambdas`` slots from the prev frame.

    For each active slot of each current-frame contact column:

    * The contact's sorted-buffer index ``k`` is
      ``pair_first[p] + col_in_pair * 6 + slot``.
    * ``prev_k = rigid_contact_match_index[k]`` is the index in the
      *previous* frame's sorted buffer that the matcher paired us
      with, or :data:`MATCH_NOT_FOUND` / :data:`MATCH_BROKEN`
      (both < 0).
    * The prev-frame ingest stored, at sorted-buffer index ``prev_k``,
      the ``(slot, cid)`` that held that contact's impulses. Look it
      up via ``(prev_slot_of_contact, prev_cid_of_contact)`` which
      was populated by the *previous* step's
      :func:`stamp_forward_contact_map`. When both return a valid
      (``>= 0``) previous location, copy the three lambdas across;
      otherwise zero-init the slot (cold warm-start).
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    cid = cid_base + tid

    p = pair_source_idx[tid]
    col_in_pair = tid - pair_col_offset[p]
    total_in_pair = pair_count[p]
    remaining = total_in_pair - col_in_pair * CONTACT_MAX_SLOTS
    slot_count = remaining
    if slot_count > CONTACT_MAX_SLOTS:
        slot_count = CONTACT_MAX_SLOTS
    start_contact = pair_first[p] + col_in_pair * CONTACT_MAX_SLOTS

    for s in range(CONTACT_MAX_SLOTS):
        if s >= slot_count:
            # Inactive slot in this column -- clear so stale data
            # from a previous step doesn't leak into the prepare
            # kernel when it sums the warm-start impulse.
            cc_set_normal_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent1_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent2_lambda(cc, s, cid, wp.float32(0.0))
            continue
        k = start_contact + s
        prev_k = rigid_contact_match_index[k]
        if prev_k < 0:
            cc_set_normal_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent1_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent2_lambda(cc, s, cid, wp.float32(0.0))
            continue

        prev_slot = prev_slot_of_contact[prev_k]
        prev_cid = prev_cid_of_contact[prev_k]
        if prev_slot < 0 or prev_cid < 0:
            cc_set_normal_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent1_lambda(cc, s, cid, wp.float32(0.0))
            cc_set_tangent2_lambda(cc, s, cid, wp.float32(0.0))
            continue

        cc_set_normal_lambda(cc, s, cid, cc_get_prev_normal_lambda(cc, prev_slot, prev_cid))
        cc_set_tangent1_lambda(cc, s, cid, cc_get_prev_tangent1_lambda(cc, prev_slot, prev_cid))
        cc_set_tangent2_lambda(cc, s, cid, cc_get_prev_tangent2_lambda(cc, prev_slot, prev_cid))


@wp.kernel
def _reset_forward_map_kernel(
    # out
    slot_of_contact: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
):
    """Clear the forward map to ``-1`` before every stamp pass.

    Separate tiny kernel so the caller doesn't need a graph-
    breaking ``fill_`` between steps.
    """
    tid = wp.tid()
    slot_of_contact[tid] = wp.int32(-1)
    cid_of_contact[tid] = wp.int32(-1)


@wp.kernel
def _stamp_slot_cid_of_contact_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_col_offset: wp.array[wp.int32],
    pair_first: wp.array[wp.int32],
    pair_count: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    cid_base: wp.int32,
    # out
    slot_of_contact: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
):
    """For each contact in this frame's sorted buffer, stamp ``(slot, cid)``.

    This is the "forward" map the *next* frame will consult when it
    swaps ``(slot_of_contact, cid_of_contact)`` into the "prev" role
    and looks up its own ``rigid_contact_match_index[k] = prev_k``.
    Written during ingest once the column layout is known.

    Entries not covered by any column keep the ``-1`` written by
    :func:`_reset_forward_map_kernel`.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return
    p = pair_source_idx[tid]
    col_in_pair = tid - pair_col_offset[p]
    total_in_pair = pair_count[p]
    remaining = total_in_pair - col_in_pair * CONTACT_MAX_SLOTS
    slot_count = remaining
    if slot_count > CONTACT_MAX_SLOTS:
        slot_count = CONTACT_MAX_SLOTS
    start_contact = pair_first[p] + col_in_pair * CONTACT_MAX_SLOTS
    cid = cid_base + tid
    for s in range(slot_count):
        slot_of_contact[start_contact + s] = s
        cid_of_contact[start_contact + s] = cid


# ---------------------------------------------------------------------------
# Host driver.
# ---------------------------------------------------------------------------


def ingest_contacts(
    contacts,  # newton._src.sim.contacts.Contacts
    shape_body: wp.array,
    num_shapes: int,
    constraints: ConstraintContainer,
    scratch: IngestScratch,
    cid_base: int,
    max_contact_columns: int,
    default_friction: float = 0.5,
    device: wp.DeviceLike = None,
) -> None:
    """Materialise contact columns for one step.

    Graph-capture safe: no host readbacks, all kernel launches have
    sizes known at host time (``rigid_contact_max`` /
    ``max_contact_columns``), and all step-varying counts are
    kept on-device in :attr:`IngestScratch.num_pairs` /
    :attr:`IngestScratch.num_contact_columns`.

    After the call, ``scratch.num_contact_columns[0]`` holds the
    number of columns emitted; the caller reads this *outside* the
    captured region (e.g. right before the next step) to resize
    the partitioner's active length.

    Args:
        contacts: Newton :class:`Contacts` buffer. Must have been
            built with ``contact_matching=True``.
        shape_body: ``model.shape_body`` array.
        num_shapes: Total shape count in the owning Newton model;
            only used to pack the (shape_a, shape_b) key into int32.
        constraints: Shared constraint storage. Contact columns are
            written at ``cid in [cid_base, cid_base +
            num_contact_columns)``.
        scratch: Reusable per-step scratch.
        cid_base: First cid in the container reserved for contacts.
        max_contact_columns: Hard cap on the number of contact
            columns this step can emit. Excess contacts are
            silently dropped by the final clamp.
        default_friction: Friction coefficient written into every
            contact column header. Per-pair lookups into the
            material tables are a future improvement.
        device: Warp device for the launches.
    """
    rigid_contact_max = int(contacts.rigid_contact_max)

    # Step 1: per-contact key.
    wp.launch(
        kernel=_contact_key_kernel,
        dim=rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            int(num_shapes),
        ],
        outputs=[scratch.keys],
        device=device,
    )

    # Step 2: graph-capture-safe RLE. Writes run_values into
    # scratch.run_values, run_lengths into scratch.pair_count, and
    # total run count into scratch.num_pairs.
    runlength_encode_variable_length(
        values=scratch.keys,
        active_length=contacts.rigid_contact_count,
        run_values=scratch.run_values,
        run_lengths=scratch.pair_count,
        run_count=scratch.num_pairs,
        sentinel=RLE_SENTINEL_INT32,
    )

    # Step 3a: unpack run_values -> pair_{shape_a, shape_b},
    # clear pair_columns tail.
    wp.launch(
        kernel=_pair_metadata_kernel,
        dim=rigid_contact_max,
        inputs=[scratch.run_values, scratch.num_pairs, int(num_shapes)],
        outputs=[scratch.pair_shape_a, scratch.pair_shape_b, scratch.pair_columns],
        device=device,
    )

    # Step 3b: pair_columns[p] = ceil(pair_count[p] / 6).
    wp.launch(
        kernel=_pair_columns_from_count_kernel,
        dim=rigid_contact_max,
        inputs=[scratch.pair_count, scratch.num_pairs],
        outputs=[scratch.pair_columns],
        device=device,
    )

    # Step 3c: exclusive scan pair_count -> pair_first.
    wp.launch(
        kernel=_pair_first_kernel,
        dim=1,
        inputs=[scratch.pair_count, scratch.num_pairs],
        outputs=[scratch.pair_first],
        device=device,
    )

    # Step 3d: exclusive scan of pair_columns -> pair_col_offset.
    wp.utils.array_scan(scratch.pair_columns, scratch.pair_col_offset, inclusive=False)

    # Step 3e: reduce to the total column count on-device.
    wp.launch(
        kernel=_total_contact_columns_kernel,
        dim=1,
        inputs=[
            scratch.pair_col_offset,
            scratch.pair_columns,
            scratch.num_pairs,
            int(max_contact_columns),
        ],
        outputs=[scratch.num_contact_columns],
        device=device,
    )

    # Step 4: build the per-column -> pair map.
    wp.launch(
        kernel=_pair_source_idx_kernel,
        dim=1,
        inputs=[
            scratch.pair_col_offset,
            scratch.pair_columns,
            scratch.num_pairs,
            int(max_contact_columns),
        ],
        outputs=[scratch.pair_source_idx],
        device=device,
    )

    # Step 5: write the contact column headers + ranges. Launches
    # at max_contact_columns and gates internally on
    # num_contact_columns[0].
    wp.launch(
        kernel=_contact_pack_columns_kernel,
        dim=max(1, max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_col_offset,
            scratch.pair_first,
            scratch.pair_count,
            scratch.pair_shape_a,
            scratch.pair_shape_b,
            shape_body,
            scratch.num_contact_columns,
            int(cid_base),
            float(default_friction),
        ],
        outputs=[constraints],
        device=device,
    )


def stamp_forward_contact_map(
    rigid_contact_max: int,
    cid_base: int,
    scratch: IngestScratch,
    slot_of_contact: wp.array,
    cid_of_contact: wp.array,
    device: wp.DeviceLike = None,
) -> None:
    """Fill the forward (sorted-index -> (slot, cid)) lookup.

    Called after :func:`ingest_contacts`; the two output arrays are
    consumed by the *next* step's warm-start gather as the "prev"
    side of the match map. Graph-capture safe: resets the entire
    map (size ``rigid_contact_max``) on every call, then stamps
    active slots gated on ``scratch.num_contact_columns[0]``.
    """
    # Reset the full table to -1.
    wp.launch(
        kernel=_reset_forward_map_kernel,
        dim=rigid_contact_max,
        inputs=[],
        outputs=[slot_of_contact, cid_of_contact],
        device=device,
    )
    # Stamp the active slots.
    wp.launch(
        kernel=_stamp_slot_cid_of_contact_kernel,
        dim=max(1, scratch.max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_col_offset,
            scratch.pair_first,
            scratch.pair_count,
            scratch.num_contact_columns,
            int(cid_base),
        ],
        outputs=[slot_of_contact, cid_of_contact],
        device=device,
    )


def gather_contact_warmstart(
    cid_base: int,
    scratch: IngestScratch,
    rigid_contact_match_index: wp.array,
    prev_slot_of_contact: wp.array,
    prev_cid_of_contact: wp.array,
    cc: ContactContainer,
    device: wp.DeviceLike = None,
) -> None:
    """Copy prev-frame lambdas into ``cc.*_lambda`` for the new columns.

    Called after the pointer-swap (``cc.prev_*`` now holds last
    step's lambdas; ``cc.*_lambda`` is scratch) but before
    :func:`contact_prepare_for_iteration_at`.
    """
    wp.launch(
        kernel=_contact_warmstart_gather_kernel,
        dim=max(1, scratch.max_contact_columns),
        inputs=[
            scratch.pair_source_idx,
            scratch.pair_col_offset,
            scratch.pair_first,
            scratch.pair_count,
            rigid_contact_match_index,
            prev_slot_of_contact,
            prev_cid_of_contact,
            scratch.num_contact_columns,
            int(cid_base),
        ],
        outputs=[cc],
        device=device,
    )
