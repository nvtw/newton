# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of PhoenX's ``UnionFindIslandBuilder``.

Direct translation of
``experimentalsim/PhoenX/src/PhoenX/UnionFindIslandBuilderHost.cs`` plus
``UnionFindIslandBuilder.cs`` and the accompanying CUDA kernels in
``PhoenX/CudaKernels/Islands/``. A union-find (disjoint-set) data
structure with path-compression + union-by-rank runs concurrently over
every constraint's body pair to group bodies into connected components
("islands"). The port preserves PhoenX's deterministic output path:

1. Atomic union-find populates the parent/rank table
   (:func:`_island_unite_kernel`).
2. :func:`_island_compute_set_nrs_kernel` pins every body to the
   representative of its component and bumps the per-representative
   size + atomically lowers the per-representative minimum body index.
3. A 1/0 bitmap of "is this representative used?" + exclusive scan
   (via :func:`wp.utils.array_scan`) produces a compact set ordering
   ``oldToNew``.
4. The compact ordering is re-sorted by *min body index* (via
   :func:`wp.utils.radix_sort_pairs`) so two runs over the same inputs
   always produce the same canonical compact ids -- this is the
   deterministic-ordering trick; the union-find's atomic races only
   affect which body wins the representative slot, not which islands
   exist.
5. An invert-map pass rewrites ``minIndexPerSetCompact`` into a
   ``oldCompact -> newCompact`` permutation.
6. :func:`_island_rewrite_set_nrs_kernel` rewrites every body's set id
   into its final deterministic index, ``setSizesCompact`` is
   inclusive-scanned into island end-offsets, and a final sort groups
   body ids by island so :meth:`UnionFindIslandBuilder.get_island`
   returns a contiguous slice.

The storage layout matches PhoenX's ``UnionFindIslandBuilderGpu``
one-for-one so future integrations can follow the same host-side
recipe. Kept isolated from the solver (no import from
:mod:`solver_jitter`); constraints plug in externally by passing their
own ``wp.array[ElementInteractionData]`` to :meth:`build_islands`.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.helpers.scan_and_sort import (
    scan_variable_length,
    sort_variable_length_int,
)

__all__ = [
    "INVALID_PARENT",
    "MAX_BODIES_PER_INTERACTION",
    "UnionFindIslandBuilder",
]


#: Maximum number of bodies a single interaction row can reference.
#: Matches the vec8i layout used by the graph-colouring partitioner's
#: :class:`ElementInteractionData` so callers can hand their existing
#: element buffers straight in as a ``wp.array2d[wp.int32]`` view --
#: the builder only needs a (num_interactions, 8) int32 table of
#: body ids with ``-1`` sentinels in unused slots.
MAX_BODIES_PER_INTERACTION = 8


#: Sentinel parent value meaning "this slot has never been touched" --
#: lets :func:`_find` short-circuit without following an un-initialised
#: pointer. PhoenX uses ``0xFFFFFFFFu`` as the sentinel; we match it
#: exactly so ports lining up against the reference kernel are easy.
INVALID_PARENT = wp.constant(wp.int32(-1))


# ---------------------------------------------------------------------------
# Packed entries layout
# ---------------------------------------------------------------------------
#
# Each entry is a 64-bit word laid out as:
#
#   bits  0..31 : parent body id (as uint32)
#   bits 32..62 : rank (positive, 31 bits enough for realistic scenes)
#   bit      63 : unused (kept 0 so the whole word remains positive
#                 when interpreted as an int64)
#
# The high-bit-clear invariant lets us store the packed entry as
# ``wp.int64`` without worrying about sign; the only comparisons we do
# on entries are equality (for the CAS), which is sign-insensitive.

_RANK_SHIFT = wp.constant(wp.int64(32))
_PARENT_MASK = wp.constant(wp.int64(0xFFFFFFFF))
_RANK_MASK_IN_ENTRY = wp.constant(wp.int64(0x7FFFFFFF00000000))
_RANK_LOW_MASK = wp.constant(wp.int64(0x7FFFFFFF))
_INT32_MAX = wp.constant(wp.int32(0x7FFFFFFF))


@wp.func
def _pack_entry(parent: wp.int32, rank: wp.int32) -> wp.int64:
    """Pack ``(parent, rank)`` into a single int64 word.

    Both operands are promoted to ``wp.int64`` *before* the shift /
    mask so Warp's "same-type" constraint on bitwise operators is
    satisfied (``int64 << int64``, not ``int64 << int32``). The
    ``& _PARENT_MASK`` on ``parent`` is what turns a negative int32
    into its unsigned low-32-bits bit pattern -- relevant only for
    defensiveness since the init path stamps non-negative ids.
    """
    return (wp.int64(rank) << _RANK_SHIFT) | (wp.int64(parent) & _PARENT_MASK)


@wp.func
def _entry_parent(entry: wp.int64) -> wp.int32:
    """Low 32 bits of an entry -> parent body id."""
    return wp.int32(entry & _PARENT_MASK)


@wp.func
def _entry_rank(entry: wp.int64) -> wp.int32:
    """High 31 bits of an entry -> rank (always non-negative)."""
    return wp.int32((entry >> _RANK_SHIFT) & _RANK_LOW_MASK)


# ---------------------------------------------------------------------------
# Union-find device funcs (wjakob/dset port)
# ---------------------------------------------------------------------------
#
# Lock-free find-with-path-compression and union-by-rank from
# https://github.com/wjakob/dset (the same algorithm PhoenX uses).
#
# * ``_find`` chases the parent chain and, where it sees an outdated
#   parent pointer, tries to compress the path with a CAS. A failed CAS
#   is fine -- someone else has compressed the same edge -- so we just
#   fall through to the next iteration.
# * ``_unite`` enforces union-by-rank: the lower-rank root is linked
#   under the higher-rank one. Ties break toward the smaller body id
#   so the tree topology is deterministic given a fixed interaction
#   ordering (atomics may still race on *which* Unite op wins, but the
#   result of the merge is the same either way).


@wp.func
def _find(entries: wp.array[wp.int64], body_id: wp.int32) -> wp.int32:
    """Return the representative of ``body_id``'s component.

    Walks parent pointers until a fixed point (``parent == self``),
    opportunistically compressing along the way. Matches PhoenX's
    ``UnionFindIslandBuilderGpu::Find``.
    """
    id_ = body_id
    while True:
        value = entries[id_]
        parent = _entry_parent(value)
        if parent == id_:
            return id_
        new_parent = _entry_parent(entries[parent])
        if new_parent == INVALID_PARENT:
            return wp.int32(-1)
        new_value = (value & _RANK_MASK_IN_ENTRY) | (wp.int64(new_parent) & _PARENT_MASK)
        if value != new_value:
            # Best-effort path compression. A failed CAS just means
            # another thread got there first; we proceed with the
            # (already-walked) new_parent either way.
            wp.atomic_cas(entries, id_, value, new_value)
        id_ = new_parent


@wp.func
def _unite(entries: wp.array[wp.int64], a: wp.int32, b: wp.int32):
    """Link the components of ``a`` and ``b`` with union-by-rank.

    Atomic CAS on ``entries`` guarantees the link is race-free under
    concurrent ``_unite`` calls. A failed CAS retries from the top
    because ``_find`` may have raced against another thread's path
    compression; the retry re-reads the latest roots before attempting
    the merge again. Matches PhoenX's
    ``UnionFindIslandBuilderGpu::Unite``.
    """
    id1 = a
    id2 = b
    while True:
        id1 = _find(entries, id1)
        id2 = _find(entries, id2)
        if id1 < 0 or id2 < 0:
            return
        if id1 == id2:
            return

        r1 = _entry_rank(entries[id1])
        r2 = _entry_rank(entries[id2])

        # Link the lower-rank root under the higher-rank one. Ties
        # resolved toward the smaller root id -- the PhoenX convention,
        # which also keeps the tree shape insensitive to the arbitrary
        # "which body did the caller pass first" order.
        if r1 > r2 or (r1 == r2 and id1 < id2):
            tmp_r = r1
            r1 = r2
            r2 = tmp_r
            tmp_id = id1
            id1 = id2
            id2 = tmp_id

        old_entry = _pack_entry(id1, r1)
        new_entry = _pack_entry(id2, r1)
        prev = wp.atomic_cas(entries, id1, old_entry, new_entry)
        if prev != old_entry:
            continue  # another thread raced us, retry

        if r1 == r2:
            # Bump the destination's rank. Best-effort -- a failing CAS
            # here just means another concurrent Unite already bumped
            # it to the same value or beyond.
            old_entry = _pack_entry(id2, r2)
            new_entry = _pack_entry(id2, r2 + wp.int32(1))
            wp.atomic_cas(entries, id2, old_entry, new_entry)
        return


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
#
# All kernels follow the same ``tid >= active_length[0]`` early-out
# pattern the rest of the solver uses so launch sizes can be fixed on
# the host and graph-capture safe.


@wp.kernel(enable_backward=False)
def _island_init_kernel(
    num_bodies: wp.array[wp.int32],
    # out
    entries: wp.array[wp.int64],
    set_nr: wp.array[wp.int32],
    old_to_new: wp.array[wp.int32],
    set_sizes: wp.array[wp.int32],
    set_sizes_compact: wp.array[wp.int32],
    min_index_per_set: wp.array[wp.int32],
):
    """Seed every body as its own singleton component.

    ``entries[i] = (rank = 0, parent = i)``. The other per-body arrays
    are cleared to the sentinels they need for the subsequent atomic
    accumulate / scan passes.
    """
    tid = wp.tid()
    if tid >= num_bodies[0]:
        return
    entries[tid] = _pack_entry(tid, wp.int32(0))
    set_nr[tid] = wp.int32(-1)
    old_to_new[tid] = wp.int32(-1)
    set_sizes[tid] = wp.int32(0)
    set_sizes_compact[tid] = wp.int32(0)
    min_index_per_set[tid] = _INT32_MAX


@wp.kernel(enable_backward=False)
def _island_unite_kernel(
    interaction_bodies: wp.array2d[wp.int32],
    num_interactions: wp.array[wp.int32],
    # inout
    entries: wp.array[wp.int64],
):
    """Walk every constraint's body list and union-find consecutive pairs.

    Same chain-based union as PhoenX's ``IslandUniteKernel``: for an
    interaction with bodies ``[b0, b1, ..., bk]`` we union ``(b0, b1)``,
    ``(b1, b2)``, ..., ``(bk-1, bk)``. Inactive slots (``-1``) end the
    chain.

    ``interaction_bodies`` is a ``(capacity, 8)`` int32 table of body
    ids -- the caller packs one row per interaction with ``-1`` in
    any unused slot. Chosen over taking a struct type because Warp
    struct visibility is module-scoped; the flat array lets callers
    hand in their existing buffers (e.g. the graph-colouring
    partitioner's :class:`ElementInteractionData` -- see the test
    helper) without the builder depending on that module.
    """
    tid = wp.tid()
    if tid >= num_interactions[0]:
        return
    prev = interaction_bodies[tid, 0]
    if prev < 0:
        return
    for j in range(1, MAX_BODIES_PER_INTERACTION):
        nxt = interaction_bodies[tid, j]
        if nxt < 0:
            return
        _unite(entries, prev, nxt)
        prev = nxt


@wp.kernel(enable_backward=False)
def _island_compute_set_nrs_kernel(
    num_bodies: wp.array[wp.int32],
    entries: wp.array[wp.int64],
    # out
    set_nr: wp.array[wp.int32],
    set_sizes: wp.array[wp.int32],
    min_index_per_set: wp.array[wp.int32],
):
    """Resolve every body's representative + tally per-representative stats.

    For each body ``i``:

    * ``set_nr[i] = find(i)`` -- the raw (pre-compaction) island id.
    * ``atomic_add(set_sizes[nr], 1)`` -- running count of bodies per
      representative. Used by :meth:`make_compact` as the raw-set size.
    * ``atomic_min(min_index_per_set[nr], i)`` -- smallest body id in
      each component. This is the deterministic-ordering key PhoenX
      uses to sort islands into a canonical order.
    """
    tid = wp.tid()
    if tid >= num_bodies[0]:
        return
    nr = _find(entries, tid)
    if nr < 0:
        return
    set_nr[tid] = nr
    wp.atomic_add(set_sizes, nr, wp.int32(1))
    wp.atomic_min(min_index_per_set, nr, tid)


@wp.kernel(enable_backward=False)
def _island_prepare_scan_kernel(
    num_bodies: wp.array[wp.int32],
    set_sizes: wp.array[wp.int32],
    # out
    old_to_new: wp.array[wp.int32],
):
    """Stamp 1/0 into ``old_to_new[i]`` so an exclusive scan gives a
    compact ``rawRepId -> compactRepId`` map on the active prefix.
    Anything past ``num_bodies[0]`` stays at its initial zero."""
    tid = wp.tid()
    if tid >= num_bodies[0]:
        return
    if set_sizes[tid] != 0:
        old_to_new[tid] = wp.int32(1)
    else:
        old_to_new[tid] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _island_reduce_num_sets_kernel(
    num_bodies: wp.array[wp.int32],
    old_to_new: wp.array[wp.int32],
    set_sizes: wp.array[wp.int32],
    # out
    num_sets: wp.array[wp.int32],
):
    """Single-thread reduction of the ``old_to_new`` inclusive-scan tail
    into ``num_sets[0]`` = total number of occupied representatives.

    ``old_to_new`` is exclusive-scanned right before this kernel, so the
    total count is ``old_to_new[n-1] + (1 if set_sizes[n-1] != 0 else 0)``
    where ``n = num_bodies[0]``. Splitting this out keeps
    ``scan_variable_length`` stateless (it doesn't know how to produce
    a total; :func:`wp.utils.array_scan` writes an exclusive prefix
    and nothing else).
    """
    tid = wp.tid()
    if tid != 0:
        return
    n = num_bodies[0]
    if n <= 0:
        num_sets[0] = wp.int32(0)
        return
    last_flag = wp.int32(0)
    if set_sizes[n - 1] != 0:
        last_flag = wp.int32(1)
    num_sets[0] = old_to_new[n - 1] + last_flag


@wp.kernel(enable_backward=False)
def _island_collect_indices_kernel(
    num_bodies: wp.array[wp.int32],
    num_sets: wp.array[wp.int32],
    set_sizes: wp.array[wp.int32],
    old_to_new: wp.array[wp.int32],
    min_index_per_set: wp.array[wp.int32],
    # out
    min_index_per_set_compact: wp.array[wp.int32],
):
    """Pack raw per-representative min-indices down into the compact
    order produced by the scan, and seed ``min_index_per_set[0..numSets)``
    as the identity ``i`` so the subsequent sort keeps both arrays in
    lockstep.

    Mirrors PhoenX's ``IslandCollectIndicesKernel``. The dual-purpose
    ``min_index_per_set`` reuse -- data-carrier first, sort-indices
    second -- matches the reference so the memory footprint stays the
    same.
    """
    tid = wp.tid()
    if tid >= num_bodies[0]:
        return
    if set_sizes[tid] != 0:
        output_index = old_to_new[tid]
        min_index_per_set_compact[output_index] = min_index_per_set[tid]
    if tid < num_sets[0]:
        min_index_per_set[tid] = tid


@wp.kernel(enable_backward=False)
def _island_invert_map_kernel(
    num_sets: wp.array[wp.int32],
    min_index_per_set: wp.array[wp.int32],
    # inout: written at min_index_per_set[i] using tid as source
    # (the pre-sort we call below leaves ``min_index_per_set`` as
    # "for compact slot i, what raw compact slot did it come from");
    # inverting it gives us ``rawCompact -> newCompact``.
    min_index_per_set_compact: wp.array[wp.int32],
):
    """Invert the sort's value permutation.

    After the sort ``min_index_per_set[i] = old_compact_slot``. We want
    ``min_index_per_set_compact[old_compact_slot] = new_compact_slot``
    so the next rewrite pass can look up each body's final island id
    with a single load.
    """
    tid = wp.tid()
    if tid >= num_sets[0]:
        return
    min_index_per_set_compact[min_index_per_set[tid]] = tid


@wp.kernel(enable_backward=False)
def _island_rewrite_set_nrs_kernel(
    num_bodies: wp.array[wp.int32],
    set_sizes: wp.array[wp.int32],
    old_to_new: wp.array[wp.int32],
    min_index_per_set_compact: wp.array[wp.int32],
    # out
    set_nr: wp.array[wp.int32],
    set_sizes_compact: wp.array[wp.int32],
):
    """PhoenX ``IslandMakeCompactStage1Kernel``: rewrite raw set ids
    into the sorted-by-min-index compact ordering and plant the
    compact-sizes scan seed.

    ``set_sizes_compact[new_nr]`` is set at most once per body; the
    guard avoids a redundant write when multiple bodies from the same
    island fight for the same slot (the data race is benign because
    every contender writes the same value, but the guard skips the
    store entirely for all but the first).
    """
    tid = wp.tid()
    if tid >= num_bodies[0]:
        return
    nr = set_nr[tid]
    new_nr = min_index_per_set_compact[old_to_new[nr]]
    set_nr[tid] = new_nr
    if set_sizes_compact[new_nr] != set_sizes[nr]:
        set_sizes_compact[new_nr] = set_sizes[nr]


@wp.kernel(enable_backward=False)
def _island_stage2_kernel(
    num_bodies: wp.array[wp.int32],
    set_nr: wp.array[wp.int32],
    # out: prepare the final "sort bodies by island id" pass
    set_sizes: wp.array[wp.int32],
    old_to_new: wp.array[wp.int32],
):
    """PhoenX ``IslandMakeCompactStage2Kernel``: seed the final
    ``set_sizes`` array with identity body indices and copy the new set
    id into ``old_to_new`` so we can feed ``(old_to_new, set_sizes)``
    into a stable key-value sort that groups body ids per island.
    """
    tid = wp.tid()
    if tid >= num_bodies[0]:
        return
    set_sizes[tid] = tid
    old_to_new[tid] = set_nr[tid]


class UnionFindIslandBuilder:
    """Warp port of PhoenX's :class:`UnionFindIslandBuilder`.

    Build connected components over bodies linked by
    :class:`ElementInteractionData` chains.

    Usage::

        builder = UnionFindIslandBuilder(num_bodies_capacity=N, device=dev)
        builder.build_islands(interactions, num_interactions, num_bodies)
        num_islands = int(builder.num_sets.numpy()[0])
        # Per-body island id, body ids grouped by island,
        # inclusive ends per island:
        island_of_body = builder.set_nr.numpy()
        body_ids = builder.set_sizes.numpy()
        ends = builder.set_sizes_compact.numpy()[:num_islands]

    Deterministic despite the race-happy atomic union-find:
    post-processing sorts island ids by each island's smallest body
    id (atomic-min + sort-by-min-index + invert-map) so the
    O(N * alpha(N)) asymptotic cost is preserved.

    All device buffers are allocated at construction; subsequent
    :meth:`build_islands` calls are fixed-size and graph-capture safe.
    """

    def __init__(
        self,
        num_bodies_capacity: int,
        device: wp.context.Devicelike = None,
    ) -> None:
        """Pre-allocate every device buffer the build needs.

        Args:
            num_bodies_capacity: Upper bound on the number of bodies
                any future :meth:`build_islands` call will reference.
                Every per-body buffer allocates ``2 * capacity``
                entries to satisfy the radix-sort ping-pong buffer
                requirement -- same convention as the helpers in
                :mod:`scan_and_sort`.
            device: Warp device. ``None`` takes the current default.
        """
        if num_bodies_capacity <= 0:
            raise ValueError(f"num_bodies_capacity must be > 0 (got {num_bodies_capacity})")
        self._capacity: int = int(num_bodies_capacity)
        self._device = wp.get_device(device)

        cap = self._capacity
        # Radix-sort ping-pong: ``sort_variable_length_int`` treats
        # ``keys.shape[0] // 2`` as the sortable count. We want to sort
        # up to ``capacity`` entries so every sort-scratch array is
        # sized ``2 * capacity``.
        sort_cap = 2 * cap

        self.entries: wp.array[wp.int64] = wp.zeros(cap, dtype=wp.int64, device=self._device)
        self.set_nr: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self._device)
        # ``set_sizes`` is reused as the body-id carrier for the final
        # sort. The sort wants a 2x ping-pong buffer so we match.
        self.set_sizes: wp.array[wp.int32] = wp.zeros(sort_cap, dtype=wp.int32, device=self._device)
        self.set_sizes_compact: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self._device)
        # ``old_to_new`` is the island-id carrier for the final sort
        # (swaps role with ``set_sizes``); same 2x ping-pong sizing.
        self.old_to_new: wp.array[wp.int32] = wp.zeros(sort_cap, dtype=wp.int32, device=self._device)
        self.num_sets: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self._device)
        self.min_index_per_set: wp.array[wp.int32] = wp.zeros(sort_cap, dtype=wp.int32, device=self._device)
        self.min_index_per_set_compact: wp.array[wp.int32] = wp.zeros(sort_cap, dtype=wp.int32, device=self._device)

    @property
    def capacity(self) -> int:
        """Maximum number of bodies the builder can handle."""
        return self._capacity

    def build_islands(
        self,
        interaction_bodies: wp.array2d[wp.int32],
        num_interactions: wp.array[wp.int32],
        num_bodies: wp.array[wp.int32],
    ) -> None:
        """Run the full PhoenX pipeline end-to-end on the device.

        Flow (per-kernel reads/writes documented on each helper):

        1. ``_island_init_kernel`` seeds sentinels.
        2. ``_island_unite_kernel`` chains bodies per interaction.
        3. ``_island_compute_set_nrs_kernel`` pins each body to its
           representative + accumulates per-rep stats.
        4. Compact ordering (``old_to_new`` bitmap -> exclusive scan
           -> reduce to ``num_sets``).
        5. ``_island_collect_indices_kernel`` packs min-indices.
        6. Sort by compact min-index.
        7. ``_island_invert_map_kernel`` -> ``rawCompact ->
           sortedCompact`` permutation.
        8. ``_island_rewrite_set_nrs_kernel`` rewrites per-body
           island ids + populates ``set_sizes_compact``.
        9. Inclusive scan -> island-end offsets.
        10. ``_island_stage2_kernel`` + key-value sort group body
            ids by island.

        Args:
            interaction_bodies: ``(>= num_interactions[0], 8)``; row
                ``k`` lists bodies connected by interaction ``k``,
                unused slots ``-1``.
            num_interactions: Device scalar, active interaction count.
            num_bodies: Device scalar, active body count. Every
                launch targets ``capacity`` threads and gates on this.
        """
        cap = self._capacity

        # Step 1: init per-body sentinels + entries.
        wp.launch(
            _island_init_kernel,
            dim=cap,
            inputs=[num_bodies],
            outputs=[
                self.entries,
                self.set_nr,
                self.old_to_new,
                self.set_sizes,
                self.set_sizes_compact,
                self.min_index_per_set,
            ],
            device=self._device,
        )

        # Step 2: atomic union-find over every interaction's body list.
        # Launch dim is the interactions buffer's leading extent -- the
        # caller sizes that to their own capacity (usually >> the body
        # capacity because there are many more constraints than bodies).
        wp.launch(
            _island_unite_kernel,
            dim=interaction_bodies.shape[0],
            inputs=[interaction_bodies, num_interactions],
            outputs=[self.entries],
            device=self._device,
        )

        # Step 3: resolve representatives + per-rep stats.
        wp.launch(
            _island_compute_set_nrs_kernel,
            dim=cap,
            inputs=[num_bodies, self.entries],
            outputs=[self.set_nr, self.set_sizes, self.min_index_per_set],
            device=self._device,
        )

        # Step 4a: bitmap of occupied representatives.
        wp.launch(
            _island_prepare_scan_kernel,
            dim=cap,
            inputs=[num_bodies, self.set_sizes],
            outputs=[self.old_to_new],
            device=self._device,
        )

        # Step 4b: exclusive scan of ``old_to_new`` -> compact indices.
        # ``scan_variable_length`` is a thin wrapper over
        # :func:`wp.utils.array_scan`; it works on the full buffer and
        # the inactive-tail entries we already zeroed in
        # :func:`_island_init_kernel` cleanly fold into the scan.
        scan_variable_length(self.old_to_new, num_bodies, inclusive=False)

        # Step 4c: derive ``num_sets[0]`` from the scan tail.
        wp.launch(
            _island_reduce_num_sets_kernel,
            dim=1,
            inputs=[num_bodies, self.old_to_new, self.set_sizes],
            outputs=[self.num_sets],
            device=self._device,
        )

        # Step 5: pack per-rep min-indices into the compact order and
        # seed the sort value array with the identity permutation.
        wp.launch(
            _island_collect_indices_kernel,
            dim=cap,
            inputs=[
                num_bodies,
                self.num_sets,
                self.set_sizes,
                self.old_to_new,
                self.min_index_per_set,
            ],
            outputs=[self.min_index_per_set_compact],
            device=self._device,
        )

        # Step 6: sort compact slots by their min-index so the final
        # island ids are deterministic regardless of union-find atomic
        # scheduling. ``sort_variable_length_int`` masks the inactive
        # tail with ``INT32_MAX`` and runs :func:`radix_sort_pairs`
        # on the full buffer, so we pass ``num_sets`` (not
        # ``num_bodies``) as the active length.
        sort_variable_length_int(
            keys=self.min_index_per_set_compact,
            values=self.min_index_per_set,
            active_length=self.num_sets,
        )

        # Step 7: invert the sort permutation so
        # ``min_index_per_set_compact[rawCompact] = newCompact``.
        wp.launch(
            _island_invert_map_kernel,
            dim=cap,
            inputs=[self.num_sets, self.min_index_per_set],
            outputs=[self.min_index_per_set_compact],
            device=self._device,
        )

        # Step 8: rewrite per-body set ids + populate compact sizes.
        wp.launch(
            _island_rewrite_set_nrs_kernel,
            dim=cap,
            inputs=[
                num_bodies,
                self.set_sizes,
                self.old_to_new,
                self.min_index_per_set_compact,
            ],
            outputs=[self.set_nr, self.set_sizes_compact],
            device=self._device,
        )

        # Step 9: inclusive scan of compact sizes -> per-island end
        # offsets. ``scan_variable_length`` operates in-place on the
        # full buffer; entries past ``num_sets[0]`` are zero (seeded
        # in init) so they contribute nothing to the scan total.
        scan_variable_length(self.set_sizes_compact, self.num_sets, inclusive=True)

        # Step 10: group body ids by island via a final key-value sort.
        # Key = island id (``old_to_new``, rewritten by stage 2),
        # value = body id (``set_sizes`` seeded with identity by
        # stage 2). radix sort is stable so bodies within an island
        # stay in ascending id order -- the property PhoenX documents
        # as needed for :meth:`get_island_lowest`.
        wp.launch(
            _island_stage2_kernel,
            dim=cap,
            inputs=[num_bodies, self.set_nr],
            outputs=[self.set_sizes, self.old_to_new],
            device=self._device,
        )
        sort_variable_length_int(
            keys=self.old_to_new,
            values=self.set_sizes,
            active_length=num_bodies,
        )

    # ------------------------------------------------------------------
    # Host-side convenience helpers
    # ------------------------------------------------------------------
    #
    # Pure Python, intended for inspection after a build. All of them
    # trigger a device->host copy; callers that want to stay on-device
    # should read :attr:`set_nr`, :attr:`set_sizes`, and
    # :attr:`set_sizes_compact` directly.

    def num_islands(self) -> int:
        """Number of islands produced by the last :meth:`build_islands`
        call. Performs a single ``int32`` device->host copy."""
        return int(self.num_sets.numpy()[0])

    def island_index_of_body(self, body_id: int) -> int:
        """Canonical island id the body was assigned to.

        Ids are in ``[0, num_islands())`` and are ordered by the
        smallest body id in each island (so
        ``island_index_of_body(min_id) == 0``). Triggers a device-to-
        host copy of the per-body ``set_nr`` array; prefer
        :attr:`set_nr` directly if you need many lookups.
        """
        return int(self.set_nr.numpy()[body_id])

    def get_island(self, island_index: int) -> list[int]:
        """Body ids belonging to island ``island_index``.

        Returned list is sorted ascending -- the radix sort at step 10
        is stable and the identity seed in ``set_sizes`` feeds it in
        ascending order, so ids come back sorted without a separate
        pass. Triggers a device->host copy of the compact offset
        table plus the grouped body id array.
        """
        ends = self.set_sizes_compact.numpy()
        n = self.num_islands()
        if not (0 <= island_index < n):
            raise IndexError(f"island_index {island_index} out of range [0, {n})")
        start = 0 if island_index == 0 else int(ends[island_index - 1])
        end = int(ends[island_index])
        bodies = self.set_sizes.numpy()[start:end]
        return [int(x) for x in bodies]

    def get_island_lowest(self, island_index: int) -> int:
        """Smallest body id in island ``island_index``.

        Cheaper than :meth:`get_island` when the caller only needs
        the deterministic per-island label -- reads one element of
        ``set_sizes``. PhoenX uses this for stable debug-render
        colour assignment.
        """
        ends = self.set_sizes_compact.numpy()
        n = self.num_islands()
        if not (0 <= island_index < n):
            raise IndexError(f"island_index {island_index} out of range [0, {n})")
        start = 0 if island_index == 0 else int(ends[island_index - 1])
        return int(self.set_sizes.numpy()[start])
