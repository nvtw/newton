# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of PhoenX's ``UnionFindIslandBuilder``.

Atomic union-find (path compression + union-by-rank) over interaction body
chains, then a deterministic post-pass sorts island ids by each island's
smallest body id so output is reproducible despite race-happy atomics.
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


#: Max bodies per interaction row. Matches the vec8i ElementInteractionData layout.
MAX_BODIES_PER_INTERACTION = 8


#: Parent sentinel = "slot never touched". Matches PhoenX's 0xFFFFFFFFu.
INVALID_PARENT = wp.constant(wp.int32(-1))


# Packed entries: int64 word, low 32 bits = parent, bits 32..62 = rank, bit 63 = 0.

_RANK_SHIFT = wp.constant(wp.int64(32))
_PARENT_MASK = wp.constant(wp.int64(0xFFFFFFFF))
_RANK_MASK_IN_ENTRY = wp.constant(wp.int64(0x7FFFFFFF00000000))
_RANK_LOW_MASK = wp.constant(wp.int64(0x7FFFFFFF))
_INT32_MAX = wp.constant(wp.int32(0x7FFFFFFF))


@wp.func
def _pack_entry(parent: wp.int32, rank: wp.int32) -> wp.int64:
    """Pack ``(parent, rank)`` into a single int64 word."""
    return (wp.int64(rank) << _RANK_SHIFT) | (wp.int64(parent) & _PARENT_MASK)


@wp.func
def _entry_parent(entry: wp.int64) -> wp.int32:
    """Low 32 bits of an entry -> parent body id."""
    return wp.int32(entry & _PARENT_MASK)


@wp.func
def _entry_rank(entry: wp.int64) -> wp.int32:
    """High 31 bits of an entry -> rank (always non-negative)."""
    return wp.int32((entry >> _RANK_SHIFT) & _RANK_LOW_MASK)


# Lock-free union-find from wjakob/dset: find-with-path-compression + union-by-rank.


@wp.func
def _find(entries: wp.array[wp.int64], body_id: wp.int32) -> wp.int32:
    """Representative of ``body_id``'s component, with opportunistic path compression."""
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
            # Best-effort path compression; failed CAS = another thread won.
            wp.atomic_cas(entries, id_, value, new_value)
        id_ = new_parent


@wp.func
def _unite(entries: wp.array[wp.int64], a: wp.int32, b: wp.int32):
    """Link components of ``a`` and ``b`` via union-by-rank with CAS-retry on race."""
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

        # Lower rank under higher; ties go to smaller id (deterministic).
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
            continue

        if r1 == r2:
            # Best-effort rank bump (race-tolerant).
            old_entry = _pack_entry(id2, r2)
            new_entry = _pack_entry(id2, r2 + wp.int32(1))
            wp.atomic_cas(entries, id2, old_entry, new_entry)
        return


# Kernels: fixed launch size, gated by tid >= active_length[0] (graph-capture safe).


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
    """Seed each body as its own singleton: entries[i] = (rank=0, parent=i)."""
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
    """Chain-union body lists per interaction: union(b0,b1), union(b1,b2), ...
    -1 ends the chain. ``interaction_bodies`` is ``(capacity, 8)`` int32 with
    -1 padding."""
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
    """Per-body: set_nr=find(i), atomic_add(set_sizes[nr],1),
    atomic_min(min_index_per_set[nr], i)."""
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
    """Stamp 1/0 into old_to_new[i] so an exclusive scan gives the compact
    rawRepId -> compactRepId map."""
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
    """num_sets[0] = old_to_new[n-1] + (set_sizes[n-1] != 0); single-thread."""
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
    """Pack raw per-rep min-indices into compact order; seed
    min_index_per_set[0..numSets) with identity i for the upcoming sort."""
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
    # After the sort, min_index_per_set[i] = old_compact_slot; we want
    # min_index_per_set_compact[old_compact_slot] = new_compact_slot.
    min_index_per_set_compact: wp.array[wp.int32],
):
    """Invert the sort's value permutation -> rawCompact -> newCompact."""
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
    """Stage1: rewrite raw set ids into sorted-by-min-index compact ordering;
    plant compact-sizes scan seed (guard skips redundant same-value writes)."""
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
    """Stage2: seed set_sizes with identity body indices and copy new set id
    into old_to_new so a stable key-value sort groups body ids per island."""
    tid = wp.tid()
    if tid >= num_bodies[0]:
        return
    set_sizes[tid] = tid
    old_to_new[tid] = set_nr[tid]


class UnionFindIslandBuilder:
    """Builds connected components over bodies linked by interaction chains.

    Usage::

        builder = UnionFindIslandBuilder(num_bodies_capacity=N, device=dev)
        builder.build_islands(interactions, num_interactions, num_bodies)
        num_islands = int(builder.num_sets.numpy()[0])
        island_of_body = builder.set_nr.numpy()
        body_ids = builder.set_sizes.numpy()
        ends = builder.set_sizes_compact.numpy()[:num_islands]

    Buffers are pre-allocated; build_islands is fixed-size and graph-capture safe.
    Output is deterministic despite atomic union-find races (post-sort by min body id).
    """

    def __init__(
        self,
        num_bodies_capacity: int,
        device: wp.context.Devicelike = None,
    ) -> None:
        """Pre-allocate device buffers. Buffers are sized 2*capacity for the
        radix-sort ping-pong requirement."""
        if num_bodies_capacity <= 0:
            raise ValueError(f"num_bodies_capacity must be > 0 (got {num_bodies_capacity})")
        self._capacity: int = int(num_bodies_capacity)
        self._device = wp.get_device(device)

        cap = self._capacity
        sort_cap = 2 * cap

        self.entries: wp.array[wp.int64] = wp.zeros(cap, dtype=wp.int64, device=self._device)
        self.set_nr: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self._device)
        # set_sizes / old_to_new double as final-sort carriers (need 2x ping-pong).
        self.set_sizes: wp.array[wp.int32] = wp.zeros(sort_cap, dtype=wp.int32, device=self._device)
        self.set_sizes_compact: wp.array[wp.int32] = wp.zeros(cap, dtype=wp.int32, device=self._device)
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
        extra_edges: wp.array2d[wp.int32] | None = None,
        num_extra_edges: wp.array[wp.int32] | None = None,
    ) -> None:
        """Full pipeline: init -> atomic union-find -> compact -> sort by min-index
        -> invert map -> rewrite per-body ids -> scan -> group bodies per island.

        ``interaction_bodies`` is ``(>= num_interactions[0], 8)`` with -1 padding.

        ``extra_edges`` is an optional second interaction array unioned
        in alongside the regular elements (same row layout, ``-1``-padded).
        Used by the sleeping pipeline to inject artificial chain edges
        ``(body, island_root)`` that pull active sleeping islands back
        into the live union-find. Caller must also pass
        ``num_extra_edges``.
        """
        cap = self._capacity

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

        # Atomic union-find. Launch dim = interactions buffer leading extent.
        wp.launch(
            _island_unite_kernel,
            dim=interaction_bodies.shape[0],
            inputs=[interaction_bodies, num_interactions],
            outputs=[self.entries],
            device=self._device,
        )

        if extra_edges is not None and num_extra_edges is not None:
            wp.launch(
                _island_unite_kernel,
                dim=extra_edges.shape[0],
                inputs=[extra_edges, num_extra_edges],
                outputs=[self.entries],
                device=self._device,
            )

        wp.launch(
            _island_compute_set_nrs_kernel,
            dim=cap,
            inputs=[num_bodies, self.entries],
            outputs=[self.set_nr, self.set_sizes, self.min_index_per_set],
            device=self._device,
        )

        # Bitmap of occupied reps -> exclusive scan -> num_sets.
        wp.launch(
            _island_prepare_scan_kernel,
            dim=cap,
            inputs=[num_bodies, self.set_sizes],
            outputs=[self.old_to_new],
            device=self._device,
        )

        scan_variable_length(self.old_to_new, num_bodies, inclusive=False)

        wp.launch(
            _island_reduce_num_sets_kernel,
            dim=1,
            inputs=[num_bodies, self.old_to_new, self.set_sizes],
            outputs=[self.num_sets],
            device=self._device,
        )

        # Pack per-rep min-indices into compact order; seed identity sort values.
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

        # Sort compact slots by min-index for deterministic island ids.
        sort_variable_length_int(
            keys=self.min_index_per_set_compact,
            values=self.min_index_per_set,
            active_length=self.num_sets,
        )

        wp.launch(
            _island_invert_map_kernel,
            dim=cap,
            inputs=[self.num_sets, self.min_index_per_set],
            outputs=[self.min_index_per_set_compact],
            device=self._device,
        )

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

        # Inclusive scan -> per-island end offsets.
        scan_variable_length(self.set_sizes_compact, self.num_sets, inclusive=True)

        # Final stable key-value sort: key = island id, value = body id.
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

    # Host-side helpers (all trigger device->host copies).

    def num_islands(self) -> int:
        """Number of islands produced by the last :meth:`build_islands`."""
        return int(self.num_sets.numpy()[0])

    def island_index_of_body(self, body_id: int) -> int:
        """Canonical island id the body was assigned to (in [0, num_islands()),
        ordered by smallest body id)."""
        return int(self.set_nr.numpy()[body_id])

    def get_island(self, island_index: int) -> list[int]:
        """Body ids in island ``island_index``, sorted ascending."""
        ends = self.set_sizes_compact.numpy()
        n = self.num_islands()
        if not (0 <= island_index < n):
            raise IndexError(f"island_index {island_index} out of range [0, {n})")
        start = 0 if island_index == 0 else int(ends[island_index - 1])
        end = int(ends[island_index])
        bodies = self.set_sizes.numpy()[start:end]
        return [int(x) for x in bodies]

    def get_island_lowest(self, island_index: int) -> int:
        """Smallest body id in island ``island_index``."""
        ends = self.set_sizes_compact.numpy()
        n = self.num_islands()
        if not (0 <= island_index < n):
            raise IndexError(f"island_index {island_index} out of range [0, {n})")
        start = 0 if island_index == 0 else int(ends[island_index - 1])
        return int(self.set_sizes.numpy()[start])
