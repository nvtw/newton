# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""On-device builder for the mass-splitting interaction graph.

Mirrors C# ``MassSplittingRigidBodyInteractionGraph.BuildInteractionGraph``
(``MassSplitting/MassSplittingRigidBodyInteractionGraph.cs:129``) plus
``MassSplittingKernels.BuildInteractionGraph{Prepare,Build,Finalize}Kernel``
(``CudaKernels/MassSplitting/MassSplittingKernels.cs``).

## Pipeline (all stages CUDA-graph-capturable)

The caller deposits ``(unified_node_id, partition_key)`` pairs into
``InteractionGraphScratch.packed_keys`` via :func:`emit_pair` from a
per-constraint kernel (or hand-build the array in tests). Then
:func:`build_interaction_graph` runs the canonical chain:

1. ``_mask_tail_int64`` — slots past ``num_pairs[0]`` are stamped with
   ``INT64_MAX`` so radix sort pushes them to the tail.
2. ``wp.utils.radix_sort_pairs`` on ``packed_keys`` (high 32 bits =
   node id, low 32 = partition key) — drives both the node-id grouping
   and the per-node partition-key ordering.
3. ``_mark_boundaries_and_count_kernel`` — per slot ``k``: flag the
   first occurrence of each unique ``(node, partition)`` pair; for each
   flagged slot atomic-add the node's bucket in ``section_end``.
4. ``wp.utils.array_scan`` on ``is_boundary`` (exclusive) →
   ``dest_idx``, then ``_compact_partition_list_kernel`` writes the
   deduplicated partition keys into ``CopyStateContainer.partition_list``
   and stamps ``highest_index_in_use[0]``.
5. ``wp.utils.array_scan`` on ``section_end`` (inclusive) — converts
   per-node counts to cumulative-end offsets, filling empty-node holes
   with their predecessor's value. Mirrors the C#
   ``maxScan.inclusiveScan(StateSectionEndIndices, ...)`` step except
   we use sum-scan over counts which is equivalent and lets us reuse
   ``array_scan``.

The C# build uses an inclusive *max* scan over an array that's already
written with the run-end at each boundary; we use an inclusive *sum*
scan over per-node counts, which collapses to the same result without
needing a custom max-scan kernel.

## Disabled fast path

Constructions where the caller leaves ``num_pairs[0] == 0`` (mass
splitting off) produce ``highest_index_in_use[0] == 0`` after build,
which is the short-circuit gate that :func:`get_state_index` (Step 4)
checks before touching any of the slot arrays.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer

__all__ = [
    "InteractionGraphScratch",
    "build_interaction_graph",
    "emit_pair",
    "interaction_graph_scratch_zeros",
]


#: Tail sentinel pushed past ``num_pairs[0]`` so radix sort lands those
#: slots at the end of the buffer. Picked to be ``INT64_MAX`` since no
#: valid packed key reaches the sign bit (node_ids are non-negative
#: int32 → high 32 bits of a real key are in ``[0, 2^31)``).
_INT64_MAX_LITERAL: int = 9223372036854775807


@wp.struct
class InteractionGraphScratch:
    """Per-step scratch buffers driving :func:`build_interaction_graph`.

    Sized for at most ``capacity`` ``(node, partition)`` pairs. Arrays
    that feed ``wp.utils.radix_sort_pairs`` are doubled (``2 *
    capacity``) per Warp's ping-pong scratch requirement.
    """

    #: Packed ``(node_id << 32) | partition_key`` keys. Slot 0..``num_pairs[0]``
    #: is live; the tail is masked by :func:`build_interaction_graph`'s
    #: prep stage and ignored by all downstream kernels. Length:
    #: ``2 * capacity`` (radix-sort ping-pong).
    packed_keys: wp.array[wp.int64]

    #: Dummy int32 payload for ``radix_sort_pairs`` (which insists on a
    #: key+value API even when only the keys matter). Length:
    #: ``2 * capacity``.
    packed_values: wp.array[wp.int32]

    #: Atomic emit counter; the per-constraint kernel calls :func:`emit_pair`
    #: which ``wp.atomic_add(num_pairs, 0, 1)``s here. Length: 1.
    num_pairs: wp.array[wp.int32]

    #: Per-slot first-occurrence flag stamped during boundary detect.
    #: Length: ``capacity``.
    is_boundary: wp.array[wp.int32]

    #: Exclusive prefix sum of ``is_boundary``; gives the write index in
    #: ``CopyStateContainer.partition_list`` for each unique pair.
    #: Length: ``capacity``.
    dest_idx: wp.array[wp.int32]


def interaction_graph_scratch_zeros(
    capacity: int,
    device: wp.context.Devicelike = None,
) -> InteractionGraphScratch:
    """Allocate a zero-initialised :class:`InteractionGraphScratch`.

    ``capacity`` is the maximum number of ``(node, partition)`` pairs
    the caller will emit in one build — an upper bound is
    ``sum(endpoints_per_constraint)`` over all active constraints. The
    radix-sort ping-pong buffer is allocated at ``2 * capacity`` per
    Warp's API.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1 (got {capacity})")
    s = InteractionGraphScratch()
    s.packed_keys = wp.zeros(2 * capacity, dtype=wp.int64, device=device)
    s.packed_values = wp.zeros(2 * capacity, dtype=wp.int32, device=device)
    s.num_pairs = wp.zeros(1, dtype=wp.int32, device=device)
    s.is_boundary = wp.zeros(capacity, dtype=wp.int32, device=device)
    s.dest_idx = wp.zeros(capacity, dtype=wp.int32, device=device)
    return s


@wp.func
def _pack_key(node_id: wp.int32, partition_key: wp.int32) -> wp.int64:
    """Pack ``(node_id, partition_key)`` into one ``int64`` with
    ``node_id`` in the high 32 bits so radix sort groups by node first.
    """
    return (wp.int64(node_id) << wp.int64(32)) | (wp.int64(partition_key) & wp.int64(0xFFFFFFFF))


@wp.func
def _unpack_node(key: wp.int64) -> wp.int32:
    """Recover the ``node_id`` half of a packed key."""
    return wp.int32(key >> wp.int64(32))


@wp.func
def _unpack_partition(key: wp.int64) -> wp.int32:
    """Recover the ``partition_key`` half of a packed key."""
    return wp.int32(key & wp.int64(0xFFFFFFFF))


@wp.func
def emit_pair(
    scratch: InteractionGraphScratch,
    node_id: wp.int32,
    partition_key: wp.int32,
):
    """Append one ``(node_id, partition_key)`` pair to the scratch.

    Hot-path helper for per-constraint emit kernels. The atomic counter
    bounds the index against the scratch capacity; over-emit silently
    drops past the end (matches C# ``DuplicateFreeList`` ``Add``
    semantics — capacity is the caller's responsibility). Static
    bodies should be filtered by the caller before calling this.
    """
    cap = scratch.packed_keys.shape[0] // 2  # scratch is 2*capacity for radix ping-pong
    slot = wp.atomic_add(scratch.num_pairs, 0, wp.int32(1))
    if slot < cap:
        scratch.packed_keys[slot] = _pack_key(node_id, partition_key)


# -----------------------------------------------------------------------------
# Build stages.
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _mask_tail_kernel(keys: wp.array[wp.int64], num_pairs: wp.array[wp.int32]):
    """Stamp ``INT64_MAX`` past ``num_pairs[0]`` so radix sort tails the slots.

    Mirrors :func:`helpers.scan_and_sort.sort_variable_length_int64`'s
    masking pass but here ``num_pairs`` is the build-time counter.
    """
    tid = wp.tid()
    if tid >= num_pairs[0]:
        keys[tid] = wp.int64(_INT64_MAX_LITERAL)


@wp.kernel(enable_backward=False)
def _zero_section_end_and_is_boundary_kernel(
    section_end: wp.array[wp.int32],
    is_boundary: wp.array[wp.int32],
):
    """Prep: zero per-node counts and per-slot boundary flags.

    Launch ``dim = max(num_nodes, capacity)``; per-thread cheap branches
    skip the tail. ``CopyStateContainer.section_end`` doubles as the
    per-node count buffer during build, then gets in-place scanned into
    the cumulative-end offsets.
    """
    tid = wp.tid()
    if tid < section_end.shape[0]:
        section_end[tid] = wp.int32(0)
    if tid < is_boundary.shape[0]:
        is_boundary[tid] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _zero_highest_index_kernel(highest_index_in_use: wp.array[wp.int32]):
    """One-thread reset of the build's scalar output."""
    if wp.tid() == 0:
        highest_index_in_use[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _reset_num_pairs_kernel(num_pairs: wp.array[wp.int32]):
    """One-thread reset of the emit counter at the start of a build."""
    if wp.tid() == 0:
        num_pairs[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _mark_boundaries_and_count_kernel(
    keys: wp.array[wp.int64],
    is_boundary: wp.array[wp.int32],
    section_end: wp.array[wp.int32],
):
    """Per-slot first-occurrence detection and per-node count atomic-add.

    Run after sort. Two outputs in one pass:

    * ``is_boundary[k] = 1`` iff slot ``k`` is the first occurrence of
      its packed key (i.e. ``k == 0`` or ``keys[k] != keys[k-1]``) AND
      ``keys[k] != INT64_MAX`` (i.e. not a masked tail slot).

    * For every boundary slot, ``atomic_add(section_end[node_id], 1)`` —
      the per-node count of unique partition keys it touches. The
      following ``array_scan`` turns counts into cumulative ends.

    Static bodies / dropped endpoints were filtered before
    :func:`emit_pair`, so every surviving boundary key contributes
    exactly one slot to its node's run.
    """
    tid = wp.tid()
    if tid >= keys.shape[0]:
        return
    k = keys[tid]
    if k == wp.int64(_INT64_MAX_LITERAL):
        return
    is_first = tid == 0
    if not is_first:
        is_first = keys[tid - 1] != k
    if not is_first:
        return
    is_boundary[tid] = wp.int32(1)
    node_id = _unpack_node(k)
    if node_id >= 0 and node_id < section_end.shape[0]:
        wp.atomic_add(section_end, node_id, wp.int32(1))


@wp.kernel(enable_backward=False)
def _compact_partition_list_kernel(
    keys: wp.array[wp.int64],
    is_boundary: wp.array[wp.int32],
    dest_idx: wp.array[wp.int32],
    partition_list: wp.array[wp.int32],
    highest_index_in_use: wp.array[wp.int32],
):
    """Scatter unique partition keys into ``partition_list`` and stamp
    the build-final ``highest_index_in_use`` scalar.

    ``dest_idx`` is the exclusive prefix sum of ``is_boundary``, so the
    final write index for a boundary slot is ``dest_idx[k]``. The last
    boundary slot's ``dest_idx + 1`` is the total unique count — i.e.
    the new ``highest_index_in_use``.
    """
    tid = wp.tid()
    if tid >= keys.shape[0]:
        return
    if is_boundary[tid] == wp.int32(0):
        return
    out = dest_idx[tid]
    k = keys[tid]
    if out < partition_list.shape[0]:
        partition_list[out] = _unpack_partition(k)
    # Single-thread write of the total count: the last boundary slot
    # wins the race for highest_index_in_use[0]. Using atomic_max so
    # any tied write produces a deterministic result.
    wp.atomic_max(highest_index_in_use, 0, out + wp.int32(1))


def build_interaction_graph(
    scratch: InteractionGraphScratch,
    copy_state: CopyStateContainer,
) -> None:
    """Run the on-device build pipeline once.

    Preconditions:

    * Caller has written ``scratch.num_pairs[0] = N`` and filled
      ``scratch.packed_keys[0:N]`` with packed ``(node_id, partition_key)``
      pairs via :func:`emit_pair` (or directly in tests).

    Postconditions:

    * ``copy_state.section_end`` = inclusive cumulative-end per node.
    * ``copy_state.partition_list[0:highest_index_in_use[0]]`` = sorted
      unique partition keys per node.
    * ``copy_state.highest_index_in_use[0]`` = total unique slot count.

    Graph-capture safe: all stages are ``wp.launch`` / Warp utility
    calls that work inside ``wp.ScopedCapture``.
    """
    device = scratch.packed_keys.device
    capacity = scratch.packed_keys.shape[0] // 2
    num_nodes = copy_state.section_end.shape[0]

    # Stage 1: prep — mask sort tail, zero accumulators.
    prep_dim = max(num_nodes, capacity)
    wp.launch(
        _zero_section_end_and_is_boundary_kernel,
        dim=prep_dim,
        inputs=[copy_state.section_end, scratch.is_boundary],
        device=device,
    )
    wp.launch(
        _zero_highest_index_kernel,
        dim=1,
        inputs=[copy_state.highest_index_in_use],
        device=device,
    )
    wp.launch(
        _mask_tail_kernel,
        dim=capacity,
        inputs=[scratch.packed_keys, scratch.num_pairs],
        device=device,
    )

    # Stage 2: sort. radix_sort_pairs requires count == buffer_size / 2
    # and uses the second half as ping-pong scratch.
    wp.utils.radix_sort_pairs(scratch.packed_keys, scratch.packed_values, capacity)

    # Stage 3: boundary detect + per-node count.
    wp.launch(
        _mark_boundaries_and_count_kernel,
        dim=capacity,
        inputs=[scratch.packed_keys, scratch.is_boundary, copy_state.section_end],
        device=device,
    )

    # Stage 4: exclusive scan boundary → dest_idx, then compact.
    wp.utils.array_scan(scratch.is_boundary, scratch.dest_idx, inclusive=False)
    wp.launch(
        _compact_partition_list_kernel,
        dim=capacity,
        inputs=[
            scratch.packed_keys,
            scratch.is_boundary,
            scratch.dest_idx,
            copy_state.partition_list,
            copy_state.highest_index_in_use,
        ],
        device=device,
    )

    # Stage 5: inclusive scan per-node counts → cumulative-end offsets.
    # Empty-node holes are filled automatically (sum scan over zeros).
    wp.utils.array_scan(copy_state.section_end, copy_state.section_end, inclusive=True)

    # Stage 6: reset the emit counter so the next build starts clean.
    # Done last so callers can still read num_pairs immediately after
    # build for diagnostics.
    wp.launch(
        _reset_num_pairs_kernel,
        dim=1,
        inputs=[scratch.num_pairs],
        device=device,
    )
