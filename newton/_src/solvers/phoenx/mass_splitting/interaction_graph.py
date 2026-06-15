# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""On-device builder for the mass-splitting interaction graph.

Mirrors C# ``MassSplittingRigidBodyInteractionGraph.BuildInteractionGraph``
(``MassSplitting/MassSplittingRigidBodyInteractionGraph.cs:129``) plus
``MassSplittingKernels.BuildInteractionGraph{Prepare,Build,Finalize}Kernel``.

Pipeline (graph-capturable): callers deposit ``(node_id, partition_key)``
pairs into ``InteractionGraphScratch.packed_keys`` via :func:`emit_pair`,
then :func:`build_interaction_graph`:

1. Mask tail past ``num_pairs[0]`` with ``INT64_MAX`` so the sort tails it.
2. ``radix_sort_pairs`` on ``packed_keys`` (high 32 = node, low 32 =
   partition_key) -- groups by node, sorts partitions within node.
3. ``_mark_boundaries_and_count_kernel`` flags first occurrence of each
   unique ``(node, partition)`` and atomic-adds the node bucket count.
4. Exclusive scan on ``is_boundary`` -> ``dest_idx``;
   ``_compact_partition_list_kernel`` writes deduplicated partition
   keys into ``CopyStateContainer.partition_list``.
5. Inclusive scan on ``section_end`` -- per-node counts -> cumulative
   end offsets; this auto-fills empty-node holes with the predecessor.

Disabled fast path: ``num_pairs[0] == 0`` => ``highest_index_in_use[0]
== 0`` after build, the short-circuit gate :func:`get_state_index`
checks.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_get,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer

__all__ = [
    "InteractionGraphScratch",
    "build_interaction_graph",
    "emit_pair",
    "interaction_graph_scratch_zeros",
    "record_all_interactions_kernel",
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


@wp.kernel(enable_backward=False)
def record_all_interactions_kernel(
    elements: wp.array[ElementInteractionData],
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    interaction_id_to_partition: wp.array[wp.int32],
    max_colored_partitions: wp.int32,
    batch_size: wp.int32,
    scratch: InteractionGraphScratch,
):
    """Per-step emit: walk the active CSR slots and stamp
    ``(node_id, partition_key)`` pairs into ``scratch.packed_keys``.

    Direct port of C# ``PartitioningKernels.RecordAllInteractionsKernel``
    (``CudaKernels/MassSplitting/PartitioningKernels.cs:618``):

    * Slot ``t`` lives in colour
      ``interaction_id_to_partition[element_ids_by_color[t]]``.
    * If colour ``< K`` (a regular MIS bucket): ``partition_key = 0``
      so every body in any regular bucket shares one copy slot.
    * If colour ``== K`` (overflow): ``partition_key = (t -
      color_starts[K]) // batch_size`` — overflow constraints group
      into batches of ``batch_size`` consecutive CSR slots, all
      sharing one partition copy. The iterate kernel processes each
      batch sequentially in one thread (GS within the batch) so the
      shared slot is race-free; across batches, processing is
      parallel (Jacobi). ``batch_size = 1`` (one constraint per
      partition) is the strongest splitting; ``batch_size = 8``
      matches C# PhoenX's ``BatchingThreshold = 8``.

    Nodes are emitted under the unified body-or-particle index space
    (rigid bodies in ``[0, num_bodies)``, particles in
    ``[num_bodies, num_bodies + num_particles)``). Particles
    participate in mass splitting just like rigid bodies — they own
    slots in the copy state and are averaged-and-broadcast back at
    the end of each iteration. Cloth-triangle / soft-tet elasticity
    must compensate for Tonge's ``1/N`` average dilution to preserve
    XPBD position-level correctness (see
    :mod:`newton._src.solvers.phoenx.constraints.constraint_cloth_triangle`).

    One thread per CSR slot in ``[0, num_active_constraints)``. Each
    thread emits up to ``MAX_BODIES`` entries via :func:`emit_pair`
    (which atomic-adds the emit counter and drops past capacity).
    """
    tid = wp.tid()
    if tid >= num_active_constraints[0]:
        return
    eid = element_ids_by_color[tid]
    color = interaction_id_to_partition[eid]
    partition_key = wp.int32(0)
    if color >= max_colored_partitions:
        # Overflow slice. Group every ``batch_size`` consecutive
        # CSR slots into one partition copy.
        overflow_offset = tid - color_starts[max_colored_partitions]
        partition_key = overflow_offset / batch_size
    el = elements[eid]
    for j in range(MAX_BODIES):
        v = element_interaction_data_get(el, j)
        if v < wp.int32(0):
            break
        emit_pair(scratch, v, partition_key)


# -----------------------------------------------------------------------------
# Build stages.
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _zero_section_end_and_is_boundary_kernel(
    section_end: wp.array[wp.int32],
    is_boundary: wp.array[wp.int32],
    highest_index_in_use: wp.array[wp.int32],
):
    """Prep: zero per-node counts, boundary flags, and scalar output.

    Launch ``dim = max(num_nodes, capacity, 1)``; cheap branches
    skip the tail. ``CopyStateContainer.section_end`` doubles as the
    per-node count buffer during build, then gets in-place scanned into
    the cumulative-end offsets.
    """
    tid = wp.tid()
    if tid == wp.int32(0):
        highest_index_in_use[0] = wp.int32(0)
    if tid < section_end.shape[0]:
        section_end[tid] = wp.int32(0)
    if tid < is_boundary.shape[0]:
        is_boundary[tid] = wp.int32(0)


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
def _build_pid0_cache_kernel(
    section_end: wp.array[wp.int32],
    partition_list: wp.array[wp.int32],
    highest_index_in_use: wp.array[wp.int32],
    slot_for_pid0: wp.array[wp.int32],
    count_per_node: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
):
    """Stamp per-node caches read by :func:`get_state_index` on the hot
    path. Runs after Stage 5's inclusive scan, so ``section_end`` is
    cumulative and ``partition_list[0:highest_index_in_use[0]]`` is
    dense / sorted.

    For each node ``i``:
        start = section_end[i-1]   (0 if i==0)
        end = section_end[i]
        count = end - start
        slot = start if (count > 0 and partition_list[start] == 0) else -1

    The slot is the location of the ``parallel_id == 0`` entry if it
    exists. Sort order guarantees the smallest partition key sits at
    ``start``, so a single equality check picks the regular-colour slot.
    ``count`` is the inv_factor return value used by prepare-phase
    code to apply Tonge's ``1/N`` mass scaling.

    No-op when the build emitted zero pairs (``highest_index_in_use[0]
    == 0``) -- ``slot_for_pid0`` was zero-initialised to -1 and stays so,
    matching the disabled-fast-path semantics of :func:`get_state_index`.
    Lane 0 also resets the emit counter for the next build.
    """
    node_id = wp.tid()
    if node_id == wp.int32(0):
        num_pairs[0] = wp.int32(0)
    if node_id >= section_end.shape[0]:
        return
    if highest_index_in_use[0] == wp.int32(0):
        # Mass splitting disabled / no pairs emitted: keep cache at the
        # zero-init values (slot=-1, count=0).
        slot_for_pid0[node_id] = wp.int32(-1)
        count_per_node[node_id] = wp.int32(0)
        return
    start = wp.int32(0)
    if node_id > wp.int32(0):
        start = section_end[node_id - wp.int32(1)]
    end = section_end[node_id]
    count = end - start
    count_per_node[node_id] = count
    if count == wp.int32(0):
        slot_for_pid0[node_id] = wp.int32(-1)
        return
    # Sort places the smallest partition key first; check it.
    if partition_list[start] == wp.int32(0):
        slot_for_pid0[node_id] = start
    else:
        slot_for_pid0[node_id] = wp.int32(-1)


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

    # Stage 1: prep — zero per-node accumulators, boundary flags, and scalar output.
    prep_dim = max(num_nodes, capacity, 1)
    wp.launch(
        _zero_section_end_and_is_boundary_kernel,
        dim=prep_dim,
        inputs=[copy_state.section_end, scratch.is_boundary, copy_state.highest_index_in_use],
        device=device,
    )

    # Stage 2: variable-length sort. Helper masks the tail past
    # ``num_pairs[0]`` with ``INT64_MAX`` then runs radix_sort_pairs.
    sort_variable_length_int64(scratch.packed_keys, scratch.packed_values, scratch.num_pairs)

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

    # Stage 6: stamp the per-node parallel_id=0 / count caches that
    # :func:`get_state_index` reads on the hot path, and reset the
    # emit counter for the next build. Collapses 3 dependent loads
    # (section_end[node-1], section_end[node], partition_list[start])
    # to 1-2 loads per call.
    wp.launch(
        _build_pid0_cache_kernel,
        dim=max(num_nodes, 1),
        inputs=[
            copy_state.section_end,
            copy_state.partition_list,
            copy_state.highest_index_in_use,
            copy_state.slot_for_pid0,
            copy_state.count_per_node,
            scratch.num_pairs,
        ],
        device=device,
    )
