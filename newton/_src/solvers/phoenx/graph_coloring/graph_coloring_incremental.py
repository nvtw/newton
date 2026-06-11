"""Incremental Jones-Plassmann graph partitioner.

One partition per :meth:`launch` call. No max_num_partitions budget; loop state
lives in 1-elem device arrays so a fixed-length launch sequence is graph-capturable.

Usage::

    partitioner = IncrementalContactPartitioner(N, num_bodies, device=d)
    partitioner.reset(elements_arr, num_elements_arr)
    while int(partitioner.num_remaining.numpy()[0]) > 0:
        partitioner.launch()
"""

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_contact_count,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    GREEDY_MAX_COLORS,
    MAX_BODIES,
    MAX_GREEDY_OUTER_ITERS,
    TILE_SCAN_BLOCK_DIM,
    ElementInteractionData,
    element_interaction_data_get,
    greedy_clear_int_kernel,
    greedy_color_histogram_kernel,
    greedy_count_and_scan_color_starts_kernel,
    greedy_overflow_spill_kernel,
    greedy_reset_init_kernel,
    greedy_scatter_elements_by_color_kernel,
    incremental_begin_sweep_kernel,
    incremental_fill_minus_one_kernel,
    incremental_init_kernel,
    incremental_init_remaining_ids_kernel,
    incremental_reset_loop_state_csr_kernel,
    incremental_reset_loop_state_kernel,
    incremental_tile_compact_csr_and_advance_kernel,
    incremental_tile_compact_remaining_and_advance_kernel,
    partitioning_adjacency_count_kernel,
    partitioning_adjacency_store_kernel,
    partitioning_coloring_incremental_greedy_kernel,
    partitioning_coloring_incremental_kernel,
    partitioning_prepare_kernel,
    speculative_overflow_exit_kernel,
    speculative_pick_kernel,
    speculative_validate_commit_kernel,
    warm_start_periodic_invalidate_kernel,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import scan_variable_length
from newton._src.solvers.phoenx.solver_config import NUM_INNER_WHILE_ITERATIONS

__all__ = ["MAX_COLORS", "IncrementalContactPartitioner"]

# Upper bound on colour count per pass. Real graphs stay below 200; 1024 gives
# comfortable margin at negligible cost (color_starts is (MAX_COLORS+1)*4 B).
MAX_COLORS = 1024

_GREEDY_BLOCK_DIM: int = 256


def _greedy_coloring_grid_size(max_num_interactions: int, device: wp.DeviceLike) -> int:
    """Persistent-grid thread count for the greedy coloring kernel:
    min(capacity_blocks, 1 block/SM) * 256, floor 8 blocks."""
    block_dim = _GREEDY_BLOCK_DIM
    device_obj = wp.get_device(device)
    if device_obj.is_cuda:
        max_blocks_limit = max(8, device_obj.sm_count)
    else:
        max_blocks_limit = 256
    capacity_blocks = max(1, (max(1, int(max_num_interactions)) + block_dim - 1) // block_dim)
    num_blocks = max(8, min(capacity_blocks, max_blocks_limit))
    return num_blocks * block_dim


# INT64_MAX sentinel: pads inactive slots so the radix sort lands them
# at the tail (they're never read by the iterate kernel).
_LOCALITY_TAIL_KEY: int = 0x7FFFFFFFFFFFFFFF

_LOCALITY_FAMILY_BITS = wp.constant(wp.int64(4))
_LOCALITY_BODY_MIN_BITS = wp.constant(wp.int64(24))
_LOCALITY_EID_BITS = wp.constant(wp.int64(25))
_LOCALITY_FAMILY_MASK = wp.constant(wp.int64((1 << 4) - 1))
_LOCALITY_BODY_MIN_MASK = wp.constant(wp.int64((1 << 24) - 1))
_LOCALITY_EID_MASK = wp.constant(wp.int64((1 << 25) - 1))
_LOCALITY_FAMILY_COUNT_HOST = 6
_LOCALITY_FAMILY_COUNT = wp.constant(_LOCALITY_FAMILY_COUNT_HOST)


@wp.kernel(enable_backward=False)
def _locality_combined_keys_kernel(
    elements: wp.array[ElementInteractionData],
    element_ids_by_color: wp.array[wp.int32],
    interaction_id_to_partition: wp.array[wp.int32],
    element_family: wp.array[wp.int32],
    family_sort_cutoff: wp.int32,
    num_elements: wp.array[wp.int32],
    keys: wp.array[wp.int64],
    values: wp.array[wp.int32],
):
    """Pack ``(colour, family, body_min, eid)`` into one int64 key so a single
    radix sort preserves colour slices while grouping same-family rows.

    ``family_sort_cutoff`` disables the family bits for overflow colours,
    preserving their original body-local solve order.

    Bit layout: colour (bits 53..63) | family (49..52, 4b) |
    body_min (25..48, 24b) | eid (0..24, 25b). Slots past
    ``num_elements[0]`` get ``INT64_MAX`` and sort to the tail.
    """
    tid = wp.tid()
    if tid >= keys.shape[0]:
        return
    if tid >= num_elements[0]:
        keys[tid] = wp.int64(_LOCALITY_TAIL_KEY)
        values[tid] = wp.int32(0)
        return
    eid = element_ids_by_color[tid]
    color = interaction_id_to_partition[eid]
    family = element_family[eid]
    if family_sort_cutoff >= wp.int32(0) and color >= family_sort_cutoff:
        family = wp.int32(0)
    el = elements[eid]
    body_min = wp.int32(0x7FFFFF)  # 24-bit sentinel for "no body"
    for j in range(MAX_BODIES):
        b = element_interaction_data_get(el, j)
        if b < wp.int32(0):
            break
        if b < body_min:
            body_min = b
    key = (
        (wp.int64(color) << (_LOCALITY_FAMILY_BITS + _LOCALITY_BODY_MIN_BITS + _LOCALITY_EID_BITS))
        | ((wp.int64(family) & _LOCALITY_FAMILY_MASK) << (_LOCALITY_BODY_MIN_BITS + _LOCALITY_EID_BITS))
        | ((wp.int64(body_min) & _LOCALITY_BODY_MIN_MASK) << _LOCALITY_EID_BITS)
        | (wp.int64(eid) & _LOCALITY_EID_MASK)
    )
    keys[tid] = key
    values[tid] = eid


@wp.kernel(enable_backward=False)
def _locality_writeback_kernel(
    values: wp.array[wp.int32],
    num_elements: wp.array[wp.int32],
    element_ids_by_color: wp.array[wp.int32],
):
    """Copy the sorted element-id stream back into the CSR slot array."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    element_ids_by_color[tid] = values[tid]


@wp.kernel(enable_backward=False)
def _zero_color_family_count_kernel(color_family_count: wp.array[wp.int32]):
    slot = wp.tid()
    if slot < color_family_count.shape[0]:
        color_family_count[slot] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _count_color_families_kernel(
    element_ids_by_color: wp.array[wp.int32],
    interaction_id_to_partition: wp.array[wp.int32],
    element_family: wp.array[wp.int32],
    family_sort_cutoff: wp.int32,
    num_elements: wp.array[wp.int32],
    color_family_count: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    eid = element_ids_by_color[tid]
    color = interaction_id_to_partition[eid]
    if color < wp.int32(0) or color >= wp.int32(MAX_COLORS):
        return
    family = element_family[eid]
    if family_sort_cutoff >= wp.int32(0) and color >= family_sort_cutoff:
        family = wp.int32(0)
    if family < wp.int32(0):
        family = wp.int32(0)
    if family >= wp.int32(_LOCALITY_FAMILY_COUNT):
        family = wp.int32(_LOCALITY_FAMILY_COUNT - 1)
    wp.atomic_add(color_family_count, color * wp.int32(_LOCALITY_FAMILY_COUNT) + family, wp.int32(1))


@wp.kernel(enable_backward=False)
def _scan_color_family_starts_kernel(
    color_starts: wp.array[wp.int32],
    color_family_count: wp.array[wp.int32],
    color_family_starts: wp.array[wp.int32],
):
    color = wp.tid()
    if color >= wp.int32(MAX_COLORS):
        return
    base = color * wp.int32(_LOCALITY_FAMILY_COUNT)
    running = color_starts[color]
    for family in range(_LOCALITY_FAMILY_COUNT_HOST):
        slot = base + wp.int32(family)
        color_family_starts[slot] = running
        running = running + color_family_count[slot]


@wp.kernel(enable_backward=False, module="unique")
def _fill_packed_priorities_from_contacts_kernel(
    packed_priorities: wp.array[wp.int32],
    random_values: wp.array[wp.int32],
    contact_cols: ContactColumnContainer,
    num_contact_columns: wp.array[wp.int32],
    num_joints: wp.int32,
):
    """Refresh per-cid packed JP priority from contact column counts.

    Layout: ``packed[i] = (cost << 24) | (random[i] & 0xFFFFFF)``.

    ``cost`` is the contact-count bias (joints get 0, contacts get the
    column's contact count clamped to 0..255); the random tiebreaker
    is read straight from the per-build permutation. This replaces the
    older two-array layout that needed a re-shift/OR per kernel access.
    """
    tid = wp.tid()
    local_cid = tid - num_joints
    if local_cid >= wp.int32(0) and local_cid < num_contact_columns[0]:
        cost = contact_get_contact_count(contact_cols, local_cid)
    else:
        cost = wp.int32(0)
    # Clamp cost to fit in the 8-bit high byte (255+ contacts on a single
    # column is unrealistic but cap defensively).
    if cost > wp.int32(255):
        cost = wp.int32(255)
    rand = random_values[tid] & wp.int32(0x00FFFFFF)
    packed_priorities[tid] = (cost << wp.int32(24)) | rand


class IncrementalContactPartitioner:
    """JP partitioner with two modes.

    Mode A — per-colour streaming: :meth:`reset` then call :meth:`launch` per
    colour, reading :attr:`partition_element_ids`/:attr:`partition_count`/
    :attr:`current_color`. Stop when num_remaining == 0.

    Mode B — CSR build via :meth:`build_csr`: drives the full JP loop in a
    captured graph, writes :attr:`element_ids_by_color` (flat) +
    :attr:`color_starts` (exclusive prefix). Build once per :class:`World.step`,
    replay across substeps/PGS iterations.
    """

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.DeviceLike = None,
        seed: int = 0,
        use_tile_scan: bool = True,
        max_colored_partitions: int | None = None,
        max_greedy_outer_iters: int | None = None,
        enable_warm_start: bool = False,
    ) -> None:
        """``max_colored_partitions``: soft cap for the overflow bucket. When
        set to ``K``, colours ``0..K-1`` are produced by normal MIS coloring
        and any still-uncoloured remainder lands in colour ``K`` (the
        overflow bucket). Mass splitting consumes the bucket and resolves
        the within-colour conflicts via copy states. ``None`` (default)
        disables the cap — overshooting :data:`MAX_COLORS` raises.
        ``K`` must be ``<= GREEDY_MAX_COLORS - 1`` for the greedy path
        (the int64 forbidden mask covers ``[0, GREEDY_MAX_COLORS)``) and
        ``<= MAX_COLORS - 1`` for the JP path so ``color_starts[K + 1]``
        stays in-bounds.
        """
        self.max_num_interactions = max_num_interactions
        self.max_num_nodes = max_num_nodes
        # Tile-scan: single-block, graph-capture safe, no implicit allocations.
        self.use_tile_scan = use_tile_scan
        # Validate the soft cap (None disables; int picks the overflow
        # bucket colour). Pass-through to kernels uses -1 for "disabled".
        if max_colored_partitions is not None:
            if max_colored_partitions < 0:
                raise ValueError(f"max_colored_partitions must be >= 0 or None (got {max_colored_partitions})")
            if max_colored_partitions >= int(GREEDY_MAX_COLORS):
                raise ValueError(
                    f"max_colored_partitions must be < GREEDY_MAX_COLORS ({int(GREEDY_MAX_COLORS)}) "
                    f"for the greedy path's int64 forbidden mask; got {max_colored_partitions}."
                )
            if max_colored_partitions >= MAX_COLORS:
                raise ValueError(
                    f"max_colored_partitions must be < MAX_COLORS ({MAX_COLORS}) so "
                    f"color_starts[K + 1] stays in-bounds; got {max_colored_partitions}."
                )
        self.max_colored_partitions = max_colored_partitions
        # Per-instance override for the host-side greedy loop bound. ``None``
        # uses the module-level :data:`MAX_GREEDY_OUTER_ITERS` (default 16).
        # Mass-splitting scenes can safely lower this -- excess uncoloured
        # elements just spill to the overflow bucket. Non-MS scenes
        # (``max_colored_partitions is None``) need the full 16 to find a
        # valid coloring with no fallback.
        self._max_greedy_outer_iters_override: int | None = (
            int(max_greedy_outer_iters) if max_greedy_outer_iters is not None else None
        )
        self._max_colored_partitions_kernel_arg: int = (
            -1 if max_colored_partitions is None else int(max_colored_partitions)
        )
        # Warm-start cache for cross-frame colour reuse. When enabled,
        # ``build_csr`` seeds ``color_tags`` from the cache and runs a
        # validation pass before the greedy MIS loop -- constraints
        # whose previous-frame colour is still legal skip the MIS
        # entirely. See :mod:`warm_start` for the design.
        self.enable_warm_start: bool = bool(enable_warm_start)
        if self.enable_warm_start:
            from newton._src.solvers.phoenx.graph_coloring.warm_start import (  # noqa: PLC0415
                warm_start_cache_zeros,
            )

            self._warm_start_cache = warm_start_cache_zeros(max_num_interactions, device=device)
            # Per-frame scratch for the validation pass. ``mark`` is
            # written by mark_kernel, consumed + cleared by
            # apply_kernel each step.
            self._warm_start_invalid_mark = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
            # Persist-pipeline scratch (radix sort + boundary scan +
            # compact). Radix sort needs 2*N ping-pong; the boundary
            # / dest_idx arrays are 1*N.
            self._ws_pair_keys = wp.zeros(2 * max_num_interactions, dtype=wp.int64, device=device)
            self._ws_pair_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)
            self._ws_is_boundary = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
            self._ws_dest_idx = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        else:
            self._warm_start_cache = None
            self._warm_start_invalid_mark = None
            self._ws_pair_keys = None
            self._ws_pair_values = None
            self._ws_is_boundary = None
            self._ws_dest_idx = None

        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        # The packed priority (cost << 24) | (random & 0xFFFFFF) caps the
        # random tiebreaker at 2^24 = 16M; ensure the permutation fits.
        if priorities.max() >= (1 << 24):
            raise ValueError(
                f"max_num_interactions ({max_num_interactions}) exceeds the 2^24 "
                "packed-priority limit. Lower the partitioner capacity or widen the "
                "PACKED_PRIO_RANDOM_MASK / cost shift in graph_coloring_common.py."
            )
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)
        # Per-cid packed (cost, random) priority. Initialised with cost=0
        # so non-contact rows already have a usable value; refreshed
        # each step from contact column counts via
        # :func:`_fill_packed_priorities_from_contacts_kernel`.
        packed_init = (priorities & 0x00FFFFFF).astype(np.int32)
        self._packed_priorities = wp.from_numpy(packed_init, dtype=wp.int32, device=device)

        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(
            max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device
        )

        # Per-element packed tag: bit 62 = unpartitioned marker, bits 32..61 = color+1, bits 0..31 = element id.
        self._partition_data_concat = wp.zeros(max_num_interactions, dtype=wp.int64, device=device)

        # int32 mirror of colour bits (0 = uncoloured) for greedy hot-path 4-byte reads.
        self._color_tags = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # Per-element assigned color (-1 = unassigned).
        self._interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # Tile-scan flag/offset buffers padded to a multiple of TILE_SCAN_BLOCK_DIM.
        if use_tile_scan:
            tile_dim = int(TILE_SCAN_BLOCK_DIM)
            padded_len = ((max_num_interactions + tile_dim - 1) // tile_dim) * tile_dim
            padded_len = max(padded_len, tile_dim)
        else:
            padded_len = max_num_interactions
        self._scan_len = padded_len
        self._flags = wp.zeros(padded_len, dtype=wp.int32, device=device)
        self._offsets = wp.zeros(padded_len, dtype=wp.int32, device=device)
        self._partition_element_ids = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # In-place compacted remaining ids (survivors pack leftward).
        self._remaining_ids = wp.zeros(padded_len, dtype=wp.int32, device=device)

        # Body-locality post-sort scratch. After the colour build
        # writes :attr:`_element_ids_by_color`, this pair-sort
        # reorders entries WITHIN each colour so consecutive slots
        # access nearby body indices. Cuts the per-thread body-data
        # scatter that dominates the iterate kernel. ``radix_sort_pairs``
        # needs 2*N buffers (ping-pong); we sort the entire
        # element-by-colour array in one call with a packed
        # ``(colour | body_min)`` key. Colour is the high 32 bits so
        # the CSR colour ordering is preserved.
        self._locality_keys = wp.zeros(2 * max_num_interactions, dtype=wp.int64, device=device)
        self._locality_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)
        self._locality_family = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        family_slots = int(MAX_COLORS) * _LOCALITY_FAMILY_COUNT_HOST
        self._color_family_count = wp.zeros(family_slots, dtype=wp.int32, device=device)
        self._color_family_starts = wp.zeros(family_slots, dtype=wp.int32, device=device)
        self._compute_color_family_starts = False

        self._current_color = wp.zeros(1, dtype=wp.int32, device=device)
        self._num_remaining = wp.zeros(1, dtype=wp.int32, device=device)
        self._partition_count = wp.zeros(1, dtype=wp.int32, device=device)

        # CSR layout (Mode B). color_starts is sized MAX_COLORS+1 so
        # color_starts[num_colors] is always addressable.
        self._element_ids_by_color = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._color_starts = wp.zeros(MAX_COLORS + 1, dtype=wp.int32, device=device)
        self._num_colors = wp.zeros(1, dtype=wp.int32, device=device)
        # Set to 1 if num_colors would exceed MAX_COLORS; checked host-side post-build.
        self._overflow_flag = wp.zeros(1, dtype=wp.int32, device=device)

        # Greedy build scratch. Bounded at GREEDY_MAX_COLORS by the int64 forbidden mask.
        self._greedy_color_count: wp.array[wp.int32] = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        self._greedy_color_offsets: wp.array[wp.int32] = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        self._greedy_grid_size: int = _greedy_coloring_grid_size(max_num_interactions, device)
        # In-graph greedy-overflow fallback predicate.
        self._fallback_flag: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=device)
        # Sweep-time colour cursor (decremented by PGS kernels).
        self._color_cursor = wp.zeros(1, dtype=wp.int32, device=device)

        # Symmetric Gauss-Seidel: flip 0/1 per begin_sweep when enabled
        # so the PGS sweep alternates forward/reverse colour order.
        # See :meth:`set_symmetric_sweep`.
        self._sweep_direction: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=device)
        self._symmetric_sweep: bool = False

        # Warm-start cache stir: periodic full invalidate +
        # round-robin per-step skip-colour to break the colouring
        # lock-in. See :meth:`set_warm_start_invalidate_period`,
        # :meth:`set_warm_start_rotate_skip`. Counter is device-side
        # so the schedule is graph-capture safe.
        self._warm_start_invalidate_counter: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=device)
        self._warm_start_invalidate_period: int = 0
        self._warm_start_skip_color_start_plus_one: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=device)
        self._warm_start_skip_color_end_plus_one: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=device)
        self._warm_start_rotate_skip_width: int = 0

        # Greedy outer-loop: ``capture_while`` (exits when
        # ``num_remaining`` hits 0) vs fixed ``MAX_GREEDY_OUTER_ITERS``
        # unroll. See :meth:`set_capture_while_greedy`.
        self._use_capture_while_greedy: bool = False

        # Speculative coloring (Çatalyürek-style): 2-phase per round
        # (pick + fused validate+commit) instead of JP-MIS's local-max
        # scan. Commits at multiple colours per round so dense graphs
        # drain in ~6-10 rounds instead of ~80. Same priority
        # permutation as MIS so determinism is preserved. Scratch
        # buffers allocated once for capture-safe pointers.
        self._use_speculative_coloring: bool = False
        self._spec_tentative_color: wp.array[wp.int32] = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # Dummies for reusing partitioning_prepare_kernel as adjacency zeroer.
        self._prepare_partition_ends_dummy = wp.zeros(1, dtype=wp.int32, device=device)
        self._prepare_max_used_color_dummy = wp.zeros(1, dtype=wp.int32, device=device)

        # Populated by :meth:`reset`.
        self._elements: wp.array | None = None
        self._num_elements: wp.array | None = None

    # ------------------------------------------------------------------
    # Setup / loop control
    # ------------------------------------------------------------------

    def set_costs_from_contacts(
        self,
        num_joints: int,
        num_contact_columns: wp.array[wp.int32],
        contact_cols: ContactColumnContainer,
    ) -> None:
        """Refresh per-cid packed JP priorities from contact-column counts.
        Joints and inactive tail cids get cost=0; contacts get the column's
        contact count clamped to 0..255."""
        wp.launch(
            _fill_packed_priorities_from_contacts_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._packed_priorities,
                self._random_values,
                contact_cols,
                num_contact_columns,
                wp.int32(num_joints),
            ],
            device=self._packed_priorities.device,
        )

    def set_locality_family(self, element_family: wp.array[wp.int32]) -> None:
        """Install per-element family tags used only by the locality post-sort.

        Generic graph-coloring tests leave the default all-zero buffer in
        place; PhoenXWorld supplies joint/contact/cloth/soft tags so the
        CSR stays grouped by row family within each colour.
        """
        self._locality_family = element_family

    def reset(
        self,
        elements: wp.array[ElementInteractionData],
        num_elements: wp.array[int],
    ) -> None:
        """Rebuild adjacency and reset loop state. Holds references to the
        passed arrays; callers must keep them alive until the next reset."""
        self._elements = elements
        self._num_elements = num_elements

        # Reuse partitioning_prepare_kernel to clear adjacency_section_end_indices.
        prepare_dim = max(1, self.max_num_nodes)
        wp.launch(
            partitioning_prepare_kernel,
            dim=prepare_dim,
            inputs=[
                self._prepare_partition_ends_dummy,
                self._prepare_max_used_color_dummy,
                self._adjacency_section_end_indices,
                0,
                self.max_num_nodes,
            ],
        )

        wp.launch(
            partitioning_adjacency_count_kernel,
            dim=self.max_num_interactions,
            inputs=[self._adjacency_section_end_indices, elements, num_elements],
        )

        scan_variable_length(self._adjacency_section_end_indices, num_elements, inclusive=False)

        wp.launch(
            partitioning_adjacency_store_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._adjacency_section_end_indices,
                self._vertex_to_adjacent_elements,
                self._partition_data_concat,
                self._color_tags,
                elements,
                num_elements,
            ],
        )

        wp.launch(
            incremental_fill_minus_one_kernel,
            dim=self.max_num_interactions,
            inputs=[self._interaction_id_to_partition, num_elements],
        )

        wp.launch(
            incremental_init_remaining_ids_kernel,
            dim=self.max_num_interactions,
            inputs=[self._remaining_ids, num_elements],
        )

        wp.launch(
            incremental_init_kernel,
            dim=1,
            inputs=[self._current_color, self._num_remaining, self._partition_count, num_elements],
        )

    def reset_loop_state_only(self) -> None:
        """Fast reset preserving the adjacency. Use when the constraint set
        is unchanged since the last :meth:`reset` (common PGS case)."""
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before reset_loop_state_only()"
        )

        wp.launch(
            incremental_reset_loop_state_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._color_tags,
                self._interaction_id_to_partition,
                self._num_elements,
            ],
        )

        wp.launch(
            incremental_init_remaining_ids_kernel,
            dim=self.max_num_interactions,
            inputs=[self._remaining_ids, self._num_elements],
        )

        wp.launch(
            incremental_init_kernel,
            dim=1,
            inputs=[
                self._current_color,
                self._num_remaining,
                self._partition_count,
                self._num_elements,
            ],
        )

    def launch(self) -> None:
        """Produce one partition at the current colour. Updates partition_element_ids,
        partition_count, interaction_id_to_partition; advances current_color and
        decrements num_remaining."""
        assert self._elements is not None and self._num_elements is not None, "reset() must be called before launch()"

        wp.launch(
            partitioning_coloring_incremental_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._packed_priorities,
                self._adjacency_section_end_indices,
                self._vertex_to_adjacent_elements,
                self._elements,
                self._remaining_ids,
                self._num_remaining,
                self._current_color,
            ],
        )

        # Fused compact + commit + advance, single-block tile-scan (graph-capture safe).
        assert self.use_tile_scan, (
            "IncrementalContactPartitioner.launch requires use_tile_scan=True. "
            "The non-tile fallback was removed when the remaining-ids buffer was "
            "introduced (wp.utils.array_scan cannot run inside wp.capture_while)."
        )
        wp.launch_tiled(
            incremental_tile_compact_remaining_and_advance_kernel,
            dim=[1],
            inputs=[
                self._partition_data_concat,
                self._remaining_ids,
                self._current_color,
                self._num_remaining,
                self._partition_count,
                self._partition_element_ids,
                self._interaction_id_to_partition,
            ],
            block_dim=int(TILE_SCAN_BLOCK_DIM),
        )

    # Mode B: build coloring once, replay across many sweeps.

    def build_csr(self, *, compute_family_starts: bool = False) -> None:
        """Run JP to completion; write element_ids_by_color, color_starts,
        num_colors. Graph-capture safe via wp.capture_while."""
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before build_csr()"
        )
        self._compute_color_family_starts = bool(compute_family_starts)
        self._build_csr_inner()

        # Surface MAX_COLORS overflow as an exception when not
        # capturing (D2H reads are illegal during capture).
        device = self._overflow_flag.device
        if device.is_cuda and device.is_capturing:
            return
        if int(self._overflow_flag.numpy()[0]) != 0:
            raise RuntimeError(
                f"PhoenX graph coloring exceeded MAX_COLORS (={MAX_COLORS}). "
                "Likely a super-hub contact body or cross-world ingest bug."
            )

    def _build_csr_inner(self) -> None:
        """Capture-safe body of :meth:`build_csr` (no host reads)."""
        wp.launch(
            incremental_reset_loop_state_csr_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._color_tags,
                self._interaction_id_to_partition,
                self._current_color,
                self._num_remaining,
                self._num_colors,
                self._color_starts,
                self._overflow_flag,
                self._num_elements,
            ],
        )
        wp.launch(
            incremental_init_remaining_ids_kernel,
            dim=self.max_num_interactions,
            inputs=[self._remaining_ids, self._num_elements],
        )
        wp.capture_while(
            self._num_remaining,
            self._capture_build_csr_step,
        )
        self._sort_csr_by_body_locality()

    def _capture_build_csr_step(self) -> None:
        """build_csr capture_while body, unrolled NUM_INNER_WHILE_ITERATIONS times.
        Tail rounds after convergence early-exit cheaply."""
        for _ in range(NUM_INNER_WHILE_ITERATIONS):
            wp.launch(
                partitioning_coloring_incremental_kernel,
                dim=self.max_num_interactions,
                inputs=[
                    self._partition_data_concat,
                    self._packed_priorities,
                    self._adjacency_section_end_indices,
                    self._vertex_to_adjacent_elements,
                    self._elements,
                    self._remaining_ids,
                    self._num_remaining,
                    self._current_color,
                ],
            )
            wp.launch_tiled(
                incremental_tile_compact_csr_and_advance_kernel,
                dim=[1],
                inputs=[
                    self._partition_data_concat,
                    self._remaining_ids,
                    self._current_color,
                    self._num_remaining,
                    self._num_colors,
                    self._element_ids_by_color,
                    self._color_starts,
                    self._interaction_id_to_partition,
                    int(MAX_COLORS),
                    self._overflow_flag,
                    wp.int32(self._max_colored_partitions_kernel_arg),
                ],
                block_dim=int(TILE_SCAN_BLOCK_DIM),
            )

    # Greedy Mode B: JP-MIS + smallest-free-colour.

    def build_csr_greedy(self, *, compute_family_starts: bool = False) -> None:
        """Greedy build, capped at :data:`GREEDY_MAX_COLORS` (64) by the int64
        forbidden mask. Raises ``RuntimeError`` on overflow. Prefer
        :meth:`build_csr_greedy_with_jp_fallback` for in-graph fallback."""
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before build_csr_greedy()"
        )
        self._compute_color_family_starts = bool(compute_family_starts)
        self._build_csr_greedy_inner()
        device = self._overflow_flag.device
        if device.is_cuda and device.is_capturing:
            return
        if int(self._overflow_flag.numpy()[0]) != 0:
            raise RuntimeError(
                f"PhoenX greedy coloring exceeded GREEDY_MAX_COLORS (={int(GREEDY_MAX_COLORS)}). "
                "Use build_csr_greedy_with_jp_fallback or build_csr instead."
            )

    def build_csr_greedy_with_jp_fallback(self, *, compute_family_starts: bool = False) -> None:
        """Greedy build with in-graph JP fallback on bitmask overflow.
        Uses _fallback_flag (separate from _overflow_flag) so JP body can clear
        it without racing JP's own MAX_COLORS accounting."""
        self._compute_color_family_starts = bool(compute_family_starts)
        self._build_csr_greedy_inner()
        wp.copy(self._fallback_flag, self._overflow_flag)
        wp.capture_while(self._fallback_flag, self._capture_jp_fallback_step)

    def _build_csr_greedy_inner(self) -> None:
        """Capture-safe body of :meth:`build_csr_greedy`."""
        wp.launch(
            greedy_reset_init_kernel,
            dim=int(GREEDY_MAX_COLORS),
            inputs=[
                self._overflow_flag,
                self._greedy_color_count,
                self._greedy_color_offsets,
                int(GREEDY_MAX_COLORS),
            ],
        )
        wp.launch(
            incremental_reset_loop_state_csr_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._color_tags,
                self._interaction_id_to_partition,
                self._current_color,
                self._num_remaining,
                self._num_colors,
                self._color_starts,
                self._overflow_flag,
                self._num_elements,
            ],
        )
        # Warm-start: seed ``color_tags`` from the cross-frame cache
        # before the MIS loop. Constraints whose previous-frame
        # colour is still legal under the current adjacency keep
        # their colour (validation pass below catches the rest);
        # constraints not in the cache stay at 0 and get coloured
        # normally by MIS.
        if self.enable_warm_start:
            from newton._src.solvers.phoenx.graph_coloring.warm_start import (  # noqa: PLC0415
                seed_warm_start_kernel,
                warm_start_invalidate_apply_kernel,
                warm_start_invalidate_mark_kernel,
            )

            # Pre-seed pass: bump the step counter and apply the
            # configured strategy for breaking the warm-start
            # coloring lock-in (periodic full invalidate, rotating
            # skip-colour, both, or neither). Capture-safe -- a
            # single-thread device-side kernel; no host roundtrip.
            wp.launch(
                warm_start_periodic_invalidate_kernel,
                dim=1,
                inputs=[
                    self._warm_start_invalidate_counter,
                    self._warm_start_cache.num_entries,
                    self._num_colors,
                    self._warm_start_skip_color_start_plus_one,
                    self._warm_start_skip_color_end_plus_one,
                    wp.int32(self._warm_start_invalidate_period),
                    wp.int32(self._warm_start_rotate_skip_width),
                ],
            )

            wp.launch(
                seed_warm_start_kernel,
                dim=self.max_num_interactions,
                inputs=[
                    self._elements,
                    self._num_elements,
                    self._warm_start_cache.keys,
                    self._warm_start_cache.colors,
                    self._warm_start_cache.num_entries,
                    self._color_tags,
                    self._partition_data_concat,
                    self._warm_start_skip_color_start_plus_one,
                    self._warm_start_skip_color_end_plus_one,
                ],
            )
            # Validation pass: detect colour conflicts under the
            # current adjacency. Two phases (mark + apply) so the
            # tie-break read-side stays race-free.
            wp.launch(
                warm_start_invalidate_mark_kernel,
                dim=self.max_num_interactions,
                inputs=[
                    self._color_tags,
                    self._elements,
                    self._adjacency_section_end_indices,
                    self._vertex_to_adjacent_elements,
                    self._num_elements,
                    wp.int32(self._max_colored_partitions_kernel_arg),
                    self._warm_start_invalid_mark,
                ],
            )
            wp.launch(
                warm_start_invalidate_apply_kernel,
                dim=self.max_num_interactions,
                inputs=[
                    self._warm_start_invalid_mark,
                    self._color_tags,
                    self._partition_data_concat,
                    self._num_elements,
                    self._num_remaining,
                ],
            )
        # Outer MIS loop. Three paths:
        #
        # 1. **Speculative coloring** (``use_speculative_coloring`` on):
        #    3-phase round (pick + validate + commit) commits at
        #    multiple colours per round; ~6-10 rounds x 3 kernels = 18-30
        #    launches on Kapla vs ~80 with JP-MIS. Race-free by
        #    construction (no concurrent ``color_tags`` writes within a
        #    phase).
        # 2. **JP-MIS + capture_while** (``use_capture_while_greedy``):
        #    legacy strict-MIS loop with ``wp.capture_while`` -- exits
        #    as soon as ``num_remaining[0]`` hits 0 instead of running
        #    the fixed cap.
        # 3. **JP-MIS + fixed unroll** (neither flag set): legacy
        #    behaviour for contexts where capture_while overhead
        #    dominates.
        # K=0 means every uncoloured row must spill to the overflow bucket;
        # skip the greedy/speculative neighbour scans and let the spill kernel
        # stamp the same overflow colour deterministically.
        if self._max_colored_partitions_kernel_arg != 0:
            if self._use_speculative_coloring:
                if self._use_capture_while_greedy:
                    wp.capture_while(self._num_remaining, self._capture_speculative_step)
                else:
                    outer_iters = (
                        self._max_greedy_outer_iters_override
                        if self._max_greedy_outer_iters_override is not None
                        else int(MAX_GREEDY_OUTER_ITERS)
                    )
                    for _ in range(outer_iters):
                        self._capture_speculative_step()
            elif self._use_capture_while_greedy:
                wp.capture_while(self._num_remaining, self._capture_build_csr_greedy_step)
            else:
                outer_iters = (
                    self._max_greedy_outer_iters_override
                    if self._max_greedy_outer_iters_override is not None
                    else int(MAX_GREEDY_OUTER_ITERS)
                )
                for _ in range(outer_iters):
                    self._capture_build_csr_greedy_step()
        # Force-spill anything still uncoloured into the overflow colour.
        # No-op when capture_while drained num_remaining naturally; only
        # fires when the iter-cap watcher triggered the early exit. Mass
        # splitting handles the overflow bucket via copy states.
        wp.launch(
            greedy_overflow_spill_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._color_tags,
                self._partition_data_concat,
                self._num_elements,
                self._num_remaining,
                self._overflow_flag,
                wp.int32(self._max_colored_partitions_kernel_arg),
            ],
        )
        if self._max_colored_partitions_kernel_arg == 0:
            wp.launch(greedy_clear_int_kernel, dim=1, inputs=[self._num_remaining])
        wp.launch(
            greedy_color_histogram_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._num_elements,
                self._greedy_color_count,
                self._interaction_id_to_partition,
            ],
        )
        wp.launch(
            greedy_count_and_scan_color_starts_kernel,
            dim=1,
            inputs=[
                self._greedy_color_count,
                self._color_starts,
                self._num_colors,
                int(GREEDY_MAX_COLORS),
            ],
        )
        wp.launch(
            greedy_scatter_elements_by_color_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._color_starts,
                self._greedy_color_offsets,
                self._element_ids_by_color,
                self._num_elements,
            ],
        )
        self._sort_csr_by_body_locality()
        # Persist this build's coloring into the warm-start cache for
        # the next frame. Reuses radix-sort + boundary-scan pipeline
        # similar to mass_splitting's build_interaction_graph.
        if self.enable_warm_start:
            self._persist_warm_start_cache()

    def _persist_warm_start_cache(self) -> None:
        """Compact ``(body_pair_key, colour)`` pairs into
        :attr:`_warm_start_cache` for the next frame.

        Pipeline: emit per coloured element (inactive tails get
        ``INT64_MAX`` keys) -> radix sort by key -> mark first
        occurrence of each unique key -> exclusive scan -> compact
        boundary entries into the cache (atomic-max sets
        ``num_entries``).
        """
        from newton._src.solvers.phoenx.graph_coloring.warm_start import (  # noqa: PLC0415
            warm_start_dedup_pairs_kernel,
            warm_start_emit_pairs_kernel,
            warm_start_mark_boundaries_kernel,
            warm_start_reset_count_kernel,
        )

        # Step 1: emit pairs.
        wp.launch(
            warm_start_emit_pairs_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._elements,
                self._color_tags,
                self._num_elements,
                self._ws_pair_keys,
                self._ws_pair_values,
            ],
        )
        # Step 2: radix sort. ``radix_sort_pairs`` needs 2*N buffers
        # (ping-pong); the emit kernel only writes the first N slots
        # but the buffer is sized 2*N already.
        wp.utils.radix_sort_pairs(self._ws_pair_keys, self._ws_pair_values, self.max_num_interactions)
        # Step 3: mark boundaries.
        wp.launch(
            warm_start_mark_boundaries_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._ws_pair_keys,
                self._num_elements,
                self._ws_is_boundary,
            ],
        )
        # Step 4: exclusive prefix scan.
        wp.utils.array_scan(self._ws_is_boundary, self._ws_dest_idx, inclusive=False)
        # Step 5: reset cache counter, then compact.
        wp.launch(
            warm_start_reset_count_kernel,
            dim=1,
            inputs=[self._warm_start_cache.num_entries],
        )
        wp.launch(
            warm_start_dedup_pairs_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._ws_pair_keys,
                self._ws_pair_values,
                self._ws_is_boundary,
                self._ws_dest_idx,
                self._color_tags,
                self._num_elements,
                self._warm_start_cache.keys,
                self._warm_start_cache.colors,
                self._warm_start_cache.num_entries,
            ],
        )

    def _sort_csr_by_body_locality(self) -> None:
        """Sort each colour slice of ``element_ids_by_color`` for iterate
        locality.

        Regular colours use ``(colour, family, body_min, eid)`` so rows
        with the same dispatch path run together while retaining body
        locality inside each family. The mass-splitting overflow colour
        disables the family bits and preserves the previous
        ``(colour, body_min, eid)`` order, because overflow copy-state
        batch ids are derived from CSR position.
        """
        n = self.max_num_interactions
        wp.launch(
            _locality_combined_keys_kernel,
            dim=n,
            inputs=[
                self._elements,
                self._element_ids_by_color,
                self._interaction_id_to_partition,
                self._locality_family,
                wp.int32(self._max_colored_partitions_kernel_arg),
                self._num_elements,
                self._locality_keys,
                self._locality_values,
            ],
        )
        wp.utils.radix_sort_pairs(self._locality_keys, self._locality_values, n)
        wp.launch(
            _locality_writeback_kernel,
            dim=n,
            inputs=[
                self._locality_values,
                self._num_elements,
                self._element_ids_by_color,
            ],
        )
        if not self._compute_color_family_starts:
            return
        wp.launch(
            _zero_color_family_count_kernel,
            dim=self._color_family_count.shape[0],
            inputs=[self._color_family_count],
        )
        wp.launch(
            _count_color_families_kernel,
            dim=n,
            inputs=[
                self._element_ids_by_color,
                self._interaction_id_to_partition,
                self._locality_family,
                wp.int32(self._max_colored_partitions_kernel_arg),
                self._num_elements,
                self._color_family_count,
            ],
        )
        wp.launch(
            _scan_color_family_starts_kernel,
            dim=MAX_COLORS,
            inputs=[
                self._color_starts,
                self._color_family_count,
                self._color_family_starts,
            ],
        )

    def _capture_jp_fallback_step(self) -> None:
        """Greedy->JP fallback capture_while body. Clears fallback flag so the
        outer capture_while exits after one iteration."""
        self._build_csr_inner()
        wp.launch(greedy_clear_int_kernel, dim=1, inputs=[self._fallback_flag])

    def _capture_build_csr_greedy_step(self) -> None:
        """build_csr_greedy capture_while body. Unrolled NUM_INNER_WHILE_ITERATIONS
        times to amortise the outer overhead. After the unrolled launches a
        single-thread watcher kernel bumps the outer-iteration counter and
        force-exits the capture_while when ``MAX_GREEDY_OUTER_ITERS`` is hit
        (only when an overflow bucket is configured -- otherwise the loop
        runs to natural convergence)."""
        for _ in range(NUM_INNER_WHILE_ITERATIONS):
            wp.launch(
                partitioning_coloring_incremental_greedy_kernel,
                dim=self._greedy_grid_size,
                inputs=[
                    self._partition_data_concat,
                    self._color_tags,
                    self._packed_priorities,
                    self._adjacency_section_end_indices,
                    self._vertex_to_adjacent_elements,
                    self._elements,
                    self._num_elements,
                    wp.int32(self._greedy_grid_size),
                    self._num_remaining,
                    self._overflow_flag,
                    wp.int32(self._max_colored_partitions_kernel_arg),
                ],
                block_dim=_GREEDY_BLOCK_DIM,
            )
        # No iter-cap watcher needed: the outer host-side loop in
        # :meth:`_build_csr_greedy_inner` is bounded by
        # ``MAX_GREEDY_OUTER_ITERS`` directly. Per-thread ``color_tags
        # [tid] != 0`` check in the greedy kernel makes post-
        # convergence iters cheap no-ops.

    def _capture_speculative_step(self) -> None:
        """Speculative coloring outer-loop body (one round = 2 phases).

        Pick -> ValidateCommit. The fused validate+commit kernel is
        race-safe under the priority-tiebreak rule: when two
        constraints sharing a body both pick colour ``c``, the
        lower-priority one always aborts regardless of whether it
        reads the higher-priority one as "uncoloured but higher
        priority" (pre-commit) or "coloured at same colour"
        (post-commit). Cuts one launch per round vs the 3-phase
        pipeline.

        Each round commits a spread of colours, so Kapla drains in
        ~6-10 rounds vs ~80 for strict JP-MIS. Unrolled by
        ``NUM_INNER_WHILE_ITERATIONS`` to amortise the outer
        ``capture_while`` predicate-check overhead.
        """
        for _ in range(NUM_INNER_WHILE_ITERATIONS):
            wp.launch(
                speculative_pick_kernel,
                dim=self._greedy_grid_size,
                inputs=[
                    self._color_tags,
                    self._adjacency_section_end_indices,
                    self._vertex_to_adjacent_elements,
                    self._elements,
                    self._num_elements,
                    wp.int32(self._greedy_grid_size),
                    self._num_remaining,
                    self._overflow_flag,
                    wp.int32(self._max_colored_partitions_kernel_arg),
                    self._spec_tentative_color,
                ],
                block_dim=_GREEDY_BLOCK_DIM,
            )
            wp.launch(
                speculative_validate_commit_kernel,
                dim=self._greedy_grid_size,
                inputs=[
                    self._partition_data_concat,
                    self._color_tags,
                    self._spec_tentative_color,
                    self._packed_priorities,
                    self._adjacency_section_end_indices,
                    self._vertex_to_adjacent_elements,
                    self._elements,
                    self._num_elements,
                    wp.int32(self._greedy_grid_size),
                    self._num_remaining,
                    wp.int32(self._max_colored_partitions_kernel_arg),
                ],
                block_dim=_GREEDY_BLOCK_DIM,
            )
            # Force exit when the forbidden mask saturated -- without
            # this the capture_while spins forever on dense graphs
            # whose chromatic number exceeds GREEDY_MAX_COLORS (64).
            # JP fallback picks up the still-uncoloured stragglers.
            wp.launch(
                speculative_overflow_exit_kernel,
                dim=1,
                inputs=[self._overflow_flag, self._num_remaining],
            )

    def begin_sweep(self) -> None:
        """Reset the sweep-time colour cursor (copy num_colors -> color_cursor).
        Call before every PGS sweep. Also bumps ``sweep_direction``
        when cyclic color sweep is enabled (see :meth:`set_symmetric_sweep`)."""
        advance = wp.int32(1) if self._symmetric_sweep else wp.int32(0)
        wp.launch(
            incremental_begin_sweep_kernel,
            dim=1,
            inputs=[self._num_colors, self._color_cursor, self._sweep_direction, advance],
        )

    def set_warm_start_invalidate_period(self, period: int) -> None:
        """Force a full cold-start re-coloring every ``period``
        ``build_csr`` calls. ``0`` disables. Breaks the warm-start
        coloring lock-in that biases the PGS solve on stable scenes."""
        if period < 0:
            raise ValueError(f"period must be >= 0 (got {period})")
        self._warm_start_invalidate_period = int(period)

    def set_capture_while_greedy(self, enabled: bool) -> None:
        """Use ``wp.capture_while(num_remaining, ...)`` instead of the
        fixed ``MAX_GREEDY_OUTER_ITERS`` unroll on the greedy MIS outer
        loop. Skips post-convergence launches when the MIS finishes
        early."""
        self._use_capture_while_greedy = bool(enabled)

    def set_speculative_coloring(self, enabled: bool) -> None:
        """Use Çatalyürek-style speculative coloring (pick + fused
        validate-commit per round) instead of JP-MIS. Commits at
        multiple colours per round so dense graphs drain in fewer
        rounds. Deterministic via the same fixed priority permutation."""
        self._use_speculative_coloring = bool(enabled)

    def set_warm_start_rotate_skip(self, enabled: bool, width: int = 1) -> None:
        """Each ``build_csr`` skips re-seeding ``width`` consecutive
        cached colours (round-robin), forcing them to re-MIS while
        the rest stay warm-started. Cost ~``width / num_colors`` of
        a cold-start per step. ``enabled=False`` disables."""
        if width < 0:
            raise ValueError(f"width must be >= 0 (got {width})")
        self._warm_start_rotate_skip_width = int(width) if enabled else 0

    def set_symmetric_sweep(self, enabled: bool) -> None:
        """Flip the PGS colour sweep direction on every ``begin_sweep``
        (symmetric Gauss-Seidel). Counters iteration-order bias from a
        locked-in warm-start coloring."""
        self._symmetric_sweep = bool(enabled)

    # Public device arrays (results).

    @property
    def num_remaining(self) -> wp.array:
        """Elements not yet assigned. Stop launching when 0."""
        return self._num_remaining

    @property
    def current_color(self) -> wp.array:
        """Index of the *next* partition launch will produce."""
        return self._current_color

    @property
    def partition_count(self) -> wp.array:
        """Element count from the most recent launch."""
        return self._partition_count

    @property
    def partition_element_ids(self) -> wp.array:
        """Element ids from the most recent launch (first partition_count valid)."""
        return self._partition_element_ids

    # Mode B (CSR) results.

    @property
    def element_ids_by_color(self) -> wp.array:
        """Colour ``c`` owns element_ids_by_color[color_starts[c]:color_starts[c+1]]."""
        return self._element_ids_by_color

    @property
    def color_starts(self) -> wp.array:
        """CSR exclusive-prefix offsets. color_starts[num_colors[0]] = total."""
        return self._color_starts

    @property
    def color_family_starts(self) -> wp.array:
        """Family starts for each colour, length ``MAX_COLORS * 6``."""
        return self._color_family_starts

    @property
    def num_colors(self) -> wp.array:
        """Number of colours produced by the last build_csr call."""
        return self._num_colors

    @property
    def color_cursor(self) -> wp.array:
        """Sweep-time colour countdown (init by begin_sweep, decremented by PGS)."""
        return self._color_cursor

    @property
    def sweep_direction(self) -> wp.array:
        """Symmetric-GS flag (length 1): 0=forward, 1=reverse. The
        iterate kernels read this when computing which colour to
        visit. ``begin_sweep`` toggles it (when symmetric sweep is
        enabled). See :meth:`set_symmetric_sweep`.
        """
        return self._sweep_direction

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """Per-element colour, -1 if uncoloured."""
        return self._interaction_id_to_partition
