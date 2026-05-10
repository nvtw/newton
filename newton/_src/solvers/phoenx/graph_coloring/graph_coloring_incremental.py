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
    TILE_SCAN_BLOCK_DIM,
    ElementInteractionData,
    greedy_clear_int_kernel,
    greedy_color_histogram_kernel,
    greedy_count_and_scan_color_starts_kernel,
    greedy_reset_init_kernel,
    greedy_scatter_elements_by_color_kernel,
    incremental_begin_sweep_kernel,
    incremental_fill_minus_one_kernel,
    incremental_init_csr_kernel,
    incremental_init_kernel,
    incremental_init_remaining_ids_kernel,
    incremental_reset_loop_state_kernel,
    incremental_tile_compact_csr_and_advance_kernel,
    incremental_tile_compact_remaining_and_advance_kernel,
    partitioning_adjacency_count_kernel,
    partitioning_adjacency_store_kernel,
    partitioning_coloring_incremental_greedy_kernel,
    partitioning_coloring_incremental_kernel,
    partitioning_prepare_kernel,
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


@wp.kernel(enable_backward=False)
def _fill_cost_values_from_contacts_kernel(
    cost_values: wp.array[wp.int32],
    contact_cols: ContactColumnContainer,
    num_contact_columns: wp.array[wp.int32],
    num_joints: wp.int32,
):
    tid = wp.tid()
    local_cid = tid - num_joints
    if local_cid >= wp.int32(0) and local_cid < num_contact_columns[0]:
        cost_values[tid] = contact_get_contact_count(contact_cols, local_cid)
    else:
        cost_values[tid] = wp.int32(0)


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
        max_partitions: int | None = None,
    ) -> None:
        self.max_num_interactions = max_num_interactions
        self.max_num_nodes = max_num_nodes
        # Tile-scan: single-block, graph-capture safe, no implicit allocations.
        self.use_tile_scan = use_tile_scan

        # Mass-splitting cap: when set, the build clamps to this many
        # MIS partitions (Jitter2 ``MaxNumPartitions`` -- typically 8 or
        # 12) and dumps any leftover elements into one extra "overflow"
        # colour at index ``max_partitions``. Mass splitting consumes
        # the overflow with copy states. ``None`` keeps the legacy
        # uncapped behaviour (raise on hitting MAX_COLORS).
        if max_partitions is not None:
            if max_partitions <= 0:
                raise ValueError(f"max_partitions must be > 0 (got {max_partitions})")
            # The overflow bucket sits at index ``max_partitions``, so
            # we need ``color_starts[max_partitions + 1]`` to be a
            # valid slot. ``color_starts`` is sized ``MAX_COLORS + 1``;
            # ensure ``max_partitions + 1 <= MAX_COLORS``.
            if max_partitions + 1 > MAX_COLORS:
                raise ValueError(
                    f"max_partitions {max_partitions} + overflow exceeds MAX_COLORS {MAX_COLORS}"
                )
            # Greedy-mode forbidden-mask bound. ``color_cap`` is
            # clamped against the int64 budget regardless.
            if max_partitions > int(GREEDY_MAX_COLORS):
                raise ValueError(
                    f"max_partitions {max_partitions} exceeds GREEDY_MAX_COLORS {int(GREEDY_MAX_COLORS)}"
                )
        self.max_partitions: int | None = max_partitions

        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)
        # Per-cid JP cost; refreshed each step from contact counts (or zero).
        self._cost_values = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

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
        # Mass-splitting overflow bucket flag. When ``max_partitions``
        # is set and the build hit the cap, this is 1 and the last
        # colour (``num_colors[0] - 1`` == ``max_partitions``) holds
        # the elements that didn't fit into any independent set.
        self._has_overflow_partition = wp.zeros(1, dtype=wp.int32, device=device)

        # Greedy build scratch. Bounded at GREEDY_MAX_COLORS by the int64 forbidden mask.
        self._greedy_color_count: wp.array[wp.int32] = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        self._greedy_color_offsets: wp.array[wp.int32] = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        self._greedy_grid_size: int = _greedy_coloring_grid_size(max_num_interactions, device)
        # In-graph greedy-overflow fallback predicate.
        self._fallback_flag: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=device)

        # Sweep-time colour cursor (decremented by PGS kernels).
        self._color_cursor = wp.zeros(1, dtype=wp.int32, device=device)

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
        """Refresh per-cid JP costs from contact-column contact counts.
        Joints and inactive tail cids get 0."""
        wp.launch(
            _fill_cost_values_from_contacts_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._cost_values,
                contact_cols,
                num_contact_columns,
                wp.int32(num_joints),
            ],
            device=self._cost_values.device,
        )

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
                self._random_values,
                self._cost_values,
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

    def build_csr(self) -> None:
        """Run JP to completion; write element_ids_by_color, color_starts,
        num_colors. Graph-capture safe via wp.capture_while."""
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before build_csr()"
        )
        self._build_csr_inner()

        # Surface MAX_COLORS overflow as an exception when not
        # capturing (D2H reads are illegal during capture).
        device = self._overflow_flag.device
        if device.is_cuda and device.is_capturing:
            return
        # In mass-splitting mode the kernel handles "cap exceeded" by
        # writing the survivors into the overflow bucket instead of
        # setting ``overflow_flag``; nothing to raise here.
        if self.max_partitions is not None:
            return
        if int(self._overflow_flag.numpy()[0]) != 0:
            raise RuntimeError(
                f"PhoenX graph coloring exceeded MAX_COLORS (={MAX_COLORS}). "
                "Likely a super-hub contact body or cross-world ingest bug."
            )

    def _build_csr_inner(self) -> None:
        """Capture-safe body of :meth:`build_csr` (no host reads)."""
        wp.launch(
            incremental_init_csr_kernel,
            dim=1,
            inputs=[
                self._current_color,
                self._num_remaining,
                self._num_colors,
                self._color_starts,
                self._num_elements,
            ],
        )
        # Reset the MAX_COLORS overflow flag for this build.
        self._overflow_flag.zero_()
        # Reset the mass-splitting overflow-bucket flag too. Must run
        # every build (not just when ``max_partitions`` is set) because
        # the JP fallback path may flip into mass-splitting mode after
        # a previous greedy build set this flag.
        self._has_overflow_partition.zero_()
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
        wp.capture_while(
            self._num_remaining,
            self._capture_build_csr_step,
        )

    def _capture_build_csr_step(self) -> None:
        """build_csr capture_while body, unrolled NUM_INNER_WHILE_ITERATIONS times.
        Tail rounds after convergence early-exit cheaply."""
        # When mass-splitting cap is active, the JP loop runs at most
        # ``max_partitions`` rounds; round ``max_partitions`` itself
        # drains the survivors into the overflow bucket. ``max_colors``
        # passed to the kernel is the *per-MIS-colour* cap so it
        # triggers the overflow path on the (max_partitions+1)-th
        # round, not after MAX_COLORS rounds.
        if self.max_partitions is not None:
            jp_max_colors = int(self.max_partitions)
            overflow_partition_id = int(self.max_partitions)
        else:
            jp_max_colors = int(MAX_COLORS)
            overflow_partition_id = -1
        for _ in range(NUM_INNER_WHILE_ITERATIONS):
            wp.launch(
                partitioning_coloring_incremental_kernel,
                dim=self.max_num_interactions,
                inputs=[
                    self._partition_data_concat,
                    self._random_values,
                    self._cost_values,
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
                    jp_max_colors,
                    self._overflow_flag,
                    overflow_partition_id,
                    self._has_overflow_partition,
                ],
                block_dim=int(TILE_SCAN_BLOCK_DIM),
            )

    # Greedy Mode B: JP-MIS + smallest-free-colour.

    def build_csr_greedy(self) -> None:
        """Greedy build, capped at :data:`GREEDY_MAX_COLORS` (64) by the int64
        forbidden mask. Raises ``RuntimeError`` on overflow. Prefer
        :meth:`build_csr_greedy_with_jp_fallback` for in-graph fallback.

        With ``max_partitions`` set, this method has no overflow
        bucket of its own -- if greedy can't place every element in
        the first ``max_partitions`` colours, the user must use
        :meth:`build_csr_greedy_with_jp_fallback` (the JP fallback
        is what materialises the overflow bucket)."""
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before build_csr_greedy()"
        )
        self._build_csr_greedy_inner()
        device = self._overflow_flag.device
        if device.is_cuda and device.is_capturing:
            return
        if int(self._overflow_flag.numpy()[0]) != 0:
            cap = int(self.max_partitions) if self.max_partitions is not None else int(GREEDY_MAX_COLORS)
            raise RuntimeError(
                f"PhoenX greedy coloring exceeded color cap (={cap}). "
                "Use build_csr_greedy_with_jp_fallback or build_csr instead."
            )

    def build_csr_greedy_with_jp_fallback(self) -> None:
        """Greedy build with in-graph JP fallback on bitmask overflow.
        Uses _fallback_flag (separate from _overflow_flag) so JP body can clear
        it without racing JP's own MAX_COLORS accounting."""
        self._build_csr_greedy_inner()
        wp.copy(self._fallback_flag, self._overflow_flag)
        wp.capture_while(self._fallback_flag, self._capture_jp_fallback_step)

    def _build_csr_greedy_inner(self) -> None:
        """Capture-safe body of :meth:`build_csr_greedy`."""
        wp.launch(
            incremental_init_csr_kernel,
            dim=1,
            inputs=[
                self._current_color,
                self._num_remaining,
                self._num_colors,
                self._color_starts,
                self._num_elements,
            ],
        )
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
            incremental_reset_loop_state_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._color_tags,
                self._interaction_id_to_partition,
                self._num_elements,
            ],
        )
        wp.capture_while(
            self._num_remaining,
            self._capture_build_csr_greedy_step,
        )
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

    def _capture_jp_fallback_step(self) -> None:
        """Greedy->JP fallback capture_while body. Clears fallback flag so the
        outer capture_while exits after one iteration."""
        self._build_csr_inner()
        wp.launch(greedy_clear_int_kernel, dim=1, inputs=[self._fallback_flag])

    def _capture_build_csr_greedy_step(self) -> None:
        """build_csr_greedy capture_while body. Unrolled NUM_INNER_WHILE_ITERATIONS
        times to amortise the outer overhead."""
        # Mass-splitting cap: forbid greedy colour assignments past
        # ``max_partitions``. Anything that would overflow trips the
        # JP fallback (where the explicit overflow bucket lives), so
        # the cap propagates correctly through both build paths.
        if self.max_partitions is not None:
            color_cap = wp.int32(int(self.max_partitions))
        else:
            color_cap = wp.int32(int(GREEDY_MAX_COLORS))
        for _ in range(NUM_INNER_WHILE_ITERATIONS):
            wp.launch(
                partitioning_coloring_incremental_greedy_kernel,
                dim=self._greedy_grid_size,
                inputs=[
                    self._partition_data_concat,
                    self._color_tags,
                    self._random_values,
                    self._cost_values,
                    self._adjacency_section_end_indices,
                    self._vertex_to_adjacent_elements,
                    self._elements,
                    self._num_elements,
                    wp.int32(self._greedy_grid_size),
                    self._num_remaining,
                    self._overflow_flag,
                    color_cap,
                ],
                block_dim=_GREEDY_BLOCK_DIM,
            )

    def begin_sweep(self) -> None:
        """Reset the sweep-time colour cursor (copy num_colors -> color_cursor).
        Call before every PGS sweep."""
        wp.launch(
            incremental_begin_sweep_kernel,
            dim=1,
            inputs=[self._num_colors, self._color_cursor],
        )

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
    def num_colors(self) -> wp.array:
        """Number of colours produced by the last build_csr call."""
        return self._num_colors

    @property
    def has_overflow_partition(self) -> wp.array:
        """1 if the last build hit the ``max_partitions`` cap and dumped
        survivors into the overflow bucket at colour ``num_colors[0]-1``,
        0 otherwise. Always 0 when ``max_partitions is None``.
        Length-1 device array."""
        return self._has_overflow_partition

    @property
    def color_cursor(self) -> wp.array:
        """Sweep-time colour countdown (init by begin_sweep, decremented by PGS)."""
        return self._color_cursor

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """Per-element colour, -1 if uncoloured."""
        return self._interaction_id_to_partition
