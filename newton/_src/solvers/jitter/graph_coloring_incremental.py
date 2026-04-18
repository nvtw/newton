"""Incremental Jones-Plassmann graph partitioner.

Companion to :class:`newton._src.solvers.jitter.graph_coloring.ContactPartitioner`
that produces **one** partition per :meth:`launch` call instead of running the
entire colouring pipeline up front. Compared to the batch version:

* No ``max_num_partitions`` budget and no "overflow" partition: the caller
  repeatedly calls :meth:`launch` until ``num_remaining[0] == 0``.
* No key-value sort. The per-call compaction uses an exclusive prefix scan,
  which is deterministic; two identical reset+launch sequences therefore
  produce byte-identical outputs on every buffer.
* All loop state (current colour, remaining count, per-call partition count)
  lives in 1-element device arrays, so a fixed-length sequence of launches can
  be captured into a single Warp graph.

Typical usage::

    partitioner = IncrementalContactPartitioner(N, num_bodies, device=d)
    partitioner.reset(elements_arr, num_elements_arr)
    while int(partitioner.num_remaining.numpy()[0]) > 0:
        partitioner.launch()
        # partitioner.partition_element_ids[:partitioner.partition_count[0]]
        # holds the element ids assigned to partition partitioner.current_color[0]
"""

import warp as wp

from newton._src.solvers.jitter.graph_coloring_common import (
    MAX_BODIES,
    TILE_SCAN_BLOCK_DIM,
    ElementInteractionData,
    incremental_advance_kernel,
    incremental_compact_kernel,
    incremental_fill_minus_one_kernel,
    incremental_flag_kernel,
    incremental_init_kernel,
    incremental_zero_int_kernel,
    partitioning_adjacency_count_kernel,
    partitioning_adjacency_store_kernel,
    partitioning_coloring_kernel,
    partitioning_prepare_kernel,
    tile_scan_exclusive_block_kernel,
)
from newton._src.solvers.jitter.scan_and_sort import scan_variable_length

__all__ = ["IncrementalContactPartitioner"]


class IncrementalContactPartitioner:
    """Jones-Plassmann partitioner that produces one partition per launch.

    Workflow:

    1. Construct once with the capacities (``max_num_interactions``,
       ``max_num_nodes``). All device buffers are allocated up front.
    2. Call :meth:`reset` whenever the element set changes. This rebuilds the
       adjacency structure and resets the loop state.
    3. Call :meth:`launch` to produce the next partition. The resulting
       element list is exposed via :attr:`partition_element_ids` (length
       :attr:`partition_count`); the colour used is :attr:`current_color`
       (which is the colour of the most recent call).
    4. Stop when :attr:`num_remaining` reaches zero.
    """

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.DeviceLike = None,
        seed: int = 0,
        use_tile_scan: bool = False,
    ) -> None:
        self.max_num_interactions = max_num_interactions
        self.max_num_nodes = max_num_nodes
        # C# convention: max_num_contacts == max_num_interactions in single-section mode.
        self.max_num_contacts = max_num_interactions
        # When True, :meth:`launch` uses a single-block tile-scan kernel
        # (graph-capture safe, no implicit allocations) instead of
        # ``wp.utils.array_scan``. Flag/offset buffers are padded to a
        # multiple of ``TILE_SCAN_BLOCK_DIM`` so the fixed-size tile loads
        # never read past the end of the allocation.
        self.use_tile_scan = use_tile_scan

        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)

        # Single-section mode: marker at the end disables the offset in
        # ``contact_partitions_get_random_value``.
        self._section_marker = wp.array([max_num_interactions], dtype=wp.int32, device=device)

        # Adjacency storage.
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(
            max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device
        )

        # Per-element packed tag: bit 62 unpartitioned marker + (color+1) in
        # bits 32..61 + element id in bits 0..31. No ping-pong 2*N buffer is
        # needed because we never sort.
        self._partition_data_concat = wp.zeros(max_num_interactions, dtype=wp.int64, device=device)

        # Per-element color (-1 until assigned).
        self._interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # Per-call compaction scratch / outputs. When tile-scan is enabled,
        # the flag/offset buffers are rounded up to a multiple of the tile
        # size so ``wp.tile_load`` never reads past the allocation. The
        # padded tail stays zero-filled (incremental_zero_int_kernel only
        # writes up to num_elements[0], but the initial wp.zeros guarantees
        # the tail is zero for the lifetime of the object).
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

        # Scalar device-side loop state (1-element arrays).
        self._current_color = wp.zeros(1, dtype=wp.int32, device=device)
        self._num_remaining = wp.zeros(1, dtype=wp.int32, device=device)
        self._partition_count = wp.zeros(1, dtype=wp.int32, device=device)

        # Dummies so partitioning_prepare_kernel can be reused for adjacency zeroing.
        # max_num_partitions=0 -> partition_ends slot [0] is written; we never read it.
        self._prepare_partition_ends_dummy = wp.zeros(1, dtype=wp.int32, device=device)
        self._prepare_max_used_color_dummy = wp.zeros(1, dtype=wp.int32, device=device)

        # Populated by :meth:`reset`.
        self._elements: wp.array | None = None
        self._num_elements: wp.array | None = None

    # ------------------------------------------------------------------
    # Setup / loop control
    # ------------------------------------------------------------------

    def reset(
        self,
        elements: wp.array[ElementInteractionData],
        num_elements: wp.array[int],
    ) -> None:
        """Rebuild adjacency for the supplied element set and reset loop state.

        Must be called before the first :meth:`launch` and whenever the
        ``elements`` layout changes. Keeps references to ``elements`` and
        ``num_elements``; callers must not free them until the partitioner
        is discarded or :meth:`reset` is called again with new arrays.
        """
        self._elements = elements
        self._num_elements = num_elements

        # 1. Clear adjacency_section_end_indices[0..max_num_nodes).
        # Re-using the batch ``partitioning_prepare_kernel`` keeps the set of
        # compiled kernels small; max_num_partitions=0 means only partition
        # ends slot [0] is written (into a throwaway dummy buffer).
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

        # 2. Count vertex references per element.
        wp.launch(
            partitioning_adjacency_count_kernel,
            dim=self.max_num_interactions,
            inputs=[self._adjacency_section_end_indices, elements, num_elements],
        )

        # 3. Exclusive scan -> adjacency_section_end_indices[v] is now the start
        #    offset for vertex v's element list. (Scan writes into the full
        #    array; the store kernel only uses the first num_elements[0]
        #    active entries.)
        scan_variable_length(self._adjacency_section_end_indices, num_elements, inclusive=False)

        # 4. Scatter element ids into the adjacency lists and initialize
        #    partition_data_concat[tid] = _UNPARTITIONED | tid.
        wp.launch(
            partitioning_adjacency_store_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._adjacency_section_end_indices,
                self._vertex_to_adjacent_elements,
                self._partition_data_concat,
                elements,
                num_elements,
            ],
        )

        # 5. Reset per-element assigned-color to -1 (so unassigned elements
        #    are visible in interaction_id_to_partition).
        wp.launch(
            incremental_fill_minus_one_kernel,
            dim=self.max_num_interactions,
            inputs=[self._interaction_id_to_partition, num_elements],
        )

        # 6. Initialize device-side loop state (current_color=0,
        #    num_remaining=num_elements[0], partition_count=0).
        wp.launch(
            incremental_init_kernel,
            dim=1,
            inputs=[self._current_color, self._num_remaining, self._partition_count, num_elements],
        )

    def launch(self) -> None:
        """Produce one partition using Jones-Plassmann at the current colour.

        After the call:

        * :attr:`partition_element_ids` holds the assigned element ids,
          :attr:`partition_count` their count.
        * :attr:`interaction_id_to_partition` has the new colour recorded for
          each committed element.
        * :attr:`num_remaining` is decremented by :attr:`partition_count`.
        * :attr:`current_color` is advanced to the next colour.

        :meth:`reset` must have been called at least once before this method.
        """
        assert self._elements is not None and self._num_elements is not None, "reset() must be called before launch()"

        # 1. JP coloring pass for partition current_color[0].
        # ``max_used_color`` is not consumed by the incremental API, so we
        # reuse the prepare dummy to satisfy the shared kernel's signature.
        wp.launch(
            partitioning_coloring_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._prepare_partition_ends_dummy,
                self._prepare_max_used_color_dummy,
                self._random_values,
                self._adjacency_section_end_indices,
                self._vertex_to_adjacent_elements,
                self.max_num_contacts,
                self._elements,
                self._num_elements,
                self._section_marker,
                self._current_color,
            ],
        )

        # 2. Zero flags[0..num_elements[0]) then set flags[tid]=1 for
        #    elements just committed to the current partition.
        wp.launch(
            incremental_zero_int_kernel,
            dim=self.max_num_interactions,
            inputs=[self._flags, self._num_elements],
        )
        wp.launch(
            incremental_flag_kernel,
            dim=self.max_num_interactions,
            inputs=[self._partition_data_concat, self._current_color, self._flags, self._num_elements],
        )

        # 3. Exclusive prefix scan of flags -> offsets. Deterministic, which is
        #    what gives the partitioner byte-for-byte repeatable outputs.
        if self.use_tile_scan:
            # Single-block tile scan: graph-capture safe (no implicit
            # allocations) but sequential across the grid-stride tail.
            wp.launch_tiled(
                tile_scan_exclusive_block_kernel,
                dim=[1],
                inputs=[self._flags, self._offsets],
                block_dim=int(TILE_SCAN_BLOCK_DIM),
            )
        else:
            wp.utils.array_scan(self._flags, self._offsets, inclusive=False)

        # 4. Compact: write partition_element_ids[offsets[tid]] = tid for
        #    flagged threads, update interaction_id_to_partition, publish count.
        wp.launch(
            incremental_compact_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._flags,
                self._offsets,
                self._current_color,
                self._partition_element_ids,
                self._interaction_id_to_partition,
                self._partition_count,
                self._num_elements,
            ],
        )

        # 5. Advance device-side loop state: num_remaining -= partition_count,
        #    current_color += 1.
        wp.launch(
            incremental_advance_kernel,
            dim=1,
            inputs=[self._current_color, self._num_remaining, self._partition_count],
        )

    # ------------------------------------------------------------------
    # Public device arrays (results)
    # ------------------------------------------------------------------

    @property
    def num_remaining(self) -> wp.array:
        """Number of elements not yet assigned to any partition (device
        scalar array). Stop calling :meth:`launch` once this reaches 0."""
        return self._num_remaining

    @property
    def current_color(self) -> wp.array:
        """Index of the *next* partition that :meth:`launch` will produce
        (device scalar array). After a :meth:`launch` call returns, the colour
        of the partition just produced is ``current_color[0] - 1``."""
        return self._current_color

    @property
    def partition_count(self) -> wp.array:
        """Number of element ids written to :attr:`partition_element_ids` by
        the most recent :meth:`launch` call (device scalar array)."""
        return self._partition_count

    @property
    def partition_element_ids(self) -> wp.array:
        """Compacted list of element ids assigned to the most recently
        produced partition. Only the first :attr:`partition_count` entries
        are meaningful (device int32 array)."""
        return self._partition_element_ids

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """``interaction_id_to_partition[i]`` is the colour that element ``i``
        was assigned to, or ``-1`` if the element has not been colored yet
        (device int32 array)."""
        return self._interaction_id_to_partition
