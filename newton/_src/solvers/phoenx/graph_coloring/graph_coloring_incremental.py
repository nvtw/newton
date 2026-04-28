"""Incremental Jones-Plassmann graph partitioner.

Companion to :class:`newton._src.solvers.phoenx.graph_coloring.graph_coloring.ContactPartitioner`
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
    incremental_begin_sweep_rotate_kernel,
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

# Upper bound on the number of colours (partitions) the incremental
# partitioner can produce in one coloring pass. Picked large enough
# that no realistic physics constraint graph can exceed it -- contact
# graphs from dense stacks top out at ~60-80 colours, and pathological
# synthetic meshes stay well below 200. 1024 gives a comfortable
# margin at negligible memory cost (``color_starts`` is
# ``(MAX_COLORS + 1) * 4 B = 4100 B``).
MAX_COLORS = 1024

#: Block dim for the greedy MIS+colour kernel's persistent grid.
#: Matches the value :mod:`solver_phoenx_kernels` uses for the PGS
#: sweeps; 256 saturates Blackwell SMs without dropping occupancy on
#: register-heavy bodies.
_GREEDY_BLOCK_DIM: int = 256


def _greedy_coloring_grid_size(max_num_interactions: int, device: wp.DeviceLike) -> int:
    """Pick a persistent-grid thread count for the greedy coloring kernel.

    The MIS+colour kernel is *light* per element (one tag-word read +
    a few neighbour reads when uncoloured) and uses a grid-stride
    loop, so we deliberately under-provision relative to the
    constraint capacity: a smaller grid means fewer warps to
    schedule per launch, fewer empty lanes when most elements have
    already been coloured (late JP rounds), and better register
    reuse across grid-strided iterations on a single thread.

    Picks ``min(capacity_blocks, 1 block per SM)`` blocks of 256
    threads. On Blackwell sm_120 (~120 SMs) that's ~30k threads
    max; for a 73k-element Kapla graph each thread runs ~3
    iterations, an order of magnitude fewer launched threads than
    the pre-grid-stride 73k-thread launch with the same throughput.
    The 8-block floor keeps tiny graphs from running on a single
    warp.
    """
    block_dim = _GREEDY_BLOCK_DIM
    device_obj = wp.get_device(device)
    if device_obj.is_cuda:
        # 1 block per SM. The kernel is light enough that 4 blocks/SM
        # buys nothing -- the grid-stride loop already amortises the
        # launch overhead, and each colour-round commit serialises
        # behind a global atomic on ``any_uncolored`` regardless of
        # block count.
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
    """Jones-Plassmann partitioner with two usage modes.

    Mode A -- per-colour streaming (standalone coloring tests):
        Construct with capacities, :meth:`reset` whenever the element
        set changes, then call :meth:`launch` per colour. Each call
        publishes one partition to :attr:`partition_element_ids`
        (length :attr:`partition_count`, colour
        :attr:`current_color`). Stop when :attr:`num_remaining == 0`.

    Mode B -- CSR build once, replay many (used by :class:`World`):
        After :meth:`reset`, :meth:`build_csr` drives the full
        Jones-Plassmann loop via ``wp.capture_while`` and writes two
        CSR arrays:

        * :attr:`element_ids_by_color` -- flat concat of every
          colour's element ids.
        * :attr:`color_starts` -- exclusive prefix; colour ``c`` owns
          ``element_ids_by_color[color_starts[c]:color_starts[c+1]]``.

        :attr:`num_colors` (capped at :data:`MAX_COLORS`) holds the
        final count. The constraint graph is frozen inside a
        ``World.step`` call, so one build is valid for every substep
        and PGS iteration of the step.
    """

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.DeviceLike = None,
        seed: int = 0,
        use_tile_scan: bool = True,
    ) -> None:
        self.max_num_interactions = max_num_interactions
        self.max_num_nodes = max_num_nodes
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
        # Per-cid JP cost. The solver refreshes contacts each step from
        # ContactColumnContainer.contact_count; standalone partitioner tests
        # leave this zero-filled for jitter-only colouring.
        self._cost_values = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # Adjacency storage.
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(
            max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device
        )

        # Per-element packed tag: bit 62 unpartitioned marker + (color+1) in
        # bits 32..61 + element id in bits 0..31. No ping-pong 2*N buffer is
        # needed because we never sort.
        self._partition_data_concat = wp.zeros(max_num_interactions, dtype=wp.int64, device=device)

        # Parallel int32 mirror of the colour bits of
        # ``partition_data_concat`` (0 = uncoloured, 1+ = colour+1).
        # Half the per-read width on the greedy kernel's hot inner
        # adjacency walk -- on dense kapla-style graphs each
        # uncoloured vertex reads ~28 neighbours' colour status per
        # round, so the bandwidth saving compounds.
        self._color_tags = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

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

        # Compact remaining-elements ids for the next JP round,
        # updated in place by
        # ``incremental_tile_compact_remaining_and_advance_kernel``.
        # Safe in-place (survivors pack leftward, writes <= reads).
        # A single buffer (no host-side ping-pong) is required for
        # graph capture, which freezes kernel args at record time.
        self._remaining_ids = wp.zeros(padded_len, dtype=wp.int32, device=device)

        # Scalar device-side loop state (1-element arrays).
        self._current_color = wp.zeros(1, dtype=wp.int32, device=device)
        self._num_remaining = wp.zeros(1, dtype=wp.int32, device=device)
        self._partition_count = wp.zeros(1, dtype=wp.int32, device=device)

        # ----- Mode B: CSR coloring layout -----------------------------
        #
        # Populated by :meth:`build_csr`. Consumed by the PGS constraint
        # kernels, which read their work list directly out of these
        # arrays instead of recomputing the coloring once per sweep.
        #
        # Sized with the same ``max_num_interactions`` capacity as the
        # per-colour streaming buffers so every element in the graph
        # has a slot regardless of the final colour count.
        self._element_ids_by_color = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # CSR offsets, one slot per colour plus a sentinel end. Size
        # ``MAX_COLORS + 1`` so ``color_starts[num_colors]`` is always
        # addressable and holds the total element count -- i.e. callers
        # can always read ``end = color_starts[c + 1]`` without a
        # bounds branch.
        self._color_starts = wp.zeros(MAX_COLORS + 1, dtype=wp.int32, device=device)

        # Device-side colour count produced by :meth:`build_csr`.
        # ``capture_while(condition=self._color_cursor, ...)`` uses this
        # as the upper bound: a sibling scalar ``color_cursor`` is
        # initialised to ``num_colors`` and decremented to 0 by the
        # sweep kernels, mirroring how ``num_remaining`` drives the
        # per-colour streaming mode.
        self._num_colors = wp.zeros(1, dtype=wp.int32, device=device)

        # Overflow detector for the :data:`MAX_COLORS` cap. The compact
        # kernel sets ``[0] = 1`` and forces ``num_remaining[0] = 0`` if
        # the colour count would exceed the allocated ``color_starts``
        # range, preventing an out-of-bounds write. :meth:`build_csr`
        # reads the flag on the host after ``capture_while`` exits and
        # raises a descriptive error -- turning what used to be silent
        # memory corruption (see ``Bug.md`` #2) into a loud failure.
        self._overflow_flag = wp.zeros(1, dtype=wp.int32, device=device)

        # ----- Greedy build (build_csr_greedy) scratch ------------------
        #
        # Greedy mode does not commit colours in round order, so it can't
        # use the running-prefix CSR scatter that
        # ``incremental_tile_compact_csr_and_advance_kernel`` does. Instead
        # we histogram colours, exclusive-scan into ``color_starts``, and
        # atomic-scatter elements. ``_greedy_color_count`` is the
        # histogram bucket and ``_greedy_color_offsets`` is the live
        # write cursor used by the scatter.
        #
        # Sized at :data:`GREEDY_MAX_COLORS` because the greedy kernel's
        # forbidden-mask bitmask is bounded at that width; the build
        # raises ``RuntimeError`` (host-side) if the algorithm wanted
        # more colours than the bitmask can hold.
        self._greedy_color_count: wp.array[wp.int32] = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        self._greedy_color_offsets: wp.array[wp.int32] = wp.zeros(int(GREEDY_MAX_COLORS), dtype=wp.int32, device=device)
        # Persistent grid size for the greedy MIS+colour pass. Sized
        # like the PGS sweep kernels: 4 blocks/SM with a small
        # min-block floor and a per-capacity ceiling. Each launch
        # uses a grid-stride loop over ``[0, num_elements)`` so the
        # grid can stay much smaller than the constraint capacity --
        # over-launching here is pure warp-scheduling overhead.
        self._greedy_grid_size: int = _greedy_coloring_grid_size(max_num_interactions, device)
        # Predicate for the in-graph greedy-overflow fallback. Copied
        # from ``_overflow_flag`` after the greedy build so the
        # ``wp.capture_while`` body that runs the round-based JP
        # rebuild can clear it independently of the JP path's own
        # overflow accounting.
        self._fallback_flag: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=device)

        # Sweep cursor used by :meth:`begin_sweep` / the PGS kernels.
        # Holds the number of colours still to process in the current
        # sweep; decremented by the constraint kernel after it finishes
        # a colour. ``capture_while`` on this array terminates the
        # sweep's colour loop when the counter reaches zero.
        self._color_cursor = wp.zeros(1, dtype=wp.int32, device=device)

        # Per-sweep colour rotation offset. Bumped by :meth:`begin_sweep`
        # modulo ``_SWEEP_OFFSET_WRAP``; the constraint kernel decodes
        # the active colour as ``(n_colors - cursor + offset) %
        # n_colors`` so successive sweeps rotate which colour fires
        # first (symmetric Gauss-Seidel; evens out PGS's earlier /
        # later-colour bias on chains).
        self._sweep_offset = wp.zeros(1, dtype=wp.int32, device=device)

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

    def set_costs_from_contacts(
        self,
        num_joints: int,
        num_contact_columns: wp.array[wp.int32],
        contact_cols: ContactColumnContainer,
    ) -> None:
        """Refresh per-cid JP costs from contact column contact counts.

        Joints and inactive tail cids get cost 0. Active contact cids get
        the number of contacts in their shape-pair column.
        """
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
                self._color_tags,
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

        # 6. Initialize the compact remaining-ids index buffer with the
        #    identity permutation so the first JP round visits every
        #    active element in [0, num_elements[0]).
        wp.launch(
            incremental_init_remaining_ids_kernel,
            dim=self.max_num_interactions,
            inputs=[self._remaining_ids, num_elements],
        )

        # 7. Initialize device-side loop state (current_color=0,
        #    num_remaining=num_elements[0], partition_count=0).
        wp.launch(
            incremental_init_kernel,
            dim=1,
            inputs=[self._current_color, self._num_remaining, self._partition_count, num_elements],
        )

    def reset_loop_state_only(self) -> None:
        """Fast reset that keeps the previously-built adjacency intact.

        Safe whenever the constraint set hasn't changed since the
        last :meth:`reset` -- the common PGS inter-iteration case.
        Skips the expensive adjacency rebuild (prepare + count + scan
        + store) and only resets:

        * ``partition_data_concat[tid] = _UNPARTITIONED | tid``
        * ``interaction_id_to_partition[tid] = -1``
        * ``current_color``, ``num_remaining``, ``partition_count``
          scalars.

        :meth:`reset` must have been called at least once.
        """
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before reset_loop_state_only()"
        )

        # Single fused launch that resets both per-element arrays the
        # JP coloring pass consumes. Cheaper than two separate launches
        # (one for partition_data_concat, one for
        # interaction_id_to_partition) and touches exactly the active
        # prefix, so the padded tails stay at their construction zeros.
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

        # Reinitialise the compact remaining-ids buffer with the
        # identity permutation so the first JP round after the reset
        # visits every active element. The previous loop's contents
        # are stale because ``partition_data_concat`` was just wiped;
        # regenerating is strictly cheaper than trying to preserve them.
        wp.launch(
            incremental_init_remaining_ids_kernel,
            dim=self.max_num_interactions,
            inputs=[self._remaining_ids, self._num_elements],
        )

        # Re-initialise the device-side scalar loop state. This is the
        # same kernel ``reset`` uses in its final step.
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

        # 1. JP coloring pass over the compact remaining-ids list.
        # Iterating ``remaining_ids_front[0..num_remaining[0])`` instead
        # of the full ``partition_data_concat`` packs active lanes
        # contiguously, killing the thread divergence that dominated
        # the classic kernel in later rounds.
        # Launch dim must stay static (``num_remaining[0]`` is on-
        # device under ``capture_while``); lanes beyond that early-
        # exit cheaply.
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

        # 2. Fused in-place compact of remaining_ids + commit current
        # partition + advance device loop state -- four kernels
        # collapsed into one single-block grid-stride launch. Safe in
        # place: every write lands at an index <= the index it read
        # (see kernel docstring).
        # Requires tile-scan; the non-tile fallback was removed with
        # the remaining-ids buffer (it was graph-capture-unsafe).
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

    # ------------------------------------------------------------------
    # Mode B: build coloring once, replay across many sweeps
    # ------------------------------------------------------------------

    def build_csr(self) -> None:
        """Run Jones-Plassmann to completion and materialise the CSR coloring.

        Call after :meth:`reset` (which builds the adjacency).
        Writes :attr:`element_ids_by_color` (element ids grouped by
        colour), :attr:`color_starts` (exclusive prefix;
        ``color_starts[num_colors[0]]`` is the inclusive end), and
        :attr:`num_colors`. The loop is a ``wp.capture_while`` so the
        build is graph-capture safe; the colour count stays
        device-resident.
        """
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before build_csr()"
        )
        self._build_csr_inner()

        # Surface MAX_COLORS overflow as an exception when not
        # capturing (D2H reads are illegal during capture).
        device = self._overflow_flag.device
        if device.is_cuda and device.is_capturing:
            return
        if int(self._overflow_flag.numpy()[0]) != 0:
            raise RuntimeError(
                "PhoenX graph coloring exceeded MAX_COLORS "
                f"(={MAX_COLORS}). The constraint graph requires more "
                "partitions than the fixed-size color_starts buffer can "
                "hold. This usually indicates a pathological contact "
                "graph (e.g. a super-hub body touching many contacts) "
                "or an upstream ingest bug that cross-links worlds."
            )

    def _build_csr_inner(self) -> None:
        """Captured-graph-safe body of :meth:`build_csr`.

        Does init + capture_while + post-pass. No host-side reads, so
        callers can stuff this inside another captured region (e.g.
        the JP fallback path of
        :meth:`build_csr_greedy_with_jp_fallback`).
        """
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
        """Body of the ``build_csr`` capture_while.

        Runs up to :data:`NUM_INNER_WHILE_ITERATIONS` JP colour rounds
        back-to-back so the outer ``wp.capture_while`` only pays its
        per-iteration graph-traversal cost once per ``N`` rounds. Both
        inner kernels honour an early-exit contract when
        ``num_remaining[0] == 0`` (the MIS pass is per-thread skipped
        by its existing ``slot >= num_remaining[0]`` guard; the compact
        kernel returns before any write when ``n == 0``), so the tail
        rounds past convergence are cheap no-ops.
        """
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
                    int(MAX_COLORS),
                    self._overflow_flag,
                ],
                block_dim=int(TILE_SCAN_BLOCK_DIM),
            )

    # ------------------------------------------------------------------
    # Greedy build (Mode B alternative): JP-MIS + smallest-free-colour
    # ------------------------------------------------------------------

    def build_csr_greedy(self) -> None:
        """Greedy build (JP-MIS + smallest-free-colour). Bounded at
        :data:`GREEDY_MAX_COLORS` (64) by the int64 forbidden mask;
        raises ``RuntimeError`` if the graph wants more.

        Most callers should use
        :meth:`build_csr_greedy_with_jp_fallback` instead, which
        gracefully degrades to round-based JP without a host poll.
        """
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before build_csr_greedy()"
        )
        self._build_csr_greedy_inner()
        device = self._overflow_flag.device
        if device.is_cuda and device.is_capturing:
            return
        if int(self._overflow_flag.numpy()[0]) != 0:
            raise RuntimeError(
                "PhoenX greedy graph coloring exceeded GREEDY_MAX_COLORS "
                f"(={int(GREEDY_MAX_COLORS)}). The constraint graph "
                "requires more partitions than a single int64 forbidden "
                "mask can track. Use build_csr_greedy_with_jp_fallback "
                "or build_csr instead."
            )

    def build_csr_greedy_with_jp_fallback(self) -> None:
        """Greedy build with an in-graph round-based-JP fallback.

        Runs the greedy build first; if its 64-colour bitmask
        overflows on this graph, runs the round-based ``build_csr``
        inside a ``wp.capture_while`` keyed on the overflow flag so
        the corrupt CSR is overwritten with a valid one inside the
        same captured frame. No host poll, no probe step, no
        per-step branching from Python.
        """
        self._build_csr_greedy_inner()
        # Stage the fallback predicate. ``_fallback_flag`` is a
        # separate buffer from ``_overflow_flag`` so the JP body can
        # clear it without racing the JP path's own MAX_COLORS
        # overflow accounting (which writes back to
        # ``_overflow_flag``).
        wp.copy(self._fallback_flag, self._overflow_flag)
        wp.capture_while(self._fallback_flag, self._capture_jp_fallback_step)

    def _build_csr_greedy_inner(self) -> None:
        """Captured-graph-safe body of :meth:`build_csr_greedy`.

        Init + capture_while + post-pass. Splits cleanly so the
        in-graph fallback orchestrator above can call this without
        the host overflow-raise that the public method appends.
        """
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
        """Body of the in-graph greedy->JP fallback capture_while.

        Runs the round-based JP rebuild and clears the fallback
        predicate so the surrounding ``wp.capture_while`` exits after
        exactly one iteration regardless of whether JP itself
        overflows ``MAX_COLORS`` (that's a separate, much more
        permissive cap).
        """
        self._build_csr_inner()
        wp.launch(greedy_clear_int_kernel, dim=1, inputs=[self._fallback_flag])

    def _capture_build_csr_greedy_step(self) -> None:
        """Body of the ``build_csr_greedy`` capture_while.

        Each iteration zeroes the per-round ``any_uncolored`` flag,
        then runs the greedy MIS+colour kernel. The kernel iterates
        ``[0, num_elements)`` via grid-stride; uncoloured threads set
        the flag to 1, coloured threads early-exit. No compaction
        kernel: the round-based JP path's compact + remaining_ids
        bookkeeping is replaced by the on-tag-word coloured check at
        the top of each strided iteration.

        Inner body is unrolled :data:`NUM_INNER_WHILE_ITERATIONS` times
        so the outer ``wp.capture_while`` overhead amortises across N
        rounds. Trailing iterations after convergence cost only the
        flag-reset + an empty greedy launch (which short-circuits in
        the early-exit branch on every strided iteration).
        """
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
                ],
                block_dim=_GREEDY_BLOCK_DIM,
            )

    def begin_sweep(self, rotate: bool = False) -> None:
        """Reset the sweep-time colour cursor before a PGS sweep.

        Copies ``num_colors`` into ``color_cursor`` so downstream
        ``wp.capture_while(color_cursor, ...)`` loops see a fresh
        per-sweep countdown. The sweep body decrements
        ``color_cursor`` as each colour completes; when it reaches 0
        the capture_while exits.

        ``rotate=True`` additionally bumps ``sweep_offset`` so
        successive sweeps rotate which colour fires first (symmetric
        Gauss-Seidel; opt-in via
        ``SolverPhoenX(rotate_color_order=True)``).

        Must be called before every PGS sweep. Cheap -- a single
        1-thread kernel writing one or two scalars -- and
        graph-capture friendly.
        """
        if rotate:
            wp.launch(
                incremental_begin_sweep_rotate_kernel,
                dim=1,
                inputs=[self._num_colors, self._color_cursor, self._sweep_offset],
            )
        else:
            wp.launch(
                incremental_begin_sweep_kernel,
                dim=1,
                inputs=[self._num_colors, self._color_cursor],
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

    # ---- Mode B (CSR) results ----------------------------------------

    @property
    def element_ids_by_color(self) -> wp.array:
        """CSR element-id buffer written by :meth:`build_csr`.

        Colour ``c`` owns
        ``element_ids_by_color[color_starts[c]:color_starts[c+1]]``.
        Device ``int32`` array of length ``max_num_interactions``.
        """
        return self._element_ids_by_color

    @property
    def color_starts(self) -> wp.array:
        """CSR exclusive-prefix offsets written by :meth:`build_csr`.

        ``color_starts[num_colors[0]]`` holds the total element count
        (= sum of all colour sizes). Device ``int32`` array of length
        :data:`MAX_COLORS` + 1.
        """
        return self._color_starts

    @property
    def num_colors(self) -> wp.array:
        """Number of colours produced by the last :meth:`build_csr` call.

        Device ``int32`` scalar array; intended to be consumed by
        sweep-time ``wp.capture_while`` loops via :meth:`begin_sweep`
        + :attr:`color_cursor`. Cheap to read on the host too (see
        :meth:`num_colors_used` on ``World``).
        """
        return self._num_colors

    @property
    def color_cursor(self) -> wp.array:
        """Device-side colour countdown for the current PGS sweep.

        Initialised to :attr:`num_colors` by :meth:`begin_sweep`;
        decremented by the PGS constraint kernels after each colour
        completes. Pass to ``wp.capture_while`` as the termination
        condition for the sweep's per-colour launch loop.
        """
        return self._color_cursor

    @property
    def sweep_offset(self) -> wp.array:
        """Device-side per-sweep colour rotation offset.

        Bumped by :meth:`begin_sweep` modulo ``_SWEEP_OFFSET_WRAP``;
        the PGS constraint kernels read it to decode the active
        colour as ``(n_colors - cursor + offset) % n_colors``,
        which rotates which colour fires first across successive
        sweeps. Symmetric Gauss-Seidel / chain anti-bias.
        """
        return self._sweep_offset

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """``interaction_id_to_partition[i]`` is the colour that element ``i``
        was assigned to, or ``-1`` if the element has not been colored yet
        (device int32 array)."""
        return self._interaction_id_to_partition
