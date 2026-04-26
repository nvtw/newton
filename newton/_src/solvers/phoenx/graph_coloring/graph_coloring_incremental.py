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

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    TILE_SCAN_BLOCK_DIM,
    ElementInteractionData,
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
        contact_cid_start: int = 0,
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

        # Contacts live at cid >= ``contact_cid_start``. The JP kernels
        # OR bit 30 into their priority on the fly (no extra storage)
        # so contacts always outrank joints -- contacts cluster into
        # earlier colours, joints into later colours, cutting intra-warp
        # divergence in the fast-tail constraint iterate kernel.
        self._contact_cid_start = int(contact_cid_start)

        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)

        # Two-section priority: joints live in [0, contact_cid_start),
        # contacts in [contact_cid_start, max_num_interactions).
        # ``contact_partitions_get_random_value`` adds
        # ``max_num_contacts`` to contact priorities, so every contact
        # beats every joint in Jones-Plassmann -- contacts cluster
        # into earlier colours, joints into later, cutting intra-warp
        # divergence in the fast-tail iterate dispatch.
        # Setting the marker to ``max_num_interactions`` disables the bias
        # (single-section behaviour, used when there are no joints).
        marker = int(contact_cid_start) if 0 < int(contact_cid_start) < max_num_interactions else max_num_interactions
        self._section_marker = wp.array([marker], dtype=wp.int32, device=device)

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

        # Sweep cursor used by :meth:`begin_sweep` / the PGS kernels.
        # Holds the number of colours still to process in the current
        # sweep; decremented by the constraint kernel after it finishes
        # a colour. ``capture_while`` on this array terminates the
        # sweep's colour loop when the counter reaches zero.
        self._color_cursor = wp.zeros(1, dtype=wp.int32, device=device)

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
                self._adjacency_section_end_indices,
                self._vertex_to_adjacent_elements,
                self.max_num_contacts,
                self._elements,
                self._remaining_ids,
                self._num_remaining,
                self._section_marker,
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

        Called exactly once per ``World.step()``: the constraint
        graph is frozen inside a step (contacts ingested before the
        substep loop, joints static) so one CSR covers every substep
        and PGS iteration. Leaves the per-colour streaming state
        consistent with a full JP loop, so Mode A :meth:`launch` can
        be intermixed across steps (not within one build).
        """
        assert self._elements is not None and self._num_elements is not None, (
            "reset() must be called before build_csr()"
        )

        # Initialise CSR-specific state: zero the colour count, stamp
        # color_starts[0] = 0, reset current_color / num_remaining /
        # num_colors. Also wipe the per-colour partition_count so Mode
        # A readers never see stale data after a Mode B build.
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
        # Reset the MAX_COLORS overflow flag for this build. If the
        # capture_while loop hits the cap, the compact kernel sets this
        # to 1 and we raise below, turning a previously-silent buffer
        # overrun into a clear error.
        self._overflow_flag.zero_()

        # Reset per-element state that the JP coloring loop consumes.
        # This mirrors what reset() does after the adjacency rebuild;
        # we cannot reuse ``reset`` itself because build_csr can be
        # called repeatedly on the same unchanged adjacency (e.g. if
        # the caller invoked ``reset_loop_state_only`` earlier).
        wp.launch(
            incremental_reset_loop_state_kernel,
            dim=self.max_num_interactions,
            inputs=[
                self._partition_data_concat,
                self._interaction_id_to_partition,
                self._num_elements,
            ],
        )
        wp.launch(
            incremental_init_remaining_ids_kernel,
            dim=self.max_num_interactions,
            inputs=[self._remaining_ids, self._num_elements],
        )

        # Drive the coloring to completion. Each iteration: (1) run the
        # JP MIS pass over the current compact remaining-ids list,
        # (2) append the just-committed elements to
        # ``element_ids_by_color`` at ``color_starts[cc]`` and advance
        # loop state. Terminates when ``num_remaining`` hits zero, OR
        # when the compact kernel zeros ``num_remaining`` after hitting
        # the ``MAX_COLORS`` cap.
        wp.capture_while(
            self._num_remaining,
            self._capture_build_csr_step,
        )

        # Host-side overflow check via a ``.numpy()`` sync (~10 us
        # @256 worlds) -- the only way to surface a graph-captured
        # overflow as a Python exception. Skipped inside a user-owned
        # capture (D2H would fail with "legacy stream depending on a
        # capturing blocking stream"); the kernel early-exit still
        # protects the buffer, user sees the raise on the next
        # uncaptured step. Without this, overflow silently returned
        # corrupted state -- see Bug #2 in ``Bug.md``.
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
                "or an upstream ingest bug that cross-links worlds. "
                "Inspect `bodies[0]` frequency in the element table to "
                "diagnose; raising MAX_COLORS is a stop-gap, not a "
                "fix."
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
                    self._adjacency_section_end_indices,
                    self._vertex_to_adjacent_elements,
                    self.max_num_contacts,
                    self._elements,
                    self._remaining_ids,
                    self._num_remaining,
                    self._section_marker,
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

    def begin_sweep(self) -> None:
        """Reset the sweep-time colour cursor before a PGS sweep.

        Copies ``num_colors`` into ``color_cursor`` so downstream
        ``wp.capture_while(color_cursor, ...)`` loops see a fresh
        per-sweep countdown. The sweep body decrements
        ``color_cursor`` as each colour completes; when it reaches 0
        the capture_while exits.

        Must be called before every PGS sweep. Cheap -- a single
        1-thread kernel writing one scalar -- and graph-capture
        friendly.
        """
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
    def interaction_id_to_partition(self) -> wp.array:
        """``interaction_id_to_partition[i]`` is the colour that element ``i``
        was assigned to, or ``-1`` if the element has not been colored yet
        (device int32 array)."""
        return self._interaction_id_to_partition
