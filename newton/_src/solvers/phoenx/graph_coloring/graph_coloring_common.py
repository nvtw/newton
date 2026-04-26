"""Shared primitives for the batch and incremental graph-coloring partitioners.

This module holds the data structures, bit-packing constants, adjacency-building
kernels, and Jones-Plassmann MIS kernel that are used by both
:mod:`graph_coloring` (batch, budgeted, sorted output) and
:mod:`graph_coloring_incremental` (one partition per launch, no sort).

Separating these primitives keeps the two entry-point modules thin and makes
it explicit which kernels are safe to reuse.
"""

import warp as wp

# Bit layout of ``partition_data_concat`` entries (``wp.int64``):
#   bits  0..31  : element id (tid)
#   bits 32..61  : color + 1  (partitioned tag, 1..~1e9)
#   bit  62      : unpartitioned marker (set by the adjacency-store kernel;
#                  cleared as soon as the element is assigned a color)
#
# int64 so the encoding supports an essentially unlimited number of partitions.
# The previous int32 encoding packed color+1 into bits 26..30 and used bit 30
# as the unpartitioned marker; that made "color+1 == 16" alias the marker bit
# and silently double-assigned elements once more than 15 partitions were used.
_COLOR_SHIFT = wp.constant(wp.int64(32))
_ID_MASK = wp.constant(wp.int64((1 << 32) - 1))
_UNPARTITIONED = wp.constant(wp.int64(1 << 62))
# Mask for bits 0..61 (everything except the unpartitioned marker). Used in
# place of ``~_UNPARTITIONED`` because Warp's codegen does not reliably emit a
# 64-bit bitwise NOT for int64 constants.
_TAG_MASK = wp.constant(wp.int64((1 << 62) - 1))

MAX_BODIES = wp.constant(8)

vec8i = wp.types.vector(length=8, dtype=wp.int32)


@wp.struct
class ElementInteractionData:
    # Body slots. Inactive slots hold -1.
    # Index 0..1 are the primary pair; indices 2..7 are optional.
    bodies: vec8i


@wp.func
def element_interaction_data_empty() -> ElementInteractionData:
    d = ElementInteractionData()
    d.bodies = vec8i(-1, -1, -1, -1, -1, -1, -1, -1)
    return d


@wp.func
def element_interaction_data_make(
    body1: int, body2: int, body3: int, body4: int, body5: int, body6: int, body7: int, body8: int
) -> ElementInteractionData:
    d = ElementInteractionData()
    d.bodies = vec8i(body1, body2, body3, body4, body5, body6, body7, body8)
    return d


@wp.func
def element_interaction_data_get(d: ElementInteractionData, index: int) -> int:
    if index >= MAX_BODIES:
        return -1
    return d.bodies[index]


@wp.func
def element_interaction_data_add(d: ElementInteractionData, body_id: int) -> ElementInteractionData:
    # Returns updated struct; callers detect success by a count comparison.
    for i in range(MAX_BODIES):
        if d.bodies[i] < 0:
            d.bodies[i] = body_id
            return d
    return d


@wp.func
def element_interaction_data_count(d: ElementInteractionData) -> int:
    for i in range(MAX_BODIES):
        if d.bodies[i] < 0:
            return i
    return MAX_BODIES


@wp.func
def element_interaction_data_contains(d: ElementInteractionData, body_id: int) -> bool:
    for i in range(MAX_BODIES):
        b = d.bodies[i]
        if b < 0:
            return False
        if body_id == b:
            return True
    return False


# ---------------------------------------------------------------------------
# Adjacency-building kernels (shared by batch and incremental paths).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def partitioning_prepare_kernel(
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    max_num_partitions: int,
    max_num_nodes: int,
):
    tid = wp.tid()

    if tid <= max_num_partitions:
        partition_ends[tid] = 0

    if tid == 0:
        max_used_color[0] = -1

    if tid < max_num_nodes:
        adjacency_section_end_indices[tid] = 0


# TODO: Is a bit heavy on atomics
@wp.kernel(enable_backward=False)
def partitioning_adjacency_count_kernel(
    adjacency_section_end_indices: wp.array[int],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
):
    tid = wp.tid()

    total_num_work_packages = num_elements[0]
    if tid >= total_num_work_packages:
        return

    el = elements[tid]
    for j in range(MAX_BODIES):
        vertex = element_interaction_data_get(el, j)
        if vertex < 0:
            break
        wp.atomic_add(adjacency_section_end_indices, vertex, 1)


@wp.kernel(enable_backward=False)
def partitioning_adjacency_store_kernel(
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    partition_data_concat: wp.array[wp.int64],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
):
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    el = elements[tid]
    for j in range(MAX_BODIES):
        vertex = element_interaction_data_get(el, j)
        if vertex < 0:
            break
        index = wp.atomic_add(adjacency_section_end_indices, vertex, 1)
        vertex_to_adjacent_elements[index] = tid

    # Assign a high value so unpartitioned (overflow) elements sort to the end.
    # Bits 32..61 stay zero until the coloring kernel fills in (color + 1).
    partition_data_concat[tid] = _UNPARTITIONED | wp.int64(tid)


# ---------------------------------------------------------------------------
# Jones-Plassmann MIS kernel (shared).
# ---------------------------------------------------------------------------


@wp.func
def contact_partitions_is_removed(
    partition_data_concat: wp.array[wp.int64],
    i: int,
    color: int,
) -> bool:
    """Returns True iff element ``i`` is settled for the current Jones-Plassmann
    round at partition ``color``.

    An element is settled iff it was colored in an *earlier* round; elements
    colored this round are treated as conflicting so races between concurrent
    writers are resolved by priority.
    """
    rem = partition_data_concat[i] & _TAG_MASK
    rem = rem >> _COLOR_SHIFT
    # Unpartitioned OR just-committed-this-round -> still "active" (conflicting).
    # Only elements colored in a previous round are removed.
    return rem != wp.int64(0) and rem != wp.int64(color + 1)


@wp.func
def contact_partitions_get_random_value(
    random_values: wp.array[int], i: int, section_marker: int, max_num_contacts: int
) -> int:
    r = random_values[i]
    if i >= section_marker:
        r += max_num_contacts
    return r


@wp.kernel(enable_backward=False)
def partitioning_coloring_kernel(
    partition_data_concat: wp.array[wp.int64],
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    random_values: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    max_num_contacts: int,
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    section_marker_single_el_arr: wp.array[int],
    color_arr: wp.array[int],
):
    # Jones-Plassmann independent-set pass: a vertex joins partition
    # ``color_arr[0]`` iff its priority is strictly greater than all
    # *uncolored* neighbours' priorities. ``color_arr`` lives on the device
    # so the launcher can iterate without host readbacks (graph-capture safe).
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    color_copy = color_arr[0]
    section_marker = section_marker_single_el_arr[0]

    if contact_partitions_is_removed(partition_data_concat, tid, color_copy):
        return

    if max_used_color[0] != color_copy:
        max_used_color[0] = color_copy

    is_local_max = bool(True)

    self_prio = contact_partitions_get_random_value(random_values, tid, section_marker, max_num_contacts)
    el = elements[tid]

    for j in range(MAX_BODIES):
        if not is_local_max:
            break
        v = element_interaction_data_get(el, j)
        if v < 0:
            break

        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        else:
            start = 0
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            if contact_partitions_is_removed(partition_data_concat, neighbor, color_copy):
                continue
            if (
                contact_partitions_get_random_value(random_values, neighbor, section_marker, max_num_contacts)
                > self_prio
            ):
                is_local_max = False
                break

    if is_local_max:
        partition_data_concat[tid] = (wp.int64(color_copy + 1) << _COLOR_SHIFT) | wp.int64(tid)
        wp.atomic_add(partition_ends, color_copy, 1)


@wp.kernel(enable_backward=False)
def partitioning_coloring_incremental_kernel(
    partition_data_concat: wp.array[wp.int64],
    random_values: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    max_num_contacts: int,
    elements: wp.array[ElementInteractionData],
    remaining_ids: wp.array[int],
    num_remaining: wp.array[int],
    section_marker_single_el_arr: wp.array[int],
    color_arr: wp.array[int],
):
    """Jones-Plassmann MIS pass iterating over a compact remaining-ids list.

    Functionally identical to :func:`partitioning_coloring_kernel` but reads
    the set of candidate elements from ``remaining_ids[0..num_remaining[0])``
    instead of filtering ``[0, num_elements[0])`` via an ``is_removed``
    check at the top of every thread.

    Why this matters:

    * **No thread divergence from self-filtering.** In the classic
      kernel every lane must load ``partition_data_concat[tid]`` to
      decide whether to bail. With a compact remaining list every
      launched lane has real work -- warps stay fully utilised and the
      adjacency walk (the hot inner loop) runs on all 32 lanes
      together instead of a sparse subset.
    * **Fewer spurious launches.** The launch grid is ``num_remaining``
      rounded up to a tile, not ``max_num_interactions``. After a few
      colour rounds this is already a sizeable cut.

    Neighbours referenced via the adjacency list are still checked with
    ``contact_partitions_is_removed`` because they can come from any
    earlier round (and thus may or may not be settled); that filter is
    cheap and unavoidable for correctness.
    """
    slot = wp.tid()

    if slot >= num_remaining[0]:
        return

    tid = remaining_ids[slot]
    color_copy = color_arr[0]
    section_marker = section_marker_single_el_arr[0]

    # Elements in the compact list are -- by construction -- not yet
    # assigned, so the outer ``is_removed`` self-check that the classic
    # kernel runs is redundant here and is skipped.

    is_local_max = bool(True)

    self_prio = contact_partitions_get_random_value(random_values, tid, section_marker, max_num_contacts)
    el = elements[tid]

    for j in range(MAX_BODIES):
        if not is_local_max:
            break
        v = element_interaction_data_get(el, j)
        if v < 0:
            break

        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        else:
            start = 0
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            # Neighbours come from the raw id space, so they may still
            # belong to an earlier round's partition. Filter them out.
            if contact_partitions_is_removed(partition_data_concat, neighbor, color_copy):
                continue
            if (
                contact_partitions_get_random_value(random_values, neighbor, section_marker, max_num_contacts)
                > self_prio
            ):
                is_local_max = False
                break

    if is_local_max:
        partition_data_concat[tid] = (wp.int64(color_copy + 1) << _COLOR_SHIFT) | wp.int64(tid)


# ---------------------------------------------------------------------------
# Incremental-path helper kernels.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def incremental_init_kernel(
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    partition_count: wp.array[int],
    num_elements: wp.array[int],
):
    # Single-thread launch: set up device-side loop state after adjacency is
    # built. Using a kernel rather than host writes keeps reset() graph-capture
    # friendly if the caller ever wants to capture reset + launch together.
    current_color[0] = 0
    num_remaining[0] = num_elements[0]
    partition_count[0] = 0


@wp.kernel(enable_backward=False)
def incremental_init_csr_kernel(
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    num_colors: wp.array[int],
    color_starts: wp.array[int],
    num_elements: wp.array[int],
):
    # Single-thread launch: initialise the CSR coloring state before
    # :meth:`IncrementalContactPartitioner.build_csr` begins. Clears
    # ``color_starts[0] = 0`` so the first colour lays its elements
    # down at slot 0 of ``element_ids_by_color``; zeroes
    # ``num_colors`` so partial-state reads (e.g. from the optional
    # diagnostic ``World.num_colors_used``) return a coherent value
    # while the JP loop is still running.
    current_color[0] = 0
    num_remaining[0] = num_elements[0]
    num_colors[0] = 0
    color_starts[0] = 0


@wp.kernel(enable_backward=False)
def incremental_begin_sweep_kernel(
    num_colors: wp.array[int],
    color_cursor: wp.array[int],
):
    # Single-thread launch: copy ``num_colors`` into ``color_cursor``
    # so the sweep-time capture_while has a fresh counter to decrement.
    # Done as a kernel (rather than a host-side ``copy_``) so the setup
    # can live inside a captured region when the caller wants to bake
    # the whole substep into a CUDA graph.
    color_cursor[0] = num_colors[0]


@wp.kernel(enable_backward=False)
def incremental_fill_minus_one_kernel(
    arr: wp.array[int],
    num_elements: wp.array[int],
):
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    arr[tid] = -1


@wp.kernel(enable_backward=False)
def incremental_init_remaining_ids_kernel(
    remaining_ids: wp.array[int],
    num_elements: wp.array[int],
):
    """Initialise the compact remaining-ids index buffer: ``remaining_ids[i] = i``.

    Written by :meth:`IncrementalContactPartitioner.reset` (and its
    adjacency-preserving sibling :meth:`reset_loop_state_only`) so the
    first JP round iterates over every active element. Subsequent
    rounds overwrite this buffer from
    :func:`incremental_tile_compact_remaining_and_advance_kernel`,
    shrinking the list as elements get partitioned.
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    remaining_ids[tid] = tid


@wp.kernel(enable_backward=False)
def incremental_reset_loop_state_kernel(
    partition_data_concat: wp.array[wp.int64],
    interaction_id_to_partition: wp.array[int],
    num_elements: wp.array[int],
):
    """Reset the per-element partitioner state that is consumed by the
    JP coloring pass, without touching the adjacency structure.

    Between PGS iterations the constraint set is unchanged -- only the
    solver's accumulated impulses evolve -- so the expensive adjacency
    rebuild (``partitioning_prepare_kernel`` +
    ``partitioning_adjacency_count_kernel`` + exclusive scan +
    ``partitioning_adjacency_store_kernel``) can be skipped entirely.
    What *does* have to be reset is:

    * ``partition_data_concat[tid] = _UNPARTITIONED | tid``, the
      packed per-element (color, tid) tag the coloring kernel reads
      and writes.
    * ``interaction_id_to_partition[tid] = -1``, so callers that read
      the per-element assigned color get a well-defined sentinel for
      elements not yet touched by the current pass.

    Fused into a single launch so the "skip-adjacency" reset path
    costs one kernel launch instead of two (``partitioning_adjacency_store``'s
    scatter half + ``incremental_fill_minus_one``).
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    partition_data_concat[tid] = _UNPARTITIONED | wp.int64(tid)
    interaction_id_to_partition[tid] = -1


@wp.kernel(enable_backward=False)
def incremental_zero_int_kernel(
    arr: wp.array[int],
    num_elements: wp.array[int],
):
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    arr[tid] = 0


@wp.kernel(enable_backward=False)
def incremental_flag_kernel(
    partition_data_concat: wp.array[wp.int64],
    current_color: wp.array[int],
    flags: wp.array[int],
    num_elements: wp.array[int],
):
    # flags[tid] = 1 iff element tid was just committed to partition
    # current_color[0] by the most recent JP coloring pass. Elements from
    # earlier rounds have a higher "color+1" stored, unpartitioned elements
    # still carry the _UNPARTITIONED marker; masking with _TAG_MASK strips
    # the marker and the comparison identifies only this round's new entries.
    tid = wp.tid()
    n = num_elements[0]
    if tid >= n:
        flags[tid] = 0
        return

    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = tagged >> _COLOR_SHIFT
    if color_plus_one == wp.int64(current_color[0] + 1):
        flags[tid] = 1
    else:
        flags[tid] = 0


@wp.kernel(enable_backward=False)
def incremental_compact_kernel(
    flags: wp.array[int],
    offsets: wp.array[int],
    current_color: wp.array[int],
    partition_element_ids: wp.array[int],
    interaction_id_to_partition: wp.array[int],
    partition_count: wp.array[int],
    num_elements: wp.array[int],
):
    # Given flags (0/1 just-committed) and their exclusive prefix sum, write
    # the compacted element list for the current partition and record the
    # assigned color per element. One thread stores the total count.
    tid = wp.tid()
    n = num_elements[0]

    # Handle degenerate n == 0 case (no work) and publish zero count.
    if n == 0:
        if tid == 0:
            partition_count[0] = 0
        return

    if tid >= n:
        return

    if flags[tid] == 1:
        partition_element_ids[offsets[tid]] = tid
        interaction_id_to_partition[tid] = current_color[0]

    # Last active thread publishes the partition size.
    if tid == n - 1:
        partition_count[0] = offsets[tid] + flags[tid]


@wp.kernel(enable_backward=False)
def incremental_advance_kernel(
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    partition_count: wp.array[int],
):
    num_remaining[0] = num_remaining[0] - partition_count[0]
    current_color[0] = current_color[0] + 1


@wp.kernel(enable_backward=False)
def set_int_array_kernel(arr: wp.array[int], value: int):
    # Tiny helper used by batch launcher to push the host-side color counter
    # into the device array consumed by ``partitioning_coloring_kernel``.
    arr[0] = value


# Block size for the single-block tile-scan kernel. 1024 = largest CUDA block
# that Warp supports reliably, keeps the running-prefix sequential tail short.
TILE_SCAN_BLOCK_DIM = wp.constant(1024)


@wp.kernel(enable_backward=False)
def tile_scan_exclusive_block_kernel(
    input: wp.array[int],
    output: wp.array[int],
):
    """Single-block exclusive prefix scan over the entire ``input`` array.

    Launched with ``dim=[1]`` and ``block_dim=TILE_SCAN_BLOCK_DIM``. The
    block walks ``input`` in a grid-stride loop, scanning one tile of
    ``TILE_SCAN_BLOCK_DIM`` elements per iteration and threading a
    running-prefix accumulator across tiles.

    Callers are responsible for padding ``input`` and ``output`` to a
    multiple of ``TILE_SCAN_BLOCK_DIM`` and for zero-initialising any
    padded tail (so scans of sparsely-filled flag arrays are still
    correct on the leading ``num_elements`` prefix).

    This is an allocation-free alternative to :func:`wp.utils.array_scan`
    that is safe to invoke inside a ``wp.capture_while`` body.
    """
    # ``wp.tid()`` inside a launch_tiled kernel returns the block index; to
    # get the lane/thread index within the block we use ``wp.tid_lane()`` via
    # the standard (block, lane) unpacking below.
    _block, lane = wp.tid()

    n = input.shape[0]

    # Per-thread running (inclusive) prefix carried between tiles. All threads
    # in the block hold the same value thanks to the block-wide ``tile_sum``
    # below (``tile_sum`` returns a 1-element tile whose scalar is visible
    # identically to every lane).
    running = int(0)

    offset = int(0)
    while offset < n:
        a = wp.tile_load(input, shape=TILE_SCAN_BLOCK_DIM, offset=offset)
        # Tile-wide exclusive scan of the current 1024-element window. Each
        # lane ends up owning one element: ``s[lane]`` is the exclusive-scan
        # value for position ``offset + lane``.
        s = wp.tile_scan_exclusive(a)
        # Shift by the running prefix accumulated over previous tiles and
        # scatter back to the output array one element per lane. Storing
        # lane-wise (instead of wp.tile_store) lets us mix the SIMT scalar
        # ``running`` with the tile-resident scan result without needing a
        # tile broadcast.
        output[offset + lane] = s[lane] + running
        # Advance the running prefix by the *inclusive* sum of this tile.
        # ``tile_sum`` is a block-wide reduction so all lanes observe the
        # same ``block_sum[0]`` and therefore the same ``running`` value.
        block_sum = wp.tile_sum(a)
        running = running + block_sum[0]
        offset = offset + TILE_SCAN_BLOCK_DIM


@wp.kernel(enable_backward=False)
def incremental_tile_compact_remaining_and_advance_kernel(
    partition_data_concat: wp.array[wp.int64],
    remaining_ids: wp.array[int],
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    partition_count: wp.array[int],
    partition_element_ids: wp.array[int],
    interaction_id_to_partition: wp.array[int],
):
    """Fused compact + remaining-list update + advance for one JP round.

    Replaces four back-to-back kernels
    (``incremental_flag_kernel`` -> tile scan -> ``incremental_compact_kernel``
    -> ``incremental_advance_kernel``) with a single single-block
    grid-stride launch and additionally maintains a compact **index
    buffer** of still-active elements so the next round's coloring pass
    only visits lanes that actually have work.

    Why the compact index buffer matters
    ------------------------------------

    Before this optimisation, every colour round launched
    ``partitioning_coloring_kernel`` with ``dim=max_num_interactions``.
    Each lane loaded ``partition_data_concat[tid]`` and early-exited if
    the element was already settled by an earlier round. At ~18 colours
    per sweep the average warp therefore had only ~50% of lanes doing
    real work, the rest stalled on the early-exit branch.

    By compacting ``remaining_ids[0..num_remaining)`` in place here --
    dropping the elements just committed to partition ``cc`` -- the
    next round can drive ``partitioning_coloring_incremental_kernel``
    over a dense work list. Every launched warp is fully utilised and
    the hot adjacency walk runs on all 32 lanes together.

    In-place update safety
    ----------------------

    The kernel reads from and writes to the *same* ``remaining_ids``
    buffer. The compaction is correct because survivors are packed
    leftward, i.e. every write goes to an index ``<=`` the index it
    read from:

    * Within a tile (1024 lanes), the exclusive-scan primitive acts as
      an implicit block sync: every lane's read of
      ``remaining_ids[offset+lane]`` completes before any lane's
      survivor store executes.
    * Across tiles: tile N writes survivors into indices ``[running,
      running + tile_survivors)`` where ``running <= N*1024``
      (survivors never exceed slots processed). Tile N+1 reads from
      ``[(N+1)*1024, (N+2)*1024)``. Since
      ``running + tile_survivors <= (N+1)*1024``, the writes stay
      strictly below the next tile's read window.

    Per-tile algorithm
    ------------------

    Each tile computes two per-lane 0/1 flags and two tile scans:

    * ``committed_flag = 1`` iff the element was just coloured with
      ``cc = current_color[0]``; its exclusive scan drives the
      compacted write into ``partition_element_ids`` and indexes
      ``interaction_id_to_partition``.
    * ``survivor_flag = in_range ? 1 - committed_flag : 0``; its
      exclusive scan drives the in-place compaction of
      ``remaining_ids``.

    Two running prefixes (``committed_running``, ``survivor_running``)
    are threaded across tiles via ``wp.tile_sum``.

    After the last tile, lane 0 publishes the totals and advances
    ``current_color``. Race-free: every lane reads ``current_color[0]``
    once into register ``cc`` at the top, and the only write happens
    from lane 0 after all reads have completed in every tile.
    """
    _block, lane = wp.tid()

    n = num_remaining[0]
    cc = current_color[0]

    # Running exclusive-prefix counts maintained identically on every
    # lane via ``wp.tile_sum`` at the end of each tile iteration.
    committed_running = int(0)
    survivor_running = int(0)

    offset = int(0)
    while offset < n:
        slot = offset + lane

        # Per-lane: fetch the raw element id from the remaining list,
        # check whether it was just committed to partition ``cc``. Out-
        # of-range lanes contribute zero flags so tail tiles stay
        # correct without a separate zero pass.
        eid = int(0)
        committed_flag = int(0)
        survivor_flag = int(0)
        if slot < n:
            eid = remaining_ids[slot]
            tagged = partition_data_concat[eid] & _TAG_MASK
            color_plus_one = tagged >> _COLOR_SHIFT
            if color_plus_one == wp.int64(cc + 1):
                committed_flag = 1
            else:
                survivor_flag = 1

        # Two exclusive scans over this tile. Each scan is a block-wide
        # collective with implicit sync, so the two calls do not step
        # on each other. The scans also synchronise all lanes' reads of
        # ``remaining_ids[slot]`` above before any lane performs the
        # in-place survivor store below, which is what makes the
        # in-place compaction safe within a tile.
        committed_tile = wp.tile(committed_flag)
        committed_scan = wp.tile_scan_exclusive(committed_tile)
        committed_local = wp.untile(committed_scan)

        survivor_tile = wp.tile(survivor_flag)
        survivor_scan = wp.tile_scan_exclusive(survivor_tile)
        survivor_local = wp.untile(survivor_scan)

        # Compact committed elements into partition_element_ids and
        # record the colour. Only flagged lanes write.
        if committed_flag == 1:
            out_idx = committed_running + committed_local
            partition_element_ids[out_idx] = eid
            interaction_id_to_partition[eid] = cc

        # In-place compaction of survivors. The destination index is
        # always <= ``slot`` (the source index), so the store cannot
        # clobber data still needed by the current tile. The running
        # prefix is maintained identically on every lane.
        if survivor_flag == 1:
            out_idx = survivor_running + survivor_local
            remaining_ids[out_idx] = eid

        # Advance running prefixes via block-wide inclusive sums.
        committed_total = wp.tile_sum(committed_tile)
        survivor_total = wp.tile_sum(survivor_tile)
        committed_running = committed_running + committed_total[0]
        survivor_running = survivor_running + survivor_total[0]
        offset = offset + TILE_SCAN_BLOCK_DIM

    # Publish per-round totals and advance device-side loop state.
    # All other lanes have finished reading ``current_color`` and
    # ``num_remaining``; the writes below cannot race.
    if lane == 0:
        partition_count[0] = committed_running
        num_remaining[0] = survivor_running
        current_color[0] = cc + 1


@wp.kernel(enable_backward=False)
def incremental_tile_compact_csr_and_advance_kernel(
    partition_data_concat: wp.array[wp.int64],
    remaining_ids: wp.array[int],
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    num_colors: wp.array[int],
    element_ids_by_color: wp.array[int],
    color_starts: wp.array[int],
    interaction_id_to_partition: wp.array[int],
    max_colors: int,
    overflow_flag: wp.array[int],
):
    """CSR variant of :func:`incremental_tile_compact_remaining_and_advance_kernel`.

    Same per-tile algorithm (in-place compaction of ``remaining_ids``,
    block-wide exclusive scans, lane-0 scalar publication) but the
    committed-elements compacted output is appended to the CSR layout
    ``element_ids_by_color`` at offset ``color_starts[cc]`` rather than
    written into a one-shot ``partition_element_ids`` buffer. After
    the last tile lane 0 additionally writes
    ``color_starts[cc + 1] = color_starts[cc] + committed_running``
    so the next colour -- and the final ``num_colors`` sentinel slot --
    picks up the correct starting offset.

    ``num_colors`` mirrors ``current_color`` as each colour completes
    and is read by :class:`~newton._src.solvers.phoenx.World` to drive
    the sweep-time ``capture_while`` that replays the CSR colour by
    colour across all PGS iterations and substeps of a ``step()``.

    Race-free by the same argument as the non-CSR sibling: every lane
    reads ``current_color[0]`` once at the top into ``cc``, and the
    only writes to ``current_color``, ``color_starts[cc+1]``, and
    ``num_colors`` happen from lane 0 after all reads have completed
    across every tile.
    """
    _block, lane = wp.tid()

    n = num_remaining[0]
    # Early-exit contract: tolerate extra launches past convergence so
    # the host side can safely unroll the capture-while body
    # ``NUM_INNER_WHILE_ITERATIONS`` times. When ``num_remaining == 0``
    # no element is still uncoloured and every per-colour counter
    # (``current_color``, ``num_colors``, ``color_starts``) has already
    # been written to its final value by the round that drained the
    # list. Returning without any writes keeps those counters stable
    # so the surplus launches are true no-ops.
    if n == 0:
        return
    cc = current_color[0]
    # Overflow guard. ``color_starts`` is sized ``max_colors + 1`` so
    # the largest writable end-offset slot is ``color_starts[max_colors]``
    # -- i.e. valid ``cc`` values are ``0 .. max_colors - 1``. If the
    # coloring exhausts the budget, raise an overflow flag, force the
    # outer ``capture_while`` to terminate by zeroing ``num_remaining``,
    # and early-return before any buffer writes. The host side of
    # ``build_csr`` reads the flag after the capture_while exits and
    # raises a descriptive error so this cannot silently corrupt memory.
    if cc >= max_colors:
        if lane == 0:
            overflow_flag[0] = 1
            num_remaining[0] = 0
        return
    # Base offset into the CSR buffer for this colour. Every lane
    # captures it into a register up front so the later survivor scan's
    # implicit block sync double-duties as a read barrier for it.
    base = color_starts[cc]

    committed_running = int(0)
    survivor_running = int(0)

    offset = int(0)
    while offset < n:
        slot = offset + lane

        eid = int(0)
        committed_flag = int(0)
        survivor_flag = int(0)
        if slot < n:
            eid = remaining_ids[slot]
            tagged = partition_data_concat[eid] & _TAG_MASK
            color_plus_one = tagged >> _COLOR_SHIFT
            if color_plus_one == wp.int64(cc + 1):
                committed_flag = 1
            else:
                survivor_flag = 1

        committed_tile = wp.tile(committed_flag)
        committed_scan = wp.tile_scan_exclusive(committed_tile)
        committed_local = wp.untile(committed_scan)

        survivor_tile = wp.tile(survivor_flag)
        survivor_scan = wp.tile_scan_exclusive(survivor_tile)
        survivor_local = wp.untile(survivor_scan)

        # Append committed elements into the CSR slice for colour cc.
        # Writes target ``[base + committed_running, base + committed_running +
        # tile_committed)`` which lies strictly inside this colour's CSR
        # range; different colours write into disjoint ranges so there
        # is no cross-kernel hazard.
        if committed_flag == 1:
            out_idx = base + committed_running + committed_local
            element_ids_by_color[out_idx] = eid
            interaction_id_to_partition[eid] = cc

        if survivor_flag == 1:
            out_idx = survivor_running + survivor_local
            remaining_ids[out_idx] = eid

        committed_total = wp.tile_sum(committed_tile)
        survivor_total = wp.tile_sum(survivor_tile)
        committed_running = committed_running + committed_total[0]
        survivor_running = survivor_running + survivor_total[0]
        offset = offset + TILE_SCAN_BLOCK_DIM

    if lane == 0:
        # Write this colour's end offset = next colour's start offset.
        # ``color_starts[cc + 1]`` is guaranteed addressable because
        # ``color_starts`` is sized ``MAX_COLORS + 1``.
        color_starts[cc + 1] = base + committed_running
        num_remaining[0] = survivor_running
        current_color[0] = cc + 1
        # ``num_colors`` trails ``current_color`` by one step and
        # settles on the final colour count when the outer
        # capture_while exits (i.e. when num_remaining hits 0). Writing
        # it unconditionally every round keeps the bookkeeping simple;
        # the last round's write is the one callers observe.
        num_colors[0] = cc + 1
