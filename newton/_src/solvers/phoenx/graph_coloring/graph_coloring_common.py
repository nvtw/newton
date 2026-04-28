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
    color_tags: wp.array[wp.int32],
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
    # ``color_tags`` mirrors the colour bits of partition_data_concat
    # for the greedy kernel's hot-path 4-byte reads.
    color_tags[tid] = wp.int32(0)


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
    random_values: wp.array[wp.int32],
    cost_values: wp.array[wp.int32],
    i: int,
) -> wp.int64:
    cost = wp.int64(cost_values[i])
    jitter = wp.int64(random_values[i]) & _ID_MASK
    return (cost << _COLOR_SHIFT) | jitter


@wp.kernel(enable_backward=False)
def partitioning_coloring_kernel(
    partition_data_concat: wp.array[wp.int64],
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    random_values: wp.array[wp.int32],
    cost_values: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
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

    if contact_partitions_is_removed(partition_data_concat, tid, color_copy):
        return

    if max_used_color[0] != color_copy:
        max_used_color[0] = color_copy

    is_local_max = bool(True)

    self_prio = contact_partitions_get_random_value(random_values, cost_values, tid)
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
            if contact_partitions_get_random_value(random_values, cost_values, neighbor) > self_prio:
                is_local_max = False
                break

    if is_local_max:
        partition_data_concat[tid] = (wp.int64(color_copy + 1) << _COLOR_SHIFT) | wp.int64(tid)
        wp.atomic_add(partition_ends, color_copy, 1)


# Maximum colour count representable in a single int64 forbidden mask.
# The greedy variant of the JP coloring (see
# :func:`partitioning_coloring_incremental_greedy_kernel`) tracks the set of
# colours already taken by an element's coloured neighbours in this many
# bits. Real-world physics constraint graphs comfortably fit (Kapla tower's
# max body degree is 28; ragdolls and robots stay well under 32). If a
# graph would actually need more than this, the greedy kernel sets an
# overflow flag and the caller raises a descriptive error before the
# coloring corrupts (analogous to the MAX_COLORS guard in the CSR builder).
GREEDY_MAX_COLORS = wp.constant(64)
# Constant mask used to flip the forbidden mask in lieu of the unreliable
# 64-bit bitwise NOT on int64 constants (see _TAG_MASK comment for the
# original codegen issue).
_FREE_COLOR_FLIP = wp.constant(wp.int64(-1))  # all ones, two's complement


@wp.kernel(enable_backward=False)
def partitioning_coloring_incremental_greedy_kernel(
    partition_data_concat: wp.array[wp.int64],
    color_tags: wp.array[wp.int32],
    random_values: wp.array[wp.int32],
    cost_values: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    total_num_threads: wp.int32,
    num_remaining: wp.array[int],
    overflow_flag: wp.array[int],
):
    """JP-MIS combined with greedy color selection.

    Same parallelism contract as
    :func:`partitioning_coloring_incremental_kernel`: a vertex commits
    iff it has the highest priority among its still-uncoloured neighbours
    (so commits within a round are an independent set, identical to the
    classic JP step).

    Difference: instead of tagging every committed vertex with the
    *round number*, the kernel computes the smallest colour not used by
    its already-coloured neighbours and writes that colour. Result: a
    run of the loop converges to near-greedy colour counts (often 2-3x
    fewer colours than vanilla JP) at the same rounds-per-build cost.

    Persistent-grid grid-stride loop driven by ``total_num_threads``,
    matching the layout of :func:`_constraint_iterate_singleworld_kernel`
    in :mod:`solver_phoenx_kernels`. The launch dim is sized
    independently of ``num_elements`` so the kernel runs on a
    warp-aligned, SM-tuned grid regardless of the constraint capacity.
    Each thread strides through ``[wp.tid(), num_elements,
    total_num_threads)`` and processes the elements it lands on,
    eliminating the partial-warp tail caused by launching at
    ``num_elements`` directly when ``num_elements`` is not a multiple
    of the warp size.

    Drops the compacted ``remaining_ids`` list -- each strided index
    reads its packed tag once and skips already-coloured entries. The
    same divergence pattern the round-based JP kernel paid via
    ``slot >= num_remaining`` plus a stale-list dereference, minus
    the per-round compact kernel. On Kapla Tower the compact kernel
    was ~16% of the captured frame time before this change.

    Convergence is driven by the same ``num_remaining`` counter the
    round-based JP path uses: initialised to ``num_elements`` by the
    build's reset, atomically decremented by every commit in this
    kernel, read by the outer ``wp.capture_while`` as the loop
    predicate. Strictly monotonic (commits only decrease it; never
    increase) so the conditional CUDA graph stays well-defined and
    converges to 0 once every vertex is coloured.

    Forbidden colours are tracked in a single ``int64`` bitmask, which
    bounds the achievable colour count at :data:`GREEDY_MAX_COLORS`
    (= 64). Going past 64 colours raises ``overflow_flag[0] = 1``; the
    host-side build path reads it after the build loop exits and
    raises a descriptive error before any consumer sees corrupt
    output. This is the same overflow contract as the CSR build's
    ``MAX_COLORS`` cap.
    """
    n = num_elements[0]
    for tid in range(wp.tid(), n, total_num_threads):
        # 4-byte read of the per-vertex colour tag (0 = uncoloured,
        # 1+ = colour+1). Replaces the int64 read of
        # ``partition_data_concat`` -- on dense kapla-style graphs
        # the inner adjacency walk does ~28 of these reads per
        # uncoloured vertex per round, so halving the per-read width
        # is the dominant gmem-bandwidth saving.
        if color_tags[tid] != wp.int32(0):
            continue

        self_prio = contact_partitions_get_random_value(random_values, cost_values, tid)
        el = elements[tid]

        is_local_max = bool(True)
        forbidden_mask = wp.int64(0)

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
                # Inspect this neighbour's colour tag once and
                # dispatch on whether they're already coloured. Two
                # distinct cases:
                #
                #   tag == 0 -> neighbour is still uncoloured. Apply
                #     the standard JP MIS check; if the neighbour
                #     outranks us, bail (we won't commit this round).
                #   tag != 0 -> neighbour is already coloured. Mark
                #     their colour as forbidden in the bitmask so we
                #     don't reuse it.
                ntag = color_tags[neighbor]
                if ntag == wp.int32(0):
                    # Uncoloured -> MIS tiebreak.
                    if contact_partitions_get_random_value(random_values, cost_values, neighbor) > self_prio:
                        is_local_max = False
                        break
                else:
                    # Coloured -> forbid that colour. Cap at the bitmask
                    # capacity; if a graph really needs more colours we
                    # raise overflow below.
                    ncolor = ntag - wp.int32(1)
                    if ncolor < GREEDY_MAX_COLORS:
                        forbidden_mask = forbidden_mask | (wp.int64(1) << wp.int64(ncolor))

        if is_local_max:
            # Smallest free colour = position of the lowest 0-bit in
            # the forbidden mask. Free positions are encoded as 1s in
            # the complement; ``forbidden_mask ^ -1`` is the all-ones
            # flip without relying on int64 unary NOT (see the
            # _TAG_MASK comment block for why we avoid ``~``).
            free_mask = forbidden_mask ^ _FREE_COLOR_FLIP
            # Linear scan for the first set bit. 64 iterations max --
            # well under the kernel's per-vertex adjacency walk -- and
            # avoids depending on a Warp-side ``ffs`` builtin we don't
            # otherwise need.
            c = wp.int32(0)
            for _ in range(GREEDY_MAX_COLORS):
                if (free_mask & (wp.int64(1) << wp.int64(c))) != wp.int64(0):
                    break
                c = c + wp.int32(1)
            if c >= GREEDY_MAX_COLORS:
                # Mask saturated: graph wants > GREEDY_MAX_COLORS
                # distinct colours. We MUST still commit something
                # and decrement ``num_remaining`` -- the outer
                # ``wp.capture_while`` runs inside a captured CUDA
                # graph with no host poll site, so leaving the
                # counter non-zero would loop forever rather than
                # giving the host a chance to raise. Stamp the
                # vertex with the highest-bit colour as a poison
                # value (distinct from any legitimate commit, so the
                # post-pass histogram lights it up at index
                # ``GREEDY_MAX_COLORS - 1`` even when the
                # legitimate maximum is lower) and let the host
                # observe the flag after the build returns.
                overflow_flag[0] = 1
                fallback_c = wp.int32(GREEDY_MAX_COLORS - wp.int32(1))
                color_tags[tid] = fallback_c + wp.int32(1)
                partition_data_concat[tid] = (wp.int64(fallback_c + wp.int32(1)) << _COLOR_SHIFT) | wp.int64(tid)
                wp.atomic_sub(num_remaining, 0, 1)
            else:
                color_tags[tid] = c + wp.int32(1)
                partition_data_concat[tid] = (wp.int64(c + 1) << _COLOR_SHIFT) | wp.int64(tid)
                # Strictly-monotonic capture_while predicate update:
                # one decrement per commit, no off-thread interaction.
                wp.atomic_sub(num_remaining, 0, 1)


@wp.kernel(enable_backward=False)
def partitioning_coloring_incremental_kernel(
    partition_data_concat: wp.array[wp.int64],
    random_values: wp.array[wp.int32],
    cost_values: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    remaining_ids: wp.array[int],
    num_remaining: wp.array[int],
    color_arr: wp.array[int],
):
    """Jones-Plassmann MIS pass over a compact remaining-ids list.

    Same MIS logic as :func:`partitioning_coloring_kernel` but drives
    from ``remaining_ids[0..num_remaining[0])`` instead of filtering
    ``[0, max_num_interactions)`` by ``is_removed``. Every launched
    lane has real work (full-warp utilisation on the hot adjacency
    walk) and the launch grid shrinks with each round.

    Neighbours from the adjacency list still go through
    ``contact_partitions_is_removed`` since they can come from any
    earlier round.
    """
    slot = wp.tid()

    if slot >= num_remaining[0]:
        return

    tid = remaining_ids[slot]
    color_copy = color_arr[0]

    # Elements in the compact list are -- by construction -- not yet
    # assigned, so the outer ``is_removed`` self-check that the classic
    # kernel runs is redundant here and is skipped.

    is_local_max = bool(True)

    self_prio = contact_partitions_get_random_value(random_values, cost_values, tid)
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
            if contact_partitions_get_random_value(random_values, cost_values, neighbor) > self_prio:
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
    color_tags: wp.array[wp.int32],
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
      packed per-element (color, tid) tag the round-based JP path
      reads and writes.
    * ``color_tags[tid] = 0``, the int32 colour-only mirror the
      greedy kernel reads on the hot path.
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
    color_tags[tid] = wp.int32(0)
    interaction_id_to_partition[tid] = -1


@wp.kernel(enable_backward=False)
def set_int_array_kernel(arr: wp.array[int], value: int):
    # Tiny helper used by batch launcher to push the host-side color counter
    # into the device array consumed by ``partitioning_coloring_kernel``.
    arr[0] = value


# Block size for the single-block tile-scan kernel. 1024 = largest CUDA block
# that Warp supports reliably, keeps the running-prefix sequential tail short.
TILE_SCAN_BLOCK_DIM = wp.constant(1024)


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

    Single-block tile-scan launch that replaces the old
    flag / scan / compact / advance kernel chain. In-place compacts
    ``remaining_ids[0..num_remaining)`` -- dropping elements just
    committed to colour ``cc = current_color[0]`` -- so the next
    coloring round visits only active lanes (keeps warps fully
    utilised; previously ~50% of lanes early-exited).

    Per-tile flags + two threaded tile scans:

    * ``committed_flag`` (just coloured): exclusive scan drives the
      writes into ``partition_element_ids`` /
      ``interaction_id_to_partition``.
    * ``survivor_flag`` (still active): exclusive scan drives the
      in-place compaction of ``remaining_ids``.

    In-place safety: survivors are packed leftward, so every write
    goes to an index ``<=`` the read it came from. Within a tile the
    scan primitive block-syncs reads vs writes; across tiles tile
    N's write window ``[running, running + tile_survivors)`` is
    strictly below tile N+1's read window
    ``[(N+1)*1024, (N+2)*1024)`` because survivors never exceed
    slots processed.

    Lane 0 publishes totals and advances ``current_color`` only
    after the last tile (every lane latched ``cc`` into a register
    at the top, so the single late write is race-free).
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

    Same per-tile algorithm but committed elements are appended to
    the CSR layout at ``element_ids_by_color[color_starts[cc]..]``
    instead of a one-shot buffer. Lane 0 additionally writes
    ``color_starts[cc + 1]`` (so the next colour -- and the trailing
    ``num_colors`` sentinel slot -- starts at the right offset) and
    mirrors ``current_color`` into ``num_colors`` for the outer
    sweep-time ``capture_while``.
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


# ---------------------------------------------------------------------------
# Greedy-mode helper kernels.
#
# The greedy variant of the JP coloring assigns each committed vertex the
# smallest colour not used by its neighbours, so commits within one round
# can land in any colour (not just the round number). This breaks the
# colour-equals-round assumption baked into
# ``incremental_tile_compact_csr_and_advance_kernel``. The kernels below
# replace that compaction with: (1) a per-colour histogram, (2) an
# exclusive-prefix scan to derive ``color_starts``, (3) a deterministic
# scatter into ``element_ids_by_color`` keyed by the per-element colour
# and ordered by element id within each colour.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def greedy_clear_int_kernel(arr: wp.array[int]):
    """Single-thread no-op zero of ``arr[0]``.

    Used by ``build_csr_greedy_with_jp_fallback`` to clear the
    fallback-decision flag at the end of its in-graph
    ``wp.capture_while`` body so the conditional graph runs the JP
    fallback at most once.
    """
    if wp.tid() == 0:
        arr[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def greedy_reset_init_kernel(
    overflow_flag: wp.array[int],
    color_count: wp.array[int],
    color_offsets: wp.array[int],
    max_colors: int,
):
    """One-shot reset of the build-wide control state.

    Called once per :meth:`IncrementalContactPartitioner.build_csr_greedy`
    before the JP loop starts. Zeroes:

      * the overflow flag,
      * the per-colour histogram bucket and the live scatter cursor
        (both indexed ``[0, max_colors)``).

    The ``num_remaining`` capture_while predicate is initialised by
    :func:`incremental_init_csr_kernel` (set to ``num_elements``)
    before this kernel runs; the greedy kernel then atomically
    decrements it once per commit until it hits zero.
    """
    tid = wp.tid()
    if tid < max_colors:
        color_count[tid] = wp.int32(0)
        color_offsets[tid] = wp.int32(0)
    if tid == 0:
        overflow_flag[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def greedy_color_histogram_kernel(
    partition_data_concat: wp.array[wp.int64],
    num_elements: wp.array[int],
    color_count: wp.array[int],
    interaction_id_to_partition: wp.array[int],
):
    """Count vertices per colour and stamp ``interaction_id_to_partition``.

    Walks every element ``tid < num_elements[0]``, reads its packed
    colour out of ``partition_data_concat``, atomically increments
    ``color_count[colour]``, and writes the colour into
    ``interaction_id_to_partition`` so consumers (e.g.
    ``World.num_colors_used``) see the same per-element assignments
    they did under JP. Uncoloured elements (colour == 0) are not
    counted -- if any survive at this point it is a build bug; the
    overflow path on the greedy kernel raises before we reach here.

    Reads from the int64 ``partition_data_concat`` rather than the
    parallel int32 ``color_tags`` mirror so the JP-fallback path
    (which only writes ``partition_data_concat``) sees consistent
    values when the greedy build overflows and the round-based JP
    rebuild fires inside the same captured frame.
    """
    tid = wp.tid()
    n = num_elements[0]
    if tid >= n:
        return
    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = tagged >> _COLOR_SHIFT
    if color_plus_one == wp.int64(0):
        # Should be unreachable on a converged greedy build. Stamp -1
        # so downstream code at least sees a distinct sentinel.
        interaction_id_to_partition[tid] = wp.int32(-1)
        return
    c = wp.int32(color_plus_one - wp.int64(1))
    interaction_id_to_partition[tid] = c
    wp.atomic_add(color_count, c, 1)


@wp.kernel(enable_backward=False)
def greedy_count_and_scan_color_starts_kernel(
    color_count: wp.array[int],
    color_starts: wp.array[int],
    num_colors: wp.array[int],
    max_colors: int,
):
    """Single-thread fused exclusive-scan over ``color_count`` plus
    ``num_colors`` derivation.

    Walks ``color_count[0..max_colors)`` once and writes:

      * ``color_starts[c] = exclusive prefix sum of color_count[0..c)``,
        for ``c`` in ``[0, max_colors]`` (so ``color_starts[max_colors]``
        holds the total).
      * ``num_colors[0] = 1 + max c such that color_count[c] > 0``.

    Replaces two separate one-thread kernels (``greedy_count_num_colors``
    and ``greedy_color_starts_scan``). With ``max_colors <=
    GREEDY_MAX_COLORS`` (= 64) the inner walk is 64 iterations on one
    thread, well within the same launch overhead as either kernel
    alone -- the build saves one ~25us kernel launch per ``World.step``
    by collapsing the two passes.
    """
    tid = wp.tid()
    if tid != 0:
        return
    running = wp.int32(0)
    last = wp.int32(-1)
    for c in range(max_colors + 1):
        color_starts[c] = running
        if c < max_colors:
            cnt = color_count[c]
            if cnt > 0:
                last = c
            running = running + cnt
    num_colors[0] = last + 1


@wp.kernel(enable_backward=False)
def greedy_scatter_elements_by_color_kernel(
    partition_data_concat: wp.array[wp.int64],
    color_starts: wp.array[int],
    color_offsets: wp.array[int],
    element_ids_by_color: wp.array[int],
    num_elements: wp.array[int],
):
    """Scatter each coloured element into the CSR slot assigned to its colour.

    For each element ``tid``:
      1. Read its colour ``c`` from the packed tag word.
      2. Atomic-add 1 to ``color_offsets[c]`` to claim a slot.
      3. Write its id into
         ``element_ids_by_color[color_starts[c] + slot]``.

    ``color_offsets`` is a separate scratch array initialised to 0
    at the start of the scatter so the atomics produce a 1:1 map
    onto the CSR range owned by colour ``c``. Order of element ids
    within a colour is non-deterministic (atomics) but the *set* of
    elements per colour is fully determined by the greedy build.
    Downstream PGS sweeps already iterate the colour slice as an
    independent set, so order does not matter.

    Reads from ``partition_data_concat`` (not ``color_tags``) for the
    same reason as :func:`greedy_color_histogram_kernel`: the JP-
    fallback path leaves ``color_tags`` stale.
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = tagged >> _COLOR_SHIFT
    if color_plus_one == wp.int64(0):
        return
    c = wp.int32(color_plus_one - wp.int64(1))
    slot = wp.atomic_add(color_offsets, c, 1)
    element_ids_by_color[color_starts[c] + slot] = tid
