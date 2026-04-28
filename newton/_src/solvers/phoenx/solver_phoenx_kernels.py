# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for :class:`PhoenXWorld`.

Split from :mod:`solver_phoenx` so the driver class stays readable.
Dispatches only the two constraint types the solver supports:
:data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` and
:data:`CONSTRAINT_TYPE_CONTACT`.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    actuated_double_ball_socket_iterate,
    actuated_double_ball_socket_iterate_multi,
    actuated_double_ball_socket_prepare_for_iteration,
    actuated_double_ball_socket_world_error,
    actuated_double_ball_socket_world_wrench,
    revolute_iterate,
    revolute_iterate_multi,
    revolute_prepare_for_iteration,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_iterate,
    contact_iterate_multi,
    contact_prepare_for_iteration,
    contact_world_error,
    contact_world_wrench,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    GREEDY_MAX_COLORS,
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_make,
)
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia

__all__ = [
    "_PER_WORLD_COLORING_BLOCK_DIM",
    "_STRAGGLER_BLOCK_DIM",
    "_build_scatter_keys_kernel",
    "_choose_fast_tail_worlds_per_block",
    "_constraint_gather_errors_kernel",
    "_constraint_gather_wrenches_kernel",
    "_constraint_iterate_singleworld_fused_revolute_kernel",
    "_constraint_iterate_singleworld_kernel",
    "_constraint_iterate_singleworld_revolute_kernel",
    "_constraint_prepare_plus_iterate_fast_tail_kernel",
    "_constraint_prepare_plus_iterate_fast_tail_revolute_kernel",
    "_constraint_prepare_singleworld_fused_revolute_kernel",
    "_constraint_prepare_singleworld_kernel",
    "_constraint_prepare_singleworld_revolute_kernel",
    "_constraint_relax_fast_tail_kernel",
    "_constraint_relax_fast_tail_revolute_kernel",
    "_constraint_relax_singleworld_fused_revolute_kernel",
    "_constraint_relax_singleworld_kernel",
    "_constraint_relax_singleworld_revolute_kernel",
    "_constraints_to_elements_kernel",
    "_count_elements_per_world_kernel",
    "_integrate_velocities_kernel",
    "_kinematic_interpolate_substep_kernel",
    "_kinematic_prepare_step_kernel",
    "_per_world_greedy_coloring_kernel",
    "_per_world_jp_coloring_kernel",
    "_phoenx_apply_forces_and_gravity_kernel",
    "_phoenx_apply_global_damping_kernel",
    "_phoenx_refresh_world_inertia_kernel",
    "_phoenx_update_inertia_and_clear_forces_kernel",
    "_pick_threads_per_world_kernel",
    "_reduce_total_colours_kernel",
    "_rotation_quaternion",
    "_set_kinematic_pose_batch_kernel",
    "_sync_num_active_constraints_kernel",
    "pack_body_xforms_kernel",
]


#: Maximum threads-per-world the fast-tail kernels ever use. Equal to
#: warp size; the launch dim is always sized to ``num_worlds *
#: _STRAGGLER_BLOCK_DIM`` so the per-step adaptive picker can drop the
#: effective threads-per-world to 16 or 8 without re-launching with a
#: different grid -- the surplus threads early-exit on
#: ``world_id >= num_worlds``.
_STRAGGLER_BLOCK_DIM: int = 32

# PGS sweeps that ``*_iterate_multi`` runs per call. Must evenly
# divide ``solver_iterations``; each value > 1 amortises per-cid body
# / constraint reloads (the body state and per-cid constraint
# constants are loaded once and held in registers for the whole
# multi-sweep) but *shrinks* cross-colour PGS feedback to
# ``solver_iterations / _FUSED_INNER_SWEEPS`` rounds.
#
# Empirically: ``2`` gives a clean +17-21% on g1_flat / h1_flat
# multi-world and still passes the full stacking / articulation /
# high-mass-ratio test suite (32 cases). ``4`` halves outer rounds
# again and saves a bit more bandwidth, but ``test_slam_ball_into_stack``
# starts failing -- a heavy ball impacting a tower needs the finer
# cross-colour feedback to dissipate the impulse without driving
# bodies through neighbours.
_FUSED_INNER_SWEEPS: int = 2

_PRIORITY_COST_SHIFT = wp.constant(wp.int64(32))
_PRIORITY_JITTER_MASK = wp.constant(wp.int64((1 << 32) - 1))


def _choose_fast_tail_worlds_per_block(num_worlds: int) -> int:
    """Worlds per physical block in the fast-tail kernels.

    Each world owns one warp (32 threads); block size is ``32 * wpb``
    so ``__syncwarp()`` stays valid. Three-tier by world count,
    empirically tuned on RTX PRO 6000 (sm_120, 188 SMs):
    ``wpb = 2`` below 512 worlds, ``wpb = 4`` up to 2048, ``wpb = 8``
    above.
    """
    if num_worlds < 512:
        return 2
    if num_worlds < 2048:
        return 4
    return 8


#: Upper bound on the block size the fast-tail launches will ever use.
#: Exposed for downstream code that wants to bound padded launch dim
#: without calling :func:`_choose_fast_tail_worlds_per_block` per-launch.
_FAST_TAIL_MAX_BLOCK_DIM: int = _STRAGGLER_BLOCK_DIM * 8


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp();
#endif
""")
def _sync_warp(): ...


# ---------------------------------------------------------------------------
# Adaptive threads-per-world picker
# ---------------------------------------------------------------------------
#
# Fast-tail kernels launch at a fixed ``num_worlds *
# _STRAGGLER_BLOCK_DIM`` grid (one warp slot per world at tpw=32) and
# read effective tpw from a 1-element buffer written once per step by
# :func:`_pick_threads_per_world_kernel`. Smaller tpw maps the top
# lanes to ``world_id >= num_worlds`` and early-exits them. Sparse-
# colour scenes (h1_flat ~5 cids/colour/world) double lane utilisation
# by packing two worlds per warp -- without changing host launch
# shape, so the whole step stays in a single CUDA graph.


@wp.kernel(enable_backward=False)
def _reduce_total_colours_kernel(
    world_num_colors: wp.array[wp.int32],
    num_worlds: wp.int32,
    # out
    total_colours: wp.array[wp.int32],
):
    """Parallel atomic-sum of ``world_num_colors`` into a 1-element scalar.

    Caller must zero ``total_colours`` before launch (``zero_()``). Each
    thread handles one world; threads beyond ``num_worlds`` early-exit
    so the launch grid can be over-sized to a fixed number for
    graph-capture stability.
    """
    tid = wp.tid()
    if tid >= num_worlds:
        return
    nc = world_num_colors[tid]
    if nc > 0:
        wp.atomic_add(total_colours, 0, nc)


@wp.kernel(enable_backward=False)
def _pick_threads_per_world_kernel(
    world_csr_offsets: wp.array[wp.int32],
    total_colours: wp.array[wp.int32],
    num_worlds: wp.int32,
    sm_count: wp.int32,
    # out
    tpw_choice: wp.array[wp.int32],
):
    """One thread; picks tpw in {16, 32} from precomputed totals.

    Two-tier heuristic (tuned on RTX PRO 6000 sm_count=188):

    * **tpw=32** (default, one warp per world) when warps/SM < 8
      (occupancy-bound) or colours are dense
      (mean >= 6 cids/colour).
    * **tpw=16** (two worlds per warp) when both gates hit: enough
      warps per SM to halve (>= 8 warps/SM at tpw=32) AND sparse
      colours (mean <= 6 cids/colour). +4-12% end-to-end on
      h1_flat-class fleets above 2048 worlds.

    ``tpw=8`` is reachable via the static
    ``threads_per_world=8`` arg but the auto picker never emits it
    (gap over tpw=16 is noise, occasional 12% -> 3% regressions).
    All comparisons in fixed-point (x16) so no FP ops; O(1) inside
    the captured graph.
    """
    if wp.tid() != 0:
        return

    # ``world_csr_offsets`` is the inclusive prefix scan of per-world
    # cid counts; the entry past the last world holds the grand total.
    total_cids = world_csr_offsets[num_worlds]
    nc = total_colours[0]

    if nc <= 0 or num_worlds <= 0:
        tpw_choice[0] = 32
        return

    # Mean cids per (world, colour). Integer math (mean_x16 = mean *
    # 16) so the thresholds below stay in int32 land.
    mean_x16 = (total_cids * wp.int32(16)) / nc

    # Saturation: warps available per SM at tpw=32. Below ~8 we can't
    # afford to halve them without losing SM occupancy.
    warps_at_tpw32 = num_worlds  # 1 warp/world at tpw=32
    saturation_x16 = (warps_at_tpw32 * wp.int32(16)) / wp.max(sm_count, wp.int32(1))

    pick = wp.int32(32)
    if mean_x16 <= wp.int32(6 * 16) and saturation_x16 >= wp.int32(8 * 16):
        pick = wp.int32(16)

    tpw_choice[0] = pick


# ---------------------------------------------------------------------------
# Per-world graph coloring (one block per world)
# ---------------------------------------------------------------------------
#
# Worlds are independent after static-body nullification -- no element
# in world w references a body in any other world. Jones-Plassmann MIS
# partitioning runs per-world in parallel: one block per world, each
# block runs the full JP loop on its world's element subset. Output
# lands directly in the per-world CSR (``world_element_ids_by_color``,
# ``world_color_starts[w, c]``, ``world_csr_offsets[w]``,
# ``world_num_colors[w]``) that the fast-tail dispatchers consume.


_PER_WORLD_COLORING_BLOCK_DIM: int = 64


@wp.kernel(enable_backward=False)
def _count_elements_per_world_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    bodies: BodyContainer,
    # out
    world_element_count: wp.array[wp.int32],  # [nw]
    world_element_offsets_shifted: wp.array[wp.int32],  # [nw+1], will be inclusive-scanned
):
    """Atomic count of per-world elements.

    Writes both the raw count (consumed by the scatter kernel) AND
    the shifted form (``shifted[w + 1] += 1``) so an inclusive scan
    produces an exclusive prefix + trailing total in a single pass.
    Thread 0 additionally stamps ``shifted[0] = 0`` (the first
    scan-input slot).

    Elements with ``bodies[0] == -1`` (both bodies static -- should not
    happen in active constraints but guard anyway) contribute to no
    world and are skipped downstream.
    """
    tid = wp.tid()
    n = num_elements[0]
    if tid == wp.int32(0):
        # The inclusive scan reads shifted[0] as the base value; stamp
        # 0 once so offsets[0] = 0 regardless of the tail garbage.
        world_element_offsets_shifted[0] = wp.int32(0)
    if tid >= n:
        return
    b = elements[tid].bodies[0]
    if b < 0:
        return
    w = bodies.world_id[b]
    wp.atomic_add(world_element_count, w, wp.int32(1))
    wp.atomic_add(world_element_offsets_shifted, w + wp.int32(1), wp.int32(1))


@wp.kernel(enable_backward=False)
def _build_scatter_keys_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    bodies: BodyContainer,
    cap: wp.int32,
    # out (size ``2 * cap`` -- ping-pong buffer for ``radix_sort_pairs``)
    keys: wp.array[wp.int32],
    values: wp.array[wp.int32],
):
    """Populate (key, value) pairs for the per-world scatter sort.

    The downstream radix sort is stable, so within a key (= world id)
    the cids land in their original index order. That makes the
    per-world bucket assignment deterministic without any atomics --
    replacing the older atomic-cursor scatter that produced the same
    counts but a thread-scheduling-dependent order.

    Inactive cids (``bodies[0] < 0``) and tail slots beyond
    ``num_elements`` are tagged with ``INT32_MAX`` so they sort to
    the end of the active region; the JP coloring only reads the
    first ``world_element_count[w]`` entries of each bucket so the
    tail is ignored.
    """
    tid = wp.tid()
    if tid >= cap:
        return
    n = num_elements[0]
    if tid >= n:
        keys[tid] = wp.int32(2147483647)
        values[tid] = wp.int32(-1)
        return
    b = elements[tid].bodies[0]
    if b < 0:
        keys[tid] = wp.int32(2147483647)
        values[tid] = wp.int32(-1)
        return
    keys[tid] = bodies.world_id[b]
    values[tid] = tid


@wp.func
def _cost_biased_priority(
    random_values: wp.array[wp.int32],
    cost_values: wp.array[wp.int32],
    cid: wp.int32,
) -> wp.int64:
    """Return lexicographic JP priority: high 32 bits cost, low 32 bits jitter."""
    cost = wp.int64(cost_values[cid])
    jitter = wp.int64(random_values[cid]) & _PRIORITY_JITTER_MASK
    return (cost << _PRIORITY_COST_SHIFT) | jitter


@wp.kernel(enable_backward=False)
def _per_world_jp_coloring_kernel(
    # per-world bucketing (input from the two kernels above)
    world_element_offsets: wp.array[wp.int32],  # [nw+1] (exclusive prefix of counts)
    world_element_count: wp.array[wp.int32],  # [nw] (raw per-world count)
    world_elements: wp.array[wp.int32],  # [total] flat cid stream, sorted by world
    # graph data
    elements: wp.array[ElementInteractionData],
    adjacency_end: wp.array[wp.int32],  # [num_bodies]
    vertex_to_elements: wp.array[wp.int32],  # [cap * MAX_BODIES]
    random_values: wp.array[wp.int32],  # [capacity] JP priorities
    cost_values: wp.array[wp.int32],  # [capacity] JP costs (contacts use contact_count)
    max_colors: wp.int32,
    # scratch (caller zeros each step)
    assigned: wp.array[wp.int32],  # [capacity] 0 unassigned, (c+1) = coloured
    # outputs
    world_element_ids_by_color: wp.array[wp.int32],  # [total] sorted-by-colour per world
    world_color_starts: wp.array2d[wp.int32],  # [nw, MAX_COLORS+1] per-world prefix
    world_num_colors: wp.array[wp.int32],  # [nw]
):
    """Run Jones-Plassmann MIS coloring independently in each world.

    One block per world. Inside the block:

    1. Clear ``assigned`` for the world's elements.
    2. Loop: pick local maxima in priority, commit them as the next
       colour, repeat until everything is assigned.
    3. Each committed element writes into ``world_element_ids_by_color``
       at offset ``world_element_offsets[w] + color_base + slot`` where
       ``slot`` is an atomic-bump on a per-world cursor (reset each
       round) and ``color_base`` is the cumulative committed count up
       to this round.

    Correctness (same as the original global JP kernel):
    ``contact_partitions_is_removed``-style check on neighbours ignores
    any neighbour settled in a **prior** colour. Neighbours settled
    **this** colour still contribute via the priority comparison --
    since JP commits only elements that are the local maximum among
    still-active neighbours, at most one of any pair of neighbours
    commits per round.
    """
    block, lane = wp.tid()
    w = block
    base = world_element_offsets[w]
    count = world_element_count[w]

    if count == 0:
        if lane == wp.int32(0):
            world_num_colors[w] = wp.int32(0)
            world_color_starts[w, 0] = wp.int32(0)
        return

    # Phase 1: zero per-element assigned flags for this world's elements.
    stride = _PER_WORLD_COLORING_BLOCK_DIM
    offset = wp.int32(0)
    while offset < count:
        slot = offset + lane
        if slot < count:
            eid = world_elements[base + slot]
            assigned[eid] = wp.int32(0)
        offset = offset + stride

    if lane == wp.int32(0):
        world_color_starts[w, 0] = wp.int32(0)

    _sync_threads()

    current_color = wp.int32(0)
    num_remaining = count
    color_base = wp.int32(0)

    while num_remaining > wp.int32(0) and current_color < max_colors:
        # Phase 2: find local maxima and commit them. All lanes run
        # the stride loop the same number of times so the
        # block-collective tile reduction/scan sees every lane at the
        # same point. Slot assignment uses ``tile_scan_exclusive``
        # (deterministic, depends only on ``committed_here``) rather
        # than an atomic cursor, at the same per-step cost.
        committed_this_round = wp.int32(0)
        offset = wp.int32(0)
        while offset < count:
            slot = offset + lane
            committed_here = wp.int32(0)
            committed_eid = wp.int32(0)
            if slot < count:
                eid = world_elements[base + slot]
                if assigned[eid] == wp.int32(0):
                    self_prio = _cost_biased_priority(random_values, cost_values, eid)
                    is_local_max = bool(True)
                    for j in range(MAX_BODIES):
                        if not is_local_max:
                            break
                        b = elements[eid].bodies[j]
                        if b < 0:
                            break
                        adj_start = wp.int32(0)
                        if b > 0:
                            adj_start = adjacency_end[b - 1]
                        adj_end_b = adjacency_end[b]
                        for k in range(adj_start, adj_end_b):
                            if not is_local_max:
                                break
                            neighbor = vertex_to_elements[k]
                            if neighbor == eid:
                                continue
                            a = assigned[neighbor]
                            # Settled in a prior colour -> skip
                            # (graph edge can't conflict).
                            if a != wp.int32(0) and a != current_color + wp.int32(1):
                                continue
                            if _cost_biased_priority(random_values, cost_values, neighbor) > self_prio:
                                is_local_max = bool(False)

                    if is_local_max:
                        assigned[eid] = current_color + wp.int32(1)
                        committed_here = wp.int32(1)
                        committed_eid = eid

            # Block-wide deterministic slot assignment for the lanes
            # that committed this iteration. ``tile_scan_exclusive``
            # turns the per-lane 0/1 ``committed_here`` flags into
            # per-lane prefix offsets; ``tile_sum`` gives the
            # iteration's total so we can advance the colour-base
            # offset for the next iteration.
            committed_tile = wp.tile(committed_here)
            iter_prefix = wp.tile_scan_exclusive(committed_tile)
            iter_total_tile = wp.tile_sum(committed_tile)
            iter_total = iter_total_tile[0]
            if committed_here == wp.int32(1):
                world_element_ids_by_color[base + color_base + committed_this_round + iter_prefix[lane]] = committed_eid
            committed_this_round = committed_this_round + iter_total
            offset = offset + stride

        _sync_threads()

        color_base = color_base + committed_this_round
        num_remaining = num_remaining - committed_this_round
        current_color = current_color + wp.int32(1)

        if lane == wp.int32(0):
            world_color_starts[w, current_color] = color_base

        if committed_this_round == wp.int32(0):
            break

    if lane == wp.int32(0):
        world_num_colors[w] = current_color


# Constant used to flip a 64-bit forbidden-color mask without a unary
# bitwise NOT (Warp's int64 codegen is unreliable; mirrors the
# ``_FREE_COLOR_FLIP`` constant in graph_coloring_common.py).
_PER_WORLD_FREE_COLOR_FLIP = wp.constant(wp.int64(-1))


@wp.kernel(enable_backward=False)
def _per_world_greedy_coloring_kernel(
    # per-world bucketing (input from the two kernels above)
    world_element_offsets: wp.array[wp.int32],  # [nw+1] (exclusive prefix of counts)
    world_element_count: wp.array[wp.int32],  # [nw] (raw per-world count)
    world_elements: wp.array[wp.int32],  # [total] flat cid stream, sorted by world
    # graph data
    elements: wp.array[ElementInteractionData],
    adjacency_end: wp.array[wp.int32],  # [num_bodies]
    vertex_to_elements: wp.array[wp.int32],  # [cap * MAX_BODIES]
    random_values: wp.array[wp.int32],  # [capacity] JP priorities
    cost_values: wp.array[wp.int32],  # [capacity] JP costs (unused in greedy mode)
    max_colors: wp.int32,  # = GREEDY_MAX_COLORS, kept for parity with JP variant
    # scratch (caller zeros each step)
    assigned: wp.array[wp.int32],  # [capacity] 0 unassigned, (c+1) = coloured
    color_count: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS] histogram bucket
    color_offsets: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS] live cursor for scatter
    # outputs
    world_element_ids_by_color: wp.array[wp.int32],  # [total] sorted-by-colour per world
    world_color_starts: wp.array2d[wp.int32],  # [nw, MAX_COLORS+1] per-world prefix
    world_num_colors: wp.array[wp.int32],  # [nw]
    overflow_flag: wp.array[wp.int32],  # [1] set if any world exceeds GREEDY_MAX_COLORS
):
    """JP-MIS + smallest-free-color greedy variant of
    :func:`_per_world_jp_coloring_kernel`.

    Same per-world dispatch (one block per world, lanes stride over
    the world's element list) and the same MIS contract (a vertex
    commits iff it is the highest priority among its still-uncolored
    neighbours -- never two within one round). What changes:

      * Each committed vertex gets the smallest colour not already
        used by its already-coloured neighbours, computed from a
        per-vertex int64 forbidden-color bitmask. This drops the
        per-world colour count to the chromatic lower bound on dense
        sub-graphs.
      * Because commits within one round can land in different
        colours, the round-equals-colour CSR scatter the JP variant
        uses no longer applies. We make two extra passes: a
        per-world histogram + exclusive prefix scan to derive
        ``world_color_starts``, then an atomic scatter into
        ``world_element_ids_by_color``. Both passes stay inside the
        block; no cross-block synchronisation is required.

    Order of element ids within a colour is non-deterministic
    (atomics on ``color_offsets``), but the *set* of elements per
    colour is fully determined by the input. PGS sweeps consume each
    colour as an unordered independent set so simulation outputs are
    bit-deterministic.
    """
    block, lane = wp.tid()
    w = block
    base = world_element_offsets[w]
    count = world_element_count[w]

    if count == 0:
        if lane == wp.int32(0):
            world_num_colors[w] = wp.int32(0)
            world_color_starts[w, 0] = wp.int32(0)
        return

    # Phase 1: zero per-element ``assigned`` flags + per-world
    # histogram / cursor buckets. The histogram is sized
    # ``GREEDY_MAX_COLORS`` so the reset always covers the full row
    # regardless of how many colours this world ends up needing.
    stride = _PER_WORLD_COLORING_BLOCK_DIM
    offset = wp.int32(0)
    while offset < count:
        slot = offset + lane
        if slot < count:
            eid = world_elements[base + slot]
            assigned[eid] = wp.int32(0)
        offset = offset + stride

    offset = wp.int32(0)
    while offset < GREEDY_MAX_COLORS:
        slot = offset + lane
        if slot < GREEDY_MAX_COLORS:
            color_count[w, slot] = wp.int32(0)
            color_offsets[w, slot] = wp.int32(0)
        offset = offset + stride

    if lane == wp.int32(0):
        world_color_starts[w, 0] = wp.int32(0)

    _sync_threads()

    # Phase 2: greedy MIS+colour rounds. Each round picks an
    # independent set (MIS) and assigns each picked vertex the
    # smallest colour not used by its colored neighbours. Loop
    # terminates when no vertex remains uncoloured.
    num_remaining = count
    overflow_local = wp.int32(0)

    # Bound the outer loop with a hard cap to keep the kernel
    # finite-launch-safe; in practice greedy converges in tens of
    # rounds, well below ``count``.
    round_idx = wp.int32(0)
    while num_remaining > wp.int32(0) and round_idx < count:
        committed_this_round = wp.int32(0)
        offset = wp.int32(0)
        while offset < count:
            slot = offset + lane
            committed_here = wp.int32(0)
            if slot < count:
                eid = world_elements[base + slot]
                if assigned[eid] == wp.int32(0):
                    self_prio = _cost_biased_priority(random_values, cost_values, eid)
                    is_local_max = bool(True)
                    forbidden_mask = wp.int64(0)
                    for j in range(MAX_BODIES):
                        if not is_local_max:
                            break
                        b = elements[eid].bodies[j]
                        if b < 0:
                            break
                        adj_start = wp.int32(0)
                        if b > 0:
                            adj_start = adjacency_end[b - 1]
                        adj_end_b = adjacency_end[b]
                        for k in range(adj_start, adj_end_b):
                            if not is_local_max:
                                break
                            neighbor = vertex_to_elements[k]
                            if neighbor == eid:
                                continue
                            a = assigned[neighbor]
                            if a == wp.int32(0):
                                # Uncoloured neighbour: MIS tiebreak.
                                if _cost_biased_priority(random_values, cost_values, neighbor) > self_prio:
                                    is_local_max = bool(False)
                            else:
                                # Coloured neighbour: forbid that colour.
                                ncolor = a - wp.int32(1)
                                if ncolor < GREEDY_MAX_COLORS:
                                    forbidden_mask = forbidden_mask | (wp.int64(1) << wp.int64(ncolor))

                    if is_local_max:
                        # Smallest free colour = first 0-bit in mask.
                        free_mask = forbidden_mask ^ _PER_WORLD_FREE_COLOR_FLIP
                        c = wp.int32(0)
                        for _ in range(GREEDY_MAX_COLORS):
                            if (free_mask & (wp.int64(1) << wp.int64(c))) != wp.int64(0):
                                break
                            c = c + wp.int32(1)
                        if c >= GREEDY_MAX_COLORS:
                            overflow_local = wp.int32(1)
                        else:
                            assigned[eid] = c + wp.int32(1)
                            wp.atomic_add(color_count, w, c, wp.int32(1))
                            committed_here = wp.int32(1)

            committed_tile = wp.tile(committed_here)
            iter_total_tile = wp.tile_sum(committed_tile)
            iter_total = iter_total_tile[0]
            committed_this_round = committed_this_round + iter_total
            offset = offset + stride

        _sync_threads()
        num_remaining = num_remaining - committed_this_round
        if committed_this_round == wp.int32(0):
            # Either the world converged or the bitmask is saturated
            # for some vertex. Either way, stop -- the overflow flag
            # below catches the saturated case.
            break
        round_idx = round_idx + wp.int32(1)

    if overflow_local != wp.int32(0) and lane == wp.int32(0):
        overflow_flag[0] = wp.int32(1)

    # Phase 3: histogram-driven CSR build for this world.
    # ``color_count[w, c]`` already holds the bucket size from the
    # atomic_adds above. Compute exclusive prefix into
    # ``world_color_starts[w, :]`` on lane 0 (cheap, GREEDY_MAX_COLORS
    # = 64 entries) so subsequent scatter sees stable offsets.
    if lane == wp.int32(0):
        running = wp.int32(0)
        last_used = wp.int32(-1)
        for c in range(GREEDY_MAX_COLORS):
            world_color_starts[w, c] = running
            cnt = color_count[w, c]
            if cnt > wp.int32(0):
                last_used = c
            running = running + cnt
        world_color_starts[w, GREEDY_MAX_COLORS] = running
        world_num_colors[w] = last_used + wp.int32(1)
    _sync_threads()

    # Phase 4: scatter element ids into the per-world CSR slice. Each
    # lane walks a stride of the world's element list, looks up its
    # assigned colour, and atomic-bumps ``color_offsets[w, c]`` to
    # claim a unique slot in the colour's range.
    offset = wp.int32(0)
    while offset < count:
        slot = offset + lane
        if slot < count:
            eid = world_elements[base + slot]
            a = assigned[eid]
            if a > wp.int32(0):
                c = a - wp.int32(1)
                local_slot = wp.atomic_add(color_offsets, w, c, wp.int32(1))
                start = world_color_starts[w, c]
                world_element_ids_by_color[base + start + local_slot] = eid
        offset = offset + stride


# ---------------------------------------------------------------------------
# Fast-path unified single-block dispatchers
# ---------------------------------------------------------------------------
#
# One block per world; each block walks its world's full CSR internally.
# Inside each block: outer iteration loop, middle colour sweep with
# ``__syncthreads`` between colours, inner block-stride lane loop.
# Partitioner guarantees no two same-colour elements share a body, so
# per-lane RMW on body velocities is race-free.


# Multi-world fast-tail kernels: factory + revolute / generic variants.
#
# Same body for the prepare-plus-iterate and the relax kernel; they
# differ in (a) whether the prepare pass runs once before the iterate
# sweeps and (b) whether ``use_bias`` is on (iterate) or off (relax).
# Each is parameterised on ``revolute_only`` to skip the per-cid
# ``read_int(_OFF_JOINT_MODE)`` global load and joint-mode branch in
# all-revolute scenes (G1 / H1 / chain / ragdoll).


def _make_fast_tail_prepare_plus_iterate_kernel(*, revolute_only: bool):
    """Build the multi-world fused prepare + iterate fast-tail kernel
    for the requested joint specialisation."""

    @wp.kernel(enable_backward=False)
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        world_element_ids_by_color: wp.array[wp.int32],
        world_color_starts: wp.array2d[wp.int32],
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        tpw_buf: wp.array[wp.int32],
    ):
        tid = wp.tid()
        tpw = tpw_buf[0]
        local_tid = tid % tpw
        world_id = tid / tpw
        if world_id >= num_worlds:
            return

        n_colors = world_num_colors[world_id]
        world_base = world_csr_offsets[world_id]

        # ---- Prepare phase ----------------------------------------
        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + 1]
            count = end - start

            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                if cid < num_joints:
                    if wp.static(revolute_only):
                        revolute_prepare_for_iteration(constraints, cid, bodies, idt)
                    else:
                        actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
                else:
                    contact_prepare_for_iteration(contact_cols, cid - num_joints, bodies, idt, cc, contacts)
                base += tpw

            _sync_warp()
            c += 1

        # ---- Iterate phase ----------------------------------------
        # Split ``num_iterations`` into
        # ``outer_iters = num_iterations / _FUSED_INNER_SWEEPS`` outer
        # rounds of cross-colour PGS feedback, each running
        # ``_FUSED_INNER_SWEEPS`` sweeps on every cid with body state
        # and per-cid constraint constants held in registers
        # (``*_iterate_multi`` functions). For revolute joints (the
        # common G1 / H1 case) and contacts, the register-resident
        # path eliminates ``(_FUSED_INNER_SWEEPS - 1)`` redundant body
        # / constraint reads per cid per outer iteration.
        inner_sweeps = wp.int32(_FUSED_INNER_SWEEPS)
        outer_iters = num_iterations / inner_sweeps
        it_outer = wp.int32(0)
        while it_outer < outer_iters:
            c = wp.int32(0)
            while c < n_colors:
                start = world_base + world_color_starts[world_id, c]
                end = world_base + world_color_starts[world_id, c + 1]
                count = end - start

                base = local_tid
                while base < count:
                    cid = world_element_ids_by_color[start + base]
                    if cid < num_joints:
                        if wp.static(revolute_only):
                            revolute_iterate_multi(constraints, cid, bodies, idt, True, inner_sweeps)
                        else:
                            actuated_double_ball_socket_iterate_multi(constraints, cid, bodies, idt, True, inner_sweeps)
                    else:
                        contact_iterate_multi(contact_cols, cid - num_joints, bodies, idt, cc, contacts, True, inner_sweeps)
                    base += tpw

                _sync_warp()
                c += 1

            it_outer += 1

    return kernel


def _make_fast_tail_relax_kernel(*, revolute_only: bool):
    """Build the multi-world relax fast-tail kernel
    (``use_bias = False``, ``num_sweeps = num_iterations``) for the
    requested joint specialisation."""

    @wp.kernel(enable_backward=False)
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        world_element_ids_by_color: wp.array[wp.int32],
        world_color_starts: wp.array2d[wp.int32],
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        tpw_buf: wp.array[wp.int32],
    ):
        tid = wp.tid()
        tpw = tpw_buf[0]
        local_tid = tid % tpw
        world_id = tid / tpw
        if world_id >= num_worlds:
            return

        n_colors = world_num_colors[world_id]
        world_base = world_csr_offsets[world_id]

        # Dispatch via the register-cached ``*_iterate_multi`` path
        # with ``num_sweeps = num_iterations`` so each cid gets one
        # body + constraint data load amortised over the whole relax
        # sweep. ``velocity_iterations`` is typically 1 so there's no
        # real cross-colour feedback to preserve -- running the full
        # relax in one multi call is equivalent to the classic outer-
        # inner split.
        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + 1]
            count = end - start

            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                if cid < num_joints:
                    if wp.static(revolute_only):
                        revolute_iterate_multi(constraints, cid, bodies, idt, False, num_iterations)
                    else:
                        actuated_double_ball_socket_iterate_multi(constraints, cid, bodies, idt, False, num_iterations)
                else:
                    contact_iterate_multi(contact_cols, cid - num_joints, bodies, idt, cc, contacts, False, num_iterations)
                base += tpw

            _sync_warp()
            c += 1

    return kernel


_constraint_prepare_plus_iterate_fast_tail_kernel = _make_fast_tail_prepare_plus_iterate_kernel(
    revolute_only=False
)
_constraint_prepare_plus_iterate_fast_tail_revolute_kernel = (
    _make_fast_tail_prepare_plus_iterate_kernel(revolute_only=True)
)
_constraint_relax_fast_tail_kernel = _make_fast_tail_relax_kernel(revolute_only=False)
_constraint_relax_fast_tail_revolute_kernel = _make_fast_tail_relax_kernel(revolute_only=True)


@wp.kernel(enable_backward=False)
def _constraints_to_elements_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    num_constraints: wp.array[wp.int32],
    num_joints: wp.int32,
    elements: wp.array[ElementInteractionData],
):
    """Project every active constraint into ``ElementInteractionData``.

    Only the two body indices matter to the graph colourer; static
    bodies get collapsed to ``-1`` and the dynamic body (if any) is
    compacted to slot 0.

    Joint cids (``tid < num_joints``) read from the joint-wide
    :class:`ConstraintContainer`; contact cids read from the narrow
    :class:`ContactColumnContainer` at local offset
    ``tid - num_joints``. Launched at ``dim = constraint_capacity``
    because the active count is device-held; inactive lanes early-out.
    """
    tid = wp.tid()
    n = num_constraints[0]
    if tid >= n:
        return
    if tid < num_joints:
        b1 = constraint_get_body1(constraints, tid)
        b2 = constraint_get_body2(constraints, tid)
    else:
        local_cid = tid - num_joints
        b1 = contact_get_body1(contact_cols, local_cid)
        b2 = contact_get_body2(contact_cols, local_cid)
    if b1 >= 0 and bodies.inverse_mass[b1] == 0.0:
        b1 = -1
    if b2 >= 0 and bodies.inverse_mass[b2] == 0.0:
        b2 = -1
    # Compact: non-negative IDs must come first so the adjacency loop
    # (which stops on the first -1) doesn't miss a dynamic body when
    # the static one happens to sit in slot 0.
    if b1 < 0 and b2 >= 0:
        b1 = b2
        b2 = -1
    elements[tid] = element_interaction_data_make(b1, b2, -1, -1, -1, -1, -1, -1)


@wp.kernel(enable_backward=False)
def _constraint_gather_wrenches_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    num_joints: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    out: wp.array[wp.spatial_vector],
):
    """Per-cid world-frame wrench on ``body2``: ``top = force [N]``,
    ``bottom = torque [N·m]``. ``idt = 1 / substep_dt``."""
    cid = wp.tid()
    if cid >= num_constraints:
        return
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    if cid < num_joints:
        force, torque = actuated_double_ball_socket_world_wrench(constraints, cid, idt)
    else:
        force, torque = contact_world_wrench(contact_cols, cid - num_joints, bodies, idt, cc, contacts)
    out[cid] = wp.spatial_vector(force, torque)


@wp.kernel(enable_backward=False)
def _constraint_gather_errors_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    num_joints: wp.int32,
    # out
    out: wp.array[wp.spatial_vector],
):
    """Per-cid position-level constraint residual: ``top`` = linear
    [m], ``bottom`` = angular [rad]. Pure read from current body pose
    + persisted per-type state; no body mutation."""
    cid = wp.tid()
    if cid >= num_constraints:
        return
    zero = wp.spatial_vector(wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0))
    err = zero
    if cid < num_joints:
        err = actuated_double_ball_socket_world_error(constraints, cid, bodies)
    else:
        err = contact_world_error(contact_cols, cid - num_joints)
    out[cid] = err


@wp.func
def _rotation_quaternion(omega: wp.vec3f, dt: wp.float32) -> wp.quatf:
    """Axis-angle rotation quaternion for ``omega * dt``. Unit norm by
    construction, stable across many substeps."""
    omega_len = wp.length(omega)
    theta = omega_len * dt
    if theta < 1.0e-9:
        return wp.quatf(0.0, 0.0, 0.0, 1.0)
    half = theta * 0.5
    s = wp.sin(half) / omega_len
    return wp.quatf(omega[0] * s, omega[1] * s, omega[2] * s, wp.cos(half))


@wp.kernel(enable_backward=False)
def _integrate_velocities_kernel(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Advance position + orientation for dynamic bodies only.

    Static bodies skipped unconditionally. Kinematic bodies are
    *also* skipped here -- their pose advances via explicit lerp /
    slerp interpolation between ``position_prev`` and
    ``kinematic_target_pos`` in
    :func:`_kinematic_interpolate_substep_kernel`, so running the
    velocity integration on them would double-advance the pose.
    """
    i = wp.tid()
    mt = bodies.motion_type[i]
    if mt == MOTION_STATIC or mt == MOTION_KINEMATIC:
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt
    q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
    bodies.orientation[i] = wp.normalize(q_rot * bodies.orientation[i])


@wp.kernel(enable_backward=False)
def _kinematic_prepare_step_kernel(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Once-per-step kinematic prepare, called before the substep loop.

    For each kinematic body: resolve this step's pose target, infer
    the linear / angular velocity to expose to contacts, and
    snapshot the current pose into ``position_prev`` /
    ``orientation_prev`` as the per-substep lerp / slerp origin.

    Target resolution:

    * ``kinematic_target_valid[i] == 1`` -- read from
      ``kinematic_target_{pos,orient}`` and clear the flag.
    * ``0`` -- synthesise from ``position_prev + velocity * dt`` and
      axis-angle ``angular_velocity * dt`` (constant-velocity
      backward-compat).

    Velocity inference uses the quaternion log-map
    (``angle = 2 * atan2(|xyz|, w); omega = axis * angle / dt``) so
    large rotations are exact, not just small-angle.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_KINEMATIC:
        return

    pos_prev = bodies.position[i]
    orient_prev = bodies.orientation[i]
    bodies.position_prev[i] = pos_prev
    bodies.orientation_prev[i] = orient_prev

    if bodies.kinematic_target_valid[i] == 1:
        pos_target = bodies.kinematic_target_pos[i]
        orient_target = bodies.kinematic_target_orient[i]
        # One-shot consumption: the user must re-assert the target
        # each step (this matches the Newton adapter, which flags
        # valid=1 every import).
        bodies.kinematic_target_valid[i] = 0
    else:
        # Constant-velocity path: advance from ``(pos, orient)`` by
        # ``(velocity, angular_velocity) * dt``. Uses the same
        # axis-angle rotation the dynamic integrator uses so a
        # constant-omega kinematic traces exactly the same orientation
        # trajectory as the legacy code.
        pos_target = pos_prev + bodies.velocity[i] * dt
        q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
        orient_target = wp.normalize(q_rot * orient_prev)
        bodies.kinematic_target_pos[i] = pos_target
        bodies.kinematic_target_orient[i] = orient_target

    # Infer velocity from pose delta. For the constant-velocity path
    # this round-trips exactly (target = pos_prev + velocity * dt ->
    # inferred velocity == original velocity). For the scripted path
    # it exposes the pose-derivative for contact response.
    inv_dt = wp.float32(1.0) / dt
    v = (pos_target - pos_prev) * inv_dt

    # Angular velocity via log-map of ``q_rel = target * inv(prev)``.
    # Canonicalise to the shortest-path hemisphere first so the
    # ``atan2`` branch cut doesn't flip the sign.
    q_rel = orient_target * wp.quat_inverse(orient_prev)
    if q_rel[3] < 0.0:
        q_rel = -q_rel
    xyz = wp.vec3f(q_rel[0], q_rel[1], q_rel[2])
    xyz_len = wp.length(xyz)
    if xyz_len > 1.0e-9:
        angle = 2.0 * wp.atan2(xyz_len, q_rel[3])
        omega = xyz * (angle * inv_dt / xyz_len)
    else:
        omega = wp.vec3f(0.0, 0.0, 0.0)

    bodies.velocity[i] = v
    bodies.angular_velocity[i] = omega


@wp.kernel(enable_backward=False)
def _set_kinematic_pose_batch_kernel(
    bodies: BodyContainer,
    body_ids: wp.array[wp.int32],
    target_positions: wp.array[wp.vec3f],
    target_orientations: wp.array[wp.quatf],
):
    """Batched writeback for :meth:`PhoenXWorld.set_kinematic_pose`.

    One thread per entry in ``body_ids``. Writes the target pose into
    the kinematic-scripting slots and flags
    ``kinematic_target_valid = 1`` so the next
    :func:`_kinematic_prepare_step_kernel` picks it up.

    Attempting to script a non-kinematic body is a no-op (silent);
    callers should validate on the host side and raise clearly.
    """
    k = wp.tid()
    b = body_ids[k]
    if bodies.motion_type[b] != MOTION_KINEMATIC:
        return
    bodies.kinematic_target_pos[b] = target_positions[k]
    bodies.kinematic_target_orient[b] = target_orientations[k]
    bodies.kinematic_target_valid[b] = 1


@wp.kernel(enable_backward=False)
def _kinematic_interpolate_substep_kernel(
    bodies: BodyContainer,
    alpha: wp.float32,
):
    """Per-substep kinematic pose update.

    Called *after* :func:`_integrate_velocities_kernel` inside each
    substep with ``alpha = (substep_index + 1) / num_substeps``. Writes

    .. math::
        \\text{position}    &= \\text{lerp}(\\text{position}_{\\text{prev}},
                                 \\text{kinematic\\_target\\_pos}, \\alpha) \\\\
        \\text{orientation} &= \\text{slerp}(\\text{orientation}_{\\text{prev}},
                                 \\text{kinematic\\_target\\_orient}, \\alpha)

    At ``alpha = 1`` the body lands exactly on its target. Dynamic and
    static bodies are skipped (dynamic pose already advanced by
    :func:`_integrate_velocities_kernel`; static never moves).
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_KINEMATIC:
        return
    prev_pos = bodies.position_prev[i]
    target_pos = bodies.kinematic_target_pos[i]
    prev_orient = bodies.orientation_prev[i]
    target_orient = bodies.kinematic_target_orient[i]
    bodies.position[i] = (1.0 - alpha) * prev_pos + alpha * target_pos
    bodies.orientation[i] = wp.quat_slerp(prev_orient, target_orient, alpha)


@wp.kernel(enable_backward=False)
def pack_body_xforms_kernel(
    bodies: BodyContainer,
    xforms: wp.array[wp.transform],
):
    """Pack ``(position, orientation)`` into a flat ``wp.transform``
    array for ``viewer.log_shapes``."""
    i = wp.tid()
    xforms[i] = wp.transform(bodies.position[i], bodies.orientation[i])


# ---------------------------------------------------------------------------
# Per-step body kernels (forces + gravity, inertia refresh, force clear) and
# the on-device active-constraint count fuse. Driven from ``PhoenXWorld.step``.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _sync_num_active_constraints_kernel(
    num_contact_columns: wp.array[wp.int32],
    joint_constraint_count: wp.int32,
    # out
    num_active_constraints: wp.array[wp.int32],
):
    """``num_active_constraints = num_joints + num_contact_columns``,
    on-device. Single-thread; safe inside graph capture."""
    tid = wp.tid()
    if tid != 0:
        return
    num_active_constraints[0] = joint_constraint_count + num_contact_columns[0]


@wp.kernel(enable_backward=False)
def _phoenx_apply_forces_and_gravity_kernel(
    bodies: BodyContainer,
    gravity: wp.array[wp.vec3f],
    substep_dt: wp.float32,
):
    """Fused per-body velocity update at the top of every substep.

    Combines the old ``apply_external_forces`` and ``integrate_gravity``
    passes. Both kernels had the same dim / same gate / same per-body
    velocity read-modify-write, so running two launches instead of one
    only paid extra CUDA launch overhead (~4us each at 256 worlds, i.e.
    ~32us / frame at substeps=4). Force accumulators are NOT cleared
    here -- :func:`_phoenx_update_inertia_and_clear_forces_kernel`
    runs once at the end of :meth:`PhoenXWorld.step` and zeros them.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return
    if bodies.inverse_mass[i] == 0.0:
        return
    v = bodies.velocity[i]
    w = bodies.angular_velocity[i]
    inv_mass = bodies.inverse_mass[i]
    inv_inertia_world = bodies.inverse_inertia_world[i]
    v = v + bodies.force[i] * (inv_mass * substep_dt)
    w = w + (inv_inertia_world * bodies.torque[i]) * substep_dt
    if bodies.affected_by_gravity[i] != 0:
        v = v + gravity[bodies.world_id[i]] * substep_dt
    bodies.velocity[i] = v
    bodies.angular_velocity[i] = w


@wp.kernel(enable_backward=False)
def _phoenx_update_inertia_and_clear_forces_kernel(
    bodies: BodyContainer,
):
    """End-of-step per-body kernel: damping + rotated inertia refresh
    **plus** force/torque accumulator zeroing. Runs once per step,
    after the substep loop. Damping uses ``linear_damping`` /
    ``angular_damping`` per body; the world-frame inertia is rebuilt
    from the final orientation (``R * I^-1 * R^T``)."""
    i = wp.tid()
    # Damping + rotated inertia: dynamic-only.
    if bodies.motion_type[i] == MOTION_DYNAMIC:
        bodies.velocity[i] = bodies.velocity[i] * bodies.linear_damping[i]
        bodies.angular_velocity[i] = bodies.angular_velocity[i] * bodies.angular_damping[i]
        r = wp.quat_to_matrix(bodies.orientation[i])
        bodies.inverse_inertia_world[i] = rotate_inertia(r, bodies.inverse_inertia[i])
    # Force / torque clear: every body slot, including kinematic / static.
    bodies.force[i] = wp.vec3f(0.0, 0.0, 0.0)
    bodies.torque[i] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _phoenx_refresh_world_inertia_kernel(
    bodies: BodyContainer,
):
    """Per-substep ``inverse_inertia_world`` refresh.

    The constraint kernels read ``bodies.inverse_inertia_world[i]``
    for each body's effective-mass projection. After
    :func:`_integrate_velocities_kernel` rotates the body via
    ``q_new = dq(omega * substep_dt) * q_old``, the previously
    cached ``inverse_inertia_world`` no longer matches the current
    orientation. For anisotropic bodies (``I_xx != I_yy != I_zz`` --
    e.g. robot torsos) running multiple substeps without refreshing
    biases the angular-impulse projection in subsequent substeps,
    producing a small but cumulative drift in the angular momentum
    direction. This kernel rebuilds ``R * I^-1 * R^T`` from the
    current orientation so the next substep's solve uses the right
    rotated inertia.

    No damping, no force-clear -- those still run once per outer
    step in :func:`_phoenx_update_inertia_and_clear_forces_kernel`.
    """
    i = wp.tid()
    if bodies.motion_type[i] == MOTION_DYNAMIC:
        r = wp.quat_to_matrix(bodies.orientation[i])
        bodies.inverse_inertia_world[i] = rotate_inertia(r, bodies.inverse_inertia[i])


@wp.kernel(enable_backward=False)
def _phoenx_apply_global_damping_kernel(
    bodies: BodyContainer,
    global_damping: wp.array[wp.float32],
):
    """Per-substep global damping for dynamic bodies.

    ``global_damping`` is a length-2 device array: ``[0]`` = linear,
    ``[1]`` = angular. Applies ``v *= 1 - global_damping[0]`` and
    ``w *= 1 - global_damping[1]``. Default values of ``0`` are a
    no-op; ``1`` zeroes the corresponding velocity component every
    substep (Kapla-style settle warm-up). Stored in a device array so
    the host can rewrite it between graph replays without re-capture.
    """
    i = wp.tid()
    if bodies.motion_type[i] == MOTION_DYNAMIC:
        lin = 1.0 - global_damping[0]
        ang = 1.0 - global_damping[1]
        bodies.velocity[i] = bodies.velocity[i] * lin
        bodies.angular_velocity[i] = bodies.angular_velocity[i] * ang


# ---------------------------------------------------------------------------
# Single-world step path: per-colour grid launches via ``wp.capture_while``
# ---------------------------------------------------------------------------
#
# Persistent grid walks the partitioner's GLOBAL CSR colour-by-colour,
# driven by a host-side ``wp.capture_while`` on ``head_active``. One
# launch per colour; launch dim sized once at construction (see
# :attr:`PhoenXWorld._singleworld_total_threads`) to a multiple of
# SM count, same strategy as :class:`NarrowPhase`. Wins for one or a
# few big worlds; the multi-world fast-tail path leaves SMs idle there.
#
# Head capture-while termination -- ``head_active[0]`` starts at 1 and
# the kernel clears it in two cases:
#
#   (a) ``color_cursor[0] <= 0``  -- sweep drained every colour.
#   (b) ``count <= fuse_threshold`` -- hand off to the fused tail
#       kernel, which resumes at the same cursor.
#
# Neither case touches ``color_cursor``; the dedicated ``head_active``
# flag is what lets the tail kernel pick up where (b) stopped. During
# normal work thread 0 decrements ``color_cursor`` at end-of-kernel.
# ``fuse_threshold = 0`` disables (b) and preserves the original
# one-kernel-per-colour behaviour.
#
# Early-exit contract (required for ``NUM_INNER_WHILE_ITERATIONS > 1``):
# launches after the one that clears ``head_active`` still fire; they
# re-enter the early-exit branches as cheap no-ops until the outer
# capture-while observes ``head_active[0] == 0``.


@wp.func
def _singleworld_color_range(
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    color_cursor: wp.array[wp.int32],
):
    """Decode the current colour's cid range from the cursor.

    Returns ``(start, count, cursor)``: ``start`` is the index into
    ``element_ids_by_color`` of the colour's first cid; ``count`` is
    how many cids belong to it; ``cursor`` is the snapshot of
    ``color_cursor[0]`` for the cursor-decrement at end of kernel.
    """
    cursor = color_cursor[0]
    n_colors = num_colors[0]
    c = n_colors - cursor
    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start
    return start, count, cursor


# Persistent-grid head kernels are factory-generated below
# (:func:`_make_singleworld_persistent_kernel`); see
# :data:`_constraint_prepare_singleworld_kernel` etc. for the bound
# names. Same shape as the previous hand-written kernels, with one
# extra ``num_sweeps`` parameter that drives the per-cid multi-sweep
# register cache via ``*_iterate_multi``.


# ---------------------------------------------------------------------------
# Single-world step path: fused tail kernels (one block, many colours)
# ---------------------------------------------------------------------------
#
# Single 1D-block kernel that walks the sweep's trailing small colours
# back-to-back, using ``__syncthreads`` in place of the per-colour
# kernel boundary. Launched via ``wp.launch_tiled(dim=[1],
# block_dim=FUSE_TAIL_BLOCK_DIM)``.
#
# Correctness:
#
# * Each iteration of the internal ``while`` handles one colour of
#   size ``<= FUSE_TAIL_MAX_COLOR_SIZE`` so every cid owns a distinct
#   lane. ``_sync_threads`` orders each colour's body-velocity writes
#   before the next colour's reads; the coloring invariant already
#   rules out intra-colour body-sharing races.
# * On encountering a colour with size > ``FUSE_TAIL_MAX_COLOR_SIZE``
#   the kernel exits WITHOUT decrementing ``color_cursor``, handing
#   off to the persistent-grid kernel. That kernel mirrors the
#   hand-off from the other direction (exits on
#   ``count <= fuse_threshold``), so the two kernels partition the
#   colour sequence by size.
# * The outer ``wp.capture_while(color_cursor, ...)`` terminates
#   when the fused kernel drains the last small colour.


@wp.kernel(enable_backward=False)
def _reset_head_active_kernel(head_active: wp.array[wp.int32]):
    """Reset ``head_active[0] = 1`` before a single-world head sweep.

    Called once before each ``wp.capture_while(head_active, ...)`` so
    the predicate starts truthy and the head kernel gets at least one
    launch in which to decide (converge, hand off, or do real work).
    """
    head_active[0] = 1


@wp.func
def _singleworld_color_range_from_cursor(
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    cursor: wp.int32,
):
    """Variant of :func:`_singleworld_color_range` that takes a cursor
    value directly instead of reading it from an array.

    Used by the fused tail kernel which threads the cursor through
    registers across the internal colour loop so every lane observes
    the same value without re-reading global memory.
    """
    n_colors = num_colors[0]
    c = n_colors - cursor
    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start
    return start, count


# Tail-fused (one-block) variants are likewise factory-generated below
# via :func:`_make_singleworld_fused_kernel`; the bound names retain
# the original ``_constraint_*_singleworld_fused_kernel`` symbols so
# the launch site doesn't need to change.


# ---------------------------------------------------------------------------
# Single-world kernel factories
# ---------------------------------------------------------------------------
#
# The persistent-grid (head) and single-block (fused tail) kernels for
# prepare / iterate / relax all share the same shape: they walk the
# colour CSR, then per cid dispatch on ``cid < num_joints`` to either
# the joint code or the contact code. The only axes of variation are:
#
#   * **phase**: prepare vs iterate vs relax. iterate uses
#     ``use_bias=True``; relax uses ``False``; prepare has no bias.
#   * **joint specialisation**: when every joint is revolute (or there
#     are no joints), the kernel can call :func:`revolute_iterate` /
#     :func:`revolute_prepare_for_iteration` directly, skipping the
#     ``read_int(_OFF_JOINT_MODE)`` global load and the four-way
#     ``joint_mode`` branch that
#     :func:`actuated_double_ball_socket_iterate` carries.
#
# Both axes are compile-time. Closing the kernel over Python booleans
# lets Warp constant-fold the variant selection at codegen and dead-
# code-eliminate the unused branch / unused inlined function bodies.
# Net: 2 factories produce 12 kernels (3 phases x 2 specialisations x
# 2 shapes) without copy-paste.


def _make_singleworld_persistent_kernel(*, phase: str, revolute_only: bool):
    """Build a persistent-grid PGS kernel for the requested phase /
    specialisation. ``phase`` is ``"prepare"``, ``"iterate"`` or
    ``"relax"``; ``revolute_only`` selects the joint dispatch path.

    Single-world stays on the single-sweep ``*_iterate`` /
    ``*_prepare_for_iteration`` helpers (no ``num_sweeps`` parameter):
    the multi-sweep register cache is the multi-world fast-tail's
    win and on contact-heavy single-world scenes (kapla) it shows a
    small net regression.
    """
    is_prepare = phase == "prepare"
    is_iterate = phase == "iterate"
    use_bias = is_iterate  # iterate ON, relax OFF (prepare ignores)

    @wp.kernel(enable_backward=False)
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        element_ids_by_color: wp.array[wp.int32],
        color_starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        color_cursor: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_joints: wp.int32,
        total_num_threads: wp.int32,
        fuse_threshold: wp.int32,
        head_active: wp.array[wp.int32],
    ):
        tid = wp.tid()
        if color_cursor[0] <= 0:
            if tid == 0:
                head_active[0] = 0
            return
        start, count, cursor = _singleworld_color_range(color_starts, num_colors, color_cursor)

        if count <= fuse_threshold:
            if tid == 0:
                head_active[0] = 0
            return

        for t in range(tid, count, total_num_threads):
            cid = element_ids_by_color[start + t]
            if cid < num_joints:
                if wp.static(is_prepare):
                    if wp.static(revolute_only):
                        revolute_prepare_for_iteration(constraints, cid, bodies, idt)
                    else:
                        actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
                else:
                    if wp.static(revolute_only):
                        revolute_iterate(constraints, cid, bodies, idt, use_bias)
                    else:
                        actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, use_bias)
            else:
                if wp.static(is_prepare):
                    contact_prepare_for_iteration(contact_cols, cid - num_joints, bodies, idt, cc, contacts)
                else:
                    contact_iterate(contact_cols, cid - num_joints, bodies, idt, cc, contacts, use_bias)

        if tid == 0:
            color_cursor[0] = cursor - 1

    return kernel


def _make_singleworld_fused_kernel(*, phase: str, revolute_only: bool):
    """Build a single-block tail-fused PGS kernel for the requested
    phase / specialisation. Same axes as
    :func:`_make_singleworld_persistent_kernel`."""
    is_prepare = phase == "prepare"
    is_iterate = phase == "iterate"
    use_bias = is_iterate

    @wp.kernel(enable_backward=False)
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        element_ids_by_color: wp.array[wp.int32],
        color_starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        color_cursor: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_joints: wp.int32,
        fuse_threshold: wp.int32,
    ):
        _block, lane = wp.tid()
        cursor = color_cursor[0]
        while cursor > 0:
            start, count = _singleworld_color_range_from_cursor(color_starts, num_colors, cursor)
            if count > fuse_threshold:
                break
            if lane < count:
                cid = element_ids_by_color[start + lane]
                if cid < num_joints:
                    if wp.static(is_prepare):
                        if wp.static(revolute_only):
                            revolute_prepare_for_iteration(constraints, cid, bodies, idt)
                        else:
                            actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
                    else:
                        if wp.static(revolute_only):
                            revolute_iterate(constraints, cid, bodies, idt, use_bias)
                        else:
                            actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, use_bias)
                else:
                    if wp.static(is_prepare):
                        contact_prepare_for_iteration(contact_cols, cid - num_joints, bodies, idt, cc, contacts)
                    else:
                        contact_iterate(contact_cols, cid - num_joints, bodies, idt, cc, contacts, use_bias)
            _sync_threads()
            cursor = cursor - 1
        if lane == 0:
            color_cursor[0] = cursor

    return kernel


_constraint_prepare_singleworld_kernel = _make_singleworld_persistent_kernel(
    phase="prepare", revolute_only=False
)
_constraint_iterate_singleworld_kernel = _make_singleworld_persistent_kernel(
    phase="iterate", revolute_only=False
)
_constraint_relax_singleworld_kernel = _make_singleworld_persistent_kernel(
    phase="relax", revolute_only=False
)
_constraint_prepare_singleworld_fused_kernel = _make_singleworld_fused_kernel(
    phase="prepare", revolute_only=False
)
_constraint_iterate_singleworld_fused_kernel = _make_singleworld_fused_kernel(
    phase="iterate", revolute_only=False
)
_constraint_relax_singleworld_fused_kernel = _make_singleworld_fused_kernel(
    phase="relax", revolute_only=False
)
_constraint_prepare_singleworld_revolute_kernel = _make_singleworld_persistent_kernel(
    phase="prepare", revolute_only=True
)
_constraint_iterate_singleworld_revolute_kernel = _make_singleworld_persistent_kernel(
    phase="iterate", revolute_only=True
)
_constraint_relax_singleworld_revolute_kernel = _make_singleworld_persistent_kernel(
    phase="relax", revolute_only=True
)
_constraint_prepare_singleworld_fused_revolute_kernel = _make_singleworld_fused_kernel(
    phase="prepare", revolute_only=True
)
_constraint_iterate_singleworld_fused_revolute_kernel = _make_singleworld_fused_kernel(
    phase="iterate", revolute_only=True
)
_constraint_relax_singleworld_fused_revolute_kernel = _make_singleworld_fused_kernel(
    phase="relax", revolute_only=True
)
