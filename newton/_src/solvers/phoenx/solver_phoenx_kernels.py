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
    "_constraint_iterate_singleworld_kernel",
    "_constraint_prepare_plus_iterate_fast_tail_kernel",
    "_constraint_prepare_singleworld_kernel",
    "_constraint_relax_fast_tail_kernel",
    "_constraint_relax_singleworld_kernel",
    "_constraints_to_elements_kernel",
    "_count_elements_per_world_kernel",
    "_integrate_velocities_kernel",
    "_kinematic_interpolate_substep_kernel",
    "_kinematic_prepare_step_kernel",
    "_per_world_jp_coloring_kernel",
    "_phoenx_apply_forces_and_gravity_kernel",
    "_phoenx_clear_forces_kernel",
    "_phoenx_update_inertia_kernel",
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

# How many PGS sweeps ``contact_iterate_multi`` /
# ``actuated_double_ball_socket_iterate_multi`` run per call. Every
# value saves per-cid body / constraint reloads amortised inside the
# multi function, but values > 1 also *shrink* the cross-colour PGS
# feedback from ``solver_iterations`` rounds down to
# ``solver_iterations / _FUSED_INNER_SWEEPS`` rounds -- which tall
# stacks cannot tolerate.
#
# ``2`` (4 feedback rounds at the default ``solver_iterations = 8``)
# passes every short-stack / pyramid / hinge test in the suite but
# *fails* ``test_example_tower`` (40 layers of circular planks): the
# top ring drops ~1 m in a 1 s settle because the ground reaction
# cannot propagate through 40 contact layers in 4 rounds.
#
# ``1`` keeps full 8-round feedback (math-identical to the pre-multi
# outer loop) but still exercises the register-caching infrastructure
# so the compiler can keep body / constraint constants in registers
# across the PGS row solves *within* a single sweep. Every test passes
# and the large-world gains are still significant (+34%/+57% vs the
# pre-optimisation baseline at 4096 worlds for g1/h1_flat).
# ``_FUSED_INNER_SWEEPS`` must evenly divide ``solver_iterations``.
_FUSED_INNER_SWEEPS: int = 1


def _choose_fast_tail_worlds_per_block(num_worlds: int) -> int:
    """How many worlds to cohabit one physical block in the fast-tail
    kernels. Each world always owns exactly one warp (32 threads), so
    the returned block size ``32 * wpb`` keeps ``__syncwarp()`` valid
    regardless of ``wpb``.

    Tuned on RTX PRO 6000 (sm_120), 188 SMs, after the fused
    prepare+iterate + register-caching refactor.

    Three-tier by scene size (env_fps numbers are 3-run averaged
    across g1_flat / h1_flat):

    * ``num_worlds < 512``: ``wpb = 2``. Keeps blocks spread across
      SMs. ``wpb = 4`` already regresses 5-15% here, ``wpb = 8`` is
      20-25% slower (blocks below 1 per SM).
    * ``512 <= num_worlds < 2048``: ``wpb = 4``. ``wpb = 2`` is
      slightly better on h1 (+3%) but worse on g1 (+8% with ``wpb = 4``);
      ``wpb = 4`` is the cross-scenario middle ground.
    * ``num_worlds >= 2048``: ``wpb = 8``. g1 @ 4096 picks up +20%
      vs ``wpb = 4`` (more ILP per block absorbs the register
      pressure from the register-cached revolute iterate); h1 @ 4096
      loses ~4% but h1 @ 16384 is neutral. Average across both
      scenarios is the better call.
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
# The fast-tail kernels launch with a fixed grid of ``num_worlds *
# _STRAGGLER_BLOCK_DIM`` threads (one warp slot per world at maximum
# tpw=32). Each kernel reads its effective tpw from a 1-element buffer
# written by :func:`_pick_threads_per_world_kernel` once per step. With
# tpw=32 every thread does work; with tpw=16 lanes 16-31 of each warp
# slot map to ``world_id >= num_worlds`` and early-exit; with tpw=8 the
# top 24 lanes early-exit. This lets a sparse-color scene (h1_flat:
# mean ~5 cids/colour/world) double its lane utilisation by packing two
# worlds per warp -- without changing the host-side launch shape, so
# the whole step still fits in a single CUDA graph.


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

    Two-tier heuristic, tuned on RTX PRO 6000 (sm_count=188) across
    h1_flat at 64..16384 worlds, tower at 32..256 worlds, and
    g1_flat at 64..2048 worlds:

      * **tpw=32 (default, one warp per world).** Used whenever the
        scene fills < ~8 warps per SM at tpw=32 (small / medium
        fleets where SM occupancy is the bottleneck and dropping
        warp count regresses), or when colours are dense enough that
        most of the warp's 32 lanes already do useful work
        (mean >= 6 cids/colour).
      * **tpw=16 (two worlds per warp).** Both gates must hit:
        enough warps per SM to absorb halving (>= 8 warps/SM at
        tpw=32) AND sparse-enough colours that two worlds' worth of
        cids still fit comfortably under the warp roof
        (mean <= 6 cids/colour). End-to-end gain ~+4-12% on
        h1_flat-class fleets above 2048 worlds.

    ``tpw=8`` is reachable via the static ``threads_per_world=8``
    constructor argument but the auto picker never emits it: empirical
    data shows tpw=8 wins occasionally at extreme world counts but
    the gap over tpw=16 is dwarfed by run-to-run noise, while
    incorrectly picking tpw=8 over tpw=16 (e.g. h1_8192, where
    tpw=16 saved +12% but tpw=8 only +3%) is a noticeable regression.

    All comparisons in fixed-point (x16) so the kernel runs without
    floating-point ops. Inputs come from the scan / reduction
    kernels that ran before this one, so the picker itself is O(1)
    and stays cheap inside the captured graph.
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
def _biased_priority(
    random_values: wp.array[wp.int32],
    cid: wp.int32,
    contact_cid_start: wp.int32,
    contact_bias: wp.int32,
) -> wp.int32:
    """Return JP priority with an on-the-fly bias that lifts contact cids
    above every joint cid. ``contact_cid_start`` is the first contact cid
    (joints live below). ``contact_bias`` is the amount to add -- typically
    ``max_num_interactions`` so the biased contact range sits strictly
    above the unbiased joint range. Mirrors the section-marker trick used
    in the C# PhoenX partitioning kernel, without the extra storage."""
    r = random_values[cid]
    if cid >= contact_cid_start:
        r = r + contact_bias
    return r


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
    max_colors: wp.int32,
    # ``cid >= contact_cid_start`` gets ``contact_bias`` added to its
    # priority, so contacts commit before joints in every JP round. This
    # clusters contacts into earlier colours, cutting intra-warp
    # divergence in the fast-tail constraint iterate kernel (contacts and
    # joints take different branches). When joints are absent
    # ``contact_cid_start = capacity`` disables the bias.
    contact_cid_start: wp.int32,
    contact_bias: wp.int32,
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
        # Phase 2: find local maxima and commit them. Every lane runs
        # the stride loop the same number of times so that the
        # ``wp.tile`` reduction / scan below sees all lanes at the
        # same point regardless of count -- Warp tile ops are
        # block-collective.
        #
        # Slot assignment within a colour is via ``tile_scan_exclusive``
        # rather than an atomic cursor: the scan is deterministic
        # (depends only on per-lane ``committed_here``, which is a
        # function of the priority graph), where atomic_add ordering
        # depends on which lane reaches the atomic first. The tile
        # primitives carry the same per-step cost.
        committed_this_round = wp.int32(0)
        offset = wp.int32(0)
        while offset < count:
            slot = offset + lane
            committed_here = wp.int32(0)
            committed_eid = wp.int32(0)
            if slot < count:
                eid = world_elements[base + slot]
                if assigned[eid] == wp.int32(0):
                    self_prio = _biased_priority(random_values, eid, contact_cid_start, contact_bias)
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
                            if _biased_priority(random_values, neighbor, contact_cid_start, contact_bias) > self_prio:
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


# ---------------------------------------------------------------------------
# Fast-path unified single-block dispatchers
# ---------------------------------------------------------------------------
#
# One block per world; each block walks its world's full CSR internally.
# Inside each block: outer iteration loop, middle colour sweep with
# ``__syncthreads`` between colours, inner block-stride lane loop.
# Partitioner guarantees no two same-colour elements share a body, so
# per-lane RMW on body velocities is race-free.


@wp.kernel(enable_backward=False)
def _constraint_prepare_plus_iterate_fast_tail_kernel(
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
    """Fused prepare + main-solve dispatcher.

    Runs the prepare pass and the ``num_iterations`` PGS sweeps in a
    single kernel launch. Saves one kernel launch per substep; more
    importantly, per-world setup (``world_id``, ``n_colors``,
    ``world_base``) is computed once and the register-resident
    per-world state survives across the prepare -> iterate transition
    for free.

    ``tpw_buf[0]`` holds the effective threads-per-world chosen by
    :func:`_pick_threads_per_world_kernel`. Threads with
    ``world_id >= num_worlds`` early-exit; this is how the launch grid
    of ``num_worlds * _STRAGGLER_BLOCK_DIM`` retracts to ``num_worlds *
    tpw`` lanes when ``tpw < _STRAGGLER_BLOCK_DIM``.

    The prepare outputs (``cc.derived``) still round-trip through
    global memory between prepare and iterate -- different cids in a
    warp touch different contacts, so we can't keep them in
    registers. But the kernel-launch fixed cost is eliminated and
    some loop-boundary scalars may be reused by the compiler.
    """
    tid = wp.tid()
    tpw = tpw_buf[0]
    local_tid = tid % tpw
    world_id = tid / tpw
    if world_id >= num_worlds:
        return

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    # ---- Prepare phase --------------------------------------------
    c = wp.int32(0)
    while c < n_colors:
        start = world_base + world_color_starts[world_id, c]
        end = world_base + world_color_starts[world_id, c + 1]
        count = end - start

        base = local_tid
        while base < count:
            cid = world_element_ids_by_color[start + base]
            if cid < num_joints:
                actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
            else:
                contact_prepare_for_iteration(contact_cols, cid - num_joints, bodies, idt, cc, contacts)
            base += tpw

        _sync_warp()
        c += 1

    # ---- Iterate phase -------------------------------------------
    # Split ``num_iterations`` into
    # ``outer_iters = num_iterations / _FUSED_INNER_SWEEPS`` outer
    # rounds of cross-colour PGS feedback, each running
    # ``_FUSED_INNER_SWEEPS`` sweeps on every cid with body state and
    # per-cid constraint constants held in registers (``*_iterate_multi``
    # functions). For revolute joints (the common G1 / H1 case) and
    # contacts, the register-resident path eliminates
    # ``(_FUSED_INNER_SWEEPS - 1)`` redundant body / constraint reads
    # per cid per outer iteration.
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
                    actuated_double_ball_socket_iterate_multi(constraints, cid, bodies, idt, True, inner_sweeps)
                else:
                    contact_iterate_multi(contact_cols, cid - num_joints, bodies, idt, cc, contacts, True, inner_sweeps)
                base += tpw

            _sync_warp()
            c += 1

        it_outer += 1


@wp.kernel(enable_backward=False)
def _constraint_relax_fast_tail_kernel(
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
    """Box2D v3 TGS-soft relax dispatcher: ``num_iterations`` sweeps
    with positional bias OFF (enforces ``Jv = 0``).

    See :func:`_constraint_prepare_plus_iterate_fast_tail_kernel` for
    the ``tpw_buf`` adaptive-launch convention.
    """
    tid = wp.tid()
    tpw = tpw_buf[0]
    local_tid = tid % tpw
    world_id = tid / tpw
    if world_id >= num_worlds:
        return

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    # Dispatch via the register-cached ``*_iterate_multi`` path with
    # ``num_sweeps = num_iterations`` so each cid gets one body +
    # constraint data load amortised over the whole relax sweep.
    # ``velocity_iterations`` is typically 1 so there's no real
    # cross-colour feedback to preserve -- running the full relax in
    # one multi call is equivalent to the classic outer-inner split.
    c = wp.int32(0)
    while c < n_colors:
        start = world_base + world_color_starts[world_id, c]
        end = world_base + world_color_starts[world_id, c + 1]
        count = end - start

        base = local_tid
        while base < count:
            cid = world_element_ids_by_color[start + base]
            if cid < num_joints:
                actuated_double_ball_socket_iterate_multi(constraints, cid, bodies, idt, False, num_iterations)
            else:
                contact_iterate_multi(contact_cols, cid - num_joints, bodies, idt, cc, contacts, False, num_iterations)
            base += tpw

        _sync_warp()
        c += 1


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
    """Once-per-step kinematic prepare, called at
    :meth:`PhoenXWorld.step` entry *before* the substep loop.

    Resolves this step's pose target for every kinematic body, infers
    the linear + angular velocity the solver should expose to contacts,
    and snapshots the body's current pose as ``position_prev`` /
    ``orientation_prev`` so the per-substep interpolator has a stable
    ``lerp`` / ``slerp`` origin.

    Target resolution:

    * ``kinematic_target_valid[i] == 1`` (user called
      :meth:`PhoenXWorld.set_kinematic_pose` or the Newton adapter
      flagged a pose import) -- read the user-set target out of
      ``kinematic_target_{pos,orient}`` and clear the flag.
    * ``kinematic_target_valid[i] == 0`` (no explicit script this
      step, constant-velocity backward-compat path) -- synthesise
      a target from ``position_prev + velocity * dt`` and the
      axis-angle integration of ``angular_velocity * dt``.

    Velocity inference uses the quaternion log-map so large
    rotations are handled correctly (small-angle ``omega ~= 2 *
    q_rel.xyz / dt`` is exact only at the limit; the full formula
    ``angle = 2 * atan2(|xyz|, w); omega = axis * angle / dt``
    generalises without drift).
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
    here -- :func:`_phoenx_clear_forces_kernel` runs once at the end
    of :meth:`PhoenXWorld.step`.
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
def _phoenx_update_inertia_kernel(
    bodies: BodyContainer,
):
    """Apply linear/angular damping and refresh
    ``inverse_inertia_world = R * I^-1 * R^T`` from the current
    orientation. Once per step, after the substep loop."""
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return
    bodies.velocity[i] = bodies.velocity[i] * bodies.linear_damping[i]
    bodies.angular_velocity[i] = bodies.angular_velocity[i] * bodies.angular_damping[i]
    r = wp.quat_to_matrix(bodies.orientation[i])
    bodies.inverse_inertia_world[i] = rotate_inertia(r, bodies.inverse_inertia[i])


@wp.kernel(enable_backward=False)
def _phoenx_clear_forces_kernel(
    bodies: BodyContainer,
):
    """Zero per-body force/torque accumulators. Runs once at the end
    of :meth:`PhoenXWorld.step`."""
    i = wp.tid()
    bodies.force[i] = wp.vec3f(0.0, 0.0, 0.0)
    bodies.torque[i] = wp.vec3f(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Single-world step path: per-colour grid launches via ``wp.capture_while``
# ---------------------------------------------------------------------------
#
# These kernels treat the partitioner's GLOBAL CSR (``element_ids_by_color``
# / ``color_starts`` / ``num_colors``) as a single coloured graph and
# drive the colour walk via a host-side ``wp.capture_while`` loop on
# ``color_cursor``. One grid launch per colour; each launch is a
# *persistent* grid of fixed size and uses an internal grid-stride loop
# to cover the current colour's active cid range. Wins when the scene
# has one big world (or a small number of large worlds) -- the
# multi-world fast-tail kernels pin one warp per world and leave most
# SMs idle in that regime.
#
# The launch dim is sized once per :class:`PhoenXWorld` at construction
# (see :attr:`PhoenXWorld._singleworld_total_threads`) to a reasonable
# multiple of the device's SM count, following the same strategy as
# :class:`newton._src.geometry.narrow_phase.NarrowPhase`. Launching one
# thread per ``constraint_capacity`` element used to produce ~900K-thread
# grids on Kapla-scale scenes where a single colour has only a few
# hundred active cids -- massive overprovisioning that tanked occupancy
# (168 reg/thread * 256-wide blocks at 16% theoretical occupancy).
#
# Usage from the host:
#
#     partitioner.begin_sweep()
#     wp.capture_while(
#         partitioner.color_cursor,
#         lambda: [wp.launch(_constraint_prepare_singleworld_kernel,
#                            dim=total_num_threads, ...,
#                            total_num_threads=total_num_threads)
#                  for _ in range(NUM_INNER_WHILE_ITERATIONS)],
#     )
#
# The kernel decrements ``color_cursor`` by 1 at the end (thread 0
# only); when it reaches 0 the capture-while loop exits.
#
# Early-exit contract (required for ``NUM_INNER_WHILE_ITERATIONS`` > 1):
# every kernel checks ``color_cursor[0] <= 0`` at the top and returns
# immediately without touching any state, so the tail launches inside
# the capture-while body -- which the host issues unconditionally --
# become cheap no-ops once the sweep has converged within the same
# outer iteration. That is what makes it safe to amortise the
# capture-while overhead by unrolling the body ``NUM_INNER_WHILE_ITERATIONS``
# times.


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


@wp.kernel(enable_backward=False)
def _constraint_prepare_singleworld_kernel(
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
):
    """Prepare-for-iteration dispatcher, single-world / one-colour-per-launch.

    Launched as a persistent grid of ``total_num_threads`` threads that
    grid-strides over the current colour's active cid range. Thread 0
    decrements ``color_cursor`` by 1 so the host-side
    :func:`wp.capture_while` exits when every colour has been swept.

    Early-exits without touching state when ``color_cursor[0] <= 0`` so
    tail launches past convergence (from the inner unrolling inside the
    capture-while body) are no-ops.
    """
    if color_cursor[0] <= 0:
        return
    tid = wp.tid()
    start, count, cursor = _singleworld_color_range(color_starts, num_colors, color_cursor)

    for t in range(tid, count, total_num_threads):
        cid = element_ids_by_color[start + t]
        if cid < num_joints:
            actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
        else:
            contact_prepare_for_iteration(contact_cols, cid - num_joints, bodies, idt, cc, contacts)

    if tid == 0:
        color_cursor[0] = cursor - 1


@wp.kernel(enable_backward=False)
def _constraint_iterate_singleworld_kernel(
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
):
    """Main-solve PGS dispatcher (bias ON), one colour per launch.

    Same launch contract as :func:`_constraint_prepare_singleworld_kernel`;
    this is the iterate counterpart with ``use_bias=True``. Inherits the
    same early-exit contract.
    """
    if color_cursor[0] <= 0:
        return
    tid = wp.tid()
    start, count, cursor = _singleworld_color_range(color_starts, num_colors, color_cursor)

    for t in range(tid, count, total_num_threads):
        cid = element_ids_by_color[start + t]
        if cid < num_joints:
            actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, True)
        else:
            contact_iterate(contact_cols, cid - num_joints, bodies, idt, cc, contacts, True)

    if tid == 0:
        color_cursor[0] = cursor - 1


@wp.kernel(enable_backward=False)
def _constraint_relax_singleworld_kernel(
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
):
    """Relax-pass dispatcher (bias OFF), one colour per launch.

    Inherits the early-exit contract from
    :func:`_constraint_prepare_singleworld_kernel`.
    """
    if color_cursor[0] <= 0:
        return
    tid = wp.tid()
    start, count, cursor = _singleworld_color_range(color_starts, num_colors, color_cursor)

    for t in range(tid, count, total_num_threads):
        cid = element_ids_by_color[start + t]
        if cid < num_joints:
            actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, False)
        else:
            contact_iterate(contact_cols, cid - num_joints, bodies, idt, cc, contacts, False)

    if tid == 0:
        color_cursor[0] = cursor - 1
