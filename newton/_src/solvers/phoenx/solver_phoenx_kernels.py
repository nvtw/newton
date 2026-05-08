# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for :class:`PhoenXWorld`. Dispatches only ADBS and CONTACT."""

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
    _lowest_set_bit,
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


#: Max threads-per-world for fast-tail kernels (= warp size). The grid is
#: always num_worlds * _STRAGGLER_BLOCK_DIM; surplus threads early-exit.
_STRAGGLER_BLOCK_DIM: int = 32

# PGS sweeps per *_iterate_multi call. Must evenly divide solver_iterations.
# 2 amortises body/constraint reloads at +17-21% on g1_flat/h1_flat without
# breaking stacking/articulation tests; 4 halves outer rounds further but
# breaks test_slam_ball_into_stack.
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


#: Upper bound on fast-tail block size. Lets callers bound padded launch dim
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


# Adaptive threads-per-world picker. Fast-tail grid is fixed; effective tpw
# read from a 1-elem buffer per step. Smaller tpw early-exits surplus lanes.


@wp.kernel(enable_backward=False)
def _reduce_total_colours_kernel(
    world_num_colors: wp.array[wp.int32],
    num_worlds: wp.int32,
    # out
    total_colours: wp.array[wp.int32],
):
    """Atomic-sum world_num_colors into a 1-elem scalar. Caller must zero ``total_colours``."""
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
    """One-thread pick of tpw in {16, 32}. tpw=16 wins when warps/SM >= 8 AND
    mean cids/colour <= 6 (sparse colours, saturated SMs); else tpw=32.
    Auto picker never emits tpw=8 (the static arg can)."""
    if wp.tid() != 0:
        return

    total_cids = world_csr_offsets[num_worlds]
    nc = total_colours[0]

    if nc <= 0 or num_worlds <= 0:
        tpw_choice[0] = 32
        return

    # Fixed-point x16 so thresholds stay int32.
    mean_x16 = (total_cids * wp.int32(16)) / nc
    warps_at_tpw32 = num_worlds  # 1 warp/world at tpw=32
    saturation_x16 = (warps_at_tpw32 * wp.int32(16)) / wp.max(sm_count, wp.int32(1))

    pick = wp.int32(32)
    if mean_x16 <= wp.int32(6 * 16) and saturation_x16 >= wp.int32(8 * 16):
        pick = wp.int32(16)

    tpw_choice[0] = pick


# Per-world JP MIS coloring: worlds are independent (static-body nullification),
# one block per world, output goes straight to per-world CSR.


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
    """Atomic per-world element count. Writes raw count + shifted form so a
    single inclusive scan produces (exclusive prefix, total)."""
    tid = wp.tid()
    n = num_elements[0]
    if tid == wp.int32(0):
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
    """(key=world_id, value=cid) pairs for the per-world scatter sort.
    Inactive/tail entries get key=INT32_MAX so they sort to the end."""
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
    """JP MIS coloring per world (one block per world). Each round picks
    local-priority maxima and commits them as the next colour, writing into
    world_element_ids_by_color via a tile-scan exclusive prefix."""
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


# All-ones int64; flips a forbidden-color mask without unary NOT (Warp's int64
# codegen is unreliable). Mirrors _FREE_COLOR_FLIP in graph_coloring_common.py.
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
    """JP-MIS + smallest-free-color (greedy) per world. One block/world.
    Replaces the JP round-equals-colour scatter with histogram + prefix scan +
    atomic scatter (intra-block, no cross-block sync). Within-colour element
    order is non-deterministic but irrelevant (PGS treats colours as sets)."""
    block, lane = wp.tid()
    w = block
    base = world_element_offsets[w]
    count = world_element_count[w]

    if count == 0:
        if lane == wp.int32(0):
            world_num_colors[w] = wp.int32(0)
            world_color_starts[w, 0] = wp.int32(0)
        return

    # Reset per-element assigned + per-world histogram/cursor buckets.
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

    # Greedy MIS+colour rounds. Outer loop hard-capped at ``count`` for safety.
    num_remaining = count
    overflow_local = wp.int32(0)
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
                                # Uncoloured: MIS tiebreak.
                                if _cost_biased_priority(random_values, cost_values, neighbor) > self_prio:
                                    is_local_max = bool(False)
                            else:
                                # Coloured: forbid that colour.
                                ncolor = a - wp.int32(1)
                                if ncolor < GREEDY_MAX_COLORS:
                                    forbidden_mask = forbidden_mask | (wp.int64(1) << wp.int64(ncolor))

                    if is_local_max:
                        # Saturated mask -> c < 0; treat as overflow (don't write
                        # assigned[eid]=0 or fire an OOB atomic_add).
                        free_mask = forbidden_mask ^ _PER_WORLD_FREE_COLOR_FLIP
                        c = _lowest_set_bit(free_mask)
                        if c < wp.int32(0) or c >= GREEDY_MAX_COLORS:
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
            # Converged or saturated; overflow flag catches the latter.
            break
        round_idx = round_idx + wp.int32(1)

    if overflow_local != wp.int32(0) and lane == wp.int32(0):
        overflow_flag[0] = wp.int32(1)

    # CSR build: exclusive prefix on color_count[w, :] (lane 0; 64 entries).
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

    # Scatter ids into per-world CSR slices via atomic_add(color_offsets[w, c]).
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


# Fast-path single-block-per-world dispatchers. Each block walks its world's
# full CSR with __syncthreads between colours; same-colour cids never share a
# body so per-lane RMW is race-free.
#
# Multi-world fast-tail kernels: revolute_only skips the joint-mode branch.


def _make_fast_tail_prepare_plus_iterate_kernel(*, revolute_only: bool):
    """Build the multi-world fused prepare + iterate fast-tail kernel."""

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

        # Iterate phase: outer = num_iterations / _FUSED_INNER_SWEEPS, each
        # outer round runs *_iterate_multi to hold state in registers.
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
    """Multi-world relax fast-tail kernel (use_bias=False, num_sweeps=num_iterations)."""

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

        # *_iterate_multi with num_sweeps=num_iterations folds the whole relax
        # into one register-cached call (velocity_iterations is typically 1).
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
    """Project active constraints into ElementInteractionData. Static bodies
    collapse to -1; the dynamic body compacts to slot 0."""
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
    # Adjacency loop stops on the first -1, so compact non-negative ids first.
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
    """Per-cid position-level residual: top=linear [m], bottom=angular [rad]."""
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
    """Axis-angle rotation quaternion for ``omega * dt``. Unit norm by construction."""
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
    """Advance pose for dynamic bodies only. Kinematic bodies advance via
    lerp/slerp in :func:`_kinematic_interpolate_substep_kernel`."""
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
    """Per-step kinematic prepare. Resolves target (scripted vs constant-vel),
    snapshots prev pose for substep lerp/slerp, infers velocity via quaternion
    log-map (exact for large rotations)."""
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
        # One-shot: user must re-assert the target each step.
        bodies.kinematic_target_valid[i] = 0
    else:
        # Constant-velocity fallthrough: advance pose by velocity*dt.
        pos_target = pos_prev + bodies.velocity[i] * dt
        q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
        orient_target = wp.normalize(q_rot * orient_prev)
        bodies.kinematic_target_pos[i] = pos_target
        bodies.kinematic_target_orient[i] = orient_target

    # Infer velocity from pose delta. Round-trips exactly on the constant-
    # velocity path; exposes pose derivative for the scripted path.
    inv_dt = wp.float32(1.0) / dt
    v = (pos_target - pos_prev) * inv_dt

    # Canonicalise to shortest-path hemisphere before the atan2.
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
    """Batched writeback for :meth:`PhoenXWorld.set_kinematic_pose`. Silently
    no-ops on non-kinematic bodies (host should validate and raise)."""
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
    """Per-substep kinematic pose: position = lerp(prev, target, alpha),
    orientation = slerp(prev, target, alpha). alpha = (substep+1)/num_substeps;
    at alpha=1 the body lands exactly on its target."""
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


# Per-step body kernels (forces + gravity, inertia refresh, force clear) plus
# the on-device active-constraint count fuse. Driven from PhoenXWorld.step.


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
    """Per-body velocity update at the top of every substep: external forces +
    gravity, fused. Force accumulators are zeroed in
    :func:`_phoenx_update_inertia_and_clear_forces_kernel` at end-of-step."""
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
    """End-of-step: per-body damping + world-inertia rebuild (R * I^-1 * R^T)
    + force/torque accumulator zeroing. Runs once per step."""
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
    """Per-substep refresh of inverse_inertia_world (R * I^-1 * R^T) so the
    next substep's solve sees the rotated inertia. Anisotropic bodies drift in
    angular momentum without this when running multiple substeps."""
    i = wp.tid()
    if bodies.motion_type[i] == MOTION_DYNAMIC:
        r = wp.quat_to_matrix(bodies.orientation[i])
        bodies.inverse_inertia_world[i] = rotate_inertia(r, bodies.inverse_inertia[i])


@wp.kernel(enable_backward=False)
def _phoenx_apply_global_damping_kernel(
    bodies: BodyContainer,
    global_damping: wp.array[wp.float32],
):
    """Per-substep global damping for dynamic bodies. ``global_damping`` is
    [linear, angular]; v *= 1 - linear, w *= 1 - angular. Device-stored so the
    host can rewrite without re-capture."""
    i = wp.tid()
    if bodies.motion_type[i] == MOTION_DYNAMIC:
        lin = 1.0 - global_damping[0]
        ang = 1.0 - global_damping[1]
        bodies.velocity[i] = bodies.velocity[i] * lin
        bodies.angular_velocity[i] = bodies.angular_velocity[i] * ang


# Single-world step path: per-colour grid launches via wp.capture_while on
# head_active. Persistent grid sized once at construction.
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
# Tail launches after head_active clears stay cheap (early-exit no-ops) until
# the outer capture_while observes head_active[0] == 0.


@wp.func
def _singleworld_color_range(
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    color_cursor: wp.array[wp.int32],
):
    """Decode current colour's cid range from cursor. Returns (start, count, cursor)."""
    cursor = color_cursor[0]
    n_colors = num_colors[0]
    c = n_colors - cursor
    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start
    return start, count, cursor


# Single-world fused-tail kernels: one 1D block walks trailing small colours,
# __syncthreads between colours. Hands off to the persistent kernel on the
# first colour > FUSE_TAIL_MAX_COLOR_SIZE without decrementing the cursor.


@wp.kernel(enable_backward=False)
def _reset_head_active_kernel(head_active: wp.array[wp.int32]):
    """Reset head_active[0] = 1 so the next capture_while gets at least one launch."""
    head_active[0] = 1


@wp.func
def _singleworld_color_range_from_cursor(
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    cursor: wp.int32,
):
    """:func:`_singleworld_color_range` taking the cursor as a register value."""
    n_colors = num_colors[0]
    c = n_colors - cursor
    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start
    return start, count


# Single-world kernel factories: persistent (head) + single-block (fused tail)
# for prepare/iterate/relax x revolute_only/generic. ``phase`` and ``revolute_only``
# are compile-time so Warp constant-folds + dead-code-eliminates the unused branch.


def _make_singleworld_persistent_kernel(*, phase: str, revolute_only: bool):
    """Persistent-grid PGS kernel for the requested phase + specialisation.
    Single-world uses the single-sweep ``*_iterate`` helpers (multi-sweep
    register cache regresses on contact-heavy single-world like kapla)."""
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
    """Single-block tail-fused PGS kernel; same axes as
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
