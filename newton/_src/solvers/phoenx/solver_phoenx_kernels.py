# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for :class:`PhoenXWorld`. Dispatches only ADBS and CONTACT."""

from __future__ import annotations

import functools

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
)
from newton._src.solvers.phoenx.array_helper import read1d_i32
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
    body_set_access_mode,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    ADBS_TIME_US_OFFSET,
    actuated_double_ball_socket_cached_warmstart,
    actuated_double_ball_socket_iterate,
    actuated_double_ball_socket_iterate_multi,
    actuated_double_ball_socket_prepare_for_iteration,
    actuated_double_ball_socket_world_error,
    actuated_double_ball_socket_world_wrench,
    revolute_cached_warmstart,
    revolute_iterate,
    revolute_iterate_multi,
    revolute_prepare_for_iteration,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_bending import (
    CLOTH_BENDING_TIME_US_OFFSET,
    cloth_bending_iterate_at,
    cloth_bending_prepare_for_iteration_at,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    CLOTH_TRIANGLE_TIME_US_OFFSET,
    cloth_triangle_iterate_at,
    cloth_triangle_prepare_for_iteration_at,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_accumulate_time_us,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_side0_kind,
    contact_get_side0_nodes_extra,
    contact_get_side1_kind,
    contact_get_side1_nodes_extra,
    contact_iterate_multi,
    contact_iterate_multi_no_soft_pd,
    contact_world_error,
    contact_world_wrench,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    contact_cached_warmstart_lean,
    contact_iterate,
    contact_iterate_cloth_aware,
    contact_iterate_lean,
    contact_iterate_lean_no_sleep,
    contact_iterate_lean_no_sleep_no_soft_pd,
    contact_iterate_lean_no_soft_pd,
    contact_iterate_no_sleep,
    contact_iterate_no_sleep_no_soft_pd,
    contact_iterate_no_soft_pd,
    contact_prepare_for_iteration,
    contact_prepare_for_iteration_cloth_aware,
    contact_prepare_for_iteration_lean,
    contact_prepare_for_iteration_lean_no_soft_pd,
    contact_prepare_for_iteration_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CLOTH_BENDING,
    CONSTRAINT_TYPE_CLOTH_TRIANGLE,
    CONSTRAINT_TYPE_SOFT_HEXAHEDRON,
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON,
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN,
    ConstraintContainer,
    constraint_accumulate_time_us,
    constraint_get_body1,
    constraint_get_body2,
    constraint_get_type,
    read_int,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_hexahedron import (
    SOFT_HEX_TIME_US_OFFSET,
    soft_hexahedron_iterate_at,
    soft_hexahedron_prepare_for_iteration_at,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tet_neohookean import (
    SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET,
    soft_tet_neohookean_iterate_at,
    soft_tet_neohookean_prepare_for_iteration_at,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    SOFT_TET_TIME_US_OFFSET,
    soft_tetrahedron_iterate_at,
    soft_tetrahedron_prepare_for_iteration_at,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_side0_bary,
    cc_get_side1_bary,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    GREEDY_MAX_COLORS,
    MAX_BODIES,
    ElementInteractionData,
    _lowest_set_bit,
    element_interaction_data_make,
)
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.timer import elapsed_us, read_global_timer_ns

# Body-N dword offsets in the per-constraint header. Each constraint
# type stores (type, body1, body2) at dwords 0/1/2 then extra bodies
# at dwords 3/4 (cloth-tri uses body3 only; soft-tet and cloth-bend
# use body3 + body4).
_CLOTH_TRIANGLE_OFF_BODY3 = wp.constant(wp.int32(3))
_SOFT_TET_OFF_BODY3 = wp.constant(wp.int32(3))
_SOFT_TET_OFF_BODY4 = wp.constant(wp.int32(4))
_CLOTH_BENDING_OFF_BODY3 = wp.constant(wp.int32(3))
_CLOTH_BENDING_OFF_BODY4 = wp.constant(wp.int32(4))
# Soft hex stamps body1/body2 at dwords 1/2 (header) and body3..body8 at
# dwords 3..8. Used by ``_constraints_to_elements_kernel`` to emit the
# 8-body element interaction.
_SOFT_HEX_OFF_BODY3 = wp.constant(wp.int32(3))
_SOFT_HEX_OFF_BODY4 = wp.constant(wp.int32(4))
_SOFT_HEX_OFF_BODY5 = wp.constant(wp.int32(5))
_SOFT_HEX_OFF_BODY6 = wp.constant(wp.int32(6))
_SOFT_HEX_OFF_BODY7 = wp.constant(wp.int32(7))
_SOFT_HEX_OFF_BODY8 = wp.constant(wp.int32(8))


__all__ = [
    "_PER_WORLD_COLORING_BLOCK_DIM",
    "_PER_WORLD_FAST_FAMILIES",
    "_STRAGGLER_BLOCK_DIM",
    "_build_scatter_keys_kernel",
    "_choose_fast_tail_worlds_per_block",
    "_constraint_gather_errors_kernel",
    "_constraint_gather_wrenches_kernel",
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
    "_reduce_constraint_time_us_kernel",
    "_reduce_contact_time_us_kernel",
    "_reduce_total_colours_kernel",
    "_set_kinematic_pose_batch_kernel",
    "_zero_constraint_time_us_kernel",
    "_zero_contact_time_us_kernel",
    "get_block_world_kernel",
    "get_fast_tail_kernel",
    "get_singleworld_kernel",
    "pack_body_xforms_kernel",
]


#: Warp size and default max threads-per-world for fast-tail kernels.
#: Dynamic auto launches keep this upper bound; fixed launches may use less.
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

    Dynamic launches reserve one warp (32 threads) per world; fixed-tpw
    launches can pack multiple worlds per warp. Block size is still an integer
    warp count, so ``__syncwarp()`` stays valid. Three-tier by world count,
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


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp(mask);
#endif
""")
def _sync_warp_mask(mask: wp.uint32): ...


# Adaptive threads-per-world picker for dynamic auto launches. The effective
# tpw is read from a 1-elem buffer per step; fixed-tpw kernels bypass it.


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
    mean cids/colour <= 10 (sparse colours, saturated SMs); else tpw=32.
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
    if mean_x16 <= wp.int32(10 * 16) and saturation_x16 >= wp.int32(8 * 16):
        pick = wp.int32(16)

    tpw_choice[0] = pick


# Per-world JP MIS coloring: worlds are independent (static-body nullification),
# one block per world, output goes straight to per-world CSR.


_PER_WORLD_COLORING_BLOCK_DIM: int = 64
# Per-world color CSR subranges used to keep rows with the same solver
# dispatcher adjacent. Slot 0/1 must stay joint/contact for the rigid
# family-split fast path; slots 2..5 preserve cloth/soft family order for
# mixed scenes even when the solve loop consumes the full color range.
_PER_WORLD_FAST_FAMILIES: int = 6


@wp.func
def _element_fast_family(family: wp.int32) -> wp.int32:
    if family < wp.int32(0):
        return wp.int32(1)
    if family >= wp.int32(_PER_WORLD_FAST_FAMILIES):
        return wp.int32(_PER_WORLD_FAST_FAMILIES - 1)
    return family


@wp.func
def _node_world_id(
    node: wp.int32,
    bodies: BodyContainer,
    particle_world_id: wp.array[wp.int32],
    num_bodies: wp.int32,
) -> wp.int32:
    if node < wp.int32(0):
        return wp.int32(-1)
    if node < num_bodies:
        return bodies.world_id[node]
    return particle_world_id[node - num_bodies]


@wp.func
def _element_world_id(
    element: ElementInteractionData,
    bodies: BodyContainer,
    particle_world_id: wp.array[wp.int32],
    num_bodies: wp.int32,
) -> wp.int32:
    seen_node = wp.int32(0)
    for i in range(MAX_BODIES):
        node = element.bodies[i]
        if node < wp.int32(0):
            break
        seen_node = wp.int32(1)
        world_id = _node_world_id(node, bodies, particle_world_id, num_bodies)
        if world_id >= wp.int32(0):
            return world_id
    if seen_node != wp.int32(0):
        return wp.int32(0)
    return wp.int32(-1)


@wp.kernel(enable_backward=False)
def _count_elements_per_world_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    bodies: BodyContainer,
    particle_world_id: wp.array[wp.int32],
    num_bodies: wp.int32,
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
    w = _element_world_id(elements[tid], bodies, particle_world_id, num_bodies)
    if w < wp.int32(0):
        return
    wp.atomic_add(world_element_count, w, wp.int32(1))
    wp.atomic_add(world_element_offsets_shifted, w + wp.int32(1), wp.int32(1))


@wp.kernel(enable_backward=False)
def _build_scatter_keys_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    bodies: BodyContainer,
    particle_world_id: wp.array[wp.int32],
    num_bodies: wp.int32,
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
    w = _element_world_id(elements[tid], bodies, particle_world_id, num_bodies)
    if w < wp.int32(0):
        keys[tid] = wp.int32(2147483647)
        values[tid] = wp.int32(-1)
        return
    keys[tid] = w
    values[tid] = tid


@wp.func
def _cost_biased_priority(
    packed_priorities: wp.array[wp.int32],
    cid: wp.int32,
) -> wp.int32:
    """Read prepacked (cost << 24) | (random & 0xFFFFFF) priority.

    The pack happens once per step in the partitioner; coloring just
    reads. See :func:`pack_priorities_kernel` in :mod:`graph_coloring_common`.
    Bit layout makes plain int32 lexicographic comparison equivalent
    to ``(cost, random)`` lex order.
    """
    return packed_priorities[cid]


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
    packed_priorities: wp.array[wp.int32],  # [capacity] (cost << 24) | (random & 0xFFFFFF)
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
                    self_prio = _cost_biased_priority(packed_priorities, eid)
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
                            if _cost_biased_priority(packed_priorities, neighbor) > self_prio:
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


@wp.kernel(enable_backward=False, module="unique")
def _per_world_greedy_coloring_kernel(
    # per-world bucketing (input from the two kernels above)
    world_element_offsets: wp.array[wp.int32],  # [nw+1] (exclusive prefix of counts)
    world_element_count: wp.array[wp.int32],  # [nw] (raw per-world count)
    world_elements: wp.array[wp.int32],  # [total] flat cid stream, sorted by world
    # graph data
    elements: wp.array[ElementInteractionData],
    element_family: wp.array[wp.int32],
    adjacency_end: wp.array[wp.int32],  # [num_bodies]
    vertex_to_elements: wp.array[wp.int32],  # [cap * MAX_BODIES]
    packed_priorities: wp.array[wp.int32],  # [capacity] (cost << 24) | (random & 0xFFFFFF)
    max_colors: wp.int32,  # = GREEDY_MAX_COLORS, kept for parity with JP variant
    # scratch (caller zeros each step)
    assigned: wp.array[wp.int32],  # [capacity] 0 unassigned, (c+1) = coloured
    color_count: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS] histogram bucket
    color_offsets: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS] live cursor for scatter
    color_family_count: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS * _PER_WORLD_FAST_FAMILIES]
    color_family_offsets: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS * _PER_WORLD_FAST_FAMILIES]
    # outputs
    world_element_ids_by_color: wp.array[wp.int32],  # [total] sorted-by-colour per world
    world_color_starts: wp.array2d[wp.int32],  # [nw, MAX_COLORS+1] per-world prefix
    world_color_family_starts: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS * _PER_WORLD_FAST_FAMILIES]
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

    family_slots = wp.int32(GREEDY_MAX_COLORS * _PER_WORLD_FAST_FAMILIES)
    offset = wp.int32(0)
    while offset < family_slots:
        slot = offset + lane
        if slot < family_slots:
            color_family_count[w, slot] = wp.int32(0)
            color_family_offsets[w, slot] = wp.int32(0)
            world_color_family_starts[w, slot] = wp.int32(0)
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
                    self_prio = _cost_biased_priority(packed_priorities, eid)
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
                                if _cost_biased_priority(packed_priorities, neighbor) > self_prio:
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
                            family = _element_fast_family(element_family[eid])
                            wp.atomic_add(
                                color_family_count,
                                w,
                                c * wp.int32(_PER_WORLD_FAST_FAMILIES) + family,
                                wp.int32(1),
                            )
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
    # Each colour is also split into solver-family subranges. The rigid
    # fast-tail path consumes joint/contact slots directly; mixed scenes still
    # benefit because same-family rows stay adjacent inside the color.
    if lane == wp.int32(0):
        running = wp.int32(0)
        last_used = wp.int32(-1)
        for c in range(GREEDY_MAX_COLORS):
            world_color_starts[w, c] = running
            family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
            family_running = wp.int32(0)
            for f in range(_PER_WORLD_FAST_FAMILIES):
                slot = family_base + wp.int32(f)
                color_family_offsets[w, slot] = running + family_running
                world_color_family_starts[w, slot] = running + family_running
                family_running = family_running + color_family_count[w, slot]
            if family_running > wp.int32(0):
                last_used = c
            running = running + family_running
        world_color_starts[w, GREEDY_MAX_COLORS] = running
        world_num_colors[w] = last_used + wp.int32(1)
    _sync_threads()

    # Scatter ids into per-world CSR slices via per-family cursors.
    offset = wp.int32(0)
    while offset < count:
        slot = offset + lane
        if slot < count:
            eid = world_elements[base + slot]
            a = assigned[eid]
            if a > wp.int32(0):
                c = a - wp.int32(1)
                family = _element_fast_family(element_family[eid])
                family_slot = c * wp.int32(_PER_WORLD_FAST_FAMILIES) + family
                local_slot = wp.atomic_add(color_family_offsets, w, family_slot, wp.int32(1))
                world_element_ids_by_color[base + local_slot] = eid
        offset = offset + stride


# Fast-path single-block-per-world dispatchers. Each block walks its world's
# full CSR with __syncthreads between colours; same-colour cids never share a
# body so per-lane RMW is race-free.
#
# Multi-world fast-tail kernels: revolute_only skips the joint-mode branch.


@functools.cache
def _make_multiworld_rigid_prepare_dispatch_func(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    has_soft_contact_pd: bool,
    cached_prepare: bool,
    enable_column_timers: bool,
):
    """Generated rigid multi-world prepare dispatch."""

    @wp.func
    def _dispatch_prepare_joint(
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        cid: wp.int32,
    ):
        if wp.static(cached_prepare):
            if wp.static(revolute_only):
                revolute_cached_warmstart(constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt)
            else:
                actuated_double_ball_socket_cached_warmstart(
                    constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                )
        elif wp.static(revolute_only):
            revolute_prepare_for_iteration(
                constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
            )
        else:
            actuated_double_ball_socket_prepare_for_iteration(
                constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
            )

    @wp.func
    def _dispatch_prepare_contact(
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        local_cid: wp.int32,
    ):
        if wp.static(cached_prepare):
            contact_cached_warmstart_lean(
                contact_cols,
                local_cid,
                bodies,
                particles,
                num_bodies,
                idt,
                cc,
                contacts,
                copy_state,
                wp.int32(0),
            )
        else:
            if wp.static(has_soft_contact_pd):
                contact_prepare_for_iteration_lean(
                    contact_cols,
                    local_cid,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    copy_state,
                    wp.int32(0),
                )
            else:
                contact_prepare_for_iteration_lean_no_soft_pd(
                    contact_cols,
                    local_cid,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    copy_state,
                    wp.int32(0),
                )

    @wp.func
    def _dispatch_prepare_cid(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        cid: wp.int32,
        num_joints: wp.int32,
    ):
        t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            t0 = read_global_timer_ns()

        if wp.static(has_joints and not has_contacts):
            _dispatch_prepare_joint(constraints, bodies, particles, copy_state, num_bodies, idt, cid)
            if wp.static(enable_column_timers):
                constraint_accumulate_time_us(
                    constraints, ADBS_TIME_US_OFFSET, cid, elapsed_us(t0, read_global_timer_ns())
                )
        elif cid < num_joints:
            _dispatch_prepare_joint(constraints, bodies, particles, copy_state, num_bodies, idt, cid)
            if wp.static(enable_column_timers):
                constraint_accumulate_time_us(
                    constraints, ADBS_TIME_US_OFFSET, cid, elapsed_us(t0, read_global_timer_ns())
                )
        else:
            local_cid = cid - num_joints
            _dispatch_prepare_contact(
                contact_cols,
                bodies,
                particles,
                cc,
                contacts,
                copy_state,
                num_bodies,
                idt,
                local_cid,
            )
            if wp.static(enable_column_timers):
                contact_accumulate_time_us(contact_cols, local_cid, elapsed_us(t0, read_global_timer_ns()))

    return _dispatch_prepare_cid


@functools.cache
def _make_multiworld_rigid_iterate_dispatch_funcs(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    enable_column_timers: bool,
    use_bias: bool,
):
    """Generated rigid multi-world multi-sweep iterate dispatch."""

    @wp.func
    def _dispatch_iterate_joint(
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        cid: wp.int32,
        num_sweeps: wp.int32,
    ):
        t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            t0 = read_global_timer_ns()
        if wp.static(revolute_only):
            revolute_iterate_multi(
                constraints,
                cid,
                bodies,
                particles,
                copy_state,
                num_bodies,
                wp.int32(0),
                idt,
                sor_boost,
                use_bias,
                num_sweeps,
            )
        else:
            actuated_double_ball_socket_iterate_multi(
                constraints,
                cid,
                bodies,
                particles,
                copy_state,
                num_bodies,
                wp.int32(0),
                idt,
                sor_boost,
                use_bias,
                num_sweeps,
            )
        if wp.static(enable_column_timers):
            constraint_accumulate_time_us(constraints, ADBS_TIME_US_OFFSET, cid, elapsed_us(t0, read_global_timer_ns()))

    @wp.func
    def _dispatch_iterate_contact(
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        local_cid: wp.int32,
        num_sweeps: wp.int32,
    ):
        t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            t0 = read_global_timer_ns()
        skip_frozen = False
        if wp.static(has_sleeping):
            cb1 = contact_get_body1(contact_cols, local_cid)
            cb2 = contact_get_body2(contact_cols, local_cid)
            if cb1 >= 0 and cb1 < num_bodies and cb2 >= 0 and cb2 < num_bodies:
                fr1 = (bodies.motion_type[cb1] != MOTION_DYNAMIC) or (bodies.island_root[cb1] >= wp.int32(0))
                fr2 = (bodies.motion_type[cb2] != MOTION_DYNAMIC) or (bodies.island_root[cb2] >= wp.int32(0))
                if fr1 and fr2:
                    skip_frozen = True
        if not skip_frozen:
            if wp.static(has_soft_contact_pd):
                contact_iterate_multi(
                    contact_cols,
                    local_cid,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    use_bias,
                    num_sweeps,
                    copy_state,
                    wp.int32(0),
                    sor_boost,
                )
            else:
                contact_iterate_multi_no_soft_pd(
                    contact_cols,
                    local_cid,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    use_bias,
                    num_sweeps,
                    copy_state,
                    wp.int32(0),
                    sor_boost,
                )
        if wp.static(enable_column_timers):
            contact_accumulate_time_us(contact_cols, local_cid, elapsed_us(t0, read_global_timer_ns()))

    @wp.func
    def _dispatch_iterate_cid(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        cid: wp.int32,
        num_joints: wp.int32,
        num_sweeps: wp.int32,
    ):
        if wp.static(has_joints and not has_contacts):
            _dispatch_iterate_joint(
                constraints, bodies, particles, copy_state, num_bodies, idt, sor_boost, cid, num_sweeps
            )
        elif cid < num_joints:
            _dispatch_iterate_joint(
                constraints, bodies, particles, copy_state, num_bodies, idt, sor_boost, cid, num_sweeps
            )
        else:
            _dispatch_iterate_contact(
                contact_cols,
                bodies,
                particles,
                cc,
                contacts,
                copy_state,
                num_bodies,
                idt,
                sor_boost,
                cid - num_joints,
                num_sweeps,
            )

    return _dispatch_iterate_cid, _dispatch_iterate_joint, _dispatch_iterate_contact


@functools.cache
def _make_fast_tail_prepare_plus_iterate_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_only: bool = False,
    cloth_only: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
):
    """Build the multi-world fused prepare + iterate fast-tail kernel."""
    _dispatch_prepare_cid = _make_multiworld_rigid_prepare_dispatch_func(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        has_soft_contact_pd=has_soft_contact_pd,
        cached_prepare=cached_prepare,
        enable_column_timers=enable_column_timers,
    )
    (
        _dispatch_iterate_cid,
        _dispatch_iterate_joint,
        _dispatch_iterate_contact,
    ) = _make_multiworld_rigid_iterate_dispatch_funcs(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        enable_column_timers=enable_column_timers,
        use_bias=True,
    )
    _dispatch_prepare_any_cid = None
    _dispatch_iterate_any_cid = None
    if cloth_support:
        _dispatch_prepare_any_cid = _make_singleworld_dispatch_func(
            revolute_only=revolute_only,
            cloth_support=cloth_support,
            enable_column_timers=enable_column_timers,
            soft_tet_only=soft_tet_only,
            cloth_only=cloth_only,
            has_joints=has_joints,
            has_mass_splitting=False,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            is_prepare=True,
            is_cached_prepare=False,
            use_bias=True,
        )
        _dispatch_iterate_any_cid = _make_singleworld_dispatch_func(
            revolute_only=revolute_only,
            cloth_support=cloth_support,
            enable_column_timers=enable_column_timers,
            soft_tet_only=soft_tet_only,
            cloth_only=cloth_only,
            has_joints=has_joints,
            has_mass_splitting=False,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            is_prepare=False,
            is_cached_prepare=False,
            use_bias=True,
        )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        world_element_ids_by_color: wp.array[wp.int32],
        world_color_starts: wp.array2d[wp.int32],
        world_color_family_starts: wp.array2d[wp.int32],
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        num_cloth_triangles: wp.int32,
        num_cloth_bending: wp.int32,
        num_soft_tetrahedra: wp.int32,
        num_soft_hexahedra: wp.int32,
        num_bodies: wp.int32,
        tpw_buf: wp.array[wp.int32],
        copy_state: CopyStateContainer,
    ):
        tid = wp.tid()
        if wp.static(fixed_tpw > 0):
            if wp.static(guard_tpw):
                if tpw_buf[0] != wp.int32(fixed_tpw):
                    return
            tpw = wp.int32(fixed_tpw)
        else:
            tpw = tpw_buf[0]
        local_tid = tid % tpw
        world_id = tid / tpw
        if world_id >= num_worlds:
            return

        sync_mask = wp.uint32(0xFFFFFFFF)
        if wp.static(fixed_tpw == 8):
            warp_lane = tid & wp.int32(31)
            if warp_lane < wp.int32(8):
                sync_mask = wp.uint32(0x000000FF)
            elif warp_lane < wp.int32(16):
                sync_mask = wp.uint32(0x0000FF00)
            elif warp_lane < wp.int32(24):
                sync_mask = wp.uint32(0x00FF0000)
            else:
                sync_mask = wp.uint32(0xFF000000)
        elif wp.static(fixed_tpw == 16):
            if (tid & wp.int32(31)) < wp.int32(16):
                sync_mask = wp.uint32(0x0000FFFF)
            else:
                sync_mask = wp.uint32(0xFFFF0000)

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
                if wp.static(cloth_support):
                    _dispatch_prepare_any_cid(
                        constraints,
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_joints,
                        num_cloth_triangles,
                        num_cloth_bending,
                        num_soft_tetrahedra,
                        num_soft_hexahedra,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid,
                        wp.int32(0),
                    )
                else:
                    _dispatch_prepare_cid(
                        constraints,
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_bodies,
                        idt,
                        cid,
                        num_joints,
                    )
                base += tpw

            _sync_warp_mask(sync_mask)
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

                if wp.static(family_split and not cloth_support):
                    family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
                    joint_start = world_base + world_color_family_starts[world_id, family_base]
                    contact_start = world_base + world_color_family_starts[world_id, family_base + wp.int32(1)]
                    count_joints = contact_start - joint_start

                    base = local_tid
                    while base < count_joints:
                        cid = world_element_ids_by_color[joint_start + base]
                        _dispatch_iterate_joint(
                            constraints,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            idt,
                            sor_boost,
                            cid,
                            inner_sweeps,
                        )
                        base += tpw

                    count_contacts = end - contact_start
                    base = local_tid
                    while base < count_contacts:
                        cid = world_element_ids_by_color[contact_start + base]
                        local_cid = cid - num_joints
                        _dispatch_iterate_contact(
                            contact_cols,
                            bodies,
                            particles,
                            cc,
                            contacts,
                            copy_state,
                            num_bodies,
                            idt,
                            sor_boost,
                            local_cid,
                            inner_sweeps,
                        )
                        base += tpw
                else:
                    base = local_tid
                    while base < count:
                        cid = world_element_ids_by_color[start + base]
                        if wp.static(cloth_support):
                            sweep = wp.int32(0)
                            while sweep < inner_sweeps:
                                _dispatch_iterate_any_cid(
                                    constraints,
                                    contact_cols,
                                    bodies,
                                    particles,
                                    cc,
                                    contacts,
                                    copy_state,
                                    num_joints,
                                    num_cloth_triangles,
                                    num_cloth_bending,
                                    num_soft_tetrahedra,
                                    num_soft_hexahedra,
                                    num_bodies,
                                    idt,
                                    sor_boost,
                                    cid,
                                    wp.int32(0),
                                )
                                sweep += wp.int32(1)
                        else:
                            _dispatch_iterate_cid(
                                constraints,
                                contact_cols,
                                bodies,
                                particles,
                                cc,
                                contacts,
                                copy_state,
                                num_bodies,
                                idt,
                                sor_boost,
                                cid,
                                num_joints,
                                inner_sweeps,
                            )
                        base += tpw

                _sync_warp_mask(sync_mask)
                c += 1

            it_outer += 1

    return kernel


@functools.cache
def _make_fast_tail_relax_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_only: bool = False,
    cloth_only: bool = False,
    enable_column_timers: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
):
    """Multi-world relax fast-tail kernel (use_bias=False, num_sweeps=num_iterations)."""
    (
        _dispatch_iterate_cid,
        _dispatch_iterate_joint,
        _dispatch_iterate_contact,
    ) = _make_multiworld_rigid_iterate_dispatch_funcs(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        enable_column_timers=enable_column_timers,
        use_bias=False,
    )
    _dispatch_relax_any_cid = None
    if cloth_support:
        _dispatch_relax_any_cid = _make_singleworld_dispatch_func(
            revolute_only=revolute_only,
            cloth_support=cloth_support,
            enable_column_timers=enable_column_timers,
            soft_tet_only=soft_tet_only,
            cloth_only=cloth_only,
            has_joints=has_joints,
            has_mass_splitting=False,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            is_prepare=False,
            is_cached_prepare=False,
            use_bias=False,
        )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        world_element_ids_by_color: wp.array[wp.int32],
        world_color_starts: wp.array2d[wp.int32],
        world_color_family_starts: wp.array2d[wp.int32],
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        num_cloth_triangles: wp.int32,
        num_cloth_bending: wp.int32,
        num_soft_tetrahedra: wp.int32,
        num_soft_hexahedra: wp.int32,
        tpw_buf: wp.array[wp.int32],
        copy_state: CopyStateContainer,
    ):
        tid = wp.tid()
        if wp.static(fixed_tpw > 0):
            if wp.static(guard_tpw):
                if tpw_buf[0] != wp.int32(fixed_tpw):
                    return
            tpw = wp.int32(fixed_tpw)
        else:
            tpw = tpw_buf[0]
        local_tid = tid % tpw
        world_id = tid / tpw
        if world_id >= num_worlds:
            return

        sync_mask = wp.uint32(0xFFFFFFFF)
        if wp.static(fixed_tpw == 8):
            warp_lane = tid & wp.int32(31)
            if warp_lane < wp.int32(8):
                sync_mask = wp.uint32(0x000000FF)
            elif warp_lane < wp.int32(16):
                sync_mask = wp.uint32(0x0000FF00)
            elif warp_lane < wp.int32(24):
                sync_mask = wp.uint32(0x00FF0000)
            else:
                sync_mask = wp.uint32(0xFF000000)
        elif wp.static(fixed_tpw == 16):
            if (tid & wp.int32(31)) < wp.int32(16):
                sync_mask = wp.uint32(0x0000FFFF)
            else:
                sync_mask = wp.uint32(0xFFFF0000)

        n_colors = world_num_colors[world_id]
        world_base = world_csr_offsets[world_id]

        # *_iterate_multi with num_sweeps=num_iterations folds the whole relax
        # into one register-cached call (velocity_iterations is typically 1).
        c = wp.int32(0)
        while c < n_colors:
            if wp.static(family_split and not cloth_support):
                family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
                joint_start = world_base + world_color_family_starts[world_id, family_base]
                contact_start = world_base + world_color_family_starts[world_id, family_base + wp.int32(1)]
                end = world_base + world_color_starts[world_id, c + 1]
                count_joints = contact_start - joint_start

                base = local_tid
                while base < count_joints:
                    cid = world_element_ids_by_color[joint_start + base]
                    _dispatch_iterate_joint(
                        constraints,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid,
                        num_iterations,
                    )
                    base += tpw

                count_contacts = end - contact_start
                base = local_tid
                while base < count_contacts:
                    cid = world_element_ids_by_color[contact_start + base]
                    local_cid = cid - num_joints
                    _dispatch_iterate_contact(
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        local_cid,
                        num_iterations,
                    )
                    base += tpw
            else:
                start = world_base + world_color_starts[world_id, c]
                end = world_base + world_color_starts[world_id, c + 1]
                count = end - start

                base = local_tid
                while base < count:
                    cid = world_element_ids_by_color[start + base]
                    if wp.static(cloth_support):
                        sweep = wp.int32(0)
                        while sweep < num_iterations:
                            _dispatch_relax_any_cid(
                                constraints,
                                contact_cols,
                                bodies,
                                particles,
                                cc,
                                contacts,
                                copy_state,
                                num_joints,
                                num_cloth_triangles,
                                num_cloth_bending,
                                num_soft_tetrahedra,
                                num_soft_hexahedra,
                                num_bodies,
                                idt,
                                sor_boost,
                                cid,
                                wp.int32(0),
                            )
                            sweep += wp.int32(1)
                    else:
                        _dispatch_iterate_cid(
                            constraints,
                            contact_cols,
                            bodies,
                            particles,
                            cc,
                            contacts,
                            copy_state,
                            num_bodies,
                            idt,
                            sor_boost,
                            cid,
                            num_joints,
                            num_iterations,
                        )
                    base += tpw

            _sync_warp_mask(sync_mask)
            c += 1

    return kernel


@functools.cache
def _make_block_world_prepare_plus_iterate_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    family_split: bool = False,
    block_dim: int = 128,
):
    """Build a multi-world kernel where one physical block owns one world."""
    _dispatch_prepare_cid = _make_multiworld_rigid_prepare_dispatch_func(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        has_soft_contact_pd=has_soft_contact_pd,
        cached_prepare=cached_prepare,
        enable_column_timers=enable_column_timers,
    )
    (
        _dispatch_iterate_cid,
        _dispatch_iterate_joint,
        _dispatch_iterate_contact,
    ) = _make_multiworld_rigid_iterate_dispatch_funcs(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        enable_column_timers=enable_column_timers,
        use_bias=True,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        world_element_ids_by_color: wp.array[wp.int32],
        world_color_starts: wp.array2d[wp.int32],
        world_color_family_starts: wp.array2d[wp.int32],
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        num_bodies: wp.int32,
        copy_state: CopyStateContainer,
    ):
        tid = wp.tid()
        local_tid = tid % wp.int32(block_dim)
        world_id = tid / wp.int32(block_dim)
        if world_id >= num_worlds:
            return

        n_colors = world_num_colors[world_id]
        world_base = world_csr_offsets[world_id]

        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + wp.int32(1)]
            count = end - start
            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                _dispatch_prepare_cid(
                    constraints,
                    contact_cols,
                    bodies,
                    particles,
                    cc,
                    contacts,
                    copy_state,
                    num_bodies,
                    idt,
                    cid,
                    num_joints,
                )
                base += wp.int32(block_dim)

            _sync_threads()
            c += wp.int32(1)

        inner_sweeps = wp.int32(_FUSED_INNER_SWEEPS)
        outer_iters = num_iterations / inner_sweeps
        it_outer = wp.int32(0)
        while it_outer < outer_iters:
            c = wp.int32(0)
            while c < n_colors:
                start = world_base + world_color_starts[world_id, c]
                end = world_base + world_color_starts[world_id, c + wp.int32(1)]

                if wp.static(family_split):
                    family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
                    joint_start = world_base + world_color_family_starts[world_id, family_base]
                    contact_start = world_base + world_color_family_starts[world_id, family_base + wp.int32(1)]
                    count_joints = contact_start - joint_start

                    base = local_tid
                    while base < count_joints:
                        cid = world_element_ids_by_color[joint_start + base]
                        _dispatch_iterate_joint(
                            constraints,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            idt,
                            sor_boost,
                            cid,
                            inner_sweeps,
                        )
                        base += wp.int32(block_dim)

                    count_contacts = end - contact_start
                    base = local_tid
                    while base < count_contacts:
                        cid = world_element_ids_by_color[contact_start + base]
                        local_cid = cid - num_joints
                        _dispatch_iterate_contact(
                            contact_cols,
                            bodies,
                            particles,
                            cc,
                            contacts,
                            copy_state,
                            num_bodies,
                            idt,
                            sor_boost,
                            local_cid,
                            inner_sweeps,
                        )
                        base += wp.int32(block_dim)
                else:
                    count = end - start
                    base = local_tid
                    while base < count:
                        cid = world_element_ids_by_color[start + base]
                        _dispatch_iterate_cid(
                            constraints,
                            contact_cols,
                            bodies,
                            particles,
                            cc,
                            contacts,
                            copy_state,
                            num_bodies,
                            idt,
                            sor_boost,
                            cid,
                            num_joints,
                            inner_sweeps,
                        )
                        base += wp.int32(block_dim)

                _sync_threads()
                c += wp.int32(1)
            it_outer += wp.int32(1)

    return kernel


@functools.cache
def _make_block_world_relax_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    enable_column_timers: bool = False,
    family_split: bool = False,
    block_dim: int = 128,
):
    """Build a multi-world bias-off relax kernel with one block per world."""
    (
        _dispatch_iterate_cid,
        _dispatch_iterate_joint,
        _dispatch_iterate_contact,
    ) = _make_multiworld_rigid_iterate_dispatch_funcs(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        enable_column_timers=enable_column_timers,
        use_bias=False,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        world_element_ids_by_color: wp.array[wp.int32],
        world_color_starts: wp.array2d[wp.int32],
        world_color_family_starts: wp.array2d[wp.int32],
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        copy_state: CopyStateContainer,
    ):
        tid = wp.tid()
        local_tid = tid % wp.int32(block_dim)
        world_id = tid / wp.int32(block_dim)
        if world_id >= num_worlds:
            return

        n_colors = world_num_colors[world_id]
        world_base = world_csr_offsets[world_id]

        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + wp.int32(1)]

            if wp.static(family_split):
                family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
                joint_start = world_base + world_color_family_starts[world_id, family_base]
                contact_start = world_base + world_color_family_starts[world_id, family_base + wp.int32(1)]
                count_joints = contact_start - joint_start

                base = local_tid
                while base < count_joints:
                    cid = world_element_ids_by_color[joint_start + base]
                    _dispatch_iterate_joint(
                        constraints,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid,
                        num_iterations,
                    )
                    base += wp.int32(block_dim)

                count_contacts = end - contact_start
                base = local_tid
                while base < count_contacts:
                    cid = world_element_ids_by_color[contact_start + base]
                    local_cid = cid - num_joints
                    _dispatch_iterate_contact(
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        local_cid,
                        num_iterations,
                    )
                    base += wp.int32(block_dim)
            else:
                count = end - start
                base = local_tid
                while base < count:
                    cid = world_element_ids_by_color[start + base]
                    _dispatch_iterate_cid(
                        constraints,
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid,
                        num_joints,
                        num_iterations,
                    )
                    base += wp.int32(block_dim)

            _sync_threads()
            c += wp.int32(1)

    return kernel


# Fast-tail kernels are no longer eagerly built at module import. Callers
# should use :func:`get_fast_tail_kernel` (or hit module ``__getattr__`` for
# the legacy names) so only the revolute variant the scene actually needs
# gets compiled.


@wp.kernel(enable_backward=False, module="unique")
def _zero_constraint_time_us_kernel(
    constraints: ConstraintContainer,
    num_active: wp.array[wp.int32],
    adbs_off: wp.int32,
    cloth_tri_off: wp.int32,
    cloth_bend_off: wp.int32,
    soft_tet_off: wp.int32,
    soft_hex_off: wp.int32,
    num_joints: wp.int32,
    num_cloth_triangles: wp.int32,
    num_cloth_bending: wp.int32,
    num_soft_tetrahedra: wp.int32,
    num_soft_hexahedra: wp.int32,
):
    """Zero every constraint column's ``time_us`` slot at step start.

    Walks ``[0, num_active)`` once and selects the correct per-schema
    dword offset from the constraint-type tag (dword 0)."""
    cid = wp.tid()
    total = num_active[0]
    if cid >= total:
        return
    if cid < num_joints:
        off = adbs_off
    elif cid < num_joints + num_cloth_triangles:
        off = cloth_tri_off
    elif cid < num_joints + num_cloth_triangles + num_cloth_bending:
        off = cloth_bend_off
    elif cid < num_joints + num_cloth_triangles + num_cloth_bending + num_soft_tetrahedra:
        off = soft_tet_off
    elif cid < num_joints + num_cloth_triangles + num_cloth_bending + num_soft_tetrahedra + num_soft_hexahedra:
        off = soft_hex_off
    else:
        return
    constraints.data[off, cid] = wp.float32(0.0)


@wp.kernel(enable_backward=False, module="unique")
def _zero_contact_time_us_kernel(
    contact_cols: ContactColumnContainer,
    num_columns: wp.int32,
    off: wp.int32,
):
    """Zero every contact column's ``time_us`` slot at step start."""
    local_cid = wp.tid()
    if local_cid >= num_columns:
        return
    contact_cols.data[off, local_cid] = wp.float32(0.0)


@wp.kernel(enable_backward=False, module="unique")
def _reduce_constraint_time_us_kernel(
    constraints: ConstraintContainer,
    adbs_off: wp.int32,
    cloth_tri_off: wp.int32,
    cloth_bend_off: wp.int32,
    soft_tet_off: wp.int32,
    soft_hex_off: wp.int32,
    num_joints: wp.int32,
    num_cloth_triangles: wp.int32,
    num_cloth_bending: wp.int32,
    num_soft_tetrahedra: wp.int32,
    num_soft_hexahedra: wp.int32,
    totals: wp.array[wp.float32],
):
    """Atomic-sum every constraint column's ``time_us`` slot into
    ``totals[0..5]`` = (joints, cloth_tri, cloth_bend, soft_tet, contacts, soft_hex).

    Slot 4 (contacts) is filled by :func:`_reduce_contact_time_us_kernel`."""
    cid = wp.tid()
    if cid < num_joints:
        wp.atomic_add(totals, 0, constraints.data[adbs_off, cid])
    elif cid < num_joints + num_cloth_triangles:
        wp.atomic_add(totals, 1, constraints.data[cloth_tri_off, cid])
    elif cid < num_joints + num_cloth_triangles + num_cloth_bending:
        wp.atomic_add(totals, 2, constraints.data[cloth_bend_off, cid])
    elif cid < num_joints + num_cloth_triangles + num_cloth_bending + num_soft_tetrahedra:
        wp.atomic_add(totals, 3, constraints.data[soft_tet_off, cid])
    elif cid < num_joints + num_cloth_triangles + num_cloth_bending + num_soft_tetrahedra + num_soft_hexahedra:
        wp.atomic_add(totals, 5, constraints.data[soft_hex_off, cid])


@wp.kernel(enable_backward=False, module="unique")
def _reduce_contact_time_us_kernel(
    contact_cols: ContactColumnContainer,
    num_columns: wp.int32,
    off: wp.int32,
    totals: wp.array[wp.float32],
):
    """Atomic-sum every contact column's ``time_us`` slot into
    ``totals[4]``."""
    local_cid = wp.tid()
    if local_cid >= num_columns:
        return
    wp.atomic_add(totals, 4, contact_cols.data[off, local_cid])


def get_block_world_kernel(
    *,
    kind: str,
    revolute_only: bool,
    has_joints: bool = True,
    has_contacts: bool = True,
    has_sleeping: bool = False,
    has_soft_contact_pd: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    family_split: bool = False,
    block_dim: int = 128,
):
    """Lazy block-per-world kernel builder.

    ``kind`` is ``"prepare_plus_iterate"`` or ``"relax"``. This scheduler is
    intended for multi-world rigid scenes where one block can keep a world's
    colors busy without the short lane groups used by fast-tail.
    """
    if kind == "prepare_plus_iterate":
        return _make_block_world_prepare_plus_iterate_kernel(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            cached_prepare=cached_prepare,
            enable_column_timers=enable_column_timers,
            family_split=family_split,
            block_dim=block_dim,
        )
    if kind == "relax":
        return _make_block_world_relax_kernel(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            enable_column_timers=enable_column_timers,
            family_split=family_split,
            block_dim=block_dim,
        )
    raise ValueError(f"unknown block-world kernel kind: {kind!r}")


def get_fast_tail_kernel(
    *,
    kind: str,
    revolute_only: bool,
    has_joints: bool = True,
    has_contacts: bool = True,
    has_sleeping: bool = False,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_only: bool = False,
    cloth_only: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
):
    """Lazy fast-tail kernel builder. ``kind`` is ``"prepare_plus_iterate"``
    or ``"relax"``. Each (kind, revolute_only, has_joints,
    has_contacts, has_sleeping, cached_prepare, enable_column_timers,
    fixed_tpw, guard_tpw) tuple is cached
    after first build by the underlying factory's ``functools.cache``. ``fixed_tpw=0``
    keeps the graph-capture-safe dynamic threads-per-world buffer read;
    ``guard_tpw`` keeps fixed variants selectable in auto mode."""
    if kind == "prepare_plus_iterate":
        return _make_fast_tail_prepare_plus_iterate_kernel(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            cloth_support=cloth_support,
            soft_tet_only=soft_tet_only,
            cloth_only=cloth_only,
            cached_prepare=cached_prepare,
            enable_column_timers=enable_column_timers,
            fixed_tpw=fixed_tpw,
            guard_tpw=guard_tpw,
            family_split=family_split,
        )
    if kind == "relax":
        return _make_fast_tail_relax_kernel(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            cloth_support=cloth_support,
            soft_tet_only=soft_tet_only,
            cloth_only=cloth_only,
            enable_column_timers=enable_column_timers,
            fixed_tpw=fixed_tpw,
            guard_tpw=guard_tpw,
            family_split=family_split,
        )
    raise ValueError(f"unknown fast-tail kernel kind: {kind!r}")


@wp.func
def _contact_soft_tet_active_nodes(
    cc: ContactContainer,
    contact_first: wp.int32,
    contact_count: wp.int32,
    is_side1: wp.bool,
):
    use0 = bool(False)
    use1 = bool(False)
    use2 = bool(False)
    use3 = bool(False)
    for i in range(contact_count):
        k = contact_first + i
        bary = cc_get_side1_bary(cc, k) if is_side1 else cc_get_side0_bary(cc, k)
        w0 = bary[0]
        w1 = bary[1]
        w2 = bary[2]
        w3 = wp.float32(1.0) - w0 - w1 - w2
        if w0 != wp.float32(0.0):
            use0 = bool(True)
        if w1 != wp.float32(0.0):
            use1 = bool(True)
        if w2 != wp.float32(0.0):
            use2 = bool(True)
        if w3 != wp.float32(0.0):
            use3 = bool(True)
    return use0, use1, use2, use3


@wp.kernel(enable_backward=False, module="unique")
def _constraints_to_elements_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    cc: ContactContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_constraints: wp.array[wp.int32],
    num_joints: wp.int32,
    num_cloth_triangles: wp.int32,
    num_cloth_bending: wp.int32,
    num_soft_tetrahedra: wp.int32,
    num_soft_hexahedra: wp.int32,
    num_bodies: wp.int32,
    elements: wp.array[ElementInteractionData],
    element_family: wp.array[wp.int32],
):
    """Project active constraints into ElementInteractionData. Static bodies
    collapse to -1; the dynamic body compacts to slot 0.

    Cloth-triangle, cloth-bending and soft-tetrahedron constraints
    emit 3- or 4-member elements with unified body-or-particle
    indices: rigid bodies live at ``[0, num_bodies)`` and particles at
    ``[num_bodies, num_bodies + num_particles)``. The partitioner sees
    the unified-index nodes uniformly so the same Jones-Plassmann /
    greedy coloring pass colours joints, contacts, cloth-triangles,
    cloth-bending, and soft-tetrahedra together.
    """
    tid = wp.tid()
    n = num_constraints[0]
    if tid >= n:
        return
    if tid < num_joints:
        element_family[tid] = wp.int32(0)
        b1 = constraint_get_body1(constraints, tid)
        b2 = constraint_get_body2(constraints, tid)
        if b1 >= 0 and bodies.inverse_mass[b1] == 0.0:
            b1 = -1
        if b2 >= 0 and bodies.inverse_mass[b2] == 0.0:
            b2 = -1
        # Adjacency loop stops on the first -1, so compact non-negative ids first.
        if b1 < 0 and b2 >= 0:
            b1 = b2
            b2 = -1
        elements[tid] = element_interaction_data_make(b1, b2, -1, -1, -1, -1, -1, -1)
        return
    if tid < num_joints + num_cloth_triangles:
        element_family[tid] = wp.int32(2)
        # Cloth-triangle: three unified-index particle endpoints already
        # stored in body1/body2/body3 (populate kernel did the +num_bodies
        # shift). Pinned particles (inverse_mass == 0) collapse to -1 so
        # the partitioner doesn't inflate adjacency for static anchors.
        b1 = constraint_get_body1(constraints, tid)
        b2 = constraint_get_body2(constraints, tid)
        b3 = read_int(constraints, _CLOTH_TRIANGLE_OFF_BODY3, tid)
        if b1 >= num_bodies and particles.inverse_mass[b1 - num_bodies] == 0.0:
            b1 = -1
        if b2 >= num_bodies and particles.inverse_mass[b2 - num_bodies] == 0.0:
            b2 = -1
        if b3 >= num_bodies and particles.inverse_mass[b3 - num_bodies] == 0.0:
            b3 = -1
        # Compact: drop -1s so the adjacency loop sees a contiguous prefix.
        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        slot2 = wp.int32(-1)
        if b1 >= 0:
            slot0 = b1
        for cand in range(2):
            v = b2
            if cand == 1:
                v = b3
            if v < 0:
                continue
            if slot0 < 0:
                slot0 = v
            elif slot1 < 0:
                slot1 = v
            else:
                slot2 = v
        elements[tid] = element_interaction_data_make(slot0, slot1, slot2, -1, -1, -1, -1, -1)
        return
    if tid < num_joints + num_cloth_triangles + num_cloth_bending:
        element_family[tid] = wp.int32(3)
        # Cloth-bending: 4 unified-index particle endpoints. body1 / body2
        # are the opposite vertices; body3 / body4 are the shared edge.
        b1 = constraint_get_body1(constraints, tid)
        b2 = constraint_get_body2(constraints, tid)
        b3 = read_int(constraints, _CLOTH_BENDING_OFF_BODY3, tid)
        b4 = read_int(constraints, _CLOTH_BENDING_OFF_BODY4, tid)
        if b1 >= num_bodies and particles.inverse_mass[b1 - num_bodies] == 0.0:
            b1 = -1
        if b2 >= num_bodies and particles.inverse_mass[b2 - num_bodies] == 0.0:
            b2 = -1
        if b3 >= num_bodies and particles.inverse_mass[b3 - num_bodies] == 0.0:
            b3 = -1
        if b4 >= num_bodies and particles.inverse_mass[b4 - num_bodies] == 0.0:
            b4 = -1
        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        slot2 = wp.int32(-1)
        slot3 = wp.int32(-1)
        if b1 >= 0:
            slot0 = b1
        for cand in range(3):
            v = b2
            if cand == 1:
                v = b3
            elif cand == 2:
                v = b4
            if v < 0:
                continue
            if slot0 < 0:
                slot0 = v
            elif slot1 < 0:
                slot1 = v
            elif slot2 < 0:
                slot2 = v
            else:
                slot3 = v
        elements[tid] = element_interaction_data_make(slot0, slot1, slot2, slot3, -1, -1, -1, -1)
        return
    if tid < num_joints + num_cloth_triangles + num_cloth_bending + num_soft_tetrahedra:
        element_family[tid] = wp.int32(4)
        # Soft-tetrahedron: four unified-index particle endpoints stored
        # in body1/body2/body3/body4 (populate kernel did the +num_bodies
        # shift). Pinned particles collapse to -1 so the partitioner
        # doesn't inflate adjacency for static anchors.
        b1 = constraint_get_body1(constraints, tid)
        b2 = constraint_get_body2(constraints, tid)
        b3 = read_int(constraints, _SOFT_TET_OFF_BODY3, tid)
        b4 = read_int(constraints, _SOFT_TET_OFF_BODY4, tid)
        if b1 >= num_bodies and particles.inverse_mass[b1 - num_bodies] == 0.0:
            b1 = -1
        if b2 >= num_bodies and particles.inverse_mass[b2 - num_bodies] == 0.0:
            b2 = -1
        if b3 >= num_bodies and particles.inverse_mass[b3 - num_bodies] == 0.0:
            b3 = -1
        if b4 >= num_bodies and particles.inverse_mass[b4 - num_bodies] == 0.0:
            b4 = -1
        # Compact: drop -1s so the adjacency loop sees a contiguous prefix.
        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        slot2 = wp.int32(-1)
        slot3 = wp.int32(-1)
        if b1 >= 0:
            slot0 = b1
        for cand in range(3):
            v = b2
            if cand == 1:
                v = b3
            elif cand == 2:
                v = b4
            if v < 0:
                continue
            if slot0 < 0:
                slot0 = v
            elif slot1 < 0:
                slot1 = v
            elif slot2 < 0:
                slot2 = v
            else:
                slot3 = v
        elements[tid] = element_interaction_data_make(slot0, slot1, slot2, slot3, -1, -1, -1, -1)
        return
    if tid < num_joints + num_cloth_triangles + num_cloth_bending + num_soft_tetrahedra + num_soft_hexahedra:
        element_family[tid] = wp.int32(5)
        # Soft-hexahedron: 8 unified-index particle endpoints in body1..body8.
        # Pinned particles collapse to -1 (same convention as tet/cloth).
        # ElementInteractionData's vec8i slots fit a full hex without
        # compaction-overflow risk; we still compact -1s so the
        # partitioner's adjacency loop sees a contiguous prefix.
        h0 = constraint_get_body1(constraints, tid)
        h1 = constraint_get_body2(constraints, tid)
        h2 = read_int(constraints, _SOFT_HEX_OFF_BODY3, tid)
        h3 = read_int(constraints, _SOFT_HEX_OFF_BODY4, tid)
        h4 = read_int(constraints, _SOFT_HEX_OFF_BODY5, tid)
        h5 = read_int(constraints, _SOFT_HEX_OFF_BODY6, tid)
        h6 = read_int(constraints, _SOFT_HEX_OFF_BODY7, tid)
        h7 = read_int(constraints, _SOFT_HEX_OFF_BODY8, tid)
        if h0 >= num_bodies and particles.inverse_mass[h0 - num_bodies] == 0.0:
            h0 = -1
        if h1 >= num_bodies and particles.inverse_mass[h1 - num_bodies] == 0.0:
            h1 = -1
        if h2 >= num_bodies and particles.inverse_mass[h2 - num_bodies] == 0.0:
            h2 = -1
        if h3 >= num_bodies and particles.inverse_mass[h3 - num_bodies] == 0.0:
            h3 = -1
        if h4 >= num_bodies and particles.inverse_mass[h4 - num_bodies] == 0.0:
            h4 = -1
        if h5 >= num_bodies and particles.inverse_mass[h5 - num_bodies] == 0.0:
            h5 = -1
        if h6 >= num_bodies and particles.inverse_mass[h6 - num_bodies] == 0.0:
            h6 = -1
        if h7 >= num_bodies and particles.inverse_mass[h7 - num_bodies] == 0.0:
            h7 = -1
        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        slot2 = wp.int32(-1)
        slot3 = wp.int32(-1)
        slot4 = wp.int32(-1)
        slot5 = wp.int32(-1)
        slot6 = wp.int32(-1)
        slot7 = wp.int32(-1)
        if h0 >= 0:
            slot0 = h0
        for cand in range(7):
            v = h1
            if cand == 1:
                v = h2
            elif cand == 2:
                v = h3
            elif cand == 3:
                v = h4
            elif cand == 4:
                v = h5
            elif cand == 5:
                v = h6
            elif cand == 6:
                v = h7
            if v < 0:
                continue
            if slot0 < 0:
                slot0 = v
            elif slot1 < 0:
                slot1 = v
            elif slot2 < 0:
                slot2 = v
            elif slot3 < 0:
                slot3 = v
            elif slot4 < 0:
                slot4 = v
            elif slot5 < 0:
                slot5 = v
            elif slot6 < 0:
                slot6 = v
            else:
                slot7 = v
        elements[tid] = element_interaction_data_make(slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7)
        return
    local_cid = tid - num_joints - num_cloth_triangles - num_cloth_bending - num_soft_tetrahedra - num_soft_hexahedra
    element_family[tid] = wp.int32(1)
    b1 = contact_get_body1(contact_cols, local_cid)
    b2 = contact_get_body2(contact_cols, local_cid)
    side0_kind = contact_get_side0_kind(contact_cols, local_cid)
    side1_kind = contact_get_side1_kind(contact_cols, local_cid)
    side0_extra = contact_get_side0_nodes_extra(contact_cols, local_cid)
    side1_extra = contact_get_side1_nodes_extra(contact_cols, local_cid)

    contact_first = contact_get_contact_first(contact_cols, local_cid)
    contact_count = contact_get_contact_count(contact_cols, local_cid)
    side0_use0 = bool(True)
    side0_use1 = bool(True)
    side0_use2 = bool(True)
    side0_use3 = bool(True)
    side1_use0 = bool(True)
    side1_use1 = bool(True)
    side1_use2 = bool(True)
    side1_use3 = bool(True)
    if side0_kind == wp.int32(2):  # SOFT_TETRAHEDRON
        side0_use0, side0_use1, side0_use2, side0_use3 = _contact_soft_tet_active_nodes(
            cc, contact_first, contact_count, False
        )
    if side1_kind == wp.int32(2):  # SOFT_TETRAHEDRON
        side1_use0, side1_use1, side1_use2, side1_use3 = _contact_soft_tet_active_nodes(
            cc, contact_first, contact_count, True
        )

    # Resolve a unified-index node to ``-1`` when its inverse mass is
    # zero (anchored). The lookup container depends on the side's
    # kind: rigid -> bodies; cloth -> particles (subtract num_bodies
    # to land in the particle SoA). ``b < 0`` is the "no node" case
    # (rigid sides without a body, e.g. world-attached shapes).
    #
    # KINEMATIC rigid bodies have ``inverse_mass == 0`` (the solver
    # treats them as immovable rails) but we keep them as graph nodes
    # so the per-step sleeping pass can spot sleeping-vs-kinematic
    # contacts and wake the impacted island (e.g. a camera collider
    # moving into a sleeping stack). Pure STATIC bodies still
    # collapse to -1.
    if b1 >= 0:
        if side0_kind == wp.int32(2) and not side0_use0:
            b1 = -1
        elif side0_kind == wp.int32(0):
            if bodies.inverse_mass[b1] == 0.0 and bodies.motion_type[b1] != MOTION_KINEMATIC:
                b1 = -1
        else:
            if particles.inverse_mass[b1 - num_bodies] == 0.0:
                b1 = -1
    if b2 >= 0:
        if side1_kind == wp.int32(2) and not side1_use0:
            b2 = -1
        elif side1_kind == wp.int32(0):
            if bodies.inverse_mass[b2] == 0.0 and bodies.motion_type[b2] != MOTION_KINEMATIC:
                b2 = -1
        else:
            if particles.inverse_mass[b2 - num_bodies] == 0.0:
                b2 = -1

    # Resolve up to three extra nodes per side. Rigid sides leave all
    # extras at -1; cloth-tri sides populate two; soft-tet sides populate
    # only the nonzero-barycentric nodes the iterate can read or write.
    e0a = wp.int32(-1)
    e0b = wp.int32(-1)
    e0c = wp.int32(-1)
    if side0_kind == wp.int32(1):  # CLOTH_TRIANGLE
        e0a = side0_extra[0]
        e0b = side0_extra[1]
        if e0a >= 0 and particles.inverse_mass[e0a - num_bodies] == 0.0:
            e0a = -1
        if e0b >= 0 and particles.inverse_mass[e0b - num_bodies] == 0.0:
            e0b = -1
    elif side0_kind == wp.int32(2):  # SOFT_TETRAHEDRON
        if side0_use1:
            e0a = side0_extra[0]
        if side0_use2:
            e0b = side0_extra[1]
        if side0_use3:
            e0c = side0_extra[2]
        if e0a >= 0 and particles.inverse_mass[e0a - num_bodies] == 0.0:
            e0a = -1
        if e0b >= 0 and particles.inverse_mass[e0b - num_bodies] == 0.0:
            e0b = -1
        if e0c >= 0 and particles.inverse_mass[e0c - num_bodies] == 0.0:
            e0c = -1
    e1a = wp.int32(-1)
    e1b = wp.int32(-1)
    e1c = wp.int32(-1)
    if side1_kind == wp.int32(1):  # CLOTH_TRIANGLE
        e1a = side1_extra[0]
        e1b = side1_extra[1]
        if e1a >= 0 and particles.inverse_mass[e1a - num_bodies] == 0.0:
            e1a = -1
        if e1b >= 0 and particles.inverse_mass[e1b - num_bodies] == 0.0:
            e1b = -1
    elif side1_kind == wp.int32(2):  # SOFT_TETRAHEDRON
        if side1_use1:
            e1a = side1_extra[0]
        if side1_use2:
            e1b = side1_extra[1]
        if side1_use3:
            e1c = side1_extra[2]
        if e1a >= 0 and particles.inverse_mass[e1a - num_bodies] == 0.0:
            e1a = -1
        if e1b >= 0 and particles.inverse_mass[e1b - num_bodies] == 0.0:
            e1b = -1
        if e1c >= 0 and particles.inverse_mass[e1c - num_bodies] == 0.0:
            e1c = -1

    # Compact: drop -1s into a contiguous prefix (the partitioner's
    # adjacency loop stops on the first -1). Up to 8 nodes per contact:
    # tet-tet = 4+4; tet-cloth = 4+3; tet-rigid = 4+1; cloth-cloth =
    # 3+3; cloth-rigid = 3+1; rigid-rigid = 1+1.
    s0 = wp.int32(-1)
    s1 = wp.int32(-1)
    s2 = wp.int32(-1)
    s3 = wp.int32(-1)
    s4 = wp.int32(-1)
    s5 = wp.int32(-1)
    s6 = wp.int32(-1)
    s7 = wp.int32(-1)
    cnt = wp.int32(0)
    for cand in range(8):
        v = wp.int32(-1)
        if cand == 0:
            v = b1
        elif cand == 1:
            v = b2
        elif cand == 2:
            v = e0a
        elif cand == 3:
            v = e0b
        elif cand == 4:
            v = e0c
        elif cand == 5:
            v = e1a
        elif cand == 6:
            v = e1b
        else:
            v = e1c
        if v < 0:
            continue
        if cnt == 0:
            s0 = v
        elif cnt == 1:
            s1 = v
        elif cnt == 2:
            s2 = v
        elif cnt == 3:
            s3 = v
        elif cnt == 4:
            s4 = v
        elif cnt == 5:
            s5 = v
        elif cnt == 6:
            s6 = v
        else:
            s7 = v
        cnt = cnt + 1
    elements[tid] = element_interaction_data_make(s0, s1, s2, s3, s4, s5, s6, s7)


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
    """Advance pose for dynamic bodies only AND refresh
    ``inverse_inertia_world`` from the just-rotated orientation in the
    same pass. Fusing the two reduces per-substep launches from 2 to 1,
    keeps the freshly-rotated quat in a register for the inertia
    rebuild, and lets the next relax / next substep solve see
    ``R * I^-1 * R^T`` aligned with the integrated pose. Kinematic
    bodies advance via lerp/slerp in
    :func:`_kinematic_interpolate_substep_kernel`."""
    i = wp.tid()
    mt = bodies.motion_type[i]
    if mt == MOTION_STATIC or mt == MOTION_KINEMATIC:
        return
    # Sleeping bodies must not drift. They may carry a small residual
    # velocity (anywhere below the per-island sleep threshold) at the
    # moment ``island_root`` is stamped; integrating that for many
    # substeps would slide the whole sleeping island visibly.
    if bodies.island_root[i] >= wp.int32(0):
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt
    q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
    q_new = wp.normalize(q_rot * bodies.orientation[i])
    bodies.orientation[i] = q_new
    # Refresh world-frame inverse inertia using the just-rotated quat
    # (one extra mat33 mul, no extra global load).
    r = wp.quat_to_matrix(q_new)
    bodies.inverse_inertia_world[i] = rotate_inertia(r, bodies.inverse_inertia[i])


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


# Per-step body kernels driven from PhoenXWorld.step.


@wp.kernel(enable_backward=False)
def _phoenx_apply_forces_and_gravity_kernel(
    bodies: BodyContainer,
    gravity: wp.array[wp.vec3f],
    substep_dt: wp.float32,
):
    """Per-body substep entry: snapshot pose into ``*_prev_substep``,
    set :attr:`access_mode`, then apply external forces + gravity to
    velocity (dynamic only). Force accumulators are zeroed in
    :func:`_phoenx_update_inertia_and_clear_forces_kernel` at end-of-step.

    The substep-start pose snapshot is the finite-diff anchor used by
    :mod:`newton._src.solvers.phoenx.access_mode` when a constraint
    flips a body between velocity- and position-level. It must run
    once per substep regardless of motion type so non-dynamic bodies
    also have a valid anchor.
    """
    i = wp.tid()
    bodies.position_prev_substep[i] = bodies.position[i]
    bodies.orientation_prev_substep[i] = bodies.orientation[i]
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        bodies.access_mode[i] = ACCESS_MODE_STATIC
        return
    if bodies.inverse_mass[i] == 0.0:
        bodies.access_mode[i] = ACCESS_MODE_STATIC
        return
    # Sleeping bodies: skip gravity + force application and present as
    # STATIC to the constraint solve so body_set_access_mode early-outs
    # on every constraint touch. Velocity stays at whatever value it
    # held when the island fell below the threshold (~ 0). ``island_root``
    # is always -1 (awake) when the sleeping pipeline is disabled.
    if bodies.island_root[i] >= 0:
        bodies.access_mode[i] = ACCESS_MODE_STATIC
        return
    bodies.access_mode[i] = ACCESS_MODE_VELOCITY_LEVEL
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
def _color_for_step(
    step_idx: wp.int32,
    n_colors: wp.int32,
    direction: wp.int32,
    max_colored_partitions: wp.int32,
) -> wp.int32:
    """Map a 0-based forward step index into the actual color index
    for this sweep. ``direction == 0`` => ascending order
    (``c = step_idx``); ``!= 0`` => reverse for non-overflow colours
    (symmetric Gauss-Seidel). When mass splitting is on
    (``max_colored_partitions >= 0``) the overflow bucket at index
    ``max_colored_partitions`` always runs last regardless of
    direction.

    Symmetric sweep alone is not enough to fix Kapla-tower drift
    under warm-start (it only swaps edge colours; middle colours
    stay near the middle of the sweep). It's kept as a building
    block for future experimentation; the actual drift fix lives on
    the warm-start cache side (see
    ``warm_start_periodic_invalidate_kernel`` and ``set_warm_start
    _rotate_skip``).
    """
    if direction == wp.int32(0):
        return step_idx
    # Reverse path.
    if max_colored_partitions >= wp.int32(0) and step_idx == max_colored_partitions:
        return max_colored_partitions  # overflow stays last
    n_regular = n_colors
    if max_colored_partitions >= wp.int32(0) and n_colors > max_colored_partitions:
        n_regular = max_colored_partitions
    if n_regular <= wp.int32(1):
        return step_idx
    return n_regular - wp.int32(1) - step_idx


@wp.func
def _singleworld_color_range(
    color_starts: wp.array[wp.int32],
    num_colors: wp.array[wp.int32],
    color_cursor: wp.array[wp.int32],
    sweep_direction: wp.array[wp.int32],
    max_colored_partitions: wp.int32,
):
    """Decode current colour's cid range from cursor. Returns
    ``(start, count, cursor, c)`` where ``c`` is the *actual* colour
    index (post-direction remap). When ``sweep_direction[0] != 0``
    the regular colours are visited in reverse order; the overflow
    bucket (if any) stays last."""
    cursor = color_cursor[0]
    n_colors = num_colors[0]
    step_idx = n_colors - cursor
    c = _color_for_step(step_idx, n_colors, sweep_direction[0], max_colored_partitions)
    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start
    return start, count, cursor, c


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
    sweep_direction: wp.array[wp.int32],
    max_colored_partitions: wp.int32,
):
    """:func:`_singleworld_color_range` taking the cursor as a register
    value. Returns ``(start, count, c)`` -- ``c`` is the direction-
    remapped colour index (see :func:`_color_for_step`)."""
    n_colors = num_colors[0]
    step_idx = n_colors - cursor
    c = _color_for_step(step_idx, n_colors, sweep_direction[0], max_colored_partitions)
    start = color_starts[c]
    end = color_starts[c + 1]
    count = end - start
    return start, count, c


# Single-world kernel factories: persistent (head) + single-block (fused tail)
# for prepare/iterate/relax x revolute_only/generic. ``phase`` and ``revolute_only``
# are compile-time so Warp constant-folds + dead-code-eliminates the unused branch.


@functools.cache
def _make_singleworld_rigid_contact_dispatch_func(
    *,
    has_mass_splitting: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    is_prepare: bool,
    is_cached_prepare: bool,
    use_bias: bool,
):
    @wp.func
    def _dispatch_rigid_contact(
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        local_cid: wp.int32,
        parallel_id: wp.int32,
    ):
        if wp.static(is_prepare):
            if wp.static(has_mass_splitting):
                if wp.static(has_soft_contact_pd):
                    contact_prepare_for_iteration(
                        contact_cols,
                        local_cid,
                        bodies,
                        particles,
                        num_bodies,
                        idt,
                        cc,
                        contacts,
                        copy_state,
                        parallel_id,
                    )
                else:
                    contact_prepare_for_iteration_no_soft_pd(
                        contact_cols,
                        local_cid,
                        bodies,
                        particles,
                        num_bodies,
                        idt,
                        cc,
                        contacts,
                        copy_state,
                        parallel_id,
                    )
            else:
                if wp.static(has_soft_contact_pd):
                    contact_prepare_for_iteration_lean(
                        contact_cols,
                        local_cid,
                        bodies,
                        particles,
                        num_bodies,
                        idt,
                        cc,
                        contacts,
                        copy_state,
                        parallel_id,
                    )
                else:
                    contact_prepare_for_iteration_lean_no_soft_pd(
                        contact_cols,
                        local_cid,
                        bodies,
                        particles,
                        num_bodies,
                        idt,
                        cc,
                        contacts,
                        copy_state,
                        parallel_id,
                    )
        elif wp.static(is_cached_prepare):
            contact_cached_warmstart_lean(
                contact_cols,
                local_cid,
                bodies,
                particles,
                num_bodies,
                idt,
                cc,
                contacts,
                copy_state,
                parallel_id,
            )
        else:
            if wp.static(has_mass_splitting):
                if wp.static(has_sleeping):
                    if wp.static(has_soft_contact_pd):
                        contact_iterate(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
                    else:
                        contact_iterate_no_soft_pd(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
                else:
                    if wp.static(has_soft_contact_pd):
                        contact_iterate_no_sleep(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
                    else:
                        contact_iterate_no_sleep_no_soft_pd(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
            else:
                if wp.static(has_sleeping):
                    if wp.static(has_soft_contact_pd):
                        contact_iterate_lean(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
                    else:
                        contact_iterate_lean_no_soft_pd(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
                else:
                    if wp.static(has_soft_contact_pd):
                        contact_iterate_lean_no_sleep(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
                    else:
                        contact_iterate_lean_no_sleep_no_soft_pd(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )

    return _dispatch_rigid_contact


@functools.cache
def _make_singleworld_dispatch_func(
    *,
    revolute_only: bool,
    cloth_support: bool,
    enable_column_timers: bool,
    soft_tet_only: bool,
    cloth_only: bool,
    has_joints: bool,
    has_mass_splitting: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    is_prepare: bool,
    is_cached_prepare: bool,
    use_bias: bool,
):
    """Per-cid dispatch helper used by both head and fused PGS kernels.

    Returns a ``@wp.func`` that routes one ``cid`` through the
    constraint-type dispatch tree (contact / soft-tet / cloth-tri /
    cloth-bending / joint). All static specialisation axes are captured
    here so the head + fused kernels can call this in a single line
    instead of duplicating the ~220-line tree.

    ``@functools.cache``-keyed on the static axes; every (axes-tuple)
    builds one function once.
    """

    _dispatch_rigid_contact = _make_singleworld_rigid_contact_dispatch_func(
        has_mass_splitting=has_mass_splitting,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
    )

    @wp.func
    def _dispatch_one_cid(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        num_joints: wp.int32,
        num_cloth_triangles: wp.int32,
        num_cloth_bending: wp.int32,
        num_soft_tetrahedra: wp.int32,
        num_soft_hexahedra: wp.int32,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        cid: wp.int32,
        parallel_id: wp.int32,
    ):
        # Opt-in per-cid wall-clock bracket. The %globaltimer reads and
        # atomic_add are dead-code-eliminated when
        # ``enable_column_timers=False`` thanks to ``wp.static``.
        _t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            _t0 = read_global_timer_ns()
        # Two-stage dispatch.
        # 1) Contacts live in a separate ContactColumnContainer (NOT in
        #    the joint-side ConstraintContainer), so contact cids must
        #    NOT do a `constraint_get_type` read -- the constraint
        #    container has no slot at the contact cid range. Use a
        #    cheap cid-range check.
        # 2) Joints + cloth share the ConstraintContainer; within that
        #    range we dispatch on the type tag (each schema stamps its
        #    type into dword 0 at populate time).
        dispatched = False
        if cid >= num_joints + num_cloth_triangles + num_cloth_bending + num_soft_tetrahedra + num_soft_hexahedra:
            local_cid = (
                cid - num_joints - num_cloth_triangles - num_cloth_bending - num_soft_tetrahedra - num_soft_hexahedra
            )
            if wp.static(cloth_support):
                # Mixed cloth/soft scenes still produce many rigid-rigid
                # contact columns. Route those through the lean rigid path;
                # only columns with a non-rigid endpoint need endpoint helpers.
                side0_kind = contact_get_side0_kind(contact_cols, local_cid)
                side1_kind = contact_get_side1_kind(contact_cols, local_cid)
                if side0_kind == wp.int32(0) and side1_kind == wp.int32(0):
                    _dispatch_rigid_contact(
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_bodies,
                        idt,
                        sor_boost,
                        local_cid,
                        parallel_id,
                    )
                else:
                    if wp.static(is_prepare):
                        contact_prepare_for_iteration_cloth_aware(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            copy_state,
                            parallel_id,
                        )
                    else:
                        contact_iterate_cloth_aware(
                            contact_cols,
                            local_cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            use_bias,
                            copy_state,
                            parallel_id,
                            sor_boost,
                        )
            else:
                _dispatch_rigid_contact(
                    contact_cols,
                    bodies,
                    particles,
                    cc,
                    contacts,
                    copy_state,
                    num_bodies,
                    idt,
                    sor_boost,
                    local_cid,
                    parallel_id,
                )
            dispatched = True
            if wp.static(enable_column_timers):
                contact_accumulate_time_us(contact_cols, local_cid, elapsed_us(_t0, read_global_timer_ns()))
        if not dispatched:
            if wp.static(cloth_only):
                # ConstraintContainer only holds cloth triangle and optional
                # bending rows, so avoid the ctype-tag load in the hot path.
                if cid < num_cloth_triangles:
                    if wp.static(is_prepare):
                        cloth_triangle_prepare_for_iteration_at(
                            constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                        )
                    else:
                        cloth_triangle_iterate_at(
                            constraints,
                            cid,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            parallel_id,
                            idt,
                            sor_boost,
                        )
                    if wp.static(enable_column_timers):
                        constraint_accumulate_time_us(
                            constraints,
                            CLOTH_TRIANGLE_TIME_US_OFFSET,
                            cid,
                            elapsed_us(_t0, read_global_timer_ns()),
                        )
                else:
                    if wp.static(is_prepare):
                        cloth_bending_prepare_for_iteration_at(
                            constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                        )
                    else:
                        cloth_bending_iterate_at(
                            constraints,
                            cid,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            parallel_id,
                            idt,
                            sor_boost,
                        )
                    if wp.static(enable_column_timers):
                        constraint_accumulate_time_us(
                            constraints,
                            CLOTH_BENDING_TIME_US_OFFSET,
                            cid,
                            elapsed_us(_t0, read_global_timer_ns()),
                        )
            elif wp.static(soft_tet_only):
                # Specialised path: ConstraintContainer only holds soft-tets
                # (no joints / cloth-tri / cloth-bend rows), so the ctype
                # read + 3-way compare is dead code. cid here is by
                # construction a soft-tet.
                if wp.static(is_prepare):
                    soft_tetrahedron_prepare_for_iteration_at(
                        constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                    )
                else:
                    soft_tetrahedron_iterate_at(
                        constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt, sor_boost
                    )
                if wp.static(enable_column_timers):
                    constraint_accumulate_time_us(
                        constraints, SOFT_TET_TIME_US_OFFSET, cid, elapsed_us(_t0, read_global_timer_ns())
                    )
            else:
                ctype = constraint_get_type(constraints, cid)
                if wp.static(cloth_support):
                    if ctype == CONSTRAINT_TYPE_CLOTH_TRIANGLE:
                        if wp.static(is_prepare):
                            cloth_triangle_prepare_for_iteration_at(
                                constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                            )
                        else:
                            cloth_triangle_iterate_at(
                                constraints,
                                cid,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                parallel_id,
                                idt,
                                sor_boost,
                            )
                        dispatched = True
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints,
                                CLOTH_TRIANGLE_TIME_US_OFFSET,
                                cid,
                                elapsed_us(_t0, read_global_timer_ns()),
                            )
                    elif ctype == CONSTRAINT_TYPE_SOFT_TETRAHEDRON:
                        if wp.static(is_prepare):
                            soft_tetrahedron_prepare_for_iteration_at(
                                constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                            )
                        else:
                            soft_tetrahedron_iterate_at(
                                constraints,
                                cid,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                parallel_id,
                                idt,
                                sor_boost,
                            )
                        dispatched = True
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints,
                                SOFT_TET_TIME_US_OFFSET,
                                cid,
                                elapsed_us(_t0, read_global_timer_ns()),
                            )
                    elif ctype == CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN:
                        if wp.static(is_prepare):
                            soft_tet_neohookean_prepare_for_iteration_at(
                                constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                            )
                        else:
                            soft_tet_neohookean_iterate_at(
                                constraints,
                                cid,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                parallel_id,
                                idt,
                                sor_boost,
                            )
                        dispatched = True
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints,
                                SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET,
                                cid,
                                elapsed_us(_t0, read_global_timer_ns()),
                            )
                    elif ctype == CONSTRAINT_TYPE_SOFT_HEXAHEDRON:
                        if wp.static(is_prepare):
                            soft_hexahedron_prepare_for_iteration_at(
                                constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                            )
                        else:
                            soft_hexahedron_iterate_at(
                                constraints,
                                cid,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                parallel_id,
                                idt,
                                sor_boost,
                            )
                        dispatched = True
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints,
                                SOFT_HEX_TIME_US_OFFSET,
                                cid,
                                elapsed_us(_t0, read_global_timer_ns()),
                            )
                    elif ctype == CONSTRAINT_TYPE_CLOTH_BENDING:
                        if wp.static(is_prepare):
                            cloth_bending_prepare_for_iteration_at(
                                constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                            )
                        else:
                            cloth_bending_iterate_at(
                                constraints,
                                cid,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                parallel_id,
                                idt,
                                sor_boost,
                            )
                        dispatched = True
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints,
                                CLOTH_BENDING_TIME_US_OFFSET,
                                cid,
                                elapsed_us(_t0, read_global_timer_ns()),
                            )
                # Joint (ADBS or revolute specialisation). The outer
                # ``if wp.static(has_joints):`` is a true compile-time
                # gate -- when the scene has zero joints the entire
                # 8-mode ADBS dispatcher (~5 kLOC of inlined code)
                # disappears from the kernel. The static MUST stand
                # alone as its own ``if``; compound conditions like
                # ``not dispatched and wp.static(...)`` do not
                # constant-fold reliably in Warp's codegen.
                if wp.static(has_joints):
                    if not dispatched:
                        # Joint constraints are velocity-level. In cloth scenes
                        # (``cloth_support=True``) a prior position-level write
                        # (cloth iterate) may have left these bodies in
                        # ``ACCESS_MODE_POSITION_LEVEL``; flip them back. The
                        # ``wp.static`` gate compile-time-eliminates both the
                        # b1/b2 reads and the inlined flip body for rigid-only
                        # scenes, restoring the dr_legs hot path.
                        if wp.static(cloth_support):
                            _b1_flip = constraint_get_body1(constraints, cid)
                            _b2_flip = constraint_get_body2(constraints, cid)
                            body_set_access_mode(bodies, _b1_flip, ACCESS_MODE_VELOCITY_LEVEL, idt)
                            body_set_access_mode(bodies, _b2_flip, ACCESS_MODE_VELOCITY_LEVEL, idt)
                        if wp.static(is_cached_prepare):
                            if wp.static(revolute_only):
                                revolute_cached_warmstart(
                                    constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                                )
                            else:
                                actuated_double_ball_socket_cached_warmstart(
                                    constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                                )
                        elif wp.static(is_prepare):
                            if wp.static(revolute_only):
                                revolute_prepare_for_iteration(
                                    constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                                )
                            else:
                                actuated_double_ball_socket_prepare_for_iteration(
                                    constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
                                )
                        else:
                            if wp.static(revolute_only):
                                revolute_iterate(
                                    constraints,
                                    cid,
                                    bodies,
                                    particles,
                                    copy_state,
                                    num_bodies,
                                    parallel_id,
                                    idt,
                                    sor_boost,
                                    use_bias,
                                )
                            else:
                                actuated_double_ball_socket_iterate(
                                    constraints,
                                    cid,
                                    bodies,
                                    particles,
                                    copy_state,
                                    num_bodies,
                                    parallel_id,
                                    idt,
                                    sor_boost,
                                    use_bias,
                                )
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints, ADBS_TIME_US_OFFSET, cid, elapsed_us(_t0, read_global_timer_ns())
                            )

    return _dispatch_one_cid


@functools.cache
def _make_singleworld_persistent_kernel(
    *,
    phase: str,
    revolute_only: bool,
    cloth_support: bool,
    enable_column_timers: bool = False,
    soft_tet_only: bool = False,
    cloth_only: bool = False,
    has_joints: bool = True,
    has_mass_splitting: bool = True,
    has_sleeping: bool = True,
    has_soft_contact_pd: bool = True,
):
    """Persistent-grid PGS kernel for the requested phase + specialisation.

    Per-cid dispatch reads the constraint type tag (dword 0) and routes
    to the matching ``@wp.func`` -- joint, cloth-triangle, or contact.
    The cloth branch is gated by the compile-time ``cloth_support``
    flag so rigid-only scenes get a binary with no cloth code at all.

    ``soft_tet_only=True`` skips the ctype dispatch tree: when the
    ConstraintContainer only ever holds soft-tetrahedron rows (no
    joints, no cloth triangles, no cloth bending), every non-contact
    cid is by construction a soft-tet. Eliminates one dword load
    (constraint type) and the cloth_tri / cloth_bend / joint
    fallback branches from the iterate / prepare / relax hot path.

    ``enable_column_timers`` is a static axis: when True, each per-cid
    dispatch is bracketed with ``%globaltimer`` reads and the elapsed
    microseconds are atomic-added to the column's ``time_us`` slot.
    """
    is_prepare = phase == "prepare"
    is_cached_prepare = phase == "cached_prepare"
    is_iterate = phase == "iterate"
    use_bias = is_iterate  # iterate ON, relax OFF (prepare ignores)

    _dispatch_one_cid = _make_singleworld_dispatch_func(
        revolute_only=revolute_only,
        cloth_support=cloth_support,
        enable_column_timers=enable_column_timers,
        soft_tet_only=soft_tet_only,
        cloth_only=cloth_only,
        has_joints=has_joints,
        has_mass_splitting=has_mass_splitting,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        element_ids_by_color: wp.array[wp.int32],
        color_starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        color_cursor: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_joints: wp.int32,
        num_cloth_triangles: wp.int32,
        num_cloth_bending: wp.int32,
        num_soft_tetrahedra: wp.int32,
        num_soft_hexahedra: wp.int32,
        num_bodies: wp.int32,
        total_num_threads: wp.int32,
        fuse_threshold: wp.int32,
        head_active: wp.array[wp.int32],
        copy_state: CopyStateContainer,
        max_colored_partitions: wp.int32,
        ms_batch_size: wp.int32,
        sweep_direction: wp.array[wp.int32],
    ):
        tid = wp.tid()
        if color_cursor[0] <= 0:
            if tid == 0:
                head_active[0] = 0
            return
        start, count, cursor, c = _singleworld_color_range(
            color_starts, num_colors, color_cursor, sweep_direction, max_colored_partitions
        )

        if count <= fuse_threshold:
            if tid == 0:
                head_active[0] = 0
            return

        # Overflow detection. ``c`` is the direction-remapped colour
        # index returned by ``_singleworld_color_range``; the overflow
        # bucket lives at ``max_colored_partitions`` and stays there
        # regardless of sweep direction (see ``_color_for_step``).
        # Overflow constraints are grouped into batches of
        # ``ms_batch_size`` consecutive CSR slots; one thread processes
        # a whole batch sequentially (Gauss-Seidel within the batch on
        # a shared slot) while across batches processing is parallel
        # (Jacobi via distinct slots). Regular colours stay at
        # ``parallel_id=0`` with one thread per constraint (independent
        # set, no need for splitting).
        is_overflow_color = max_colored_partitions >= wp.int32(0) and c == max_colored_partitions

        # For overflow: each thread covers ``ms_batch_size`` consecutive
        # CSR slots starting at ``tid * ms_batch_size``, grid-stride
        # by ``total_num_threads * ms_batch_size``. For regular:
        # one slot per thread, grid-stride by total_num_threads.
        thread_start = tid
        stride = total_num_threads
        if is_overflow_color:
            thread_start = tid * ms_batch_size
            stride = total_num_threads * ms_batch_size

        for t in range(thread_start, count, stride):
            inner_end = wp.int32(1)
            if is_overflow_color:
                inner_end = ms_batch_size
            for inner in range(inner_end):
                t_slot = t + inner
                if t_slot >= count:
                    break
                parallel_id = wp.int32(0)
                if is_overflow_color:
                    # Batch index = partition_key stamped by emit.
                    parallel_id = t_slot / ms_batch_size
                cid = read1d_i32(element_ids_by_color, start + t_slot)
                _dispatch_one_cid(
                    constraints,
                    contact_cols,
                    bodies,
                    particles,
                    cc,
                    contacts,
                    copy_state,
                    num_joints,
                    num_cloth_triangles,
                    num_cloth_bending,
                    num_soft_tetrahedra,
                    num_soft_hexahedra,
                    num_bodies,
                    idt,
                    sor_boost,
                    cid,
                    parallel_id,
                )

        if tid == 0:
            color_cursor[0] = cursor - 1

    return kernel


@functools.cache
def _make_singleworld_fused_kernel(
    *,
    phase: str,
    revolute_only: bool,
    cloth_support: bool,
    enable_column_timers: bool = False,
    soft_tet_only: bool = False,
    cloth_only: bool = False,
    has_joints: bool = True,
    has_mass_splitting: bool = True,
    has_sleeping: bool = True,
    has_soft_contact_pd: bool = True,
):
    """Single-block tail-fused PGS kernel; same axes as
    :func:`_make_singleworld_persistent_kernel`."""
    is_prepare = phase == "prepare"
    is_cached_prepare = phase == "cached_prepare"
    is_iterate = phase == "iterate"
    use_bias = is_iterate

    _dispatch_one_cid = _make_singleworld_dispatch_func(
        revolute_only=revolute_only,
        cloth_support=cloth_support,
        enable_column_timers=enable_column_timers,
        soft_tet_only=soft_tet_only,
        cloth_only=cloth_only,
        has_joints=has_joints,
        has_mass_splitting=has_mass_splitting,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        element_ids_by_color: wp.array[wp.int32],
        color_starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        color_cursor: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_joints: wp.int32,
        num_cloth_triangles: wp.int32,
        num_cloth_bending: wp.int32,
        num_soft_tetrahedra: wp.int32,
        num_soft_hexahedra: wp.int32,
        num_bodies: wp.int32,
        fuse_threshold: wp.int32,
        copy_state: CopyStateContainer,
        max_colored_partitions: wp.int32,
        ms_batch_size: wp.int32,
        sweep_direction: wp.array[wp.int32],
        # Re-arm the head_active flag for the next outer round so the
        # caller no longer needs a dedicated 1-thread ``_reset_head_active``
        # launch between rounds. The head kernel zeros this flag when it
        # can't make progress; the outer ``wp.capture_while(color_cursor)``
        # only re-enters the body when there's still work, so the unread
        # writes when ``cursor==0`` are harmless.
        head_active: wp.array[wp.int32],
    ):
        _block, lane = wp.tid()
        cursor = color_cursor[0]
        while cursor > 0:
            start, count, c = _singleworld_color_range_from_cursor(
                color_starts, num_colors, cursor, sweep_direction, max_colored_partitions
            )
            if count > fuse_threshold:
                break
            is_overflow_color = max_colored_partitions >= wp.int32(0) and c == max_colored_partitions
            # Overflow handling mirrors the head kernel's batched
            # dispatch (see ``_make_singleworld_persistent_kernel``):
            # one lane covers ``ms_batch_size`` consecutive CSR slots
            # sequentially (Gauss-Seidel within the batch on a shared
            # ``parallel_id``), and different lanes own different
            # batches in parallel (Jacobi). ``parallel_id == lane``
            # matches the ``partition_key = overflow_offset /
            # batch_size`` the emit stamped, so each lane reads /
            # writes its own ``copy_state`` slot -- no intra-block
            # race. The earlier shortcut bailed back to the head on
            # overflow + ``batch_size > 1`` and deadlocked when the
            # head also bailed to the tail on the same small overflow
            # colour.
            inner_steps = wp.int32(1)
            num_units = count
            if is_overflow_color:
                inner_steps = ms_batch_size
                num_units = (count + ms_batch_size - wp.int32(1)) / ms_batch_size
            if lane < num_units:
                parallel_id = wp.int32(0)
                if is_overflow_color:
                    parallel_id = lane
                t_slot_base = lane * inner_steps
                for inner in range(inner_steps):
                    t_slot = t_slot_base + inner
                    if t_slot >= count:
                        break
                    cid = read1d_i32(element_ids_by_color, start + t_slot)
                    _dispatch_one_cid(
                        constraints,
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        num_joints,
                        num_cloth_triangles,
                        num_cloth_bending,
                        num_soft_tetrahedra,
                        num_soft_hexahedra,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid,
                        parallel_id,
                    )
            _sync_threads()
            cursor = cursor - 1
        if lane == 0:
            color_cursor[0] = cursor
            # Re-arm head_active for the NEXT outer round; the head
            # kernel will zero it again if it has no work. Replaces the
            # dedicated 1-thread ``_reset_head_active_kernel`` launch the
            # caller used to issue between rounds.
            head_active[0] = 1

    return kernel


def get_singleworld_kernel(
    *,
    phase: str,
    fused: bool,
    revolute_only: bool,
    cloth_support: bool,
    enable_column_timers: bool = False,
    soft_tet_only: bool = False,
    cloth_only: bool = False,
    has_joints: bool = True,
    has_mass_splitting: bool = True,
    has_sleeping: bool = True,
    has_soft_contact_pd: bool = True,
):
    """Lazy singleworld kernel builder. Each axis combination is cached
    after first build by the underlying factory's ``functools.cache``."""
    factory = _make_singleworld_fused_kernel if fused else _make_singleworld_persistent_kernel
    return factory(
        phase=phase,
        revolute_only=revolute_only,
        cloth_support=cloth_support,
        enable_column_timers=enable_column_timers,
        soft_tet_only=soft_tet_only,
        cloth_only=cloth_only,
        has_joints=has_joints,
        has_mass_splitting=has_mass_splitting,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
    )
