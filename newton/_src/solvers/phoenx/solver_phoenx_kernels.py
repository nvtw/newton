# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels and dispatch factories for :class:`PhoenXWorld`."""

from __future__ import annotations

import functools

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
)
from newton._src.solvers.phoenx.body import (
    MOTION_ARTICULATED,
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
    body_set_access_mode,
    mat33_from_sym6,
    sym6_from_mat33,
)
from newton._src.solvers.phoenx.cloth_collision import (
    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE,
    SHAPE_ENDPOINT_KIND_RIGID,
    SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON,
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
    contact_iterate_patch_lean_no_sleep,
    contact_iterate_patch_lean_no_sleep_no_soft_pd,
    contact_iterate_patch_multi,
    contact_iterate_patch_multi_no_soft_pd,
    contact_prepare_for_iteration,
    contact_prepare_for_iteration_cloth_aware,
    contact_prepare_for_iteration_lean,
    contact_prepare_for_iteration_lean_no_soft_pd,
    contact_prepare_for_iteration_no_soft_pd,
    contact_prepare_for_iteration_packed_rows,
    contact_prepare_for_iteration_packed_rows_no_soft_pd,
    contact_prepare_for_iteration_patch_lean,
    contact_prepare_for_iteration_patch_lean_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_accumulate_time_us,
    constraint_get_body1,
    constraint_get_body2,
    read_int,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
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
    MAX_BODIES,
    ElementInteractionData,
    _lowest_set_bit,
    element_interaction_data_make,
)
from newton._src.solvers.phoenx.helpers.array_access import read1d_i32
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.timer import elapsed_us, read_global_timer_ns

# A PGS iteration is one complete traversal of all colors. Repeating a local
# row before adjacent colors observe its update wastes propagation and changes
# the meaning of ``solver_iterations``.
_FAST_TAIL_SOLVE_JOINT_INNER_SWEEPS = 1
_FAST_TAIL_SOLVE_CONTACT_INNER_SWEEPS = 1
_FAST_TAIL_SOLVE_OUTER_ITERATION_CHUNK = 1
_BLOCK_WORLD_SOLVE_INNER_SWEEPS = 1

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
    "_choose_fast_tail_worlds_per_block",
    "_constraint_gather_errors_kernel",
    "_constraint_gather_wrenches_kernel",
    "_constraints_to_elements_kernel",
    "_count_and_mark_world_runs_kernel",
    "_initialize_rigid_topology_rebuild_kernel",
    "_integrate_velocities_kernel",
    "_kinematic_interpolate_substep_kernel",
    "_kinematic_prepare_step_kernel",
    "_merge_monotone_world_runs_kernel",
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
    "get_per_world_greedy_coloring_kernel",
    "get_singleworld_kernel",
    "pack_body_xforms_kernel",
]


#: Warp size and default max threads-per-world for fast-tail kernels.
#: Dynamic auto launches keep this upper bound; fixed launches may use less.
_STRAGGLER_BLOCK_DIM: int = 32


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
def _count_and_mark_world_runs_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    bodies: BodyContainer,
    particle_world_id: wp.array[wp.int32],
    num_bodies: wp.int32,
    # out
    world_element_count: wp.array[wp.int32],  # [nw]
    world_element_offsets_shifted: wp.array[wp.int32],  # [nw+1], will be inclusive-scanned
    run_flags: wp.array[wp.int32],  # [constraint_capacity]
):
    """Count elements per world and mark nondecreasing world-id runs."""
    tid = wp.tid()
    n = num_elements[0]
    if tid == wp.int32(0):
        world_element_offsets_shifted[0] = wp.int32(0)
    if tid >= n:
        run_flags[tid] = wp.int32(0)
        return

    world_id = _element_world_id(elements[tid], bodies, particle_world_id, num_bodies)
    run_flags[tid] = wp.int32(0)
    if tid == wp.int32(0):
        run_flags[tid] = wp.int32(1)
    else:
        previous_world_id = _element_world_id(elements[tid - wp.int32(1)], bodies, particle_world_id, num_bodies)
        if world_id < previous_world_id:
            run_flags[tid] = wp.int32(1)

    if world_id < wp.int32(0):
        return
    wp.atomic_add(world_element_count, world_id, wp.int32(1))
    wp.atomic_add(world_element_offsets_shifted, world_id + wp.int32(1), wp.int32(1))


@wp.kernel(enable_backward=False)
def _scatter_monotone_world_run_starts_kernel(
    num_elements: wp.array[wp.int32],
    run_flags: wp.array[wp.int32],
    run_ids: wp.array[wp.int32],
    run_starts: wp.array[wp.int32],
    num_runs: wp.array[wp.int32],
):
    """Scatter scanned run boundaries and record the run count."""
    tid = wp.tid()
    n = num_elements[0]
    if n == wp.int32(0):
        if tid == wp.int32(0):
            run_starts[0] = wp.int32(0)
            num_runs[0] = wp.int32(0)
        return
    if tid >= n:
        return
    if run_flags[tid] != wp.int32(0):
        run_starts[run_ids[tid] - wp.int32(1)] = tid
    if tid == n - wp.int32(1):
        num_runs[0] = run_ids[tid]


@wp.func
def _lower_bound_element_world(
    elements: wp.array[ElementInteractionData],
    bodies: BodyContainer,
    particle_world_id: wp.array[wp.int32],
    num_bodies: wp.int32,
    start: wp.int32,
    end: wp.int32,
    target_world: wp.int32,
) -> wp.int32:
    lo = start
    hi = end
    while lo < hi:
        mid = lo + (hi - lo) / wp.int32(2)
        world_id = _element_world_id(elements[mid], bodies, particle_world_id, num_bodies)
        if world_id < target_world:
            lo = mid + wp.int32(1)
        else:
            hi = mid
    return lo


@wp.kernel(enable_backward=False)
def _merge_monotone_world_runs_kernel(
    elements: wp.array[ElementInteractionData],
    bodies: BodyContainer,
    particle_world_id: wp.array[wp.int32],
    num_bodies: wp.int32,
    num_worlds: wp.int32,
    world_element_offsets: wp.array[wp.int32],
    num_elements: wp.array[wp.int32],
    run_starts: wp.array[wp.int32],
    num_runs: wp.array[wp.int32],
    world_elements: wp.array[wp.int32],
):
    """Stable-merge monotone CID runs into deterministic world buckets."""
    world_id = wp.tid()
    if world_id >= num_worlds:
        return
    output = world_element_offsets[world_id]
    run_count = num_runs[0]
    run = wp.int32(0)
    while run < run_count:
        start = run_starts[run]
        end = num_elements[0]
        if run + wp.int32(1) < run_count:
            end = run_starts[run + wp.int32(1)]
        first = _lower_bound_element_world(elements, bodies, particle_world_id, num_bodies, start, end, world_id)
        last = _lower_bound_element_world(
            elements, bodies, particle_world_id, num_bodies, first, end, world_id + wp.int32(1)
        )
        cid = first
        while cid < last:
            world_elements[output] = cid
            output += wp.int32(1)
            cid += wp.int32(1)
        run += wp.int32(1)


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


@functools.cache
def get_per_world_greedy_coloring_kernel(group_families: bool):
    """Build the shared world-greedy colorer with optional family grouping."""

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        # per-world bucketing (input from the two kernels above)
        world_element_offsets: wp.array[wp.int32],  # [nw+1] (exclusive prefix of counts)
        world_element_count: wp.array[wp.int32],  # [nw] (raw per-world count)
        world_elements: wp.array[wp.int32],  # [total] flat cid stream, sorted by world
        # graph data
        elements: wp.array[ElementInteractionData],
        element_family: wp.array[wp.int32],
        node_color_mask: wp.array[wp.uint64],  # one 64-color bit mask per body/particle
        max_colors: wp.int32,  # = GREEDY_MAX_COLORS, kept for parity with JP variant
        # scratch (caller zeros each step)
        assigned: wp.array[wp.int32],  # [capacity] 0 unassigned, (c+1) = coloured
        color_family_count: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS * _PER_WORLD_FAST_FAMILIES]
        color_family_offsets: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS * _PER_WORLD_FAST_FAMILIES]
        # outputs
        world_element_ids_by_color: wp.array[wp.int32],  # [total] sorted-by-colour per world
        world_color_starts: wp.array2d[wp.int32],  # [nw, MAX_COLORS+1] per-world prefix
        world_color_family_starts: wp.array2d[wp.int32],  # [nw, GREEDY_MAX_COLORS * _PER_WORLD_FAST_FAMILIES]
        world_num_colors: wp.array[wp.int32],  # [nw]
        overflow_flag: wp.array[wp.int32],  # [1] set if any world exceeds GREEDY_MAX_COLORS
    ):
        """Deterministic smallest-free-color greedy pass, one thread per world.

        RL workloads provide enough worlds to saturate the GPU without exposing
        the tiny graph inside each world as a cooperative block. Serial ownership
        removes MIS rounds, block barriers, and atomics while preserving parallel
        PGS correctness between body-disjoint colors.
        """
        w = wp.tid()
        base = world_element_offsets[w]
        count = world_element_count[w]

        if count == 0:
            world_num_colors[w] = wp.int32(0)
            world_color_starts[w, 0] = wp.int32(0)
            return

        # Only active elements need reset; full-capacity clears are deliberately
        # avoided because each world owns and immediately consumes this slice.
        offset = wp.int32(0)
        while offset < count:
            eid = world_elements[base + offset]
            assigned[eid] = wp.int32(0)
            offset += wp.int32(1)

        num_colors = wp.int32(0)
        offset = wp.int32(0)
        while offset < count:
            eid = world_elements[base + offset]
            forbidden_mask = wp.uint64(0)
            for j in range(MAX_BODIES):
                node = elements[eid].bodies[j]
                if node < wp.int32(0):
                    break
                forbidden_mask |= node_color_mask[node]

            color = _lowest_set_bit(wp.int64(forbidden_mask) ^ _PER_WORLD_FREE_COLOR_FLIP)
            if color < wp.int32(0) or color >= max_colors:
                overflow_flag[0] = wp.int32(1)
                assigned[eid] = wp.int32(-1)
            else:
                # Smallest-free greedy produces a contiguous color range. Initialize
                # a bucket exactly once when its first element appears.
                if color >= num_colors:
                    c = num_colors
                    while c <= color:
                        family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
                        family_count = wp.int32(1)
                        if wp.static(group_families):
                            family_count = wp.int32(_PER_WORLD_FAST_FAMILIES)
                        for f in range(_PER_WORLD_FAST_FAMILIES):
                            if wp.int32(f) < family_count:
                                family_slot = family_base + wp.int32(f)
                                color_family_count[w, family_slot] = wp.int32(0)
                                color_family_offsets[w, family_slot] = wp.int32(0)
                                world_color_family_starts[w, family_slot] = wp.int32(0)
                        c += wp.int32(1)
                    num_colors = color + wp.int32(1)

                assigned[eid] = color + wp.int32(1)
                color_bit = wp.uint64(1) << wp.uint64(color)
                for j in range(MAX_BODIES):
                    node = elements[eid].bodies[j]
                    if node < wp.int32(0):
                        break
                    node_color_mask[node] |= color_bit
                family = wp.int32(0)
                if wp.static(group_families):
                    family = _element_fast_family(element_family[eid])
                family_slot = color * wp.int32(_PER_WORLD_FAST_FAMILIES) + family
                color_family_count[w, family_slot] += wp.int32(1)
            offset += wp.int32(1)

        # Build color/family prefixes, retaining one mutable cursor per family for
        # the deterministic scatter below.
        running = wp.int32(0)
        color = wp.int32(0)
        while color < num_colors:
            world_color_starts[w, color] = running
            family_base = color * wp.int32(_PER_WORLD_FAST_FAMILIES)
            family_count = wp.int32(1)
            if wp.static(group_families):
                family_count = wp.int32(_PER_WORLD_FAST_FAMILIES)
            family_running = wp.int32(0)
            for f in range(_PER_WORLD_FAST_FAMILIES):
                if wp.int32(f) < family_count:
                    family_slot = family_base + wp.int32(f)
                    family_start = running + family_running
                    color_family_offsets[w, family_slot] = family_start
                    world_color_family_starts[w, family_slot] = family_start
                    family_running += color_family_count[w, family_slot]
            running += family_running
            color += wp.int32(1)
        world_color_starts[w, num_colors] = running
        world_num_colors[w] = num_colors

        # Stable world/cid order inside every family bucket.
        offset = wp.int32(0)
        while offset < count:
            eid = world_elements[base + offset]
            color = assigned[eid] - wp.int32(1)
            if color >= wp.int32(0):
                family = wp.int32(0)
                if wp.static(group_families):
                    family = _element_fast_family(element_family[eid])
                family_slot = color * wp.int32(_PER_WORLD_FAST_FAMILIES) + family
                local_slot = color_family_offsets[w, family_slot]
                color_family_offsets[w, family_slot] = local_slot + wp.int32(1)
                world_element_ids_by_color[base + local_slot] = eid
            offset += wp.int32(1)

    return kernel


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
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_soft_contact_pd: bool,
    cached_prepare: bool,
    enable_column_timers: bool,
    patch_friction: bool = False,
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
        t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            t0 = read_global_timer_ns()
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
        if wp.static(enable_column_timers):
            constraint_accumulate_time_us(constraints, ADBS_TIME_US_OFFSET, cid, elapsed_us(t0, read_global_timer_ns()))

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
        t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            t0 = read_global_timer_ns()
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
            if wp.static(patch_friction and has_soft_contact_pd):
                contact_prepare_for_iteration_patch_lean(
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
            elif wp.static(patch_friction):
                contact_prepare_for_iteration_patch_lean_no_soft_pd(
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
            elif wp.static(has_soft_contact_pd):
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
        if wp.static(enable_column_timers):
            contact_accumulate_time_us(contact_cols, local_cid, elapsed_us(t0, read_global_timer_ns()))

    @wp.func
    def _joint_pgs_enabled(cid: wp.int32, joint_pgs_enabled: wp.array[wp.int32]) -> bool:
        if wp.static(selective_joint_pgs):
            return joint_pgs_enabled[cid] != wp.int32(0)
        return True

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
        joint_pgs_enabled: wp.array[wp.int32],
    ):
        if wp.static(has_joints and not has_contacts):
            if wp.static(not skip_joint_pgs):
                if _joint_pgs_enabled(cid, joint_pgs_enabled):
                    _dispatch_prepare_joint(constraints, bodies, particles, copy_state, num_bodies, idt, cid)
        elif cid < num_joints:
            if wp.static(not skip_joint_pgs):
                if _joint_pgs_enabled(cid, joint_pgs_enabled):
                    _dispatch_prepare_joint(constraints, bodies, particles, copy_state, num_bodies, idt, cid)
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

    return _dispatch_prepare_cid, _dispatch_prepare_joint, _dispatch_prepare_contact


@functools.cache
def _make_multiworld_rigid_iterate_dispatch_funcs(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    enable_column_timers: bool,
    use_bias: bool,
    patch_friction: bool = False,
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
            if wp.static(patch_friction and has_soft_contact_pd):
                contact_iterate_patch_multi(
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
            elif wp.static(patch_friction):
                contact_iterate_patch_multi_no_soft_pd(
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
            elif wp.static(has_soft_contact_pd):
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
    def _joint_pgs_enabled(cid: wp.int32, joint_pgs_enabled: wp.array[wp.int32]) -> bool:
        if wp.static(selective_joint_pgs):
            return joint_pgs_enabled[cid] == wp.int32(1)
        return True

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
        joint_pgs_enabled: wp.array[wp.int32],
        num_sweeps: wp.int32,
    ):
        if wp.static(has_joints and not has_contacts):
            if wp.static(not skip_joint_pgs):
                if _joint_pgs_enabled(cid, joint_pgs_enabled):
                    _dispatch_iterate_joint(
                        constraints, bodies, particles, copy_state, num_bodies, idt, sor_boost, cid, num_sweeps
                    )
        elif cid < num_joints:
            if wp.static(not skip_joint_pgs):
                if _joint_pgs_enabled(cid, joint_pgs_enabled):
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
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_neohookean: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    patch_friction: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
    solve_joint_inner_sweeps: int = _FAST_TAIL_SOLVE_JOINT_INNER_SWEEPS,
    solve_contact_inner_sweeps: int = _FAST_TAIL_SOLVE_CONTACT_INNER_SWEEPS,
    solve_outer_iteration_chunk: int = _FAST_TAIL_SOLVE_OUTER_ITERATION_CHUNK,
):
    """Build the multi-world fused prepare + iterate fast-tail kernel."""
    (
        _dispatch_prepare_cid,
        _dispatch_prepare_joint,
        _dispatch_prepare_contact,
    ) = _make_multiworld_rigid_prepare_dispatch_func(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_soft_contact_pd=has_soft_contact_pd,
        cached_prepare=cached_prepare,
        enable_column_timers=enable_column_timers,
        patch_friction=patch_friction,
    )
    (
        _dispatch_iterate_cid,
        _dispatch_iterate_joint,
        _dispatch_iterate_contact,
    ) = _make_multiworld_rigid_iterate_dispatch_funcs(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        enable_column_timers=enable_column_timers,
        use_bias=True,
        patch_friction=patch_friction,
    )
    _dispatch_prepare_any_cid = None
    _dispatch_iterate_any_cid = None
    if cloth_support:
        _dispatch_prepare_any_cid = _make_singleworld_dispatch_func(
            revolute_only=revolute_only,
            cloth_support=cloth_support,
            enable_column_timers=enable_column_timers,
            soft_tet_neohookean=soft_tet_neohookean,
            has_joints=has_joints,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_mass_splitting=False,
            packed_contact_headers=False,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            is_prepare=True,
            is_cached_prepare=False,
            use_bias=True,
        )[0]
        _dispatch_iterate_any_cid = _make_singleworld_dispatch_func(
            revolute_only=revolute_only,
            cloth_support=cloth_support,
            enable_column_timers=enable_column_timers,
            soft_tet_neohookean=soft_tet_neohookean,
            has_joints=has_joints,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_mass_splitting=False,
            packed_contact_headers=False,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            is_prepare=False,
            is_cached_prepare=False,
            use_bias=True,
        )[0]

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
        joint_pgs_enabled: wp.array[wp.int32],
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

            if wp.static(family_split):
                family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
                if wp.static(cloth_support):
                    family = wp.int32(0)
                    while family < wp.int32(_PER_WORLD_FAST_FAMILIES):
                        family_start = world_base + world_color_family_starts[world_id, family_base + family]
                        family_end = end
                        if family + wp.int32(1) < wp.int32(_PER_WORLD_FAST_FAMILIES):
                            family_end = (
                                world_base + world_color_family_starts[world_id, family_base + family + wp.int32(1)]
                            )
                        family_count = family_end - family_start

                        base = local_tid
                        while base < family_count:
                            cid = world_element_ids_by_color[family_start + base]
                            _dispatch_prepare_any_cid(
                                constraints,
                                contact_cols,
                                bodies,
                                particles,
                                cc,
                                contacts,
                                copy_state,
                                num_joints,
                                joint_pgs_enabled,
                                num_cloth_triangles,
                                num_cloth_bending,
                                num_soft_tetrahedra,
                                num_soft_hexahedra,
                                num_bodies,
                                idt,
                                sor_boost,
                                cid,
                                family_start + base,
                                wp.int32(0),
                            )
                            base += tpw
                        family += wp.int32(1)
                else:
                    joint_start = world_base + world_color_family_starts[world_id, family_base]
                    contact_start = world_base + world_color_family_starts[world_id, family_base + wp.int32(1)]
                    count_joints = contact_start - joint_start

                    if wp.static(not skip_joint_pgs):
                        base = local_tid
                        while base < count_joints:
                            cid = world_element_ids_by_color[joint_start + base]
                            if wp.static(not selective_joint_pgs) or joint_pgs_enabled[cid] != wp.int32(0):
                                _dispatch_prepare_joint(
                                    constraints,
                                    bodies,
                                    particles,
                                    copy_state,
                                    num_bodies,
                                    idt,
                                    cid,
                                )
                            base += tpw

                    count_contacts = end - contact_start
                    base = local_tid
                    while base < count_contacts:
                        cid = world_element_ids_by_color[contact_start + base]
                        _dispatch_prepare_contact(
                            contact_cols,
                            bodies,
                            particles,
                            cc,
                            contacts,
                            copy_state,
                            num_bodies,
                            idt,
                            cid - num_joints,
                        )
                        base += tpw
            else:
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
                            joint_pgs_enabled,
                            num_cloth_triangles,
                            num_cloth_bending,
                            num_soft_tetrahedra,
                            num_soft_hexahedra,
                            num_bodies,
                            idt,
                            sor_boost,
                            cid,
                            start + base,
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
                            joint_pgs_enabled,
                        )
                    base += tpw

            _sync_warp_mask(sync_mask)
            c += 1

        solve_outer_chunk = wp.int32(solve_outer_iteration_chunk)
        solve_outer_iterations = (num_iterations + solve_outer_chunk - wp.int32(1)) / solve_outer_chunk
        it_outer = wp.int32(0)
        while it_outer < solve_outer_iterations:
            c = wp.int32(0)
            while c < n_colors:
                color = c
                if (it_outer & wp.int32(1)) != wp.int32(0):
                    color = n_colors - wp.int32(1) - c
                start = world_base + world_color_starts[world_id, color]
                end = world_base + world_color_starts[world_id, color + 1]
                count = end - start

                if wp.static(family_split):
                    family_base = color * wp.int32(_PER_WORLD_FAST_FAMILIES)
                    if wp.static(cloth_support):
                        family = wp.int32(0)
                        while family < wp.int32(_PER_WORLD_FAST_FAMILIES):
                            family_start = world_base + world_color_family_starts[world_id, family_base + family]
                            family_end = end
                            if family + wp.int32(1) < wp.int32(_PER_WORLD_FAST_FAMILIES):
                                family_end = (
                                    world_base + world_color_family_starts[world_id, family_base + family + wp.int32(1)]
                                )
                            family_count = family_end - family_start

                            base = local_tid
                            while base < family_count:
                                cid = world_element_ids_by_color[family_start + base]
                                _dispatch_iterate_any_cid(
                                    constraints,
                                    contact_cols,
                                    bodies,
                                    particles,
                                    cc,
                                    contacts,
                                    copy_state,
                                    num_joints,
                                    joint_pgs_enabled,
                                    num_cloth_triangles,
                                    num_cloth_bending,
                                    num_soft_tetrahedra,
                                    num_soft_hexahedra,
                                    num_bodies,
                                    idt,
                                    sor_boost,
                                    cid,
                                    family_start + base,
                                    wp.int32(0),
                                )
                                base += tpw
                            family += wp.int32(1)
                    else:
                        joint_start = world_base + world_color_family_starts[world_id, family_base]
                        contact_start = world_base + world_color_family_starts[world_id, family_base + wp.int32(1)]
                        count_joints = contact_start - joint_start

                        if wp.static(not skip_joint_pgs):
                            base = local_tid
                            while base < count_joints:
                                cid = world_element_ids_by_color[joint_start + base]
                                if wp.static(not selective_joint_pgs) or joint_pgs_enabled[cid] == wp.int32(1):
                                    _dispatch_iterate_joint(
                                        constraints,
                                        bodies,
                                        particles,
                                        copy_state,
                                        num_bodies,
                                        idt,
                                        sor_boost,
                                        cid,
                                        wp.int32(solve_joint_inner_sweeps),
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
                                wp.int32(solve_contact_inner_sweeps),
                            )
                            base += tpw
                else:
                    base = local_tid
                    while base < count:
                        cid = world_element_ids_by_color[start + base]
                        if wp.static(cloth_support):
                            _dispatch_iterate_any_cid(
                                constraints,
                                contact_cols,
                                bodies,
                                particles,
                                cc,
                                contacts,
                                copy_state,
                                num_joints,
                                joint_pgs_enabled,
                                num_cloth_triangles,
                                num_cloth_bending,
                                num_soft_tetrahedra,
                                num_soft_hexahedra,
                                num_bodies,
                                idt,
                                sor_boost,
                                cid,
                                start + base,
                                wp.int32(0),
                            )
                        else:
                            if wp.static(has_joints and not has_contacts):
                                if wp.static(not skip_joint_pgs):
                                    if wp.static(not selective_joint_pgs) or joint_pgs_enabled[cid] == wp.int32(1):
                                        _dispatch_iterate_joint(
                                            constraints,
                                            bodies,
                                            particles,
                                            copy_state,
                                            num_bodies,
                                            idt,
                                            sor_boost,
                                            cid,
                                            wp.int32(solve_joint_inner_sweeps),
                                        )
                            elif cid < num_joints:
                                if wp.static(not skip_joint_pgs):
                                    if wp.static(not selective_joint_pgs) or joint_pgs_enabled[cid] == wp.int32(1):
                                        _dispatch_iterate_joint(
                                            constraints,
                                            bodies,
                                            particles,
                                            copy_state,
                                            num_bodies,
                                            idt,
                                            sor_boost,
                                            cid,
                                            wp.int32(solve_joint_inner_sweeps),
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
                                    wp.int32(solve_contact_inner_sweeps),
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
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_neohookean: bool = False,
    enable_column_timers: bool = False,
    patch_friction: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
):
    """Multi-world relax fast-tail kernel with global color sweep order."""
    (
        _dispatch_iterate_cid,
        _dispatch_iterate_joint,
        _dispatch_iterate_contact,
    ) = _make_multiworld_rigid_iterate_dispatch_funcs(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        enable_column_timers=enable_column_timers,
        use_bias=False,
        patch_friction=patch_friction,
    )
    _dispatch_relax_any_cid = None
    if cloth_support:
        _dispatch_relax_any_cid = _make_singleworld_dispatch_func(
            revolute_only=revolute_only,
            cloth_support=cloth_support,
            enable_column_timers=enable_column_timers,
            soft_tet_neohookean=soft_tet_neohookean,
            has_joints=has_joints,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_mass_splitting=False,
            packed_contact_headers=False,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            is_prepare=False,
            is_cached_prepare=False,
            use_bias=False,
        )[0]

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
        joint_pgs_enabled: wp.array[wp.int32],
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

        relax_iterations = num_iterations
        sweeps_per_dispatch = wp.int32(1)

        # Every relax iteration visits all colors once before any color
        # repeats, matching the solve-phase PGS ordering.
        it = wp.int32(0)
        while it < relax_iterations:
            c = wp.int32(0)
            while c < n_colors:
                color = c
                if (it & wp.int32(1)) != wp.int32(0):
                    color = n_colors - wp.int32(1) - c
                start = world_base + world_color_starts[world_id, color]
                end = world_base + world_color_starts[world_id, color + 1]
                if wp.static(family_split):
                    family_base = color * wp.int32(_PER_WORLD_FAST_FAMILIES)
                    if wp.static(cloth_support):
                        family = wp.int32(0)
                        while family < wp.int32(_PER_WORLD_FAST_FAMILIES):
                            family_start = world_base + world_color_family_starts[world_id, family_base + family]
                            family_end = end
                            if family + wp.int32(1) < wp.int32(_PER_WORLD_FAST_FAMILIES):
                                family_end = (
                                    world_base + world_color_family_starts[world_id, family_base + family + wp.int32(1)]
                                )
                            family_count = family_end - family_start

                            base = local_tid
                            while base < family_count:
                                cid = world_element_ids_by_color[family_start + base]
                                _dispatch_relax_any_cid(
                                    constraints,
                                    contact_cols,
                                    bodies,
                                    particles,
                                    cc,
                                    contacts,
                                    copy_state,
                                    num_joints,
                                    joint_pgs_enabled,
                                    num_cloth_triangles,
                                    num_cloth_bending,
                                    num_soft_tetrahedra,
                                    num_soft_hexahedra,
                                    num_bodies,
                                    idt,
                                    sor_boost,
                                    cid,
                                    family_start + base,
                                    wp.int32(0),
                                )
                                base += tpw
                            family += wp.int32(1)
                    else:
                        joint_start = world_base + world_color_family_starts[world_id, family_base]
                        contact_start = world_base + world_color_family_starts[world_id, family_base + wp.int32(1)]
                        count_joints = contact_start - joint_start

                        if wp.static(not skip_joint_pgs):
                            base = local_tid
                            while base < count_joints:
                                cid = world_element_ids_by_color[joint_start + base]
                                if wp.static(not selective_joint_pgs) or joint_pgs_enabled[cid] == wp.int32(1):
                                    _dispatch_iterate_joint(
                                        constraints,
                                        bodies,
                                        particles,
                                        copy_state,
                                        num_bodies,
                                        idt,
                                        sor_boost,
                                        cid,
                                        sweeps_per_dispatch,
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
                                sweeps_per_dispatch,
                            )
                            base += tpw
                else:
                    count = end - start

                    base = local_tid
                    while base < count:
                        cid = world_element_ids_by_color[start + base]
                        if wp.static(cloth_support):
                            _dispatch_relax_any_cid(
                                constraints,
                                contact_cols,
                                bodies,
                                particles,
                                cc,
                                contacts,
                                copy_state,
                                num_joints,
                                joint_pgs_enabled,
                                num_cloth_triangles,
                                num_cloth_bending,
                                num_soft_tetrahedra,
                                num_soft_hexahedra,
                                num_bodies,
                                idt,
                                sor_boost,
                                cid,
                                start + base,
                                wp.int32(0),
                            )
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
                                joint_pgs_enabled,
                                sweeps_per_dispatch,
                            )
                        base += tpw

                _sync_warp_mask(sync_mask)
                c += 1
            it += wp.int32(1)

    return kernel


@functools.cache
def _make_block_world_prepare_plus_iterate_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    block_dim: int = 128,
    solve_inner_sweeps: int = 1,
):
    """Build a multi-world kernel where one physical block owns one world."""
    (
        _dispatch_prepare_cid,
        _dispatch_prepare_joint,
        _dispatch_prepare_contact,
    ) = _make_multiworld_rigid_prepare_dispatch_func(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
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
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
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
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        joint_pgs_enabled: wp.array[wp.int32],
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
                    joint_pgs_enabled,
                )
                base += wp.int32(block_dim)

            if wp.static(block_dim == 32):
                _sync_warp()
            else:
                _sync_threads()
            c += wp.int32(1)

        # Outer rounds run ``solve_inner_sweeps`` register-cached sweeps per
        # dispatch; a final short round mops up any remainder so the total
        # sweep count always equals ``num_iterations``. ``sweeps`` is uniform
        # across the block, so the per-colour barriers stay collective.
        done_sweeps = wp.int32(0)
        while done_sweeps < num_iterations:
            sweeps = wp.int32(solve_inner_sweeps)
            if done_sweeps + sweeps > num_iterations:
                sweeps = num_iterations - done_sweeps
            c = wp.int32(0)
            while c < n_colors:
                color = c
                if (done_sweeps & wp.int32(1)) != wp.int32(0):
                    color = n_colors - wp.int32(1) - c
                start = world_base + world_color_starts[world_id, color]
                end = world_base + world_color_starts[world_id, color + wp.int32(1)]

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
                        joint_pgs_enabled,
                        sweeps,
                    )
                    base += wp.int32(block_dim)

                if wp.static(block_dim == 32):
                    _sync_warp()
                else:
                    _sync_threads()
                c += wp.int32(1)
            done_sweeps += sweeps

    return kernel


@functools.cache
def _make_block_world_relax_kernel(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool = False,
    enable_column_timers: bool = False,
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
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
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
        world_csr_offsets: wp.array[wp.int32],
        world_num_colors: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_iterations: wp.int32,
        num_worlds: wp.int32,
        num_joints: wp.int32,
        joint_pgs_enabled: wp.array[wp.int32],
        copy_state: CopyStateContainer,
    ):
        tid = wp.tid()
        local_tid = tid % wp.int32(block_dim)
        world_id = tid / wp.int32(block_dim)
        if world_id >= num_worlds:
            return

        n_colors = world_num_colors[world_id]
        world_base = world_csr_offsets[world_id]

        # Alternate forward and reverse color order between iterations.
        it = wp.int32(0)
        while it < num_iterations:
            c = wp.int32(0)
            while c < n_colors:
                color = c
                if (it & wp.int32(1)) != wp.int32(0):
                    color = n_colors - wp.int32(1) - c
                start = world_base + world_color_starts[world_id, color]
                end = world_base + world_color_starts[world_id, color + wp.int32(1)]

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
                        joint_pgs_enabled,
                        wp.int32(1),
                    )
                    base += wp.int32(block_dim)

                if wp.static(block_dim == 32):
                    _sync_warp()
                else:
                    _sync_threads()
                c += wp.int32(1)
            it += wp.int32(1)

    return kernel


# Fast-tail kernels are built lazily so only the scene-specific
# revolute/generic variant gets compiled.


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
    skip_joint_pgs: bool = False,
    selective_joint_pgs: bool = False,
    has_sleeping: bool = False,
    has_soft_contact_pd: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    block_dim: int = 128,
    solve_inner_sweeps: int = 1,
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
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            cached_prepare=cached_prepare,
            enable_column_timers=enable_column_timers,
            block_dim=block_dim,
            solve_inner_sweeps=solve_inner_sweeps,
        )
    if kind == "relax":
        return _make_block_world_relax_kernel(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            enable_column_timers=enable_column_timers,
            block_dim=block_dim,
        )
    raise ValueError(f"unknown block-world kernel kind: {kind!r}")


def get_fast_tail_kernel(
    *,
    kind: str,
    revolute_only: bool,
    has_joints: bool = True,
    has_contacts: bool = True,
    skip_joint_pgs: bool = False,
    selective_joint_pgs: bool = False,
    has_sleeping: bool = False,
    has_soft_contact_pd: bool = False,
    cloth_support: bool = False,
    soft_tet_neohookean: bool = False,
    cached_prepare: bool = False,
    enable_column_timers: bool = False,
    patch_friction: bool = False,
    fixed_tpw: int = 0,
    guard_tpw: bool = True,
    family_split: bool = False,
    solve_joint_inner_sweeps: int = _FAST_TAIL_SOLVE_JOINT_INNER_SWEEPS,
    solve_contact_inner_sweeps: int = _FAST_TAIL_SOLVE_CONTACT_INNER_SWEEPS,
    solve_outer_iteration_chunk: int = _FAST_TAIL_SOLVE_OUTER_ITERATION_CHUNK,
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
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            cloth_support=cloth_support,
            soft_tet_neohookean=soft_tet_neohookean,
            cached_prepare=cached_prepare,
            enable_column_timers=enable_column_timers,
            patch_friction=patch_friction,
            fixed_tpw=fixed_tpw,
            guard_tpw=guard_tpw,
            family_split=family_split,
            solve_joint_inner_sweeps=solve_joint_inner_sweeps,
            solve_contact_inner_sweeps=solve_contact_inner_sweeps,
            solve_outer_iteration_chunk=solve_outer_iteration_chunk,
        )
    if kind == "relax":
        return _make_fast_tail_relax_kernel(
            revolute_only=revolute_only,
            has_joints=has_joints,
            has_contacts=has_contacts,
            skip_joint_pgs=skip_joint_pgs,
            selective_joint_pgs=selective_joint_pgs,
            has_sleeping=has_sleeping,
            has_soft_contact_pd=has_soft_contact_pd,
            cloth_support=cloth_support,
            soft_tet_neohookean=soft_tet_neohookean,
            enable_column_timers=enable_column_timers,
            patch_friction=patch_friction,
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


@wp.func
def _rigid_graph_node(node: wp.int32, bodies: BodyContainer):
    if node >= 0:
        if bodies.inverse_mass[node] == 0.0:
            node = -1
        else:
            node = bodies.constraint_node[node]
    return node


@wp.func
def _particle_graph_node(node: wp.int32, particles: ParticleContainer, num_bodies: wp.int32):
    if node >= num_bodies and particles.inverse_mass[node - num_bodies] == 0.0:
        node = -1
    return node


@wp.func
def _contact_primary_graph_node(
    node: wp.int32,
    side_kind: wp.int32,
    use_node: wp.bool,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
):
    if node >= 0:
        if side_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON) and not use_node:
            node = -1
        elif side_kind == wp.int32(SHAPE_ENDPOINT_KIND_RIGID):
            if bodies.inverse_mass[node] == 0.0 and bodies.motion_type[node] != MOTION_KINEMATIC:
                node = -1
            elif bodies.motion_type[node] != MOTION_KINEMATIC:
                node = bodies.constraint_node[node]
        else:
            if particles.inverse_mass[node - num_bodies] == 0.0:
                node = -1
    return node


@wp.func
def _contact_extra_graph_nodes(
    side_kind: wp.int32,
    extra: wp.vec3i,
    use1: wp.bool,
    use2: wp.bool,
    use3: wp.bool,
    particles: ParticleContainer,
    num_bodies: wp.int32,
):
    e0 = wp.int32(-1)
    e1 = wp.int32(-1)
    e2 = wp.int32(-1)
    if side_kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        e0 = _particle_graph_node(extra[0], particles, num_bodies)
        e1 = _particle_graph_node(extra[1], particles, num_bodies)
    elif side_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        if use1:
            e0 = extra[0]
        if use2:
            e1 = extra[1]
        if use3:
            e2 = extra[2]
        e0 = _particle_graph_node(e0, particles, num_bodies)
        e1 = _particle_graph_node(e1, particles, num_bodies)
        e2 = _particle_graph_node(e2, particles, num_bodies)
    return e0, e1, e2


@wp.func
def _element_data_compact2(v0: wp.int32, v1: wp.int32):
    s0 = wp.int32(-1)
    s1 = wp.int32(-1)
    if v0 >= 0:
        s0 = v0
    if v1 >= 0:
        if s0 < 0:
            s0 = v1
        else:
            s1 = v1
    return element_interaction_data_make(s0, s1, -1, -1, -1, -1, -1, -1)


@wp.func
def _element_data_compact3(v0: wp.int32, v1: wp.int32, v2: wp.int32):
    s0 = wp.int32(-1)
    s1 = wp.int32(-1)
    s2 = wp.int32(-1)
    cnt = wp.int32(0)
    for cand in range(3):
        v = v0
        if cand == 1:
            v = v1
        elif cand == 2:
            v = v2
        if v < 0:
            continue
        if cnt == 0:
            s0 = v
        elif cnt == 1:
            s1 = v
        else:
            s2 = v
        cnt = cnt + 1
    return element_interaction_data_make(s0, s1, s2, -1, -1, -1, -1, -1)


@wp.func
def _element_data_compact4(v0: wp.int32, v1: wp.int32, v2: wp.int32, v3: wp.int32):
    s0 = wp.int32(-1)
    s1 = wp.int32(-1)
    s2 = wp.int32(-1)
    s3 = wp.int32(-1)
    cnt = wp.int32(0)
    for cand in range(4):
        v = v0
        if cand == 1:
            v = v1
        elif cand == 2:
            v = v2
        elif cand == 3:
            v = v3
        if v < 0:
            continue
        if cnt == 0:
            s0 = v
        elif cnt == 1:
            s1 = v
        elif cnt == 2:
            s2 = v
        else:
            s3 = v
        cnt = cnt + 1
    return element_interaction_data_make(s0, s1, s2, s3, -1, -1, -1, -1)


@wp.func
def _element_data_compact8(
    v0: wp.int32,
    v1: wp.int32,
    v2: wp.int32,
    v3: wp.int32,
    v4: wp.int32,
    v5: wp.int32,
    v6: wp.int32,
    v7: wp.int32,
):
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
        v = v0
        if cand == 1:
            v = v1
        elif cand == 2:
            v = v2
        elif cand == 3:
            v = v3
        elif cand == 4:
            v = v4
        elif cand == 5:
            v = v5
        elif cand == 6:
            v = v6
        elif cand == 7:
            v = v7
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
    return element_interaction_data_make(s0, s1, s2, s3, s4, s5, s6, s7)


@wp.kernel(enable_backward=False)
def _initialize_rigid_topology_rebuild_kernel(
    num_constraints: wp.array[wp.int32],
    previous_num_constraints: wp.array[wp.int32],
    topology_rebuild: wp.array[wp.int32],
):
    n = num_constraints[0]
    rebuild = wp.int32(0)
    if previous_num_constraints[0] != n:
        rebuild = wp.int32(1)
    previous_num_constraints[0] = n
    topology_rebuild[0] = rebuild


@wp.func
def _record_rigid_topology(
    tid: wp.int32,
    body0: wp.int32,
    body1: wp.int32,
    previous_topology: wp.array[wp.int64],
    topology_rebuild: wp.array[wp.int32],
):
    key = (wp.int64(body0) << wp.int64(32)) | (wp.int64(body1) & wp.int64(0xFFFFFFFF))
    if previous_topology[tid] != key:
        wp.atomic_max(topology_rebuild, 0, wp.int32(1))
    previous_topology[tid] = key


@wp.func
def _stable_contact_pair_priority(shape_a: wp.int32, shape_b: wp.int32) -> wp.int32:
    """Return a model-stable 24-bit priority for a contact shape pair."""
    lo = shape_a
    hi = shape_b
    if lo > hi:
        lo = shape_b
        hi = shape_a
    h = (lo + wp.int32(1)) * wp.int32(73856093)
    h = h ^ ((hi + wp.int32(1)) * wp.int32(19349663))
    h = h ^ (h >> wp.int32(13))
    h = h * wp.int32(83492791)
    return h & wp.int32(0x00FFFFFF)


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
    pair_source_idx: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    random_values: wp.array[wp.int32],
    track_rigid_topology: wp.int32,
    elements: wp.array[ElementInteractionData],
    element_family: wp.array[wp.int32],
    packed_priorities: wp.array[wp.int32],
    previous_topology: wp.array[wp.int64],
    topology_rebuild: wp.array[wp.int32],
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
    packed_priorities[tid] = random_values[tid] & wp.int32(0x00FFFFFF)
    if tid < num_joints:
        element_family[tid] = wp.int32(0)
        b1 = constraint_get_body1(constraints, tid)
        b2 = constraint_get_body2(constraints, tid)
        b1 = _rigid_graph_node(b1, bodies)
        b2 = _rigid_graph_node(b2, bodies)
        elements[tid] = _element_data_compact2(b1, b2)
        if track_rigid_topology != wp.int32(0):
            _record_rigid_topology(tid, b1, b2, previous_topology, topology_rebuild)
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
        b1 = _particle_graph_node(b1, particles, num_bodies)
        b2 = _particle_graph_node(b2, particles, num_bodies)
        b3 = _particle_graph_node(b3, particles, num_bodies)
        elements[tid] = _element_data_compact3(b1, b2, b3)
        return
    if tid < num_joints + num_cloth_triangles + num_cloth_bending:
        element_family[tid] = wp.int32(3)
        # Cloth-bending: 4 unified-index particle endpoints. body1 / body2
        # are the opposite vertices; body3 / body4 are the shared edge.
        b1 = constraint_get_body1(constraints, tid)
        b2 = constraint_get_body2(constraints, tid)
        b3 = read_int(constraints, _CLOTH_BENDING_OFF_BODY3, tid)
        b4 = read_int(constraints, _CLOTH_BENDING_OFF_BODY4, tid)
        b1 = _particle_graph_node(b1, particles, num_bodies)
        b2 = _particle_graph_node(b2, particles, num_bodies)
        b3 = _particle_graph_node(b3, particles, num_bodies)
        b4 = _particle_graph_node(b4, particles, num_bodies)
        elements[tid] = _element_data_compact4(b1, b2, b3, b4)
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
        b1 = _particle_graph_node(b1, particles, num_bodies)
        b2 = _particle_graph_node(b2, particles, num_bodies)
        b3 = _particle_graph_node(b3, particles, num_bodies)
        b4 = _particle_graph_node(b4, particles, num_bodies)
        elements[tid] = _element_data_compact4(b1, b2, b3, b4)
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
        h0 = _particle_graph_node(h0, particles, num_bodies)
        h1 = _particle_graph_node(h1, particles, num_bodies)
        h2 = _particle_graph_node(h2, particles, num_bodies)
        h3 = _particle_graph_node(h3, particles, num_bodies)
        h4 = _particle_graph_node(h4, particles, num_bodies)
        h5 = _particle_graph_node(h5, particles, num_bodies)
        h6 = _particle_graph_node(h6, particles, num_bodies)
        h7 = _particle_graph_node(h7, particles, num_bodies)
        elements[tid] = _element_data_compact8(h0, h1, h2, h3, h4, h5, h6, h7)
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
    priority_cost = wp.min(contact_count, wp.int32(255))
    pair = pair_source_idx[local_cid]
    priority_rand = _stable_contact_pair_priority(pair_shape_a[pair], pair_shape_b[pair])
    packed_priorities[tid] = (priority_cost << wp.int32(24)) | priority_rand
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
    b1 = _contact_primary_graph_node(b1, side0_kind, side0_use0, bodies, particles, num_bodies)
    b2 = _contact_primary_graph_node(b2, side1_kind, side1_use0, bodies, particles, num_bodies)

    # Resolve up to three extra nodes per side. Rigid sides leave all
    # extras at -1; cloth-tri sides populate two; soft-tet sides populate
    # only the nonzero-barycentric nodes the iterate can read or write.
    e0a, e0b, e0c = _contact_extra_graph_nodes(
        side0_kind, side0_extra, side0_use1, side0_use2, side0_use3, particles, num_bodies
    )
    e1a, e1b, e1c = _contact_extra_graph_nodes(
        side1_kind, side1_extra, side1_use1, side1_use2, side1_use3, particles, num_bodies
    )

    # Compact: drop -1s into a contiguous prefix (the partitioner's
    # adjacency loop stops on the first -1). Up to 8 nodes per contact:
    # tet-tet = 4+4; tet-cloth = 4+3; tet-rigid = 4+1; cloth-cloth =
    # 3+3; cloth-rigid = 3+1; rigid-rigid = 1+1.
    elements[tid] = _element_data_compact8(b1, b2, e0a, e0b, e0c, e1a, e1b, e1c)
    if track_rigid_topology != wp.int32(0):
        _record_rigid_topology(tid, b1, b2, previous_topology, topology_rebuild)


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
    if mt != MOTION_DYNAMIC:
        return
    # Sleeping bodies must not drift. They may carry a small residual
    # velocity (anywhere below the per-island sleep threshold) at the
    # moment ``island_root`` is stamped; integrating that for many
    # substeps would slide the whole sleeping island visibly.
    if bodies.island_root[i] >= wp.int32(0):
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt

    # Integrate torque-free rotation with an implicit midpoint update. Contact
    # and external impulses have already changed omega, so preserve their
    # resulting world angular momentum while the anisotropic inertia rotates.
    q0 = bodies.orientation[i]
    inv_inertia0 = mat33_from_sym6(bodies.inverse_inertia_world[i])
    angular_momentum = wp.inverse(inv_inertia0) * bodies.angular_velocity[i]
    omega_mid = bodies.angular_velocity[i]
    for _ in range(3):
        q_half = wp.normalize(_rotation_quaternion(omega_mid, dt * wp.float32(0.5)) * q0)
        r_half = wp.quat_to_matrix(q_half)
        inv_inertia_half = rotate_inertia(r_half, bodies.inverse_inertia[i])
        omega_mid = inv_inertia_half * angular_momentum

    q1 = wp.normalize(_rotation_quaternion(omega_mid, dt) * q0)
    r1 = wp.quat_to_matrix(q1)
    inv_inertia1 = rotate_inertia(r1, bodies.inverse_inertia[i])
    bodies.orientation[i] = q1
    bodies.inverse_inertia_world[i] = sym6_from_mat33(inv_inertia1)
    bodies.angular_velocity[i] = inv_inertia1 * angular_momentum


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
        if bodies.motion_type[i] == MOTION_STATIC:
            bodies.velocity[i] = wp.vec3f(0.0, 0.0, 0.0)
            bodies.angular_velocity[i] = wp.vec3f(0.0, 0.0, 0.0)
        return
    if bodies.inverse_mass[i] == 0.0:
        bodies.access_mode[i] = ACCESS_MODE_STATIC
        bodies.velocity[i] = wp.vec3f(0.0, 0.0, 0.0)
        bodies.angular_velocity[i] = wp.vec3f(0.0, 0.0, 0.0)
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
    inv_inertia_world = mat33_from_sym6(bodies.inverse_inertia_world[i])
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
        bodies.inverse_inertia_world[i] = sym6_from_mat33(rotate_inertia(r, bodies.inverse_inertia[i]))
    # Force / torque clear: every body slot, including kinematic / static.
    bodies.force[i] = wp.vec3f(0.0, 0.0, 0.0)
    bodies.torque[i] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _phoenx_refresh_world_inertia_kernel(
    bodies: BodyContainer,
):
    """Refresh world inverse inertia after pose integration.

    Angular velocity is left unchanged. Transporting angular momentum through
    an explicitly integrated pose injects rotational energy into strongly
    anisotropic bodies such as Kapla planks.
    """
    i = wp.tid()
    if bodies.motion_type[i] == MOTION_DYNAMIC:
        r = wp.quat_to_matrix(bodies.orientation[i])
        bodies.inverse_inertia_world[i] = sym6_from_mat33(rotate_inertia(r, bodies.inverse_inertia[i]))


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
    packed_contact_headers: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    is_prepare: bool,
    is_cached_prepare: bool,
    use_bias: bool,
    patch_friction: bool = False,
):
    if is_prepare:
        if patch_friction:
            prepare_func = (
                contact_prepare_for_iteration_patch_lean
                if has_soft_contact_pd
                else contact_prepare_for_iteration_patch_lean_no_soft_pd
            )
        elif has_mass_splitting:
            if packed_contact_headers:
                prepare_func = (
                    contact_prepare_for_iteration_packed_rows
                    if has_soft_contact_pd
                    else contact_prepare_for_iteration_packed_rows_no_soft_pd
                )
            else:
                prepare_func = (
                    contact_prepare_for_iteration if has_soft_contact_pd else contact_prepare_for_iteration_no_soft_pd
                )
        else:
            prepare_func = (
                contact_prepare_for_iteration_lean
                if has_soft_contact_pd
                else contact_prepare_for_iteration_lean_no_soft_pd
            )

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
            colored_slot: wp.int32,
            parallel_id: wp.int32,
        ):
            solve_cid = local_cid
            if wp.static(packed_contact_headers):
                solve_cid = colored_slot
            prepare_func(
                contact_cols,
                solve_cid,
                bodies,
                particles,
                num_bodies,
                idt,
                cc,
                contacts,
                copy_state,
                parallel_id,
            )

    elif is_cached_prepare:

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
            colored_slot: wp.int32,
            parallel_id: wp.int32,
        ):
            solve_cid = local_cid
            if wp.static(packed_contact_headers):
                solve_cid = colored_slot
            contact_cached_warmstart_lean(
                contact_cols,
                solve_cid,
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
        if patch_friction:
            iterate_func = (
                contact_iterate_patch_lean_no_sleep
                if has_soft_contact_pd
                else contact_iterate_patch_lean_no_sleep_no_soft_pd
            )
        elif has_mass_splitting:
            if has_sleeping:
                iterate_func = contact_iterate if has_soft_contact_pd else contact_iterate_no_soft_pd
            else:
                iterate_func = contact_iterate_no_sleep if has_soft_contact_pd else contact_iterate_no_sleep_no_soft_pd
        elif has_sleeping:
            iterate_func = contact_iterate_lean if has_soft_contact_pd else contact_iterate_lean_no_soft_pd
        else:
            iterate_func = (
                contact_iterate_lean_no_sleep if has_soft_contact_pd else contact_iterate_lean_no_sleep_no_soft_pd
            )

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
            colored_slot: wp.int32,
            parallel_id: wp.int32,
        ):
            solve_cid = local_cid
            if wp.static(packed_contact_headers):
                solve_cid = colored_slot
            iterate_func(
                contact_cols,
                solve_cid,
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
def _make_singleworld_rigid_joint_dispatch_func(
    *,
    revolute_only: bool,
    is_prepare: bool,
    is_cached_prepare: bool,
    use_bias: bool,
    enable_column_timers: bool,
):
    """Generated rigid-joint dispatch for single-world kernels."""

    if is_cached_prepare:
        joint_func = revolute_cached_warmstart if revolute_only else actuated_double_ball_socket_cached_warmstart
    elif is_prepare:
        joint_func = (
            revolute_prepare_for_iteration if revolute_only else actuated_double_ball_socket_prepare_for_iteration
        )
    else:
        joint_func = revolute_iterate if revolute_only else actuated_double_ball_socket_iterate

    @wp.func
    def _dispatch_rigid_joint(
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        cid: wp.int32,
        parallel_id: wp.int32,
    ):
        t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            t0 = read_global_timer_ns()

        if wp.static(is_prepare or is_cached_prepare):
            joint_func(constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt)
        else:
            joint_func(
                constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt, sor_boost, use_bias
            )

        if wp.static(enable_column_timers):
            constraint_accumulate_time_us(constraints, ADBS_TIME_US_OFFSET, cid, elapsed_us(t0, read_global_timer_ns()))

    return _dispatch_rigid_joint


@functools.cache
def _make_singleworld_dispatch_func(
    *,
    revolute_only: bool,
    cloth_support: bool,
    soft_tet_neohookean: bool,
    enable_column_timers: bool,
    has_joints: bool,
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_mass_splitting: bool,
    packed_contact_headers: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    is_prepare: bool,
    is_cached_prepare: bool,
    use_bias: bool,
    patch_friction: bool = False,
):
    """Per-cid dispatch helper used by head and fused PGS kernels.

    Non-contact rows are stored in fixed family ranges, so dispatch uses
    ``cid`` bounds instead of reading the stamped type tag from global memory.
    """

    _dispatch_rigid_contact = _make_singleworld_rigid_contact_dispatch_func(
        has_mass_splitting=has_mass_splitting,
        packed_contact_headers=packed_contact_headers,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
        patch_friction=patch_friction,
    )
    _dispatch_rigid_joint = _make_singleworld_rigid_joint_dispatch_func(
        revolute_only=revolute_only,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
        enable_column_timers=enable_column_timers,
    )

    @wp.func
    def _joint_pgs_enabled(cid: wp.int32, joint_pgs_enabled: wp.array[wp.int32]) -> bool:
        if wp.static(selective_joint_pgs):
            if wp.static(is_prepare or is_cached_prepare):
                return joint_pgs_enabled[cid] != wp.int32(0)
            return joint_pgs_enabled[cid] == wp.int32(1)
        return True

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
        joint_pgs_enabled: wp.array[wp.int32],
        num_cloth_triangles: wp.int32,
        num_cloth_bending: wp.int32,
        num_soft_tetrahedra: wp.int32,
        num_soft_hexahedra: wp.int32,
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        cid: wp.int32,
        colored_slot: wp.int32,
        parallel_id: wp.int32,
    ):
        t0 = wp.uint64(0)
        if wp.static(enable_column_timers):
            t0 = read_global_timer_ns()

        cloth_tri_end = num_joints + num_cloth_triangles
        cloth_bend_end = cloth_tri_end + num_cloth_bending
        soft_tet_end = cloth_bend_end + num_soft_tetrahedra
        contact_start = soft_tet_end + num_soft_hexahedra

        dispatched = False
        if cid >= contact_start:
            local_cid = cid - contact_start
            if wp.static(cloth_support):
                side0_kind = contact_get_side0_kind(contact_cols, local_cid)
                side1_kind = contact_get_side1_kind(contact_cols, local_cid)
                use_endpoint_path = side0_kind != wp.int32(0) or side1_kind != wp.int32(0)
                if not use_endpoint_path:
                    b1 = contact_get_body1(contact_cols, local_cid)
                    b2 = contact_get_body2(contact_cols, local_cid)
                    use_endpoint_path = (
                        bodies.motion_type[b1] == MOTION_ARTICULATED or bodies.motion_type[b2] == MOTION_ARTICULATED
                    )
                if use_endpoint_path:
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
                        colored_slot,
                        parallel_id,
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
                    colored_slot,
                    parallel_id,
                )
            dispatched = True
            if wp.static(enable_column_timers):
                contact_accumulate_time_us(contact_cols, local_cid, elapsed_us(t0, read_global_timer_ns()))

        if not dispatched:
            if cid < num_joints:
                dispatched = True
                if wp.static(has_joints and not skip_joint_pgs):
                    if _joint_pgs_enabled(cid, joint_pgs_enabled):
                        if wp.static(cloth_support):
                            b1 = constraint_get_body1(constraints, cid)
                            b2 = constraint_get_body2(constraints, cid)
                            body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
                            body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
                        _dispatch_rigid_joint(
                            constraints, bodies, particles, copy_state, num_bodies, idt, sor_boost, cid, parallel_id
                        )

        if not dispatched:
            if wp.static(cloth_support):
                if cid < cloth_tri_end:
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
                            elapsed_us(t0, read_global_timer_ns()),
                        )
                elif cid < cloth_bend_end:
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
                            elapsed_us(t0, read_global_timer_ns()),
                        )
                elif cid < soft_tet_end:
                    if wp.static(soft_tet_neohookean):
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
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints,
                                SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET,
                                cid,
                                elapsed_us(t0, read_global_timer_ns()),
                            )
                    else:
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
                        if wp.static(enable_column_timers):
                            constraint_accumulate_time_us(
                                constraints, SOFT_TET_TIME_US_OFFSET, cid, elapsed_us(t0, read_global_timer_ns())
                            )
                else:
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
                    if wp.static(enable_column_timers):
                        constraint_accumulate_time_us(
                            constraints, SOFT_HEX_TIME_US_OFFSET, cid, elapsed_us(t0, read_global_timer_ns())
                        )

    return _dispatch_one_cid, _dispatch_rigid_contact


@functools.cache
def _make_singleworld_rigid_direct_color_func(
    *,
    revolute_only: bool,
    has_joints: bool,
    has_contacts: bool,
    skip_joint_pgs: bool,
    selective_joint_pgs: bool,
    has_mass_splitting: bool,
    packed_contact_headers: bool,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    is_prepare: bool,
    is_cached_prepare: bool,
    use_bias: bool,
    enable_column_timers: bool,
):
    """Generated rigid-only color dispatch for single-world kernels."""

    _dispatch_rigid_contact = _make_singleworld_rigid_contact_dispatch_func(
        has_mass_splitting=has_mass_splitting,
        packed_contact_headers=packed_contact_headers,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
    )
    _, _dispatch_prepare_rigid_joint, _ = _make_multiworld_rigid_prepare_dispatch_func(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_soft_contact_pd=has_soft_contact_pd,
        cached_prepare=is_cached_prepare,
        enable_column_timers=enable_column_timers,
    )
    _, _dispatch_iterate_rigid_joint, _ = _make_multiworld_rigid_iterate_dispatch_funcs(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        enable_column_timers=enable_column_timers,
        use_bias=use_bias,
    )

    @wp.func
    def _joint_pgs_enabled(cid: wp.int32, joint_pgs_enabled: wp.array[wp.int32]) -> bool:
        if wp.static(selective_joint_pgs):
            if wp.static(is_prepare or is_cached_prepare):
                return joint_pgs_enabled[cid] != wp.int32(0)
            return joint_pgs_enabled[cid] == wp.int32(1)
        return True

    @wp.func
    def _dispatch_rigid_direct_color(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        element_ids_by_color: wp.array[wp.int32],
        color_family_starts: wp.array[wp.int32],
        start: wp.int32,
        count: wp.int32,
        c: wp.int32,
        num_joints: wp.int32,
        joint_pgs_enabled: wp.array[wp.int32],
        num_bodies: wp.int32,
        idt: wp.float32,
        sor_boost: wp.float32,
        lane: wp.int32,
        stride: wp.int32,
    ):
        color_end = start + count
        joint_start = start
        contact_start = start
        if wp.static(has_joints and has_contacts):
            family_base = c * wp.int32(_PER_WORLD_FAST_FAMILIES)
            joint_start = color_family_starts[family_base]
            contact_start = color_family_starts[family_base + wp.int32(1)]
        elif wp.static(has_joints):
            contact_start = color_end

        if wp.static(has_joints and not skip_joint_pgs):
            count_joints = contact_start - joint_start
            base = lane
            while base < count_joints:
                cid = read1d_i32(element_ids_by_color, joint_start + base)
                if _joint_pgs_enabled(cid, joint_pgs_enabled):
                    if wp.static(is_prepare or is_cached_prepare):
                        _dispatch_prepare_rigid_joint(constraints, bodies, particles, copy_state, num_bodies, idt, cid)
                    else:
                        _dispatch_iterate_rigid_joint(
                            constraints,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            idt,
                            sor_boost,
                            cid,
                            wp.int32(1),
                        )
                base = base + stride

        if wp.static(has_contacts):
            count_contacts = color_end - contact_start
            base = lane
            while base < count_contacts:
                cid = read1d_i32(element_ids_by_color, contact_start + base)
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
                    cid - num_joints,
                    contact_start + base,
                    wp.int32(0),
                )
                base = base + stride

    return _dispatch_rigid_direct_color


@functools.cache
def _make_singleworld_persistent_kernel(
    *,
    phase: str,
    revolute_only: bool,
    cloth_support: bool,
    enable_column_timers: bool = False,
    soft_tet_neohookean: bool = False,
    has_joints: bool = True,
    has_contacts: bool = True,
    skip_joint_pgs: bool = False,
    selective_joint_pgs: bool = False,
    has_mass_splitting: bool = True,
    packed_contact_headers: bool = False,
    has_sleeping: bool = True,
    has_soft_contact_pd: bool = True,
    rigid_direct: bool = False,
    patch_friction: bool = False,
):
    """Persistent-grid PGS kernel for the requested phase.

    Per-cid dispatch uses fixed constraint family ranges. The static
    ``cloth_support`` flag keeps deformable code out of rigid-only binaries;
    ``soft_tet_neohookean`` chooses the soft-tet block variant.

    ``enable_column_timers`` brackets each per-cid dispatch with
    ``%globaltimer`` and atomic-adds elapsed microseconds to ``time_us``.
    """
    is_prepare = phase == "prepare"
    is_cached_prepare = phase == "cached_prepare"
    is_iterate = phase == "iterate"
    use_bias = is_iterate  # iterate ON, relax OFF (prepare ignores)

    _dispatch_one_cid, _ = _make_singleworld_dispatch_func(
        revolute_only=revolute_only,
        cloth_support=cloth_support,
        enable_column_timers=enable_column_timers,
        soft_tet_neohookean=soft_tet_neohookean,
        has_joints=has_joints,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_mass_splitting=has_mass_splitting,
        packed_contact_headers=packed_contact_headers,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
        patch_friction=patch_friction,
    )
    _dispatch_rigid_direct_color = _make_singleworld_rigid_direct_color_func(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_mass_splitting=has_mass_splitting,
        packed_contact_headers=packed_contact_headers,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
        enable_column_timers=enable_column_timers,
    )

    @wp.kernel(enable_backward=False, module="unique", grid_stride=False)
    def kernel(
        constraints: ConstraintContainer,
        contact_cols: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        element_ids_by_color: wp.array[wp.int32],
        color_starts: wp.array[wp.int32],
        color_family_starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        color_cursor: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_joints: wp.int32,
        joint_pgs_enabled: wp.array[wp.int32],
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

        if wp.static(rigid_direct):
            if not is_overflow_color:
                _dispatch_rigid_direct_color(
                    constraints,
                    contact_cols,
                    bodies,
                    particles,
                    cc,
                    contacts,
                    copy_state,
                    element_ids_by_color,
                    color_family_starts,
                    start,
                    count,
                    c,
                    num_joints,
                    joint_pgs_enabled,
                    num_bodies,
                    idt,
                    sor_boost,
                    tid,
                    total_num_threads,
                )
                if tid == 0:
                    color_cursor[0] = cursor - 1
                return

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
                    joint_pgs_enabled,
                    num_cloth_triangles,
                    num_cloth_bending,
                    num_soft_tetrahedra,
                    num_soft_hexahedra,
                    num_bodies,
                    idt,
                    sor_boost,
                    cid,
                    start + t_slot,
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
    soft_tet_neohookean: bool = False,
    has_joints: bool = True,
    has_contacts: bool = True,
    skip_joint_pgs: bool = False,
    selective_joint_pgs: bool = False,
    has_mass_splitting: bool = True,
    packed_contact_headers: bool = False,
    has_sleeping: bool = True,
    has_soft_contact_pd: bool = True,
    rigid_direct: bool = False,
    patch_friction: bool = False,
):
    """Single-block tail-fused PGS kernel; same axes as
    :func:`_make_singleworld_persistent_kernel`."""
    is_prepare = phase == "prepare"
    is_cached_prepare = phase == "cached_prepare"
    is_iterate = phase == "iterate"
    use_bias = is_iterate

    _dispatch_one_cid, _ = _make_singleworld_dispatch_func(
        revolute_only=revolute_only,
        cloth_support=cloth_support,
        enable_column_timers=enable_column_timers,
        soft_tet_neohookean=soft_tet_neohookean,
        has_joints=has_joints,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_mass_splitting=has_mass_splitting,
        packed_contact_headers=packed_contact_headers,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
        patch_friction=patch_friction,
    )
    _dispatch_rigid_direct_color = _make_singleworld_rigid_direct_color_func(
        revolute_only=revolute_only,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_mass_splitting=has_mass_splitting,
        packed_contact_headers=packed_contact_headers,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        is_prepare=is_prepare,
        is_cached_prepare=is_cached_prepare,
        use_bias=use_bias,
        enable_column_timers=enable_column_timers,
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
        color_family_starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        color_cursor: wp.array[wp.int32],
        cc: ContactContainer,
        contacts: ContactViews,
        num_joints: wp.int32,
        joint_pgs_enabled: wp.array[wp.int32],
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
            if wp.static(rigid_direct):
                if not is_overflow_color:
                    _dispatch_rigid_direct_color(
                        constraints,
                        contact_cols,
                        bodies,
                        particles,
                        cc,
                        contacts,
                        copy_state,
                        element_ids_by_color,
                        color_family_starts,
                        start,
                        count,
                        c,
                        num_joints,
                        joint_pgs_enabled,
                        num_bodies,
                        idt,
                        sor_boost,
                        lane,
                        fuse_threshold,
                    )
                    _sync_threads()
                    cursor = cursor - 1
                    continue
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
                        joint_pgs_enabled,
                        num_cloth_triangles,
                        num_cloth_bending,
                        num_soft_tetrahedra,
                        num_soft_hexahedra,
                        num_bodies,
                        idt,
                        sor_boost,
                        cid,
                        start + t_slot,
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
    soft_tet_neohookean: bool = False,
    has_joints: bool = True,
    has_contacts: bool = True,
    skip_joint_pgs: bool = False,
    selective_joint_pgs: bool = False,
    has_mass_splitting: bool = True,
    packed_contact_headers: bool = False,
    has_sleeping: bool = True,
    has_soft_contact_pd: bool = True,
    rigid_direct: bool = False,
    patch_friction: bool = False,
):
    """Lazy singleworld kernel builder. Each axis combination is cached
    after first build by the underlying factory's ``functools.cache``."""
    factory = _make_singleworld_fused_kernel if fused else _make_singleworld_persistent_kernel
    return factory(
        phase=phase,
        revolute_only=revolute_only,
        cloth_support=cloth_support,
        enable_column_timers=enable_column_timers,
        soft_tet_neohookean=soft_tet_neohookean,
        has_joints=has_joints,
        has_contacts=has_contacts,
        skip_joint_pgs=skip_joint_pgs,
        selective_joint_pgs=selective_joint_pgs,
        has_mass_splitting=has_mass_splitting,
        packed_contact_headers=packed_contact_headers,
        has_sleeping=has_sleeping,
        has_soft_contact_pd=has_soft_contact_pd,
        rigid_direct=rigid_direct,
        patch_friction=patch_friction,
    )
