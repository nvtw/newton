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
    ContactViews,
    contact_iterate,
    contact_iterate_multi,
    contact_position_iterate,
    contact_prepare_for_iteration,
    contact_world_error,
    contact_world_wrench,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET,
    CONSTRAINT_TYPE_CONTACT,
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
    constraint_get_type,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_make,
)

__all__ = [
    "_PER_WORLD_COLORING_BLOCK_DIM",
    "_STRAGGLER_BLOCK_DIM",
    "_choose_fast_tail_worlds_per_block",
    "_constraint_gather_errors_kernel",
    "_constraint_gather_wrenches_kernel",
    "_constraint_iterate_fast_tail_kernel",
    "_constraint_position_iterate_fast_tail_kernel",
    "_constraint_prepare_fast_tail_kernel",
    "_constraint_relax_fast_tail_kernel",
    "_constraints_to_elements_kernel",
    "_count_elements_per_world_kernel",
    "_integrate_velocities_kernel",
    "_kinematic_interpolate_substep_kernel",
    "_kinematic_prepare_step_kernel",
    "_per_world_jp_coloring_kernel",
    "_rotation_quaternion",
    "_scatter_elements_to_worlds_kernel",
    "_set_kinematic_pose_batch_kernel",
    "_world_csr_count_kernel",
    "_world_csr_scan_kernel",
    "_world_csr_scatter_kernel",
    "pack_body_xforms_kernel",
]


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
    prepare+iterate + register-caching refactor. Overridable via
    ``NEWTON_PHOENX_WORLDS_PER_BLOCK`` for A/B tests.

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
    import os as _os  # noqa: PLC0415

    override = _os.environ.get("NEWTON_PHOENX_WORLDS_PER_BLOCK")
    if override is not None:
        return max(1, int(override))
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
# Per-world CSR bucketing
# ---------------------------------------------------------------------------
#
# The incremental partitioner produces a single global Jones-Plassmann
# coloring over every active constraint. Worlds share no bodies, so
# per-world slices can be carved out in a cheap post-pass instead of
# colouring once per world.
#
# Pipeline: ``_world_csr_count_kernel`` -> ``_world_csr_scan_kernel``
# -> ``_world_csr_prefix_offsets_kernel`` -> ``_world_csr_scatter_kernel``.
# Output layout: ``world_element_ids_by_color[capacity]`` (flat,
# bucketed by world); ``world_color_starts[w, c]`` (per-world
# exclusive-prefix); ``world_csr_offsets[w]`` (base index for world
# ``w``'s slice); ``world_num_colors[w]`` (highest non-empty colour).


@wp.func
def _find_color_of_position(
    color_starts: wp.array[wp.int32],
    num_colors: wp.int32,
    pos: wp.int32,
) -> wp.int32:
    """Return ``c`` s.t. ``color_starts[c] <= pos < color_starts[c + 1]``.
    ``O(log num_colors)`` binary search."""
    lo = wp.int32(0)
    hi = num_colors
    while lo < hi:
        mid = (lo + hi) >> 1
        if color_starts[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@wp.kernel(enable_backward=False)
def _world_csr_count_kernel(
    bodies: BodyContainer,
    elements: wp.array[ElementInteractionData],
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors_arr: wp.array[wp.int32],
    # out
    world_color_counts: wp.array2d[wp.int32],
):
    """Atomic count of per-(world, colour) elements.

    One thread per position in the global ``element_ids_by_color``.
    Reads the compacted primary body from ``elements[cid].bodies[0]``
    so worlds sharing a static world body still route via the dynamic
    body's world id.
    """
    tid = wp.tid()
    n_colors = num_colors_arr[0]
    if n_colors == 0:
        return
    total_elements = color_starts[n_colors]
    if tid >= total_elements:
        return
    c = _find_color_of_position(color_starts, n_colors, tid)
    cid = element_ids_by_color[tid]
    b_primary = elements[cid].bodies[0]
    if b_primary < 0:
        return
    w = bodies.world_id[b_primary]
    wp.atomic_add(world_color_counts, w, c, wp.int32(1))


@wp.kernel(enable_backward=False)
def _world_csr_scan_kernel(
    world_color_counts: wp.array2d[wp.int32],
    num_colors_arr: wp.array[wp.int32],
    # out
    world_color_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    world_totals_shifted: wp.array[wp.int32],
):
    """Per-world serial prefix scan -> ``world_color_starts`` +
    ``world_num_colors`` (highest non-empty colour index). One thread
    per world.

    Additionally stages the per-world total into
    ``world_totals_shifted[w + 1]`` (and ``[0] = 0`` from thread 0)
    so an inclusive ``wp.utils.array_scan`` over the result produces
    ``world_csr_offsets`` directly -- replaces a serial O(num_worlds)
    prefix kernel that scaled poorly past 256 worlds (206 us at 1024
    worlds). The scan writes under O(log num_worlds) kernel work.
    """
    w = wp.tid()
    n_colors = num_colors_arr[0]
    acc = wp.int32(0)
    highest = wp.int32(0)
    for c in range(n_colors):
        world_color_starts[w, c] = acc
        count = world_color_counts[w, c]
        if count > 0:
            highest = c + 1
        acc += count
    world_color_starts[w, n_colors] = acc
    world_num_colors[w] = highest
    # Shifted staging: ``shifted[w + 1] = total_w``; thread 0 stamps
    # ``[0] = 0`` so the downstream inclusive scan lands
    # ``world_csr_offsets[0] = 0`` and
    # ``world_csr_offsets[w + 1] = sum(totals[0..w])``.
    world_totals_shifted[w + 1] = acc
    if w == 0:
        world_totals_shifted[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _world_csr_scatter_kernel(
    bodies: BodyContainer,
    elements: wp.array[ElementInteractionData],
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors_arr: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    # scratch (caller zeroes; atomic cursors -- one per (world, colour))
    world_color_cursor: wp.array2d[wp.int32],
    # out
    world_element_ids_by_color: wp.array[wp.int32],
):
    """Scatter each global-CSR element into its per-world slot.

    Runs after :func:`_world_csr_scan_kernel`. Atomic-cursor bump per
    ``(world, colour)`` gives a unique within-colour index; within-
    colour order is non-deterministic but the partitioner already
    guarantees no two same-colour elements share a body.
    """
    tid = wp.tid()
    n_colors = num_colors_arr[0]
    if n_colors == 0:
        return
    total_elements = color_starts[n_colors]
    if tid >= total_elements:
        return
    c = _find_color_of_position(color_starts, n_colors, tid)
    cid = element_ids_by_color[tid]
    b_primary = elements[cid].bodies[0]
    if b_primary < 0:
        return
    w = bodies.world_id[b_primary]
    local_slot = wp.atomic_add(world_color_cursor, w, c, wp.int32(1))
    dst = world_csr_offsets[w] + world_color_starts[w, c] + local_slot
    world_element_ids_by_color[dst] = cid


# ---------------------------------------------------------------------------
# Per-world graph coloring (one block per world).
# ---------------------------------------------------------------------------
#
# Worlds are independent after static-body nullification -- no element
# in world w references a body in any other world. So Jones-Plassmann
# MIS partitioning can run per-world in parallel instead of device-wide.
# One block per world, each block runs the full JP loop on its world's
# element subset. Output lands directly in the per-world CSR that the
# fast-tail dispatchers consume, skipping the post-coloring bucketing
# pass used by the global-coloring path.
#
# This typically produces the same colour count per world as the
# global path (they're the same local subgraph), but the work is
# parallel across worlds instead of serialized inside a single-block
# scan / compact kernel. Dramatic win at large world counts.


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
def _scatter_elements_to_worlds_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    bodies: BodyContainer,
    world_element_offsets: wp.array[wp.int32],
    # scratch (zeroed by caller each step)
    world_element_cursor: wp.array[wp.int32],
    # out
    world_elements: wp.array[wp.int32],
):
    """Scatter each element's cid into its per-world slot.

    Within-world order is non-deterministic (atomic cursor bumps) --
    that's fine: the downstream JP coloring re-orders by colour, and
    within one colour the ordering doesn't affect PGS convergence
    (graph coloring guarantees no body is shared within a colour).
    """
    tid = wp.tid()
    n = num_elements[0]
    if tid >= n:
        return
    b = elements[tid].bodies[0]
    if b < 0:
        return
    w = bodies.world_id[b]
    slot = wp.atomic_add(world_element_cursor, w, wp.int32(1))
    world_elements[world_element_offsets[w] + slot] = tid


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
    world_commit_count: wp.array[wp.int32],  # [nw] per-round atomic cursor
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
        # Reset per-world round cursor.
        if lane == wp.int32(0):
            world_commit_count[w] = wp.int32(0)
        _sync_threads()

        # Phase 2: find local maxima and commit them. Every lane runs
        # the stride loop the same number of times so that the
        # ``wp.tile`` reduction below sees all lanes at the same point
        # regardless of count -- Warp tile ops are block-collective.
        committed_this_lane = wp.int32(0)
        offset = wp.int32(0)
        while offset < count:
            slot = offset + lane
            committed_here = wp.int32(0)
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
                        out_slot = wp.atomic_add(world_commit_count, w, wp.int32(1))
                        world_element_ids_by_color[base + color_base + out_slot] = eid
                        committed_here = wp.int32(1)
            committed_this_lane = committed_this_lane + committed_here
            offset = offset + stride

        _sync_threads()

        # Block-wide reduce committed count via a tile sum. Using
        # ``wp.tile`` builds a tile from each lane's scalar; ``tile_sum``
        # returns a 1-tile whose scalar is the block-wide total. All
        # lanes see the same value so we can branch uniformly on it.
        committed_tile = wp.tile(committed_this_lane)
        committed_total = wp.tile_sum(committed_tile)
        committed_this_round = committed_total[0]

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
def _constraint_iterate_fast_tail_kernel(
    constraints: ConstraintContainer,
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
    # First cid assigned to a contact column. Joints occupy
    # ``[0, num_joints)``, contacts ``[num_joints, capacity)``, so the
    # dispatch branch can be a cheap register compare instead of a
    # per-cid ``constraint_get_type`` memory read.
    num_joints: wp.int32,
):
    """Main-solve dispatcher: ``num_iterations`` PGS sweeps per world,
    positional bias ON.

    One warp per world; worlds don't share bodies, so per-colour sync is
    warp-local (``__syncwarp``) and several worlds can cohabit one
    128-thread block without risking cross-world ``__syncthreads``
    deadlocks when their ``num_colors`` differ.
    """
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM
    # Padding lanes (launch dim rounded up to the block size) fall past
    # the world count -- early-out before any OOB reads.
    if world_id >= num_worlds:
        return

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    it = wp.int32(0)
    while it < num_iterations:
        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + 1]
            count = end - start

            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                if cid < num_joints:
                    actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, True)
                else:
                    contact_iterate(constraints, cid, bodies, idt, cc, contacts, True)
                base += _STRAGGLER_BLOCK_DIM

            _sync_warp()
            c += 1

        it += 1


@wp.kernel(enable_backward=False)
def _constraint_prepare_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_worlds: wp.int32,
    num_joints: wp.int32,
):
    """Prepare dispatcher: one sweep per world. Computes effective
    masses, velocity bias, applies warm-start impulse."""
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM
    if world_id >= num_worlds:
        return

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

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
                contact_prepare_for_iteration(constraints, cid, bodies, idt, cc, contacts)
            base += _STRAGGLER_BLOCK_DIM

        _sync_warp()
        c += 1


@wp.kernel(enable_backward=False)
def _constraint_prepare_plus_iterate_fast_tail_kernel(
    constraints: ConstraintContainer,
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
):
    """Fused prepare + main-solve dispatcher.

    Runs the prepare pass and the ``num_iterations`` PGS sweeps in a
    single kernel launch. Saves one kernel launch per substep; more
    importantly, per-world setup (``world_id``, ``n_colors``,
    ``world_base``) is computed once and the register-resident
    per-world state survives across the prepare -> iterate transition
    for free.

    The prepare outputs (``cc.derived``) still round-trip through
    global memory between prepare and iterate -- different cids in a
    warp touch different contacts, so we can't keep them in
    registers. But the kernel-launch fixed cost is eliminated and
    some loop-boundary scalars may be reused by the compiler.
    """
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM
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
                contact_prepare_for_iteration(constraints, cid, bodies, idt, cc, contacts)
            base += _STRAGGLER_BLOCK_DIM

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
                    actuated_double_ball_socket_iterate_multi(
                        constraints, cid, bodies, idt, True, inner_sweeps
                    )
                else:
                    contact_iterate_multi(
                        constraints, cid, bodies, idt, cc, contacts, True, inner_sweeps
                    )
                base += _STRAGGLER_BLOCK_DIM

            _sync_warp()
            c += 1

        it_outer += 1


@wp.kernel(enable_backward=False)
def _constraint_position_iterate_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    num_iterations: wp.int32,
    num_worlds: wp.int32,
    num_joints: wp.int32,
):
    """XPBD position-iteration dispatcher for contact tangent drift.

    Runs between ``integrate_velocities`` and ``relax_velocities``.
    Each per-slot position iterate is gated by the slip threshold so
    sliding pairs are skipped; the Coulomb-clamped velocity friction
    handles them.
    """
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM
    if world_id >= num_worlds:
        return

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    it = wp.int32(0)
    while it < num_iterations:
        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + 1]
            count = end - start

            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                # Only contact cids participate in XPBD position drift;
                # joint cids at ``cid < num_joints`` are skipped.
                if cid >= num_joints:
                    contact_position_iterate(constraints, cid, bodies, cc)
                base += _STRAGGLER_BLOCK_DIM

            _sync_warp()
            c += 1

        it += 1


@wp.kernel(enable_backward=False)
def _constraint_relax_fast_tail_kernel(
    constraints: ConstraintContainer,
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
):
    """Box2D v3 TGS-soft relax dispatcher: ``num_iterations`` sweeps
    with positional bias OFF (enforces ``Jv = 0``)."""
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM
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
                actuated_double_ball_socket_iterate_multi(
                    constraints, cid, bodies, idt, False, num_iterations
                )
            else:
                contact_iterate_multi(
                    constraints, cid, bodies, idt, cc, contacts, False, num_iterations
                )
            base += _STRAGGLER_BLOCK_DIM

        _sync_warp()
        c += 1


@wp.kernel(enable_backward=False)
def _constraints_to_elements_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.array[wp.int32],
    elements: wp.array[ElementInteractionData],
):
    """Project every active constraint into ``ElementInteractionData``.

    Only the two body indices matter to the graph colourer; static
    bodies get collapsed to ``-1`` and the dynamic body (if any) is
    compacted to slot 0.

    Launched at ``dim = constraint_capacity`` because the active count
    is device-held; inactive lanes early-out without writing. Every
    downstream reader (``partitioning_adjacency_{count,store}``,
    ``_count_elements_per_world``, ``_scatter_elements_to_worlds``,
    ``_per_world_jp_coloring``) gates on ``num_active_constraints``,
    so tail slots are never read and don't need a sentinel value.
    """
    tid = wp.tid()
    n = num_constraints[0]
    if tid >= n:
        return
    b1 = constraint_get_body1(constraints, tid)
    b2 = constraint_get_body2(constraints, tid)
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
    bodies: BodyContainer,
    num_constraints: wp.int32,
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
    t = constraint_get_type(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
        force, torque = actuated_double_ball_socket_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_CONTACT:
        force, torque = contact_world_wrench(constraints, cid, idt, cc, contacts)
    out[cid] = wp.spatial_vector(force, torque)


@wp.kernel(enable_backward=False)
def _constraint_gather_errors_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    # out
    out: wp.array[wp.spatial_vector],
):
    """Per-cid position-level constraint residual: ``top`` = linear
    [m], ``bottom`` = angular [rad]. Pure read from current body pose
    + persisted per-type state; no body mutation."""
    cid = wp.tid()
    if cid >= num_constraints:
        return
    t = constraint_get_type(constraints, cid)
    zero = wp.spatial_vector(wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0))
    err = zero
    if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
        err = actuated_double_ball_socket_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_CONTACT:
        err = contact_world_error(constraints, cid)
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
