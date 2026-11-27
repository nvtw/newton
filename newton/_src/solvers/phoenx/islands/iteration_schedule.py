# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Schedule the colors of a rigid PGS iteration independently per island.

Only shared writable bodies impose an order between columns. A stable sort of
the existing color-ordered columns by dynamic component preserves that order
inside each component. Its block can then advance colors with a block barrier,
so one launch traverses all colors without a device-wide boundary between them.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import warp as wp

from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_owned_elsewhere,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.islands.island_builder import _find, _pack_entry, _unite
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.simulation_kernels import (
    _PER_WORLD_FAST_FAMILIES,
    ONE_LAUNCH_BLOCK_DIM,
    _make_singleworld_head_color_func,
    _sync_threads,
)

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.simulation import PhoenXWorld


@wp.kernel(enable_backward=False)
def _reset_roots(
    entries: wp.array[wp.int64],
    sizes: wp.array[int],
    first_color: wp.array[int],
    last_color: wp.array[int],
    starts: wp.array[int],
    ends: wp.array[int],
    count: wp.array[int],
):
    """Make every body its own root and clear the per-root island census.

    The last thread clears the merged island, keyed past every body, so it
    keeps no stale range when this step has none.
    """
    body = wp.tid()
    sizes[body] = 0
    first_color[body] = 2147483647
    last_color[body] = -1
    if body < entries.shape[0]:
        entries[body] = _pack_entry(body, 0)
    else:
        starts[body] = 0
        ends[body] = 0
        count[0] = 0


@wp.kernel(enable_backward=False)
def _join_columns(
    entries: wp.array[wp.int64],
    witnesses: wp.array[int],
    columns: wp.array[int],
    starts: wp.array[int],
    num_colors: wp.array[int],
    bodies: BodyContainer,
    constraints: ConstraintContainer,
    contacts: ContactColumnContainer,
    num_joints: int,
    joint_pgs_enabled: wp.array[int],
    sleeping: bool,
):
    slot = wp.tid()
    if slot >= starts[num_colors[0]]:
        return
    column = columns[slot]
    witnesses[slot] = -1
    first = int(0)
    second = int(0)
    if column < num_joints:
        # Prepare-only joints also write rows during the warm-start sweep.
        if joint_pgs_enabled[column] == 0:
            return
        first = constraint_get_body1(constraints, column)
        second = constraint_get_body2(constraints, column)
    else:
        local = column - num_joints
        if contact_owned_elsewhere(contacts, local):
            return
        first = contact_get_body1(contacts, local)
        second = contact_get_body2(contacts, local)
    first_dynamic = first >= 0 and bodies.motion_type[first] == MOTION_DYNAMIC
    second_dynamic = second >= 0 and bodies.motion_type[second] == MOTION_DYNAMIC
    if (
        sleeping
        and (not first_dynamic or bodies.island_root[first] >= 0)
        and (not second_dynamic or bodies.island_root[second] >= 0)
    ):
        return
    if first_dynamic:
        witnesses[slot] = first
        if second_dynamic:
            _unite(entries, first, second)
    elif second_dynamic:
        witnesses[slot] = second
    else:
        # Keep the original row math even for a pair with two immovable endpoints.
        witnesses[slot] = bodies.motion_type.shape[0]


@wp.kernel(enable_backward=False)
def _key_columns(
    entries: wp.array[wp.int64],
    witnesses: wp.array[int],
    starts: wp.array[int],
    num_colors: wp.array[int],
    keys: wp.array[int],
    slots: wp.array[int],
    num_bodies: int,
    sizes: wp.array[int],
    first_color: wp.array[int],
    last_color: wp.array[int],
):
    slot = wp.tid()
    key = num_bodies + 1
    if slot < starts[num_colors[0]]:
        witness = witnesses[slot]
        if witness == num_bodies:
            key = witness
        elif witness >= 0:
            key = _find(entries, witness)
        if key <= num_bodies:
            wp.atomic_add(sizes, key, 1)
            color = wp.lower_bound(starts, 1, num_colors[0] + 1, slot + 1) - 1
            wp.atomic_min(first_color, key, color)
            wp.atomic_max(last_color, key, color)
    keys[slot] = key
    slots[slot] = slot


@wp.kernel(enable_backward=False)
def _merge_wide_islands(
    keys: wp.array[int],
    sizes: wp.array[int],
    first_color: wp.array[int],
    last_color: wp.array[int],
    num_bodies: int,
    merge_width: int,
):
    slot = wp.tid()
    key = keys[slot]
    # An island whose colors are on average wider than one block's lanes needs
    # several block passes per color; narrower, deeper ones would only add grid
    # barriers. Wide islands join the shared key of the immovable pairs, which
    # may run in any color group: sorted by slot, they form one island whose
    # color runs span all of them.
    if key < num_bodies and sizes[key] > merge_width * (last_color[key] - first_color[key] + 1):
        keys[slot] = num_bodies


@wp.kernel(enable_backward=False)
def _request_island_iterations(
    keys: wp.array[int],
    columns: wp.array[int],
    starts: wp.array[int],
    num_colors: wp.array[int],
    bodies: BodyContainer,
    constraints: ConstraintContainer,
    contacts: ContactColumnContainer,
    num_joints: int,
    body_iterations: wp.array[wp.int32],
    island_iterations: wp.array[int],
):
    """Raise each island's sweep count to the largest request of its dynamic bodies."""
    slot = wp.tid()
    if slot >= starts[num_colors[0]]:
        return
    key = keys[slot]
    if key >= island_iterations.shape[0]:
        return
    column = columns[slot]
    first = int(0)
    second = int(0)
    if column < num_joints:
        first = constraint_get_body1(constraints, column)
        second = constraint_get_body2(constraints, column)
    else:
        first = contact_get_body1(contacts, column - num_joints)
        second = contact_get_body2(contacts, column - num_joints)
    request = int(0)
    if first >= 0 and bodies.motion_type[first] == MOTION_DYNAMIC:
        request = int(body_iterations[first])
    if second >= 0 and bodies.motion_type[second] == MOTION_DYNAMIC:
        request = wp.max(request, int(body_iterations[second]))
    if request > 0:
        wp.atomic_max(island_iterations, key, request)


@wp.kernel(enable_backward=False)
def _index_islands(
    keys: wp.array[int],
    slots: wp.array[int],
    columns: wp.array[int],
    ordered: wp.array[int],
    starts: wp.array[int],
    ends: wp.array[int],
    roots: wp.array[int],
    count: wp.array[int],
    sentinel: int,
):
    i = wp.tid()
    root = keys[i]
    if root == sentinel:
        return
    ordered[i] = columns[slots[i]]
    if i == 0 or keys[i - 1] != root:
        starts[root] = i
        roots[wp.atomic_add(count, 0, 1)] = root
    if i + 1 == ordered.shape[0] or keys[i + 1] != root:
        ends[root] = i + 1


@wp.func
def _first_slot_at_least(slots: wp.array[int], start: int, stop: int, value: int):
    # Warp's lower_bound clamps to the last element. Here the insertion
    # point may be one past the end (e.g. a color containing only joints).
    if slots[stop - 1] < value:
        return stop
    return wp.lower_bound(slots, start, stop, value)


@wp.kernel(enable_backward=False)
def _index_colors(
    keys: wp.array[int],
    slots: wp.array[int],
    island_ends: wp.array[int],
    sentinel: int,
    starts: wp.array[int],
    num_colors: wp.array[int],
    family_starts: wp.array[int],
    mixed: bool,
    colors: wp.array[int],
    ends: wp.array[int],
    families: wp.array[int],
):
    i = wp.tid()
    root = keys[i]
    if root == sentinel:
        return
    slot = slots[i]
    color = wp.lower_bound(starts, 1, num_colors[0] + 1, slot + 1) - 1
    colors[i] = color
    if i > 0 and keys[i - 1] == root and slots[i - 1] >= starts[color]:
        return
    end = _first_slot_at_least(slots, i, island_ends[root], starts[color + 1])
    ends[i] = end
    if mixed:
        family = wp.static(_PER_WORLD_FAST_FAMILIES)
        families[i * family] = i
        families[i * family + 1] = _first_slot_at_least(slots, i, end, family_starts[color * family + 1])


@wp.kernel(enable_backward=False)
def _key_island_order(
    roots: wp.array[int],
    count: wp.array[int],
    first_color: wp.array[int],
    last_color: wp.array[int],
    color_capacity: int,
    keys: wp.array[int],
    order: wp.array[int],
):
    """Key every island by its color span, longest first; unused slots sort last.

    Spans are below ``color_capacity``, which bounds the keys and the sort's bits.
    """
    i = wp.tid()
    if i >= count[0]:
        keys[i] = color_capacity
        order[i] = -1
        return
    root = roots[i]
    keys[i] = color_capacity - (last_color[root] - first_color[root] + 1)
    order[i] = root


#: Warps of each block that sweep the merged island; the others sweep small islands.
_MERGED_WARPS = 4
_MERGED_THREADS = wp.constant(32 * _MERGED_WARPS)
_ISLAND_THREADS = wp.constant(ONE_LAUNCH_BLOCK_DIM - 32 * _MERGED_WARPS)

# Hardware barrier 1 joins the merged warps of a block, 2 its small-island
# warps. The ids must be literals: a register id makes the compiler reserve
# all sixteen barriers per block, which caps residency at one block per SM.
_MERGED_SYNC = f'asm volatile("barrier.sync 1, {32 * _MERGED_WARPS};" ::: "memory");'
_ISLAND_SYNC = f'asm volatile("barrier.sync 2, {ONE_LAUNCH_BLOCK_DIM - 32 * _MERGED_WARPS};" ::: "memory");'


@wp.func_native(f"""
#if defined(__CUDA_ARCH__)
{_MERGED_SYNC}
#endif
""")
def _sync_merged_warps(): ...


@wp.func_native(f"""
#if defined(__CUDA_ARCH__)
{_ISLAND_SYNC}
#endif
""")
def _sync_island_warps(): ...


# The grid barrier of ``_grid_sync`` for the merged warps of every block,
# which lead with thread 0; the small-island warps of the block go on.
@wp.func_native(f"""
#if defined(__CUDA_ARCH__)
{_MERGED_SYNC}
if (threadIdx.x == 0) {{
    int* count = (int*)state.data;
    volatile int* generation = count + 1;
    const int seen = *generation;
    __threadfence();
    if (atomicAdd(count, 1) == num_blocks - 1) {{
        atomicExch(count, 0);
        __threadfence();
        atomicAdd((int*)generation, 1);
    }} else {{
        unsigned int backoff = 8;
        while (*generation == seen) {{
            __nanosleep(backoff);
            backoff = backoff < 128 ? 2 * backoff : 128;
        }}
    }}
    __threadfence();
}}
{_MERGED_SYNC}
#endif
""")
def _merged_grid_sync(state: wp.array[wp.int32], num_blocks: wp.int32): ...


@wp.func
def _sync_island_group(has_merged: bool):
    """Join the warps that walk small islands: all of the block without a merged island."""
    if has_merged:
        _sync_island_warps()
    else:
        _sync_threads()


@functools.cache
def _get_island_iteration(axes):
    phase = dict(axes)["phase"]
    solve_color = _make_singleworld_head_color_func(**dict(axes))

    @wp.kernel(
        name=f"phoenx_{phase}_islands",
        enable_backward=False,
        module="unique",
        module_options={"fast_math": phase == "iterate"},
        grid_stride=False,
    )
    def iterate(
        constraints: ConstraintContainer,
        contacts: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        idt: float,
        sor: float,
        cc: ContactContainer,
        views: ContactViews,
        num_joints: int,
        joint_enabled: wp.array[int],
        num_bodies: int,
        copies: CopyStateContainer,
        roots: wp.array[int],
        count: wp.array[int],
        starts: wp.array[int],
        ends: wp.array[int],
        colors: wp.array[int],
        color_ends: wp.array[int],
        columns: wp.array[int],
        families: wp.array[int],
        num_blocks: int,
        merged: int,
        fuse_columns: int,
        barrier: wp.array[wp.int32],
        claims: wp.array[wp.int32],
        iteration: int,
        island_iterations: wp.array[int],
    ):
        block, lane = wp.tid()
        has_merged = ends[merged] > starts[merged]
        # Every island runs the first sweep; ``island_iterations`` is only
        # read for later sweeps, which islands with fewer requested ones skip.
        if iteration > 0 and has_merged:
            has_merged = iteration < island_iterations[merged]
        # The merged large islands would leave one block walking them long
        # after the others finish. The merged warps of the whole co-resident
        # grid share each of their wide colors, with a grid barrier between
        # colors; block 0 alone walks runs of narrow colors. Both are chains
        # of latency-bound color steps, so the remaining warps of every block
        # walk the independent small islands meanwhile.
        if has_merged and lane < _MERGED_THREADS:
            i = starts[merged]
            # 0: nothing written, 1: block 0 wrote a narrow color, 2: the grid wrote a wide one.
            pending = int(0)
            while i < ends[merged]:
                end = color_ends[i]
                wide = end - i > fuse_columns
                thread = lane
                threads = int(_MERGED_THREADS)
                if wide:
                    if pending != 0:
                        _merged_grid_sync(barrier, wp.int32(num_blocks))
                    thread = block * _MERGED_THREADS + lane
                    threads = num_blocks * _MERGED_THREADS
                elif pending == 2:
                    _merged_grid_sync(barrier, wp.int32(num_blocks))
                if wide or block == 0:
                    solve_color(
                        constraints,
                        contacts,
                        bodies,
                        particles,
                        idt,
                        sor,
                        columns,
                        families,
                        colors,
                        cc,
                        views,
                        num_joints,
                        joint_enabled,
                        0,
                        0,
                        0,
                        0,
                        num_bodies,
                        copies,
                        1,
                        i,
                        end - i,
                        i,
                        False,
                        thread,
                        threads,
                    )
                if wide:
                    pending = 2
                else:
                    _sync_merged_warps()
                    pending = 1
                i = end
            return
        # Without a merged island every warp walks small islands.
        island_lane = lane
        island_threads = int(ONE_LAUNCH_BLOCK_DIM)
        if has_merged:
            island_lane = lane - _MERGED_THREADS
            island_threads = int(_ISLAND_THREADS)
        # Islands are sorted longest first. Each block's island warps claim
        # the next one when they finish their last, which keeps the longest
        # serial chains of colors on different blocks.
        total = count[0]
        while True:
            if island_lane == 0:
                claim = wp.atomic_add(barrier, 2, 1)
                # Every block makes one claim past the last island; the final
                # one resets the cursor for the next launch.
                if claim == total + num_blocks - 1:
                    barrier[2] = 0
                claims[block] = claim
            _sync_island_group(has_merged)
            index = claims[block]
            _sync_island_group(has_merged)
            if index >= total:
                break
            root = roots[index]
            if root == merged:
                continue
            if iteration > 0:
                if iteration >= island_iterations[root]:
                    continue
            i = starts[root]
            end_of_island = ends[root]
            while i < end_of_island:
                end = color_ends[i]
                solve_color(
                    constraints,
                    contacts,
                    bodies,
                    particles,
                    idt,
                    sor,
                    columns,
                    families,
                    colors,
                    cc,
                    views,
                    num_joints,
                    joint_enabled,
                    0,
                    0,
                    0,
                    0,
                    num_bodies,
                    copies,
                    1,
                    i,
                    end - i,
                    i,
                    False,
                    island_lane,
                    island_threads,
                )
                _sync_island_group(has_merged)
                i = end

    return iterate


class IslandIterationSchedule:
    """CUDA graph scratch for the current collision step's dynamic-body components."""

    def __init__(self, world: PhoenXWorld):
        device = world.device
        capacity = world._constraint_capacity
        num_bodies = world.num_bodies
        self.entries = wp.empty(num_bodies, dtype=wp.int64, device=device)
        self.witnesses = wp.empty(capacity, dtype=int, device=device)
        # Warp's stable radix sort needs a second capacity-sized buffer in each array.
        self.keys = wp.empty(2 * capacity, dtype=int, device=device)
        self.slots = wp.empty_like(self.keys)
        self.columns = wp.empty_like(self.witnesses)
        self.colors = wp.empty_like(self.witnesses)
        self.color_ends = wp.empty_like(self.witnesses)
        self.mixed = world.num_joints > 0 and world.max_contact_columns > 0
        self.families = wp.empty(capacity * _PER_WORLD_FAST_FAMILIES if self.mixed else 1, dtype=int, device=device)
        self.roots = wp.empty(num_bodies + 1, dtype=int, device=device)
        # Warp's stable radix sort needs a second buffer in each array.
        self.order_keys = wp.empty(2 * (num_bodies + 1), dtype=int, device=device)
        self.order = wp.empty(2 * (num_bodies + 1), dtype=int, device=device)
        self.starts = wp.empty_like(self.roots)
        self.ends = wp.empty_like(self.roots)
        self.count = wp.empty(1, dtype=int, device=device)
        self.sizes = wp.empty(num_bodies + 1, dtype=int, device=device)
        self.first_color = wp.empty_like(self.sizes)
        self.last_color = wp.empty_like(self.sizes)
        # Merged grid barrier (arrivals, generation) and the small-island cursor.
        self.barrier = wp.zeros(3, dtype=wp.int32, device=device)
        #: The island each block's island warps claimed last.
        self.claims = wp.zeros(4 * max(1, int(device.sm_count)), dtype=wp.int32, device=device)
        self._blocks = {}
        #: Islands with more columns per color are merged and swept by the grid.
        self.merge_width = ONE_LAUNCH_BLOCK_DIM
        #: Colors of the merged island with more columns use the whole grid;
        #: ``None`` uses the one-launch sweep's fuse threshold.
        self.fuse_columns = None
        #: Biased sweeps of each island (merged one at ``num_bodies``) when
        #: bodies request more than ``solver_iterations``; unread otherwise.
        self.island_iterations = wp.empty(
            num_bodies + 1 if world.body_solver_iterations is not None else 1, dtype=int, device=device
        )

    def build(self, world: PhoenXWorld):
        partition = world._partitioner
        capacity = self.columns.size
        num_bodies = world.num_bodies
        sentinel = num_bodies + 1
        wp.launch(
            _reset_roots,
            num_bodies + 1,
            [self.entries, self.sizes, self.first_color, self.last_color, self.starts, self.ends, self.count],
            device=world.device,
        )
        wp.launch(
            _join_columns,
            capacity,
            [
                self.entries,
                self.witnesses,
                partition.element_ids_by_color,
                partition.color_starts,
                partition.num_colors,
                world.bodies,
                world.constraints,
                world._contact_cols,
                world.num_joints,
                world._joint_pgs_enabled,
                world._sleeping_enabled,
            ],
            device=world.device,
        )
        wp.launch(
            _key_columns,
            capacity,
            [
                self.entries,
                self.witnesses,
                partition.color_starts,
                partition.num_colors,
                self.keys,
                self.slots,
                num_bodies,
                self.sizes,
                self.first_color,
                self.last_color,
            ],
            device=world.device,
        )
        wp.launch(
            _merge_wide_islands,
            capacity,
            [self.keys, self.sizes, self.first_color, self.last_color, num_bodies, int(self.merge_width)],
            device=world.device,
        )
        if world.body_solver_iterations is not None:
            self.island_iterations.fill_(world.solver_iterations)
            wp.launch(
                _request_island_iterations,
                capacity,
                [
                    self.keys,
                    partition.element_ids_by_color,
                    partition.color_starts,
                    partition.num_colors,
                    world.bodies,
                    world.constraints,
                    world._contact_cols,
                    world.num_joints,
                    world.body_solver_iterations,
                    self.island_iterations,
                ],
                device=world.device,
            )
        wp.utils.radix_sort_pairs(self.keys, self.slots, capacity, end_bit=sentinel.bit_length())
        wp.launch(
            _index_islands,
            capacity,
            [
                self.keys,
                self.slots,
                partition.element_ids_by_color,
                self.columns,
                self.starts,
                self.ends,
                self.roots,
                self.count,
                sentinel,
            ],
            device=world.device,
        )
        wp.launch(
            _index_colors,
            capacity,
            [
                self.keys,
                self.slots,
                self.ends,
                sentinel,
                partition.color_starts,
                partition.num_colors,
                partition.color_family_starts,
                self.mixed,
                self.colors,
                self.color_ends,
                self.families,
            ],
            device=world.device,
        )
        # Color spans stay below the color table's length.
        color_capacity = partition.color_starts.shape[0]
        wp.launch(
            _key_island_order,
            num_bodies + 1,
            [self.roots, self.count, self.first_color, self.last_color, color_capacity, self.order_keys, self.order],
            device=world.device,
        )
        wp.utils.radix_sort_pairs(self.order_keys, self.order, num_bodies + 1, end_bit=color_capacity.bit_length())

    def _grid_blocks(self, world: PhoenXWorld, kernel) -> int:
        """Blocks of the island sweep, all co-resident for its grid barrier.

        Up to four blocks per SM, as many as the registers of ``kernel`` fit;
        see ``PhoenXWorld._one_launch_grid_blocks`` for why fitting blocks are
        eventually all resident.
        """
        blocks = self._blocks.get(kernel)
        if blocks is None:
            block_dim = ONE_LAUNCH_BLOCK_DIM
            try:
                regs = int(wp.get_cuda_kernel_properties(kernel, world.device, block_dim=block_dim)["register_count"])
            except Exception:
                regs = 255
            regs_per_block = (-(-max(regs, 1) // 8) * 8) * block_dim
            fit = min(65536 // regs_per_block, 1024 // block_dim)
            blocks = max(1, int(world.device.sm_count)) * max(1, min(4, fit))
            self._blocks[kernel] = blocks
        return blocks

    def iterate(self, world: PhoenXWorld, kernel, idt: float, iteration: int = 0):
        """Advance every island through all of its colors, in their original order.

        The merged island needs a co-resident grid for its barrier. Its warps
        share each block with warps that walk the small islands, which share
        no body with it and need no grid barrier, so both chains of color
        steps proceed in one launch.

        Islands are independent sets, so each may run its own number of
        Gauss-Seidel sweeps: biased sweep ``iteration`` skips the islands
        whose bodies requested fewer (see ``body_solver_iterations``). Other
        phases pass ``0`` and sweep every island.
        """
        if world.body_solver_iterations is None:
            iteration = 0
        island_kernel = _get_island_iteration(kernel.phoenx_axes)
        blocks = self._grid_blocks(world, island_kernel)
        wp.launch(
            island_kernel,
            (blocks, ONE_LAUNCH_BLOCK_DIM),
            [
                world.constraints,
                world._contact_cols,
                world.bodies,
                world._particles_or_sentinel(),
                idt,
                world.sor_boost,
                world._contact_container,
                world._active_contact_views(),
                world.num_joints,
                world._joint_pgs_enabled,
                world.num_bodies,
                world._copy_state,
                self.order,
                self.count,
                self.starts,
                self.ends,
                self.colors,
                self.color_ends,
                self.columns,
                self.families,
                blocks,
                world.num_bodies,
                int(world._fuse_threshold if self.fuse_columns is None else self.fuse_columns),
                self.barrier,
                self.claims,
                iteration,
                self.island_iterations,
            ],
            block_dim=ONE_LAUNCH_BLOCK_DIM,
            device=world.device,
        )
