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
def _reset_roots(entries: wp.array[wp.int64]):
    body = wp.tid()
    entries[body] = _pack_entry(body, 0)


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
):
    slot = wp.tid()
    key = num_bodies + 1
    if slot < starts[num_colors[0]]:
        witness = witnesses[slot]
        if witness == num_bodies:
            key = witness
        elif witness >= 0:
            key = _find(entries, witness)
    keys[slot] = key
    slots[slot] = slot


# A dozen columns can already span a long sequence of dependent contact colors.
_LARGE_ISLAND_COLUMNS = wp.constant(12)


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
        # Start long islands early so their final colors overlap shorter islands.
        last = i + _LARGE_ISLAND_COLUMNS - 1
        if last < ordered.shape[0] and keys[last] == root:
            roots[wp.atomic_add(count, 0, 1)] = root
        else:
            roots[roots.shape[0] - 1 - wp.atomic_add(count, 1, 1)] = root
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
    count: wp.array[int],
):
    i = wp.tid()
    if i == 0:
        count[0] += count[1]
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
    ):
        block, lane = wp.tid()
        total = count[0]
        large = total - count[1]
        for index in range(block, total, num_blocks):
            slot = index
            if index >= large:
                slot = roots.shape[0] - 1 - (index - large)
            root = roots[slot]
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
                    lane,
                    wp.static(ONE_LAUNCH_BLOCK_DIM),
                )
                _sync_threads()
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
        self.starts = wp.empty_like(self.roots)
        self.ends = wp.empty_like(self.roots)
        self.count = wp.empty(2, dtype=int, device=device)
        self.blocks = min(self.roots.size, 4 * device.sm_count)

    def build(self, world: PhoenXWorld):
        partition = world._partitioner
        capacity = self.columns.size
        num_bodies = world.num_bodies
        sentinel = num_bodies + 1
        wp.launch(_reset_roots, num_bodies, [self.entries], device=world.device)
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
            ],
            device=world.device,
        )
        wp.utils.radix_sort_pairs(self.keys, self.slots, capacity, end_bit=sentinel.bit_length())
        self.count.zero_()
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
                self.count,
            ],
            device=world.device,
        )

    def iterate(self, world: PhoenXWorld, kernel, idt: float):
        """Advance every island through all of its colors, in their original order."""
        wp.launch(
            _get_island_iteration(kernel.phoenx_axes),
            (self.blocks, ONE_LAUNCH_BLOCK_DIM),
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
                self.roots,
                self.count,
                self.starts,
                self.ends,
                self.colors,
                self.color_ends,
                self.columns,
                self.families,
                self.blocks,
            ],
            block_dim=ONE_LAUNCH_BLOCK_DIM,
            device=world.device,
        )
