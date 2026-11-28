# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental broad phase on an implicit binary BVH with stackless traversal.

The tree itself is :class:`~newton._src.geometry.implicit_bvh.ImplicitBvh`;
this module only adds the broad-phase leaf bounds (gap, swept displacement and
float rounding slack) and the pair filter. Its complete binary topology
isolates the cost of BVH pair-tree traversal; it is not an implementation of
the compact Chitalu et al. node-index mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .broad_phase_common import (
    EmptyFilterData,
    PairExclusionIndex,
    is_shape_pair_same_body_filtered,
    keep_all_filter,
    precompute_world_map,
    write_pair,
)
from .broad_phase_sap import _make_sap_process_pair_func
from .flags import ShapeFlags
from .implicit_bvh import (
    ImplicitBvh,
    ImplicitBvhData,
    ImplicitBvhNode,
    _node_load,
    aabb_disjoint,
    implicit_bvh_next_subtree,
)

wp.set_module_options({"enable_backward": False})

# Query lanes loop over the (active leaf, ancestor level) work items.
_QUERY_MAX_THREADS = 1 << 21
# Threads of the kept-pair compaction, which loops over the kept pairs.
_COMPACT_THREADS = 1 << 18


@wp.func
def _spread_20(v: wp.uint64) -> wp.uint64:
    v = (v | (v << wp.uint64(32))) & wp.uint64(0x001F00000000FFFF)
    v = (v | (v << wp.uint64(16))) & wp.uint64(0x001F0000FF0000FF)
    v = (v | (v << wp.uint64(8))) & wp.uint64(0x100F00F00F00F00F)
    v = (v | (v << wp.uint64(4))) & wp.uint64(0x10C30C30C30C30C3)
    return (v | (v << wp.uint64(2))) & wp.uint64(0x1249249249249249)


@wp.kernel(enable_backward=False)
def _tree_keys(
    shape_index: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    gap: wp.array[float],
    displacement: wp.array[wp.vec3],
    keys: wp.array[wp.uint64],
    values: wp.array[int],
    active_count: wp.array[wp.int32],
):
    leaf = wp.tid()
    if leaf == 0:
        active_count[0] = 0
    shape = shape_index[leaf]
    center = (lower[shape] + upper[shape]) * 0.5
    if displacement.shape[0] > 0:
        center += displacement[shape] * 0.5
    # Quantization affects traversal locality only; node bounds remain exact.
    # 16 bits per axis on a 2 cm grid span 1.3 km; six radix passes sort the 48-bit keys.
    x = wp.uint64(wp.int32(wp.clamp(wp.floor(center[0] * 50.0) + 32768.0, 0.0, 65535.0)))
    y = wp.uint64(wp.int32(wp.clamp(wp.floor(center[1] * 50.0) + 32768.0, 0.0, 65535.0)))
    z = wp.uint64(wp.int32(wp.clamp(wp.floor(center[2] * 50.0) + 32768.0, 0.0, 65535.0)))
    keys[leaf] = (_spread_20(x) << wp.uint64(2)) | (_spread_20(y) << wp.uint64(1)) | _spread_20(z)
    values[leaf] = leaf


@wp.func
def _leaf_bounds(
    shape: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    gap: wp.array[float],
    displacement: wp.array[wp.vec3],
    reuse_slack: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Return a leaf's swept bound; a positive ``reuse_slack`` [m] also covers the next pass's sweep."""
    margin = wp.float32(0.0)
    if gap.shape[0] > 0:
        margin = gap[shape]
    shift = wp.vec3(0.0, 0.0, 0.0)
    if displacement.shape[0] > 0:
        shift = displacement[shape]
    expand = wp.vec3(margin, margin, margin)
    low = wp.min(lower[shape], lower[shape] + shift) - expand
    high = wp.max(upper[shape], upper[shape] + shift) + expand
    if reuse_slack > 0.0:
        # The next pass sweeps from about ``lower + shift`` by about ``shift`` again;
        # the slack absorbs the change of motion over one pass.
        pad = wp.vec3(reuse_slack + 0.25 * wp.max(wp.abs(shift[0]), wp.max(wp.abs(shift[1]), wp.abs(shift[2]))))
        low = wp.min(low, lower[shape] + 2.0 * shift - expand) - pad
        high = wp.max(high, upper[shape] + 2.0 * shift + expand) + pad
    # At distant world coordinates, float32 rounding can make a swept bound
    # narrower than the equivalent moving-AABB test (which combines the two
    # gaps before subtraction). Expand by a few ulps of the coordinate scale;
    # the shared exact pair filter still decides which contacts are accepted.
    rounding = wp.vec3(
        wp.max(wp.abs(low[0]), wp.abs(high[0])) * wp.float32(4.76837158e-7) + wp.float32(1.0e-6),
        wp.max(wp.abs(low[1]), wp.abs(high[1])) * wp.float32(4.76837158e-7) + wp.float32(1.0e-6),
        wp.max(wp.abs(low[2]), wp.abs(high[2])) * wp.float32(4.76837158e-7) + wp.float32(1.0e-6),
    )
    return low - rounding, high + rounding


def _make_pair_kernels(filter_func: Any, filter_data_type: Any, passive_shape: Any = None, reuse: bool = False):
    process_pair = _make_sap_process_pair_func(filter_func)
    module = f"implicit_tree_{filter_func.__name__}_{filter_data_type.__name__}"

    if passive_shape is not None:
        module += f"_{passive_shape.__name__}"
    if reuse:
        module += "_reuse"

    @wp.kernel(enable_backward=False, module=module)
    def leaves(
        shape_index: wp.array[int],
        sorted_values: wp.array[int],
        lower: wp.array[wp.vec3],
        upper: wp.array[wp.vec3],
        gap: wp.array[float],
        displacement: wp.array[wp.vec3],
        flags: wp.array[wp.int32],
        nodes: wp.array[ImplicitBvhNode],
        leaf_item: wp.array[wp.int32],
        leaf_base: int,
        shape_count: int,
        filter_data: Any,
        active_slots: wp.array[wp.int32],
        active_count: wp.array[wp.int32],
        reuse_slack: float,
        rebuild: wp.array[wp.int32],
    ):
        slot = wp.tid()
        node = leaf_base + slot
        if wp.static(reuse):
            if slot == 0:
                rebuild[0] = 0
                rebuild[1] += 1
        record = ImplicitBvhNode()
        record.lower = wp.vec3(1.0e30, 1.0e30, 1.0e30)
        record.upper = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
        record.max_span = wp.float32(-1.0)
        record.last_leaf = -1
        leaf_item[slot] = -1
        if slot >= shape_count:
            nodes[node] = record
            return
        shape = shape_index[sorted_values[slot]]
        leaf_item[slot] = shape
        record.last_leaf = slot
        if flags.shape[0] > 0 and (flags[shape] & wp.static(int(ShapeFlags.COLLIDE_SHAPES))) == 0:
            # A row for a runtime shape that holds none: a leaf that overlaps nothing.
            nodes[node] = record
            return
        low, high = _leaf_bounds(shape, lower, upper, gap, displacement, reuse_slack)
        record.lower = low
        record.upper = high
        span = record.upper - record.lower
        record.max_span = wp.max(span[0], wp.max(span[1], span[2]))
        active = True
        if wp.static(passive_shape is not None):
            if passive_shape(shape, filter_data):
                # Active leaves own pairs with passive leaves regardless of their AABB sizes.
                record.max_span = wp.inf
                active = False
            else:
                # Finite endpoints can still overflow when subtracting them. Reserve infinity for passive leaves.
                record.max_span = wp.min(record.max_span, wp.static(float(np.finfo(np.float32).max)))
        nodes[node] = record
        if active:
            # Only active leaves query; listing them keeps every query lane busy.
            active_slots[wp.atomic_add(active_count, 0, 1)] = slot

    @wp.kernel(enable_backward=False, module=module)
    def query(
        tree: ImplicitBvhData,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float],
        shape_displacement: wp.array[wp.vec3],
        collision_group: wp.array[int],
        shape_world: wp.array[int],
        filter_pairs: Any,
        num_filter_pairs: int,
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        filter_data: Any,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: int,
        active_slots: wp.array[wp.int32],
        active_count: wp.array[wp.int32],
        depth: int,
        thread_count: int,
    ):
        # An ancestor always contains the query leaf, so its bound cannot
        # reject it. Visit only the sibling subtrees along the leaf-to-root
        # path. These disjoint subtrees contain every other leaf exactly once,
        # so each (leaf, level) work item walks one of them in its own lane:
        # a lane walks one short chain instead of the whole path, and lanes of
        # a warp share a level, whose sibling subtrees neighboring leaves share.
        count = active_count[0]
        for work in range(wp.tid(), count * depth, thread_count):
            level = work // count
            leaf = active_slots[work - level * count]
            query_node = tree.leaf_base + leaf
            query_lower, query_upper, query_span, _query_last = _node_load(tree, query_node)
            shape1 = tree.leaf_item[leaf]
            node = (query_node >> level) ^ wp.int32(1)
            stop = implicit_bvh_next_subtree(node)
            while node != stop:
                lower, upper, max_span, last_leaf = _node_load(tree, node)
                # The leaf with the smaller span (ties: smaller slot) reports a pair.
                if (
                    max_span < query_span
                    or (max_span == query_span and last_leaf <= leaf)
                    or aabb_disjoint(lower, upper, query_lower, query_upper)
                ):
                    node = implicit_bvh_next_subtree(node)
                elif node < tree.leaf_base:
                    node = node * wp.int32(2)
                elif wp.static(reuse):
                    # Keep every pair of overlapping reuse bounds; ``compact`` filters them on each pass.
                    shape2 = tree.leaf_item[node - tree.leaf_base]
                    if not is_shape_pair_same_body_filtered(shape1, shape2, shape_body):
                        write_pair(
                            wp.vec2i(wp.min(shape1, shape2), wp.max(shape1, shape2)),
                            candidate_pair,
                            candidate_pair_count,
                            max_candidate_pair,
                        )
                    node = implicit_bvh_next_subtree(node)
                else:
                    shape2 = tree.leaf_item[node - tree.leaf_base]
                    pair_world_id = wp.int32(0)
                    if shape_world[shape1] == wp.int32(-1) and shape_world[shape2] == wp.int32(-1):
                        pair_world_id = wp.int32(1)
                    process_pair(
                        shape1,
                        shape2,
                        pair_world_id,
                        wp.int32(1),
                        shape_lower,
                        shape_upper,
                        shape_gap,
                        shape_displacement,
                        collision_group,
                        shape_world,
                        filter_pairs,
                        num_filter_pairs,
                        shape_body,
                        body_flags,
                        include_static_kinematic_pairs,
                        filter_data,
                        candidate_pair,
                        candidate_pair_count,
                        max_candidate_pair,
                    )
                    node = implicit_bvh_next_subtree(node)

    @wp.kernel(enable_backward=False, module=module)
    def check(
        tree: ImplicitBvhData,
        lower: wp.array[wp.vec3],
        upper: wp.array[wp.vec3],
        gap: wp.array[float],
        displacement: wp.array[wp.vec3],
        flags: wp.array[wp.int32],
        shape_count: int,
        filter_data: Any,
        kept_pair_count: wp.array[int],
        kept_pair_capacity: int,
        rebuild: wp.array[wp.int32],
    ):
        """Request a rebuild once a leaf leaves its reuse bound, starts colliding, or becomes active."""
        slot = wp.tid()
        if slot >= shape_count or rebuild[0] != 0:
            return
        if slot == 0 and kept_pair_count[0] > kept_pair_capacity:
            rebuild[0] = 1
            return
        shape = tree.leaf_item[slot]
        if flags.shape[0] > 0 and (flags[shape] & wp.static(int(ShapeFlags.COLLIDE_SHAPES))) == 0:
            return
        kept_lower, kept_upper, kept_span, _last = _node_load(tree, tree.leaf_base + slot)
        escaped = kept_span < 0.0
        if wp.static(passive_shape is not None):
            if kept_span == wp.inf and not passive_shape(shape, filter_data):
                escaped = True
        if not escaped:
            low, high = _leaf_bounds(shape, lower, upper, gap, displacement, 0.0)
            escaped = (
                low[0] < kept_lower[0]
                or low[1] < kept_lower[1]
                or low[2] < kept_lower[2]
                or high[0] > kept_upper[0]
                or high[1] > kept_upper[1]
                or high[2] > kept_upper[2]
            )
        if escaped:
            rebuild[0] = 1

    @wp.kernel(enable_backward=False, module=module)
    def compact(
        kept_pair: wp.array[wp.vec2i],
        kept_pair_count: wp.array[int],
        flags: wp.array[wp.int32],
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float],
        shape_displacement: wp.array[wp.vec3],
        collision_group: wp.array[int],
        shape_world: wp.array[int],
        filter_pairs: Any,
        num_filter_pairs: int,
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        filter_data: Any,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: int,
        thread_count: int,
    ):
        """Apply the exact pair test of ``query`` to the kept pairs."""
        kept = kept_pair_count[0]
        count = wp.min(kept, kept_pair.shape[0])
        if wp.tid() == 0 and kept > count:
            # Truncated kept pairs may miss candidates: report the overflow as the full query would.
            wp.atomic_add(candidate_pair_count, 0, max_candidate_pair + 1)
        for i in range(wp.tid(), count, thread_count):
            pair = kept_pair[i]
            shape1 = pair[0]
            shape2 = pair[1]
            if flags.shape[0] > 0:
                collide = wp.static(int(ShapeFlags.COLLIDE_SHAPES))
                if (flags[shape1] & collide) == 0 or (flags[shape2] & collide) == 0:
                    continue
            pair_world_id = wp.int32(0)
            if shape_world[shape1] == wp.int32(-1) and shape_world[shape2] == wp.int32(-1):
                pair_world_id = wp.int32(1)
            process_pair(
                shape1,
                shape2,
                pair_world_id,
                wp.int32(1),
                shape_lower,
                shape_upper,
                shape_gap,
                shape_displacement,
                collision_group,
                shape_world,
                filter_pairs,
                num_filter_pairs,
                shape_body,
                body_flags,
                include_static_kinematic_pairs,
                filter_data,
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            )

    return leaves, query, check, compact


_DEFAULT_KERNELS = _make_pair_kernels(keep_all_filter, EmptyFilterData)


class BroadPhaseImplicitTree:
    """Experimental graph-capturable implicit BVH with exact pair traversal."""

    def __init__(
        self,
        shape_world: wp.array[wp.int32] | np.ndarray,
        shape_flags: wp.array[wp.int32] | np.ndarray | None = None,
        *,
        queue_capacity_factor: int = 64,
        device: wp.Device | str | None = None,
        filter_func: Any | None = None,
        filter_data_type: Any | None = None,
        dynamic_rows: np.ndarray | None = None,
        passive_shape: Any | None = None,
        pair_reuse_slack: float | None = None,
    ) -> None:
        """
        Args:
            pair_reuse_slack: Enables pair reuse when not ``None``. Each tree build
                then bounds every shape's sweep over this pass and the next one, widened
                by this slack [m], and keeps the pairs of overlapping bounds. Later launches
                apply the exact pair test to the kept pairs, and rebuild only once a shape
                leaves its bound, starts colliding, or becomes active. The candidate pairs
                equal those of a rebuild on every launch, in a different order.
            passive_shape: Optional Warp predicate taking a shape index and the pair filter's data.
                Passive shapes are visible to active queries. The pair filter must reject all
                pairs of passive shapes.
            dynamic_rows: Mask of the shapes whose flags change at runtime. They always have a
                leaf, which overlaps nothing while ``shape_flags``, then a device array that
                is read on every launch, does not let the shape collide.
        """
        if queue_capacity_factor < 1:
            raise ValueError("queue_capacity_factor must be positive")
        if (filter_func is None) != (filter_data_type is None):
            raise ValueError("filter_func and filter_data_type must be provided together")
        if isinstance(shape_world, wp.array):
            world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            world_np = shape_world
        flags_np = shape_flags.numpy() if isinstance(shape_flags, wp.array) else shape_flags
        index_map, _ = precompute_world_map(world_np, flags_np)
        valid = np.unique(index_map).astype(np.int32)
        self._flags = wp.empty(0, dtype=wp.int32, device=device)
        if dynamic_rows is not None:
            if not isinstance(shape_flags, wp.array):
                raise ValueError("dynamic_rows requires shape_flags as a device array")
            valid = np.union1d(valid, np.flatnonzero(dynamic_rows)).astype(np.int32)
            self._flags = shape_flags
        self.shape_index = wp.array(valid, dtype=wp.int32, device=device)
        self.shape_count = len(valid)
        # Retain the constructor argument for API compatibility. Stackless
        # traversal allocates no per-level pair queues or overflow fallback.
        self.capacity = max(1, queue_capacity_factor * self.shape_count)
        # The leaf kernel below places sorted shapes by slot, which needs all
        # leaves at one depth.
        self.tree = ImplicitBvh(max(1, self.shape_count), device=device, slack=1.0, power_of_two=True)
        self.tree.item_count = self.shape_count
        self.tree.data.item_count = self.shape_count
        # 48-bit Morton keys on a fixed 2 cm grid; the tree's sort order buffer
        # holds the sorted leaf -> shape_index permutation.
        self.keys = wp.empty(2 * self.shape_count, dtype=wp.uint64, device=device)
        # Own sort values: the tree's build scratch aliases its node buffer,
        # which the leaf kernel below writes while reading this order.
        self.values = wp.empty(2 * self.shape_count, dtype=wp.int32, device=device)
        self.leaf_base = self.tree.leaf_base
        # Query work items are (active leaf, ancestor level) pairs.
        self._depth = self.leaf_base.bit_length() - 1
        self._query_threads = max(1, min(self.shape_count * self._depth, _QUERY_MAX_THREADS))
        self._active_slots = wp.empty(self.shape_count, dtype=wp.int32, device=device)
        self._active_count = wp.zeros(1, dtype=wp.int32, device=device)
        self._empty_filter_data = EmptyFilterData()
        if pair_reuse_slack is not None and not (np.isfinite(pair_reuse_slack) and pair_reuse_slack > 0.0):
            raise ValueError("pair_reuse_slack must be a positive finite number or None")
        self._pair_reuse_slack = pair_reuse_slack
        # [0]: rebuild before the next query, [1]: builds so far.
        self._rebuild = wp.array([1, 0], dtype=wp.int32, device=device)
        self._kept_pairs = None
        self._kept_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
        self._set_filter_kernels(filter_func, filter_data_type, passive_shape=passive_shape)

    def _set_filter_kernels(
        self, filter_func: Any | None, filter_data_type: Any | None, *, passive_shape: Any | None = None
    ) -> None:
        if passive_shape is not None and filter_func is None:
            raise ValueError("passive_shape requires a pair filter")
        self._has_custom_filter = filter_func is not None
        reuse = self._pair_reuse_slack is not None
        self._kernels = (
            _make_pair_kernels(filter_func, filter_data_type, passive_shape, reuse)
            if self._has_custom_filter
            else (_make_pair_kernels(keep_all_filter, EmptyFilterData, reuse=True) if reuse else _DEFAULT_KERNELS)
        )

    def update_world_map(
        self, shape_world: wp.array[wp.int32], shape_flags: wp.array[wp.int32], dirty: wp.array[wp.int32]
    ) -> None:
        """Do nothing: every launch reads the worlds and flags of runtime shapes anew."""

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float] | None,
        shape_collision_group: wp.array[int],
        shape_world: wp.array[int],
        shape_count: int,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        device: wp.Device | str | None = None,
        filter_pairs: wp.array[wp.vec2i] | PairExclusionIndex | None = None,
        num_filter_pairs: int | None = None,
        skip_count_zero: bool = False,
        filter_data: Any | None = None,
        *,
        shape_body: wp.array[int] | None = None,
        body_flags: wp.array[int] | None = None,
        include_static_kinematic_pairs: bool = True,
        shape_displacement: wp.array[wp.vec3] | None = None,
        sort_axis_displacement_limit: float | None = None,
    ) -> None:
        if device is None:
            device = shape_lower.device
        if not skip_count_zero:
            candidate_pair_count.zero_()
        if shape_count < 2 or self.shape_count < 2:
            return
        if self._has_custom_filter and filter_data is None:
            raise ValueError("BroadPhaseImplicitTree requires filter_data for its custom filter")
        shape_gap = shape_gap if shape_gap is not None else wp.empty(0, dtype=wp.float32, device=device)
        shape_displacement = (
            shape_displacement if shape_displacement is not None else wp.empty(0, dtype=wp.vec3, device=device)
        )
        shape_body = shape_body if shape_body is not None else wp.empty(0, dtype=wp.int32, device=device)
        body_flags = body_flags if body_flags is not None else wp.empty(0, dtype=wp.int32, device=device)
        filter_pairs = filter_pairs if filter_pairs is not None else wp.empty(0, dtype=wp.vec2i, device=device)
        n_filter = num_filter_pairs
        if n_filter is None:
            if isinstance(filter_pairs, wp.array):
                n_filter = filter_pairs.shape[0]
            else:
                n_filter = filter_pairs.partners.shape[0]
        pair_data = filter_data if self._has_custom_filter else self._empty_filter_data
        reuse = self._pair_reuse_slack is not None
        pair_inputs = [
            shape_lower,
            shape_upper,
            shape_gap,
            shape_displacement,
            shape_collision_group,
            shape_world,
            filter_pairs,
            n_filter,
            shape_body,
            body_flags,
            include_static_kinematic_pairs,
            pair_data,
        ]
        if reuse:
            if self._kept_pairs is None or self._kept_pairs.shape[0] != candidate_pair.shape[0]:
                self._kept_pairs = wp.empty(candidate_pair.shape[0], dtype=wp.vec2i, device=device)
            wp.launch(
                self._kernels[2],
                dim=self.shape_count,
                inputs=[
                    self.tree.data,
                    shape_lower,
                    shape_upper,
                    shape_gap,
                    shape_displacement,
                    self._flags,
                    self.shape_count,
                    pair_data,
                    self._kept_pair_count,
                    self._kept_pairs.shape[0],
                    self._rebuild,
                ],
                device=device,
                record_tape=False,
            )
            wp.capture_if(
                self._rebuild,
                on_true=lambda: self._build_and_query(
                    shape_lower,
                    shape_upper,
                    shape_gap,
                    shape_displacement,
                    pair_data,
                    pair_inputs,
                    self._kept_pairs,
                    self._kept_pair_count,
                    device,
                ),
            )
            wp.launch(
                self._kernels[3],
                dim=_COMPACT_THREADS,
                inputs=[
                    self._kept_pairs,
                    self._kept_pair_count,
                    self._flags,
                    *pair_inputs,
                    candidate_pair,
                    candidate_pair_count,
                    candidate_pair.shape[0],
                    _COMPACT_THREADS,
                ],
                device=device,
                record_tape=False,
            )
        else:
            self._build_and_query(
                shape_lower,
                shape_upper,
                shape_gap,
                shape_displacement,
                pair_data,
                pair_inputs,
                candidate_pair,
                candidate_pair_count,
                device,
            )

    def _build_and_query(
        self,
        shape_lower,
        shape_upper,
        shape_gap,
        shape_displacement,
        pair_data,
        pair_inputs,
        candidate_pair,
        candidate_pair_count,
        device,
    ) -> None:
        reuse = self._pair_reuse_slack is not None
        if reuse:
            candidate_pair_count.zero_()
        wp.launch(
            _tree_keys,
            dim=self.shape_count,
            inputs=[
                self.shape_index,
                shape_lower,
                shape_upper,
                shape_gap,
                shape_displacement,
                self.keys,
                self.values,
                self._active_count,
            ],
            device=device,
            record_tape=False,
        )
        wp.utils.radix_sort_pairs(self.keys, self.values, self.shape_count, end_bit=48)
        wp.launch(
            self._kernels[0],
            dim=self.leaf_base,
            inputs=[
                self.shape_index,
                self.values,
                shape_lower,
                shape_upper,
                shape_gap,
                shape_displacement,
                self._flags,
                self.tree.nodes,
                self.tree.leaf_item,
                self.leaf_base,
                self.shape_count,
                pair_data,
                self._active_slots,
                self._active_count,
                self._pair_reuse_slack if reuse else 0.0,
                self._rebuild,
            ],
            device=device,
            record_tape=False,
        )
        self.tree.update_internal_nodes()
        wp.launch(
            self._kernels[1],
            dim=self._query_threads,
            inputs=[
                self.tree.data,
                *pair_inputs,
                candidate_pair,
                candidate_pair_count,
                candidate_pair.shape[0],
                self._active_slots,
                self._active_count,
                self._depth,
                self._query_threads,
            ],
            device=device,
            record_tape=False,
        )
