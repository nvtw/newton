# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit binary BVH with stackless per-leaf traversal.

This module is intentionally separate from the production broad phases. Its
complete binary topology isolates the cost of BVH pair-tree traversal; it is
not an implementation of the compact Chitalu et al. node-index mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .broad_phase_common import EmptyFilterData, keep_all_filter, precompute_world_map
from .broad_phase_sap import _make_sap_process_pair_func

wp.set_module_options({"enable_backward": False})


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
):
    leaf = wp.tid()
    shape = shape_index[leaf]
    center = (lower[shape] + upper[shape]) * 0.5
    if displacement.shape[0] > 0:
        center += displacement[shape] * 0.5
    # Quantization affects traversal locality only; node bounds remain exact.
    x = wp.uint64(wp.int32(wp.clamp(wp.floor(center[0] * 100.0) + 524288.0, 0.0, 1048575.0)))
    y = wp.uint64(wp.int32(wp.clamp(wp.floor(center[1] * 100.0) + 524288.0, 0.0, 1048575.0)))
    z = wp.uint64(wp.int32(wp.clamp(wp.floor(center[2] * 100.0) + 524288.0, 0.0, 1048575.0)))
    keys[leaf] = (_spread_20(x) << wp.uint64(2)) | (_spread_20(y) << wp.uint64(1)) | _spread_20(z)
    values[leaf] = leaf


@wp.kernel(enable_backward=False)
def _tree_leaves(
    shape_index: wp.array[int],
    sorted_values: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    gap: wp.array[float],
    displacement: wp.array[wp.vec3],
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    node_shape: wp.array[int],
    node_last_leaf: wp.array[int],
    node_max_span: wp.array[float],
    leaf_base: int,
    shape_count: int,
):
    slot = wp.tid()
    node = leaf_base + slot
    node_shape[node] = -1
    node_last_leaf[node] = -1
    node_max_span[node] = wp.float32(-1.0)
    node_lower[node] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    node_upper[node] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    if slot >= shape_count:
        return
    shape = shape_index[sorted_values[slot]]
    node_shape[node] = shape
    node_last_leaf[node] = slot
    margin = wp.float32(0.0)
    if gap.shape[0] > 0:
        margin = gap[shape]
    shift = wp.vec3(0.0, 0.0, 0.0)
    if displacement.shape[0] > 0:
        shift = displacement[shape]
    expand = wp.vec3(margin, margin, margin)
    node_lower[node] = wp.min(lower[shape], lower[shape] + shift) - expand
    node_upper[node] = wp.max(upper[shape], upper[shape] + shift) + expand
    # At distant world coordinates, float32 rounding can make a swept bound
    # narrower than the equivalent moving-AABB test (which combines the two
    # gaps before subtraction). Expand by a few ulps of the coordinate scale;
    # the shared exact pair filter still decides which contacts are accepted.
    low = node_lower[node]
    high = node_upper[node]
    rounding = wp.vec3(
        wp.max(wp.abs(low[0]), wp.abs(high[0])) * wp.float32(4.76837158e-7) + wp.float32(1.0e-6),
        wp.max(wp.abs(low[1]), wp.abs(high[1])) * wp.float32(4.76837158e-7) + wp.float32(1.0e-6),
        wp.max(wp.abs(low[2]), wp.abs(high[2])) * wp.float32(4.76837158e-7) + wp.float32(1.0e-6),
    )
    node_lower[node] = low - rounding
    node_upper[node] = high + rounding
    span = node_upper[node] - node_lower[node]
    node_max_span[node] = wp.max(span[0], wp.max(span[1], span[2]))


@wp.kernel(enable_backward=False)
def _tree_build_level(
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    node_last_leaf: wp.array[int],
    node_max_span: wp.array[float],
    start: int,
):
    node = start + wp.tid()
    node_lower[node] = wp.min(node_lower[node * 2], node_lower[node * 2 + 1])
    node_upper[node] = wp.max(node_upper[node * 2], node_upper[node * 2 + 1])
    node_last_leaf[node] = wp.max(node_last_leaf[node * 2], node_last_leaf[node * 2 + 1])
    node_max_span[node] = wp.max(node_max_span[node * 2], node_max_span[node * 2 + 1])


@wp.func
def _tree_next_dfs(node: wp.int32) -> wp.int32:
    """Advance past a visited/pruned subtree in the complete binary tree."""
    while node > wp.int32(1):
        if (node & wp.int32(1)) == wp.int32(0):
            return node + wp.int32(1)
        node = node / wp.int32(2)
    return wp.int32(0)


def _make_pair_kernels(filter_func: Any, filter_data_type: Any):
    process_pair = _make_sap_process_pair_func(filter_func)
    module = f"implicit_tree_{filter_func.__name__}_{filter_data_type.__name__}"

    @wp.kernel(enable_backward=False, module=module)
    def query(
        node_lower: wp.array[wp.vec3],
        node_upper: wp.array[wp.vec3],
        node_shape: wp.array[int],
        node_last_leaf: wp.array[int],
        node_max_span: wp.array[float],
        leaf_base: int,
        shape_count: int,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float],
        shape_displacement: wp.array[wp.vec3],
        collision_group: wp.array[int],
        shape_world: wp.array[int],
        filter_pairs: wp.array[wp.vec2i],
        num_filter_pairs: int,
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        filter_data: Any,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: int,
    ):
        leaf = wp.tid()
        if leaf >= shape_count:
            return
        query_node = leaf_base + leaf
        query_lower = node_lower[query_node]
        query_upper = node_upper[query_node]
        query_span = node_max_span[query_node]
        shape1 = node_shape[query_node]
        # An ancestor always contains the query leaf, so its bound cannot
        # reject it. Visit only the sibling subtrees along the leaf-to-root
        # path. These disjoint subtrees contain every other leaf exactly once.
        ancestor = query_node
        while ancestor > wp.int32(1):
            sibling = ancestor ^ wp.int32(1)
            stop = _tree_next_dfs(sibling)
            node = sibling
            while node != stop:
                max_span = node_max_span[node]
                if max_span < query_span or (max_span == query_span and node_last_leaf[node] <= leaf):
                    node = _tree_next_dfs(node)
                    continue
                if (
                    node_lower[node][0] > query_upper[0]
                    or query_lower[0] > node_upper[node][0]
                    or node_lower[node][1] > query_upper[1]
                    or query_lower[1] > node_upper[node][1]
                    or node_lower[node][2] > query_upper[2]
                    or query_lower[2] > node_upper[node][2]
                ):
                    node = _tree_next_dfs(node)
                    continue
                if node < leaf_base:
                    node = node * wp.int32(2)
                    continue
                other_leaf = node - leaf_base
                if other_leaf < shape_count and (
                    max_span > query_span or (max_span == query_span and other_leaf > leaf)
                ):
                    shape2 = node_shape[node]
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
                node = _tree_next_dfs(node)
            ancestor = ancestor / wp.int32(2)

    return query


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
    ) -> None:
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
        self.shape_index = wp.array(valid, dtype=wp.int32, device=device)
        self.shape_count = len(valid)
        self.leaf_base = 1 << max(1, (self.shape_count - 1).bit_length())
        self.depth = self.leaf_base.bit_length() - 1
        # Retain the constructor argument for API compatibility. Stackless
        # traversal allocates no per-level pair queues or overflow fallback.
        self.capacity = max(1, queue_capacity_factor * self.shape_count)
        self.keys = wp.empty(2 * self.shape_count, dtype=wp.uint64, device=device)
        self.values = wp.empty(2 * self.shape_count, dtype=wp.int32, device=device)
        self.node_lower = wp.empty(2 * self.leaf_base, dtype=wp.vec3, device=device)
        self.node_upper = wp.empty(2 * self.leaf_base, dtype=wp.vec3, device=device)
        self.node_shape = wp.empty(2 * self.leaf_base, dtype=wp.int32, device=device)
        self.node_last_leaf = wp.empty(2 * self.leaf_base, dtype=wp.int32, device=device)
        self.node_max_span = wp.empty(2 * self.leaf_base, dtype=wp.float32, device=device)
        self._has_custom_filter = filter_func is not None
        self._empty_filter_data = EmptyFilterData()
        self._kernels = (
            _make_pair_kernels(filter_func, filter_data_type) if self._has_custom_filter else _DEFAULT_KERNELS
        )

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
        filter_pairs: wp.array[wp.vec2i] | None = None,
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
        n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]
        pair_data = filter_data if self._has_custom_filter else self._empty_filter_data
        wp.launch(
            _tree_keys,
            dim=self.shape_count,
            inputs=[self.shape_index, shape_lower, shape_upper, shape_gap, shape_displacement, self.keys, self.values],
            device=device,
            record_tape=False,
        )
        wp.utils.radix_sort_pairs(self.keys, self.values, self.shape_count)
        wp.launch(
            _tree_leaves,
            dim=self.leaf_base,
            inputs=[
                self.shape_index,
                self.values,
                shape_lower,
                shape_upper,
                shape_gap,
                shape_displacement,
                self.node_lower,
                self.node_upper,
                self.node_shape,
                self.node_last_leaf,
                self.node_max_span,
                self.leaf_base,
                self.shape_count,
            ],
            device=device,
            record_tape=False,
        )
        for level in range(self.depth - 1, -1, -1):
            start = 1 << level
            wp.launch(
                _tree_build_level,
                dim=start,
                inputs=[self.node_lower, self.node_upper, self.node_last_leaf, self.node_max_span, start],
                device=device,
                record_tape=False,
            )
        query = self._kernels
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
            candidate_pair,
            candidate_pair_count,
            candidate_pair.shape[0],
        ]
        wp.launch(
            query,
            dim=self.shape_count,
            inputs=[
                self.node_lower,
                self.node_upper,
                self.node_shape,
                self.node_last_leaf,
                self.node_max_span,
                self.leaf_base,
                self.shape_count,
                *pair_inputs,
            ],
            device=device,
            record_tape=False,
        )
