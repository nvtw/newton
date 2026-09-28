# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit binary BVH with breadth-first self traversal.

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
    leaf_base: int,
    shape_count: int,
):
    slot = wp.tid()
    node = leaf_base + slot
    node_shape[node] = -1
    node_lower[node] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    node_upper[node] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    if slot >= shape_count:
        return
    shape = shape_index[sorted_values[slot]]
    node_shape[node] = shape
    margin = wp.float32(0.0)
    if gap.shape[0] > 0:
        margin = gap[shape]
    shift = wp.vec3(0.0, 0.0, 0.0)
    if displacement.shape[0] > 0:
        shift = displacement[shape]
    expand = wp.vec3(margin, margin, margin)
    node_lower[node] = wp.min(lower[shape], lower[shape] + shift) - expand
    node_upper[node] = wp.max(upper[shape], upper[shape] + shift) + expand


@wp.kernel(enable_backward=False)
def _tree_build_level(node_lower: wp.array[wp.vec3], node_upper: wp.array[wp.vec3], start: int):
    node = start + wp.tid()
    node_lower[node] = wp.min(node_lower[node * 2], node_lower[node * 2 + 1])
    node_upper[node] = wp.max(node_upper[node * 2], node_upper[node * 2 + 1])


@wp.kernel(enable_backward=False)
def _tree_root(queue: wp.array[wp.vec2i], count: wp.array[int]):
    queue[0] = wp.vec2i(1, 1)
    count[0] = 1


@wp.kernel(enable_backward=False)
def _tree_save_count(candidate_pair_count: wp.array[int], initial_count: wp.array[int]):
    initial_count[0] = candidate_pair_count[0]


@wp.func
def _tree_push(queue: wp.array[wp.vec2i], count: wp.array[int], overflow: wp.array[int], capacity: int, a: int, b: int):
    slot = wp.atomic_add(count, 0, 1)
    if slot < capacity:
        queue[slot] = wp.vec2i(a, b)
    else:
        overflow[0] = 1


def _make_pair_kernels(filter_func: Any, filter_data_type: Any):
    process_pair = _make_sap_process_pair_func(filter_func)
    module = f"implicit_tree_{filter_func.__name__}_{filter_data_type.__name__}"

    @wp.kernel(enable_backward=False, module=module)
    def expand(
        current: wp.array[wp.vec2i],
        current_count: wp.array[int],
        following: wp.array[wp.vec2i],
        following_count: wp.array[int],
        overflow: wp.array[int],
        node_lower: wp.array[wp.vec3],
        node_upper: wp.array[wp.vec3],
        capacity: int,
    ):
        slot = wp.tid()
        if slot >= current_count[0] or overflow[0] != 0:
            return
        pair = current[slot]
        a = pair[0]
        b = pair[1]
        if (
            node_lower[a][0] > node_upper[b][0]
            or node_lower[b][0] > node_upper[a][0]
            or node_lower[a][1] > node_upper[b][1]
            or node_lower[b][1] > node_upper[a][1]
            or node_lower[a][2] > node_upper[b][2]
            or node_lower[b][2] > node_upper[a][2]
        ):
            return
        if a == b:
            _tree_push(following, following_count, overflow, capacity, 2 * a, 2 * a)
            _tree_push(following, following_count, overflow, capacity, 2 * a, 2 * a + 1)
            _tree_push(following, following_count, overflow, capacity, 2 * a + 1, 2 * a + 1)
        else:
            _tree_push(following, following_count, overflow, capacity, 2 * a, 2 * b)
            _tree_push(following, following_count, overflow, capacity, 2 * a, 2 * b + 1)
            _tree_push(following, following_count, overflow, capacity, 2 * a + 1, 2 * b)
            _tree_push(following, following_count, overflow, capacity, 2 * a + 1, 2 * b + 1)

    @wp.kernel(enable_backward=False, module=module)
    def finish(
        current: wp.array[wp.vec2i],
        current_count: wp.array[int],
        overflow: wp.array[int],
        node_lower: wp.array[wp.vec3],
        node_upper: wp.array[wp.vec3],
        node_shape: wp.array[int],
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
        slot = wp.tid()
        if slot >= current_count[0] or overflow[0] != 0:
            return
        nodes = current[slot]
        a = nodes[0]
        b = nodes[1]
        if a == b or node_shape[a] < 0 or node_shape[b] < 0:
            return
        if (
            node_lower[a][0] > node_upper[b][0]
            or node_lower[b][0] > node_upper[a][0]
            or node_lower[a][1] > node_upper[b][1]
            or node_lower[b][1] > node_upper[a][1]
            or node_lower[a][2] > node_upper[b][2]
            or node_lower[b][2] > node_upper[a][2]
        ):
            return
        shape1 = node_shape[a]
        shape2 = node_shape[b]
        world_id = wp.int32(0)
        if shape_world[shape1] == -1 and shape_world[shape2] == -1:
            world_id = 1
        process_pair(
            shape1,
            shape2,
            world_id,
            1,
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

    @wp.kernel(enable_backward=False, module=module)
    def fallback_reset(overflow: wp.array[int], candidate_pair_count: wp.array[int], initial_count: wp.array[int]):
        if overflow[0] != 0:
            candidate_pair_count[0] = initial_count[0]

    @wp.kernel(enable_backward=False, module=module)
    def fallback_pairs(
        overflow: wp.array[int],
        shape_index: wp.array[int],
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
        i = wp.tid()
        if overflow[0] == 0:
            return
        shape1 = shape_index[i]
        for j in range(i + 1, shape_count):
            shape2 = shape_index[j]
            world_id = wp.int32(0)
            if shape_world[shape1] == -1 and shape_world[shape2] == -1:
                world_id = 1
            process_pair(
                shape1,
                shape2,
                world_id,
                1,
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

    return expand, finish, fallback_reset, fallback_pairs


_DEFAULT_KERNELS = _make_pair_kernels(keep_all_filter, EmptyFilterData)


class BroadPhaseImplicitTree:
    """Experimental graph-capturable implicit BVH for exact pair-set studies."""

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
        self.capacity = max(1, queue_capacity_factor * self.shape_count)
        self.keys = wp.empty(2 * self.shape_count, dtype=wp.uint64, device=device)
        self.values = wp.empty(2 * self.shape_count, dtype=wp.int32, device=device)
        self.node_lower = wp.empty(2 * self.leaf_base, dtype=wp.vec3, device=device)
        self.node_upper = wp.empty(2 * self.leaf_base, dtype=wp.vec3, device=device)
        self.node_shape = wp.empty(2 * self.leaf_base, dtype=wp.int32, device=device)
        self.queues = [wp.empty(self.capacity, dtype=wp.vec2i, device=device) for _ in range(2)]
        self.counts = [wp.zeros(1, dtype=wp.int32, device=device) for _ in range(2)]
        self.overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self.initial_count = wp.zeros(1, dtype=wp.int32, device=device)
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
        wp.launch(
            _tree_save_count,
            dim=1,
            inputs=[candidate_pair_count, self.initial_count],
            device=device,
            record_tape=False,
        )
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
                inputs=[self.node_lower, self.node_upper, start],
                device=device,
                record_tape=False,
            )
        self.overflow.zero_()
        wp.launch(_tree_root, dim=1, inputs=[self.queues[0], self.counts[0]], device=device, record_tape=False)
        expand, finish, fallback_reset, fallback_pairs = self._kernels
        for level in range(self.depth):
            src = level & 1
            dst = 1 - src
            self.counts[dst].zero_()
            launch_dim = min(self.capacity, 1 << min(30, 2 * (level + 1)))
            wp.launch(
                expand,
                dim=launch_dim,
                inputs=[
                    self.queues[src],
                    self.counts[src],
                    self.queues[dst],
                    self.counts[dst],
                    self.overflow,
                    self.node_lower,
                    self.node_upper,
                    self.capacity,
                ],
                device=device,
                record_tape=False,
            )
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
            finish,
            dim=self.capacity,
            inputs=[
                self.queues[self.depth & 1],
                self.counts[self.depth & 1],
                self.overflow,
                self.node_lower,
                self.node_upper,
                self.node_shape,
                *pair_inputs,
            ],
            device=device,
            record_tape=False,
        )
        wp.launch(
            fallback_reset,
            dim=1,
            inputs=[self.overflow, candidate_pair_count, self.initial_count],
            device=device,
            record_tape=False,
        )
        wp.launch(
            fallback_pairs,
            dim=self.shape_count,
            inputs=[self.overflow, self.shape_index, self.shape_count, *pair_inputs],
            device=device,
            record_tape=False,
        )
