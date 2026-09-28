# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Uniform-grid broad phase with unique ownership of overlapping shape pairs."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .broad_phase_common import EmptyFilterData, keep_all_filter, precompute_world_map
from .broad_phase_sap import _make_sap_process_pair_func

wp.set_module_options({"enable_backward": False})

_CELL_BIAS = wp.constant(wp.int32(1 << 20))
_MAX_CELLS_PER_SHAPE = wp.constant(wp.int32(8))
_EMPTY_CELL_KEY = wp.constant(wp.uint64((1 << 64) - 1))


@wp.func
def _cell_key(x: wp.int32, y: wp.int32, z: wp.int32) -> wp.uint64:
    return (
        (wp.uint64(x + _CELL_BIAS) << wp.uint64(42))
        | (wp.uint64(y + _CELL_BIAS) << wp.uint64(21))
        | wp.uint64(z + _CELL_BIAS)
    )


@wp.kernel(enable_backward=False)
def _grid_prepare_bounds(
    shape_index: wp.array[int],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],
    shape_displacement: wp.array[wp.vec3],
    inverse_cell_width: float,
    cell_lower: wp.array[wp.vec3i],
    cell_upper: wp.array[wp.vec3i],
    cell_count: wp.array[int],
    large_flag: wp.array[int],
):
    leaf = wp.tid()
    shape = shape_index[leaf]
    lower = shape_lower[shape]
    upper = shape_upper[shape]
    gap = wp.float32(0.0)
    if shape_gap.shape[0] > 0:
        gap = shape_gap[shape]
    displacement = wp.vec3(0.0, 0.0, 0.0)
    if shape_displacement.shape[0] > 0:
        displacement = shape_displacement[shape]
    expansion = wp.vec3(gap, gap, gap)
    swept_lower = wp.min(lower, lower + displacement) - expansion
    swept_upper = wp.max(upper, upper + displacement) + expansion
    scaled_lower = swept_lower * inverse_cell_width
    scaled_upper = swept_upper * inverse_cell_width
    if (
        scaled_lower[0] <= -wp.float32(_CELL_BIAS)
        or scaled_lower[1] <= -wp.float32(_CELL_BIAS)
        or scaled_lower[2] <= -wp.float32(_CELL_BIAS)
        or scaled_upper[0] >= wp.float32(_CELL_BIAS)
        or scaled_upper[1] >= wp.float32(_CELL_BIAS)
        or scaled_upper[2] >= wp.float32(_CELL_BIAS)
    ):
        cell_count[leaf] = wp.int32(0)
        large_flag[leaf] = wp.int32(1)
        return

    lo = wp.vec3i(
        wp.int32(wp.floor(scaled_lower[0])),
        wp.int32(wp.floor(scaled_lower[1])),
        wp.int32(wp.floor(scaled_lower[2])),
    )
    hi = wp.vec3i(
        wp.int32(wp.floor(scaled_upper[0])),
        wp.int32(wp.floor(scaled_upper[1])),
        wp.int32(wp.floor(scaled_upper[2])),
    )
    cell_lower[leaf] = lo
    cell_upper[leaf] = hi
    nx = hi[0] - lo[0] + wp.int32(1)
    ny = hi[1] - lo[1] + wp.int32(1)
    nz = hi[2] - lo[2] + wp.int32(1)
    if nx > _MAX_CELLS_PER_SHAPE or ny > _MAX_CELLS_PER_SHAPE or nz > _MAX_CELLS_PER_SHAPE:
        cell_count[leaf] = wp.int32(0)
        large_flag[leaf] = wp.int32(1)
        return
    count = nx * ny * nz
    if count > _MAX_CELLS_PER_SHAPE:
        cell_count[leaf] = wp.int32(0)
        large_flag[leaf] = wp.int32(1)
    else:
        cell_count[leaf] = count
        large_flag[leaf] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _grid_clear_entries(keys: wp.array[wp.uint64], values: wp.array[int]):
    tid = wp.tid()
    keys[tid] = _EMPTY_CELL_KEY
    values[tid] = wp.int32(-1)


@wp.kernel(enable_backward=False)
def _grid_scatter_entries(
    cell_lower: wp.array[wp.vec3i],
    cell_upper: wp.array[wp.vec3i],
    cell_count: wp.array[int],
    cell_prefix: wp.array[int],
    capacity: int,
    keys: wp.array[wp.uint64],
    values: wp.array[int],
    large_flag: wp.array[int],
):
    leaf = wp.tid()
    count = cell_count[leaf]
    if count <= wp.int32(0):
        return
    start = cell_prefix[leaf] - count
    if start + count > capacity:
        large_flag[leaf] = wp.int32(1)
        return
    lo = cell_lower[leaf]
    hi = cell_upper[leaf]
    offset = wp.int32(0)
    for x in range(lo[0], hi[0] + wp.int32(1)):
        for y in range(lo[1], hi[1] + wp.int32(1)):
            for z in range(lo[2], hi[2] + wp.int32(1)):
                keys[start + offset] = _cell_key(x, y, z)
                values[start + offset] = leaf
                offset += wp.int32(1)


@wp.kernel(enable_backward=False)
def _grid_scatter_large(
    large_flag: wp.array[int],
    large_prefix: wp.array[int],
    large_index: wp.array[int],
):
    leaf = wp.tid()
    if large_flag[leaf] != wp.int32(0):
        large_index[large_prefix[leaf] - wp.int32(1)] = leaf


def _make_grid_pair_kernels(filter_func: Any, filter_data_type: Any):
    _module = f"grid_broadphase_{filter_func.__name__}_{filter_data_type.__name__}"
    process_pair = _make_sap_process_pair_func(filter_func)

    @wp.func
    def process_leaf_pair(
        leaf1: wp.int32,
        leaf2: wp.int32,
        shape_index: wp.array[int],
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float],
        shape_displacement: wp.array[wp.vec3],
        collision_group: wp.array[int],
        shape_world: wp.array[int],
        filter_pairs: wp.array[wp.vec2i],
        num_filter_pairs: wp.int32,
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: wp.bool,
        filter_data: Any,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: wp.int32,
    ):
        shape1 = shape_index[leaf1]
        shape2 = shape_index[leaf2]
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

    @wp.kernel(enable_backward=False, module=_module, grid_stride=False)
    def grid_pairs(
        keys: wp.array[wp.uint64],
        values: wp.array[int],
        cell_lower: wp.array[wp.vec3i],
        shape_index: wp.array[int],
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
        capacity: int,
    ):
        i = wp.tid()
        key = keys[i]
        if key == _EMPTY_CELL_KEY:
            return
        leaf1 = values[i]
        lo1 = cell_lower[leaf1]
        j = i + wp.int32(1)
        while j < capacity and keys[j] == key:
            leaf2 = values[j]
            lo2 = cell_lower[leaf2]
            owner_key = _cell_key(
                wp.max(lo1[0], lo2[0]),
                wp.max(lo1[1], lo2[1]),
                wp.max(lo1[2], lo2[2]),
            )
            if owner_key == key:
                process_leaf_pair(
                    leaf1,
                    leaf2,
                    shape_index,
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
            j += wp.int32(1)

    @wp.kernel(enable_backward=False, module=_module, grid_stride=False)
    def large_pairs(
        large_flag: wp.array[int],
        large_prefix: wp.array[int],
        large_index: wp.array[int],
        shape_index: wp.array[int],
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
        shape_count: int,
    ):
        leaf1 = wp.tid()
        large_count = large_prefix[shape_count - wp.int32(1)]
        for large_slot in range(large_count):
            leaf2 = large_index[large_slot]
            if leaf1 == leaf2:
                continue
            # Small-large pairs are emitted by the small leaf. Large-large
            # pairs are emitted by the smaller leaf exactly once.
            if large_flag[leaf1] != wp.int32(0) and leaf1 > leaf2:
                continue
            process_leaf_pair(
                leaf1,
                leaf2,
                shape_index,
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

    return grid_pairs, large_pairs


_grid_pair_kernel, _large_pair_kernel = _make_grid_pair_kernels(keep_all_filter, EmptyFilterData)


class BroadPhaseGrid:
    """Dynamic 3D grid with a conservative large-shape side pass.

    Each pair belongs to the first cell shared by its swept bounds. Large
    shapes that span more than eight cells, and shapes exceeding the compact
    member buffer, are compared directly with every shape.
    """

    def __init__(
        self,
        shape_world: wp.array[wp.int32] | np.ndarray,
        shape_flags: wp.array[wp.int32] | np.ndarray | None = None,
        *,
        cell_width: float = 0.3,
        capacity_factor: int = 4,
        device: wp.Device | str | None = None,
        filter_func: Any | None = None,
        filter_data_type: Any | None = None,
    ) -> None:
        if cell_width <= 0.0:
            raise ValueError("cell_width must be positive")
        if capacity_factor < 1:
            raise ValueError("capacity_factor must be positive")
        if (filter_func is None) != (filter_data_type is None):
            raise ValueError("filter_func and filter_data_type must be provided together")
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
        shape_flags_np = shape_flags.numpy() if isinstance(shape_flags, wp.array) else shape_flags
        index_map, _ = precompute_world_map(shape_world_np, shape_flags_np)
        valid_indices = np.unique(index_map).astype(np.int32)
        self.shape_index = wp.array(valid_indices, dtype=wp.int32, device=device)
        self.shape_count = len(valid_indices)
        self.cell_width = float(cell_width)
        self.capacity = max(1, capacity_factor * self.shape_count)
        self.cell_lower = wp.empty(self.shape_count, dtype=wp.vec3i, device=device)
        self.cell_upper = wp.empty(self.shape_count, dtype=wp.vec3i, device=device)
        self.cell_count = wp.empty(self.shape_count, dtype=wp.int32, device=device)
        self.cell_prefix = wp.empty(self.shape_count, dtype=wp.int32, device=device)
        self.large_flag = wp.empty(self.shape_count, dtype=wp.int32, device=device)
        self.large_prefix = wp.empty(self.shape_count, dtype=wp.int32, device=device)
        self.large_index = wp.empty(self.shape_count, dtype=wp.int32, device=device)
        self.keys = wp.empty(2 * self.capacity, dtype=wp.uint64, device=device)
        self.values = wp.empty(2 * self.capacity, dtype=wp.int32, device=device)
        self._has_custom_filter = filter_func is not None
        self._empty_filter_data = EmptyFilterData()
        if self._has_custom_filter:
            self._grid_kernel, self._large_kernel = _make_grid_pair_kernels(filter_func, filter_data_type)
        else:
            self._grid_kernel, self._large_kernel = _grid_pair_kernel, _large_pair_kernel

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
        """Emit exact overlapping pairs from conservative swept-grid candidates.

        The SAP-compatible ``sort_axis_displacement_limit`` does not apply to a
        3D grid. Full displacement is used when assigning cells, then the
        shared pair filter checks continuous overlap exactly.
        """
        if device is None:
            device = shape_lower.device
        if not skip_count_zero:
            candidate_pair_count.zero_()
        if shape_count < 2 or self.shape_count < 2:
            return
        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)
        if shape_displacement is None:
            shape_displacement = wp.empty(0, dtype=wp.vec3, device=device)
        if shape_body is None:
            shape_body = wp.empty(0, dtype=wp.int32, device=device)
        if body_flags is None:
            body_flags = wp.empty(0, dtype=wp.int32, device=device)
        if filter_pairs is None:
            filter_pairs = wp.empty(0, dtype=wp.vec2i, device=device)
        n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]
        if self._has_custom_filter and filter_data is None:
            raise ValueError("BroadPhaseGrid was constructed with filter_func=...; launch() requires filter_data")
        pair_filter_data = filter_data if self._has_custom_filter else self._empty_filter_data

        wp.launch(
            _grid_prepare_bounds,
            dim=self.shape_count,
            inputs=[
                self.shape_index,
                shape_lower,
                shape_upper,
                shape_gap,
                shape_displacement,
                1.0 / self.cell_width,
                self.cell_lower,
                self.cell_upper,
                self.cell_count,
                self.large_flag,
            ],
            device=device,
            record_tape=False,
        )
        wp.utils.array_scan(self.cell_count, self.cell_prefix, True)
        wp.launch(
            _grid_clear_entries,
            dim=self.capacity,
            inputs=[self.keys, self.values],
            device=device,
            record_tape=False,
        )
        wp.launch(
            _grid_scatter_entries,
            dim=self.shape_count,
            inputs=[
                self.cell_lower,
                self.cell_upper,
                self.cell_count,
                self.cell_prefix,
                self.capacity,
                self.keys,
                self.values,
                self.large_flag,
            ],
            device=device,
            record_tape=False,
        )
        wp.utils.array_scan(self.large_flag, self.large_prefix, True)
        wp.launch(
            _grid_scatter_large,
            dim=self.shape_count,
            inputs=[self.large_flag, self.large_prefix, self.large_index],
            device=device,
            record_tape=False,
        )
        wp.utils.radix_sort_pairs(self.keys, self.values, self.capacity)
        common_inputs = [
            self.shape_index,
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
            pair_filter_data,
            candidate_pair,
            candidate_pair_count,
            candidate_pair.shape[0],
        ]
        wp.launch(
            self._grid_kernel,
            dim=self.capacity,
            inputs=[self.keys, self.values, self.cell_lower, *common_inputs, self.capacity],
            device=device,
            record_tape=False,
        )
        wp.launch(
            self._large_kernel,
            dim=self.shape_count,
            inputs=[self.large_flag, self.large_prefix, self.large_index, *common_inputs, self.shape_count],
            device=device,
            record_tape=False,
        )
