# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-owned scalar symbolic patterns for batched Cholesky factors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from .llt_packed import _block_sync

__all__ = []

wp.set_module_options({"enable_backward": False})


@wp.func
def _row_start(base: wp.int64, row: wp.int32) -> wp.int64:
    r = wp.int64(row)
    return base + r * (r + wp.int64(1)) // wp.int64(2)


@wp.kernel
def _mark_pairs(
    pair_world: wp.array[wp.int32],
    pair_row: wp.array[wp.int32],
    pair_col: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    offsets: wp.array[wp.int64],
    pattern: wp.array[wp.int32],
):
    entry = wp.tid()
    world = pair_world[entry]
    a = inverse[vector_offsets[world] + pair_row[entry]]
    b = inverse[vector_offsets[world] + pair_col[entry]]
    index = _row_start(offsets[world], wp.max(a, b)) + wp.int64(wp.min(a, b))
    wp.atomic_or(pattern, index, wp.int32(1))


@wp.kernel
def _fill_and_compact(
    dimensions: wp.array[wp.int32],
    offsets: wp.array[wp.int64],
    row_offsets: wp.array[wp.int32],
    pattern: wp.array[wp.int32],
    counts: wp.array[wp.int32],
):
    world, lane = wp.tid()
    n = dimensions[world]
    base = offsets[world]
    for row in range(lane, n, 128):
        pattern[_row_start(base, row) + wp.int64(row)] = wp.int32(1)
    _block_sync()
    # Each row has one writer. Only columns after k change, leaving all
    # pivot-column reads immutable until the next synchronized pivot.
    for k in range(n):
        for row in range(k + 1 + lane, n, 128):
            destination = _row_start(base, row)
            if pattern[destination + wp.int64(k)] != 0:
                for col in range(k + 1, row + 1):
                    if pattern[_row_start(base, col) + wp.int64(k)] != 0:
                        pattern[destination + wp.int64(col)] = wp.int32(1)
        _block_sync()
    # Reuse scratch for sorted column indices. In-place writes stay behind
    # unread entries, and no other thread reads this row after elimination.
    for row in range(lane, n, 128):
        start = _row_start(base, row)
        count = wp.int32(0)
        for col in range(row + 1):
            if pattern[start + wp.int64(col)] != 0:
                pattern[start + wp.int64(count)] = col
                count += wp.int32(1)
        counts[row_offsets[world] + row] = count


@wp.kernel
def _scan_counts(counts: wp.array[wp.int32], starts: wp.array[wp.int32]):
    # Initialization-only scan: avoiding library scratch allocations allows
    # this work to execute inside a CUDA conditional graph after reset.
    total = wp.int32(0)
    for row in range(counts.shape[0]):
        starts[row] = total
        total += counts[row]


@wp.kernel
def _copy_columns(
    dimensions: wp.array[wp.int32],
    offsets: wp.array[wp.int64],
    pattern: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    columns: wp.array[wp.int32],
):
    world, row = wp.tid()
    if row < dimensions[world]:
        begin = starts[row_offsets[world] + row]
        end = starts[row_offsets[world] + row + 1]
        source = _row_start(offsets[world], row)
        for entry in range(end - begin):
            columns[begin + entry] = pattern[source + wp.int64(entry)]


class _ScalarFactorPattern:
    """Keep tight scalar factor indices and persistent CUDA initialization scratch.

    Allocate outside capture. Initialize after numerical permutation selection,
    and rebuild whenever that permutation is invalidated. Structural pairs must
    include currently cancelling entries. Changed dimensions require allocation
    and graph recapture; changed pairs require initialization and recapture.
    """

    def __init__(self, dimensions: Sequence[int], device: str | wp.context.Device):
        dimensions = [int(n) for n in dimensions]
        if any(n < 0 for n in dimensions):
            raise ValueError("Scalar factor dimensions must be nonnegative")
        capacities = [n * (n + 1) // 2 for n in dimensions]
        capacity = sum(capacities)
        row_capacity = sum(dimensions) + len(dimensions)
        if max(capacity, row_capacity) > np.iinfo(np.int32).max:
            raise ValueError("Scalar factor pattern exceeds int32 index capacity")
        self._device = device
        self._worlds = len(dimensions)
        self._max_dimension = max(dimensions, default=0)
        self._dimensions = wp.array(dimensions, dtype=wp.int32, device=device)
        self._offsets = wp.array(np.cumsum([0, *capacities])[:-1], dtype=wp.int64, device=device)
        self.row_offsets = wp.array(np.cumsum([0, *(n + 1 for n in dimensions)])[:-1], dtype=wp.int32, device=device)
        self._pattern = wp.zeros(max(1, capacity), dtype=wp.int32, device=device)
        # One zero sentinel per world makes starts[row+1] valid for its last row.
        self._counts = wp.zeros(max(1, row_capacity), dtype=wp.int32, device=device)
        self.starts = wp.zeros_like(self._counts)
        self.columns = wp.empty(max(1, capacity), dtype=wp.int32, device=device)
        self.inputs = [self.row_offsets, self.starts, self.columns]

    def initialize(
        self,
        pair_world: wp.array[wp.int32],
        pair_row: wp.array[wp.int32],
        pair_col: wp.array[wp.int32],
        inverse: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
    ) -> None:
        """Rebuild sorted indices from topology in the current numerical order."""
        self._pattern.zero_()
        self._counts.zero_()
        if pair_world.size:
            wp.launch(
                _mark_pairs,
                dim=pair_world.size,
                inputs=[pair_world, pair_row, pair_col, vector_offsets, inverse, self._offsets, self._pattern],
                device=self._device,
            )
        if self._worlds:
            wp.launch(
                _fill_and_compact,
                dim=(self._worlds, 128),
                block_dim=128,
                inputs=[self._dimensions, self._offsets, self.row_offsets, self._pattern, self._counts],
                device=self._device,
            )
        wp.launch(_scan_counts, dim=1, inputs=[self._counts, self.starts], device=self._device)
        if self._worlds and self._max_dimension:
            wp.launch(
                _copy_columns,
                dim=(self._worlds, self._max_dimension),
                inputs=[self._dimensions, self._offsets, self._pattern, self.row_offsets, self.starts, self.columns],
                device=self._device,
            )
