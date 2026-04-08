# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-efficient reduction primitives for PufferLib on Warp.

Provides :class:`ArraySum` for sum reduction.  For small arrays
(up to 4096 elements) a single-thread-block tile kernel is used;
for larger arrays the implementation falls back to an inclusive
prefix scan + last-element read.

:func:`array_prefix_sum` wraps ``wp.utils.array_scan``.

All buffers are pre-allocated at construction time so every operation
is CUDA-graph-capture compatible.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

_TILE_SUM_THRESHOLD = 4096
_VALID_BLOCK_DIMS = (128, 256, 512, 1024)


def _pick_block_dim(length: int) -> int:
    """Choose the smallest power-of-2 block dim in [128..1024] >= *length*, or 1024."""
    for bd in _VALID_BLOCK_DIMS:
        if bd >= length:
            return bd
    return 1024


def _make_tile_sum_kernel(block_dim: int):
    """Return a cached single-block tile-sum kernel for *block_dim* threads."""

    @wp.kernel(module="unique")
    def _tile_sum_kernel(
        data: wp.array(dtype=float),
        length: int,
        result: wp.array(dtype=float),
    ):
        _tid, lane = wp.tid()

        num_threads = wp.block_dim()
        partial = float(0.0)

        upper = ((length + num_threads - 1) // num_threads) * num_threads
        for idx in range(lane, upper, num_threads):
            val = float(0.0)
            if idx < length:
                val = data[idx]
            t = wp.tile(val)
            s = wp.tile_sum(t)
            partial += s[0]

        if lane == 0:
            result[0] = partial

    return _tile_sum_kernel


_tile_sum_kernels: dict[int, object] = {}


def _get_tile_sum_kernel(block_dim: int):
    """Get or create the tile-sum kernel for the given block dim."""
    kernel = _tile_sum_kernels.get(block_dim)
    if kernel is None:
        kernel = _make_tile_sum_kernel(block_dim)
        _tile_sum_kernels[block_dim] = kernel
    return kernel


@wp.kernel
def _copy_last_kernel(
    scan_out: wp.array(dtype=float, ndim=1),
    last_idx: int,
    result: wp.array(dtype=float, ndim=1),
):
    """Copy the last element of the inclusive scan to *result*."""
    result[0] = scan_out[last_idx]


class ArraySum:
    """Sum reduction with a fast single-block path for small arrays.

    For arrays up to 4096 elements a single-thread-block tile kernel
    is launched (no scratch buffer needed).  For larger arrays the
    implementation falls back to an inclusive prefix scan.

    Pre-allocates all scratch buffers so :meth:`compute` is
    graph-capture safe.

    Args:
        max_length: Maximum array length this reducer will handle.
        device: Warp device string.
    """

    def __init__(self, max_length: int, device: str = "cuda:0"):
        self.max_length = max_length
        self.device = device

        if max_length > _TILE_SUM_THRESHOLD:
            self._scan_buf = wp.zeros(max_length, dtype=float, device=device)
        else:
            self._scan_buf = None
        self.result = wp.zeros(1, dtype=float, device=device)

    def compute(self, data: wp.array, length: int | None = None) -> wp.array:
        """Compute the sum of the first *length* elements of *data*.

        Returns ``self.result`` — a (1,) array reused across calls.
        """
        if length is None:
            length = data.shape[0]

        if length <= _TILE_SUM_THRESHOLD:
            block_dim = _pick_block_dim(length)
            kernel = _get_tile_sum_kernel(block_dim)
            wp.launch_tiled(
                kernel,
                dim=[1],
                inputs=[data, length, self.result],
                block_dim=block_dim,
                device=self.device,
            )
        else:
            if self._scan_buf is None:
                self._scan_buf = wp.zeros(self.max_length, dtype=float, device=self.device)
            wp.utils.array_scan(data, self._scan_buf, inclusive=True)
            wp.launch(
                _copy_last_kernel,
                dim=1,
                inputs=[self._scan_buf, length - 1, self.result],
                device=self.device,
            )
        return self.result


def array_prefix_sum(src: wp.array, dst: wp.array):
    """Inclusive prefix sum using Warp's built-in GPU scan.

    Graph-capture safe as long as *src* and *dst* are pre-allocated.
    """
    wp.utils.array_scan(src, dst, inclusive=True)
