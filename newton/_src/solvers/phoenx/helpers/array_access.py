# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Raw-pointer ``wp.func_native`` accessors for hot Warp array reads/writes.

Warp's bounds-checked indexing compiles to a strided load with a
debug-build range assert and a cached load / store wrapper. For the
inner loops in the constraint-container accessors and CSR dispatchers
we only ever touch contiguous buffers with proven in-range indices,
where the range check adds nothing. These helpers drop the bounds
check by casting ``arr.data`` directly and indexing flat memory.

The 2D row pitch is read straight from ``arr.shape.dims[1]``
(elements per row), which matches Warp's contiguous row-major layout.
Callers must therefore only pass contiguous, non-padded arrays.

The cast also marks the underlying pointer ``__restrict__``, which
helps nvcc avoid conservative aliasing around these scalar accesses.
These helpers do not force a separate ``__ldg`` path; keep new uses to
buffers whose indexing and aliasing have been checked in the calling
kernel. In particular, mutable PGS state should not be routed through
a read-only helper unless the kernel never writes that same storage.

GPU-only -- ``func_native`` snippets have no CPU fallback.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "read1d_i32",
    "read2d_f32",
    "write2d_f32",
]


_READ1D_I32_SNIPPET = """
    const int* __restrict__ p = (const int*)arr.data;
    return p[i];
"""

_READ2D_F32_SNIPPET = """
    const float* __restrict__ p = (const float*)arr.data;
    return p[i * arr.shape.dims[1] + j];
"""

_WRITE2D_F32_SNIPPET = """
    float* __restrict__ p = (float*)arr.data;
    p[i * arr.shape.dims[1] + j] = v;
"""


@wp.func_native(_READ1D_I32_SNIPPET)
def read1d_i32(arr: wp.array[wp.int32], i: wp.int32) -> wp.int32:
    """Bounds-check-free read of ``arr[i]`` for a contiguous 1D
    ``int32`` array. Caller is responsible for keeping ``i`` in-range."""
    ...


@wp.func_native(_READ2D_F32_SNIPPET)
def read2d_f32(arr: wp.array2d[wp.float32], i: wp.int32, j: wp.int32) -> wp.float32:
    """Bounds-check-free read of ``arr[i, j]`` for a contiguous 2D
    ``float32`` array. Caller is responsible for keeping ``(i, j)``
    in-range."""
    ...


@wp.func_native(_WRITE2D_F32_SNIPPET)
def write2d_f32(arr: wp.array2d[wp.float32], i: wp.int32, j: wp.int32, v: wp.float32):
    """Bounds-check-free write of ``v`` to ``arr[i, j]`` for a
    contiguous 2D ``float32`` array. Caller is responsible for keeping
    ``(i, j)`` in-range."""
    ...
