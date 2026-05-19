# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Raw-pointer ``wp.func_native`` accessors for 2D Warp arrays.

Warp's bounds-checked ``arr[i, j]`` indexing compiles to a strided
load with a debug-build range assert and a ``__ldg`` / store wrapper.
For the inner loops in the constraint-container accessors we only ever
touch contiguous, row-major ``wp.array2d[wp.float32]`` buffers, and we
walk them with hot, compile-time-constant dword offsets where the
range check adds nothing. These helpers drop the bounds check by
casting ``arr.data`` directly and indexing flat memory.

The row pitch is read straight from ``arr.shape.dims[1]`` (elements
per row), which matches Warp's contiguous row-major layout. Callers
must therefore only pass contiguous, non-padded 2D arrays.

The cast also marks the underlying pointer ``__restrict__``. nvcc
uses that promise to issue the read-only cached load path on the
:func:`read2d_f32` side and to hoist independent loads past stores on
the :func:`write2d_f32` side. Callers must guarantee that, within a
single kernel, no other pointer aliases the buffer being read /
written through these helpers; the phoenx ``cc_*`` / ``ic_*``
accessors satisfy this trivially because each call passes a distinct
``ContactContainer`` member (``lambdas`` / ``prev_lambdas`` /
``derived``) and no kernel binds two of them to the same address.

GPU-only -- ``func_native`` snippets have no CPU fallback.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "read2d_f32",
    "write2d_f32",
]


_READ2D_F32_SNIPPET = """
    const float* __restrict__ p = (const float*)arr.data;
    return p[i * arr.shape.dims[1] + j];
"""

_WRITE2D_F32_SNIPPET = """
    float* __restrict__ p = (float*)arr.data;
    p[i * arr.shape.dims[1] + j] = v;
"""


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
