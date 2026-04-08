# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core computational kernels for PufferLib on Warp.

All kernels are compiled with ``enable_backward=False`` for fast JIT times.
Manual backward passes are provided as separate kernels and registered on the
tape via ``wp.Tape.record_func``.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Activation helpers (device functions)
# ---------------------------------------------------------------------------


@wp.func
def relu(x: float) -> float:
    return wp.max(x, 0.0)


@wp.func
def relu_backward(x: float, grad_out: float) -> float:
    if x > 0.0:
        return grad_out
    return 0.0


@wp.func
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + wp.exp(-x))


@wp.func
def softplus(x: float) -> float:
    if x > 20.0:
        return x
    return wp.log(1.0 + wp.exp(x))


@wp.func
def fast_tanh(x: float) -> float:
    return wp.tanh(x)


@wp.func
def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + wp.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


@wp.func
def logaddexp(a: float, b: float) -> float:
    m = wp.max(a, b)
    return m + wp.log(wp.exp(a - m) + wp.exp(b - m))


@wp.func
def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------


@wp.kernel
def kaiming_uniform_init_kernel(w: wp.array(dtype=float, ndim=1), fan_in: int, seed: int, gain: float):
    """Matches C++ PufferLib puf_kaiming_init: Uniform(-gain/sqrt(fan_in), gain/sqrt(fan_in))."""
    i = wp.tid()
    state = wp.rand_init(seed, i)
    bound = gain / wp.sqrt(float(fan_in))
    w[i] = wp.randf(state, -bound, bound)


# ---------------------------------------------------------------------------
# Tile constants
# ---------------------------------------------------------------------------

TILE_M = wp.constant(32)
TILE_N = wp.constant(32)
TILE_K = wp.constant(32)
BLOCK_DIM = wp.constant(128)

# ---------------------------------------------------------------------------
# Tiled GEMM (fixed 16×16×16 tiles — compiles exactly 3 kernels)
# ---------------------------------------------------------------------------

_TILE = 16
_TILE_C = wp.constant(_TILE)
_TILE_THREADS = 128


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@wp.kernel
def _tiled_gemm(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    """``C = A @ B``."""
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(_TILE_C, _TILE_C), dtype=wp.float32)
    K = A.shape[1]
    num_k = (K + _TILE_C - 1) // _TILE_C
    for k in range(num_k):
        a = wp.tile_load(A, shape=(_TILE_C, _TILE_C), offset=(i * _TILE_C, k * _TILE_C))
        b = wp.tile_load(B, shape=(_TILE_C, _TILE_C), offset=(k * _TILE_C, j * _TILE_C))
        wp.tile_matmul(a, b, acc)
    wp.tile_store(C, acc, offset=(i * _TILE_C, j * _TILE_C))


@wp.kernel
def _tiled_gemm_transA(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    """``C = A^T @ B``."""
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(_TILE_C, _TILE_C), dtype=wp.float32)
    K = A.shape[0]
    num_k = (K + _TILE_C - 1) // _TILE_C
    for k in range(num_k):
        a = wp.tile_load(A, shape=(_TILE_C, _TILE_C), offset=(k * _TILE_C, i * _TILE_C))
        at = wp.tile_transpose(a)
        b = wp.tile_load(B, shape=(_TILE_C, _TILE_C), offset=(k * _TILE_C, j * _TILE_C))
        wp.tile_matmul(at, b, acc)
    wp.tile_store(C, acc, offset=(i * _TILE_C, j * _TILE_C))


@wp.kernel
def _tiled_gemm_transB(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    """``C = A @ B^T``."""
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(_TILE_C, _TILE_C), dtype=wp.float32)
    K = A.shape[1]
    num_k = (K + _TILE_C - 1) // _TILE_C
    for k in range(num_k):
        a = wp.tile_load(A, shape=(_TILE_C, _TILE_C), offset=(i * _TILE_C, k * _TILE_C))
        b = wp.tile_load(B, shape=(_TILE_C, _TILE_C), offset=(j * _TILE_C, k * _TILE_C))
        bt = wp.tile_transpose(b)
        wp.tile_matmul(a, bt, acc)
    wp.tile_store(C, acc, offset=(i * _TILE_C, j * _TILE_C))


def matmul(A: wp.array, B: wp.array, C: wp.array,
           transpose_a: bool = False, transpose_b: bool = False) -> None:
    """Dispatch GEMM: ``C = op(A) @ op(B)``.

    Uses a fixed 16×16×16 tile so only 3 kernels are ever compiled.
    Bounds-checked tile_load/tile_store handles non-dividing dimensions.
    """
    M, N = C.shape[0], C.shape[1]
    grid = [_ceil_div(M, _TILE), _ceil_div(N, _TILE)]

    if transpose_a and not transpose_b:
        kernel = _tiled_gemm_transA
    elif not transpose_a and transpose_b:
        kernel = _tiled_gemm_transB
    elif transpose_a and transpose_b:
        raise NotImplementedError("matmul with both transpose_a and transpose_b")
    else:
        kernel = _tiled_gemm

    wp.launch_tiled(kernel, dim=grid, inputs=[A, B, C],
                    block_dim=_TILE_THREADS, device=C.device)


# ---------------------------------------------------------------------------
# Elementwise kernels
# ---------------------------------------------------------------------------


@wp.kernel
def add_kernel(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), c: wp.array2d(dtype=float)):
    i, j = wp.tid()
    c[i, j] = a[i, j] + b[i, j]


@wp.kernel
def scale_kernel(x: wp.array2d(dtype=float), alpha: float, y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    y[i, j] = x[i, j] * alpha


@wp.kernel
def relu_kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    y[i, j] = wp.max(x[i, j], 0.0)


@wp.kernel
def relu_backward_kernel(
    y: wp.array2d(dtype=float),
    grad_y: wp.array2d(dtype=float),
    grad_x: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if y[i, j] > 0.0:
        grad_x[i, j] = grad_y[i, j]
    else:
        grad_x[i, j] = 0.0
