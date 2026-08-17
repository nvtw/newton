# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import ctypes.util
import threading

import warp as wp

_CUDA_R_32F = 0
_CUDA_R_16F = 2
_CUDA_R_16BF = 14
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_GEMM_DEFAULT = -1


class _Cublas:
    def __init__(self):
        path = ctypes.util.find_library("cublas")
        if path is None:
            raise OSError("cuBLAS library was not found")
        self.lib = ctypes.CDLL(path)
        self.lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cublasCreate_v2.restype = ctypes.c_int
        self.lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cublasSetStream_v2.restype = ctypes.c_int
        self.lib.cublasGemmEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cublasGemmEx.restype = ctypes.c_int
        self.lib.cublasGemmStridedBatchedEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cublasGemmStridedBatchedEx.restype = ctypes.c_int
        self.local = threading.local()

    def handle(self, device: wp.context.Device) -> ctypes.c_void_p:
        handles = getattr(self.local, "handles", None)
        if handles is None:
            handles = {}
            self.local.handles = handles
        key = int(device.ordinal)
        handle = handles.get(key)
        if handle is None:
            handle = ctypes.c_void_p()
            with wp.ScopedDevice(device):
                status = self.lib.cublasCreate_v2(ctypes.byref(handle))
            if status != 0:
                raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
            handles[key] = handle
        return handle


_cublas_cache: list[_Cublas | bool] = []
_cublas_lock = threading.Lock()
_alpha = ctypes.c_float(1.0)
_beta = ctypes.c_float(0.0)


def _get_cublas() -> _Cublas | None:
    if not _cublas_cache:
        with _cublas_lock:
            if not _cublas_cache:
                try:
                    cublas: _Cublas | bool = _Cublas()
                except (AttributeError, OSError):
                    cublas = False
                _cublas_cache.append(cublas)
    cublas = _cublas_cache[0]
    return cublas if isinstance(cublas, _Cublas) else None


def is_cublas_available(device: wp.context.Device) -> bool:
    """Return whether device-wide cuBLAS contractions are available."""

    if not device.is_cuda:
        return False
    cublas = _get_cublas()
    if cublas is None:
        return False
    try:
        cublas.handle(device)
    except RuntimeError:
        return False
    return True


def _gemm(
    lhs: wp.array2d[wp.bfloat16] | wp.array2d[wp.float32],
    rhs: wp.array2d[wp.bfloat16] | wp.array2d[wp.float32],
    out: wp.array2d[wp.bfloat16] | wp.array2d[wp.float32],
    rows: int,
    cols: int,
    inner: int,
    input_type: int,
    transpose_lhs: bool,
    transpose_rhs: bool,
    output_type: int = _CUDA_R_32F,
) -> None:
    cublas = _get_cublas()
    if cublas is None:
        raise RuntimeError("cuBLAS is not available")
    device = out.device
    handle = cublas.handle(device)
    op_lhs = _CUBLAS_OP_T if transpose_lhs else _CUBLAS_OP_N
    op_rhs = _CUBLAS_OP_T if transpose_rhs else _CUBLAS_OP_N
    lhs_stride = rows if transpose_lhs else inner
    rhs_stride = inner if transpose_rhs else cols
    with wp.ScopedDevice(device):
        stream = device.stream
        status = cublas.lib.cublasSetStream_v2(handle, ctypes.c_void_p(stream.cuda_stream))
        if status != 0:
            raise RuntimeError(f"cublasSetStream_v2 failed with status {status}")
        status = cublas.lib.cublasGemmEx(
            handle,
            op_rhs,
            op_lhs,
            cols,
            rows,
            inner,
            ctypes.byref(_alpha),
            ctypes.c_void_p(rhs.ptr),
            input_type,
            rhs_stride,
            ctypes.c_void_p(lhs.ptr),
            input_type,
            lhs_stride,
            ctypes.byref(_beta),
            ctypes.c_void_p(out.ptr),
            output_type,
            cols,
            _CUBLAS_COMPUTE_32F,
            _CUBLAS_GEMM_DEFAULT,
        )
    if status != 0:
        raise RuntimeError(f"cublasGemmEx failed with status {status}")


def _gemm_strided_batched(
    lhs: wp.array,
    rhs: wp.array,
    out: wp.array,
    rows: int,
    cols: int,
    inner: int,
    batch_count: int,
    lhs_batch_stride: int,
    rhs_batch_stride: int,
    out_batch_stride: int,
    transpose_lhs: bool,
    transpose_rhs: bool,
    input_type: int = _CUDA_R_32F,
    output_type: int = _CUDA_R_32F,
) -> None:
    cublas = _get_cublas()
    if cublas is None:
        raise RuntimeError("cuBLAS is not available")
    device = out.device
    handle = cublas.handle(device)
    op_lhs = _CUBLAS_OP_T if transpose_lhs else _CUBLAS_OP_N
    op_rhs = _CUBLAS_OP_T if transpose_rhs else _CUBLAS_OP_N
    lhs_stride = rows if transpose_lhs else inner
    rhs_stride = inner if transpose_rhs else cols
    with wp.ScopedDevice(device):
        stream = device.stream
        status = cublas.lib.cublasSetStream_v2(handle, ctypes.c_void_p(stream.cuda_stream))
        if status != 0:
            raise RuntimeError(f"cublasSetStream_v2 failed with status {status}")
        status = cublas.lib.cublasGemmStridedBatchedEx(
            handle,
            op_rhs,
            op_lhs,
            cols,
            rows,
            inner,
            ctypes.byref(_alpha),
            ctypes.c_void_p(rhs.ptr),
            input_type,
            rhs_stride,
            int(rhs_batch_stride),
            ctypes.c_void_p(lhs.ptr),
            input_type,
            lhs_stride,
            int(lhs_batch_stride),
            ctypes.byref(_beta),
            ctypes.c_void_p(out.ptr),
            output_type,
            cols,
            int(out_batch_stride),
            int(batch_count),
            _CUBLAS_COMPUTE_32F,
            _CUBLAS_GEMM_DEFAULT,
        )
    if status != 0:
        raise RuntimeError(f"cublasGemmStridedBatchedEx failed with status {status}")


def gemm_float16(
    lhs: wp.array2d[wp.float16],
    rhs: wp.array2d[wp.float16],
    out: wp.array2d[wp.float32],
    rows: int,
    cols: int,
    inner: int,
    *,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue a row-major FP16 GEMM with FP32 accumulation and output."""

    _gemm(lhs, rhs, out, rows, cols, inner, _CUDA_R_16F, transpose_lhs, transpose_rhs)


def gemm_float16_output(
    lhs: wp.array2d[wp.float16],
    rhs: wp.array2d[wp.float16],
    out: wp.array2d[wp.float16],
    rows: int,
    cols: int,
    inner: int,
    *,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue a row-major FP16 GEMM with FP32 accumulation and FP16 output."""

    _gemm(lhs, rhs, out, rows, cols, inner, _CUDA_R_16F, transpose_lhs, transpose_rhs, output_type=_CUDA_R_16F)


def gemm_bfloat16(
    lhs: wp.array2d[wp.bfloat16],
    rhs: wp.array2d[wp.bfloat16],
    out: wp.array2d[wp.float32],
    rows: int,
    cols: int,
    inner: int,
    *,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue a row-major BF16 GEMM with FP32 accumulation and output."""

    _gemm(lhs, rhs, out, rows, cols, inner, _CUDA_R_16BF, transpose_lhs, transpose_rhs)


def gemm_bfloat16_output(
    lhs: wp.array2d[wp.bfloat16],
    rhs: wp.array2d[wp.bfloat16],
    out: wp.array2d[wp.bfloat16],
    rows: int,
    cols: int,
    inner: int,
    *,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue a row-major BF16 GEMM with FP32 accumulation and BF16 output."""

    _gemm(
        lhs,
        rhs,
        out,
        rows,
        cols,
        inner,
        _CUDA_R_16BF,
        transpose_lhs,
        transpose_rhs,
        output_type=_CUDA_R_16BF,
    )


def gemm_float32(
    lhs: wp.array2d[wp.float32],
    rhs: wp.array2d[wp.float32],
    out: wp.array2d[wp.float32],
    rows: int,
    cols: int,
    inner: int,
    *,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue a row-major FP32 GEMM."""

    _gemm(lhs, rhs, out, rows, cols, inner, _CUDA_R_32F, transpose_lhs, transpose_rhs)


def gemm_float32_strided_batched(
    lhs: wp.array3d[wp.float32] | wp.array2d[wp.float32],
    rhs: wp.array3d[wp.float32],
    out: wp.array3d[wp.float32],
    rows: int,
    cols: int,
    inner: int,
    batch_count: int,
    *,
    broadcast_lhs: bool = False,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue row-major strided-batched FP32 GEMMs."""

    _gemm_strided_batched(
        lhs,
        rhs,
        out,
        rows,
        cols,
        inner,
        batch_count,
        0 if broadcast_lhs else rows * inner,
        inner * cols,
        rows * cols,
        transpose_lhs,
        transpose_rhs,
    )


def gemm_float16_strided_batched_output(
    lhs: wp.array3d[wp.float16] | wp.array2d[wp.float16],
    rhs: wp.array3d[wp.float16],
    out: wp.array3d[wp.float16],
    rows: int,
    cols: int,
    inner: int,
    batch_count: int,
    *,
    broadcast_lhs: bool = False,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue row-major strided-batched FP16 GEMMs with FP16 output."""

    _gemm_strided_batched(
        lhs,
        rhs,
        out,
        rows,
        cols,
        inner,
        batch_count,
        0 if broadcast_lhs else rows * inner,
        inner * cols,
        rows * cols,
        transpose_lhs,
        transpose_rhs,
        input_type=_CUDA_R_16F,
        output_type=_CUDA_R_16F,
    )


def gemm_float16_strided_batched(
    lhs: wp.array3d[wp.float16] | wp.array2d[wp.float16],
    rhs: wp.array3d[wp.float16],
    out: wp.array3d[wp.float32],
    rows: int,
    cols: int,
    inner: int,
    batch_count: int,
    *,
    broadcast_lhs: bool = False,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> None:
    """Enqueue row-major strided-batched FP16 GEMMs with FP32 output."""

    _gemm_strided_batched(
        lhs,
        rhs,
        out,
        rows,
        cols,
        inner,
        batch_count,
        0 if broadcast_lhs else rows * inner,
        inner * cols,
        rows * cols,
        transpose_lhs,
        transpose_rhs,
        input_type=_CUDA_R_16F,
        output_type=_CUDA_R_32F,
    )
