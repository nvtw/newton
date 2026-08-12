# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Narrow graph-capturable cuBLAS support for batched DVI response solves."""

from __future__ import annotations

import ctypes
import ctypes.util
import threading

import warp as wp

_CUBLAS_SIDE_LEFT = 0
_CUBLAS_FILL_UPPER = 1
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUBLAS_DIAG_NON_UNIT = 0


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
        self.lib.cublasStrsmBatched.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cublasStrsmBatched.restype = ctypes.c_int
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


_cache: list[_Cublas | bool] = []
_lock = threading.Lock()
_alpha = ctypes.c_float(1.0)


def _get_cublas() -> _Cublas | None:
    if not _cache:
        with _lock:
            if not _cache:
                try:
                    library: _Cublas | bool = _Cublas()
                except (AttributeError, OSError):
                    library = False
                _cache.append(library)
    library = _cache[0]
    return library if isinstance(library, _Cublas) else None


def is_batched_trsm_available(device: wp.context.Device) -> bool:
    """Return whether graph-capturable batched triangular solves are available."""
    if not device.is_cuda:
        return False
    library = _get_cublas()
    if library is None:
        return False
    try:
        library.handle(device)
    except RuntimeError:
        return False
    return True


def solve_llt_batched(
    factor_ptrs: wp.array[wp.uint64],
    rhs_ptrs: wp.array[wp.uint64],
    rows: int,
    rhs_count: int,
    batch_count: int,
) -> None:
    """Enqueue ``L L^T X = B`` for row-major ``L`` and RHS-major ``B`` batches."""
    library = _get_cublas()
    if library is None:
        raise RuntimeError("cuBLAS is not available")
    device = rhs_ptrs.device
    handle = library.handle(device)
    with wp.ScopedDevice(device):
        stream = device.stream
        status = library.lib.cublasSetStream_v2(handle, ctypes.c_void_p(stream.cuda_stream))
        if status != 0:
            raise RuntimeError(f"cublasSetStream_v2 failed with status {status}")
        for operation in (_CUBLAS_OP_T, _CUBLAS_OP_N):
            status = library.lib.cublasStrsmBatched(
                handle,
                _CUBLAS_SIDE_LEFT,
                _CUBLAS_FILL_UPPER,
                operation,
                _CUBLAS_DIAG_NON_UNIT,
                rows,
                rhs_count,
                ctypes.byref(_alpha),
                ctypes.c_void_p(factor_ptrs.ptr),
                rows,
                ctypes.c_void_p(rhs_ptrs.ptr),
                rows,
                batch_count,
            )
            if status != 0:
                raise RuntimeError(f"cublasStrsmBatched failed with status {status}")
