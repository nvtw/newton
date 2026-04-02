# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal ONNX inference runtime backed by Warp kernels.

Only the ``onnx`` package (pure protobuf parser) is required -- no
``onnxruntime`` or ``torch``.  Weights are loaded once onto the target
Warp device; inference executes a pre-built list of lightweight op
descriptors that dispatch to Warp kernels.

The Gemm and MatMul operators use **tiled** matrix multiplication via
``wp.tile_load`` / ``wp.tile_matmul`` / ``wp.tile_store``, which maps to
hardware tensor-core acceleration when available.  Activation and
element-wise operators are separate kernels so the ONNX graph stays
composable without forced fusion.

Supported ONNX operators (easily extensible):

* **Gemm** -- general matrix multiply with optional transpose and bias
* **MatMul** -- plain matrix multiplication
* **Elu**  -- ``y = x if x >= 0 else alpha * (exp(x) - 1)``
* **Relu** -- ``y = max(0, x)``
* **Tanh** -- ``y = tanh(x)``
* **Sigmoid** -- ``y = 1 / (1 + exp(-x))``
* **Add** -- element-wise addition (with broadcast for 1-D bias)
* **Identity** -- pass-through

Example::

    from newton._src.onnx_runtime import OnnxRuntime

    rt = OnnxRuntime("policy.onnx", device="cuda:0")
    out = rt({"observation": wp.array(obs, dtype=wp.float32, device="cuda:0")})
    actions = out["action"]
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
import warp as wp
from onnx import numpy_helper

# ---------------------------------------------------------------------------
# Tiled GEMM kernel factory
# ---------------------------------------------------------------------------

_TILE_THREADS = 256


@functools.cache
def _make_tiled_gemm_kernel(tile_m: int, tile_n: int, tile_k: int):
    """Return a Warp kernel for ``C = A @ B`` using tiles of the given size.

    The kernel is launched with ``wp.launch_tiled`` over
    ``dim=(ceil(M/tile_m), ceil(N/tile_n))`` and iterates over the K
    dimension in chunks of *tile_k*.  ``bounds_check`` on tile loads/stores
    handles matrices whose dimensions are not multiples of the tile size.
    """
    TILE_M = wp.constant(tile_m)
    TILE_N = wp.constant(tile_n)
    TILE_K = wp.constant(tile_k)

    @wp.kernel
    def _tiled_gemm(
        A: wp.array2d[float],
        B: wp.array2d[float],
        C: wp.array2d[float],
    ):
        i, j = wp.tid()
        acc = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        K = A.shape[1]
        num_k_tiles = (K + TILE_K - 1) // TILE_K

        for k in range(num_k_tiles):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
            wp.tile_matmul(a, b, acc)

        wp.tile_store(C, acc, offset=(i * TILE_M, j * TILE_N))

    return _tiled_gemm


@functools.cache
def _make_tiled_gemm_transB_kernel(tile_m: int, tile_n: int, tile_k: int):
    """``C = A @ B^T`` where B has shape (N, K) stored row-major."""
    TILE_M = wp.constant(tile_m)
    TILE_N = wp.constant(tile_n)
    TILE_K = wp.constant(tile_k)

    @wp.kernel
    def _tiled_gemm_transB(
        A: wp.array2d[float],
        B: wp.array2d[float],
        C: wp.array2d[float],
    ):
        i, j = wp.tid()
        acc = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        K = A.shape[1]
        num_k_tiles = (K + TILE_K - 1) // TILE_K

        for k in range(num_k_tiles):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
            b = wp.tile_load(B, shape=(TILE_N, TILE_K), offset=(j * TILE_N, k * TILE_K))
            bt = wp.tile_transpose(b)
            wp.tile_matmul(a, bt, acc)

        wp.tile_store(C, acc, offset=(i * TILE_M, j * TILE_N))

    return _tiled_gemm_transB


# ---------------------------------------------------------------------------
# Element-wise kernels (separate from matmul for composability)
# ---------------------------------------------------------------------------


@wp.kernel
def _bias_add_kernel(C: wp.array2d[float], bias: wp.array[float], alpha: float, beta: float):
    """``C[i,j] = alpha * C[i,j] + beta * bias[j]``."""
    i, j = wp.tid()
    C[i, j] = alpha * C[i, j] + beta * bias[j]


@wp.kernel
def _elu_kernel(x: wp.array2d[float], y: wp.array2d[float], alpha: float):
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.where(v >= 0.0, v, alpha * (wp.exp(v) - 1.0))


@wp.kernel
def _relu_kernel(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()
    y[i, j] = wp.max(x[i, j], 0.0)


@wp.kernel
def _tanh_kernel(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()
    y[i, j] = wp.tanh(x[i, j])


@wp.kernel
def _sigmoid_kernel(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()
    y[i, j] = 1.0 / (1.0 + wp.exp(-x[i, j]))


@wp.kernel
def _add_2d_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    c[i, j] = a[i, j] + b[i, j]


@wp.kernel
def _add_broadcast_1d_kernel(a: wp.array2d[float], b: wp.array[float], c: wp.array2d[float]):
    """Add with broadcast: ``c[i,j] = a[i,j] + b[j]``."""
    i, j = wp.tid()
    c[i, j] = a[i, j] + b[j]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _Op:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


def _get_attr(node, name: str, default=None):
    """Extract a named attribute from an ONNX ``NodeProto``."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == 1:  # FLOAT
                return attr.f
            if attr.type == 2:  # INT
                return attr.i
            if attr.type == 3:  # STRING
                return attr.s.decode()
            if attr.type == 6:  # FLOATS
                return list(attr.floats)
            if attr.type == 7:  # INTS
                return list(attr.ints)
    return default


def _pick_tile_sizes(M: int, N: int, K: int) -> tuple[int, int, int]:
    """Choose tile sizes that divide reasonably into the matrix dimensions.

    Prefers tiles that evenly divide each dimension.  Falls back to smaller
    tiles when the dimension is small to avoid excessive padding.  Max tile
    size is 32 to keep shared-memory usage reasonable.
    """

    def _best(dim: int, candidates: tuple[int, ...] = (32, 16, 8, 4)) -> int:
        for c in candidates:
            if dim >= c:
                return c
        return candidates[-1]

    return _best(M), _best(N), _best(K)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OnnxRuntime:
    """Lightweight ONNX inference engine using Warp kernels.

    Args:
        path: Path to an ``.onnx`` file.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.  Defaults to ``1``.
    """

    def __init__(self, path: str, device: str | None = None, batch_size: int = 1):
        self._device = wp.get_device(device)
        self._batch_size = batch_size

        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=False)
        graph = model.graph

        # -- 1. Load initializers (weights / biases) as Warp arrays ----------
        self._tensors: dict[str, wp.array] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

        for init in graph.initializer:
            arr_np = numpy_helper.to_array(init).astype(np.float32)
            if arr_np.ndim == 1:
                wa = wp.array(arr_np, dtype=wp.float32, device=self._device)
            else:
                wa = wp.array2d(arr_np, dtype=wp.float32, device=self._device)
            self._tensors[init.name] = wa
            self._shapes[init.name] = tuple(arr_np.shape)

        # -- 2. Record input / output names ----------------------------------
        initializer_names = {init.name for init in graph.initializer}
        self.input_names: list[str] = [inp.name for inp in graph.input if inp.name not in initializer_names]
        self.output_names: list[str] = [out.name for out in graph.output]

        # -- 3. Build op list ------------------------------------------------
        self._ops: list[_Op] = []
        for node in graph.node:
            attrs: dict[str, Any] = {}
            for a in node.attribute:
                attrs[a.name] = _get_attr(node, a.name)
            self._ops.append(
                _Op(
                    op_type=node.op_type,
                    inputs=list(node.input),
                    outputs=list(node.output),
                    attrs=attrs,
                )
            )

        # -- 4. Shape inference & buffer pre-allocation ----------------------
        self._preallocate_buffers(batch_size)

    # ------------------------------------------------------------------
    # Buffer pre-allocation
    # ------------------------------------------------------------------

    def _preallocate_buffers(self, batch_size: int) -> None:
        """Walk the graph symbolically to determine intermediate shapes and
        allocate reusable device buffers."""
        for op in self._ops:
            out_name = op.outputs[0]

            if op.op_type == "Gemm":
                B_shape = self._shapes[op.inputs[1]]
                transB = op.attrs.get("transB", 0)
                N = B_shape[0] if transB else B_shape[1]
                out_shape = (batch_size, N)

            elif op.op_type == "MatMul":
                B_shape = self._shapes[op.inputs[1]]
                N = B_shape[1] if len(B_shape) == 2 else B_shape[0]
                out_shape = (batch_size, N)

            elif op.op_type in ("Elu", "Relu", "Tanh", "Sigmoid"):
                out_shape = self._shapes[op.inputs[0]]

            elif op.op_type == "Add":
                out_shape = self._shapes[op.inputs[0]]

            elif op.op_type == "Identity":
                self._shapes[out_name] = self._shapes[op.inputs[0]]
                continue

            else:
                raise NotImplementedError(
                    f"OnnxRuntime: unsupported op '{op.op_type}'.  "
                    f"Supported: Gemm, MatMul, Elu, Relu, Tanh, Sigmoid, Add, Identity"
                )

            if out_name not in self._tensors:
                self._tensors[out_name] = wp.zeros(out_shape, dtype=wp.float32, device=self._device)
            self._shapes[out_name] = out_shape

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, inputs: dict[str, wp.array]) -> dict[str, wp.array]:
        """Run forward inference.

        Args:
            inputs: Mapping of ONNX input names to ``wp.array2d`` tensors
                already on the correct device.

        Returns:
            Mapping of ONNX output names to ``wp.array2d`` result tensors.
        """
        tensors = self._tensors

        for name, arr in inputs.items():
            tensors[name] = arr
            self._shapes[name] = tuple(arr.shape)

        for op in self._ops:
            self._exec_op(op, tensors)

        return {name: tensors[name] for name in self.output_names}

    def _exec_op(self, op: _Op, tensors: dict[str, wp.array]) -> None:
        dispatch = _OP_DISPATCH.get(op.op_type)
        if dispatch is None:
            raise NotImplementedError(f"OnnxRuntime: unsupported op '{op.op_type}'")
        dispatch(op, tensors, self._shapes, self._device)


# ---------------------------------------------------------------------------
# Op implementations
# ---------------------------------------------------------------------------


def _launch_tiled_gemm(
    A: wp.array,
    B: wp.array,
    C: wp.array,
    M: int,
    N: int,
    K: int,
    transB: bool,
    device: wp.context.Device,
) -> None:
    """Launch a tiled GEMM kernel for ``C = A @ B`` (or ``A @ B^T``)."""
    tile_m, tile_n, tile_k = _pick_tile_sizes(M, N, K)
    grid = (_ceil_div(M, tile_m), _ceil_div(N, tile_n))

    if transB:
        kernel = _make_tiled_gemm_transB_kernel(tile_m, tile_n, tile_k)
    else:
        kernel = _make_tiled_gemm_kernel(tile_m, tile_n, tile_k)

    wp.launch_tiled(
        kernel,
        dim=list(grid),
        inputs=[A, B, C],
        block_dim=_TILE_THREADS,
        device=device,
    )


def _exec_gemm(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    alpha = float(op.attrs.get("alpha", 1.0))
    beta = float(op.attrs.get("beta", 1.0))
    transA = int(op.attrs.get("transA", 0))
    transB = int(op.attrs.get("transB", 0))

    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]

    if transA:
        M, K = A_shape[1], A_shape[0]
    else:
        M, K = A_shape[0], A_shape[1]

    if transB:
        N = B_shape[0]
    else:
        N = B_shape[1]

    out = tensors[op.outputs[0]]

    # Tiled matmul: C = A @ B  or  C = A @ B^T
    _launch_tiled_gemm(A, B, out, M, N, K, transB=bool(transB), device=device)

    # Apply alpha scaling and bias addition if needed
    has_bias = len(op.inputs) > 2 and op.inputs[2] != ""
    if has_bias:
        bias = tensors[op.inputs[2]]
        wp.launch(_bias_add_kernel, dim=(M, N), inputs=[out, bias, alpha, beta], device=device)
    elif alpha != 1.0:
        wp.launch(
            _bias_add_kernel,
            dim=(M, N),
            inputs=[out, wp.zeros(N, dtype=wp.float32, device=device), alpha, 0.0],
            device=device,
        )

    shapes[op.outputs[0]] = (M, N)


def _exec_matmul(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    M, K = A_shape[0], A_shape[1]
    N = B_shape[1]
    out = tensors[op.outputs[0]]

    _launch_tiled_gemm(A, B, out, M, N, K, transB=False, device=device)
    shapes[op.outputs[0]] = (M, N)


def _exec_elu(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    x = tensors[op.inputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_elu_kernel, dim=shape, inputs=[x, out, alpha], device=device)
    shapes[op.outputs[0]] = shape


def _exec_relu(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    x = tensors[op.inputs[0]]
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_relu_kernel, dim=shape, inputs=[x, out], device=device)
    shapes[op.outputs[0]] = shape


def _exec_tanh(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    x = tensors[op.inputs[0]]
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_tanh_kernel, dim=shape, inputs=[x, out], device=device)
    shapes[op.outputs[0]] = shape


def _exec_sigmoid(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    x = tensors[op.inputs[0]]
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_sigmoid_kernel, dim=shape, inputs=[x, out], device=device)
    shapes[op.outputs[0]] = shape


def _exec_add(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    a = tensors[op.inputs[0]]
    b = tensors[op.inputs[1]]
    a_shape = shapes[op.inputs[0]]
    b_shape = shapes[op.inputs[1]]
    out = tensors[op.outputs[0]]

    if len(b_shape) == 1:
        wp.launch(_add_broadcast_1d_kernel, dim=a_shape, inputs=[a, b, out], device=device)
    else:
        wp.launch(_add_2d_kernel, dim=a_shape, inputs=[a, b, out], device=device)

    shapes[op.outputs[0]] = a_shape


def _exec_identity(
    op: _Op,
    tensors: dict[str, wp.array],
    shapes: dict[str, tuple[int, ...]],
    device: wp.context.Device,
) -> None:
    tensors[op.outputs[0]] = tensors[op.inputs[0]]
    shapes[op.outputs[0]] = shapes[op.inputs[0]]


_OP_DISPATCH: dict[str, Any] = {
    "Gemm": _exec_gemm,
    "MatMul": _exec_matmul,
    "Elu": _exec_elu,
    "Relu": _exec_relu,
    "Tanh": _exec_tanh,
    "Sigmoid": _exec_sigmoid,
    "Add": _exec_add,
    "Identity": _exec_identity,
}
