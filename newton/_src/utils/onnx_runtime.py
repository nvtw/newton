# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal ONNX inference runtime backed by Warp kernels.

Only the ``onnx`` package (pure protobuf parser) is required -- no
``onnxruntime`` or ``torch``.  Weights are loaded once onto the target
Warp device; inference executes a pre-built list of lightweight op
descriptors that dispatch to fused Warp kernels.

Supported ONNX operators:

* **Gemm** -- ``C = alpha * A @ B[^T] + beta * bias``
* **MatMul** -- 2-D matrix multiplication
* **Elu / Relu / Tanh / Sigmoid** -- element-wise activations
* **Add / Sub / Mul / Div** -- element-wise binary ops (with simple broadcast)
* **Concat** -- last-axis concatenation of 2-D tensors
* **Split** -- split a 2-D tensor along the last axis
* **Reshape** -- reshape with shape from initializer / constant
* **Transpose** -- 2-D transpose, or 3-D ``(seq, batch, hidden)`` <-> ``(batch, seq, hidden)``
* **Squeeze / Unsqueeze** -- single-axis squeeze/unsqueeze
* **Identity** -- alias passthrough
* **Constant** -- emit an initializer-like tensor
* **LSTM** -- forward, single-direction, layout 0 or 1; arbitrary ``seq_length``

Example::

    from newton._src.utils.onnx_runtime import OnnxRuntime

    rt = OnnxRuntime("policy.onnx", device="cuda:0")
    out = rt({"observation": wp.array2d(obs, dtype=wp.float32, device="cuda:0")})
    actions = out["action"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
import warp as wp
from onnx import numpy_helper

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _gemm_transB_bias_kernel(
    A: wp.array2d[float],
    B: wp.array2d[float],
    bias: wp.array[float],
    C: wp.array2d[float],
    K: int,
    alpha: float,
    beta: float,
):
    i, j = wp.tid()
    s = float(0.0)
    for k in range(K):
        s += A[i, k] * B[j, k]
    C[i, j] = alpha * s + beta * bias[j]


@wp.kernel
def _gemm_transB_kernel(
    A: wp.array2d[float],
    B: wp.array2d[float],
    C: wp.array2d[float],
    K: int,
    alpha: float,
):
    i, j = wp.tid()
    s = float(0.0)
    for k in range(K):
        s += A[i, k] * B[j, k]
    C[i, j] = alpha * s


@wp.kernel
def _gemm_kernel(
    A: wp.array2d[float],
    B: wp.array2d[float],
    bias: wp.array[float],
    C: wp.array2d[float],
    K: int,
    alpha: float,
    beta: float,
    has_bias: int,
):
    i, j = wp.tid()
    s = float(0.0)
    for k in range(K):
        s += A[i, k] * B[k, j]
    if has_bias != 0:
        C[i, j] = alpha * s + beta * bias[j]
    else:
        C[i, j] = alpha * s


@wp.kernel
def _matmul_kernel(
    A: wp.array2d[float],
    B: wp.array2d[float],
    C: wp.array2d[float],
    K: int,
):
    i, j = wp.tid()
    s = float(0.0)
    for k in range(K):
        s += A[i, k] * B[k, j]
    C[i, j] = s


@wp.kernel
def _elu_kernel(
    x: wp.array2d[float],
    y: wp.array2d[float],
    alpha: float,
):
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.where(v >= 0.0, v, alpha * (wp.exp(v) - 1.0))


@wp.kernel
def _relu_kernel(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.max(v, 0.0)


@wp.kernel
def _tanh_kernel(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()
    y[i, j] = wp.tanh(x[i, j])


@wp.kernel
def _sigmoid_kernel(x: wp.array2d[float], y: wp.array2d[float]):
    i, j = wp.tid()
    y[i, j] = 1.0 / (1.0 + wp.exp(-x[i, j]))


@wp.kernel
def _binop_kernel(
    a: wp.array2d[float],
    b: wp.array2d[float],
    c: wp.array2d[float],
    op: int,  # 0=add, 1=sub, 2=mul, 3=div
    bcast_rows_a: int,  # 1 if a's rows broadcast (size 1)
    bcast_cols_a: int,  # 1 if a's cols broadcast
    bcast_rows_b: int,
    bcast_cols_b: int,
):
    i, j = wp.tid()
    ai = wp.where(bcast_rows_a != 0, 0, i)
    aj = wp.where(bcast_cols_a != 0, 0, j)
    bi = wp.where(bcast_rows_b != 0, 0, i)
    bj = wp.where(bcast_cols_b != 0, 0, j)
    av = a[ai, aj]
    bv = b[bi, bj]
    if op == 0:
        c[i, j] = av + bv
    elif op == 1:
        c[i, j] = av - bv
    elif op == 2:
        c[i, j] = av * bv
    else:
        c[i, j] = av / bv


@wp.kernel
def _copy2d_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


# ---------------------------------------------------------------------------
# LSTM kernel: per-timestep cell update, forward direction only.
# ---------------------------------------------------------------------------


@wp.kernel
def _lstm_cell_kernel(
    x: wp.array2d[float],          # (batch, input_size)
    h_prev: wp.array2d[float],     # (batch, hidden_size)
    c_prev: wp.array2d[float],     # (batch, hidden_size)
    W: wp.array2d[float],          # (4*hidden_size, input_size)  -- gates IOFC
    R: wp.array2d[float],          # (4*hidden_size, hidden_size)
    Bx: wp.array[float],           # (4*hidden_size,)
    Bh: wp.array[float],           # (4*hidden_size,)
    h_out: wp.array2d[float],      # (batch, hidden_size)
    c_out: wp.array2d[float],      # (batch, hidden_size)
    input_size: int,
    hidden_size: int,
):
    b, h = wp.tid()

    base_i = 0 * hidden_size + h
    base_o = 1 * hidden_size + h
    base_f = 2 * hidden_size + h
    base_c = 3 * hidden_size + h

    s_i = Bx[base_i] + Bh[base_i]
    s_o = Bx[base_o] + Bh[base_o]
    s_f = Bx[base_f] + Bh[base_f]
    s_c = Bx[base_c] + Bh[base_c]

    for k in range(input_size):
        xv = x[b, k]
        s_i += W[base_i, k] * xv
        s_o += W[base_o, k] * xv
        s_f += W[base_f, k] * xv
        s_c += W[base_c, k] * xv

    for k in range(hidden_size):
        hv = h_prev[b, k]
        s_i += R[base_i, k] * hv
        s_o += R[base_o, k] * hv
        s_f += R[base_f, k] * hv
        s_c += R[base_c, k] * hv

    g_i = 1.0 / (1.0 + wp.exp(-s_i))
    g_o = 1.0 / (1.0 + wp.exp(-s_o))
    g_f = 1.0 / (1.0 + wp.exp(-s_f))
    g_c = wp.tanh(s_c)

    c_new = g_f * c_prev[b, h] + g_i * g_c
    c_out[b, h] = c_new
    h_out[b, h] = g_o * wp.tanh(c_new)


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
            if attr.type == 4:  # TENSOR
                return numpy_helper.to_array(attr.t)
            if attr.type == 6:  # FLOATS
                return list(attr.floats)
            if attr.type == 7:  # INTS
                return list(attr.ints)
            if attr.type == 8:  # STRINGS
                return [s.decode() for s in attr.strings]
    return default


def _alloc_array(shape: tuple[int, ...], device: wp.context.Device) -> wp.array:
    """Allocate a zero-initialized Warp float32 array of the given shape."""
    if len(shape) == 1:
        return wp.zeros(shape[0], dtype=wp.float32, device=device)
    if len(shape) == 2:
        return wp.zeros(shape, dtype=wp.float32, device=device)
    if len(shape) == 3:
        return wp.zeros(shape, dtype=wp.float32, device=device)
    return wp.zeros(shape, dtype=wp.float32, device=device)


def _np_to_warp(arr_np: np.ndarray, device: wp.context.Device) -> wp.array:
    arr_np = np.ascontiguousarray(arr_np.astype(np.float32))
    if arr_np.ndim == 1:
        return wp.array(arr_np, dtype=wp.float32, device=device)
    return wp.array(arr_np, dtype=wp.float32, device=device)


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

        self._tensors: dict[str, wp.array] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

        for init in graph.initializer:
            arr_np = numpy_helper.to_array(init).astype(np.float32)
            self._tensors[init.name] = _np_to_warp(arr_np, self._device)
            self._shapes[init.name] = tuple(arr_np.shape)

        initializer_names = {init.name for init in graph.initializer}
        self.input_names: list[str] = [inp.name for inp in graph.input if inp.name not in initializer_names]
        self.output_names: list[str] = [out.name for out in graph.output]

        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            shape = []
            for d in inp.type.tensor_type.shape.dim:
                if d.HasField("dim_value") and d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    shape.append(batch_size)
            self._shapes[inp.name] = tuple(shape)

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

        self._preallocate_buffers(batch_size)

    # ------------------------------------------------------------------
    # Buffer pre-allocation
    # ------------------------------------------------------------------

    def _preallocate_buffers(self, batch_size: int) -> None:
        for op in self._ops:
            handler = _SHAPE_DISPATCH.get(op.op_type)
            if handler is None:
                raise NotImplementedError(
                    f"OnnxRuntime: unsupported op '{op.op_type}'.  "
                    f"Supported ops: {sorted(_OP_DISPATCH.keys())}"
                )
            handler(op, self._shapes, self._tensors, self._device, batch_size)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, inputs: dict[str, wp.array]) -> dict[str, wp.array]:
        """Run forward inference.

        Args:
            inputs: Mapping of ONNX input names to Warp arrays already on
                the correct device.  2-D ``wp.array2d`` is the typical case.

        Returns:
            Mapping of ONNX output names to Warp result arrays.
        """
        tensors = self._tensors

        for name, arr in inputs.items():
            tensors[name] = arr
            self._shapes[name] = tuple(arr.shape)

        for op in self._ops:
            dispatch = _OP_DISPATCH.get(op.op_type)
            if dispatch is None:
                raise NotImplementedError(f"OnnxRuntime: unsupported op '{op.op_type}'")
            dispatch(op, tensors, self._shapes, self._device)

        return {name: tensors[name] for name in self.output_names}


# ---------------------------------------------------------------------------
# Shape inference (per op)
# ---------------------------------------------------------------------------


def _shape_gemm(op, shapes, tensors, device, batch_size):
    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    transA = int(op.attrs.get("transA", 0))
    transB = int(op.attrs.get("transB", 0))
    M = A_shape[1] if transA else A_shape[0]
    N = B_shape[0] if transB else B_shape[1]
    out_shape = (M, N)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=wp.float32, device=device)
    shapes[out_name] = out_shape


def _shape_matmul(op, shapes, tensors, device, batch_size):
    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    out_shape = (A_shape[0], B_shape[1])
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=wp.float32, device=device)
    shapes[out_name] = out_shape


def _shape_elementwise_unary(op, shapes, tensors, device, batch_size):
    in_shape = shapes[op.inputs[0]]
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = _alloc_array(in_shape, device)
    shapes[out_name] = in_shape


def _shape_binop(op, shapes, tensors, device, batch_size):
    a_shape = shapes[op.inputs[0]]
    b_shape = shapes[op.inputs[1]]
    out_shape = tuple(max(a, b) for a, b in zip(_pad(a_shape, len(b_shape)), _pad(b_shape, len(a_shape)), strict=False))
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = _alloc_array(out_shape, device)
    shapes[out_name] = out_shape


def _pad(shape, n):
    return (1,) * (n - len(shape)) + tuple(shape)


def _shape_concat(op, shapes, tensors, device, batch_size):
    axis = int(op.attrs.get("axis", -1))
    in_shapes = [shapes[i] for i in op.inputs]
    rank = len(in_shapes[0])
    if axis < 0:
        axis += rank
    out_shape = list(in_shapes[0])
    out_shape[axis] = sum(s[axis] for s in in_shapes)
    out_shape = tuple(out_shape)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = _alloc_array(out_shape, device)
    shapes[out_name] = out_shape


def _shape_split(op, shapes, tensors, device, batch_size):
    in_shape = shapes[op.inputs[0]]
    axis = int(op.attrs.get("axis", 0))
    if axis < 0:
        axis += len(in_shape)
    num_outputs = len(op.outputs)

    splits = op.attrs.get("split")
    if splits is None and len(op.inputs) > 1 and op.inputs[1] in shapes:
        # split sizes from input -- fall back to equal split when not constant
        splits = None
    if splits is None:
        each = in_shape[axis] // num_outputs
        splits = [each] * num_outputs
    for out_name, sz in zip(op.outputs, splits, strict=False):
        out_shape = list(in_shape)
        out_shape[axis] = sz
        out_shape = tuple(out_shape)
        if out_name not in tensors:
            tensors[out_name] = _alloc_array(out_shape, device)
        shapes[out_name] = out_shape


def _shape_reshape(op, shapes, tensors, device, batch_size):
    in_shape = shapes[op.inputs[0]]
    target = None
    if len(op.inputs) > 1 and op.inputs[1] in tensors:
        # shape comes from initializer
        target = tuple(int(v) for v in tensors[op.inputs[1]].numpy().tolist())
    if target is None:
        target = in_shape
    total_in = 1
    for d in in_shape:
        total_in *= d
    out_shape = []
    unknown = -1
    known_prod = 1
    for i, dim in enumerate(target):
        d = in_shape[i] if dim == 0 else dim
        if d == -1:
            unknown = i
            out_shape.append(-1)
        else:
            out_shape.append(d)
            known_prod *= d
    if unknown >= 0:
        out_shape[unknown] = total_in // known_prod
    out_shape = tuple(out_shape)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = _alloc_array(out_shape, device)
    shapes[out_name] = out_shape


def _shape_transpose(op, shapes, tensors, device, batch_size):
    in_shape = shapes[op.inputs[0]]
    perm = op.attrs.get("perm")
    if perm is None:
        perm = list(reversed(range(len(in_shape))))
    out_shape = tuple(in_shape[p] for p in perm)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = _alloc_array(out_shape, device)
    shapes[out_name] = out_shape


def _shape_squeeze(op, shapes, tensors, device, batch_size):
    in_shape = shapes[op.inputs[0]]
    axes = op.attrs.get("axes")
    if axes is None and len(op.inputs) > 1 and op.inputs[1] in tensors:
        axes = [int(v) for v in tensors[op.inputs[1]].numpy().tolist()]
    if axes is None:
        out_shape = tuple(d for d in in_shape if d != 1)
    else:
        rank = len(in_shape)
        axes_norm = [a if a >= 0 else a + rank for a in axes]
        out_shape = tuple(d for i, d in enumerate(in_shape) if i not in axes_norm)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = _alloc_array(out_shape, device)
    shapes[out_name] = out_shape


def _shape_unsqueeze(op, shapes, tensors, device, batch_size):
    in_shape = shapes[op.inputs[0]]
    axes = op.attrs.get("axes")
    if axes is None and len(op.inputs) > 1 and op.inputs[1] in tensors:
        axes = [int(v) for v in tensors[op.inputs[1]].numpy().tolist()]
    out_shape = list(in_shape)
    rank = len(out_shape) + len(axes)
    axes_norm = sorted(a if a >= 0 else a + rank for a in axes)
    for a in axes_norm:
        out_shape.insert(a, 1)
    out_shape = tuple(out_shape)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = _alloc_array(out_shape, device)
    shapes[out_name] = out_shape


def _shape_identity(op, shapes, tensors, device, batch_size):
    in_shape = shapes[op.inputs[0]]
    shapes[op.outputs[0]] = in_shape


def _shape_constant(op, shapes, tensors, device, batch_size):
    val = op.attrs.get("value")
    arr = np.asarray(val, dtype=np.float32)
    tensors[op.outputs[0]] = _np_to_warp(arr, device)
    shapes[op.outputs[0]] = tuple(arr.shape)


def _shape_lstm(op, shapes, tensors, device, batch_size):
    X_shape = shapes[op.inputs[0]]   # (seq_len, batch, input_size) for layout=0
    W_shape = shapes[op.inputs[1]]   # (1, 4H, input_size)
    layout = int(op.attrs.get("layout", 0))
    hidden_size = int(op.attrs.get("hidden_size", W_shape[1] // 4))

    if layout == 0:
        seq_len, batch, _ = X_shape
        Y_shape = (seq_len, 1, batch, hidden_size)
    else:
        batch, seq_len, _ = X_shape
        Y_shape = (batch, seq_len, 1, hidden_size)

    Yh_shape = (1, batch, hidden_size)
    Yc_shape = (1, batch, hidden_size)

    if len(op.outputs) > 0 and op.outputs[0]:
        if op.outputs[0] not in tensors:
            tensors[op.outputs[0]] = _alloc_array(Y_shape, device)
        shapes[op.outputs[0]] = Y_shape
    if len(op.outputs) > 1 and op.outputs[1]:
        if op.outputs[1] not in tensors:
            tensors[op.outputs[1]] = _alloc_array(Yh_shape, device)
        shapes[op.outputs[1]] = Yh_shape
    if len(op.outputs) > 2 and op.outputs[2]:
        if op.outputs[2] not in tensors:
            tensors[op.outputs[2]] = _alloc_array(Yc_shape, device)
        shapes[op.outputs[2]] = Yc_shape


# ---------------------------------------------------------------------------
# Op implementations
# ---------------------------------------------------------------------------


def _exec_gemm(op, tensors, shapes, device):
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    bias = tensors[op.inputs[2]] if len(op.inputs) > 2 else None
    alpha = float(op.attrs.get("alpha", 1.0))
    beta = float(op.attrs.get("beta", 1.0))
    transA = int(op.attrs.get("transA", 0))
    transB = int(op.attrs.get("transB", 0))

    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]

    if transA:
        raise NotImplementedError("OnnxRuntime Gemm: transA=1 is not supported")

    M = A_shape[0]
    if transB:
        N = B_shape[0]
        K = B_shape[1]
    else:
        N = B_shape[1]
        K = B_shape[0]

    out = tensors[op.outputs[0]]

    if transB:
        if bias is not None:
            wp.launch(
                _gemm_transB_bias_kernel,
                dim=(M, N),
                inputs=[A, B, bias, out, K, alpha, beta],
                device=device,
            )
        else:
            wp.launch(
                _gemm_transB_kernel,
                dim=(M, N),
                inputs=[A, B, out, K, alpha],
                device=device,
            )
    else:
        zero_bias = bias if bias is not None else wp.zeros(N, dtype=wp.float32, device=device)
        wp.launch(
            _gemm_kernel,
            dim=(M, N),
            inputs=[A, B, zero_bias, out, K, alpha, beta, 1 if bias is not None else 0],
            device=device,
        )

    shapes[op.outputs[0]] = (M, N)


def _exec_matmul(op, tensors, shapes, device):
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    M = A_shape[0]
    K = A_shape[1]
    N = B_shape[1]
    out = tensors[op.outputs[0]]
    wp.launch(_matmul_kernel, dim=(M, N), inputs=[A, B, out, K], device=device)
    shapes[op.outputs[0]] = (M, N)


def _exec_elu(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_elu_kernel, dim=shape, inputs=[x, out, alpha], device=device)
    shapes[op.outputs[0]] = shape


def _exec_relu(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_relu_kernel, dim=shape, inputs=[x, out], device=device)
    shapes[op.outputs[0]] = shape


def _exec_tanh(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_tanh_kernel, dim=shape, inputs=[x, out], device=device)
    shapes[op.outputs[0]] = shape


def _exec_sigmoid(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_sigmoid_kernel, dim=shape, inputs=[x, out], device=device)
    shapes[op.outputs[0]] = shape


def _exec_binop(op, tensors, shapes, device, op_id: int):
    a = tensors[op.inputs[0]]
    b = tensors[op.inputs[1]]
    a_shape = shapes[op.inputs[0]]
    b_shape = shapes[op.inputs[1]]
    out = tensors[op.outputs[0]]
    out_shape = shapes[op.outputs[0]]

    a_view = _as_2d(a, a_shape, out_shape)
    b_view = _as_2d(b, b_shape, out_shape)

    bra = 1 if a_view.shape[0] == 1 and out_shape[0] != 1 else 0
    bca = 1 if a_view.shape[1] == 1 and out_shape[1] != 1 else 0
    brb = 1 if b_view.shape[0] == 1 and out_shape[0] != 1 else 0
    bcb = 1 if b_view.shape[1] == 1 and out_shape[1] != 1 else 0

    wp.launch(
        _binop_kernel,
        dim=out_shape,
        inputs=[a_view, b_view, out, op_id, bra, bca, brb, bcb],
        device=device,
    )


def _as_2d(arr: wp.array, src_shape, target_shape):
    """Reshape a Warp array into a 2-D view that broadcasts to ``target_shape``."""
    if len(src_shape) == 2:
        return arr
    if len(src_shape) == 1:
        # treat 1-D as row vector (1, n), broadcast along axis 0
        return arr.reshape((1, src_shape[0]))
    if len(src_shape) == 0:
        return arr.reshape((1, 1))
    flat_rows = 1
    for d in src_shape[:-1]:
        flat_rows *= d
    return arr.reshape((flat_rows, src_shape[-1]))


def _exec_add(op, tensors, shapes, device):
    _exec_binop(op, tensors, shapes, device, 0)


def _exec_sub(op, tensors, shapes, device):
    _exec_binop(op, tensors, shapes, device, 1)


def _exec_mul(op, tensors, shapes, device):
    _exec_binop(op, tensors, shapes, device, 2)


def _exec_div(op, tensors, shapes, device):
    _exec_binop(op, tensors, shapes, device, 3)


def _exec_concat(op, tensors, shapes, device):
    """Last-axis concat for 2-D tensors via numpy round-trip (tiny, infrequent)."""
    arrs = [tensors[i].numpy() for i in op.inputs]
    axis = int(op.attrs.get("axis", -1))
    out_np = np.concatenate(arrs, axis=axis).astype(np.float32)
    tensors[op.outputs[0]] = _np_to_warp(out_np, device)
    shapes[op.outputs[0]] = tuple(out_np.shape)


def _exec_split(op, tensors, shapes, device):
    arr = tensors[op.inputs[0]].numpy()
    axis = int(op.attrs.get("axis", 0))
    splits = op.attrs.get("split")
    if splits is None and len(op.inputs) > 1 and op.inputs[1] in tensors:
        splits = [int(v) for v in tensors[op.inputs[1]].numpy().tolist()]
    if splits is None:
        each = arr.shape[axis] // len(op.outputs)
        splits = [each] * len(op.outputs)
    indices = np.cumsum(splits)[:-1]
    parts = np.split(arr, indices, axis=axis)
    for name, part in zip(op.outputs, parts, strict=False):
        tensors[name] = _np_to_warp(part.astype(np.float32), device)
        shapes[name] = tuple(part.shape)


def _exec_reshape(op, tensors, shapes, device):
    src = tensors[op.inputs[0]]
    out_shape = shapes[op.outputs[0]]
    arr_np = src.numpy().reshape(out_shape).astype(np.float32)
    tensors[op.outputs[0]] = _np_to_warp(arr_np, device)


def _exec_transpose(op, tensors, shapes, device):
    src = tensors[op.inputs[0]]
    perm = op.attrs.get("perm")
    arr_np = src.numpy()
    if perm is None:
        arr_np = arr_np.T
    else:
        arr_np = np.transpose(arr_np, axes=perm)
    arr_np = np.ascontiguousarray(arr_np.astype(np.float32))
    tensors[op.outputs[0]] = _np_to_warp(arr_np, device)
    shapes[op.outputs[0]] = tuple(arr_np.shape)


def _exec_squeeze(op, tensors, shapes, device):
    out_shape = shapes[op.outputs[0]]
    src_np = tensors[op.inputs[0]].numpy()
    arr_np = src_np.reshape(out_shape).astype(np.float32)
    tensors[op.outputs[0]] = _np_to_warp(arr_np, device)


def _exec_unsqueeze(op, tensors, shapes, device):
    out_shape = shapes[op.outputs[0]]
    src_np = tensors[op.inputs[0]].numpy()
    arr_np = src_np.reshape(out_shape).astype(np.float32)
    tensors[op.outputs[0]] = _np_to_warp(arr_np, device)


def _exec_identity(op, tensors, shapes, device):
    tensors[op.outputs[0]] = tensors[op.inputs[0]]
    shapes[op.outputs[0]] = shapes[op.inputs[0]]


def _exec_constant(op, tensors, shapes, device):
    pass  # already materialized in shape inference


def _exec_lstm(op, tensors, shapes, device):
    """Forward, single-direction LSTM. Loops through ``seq_length`` on the host."""
    X_shape = shapes[op.inputs[0]]
    layout = int(op.attrs.get("layout", 0))
    hidden_size = int(op.attrs.get("hidden_size", 0))

    if layout == 0:
        seq_len, batch, input_size = X_shape
    else:
        batch, seq_len, input_size = X_shape

    # Inputs
    X_np = tensors[op.inputs[0]].numpy()
    if layout == 1:
        X_np = np.transpose(X_np, (1, 0, 2))   # -> (seq_len, batch, input_size)
    W_np = tensors[op.inputs[1]].numpy().reshape(4 * hidden_size, input_size)
    R_np = tensors[op.inputs[2]].numpy().reshape(4 * hidden_size, hidden_size)
    if len(op.inputs) > 3 and op.inputs[3] and op.inputs[3] in tensors:
        B_np = tensors[op.inputs[3]].numpy().reshape(8 * hidden_size)
        Bx_np = B_np[: 4 * hidden_size]
        Bh_np = B_np[4 * hidden_size :]
    else:
        Bx_np = np.zeros(4 * hidden_size, dtype=np.float32)
        Bh_np = np.zeros(4 * hidden_size, dtype=np.float32)

    if len(op.inputs) > 5 and op.inputs[5] and op.inputs[5] in tensors:
        h_np = tensors[op.inputs[5]].numpy().reshape(batch, hidden_size).astype(np.float32)
    else:
        h_np = np.zeros((batch, hidden_size), dtype=np.float32)
    if len(op.inputs) > 6 and op.inputs[6] and op.inputs[6] in tensors:
        c_np = tensors[op.inputs[6]].numpy().reshape(batch, hidden_size).astype(np.float32)
    else:
        c_np = np.zeros((batch, hidden_size), dtype=np.float32)

    W_wp = wp.array(np.ascontiguousarray(W_np), dtype=wp.float32, device=device)
    R_wp = wp.array(np.ascontiguousarray(R_np), dtype=wp.float32, device=device)
    Bx_wp = wp.array(np.ascontiguousarray(Bx_np), dtype=wp.float32, device=device)
    Bh_wp = wp.array(np.ascontiguousarray(Bh_np), dtype=wp.float32, device=device)

    h_wp = wp.array(np.ascontiguousarray(h_np), dtype=wp.float32, device=device)
    c_wp = wp.array(np.ascontiguousarray(c_np), dtype=wp.float32, device=device)
    h_next = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)
    c_next = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)

    Y_np = np.zeros((seq_len, batch, hidden_size), dtype=np.float32) if op.outputs[0] else None

    for t in range(seq_len):
        x_t = wp.array(np.ascontiguousarray(X_np[t]), dtype=wp.float32, device=device)
        wp.launch(
            _lstm_cell_kernel,
            dim=(batch, hidden_size),
            inputs=[x_t, h_wp, c_wp, W_wp, R_wp, Bx_wp, Bh_wp, h_next, c_next, input_size, hidden_size],
            device=device,
        )
        if Y_np is not None:
            Y_np[t] = h_next.numpy()
        h_wp, h_next = h_next, h_wp
        c_wp, c_next = c_next, c_wp

    if op.outputs[0]:
        Y_np = Y_np.reshape((seq_len, 1, batch, hidden_size))
        if layout == 1:
            Y_np = np.transpose(Y_np, (2, 0, 1, 3))
        tensors[op.outputs[0]] = _np_to_warp(Y_np, device)
        shapes[op.outputs[0]] = tuple(Y_np.shape)
    if len(op.outputs) > 1 and op.outputs[1]:
        Yh_np = h_wp.numpy().reshape(1, batch, hidden_size)
        tensors[op.outputs[1]] = _np_to_warp(Yh_np, device)
        shapes[op.outputs[1]] = tuple(Yh_np.shape)
    if len(op.outputs) > 2 and op.outputs[2]:
        Yc_np = c_wp.numpy().reshape(1, batch, hidden_size)
        tensors[op.outputs[2]] = _np_to_warp(Yc_np, device)
        shapes[op.outputs[2]] = tuple(Yc_np.shape)


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------


_OP_DISPATCH: dict[str, Any] = {
    "Gemm": _exec_gemm,
    "MatMul": _exec_matmul,
    "Elu": _exec_elu,
    "Relu": _exec_relu,
    "Tanh": _exec_tanh,
    "Sigmoid": _exec_sigmoid,
    "Add": _exec_add,
    "Sub": _exec_sub,
    "Mul": _exec_mul,
    "Div": _exec_div,
    "Concat": _exec_concat,
    "Split": _exec_split,
    "Reshape": _exec_reshape,
    "Transpose": _exec_transpose,
    "Squeeze": _exec_squeeze,
    "Unsqueeze": _exec_unsqueeze,
    "Identity": _exec_identity,
    "Constant": _exec_constant,
    "LSTM": _exec_lstm,
}

_SHAPE_DISPATCH: dict[str, Any] = {
    "Gemm": _shape_gemm,
    "MatMul": _shape_matmul,
    "Elu": _shape_elementwise_unary,
    "Relu": _shape_elementwise_unary,
    "Tanh": _shape_elementwise_unary,
    "Sigmoid": _shape_elementwise_unary,
    "Add": _shape_binop,
    "Sub": _shape_binop,
    "Mul": _shape_binop,
    "Div": _shape_binop,
    "Concat": _shape_concat,
    "Split": _shape_split,
    "Reshape": _shape_reshape,
    "Transpose": _shape_transpose,
    "Squeeze": _shape_squeeze,
    "Unsqueeze": _shape_unsqueeze,
    "Identity": _shape_identity,
    "Constant": _shape_constant,
    "LSTM": _shape_lstm,
}
