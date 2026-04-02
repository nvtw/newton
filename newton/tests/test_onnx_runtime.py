# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Warp-based ONNX runtime.

Every supported ONNX operator is tested against a NumPy reference with
tight tolerances.  Tests build ONNX graphs programmatically so no
external model files are required.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
import warp as wp
from onnx import TensorProto, helper, numpy_helper

from newton._src.onnx_runtime import OnnxRuntime, _ceil_div, _pick_tile_sizes

# ---------------------------------------------------------------------------
# ONNX graph builder helpers
# ---------------------------------------------------------------------------

POLICY_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "_src", "anymal_c_walking_policy.onnx")


def _save_model(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, path)


def _make_gemm_graph(
    M, K, N, *, transB=0, alpha=1.0, beta=1.0, has_bias=True, seed=0,
):
    """Build an ONNX graph: output = alpha * (input @ W) + beta * bias.

    W is stored as (N, K) when transB=1, or (K, N) when transB=0.
    """
    rng = np.random.default_rng(seed)
    if transB:
        w_np = rng.standard_normal((N, K)).astype(np.float32)
    else:
        w_np = rng.standard_normal((K, N)).astype(np.float32)
    b_np = rng.standard_normal(N).astype(np.float32)

    inits = [numpy_helper.from_array(w_np, name="W")]
    gemm_inputs = ["X", "W"]
    if has_bias:
        inits.append(numpy_helper.from_array(b_np, name="B"))
        gemm_inputs.append("B")

    node = helper.make_node(
        "Gemm", gemm_inputs, ["Y"],
        alpha=alpha, beta=beta, transB=transB,
    )
    graph = helper.make_graph(
        [node], "test_gemm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
        initializer=inits,
    )
    return graph, w_np, b_np


def _make_matmul_graph(M, K, N, seed=0):
    rng = np.random.default_rng(seed)
    w_np = rng.standard_normal((K, N)).astype(np.float32)
    inits = [numpy_helper.from_array(w_np, name="W")]
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph(
        [node], "test_matmul",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
        initializer=inits,
    )
    return graph, w_np


def _make_activation_graph(op_type, M, N, **attrs):
    """Build: X -> Activation -> Y."""
    node = helper.make_node(op_type, ["X"], ["Y"], **attrs)
    graph = helper.make_graph(
        [node], f"test_{op_type.lower()}",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
    )
    return graph


def _make_add_graph(M, N, broadcast_1d, seed=0):
    """Build: X + B -> Y.  B is 1D (N,) if broadcast_1d else 2D (M, N)."""
    rng = np.random.default_rng(seed)
    if broadcast_1d:
        b_np = rng.standard_normal(N).astype(np.float32)
        b_shape = [N]
    else:
        b_np = rng.standard_normal((M, N)).astype(np.float32)
        b_shape = [M, N]

    inits = [numpy_helper.from_array(b_np, name="B")]
    node = helper.make_node("Add", ["X", "B"], ["Y"])
    graph = helper.make_graph(
        [node], "test_add",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
        initializer=inits,
    )
    return graph, b_np


def _make_identity_graph(M, N):
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "test_identity",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
    )
    return graph


def _make_layer_norm_graph(M, N, eps=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    gamma_np = rng.standard_normal(N).astype(np.float32)
    beta_np = rng.standard_normal(N).astype(np.float32)
    inits = [
        numpy_helper.from_array(gamma_np, name="gamma"),
        numpy_helper.from_array(beta_np, name="beta"),
    ]
    node = helper.make_node(
        "LayerNormalization", ["X", "gamma", "beta"], ["Y"],
        epsilon=eps, axis=-1,
    )
    graph = helper.make_graph(
        [node], "test_layer_norm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, N])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
        initializer=inits,
    )
    return graph, gamma_np, beta_np


def _make_mlp_graph(layer_sizes, activation="Elu", seed=0):
    """Build a multi-layer Gemm+Activation graph (no activation on last layer)."""
    rng = np.random.default_rng(seed)
    nodes = []
    inits = []
    prev = "X"
    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        w_name = f"W{i}"
        b_name = f"B{i}"
        gemm_out = f"gemm_{i}"

        w_np = rng.standard_normal((fan_out, fan_in)).astype(np.float32) * 0.1
        b_np = rng.standard_normal(fan_out).astype(np.float32) * 0.01
        weights.append(w_np)
        biases.append(b_np)

        inits.append(numpy_helper.from_array(w_np, name=w_name))
        inits.append(numpy_helper.from_array(b_np, name=b_name))
        nodes.append(helper.make_node(
            "Gemm", [prev, w_name, b_name], [gemm_out],
            alpha=1.0, beta=1.0, transB=1,
        ))

        is_last = i == len(layer_sizes) - 2
        if not is_last:
            act_out = f"act_{i}"
            nodes.append(helper.make_node(activation, [gemm_out], [act_out]))
            prev = act_out
        else:
            prev = gemm_out

    nodes[-1].output[0] = "Y"
    graph = helper.make_graph(
        nodes, "test_mlp",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", layer_sizes[0]])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", layer_sizes[-1]])],
        initializer=inits,
    )
    return graph, weights, biases


def _run_onnx(graph, x_np, batch_size=None):
    """Save graph to temp file, run through OnnxRuntime, return output NumPy array.

    Seeds the input shape into the runtime so that ops consuming the graph
    input directly (activations, Identity, Add, LayerNorm) can preallocate.
    """
    if batch_size is None:
        batch_size = x_np.shape[0]
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        _save_model(graph, path)
        rt = OnnxRuntime(path, device="cpu", batch_size=batch_size)
        x_wp = wp.array(x_np, dtype=wp.float32, device="cpu")
        out = rt({"X": x_wp})
        return out["Y"].numpy()
    finally:
        os.unlink(path)


def _run_onnx_seeded(graph, x_np, batch_size=None):
    """Like _run_onnx but seeds input shapes before construction.

    Needed for graphs where the first op directly consumes the graph input
    (e.g. standalone activation, Identity, Add, LayerNorm) since the runtime
    only learns input shapes at call time, not during buffer preallocation.
    """
    if batch_size is None:
        batch_size = x_np.shape[0]
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        _save_model(graph, path)
        rt = OnnxRuntime.__new__(OnnxRuntime)
        rt._device = wp.get_device("cpu")
        rt._batch_size = batch_size

        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=False)
        g = model.graph

        rt._tensors = {}
        rt._shapes = {}
        for init in g.initializer:
            arr_np = numpy_helper.to_array(init).astype(np.float32)
            if arr_np.ndim == 1:
                wa = wp.array(arr_np, dtype=wp.float32, device="cpu")
            else:
                wa = wp.array2d(arr_np, dtype=wp.float32, device="cpu")
            rt._tensors[init.name] = wa
            rt._shapes[init.name] = tuple(arr_np.shape)

        initializer_names = {init.name for init in g.initializer}
        rt.input_names = [inp.name for inp in g.input if inp.name not in initializer_names]
        rt.output_names = [out.name for out in g.output]

        rt._shapes["X"] = tuple(x_np.shape)

        from newton._src.onnx_runtime import _Op, _get_attr
        rt._ops = []
        for node in g.node:
            attrs = {}
            for a in node.attribute:
                attrs[a.name] = _get_attr(node, a.name)
            rt._ops.append(_Op(op_type=node.op_type, inputs=list(node.input), outputs=list(node.output), attrs=attrs))

        rt._preallocate_buffers(batch_size)

        x_wp = wp.array(x_np, dtype=wp.float32, device="cpu")
        out = rt({"X": x_wp})
        return out["Y"].numpy()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCeilDiv(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual(_ceil_div(16, 4), 4)

    def test_remainder(self):
        self.assertEqual(_ceil_div(17, 4), 5)

    def test_one(self):
        self.assertEqual(_ceil_div(1, 32), 1)

    def test_equal(self):
        self.assertEqual(_ceil_div(32, 32), 1)


class TestPickTileSizes(unittest.TestCase):
    def test_large_dims(self):
        tm, tn, tk = _pick_tile_sizes(128, 64, 256)
        self.assertEqual(tm, 32)
        self.assertEqual(tn, 32)
        self.assertEqual(tk, 32)

    def test_small_dims(self):
        tm, tn, tk = _pick_tile_sizes(3, 5, 2)
        self.assertEqual(tm, 4)
        self.assertEqual(tn, 4)
        self.assertEqual(tk, 4)

    def test_medium_dims(self):
        tm, tn, tk = _pick_tile_sizes(16, 8, 32)
        self.assertEqual(tm, 16)
        self.assertEqual(tn, 8)
        self.assertEqual(tk, 32)


class TestGemm(unittest.TestCase):
    def test_gemm_no_transpose(self):
        """Gemm with transB=0: Y = X @ W + B."""
        M, K, N = 4, 8, 6
        graph, w_np, b_np = _make_gemm_graph(M, K, N, transB=0, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_gemm_transB(self):
        """Gemm with transB=1: Y = X @ W^T + B."""
        M, K, N = 4, 8, 6
        graph, w_np, b_np = _make_gemm_graph(M, K, N, transB=1, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np.T + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_gemm_no_bias(self):
        """Gemm without bias: Y = X @ W."""
        M, K, N = 4, 8, 6
        graph, w_np, _ = _make_gemm_graph(M, K, N, transB=0, has_bias=False, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_gemm_alpha_beta(self):
        """Gemm with non-default alpha and beta: Y = 0.5 * (X @ W^T) + 2.0 * B."""
        M, K, N = 4, 8, 6
        graph, w_np, b_np = _make_gemm_graph(M, K, N, transB=1, alpha=0.5, beta=2.0, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = 0.5 * (x_np @ w_np.T) + 2.0 * b_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_gemm_non_aligned_dims(self):
        """Gemm with dimensions that are not multiples of any tile size."""
        M, K, N = 3, 7, 5
        graph, w_np, b_np = _make_gemm_graph(M, K, N, transB=1, seed=42)
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np.T + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_gemm_large_matrix(self):
        """Gemm with dimensions requiring multiple tiles."""
        M, K, N = 64, 128, 48
        graph, w_np, b_np = _make_gemm_graph(M, K, N, transB=1, seed=7)
        rng = np.random.default_rng(8)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np.T + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-4)

    def test_gemm_single_row(self):
        """Gemm with batch_size=1."""
        M, K, N = 1, 16, 8
        graph, w_np, b_np = _make_gemm_graph(M, K, N, transB=1, seed=0)
        x_np = np.ones((1, K), dtype=np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np.T + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_gemm_alpha_no_bias(self):
        """Gemm with alpha != 1.0 but no bias (exercises the alpha-only branch)."""
        M, K, N = 4, 8, 6
        graph, w_np, _ = _make_gemm_graph(M, K, N, transB=1, alpha=0.5, has_bias=False, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = 0.5 * (x_np @ w_np.T)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


class TestMatMul(unittest.TestCase):
    def test_matmul_matches_numpy(self):
        """MatMul: Y = X @ W."""
        M, K, N = 4, 8, 6
        graph, w_np = _make_matmul_graph(M, K, N, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_matmul_non_aligned(self):
        """MatMul with non-aligned dimensions."""
        M, K, N = 3, 5, 7
        graph, w_np = _make_matmul_graph(M, K, N, seed=42)
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_matmul_large(self):
        """MatMul with multi-tile dimensions."""
        M, K, N = 32, 64, 48
        graph, w_np = _make_matmul_graph(M, K, N, seed=7)
        rng = np.random.default_rng(8)
        x_np = rng.standard_normal((M, K)).astype(np.float32)

        result = _run_onnx(graph, x_np)
        expected = x_np @ w_np
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-4)


class TestActivations(unittest.TestCase):
    def _test_activation(self, op_type, numpy_fn, M=4, N=8, **attrs):
        graph = _make_activation_graph(op_type, M, N, **attrs)
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((M, N)).astype(np.float32) * 2

        result = _run_onnx_seeded(graph, x_np)
        expected = numpy_fn(x_np)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_elu_default_alpha(self):
        self._test_activation(
            "Elu",
            lambda x: np.where(x >= 0, x, 1.0 * (np.exp(x) - 1)),
        )

    def test_elu_custom_alpha(self):
        alpha = 0.5
        self._test_activation(
            "Elu",
            lambda x: np.where(x >= 0, x, alpha * (np.exp(x) - 1)),
            alpha=alpha,
        )

    def test_relu(self):
        self._test_activation("Relu", lambda x: np.maximum(x, 0))

    def test_tanh(self):
        self._test_activation("Tanh", np.tanh)

    def test_sigmoid(self):
        self._test_activation("Sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x)))

    def test_elu_preserves_positive(self):
        """ELU should pass positive values through unchanged."""
        graph = _make_activation_graph("Elu", 1, 4)
        x_np = np.array([[1.0, 2.0, 0.5, 10.0]], dtype=np.float32)
        result = _run_onnx_seeded(graph, x_np)
        np.testing.assert_allclose(result, x_np, atol=1e-6)

    def test_elu_negative_values(self):
        """ELU negative branch: alpha * (exp(x) - 1) for specific negative inputs."""
        alpha = 2.0
        graph = _make_activation_graph("Elu", 1, 4, alpha=alpha)
        x_np = np.array([[-0.5, -1.0, -3.0, -10.0]], dtype=np.float32)
        result = _run_onnx_seeded(graph, x_np)
        expected = alpha * (np.exp(x_np) - 1)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_relu_zeros_negatives(self):
        """ReLU should zero out all negative values."""
        graph = _make_activation_graph("Relu", 1, 4)
        x_np = np.array([[-1.0, 0.0, 1.0, -100.0]], dtype=np.float32)
        result = _run_onnx_seeded(graph, x_np)
        expected = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_sigmoid_range(self):
        """Sigmoid output should be in [0, 1] and match NumPy at moderate inputs."""
        graph = _make_activation_graph("Sigmoid", 2, 4)
        x_np = np.array([[-5.0, -1.0, 0.0, 1.0], [5.0, 3.0, -3.0, 0.5]], dtype=np.float32)
        result = _run_onnx_seeded(graph, x_np)
        self.assertTrue(np.all(result >= 0.0) and np.all(result <= 1.0))
        np.testing.assert_allclose(result[0, 2], 0.5, atol=1e-6)
        expected = 1.0 / (1.0 + np.exp(-x_np))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


class TestAdd(unittest.TestCase):
    def test_add_2d(self):
        """Add two 2D arrays element-wise."""
        M, N = 4, 6
        graph, b_np = _make_add_graph(M, N, broadcast_1d=False, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, N)).astype(np.float32)

        result = _run_onnx_seeded(graph, x_np)
        expected = x_np + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_add_broadcast_1d(self):
        """Add 2D + 1D with broadcast along rows."""
        M, N = 4, 6
        graph, b_np = _make_add_graph(M, N, broadcast_1d=True, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, N)).astype(np.float32)

        result = _run_onnx_seeded(graph, x_np)
        expected = x_np + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


class TestIdentity(unittest.TestCase):
    def test_identity_passthrough(self):
        """Identity op should return the input unchanged."""
        M, N = 3, 5
        graph = _make_identity_graph(M, N)
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((M, N)).astype(np.float32)

        result = _run_onnx_seeded(graph, x_np)
        np.testing.assert_array_equal(result, x_np)


class TestLayerNorm(unittest.TestCase):
    def test_layer_norm_matches_numpy(self):
        """LayerNormalization should match NumPy reference."""
        M, N = 4, 8
        eps = 1e-5
        graph, gamma_np, beta_np = _make_layer_norm_graph(M, N, eps=eps, seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((M, N)).astype(np.float32) * 3 + 1

        result = _run_onnx_seeded(graph, x_np)

        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        expected = gamma_np * (x_np - mean) / np.sqrt(var + eps) + beta_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_layer_norm_constant_input(self):
        """Constant input should produce beta (since normalized input is zero)."""
        M, N = 2, 4
        graph, gamma_np, beta_np = _make_layer_norm_graph(M, N, seed=42)
        x_np = np.full((M, N), 5.0, dtype=np.float32)

        result = _run_onnx_seeded(graph, x_np)
        expected = np.tile(beta_np, (M, 1))
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


class TestMultiLayerMLP(unittest.TestCase):
    def test_mlp_elu_matches_numpy(self):
        """Multi-layer Gemm+Elu graph should match NumPy reference."""
        layer_sizes = [8, 16, 12, 4]
        graph, weights, biases = _make_mlp_graph(layer_sizes, activation="Elu", seed=0)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 8)).astype(np.float32)

        result = _run_onnx(graph, x_np, batch_size=2)

        h = x_np
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            h = h @ w.T + b
            if i < len(weights) - 1:
                h = np.where(h >= 0, h, np.exp(h) - 1)
        np.testing.assert_allclose(result, h, rtol=1e-4, atol=1e-5)

    def test_mlp_relu_matches_numpy(self):
        """Multi-layer Gemm+Relu graph should match NumPy reference."""
        layer_sizes = [6, 10, 4]
        graph, weights, biases = _make_mlp_graph(layer_sizes, activation="Relu", seed=7)
        rng = np.random.default_rng(8)
        x_np = rng.standard_normal((3, 6)).astype(np.float32)

        result = _run_onnx(graph, x_np, batch_size=3)

        h = x_np
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            h = h @ w.T + b
            if i < len(weights) - 1:
                h = np.maximum(h, 0)
        np.testing.assert_allclose(result, h, rtol=1e-4, atol=1e-5)

    def test_mlp_tanh_matches_numpy(self):
        """Multi-layer Gemm+Tanh graph should match NumPy reference."""
        layer_sizes = [4, 8, 3]
        graph, weights, biases = _make_mlp_graph(layer_sizes, activation="Tanh", seed=99)
        rng = np.random.default_rng(100)
        x_np = rng.standard_normal((2, 4)).astype(np.float32)

        result = _run_onnx(graph, x_np, batch_size=2)

        h = x_np
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            h = h @ w.T + b
            if i < len(weights) - 1:
                h = np.tanh(h)
        np.testing.assert_allclose(result, h, rtol=1e-4, atol=1e-5)

    def test_mlp_non_aligned_dims(self):
        """MLP with dimensions not aligned to tile sizes."""
        layer_sizes = [7, 13, 5, 3]
        graph, weights, biases = _make_mlp_graph(layer_sizes, activation="Elu", seed=42)
        rng = np.random.default_rng(43)
        x_np = rng.standard_normal((3, 7)).astype(np.float32)

        result = _run_onnx(graph, x_np, batch_size=3)

        h = x_np
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            h = h @ w.T + b
            if i < len(weights) - 1:
                h = np.where(h >= 0, h, np.exp(h) - 1)
        np.testing.assert_allclose(result, h, rtol=1e-4, atol=1e-5)


class TestDeterminismAndBatch(unittest.TestCase):
    def test_deterministic(self):
        """Two runs with the same input should produce identical output."""
        layer_sizes = [4, 8, 3]
        graph, _, _ = _make_mlp_graph(layer_sizes, seed=0)
        x_np = np.ones((2, 4), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            _save_model(graph, path)
            rt = OnnxRuntime(path, device="cpu", batch_size=2)
            x_wp = wp.array(x_np, dtype=wp.float32, device="cpu")
            out1 = rt({"X": x_wp})["Y"].numpy()
            out2 = rt({"X": x_wp})["Y"].numpy()
            np.testing.assert_array_equal(out1, out2)
        finally:
            os.unlink(path)

    def test_batch_matches_single(self):
        """Batched inference should match row-by-row single inference."""
        layer_sizes = [6, 12, 4]
        graph, _, _ = _make_mlp_graph(layer_sizes, seed=0)
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 6)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            _save_model(graph, path)

            rt_batch = OnnxRuntime(path, device="cpu", batch_size=4)
            batch_out = rt_batch({"X": wp.array(x_np, dtype=wp.float32, device="cpu")})["Y"].numpy()

            rt_single = OnnxRuntime(path, device="cpu", batch_size=1)
            for i in range(4):
                row = x_np[i:i + 1]
                single_out = rt_single({"X": wp.array(row, dtype=wp.float32, device="cpu")})["Y"].numpy()
                np.testing.assert_allclose(batch_out[i:i + 1], single_out, atol=1e-5)
        finally:
            os.unlink(path)


class TestUnsupportedOp(unittest.TestCase):
    def test_unsupported_op_raises(self):
        """Loading a graph with an unsupported op should raise NotImplementedError."""
        node = helper.make_node("Softmax", ["X"], ["Y"])
        graph = helper.make_graph(
            [node], "test_unsupported",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            _save_model(graph, path)
            with self.assertRaises(NotImplementedError):
                OnnxRuntime(path, device="cpu")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# External policy tests (skipped if file not present)
# ---------------------------------------------------------------------------


class TestExternalPolicy(unittest.TestCase):
    """Tests against an external ONNX policy file."""

    def setUp(self):
        if not os.path.exists(POLICY_PATH):
            self.skipTest(f"ONNX policy not found at {POLICY_PATH}")

    def _run_on_device(self, device_str: str):
        rt = OnnxRuntime(POLICY_PATH, device=device_str)

        self.assertEqual(rt.input_names, ["observation"])
        self.assertEqual(rt.output_names, ["action"])

        rng = np.random.default_rng(42)
        obs = rng.standard_normal((1, 48)).astype(np.float32)
        obs_wp = wp.array(obs, dtype=wp.float32, device=device_str)

        result = rt({"observation": obs_wp})
        action = result["action"]

        self.assertEqual(action.shape, (1, 12))
        action_np = action.numpy()

        expected = np.array(
            [[
                0.47301683, 1.0230064, 0.6639304, 1.1513045,
                0.73161083, -1.572881, 0.6339237, -0.7481932,
                -0.22896627, 1.26267, -0.9770715, -2.3447196,
            ]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(action_np, expected, atol=1e-4)

    def test_cpu(self):
        self._run_on_device("cpu")

    def test_cuda(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")
        self._run_on_device("cuda:0")

    def test_deterministic(self):
        rt = OnnxRuntime(POLICY_PATH, device="cpu")
        obs = np.ones((1, 48), dtype=np.float32) * 0.5
        obs_wp = wp.array(obs, dtype=wp.float32, device="cpu")

        out1 = rt({"observation": obs_wp})["action"].numpy()
        out2 = rt({"observation": obs_wp})["action"].numpy()
        np.testing.assert_array_equal(out1, out2)

    def test_batch(self):
        rt = OnnxRuntime(POLICY_PATH, device="cpu", batch_size=4)
        rng = np.random.default_rng(123)
        obs = rng.standard_normal((4, 48)).astype(np.float32)
        obs_wp = wp.array(obs, dtype=wp.float32, device="cpu")

        result = rt({"observation": obs_wp})
        action = result["action"]
        self.assertEqual(action.shape, (4, 12))

        rt_single = OnnxRuntime(POLICY_PATH, device="cpu", batch_size=1)
        for i in range(4):
            single_obs = wp.array(obs[i:i + 1], dtype=wp.float32, device="cpu")
            single_out = rt_single({"observation": single_obs})["action"].numpy()
            np.testing.assert_allclose(action.numpy()[i:i + 1], single_out, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
