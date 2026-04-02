# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lightweight neural network building blocks backed by Warp kernels.

Provides a :class:`WarpMLP` that uses tiled GEMM for forward passes and
``wp.Tape`` for automatic differentiation.  Networks implement the
:class:`Network` protocol so they can be used interchangeably with
:class:`~newton._src.ppo.ActorCritic`.

ONNX export via :func:`export_to_onnx` produces models loadable by
:class:`~newton._src.onnx_runtime.OnnxRuntime`.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
import onnx
import warp as wp
from onnx import TensorProto, helper, numpy_helper

from newton._src.onnx_runtime import (
    _TILE_THREADS,
    _bias_add_kernel,
    _ceil_div,
    _elu_kernel,
    _make_tiled_gemm_transB_kernel,
    _pick_tile_sizes,
    _relu_kernel,
    _tanh_kernel,
)

# ---------------------------------------------------------------------------
# Network protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Network(Protocol):
    """Minimal interface that :class:`~newton._src.ppo.ActorCritic` expects
    from actor and critic networks."""

    def forward(self, x: wp.array) -> wp.array:
        """Run the forward pass.  Returns a 2-D ``(batch, out_dim)`` array."""
        ...

    def parameters(self) -> list[wp.array]:
        """Flat list of trainable parameter arrays (1-D views)."""
        ...

    def grad_arrays(self) -> list[wp.array]:
        """Flat list of gradient arrays matching :meth:`parameters`."""
        ...

    def alloc_intermediates(self, batch: int) -> None:
        """Pre-allocate internal buffers for a fixed batch size."""
        ...


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

_ACTIVATION_KERNELS = {
    "elu": _elu_kernel,
    "relu": _relu_kernel,
    "tanh": _tanh_kernel,
}


@wp.kernel
def _layer_norm_kernel(
    x: wp.array2d[float],
    gamma: wp.array[float],
    beta: wp.array[float],
    out: wp.array2d[float],
    width: int,
):
    """Per-row layer normalization: out[i] = gamma * (x[i] - mean) / std + beta."""
    i = wp.tid()
    mu = float(0.0)
    for j in range(width):
        mu = mu + x[i, j]
    mu = mu / float(width)
    var = float(0.0)
    for j in range(width):
        d = x[i, j] - mu
        var = var + d * d
    var = var / float(width)
    inv_std = 1.0 / wp.sqrt(var + 1.0e-5)
    for j in range(width):
        out[i, j] = gamma[j] * (x[i, j] - mu) * inv_std + beta[j]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orthogonal_init(shape: tuple[int, int], gain: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Orthogonal weight initialization (standard for PPO)."""
    rows, cols = shape
    rng = np.random.default_rng(seed)
    n = max(rows, cols)
    flat = rng.standard_normal((n, n)).astype(np.float32)
    q, r = np.linalg.qr(flat)
    q *= np.sign(np.diag(r))
    return q[:rows, :cols] * gain


# ---------------------------------------------------------------------------
# WarpMLP
# ---------------------------------------------------------------------------


class WarpMLP:
    """Dense MLP with tiled GEMM forward pass and ``wp.Tape`` autodiff.

    Weights are stored in ``(out_dim, in_dim)`` layout (transB=1 convention)
    so the same tiled kernels used by :class:`OnnxRuntime` are reused here.

    Args:
        layer_sizes: Sequence of layer widths, e.g. ``[48, 128, 128, 128, 12]``.
        activation: Activation function name (``"elu"``, ``"relu"``, ``"tanh"``).
        layer_norm: Apply layer normalization after each hidden layer
            (before the activation).  Stabilises training and reduces
            sensitivity to input scale.
        device: Warp device string.
        output_gain: Orthogonal init gain for the last layer (0.01 is common
            for policy heads).
        seed: RNG seed for weight initialization reproducibility.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "elu",
        layer_norm: bool = True,
        device: str | None = None,
        output_gain: float = 1.0,
        seed: int | None = None,
    ):
        self._device = wp.get_device(device)
        self._activation = activation
        self._layer_norm = layer_norm
        self._act_kernel = _ACTIVATION_KERNELS.get(activation)
        if self._act_kernel is None:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(_ACTIVATION_KERNELS)}")

        self.weights: list[wp.array] = []
        self.biases: list[wp.array] = []
        self.ln_gammas: list[wp.array] = []
        self.ln_betas: list[wp.array] = []
        self._intermediates: list[wp.array] = []

        num_layers = len(layer_sizes) - 1
        for i in range(num_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            gain = output_gain if i == num_layers - 1 else math.sqrt(2.0)
            layer_seed = None if seed is None else seed + i
            w_np = _orthogonal_init((fan_out, fan_in), gain=gain, seed=layer_seed)
            b_np = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(wp.array(w_np, dtype=wp.float32, device=self._device, requires_grad=True))
            self.biases.append(wp.array(b_np, dtype=wp.float32, device=self._device, requires_grad=True))
            if layer_norm and i < num_layers - 1:
                self.ln_gammas.append(
                    wp.array(np.ones(fan_out, dtype=np.float32), dtype=wp.float32, device=self._device, requires_grad=True)
                )
                self.ln_betas.append(
                    wp.array(np.zeros(fan_out, dtype=np.float32), dtype=wp.float32, device=self._device, requires_grad=True)
                )

        self._layer_sizes = layer_sizes

    def alloc_intermediates(self, batch: int) -> None:
        """Pre-allocate intermediate buffers for a fixed batch size."""
        needed = len(self.weights)
        if len(self._intermediates) == needed and self._intermediates[0].shape[0] == batch:
            return
        self._intermediates = [
            wp.zeros((batch, self._layer_sizes[i + 1]), dtype=wp.float32, device=self._device, requires_grad=True)
            for i in range(needed)
        ]

    def forward(self, x: wp.array) -> wp.array:
        batch = x.shape[0]
        if not self._intermediates or self._intermediates[0].shape[0] != batch:
            self.alloc_intermediates(batch)

        inp = x
        ln_idx = 0
        for i, (w, b, out) in enumerate(zip(self.weights, self.biases, self._intermediates, strict=True)):
            M, K = batch, self._layer_sizes[i]
            N = self._layer_sizes[i + 1]
            tile_m, tile_n, tile_k = _pick_tile_sizes(M, N, K)
            grid = (_ceil_div(M, tile_m), _ceil_div(N, tile_n))
            kernel = _make_tiled_gemm_transB_kernel(tile_m, tile_n, tile_k)

            out.zero_()
            wp.launch_tiled(kernel, dim=list(grid), inputs=[inp, w, out], block_dim=_TILE_THREADS, device=self._device)
            wp.launch(_bias_add_kernel, dim=(M, N), inputs=[out, b, 1.0, 1.0], device=self._device)

            is_last = i == len(self.weights) - 1
            if not is_last:
                if self._layer_norm:
                    wp.launch(
                        _layer_norm_kernel, dim=M,
                        inputs=[out, self.ln_gammas[ln_idx], self.ln_betas[ln_idx], out, N],
                        device=self._device,
                    )
                    ln_idx += 1
                if self._activation == "elu":
                    wp.launch(self._act_kernel, dim=(M, N), inputs=[out, out, 1.0], device=self._device)
                else:
                    wp.launch(self._act_kernel, dim=(M, N), inputs=[out, out], device=self._device)

            inp = out

        return self._intermediates[-1]

    def parameters(self) -> list[wp.array]:
        """Flat list of all trainable parameter arrays."""
        params = []
        ln_idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            params.append(w.flatten())
            params.append(b)
            if self._layer_norm and i < len(self.weights) - 1:
                params.append(self.ln_gammas[ln_idx])
                params.append(self.ln_betas[ln_idx])
                ln_idx += 1
        return params

    def grad_arrays(self) -> list[wp.array]:
        """Flat list of gradient arrays matching :meth:`parameters`."""
        grads = []
        ln_idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            grads.append(w.grad.flatten())
            grads.append(b.grad)
            if self._layer_norm and i < len(self.weights) - 1:
                grads.append(self.ln_gammas[ln_idx].grad)
                grads.append(self.ln_betas[ln_idx].grad)
                ln_idx += 1
        return grads


def clone_for_batch(source: WarpMLP, batch: int) -> WarpMLP:
    """Create a WarpMLP that shares weights/LN params but has its own intermediates."""
    clone = WarpMLP.__new__(WarpMLP)
    clone.__dict__.update(source.__dict__)
    clone._intermediates = []
    clone.alloc_intermediates(batch)
    clone.weights = source.weights
    clone.biases = source.biases
    clone.ln_gammas = source.ln_gammas
    clone.ln_betas = source.ln_betas
    return clone


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_to_onnx(actor: WarpMLP, obs_dim: int, path: str) -> None:
    """Export a trained :class:`WarpMLP` actor to ONNX format.

    The exported model can be loaded by :class:`newton._src.onnx_runtime.OnnxRuntime`.

    Args:
        actor: The trained actor MLP.
        obs_dim: Observation dimension (input width).
        path: Output ``.onnx`` file path.
    """
    nodes: list[Any] = []
    initializers = []
    prev_output = "observation"

    activation_map = {"elu": "Elu", "relu": "Relu", "tanh": "Tanh"}
    onnx_act = activation_map.get(actor._activation, "Elu")

    ln_idx = 0
    for i, (w, b) in enumerate(zip(actor.weights, actor.biases, strict=True)):
        w_name = f"actor.{i * 2}.weight"
        b_name = f"actor.{i * 2}.bias"
        gemm_out = f"/actor/{i * 2}/Gemm_output_0"

        initializers.append(numpy_helper.from_array(w.numpy(), name=w_name))
        initializers.append(numpy_helper.from_array(b.numpy(), name=b_name))
        nodes.append(helper.make_node("Gemm", [prev_output, w_name, b_name], [gemm_out], alpha=1.0, beta=1.0, transB=1))
        prev_output = gemm_out

        is_last = i == len(actor.weights) - 1
        if not is_last:
            if actor._layer_norm and ln_idx < len(actor.ln_gammas):
                gamma_name = f"actor.ln{ln_idx}.gamma"
                beta_name = f"actor.ln{ln_idx}.beta"
                ln_out = f"/actor/ln{ln_idx}/output"
                initializers.append(numpy_helper.from_array(actor.ln_gammas[ln_idx].numpy(), name=gamma_name))
                initializers.append(numpy_helper.from_array(actor.ln_betas[ln_idx].numpy(), name=beta_name))
                nodes.append(
                    helper.make_node("LayerNormalization", [prev_output, gamma_name, beta_name], [ln_out], epsilon=1e-5)
                )
                prev_output = ln_out
                ln_idx += 1
            act_out = f"/actor/{i * 2 + 1}/{onnx_act}_output_0"
            nodes.append(helper.make_node(onnx_act, [prev_output], [act_out]))
            prev_output = act_out

    nodes[-1].output[0] = "action"

    graph = helper.make_graph(
        nodes,
        "actor",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch_size", obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch_size", actor._layer_sizes[-1]])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, path)
