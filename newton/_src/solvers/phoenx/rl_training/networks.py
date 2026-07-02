# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
import warp as wp

from .kernels import (
    ACTIVATION_ELU,
    ACTIVATION_LINEAR,
    ACTIVATION_RELU,
    ACTIVATION_TANH,
    DENSE_BIAS_TILE_BATCH,
    DENSE_TILE_BATCH,
    DENSE_TILE_BLOCK_DIM,
    DENSE_TILE_IN,
    DENSE_TILE_OUT,
    DENSE_WEIGHT_GRAD_KCHUNKS,
    add_2d_kernel,
    copy_1d_kernel,
    copy_2d_kernel,
    dense_activation_grad_kernel,
    dense_bias_partial_grad_kernel,
    dense_bias_reduce_grad_kernel,
    dense_forward_tiled_kernel,
    dense_input_grad_kernel,
    dense_input_grad_tiled_kernel,
    dense_layer_kernel,
    dense_weight_bias_grad_kernel,
    dense_weight_grad_reduce_kernel,
    dense_weight_grad_splitk_tiled_kernel,
    fill_eps_kernel,
    fill_eps_seed_counter_kernel,
    gaussian_entropy_kernel,
    gaussian_log_prob_kernel,
    mingru_sequence_backward_kernel,
    mingru_sequence_forward_kernel,
    mingru_step_kernel,
    reset_mingru_state_kernel,
    sample_gaussian_actions_kernel,
    soft_update_1d_kernel,
    soft_update_2d_kernel,
    zero_2d_tail_rows_kernel,
    zero_3d_kernel,
)
from .kernels_bf16 import (
    cast_2d_float_to_bfloat16_kernel,
    dense_bias_activation_kernel,
    dense_forward_bf16_tiled_kernel,
    dense_input_grad_bf16_tiled_kernel,
    dense_weight_grad_bf16_tiled_kernel,
)

_BF16_FORWARD_MIN_BATCH = 16_384
# Minimum batch for the f32 tiled forward. The naive forward re-reads the
# weight matrix per row, so tiling only pays once the batch is large (training
# updates), not for per-step rollout inference. Gated to tile-aligned shapes so
# the rounded-up tile loads/stores stay in bounds.
_F32_FORWARD_TILED_MIN_BATCH = 16_384


def activation_code(name: str) -> int:
    """Return the integer activation code used by Warp kernels."""

    key = name.lower()
    if key == "linear":
        return ACTIVATION_LINEAR
    if key == "tanh":
        return ACTIVATION_TANH
    if key == "relu":
        return ACTIVATION_RELU
    if key == "elu":
        return ACTIVATION_ELU
    raise ValueError(f"Unsupported activation {name!r}")


class WarpMLP:
    """Fully connected MLP backed by Warp arrays and kernels.

    Args:
        layer_sizes: Network widths, including input and output dimensions.
        activation: Hidden-layer activation name.
        output_activation: Output-layer activation name.
        device: Warp device.
        seed: Initializer seed.
        gain: Weight initializer gain.
        manual_weight_grad_dtype: Accumulator input dtype for manual CUDA
            backward tile matmuls. Supports ``"float32"`` and
            ``"bfloat16"``.
        manual_forward_dtype: Input dtype for manual CUDA hidden-layer forward
            tile matmul. Supports ``"float32"`` and ``"bfloat16"``.
    """

    class _ForwardScratch:
        def __init__(self):
            self.capacity = 0
            self.outputs: list[wp.array2d[wp.float32]] = []
            self.bf16_inputs: list[wp.array2d[wp.bfloat16]] = []
            self.bf16_weights: list[wp.array2d[wp.bfloat16]] = []

    def __init__(
        self,
        layer_sizes: Iterable[int],
        *,
        activation: str = "tanh",
        output_activation: str = "linear",
        device: wp.context.Devicelike = None,
        seed: int = 0,
        gain: float = 1.0,
        manual_weight_grad_dtype: str = "float32",
        manual_forward_dtype: str = "float32",
    ):
        sizes = [int(s) for s in layer_sizes]
        if len(sizes) < 2:
            raise ValueError("WarpMLP requires at least input and output sizes")
        if any(s <= 0 for s in sizes):
            raise ValueError(f"Layer sizes must be positive, got {sizes}")

        self.layer_sizes = sizes
        self.input_dim = sizes[0]
        self.output_dim = sizes[-1]
        self.device = wp.get_device(device)
        self.activation = activation_code(activation)
        self.output_activation = activation_code(output_activation)
        self.manual_weight_grad_dtype = _manual_bfloat16_dtype(manual_weight_grad_dtype, "manual_weight_grad_dtype")
        self.manual_forward_dtype = _manual_bfloat16_dtype(manual_forward_dtype, "manual_forward_dtype")
        self._manual_batch_size = 0
        self._manual_capacity = 0
        self._manual_input: wp.array2d[wp.float32] | None = None
        self._manual_outputs: list[wp.array2d[wp.float32]] = []
        self._manual_output_grads: list[wp.array2d[wp.float32]] = []
        self._manual_pre_grads: list[wp.array2d[wp.float32]] = []
        self._manual_bias_partials: list[wp.array2d[wp.float32]] = []
        self._manual_weight_grad_partials: list[wp.array3d[wp.float32]] = []
        self._manual_bf16_inputs: list[wp.array2d[wp.bfloat16]] = []
        self._manual_bf16_pre_grads: list[wp.array2d[wp.bfloat16]] = []
        self._manual_bf16_weights: list[wp.array2d[wp.bfloat16]] = []
        self._forward_scratch: dict[str, WarpMLP._ForwardScratch] = {}

        rng = np.random.default_rng(seed)
        self.weights: list[wp.array] = []
        self.biases: list[wp.array] = []
        for in_dim, out_dim in pairwise(sizes):
            # Xavier-style uniform init keeps early policy logits in a useful range.
            limit = gain * np.sqrt(6.0 / float(in_dim + out_dim))
            w_np = rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
            b_np = np.zeros(out_dim, dtype=np.float32)
            self.weights.append(wp.array(w_np, dtype=wp.float32, device=self.device, requires_grad=True))
            self.biases.append(wp.array(b_np, dtype=wp.float32, device=self.device, requires_grad=True))

    def parameters(self) -> list[wp.array]:
        """Return trainable parameter arrays."""

        params: list[wp.array] = []
        for weight, bias in zip(self.weights, self.biases, strict=True):
            params.append(weight)
            params.append(bias)
        return params

    def forward(self, x: wp.array, *, requires_grad: bool = True) -> wp.array:
        """Evaluate the MLP for a batch of observations.

        Args:
            x: Input array with shape ``[batch, input_dim]``.
            requires_grad: Whether intermediate activations should store gradients.

        Returns:
            Output array with shape ``[batch, output_dim]``.
        """

        if x.ndim != 2:
            raise ValueError(f"Expected a 2-D input array, got ndim={x.ndim}")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {int(x.shape[1])}")

        batch_size = int(x.shape[0])
        if not requires_grad and self.manual_forward_dtype == "bfloat16":
            self._ensure_manual_buffers(batch_size)
        y = x
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            out_dim = int(weight.shape[1])
            activation = self.output_activation if layer == len(self.weights) - 1 else self.activation
            out = wp.empty((batch_size, out_dim), dtype=wp.float32, device=self.device, requires_grad=requires_grad)
            if requires_grad:
                wp.launch(
                    dense_layer_kernel,
                    dim=(batch_size, out_dim),
                    inputs=[y, weight, bias, int(weight.shape[0]), activation],
                    outputs=[out],
                    device=self.device,
                )
            else:
                self._launch_forward_layer(
                    layer,
                    y,
                    weight,
                    bias,
                    activation,
                    out,
                    batch_size,
                    bf16_inputs=self._manual_bf16_inputs,
                    bf16_weights=self._manual_bf16_weights,
                )
            y = out
        return y

    def forward_reuse(self, x: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        """Evaluate the MLP into persistent no-grad buffers."""

        if x.ndim != 2:
            raise ValueError(f"Expected a 2-D input array, got ndim={x.ndim}")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {int(x.shape[1])}")

        batch_size = int(x.shape[0])
        scratch = self._ensure_forward_scratch(batch_size, "default")
        y = x
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            activation = self.output_activation if layer == len(self.weights) - 1 else self.activation
            out = scratch.outputs[layer]
            self._launch_forward_layer(
                layer,
                y,
                weight,
                bias,
                activation,
                out,
                batch_size,
                bf16_inputs=scratch.bf16_inputs,
                bf16_weights=scratch.bf16_weights,
            )
            y = out
        return y[:batch_size]

    def forward_manual(self, x: wp.array) -> wp.array:
        """Evaluate the MLP and retain activations for manual backpropagation."""

        if x.ndim != 2:
            raise ValueError(f"Expected a 2-D input array, got ndim={x.ndim}")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {int(x.shape[1])}")

        batch_size = int(x.shape[0])
        self._ensure_manual_buffers(batch_size)
        self._manual_input = x
        y = x
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            activation = self.output_activation if layer == len(self.weights) - 1 else self.activation
            out = self._manual_outputs[layer]
            self._launch_forward_layer(
                layer,
                y,
                weight,
                bias,
                activation,
                out,
                batch_size,
                bf16_inputs=self._manual_bf16_inputs,
                bf16_weights=self._manual_bf16_weights,
            )
            y = out
        return y

    def _launch_forward_layer(
        self,
        layer: int,
        x: wp.array2d[wp.float32],
        weight: wp.array2d[wp.float32],
        bias: wp.array[wp.float32],
        activation: int,
        out: wp.array2d[wp.float32],
        batch_size: int,
        *,
        bf16_inputs: list[wp.array2d[wp.bfloat16]] | None = None,
        bf16_weights: list[wp.array2d[wp.bfloat16]] | None = None,
    ) -> None:
        rows = int(batch_size)
        cols = int(out.shape[1])
        if self._uses_bf16_forward(x, weight, rows):
            if bf16_inputs is None or bf16_weights is None:
                raise RuntimeError("BF16 forward buffers were not initialized")
            x_bf16 = bf16_inputs[layer]
            weight_bf16 = bf16_weights[layer]
            wp.launch(
                cast_2d_float_to_bfloat16_kernel,
                dim=(rows, int(x.shape[1])),
                inputs=[x],
                outputs=[x_bf16],
                device=self.device,
            )
            wp.launch(
                cast_2d_float_to_bfloat16_kernel,
                dim=weight.shape,
                inputs=[weight],
                outputs=[weight_bf16],
                device=self.device,
            )
            wp.launch_tiled(
                dense_forward_bf16_tiled_kernel,
                dim=(
                    _ceil_div(rows, DENSE_TILE_BATCH),
                    _ceil_div(cols, DENSE_TILE_OUT),
                ),
                inputs=[x_bf16, weight_bf16, int(weight.shape[0])],
                outputs=[out],
                block_dim=DENSE_TILE_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                dense_bias_activation_kernel,
                dim=(rows, cols),
                inputs=[out, bias, activation],
                device=self.device,
            )
        elif self._uses_f32_tiled_forward(weight, rows):
            wp.launch_tiled(
                dense_forward_tiled_kernel,
                dim=(_ceil_div(rows, DENSE_TILE_BATCH), _ceil_div(cols, DENSE_TILE_OUT)),
                inputs=[x, weight, int(weight.shape[0])],
                outputs=[out],
                block_dim=DENSE_TILE_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                dense_bias_activation_kernel,
                dim=(rows, cols),
                inputs=[out, bias, activation],
                device=self.device,
            )
        else:
            wp.launch(
                dense_layer_kernel,
                dim=(rows, cols),
                inputs=[x, weight, bias, int(weight.shape[0]), activation],
                outputs=[out],
                device=self.device,
            )

    def _uses_bf16_forward(self, x: wp.array2d[wp.float32], weight: wp.array2d[wp.float32], batch_size: int) -> bool:
        return (
            self.device.is_cuda
            and self.manual_forward_dtype == "bfloat16"
            and int(batch_size) >= _BF16_FORWARD_MIN_BATCH
            and int(weight.shape[0]) >= DENSE_TILE_IN
            and int(weight.shape[1]) >= 64
        )

    def _uses_f32_tiled_forward(self, weight: wp.array2d[wp.float32], batch_size: int) -> bool:
        # Tile loads/stores read rounded-up tiles, so require tile-aligned
        # shapes (batch, in_dim, out_dim) to stay in bounds without padding.
        return (
            self.device.is_cuda
            and int(batch_size) >= _F32_FORWARD_TILED_MIN_BATCH
            and int(batch_size) % DENSE_TILE_BATCH == 0
            and int(weight.shape[0]) % DENSE_TILE_IN == 0
            and int(weight.shape[1]) % DENSE_TILE_OUT == 0
        )

    def backward_manual(self, output_grad: wp.array2d[wp.float32]) -> None:
        """Backpropagate from an output gradient into parameter gradients."""

        if self._manual_input is None or not self._manual_outputs:
            raise RuntimeError("forward_manual() must be called before backward_manual()")
        if int(output_grad.shape[0]) < self._manual_batch_size or int(output_grad.shape[1]) != self.output_dim:
            raise ValueError("Manual MLP output gradient shape does not match the last forward pass")

        rows = self._manual_batch_size
        grad_y = output_grad
        for layer in reversed(range(len(self.weights))):
            weight = self.weights[layer]
            bias = self.biases[layer]
            activation = self.output_activation if layer == len(self.weights) - 1 else self.activation
            grad_pre = self._manual_pre_grads[layer]
            width = int(grad_pre.shape[1])
            wp.launch(
                dense_activation_grad_kernel,
                dim=(rows, width),
                inputs=[self._manual_outputs[layer], grad_y, activation],
                outputs=[grad_pre],
                device=self.device,
            )
            tiled_rows = rows
            if self.device.is_cuda:
                # Tiled weight-grad kernels read a rounded-up tile; clear reserved inactive rows.
                tile_end = min(int(grad_pre.shape[0]), _ceil_div(rows, DENSE_TILE_BATCH) * DENSE_TILE_BATCH)
                if tile_end > rows:
                    wp.launch(
                        zero_2d_tail_rows_kernel,
                        dim=(tile_end - rows, width),
                        inputs=[rows],
                        outputs=[grad_pre],
                        device=self.device,
                    )
                    tiled_rows = tile_end
            x = self._manual_input if layer == 0 else self._manual_outputs[layer - 1]
            if self.device.is_cuda:
                if self.manual_weight_grad_dtype == "bfloat16":
                    x_bf16 = self._manual_bf16_inputs[layer]
                    grad_pre_bf16 = self._manual_bf16_pre_grads[layer]
                    wp.launch(
                        cast_2d_float_to_bfloat16_kernel,
                        dim=(rows, int(x.shape[1])),
                        inputs=[x],
                        outputs=[x_bf16],
                        device=self.device,
                    )
                    wp.launch(
                        cast_2d_float_to_bfloat16_kernel,
                        dim=(tiled_rows, width),
                        inputs=[grad_pre],
                        outputs=[grad_pre_bf16],
                        device=self.device,
                    )
                    wp.launch_tiled(
                        dense_weight_grad_bf16_tiled_kernel,
                        dim=(
                            _ceil_div(int(weight.shape[0]), DENSE_TILE_IN),
                            _ceil_div(int(weight.shape[1]), DENSE_TILE_OUT),
                        ),
                        inputs=[x_bf16, grad_pre_bf16, rows],
                        outputs=[weight.grad],
                        block_dim=DENSE_TILE_BLOCK_DIM,
                        device=self.device,
                    )
                else:
                    weight_grad_partials = self._manual_weight_grad_partials[layer]
                    wp.launch_tiled(
                        dense_weight_grad_splitk_tiled_kernel,
                        dim=(
                            _ceil_div(int(weight.shape[0]), DENSE_TILE_IN),
                            _ceil_div(int(weight.shape[1]), DENSE_TILE_OUT),
                            DENSE_WEIGHT_GRAD_KCHUNKS,
                        ),
                        inputs=[x, grad_pre, rows],
                        outputs=[weight_grad_partials],
                        block_dim=DENSE_TILE_BLOCK_DIM,
                        device=self.device,
                    )
                    wp.launch(
                        dense_weight_grad_reduce_kernel,
                        dim=(int(weight.shape[0]), int(weight.shape[1])),
                        inputs=[weight_grad_partials, DENSE_WEIGHT_GRAD_KCHUNKS],
                        outputs=[weight.grad],
                        device=self.device,
                    )
                bias_partial = self._manual_bias_partials[layer]
                bias_tile_count = _ceil_div(rows, DENSE_BIAS_TILE_BATCH)
                wp.launch(
                    dense_bias_partial_grad_kernel,
                    dim=(bias_tile_count, int(weight.shape[1])),
                    inputs=[grad_pre, rows],
                    outputs=[bias_partial],
                    device=self.device,
                )
                wp.launch(
                    dense_bias_reduce_grad_kernel,
                    dim=int(weight.shape[1]),
                    inputs=[bias_partial, bias_tile_count],
                    outputs=[bias.grad],
                    device=self.device,
                )
            else:
                wp.launch(
                    dense_weight_bias_grad_kernel,
                    dim=weight.shape,
                    inputs=[x, grad_pre, rows],
                    outputs=[weight.grad, bias.grad],
                    device=self.device,
                )
            if layer > 0:
                grad_y = self._manual_output_grads[layer - 1]
                if self.device.is_cuda:
                    if self.manual_weight_grad_dtype == "bfloat16":
                        grad_pre_bf16 = self._manual_bf16_pre_grads[layer]
                        weight_bf16 = self._manual_bf16_weights[layer]
                        wp.launch(
                            cast_2d_float_to_bfloat16_kernel,
                            dim=weight.shape,
                            inputs=[weight],
                            outputs=[weight_bf16],
                            device=self.device,
                        )
                        wp.launch_tiled(
                            dense_input_grad_bf16_tiled_kernel,
                            dim=(
                                _ceil_div(rows, DENSE_TILE_BATCH),
                                _ceil_div(int(grad_y.shape[1]), DENSE_TILE_IN),
                            ),
                            inputs=[grad_pre_bf16, weight_bf16, int(weight.shape[1])],
                            outputs=[grad_y],
                            block_dim=DENSE_TILE_BLOCK_DIM,
                            device=self.device,
                        )
                    else:
                        wp.launch_tiled(
                            dense_input_grad_tiled_kernel,
                            dim=(
                                _ceil_div(rows, DENSE_TILE_BATCH),
                                _ceil_div(int(grad_y.shape[1]), DENSE_TILE_IN),
                            ),
                            inputs=[grad_pre, weight, int(weight.shape[1])],
                            outputs=[grad_y],
                            block_dim=DENSE_TILE_BLOCK_DIM,
                            device=self.device,
                        )
                else:
                    wp.launch(
                        dense_input_grad_kernel,
                        dim=(rows, int(grad_y.shape[1])),
                        inputs=[grad_pre, weight, int(weight.shape[1])],
                        outputs=[grad_y],
                        device=self.device,
                    )

    def _ensure_manual_buffers(self, batch_size: int) -> None:
        requested = int(batch_size)
        if requested <= 0:
            raise ValueError("batch_size must be positive")
        self._manual_batch_size = requested
        if self._manual_capacity >= requested and len(self._manual_outputs) == len(self.weights):
            return
        self._manual_capacity = requested
        self._manual_outputs = []
        self._manual_output_grads = []
        self._manual_pre_grads = []
        self._manual_bias_partials = []
        self._manual_weight_grad_partials = []
        self._manual_bf16_inputs = []
        self._manual_bf16_pre_grads = []
        self._manual_bf16_weights = []
        bias_tile_count = _ceil_div(self._manual_capacity, DENSE_BIAS_TILE_BATCH)
        for in_dim, width in pairwise(self.layer_sizes):
            shape = (self._manual_capacity, int(width))
            self._manual_outputs.append(wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=False))
            self._manual_output_grads.append(wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=False))
            self._manual_pre_grads.append(wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=False))
            if self.device.is_cuda:
                self._manual_bias_partials.append(
                    wp.empty((bias_tile_count, int(width)), dtype=wp.float32, device=self.device, requires_grad=False)
                )
                # Split-K partials, padded to the tile grid so the tiled store stays in bounds.
                padded_in = _ceil_div(int(in_dim), DENSE_TILE_IN) * DENSE_TILE_IN
                padded_out = _ceil_div(int(width), DENSE_TILE_OUT) * DENSE_TILE_OUT
                self._manual_weight_grad_partials.append(
                    wp.empty(
                        (DENSE_WEIGHT_GRAD_KCHUNKS, padded_in, padded_out),
                        dtype=wp.float32,
                        device=self.device,
                        requires_grad=False,
                    )
                )
                if self.manual_weight_grad_dtype == "bfloat16" or self.manual_forward_dtype == "bfloat16":
                    self._manual_bf16_inputs.append(
                        wp.empty(
                            (self._manual_capacity, int(in_dim)),
                            dtype=wp.bfloat16,
                            device=self.device,
                            requires_grad=False,
                        )
                    )
                    if self.manual_weight_grad_dtype == "bfloat16":
                        self._manual_bf16_pre_grads.append(
                            wp.empty(shape, dtype=wp.bfloat16, device=self.device, requires_grad=False)
                        )
                    if self.manual_weight_grad_dtype == "bfloat16" or self.manual_forward_dtype == "bfloat16":
                        self._manual_bf16_weights.append(
                            wp.empty(
                                (int(in_dim), int(width)), dtype=wp.bfloat16, device=self.device, requires_grad=False
                            )
                        )

    def _ensure_forward_scratch(self, batch_size: int, name: str) -> _ForwardScratch:
        requested = int(batch_size)
        if requested <= 0:
            raise ValueError("batch_size must be positive")
        if not name:
            raise ValueError("forward scratch name must be non-empty")
        scratch = self._forward_scratch.get(name)
        if scratch is None:
            scratch = self._ForwardScratch()
            self._forward_scratch[name] = scratch
        if scratch.capacity >= requested and len(scratch.outputs) == len(self.weights):
            return scratch

        scratch.capacity = requested
        scratch.outputs = []
        scratch.bf16_inputs = []
        scratch.bf16_weights = []
        for in_dim, width in pairwise(self.layer_sizes):
            scratch.outputs.append(
                wp.empty((scratch.capacity, int(width)), dtype=wp.float32, device=self.device, requires_grad=False)
            )
            if self.device.is_cuda and self.manual_forward_dtype == "bfloat16":
                scratch.bf16_inputs.append(
                    wp.empty(
                        (scratch.capacity, int(in_dim)),
                        dtype=wp.bfloat16,
                        device=self.device,
                        requires_grad=False,
                    )
                )
                scratch.bf16_weights.append(
                    wp.empty((int(in_dim), int(width)), dtype=wp.bfloat16, device=self.device, requires_grad=False)
                )
        return scratch

    def reserve_forward_buffers(self, batch_size: int) -> None:
        """Reserve reusable no-grad forward buffers for at least ``batch_size`` rows."""

        self._ensure_forward_scratch(batch_size, "default")

    def reserve_buffers(self, batch_size: int) -> None:
        """Reserve no-grad and manual-backward buffers for at least ``batch_size`` rows."""

        self._ensure_manual_buffers(batch_size)
        self.reserve_forward_buffers(batch_size)
        self._manual_input = None

    def copy_from(self, other: WarpMLP) -> None:
        """Copy parameters from another network with the same architecture."""

        _check_same_shapes(self.parameters(), other.parameters())
        for src, dst in zip(other.weights, self.weights, strict=True):
            wp.launch(copy_2d_kernel, dim=dst.shape, inputs=[src], outputs=[dst], device=self.device)
        for src, dst in zip(other.biases, self.biases, strict=True):
            wp.launch(copy_1d_kernel, dim=dst.shape[0], inputs=[src], outputs=[dst], device=self.device)

    def soft_update_from(self, other: WarpMLP, tau: float) -> None:
        """Move this network toward another network in place."""

        _check_same_shapes(self.parameters(), other.parameters())
        for src, dst in zip(other.weights, self.weights, strict=True):
            wp.launch(soft_update_2d_kernel, dim=dst.shape, inputs=[src, tau], outputs=[dst], device=self.device)
        for src, dst in zip(other.biases, self.biases, strict=True):
            wp.launch(soft_update_1d_kernel, dim=dst.shape[0], inputs=[src, tau], outputs=[dst], device=self.device)


class PufferMinGRUNet:
    """PufferLib-style encoder, MinGRU stack, and fused decoder.

    The network follows the native Puffer learner layout used by nanoG1:
    a bias-free linear encoder, a stack of bias-free MinGRU projections, and a
    bias-free linear decoder whose final column is the value prediction when
    used by shared actor/value PPO.
    """

    network_type = "puffer_mingru"

    class _ForwardScratch:
        def __init__(self):
            self.capacity = 0
            self.encoder_out: wp.array2d[wp.float32] | None = None
            self.combined: list[wp.array2d[wp.float32]] = []
            self.outputs: list[wp.array2d[wp.float32]] = []
            self.recurrent: list[wp.array2d[wp.float32]] = []
            self.decoder_out: wp.array2d[wp.float32] | None = None

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 3,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ):
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        if self.input_dim <= 0 or self.hidden_size <= 0 or self.output_dim <= 0 or self.num_layers <= 0:
            raise ValueError("PufferMinGRUNet dimensions must be positive")
        self.layer_sizes = [self.input_dim, self.hidden_size, self.output_dim]
        self.device = wp.get_device(device)

        rng = np.random.default_rng(seed)
        self.encoder_weight = wp.array(
            _kaiming_uniform(rng, self.input_dim, self.hidden_size, np.sqrt(2.0)),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )
        self.decoder_weight = wp.array(
            _kaiming_uniform(rng, self.hidden_size, self.output_dim, 1.0),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )
        self.recurrent_weights: list[wp.array2d[wp.float32]] = []
        for _ in range(self.num_layers):
            self.recurrent_weights.append(
                wp.array(
                    _kaiming_uniform(rng, self.hidden_size, 3 * self.hidden_size, 1.0),
                    dtype=wp.float32,
                    device=self.device,
                    requires_grad=True,
                )
            )
        self._zero_hidden = wp.zeros(self.hidden_size, dtype=wp.float32, device=self.device)
        self._zero_combined = wp.zeros(3 * self.hidden_size, dtype=wp.float32, device=self.device)
        self._zero_output = wp.zeros(self.output_dim, dtype=wp.float32, device=self.device)
        self._forward_scratch: dict[str, PufferMinGRUNet._ForwardScratch] = {}
        self._state_capacity = 0
        self._state: wp.array3d[wp.float32] | None = None

        self._manual_capacity = 0
        self._manual_rows = 0
        self._manual_steps = 1
        self._manual_envs = 0
        self._manual_input: wp.array2d[wp.float32] | None = None
        self._manual_encoder_out: wp.array2d[wp.float32] | None = None
        self._manual_combined: list[wp.array2d[wp.float32]] = []
        self._manual_outputs: list[wp.array2d[wp.float32]] = []
        self._manual_recurrent: list[wp.array2d[wp.float32]] = []
        self._manual_decoder_out: wp.array2d[wp.float32] | None = None
        self._manual_decoder_input_grad: wp.array2d[wp.float32] | None = None
        self._manual_grad_combined: list[wp.array2d[wp.float32]] = []
        self._manual_grad_highway: list[wp.array2d[wp.float32]] = []
        self._manual_grad_projected: list[wp.array2d[wp.float32]] = []
        self._manual_grad_inputs: list[wp.array2d[wp.float32]] = []
        self._manual_encoder_weight_grad_partials: wp.array3d[wp.float32] | None = None
        self._manual_decoder_weight_grad_partials: wp.array3d[wp.float32] | None = None
        self._manual_recurrent_weight_grad_partials: list[wp.array3d[wp.float32]] = []

    def parameters(self) -> list[wp.array]:
        """Return trainable parameter arrays."""

        return [self.encoder_weight, self.decoder_weight, *self.recurrent_weights]

    def set_sequence_shape(self, num_steps: int, num_envs: int) -> None:
        """Set the flat-row layout used by the next manual training forward."""

        steps = int(num_steps)
        envs = int(num_envs)
        if steps <= 0 or envs <= 0:
            raise ValueError("sequence shape must be positive")
        self._manual_steps = steps
        self._manual_envs = envs

    def reserve_forward_buffers(self, batch_size: int) -> None:
        """Reserve reusable no-grad forward buffers."""

        self._ensure_forward_scratch(int(batch_size), "default")
        self._ensure_state(int(batch_size))

    def reserve_buffers(self, batch_size: int) -> None:
        """Reserve no-grad and manual-backward buffers."""

        rows = int(batch_size)
        self.reserve_forward_buffers(rows)
        self._ensure_manual_buffers(rows)
        self._ensure_forward_scratch(rows, "sequence")

    def zero_state(self) -> None:
        """Clear all persistent rollout recurrent state."""

        if self._state is None:
            return
        wp.launch(zero_3d_kernel, dim=self._state.shape, inputs=[self._state], device=self.device)

    def reset_state(self, dones: wp.array[wp.float32] | None = None) -> None:
        """Clear all state, or only state rows whose done flag is nonzero."""

        if self._state is None:
            return
        if dones is None:
            self.zero_state()
            return
        env_count = int(dones.shape[0])
        if env_count > int(self._state.shape[1]):
            raise ValueError("done array is larger than recurrent-state capacity")
        wp.launch(
            reset_mingru_state_kernel,
            dim=(self.num_layers, env_count, self.hidden_size),
            inputs=[dones, self._state],
            device=self.device,
        )

    def forward(self, x: wp.array2d[wp.float32], *, requires_grad: bool = True) -> wp.array2d[wp.float32]:
        """Evaluate the policy network."""

        if requires_grad:
            return self.forward_manual(x)
        return self.forward_reuse(x)

    def forward_reuse(self, x: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        """Evaluate and update persistent rollout state."""

        rows = self._check_input(x)
        scratch = self._ensure_forward_scratch(rows, "default")
        state = self._ensure_state(rows)
        return self._forward_rollout(x, rows, scratch, state)

    def forward_sequence_reuse(
        self, x: wp.array2d[wp.float32], *, num_steps: int, num_envs: int
    ) -> wp.array2d[wp.float32]:
        """Evaluate a flat ``[num_steps, num_envs]`` sequence from zero state."""

        rows = self._check_input(x)
        steps = int(num_steps)
        envs = int(num_envs)
        if rows != steps * envs:
            raise ValueError("sequence shape does not match input row count")
        scratch = self._ensure_forward_scratch(rows, "sequence")
        return self._forward_sequence(x, rows, steps, envs, scratch)

    def forward_manual(self, x: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        """Evaluate a sequence and retain activations for manual BPTT."""

        rows = self._check_input(x)
        steps = int(self._manual_steps)
        envs = int(self._manual_envs) if self._manual_envs > 0 else rows
        if rows != steps * envs:
            steps = 1
            envs = rows
        self._manual_rows = rows
        self._manual_steps = steps
        self._manual_envs = envs
        self._ensure_manual_buffers(rows)
        self._manual_input = x
        if self._manual_encoder_out is None or self._manual_decoder_out is None:
            raise RuntimeError("manual buffers were not initialized")
        self._linear_forward(x, self.encoder_weight, self._zero_hidden, rows, self._manual_encoder_out)
        h = self._manual_encoder_out
        for layer, weight in enumerate(self.recurrent_weights):
            combined = self._manual_combined[layer]
            out = self._manual_outputs[layer]
            self._linear_forward(h, weight, self._zero_combined, rows, combined)
            wp.launch(
                mingru_sequence_forward_kernel,
                dim=(envs, self.hidden_size),
                inputs=[combined, h, steps, envs, self.hidden_size],
                outputs=[out, self._manual_recurrent[layer]],
                device=self.device,
            )
            h = out
        self._linear_forward(h, self.decoder_weight, self._zero_output, rows, self._manual_decoder_out)
        return self._manual_decoder_out

    def backward_manual(self, output_grad: wp.array2d[wp.float32]) -> None:
        """Backpropagate through decoder, MinGRU sequence, and encoder."""

        if self._manual_input is None or self._manual_encoder_out is None or self._manual_decoder_out is None:
            raise RuntimeError("forward_manual() must be called before backward_manual()")
        rows = int(self._manual_rows)
        steps = int(self._manual_steps)
        envs = int(self._manual_envs)
        if int(output_grad.shape[0]) < rows or int(output_grad.shape[1]) != self.output_dim:
            raise ValueError("manual output gradient shape does not match the last forward pass")
        decoder_input = self._manual_outputs[-1]
        if self._manual_decoder_input_grad is None:
            raise RuntimeError("manual decoder gradient buffer was not initialized")
        self._zero_tail(output_grad, rows, self.output_dim)
        self._weight_grad(
            decoder_input,
            output_grad,
            rows,
            self.decoder_weight.grad,
            self._manual_decoder_weight_grad_partials,
        )
        self._input_grad(output_grad, self.decoder_weight, rows, self.output_dim, self._manual_decoder_input_grad)
        grad_h = self._manual_decoder_input_grad
        self._zero_tail(grad_h, rows, self.hidden_size)

        for layer in reversed(range(self.num_layers)):
            x_in = self._manual_encoder_out if layer == 0 else self._manual_outputs[layer - 1]
            grad_combined = self._manual_grad_combined[layer]
            grad_highway = self._manual_grad_highway[layer]
            grad_projected = self._manual_grad_projected[layer]
            grad_x = self._manual_grad_inputs[layer]
            wp.launch(
                mingru_sequence_backward_kernel,
                dim=(envs, self.hidden_size),
                inputs=[
                    self._manual_combined[layer],
                    x_in,
                    self._manual_recurrent[layer],
                    grad_h,
                    steps,
                    envs,
                    self.hidden_size,
                ],
                outputs=[grad_combined, grad_highway],
                device=self.device,
            )
            self._zero_tail(grad_combined, rows, 3 * self.hidden_size)
            self._weight_grad(
                x_in,
                grad_combined,
                rows,
                self.recurrent_weights[layer].grad,
                self._manual_recurrent_weight_grad_partials[layer],
            )
            self._input_grad(
                grad_combined,
                self.recurrent_weights[layer],
                rows,
                3 * self.hidden_size,
                grad_projected,
            )
            wp.launch(
                add_2d_kernel,
                dim=(rows, self.hidden_size),
                inputs=[grad_projected, grad_highway, rows * self.hidden_size],
                outputs=[grad_x],
                device=self.device,
            )
            grad_h = grad_x
            self._zero_tail(grad_h, rows, self.hidden_size)

        self._weight_grad(
            self._manual_input,
            grad_h,
            rows,
            self.encoder_weight.grad,
            self._manual_encoder_weight_grad_partials,
        )

    def copy_from(self, other: PufferMinGRUNet) -> None:
        """Copy parameters from another MinGRU network."""

        _check_same_shapes(self.parameters(), other.parameters())
        for src, dst in zip(other.parameters(), self.parameters(), strict=True):
            wp.launch(copy_2d_kernel, dim=dst.shape, inputs=[src], outputs=[dst], device=self.device)

    def _linear_forward(
        self,
        x: wp.array2d[wp.float32],
        weight: wp.array2d[wp.float32],
        zero_bias: wp.array[wp.float32],
        rows: int,
        out: wp.array2d[wp.float32],
    ) -> None:
        in_dim = int(weight.shape[0])
        out_dim = int(weight.shape[1])
        if (
            self.device.is_cuda
            and rows >= DENSE_TILE_BATCH
            and rows % DENSE_TILE_BATCH == 0
            and in_dim % DENSE_TILE_IN == 0
            and out_dim % DENSE_TILE_OUT == 0
        ):
            wp.launch_tiled(
                dense_forward_tiled_kernel,
                dim=(rows // DENSE_TILE_BATCH, out_dim // DENSE_TILE_OUT),
                inputs=[x, weight, in_dim],
                outputs=[out],
                block_dim=DENSE_TILE_BLOCK_DIM,
                device=self.device,
            )
        else:
            wp.launch(
                dense_layer_kernel,
                dim=(rows, out_dim),
                inputs=[x, weight, zero_bias, in_dim, ACTIVATION_LINEAR],
                outputs=[out],
                device=self.device,
            )

    def _check_input(self, x: wp.array2d[wp.float32]) -> int:
        if x.ndim != 2:
            raise ValueError(f"Expected a 2-D input array, got ndim={x.ndim}")
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {int(x.shape[1])}")
        rows = int(x.shape[0])
        if rows <= 0:
            raise ValueError("batch size must be positive")
        return rows

    def _ensure_state(self, batch_size: int) -> wp.array3d[wp.float32]:
        rows = int(batch_size)
        if self._state is None or self._state_capacity < rows:
            self._state_capacity = rows
            self._state = wp.zeros(
                (self.num_layers, self._state_capacity, self.hidden_size),
                dtype=wp.float32,
                device=self.device,
                requires_grad=False,
            )
        return self._state

    def _ensure_forward_scratch(self, batch_size: int, name: str) -> _ForwardScratch:
        rows = int(batch_size)
        scratch = self._forward_scratch.get(name)
        if scratch is None:
            scratch = self._ForwardScratch()
            self._forward_scratch[name] = scratch
        if scratch.capacity >= rows and scratch.encoder_out is not None and scratch.decoder_out is not None:
            return scratch
        scratch.capacity = rows
        scratch.encoder_out = wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device)
        scratch.combined = [
            wp.empty((rows, 3 * self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        scratch.outputs = [
            wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        if name == "sequence":
            if self._manual_capacity >= rows:
                scratch.recurrent = self._manual_recurrent
            else:
                scratch.recurrent = [
                    wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device)
                    for _ in range(self.num_layers)
                ]
        else:
            scratch.recurrent = []
        scratch.decoder_out = wp.empty((rows, self.output_dim), dtype=wp.float32, device=self.device)
        return scratch

    def _ensure_manual_buffers(self, batch_size: int) -> None:
        rows = int(batch_size)
        if rows <= 0:
            raise ValueError("batch size must be positive")
        if self._manual_capacity >= rows and self._manual_encoder_out is not None:
            return
        self._manual_capacity = rows
        self._manual_encoder_out = wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device)
        self._manual_combined = [
            wp.empty((rows, 3 * self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        self._manual_outputs = [
            wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        self._manual_recurrent = [
            wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        self._manual_decoder_out = wp.empty((rows, self.output_dim), dtype=wp.float32, device=self.device)
        self._manual_decoder_input_grad = wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device)
        self._manual_grad_combined = [
            wp.empty((rows, 3 * self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        self._manual_grad_highway = [
            wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        self._manual_grad_projected = [
            wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]
        self._manual_grad_inputs = [
            wp.empty((rows, self.hidden_size), dtype=wp.float32, device=self.device) for _ in range(self.num_layers)
        ]

        def make_weight_grad_partials(weight: wp.array2d[wp.float32]) -> wp.array3d[wp.float32]:
            padded_in = _ceil_div(int(weight.shape[0]), DENSE_TILE_IN) * DENSE_TILE_IN
            padded_out = _ceil_div(int(weight.shape[1]), DENSE_TILE_OUT) * DENSE_TILE_OUT
            return wp.empty(
                (DENSE_WEIGHT_GRAD_KCHUNKS, padded_in, padded_out),
                dtype=wp.float32,
                device=self.device,
            )

        self._manual_encoder_weight_grad_partials = make_weight_grad_partials(self.encoder_weight)
        self._manual_decoder_weight_grad_partials = make_weight_grad_partials(self.decoder_weight)
        self._manual_recurrent_weight_grad_partials = [
            make_weight_grad_partials(weight) for weight in self.recurrent_weights
        ]

    def _forward_rollout(
        self,
        x: wp.array2d[wp.float32],
        rows: int,
        scratch: _ForwardScratch,
        state: wp.array3d[wp.float32],
    ) -> wp.array2d[wp.float32]:
        if scratch.encoder_out is None or scratch.decoder_out is None:
            raise RuntimeError("forward scratch was not initialized")
        self._linear_forward(x, self.encoder_weight, self._zero_hidden, rows, scratch.encoder_out)
        h = scratch.encoder_out
        for layer, weight in enumerate(self.recurrent_weights):
            combined = scratch.combined[layer]
            out = scratch.outputs[layer]
            self._linear_forward(h, weight, self._zero_combined, rows, combined)
            wp.launch(
                mingru_step_kernel,
                dim=(rows, self.hidden_size),
                inputs=[combined, h, state, layer, self.hidden_size],
                outputs=[out],
                device=self.device,
            )
            h = out
        self._linear_forward(h, self.decoder_weight, self._zero_output, rows, scratch.decoder_out)
        return scratch.decoder_out[:rows]

    def _forward_sequence(
        self,
        x: wp.array2d[wp.float32],
        rows: int,
        steps: int,
        envs: int,
        scratch: _ForwardScratch,
    ) -> wp.array2d[wp.float32]:
        if scratch.encoder_out is None or scratch.decoder_out is None:
            raise RuntimeError("forward scratch was not initialized")
        self._linear_forward(x, self.encoder_weight, self._zero_hidden, rows, scratch.encoder_out)
        h = scratch.encoder_out
        for layer, weight in enumerate(self.recurrent_weights):
            combined = scratch.combined[layer]
            out = scratch.outputs[layer]
            self._linear_forward(h, weight, self._zero_combined, rows, combined)
            wp.launch(
                mingru_sequence_forward_kernel,
                dim=(envs, self.hidden_size),
                inputs=[combined, h, steps, envs, self.hidden_size],
                outputs=[out, scratch.recurrent[layer]],
                device=self.device,
            )
            h = out
        self._linear_forward(h, self.decoder_weight, self._zero_output, rows, scratch.decoder_out)
        return scratch.decoder_out[:rows]

    def _zero_tail(self, x: wp.array2d[wp.float32], rows: int, width: int) -> None:
        if not self.device.is_cuda:
            return
        tile_end = min(int(x.shape[0]), _ceil_div(rows, DENSE_TILE_BATCH) * DENSE_TILE_BATCH)
        if tile_end > rows:
            wp.launch(
                zero_2d_tail_rows_kernel,
                dim=(tile_end - rows, width),
                inputs=[rows],
                outputs=[x],
                device=self.device,
            )

    def _weight_grad(
        self,
        x: wp.array2d[wp.float32],
        grad_pre: wp.array2d[wp.float32],
        rows: int,
        weight_grad: wp.array2d[wp.float32],
        partials: wp.array3d[wp.float32] | None,
    ) -> None:
        if self.device.is_cuda:
            if partials is None:
                raise RuntimeError("manual MinGRU weight-gradient partials were not initialized")
            wp.launch_tiled(
                dense_weight_grad_splitk_tiled_kernel,
                dim=(
                    _ceil_div(int(weight_grad.shape[0]), DENSE_TILE_IN),
                    _ceil_div(int(weight_grad.shape[1]), DENSE_TILE_OUT),
                    DENSE_WEIGHT_GRAD_KCHUNKS,
                ),
                inputs=[x, grad_pre, rows],
                outputs=[partials],
                block_dim=DENSE_TILE_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                dense_weight_grad_reduce_kernel,
                dim=weight_grad.shape,
                inputs=[partials, DENSE_WEIGHT_GRAD_KCHUNKS],
                outputs=[weight_grad],
                device=self.device,
            )
        else:
            bias_grad = wp.empty(int(weight_grad.shape[1]), dtype=wp.float32, device=self.device)
            wp.launch(
                dense_weight_bias_grad_kernel,
                dim=weight_grad.shape,
                inputs=[x, grad_pre, rows],
                outputs=[weight_grad, bias_grad],
                device=self.device,
            )

    def _input_grad(
        self,
        grad_pre: wp.array2d[wp.float32],
        weight: wp.array2d[wp.float32],
        rows: int,
        out_dim: int,
        grad_x: wp.array2d[wp.float32],
    ) -> None:
        if self.device.is_cuda:
            wp.launch_tiled(
                dense_input_grad_tiled_kernel,
                dim=(_ceil_div(rows, DENSE_TILE_BATCH), _ceil_div(int(grad_x.shape[1]), DENSE_TILE_IN)),
                inputs=[grad_pre, weight, out_dim],
                outputs=[grad_x],
                block_dim=DENSE_TILE_BLOCK_DIM,
                device=self.device,
            )
        else:
            wp.launch(
                dense_input_grad_kernel,
                dim=(rows, int(grad_x.shape[1])),
                inputs=[grad_pre, weight, out_dim],
                outputs=[grad_x],
                device=self.device,
            )


class GaussianActor:
    """Gaussian policy with optional tanh squashing.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_layers: Hidden layer widths.
        activation: Hidden-layer activation.
        state_dependent_std: If ``True``, the network outputs mean and
            log-standard-deviation per state. Otherwise log-standard-deviation
            is one trainable vector shared by all states.
        log_std_init: Initial log-standard-deviation when using shared std.
        log_std_bounds: Clamp bounds for log-standard-deviation.
        squash: Whether to tanh-squash actions into ``[-1, 1]``.
        device: Warp device.
        seed: Initializer seed.
        manual_weight_grad_dtype: Accumulator input dtype for manual CUDA
            weight-gradient tile matmul.
        manual_forward_dtype: Input dtype for manual CUDA hidden-layer forward
            tile matmul.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        state_dependent_std: bool = False,
        log_std_init: float = -0.5,
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
        squash: bool = True,
        device: wp.context.Devicelike = None,
        seed: int = 0,
        manual_weight_grad_dtype: str = "float32",
        manual_forward_dtype: str = "float32",
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.state_dependent_std = bool(state_dependent_std)
        self.squash = bool(squash)
        self.log_std_min = float(log_std_bounds[0])
        self.log_std_max = float(log_std_bounds[1])
        self.device = wp.get_device(device)

        out_dim = self.action_dim * 2 if self.state_dependent_std else self.action_dim
        self.net = WarpMLP(
            (self.obs_dim, *hidden_layers, out_dim),
            activation=activation,
            output_activation="linear",
            device=self.device,
            seed=seed,
            manual_weight_grad_dtype=manual_weight_grad_dtype,
            manual_forward_dtype=manual_forward_dtype,
        )
        log_std_np = np.full(self.action_dim, float(log_std_init), dtype=np.float32)
        self.log_std = wp.array(
            log_std_np,
            dtype=wp.float32,
            device=self.device,
            requires_grad=not self.state_dependent_std,
        )
        self._sample_reuse_capacity = 0
        self._sample_reuse_actions: wp.array2d[wp.float32] | None = None
        self._sample_reuse_log_probs: wp.array[wp.float32] | None = None
        self._sample_reuse_eps: wp.array2d[wp.float32] | None = None

    def parameters(self) -> list[wp.array]:
        """Return trainable parameter arrays."""

        params = self.net.parameters()
        if not self.state_dependent_std:
            params.append(self.log_std)
        return params

    def forward(self, obs: wp.array, *, requires_grad: bool = True) -> wp.array:
        """Return raw policy outputs for ``obs``."""

        return self.net.forward(obs, requires_grad=requires_grad)

    def log_prob(self, obs: wp.array, actions: wp.array, *, requires_grad: bool = True) -> tuple[wp.array, wp.array]:
        """Compute action log-probabilities under the current policy.

        Args:
            obs: Observation batch.
            actions: Action batch.
            requires_grad: Whether outputs should record gradients.

        Returns:
            Tuple ``(policy_out, log_probs)``.
        """

        policy_out = self.forward(obs, requires_grad=requires_grad)
        log_probs = wp.empty(obs.shape[0], dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        wp.launch(
            gaussian_log_prob_kernel,
            dim=obs.shape[0],
            inputs=[
                policy_out,
                self.log_std,
                actions,
                self.action_dim,
                int(self.state_dependent_std),
                int(self.squash),
                self.log_std_min,
                self.log_std_max,
            ],
            outputs=[log_probs],
            device=self.device,
        )
        return policy_out, log_probs

    def entropy(self, policy_out: wp.array, *, requires_grad: bool = True) -> wp.array:
        """Return Gaussian entropy before optional tanh squashing."""

        entropy = wp.empty(policy_out.shape[0], dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        wp.launch(
            gaussian_entropy_kernel,
            dim=policy_out.shape[0],
            inputs=[
                policy_out,
                self.log_std,
                self.action_dim,
                int(self.state_dependent_std),
                self.log_std_min,
                self.log_std_max,
            ],
            outputs=[entropy],
            device=self.device,
        )
        return entropy

    def sample(
        self,
        obs: wp.array,
        *,
        seed: int,
        deterministic: bool = False,
        requires_grad: bool = True,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Sample actions with fixed Warp RNG seed.

        Args:
            obs: Observation batch.
            seed: Warp RNG seed.
            deterministic: If ``True``, return policy means.
            requires_grad: Whether outputs should record gradients.

        Returns:
            Tuple ``(actions, log_probs, policy_out)``.
        """

        batch_size = int(obs.shape[0])
        policy_out = self.forward(obs, requires_grad=requires_grad)
        actions = wp.empty(
            (batch_size, self.action_dim), dtype=wp.float32, device=self.device, requires_grad=requires_grad
        )
        log_probs = wp.empty(batch_size, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        eps = wp.empty((batch_size, self.action_dim), dtype=wp.float32, device=self.device, requires_grad=False)
        if not deterministic:
            wp.launch(
                fill_eps_kernel,
                dim=(batch_size, self.action_dim),
                inputs=[int(seed)],
                outputs=[eps],
                device=self.device,
            )
        wp.launch(
            sample_gaussian_actions_kernel,
            dim=batch_size,
            inputs=[
                policy_out,
                self.log_std,
                eps,
                self.action_dim,
                int(self.state_dependent_std),
                int(self.squash),
                int(deterministic),
                self.log_std_min,
                self.log_std_max,
            ],
            outputs=[actions, log_probs],
            device=self.device,
        )
        return actions, log_probs, policy_out

    def sample_reuse(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array2d[wp.float32]]:
        """Sample actions into persistent no-grad buffers."""

        return self._sample_reuse_impl(
            obs, seed=int(seed), seed_counter=None, seed_offset=0, deterministic=deterministic
        )

    def sample_reuse_seed_counter(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed_counter: wp.array[wp.int32],
        seed_offset: int = 0,
        deterministic: bool = False,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array2d[wp.float32]]:
        """Sample actions using a graph-replay-safe device seed counter."""

        return self._sample_reuse_impl(
            obs, seed=0, seed_counter=seed_counter, seed_offset=int(seed_offset), deterministic=deterministic
        )

    def _sample_reuse_impl(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed: int,
        seed_counter: wp.array[wp.int32] | None,
        seed_offset: int,
        deterministic: bool,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array2d[wp.float32]]:
        batch_size = int(obs.shape[0])
        self._ensure_sample_reuse_buffers(batch_size)
        actions = self._sample_reuse_actions
        log_probs = self._sample_reuse_log_probs
        eps = self._sample_reuse_eps
        if actions is None or log_probs is None or eps is None:
            raise RuntimeError("sample reuse buffers were not initialized")

        policy_out = self.net.forward_reuse(obs)
        if not deterministic:
            if seed_counter is None:
                wp.launch(
                    fill_eps_kernel,
                    dim=(batch_size, self.action_dim),
                    inputs=[int(seed)],
                    outputs=[eps],
                    device=self.device,
                )
            else:
                wp.launch(
                    fill_eps_seed_counter_kernel,
                    dim=(batch_size, self.action_dim),
                    inputs=[seed_counter, int(seed_offset)],
                    outputs=[eps],
                    device=self.device,
                )
        wp.launch(
            sample_gaussian_actions_kernel,
            dim=batch_size,
            inputs=[
                policy_out,
                self.log_std,
                eps,
                self.action_dim,
                int(self.state_dependent_std),
                int(self.squash),
                int(deterministic),
                self.log_std_min,
                self.log_std_max,
            ],
            outputs=[actions, log_probs],
            device=self.device,
        )
        return actions[:batch_size], log_probs[:batch_size], policy_out

    def _ensure_sample_reuse_buffers(self, batch_size: int) -> None:
        requested = int(batch_size)
        if requested <= 0:
            raise ValueError("batch_size must be positive")
        if self._sample_reuse_capacity >= requested and self._sample_reuse_actions is not None:
            return
        self._sample_reuse_capacity = requested
        self._sample_reuse_actions = wp.empty(
            (self._sample_reuse_capacity, self.action_dim),
            dtype=wp.float32,
            device=self.device,
            requires_grad=False,
        )
        self._sample_reuse_log_probs = wp.empty(
            self._sample_reuse_capacity,
            dtype=wp.float32,
            device=self.device,
            requires_grad=False,
        )
        self._sample_reuse_eps = wp.empty(
            (self._sample_reuse_capacity, self.action_dim),
            dtype=wp.float32,
            device=self.device,
            requires_grad=False,
        )

    def reserve_reuse_buffers(self, batch_size: int) -> None:
        """Reserve no-grad policy sampling buffers for at least ``batch_size`` rows."""

        self.net.reserve_buffers(batch_size)
        self._ensure_sample_reuse_buffers(batch_size)

    def copy_from(self, other: GaussianActor) -> None:
        """Copy parameters from another actor."""

        self.net.copy_from(other.net)
        wp.launch(
            copy_1d_kernel, dim=self.action_dim, inputs=[other.log_std], outputs=[self.log_std], device=self.device
        )


def _kaiming_uniform(rng: np.random.Generator, fan_in: int, fan_out: int, gain: float) -> np.ndarray:
    bound = float(gain) / np.sqrt(float(fan_in))
    return rng.uniform(-bound, bound, size=(int(fan_in), int(fan_out))).astype(np.float32)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _manual_bfloat16_dtype(dtype: str, name: str) -> str:
    key = str(dtype).lower()
    if key in ("float32", "fp32"):
        return "float32"
    if key in ("bfloat16", "bf16"):
        return "bfloat16"
    raise ValueError(f"Unsupported {name} {dtype!r}")


def _check_same_shapes(params: list[wp.array], other_params: list[wp.array]) -> None:
    if len(params) != len(other_params):
        raise ValueError("Parameter lists have different lengths")
    for param, other in zip(params, other_params, strict=True):
        if tuple(param.shape) != tuple(other.shape):
            raise ValueError(f"Parameter shape mismatch: {param.shape} vs {other.shape}")
