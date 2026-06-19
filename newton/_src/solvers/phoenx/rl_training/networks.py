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
    copy_1d_kernel,
    copy_2d_kernel,
    dense_activation_grad_kernel,
    dense_bias_partial_grad_kernel,
    dense_bias_reduce_grad_kernel,
    dense_input_grad_kernel,
    dense_input_grad_tiled_kernel,
    dense_layer_kernel,
    dense_weight_bias_grad_kernel,
    dense_weight_grad_tiled_kernel,
    fill_eps_kernel,
    gaussian_entropy_kernel,
    gaussian_log_prob_kernel,
    sample_gaussian_actions_kernel,
    soft_update_1d_kernel,
    soft_update_2d_kernel,
)
from .kernels_bf16 import (
    cast_2d_float_to_bfloat16_kernel,
    dense_bias_activation_kernel,
    dense_forward_bf16_tiled_kernel,
    dense_weight_grad_bf16_tiled_kernel,
)

_BF16_FORWARD_MIN_BATCH = DENSE_TILE_BATCH * 256


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
            weight-gradient tile matmul. Supports ``"float32"`` and
            ``"bfloat16"``.
        manual_forward_dtype: Input dtype for manual CUDA hidden-layer forward
            tile matmul. Supports ``"float32"`` and ``"bfloat16"``.
    """

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
        self._manual_input: wp.array2d[wp.float32] | None = None
        self._manual_outputs: list[wp.array2d[wp.float32]] = []
        self._manual_output_grads: list[wp.array2d[wp.float32]] = []
        self._manual_pre_grads: list[wp.array2d[wp.float32]] = []
        self._manual_bias_partials: list[wp.array2d[wp.float32]] = []
        self._manual_bf16_inputs: list[wp.array2d[wp.bfloat16]] = []
        self._manual_bf16_pre_grads: list[wp.array2d[wp.bfloat16]] = []
        self._manual_bf16_weights: list[wp.array2d[wp.bfloat16]] = []

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
                self._launch_forward_layer(layer, y, weight, bias, activation, out)
            y = out
        return y

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
            self._launch_forward_layer(layer, y, weight, bias, activation, out)
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
    ) -> None:
        if self._uses_bf16_forward(x, weight):
            x_bf16 = self._manual_bf16_inputs[layer]
            weight_bf16 = self._manual_bf16_weights[layer]
            wp.launch(
                cast_2d_float_to_bfloat16_kernel,
                dim=x.shape,
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
                    _ceil_div(int(out.shape[0]), DENSE_TILE_BATCH),
                    _ceil_div(int(out.shape[1]), DENSE_TILE_OUT),
                ),
                inputs=[x_bf16, weight_bf16, int(weight.shape[0])],
                outputs=[out],
                block_dim=DENSE_TILE_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                dense_bias_activation_kernel,
                dim=out.shape,
                inputs=[out, bias, activation],
                device=self.device,
            )
        else:
            wp.launch(
                dense_layer_kernel,
                dim=out.shape,
                inputs=[x, weight, bias, int(weight.shape[0]), activation],
                outputs=[out],
                device=self.device,
            )

    def _uses_bf16_forward(self, x: wp.array2d[wp.float32], weight: wp.array2d[wp.float32]) -> bool:
        return (
            self.device.is_cuda
            and self.manual_forward_dtype == "bfloat16"
            and int(x.shape[0]) >= _BF16_FORWARD_MIN_BATCH
            and int(weight.shape[0]) >= DENSE_TILE_IN
            and int(weight.shape[1]) >= 64
        )

    def backward_manual(self, output_grad: wp.array2d[wp.float32]) -> None:
        """Backpropagate from an output gradient into parameter gradients."""

        if self._manual_input is None or not self._manual_outputs:
            raise RuntimeError("forward_manual() must be called before backward_manual()")
        if int(output_grad.shape[0]) != self._manual_batch_size or int(output_grad.shape[1]) != self.output_dim:
            raise ValueError("Manual MLP output gradient shape does not match the last forward pass")

        grad_y = output_grad
        for layer in reversed(range(len(self.weights))):
            weight = self.weights[layer]
            bias = self.biases[layer]
            activation = self.output_activation if layer == len(self.weights) - 1 else self.activation
            grad_pre = self._manual_pre_grads[layer]
            wp.launch(
                dense_activation_grad_kernel,
                dim=grad_pre.shape,
                inputs=[self._manual_outputs[layer], grad_y, activation],
                outputs=[grad_pre],
                device=self.device,
            )
            x = self._manual_input if layer == 0 else self._manual_outputs[layer - 1]
            if self.device.is_cuda:
                if self.manual_weight_grad_dtype == "bfloat16":
                    x_bf16 = self._manual_bf16_inputs[layer]
                    grad_pre_bf16 = self._manual_bf16_pre_grads[layer]
                    wp.launch(
                        cast_2d_float_to_bfloat16_kernel,
                        dim=x.shape,
                        inputs=[x],
                        outputs=[x_bf16],
                        device=self.device,
                    )
                    wp.launch(
                        cast_2d_float_to_bfloat16_kernel,
                        dim=grad_pre.shape,
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
                        inputs=[x_bf16, grad_pre_bf16, self._manual_batch_size],
                        outputs=[weight.grad],
                        block_dim=DENSE_TILE_BLOCK_DIM,
                        device=self.device,
                    )
                else:
                    wp.launch_tiled(
                        dense_weight_grad_tiled_kernel,
                        dim=(
                            _ceil_div(int(weight.shape[0]), DENSE_TILE_IN),
                            _ceil_div(int(weight.shape[1]), DENSE_TILE_OUT),
                        ),
                        inputs=[x, grad_pre, self._manual_batch_size],
                        outputs=[weight.grad],
                        block_dim=DENSE_TILE_BLOCK_DIM,
                        device=self.device,
                    )
                bias_partial = self._manual_bias_partials[layer]
                wp.launch(
                    dense_bias_partial_grad_kernel,
                    dim=bias_partial.shape,
                    inputs=[grad_pre, self._manual_batch_size],
                    outputs=[bias_partial],
                    device=self.device,
                )
                wp.launch(
                    dense_bias_reduce_grad_kernel,
                    dim=int(weight.shape[1]),
                    inputs=[bias_partial, int(bias_partial.shape[0])],
                    outputs=[bias.grad],
                    device=self.device,
                )
            else:
                wp.launch(
                    dense_weight_bias_grad_kernel,
                    dim=weight.shape,
                    inputs=[x, grad_pre, self._manual_batch_size],
                    outputs=[weight.grad, bias.grad],
                    device=self.device,
                )
            if layer > 0:
                grad_y = self._manual_output_grads[layer - 1]
                if self.device.is_cuda:
                    wp.launch_tiled(
                        dense_input_grad_tiled_kernel,
                        dim=(
                            _ceil_div(int(grad_y.shape[0]), DENSE_TILE_BATCH),
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
                        dim=grad_y.shape,
                        inputs=[grad_pre, weight, int(weight.shape[1])],
                        outputs=[grad_y],
                        device=self.device,
                    )

    def _ensure_manual_buffers(self, batch_size: int) -> None:
        if self._manual_batch_size == int(batch_size) and len(self._manual_outputs) == len(self.weights):
            return
        self._manual_batch_size = int(batch_size)
        self._manual_outputs = []
        self._manual_output_grads = []
        self._manual_pre_grads = []
        self._manual_bias_partials = []
        self._manual_bf16_inputs = []
        self._manual_bf16_pre_grads = []
        self._manual_bf16_weights = []
        bias_tile_count = _ceil_div(self._manual_batch_size, DENSE_BIAS_TILE_BATCH)
        for in_dim, width in pairwise(self.layer_sizes):
            shape = (self._manual_batch_size, int(width))
            self._manual_outputs.append(wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=False))
            self._manual_output_grads.append(wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=False))
            self._manual_pre_grads.append(wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=False))
            if self.device.is_cuda:
                self._manual_bias_partials.append(
                    wp.empty((bias_tile_count, int(width)), dtype=wp.float32, device=self.device, requires_grad=False)
                )
                if self.manual_weight_grad_dtype == "bfloat16" or self.manual_forward_dtype == "bfloat16":
                    self._manual_bf16_inputs.append(
                        wp.empty(
                            (self._manual_batch_size, int(in_dim)),
                            dtype=wp.bfloat16,
                            device=self.device,
                            requires_grad=False,
                        )
                    )
                    if self.manual_weight_grad_dtype == "bfloat16":
                        self._manual_bf16_pre_grads.append(
                            wp.empty(shape, dtype=wp.bfloat16, device=self.device, requires_grad=False)
                        )
                    if self.manual_forward_dtype == "bfloat16":
                        self._manual_bf16_weights.append(
                            wp.empty(
                                (int(in_dim), int(width)), dtype=wp.bfloat16, device=self.device, requires_grad=False
                            )
                        )

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
        if deterministic:
            eps.zero_()
        else:
            wp.launch(fill_eps_kernel, dim=eps.shape, inputs=[int(seed)], outputs=[eps], device=self.device)
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

    def copy_from(self, other: GaussianActor) -> None:
        """Copy parameters from another actor."""

        self.net.copy_from(other.net)
        wp.launch(
            copy_1d_kernel, dim=self.action_dim, inputs=[other.log_std], outputs=[self.log_std], device=self.device
        )


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
