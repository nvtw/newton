# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from .kernels import (
    adam_step_1d_kernel,
    adam_step_2d_kernel,
    adam_step_prepare_kernel,
    grad_sumsq_1d_kernel,
    grad_sumsq_2d_kernel,
    muon_gram_tall_kernel,
    muon_gram_wide_kernel,
    muon_nesterov_2d_kernel,
    muon_normalize_2d_kernel,
    muon_ns_tall_kernel,
    muon_ns_wide_kernel,
    muon_poly_kernel,
    muon_step_1d_kernel,
    muon_step_2d_kernel,
    optimizer_step_count_kernel,
    zero_scalar_kernel,
)


class Adam:
    """Adam optimizer for Warp arrays.

    Args:
        params: Trainable Warp arrays. Only 1-D and 2-D float arrays are
            supported because the RL networks use vectors and matrices.
        lr: Learning rate.
        beta1: First-moment decay.
        beta2: Second-moment decay.
        eps: Numerical stabilizer.
        weight_decay: Decoupled-free L2 weight decay added to gradients.
        max_grad_norm: Global gradient-norm clipping threshold. A value less
            than or equal to zero disables clipping.
    """

    def __init__(
        self,
        params: list[wp.array],
        *,
        lr: float = 3.0e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
    ):
        if not params:
            raise ValueError("Adam requires at least one parameter")
        self.params = params
        self.lr = float(lr)
        self.lr_scale = wp.ones(1, dtype=wp.float32, device=params[0].device, requires_grad=False)
        self.pbt_lr_scale = wp.ones(1, dtype=wp.float32, device=params[0].device, requires_grad=False)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self._step_count_host = 0
        self._step_count = wp.array([0], dtype=wp.int32, device=params[0].device)
        self._step_corrections = wp.zeros(2, dtype=wp.float32, device=params[0].device, requires_grad=False)
        self._grad_sumsq = wp.zeros(1, dtype=wp.float32, device=params[0].device, requires_grad=False)
        self.m = [wp.zeros_like(param, requires_grad=False) for param in params]
        self.v = [wp.zeros_like(param, requires_grad=False) for param in params]

    @property
    def step_count(self) -> int:
        """Number of optimizer steps already applied."""

        self._step_count_host = int(self._step_count.numpy()[0])
        return self._step_count_host

    @step_count.setter
    def step_count(self, value: int) -> None:
        self._step_count_host = max(int(value), 0)
        if hasattr(self, "_step_count"):
            self._step_count.assign(np.asarray([self._step_count_host], dtype=np.int32))

    def zero_grad(self) -> None:
        """Clear gradients on all parameters."""

        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """Apply one Adam update and clear gradients."""

        wp.launch(
            adam_step_prepare_kernel,
            dim=1,
            inputs=[self._step_count, self.beta1, self.beta2],
            outputs=[self._step_corrections],
            device=self._step_count.device,
        )
        max_grad_norm = float(self.max_grad_norm)
        if max_grad_norm > 0.0:
            wp.launch(zero_scalar_kernel, dim=1, outputs=[self._grad_sumsq], device=self._grad_sumsq.device)
            for param in self.params:
                grad = param.grad
                if grad is None:
                    continue
                if param.ndim == 1:
                    wp.launch(
                        grad_sumsq_1d_kernel,
                        dim=param.shape[0],
                        inputs=[grad],
                        outputs=[self._grad_sumsq],
                        device=param.device,
                    )
                elif param.ndim == 2:
                    wp.launch(
                        grad_sumsq_2d_kernel,
                        dim=param.shape,
                        inputs=[grad],
                        outputs=[self._grad_sumsq],
                        device=param.device,
                    )
                else:
                    raise ValueError(f"Adam only supports 1-D and 2-D arrays, got ndim={param.ndim}")

        for param, m, v in zip(self.params, self.m, self.v, strict=True):
            grad = param.grad
            if grad is None:
                continue
            device = param.device
            if param.ndim == 1:
                wp.launch(
                    adam_step_1d_kernel,
                    dim=param.shape[0],
                    inputs=[
                        param,
                        grad,
                        m,
                        v,
                        self._grad_sumsq,
                        self._step_corrections,
                        self.lr,
                        self.lr_scale,
                        self.pbt_lr_scale,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        self.weight_decay,
                        max_grad_norm,
                    ],
                    device=device,
                )
            elif param.ndim == 2:
                wp.launch(
                    adam_step_2d_kernel,
                    dim=param.shape,
                    inputs=[
                        param,
                        grad,
                        m,
                        v,
                        self._grad_sumsq,
                        self._step_corrections,
                        self.lr,
                        self.lr_scale,
                        self.pbt_lr_scale,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        self.weight_decay,
                        max_grad_norm,
                    ],
                    device=device,
                )
            else:
                raise ValueError(f"Adam only supports 1-D and 2-D arrays, got ndim={param.ndim}")


    def set_pbt_lr(self, new_lr: float) -> None:
        """Update the PBT learning-rate multiplier in-place (no graph rebuild)."""
        scale = float(new_lr) / (self.lr + 1e-30)
        self.pbt_lr_scale.assign(np.asarray([scale], dtype=np.float32))


NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


class Muon:
    """Muon optimizer for Warp arrays.

    Matrix parameters use the Newton-Schulz orthogonalized Nesterov update
    used by PufferLib. Vector parameters use the same Nesterov update without
    orthogonalization.

    Args:
        params: Trainable 1-D or 2-D Warp arrays.
        lr: Learning rate.
        momentum: Nesterov momentum coefficient.
        eps: Newton-Schulz normalization stabilizer.
        weight_decay: Decoupled weight decay.
        max_grad_norm: Global gradient-norm clipping threshold. A value less
            than or equal to zero disables clipping.
        matrix_transpose: Treat 2-D parameters as logical transposes for the
            matrix orthogonalization and scale. Use this for :class:`WarpMLP`
            weights, which are stored as ``[in, out]`` while PufferLib/Torch
            linear weights are ``[out, in]``.
    """

    def __init__(
        self,
        params: list[wp.array],
        *,
        lr: float = 2.0e-2,
        momentum: float = 0.9,
        eps: float = 1.0e-12,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
        matrix_transpose: bool = False,
    ):
        if not params:
            raise ValueError("Muon requires at least one parameter")
        for param in params:
            if param.ndim not in (1, 2):
                raise ValueError(f"Muon only supports 1-D and 2-D arrays, got ndim={param.ndim}")
        self.params = params
        self.lr = float(lr)
        self.lr_scale = wp.ones(1, dtype=wp.float32, device=params[0].device, requires_grad=False)
        self.pbt_lr_scale = wp.ones(1, dtype=wp.float32, device=params[0].device, requires_grad=False)
        self.momentum_coeff = float(momentum)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.matrix_transpose = bool(matrix_transpose)
        self._step_count_host = 0
        self._step_count = wp.array([0], dtype=wp.int32, device=params[0].device)
        self._grad_sumsq = wp.zeros(1, dtype=wp.float32, device=params[0].device, requires_grad=False)
        self.m = [wp.zeros_like(param, requires_grad=False) for param in params]
        self._updates: list[wp.array | None] = []
        self._scratch: list[wp.array | None] = []
        self._grams: list[wp.array | None] = []
        self._polys: list[wp.array | None] = []
        self._norms: list[wp.array | None] = []
        for param in params:
            if param.ndim == 2:
                rows = int(param.shape[0])
                cols = int(param.shape[1])
                gram_dim = min(rows, cols)
                self._updates.append(wp.zeros_like(param, requires_grad=False))
                self._scratch.append(wp.zeros_like(param, requires_grad=False))
                self._grams.append(wp.zeros((gram_dim, gram_dim), dtype=wp.float32, device=param.device))
                self._polys.append(wp.zeros((gram_dim, gram_dim), dtype=wp.float32, device=param.device))
                self._norms.append(wp.zeros(1, dtype=wp.float32, device=param.device, requires_grad=False))
            else:
                self._updates.append(None)
                self._scratch.append(None)
                self._grams.append(None)
                self._polys.append(None)
                self._norms.append(None)

    @property
    def step_count(self) -> int:
        """Number of optimizer steps already applied."""

        self._step_count_host = int(self._step_count.numpy()[0])
        return self._step_count_host

    @step_count.setter
    def step_count(self, value: int) -> None:
        self._step_count_host = max(int(value), 0)
        if hasattr(self, "_step_count"):
            self._step_count.assign(np.asarray([self._step_count_host], dtype=np.int32))

    def zero_grad(self) -> None:
        """Clear gradients on all parameters."""

        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """Apply one Muon update and clear gradients."""

        max_grad_norm = float(self.max_grad_norm)
        if max_grad_norm > 0.0:
            wp.launch(zero_scalar_kernel, dim=1, outputs=[self._grad_sumsq], device=self._grad_sumsq.device)
            for param in self.params:
                grad = param.grad
                if grad is None:
                    continue
                if param.ndim == 1:
                    wp.launch(
                        grad_sumsq_1d_kernel,
                        dim=param.shape[0],
                        inputs=[grad],
                        outputs=[self._grad_sumsq],
                        device=param.device,
                    )
                else:
                    wp.launch(
                        grad_sumsq_2d_kernel,
                        dim=param.shape,
                        inputs=[grad],
                        outputs=[self._grad_sumsq],
                        device=param.device,
                    )

        for index, (param, momentum_buffer) in enumerate(zip(self.params, self.m, strict=True)):
            grad = param.grad
            if grad is None:
                continue
            if param.ndim == 1:
                wp.launch(
                    muon_step_1d_kernel,
                    dim=param.shape[0],
                    inputs=[
                        param,
                        grad,
                        momentum_buffer,
                        self._grad_sumsq,
                        self.lr,
                        self.lr_scale,
                        self.pbt_lr_scale,
                        self.momentum_coeff,
                        self.weight_decay,
                        max_grad_norm,
                    ],
                    device=param.device,
                )
                continue

            update = self._updates[index]
            scratch = self._scratch[index]
            gram = self._grams[index]
            poly = self._polys[index]
            norm = self._norms[index]
            if update is None or scratch is None or gram is None or poly is None or norm is None:
                raise RuntimeError("Muon matrix scratch buffers were not initialized")

            rows = int(param.shape[0])
            cols = int(param.shape[1])
            wp.launch(
                muon_nesterov_2d_kernel,
                dim=param.shape,
                inputs=[grad, momentum_buffer, self._grad_sumsq, self.momentum_coeff, max_grad_norm],
                outputs=[update],
                device=param.device,
            )
            wp.launch(zero_scalar_kernel, dim=1, outputs=[norm], device=param.device)
            wp.launch(grad_sumsq_2d_kernel, dim=param.shape, inputs=[update], outputs=[norm], device=param.device)
            wp.launch(muon_normalize_2d_kernel, dim=param.shape, inputs=[update, norm, self.eps], device=param.device)

            src = update
            dst = scratch
            use_tall_kernels = rows > cols
            logical_rows = rows
            logical_cols = cols
            if self.matrix_transpose:
                use_tall_kernels = cols <= rows
                logical_rows = cols
                logical_cols = rows
            for coeff_a, coeff_b, coeff_c in NS_COEFFS:
                if use_tall_kernels:
                    wp.launch(
                        muon_gram_tall_kernel, dim=gram.shape, inputs=[src, rows], outputs=[gram], device=param.device
                    )
                    wp.launch(
                        muon_poly_kernel,
                        dim=gram.shape,
                        inputs=[gram, coeff_b, coeff_c],
                        outputs=[poly],
                        device=param.device,
                    )
                    wp.launch(
                        muon_ns_tall_kernel,
                        dim=param.shape,
                        inputs=[src, poly, coeff_a, cols],
                        outputs=[dst],
                        device=param.device,
                    )
                else:
                    wp.launch(
                        muon_gram_wide_kernel, dim=gram.shape, inputs=[src, cols], outputs=[gram], device=param.device
                    )
                    wp.launch(
                        muon_poly_kernel,
                        dim=gram.shape,
                        inputs=[gram, coeff_b, coeff_c],
                        outputs=[poly],
                        device=param.device,
                    )
                    wp.launch(
                        muon_ns_wide_kernel,
                        dim=param.shape,
                        inputs=[src, poly, coeff_a, rows],
                        outputs=[dst],
                        device=param.device,
                    )
                src, dst = dst, src

            scale = np.sqrt(max(1.0, float(logical_rows) / float(logical_cols)))
            wp.launch(
                muon_step_2d_kernel,
                dim=param.shape,
                inputs=[param, grad, src, self.lr, self.lr_scale, self.pbt_lr_scale, self.weight_decay, float(scale)],
                device=param.device,
            )

        wp.launch(optimizer_step_count_kernel, dim=1, inputs=[self._step_count], device=self._step_count.device)

    def set_pbt_lr(self, new_lr: float) -> None:
        """Update the PBT learning-rate multiplier in-place (no graph rebuild)."""
        scale = float(new_lr) / (self.lr + 1e-30)
        self.pbt_lr_scale.assign(np.asarray([scale], dtype=np.float32))
