# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from .kernels import adam_step_1d_kernel, adam_step_2d_kernel


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
    ):
        if not params:
            raise ValueError("Adam requires at least one parameter")
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.m = [wp.zeros_like(param, requires_grad=False) for param in params]
        self.v = [wp.zeros_like(param, requires_grad=False) for param in params]

    def zero_grad(self) -> None:
        """Clear gradients on all parameters."""

        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """Apply one Adam update and clear gradients."""

        self.step_count += 1
        beta1_correction = 1.0 - self.beta1**self.step_count
        beta2_correction = 1.0 - self.beta2**self.step_count

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
                        self.lr,
                        self.beta1,
                        self.beta2,
                        beta1_correction,
                        beta2_correction,
                        self.eps,
                        self.weight_decay,
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
                        self.lr,
                        self.beta1,
                        self.beta2,
                        beta1_correction,
                        beta2_correction,
                        self.eps,
                        self.weight_decay,
                    ],
                    device=device,
                )
            else:
                raise ValueError(f"Adam only supports 1-D and 2-D arrays, got ndim={param.ndim}")
