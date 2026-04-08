# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muon optimizer and Adam step for PufferLib on Warp.

Muon uses Nesterov momentum followed by Newton-Schulz orthogonalization
(5 iterations) for 2D weight matrices.  1D parameters (biases) fall back
to a simple SGD-with-momentum update.

The implementation matches PufferLib C++ ``muon.cu`` closely.

All kernels use ``enable_backward=False``.
"""

from __future__ import annotations

import math

import warp as wp

from newton._src.pufferlib.kernels import matmul

wp.set_module_options({"enable_backward": False})

# Newton-Schulz polynomial coefficients (from PufferLib muon.cu)
NS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]
NS_ITERS = 5


# ---------------------------------------------------------------------------
# Elementwise optimizer kernels
# ---------------------------------------------------------------------------


@wp.kernel
def nesterov_momentum_kernel(
    grad: wp.array(dtype=float, ndim=1),
    momentum: wp.array(dtype=float, ndim=1),
    mu: float,
):
    """Nesterov momentum: m = mu*m + g; g = g + mu*m (in-place on grad)."""
    i = wp.tid()
    m = mu * momentum[i] + grad[i]
    momentum[i] = m
    grad[i] = grad[i] + mu * m


_SUM_SQ_BLOCK_DIM = 1024


@wp.kernel
def _sum_sq_kernel(
    grad: wp.array(dtype=float, ndim=1),
    n: int,
    out: wp.array(dtype=float, ndim=1),
):
    """Parallel sum(grad[0:n]^2), result written to out[0] (overwrite).

    Launch with ``wp.launch_tiled(dim=[1], block_dim=_SUM_SQ_BLOCK_DIM)``.
    """
    _blk, lane = wp.tid()
    num_threads = wp.block_dim()
    partial = float(0.0)
    upper = ((n + num_threads - 1) // num_threads) * num_threads
    for idx in range(lane, upper, num_threads):
        val = float(0.0)
        if idx < n:
            val = grad[idx] * grad[idx]
        t = wp.tile(val)
        s = wp.tile_sum(t)
        partial += s[0]
    if lane == 0:
        out[0] = partial


@wp.kernel
def _sum_sq_acc_kernel(
    grad: wp.array(dtype=float, ndim=1),
    n: int,
    out: wp.array(dtype=float, ndim=1),
):
    """Parallel sum(grad[0:n]^2), result **accumulated** into out[0].

    Launch with ``wp.launch_tiled(dim=[1], block_dim=_SUM_SQ_BLOCK_DIM)``.
    """
    _blk, lane = wp.tid()
    num_threads = wp.block_dim()
    partial = float(0.0)
    upper = ((n + num_threads - 1) // num_threads) * num_threads
    for idx in range(lane, upper, num_threads):
        val = float(0.0)
        if idx < n:
            val = grad[idx] * grad[idx]
        t = wp.tile(val)
        s = wp.tile_sum(t)
        partial += s[0]
    if lane == 0:
        wp.atomic_add(out, 0, partial)


@wp.kernel
def clip_grad_norm_kernel(
    grad: wp.array(dtype=float, ndim=1),
    norm_sq: wp.array(dtype=float, ndim=1),
    max_norm: float,
):
    """Clip gradient by global norm (in-place)."""
    i = wp.tid()
    n = wp.sqrt(norm_sq[0])
    clip = wp.min(max_norm / (n + 1.0e-6), 1.0)
    grad[i] = grad[i] * clip


@wp.kernel
def _normalize_kernel(
    x: wp.array(dtype=float, ndim=1),
    norm_sq: wp.array(dtype=float, ndim=1),
):
    """Normalize x in-place: x /= max(||x||, eps)."""
    i = wp.tid()
    inv_norm = 1.0 / wp.max(wp.sqrt(norm_sq[0]), 1.0e-7)
    x[i] = x[i] * inv_norm


@wp.kernel
def weight_update_kernel(
    weight: wp.array(dtype=float, ndim=1),
    update: wp.array(dtype=float, ndim=1),
    lr: float,
    wd: float,
    scale: float,
):
    """w = w * (1 - lr*wd) - lr * scale * update."""
    i = wp.tid()
    weight[i] = weight[i] * (1.0 - lr * wd) - lr * scale * update[i]


@wp.kernel
def weight_update_dev_lr_kernel(
    weight: wp.array(dtype=float, ndim=1),
    update: wp.array(dtype=float, ndim=1),
    lr_arr: wp.array(dtype=float, ndim=1),
    wd: float,
    scale: float,
):
    """w = w * (1 - lr*wd) - lr * scale * update.  LR read from device array."""
    i = wp.tid()
    lr = lr_arr[0]
    weight[i] = weight[i] * (1.0 - lr * wd) - lr * scale * update[i]


@wp.kernel
def scale_flat_kernel(
    x: wp.array(dtype=float, ndim=1),
    alpha: float,
    out: wp.array(dtype=float, ndim=1),
):
    i = wp.tid()
    out[i] = x[i] * alpha


@wp.kernel
def _copy_kernel(src: wp.array(dtype=float, ndim=1), dst: wp.array(dtype=float, ndim=1)):
    i = wp.tid()
    dst[i] = src[i]


# ---------------------------------------------------------------------------
# Newton-Schulz GEMM helpers
# ---------------------------------------------------------------------------


@wp.kernel
def addmm_kernel(
    matmul_result: wp.array(dtype=float, ndim=1),
    C: wp.array(dtype=float, ndim=1),
    alpha: float,
    beta: float,
):
    """C = alpha * matmul_result + beta * C (elementwise on flat arrays)."""
    i = wp.tid()
    C[i] = alpha * matmul_result[i] + beta * C[i]


def addmm(A: wp.array, B: wp.array, C: wp.array, alpha: float = 1.0, beta: float = 0.0,
          tmp: wp.array | None = None):
    """C = alpha * A @ B + beta * C.

    Pass *tmp* (same shape as *C*) for graph-capture compatibility.
    """
    if tmp is None:
        tmp = wp.zeros_like(C)
    matmul(A, B, tmp)
    wp.launch(addmm_kernel, dim=C.size,
              inputs=[tmp.flatten(), C.flatten(), alpha, beta],
              device=C.device)


# ---------------------------------------------------------------------------
# Muon optimizer class
# ---------------------------------------------------------------------------


class Muon:
    """Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization.

    Matches PufferLib C++ ``muon.cu`` closely:

    1. Clip gradients by global norm.
    2. Apply Nesterov momentum to the full flat gradient buffer.
    3. Per 2D parameter: normalize, run 5 Newton-Schulz iterations with
       ping-pong buffers, then apply the orthogonalized update.
    4. Per 1D parameter: plain SGD with momentum.

    All scratch buffers are pre-allocated at construction time so
    :meth:`step` is CUDA-graph-capture safe.

    Args:
        params: list of Warp arrays (the model parameters).
        lr: learning rate.
        momentum: Nesterov momentum coefficient (``beta1`` in PufferLib).
        weight_decay: L2 weight decay.
        max_grad_norm: gradient clipping threshold.
        ns_iters: number of Newton-Schulz iterations (default 5).
        device_lr: if provided, a 1-element device array holding the LR.
            When set, the optimizer reads LR from this array (for graph
            capture with LR annealing).
    """

    def __init__(self, params: list[wp.array], lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0, max_grad_norm: float = 1.0, ns_iters: int = NS_ITERS,
                 device_lr: wp.array | None = None):
        self.params = params
        self.lr = lr
        self.mu = momentum
        self.wd = weight_decay
        self.max_grad_norm = max_grad_norm
        self.ns_iters = ns_iters
        self.device_lr = device_lr

        device = params[0].device
        self.device = device

        self.momentum_buffers: list[wp.array] = []
        for p in params:
            self.momentum_buffers.append(wp.zeros_like(p))

        self._norm_acc = wp.zeros(1, dtype=float, device=device)

        # Pre-allocate Newton-Schulz scratch per 2D parameter
        self._ns_bufs: dict[int, dict] = {}
        for i, p in enumerate(params):
            if p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1:
                R, C = p.shape[0], p.shape[1]
                M = min(R, C)
                self._ns_bufs[i] = {
                    "gram": wp.zeros((M, M), dtype=float, device=device),
                    "gram_buf": wp.zeros((M, M), dtype=float, device=device),
                    "x_buf": wp.zeros((R, C), dtype=float, device=device),
                    "norm_sq": wp.zeros(1, dtype=float, device=device),
                    "addmm_tmp_gram": wp.zeros((M, M), dtype=float, device=device),
                    "addmm_tmp_update": wp.zeros((R, C), dtype=float, device=device),
                    "R": R,
                    "C": C,
                    "M": M,
                    "tall": R > C,
                }

    def reset(self):
        """Zero all momentum buffers."""
        for m in self.momentum_buffers:
            m.zero_()

    def step(self, grads: list[wp.array]):
        """Perform one optimizer step given gradients for each parameter.

        Follows PufferLib ``muon_step`` order:
        1. Clip gradients by global norm.
        2. Nesterov momentum on all grads.
        3. Per-parameter: Newton-Schulz (2D) or plain update (1D).
        """
        device = self.device

        # 1. Global gradient norm clipping
        self._norm_acc.zero_()
        for g in grads:
            wp.launch_tiled(_sum_sq_acc_kernel, dim=[1],
                            inputs=[g.flatten(), g.size, self._norm_acc],
                            block_dim=_SUM_SQ_BLOCK_DIM, device=device)
        for g in grads:
            wp.launch(clip_grad_norm_kernel, dim=g.size,
                      inputs=[g.flatten(), self._norm_acc, self.max_grad_norm], device=device)

        # 2. Nesterov momentum on all parameters
        for grad, mom in zip(grads, self.momentum_buffers):
            wp.launch(nesterov_momentum_kernel, dim=grad.size,
                      inputs=[grad.flatten(), mom.flatten(), self.mu], device=device)

        # 3. Per-parameter update
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if i in self._ns_bufs:
                self._newton_schulz_update(param, grad, i)
            else:
                scale = 1.0
                if self.device_lr is not None:
                    wp.launch(weight_update_dev_lr_kernel, dim=param.size,
                              inputs=[param.flatten(), grad.flatten(),
                                      self.device_lr, self.wd, scale],
                              device=device)
                else:
                    wp.launch(weight_update_kernel, dim=param.size,
                              inputs=[param.flatten(), grad.flatten(),
                                      self.lr, self.wd, scale],
                              device=device)

    def _newton_schulz_update(self, param: wp.array, grad: wp.array, idx: int):
        """Apply Newton-Schulz orthogonalization matching PufferLib muon.cu.

        PufferLib algorithm per 2D parameter:
        1. Normalize grad: x = grad / ||grad||
        2. For 5 iterations (ping-pong between x and x_buf):
           a. gram = src^T @ src  (tall) or src @ src^T  (wide)
           b. gram_buf = copy(gram)
           c. gram = c2 * gram @ gram_buf + c1 * gram
           d. dst = copy(src)
           e. dst = c0 * (tall ? src @ gram_buf : gram_buf @ src) + dst
        3. scale = sqrt(max(1, R/C))
        4. weight -= lr * scale * update
        """
        bufs = self._ns_bufs[idx]
        gram = bufs["gram"]
        gram_buf = bufs["gram_buf"]
        x_buf = bufs["x_buf"]
        norm_sq = bufs["norm_sq"]
        R, C, M = bufs["R"], bufs["C"], bufs["M"]
        tall = bufs["tall"]
        device = self.device

        # Normalize grad in-place
        norm_sq.zero_()
        wp.launch_tiled(_sum_sq_kernel, dim=[1],
                        inputs=[grad.flatten(), grad.size, norm_sq],
                        block_dim=_SUM_SQ_BLOCK_DIM, device=device)
        wp.launch(_normalize_kernel, dim=grad.size,
                  inputs=[grad.flatten(), norm_sq], device=device)

        # Ping-pong: even iterations read from grad (src), write to x_buf (dst)
        #            odd iterations read from x_buf (src), write to grad (dst)
        for it in range(self.ns_iters):
            c0, c1, c2 = NS_COEFFS[it]
            if it % 2 == 0:
                src, dst = grad, x_buf
            else:
                src, dst = x_buf, grad

            # gram = src^T @ src (tall) or src @ src^T (wide)
            if tall:
                matmul(src, src, gram, transpose_a=True, transpose_b=False)
            else:
                matmul(src, src, gram, transpose_a=False, transpose_b=True)

            # gram_buf = copy(gram)
            wp.launch(_copy_kernel, dim=gram.size,
                      inputs=[gram.flatten(), gram_buf.flatten()], device=device)

            # C++ puf_addmm_nn(&gram, &gram, &gram_buf, c2, c1):
            #   gram_buf = c2 * (gram @ gram) + c1 * gram_buf
            addmm(gram, gram, gram_buf, alpha=c2, beta=c1, tmp=bufs["addmm_tmp_gram"])

            # dst = copy(src), then dst = 1.0 * (A @ B) + c0 * dst
            wp.launch(_copy_kernel, dim=src.size,
                      inputs=[src.flatten(), dst.flatten()], device=device)

            if tall:
                addmm(src, gram_buf, dst, alpha=1.0, beta=c0, tmp=bufs["addmm_tmp_update"])
            else:
                addmm(gram_buf, src, dst, alpha=1.0, beta=c0, tmp=bufs["addmm_tmp_update"])

        # After 5 iterations (odd count), result is in x_buf
        update = x_buf
        scale = math.sqrt(max(1.0, R / C))

        if self.device_lr is not None:
            wp.launch(weight_update_dev_lr_kernel, dim=param.size,
                      inputs=[param.flatten(), update.flatten(),
                              self.device_lr, self.wd, scale],
                      device=device)
        else:
            wp.launch(weight_update_kernel, dim=param.size,
                      inputs=[param.flatten(), update.flatten(),
                              self.lr, self.wd, scale],
                      device=device)


# ---------------------------------------------------------------------------
# AdamW optimizer (graph-capture compatible, same API as Muon)
# ---------------------------------------------------------------------------


@wp.kernel
def _adamw_step_kernel(
    param: wp.array(dtype=float, ndim=1),
    grad: wp.array(dtype=float, ndim=1),
    m: wp.array(dtype=float, ndim=1),
    v: wp.array(dtype=float, ndim=1),
    lr_arr: wp.array(dtype=float, ndim=1),
    bc_arr: wp.array(dtype=float, ndim=1),
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
):
    """AdamW step reading LR and bias-correction from device arrays.

    ``bc_arr`` layout: [beta1^t, beta2^t, 1-beta1^t, 1-beta2^t].
    """
    i = wp.tid()
    lr = lr_arr[0]
    bc1 = bc_arr[2]
    bc2 = bc_arr[3]
    g = grad[i]

    m_new = beta1 * m[i] + (1.0 - beta1) * g
    v_new = beta2 * v[i] + (1.0 - beta2) * g * g
    m[i] = m_new
    v[i] = v_new

    m_hat = m_new / bc1
    v_hat = v_new / bc2

    param[i] = param[i] * (1.0 - lr * weight_decay) - lr * m_hat / (wp.sqrt(v_hat) + eps)


@wp.kernel
def _update_bc_kernel(
    bc_arr: wp.array(dtype=float, ndim=1),
    beta1: float,
    beta2: float,
):
    """Multiply running bias-correction products: bc1 *= beta1, bc2 *= beta2,
    then store 1 - bc1 and 1 - bc2."""
    bc_arr[0] = bc_arr[0] * beta1
    bc_arr[1] = bc_arr[1] * beta2
    bc_arr[2] = 1.0 - bc_arr[0]
    bc_arr[3] = 1.0 - bc_arr[1]


class AdamW:
    """AdamW optimizer with the same API as :class:`Muon`.

    All state lives on-device so :meth:`step` is CUDA-graph-capture safe.

    Args:
        params: list of Warp arrays (the model parameters).
        lr: learning rate (used when ``device_lr`` is ``None``).
        beta1: exponential decay rate for the first moment.
        beta2: exponential decay rate for the second moment.
        eps: term added to the denominator for numerical stability.
        weight_decay: decoupled weight decay coefficient.
        max_grad_norm: gradient clipping threshold.
        device_lr: if provided, a 1-element device array holding the LR.
    """

    def __init__(self, params: list[wp.array], lr: float = 3e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0,
                 max_grad_norm: float = 1.0, device_lr: wp.array | None = None):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.max_grad_norm = max_grad_norm
        self.device_lr = device_lr

        device = params[0].device
        self.device = device

        self.m = [wp.zeros_like(p) for p in params]
        self.v = [wp.zeros_like(p) for p in params]

        # bc_arr[0] = beta1^t, bc_arr[1] = beta2^t (running products)
        # bc_arr[2] = 1 - beta1^t, bc_arr[3] = 1 - beta2^t (bias corrections)
        self._bc_arr = wp.array([1.0, 1.0, 0.0, 0.0], dtype=float, device=device)

        self._norm_acc = wp.zeros(1, dtype=float, device=device)

        if device_lr is None:
            self._lr_arr = wp.array([lr], dtype=float, device=device)
        else:
            self._lr_arr = device_lr

    def reset(self):
        """Zero all moment buffers and reset the step counter."""
        for buf in self.m:
            buf.zero_()
        for buf in self.v:
            buf.zero_()
        self._bc_arr.assign([1.0, 1.0, 0.0, 0.0])

    def step(self, grads: list[wp.array]):
        """Perform one AdamW step.

        1. Clip gradients by global norm.
        2. Update bias-correction terms on device.
        3. Per-parameter AdamW update.
        """
        device = self.device

        # 1. Global gradient norm clipping
        self._norm_acc.zero_()
        for g in grads:
            wp.launch_tiled(_sum_sq_acc_kernel, dim=[1],
                            inputs=[g.flatten(), g.size, self._norm_acc],
                            block_dim=_SUM_SQ_BLOCK_DIM, device=device)
        for g in grads:
            wp.launch(clip_grad_norm_kernel, dim=g.size,
                      inputs=[g.flatten(), self._norm_acc, self.max_grad_norm], device=device)

        # 2. Advance bias-correction on device
        wp.launch(_update_bc_kernel, dim=1,
                  inputs=[self._bc_arr, self.beta1, self.beta2], device=device)

        # 3. Per-parameter update
        for param, grad, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            wp.launch(_adamw_step_kernel, dim=param.size,
                      inputs=[param.flatten(), grad.flatten(), m.flatten(), v.flatten(),
                              self._lr_arr, self._bc_arr,
                              self.beta1, self.beta2, self.eps, self.wd],
                      device=device)


