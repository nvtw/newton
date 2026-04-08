# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Running observation normalizer for PPO (CUDA-graph-capture compatible).

Uses Welford's online algorithm to track per-feature mean and variance.
Normalization: ``obs_norm = clip((obs - mean) / sqrt(var + eps), -clip_val, clip_val)``
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _update_stats_kernel(
    obs: wp.array2d(dtype=float),
    count: wp.array(dtype=float, ndim=1),
    mean: wp.array(dtype=float, ndim=1),
    var: wp.array(dtype=float, ndim=1),
    obs_dim: int,
):
    """Update running mean/var with a batch of observations (Welford's)."""
    d = wp.tid()
    if d >= obs_dim:
        return
    N = obs.shape[0]
    # Compute batch mean and var for this feature
    batch_mean = float(0.0)
    for i in range(N):
        batch_mean = batch_mean + obs[i, d]
    batch_mean = batch_mean / float(N)

    batch_var = float(0.0)
    for i in range(N):
        diff = obs[i, d] - batch_mean
        batch_var = batch_var + diff * diff
    batch_var = batch_var / float(N)

    # Combine with running stats
    old_count = count[0]
    new_count = old_count + float(N)
    delta = batch_mean - mean[d]
    m_a = var[d] * old_count
    m_b = batch_var * float(N)
    m2 = m_a + m_b + delta * delta * old_count * float(N) / new_count
    mean[d] = mean[d] + delta * float(N) / new_count
    var[d] = m2 / new_count
    if d == 0:
        count[0] = new_count


@wp.kernel
def _normalize_obs_kernel(
    obs: wp.array2d(dtype=float),
    mean: wp.array(dtype=float, ndim=1),
    var: wp.array(dtype=float, ndim=1),
    out: wp.array2d(dtype=float),
    eps: float,
    clip_val: float,
):
    """Normalize observations in-place: (obs - mean) / sqrt(var + eps), clipped."""
    i, d = wp.tid()
    val = (obs[i, d] - mean[d]) / wp.sqrt(var[d] + eps)
    out[i, d] = wp.clamp(val, -clip_val, clip_val)


class ObsNormalizer:
    """Running observation normalizer (CUDA-graph-capture compatible).

    Call :meth:`update` during rollout collection to track statistics,
    and :meth:`normalize` to normalize a batch of observations.

    Args:
        obs_dim: Observation dimensionality.
        device: Warp device string.
        eps: Small constant for numerical stability.
        clip_val: Clipping range for normalized values.
    """

    def __init__(self, obs_dim: int, device: str, eps: float = 1e-4, clip_val: float = 10.0):
        self.obs_dim = obs_dim
        self.device = device
        self.eps = eps
        self.clip_val = clip_val
        self.mean = wp.zeros(obs_dim, dtype=float, device=device)
        self.var = wp.ones(obs_dim, dtype=float, device=device)
        self.count = wp.array([1e-4], dtype=float, device=device)

    def update(self, obs: wp.array, N: int):
        """Update running statistics with a batch of observations."""
        wp.launch(
            _update_stats_kernel,
            dim=self.obs_dim,
            inputs=[obs, self.count, self.mean, self.var, self.obs_dim],
            device=self.device,
        )

    def normalize(self, obs: wp.array, out: wp.array, N: int):
        """Normalize observations into *out*."""
        wp.launch(
            _normalize_obs_kernel,
            dim=(N, self.obs_dim),
            inputs=[obs, self.mean, self.var, out, self.eps, self.clip_val],
            device=self.device,
        )

    def get_mean_var(self):
        """Return (mean, var) as numpy arrays (for ONNX export)."""
        return self.mean.numpy(), self.var.numpy()
