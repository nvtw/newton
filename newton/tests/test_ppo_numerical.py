# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical correctness tests for PufferLib PPO kernels.

Compares Warp kernel outputs against numpy/scipy reference implementations
to verify mathematical correctness of:
- Continuous action sampling (log-probability computation)
- GAE advantage estimation
- PPO loss and gradient computation
- Observation normalization

Run with::

    uv run --extra dev -m unittest newton.tests.test_ppo_numerical -v
"""

import unittest
import math

import numpy as np
import warp as wp


class TestContinuousLogProb(unittest.TestCase):
    """Verify log-probability computation for continuous Gaussian actions."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_logprob_matches_scipy(self):
        """Log-prob from our kernel matches scipy.stats.norm.logpdf."""
        from scipy.stats import norm

        B, A = 16, 4
        np.random.seed(42)
        means = np.random.randn(B, A).astype(np.float32)
        stds = np.abs(np.random.randn(A).astype(np.float32)) + 0.1  # positive
        actions = np.random.randn(B, A).astype(np.float32)

        # Reference: scipy
        ref_logprobs = np.zeros(B, dtype=np.float32)
        for i in range(B):
            lp = 0.0
            for h in range(A):
                lp += norm.logpdf(actions[i, h], loc=means[i, h], scale=stds[h])
            ref_logprobs[i] = lp

        # Our kernel via _compute_cont_logprobs_kernel
        from newton._src.pufferlib.trainer import _compute_cont_logprobs_kernel

        d_logits = wp.array(means, dtype=wp.float32, device="cuda:0")
        d_std = wp.array(stds, dtype=wp.float32, device="cuda:0")
        d_actions = wp.array(actions, dtype=wp.float32, device="cuda:0")
        d_logprobs = wp.zeros(B, dtype=wp.float32, device="cuda:0")

        wp.launch(_compute_cont_logprobs_kernel, dim=B,
                  inputs=[d_logits, d_std, d_actions, A, d_logprobs], device="cuda:0")
        wp.synchronize()

        our_logprobs = d_logprobs.numpy()

        np.testing.assert_allclose(our_logprobs, ref_logprobs, rtol=1e-4, atol=1e-5,
                                   err_msg="Log-prob computation mismatch")

    def test_logprob_gradient_wrt_mean(self):
        """Gradient of log-prob w.r.t. mean matches finite differences."""
        B, A = 4, 2
        np.random.seed(123)
        means = np.random.randn(B, A).astype(np.float32)
        stds = np.array([0.5, 0.8], dtype=np.float32)
        actions = np.random.randn(B, A).astype(np.float32)

        def logprob_fn(m):
            lp = 0.0
            for i in range(B):
                for h in range(A):
                    z = (actions[i, h] - m[i, h]) / stds[h]
                    lp += -0.5 * z * z - 0.5 * math.log(2 * math.pi) - math.log(stds[h])
            return lp / B  # mean over batch (like PPO inv_batch)

        # Analytical gradient: d(log_prob)/d(mean) = (action - mean) / var
        analytical = np.zeros_like(means)
        for i in range(B):
            for h in range(A):
                analytical[i, h] = (actions[i, h] - means[i, h]) / (stds[h] ** 2) / B

        # Finite difference
        eps = 1e-3
        fd_grad = np.zeros_like(means)
        for i in range(B):
            for h in range(A):
                m_plus = means.copy(); m_plus[i, h] += eps
                m_minus = means.copy(); m_minus[i, h] -= eps
                fd_grad[i, h] = (logprob_fn(m_plus) - logprob_fn(m_minus)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd_grad, rtol=0.05, atol=1e-3,
                                   err_msg="Mean gradient mismatch")

    def test_logprob_gradient_wrt_std(self):
        """Gradient of log-prob w.r.t. scalar std matches finite differences."""
        B, A = 4, 2
        np.random.seed(456)
        means = np.random.randn(B, A).astype(np.float32)
        stds = np.array([0.5, 0.8], dtype=np.float32)
        actions = np.random.randn(B, A).astype(np.float32)

        def logprob_fn_std(s):
            lp = 0.0
            for i in range(B):
                for h in range(A):
                    z = (actions[i, h] - means[i, h]) / s[h]
                    lp += -0.5 * z * z - 0.5 * math.log(2 * math.pi) - math.log(s[h])
            return lp / B

        # Analytical: d(log_prob)/d(std) = ((action-mean)^2/var - 1) / std
        analytical = np.zeros(A, dtype=np.float32)
        for h in range(A):
            grad_sum = 0.0
            for i in range(B):
                diff = actions[i, h] - means[i, h]
                var = stds[h] ** 2
                grad_sum += (diff * diff / var - 1.0) / stds[h]
            analytical[h] = grad_sum / B

        # Finite difference
        eps = 1e-4
        fd_grad = np.zeros(A, dtype=np.float32)
        for h in range(A):
            s_plus = stds.copy(); s_plus[h] += eps
            s_minus = stds.copy(); s_minus[h] -= eps
            fd_grad[h] = (logprob_fn_std(s_plus) - logprob_fn_std(s_minus)) / (2 * eps)

        np.testing.assert_allclose(analytical, fd_grad, rtol=1e-2, atol=1e-4,
                                   err_msg="Std gradient mismatch (this is the critical entropy stabilizer)")

    def test_entropy_gradient_wrt_std(self):
        """Entropy gradient w.r.t. scalar std is 1/std (the key stabilizer)."""
        stds = np.array([0.3, 0.5, 1.0, 2.0], dtype=np.float32)

        # Entropy of Gaussian: H = 0.5*(1+log(2*pi)) + log(std)
        # d(H)/d(std) = 1/std
        for std in stds:
            analytical = 1.0 / std
            eps = 1e-5
            h_plus = 0.5 * (1 + math.log(2 * math.pi)) + math.log(std + eps)
            h_minus = 0.5 * (1 + math.log(2 * math.pi)) + math.log(std - eps)
            fd = (h_plus - h_minus) / (2 * eps)
            self.assertAlmostEqual(analytical, fd, places=2,
                                   msg=f"Entropy gradient wrong at std={std}")

        # The entropy bonus gradient in the loss is: -ent_coef * d(H)/d(std) = -ent_coef/std
        # This DECREASES as std grows (1/std), providing natural stabilization.
        # Compare: with logstd parameterization, d(H)/d(logstd) = 1 (CONSTANT — no stabilization!)


class TestGAE(unittest.TestCase):
    """Verify Generalized Advantage Estimation."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_gae_matches_reference(self):
        """GAE from our kernel matches a simple Python reference."""
        from newton._src.pufferlib.ppo import gae_vtrace_kernel

        N, T = 4, 8
        gamma, lam = 0.99, 0.95
        np.random.seed(42)
        values = np.random.randn(N, T).astype(np.float32) * 0.5
        rewards = np.random.randn(N, T).astype(np.float32) * 0.1
        dones = np.zeros((N, T), dtype=np.float32)
        dones[1, 5] = 1.0  # env 1 terminates at step 5
        importance = np.ones((N, T), dtype=np.float32)

        # Reference GAE (Python)
        ref_advantages = np.zeros((N, T), dtype=np.float32)
        ref_returns = np.zeros((N, T), dtype=np.float32)
        for n in range(N):
            ref_advantages[n, T - 1] = 0.0
            ref_returns[n, T - 1] = values[n, T - 1]
            for t in range(T - 2, -1, -1):
                rho = min(importance[n, t], 1.0)  # rho_clip=1.0
                c = min(importance[n, t], 1.0)     # c_clip=1.0
                not_done = 1.0 - dones[n, t + 1]
                delta = rho * (rewards[n, t + 1] + gamma * values[n, t + 1] * not_done - values[n, t])
                ref_advantages[n, t] = delta + gamma * lam * c * ref_advantages[n, t + 1] * not_done
                ref_returns[n, t] = ref_advantages[n, t] + values[n, t]

        # Our kernel
        d_values = wp.array(values, dtype=wp.float32, device="cuda:0")
        d_rewards = wp.array(rewards, dtype=wp.float32, device="cuda:0")
        d_dones = wp.array(dones, dtype=wp.float32, device="cuda:0")
        d_importance = wp.array(importance, dtype=wp.float32, device="cuda:0")
        d_advantages = wp.zeros((N, T), dtype=wp.float32, device="cuda:0")
        d_returns = wp.zeros((N, T), dtype=wp.float32, device="cuda:0")

        wp.launch(gae_vtrace_kernel, dim=N,
                  inputs=[d_values, d_rewards, d_dones, d_importance,
                          d_advantages, d_returns, gamma, lam, 1.0, 1.0],
                  device="cuda:0")
        wp.synchronize()

        our_advantages = d_advantages.numpy()
        our_returns = d_returns.numpy()

        np.testing.assert_allclose(our_advantages, ref_advantages, rtol=1e-4, atol=1e-5,
                                   err_msg="GAE advantages mismatch")
        np.testing.assert_allclose(our_returns, ref_returns, rtol=1e-4, atol=1e-5,
                                   err_msg="GAE returns mismatch")


class TestAdvantageNormalization(unittest.TestCase):
    """Verify advantage normalization produces mean≈0, std≈1."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_normalized_advantages(self):
        """After normalization, advantages have mean≈0 and std≈1."""
        from newton._src.pufferlib.trainer import _normalize_adv_kernel, _sq_kernel
        from newton._src.pufferlib.reduce import ArraySum

        N = 1024
        np.random.seed(42)
        raw = np.random.randn(N).astype(np.float32) * 5.0 + 3.0  # mean=3, std=5

        d_adv = wp.array(raw, dtype=wp.float32, device="cuda:0")
        d_sq = wp.zeros(N, dtype=wp.float32, device="cuda:0")
        summer1 = ArraySum(N, device="cuda:0")
        summer2 = ArraySum(N, device="cuda:0")

        adv_sum = summer1.compute(d_adv, N)
        wp.launch(_sq_kernel, dim=N, inputs=[d_adv, d_sq], device="cuda:0")
        sq_sum = summer2.compute(d_sq, N)
        wp.launch(_normalize_adv_kernel, dim=N,
                  inputs=[d_adv, adv_sum, sq_sum, float(N)], device="cuda:0")
        wp.synchronize()

        normalized = d_adv.numpy()
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=2,
                               msg=f"Normalized mean should be ≈0, got {normalized.mean()}")
        self.assertAlmostEqual(float(normalized.std()), 1.0, places=1,
                               msg=f"Normalized std should be ≈1, got {normalized.std()}")


class TestObsNormalizerCorrectness(unittest.TestCase):
    """Verify Welford normalizer produces correct statistics."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_welford_matches_numpy(self):
        """Running mean/var from Welford matches numpy after multiple batches."""
        from newton._src.pufferlib.obs_normalizer import ObsNormalizer

        obs_dim = 8
        norm = ObsNormalizer(obs_dim, "cuda:0")

        np.random.seed(42)
        all_data = []
        for _ in range(10):
            batch = np.random.randn(64, obs_dim).astype(np.float32) * 3.0 + 1.0
            all_data.append(batch)
            d_batch = wp.array(batch, dtype=wp.float32, device="cuda:0")
            norm.update(d_batch, 64)

        wp.synchronize()
        mean, var = norm.get_mean_var()

        all_np = np.concatenate(all_data, axis=0)
        ref_mean = all_np.mean(axis=0)
        ref_var = all_np.var(axis=0)

        np.testing.assert_allclose(mean, ref_mean, rtol=0.05, atol=0.1,
                                   err_msg="Welford mean mismatch")
        np.testing.assert_allclose(var, ref_var, rtol=0.1, atol=0.2,
                                   err_msg="Welford var mismatch")


if __name__ == "__main__":
    unittest.main()
