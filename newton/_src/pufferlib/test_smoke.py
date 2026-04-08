# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for PufferLib on Warp.

Verifies that kernels compile and produce reasonable outputs.
Run with: uv run python warp/pufferlib/test_smoke.py
"""

from __future__ import annotations

import unittest

import numpy as np

import warp as wp

wp.init()


class TestKernels(unittest.TestCase):
    """Test core computational kernels."""

    def setUp(self):
        self.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_gemm_nn(self):
        """Test tiled GEMM (NN) against NumPy."""
        from newton._src.pufferlib.kernels import matmul

        M, K, N = 128, 64, 128
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)

        A = wp.array(A_np, dtype=float, device=self.device)
        B = wp.array(B_np, dtype=float, device=self.device)
        C = wp.zeros((M, N), dtype=float, device=self.device)

        matmul(A, B, C)

        C_np = C.numpy()
        expected = A_np @ B_np
        np.testing.assert_allclose(C_np, expected, atol=1e-3, rtol=1e-3)

    def test_gemm_tn(self):
        """Test tiled GEMM (TN) against NumPy."""
        from newton._src.pufferlib.kernels import matmul

        K, M, N = 64, 128, 128
        A_np = np.random.randn(K, M).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)

        A = wp.array(A_np, dtype=float, device=self.device)
        B = wp.array(B_np, dtype=float, device=self.device)
        C = wp.zeros((M, N), dtype=float, device=self.device)

        matmul(A, B, C, transpose_a=True)

        C_np = C.numpy()
        expected = A_np.T @ B_np
        np.testing.assert_allclose(C_np, expected, atol=1e-3, rtol=1e-3)

    def test_kaiming_init(self):
        """Test that Kaiming init produces reasonable values."""
        from newton._src.pufferlib.kernels import kaiming_uniform_init_kernel

        n = 1024
        w = wp.zeros(n, dtype=float, device=self.device)
        wp.launch(kaiming_uniform_init_kernel, dim=n,
                  inputs=[w, 256, 42, 1.0], device=self.device)

        w_np = w.numpy()
        bound = 1.0 / np.sqrt(256)
        self.assertTrue(np.all(w_np >= -bound - 0.01))
        self.assertTrue(np.all(w_np <= bound + 0.01))
        self.assertGreater(np.std(w_np), 0.01)


class TestPPO(unittest.TestCase):
    """Test PPO components."""

    def setUp(self):
        self.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_sample_discrete(self):
        """Test discrete action sampling produces valid actions."""
        from newton._src.pufferlib.ppo import sample_actions_discrete

        B = 64
        num_actions = 4
        logits = wp.zeros((B, num_actions), dtype=float, device=self.device)
        act_sizes = wp.array([num_actions], dtype=int, device=self.device)

        actions, logprobs = sample_actions_discrete(logits, act_sizes, 1, seed=42)

        actions_np = actions.numpy()
        logprobs_np = logprobs.numpy()

        # All actions should be in [0, num_actions)
        self.assertTrue(np.all(actions_np >= 0))
        self.assertTrue(np.all(actions_np < num_actions))

        # With uniform logits, logprobs should be close to -log(num_actions)
        expected_logp = -np.log(num_actions)
        np.testing.assert_allclose(logprobs_np, expected_logp, atol=0.01)

    def test_sample_continuous(self):
        """Test continuous action sampling."""
        from newton._src.pufferlib.ppo import sample_actions_continuous

        B = 256
        num_actions = 3
        logits = wp.zeros((B, num_actions), dtype=float, device=self.device)
        logstd = wp.zeros(num_actions, dtype=float, device=self.device)

        actions, logprobs = sample_actions_continuous(logits, logstd, seed=42)

        actions_np = actions.numpy()
        # Actions should be roughly N(0, 1)
        self.assertAlmostEqual(np.mean(actions_np), 0.0, delta=0.3)
        self.assertAlmostEqual(np.std(actions_np), 1.0, delta=0.3)

    def test_gae_vtrace(self):
        """Test GAE+V-Trace advantage computation."""
        from newton._src.pufferlib.ppo import compute_gae_vtrace

        N, T = 4, 8
        values = wp.array(np.ones((N, T), dtype=np.float32) * 0.5, device=self.device)
        rewards = wp.array(np.ones((N, T), dtype=np.float32) * 1.0, device=self.device)
        dones = wp.zeros((N, T), dtype=float, device=self.device)
        importance = wp.array(np.ones((N, T), dtype=np.float32), device=self.device)

        advantages, returns = compute_gae_vtrace(
            values, rewards, dones, importance,
            gamma=0.99, lam=0.95,
        )

        adv_np = advantages.numpy()
        ret_np = returns.numpy()

        # Advantages should be positive (rewards > values)
        self.assertTrue(np.all(adv_np[:, :-1] > 0))
        # Returns = advantages + values
        np.testing.assert_allclose(ret_np[:, :-1], adv_np[:, :-1] + 0.5, atol=1e-5)

    def test_ppo_loss_discrete(self):
        """Test fused PPO loss kernel produces finite outputs."""
        from newton._src.pufferlib.ppo import ppo_loss_and_grad

        B = 32
        num_actions = 4
        logits = wp.array(np.random.randn(B, num_actions).astype(np.float32), device=self.device)
        actions = wp.array(np.random.randint(0, num_actions, (B, 1)).astype(np.float32), device=self.device)
        old_logprobs = wp.array(np.full(B, -np.log(num_actions), dtype=np.float32), device=self.device)
        advantages = wp.array(np.random.randn(B).astype(np.float32), device=self.device)
        values_pred = wp.array(np.random.randn(B).astype(np.float32), device=self.device)
        old_values = wp.array(np.random.randn(B).astype(np.float32), device=self.device)
        returns = wp.array(np.random.randn(B).astype(np.float32), device=self.device)
        act_sizes = wp.array([num_actions], dtype=int, device=self.device)

        loss_out, grad_logits, grad_values, _ = ppo_loss_and_grad(
            logits, actions, old_logprobs, advantages,
            values_pred, old_values, returns,
            act_sizes=act_sizes,
        )

        loss_np = loss_out.numpy()
        grad_l_np = grad_logits.numpy()
        grad_v_np = grad_values.numpy()

        # All should be finite
        self.assertTrue(np.all(np.isfinite(loss_np)))
        self.assertTrue(np.all(np.isfinite(grad_l_np)))
        self.assertTrue(np.all(np.isfinite(grad_v_np)))


class TestNetwork(unittest.TestCase):
    """Test SimpleMLP network."""

    def setUp(self):
        self.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_simple_mlp_forward(self):
        """Test SimpleMLP produces output of correct shape with finite values."""
        from newton._src.pufferlib.network import SimpleMLP

        B = 64
        obs_dim, hidden, num_actions = 16, 32, 4
        mlp = SimpleMLP(obs_dim, hidden, num_actions + 1, max_batch=B, device=self.device)
        X = wp.array(np.random.randn(B, obs_dim).astype(np.float32), device=self.device)

        Y = mlp.forward(X, B)

        self.assertEqual(Y.shape[0], B)
        self.assertEqual(Y.shape[1], num_actions + 1)
        self.assertTrue(np.all(np.isfinite(Y.numpy()[:B])))

    def test_simple_mlp_backward(self):
        """Test SimpleMLP backward produces finite gradients."""
        from newton._src.pufferlib.network import SimpleMLP

        B = 64
        obs_dim, hidden, num_actions = 16, 32, 4
        mlp = SimpleMLP(obs_dim, hidden, num_actions + 1, max_batch=B, device=self.device)
        X = wp.array(np.random.randn(B, obs_dim).astype(np.float32), device=self.device)

        mlp.forward(X, B)
        grad_out = wp.array(np.random.randn(B, num_actions + 1).astype(np.float32), device=self.device)
        grads = mlp.backward(grad_out, B)

        self.assertEqual(len(grads), 3)
        for g in grads:
            self.assertTrue(np.all(np.isfinite(g.numpy())))


class TestOptimizer(unittest.TestCase):
    """Test optimizer implementations."""

    def setUp(self):
        self.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_adamw_step(self):
        """Test AdamW optimizer modifies parameters."""
        from newton._src.pufferlib.optimizer import AdamW

        param = wp.array(np.ones(64, dtype=np.float32), device=self.device, requires_grad=True)
        grad = wp.array(np.ones(64, dtype=np.float32) * 0.1, device=self.device)

        opt = AdamW([param], lr=0.01)
        opt.step([grad])

        param_np = param.numpy()
        # Parameters should have decreased (gradient points in positive direction)
        self.assertTrue(np.all(param_np < 1.0))


if __name__ == "__main__":
    unittest.main()
