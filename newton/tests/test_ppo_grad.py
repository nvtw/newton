# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for PPO gradient stability and differentiability."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.ppo import (
    ActorCritic,
    AdamW,
    PPOTrainer,
    WarpMLP,
    _ppo_fused_loss_and_grad_kernel,
)


@wp.kernel
def _mse_loss_kernel(pred: wp.array2d[float], target: wp.array2d[float], loss: wp.array[float], n: int):
    i = wp.tid()
    d = pred[i, 0] - target[i, 0]
    wp.atomic_add(loss, 0, 0.5 * d * d / float(n))


class TestWarpMLPGradients(unittest.TestCase):
    """Test backward pass through WarpMLP for numerical stability."""

    def test_mlp_backward_large_target(self):
        """WarpMLP should not produce NaN grads on a single forward/backward."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        batch = 256
        mlp = WarpMLP([48, 128, 128, 128, 1], activation="elu", device=device, output_gain=1.0)
        mlp.alloc_intermediates(batch)

        obs = wp.array(
            np.random.default_rng(42).standard_normal((batch, 48)).astype(np.float32),
            device=device,
            requires_grad=True,
        )
        target = wp.array(np.full((batch, 1), 36.0, dtype=np.float32), device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            out = mlp.forward(obs)
            wp.launch(_mse_loss_kernel, dim=batch, inputs=[out, target, loss, batch], device=device)
        tape.backward(loss)

        grads = mlp.grad_arrays()
        has_nan_grad = any(np.any(np.isnan(g.numpy())) for g in grads)
        loss_val = loss.numpy()[0]

        self.assertFalse(np.isnan(loss_val), "Loss is NaN")
        self.assertFalse(has_nan_grad, "Gradient is NaN")
        self.assertGreater(loss_val, 0.0, "Loss should be positive for non-zero target")

    def test_mlp_backward_with_adamw(self):
        """WarpMLP + AdamW should not produce NaN during training."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        batch = 256
        mlp = WarpMLP([48, 128, 128, 128, 1], activation="elu", device=device, output_gain=1.0)
        mlp.alloc_intermediates(batch)
        optimizer = AdamW(mlp.parameters(), lr=3e-4)

        obs = wp.array(
            np.random.default_rng(42).standard_normal((batch, 48)).astype(np.float32),
            device=device,
            requires_grad=True,
        )
        target = wp.array(np.full((batch, 1), 36.0, dtype=np.float32), device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        for step in range(100):
            loss.zero_()
            tape = wp.Tape()
            with tape:
                out = mlp.forward(obs)
                wp.launch(_mse_loss_kernel, dim=batch, inputs=[out, target, loss, batch], device=device)
            tape.backward(loss)

            grads = mlp.grad_arrays()
            has_nan_grad = any(np.any(np.isnan(g.numpy())) for g in grads)
            loss_val = loss.numpy()[0]

            self.assertFalse(np.isnan(loss_val), f"Loss is NaN at step {step}")
            self.assertFalse(has_nan_grad, f"Gradient is NaN at step {step}")

            optimizer.step(grads)
            tape.zero()

        # Loss should have decreased
        final_loss = loss.numpy()[0]
        self.assertLess(final_loss, 650.0, "Loss should decrease from initial ~650")


class TestPPOLossKernelGradients(unittest.TestCase):
    """Test the PPO loss kernel for numerical stability."""

    def test_fused_kernel_large_advantages(self):
        """Fused PPO kernel should not NaN with large advantage values."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        batch = 256
        act_dim = 12

        ac = ActorCritic(obs_dim=48, act_dim=act_dim, hidden_sizes=[128, 128, 128], device=device, seed=42)
        ac.alloc_buffers(rollout_batch=batch, minibatch_size=batch)

        rng = np.random.default_rng(42)
        obs = wp.array(rng.standard_normal((batch, 48)).astype(np.float32), device=device, requires_grad=True)
        actions = wp.array(rng.standard_normal((batch, act_dim)).astype(np.float32) * 0.5, device=device)
        old_lp = wp.array(rng.standard_normal(batch).astype(np.float32) * 2 - 10, device=device)
        advantages = wp.array(rng.standard_normal(batch).astype(np.float32) * 10 + 20, device=device)
        returns = wp.array(np.full(batch, 36.0, dtype=np.float32), device=device)

        loss = wp.zeros(1, dtype=wp.float32, device=device)
        log_alpha = wp.array([np.log(0.01).astype(np.float32)], dtype=wp.float32, device=device)
        grad_mean = wp.zeros((batch, act_dim), dtype=wp.float32, device=device)
        grad_values = wp.zeros(batch, dtype=wp.float32, device=device)
        grad_log_std = wp.zeros(act_dim, dtype=wp.float32, device=device)
        grad_log_alpha = wp.zeros(1, dtype=wp.float32, device=device)

        tape = wp.Tape()
        with tape:
            mean = ac._actor_mb.forward(obs)
            values_2d = ac._critic_mb.forward(obs)

        wp.launch(
            _ppo_fused_loss_and_grad_kernel,
            dim=batch,
            inputs=[
                mean,
                ac.log_std,
                actions,
                old_lp,
                advantages,
                values_2d.flatten(),
                returns,
                log_alpha,
                0.2,
                0.5,
                ac._use_tanh_int,
                batch,
                act_dim,
                loss,
                grad_mean,
                grad_values,
                grad_log_std,
                grad_log_alpha,
            ],
            device=device,
        )
        tape.backward(grads={mean: grad_mean, values_2d: grad_values.reshape((batch, 1))})

        loss_val = loss.numpy()[0]
        has_nan_grad = any(np.any(np.isnan(w.grad.numpy())) for w in ac.actor.weights + ac.critic.weights)

        print(f"Fused PPO with large advantages: loss={loss_val:.4f}, nan_grad={has_nan_grad}")
        self.assertFalse(np.isnan(loss_val), "PPO loss should not be NaN")
        self.assertFalse(has_nan_grad, "PPO gradients should not be NaN")

    def test_ppo_update_multiple_epochs_same_data(self):
        """Multiple PPO epochs on same data should not produce NaN.

        This reproduces the failure mode seen in ANYmal training where
        the loss goes NaN after ~20 PPO updates.
        """
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        num_envs = 256
        obs_dim = 48
        act_dim = 12
        num_steps = 24

        ac = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=[128, 128, 128],
            activation="elu",
            init_log_std=-1.0,
            device=device,
            seed=42,
        )
        trainer = PPOTrainer(
            ac,
            num_envs=num_envs,
            lr=3e-4,
            num_steps=num_steps,
            num_epochs=5,
            num_minibatches=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=1.0,
        )

        # Fill buffer with realistic ANYmal-like data
        buf = trainer.buffer
        rng = np.random.default_rng(42)
        for t in range(num_steps):
            obs = wp.array(rng.standard_normal((num_envs, obs_dim)).astype(np.float32) * 3, device=device)
            actions = wp.array(rng.standard_normal((num_envs, act_dim)).astype(np.float32) * 0.5, device=device)
            log_probs = wp.array(rng.standard_normal(num_envs).astype(np.float32) * 2 - 10, device=device)
            rewards = wp.array((rng.standard_normal(num_envs).astype(np.float32) * 0.3 + 2.0), device=device)
            dones = wp.zeros(num_envs, dtype=wp.float32, device=device)
            values = wp.array(np.full(num_envs, 0.24, dtype=np.float32), device=device)
            buf.insert(t, obs, actions, actions, log_probs, rewards, dones, values)

        last_vals = wp.array(np.full(num_envs, 0.24, dtype=np.float32), device=device)
        buf.compute_gae(last_vals, 0.99, 0.95)

        trainer.update()
        stats = trainer.get_stats()
        has_nan_loss = np.isnan(stats["loss"])
        has_nan_params = any(np.any(np.isnan(p.numpy())) for p in ac.parameters())

        print(
            f"PPO update on ANYmal-like data: loss={stats['loss']:.4f}, nan_loss={has_nan_loss}, nan_params={has_nan_params}"
        )
        self.assertFalse(has_nan_loss, f"PPO update produced NaN loss: {stats['loss']}")
        self.assertFalse(has_nan_params, "PPO update produced NaN parameters")


if __name__ == "__main__":
    unittest.main()
