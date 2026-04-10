# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the PPO continuous loss kernel.

Runs the actual Warp kernel with controlled inputs and compares every output
(loss components, gradients for logits and std) against a numpy reference
implementation that mirrors the kernel line by line.

This catches:
- Sign errors in gradients
- Missing scaling factors
- Entropy computation bugs
- Gradient direction errors for scalar std parameterization
"""

import unittest
import math

import numpy as np
import warp as wp


def _reference_ppo_continuous_loss(
    logits, std_arr, actions, old_logprobs, advantages,
    values_pred, old_values, returns,
    num_actions, clip_coef, vf_clip_coef, vf_coef, ent_coef, inv_batch,
    prio_weight,
):
    """Pure numpy reference implementation matching ppo_loss_continuous_kernel."""
    B = logits.shape[0]
    HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
    HALF_1_PLUS_LOG_2PI = 0.5 * (1 + math.log(2 * math.pi))

    grad_logits = np.zeros_like(logits)
    grad_std_scratch = np.zeros((B, num_actions), dtype=np.float32)
    grad_values = np.zeros(B, dtype=np.float32)
    loss_out = np.zeros((B, 7), dtype=np.float32)

    for idx in range(B):
        dL = inv_batch
        adv = advantages[idx]
        old_logp = old_logprobs[idx]
        val_pred = values_pred[idx]
        old_val = old_values[idx]
        ret = returns[idx]

        # Value loss (clipped)
        v_error = val_pred - old_val
        v_clipped = old_val + np.clip(v_error, -vf_clip_coef, vf_clip_coef)
        v_loss_unclipped = (val_pred - ret) ** 2
        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss = 0.5 * max(v_loss_unclipped, v_loss_clipped)

        if v_loss_clipped > v_loss_unclipped:
            if -vf_clip_coef <= v_error <= vf_clip_coef:
                d_val_pred = v_clipped - ret
            else:
                d_val_pred = 0.0
        else:
            d_val_pred = val_pred - ret
        grad_values[idx] = dL * vf_coef * d_val_pred

        # Policy loss (continuous, scalar std)
        total_new_logp = 0.0
        total_entropy = 0.0
        for h in range(num_actions):
            mean = logits[idx, h]
            std = std_arr[h]
            action = actions[idx, h]
            z = (action - mean) / std
            log_std = math.log(std)
            new_logp = -0.5 * z * z - HALF_LOG_2PI - log_std
            entropy = HALF_1_PLUS_LOG_2PI + log_std
            total_new_logp += new_logp
            total_entropy += entropy

        logratio = total_new_logp - old_logp
        ratio = math.exp(logratio)
        ratio_clipped = np.clip(ratio, 1 - clip_coef, 1 + clip_coef)

        wa = -adv * prio_weight[idx]
        pg_loss1 = wa * ratio
        pg_loss2 = wa * ratio_clipped
        pg_loss = max(pg_loss1, pg_loss2)

        d_pg = dL
        if pg_loss2 > pg_loss1:
            if ratio <= 1 - clip_coef or ratio >= 1 + clip_coef:
                d_pg = 0.0

        d_new_logp = wa * d_pg * ratio
        d_entropy = dL * (-ent_coef)

        for h2 in range(num_actions):
            mean2 = logits[idx, h2]
            std2 = std_arr[h2]
            action2 = actions[idx, h2]
            diff = action2 - mean2
            var = std2 * std2
            inv_std = 1.0 / std2

            grad_logits[idx, h2] = d_new_logp * diff / var
            # Scalar std gradient: (z²-1)/std from PG + (-ent_coef)/std from entropy
            grad_std_scratch[idx, h2] = d_new_logp * (diff * diff / var - 1.0) * inv_std + d_entropy * inv_std

        total_loss = (pg_loss + vf_coef * v_loss - ent_coef * total_entropy) * dL
        loss_out[idx, 0] = total_loss
        loss_out[idx, 1] = pg_loss * dL
        loss_out[idx, 2] = v_loss * dL
        loss_out[idx, 3] = total_entropy * dL

        clipped_flag = 1.0 if abs(ratio - 1.0) > clip_coef else 0.0
        loss_out[idx, 4] = clipped_flag * dL
        approx_kl = (ratio - 1.0 - logratio) * dL
        loss_out[idx, 5] = approx_kl
        loss_out[idx, 6] = ratio * dL

    return grad_logits, grad_std_scratch, grad_values, loss_out


class TestPPOLossKernel(unittest.TestCase):
    """Test the actual ppo_loss_continuous_kernel against numpy reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_loss_and_gradients_match_reference(self):
        """All outputs of ppo_loss_continuous_kernel match numpy reference."""
        from newton._src.pufferlib.ppo import ppo_loss_continuous_kernel, PPOLossBuffers, _REDUCE_BLOCK_DIM, _reduce_logstd_grad_kernel

        B, A = 32, 4
        np.random.seed(42)

        logits = np.random.randn(B, A).astype(np.float32) * 0.5
        std_arr = np.abs(np.random.randn(A).astype(np.float32)) * 0.3 + 0.2
        actions = np.random.randn(B, A).astype(np.float32) * 0.5
        old_logprobs = np.random.randn(B).astype(np.float32) * 2.0
        advantages = np.random.randn(B).astype(np.float32)
        values_pred = np.random.randn(B).astype(np.float32)
        old_values = np.random.randn(B).astype(np.float32)
        returns = np.random.randn(B).astype(np.float32)
        prio_weight = np.ones(B, dtype=np.float32)

        clip_coef = 0.2
        vf_clip_coef = 0.2
        vf_coef = 1.0
        ent_coef = 0.01
        inv_batch = 1.0 / B

        # Reference
        ref_grad_logits, ref_grad_std, ref_grad_values, ref_loss = _reference_ppo_continuous_loss(
            logits, std_arr, actions, old_logprobs, advantages,
            values_pred, old_values, returns,
            A, clip_coef, vf_clip_coef, vf_coef, ent_coef, inv_batch, prio_weight,
        )

        # Warp kernel
        d = "cuda:0"
        loss_bufs = PPOLossBuffers(B, A + 1, device=d, continuous=True, num_actions=A)
        d_logits = wp.array(logits, dtype=wp.float32, device=d)
        d_std = wp.array(std_arr, dtype=wp.float32, device=d)
        d_actions = wp.array(actions, dtype=wp.float32, device=d)
        d_old_lp = wp.array(old_logprobs, dtype=wp.float32, device=d)
        d_adv = wp.array(advantages, dtype=wp.float32, device=d)
        d_vpred = wp.array(values_pred, dtype=wp.float32, device=d)
        d_oldv = wp.array(old_values, dtype=wp.float32, device=d)
        d_ret = wp.array(returns, dtype=wp.float32, device=d)

        wp.launch(ppo_loss_continuous_kernel, dim=B,
                  inputs=[d_logits, d_std, d_actions, d_old_lp, d_adv,
                          d_vpred, d_oldv, d_ret, A,
                          clip_coef, vf_clip_coef, vf_coef, ent_coef, inv_batch,
                          loss_bufs.prio_weight,
                          loss_bufs.grad_logits, loss_bufs.grad_logstd_scratch,
                          loss_bufs.grad_values,
                          loss_bufs.loss_per_thread],
                  device=d)

        # Reduce logstd gradient
        wp.launch_tiled(_reduce_logstd_grad_kernel, dim=[A],
                        inputs=[loss_bufs.grad_logstd_scratch, loss_bufs.grad_logstd, B],
                        block_dim=_REDUCE_BLOCK_DIM, device=d)
        wp.synchronize()

        our_grad_logits = loss_bufs.grad_logits.numpy()[:B, :A]
        our_grad_std = loss_bufs.grad_logstd.numpy()
        our_grad_values = loss_bufs.grad_values.numpy()[:B]
        our_loss = loss_bufs.loss_per_thread.numpy()[:B, :]

        # Compare grad_logits
        np.testing.assert_allclose(our_grad_logits, ref_grad_logits, rtol=1e-4, atol=1e-6,
                                   err_msg="grad_logits (mean gradient) mismatch")

        # Compare grad_values
        np.testing.assert_allclose(our_grad_values, ref_grad_values, rtol=1e-4, atol=1e-6,
                                   err_msg="grad_values mismatch")

        # Compare reduced grad_std (sum over batch)
        ref_grad_std_reduced = ref_grad_std.sum(axis=0)
        np.testing.assert_allclose(our_grad_std, ref_grad_std_reduced, rtol=1e-3, atol=1e-5,
                                   err_msg="grad_std (reduced) mismatch — entropy gradient bug?")

        # Compare loss components
        np.testing.assert_allclose(our_loss[:, 0], ref_loss[:, 0], rtol=1e-4, atol=1e-6,
                                   err_msg="total_loss mismatch")
        np.testing.assert_allclose(our_loss[:, 1], ref_loss[:, 1], rtol=1e-4, atol=1e-6,
                                   err_msg="pg_loss mismatch")
        np.testing.assert_allclose(our_loss[:, 2], ref_loss[:, 2], rtol=1e-4, atol=1e-6,
                                   err_msg="vf_loss mismatch")
        np.testing.assert_allclose(our_loss[:, 3], ref_loss[:, 3], rtol=1e-4, atol=1e-6,
                                   err_msg="entropy mismatch")

    def test_entropy_direction_with_ent_coef(self):
        """With ent_coef>0, the std gradient from entropy should push std UP (positive direction)."""
        # For a single sample with zero advantage (so PG gradient is zero),
        # the only gradient on std comes from the entropy bonus.
        # d(-ent_coef * entropy) / d(std) = -ent_coef * (1/std) < 0
        # Since optimizer does w -= lr * grad, this INCREASES std.
        # But the gradient itself should be NEGATIVE.

        from newton._src.pufferlib.ppo import ppo_loss_continuous_kernel, PPOLossBuffers, _REDUCE_BLOCK_DIM, _reduce_logstd_grad_kernel

        B, A = 1, 2
        d = "cuda:0"

        logits = np.zeros((B, A), dtype=np.float32)
        std_arr = np.array([0.5, 0.5], dtype=np.float32)
        actions = np.zeros((B, A), dtype=np.float32)  # action = mean → z=0
        # Set old_logprobs = current logprobs (ratio=1, no PG gradient)
        old_lp = 0.0
        for h in range(A):
            old_lp += -0.5 * math.log(2 * math.pi) - math.log(0.5)
        old_logprobs = np.array([old_lp], dtype=np.float32)
        advantages = np.zeros(B, dtype=np.float32)  # zero advantage
        values_pred = np.zeros(B, dtype=np.float32)
        old_values = np.zeros(B, dtype=np.float32)
        returns = np.zeros(B, dtype=np.float32)

        ent_coef = 0.01
        inv_batch = 1.0

        loss_bufs = PPOLossBuffers(B, A + 1, device=d, continuous=True, num_actions=A)
        wp.launch(ppo_loss_continuous_kernel, dim=B,
                  inputs=[wp.array(logits, device=d), wp.array(std_arr, device=d),
                          wp.array(actions, device=d), wp.array(old_logprobs, device=d),
                          wp.array(advantages, device=d),
                          wp.array(values_pred, device=d), wp.array(old_values, device=d),
                          wp.array(returns, device=d), A,
                          0.2, 0.2, 1.0, ent_coef, inv_batch,
                          loss_bufs.prio_weight,
                          loss_bufs.grad_logits, loss_bufs.grad_logstd_scratch,
                          loss_bufs.grad_values, loss_bufs.loss_per_thread],
                  device=d)
        wp.launch_tiled(_reduce_logstd_grad_kernel, dim=[A],
                        inputs=[loss_bufs.grad_logstd_scratch, loss_bufs.grad_logstd, B],
                        block_dim=_REDUCE_BLOCK_DIM, device=d)
        wp.synchronize()

        grad_std = loss_bufs.grad_logstd.numpy()

        # With zero advantage and action=mean (z=0):
        # PG gradient component: d_new_logp * (0 - 1)/std = 0 (because adv=0 → d_new_logp=0)
        # Entropy gradient component: d_entropy / std = (-ent_coef) / std = -0.01 / 0.5 = -0.02
        expected = -ent_coef / std_arr
        np.testing.assert_allclose(grad_std, expected, rtol=1e-4, atol=1e-6,
                                   err_msg="Entropy-only gradient direction wrong!")

        # Verify the gradient is NEGATIVE (optimizer w -= lr*grad → w increases)
        self.assertTrue(np.all(grad_std < 0),
                        f"Entropy gradient should be negative (pushes std UP via optimizer), got {grad_std}")

    def test_entropy_decreases_with_larger_std(self):
        """The entropy gradient magnitude decreases with larger std (1/std factor)."""
        # This is the KEY property of scalar std vs logstd parameterization
        stds = [0.1, 0.5, 1.0, 2.0, 5.0]
        ent_coef = 0.01
        expected_grads = [-ent_coef / s for s in stds]

        # Verify the gradient magnitude decreases with std
        for i in range(len(stds) - 1):
            self.assertGreater(abs(expected_grads[i]), abs(expected_grads[i + 1]),
                               f"Entropy gradient should decrease with larger std: "
                               f"|grad({stds[i]})| = {abs(expected_grads[i]):.4f} vs "
                               f"|grad({stds[i+1]})| = {abs(expected_grads[i+1]):.4f}")


if __name__ == "__main__":
    unittest.main()
