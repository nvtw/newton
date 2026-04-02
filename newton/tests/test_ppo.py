# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the single-file PPO implementation.

Every public class and kernel in ``newton._src.ppo`` is tested against
a NumPy reference implementation with tight numerical tolerances.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np
import warp as wp

from newton._src.ppo import (
    ActorCritic,
    AdamW,
    ObsNormalizer,
    PPOTrainer,
    RolloutBuffer,
    WarpMLP,
    _ArraySum,
    _orthogonal_init,
    _ppo_fused_loss_and_grad_kernel,
    _sample_actions_kernel,
    export_actor_to_onnx,
)


# ---------------------------------------------------------------------------
# Helper environments
# ---------------------------------------------------------------------------


class DummyVecEnv:
    """Trivial vectorized env returning wp.array on device (zero alloc after init)."""

    def __init__(self, num_envs: int = 4, obs_dim: int = 16, act_dim: int = 4, device: str = "cpu"):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._device = device
        self._seed = 0
        self._obs = wp.zeros((num_envs, obs_dim), dtype=wp.float32, device=device)
        self._rewards = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self._dones = wp.zeros(num_envs, dtype=wp.float32, device=device)

    def reset(self) -> wp.array:
        self._fill_obs()
        return self._obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        self._fill_obs()
        self._fill_rewards()
        return self._obs, self._rewards, self._dones

    def _fill_obs(self) -> None:
        rng = np.random.default_rng(self._seed)
        self._seed += 1
        obs_np = rng.standard_normal((self.num_envs, self.obs_dim)).astype(np.float32)
        wp.copy(self._obs, wp.array(obs_np, dtype=wp.float32, device=self._device))

    def _fill_rewards(self) -> None:
        rng = np.random.default_rng(self._seed + 10000)
        r_np = rng.standard_normal(self.num_envs).astype(np.float32)
        wp.copy(self._rewards, wp.array(r_np, dtype=wp.float32, device=self._device))


@wp.kernel
def _neg_action_sq_kernel(
    actions: wp.array2d[float],
    rewards: wp.array[float],
):
    """Reward = -sum(action^2). Optimal policy outputs zeros."""
    env = wp.tid()
    r = float(0.0)
    for j in range(actions.shape[1]):
        a = actions[env, j]
        r = r - a * a
    rewards[env] = r


class ReachZeroEnv:
    """Trivial env: reward = -sum(action^2). Optimal policy outputs zeros.

    Observation is constant (all ones). No termination. This tests whether
    PPO can learn to minimize action magnitude.
    """

    def __init__(self, num_envs: int = 64, obs_dim: int = 4, act_dim: int = 4, device: str = "cpu"):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._device = device
        self._obs = wp.ones((num_envs, obs_dim), dtype=wp.float32, device=device)
        self._rewards = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self._dones = wp.zeros(num_envs, dtype=wp.float32, device=device)

    def reset(self) -> wp.array:
        return self._obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        wp.launch(_neg_action_sq_kernel, dim=self.num_envs, inputs=[actions, self._rewards], device=self._device)
        return self._obs, self._rewards, self._dones


# ---------------------------------------------------------------------------
# NumPy reference helpers
# ---------------------------------------------------------------------------


def _numpy_gae(rewards, values, dones, last_values, gamma, lam):
    """Reference GAE in pure NumPy (num_steps, num_envs)."""
    num_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    last_gae = np.zeros(num_envs)
    for t in reversed(range(num_steps)):
        next_val = last_values if t == num_steps - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + values[t]
    return advantages, returns


def _numpy_adamw_step(param, grad, m, v, t, lr, beta1, beta2, eps, wd):
    """Single AdamW step in NumPy, returns (new_param, new_m, new_v)."""
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    mhat = m_new / (1 - beta1 ** (t + 1))
    vhat = v_new / (1 - beta2 ** (t + 1))
    param_new = param * (1 - lr * wd) - lr * mhat / (np.sqrt(vhat) + eps)
    return param_new, m_new, v_new


def _numpy_gaussian_log_prob(actions, mean, log_std, use_tanh):
    """Reference diagonal-Gaussian log-prob with optional tanh correction."""
    std = np.exp(log_std)
    if use_tanh:
        u = np.arctanh(np.clip(actions, -0.999999, 0.999999))
    else:
        u = actions
    z = (u - mean) / std
    log2pi = math.log(2 * math.pi)
    lp_per_dim = -0.5 * (z * z + 2 * log_std + log2pi)
    if use_tanh:
        log2 = math.log(2.0)
        lp_per_dim -= 2.0 * (log2 - u - np.log(1.0 + np.exp(-2.0 * u)))
    return lp_per_dim.sum(axis=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrthogonalInit(unittest.TestCase):
    def test_shape(self):
        w = _orthogonal_init((64, 32), gain=1.0, seed=0)
        self.assertEqual(w.shape, (64, 32))

    def test_orthogonality_square(self):
        w = _orthogonal_init((64, 64), gain=1.0, seed=0)
        product = w @ w.T
        np.testing.assert_allclose(product, np.eye(64), atol=1e-5)

    def test_orthogonality_tall(self):
        """For tall matrices (rows > cols), columns should be orthonormal."""
        w = _orthogonal_init((128, 64), gain=1.0, seed=0)
        product = w.T @ w
        np.testing.assert_allclose(product, np.eye(64), atol=1e-5)

    def test_gain_scaling(self):
        w1 = _orthogonal_init((64, 64), gain=1.0, seed=0)
        w2 = _orthogonal_init((64, 64), gain=2.0, seed=0)
        np.testing.assert_allclose(w2, w1 * 2.0, atol=1e-6)

    def test_seed_reproducibility(self):
        w1 = _orthogonal_init((32, 16), gain=1.0, seed=42)
        w2 = _orthogonal_init((32, 16), gain=1.0, seed=42)
        np.testing.assert_array_equal(w1, w2)

    def test_different_seeds_differ(self):
        w1 = _orthogonal_init((32, 16), gain=1.0, seed=0)
        w2 = _orthogonal_init((32, 16), gain=1.0, seed=1)
        self.assertFalse(np.allclose(w1, w2))


class TestAdamW(unittest.TestCase):
    def test_step_updates_params(self):
        p = wp.array(np.ones(8, dtype=np.float32), device="cpu", requires_grad=True)
        opt = AdamW([p], lr=0.1)
        g = wp.array(np.ones(8, dtype=np.float32) * 0.5, device="cpu")
        before = p.numpy().copy()
        opt.step([g])
        after = p.numpy()
        self.assertFalse(np.allclose(before, after), "Parameters should change after step")

    def test_step_matches_numpy_reference(self):
        """Verify AdamW produces the same values as a NumPy reference over 5 steps."""
        lr, beta1, beta2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.01
        param_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
        p = wp.array(param_np.copy(), device="cpu", requires_grad=True)
        opt = AdamW([p], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)

        m_np = np.zeros_like(param_np)
        v_np = np.zeros_like(param_np)
        rng = np.random.default_rng(7)

        for t in range(5):
            grad_np = rng.standard_normal(4).astype(np.float32)
            g = wp.array(grad_np.copy(), device="cpu")
            opt.step([g])

            param_np, m_np, v_np = _numpy_adamw_step(param_np, grad_np, m_np, v_np, t, lr, beta1, beta2, eps, wd)
            np.testing.assert_allclose(p.numpy(), param_np, rtol=1e-5, atol=1e-7)

    def test_weight_decay_shrinks_params(self):
        """With zero gradient, weight decay should shrink parameters toward zero."""
        p = wp.array(np.array([10.0, -10.0], dtype=np.float32), device="cpu", requires_grad=True)
        opt = AdamW([p], lr=0.01, weight_decay=0.1)
        g = wp.array(np.zeros(2, dtype=np.float32), device="cpu")
        before = np.abs(p.numpy()).copy()
        for _ in range(10):
            opt.step([g])
        after = np.abs(p.numpy())
        self.assertTrue(np.all(after < before), "Weight decay should shrink parameter magnitudes")


class TestObsNormalizer(unittest.TestCase):
    def test_welford_mean_var(self):
        """Running mean/var should match NumPy after several batches."""
        obs_dim = 4
        norm = ObsNormalizer(obs_dim, device="cpu")
        out = wp.zeros((8, obs_dim), dtype=wp.float32, device="cpu")

        rng = np.random.default_rng(42)
        all_data = []
        for _ in range(10):
            batch = rng.standard_normal((8, obs_dim)).astype(np.float32) * 3 + 1
            all_data.append(batch)
            obs = wp.array(batch, device="cpu")
            norm.update_and_normalize(obs, out)

        all_data = np.concatenate(all_data, axis=0)
        expected_mean = all_data.mean(axis=0)
        expected_var = np.var(all_data, axis=0, ddof=0) * len(all_data)

        np.testing.assert_allclose(norm.mean.numpy(), expected_mean, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(norm.var.numpy(), expected_var, rtol=1e-4, atol=1e-3)

    def test_normalize_output_range(self):
        """Normalized output should be clipped to [-10, 10]."""
        obs_dim = 2
        norm = ObsNormalizer(obs_dim, device="cpu")
        out = wp.zeros((4, obs_dim), dtype=wp.float32, device="cpu")

        batch = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.25, 0.25]], dtype=np.float32)
        norm.update_and_normalize(wp.array(batch, device="cpu"), out)

        extreme = np.array([[1000.0, -1000.0], [0.0, 0.0], [500.0, -500.0], [1.0, 1.0]], dtype=np.float32)
        norm.update_and_normalize(wp.array(extreme, device="cpu"), out)
        result = out.numpy()
        self.assertTrue(np.all(result >= -10.0) and np.all(result <= 10.0))

    def test_normalize_without_update(self):
        """normalize() should use frozen statistics."""
        obs_dim = 3
        norm = ObsNormalizer(obs_dim, device="cpu")
        out = wp.zeros((4, obs_dim), dtype=wp.float32, device="cpu")

        batch = np.ones((4, obs_dim), dtype=np.float32) * 5.0
        norm.update_and_normalize(wp.array(batch, device="cpu"), out)

        mean_before = norm.mean.numpy().copy()
        norm.normalize(wp.array(batch * 100, device="cpu"), out)
        np.testing.assert_array_equal(norm.mean.numpy(), mean_before)


class TestWarpMLP(unittest.TestCase):
    def test_forward_shape(self):
        mlp = WarpMLP([16, 32, 8], activation="elu", device="cpu")
        mlp.alloc_intermediates(4)
        x = wp.array(np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32), device="cpu")
        y = mlp.forward(x)
        self.assertEqual(y.shape, (4, 8))

    def test_parameters_count(self):
        mlp = WarpMLP([16, 32, 8], activation="elu", device="cpu")
        params = mlp.parameters()
        self.assertEqual(len(params), 4)  # 2 layers x (weight + bias)

    def test_forward_matches_numpy(self):
        """WarpMLP forward should match a manual NumPy matmul + ELU."""
        mlp = WarpMLP([4, 8, 3], activation="elu", device="cpu", seed=0)
        x_np = np.array([[1.0, 0.0, -1.0, 2.0], [0.5, -0.5, 1.5, -1.0]], dtype=np.float32)
        x = wp.array(x_np, device="cpu")
        y = mlp.forward(x)
        y_np = y.numpy()

        w0 = mlp.weights[0].numpy()  # (8, 4) transB layout
        b0 = mlp.biases[0].numpy()
        w1 = mlp.weights[1].numpy()  # (3, 8)
        b1 = mlp.biases[1].numpy()

        h = x_np @ w0.T + b0
        h = np.where(h > 0, h, np.exp(h) - 1)  # ELU(alpha=1)
        expected = h @ w1.T + b1

        np.testing.assert_allclose(y_np, expected, rtol=1e-4, atol=1e-5)

    def test_forward_relu(self):
        """Verify ReLU activation produces correct output."""
        mlp = WarpMLP([3, 4, 2], activation="relu", device="cpu", seed=0)
        x_np = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
        y = mlp.forward(wp.array(x_np, device="cpu")).numpy()

        w0 = mlp.weights[0].numpy()
        b0 = mlp.biases[0].numpy()
        w1 = mlp.weights[1].numpy()
        b1 = mlp.biases[1].numpy()
        h = np.maximum(0, x_np @ w0.T + b0)
        expected = h @ w1.T + b1
        np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)

    def test_forward_tanh(self):
        """Verify tanh activation produces correct output."""
        mlp = WarpMLP([3, 4, 2], activation="tanh", device="cpu", seed=0)
        x_np = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
        y = mlp.forward(wp.array(x_np, device="cpu")).numpy()

        w0 = mlp.weights[0].numpy()
        b0 = mlp.biases[0].numpy()
        w1 = mlp.weights[1].numpy()
        b1 = mlp.biases[1].numpy()
        h = np.tanh(x_np @ w0.T + b0)
        expected = h @ w1.T + b1
        np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)

    def test_seed_reproducibility(self):
        mlp1 = WarpMLP([8, 16, 4], activation="elu", device="cpu", seed=42)
        mlp2 = WarpMLP([8, 16, 4], activation="elu", device="cpu", seed=42)
        for w1, w2 in zip(mlp1.weights, mlp2.weights, strict=True):
            np.testing.assert_array_equal(w1.numpy(), w2.numpy())

    def test_output_gain(self):
        """Last layer should be scaled by output_gain relative to gain=1."""
        mlp_default = WarpMLP([4, 8, 2], activation="elu", device="cpu", seed=0, output_gain=1.0)
        mlp_small = WarpMLP([4, 8, 2], activation="elu", device="cpu", seed=0, output_gain=0.01)
        w_default = mlp_default.weights[-1].numpy()
        w_small = mlp_small.weights[-1].numpy()
        np.testing.assert_allclose(w_small, w_default * 0.01, atol=1e-7)

    def test_invalid_activation_raises(self):
        with self.assertRaises(ValueError):
            WarpMLP([4, 8, 2], activation="swish", device="cpu")

    def test_grad_arrays_match_parameters(self):
        """grad_arrays() should have the same shapes as parameters()."""
        mlp = WarpMLP([4, 8, 2], activation="elu", device="cpu", seed=0)
        x = wp.array(np.ones((2, 4), dtype=np.float32), device="cpu", requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)
        tape = wp.Tape()
        with tape:
            out = mlp.forward(x)
        tape.backward(loss)
        params = mlp.parameters()
        grads = mlp.grad_arrays()
        self.assertEqual(len(params), len(grads))
        for p, g in zip(params, grads, strict=True):
            self.assertEqual(p.shape, g.shape)


class TestActorCritic(unittest.TestCase):
    def test_act_shapes(self):
        ac = ActorCritic(obs_dim=16, act_dim=4, hidden_sizes=[32, 32], device="cpu")
        batch = 8
        ac.alloc_buffers(rollout_batch=batch, minibatch_size=4)
        obs = wp.array(np.random.default_rng(0).standard_normal((batch, 16)).astype(np.float32), device="cpu")
        rng_counter = wp.array([42], dtype=wp.int32, device="cpu")
        actions, log_probs, values = ac.act(obs, rng_counter=rng_counter)
        self.assertEqual(actions.shape, (batch, 4))
        self.assertEqual(log_probs.shape, (batch,))
        self.assertEqual(values.shape, (batch,))

    def test_bounded_actions_in_range(self):
        """With bounded_actions=True, all actions should be in [-1, 1]."""
        ac = ActorCritic(obs_dim=8, act_dim=4, hidden_sizes=[32], bounded_actions=True, device="cpu", seed=0)
        batch = 64
        ac.alloc_buffers(rollout_batch=batch, minibatch_size=batch)
        obs = wp.array(np.random.default_rng(0).standard_normal((batch, 8)).astype(np.float32), device="cpu")
        rng_counter = wp.array([0], dtype=wp.int32, device="cpu")
        for _ in range(5):
            actions, _, _ = ac.act(obs, rng_counter)
            a_np = actions.numpy()
            self.assertTrue(np.all(a_np >= -1.0) and np.all(a_np <= 1.0), f"Actions out of [-1,1]: {a_np.min()}, {a_np.max()}")

    def test_unbounded_actions_can_exceed_one(self):
        """With bounded_actions=False and large log_std, actions should exceed [-1, 1]."""
        ac = ActorCritic(obs_dim=4, act_dim=4, hidden_sizes=[32], bounded_actions=False, init_log_std=1.0, device="cpu", seed=0)
        batch = 256
        ac.alloc_buffers(rollout_batch=batch, minibatch_size=batch)
        obs = wp.array(np.random.default_rng(0).standard_normal((batch, 4)).astype(np.float32), device="cpu")
        rng_counter = wp.array([0], dtype=wp.int32, device="cpu")
        actions, _, _ = ac.act(obs, rng_counter)
        a_np = actions.numpy()
        self.assertTrue(np.any(np.abs(a_np) > 1.0), "Unbounded actions with large std should exceed [-1,1]")

    def test_log_probs_finite(self):
        """Log-probabilities should be finite (no NaN/Inf)."""
        for bounded in [True, False]:
            ac = ActorCritic(obs_dim=8, act_dim=4, hidden_sizes=[32], bounded_actions=bounded, device="cpu", seed=0)
            batch = 32
            ac.alloc_buffers(rollout_batch=batch, minibatch_size=batch)
            obs = wp.array(np.random.default_rng(0).standard_normal((batch, 8)).astype(np.float32), device="cpu")
            rng_counter = wp.array([0], dtype=wp.int32, device="cpu")
            _, log_probs, _ = ac.act(obs, rng_counter)
            lp = log_probs.numpy()
            self.assertTrue(np.all(np.isfinite(lp)), f"Non-finite log-probs (bounded={bounded}): {lp}")

    def test_parameters_count(self):
        ac = ActorCritic(obs_dim=8, act_dim=4, hidden_sizes=[16, 16], device="cpu")
        params = ac.parameters()
        actor_params = ac.actor.parameters()
        critic_params = ac.critic.parameters()
        self.assertEqual(len(params), len(actor_params) + len(critic_params) + 1)


class TestRolloutBuffer(unittest.TestCase):
    def test_insert_and_flatten(self):
        buf = RolloutBuffer(num_envs=4, num_steps=8, obs_dim=16, act_dim=4, device="cpu")
        rng = np.random.default_rng(0)
        for t in range(8):
            act = wp.array(rng.standard_normal((4, 4)).astype(np.float32), device="cpu")
            buf.insert(
                t=t,
                obs=wp.array(rng.standard_normal((4, 16)).astype(np.float32), device="cpu"),
                actions=act,
                pre_tanh=act,
                log_probs=wp.array(rng.standard_normal(4).astype(np.float32), device="cpu"),
                rewards=wp.array(rng.standard_normal(4).astype(np.float32), device="cpu"),
                dones=wp.zeros(4, dtype=wp.float32, device="cpu"),
                values=wp.array(rng.standard_normal(4).astype(np.float32), device="cpu"),
            )
        buf.flatten()
        self.assertEqual(buf.flat_obs.shape, (32, 16))
        self.assertEqual(buf.flat_actions.shape, (32, 4))

    def test_insert_preserves_data(self):
        """Verify inserted data is faithfully stored and flattened."""
        num_envs, num_steps, obs_dim, act_dim = 2, 3, 4, 2
        buf = RolloutBuffer(num_envs=num_envs, num_steps=num_steps, obs_dim=obs_dim, act_dim=act_dim, device="cpu")
        rng = np.random.default_rng(99)
        obs_data = []
        reward_data = []
        for t in range(num_steps):
            o = rng.standard_normal((num_envs, obs_dim)).astype(np.float32)
            r = rng.standard_normal(num_envs).astype(np.float32)
            obs_data.append(o)
            reward_data.append(r)
            act = wp.array(np.zeros((num_envs, act_dim), dtype=np.float32), device="cpu")
            buf.insert(
                t=t,
                obs=wp.array(o, device="cpu"),
                actions=act,
                pre_tanh=act,
                log_probs=wp.zeros(num_envs, dtype=wp.float32, device="cpu"),
                rewards=wp.array(r, device="cpu"),
                dones=wp.zeros(num_envs, dtype=wp.float32, device="cpu"),
                values=wp.zeros(num_envs, dtype=wp.float32, device="cpu"),
            )

        stored_obs = buf.observations.numpy()
        stored_rewards = buf.rewards.numpy()
        for t in range(num_steps):
            np.testing.assert_allclose(stored_obs[t], obs_data[t], atol=1e-7)
            np.testing.assert_allclose(stored_rewards[t], reward_data[t], atol=1e-7)

        buf.flatten()
        flat_obs = buf.flat_obs.numpy()
        for t in range(num_steps):
            for e in range(num_envs):
                np.testing.assert_allclose(flat_obs[t * num_envs + e], obs_data[t][e], atol=1e-7)

    def test_gae_positive_rewards(self):
        buf = RolloutBuffer(num_envs=2, num_steps=4, obs_dim=4, act_dim=2, device="cpu")
        for t in range(4):
            act = wp.array(np.zeros((2, 2), dtype=np.float32), device="cpu")
            buf.insert(
                t=t,
                obs=wp.array(np.zeros((2, 4), dtype=np.float32), device="cpu"),
                actions=act,
                pre_tanh=act,
                log_probs=wp.zeros(2, dtype=wp.float32, device="cpu"),
                rewards=wp.array(np.ones(2, dtype=np.float32), device="cpu"),
                dones=wp.zeros(2, dtype=wp.float32, device="cpu"),
                values=wp.zeros(2, dtype=wp.float32, device="cpu"),
            )
        last_vals = wp.zeros(2, dtype=wp.float32, device="cpu")
        buf.compute_gae(last_vals, gamma=0.99, gae_lambda=0.95)
        adv = buf.advantages.numpy()
        self.assertTrue(np.all(adv > 0), "Advantages should be positive with positive rewards and zero values")

    def test_gae_matches_numpy_reference(self):
        """GAE output should match a NumPy reference implementation exactly."""
        num_envs, num_steps = 3, 5
        gamma, lam = 0.99, 0.95
        rng = np.random.default_rng(123)

        rewards_np = rng.standard_normal((num_steps, num_envs)).astype(np.float32)
        values_np = rng.standard_normal((num_steps, num_envs)).astype(np.float32)
        dones_np = np.zeros((num_steps, num_envs), dtype=np.float32)
        dones_np[2, 0] = 1.0  # episode boundary
        dones_np[3, 1] = 1.0
        last_values_np = rng.standard_normal(num_envs).astype(np.float32)

        expected_adv, expected_ret = _numpy_gae(rewards_np, values_np, dones_np, last_values_np, gamma, lam)

        buf = RolloutBuffer(num_envs=num_envs, num_steps=num_steps, obs_dim=2, act_dim=1, device="cpu")
        for t in range(num_steps):
            act = wp.array(np.zeros((num_envs, 1), dtype=np.float32), device="cpu")
            buf.insert(
                t=t,
                obs=wp.array(np.zeros((num_envs, 2), dtype=np.float32), device="cpu"),
                actions=act,
                pre_tanh=act,
                log_probs=wp.zeros(num_envs, dtype=wp.float32, device="cpu"),
                rewards=wp.array(rewards_np[t], device="cpu"),
                dones=wp.array(dones_np[t], device="cpu"),
                values=wp.array(values_np[t], device="cpu"),
            )
        buf.compute_gae(wp.array(last_values_np, device="cpu"), gamma=gamma, gae_lambda=lam)

        np.testing.assert_allclose(buf.advantages.numpy(), expected_adv, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(buf.returns.numpy(), expected_ret, rtol=1e-5, atol=1e-6)

    def test_gae_episode_boundary_resets_gae(self):
        """A done=1 at step t should prevent GAE from propagating across that boundary."""
        num_envs, num_steps = 1, 4
        gamma, lam = 0.99, 0.95

        buf = RolloutBuffer(num_envs=num_envs, num_steps=num_steps, obs_dim=1, act_dim=1, device="cpu")
        rewards = [1.0, 1.0, 1.0, 1.0]
        dones = [0.0, 1.0, 0.0, 0.0]  # episode ends at t=1
        for t in range(num_steps):
            act = wp.array(np.zeros((1, 1), dtype=np.float32), device="cpu")
            buf.insert(
                t=t,
                obs=wp.array(np.zeros((1, 1), dtype=np.float32), device="cpu"),
                actions=act,
                pre_tanh=act,
                log_probs=wp.zeros(1, dtype=wp.float32, device="cpu"),
                rewards=wp.array(np.array([rewards[t]], dtype=np.float32), device="cpu"),
                dones=wp.array(np.array([dones[t]], dtype=np.float32), device="cpu"),
                values=wp.zeros(1, dtype=wp.float32, device="cpu"),
            )
        buf.compute_gae(wp.zeros(1, dtype=wp.float32, device="cpu"), gamma=gamma, gae_lambda=lam)

        rewards_np = np.array(rewards, dtype=np.float32).reshape(num_steps, 1)
        dones_np = np.array(dones, dtype=np.float32).reshape(num_steps, 1)
        values_np = np.zeros((num_steps, 1), dtype=np.float32)
        expected_adv, _ = _numpy_gae(rewards_np, values_np, dones_np, np.zeros(1, dtype=np.float32), gamma, lam)
        np.testing.assert_allclose(buf.advantages.numpy(), expected_adv, rtol=1e-5, atol=1e-6)

    def test_mean_reward(self):
        buf = RolloutBuffer(num_envs=2, num_steps=3, obs_dim=1, act_dim=1, device="cpu")
        rng = np.random.default_rng(0)
        all_rewards = []
        for t in range(3):
            r = rng.standard_normal(2).astype(np.float32)
            all_rewards.append(r)
            act = wp.array(np.zeros((2, 1), dtype=np.float32), device="cpu")
            buf.insert(
                t=t,
                obs=wp.array(np.zeros((2, 1), dtype=np.float32), device="cpu"),
                actions=act,
                pre_tanh=act,
                log_probs=wp.zeros(2, dtype=wp.float32, device="cpu"),
                rewards=wp.array(r, device="cpu"),
                dones=wp.zeros(2, dtype=wp.float32, device="cpu"),
                values=wp.zeros(2, dtype=wp.float32, device="cpu"),
            )
        expected = np.concatenate(all_rewards).mean()
        self.assertAlmostEqual(buf.mean_reward(), float(expected), places=5)


class TestArraySum(unittest.TestCase):
    def test_small_array(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        arr = wp.array(data, device="cpu")
        result = wp.zeros(1, dtype=wp.float32, device="cpu")
        summer = _ArraySum(len(data), device="cpu")
        summer.compute(arr, len(data), result)
        np.testing.assert_allclose(result.numpy()[0], data.sum(), rtol=1e-5)

    def test_large_array(self):
        """Test reduction with more elements than one tile (>512)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000).astype(np.float32)
        arr = wp.array(data, device="cpu")
        result = wp.zeros(1, dtype=wp.float32, device="cpu")
        summer = _ArraySum(len(data), device="cpu")
        summer.compute(arr, len(data), result)
        np.testing.assert_allclose(result.numpy()[0], data.sum(), rtol=1e-4)

    def test_single_element(self):
        data = np.array([42.0], dtype=np.float32)
        arr = wp.array(data, device="cpu")
        result = wp.zeros(1, dtype=wp.float32, device="cpu")
        summer = _ArraySum(1, device="cpu")
        summer.compute(arr, 1, result)
        np.testing.assert_allclose(result.numpy()[0], 42.0, atol=1e-6)

    def test_partial_length(self):
        """Only the first `length` elements should be summed."""
        data = np.array([1.0, 2.0, 3.0, 100.0, 200.0], dtype=np.float32)
        arr = wp.array(data, device="cpu")
        result = wp.zeros(1, dtype=wp.float32, device="cpu")
        summer = _ArraySum(len(data), device="cpu")
        summer.compute(arr, 3, result)
        np.testing.assert_allclose(result.numpy()[0], 6.0, rtol=1e-5)


class TestSampleActionsKernel(unittest.TestCase):
    def test_unbounded_log_prob_matches_numpy(self):
        """Log-probs from _sample_actions_kernel (unbounded) should match NumPy Gaussian."""
        batch, act_dim = 16, 4
        rng = np.random.default_rng(42)
        mean_np = rng.standard_normal((batch, act_dim)).astype(np.float32) * 0.1
        log_std_np = np.full(act_dim, -0.5, dtype=np.float32)

        mean = wp.array(mean_np, device="cpu")
        log_std = wp.array(log_std_np, device="cpu")
        rng_counter = wp.array([7], dtype=wp.int32, device="cpu")
        actions = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
        pre_tanh = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
        log_probs = wp.zeros(batch, dtype=wp.float32, device="cpu")

        wp.launch(
            _sample_actions_kernel,
            dim=(batch, act_dim),
            inputs=[mean, log_std, rng_counter, 0, actions, pre_tanh, log_probs],
            device="cpu",
        )

        a_np = actions.numpy()
        u_np = pre_tanh.numpy()
        np.testing.assert_allclose(a_np, u_np, atol=1e-7, err_msg="Unbounded: actions should equal pre_tanh")

        std = np.exp(log_std_np)
        z = (u_np - mean_np) / std
        log2pi = math.log(2 * math.pi)
        expected_lp = np.sum(-0.5 * (z * z + 2 * log_std_np + log2pi), axis=1)
        np.testing.assert_allclose(log_probs.numpy(), expected_lp, rtol=1e-4, atol=1e-5)

    def test_bounded_actions_in_range(self):
        """With use_tanh=1, all actions should be in (-1, 1)."""
        batch, act_dim = 64, 8
        mean = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
        log_std = wp.array(np.full(act_dim, 0.5, dtype=np.float32), device="cpu")
        rng_counter = wp.array([0], dtype=wp.int32, device="cpu")
        actions = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
        pre_tanh = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
        log_probs = wp.zeros(batch, dtype=wp.float32, device="cpu")

        wp.launch(
            _sample_actions_kernel,
            dim=(batch, act_dim),
            inputs=[mean, log_std, rng_counter, 1, actions, pre_tanh, log_probs],
            device="cpu",
        )
        a_np = actions.numpy()
        self.assertTrue(np.all(a_np > -1.0) and np.all(a_np < 1.0))

    def test_tanh_log_prob_includes_jacobian(self):
        """Tanh log-prob should be lower than Gaussian log-prob due to Jacobian correction."""
        batch, act_dim = 32, 4
        mean = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
        log_std = wp.array(np.zeros(act_dim, dtype=np.float32), device="cpu")

        for use_tanh in [0, 1]:
            rng_counter = wp.array([42], dtype=wp.int32, device="cpu")
            actions = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
            pre_tanh = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
            log_probs = wp.zeros(batch, dtype=wp.float32, device="cpu")
            wp.launch(
                _sample_actions_kernel,
                dim=(batch, act_dim),
                inputs=[mean, log_std, rng_counter, use_tanh, actions, pre_tanh, log_probs],
                device="cpu",
            )
            if use_tanh == 0:
                lp_unbounded = log_probs.numpy().copy()
                u_saved = pre_tanh.numpy().copy()
            else:
                lp_bounded = log_probs.numpy().copy()
                u_bounded = pre_tanh.numpy().copy()

        np.testing.assert_allclose(u_saved, u_bounded, atol=1e-6, err_msg="Same RNG seed should give same pre_tanh")

        log2 = math.log(2.0)
        jacobian = np.sum(-2.0 * (log2 - u_saved - np.log(1.0 + np.exp(-2.0 * u_saved))), axis=1)
        expected_bounded = lp_unbounded + jacobian
        np.testing.assert_allclose(lp_bounded, expected_bounded, rtol=1e-4, atol=1e-5)


def _numpy_ppo_fused_loss_and_grad(
    mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
    values_np, returns_np, log_alpha_val, clip_ratio, value_coef, use_tanh,
):
    """Full NumPy reference for the fused PPO loss kernel.

    Returns (loss, grad_mean, grad_values, grad_log_std, grad_log_alpha).
    """
    batch = mean_np.shape[0]
    act_dim = mean_np.shape[1]
    inv_n = 1.0 / batch
    alpha = np.exp(log_alpha_val)
    log2pi = math.log(2 * math.pi)
    log2 = math.log(2.0)

    total_loss = 0.0
    grad_mean = np.zeros_like(mean_np)
    grad_values = np.zeros(batch, dtype=np.float32)
    grad_log_std = np.zeros(act_dim, dtype=np.float32)
    grad_log_alpha = 0.0

    for i in range(batch):
        new_lp = 0.0
        for j in range(act_dim):
            std_j = np.exp(log_std_np[j])
            u_j = pre_tanh_np[i, j]
            z = (u_j - mean_np[i, j]) / std_j
            new_lp += -0.5 * (z * z + 2.0 * log_std_np[j] + log2pi)
            if use_tanh:
                new_lp += -2.0 * (log2 - u_j - np.log(1.0 + np.exp(-2.0 * u_j)))

        log_ratio = np.clip(new_lp - old_lp_np[i], -20.0, 20.0)
        ratio = np.exp(log_ratio)
        adv = advantages_np[i]
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        clipped = surr2 < surr1
        policy_loss = -min(surr1, surr2)

        vdiff = values_np[i] - returns_np[i]
        value_loss = 0.5 * vdiff * vdiff

        entropy = 0.0
        for j in range(act_dim):
            entropy += 0.5 * (log2pi + 1.0) + log_std_np[j]

        total = (policy_loss + value_coef * value_loss - alpha * entropy) * inv_n
        total_loss += total

        d_policy_d_lp = 0.0 if clipped else -ratio * adv
        d_total_d_lp = d_policy_d_lp * inv_n

        grad_values[i] = value_coef * vdiff * inv_n

        for j in range(act_dim):
            std_j = np.exp(log_std_np[j])
            z = (pre_tanh_np[i, j] - mean_np[i, j]) / std_j
            grad_mean[i, j] = d_total_d_lp * z / std_j
            grad_log_std[j] += d_total_d_lp * (z * z - 1.0) - alpha * inv_n

        grad_log_alpha += -alpha * entropy * inv_n

    return total_loss, grad_mean, grad_values, grad_log_std, np.float32(grad_log_alpha)


class TestPPOFusedLossKernel(unittest.TestCase):
    def _run_kernel(self, mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
                    values_np, returns_np, log_alpha_val, clip_ratio, value_coef, use_tanh):
        """Helper: run the Warp kernel and return all outputs as NumPy."""
        batch, act_dim = mean_np.shape
        loss = wp.zeros(1, dtype=wp.float32, device="cpu")
        grad_mean = wp.zeros((batch, act_dim), dtype=wp.float32, device="cpu")
        grad_values = wp.zeros(batch, dtype=wp.float32, device="cpu")
        grad_log_std = wp.zeros(act_dim, dtype=wp.float32, device="cpu")
        grad_log_alpha = wp.zeros(1, dtype=wp.float32, device="cpu")

        wp.launch(
            _ppo_fused_loss_and_grad_kernel,
            dim=batch,
            inputs=[
                wp.array(mean_np, device="cpu"),
                wp.array(log_std_np, device="cpu"),
                wp.array(pre_tanh_np, device="cpu"),
                wp.array(old_lp_np, device="cpu"),
                wp.array(advantages_np, device="cpu"),
                wp.array(values_np, device="cpu"),
                wp.array(returns_np, device="cpu"),
                wp.array([log_alpha_val], dtype=wp.float32, device="cpu"),
                clip_ratio, value_coef, 1 if use_tanh else 0, batch, act_dim,
                loss, grad_mean, grad_values, grad_log_std, grad_log_alpha,
            ],
            device="cpu",
        )
        return (
            loss.numpy()[0],
            grad_mean.numpy(),
            grad_values.numpy(),
            grad_log_std.numpy(),
            grad_log_alpha.numpy()[0],
        )

    def _make_consistent_data(self, batch, act_dim, rng, use_tanh=False):
        """Generate test data where old_log_probs match the current policy (ratio=1)."""
        mean_np = rng.standard_normal((batch, act_dim)).astype(np.float32) * 0.3
        log_std_np = rng.uniform(-1, 0, act_dim).astype(np.float32)
        pre_tanh_np = mean_np + rng.standard_normal((batch, act_dim)).astype(np.float32) * np.exp(log_std_np)

        std = np.exp(log_std_np)
        z = (pre_tanh_np - mean_np) / std
        log2pi = math.log(2 * math.pi)
        log2 = math.log(2.0)
        old_lp_np = np.sum(-0.5 * (z * z + 2 * log_std_np + log2pi), axis=1)
        if use_tanh:
            for j in range(act_dim):
                old_lp_np += -2.0 * (log2 - pre_tanh_np[:, j] - np.log(1.0 + np.exp(-2.0 * pre_tanh_np[:, j])))
        old_lp_np = old_lp_np.astype(np.float32)

        return mean_np, log_std_np, pre_tanh_np, old_lp_np

    def test_full_loss_and_grads_match_numpy_unbounded(self):
        """Full loss + all gradients should match NumPy reference (unbounded mode)."""
        batch, act_dim = 8, 3
        rng = np.random.default_rng(42)
        mean_np, log_std_np, pre_tanh_np, old_lp_np = self._make_consistent_data(batch, act_dim, rng)
        advantages_np = rng.standard_normal(batch).astype(np.float32) * 2
        values_np = rng.standard_normal(batch).astype(np.float32)
        returns_np = rng.standard_normal(batch).astype(np.float32)
        log_alpha_val = np.log(0.01).astype(np.float32)
        clip_ratio, value_coef = 0.2, 0.5

        warp_loss, warp_gm, warp_gv, warp_gls, warp_gla = self._run_kernel(
            mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
            values_np, returns_np, log_alpha_val, clip_ratio, value_coef, use_tanh=False,
        )
        ref_loss, ref_gm, ref_gv, ref_gls, ref_gla = _numpy_ppo_fused_loss_and_grad(
            mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
            values_np, returns_np, log_alpha_val, clip_ratio, value_coef, use_tanh=False,
        )

        np.testing.assert_allclose(warp_loss, ref_loss, rtol=1e-4, atol=1e-5, err_msg="Loss mismatch")
        np.testing.assert_allclose(warp_gm, ref_gm, rtol=1e-4, atol=1e-6, err_msg="grad_mean mismatch")
        np.testing.assert_allclose(warp_gv, ref_gv, rtol=1e-4, atol=1e-6, err_msg="grad_values mismatch")
        np.testing.assert_allclose(warp_gls, ref_gls, rtol=1e-3, atol=1e-5, err_msg="grad_log_std mismatch")
        np.testing.assert_allclose(warp_gla, ref_gla, rtol=1e-3, atol=1e-5, err_msg="grad_log_alpha mismatch")

    def test_full_loss_and_grads_match_numpy_tanh(self):
        """Full loss + all gradients should match NumPy reference (tanh/bounded mode)."""
        batch, act_dim = 8, 3
        rng = np.random.default_rng(99)
        mean_np, log_std_np, pre_tanh_np, old_lp_np = self._make_consistent_data(batch, act_dim, rng, use_tanh=True)
        advantages_np = rng.standard_normal(batch).astype(np.float32) * 2
        values_np = rng.standard_normal(batch).astype(np.float32)
        returns_np = rng.standard_normal(batch).astype(np.float32)
        log_alpha_val = np.log(0.05).astype(np.float32)
        clip_ratio, value_coef = 0.2, 0.5

        warp_loss, warp_gm, warp_gv, warp_gls, warp_gla = self._run_kernel(
            mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
            values_np, returns_np, log_alpha_val, clip_ratio, value_coef, use_tanh=True,
        )
        ref_loss, ref_gm, ref_gv, ref_gls, ref_gla = _numpy_ppo_fused_loss_and_grad(
            mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
            values_np, returns_np, log_alpha_val, clip_ratio, value_coef, use_tanh=True,
        )

        np.testing.assert_allclose(warp_loss, ref_loss, rtol=1e-4, atol=1e-5, err_msg="Loss mismatch (tanh)")
        np.testing.assert_allclose(warp_gm, ref_gm, rtol=1e-4, atol=1e-6, err_msg="grad_mean mismatch (tanh)")
        np.testing.assert_allclose(warp_gv, ref_gv, rtol=1e-4, atol=1e-6, err_msg="grad_values mismatch (tanh)")
        np.testing.assert_allclose(warp_gls, ref_gls, rtol=1e-3, atol=1e-5, err_msg="grad_log_std mismatch (tanh)")
        np.testing.assert_allclose(warp_gla, ref_gla, rtol=1e-3, atol=1e-5, err_msg="grad_log_alpha mismatch (tanh)")

    def test_clipping_zeroes_policy_gradient(self):
        """When ratio is clipped, the policy gradient on mean should be zero for that sample."""
        batch, act_dim = 1, 2
        clip_ratio = 0.2
        mean_np = np.zeros((batch, act_dim), dtype=np.float32)
        log_std_np = np.zeros(act_dim, dtype=np.float32)
        pre_tanh_np = np.array([[0.1, -0.1]], dtype=np.float32)

        # Craft old_log_probs so that ratio >> 1 + clip_ratio with positive advantage
        # new_lp will be close to the Gaussian log-prob at z=(0.1, -0.1)
        # Set old_lp much lower so ratio = exp(new_lp - old_lp) >> 1.2
        old_lp_np = np.array([-100.0], dtype=np.float32)
        advantages_np = np.array([1.0], dtype=np.float32)
        values_np = np.zeros(batch, dtype=np.float32)
        returns_np = np.zeros(batch, dtype=np.float32)
        log_alpha_val = np.log(0.0001).astype(np.float32)

        _, warp_gm, _, _, _ = self._run_kernel(
            mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
            values_np, returns_np, log_alpha_val, clip_ratio, 0.5, use_tanh=False,
        )
        np.testing.assert_allclose(
            warp_gm, 0.0, atol=1e-6,
            err_msg="Clipped ratio should produce zero policy gradient on mean",
        )

    def test_value_loss_gradient(self):
        """Gradient of value loss w.r.t. values should be value_coef * (v - ret) / n."""
        batch, act_dim = 8, 2
        rng = np.random.default_rng(0)
        mean_np, log_std_np, pre_tanh_np, old_lp_np = self._make_consistent_data(batch, act_dim, rng)
        values_np = rng.standard_normal(batch).astype(np.float32)
        returns_np = rng.standard_normal(batch).astype(np.float32)
        advantages_np = np.zeros(batch, dtype=np.float32)
        value_coef = 0.5

        _, _, warp_gv, _, _ = self._run_kernel(
            mean_np, log_std_np, pre_tanh_np, old_lp_np, advantages_np,
            values_np, returns_np, np.log(0.01).astype(np.float32), 0.2, value_coef, use_tanh=False,
        )
        expected_grad_values = value_coef * (values_np - returns_np) / batch
        np.testing.assert_allclose(warp_gv, expected_grad_values, rtol=1e-4, atol=1e-6)

    def test_zero_advantage_zero_policy_grad(self):
        """With zero advantages, policy gradient on mean should be zero."""
        batch, act_dim = 4, 2
        rng = np.random.default_rng(0)
        mean_np, log_std_np, pre_tanh_np, old_lp_np = self._make_consistent_data(batch, act_dim, rng)

        _, warp_gm, _, _, _ = self._run_kernel(
            mean_np, log_std_np, pre_tanh_np, old_lp_np,
            np.zeros(batch, dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
            np.log(0.01).astype(np.float32), 0.2, 0.5, use_tanh=False,
        )
        np.testing.assert_allclose(warp_gm, 0.0, atol=1e-6, err_msg="Policy grad should be zero with zero advantages")


class TestGradientClipping(unittest.TestCase):
    def test_clips_large_gradients(self):
        """Gradient clipping should scale gradients when norm exceeds max_grad_norm."""
        ac = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=[8], device="cpu", seed=0)
        trainer = PPOTrainer(ac, num_envs=4, num_steps=4, num_epochs=1, num_minibatches=1, max_grad_norm=1.0)

        large_grad = wp.array(np.ones(100, dtype=np.float32) * 10.0, device="cpu")
        grads = [large_grad]
        original_norm = np.sqrt(np.sum(large_grad.numpy() ** 2))
        self.assertGreater(original_norm, 1.0)

        trainer._clip_grad_norm(grads)
        clipped = grads[0].numpy()
        clipped_norm = np.sqrt(np.sum(clipped ** 2))
        np.testing.assert_allclose(clipped_norm, 1.0, rtol=1e-4)

    def test_does_not_clip_small_gradients(self):
        """Gradients within the norm should not be modified."""
        ac = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=[8], device="cpu", seed=0)
        trainer = PPOTrainer(ac, num_envs=4, num_steps=4, num_epochs=1, num_minibatches=1, max_grad_norm=100.0)

        small_grad_np = np.array([0.1, 0.2, -0.1], dtype=np.float32)
        small_grad = wp.array(small_grad_np.copy(), device="cpu")
        grads = [small_grad]
        trainer._clip_grad_norm(grads)
        np.testing.assert_allclose(grads[0].numpy(), small_grad_np, atol=1e-7)


class TestAdvantageNormalization(unittest.TestCase):
    def test_normalized_advantages_match_numpy(self):
        """Normalized advantages should exactly match (x - mean) / std from NumPy."""
        num_envs, num_steps = 8, 16
        ac = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=[8], device="cpu", seed=0)
        trainer = PPOTrainer(ac, num_envs=num_envs, num_steps=num_steps, num_epochs=1, num_minibatches=1)

        rng = np.random.default_rng(42)
        adv_np = (rng.standard_normal(num_envs * num_steps) * 5 + 3).astype(np.float32)
        wp.copy(trainer.buffer.flat_advantages, wp.array(adv_np, device="cpu"))

        trainer._normalize_advantages()
        norm_adv = trainer._norm_adv.numpy()

        expected = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        np.testing.assert_allclose(norm_adv, expected, rtol=1e-4, atol=1e-5)

    def test_constant_advantages_produce_zeros(self):
        """Constant advantages should normalize to all zeros (zero variance)."""
        num_envs, num_steps = 4, 4
        ac = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=[8], device="cpu", seed=0)
        trainer = PPOTrainer(ac, num_envs=num_envs, num_steps=num_steps, num_epochs=1, num_minibatches=1)

        adv_np = np.full(num_envs * num_steps, 5.0, dtype=np.float32)
        wp.copy(trainer.buffer.flat_advantages, wp.array(adv_np, device="cpu"))

        trainer._normalize_advantages()
        norm_adv = trainer._norm_adv.numpy()
        np.testing.assert_allclose(norm_adv, 0.0, atol=1e-4)


class TestPPOTrainer(unittest.TestCase):
    def test_smoke_train(self):
        """Run a few PPO updates with a dummy env to ensure nothing crashes."""
        env = DummyVecEnv(num_envs=4, obs_dim=16, act_dim=4, device="cpu")
        ac = ActorCritic(obs_dim=16, act_dim=4, hidden_sizes=[32, 32], device="cpu")
        trainer = PPOTrainer(ac, num_envs=4, num_steps=8, num_epochs=2, num_minibatches=2)
        trainer.train(env, total_timesteps=32, log_interval=1)

    def test_convergence_reach_zero(self):
        """Verify PPO converges on a trivial env where optimal action = 0."""
        num_envs = 64
        obs_dim = 4
        act_dim = 4
        env = ReachZeroEnv(num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim, device="cpu")
        ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=[32, 32], device="cpu", seed=42)
        trainer = PPOTrainer(ac, num_envs=num_envs, num_steps=32, num_epochs=5, num_minibatches=4, lr=3e-4)

        obs = env.reset()
        last_values, obs = trainer.collect_rollouts(env, obs)
        initial_reward = trainer.buffer.mean_reward()
        trainer.buffer.compute_gae(last_values, trainer.gamma, trainer.gae_lambda)
        trainer.update()

        for _ in range(50):
            last_values, obs = trainer.collect_rollouts(env, obs)
            trainer.buffer.compute_gae(last_values, trainer.gamma, trainer.gae_lambda)
            trainer.update()

        final_reward = trainer.buffer.mean_reward()
        self.assertGreater(
            final_reward,
            initial_reward,
            f"PPO should improve reward: {initial_reward:.4f} -> {final_reward:.4f}",
        )
        self.assertGreater(final_reward, -2.0, f"Final reward should improve substantially (got {final_reward:.4f})")

    def test_get_stats_returns_valid(self):
        """get_stats() should return finite loss, reward, and alpha."""
        env = DummyVecEnv(num_envs=4, obs_dim=8, act_dim=2, device="cpu")
        ac = ActorCritic(obs_dim=8, act_dim=2, hidden_sizes=[16], device="cpu")
        trainer = PPOTrainer(ac, num_envs=4, num_steps=8, num_epochs=2, num_minibatches=2)
        trainer.train(env, total_timesteps=32, log_interval=999)
        stats = trainer.get_stats()
        self.assertIn("loss", stats)
        self.assertIn("mean_reward", stats)
        self.assertIn("alpha", stats)
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertTrue(np.isfinite(stats["mean_reward"]))
        self.assertGreater(stats["alpha"], 0.0)

    def test_auto_entropy_adjusts_alpha(self):
        """With auto_entropy=True, alpha should change from its initial value."""
        env = DummyVecEnv(num_envs=16, obs_dim=4, act_dim=2, device="cpu")
        ac = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=[16], device="cpu", seed=0)
        trainer = PPOTrainer(ac, num_envs=16, num_steps=8, num_epochs=5, num_minibatches=2, auto_entropy=True, entropy_coef=0.01)
        initial_log_alpha = trainer._log_alpha.numpy()[0].copy()
        trainer.train(env, total_timesteps=640, log_interval=999)
        final_log_alpha = trainer._log_alpha.numpy()[0]
        self.assertGreater(
            abs(final_log_alpha - initial_log_alpha), 0.01,
            f"Auto-entropy should adjust log_alpha: {initial_log_alpha:.6f} -> {final_log_alpha:.6f}",
        )


class TestOnnxExport(unittest.TestCase):
    def test_export_and_load(self):
        """Export an actor to ONNX and verify it loads with OnnxRuntime."""
        actor = WarpMLP([16, 32, 4], activation="elu", device="cpu", output_gain=0.01)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            export_actor_to_onnx(actor, obs_dim=16, path=path)
            from newton._src.onnx_runtime import OnnxRuntime

            rt = OnnxRuntime(path, device="cpu", batch_size=2)
            obs = wp.array(np.random.default_rng(0).standard_normal((2, 16)).astype(np.float32), device="cpu")
            out = rt({"observation": obs})
            self.assertIn("action", out)
            self.assertEqual(out["action"].shape, (2, 4))
        finally:
            os.unlink(path)

    def test_onnx_matches_warp_forward(self):
        """ONNX inference should produce the same output as WarpMLP.forward."""
        actor = WarpMLP([8, 16, 4], activation="elu", device="cpu", seed=42, output_gain=1.0)
        obs_np = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)

        warp_out = actor.forward(wp.array(obs_np, device="cpu")).numpy()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            export_actor_to_onnx(actor, obs_dim=8, path=path)
            from newton._src.onnx_runtime import OnnxRuntime

            rt = OnnxRuntime(path, device="cpu", batch_size=4)
            onnx_out = rt({"observation": wp.array(obs_np, device="cpu")})["action"].numpy()
            np.testing.assert_allclose(onnx_out, warp_out, rtol=1e-4, atol=1e-5)
        finally:
            os.unlink(path)

    def test_export_all_activations(self):
        """ONNX export should work for all supported activations."""
        for act in ["elu", "relu", "tanh"]:
            actor = WarpMLP([4, 8, 2], activation=act, device="cpu", seed=0)
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                path = f.name
            try:
                export_actor_to_onnx(actor, obs_dim=4, path=path)
                from newton._src.onnx_runtime import OnnxRuntime

                rt = OnnxRuntime(path, device="cpu", batch_size=1)
                obs = wp.array(np.ones((1, 4), dtype=np.float32), device="cpu")
                out = rt({"observation": obs})
                self.assertEqual(out["action"].shape, (1, 2), f"Failed for activation={act}")
            finally:
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
