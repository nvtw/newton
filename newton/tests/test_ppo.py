# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the single-file PPO implementation."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import warp as wp

from newton._src.ppo import (
    ActorCritic,
    AdamW,
    PPOTrainer,
    RolloutBuffer,
    WarpMLP,
    export_actor_to_onnx,
)


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


class TestAdamW(unittest.TestCase):
    def test_step_updates_params(self):
        p = wp.array(np.ones(8, dtype=np.float32), device="cpu", requires_grad=True)
        opt = AdamW([p], lr=0.1)
        g = wp.array(np.ones(8, dtype=np.float32) * 0.5, device="cpu")
        before = p.numpy().copy()
        opt.step([g])
        after = p.numpy()
        self.assertFalse(np.allclose(before, after), "Parameters should change after step")


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
        self.assertEqual(len(params), 4)


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

    def test_evaluate_shapes(self):
        ac = ActorCritic(obs_dim=16, act_dim=4, hidden_sizes=[32, 32], device="cpu")
        batch = 4
        ac.alloc_buffers(rollout_batch=8, minibatch_size=batch)
        obs = wp.array(np.random.default_rng(0).standard_normal((batch, 16)).astype(np.float32), device="cpu")
        actions = wp.array(np.random.default_rng(1).standard_normal((batch, 4)).astype(np.float32), device="cpu")
        log_probs, entropy, values = ac.evaluate(obs, actions)
        self.assertEqual(log_probs.shape, (batch,))
        self.assertEqual(entropy.shape, (batch,))
        self.assertEqual(values.shape, (batch,))


class TestRolloutBuffer(unittest.TestCase):
    def test_insert_and_flatten(self):
        buf = RolloutBuffer(num_envs=4, num_steps=8, obs_dim=16, act_dim=4, device="cpu")
        rng = np.random.default_rng(0)
        for t in range(8):
            buf.insert(
                t=t,
                obs=wp.array(rng.standard_normal((4, 16)).astype(np.float32), device="cpu"),
                actions=wp.array(rng.standard_normal((4, 4)).astype(np.float32), device="cpu"),
                log_probs=wp.array(rng.standard_normal(4).astype(np.float32), device="cpu"),
                rewards=wp.array(rng.standard_normal(4).astype(np.float32), device="cpu"),
                dones=wp.zeros(4, dtype=wp.float32, device="cpu"),
                values=wp.array(rng.standard_normal(4).astype(np.float32), device="cpu"),
            )
        buf.flatten()
        self.assertEqual(buf.flat_obs.shape, (32, 16))
        self.assertEqual(buf.flat_actions.shape, (32, 4))

    def test_gae(self):
        buf = RolloutBuffer(num_envs=2, num_steps=4, obs_dim=4, act_dim=2, device="cpu")
        for t in range(4):
            buf.insert(
                t=t,
                obs=wp.array(np.zeros((2, 4), dtype=np.float32), device="cpu"),
                actions=wp.array(np.zeros((2, 2), dtype=np.float32), device="cpu"),
                log_probs=wp.zeros(2, dtype=wp.float32, device="cpu"),
                rewards=wp.array(np.ones(2, dtype=np.float32), device="cpu"),
                dones=wp.zeros(2, dtype=wp.float32, device="cpu"),
                values=wp.zeros(2, dtype=wp.float32, device="cpu"),
            )
        last_vals = wp.zeros(2, dtype=wp.float32, device="cpu")
        buf.compute_gae(last_vals, gamma=0.99, gae_lambda=0.95)
        adv = buf.advantages.numpy()
        self.assertTrue(np.all(adv > 0), "Advantages should be positive with positive rewards and zero values")


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

        # Collect initial reward baseline
        obs = env.reset()
        last_values, obs = trainer.collect_rollouts(env, obs)
        initial_reward = trainer.buffer.mean_reward()
        trainer.buffer.compute_gae(last_values, trainer.gamma, trainer.gae_lambda)
        trainer.update()

        # Train for many updates
        for _ in range(50):
            last_values, obs = trainer.collect_rollouts(env, obs)
            trainer.buffer.compute_gae(last_values, trainer.gamma, trainer.gae_lambda)
            trainer.update()

        final_reward = trainer.buffer.mean_reward()
        print(f"ReachZero convergence: initial_reward={initial_reward:.4f} -> final_reward={final_reward:.4f}")

        # Reward should improve significantly (closer to 0)
        self.assertGreater(
            final_reward,
            initial_reward,
            f"PPO should improve reward: {initial_reward:.4f} -> {final_reward:.4f}",
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


if __name__ == "__main__":
    unittest.main()
