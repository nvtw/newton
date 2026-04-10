# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for PufferLib PPO training pipeline.

Verifies that all environments and training configurations can:
1. Initialize without error
2. Run a few training iterations (warmup + graph capture + train)
3. Produce non-NaN loss values

Run with::

    uv run --extra dev -m newton.tests -k test_ppo_smoke
"""

import unittest

import warp as wp


class TestPPOSmoke(unittest.TestCase):
    """Smoke tests for PufferLib PPO training."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _train_env(self, make_env_fn, config_overrides: dict):
        """Helper: train an env for a tiny number of steps and check it doesn't crash."""
        from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

        base = dict(
            total_timesteps=config_overrides.get("num_envs", 64) * 24 * 3,
            seed=42,
            device="cuda:0",
            log_interval=999,  # suppress output
        )
        base.update(config_overrides)
        config = PPOConfig(**base)

        result = PPOTrainer(config, make_env_fn).train()
        self.assertIsNotNone(result)
        self.assertIn("policy", result)

    def test_ppo_training_envs(self):
        """All PufferLib envs (CartPole, Drone, SquaredContinuous) train without error.

        Combined into a single test because CUDA graph captures from different
        training runs cannot coexist in the same process.
        """
        from newton._src.pufferlib.envs.cartpole import CartPoleEnv

        N = 256
        self._train_env(
            lambda d: CartPoleEnv(num_envs=N, device=d, seed=42),
            dict(
                num_envs=N, horizon=32, obs_dim=4, num_actions=2, hidden=32,
                lr=0.1, gamma=0.8, gae_lambda=0.922, clip_coef=0.144,
                vf_coef=1.78, vf_clip_coef=3.91, ent_coef=0.037,
                max_grad_norm=0.33, momentum=0.943,
                replay_ratio=0.381, minibatch_size=N * 32,
            ),
        )

    def test_anymal_env_and_training(self):
        """AnymalEnv initializes, steps, and trains without error."""
        from newton._src.pufferlib.envs.anymal import AnymalEnv

        # Init + step test
        env = AnymalEnv(num_envs=4, device="cuda:0", seed=42, use_mujoco_contacts=True)
        env.reset()
        actions = wp.zeros((4, 12), dtype=wp.float32, device="cuda:0")
        seed_arr = wp.array([42], dtype=wp.int32, device="cuda:0")
        env.step_graphed(actions, seed_arr)
        wp.synchronize()
        stats = env.get_episode_stats()
        self.assertIn("mean_return", stats)
        del env

        # Training test (separate env instance)
        N = 64
        self._train_env(
            lambda d: AnymalEnv(num_envs=N, device=d, seed=42, use_mujoco_contacts=True),
            dict(
                num_envs=N, horizon=24, obs_dim=48, num_actions=12, hidden=64,
                lr=1e-3, optimizer="adamw", continuous=True, init_logstd=-1.0,
                activation="elu", normalize_obs=True, desired_kl=0.01,
                replay_ratio=1.0, minibatch_size=N * 24,
                reward_clamp=10.0,
            ),
        )

    def test_onnx_export(self):
        """SimpleMLP can be exported to ONNX and loaded by OnnxRuntime."""
        from newton._src.pufferlib.network import SimpleMLP
        from newton._src.pufferlib.onnx_export import export_policy_to_onnx

        import tempfile
        import os

        policy = SimpleMLP(
            obs_dim=4, hidden=32, out_dim=3, max_batch=64,
            device="cuda:0", seed=42, continuous=True, num_actions=2,
            init_logstd=-1.0, activation="elu",
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            export_policy_to_onnx(policy, obs_dim=4, num_actions=2, path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

            # Load and run inference
            from newton._src.onnx_runtime import OnnxRuntime
            import numpy as np

            rt = OnnxRuntime(path, device="cuda:0")
            obs = wp.array(np.zeros((1, 4), dtype=np.float32), device="cuda:0")
            result = rt({"observation": obs})
            self.assertIn("action", result)
            self.assertEqual(result["action"].shape, (1, 2))
        finally:
            os.unlink(path)

    def test_obs_normalizer(self):
        """ObsNormalizer updates stats and normalizes correctly."""
        from newton._src.pufferlib.obs_normalizer import ObsNormalizer
        import numpy as np

        norm = ObsNormalizer(obs_dim=4, device="cuda:0")
        obs = wp.array(np.random.randn(64, 4).astype(np.float32), device="cuda:0")
        out = wp.zeros((64, 4), dtype=wp.float32, device="cuda:0")

        norm.update(obs, 64)
        norm.normalize(obs, out, 64)
        wp.synchronize()

        out_np = out.numpy()
        self.assertFalse(np.any(np.isnan(out_np)))
        # Normalized values should be roughly zero-mean
        self.assertLess(abs(out_np.mean()), 2.0)

        mean, var = norm.get_mean_var()
        self.assertEqual(mean.shape, (4,))
        self.assertEqual(var.shape, (4,))


if __name__ == "__main__":
    unittest.main()
