# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training.env import collect_ppo_rollout_seed_counter, make_seed_counter
from newton._src.solvers.phoenx.rl_training.ppo import BufferRollout, ConfigPPO, TrainerPPO
from newton._src.solvers.phoenx.rl_training.standard_envs import (
    ConfigEnvPendulumV1Warp,
    EnvPendulumV1Warp,
    pendulum_v1_numpy_step,
)
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestPPOStandardBenchmarks(unittest.TestCase):
    def test_pendulum_v1_step_matches_numpy_reference_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX PPO standard benchmark env tests")
        config = ConfigEnvPendulumV1Warp(world_count=5, max_episode_steps=0, seed=11)
        env = EnvPendulumV1Warp(config, device=device)
        theta = np.linspace(-2.5, 2.5, env.world_count, dtype=np.float32)
        theta_dot = np.linspace(-3.0, 3.0, env.world_count, dtype=np.float32)
        actions_np = np.linspace(-1.2, 1.2, env.world_count, dtype=np.float32).reshape(env.world_count, 1)
        actions = wp.array(actions_np, dtype=wp.float32, device=device)
        env.theta.assign(theta)
        env.theta_dot.assign(theta_dot)

        with wp.ScopedCapture(device=device) as capture:
            env.step(actions)
        wp.capture_launch(capture.graph)

        expected_theta, expected_theta_dot, expected_rewards, expected_obs = pendulum_v1_numpy_step(
            theta, theta_dot, actions_np, config
        )
        np.testing.assert_allclose(env.theta.numpy(), expected_theta, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(env.theta_dot.numpy(), expected_theta_dot, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(env.rewards.numpy(), expected_rewards, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(env.obs.numpy(), expected_obs, rtol=1.0e-6, atol=1.0e-6)

    def test_ppo_collect_update_on_pendulum_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX PPO standard benchmark update tests")
        env = EnvPendulumV1Warp(ConfigEnvPendulumV1Warp(world_count=8, max_episode_steps=0, seed=13), device=device)
        trainer = TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=(8, 8),
            config=ConfigPPO(
                gamma=0.95,
                gae_lambda=0.9,
                actor_lr=1.0e-3,
                critic_lr=1.0e-3,
                train_epochs=1,
                entropy_coeff=0.0,
                normalize_advantages=True,
                max_grad_norm=0.5,
            ),
            device=device,
            seed=17,
            squash_actions=True,
            activation="tanh",
            log_std_init=-0.5,
        )
        buffer = BufferRollout(
            num_steps=4,
            num_envs=env.world_count,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            device=device,
        )
        trainer.reserve_update_buffers(buffer)
        rollout_seed = make_seed_counter(101, device=device)
        update_seed = make_seed_counter(202, device=device)

        with wp.ScopedCapture(device=device) as capture:
            collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=rollout_seed)
            trainer.update_seed_counter(buffer, seed_counter=update_seed, read_stats=False)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.isfinite(env.obs.numpy()).all())
        self.assertTrue(np.isfinite(buffer.rewards.numpy()).all())
        self.assertTrue(np.isfinite(buffer.advantages.numpy()).all())
        self.assertTrue(np.isfinite(trainer.actor.log_std.numpy()).all())


if __name__ == "__main__":
    unittest.main()
