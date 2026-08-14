# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure-Warp FlashSAC implementation."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training.flash_sac import (
    BufferReplayFlashSAC,
    ConfigFlashSAC,
    TrainerFlashSAC,
)
from newton._src.solvers.phoenx.rl_training.kernels import (
    sac_distributional_min_projection_device_alpha_kernel,
)
from newton._src.solvers.phoenx.rl_training.sac import BatchSAC
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestTrainerFlashSAC(unittest.TestCase):
    def test_flash_sac_defaults_and_replay_warmup(self) -> None:
        """Match upstream defaults and enforce replay warmup."""

        device = require_cuda_graph_capture("FlashSAC replay tests")
        config = ConfigFlashSAC()
        self.assertTrue(config.distributional_critic)
        self.assertEqual(config.distributional_atoms, 101)
        self.assertEqual(config.policy_frequency, 2)
        self.assertAlmostEqual(config.initial_alpha, 0.01)
        replay = BufferReplayFlashSAC(
            minimum_size=2,
            capacity=4,
            obs_dim=2,
            action_dim=1,
            batch_size=2,
            device=device,
        )
        self.assertFalse(replay.can_sample())
        replay.add_batch(
            wp.zeros((2, 2), dtype=wp.float32, device=device),
            wp.zeros((2, 1), dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            wp.zeros((2, 2), dtype=wp.float32, device=device),
        )
        self.assertTrue(replay.can_sample())

    def test_flash_sac_constraints_survive_update(self) -> None:
        """Keep unit incoming weights after a full FlashSAC update."""

        device = require_cuda_graph_capture("FlashSAC update tests")
        rng = np.random.default_rng(41)
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, hidden_layers=(8,), device=device, seed=43)
        expected_entropy = 0.5 * 2 * math.log(2.0 * math.pi * math.e * 0.15**2)
        self.assertAlmostEqual(trainer.target_entropy, expected_entropy)
        self.assertAlmostEqual(trainer._scheduled_learning_rate(0), 3.0e-4)
        self.assertAlmostEqual(trainer._scheduled_learning_rate(1_000_000), 1.5e-4)

        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(16, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=16).astype(np.float32), device=device),
            dones=wp.zeros(16, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), device=device),
        )
        stats = trainer.update(batch, seed=47)
        self.assertTrue(all(math.isfinite(value) for value in stats.__dict__.values()))
        for network in (trainer.actor.net, trainer.critic1, trainer.critic2):
            for weight in network.weights:
                np.testing.assert_allclose(np.linalg.norm(weight.numpy(), axis=0), 1.0, rtol=0.0, atol=2.0e-6)

    def test_flash_sac_reuses_exploration_noise(self) -> None:
        """Reuse sampled Gaussian noise for the configured repeat duration."""

        device = require_cuda_graph_capture("FlashSAC exploration tests")
        trainer = TrainerFlashSAC(obs_dim=2, action_dim=1, hidden_layers=(8,), device=device, seed=53)
        trainer._noise_repeat_steps = 3
        trainer._noise_repeat_count = 0
        obs = wp.zeros((4, 2), dtype=wp.float32, device=device)
        first = trainer.act(obs, seed=101)[0].numpy()
        second = trainer.act(obs, seed=202)[0].numpy()
        np.testing.assert_array_equal(first, second)

    def test_min_expected_q_selects_one_target_distribution(self) -> None:
        """Select the lower-Q critic distribution for both targets."""

        device = require_cuda_graph_capture("FlashSAC categorical target tests")
        logits1 = wp.array([[8.0, 0.0, 0.0]], dtype=wp.float32, device=device)
        logits2 = wp.array([[0.0, 0.0, 8.0]], dtype=wp.float32, device=device)
        targets1 = wp.zeros_like(logits1)
        targets2 = wp.zeros_like(logits2)
        wp.launch(
            sac_distributional_min_projection_device_alpha_kernel,
            dim=(1, 3),
            inputs=[
                wp.zeros(1, dtype=wp.float32, device=device),
                wp.zeros(1, dtype=wp.float32, device=device),
                logits1,
                logits2,
                wp.zeros(1, dtype=wp.float32, device=device),
                1.0,
                wp.zeros(1, dtype=wp.float32, device=device),
                3,
                -1.0,
                1.0,
            ],
            outputs=[targets1, targets2],
            device=device,
        )
        np.testing.assert_allclose(targets1.numpy(), targets2.numpy(), rtol=0.0, atol=0.0)
        self.assertEqual(int(np.argmax(targets1.numpy()[0])), 0)

    def test_n_step_replay_stops_at_truncation(self) -> None:
        """Accumulate n-step rewards while preserving truncation bootstrap semantics."""

        device = require_cuda_graph_capture("FlashSAC n-step replay tests")
        replay = BufferReplayFlashSAC(
            minimum_size=0,
            capacity=8,
            obs_dim=1,
            action_dim=1,
            batch_size=2,
            n_step=3,
            gamma=0.5,
            normalize_rewards=False,
            device=device,
        )
        for step, rewards in enumerate(([1.0, 10.0], [2.0, 20.0], [3.0, 30.0])):
            replay.add_batch(
                wp.array([[float(step)], [float(step)]], dtype=wp.float32, device=device),
                wp.zeros((2, 1), dtype=wp.float32, device=device),
                wp.array(rewards, dtype=wp.float32, device=device),
                wp.zeros(2, dtype=wp.float32, device=device),
                wp.array([[float(step + 1)], [float(step + 1)]], dtype=wp.float32, device=device),
                truncateds=wp.array([0.0, float(step == 0)], dtype=wp.float32, device=device),
            )
        np.testing.assert_allclose(replay.rewards.numpy()[:2], [2.75, 10.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(replay.dones.numpy()[:2], [0.0, 0.0])
        np.testing.assert_allclose(replay.next_obs.numpy()[:2, 0], [3.0, 1.0], rtol=0.0, atol=0.0)

    def test_reward_normalizer_bounds_discounted_return(self) -> None:
        """Bound normalized rewards by the configured return range."""

        device = require_cuda_graph_capture("FlashSAC reward normalization tests")
        replay = BufferReplayFlashSAC(
            minimum_size=0,
            capacity=2,
            obs_dim=1,
            action_dim=1,
            batch_size=1,
            normalized_return_max=5.0,
            device=device,
        )
        replay.add_batch(
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            wp.array([10.0], dtype=wp.float32, device=device),
            wp.zeros(1, dtype=wp.float32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
        )
        np.testing.assert_allclose(replay.sample(seed=1).rewards.numpy(), [5.0], rtol=0.0, atol=1.0e-6)

    def test_update_uses_upstream_network_order(self) -> None:
        """Update actor and temperature before critic and target networks."""

        device = require_cuda_graph_capture("FlashSAC update-order tests")
        trainer = TrainerFlashSAC(
            obs_dim=2,
            action_dim=1,
            hidden_layers=(4,),
            config=ConfigFlashSAC(normalize_observations=False),
            device=device,
        )
        events: list[str] = []
        trainer._update_actor = lambda batch, seed: events.append("actor")
        trainer._update_alpha = lambda batch, seed: events.append("temperature")
        trainer._update_critics = lambda batch, seed: events.append("critic")
        trainer.target_critic1.soft_update_from = lambda source, tau: events.append("target1")
        trainer.target_critic2.soft_update_from = lambda source, tau: events.append("target2")
        batch = BatchSAC(
            obs=wp.zeros((2, 2), dtype=wp.float32, device=device),
            actions=wp.zeros((2, 1), dtype=wp.float32, device=device),
            rewards=wp.zeros(2, dtype=wp.float32, device=device),
            dones=wp.zeros(2, dtype=wp.float32, device=device),
            next_obs=wp.zeros((2, 2), dtype=wp.float32, device=device),
        )
        trainer.update(batch, read_stats=False)
        self.assertEqual(events, ["actor", "temperature", "critic", "target1", "target2"])

    def test_expanded_layers_match_upstream_capacities(self) -> None:
        """Match upstream block expansion depths and distinct feature widths."""

        self.assertEqual(TrainerFlashSAC._expanded_block_layers(128, 2), (128, 512, 128, 512, 128))
        self.assertEqual(TrainerFlashSAC._expanded_block_layers(256, 2), (256, 1024, 256, 1024, 256))


if __name__ == "__main__":
    wp.init()
    unittest.main()
