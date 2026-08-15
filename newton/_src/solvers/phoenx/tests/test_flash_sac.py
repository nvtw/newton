# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure-Warp FlashSAC implementation."""

from __future__ import annotations

import math
import tempfile
import unittest

import numpy as np
import warp as wp

import newton.rl as public_rl
from newton._src.solvers.phoenx.rl_training.flash_sac import (
    BufferReplayFlashSAC,
    ConfigFlashSAC,
    TrainerFlashSAC,
    _flash_sac_alpha_loss_kernel,
)
from newton._src.solvers.phoenx.rl_training.flash_sac_networks import NetworkFlashSAC
from newton._src.solvers.phoenx.rl_training.g1 import (
    ACTION_DIM_G1,
    OBS_DIM_G1,
    ConfigEnvG1PhoenX,
    EnvG1PhoenX,
)
from newton._src.solvers.phoenx.rl_training.kernels import (
    sac_distributional_min_projection_device_alpha_kernel,
)
from newton._src.solvers.phoenx.rl_training.ppo import BufferRollout, ConfigPPO, TrainerPPO
from newton._src.solvers.phoenx.rl_training.sac import BatchSAC
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class _G1SmokeEnv:
    def __init__(self, obs: wp.array2d[wp.float32], next_obs: wp.array2d[wp.float32]):
        self.world_count = int(obs.shape[0])
        self.obs_dim = int(obs.shape[1])
        self.action_dim = ACTION_DIM_G1
        self.device = obs.device
        self._initial_obs = obs
        self._obs = obs
        self._next_obs = next_obs
        self._rewards = wp.array(np.linspace(-0.2, 0.3, self.world_count, dtype=np.float32), device=self.device)
        self._dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_next_obs = next_obs
        self.step_terminateds = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_truncateds = wp.ones(self.world_count, dtype=wp.float32, device=self.device)

    def reset(self) -> wp.array2d[wp.float32]:
        self._obs = self._initial_obs
        return self._obs

    def observe(self) -> wp.array2d[wp.float32]:
        return self._obs

    def step(self, actions: wp.array2d[wp.float32]) -> tuple[wp.array, wp.array, wp.array]:
        self._obs = self._initial_obs
        return self._obs, self._rewards, wp.ones_like(self._dones)


class TestTrainerFlashSAC(unittest.TestCase):
    def test_reference_backbone_forward_equations_and_initialization(self) -> None:
        """Match reference normalization, residual, head, and orthogonal equations."""

        device = require_cuda_graph_capture("FlashSAC reference backbone equations")
        network = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=1,
            output_dim=2,
            actor_heads=False,
            device=device,
            seed=11,
        )
        network.input_norm.running_mean.assign(np.asarray([1.0, -1.0], dtype=np.float32))
        network.input_norm.running_variance.assign(np.asarray([4.0, 9.0], dtype=np.float32))
        network.input_norm.scale.assign(np.asarray([2.0, 3.0], dtype=np.float32))
        network.input_norm.bias.assign(np.asarray([0.5, -0.5], dtype=np.float32))
        network.embed_weight.assign(np.eye(2, dtype=np.float32))
        for weight in network.block_weights[0]:
            weight.zero_()
        network.rms_scale.assign(np.asarray([2.0, 0.5], dtype=np.float32))
        network.head_weights[0].assign(np.eye(2, dtype=np.float32))
        network.head_biases[0].assign(np.asarray([0.25, -0.25], dtype=np.float32))
        x_np = np.asarray([[3.0, 2.0]], dtype=np.float32)
        output = network.forward(wp.array(x_np, device=device), requires_grad=False).numpy()

        normalized = (x_np - np.asarray([1.0, -1.0], dtype=np.float32)) / np.sqrt(
            np.asarray([4.0, 9.0], dtype=np.float32) + 1.0e-5
        )
        normalized = normalized * np.asarray([2.0, 3.0], dtype=np.float32) + np.asarray([0.5, -0.5], dtype=np.float32)
        rms = np.sqrt(np.mean(normalized * normalized, axis=-1, keepdims=True) + 1.0e-6)
        expected = normalized / rms * np.asarray([2.0, 0.5], dtype=np.float32)
        expected += np.asarray([0.25, -0.25], dtype=np.float32)
        np.testing.assert_allclose(output, expected, rtol=1.0e-6, atol=1.0e-6)

        square = NetworkFlashSAC(
            input_dim=4,
            hidden_dim=4,
            num_blocks=0,
            output_dim=2,
            actor_heads=False,
            device=device,
            seed=13,
        ).embed_weight.numpy()
        np.testing.assert_allclose(square.T @ square, np.eye(4), rtol=0.0, atol=2.0e-6)

    def test_reference_batch_norm_running_state_and_parameter_ema(self) -> None:
        """Use unbiased running variance and exclude running state from target EMA."""

        device = require_cuda_graph_capture("FlashSAC reference BatchNorm state")
        online = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=0,
            output_dim=1,
            actor_heads=False,
            device=device,
            seed=17,
        )
        target = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=0,
            output_dim=1,
            actor_heads=False,
            device=device,
            seed=19,
        )
        target.default_training = True
        values = wp.array(np.asarray([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32), device=device)
        target.forward(values, requires_grad=False)
        np.testing.assert_allclose(target.input_norm.running_mean.numpy(), [0.02, 0.05], atol=1.0e-7)
        np.testing.assert_allclose(target.input_norm.running_variance.numpy(), [1.01, 1.07], atol=1.0e-7)

        running_before = target.input_norm.running_mean.numpy().copy()
        online.input_norm.running_mean.assign(np.asarray([9.0, 8.0], dtype=np.float32))
        target.head_biases[0].zero_()
        online.head_biases[0].fill_(2.0)
        target.soft_update_from(online, 0.5)
        np.testing.assert_array_equal(target.input_norm.running_mean.numpy(), running_before)
        np.testing.assert_allclose(target.head_biases[0].numpy(), [1.0])

    def test_reference_actor_log_std_smooth_mapping(self) -> None:
        """Map raw actor standard deviations smoothly into upstream bounds."""

        device = require_cuda_graph_capture("FlashSAC smooth log-std mapping")
        network = NetworkFlashSAC(
            input_dim=1,
            hidden_dim=1,
            num_blocks=0,
            output_dim=2,
            actor_heads=True,
            device=device,
            seed=23,
        )
        network.embed_weight.fill_(1.0)
        network.head_weights[1].zero_()
        x = wp.ones((1, 1), dtype=wp.float32, device=device)
        for raw in (-20.0, 0.0, 20.0):
            network.head_biases[1].fill_(raw)
            mapped = float(network.forward(x, requires_grad=False).numpy()[0, 1])
            expected = -10.0 + 12.0 * 0.5 * (1.0 + math.tanh(raw))
            self.assertAlmostEqual(mapped, expected, places=5)

    def test_reference_backbone_backward_and_temperature_gradient(self) -> None:
        """Match a finite-difference network gradient and exact temperature gradient."""

        device = require_cuda_graph_capture("FlashSAC reference gradient equations")
        network = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=0,
            output_dim=1,
            actor_heads=False,
            device=device,
            seed=29,
        )
        x_np = np.asarray([[-1.0, 0.5], [0.2, 1.3], [1.4, -0.7]], dtype=np.float32)
        x = wp.array(x_np, device=device)
        output = network.forward_manual(x)
        network.backward_manual(wp.ones(output.shape, dtype=wp.float32, device=device))
        analytic = float(network.embed_weight.grad.numpy()[0, 0])
        weights = network.embed_weight.numpy().copy()
        epsilon = 1.0e-3
        losses = []
        for delta in (-epsilon, epsilon):
            perturbed = weights.copy()
            perturbed[0, 0] += delta
            network.embed_weight.assign(perturbed)
            losses.append(float(network.forward(x, requires_grad=False, training=True).numpy().sum()))
        network.embed_weight.assign(weights)
        finite_difference = (losses[1] - losses[0]) / (2.0 * epsilon)
        self.assertAlmostEqual(analytic, finite_difference, delta=2.0e-2)

        log_probs = wp.array(np.asarray([-2.0, -1.0], dtype=np.float32), device=device)
        log_alpha = wp.array(np.asarray([math.log(0.25)], dtype=np.float32), device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(
                _flash_sac_alpha_loss_kernel,
                dim=2,
                inputs=[log_probs, log_alpha, 2, -0.5],
                outputs=[loss],
                device=device,
            )
        tape.backward(loss)
        expected = 0.25 * ((2.0 + 0.5) + (1.0 + 0.5)) / 2.0
        self.assertAlmostEqual(float(loss.numpy()[0]), expected, places=6)
        self.assertAlmostEqual(float(log_alpha.grad.numpy()[0]), expected, places=6)

    def test_flash_sac_defaults_and_replay_warmup(self) -> None:
        """Match upstream defaults and enforce replay warmup."""

        device = require_cuda_graph_capture("FlashSAC replay tests")
        config = ConfigFlashSAC()
        self.assertTrue(config.distributional_critic)
        self.assertEqual(config.distributional_atoms, 101)
        self.assertEqual(config.policy_frequency, 2)
        self.assertAlmostEqual(config.initial_alpha, 0.01)
        self.assertEqual(config.buffer_max_length, 1_000_000)
        self.assertEqual(config.buffer_min_length, 10_000)
        self.assertEqual(config.sample_batch_size, 2048)
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

    def test_reference_backbone_integrated_update(self) -> None:
        """Run a finite update through residual reference actor and critics."""

        device = require_cuda_graph_capture("FlashSAC integrated reference update")
        rng = np.random.default_rng(31)
        config = ConfigFlashSAC(
            actor_hidden_dim=4,
            actor_num_blocks=1,
            critic_hidden_dim=4,
            critic_num_blocks=1,
            distributional_atoms=7,
            normalize_observations=True,
            normalize_rewards=False,
            policy_frequency=1,
        )
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=37)
        self.assertIsInstance(trainer.actor.net, NetworkFlashSAC)
        self.assertFalse(trainer.config.normalize_observations)
        actor_running_before = trainer.actor.net.input_norm.running_mean.numpy().copy()
        target_running_before = trainer.target_critic1.input_norm.running_mean.numpy().copy()
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(8, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=8).astype(np.float32), device=device),
            dones=wp.zeros(8, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
        )
        stats = trainer.update(batch, seed=41)

        self.assertTrue(all(math.isfinite(value) for value in stats.__dict__.values()))
        expected_actor_running = 0.01 * np.concatenate((batch.obs.numpy(), batch.next_obs.numpy()), axis=0).mean(axis=0)
        np.testing.assert_allclose(
            trainer.actor.net.input_norm.running_mean.numpy(), expected_actor_running, atol=1.0e-7
        )
        self.assertFalse(np.array_equal(trainer.actor.net.input_norm.running_mean.numpy(), actor_running_before))
        self.assertFalse(np.array_equal(trainer.target_critic1.input_norm.running_mean.numpy(), target_running_before))
        for network in (trainer.actor.net, trainer.critic1, trainer.critic2):
            for weight in network.weights:
                np.testing.assert_allclose(np.linalg.norm(weight.numpy(), axis=0), 1.0, rtol=0.0, atol=3.0e-6)
            norms = [network.input_norm]
            for norm1, norm2 in network.block_norms:
                norms.extend((norm1, norm2))
            for norm in norms:
                affine_squared = np.sum(norm.scale.numpy() ** 2) + np.sum(norm.bias.numpy() ** 2)
                self.assertAlmostEqual(float(affine_squared), float(norm.width), delta=3.0e-5)
            self.assertAlmostEqual(
                float(np.sum(network.rms_scale.numpy() ** 2)), float(network.hidden_dim), delta=3.0e-5
            )

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

    def test_flash_sac_checkpoint_round_trip(self) -> None:
        """Restore networks, targets, temperature, optimizers, counters, and config."""

        device = require_cuda_graph_capture("FlashSAC checkpoint tests")
        rng = np.random.default_rng(131)
        config = ConfigFlashSAC(normalize_observations=False, normalize_rewards=False)
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=137)
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(8, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=8).astype(np.float32), device=device),
            dones=wp.zeros(8, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
        )
        trainer.update(batch, seed=139)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/flash_sac.npz"
            public_rl.save_flash_sac_checkpoint(trainer, path)
            restored = public_rl.load_flash_sac_checkpoint(path, device=device)

        self.assertEqual(restored._update_count, trainer._update_count)
        self.assertEqual(restored._gradient_update_count, trainer._gradient_update_count)
        self.assertEqual(restored.config, trainer.config)
        self.assertIsNone(restored.replay_buffer)
        np.testing.assert_array_equal(restored.log_alpha.numpy(), trainer.log_alpha.numpy())
        np.testing.assert_array_equal(restored._alpha.numpy(), trainer._alpha.numpy())

        for expected_network, actual_network in (
            (trainer.actor.net, restored.actor.net),
            (trainer.critic1, restored.critic1),
            (trainer.critic2, restored.critic2),
            (trainer.target_critic1, restored.target_critic1),
            (trainer.target_critic2, restored.target_critic2),
        ):
            for expected, actual in zip(expected_network.parameters(), actual_network.parameters(), strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

        for expected_optimizer, actual_optimizer in (
            (trainer.actor_optimizer, restored.actor_optimizer),
            (trainer.critic1_optimizer, restored.critic1_optimizer),
            (trainer.critic2_optimizer, restored.critic2_optimizer),
            (trainer.alpha_optimizer, restored.alpha_optimizer),
        ):
            self.assertEqual(actual_optimizer.step_count, expected_optimizer.step_count)
            for expected, actual in zip(expected_optimizer.m, actual_optimizer.m, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            for expected, actual in zip(expected_optimizer.v, actual_optimizer.v, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_reference_backbone_checkpoint_round_trip(self) -> None:
        """Restore reference affine, running-statistic, RMS, and head state."""

        device = require_cuda_graph_capture("FlashSAC reference checkpoint tests")
        config = ConfigFlashSAC(
            actor_hidden_dim=4,
            actor_num_blocks=1,
            critic_hidden_dim=4,
            critic_num_blocks=1,
            distributional_atoms=7,
            normalize_rewards=False,
        )
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=149)
        values = wp.array(
            np.asarray([[-1.0, 0.5, 2.0], [0.0, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=np.float32),
            device=device,
        )
        trainer.actor.net.forward(values, training=True, requires_grad=False)
        trainer.actor.net.rms_scale.assign(np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/flash_sac_reference.npz"
            trainer.save_checkpoint(path)
            restored = TrainerFlashSAC.load_checkpoint(path, device=device)

        self.assertIsInstance(restored.actor.net, NetworkFlashSAC)
        for expected_network, actual_network in (
            (trainer.actor.net, restored.actor.net),
            (trainer.critic1, restored.critic1),
            (trainer.critic2, restored.critic2),
            (trainer.target_critic1, restored.target_critic1),
            (trainer.target_critic2, restored.target_critic2),
        ):
            for expected, actual in zip(expected_network.state_arrays(), actual_network.state_arrays(), strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

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

    def test_g1_timeout_preserves_flash_sac_transition(self) -> None:
        """Expose timeout truncation and pre-reset next observations for FlashSAC."""

        device = require_cuda_graph_capture("G1 FlashSAC timeout transition test")
        env = EnvG1PhoenX(
            ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                max_episode_steps=1,
                auto_reset=True,
                randomize_commands_on_reset=False,
                command_resample_steps=0,
                parse_visuals=False,
            ),
            device=device,
        )
        actions = wp.zeros((1, ACTION_DIM_G1), dtype=wp.float32, device=device)
        returned_obs, _rewards, dones = env.step(actions)
        np.testing.assert_array_equal(dones.numpy(), [1.0])
        np.testing.assert_array_equal(env.step_truncateds.numpy(), [1.0])
        np.testing.assert_array_equal(env.step_terminateds.numpy(), [0.0])
        self.assertEqual(env.step_next_obs.shape, returned_obs.shape)
        self.assertTrue(np.isfinite(env.step_next_obs.numpy()).all())

    def test_g1_flash_sac_and_ppo_trainer_workflows_smoke(self) -> None:
        """Exercise one PPO and FlashSAC update with G1-sized synthetic transitions."""

        device = require_cuda_graph_capture("G1 FlashSAC and PPO trainer smoke test")
        world_count = 4
        obs_np = np.linspace(-0.5, 0.5, world_count * OBS_DIM_G1, dtype=np.float32).reshape(world_count, OBS_DIM_G1)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        next_obs = wp.array(obs_np * 0.9, dtype=wp.float32, device=device)

        flash_config = ConfigFlashSAC(
            buffer_max_length=8,
            buffer_min_length=world_count,
            sample_batch_size=world_count,
            normalize_observations=False,
            normalize_rewards=False,
        )
        flash = TrainerFlashSAC(
            obs_dim=OBS_DIM_G1,
            action_dim=ACTION_DIM_G1,
            hidden_layers=(8,),
            config=flash_config,
            device=device,
            seed=71,
        )
        ppo = TrainerPPO(
            obs_dim=OBS_DIM_G1,
            action_dim=ACTION_DIM_G1,
            hidden_layers=(8,),
            config=ConfigPPO(
                train_epochs=1,
                normalize_advantages=False,
                normalize_observations=False,
            ),
            device=device,
            seed=73,
        )

        flash_actions, _flash_log_probs = flash.act(obs, seed=79)
        ppo_actions, ppo_log_probs, _ppo_values = ppo.act(obs, seed=83)
        self.assertEqual(flash_actions.shape, (world_count, ACTION_DIM_G1))
        self.assertEqual(ppo_actions.shape, (world_count, ACTION_DIM_G1))
        for actions in (flash_actions.numpy(), ppo_actions.numpy()):
            self.assertTrue(np.isfinite(actions).all())
            self.assertLessEqual(float(np.max(np.abs(actions))), 1.0)

        flash_updates = public_rl.train_flash_sac(_G1SmokeEnv(obs, next_obs), flash, interaction_steps=1, seed=97)
        self.assertTrue(flash.can_start_training())
        if flash.replay_buffer is None:
            self.fail("public FlashSAC workflow did not initialize replay")
        np.testing.assert_allclose(flash.replay_buffer.next_obs.numpy()[:world_count], next_obs.numpy())
        np.testing.assert_array_equal(flash.replay_buffer.dones.numpy()[:world_count], 0.0)

        buffer = BufferRollout(
            num_steps=1, num_envs=world_count, obs_dim=OBS_DIM_G1, action_dim=ACTION_DIM_G1, device=device
        )
        buffer.obs.assign(obs)
        buffer.actions.assign(ppo_actions)
        buffer.old_log_probs.assign(ppo_log_probs)
        buffer.advantages.assign(np.linspace(-0.2, 0.2, world_count, dtype=np.float32))
        buffer.returns.assign(np.linspace(-0.1, 0.3, world_count, dtype=np.float32))
        buffer.old_values.zero_()

        ppo_stats = ppo.update(buffer)
        flash_stats = flash_updates[0]
        self.assertTrue(all(math.isfinite(value) for value in ppo_stats.__dict__.values()))
        self.assertTrue(all(math.isfinite(value) for value in flash_stats.__dict__.values()))

    def test_real_g1_flash_sac_and_ppo_workflow_smoke(self) -> None:
        """Exercise one real G1 collection and update for each trainer workflow."""

        device = require_cuda_graph_capture("real G1 FlashSAC and PPO workflow smoke")
        env_kwargs = {
            "world_count": 1,
            "sim_substeps": 1,
            "solver_iterations": 1,
            "max_episode_steps": 1,
            "auto_reset": True,
            "randomize_commands_on_reset": False,
            "command_resample_steps": 0,
            "parse_visuals": False,
        }
        flash_env = EnvG1PhoenX(ConfigEnvG1PhoenX(**env_kwargs), device=device)
        ppo_env = EnvG1PhoenX(ConfigEnvG1PhoenX(**env_kwargs), device=device)
        flash_obs = flash_env.reset_noisy(seed=101)
        ppo_obs = ppo_env.reset_noisy(seed=211)
        self.assertEqual(flash_obs.shape, (1, OBS_DIM_G1))
        self.assertEqual(ppo_obs.shape, (1, OBS_DIM_G1))

        flash = TrainerFlashSAC(
            obs_dim=OBS_DIM_G1,
            action_dim=ACTION_DIM_G1,
            hidden_layers=(4,),
            config=ConfigFlashSAC(
                buffer_max_length=2,
                buffer_min_length=1,
                sample_batch_size=1,
                normalize_observations=False,
                normalize_rewards=False,
            ),
            device=device,
            seed=103,
        )
        ppo = TrainerPPO(
            obs_dim=OBS_DIM_G1,
            action_dim=ACTION_DIM_G1,
            hidden_layers=(4,),
            config=ConfigPPO(
                train_epochs=1,
                normalize_advantages=False,
                normalize_observations=False,
            ),
            device=device,
            seed=223,
        )

        flash_updates = public_rl.train_flash_sac(
            flash_env,
            flash,
            interaction_steps=1,
            seed=107,
            reset_at_start=False,
        )
        self.assertEqual(len(flash_updates), 1)
        if flash.replay_buffer is None:
            self.fail("real G1 FlashSAC collection did not initialize replay")
        self.assertEqual(flash.replay_buffer.actions.shape, (2, ACTION_DIM_G1))
        self.assertEqual(flash_env.step_next_obs.shape, (1, OBS_DIM_G1))
        np.testing.assert_allclose(flash.replay_buffer.next_obs.numpy()[:1], flash_env.step_next_obs.numpy())
        np.testing.assert_array_equal(flash_env.step_terminateds.numpy(), [0.0])
        np.testing.assert_array_equal(flash_env.step_truncateds.numpy(), [1.0])
        np.testing.assert_array_equal(flash.replay_buffer.dones.numpy()[:1], [0.0])

        rollout = BufferRollout(
            num_steps=1,
            num_envs=1,
            obs_dim=OBS_DIM_G1,
            action_dim=ACTION_DIM_G1,
            device=device,
        )
        ppo_env.collect_ppo_rollout(ppo, rollout, seed=227)
        self.assertEqual(rollout.obs.shape, (1, OBS_DIM_G1))
        self.assertEqual(rollout.actions.shape, (1, ACTION_DIM_G1))
        ppo_stats = ppo.update(rollout)
        self.assertTrue(all(math.isfinite(value) for value in flash_updates[0].__dict__.values()))
        self.assertTrue(all(math.isfinite(value) for value in ppo_stats.__dict__.values()))
        self.assertTrue(np.isfinite(rollout.obs.numpy()).all())
        self.assertTrue(np.isfinite(rollout.actions.numpy()).all())


if __name__ == "__main__":
    wp.init()
    unittest.main()
