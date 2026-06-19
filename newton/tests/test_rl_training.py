# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import tempfile
import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training.kernels import (
    mirrored_action_mse_grad_kernel,
    ppo_actor_loss_backward_kernel,
    zero_scalar_kernel,
)


def _rl_cuda_device():
    device = wp.get_preferred_device()
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise unittest.SkipTest("RL tests require CUDA graph capture with Warp mempool enabled")
    return device


class TestRolloutBuffer(unittest.TestCase):
    def test_compute_returns_matches_numpy_gae(self) -> None:
        device = _rl_cuda_device()
        buffer = rl.BufferRollout(num_steps=3, num_envs=2, obs_dim=4, action_dim=2, device=device)
        rewards = np.array([1.0, 0.5, 0.25, -0.25, 2.0, 1.0], dtype=np.float32)
        dones = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3], dtype=np.float32)
        successes = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buffer.rewards.assign(rewards)
        buffer.dones.assign(dones)
        buffer.successes.assign(successes)
        buffer.values.assign(values)

        gamma = 0.9
        gae_lambda = 0.8
        buffer.compute_returns(gamma=gamma, gae_lambda=gae_lambda)
        buffer.advantages.zero_()
        buffer.returns.zero_()
        with wp.ScopedCapture(device=device) as capture:
            buffer.compute_returns(gamma=gamma, gae_lambda=gae_lambda)
        wp.capture_launch(capture.graph)

        expected_adv = np.zeros(6, dtype=np.float32)
        expected_returns = np.zeros(6, dtype=np.float32)
        for env in range(2):
            gae = 0.0
            for t in reversed(range(3)):
                idx = t * 2 + env
                next_idx = (t + 1) * 2 + env
                non_terminal = 1.0 - float(dones[idx])
                delta = float(rewards[idx]) + gamma * float(values[next_idx]) * non_terminal - float(values[idx])
                gae = delta + gamma * gae_lambda * non_terminal * gae
                expected_adv[idx] = gae
                expected_returns[idx] = gae + float(values[idx])

        np.testing.assert_allclose(buffer.advantages.numpy(), expected_adv, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(buffer.returns.numpy(), expected_returns, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(buffer.successes.numpy(), successes, rtol=0.0, atol=0.0)

        buffer.advantages.zero_()
        buffer.returns.zero_()
        clipped_rewards = np.clip(rewards, -0.5, 0.5)
        with wp.ScopedCapture(device=device) as clipped_capture:
            buffer.compute_returns(gamma=gamma, gae_lambda=gae_lambda, reward_clip=0.5)
        wp.capture_launch(clipped_capture.graph)
        expected_clipped_adv = np.zeros(6, dtype=np.float32)
        expected_clipped_returns = np.zeros(6, dtype=np.float32)
        for env in range(2):
            gae = 0.0
            for t in reversed(range(3)):
                idx = t * 2 + env
                next_idx = (t + 1) * 2 + env
                non_terminal = 1.0 - float(dones[idx])
                delta = (
                    float(clipped_rewards[idx]) + gamma * float(values[next_idx]) * non_terminal - float(values[idx])
                )
                gae = delta + gamma * gae_lambda * non_terminal * gae
                expected_clipped_adv[idx] = gae
                expected_clipped_returns[idx] = gae + float(values[idx])

        np.testing.assert_allclose(buffer.advantages.numpy(), expected_clipped_adv, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(buffer.returns.numpy(), expected_clipped_returns, rtol=1.0e-6, atol=1.0e-6)

        ratios = np.array([0.5, 1.5, 4.0, 0.25, 2.0, 0.75], dtype=np.float32)
        buffer.ratios.assign(ratios)
        buffer.compute_vtrace_returns(gamma=gamma, gae_lambda=gae_lambda, rho_clip=1.5, c_clip=1.0, reward_clip=0.5)
        expected_vtrace_adv = np.zeros(6, dtype=np.float32)
        expected_vtrace_returns = np.zeros(6, dtype=np.float32)
        for env in range(2):
            trace = 0.0
            for t in reversed(range(3)):
                idx = t * 2 + env
                next_idx = (t + 1) * 2 + env
                non_terminal = 1.0 - float(dones[idx])
                rho = min(float(ratios[idx]), 1.5)
                c = min(float(ratios[idx]), 1.0)
                delta = rho * (
                    float(clipped_rewards[idx]) + gamma * float(values[next_idx]) * non_terminal - float(values[idx])
                )
                trace = delta + gamma * gae_lambda * c * trace * non_terminal
                expected_vtrace_adv[idx] = trace
                expected_vtrace_returns[idx] = trace + float(values[idx])

        np.testing.assert_allclose(buffer.advantages.numpy(), expected_vtrace_adv, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(buffer.returns.numpy(), expected_vtrace_returns, rtol=1.0e-6, atol=1.0e-6)


class TestTrainerPPO(unittest.TestCase):
    def test_manual_actor_backward_matches_tape_update(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(31)
        mirror_map = rl.MirrorMapPPO(
            obs_src=(0, 1, 2, 3),
            obs_sign=(1.0, 1.0, 1.0, 1.0),
            action_src=(1, 0),
            action_sign=(1.0, 1.0),
        )
        common = {
            "train_epochs": 1,
            "normalize_advantages": False,
            "actor_lr": 1.0e-3,
            "critic_lr": 1.0e-3,
            "entropy_coeff": 1.0e-4,
            "max_grad_norm": 0.0,
            "mirror_loss_coeff": 0.1,
        }
        trainer_tape = rl.TrainerPPO(
            obs_dim=4,
            action_dim=2,
            hidden_layers=(8,),
            config=rl.ConfigPPO(**common, manual_actor_backward=False),
            device=device,
            seed=13,
            mirror_map=mirror_map,
        )
        trainer_manual = rl.TrainerPPO(
            obs_dim=4,
            action_dim=2,
            hidden_layers=(8,),
            config=rl.ConfigPPO(**common, manual_actor_backward=True),
            device=device,
            seed=13,
            mirror_map=mirror_map,
        )
        buffers = [rl.BufferRollout(num_steps=4, num_envs=3, obs_dim=4, action_dim=2, device=device) for _ in range(2)]
        obs = rng.normal(size=(12, 4)).astype(np.float32)
        actions = np.tanh(0.35 * rng.normal(size=(12, 2))).astype(np.float32)
        advantages = rng.normal(loc=0.2, scale=0.4, size=12).astype(np.float32)
        for buffer in buffers:
            buffer.obs.assign(obs)
            buffer.actions.assign(actions)
            buffer.advantages.assign(advantages)
        _policy_out, old_log_probs = trainer_tape.actor.log_prob(
            buffers[0].obs, buffers[0].actions, requires_grad=False
        )
        old_log_probs_np = old_log_probs.numpy()
        buffers[0].old_log_probs.assign(old_log_probs_np)
        buffers[1].old_log_probs.assign(old_log_probs_np)

        stats_tape = trainer_tape._update_actor(buffers[0])
        stats_manual = trainer_manual._update_actor(buffers[1])

        self.assertAlmostEqual(stats_manual[0], stats_tape[0], places=5)
        self.assertAlmostEqual(stats_manual[1], stats_tape[1], places=5)
        self.assertAlmostEqual(stats_manual[2], stats_tape[2], places=6)
        for manual_param, tape_param in zip(
            trainer_manual.actor.parameters(), trainer_tape.actor.parameters(), strict=True
        ):
            np.testing.assert_allclose(manual_param.numpy(), tape_param.numpy(), rtol=2.0e-4, atol=2.0e-5)

    def test_manual_actor_loss_backward_is_graph_capturable(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(47)
        rows = 8
        action_dim = 2
        policy_out = wp.array(rng.normal(size=(rows, action_dim)).astype(np.float32), dtype=wp.float32, device=device)
        mirrored_policy_out = wp.array(
            rng.normal(size=(rows, action_dim)).astype(np.float32), dtype=wp.float32, device=device
        )
        log_std = wp.array(np.array([-0.2, 0.1], dtype=np.float32), dtype=wp.float32, device=device)
        actions = wp.array(
            np.tanh(0.3 * rng.normal(size=(rows, action_dim))).astype(np.float32), dtype=wp.float32, device=device
        )
        old_log_probs = wp.array(rng.normal(size=rows).astype(np.float32), dtype=wp.float32, device=device)
        advantages = wp.array(rng.normal(size=rows).astype(np.float32), dtype=wp.float32, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device)
        approx_kl = wp.zeros(1, dtype=wp.float32, device=device)
        clip_fraction = wp.zeros(1, dtype=wp.float32, device=device)
        ratios = wp.zeros(rows, dtype=wp.float32, device=device)
        policy_out_grad = wp.zeros((rows, action_dim), dtype=wp.float32, device=device)
        log_std_grad = wp.zeros(action_dim, dtype=wp.float32, device=device)
        mirror_src = wp.array(np.array([1, 0], dtype=np.int32), dtype=wp.int32, device=device)
        mirror_sign = wp.array(np.array([1.0, 1.0], dtype=np.float32), dtype=wp.float32, device=device)

        def launch_manual_backward() -> None:
            wp.launch(zero_scalar_kernel, dim=1, outputs=[loss], device=device)
            wp.launch(zero_scalar_kernel, dim=1, outputs=[approx_kl], device=device)
            wp.launch(zero_scalar_kernel, dim=1, outputs=[clip_fraction], device=device)
            log_std_grad.zero_()
            wp.launch(
                ppo_actor_loss_backward_kernel,
                dim=rows,
                inputs=[
                    policy_out,
                    log_std,
                    actions,
                    old_log_probs,
                    advantages,
                    0.2,
                    1.0e-4,
                    action_dim,
                    0,
                    1,
                    -5.0,
                    2.0,
                    rows,
                ],
                outputs=[loss, approx_kl, clip_fraction, ratios, policy_out_grad, log_std_grad],
                device=device,
            )
            wp.launch(
                mirrored_action_mse_grad_kernel,
                dim=(rows, action_dim),
                inputs=[policy_out, mirrored_policy_out, mirror_src, mirror_sign, action_dim, 0.1, rows],
                outputs=[policy_out_grad, loss],
                device=device,
            )

        launch_manual_backward()
        policy_out_grad.zero_()
        with wp.ScopedCapture(device=device) as capture:
            launch_manual_backward()
        wp.capture_launch(capture.graph)

        self.assertTrue(math.isfinite(float(loss.numpy()[0])))
        self.assertTrue(math.isfinite(float(approx_kl.numpy()[0])))
        self.assertGreater(float(np.linalg.norm(policy_out_grad.numpy())), 0.0)
        self.assertGreater(float(np.linalg.norm(log_std_grad.numpy())), 0.0)

    def test_update_changes_actor_and_returns_finite_stats(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(3)
        config = rl.ConfigPPO(
            train_epochs=2,
            normalize_advantages=False,
            actor_lr=3.0e-3,
            critic_lr=3.0e-3,
            entropy_coeff=0.0,
            minibatch_size=8,
            replay_ratio=1.0,
            priority_alpha=0.4,
            priority_beta=1.0,
            vtrace_rho_clip=3.0,
            vtrace_c_clip=3.0,
            max_grad_norm=0.3,
            mirror_loss_coeff=0.1,
            manual_actor_backward=True,
        )
        mirror_map = rl.MirrorMapPPO(
            obs_src=(0, 1, 2, 3, 4),
            obs_sign=(1.0, 1.0, 1.0, 1.0, 1.0),
            action_src=(1, 0),
            action_sign=(1.0, 1.0),
        )
        trainer = rl.TrainerPPO(
            obs_dim=5, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=7, mirror_map=mirror_map
        )
        buffer = rl.BufferRollout(num_steps=4, num_envs=4, obs_dim=5, action_dim=2, device=device)
        n = buffer.num_samples
        obs = rng.normal(size=(n, 5)).astype(np.float32)
        actions = np.tanh(0.5 * rng.normal(size=(n, 2))).astype(np.float32)
        advantages = rng.normal(loc=0.5, scale=0.25, size=n).astype(np.float32)
        returns = rng.normal(size=n).astype(np.float32)
        buffer.obs.assign(obs)
        buffer.actions.assign(actions)
        buffer.advantages.assign(advantages)
        buffer.returns.assign(returns)
        _policy_out, old_log_probs = trainer.actor.log_prob(buffer.obs, buffer.actions, requires_grad=False)
        buffer.old_log_probs.assign(old_log_probs.numpy())

        actor_before = trainer.actor.net.weights[0].numpy().copy()
        stats = trainer.update(buffer)
        actor_after = trainer.actor.net.weights[0].numpy()

        self.assertTrue(math.isfinite(stats.policy_loss))
        self.assertTrue(math.isfinite(stats.value_loss))
        self.assertTrue(math.isfinite(stats.approx_kl))
        self.assertTrue(math.isfinite(stats.clip_fraction))
        self.assertGreater(float(np.max(np.abs(actor_after - actor_before))), 0.0)

    def test_checkpoint_round_trip_preserves_parameters(self) -> None:
        device = _rl_cuda_device()
        config = rl.ConfigPPO(train_epochs=1, normalize_advantages=False)
        trainer = rl.TrainerPPO(obs_dim=5, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=17)
        actor_before = [param.numpy().copy() for param in trainer.actor.parameters()]
        critic_before = [param.numpy().copy() for param in trainer.critic.parameters()]
        trainer.actor_optimizer.step_count = 3
        trainer.critic_optimizer.step_count = 5
        trainer.iteration = 9

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/ppo_checkpoint.npz"
            trainer.save_checkpoint(path, iteration=11)
            restored = rl.load_ppo_checkpoint(path, device=device)

        actor_after = [param.numpy().copy() for param in restored.actor.parameters()]
        critic_after = [param.numpy().copy() for param in restored.critic.parameters()]
        for before, after in zip(actor_before, actor_after, strict=True):
            np.testing.assert_allclose(after, before, rtol=0.0, atol=0.0)
        for before, after in zip(critic_before, critic_after, strict=True):
            np.testing.assert_allclose(after, before, rtol=0.0, atol=0.0)
        self.assertEqual(restored.actor_optimizer.step_count, 3)
        self.assertEqual(restored.critic_optimizer.step_count, 5)
        self.assertEqual(restored.iteration, 11)
        self.assertEqual(restored.config.minibatch_size, 0)
        self.assertEqual(restored.config.replay_ratio, 0.0)
        self.assertEqual(restored.config.priority_alpha, 0.0)
        self.assertEqual(restored.config.priority_beta, 0.0)
        self.assertFalse(restored.config.manual_actor_backward)
        self.assertEqual(restored.config.vtrace_rho_clip, 0.0)
        self.assertEqual(restored.config.vtrace_c_clip, 0.0)
        self.assertEqual(restored.config.reward_clip, 0.0)
        self.assertEqual(restored.config.max_grad_norm, 0.0)
        self.assertEqual(restored.config.mirror_loss_coeff, 0.0)


class TestReplayBufferSAC(unittest.TestCase):
    def test_sample_returns_inserted_rows(self) -> None:
        device = _rl_cuda_device()
        replay = rl.BufferReplaySAC(capacity=8, obs_dim=3, action_dim=2, batch_size=4, device=device)
        obs = np.arange(18, dtype=np.float32).reshape(6, 3)
        actions = np.arange(12, dtype=np.float32).reshape(6, 2) * 0.1
        rewards = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
        dones = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        next_obs = obs + 100.0
        replay.add_batch(
            wp.array(obs, dtype=wp.float32, device=device),
            wp.array(actions, dtype=wp.float32, device=device),
            wp.array(rewards, dtype=wp.float32, device=device),
            wp.array(dones, dtype=wp.float32, device=device),
            wp.array(next_obs, dtype=wp.float32, device=device),
        )

        batch = replay.sample(seed=11)
        self.assertEqual(batch.obs.shape, (4, 3))
        self.assertEqual(batch.actions.shape, (4, 2))
        sampled_obs = batch.obs.numpy()
        valid_first_columns = {float(v) for v in obs[:, 0]}
        self.assertTrue(all(float(v) in valid_first_columns for v in sampled_obs[:, 0]))
        np.testing.assert_allclose(batch.next_obs.numpy(), sampled_obs + 100.0, rtol=0.0, atol=0.0)


class TestTrainerSAC(unittest.TestCase):
    def test_update_changes_networks_and_returns_finite_stats(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(5)
        config = rl.ConfigSAC(update_steps=2, tau=0.5, actor_lr=1.0e-3, critic_lr=1.0e-3, alpha_lr=1.0e-3)
        trainer = rl.TrainerSAC(obs_dim=4, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=13)
        obs = wp.array(rng.normal(size=(16, 4)).astype(np.float32), dtype=wp.float32, device=device)
        actions = wp.array(np.tanh(rng.normal(size=(16, 2))).astype(np.float32), dtype=wp.float32, device=device)
        rewards = wp.array(rng.normal(size=16).astype(np.float32), dtype=wp.float32, device=device)
        dones = wp.array(np.zeros(16, dtype=np.float32), dtype=wp.float32, device=device)
        next_obs = wp.array(rng.normal(size=(16, 4)).astype(np.float32), dtype=wp.float32, device=device)
        batch = rl.BatchSAC(obs=obs, actions=actions, rewards=rewards, dones=dones, next_obs=next_obs)

        actor_before = [param.numpy().copy() for param in trainer.actor.parameters()]
        critic_before = [param.numpy().copy() for param in trainer.critic1.parameters()]
        target_before = [param.numpy().copy() for param in trainer.target_critic1.parameters()]
        stats = trainer.update(batch, seed=17)

        actor_delta = max(
            float(np.max(np.abs(param.numpy() - before)))
            for param, before in zip(trainer.actor.parameters(), actor_before, strict=True)
        )
        critic_delta = max(
            float(np.max(np.abs(param.numpy() - before)))
            for param, before in zip(trainer.critic1.parameters(), critic_before, strict=True)
        )
        target_delta = max(
            float(np.max(np.abs(param.numpy() - before)))
            for param, before in zip(trainer.target_critic1.parameters(), target_before, strict=True)
        )
        self.assertTrue(math.isfinite(stats.actor_loss))
        self.assertTrue(math.isfinite(stats.critic_loss))
        self.assertTrue(math.isfinite(stats.alpha_loss))
        self.assertGreater(stats.alpha, 0.0)
        self.assertGreater(actor_delta, 0.0)
        self.assertGreater(critic_delta, 0.0)
        self.assertGreater(target_delta, 0.0)


if __name__ == "__main__":
    wp.init()
    unittest.main()
