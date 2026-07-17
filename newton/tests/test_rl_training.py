# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import tempfile
import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training.env import advance_seed_counter, make_seed_counter
from newton._src.solvers.phoenx.rl_training.kernels import (
    PPO_LOG_STD_PARTIAL_BATCH,
    gather_flat_minibatch_kernel,
    mirrored_action_mse_grad_kernel,
    ppo_actor_loss_backward_kernel,
    ppo_log_std_grad_partials_kernel,
    reduce_ppo_log_std_grad_kernel,
    sac_actor_q_backward_kernel,
    sac_distributional_projection_kernel,
    sac_distributional_q_value_kernel,
    sac_q_target_kernel,
    value_column_loss_grad_kernel,
    value_column_symmetry_loss_grad_kernel,
    value_loss_grad_kernel,
    value_symmetry_loss_grad_kernel,
    zero_scalar_kernel,
)


@wp.kernel
def _fill_1d_float_kernel(value: wp.float32, out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = value


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

    def test_normalize_advantages_is_graph_capturable(self) -> None:
        device = _rl_cuda_device()
        buffer = rl.BufferRollout(num_steps=3, num_envs=4, obs_dim=2, action_dim=1, device=device)
        advantages_np = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 4.0, 2.0, -3.0, 0.25, 1.5, -0.5, 2.5], dtype=np.float32)
        buffer.advantages.assign(advantages_np)

        with wp.ScopedCapture(device=device) as capture:
            buffer.normalize_advantages()
        wp.capture_launch(capture.graph)

        expected = (advantages_np - float(np.mean(advantages_np))) / float(
            np.sqrt(np.var(advantages_np, ddof=1) + 1.0e-8)
        )
        np.testing.assert_allclose(buffer.advantages.numpy(), expected, rtol=2.0e-6, atol=2.0e-6)

    def test_reward_done_success_sums_are_graph_capturable(self) -> None:
        device = _rl_cuda_device()
        buffer = rl.BufferRollout(num_steps=3, num_envs=2, obs_dim=2, action_dim=1, device=device)
        rewards = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 4.0], dtype=np.float32)
        dones = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        successes = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.0], dtype=np.float32)
        buffer.rewards.assign(rewards)
        buffer.dones.assign(dones)
        buffer.successes.assign(successes)

        buffer.compute_reward_done_success_sums()
        with wp.ScopedCapture(device=device) as capture:
            buffer.compute_reward_done_success_sums()
        wp.capture_launch(capture.graph)

        expected_sums = np.array([np.sum(rewards), np.sum(dones), np.sum(successes)], dtype=np.float32)
        np.testing.assert_allclose(
            buffer.compute_reward_done_success_sums().numpy(), expected_sums, rtol=0.0, atol=1.0e-6
        )

        with wp.ScopedCapture(device=device) as copy_capture:
            buffer.copy_reward_done_success_sums_to_host()
        wp.capture_launch(copy_capture.graph)
        wp.synchronize_device(device)
        np.testing.assert_allclose(buffer._metric_sums_host.numpy(), expected_sums, rtol=0.0, atol=1.0e-6)

        mean_reward, mean_done, mean_success = buffer.reward_done_success_means()
        self.assertAlmostEqual(mean_reward, float(np.mean(rewards)), places=6)
        self.assertAlmostEqual(mean_done, float(np.mean(dones)), places=6)
        self.assertAlmostEqual(mean_success, float(np.mean(successes)), places=6)


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
            "actor_lr": 1.0e-5,
            "critic_lr": 1.0e-5,
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
        d_log_prob_rows = wp.zeros(rows, dtype=wp.float32, device=device)
        partial_count = max((rows + PPO_LOG_STD_PARTIAL_BATCH - 1) // PPO_LOG_STD_PARTIAL_BATCH, 1)
        log_std_grad_partials = wp.zeros((partial_count, action_dim), dtype=wp.float32, device=device)
        mirror_src = wp.array(np.array([1, 0], dtype=np.int32), dtype=wp.int32, device=device)
        mirror_sign = wp.array(np.array([1.0, 1.0], dtype=np.float32), dtype=wp.float32, device=device)

        entropy_coeff_buf = wp.array([1.0e-4], dtype=wp.float32, device=device)
        mirror_coeff_buf = wp.array([0.1], dtype=wp.float32, device=device)

        def launch_manual_backward() -> None:
            wp.launch(zero_scalar_kernel, dim=1, outputs=[loss], device=device)
            wp.launch(zero_scalar_kernel, dim=1, outputs=[approx_kl], device=device)
            wp.launch(zero_scalar_kernel, dim=1, outputs=[clip_fraction], device=device)
            log_std_grad_partials.zero_()
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
                    entropy_coeff_buf,
                    action_dim,
                    0,
                    1,
                    -5.0,
                    2.0,
                    rows,
                ],
                outputs=[loss, approx_kl, clip_fraction, ratios, policy_out_grad, d_log_prob_rows],
                device=device,
            )
            wp.launch(
                ppo_log_std_grad_partials_kernel,
                dim=(partial_count, action_dim),
                inputs=[
                    policy_out,
                    log_std,
                    actions,
                    d_log_prob_rows,
                    entropy_coeff_buf,
                    action_dim,
                    1,
                    -5.0,
                    2.0,
                    rows,
                ],
                outputs=[log_std_grad_partials],
                device=device,
            )
            wp.launch(
                reduce_ppo_log_std_grad_kernel,
                dim=action_dim,
                inputs=[log_std_grad_partials, partial_count],
                outputs=[log_std_grad],
                device=device,
            )
            wp.launch(
                mirrored_action_mse_grad_kernel,
                dim=rows,
                inputs=[policy_out, mirrored_policy_out, mirror_src, mirror_sign, action_dim, mirror_coeff_buf, rows],
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

    def test_manual_critic_loss_backward_is_graph_capturable(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(53)
        rows = 8
        coeff = 0.25
        values_np = rng.normal(size=(rows, 1)).astype(np.float32)
        returns_np = rng.normal(size=rows).astype(np.float32)
        mirrored_np = rng.normal(size=(rows, 1)).astype(np.float32)
        values = wp.array(values_np, dtype=wp.float32, device=device)
        returns = wp.array(returns_np, dtype=wp.float32, device=device)
        mirrored = wp.array(mirrored_np, dtype=wp.float32, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device)
        value_grad = wp.zeros((rows, 1), dtype=wp.float32, device=device)

        coeff_buf = wp.array([coeff], dtype=wp.float32, device=device)
        old_values = wp.zeros(rows, dtype=wp.float32, device=device)

        def launch_manual_backward() -> None:
            wp.launch(zero_scalar_kernel, dim=1, outputs=[loss], device=device)
            wp.launch(
                value_loss_grad_kernel,
                dim=rows,
                inputs=[values, old_values, returns, 1.0, 0.0, rows],
                outputs=[loss, value_grad],
                device=device,
            )
            wp.launch(
                value_symmetry_loss_grad_kernel,
                dim=rows,
                inputs=[values, mirrored, coeff_buf, rows],
                outputs=[loss, value_grad],
                device=device,
            )

        launch_manual_backward()
        value_grad.zero_()
        with wp.ScopedCapture(device=device) as capture:
            launch_manual_backward()
        wp.capture_launch(capture.graph)

        expected_delta = values_np[:, 0] - returns_np
        expected_mirror_delta = values_np[:, 0] - mirrored_np[:, 0]
        expected_loss = 0.5 * np.mean(expected_delta * expected_delta)
        expected_loss += 0.5 * coeff * np.mean(expected_mirror_delta * expected_mirror_delta)
        expected_grad = ((expected_delta + coeff * expected_mirror_delta) / float(rows)).reshape(rows, 1)
        self.assertAlmostEqual(float(loss.numpy()[0]), float(expected_loss), places=6)
        np.testing.assert_allclose(value_grad.numpy(), expected_grad, rtol=1.0e-6, atol=1.0e-6)

    def test_shared_value_column_loss_backward_is_graph_capturable(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(59)
        rows = 8
        action_dim = 2
        value_col = action_dim
        coeff = 0.25
        values_np = rng.normal(size=(rows, action_dim + 1)).astype(np.float32)
        mirrored_np = rng.normal(size=(rows, action_dim + 1)).astype(np.float32)
        returns_np = rng.normal(size=rows).astype(np.float32)
        values = wp.array(values_np, dtype=wp.float32, device=device)
        mirrored = wp.array(mirrored_np, dtype=wp.float32, device=device)
        returns = wp.array(returns_np, dtype=wp.float32, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device)
        output_grad = wp.zeros((rows, action_dim + 1), dtype=wp.float32, device=device)

        coeff_buf = wp.array([coeff], dtype=wp.float32, device=device)
        old_values = wp.zeros(rows, dtype=wp.float32, device=device)

        def launch_manual_backward() -> None:
            wp.launch(zero_scalar_kernel, dim=1, outputs=[loss], device=device)
            output_grad.zero_()
            wp.launch(
                value_column_loss_grad_kernel,
                dim=rows,
                inputs=[values, value_col, old_values, returns, 1.0, 0.0, rows],
                outputs=[loss, output_grad],
                device=device,
            )
            wp.launch(
                value_column_symmetry_loss_grad_kernel,
                dim=rows,
                inputs=[values, value_col, mirrored, coeff_buf, rows],
                outputs=[loss, output_grad],
                device=device,
            )

        launch_manual_backward()
        with wp.ScopedCapture(device=device) as capture:
            launch_manual_backward()
        wp.capture_launch(capture.graph)

        expected_delta = values_np[:, value_col] - returns_np
        expected_mirror_delta = values_np[:, value_col] - mirrored_np[:, value_col]
        expected_loss = 0.5 * np.mean(expected_delta * expected_delta)
        expected_loss += 0.5 * coeff * np.mean(expected_mirror_delta * expected_mirror_delta)
        expected_grad = np.zeros((rows, action_dim + 1), dtype=np.float32)
        expected_grad[:, value_col] = (expected_delta + coeff * expected_mirror_delta) / float(rows)
        self.assertAlmostEqual(float(loss.numpy()[0]), float(expected_loss), places=6)
        np.testing.assert_allclose(output_grad.numpy(), expected_grad, rtol=1.0e-6, atol=1.0e-6)

    def test_manual_mlp_backward_matches_numpy_and_graph_captures(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(71)
        x_np = rng.normal(size=(9, 3)).astype(np.float32)
        output_grad_np = rng.normal(size=(9, 2)).astype(np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=device)
        output_grad = wp.array(output_grad_np, dtype=wp.float32, device=device)
        manual_mlp = rl.WarpMLP((3, 5, 4, 2), activation="tanh", output_activation="tanh", device=device, seed=19)

        manual_mlp.forward_manual(x)
        manual_mlp.backward_manual(output_grad)
        with wp.ScopedCapture(device=device) as capture:
            manual_mlp.forward_manual(x)
            manual_mlp.backward_manual(output_grad)
        wp.capture_launch(capture.graph)
        manual_out = manual_mlp._manual_outputs[-1].numpy()
        manual_grads = [param.grad.numpy().copy() for param in manual_mlp.parameters()]

        weights = [weight.numpy() for weight in manual_mlp.weights]
        biases = [bias.numpy() for bias in manual_mlp.biases]
        activations = [x_np]
        for weight, bias in zip(weights, biases, strict=True):
            activations.append(np.tanh(activations[-1] @ weight + bias))

        grad = output_grad_np * (1.0 - activations[-1] * activations[-1])
        expected_pairs = []
        for layer in reversed(range(len(weights))):
            expected_pairs.append((activations[layer].T @ grad, grad.sum(axis=0)))
            if layer > 0:
                grad = (grad @ weights[layer].T) * (1.0 - activations[layer] * activations[layer])
        expected_grads = [grad for pair in reversed(expected_pairs) for grad in pair]

        np.testing.assert_allclose(manual_out, activations[-1], rtol=1.0e-6, atol=1.0e-6)
        for manual_grad, expected_grad in zip(manual_grads, expected_grads, strict=True):
            np.testing.assert_allclose(manual_grad, expected_grad, rtol=1.0e-5, atol=1.0e-5)

    def test_manual_mlp_reserved_capacity_graph_captures(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(75)
        x_np = rng.normal(size=(64, 3)).astype(np.float32)
        output_grad_np = rng.normal(size=(64, 2)).astype(np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=device)
        output_grad = wp.array(output_grad_np, dtype=wp.float32, device=device)
        manual_mlp = rl.WarpMLP((3, 5, 4, 2), activation="tanh", output_activation="tanh", device=device, seed=21)
        manual_mlp.reserve_buffers(128)

        manual_mlp.forward_manual(x)
        manual_mlp.backward_manual(output_grad)
        with wp.ScopedCapture(device=device) as capture:
            manual_mlp.forward_manual(x)
            manual_mlp.backward_manual(output_grad)
        wp.capture_launch(capture.graph)
        manual_out = manual_mlp._manual_outputs[-1].numpy()[: x_np.shape[0]]
        manual_grads = [param.grad.numpy().copy() for param in manual_mlp.parameters()]

        weights = [weight.numpy() for weight in manual_mlp.weights]
        biases = [bias.numpy() for bias in manual_mlp.biases]
        activations = [x_np]
        for weight, bias in zip(weights, biases, strict=True):
            activations.append(np.tanh(activations[-1] @ weight + bias))

        grad = output_grad_np * (1.0 - activations[-1] * activations[-1])
        expected_pairs = []
        for layer in reversed(range(len(weights))):
            expected_pairs.append((activations[layer].T @ grad, grad.sum(axis=0)))
            if layer > 0:
                grad = (grad @ weights[layer].T) * (1.0 - activations[layer] * activations[layer])
        expected_grads = [grad for pair in reversed(expected_pairs) for grad in pair]

        np.testing.assert_allclose(manual_out, activations[-1], rtol=1.0e-6, atol=1.0e-6)
        for manual_grad, expected_grad in zip(manual_grads, expected_grads, strict=True):
            np.testing.assert_allclose(manual_grad, expected_grad, rtol=1.0e-5, atol=1.0e-5)

    def test_mlp_reuse_forward_survives_manual_backward_graph_capture(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(79)
        x_reuse_np = rng.normal(size=(16, 4)).astype(np.float32)
        x_manual_np = rng.normal(size=(16, 4)).astype(np.float32)
        output_grad_np = rng.normal(size=(16, 3)).astype(np.float32)
        x_reuse = wp.array(x_reuse_np, dtype=wp.float32, device=device)
        x_manual = wp.array(x_manual_np, dtype=wp.float32, device=device)
        output_grad = wp.array(output_grad_np, dtype=wp.float32, device=device)
        manual_mlp = rl.WarpMLP((4, 6, 3), activation="tanh", output_activation="tanh", device=device, seed=23)
        manual_mlp.reserve_buffers(64)

        def run_reuse_then_manual() -> wp.array2d[wp.float32]:
            reuse_out = manual_mlp.forward_reuse(x_reuse)
            manual_mlp.forward_manual(x_manual)
            manual_mlp.backward_manual(output_grad)
            return reuse_out

        reuse_out = run_reuse_then_manual()
        with wp.ScopedCapture(device=device) as capture:
            reuse_out = run_reuse_then_manual()
        wp.capture_launch(capture.graph)

        weights = [weight.numpy() for weight in manual_mlp.weights]
        biases = [bias.numpy() for bias in manual_mlp.biases]
        expected = x_reuse_np
        for weight, bias in zip(weights, biases, strict=True):
            expected = np.tanh(expected @ weight + bias)

        np.testing.assert_allclose(reuse_out.numpy()[: x_reuse_np.shape[0]], expected, rtol=1.0e-6, atol=1.0e-6)

    def test_bfloat16_manual_mlp_weight_grad_graph_captures(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(83)
        # Exercise every split-K chunk, not just the first batch tile.
        x_np = rng.normal(size=(1536, 17)).astype(np.float32)
        output_grad_np = rng.normal(size=(1536, 5)).astype(np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=device)
        output_grad = wp.array(output_grad_np, dtype=wp.float32, device=device)
        fp32_mlp = rl.WarpMLP(
            (17, 19, 5),
            activation="relu",
            output_activation="linear",
            device=device,
            seed=29,
            manual_weight_grad_dtype="float32",
        )
        bf16_mlp = rl.WarpMLP(
            (17, 19, 5),
            activation="relu",
            output_activation="linear",
            device=device,
            seed=29,
            manual_weight_grad_dtype="bfloat16",
        )

        fp32_mlp.forward_manual(x)
        fp32_mlp.backward_manual(output_grad)
        with wp.ScopedCapture(device=device) as capture:
            bf16_mlp.forward_manual(x)
            bf16_mlp.backward_manual(output_grad)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(
            bf16_mlp._manual_outputs[-1].numpy(), fp32_mlp._manual_outputs[-1].numpy(), rtol=0.0, atol=0.0
        )
        bf16_grads = []
        for bf16_param, fp32_param in zip(bf16_mlp.parameters(), fp32_mlp.parameters(), strict=True):
            bf16_grad = bf16_param.grad.numpy()
            np.testing.assert_allclose(bf16_grad, fp32_param.grad.numpy(), rtol=3.0e-2, atol=2.0e-1)
            bf16_grads.append(bf16_grad.copy())
        wp.capture_launch(capture.graph)
        for bf16_param, expected_grad in zip(bf16_mlp.parameters(), bf16_grads, strict=True):
            np.testing.assert_array_equal(bf16_param.grad.numpy(), expected_grad)
        with self.assertRaises(ValueError):
            rl.WarpMLP((2, 3), device=device, manual_weight_grad_dtype="float16")

    def test_bfloat16_manual_mlp_forward_graph_captures(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(89)
        x_np = rng.normal(size=(16_384, 17)).astype(np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=device)
        fp32_mlp = rl.WarpMLP(
            (17, 64, 5),
            activation="relu",
            output_activation="linear",
            device=device,
            seed=31,
            manual_forward_dtype="float32",
        )
        bf16_mlp = rl.WarpMLP(
            (17, 64, 5),
            activation="relu",
            output_activation="linear",
            device=device,
            seed=31,
            manual_forward_dtype="bfloat16",
        )

        fp32_out = fp32_mlp.forward_manual(x)
        with wp.ScopedCapture(device=device) as capture:
            bf16_out = bf16_mlp.forward_manual(x)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(bf16_out.numpy(), fp32_out.numpy(), rtol=8.0e-3, atol=2.0e-2)
        with self.assertRaises(ValueError):
            rl.WarpMLP((2, 64, 3), device=device, manual_forward_dtype="float16")

    def test_adam_step_count_advances_inside_graph(self) -> None:
        device = _rl_cuda_device()
        initial = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        param_graph = wp.array(initial, dtype=wp.float32, device=device, requires_grad=True)
        param_eager = wp.array(initial, dtype=wp.float32, device=device, requires_grad=True)
        opt_graph = rl.Adam([param_graph], lr=0.01, beta1=0.8, beta2=0.9)
        opt_eager = rl.Adam([param_eager], lr=0.01, beta1=0.8, beta2=0.9)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(_fill_1d_float_kernel, dim=param_graph.shape[0], inputs=[0.25], outputs=[param_graph.grad])
            opt_graph.step()

        for _ in range(2):
            wp.capture_launch(capture.graph)
        for _ in range(2):
            wp.launch(
                _fill_1d_float_kernel,
                dim=param_eager.shape[0],
                inputs=[0.25],
                outputs=[param_eager.grad],
                device=device,
            )
            opt_eager.step()

        np.testing.assert_allclose(param_graph.numpy(), param_eager.numpy(), rtol=1.0e-6, atol=1.0e-6)
        self.assertEqual(opt_graph.step_count, 2)
        self.assertEqual(opt_eager.step_count, 2)

    def test_actor_seed_counter_advances_inside_graph(self) -> None:
        device = _rl_cuda_device()
        actor = rl.GaussianActor(obs_dim=3, action_dim=2, hidden_layers=(8,), device=device, seed=11)
        obs = wp.array(np.ones((4, 3), dtype=np.float32), dtype=wp.float32, device=device)
        seed_counter = make_seed_counter(17, device=device)
        actor.reserve_reuse_buffers(4)

        with wp.ScopedCapture(device=device) as capture:
            actor.sample_reuse_seed_counter(obs, seed_counter=seed_counter)
            advance_seed_counter(seed_counter, 1, device=device)

        wp.capture_launch(capture.graph)
        first = actor._sample_reuse_actions.numpy().copy()
        wp.capture_launch(capture.graph)
        second = actor._sample_reuse_actions.numpy().copy()

        self.assertFalse(np.allclose(first, second))
        np.testing.assert_array_equal(seed_counter.numpy(), np.array([19], dtype=np.int32))

    def test_adaptive_kl_lr_updates_inside_graph(self) -> None:
        device = _rl_cuda_device()
        config = rl.ConfigPPO(
            adaptive_kl_target=0.01,
            adaptive_kl_factor=1.5,
            adaptive_kl_min_lr_ratio=0.02,
            adaptive_kl_max_lr_ratio=20.0,
        )
        trainer = rl.TrainerPPO(obs_dim=2, action_dim=1, hidden_layers=(4,), config=config, device=device)
        trainer._approx_kl.assign(np.array([0.03], dtype=np.float32))
        trainer._adapt_lr_to_kl()
        trainer._adaptive_lr_scale.assign(np.array([1.0], dtype=np.float32))
        trainer._apply_lr_schedule(8)

        with wp.ScopedCapture(device=device) as capture:
            trainer._adapt_lr_to_kl()
        wp.capture_launch(capture.graph)

        expected_reduced = 1.0 / 1.5
        self.assertAlmostEqual(float(trainer._adaptive_lr_scale.numpy()[0]), expected_reduced, places=6)
        self.assertAlmostEqual(float(trainer.actor_optimizer.lr_scale.numpy()[0]), expected_reduced, places=6)
        self.assertAlmostEqual(float(trainer.critic_optimizer.lr_scale.numpy()[0]), expected_reduced, places=6)

        trainer._approx_kl.assign(np.array([0.0], dtype=np.float32))
        trainer._adapt_lr_to_kl()
        self.assertAlmostEqual(float(trainer._adaptive_lr_scale.numpy()[0]), expected_reduced, places=6)

        trainer._approx_kl.assign(np.array([0.001], dtype=np.float32))
        trainer._adapt_lr_to_kl()
        self.assertAlmostEqual(float(trainer._adaptive_lr_scale.numpy()[0]), 1.0, places=6)

    def test_flat_minibatches_partition_every_transition_once(self) -> None:
        device = _rl_cuda_device()
        buffer = rl.BufferRollout(num_steps=2, num_envs=4, obs_dim=1, action_dim=1, device=device)
        rows = np.arange(buffer.num_samples, dtype=np.float32)
        buffer.obs.assign(rows[:, None])
        buffer.actions.assign((-rows)[:, None])
        buffer.old_log_probs.assign(rows + 10.0)
        buffer.advantages.assign(rows + 20.0)
        buffer.returns.assign(rows + 30.0)
        buffer.values.assign(np.arange(buffer.values.shape[0], dtype=np.float32) + 40.0)
        buffer.dones.assign(rows % 2.0)
        trainer = rl.TrainerPPO(obs_dim=1, action_dim=1, hidden_layers=(4,), device=device, seed=3)
        batch = trainer._ensure_minibatch(buffer, segment_count=2)
        iteration = wp.array([0], dtype=wp.int32, device=device)

        gathered = []
        for minibatch_id in range(2):
            wp.launch(
                gather_flat_minibatch_kernel,
                dim=(batch.num_samples, 1),
                inputs=[
                    iteration,
                    0,
                    minibatch_id,
                    batch.num_samples,
                    buffer.num_samples,
                    3,
                    1,
                    1,
                    buffer.obs,
                    buffer.actions,
                    buffer.old_log_probs,
                    buffer.advantages,
                    buffer.returns,
                    buffer.old_values,
                    buffer.dones,
                ],
                outputs=[
                    batch.obs,
                    batch.actions,
                    batch.old_log_probs,
                    batch.advantages,
                    batch.returns,
                    batch.old_values,
                    batch.dones,
                ],
                device=device,
            )
            gathered.append(batch.obs.numpy()[:, 0].copy())

        first, second = gathered
        self.assertEqual(len(np.intersect1d(first, second)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate(gathered)), rows)

    def test_priority_replay_sampling_seed_counter_advances_inside_graph(self) -> None:
        device = _rl_cuda_device()
        config = rl.ConfigPPO(
            minibatch_size=128,
            replay_ratio=1.0,
            priority_alpha=0.5,
            priority_beta=0.75,
            normalize_advantages=False,
        )
        trainer = rl.TrainerPPO(obs_dim=3, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=101)
        buffer = rl.BufferRollout(num_steps=4, num_envs=5, obs_dim=3, action_dim=2, device=device)
        per_env_priority = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
        buffer.advantages.assign(np.tile(per_env_priority / float(buffer.num_steps), buffer.num_steps))
        batch = trainer._ensure_minibatch(buffer, segment_count=32)
        seed_counter = make_seed_counter(17, device=device)

        trainer._prepare_trajectory_priority_weights(buffer)
        with wp.ScopedCapture(device=device) as capture:
            use_priority = trainer._prepare_trajectory_priority_weights(buffer)
            trainer._sample_minibatch_env_ids_seed_counter(
                buffer, batch, seed_counter=seed_counter, seed_offset=0, use_priority=use_priority
            )
            advance_seed_counter(seed_counter, 1, device=device)

        wp.capture_launch(capture.graph)
        first = trainer._minibatch_env_ids.numpy().copy()
        wp.capture_launch(capture.graph)
        second = trainer._minibatch_env_ids.numpy().copy()

        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(seed_counter.numpy(), np.array([19], dtype=np.int32))

    def test_priority_replay_sampling_is_graph_capturable(self) -> None:
        device = _rl_cuda_device()
        config = rl.ConfigPPO(
            minibatch_size=128,
            replay_ratio=1.0,
            priority_alpha=0.5,
            priority_beta=0.75,
            normalize_advantages=False,
        )
        trainer = rl.TrainerPPO(obs_dim=3, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=101)
        buffer = rl.BufferRollout(num_steps=4, num_envs=5, obs_dim=3, action_dim=2, device=device)
        per_env_priority = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
        buffer.advantages.assign(np.tile(per_env_priority / float(buffer.num_steps), buffer.num_steps))
        batch = trainer._ensure_minibatch(buffer, segment_count=32)

        trainer._prepare_trajectory_priority_weights(buffer)
        trainer._sample_minibatch_env_ids(buffer, batch, seed=17, use_priority=True)
        with wp.ScopedCapture(device=device) as capture:
            use_priority = trainer._prepare_trajectory_priority_weights(buffer)
            trainer._sample_minibatch_env_ids(buffer, batch, seed=19, use_priority=use_priority)
        wp.capture_launch(capture.graph)

        env_ids = trainer._minibatch_env_ids.numpy()
        importance = batch.priority_weights.numpy()
        weights = np.power(np.maximum(per_env_priority, 0.0) + 1.0e-6, config.priority_alpha)
        total = float(np.sum(weights))
        expected = np.power(
            np.maximum(float(buffer.num_envs) * weights[env_ids] / total, 1.0e-6), -config.priority_beta
        )

        self.assertTrue(np.all(env_ids >= 0))
        self.assertTrue(np.all(env_ids < buffer.num_envs))
        np.testing.assert_allclose(importance, expected, rtol=2.0e-6, atol=2.0e-6)
        self.assertGreater(float(np.max(importance) - np.min(importance)), 0.0)

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
            manual_critic_backward=True,
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

    def test_minibatch_update_reserve_graph_captures_without_readback(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(29)
        config = rl.ConfigPPO(
            train_epochs=1,
            minibatch_size=4,
            replay_ratio=1.0,
            priority_alpha=0.4,
            priority_beta=0.5,
            normalize_advantages=True,
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            entropy_coeff=0.0,
            max_grad_norm=0.3,
            manual_actor_backward=True,
            manual_critic_backward=True,
            shared_value_network=True,
        )
        trainer = rl.TrainerPPO(obs_dim=4, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=17)
        buffer = rl.BufferRollout(num_steps=2, num_envs=4, obs_dim=4, action_dim=2, device=device)
        n = buffer.num_samples
        buffer.obs.assign(rng.normal(size=(n, 4)).astype(np.float32))
        buffer.actions.assign(np.tanh(0.4 * rng.normal(size=(n, 2))).astype(np.float32))
        buffer.advantages.assign(rng.normal(loc=0.2, scale=0.5, size=n).astype(np.float32))
        buffer.returns.assign(rng.normal(size=n).astype(np.float32))
        _policy_out, old_log_probs = trainer.actor.log_prob(buffer.obs, buffer.actions, requires_grad=False)
        buffer.old_log_probs.assign(old_log_probs.numpy())

        trainer.reserve_update_buffers(buffer)
        before = trainer.actor.net.weights[0].numpy().copy()
        with wp.ScopedCapture(device=device) as capture:
            stats = trainer.update(buffer, read_stats=False)
        wp.capture_launch(capture.graph)
        after = trainer.actor.net.weights[0].numpy()

        self.assertEqual(stats.policy_loss, 0.0)
        self.assertEqual(stats.value_loss, 0.0)
        self.assertEqual(stats.approx_kl, 0.0)
        self.assertEqual(stats.clip_fraction, 0.0)
        self.assertGreater(float(np.max(np.abs(after - before))), 0.0)

    def test_shared_value_network_update_graph_capture_and_checkpoint(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(11)
        config = rl.ConfigPPO(
            train_epochs=1,
            normalize_advantages=False,
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            entropy_coeff=0.0,
            max_grad_norm=0.3,
            mirror_loss_coeff=0.1,
            manual_actor_backward=True,
            manual_critic_backward=True,
            manual_mlp_weight_grad_dtype="bfloat16",
            manual_mlp_forward_dtype="bfloat16",
            shared_value_network=True,
        )
        mirror_map = rl.MirrorMapPPO(
            obs_src=(0, 1, 2, 3, 4),
            obs_sign=(1.0, 1.0, 1.0, 1.0, 1.0),
            action_src=(1, 0),
            action_sign=(1.0, 1.0),
        )
        trainer = rl.TrainerPPO(
            obs_dim=5, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=13, mirror_map=mirror_map
        )
        buffer = rl.BufferRollout(num_steps=4, num_envs=4, obs_dim=5, action_dim=2, device=device)
        n = buffer.num_samples
        buffer.obs.assign(rng.normal(size=(n, 5)).astype(np.float32))
        buffer.actions.assign(np.tanh(0.5 * rng.normal(size=(n, 2))).astype(np.float32))
        buffer.advantages.assign(rng.normal(loc=0.5, scale=0.25, size=n).astype(np.float32))
        buffer.returns.assign(rng.normal(size=n).astype(np.float32))
        _policy_out, old_log_probs = trainer.actor.log_prob(buffer.obs, buffer.actions, requires_grad=False)
        buffer.old_log_probs.assign(old_log_probs.numpy())

        self.assertIsNone(trainer.critic)
        self.assertEqual(trainer.value_column, 2)
        self.assertEqual(trainer.actor.net.output_dim, 3)

        trainer.reserve_update_buffers(buffer)
        actor_before = trainer.actor.net.weights[0].numpy().copy()
        stats = trainer._update_shared_manual(buffer)
        actor_after = trainer.actor.net.weights[0].numpy()
        self.assertTrue(math.isfinite(stats.policy_loss))
        self.assertTrue(math.isfinite(stats.value_loss))
        self.assertTrue(math.isfinite(stats.approx_kl))
        self.assertTrue(math.isfinite(stats.clip_fraction))
        self.assertGreater(float(np.max(np.abs(actor_after - actor_before))), 0.0)

        with wp.ScopedCapture(device=device) as stats_capture:
            trainer.copy_update_stats_to_host()
        wp.capture_launch(stats_capture.graph)
        wp.synchronize_device(device)
        np.testing.assert_allclose(
            trainer._update_stats_host.numpy(),
            np.array([stats.policy_loss, stats.value_loss, stats.approx_kl, stats.clip_fraction], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )

        graph_before = trainer.actor.net.weights[0].numpy().copy()
        with wp.ScopedCapture(device=device) as capture:
            trainer.update(buffer, read_stats=False)
        wp.capture_launch(capture.graph)
        graph_after = trainer.actor.net.weights[0].numpy()
        self.assertGreater(float(np.max(np.abs(graph_after - graph_before))), 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/shared_ppo_checkpoint.npz"
            trainer.save_checkpoint(path, iteration=7)
            restored = rl.load_ppo_checkpoint(path, device=device)

        self.assertTrue(restored.config.shared_value_network)
        self.assertIsNone(restored.critic)
        self.assertEqual(restored.actor.net.output_dim, 3)
        self.assertEqual(restored.actor_optimizer.step_count, trainer.actor_optimizer.step_count)
        self.assertEqual(restored.iteration, 7)
        for before, after in zip(trainer.actor.parameters(), restored.actor.parameters(), strict=True):
            np.testing.assert_allclose(after.numpy(), before.numpy(), rtol=0.0, atol=0.0)

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
        self.assertFalse(restored.config.manual_critic_backward)
        self.assertEqual(restored.config.manual_mlp_weight_grad_dtype, "float32")
        self.assertEqual(restored.config.manual_mlp_forward_dtype, "float32")
        self.assertEqual(restored.config.vtrace_rho_clip, 0.0)
        self.assertEqual(restored.config.vtrace_c_clip, 0.0)
        self.assertEqual(restored.config.reward_clip, 0.0)
        self.assertEqual(restored.config.max_grad_norm, 0.0)
        self.assertEqual(restored.config.mirror_loss_coeff, 0.0)
        self.assertFalse(restored.config.shared_value_network)


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
    def test_distributional_projection_preserves_mass_and_terminal_reward(self) -> None:
        device = _rl_cuda_device()
        logits = wp.zeros((2, 5), dtype=wp.float32, device=device)
        rewards = wp.array([1.0, 0.0], dtype=wp.float32, device=device)
        dones = wp.array([1.0, 0.0], dtype=wp.float32, device=device)
        log_probs = wp.zeros(2, dtype=wp.float32, device=device)
        targets1 = wp.zeros_like(logits)
        targets2 = wp.zeros_like(logits)
        values = wp.empty((2, 1), dtype=wp.float32, device=device)

        wp.launch(
            sac_distributional_projection_kernel,
            dim=(2, 5),
            inputs=[rewards, dones, logits, logits, log_probs, 1.0, 0.0, 5, -2.0, 2.0],
            outputs=[targets1, targets2],
            device=device,
        )
        wp.launch(
            sac_distributional_q_value_kernel,
            dim=2,
            inputs=[logits, 5, -2.0, 2.0],
            outputs=[values],
            device=device,
        )

        expected_terminal = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        expected_uniform = np.full(5, 0.2, dtype=np.float32)
        np.testing.assert_allclose(targets1.numpy()[0], expected_terminal, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(targets1.numpy()[1], expected_uniform, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(targets2.numpy(), targets1.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(targets1.numpy().sum(axis=1), np.ones(2), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(values.numpy(), np.zeros((2, 1)), rtol=0.0, atol=1.0e-6)

    def test_average_critics_uses_mean_for_targets_and_actor_gradients(self) -> None:
        device = _rl_cuda_device()
        q1 = wp.array([[1.0], [5.0]], dtype=wp.float32, device=device)
        q2 = wp.array([[3.0], [1.0]], dtype=wp.float32, device=device)
        rewards = wp.array([0.0, 2.0], dtype=wp.float32, device=device)
        dones = wp.array([0.0, 1.0], dtype=wp.float32, device=device)
        log_probs = wp.zeros(2, dtype=wp.float32, device=device)
        targets = wp.empty(2, dtype=wp.float32, device=device)
        q1_grad = wp.empty_like(q1)
        q2_grad = wp.empty_like(q2)

        wp.launch(
            sac_q_target_kernel,
            dim=2,
            inputs=[rewards, dones, q1, q2, log_probs, 0.5, 0.0, True],
            outputs=[targets],
            device=device,
        )
        wp.launch(
            sac_actor_q_backward_kernel,
            dim=2,
            inputs=[q1, q2, 2, True],
            outputs=[q1_grad, q2_grad],
            device=device,
        )

        np.testing.assert_allclose(targets.numpy(), np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(q1_grad.numpy(), -0.25)
        np.testing.assert_allclose(q2_grad.numpy(), -0.25)

    def test_policy_frequency_delays_actor_updates(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(19)
        trainer = rl.TrainerSAC(
            obs_dim=3,
            action_dim=1,
            hidden_layers=(8,),
            config=rl.ConfigSAC(policy_frequency=2, normalize_observations=False),
            device=device,
            seed=23,
        )
        batch = rl.BatchSAC(
            obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), dtype=wp.float32, device=device),
            actions=wp.array(np.tanh(rng.normal(size=(16, 1))).astype(np.float32), dtype=wp.float32, device=device),
            rewards=wp.array(rng.normal(size=16).astype(np.float32), dtype=wp.float32, device=device),
            dones=wp.zeros(16, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), dtype=wp.float32, device=device),
        )

        trainer.update(batch, seed=1)
        after_first = [parameter.numpy().copy() for parameter in trainer.actor.parameters()]
        trainer.update(batch, seed=2)
        after_second = [parameter.numpy().copy() for parameter in trainer.actor.parameters()]
        trainer.update(batch, seed=3)
        after_third = [parameter.numpy().copy() for parameter in trainer.actor.parameters()]

        for first, second in zip(after_first, after_second, strict=True):
            np.testing.assert_array_equal(first, second)
        self.assertTrue(any(np.any(second != third) for second, third in zip(after_second, after_third, strict=True)))

    def test_update_without_stats_matches_readback_update(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(29)
        config = rl.ConfigSAC(normalize_observations=False)
        with_stats = rl.TrainerSAC(obs_dim=4, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=31)
        without_stats = rl.TrainerSAC(
            obs_dim=4, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=31
        )
        batch = rl.BatchSAC(
            obs=wp.array(rng.normal(size=(16, 4)).astype(np.float32), dtype=wp.float32, device=device),
            actions=wp.array(np.tanh(rng.normal(size=(16, 2))).astype(np.float32), dtype=wp.float32, device=device),
            rewards=wp.array(rng.normal(size=16).astype(np.float32), dtype=wp.float32, device=device),
            dones=wp.zeros(16, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(16, 4)).astype(np.float32), dtype=wp.float32, device=device),
        )

        stats = with_stats.update(batch, seed=37)
        skipped = without_stats.update(batch, seed=37, read_stats=False)

        self.assertTrue(all(math.isfinite(value) for value in stats.__dict__.values()))
        self.assertEqual(tuple(skipped.__dict__.values()), (0.0, 0.0, 0.0, 0.0))
        networks_with = (
            with_stats.actor,
            with_stats.critic1,
            with_stats.critic2,
            with_stats.target_critic1,
            with_stats.target_critic2,
        )
        networks_without = (
            without_stats.actor,
            without_stats.critic1,
            without_stats.critic2,
            without_stats.target_critic1,
            without_stats.target_critic2,
        )
        for left, right in zip(networks_with, networks_without, strict=True):
            for left_param, right_param in zip(left.parameters(), right.parameters(), strict=True):
                np.testing.assert_array_equal(left_param.numpy(), right_param.numpy())
        np.testing.assert_array_equal(with_stats.log_alpha.numpy(), without_stats.log_alpha.numpy())
        np.testing.assert_array_equal(with_stats._alpha.numpy(), without_stats._alpha.numpy())

    def test_observation_normalization_tracks_batch_moments(self) -> None:
        device = _rl_cuda_device()
        rng = np.random.default_rng(7)
        obs_np = np.column_stack(
            (
                rng.normal(2.0, 3.0, 8192),
                rng.normal(-3.0, 0.5, 8192),
            )
        ).astype(np.float32)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        zeros = wp.zeros(8192, dtype=wp.float32, device=device)
        trainer = rl.TrainerSAC(obs_dim=2, action_dim=1, hidden_layers=(8,), device=device)
        batch = rl.BatchSAC(
            obs=obs,
            actions=wp.zeros((8192, 1), dtype=wp.float32, device=device),
            rewards=zeros,
            dones=zeros,
            next_obs=obs,
        )

        normalized = trainer._normalize_batch(batch).obs.numpy()

        np.testing.assert_allclose(trainer._obs_mean.numpy(), obs_np.mean(axis=0), rtol=0.0, atol=2.0e-3)
        np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(2), rtol=0.0, atol=2.0e-3)
        np.testing.assert_allclose(normalized.std(axis=0), np.ones(2), rtol=0.0, atol=6.0e-3)

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
        actor_first_layer_delta = float(np.max(np.abs(trainer.actor.net.weights[0].numpy() - actor_before[0])))
        critic_first_layer_delta = float(np.max(np.abs(trainer.critic1.weights[0].numpy() - critic_before[0])))
        self.assertGreater(actor_delta, 0.0)
        self.assertGreater(critic_delta, 0.0)
        self.assertGreater(target_delta, 0.0)
        self.assertGreater(actor_first_layer_delta, 0.0)
        self.assertGreater(critic_first_layer_delta, 0.0)

    def test_distributional_sac_learns_known_continuous_control_optimum(self) -> None:
        device = _rl_cuda_device()
        seed = 3
        rng = np.random.default_rng(seed)
        world_count = 1024
        trainer = rl.TrainerSAC(
            obs_dim=2,
            action_dim=1,
            hidden_layers=(64, 64),
            config=rl.ConfigSAC(
                gamma=0.0,
                initial_alpha=0.01,
                target_entropy=0.0,
                critic_lr=1.0e-3,
                average_critics=True,
                distributional_critic=True,
            ),
            device=device,
            seed=seed,
        )
        replay = rl.BufferReplaySAC(capacity=65536, obs_dim=2, action_dim=1, batch_size=2048, device=device)
        eval_obs_np = rng.uniform(-1.0, 1.0, (2048, 2)).astype(np.float32)
        eval_target = np.tanh(1.2 * eval_obs_np[:, 0] - 0.7 * eval_obs_np[:, 1])
        eval_obs = wp.array(eval_obs_np, dtype=wp.float32, device=device)
        initial_actions = trainer.act(eval_obs, seed=0, deterministic=True)[0].numpy()[:, 0]
        initial_mse = float(np.mean((initial_actions - eval_target) ** 2))

        for update in range(100):
            obs_np = rng.uniform(-1.0, 1.0, (world_count, 2)).astype(np.float32)
            obs = wp.array(obs_np, dtype=wp.float32, device=device)
            if update < 4:
                actions_np = rng.uniform(-1.0, 1.0, (world_count, 1)).astype(np.float32)
            else:
                actions_np = trainer.act(obs, seed=seed * 1000 + update)[0].numpy()
            target = np.tanh(1.2 * obs_np[:, 0] - 0.7 * obs_np[:, 1])
            rewards_np = -((actions_np[:, 0] - target) ** 2)
            replay.add_batch(
                obs,
                wp.array(actions_np, dtype=wp.float32, device=device),
                wp.array(rewards_np, dtype=wp.float32, device=device),
                wp.ones(world_count, dtype=wp.float32, device=device),
                obs,
            )
            trainer.update(replay.sample(seed=update), seed=10000 + update)

        learned_actions = trainer.act(eval_obs, seed=0, deterministic=True)[0].numpy()[:, 0]
        learned_mse = float(np.mean((learned_actions - eval_target) ** 2))
        self.assertGreater(initial_mse, 0.2)
        self.assertLess(learned_mse, 0.05)

    def test_learns_known_continuous_control_optimum(self) -> None:
        device = _rl_cuda_device()
        seed = 11
        rng = np.random.default_rng(seed)
        world_count = 1024
        trainer = rl.TrainerSAC(
            obs_dim=2,
            action_dim=1,
            hidden_layers=(64, 64),
            config=rl.ConfigSAC(gamma=0.0, initial_alpha=0.01, target_entropy=0.0, critic_lr=1.0e-3),
            device=device,
            seed=seed,
        )
        replay = rl.BufferReplaySAC(capacity=65536, obs_dim=2, action_dim=1, batch_size=2048, device=device)
        eval_obs_np = rng.uniform(-1.0, 1.0, (2048, 2)).astype(np.float32)
        eval_target = np.tanh(1.2 * eval_obs_np[:, 0] - 0.7 * eval_obs_np[:, 1])
        eval_obs = wp.array(eval_obs_np, dtype=wp.float32, device=device)
        initial_actions = trainer.act(eval_obs, seed=0, deterministic=True)[0].numpy()[:, 0]
        initial_mse = float(np.mean((initial_actions - eval_target) ** 2))

        for update in range(100):
            obs_np = rng.uniform(-1.0, 1.0, (world_count, 2)).astype(np.float32)
            obs = wp.array(obs_np, dtype=wp.float32, device=device)
            if update < 4:
                actions_np = rng.uniform(-1.0, 1.0, (world_count, 1)).astype(np.float32)
            else:
                actions_np = trainer.act(obs, seed=seed * 1000 + update)[0].numpy()
            target = np.tanh(1.2 * obs_np[:, 0] - 0.7 * obs_np[:, 1])
            rewards_np = -((actions_np[:, 0] - target) ** 2)
            replay.add_batch(
                obs,
                wp.array(actions_np, dtype=wp.float32, device=device),
                wp.array(rewards_np, dtype=wp.float32, device=device),
                wp.ones(world_count, dtype=wp.float32, device=device),
                obs,
            )
            trainer.update(replay.sample(seed=update), seed=10000 + update)

        learned_actions = trainer.act(eval_obs, seed=0, deterministic=True)[0].numpy()[:, 0]
        learned_mse = float(np.mean((learned_actions - eval_target) ** 2))
        self.assertGreater(initial_mse, 0.2)
        self.assertLess(learned_mse, 0.02)


if __name__ == "__main__":
    wp.init()
    unittest.main()
