# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training.env import collect_ppo_rollout_seed_counter, make_seed_counter
from newton._src.solvers.phoenx.rl_training.kernels import (
    PPO_LOG_STD_PARTIAL_BATCH,
    TANH_EPS,
    gaussian_log_prob_kernel,
    ppo_actor_loss_backward_kernel,
    reduce_ppo_log_std_grad_kernel,
    sample_gaussian_actions_kernel,
    zero_ppo_actor_stats_kernel,
)
from newton._src.solvers.phoenx.rl_training.ppo import BufferRollout, ConfigPPO, TrainerPPO
from newton._src.solvers.phoenx.rl_training.standard_envs import (
    ConfigEnvPendulumV1Warp,
    EnvPendulumV1Warp,
    pendulum_v1_numpy_step,
)
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


def _squashed_gaussian_log_prob_numpy(
    actions: np.ndarray,
    mean: np.ndarray,
    log_std: np.ndarray,
) -> np.ndarray:
    pre_tanh = np.arctanh(np.clip(actions, -1.0 + TANH_EPS, 1.0 - TANH_EPS)).astype(np.float32)
    std = np.exp(log_std).astype(np.float32)
    normal_log_prob = -0.5 * ((pre_tanh - mean) / std) ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std
    squash_correction = np.log(1.0 - actions * actions + TANH_EPS)
    return np.sum(normal_log_prob - squash_correction, axis=1, dtype=np.float32).astype(np.float32)


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

    def test_squashed_gaussian_log_prob_matches_numpy_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX squashed Gaussian log-prob tests")
        rows = 3
        action_dim = 2
        policy_out_np = np.asarray([[0.4, -0.25], [0.1, 0.5], [-0.6, 0.2]], dtype=np.float32)
        log_std_np = np.asarray([-0.5, 0.25], dtype=np.float32)
        actions_np = np.asarray([[0.0, 0.25], [-0.4, 0.75], [0.85, -0.65]], dtype=np.float32)
        eps_np = np.zeros_like(actions_np)

        policy_out = wp.array(policy_out_np, dtype=wp.float32, device=device)
        log_std = wp.array(log_std_np, dtype=wp.float32, device=device)
        actions = wp.array(actions_np, dtype=wp.float32, device=device)
        eps = wp.array(eps_np, dtype=wp.float32, device=device)
        log_probs = wp.zeros(rows, dtype=wp.float32, device=device)
        sampled_actions = wp.zeros_like(actions)
        sampled_log_probs = wp.zeros(rows, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                gaussian_log_prob_kernel,
                dim=rows,
                inputs=[policy_out, log_std, actions, action_dim, 0, 1, -20.0, 2.0],
                outputs=[log_probs],
                device=device,
            )
            wp.launch(
                sample_gaussian_actions_kernel,
                dim=rows,
                inputs=[policy_out, log_std, eps, action_dim, 0, 1, 1, -20.0, 2.0],
                outputs=[sampled_actions, sampled_log_probs],
                device=device,
            )
        wp.capture_launch(capture.graph)

        expected_log_probs = _squashed_gaussian_log_prob_numpy(actions_np, policy_out_np, log_std_np)
        expected_sampled_actions = np.tanh(policy_out_np).astype(np.float32)
        expected_sampled_log_probs = _squashed_gaussian_log_prob_numpy(
            expected_sampled_actions, policy_out_np, log_std_np
        )
        np.testing.assert_allclose(log_probs.numpy(), expected_log_probs, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(sampled_actions.numpy(), expected_sampled_actions, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(sampled_log_probs.numpy(), expected_sampled_log_probs, rtol=1.0e-6, atol=1.0e-6)

    def test_squashed_ppo_actor_loss_backward_matches_numpy_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX squashed PPO actor loss tests")
        rows = 4
        action_dim = 2
        clip_ratio = np.float32(0.2)
        entropy_coeff = np.float32(0.03)
        policy_out_np = np.asarray([[0.2, -0.4], [0.1, 0.3], [-0.25, 0.35], [0.45, -0.15]], dtype=np.float32)
        log_std_np = np.asarray([-0.3, 0.25], dtype=np.float32)
        actions_np = np.asarray([[0.35, -0.2], [-0.15, 0.65], [0.4, -0.05], [0.2, -0.55]], dtype=np.float32)
        advantages_np = np.asarray([0.5, -0.25, 0.8, -0.4], dtype=np.float32)
        desired_ratios = np.asarray([1.0, 1.35, 1.35, 0.72], dtype=np.float32)
        new_log_probs_np = _squashed_gaussian_log_prob_numpy(actions_np, policy_out_np, log_std_np)
        old_log_probs_np = (new_log_probs_np - np.log(desired_ratios).astype(np.float32)).astype(np.float32)

        policy_out = wp.array(policy_out_np, dtype=wp.float32, device=device)
        log_std = wp.array(log_std_np, dtype=wp.float32, device=device)
        actions = wp.array(actions_np, dtype=wp.float32, device=device)
        old_log_probs = wp.array(old_log_probs_np, dtype=wp.float32, device=device)
        advantages = wp.array(advantages_np, dtype=wp.float32, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device)
        approx_kl = wp.zeros(1, dtype=wp.float32, device=device)
        clip_fraction = wp.zeros(1, dtype=wp.float32, device=device)
        ratios = wp.zeros(rows, dtype=wp.float32, device=device)
        policy_out_grad = wp.zeros_like(policy_out)
        partial_count = (rows + PPO_LOG_STD_PARTIAL_BATCH - 1) // PPO_LOG_STD_PARTIAL_BATCH
        log_std_grad_partials = wp.zeros((partial_count, action_dim), dtype=wp.float32, device=device)
        log_std_grad = wp.zeros(action_dim, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                zero_ppo_actor_stats_kernel,
                dim=max(partial_count * action_dim, 1),
                inputs=[partial_count, action_dim],
                outputs=[loss, approx_kl, clip_fraction, log_std_grad_partials],
                device=device,
            )
            wp.launch(
                ppo_actor_loss_backward_kernel,
                dim=rows,
                inputs=[
                    policy_out,
                    log_std,
                    actions,
                    old_log_probs,
                    advantages,
                    float(clip_ratio),
                    float(entropy_coeff),
                    action_dim,
                    0,
                    1,
                    -20.0,
                    2.0,
                    rows,
                ],
                outputs=[loss, approx_kl, clip_fraction, ratios, policy_out_grad, log_std_grad_partials],
                device=device,
            )
            wp.launch(
                reduce_ppo_log_std_grad_kernel,
                dim=action_dim,
                inputs=[log_std_grad_partials, partial_count],
                outputs=[log_std_grad],
                device=device,
            )
        wp.capture_launch(capture.graph)

        pre_tanh_actions_np = np.arctanh(np.clip(actions_np, -1.0 + TANH_EPS, 1.0 - TANH_EPS)).astype(np.float32)
        std = np.exp(log_std_np).astype(np.float32)
        var = std * std
        expected_loss = np.float32(0.0)
        expected_approx_kl = np.float32(0.0)
        expected_clip_fraction = np.float32(0.0)
        expected_policy_grad = np.zeros_like(policy_out_np)
        expected_log_std_grad = np.zeros(action_dim, dtype=np.float32)
        entropy = np.sum(np.float32(0.5 * math.log(2.0 * math.pi * math.e)) + log_std_np, dtype=np.float32)
        inv_batch = np.float32(1.0 / rows)
        expected_ratios = np.zeros(rows, dtype=np.float32)
        for row in range(rows):
            log_ratio = np.float32(new_log_probs_np[row] - old_log_probs_np[row])
            ratio = np.float32(math.exp(float(log_ratio)))
            expected_ratios[row] = ratio
            clipped = np.clip(ratio, np.float32(1.0) - clip_ratio, np.float32(1.0) + clip_ratio)
            adv = advantages_np[row]
            pg_loss_unclipped = np.float32(-adv * ratio)
            pg_loss_clipped = np.float32(-adv * clipped)
            pg_loss = max(pg_loss_unclipped, pg_loss_clipped)
            expected_loss += np.float32((pg_loss - entropy_coeff * entropy) * inv_batch)
            expected_approx_kl += np.float32(((ratio - np.float32(1.0)) - log_ratio) * inv_batch)
            if abs(ratio - np.float32(1.0)) > clip_ratio:
                expected_clip_fraction += inv_batch
            d_log_prob = np.float32(-adv * ratio * inv_batch)
            clipped_branch = pg_loss_clipped > pg_loss_unclipped
            outside_clip = ratio <= np.float32(1.0) - clip_ratio or ratio >= np.float32(1.0) + clip_ratio
            if clipped_branch and outside_clip:
                d_log_prob = np.float32(0.0)
            diff = pre_tanh_actions_np[row] - policy_out_np[row]
            expected_policy_grad[row] = d_log_prob * diff / var
            expected_log_std_grad += d_log_prob * (diff * diff / var - np.float32(1.0)) - entropy_coeff * inv_batch

        np.testing.assert_allclose(loss.numpy()[0], expected_loss, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(approx_kl.numpy()[0], expected_approx_kl, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(clip_fraction.numpy()[0], expected_clip_fraction, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(ratios.numpy(), expected_ratios, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(policy_out_grad.numpy(), expected_policy_grad, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(log_std_grad.numpy(), expected_log_std_grad, rtol=1.0e-6, atol=1.0e-6)

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
