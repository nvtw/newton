# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import tempfile
import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


def _g1_test_env(world_count: int = 1) -> rl.EnvG1PhoenX:
    device = require_cuda_graph_capture("PhoenX G1 RL tests")
    config = rl.ConfigEnvG1PhoenX(
        world_count=int(world_count),
        sim_substeps=5,
        solver_iterations=2,
        velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
        max_episode_steps=0,
        auto_reset=False,
    )
    return rl.EnvG1PhoenX(config, device=device)


class TestG1PhoenXRL(unittest.TestCase):
    def test_default_recipe_enables_mirror_and_matches_config_defaults(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL recipe tests")

        env_config = g1_recipe.default_g1_env_config(world_count=1)
        train_config = rl.ConfigTrainG1PPO()
        ppo_config = g1_recipe.default_g1_ppo_config()

        self.assertEqual(env_config.sim_substeps, g1_recipe.SIM_SUBSTEPS)
        self.assertEqual(rl.EnvG1PhoenX(env_config, device=device).solver.world.substeps, 1)
        self.assertEqual(env_config.solver_iterations, g1_recipe.SOLVER_ITERATIONS)
        self.assertEqual(g1_recipe.VELOCITY_ITERATIONS, 1)
        self.assertEqual(env_config.velocity_iterations, g1_recipe.VELOCITY_ITERATIONS)
        self.assertEqual(env_config.w_track_lin, g1_recipe.W_TRACK_LIN)
        self.assertEqual(env_config.w_action_rate, g1_recipe.W_ACTION_RATE)
        self.assertEqual(env_config.rigid_contact_max_per_world, g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env_config.threads_per_world, g1_recipe.THREADS_PER_WORLD)
        self.assertEqual(env_config.multi_world_scheduler, g1_recipe.MULTI_WORLD_SCHEDULER)
        self.assertEqual(env_config.prepare_refresh_stride, g1_recipe.PREPARE_REFRESH_STRIDE)
        self.assertEqual(train_config.hidden_layers, g1_recipe.HIDDEN_LAYERS)
        self.assertEqual(train_config.activation, g1_recipe.ACTIVATION)
        self.assertEqual(train_config.rollout_steps, g1_recipe.ROLLOUT_STEPS)
        self.assertGreater(ppo_config.mirror_loss_coeff, 0.0)
        self.assertEqual(ppo_config.mirror_loss_coeff, g1_recipe.MIRROR_LOSS_COEFF)
        self.assertTrue(ppo_config.shared_value_network)
        self.assertEqual(ppo_config.shared_value_network, g1_recipe.SHARED_VALUE_NETWORK)
        self.assertEqual(ppo_config.minibatch_size, g1_recipe.MINIBATCH_SIZE)
        self.assertEqual(ppo_config.manual_mlp_forward_dtype, g1_recipe.MANUAL_MLP_FORWARD_DTYPE)

    def test_rejects_negative_contact_capacity(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL capacity tests")
        config = rl.ConfigEnvG1PhoenX(world_count=1, rigid_contact_max_per_world=-1)

        with self.assertRaisesRegex(ValueError, "rigid_contact_max_per_world"):
            rl.EnvG1PhoenX(config, device=device)

    def test_step_graph_capture_shapes_and_masks_actions(self) -> None:
        env = _g1_test_env(world_count=2)
        actions_np = np.full((env.world_count, env.action_dim), 1.5, dtype=np.float32)
        actions = wp.array(actions_np, dtype=wp.float32, device=env.device)

        graph = rl.capture_env_steps(env, actions, steps_per_graph=2, warmup_steps=1)
        wp.capture_launch(graph)
        wp.synchronize_device(env.device)

        self.assertFalse(env.solver.world.articulation_dvi_host)
        self.assertFalse(env.solver.world.articulation_dvi_replaces_joint_pgs)
        self.assertIsNone(env.solver.world.articulation_topology)
        self.assertEqual(env.solver.world.rigid_contact_max, env.world_count * g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env.contacts.rigid_contact_max, env.world_count * g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env.obs.shape, (2, rl.OBS_DIM_G1))
        self.assertEqual(env.rewards.shape, (2,))
        self.assertEqual(env.dones.shape, (2,))
        self.assertTrue(np.isfinite(env.obs.numpy()).all())
        self.assertTrue(np.isfinite(env.rewards.numpy()).all())
        self.assertTrue(np.isfinite(env.dones.numpy()).all())

        current_actions = env.current_actions.numpy()
        np.testing.assert_allclose(current_actions[:, :12], 1.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(current_actions[:, 12:], 0.0, rtol=0.0, atol=0.0)

    def test_graph_replay_advances_policy_steps(self) -> None:
        env = _g1_test_env(world_count=1)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)

        graph = rl.capture_env_steps(env, actions, steps_per_graph=2, warmup_steps=1)
        before = int(env.episode_steps.numpy()[0])
        for _ in range(3):
            wp.capture_launch(graph)
        wp.synchronize_device(env.device)
        after = int(env.episode_steps.numpy()[0])

        self.assertEqual(after - before, 6)
        self.assertTrue(np.isfinite(env.obs.numpy()).all())

    def test_randomize_commands_graph_capture(self) -> None:
        env = _g1_test_env(world_count=4)
        command_x_range = (-0.4, 0.8)
        command_y_range = (-0.2, 0.3)
        command_yaw_range = (-0.7, 0.6)

        with wp.ScopedCapture(device=env.device) as capture:
            env.randomize_commands(
                seed=17,
                command_x_range=command_x_range,
                command_y_range=command_y_range,
                command_yaw_range=command_yaw_range,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize_device(env.device)

        commands = env.command.numpy()
        self.assertTrue(np.all(commands[:, 0] >= command_x_range[0]))
        self.assertTrue(np.all(commands[:, 0] <= command_x_range[1]))
        self.assertTrue(np.all(commands[:, 1] >= command_y_range[0]))
        self.assertTrue(np.all(commands[:, 1] <= command_y_range[1]))
        self.assertTrue(np.all(commands[:, 2] >= command_yaw_range[0]))
        self.assertTrue(np.all(commands[:, 2] <= command_yaw_range[1]))

    def test_observe_clamps_extreme_state_to_finite_metrics(self) -> None:
        env = _g1_test_env(world_count=1)
        huge_qd = np.full(env.state_0.joint_qd.shape, 1.0e20, dtype=np.float32)
        env.state_0.joint_qd.assign(huge_qd)

        obs = env.observe()

        self.assertTrue(np.isfinite(obs.numpy()).all())
        self.assertTrue(np.isfinite(env.rewards.numpy()).all())
        self.assertTrue(np.isfinite(env.dones.numpy()).all())

    def test_train_save_load_evaluate_and_resume(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL training tests")
        env_config = rl.ConfigEnvG1PhoenX(
            world_count=1,
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            max_episode_steps=0,
            auto_reset=False,
        )
        ppo_config = rl.ConfigPPO(
            train_epochs=1,
            normalize_advantages=False,
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            entropy_coeff=0.0,
            reward_clip=1.0,
            max_grad_norm=0.3,
            mirror_loss_coeff=0.25,
            shared_value_network=True,
            manual_actor_backward=True,
            manual_critic_backward=True,
            manual_mlp_weight_grad_dtype="bfloat16",
            manual_mlp_forward_dtype="bfloat16",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_template = f"{tmpdir}/g1_{{iteration}}.npz"
            train_config = rl.ConfigTrainG1PPO(
                iterations=1,
                rollout_steps=1,
                hidden_layers=(8,),
                env_config=env_config,
                ppo_config=ppo_config,
                device=device,
                seed=23,
                log_interval=0,
                randomize_commands=True,
                checkpoint_path=checkpoint_template,
                checkpoint_interval=1,
                readback_diagnostics=True,
            )
            first = rl.train_g1_ppo(train_config)
            first_path = f"{tmpdir}/g1_1.npz"
            restored = rl.load_ppo_checkpoint(first_path, device=device)

            self.assertEqual(first.history[0].iteration, 0)
            self.assertTrue(math.isfinite(first.history[0].policy_loss))
            self.assertTrue(math.isfinite(first.history[0].value_loss))
            self.assertTrue(math.isfinite(first.history[0].mean_reward))
            self.assertTrue(math.isfinite(first.history[0].mean_tracking_perf))
            self.assertTrue(math.isfinite(first.history[0].mean_command_x))
            self.assertTrue(math.isfinite(first.history[0].mean_command_y))
            self.assertTrue(math.isfinite(first.history[0].mean_command_yaw))
            self.assertEqual(first.trainer.iteration, 1)
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.config.minibatch_size, 0)
            self.assertEqual(restored.config.replay_ratio, 0.0)
            self.assertEqual(restored.config.priority_alpha, 0.0)
            self.assertEqual(restored.config.priority_beta, 0.0)
            self.assertTrue(restored.config.manual_actor_backward)
            self.assertTrue(restored.config.manual_critic_backward)
            self.assertEqual(restored.config.manual_mlp_weight_grad_dtype, "bfloat16")
            self.assertEqual(restored.config.manual_mlp_forward_dtype, "bfloat16")
            self.assertEqual(restored.config.vtrace_rho_clip, 0.0)
            self.assertEqual(restored.config.vtrace_c_clip, 0.0)
            self.assertEqual(restored.config.reward_clip, 1.0)
            self.assertEqual(restored.config.max_grad_norm, 0.3)
            self.assertEqual(restored.config.mirror_loss_coeff, 0.25)
            self.assertTrue(restored.config.shared_value_network)
            self.assertIsNone(restored.critic)
            for before, after in zip(first.trainer.actor.parameters(), restored.actor.parameters(), strict=True):
                np.testing.assert_allclose(after.numpy(), before.numpy(), rtol=0.0, atol=0.0)

            eval_result = rl.evaluate_g1_ppo(
                restored,
                rl.ConfigEvaluateG1PPO(env_config=env_config, steps=1, device=device, deterministic=True),
            )
            self.assertTrue(math.isfinite(eval_result.stats.mean_reward))
            self.assertTrue(math.isfinite(eval_result.stats.mean_done))
            self.assertGreater(eval_result.stats.samples_per_second, 0.0)

            gate_result = rl.evaluate_g1_gate_ppo(
                restored,
                rl.ConfigEvaluateG1GatePPO(
                    env_config=env_config,
                    battery_commands=((0.0, 0.0, 0.0),),
                    seeds_per_command=1,
                    battery_steps=1,
                    diagnostic_steps=1,
                    diagnostic_world_count=1,
                    device=device,
                    deterministic=True,
                    min_battery_perf=2.0,
                ),
            )
            gate_stats = gate_result.stats
            self.assertEqual(gate_stats.battery_samples, 1)
            self.assertEqual(len(gate_stats.per_command), 1)
            self.assertFalse(gate_stats.pass_gate)
            self.assertTrue(math.isfinite(gate_stats.battery_perf))
            self.assertTrue(math.isfinite(gate_stats.action_jerk_rms))
            self.assertTrue(math.isfinite(gate_stats.ang_vel_xy_rms))
            self.assertTrue(math.isfinite(gate_stats.yaw_rate_rms))
            self.assertTrue(math.isfinite(gate_stats.leg_qvel_rms))
            self.assertGreater(gate_stats.samples_per_second, 0.0)

            resumed = rl.train_g1_ppo(
                rl.ConfigTrainG1PPO(
                    iterations=1,
                    rollout_steps=1,
                    hidden_layers=(8,),
                    env_config=env_config,
                    ppo_config=ppo_config,
                    device=device,
                    seed=23,
                    log_interval=0,
                    randomize_commands=False,
                    resume_checkpoint=first_path,
                    checkpoint_path=checkpoint_template,
                    checkpoint_interval=1,
                    readback_diagnostics=False,
                )
            )
            second_path = f"{tmpdir}/g1_2.npz"
            second_restored = rl.load_ppo_checkpoint(second_path, device=device)

            self.assertEqual(resumed.history[0].iteration, 1)
            self.assertEqual(resumed.trainer.iteration, 2)
            self.assertEqual(second_restored.iteration, 2)
            self.assertTrue(second_restored.config.shared_value_network)
            self.assertIsNone(second_restored.critic)


if __name__ == "__main__":
    wp.init()
    unittest.main()
