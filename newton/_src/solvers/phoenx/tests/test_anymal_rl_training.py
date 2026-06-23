# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.experimental import train_anymal_full_control_curriculum as anymal_curriculum
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestAnymalPhoenXRL(unittest.TestCase):
    def test_full_control_curriculum_phase_env_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        phases = anymal_curriculum.build_full_control_curriculum()
        names = [phase.name for phase in phases]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "balance_and_step_forward")
        self.assertEqual(names[-1], "robust_full_control")
        self.assertLess(names.index("curved_forward"), names.index("reverse_walk"))
        for phase in phases:
            if phase.name in {"balance_and_step_forward", "walk_forward", "fast_efficient_forward"}:
                overrides = dict(phase.env_overrides)
                self.assertTrue(phase.randomize_commands)
                self.assertEqual(overrides["forward_progress_reward_scale"], 0.0)
                self.assertEqual(phase.command_x_range[0], 0.0)
                self.assertGreater(phase.command_zero_probability, 0.0)
        reverse = next(phase for phase in phases if phase.name == "reverse_walk")
        reverse_overrides = dict(reverse.env_overrides)
        self.assertLessEqual(reverse.command_x_range[1], 0.0)
        self.assertGreaterEqual(reverse.command_x_range[0], -0.25)
        self.assertLessEqual(abs(reverse.command[0]), 0.25)
        self.assertLess(reverse_overrides["lin_vel_tracking_sigma"], 0.5)
        self.assertGreater(reverse_overrides["forward_progress_reward_scale"], 0.0)
        side = next(phase for phase in phases if phase.name == "side_step")
        side_overrides = dict(side.env_overrides)
        self.assertEqual(side.command_x_range, (0.0, 0.0))
        self.assertLess(side_overrides["lin_vel_tracking_sigma"], 0.5)
        self.assertGreater(side_overrides["forward_progress_reward_scale"], 0.0)
        robust = phases[-1]
        robust_overrides = dict(robust.env_overrides)
        self.assertIn("lin_vel_tracking_sigma", robust_overrides)
        self.assertGreater(robust_overrides["forward_progress_reward_scale"], 0.0)

        args = anymal_curriculum._make_parser().parse_args(
            [
                "--world-count",
                "2",
                "--sim-substeps",
                "1",
                "--solver-iterations",
                "1",
                "--velocity-iterations",
                "1",
                "--max-episode-steps",
                "0",
            ]
        )
        env_config = anymal_curriculum.build_phase_env_config(
            args,
            phases[-1],
            world_count=2,
            auto_reset=False,
        )
        env = rl.EnvAnymalPhoenX(env_config, device=device)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            obs, rewards, dones = env.step(actions)
        wp.capture_launch(capture.graph)

        self.assertEqual(obs.shape, (env.world_count, env.obs_dim))
        self.assertEqual(rewards.shape, (env.world_count,))
        self.assertEqual(dones.shape, (env.world_count,))
        self.assertTrue(np.all(np.isfinite(obs.numpy())))

    def test_dense_command_ppo_step_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(
                world_count=4,
                reward_mode="dense_command",
                command=(0.6, 0.0, 0.0),
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=1,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=device,
        )
        trainer = rl.TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=(16,),
            config=rl.ConfigPPO(mirror_loss_coeff=1.0e-3),
            device=device,
            seed=19,
            activation="elu",
            mirror_map=rl.anymal_mirror_map_ppo(),
        )
        trainer.reserve_buffers(8)
        warmup_actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
        env.step(warmup_actions)
        obs = env.observe()

        with wp.ScopedCapture(device=device) as capture:
            actions, log_probs, values = trainer.act_reuse(obs, seed=23, deterministic=True)
            next_obs, rewards, dones = env.step(actions)
        wp.capture_launch(capture.graph)

        self.assertEqual(actions.shape, (env.world_count, env.action_dim))
        self.assertEqual(log_probs.shape, (env.world_count,))
        self.assertEqual(values.shape, (env.world_count, 1))
        self.assertEqual(next_obs.shape, (env.world_count, env.obs_dim))
        self.assertTrue(np.all(np.isfinite(next_obs.numpy())))
        self.assertTrue(np.all(np.isfinite(rewards.numpy())))
        self.assertTrue(np.all(np.isfinite(dones.numpy())))

    def test_dense_command_success_metric_tracks_velocity_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(
                world_count=2,
                reward_mode="dense_command",
                command=(0.0, 0.0, 0.0),
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=1,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=device,
        )

        with wp.ScopedCapture(device=device) as zero_capture:
            env.observe()
        wp.capture_launch(zero_capture.graph)
        zero_command_success = env.successes.numpy()

        env.set_command((1.0, 0.0, 0.0))
        with wp.ScopedCapture(device=device) as fast_capture:
            env.observe()
        wp.capture_launch(fast_capture.graph)
        fast_command_success = env.successes.numpy()

        self.assertGreater(float(np.min(zero_command_success)), 0.95)
        self.assertLess(float(np.max(fast_command_success)), 0.05)

    def test_dense_command_tracking_sigma_sharpens_standstill_reward_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        common = {
            "world_count": 1,
            "reward_mode": "dense_command",
            "command": (-0.3, 0.0, 0.0),
            "sim_substeps": 1,
            "solver_iterations": 1,
            "velocity_iterations": 1,
            "max_episode_steps": 0,
            "auto_reset": False,
            "lin_vel_reward_scale": 1.0,
            "yaw_rate_reward_scale": 0.0,
            "z_vel_reward_scale": 0.0,
            "ang_vel_reward_scale": 0.0,
            "action_rate_reward_scale": 0.0,
            "joint_speed_reward_scale": 0.0,
            "flat_orientation_reward_scale": 0.0,
            "forward_progress_reward_scale": 0.0,
            "fall_reward_scale": 0.0,
            "energy_reward_scale": 0.0,
        }
        default_env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(**common, lin_vel_tracking_sigma=0.5),
            device=device,
        )
        sharp_env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(**common, lin_vel_tracking_sigma=0.2),
            device=device,
        )

        with wp.ScopedCapture(device=device) as capture:
            default_env.observe()
            sharp_env.observe()
        wp.capture_launch(capture.graph)

        default_reward = float(default_env.rewards.numpy()[0])
        sharp_reward = float(sharp_env.rewards.numpy()[0])
        self.assertGreater(default_reward, 0.6)
        self.assertLess(sharp_reward, 0.2)
        self.assertGreater(default_reward, sharp_reward + 0.45)

    def test_dense_command_observes_per_world_steering_commands_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(
                world_count=4,
                reward_mode="dense_command",
                command=(0.0, 0.0, 0.0),
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=1,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=device,
        )
        commands = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.6, 0.0, 0.0),
                (0.0, 0.35, 0.0),
                (0.0, 0.0, 0.75),
            ),
            dtype=np.float32,
        )
        env.set_commands(commands)

        with wp.ScopedCapture(device=device) as capture:
            obs = env.observe()
        wp.capture_launch(capture.graph)

        obs_np = obs.numpy()
        np.testing.assert_allclose(obs_np[:, 9:12], commands, rtol=0.0, atol=1.0e-6)
        success = env.successes.numpy()
        self.assertGreater(float(success[0]), 0.95)
        self.assertLess(float(success[3]), 0.25)

    def test_velocity_disturbance_inside_graph_changes_root_velocity(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(
                world_count=3,
                reward_mode="dense_command",
                command=(0.6, 0.0, 0.0),
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=1,
                max_episode_steps=0,
                auto_reset=False,
                disturbance_warmup_steps=0,
                disturbance_noise_velocity_xy=0.0,
                disturbance_noise_yaw_velocity=0.0,
                disturbance_kick_probability=1.0,
                disturbance_kick_velocity_xy=0.25,
                disturbance_kick_yaw_velocity=0.10,
                disturbance_seed=7,
            ),
            device=device,
        )
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
        qd_before = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)[:, :6].copy()

        with wp.ScopedCapture(device=device) as capture:
            env.step(actions)
        wp.capture_launch(capture.graph)

        qd_after = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)[:, :6]
        delta = np.abs(qd_after - qd_before)
        self.assertGreater(float(np.max(delta[:, :2])), 1.0e-3)
        self.assertTrue(np.all(np.isfinite(qd_after)))


if __name__ == "__main__":
    unittest.main()
