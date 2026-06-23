# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestAnymalPhoenXRL(unittest.TestCase):
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
            config=rl.ConfigPPO(),
            device=device,
            seed=19,
            activation="elu",
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
