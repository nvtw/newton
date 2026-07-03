# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestHumanoidPhoenXRL(unittest.TestCase):
    def test_direct_task_step_inside_cuda_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Humanoid RL tests")
        env = rl.EnvHumanoidPhoenX(
            rl.ConfigEnvHumanoidPhoenX(
                world_count=2,
                sim_substeps=2,
                solver_iterations=4,
                velocity_iterations=1,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=device,
        )
        action_row = np.linspace(-0.2, 0.2, env.action_dim, dtype=np.float32)
        actions = wp.array(np.tile(action_row, (env.world_count, 1)), dtype=wp.float32, device=device)

        env.step(actions)
        env.reset()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(3):
                env.step(actions)
        wp.capture_launch(capture.graph)

        obs = env.obs.numpy()
        rewards = env.step_rewards.numpy()
        dones = env.step_dones.numpy()
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        joint_f = env.control.joint_f.numpy().reshape(env.world_count, env.dof_stride)
        expected_f = action_row * env.joint_gears.numpy()

        self.assertEqual(obs.shape, (env.world_count, rl.OBS_DIM_HUMANOID))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(dones, 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(joint_f[:, 6:27], np.tile(expected_f, (env.world_count, 1)), rtol=0.0, atol=1.0e-5)
        self.assertGreater(float(np.min(q[:, 2])), 1.0)

    def test_initial_reward_matches_direct_task_terms(self) -> None:
        device = require_cuda_graph_capture("PhoenX Humanoid RL tests")
        env = rl.EnvHumanoidPhoenX(
            rl.ConfigEnvHumanoidPhoenX(world_count=2, max_episode_steps=0, auto_reset=False),
            device=device,
        )
        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        rewards = env.rewards.numpy()
        np.testing.assert_allclose(rewards, 2.6, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.obs.numpy()[:, 0], 1.34, rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
