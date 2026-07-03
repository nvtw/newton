# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestH1PhoenXRL(unittest.TestCase):
    def test_step_and_position_targets_inside_cuda_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX H1 RL tests")
        config = rl.ConfigEnvH1PhoenX(
            world_count=2,
            sim_substeps=2,
            solver_iterations=4,
            velocity_iterations=1,
            max_episode_steps=0,
            auto_reset=False,
        )
        env = rl.EnvH1PhoenX(config, device=device)
        action_row = np.linspace(-0.1, 0.1, env.action_dim, dtype=np.float32)
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
        targets = env.control.joint_target_q.numpy().reshape(env.world_count, -1)
        expected_targets = env.default_joint_pos.numpy() + config.action_scale * action_row

        self.assertEqual(obs.shape, (env.world_count, rl.OBS_DIM_H1))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(dones, 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            targets[:, 6:25], np.tile(expected_targets, (env.world_count, 1)), rtol=0.0, atol=1.0e-6
        )
        self.assertGreater(float(np.min(q[:, 2])), 0.8)

    def test_initial_observation_matches_nominal_pose(self) -> None:
        device = require_cuda_graph_capture("PhoenX H1 RL tests")
        env = rl.EnvH1PhoenX(
            rl.ConfigEnvH1PhoenX(world_count=2, max_episode_steps=0, auto_reset=False),
            device=device,
        )
        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        obs = env.obs.numpy()
        np.testing.assert_allclose(obs[:, 6:9], np.asarray([[0.0, 0.0, -1.0]] * 2), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(obs[:, 12:31], 0.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.dones.numpy(), 0.0, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
