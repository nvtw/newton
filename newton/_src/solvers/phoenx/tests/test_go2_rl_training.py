# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestGo2PhoenXRL(unittest.TestCase):
    def test_step_and_action_mapping_inside_cuda_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Go2 RL tests")
        config = rl.ConfigEnvGo2PhoenX(
            world_count=2,
            sim_substeps=2,
            solver_iterations=2,
            velocity_iterations=1,
            max_episode_steps=0,
            auto_reset=False,
        )
        env = rl.EnvGo2PhoenX(config, device=device)
        action_np = np.linspace(-0.4, 0.4, env.action_dim, dtype=np.float32)
        actions = wp.array(np.tile(action_np, (env.world_count, 1)), dtype=wp.float32, device=device)

        env.step(actions)
        env.reset()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(4):
                env.step(actions)
        wp.capture_launch(capture.graph)

        obs = env.obs.numpy()
        rewards = env.step_rewards.numpy()
        dones = env.step_dones.numpy()
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        targets = env.control.joint_target_q.numpy().reshape(env.world_count, -1)
        default_q = env.default_joint_pos.numpy()
        policy_to_model = np.asarray(env._policy_to_model_joint, dtype=np.int32)
        expected_targets = default_q.copy()
        expected_targets[policy_to_model] += config.action_scale * action_np

        self.assertEqual(obs.shape, (env.world_count, env.obs_dim))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(dones, 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            targets[:, 6:18], np.tile(expected_targets, (env.world_count, 1)), rtol=0.0, atol=1.0e-6
        )
        self.assertGreater(float(np.min(q[:, 2])), 0.2)


if __name__ == "__main__":
    unittest.main()
