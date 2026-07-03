# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestAntPhoenXRL(unittest.TestCase):
    def test_ant_step_is_cuda_graph_capturable(self) -> None:
        device = require_cuda_graph_capture("PhoenX Ant RL tests")
        env = rl.EnvAntPhoenX(
            rl.ConfigEnvAntPhoenX(world_count=8, max_episode_steps=0, auto_reset=False),
            device=device,
        )
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        np.testing.assert_allclose(env.dones.numpy(), 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(env.obs.numpy()[:, 8], -1.0, rtol=0.0, atol=1.0e-6)
        env.step(actions)
        env.reset()
        with wp.ScopedCapture(device=device) as capture:
            env.step(actions)
        wp.capture_launch(capture.graph)

        obs = env.obs.numpy()
        rewards = env.step_rewards.numpy()
        dones = env.step_dones.numpy()
        heights = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)[:, 1]
        self.assertEqual(obs.shape, (8, env.obs_dim))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(np.isfinite(rewards)))
        np.testing.assert_allclose(dones, np.zeros(env.world_count, dtype=np.float32), atol=0.0, rtol=0.0)
        self.assertGreater(float(np.min(heights)), 0.5)


if __name__ == "__main__":
    unittest.main()
