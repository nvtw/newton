# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestAntPhoenXRL(unittest.TestCase):
    def test_ant_finite_state_explosion_terminates_and_resets(self) -> None:
        device = require_cuda_graph_capture("PhoenX Ant RL tests")
        env = rl.EnvAntPhoenX(
            rl.ConfigEnvAntPhoenX(world_count=2, max_episode_steps=0, auto_reset=False),
            device=device,
        )
        qd = env.state_0.joint_qd.numpy()
        qd[0] = np.float32(env.config.max_abs_root_linear_velocity + 1.0)
        env.state_0.joint_qd.assign(qd)

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        np.testing.assert_array_equal(env.dones.numpy(), np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(float(env.rewards.numpy()[0]), env.config.termination_reward)

        env.reset_done()
        env.observe()
        self.assertTrue(np.isfinite(env.obs.numpy()).all())
        np.testing.assert_array_equal(env.dones.numpy(), np.zeros(2, dtype=np.float32))

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
