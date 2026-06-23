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


if __name__ == "__main__":
    unittest.main()
