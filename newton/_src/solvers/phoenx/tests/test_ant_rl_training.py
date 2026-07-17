# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training.ant import _make_parser, _make_trainer
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestAntPhoenXRL(unittest.TestCase):
    def test_ant_mraksha_profile_matches_task_contract(self) -> None:
        device = require_cuda_graph_capture("PhoenX Ant RL tests")
        env = rl.EnvAntPhoenX(
            rl.ConfigEnvAntPhoenX(
                world_count=2,
                task_profile="mraksha",
                frame_dt=1.0 / 60.0,
                sim_substeps=2,
                articulation_mode="reduced",
                torque_limit=7.5,
                max_episode_steps=900,
                min_height=0.31,
                max_height=0.0,
                ground_friction=1.0,
                joint_noise=0.0,
                velocity_noise=0.0,
                joint_damping=0.1,
                joint_armature=0.05,
                contact_gap=0.005,
                healthy_reward=0.5,
                termination_reward=-2.0,
                auto_reset=False,
            ),
            device=device,
        )

        self.assertEqual(env.obs.shape, (2, 36))
        self.assertEqual(env.solver.articulation_mode, "reduced")
        np.testing.assert_allclose(env.obs.numpy()[:, 0], 0.5, rtol=0.0, atol=1.0e-6)
        expected_joint_q = np.array([0.0, np.pi / 4.0, 0.0, -np.pi / 4.0, 0.0, -np.pi / 4.0, 0.0, np.pi / 4.0])
        joint_q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        np.testing.assert_allclose(
            joint_q[:, 7:15], np.broadcast_to(expected_joint_q, (env.world_count, 8)), rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(env.obs.numpy()[:, 10:12], 1.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.rewards.numpy(), 1.1, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.model.joint_damping.numpy()[6:14], 0.1, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.model.joint_armature.numpy()[6:14], 0.05, rtol=0.0, atol=1.0e-6)
        args = _make_parser().parse_args([])
        self.assertEqual(args.articulation_mode, "reduced")
        self.assertEqual(args.solver_iterations, 8)
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(args.rollout_steps, 32)
        self.assertEqual(args.hidden_layers, [400, 200, 100])
        self.assertEqual(args.log_std_init, 0.0)
        self.assertEqual(args.min_upright_cos, -1.0)
        trainer = _make_trainer(args, env)
        self.assertEqual(trainer.hidden_layers, (400, 200, 100))
        self.assertEqual(trainer.config.minibatch_size, 32768)
        self.assertEqual(trainer.config.replay_ratio, 5.0)
        self.assertEqual(trainer.config.value_clip_range, 0.2)
        self.assertEqual(trainer.config.adaptive_kl_target, 0.0)
        self.assertEqual(trainer.config.manual_mlp_weight_grad_dtype, "bfloat16")
        self.assertEqual(trainer.config.manual_mlp_forward_dtype, "bfloat16")
        self.assertFalse(trainer.squash_actions)

        actions = wp.full((env.world_count, env.action_dim), 2.0, dtype=wp.float32, device=device)
        env.step(actions)
        joint_forces = env.control.joint_f.numpy().reshape(env.world_count, env.dof_stride)
        np.testing.assert_allclose(joint_forces[:, 6:14], 7.5, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.current_actions.numpy(), 2.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.obs.numpy()[:, 28:36], 2.0, rtol=0.0, atol=1.0e-6)

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
