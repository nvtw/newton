# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.rl as rl
from newton._src.solvers.phoenx.benchmarks.bench_humanoid_train_to_gate import (
    StatsEvaluateHumanoid,
    check_gate,
)
from newton._src.solvers.phoenx.rl_training.examples.train_humanoid_phoenx_ppo import (
    _make_parser,
    build_ppo_config,
)
from newton._src.solvers.phoenx.rl_training.humanoid import _resolve_humanoid_actuators
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestHumanoidPhoenXRL(unittest.TestCase):
    def test_actuator_contract_uses_imported_metadata(self) -> None:
        """Resolve reordered motors and reject duplicate DOF targets."""
        robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
        robot.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            up_axis="Z",
            parse_meshes=False,
            parse_visuals=False,
            ignore_names=("floor", "ground"),
            enable_self_collisions=True,
            parse_mujoco_options=False,
        )
        expected_gears = np.asarray(
            [
                67.5,
                67.5,
                67.5,
                45.0,
                45.0,
                135.0,
                90.0,
                22.5,
                22.5,
                45.0,
                45.0,
                135.0,
                90.0,
                22.5,
                22.5,
                67.5,
                67.5,
                45.0,
                67.5,
                67.5,
                45.0,
            ],
            dtype=np.float32,
        )

        dofs, gears = _resolve_humanoid_actuators(robot)
        np.testing.assert_array_equal(dofs, np.arange(6, 27, dtype=np.int32))
        np.testing.assert_array_equal(gears, expected_gears)

        actuator_keys = (
            "mujoco:actuator_trnid",
            "mujoco:actuator_target_label",
            "mujoco:actuator_trntype",
            "mujoco:actuator_gear",
        )
        for key in actuator_keys:
            values = robot.custom_attributes[key].values
            if isinstance(values, list):
                values.reverse()
        reordered_dofs, reordered_gears = _resolve_humanoid_actuators(robot)
        np.testing.assert_array_equal(reordered_dofs, dofs)
        np.testing.assert_array_equal(reordered_gears, gears)

        trnid = robot.custom_attributes["mujoco:actuator_trnid"].values
        trnid[1] = trnid[0]
        with self.assertRaisesRegex(RuntimeError, "duplicates joint|cover every"):
            _resolve_humanoid_actuators(robot)

        for key in actuator_keys:
            values = robot.custom_attributes[key].values
            if isinstance(values, list):
                values.pop()
        with self.assertRaisesRegex(RuntimeError, "requires 21 complete joint actuators"):
            _resolve_humanoid_actuators(robot)

    def test_direct_task_step_inside_cuda_graph(self) -> None:
        """Apply source-MJCF torques in model DOF order inside a CUDA graph."""
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
        expected_gears = np.asarray(
            [
                67.5,
                67.5,
                67.5,
                45.0,
                45.0,
                135.0,
                90.0,
                22.5,
                22.5,
                45.0,
                45.0,
                135.0,
                90.0,
                22.5,
                22.5,
                67.5,
                67.5,
                45.0,
                67.5,
                67.5,
                45.0,
            ],
            dtype=np.float32,
        )
        expected_kp = np.asarray(
            [20, 20, 10, 10, 10, 20, 5, 2, 2, 10, 10, 20, 5, 2, 2, 10, 10, 2, 10, 10, 2],
            dtype=np.float32,
        )
        expected_kd = np.asarray([5, 5, 5, 5, 5, 5, 0.1, 1, 1, 5, 5, 5, 0.1, 1, 1, 5, 5, 1, 5, 5, 1], dtype=np.float32)

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
        expected_f = action_row * expected_gears

        self.assertEqual(obs.shape, (env.world_count, rl.OBS_DIM_HUMANOID))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_array_equal(env.joint_gears.numpy(), expected_gears)
        np.testing.assert_allclose(dones, 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(env.model.joint_target_ke.numpy()[6:27], expected_kp)
        np.testing.assert_array_equal(env.model.joint_target_kd.numpy()[6:27], expected_kd)
        np.testing.assert_array_equal(env.model.joint_effort_limit.numpy()[6:27], np.full(21, 150.0, np.float32))
        np.testing.assert_allclose(joint_f[:, 6:27], np.tile(expected_f, (env.world_count, 1)), rtol=0.0, atol=1.0e-5)
        self.assertGreater(float(np.min(q[:, 2])), 1.0)

    def test_initial_reward_matches_direct_task_terms(self) -> None:
        """Match the initial reward and source MJCF integration timestep."""
        device = require_cuda_graph_capture("PhoenX Humanoid RL tests")
        env = rl.EnvHumanoidPhoenX(
            rl.ConfigEnvHumanoidPhoenX(world_count=2, max_episode_steps=0, auto_reset=False),
            device=device,
        )
        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)
        self.assertAlmostEqual(env.config.frame_dt / env.config.sim_substeps, 0.00555, delta=1.0e-5)

        rewards = env.rewards.numpy()
        np.testing.assert_allclose(rewards, 2.6, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.obs.numpy()[:, 0], 1.34, rtol=0.0, atol=1.0e-6)

    def test_humanoid_training_matches_mraksha_ppo_contract(self) -> None:
        args = _make_parser().parse_args([])
        config = build_ppo_config(args, 4096 * 32)

        self.assertEqual(args.hidden_layers, (400, 200, 100))
        self.assertEqual(args.rollout_steps, 32)
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(config.train_epochs, 5)
        self.assertEqual(config.minibatch_size, 4096 * 32 // 4)
        self.assertEqual(config.actor_lr, 1.0e-4)
        self.assertEqual(config.critic_lr, 1.0e-4)
        self.assertEqual(config.value_clip_range, 0.2)
        self.assertEqual(config.adaptive_kl_target, 0.008)
        self.assertTrue(config.normalize_observations)
        self.assertEqual(config.observation_clip, 10.0)

    def test_ppo_observation_normalization_round_trips_checkpoint(self) -> None:
        device = require_cuda_graph_capture("PhoenX Humanoid RL tests")
        config = rl.ConfigPPO(normalize_observations=True, observation_clip=10.0)
        trainer = rl.TrainerPPO(
            obs_dim=2,
            action_dim=1,
            hidden_layers=(4,),
            config=config,
            device=device,
            seed=7,
        )
        obs = wp.array(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )

        normalized = trainer.prepare_observations(obs, update_stats=True).numpy()
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_allclose(trainer._obs_count.numpy(), 6.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(trainer._obs_mean.numpy(), [8.0 / 3.0, 10.0 / 3.0], rtol=0.0, atol=1.0e-6)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = f"{tmpdir}/normalized_ppo.npz"
            trainer.save_checkpoint(checkpoint)
            restored = rl.load_ppo_checkpoint(checkpoint, config=config, device=device)

        np.testing.assert_allclose(restored._obs_mean.numpy(), trainer._obs_mean.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(restored._obs_m2.numpy(), trainer._obs_m2.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(restored._obs_count.numpy(), trainer._obs_count.numpy(), rtol=0.0, atol=0.0)
        actions, _log_probs, _values = trainer.act(obs, seed=11, deterministic=True)
        restored_actions, _log_probs, _values = restored.act(obs, seed=11, deterministic=True)
        np.testing.assert_allclose(restored_actions.numpy(), actions.numpy(), rtol=0.0, atol=1.0e-6)

    def test_humanoid_gate_rejects_stationary_and_accepts_walking(self) -> None:
        args = argparse.Namespace(
            gate_max_fall_fraction=0.05,
            gate_min_survival_fraction=0.95,
            gate_min_forward_velocity=0.4,
            gate_min_displacement=3.0,
            gate_min_forward_fraction=0.9,
            gate_min_success=0.4,
            gate_min_upright_cos=0.8,
        )
        walking = StatsEvaluateHumanoid(
            steps=600,
            fall_fraction=0.0,
            survival_fraction=1.0,
            mean_forward_velocity=0.7,
            mean_displacement_x=7.0,
            forward_fraction=1.0,
            mean_success=0.7,
            mean_upright_cos=0.95,
            mean_action_rms=0.25,
            finite=True,
        )
        stationary = StatsEvaluateHumanoid(
            steps=600,
            fall_fraction=0.0,
            survival_fraction=1.0,
            mean_forward_velocity=0.0,
            mean_displacement_x=0.0,
            forward_fraction=0.0,
            mean_success=0.0,
            mean_upright_cos=1.0,
            mean_action_rms=0.0,
            finite=True,
        )

        self.assertEqual(check_gate(walking, args), [])
        self.assertEqual(
            check_gate(stationary, args),
            ["mean forward velocity", "mean forward displacement", "forward-moving fraction", "locomotion success"],
        )


if __name__ == "__main__":
    unittest.main()
