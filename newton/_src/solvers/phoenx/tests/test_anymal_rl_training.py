# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training.examples import train_anymal_full_control_curriculum as anymal_curriculum
from newton._src.solvers.phoenx.rl_training.examples import train_anymal_walk_phoenx_ppo as anymal_walk
from newton._src.solvers.phoenx.rl_training.reward_functions import (
    command_progress_2d,
    fall_indicator,
    gaussian_reward,
    pd_torque,
    projected_gravity_upright_reward,
    tracking_reward_2d,
)
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


@wp.kernel
def reward_functions_smoke_kernel(out: wp.array[wp.float32]):
    out[0] = tracking_reward_2d(wp.float32(1.0), wp.float32(0.0), wp.float32(1.0), wp.float32(0.0), wp.float32(0.5))
    out[1] = tracking_reward_2d(wp.float32(0.0), wp.float32(0.0), wp.float32(1.0), wp.float32(0.0), wp.float32(0.5))
    out[2] = gaussian_reward(wp.float32(0.5), wp.float32(0.25))
    out[3] = projected_gravity_upright_reward(wp.vec3(0.0, 0.0, -1.0))
    out[4] = fall_indicator(wp.float32(0.2), wp.float32(0.3), wp.float32(1.0), wp.float32(0.3))
    out[5] = command_progress_2d(wp.float32(0.4), wp.float32(0.2), wp.float32(0.5), wp.float32(-0.5))
    out[6] = pd_torque(wp.float32(1.0), wp.float32(0.5), wp.float32(0.2), wp.float32(100.0), wp.float32(2.0))


class TestAnymalPhoenXRL(unittest.TestCase):
    def test_robot_runners_default_to_graph_capture(self) -> None:
        self.assertEqual(anymal_curriculum._make_parser().parse_args([]).execution_mode, "graph_leapfrog")
        self.assertEqual(anymal_walk._make_parser().parse_args([]).execution_mode, "graph_leapfrog")

    def test_full_control_curriculum_phase_env_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        phases = anymal_curriculum.build_full_control_curriculum()
        names = [phase.name for phase in phases]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "balance_and_step_forward")
        self.assertEqual(names[-1], "robust_full_control")
        self.assertLess(names.index("robust_forward"), names.index("run_forward"))
        self.assertLess(names.index("run_forward"), names.index("base_height_control"))
        self.assertLess(names.index("base_height_control"), names.index("turn_in_place"))
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
        run = next(phase for phase in phases if phase.name == "run_forward")
        run_overrides = dict(run.env_overrides)
        self.assertGreater(run.command_x_range[1], 1.0)
        self.assertLess(run_overrides["hip_abduction_reward_scale"], 0.0)
        self.assertLess(run_overrides["joint_position_reward_scale"], 0.0)
        self.assertIn(anymal_curriculum.FORWARD_RUN, run.eval_commands)
        side = next(phase for phase in phases if phase.name == "side_step")
        side_overrides = dict(side.env_overrides)
        self.assertEqual(side.command_x_range, (0.0, 0.0))
        self.assertLess(side_overrides["lin_vel_tracking_sigma"], 0.5)
        self.assertGreater(side_overrides["forward_progress_reward_scale"], 0.0)
        robust = phases[-1]
        robust_overrides = dict(robust.env_overrides)
        self.assertIn("lin_vel_tracking_sigma", robust_overrides)
        self.assertNotEqual(robust.command_height_range, (0.0, 0.0))
        self.assertGreater(robust.command_x_range[1], 1.0)
        self.assertLess(robust_overrides["joint_position_reward_scale"], 0.0)
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

    def test_collide_once_per_frame_steps_and_differs_from_per_substep(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")

        def run(collide_once: bool) -> np.ndarray:
            cfg = rl.ConfigEnvAnymalPhoenX(
                world_count=2,
                reward_mode="dense_command",
                command=(0.6, 0.0, 0.0, 0.0),
                sim_substeps=4,
                solver_iterations=2,
                velocity_iterations=1,
                max_episode_steps=0,
                collide_once_per_frame=collide_once,
                auto_reset=False,
            )
            env = rl.EnvAnymalPhoenX(cfg, device=device)
            env.reset()
            actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(8):
                    env.step(actions)
            wp.capture_launch(capture.graph)
            q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
            self.assertTrue(np.all(np.isfinite(q)))
            return q[:, 2].copy()  # base heights

        per_substep = run(False)
        once = run(True)
        # Both modes stay finite and upright (no fall-through), but the coarser
        # contact cadence yields a measurably different trajectory.
        self.assertTrue(np.all(per_substep > 0.2))
        self.assertTrue(np.all(once > 0.2))
        self.assertFalse(np.allclose(per_substep, once))

    def test_reward_functions_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        out = wp.zeros(7, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(reward_functions_smoke_kernel, dim=1, outputs=[out], device=device)
        wp.capture_launch(capture.graph)

        values = out.numpy()
        self.assertGreater(float(values[0]), 0.99)
        self.assertLess(float(values[1]), 0.05)
        self.assertLess(float(values[2]), 0.02)
        self.assertAlmostEqual(float(values[3]), 1.0, places=6)
        self.assertAlmostEqual(float(values[4]), 1.0, places=6)
        self.assertAlmostEqual(float(values[5]), 0.10, places=6)
        self.assertAlmostEqual(float(values[6]), 49.6, places=4)

    def test_ppo_checkpoint_insert_input_preserves_zero_inserted_policy_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        old_obs_dim = rl.OBS_DIM_ANYMAL - 1
        row_count = 3
        rng = np.random.default_rng(123)
        old_obs_np = rng.normal(size=(row_count, old_obs_dim)).astype(np.float32)
        insert_at = rl.COMMAND_OBS_OFFSET_ANYMAL + 3
        new_obs_np = np.concatenate(
            (old_obs_np[:, :insert_at], np.zeros((row_count, 1), dtype=np.float32), old_obs_np[:, insert_at:]),
            axis=1,
        )

        old_trainer = rl.TrainerPPO(
            obs_dim=old_obs_dim,
            action_dim=rl.ACTION_DIM_ANYMAL,
            hidden_layers=(16,),
            config=rl.ConfigPPO(),
            device=device,
            seed=37,
            activation="elu",
        )
        old_trainer.reserve_buffers(row_count)
        with TemporaryDirectory() as tmpdir:
            old_path = Path(tmpdir) / "old.npz"
            new_path = Path(tmpdir) / "new.npz"
            old_trainer.save_checkpoint(old_path, iteration=17)
            rl.insert_ppo_checkpoint_inputs(old_path, new_path, index=insert_at)

            with np.load(old_path, allow_pickle=False) as old_data, np.load(new_path, allow_pickle=False) as new_data:
                self.assertEqual(int(new_data["obs_dim"]), rl.OBS_DIM_ANYMAL)
                self.assertEqual(new_data["actor_weight_0"].shape[0], rl.OBS_DIM_ANYMAL)
                np.testing.assert_allclose(
                    new_data["actor_weight_0"][:insert_at], old_data["actor_weight_0"][:insert_at]
                )
                np.testing.assert_allclose(new_data["actor_weight_0"][insert_at], 0.0)
                np.testing.assert_allclose(
                    new_data["actor_weight_0"][insert_at + 1 :], old_data["actor_weight_0"][insert_at:]
                )
                np.testing.assert_allclose(new_data["actor_optimizer_m_0"][insert_at], 0.0)

            new_trainer = rl.load_ppo_checkpoint(new_path, device=device)
            new_trainer.reserve_buffers(row_count)

        old_obs = wp.array(old_obs_np, dtype=wp.float32, device=device)
        new_obs = wp.array(new_obs_np, dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            old_actions, old_log_probs, old_values = old_trainer.act_reuse(old_obs, seed=9, deterministic=True)
            new_actions, new_log_probs, new_values = new_trainer.act_reuse(new_obs, seed=9, deterministic=True)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(new_actions.numpy(), old_actions.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(new_log_probs.numpy(), old_log_probs.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(new_values.numpy(), old_values.numpy(), rtol=0.0, atol=1.0e-6)

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

    def test_train_dense_command_graph_leapfrog_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal graph-leapfrog training tests")
        env_config = rl.ConfigEnvAnymalPhoenX(
            world_count=2,
            reward_mode="dense_command",
            command=(0.0, 0.0, 0.0, 0.0),
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
            max_episode_steps=0,
            auto_reset=False,
        )
        ppo_config = rl.ConfigPPO(
            train_epochs=1,
            normalize_advantages=False,
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            entropy_coeff=0.0,
            mirror_loss_coeff=0.0,
            shared_value_network=True,
            manual_actor_backward=True,
            manual_critic_backward=True,
        )

        result = rl.train_anymal_ppo(
            rl.ConfigTrainAnymalPPO(
                iterations=2,
                rollout_steps=1,
                hidden_layers=(8,),
                env_config=env_config,
                ppo_config=ppo_config,
                device=device,
                seed=41,
                log_interval=0,
                use_target_curriculum=False,
                randomize_target_positions=False,
                randomize_commands=True,
                command_x_range=(-0.10, 0.10),
                command_y_range=(-0.05, 0.05),
                command_yaw_range=(-0.20, 0.20),
                command_height_range=(-0.02, 0.02),
                command_yaw_min_abs=0.05,
                command_zero_probability=0.10,
                execution_mode="graph_leapfrog",
            )
        )

        self.assertEqual([stat.iteration for stat in result.history], [0, 1])
        self.assertEqual(result.trainer.iteration, 2)
        self.assertGreater(result.trainer.actor_optimizer.step_count, 0)
        for stat in result.history:
            self.assertTrue(np.isfinite(stat.mean_reward))
            self.assertTrue(np.isfinite(stat.mean_done))
            self.assertTrue(np.isfinite(stat.mean_forward_velocity))
            self.assertTrue(np.isfinite(stat.policy_loss))
            self.assertTrue(np.isfinite(stat.value_loss))

    def test_train_dense_command_graph_leapfrog_repeatable_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal deterministic graph-leapfrog training tests")

        def run_once():
            env_config = rl.ConfigEnvAnymalPhoenX(
                world_count=2,
                reward_mode="dense_command",
                command=(0.0, 0.0, 0.0, 0.0),
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=1,
                max_episode_steps=0,
                auto_reset=False,
            )
            ppo_config = rl.ConfigPPO(
                train_epochs=1,
                normalize_advantages=False,
                actor_lr=1.0e-3,
                critic_lr=1.0e-3,
                entropy_coeff=0.0,
                mirror_loss_coeff=0.0,
                shared_value_network=True,
                manual_actor_backward=True,
                manual_critic_backward=True,
            )
            return rl.train_anymal_ppo(
                rl.ConfigTrainAnymalPPO(
                    iterations=2,
                    rollout_steps=2,
                    hidden_layers=(8,),
                    env_config=env_config,
                    ppo_config=ppo_config,
                    device=device,
                    seed=43,
                    log_interval=0,
                    use_target_curriculum=False,
                    randomize_target_positions=False,
                    randomize_commands=True,
                    command_x_range=(-0.10, 0.10),
                    command_y_range=(-0.05, 0.05),
                    command_yaw_range=(-0.20, 0.20),
                    command_height_range=(-0.02, 0.02),
                    command_yaw_min_abs=0.05,
                    command_zero_probability=0.10,
                    execution_mode="graph_leapfrog",
                )
            )

        first = run_once()
        second = run_once()

        for first_stat, second_stat in zip(first.history, second.history, strict=True):
            np.testing.assert_allclose(
                tuple(first_stat.__dict__.values()), tuple(second_stat.__dict__.values()), rtol=0.0, atol=0.0
            )

        for first_weight, second_weight in zip(
            first.trainer.actor.net.weights, second.trainer.actor.net.weights, strict=True
        ):
            np.testing.assert_allclose(first_weight.numpy(), second_weight.numpy(), rtol=0.0, atol=0.0)
        for first_bias, second_bias in zip(
            first.trainer.actor.net.biases, second.trainer.actor.net.biases, strict=True
        ):
            np.testing.assert_allclose(first_bias.numpy(), second_bias.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            first.trainer.actor.log_std.numpy(), second.trainer.actor.log_std.numpy(), rtol=0.0, atol=0.0
        )

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
                command=(0.0, 0.0, 0.0, 0.0),
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
                (0.0, 0.0, 0.0, 0.0),
                (0.6, 0.0, 0.0, 0.02),
                (0.0, 0.35, 0.0, -0.03),
                (0.0, 0.0, 0.75, 0.04),
            ),
            dtype=np.float32,
        )
        env.set_commands(commands)

        with wp.ScopedCapture(device=device) as capture:
            obs = env.observe()
        wp.capture_launch(capture.graph)

        obs_np = obs.numpy()
        np.testing.assert_allclose(
            obs_np[:, rl.COMMAND_OBS_OFFSET_ANYMAL : rl.COMMAND_OBS_OFFSET_ANYMAL + rl.COMMAND_DIM_ANYMAL],
            commands,
            rtol=0.0,
            atol=1.0e-6,
        )
        success = env.successes.numpy()
        self.assertGreater(float(success[0]), 0.95)
        self.assertLess(float(success[3]), 0.25)

    def test_dense_command_accepts_legacy_three_value_commands_inside_graph(self) -> None:
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
        commands = np.asarray(((0.2, 0.0, 0.0), (0.0, -0.2, 0.1)), dtype=np.float32)
        expected = np.concatenate((commands, np.zeros((env.world_count, 1), dtype=np.float32)), axis=1)
        env.set_commands(commands)

        with wp.ScopedCapture(device=device) as capture:
            obs = env.observe()
        wp.capture_launch(capture.graph)

        obs_np = obs.numpy()
        np.testing.assert_allclose(
            obs_np[:, rl.COMMAND_OBS_OFFSET_ANYMAL : rl.COMMAND_OBS_OFFSET_ANYMAL + rl.COMMAND_DIM_ANYMAL],
            expected,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_dense_command_height_reward_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX Anymal RL tests")
        common = {
            "world_count": 1,
            "reward_mode": "dense_command",
            "sim_substeps": 1,
            "solver_iterations": 1,
            "velocity_iterations": 1,
            "max_episode_steps": 0,
            "auto_reset": False,
            "lin_vel_reward_scale": 0.0,
            "yaw_rate_reward_scale": 0.0,
            "base_height_reward_scale": 1.0,
            "base_height_tracking_sigma": 0.04,
            "z_vel_reward_scale": 0.0,
            "ang_vel_reward_scale": 0.0,
            "action_rate_reward_scale": 0.0,
            "joint_speed_reward_scale": 0.0,
            "flat_orientation_reward_scale": 0.0,
            "forward_progress_reward_scale": 0.0,
            "fall_reward_scale": 0.0,
            "energy_reward_scale": 0.0,
        }
        nominal_env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(**common, command=(0.0, 0.0, 0.0, 0.0)),
            device=device,
        )
        high_env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(**common, command=(0.0, 0.0, 0.0, 0.10)),
            device=device,
        )

        with wp.ScopedCapture(device=device) as capture:
            nominal_env.observe()
            high_env.observe()
        wp.capture_launch(capture.graph)

        nominal_reward = float(nominal_env.rewards.numpy()[0])
        high_reward = float(high_env.rewards.numpy()[0])
        self.assertGreater(nominal_reward, 0.9)
        self.assertLess(high_reward, 0.01)

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
