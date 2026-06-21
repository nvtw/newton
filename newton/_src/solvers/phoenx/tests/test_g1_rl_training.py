# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.rl as rl
from newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate import benchmark_train_to_gate
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.env import make_seed_counter
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture

_NANOG1_ROOT = Path("/home/twidmer/Documents/git/nanoG1")
_PUFFERLIB_ROOT = Path("/home/twidmer/Documents/git/PufferLib")


def _g1_test_env(world_count: int = 1) -> rl.EnvG1PhoenX:
    device = require_cuda_graph_capture("PhoenX G1 RL tests")
    config = rl.ConfigEnvG1PhoenX(
        world_count=int(world_count),
        sim_substeps=5,
        solver_iterations=2,
        velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
        max_episode_steps=0,
        auto_reset=False,
    )
    return rl.EnvG1PhoenX(config, device=device)


def _load_reference_module(path: Path, name: str):
    if not path.is_file():
        raise unittest.SkipTest(f"missing nanoG1 reference file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"could not import nanoG1 reference file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_nanog1_recipe():
    return _load_reference_module(_NANOG1_ROOT / "recipe.py", "nanog1_recipe_reference")


def _load_nanog1_deploy():
    return _load_reference_module(_NANOG1_ROOT / "deploy" / "deploy_g1.py", "nanog1_deploy_reference")


def _read_c_array(path: Path, name: str, *, dtype: type = float) -> np.ndarray:
    if not path.is_file():
        raise unittest.SkipTest(f"missing nanoG1 reference file: {path}")
    text = path.read_text()
    pattern = rf"static const (?:double|int) {re.escape(name)}\[[^\]]+\] = \{{([^}}]+)\}};"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"missing C array {name!r} in {path}")
    values = [item.strip() for item in match.group(1).replace("\n", " ").split(",") if item.strip()]
    np_dtype = np.int64 if dtype is int else np.float64
    return np.asarray([dtype(value) for value in values], dtype=np_dtype)


def _reference_obs_from_nanog1_deploy(
    deploy,
    q: np.ndarray,
    qd: np.ndarray,
    command: np.ndarray,
    prev_action: np.ndarray,
    episode_step: int,
) -> np.ndarray:
    obs = np.zeros(rl.OBS_DIM_G1, dtype=np.float32)
    # Newton and nanoG1 host qpos store the base-to-world root quaternion.
    # The deploy helper consumes the Unitree IMU convention, so use the
    # conjugate to compare the same projected-gravity observation.
    unitree_quat_wxyz = np.asarray((q[6], -q[3], -q[4], -q[5]), dtype=np.float64)
    phase = 2.0 * math.pi * float(episode_step % deploy.PHASE_PERIOD) / float(deploy.PHASE_PERIOD)
    obs[0:3] = np.float32(deploy.ANG_VEL_SCALE) * qd[3:6]
    obs[3:6] = deploy.projected_gravity(unitree_quat_wxyz).astype(np.float32)
    obs[6:9] = command
    obs[9:38] = q[7 : 7 + rl.ACTION_DIM_G1] - deploy.HOME.astype(np.float32)
    obs[38:67] = np.float32(deploy.DOF_VEL_SCALE) * qd[6 : 6 + rl.ACTION_DIM_G1]
    obs[67:96] = prev_action
    obs[96] = math.sin(phase)
    obs[97] = math.cos(phase)
    return obs


class TestG1PhoenXRL(unittest.TestCase):
    def test_nanog1_recipe_constants_match_reference(self) -> None:
        reference = _load_nanog1_recipe()
        recipe = reference.RECIPE
        dt_match = re.search(r"-DG1_DT=([0-9.]+)f", reference.TASK_FLAGS)
        decimation_match = re.search(r"-DENV_DECIMATION=([0-9]+)", reference.TASK_FLAGS)
        solver_match = re.search(r"-DSOL_ITER=([0-9]+)", reference.TASK_FLAGS)
        mirror_match = re.search(r"-DG1_MIRROR_LOSS=([0-9.]+)", reference.TRAIN_FLAGS)
        self.assertIsNotNone(dt_match)
        self.assertIsNotNone(decimation_match)
        self.assertIsNotNone(solver_match)
        self.assertIsNotNone(mirror_match)

        reference_frame_dt = float(dt_match.group(1)) * int(decimation_match.group(1))
        self.assertAlmostEqual(g1_recipe.FRAME_DT, reference_frame_dt)
        self.assertGreaterEqual(g1_recipe.SIM_SUBSTEPS, int(decimation_match.group(1)))
        self.assertGreaterEqual(g1_recipe.SOLVER_ITERATIONS, int(solver_match.group(1)))
        self.assertEqual(g1_recipe.CONTROLLED_ACTION_COUNT, 12)
        self.assertIn("-DG1_TASK_V3", reference.TASK_FLAGS)
        self.assertIn("-DG1_PD_UNITREE", reference.TASK_FLAGS)

        self.assertAlmostEqual(g1_recipe.ACTION_SCALE, float(recipe["env.action_scale"]))
        self.assertEqual(g1_recipe.MAX_EPISODE_STEPS, int(recipe["env.max_episode_len"]))
        self.assertAlmostEqual(g1_recipe.W_TRACK_LIN, float(recipe["env.w_track_lin"]))
        self.assertAlmostEqual(g1_recipe.W_TRACK_ANG, float(recipe["env.w_track_ang"]))
        self.assertAlmostEqual(g1_recipe.W_LIN_VEL_Z, float(recipe["env.w_lin_vel_z"]))
        self.assertAlmostEqual(g1_recipe.W_ANG_VEL_XY, float(recipe["env.w_ang_vel_xy"]))
        self.assertAlmostEqual(g1_recipe.W_ORIENTATION, float(recipe["env.w_orientation"]))
        self.assertAlmostEqual(g1_recipe.W_TORQUE, float(recipe["env.w_torque"]))
        self.assertAlmostEqual(g1_recipe.W_ACTION_RATE, float(recipe["env.w_action_rate"]))
        self.assertAlmostEqual(g1_recipe.W_ALIVE, float(recipe["env.w_alive"]))
        self.assertAlmostEqual(g1_recipe.W_TERMINATION, float(recipe["env.w_termination"]))

        self.assertEqual(g1_recipe.SEED, int(recipe["train.seed"]))
        self.assertAlmostEqual(g1_recipe.GAMMA, float(recipe["train.gamma"]))
        self.assertAlmostEqual(g1_recipe.GAE_LAMBDA, float(recipe["train.gae_lambda"]))
        self.assertAlmostEqual(g1_recipe.CLIP_RATIO, float(recipe["train.clip_coef"]))
        self.assertAlmostEqual(g1_recipe.ENTROPY_COEFF, float(recipe["train.ent_coef"]))
        self.assertAlmostEqual(g1_recipe.REPLAY_RATIO, float(recipe["train.replay_ratio"]))
        self.assertAlmostEqual(g1_recipe.VTRACE_RHO_CLIP, float(recipe["train.vtrace_rho_clip"]))
        self.assertAlmostEqual(g1_recipe.VTRACE_C_CLIP, float(recipe["train.vtrace_c_clip"]))
        self.assertAlmostEqual(g1_recipe.PRIORITY_ALPHA, float(recipe["train.prio_alpha"]))
        self.assertAlmostEqual(g1_recipe.PRIORITY_BETA, float(recipe["train.prio_beta0"]))
        self.assertEqual(g1_recipe.MINIBATCH_SIZE, int(recipe["train.minibatch_size"]))
        self.assertEqual(g1_recipe.ROLLOUT_STEPS, int(recipe["train.horizon"]))
        self.assertAlmostEqual(g1_recipe.MAX_GRAD_NORM, float(recipe["train.max_grad_norm"]))
        self.assertAlmostEqual(g1_recipe.MIRROR_LOSS_COEFF, float(mirror_match.group(1)))

    def test_nanog1_model_and_deploy_constants_match_env(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 nanoG1 constant parity tests")
        deploy = _load_nanog1_deploy()
        header = _NANOG1_ROOT / "web" / "g1_model_const.h"
        key_qpos = _read_c_array(header, "hc_key_qpos")
        ctrl_range = _read_c_array(header, "hc_act_ctrlrange").reshape(rl.ACTION_DIM_G1, 2)
        dof_damping = _read_c_array(header, "hc_dof_damping")

        env = rl.EnvG1PhoenX(g1_recipe.default_g1_env_config(world_count=1), device=device)
        np.testing.assert_allclose(env.default_joint_pos.numpy(), deploy.HOME, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.default_joint_pos.numpy(), key_qpos[7:], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.ctrl_lower.numpy(), deploy.CTRL_RANGE[:, 0], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_upper.numpy(), deploy.CTRL_RANGE[:, 1], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_lower.numpy(), ctrl_range[:, 0], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_upper.numpy(), ctrl_range[:, 1], rtol=0.0, atol=1.0e-5)

        expected_kp = deploy.KP.astype(np.float32)
        expected_kd = deploy.KD.astype(np.float32)
        expected_kd[: g1_recipe.CONTROLLED_ACTION_COUNT] += dof_damping[6 : 6 + g1_recipe.CONTROLLED_ACTION_COUNT]
        np.testing.assert_allclose(env.actuator_ke.numpy(), expected_kp, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.actuator_kd.numpy(), expected_kd, rtol=0.0, atol=1.0e-6)
        self.assertEqual(deploy.NU, rl.ACTION_DIM_G1)
        self.assertEqual(deploy.LEG_DOF, g1_recipe.CONTROLLED_ACTION_COUNT)
        self.assertAlmostEqual(deploy.CONTROL_DT, g1_recipe.FRAME_DT)
        self.assertEqual(deploy.PHASE_PERIOD, g1_recipe.PHASE_PERIOD)

    def test_nanog1_observation_contract_matches_graph_observe(self) -> None:
        env = _g1_test_env(world_count=1)
        deploy = _load_nanog1_deploy()
        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        quat_wxyz = np.asarray((0.94, 0.12, -0.25, 0.20), dtype=np.float32)
        quat_wxyz /= np.linalg.norm(quat_wxyz)
        q[:] = 0.0
        q[0:3] = np.asarray((0.11, -0.07, 0.82), dtype=np.float32)
        q[3:7] = np.asarray((quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]), dtype=np.float32)
        q[7:] = deploy.HOME.astype(np.float32) + np.linspace(-0.03, 0.03, rl.ACTION_DIM_G1, dtype=np.float32)
        qd[:] = np.linspace(-0.4, 0.6, qd.size, dtype=np.float32)
        command = np.asarray((0.42, -0.13, 0.70), dtype=np.float32)
        prev_action = np.linspace(-1.0, 1.0, rl.ACTION_DIM_G1, dtype=np.float32)
        episode_step = 13

        env.state_0.joint_q.assign(q)
        env.state_0.joint_qd.assign(qd)
        env.command.assign(command.reshape(1, 3))
        env.current_actions.assign(prev_action.reshape(1, rl.ACTION_DIM_G1))
        env.previous_actions.zero_()
        env.episode_steps.assign(np.asarray([episode_step], dtype=np.int32))
        with wp.ScopedCapture(device=env.device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(env.device)

        expected = _reference_obs_from_nanog1_deploy(deploy, q, qd, command, prev_action, episode_step)
        np.testing.assert_allclose(env.obs.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)

    def test_nanog1_action_target_contract_matches_graph_step(self) -> None:
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=require_cuda_graph_capture("PhoenX G1 nanoG1 action parity tests"),
        )
        deploy = _load_nanog1_deploy()
        raw_actions = np.linspace(-1.6, 1.6, rl.ACTION_DIM_G1, dtype=np.float32).reshape(1, rl.ACTION_DIM_G1)
        actions = wp.array(raw_actions, dtype=wp.float32, device=env.device)

        with wp.ScopedCapture(device=env.device) as capture:
            env.step(actions)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(env.device)

        expected_actions = np.clip(raw_actions[0], -1.0, 1.0)
        expected_actions[g1_recipe.CONTROLLED_ACTION_COUNT :] = 0.0
        expected_target = deploy.HOME.astype(np.float32) + np.float32(deploy.ACTION_SCALE) * expected_actions
        expected_target = np.clip(expected_target, deploy.CTRL_RANGE[:, 0], deploy.CTRL_RANGE[:, 1])
        np.testing.assert_allclose(env.current_actions.numpy()[0], expected_actions, rtol=0.0, atol=0.0)

        target_q = env.control.joint_target_q.numpy()
        if env.model.use_coord_layout_targets:
            actual_target = target_q[7 : 7 + rl.ACTION_DIM_G1]
        else:
            actual_target = target_q[6 : 6 + rl.ACTION_DIM_G1]
        np.testing.assert_allclose(actual_target, expected_target, rtol=0.0, atol=1.0e-6)

    def test_shared_value_mlp_forward_matches_numpy_for_nanog1_shape(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL network parity tests")
        puffernet = _PUFFERLIB_ROOT / "src" / "puffernet.h"
        if not puffernet.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {puffernet}")
        text = puffernet.read_text()
        self.assertIn("decoder output", text)
        self.assertIn("_gaussian_mean", text)
        self.assertIn("MinGRU* mingru", text)

        trainer = rl.TrainerPPO(
            obs_dim=rl.OBS_DIM_G1,
            action_dim=rl.ACTION_DIM_G1,
            hidden_layers=(128, 128, 128),
            config=rl.ConfigPPO(
                shared_value_network=True,
                manual_actor_backward=True,
                manual_critic_backward=True,
                manual_mlp_forward_dtype="float32",
                manual_mlp_weight_grad_dtype="float32",
            ),
            device=device,
            seed=101,
            squash_actions=True,
            activation="relu",
            log_std_init=g1_recipe.LOG_STD_INIT,
        )
        self.assertEqual(trainer.value_column, rl.ACTION_DIM_G1)
        self.assertEqual(trainer.actor.net.output_dim, rl.ACTION_DIM_G1 + 1)

        obs_np = np.linspace(-0.8, 0.9, 4 * rl.OBS_DIM_G1, dtype=np.float32).reshape(4, rl.OBS_DIM_G1)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        out = trainer.actor.net.forward_reuse(obs)
        with wp.ScopedCapture(device=device) as capture:
            trainer.actor.net.forward_reuse(obs)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        expected = obs_np
        for layer, (weight, bias) in enumerate(zip(trainer.actor.net.weights, trainer.actor.net.biases, strict=True)):
            expected = expected @ weight.numpy() + bias.numpy()
            if layer < len(trainer.actor.net.weights) - 1:
                expected = np.maximum(expected, 0.0)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2.0e-5, atol=2.0e-5)

    def test_default_recipe_enables_mirror_and_matches_config_defaults(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL recipe tests")

        env_config = g1_recipe.default_g1_env_config(world_count=1)
        env = rl.EnvG1PhoenX(env_config, device=device)
        train_config = rl.ConfigTrainG1PPO()
        ppo_config = g1_recipe.default_g1_ppo_config()

        self.assertEqual(env_config.sim_substeps, g1_recipe.SIM_SUBSTEPS)
        self.assertEqual(env.solver.world.substeps, 1)
        self.assertEqual(env_config.solver_iterations, g1_recipe.SOLVER_ITERATIONS)
        self.assertEqual(g1_recipe.VELOCITY_ITERATIONS, 1)
        self.assertEqual(env_config.velocity_iterations, g1_recipe.VELOCITY_ITERATIONS)
        self.assertEqual(env_config.w_track_lin, g1_recipe.W_TRACK_LIN)
        self.assertEqual(env_config.w_action_rate, g1_recipe.W_ACTION_RATE)
        self.assertEqual(env_config.rigid_contact_max_per_world, g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env_config.contact_geometry, g1_recipe.CONTACT_GEOMETRY)
        self.assertEqual(env_config.contact_geometry, "nanog1_foot_boxes")
        self.assertEqual(env_config.threads_per_world, g1_recipe.THREADS_PER_WORLD)
        self.assertEqual(env_config.multi_world_scheduler, g1_recipe.MULTI_WORLD_SCHEDULER)
        self.assertEqual(env_config.prepare_refresh_stride, g1_recipe.PREPARE_REFRESH_STRIDE)
        expected_leg_kd = np.array([4.0, 4.0, 4.0, 6.0, 3.0, 2.2] * 2, dtype=np.float32)
        np.testing.assert_allclose(env.model.joint_target_kd.numpy()[6:18], expected_leg_kd, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.model.joint_q.numpy()[3:7], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=0.0)
        labels = list(env.model.shape_label)
        self.assertIn("left_nanog1_foot_box", labels)
        self.assertIn("right_nanog1_foot_box", labels)
        flags = env.model.shape_flags.numpy()
        collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for shape_index, label in enumerate(labels):
            if "ankle_roll_link_geom_" in label:
                self.assertEqual(int(flags[shape_index]) & collide_bit, 0)
        with wp.ScopedCapture(device=device) as capture:
            env.model.collide(env.state_0, env.contacts)
        wp.capture_launch(capture.graph)
        self.assertGreater(int(env.contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(train_config.hidden_layers, g1_recipe.HIDDEN_LAYERS)
        self.assertEqual(train_config.activation, g1_recipe.ACTIVATION)
        self.assertEqual(train_config.rollout_steps, g1_recipe.ROLLOUT_STEPS)
        self.assertGreater(ppo_config.mirror_loss_coeff, 0.0)
        self.assertEqual(ppo_config.mirror_loss_coeff, g1_recipe.MIRROR_LOSS_COEFF)
        self.assertTrue(ppo_config.shared_value_network)
        self.assertEqual(ppo_config.shared_value_network, g1_recipe.SHARED_VALUE_NETWORK)
        self.assertEqual(ppo_config.minibatch_size, g1_recipe.MINIBATCH_SIZE)
        self.assertEqual(ppo_config.manual_mlp_forward_dtype, g1_recipe.MANUAL_MLP_FORWARD_DTYPE)

    def test_rejects_unknown_contact_geometry(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL contact geometry tests")
        config = rl.ConfigEnvG1PhoenX(world_count=1, contact_geometry="unknown")

        with self.assertRaisesRegex(ValueError, "contact_geometry"):
            rl.EnvG1PhoenX(config, device=device)

    def test_rejects_negative_contact_capacity(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL capacity tests")
        config = rl.ConfigEnvG1PhoenX(world_count=1, rigid_contact_max_per_world=-1)

        with self.assertRaisesRegex(ValueError, "rigid_contact_max_per_world"):
            rl.EnvG1PhoenX(config, device=device)

    def test_step_graph_capture_shapes_and_masks_actions(self) -> None:
        env = _g1_test_env(world_count=2)
        actions_np = np.full((env.world_count, env.action_dim), 1.5, dtype=np.float32)
        actions = wp.array(actions_np, dtype=wp.float32, device=env.device)

        graph = rl.capture_env_steps(env, actions, steps_per_graph=2, warmup_steps=1)
        wp.capture_launch(graph)
        wp.synchronize_device(env.device)

        self.assertFalse(env.solver.world.articulation_dvi_host)
        self.assertFalse(env.solver.world.articulation_dvi_replaces_joint_pgs)
        self.assertIsNone(env.solver.world.articulation_topology)
        self.assertEqual(env.solver.world.rigid_contact_max, env.world_count * g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env.contacts.rigid_contact_max, env.world_count * g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env.obs.shape, (2, rl.OBS_DIM_G1))
        self.assertEqual(env.rewards.shape, (2,))
        self.assertEqual(env.dones.shape, (2,))
        self.assertTrue(np.isfinite(env.obs.numpy()).all())
        self.assertTrue(np.isfinite(env.rewards.numpy()).all())
        self.assertTrue(np.isfinite(env.dones.numpy()).all())

        current_actions = env.current_actions.numpy()
        np.testing.assert_allclose(current_actions[:, :12], 1.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(current_actions[:, 12:], 0.0, rtol=0.0, atol=0.0)

    def test_graph_replay_advances_policy_steps(self) -> None:
        env = _g1_test_env(world_count=1)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)

        graph = rl.capture_env_steps(env, actions, steps_per_graph=2, warmup_steps=1)
        before = int(env.episode_steps.numpy()[0])
        for _ in range(3):
            wp.capture_launch(graph)
        wp.synchronize_device(env.device)
        after = int(env.episode_steps.numpy()[0])

        self.assertEqual(after - before, 6)
        self.assertTrue(np.isfinite(env.obs.numpy()).all())

    def test_randomize_commands_graph_capture(self) -> None:
        env = _g1_test_env(world_count=4)
        command_x_range = (-0.4, 0.8)
        command_y_range = (-0.2, 0.3)
        command_yaw_range = (-0.7, 0.6)

        with wp.ScopedCapture(device=env.device) as capture:
            env.randomize_commands(
                seed=17,
                command_x_range=command_x_range,
                command_y_range=command_y_range,
                command_yaw_range=command_yaw_range,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize_device(env.device)

        commands = env.command.numpy()
        self.assertTrue(np.all(commands[:, 0] >= command_x_range[0]))
        self.assertTrue(np.all(commands[:, 0] <= command_x_range[1]))
        self.assertTrue(np.all(commands[:, 1] >= command_y_range[0]))
        self.assertTrue(np.all(commands[:, 1] <= command_y_range[1]))
        self.assertTrue(np.all(commands[:, 2] >= command_yaw_range[0]))
        self.assertTrue(np.all(commands[:, 2] <= command_yaw_range[1]))

    def test_randomize_commands_seed_counter_advances_inside_graph(self) -> None:
        env = _g1_test_env(world_count=4)
        command_x_range = (-0.4, 0.8)
        command_y_range = (-0.2, 0.3)
        command_yaw_range = (-0.7, 0.6)
        seed_counter = make_seed_counter(17, device=env.device)

        with wp.ScopedCapture(device=env.device) as capture:
            env.randomize_commands_seed_counter(
                seed_counter=seed_counter,
                command_x_range=command_x_range,
                command_y_range=command_y_range,
                command_yaw_range=command_yaw_range,
            )

        wp.capture_launch(capture.graph)
        first = env.command.numpy().copy()
        wp.capture_launch(capture.graph)
        second = env.command.numpy().copy()

        self.assertFalse(np.allclose(first, second))
        np.testing.assert_array_equal(seed_counter.numpy(), np.array([19], dtype=np.int32))
        for commands in (first, second):
            self.assertTrue(np.all(commands[:, 0] >= command_x_range[0]))
            self.assertTrue(np.all(commands[:, 0] <= command_x_range[1]))
            self.assertTrue(np.all(commands[:, 1] >= command_y_range[0]))
            self.assertTrue(np.all(commands[:, 1] <= command_y_range[1]))
            self.assertTrue(np.all(commands[:, 2] >= command_yaw_range[0]))
            self.assertTrue(np.all(commands[:, 2] <= command_yaw_range[1]))

    def test_observe_clamps_extreme_state_to_finite_metrics(self) -> None:
        env = _g1_test_env(world_count=1)
        huge_qd = np.full(env.state_0.joint_qd.shape, 1.0e20, dtype=np.float32)
        env.state_0.joint_qd.assign(huge_qd)

        obs = env.observe()

        self.assertTrue(np.isfinite(obs.numpy()).all())
        self.assertTrue(np.isfinite(env.rewards.numpy()).all())
        self.assertTrue(np.isfinite(env.dones.numpy()).all())

    def test_train_save_load_evaluate_and_resume(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL training tests")
        env_config = rl.ConfigEnvG1PhoenX(
            world_count=1,
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            max_episode_steps=0,
            auto_reset=False,
        )
        ppo_config = rl.ConfigPPO(
            train_epochs=1,
            normalize_advantages=False,
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            entropy_coeff=0.0,
            reward_clip=1.0,
            max_grad_norm=0.3,
            mirror_loss_coeff=0.25,
            shared_value_network=True,
            manual_actor_backward=True,
            manual_critic_backward=True,
            manual_mlp_weight_grad_dtype="bfloat16",
            manual_mlp_forward_dtype="bfloat16",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_template = f"{tmpdir}/g1_{{iteration}}.npz"
            train_config = rl.ConfigTrainG1PPO(
                iterations=1,
                rollout_steps=1,
                hidden_layers=(8,),
                env_config=env_config,
                ppo_config=ppo_config,
                device=device,
                seed=23,
                log_interval=0,
                randomize_commands=True,
                checkpoint_path=checkpoint_template,
                checkpoint_interval=1,
                readback_diagnostics=True,
            )
            first = rl.train_g1_ppo(train_config)
            first_path = f"{tmpdir}/g1_1.npz"
            restored = rl.load_ppo_checkpoint(first_path, device=device)

            self.assertEqual(first.history[0].iteration, 0)
            self.assertTrue(math.isfinite(first.history[0].policy_loss))
            self.assertTrue(math.isfinite(first.history[0].value_loss))
            self.assertTrue(math.isfinite(first.history[0].mean_reward))
            self.assertTrue(math.isfinite(first.history[0].mean_tracking_perf))
            self.assertTrue(math.isfinite(first.history[0].mean_command_x))
            self.assertTrue(math.isfinite(first.history[0].mean_command_y))
            self.assertTrue(math.isfinite(first.history[0].mean_command_yaw))
            self.assertEqual(first.trainer.iteration, 1)
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.config.minibatch_size, 0)
            self.assertEqual(restored.config.replay_ratio, 0.0)
            self.assertEqual(restored.config.priority_alpha, 0.0)
            self.assertEqual(restored.config.priority_beta, 0.0)
            self.assertTrue(restored.config.manual_actor_backward)
            self.assertTrue(restored.config.manual_critic_backward)
            self.assertEqual(restored.config.manual_mlp_weight_grad_dtype, "bfloat16")
            self.assertEqual(restored.config.manual_mlp_forward_dtype, "bfloat16")
            self.assertEqual(restored.config.vtrace_rho_clip, 0.0)
            self.assertEqual(restored.config.vtrace_c_clip, 0.0)
            self.assertEqual(restored.config.reward_clip, 1.0)
            self.assertEqual(restored.config.max_grad_norm, 0.3)
            self.assertEqual(restored.config.mirror_loss_coeff, 0.25)
            self.assertTrue(restored.config.shared_value_network)
            self.assertIsNone(restored.critic)
            for before, after in zip(first.trainer.actor.parameters(), restored.actor.parameters(), strict=True):
                np.testing.assert_allclose(after.numpy(), before.numpy(), rtol=0.0, atol=0.0)

            eval_result = rl.evaluate_g1_ppo(
                restored,
                rl.ConfigEvaluateG1PPO(env_config=env_config, steps=1, device=device, deterministic=True),
            )
            self.assertTrue(math.isfinite(eval_result.stats.mean_reward))
            self.assertTrue(math.isfinite(eval_result.stats.mean_done))
            self.assertGreater(eval_result.stats.samples_per_second, 0.0)

            gate_result = rl.evaluate_g1_gate_ppo(
                restored,
                rl.ConfigEvaluateG1GatePPO(
                    env_config=env_config,
                    battery_commands=((0.0, 0.0, 0.0),),
                    seeds_per_command=1,
                    battery_steps=1,
                    diagnostic_steps=1,
                    diagnostic_world_count=1,
                    device=device,
                    deterministic=True,
                    min_battery_perf=2.0,
                ),
            )
            gate_stats = gate_result.stats
            self.assertEqual(gate_stats.battery_samples, 1)
            self.assertEqual(len(gate_stats.per_command), 1)
            self.assertFalse(gate_stats.pass_gate)
            self.assertTrue(math.isfinite(gate_stats.battery_perf))
            self.assertTrue(math.isfinite(gate_stats.action_jerk_rms))
            self.assertTrue(math.isfinite(gate_stats.ang_vel_xy_rms))
            self.assertTrue(math.isfinite(gate_stats.yaw_rate_rms))
            self.assertTrue(math.isfinite(gate_stats.leg_qvel_rms))
            self.assertGreater(gate_stats.samples_per_second, 0.0)

            resumed = rl.train_g1_ppo(
                rl.ConfigTrainG1PPO(
                    iterations=1,
                    rollout_steps=1,
                    hidden_layers=(8,),
                    env_config=env_config,
                    ppo_config=ppo_config,
                    device=device,
                    seed=23,
                    log_interval=0,
                    randomize_commands=False,
                    resume_checkpoint=first_path,
                    checkpoint_path=checkpoint_template,
                    checkpoint_interval=1,
                    readback_diagnostics=False,
                )
            )
            second_path = f"{tmpdir}/g1_2.npz"
            second_restored = rl.load_ppo_checkpoint(second_path, device=device)

            self.assertEqual(resumed.history[0].iteration, 1)
            self.assertEqual(resumed.trainer.iteration, 2)
            self.assertEqual(second_restored.iteration, 2)
            self.assertTrue(second_restored.config.shared_value_network)
            self.assertIsNone(second_restored.critic)

    def test_train_graph_leapfrog_save_and_resume(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 graph-leapfrog training tests")
        env_config = rl.ConfigEnvG1PhoenX(
            world_count=2,
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            max_episode_steps=0,
            auto_reset=False,
        )
        ppo_config = rl.ConfigPPO(
            train_epochs=1,
            minibatch_size=1,
            replay_ratio=1.0,
            priority_alpha=0.4,
            priority_beta=1.0,
            normalize_advantages=False,
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            entropy_coeff=0.0,
            reward_clip=1.0,
            max_grad_norm=0.3,
            mirror_loss_coeff=0.25,
            shared_value_network=True,
            manual_actor_backward=True,
            manual_critic_backward=True,
            manual_mlp_weight_grad_dtype="bfloat16",
            manual_mlp_forward_dtype="bfloat16",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_template = f"{tmpdir}/g1_graph_{{iteration}}.npz"
            result = rl.train_g1_ppo(
                rl.ConfigTrainG1PPO(
                    iterations=2,
                    rollout_steps=1,
                    hidden_layers=(8,),
                    env_config=env_config,
                    ppo_config=ppo_config,
                    device=device,
                    seed=29,
                    log_interval=0,
                    randomize_commands=True,
                    checkpoint_path=checkpoint_template,
                    checkpoint_interval=1,
                    readback_diagnostics=True,
                    execution_mode="graph_leapfrog",
                )
            )
            restored = rl.load_ppo_checkpoint(f"{tmpdir}/g1_graph_2.npz", device=device)

            self.assertEqual([stat.iteration for stat in result.history], [0, 1])
            self.assertEqual(result.trainer.iteration, 2)
            self.assertEqual(restored.iteration, 2)
            self.assertGreater(result.trainer.actor_optimizer.step_count, 0)
            self.assertTrue(math.isfinite(result.history[-1].mean_reward))
            self.assertTrue(math.isfinite(result.history[-1].policy_loss))
            self.assertTrue(math.isfinite(result.history[-1].value_loss))
            self.assertGreater(result.history[-1].samples_per_second, 0.0)
            for before, after in zip(result.trainer.actor.parameters(), restored.actor.parameters(), strict=True):
                np.testing.assert_allclose(after.numpy(), before.numpy(), rtol=0.0, atol=0.0)

            resumed = rl.train_g1_ppo(
                rl.ConfigTrainG1PPO(
                    iterations=1,
                    rollout_steps=1,
                    hidden_layers=(8,),
                    env_config=env_config,
                    ppo_config=ppo_config,
                    device=device,
                    seed=29,
                    log_interval=0,
                    randomize_commands=False,
                    resume_checkpoint=f"{tmpdir}/g1_graph_2.npz",
                    checkpoint_path=checkpoint_template,
                    checkpoint_interval=1,
                    readback_diagnostics=False,
                    execution_mode="graph_leapfrog",
                )
            )
            second_restored = rl.load_ppo_checkpoint(f"{tmpdir}/g1_graph_3.npz", device=device)

            self.assertEqual(resumed.history[0].iteration, 2)
            self.assertEqual(resumed.trainer.iteration, 3)
            self.assertEqual(second_restored.iteration, 3)
            self.assertEqual(second_restored.actor_optimizer.step_count, resumed.trainer.actor_optimizer.step_count)

    def test_train_to_gate_benchmark_smoke_graph_leapfrog(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 train-to-gate benchmark tests")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_template = f"{tmpdir}/gate_{{iteration}}.npz"
            args = argparse.Namespace(
                world_count=2,
                rollout_steps=1,
                target_samples=4,
                max_iterations=1,
                chunk_iterations=1,
                hidden_layers=(8,),
                train_epochs=1,
                mirror_loss_coeff=0.25,
                minibatch_size=1,
                replay_ratio=1.0,
                priority_alpha=0.4,
                priority_beta=1.0,
                no_manual_actor_backward=False,
                no_manual_critic_backward=False,
                manual_mlp_weight_grad_dtype="bfloat16",
                manual_mlp_forward_dtype="bfloat16",
                vtrace_rho_clip=3.0,
                vtrace_c_clip=3.0,
                reward_clip=1.0,
                max_grad_norm=0.3,
                command_x=0.8,
                command_y=0.0,
                command_yaw=0.0,
                command_x_range=(-0.5, 0.8),
                command_y_range=(-0.4, 0.4),
                command_yaw_range=(-1.0, 1.0),
                no_command_randomization=True,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                controlled_action_count=g1_recipe.CONTROLLED_ACTION_COUNT,
                parse_meshes=False,
                rigid_contact_max_per_world=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
                threads_per_world="auto",
                multi_world_scheduler="auto",
                prepare_refresh_stride="auto",
                execution_mode="graph_leapfrog",
                readback_diagnostics=False,
                checkpoint_path=checkpoint_template,
                device=device,
                seed=31,
                battery_steps=1,
                seeds_per_command=1,
                diagnostic_steps=1,
                diagnostic_world_count=1,
                stochastic_gate=False,
                gate_seed=41,
                max_battery_falls=1,
                min_battery_perf=2.0,
                max_action_jerk_rms=0.21,
                max_ang_vel_xy_rms=0.21,
                max_yaw_rate_rms=0.20,
                max_leg_qvel_rms=1.22,
                keep_going_after_pass=False,
                fail_on_miss=False,
                include_train_history=True,
            )
            result = benchmark_train_to_gate(args)
            checkpoint_path = Path(checkpoint_template.format(iteration=1))
            restored = rl.load_ppo_checkpoint(checkpoint_path, device=device)

            self.assertEqual(result["execution_mode"], "graph_leapfrog")
            self.assertEqual(result["completed_iterations"], 1)
            self.assertEqual(result["trained_samples"], 2)
            self.assertEqual(restored.iteration, 1)
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(len(result["gate_history"]), 1)
            self.assertEqual(len(result["train_history"]), 1)
            self.assertFalse(result["pass_gate"])
            self.assertGreater(result["train_env_samples_per_s"], 0.0)
            self.assertTrue(math.isfinite(result["gate_history"][0]["stats"]["battery_perf"]))


if __name__ == "__main__":
    wp.init()
    unittest.main()
