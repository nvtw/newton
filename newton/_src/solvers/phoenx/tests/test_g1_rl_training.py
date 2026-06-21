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
from newton._src.solvers.phoenx.model_adapter import build_adbs_init_arrays
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.env import collect_ppo_rollout_seed_counter, make_seed_counter
from newton._src.solvers.phoenx.rl_training.kernels import (
    compute_vtrace_returns_kernel,
    value_column_loss_grad_kernel,
    zero_scalar_kernel,
)
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


class _ConstantPPOEnv:
    def __init__(self, *, world_count: int, obs_dim: int, action_dim: int, device: wp.context.Device):
        self.world_count = int(world_count)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        obs = np.linspace(-0.3, 0.5, self.world_count * self.obs_dim, dtype=np.float32).reshape(
            self.world_count, self.obs_dim
        )
        self.obs = wp.array(obs, dtype=wp.float32, device=device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=device)
        self.step_successes = wp.zeros(self.world_count, dtype=wp.float32, device=device)

    def observe(self) -> wp.array2d[wp.float32]:
        return self.obs

    def step(
        self, actions: wp.array2d[wp.float32]
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array[wp.float32]]:
        return self.obs, self.rewards, self.dones


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


def _muon_reference_step(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    momentum: list[np.ndarray],
    *,
    lr: float,
    mu: float,
    eps: float,
    weight_decay: float,
    max_grad_norm: float,
    matrix_transpose: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    grad_sumsq = sum(float(np.sum(g * g)) for g in grads)
    clip = 1.0
    if max_grad_norm > 0.0:
        clip = min(float(max_grad_norm) / (math.sqrt(grad_sumsq) + 1.0e-6), 1.0)
    out_params: list[np.ndarray] = []
    out_momentum: list[np.ndarray] = []
    coeffs = (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    )
    for param, grad, mom in zip(params, grads, momentum, strict=True):
        clipped = clip * grad
        next_mom = mu * mom + clipped
        update = clipped + mu * next_mom
        if param.ndim >= 2:
            x = update.T if matrix_transpose else update
            x = x.reshape(x.shape[0], -1).astype(np.float32)
            x = x / max(float(np.linalg.norm(x)), eps)
            rows, cols = x.shape
            for a, b, c in coeffs:
                if rows > cols:
                    gram = x.T @ x
                    poly = c * (gram @ gram) + b * gram
                    x = a * x + x @ poly
                else:
                    gram = x @ x.T
                    poly = c * (gram @ gram) + b * gram
                    x = a * x + poly @ x
            update = (x * math.sqrt(max(1.0, rows / cols))).astype(np.float32)
            if matrix_transpose:
                update = update.T
            update = update.reshape(param.shape)
        out_params.append((param * (1.0 - lr * weight_decay) - lr * update).astype(np.float32))
        out_momentum.append(next_mom.astype(np.float32))
    return out_params, out_momentum


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

        g1_gpu_path = _NANOG1_ROOT / "ocean" / "g1gpu" / "g1_gpu.cu"
        if not g1_gpu_path.is_file():
            raise unittest.SkipTest(f"missing nanoG1 reference file: {g1_gpu_path}")
        g1_gpu = g1_gpu_path.read_text()

        def read_define_number(name: str) -> float:
            match = re.search(rf"#define {re.escape(name)}\s+\(?(-?[0-9.]+)f?\)?", g1_gpu)
            self.assertIsNotNone(match, f"missing define {name}")
            return float(match.group(1))

        self.assertAlmostEqual(g1_recipe.GAIT_STANCE_FRACTION, read_define_number("G1_V3_STANCE"))
        self.assertAlmostEqual(g1_recipe.W_GAIT_CONTACT, read_define_number("G1_V3_W_CONTACT"))
        self.assertAlmostEqual(g1_recipe.W_GAIT_SWING, read_define_number("G1_V3_W_SWING"))
        self.assertAlmostEqual(g1_recipe.W_GAIT_HIP, read_define_number("G1_V3_W_HIP"))
        self.assertAlmostEqual(g1_recipe.GAIT_FOOT_HEIGHT, read_define_number("G1_V3_FOOT_Z0"))
        self.assertAlmostEqual(g1_recipe.W_BASE_HEIGHT, read_define_number("G1_V3_W_BASE_HEIGHT"))
        self.assertAlmostEqual(g1_recipe.BASE_HEIGHT_TARGET, read_define_number("G1_V3_BASE_Z0"))

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
        self.assertAlmostEqual(g1_recipe.ACTOR_LR, float(recipe["train.learning_rate"]))
        self.assertEqual(g1_recipe.OPTIMIZER, "muon")
        self.assertAlmostEqual(g1_recipe.MUON_MOMENTUM, float(recipe["train.beta1"]))
        self.assertAlmostEqual(g1_recipe.OPTIMIZER_EPS, float(recipe["train.eps"]))
        self.assertAlmostEqual(g1_recipe.VALUE_LOSS_COEFF, float(recipe["train.vf_coef"]))
        self.assertAlmostEqual(g1_recipe.VALUE_CLIP_RANGE, float(recipe["train.vf_clip_coef"]))
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
        dof_frictionloss = _read_c_array(header, "hc_dof_frictionloss")

        env = rl.EnvG1PhoenX(g1_recipe.default_g1_env_config(world_count=1), device=device)
        np.testing.assert_allclose(env.default_joint_pos.numpy(), deploy.HOME, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.default_joint_pos.numpy(), key_qpos[7:], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.ctrl_lower.numpy(), deploy.CTRL_RANGE[:, 0], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_upper.numpy(), deploy.CTRL_RANGE[:, 1], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_lower.numpy(), ctrl_range[:, 0], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_upper.numpy(), ctrl_range[:, 1], rtol=0.0, atol=1.0e-5)

        expected_kp = deploy.KP.astype(np.float32)
        expected_kd = deploy.KD.astype(np.float32) + dof_damping[6 : 6 + rl.ACTION_DIM_G1].astype(np.float32)
        np.testing.assert_allclose(env.actuator_ke.numpy(), expected_kp, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.actuator_kd.numpy(), expected_kd, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            env.model.joint_friction.numpy()[6 : 6 + rl.ACTION_DIM_G1],
            dof_frictionloss[6 : 6 + rl.ACTION_DIM_G1],
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertEqual(env.config.joint_friction_model, "mujoco")
        hard_adbs = build_adbs_init_arrays(env.model, device=device, joint_friction_model="hard")
        mujoco_adbs = build_adbs_init_arrays(env.model, device=device, joint_friction_model="mujoco")
        np.testing.assert_allclose(
            hard_adbs.friction_slip_scale.numpy()[: rl.ACTION_DIM_G1],
            -np.ones(rl.ACTION_DIM_G1, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(float(np.min(mujoco_adbs.friction_slip_scale.numpy()[: rl.ACTION_DIM_G1])), 0.0)
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

    def test_pufferlib_continuous_actor_matches_unsquashed_warp_gaussian(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 PufferLib actor parity tests")
        models_py = _PUFFERLIB_ROOT / "pufferlib" / "models.py"
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not models_py.is_file() or not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference files under {_PUFFERLIB_ROOT}")
        models_text = models_py.read_text()
        self.assertIn("torch.distributions.Normal(mean, torch.exp(logstd))", models_text)
        self.assertIn("decoder_logstd = nn.Parameter(torch.zeros(1, num_atns))", models_text)
        self.assertEqual(g1_recipe.LOG_STD_INIT, 0.0)
        cuda_text = pufferlib_cu.read_text()
        self.assertIn("action = mean + std * noise", cuda_text)
        self.assertIn("normalized = (action - mean) / std", cuda_text)
        self.assertIn("*out_logp = -0.5f * normalized", cuda_text)

        actor = rl.GaussianActor(
            obs_dim=rl.OBS_DIM_G1,
            action_dim=rl.ACTION_DIM_G1,
            hidden_layers=(),
            activation="relu",
            squash=False,
            device=device,
            seed=5,
            log_std_init=g1_recipe.LOG_STD_INIT,
        )
        mean = np.linspace(-1.35, 1.35, rl.ACTION_DIM_G1, dtype=np.float32)
        log_std = np.linspace(-0.7, -0.2, rl.ACTION_DIM_G1, dtype=np.float32)
        actor.net.weights[0].assign(np.zeros((rl.OBS_DIM_G1, rl.ACTION_DIM_G1), dtype=np.float32))
        actor.net.biases[0].assign(mean)
        actor.log_std.assign(log_std)

        obs = wp.zeros((2, rl.OBS_DIM_G1), dtype=wp.float32, device=device)
        actor.reserve_reuse_buffers(2)
        with wp.ScopedCapture(device=device) as capture:
            actions, log_probs, _ = actor.sample_reuse(obs, seed=19, deterministic=True)
        wp.capture_launch(capture.graph)

        expected_actions = np.broadcast_to(mean, (2, rl.ACTION_DIM_G1))
        np.testing.assert_allclose(actions.numpy(), expected_actions, rtol=0.0, atol=0.0)
        self.assertGreater(float(np.max(expected_actions)), 1.0)
        expected_log_prob = np.full(
            2,
            np.sum(-0.5 * np.log(2.0 * np.pi) - log_std),
            dtype=np.float32,
        )
        np.testing.assert_allclose(log_probs.numpy(), expected_log_prob, rtol=1.0e-6, atol=1.0e-6)

        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=2,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=device,
        )
        with wp.ScopedCapture(device=device) as step_capture:
            env.step(actions)
        wp.capture_launch(step_capture.graph)

        expected_clipped = np.clip(expected_actions, -1.0, 1.0)
        expected_clipped[:, g1_recipe.CONTROLLED_ACTION_COUNT :] = 0.0
        np.testing.assert_allclose(env.current_actions.numpy(), expected_clipped, rtol=0.0, atol=0.0)

    def test_pufferlib_muon_optimizer_matches_numpy_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 Muon optimizer parity tests")
        muon_py = _PUFFERLIB_ROOT / "pufferlib" / "muon.py"
        muon_cu = _PUFFERLIB_ROOT / "src" / "muon.cu"
        if not muon_py.is_file() or not muon_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib Muon reference files under {_PUFFERLIB_ROOT}")
        self.assertIn("zeropower_via_newtonschulz5", muon_py.read_text())
        self.assertIn("muon_nesterov", muon_cu.read_text())

        param_np = [
            np.asarray([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]], dtype=np.float32),
            np.asarray([[0.5, -0.2, 0.1], [-0.6, 0.3, 0.8]], dtype=np.float32),
            np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
        ]
        grad_np = [
            np.asarray([[0.3, -0.7], [0.2, 0.4], [-0.5, 0.6]], dtype=np.float32),
            np.asarray([[-0.1, 0.2, 0.3], [0.4, -0.6, 0.5]], dtype=np.float32),
            np.asarray([0.7, -0.4, 0.2], dtype=np.float32),
        ]
        params = [wp.array(value, dtype=wp.float32, device=device, requires_grad=True) for value in param_np]
        for param, grad in zip(params, grad_np, strict=True):
            param.grad.assign(grad)
        optimizer = rl.Muon(
            params,
            lr=0.02,
            momentum=0.9,
            eps=1.0e-12,
            weight_decay=0.01,
            max_grad_norm=1.2,
        )
        with wp.ScopedCapture(device=device) as capture:
            optimizer.step()
        wp.capture_launch(capture.graph)

        expected_params, expected_momentum = _muon_reference_step(
            param_np,
            grad_np,
            [np.zeros_like(value) for value in param_np],
            lr=0.02,
            mu=0.9,
            eps=1.0e-12,
            weight_decay=0.01,
            max_grad_norm=1.2,
        )
        for actual, expected in zip(params, expected_params, strict=True):
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-5, atol=2.0e-5)
            np.testing.assert_allclose(actual.grad.numpy(), np.zeros_like(expected), rtol=0.0, atol=0.0)
        for actual, expected in zip(optimizer.m, expected_momentum, strict=True):
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-6, atol=2.0e-6)
        self.assertFalse(optimizer.matrix_transpose)
        self.assertEqual(optimizer.step_count, 1)

    def test_pufferlib_muon_matches_warp_mlp_transposed_weight_layout(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 Muon WarpMLP layout parity tests")
        muon_py = _PUFFERLIB_ROOT / "pufferlib" / "muon.py"
        puffernet = _PUFFERLIB_ROOT / "src" / "puffernet.h"
        if not muon_py.is_file() or not puffernet.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference files under {_PUFFERLIB_ROOT}")
        self.assertIn("grad.view(grad.shape[0], -1)", muon_py.read_text())
        self.assertIn("weights[o*input_dim + i]", puffernet.read_text())

        param_np = np.linspace(-0.3, 0.4, 7 * 11, dtype=np.float32).reshape(7, 11)
        grad_np = np.linspace(0.2, -0.5, 7 * 11, dtype=np.float32).reshape(7, 11)
        param = wp.array(param_np, dtype=wp.float32, device=device, requires_grad=True)
        param.grad.assign(grad_np)
        optimizer = rl.Muon(
            [param],
            lr=0.02,
            momentum=0.9,
            eps=1.0e-12,
            weight_decay=0.0,
            max_grad_norm=0.3,
            matrix_transpose=True,
        )
        with wp.ScopedCapture(device=device) as capture:
            optimizer.step()
        wp.capture_launch(capture.graph)

        expected_params, expected_momentum = _muon_reference_step(
            [param_np],
            [grad_np],
            [np.zeros_like(param_np)],
            lr=0.02,
            mu=0.9,
            eps=1.0e-12,
            weight_decay=0.0,
            max_grad_norm=0.3,
            matrix_transpose=True,
        )
        np.testing.assert_allclose(param.numpy(), expected_params[0], rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(param.grad.numpy(), np.zeros_like(param_np), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(optimizer.m[0].numpy(), expected_momentum[0], rtol=2.0e-6, atol=2.0e-6)
        self.assertTrue(optimizer.matrix_transpose)
        self.assertEqual(optimizer.step_count, 1)

    def test_pufferlib_value_clip_formula_matches_warp_kernel(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 PufferLib value loss parity tests")
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {pufferlib_cu}")
        cuda_text = pufferlib_cu.read_text()
        self.assertIn("v_clipped = val + fmaxf(-a.vf_clip_coef", cuda_text)
        self.assertIn("v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped)", cuda_text)
        self.assertIn("a.grad_values_pred[nt] = dL * a.vf_coef * d_val_pred", cuda_text)

        values_np = np.asarray(
            [
                [0.0, 15.0, 0.0],
                [0.0, 35.0, 0.0],
                [0.0, -45.0, 0.0],
                [0.0, -5.0, 0.0],
            ],
            dtype=np.float32,
        )
        old_values_np = np.asarray([0.0, 10.0, 0.0, -10.0], dtype=np.float32)
        returns_np = np.asarray([5.0, 0.0, -10.0, -8.0], dtype=np.float32)
        value_col = 1
        coeff = np.float32(g1_recipe.VALUE_LOSS_COEFF)
        clip_range = np.float32(g1_recipe.VALUE_CLIP_RANGE)

        values = wp.array(values_np, dtype=wp.float32, device=device)
        old_values = wp.array(old_values_np, dtype=wp.float32, device=device)
        returns = wp.array(returns_np, dtype=wp.float32, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device)
        grad = wp.zeros_like(values)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(zero_scalar_kernel, dim=1, outputs=[loss], device=device)
            wp.launch(
                value_column_loss_grad_kernel,
                dim=values_np.shape[0],
                inputs=[
                    values,
                    value_col,
                    old_values,
                    returns,
                    float(coeff),
                    float(clip_range),
                    values_np.shape[0],
                ],
                outputs=[loss, grad],
                device=device,
            )
        wp.capture_launch(capture.graph)

        pred = values_np[:, value_col]
        value_error = pred - old_values_np
        clipped = old_values_np + np.clip(value_error, -clip_range, clip_range)
        loss_unclipped = (pred - returns_np) ** 2
        loss_clipped = (clipped - returns_np) ** 2
        expected_loss = float(np.mean(0.5 * coeff * np.maximum(loss_unclipped, loss_clipped)))
        expected_grad_col = coeff * (pred - returns_np) / values_np.shape[0]
        use_clipped = loss_clipped > loss_unclipped
        outside_clip = np.logical_or(value_error < -clip_range, value_error > clip_range)
        expected_grad_col = np.where(use_clipped & outside_clip, 0.0, expected_grad_col)
        expected_grad_col = np.where(
            use_clipped & ~outside_clip, coeff * (clipped - returns_np) / values_np.shape[0], expected_grad_col
        )
        expected_grad = np.zeros_like(values_np)
        expected_grad[:, value_col] = expected_grad_col

        np.testing.assert_allclose(loss.numpy()[0], expected_loss, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(grad.numpy(), expected_grad, rtol=1.0e-6, atol=1.0e-6)

    def test_pufferlib_vtrace_advantage_matches_shifted_warp_layout(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 V-trace parity tests")
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {pufferlib_cu}")
        text = pufferlib_cu.read_text()
        self.assertIn("float r_nxt = to_float(rewards[t_next])", text)
        self.assertIn("float rho_t = fminf(imp, rho_clip)", text)
        self.assertIn("lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal", text)

        num_steps = 4
        num_envs = 2
        gamma = np.float32(0.97)
        gae_lambda = np.float32(0.9)
        rho_clip = np.float32(1.4)
        c_clip = np.float32(1.1)
        reward_clip = np.float32(1.0)
        rewards_np = np.asarray([[1.2, -0.4], [0.3, 2.5], [-1.7, 0.2], [0.5, -0.8]], dtype=np.float32)
        dones_np = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        values_np = np.asarray([[0.1, -0.2], [0.4, 0.3], [-0.1, 0.7], [0.5, -0.6], [0.2, 0.9]], dtype=np.float32)
        ratios_np = np.asarray([[0.8, 1.6], [1.2, 0.7], [2.0, 1.0], [0.5, 1.3]], dtype=np.float32)

        rewards = wp.array(rewards_np.reshape(-1), dtype=wp.float32, device=device)
        dones = wp.array(dones_np.reshape(-1), dtype=wp.float32, device=device)
        values = wp.array(values_np.reshape(-1), dtype=wp.float32, device=device)
        ratios = wp.array(ratios_np.reshape(-1), dtype=wp.float32, device=device)
        advantages = wp.zeros(num_steps * num_envs, dtype=wp.float32, device=device)
        returns = wp.zeros(num_steps * num_envs, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                compute_vtrace_returns_kernel,
                dim=num_envs,
                inputs=[
                    rewards,
                    dones,
                    values,
                    ratios,
                    num_steps,
                    num_envs,
                    float(gamma),
                    float(gae_lambda),
                    float(rho_clip),
                    float(c_clip),
                    float(reward_clip),
                ],
                outputs=[advantages, returns],
                device=device,
            )
        wp.capture_launch(capture.graph)

        puffer_rewards = np.zeros((num_steps + 1, num_envs), dtype=np.float32)
        puffer_dones = np.zeros((num_steps + 1, num_envs), dtype=np.float32)
        puffer_rewards[1:] = np.clip(rewards_np, -reward_clip, reward_clip)
        puffer_dones[1:] = dones_np
        expected_adv = np.zeros((num_steps, num_envs), dtype=np.float32)
        for env_id in range(num_envs):
            trace = np.float32(0.0)
            for t in range(num_steps - 1, -1, -1):
                next_nonterminal = np.float32(1.0) - puffer_dones[t + 1, env_id]
                rho = min(ratios_np[t, env_id], rho_clip)
                c = min(ratios_np[t, env_id], c_clip)
                delta = rho * (
                    puffer_rewards[t + 1, env_id]
                    + gamma * values_np[t + 1, env_id] * next_nonterminal
                    - values_np[t, env_id]
                )
                trace = delta + gamma * gae_lambda * c * trace * next_nonterminal
                expected_adv[t, env_id] = trace
        expected_returns = expected_adv + values_np[:-1]

        np.testing.assert_allclose(
            advantages.numpy().reshape(num_steps, num_envs), expected_adv, rtol=1.0e-6, atol=1.0e-6
        )
        np.testing.assert_allclose(
            returns.numpy().reshape(num_steps, num_envs), expected_returns, rtol=1.0e-6, atol=1.0e-6
        )

    def test_collect_rollout_resets_recurrent_state_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX recurrent rollout reset tests")
        env = _ConstantPPOEnv(world_count=2, obs_dim=3, action_dim=1, device=device)
        buffer = rl.BufferRollout(
            num_steps=1, num_envs=env.world_count, obs_dim=env.obs_dim, action_dim=1, device=device
        )
        trainer = rl.TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=(2,),
            config=rl.ConfigPPO(
                gamma=0.9,
                gae_lambda=0.8,
                shared_value_network=True,
                policy_network="puffer_mingru",
                manual_actor_backward=True,
                manual_critic_backward=True,
            ),
            device=device,
            seed=3,
            squash_actions=False,
            log_std_init=-5.0,
        )
        trainer.actor.net.encoder_weight.assign(np.asarray([[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]], dtype=np.float32))
        trainer.actor.net.recurrent_weights[0].assign(
            np.asarray([[0.2, -0.1, 0.5, -0.4, 0.6, -0.3], [-0.3, 0.4, -0.2, 0.7, -0.5, 0.2]], dtype=np.float32)
        )
        trainer.actor.net.decoder_weight.assign(np.asarray([[0.6, 0.2], [-0.4, 0.3]], dtype=np.float32))
        trainer.reserve_update_buffers(buffer)

        trainer.reset_rollout_state()
        expected_actions, _expected_log_probs, _expected_values = trainer.act_reuse(env.obs, seed=11)
        expected = expected_actions.numpy().copy()
        stale_actions, _stale_log_probs, _stale_values = trainer.act_reuse(env.obs, seed=11)
        stale = stale_actions.numpy().copy()
        self.assertFalse(np.allclose(stale, expected, rtol=1.0e-5, atol=1.0e-5))

        seed_counter = make_seed_counter(11, device=device)
        with wp.ScopedCapture(device=device) as capture:
            collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=seed_counter)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(buffer.actions.numpy(), expected, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_array_equal(seed_counter.numpy(), np.array([12], dtype=np.int32))

    def test_puffer_mingru_forward_and_reset_match_numpy_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 MinGRU network tests")
        net = rl.PufferMinGRUNet(input_dim=3, hidden_size=2, output_dim=2, num_layers=1, device=device, seed=3)
        net.encoder_weight.assign(np.asarray([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.7]], dtype=np.float32))
        net.recurrent_weights[0].assign(
            np.asarray(
                [
                    [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
                    [-0.3, 0.2, -0.1, 0.6, -0.5, 0.4],
                ],
                dtype=np.float32,
            )
        )
        net.decoder_weight.assign(np.asarray([[0.3, -0.2], [-0.4, 0.5]], dtype=np.float32))
        obs_np = np.asarray([[0.2, -0.4, 0.6], [0.1, 0.3, -0.2]], dtype=np.float32)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        dones = wp.array(np.asarray([1.0, 0.0], dtype=np.float32), dtype=wp.float32, device=device)
        net.reserve_buffers(2)
        out0_snapshot = wp.empty((2, 2), dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            out0 = net.forward_reuse(obs)
            wp.copy(out0_snapshot, out0)
            net.reset_state(dones)
            out1 = net.forward_reuse(obs)
        wp.capture_launch(capture.graph)

        def step_numpy(state: np.ndarray, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            h = observations @ net.encoder_weight.numpy()
            combined = h @ net.recurrent_weights[0].numpy()
            hidden = combined[:, 0:2]
            gate = combined[:, 2:4]
            proj = combined[:, 4:6]
            candidate = np.where(hidden >= 0.0, hidden + 0.5, 1.0 / (1.0 + np.exp(-hidden)))
            gate_sig = 1.0 / (1.0 + np.exp(-gate))
            recurrent = state + gate_sig * (candidate - state)
            proj_sig = 1.0 / (1.0 + np.exp(-proj))
            y = proj_sig * recurrent + (1.0 - proj_sig) * h
            return y @ net.decoder_weight.numpy(), recurrent.astype(np.float32)

        zero_state = np.zeros((2, 2), dtype=np.float32)
        expected0, state0 = step_numpy(zero_state, obs_np)
        state0[0] = 0.0
        expected1, _state1 = step_numpy(state0, obs_np)
        np.testing.assert_allclose(out0_snapshot.numpy(), expected0, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(out1.numpy(), expected1, rtol=1.0e-6, atol=1.0e-6)

    def test_puffer_mingru_backward_matches_finite_difference_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX MinGRU backward tests")
        net = rl.PufferMinGRUNet(input_dim=2, hidden_size=2, output_dim=1, num_layers=1, device=device, seed=5)
        encoder = np.asarray([[0.2, -0.3], [0.4, 0.1]], dtype=np.float32)
        recurrent = np.asarray(
            [[0.1, -0.2, 0.3, 0.15, -0.25, 0.35], [-0.4, 0.2, -0.1, 0.45, 0.05, -0.15]],
            dtype=np.float32,
        )
        decoder = np.asarray([[0.25], [-0.35]], dtype=np.float32)
        obs_np = np.asarray([[0.2, -0.1], [0.4, 0.3]], dtype=np.float32)
        upstream_np = np.asarray([[0.7], [-0.2]], dtype=np.float32)
        net.encoder_weight.assign(encoder)
        net.recurrent_weights[0].assign(recurrent)
        net.decoder_weight.assign(decoder)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        upstream = wp.array(upstream_np, dtype=wp.float32, device=device)
        net.set_sequence_shape(num_steps=2, num_envs=1)
        net.reserve_buffers(2)

        with wp.ScopedCapture(device=device) as capture:
            net.forward_manual(obs)
            net.backward_manual(upstream)
        wp.capture_launch(capture.graph)

        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x))

        def loss(enc: np.ndarray, rec: np.ndarray, dec: np.ndarray) -> float:
            h = obs_np @ enc
            state = np.zeros((1, 2), dtype=np.float32)
            outputs = []
            for row in range(2):
                combined = h[row : row + 1] @ rec
                hidden = combined[:, 0:2]
                gate = sigmoid(combined[:, 2:4])
                proj = sigmoid(combined[:, 4:6])
                candidate = np.where(hidden >= 0.0, hidden + 0.5, sigmoid(hidden))
                state = state + gate * (candidate - state)
                outputs.append(proj * state + (1.0 - proj) * h[row : row + 1])
            out = np.concatenate(outputs, axis=0) @ dec
            return float(np.sum(out * upstream_np))

        def finite_difference(param: np.ndarray, fn) -> np.ndarray:
            grad = np.zeros_like(param)
            eps = 1.0e-3
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            for value in it:
                index = it.multi_index
                old = float(value)
                param[index] = old + eps
                plus = fn()
                param[index] = old - eps
                minus = fn()
                param[index] = old
                grad[index] = (plus - minus) / (2.0 * eps)
            return grad

        encoder_fd = encoder.copy()
        recurrent_fd = recurrent.copy()
        decoder_fd = decoder.copy()
        enc_grad = finite_difference(encoder_fd, lambda: loss(encoder_fd, recurrent, decoder))
        rec_grad = finite_difference(recurrent_fd, lambda: loss(encoder, recurrent_fd, decoder))
        dec_grad = finite_difference(decoder_fd, lambda: loss(encoder, recurrent, decoder_fd))

        np.testing.assert_allclose(net.encoder_weight.grad.numpy(), enc_grad, rtol=2.0e-2, atol=2.0e-2)
        np.testing.assert_allclose(net.recurrent_weights[0].grad.numpy(), rec_grad, rtol=2.0e-2, atol=2.0e-2)
        np.testing.assert_allclose(net.decoder_weight.grad.numpy(), dec_grad, rtol=2.0e-2, atol=2.0e-2)

    def test_puffer_mingru_ppo_train_save_load_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX recurrent PPO tests")
        buffer = rl.BufferRollout(num_steps=2, num_envs=2, obs_dim=4, action_dim=2, device=device)
        obs = np.linspace(-0.5, 0.6, buffer.num_samples * buffer.obs_dim, dtype=np.float32).reshape(
            buffer.num_samples, buffer.obs_dim
        )
        buffer.obs.assign(obs)
        buffer.actions.assign(
            np.linspace(-0.2, 0.2, buffer.num_samples * buffer.action_dim, dtype=np.float32).reshape(
                buffer.num_samples, buffer.action_dim
            )
        )
        buffer.old_log_probs.assign(np.zeros(buffer.num_samples, dtype=np.float32))
        buffer.advantages.assign(np.asarray([0.4, -0.1, 0.2, -0.3], dtype=np.float32))
        buffer.returns.assign(np.asarray([0.3, -0.2, 0.1, -0.4], dtype=np.float32))
        buffer.old_values.assign(np.zeros((buffer.num_steps + 1) * buffer.num_envs, dtype=np.float32))

        config = rl.ConfigPPO(
            train_epochs=1,
            normalize_advantages=False,
            actor_lr=1.0e-3,
            entropy_coeff=0.0,
            value_loss_coeff=0.5,
            value_clip_range=1.0,
            max_grad_norm=0.3,
            shared_value_network=True,
            policy_network="puffer_mingru",
            manual_actor_backward=True,
            manual_critic_backward=True,
            mirror_loss_coeff=0.0,
            optimizer="adam",
        )
        trainer = rl.TrainerPPO(
            obs_dim=buffer.obs_dim,
            action_dim=buffer.action_dim,
            hidden_layers=(8, 8),
            config=config,
            device=device,
            seed=7,
            squash_actions=False,
        )
        trainer.reserve_update_buffers(buffer)
        seed_counter = make_seed_counter(17, device=device)

        with wp.ScopedCapture(device=device) as capture:
            trainer.update_seed_counter(buffer, seed_counter=seed_counter, read_stats=False)
        before = [param.numpy().copy() for param in trainer.actor.parameters()]
        wp.capture_launch(capture.graph)
        after = [param.numpy().copy() for param in trainer.actor.parameters()]
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(after, before, strict=True)))
        self.assertEqual(trainer.actor.net.network_type, "puffer_mingru")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/recurrent.npz"
            trainer.save_checkpoint(path, iteration=1)
            restored = rl.load_ppo_checkpoint(path, device=device)
            self.assertEqual(restored.config.policy_network, "puffer_mingru")
            self.assertEqual(restored.actor.net.network_type, "puffer_mingru")
            for expected, actual in zip(trainer.actor.parameters(), restored.actor.parameters(), strict=True):
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.0, atol=0.0)

    def test_shared_value_mlp_forward_matches_numpy_for_nanog1_shape(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL network parity tests")
        puffernet = _PUFFERLIB_ROOT / "src" / "puffernet.h"
        nanog1_policy = _NANOG1_ROOT / "deploy" / "nanog1_policy.c"
        if not puffernet.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {puffernet}")
        if not nanog1_policy.is_file():
            raise unittest.SkipTest(f"missing nanoG1 policy reference file: {nanog1_policy}")
        text = puffernet.read_text()
        self.assertIn("decoder output", text)
        self.assertIn("_gaussian_mean", text)
        self.assertIn("MinGRU* mingru", text)
        policy_text = nanog1_policy.read_text()
        self.assertIn("NANOG1_OBS 98", policy_text)
        self.assertIn("NANOG1_NU  29", policy_text)
        self.assertIn("make_puffernet(w, 1, NANOG1_OBS, 128, 3", policy_text)

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

        expected = obs_np
        for layer, (weight, bias) in enumerate(zip(trainer.actor.net.weights, trainer.actor.net.biases, strict=True)):
            expected = expected @ weight.numpy() + bias.numpy()
            if layer < len(trainer.actor.net.weights) - 1:
                expected = np.maximum(expected, 0.0)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2.0e-5, atol=2.0e-5)

    def test_puffernet_linear_layer_matches_warp_mlp_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX PufferNet linear parity tests")
        puffernet = _PUFFERLIB_ROOT / "src" / "puffernet.h"
        if not puffernet.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {puffernet}")
        text = puffernet.read_text()
        self.assertIn("output[b*output_dim + o] = sum + bias[o];", text)
        self.assertIn("weights[o*input_dim + i]", text)

        batch_size = 5
        input_dim = 7
        output_dim = 11
        obs_np = np.linspace(-0.6, 0.7, batch_size * input_dim, dtype=np.float32).reshape(batch_size, input_dim)
        puffer_weight_np = np.linspace(-0.4, 0.5, output_dim * input_dim, dtype=np.float32).reshape(
            output_dim, input_dim
        )
        bias_np = np.linspace(-0.2, 0.3, output_dim, dtype=np.float32)

        net = rl.WarpMLP((input_dim, output_dim), activation="linear", device=device, seed=7)
        net.weights[0].assign(puffer_weight_np.T.copy())
        net.biases[0].assign(bias_np)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        out = net.forward_reuse(obs)
        with wp.ScopedCapture(device=device) as capture:
            net.forward_reuse(obs)
        wp.capture_launch(capture.graph)

        expected = obs_np @ puffer_weight_np.T + bias_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1.0e-6, atol=1.0e-6)

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
        self.assertEqual(env_config.gait_stance_fraction, g1_recipe.GAIT_STANCE_FRACTION)
        self.assertEqual(env_config.w_gait_contact, g1_recipe.W_GAIT_CONTACT)
        self.assertEqual(env_config.w_gait_swing, g1_recipe.W_GAIT_SWING)
        self.assertEqual(env_config.w_gait_hip, g1_recipe.W_GAIT_HIP)
        self.assertEqual(env_config.gait_foot_height, g1_recipe.GAIT_FOOT_HEIGHT)
        self.assertEqual(env_config.w_base_height, g1_recipe.W_BASE_HEIGHT)
        self.assertEqual(env_config.base_height_target, g1_recipe.BASE_HEIGHT_TARGET)
        self.assertEqual(env_config.rigid_contact_max_per_world, g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env_config.contact_geometry, g1_recipe.CONTACT_GEOMETRY)
        self.assertEqual(env_config.contact_geometry, "nanog1_foot_boxes")
        self.assertEqual(env_config.threads_per_world, g1_recipe.THREADS_PER_WORLD)
        self.assertEqual(env_config.multi_world_scheduler, g1_recipe.MULTI_WORLD_SCHEDULER)
        self.assertEqual(env_config.prepare_refresh_stride, g1_recipe.PREPARE_REFRESH_STRIDE)
        expected_leg_kd = np.array([4.0, 4.0, 4.0, 6.0, 3.0, 2.2] * 2, dtype=np.float32)
        np.testing.assert_allclose(env.model.joint_target_kd.numpy()[6:18], expected_leg_kd, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.model.joint_friction.numpy()[6:35], 0.1, rtol=0.0, atol=1.0e-6)
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
        self.assertEqual(train_config.squash_actions, g1_recipe.SQUASH_ACTIONS)
        self.assertFalse(train_config.squash_actions)
        self.assertGreater(ppo_config.mirror_loss_coeff, 0.0)
        self.assertEqual(ppo_config.mirror_loss_coeff, g1_recipe.MIRROR_LOSS_COEFF)
        self.assertTrue(ppo_config.shared_value_network)
        self.assertEqual(ppo_config.shared_value_network, g1_recipe.SHARED_VALUE_NETWORK)
        self.assertEqual(ppo_config.policy_network, g1_recipe.POLICY_NETWORK)
        self.assertEqual(ppo_config.policy_network, "puffer_mingru")
        self.assertEqual(ppo_config.minibatch_size, g1_recipe.MINIBATCH_SIZE)
        self.assertEqual(ppo_config.value_loss_coeff, g1_recipe.VALUE_LOSS_COEFF)
        self.assertEqual(ppo_config.value_clip_range, g1_recipe.VALUE_CLIP_RANGE)
        self.assertEqual(ppo_config.optimizer, g1_recipe.OPTIMIZER)
        self.assertEqual(ppo_config.optimizer_eps, g1_recipe.OPTIMIZER_EPS)
        self.assertEqual(ppo_config.optimizer_weight_decay, g1_recipe.OPTIMIZER_WEIGHT_DECAY)
        self.assertEqual(ppo_config.muon_momentum, g1_recipe.MUON_MOMENTUM)
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

    def test_reset_done_samples_commands_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 command reset tests")
        config = rl.ConfigEnvG1PhoenX(
            world_count=2,
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            max_episode_steps=0,
            auto_reset=False,
            randomize_commands_on_reset=True,
            command_x_range=(0.25, 0.25),
            command_y_range=(-0.15, -0.15),
            command_yaw_range=(0.4, 0.4),
            command_zero_probability=0.0,
        )
        env = rl.EnvG1PhoenX(config, device=device)
        env.set_commands(np.zeros((env.world_count, 3), dtype=np.float32))
        env.dones.assign(np.array([1.0, 0.0], dtype=np.float32))
        reset_seed_counter = make_seed_counter(11, device=device)
        command_seed_counter = make_seed_counter(21, device=device)
        env.use_command_seed_counter(command_seed_counter)

        with wp.ScopedCapture(device=device) as capture:
            env.reset_done_seed_counter(reset_seed_counter)

        wp.capture_launch(capture.graph)
        commands = env.command.numpy()
        np.testing.assert_allclose(commands[0], np.array([0.25, -0.15, 0.4], dtype=np.float32))
        np.testing.assert_allclose(commands[1], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(reset_seed_counter.numpy(), np.array([12], dtype=np.int32))
        np.testing.assert_array_equal(command_seed_counter.numpy(), np.array([22], dtype=np.int32))

    def test_periodic_command_resample_updates_next_obs_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 periodic command tests")
        config = rl.ConfigEnvG1PhoenX(
            world_count=1,
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            max_episode_steps=0,
            auto_reset=True,
            randomize_commands_on_reset=True,
            command_x_range=(0.3, 0.3),
            command_y_range=(0.0, 0.0),
            command_yaw_range=(-0.2, -0.2),
            command_zero_probability=0.0,
            command_resample_steps=2,
        )
        env = rl.EnvG1PhoenX(config, device=device)
        env.set_command((0.0, 0.0, 0.0))
        env.episode_steps.assign(np.array([1], dtype=np.int32))
        env.use_reset_seed_counter(make_seed_counter(31, device=device))
        env.use_command_seed_counter(make_seed_counter(41, device=device))
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            env.step(actions)

        wp.capture_launch(capture.graph)
        expected = np.array([0.3, 0.0, -0.2], dtype=np.float32)
        np.testing.assert_allclose(env.command.numpy()[0], expected)
        np.testing.assert_allclose(env.obs.numpy()[0, 6:9], expected)

    def test_reset_done_seed_counter_advances_inside_graph(self) -> None:
        env = _g1_test_env(world_count=2)
        seed_counter = make_seed_counter(21, device=env.device)
        env.dones.assign(np.ones(env.world_count, dtype=np.float32))

        with wp.ScopedCapture(device=env.device) as capture:
            env.reset_done_seed_counter(seed_counter=seed_counter)

        wp.capture_launch(capture.graph)
        first = env.state_0.joint_q.numpy().copy()
        wp.capture_launch(capture.graph)
        second = env.state_0.joint_q.numpy().copy()

        first_joints = first.reshape(env.world_count, env.coord_stride)[:, 7 : 7 + env.action_dim]
        second_joints = second.reshape(env.world_count, env.coord_stride)[:, 7 : 7 + env.action_dim]
        self.assertFalse(np.allclose(first_joints, second_joints))
        np.testing.assert_array_equal(seed_counter.numpy(), np.array([23], dtype=np.int32))

    def test_ppo_lr_anneal_updates_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX PPO LR anneal test")
        buffer = rl.BufferRollout(num_steps=2, num_envs=2, obs_dim=4, action_dim=2, device=device)
        obs = np.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0, 1.1],
                [1.2, 1.3, 1.4, 1.5],
            ],
            dtype=np.float32,
        )
        buffer.obs.assign(obs)
        buffer.actions.assign(np.zeros((buffer.num_samples, buffer.action_dim), dtype=np.float32))
        buffer.old_log_probs.assign(np.zeros(buffer.num_samples, dtype=np.float32))
        buffer.advantages.assign(np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32))
        buffer.returns.assign(np.array([0.2, -0.2, 0.1, -0.1], dtype=np.float32))

        config = rl.ConfigPPO(
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            anneal_lr=True,
            lr_anneal_timesteps=buffer.num_samples * 2,
            min_lr_ratio=0.0,
            train_epochs=1,
            manual_actor_backward=True,
            manual_critic_backward=True,
            normalize_advantages=True,
            value_loss_coeff=0.5,
            max_grad_norm=0.3,
        )
        trainer = rl.TrainerPPO(
            obs_dim=buffer.obs_dim,
            action_dim=buffer.action_dim,
            hidden_layers=(8,),
            config=config,
            device=device,
            seed=5,
            squash_actions=False,
            activation="relu",
        )
        trainer.reserve_update_buffers(buffer)
        seed_counter = make_seed_counter(123, device=device)

        with wp.ScopedCapture(device=device) as capture:
            trainer.update_seed_counter(buffer, seed_counter=seed_counter, read_stats=False)

        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(trainer.actor_optimizer.lr_scale.numpy(), np.array([1.0], dtype=np.float32))
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(trainer.actor_optimizer.lr_scale.numpy(), np.array([0.5], dtype=np.float32))
        np.testing.assert_array_equal(trainer._iteration_counter.numpy(), np.array([2], dtype=np.int32))

    def test_reset_done_clears_contact_warmstart_inside_graph(self) -> None:
        env = _g1_test_env(world_count=2)
        shape_world = env.model.shape_world.numpy()
        world0_shape = int(np.flatnonzero(shape_world == 0)[0])
        world1_shape = int(np.flatnonzero(shape_world == 1)[0])
        ground_shape = int(np.flatnonzero(shape_world < 0)[0])

        shape0 = np.full(env.contacts.rigid_contact_max, ground_shape, dtype=np.int32)
        shape1 = np.full(env.contacts.rigid_contact_max, ground_shape, dtype=np.int32)
        shape0[:2] = np.array([world0_shape, world1_shape], dtype=np.int32)
        env.contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
        env.contacts.rigid_contact_shape0.assign(shape0)
        env.contacts.rigid_contact_shape1.assign(shape1)

        world = env.solver.world
        self.assertIsNotNone(world._cid_of_contact_cur)
        self.assertIsNotNone(world._cid_of_contact_prev)
        world._contact_container.impulses.assign(np.ones(world._contact_container.impulses.shape, dtype=np.float32))
        world._contact_container.prev_impulses.assign(
            np.ones(world._contact_container.prev_impulses.shape, dtype=np.float32)
        )
        world._contact_container.lambdas.assign(np.ones(world._contact_container.lambdas.shape, dtype=np.float32))
        world._contact_container.prev_lambdas.assign(
            np.ones(world._contact_container.prev_lambdas.shape, dtype=np.float32)
        )
        world._contact_container.derived.assign(np.ones(world._contact_container.derived.shape, dtype=np.float32))
        world._cid_of_contact_cur.assign(np.full(world._cid_of_contact_cur.shape, 17, dtype=np.int32))
        world._cid_of_contact_prev.assign(np.full(world._cid_of_contact_prev.shape, 19, dtype=np.int32))

        seed_counter = make_seed_counter(7, device=env.device)
        env.dones.assign(np.array([1.0, 0.0], dtype=np.float32))

        with wp.ScopedCapture(device=env.device) as capture:
            env.reset_done_seed_counter(seed_counter=seed_counter)

        wp.capture_launch(capture.graph)

        impulses = world._contact_container.impulses.numpy()
        prev_impulses = world._contact_container.prev_impulses.numpy()
        lambdas = world._contact_container.lambdas.numpy()
        prev_lambdas = world._contact_container.prev_lambdas.numpy()
        derived = world._contact_container.derived.numpy()
        cid_cur = world._cid_of_contact_cur.numpy()
        cid_prev = world._cid_of_contact_prev.numpy()

        np.testing.assert_array_equal(impulses[:, 0], np.zeros(impulses.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(prev_impulses[:, 0], np.zeros(prev_impulses.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(lambdas[:, 0], np.zeros(lambdas.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(prev_lambdas[:, 0], np.zeros(prev_lambdas.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(derived[:, 0], np.zeros(derived.shape[0], dtype=np.float32))
        self.assertEqual(cid_cur[0], -1)
        self.assertEqual(cid_prev[0], -1)

        np.testing.assert_array_equal(impulses[:, 1], np.ones(impulses.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(prev_impulses[:, 1], np.ones(prev_impulses.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(lambdas[:, 1], np.ones(lambdas.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(prev_lambdas[:, 1], np.ones(prev_lambdas.shape[0], dtype=np.float32))
        np.testing.assert_array_equal(derived[:, 1], np.ones(derived.shape[0], dtype=np.float32))
        self.assertEqual(cid_cur[1], 17)
        self.assertEqual(cid_prev[1], 19)

    def test_reward_matches_nanog1_dt_scaling_inside_graph(self) -> None:
        env = _g1_test_env(world_count=1)
        env.set_command((0.0, 0.0, 0.0))
        env.config.w_gait_contact = 0.0
        env.config.w_gait_swing = 0.0
        env.config.w_gait_hip = 0.0
        env.config.w_base_height = 0.0

        with wp.ScopedCapture(device=env.device) as capture:
            env.observe()

        wp.capture_launch(capture.graph)
        expected = (env.config.w_alive + env.config.w_track_lin + env.config.w_track_ang) * env.config.frame_dt
        self.assertAlmostEqual(float(env.rewards.numpy()[0]), expected, places=5)
        self.assertEqual(float(env.dones.numpy()[0]), 0.0)

        joint_q = env.state_0.joint_q.numpy()
        joint_q[2] = env.config.min_base_height - 0.01
        env.state_0.joint_q.assign(joint_q)

        with wp.ScopedCapture(device=env.device) as fall_capture:
            env.observe()

        wp.capture_launch(fall_capture.graph)
        self.assertAlmostEqual(float(env.rewards.numpy()[0]), expected + env.config.w_termination, places=5)
        self.assertEqual(float(env.dones.numpy()[0]), 1.0)

    def test_gait_reward_uses_foot_contacts_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 gait reward tests")
        config = rl.ConfigEnvG1PhoenX(
            world_count=1,
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            max_episode_steps=0,
            auto_reset=False,
            command=(0.0, 0.0, 0.0),
            w_track_lin=0.0,
            w_track_ang=0.0,
            w_lin_vel_z=0.0,
            w_ang_vel_xy=0.0,
            w_orientation=0.0,
            w_torque=0.0,
            w_action_rate=0.0,
            w_alive=0.0,
        )
        env = rl.EnvG1PhoenX(config, device=device)
        labels = list(env.model.shape_label)
        left_shape = labels.index("left_nanog1_foot_box")
        right_shape = labels.index("right_nanog1_foot_box")
        ground_shape = int(np.flatnonzero(env.model.shape_body.numpy() < 0)[0])

        shape0 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape1 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape0[:2] = np.array([left_shape, right_shape], dtype=np.int32)
        env.contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
        env.contacts.rigid_contact_shape0.assign(shape0)
        env.contacts.rigid_contact_shape1.assign(shape1)

        with wp.ScopedCapture(device=device) as capture:
            env.observe()

        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(env.foot_contacts.numpy()[0], np.array([1.0, 1.0], dtype=np.float32))

        base_z = float(env.state_0.joint_q.numpy()[2])
        expected_shaped = 2.0 * env.config.w_gait_contact
        expected_shaped += env.config.w_base_height * (base_z - env.config.base_height_target) ** 2
        self.assertAlmostEqual(float(env.rewards.numpy()[0]), expected_shaped * env.config.frame_dt, places=5)
        self.assertEqual(float(env.dones.numpy()[0]), 0.0)

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
            value_loss_coeff=0.5,
            value_clip_range=20.0,
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
            self.assertFalse(first.trainer.squash_actions)
            self.assertFalse(restored.squash_actions)
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
            self.assertEqual(restored.config.value_loss_coeff, 0.5)
            self.assertEqual(restored.config.value_clip_range, 20.0)
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
            self.assertFalse(result.trainer.squash_actions)
            self.assertFalse(restored.squash_actions)
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
                policy_network=g1_recipe.POLICY_NETWORK,
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
                value_loss_coeff=g1_recipe.VALUE_LOSS_COEFF,
                value_clip_range=g1_recipe.VALUE_CLIP_RANGE,
                optimizer=g1_recipe.OPTIMIZER,
                optimizer_eps=g1_recipe.OPTIMIZER_EPS,
                optimizer_weight_decay=g1_recipe.OPTIMIZER_WEIGHT_DECAY,
                muon_momentum=g1_recipe.MUON_MOMENTUM,
                squash_actions=g1_recipe.SQUASH_ACTIONS,
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
            self.assertEqual(result["squash_actions"], g1_recipe.SQUASH_ACTIONS)
            self.assertEqual(result["value_loss_coeff"], g1_recipe.VALUE_LOSS_COEFF)
            self.assertEqual(result["value_clip_range"], g1_recipe.VALUE_CLIP_RANGE)
            self.assertEqual(result["optimizer"], g1_recipe.OPTIMIZER)
            self.assertEqual(result["optimizer_eps"], g1_recipe.OPTIMIZER_EPS)
            self.assertEqual(result["optimizer_weight_decay"], g1_recipe.OPTIMIZER_WEIGHT_DECAY)
            self.assertEqual(result["muon_momentum"], g1_recipe.MUON_MOMENTUM)
            self.assertEqual(result["completed_iterations"], 1)
            self.assertEqual(result["trained_samples"], 2)
            self.assertEqual(restored.iteration, 1)
            self.assertTrue(restored.actor_optimizer.matrix_transpose)
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(len(result["gate_history"]), 1)
            self.assertEqual(len(result["train_history"]), 1)
            self.assertFalse(result["pass_gate"])
            self.assertGreater(result["train_env_samples_per_s"], 0.0)
            self.assertTrue(math.isfinite(result["gate_history"][0]["stats"]["battery_perf"]))


if __name__ == "__main__":
    wp.init()
    unittest.main()
