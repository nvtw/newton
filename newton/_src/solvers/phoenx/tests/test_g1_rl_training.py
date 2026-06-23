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
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_ACC_DRIVE,
    _OFF_ACC_FRICTION,
    _OFF_ACC_IMP1,
    _OFF_ACC_LIMIT,
    _OFF_AXIS_WORLD,
    _OFF_BIAS1,
    _OFF_BIAS_DRIVE,
    _OFF_CLAMP,
    _OFF_EFF_INV_AXIAL,
    _OFF_EFF_MASS_DRIVE_SOFT,
    _OFF_GAMMA_DRIVE,
    _OFF_IMPULSE_COEFF,
    _OFF_LIMIT_CACHE,
    _OFF_MASS_COEFF,
    _OFF_MODE_CACHE,
    _OFF_PREVIOUS_QUATERNION_ANGLE,
    _OFF_R1_B1,
    _OFF_R3_B1,
    _OFF_REVOLUTION_COUNTER,
)
from newton._src.solvers.phoenx.experimental.nanog1_import import (
    PufferNetWeights,
    assign_puffernet_weights,
    load_puffernet_weights,
    puffernet_numpy_forward,
)
from newton._src.solvers.phoenx.model_adapter import build_adbs_init_arrays
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.env import collect_ppo_rollout_seed_counter, make_seed_counter
from newton._src.solvers.phoenx.rl_training.g1_diagnostics import (
    G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT,
    G1_FOOT_CONTACT_METRIC_ACTIVE_TANGENT_COUNT,
    G1_FOOT_CONTACT_METRIC_COUNT,
    G1_FOOT_CONTACT_METRIC_COUNT_TOTAL,
    G1_FOOT_CONTACT_METRIC_FRICTION_LOAD,
    G1_FOOT_CONTACT_METRIC_FRICTION_LOAD_RATIO_SUM,
    G1_FOOT_CONTACT_METRIC_HIGH_TANGENT_RATIO_COUNT,
    G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE,
    G1_FOOT_CONTACT_METRIC_SPECULATIVE_COUNT,
    G1_FOOT_CONTACT_METRIC_TANGENT_BIAS,
    G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE,
    G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM,
    scan_g1_foot_contact_metrics,
)
from newton._src.solvers.phoenx.rl_training.kernels import (
    DENSE_TILE_IN,
    DENSE_TILE_OUT,
    PPO_LOG_STD_PARTIAL_BATCH,
    compute_puffer_vtrace_returns_kernel,
    gather_trajectory_minibatch_kernel,
    mingru_sequence_backward_kernel,
    ppo_actor_loss_backward_kernel,
    reduce_ppo_log_std_grad_kernel,
    sample_trajectory_env_ids_kernel,
    scatter_trajectory_ratios_kernel,
    scatter_trajectory_values_kernel,
    trajectory_priority_kernel,
    trajectory_priority_weight_kernel,
    value_column_loss_grad_kernel,
    zero_ppo_actor_stats_kernel,
    zero_scalar_kernel,
)
from newton._src.solvers.phoenx.rl_training.networks import _BF16_FORWARD_MIN_BATCH
from newton._src.solvers.phoenx.rl_training.training import (
    _g1_root_origin_linear_velocity_body_np,
    _quat_rotate_inverse_xyzw_np,
)
from newton._src.solvers.phoenx.solver_config import PHOENX_BOOST_REVOLUTE_DRIVE
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture

_NANOG1_ROOT = Path("/home/twidmer/Documents/git/nanoG1")
_PUFFERLIB_ROOT = Path("/home/twidmer/Documents/git/PufferLib")
_PUFFERLIB_G1_ROOT = Path("/tmp/pufferlib-g1")
_G1_GROUND_SLOT = 0
_G1_STANDING_HOLD_KP = 500.0
_G1_STANDING_HOLD_KD = 10.0
_G = 9.81


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


def _set_g1_contact_normal_impulses(env: rl.EnvG1PhoenX, values: list[float]) -> None:
    impulses = env.solver.world._contact_container.impulses.numpy()
    impulses[...] = 0.0
    count = min(len(values), impulses.shape[1])
    impulses[0, :count] = np.asarray(values[:count], dtype=np.float32)
    env.solver.world._contact_container.impulses.assign(impulses)


def _set_g1_standing_position_hold_gains(env: rl.EnvG1PhoenX) -> None:
    """Use strong position-hold gains for a standing force-balance check."""

    joint_target_ke = env.model.joint_target_ke.numpy()
    joint_target_kd = env.model.joint_target_kd.numpy()
    start = 6
    stop = start + env.action_dim
    joint_target_ke[start:stop] = _G1_STANDING_HOLD_KP
    joint_target_kd[start:stop] = _G1_STANDING_HOLD_KD
    env.model.joint_target_ke.assign(joint_target_ke)
    env.model.joint_target_kd.assign(joint_target_kd)


def _gather_g1_ground_contact_force(env: rl.EnvG1PhoenX) -> tuple[np.ndarray, int, int]:
    """Return net force [N] on robot bodies from the static ground slot."""

    n_cols = env.solver.world.max_contact_columns
    pair_w = wp.zeros(n_cols, dtype=wp.spatial_vector, device=env.device)
    pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=env.device)
    pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=env.device)
    pair_count = wp.zeros(n_cols, dtype=wp.int32, device=env.device)

    env.solver.world.gather_contact_pair_wrenches(pair_w, pair_b1, pair_b2, pair_count)
    with wp.ScopedCapture(device=env.device) as capture:
        env.solver.world.gather_contact_pair_wrenches(pair_w, pair_b1, pair_b2, pair_count)
    wp.capture_launch(capture.graph)

    force_pairs = pair_w.numpy()[:, :3].astype(np.float64)
    body1 = pair_b1.numpy()
    body2 = pair_b2.numpy()
    counts = pair_count.numpy()

    ground_to_body2 = (counts > 0) & (body1 == _G1_GROUND_SLOT) & (body2 != _G1_GROUND_SLOT)
    body1_to_ground = (counts > 0) & (body2 == _G1_GROUND_SLOT) & (body1 != _G1_GROUND_SLOT)
    total = force_pairs[ground_to_body2].sum(axis=0) - force_pairs[body1_to_ground].sum(axis=0)
    pair_total = int(np.count_nonzero(ground_to_body2) + np.count_nonzero(body1_to_ground))
    point_total = int(counts[ground_to_body2].sum() + counts[body1_to_ground].sum())
    return total, pair_total, point_total


@wp.kernel(enable_backward=False)
def _g1_apply_equal_foot_push_kernel(
    force_x: wp.array[wp.float32],
    body_stride: wp.int32,
    left_foot_body: wp.int32,
    right_foot_body: wp.int32,
    body_f: wp.array[wp.spatial_vector],
):
    world = wp.tid()
    force = wp.vec3(force_x[world], wp.float32(0.0), wp.float32(0.0))
    wrench = wp.spatial_vector(force, wp.vec3(0.0, 0.0, 0.0))
    body_f[world * body_stride + left_foot_body] = wrench
    body_f[world * body_stride + right_foot_body] = wrench


def _gather_g1_foot_ground_contact_forces(env: rl.EnvG1PhoenX) -> tuple[np.ndarray, np.ndarray]:
    """Return per-foot ground contact forces [N] and contact counts."""

    n_cols = env.solver.world.max_contact_columns
    pair_w = wp.zeros(n_cols, dtype=wp.spatial_vector, device=env.device)
    pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=env.device)
    pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=env.device)
    pair_count = wp.zeros(n_cols, dtype=wp.int32, device=env.device)

    env.solver.world.gather_contact_pair_wrenches(pair_w, pair_b1, pair_b2, pair_count)
    with wp.ScopedCapture(device=env.device) as capture:
        env.solver.world.gather_contact_pair_wrenches(pair_w, pair_b1, pair_b2, pair_count)
    wp.capture_launch(capture.graph)

    foot_by_slot: dict[int, tuple[int, int]] = {}
    for world in range(env.world_count):
        base = 1 + world * env.body_stride
        foot_by_slot[base + env._left_foot_body_local] = (world, 0)
        foot_by_slot[base + env._right_foot_body_local] = (world, 1)

    forces = np.zeros((env.world_count, 2, 3), dtype=np.float64)
    counts_out = np.zeros((env.world_count, 2), dtype=np.int32)
    force_pairs = pair_w.numpy()[:, :3].astype(np.float64)
    body1 = pair_b1.numpy()
    body2 = pair_b2.numpy()
    counts = pair_count.numpy()
    for pair_index, count in enumerate(counts):
        if int(count) <= 0:
            continue
        b1 = int(body1[pair_index])
        b2 = int(body2[pair_index])
        if b1 == _G1_GROUND_SLOT and b2 in foot_by_slot:
            world, foot = foot_by_slot[b2]
            forces[world, foot] += force_pairs[pair_index]
            counts_out[world, foot] += int(count)
        elif b2 == _G1_GROUND_SLOT and b1 in foot_by_slot:
            world, foot = foot_by_slot[b1]
            forces[world, foot] -= force_pairs[pair_index]
            counts_out[world, foot] += int(count)
    return forces, counts_out


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


@wp.kernel(enable_backward=False)
def _sequence_env_step_values_kernel(
    step_counter: wp.array[wp.int32],
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
    step_rewards: wp.array[wp.float32],
    step_dones: wp.array[wp.float32],
    step_successes: wp.array[wp.float32],
):
    env = wp.tid()
    step = step_counter[0]
    reward = wp.float32(10.0) * wp.float32(step + wp.int32(1)) + wp.float32(env)
    done = wp.float32(0.0)
    if env == wp.int32(1) and step == wp.int32(0):
        done = wp.float32(1.0)
    success = wp.float32(0.25) * wp.float32(step + wp.int32(1)) + wp.float32(env)
    rewards[env] = reward
    dones[env] = done
    successes[env] = success
    step_rewards[env] = reward
    step_dones[env] = done
    step_successes[env] = success
    obs[env, 0] = wp.float32(step + wp.int32(1))


@wp.kernel(enable_backward=False)
def _sequence_env_increment_kernel(step_counter: wp.array[wp.int32]):
    step_counter[0] = step_counter[0] + wp.int32(1)


class _SequenceRewardPPOEnv:
    def __init__(self, *, world_count: int, obs_dim: int, action_dim: int, device: wp.context.Device):
        self.world_count = int(world_count)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.obs = wp.zeros((self.world_count, self.obs_dim), dtype=wp.float32, device=device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=device)
        self.successes = wp.zeros(self.world_count, dtype=wp.float32, device=device)
        self.step_rewards = wp.array(np.asarray([1.0, 2.0], dtype=np.float32), dtype=wp.float32, device=device)
        self.step_dones = wp.array(np.asarray([0.0, 1.0], dtype=np.float32), dtype=wp.float32, device=device)
        self.step_successes = wp.array(np.asarray([0.5, 1.5], dtype=np.float32), dtype=wp.float32, device=device)
        self.step_counter = wp.zeros(1, dtype=wp.int32, device=device)

    def observe(self) -> wp.array2d[wp.float32]:
        return self.obs

    def step(
        self, actions: wp.array2d[wp.float32]
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array[wp.float32]]:
        wp.launch(
            _sequence_env_step_values_kernel,
            dim=self.world_count,
            inputs=[self.step_counter],
            outputs=[
                self.obs,
                self.rewards,
                self.dones,
                self.successes,
                self.step_rewards,
                self.step_dones,
                self.step_successes,
            ],
            device=self.device,
        )
        wp.launch(_sequence_env_increment_kernel, dim=1, outputs=[self.step_counter], device=self.device)
        return self.obs, self.rewards, self.dones


@wp.kernel(enable_backward=False)
def _poison_adbs_reset_runtime_state_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    joint_count: wp.int32,
):
    cid = wp.tid()
    poison = wp.float32(0.0) / wp.float32(0.0)
    if cid == wp.int32(0):
        bodies.velocity[0] = wp.vec3f(poison, poison, poison)
        bodies.angular_velocity[0] = wp.vec3f(poison, poison, poison)
    if cid >= joint_count:
        return

    for row in range(18):
        constraints.data[_OFF_R1_B1 + row, cid] = poison
    constraints.data[_OFF_MASS_COEFF, cid] = poison
    constraints.data[_OFF_IMPULSE_COEFF, cid] = poison
    for row in range(6):
        constraints.data[_OFF_BIAS1 + row, cid] = poison
    for row in range(27):
        constraints.data[_OFF_MODE_CACHE + row, cid] = poison
    constraints.data[_OFF_REVOLUTION_COUNTER, cid] = poison
    constraints.data[_OFF_PREVIOUS_QUATERNION_ANGLE, cid] = poison
    for row in range(10):
        constraints.data[_OFF_R3_B1 + row, cid] = poison
    for row in range(6):
        constraints.data[_OFF_ACC_IMP1 + row, cid] = poison
    constraints.data[_OFF_EFF_INV_AXIAL, cid] = poison
    constraints.data[_OFF_BIAS_DRIVE, cid] = poison
    constraints.data[_OFF_GAMMA_DRIVE, cid] = poison
    constraints.data[_OFF_EFF_MASS_DRIVE_SOFT, cid] = poison
    for row in range(3):
        constraints.data[_OFF_LIMIT_CACHE + row, cid] = poison
    constraints.data[_OFF_CLAMP, cid] = poison
    for row in range(3):
        constraints.data[_OFF_AXIS_WORLD + row, cid] = poison
    constraints.data[_OFF_ACC_DRIVE, cid] = poison
    constraints.data[_OFF_ACC_LIMIT, cid] = poison
    constraints.data[_OFF_ACC_FRICTION, cid] = poison


@wp.kernel(enable_backward=False)
def _adbs_runtime_nonfinite_flags_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    joint_count: wp.int32,
    check_body_velocity: wp.int32,
    flags: wp.array[wp.int32],
):
    cid = wp.tid()
    if cid >= joint_count:
        return

    bad = wp.int32(0)
    if check_body_velocity != wp.int32(0) and cid == wp.int32(0):
        velocity = bodies.velocity[0]
        angular_velocity = bodies.angular_velocity[0]
        if not wp.isfinite(velocity[0]):
            bad = wp.int32(1)
        if not wp.isfinite(velocity[1]):
            bad = wp.int32(1)
        if not wp.isfinite(velocity[2]):
            bad = wp.int32(1)
        if not wp.isfinite(angular_velocity[0]):
            bad = wp.int32(1)
        if not wp.isfinite(angular_velocity[1]):
            bad = wp.int32(1)
        if not wp.isfinite(angular_velocity[2]):
            bad = wp.int32(1)

    for row in range(18):
        if not wp.isfinite(constraints.data[_OFF_R1_B1 + row, cid]):
            bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_MASS_COEFF, cid]):
        bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_IMPULSE_COEFF, cid]):
        bad = wp.int32(1)
    for row in range(6):
        if not wp.isfinite(constraints.data[_OFF_BIAS1 + row, cid]):
            bad = wp.int32(1)
    for row in range(27):
        if not wp.isfinite(constraints.data[_OFF_MODE_CACHE + row, cid]):
            bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_PREVIOUS_QUATERNION_ANGLE, cid]):
        bad = wp.int32(1)
    for row in range(6):
        if not wp.isfinite(constraints.data[_OFF_ACC_IMP1 + row, cid]):
            bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_EFF_INV_AXIAL, cid]):
        bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_BIAS_DRIVE, cid]):
        bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_GAMMA_DRIVE, cid]):
        bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_EFF_MASS_DRIVE_SOFT, cid]):
        bad = wp.int32(1)
    for row in range(3):
        if not wp.isfinite(constraints.data[_OFF_LIMIT_CACHE + row, cid]):
            bad = wp.int32(1)
    for row in range(3):
        if not wp.isfinite(constraints.data[_OFF_AXIS_WORLD + row, cid]):
            bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_ACC_DRIVE, cid]):
        bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_ACC_LIMIT, cid]):
        bad = wp.int32(1)
    if not wp.isfinite(constraints.data[_OFF_ACC_FRICTION, cid]):
        bad = wp.int32(1)

    flags[cid] = bad


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


def _read_c_constant_array(path: Path, name: str, *, dtype: type = float) -> np.ndarray:
    if not path.is_file():
        raise unittest.SkipTest(f"missing nanoG1 reference file: {path}")
    text = path.read_text()
    pattern = rf"{re.escape(name)}\[[^\]]+\]\s*=\s*\{{([^}}]+)\}};"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"missing C constant array {name!r} in {path}")
    raw = re.sub(r"//.*", "", match.group(1).replace("\n", " "))
    values = [item.strip().removesuffix("f") for item in raw.split(",") if item.strip()]
    np_dtype = np.int64 if dtype is int else np.float64
    return np.asarray([dtype(value) for value in values], dtype=np_dtype)


def _quat_wxyz_to_matrix_np(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


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
    ang_b = _quat_rotate_inverse_xyzw_np(q[3:7].reshape(1, 4), qd[3:6].reshape(1, 3))[0]
    phase = 2.0 * math.pi * float(episode_step % deploy.PHASE_PERIOD) / float(deploy.PHASE_PERIOD)
    obs[0:3] = np.float32(deploy.ANG_VEL_SCALE) * ang_b
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

        reference_decimation = int(decimation_match.group(1))
        reference_solver_iterations = int(solver_match.group(1))
        reference_frame_dt = float(dt_match.group(1)) * reference_decimation
        self.assertAlmostEqual(g1_recipe.FRAME_DT, reference_frame_dt)
        self.assertEqual(reference_decimation, 5)
        self.assertEqual(reference_solver_iterations, 2)
        self.assertGreaterEqual(g1_recipe.SIM_SUBSTEPS, reference_decimation)
        self.assertGreaterEqual(g1_recipe.SOLVER_ITERATIONS, reference_solver_iterations)
        self.assertEqual(g1_recipe.CONTROLLED_ACTION_COUNT, 12)
        self.assertIn("-DG1_TASK_V3", reference.TASK_FLAGS)
        self.assertIn("-DG1_PD_UNITREE", reference.TASK_FLAGS)

        g1_gpu_path = _PUFFERLIB_G1_ROOT / "ocean" / "g1gpu" / "g1_gpu.cu"
        if g1_gpu_path.is_file():
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
            self.assertNotIn("W_SWING_CONTACT", g1_gpu)
            self.assertEqual(g1_recipe.W_GAIT_SWING_CONTACT, 0.0)
            self.assertAlmostEqual(g1_recipe.RESET_NOISE, read_define_number("ENV_RESET_NOISE"))
            self.assertEqual(g1_recipe.COMMAND_RESAMPLE_STEPS, int(read_define_number("ENV_CMD_RESAMPLE")))
            self.assertIn("urand_01(&rng) < 0.1f", g1_gpu)
            self.assertIn("g1e_cmd_scale * 1.0f * urand_pm1", g1_gpu)
            self.assertIn("g1e_cmd_scale * 0.6f * urand_pm1", g1_gpu)
            self.assertIn("const float  CS_START = 0.4f", g1_gpu)
            self.assertIn("const double CS_RAMP  = 40.0e6", g1_gpu)

        self.assertEqual(g1_recipe.COMMAND_X_RANGE, (-1.0, 1.0))
        self.assertEqual(g1_recipe.COMMAND_Y_RANGE, (-0.6, 0.6))
        self.assertEqual(g1_recipe.COMMAND_YAW_RANGE, (-1.0, 1.0))
        self.assertAlmostEqual(g1_recipe.COMMAND_ZERO_PROBABILITY, 0.1)
        self.assertAlmostEqual(g1_recipe.COMMAND_CURRICULUM_START, 0.4)
        self.assertEqual(g1_recipe.COMMAND_CURRICULUM_SAMPLES, 40_000_000)
        self.assertAlmostEqual(g1_recipe.ACTION_SCALE, float(recipe["env.action_scale"]))
        self.assertEqual(g1_recipe.MAX_EPISODE_STEPS, int(recipe["env.max_episode_len"]))
        self.assertAlmostEqual(g1_recipe.W_TRACK_LIN, float(recipe["env.w_track_lin"]))
        self.assertAlmostEqual(g1_recipe.W_TRACK_ANG, float(recipe["env.w_track_ang"]))
        self.assertEqual(g1_recipe.W_COMMAND_PROGRESS, 0.0)
        self.assertNotIn("env.w_command_progress", recipe)
        self.assertEqual(g1_recipe.W_TARGET_PROGRESS, 0.0)
        self.assertNotIn("env.w_target_progress", recipe)
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
        self.assertTrue(g1_recipe.default_g1_ppo_config().puffer_vtrace_advantage)
        self.assertFalse(rl.ConfigPPO().puffer_vtrace_advantage)
        self.assertAlmostEqual(g1_recipe.PRIORITY_ALPHA, float(recipe["train.prio_alpha"]))
        self.assertAlmostEqual(g1_recipe.PRIORITY_BETA, float(recipe["train.prio_beta0"]))
        self.assertEqual(g1_recipe.MINIBATCH_SIZE, int(recipe["train.minibatch_size"]))
        self.assertEqual(g1_recipe.ROLLOUT_STEPS, int(recipe["train.horizon"]))
        self.assertAlmostEqual(g1_recipe.MAX_GRAD_NORM, float(recipe["train.max_grad_norm"]))
        self.assertAlmostEqual(g1_recipe.MIRROR_LOSS_COEFF, float(mirror_match.group(1)))
        self.assertTrue(g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START)
        self.assertTrue(rl.ConfigTrainG1PPO().reset_recurrent_state_on_rollout_start)

    def test_nanog1_mirror_map_matches_pinned_pufferlib(self) -> None:
        mirror_path = _PUFFERLIB_G1_ROOT / "ocean" / "g1gpu" / "g1_mirror.h"
        obs_src = _read_c_constant_array(mirror_path, "G1_OBS_MIRROR_SRC", dtype=int)
        obs_sign = _read_c_constant_array(mirror_path, "G1_OBS_MIRROR_SIGN")
        action_src = _read_c_constant_array(mirror_path, "G1_ACT_MIRROR_SRC", dtype=int)
        action_sign = _read_c_constant_array(mirror_path, "G1_ACT_MIRROR_SIGN")
        mirror = rl.g1_mirror_map_ppo()

        np.testing.assert_array_equal(np.asarray(mirror.obs_src, dtype=np.int64), obs_src)
        np.testing.assert_allclose(np.asarray(mirror.obs_sign, dtype=np.float64), obs_sign, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(np.asarray(mirror.action_src, dtype=np.int64), action_src)
        np.testing.assert_allclose(np.asarray(mirror.action_sign, dtype=np.float64), action_sign, rtol=0.0, atol=0.0)
        self.assertEqual(obs_src.size, rl.OBS_DIM_G1)
        self.assertEqual(action_src.size, rl.ACTION_DIM_G1)

    def test_nanog1_model_and_deploy_constants_match_env(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 nanoG1 constant parity tests")
        deploy = _load_nanog1_deploy()
        header = _NANOG1_ROOT / "web" / "g1_model_const.h"
        qpos0 = _read_c_array(header, "hc_qpos0")
        key_qpos = _read_c_array(header, "hc_key_qpos")
        ctrl_range = _read_c_array(header, "hc_act_ctrlrange").reshape(rl.ACTION_DIM_G1, 2)
        act_gain0 = _read_c_array(header, "hc_act_gain0")
        act_bias1 = _read_c_array(header, "hc_act_bias1")
        act_bias2 = _read_c_array(header, "hc_act_bias2")
        jnt_actfrcrange = _read_c_array(header, "hc_jnt_actfrcrange").reshape(rl.ACTION_DIM_G1 + 1, 2)[1:]
        dof_damping = _read_c_array(header, "hc_dof_damping")
        dof_armature = _read_c_array(header, "hc_dof_armature")
        dof_frictionloss = _read_c_array(header, "hc_dof_frictionloss")
        body_mass = _read_c_array(header, "hc_body_mass")
        body_ipos = _read_c_array(header, "hc_body_ipos").reshape(-1, 3)
        body_inertia_diag = _read_c_array(header, "hc_body_inertia").reshape(-1, 3)
        body_iquat = _read_c_array(header, "hc_body_iquat").reshape(-1, 4)
        body_parentid = _read_c_array(header, "hc_body_parentid", dtype=int)
        jnt_qposadr = _read_c_array(header, "hc_jnt_qposadr", dtype=int)
        jnt_dofadr = _read_c_array(header, "hc_jnt_dofadr", dtype=int)
        jnt_bodyid = _read_c_array(header, "hc_jnt_bodyid", dtype=int)
        jnt_axis = _read_c_array(header, "hc_jnt_axis").reshape(-1, 3)
        jnt_range = _read_c_array(header, "hc_jnt_range").reshape(-1, 2)
        jnt_actfrclimited = _read_c_array(header, "hc_jnt_actfrclimited", dtype=int)
        act_trnid = _read_c_array(header, "hc_act_trnid", dtype=int)

        env = rl.EnvG1PhoenX(g1_recipe.default_g1_env_config(world_count=1), device=device)
        self.assertEqual(env.model.body_count, body_mass.size - 1)
        self.assertEqual(env.model.joint_count, jnt_qposadr.size)
        np.testing.assert_allclose(env.model.body_mass.numpy(), body_mass[1:], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.model.body_com.numpy(), body_ipos[1:], rtol=0.0, atol=1.0e-6)
        expected_body_inertia = np.stack(
            [
                _quat_wxyz_to_matrix_np(quat) @ np.diag(diag) @ _quat_wxyz_to_matrix_np(quat).T
                for diag, quat in zip(body_inertia_diag[1:], body_iquat[1:], strict=True)
            ]
        )
        np.testing.assert_allclose(env.model.body_inertia.numpy(), expected_body_inertia, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(env.model.joint_q_start.numpy()[: env.model.joint_count], jnt_qposadr)
        np.testing.assert_array_equal(env.model.joint_qd_start.numpy()[: env.model.joint_count], jnt_dofadr)
        np.testing.assert_array_equal(env.model.joint_child.numpy(), jnt_bodyid - 1)
        np.testing.assert_array_equal(env.model.joint_parent.numpy(), body_parentid[jnt_bodyid] - 1)
        np.testing.assert_allclose(
            env.model.joint_axis.numpy()[6 : 6 + rl.ACTION_DIM_G1], jnt_axis[1:], rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            env.model.joint_limit_lower.numpy()[6 : 6 + rl.ACTION_DIM_G1], jnt_range[1:, 0], rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            env.model.joint_limit_upper.numpy()[6 : 6 + rl.ACTION_DIM_G1], jnt_range[1:, 1], rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            env.model.joint_effort_limit.numpy()[6 : 6 + rl.ACTION_DIM_G1],
            jnt_actfrcrange[:, 1],
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            env.model.joint_gear.numpy()[6 : 6 + rl.ACTION_DIM_G1],
            np.ones(rl.ACTION_DIM_G1, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(jnt_actfrclimited[1:], np.ones(rl.ACTION_DIM_G1, dtype=np.int64))
        np.testing.assert_array_equal(act_trnid, np.arange(1, 1 + rl.ACTION_DIM_G1, dtype=np.int64))
        key_qpos_newton = key_qpos.copy()
        key_qpos_newton[3:7] = np.asarray((key_qpos[4], key_qpos[5], key_qpos[6], key_qpos[3]))
        np.testing.assert_allclose(
            env.model.joint_q.numpy()[: env.coord_stride], key_qpos_newton, rtol=0.0, atol=1.0e-6
        )
        self.assertNotAlmostEqual(float(qpos0[2]), float(key_qpos[2]))
        np.testing.assert_allclose(env.default_joint_pos.numpy(), deploy.HOME, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.default_joint_pos.numpy(), key_qpos[7:], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.ctrl_lower.numpy(), deploy.CTRL_RANGE[:, 0], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_upper.numpy(), deploy.CTRL_RANGE[:, 1], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_lower.numpy(), ctrl_range[:, 0], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(env.ctrl_upper.numpy(), ctrl_range[:, 1], rtol=0.0, atol=1.0e-5)

        expected_kp = deploy.KP.astype(np.float32)
        expected_damping = dof_damping[6 : 6 + rl.ACTION_DIM_G1].astype(np.float32)
        expected_kd = deploy.KD.astype(np.float32) + expected_damping
        expected_force_kd = np.zeros(rl.ACTION_DIM_G1, dtype=np.float32)
        expected_force_kd[: g1_recipe.CONTROLLED_ACTION_COUNT] = deploy.KD[: g1_recipe.CONTROLLED_ACTION_COUNT]
        np.testing.assert_allclose(env.actuator_ke.numpy(), expected_kp, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.actuator_kd.numpy(), expected_kd, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            env.model.joint_target_ke.numpy()[6 : 6 + rl.ACTION_DIM_G1], expected_kp, rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            env.model.joint_target_kd.numpy()[6 : 6 + rl.ACTION_DIM_G1], expected_kd, rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(env.actuator_force_kp.numpy(), expected_kp, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.actuator_force_kp.numpy()[12:], act_gain0[12:], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(act_bias1, -act_gain0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.actuator_force_kd.numpy(), expected_force_kd, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.passive_damping.numpy(), expected_damping, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(act_bias2, np.zeros_like(act_bias2), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(env.actuator_force_lower.numpy(), jnt_actfrcrange[:, 0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.actuator_force_upper.numpy(), jnt_actfrcrange[:, 1], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            env.model.joint_damping.numpy()[6 : 6 + rl.ACTION_DIM_G1], expected_damping, rtol=0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            env.model.joint_armature.numpy()[6 : 6 + rl.ACTION_DIM_G1],
            dof_armature[6 : 6 + rl.ACTION_DIM_G1],
            rtol=0.0,
            atol=1.0e-8,
        )
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

        expected = _reference_obs_from_nanog1_deploy(deploy, q, qd, command, prev_action, episode_step)
        np.testing.assert_allclose(env.obs.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)

    def test_g1_gate_velocity_metric_uses_phoenx_xyzw_quaternions(self) -> None:
        env = _g1_test_env(world_count=1)
        identity_xyzw = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        x_world = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(_quat_rotate_inverse_xyzw_np(identity_xyzw, x_world), x_world, rtol=0.0, atol=0.0)

        half_yaw = np.float32(0.25 * np.pi)
        yaw_90_xyzw = np.asarray([[0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)]], dtype=np.float32)
        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        q[3:7] = yaw_90_xyzw[0]
        qd[:] = 0.0
        qd[0:3] = x_world[0]
        env.state_0.joint_q.assign(q)
        env.state_0.joint_qd.assign(qd)
        with wp.ScopedCapture(device=env.device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        expected_body = np.asarray([[0.0, -1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(
            _quat_rotate_inverse_xyzw_np(q.reshape(1, env.coord_stride)[:, 3:7], qd.reshape(1, env.dof_stride)[:, 0:3]),
            expected_body,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_g1_linear_tracking_uses_root_origin_velocity_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 root velocity convention tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                w_track_lin=1.0,
                w_track_ang=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=0.0,
                w_base_height=0.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
            ),
            device=device,
        )
        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        q[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        qd[:] = 0.0
        qd[0:6] = np.asarray([1.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float32)
        root_com = env.model.body_com.numpy().reshape(env.world_count, env.body_stride, 3)[:, 0, :].copy()
        lin_origin_b = _g1_root_origin_linear_velocity_body_np(
            q.reshape(1, env.coord_stride), qd.reshape(1, env.dof_stride), root_com
        )
        self.assertGreater(float(abs(lin_origin_b[0, 0] - qd[0])), 0.05)

        env.state_0.joint_q.assign(q)
        env.state_0.joint_qd.assign(qd)
        env.command.assign(np.asarray([[lin_origin_b[0, 0], lin_origin_b[0, 1], 0.0]], dtype=np.float32))
        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.successes.numpy(), np.ones(1, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)

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

    def test_pufferlib_ppo_actor_loss_backward_matches_warp_kernel(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 PPO actor loss parity tests")
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {pufferlib_cu}")
        cuda_text = pufferlib_cu.read_text()
        self.assertIn("float wa = -w * adv_normalized", cuda_text)
        self.assertIn("pg_loss = fmaxf(pg_loss1, pg_loss2)", cuda_text)
        self.assertIn("a.grad_logits[grad_logits_base + h] = d_new_logp * diff / var", cuda_text)
        self.assertIn(
            "a.grad_logstd[nt * a.num_atns + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term",
            cuda_text,
        )

        rows = 4
        action_dim = 2
        clip_ratio = np.float32(0.2)
        entropy_coeff = np.float32(0.03)
        policy_out_np = np.asarray([[0.2, -0.4], [0.1, 0.3], [-0.25, 0.35], [0.45, -0.15]], dtype=np.float32)
        log_std_np = np.asarray([-0.3, 0.25], dtype=np.float32)
        actions_np = np.asarray([[0.35, -0.2], [-0.15, 0.65], [0.4, -0.05], [0.2, -0.55]], dtype=np.float32)
        advantages_np = np.asarray([0.5, -0.25, 0.8, -0.4], dtype=np.float32)
        desired_ratios = np.asarray([1.0, 1.35, 1.35, 0.72], dtype=np.float32)
        std = np.exp(log_std_np).astype(np.float32)
        var = std * std
        new_log_probs_np = np.sum(
            -0.5 * ((actions_np - policy_out_np) / std) ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std_np,
            axis=1,
            dtype=np.float32,
        ).astype(np.float32)
        old_log_probs_np = (new_log_probs_np - np.log(desired_ratios).astype(np.float32)).astype(np.float32)

        policy_out = wp.array(policy_out_np, dtype=wp.float32, device=device)
        log_std = wp.array(log_std_np, dtype=wp.float32, device=device)
        actions = wp.array(actions_np, dtype=wp.float32, device=device)
        old_log_probs = wp.array(old_log_probs_np, dtype=wp.float32, device=device)
        advantages = wp.array(advantages_np, dtype=wp.float32, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device)
        approx_kl = wp.zeros(1, dtype=wp.float32, device=device)
        clip_fraction = wp.zeros(1, dtype=wp.float32, device=device)
        ratios = wp.zeros(rows, dtype=wp.float32, device=device)
        policy_out_grad = wp.zeros_like(policy_out)
        partial_count = (rows + PPO_LOG_STD_PARTIAL_BATCH - 1) // PPO_LOG_STD_PARTIAL_BATCH
        log_std_grad_partials = wp.zeros((partial_count, action_dim), dtype=wp.float32, device=device)
        log_std_grad = wp.zeros(action_dim, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                zero_ppo_actor_stats_kernel,
                dim=max(partial_count * action_dim, 1),
                inputs=[partial_count, action_dim],
                outputs=[loss, approx_kl, clip_fraction, log_std_grad_partials],
                device=device,
            )
            wp.launch(
                ppo_actor_loss_backward_kernel,
                dim=rows,
                inputs=[
                    policy_out,
                    log_std,
                    actions,
                    old_log_probs,
                    advantages,
                    float(clip_ratio),
                    float(entropy_coeff),
                    action_dim,
                    0,
                    0,
                    -20.0,
                    2.0,
                    rows,
                ],
                outputs=[loss, approx_kl, clip_fraction, ratios, policy_out_grad, log_std_grad_partials],
                device=device,
            )
            wp.launch(
                reduce_ppo_log_std_grad_kernel,
                dim=action_dim,
                inputs=[log_std_grad_partials, partial_count],
                outputs=[log_std_grad],
                device=device,
            )
        wp.capture_launch(capture.graph)

        expected_loss = np.float32(0.0)
        expected_approx_kl = np.float32(0.0)
        expected_clip_fraction = np.float32(0.0)
        expected_policy_grad = np.zeros_like(policy_out_np)
        expected_log_std_grad = np.zeros(action_dim, dtype=np.float32)
        entropy = np.sum(np.float32(0.5 * math.log(2.0 * math.pi * math.e)) + log_std_np, dtype=np.float32)
        inv_batch = np.float32(1.0 / rows)
        expected_ratios = np.zeros(rows, dtype=np.float32)
        for row in range(rows):
            log_ratio = np.float32(new_log_probs_np[row] - old_log_probs_np[row])
            ratio = np.float32(math.exp(float(log_ratio)))
            expected_ratios[row] = ratio
            clipped = np.clip(ratio, np.float32(1.0) - clip_ratio, np.float32(1.0) + clip_ratio)
            adv = advantages_np[row]
            pg_loss_unclipped = np.float32(-adv * ratio)
            pg_loss_clipped = np.float32(-adv * clipped)
            pg_loss = max(pg_loss_unclipped, pg_loss_clipped)
            expected_loss += np.float32((pg_loss - entropy_coeff * entropy) * inv_batch)
            expected_approx_kl += np.float32(((ratio - np.float32(1.0)) - log_ratio) * inv_batch)
            if abs(ratio - np.float32(1.0)) > clip_ratio:
                expected_clip_fraction += inv_batch
            d_log_prob = np.float32(-adv * ratio * inv_batch)
            clipped_branch = pg_loss_clipped > pg_loss_unclipped
            outside_clip = ratio <= np.float32(1.0) - clip_ratio or ratio >= np.float32(1.0) + clip_ratio
            if clipped_branch and outside_clip:
                d_log_prob = np.float32(0.0)
            diff = actions_np[row] - policy_out_np[row]
            expected_policy_grad[row] = d_log_prob * diff / var
            expected_log_std_grad += d_log_prob * (diff * diff / var - np.float32(1.0)) - entropy_coeff * inv_batch

        np.testing.assert_allclose(loss.numpy()[0], expected_loss, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(approx_kl.numpy()[0], expected_approx_kl, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(clip_fraction.numpy()[0], expected_clip_fraction, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(ratios.numpy(), expected_ratios, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(policy_out_grad.numpy(), expected_policy_grad, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(log_std_grad.numpy(), expected_log_std_grad, rtol=1.0e-6, atol=1.0e-6)

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

    def test_pufferlib_advantage_normalization_matches_warp_buffer(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 advantage normalization tests")
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {pufferlib_cu}")
        text = pufferlib_cu.read_text()
        self.assertIn("*var_out = sdata[0] / (float)(n - 1)", text)
        self.assertIn("float adv_std = sqrtf(float(a.adv_var[0]))", text)
        self.assertIn("adv_normalized = (adv - float(a.adv_mean[0])) / (adv_std + 1e-8f)", text)

        raw = np.asarray([1.0, 2.0, 4.0, -1.0, 0.5, 3.0], dtype=np.float32)
        buffer = rl.BufferRollout(num_steps=2, num_envs=3, obs_dim=1, action_dim=1, device=device)
        buffer.advantages.assign(raw)
        with wp.ScopedCapture(device=device) as capture:
            buffer.normalize_advantages()
        wp.capture_launch(capture.graph)

        expected = (raw - np.mean(raw)) / (np.std(raw, ddof=1) + np.float32(1.0e-8))
        np.testing.assert_allclose(buffer.advantages.numpy(), expected, rtol=1.0e-6, atol=1.0e-6)

    def test_pufferlib_priority_weights_match_warp_kernels(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 priority replay parity tests")
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {pufferlib_cu}")
        text = pufferlib_cu.read_text()
        self.assertIn("float pw = __powf(local_sum, prio_alpha)", text)
        self.assertIn("prio_weights[t] = (prio_weights[t] + eps) / block_sum", text)
        self.assertIn("mb_prio[tx] = __powf(value, -anneal_beta)", text)

        num_steps = 3
        num_envs = 4
        alpha = np.float32(0.4)
        advantages_np = np.asarray(
            [[1.0, -2.0, 0.0, 0.25], [-0.5, 3.0, 0.0, -0.75], [0.25, -1.0, 0.0, 1.5]],
            dtype=np.float32,
        )
        advantages = wp.array(advantages_np.reshape(-1), dtype=wp.float32, device=device)
        priorities = wp.zeros(num_envs, dtype=wp.float32, device=device)
        weights = wp.zeros(num_envs, dtype=wp.float32, device=device)
        total_weight = wp.zeros(1, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                trajectory_priority_kernel,
                dim=num_envs,
                inputs=[advantages, num_steps, num_envs],
                outputs=[priorities],
                device=device,
            )
            wp.launch(
                trajectory_priority_weight_kernel,
                dim=num_envs,
                inputs=[priorities, float(alpha)],
                outputs=[weights, total_weight],
                device=device,
            )
        wp.capture_launch(capture.graph)

        expected_priorities = np.sum(np.abs(advantages_np), axis=0, dtype=np.float32)
        expected_weights = np.power(np.maximum(expected_priorities, np.float32(0.0)), alpha).astype(np.float32)
        expected_probs = (expected_weights + np.float32(1.0e-6)) / (
            np.sum(expected_weights, dtype=np.float32) + np.float32(1.0e-6)
        )

        np.testing.assert_allclose(priorities.numpy(), expected_priorities, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(weights.numpy(), expected_weights, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            total_weight.numpy()[0], np.sum(expected_weights, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6
        )
        self.assertEqual(float(expected_weights[2]), 0.0)
        self.assertLess(float(expected_probs[2]), 1.0e-6)

        sample_count = 16
        beta = np.float32(1.0)
        manual_weights_np = np.asarray([0.0, 1.0e-8, 2.0e-8, 0.0], dtype=np.float32)
        manual_total_np = np.asarray([np.sum(manual_weights_np, dtype=np.float32)], dtype=np.float32)
        manual_weights = wp.array(manual_weights_np, dtype=wp.float32, device=device)
        manual_total = wp.array(manual_total_np, dtype=wp.float32, device=device)
        env_ids = wp.zeros(sample_count, dtype=wp.int32, device=device)
        importance_weights = wp.zeros(sample_count, dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as sample_capture:
            wp.launch(
                sample_trajectory_env_ids_kernel,
                dim=sample_count,
                inputs=[manual_weights, manual_total, num_envs, 17, float(beta), 1],
                outputs=[env_ids, importance_weights],
                device=device,
            )
        wp.capture_launch(sample_capture.graph)

        env_ids_np = env_ids.numpy()
        manual_probs = (manual_weights_np + np.float32(1.0e-6)) / (manual_total_np[0] + np.float32(1.0e-6))
        expected_importance = np.power(np.float32(num_envs) * manual_probs[env_ids_np], -beta).astype(np.float32)
        self.assertIn(0, env_ids_np.tolist())
        np.testing.assert_allclose(importance_weights.numpy(), expected_importance, rtol=1.0e-6, atol=1.0e-6)

    def test_trajectory_minibatch_gather_scatter_matches_puffer_layout(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 trajectory minibatch layout tests")
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {pufferlib_cu}")
        text = pufferlib_cu.read_text()
        self.assertIn("Transpose from rollout layout (T, B, ...) to train layout (B, T, ...)", text)
        self.assertIn("select_copy<<<dim3(mb_segs, channels)", text)
        self.assertIn("index_copy<<<grid_size(num_idx)", text)

        num_steps = 3
        num_envs = 4
        segment_count = 2
        obs_dim = 3
        action_dim = 2
        env_ids_np = np.asarray([3, 1], dtype=np.int32)
        sample_count = num_steps * segment_count
        src_count = num_steps * num_envs
        obs_np = np.arange(src_count * obs_dim, dtype=np.float32).reshape(src_count, obs_dim)
        actions_np = (100.0 + np.arange(src_count * action_dim, dtype=np.float32)).reshape(src_count, action_dim)
        log_probs_np = 200.0 + np.arange(src_count, dtype=np.float32)
        advantages_np = (300.0 + np.arange(src_count, dtype=np.float32)).reshape(num_steps, num_envs)
        returns_np = (400.0 + np.arange(src_count, dtype=np.float32)).reshape(num_steps, num_envs)
        values_np = (500.0 + np.arange((num_steps + 1) * num_envs, dtype=np.float32)).reshape(num_steps + 1, num_envs)

        env_ids = wp.array(env_ids_np, dtype=wp.int32, device=device)
        obs_src = wp.array(obs_np, dtype=wp.float32, device=device)
        actions_src = wp.array(actions_np, dtype=wp.float32, device=device)
        log_probs_src = wp.array(log_probs_np, dtype=wp.float32, device=device)
        advantages_src = wp.array(advantages_np.reshape(-1), dtype=wp.float32, device=device)
        returns_src = wp.array(returns_np.reshape(-1), dtype=wp.float32, device=device)
        values_src = wp.array(values_np.reshape(-1), dtype=wp.float32, device=device)
        obs_dst = wp.zeros((sample_count, obs_dim), dtype=wp.float32, device=device)
        actions_dst = wp.zeros((sample_count, action_dim), dtype=wp.float32, device=device)
        log_probs_dst = wp.zeros(sample_count, dtype=wp.float32, device=device)
        advantages_dst = wp.zeros(sample_count, dtype=wp.float32, device=device)
        returns_dst = wp.zeros(sample_count, dtype=wp.float32, device=device)
        old_values_dst = wp.zeros(sample_count, dtype=wp.float32, device=device)
        ratios_src_np = 600.0 + np.arange(sample_count, dtype=np.float32)
        ratios_src = wp.array(ratios_src_np, dtype=wp.float32, device=device)
        ratios_dst = wp.full(src_count, -1.0, dtype=wp.float32, device=device)
        new_values_src_np = np.stack(
            (
                700.0 + np.arange(sample_count, dtype=np.float32),
                800.0 + np.arange(sample_count, dtype=np.float32),
            ),
            axis=1,
        ).astype(np.float32)
        new_values_src = wp.array(new_values_src_np, dtype=wp.float32, device=device)
        values_dst = wp.full(src_count, -1.0, dtype=wp.float32, device=device)
        max_cols = max(obs_dim, action_dim)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                gather_trajectory_minibatch_kernel,
                dim=(sample_count, max_cols),
                inputs=[
                    env_ids,
                    num_envs,
                    segment_count,
                    obs_dim,
                    action_dim,
                    obs_src,
                    actions_src,
                    log_probs_src,
                    advantages_src,
                    returns_src,
                    values_src,
                ],
                outputs=[obs_dst, actions_dst, log_probs_dst, advantages_dst, returns_dst, old_values_dst],
                device=device,
            )
            wp.launch(
                scatter_trajectory_ratios_kernel,
                dim=sample_count,
                inputs=[env_ids, num_envs, segment_count, ratios_src],
                outputs=[ratios_dst],
                device=device,
            )
            wp.launch(
                scatter_trajectory_values_kernel,
                dim=sample_count,
                inputs=[env_ids, num_envs, segment_count, new_values_src, 1],
                outputs=[values_dst],
                device=device,
            )
        wp.capture_launch(capture.graph)

        selected_rows = []
        for step in range(num_steps):
            for env_id in env_ids_np:
                selected_rows.append(step * num_envs + int(env_id))
        selected_rows_np = np.asarray(selected_rows, dtype=np.int32)
        np.testing.assert_allclose(obs_dst.numpy(), obs_np[selected_rows_np], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actions_dst.numpy(), actions_np[selected_rows_np], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(log_probs_dst.numpy(), log_probs_np[selected_rows_np], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            advantages_dst.numpy(), advantages_np.reshape(-1)[selected_rows_np], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(returns_dst.numpy(), returns_np.reshape(-1)[selected_rows_np], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(old_values_dst.numpy(), values_np.reshape(-1)[selected_rows_np], rtol=0.0, atol=0.0)

        expected_ratios = np.full(src_count, -1.0, dtype=np.float32)
        expected_values = np.full(src_count, -1.0, dtype=np.float32)
        expected_ratios[selected_rows_np] = ratios_src_np
        expected_values[selected_rows_np] = new_values_src_np[:, 1]
        np.testing.assert_allclose(ratios_dst.numpy(), expected_ratios, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(values_dst.numpy(), expected_values, rtol=0.0, atol=0.0)

    def test_compact_rollout_and_update_metric_readbacks_reuse_pinned_buffers(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 compact metric readback tests")
        buffer = rl.BufferRollout(num_steps=2, num_envs=3, obs_dim=4, action_dim=2, device=device)
        buffer.rewards.assign(np.linspace(-0.3, 0.4, buffer.num_samples, dtype=np.float32))
        buffer.dones.assign(np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        buffer.successes.assign(np.linspace(0.1, 0.6, buffer.num_samples, dtype=np.float32))

        metric_host_a = buffer.copy_reward_done_success_sums_to_host()
        metric_host_b = buffer.copy_reward_done_success_sums_to_host()
        self.assertIs(metric_host_a, metric_host_b)
        self.assertEqual(metric_host_a.device, wp.get_device("cpu"))
        self.assertTrue(metric_host_a.pinned)
        np.testing.assert_allclose(
            metric_host_a.numpy(),
            np.asarray(
                [
                    np.sum(buffer.rewards.numpy(), dtype=np.float32),
                    np.sum(buffer.dones.numpy(), dtype=np.float32),
                    np.sum(buffer.successes.numpy(), dtype=np.float32),
                ],
                dtype=np.float32,
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        trainer = rl.TrainerPPO(
            obs_dim=buffer.obs_dim,
            action_dim=buffer.action_dim,
            hidden_layers=(8,),
            config=rl.ConfigPPO(manual_actor_backward=True, manual_critic_backward=True),
            device=device,
            seed=7,
            squash_actions=False,
        )
        stats_host_a = trainer.copy_update_stats_to_host()
        stats_host_b = trainer.copy_update_stats_to_host()
        self.assertIs(stats_host_a, stats_host_b)
        self.assertEqual(stats_host_a.device, wp.get_device("cpu"))
        self.assertTrue(stats_host_a.pinned)
        self.assertEqual(stats_host_a.shape, (4,))

    def test_standard_gae_uses_current_reward_and_done_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX PPO GAE regression tests")
        num_steps = 3
        num_envs = 2
        gamma = np.float32(0.91)
        gae_lambda = np.float32(0.73)
        reward_clip = np.float32(1.0)
        rewards_np = np.asarray([[1.2, -0.3], [0.5, 2.4], [-1.7, 0.8]], dtype=np.float32)
        dones_np = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        values_np = np.asarray([[0.1, -0.2], [0.4, 0.3], [-0.1, 0.7], [0.2, 0.9]], dtype=np.float32)

        buffer = rl.BufferRollout(num_steps=num_steps, num_envs=num_envs, obs_dim=1, action_dim=1, device=device)
        buffer.rewards.assign(rewards_np.reshape(-1))
        buffer.dones.assign(dones_np.reshape(-1))
        buffer.values.assign(values_np.reshape(-1))

        with wp.ScopedCapture(device=device) as capture:
            buffer.compute_returns(gamma=float(gamma), gae_lambda=float(gae_lambda), reward_clip=float(reward_clip))
        wp.capture_launch(capture.graph)

        expected_adv = np.zeros((num_steps, num_envs), dtype=np.float32)
        for env_id in range(num_envs):
            trace = np.float32(0.0)
            for t in range(num_steps - 1, -1, -1):
                reward = np.clip(rewards_np[t, env_id], -reward_clip, reward_clip)
                non_terminal = np.float32(1.0) - dones_np[t, env_id]
                delta = reward + gamma * values_np[t + 1, env_id] * non_terminal - values_np[t, env_id]
                trace = delta + gamma * gae_lambda * non_terminal * trace
                expected_adv[t, env_id] = trace
        expected_returns = expected_adv + values_np[:-1]

        np.testing.assert_allclose(
            buffer.advantages.numpy().reshape(num_steps, num_envs), expected_adv, rtol=1.0e-6, atol=1.0e-6
        )
        np.testing.assert_allclose(
            buffer.returns.numpy().reshape(num_steps, num_envs), expected_returns, rtol=1.0e-6, atol=1.0e-6
        )

    def test_rollout_uses_puffer_reward_layout_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX PPO Puffer rollout layout tests")
        num_steps = 2
        num_envs = 2

        def collect_with_layout(*, puffer_layout: bool) -> rl.BufferRollout:
            env = _SequenceRewardPPOEnv(world_count=num_envs, obs_dim=3, action_dim=2, device=device)
            buffer = rl.BufferRollout(
                num_steps=num_steps, num_envs=num_envs, obs_dim=env.obs_dim, action_dim=env.action_dim, device=device
            )
            trainer = rl.TrainerPPO(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                hidden_layers=(4,),
                config=rl.ConfigPPO(puffer_vtrace_advantage=puffer_layout),
                device=device,
                seed=17,
                squash_actions=False,
            )
            seed_counter = make_seed_counter(123, device=device)
            with wp.ScopedCapture(device=device) as capture:
                collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=seed_counter)
            wp.capture_launch(capture.graph)
            return buffer

        standard = collect_with_layout(puffer_layout=False)
        puffer = collect_with_layout(puffer_layout=True)

        np.testing.assert_allclose(
            standard.rewards.numpy().reshape(num_steps, num_envs),
            np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            puffer.rewards.numpy().reshape(num_steps, num_envs),
            np.asarray([[1.0, 2.0], [10.0, 11.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            standard.dones.numpy().reshape(num_steps, num_envs),
            np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            puffer.dones.numpy().reshape(num_steps, num_envs),
            np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            puffer.successes.numpy().reshape(num_steps, num_envs),
            np.asarray([[0.5, 1.5], [0.25, 1.25]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_pufferlib_vtrace_matches_puffer_reward_layout(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 V-trace parity tests")
        pufferlib_cu = _PUFFERLIB_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib reference file: {pufferlib_cu}")
        text = pufferlib_cu.read_text()
        self.assertIn("float r_nxt = to_float(rewards[t_next])", text)
        self.assertIn("int start_idx = (chunk == num_chunks - 1) ? (N - 2) : (N - 1)", text)
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
                compute_puffer_vtrace_returns_kernel,
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

        expected_adv = np.zeros((num_steps, num_envs), dtype=np.float32)
        for env_id in range(num_envs):
            trace = np.float32(0.0)
            for t in range(num_steps - 2, -1, -1):
                next_t = t + 1
                reward = np.clip(rewards_np[next_t, env_id], -reward_clip, reward_clip)
                next_nonterminal = np.float32(1.0) - dones_np[next_t, env_id]
                rho = min(ratios_np[t, env_id], rho_clip)
                c = min(ratios_np[t, env_id], c_clip)
                delta = rho * (reward + gamma * values_np[next_t, env_id] * next_nonterminal - values_np[t, env_id])
                trace = delta + gamma * gae_lambda * c * trace * next_nonterminal
                expected_adv[t, env_id] = trace
        expected_returns = expected_adv + values_np[:-1]
        self.assertTrue(np.all(expected_adv[-1] == 0.0))
        self.assertNotAlmostEqual(float(expected_adv[0, 0]), float(rewards_np[0, 0]))

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

    def test_rollout_seed_counter_changes_actions_across_graph_replays(self) -> None:
        device = require_cuda_graph_capture("PhoenX PPO stochastic rollout seed tests")
        env = _ConstantPPOEnv(world_count=3, obs_dim=4, action_dim=2, device=device)
        buffer = rl.BufferRollout(
            num_steps=2, num_envs=env.world_count, obs_dim=env.obs_dim, action_dim=env.action_dim, device=device
        )
        trainer = rl.TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=(8,),
            config=rl.ConfigPPO(),
            device=device,
            seed=5,
            squash_actions=False,
            log_std_init=-0.1,
        )
        seed_counter = make_seed_counter(101, device=device)

        with wp.ScopedCapture(device=device) as capture:
            collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=seed_counter)

        wp.capture_launch(capture.graph)
        first_actions = buffer.actions.numpy().copy()
        wp.capture_launch(capture.graph)
        second_actions = buffer.actions.numpy().copy()

        self.assertFalse(np.allclose(first_actions, second_actions))
        np.testing.assert_array_equal(seed_counter.numpy(), np.array([105], dtype=np.int32))

    def test_collect_rollout_can_preserve_recurrent_state_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX recurrent rollout preserve-state tests")
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
        first_actions, _first_log_probs, _first_values = trainer.act_reuse(env.obs, seed=11)
        reset_expected = first_actions.numpy().copy()
        preserved_actions, _preserved_log_probs, _preserved_values = trainer.act_reuse(env.obs, seed=11)
        preserved_expected = preserved_actions.numpy().copy()
        self.assertFalse(np.allclose(preserved_expected, reset_expected, rtol=1.0e-5, atol=1.0e-5))

        trainer.reset_rollout_state()
        trainer.act_reuse(env.obs, seed=11)
        seed_counter = make_seed_counter(11, device=device)
        with wp.ScopedCapture(device=device) as capture:
            collect_ppo_rollout_seed_counter(
                env, trainer, buffer, seed_counter=seed_counter, reset_state_at_start=False
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(buffer.actions.numpy(), preserved_expected, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_array_equal(seed_counter.numpy(), np.array([12], dtype=np.int32))

    def test_recurrent_rollout_reset_keeps_update_replay_consistent_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX recurrent PPO update replay tests")
        env = _ConstantPPOEnv(world_count=2, obs_dim=3, action_dim=1, device=device)
        buffer = rl.BufferRollout(
            num_steps=2, num_envs=env.world_count, obs_dim=env.obs_dim, action_dim=1, device=device
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
        trainer.act_reuse(env.obs, seed=7)
        seed_counter = make_seed_counter(11, device=device)
        with wp.ScopedCapture(device=device) as capture:
            collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=seed_counter)
            reset_replay = trainer._policy_update_reuse(buffer.obs, buffer)
        wp.capture_launch(capture.graph)
        reset_values = buffer.values.numpy()[: buffer.num_samples].copy()
        reset_replay_values = reset_replay.numpy()[: buffer.num_samples, trainer.value_column].copy()
        np.testing.assert_allclose(reset_values, reset_replay_values, rtol=1.0e-6, atol=1.0e-6)

        trainer.reset_rollout_state()
        trainer.act_reuse(env.obs, seed=7)
        seed_counter = make_seed_counter(11, device=device)
        with wp.ScopedCapture(device=device) as capture:
            collect_ppo_rollout_seed_counter(
                env, trainer, buffer, seed_counter=seed_counter, reset_state_at_start=False
            )
            preserved_replay = trainer._policy_update_reuse(buffer.obs, buffer)
        wp.capture_launch(capture.graph)
        preserved_values = buffer.values.numpy()[: buffer.num_samples].copy()
        preserved_replay_values = preserved_replay.numpy()[: buffer.num_samples, trainer.value_column].copy()
        self.assertFalse(np.allclose(preserved_values, preserved_replay_values, rtol=1.0e-5, atol=1.0e-5))

    def test_puffer_mingru_forward_and_reset_match_numpy_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 MinGRU network tests")
        models_py = _PUFFERLIB_ROOT / "pufferlib" / "models.py"
        models_cu = _PUFFERLIB_ROOT / "src" / "models.cu"
        if not models_py.is_file() or not models_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib MinGRU reference files under {_PUFFERLIB_ROOT}")
        models_text = models_py.read_text()
        self.assertIn("return torch.where(x >= 0, x + 0.5, x.sigmoid())", models_text)
        self.assertIn("return g * out + (1.0 - g) * x", models_text)
        self.assertIn("log_coeffs = -F.softplus(gate)", models_text)
        cuda_text = models_cu.read_text()
        self.assertIn("float hidden_tilde = (hidden >= 0.0f) ? hidden + 0.5f : fast_sigmoid(hidden)", cuda_text)
        self.assertIn("float mingru_out = lerp(state, hidden_tilde, gate_sigmoid)", cuda_text)

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

    def test_puffer_mingru_initialization_matches_native_shape_order(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 MinGRU initializer tests")
        models_cu = _PUFFERLIB_ROOT / "src" / "models.cu"
        if not models_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib MinGRU reference file: {models_cu}")
        text = models_cu.read_text()
        self.assertIn("p->encoder.init_weights(w.encoder, seed, stream)", text)
        self.assertIn("p->decoder.init_weights(w.decoder, seed, stream)", text)
        self.assertIn("p->network.init_weights(w.network, seed, stream)", text)
        self.assertIn(".shape = {ew->out_dim, ew->in_dim}", text)
        self.assertIn(".shape = {dw->output_dim + 1, dw->hidden_dim}", text)
        self.assertIn(".shape = {3 * m->hidden, m->hidden}", text)

        seed = 42
        hidden_size = 32
        num_layers = 3
        net = rl.PufferMinGRUNet(
            input_dim=rl.OBS_DIM_G1,
            hidden_size=hidden_size,
            output_dim=rl.ACTION_DIM_G1 + 1,
            num_layers=num_layers,
            device=device,
            seed=seed,
        )
        self.assertEqual(net.encoder_weight.shape, (rl.OBS_DIM_G1, hidden_size))
        self.assertEqual(net.decoder_weight.shape, (hidden_size, rl.ACTION_DIM_G1 + 1))
        self.assertEqual([weight.shape for weight in net.recurrent_weights], [(hidden_size, 3 * hidden_size)] * 3)

        rng = np.random.default_rng(seed)
        encoder_bound = np.sqrt(np.float32(2.0)) / np.sqrt(np.float32(rl.OBS_DIM_G1))
        decoder_bound = np.float32(1.0) / np.sqrt(np.float32(hidden_size))
        expected_encoder = rng.uniform(-encoder_bound, encoder_bound, size=(rl.OBS_DIM_G1, hidden_size)).astype(
            np.float32
        )
        expected_decoder = rng.uniform(-decoder_bound, decoder_bound, size=(hidden_size, rl.ACTION_DIM_G1 + 1)).astype(
            np.float32
        )
        expected_recurrent = [
            rng.uniform(-decoder_bound, decoder_bound, size=(hidden_size, 3 * hidden_size)).astype(np.float32)
            for _ in range(num_layers)
        ]

        np.testing.assert_allclose(net.encoder_weight.numpy(), expected_encoder, rtol=0.0, atol=2.0e-8)
        np.testing.assert_allclose(net.decoder_weight.numpy(), expected_decoder, rtol=0.0, atol=2.0e-8)
        for actual, expected in zip(net.recurrent_weights, expected_recurrent, strict=True):
            np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=2.0e-8)

    def test_puffernet_weight_import_matches_numpy_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX PufferNet import tests")
        input_dim = 3
        hidden_size = 2
        action_dim = 2
        num_layers = 2
        encoder_c = np.asarray(
            [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.7]],
            dtype=np.float32,
        )
        decoder_c = np.asarray(
            [[0.3, -0.2], [-0.4, 0.6], [0.1, 0.5]],
            dtype=np.float32,
        )
        log_std = np.asarray([-0.7, -0.2], dtype=np.float32)
        recurrent_c = (
            np.linspace(-0.5, 0.6, 3 * hidden_size * hidden_size, dtype=np.float32).reshape(
                3 * hidden_size, hidden_size
            ),
            np.linspace(0.4, -0.3, 3 * hidden_size * hidden_size, dtype=np.float32).reshape(
                3 * hidden_size, hidden_size
            ),
        )

        raw_parts = []
        raw_count = 0

        def append_aligned(values: np.ndarray) -> None:
            nonlocal raw_count
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            raw_parts.append(flat)
            raw_count += int(flat.size)
            pad = (-raw_count) % 8
            if pad:
                raw_parts.append(np.full(pad, 99.0, dtype=np.float32))
                raw_count += pad

        append_aligned(encoder_c)
        append_aligned(decoder_c)
        append_aligned(log_std)
        for recurrent in recurrent_c:
            append_aligned(recurrent)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "puffernet.bin"
            np.concatenate(raw_parts).astype(np.float32).tofile(path)
            weights = load_puffernet_weights(
                path,
                input_dim=input_dim,
                hidden_size=hidden_size,
                action_dim=action_dim,
                num_layers=num_layers,
            )

        expected_weights = PufferNetWeights(
            encoder_weight=encoder_c.T.copy(),
            decoder_weight=decoder_c.T.copy(),
            log_std=log_std.copy(),
            recurrent_weights=tuple(recurrent.T.copy() for recurrent in recurrent_c),
            raw_float_count=raw_count,
            aligned_float_count=raw_count,
        )
        np.testing.assert_allclose(weights.encoder_weight, expected_weights.encoder_weight, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(weights.decoder_weight, expected_weights.decoder_weight, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(weights.log_std, expected_weights.log_std, rtol=0.0, atol=0.0)
        for actual, expected in zip(weights.recurrent_weights, expected_weights.recurrent_weights, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

        trainer = rl.TrainerPPO(
            obs_dim=input_dim,
            action_dim=action_dim,
            hidden_layers=(hidden_size, hidden_size),
            config=rl.ConfigPPO(
                shared_value_network=True,
                policy_network="puffer_mingru",
                manual_actor_backward=True,
                manual_critic_backward=True,
            ),
            device=device,
            seed=13,
            squash_actions=False,
            log_std_init=-3.0,
        )
        assign_puffernet_weights(trainer, weights)
        np.testing.assert_allclose(trainer.actor.log_std.numpy(), log_std, rtol=0.0, atol=0.0)

        obs_np = np.asarray(
            [[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.5, 0.2]],
            dtype=np.float32,
        )
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        out0_snapshot = wp.empty((obs_np.shape[0], action_dim + 1), dtype=wp.float32, device=device)
        out1_snapshot = wp.empty((obs_np.shape[0], action_dim + 1), dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            trainer.actor.net.zero_state()
            out0 = trainer.actor.net.forward_reuse(obs)
            wp.copy(out0_snapshot, out0)
            out1 = trainer.actor.net.forward_reuse(obs)
            wp.copy(out1_snapshot, out1)
        wp.capture_launch(capture.graph)

        expected0, state0 = puffernet_numpy_forward(weights, obs_np)
        expected1, _state1 = puffernet_numpy_forward(weights, obs_np, state=state0)
        np.testing.assert_allclose(out0_snapshot.numpy(), expected0, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(out1_snapshot.numpy(), expected1, rtol=1.0e-6, atol=1.0e-6)

    def test_mingru_backward_does_not_leak_future_grad_into_highway_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX MinGRU BPTT gradient routing tests")
        combined_np = np.asarray([[0.2, -0.4, 0.7], [-0.3, 0.5, -0.2]], dtype=np.float32)
        x_np = np.asarray([[0.25], [-0.15]], dtype=np.float32)
        grad_out_np = np.asarray([[0.0], [1.0]], dtype=np.float32)
        combined = wp.array(combined_np, dtype=wp.float32, device=device)
        x = wp.array(x_np, dtype=wp.float32, device=device)
        grad_out = wp.array(grad_out_np, dtype=wp.float32, device=device)
        grad_combined = wp.zeros_like(combined)
        grad_highway = wp.zeros_like(x)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                mingru_sequence_backward_kernel,
                dim=(1, 1),
                inputs=[combined, x, grad_out, 2, 1, 1],
                outputs=[grad_combined, grad_highway],
                device=device,
            )
        wp.capture_launch(capture.graph)

        def sigmoid(value: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-value))

        hidden = combined_np[:, 0:1]
        gate = sigmoid(combined_np[:, 1:2])
        proj = sigmoid(combined_np[:, 2:3])
        candidate = np.where(hidden >= 0.0, hidden + 0.5, sigmoid(hidden))
        candidate_grad = np.where(hidden >= 0.0, 1.0, candidate * (1.0 - candidate))
        prev = np.zeros((2, 1), dtype=np.float32)
        recurrent = np.zeros((2, 1), dtype=np.float32)
        state = np.zeros((1, 1), dtype=np.float32)
        for step in range(2):
            prev[step] = state
            state = state + gate[step : step + 1] * (candidate[step : step + 1] - state)
            recurrent[step] = state

        expected_combined = np.zeros_like(combined_np)
        expected_highway = np.zeros_like(x_np)
        grad_recurrent_next = np.zeros((1, 1), dtype=np.float32)
        for step in range(1, -1, -1):
            grad_y = grad_out_np[step : step + 1]
            grad_proj = grad_y * (recurrent[step : step + 1] - x_np[step : step + 1])
            grad_recurrent = grad_y * proj[step : step + 1] + grad_recurrent_next
            expected_highway[step] = grad_y * (1.0 - proj[step : step + 1])
            grad_gate = grad_recurrent * (candidate[step : step + 1] - prev[step : step + 1])
            grad_candidate = grad_recurrent * gate[step : step + 1]
            grad_recurrent_next = grad_recurrent * (1.0 - gate[step : step + 1])
            expected_combined[step, 0:1] = grad_candidate * candidate_grad[step : step + 1]
            expected_combined[step, 1:2] = grad_gate * gate[step : step + 1] * (1.0 - gate[step : step + 1])
            expected_combined[step, 2:3] = grad_proj * proj[step : step + 1] * (1.0 - proj[step : step + 1])

        self.assertAlmostEqual(float(expected_combined[0, 2]), 0.0, places=7)
        self.assertAlmostEqual(float(expected_highway[0, 0]), 0.0, places=7)
        self.assertGreater(abs(float(expected_combined[0, 0])) + abs(float(expected_combined[0, 1])), 1.0e-4)
        np.testing.assert_allclose(grad_combined.numpy(), expected_combined, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(grad_highway.numpy(), expected_highway, rtol=1.0e-6, atol=1.0e-6)

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
            puffer_vtrace_advantage=True,
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
            self.assertTrue(restored.config.puffer_vtrace_advantage)
            self.assertEqual(restored.actor.net.network_type, "puffer_mingru")
            for expected, actual in zip(trainer.actor.parameters(), restored.actor.parameters(), strict=True):
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.0, atol=0.0)

    def test_puffer_mingru_mirror_forward_uses_zero_sequence_state(self) -> None:
        device = require_cuda_graph_capture("PhoenX recurrent mirror PPO tests")
        pufferlib_cu = _PUFFERLIB_G1_ROOT / "src" / "pufferlib.cu"
        if not pufferlib_cu.is_file():
            raise unittest.SkipTest(f"missing PufferLib G1 reference file: {pufferlib_cu}")
        self.assertIn("puf_zero(&graph.mb_state_mir", pufferlib_cu.read_text())

        buffer = rl.BufferRollout(num_steps=2, num_envs=2, obs_dim=2, action_dim=2, device=device)
        buffer.obs.assign(
            np.asarray(
                [
                    [0.1, -0.2],
                    [0.3, -0.4],
                    [0.5, -0.6],
                    [0.7, -0.8],
                ],
                dtype=np.float32,
            )
        )
        mirror_map = rl.MirrorMapPPO(
            obs_src=(1, 0),
            obs_sign=(1.0, 1.0),
            action_src=(1, 0),
            action_sign=(1.0, 1.0),
        )
        trainer = rl.TrainerPPO(
            obs_dim=buffer.obs_dim,
            action_dim=buffer.action_dim,
            hidden_layers=(3,),
            config=rl.ConfigPPO(
                train_epochs=1,
                normalize_advantages=False,
                entropy_coeff=0.0,
                shared_value_network=True,
                policy_network="puffer_mingru",
                manual_actor_backward=True,
                manual_critic_backward=True,
                mirror_loss_coeff=0.25,
                optimizer="adam",
            ),
            device=device,
            seed=11,
            squash_actions=False,
            mirror_map=mirror_map,
        )
        trainer.reserve_update_buffers(buffer)
        prime_obs = wp.array(
            np.asarray(
                [
                    [1.0, -0.5],
                    [0.8, -0.3],
                    [0.6, -0.1],
                    [0.4, 0.1],
                ],
                dtype=np.float32,
            ),
            dtype=wp.float32,
            device=device,
        )

        with wp.ScopedCapture(device=device) as prime_capture:
            trainer.actor.net.forward_reuse(prime_obs)
        wp.capture_launch(prime_capture.graph)
        state_before = trainer.actor.net._state.numpy().copy()

        trainer._set_update_sequence_shape(buffer)
        with wp.ScopedCapture(device=device) as mirror_capture:
            mirror_obs = trainer._mirrored_obs(buffer)
            mirror_out = trainer._policy_update_reuse(mirror_obs, buffer)
        wp.capture_launch(mirror_capture.graph)
        mirror_actual = mirror_out.numpy().copy()
        np.testing.assert_allclose(trainer.actor.net._state.numpy(), state_before, rtol=0.0, atol=0.0)

        with wp.ScopedCapture(device=device) as expected_capture:
            expected_out = trainer.actor.net.forward_sequence_reuse(
                mirror_obs, num_steps=buffer.num_steps, num_envs=buffer.num_envs
            )
        wp.capture_launch(expected_capture.graph)
        np.testing.assert_allclose(mirror_actual, expected_out.numpy(), rtol=0.0, atol=0.0)

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

    def test_bf16_tiled_mlp_forward_matches_numpy_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 BF16 MLP forward tests")
        batch_size = _BF16_FORWARD_MIN_BATCH
        input_dim = DENSE_TILE_IN
        output_dim = max(64, DENSE_TILE_OUT)
        obs_np = ((np.arange(batch_size * input_dim, dtype=np.float32) % 17.0) - 8.0).reshape(
            batch_size, input_dim
        ) / np.float32(16.0)
        weight_np = ((np.arange(input_dim * output_dim, dtype=np.float32) % 13.0) - 6.0).reshape(
            input_dim, output_dim
        ) / np.float32(32.0)
        bias_np = ((np.arange(output_dim, dtype=np.float32) % 7.0) - 3.0) / np.float32(64.0)

        net = rl.WarpMLP(
            (input_dim, output_dim),
            activation="linear",
            device=device,
            seed=7,
            manual_forward_dtype="bfloat16",
        )
        net.weights[0].assign(weight_np)
        net.biases[0].assign(bias_np)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        out = net.forward_reuse(obs)
        self.assertTrue(net._uses_bf16_forward(obs, net.weights[0], batch_size))

        with wp.ScopedCapture(device=device) as capture:
            net.forward_reuse(obs)
        wp.capture_launch(capture.graph)

        expected = obs_np @ weight_np + bias_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=2.0e-5, atol=2.0e-5)

    def test_elu_mlp_forward_and_manual_backward_match_numpy_in_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX ELU MLP graph tests")
        batch_size = 4
        input_dim = 3
        hidden_dim = 5
        output_dim = 2
        obs_np = np.linspace(-0.7, 0.8, batch_size * input_dim, dtype=np.float32).reshape(batch_size, input_dim)
        w0_np = np.linspace(-0.35, 0.45, input_dim * hidden_dim, dtype=np.float32).reshape(input_dim, hidden_dim)
        b0_np = np.linspace(-0.2, 0.3, hidden_dim, dtype=np.float32)
        w1_np = np.linspace(-0.25, 0.4, hidden_dim * output_dim, dtype=np.float32).reshape(hidden_dim, output_dim)
        b1_np = np.asarray([0.15, -0.05], dtype=np.float32)
        grad_out_np = np.linspace(-0.4, 0.5, batch_size * output_dim, dtype=np.float32).reshape(batch_size, output_dim)

        net = rl.WarpMLP((input_dim, hidden_dim, output_dim), activation="elu", device=device, seed=13)
        net.weights[0].assign(w0_np)
        net.biases[0].assign(b0_np)
        net.weights[1].assign(w1_np)
        net.biases[1].assign(b1_np)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        grad_out = wp.array(grad_out_np, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            out = net.forward_manual(obs)
            net.backward_manual(grad_out)
        wp.capture_launch(capture.graph)

        hidden_pre = obs_np @ w0_np + b0_np
        hidden = np.where(hidden_pre > 0.0, hidden_pre, np.exp(hidden_pre) - 1.0).astype(np.float32)
        expected_out = hidden @ w1_np + b1_np
        grad_hidden = grad_out_np @ w1_np.T
        grad_hidden_pre = grad_hidden * np.where(hidden > 0.0, 1.0, hidden + 1.0).astype(np.float32)
        expected_w0_grad = obs_np.T @ grad_hidden_pre
        expected_b0_grad = grad_hidden_pre.sum(axis=0)
        expected_w1_grad = hidden.T @ grad_out_np
        expected_b1_grad = grad_out_np.sum(axis=0)

        np.testing.assert_allclose(out.numpy(), expected_out, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(net.weights[0].grad.numpy(), expected_w0_grad, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(net.biases[0].grad.numpy(), expected_b0_grad, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(net.weights[1].grad.numpy(), expected_w1_grad, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(net.biases[1].grad.numpy(), expected_b1_grad, rtol=1.0e-6, atol=1.0e-6)

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
        self.assertEqual(g1_recipe.VELOCITY_ITERATIONS, 2)
        self.assertEqual(env_config.velocity_iterations, g1_recipe.VELOCITY_ITERATIONS)
        self.assertEqual(env_config.actuation_model, g1_recipe.ACTUATION_MODEL)
        self.assertEqual(env_config.actuation_model, "explicit_torque")
        self.assertEqual(env_config.w_track_lin, g1_recipe.W_TRACK_LIN)
        self.assertEqual(env_config.w_action_rate, g1_recipe.W_ACTION_RATE)
        self.assertEqual(env_config.reward_mode, g1_recipe.REWARD_MODE)
        self.assertEqual(env_config.reward_mode, "nanog1_dense")
        self.assertEqual(env_config.w_sparse_command_success, g1_recipe.W_SPARSE_COMMAND_SUCCESS)
        self.assertEqual(env_config.sparse_command_velocity_tolerance, g1_recipe.SPARSE_COMMAND_VELOCITY_TOLERANCE)
        self.assertEqual(env_config.sparse_command_yaw_tolerance, g1_recipe.SPARSE_COMMAND_YAW_TOLERANCE)
        self.assertEqual(env_config.sparse_target_position, g1_recipe.SPARSE_TARGET_POSITION)
        self.assertEqual(env_config.sparse_target_radius, g1_recipe.SPARSE_TARGET_RADIUS)
        self.assertEqual(env_config.sparse_target_success_upright_cos, g1_recipe.SPARSE_TARGET_SUCCESS_UPRIGHT_COS)
        self.assertEqual(
            env_config.sparse_target_success_min_base_height, g1_recipe.SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT
        )
        self.assertEqual(
            env_config.sparse_target_success_max_base_height, g1_recipe.SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT
        )
        np.testing.assert_allclose(env.target_position.numpy()[0], np.asarray(g1_recipe.SPARSE_TARGET_POSITION))
        self.assertEqual(env_config.w_mechanical_power, g1_recipe.W_MECHANICAL_POWER)
        self.assertEqual(env_config.max_abs_root_position, g1_recipe.MAX_ABS_ROOT_POSITION)
        self.assertEqual(env_config.max_abs_root_linear_velocity, g1_recipe.MAX_ABS_ROOT_LINEAR_VELOCITY)
        self.assertEqual(env_config.max_abs_root_angular_velocity, g1_recipe.MAX_ABS_ROOT_ANGULAR_VELOCITY)
        self.assertEqual(env_config.max_abs_joint_position, g1_recipe.MAX_ABS_JOINT_POSITION)
        self.assertEqual(env_config.max_abs_joint_velocity, g1_recipe.MAX_ABS_JOINT_VELOCITY)
        self.assertEqual(env_config.gait_stance_fraction, g1_recipe.GAIT_STANCE_FRACTION)
        self.assertEqual(env_config.w_gait_contact, g1_recipe.W_GAIT_CONTACT)
        self.assertEqual(env_config.w_gait_swing, g1_recipe.W_GAIT_SWING)
        self.assertEqual(env_config.w_gait_swing_contact, g1_recipe.W_GAIT_SWING_CONTACT)
        self.assertEqual(env_config.w_gait_hip, g1_recipe.W_GAIT_HIP)
        self.assertEqual(env_config.gait_foot_height, g1_recipe.GAIT_FOOT_HEIGHT)
        self.assertEqual(env_config.w_base_height, g1_recipe.W_BASE_HEIGHT)
        self.assertEqual(env_config.base_height_target, g1_recipe.BASE_HEIGHT_TARGET)
        self.assertEqual(env_config.w_feet_air_time, g1_recipe.W_FEET_AIR_TIME)
        self.assertEqual(env_config.feet_air_time_threshold, g1_recipe.FEET_AIR_TIME_THRESHOLD)
        self.assertEqual(env_config.w_feet_slide, g1_recipe.W_FEET_SLIDE)
        self.assertEqual(
            env_config.foot_contact_normal_impulse_threshold, g1_recipe.FOOT_CONTACT_NORMAL_IMPULSE_THRESHOLD
        )
        self.assertEqual(env_config.w_joint_deviation_hip, g1_recipe.W_JOINT_DEVIATION_HIP)
        self.assertEqual(env_config.w_joint_deviation_waist, g1_recipe.W_JOINT_DEVIATION_WAIST)
        self.assertEqual(env_config.w_joint_deviation_upper, g1_recipe.W_JOINT_DEVIATION_UPPER)
        self.assertEqual(env_config.w_joint_acc_legs, g1_recipe.W_JOINT_ACC_LEGS)
        self.assertEqual(env_config.w_joint_pos_limit_ankle, g1_recipe.W_JOINT_POS_LIMIT_ANKLE)
        self.assertEqual(env_config.rigid_contact_max_per_world, g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env_config.contact_geometry, g1_recipe.CONTACT_GEOMETRY)
        self.assertEqual(env_config.ground_friction, g1_recipe.GROUND_FRICTION)
        self.assertEqual(env_config.foot_box_xy_scale, g1_recipe.FOOT_BOX_XY_SCALE)
        self.assertTrue(train_config.reset_recurrent_state_on_rollout_start)
        self.assertEqual(train_config.target_distance_start, g1_recipe.SPARSE_TARGET_CURRICULUM_START)
        self.assertEqual(train_config.target_distance_end, g1_recipe.SPARSE_TARGET_CURRICULUM_END)
        self.assertEqual(train_config.target_curriculum_samples, g1_recipe.SPARSE_TARGET_CURRICULUM_SAMPLES)
        self.assertEqual(train_config.randomize_target_positions, g1_recipe.SPARSE_TARGET_RANDOMIZE)
        self.assertEqual(train_config.target_angle_min, g1_recipe.SPARSE_TARGET_ANGLE_MIN)
        self.assertEqual(train_config.target_angle_max, g1_recipe.SPARSE_TARGET_ANGLE_MAX)
        self.assertEqual(env_config.contact_geometry, "nanog1_foot_boxes")
        self.assertEqual(env_config.threads_per_world, g1_recipe.THREADS_PER_WORLD)
        self.assertEqual(env_config.multi_world_scheduler, g1_recipe.MULTI_WORLD_SCHEDULER)
        self.assertEqual(env_config.prepare_refresh_stride, g1_recipe.PREPARE_REFRESH_STRIDE)
        expected_leg_kd = np.array([4.0, 4.0, 4.0, 6.0, 3.0, 2.2] * 2, dtype=np.float32)
        expected_leg_damping = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 0.2] * 2, dtype=np.float32)
        expected_leg_armature = np.array(
            [0.01017752004, 0.025101925, 0.01017752004, 0.025101925, 0.00721945, 0.00721945] * 2, dtype=np.float32
        )
        np.testing.assert_allclose(env.model.joint_target_kd.numpy()[6:18], expected_leg_kd, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.model.joint_damping.numpy()[6:18], expected_leg_damping, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.passive_damping.numpy()[:12], expected_leg_damping, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.model.joint_armature.numpy()[6:18], expected_leg_armature, rtol=0.0, atol=1.0e-8)
        np.testing.assert_allclose(env.model.joint_friction.numpy()[6:35], 0.1, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(
            env.model.joint_target_mode.numpy()[6:35],
            np.full(rl.ACTION_DIM_G1, int(newton.JointTargetMode.EFFORT), dtype=np.int32),
        )
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

        constraint_drive_env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(world_count=1, actuation_model="constraint_drive"), device=device
        )
        np.testing.assert_array_equal(
            constraint_drive_env.model.joint_target_mode.numpy()[6:35],
            np.full(rl.ACTION_DIM_G1, int(newton.JointTargetMode.POSITION), dtype=np.int32),
        )

    def test_command_tracking_reward_requires_upright_gate_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 command upright-gated reward tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=2,
                reward_mode="dense_sparse_command",
                command=(0.3, 0.0, 0.0),
                w_track_lin=1.0,
                w_track_ang=0.0,
                w_command_progress=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=0.0,
                w_sparse_command_success=5.0,
                w_mechanical_power=0.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
                sparse_command_velocity_tolerance=0.05,
                sparse_command_yaw_tolerance=0.05,
                min_upright_cos=0.5,
                auto_reset=False,
            ),
            device=device,
        )
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        qd = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
        q[:, 3:7] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        tilt = math.radians(45.0)
        q[1, 3:7] = np.asarray((math.sin(0.5 * tilt), 0.0, 0.0, math.cos(0.5 * tilt)), dtype=np.float32)
        qd[:, :] = 0.0
        qd[:, 0] = 0.3
        env.state_0.joint_q.assign(q.reshape(-1))
        env.state_0.joint_qd.assign(qd.reshape(-1))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        expected_upright = np.float32((1.0 + 5.0) * env.config.frame_dt)
        rewards = env.rewards.numpy()
        successes = env.successes.numpy()
        np.testing.assert_allclose(rewards[0:1], np.asarray([expected_upright], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(rewards[1:2], np.zeros(1, dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(successes, np.asarray([1.0, 0.0], dtype=np.float32), atol=1.0e-6)

    def test_sparse_command_reward_is_boolean_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 sparse command reward tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=2,
                reward_mode="sparse_command",
                command=(0.8, 0.0, 0.0),
                w_sparse_command_success=5.0,
                sparse_command_velocity_tolerance=0.35,
                sparse_command_yaw_tolerance=0.4,
                w_mechanical_power=0.0,
                auto_reset=False,
            ),
            device=device,
        )
        qd = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
        qd[:, :] = 0.0
        qd[0, 0] = 0.8
        env.state_0.joint_qd.assign(qd.reshape(-1))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.successes.numpy(), np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(env.rewards.numpy(), np.array([0.1, 0.0], dtype=np.float32), atol=1.0e-6)

    def test_dense_sparse_command_reward_adds_boolean_bonus_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 dense sparse command reward tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=2,
                reward_mode="dense_sparse_command",
                command=(0.8, 0.0, 0.0),
                w_sparse_command_success=5.0,
                sparse_command_velocity_tolerance=0.35,
                sparse_command_yaw_tolerance=0.4,
                w_track_lin=0.0,
                w_track_ang=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=1.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
                w_mechanical_power=0.0,
                auto_reset=False,
            ),
            device=device,
        )
        qd = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
        qd[:, :] = 0.0
        qd[0, 0] = 0.8
        env.state_0.joint_qd.assign(qd.reshape(-1))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.successes.numpy(), np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(env.rewards.numpy(), np.array([0.12, 0.02], dtype=np.float32), atol=1.0e-6)

    def test_sparse_target_reward_is_terminal_boolean_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 sparse target reward tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=2,
                reward_mode="sparse_target",
                command=(0.0, 0.0, 0.0),
                sparse_target_position=(0.0, 0.0),
                sparse_target_radius=0.4,
                w_sparse_command_success=5.0,
                w_mechanical_power=0.0,
                auto_reset=False,
            ),
            device=device,
        )
        env.set_target_positions(np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.successes.numpy(), np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(env.dones.numpy(), np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(env.rewards.numpy(), np.array([5.0, 0.0], dtype=np.float32), atol=1.0e-6)
        obs = env.obs.numpy()
        np.testing.assert_allclose(obs[:, 6:8], np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(obs[:, 8], np.zeros(2, dtype=np.float32))

    def test_sparse_target_progress_reward_requires_upright_gate_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 sparse target progress reward tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=2,
                reward_mode="sparse_target",
                command=(0.0, 0.0, 0.0),
                sparse_target_position=(1.0, 0.0),
                sparse_target_radius=0.1,
                w_sparse_command_success=0.0,
                w_target_progress=3.0,
                w_mechanical_power=0.0,
                auto_reset=False,
            ),
            device=device,
        )
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        qd = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
        q[:, 0:2] = np.asarray((0.0, 0.0), dtype=np.float32)
        q[:, 3:7] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        tilt = math.radians(45.0)
        q[1, 3:7] = np.asarray((math.sin(0.5 * tilt), 0.0, 0.0, math.cos(0.5 * tilt)), dtype=np.float32)
        qd[:, :] = 0.0
        qd[:, 0] = 0.5
        env.state_0.joint_q.assign(q.reshape(-1))
        env.state_0.joint_qd.assign(qd.reshape(-1))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        expected = np.asarray([3.0 * 0.5 * env.config.frame_dt, 0.0], dtype=np.float32)
        np.testing.assert_allclose(env.successes.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(env.dones.numpy(), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(env.rewards.numpy(), expected, atol=1.0e-6)

    def test_dense_target_reward_combines_progress_and_stability_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 dense target reward tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=2,
                reward_mode="dense_target",
                command=(0.0, 0.0, 0.0),
                sparse_target_position=(1.0, 0.0),
                sparse_target_radius=0.2,
                w_sparse_command_success=5.0,
                w_target_progress=3.0,
                w_track_ang=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=1.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
                w_mechanical_power=0.0,
                auto_reset=False,
            ),
            device=device,
        )
        env.set_target_positions(np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        qd = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
        q[:, 0:2] = 0.0
        q[:, 3:7] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        qd[:, :] = 0.0
        qd[0, 0] = 0.5
        env.state_0.joint_q.assign(q.reshape(-1))
        env.state_0.joint_qd.assign(qd.reshape(-1))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        expected = np.asarray(
            [
                (3.0 * 0.5 + 1.0) * env.config.frame_dt,
                5.0 + 1.0 * env.config.frame_dt,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(env.successes.numpy(), np.array([0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(env.dones.numpy(), np.array([0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(env.rewards.numpy(), expected, atol=1.0e-6)
        obs = env.obs.numpy()
        np.testing.assert_allclose(obs[:, 6:8], np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))

    def test_sparse_target_requires_balanced_posture_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 sparse target balanced success tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=2,
                reward_mode="sparse_target",
                command=(0.0, 0.0, 0.0),
                sparse_target_position=(0.0, 0.0),
                sparse_target_radius=0.4,
                sparse_target_success_upright_cos=math.cos(math.radians(30.0)),
                w_sparse_command_success=5.0,
                w_mechanical_power=0.0,
                auto_reset=False,
            ),
            device=device,
        )
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        tilt = math.radians(45.0)
        q[0, 3] = math.sin(0.5 * tilt)
        q[0, 4] = 0.0
        q[0, 5] = 0.0
        q[0, 6] = math.cos(0.5 * tilt)
        env.state_0.joint_q.assign(q.reshape(-1))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.successes.numpy(), np.array([0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(env.dones.numpy(), np.array([0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(env.rewards.numpy(), np.array([0.0, 5.0], dtype=np.float32), atol=1.0e-6)

    def test_sparse_target_randomizes_distance_range_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 sparse target distance range tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=16,
                reward_mode="sparse_target",
                sparse_target_position=(1.0, 0.0),
                auto_reset=False,
            ),
            device=device,
        )
        seed_counter = make_seed_counter(321, device=device)

        with wp.ScopedCapture(device=device) as capture:
            env.randomize_target_positions_seed_counter(
                seed_counter=seed_counter,
                target_angle_range=(0.0, 0.0),
                target_distance_min=0.5,
            )
        wp.capture_launch(capture.graph)

        targets = env.target_position.numpy()
        self.assertGreater(float(targets[:, 0].min()), 0.5 - 1.0e-6)
        self.assertLess(float(targets[:, 0].max()), 1.0 + 1.0e-6)
        self.assertGreater(float(np.ptp(targets[:, 0])), 0.05)
        np.testing.assert_allclose(targets[:, 1], np.zeros(env.world_count, dtype=np.float32), atol=1.0e-6)

    def test_sparse_target_curriculum_updates_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 sparse target curriculum tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=4,
                reward_mode="sparse_target",
                sparse_target_position=(1.0, 0.0),
                auto_reset=False,
            ),
            device=device,
        )
        sample_counter = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device=device)
        seed_counter = make_seed_counter(123, device=device)

        with wp.ScopedCapture(device=device) as capture:
            env.update_sparse_target_distance_curriculum(
                sample_counter, sample_delta=10, start_distance=0.5, end_distance=1.0, ramp_samples=20
            )
            env.randomize_target_positions_seed_counter(seed_counter=seed_counter, target_angle_range=(0.0, 0.0))
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.target_distance.numpy(), np.array([0.75], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(
            env.target_position.numpy(),
            np.tile(np.array([[0.75, 0.0]], dtype=np.float32), (env.world_count, 1)),
            atol=1.0e-6,
        )

        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.target_distance.numpy(), np.array([1.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(
            env.target_position.numpy(),
            np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (env.world_count, 1)),
            atol=1.0e-6,
        )

    def test_gait_swing_contact_penalizes_double_stance_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 gait contact reward tests")

        def reward_with_swing_contact(weight: float) -> float:
            env = rl.EnvG1PhoenX(
                g1_recipe.default_g1_env_config(
                    world_count=1,
                    w_gait_swing_contact=weight,
                    auto_reset=False,
                ),
                device=device,
            )
            env.episode_steps.assign(np.array([10], dtype=np.int32))
            env.model.collide(env.state_0, env.contacts)
            with wp.ScopedCapture(device=device) as capture:
                env.observe()
            wp.capture_launch(capture.graph)
            return float(env.rewards.numpy()[0])

        neutral_reward = reward_with_swing_contact(0.0)
        penalized_reward = reward_with_swing_contact(-2.0)
        self.assertLess(penalized_reward, neutral_reward - 0.03)

    def test_visual_mesh_mode_keeps_meshes_collision_free_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 visual mesh tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                parse_visuals=True,
                parse_meshes=False,
            ),
            device=device,
        )

        shape_types = env.model.shape_type.numpy()
        flags = env.model.shape_flags.numpy()
        labels = list(env.model.shape_label)
        mesh_mask = shape_types == int(newton.GeoType.MESH)
        collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        visible_bit = int(newton.ShapeFlags.VISIBLE)

        self.assertGreater(int(np.count_nonzero(mesh_mask)), 0)
        self.assertFalse(np.any((flags[mesh_mask] & collide_bit) != 0))
        self.assertTrue(np.any((flags[mesh_mask] & visible_bit) != 0))
        for shape_index, label in enumerate(labels):
            if label in ("left_nanog1_foot_box", "right_nanog1_foot_box"):
                self.assertNotEqual(int(flags[shape_index]) & collide_bit, 0)
                self.assertEqual(int(flags[shape_index]) & visible_bit, 0)

        with wp.ScopedCapture(device=device) as capture:
            env.model.collide(env.state_0, env.contacts)
        wp.capture_launch(capture.graph)
        self.assertGreater(int(env.contacts.rigid_contact_count.numpy()[0]), 0)

    def test_nanog1_foot_box_geometry_matches_reference_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 nanoG1 foot geometry tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                parse_visuals=True,
                parse_meshes=False,
                contact_geometry="nanog1_foot_boxes",
            ),
            device=device,
        )

        labels = list(env.model.shape_label)
        shape_body = env.model.shape_body.numpy()
        shape_scale = env.model.shape_scale.numpy()
        shape_transform = env.model.shape_transform.numpy()
        shape_mu = env.model.shape_material_mu.numpy()
        shape_flags = env.model.shape_flags.numpy()
        collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        visible_bit = int(newton.ShapeFlags.VISIBLE)
        expected_pos_quat = np.array([0.04, 0.0, -0.029, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        expected_half_extents = np.array([0.09, 0.03, 0.008], dtype=np.float32)
        expected_body_by_label = {
            "left_nanog1_foot_box": env._left_foot_body_local,
            "right_nanog1_foot_box": env._right_foot_body_local,
        }

        for label, expected_body in expected_body_by_label.items():
            shape_index = labels.index(label)
            self.assertEqual(int(shape_body[shape_index]), expected_body)
            np.testing.assert_allclose(shape_scale[shape_index], expected_half_extents, rtol=0.0, atol=1.0e-7)
            np.testing.assert_allclose(shape_transform[shape_index], expected_pos_quat, rtol=0.0, atol=1.0e-7)
            self.assertAlmostEqual(float(shape_mu[shape_index]), 0.6, places=6)
            self.assertNotEqual(int(shape_flags[shape_index]) & collide_bit, 0)
            self.assertEqual(int(shape_flags[shape_index]) & visible_bit, 0)

        for shape_index, label in enumerate(labels):
            if "ankle_roll_link_geom_" in label:
                self.assertEqual(int(shape_flags[shape_index]) & collide_bit, 0)

        with wp.ScopedCapture(device=device) as capture:
            env.model.collide(env.state_0, env.contacts)
        wp.capture_launch(capture.graph)
        self.assertGreater(int(env.contacts.rigid_contact_count.numpy()[0]), 0)

    def test_nanog1_foot_box_friction_material_matches_recipe_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 foot friction material tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(world_count=1, auto_reset=False),
            device=device,
        )

        labels = list(env.model.shape_label)
        foot_shapes = [labels.index("left_nanog1_foot_box"), labels.index("right_nanog1_foot_box")]
        shape_body = env.model.shape_body.numpy()
        ground_shape = int(np.flatnonzero(shape_body < 0)[0])
        shape_mu = env.model.shape_material_mu.numpy()
        # PhoenX currently exposes one Coulomb sliding coefficient; this
        # value is used for the static cone limit and dynamic sliding. Both
        # sides of the G1 foot-floor pair must be 0.6 because PhoenX combines
        # material friction by averaging by default, while nanoG1 authors the
        # pair coefficient directly as 0.6.
        np.testing.assert_allclose(
            shape_mu[foot_shapes],
            np.full(2, 0.6, dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertAlmostEqual(float(shape_mu[ground_shape]), env.config.ground_friction, places=6)
        self.assertAlmostEqual(float(shape_mu[ground_shape]), 0.6, places=6)

        with wp.ScopedCapture(device=device) as capture:
            env.model.collide(env.state_0, env.contacts)
        wp.capture_launch(capture.graph)

        self.assertGreater(int(env.contacts.rigid_contact_count.numpy()[0]), 0)

    def test_g1_ground_normal_force_matches_weight_in_standing_equilibrium_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 normal force balance tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(
                world_count=1,
                auto_reset=False,
                randomize_commands_on_reset=False,
                reset_noise=0.0,
                actuation_model="constraint_drive",
                max_episode_steps=0,
            ),
            device=device,
        )
        _set_g1_standing_position_hold_gains(env)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)

        settle_graph = rl.capture_env_steps(env, actions, steps_per_graph=400, warmup_steps=1)
        wp.capture_launch(settle_graph)
        settled_q = env.state_0.joint_q.numpy().copy()
        drift_graph = rl.capture_env_steps(env, actions, steps_per_graph=20, warmup_steps=0)
        wp.capture_launch(drift_graph)
        drifted_q = env.state_0.joint_q.numpy().copy()

        base_drift = float(np.max(np.abs(drifted_q[:7] - settled_q[:7])))
        joint_drift = float(np.max(np.abs(drifted_q[7 : 7 + env.action_dim] - settled_q[7 : 7 + env.action_dim])))
        self.assertLess(base_drift, 5.0e-4)
        self.assertLess(joint_drift, 5.0e-4)
        self.assertEqual(float(env.dones.numpy()[0]), 0.0)

        ground_force, npairs, npoints = _gather_g1_ground_contact_force(env)
        expected_weight = float(np.sum(env.model.body_mass.numpy(), dtype=np.float64) * _G)
        rel_err = abs(float(ground_force[2]) - expected_weight) / expected_weight

        self.assertGreaterEqual(npairs, 1)
        self.assertGreater(npoints, 0)
        self.assertLess(
            rel_err,
            0.02,
            f"ground normal Fz = {float(ground_force[2]):.3f} N vs G1 weight {expected_weight:.3f} N",
        )
        lateral = math.hypot(float(ground_force[0]), float(ground_force[1]))
        self.assertLess(lateral, 0.02 * expected_weight)

    def test_g1_enlarged_foot_static_friction_threshold_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 foot friction threshold tests")

        def run_force_case(force_ratio: float):
            env = rl.EnvG1PhoenX(
                g1_recipe.default_g1_env_config(
                    world_count=1,
                    sim_substeps=10,
                    solver_iterations=8,
                    velocity_iterations=2,
                    auto_reset=False,
                    randomize_commands_on_reset=False,
                    reset_noise=0.0,
                    actuation_model="constraint_drive",
                    max_episode_steps=0,
                    foot_box_xy_scale=2.5,
                ),
                device=device,
            )
            _set_g1_standing_position_hold_gains(env)
            actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)

            settle_graph = rl.capture_env_steps(env, actions, steps_per_graph=180, warmup_steps=1)
            wp.capture_launch(settle_graph)

            baseline_forces, baseline_counts = _gather_g1_foot_ground_contact_forces(env)
            self.assertTrue(np.all(baseline_counts > 0), msg=f"missing standing foot contacts: {baseline_counts}")
            baseline_normal = baseline_forces[:, :, 2]
            self.assertGreater(float(np.min(baseline_normal)), 50.0)

            mu = float(env.config.ground_friction)
            per_foot_limit = mu * float(np.min(baseline_normal))
            applied_force = force_ratio * per_foot_limit
            force_x = wp.array(np.asarray([applied_force], dtype=np.float32), dtype=wp.float32, device=env.device)

            def forced_step_pair() -> None:
                for _ in range(2):
                    env.state_0.clear_forces()
                    wp.launch(
                        _g1_apply_equal_foot_push_kernel,
                        dim=env.world_count,
                        inputs=[
                            force_x,
                            env.body_stride,
                            env._left_foot_body_local,
                            env._right_foot_body_local,
                        ],
                        outputs=[env.state_0.body_f],
                        device=env.device,
                    )
                    env.model.collide(env.state_0, env.contacts)
                    env.solver.step(
                        env.state_0,
                        env.state_1,
                        env.control,
                        env.contacts,
                        float(env.config.frame_dt) / float(env.config.sim_substeps),
                    )
                    env.state_0, env.state_1 = env.state_1, env.state_0

            forced_step_pair()
            with wp.ScopedCapture(device=env.device) as capture:
                forced_step_pair()
            for _ in range(10):
                wp.capture_launch(capture.graph)

            foot_forces, foot_counts = _gather_g1_foot_ground_contact_forces(env)
            normal = foot_forces[:, :, 2]
            tangent = np.linalg.norm(foot_forces[:, :, :2], axis=2)
            limit = mu * normal
            body_qd = env.state_0.body_qd.numpy()
            left_id = env._left_foot_body_local
            right_id = env._right_foot_body_local
            foot_vx = np.asarray([body_qd[left_id, 0], body_qd[right_id, 0]], dtype=np.float64)
            return applied_force, foot_forces, foot_counts, normal, tangent, limit, foot_vx

        low_force, _low_forces, low_counts, low_normal, low_tangent, low_limit, low_vx = run_force_case(0.35)
        high_force, _high_forces, high_counts, high_normal, high_tangent, high_limit, high_vx = run_force_case(1.35)

        self.assertTrue(np.all(low_counts > 0), msg=f"missing low-force foot contacts: {low_counts}")
        self.assertTrue(np.all(high_counts > 0), msg=f"missing high-force foot contacts: {high_counts}")
        self.assertGreater(float(np.min(low_normal)), 50.0)
        self.assertGreater(float(np.min(high_normal)), 50.0)

        np.testing.assert_allclose(low_tangent[0], low_force, rtol=0.08, atol=2.0)
        self.assertTrue(np.all(low_tangent[0] < 0.50 * low_limit[0]))
        self.assertLess(float(np.max(np.abs(low_vx))), 0.02)

        self.assertTrue(
            np.all(high_tangent[0] <= 1.20 * high_limit[0] + 5.0),
            msg=f"high-force tangent exceeded Coulomb limit: tangent={high_tangent[0]}, limit={high_limit[0]}",
        )
        self.assertTrue(
            np.all(high_tangent[0] >= 0.85 * high_limit[0]),
            msg=f"high-force contacts did not approach friction limit: tangent={high_tangent[0]}, limit={high_limit[0]}",
        )
        self.assertGreater(float(np.mean(high_vx)), max(0.10, 5.0 * float(np.max(np.abs(low_vx)))))
        self.assertGreater(high_force, float(np.max(high_limit[0])))

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

    def test_rejects_unknown_observation_mode(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL observation-mode tests")
        config = rl.ConfigEnvG1PhoenX(world_count=1, observation_mode="unknown")

        with self.assertRaisesRegex(ValueError, "observation_mode"):
            rl.EnvG1PhoenX(config, device=device)

    def test_isaaclab_flat_observation_mode_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 RL IsaacLab observation tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=2,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                observation_mode="isaaclab_flat",
            ),
            device=device,
        )
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        graph = rl.capture_env_steps(env, actions, steps_per_graph=1, warmup_steps=1)
        wp.capture_launch(graph)

        self.assertEqual(rl.OBS_DIM_G1, rl.OBS_DIM_G1_NANOG1)
        self.assertEqual(env.obs.shape, (2, rl.OBS_DIM_G1_ISAACLAB_FLAT))
        obs = env.obs.numpy()
        self.assertTrue(np.isfinite(obs).all())
        np.testing.assert_allclose(obs[:, 70:], 0.0, rtol=0.0, atol=0.0)

    def test_step_graph_capture_shapes_and_masks_actions(self) -> None:
        env = _g1_test_env(world_count=2)
        actions_np = np.full((env.world_count, env.action_dim), 1.5, dtype=np.float32)
        actions = wp.array(actions_np, dtype=wp.float32, device=env.device)

        graph = rl.capture_env_steps(env, actions, steps_per_graph=2, warmup_steps=1)
        wp.capture_launch(graph)

        self.assertFalse(env.solver.world.articulation_dvi_host)
        self.assertFalse(env.solver.world.articulation_dvi_replaces_joint_pgs)
        self.assertIsNone(env.solver.world.articulation_topology)
        self.assertEqual(env.solver.world.rigid_contact_max, env.world_count * g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env.contacts.rigid_contact_max, env.world_count * g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        self.assertEqual(env.obs.shape, (2, rl.OBS_DIM_G1))
        self.assertEqual(env.rewards.shape, (2,))
        self.assertEqual(env.dones.shape, (2,))
        self.assertEqual(env.actuator_force.shape, (2, rl.ACTION_DIM_G1))
        self.assertTrue(np.isfinite(env.obs.numpy()).all())
        self.assertTrue(np.isfinite(env.rewards.numpy()).all())
        self.assertTrue(np.isfinite(env.dones.numpy()).all())
        self.assertTrue(np.isfinite(env.actuator_force.numpy()).all())

        current_actions = env.current_actions.numpy()
        np.testing.assert_allclose(current_actions[:, :12], 1.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(current_actions[:, 12:], 0.0, rtol=0.0, atol=0.0)

    def test_actuator_force_matches_nanog1_model_formula_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 actuator-force gather tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=device,
        )
        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        q_before = q.copy()
        qd_before = qd.copy()
        q_before[7 : 7 + rl.ACTION_DIM_G1] = env.default_joint_pos.numpy()
        q_before[8] = env.default_joint_pos.numpy()[1] - np.float32(3.0)
        q_before[20] = env.default_joint_pos.numpy()[13] + np.float32(1.5)
        qd_before[:] = 0.0
        qd_before[6 : 6 + rl.ACTION_DIM_G1] = np.linspace(-3.0, 2.0, rl.ACTION_DIM_G1, dtype=np.float32)
        env.state_0.joint_q.assign(q_before)
        env.state_0.joint_qd.assign(qd_before)

        actions_np = np.zeros((1, rl.ACTION_DIM_G1), dtype=np.float32)
        actions_np[0, : g1_recipe.CONTROLLED_ACTION_COUNT] = np.linspace(0.8, -0.8, 12, dtype=np.float32)
        actions = wp.array(actions_np, dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            env.step(actions)
        wp.capture_launch(capture.graph)

        expected_actions = np.clip(actions_np[0], -1.0, 1.0)
        expected_actions[g1_recipe.CONTROLLED_ACTION_COUNT :] = 0.0
        target = env.default_joint_pos.numpy() + np.float32(env.config.action_scale) * expected_actions
        target = np.clip(target, env.ctrl_lower.numpy(), env.ctrl_upper.numpy())
        expected_force = env.actuator_force_kp.numpy() * (target - q_before[7 : 7 + rl.ACTION_DIM_G1])
        expected_force -= env.actuator_force_kd.numpy() * qd_before[6 : 6 + rl.ACTION_DIM_G1]
        expected_force = np.clip(expected_force, env.actuator_force_lower.numpy(), env.actuator_force_upper.numpy())
        expected_joint_f = expected_force - env.passive_damping.numpy() * qd_before[6 : 6 + rl.ACTION_DIM_G1]
        np.testing.assert_allclose(env.actuator_force.numpy()[0], expected_force, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            env.control.joint_f.numpy()[6 : 6 + rl.ACTION_DIM_G1], expected_joint_f, rtol=1.0e-6, atol=1.0e-6
        )
        self.assertEqual(float(env.actuator_force.numpy()[0, 1]), float(env.actuator_force_upper.numpy()[1]))
        self.assertEqual(float(env.actuator_force.numpy()[0, 13]), float(env.actuator_force_lower.numpy()[13]))

        old_drive_proxy = env.actuator_ke.numpy() * (target - q_before[7 : 7 + rl.ACTION_DIM_G1])
        old_drive_proxy -= env.actuator_kd.numpy() * qd_before[6 : 6 + rl.ACTION_DIM_G1]
        self.assertGreater(abs(float(old_drive_proxy[13] - expected_force[13])), 1.0)

        env.dones.assign(np.ones(env.world_count, dtype=np.float32))
        seed_counter = make_seed_counter(11, device=device)
        with wp.ScopedCapture(device=device) as reset_capture:
            env.reset_done_seed_counter(seed_counter)
        wp.capture_launch(reset_capture.graph)
        np.testing.assert_allclose(
            env.actuator_force.numpy(), np.zeros_like(env.actuator_force.numpy()), rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            env.control.joint_f.numpy(),
            np.zeros_like(env.control.joint_f.numpy()),
            rtol=0.0,
            atol=0.0,
        )

    def test_explicit_torque_matches_analytical_g1_drive_forces_joint_by_joint_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 joint-by-joint drive-force tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=rl.ACTION_DIM_G1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
            ),
            device=device,
        )
        self.assertEqual(env.config.actuation_model, "explicit_torque")
        np.testing.assert_array_equal(
            env.model.joint_target_mode.numpy()[6:35],
            np.full(rl.ACTION_DIM_G1, int(newton.JointTargetMode.EFFORT), dtype=np.int32),
        )

        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        qd = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
        q_before = q.copy()
        qd_before = qd.copy()
        q_before[:, 7 : 7 + rl.ACTION_DIM_G1] = env.default_joint_pos.numpy()
        qd_before[:, :] = 0.0
        for world in range(env.world_count):
            q_before[world, 7 + world] += np.float32(0.05)
            qd_before[world, 6 : 6 + rl.ACTION_DIM_G1] = np.linspace(-1.5, 1.5, rl.ACTION_DIM_G1, dtype=np.float32)
        env.state_0.joint_q.assign(q_before.reshape(-1))
        env.state_0.joint_qd.assign(qd_before.reshape(-1))

        actions_np = np.zeros((env.world_count, env.action_dim), dtype=np.float32)
        actions = wp.array(actions_np, dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            env.step(actions)
        wp.capture_launch(capture.graph)

        expected_force = env.actuator_force_kp.numpy()[None, :] * (
            env.default_joint_pos.numpy()[None, :] - q_before[:, 7 : 7 + rl.ACTION_DIM_G1]
        )
        expected_force -= env.actuator_force_kd.numpy()[None, :] * qd_before[:, 6 : 6 + rl.ACTION_DIM_G1]
        expected_force = np.clip(expected_force, env.actuator_force_lower.numpy(), env.actuator_force_upper.numpy())
        expected_joint_f = (
            expected_force - env.passive_damping.numpy()[None, :] * qd_before[:, 6 : 6 + rl.ACTION_DIM_G1]
        )
        joint_f = env.control.joint_f.numpy().reshape(env.world_count, env.dof_stride)
        np.testing.assert_allclose(env.actuator_force.numpy(), expected_force, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(joint_f[:, :6], 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(joint_f[:, 6 : 6 + rl.ACTION_DIM_G1], expected_joint_f, rtol=1.0e-6, atol=1.0e-6)

    def test_constraint_drive_softness_matches_implicit_pd_coefficients_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 implicit-drive coefficient tests")
        perturb = np.float32(0.05)
        substep_dt = np.float32(g1_recipe.FRAME_DT / g1_recipe.SIM_SUBSTEPS)
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=rl.ACTION_DIM_G1,
                frame_dt=float(substep_dt),
                sim_substeps=1,
                solver_iterations=g1_recipe.SOLVER_ITERATIONS,
                velocity_iterations=0,
                max_episode_steps=0,
                auto_reset=False,
                actuation_model="constraint_drive",
            ),
            device=device,
        )
        self.assertEqual(env.config.actuation_model, "constraint_drive")

        q = env.model.joint_q.numpy().reshape(env.world_count, env.coord_stride).copy()
        qd = env.model.joint_qd.numpy().reshape(env.world_count, env.dof_stride).copy()
        qd[:, :] = 0.0
        q[:, 2] = 3.0
        for world in range(env.world_count):
            q[world, 7 + world] += perturb
        env.state_0.joint_q.assign(q.reshape(-1))
        env.state_0.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)

        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            env.step(actions)
        wp.capture_launch(capture.graph)

        cids = env.solver._adbs.drive_cid.numpy().reshape(env.world_count, env.action_dim)
        data = env.solver._constraints.data.numpy()
        eff_inv = data[int(_OFF_EFF_INV_AXIAL), cids]
        gamma = data[int(_OFF_GAMMA_DRIVE), cids]
        bias = data[int(_OFF_BIAS_DRIVE), cids]
        eff_mass_soft = data[int(_OFF_EFF_MASS_DRIVE_SOFT), cids]
        acc_drive = data[int(_OFF_ACC_DRIVE), cids]

        kp = env.actuator_ke.numpy()[None, :]
        kd = env.actuator_kd.numpy()[None, :]
        drive_c = np.zeros((env.world_count, env.action_dim), dtype=np.float32)
        np.fill_diagonal(drive_c, perturb)
        boost = np.float32(float(PHOENX_BOOST_REVOLUTE_DRIVE))
        k_max = boost / (eff_inv * substep_dt * substep_dt)
        k_clamped = np.minimum(kp, k_max)
        softness = np.float32(1.0) / (kd + substep_dt * k_clamped)
        expected_gamma = softness / substep_dt
        expected_bias = drive_c * substep_dt * k_clamped * softness / substep_dt
        expected_eff_mass = np.float32(1.0) / (eff_inv + expected_gamma)

        np.testing.assert_allclose(gamma, expected_gamma, rtol=1.0e-6, atol=1.0e-5)
        np.testing.assert_allclose(bias, expected_bias, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(eff_mass_soft, expected_eff_mass, rtol=1.0e-6, atol=1.0e-7)

        coefficient_force_ratio = np.diag(expected_eff_mass / (substep_dt * (kd + substep_dt * k_clamped)))
        observed_force_ratio = np.diag(acc_drive) / (substep_dt * env.actuator_ke.numpy() * perturb)
        self.assertLess(float(np.max(coefficient_force_ratio)), 0.75)
        self.assertLess(float(np.mean(coefficient_force_ratio)), 0.45)
        self.assertLess(float(np.max(observed_force_ratio)), 0.95)
        self.assertLess(float(np.mean(observed_force_ratio)), 0.65)

    def test_graph_replay_advances_policy_steps(self) -> None:
        env = _g1_test_env(world_count=1)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)

        graph = rl.capture_env_steps(env, actions, steps_per_graph=2, warmup_steps=1)
        before = int(env.episode_steps.numpy()[0])
        for _ in range(3):
            wp.capture_launch(graph)
        after = int(env.episode_steps.numpy()[0])

        self.assertEqual(after - before, 6)
        self.assertTrue(np.isfinite(env.obs.numpy()).all())

    def test_zero_action_full_g1_solver_survives_short_graph_rollout(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 zero-action full solver tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(world_count=2, auto_reset=False, max_episode_steps=0),
            device=device,
        )
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            for _ in range(8):
                env.step(actions)
        wp.capture_launch(capture.graph)

        self.assertFalse(np.any(env.dones.numpy() > 0.0))
        self.assertTrue(np.isfinite(env.obs.numpy()).all())
        self.assertTrue(np.isfinite(env.rewards.numpy()).all())
        self.assertTrue(np.all(env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)[:, 2] > 0.5))

    def test_reset_clears_poisoned_adbs_runtime_state_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 poisoned ADBS reset tests")
        env = rl.EnvG1PhoenX(
            g1_recipe.default_g1_env_config(world_count=32, auto_reset=False, max_episode_steps=0),
            device=device,
        )
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
        joint_count = env.solver.world.num_joints
        bad_before_reset = wp.zeros(joint_count, dtype=wp.int32, device=device)
        bad_after_reset = wp.zeros(joint_count, dtype=wp.int32, device=device)
        bad_after_step = wp.zeros(joint_count, dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _poison_adbs_reset_runtime_state_kernel,
                dim=joint_count,
                inputs=[env.solver.world.constraints, env.solver.world.bodies, joint_count],
                device=device,
            )
            wp.launch(
                _adbs_runtime_nonfinite_flags_kernel,
                dim=joint_count,
                inputs=[
                    env.solver.world.constraints,
                    env.solver.world.bodies,
                    joint_count,
                    wp.int32(1),
                    bad_before_reset,
                ],
                device=device,
            )
            env.reset()
            wp.launch(
                _adbs_runtime_nonfinite_flags_kernel,
                dim=joint_count,
                inputs=[
                    env.solver.world.constraints,
                    env.solver.world.bodies,
                    joint_count,
                    wp.int32(0),
                    bad_after_reset,
                ],
                device=device,
            )
            env.step(actions)
            wp.launch(
                _adbs_runtime_nonfinite_flags_kernel,
                dim=joint_count,
                inputs=[
                    env.solver.world.constraints,
                    env.solver.world.bodies,
                    joint_count,
                    wp.int32(1),
                    bad_after_step,
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)

        self.assertTrue(np.any(bad_before_reset.numpy() != 0))
        self.assertFalse(np.any(bad_after_reset.numpy() != 0))
        self.assertFalse(np.any(bad_after_step.numpy() != 0))
        self.assertFalse(np.any(env.step_dones.numpy() > 0.0))
        self.assertTrue(np.isfinite(env.state_0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(env.state_0.joint_qd.numpy()).all())
        self.assertTrue(np.isfinite(env.solver.world.bodies.velocity.numpy()).all())
        self.assertTrue(np.isfinite(env.solver.world.bodies.angular_velocity.numpy()).all())

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

        commands = env.command.numpy()
        self.assertTrue(np.all(commands[:, 0] >= command_x_range[0]))
        self.assertTrue(np.all(commands[:, 0] <= command_x_range[1]))
        self.assertTrue(np.all(commands[:, 1] >= command_y_range[0]))
        self.assertTrue(np.all(commands[:, 1] <= command_y_range[1]))
        self.assertTrue(np.all(commands[:, 2] >= command_yaw_range[0]))
        self.assertTrue(np.all(commands[:, 2] <= command_yaw_range[1]))

    def test_command_curriculum_scales_sampled_commands_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 command curriculum tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(world_count=2, sim_substeps=1, solver_iterations=1, max_episode_steps=0),
            device=device,
        )
        sample_counter = wp.array(np.zeros(1, dtype=np.int32), dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            env.update_command_curriculum(sample_counter, sample_delta=20, start_scale=0.4, ramp_samples=40.0)
            env.randomize_commands(
                seed=17,
                command_x_range=(1.0, 1.0),
                command_y_range=(0.6, 0.6),
                command_yaw_range=(-1.0, -1.0),
                zero_probability=0.0,
            )

        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(env.command_scale.numpy(), np.array([0.7], dtype=np.float32), rtol=0.0, atol=1.0e-6)
        expected = np.array([[0.7, 0.42, -0.7], [0.7, 0.42, -0.7]], dtype=np.float32)
        np.testing.assert_allclose(env.command.numpy(), expected, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(sample_counter.numpy(), np.array([20], dtype=np.int32))

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

    def test_reset_done_masks_fk_to_done_worlds_inside_graph(self) -> None:
        env = _g1_test_env(world_count=2)
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        body_q_before = env.state_0.body_q.numpy().copy()
        q[1, 2] += np.float32(1.0)
        env.state_0.joint_q.assign(q.reshape(-1))
        env.dones.assign(np.array([1.0, 0.0], dtype=np.float32))
        seed_counter = make_seed_counter(33, device=env.device)

        with wp.ScopedCapture(device=env.device) as capture:
            env.reset_done_seed_counter(seed_counter=seed_counter)

        wp.capture_launch(capture.graph)
        body_q_after = env.state_0.body_q.numpy()
        mask = env._reset_articulation_mask.numpy()
        non_done_root = env.body_stride

        np.testing.assert_array_equal(mask, np.array([True, False]))
        np.testing.assert_allclose(body_q_after[non_done_root], body_q_before[non_done_root], rtol=0.0, atol=0.0)
        self.assertGreater(abs(float(q[1, 2] - body_q_after[non_done_root, 2])), 0.5)

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

    def test_command_progress_reward_is_optional_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 command progress reward tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                command=(0.5, 0.0, 0.0),
                w_track_lin=0.0,
                w_track_ang=0.0,
                w_command_progress=2.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=0.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
            ),
            device=device,
        )
        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        q[3:7] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        qd[:] = 0.0
        qd[0] = 0.4
        env.state_0.joint_q.assign(q)
        env.state_0.joint_qd.assign(qd)

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        expected = np.float32(2.0 * 0.5 * 0.4 * env.config.frame_dt)
        np.testing.assert_allclose(env.rewards.numpy(), np.asarray([expected], dtype=np.float32), atol=1.0e-6)

    def test_reward_decomposition_matches_nanog1_v3_equations_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 reward decomposition tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=100,
                auto_reset=False,
                w_torque=-0.01,
            ),
            device=device,
        )
        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        default_joint_pos = env.default_joint_pos.numpy()
        q[:] = 0.0
        q[2] = 0.79
        q[3:7] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        joint_delta = np.linspace(-0.015, 0.015, rl.ACTION_DIM_G1, dtype=np.float32)
        q[7 : 7 + rl.ACTION_DIM_G1] = default_joint_pos + joint_delta
        qd[:] = 0.0
        qd[0:6] = np.asarray([0.35, -0.2, 0.15, 0.4, -0.3, 0.2], dtype=np.float32)
        qd[6 : 6 + rl.ACTION_DIM_G1] = np.linspace(-0.25, 0.35, rl.ACTION_DIM_G1, dtype=np.float32)
        current_actions_np = np.zeros((1, rl.ACTION_DIM_G1), dtype=np.float32)
        previous_actions_np = np.zeros((1, rl.ACTION_DIM_G1), dtype=np.float32)
        current_actions_np[0, : g1_recipe.CONTROLLED_ACTION_COUNT] = np.linspace(
            -0.2, 0.2, g1_recipe.CONTROLLED_ACTION_COUNT, dtype=np.float32
        )
        previous_actions_np[0, : g1_recipe.CONTROLLED_ACTION_COUNT] = np.linspace(
            0.1, -0.1, g1_recipe.CONTROLLED_ACTION_COUNT, dtype=np.float32
        )
        command_np = np.asarray([[0.1, -0.05, 0.0]], dtype=np.float32)
        actuator_force_np = np.linspace(-14.0, 17.0, rl.ACTION_DIM_G1, dtype=np.float32).reshape(1, rl.ACTION_DIM_G1)
        episode_step = 5
        labels = list(env.model.shape_label)
        left_shape = labels.index("left_nanog1_foot_box")
        ground_shape = int(np.flatnonzero(env.model.shape_body.numpy() < 0)[0])
        shape0 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape1 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape0[0] = left_shape
        shape1[0] = ground_shape
        env.contacts.rigid_contact_count.assign(np.asarray([1], dtype=np.int32))
        env.contacts.rigid_contact_shape0.assign(shape0)
        env.contacts.rigid_contact_shape1.assign(shape1)
        _set_g1_contact_normal_impulses(env, [1.0])

        env.state_0.joint_q.assign(q)
        env.state_0.joint_qd.assign(qd)
        env.current_actions.assign(current_actions_np)
        env.previous_actions.assign(previous_actions_np)
        env.actuator_force.assign(actuator_force_np)
        env.command.assign(command_np)
        env.episode_steps.assign(np.asarray([episode_step], dtype=np.int32))
        with wp.ScopedCapture(device=device) as capture:
            newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
            env.observe()
        wp.capture_launch(capture.graph)

        body_q = env.state_0.body_q.numpy()
        right_foot_z = float(body_q[env._right_foot_body_local, 2])
        root_com = env.model.body_com.numpy().reshape(env.world_count, env.body_stride, 3)[:, 0, :].copy()
        lin_b = _g1_root_origin_linear_velocity_body_np(
            q.reshape(1, env.coord_stride), qd.reshape(1, env.dof_stride), root_com
        )[0]
        lin_com_b = _quat_rotate_inverse_xyzw_np(q[3:7].reshape(1, 4), qd[0:3].reshape(1, 3))[0]
        ang = _quat_rotate_inverse_xyzw_np(q[3:7].reshape(1, 4), qd[3:6].reshape(1, 3))[0]
        self.assertGreater(float(np.linalg.norm(lin_b - lin_com_b)), 0.005)
        self.assertGreater(float(np.linalg.norm(lin_b - qd[0:3])), 0.01)
        vx_err = command_np[0, 0] - lin_b[0]
        vy_err = command_np[0, 1] - lin_b[1]
        yaw_err = command_np[0, 2] - ang[2]
        track_lin = math.exp(-float(vx_err * vx_err + vy_err * vy_err) / 0.25)
        track_ang = math.exp(-float(yaw_err * yaw_err) / 0.25)
        lin_vel_z_penalty = float(lin_b[2] * lin_b[2])
        ang_vel_xy_penalty = float(ang[0] * ang[0] + ang[1] * ang[1])
        obs_np = env.obs.numpy()[0]
        gravity_b = obs_np[3:6]
        upright_cos = float(np.clip(-gravity_b[2], 0.0, 1.0))
        upright_gate = float(np.clip((upright_cos - 0.75) * 4.0, 0.0, 1.0))
        orientation_penalty = float(gravity_b[0] * gravity_b[0] + gravity_b[1] * gravity_b[1])
        action_rate_penalty = float(np.sum((current_actions_np[0] - previous_actions_np[0]) ** 2, dtype=np.float32))
        actuator_ke = env.actuator_ke.numpy()
        actuator_kd = env.actuator_kd.numpy()
        targets = default_joint_pos + current_actions_np[0] * np.float32(env.config.action_scale)
        tau_proxy = actuator_ke * (targets - q[7 : 7 + rl.ACTION_DIM_G1]) - actuator_kd * qd[6 : 6 + rl.ACTION_DIM_G1]
        proxy_sq_penalty = float(np.sum(tau_proxy * tau_proxy, dtype=np.float32))
        torque_sq_penalty = float(np.sum(actuator_force_np[0] * actuator_force_np[0], dtype=np.float32))
        self.assertGreater(abs(proxy_sq_penalty - torque_sq_penalty), 100.0)

        left_phase = float(episode_step % env.config.phase_period) / float(env.config.phase_period)
        right_phase = left_phase + 0.5
        if right_phase >= 1.0:
            right_phase -= 1.0
        gait_reward = 0.0
        gait_reward += env.config.w_gait_contact if left_phase < env.config.gait_stance_fraction else 0.0
        gait_reward += env.config.w_gait_contact if right_phase >= env.config.gait_stance_fraction else 0.0
        right_dz = right_foot_z - env.config.gait_foot_height
        gait_reward += env.config.w_gait_swing * right_dz * right_dz
        hip_indices = np.asarray([1, 2, 7, 8], dtype=np.int32)
        hip_penalty = float(np.sum(joint_delta[hip_indices] * joint_delta[hip_indices], dtype=np.float32))
        gait_reward += env.config.w_gait_hip * hip_penalty
        base_dz = float(q[2] - env.config.base_height_target)
        gait_reward += env.config.w_base_height * base_dz * base_dz

        shaped_reward = (
            env.config.w_track_lin * track_lin * upright_gate
            + env.config.w_track_ang * track_ang * upright_gate
            + env.config.w_lin_vel_z * lin_vel_z_penalty
            + env.config.w_ang_vel_xy * ang_vel_xy_penalty * upright_gate
            + env.config.w_orientation * orientation_penalty * upright_gate
            + env.config.w_torque * torque_sq_penalty
            + env.config.w_action_rate * action_rate_penalty
            + gait_reward
            + env.config.w_alive
        )
        expected_reward = np.float32(shaped_reward * env.config.frame_dt)
        expected_phase = 2.0 * math.pi * float(episode_step % env.config.phase_period) / float(env.config.phase_period)

        np.testing.assert_allclose(
            env.rewards.numpy(), np.asarray([expected_reward], dtype=np.float32), rtol=2.0e-5, atol=2.0e-6
        )
        np.testing.assert_allclose(
            env.successes.numpy(), np.asarray([track_lin * upright_gate], dtype=np.float32), rtol=1.0e-6, atol=1.0e-6
        )
        np.testing.assert_allclose(env.dones.numpy(), np.zeros(1, dtype=np.float32), rtol=0.0, atol=0.0)
        obs = obs_np
        np.testing.assert_allclose(obs[6:9], command_np[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(obs[67:96], current_actions_np[0], rtol=0.0, atol=0.0)
        self.assertAlmostEqual(float(obs[96]), math.sin(expected_phase), places=6)
        self.assertAlmostEqual(float(obs[97]), math.cos(expected_phase), places=6)

    def test_g1_foot_contact_scan_requires_normal_impulse_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 active foot-contact tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                command=(0.0, 0.0, 0.0),
                w_track_lin=0.0,
                w_track_ang=0.0,
                w_command_progress=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=0.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
            ),
            device=device,
        )
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
        _set_g1_contact_normal_impulses(env, [0.0, 0.5 * env.config.foot_contact_normal_impulse_threshold])

        with wp.ScopedCapture(device=device) as gap_capture:
            env.observe()
        wp.capture_launch(gap_capture.graph)

        np.testing.assert_allclose(env.foot_contacts.numpy()[0], np.zeros(2, dtype=np.float32), rtol=0.0, atol=0.0)

        _set_g1_contact_normal_impulses(
            env,
            [
                2.0 * env.config.foot_contact_normal_impulse_threshold,
                3.0 * env.config.foot_contact_normal_impulse_threshold,
            ],
        )
        with wp.ScopedCapture(device=device) as active_capture:
            env.observe()
        wp.capture_launch(active_capture.graph)

        np.testing.assert_allclose(env.foot_contacts.numpy()[0], np.ones(2, dtype=np.float32), rtol=0.0, atol=0.0)

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
        _set_g1_contact_normal_impulses(env, [1.0, 1.0])

        with wp.ScopedCapture(device=device) as capture:
            env.observe()

        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(env.foot_contacts.numpy()[0], np.array([1.0, 1.0], dtype=np.float32))

        base_z = float(env.state_0.joint_q.numpy()[2])
        expected_shaped = 2.0 * env.config.w_gait_contact
        expected_shaped += env.config.w_base_height * (base_z - env.config.base_height_target) ** 2
        self.assertAlmostEqual(float(env.rewards.numpy()[0]), expected_shaped * env.config.frame_dt, places=5)
        self.assertEqual(float(env.dones.numpy()[0]), 0.0)

    def test_isaaclab_feet_air_time_positive_biped_reward_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 IsaacLab feet air-time reward tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                command=(0.5, 0.0, 0.0),
                w_track_lin=0.0,
                w_track_ang=0.0,
                w_command_progress=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=0.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
                w_feet_air_time=1.0,
                feet_air_time_threshold=0.4,
                w_feet_slide=0.0,
            ),
            device=device,
        )
        labels = list(env.model.shape_label)
        left_shape = labels.index("left_nanog1_foot_box")
        ground_shape = int(np.flatnonzero(env.model.shape_body.numpy() < 0)[0])
        shape0 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape1 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape0[0] = left_shape
        env.contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        env.contacts.rigid_contact_shape0.assign(shape0)
        env.contacts.rigid_contact_shape1.assign(shape1)
        _set_g1_contact_normal_impulses(env, [1.0])
        env.episode_steps.assign(np.array([1], dtype=np.int32))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
            env.observe()
        wp.capture_launch(capture.graph)

        expected_time = np.float32(env.config.frame_dt)
        np.testing.assert_allclose(env.foot_contact_time.numpy(), np.array([[expected_time, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(env.foot_air_time.numpy(), np.array([[0.0, expected_time]], dtype=np.float32))
        np.testing.assert_allclose(env.foot_timer_episode_step.numpy(), np.array([1], dtype=np.int32))
        expected_reward = np.float32(expected_time * env.config.frame_dt)
        np.testing.assert_allclose(env.rewards.numpy(), np.array([expected_reward], dtype=np.float32), atol=1.0e-7)

        env.dones.assign(np.ones(1, dtype=np.float32))
        with wp.ScopedCapture(device=device) as reset_capture:
            env.reset_done()
        wp.capture_launch(reset_capture.graph)
        np.testing.assert_allclose(env.foot_air_time.numpy(), np.zeros((1, 2), dtype=np.float32))
        np.testing.assert_allclose(env.foot_contact_time.numpy(), np.zeros((1, 2), dtype=np.float32))
        np.testing.assert_allclose(env.foot_timer_episode_step.numpy(), np.array([-1], dtype=np.int32))

    def test_isaaclab_feet_slide_penalty_uses_contact_body_velocity_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 IsaacLab feet slide reward tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                command=(0.0, 0.0, 0.0),
                w_track_lin=0.0,
                w_track_ang=0.0,
                w_command_progress=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=0.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
                w_feet_air_time=0.0,
                w_feet_slide=-2.0,
            ),
            device=device,
        )
        labels = list(env.model.shape_label)
        left_shape = labels.index("left_nanog1_foot_box")
        ground_shape = int(np.flatnonzero(env.model.shape_body.numpy() < 0)[0])
        shape0 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape1 = np.full(int(env.contacts.rigid_contact_max), ground_shape, dtype=np.int32)
        shape0[0] = left_shape
        env.contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        env.contacts.rigid_contact_shape0.assign(shape0)
        env.contacts.rigid_contact_shape1.assign(shape1)
        _set_g1_contact_normal_impulses(env, [1.0])

        body_qd = env.state_0.body_qd.numpy()
        body_qd[env._left_foot_body_local, :6] = np.array([0.3, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        env.state_0.body_qd.assign(body_qd)
        env.episode_steps.assign(np.array([1], dtype=np.int32))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        expected_reward = np.float32(-2.0 * 0.5 * env.config.frame_dt)
        np.testing.assert_allclose(env.rewards.numpy(), np.array([expected_reward], dtype=np.float32), atol=1.0e-6)

    def test_isaaclab_joint_regularizers_inside_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 IsaacLab joint regularizer tests")
        env = rl.EnvG1PhoenX(
            rl.ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                max_episode_steps=0,
                auto_reset=False,
                command=(0.0, 0.0, 0.0),
                w_track_lin=0.0,
                w_track_ang=0.0,
                w_command_progress=0.0,
                w_lin_vel_z=0.0,
                w_ang_vel_xy=0.0,
                w_orientation=0.0,
                w_torque=0.0,
                w_action_rate=0.0,
                w_alive=0.0,
                w_gait_contact=0.0,
                w_gait_swing=0.0,
                w_gait_swing_contact=0.0,
                w_gait_hip=0.0,
                w_base_height=0.0,
                w_feet_air_time=0.0,
                w_feet_slide=0.0,
                w_joint_deviation_hip=-1.0,
                w_joint_deviation_waist=-2.0,
                w_joint_deviation_upper=-0.5,
                w_joint_acc_legs=-1.0e-4,
                w_joint_pos_limit_ankle=-3.0,
            ),
            device=device,
        )

        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        default = env.default_joint_pos.numpy()
        ctrl_lower = env.ctrl_lower.numpy()
        ctrl_upper = env.ctrl_upper.numpy()

        hip_delta = {1: 0.1, 2: -0.2, 7: 0.05, 8: -0.1}
        waist_delta = {12: 0.2, 13: -0.1, 14: 0.05}
        upper_delta = {15: 0.2, 22: -0.1}
        all_delta = {**hip_delta, **waist_delta, **upper_delta}
        for joint, delta in all_delta.items():
            q[7 + joint] = default[joint] + np.float32(delta)
        q[7 + 4] = ctrl_upper[4] + np.float32(0.02)
        q[7 + 10] = ctrl_lower[10] - np.float32(0.03)

        leg_acc_indices = np.array([0, 1, 2, 3, 6, 7, 8, 9], dtype=np.int32)
        qd_values = np.linspace(0.001, 0.008, leg_acc_indices.size, dtype=np.float32)
        qd[6 + leg_acc_indices] = qd_values
        env.state_0.joint_q.assign(q)
        env.state_0.joint_qd.assign(qd)
        env.previous_joint_qd.zero_()
        env.joint_acc_l2.zero_()
        env.joint_regularizer_episode_step.assign(np.array([-1], dtype=np.int32))
        env.episode_steps.assign(np.array([1], dtype=np.int32))

        with wp.ScopedCapture(device=device) as capture:
            env.observe()
            env.observe()
        wp.capture_launch(capture.graph)

        hip_l1 = sum(abs(v) for v in hip_delta.values())
        waist_l1 = sum(abs(v) for v in waist_delta.values())
        upper_l1 = sum(abs(v) for v in upper_delta.values())
        ankle_limit = 0.02 + 0.03
        joint_acc_l2 = float(np.sum((qd_values / np.float32(env.config.frame_dt)) ** 2, dtype=np.float32))
        shaped = (
            env.config.w_joint_deviation_hip * hip_l1
            + env.config.w_joint_deviation_waist * waist_l1
            + env.config.w_joint_deviation_upper * upper_l1
            + env.config.w_joint_acc_legs * joint_acc_l2
            + env.config.w_joint_pos_limit_ankle * ankle_limit
        )
        expected_reward = np.float32(shaped * env.config.frame_dt)

        np.testing.assert_allclose(env.previous_joint_qd.numpy()[0], qd[6 : 6 + rl.ACTION_DIM_G1], atol=0.0)
        np.testing.assert_allclose(env.joint_regularizer_episode_step.numpy(), np.array([1], dtype=np.int32))
        np.testing.assert_allclose(env.joint_acc_l2.numpy(), np.array([joint_acc_l2], dtype=np.float32), rtol=1.0e-6)
        np.testing.assert_allclose(env.rewards.numpy(), np.array([expected_reward], dtype=np.float32), rtol=1.0e-6)

        env.dones.assign(np.ones(1, dtype=np.float32))
        with wp.ScopedCapture(device=device) as reset_capture:
            env.reset_done()
        wp.capture_launch(reset_capture.graph)
        np.testing.assert_allclose(env.previous_joint_qd.numpy(), np.zeros((1, rl.ACTION_DIM_G1), dtype=np.float32))
        np.testing.assert_allclose(env.joint_acc_l2.numpy(), np.zeros(1, dtype=np.float32))
        np.testing.assert_allclose(env.joint_regularizer_episode_step.numpy(), np.array([-1], dtype=np.int32))

    def test_g1_foot_contact_support_metrics_inside_graph(self) -> None:
        env = _g1_test_env(world_count=1)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)
        foot_metrics = wp.zeros(
            (env.world_count, 2, G1_FOOT_CONTACT_METRIC_COUNT_TOTAL), dtype=wp.float32, device=env.device
        )

        with wp.ScopedCapture(device=env.device) as capture:
            for _ in range(4):
                env.step(actions)
            scan_g1_foot_contact_metrics(env, foot_metrics)

        wp.capture_launch(capture.graph)

        metrics = foot_metrics.numpy()
        self.assertTrue(np.isfinite(metrics).all())
        self.assertGreater(metrics[0, 0, G1_FOOT_CONTACT_METRIC_COUNT], 0.0)
        self.assertGreater(metrics[0, 1, G1_FOOT_CONTACT_METRIC_COUNT], 0.0)
        self.assertGreater(float(np.sum(metrics[0, :, G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE])), 0.0)
        foot_counts = metrics[0, :, G1_FOOT_CONTACT_METRIC_COUNT]
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE] >= 0.0))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM] >= 0.0))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_SPECULATIVE_COUNT] >= 0.0))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_SPECULATIVE_COUNT] <= foot_counts + 1.0e-5))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_TANGENT_BIAS] >= 0.0))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_HIGH_TANGENT_RATIO_COUNT] >= 0.0))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT] >= 0.0))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT] <= foot_counts + 1.0e-5))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_ACTIVE_TANGENT_COUNT] >= 0.0))
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_FRICTION_LOAD] >= 0.0))
        self.assertTrue(
            np.all(
                metrics[0, :, G1_FOOT_CONTACT_METRIC_FRICTION_LOAD]
                <= metrics[0, :, G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE] + 1.0e-5
            )
        )
        self.assertTrue(np.all(metrics[0, :, G1_FOOT_CONTACT_METRIC_FRICTION_LOAD_RATIO_SUM] >= 0.0))
        self.assertTrue(
            np.all(
                metrics[0, :, G1_FOOT_CONTACT_METRIC_FRICTION_LOAD_RATIO_SUM]
                <= metrics[0, :, G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT] + 1.0e-5
            )
        )
        self.assertTrue(
            np.all(
                metrics[0, :, G1_FOOT_CONTACT_METRIC_HIGH_TANGENT_RATIO_COUNT]
                <= metrics[0, :, G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT] + 1.0e-5
            )
        )
        ratio_mean = metrics[0, :, G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM] / foot_counts
        self.assertTrue(np.isfinite(ratio_mean).all())

    def test_observe_clamps_extreme_state_to_finite_metrics(self) -> None:
        env = _g1_test_env(world_count=1)
        huge_qd = np.full(env.state_0.joint_qd.shape, 1.0e20, dtype=np.float32)
        env.state_0.joint_qd.assign(huge_qd)

        obs = env.observe()

        self.assertTrue(np.isfinite(obs.numpy()).all())
        self.assertTrue(np.isfinite(env.rewards.numpy()).all())
        self.assertTrue(np.isfinite(env.dones.numpy()).all())

    def test_g1_observe_terminates_finite_explosions_inside_graph(self) -> None:
        env = _g1_test_env(world_count=2)
        q = env.state_0.joint_q.numpy()
        qd = env.state_0.joint_qd.numpy()
        q[0] = np.float32(env.config.max_abs_root_position + 1.0)
        qd[env.dof_stride + 6] = np.float32(env.config.max_abs_joint_velocity + 1.0)
        env.state_0.joint_q.assign(q)
        env.state_0.joint_qd.assign(qd)

        with wp.ScopedCapture(device=env.device) as capture:
            env.observe()
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(env.dones.numpy(), np.ones(2, dtype=np.float32), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            env.rewards.numpy(),
            np.full(2, env.config.w_termination, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(env.successes.numpy(), np.zeros(2, dtype=np.float32), rtol=0.0, atol=0.0)

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
            self.assertTrue(math.isfinite(eval_result.stats.fall_fraction))
            self.assertTrue(math.isfinite(eval_result.stats.mean_survival_steps))
            self.assertTrue(math.isfinite(eval_result.stats.mean_command_aligned_displacement))
            self.assertTrue(math.isfinite(eval_result.stats.mean_command_aligned_velocity))
            self.assertTrue(math.isfinite(eval_result.stats.mean_lateral_displacement_abs))
            self.assertTrue(math.isfinite(eval_result.stats.mean_path_length))
            self.assertGreaterEqual(eval_result.stats.fall_fraction, 0.0)
            self.assertLessEqual(eval_result.stats.fall_fraction, 1.0)
            self.assertGreaterEqual(eval_result.stats.mean_survival_steps, 0.0)
            self.assertGreaterEqual(eval_result.stats.mean_path_length, 0.0)
            self.assertGreater(eval_result.stats.samples_per_second, 0.0)

            target_result = rl.evaluate_g1_target_ppo(
                restored,
                rl.ConfigEvaluateG1TargetPPO(
                    env_config=env_config,
                    target_positions=((0.6, 0.0),),
                    steps=1,
                    device=device,
                    deterministic=True,
                ),
            )
            self.assertEqual(len(target_result.stats), 1)
            self.assertTrue(math.isfinite(target_result.stats[0].success_fraction))
            self.assertTrue(math.isfinite(target_result.stats[0].fall_fraction))
            self.assertTrue(math.isfinite(target_result.stats[0].mean_path_length))

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

    def test_g1_gate_fixed_commands_survive_randomized_reset(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 gate command reset regression tests")

        class RecorderPolicy:
            def __init__(self, device):
                self.device = device
                self.obs_dim = rl.OBS_DIM_G1
                self.action_dim = rl.ACTION_DIM_G1
                self.battery_commands = []
                self._actions = None
                self._log_probs = None
                self._values = None

            def reset_rollout_state(self, dones=None):
                return None

            def act(self, obs, *, seed: int, deterministic: bool):
                rows = int(obs.shape[0])
                obs_np = obs.numpy()
                if rows == 2:
                    self.battery_commands.append(obs_np[:, 6:9].copy())
                if self._actions is None or int(self._actions.shape[0]) != rows:
                    self._actions = wp.zeros((rows, self.action_dim), dtype=wp.float32, device=self.device)
                    self._log_probs = wp.zeros(rows, dtype=wp.float32, device=self.device)
                    self._values = wp.zeros(rows, dtype=wp.float32, device=self.device)
                return self._actions, self._log_probs, self._values

        fixed_command = (0.75, -0.25, 0.5)
        env_config = rl.ConfigEnvG1PhoenX(
            world_count=2,
            sim_substeps=1,
            solver_iterations=1,
            velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
            randomize_commands_on_reset=True,
            command_x_range=(0.0, 0.0),
            command_y_range=(0.0, 0.0),
            command_yaw_range=(0.0, 0.0),
            command_zero_probability=0.0,
            command_resample_steps=0,
            max_episode_steps=0,
            auto_reset=False,
        )
        policy = RecorderPolicy(device)

        rl.evaluate_g1_gate_ppo(
            policy,
            rl.ConfigEvaluateG1GatePPO(
                env_config=env_config,
                battery_commands=(fixed_command,),
                seeds_per_command=2,
                battery_steps=1,
                diagnostic_steps=1,
                diagnostic_world_count=1,
                device=device,
                deterministic=True,
                seed=97,
                min_battery_perf=2.0,
            ),
        )

        self.assertEqual(len(policy.battery_commands), 1)
        expected = np.tile(np.asarray(fixed_command, dtype=np.float32), (2, 1))
        np.testing.assert_allclose(policy.battery_commands[0], expected, rtol=0.0, atol=0.0)

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

    def test_full_solver_replay_training_graph_does_not_collapse_to_all_done(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 full-solver PPO collapse regression tests")
        env_config = g1_recipe.default_g1_env_config(world_count=32)
        ppo_config = g1_recipe.default_g1_ppo_config(minibatch_size=64)

        result = rl.train_g1_ppo(
            rl.ConfigTrainG1PPO(
                iterations=2,
                rollout_steps=8,
                hidden_layers=(32, 32, 32),
                env_config=env_config,
                ppo_config=ppo_config,
                device=device,
                seed=37,
                log_interval=0,
                randomize_commands=False,
                readback_diagnostics=True,
                execution_mode="graph_leapfrog",
            )
        )

        self.assertEqual([stat.iteration for stat in result.history], [0, 1])
        self.assertEqual(result.trainer.actor.net.network_type, "puffer_mingru")
        self.assertTrue(result.trainer.actor_optimizer.matrix_transpose)
        for stat in result.history:
            self.assertTrue(math.isfinite(stat.mean_reward))
            self.assertTrue(math.isfinite(stat.mean_done))
            self.assertTrue(math.isfinite(stat.mean_tracking_perf))
            self.assertTrue(math.isfinite(stat.approx_kl))
            self.assertLess(stat.mean_done, 0.95)

    def test_default_stochastic_rollout_graph_stays_nonterminal_without_update(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 default stochastic rollout regression tests")
        env_config = g1_recipe.default_g1_env_config(
            world_count=32,
            max_episode_steps=0,
            randomize_commands_on_reset=False,
            command_resample_steps=0,
        )
        env = rl.EnvG1PhoenX(env_config, device=device)
        rollout_seed_counter = make_seed_counter(37, device=device)
        reset_seed_counter = make_seed_counter(113, device=device)
        env.use_reset_seed_counter(reset_seed_counter)
        trainer = rl.TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=(32, 32, 32),
            config=g1_recipe.default_g1_ppo_config(minibatch_size=0, replay_ratio=0.0),
            device=device,
            seed=37,
            squash_actions=g1_recipe.SQUASH_ACTIONS,
            activation=g1_recipe.ACTIVATION,
            log_std_init=g1_recipe.LOG_STD_INIT,
            mirror_map=rl.g1_mirror_map_ppo(),
        )
        buffer = rl.BufferRollout(
            num_steps=8,
            num_envs=env.world_count,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            device=device,
        )

        env.collect_ppo_rollout_seed_counter(trainer, buffer, seed_counter=rollout_seed_counter)
        env.reset()
        rollout_seed_counter.assign(np.asarray([37], dtype=np.int32))
        reset_seed_counter.assign(np.asarray([113], dtype=np.int32))
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(3):
                env.collect_ppo_rollout_seed_counter(trainer, buffer, seed_counter=rollout_seed_counter)
        wp.capture_launch(capture.graph)

        reward_mean, done_mean, success_mean = buffer.reward_done_success_means()
        self.assertTrue(math.isfinite(reward_mean))
        self.assertTrue(math.isfinite(done_mean))
        self.assertTrue(math.isfinite(success_mean))
        self.assertLess(done_mean, 0.35)
        self.assertGreater(reward_mean, -5.0)

    def test_default_zero_action_no_reset_stays_finite_until_terminal_fall(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 no-reset stability regression tests")
        env_config = g1_recipe.default_g1_env_config(
            world_count=2,
            max_episode_steps=0,
            auto_reset=False,
            randomize_commands_on_reset=False,
            command_resample_steps=0,
            parse_visuals=False,
        )
        env = rl.EnvG1PhoenX(env_config, device=device)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            for _ in range(60):
                env.step(actions)
        wp.capture_launch(capture.graph)

        joint_q = env.state_0.joint_q.numpy()
        joint_qd = env.state_0.joint_qd.numpy()
        self.assertTrue(np.isfinite(joint_q).all())
        self.assertTrue(np.isfinite(joint_qd).all())
        self.assertLess(float(np.max(np.abs(joint_qd))), 100.0)
        self.assertFalse(np.any(env.step_dones.numpy() > 0.0))
        self.assertTrue(np.isfinite(env.step_rewards.numpy()).all())

    def test_default_zero_action_auto_reset_stays_finite_after_terminal_fall(self) -> None:
        device = require_cuda_graph_capture("PhoenX G1 auto-reset stability regression tests")
        env_config = g1_recipe.default_g1_env_config(
            world_count=2,
            max_episode_steps=0,
            auto_reset=True,
            randomize_commands_on_reset=False,
            command_resample_steps=0,
            parse_visuals=False,
        )
        env = rl.EnvG1PhoenX(env_config, device=device)
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            for _ in range(120):
                env.step(actions)
        wp.capture_launch(capture.graph)

        joint_q = env.state_0.joint_q.numpy()
        joint_qd = env.state_0.joint_qd.numpy()
        self.assertTrue(np.isfinite(joint_q).all())
        self.assertTrue(np.isfinite(joint_qd).all())
        self.assertLess(float(np.max(np.abs(joint_qd))), 100.0)
        self.assertTrue(np.isfinite(env.step_rewards.numpy()).all())

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
                actor_lr=g1_recipe.ACTOR_LR,
                critic_lr=g1_recipe.CRITIC_LR,
                anneal_lr=g1_recipe.ANNEAL_LR,
                lr_anneal_timesteps=g1_recipe.LR_ANNEAL_TIMESTEPS,
                min_lr_ratio=g1_recipe.MIN_LR_RATIO,
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
                log_std_init=g1_recipe.LOG_STD_INIT,
                reset_recurrent_state_on_rollout_start=g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
                command_x=0.8,
                command_y=0.0,
                command_yaw=0.0,
                command_x_range=(-0.5, 0.8),
                command_y_range=(-0.4, 0.4),
                command_yaw_range=(-1.0, 1.0),
                command_zero_probability=g1_recipe.COMMAND_ZERO_PROBABILITY,
                command_curriculum_start=g1_recipe.COMMAND_CURRICULUM_START,
                command_curriculum_samples=g1_recipe.COMMAND_CURRICULUM_SAMPLES,
                no_command_randomization=True,
                sim_substeps=1,
                solver_iterations=1,
                velocity_iterations=g1_recipe.VELOCITY_ITERATIONS,
                reward_mode=g1_recipe.REWARD_MODE,
                w_alive=g1_recipe.W_ALIVE,
                w_track_lin=g1_recipe.W_TRACK_LIN,
                w_track_ang=g1_recipe.W_TRACK_ANG,
                w_lin_vel_z=g1_recipe.W_LIN_VEL_Z,
                w_ang_vel_xy=g1_recipe.W_ANG_VEL_XY,
                w_orientation=g1_recipe.W_ORIENTATION,
                w_torque=g1_recipe.W_TORQUE,
                w_action_rate=g1_recipe.W_ACTION_RATE,
                w_sparse_command_success=g1_recipe.W_SPARSE_COMMAND_SUCCESS,
                sparse_command_velocity_tolerance=g1_recipe.SPARSE_COMMAND_VELOCITY_TOLERANCE,
                sparse_command_yaw_tolerance=g1_recipe.SPARSE_COMMAND_YAW_TOLERANCE,
                target_x=g1_recipe.SPARSE_TARGET_POSITION[0],
                target_y=g1_recipe.SPARSE_TARGET_POSITION[1],
                sparse_target_radius=g1_recipe.SPARSE_TARGET_RADIUS,
                sparse_target_success_upright_cos=g1_recipe.SPARSE_TARGET_SUCCESS_UPRIGHT_COS,
                sparse_target_success_min_base_height=g1_recipe.SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT,
                sparse_target_success_max_base_height=g1_recipe.SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT,
                no_target_curriculum=False,
                target_distance_start=g1_recipe.SPARSE_TARGET_CURRICULUM_START,
                target_distance_end=g1_recipe.SPARSE_TARGET_CURRICULUM_END,
                target_curriculum_samples=g1_recipe.SPARSE_TARGET_CURRICULUM_SAMPLES,
                randomize_target_positions=g1_recipe.SPARSE_TARGET_RANDOMIZE,
                target_angle_min=g1_recipe.SPARSE_TARGET_ANGLE_MIN,
                target_angle_max=g1_recipe.SPARSE_TARGET_ANGLE_MAX,
                w_mechanical_power=g1_recipe.W_MECHANICAL_POWER,
                w_gait_contact=g1_recipe.W_GAIT_CONTACT,
                w_gait_swing=g1_recipe.W_GAIT_SWING,
                w_gait_swing_contact=g1_recipe.W_GAIT_SWING_CONTACT,
                w_gait_hip=g1_recipe.W_GAIT_HIP,
                w_base_height=g1_recipe.W_BASE_HEIGHT,
                actuation_model=g1_recipe.ACTUATION_MODEL,
                controlled_action_count=g1_recipe.CONTROLLED_ACTION_COUNT,
                parse_meshes=False,
                contact_geometry=g1_recipe.CONTACT_GEOMETRY,
                rigid_contact_max_per_world=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
                threads_per_world="auto",
                multi_world_scheduler="auto",
                prepare_refresh_stride="auto",
                execution_mode="graph_leapfrog",
                readback_diagnostics=False,
                resume_checkpoint=None,
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
            self.assertFalse(result["readback_diagnostics"])
            self.assertEqual(result["squash_actions"], g1_recipe.SQUASH_ACTIONS)
            self.assertEqual(result["log_std_init"], g1_recipe.LOG_STD_INIT)
            self.assertEqual(
                result["reset_recurrent_state_on_rollout_start"],
                g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
            )
            self.assertEqual(result["value_loss_coeff"], g1_recipe.VALUE_LOSS_COEFF)
            self.assertEqual(result["value_clip_range"], g1_recipe.VALUE_CLIP_RANGE)
            self.assertEqual(result["target_x"], g1_recipe.SPARSE_TARGET_POSITION[0])
            self.assertEqual(result["target_y"], g1_recipe.SPARSE_TARGET_POSITION[1])
            self.assertEqual(result["sparse_target_radius"], g1_recipe.SPARSE_TARGET_RADIUS)
            self.assertEqual(result["sparse_target_success_upright_cos"], g1_recipe.SPARSE_TARGET_SUCCESS_UPRIGHT_COS)
            self.assertEqual(
                result["sparse_target_success_min_base_height"],
                g1_recipe.SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT,
            )
            self.assertEqual(
                result["sparse_target_success_max_base_height"],
                g1_recipe.SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT,
            )
            self.assertEqual(result["target_distance_start"], g1_recipe.SPARSE_TARGET_CURRICULUM_START)
            self.assertEqual(result["target_distance_end"], g1_recipe.SPARSE_TARGET_CURRICULUM_END)
            self.assertEqual(result["target_curriculum_samples"], g1_recipe.SPARSE_TARGET_CURRICULUM_SAMPLES)
            self.assertEqual(result["randomize_target_positions"], g1_recipe.SPARSE_TARGET_RANDOMIZE)
            self.assertEqual(result["target_angle_min"], g1_recipe.SPARSE_TARGET_ANGLE_MIN)
            self.assertEqual(result["target_angle_max"], g1_recipe.SPARSE_TARGET_ANGLE_MAX)
            self.assertEqual(result["actor_lr"], g1_recipe.ACTOR_LR)
            self.assertEqual(result["critic_lr"], g1_recipe.CRITIC_LR)
            self.assertEqual(result["anneal_lr"], g1_recipe.ANNEAL_LR)
            self.assertEqual(result["lr_anneal_timesteps"], g1_recipe.LR_ANNEAL_TIMESTEPS)
            self.assertEqual(result["min_lr_ratio"], g1_recipe.MIN_LR_RATIO)
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
