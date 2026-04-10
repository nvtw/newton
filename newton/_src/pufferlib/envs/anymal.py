# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ANYmal C walking environment for PufferLib PPO training.

Integrates Newton's MuJoCo-Warp solver with PufferLib's env interface.
All observation, reward, and reset kernels run on GPU and are CUDA-graph
capturable.

The reward function follows the IsaacLab AnymalCFlatEnvCfg implementation:
  - Linear/angular velocity tracking (exponential mapping)
  - Penalties for z-velocity, roll/pitch angular velocity, joint
    torques, joint acceleration, action rate, flat orientation
  - Feet air time bonus and undesired thigh contact penalty
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import GeoType

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBS_DIM = 48
_ACT_DIM = 12
_Q_STRIDE = 19  # 7 (free joint) + 12 (revolute)
_QD_STRIDE = 18  # 6 (free joint) + 12 (revolute)
_NUM_BODIES = 13
_MAX_EPISODE_LENGTH = 1000

# Joint ordering: Newton URDF order -> IsaacLab (lab) order and back.
_LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
_MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

# Reward scales — alive bonus + high tracking drives locomotion.
_ALIVE_BONUS = 1.0
_LIN_VEL_REWARD_SCALE = 5.0
_YAW_RATE_REWARD_SCALE = 2.0
_Z_VEL_REWARD_SCALE = -2.0
_ANG_VEL_REWARD_SCALE = -0.05
_JOINT_TORQUE_REWARD_SCALE = -2.5e-5
_JOINT_ACCEL_REWARD_SCALE = -2.5e-7
_ACTION_RATE_REWARD_SCALE = -0.01
_FEET_AIR_TIME_REWARD_SCALE = 0.5
_UNDESIRED_CONTACT_REWARD_SCALE = -1.0
_FLAT_ORIENTATION_REWARD_SCALE = -5.0

_ACTION_SCALE = 0.5
_FRAME_DT = 0.02  # sim_dt(0.005) * substeps(4)
_SIM_DT = 0.005
_SIM_SUBSTEPS = 4

# PD gains
_KE = 150.0
_KD = 5.0

# Height thresholds for contact detection via body_q
_FOOT_CONTACT_THRESHOLD = 0.07
_THIGH_CONTACT_THRESHOLD = 0.18

_INITIAL_JOINT_Q_NAMES = {
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
}


# ---------------------------------------------------------------------------
# Helper: rotate vector by inverse quaternion (world -> body frame)
# ---------------------------------------------------------------------------


@wp.func
def _quat_rotate_inv(qx: float, qy: float, qz: float, qw: float, vx: float, vy: float, vz: float):
    """Rotate (vx,vy,vz) by the inverse of quaternion (qx,qy,qz,qw)."""
    q2w2m1 = 2.0 * qw * qw - 1.0
    cross_x = qy * vz - qz * vy
    cross_y = qz * vx - qx * vz
    cross_z = qx * vy - qy * vx
    dot_qv = qx * vx + qy * vy + qz * vz
    bx = vx * q2w2m1 - 2.0 * qw * cross_x + 2.0 * qx * dot_qv
    by = vy * q2w2m1 - 2.0 * qw * cross_y + 2.0 * qy * dot_qv
    bz = vz * q2w2m1 - 2.0 * qw * cross_z + 2.0 * qz * dot_qv
    return bx, by, bz


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_obs_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    last_actions: wp.array2d[float],
    commands: wp.array2d[float],
    joint_pos_initial: wp.array[float],
    lab_to_mujoco: wp.array[int],
    obs: wp.array2d[float],
):
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE

    qx = joint_q[q_off + 3]
    qy = joint_q[q_off + 4]
    qz = joint_q[q_off + 5]
    qw = joint_q[q_off + 6]

    # Base linear velocity in body frame
    vx = joint_qd[qd_off + 0]
    vy = joint_qd[qd_off + 1]
    vz = joint_qd[qd_off + 2]
    bvx, bvy, bvz = _quat_rotate_inv(qx, qy, qz, qw, vx, vy, vz)
    obs[env, 0] = bvx
    obs[env, 1] = bvy
    obs[env, 2] = bvz

    # Base angular velocity in body frame
    wx = joint_qd[qd_off + 3]
    wy = joint_qd[qd_off + 4]
    wz = joint_qd[qd_off + 5]
    bwx, bwy, bwz = _quat_rotate_inv(qx, qy, qz, qw, wx, wy, wz)
    obs[env, 3] = bwx
    obs[env, 4] = bwy
    obs[env, 5] = bwz

    # Projected gravity in body frame: rotate (0, 0, -1) by inverse quat
    gx, gy, gz = _quat_rotate_inv(qx, qy, qz, qw, 0.0, 0.0, -1.0)
    obs[env, 6] = gx
    obs[env, 7] = gy
    obs[env, 8] = gz

    # Velocity commands
    obs[env, 9] = commands[env, 0]
    obs[env, 10] = commands[env, 1]
    obs[env, 11] = commands[env, 2]

    # Joint positions (deviation from default) and velocities -- reordered
    for j in range(12):
        mj = lab_to_mujoco[j]
        obs[env, 12 + mj] = joint_q[q_off + 7 + j] - joint_pos_initial[j]
        obs[env, 24 + mj] = joint_qd[qd_off + 6 + j]

    # Previous actions
    for j in range(12):
        obs[env, 36 + j] = last_actions[env, j]


@wp.kernel
def _compute_rewards_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    last_actions: wp.array2d[float],
    prev_actions: wp.array2d[float],
    prev_joint_vel: wp.array2d[float],
    commands: wp.array2d[float],
    joint_pos_initial: wp.array[float],
    joint_target_pos: wp.array[float],
    foot_ids: wp.array[int],
    thigh_ids: wp.array[int],
    foot_air_time: wp.array2d[float],
    foot_was_contact: wp.array2d[float],
    episode_lengths: wp.array[int],
    frame_dt: float,
    rewards: wp.array[float],
    dones: wp.array[float],
):
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE
    body_off = env * _NUM_BODIES

    height = joint_q[q_off + 2]
    qx = joint_q[q_off + 3]
    qy = joint_q[q_off + 4]
    qz = joint_q[q_off + 5]
    qw = joint_q[q_off + 6]

    vx_w = joint_qd[qd_off + 0]
    vy_w = joint_qd[qd_off + 1]
    vz_w = joint_qd[qd_off + 2]
    wx_w = joint_qd[qd_off + 3]
    wy_w = joint_qd[qd_off + 4]
    wz_w = joint_qd[qd_off + 5]

    vel_bx, vel_by, vel_bz = _quat_rotate_inv(qx, qy, qz, qw, vx_w, vy_w, vz_w)
    avel_bx, avel_by, avel_bz = _quat_rotate_inv(qx, qy, qz, qw, wx_w, wy_w, wz_w)

    grav_bx, grav_by, _grav_bz = _quat_rotate_inv(qx, qy, qz, qw, 0.0, 0.0, -1.0)

    cmd_vx = commands[env, 0]
    cmd_vy = commands[env, 1]
    cmd_yaw = commands[env, 2]

    # 1. Linear velocity tracking (exp mapping, standard IsaacLab formulation)
    lin_vel_err = (cmd_vx - vel_bx) * (cmd_vx - vel_bx) + (cmd_vy - vel_by) * (cmd_vy - vel_by)
    track_lin_vel = wp.exp(-lin_vel_err / 0.25)

    # 2. Yaw rate tracking (exp mapping)
    yaw_err = (cmd_yaw - avel_bz) * (cmd_yaw - avel_bz)
    track_ang_vel = wp.exp(-yaw_err / 0.25)

    # 3. Z velocity penalty
    z_vel_l2 = vel_bz * vel_bz

    # 4. Angular velocity xy penalty
    ang_vel_xy_l2 = avel_bx * avel_bx + avel_by * avel_by

    # 5. Joint torque penalty (PD torque approximation)
    torque_l2 = float(0.0)
    for j in range(12):
        pos_j = joint_q[q_off + 7 + j]
        vel_j = joint_qd[qd_off + 6 + j]
        target_j = joint_target_pos[qd_off + 6 + j]
        tau = _KE * (target_j - pos_j) - _KD * vel_j
        torque_l2 = torque_l2 + tau * tau

    # 6. Joint acceleration penalty
    accel_l2 = float(0.0)
    inv_dt = 1.0 / frame_dt
    for j in range(12):
        vel_j = joint_qd[qd_off + 6 + j]
        prev_vel_j = prev_joint_vel[env, j]
        acc = (vel_j - prev_vel_j) * inv_dt
        accel_l2 = accel_l2 + acc * acc

    # 7. Action rate penalty
    action_rate_l2 = float(0.0)
    for j in range(12):
        diff = last_actions[env, j] - prev_actions[env, j]
        action_rate_l2 = action_rate_l2 + diff * diff

    # 8. Feet air time reward
    cmd_speed = wp.sqrt(cmd_vx * cmd_vx + cmd_vy * cmd_vy)
    feet_air_reward = float(0.0)
    for foot_idx in range(4):
        foot_pos = wp.transform_get_translation(body_q[body_off + foot_ids[foot_idx]])
        in_contact = float(0.0)
        if foot_pos[2] < _FOOT_CONTACT_THRESHOLD:
            in_contact = 1.0

        was_airborne = 1.0 - foot_was_contact[env, foot_idx]
        first_contact = in_contact * was_airborne

        feet_air_reward = feet_air_reward + (foot_air_time[env, foot_idx] - 0.5) * first_contact

        if in_contact > 0.5:
            foot_air_time[env, foot_idx] = 0.0
        else:
            foot_air_time[env, foot_idx] = foot_air_time[env, foot_idx] + frame_dt

        foot_was_contact[env, foot_idx] = in_contact

    if cmd_speed < 0.1:
        feet_air_reward = 0.0

    # 9. Undesired contacts penalty (thigh bodies)
    undesired_contacts = float(0.0)
    for leg in range(4):
        thigh_pos = wp.transform_get_translation(body_q[body_off + thigh_ids[leg]])
        if thigh_pos[2] < _THIGH_CONTACT_THRESHOLD:
            undesired_contacts = undesired_contacts + 1.0

    # 10. Flat orientation penalty
    flat_orientation_l2 = grav_bx * grav_bx + grav_by * grav_by

    # Combine all reward terms: alive bonus + tracking + penalties
    reward = (
        _ALIVE_BONUS
        + track_lin_vel * _LIN_VEL_REWARD_SCALE
        + track_ang_vel * _YAW_RATE_REWARD_SCALE
        + z_vel_l2 * _Z_VEL_REWARD_SCALE
        + ang_vel_xy_l2 * _ANG_VEL_REWARD_SCALE
        + torque_l2 * _JOINT_TORQUE_REWARD_SCALE
        + accel_l2 * _JOINT_ACCEL_REWARD_SCALE
        + action_rate_l2 * _ACTION_RATE_REWARD_SCALE
        + feet_air_reward * _FEET_AIR_TIME_REWARD_SCALE
        + undesired_contacts * _UNDESIRED_CONTACT_REWARD_SCALE
        + flat_orientation_l2 * _FLAT_ORIENTATION_REWARD_SCALE
    )
    rewards[env] = reward

    # Termination
    terminated = float(0.0)
    if height < 0.25:
        terminated = 1.0
    if flat_orientation_l2 > 0.49:
        terminated = 1.0
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0
    dones[env] = terminated

    # Store current joint velocities for next step's acceleration
    for j in range(12):
        prev_joint_vel[env, j] = joint_qd[qd_off + 6 + j]


@wp.kernel
def _apply_actions_kernel(
    actions: wp.array2d[float],
    mujoco_to_lab: wp.array[int],
    joint_pos_initial: wp.array[float],
    joint_target_pos: wp.array[float],
):
    """Map network output to joint targets (matching IsaacLab: no tanh squashing)."""
    env, j = wp.tid()
    lab_j = mujoco_to_lab[j]
    joint_target_pos[env * _QD_STRIDE + 6 + lab_j] = joint_pos_initial[lab_j] + _ACTION_SCALE * actions[env, j]


@wp.kernel
def _copy_actions_kernel(
    new_actions: wp.array2d[float],
    last_actions: wp.array2d[float],
    prev_actions: wp.array2d[float],
):
    env, j = wp.tid()
    prev_actions[env, j] = last_actions[env, j]
    last_actions[env, j] = new_actions[env, j]


@wp.kernel
def _reset_envs_kernel(
    dones: wp.array[float],
    initial_q: wp.array[float],
    initial_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    episode_lengths: wp.array[int],
    last_actions: wp.array2d[float],
    prev_actions: wp.array2d[float],
    prev_joint_vel: wp.array2d[float],
    foot_air_time: wp.array2d[float],
    foot_was_contact: wp.array2d[float],
    running_return: wp.array[float],
    seed_arr: wp.array[int],
):
    """Reset done envs with randomized initial state (matching IsaacLab).

    Adds small noise to joint positions (±0.25 rad) and velocities (±0.1 rad/s)
    to break correlations between environments.  Uses seed_arr for deterministic
    randomization compatible with CUDA graph capture.
    """
    env = wp.tid()
    if dones[env] > 0.5:
        q_off = env * _Q_STRIDE
        qd_off = env * _QD_STRIDE
        rng = wp.rand_init(seed_arr[0], env * 31 + 7)
        # Reset base position/orientation (no noise on free joint)
        for i in range(7):
            joint_q[q_off + i] = initial_q[i]
        # Reset joint positions with small noise
        for i in range(12):
            joint_q[q_off + 7 + i] = initial_q[7 + i] + wp.randf(rng, -0.25, 0.25)
        # Reset base velocity (no noise)
        for i in range(6):
            joint_qd[qd_off + i] = 0.0
        # Reset joint velocities with small noise
        for i in range(12):
            joint_qd[qd_off + 6 + i] = wp.randf(rng, -0.1, 0.1)
        episode_lengths[env] = 0
        running_return[env] = 0.0
        for i in range(12):
            last_actions[env, i] = 0.0
            prev_actions[env, i] = 0.0
            prev_joint_vel[env, i] = 0.0
        for i in range(4):
            foot_air_time[env, i] = 0.0
            foot_was_contact[env, i] = 0.0


@wp.kernel
def _randomize_commands_kernel(dones: wp.array[float], commands: wp.array2d[float], rng_counter: wp.array[int]):
    env = wp.tid()
    if dones[env] > 0.5:
        step_offset = rng_counter[0]
        rng = wp.rand_init(42, step_offset * 1000 + env)
        commands[env, 0] = wp.randf(rng) * 2.0 - 0.5  # forward: [-0.5, 1.5]
        commands[env, 1] = wp.randf(rng) * 1.0 - 0.5  # lateral: [-0.5, 0.5]
        commands[env, 2] = wp.randf(rng) * 1.0 - 0.5  # yaw:     [-0.5, 0.5]


@wp.kernel
def _init_all_commands_kernel(commands: wp.array2d[float], rng_counter: wp.array[int]):
    env = wp.tid()
    seed_val = rng_counter[0]
    rng = wp.rand_init(seed_val, env * 3)
    commands[env, 0] = wp.randf(rng) * 2.0 - 0.5
    commands[env, 1] = wp.randf(rng) * 1.0 - 0.5
    commands[env, 2] = wp.randf(rng) * 1.0 - 0.5


@wp.kernel
def _track_episodes_kernel(
    rewards: wp.array[float],
    dones: wp.array[float],
    running_return: wp.array[float],
    episode_returns: wp.array[float],
    episode_lengths_int: wp.array[int],
    episode_lengths_out: wp.array[float],
):
    """Update episode return/length tracking for PufferLib stats."""
    env = wp.tid()
    running_return[env] = running_return[env] + rewards[env]
    episode_lengths_int[env] = episode_lengths_int[env] + 1
    episode_returns[env] = running_return[env]
    episode_lengths_out[env] = float(episode_lengths_int[env])


@wp.kernel
def _tile_initial_state_kernel(
    initial_q: wp.array[float],
    initial_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Tile single-world initial state across all environments."""
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE
    for i in range(wp.static(_Q_STRIDE)):
        joint_q[q_off + i] = initial_q[i]
    for i in range(wp.static(_QD_STRIDE)):
        joint_qd[qd_off + i] = initial_qd[i]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AnymalEnv:
    """ANYmal C walking environment compatible with PufferLib's PPOTrainer.

    Wraps Newton's MuJoCo-Warp solver with the env interface expected by
    :class:`~warp.pufferlib.trainer.PPOTrainer`: pre-allocated ``obs``,
    ``rewards``, ``dones``, ``episode_returns``, ``episode_lengths`` arrays,
    plus ``step_graphed()``, ``reset()``, and ``get_episode_stats()`` methods.

    All operations are CUDA-graph capturable.
    """

    NUM_ACTIONS = _ACT_DIM
    OBS_SIZE = _OBS_DIM

    def __init__(self, num_envs: int = 4096, device: str = "cuda:0", seed: int = 123,
                 use_mujoco_contacts: bool = False):
        self.num_envs = num_envs
        self.device = device
        self.sim_time = 0.0

        # --- Build single-robot model ---
        art = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(art)
        art.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        art.default_shape_cfg.ke = 5.0e4
        art.default_shape_cfg.kd = 5.0e2
        art.default_shape_cfg.kf = 1.0e3
        art.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        art.add_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.62),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
            hide_visuals=True,
        )

        # Remove mesh shapes entirely to avoid massive memory use when
        # replicating across thousands of worlds.  Only primitive collision
        # shapes (spheres, boxes, cylinders) are needed for RL training.
        n_shapes = len(art.shape_type)
        keep = [i for i in range(n_shapes) if art.shape_type[i] != GeoType.MESH]
        keep_set = set(keep)
        # Build old→new index mapping for collision filter pairs
        old_to_new = {old: new for new, old in enumerate(keep)}
        for attr_name in [a for a in dir(art) if a.startswith("shape_") and isinstance(getattr(art, a), list)]:
            lst = getattr(art, attr_name)
            if len(lst) == n_shapes:
                setattr(art, attr_name, [lst[i] for i in keep])
        # Remap collision filter pairs (drop pairs referencing removed shapes)
        art.shape_collision_filter_pairs = [
            (old_to_new[a], old_to_new[b])
            for a, b in art.shape_collision_filter_pairs
            if a in keep_set and b in keep_set
        ]

        # Enlarge foot collision spheres for stability
        for i in range(len(art.shape_type)):
            if art.shape_type[i] == GeoType.SPHERE:
                r = art.shape_scale[i][0]
                art.shape_scale[i] = (r * 2.0, 0.0, 0.0)

        # Set initial joint positions
        for name, value in _INITIAL_JOINT_Q_NAMES.items():
            idx = next((i for i, lbl in enumerate(art.joint_label) if lbl.endswith(f"/{name}")), None)
            if idx is None:
                raise ValueError(f"Joint '{name}' not found")
            art.joint_q[idx + 6] = value

        # Set PD gains
        for i in range(len(art.joint_target_ke)):
            art.joint_target_ke[i] = _KE
            art.joint_target_kd[i] = _KD

        # --- Replicate for multi-world ---
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(art, num_envs)
        builder.add_ground_plane()

        self.model = builder.finalize(device=device)

        # --- Create solver ---
        self._use_mujoco_contacts = use_mujoco_contacts
        if use_mujoco_contacts:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=True,
                solver="newton",
                ls_iterations=50,
            )
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                ls_parallel=False,
                ls_iterations=50,
                njmax=50 * num_envs,
                nconmax=100 * num_envs,
            )

        # --- Create states, control, contacts ---
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None if use_mujoco_contacts else self.model.contacts()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # --- Store initial state for resets (single world) ---
        full_q = self.state_0.joint_q.numpy()[:_Q_STRIDE]
        full_qd = self.state_0.joint_qd.numpy()[:_QD_STRIDE]
        self.initial_joint_q = wp.array(full_q.astype(np.float32), dtype=wp.float32, device=device)
        self.initial_joint_qd = wp.array(full_qd.astype(np.float32), dtype=wp.float32, device=device)
        self._joint_pos_initial = wp.array(full_q[7:_Q_STRIDE].astype(np.float32), dtype=wp.float32, device=device)

        # --- Resolve body indices ---
        body_names = [lbl.split("/")[-1] for lbl in self.model.body_label[:_NUM_BODIES]]
        self._foot_ids_wp = wp.array(
            [body_names.index(n) for n in ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]],
            dtype=wp.int32,
            device=device,
        )
        self._thigh_ids_wp = wp.array(
            [body_names.index(n) for n in ["LF_THIGH", "RF_THIGH", "LH_THIGH", "RH_THIGH"]],
            dtype=wp.int32,
            device=device,
        )

        # --- Joint ordering arrays ---
        self._lab_to_mujoco = wp.array(_LAB_TO_MUJOCO, dtype=wp.int32, device=device)
        self._mujoco_to_lab = wp.array(_MUJOCO_TO_LAB, dtype=wp.int32, device=device)

        # --- PufferLib interface arrays ---
        self.obs = wp.zeros((num_envs, _OBS_DIM), dtype=wp.float32, device=device)
        self.rewards = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.dones = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.episode_returns = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.episode_lengths = wp.zeros(num_envs, dtype=wp.float32, device=device)

        # --- Internal env buffers ---
        self._episode_lengths_int = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self._running_return = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.last_actions = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=device)
        self.prev_actions = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=device)
        self.prev_joint_vel = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=device)
        self.commands = wp.zeros((num_envs, 3), dtype=wp.float32, device=device)
        self.foot_air_time = wp.zeros((num_envs, 4), dtype=wp.float32, device=device)
        self.foot_was_contact = wp.zeros((num_envs, 4), dtype=wp.float32, device=device)
        self.rng_counter = wp.array([seed], dtype=wp.int32, device=device)

    def reset(self):
        """Reset all environments to initial state."""
        d = self.device
        N = self.num_envs

        # Tile initial state across all envs
        wp.launch(_tile_initial_state_kernel, dim=N, inputs=[
            self.initial_joint_q, self.initial_joint_qd,
            self.state_0.joint_q, self.state_0.joint_qd,
        ], device=d)

        # Zero all buffers
        self._episode_lengths_int.zero_()
        self._running_return.zero_()
        self.last_actions.zero_()
        self.prev_actions.zero_()
        self.prev_joint_vel.zero_()
        self.foot_air_time.zero_()
        self.foot_was_contact.zero_()
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        self.rewards.zero_()
        self.dones.zero_()

        # Initialize commands
        wp.launch(_init_all_commands_kernel, dim=N, inputs=[self.commands, self.rng_counter], device=d)

        # Update body poses from joint state
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Compute initial observations
        wp.launch(
            _compute_obs_kernel,
            dim=N,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.last_actions,
                self.commands,
                self._joint_pos_initial,
                self._lab_to_mujoco,
                self.obs,
            ],
            device=d,
        )

        self.sim_time = 0.0
        return self.obs

    def step_graphed(self, actions: wp.array, seed_arr: wp.array):
        """Graph-capture-compatible step.

        Args:
            actions: ``(N, 12)`` continuous joint action commands.
            seed_arr: Device-side 1-element seed array (used by trainer for RNG).
        """
        d = self.device
        N = self.num_envs

        # 1. Copy actions to history buffers
        wp.launch(
            _copy_actions_kernel,
            dim=(N, _ACT_DIM),
            inputs=[actions, self.last_actions, self.prev_actions],
            device=d,
        )

        # 2. Apply actions with tanh squashing to joint targets
        wp.launch(
            _apply_actions_kernel,
            dim=(N, _ACT_DIM),
            inputs=[actions, self._mujoco_to_lab, self._joint_pos_initial, self.control.joint_target_pos],
            device=d,
        )

        # 3. Physics substeps (inlined loop unrolls during graph capture)
        for _ in range(_SIM_SUBSTEPS):
            self.state_0.clear_forces()
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, _SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # 4. Compute rewards and dones
        wp.launch(
            _compute_rewards_kernel,
            dim=N,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0.body_q,
                self.last_actions,
                self.prev_actions,
                self.prev_joint_vel,
                self.commands,
                self._joint_pos_initial,
                self.control.joint_target_pos,
                self._foot_ids_wp,
                self._thigh_ids_wp,
                self.foot_air_time,
                self.foot_was_contact,
                self._episode_lengths_int,
                _FRAME_DT,
                self.rewards,
                self.dones,
            ],
            device=d,
        )

        # 5. Track episode stats (before reset clears running_return)
        wp.launch(
            _track_episodes_kernel,
            dim=N,
            inputs=[
                self.rewards,
                self.dones,
                self._running_return,
                self.episode_returns,
                self._episode_lengths_int,
                self.episode_lengths,
            ],
            device=d,
        )

        # 6. Reset done envs
        wp.launch(
            _reset_envs_kernel,
            dim=N,
            inputs=[
                self.dones,
                self.initial_joint_q,
                self.initial_joint_qd,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._episode_lengths_int,
                self.last_actions,
                self.prev_actions,
                self.prev_joint_vel,
                self.foot_air_time,
                self.foot_was_contact,
                self._running_return,
                seed_arr,
            ],
            device=d,
        )

        # 7. Randomize commands for reset envs
        wp.launch(
            _randomize_commands_kernel,
            dim=N,
            inputs=[self.dones, self.commands, self.rng_counter],
            device=d,
        )

        # 8. Update body poses (needed for contact height checks after reset)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # 9. Compute observations from new state
        wp.launch(
            _compute_obs_kernel,
            dim=N,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.last_actions,
                self.commands,
                self._joint_pos_initial,
                self._lab_to_mujoco,
                self.obs,
            ],
            device=d,
        )

        self.sim_time += _FRAME_DT

    def get_episode_stats(self) -> dict:
        """Return episode statistics for PufferLib logging.

        Only call at log intervals — triggers a GPU sync via ``.numpy()``.
        """
        dones_np = self.dones.numpy()
        done_mask = dones_np > 0.5
        if np.any(done_mask):
            returns_np = self.episode_returns.numpy()
            lengths_np = self.episode_lengths.numpy()
            return {
                "mean_return": float(np.mean(returns_np[done_mask])),
                "mean_length": float(np.mean(lengths_np[done_mask])),
                "num_episodes": int(np.sum(done_mask)),
            }
        return {"mean_return": 0.0, "mean_length": 0.0, "num_episodes": 0}
