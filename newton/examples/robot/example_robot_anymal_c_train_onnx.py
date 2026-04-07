# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Train (ONNX)
#
# Trains an ANYmal C walking policy from scratch using the built-in PPO
# trainer and Warp kernels.  No PyTorch dependency.  The trained policy
# is exported to ONNX and can be loaded by OnnxRuntime for inference.
#
# The reward function follows the IsaacLab AnymalCEnv implementation:
#   - Linear/angular velocity tracking (exponential mapping)
#   - Penalties for z-velocity, roll/pitch angular velocity, joint
#     torques, joint acceleration, action rate, flat orientation
#   - Feet air time bonus and undesired thigh contact penalty
#
# Command: python -m newton.examples robot_anymal_c_train_onnx
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import GeoType
from newton._src.ppo import ActorCritic, PPOTrainer
from newton._src.robot_env import RobotEnv
from newton._src.warp_nn import export_to_onnx

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

# Body indices within a single environment (see test_final in walk example):
#   0: base
#   1: LF_HIP   2: LF_THIGH   3: LF_SHANK
#   4: RF_HIP   5: RF_THIGH   6: RF_SHANK
#   7: LH_HIP   8: LH_THIGH   9: LH_SHANK
#  10: RH_HIP  11: RH_THIGH  12: RH_SHANK

# IsaacLab reward scales (AnymalCFlatEnvCfg)
# Reward scales -- increased tracking rewards relative to IsaacLab defaults
# to accelerate learning with fewer envs. Penalty scales kept as-is.
_LIN_VEL_REWARD_SCALE = 3.0   # IsaacLab: 1.0 -- boosted to encourage movement
_YAW_RATE_REWARD_SCALE = 1.5  # IsaacLab: 0.5
_Z_VEL_REWARD_SCALE = -2.0
_ANG_VEL_REWARD_SCALE = -0.05
_JOINT_TORQUE_REWARD_SCALE = -2.5e-5
_JOINT_ACCEL_REWARD_SCALE = -2.5e-7
_ACTION_RATE_REWARD_SCALE = -0.01
_FEET_AIR_TIME_REWARD_SCALE = 1.5  # IsaacLab: 0.5 -- boosted to encourage gait
_UNDESIRED_CONTACT_REWARD_SCALE = -1.0
_FLAT_ORIENTATION_REWARD_SCALE = -5.0

_ACTION_SCALE = 0.5  # IsaacLab action_scale
_FRAME_DT = 0.02  # sim_dt(0.005) * substeps(4)

# PD gains (set in build_robot)
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
    # v_body = v*(2w^2 - 1) - 2w*(q x v) + 2q*(q . v)
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

    # --- Body-frame velocities ---
    vx_w = joint_qd[qd_off + 0]
    vy_w = joint_qd[qd_off + 1]
    vz_w = joint_qd[qd_off + 2]
    wx_w = joint_qd[qd_off + 3]
    wy_w = joint_qd[qd_off + 4]
    wz_w = joint_qd[qd_off + 5]

    vel_bx, vel_by, vel_bz = _quat_rotate_inv(qx, qy, qz, qw, vx_w, vy_w, vz_w)
    avel_bx, avel_by, avel_bz = _quat_rotate_inv(qx, qy, qz, qw, wx_w, wy_w, wz_w)

    # --- Projected gravity in body frame ---
    grav_bx, grav_by, _grav_bz = _quat_rotate_inv(qx, qy, qz, qw, 0.0, 0.0, -1.0)

    cmd_vx = commands[env, 0]
    cmd_vy = commands[env, 1]
    cmd_yaw = commands[env, 2]

    # --- 1. Linear velocity tracking (exp mapping) ---
    lin_vel_err = (cmd_vx - vel_bx) * (cmd_vx - vel_bx) + (cmd_vy - vel_by) * (cmd_vy - vel_by)
    track_lin_vel = wp.exp(-lin_vel_err / 0.25)

    # --- 2. Yaw rate tracking (exp mapping) ---
    yaw_err = (cmd_yaw - avel_bz) * (cmd_yaw - avel_bz)
    track_ang_vel = wp.exp(-yaw_err / 0.25)

    # --- 3. Z velocity penalty ---
    z_vel_l2 = vel_bz * vel_bz

    # --- 4. Angular velocity xy penalty ---
    ang_vel_xy_l2 = avel_bx * avel_bx + avel_by * avel_by

    # --- 5. Joint torque penalty (PD torque approximation) ---
    torque_l2 = float(0.0)
    for j in range(12):
        pos_j = joint_q[q_off + 7 + j]
        vel_j = joint_qd[qd_off + 6 + j]
        target_j = joint_target_pos[qd_off + 6 + j]
        tau = _KE * (target_j - pos_j) - _KD * vel_j
        torque_l2 = torque_l2 + tau * tau

    # --- 6. Joint acceleration penalty ---
    accel_l2 = float(0.0)
    inv_dt = 1.0 / frame_dt
    for j in range(12):
        vel_j = joint_qd[qd_off + 6 + j]
        prev_vel_j = prev_joint_vel[env, j]
        acc = (vel_j - prev_vel_j) * inv_dt
        accel_l2 = accel_l2 + acc * acc

    # --- 7. Action rate penalty ---
    action_rate_l2 = float(0.0)
    for j in range(12):
        diff = last_actions[env, j] - prev_actions[env, j]
        action_rate_l2 = action_rate_l2 + diff * diff

    # --- 8. Feet air time reward ---
    cmd_speed = wp.sqrt(cmd_vx * cmd_vx + cmd_vy * cmd_vy)
    feet_air_reward = float(0.0)
    for foot_idx in range(4):
        foot_pos = wp.transform_get_translation(body_q[body_off + foot_ids[foot_idx]])
        in_contact = float(0.0)
        if foot_pos[2] < _FOOT_CONTACT_THRESHOLD:
            in_contact = 1.0

        was_airborne = 1.0 - foot_was_contact[env, foot_idx]
        first_contact = in_contact * was_airborne

        # Reward on first contact: (air_time - 0.5)
        feet_air_reward = feet_air_reward + (foot_air_time[env, foot_idx] - 0.5) * first_contact

        # Update air time tracking
        if in_contact > 0.5:
            foot_air_time[env, foot_idx] = 0.0
        else:
            foot_air_time[env, foot_idx] = foot_air_time[env, foot_idx] + frame_dt

        foot_was_contact[env, foot_idx] = in_contact

    # Only reward air time when the robot has a non-trivial velocity command
    if cmd_speed < 0.1:
        feet_air_reward = 0.0

    # --- 9. Undesired contacts penalty (thigh bodies) ---
    undesired_contacts = float(0.0)
    for leg in range(4):
        thigh_pos = wp.transform_get_translation(body_q[body_off + thigh_ids[leg]])
        if thigh_pos[2] < _THIGH_CONTACT_THRESHOLD:
            undesired_contacts = undesired_contacts + 1.0

    # --- 10. Flat orientation penalty ---
    flat_orientation_l2 = grav_bx * grav_bx + grav_by * grav_by

    # --- Combine all reward terms (all multiplied by frame_dt as in IsaacLab) ---
    reward = (
        track_lin_vel * _LIN_VEL_REWARD_SCALE * frame_dt
        + track_ang_vel * _YAW_RATE_REWARD_SCALE * frame_dt
        + z_vel_l2 * _Z_VEL_REWARD_SCALE * frame_dt
        + ang_vel_xy_l2 * _ANG_VEL_REWARD_SCALE * frame_dt
        + torque_l2 * _JOINT_TORQUE_REWARD_SCALE * frame_dt
        + accel_l2 * _JOINT_ACCEL_REWARD_SCALE * frame_dt
        + action_rate_l2 * _ACTION_RATE_REWARD_SCALE * frame_dt
        + feet_air_reward * _FEET_AIR_TIME_REWARD_SCALE * frame_dt
        + undesired_contacts * _UNDESIRED_CONTACT_REWARD_SCALE * frame_dt
        + flat_orientation_l2 * _FLAT_ORIENTATION_REWARD_SCALE * frame_dt
    )
    rewards[env] = reward

    # --- Termination ---
    terminated = float(0.0)
    if height < 0.25:
        terminated = 1.0
    # Terminate if orientation is too far from upright (base contact proxy)
    if flat_orientation_l2 > 0.49:
        terminated = 1.0
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0
    dones[env] = terminated

    # --- Store current joint velocities for next step's acceleration ---
    for j in range(12):
        prev_joint_vel[env, j] = joint_qd[qd_off + 6 + j]


@wp.kernel
def _apply_actions_kernel(
    actions: wp.array2d[float],
    mujoco_to_lab: wp.array[int],
    joint_pos_initial: wp.array[float],
    joint_target_pos: wp.array[float],
):
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
):
    env = wp.tid()
    if dones[env] > 0.5:
        q_off = env * _Q_STRIDE
        qd_off = env * _QD_STRIDE
        for i in range(wp.static(_Q_STRIDE)):
            joint_q[q_off + i] = initial_q[i]
        for i in range(wp.static(_QD_STRIDE)):
            joint_qd[qd_off + i] = initial_qd[i]
        episode_lengths[env] = 0
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AnymalEnv(RobotEnv):
    obs_dim = _OBS_DIM
    act_dim = _ACT_DIM
    sim_substeps = 4
    sim_dt = 0.005
    max_episode_length = _MAX_EPISODE_LENGTH

    def __init__(self, num_envs: int, device: str | None = None, seed: int = 123):
        super().__init__(num_envs, device=device, seed=seed)
        d = self.device
        self.last_actions = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=d)
        self.prev_actions = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=d)
        self.prev_joint_vel = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=d)
        self.commands = wp.zeros((num_envs, 3), dtype=wp.float32, device=d)
        self.foot_air_time = wp.zeros((num_envs, 4), dtype=wp.float32, device=d)
        self.foot_was_contact = wp.zeros((num_envs, 4), dtype=wp.float32, device=d)
        self._lab_to_mujoco = wp.array(_LAB_TO_MUJOCO, dtype=wp.int32, device=d)
        self._mujoco_to_lab = wp.array(_MUJOCO_TO_LAB, dtype=wp.int32, device=d)
        full_q = self.initial_joint_q.numpy()
        self._joint_pos_initial = wp.array(full_q[7:_Q_STRIDE].astype(np.float32), dtype=wp.float32, device=d)

        # Resolve body indices by name (per single world, then used with body_off)
        body_names = [lbl.split("/")[-1] for lbl in self.model.body_label[: _NUM_BODIES]]
        self._foot_body_ids = [body_names.index(n) for n in ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]]
        self._thigh_body_ids = [body_names.index(n) for n in ["LF_THIGH", "RF_THIGH", "LH_THIGH", "RH_THIGH"]]
        self._foot_ids_wp = wp.array(self._foot_body_ids, dtype=wp.int32, device=d)
        self._thigh_ids_wp = wp.array(self._thigh_body_ids, dtype=wp.int32, device=d)

    def build_robot(self, builder):
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        builder.add_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        for i in range(len(builder.shape_type)):
            if builder.shape_type[i] == GeoType.SPHERE:
                r = builder.shape_scale[i][0]
                builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)
        for name, value in _INITIAL_JOINT_Q_NAMES.items():
            idx = next((i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}")), None)
            if idx is None:
                raise ValueError(f"Joint '{name}' not found")
            builder.joint_q[idx + 6] = value
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = _KE
            builder.joint_target_kd[i] = _KD

    def compute_obs(self):
        wp.launch(
            _compute_obs_kernel,
            dim=self.num_envs,
            inputs=[
                self.state.joint_q,
                self.state.joint_qd,
                self.last_actions,
                self.commands,
                self._joint_pos_initial,
                self._lab_to_mujoco,
                self.obs,
            ],
            device=self.device,
        )

    def compute_reward(self):
        wp.launch(
            _compute_rewards_kernel,
            dim=self.num_envs,
            inputs=[
                self.state.joint_q,
                self.state.joint_qd,
                self.state.body_q,
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
                self.episode_lengths,
                _FRAME_DT,
                self.rewards,
                self.dones,
            ],
            device=self.device,
        )

    def apply_actions(self, actions):
        wp.launch(
            _copy_actions_kernel,
            dim=(self.num_envs, _ACT_DIM),
            inputs=[actions, self.last_actions, self.prev_actions],
            device=self.device,
        )
        wp.launch(
            _apply_actions_kernel,
            dim=(self.num_envs, _ACT_DIM),
            inputs=[actions, self._mujoco_to_lab, self._joint_pos_initial, self.control.joint_target_pos],
            device=self.device,
        )

    def reset_done_envs(self):
        wp.launch(
            _reset_envs_kernel,
            dim=self.num_envs,
            inputs=[
                self.dones,
                self.initial_joint_q,
                self.initial_joint_qd,
                self.state.joint_q,
                self.state.joint_qd,
                self.episode_lengths,
                self.last_actions,
                self.prev_actions,
                self.prev_joint_vel,
                self.foot_air_time,
                self.foot_was_contact,
            ],
            device=self.device,
        )
        wp.launch(
            _randomize_commands_kernel,
            dim=self.num_envs,
            inputs=[self.dones, self.commands, self.rng_counter],
            device=self.device,
        )

    def get_reward_breakdown(self) -> dict[str, float]:
        """Compute individual reward terms on host for logging (NOT in hot path)."""
        q = self.state.joint_q.numpy()
        qd = self.state.joint_qd.numpy()
        ne = self.num_envs
        dt = self.frame_dt

        # Base velocities (world frame, first env as sample)
        heights = q[2::_Q_STRIDE]
        # Average over envs for each term
        lin_vel_xy = np.mean(qd[0::_QD_STRIDE] ** 2 + qd[1::_QD_STRIDE] ** 2)
        z_vel = np.mean(qd[2::_QD_STRIDE] ** 2)
        ang_vel_xy = np.mean(qd[3::_QD_STRIDE] ** 2 + qd[4::_QD_STRIDE] ** 2)

        # Joint velocities
        joint_vels = np.stack([qd[6 + j :: _QD_STRIDE] for j in range(12)], axis=1)
        act_np = self.last_actions.numpy()
        prev_np = self.prev_actions.numpy()
        action_rate = np.mean(np.sum((act_np - prev_np) ** 2, axis=1))

        # Orientation
        grav_xy_sq = []
        for e in range(ne):
            off = e * _Q_STRIDE
            qx, qy, qz, qw = q[off + 3], q[off + 4], q[off + 5], q[off + 6]
            q2w = 2 * qw * qw - 1
            gx = -2 * qw * (qy * (-1)) + 2 * qx * (qz * (-1))
            gy = (-1) * q2w - 2 * qw * (-qx * (-1)) + 2 * qy * (qz * (-1))
            grav_xy_sq.append(gx * gx + gy * gy)
        flat_orient = np.mean(grav_xy_sq)

        return {
            "height": float(heights.mean()),
            "lin_vel_xy": float(lin_vel_xy),
            "z_vel": float(z_vel),
            "ang_vel_xy": float(ang_vel_xy),
            "action_rate": float(action_rate),
            "flat_orient": float(flat_orient),
            "alive_frac": float(np.mean(self.episode_lengths.numpy() > 0)),
        }

    def on_reset(self):
        self.last_actions.zero_()
        self.prev_actions.zero_()
        self.prev_joint_vel.zero_()
        self.foot_air_time.zero_()
        self.foot_was_contact.zero_()
        wp.launch(
            _init_all_commands_kernel, dim=self.num_envs, inputs=[self.commands, self.rng_counter], device=self.device
        )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        num_envs = getattr(args, "num_envs", 4096)
        total_timesteps = getattr(args, "total_timesteps", 10_000_000)
        self.onnx_path = getattr(args, "onnx_output", "anymal_c_trained.onnx")

        if self.is_test:
            total_timesteps = min(total_timesteps, num_envs * 24 * 3)

        self.env = AnymalEnv(num_envs, device=str(self.device))

        self.ac = ActorCritic(
            obs_dim=_OBS_DIM,
            act_dim=_ACT_DIM,
            hidden_sizes=[128, 128, 128],
            activation="elu",
            init_log_std=-1.0,
            bounded_actions=True,
            device=str(self.device),
            seed=42,
        )
        num_steps = 24
        self.trainer = PPOTrainer(
            self.ac,
            num_envs,
            lr=3e-4,
            num_steps=num_steps,
            num_epochs=5,
            num_minibatches=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=1.0,
        )

        self.steps_per_update = num_envs * num_steps
        self.total_updates = total_timesteps // self.steps_per_update
        self.update_idx = 0
        self.training = True
        self.obs = None
        self.viewer.set_model(self.env.model)

    def step(self):
        if self.training and self.update_idx < self.total_updates:
            last_values, self.obs = self.trainer.collect_rollouts(self.env, self.obs)
            self.trainer.buffer.compute_gae(last_values, self.trainer.gamma, self.trainer.gae_lambda)
            self.trainer.update()
            self.update_idx += 1

            if self.update_idx % 20 == 0:
                import resource  # noqa: PLC0415

                stats = self.trainer.get_stats()
                steps = self.update_idx * self.steps_per_update
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
                ep_lens = self.env.episode_lengths.numpy()
                alive = int(np.sum(ep_lens > 0))
                bd = self.env.get_reward_breakdown()
                print(
                    f"Update {self.update_idx:5d}/{self.total_updates}"
                    f" | steps={steps:>10d}"
                    f" | rew={stats['mean_reward']:+.6f}"
                    f" | loss={stats['loss']:.4f}"
                    f" | h={bd['height']:.3f}"
                    f" | alive={alive}/{self.env.num_envs}"
                    f" | ep={ep_lens.mean():.0f}"
                    f" | RSS={rss}MB"
                )
            if self.update_idx % 100 == 0:
                bd = self.env.get_reward_breakdown()
                print(
                    f"  [reward breakdown] lin_vel_xy={bd['lin_vel_xy']:.4f}"
                    f" z_vel={bd['z_vel']:.4f}"
                    f" ang_vel_xy={bd['ang_vel_xy']:.4f}"
                    f" action_rate={bd['action_rate']:.4f}"
                    f" flat_orient={bd['flat_orient']:.4f}"
                )

            if self.update_idx >= self.total_updates:
                self.training = False
                norm = self.trainer.obs_normalizer
                export_to_onnx(
                    self.ac.actor,
                    obs_dim=_OBS_DIM,
                    path=self.onnx_path,
                    obs_mean=norm.mean.numpy(),
                    obs_inv_std=norm.inv_std.numpy(),
                )
                print(f"Training complete. Policy saved to {self.onnx_path}")
        else:
            if self.obs is None:
                self.obs = self.env.reset()
            self.trainer.obs_normalizer.normalize(self.obs, self.trainer._norm_obs)
            mean = self.ac.actor.forward(self.trainer._norm_obs)
            self.obs, _, _ = self.env.step(mean)

    def render(self):
        self.viewer.begin_frame(self.env.sim_time)
        self.viewer.log_state(self.env.state_0)
        self.viewer.end_frame()

    def test_final(self):
        assert self.update_idx > 0, "No training updates were performed"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--num-envs", type=int, default=4096)
        parser.add_argument("--total-timesteps", type=int, default=10_000_000)
        parser.add_argument("--onnx-output", type=str, default="anymal_c_trained.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
