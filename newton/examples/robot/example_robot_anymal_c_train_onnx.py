# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Train (ONNX)
#
# Trains an ANYmal C walking policy from scratch using the built-in PPO
# trainer and Warp kernels.  No PyTorch dependency.  The trained policy
# is exported to ONNX and can be loaded by OnnxRuntime for inference.
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
from newton._src.ppo import ActorCritic, PPOTrainer, _increment_counter_kernel, export_actor_to_onnx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBS_DIM = 48
_ACT_DIM = 12
_Q_STRIDE = 19  # 7 (free joint) + 12 (revolute)
_QD_STRIDE = 18  # 6 (free joint) + 12 (revolute)
_NUM_BODIES = 13
_SIM_SUBSTEPS = 4
_SIM_DT = 0.005  # per substep
_FRAME_DT = _SIM_DT * _SIM_SUBSTEPS  # 0.02s -> 50 Hz
_MAX_EPISODE_LENGTH = 1000  # 20 seconds at 50 Hz

# Joint ordering remapping (lab <-> mujoco convention)
_LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
_MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

# Initial joint positions (lab ordering, matching the walk example)
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
    """Compute 48-D observation for each environment entirely on device."""
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE

    # Root quaternion (XYZW convention in Newton)
    qx = joint_q[q_off + 3]
    qy = joint_q[q_off + 4]
    qz = joint_q[q_off + 5]
    qw = joint_q[q_off + 6]

    # Root linear velocity (world frame)
    vx = joint_qd[qd_off + 0]
    vy = joint_qd[qd_off + 1]
    vz = joint_qd[qd_off + 2]

    # Root angular velocity (world frame)
    wx = joint_qd[qd_off + 3]
    wy = joint_qd[qd_off + 4]
    wz = joint_qd[qd_off + 5]

    # quat_rotate_inverse: rotate vector by inverse of quaternion
    # v_body = q^{-1} * v * q
    q2w2m1 = 2.0 * qw * qw - 1.0

    # Linear velocity in body frame
    cross_vx = qy * vz - qz * vy
    cross_vy = qz * vx - qx * vz
    cross_vz = qx * vy - qy * vx
    dot_qv = qx * vx + qy * vy + qz * vz
    vel_bx = vx * q2w2m1 - 2.0 * qw * cross_vx + 2.0 * qx * dot_qv
    vel_by = vy * q2w2m1 - 2.0 * qw * cross_vy + 2.0 * qy * dot_qv
    vel_bz = vz * q2w2m1 - 2.0 * qw * cross_vz + 2.0 * qz * dot_qv

    # Angular velocity in body frame
    cross_wx = qy * wz - qz * wy
    cross_wy = qz * wx - qx * wz
    cross_wz = qx * wy - qy * wx
    dot_qw = qx * wx + qy * wy + qz * wz
    avel_bx = wx * q2w2m1 - 2.0 * qw * cross_wx + 2.0 * qx * dot_qw
    avel_by = wy * q2w2m1 - 2.0 * qw * cross_wy + 2.0 * qy * dot_qw
    avel_bz = wz * q2w2m1 - 2.0 * qw * cross_wz + 2.0 * qz * dot_qw

    # Gravity vector in body frame (world gravity is [0, 0, -1])
    gx = 0.0
    gy = 0.0
    gz = -1.0
    cross_gx = qy * gz - qz * gy
    cross_gy = qz * gx - qx * gz
    cross_gz = qx * gy - qy * gx
    dot_qg = qx * gx + qy * gy + qz * gz
    grav_bx = gx * q2w2m1 - 2.0 * qw * cross_gx + 2.0 * qx * dot_qg
    grav_by = gy * q2w2m1 - 2.0 * qw * cross_gy + 2.0 * qy * dot_qg
    grav_bz = gz * q2w2m1 - 2.0 * qw * cross_gz + 2.0 * qz * dot_qg

    # Write observation: [vel_b(3), avel_b(3), grav_b(3), cmd(3),
    #                     joint_pos_rel(12), joint_vel(12), last_act(12)]
    obs[env, 0] = vel_bx
    obs[env, 1] = vel_by
    obs[env, 2] = vel_bz
    obs[env, 3] = avel_bx
    obs[env, 4] = avel_by
    obs[env, 5] = avel_bz
    obs[env, 6] = grav_bx
    obs[env, 7] = grav_by
    obs[env, 8] = grav_bz
    obs[env, 9] = commands[env, 0]
    obs[env, 10] = commands[env, 1]
    obs[env, 11] = commands[env, 2]

    # Joint positions relative to initial (remapped to mujoco ordering)
    for j in range(12):
        mj = lab_to_mujoco[j]
        pos = joint_q[q_off + 7 + j] - joint_pos_initial[j]
        vel = joint_qd[qd_off + 6 + j]
        obs[env, 12 + mj] = pos
        obs[env, 24 + mj] = vel

    # Last actions (already in mujoco ordering)
    for j in range(12):
        obs[env, 36 + j] = last_actions[env, j]


@wp.kernel
def _compute_rewards_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    last_actions: wp.array2d[float],
    prev_actions: wp.array2d[float],
    commands: wp.array2d[float],
    episode_lengths: wp.array[int],
    rewards: wp.array[float],
    dones: wp.array[float],
):
    """Compute per-environment reward and termination."""
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE

    # Root height
    height = joint_q[q_off + 2]

    # Root quaternion
    qx = joint_q[q_off + 3]
    qy = joint_q[q_off + 4]
    qz = joint_q[q_off + 5]
    qw = joint_q[q_off + 6]

    # Velocities in world frame
    vx = joint_qd[qd_off + 0]
    vy = joint_qd[qd_off + 1]
    wx = joint_qd[qd_off + 3]
    wy = joint_qd[qd_off + 4]
    wz = joint_qd[qd_off + 5]

    # Body-frame velocities via quat_rotate_inverse
    q2w2m1 = 2.0 * qw * qw - 1.0

    cross_vx = qy * 0.0 - qz * vy
    cross_vy = qz * vx - qx * 0.0
    dot_qv = qx * vx + qy * vy
    vel_bx = vx * q2w2m1 - 2.0 * qw * cross_vx + 2.0 * qx * dot_qv
    vel_by = vy * q2w2m1 - 2.0 * qw * cross_vy + 2.0 * qy * dot_qv

    cross_wz = qx * wy - qy * wx
    dot_qw_z = qx * wx + qy * wy + qz * wz
    avel_bz = wz * q2w2m1 - 2.0 * qw * cross_wz + 2.0 * qz * dot_qw_z

    # Projected gravity (only need x, y components for tilt check)
    gz = -1.0
    cross_gx = qy * gz
    cross_gy = -qx * gz
    dot_qg = qz * gz
    grav_bx = -2.0 * qw * cross_gx + 2.0 * qx * dot_qg
    grav_by = gz * q2w2m1 - 2.0 * qw * cross_gy + 2.0 * qy * dot_qg

    # Commands
    cmd_fwd = commands[env, 0]
    cmd_lat = commands[env, 1]
    cmd_yaw = commands[env, 2]

    # Reward terms
    fwd_reward = wp.exp(-4.0 * (vel_bx - cmd_fwd) * (vel_bx - cmd_fwd))
    lat_reward = wp.exp(-4.0 * (vel_by - cmd_lat) * (vel_by - cmd_lat)) * 0.5
    yaw_reward = wp.exp(-4.0 * (avel_bz - cmd_yaw) * (avel_bz - cmd_yaw)) * 0.5
    alive_bonus = 1.0
    orientation_penalty = -5.0 * (grav_bx * grav_bx + grav_by * grav_by)

    # Action rate penalty
    action_rate = float(0.0)
    for j in range(12):
        diff = last_actions[env, j] - prev_actions[env, j]
        action_rate = action_rate + diff * diff
    action_rate_penalty = -0.01 * action_rate

    total_reward = fwd_reward + lat_reward + yaw_reward + alive_bonus + orientation_penalty + action_rate_penalty

    # Termination conditions
    tilt_sq = grav_bx * grav_bx + grav_by * grav_by
    terminated = float(0.0)
    if height < 0.25:
        terminated = 1.0
    if tilt_sq > 0.49:  # ~45 deg tilt
        terminated = 1.0
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0

    rewards[env] = total_reward
    dones[env] = terminated


@wp.kernel
def _apply_actions_kernel(
    actions: wp.array2d[float],
    mujoco_to_lab: wp.array[int],
    joint_pos_initial: wp.array[float],
    joint_target_pos: wp.array[float],
):
    """Convert policy actions to joint position targets."""
    env, j = wp.tid()
    lab_j = mujoco_to_lab[j]
    target = joint_pos_initial[lab_j] + 0.5 * actions[env, j]
    joint_target_pos[env * _QD_STRIDE + 6 + lab_j] = target


@wp.kernel
def _reset_envs_kernel(
    dones: wp.array[float],
    initial_joint_q: wp.array[float],
    initial_joint_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    episode_lengths: wp.array[int],
    last_actions: wp.array2d[float],
    prev_actions: wp.array2d[float],
):
    """Reset environments where done flag is set."""
    env = wp.tid()
    if dones[env] > 0.5:
        q_off = env * _Q_STRIDE
        qd_off = env * _QD_STRIDE
        for i in range(wp.static(_Q_STRIDE)):
            joint_q[q_off + i] = initial_joint_q[i]
        for i in range(wp.static(_QD_STRIDE)):
            joint_qd[qd_off + i] = initial_joint_qd[i]
        episode_lengths[env] = 0
        for i in range(12):
            last_actions[env, i] = 0.0
            prev_actions[env, i] = 0.0


@wp.kernel
def _randomize_commands_kernel(
    dones: wp.array[float],
    commands: wp.array2d[float],
    rng_counter: wp.array[int],
):
    """Randomize velocity commands for reset environments."""
    env = wp.tid()
    if dones[env] > 0.5:
        step_offset = rng_counter[0]
        rng = wp.rand_init(42, step_offset * 1000 + env)
        # Forward: [-0.5, 1.5], biased toward forward walking
        commands[env, 0] = wp.randf(rng) * 2.0 - 0.5
        # Lateral: [-0.5, 0.5]
        commands[env, 1] = wp.randf(rng) * 1.0 - 0.5
        # Yaw rate: [-0.5, 0.5]
        commands[env, 2] = wp.randf(rng) * 1.0 - 0.5


@wp.kernel
def _copy_actions_kernel(
    new_actions: wp.array2d[float],
    last_actions: wp.array2d[float],
    prev_actions: wp.array2d[float],
):
    """Copy last_actions -> prev_actions, new_actions -> last_actions."""
    env, j = wp.tid()
    prev_actions[env, j] = last_actions[env, j]
    last_actions[env, j] = new_actions[env, j]


@wp.kernel
def _increment_episode_kernel(episode_lengths: wp.array[int]):
    env = wp.tid()
    episode_lengths[env] = episode_lengths[env] + 1


@wp.kernel
def _init_all_commands_kernel(
    commands: wp.array2d[float],
    rng_counter: wp.array[int],
):
    """Randomize commands for all environments at init."""
    env = wp.tid()
    seed_val = rng_counter[0]
    rng = wp.rand_init(seed_val, env * 3)
    commands[env, 0] = wp.randf(rng) * 2.0 - 0.5
    commands[env, 1] = wp.randf(rng) * 1.0 - 0.5
    commands[env, 2] = wp.randf(rng) * 1.0 - 0.5


# ---------------------------------------------------------------------------
# Vectorized environment
# ---------------------------------------------------------------------------


class AnymalVecEnv:
    """Vectorized ANYmal C environment for PPO training.

    All arrays are pre-allocated ``wp.array`` on the target device.
    The physics substep loop is CUDA-graph captured when running on GPU.
    """

    num_envs: int
    obs_dim: int = _OBS_DIM
    act_dim: int = _ACT_DIM

    def __init__(self, num_envs: int, device: str | None = None, seed: int = 123):
        self.num_envs = num_envs
        self._device = wp.get_device(device)
        self.sim_time = 0.0

        # -- Build single-robot articulation builder --
        art_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(art_builder)
        art_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        art_builder.default_shape_cfg.ke = 5.0e4
        art_builder.default_shape_cfg.kd = 5.0e2
        art_builder.default_shape_cfg.kf = 1.0e3
        art_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        art_builder.add_urdf(
            stage_path,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.62),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Enlarge foot collision spheres
        for i in range(len(art_builder.shape_type)):
            if art_builder.shape_type[i] == GeoType.SPHERE:
                r = art_builder.shape_scale[i][0]
                art_builder.shape_scale[i] = (r * 2.0, 0.0, 0.0)

        # Set initial joint positions
        for name, value in _INITIAL_JOINT_Q_NAMES.items():
            idx = next(
                (i for i, lbl in enumerate(art_builder.joint_label) if lbl.endswith(f"/{name}")),
                None,
            )
            if idx is None:
                raise ValueError(f"Joint '{name}' not found")
            art_builder.joint_q[idx + 6] = value

        # PD gains
        for i in range(len(art_builder.joint_target_ke)):
            art_builder.joint_target_ke[i] = 150
            art_builder.joint_target_kd[i] = 5

        # -- Replicate into N worlds --
        builder = newton.ModelBuilder()
        builder.replicate(art_builder, num_envs)
        builder.add_ground_plane()

        self.model = builder.finalize(device=self._device)

        # -- Solver --
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            ls_parallel=False,
            ls_iterations=50,
            njmax=50 * num_envs,
            nconmax=100 * num_envs,
        )

        # -- State and control --
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # -- Store initial joint state (single world) for resets --
        full_q = self.state_0.joint_q.numpy()
        full_qd = self.state_0.joint_qd.numpy()
        self._initial_joint_q = wp.array(full_q[:_Q_STRIDE].astype(np.float32), dtype=wp.float32, device=self._device)
        self._initial_joint_qd = wp.array(
            full_qd[:_QD_STRIDE].astype(np.float32), dtype=wp.float32, device=self._device
        )

        # Extract initial joint positions (12 revolute DOFs, lab ordering)
        self._joint_pos_initial = wp.array(
            full_q[7:_Q_STRIDE].astype(np.float32), dtype=wp.float32, device=self._device
        )

        # -- On-device index arrays --
        self._lab_to_mujoco = wp.array(_LAB_TO_MUJOCO, dtype=wp.int32, device=self._device)
        self._mujoco_to_lab = wp.array(_MUJOCO_TO_LAB, dtype=wp.int32, device=self._device)

        # -- Pre-allocated buffers --
        d = self._device
        self.obs = wp.zeros((num_envs, _OBS_DIM), dtype=wp.float32, device=d)
        self.rewards = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.dones = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.last_actions = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=d)
        self.prev_actions = wp.zeros((num_envs, _ACT_DIM), dtype=wp.float32, device=d)
        self.episode_lengths = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self.commands = wp.zeros((num_envs, 3), dtype=wp.float32, device=d)
        self._rng_counter = wp.array([seed], dtype=wp.int32, device=d)

        # -- CUDA graph capture for physics substeps --
        self._graph = None
        if self._device.is_cuda:
            # Pre-allocate control target buffer for graph capture
            self.control.joint_target_pos = wp.zeros(num_envs * _QD_STRIDE, dtype=wp.float32, device=d)
            with wp.ScopedCapture() as capture:
                self._simulate()
            self._graph = capture.graph

    def _simulate(self):
        """Run physics substeps (captured as CUDA graph)."""
        for _ in range(_SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, _SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self) -> wp.array:
        """Reset all environments and return initial observations."""
        # Copy initial state to all worlds
        full_q = self._initial_joint_q.numpy()
        full_qd = self._initial_joint_qd.numpy()
        q_tiled = np.tile(full_q, self.num_envs).astype(np.float32)
        qd_tiled = np.tile(full_qd, self.num_envs).astype(np.float32)
        wp.copy(self.state_0.joint_q, wp.array(q_tiled, dtype=wp.float32, device=self._device))
        wp.copy(self.state_0.joint_qd, wp.array(qd_tiled, dtype=wp.float32, device=self._device))

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Zero buffers
        self.last_actions.zero_()
        self.prev_actions.zero_()
        self.episode_lengths.zero_()
        self.dones.zero_()
        self.sim_time = 0.0

        # Randomize commands
        wp.launch(
            _init_all_commands_kernel,
            dim=self.num_envs,
            inputs=[self.commands, self._rng_counter],
            device=self._device,
        )

        # Compute initial observations
        wp.launch(
            _compute_obs_kernel,
            dim=self.num_envs,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.last_actions,
                self.commands,
                self._joint_pos_initial,
                self._lab_to_mujoco,
                self.obs,
            ],
            device=self._device,
        )
        return self.obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Step all environments with the given actions.

        Args:
            actions: Policy output, shape ``(num_envs, 12)`` in mujoco ordering.

        Returns:
            ``(obs, rewards, dones)`` -- all pre-allocated ``wp.array``.
        """
        d = self._device
        ne = self.num_envs

        # 1. Copy actions to history buffers
        wp.launch(
            _copy_actions_kernel, dim=(ne, _ACT_DIM), inputs=[actions, self.last_actions, self.prev_actions], device=d
        )

        # 2. Apply actions as joint position targets
        wp.launch(
            _apply_actions_kernel,
            dim=(ne, _ACT_DIM),
            inputs=[actions, self._mujoco_to_lab, self._joint_pos_initial, self.control.joint_target_pos],
            device=d,
        )

        # 3. Physics simulation (CUDA graph captured)
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        # 4. Increment episode counters
        wp.launch(_increment_episode_kernel, dim=ne, inputs=[self.episode_lengths], device=d)

        # 5. Compute rewards and termination
        wp.launch(
            _compute_rewards_kernel,
            dim=ne,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.last_actions,
                self.prev_actions,
                self.commands,
                self.episode_lengths,
                self.rewards,
                self.dones,
            ],
            device=d,
        )

        # 6. Reset terminated environments
        wp.launch(
            _reset_envs_kernel,
            dim=ne,
            inputs=[
                self.dones,
                self._initial_joint_q,
                self._initial_joint_qd,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_lengths,
                self.last_actions,
                self.prev_actions,
            ],
            device=d,
        )

        # 7. Update body transforms for reset environments
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # 8. Randomize commands for reset environments
        wp.launch(_increment_counter_kernel, dim=1, inputs=[self._rng_counter], device=d)
        wp.launch(
            _randomize_commands_kernel,
            dim=ne,
            inputs=[self.dones, self.commands, self._rng_counter],
            device=d,
        )

        # 9. Compute observations
        wp.launch(
            _compute_obs_kernel,
            dim=ne,
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
        return self.obs, self.rewards, self.dones


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        num_envs = getattr(args, "num_envs", 64)
        total_timesteps = getattr(args, "total_timesteps", 1_000_000)
        self.onnx_path = getattr(args, "onnx_output", "anymal_c_trained.onnx")

        # Build vectorized environment
        self.env = AnymalVecEnv(num_envs, device=str(self.device))

        # For tests, reduce training
        if self.is_test:
            total_timesteps = min(total_timesteps, num_envs * 24 * 3)

        # Build actor-critic and trainer
        num_steps = 24
        num_minibatches = 4
        self.ac = ActorCritic(
            obs_dim=_OBS_DIM,
            act_dim=_ACT_DIM,
            hidden_sizes=[128, 128, 128],
            activation="elu",
            init_log_std=-1.0,
            device=str(self.device),
            seed=42,
        )
        self.trainer = PPOTrainer(
            self.ac,
            num_envs,
            lr=3e-4,
            num_steps=num_steps,
            num_epochs=5,
            num_minibatches=num_minibatches,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=1.0,
        )

        # Training state
        self.steps_per_update = num_envs * num_steps
        self.total_updates = total_timesteps // self.steps_per_update
        self.update_idx = 0
        self.training = True
        self.obs: wp.array | None = None

        self.viewer.set_model(self.env.model)

    def step(self):
        if self.training and self.update_idx < self.total_updates:
            # One PPO update per frame
            last_values, self.obs = self.trainer.collect_rollouts(self.env, self.obs)
            self.trainer.buffer.compute_gae(last_values, self.trainer.gamma, self.trainer.gae_lambda)
            avg_loss = self.trainer.update()

            self.update_idx += 1
            steps = self.update_idx * self.steps_per_update

            if self.update_idx % 10 == 0:
                mean_rew = self.trainer.buffer.mean_reward()
                print(
                    f"Update {self.update_idx}/{self.total_updates} | steps={steps}"
                    f" | loss={avg_loss:.4f} | mean_reward={mean_rew:.4f}"
                )

            if self.update_idx >= self.total_updates:
                self.training = False
                export_actor_to_onnx(self.ac.actor, obs_dim=_OBS_DIM, path=self.onnx_path)
                print(f"Training complete. Policy saved to {self.onnx_path}")
        else:
            # Inference: use deterministic policy (mean, no noise)
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
        # Verify training ran without errors
        assert self.update_idx > 0, "No training updates were performed"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments.")
        parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps.")
        parser.add_argument("--onnx-output", type=str, default="anymal_c_trained.onnx", help="Output ONNX path.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
