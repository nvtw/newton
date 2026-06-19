# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.utils

from .env import collect_ppo_rollout
from .ppo import BufferRollout, TrainerPPO

ACTION_DIM_G1 = 29
OBS_DIM_G1 = 98
NANOG1_PHASE_PERIOD = 40

_DEFAULT_JOINT_POS_G1 = (
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    0.0,
    0.0,
    0.0,
    0.20,
    0.20,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
    0.20,
    -0.20,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
)

_UNITREE_KP_G1 = (
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    2.0,
    2.0,
    2.0,
    75.0,
    75.0,
    75.0,
    75.0,
    2.0,
    2.0,
    2.0,
)

_UNITREE_KD_G1 = (
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
)

_CTRL_LO_G1 = (
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.618,
    -0.52,
    -0.52,
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
)

_CTRL_HI_G1 = (
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.618,
    0.52,
    0.52,
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
)


@wp.func
def _clip_float(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _quat_rotate_inverse_wxyz(qw: wp.float32, qx: wp.float32, qy: wp.float32, qz: wp.float32, v: wp.vec3) -> wp.vec3:
    q = wp.vec3(qx, qy, qz)
    a = v * (wp.float32(2.0) * qw * qw - wp.float32(1.0))
    b = wp.cross(q, v) * qw * wp.float32(2.0)
    c = q * (wp.dot(q, v) * wp.float32(2.0))
    return a - b + c


@wp.kernel
def g1_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    ctrl_lower: wp.array[wp.float32],
    ctrl_upper: wp.array[wp.float32],
    action_scale: wp.float32,
    controlled_action_count: wp.int32,
    dof_stride: wp.int32,
    coord_stride: wp.int32,
    target_uses_coord_layout: wp.int32,
    current_actions: wp.array2d[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    world, action = wp.tid()
    value = _clip_float(actions[world, action], wp.float32(-1.0), wp.float32(1.0))
    if action >= controlled_action_count:
        value = wp.float32(0.0)
    current_actions[world, action] = value

    target = default_joint_pos[action] + action_scale * value
    target = _clip_float(target, ctrl_lower[action], ctrl_upper[action])
    if target_uses_coord_layout != 0:
        joint_target_q[world * coord_stride + wp.int32(7) + action] = target
    else:
        joint_target_q[world * dof_stride + wp.int32(6) + action] = target


@wp.kernel
def g1_observe_reward_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    command: wp.array2d[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    phase_period: wp.int32,
    min_base_height: wp.float32,
    min_upright_cos: wp.float32,
    action_scale: wp.float32,
    w_track_lin: wp.float32,
    w_track_ang: wp.float32,
    w_lin_vel_z: wp.float32,
    w_ang_vel_xy: wp.float32,
    w_orientation: wp.float32,
    w_torque: wp.float32,
    w_action_rate: wp.float32,
    w_alive: wp.float32,
    w_termination: wp.float32,
    actuator_ke: wp.array[wp.float32],
    actuator_kd: wp.array[wp.float32],
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
):
    world, col = wp.tid()
    q_base = world * coord_stride
    qd_base = world * dof_stride

    qw = joint_q[q_base + wp.int32(3)]
    qx = joint_q[q_base + wp.int32(4)]
    qy = joint_q[q_base + wp.int32(5)]
    qz = joint_q[q_base + wp.int32(6)]
    lin_w = wp.vec3(joint_qd[qd_base], joint_qd[qd_base + wp.int32(1)], joint_qd[qd_base + wp.int32(2)])
    ang = wp.vec3(
        joint_qd[qd_base + wp.int32(3)],
        joint_qd[qd_base + wp.int32(4)],
        joint_qd[qd_base + wp.int32(5)],
    )
    lin_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, lin_w)
    gravity_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, wp.vec3(0.0, 0.0, -1.0))

    value = wp.float32(0.0)
    if col < wp.int32(3):
        value = wp.float32(0.25) * ang[col]
    elif col < wp.int32(6):
        value = gravity_b[col - wp.int32(3)]
    elif col < wp.int32(9):
        value = command[world, col - wp.int32(6)]
    elif col < wp.int32(38):
        j = col - wp.int32(9)
        value = joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j]
    elif col < wp.int32(67):
        j = col - wp.int32(38)
        value = wp.float32(0.05) * joint_qd[qd_base + wp.int32(6) + j]
    elif col < wp.int32(96):
        value = current_actions[world, col - wp.int32(67)]
    elif col == wp.int32(96):
        phase_step = episode_steps[world] % phase_period
        phase = wp.float32(6.283185307179586) * wp.float32(phase_step) / wp.float32(phase_period)
        value = wp.sin(phase)
    else:
        phase_step = episode_steps[world] % phase_period
        phase = wp.float32(6.283185307179586) * wp.float32(phase_step) / wp.float32(phase_period)
        value = wp.cos(phase)
    obs[world, col] = value

    if col == 0:
        vx_err = command[world, 0] - lin_b[0]
        vy_err = command[world, 1] - lin_b[1]
        yaw_err = command[world, 2] - ang[2]
        track_lin = wp.exp(-(vx_err * vx_err + vy_err * vy_err) / wp.float32(0.25))
        track_ang = wp.exp(-(yaw_err * yaw_err) / wp.float32(0.25))
        lin_vel_z_penalty = lin_b[2] * lin_b[2]
        ang_vel_xy_penalty = ang[0] * ang[0] + ang[1] * ang[1]
        orientation_penalty = gravity_b[0] * gravity_b[0] + gravity_b[1] * gravity_b[1]
        upright = _clip_float(-gravity_b[2], wp.float32(0.0), wp.float32(1.0))

        action_rate_penalty = wp.float32(0.0)
        power_proxy = wp.float32(0.0)
        for j in range(ACTION_DIM_G1):
            da = current_actions[world, j] - previous_actions[world, j]
            action_rate_penalty = action_rate_penalty + da * da
            q_idx = q_base + wp.int32(7) + j
            qd_idx = qd_base + wp.int32(6) + j
            qd = joint_qd[qd_idx]
            target = default_joint_pos[j] + current_actions[world, j] * action_scale
            tau_proxy = actuator_ke[j] * (target - joint_q[q_idx]) - actuator_kd[j] * qd
            power_proxy = power_proxy + wp.abs(tau_proxy * qd)

        fall = wp.float32(0.0)
        if joint_q[q_base + wp.int32(2)] < min_base_height or upright < min_upright_cos:
            fall = wp.float32(1.0)

        reward = (
            w_track_lin * track_lin
            + w_track_ang * track_ang
            + w_lin_vel_z * lin_vel_z_penalty
            + w_ang_vel_xy * ang_vel_xy_penalty
            + w_orientation * orientation_penalty
            + w_torque * power_proxy
            + w_action_rate * action_rate_penalty
            + w_alive
            + w_termination * fall
        )
        rewards[world] = reward
        successes[world] = track_lin

        done = wp.float32(0.0)
        if fall > wp.float32(0.5):
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0):
            if episode_steps[world] >= max_episode_steps:
                done = wp.float32(1.0)
        dones[world] = done


@wp.kernel
def g1_increment_episode_steps_kernel(episode_steps: wp.array[wp.int32]):
    world = wp.tid()
    episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel
def g1_reset_done_worlds_kernel(
    seed: wp.int32,
    reset_noise: wp.float32,
    dones: wp.array[wp.float32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    action_dim: wp.int32,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    previous_actions: wp.array2d[wp.float32],
    current_actions: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    if dones[world] <= wp.float32(0.5):
        return
    if col < coord_stride:
        idx = world * coord_stride + col
        value = default_joint_q[idx]
        if col >= wp.int32(7) and col < wp.int32(7) + action_dim and reset_noise > wp.float32(0.0):
            rng = wp.rand_init(seed, world * coord_stride + col)
            value = value + reset_noise * (wp.float32(2.0) * wp.randf(rng) - wp.float32(1.0))
        joint_q[idx] = value
    if col < dof_stride:
        idx = world * dof_stride + col
        joint_qd[idx] = default_joint_qd[idx]
    if col < action_dim:
        previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
    if col == 0:
        episode_steps[world] = wp.int32(0)


@dataclass
class ConfigEnvG1PhoenX:
    """Configuration for :class:`EnvG1PhoenX`.

    Args:
        world_count: Number of vectorized G1 worlds.
        frame_dt: Policy step duration [s].
        sim_substeps: PhoenX physics steps per policy step.
        solver_iterations: PhoenX position iterations per substep.
        velocity_iterations: PhoenX velocity iterations per substep.
        action_scale: Position target scale [rad].
        controlled_action_count: Number of leading policy actions applied to joints.
        command: Target body-frame command ``(vx, vy, yaw_rate)`` [m/s, m/s, rad/s].
        max_episode_steps: Episode timeout in policy steps. Use ``0`` to disable.
        reset_noise: Uniform joint-position reset noise half-width [rad].
        min_base_height: Episode ends below this base height [m].
        min_upright_cos: Episode ends below this base-upright cosine threshold.
        phase_period: Gait clock period in policy steps.
        w_track_lin: Linear XY velocity tracking reward scale.
        w_track_ang: Yaw-rate tracking reward scale.
        w_lin_vel_z: Vertical body-velocity penalty scale.
        w_ang_vel_xy: Base roll/pitch angular-velocity penalty scale.
        w_orientation: Projected-gravity tilt penalty scale.
        w_torque: Mechanical power proxy penalty scale.
        w_action_rate: Action-rate penalty scale.
        w_alive: Alive reward per policy step.
        w_termination: Termination reward applied on fall.
        parse_meshes: Import MJCF mesh collision geoms. Disable for fast
            RL runs that use the primitive foot and arm geoms only.
        auto_reset: Reset worlds whose done flag is set after each step.
    """

    world_count: int = 4096
    frame_dt: float = 1.0 / 50.0
    sim_substeps: int = 5
    solver_iterations: int = 2
    velocity_iterations: int = 1
    action_scale: float = 0.25
    controlled_action_count: int = 12
    command: tuple[float, float, float] = (0.8, 0.0, 0.0)
    max_episode_steps: int = 1000
    reset_noise: float = 0.05
    min_base_height: float = 0.35
    min_upright_cos: float = 0.6
    phase_period: int = NANOG1_PHASE_PERIOD
    w_track_lin: float = 2.5
    w_track_ang: float = 1.25
    w_lin_vel_z: float = -2.0
    w_ang_vel_xy: float = -1.3
    w_orientation: float = -10.0
    w_torque: float = -2.0e-5
    w_action_rate: float = -0.01
    w_alive: float = 3.0
    w_termination: float = -1.0
    parse_meshes: bool = False
    auto_reset: bool = True


class EnvG1PhoenX:
    """Warp-only Unitree G1 locomotion environment backed by SolverPhoenX.

    The observation layout follows nanoG1 v3: scaled base angular velocity,
    projected gravity, command, joint position deltas, scaled joint velocities,
    previous actions, and a sin/cos gait phase clock.

    Args:
        config: Environment and reward configuration.
        device: Warp device.
    """

    obs_dim = OBS_DIM_G1
    action_dim = ACTION_DIM_G1

    def __init__(self, config: ConfigEnvG1PhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvG1PhoenX()
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if not 0 < int(self.config.controlled_action_count) <= ACTION_DIM_G1:
            raise ValueError("controlled_action_count must be in [1, ACTION_DIM_G1]")
        if int(self.config.phase_period) <= 0:
            raise ValueError("phase_period must be positive")

        self.model = self._build_model()
        self.coord_stride = int(self.model.joint_coord_count) // self.world_count
        self.dof_stride = int(self.model.joint_dof_count) // self.world_count
        if self.coord_stride != 7 + ACTION_DIM_G1 or self.dof_stride != 6 + ACTION_DIM_G1:
            raise RuntimeError(
                f"Expected nanoG1 dimensions coord={7 + ACTION_DIM_G1}, dof={6 + ACTION_DIM_G1}; "
                f"got coord={self.coord_stride}, dof={self.dof_stride}"
            )

        self.solver = self._make_solver()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.default_joint_pos = wp.array(
            np.asarray(_DEFAULT_JOINT_POS_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.ctrl_lower = wp.array(np.asarray(_CTRL_LO_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.ctrl_upper = wp.array(np.asarray(_CTRL_HI_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.actuator_ke = wp.array(np.asarray(_UNITREE_KP_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.actuator_kd = wp.array(np.asarray(_UNITREE_KD_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        command_np = np.tile(np.asarray(self.config.command, dtype=np.float32), (self.world_count, 1))
        self.command = wp.array(command_np, dtype=wp.float32, device=self.device)
        self.episode_steps = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros((self.world_count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._reset_seed = 0
        self.sim_time = 0.0

        self.reset()

    def _build_model(self):
        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75
        articulation_builder.default_shape_cfg.gap = 0.0

        asset_path = newton.utils.download_asset("unitree_g1")
        articulation_builder.add_mjcf(
            str(asset_path / "mjcf" / "g1_29dof.xml"),
            floating=None,
            enable_self_collisions=False,
            ignore_names=("floor", "ground"),
            parse_visuals=False,
            parse_meshes=bool(self.config.parse_meshes),
            ignore_inertial_definitions=False,
        )
        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.785]
        articulation_builder.joint_q[3:7] = [1.0, 0.0, 0.0, 0.0]
        articulation_builder.joint_q[7 : 7 + ACTION_DIM_G1] = list(_DEFAULT_JOINT_POS_G1)
        for i in range(ACTION_DIM_G1):
            dof = i + 6
            articulation_builder.joint_target_ke[dof] = _UNITREE_KP_G1[i]
            articulation_builder.joint_target_kd[dof] = _UNITREE_KD_G1[i]
            articulation_builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(articulation_builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.default_shape_cfg.gap = 0.0
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        return model

    def _make_solver(self):
        return newton.solvers.SolverPhoenX(
            self.model,
            substeps=int(self.config.sim_substeps),
            solver_iterations=int(self.config.solver_iterations),
            velocity_iterations=int(self.config.velocity_iterations),
            articulation_dvi=False,
            articulation_dvi_replaces_joint_pgs=False,
        )

    def set_command(self, command: tuple[float, float, float]) -> None:
        """Set the same body-frame command for every world [m/s, m/s, rad/s]."""

        cmd = (float(command[0]), float(command[1]), float(command[2]))
        self.config.command = cmd
        command_np = np.tile(np.asarray(cmd, dtype=np.float32), (self.world_count, 1))
        self.command.assign(command_np)

    def set_commands(self, commands: np.ndarray) -> None:
        """Set per-world body-frame commands [m/s, m/s, rad/s]."""

        cmds = np.asarray(commands, dtype=np.float32)
        if cmds.shape != (self.world_count, 3):
            raise ValueError(f"Expected commands with shape {(self.world_count, 3)}, got {cmds.shape}")
        self.config.command = (float(cmds[0, 0]), float(cmds[0, 1]), float(cmds[0, 2]))
        self.command.assign(cmds)

    def observe(self) -> wp.array:
        """Update and return the current observation array."""

        wp.launch(
            g1_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.default_joint_pos,
                self.current_actions,
                self.previous_actions,
                self.command,
                self.coord_stride,
                self.dof_stride,
                self.episode_steps,
                self.config.max_episode_steps,
                self.config.phase_period,
                self.config.min_base_height,
                self.config.min_upright_cos,
                self.config.action_scale,
                self.config.w_track_lin,
                self.config.w_track_ang,
                self.config.w_lin_vel_z,
                self.config.w_ang_vel_xy,
                self.config.w_orientation,
                self.config.w_torque,
                self.config.w_action_rate,
                self.config.w_alive,
                self.config.w_termination,
                self.actuator_ke,
                self.actuator_kd,
            ],
            outputs=[self.obs, self.rewards, self.dones, self.successes],
            device=self.device,
        )
        return self.obs

    def reset(self) -> wp.array:
        """Reset all worlds and return observations."""

        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        self.current_actions.zero_()
        self.previous_actions.zero_()
        self.episode_steps.zero_()
        self.dones.zero_()
        self.successes.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        return self.observe()

    def reset_done(self) -> None:
        """Reset worlds whose done flag is set."""

        max_cols = max(self.coord_stride, self.dof_stride, self.action_dim)
        wp.launch(
            g1_reset_done_worlds_kernel,
            dim=(self.world_count, max_cols),
            inputs=[
                int(self._reset_seed),
                self.config.reset_noise,
                self.dones,
                self.model.joint_q,
                self.model.joint_qd,
                self.coord_stride,
                self.dof_stride,
                self.action_dim,
            ],
            outputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_steps,
                self.previous_actions,
                self.current_actions,
            ],
            device=self.device,
        )
        self._reset_seed += 1
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Apply actions, advance PhoenX, and return ``(obs, rewards, dones)``."""

        wp.launch(
            g1_apply_actions_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[
                actions,
                self.default_joint_pos,
                self.ctrl_lower,
                self.ctrl_upper,
                self.config.action_scale,
                int(self.config.controlled_action_count),
                self.dof_stride,
                self.coord_stride,
                int(bool(self.model.use_coord_layout_targets)),
            ],
            outputs=[self.current_actions, self.control.joint_target_q],
            device=self.device,
        )

        sub_dt = float(self.config.frame_dt) / float(self.config.sim_substeps)
        for _ in range(int(self.config.sim_substeps)):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            g1_increment_episode_steps_kernel,
            dim=self.world_count,
            outputs=[self.episode_steps],
            device=self.device,
        )
        self.observe()
        wp.copy(self.step_rewards, self.rewards)
        wp.copy(self.step_dones, self.dones)
        wp.copy(self.step_successes, self.successes)
        wp.copy(self.previous_actions, self.current_actions)
        if self.config.auto_reset:
            self.reset_done()
            self.observe()
        self.sim_time += float(self.config.frame_dt)
        return self.obs, self.step_rewards, self.step_dones

    def collect_ppo_rollout(self, trainer: TrainerPPO, buffer: BufferRollout, *, seed: int) -> None:
        """Collect one rollout and compute GAE returns for PPO."""

        collect_ppo_rollout(self, trainer, buffer, seed=seed)
