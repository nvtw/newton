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
from newton._src.ppo import ActorCritic, PPOTrainer
from newton._src.robot_env import RobotEnv
from newton._src.training_monitor import TrainingMonitor
from newton._src.warp_nn import export_to_onnx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBS_DIM = 48
_ACT_DIM = 12
_Q_STRIDE = 19  # 7 (free joint) + 12 (revolute)
_QD_STRIDE = 18  # 6 (free joint) + 12 (revolute)
_MAX_EPISODE_LENGTH = 1000

_LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
_MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

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
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE

    qx = joint_q[q_off + 3]
    qy = joint_q[q_off + 4]
    qz = joint_q[q_off + 5]
    qw = joint_q[q_off + 6]

    vx = joint_qd[qd_off + 0]
    vy = joint_qd[qd_off + 1]
    vz = joint_qd[qd_off + 2]
    wx = joint_qd[qd_off + 3]
    wy = joint_qd[qd_off + 4]
    wz = joint_qd[qd_off + 5]

    q2w2m1 = 2.0 * qw * qw - 1.0

    cross_vx = qy * vz - qz * vy
    cross_vy = qz * vx - qx * vz
    cross_vz = qx * vy - qy * vx
    dot_qv = qx * vx + qy * vy + qz * vz
    obs[env, 0] = vx * q2w2m1 - 2.0 * qw * cross_vx + 2.0 * qx * dot_qv
    obs[env, 1] = vy * q2w2m1 - 2.0 * qw * cross_vy + 2.0 * qy * dot_qv
    obs[env, 2] = vz * q2w2m1 - 2.0 * qw * cross_vz + 2.0 * qz * dot_qv

    cross_wx = qy * wz - qz * wy
    cross_wy = qz * wx - qx * wz
    cross_wz = qx * wy - qy * wx
    dot_qw = qx * wx + qy * wy + qz * wz
    obs[env, 3] = wx * q2w2m1 - 2.0 * qw * cross_wx + 2.0 * qx * dot_qw
    obs[env, 4] = wy * q2w2m1 - 2.0 * qw * cross_wy + 2.0 * qy * dot_qw
    obs[env, 5] = wz * q2w2m1 - 2.0 * qw * cross_wz + 2.0 * qz * dot_qw

    gz = -1.0
    cross_gx = qy * gz
    cross_gy = -qx * gz
    dot_qg = qz * gz
    obs[env, 6] = -2.0 * qw * cross_gx + 2.0 * qx * dot_qg
    obs[env, 7] = gz * q2w2m1 - 2.0 * qw * cross_gy + 2.0 * qy * dot_qg
    obs[env, 8] = gz * q2w2m1 - 2.0 * qw * (qx * gz) + 2.0 * qz * dot_qg

    obs[env, 9] = commands[env, 0]
    obs[env, 10] = commands[env, 1]
    obs[env, 11] = commands[env, 2]

    for j in range(12):
        mj = lab_to_mujoco[j]
        obs[env, 12 + mj] = joint_q[q_off + 7 + j] - joint_pos_initial[j]
        obs[env, 24 + mj] = joint_qd[qd_off + 6 + j]

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
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE

    height = joint_q[q_off + 2]
    qx = joint_q[q_off + 3]
    qy = joint_q[q_off + 4]
    qz = joint_q[q_off + 5]
    qw = joint_q[q_off + 6]
    vx = joint_qd[qd_off + 0]
    vy = joint_qd[qd_off + 1]
    wx = joint_qd[qd_off + 3]
    wy = joint_qd[qd_off + 4]
    wz = joint_qd[qd_off + 5]

    q2w2m1 = 2.0 * qw * qw - 1.0
    cross_vx = -qz * vy
    cross_vy = qz * vx
    dot_qv = qx * vx + qy * vy
    vel_bx = vx * q2w2m1 - 2.0 * qw * cross_vx + 2.0 * qx * dot_qv
    vel_by = vy * q2w2m1 - 2.0 * qw * cross_vy + 2.0 * qy * dot_qv

    cross_wz = qx * wy - qy * wx
    dot_qw_z = qx * wx + qy * wy + qz * wz
    avel_bz = wz * q2w2m1 - 2.0 * qw * cross_wz + 2.0 * qz * dot_qw_z

    gz = -1.0
    grav_bx = -2.0 * qw * (qy * gz) + 2.0 * qx * (qz * gz)
    grav_by = gz * q2w2m1 - 2.0 * qw * (-qx * gz) + 2.0 * qy * (qz * gz)

    cmd_fwd = commands[env, 0]
    cmd_lat = commands[env, 1]
    cmd_yaw = commands[env, 2]

    fwd_reward = wp.exp(-4.0 * (vel_bx - cmd_fwd) * (vel_bx - cmd_fwd))
    lat_reward = wp.exp(-4.0 * (vel_by - cmd_lat) * (vel_by - cmd_lat)) * 0.5
    yaw_reward = wp.exp(-4.0 * (avel_bz - cmd_yaw) * (avel_bz - cmd_yaw)) * 0.5
    orientation_penalty = -5.0 * (grav_bx * grav_bx + grav_by * grav_by)

    action_rate = float(0.0)
    for j in range(12):
        diff = last_actions[env, j] - prev_actions[env, j]
        action_rate = action_rate + diff * diff

    rewards[env] = fwd_reward + lat_reward + yaw_reward + 1.0 + orientation_penalty - 0.01 * action_rate

    terminated = float(0.0)
    if height < 0.25:
        terminated = 1.0
    if grav_bx * grav_bx + grav_by * grav_by > 0.49:
        terminated = 1.0
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0
    dones[env] = terminated


@wp.kernel
def _apply_actions_kernel(
    actions: wp.array2d[float],
    mujoco_to_lab: wp.array[int],
    joint_pos_initial: wp.array[float],
    joint_target_pos: wp.array[float],
):
    env, j = wp.tid()
    lab_j = mujoco_to_lab[j]
    joint_target_pos[env * _QD_STRIDE + 6 + lab_j] = joint_pos_initial[lab_j] + 0.5 * actions[env, j]


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


@wp.kernel
def _randomize_commands_kernel(dones: wp.array[float], commands: wp.array2d[float], rng_counter: wp.array[int]):
    env = wp.tid()
    if dones[env] > 0.5:
        step_offset = rng_counter[0]
        rng = wp.rand_init(42, step_offset * 1000 + env)
        commands[env, 0] = wp.randf(rng) * 2.0 - 0.5
        commands[env, 1] = wp.randf(rng) * 1.0 - 0.5
        commands[env, 2] = wp.randf(rng) * 1.0 - 0.5


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
        self.commands = wp.zeros((num_envs, 3), dtype=wp.float32, device=d)
        self._lab_to_mujoco = wp.array(_LAB_TO_MUJOCO, dtype=wp.int32, device=d)
        self._mujoco_to_lab = wp.array(_MUJOCO_TO_LAB, dtype=wp.int32, device=d)
        full_q = self.initial_joint_q.numpy()
        self._joint_pos_initial = wp.array(full_q[7:_Q_STRIDE].astype(np.float32), dtype=wp.float32, device=d)

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
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

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
                self.last_actions,
                self.prev_actions,
                self.commands,
                self.episode_lengths,
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
            ],
            device=self.device,
        )
        wp.launch(
            _randomize_commands_kernel,
            dim=self.num_envs,
            inputs=[self.dones, self.commands, self.rng_counter],
            device=self.device,
        )

    def on_reset(self):
        self.last_actions.zero_()
        self.prev_actions.zero_()
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

        num_envs = getattr(args, "num_envs", 1024)
        total_timesteps = getattr(args, "total_timesteps", 1_000_000)
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
        self.monitor = TrainingMonitor(self.viewer)

    def step(self):
        if self.training and self.update_idx < self.total_updates:
            last_values, self.obs = self.trainer.collect_rollouts(self.env, self.obs)
            self.trainer.buffer.compute_gae(last_values, self.trainer.gamma, self.trainer.gae_lambda)
            self.trainer.update()
            self.update_idx += 1

            stats = self.trainer.get_stats()
            self.monitor.log(stats)

            if self.update_idx % 10 == 0:
                steps = self.update_idx * self.steps_per_update
                print(
                    f"Update {self.update_idx}/{self.total_updates} | steps={steps}"
                    f" | loss={stats['loss']:.4f} | mean_reward={stats['mean_reward']:.4f}"
                )

            if self.update_idx >= self.total_updates:
                self.training = False
                export_to_onnx(self.ac.actor, obs_dim=_OBS_DIM, path=self.onnx_path)
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
        parser.add_argument("--num-envs", type=int, default=1024)
        parser.add_argument("--total-timesteps", type=int, default=1_000_000)
        parser.add_argument("--onnx-output", type=str, default="anymal_c_trained.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
