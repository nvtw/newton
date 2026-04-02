# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Cartpole Train
#
# Trains a double-pendulum cartpole balancing policy from scratch using
# the built-in PPO trainer and Warp kernels.  No PyTorch dependency.
# This is a fast-converging validation scene for the PPO implementation.
#
# Command: python -m newton.examples robot_cartpole_train
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.ppo import ActorCritic, PPOTrainer
from newton._src.robot_env import RobotEnv
from newton._src.training_monitor import TrainingMonitor
from newton._src.warp_nn import export_to_onnx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_Q_STRIDE = 3
_QD_STRIDE = 3
_MAX_EPISODE_LENGTH = 500


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_obs_kernel(joint_q: wp.array[float], joint_qd: wp.array[float], obs: wp.array2d[float]):
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE
    for j in range(3):
        obs[env, j] = joint_q[q_off + j]
        obs[env, 3 + j] = joint_qd[qd_off + j]


@wp.kernel
def _compute_rewards_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    episode_lengths: wp.array[int],
    rewards: wp.array[float],
    dones: wp.array[float],
):
    env = wp.tid()
    q_off = env * _Q_STRIDE
    qd_off = env * _QD_STRIDE

    cart_pos = joint_q[q_off + 0]
    pole1_angle = joint_q[q_off + 1]
    pole2_angle = joint_q[q_off + 2]
    cart_vel = joint_qd[qd_off + 0]
    pole1_vel = joint_qd[qd_off + 1]
    pole2_vel = joint_qd[qd_off + 2]

    upright1 = wp.cos(pole1_angle)
    upright2 = wp.cos(pole1_angle + pole2_angle)
    rewards[env] = (
        upright1
        + upright2
        - 0.01 * cart_pos * cart_pos
        - 0.001 * (cart_vel * cart_vel + pole1_vel * pole1_vel + pole2_vel * pole2_vel)
    )

    terminated = float(0.0)
    if wp.abs(cart_pos) > 3.0:
        terminated = 1.0
    if upright1 < -0.2:
        terminated = 1.0
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0
    dones[env] = terminated


@wp.kernel
def _apply_actions_kernel(actions: wp.array2d[float], joint_target_pos: wp.array[float]):
    env = wp.tid()
    joint_target_pos[env * _QD_STRIDE] = actions[env, 0] * 2.0


@wp.kernel
def _reset_envs_kernel(
    dones: wp.array[float],
    initial_q: wp.array[float],
    initial_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    episode_lengths: wp.array[int],
    rng_counter: wp.array[int],
):
    env = wp.tid()
    if dones[env] > 0.5:
        q_off = env * _Q_STRIDE
        qd_off = env * _QD_STRIDE
        seed = rng_counter[0]
        rng = wp.rand_init(seed, env * 10)
        for i in range(wp.static(_Q_STRIDE)):
            joint_q[q_off + i] = initial_q[i] + (wp.randf(rng) - 0.5) * 0.1
        for i in range(wp.static(_QD_STRIDE)):
            joint_qd[qd_off + i] = initial_qd[i]
        episode_lengths[env] = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CartpoleEnv(RobotEnv):
    obs_dim = 6
    act_dim = 1
    sim_substeps = 10
    sim_dt = 1.0 / 60.0 / 10
    max_episode_length = _MAX_EPISODE_LENGTH
    use_collisions = False

    def build_robot(self, builder):
        builder.default_shape_cfg.density = 100.0
        builder.default_joint_cfg.armature = 0.1
        builder.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        body_armature = 0.1
        for body in range(builder.body_count):
            inertia_np = np.asarray(builder.body_inertia[body], dtype=np.float32).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * body_armature
            builder.body_inertia[body] = wp.mat33(inertia_np)
        builder.joint_q[-3:] = [0.0, 0.3, 0.0]
        builder.joint_target_ke[0] = 1000.0
        builder.joint_target_kd[0] = 100.0

    def compute_obs(self):
        wp.launch(
            _compute_obs_kernel,
            dim=self.num_envs,
            inputs=[self.state.joint_q, self.state.joint_qd, self.obs],
            device=self.device,
        )

    def compute_reward(self):
        wp.launch(
            _compute_rewards_kernel,
            dim=self.num_envs,
            inputs=[self.state.joint_q, self.state.joint_qd, self.episode_lengths, self.rewards, self.dones],
            device=self.device,
        )

    def apply_actions(self, actions):
        wp.launch(
            _apply_actions_kernel,
            dim=self.num_envs,
            inputs=[actions, self.control.joint_target_pos],
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
                self.rng_counter,
            ],
            device=self.device,
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
        total_timesteps = getattr(args, "total_timesteps", 500_000)
        self.onnx_path = getattr(args, "onnx_output", "cartpole_trained.onnx")

        if self.is_test:
            total_timesteps = min(total_timesteps, num_envs * 24 * 3)

        self.env = CartpoleEnv(num_envs, device=str(self.device))

        self.ac = ActorCritic(
            obs_dim=6,
            act_dim=1,
            hidden_sizes=[64, 64],
            activation="elu",
            init_log_std=-1.0,
            device=str(self.device),
            seed=42,
        )
        self.trainer = PPOTrainer(
            self.ac,
            num_envs,
            lr=3e-4,
            num_steps=32,
            num_epochs=5,
            num_minibatches=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=1.0,
        )

        self.steps_per_update = num_envs * 32
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

            if self.update_idx % 5 == 0:
                steps = self.update_idx * self.steps_per_update
                print(
                    f"Update {self.update_idx}/{self.total_updates} | steps={steps}"
                    f" | loss={stats['loss']:.4f} | mean_reward={stats['mean_reward']:.4f}"
                )

            if self.update_idx >= self.total_updates:
                self.training = False
                export_to_onnx(self.ac.actor, obs_dim=6, path=self.onnx_path)
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
        parser.add_argument("--total-timesteps", type=int, default=500_000)
        parser.add_argument("--onnx-output", type=str, default="cartpole_trained.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
