# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Pendulum Train
#
# Trains a policy to balance the double-pendulum cartpole (from the
# robot_cartpole example) using PPO.  Only the cart is actuated.
#
# Command: python -m newton.examples basic_pendulum_train
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.ppo import ActorCritic, PPOTrainer, _increment_counter_kernel
from newton._src.robot_env import RobotEnv
from newton._src.warp_nn import export_to_onnx

_Q_STRIDE = 3  # cart_pos, pole1_angle, pole2_angle
_QD_STRIDE = 3
_MAX_EPISODE_LENGTH = 1000000  # no timeout -- continuous training


@wp.kernel
def _compute_obs_kernel(joint_q: wp.array[float], joint_qd: wp.array[float], obs: wp.array2d[float]):
    """Obs: [cart_pos, sin(p1), cos(p1), sin(p2), cos(p2), cart_vel, p1_vel, p2_vel]."""
    env = wp.tid()
    q = env * _Q_STRIDE
    qd = env * _QD_STRIDE
    obs[env, 0] = joint_q[q]              # cart position
    obs[env, 1] = wp.sin(joint_q[q + 1])  # pole1 sin
    obs[env, 2] = wp.cos(joint_q[q + 1])  # pole1 cos
    obs[env, 3] = wp.sin(joint_q[q + 2])  # pole2 sin
    obs[env, 4] = wp.cos(joint_q[q + 2])  # pole2 cos
    obs[env, 5] = joint_qd[qd] * 0.1      # cart vel (scaled)
    obs[env, 6] = joint_qd[qd + 1] * 0.1  # pole1 vel
    obs[env, 7] = joint_qd[qd + 2] * 0.1  # pole2 vel


@wp.kernel
def _compute_rewards_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    episode_lengths: wp.array[int],
    rewards: wp.array[float],
    dones: wp.array[float],
):
    env = wp.tid()
    q = env * _Q_STRIDE
    qd = env * _QD_STRIDE

    cart_pos = joint_q[q]
    p1 = joint_q[q + 1]
    p2 = joint_q[q + 2]
    p1_vel = joint_qd[qd + 1]
    p2_vel = joint_qd[qd + 2]

    # Height: cos(p1) + cos(p1+p2), max=2 when both upright
    upright1 = wp.cos(p1)
    upright2 = wp.cos(p1 + p2)
    height = upright1 + upright2

    # Velocity penalty scales with how upright -- free to swing below, must be still at top
    near_top = wp.max(height - 1.0, 0.0)  # 0 below halfway, 1 at top
    vel_penalty = near_top * 0.05 * (p1_vel * p1_vel + p2_vel * p2_vel)

    rewards[env] = height - vel_penalty - 0.1 * cart_pos * cart_pos

    # No early termination -- continuous reward shaping
    terminated = float(0.0)
    if wp.abs(cart_pos) > 3.0:
        terminated = 1.0
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0
    dones[env] = terminated


@wp.kernel
def _apply_actions_kernel(actions: wp.array2d[float], joint_act: wp.array[float]):
    """Apply force to cart only (joint 0). Poles are unactuated."""
    env = wp.tid()
    joint_act[env * _QD_STRIDE] = actions[env, 0] * 200.0


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
        q = env * _Q_STRIDE
        qd = env * _QD_STRIDE
        seed = rng_counter[0]
        rng = wp.rand_init(seed, env * 7)
        for i in range(wp.static(_Q_STRIDE)):
            joint_q[q + i] = initial_q[i] + (wp.randf(rng) - 0.5) * 0.2
        for i in range(wp.static(_QD_STRIDE)):
            joint_qd[qd + i] = (wp.randf(rng) - 0.5) * 0.3
        episode_lengths[env] = 0


class CartpoleSwingUpEnv(RobotEnv):
    obs_dim = 8
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
        # Start with poles hanging down
        builder.joint_q[-3:] = [0.0, 0.0, 0.0]
        # PD on cart for position control
        builder.joint_target_ke[0] = 500.0
        builder.joint_target_kd[0] = 50.0

    def compute_obs(self):
        wp.launch(_compute_obs_kernel, dim=self.num_envs,
                  inputs=[self.state.joint_q, self.state.joint_qd, self.obs], device=self.device)

    def compute_reward(self):
        wp.launch(_compute_rewards_kernel, dim=self.num_envs,
                  inputs=[self.state.joint_q, self.state.joint_qd,
                          self.episode_lengths, self.rewards, self.dones], device=self.device)

    def apply_actions(self, actions):
        wp.launch(_apply_actions_kernel, dim=self.num_envs,
                  inputs=[actions, self.control.joint_act], device=self.device)

    def reset_done_envs(self):
        wp.launch(_reset_envs_kernel, dim=self.num_envs,
                  inputs=[self.dones, self.initial_joint_q, self.initial_joint_qd,
                          self.state.joint_q, self.state.joint_qd,
                          self.episode_lengths, self.rng_counter], device=self.device)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        num_envs = getattr(args, "num_envs", 4096)
        total_timesteps = getattr(args, "total_timesteps", 50_000_000)
        self.onnx_path = getattr(args, "onnx_output", "pendulum_trained.onnx")

        if self.is_test:
            total_timesteps = min(total_timesteps, num_envs * 64 * 3)

        self.env = CartpoleSwingUpEnv(num_envs, device=str(self.device))

        self.ac = ActorCritic(
            obs_dim=8, act_dim=1, hidden_sizes=[128, 128],
            activation="elu", init_log_std=0.0,
            bounded_actions=True, device=str(self.device), seed=42,
        )
        num_steps = 128
        self.trainer = PPOTrainer(
            self.ac, num_envs, lr=3e-4, num_steps=num_steps, num_epochs=5,
            num_minibatches=4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
            entropy_coef=0.01, auto_entropy=False,
            value_coef=0.5, max_grad_norm=1.0,
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

            if self.update_idx % 10 == 0:
                stats = self.trainer.get_stats()
                steps = self.update_idx * self.steps_per_update
                print(
                    f"Update {self.update_idx}/{self.total_updates} | steps={steps}"
                    f" | loss={stats['loss']:.4f} | mean_reward={stats['mean_reward']:.4f}"
                )

            if self.update_idx >= self.total_updates:
                self.training = False
                norm = self.trainer.obs_normalizer
                export_to_onnx(
                    self.ac.actor, obs_dim=8, path=self.onnx_path,
                    obs_mean=norm.mean.numpy(), obs_inv_std=norm.inv_std.numpy(),
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
        assert self.update_idx > 0

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--num-envs", type=int, default=4096)
        parser.add_argument("--total-timesteps", type=int, default=50_000_000)
        parser.add_argument("--onnx-output", type=str, default="pendulum_trained.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
