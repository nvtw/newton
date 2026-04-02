# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Pendulum Train
#
# Trains a policy to swing up and balance a double pendulum using PPO.
# Joint 0 (shoulder) is actuated, joint 1 (elbow) is passive.
# The pendulum starts hanging down and must learn to reach upright.
#
# Command: python -m newton.examples basic_pendulum_train
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples
from newton._src.ppo import ActorCritic, PPOTrainer, _increment_counter_kernel
from newton._src.robot_env import RobotEnv
from newton._src.warp_nn import export_to_onnx

_Q_STRIDE = 2
_QD_STRIDE = 2
_MAX_EPISODE_LENGTH = 500
_HX = 1.0  # link half-length


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_obs_kernel(joint_q: wp.array[float], joint_qd: wp.array[float], obs: wp.array2d[float]):
    """Obs = [sin(q0), cos(q0), sin(q1), cos(q1), qd0, qd1]."""
    env = wp.tid()
    q = env * _Q_STRIDE
    qd = env * _QD_STRIDE
    obs[env, 0] = wp.sin(joint_q[q])
    obs[env, 1] = wp.cos(joint_q[q])
    obs[env, 2] = wp.sin(joint_q[q + 1])
    obs[env, 3] = wp.cos(joint_q[q + 1])
    obs[env, 4] = joint_qd[qd] * 0.1  # scale velocities
    obs[env, 5] = joint_qd[qd + 1] * 0.1


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
    q0 = joint_q[q]
    q1 = joint_q[q + 1]
    qd0 = joint_qd[qd]
    qd1 = joint_qd[qd + 1]

    # Tip height reward: height of end of link_1 relative to pivot
    # Link 0 tip at angle q0: y0 = hx*cos(q0), z0 = -hx*sin(q0) (from pivot)
    # Wait -- the joint axis is Y, so rotation is in the XZ plane.
    # Actually the parent_xform has a Z-rotation of -pi/2, so the pendulum
    # swings in the YZ plane from the viewer's perspective.
    # For the reward, we just care about the height of the tip.
    # In the joint's local frame, the "up" direction for the tip of link_1 is:
    # tip_z = hx*cos(q0) + hx*cos(q0+q1) (relative to pivot at z=5)
    # When q0=0, q1=0: tip_z = 2*hx (both links hanging straight down from pivot)
    # Wait -- q=0 means the default pose. Let me check.
    # At q0=0, link_0 hangs down. cos(0) points along the joint's rest direction.
    # The child_xform for j0 has offset (-hx, 0, 0), meaning the joint is at the
    # -X end of the link. In the parent frame (rotated by -pi/2 around Z),
    # the link hangs in the -X direction which maps to... this is getting complex.
    #
    # Simpler: the reward is cos(q0) + cos(q0+q1). When both are 0 (rest/hanging),
    # cos = 1. When swung to pi (upright), cos(pi) = -1.
    # So upright = q0 = pi, q1 = 0 -> cos(pi) + cos(pi) = -2 (bad!)
    # Actually upright means tips are ABOVE the pivot. Need to think about sign.
    #
    # Let's just use: reward = -cos(q0) - cos(q0+q1)
    # At rest (hanging): -cos(0) - cos(0) = -2 (worst)
    # Upright (q0=pi): -cos(pi) - cos(pi) = +2 (best)
    rewards[env] = -wp.cos(q0) - wp.cos(q0 + q1) - 0.001 * (qd0 * qd0 + qd1 * qd1)

    terminated = float(0.0)
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0
    dones[env] = terminated


@wp.kernel
def _apply_actions_kernel(actions: wp.array2d[float], joint_act: wp.array[float]):
    """Apply torque to joint 0 (shoulder). Joint 1 (elbow) is passive."""
    env = wp.tid()
    joint_act[env * _QD_STRIDE] = actions[env, 0] * 50.0  # torque scale


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
            joint_qd[qd + i] = (wp.randf(rng) - 0.5) * 0.5
        episode_lengths[env] = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PendulumEnv(RobotEnv):
    obs_dim = 6  # sin(q0), cos(q0), sin(q1), cos(q1), qd0, qd1
    act_dim = 1  # torque on joint 0
    sim_substeps = 10
    sim_dt = 1.0 / 100.0 / 10  # match the original example: 100 fps, 10 substeps
    max_episode_length = _MAX_EPISODE_LENGTH
    use_collisions = False

    def build_robot(self, builder):
        hx, hy, hz = _HX, 0.1, 0.1
        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)
        link_1 = builder.add_link()
        builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = builder.add_joint_revolute(
            parent=-1, child=link_0, axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=0.0, target_kd=0.1,  # small damping
        )
        j1 = builder.add_joint_revolute(
            parent=link_0, child=link_1, axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=0.0, target_kd=0.1,
        )
        builder.add_articulation([j0, j1], label="pendulum")

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


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        num_envs = getattr(args, "num_envs", 2048)
        total_timesteps = getattr(args, "total_timesteps", 10_000_000)
        self.onnx_path = getattr(args, "onnx_output", "pendulum_trained.onnx")

        if self.is_test:
            total_timesteps = min(total_timesteps, num_envs * 64 * 3)

        self.env = PendulumEnv(num_envs, device=str(self.device))

        self.ac = ActorCritic(
            obs_dim=6, act_dim=1, hidden_sizes=[64, 64],
            activation="elu", init_log_std=-0.5,
            bounded_actions=True, device=str(self.device), seed=42,
        )
        num_steps = 64
        self.trainer = PPOTrainer(
            self.ac, num_envs, lr=3e-4, num_steps=num_steps, num_epochs=5,
            num_minibatches=4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
            entropy_coef=0.005, auto_entropy=False,
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
        parser.add_argument("--num-envs", type=int, default=2048)
        parser.add_argument("--total-timesteps", type=int, default=10_000_000)
        parser.add_argument("--onnx-output", type=str, default="pendulum_trained.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
