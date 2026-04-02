# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Cartpole Train
#
# Trains an inverted pendulum (cart-pole) balancing policy from scratch.
# The cart slides on a rail and must keep a single unactuated pole upright.
# Built from primitives -- no external asset files needed.
#
# Command: python -m newton.examples robot_cartpole_train
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples
from newton._src.ppo import ActorCritic, PPOTrainer, _increment_counter_kernel
from newton._src.robot_env import RobotEnv
from newton._src.warp_nn import export_to_onnx

_Q_STRIDE = 2  # cart_pos, pole_angle
_QD_STRIDE = 2  # cart_vel, pole_angular_vel
_MAX_EPISODE_LENGTH = 500


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_obs_kernel(joint_q: wp.array[float], joint_qd: wp.array[float], obs: wp.array2d[float]):
    """Obs = [cart_pos, sin(pole_angle), cos(pole_angle), cart_vel, pole_vel]."""
    env = wp.tid()
    q = env * _Q_STRIDE
    qd = env * _QD_STRIDE
    obs[env, 0] = joint_q[q]
    obs[env, 1] = wp.sin(joint_q[q + 1])
    obs[env, 2] = wp.cos(joint_q[q + 1])
    obs[env, 3] = joint_qd[qd]
    obs[env, 4] = joint_qd[qd + 1]


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
    pole_angle = joint_q[q + 1]
    cart_vel = joint_qd[qd]
    pole_vel = joint_qd[qd + 1]

    # Swing-up + balance reward (standard formulation).
    # Upright bonus: cos(angle) ranges from -1 (hanging) to +1 (upright).
    # Velocity penalty scales with how upright the pole is -- near the top
    # angular velocity must be low (balance), but during swing-up it's OK.
    upright = wp.cos(pole_angle)
    # upright_weight: 0 when hanging, 1 when upright
    upright_weight = wp.max(upright, 0.0)
    vel_penalty = upright_weight * 0.1 * pole_vel * pole_vel
    rewards[env] = upright - vel_penalty - 0.5 * cart_pos * cart_pos

    # Only terminate if cart escapes or episode timeout
    terminated = float(0.0)
    if wp.abs(cart_pos) > 2.0:
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
        q = env * _Q_STRIDE
        qd = env * _QD_STRIDE
        seed = rng_counter[0]
        rng = wp.rand_init(seed, env * 7)
        # Small random perturbation on all joints
        for i in range(wp.static(_Q_STRIDE)):
            joint_q[q + i] = initial_q[i] + (wp.randf(rng) - 0.5) * 0.1
        for i in range(wp.static(_QD_STRIDE)):
            joint_qd[qd + i] = (wp.randf(rng) - 0.5) * 0.2
        episode_lengths[env] = 0


@wp.kernel
def _apply_perturbation_kernel(
    joint_qd: wp.array[float],
    rng_counter: wp.array[int],
    kick_prob: float,
    kick_strength: float,
):
    """Random velocity kicks to a subset of envs each step."""
    env = wp.tid()
    seed = rng_counter[0]
    rng = wp.rand_init(seed, env * 7 + 12345)
    if wp.randf(rng) < kick_prob:
        qd = env * _QD_STRIDE
        joint_qd[qd + 1] = joint_qd[qd + 1] + (wp.randf(rng) - 0.5) * kick_strength


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CartpoleEnv(RobotEnv):
    obs_dim = 5  # cart_pos, sin(angle), cos(angle), cart_vel, pole_vel
    act_dim = 1
    sim_substeps = 4
    sim_dt = 1.0 / 60.0 / 4
    max_episode_length = _MAX_EPISODE_LENGTH
    use_collisions = False

    def build_robot(self, builder):
        # Cart: slides along Y axis
        cart = builder.add_link()
        builder.add_shape_box(cart, hx=0.3, hy=0.15, hz=0.1)
        j0 = builder.add_joint_prismatic(
            parent=-1,
            child=cart,
            axis=(0.0, 1.0, 0.0),
            target_ke=500.0,
            target_kd=50.0,
        )

        # Pole: hinges on the cart, unactuated
        pole = builder.add_link()
        builder.add_shape_box(pole, hx=0.04, hy=0.04, hz=0.5)
        j1 = builder.add_joint_revolute(
            parent=cart,
            child=pole,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=(0.0, 0.0, 0.1), q=wp.quat_identity()),
            child_xform=wp.transform(p=(0.0, 0.0, -0.5), q=wp.quat_identity()),
            target_ke=0.0,
            target_kd=0.05,
        )
        builder.add_articulation([j0, j1])

        # Start with pole hanging down (angle = pi)
        import math  # noqa: PLC0415

        builder.joint_q[1] = math.pi

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
        # Random perturbation kicks to ~10% of envs
        wp.launch(_increment_counter_kernel, dim=1, inputs=[self.rng_counter], device=self.device)
        wp.launch(
            _apply_perturbation_kernel,
            dim=self.num_envs,
            inputs=[self.state.joint_qd, self.rng_counter, 0.1, 2.0],
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

        num_envs = getattr(args, "num_envs", 2048)
        total_timesteps = getattr(args, "total_timesteps", 5_000_000)
        self.onnx_path = getattr(args, "onnx_output", "cartpole_trained.onnx")

        if self.is_test:
            total_timesteps = min(total_timesteps, num_envs * 64 * 3)

        self.env = CartpoleEnv(num_envs, device=str(self.device))

        self.ac = ActorCritic(
            obs_dim=5,
            act_dim=1,
            hidden_sizes=[64, 64],
            activation="elu",
            init_log_std=-0.5,
            bounded_actions=True,
            device=str(self.device),
            seed=42,
        )
        num_steps = 64
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
            entropy_coef=0.005,
            auto_entropy=False,
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

            if self.update_idx % 5 == 0:
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
                    self.ac.actor,
                    obs_dim=5,
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
        parser.add_argument("--num-envs", type=int, default=2048)
        parser.add_argument("--total-timesteps", type=int, default=5_000_000)
        parser.add_argument("--onnx-output", type=str, default="cartpole_trained.onnx")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
