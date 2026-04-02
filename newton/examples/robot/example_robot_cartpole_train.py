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
from newton._src.ppo import ActorCritic, PPOTrainer, _increment_counter_kernel, export_actor_to_onnx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBS_DIM = 6  # joint_q (3) + joint_qd (3)
_ACT_DIM = 1  # force on cart prismatic joint
_Q_STRIDE = 3  # 3 joint DOFs (prismatic + 2 revolute)
_QD_STRIDE = 3
_NUM_BODIES = 3
_SIM_SUBSTEPS = 10
_FRAME_DT = 1.0 / 60.0
_SIM_DT = _FRAME_DT / _SIM_SUBSTEPS
_MAX_EPISODE_LENGTH = 500  # ~8.3 seconds at 60 Hz


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_obs_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    obs: wp.array2d[float],
):
    """Observation = [joint_q(3), joint_qd(3)]."""
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
    """Reward: upright bonus - velocity penalty - cart position penalty."""
    env = wp.tid()
    q_off = env * _Q_STRIDE

    cart_pos = joint_q[q_off + 0]
    pole1_angle = joint_q[q_off + 1]
    pole2_angle = joint_q[q_off + 2]

    qd_off = env * _QD_STRIDE
    cart_vel = joint_qd[qd_off + 0]
    pole1_vel = joint_qd[qd_off + 1]
    pole2_vel = joint_qd[qd_off + 2]

    # Reward: poles upright (cos(angle) close to 1)
    upright1 = wp.cos(pole1_angle)
    upright2 = wp.cos(pole1_angle + pole2_angle)
    upright_reward = upright1 + upright2  # max = 2.0 when both vertical

    # Penalties
    cart_penalty = -0.01 * cart_pos * cart_pos
    vel_penalty = -0.001 * (cart_vel * cart_vel + pole1_vel * pole1_vel + pole2_vel * pole2_vel)

    rewards[env] = upright_reward + cart_penalty + vel_penalty

    # Termination
    terminated = float(0.0)
    if wp.abs(cart_pos) > 3.0:
        terminated = 1.0
    if upright1 < -0.2:  # pole1 more than ~100 deg from vertical
        terminated = 1.0
    if episode_lengths[env] >= _MAX_EPISODE_LENGTH:
        terminated = 1.0

    dones[env] = terminated


@wp.kernel
def _apply_actions_kernel(
    actions: wp.array2d[float],
    joint_target_pos: wp.array[float],
):
    """Apply cart force via joint target position (PD control on prismatic joint)."""
    env = wp.tid()
    # Scale action to reasonable cart position target
    joint_target_pos[env * _QD_STRIDE] = actions[env, 0] * 2.0


@wp.kernel
def _reset_envs_kernel(
    dones: wp.array[float],
    initial_joint_q: wp.array[float],
    initial_joint_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    episode_lengths: wp.array[int],
    rng_counter: wp.array[int],
):
    """Reset terminated environments with small random perturbation."""
    env = wp.tid()
    if dones[env] > 0.5:
        q_off = env * _Q_STRIDE
        qd_off = env * _QD_STRIDE
        seed = rng_counter[0]
        rng = wp.rand_init(seed, env * 10)
        for i in range(wp.static(_Q_STRIDE)):
            noise = (wp.randf(rng) - 0.5) * 0.1
            joint_q[q_off + i] = initial_joint_q[i] + noise
        for i in range(wp.static(_QD_STRIDE)):
            joint_qd[qd_off + i] = initial_joint_qd[i]
        episode_lengths[env] = 0


@wp.kernel
def _increment_episode_kernel(episode_lengths: wp.array[int]):
    env = wp.tid()
    episode_lengths[env] = episode_lengths[env] + 1


# ---------------------------------------------------------------------------
# Vectorized environment
# ---------------------------------------------------------------------------


class CartpoleVecEnv:
    """Vectorized double-pendulum cartpole for PPO training."""

    num_envs: int
    obs_dim: int = _OBS_DIM
    act_dim: int = _ACT_DIM

    def __init__(self, num_envs: int, device: str | None = None, seed: int = 123):
        self.num_envs = num_envs
        self._device = wp.get_device(device)
        self.sim_time = 0.0

        # Build single cartpole
        cartpole = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1

        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # Inertia augmentation for stability
        body_armature = 0.1
        for body in range(cartpole.body_count):
            inertia_np = np.asarray(cartpole.body_inertia[body], dtype=np.float32).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * body_armature
            cartpole.body_inertia[body] = wp.mat33(inertia_np)

        # Initial state: poles slightly tilted
        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        # PD gains for cart prismatic joint
        cartpole.joint_target_ke[0] = 1000.0
        cartpole.joint_target_kd[0] = 100.0

        # Replicate
        builder = newton.ModelBuilder()
        builder.replicate(cartpole, num_envs)
        self.model = builder.finalize(device=self._device)

        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Store initial state for resets
        full_q = self.state_0.joint_q.numpy()
        full_qd = self.state_0.joint_qd.numpy()
        self._initial_joint_q = wp.array(full_q[:_Q_STRIDE].astype(np.float32), dtype=wp.float32, device=self._device)
        self._initial_joint_qd = wp.array(
            full_qd[:_QD_STRIDE].astype(np.float32), dtype=wp.float32, device=self._device
        )

        # Pre-allocated buffers
        d = self._device
        self.obs = wp.zeros((num_envs, _OBS_DIM), dtype=wp.float32, device=d)
        self.rewards = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.dones = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.episode_lengths = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self._rng_counter = wp.array([seed], dtype=wp.int32, device=d)

        # CUDA graph for physics
        self._graph = None
        if self._device.is_cuda:
            self.control.joint_target_pos = wp.zeros(num_envs * _QD_STRIDE, dtype=wp.float32, device=d)
            with wp.ScopedCapture() as capture:
                self._simulate()
            self._graph = capture.graph

    def _simulate(self):
        for _ in range(_SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, _SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self) -> wp.array:
        full_q = self._initial_joint_q.numpy()
        full_qd = self._initial_joint_qd.numpy()
        q_tiled = np.tile(full_q, self.num_envs).astype(np.float32)
        qd_tiled = np.tile(full_qd, self.num_envs).astype(np.float32)
        wp.copy(self.state_0.joint_q, wp.array(q_tiled, dtype=wp.float32, device=self._device))
        wp.copy(self.state_0.joint_qd, wp.array(qd_tiled, dtype=wp.float32, device=self._device))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.episode_lengths.zero_()
        self.dones.zero_()
        self.sim_time = 0.0

        wp.launch(
            _compute_obs_kernel,
            dim=self.num_envs,
            inputs=[self.state_0.joint_q, self.state_0.joint_qd, self.obs],
            device=self._device,
        )
        return self.obs

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        d = self._device
        ne = self.num_envs

        wp.launch(_apply_actions_kernel, dim=ne, inputs=[actions, self.control.joint_target_pos], device=d)

        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        wp.launch(_increment_episode_kernel, dim=ne, inputs=[self.episode_lengths], device=d)

        wp.launch(
            _compute_rewards_kernel,
            dim=ne,
            inputs=[self.state_0.joint_q, self.state_0.joint_qd, self.episode_lengths, self.rewards, self.dones],
            device=d,
        )

        wp.launch(_increment_counter_kernel, dim=1, inputs=[self._rng_counter], device=d)
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
                self._rng_counter,
            ],
            device=d,
        )

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        wp.launch(_compute_obs_kernel, dim=ne, inputs=[self.state_0.joint_q, self.state_0.joint_qd, self.obs], device=d)

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

        num_envs = getattr(args, "num_envs", 1024)
        total_timesteps = getattr(args, "total_timesteps", 500_000)
        self.onnx_path = getattr(args, "onnx_output", "cartpole_trained.onnx")

        if self.is_test:
            total_timesteps = min(total_timesteps, num_envs * 24 * 3)

        self.env = CartpoleVecEnv(num_envs, device=str(self.device))

        self.ac = ActorCritic(
            obs_dim=_OBS_DIM,
            act_dim=_ACT_DIM,
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
        self.obs: wp.array | None = None

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
                export_actor_to_onnx(self.ac.actor, obs_dim=_OBS_DIM, path=self.onnx_path)
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
        parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments.")
        parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total training timesteps.")
        parser.add_argument("--onnx-output", type=str, default="cartpole_trained.onnx", help="Output ONNX path.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
