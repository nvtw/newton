# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Train (PufferLib PPO)
#
# Trains an ANYmal C walking policy from scratch using PufferLib's PPO
# trainer with Muon optimizer and CUDA graph capture.  No PyTorch
# dependency.
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

import warp as wp

import newton
import newton.examples
from newton._src.pufferlib.envs.anymal import AnymalEnv
from newton._src.pufferlib.onnx_export import export_policy_to_onnx, save_checkpoint
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.is_test = args is not None and args.test

        num_envs = getattr(args, "num_envs", 4096)
        total_timesteps = getattr(args, "total_timesteps", 100_000_000)
        self.onnx_path = getattr(args, "onnx_output", "anymal_c_trained.onnx")
        use_mujoco_contacts = getattr(args, "mujoco_contacts", True)

        if self.is_test:
            num_envs = min(num_envs, 64)
            total_timesteps = min(total_timesteps, num_envs * 24 * 3)

        device_str = str(self.device)

        best_so_far = [-1000.0]
        onnx_path = self.onnx_path

        def checkpoint_fn(policy, iteration, total_steps, best_return, obs_normalizer):
            if best_return > best_so_far[0]:
                best_so_far[0] = best_return
                save_checkpoint(policy, "anymal_checkpoint.npz")
                obs_mean, obs_var = None, None
                if obs_normalizer is not None:
                    obs_mean, obs_var = obs_normalizer.get_mean_var()
                export_policy_to_onnx(policy, obs_dim=48, num_actions=12, path=onnx_path,
                                      obs_mean=obs_mean, obs_var=obs_var)

        config = PPOConfig(
            num_envs=num_envs,
            horizon=24,
            obs_dim=48,
            num_actions=12,
            hidden=128,
            total_timesteps=total_timesteps,
            seed=42,
            optimizer="adamw",
            lr=1e-3,
            anneal_lr=True,
            min_lr_ratio=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            vf_coef=0.5,
            vf_clip_coef=0.2,
            ent_coef=0.01,
            rho_clip=1.0,
            c_clip=1.0,
            max_grad_norm=1.0,
            momentum=0.9,
            replay_ratio=5.0,
            minibatch_size=max(min(32768, num_envs * 24) // 4, 1024),
            continuous=True,
            init_logstd=-1.0,
            normalize_obs=True,
            reward_clamp=10.0,
            env_name="ANYmal C Walk",
            best_return_init=-1000.0,
            return_format="8.1f",
            step_width=11,
            log_interval=10,
            checkpoint_fn=checkpoint_fn,
            device=device_str,
        )

        def make_env(device):
            return AnymalEnv(
                num_envs=config.num_envs,
                device=device,
                seed=config.seed,
                use_mujoco_contacts=use_mujoco_contacts,
            )

        # Set viewer model from a temporary env for visualization
        temp_env = AnymalEnv(num_envs=1, device=device_str, seed=config.seed)
        self.viewer.set_model(temp_env.model)
        self._single_env = temp_env

        # Run training (blocks until complete)
        trainer = PPOTrainer(config, make_env)
        trainer.train()
        self._trained = True

    def step(self):
        pass

    def render(self):
        self.viewer.begin_frame(0.0)
        self.viewer.end_frame()

    def test_final(self):
        assert self._trained, "Training did not complete"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--num-envs", type=int, default=4096)
        parser.add_argument("--total-timesteps", type=int, default=100_000_000)
        parser.add_argument("--onnx-output", type=str, default="anymal_c_trained.onnx")
        parser.add_argument("--mujoco-contacts", action="store_true", default=True)
        parser.add_argument("--newton-contacts", dest="mujoco_contacts", action="store_false")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
