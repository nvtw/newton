# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Anymal RL Playback (PhoenX)
#
# Replays a Warp-only PPO Anymal policy trained with PhoenX physics. The
# command interface is body-frame velocity tracking: I/K forward/backward,
# J/L left/right, U/O yaw-rate about the up axis, and N/M lower/raise
# the body-height command. No key means zero velocity and nominal height.
#
# Command: python -m newton.examples robot_anymal_rl_phoenx --checkpoint policy.npz
#
###########################################################################

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton.examples
import newton.rl as rl


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device(args.device)
        if not self.device.is_cuda:
            raise RuntimeError(
                "Anymal PhoenX RL playback requires CUDA; pass --device cuda:0 or set Warp's default CUDA device"
            )

        self.seed = int(args.seed)
        self.step_count = 0
        self.steps_per_frame = max(int(args.steps_per_frame), 1)
        self.deterministic = bool(args.deterministic)
        self.interactive_command = bool(args.interactive_command)
        self.forward_speed = float(args.forward_speed)
        self.backward_speed = float(args.backward_speed)
        self.lateral_speed = float(args.lateral_speed)
        self.yaw_rate = float(args.yaw_rate)
        self.height_offset = float(args.height_offset)
        self.fallback_command = (
            float(args.command_x),
            float(args.command_y),
            float(args.command_yaw),
            float(args.command_height),
        )
        self.forward_key = str(args.forward_key).lower()
        self.backward_key = str(args.backward_key).lower()
        self.left_key = str(args.left_key).lower()
        self.right_key = str(args.right_key).lower()
        self.yaw_left_key = str(args.yaw_left_key).lower()
        self.yaw_right_key = str(args.yaw_right_key).lower()
        self.height_down_key = str(args.height_down_key).lower()
        self.height_up_key = str(args.height_up_key).lower()
        self.render_contacts = bool(args.render_contacts)
        self.log_interval = max(int(args.log_interval), 0)
        self._last_command: tuple[float, float, float, float] | None = None
        self._last_stats: tuple[float, float, float] | None = None

        self.env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(
                world_count=1,
                frame_dt=float(args.frame_dt),
                sim_substeps=int(args.sim_substeps),
                solver_iterations=int(args.solver_iterations),
                velocity_iterations=int(args.velocity_iterations),
                action_scale=float(args.action_scale),
                reward_mode="dense_command",
                command=(0.0, 0.0, 0.0, 0.0),
                max_episode_steps=int(args.max_episode_steps),
                target_base_height=float(args.target_base_height),
                min_base_height=float(args.min_base_height),
                min_upright_cos=float(args.min_upright_cos),
                actuator_ke=float(args.actuator_ke),
                actuator_kd=float(args.actuator_kd),
                auto_reset=True,
            ),
            device=self.device,
        )
        self.trainer = rl.load_ppo_checkpoint(args.checkpoint, device=self.device)
        if self.trainer.obs_dim != self.env.obs_dim or self.trainer.action_dim != self.env.action_dim:
            raise ValueError("Checkpoint dimensions do not match the Anymal PhoenX environment")
        self.trainer.reset_rollout_state()
        self.env.set_command((0.0, 0.0, 0.0, 0.0))

        self.viewer.set_model(self.env.model)
        self.viewer.set_visible_worlds(range(1))
        self.viewer.set_camera(
            wp.vec3(float(args.camera_x), float(args.camera_y), float(args.camera_z)),
            float(args.camera_pitch),
            float(args.camera_yaw),
        )

    def _key_down(self, key: str) -> bool:
        return bool(hasattr(self.viewer, "is_key_down") and self.viewer.is_key_down(key))

    def _keyboard_command(self) -> tuple[float, float, float, float]:
        if not self.interactive_command:
            return self.fallback_command

        forward = 0.0
        lateral = 0.0
        yaw = 0.0
        height = 0.0
        used = False
        if self._key_down(self.forward_key):
            forward += self.forward_speed
            used = True
        if self._key_down(self.backward_key):
            forward -= self.backward_speed
            used = True
        if self._key_down(self.left_key):
            lateral += self.lateral_speed
            used = True
        if self._key_down(self.right_key):
            lateral -= self.lateral_speed
            used = True
        if self._key_down(self.yaw_left_key):
            yaw += self.yaw_rate
            used = True
        if self._key_down(self.yaw_right_key):
            yaw -= self.yaw_rate
            used = True
        if self._key_down(self.height_down_key):
            height -= self.height_offset
            used = True
        if self._key_down(self.height_up_key):
            height += self.height_offset
            used = True
        if used:
            return (forward, lateral, yaw, height)
        return (0.0, 0.0, 0.0, 0.0)

    def _apply_command(self, command: tuple[float, ...]) -> None:
        command = (float(command[0]), float(command[1]), float(command[2]), float(command[3]))
        if self._last_command == command:
            return
        self.env.set_command(command)
        self._last_command = command

    def step(self):
        for _ in range(self.steps_per_frame):
            self._apply_command(self._keyboard_command())
            actions, _log_probs, _values = self.trainer.act_reuse(
                self.env.obs,
                seed=self.seed + 1_000_003 + self.step_count,
                deterministic=self.deterministic,
            )
            _obs, rewards, dones = self.env.step(actions)
            self.trainer.reset_rollout_state(dones)
            self.step_count += 1
            if self.log_interval > 0 and self.step_count % self.log_interval == 0:
                reward = float(np.mean(rewards.numpy()))
                done = float(np.mean(dones.numpy()))
                perf = float(np.mean(self.env.step_successes.numpy()))
                self._last_stats = (reward, done, perf)
                cmd = self._last_command or (0.0, 0.0, 0.0, 0.0)
                print(
                    f"step={self.step_count:06d} cmd=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f},{cmd[3]:.2f}) "
                    f"reward={reward:.4f} perf={perf:.3f} done={done:.4f}"
                )

    def render(self):
        self.viewer.begin_frame(self.env.sim_time)
        self.viewer.log_state(self.env.state_0)
        if self.render_contacts:
            self.viewer.log_contacts(self.env.contacts, self.env.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.step_count <= 0:
            raise RuntimeError("Expected at least one Anymal policy playback step")
        obs = self.env.obs.numpy()
        if not np.isfinite(obs).all():
            raise RuntimeError("Anymal policy observations contain non-finite values")
        joint_q = self.env.state_0.joint_q.numpy()
        joint_qd = self.env.state_0.joint_qd.numpy()
        if not np.isfinite(joint_q).all() or not np.isfinite(joint_qd).all():
            raise RuntimeError("Anymal policy state contains non-finite values")
        if self._last_stats is not None and not np.isfinite(np.asarray(self._last_stats)).all():
            raise RuntimeError("Anymal playback diagnostics contain non-finite values")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=100000)
        parser.add_argument("--checkpoint", "--resume-checkpoint", dest="checkpoint", required=True)
        parser.add_argument("--seed", type=int, default=12_321)
        parser.add_argument("--steps-per-frame", type=int, default=1)
        parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--interactive-command", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--forward-speed", type=float, default=0.75)
        parser.add_argument("--backward-speed", type=float, default=0.20)
        parser.add_argument("--lateral-speed", type=float, default=0.20)
        parser.add_argument("--yaw-rate", type=float, default=0.65)
        parser.add_argument("--height-offset", type=float, default=0.07)
        parser.add_argument("--forward-key", default="i")
        parser.add_argument("--backward-key", default="k")
        parser.add_argument("--left-key", default="j")
        parser.add_argument("--right-key", default="l")
        parser.add_argument("--yaw-left-key", default="u")
        parser.add_argument("--yaw-right-key", default="o")
        parser.add_argument("--height-down-key", default="n")
        parser.add_argument("--height-up-key", default="m")
        parser.add_argument("--command-x", type=float, default=0.0)
        parser.add_argument("--command-y", type=float, default=0.0)
        parser.add_argument("--command-yaw", type=float, default=0.0)
        parser.add_argument("--command-height", type=float, default=0.0)
        parser.add_argument("--frame-dt", type=float, default=1.0 / 50.0)
        parser.add_argument("--sim-substeps", type=int, default=4)
        parser.add_argument("--solver-iterations", type=int, default=8)
        parser.add_argument("--velocity-iterations", type=int, default=1)
        parser.add_argument("--action-scale", type=float, default=0.5)
        parser.add_argument("--target-base-height", type=float, default=0.62)
        parser.add_argument("--min-base-height", type=float, default=0.30)
        parser.add_argument("--min-upright-cos", type=float, default=0.35)
        parser.add_argument("--actuator-ke", type=float, default=150.0)
        parser.add_argument("--actuator-kd", type=float, default=5.0)
        parser.add_argument("--max-episode-steps", type=int, default=0)
        parser.add_argument("--log-interval", type=int, default=50)
        parser.add_argument("--render-contacts", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--camera-x", type=float, default=2.2)
        parser.add_argument("--camera-y", type=float, default=-4.0)
        parser.add_argument("--camera-z", type=float, default=1.7)
        parser.add_argument("--camera-pitch", type=float, default=-22.0)
        parser.add_argument("--camera-yaw", type=float, default=28.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
