# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot MuJoCo Warp G1 (PhoenX)
#
# Loads the Unitree G1 flat benchmark MJCF from MuJoCo Warp and drives the
# imported position actuators from the bundled shuffle-dance replay.
#
# Command: python -m newton.examples robot_mjwarp_g1_phoenx
#
###########################################################################

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.robot.mjwarp_benchmark_utils import (
    advance_sim_time,
    benchmark_asset_dir,
    download_menagerie_folder,
    make_path_resolver,
    parse_prepare_refresh_stride,
    scatter_replay_targets,
)


def _mujoco_free_qpos_to_newton(qpos: np.ndarray) -> np.ndarray:
    converted = np.asarray(qpos, dtype=np.float32).copy()
    qw, qx, qy, qz = converted[3:7]
    converted[3:7] = (qx, qy, qz, qw)
    return converted


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.world_count = args.world_count
        self.viewer = viewer

        asset_dir = benchmark_asset_dir("unitree_g1")
        menagerie_assets = download_menagerie_folder("unitree_g1/assets")
        path_resolver = make_path_resolver(asset_dir, menagerie_assets)

        replay = np.load(asset_dir / "shuffle_dance.npz")
        replay_ctrl = np.asarray(replay["ctrl"], dtype=np.float32)
        replay_qpos = np.asarray(replay["qpos"], dtype=np.float32)
        replay_times = np.asarray(replay["times"], dtype=np.float64)
        self._replay_channels = int(replay_ctrl.shape[1])
        self._replay_n_frames = int(replay_ctrl.shape[0])
        self._replay_fps = float(1.0 / np.median(np.diff(replay_times)))
        self._replay_loop = 1 if args.loop_replay else 0
        self._replay_speed = float(args.replay_speed)

        g1 = newton.ModelBuilder(up_axis=newton.Axis.Z)
        g1.add_mjcf(
            str(asset_dir / "scene_flat.xml"),
            path_resolver=path_resolver,
            ignore_names=("floor",),
            parse_visuals=args.parse_visuals,
            parse_meshes=args.parse_meshes,
            enable_self_collisions=args.enable_self_collisions,
        )

        qpos0 = _mujoco_free_qpos_to_newton(replay_qpos[0])
        if qpos0.shape[0] != g1.joint_coord_count:
            raise RuntimeError(f"G1 replay qpos has {qpos0.shape[0]} coordinates, expected {g1.joint_coord_count}")
        if self._replay_channels != g1.joint_dof_count - 6:
            raise RuntimeError(f"G1 replay has {self._replay_channels} controls, expected {g1.joint_dof_count - 6}")

        for i, value in enumerate(qpos0):
            g1.joint_q[i] = float(value)
            g1.joint_target_q[i] = float(value)
        for i in range(g1.joint_dof_count):
            g1.joint_qd[i] = 0.0
            g1.joint_target_qd[i] = 0.0
        for channel, value in enumerate(replay_ctrl[0]):
            g1.joint_target_q[7 + channel] = float(value)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(g1, self.world_count)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 2.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            prepare_refresh_stride=args.prepare_refresh_stride,
        )

        self.state_0 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()

        self._sim_time_wp = wp.array([0.0], dtype=float)
        self._replay_ctrl_wp = wp.array(replay_ctrl, dtype=float)
        self._replay_indices_wp = wp.array(self._make_replay_target_indices(), dtype=int)

        self.viewer.set_model(self.model)
        if self.world_count > 1:
            self.viewer.set_world_offsets((2.0, 2.0, 0.0))

        self.capture()

    def _make_replay_target_indices(self) -> np.ndarray:
        joint_count_per_world = self.model.joint_count // self.world_count
        target_q_start = self.model.joint_target_q_start.numpy()

        indices = np.empty((self.world_count, self._replay_channels), dtype=np.int32)
        for world_idx in range(self.world_count):
            joint_base = world_idx * joint_count_per_world
            first_actuated_joint = joint_base + 1
            joints = first_actuated_joint + np.arange(self._replay_channels, dtype=np.int32)
            indices[world_idx, :] = target_q_start[joints]
        return indices

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(
            scatter_replay_targets,
            dim=(self.world_count, self._replay_channels),
            inputs=[
                self._sim_time_wp,
                self._replay_speed,
                self._replay_fps,
                self._replay_n_frames,
                self._replay_loop,
                self._replay_ctrl_wp,
                self._replay_indices_wp,
                self.control.joint_target_q,
            ],
        )
        self.model.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)
        wp.launch(advance_sim_time, dim=1, inputs=[self._sim_time_wp, self.frame_dt])

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "benchmark bodies stayed in a finite visual range",
            lambda q, qd: q[2] > -5.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--fps", type=int, default=60, help="Simulation frame rate.")
        parser.add_argument("--sim-substeps", type=int, default=6, help="PhoenX internal substeps per frame.")
        parser.add_argument("--solver-iterations", type=int, default=8, help="PhoenX PGS iterations per substep.")
        parser.add_argument("--velocity-iterations", type=int, default=1, help="PhoenX velocity iterations.")
        parser.add_argument(
            "--prepare-refresh-stride",
            type=parse_prepare_refresh_stride,
            default="auto",
            help="PhoenX prepare refresh stride, or 'auto'.",
        )
        parser.add_argument(
            "--loop-replay",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Loop the MuJoCo Warp replay after the final frame.",
        )
        parser.add_argument("--replay-speed", type=float, default=1.0, help="Playback speed for the replay controls.")
        parser.add_argument(
            "--parse-visuals",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Import visual geometries from the MJCF.",
        )
        parser.add_argument(
            "--parse-meshes",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Import mesh geometries from the MJCF.",
        )
        parser.add_argument(
            "--enable-self-collisions",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable self-collision while importing G1.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
