# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot MuJoCo Warp Panda (PhoenX)
#
# Loads the Franka Emika Panda benchmark MJCF from MuJoCo Warp and simulates
# it with SolverPhoenX.
#
# Command: python -m newton.examples robot_mjwarp_panda_phoenx
#
###########################################################################

import argparse

import warp as wp

import newton
import newton.examples
from newton.examples.robot.mjwarp_benchmark_utils import (
    benchmark_asset_dir,
    download_menagerie_folder,
    make_path_resolver,
    parse_prepare_refresh_stride,
)


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.world_count = args.world_count
        self.viewer = viewer

        asset_dir = benchmark_asset_dir("franka_emika_panda")
        menagerie_assets = download_menagerie_folder("franka_emika_panda/assets")
        path_resolver = make_path_resolver(asset_dir, menagerie_assets)

        panda = newton.ModelBuilder(up_axis=newton.Axis.Z)
        panda.add_mjcf(
            str(asset_dir / "scene.xml"),
            path_resolver=path_resolver,
            ignore_names=("floor",),
            parse_visuals=args.parse_visuals,
            parse_meshes=args.parse_meshes,
            enable_self_collisions=args.enable_self_collisions,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(panda, self.world_count)
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

        self.viewer.set_model(self.model)
        if self.world_count > 1:
            self.viewer.set_world_offsets((1.5, 1.5, 0.0))

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)

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
        parser.add_argument("--sim-substeps", type=int, default=4, help="PhoenX internal substeps per frame.")
        parser.add_argument("--solver-iterations", type=int, default=8, help="PhoenX PGS iterations per substep.")
        parser.add_argument("--velocity-iterations", type=int, default=1, help="PhoenX velocity iterations.")
        parser.add_argument(
            "--prepare-refresh-stride",
            type=parse_prepare_refresh_stride,
            default="auto",
            help="PhoenX prepare refresh stride, or 'auto'.",
        )
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
            help="Enable self-collision while importing Panda.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
