# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX humanoid MJCF demo.
#
# Recreates the randomized nv_humanoid.xml batch from
# ``newton.examples.basic.example_recording``, but runs it directly in the
# OpenGL viewer instead of writing a ViewerFile recording.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_humanoid
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.world_count = args.world_count
        self.frame_substeps = args.frame_substeps
        self.sim_dt = self.frame_dt / self.frame_substeps

        rng = np.random.default_rng(args.seed)
        start_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.uniform(-1.0, 1.0, size=3))), -wp.pi * 0.5)

        articulation_builder = newton.ModelBuilder()
        articulation_builder.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            ignore_names=["floor", "ground"],
            up_axis="Z",
            parse_sites=False,
            enable_self_collisions=args.self_collisions,
        )

        # Joint initial positions: free-root translation/orientation plus
        # randomized internal joint angles for each world.
        articulation_builder.joint_q[:7] = [0.0, 0.0, 1.5, *start_rot]

        builder = newton.ModelBuilder()
        for _ in range(self.world_count):
            articulation_builder.joint_q[7:] = rng.uniform(
                -1.0, 1.0, size=(len(articulation_builder.joint_q) - 7,)
            ).tolist()
            builder.add_world(articulation_builder)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.control = self.model.control()

        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=args.solver_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            prepare_refresh_stride="auto",
        )
        self.contacts = self.model.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.0, -10.0, 5.0), pitch=-18.0, yaw=140.0)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.frame_substeps):
            self.model.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
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
            "all bodies have finite state",
            lambda q, qd: (
                not (
                    wp.isnan(q[0])
                    or wp.isnan(q[1])
                    or wp.isnan(q[2])
                    or wp.isnan(q[3])
                    or wp.isnan(q[4])
                    or wp.isnan(q[5])
                    or wp.isnan(q[6])
                    or wp.isnan(qd[0])
                    or wp.isnan(qd[1])
                    or wp.isnan(qd[2])
                    or wp.isnan(qd[3])
                    or wp.isnan(qd[4])
                    or wp.isnan(qd[5])
                )
            ),
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--seed", type=int, default=123, help="Random seed for initial humanoid poses.")
        parser.add_argument("--self-collisions", action="store_true", help="Enable MJCF self-collisions.")
        parser.add_argument(
            "--frame-substeps",
            type=int,
            default=2,
            help="Outer collision/solver steps per rendered frame.",
        )
        parser.add_argument(
            "--solver-substeps",
            type=int,
            default=4,
            help="PhoenX internal PGS substeps per solver step.",
        )
        parser.add_argument("--solver-iterations", type=int, default=8, help="PhoenX PGS iterations per substep.")
        parser.add_argument(
            "--velocity-iterations",
            type=int,
            default=1,
            help="PhoenX TGS-soft velocity-relaxation sweeps per substep.",
        )
        parser.set_defaults(world_count=100, num_frames=1000, viewer="gl")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
