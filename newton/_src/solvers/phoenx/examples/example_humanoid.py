# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX humanoid MJCF demo.
#
# Runs nv_humanoid.xml directly in the OpenGL viewer. The default is a
# neutral batch; use --stress-random-poses for the old randomized ragdoll
# stress scene.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_humanoid
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_CONTACT_GAP = 0.005


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.world_count = args.world_count
        self.show_contacts = args.show_contacts
        self.sim_dt = self.frame_dt

        rng = np.random.default_rng(args.seed)
        root_rot = wp.quat_identity()
        if args.stress_random_poses:
            axis_np = rng.uniform(-1.0, 1.0, size=3)
            axis_len = np.linalg.norm(axis_np)
            if axis_len > 1.0e-6:
                root_rot = wp.quat_from_axis_angle(wp.vec3(*(axis_np / axis_len)), -wp.pi * 0.5)

        contact_gap = float(args.contact_gap)

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.gap = contact_gap
        articulation_builder.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            ignore_names=["floor", "ground"],
            up_axis="Z",
            parse_sites=False,
            enable_self_collisions=args.self_collisions,
        )

        neutral_q = list(articulation_builder.joint_q)
        articulation_builder.joint_q[:7] = [0.0, 0.0, args.root_height, *root_rot]

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = contact_gap
        for _ in range(self.world_count):
            if args.stress_random_poses:
                articulation_builder.joint_q[7:] = rng.uniform(
                    -args.joint_random_range,
                    args.joint_random_range,
                    size=(len(articulation_builder.joint_q) - 7,),
                ).tolist()
            else:
                articulation_builder.joint_q[7:] = neutral_q[7:]
            builder.add_world(articulation_builder)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.control = self.model.control()

        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=args.solver_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            prepare_refresh_stride=args.prepare_refresh_stride,
        )
        self.contacts = self.model.contacts()

        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        if self.world_count > 1:
            self.viewer.set_world_offsets((3.0, 3.0, 0.0))
        self.viewer.set_camera(pos=wp.vec3(8.0, -10.0, 5.0), pitch=-18.0, yaw=140.0)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.show_contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies have bounded finite state",
            lambda q, qd: (
                (
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
                )
                and q[2] > -1.0
                and q[2] < 5.0
                and max(abs(qd)) < 200.0
            ),
        )

    def test_final(self):
        self.test_post_step()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--seed", type=int, default=123, help="Random seed for stress poses.")
        parser.add_argument("--stress-random-poses", action="store_true", help="Use randomized tilted ragdoll starts.")
        parser.add_argument("--joint-random-range", type=float, default=1.0, help="Stress joint angle range [rad].")
        parser.add_argument("--root-height", type=float, default=1.4, help="Root start height [m].")
        parser.add_argument("--self-collisions", action="store_true", help="Enable MJCF self-collisions.")
        parser.add_argument("--show-contacts", action="store_true", help="Draw contact arrows.")
        parser.add_argument("--contact-gap", type=float, default=DEFAULT_CONTACT_GAP, help="Contact detection gap [m].")
        parser.add_argument(
            "--solver-substeps",
            type=int,
            default=10,
            help="PhoenX internal PGS substeps per solver step.",
        )
        parser.add_argument("--solver-iterations", type=int, default=8, help="PhoenX PGS iterations per substep.")
        parser.add_argument(
            "--velocity-iterations",
            type=int,
            default=1,
            help="PhoenX TGS-soft velocity-relaxation sweeps per substep.",
        )
        parser.add_argument(
            "--prepare-refresh-stride",
            type=int,
            default=1,
            help="Substep interval for refreshing contact effective masses and bias.",
        )
        parser.set_defaults(world_count=100, num_frames=1000, viewer="gl")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
