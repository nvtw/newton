# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX MuJoCo humanoid pile.
#
# Drops multiple MJCF humanoids into one shared collision world. Select
# maximal (full-coordinate) or reduced-coordinate articulation dynamics.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_humanoid_pile
#   python -m newton._src.solvers.phoenx.examples.example_humanoid_pile --articulation-mode reduced
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / args.fps
        self.sim_time = 0.0
        self.show_contacts = args.show_contacts

        humanoid = newton.ModelBuilder(up_axis=newton.Axis.Z)
        humanoid.default_shape_cfg.gap = args.contact_gap
        humanoid.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            ignore_names=("floor", "ground"),
            up_axis="Z",
            parse_sites=False,
            enable_self_collisions=args.self_collisions,
        )
        humanoid.joint_armature = [0.0] * len(humanoid.joint_armature)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.gap = args.contact_gap
        for i in range(args.humanoid_count):
            angle = i * math.pi * (3.0 - math.sqrt(5.0))
            radius = args.spawn_radius * math.sqrt((i % 5) / 4.0)
            position = wp.vec3(
                radius * math.cos(angle),
                radius * math.sin(angle),
                args.spawn_height + i * args.vertical_spacing,
            )
            yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
            tilt_axis = wp.normalize(wp.vec3(math.cos(angle), math.sin(angle), 0.35))
            tilt = wp.quat_from_axis_angle(tilt_axis, args.spawn_tilt * (1.0 if i % 2 == 0 else -1.0))
            builder.add_builder(
                humanoid,
                xform=wp.transform(position, yaw * tilt),
                label_prefix=f"humanoid_{i}",
            )
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            articulation_mode=args.articulation_mode,
            substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            prepare_refresh_stride="auto",
        )
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(9.0, -12.0, 7.0), pitch=-20.0, yaw=145.0)

        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.model.collide(self.state, self.contacts)
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        self.solver.step(self.state, self.state, self.control, self.contacts, self.frame_dt)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.show_contacts:
            self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state,
            "all pile bodies have bounded finite state",
            lambda q, qd: q[2] > -2.0 and q[2] < 30.0 and max(abs(qd)) < 500.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--articulation-mode",
            choices=("maximal", "reduced"),
            default="maximal",
            help="PhoenX articulation representation: full-coordinate 'maximal' or 'reduced'.",
        )
        parser.add_argument("--humanoid-count", type=int, default=12, help="Humanoids in the shared collision world.")
        parser.add_argument("--fps", type=int, default=60, help="Simulation frame rate.")
        parser.add_argument("--sim-substeps", type=int, default=8, help="PhoenX internal substeps per frame.")
        parser.add_argument("--solver-iterations", type=int, default=8, help="PhoenX PGS iterations per substep.")
        parser.add_argument("--velocity-iterations", type=int, default=1, help="PhoenX velocity iterations.")
        parser.add_argument("--spawn-height", type=float, default=1.8, help="Lowest humanoid root height [m].")
        parser.add_argument("--vertical-spacing", type=float, default=0.65, help="Vertical root spacing [m].")
        parser.add_argument("--spawn-radius", type=float, default=0.55, help="Horizontal spawn radius [m].")
        parser.add_argument("--spawn-tilt", type=float, default=0.65, help="Alternating initial root tilt [rad].")
        parser.add_argument("--contact-gap", type=float, default=0.005, help="Contact detection gap [m].")
        parser.add_argument("--self-collisions", action="store_true", help="Enable contacts within each humanoid.")
        parser.add_argument("--show-contacts", action="store_true", help="Draw contact arrows.")
        parser.set_defaults(num_frames=600, viewer="gl")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
