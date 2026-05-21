# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Shapes
#
# Shows how to programmatically create a variety of
# collision shapes using the newton.ModelBuilder() API.
# Supports XPBD (default), VBD, and PhoenX solvers.
#
# Command:    python -m newton.examples basic_shapes
# With VBD:    python -m newton.examples basic_shapes --solver vbd
# With PhoenX: python -m newton.examples basic_shapes --solver phoenx
#
#
###########################################################################

import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        # Per-solver substepping. PhoenX handles substeps internally
        # (one kernel launch per outer step, vs. one per inner step for
        # XPBD/VBD), so we want as few outer steps as possible to keep
        # graph-capture overhead and ``model.collide()`` work down. The
        # outer loop swaps ``state_0`` and ``state_1`` after each
        # ``solver.step``; CUDA graph capture bakes those references in,
        # so the outer count must be EVEN -- otherwise the captured
        # graph reads from the same stale slot every replay and the
        # simulation freezes after one frame. We use the smallest even
        # count: 2 outer steps per frame for PhoenX, plus
        # ``substeps=5`` internally for ~10 effective substeps per
        # frame (parity with the XPBD branch's 10x outer loop).
        if self.solver_type == "phoenx":
            self.sim_substeps = 2
            self.sim_dt = self.frame_dt / self.sim_substeps
        else:
            self.sim_substeps = 10
            self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()

        builder.default_shape_cfg.mu = 0.5  # Friction coefficient

        if self.solver_type == "vbd":
            # VBD: Higher stiffness for stable rigid body contacts
            builder.default_shape_cfg.ke = 1.0e6  # Contact stiffness
            builder.default_shape_cfg.kd = 1.0e1  # Contact damping
        elif self.solver_type == "phoenx":
            # PhoenX: PGS solver consumes contacts from the shared
            # collision pipeline; it does not use ke/kd (rigid contact),
            # but it does honour mu_torsional / mu_rolling when the
            # adapter wires them through. Use the same XPBD defaults
            # for cross-solver consistency.
            builder.default_shape_cfg.mu_torsional = 0.01
            builder.default_shape_cfg.mu_rolling = 3e-3
        else:
            builder.default_shape_cfg.mu_torsional = 0.01  # Contact stiffness
            builder.default_shape_cfg.mu_rolling = 3e-3  # Contact stiffness

        # add ground plane
        builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        self.sphere_pos = wp.vec3(0.0, -2.0, drop_z)
        body_sphere = builder.add_body(xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()), label="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # ELLIPSOID (flat disk shape: a=b > c for stability when resting on ground)
        self.ellipsoid_pos = wp.vec3(0.0, -6.0, drop_z)
        body_ellipsoid = builder.add_body(
            xform=wp.transform(p=self.ellipsoid_pos, q=wp.quat_identity()), label="ellipsoid"
        )
        builder.add_shape_ellipsoid(body_ellipsoid, rx=0.5, ry=0.5, rz=0.25)

        # CAPSULE (tilted slightly off vertical so it doesn't drop perfectly axis-aligned)
        self.capsule_pos = wp.vec3(0.0, 0.0, drop_z)
        self.capsule_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.1)
        body_capsule = builder.add_body(xform=wp.transform(p=self.capsule_pos, q=self.capsule_rot), label="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # CYLINDER
        self.cylinder_pos = wp.vec3(0.0, -4.0, drop_z)
        body_cylinder = builder.add_body(
            xform=wp.transform(p=self.cylinder_pos, q=wp.quat_identity()), label="cylinder"
        )
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

        # BOX
        self.box_pos = wp.vec3(0.0, 2.0, drop_z)
        body_box = builder.add_body(xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), label="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)

        # MESH (bunny)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        self.mesh_pos = wp.vec3(0.0, 4.0, drop_z - 0.5)
        body_mesh = builder.add_body(xform=wp.transform(p=self.mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), label="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        # CONE (no collision support in the standard collision pipeline)
        self.cone_pos = wp.vec3(0.0, 6.0, drop_z)
        body_cone = builder.add_body(xform=wp.transform(p=self.cone_pos, q=wp.quat_identity()), label="cone")
        builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)

        # EXTRA CAPSULES: a column of capsules stepping out in +y and up in +z.
        # Each is offset 5 m further along y and starts 5 m higher than the
        # previous one so they drop in a cascade. All share the original
        # capsule's small tilt so they tip onto their sides instead of
        # balancing on end.
        self.extra_capsule_count = 10
        self.extra_capsule_y_spacing = 5.0
        self.extra_capsule_z_spacing = 5.0
        self.extra_capsule_y0 = 10.0  # first extra capsule sits past the cone (y=6)
        self.extra_capsule_positions: list[wp.vec3] = []
        for i in range(self.extra_capsule_count):
            pos = wp.vec3(
                0.0,
                self.extra_capsule_y0 + i * self.extra_capsule_y_spacing,
                drop_z + (i + 1) * self.extra_capsule_z_spacing,
            )
            self.extra_capsule_positions.append(pos)
            body = builder.add_body(
                xform=wp.transform(p=pos, q=self.capsule_rot),
                label=f"capsule_{i + 1}",
            )
            builder.add_shape_capsule(body, radius=0.3, half_height=0.7)

        # Color rigid bodies for VBD solver
        if self.solver_type == "vbd":
            builder.color()

        # finalize model
        self.model = builder.finalize()

        # Create solver based on type
        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=10,
            )
        elif self.solver_type == "phoenx":
            # PhoenX handles its substep loop internally, so the outer
            # ``sim_substeps`` is just 2 (the minimum even count
            # required for the swap-based graph-capture pattern in
            # ``simulate``). 2 outer * 5 inner = 10 effective substeps
            # per frame -- parity with the XPBD branch.
            #
            # ``step_layout="single_world"`` is right for a single big
            # scene like this; the default ``"multi_world"`` is tuned
            # for many small worlds (~256+) and pays per-world dispatch
            # overhead with no upside on a 7-body scene.
            self.solver = newton.solvers.SolverPhoenX(
                self.model,
                substeps=3,
                solver_iterations=5,
                step_layout="single_world",
            )
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        # Set camera to view all the shapes
        self.viewer.set_camera(
            pos=wp.vec3(10.0, -1.3, 2.0),
            pitch=0.0,
            yaw=-180.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 70.0

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        # Run collision once per frame and reuse the contact set
        # across all substeps. ``model.collide()`` is by far the most
        # expensive op in this example (broadphase + narrowphase over
        # 7 shapes + ground), and the contact geometry barely shifts
        # within a single frame, so re-running it per substep is wasted
        # work for all three solver backends.
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.model.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        self.sphere_pos[2] = 0.5
        sphere_q = wp.transform(self.sphere_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "sphere at rest pose",
            lambda q, qd: newton.math.vec_allclose(q, sphere_q, atol=2e-4),
            [0],
        )
        # Ellipsoid with a=b=0.5, c=0.25 is stable (flat disk), rests at z=0.25
        self.ellipsoid_pos[2] = 0.25
        ellipsoid_q = wp.transform(self.ellipsoid_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "ellipsoid at rest pose",
            lambda q, qd: newton.math.vec_allclose(q, ellipsoid_q, atol=2e-2),
            [1],
        )
        # Capsule starts tilted ~0.1 rad off vertical, so it tips and settles
        # on its side (body center at z = radius = 0.3) rather than upright.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "capsule at rest pose",
            lambda q, qd: abs(q[0]) < 0.1 and abs(q[1]) < 0.1 and q[2] > 0.0 and q[2] < 0.35,
            [2],
        )
        # Custom test for cylinder: allow 0.01 error for X and Y, strict for Z and rotation
        self.cylinder_pos[2] = 0.6
        cylinder_q = wp.transform(self.cylinder_pos, wp.quat_identity())
        # fmt: off
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cylinder at rest pose",
            lambda q, qd: abs(q[0] - cylinder_q[0]) < 0.01
            and abs(q[1] - cylinder_q[1]) < 0.01
            and abs(q[2] - cylinder_q[2]) < 1e-4
            and abs(q[3] - cylinder_q[3]) < 1e-4
            and abs(q[4] - cylinder_q[4]) < 1e-4
            and abs(q[5] - cylinder_q[5]) < 1e-4
            and abs(q[6] - cylinder_q[6]) < 1e-4,
            [3],
        )
        # fmt: on
        self.box_pos[2] = 0.25
        box_q = wp.transform(self.box_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "box at rest pose",
            lambda q, qd: newton.math.vec_allclose(q, box_q, atol=0.1),
            [4],
        )
        # we only test that the bunny didn't fall through the ground and didn't slide too far
        # Allow slight penetration (z > -0.05) due to contact reduction
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "bunny at rest pose",
            lambda q, qd: q[2] > -0.05 and abs(q[0]) < 0.1 and abs(q[1] - 4.0) < 0.1,
            [5],
        )
        # Extra capsules start at body index 7 (after sphere, ellipsoid, capsule,
        # cylinder, box, mesh, cone). They each drop from progressively higher
        # ``z`` and tip onto their sides, settling with the body center at
        # ``z ~= radius = 0.3`` and roughly at their initial ``y`` (small
        # roll-out allowed because the tilt is around the +y axis).
        for i, pos in enumerate(self.extra_capsule_positions):
            expected_y = pos[1]
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                f"capsule_{i + 1} at rest pose",
                lambda q, qd, ey=expected_y: abs(q[0]) < 1.0 and abs(q[1] - ey) < 1.0 and q[2] > 0.0 and q[2] < 0.35,
                [7 + i],
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Extend the shared examples parser with a solver choice
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="phoenx",
        choices=["vbd", "xpbd", "phoenx"],
        help="Solver type: xpbd (default), vbd, or phoenx",
    )

    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
