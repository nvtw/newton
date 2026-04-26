# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX scale (kitchen / bathroom scale) demo.
#
# A 1 m x 1 m x 0.05 m platform is held above a static base by a +z
# prismatic joint with a *very* short stroke (5 cm) and a PD spring-
# damper at the lower limit (``limit_ke``, ``limit_kd``). The platform's
# weight plus 25 small cubes piled on top compresses the spring; at
# static equilibrium the deflection past the limit equals the linear
# Hooke's law result::
#
#     dx = (m_platform + n * m_cube) * g / limit_ke
#
# The unit test in
# :mod:`newton._src.solvers.phoenx.tests.test_scale_hookes_law`
# verifies this analytic prediction across a sweep of platform / cube
# masses and stiffness values.
#
# Implementation note: ``PortedExample`` (the shared example base used
# by every other Box2D / JitterPhysics2 port) drives a
# ``PhoenXWorld`` directly with ``num_joints=0`` -- it's a
# contact-only harness. This scale demo needs an articulated
# prismatic joint, so it builds the scene via Newton's
# :class:`newton.ModelBuilder` and steps via
# :class:`newton.solvers.SolverPhoenX` instead.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_phoenx_scale
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples

# ---- Geometry ------------------------------------------------------------
PLATFORM_HX = 0.5
PLATFORM_HY = 0.5
PLATFORM_HZ = 0.025
PLATFORM_MASS = 5.0  # kg

CUBE_HE = 0.07
CUBE_MASS = 0.5  # kg
CUBE_GRID = 5    # 5 x 5 = 25 cubes

# ---- Spring config -------------------------------------------------------
LIMIT_KE = 50_000.0   # N/m
# Critical damping: ``kd_crit = 2 * sqrt(ke * M)`` with M = 17.5 kg
# (5 kg platform + 25 * 0.5 kg cubes) ~= 1870 N*s/m. Round up so the
# platform settles inside ~1 second instead of oscillating.
LIMIT_KD = 2000.0     # N*s/m
LIMIT_LOWER = -0.05   # m
LIMIT_UPPER = 0.05    # m

GRAVITY = 9.81
SUBSTEPS = 16
SOLVER_ITERATIONS = 32
FPS = 60
# Two ``solver.step`` calls per ``simulate()`` -- with the
# ``state_0 <-> state_1`` swap inside the loop, an even substep count
# keeps the original ``state_0`` device pointer pointing at the
# latest data after each captured-graph replay (an odd count leaves
# the latest data in ``state_1`` and the captured replay then reads
# stale memory). See the Anymal walk example for the same pattern.
GRAPH_SUBSTEPS = 2


def _build_scale_model() -> newton.Model:
    """Build the scale's Newton model: static base, platform on a
    prismatic joint, 5x5 cube grid sitting on top."""
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    cfg_static = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6)

    # Static base (visual footprint under the platform).
    mb.add_shape_box(
        -1,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.05), q=wp.quat_identity()),
        hx=PLATFORM_HX + 0.05,
        hy=PLATFORM_HY + 0.05,
        hz=0.05,
        cfg=cfg_static,
    )

    # Platform link (the weighing surface). Spawn at z = 0.20 so the
    # joint's neutral position (q=0) puts the platform a bit above the
    # base; gravity pulls q < 0 until the lower-limit spring holds it.
    platform_z0 = 0.20
    ixx = PLATFORM_MASS / 3.0 * (PLATFORM_HY ** 2 + PLATFORM_HZ ** 2)
    iyy = PLATFORM_MASS / 3.0 * (PLATFORM_HX ** 2 + PLATFORM_HZ ** 2)
    izz = PLATFORM_MASS / 3.0 * (PLATFORM_HX ** 2 + PLATFORM_HY ** 2)
    plat = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, platform_z0), q=wp.quat_identity()),
        mass=PLATFORM_MASS,
        inertia=((ixx, 0, 0), (0, iyy, 0), (0, 0, izz)),
    )
    mb.add_shape_box(plat, hx=PLATFORM_HX, hy=PLATFORM_HY, hz=PLATFORM_HZ, cfg=cfg_static)

    joint = mb.add_joint_prismatic(
        parent=-1,
        child=plat,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, platform_z0), q=wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=(0.0, 0.0, 1.0),
        limit_lower=LIMIT_LOWER,
        limit_upper=LIMIT_UPPER,
        limit_ke=LIMIT_KE,
        limit_kd=LIMIT_KD,
    )
    mb.add_articulation([joint])

    # Pile 5 x 5 = 25 cubes on the platform.
    cixx = CUBE_MASS / 3.0 * (CUBE_HE ** 2 + CUBE_HE ** 2)
    cinertia = ((cixx, 0.0, 0.0), (0.0, cixx, 0.0), (0.0, 0.0, cixx))
    spacing = 2.2 * CUBE_HE
    x0 = -((CUBE_GRID - 1) * spacing) * 0.5
    for i in range(CUBE_GRID):
        for j in range(CUBE_GRID):
            cube = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(
                        x0 + i * spacing,
                        x0 + j * spacing,
                        platform_z0 + PLATFORM_HZ + CUBE_HE + 0.005,
                    ),
                    q=wp.quat_identity(),
                ),
                mass=CUBE_MASS,
                inertia=cinertia,
            )
            mb.add_shape_box(cube, hx=CUBE_HE, hy=CUBE_HE, hz=CUBE_HE, cfg=cfg_static)

    mb.gravity = -GRAVITY
    return mb.finalize()


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.frame_dt = 1.0 / FPS
        self.sim_time = 0.0

        self.model = _build_scale_model()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching="sticky"
        )
        self.contacts = self.collision_pipeline.contacts()

        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=SUBSTEPS,
            solver_iterations=SOLVER_ITERATIONS,
            velocity_iterations=1,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 0.7), pitch=-15.0, yaw=135.0)

        # CUDA graph capture for the per-frame step.
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self) -> None:
        sub_dt = self.frame_dt / GRAPH_SUBSTEPS
        for _ in range(GRAPH_SUBSTEPS):
            self.state_0.clear_forces()
            self.model.collide(
                self.state_0,
                contacts=self.contacts,
                collision_pipeline=self.collision_pipeline,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
