# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Body Chain
#
# Ten unit cubes arranged in a zig-zag along +x and connected with ball
# joints at *diagonal* corners. Cube j's centre is placed at
#   (2j+1)*h, +h if j even else -h, 0
# so consecutive cubes share exactly one corner: the joint between cube
# k-1 and cube k sits on the centerline at (2k*h, 0, 0), which is the
# +x/-y (or +x/+y) corner of cube k-1 and the opposite -x/+y (or -x/-y)
# corner of cube k. Cube 0's bottom-left-back corner (-h, -h, 0) is
# anchored to a static "world" body at the origin so the chain hangs in
# mid-air. Gravity (-y) pulls everything down and the joints keep the
# zig-zag together.
#
# The chain joint type is chosen by the module-level ``JOINT_KIND``
# selector (:class:`JointKind`) so you can directly compare the three
# ball-socket implementations available in jitter: the legacy fused
# ball-socket, the unified ``ACTUATED_DOUBLE_BALL_SOCKET`` in
# ball-socket mode (which we want to eventually consolidate all joints
# onto), and the 6-DoF D6 configured as a ball-socket (rigid linear,
# free angular). All three lock exactly the same 3 positional DoF at
# the shared corner and leave all 3 rotational DoF free -- the only
# differences are solver structure and feature set.
#
# Constructed via :class:`WorldBuilder` -- the recommended high-level
# API. The builder packs descriptors into the SoA body container and
# the column-major constraint container, then hands an assembled
# :class:`World` back ready to step.
#
# Run:  python -m newton._src.solvers.jitter.example_body_chain
#
###########################################################################

import enum
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel
from newton._src.solvers.jitter.world_builder import (
    D6AxisDrive,
    JointMode,
    WorldBuilder,
)


class JointKind(enum.Enum):
    """Which kind of ball-socket the chain is built with.

    All three modes lock the same 3 translational DoF at the shared
    corner and leave all 3 rotational DoF free, but reach that
    constraint with different solver structures:

    * :attr:`BALL_SOCKET`                 -- fused
      ``CONSTRAINT_TYPE_BALL_SOCKET`` (single 3x3 direct solve per
      column, the oldest and most specialized path).
    * :attr:`ACTUATED_DOUBLE_BALL_SOCKET` -- the unified joint in
      :attr:`JointMode.BALL_SOCKET` mode: also a single 3x3 direct
      solve per column, but shares one constraint schema / dispatch
      arm with the revolute and prismatic modes of the same constraint
      type. Intended to eventually replace the legacy
      :attr:`BALL_SOCKET`.
    * :attr:`D6_BALL_SOCKET`              -- generalized D6 with the
      three linear axes rigidly locked and the three angular axes
      free (``D6AxisDrive(max_force=0)``). Internally dispatches to
      the fused 3-DoF point constraint for the linear block and three
      1-DoF free axes for the angular block (Jolt ``SixDOFConstraint``
      style).
    """

    BALL_SOCKET = "ball_socket"
    ACTUATED_DOUBLE_BALL_SOCKET = "actuated_double_ball_socket"
    D6_BALL_SOCKET = "d6_ball_socket"


# Selects which constraint type every joint in the chain is built
# with. Switch this to compare solver behaviour / robustness across
# the three implementations of a ball-socket.
JOINT_KIND = JointKind.ACTUATED_DOUBLE_BALL_SOCKET

# Toggle the initial state of the chain.
#
# False (default): zig-zag along +x with identity-oriented cubes touching
#     diagonally on the y=0 centerline. Pretty starting pose; gravity
#     immediately swings it into motion -- useful as a dynamics demo.
#
# True: cubes hang *straight down* from the world anchor with each cube
#     rotated 45 degrees about z so its body-frame diagonal aligns with
#     gravity. This is the static equilibrium configuration: in body
#     frame the joints sit at ``(+h, +h, 0)`` (top corner) and
#     ``(-h, -h, 0)`` (bottom corner); the 45-degree rotation puts both
#     of those corners on the world y axis at distance ``h*sqrt(2)``
#     from the cube centre, so adjacent cubes meet corner-to-corner
#     along -y. Cube j's centre is at ``y = -(2j+1)*h*sqrt(2)``; joint k
#     between cube k-1 and cube k sits at ``y = -k * 2*h*sqrt(2)``.
#     Gravity is balanced by tension in the joints, so the chain just
#     hangs there (modulo PGS jitter). Useful for verifying the solver
#     doesn't drift from a known equilibrium.
START_AT_EQUILIBRIUM = True

NUM_CUBES = 10
HALF_EXTENT = 0.5  # unit cube -> half-extent 0.5 on each axis
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor body
NUM_BALL_SOCKETS = NUM_CUBES  # 1 world->cube0 + 9 cube_i->cube_(i+1)

# Identity body-frame inverse inertia. A unit cube of mass 1 actually has
# I = m*s^2/6 * I_3 = 1/6 * I_3 so I^-1 = 6 * I_3, but identity keeps
# the demo numbers round and is fine for an unstressed visual test.
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# Quaternion for a 45-degree rotation about +z (xyzw convention; matches
# Newton/Warp). After this rotation the body-frame diagonal corners
# (+h, +h, 0) and (-h, -h, 0) end up on the world +y / -y axis at
# distance h*sqrt(2) from the cube centre.
_DIAGONAL_HALF = HALF_EXTENT * math.sqrt(2.0)
_HALF_ANGLE = math.pi / 8.0  # half of 45 degrees
_DIAGONAL_QUAT = (0.0, 0.0, math.sin(_HALF_ANGLE), math.cos(_HALF_ANGLE))


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # ---- Build the Jitter World via WorldBuilder ------------------
        b = WorldBuilder()
        world_body = b.world_body  # body 0, auto-created static anchor

        # Pre-build the all-rigid linear / all-free angular specs for
        # the D6 ball-socket once -- same for every joint in the chain.
        if JOINT_KIND is JointKind.D6_BALL_SOCKET:
            d6_rigid_linear = (D6AxisDrive(), D6AxisDrive(), D6AxisDrive())
            d6_free_angular = (
                D6AxisDrive(max_force=0.0),
                D6AxisDrive(max_force=0.0),
                D6AxisDrive(max_force=0.0),
            )
        else:
            d6_rigid_linear = None
            d6_free_angular = None

        def add_chain_joint(body_a: int, body_b: int, anchor: tuple[float, float, float]) -> None:
            """Insert one ball-socket between ``body_a`` and ``body_b``.

            Dispatches on :data:`JOINT_KIND` so the caller is spared the
            per-kind book-keeping. All three branches produce the same
            physical joint (3-DoF point lock, free rotations).
            """
            if JOINT_KIND is JointKind.BALL_SOCKET:
                b.add_ball_socket(body_a, body_b, anchor)
            elif JOINT_KIND is JointKind.ACTUATED_DOUBLE_BALL_SOCKET:
                b.add_joint(
                    body1=body_a,
                    body2=body_b,
                    anchor1=anchor,
                    mode=JointMode.BALL_SOCKET,
                )
            elif JOINT_KIND is JointKind.D6_BALL_SOCKET:
                b.add_d6(
                    body1=body_a,
                    body2=body_b,
                    anchor=anchor,
                    angular=d6_free_angular,
                    linear=d6_rigid_linear,
                )
            else:
                raise ValueError(f"unknown JOINT_KIND: {JOINT_KIND}")

        cube_ids: list[int] = []
        if START_AT_EQUILIBRIUM:
            # Diamond-rotated cubes hanging straight down from origin.
            # Cube j centre at (0, -(2j+1)*h*sqrt(2), 0), all share the
            # same 45-degree-about-z orientation. Joints land on the
            # world y axis between consecutive cubes.
            for j in range(NUM_CUBES):
                cube_ids.append(
                    b.add_dynamic_body(
                        position=(0.0, -(2 * j + 1) * _DIAGONAL_HALF, 0.0),
                        orientation=_DIAGONAL_QUAT,
                        inverse_mass=1.0,
                        inverse_inertia=_INV_INERTIA,
                    )
                )

            for k in range(NUM_BALL_SOCKETS):
                body_a = world_body if k == 0 else cube_ids[k - 1]
                body_b = cube_ids[k]
                # Joint k sits on the y axis at -k * 2 * h * sqrt(2):
                # k = 0 is at the origin (the world anchor), and each
                # subsequent joint is one full diagonal further down.
                anchor = (0.0, -k * 2.0 * _DIAGONAL_HALF, 0.0)
                add_chain_joint(body_a, body_b, anchor)
        else:
            # Cube j sits at x = (2j+1)*h on +x and alternates y = +h
            # (j even) / -h (j odd), so consecutive cubes meet at
            # exactly one diagonal corner on the y=0 line.
            for j in range(NUM_CUBES):
                sign = 1.0 if j % 2 == 0 else -1.0
                cube_ids.append(
                    b.add_dynamic_body(
                        position=((2 * j + 1) * HALF_EXTENT, sign * HALF_EXTENT, 0.0),
                        inverse_mass=1.0,
                        inverse_inertia=_INV_INERTIA,
                    )
                )

            # Joint k anchors:
            #   * k = 0 : world (origin) <-> cube 0's (-h, -h, 0) corner at (0,0,0)
            #   * k>=1 : cube k-1 <-> cube k at (2*k*h, 0, 0) -- their shared
            #            corner on the centerline.
            for k in range(NUM_BALL_SOCKETS):
                body_a = world_body if k == 0 else cube_ids[k - 1]
                body_b = cube_ids[k]
                anchor = (0.0, 0.0, 0.0) if k == 0 else (2.0 * k * HALF_EXTENT, 0.0, 0.0)
                add_chain_joint(body_a, body_b, anchor)

        # substeps=1 here because we drive substepping ourselves from
        # simulate() (matches the basic-pendulum pattern).
        self.world = b.finalize(
            substeps=1,
            solver_iterations=8,
            device=self.device,
        )

        # ---- Rendering scratch ---------------------------------------
        # Reused per frame so we don't allocate inside render().
        self._xforms = wp.zeros(NUM_BODIES, dtype=wp.transform, device=self.device)

        # ---- Picking --------------------------------------------------
        # Half-extents per body in body-local frame; (0, 0, 0) marks the
        # world anchor as non-pickable. All cubes are unit -> half=0.5.
        half_extents_np = np.zeros((NUM_BODIES, 3), dtype=np.float32)
        half_extents_np[1:] = HALF_EXTENT
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = JitterPicking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Re-inject the picking force every Jitter step. The force
            # accumulator is consumed and zeroed at the end of each
            # ``step()`` (Jitter's two-stage IntegrateForces split), so
            # without re-injection the user would feel a 1-frame pulse
            # rather than a continuous spring. Also a no-op kernel
            # launch when nothing is picked, keeping the path graph-
            # capture-friendly.
            self.picking.apply_force()
            self.world.step(self.sim_dt)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        wp.launch(
            pack_body_xforms_kernel,
            dim=NUM_BODIES,
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )

        self.viewer.begin_frame(self.sim_time)
        # Render the dynamic cubes only (skip the static world body at idx 0).
        self.viewer.log_shapes(
            "/world/cubes",
            newton.GeoType.BOX,
            (HALF_EXTENT, HALF_EXTENT, HALF_EXTENT),
            self._xforms[1:],
        )
        self.viewer.end_frame()

    def test_final(self):
        # After the run the chain should have swung from the world anchor
        # but no body should have fallen below y = -10 (a generous lower
        # bound that catches outright explosions / NaNs).
        positions = self.world.bodies.position.numpy()
        for i in range(1, NUM_BODIES):
            assert np.isfinite(positions[i]).all(), f"body {i} produced non-finite position"
            assert positions[i, 1] > -10.0, f"body {i} fell unreasonably far ({positions[i, 1]})"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
