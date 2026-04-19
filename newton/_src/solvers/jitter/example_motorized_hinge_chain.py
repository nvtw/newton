# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Motorized Hinge Chain
#
# Same scene as :mod:`example_body_chain` with ``START_AT_EQUILIBRIUM=True``
# (ten unit cubes rotated 45 degrees about z so they hang in a straight
# diamond column from the world anchor along -y), but every joint is now
# a *motorized* hinge instead of a plain ball-socket.
#
# Each hinge:
#   * is composed -- at the WorldBuilder level -- of a HingeAngle (locks
#     the two angular DoFs orthogonal to the axis), a BallSocket (locks
#     the three positional DoFs at the shared corner), and an
#     AngularMotor (drives the relative angular velocity along the axis
#     toward zero), exactly like Jitter2's ``HingeJoint``;
#   * uses a hinge axis aligned with the *cube edge* that passes through
#     the joint corner. With the 45-degree-about-z rotation the four
#     vertical body-frame edges (the ones parallel to body z) all stay
#     aligned with world ``+z``, so the world-space hinge axis is just
#     ``(0, 0, 1)``;
#   * runs in *velocity mode* with ``target_velocity = 0``: the motor
#     fights any relative spin around the hinge axis and acts as a
#     strong angular damper. This is not a position spring -- so a small
#     residual angle can slowly drift -- but in the equilibrium pose
#     gravity does not torque the chain about z, so the motors should
#     stay near zero impulse and the chain hangs as before.
#     A position-mode (PD) motor will be added later.
#
# The picking, viewer, render, and ``test_final`` plumbing all match
# example_body_chain.py so the two examples can be compared side by side.
#
# Run:  python -m newton._src.solvers.jitter.example_motorized_hinge_chain
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel
from newton._src.solvers.jitter.world_builder import WorldBuilder

NUM_CUBES = 10
HALF_EXTENT = 0.5
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor body
NUM_HINGES = NUM_CUBES  # 1 world->cube0 + 9 cube_i->cube_(i+1)

# Identity body-frame inverse inertia (matches example_body_chain).
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# 45-degree rotation about +z (xyzw); puts the body-frame diagonal
# corners on the world y axis at distance h*sqrt(2) from the COM. Note
# that this rotation leaves the body-frame z axis pointing along world
# +z, so the four vertical cube edges -- which run from (+-h, +-h, -h)
# to (+-h, +-h, +h) in body frame -- stay parallel to world z.
_DIAGONAL_HALF = HALF_EXTENT * math.sqrt(2.0)
_HALF_ANGLE = math.pi / 8.0  # half of 45 degrees
_DIAGONAL_QUAT = (0.0, 0.0, math.sin(_HALF_ANGLE), math.cos(_HALF_ANGLE))

# Hinge axis: the cube edge that passes through the shared corner
# between consecutive diamond-rotated cubes. With the rotation above
# this is the world z axis.
_HINGE_AXIS = (0.0, 0.0, 1.0)

# Maximum motor torque [N·m]. Generous so the motor can hold zero
# relative axial spin even under the (small) precession the chain feels
# from PGS jitter; the motor is still velocity-mode so it doesn't fight
# slow drift, only motion.
_MOTOR_MAX_FORCE = 50.0


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

        cube_ids: list[int] = []
        for j in range(NUM_CUBES):
            cube_ids.append(
                b.add_dynamic_body(
                    position=(0.0, -(2 * j + 1) * _DIAGONAL_HALF, 0.0),
                    orientation=_DIAGONAL_QUAT,
                    inverse_mass=1.0,
                    inverse_inertia=_INV_INERTIA,
                )
            )

        # One motorized hinge joint per junction. Joint k sits on the
        # world y axis at -k * 2 * h*sqrt(2) -- the meeting corner of
        # cube k-1 and cube k -- with the hinge axis along world z.
        # Keep the handles around so callers can later read the
        # per-component reaction wrenches via gather_constraint_wrenches.
        self.hinge_handles = []
        for k in range(NUM_HINGES):
            body_a = world_body if k == 0 else cube_ids[k - 1]
            body_b = cube_ids[k]
            anchor = (0.0, -k * 2.0 * _DIAGONAL_HALF, 0.0)
            self.hinge_handles.append(
                b.add_hinge_joint(
                    body1=body_a,
                    body2=body_b,
                    hinge_center=anchor,
                    hinge_axis=_HINGE_AXIS,
                    motor=True,
                    target_velocity=0.0,
                    max_force=_MOTOR_MAX_FORCE,
                )
            )

        # substeps=1 here because we drive substepping ourselves from
        # simulate() (matches example_body_chain).
        self.world = b.finalize(
            substeps=1,
            solver_iterations=8,
            device=self.device,
        )

        # ---- Rendering scratch ---------------------------------------
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
        # After the run the chain should still be hanging from the world
        # anchor (no body fell unreasonably far, no NaNs).
        positions = self.world.bodies.position.numpy()
        for i in range(1, NUM_BODIES):
            assert np.isfinite(positions[i]).all(), f"body {i} produced non-finite position"
            assert positions[i, 1] > -10.0 * NUM_CUBES, (
                f"body {i} fell unreasonably far ({positions[i, 1]})"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
