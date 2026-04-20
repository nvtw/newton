# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 13 -- Motor and Limit
#
# Adapted from ``JitterDemo.Demos.Demo13`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo13.cs``). The C#
# demo has three sub-scenes; we ship two of them:
#
#   Scene A -- motorised hinge with a coupled second hinge.
#       Two thin planks lying along +X float at height 4 m. A motorised
#       revolute hinge at b0's COM (world <-> b0) spins about +X at
#       ``MOTOR_TARGET_VELOCITY`` rad/s with a ``MOTOR_MAX_TORQUE`` N*m
#       torque cap; a second (unactuated) revolute along the same axis
#       couples b0 and b1 so they swing together. The C# reference
#       glues the two planks with a ``UniversalJoint``; we approximate
#       it with a revolute (the resulting DoF count is the same: a
#       single free axial rotation shared between the pair).
#
#   Scene B -- hinge with a +-120 deg angular limit.
#       A single thin plank ("long box") is pinned to the world by a
#       revolute about +Z at the plank's initial COM. The joint's
#       ``min_value`` / ``max_value`` clamp the rotation to
#       ``[-120, 120]`` degrees. Gravity drives the plank back and
#       forth; the limits prevent it from spinning past the clamp.
#
# Sub-scene 3 in the C# source is a pair of free-spinning wheels
# coupled by ``TwistAngle`` so they rotate in lock-step. The coupling
# is a relative-twist constraint we do not ship; two *uncoupled*
# revolute wheels would not reproduce the intended behaviour (they'd
# just free-fall together), so that scene is intentionally omitted.
#
# Run:  python -m newton._src.solvers.jitter.example_demo_13
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.example_jitter_common import (
    WORLD_BODY,
    DemoConfig,
    DemoExample,
)
from newton._src.solvers.jitter.world_builder import DriveMode, JointMode

# ---------------------------------------------------------------------------
# Scene A (motorised hinge + coupling)
# ---------------------------------------------------------------------------

PLANK_HX = 1.0
PLANK_HY = 0.05
PLANK_HZ = 0.05
HINGE_Z = 4.0
HALF_GAP = 1.1  # half-step along +X between the two plank centres

MOTOR_TARGET_VELOCITY = 4.0  # [rad/s]
MOTOR_MAX_TORQUE = 1.0  # [N*m]

# ---------------------------------------------------------------------------
# Scene B (hinge with +-120 deg limit)
# ---------------------------------------------------------------------------

# Scene B is placed 5 m along +Y of scene A so both are visible from the
# default camera (matches the C# ``x = -5`` offset after Y<->Z swap).
LIMIT_PLANK_CENTER = (0.0, -5.0, 3.0)
# The C# source uses BoxShape(2, 0.1, 3): the 2 m axis lies along +X,
# the 0.1 m axis along +Y, and the 3 m axis along +Z (hinge axis).
LIMIT_PLANK_HX = 1.0
LIMIT_PLANK_HY = 0.05
LIMIT_PLANK_HZ = 1.5
LIMIT_ANGLE_RAD = math.radians(120.0)


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Motor and Limit",
            camera_pos=(7.0, -2.0, 6.0),
            camera_pitch=-20.0,
            camera_yaw=-60.0,
            fps=60,
            substeps=3,
            solver_iterations=8,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._build_scene_a()
        self._build_scene_b()

    # ------------------------------------------------------------------
    # Scene A -- motorised hinge + passive coupling
    # ------------------------------------------------------------------

    def _build_scene_a(self) -> None:
        mb = self.model_builder
        self._b0 = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(-HALF_GAP, 0.0, HINGE_Z), q=wp.quat_identity()
            ),
            mass=1.0,
        )
        mb.add_shape_box(self._b0, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
        self.register_body_extent(self._b0, (PLANK_HX, PLANK_HY, PLANK_HZ))

        self._b1 = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(HALF_GAP, 0.0, HINGE_Z), q=wp.quat_identity()
            ),
            mass=1.0,
        )
        mb.add_shape_box(self._b1, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
        self.register_body_extent(self._b1, (PLANK_HX, PLANK_HY, PLANK_HZ))

        # Motorised hinge: world <-> b0 about +X, velocity-drive at
        # +MOTOR_TARGET_VELOCITY rad/s, torque cap MOTOR_MAX_TORQUE.
        self.add_joint(
            body1=WORLD_BODY,
            body2=self._b0,
            anchor1=(-HALF_GAP, 0.0, HINGE_Z),
            anchor2=(-HALF_GAP + 1.0, 0.0, HINGE_Z),  # +X axis
            mode=JointMode.REVOLUTE,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=MOTOR_TARGET_VELOCITY,
            max_force_drive=MOTOR_MAX_TORQUE,
        )
        # Coupling revolute along the same +X axis between b0 and b1.
        self.add_joint(
            body1=self._b0,
            body2=self._b1,
            anchor1=(0.0, 0.0, HINGE_Z),
            anchor2=(1.0, 0.0, HINGE_Z),
            mode=JointMode.REVOLUTE,
        )

    # ------------------------------------------------------------------
    # Scene B -- hinge with +-120 deg angular limit
    # ------------------------------------------------------------------

    def _build_scene_b(self) -> None:
        mb = self.model_builder
        cx, cy, cz = LIMIT_PLANK_CENTER

        # Rotate the plank 90 deg about +Z so the 3 m axis (its long
        # axis in the C# BoxShape) lies along +Y at rest -- that way
        # gravity has a lever arm about the +Z hinge axis and the limit
        # actually engages.
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)

        self._b_limit = mb.add_body(
            xform=wp.transform(p=wp.vec3(cx, cy, cz), q=q),
            mass=1.0,
        )
        mb.add_shape_box(
            self._b_limit,
            hx=LIMIT_PLANK_HX,
            hy=LIMIT_PLANK_HY,
            hz=LIMIT_PLANK_HZ,
        )
        self.register_body_extent(
            self._b_limit, (LIMIT_PLANK_HX, LIMIT_PLANK_HY, LIMIT_PLANK_HZ)
        )

        # Revolute about +Z at the plank's COM with +-120 deg limits.
        self.add_joint(
            body1=WORLD_BODY,
            body2=self._b_limit,
            anchor1=(cx, cy, cz),
            anchor2=(cx, cy, cz + 1.0),  # +Z axis
            mode=JointMode.REVOLUTE,
            min_value=-LIMIT_ANGLE_RAD,
            max_value=LIMIT_ANGLE_RAD,
        )

    # ------------------------------------------------------------------
    # Asserts
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """Both sub-scenes must stay pinned to their hinge anchors --
        that's the tightest invariant the unified joint guarantees.

        * Scene A: motorised plank stays within a small envelope of its
          hinge anchor at ``(-HALF_GAP, 0, HINGE_Z)`` (the motor spins
          it, so we don't constrain its orientation).
        * Scene B: limited plank stays within its hinge anchor envelope
          at ``LIMIT_PLANK_CENTER`` -- the +-120 deg limit keeps it
          inside a cone but the COM should not drift.
        """
        for body, anchor, tol in (
            (self._b0, (-HALF_GAP, 0.0, HINGE_Z), 0.2),
            (self._b_limit, LIMIT_PLANK_CENTER, 0.2),
        ):
            pos, vel = self.jitter_body_state(body)
            assert np.isfinite(pos).all() and np.isfinite(vel).all()
            offset = float(np.linalg.norm(pos - np.asarray(anchor, dtype=np.float32)))
            assert offset < tol, (
                f"body {body} drifted {offset:.2f} m from its hinge anchor {anchor}"
            )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
