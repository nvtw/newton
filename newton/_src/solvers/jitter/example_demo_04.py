# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 04 -- Ragdolls (simplified)
#
# Adapted from ``JitterDemo.Demos.Demo04`` (which itself builds on
# ``Common.BuildRagdoll`` in
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Common.cs``). The C#
# ragdoll uses three Jitter constraint types we don't ship:
#
#   * ``TwistAngle``     -- 1-DoF relative-twist limit (hip / shoulder).
#   * ``ConeLimit``      -- 1-DoF cone-half-angle limit (neck / hip).
#   * ``HingeJoint``     -- fused 3-sub-constraint hinge (knee / elbow).
#
# Our Jitter port exposes a unified :class:`JointMode` (ball-socket,
# revolute, prismatic) with optional one-sided angle limits on the
# free DoF. ``HingeJoint`` can be reproduced exactly as
# ``JointMode.REVOLUTE`` with ``min_value`` / ``max_value``;
# ``TwistAngle`` and ``ConeLimit`` have no direct equivalent. We
# therefore trade the twist + cone limits for plain ball-sockets at
# the shoulders / hips / neck. The result is a looser ragdoll than
# Jitter's -- limbs can twist further and splay more -- but it still
# produces the canonical "pile of floppy bodies" behaviour that makes
# this demo useful as a stress test for the solver's contact +
# articulation interaction.
#
# Built-in collision filtering between limbs (Jitter's
# ``IgnoreCollisionBetweenFilter``) is likewise not available in the
# Newton ``CollisionPipeline`` we use, so limb pairs that the C# demo
# masks out still interact through contacts. The ball-socket pins are
# strong enough to keep adjacent parts from pushing each other apart,
# but you will see more contact activity between e.g. torso and thigh
# than the C# reference.
#
# Layout -- a 2x2 grid of four ragdolls stacked to fall into each
# other (reduced from 100 in C# to keep the contact counts manageable
# on default budgets; tweak ``GRID_X`` / ``GRID_Y`` / ``STACK_COUNT``
# to match the original 100 if desired).
#
# Run:  python -m newton._src.solvers.jitter.example_demo_04
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.example_jitter_common import (
    DemoConfig,
    DemoExample,
)
from newton._src.solvers.jitter.world_builder import JointMode

# ---------------------------------------------------------------------------
# Coordinate system:
#   Jitter is +Y-up, Newton is +Z-up. Positions (x, y, z)_C# map to
#   (x, z, y)_Newton. Rotations about Jitter +Z (horizontal) map to
#   rotations about Newton +Y (also horizontal). Capsules are oriented
#   along +Y in Jitter and along +Z in Newton, so the axis swap keeps
#   them pointing "up" without any extra rotation.
# ---------------------------------------------------------------------------

# Torso: C# BoxShape(0.35, 0.6, 0.2) -> full sizes (sx, sy, sz). Newton
# half-extents after the axis swap: HX = 0.175, HY = 0.1, HZ = 0.3.
TORSO_HX, TORSO_HY, TORSO_HZ = 0.175, 0.1, 0.3

HEAD_RADIUS = 0.15
# Capsule radii + half-lengths (Jitter CapsuleShape(radius, length)
# with length = 2 * half_height in Newton).
UPPER_LEG_R, UPPER_LEG_HH = 0.08, 0.15
LOWER_LEG_R, LOWER_LEG_HH = 0.08, 0.15
UPPER_ARM_R, UPPER_ARM_HH = 0.07, 0.10
LOWER_ARM_R, LOWER_ARM_HH = 0.06, 0.10

# Grid of ragdolls. A single ragdoll's extreme vertical span is ~1.35 m
# (head at 0 down to lower-leg tip at ~-1.35 m); we stack them 2 m
# apart vertically so the upper stack has a brief fall before landing
# on the lower one, reproducing the Demo04 visual of ragdolls raining
# onto each other.
GRID_X = 2
GRID_Y = 2
STACK_COUNT = 2
GRID_SPACING = 3.0  # [m] between ragdoll roots along +X / +Y
STACK_SPACING = 2.5  # [m] between stacked ragdoll roots along +Z
BASE_Z = 4.0  # head height of the lowest ragdoll [m]

KNEE_MIN_RAD = math.radians(-120.0)
KNEE_MAX_RAD = 0.0
ELBOW_LEFT_MIN_RAD = math.radians(-160.0)
ELBOW_LEFT_MAX_RAD = 0.0
ELBOW_RIGHT_MIN_RAD = 0.0
ELBOW_RIGHT_MAX_RAD = math.radians(160.0)


def _xform_newton(x_c: float, y_c: float, z_c: float, q=None) -> wp.transform:
    """Pack a C#-convention ``(x, y, z)`` triple into a Newton +Z-up
    world transform. Jitter +Y-up -> Newton +Z-up means the Jitter
    ``y`` (vertical) becomes Newton ``z``, and Jitter ``z``
    (horizontal) becomes Newton ``y``.
    """
    return wp.transform(
        p=wp.vec3(float(x_c), float(z_c), float(y_c)),
        q=wp.quat_identity() if q is None else q,
    )


def _anchor_newton(x_c: float, y_c: float, z_c: float) -> tuple[float, float, float]:
    """Same swap as :func:`_xform_newton` but for hinge / ball-socket
    anchors -- three-tuples straight into :meth:`add_joint`.
    """
    return (float(x_c), float(z_c), float(y_c))


class _Parts:
    """Handle bag for one ragdoll's 10 bodies."""

    __slots__ = (
        "head",
        "torso",
        "upper_leg_left",
        "upper_leg_right",
        "lower_leg_left",
        "lower_leg_right",
        "upper_arm_left",
        "upper_arm_right",
        "lower_arm_left",
        "lower_arm_right",
    )


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Ragdolls (simplified)",
            camera_pos=(4.5, -4.5, 4.5),
            camera_pitch=-15.0,
            camera_yaw=-45.0,
            fps=60,
            # C# Demo04 uses SolverIterations=(8, 4); we keep 4 substeps
            # and 8 iterations (similar total PGS work) to give the
            # heavy ball-socket chain enough time to settle.
            substeps=4,
            solver_iterations=10,
        )
        super().__init__(viewer, args, cfg)
        # All ragdoll handles live here so ``test_final`` can walk them.
        self._ragdolls: list[_Parts] = []
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        for ix in range(GRID_X):
            for iy in range(GRID_Y):
                for iz in range(STACK_COUNT):
                    origin_c = (
                        (ix - (GRID_X - 1) * 0.5) * GRID_SPACING,
                        BASE_Z + iz * STACK_SPACING,  # C# +Y (vertical)
                        (iy - (GRID_Y - 1) * 0.5) * GRID_SPACING,  # C# +Z
                    )
                    self._ragdolls.append(self._build_ragdoll(origin_c))

    # ------------------------------------------------------------------
    # One ragdoll
    # ------------------------------------------------------------------

    def _build_ragdoll(self, origin_c: tuple[float, float, float]) -> _Parts:
        mb = self.model_builder
        ox, oy, oz = origin_c

        p = _Parts()

        # --- Head (sphere) at C# (0, 0, 0) -> Newton (0, 0, 0).
        p.head = mb.add_body(
            xform=_xform_newton(ox + 0.0, oy + 0.0, oz + 0.0),
            mass=1.0,
        )
        mb.add_shape_sphere(p.head, radius=HEAD_RADIUS)
        self.register_body_extent(p.head, (HEAD_RADIUS, HEAD_RADIUS, HEAD_RADIUS))

        # --- Torso box at C# (0, -0.46, 0).
        p.torso = mb.add_body(
            xform=_xform_newton(ox + 0.0, oy - 0.46, oz + 0.0),
            mass=3.0,
        )
        mb.add_shape_box(p.torso, hx=TORSO_HX, hy=TORSO_HY, hz=TORSO_HZ)
        self.register_body_extent(p.torso, (TORSO_HX, TORSO_HY, TORSO_HZ))

        # --- Legs (capsules along +Y in C# -> +Z in Newton, no extra rot).
        p.upper_leg_left = self._add_capsule_body(
            ox + 0.11, oy - 0.85, oz + 0.0, UPPER_LEG_R, UPPER_LEG_HH, mass=1.5
        )
        p.upper_leg_right = self._add_capsule_body(
            ox - 0.11, oy - 0.85, oz + 0.0, UPPER_LEG_R, UPPER_LEG_HH, mass=1.5
        )
        p.lower_leg_left = self._add_capsule_body(
            ox + 0.11, oy - 1.2, oz + 0.0, LOWER_LEG_R, LOWER_LEG_HH, mass=1.0
        )
        p.lower_leg_right = self._add_capsule_body(
            ox - 0.11, oy - 1.2, oz + 0.0, LOWER_LEG_R, LOWER_LEG_HH, mass=1.0
        )

        # --- Arms (rotated +90 deg about C# Z = Newton Y so the capsule
        # axis goes from +Z (Newton default) to +X).
        q_arm = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0)
        p.upper_arm_left = mb.add_body(
            xform=_xform_newton(ox + 0.30, oy - 0.2, oz + 0.0, q_arm),
            mass=0.8,
        )
        mb.add_shape_capsule(
            p.upper_arm_left, radius=UPPER_ARM_R, half_height=UPPER_ARM_HH
        )
        self.register_body_extent(
            p.upper_arm_left, (UPPER_ARM_HH, UPPER_ARM_R, UPPER_ARM_R)
        )
        p.upper_arm_right = mb.add_body(
            xform=_xform_newton(ox - 0.30, oy - 0.2, oz + 0.0, q_arm),
            mass=0.8,
        )
        mb.add_shape_capsule(
            p.upper_arm_right, radius=UPPER_ARM_R, half_height=UPPER_ARM_HH
        )
        self.register_body_extent(
            p.upper_arm_right, (UPPER_ARM_HH, UPPER_ARM_R, UPPER_ARM_R)
        )
        p.lower_arm_left = mb.add_body(
            xform=_xform_newton(ox + 0.55, oy - 0.2, oz + 0.0, q_arm),
            mass=0.6,
        )
        mb.add_shape_capsule(
            p.lower_arm_left, radius=LOWER_ARM_R, half_height=LOWER_ARM_HH
        )
        self.register_body_extent(
            p.lower_arm_left, (LOWER_ARM_HH, LOWER_ARM_R, LOWER_ARM_R)
        )
        p.lower_arm_right = mb.add_body(
            xform=_xform_newton(ox - 0.55, oy - 0.2, oz + 0.0, q_arm),
            mass=0.6,
        )
        mb.add_shape_capsule(
            p.lower_arm_right, radius=LOWER_ARM_R, half_height=LOWER_ARM_HH
        )
        self.register_body_extent(
            p.lower_arm_right, (LOWER_ARM_HH, LOWER_ARM_R, LOWER_ARM_R)
        )

        # --- Joints.
        # Neck: ball-socket head <-> torso at C# (0, -0.15, 0).
        self.add_joint(
            body1=p.head,
            body2=p.torso,
            anchor1=_anchor_newton(ox + 0.0, oy - 0.15, oz + 0.0),
            mode=JointMode.BALL_SOCKET,
        )

        # Hips: ball-socket torso <-> upper_leg_*.
        self.add_joint(
            body1=p.torso,
            body2=p.upper_leg_left,
            anchor1=_anchor_newton(ox + 0.11, oy - 0.7, oz + 0.0),
            mode=JointMode.BALL_SOCKET,
        )
        self.add_joint(
            body1=p.torso,
            body2=p.upper_leg_right,
            anchor1=_anchor_newton(ox - 0.11, oy - 0.7, oz + 0.0),
            mode=JointMode.BALL_SOCKET,
        )

        # Knees: hinge with +-120 deg limit, axis C# +X (unchanged).
        self._add_hinge_with_limit(
            p.upper_leg_left,
            p.lower_leg_left,
            anchor_c=(ox + 0.11, oy - 1.05, oz + 0.0),
            axis_dir=(1.0, 0.0, 0.0),
            lo_rad=KNEE_MIN_RAD,
            hi_rad=KNEE_MAX_RAD,
        )
        self._add_hinge_with_limit(
            p.upper_leg_right,
            p.lower_leg_right,
            anchor_c=(ox - 0.11, oy - 1.05, oz + 0.0),
            axis_dir=(1.0, 0.0, 0.0),
            lo_rad=KNEE_MIN_RAD,
            hi_rad=KNEE_MAX_RAD,
        )

        # Shoulders: ball-socket upper_arm <-> torso at C# (+-0.20, -0.2, 0).
        self.add_joint(
            body1=p.upper_arm_left,
            body2=p.torso,
            anchor1=_anchor_newton(ox + 0.20, oy - 0.2, oz + 0.0),
            mode=JointMode.BALL_SOCKET,
        )
        self.add_joint(
            body1=p.upper_arm_right,
            body2=p.torso,
            anchor1=_anchor_newton(ox - 0.20, oy - 0.2, oz + 0.0),
            mode=JointMode.BALL_SOCKET,
        )

        # Elbows: hinge with limits, axis C# +Y -> Newton +Z.
        self._add_hinge_with_limit(
            p.lower_arm_left,
            p.upper_arm_left,
            anchor_c=(ox + 0.42, oy - 0.2, oz + 0.0),
            axis_dir=(0.0, 0.0, 1.0),
            lo_rad=ELBOW_LEFT_MIN_RAD,
            hi_rad=ELBOW_LEFT_MAX_RAD,
        )
        self._add_hinge_with_limit(
            p.lower_arm_right,
            p.upper_arm_right,
            anchor_c=(ox - 0.42, oy - 0.2, oz + 0.0),
            axis_dir=(0.0, 0.0, 1.0),
            lo_rad=ELBOW_RIGHT_MIN_RAD,
            hi_rad=ELBOW_RIGHT_MAX_RAD,
        )

        return p

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_capsule_body(
        self,
        x_c: float,
        y_c: float,
        z_c: float,
        radius: float,
        half_height: float,
        mass: float,
    ) -> int:
        mb = self.model_builder
        body = mb.add_body(xform=_xform_newton(x_c, y_c, z_c), mass=mass)
        mb.add_shape_capsule(body, radius=radius, half_height=half_height)
        self.register_body_extent(body, (radius, radius, half_height))
        return body

    def _add_hinge_with_limit(
        self,
        body1: int,
        body2: int,
        anchor_c: tuple[float, float, float],
        axis_dir: tuple[float, float, float],
        lo_rad: float,
        hi_rad: float,
    ) -> None:
        """Add a revolute hinge at ``anchor_c`` (C# coordinates) with
        ``axis_dir`` given in *Newton* coordinates.

        ``anchor2`` is synthesised by stepping 1 m along ``axis_dir``
        from the Newton-space ``anchor1`` -- the unified joint just
        needs two collinear points on the hinge axis.
        """
        ax1 = _anchor_newton(*anchor_c)
        ax2 = (
            ax1[0] + axis_dir[0],
            ax1[1] + axis_dir[1],
            ax1[2] + axis_dir[2],
        )
        self.add_joint(
            body1=body1,
            body2=body2,
            anchor1=ax1,
            anchor2=ax2,
            mode=JointMode.REVOLUTE,
            min_value=float(lo_rad),
            max_value=float(hi_rad),
        )

    # ------------------------------------------------------------------
    # Asserts
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """Every part of every ragdoll must still be at a finite pose
        and within a generous bounding box of the spawn volume. The
        loosely-constrained version allows limbs to splay farther than
        the C# reference, so tolerances are wide.
        """
        half_span = (
            GRID_X * GRID_SPACING + 2.0,
            GRID_Y * GRID_SPACING + 2.0,
            STACK_COUNT * STACK_SPACING + BASE_Z + 2.0,
        )
        for idx, parts in enumerate(self._ragdolls):
            for attr in _Parts.__slots__:
                body = getattr(parts, attr)
                pos, vel = self.jitter_body_state(body)
                assert np.isfinite(pos).all() and np.isfinite(vel).all(), (
                    f"ragdoll {idx}.{attr} non-finite state {pos=} {vel=}"
                )
                for axis_i, bound in enumerate(half_span):
                    assert abs(float(pos[axis_i])) < bound + half_span[axis_i], (
                        f"ragdoll {idx}.{attr} axis {axis_i} out of bounds: "
                        f"pos={pos}, bound=+-{bound}"
                    )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
