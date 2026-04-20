# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Motorized Prismatic Chain
#
# A vertical column of ``NUM_CUBES`` cubes joined head-to-tail by
# prismatic (slider) joints. Each slider's free DoF is translation along
# world ``+y``, so the stack behaves like a set of concentric telescopic
# sections. The top joint anchors cube 0 to the static world; every
# other joint connects cube ``k-1`` to cube ``k``.
#
# The chain joint type is chosen by the module-level ``JOINT_KIND``
# (see :class:`JointKind` below) so the same scene can be used to
# compare three implementations side by side:
#
#   * :attr:`JointKind.ACTUATED_DOUBLE_BALL_SOCKET` -- unified
#     revolute/prismatic constraint in :attr:`JointMode.PRISMATIC`
#     (``constraint_actuated_double_ball_socket``). Pure point-matching
#     2+2+1 positional lock plus an optional scalar PGS row driving the
#     free axial translation with a soft position / velocity drive and
#     optional one-sided spring-damper limits. The *only* kind that
#     exposes ``TARGET_POSITION`` / ``_DRIVE_MAX_FORCE`` to the solver.
#   * :attr:`JointKind.PRISMATIC`                  -- legacy standalone
#     ``constraint_prismatic``. 3 rotational rows (quaternion-error)
#     + 2 tangent-translation rows (point-matching), solved as a
#     3x3 + 2x2 Schur. No drive or limit -- pure passive slider, so
#     under gravity the chain extends freely along +y.
#   * :attr:`JointKind.D6_PRISMATIC`               -- generalised D6 with
#     all 3 angular axes locked, linear x/z locked, and linear y free
#     (or driven, if ``TARGET_POSITION != 0``). Internally dispatches
#     to Jolt-style 1-DoF axis constraints with an optional implicit-PD
#     position drive on the free linear axis.
#
# All three modes lock the same 5 DoF and free translation along +y;
# they just reach that constraint with different solver structures
# and different feature sets.
#
# Run: ``python -m newton._src.solvers.jitter.example_motorized_prismatic_chain``
###########################################################################

import enum

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel
from newton._src.solvers.jitter.world_builder import (
    D6AxisDrive,
    DriveMode,
    JointMode,
    WorldBuilder,
)


class JointKind(enum.Enum):
    """Which kind of "slider between two cubes" the chain is built with.

    All three modes lock the same 5 DoF (3 rotational, 2 translational
    perpendicular to the slide axis) and leave translation along the
    slide axis free, but they reach that constraint through different
    solver structures and feature sets:

    * :attr:`ACTUATED_DOUBLE_BALL_SOCKET` -- unified revolute/prismatic
      constraint in :attr:`JointMode.PRISMATIC` (default). Pure
      point-matching 2+2+1 positional lock plus an optional scalar
      PGS row driving the free axial translation with a soft
      position / velocity drive and optional one-sided spring-damper
      limits. Exposes ``TARGET_POSITION`` / ``_DRIVE_MAX_FORCE``.
    * :attr:`PRISMATIC`                   -- legacy standalone slider
      constraint (``constraint_prismatic``). 3 rotational rows
      (quaternion-error) + 2 tangent-translation rows (point-matching),
      solved as a 3x3 + 2x2 Schur complement. Passive: no drive, no
      limits -- under gravity the chain extends freely along +y.
    * :attr:`D6_PRISMATIC`                -- generalised D6 with all
      angular axes locked, linear x/z locked, linear y free (or driven
      if ``TARGET_POSITION != 0``). Internally dispatches to Jolt-style
      1-DoF axis constraints with an optional implicit-PD position
      drive on the free linear axis.
    """

    ACTUATED_DOUBLE_BALL_SOCKET = "actuated_double_ball_socket"
    PRISMATIC = "prismatic"
    D6_PRISMATIC = "d6_prismatic"


# Selects which constraint type every joint in the chain is built
# with. Switch this to compare solver behaviour / robustness across
# the three implementations of "slider between two cubes".
JOINT_KIND = JointKind.ACTUATED_DOUBLE_BALL_SOCKET

# Selects how the drive on the free linear axis behaves. Honoured by
# :attr:`JointKind.ACTUATED_DOUBLE_BALL_SOCKET` and
# :attr:`JointKind.D6_PRISMATIC`; the passive :attr:`JointKind.PRISMATIC`
# legacy kind ignores this (no drive available).
#
#   * :attr:`DriveMode.POSITION` (default) -- soft-spring position drive;
#     pulls each joint's axial displacement towards ``TARGET_POSITION``
#     metres with a critically damped spring. This is the only mode
#     that can *hold* the chain against gravity along a vertical slide
#     axis; velocity/OFF drives let the stack fall freely since they
#     produce no restoring force at zero velocity.
#   * :attr:`DriveMode.VELOCITY` -- soft-damper velocity drive; pushes
#     the axial rate towards ``TARGET_VELOCITY`` [m/s]. With the slide
#     axis along gravity this only slows the fall -- it does not stop
#     it -- because a pure damper has no spring term.
#   * :attr:`DriveMode.OFF` -- passive slider; indistinguishable from
#     :attr:`JointKind.PRISMATIC` under gravity (the chain free-falls
#     along the slide axis).
DRIVE_MODE = DriveMode.POSITION

NUM_CUBES = 6
HALF_EXTENT = 0.5
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor body
NUM_JOINTS = NUM_CUBES

_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Slide axis: world +y. For the ACTUATED_DOUBLE_BALL_SOCKET path the
# second anchor is one metre away from the first (rest_length = 1 m)
# so the unit-axis convention recommended by :class:`JointDescriptor`
# applies: any value of ``target`` is measured as "metres of
# displacement along the slide axis". The legacy PRISMATIC and
# D6_PRISMATIC paths use the same world-space axis (they take an
# ``axis`` vector directly rather than two anchor points).
_SLIDE_AXIS = (0.0, 1.0, 0.0)

# Spacing between consecutive cubes at rest [m]. 2*HALF_EXTENT stacks
# them edge-to-edge.
_REST_SPACING = 2.0 * HALF_EXTENT

# Peak linear force the drive may apply per substep [N]. Generous so
# the chain converges to ``TARGET_POSITION`` under gravity without
# explosion. Ignored by the passive-only PRISMATIC legacy kind.
_DRIVE_MAX_FORCE = 200.0

# Soft-drive knobs. ``hertz_drive = 8`` / ``damping_ratio_drive = 2``
# gives a per-joint over-damped position spring at 8 Hz. We pick
# ``damping_ratio > 1`` deliberately: the per-joint critical ratio
# does *not* carry over to the chain's normal modes (see the analysis
# in :mod:`constraint_actuated_double_ball_socket`), so over-damping
# each joint is necessary to keep the collective chain modes from
# oscillating under mouse-picking perturbations and gravity loading.
_HERTZ_DRIVE = 8.0
_DAMPING_RATIO_DRIVE = 2.0

# Per-joint target displacement along +y relative to the joint's
# anchor1 (i.e. body1's end of the slider) [m]. Zero = rest pose; >0
# = extend; <0 = retract. The chain's total extension is
# ``NUM_JOINTS * TARGET_POSITION``. Honoured by :attr:`DriveMode.POSITION`.
TARGET_POSITION = 0.0

# Per-joint target axial velocity [m/s]. Honoured by
# :attr:`DriveMode.VELOCITY`.
#
#   * 0.0 -> velocity damper (slows the chain's collapse under
#     gravity but does not stop it; the chain drifts down at
#     terminal velocity).
#   * >0  -> actively extend at that rate; chain pushes up against
#     gravity until ``_DRIVE_MAX_FORCE`` stalls it.
TARGET_VELOCITY = 0.0


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # ---- Build the Jitter World via WorldBuilder ------------------
        b = WorldBuilder()
        world_body = b.world_body  # body 0, auto-created static anchor

        # Cubes stacked below the world anchor along -y. Identity
        # orientation: no rotation needed because the slide axis is
        # world-aligned, and for D6_PRISMATIC the body-1-local linear
        # axes then coincide with the world axes.
        cube_ids: list[int] = []
        for j in range(NUM_CUBES):
            cube_ids.append(
                b.add_dynamic_body(
                    position=(0.0, -(j + 1) * _REST_SPACING, 0.0),
                    inverse_mass=1.0,
                    inverse_inertia=_INV_INERTIA,
                )
            )

        # D6 axis presets, hoisted out of the per-joint loop so the
        # tuples are built once and reused across all NUM_JOINTS joints.
        # Linear free axis is the body-1-local +y (which under the
        # identity cube orientation coincides with world +y). The
        # soft-spring drive on that axis is selected by ``DRIVE_MODE``
        # and capped at ``_DRIVE_MAX_FORCE`` (Jolt-style 1-DoF linear
        # ``MotorConstraintPart`` + ``SpringPart``).
        d6_lock = D6AxisDrive()
        if DRIVE_MODE is DriveMode.POSITION:
            d6_linear_y = D6AxisDrive(
                hertz=_HERTZ_DRIVE,
                damping_ratio=_DAMPING_RATIO_DRIVE,
                target_position=TARGET_POSITION,
                max_force=_DRIVE_MAX_FORCE,
            )
        elif DRIVE_MODE is DriveMode.VELOCITY:
            d6_linear_y = D6AxisDrive(
                hertz=_HERTZ_DRIVE,
                damping_ratio=_DAMPING_RATIO_DRIVE,
                target_velocity=TARGET_VELOCITY,
                max_force=_DRIVE_MAX_FORCE,
            )
        else:  # DriveMode.OFF
            d6_linear_y = D6AxisDrive(max_force=0.0)
        d6_angular = (d6_lock, d6_lock, d6_lock)
        d6_linear = (d6_lock, d6_linear_y, d6_lock)

        # One prismatic joint per junction. Joint k connects body_a
        # (world anchor for k=0, else cube k-1) to cube k. The
        # ``anchor`` used by PRISMATIC / D6_PRISMATIC is the midpoint
        # between body_a and body_b at rest; the unified variant
        # additionally takes a second anchor one metre along +y so the
        # slide axis is implicit in the two points.
        self.joint_handles = []
        for k in range(NUM_JOINTS):
            body_a = world_body if k == 0 else cube_ids[k - 1]
            body_b = cube_ids[k]
            y_mid = -(k + 0.5) * _REST_SPACING if k > 0 else -0.5 * _REST_SPACING
            anchor = (0.0, y_mid, 0.0)
            if JOINT_KIND is JointKind.ACTUATED_DOUBLE_BALL_SOCKET:
                anchor2 = (
                    anchor[0] + _SLIDE_AXIS[0],
                    anchor[1] + _SLIDE_AXIS[1],
                    anchor[2] + _SLIDE_AXIS[2],
                )
                self.joint_handles.append(
                    b.add_joint(
                        body1=body_a,
                        body2=body_b,
                        anchor1=anchor,
                        anchor2=anchor2,
                        mode=JointMode.PRISMATIC,
                        drive_mode=DRIVE_MODE,
                        target=TARGET_POSITION,
                        target_velocity=TARGET_VELOCITY,
                        max_force_drive=_DRIVE_MAX_FORCE,
                        hertz_drive=_HERTZ_DRIVE,
                        damping_ratio_drive=_DAMPING_RATIO_DRIVE,
                    )
                )
            elif JOINT_KIND is JointKind.PRISMATIC:
                # Legacy standalone slider. No drive / limits -- this is
                # a pure passive 5-DoF lock, so under gravity the chain
                # extends freely along +y.
                self.joint_handles.append(
                    b.add_prismatic(
                        body1=body_a,
                        body2=body_b,
                        anchor=anchor,
                        axis=_SLIDE_AXIS,
                    )
                )
            elif JOINT_KIND is JointKind.D6_PRISMATIC:
                self.joint_handles.append(
                    b.add_d6(
                        body1=body_a,
                        body2=body_b,
                        anchor=anchor,
                        angular=d6_angular,
                        linear=d6_linear,
                    )
                )
            else:
                raise ValueError(f"unknown JOINT_KIND: {JOINT_KIND}")

        self.world = b.finalize(
            substeps=self.sim_substeps,
            solver_iterations=8,
            device=self.device,
        )

        # ---- Rendering scratch ---------------------------------------
        self._xforms = wp.zeros(NUM_BODIES, dtype=wp.transform, device=self.device)

        # ---- Picking --------------------------------------------------
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
        self.world.step(self.frame_dt, picking=self.picking)

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
        self.viewer.log_shapes(
            "/world/cubes",
            newton.GeoType.BOX,
            (HALF_EXTENT, HALF_EXTENT, HALF_EXTENT),
            self._xforms[1:],
        )
        self.viewer.end_frame()

    def test_final(self):
        # Sanity: no cube flew off to infinity and no NaNs. Under a
        # passive PRISMATIC joint (no drive) the chain slides freely
        # under gravity, so the "floor" bound below accommodates the
        # unbounded fall of ``0.5 * g * t^2`` over the default
        # ``frames`` budget of the test runner.
        positions = self.world.bodies.position.numpy()
        max_fall = 0.5 * 9.81 * (NUM_CUBES * self.frame_dt * 100.0) ** 2
        floor = -(NUM_CUBES * _REST_SPACING + max_fall + 10.0)
        # Guard against numerical blowup (absurd overflow in addition
        # to the free-fall envelope). ``math.isfinite`` catches NaN /
        # +-inf, the ``> floor`` check catches slower drifts.
        for i in range(1, NUM_BODIES):
            assert np.isfinite(positions[i]).all(), (
                f"body {i} produced non-finite position"
            )
            assert positions[i, 1] > floor, (
                f"body {i} fell below prismatic-chain floor "
                f"({positions[i, 1]} < {floor})"
            )
            assert abs(positions[i, 0]) < 1.0, (
                f"body {i} drifted laterally along x ({positions[i, 0]})"
            )
            assert abs(positions[i, 2]) < 1.0, (
                f"body {i} drifted laterally along z ({positions[i, 2]})"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
