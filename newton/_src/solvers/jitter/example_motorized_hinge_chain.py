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
# The chain joint type is chosen by the module-level ``JOINT_KIND``
# (see :class:`JointKind` below) so the same scene can be used to
# compare three implementations side by side: the fused motorized
# ``HingeJoint``, the rank-5 fused two-anchor ``DoubleBallSocketHinge``,
# and the generalized ``D6`` solved as a single revolute axis.
#
# Each hinge:
#   * is a *fused* CONSTRAINT_TYPE_HINGE_JOINT -- a single column in the
#     constraint container that packs the HingeAngle (locks the two
#     angular DoFs orthogonal to the axis), the BallSocket (locks the
#     three positional DoFs at the shared corner), and the AngularMotor
#     (drives the relative angular velocity along the axis) into one
#     PGS-thread-owned slot. This converges noticeably better than three
#     separate constraints touching the same body pair because the
#     partitioner colours one fused joint per partition (instead of
#     three) and the body data stays in registers across the three
#     sub-iterations.
#   * uses a hinge axis aligned with the *cube edge* that passes through
#     the joint corner. With the 45-degree-about-z rotation the four
#     vertical body-frame edges (the ones parallel to body z) all stay
#     aligned with world ``+z``, so the world-space hinge axis is just
#     ``(0, 0, 1)``;
#   * carries an *optional actuator* on the free axial spin, driven by
#     the module-level ``DRIVE_MODE``:
#       * :attr:`DriveMode.VELOCITY` (default) -- tracks ``TARGET_VELOCITY``
#         [rad/s]. 0.0 makes the motor fight any relative spin around the
#         hinge axis and act as a strong angular damper, so the chain
#         hangs in equilibrium. Non-zero spins the cubes about their
#         hinges; the chain itself starts swinging because each motor
#         applies a net external torque about world z.
#       * :attr:`DriveMode.POSITION` -- tracks ``TARGET_ANGLE`` [rad]
#         with a critically damped soft-spring (ACTUATED_DOUBLE_BALL_SOCKET
#         and D6_REVOLUTE kinds only). Non-zero twists the chain into a
#         helix.
#       * :attr:`DriveMode.OFF`      -- no motor, free-spin axis.
#
# The picking, viewer, render, and ``test_final`` plumbing all match
# example_body_chain.py so the two examples can be compared side by side.
#
# Run:  python -m newton._src.solvers.jitter.example_motorized_hinge_chain
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
from newton._src.solvers.jitter.world_builder import D6AxisDrive, DriveMode, JointMode, WorldBuilder


class JointKind(enum.Enum):
    """Which kind of "hinge between two cubes" the chain is built with.

    All four modes lock the same 5 DoF (3 positional at the shared
    corner, 2 angular orthogonal to the cube edge) and leave the same
    1 DoF free (relative spin about the cube edge / world +z), but
    they reach that constraint with different solver structures and
    different feature sets:

    * :attr:`HINGE_JOINT`           -- fused
      ``CONSTRAINT_TYPE_HINGE_JOINT`` (BallSocket + HingeAngle +
      AngularMotor in one column). Velocity-only motor
      (``TARGET_VELOCITY`` / ``_MOTOR_MAX_FORCE``); position-drive
      mode is silently ignored.
    * :attr:`DOUBLE_BALL_SOCKET`    -- fused two-anchor ball-socket
      ("rank-5 Schur" hinge, no motor). The hinge axis is implicit
      in the line through the two anchors.
    * :attr:`D6_REVOLUTE`           -- generalized D6 with 5 of the 6
      axes locked and the free axis = body-1-local +z. Internally
      dispatches to a fused 3-DoF point-constraint (linear) plus
      per-axis 1-DoF angle constraints (angular), Jolt-style;
      supports both velocity and position drives on the free axis.
    * :attr:`ACTUATED_DOUBLE_BALL_SOCKET` -- same rank-5 Schur lock as
      :attr:`DOUBLE_BALL_SOCKET` plus a soft scalar PGS row driving the
      free axial twist. Supports both :attr:`DriveMode.VELOCITY`
      (``TARGET_VELOCITY``) and :attr:`DriveMode.POSITION`
      (``TARGET_ANGLE``); see ``DRIVE_MODE`` below.
    """

    HINGE_JOINT = "hinge_joint"
    DOUBLE_BALL_SOCKET = "double_ball_socket"
    D6_REVOLUTE = "d6_revolute"
    ACTUATED_DOUBLE_BALL_SOCKET = "actuated_double_ball_socket"


# Selects which constraint type every joint in the chain is built
# with. Switch this to compare solver behaviour / robustness across
# the four implementations of "hinge between two cubes".
JOINT_KIND = JointKind.DOUBLE_BALL_SOCKET

# Selects how the motor drives the free axial spin. Only honoured by
# the kinds that carry a motor row (``HINGE_JOINT``,
# ``ACTUATED_DOUBLE_BALL_SOCKET``, and ``D6_REVOLUTE`` -- see each
# kind's entry in :class:`JointKind`):
#
#   * :attr:`DriveMode.OFF`      -- no motor, free-spin axis.
#   * :attr:`DriveMode.VELOCITY` -- velocity motor; tracks
#     ``TARGET_VELOCITY`` [rad/s]. Supported by all three actuated
#     kinds.
#   * :attr:`DriveMode.POSITION` -- soft-spring position drive; pulls
#     the axial angle towards ``TARGET_ANGLE`` [rad] with a critically
#     damped spring. Only supported by :attr:`JointKind.ACTUATED_DOUBLE_BALL_SOCKET`
#     and :attr:`JointKind.D6_REVOLUTE`; silently ignored by the other
#     kinds (HingeJoint's AngularMotor sub-block is a direct
#     Jitter2 port and is velocity-only).
#
# With the chain hanging in equilibrium along -y the axial spin is
# initially zero; POSITION with a non-zero ``TARGET_ANGLE`` therefore
# visibly twists the column into a helix.
DRIVE_MODE = DriveMode.VELOCITY

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

# Maximum motor torque [N·m]. Generous so the motor can hold the
# target state (velocity setpoint or position setpoint) even under
# the (small) precession the chain feels from PGS jitter.
_MOTOR_MAX_FORCE = 50.0

# Position-drive soft-spring knobs (honoured only by
# :attr:`DriveMode.POSITION`). The unified ACTUATED_DOUBLE_BALL_SOCKET
# drive is a straight Jitter2 AngularMotor PD: ``tau = kp*(theta -
# theta*) + kd*theta_dot``. For unit-inertia cubes a 4 Hz critically
# damped angular spring (Box2D convention: ``omega = 2*pi*hertz``,
# ``zeta = 1``) maps to
#
#   kp = I * omega^2 = (2*pi*4)^2 ~= 631 N*m/rad
#   kd = 2*I*zeta*omega = 2*(2*pi*4) ~= 50.3 N*m*s/rad
#
# As noted in :mod:`constraint_actuated_double_ball_socket`, the
# chain as a coupled system has its own normal modes and is not
# guaranteed to be critically damped at these per-joint settings.
# The D6_REVOLUTE legacy drive still takes ``hertz`` /
# ``damping_ratio``.
_HERTZ_DRIVE = 4.0
_DAMPING_RATIO_DRIVE = 1.0
_STIFFNESS_DRIVE = (2.0 * math.pi * _HERTZ_DRIVE) ** 2  # kp [N*m/rad] for I=1
_DAMPING_DRIVE = 2.0 * _DAMPING_RATIO_DRIVE * (2.0 * math.pi * _HERTZ_DRIVE)

# User-configurable target relative angular velocity for every motor in
# the chain [rad/s]. Honoured by :attr:`DriveMode.VELOCITY`. Each hinge
# drives the *relative* spin of body 2 vs. body 1 around the hinge
# axis toward this value.
#
#   * 0.0  -> hold pose (default; chain stays in equilibrium).
#   * 0.5  -> mild continuous rotation; cubes precess about the chain.
#   * 5.0  -> aggressive spin; the chain whips around quickly.
#
# Note: even a small nonzero value is felt by *every* hinge in the
# chain, including the one connecting cube 0 to the static world body
# (which acts as an infinite reaction sink), so the chain quickly
# accumulates large angular momentum.
TARGET_VELOCITY = 0.0

# User-configurable target relative axial angle for every hinge
# [rad]. Honoured by :attr:`DriveMode.POSITION`. Each hinge pulls the
# relative spin of body 2 vs. body 1 around the hinge axis towards
# this value with a soft spring of stiffness ``_HERTZ_DRIVE``. Because
# *each* joint in the chain carries the same target, non-zero values
# compound down the chain (body k sits at ``k * TARGET_ANGLE`` world
# angle if the chain is rigid), so the stack visibly coils.
TARGET_ANGLE = 0.0


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
        # Each handle exposes the joint's global cid; pass it to
        # World.gather_constraint_wrenches via wrenches[handle.cid] to
        # read the joint's combined reaction wrench (force + torque on
        # body 2, summed across the BallSocket / HingeAngle /
        # AngularMotor sub-contributions).
        # D6 axis presets, hoisted out of the per-joint loop so the
        # tuples are built once and reused across all NUM_HINGES
        # joints. The free axis is angular[2] = body-1-local +z (the
        # cube edge), which under the _DIAGONAL_QUAT rotation is
        # parallel to world +z and so matches _HINGE_AXIS. If the user
        # asked for a motor the free axis becomes a position / velocity
        # drive capped at ``_MOTOR_MAX_FORCE`` (Jolt-style 1-DoF
        # AngleConstraintPart with a soft-spring drive row).
        d6_lock = D6AxisDrive()
        if DRIVE_MODE is DriveMode.POSITION:
            d6_axial = D6AxisDrive(
                hertz=_HERTZ_DRIVE,
                damping_ratio=_DAMPING_RATIO_DRIVE,
                target_position=TARGET_ANGLE,
                max_force=_MOTOR_MAX_FORCE,
            )
        elif DRIVE_MODE is DriveMode.VELOCITY:
            d6_axial = D6AxisDrive(
                hertz=_HERTZ_DRIVE,
                damping_ratio=_DAMPING_RATIO_DRIVE,
                target_velocity=TARGET_VELOCITY,
                max_force=_MOTOR_MAX_FORCE,
            )
        else:  # DriveMode.OFF
            d6_axial = D6AxisDrive(max_force=0.0)
        d6_revolute_angular = (d6_lock, d6_lock, d6_axial)
        d6_revolute_linear = (d6_lock, d6_lock, d6_lock)

        self.hinge_handles = []
        for k in range(NUM_HINGES):
            body_a = world_body if k == 0 else cube_ids[k - 1]
            body_b = cube_ids[k]
            anchor = (0.0, -k * 2.0 * _DIAGONAL_HALF, 0.0)
            if JOINT_KIND is JointKind.DOUBLE_BALL_SOCKET:
                # Two anchors along the cube edge (world +z); the implicit
                # hinge axis = anchor2 - anchor1 matches _HINGE_AXIS.
                a1 = (anchor[0], anchor[1], anchor[2] - HALF_EXTENT)
                a2 = (anchor[0], anchor[1], anchor[2] + HALF_EXTENT)
                self.hinge_handles.append(
                    b.add_joint(
                        body1=body_a,
                        body2=body_b,
                        anchor1=a1,
                        anchor2=a2,
                        mode=JointMode.REVOLUTE,
                    )
                )
            elif JOINT_KIND is JointKind.ACTUATED_DOUBLE_BALL_SOCKET:
                # Same two-anchor lock as DOUBLE_BALL_SOCKET, with the
                # axial twist now under either a position drive
                # (``TARGET_ANGLE``) or a velocity drive
                # (``TARGET_VELOCITY``), capped at ``_MOTOR_MAX_FORCE``.
                a1 = (anchor[0], anchor[1], anchor[2] - HALF_EXTENT)
                a2 = (anchor[0], anchor[1], anchor[2] + HALF_EXTENT)
                self.hinge_handles.append(
                    b.add_joint(
                        body1=body_a,
                        body2=body_b,
                        anchor1=a1,
                        anchor2=a2,
                        mode=JointMode.REVOLUTE,
                        drive_mode=DRIVE_MODE,
                        target=TARGET_ANGLE,
                        target_velocity=TARGET_VELOCITY,
                        max_force_drive=_MOTOR_MAX_FORCE,
                        stiffness_drive=_STIFFNESS_DRIVE,
                        damping_drive=_DAMPING_DRIVE,
                    )
                )
            elif JOINT_KIND is JointKind.D6_REVOLUTE:
                self.hinge_handles.append(
                    b.add_d6(
                        body1=body_a,
                        body2=body_b,
                        anchor=anchor,
                        angular=d6_revolute_angular,
                        linear=d6_revolute_linear,
                    )
                )
            elif JOINT_KIND is JointKind.HINGE_JOINT:
                # Fused HingeJoint's AngularMotor sub-block is a direct
                # Jitter2 port (velocity-only). ``DriveMode.POSITION``
                # is silently downgraded to a free-spin axis here; use
                # :attr:`JointKind.ACTUATED_DOUBLE_BALL_SOCKET` or
                # :attr:`JointKind.D6_REVOLUTE` for position-PD drives.
                has_motor = DRIVE_MODE is DriveMode.VELOCITY
                self.hinge_handles.append(
                    b.add_hinge_joint(
                        body1=body_a,
                        body2=body_b,
                        hinge_center=anchor,
                        hinge_axis=_HINGE_AXIS,
                        motor=has_motor,
                        target_velocity=TARGET_VELOCITY,
                        max_force=_MOTOR_MAX_FORCE,
                    )
                )
            else:
                raise ValueError(f"unknown JOINT_KIND: {JOINT_KIND}")

        # Substepping lives inside ``World.step`` now, so pass the
        # substep count through and let the solver's step driver apply
        # picking once per internal substep. Coloring runs once per
        # full step and is reused across substeps + PGS iterations.
        self.world = b.finalize(
            substeps=self.sim_substeps,
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
