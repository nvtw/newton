# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side builder that assembles a :class:`World` from plain Python.

Construction is split in two phases:

1. *Append* phase -- ``add_static_body`` / ``add_dynamic_body`` /
   ``add_ball_socket`` / ... record per-body / per-constraint
   *descriptors* in plain Python lists. Nothing touches the GPU.
2. *Finalize* phase -- :meth:`WorldBuilder.finalize` allocates a
   :class:`BodyContainer` and a :class:`ConstraintContainer` sized
   exactly to the appended counts, packs every descriptor into the SoA
   body arrays via NumPy, computes the per-constraint-type cid ranges,
   uploads per-type batch arrays, and launches one initialise kernel
   per type to fill the column-major constraint storage. Returns a
   ready-to-step :class:`World`.

This keeps the user-facing API trivial (regular Python objects, no
Warp types until the very end) while still hitting the GPU in batched
launches at finalize time.

Body 0 is always the static world body (created in ``__init__``); user
descriptor indices start at 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.jitter.constraints.constraint_actuated_double_ball_socket import (
    ADBS_DWORDS,
    DRIVE_MODE_OFF,
    DRIVE_MODE_POSITION,
    DRIVE_MODE_VELOCITY,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    actuated_double_ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_angular_limit import (
    AL_DWORDS,
    angular_limit_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_angular_motor import (
    AM_DWORDS,
    angular_motor_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_ball_socket import (
    BS_DWORDS,
    ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_contact import CONTACT_DWORDS
from newton._src.solvers.jitter.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_ANGULAR,
    DEFAULT_HERTZ_LIMIT,
    DEFAULT_HERTZ_LINEAR,
    DEFAULT_HERTZ_MOTOR,
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.jitter.constraints.constraint_d6 import (
    D6_DWORDS,
    d6_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_double_ball_socket import (
    DBS_DWORDS,
    double_ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_double_ball_socket_prismatic import (
    DBS_PRISMATIC_DWORDS,
    double_ball_socket_prismatic_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_linear_limit import (
    LL_DWORDS,
    linear_limit_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_linear_motor import (
    LM_DWORDS,
    linear_motor_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_hinge_angle import (
    HA_DWORDS,
    hinge_angle_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_hinge_joint import (
    HJ_DWORDS,
    hinge_joint_initialize_kernel,
)
from newton._src.solvers.jitter.constraints.constraint_prismatic import (
    PR_DWORDS,
    prismatic_initialize_kernel,
)
from newton._src.solvers.jitter.solver_jitter import World

__all__ = [
    "AngularLimitDescriptor",
    "AngularMotorDescriptor",
    "BallSocketDescriptor",
    "D6AxisDrive",
    "D6Descriptor",
    "D6Handle",
    "DoubleBallSocketDescriptor",
    "DoubleBallSocketPrismaticDescriptor",
    "DriveMode",
    "HingeAngleDescriptor",
    "HingeJointDescriptor",
    "HingeJointHandle",
    "JointDescriptor",
    "JointHandle",
    "JointMode",
    "LinearLimitDescriptor",
    "LinearMotorDescriptor",
    "PrismaticDescriptor",
    "PrismaticHandle",
    "RigidBodyDescriptor",
    "WorldBuilder",
]


_IDENTITY_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# Validation tolerances used by :func:`_validate_body_descriptor`. These
# are intentionally *loose* -- the goal is to flag obviously-broken
# descriptors (NaN, sign errors, static-body masses), not to quibble
# about the last bit of float precision.
_QUAT_NORM_TOL = 1e-3
_INERTIA_SYMMETRY_TOL = 1e-5
_INERTIA_POSDEF_TOL = -1e-8  # diag values slightly below 0 are OK (float noise)


def _is_finite(x: float) -> bool:
    """Return True iff ``x`` is a finite float (no NaN / +/-inf)."""
    return math.isfinite(float(x))


def _all_finite(seq) -> bool:
    """Recursive finite-check for scalars / tuples / nested tuples."""
    if isinstance(seq, (tuple, list)):
        return all(_all_finite(v) for v in seq)
    return _is_finite(seq)


def _validate_body_descriptor(desc: "RigidBodyDescriptor", body_index: int) -> None:
    """Reject obviously-broken :class:`RigidBodyDescriptor`s.

    The historical failure mode this guards against: a caller silently
    hands in a ``1/mass`` that disagrees with what they *think* the
    body's mass is (e.g. Newton's ``ModelBuilder`` adding density-derived
    shape mass on top of an explicit ``add_body(mass=...)`` argument).
    Rather than let a bogus body slip onto the GPU and produce mystery
    scale factors in later contact-force checks, we raise early with a
    pointed error message identifying the offending body index and field.

    Rules (matches Jitter2's RigidBody invariants):
        - All numeric fields must be finite (no NaN / inf).
        - ``inverse_mass >= 0``.
        - ``inverse_inertia`` diagonal entries must be >= 0 (within a
          float-noise tolerance) and the matrix must be symmetric.
        - ``motion_type == DYNAMIC`` requires a strictly positive
          ``inverse_mass`` AND at least one strictly positive diagonal
          inertia term (otherwise the body can't respond to forces and
          torques, which is surely not what the caller wants -- they
          should have used ``add_static_body``/``add_kinematic_body``).
        - ``motion_type == STATIC`` requires ``inverse_mass == 0`` and
          a fully-zero ``inverse_inertia`` (non-zero values here are
          silently ignored on the GPU -- fail loudly instead).
        - Orientation quaternion must be (near-)unit length so the
          solver's rotation pipeline stays well-defined.
        - ``linear_damping`` / ``angular_damping`` in [0, 1] (Jitter2's
          Update multiplies velocity by these every frame; values
          outside that band either do nothing or blow up).
    """
    # Finite-check everything first so downstream checks don't have to
    # second-guess NaN-polluted comparisons.
    for name, val in (
        ("position", desc.position),
        ("orientation", desc.orientation),
        ("inverse_mass", desc.inverse_mass),
        ("inverse_inertia", desc.inverse_inertia),
        ("linear_damping", desc.linear_damping),
        ("angular_damping", desc.angular_damping),
        ("velocity", desc.velocity),
        ("angular_velocity", desc.angular_velocity),
    ):
        if not _all_finite(val):
            raise ValueError(
                f"add_body(body_index={body_index}): non-finite value in "
                f"``{name}`` ({val!r}). NaN/inf fields are rejected to "
                "prevent silent solver corruption."
            )

    motion_type = int(desc.motion_type)
    inv_m = float(desc.inverse_mass)

    if inv_m < 0.0:
        raise ValueError(
            f"add_body(body_index={body_index}): ``inverse_mass`` must be "
            f">= 0 (got {inv_m}). Remember this is 1/mass, not mass."
        )

    inv_I = desc.inverse_inertia
    if len(inv_I) != 3 or any(len(row) != 3 for row in inv_I):
        raise ValueError(
            f"add_body(body_index={body_index}): ``inverse_inertia`` must "
            f"be a 3x3 nested tuple (got shape "
            f"{(len(inv_I), len(inv_I[0]) if inv_I else 0)})."
        )
    # Symmetry + non-negative diagonal.
    for i in range(3):
        if float(inv_I[i][i]) < _INERTIA_POSDEF_TOL:
            raise ValueError(
                f"add_body(body_index={body_index}): "
                f"``inverse_inertia[{i}][{i}]`` = {float(inv_I[i][i])} is "
                "negative. Inverse inertia tensor must be positive "
                "semi-definite."
            )
        for j in range(i + 1, 3):
            a = float(inv_I[i][j])
            b = float(inv_I[j][i])
            if abs(a - b) > _INERTIA_SYMMETRY_TOL:
                raise ValueError(
                    f"add_body(body_index={body_index}): "
                    f"``inverse_inertia`` is not symmetric: "
                    f"[{i}][{j}]={a} but [{j}][{i}]={b}. Inertia tensors "
                    "must be symmetric."
                )

    # Quaternion normalisation -- the solver assumes unit quaternions so
    # a malformed one yields garbage rotations with no warning.
    qx, qy, qz, qw = (
        float(desc.orientation[0]),
        float(desc.orientation[1]),
        float(desc.orientation[2]),
        float(desc.orientation[3]),
    )
    qn2 = qx * qx + qy * qy + qz * qz + qw * qw
    if abs(qn2 - 1.0) > _QUAT_NORM_TOL:
        raise ValueError(
            f"add_body(body_index={body_index}): ``orientation`` quaternion "
            f"has norm^2 = {qn2:.6f}, expected 1. Pass a normalized "
            "quaternion (xyzw)."
        )

    # Damping scalars are dimensionless per-frame multipliers in Jitter2
    # (velocity *= (1 - damping*dt) each step). Outside [0, 1] this is
    # either a no-op sign error or an explosion.
    for name, val in (
        ("linear_damping", desc.linear_damping),
        ("angular_damping", desc.angular_damping),
    ):
        f = float(val)
        if f < 0.0 or f > 1.0:
            raise ValueError(
                f"add_body(body_index={body_index}): ``{name}`` = {f} out "
                "of range [0, 1]."
            )

    # Motion-type cross-checks -- the main safety net.
    if motion_type == int(MOTION_DYNAMIC):
        if inv_m == 0.0:
            raise ValueError(
                f"add_body(body_index={body_index}): motion_type=DYNAMIC "
                "requires a strictly positive ``inverse_mass`` (got 0, "
                "i.e. infinite mass). Use ``add_static_body()`` for an "
                "immovable body, or pass the correct mass."
            )
        # At least one axis must have finite inertia, else torques are
        # silently discarded and the body spins freely.
        diag_sum = sum(float(inv_I[i][i]) for i in range(3))
        if diag_sum <= 0.0:
            raise ValueError(
                f"add_body(body_index={body_index}): motion_type=DYNAMIC "
                "requires at least one positive diagonal entry in "
                "``inverse_inertia`` (got all zeros, i.e. infinite "
                "inertia). Use ``add_static_body()`` instead, or set "
                "per-axis inverse inertias."
            )
    elif motion_type == int(MOTION_STATIC):
        if inv_m != 0.0:
            raise ValueError(
                f"add_body(body_index={body_index}): motion_type=STATIC "
                f"but ``inverse_mass`` = {inv_m} (expected 0). Static "
                "bodies are immovable by definition -- did you mean "
                "DYNAMIC?"
            )
        if any(float(inv_I[i][j]) != 0.0 for i in range(3) for j in range(3)):
            raise ValueError(
                f"add_body(body_index={body_index}): motion_type=STATIC "
                "but ``inverse_inertia`` is non-zero. Static bodies must "
                "have zero inverse inertia."
            )
        if any(float(v) != 0.0 for v in desc.velocity) or any(
            float(v) != 0.0 for v in desc.angular_velocity
        ):
            raise ValueError(
                f"add_body(body_index={body_index}): motion_type=STATIC "
                "but ``velocity`` / ``angular_velocity`` is non-zero. "
                "Static bodies never integrate -- use KINEMATIC for a "
                "scripted-velocity body."
            )
    elif motion_type == int(MOTION_KINEMATIC):
        if inv_m != 0.0:
            raise ValueError(
                f"add_body(body_index={body_index}): motion_type=KINEMATIC "
                f"but ``inverse_mass`` = {inv_m} (expected 0). Kinematic "
                "bodies follow user-scripted velocities and must be "
                "treated as infinite-mass by the solver."
            )
    else:
        raise ValueError(
            f"add_body(body_index={body_index}): unknown motion_type "
            f"{motion_type}. Expected one of "
            f"{{STATIC={int(MOTION_STATIC)}, "
            f"KINEMATIC={int(MOTION_KINEMATIC)}, "
            f"DYNAMIC={int(MOTION_DYNAMIC)}}}."
        )


_ZERO_INERTIA = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass
class RigidBodyDescriptor:
    """Plain-Python description of one rigid body.

    Mirrors the writable subset of Jitter2's ``RigidBody`` constructor
    arguments. Defaults produce a *valid static body* sitting at the
    origin with no integration -- i.e. the caller can drop a bare
    ``RigidBodyDescriptor()`` into :meth:`WorldBuilder.add_body` and the
    body-descriptor validator will accept it.

    Mass / inertia fields are intentionally *zero* by default so the
    only way to get a dynamic body is to explicitly supply a non-zero
    ``inverse_mass`` *and* ``inverse_inertia`` *and* flip ``motion_type``
    to :data:`MOTION_DYNAMIC` (or, more readably, go through
    :meth:`WorldBuilder.add_dynamic_body`). This matches Jitter2's
    "opt-in dynamic" convention and prevents the class of bug where a
    silently-infinite-inertia body fails to respond to torques.

    Units (where applicable):
        * ``position`` [m]
        * ``inverse_mass`` [1/kg]
        * ``inverse_inertia`` [1/(kg m^2)] in the *body* frame; the
          solver rotates this into world space every step.
        * ``velocity`` [m/s]
        * ``angular_velocity`` [rad/s]
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    motion_type: int = int(MOTION_STATIC)
    inverse_mass: float = 0.0
    inverse_inertia: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = _ZERO_INERTIA
    linear_damping: float = 1.0
    angular_damping: float = 1.0
    affected_by_gravity: bool = True
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BallSocketDescriptor:
    """Plain-Python description of one ball-and-socket constraint.

    The ``anchor`` is a single point in *world* space; finalize() asks
    each body to express it in its own local frame (mirrors
    ``BallSocket.Initialize``). ``hertz`` and ``damping_ratio`` follow
    the Box2D v3 / Bepu soft-constraint formulation: ``hertz`` is the
    undamped natural frequency [Hz] of the joint as a virtual spring,
    ``damping_ratio`` is non-dimensional (1 = critically damped). See
    :func:`soft_constraint_coefficients`.
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class DoubleBallSocketDescriptor:
    """Plain-Python description of one double-ball-socket hinge lock.

    A pair of ball-and-socket rows sharing a single column that locks
    the relative linear DoFs at *two* points, ``anchor1`` and
    ``anchor2``. Physically this is a 5-DoF hinge: position at both
    anchors is clamped so the only remaining free motion is rotation
    about the line ``anchor1 -> anchor2``. See
    :class:`~newton._src.solvers.jitter.constraints.constraint_double_ball_socket.DoubleBallSocketData`
    for the full derivation.

    Both anchors are in *world* space at finalize() time; each is
    expressed in each body's local frame the same way
    :class:`BallSocketDescriptor` handles its single anchor.

    ``hertz`` / ``damping_ratio`` follow the Box2D v3 / Bepu soft-
    constraint formulation. ``hertz = 0`` (default) gives a rigid lock
    (no compliance on either anchor), matching the rigid-anchor choice
    used by :class:`HingeJointDescriptor`.
    """

    body1: int
    body2: int
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class DoubleBallSocketPrismaticDescriptor:
    """Plain-Python description of one standalone prismatic DBS lock.

    Translational twin of :class:`DoubleBallSocketDescriptor`: locks
    5 DoF so the only free relative motion is translation along the
    line from ``anchor1`` to ``anchor2``. Rotation is locked entirely
    via a 4+1 Schur-complement solve (two tangent rows per on-axis
    anchor plus a scalar row at an auto-derived off-axis point); see
    :class:`~newton._src.solvers.jitter.constraints.constraint_double_ball_socket_prismatic.DoubleBallSocketPrismaticData`
    for the full derivation.

    Both anchors are in *world* space at finalize() time. Pair with
    :meth:`WorldBuilder.add_linear_motor` (velocity or PD-drive mode)
    to build a motorised prismatic joint out of standalone pieces.

    ``hertz`` / ``damping_ratio`` follow the Box2D v3 / Bepu soft-
    constraint formulation and apply uniformly to all 5 lock rows.
    ``hertz = 0`` (default) gives a rigid lock.
    """

    body1: int
    body2: int
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class HingeAngleDescriptor:
    """Plain-Python description of one hinge-angle (2-DoF) constraint.

    The ``axis`` is in *world* space at construction time; finalize()
    transforms it into body 2's local frame, mirroring
    ``HingeAngle.Initialize``. ``min_angle`` / ``max_angle`` are in
    radians.

    ``hertz_lock`` / ``damping_ratio_lock`` parametrise the perpendicular
    -to-hinge angular lock (rows 0 / 1 of the Jacobian) as a Box2D / Bepu
    soft-constraint pair; ``hertz_limit`` / ``damping_ratio_limit`` do
    the same for the axial min / max limit (row 2). See
    :func:`soft_constraint_coefficients` for the formulation.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    min_angle: float = -math.pi
    max_angle: float = math.pi
    hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR)
    damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO)
    hertz_limit: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class AngularMotorDescriptor:
    """Plain-Python description of one angular-motor constraint.

    The motor drives the relative rotation between ``body1`` and
    ``body2`` around ``axis`` (world space at construction time) in one
    of two modes, selected automatically at init time:

    * **Velocity target** (``stiffness == 0`` and ``damping == 0``):
      the Jitter2 default. Drives the relative angular velocity to
      ``target_velocity`` [rad/s] using at most ``max_force`` [N·m] of
      torque. ``hertz`` / ``damping_ratio`` follow the Box2D v3 / Bepu
      soft-constraint formulation (see
      :func:`soft_constraint_coefficients`). ``max_force = 0`` (Jitter
      default) yields a disabled motor that does nothing.

    * **PD position target** (``stiffness > 0`` or ``damping > 0``):
      behaves like a rotational spring-damper that holds the relative
      twist angle at ``target_angle`` [rad], measured from the
      *initial* relative pose of the two bodies at finalize() time.
      ``stiffness`` is in N·m/rad, ``damping`` in N·m·s/rad. Unbounded
      angle tracking across 2*pi wraps uses PhoenX's
      ``FullRevolutionTracker`` (see
      :mod:`newton._src.solvers.jitter.helpers.math_helpers`) so arbitrarily
      large target angles are well-defined. ``target_velocity`` is then
      interpreted as an *additional* feed-forward angular velocity
      (usually 0); ``max_force`` still caps the per-substep impulse.
      ``hertz`` / ``damping_ratio`` are ignored on this path.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    target_velocity: float = 0.0
    max_force: float = 0.0
    hertz: float = float(DEFAULT_HERTZ_MOTOR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    target_angle: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class LinearMotorDescriptor:
    """Plain-Python description of one linear-motor constraint.

    Translational twin of :class:`AngularMotorDescriptor`. Drives the
    relative translation between ``body1`` and ``body2`` along ``axis``
    (world space at construction time) in one of two modes:

    * **Velocity target** (``stiffness == 0`` and ``damping == 0``):
      Jitter2 default. Drives the linear-velocity difference along
      ``axis`` to ``target_velocity`` [m/s] using at most ``max_force``
      [N]. ``hertz`` / ``damping_ratio`` parametrise the Box2D /
      Bepu soft-constraint velocity-tracking spring. ``max_force = 0``
      disables the motor (no-op) on this path.

    * **PD position target** (``stiffness > 0`` or ``damping > 0``):
      Linear spring-damper that holds the slide at ``target_position``
      [m], measured from the initial relative pose of
      ``anchor1`` -> ``anchor2`` at finalize() time. ``stiffness`` is
      in N/m, ``damping`` in N·s/m. ``target_velocity`` is then a
      feed-forward slide velocity (usually 0); ``max_force`` still
      caps the per-substep impulse. ``hertz`` / ``damping_ratio`` are
      ignored on this path.

    The anchors are used by the PD path to reconstruct the current
    slide from world coordinates
    (``slide_now = hat_n . (anchor2_world - anchor1_world)``). In
    pure-velocity mode they can point to any two collocated world
    points on the slide axis.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    target_velocity: float = 0.0
    max_force: float = 0.0
    hertz: float = float(DEFAULT_HERTZ_MOTOR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    target_position: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class AngularLimitDescriptor:
    """Plain-Python description of one angular-limit constraint.

    Clamps the relative twist between ``body1`` and ``body2`` around
    ``axis`` (world space at finalize time) to ``[min_value,
    max_value]`` (rad, measured from the initial relative pose).

    Two soft-constraint conventions are supported, selected
    automatically at init time:

    * **Box2D / Bepu** (``stiffness == 0`` and ``damping == 0``): the
      stop spring is parametrised by ``hertz`` [Hz] and
      ``damping_ratio`` [dimensionless]. The effective mass of the row
      is baked into the gains -- see
      :func:`constraint_container.soft_constraint_coefficients`.
    * **PD spring-damper** (``stiffness > 0`` or ``damping > 0``):
      absolute SI gains with ``stiffness`` in N*m/rad and ``damping``
      in N*m*s/rad, plugged through
      :func:`constraint_container.pd_coefficients`. Produces the same
      per-unit-inertia response regardless of body mass, at the cost
      of the caller having to think in physical units.

    The clamp is unilateral: the row only ever pushes the joint back
    into ``[min_value, max_value]``, never out of it. Set
    ``min_value > max_value`` to disable the row entirely; use a
    large sentinel (e.g. ``min_value = -1e9``) to express a one-sided
    limit on one direction only.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    min_value: float
    max_value: float
    hertz: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class LinearLimitDescriptor:
    """Plain-Python description of one linear-limit constraint.

    Translational analogue of :class:`AngularLimitDescriptor`. Clamps
    the relative slide along ``axis`` between two anchor points to
    ``[min_value, max_value]`` (m, measured from the initial relative
    slide at finalize time).

    Same dual-convention soft plumbing:

    * **Box2D / Bepu** (``stiffness == 0`` and ``damping == 0``):
      ``hertz`` [Hz] and ``damping_ratio``.
    * **PD spring-damper** (``stiffness > 0`` or ``damping > 0``):
      ``stiffness`` [N/m] and ``damping`` [N*s/m].

    The anchors serve exactly the same role as in
    :class:`LinearMotorDescriptor`: the current slide is
    reconstructed each substep as
    ``slide = axis . (anchor2_world - anchor1_world)``. For a typical
    prismatic joint ``anchor1 == anchor2`` at finalize, giving a rest
    slide of zero.

    Set ``min_value > max_value`` to disable the row; use a large
    sentinel on either side for a one-sided stop.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    min_value: float
    max_value: float
    hertz: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class HingeJointDescriptor:
    """Plain-Python description of one fused hinge-joint constraint.

    A Jitter-style hinge joint composed of a HingeAngle (2-DoF angular
    lock, optionally with limits), a BallSocket (3-DoF positional lock),
    and an AngularMotor (drives the remaining axial DoF). All three
    sub-constraints live in *one* column of the shared
    :class:`ConstraintContainer` and are owned by a single PGS thread
    -- see :mod:`newton._src.solvers.jitter.constraints.constraint_hinge_joint` for
    why this fuses better than three separate constraints.

    ``hinge_center`` and ``hinge_axis`` are in *world* space at
    finalize() time. Set ``max_force = 0.0`` for a passive (unmotorised)
    joint -- the AngularMotor sub then applies zero corrective impulse
    and acts as a no-op.
    """

    body1: int
    body2: int
    hinge_center: tuple[float, float, float]
    hinge_axis: tuple[float, float, float]
    min_angle: float = -math.pi
    max_angle: float = math.pi
    target_velocity: float = 0.0
    max_force: float = 0.0
    hertz_linear: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO)
    hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR)
    damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO)
    hertz_limit: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO)
    hertz_motor: float = float(DEFAULT_HERTZ_MOTOR)
    damping_ratio_motor: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class HingeJointHandle:
    """Global cid of the fused hinge-joint constraint created by
    :meth:`WorldBuilder.add_hinge_joint`.

    Returned with a sentinel cid (``-1``) at the time of the
    ``add_hinge_joint`` call and rewritten in place by
    :meth:`WorldBuilder.finalize` to the actual cid in the shared
    :class:`ConstraintContainer`. Pass it to
    :meth:`World.gather_constraint_wrenches` (indirectly via
    ``wrenches[handle.cid]``) to read the fused joint's combined
    reaction wrench -- the sum of its BallSocket / HingeAngle /
    AngularMotor sub-contributions on body 2.
    """

    cid: int = -1


class DriveMode(IntEnum):
    """Drive mode for an actuated joint's free DoF.

    See :mod:`newton._src.solvers.jitter.constraints.constraint_actuated_double_ball_socket`
    for the per-mode formulation. Values match the underlying
    ``DRIVE_MODE_*`` Warp constants exactly so they can be passed
    through ``np.int32`` arrays without translation.
    """

    OFF = int(DRIVE_MODE_OFF)
    POSITION = int(DRIVE_MODE_POSITION)
    VELOCITY = int(DRIVE_MODE_VELOCITY)


class JointMode(IntEnum):
    """Which physical joint a :class:`JointDescriptor` materialises.

    A single unified constraint schema (3- or 5-DoF positional lock +
    optional scalar actuator row) covers ball-socket, revolute (hinge),
    and prismatic (slider) joints; the mode picks which pure-point
    formulation the runtime kernel uses and what units the drive /
    limit are in.

    See :mod:`newton._src.solvers.jitter.constraints.constraint_actuated_double_ball_socket`
    for the per-mode math. Values match the underlying
    ``JOINT_MODE_*`` Warp constants exactly so they can be passed
    through ``np.int32`` arrays without translation.
    """

    REVOLUTE = int(JOINT_MODE_REVOLUTE)
    PRISMATIC = int(JOINT_MODE_PRISMATIC)
    BALL_SOCKET = int(JOINT_MODE_BALL_SOCKET)


@dataclass
class JointDescriptor:
    """Plain-Python description of one unified ball-socket / revolute / prismatic joint.

    A single pure-point positional lock (3-DoF for ball-socket, 5-DoF
    for revolute / prismatic -- solved as a direct 3x3 inverse or a
    rank-5 Schur complement) plus an *optional* scalar actuator row
    that drives and/or clamps the free DoF (revolute / prismatic
    only). The ``mode`` selects which physical joint this is:

    * :attr:`JointMode.BALL_SOCKET` -- a single ball-socket. Locks 3
      translational DoF at ``anchor1``; all 3 rotational DoF are free.
      No ``anchor2``, no drive, no limit -- the two bodies are pinned
      together at one world point and rotate independently. ``anchor2``
      is ignored; drive / limit fields must be left at their defaults.
    * :attr:`JointMode.REVOLUTE` -- a double ball-socket hinge. Locks
      3 translational + 2 rotational DoF; the free DoF is rotation
      about the line through ``anchor1`` and ``anchor2``. The drive
      and limit, if active, interpret ``target`` / ``min_value`` /
      ``max_value`` as angles [rad] and ``max_force_drive`` as a torque
      cap [N*m].
    * :attr:`JointMode.PRISMATIC` -- a slider. Locks 3 rotational + 2
      translational DoF; the free DoF is translation along the line
      through ``anchor1`` and ``anchor2``. The drive and limit, if
      active, interpret ``target`` / ``min_value`` / ``max_value`` as
      displacements [m] along the slide axis (measured from
      ``anchor1``) and ``max_force_drive`` as a linear force cap [N].

    For revolute and prismatic modes, ``anchor1`` and ``anchor2`` are
    two world-space points on the joint axis at :meth:`finalize` time.
    The init kernel snapshots them into each body's local frame; the
    line between them defines the hinge / slide direction. For
    prismatic mode the init kernel additionally auto-derives a third
    anchor (perpendicular to the slide axis, at distance
    ``|anchor2 - anchor1|`` -- i.e. the *rest length*) so the point-
    matching 2+2+1 formulation has a non-degenerate twist-lock row
    without any user-visible API change.

    Passing ``anchor2 = anchor1 + axis`` with a unit ``axis`` is the
    common convention -- it yields ``rest_length = 1 m`` and makes the
    slide / hinge axis independent of any scene-wide length unit.

    For ball-socket mode ``anchor2`` is set to ``anchor1`` internally
    (so the init kernel's axis snapshot is degenerate but harmless;
    the runtime dispatch ignores anchor-2 state entirely).

    The positional block uses the Box2D v3 / Bepu / Nordby
    soft-constraint formulation (see :func:`soft_constraint_coefficients`)
    with mass-independent ``(hertz, damping_ratio)`` knobs. The
    default ``hertz = DEFAULT_HERTZ_LINEAR`` / critical damping applies
    to the positional block (3x3 for ball-socket, rank-5 Schur
    otherwise).

    Actuator (drive) -- **normal PD, absolute SI gains**
    ---------------------------------------------------
    The drive row is a Jitter2-style implicit-Euler spring-damper:

    .. math::

        \\tau = -k_p\\, (x - x_{\\text{target}}) - k_d\\, (v -
                 v_{\\text{target}})

    parameterised by:

      * :attr:`stiffness_drive` -- ``kp`` [N/m for prismatic, N*m/rad
        for revolute]. Default ``0`` (drive off).
      * :attr:`damping_drive`   -- ``kd`` [N*s/m or N*m*s/rad].
        Default ``0``.

    ``stiffness_drive == damping_drive == 0`` disables the drive row
    entirely -- no impulse is applied even if ``drive_mode != OFF``.
    That's the "drive off" idiom; use it to construct a free-DoF
    revolute or prismatic joint.

    Limit -- **dual convention**
    ----------------------------
    The unilateral ``[min_value, max_value]`` limit row accepts two
    parameter conventions, auto-selected by whether either PD gain
    is strictly positive (matching the standalone
    :class:`AngularLimitDescriptor` / :class:`LinearLimitDescriptor`):

      * ``stiffness_limit == damping_limit == 0`` -- Box2D
        mass-independent knobs ``(hertz_limit, damping_ratio_limit)``;
        interpretation identical to the positional block.
      * ``stiffness_limit > 0`` or ``damping_limit > 0`` -- Jitter2
        PD spring-damper with absolute SI gains, same implicit-Euler
        math as the drive row above.

    ``min_value > max_value`` disables the limit row. Leaving
    ``drive_mode = DriveMode.OFF`` *and*
    ``stiffness_drive == damping_drive == 0`` *and*
    ``min_value > max_value`` gives a *non-actuated* revolute or
    prismatic joint. Ball-socket has no actuator row -- drive /
    limit fields are required to stay at their defaults.
    """

    body1: int
    body2: int
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    mode: JointMode = JointMode.REVOLUTE
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    drive_mode: DriveMode = DriveMode.OFF
    target: float = 0.0
    target_velocity: float = 0.0
    max_force_drive: float = 0.0
    stiffness_drive: float = 0.0
    damping_drive: float = 0.0
    min_value: float = 1.0
    max_value: float = -1.0
    hertz_limit: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO)
    stiffness_limit: float = 0.0
    damping_limit: float = 0.0


@dataclass
class JointHandle:
    """Global cid of a unified joint created by :meth:`WorldBuilder.add_joint`.

    Returned with a sentinel cid (``-1``) and rewritten in place by
    :meth:`WorldBuilder.finalize`. Pass to
    :meth:`World.gather_constraint_wrenches` (via
    ``wrenches[handle.cid]``) to read the joint's reaction wrench on
    body 2 (linear force + angular torque, both in world frame). In
    prismatic mode the axial impulse shows up on the force component;
    in revolute mode it shows up on the torque component.
    """

    cid: int = -1


@dataclass
class PrismaticDescriptor:
    """Plain-Python description of one *legacy* standalone prismatic joint.

    Represents a 5-DoF slider constraint implemented by
    :mod:`newton._src.solvers.jitter.constraints.constraint_prismatic` -- 3
    rotational rows + 2 perpendicular-translation rows, solved as one
    PGS column via a 3x3 + 2x2 Schur complement with quaternion-error
    rotational rows (not pure point-matching).

    Kept alongside the newer unified :class:`JointDescriptor`
    (``JointMode.PRISMATIC`` -- pure 2+2+1 point-matching formulation)
    so callers can benchmark both implementations against each other;
    see ``example_motorized_prismatic_chain`` for a side-by-side
    comparison. New code should prefer :meth:`WorldBuilder.add_joint`.

    The anchor is in *world* space at finalize() time; the init
    kernel snapshots it into each body's local frame. The slide axis
    is likewise a world-space direction at finalize() time and is
    snapshotted into body 1's local frame (the slide direction "rides"
    body 1).

    ``hertz_angular`` / ``damping_ratio_angular`` set the soft-constraint
    spring/damper for the 3-row rotational lock; the separate
    ``hertz_linear`` / ``damping_ratio_linear`` knobs do the same for
    the 2-row perpendicular-translation lock.
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    hertz_angular: float = float(DEFAULT_HERTZ_ANGULAR)
    damping_ratio_angular: float = float(DEFAULT_DAMPING_RATIO)
    hertz_linear: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class PrismaticHandle:
    """Global cid of a legacy prismatic created by :meth:`WorldBuilder.add_prismatic`.

    Returned with a sentinel cid (``-1``) and rewritten in place by
    :meth:`WorldBuilder.finalize`. Pass to
    :meth:`World.gather_constraint_wrenches` (via
    ``wrenches[handle.cid]``) to read the joint's combined reaction
    wrench on body 2 (linear constraint force + angular torque, both
    in world frame).
    """

    cid: int = -1


@dataclass
class D6AxisDrive:
    """Per-axis configuration for one axis of a :class:`D6Descriptor`.

    Each of the 6 axes (3 angular + 3 linear) is independently
    configurable as:

    * **Rigid lock**       -- ``hertz=0`` (default) and
      ``max_force=math.inf`` (default). The default
      ``D6AxisDrive()`` is a rigid lock; an unconfigured D6 is a
      6-DoF rigid weld.
    * **Soft lock**        -- ``hertz>0``, ``target_position=0``,
      ``target_velocity=0``, ``max_force=inf``. A spring-and-damper
      restoring the rest pose with the Box2D / Bepu / Nordby implicit
      formulation.
    * **Position drive**   -- ``hertz>0``, ``target_position!=0``,
      ``max_force`` finite. Implicit-PD drive that pulls the axis
      towards the target with the given soft-spring stiffness, capped
      at ``max_force``.
    * **Velocity drive**   -- ``hertz>0``, ``target_velocity!=0``,
      ``max_force`` finite. Implicit-PD velocity-tracking drive (the
      "motor" mode); ``hertz`` controls how aggressively the motor
      tracks the setpoint.
    * **Free**             -- ``max_force=0``. The accumulated impulse
      is pinned to zero every PGS iteration regardless of ``hertz``;
      this is the canonical way to mark an axis as un-constrained.

    Position + velocity targets compose: setting both gives an
    over-damped spring with a steady-state velocity (useful for
    e.g. "open the door at 1 rad/s while pulling it towards 90°").

    Units (per axis):
        * Angular axes: ``target_position`` [rad], ``target_velocity``
          [rad/s], ``max_force`` [N*m].
        * Linear axes:  ``target_position`` [m],   ``target_velocity``
          [m/s],   ``max_force`` [N].
        * ``hertz`` is always [Hz]; ``damping_ratio`` is dimensionless
          (1 = critically damped).
    """

    hertz: float = 0.0
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    target_position: float = 0.0
    target_velocity: float = 0.0
    max_force: float = math.inf


_D6_RIGID_LOCK = D6AxisDrive()


@dataclass
class D6Descriptor:
    """Plain-Python description of one 6-DoF generalised ("D6") joint.

    A single D6 owns *all* 6 relative DoF of body 2 with respect to
    body 1 and configures each axis independently via a
    :class:`D6AxisDrive`. The default ``D6Descriptor`` (no per-axis
    overrides) is a 6-DoF rigid weld at ``anchor``; pass non-default
    drives to soften axes, add PD setpoints, cap forces, or free
    individual axes.

    Internally dispatches the translation and rotation blocks to
    specialized sub-parts at ``finalize()`` time -- a fused 3-DoF
    point/euler constraint when every axis in the block is rigid, or
    three independent 1-DoF axis/angle constraints otherwise (Jolt
    ``SixDOFConstraint`` style). See
    :mod:`newton._src.solvers.jitter.constraints.constraint_d6` for the math.

    The 3 angular axes and 3 linear axes are interpreted in *body 1's
    local frame*: the joint frame "rides" body 1 rigidly. Position
    targets are folded into the rest pose at finalize() time so the
    runtime kernel never branches on "is there a position target".

    ``anchor`` is in *world* space at finalize() time. The init
    kernel snapshots it into both bodies' local frames; the linear
    position targets are added on top of the body-1-local anchor so
    that "drive to rest pose" tracks the user-specified offset
    automatically.
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]
    angular: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] = (
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
    )
    linear: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] = (
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
    )


@dataclass
class D6Handle:
    """Global cid of the D6 joint created by :meth:`WorldBuilder.add_d6`.

    Returned with a sentinel cid (``-1``) and rewritten in place by
    :meth:`WorldBuilder.finalize`. Pass to
    :meth:`World.gather_constraint_wrenches` (via
    ``wrenches[handle.cid]``) to read the joint's combined reaction
    wrench on body 2 (linear constraint force + angular reaction
    torque, both in world frame). Per-axis force/torque limits show up
    as saturated (``+/- max_force``) components in the reported wrench.
    """

    cid: int = -1


class WorldBuilder:
    """Append bodies and constraints, then materialise a :class:`World`.

    Usage::

        b = WorldBuilder()
        anchor = b.add_static_body()                # body 0 (auto)
        body_a = b.add_dynamic_body(position=(0, 0, 0), inverse_mass=1.0)
        body_b = b.add_dynamic_body(position=(1, 0, 0), inverse_mass=1.0)
        b.add_ball_socket(anchor, body_a, anchor=(0, 0, 0))
        b.add_ball_socket(body_a, body_b, anchor=(0.5, 0, 0))
        world = b.finalize(substeps=1, solver_iterations=8)

    The builder is single-use; call :meth:`finalize` exactly once.
    """

    def __init__(self):
        # Body 0 is the static world body; user calls to add_*_body
        # start at index 1. We seed it with the canonical static
        # defaults (origin, identity rotation, MOTION_STATIC, zero
        # inverse mass *and* zero inverse inertia, no gravity). The
        # zero inertia is critical: constraints attached to the world
        # body read ``inverse_inertia_world[0]`` for their effective-
        # mass terms, and a non-zero value would split the impulse as
        # if the world body were dynamic.
        self._bodies: list[RigidBodyDescriptor] = [
            RigidBodyDescriptor(
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
            )
        ]
        self._ball_sockets: list[BallSocketDescriptor] = []
        self._double_ball_sockets: list[DoubleBallSocketDescriptor] = []
        self._double_ball_socket_prismatics: list[DoubleBallSocketPrismaticDescriptor] = []
        self._hinge_angles: list[HingeAngleDescriptor] = []
        self._angular_motors: list[AngularMotorDescriptor] = []
        self._linear_motors: list[LinearMotorDescriptor] = []
        self._angular_limits: list[AngularLimitDescriptor] = []
        self._linear_limits: list[LinearLimitDescriptor] = []
        # Fused hinge-joint descriptors and their handles. The handle
        # list is parallel to the descriptor list -- finalize() walks
        # both to patch each handle with its global cid in place so the
        # user-held reference becomes valid after finalize() returns.
        self._hinge_joint_descriptors: list[HingeJointDescriptor] = []
        self._hinge_joint_handles: list[HingeJointHandle] = []
        # Unified revolute / prismatic joint descriptors. One list
        # serves both modes -- the single ``actuated_double_ball_socket``
        # Warp kernel dispatches on ``JointMode`` at runtime, so there
        # is no per-mode ingestion path. Same parallel-list pattern as
        # the fused HingeJoint above.
        self._joint_descriptors: list[JointDescriptor] = []
        self._joint_handles: list[JointHandle] = []
        # Legacy standalone prismatic joint descriptors. Kept alongside
        # the unified ``_joint_descriptors`` (which covers prismatic
        # via ``JointMode.PRISMATIC``) so callers can compare the two
        # implementations. Same parallel-list pattern.
        self._prismatic_descriptors: list[PrismaticDescriptor] = []
        self._prismatic_handles: list[PrismaticHandle] = []
        # 6-DoF generalised (D6) joint descriptors. Same parallel-list
        # pattern.
        self._d6_descriptors: list[D6Descriptor] = []
        self._d6_handles: list[D6Handle] = []

        # Pairwise contact filtering: canonical (min, max) body-index
        # tuples whose contacts the solver must ignore. See
        # :meth:`add_collision_filter_pair`.
        self._collision_filter_pairs: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Body API
    # ------------------------------------------------------------------

    @property
    def world_body(self) -> int:
        """Index of the auto-created static world body."""
        return 0

    def add_body(self, descriptor: RigidBodyDescriptor) -> int:
        """Append a fully-formed descriptor and return its body index.

        The descriptor is validated eagerly via
        :func:`_validate_body_descriptor`; obvious errors (NaN fields,
        negative inverse masses, DYNAMIC bodies with zero inverse mass,
        STATIC bodies with non-zero mass / velocity, non-unit
        quaternions, etc.) raise :class:`ValueError` here rather than
        producing silent corruption deep inside the solver. Valid
        descriptors are passed through unchanged.
        """
        next_index = len(self._bodies)
        _validate_body_descriptor(descriptor, next_index)
        self._bodies.append(descriptor)
        return next_index

    def add_static_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        """Append a static body (zero inverse mass / inertia, no integration)."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_STATIC),
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
            )
        )

    def add_kinematic_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Append a kinematic body: integrates its user-set velocities but
        ignores forces/contacts (matches Jitter ``MotionType.Kinematic``)."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_KINEMATIC),
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
                velocity=velocity,
                angular_velocity=angular_velocity,
            )
        )

    def add_dynamic_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        inverse_mass: float = 1.0,
        inverse_inertia: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = _IDENTITY_INERTIA,
        linear_damping: float = 1.0,
        angular_damping: float = 1.0,
        affected_by_gravity: bool = True,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Append a fully dynamic body. ``inverse_inertia`` is in the
        *body* frame; the solver rotates it to world space each step."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_DYNAMIC),
                inverse_mass=inverse_mass,
                inverse_inertia=inverse_inertia,
                linear_damping=linear_damping,
                angular_damping=angular_damping,
                affected_by_gravity=affected_by_gravity,
                velocity=velocity,
                angular_velocity=angular_velocity,
            )
        )

    # ------------------------------------------------------------------
    # Constraint API
    # ------------------------------------------------------------------

    def add_ball_socket(
        self,
        body1: int,
        body2: int,
        anchor: tuple[float, float, float],
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
    ) -> int:
        """Append a ball-and-socket constraint and return its (per-type)
        index. Both ``body1`` and ``body2`` must be valid body indices
        (i.e. returned from one of the ``add_*_body`` methods or 0 for
        the static world body).

        Args:
            body1: First body index.
            body2: Second body index.
            anchor: World-space anchor point [m].
            hertz: Soft-constraint stiffness target [Hz]; lower values
                make the joint more compliant. Defaults to
                :data:`DEFAULT_HERTZ_LINEAR`.
            damping_ratio: Non-dimensional damping ratio (1 is critical,
                <1 underdamped, >1 overdamped). Defaults to
                :data:`DEFAULT_DAMPING_RATIO`.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._ball_sockets.append(
            BallSocketDescriptor(body1, body2, anchor, hertz, damping_ratio)
        )
        return len(self._ball_sockets) - 1

    def add_double_ball_socket(
        self,
        body1: int,
        body2: int,
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float],
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
    ) -> int:
        """Append a double-ball-socket hinge lock and return its
        (per-type) index.

        Locks the relative linear position of ``body1`` and ``body2``
        at *two* world-space points ``anchor1`` and ``anchor2``. This
        is the 5-DoF hinge half of a revolute joint -- the remaining
        free DoF is rotation about the line connecting the two anchors.
        Pair with :meth:`add_angular_motor` (in either velocity or
        PD-drive mode) to build a motorised revolute joint out of
        stand-alone pieces.

        Args:
            body1: First body index.
            body2: Second body index.
            anchor1: World-space first anchor [m]. Defines the hinge
                line together with ``anchor2``.
            anchor2: World-space second anchor [m]. ``anchor1`` and
                ``anchor2`` must not coincide.
            hertz: Soft-constraint stiffness target [Hz]; ``0``
                gives a rigid lock. Defaults to
                :data:`DEFAULT_HERTZ_LINEAR`.
            damping_ratio: Non-dimensional damping ratio (1 is
                critical). Only consumed when ``hertz > 0``.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._double_ball_sockets.append(
            DoubleBallSocketDescriptor(
                body1, body2, anchor1, anchor2, hertz, damping_ratio
            )
        )
        return len(self._double_ball_sockets) - 1

    def add_hinge_angle(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        min_angle: float = -math.pi,
        max_angle: float = math.pi,
        hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR),
        damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO),
        hertz_limit: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO),
    ) -> int:
        """Append a stand-alone hinge-angle (2-DoF) constraint and return
        its per-type index. ``axis`` is interpreted in *world* space at
        finalize() time and snapshotted into body 2's local frame, just
        like ``HingeAngle.Initialize``. Limits are in radians; the
        defaults (``+- pi``) effectively disable the limit.

        ``hertz_lock`` / ``damping_ratio_lock`` set the soft-constraint
        spring/damper for the two perpendicular angular DoFs (the lock
        itself); ``hertz_limit`` / ``damping_ratio_limit`` do the same
        for the axial min/max limit only.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._hinge_angles.append(
            HingeAngleDescriptor(
                body1,
                body2,
                axis,
                min_angle,
                max_angle,
                hertz_lock,
                damping_ratio_lock,
                hertz_limit,
                damping_ratio_limit,
            )
        )
        return len(self._hinge_angles) - 1

    def add_angular_motor(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        target_velocity: float = 0.0,
        max_force: float = 0.0,
        hertz: float = float(DEFAULT_HERTZ_MOTOR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
        target_angle: float = 0.0,
        stiffness: float = 0.0,
        damping: float = 0.0,
    ) -> int:
        """Append a stand-alone angular motor and return its per-type
        index. ``axis`` is in world space. The motor has two modes,
        selected at init time:

        * **Velocity target** (default): drives the relative angular
          velocity along ``axis`` toward ``target_velocity`` [rad/s]
          using at most ``max_force`` [N·m] of torque. ``hertz`` /
          ``damping_ratio`` parametrise the Box2D v3 / Bepu velocity-
          tracking spring.
        * **PD position target**: set ``stiffness`` > 0 (N·m/rad)
          and/or ``damping`` > 0 (N·m·s/rad) to switch the motor into
          a rotational spring-damper that holds the relative twist at
          ``target_angle`` [rad], measured from the initial relative
          pose. ``max_force`` still caps the per-substep impulse.

        See :class:`AngularMotorDescriptor` for the full contract.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._angular_motors.append(
            AngularMotorDescriptor(
                body1,
                body2,
                axis,
                target_velocity,
                max_force,
                hertz,
                damping_ratio,
                target_angle,
                stiffness,
                damping,
            )
        )
        return len(self._angular_motors) - 1

    def add_double_ball_socket_prismatic(
        self,
        body1: int,
        body2: int,
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float],
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
    ) -> int:
        """Append a standalone prismatic double-ball-socket lock and
        return its (per-type) index.

        Locks 5 DoF so the only free motion between ``body1`` and
        ``body2`` is translation along the line from ``anchor1`` to
        ``anchor2``; rotation is locked entirely (4+1 Schur solve, see
        :class:`DoubleBallSocketPrismaticDescriptor`). Pair with
        :meth:`add_linear_motor` to build a motorised prismatic joint
        out of standalone pieces.

        Args:
            body1: First body index.
            body2: Second body index.
            anchor1: World-space first anchor [m]. Defines the slide
                axis together with ``anchor2``.
            anchor2: World-space second anchor [m]. Must not coincide
                with ``anchor1``.
            hertz: Soft-constraint stiffness target [Hz]; ``0`` gives
                a rigid lock (default). Applies uniformly to all 5
                lock rows.
            damping_ratio: Non-dimensional damping ratio (1 is
                critical). Only consumed when ``hertz > 0``.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._double_ball_socket_prismatics.append(
            DoubleBallSocketPrismaticDescriptor(
                body1, body2, anchor1, anchor2, hertz, damping_ratio
            )
        )
        return len(self._double_ball_socket_prismatics) - 1

    def add_linear_motor(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float],
        target_velocity: float = 0.0,
        max_force: float = 0.0,
        hertz: float = float(DEFAULT_HERTZ_MOTOR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
        target_position: float = 0.0,
        stiffness: float = 0.0,
        damping: float = 0.0,
    ) -> int:
        """Append a standalone linear motor and return its per-type
        index. Translational twin of :meth:`add_angular_motor`.

        ``axis`` is in world space and picks the translational
        direction that the motor drives; ``anchor1`` / ``anchor2`` are
        the world-space reference points used by the PD-position path
        to reconstruct the current slide
        (``slide_now = hat_axis . (anchor2_now - anchor1_now)``). In
        pure-velocity mode they can point to any two collocated points
        on the slide axis.

        Modes:

        * **Velocity target** (default): drive the relative linear
          velocity along ``axis`` toward ``target_velocity`` [m/s]
          using at most ``max_force`` [N]. ``hertz`` / ``damping_ratio``
          parametrise the Box2D / Bepu velocity-tracking spring.
        * **PD position target**: set ``stiffness`` > 0 (N/m) and/or
          ``damping`` > 0 (N·s/m) to switch into a linear spring-damper
          that holds the slide at ``target_position`` [m] measured
          from the initial
          ``hat_axis . (anchor2 - anchor1)`` offset at finalize() time.
          ``max_force`` still caps the per-substep impulse.

        See :class:`LinearMotorDescriptor` for the full contract.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._linear_motors.append(
            LinearMotorDescriptor(
                body1,
                body2,
                axis,
                anchor1,
                anchor2,
                target_velocity,
                max_force,
                hertz,
                damping_ratio,
                target_position,
                stiffness,
                damping,
            )
        )
        return len(self._linear_motors) - 1

    def add_angular_limit(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        min_value: float,
        max_value: float,
        hertz: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
        stiffness: float = 0.0,
        damping: float = 0.0,
    ) -> int:
        """Append a standalone angular limit and return its per-type index.

        Clamps the relative twist between ``body1`` and ``body2``
        around ``axis`` (world space at finalize time) to
        ``[min_value, max_value]`` (rad, measured from the initial
        relative pose). See :class:`AngularLimitDescriptor` for the
        dual softness conventions and sentinel rules for disabling
        / one-siding the row.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._angular_limits.append(
            AngularLimitDescriptor(
                body1,
                body2,
                axis,
                min_value,
                max_value,
                hertz,
                damping_ratio,
                stiffness,
                damping,
            )
        )
        return len(self._angular_limits) - 1

    def add_linear_limit(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float],
        min_value: float,
        max_value: float,
        hertz: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
        stiffness: float = 0.0,
        damping: float = 0.0,
    ) -> int:
        """Append a standalone linear limit and return its per-type index.

        Translational analogue of :meth:`add_angular_limit`. Clamps
        the slide along ``axis`` between ``anchor1`` and ``anchor2``
        (world-space at finalize) to ``[min_value, max_value]`` (m,
        measured from the initial slide). See
        :class:`LinearLimitDescriptor` for the contract.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._linear_limits.append(
            LinearLimitDescriptor(
                body1,
                body2,
                axis,
                anchor1,
                anchor2,
                min_value,
                max_value,
                hertz,
                damping_ratio,
                stiffness,
                damping,
            )
        )
        return len(self._linear_limits) - 1

    def add_hinge_joint(
        self,
        body1: int,
        body2: int,
        hinge_center: tuple[float, float, float],
        hinge_axis: tuple[float, float, float],
        min_angle: float = -math.pi,
        max_angle: float = math.pi,
        motor: bool = False,
        target_velocity: float = 0.0,
        max_force: float = 0.0,
        hertz_linear: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO),
        hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR),
        damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO),
        hertz_limit: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO),
        hertz_motor: float = float(DEFAULT_HERTZ_MOTOR),
        damping_ratio_motor: float = float(DEFAULT_DAMPING_RATIO),
    ) -> HingeJointHandle:
        """Append a Jitter-style hinge joint and return its handle.

        Builds a *fused* :data:`CONSTRAINT_TYPE_HINGE_JOINT` constraint
        that packs the HingeAngle (2-DoF angular lock with optional
        limits), BallSocket (3-DoF positional lock at ``hinge_center``),
        and AngularMotor (drives the axial velocity) into one column.
        The single-thread fused dispatch typically converges noticeably
        better than the same triple as three separate constraints --
        see :mod:`newton._src.solvers.jitter.constraints.constraint_hinge_joint`
        for the rationale.

        Mirrors Jitter2's ``HingeJoint`` constructor
        (``Jitter2/Dynamics/Joints/HingeJoint.cs``); the ``hasMotor``
        flag is spelt ``motor`` here for Python ergonomics. When
        ``motor`` is false the AngularMotor sub stores ``max_force = 0``
        and applies no torque, matching the C# ``hasMotor=false`` path.
        Both ``hinge_center`` and ``hinge_axis`` are in *world* space at
        finalize() time.

        Returns a :class:`HingeJointHandle` whose ``cid`` field is
        rewritten in place by :meth:`finalize` to the joint's global
        cid in the shared :class:`ConstraintContainer`.
        """
        self._validate_body(body1)
        self._validate_body(body2)

        if motor:
            mf = max_force
            tv = target_velocity
        else:
            mf = 0.0
            tv = 0.0
        descriptor = HingeJointDescriptor(
            body1=body1,
            body2=body2,
            hinge_center=hinge_center,
            hinge_axis=hinge_axis,
            min_angle=min_angle,
            max_angle=max_angle,
            target_velocity=tv,
            max_force=mf,
            hertz_linear=hertz_linear,
            damping_ratio_linear=damping_ratio_linear,
            hertz_lock=hertz_lock,
            damping_ratio_lock=damping_ratio_lock,
            hertz_limit=hertz_limit,
            damping_ratio_limit=damping_ratio_limit,
            hertz_motor=hertz_motor,
            damping_ratio_motor=damping_ratio_motor,
        )
        handle = HingeJointHandle(cid=-1)
        self._hinge_joint_descriptors.append(descriptor)
        self._hinge_joint_handles.append(handle)
        return handle

    def add_joint(
        self,
        body1: int,
        body2: int,
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float] | None = None,
        mode: JointMode = JointMode.REVOLUTE,
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
        drive_mode: DriveMode = DriveMode.OFF,
        target: float = 0.0,
        target_velocity: float = 0.0,
        max_force_drive: float = 0.0,
        stiffness_drive: float = 0.0,
        damping_drive: float = 0.0,
        min_value: float = 1.0,
        max_value: float = -1.0,
        hertz_limit: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO),
        stiffness_limit: float = 0.0,
        damping_limit: float = 0.0,
    ) -> JointHandle:
        """Append a unified ball-socket / revolute / prismatic joint and return its handle.

        One entry point materialises:

        * a 3-DoF ball-socket (ball-socket mode -- single anchor, pure
          point lock, all 3 rotations free, no drive / limit),
        * a 5-DoF hinge (revolute mode -- double ball-socket), or
        * a 5-DoF slider (prismatic mode -- 2+2+1 tangent-plane triad),

        plus an optional scalar actuator row (revolute / prismatic
        only) that drives and/or clamps the free DoF. All three modes
        share one constraint schema and one Warp kernel; the runtime
        picks the per-mode prepare / iterate math from ``mode``.

        For revolute and prismatic, ``anchor1`` and ``anchor2`` are
        two world-space points on the joint axis at :meth:`finalize`
        time. The line between them defines the hinge axis (revolute)
        or slide direction (prismatic). For a unit-length axis
        convention, pass ``anchor2 = anchor1 + axis`` (yields
        ``rest_length = 1 m`` independent of scene units); any two
        distinct points work.

        For ball-socket, ``anchor2`` must be ``None`` (the default);
        ``drive_mode`` / ``min_value`` / ``max_value`` must be left at
        their defaults -- ball-socket has no actuator row, and passing
        non-default values raises ``ValueError`` to fail loudly rather
        than silently ignoring them.

        For revolute / prismatic, leaving ``drive_mode = DriveMode.OFF``
        and ``min_value == max_value == 0`` disables the actuator row
        entirely (non-actuated joint). Otherwise:

        * :attr:`DriveMode.POSITION` / :attr:`DriveMode.VELOCITY` on the
          free DoF, capped by ``max_force_drive`` (torque [N*m] for
          revolute, force [N] for prismatic), with an implicit-Euler
          PD spring-damper (``stiffness_drive`` / ``damping_drive``).
        * One-sided ``[min_value, max_value]`` spring-damper limits on
          the same DoF (``min_value == max_value == 0`` disables). Units
          are rad for revolute and m for prismatic.

        Drive and limit are independent and can be active
        simultaneously; the limit always wins because it's unilateral.
        See :mod:`newton._src.solvers.jitter.constraints.constraint_actuated_double_ball_socket`
        for the derivation of all three modes.

        Args:
            body1: Index of the first body. For prismatic mode the
                slide axis is snapshotted into this body's local frame.
            body2: Index of the second body.
            anchor1: First anchor in *world* space at finalize() time [m].
                Both bodies are expected to coincide at this point at
                finalize time (body positions / orientations are used
                to derive the body-local snapshots).
            anchor2: Second anchor in *world* space at finalize() time [m];
                the line ``anchor1 -> anchor2`` defines the joint axis
                for revolute / prismatic modes. Must be ``None`` for
                ball-socket (and is required for the other two modes).
            mode: :attr:`JointMode.BALL_SOCKET`,
                :attr:`JointMode.REVOLUTE` (hinge), or
                :attr:`JointMode.PRISMATIC` (slider). Default revolute.
            hertz: Positional block soft-constraint frequency [Hz]
                (3x3 block for ball-socket, rank-5 Schur otherwise).
            damping_ratio: Positional block damping ratio.
            drive_mode: One of :class:`DriveMode`. Must be
                :attr:`DriveMode.OFF` for ball-socket.
            target: Position-drive setpoint. Units [rad] for revolute,
                [m] for prismatic (measured from ``anchor1`` along the
                axis). Ignored for ball-socket (must be 0).
            target_velocity: Velocity-drive setpoint. Units [rad/s] for
                revolute, [m/s] for prismatic. Ignored for ball-socket
                (must be 0).
            max_force_drive: Drive impulse cap, in torque [N*m] for
                revolute or linear force [N] for prismatic. Active for
                both :attr:`DriveMode.POSITION` and
                :attr:`DriveMode.VELOCITY`: ``> 0`` clamps the
                per-substep impulse to ``max_force_drive * dt``, ``0``
                means *unlimited* for POSITION and *disables* the drive
                for VELOCITY. Must be 0 for ball-socket.
            stiffness_drive: Drive PD stiffness ``kp``; [N*m/rad] for
                revolute or [N/m] for prismatic. Default ``0`` (drive
                off -- no impulse even if ``drive_mode != OFF``).
            damping_drive: Drive PD damping ``kd``; [N*m*s/rad] for
                revolute or [N*s/m] for prismatic. Default ``0``.
                See :class:`JointDescriptor` for the full PD spec.
            min_value: Lower limit. Units [rad] for revolute, [m] for
                prismatic. ``min_value > max_value`` disables the
                limit row (default). Must leave the limit disabled
                for ball-socket.
            max_value: Upper limit. Same units as ``min_value``.
            hertz_limit: Box2D limit row frequency [Hz]. Consumed
                only when ``stiffness_limit == damping_limit == 0``.
            damping_ratio_limit: Box2D limit row damping ratio.
                Consumed only when ``stiffness_limit ==
                damping_limit == 0``.
            stiffness_limit: Limit PD stiffness ``kp`` [N*m/rad]
                (revolute) or [N/m] (prismatic). Default ``0``.
            damping_limit: Limit PD damping ``kd`` [N*m*s/rad]
                (revolute) or [N*s/m] (prismatic). Default ``0``.
                ``stiffness_limit > 0`` *or* ``damping_limit > 0``
                selects the PD path (same math as the drive row);
                both zero falls back to the Box2D path.

        Returns:
            A :class:`JointHandle` whose ``cid`` field is rewritten in
            place by :meth:`finalize` to the joint's global cid in the
            shared :class:`ConstraintContainer`.

        Raises:
            ValueError: If ``mode`` and the other arguments are
                inconsistent -- e.g. ``anchor2`` missing in revolute /
                prismatic mode, or drive / limit / ``anchor2`` set in
                ball-socket mode.
        """
        self._validate_body(body1)
        self._validate_body(body2)

        mode_enum = JointMode(int(mode))
        drive_mode_enum = DriveMode(int(drive_mode))

        if mode_enum is JointMode.BALL_SOCKET:
            if anchor2 is not None:
                raise ValueError(
                    "add_joint(mode=JointMode.BALL_SOCKET) must not receive an "
                    "``anchor2`` argument -- a ball-socket is a single-point pin."
                )
            if drive_mode_enum is not DriveMode.OFF:
                raise ValueError(
                    "add_joint(mode=JointMode.BALL_SOCKET) has no actuator row; "
                    "``drive_mode`` must be DriveMode.OFF (got "
                    f"{drive_mode_enum.name})."
                )
            if target != 0.0 or target_velocity != 0.0 or max_force_drive != 0.0:
                raise ValueError(
                    "add_joint(mode=JointMode.BALL_SOCKET) has no actuator row; "
                    "``target`` / ``target_velocity`` / ``max_force_drive`` must "
                    "be left at their defaults (0.0)."
                )
            if min_value <= max_value:
                raise ValueError(
                    "add_joint(mode=JointMode.BALL_SOCKET) has no limit row; "
                    "``min_value`` / ``max_value`` must stay at the "
                    "limit-disabled defaults (``min_value > max_value``)."
                )
            if stiffness_limit != 0.0 or damping_limit != 0.0:
                raise ValueError(
                    "add_joint(mode=JointMode.BALL_SOCKET) has no limit row; "
                    "``stiffness_limit`` / ``damping_limit`` must be left "
                    "at their defaults (0.0)."
                )
            # Internally stash anchor2 = anchor1 so the shared init kernel
            # (which unconditionally reads both anchors) has something valid
            # to snapshot. The dispatcher's ball-socket branch ignores all
            # anchor-2 state anyway.
            anchor2_effective: tuple[float, float, float] = tuple(anchor1)  # type: ignore[assignment]
        else:
            if anchor2 is None:
                raise ValueError(
                    f"add_joint(mode={mode_enum.name}) requires an ``anchor2`` "
                    "argument; the line anchor1 -> anchor2 defines the joint axis."
                )
            anchor2_effective = anchor2
            # The unified joint's drive row is PD-only. VELOCITY mode
            # recovers pure velocity tracking via
            # ``stiffness_drive == 0, damping_drive > 0`` (the iterate
            # collapses to ``lam = -(1 / (1/eff_inv + kd/dt)) *
            # (jv + target_velocity)`` -- the canonical PD velocity
            # servo). A zero-damping VELOCITY drive used to fall
            # through to a rigid Jitter2-style pure-velocity motor;
            # that path is gone, so reject it up front with a message
            # that points the user at the replacement.
            if drive_mode_enum is DriveMode.VELOCITY and damping_drive <= 0.0:
                raise ValueError(
                    f"add_joint(mode={mode_enum.name}, "
                    "drive_mode=DriveMode.VELOCITY) requires "
                    "``damping_drive > 0``; the unified joint no longer "
                    "exposes a rigid pure-velocity motor. Set "
                    "``stiffness_drive=0`` and ``damping_drive`` to the "
                    "desired PD damping gain [N*s/m or N*m*s/rad] -- the "
                    "iterate then reduces to a PD velocity servo with "
                    "time constant tau = I_eff / damping_drive."
                )

        descriptor = JointDescriptor(
            body1=body1,
            body2=body2,
            anchor1=anchor1,
            anchor2=anchor2_effective,
            mode=mode_enum,
            hertz=hertz,
            damping_ratio=damping_ratio,
            drive_mode=drive_mode_enum,
            target=target,
            target_velocity=target_velocity,
            max_force_drive=max_force_drive,
            stiffness_drive=stiffness_drive,
            damping_drive=damping_drive,
            min_value=min_value,
            max_value=max_value,
            hertz_limit=hertz_limit,
            damping_ratio_limit=damping_ratio_limit,
            stiffness_limit=stiffness_limit,
            damping_limit=damping_limit,
        )
        handle = JointHandle(cid=-1)
        self._joint_descriptors.append(descriptor)
        self._joint_handles.append(handle)
        return handle

    def add_prismatic(
        self,
        body1: int,
        body2: int,
        anchor: tuple[float, float, float],
        axis: tuple[float, float, float],
        hertz_angular: float = float(DEFAULT_HERTZ_ANGULAR),
        damping_ratio_angular: float = float(DEFAULT_DAMPING_RATIO),
        hertz_linear: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO),
    ) -> PrismaticHandle:
        """Append a *legacy* standalone prismatic (slider) joint and return its handle.

        Materialises a :data:`CONSTRAINT_TYPE_PRISMATIC` column: 5 DoF
        locked (3 rotational + 2 perpendicular-translation), 1 free
        (translation along ``axis``). The rotational block is
        quaternion-error; the translational block is a tangent-plane
        2-row point lock. See
        :mod:`newton._src.solvers.jitter.constraints.constraint_prismatic` for the
        math.

        This entry point is kept alongside the newer unified
        :meth:`add_joint` (with ``mode=JointMode.PRISMATIC``, which
        uses a pure 2+2+1 point-matching formulation) so callers can
        benchmark both implementations side-by-side. New code should
        prefer :meth:`add_joint`.

        ``anchor`` and ``axis`` are in *world* space at finalize()
        time; the init kernel snapshots ``anchor`` into each body's
        local frame and ``axis`` into body 1's local frame. ``axis``
        does *not* have to be unit length -- the kernel normalises it.

        Args:
            body1: First body index. The slide direction "rides"
                this body rigidly.
            body2: Second body index.
            anchor: World-space anchor point [m]. Lies on the slide
                axis; both bodies' lever arms are measured from it.
            axis: World-space slide direction; any non-zero vector.
            hertz_angular: Rotational-lock stiffness [Hz]; applies to
                the 3-row rotational block.
            damping_ratio_angular: Rotational-lock damping ratio
                (1 is critical).
            hertz_linear: Perpendicular-translation lock stiffness
                [Hz]; applies to the 2-row translational block.
            damping_ratio_linear: Translational-lock damping ratio.

        Returns:
            A :class:`PrismaticHandle` whose ``cid`` field is
            rewritten in place by :meth:`finalize` to the joint's
            global cid.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        descriptor = PrismaticDescriptor(
            body1=body1,
            body2=body2,
            anchor=anchor,
            axis=axis,
            hertz_angular=hertz_angular,
            damping_ratio_angular=damping_ratio_angular,
            hertz_linear=hertz_linear,
            damping_ratio_linear=damping_ratio_linear,
        )
        handle = PrismaticHandle(cid=-1)
        self._prismatic_descriptors.append(descriptor)
        self._prismatic_handles.append(handle)
        return handle

    def add_d6(
        self,
        body1: int,
        body2: int,
        anchor: tuple[float, float, float],
        *,
        angular: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
        linear: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
    ) -> D6Handle:
        """Append a 6-DoF generalised ("D6") joint and return its handle.

        Owns *all* six relative DoF between ``body1`` and ``body2`` and
        configures each axis independently via a :class:`D6AxisDrive`:
        rigid lock, soft lock, position-PD drive, velocity-PD drive,
        force-limited drive, or free axis. Internally dispatches each
        block (translation / rotation) to specialized sub-parts at
        ``finalize()`` time -- a fused 3-DoF point/euler constraint
        when every axis in the block is rigid, or three independent
        1-DoF axis/angle constraints otherwise (Jolt
        ``SixDOFConstraint`` style). See
        :mod:`newton._src.solvers.jitter.constraints.constraint_d6` for the math.

        The default call ``b.add_d6(b1, b2, anchor)`` (no per-axis
        overrides) is a 6-DoF *rigid weld*: all 6 axes default to
        ``D6AxisDrive(hertz=0, max_force=inf)`` which collapses to a
        plain-PGS rigid lock on every axis. Pass per-axis
        :class:`D6AxisDrive` triples to soften axes, set position /
        velocity targets, cap the force/torque per axis, or free
        individual axes (``max_force=0``).

        ``anchor`` is in *world* space at finalize() time. The 3
        angular and 3 linear axes are interpreted in body-1's local
        frame ("frame A" in PhysX D6 / Bullet 6Dof2 convention). All
        position targets and velocity targets are in this body-1-local
        frame.

        Args:
            body1: First body index. The joint frame rides this body
                rigidly.
            body2: Second body index. Driven against the joint frame
                by the per-axis spec.
            anchor: World-space anchor point [m]. Both bodies measure
                their lever arms from this point at finalize() time.
            angular: 3-tuple of :class:`D6AxisDrive` for the
                (body-1-local) ``e_x``, ``e_y``, ``e_z`` angular axes.
                ``None`` -> all-rigid (default).
            linear: 3-tuple of :class:`D6AxisDrive` for the
                (body-1-local) ``e_x``, ``e_y``, ``e_z`` linear axes.
                ``None`` -> all-rigid (default).

        Returns:
            A :class:`D6Handle` whose ``cid`` field is rewritten in
            place by :meth:`finalize` to the joint's global cid in
            the shared :class:`ConstraintContainer`.

        Examples:

        Rigid weld (no kwargs needed)::

            b.add_d6(b1, b2, anchor=(0.0, 0.0, 0.0))

        Position-PD drive on the body-1-local +y angular axis (90 deg
        target, 5 Hz spring, 10 N*m torque cap), other 5 axes rigid::

            b.add_d6(
                b1, b2, anchor=(0.0, 0.0, 0.0),
                angular=(
                    D6AxisDrive(),
                    D6AxisDrive(hertz=5.0, target_position=math.pi/2,
                                max_force=10.0),
                    D6AxisDrive(),
                ),
            )

        A "free hinge" about body-1-local +y (3 angular DoF: x and z
        rigid, y free), positionally locked::

            b.add_d6(
                b1, b2, anchor=(0.0, 0.0, 0.0),
                angular=(
                    D6AxisDrive(),                        # rigid lock
                    D6AxisDrive(max_force=0.0),           # FREE
                    D6AxisDrive(),                        # rigid lock
                ),
            )
        """
        self._validate_body(body1)
        self._validate_body(body2)

        if angular is None:
            angular = (_D6_RIGID_LOCK, _D6_RIGID_LOCK, _D6_RIGID_LOCK)
        if linear is None:
            linear = (_D6_RIGID_LOCK, _D6_RIGID_LOCK, _D6_RIGID_LOCK)
        if len(angular) != 3:
            raise ValueError(
                f"D6 'angular' must be a 3-tuple of D6AxisDrive (got len={len(angular)})"
            )
        if len(linear) != 3:
            raise ValueError(
                f"D6 'linear' must be a 3-tuple of D6AxisDrive (got len={len(linear)})"
            )

        descriptor = D6Descriptor(
            body1=body1,
            body2=body2,
            anchor=anchor,
            angular=tuple(angular),
            linear=tuple(linear),
        )
        handle = D6Handle(cid=-1)
        self._d6_descriptors.append(descriptor)
        self._d6_handles.append(handle)
        return handle

    # ------------------------------------------------------------------
    # Pairwise contact filtering
    # ------------------------------------------------------------------

    def add_collision_filter_pair(self, body_a: int, body_b: int) -> None:
        """Ignore contacts between bodies ``body_a`` and ``body_b``.

        Mirrors Jitter2's ``IgnoreCollisionBetweenFilter`` and PhysX /
        Bullet's per-body-pair contact mask. After :meth:`finalize`,
        every contact emitted by the upstream collision pipeline whose
        ``(shape_body[shape_a], shape_body[shape_b])`` pair matches any
        registered filter (in either order) is dropped during ingest --
        no constraint column is allocated, no warm-start is gathered,
        and the dispatcher never sees it. The filter is the recommended
        mechanism for suppressing self-collision between jointed limbs
        of a ragdoll, sibling shapes on the same rigid body, etc.
        (jointed bodies typically overlap at the joint anchor, and the
        resulting spurious contacts fight the joint's positional
        constraint).

        The pair is stored in canonical order (``min(a, b), max(a, b)``)
        so either argument order produces the same filter. Duplicate
        registrations are idempotent.

        ``body_a`` and ``body_b`` may be any two bodies registered with
        this builder (including the static world body at index 0). A
        self-pair ``(b, b)`` is rejected: a body can never generate
        contacts with itself, so the filter would be a no-op and is
        almost certainly a caller mistake.

        Args:
            body_a: First body index (``>= 0``, ``< num_bodies``).
            body_b: Second body index (``>= 0``, ``< num_bodies``,
                ``!= body_a``).

        Raises:
            IndexError: If either index is out of range for this
                builder's body list.
            ValueError: If ``body_a == body_b``.
        """
        self._validate_body(body_a)
        self._validate_body(body_b)
        if body_a == body_b:
            raise ValueError(
                f"add_collision_filter_pair: body_a and body_b must differ "
                f"(got both = {body_a})"
            )
        self._collision_filter_pairs.add(
            (min(int(body_a), int(body_b)), max(int(body_a), int(body_b)))
        )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(
        self,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_relaxations: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        max_contact_columns: int = 0,
        rigid_contact_max: int = 0,
        num_shapes: int = 0,
        default_friction: float = 0.5,
        enable_all_constraints: bool = False,
        device: wp.context.Devicelike = None,
    ) -> World:
        """Allocate GPU storage and build a ready-to-step :class:`World`.

        The current registered descriptor lists are consumed in place
        (cleared after success). Calling :meth:`finalize` twice produces
        an empty world the second time -- clone the builder first if
        that's not what you want.

        Args:
            substeps: Forwarded to :class:`World`. Number of substeps
                per :meth:`World.step` call.
            solver_iterations: PGS iterations per substep.
            velocity_relaxations: Relaxation passes (currently a stub
                inside ``World``).
            gravity: World gravity vector [m/s^2].
            max_contact_columns: Upper bound on the number of
                :data:`CONSTRAINT_TYPE_CONTACT` columns the solver
                will ever pack per step. Sized by the user to match
                the worst-case shape-pair + slot-split count
                (``max_contact_pairs * ceil(max_contacts_per_pair /
                6)``). Sets aside that many trailing cids in the
                shared :class:`ConstraintContainer` and allocates the
                parallel :class:`~newton._src.solvers.jitter.constraints.contact_container.ContactContainer`
                for persistent lambdas. ``0`` disables the contact
                code paths entirely.
            rigid_contact_max: Upper bound on the number of *contacts*
                (not columns) the upstream Newton :class:`Contacts`
                buffer can hold. Sizes the forward ``(slot, cid)``
                lookup tables used by warm-starting. Defaults to
                ``max_contact_columns * 6`` if left at ``0`` --
                matches the worst case where every column is fully
                packed.
            num_shapes: Total number of Newton shapes in the model.
                Only used to pack the ``(shape_a, shape_b)`` RLE key
                into int32 during ingest; must satisfy
                ``num_shapes * num_shapes < 2**31``. Ignored when
                ``max_contact_columns == 0``.
            device: Warp device. ``None`` selects the current default
                device (typically ``cuda:0``).
        """
        device = wp.get_device(device)

        bodies = self._build_body_container(device)
        constraints, num_constraints = self._build_constraint_container(
            bodies, device, max_contact_columns=max_contact_columns
        )

        if max_contact_columns > 0 and rigid_contact_max == 0:
            # Default: assume every column could hold a full 6-contact
            # slot set. This is the worst case; user can override to
            # whatever they actually allocated on their Contacts.
            rigid_contact_max = max_contact_columns * 6

        world = World(
            bodies=bodies,
            constraints=constraints,
            num_constraints=num_constraints,
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_relaxations=velocity_relaxations,
            gravity=gravity,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            joint_constraint_count=num_constraints,
            collision_filter_pairs=self._collision_filter_pairs,
            default_friction=default_friction,
            enable_all_constraints=enable_all_constraints,
            device=device,
        )

        # Reset internal state so the builder can't accidentally leak
        # references into a second finalize call.
        self._bodies = [
            RigidBodyDescriptor(
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
            )
        ]
        self._ball_sockets = []
        self._double_ball_sockets = []
        self._double_ball_socket_prismatics = []
        self._hinge_angles = []
        self._angular_motors = []
        self._linear_motors = []
        self._angular_limits = []
        self._linear_limits = []
        self._hinge_joint_descriptors = []
        self._hinge_joint_handles = []
        self._joint_descriptors = []
        self._joint_handles = []
        self._prismatic_descriptors = []
        self._prismatic_handles = []
        self._d6_descriptors = []
        self._d6_handles = []
        self._collision_filter_pairs = set()
        return world

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_body(self, idx: int) -> None:
        if not (0 <= idx < len(self._bodies)):
            raise IndexError(
                f"body index {idx} out of range [0, {len(self._bodies)})"
            )

    def _build_body_container(self, device: wp.context.Device) -> BodyContainer:
        """Pack the descriptor list into a :class:`BodyContainer` via NumPy.

        Builds one NumPy array per field and uploads it. This is one
        host-to-device transfer per field, which dwarfs anything we
        could save by interleaving -- finalize() is a one-shot setup
        path, not a hot loop.
        """
        n = len(self._bodies)
        positions = np.zeros((n, 3), dtype=np.float32)
        orientations = np.zeros((n, 4), dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        angular_velocities = np.zeros((n, 3), dtype=np.float32)
        inverse_inertia = np.zeros((n, 3, 3), dtype=np.float32)
        inverse_inertia_world = np.zeros((n, 3, 3), dtype=np.float32)
        inverse_mass = np.zeros(n, dtype=np.float32)
        linear_damping = np.ones(n, dtype=np.float32)
        angular_damping = np.ones(n, dtype=np.float32)
        affected_by_gravity = np.ones(n, dtype=np.int32)
        motion_type = np.full(n, int(MOTION_STATIC), dtype=np.int32)

        for i, b in enumerate(self._bodies):
            positions[i] = b.position
            orientations[i] = b.orientation
            velocities[i] = b.velocity
            angular_velocities[i] = b.angular_velocity
            inverse_inertia[i] = b.inverse_inertia
            # Seed inverse_inertia_world with the body-frame value; the
            # first _update_bodies launch will rotate it into world
            # space using the actual orientation. Without this seed the
            # very first prepare_for_iteration sees zero effective mass.
            inverse_inertia_world[i] = b.inverse_inertia
            inverse_mass[i] = b.inverse_mass
            linear_damping[i] = b.linear_damping
            angular_damping[i] = b.angular_damping
            affected_by_gravity[i] = 1 if b.affected_by_gravity else 0
            motion_type[i] = int(b.motion_type)

        c = BodyContainer()
        c.position = wp.array(positions, dtype=wp.vec3f, device=device)
        c.velocity = wp.array(velocities, dtype=wp.vec3f, device=device)
        c.angular_velocity = wp.array(angular_velocities, dtype=wp.vec3f, device=device)
        c.orientation = wp.array(orientations, dtype=wp.quatf, device=device)
        c.inverse_inertia_world = wp.array(inverse_inertia_world, dtype=wp.mat33f, device=device)
        c.inverse_inertia = wp.array(inverse_inertia, dtype=wp.mat33f, device=device)
        c.inverse_mass = wp.array(inverse_mass, dtype=wp.float32, device=device)
        c.force = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.torque = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.delta_velocity = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.delta_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.linear_damping = wp.array(linear_damping, dtype=wp.float32, device=device)
        c.angular_damping = wp.array(angular_damping, dtype=wp.float32, device=device)
        c.affected_by_gravity = wp.array(affected_by_gravity, dtype=wp.int32, device=device)
        c.motion_type = wp.array(motion_type, dtype=wp.int32, device=device)
        # Single-world scenes: every body is in world 0. Multi-world
        # callers override this array after finalize() (stage 4 wires
        # per-body world assignment through the builder).
        c.world_id = wp.zeros(n, dtype=wp.int32, device=device)
        return c

    def _build_constraint_container(
        self,
        bodies: BodyContainer,
        device: wp.context.Device,
        max_contact_columns: int = 0,
    ) -> tuple[ConstraintContainer, int]:
        """Allocate the shared :class:`ConstraintContainer`, pack every
        constraint type into a contiguous cid range, and patch any
        outstanding :class:`HingeJointHandle` with their global cid.

        Layout:

            * cids ``[0, n_ball)``                              -> ball-sockets
            * cids ``[..., +n_dbs)``                            -> double-ball-sockets (5-DoF hinge lock)
            * cids ``[..., +n_dbs_pris)``                       -> double-ball-socket prismatics (5-DoF slide lock)
            * cids ``[..., +n_hinge)``                          -> hinge-angles
            * cids ``[..., +n_motor)``                          -> angular-motors
            * cids ``[..., +n_linear_motor)``                   -> linear-motors
            * cids ``[..., +n_hinge_joint)``                    -> fused hinge-joints
            * cids ``[..., +n_joint)``                          -> unified revolute/prismatic joints
            * cids ``[..., +n_prismatic)``                      -> legacy standalone prismatic joints
            * cids ``[..., +n_d6)``                             -> D6 (6-DoF generalised) joints

        The container's per-column dword count is ``max`` of all
        registered constraint schemas so any column can hold any type.
        The unused trailing rows of the smaller types are simply
        ignored by their kernels (per-type ``*_DWORDS`` constants only
        describe the populated prefix).
        """
        n_ball = len(self._ball_sockets)
        n_dbs = len(self._double_ball_sockets)
        n_dbs_pris = len(self._double_ball_socket_prismatics)
        n_hinge = len(self._hinge_angles)
        n_motor = len(self._angular_motors)
        n_linear_motor = len(self._linear_motors)
        n_angular_limit = len(self._angular_limits)
        n_linear_limit = len(self._linear_limits)
        n_hinge_joint = len(self._hinge_joint_descriptors)
        n_joint = len(self._joint_descriptors)
        n_prismatic = len(self._prismatic_descriptors)
        n_d6 = len(self._d6_descriptors)
        total = (
            n_ball
            + n_dbs
            + n_dbs_pris
            + n_hinge
            + n_motor
            + n_linear_motor
            + n_angular_limit
            + n_linear_limit
            + n_hinge_joint
            + n_joint
            + n_prismatic
            + n_d6
        )

        # All types share the column-major storage so the unified
        # dispatcher in solver_jitter_kernels can index any column the
        # same way; size to the widest schema.
        per_constraint_dwords = max(
            BS_DWORDS,
            DBS_DWORDS,
            DBS_PRISMATIC_DWORDS,
            HA_DWORDS,
            AM_DWORDS,
            LM_DWORDS,
            AL_DWORDS,
            LL_DWORDS,
            HJ_DWORDS,
            ADBS_DWORDS,
            PR_DWORDS,
            D6_DWORDS,
            CONTACT_DWORDS,
        )

        # Reserve trailing capacity for contact columns. They live at
        # cids [total, total + max_contact_columns) and are fully
        # overwritten by the ingest pipeline every
        # :meth:`World.step`; the initial zero-init doesn't matter
        # (except that constraint_type=0 stays outside any of the
        # live CONSTRAINT_TYPE_* tags so the dispatcher no-ops on
        # unused trailing cids).
        num_columns = max(1, total + int(max_contact_columns))
        container = constraint_container_zeros(
            num_constraints=num_columns,
            num_dwords=per_constraint_dwords,
            device=device,
        )

        ball_socket_offset = 0
        double_ball_socket_offset = n_ball
        double_ball_socket_prismatic_offset = n_ball + n_dbs
        hinge_angle_offset = n_ball + n_dbs + n_dbs_pris
        angular_motor_offset = n_ball + n_dbs + n_dbs_pris + n_hinge
        linear_motor_offset = n_ball + n_dbs + n_dbs_pris + n_hinge + n_motor
        angular_limit_offset = (
            n_ball + n_dbs + n_dbs_pris + n_hinge + n_motor + n_linear_motor
        )
        linear_limit_offset = angular_limit_offset + n_angular_limit
        hinge_joint_offset = linear_limit_offset + n_linear_limit
        joint_offset = hinge_joint_offset + n_hinge_joint
        prismatic_offset = joint_offset + n_joint
        d6_offset = prismatic_offset + n_prismatic

        if n_ball > 0:
            self._init_ball_sockets(container, bodies, ball_socket_offset, device)
        if n_dbs > 0:
            self._init_double_ball_sockets(
                container, bodies, double_ball_socket_offset, device
            )
        if n_dbs_pris > 0:
            self._init_double_ball_socket_prismatics(
                container, bodies, double_ball_socket_prismatic_offset, device
            )
        if n_hinge > 0:
            self._init_hinge_angles(container, bodies, hinge_angle_offset, device)
        if n_motor > 0:
            self._init_angular_motors(container, bodies, angular_motor_offset, device)
        if n_linear_motor > 0:
            self._init_linear_motors(container, bodies, linear_motor_offset, device)
        if n_angular_limit > 0:
            self._init_angular_limits(container, bodies, angular_limit_offset, device)
        if n_linear_limit > 0:
            self._init_linear_limits(container, bodies, linear_limit_offset, device)
        if n_hinge_joint > 0:
            self._init_hinge_joints(container, bodies, hinge_joint_offset, device)
        if n_joint > 0:
            self._init_joints(container, bodies, joint_offset, device)
        if n_prismatic > 0:
            self._init_prismatics(container, bodies, prismatic_offset, device)
        if n_d6 > 0:
            self._init_d6s(container, bodies, d6_offset, device)

        # Resolve fused-joint handles in place. Each handle was created
        # with cid=-1 and is parallel to its descriptor list; rewrite
        # to the joint's global cid so user-held references become
        # valid after finalize() returns.
        for i, h in enumerate(self._hinge_joint_handles):
            h.cid = hinge_joint_offset + i
        for i, h in enumerate(self._joint_handles):
            h.cid = joint_offset + i
        for i, h in enumerate(self._prismatic_handles):
            h.cid = prismatic_offset + i
        for i, h in enumerate(self._d6_handles):
            h.cid = d6_offset + i

        return container, total

    def _init_ball_sockets(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        n = len(self._ball_sockets)

        body1 = np.asarray([d.body1 for d in self._ball_sockets], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._ball_sockets], dtype=np.int32)
        anchor = np.asarray([d.anchor for d in self._ball_sockets], dtype=np.float32)
        hertz = np.asarray([d.hertz for d in self._ball_sockets], dtype=np.float32)
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._ball_sockets], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)

        wp.launch(
            ball_socket_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                hertz_d,
                damping_ratio_d,
            ],
            device=device,
        )

    def _init_double_ball_sockets(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the double-ball-socket descriptors into ``constraints``.

        Mirrors :meth:`_init_ball_sockets` but carries *two* world-space
        anchors per constraint (the hinge endpoints).
        """
        n = len(self._double_ball_sockets)

        body1 = np.asarray(
            [d.body1 for d in self._double_ball_sockets], dtype=np.int32
        )
        body2 = np.asarray(
            [d.body2 for d in self._double_ball_sockets], dtype=np.int32
        )
        anchor1 = np.asarray(
            [d.anchor1 for d in self._double_ball_sockets], dtype=np.float32
        )
        anchor2 = np.asarray(
            [d.anchor2 for d in self._double_ball_sockets], dtype=np.float32
        )
        hertz = np.asarray(
            [d.hertz for d in self._double_ball_sockets], dtype=np.float32
        )
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._double_ball_sockets], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)

        wp.launch(
            double_ball_socket_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor1_d,
                anchor2_d,
                hertz_d,
                damping_ratio_d,
            ],
            device=device,
        )

    def _init_double_ball_socket_prismatics(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack standalone prismatic-DBS descriptors into ``constraints``.

        Same shape as :meth:`_init_double_ball_sockets` -- the kernel
        itself auto-derives the third off-axis anchor used for the
        4+1 Schur rotation lock, so there is no extra per-descriptor
        input.
        """
        n = len(self._double_ball_socket_prismatics)

        body1 = np.asarray(
            [d.body1 for d in self._double_ball_socket_prismatics], dtype=np.int32
        )
        body2 = np.asarray(
            [d.body2 for d in self._double_ball_socket_prismatics], dtype=np.int32
        )
        anchor1 = np.asarray(
            [d.anchor1 for d in self._double_ball_socket_prismatics], dtype=np.float32
        )
        anchor2 = np.asarray(
            [d.anchor2 for d in self._double_ball_socket_prismatics], dtype=np.float32
        )
        hertz = np.asarray(
            [d.hertz for d in self._double_ball_socket_prismatics], dtype=np.float32
        )
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._double_ball_socket_prismatics],
            dtype=np.float32,
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)

        wp.launch(
            double_ball_socket_prismatic_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor1_d,
                anchor2_d,
                hertz_d,
                damping_ratio_d,
            ],
            device=device,
        )

    def _init_hinge_angles(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        n = len(self._hinge_angles)

        body1 = np.asarray([d.body1 for d in self._hinge_angles], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._hinge_angles], dtype=np.int32)
        axis = np.asarray([d.axis for d in self._hinge_angles], dtype=np.float32)
        min_angle = np.asarray([d.min_angle for d in self._hinge_angles], dtype=np.float32)
        max_angle = np.asarray([d.max_angle for d in self._hinge_angles], dtype=np.float32)
        hertz_lock = np.asarray(
            [d.hertz_lock for d in self._hinge_angles], dtype=np.float32
        )
        damping_ratio_lock = np.asarray(
            [d.damping_ratio_lock for d in self._hinge_angles], dtype=np.float32
        )
        hertz_limit = np.asarray(
            [d.hertz_limit for d in self._hinge_angles], dtype=np.float32
        )
        damping_ratio_limit = np.asarray(
            [d.damping_ratio_limit for d in self._hinge_angles], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        min_angle_d = wp.array(min_angle, dtype=wp.float32, device=device)
        max_angle_d = wp.array(max_angle, dtype=wp.float32, device=device)
        hertz_lock_d = wp.array(hertz_lock, dtype=wp.float32, device=device)
        damping_ratio_lock_d = wp.array(damping_ratio_lock, dtype=wp.float32, device=device)
        hertz_limit_d = wp.array(hertz_limit, dtype=wp.float32, device=device)
        damping_ratio_limit_d = wp.array(
            damping_ratio_limit, dtype=wp.float32, device=device
        )

        wp.launch(
            hinge_angle_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                axis_d,
                min_angle_d,
                max_angle_d,
                hertz_lock_d,
                damping_ratio_lock_d,
                hertz_limit_d,
                damping_ratio_limit_d,
            ],
            device=device,
        )

    def _init_angular_motors(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        n = len(self._angular_motors)

        body1 = np.asarray([d.body1 for d in self._angular_motors], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._angular_motors], dtype=np.int32)
        # Jitter's HingeJoint passes the same world-space axis to body1
        # and body2 (axis1 == axis2 == hinge_axis); mirror that here so
        # both bodies snapshot the same initial rotational reference.
        axis = np.asarray([d.axis for d in self._angular_motors], dtype=np.float32)
        target_velocity = np.asarray(
            [d.target_velocity for d in self._angular_motors], dtype=np.float32
        )
        max_force = np.asarray(
            [d.max_force for d in self._angular_motors], dtype=np.float32
        )
        hertz = np.asarray(
            [d.hertz for d in self._angular_motors], dtype=np.float32
        )
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._angular_motors], dtype=np.float32
        )
        target_angle = np.asarray(
            [d.target_angle for d in self._angular_motors], dtype=np.float32
        )
        stiffness = np.asarray(
            [d.stiffness for d in self._angular_motors], dtype=np.float32
        )
        damping = np.asarray(
            [d.damping for d in self._angular_motors], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis1_d = wp.array(axis, dtype=wp.vec3f, device=device)
        axis2_d = wp.array(axis, dtype=wp.vec3f, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_d = wp.array(max_force, dtype=wp.float32, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)
        target_angle_d = wp.array(target_angle, dtype=wp.float32, device=device)
        stiffness_d = wp.array(stiffness, dtype=wp.float32, device=device)
        damping_d = wp.array(damping, dtype=wp.float32, device=device)

        wp.launch(
            angular_motor_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                axis1_d,
                axis2_d,
                target_velocity_d,
                max_force_d,
                hertz_d,
                damping_ratio_d,
                target_angle_d,
                stiffness_d,
                damping_d,
            ],
            device=device,
        )

    def _init_linear_motors(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack linear-motor descriptors into ``constraints``.

        Translational twin of :meth:`_init_angular_motors`: the kernel
        snapshots a separate axis per body (same world axis here, so
        both bodies see the same reference direction), plus a pair of
        world-space anchors for the PD-path slide reconstruction.
        """
        n = len(self._linear_motors)

        body1 = np.asarray([d.body1 for d in self._linear_motors], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._linear_motors], dtype=np.int32)
        axis = np.asarray([d.axis for d in self._linear_motors], dtype=np.float32)
        anchor1 = np.asarray(
            [d.anchor1 for d in self._linear_motors], dtype=np.float32
        )
        anchor2 = np.asarray(
            [d.anchor2 for d in self._linear_motors], dtype=np.float32
        )
        target_velocity = np.asarray(
            [d.target_velocity for d in self._linear_motors], dtype=np.float32
        )
        max_force = np.asarray(
            [d.max_force for d in self._linear_motors], dtype=np.float32
        )
        hertz = np.asarray([d.hertz for d in self._linear_motors], dtype=np.float32)
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._linear_motors], dtype=np.float32
        )
        target_position = np.asarray(
            [d.target_position for d in self._linear_motors], dtype=np.float32
        )
        stiffness = np.asarray(
            [d.stiffness for d in self._linear_motors], dtype=np.float32
        )
        damping = np.asarray(
            [d.damping for d in self._linear_motors], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis1_d = wp.array(axis, dtype=wp.vec3f, device=device)
        axis2_d = wp.array(axis, dtype=wp.vec3f, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_d = wp.array(max_force, dtype=wp.float32, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)
        target_position_d = wp.array(target_position, dtype=wp.float32, device=device)
        stiffness_d = wp.array(stiffness, dtype=wp.float32, device=device)
        damping_d = wp.array(damping, dtype=wp.float32, device=device)

        wp.launch(
            linear_motor_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                axis1_d,
                axis2_d,
                anchor1_d,
                anchor2_d,
                target_velocity_d,
                max_force_d,
                hertz_d,
                damping_ratio_d,
                target_position_d,
                stiffness_d,
                damping_d,
            ],
            device=device,
        )

    def _init_angular_limits(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack angular-limit descriptors into ``constraints``.

        Mirrors :meth:`_init_angular_motors` but with the
        ``(min_value, max_value)`` / ``(hertz, damping_ratio,
        stiffness, damping)`` fields instead of the motor drive
        inputs.
        """
        n = len(self._angular_limits)

        body1 = np.asarray([d.body1 for d in self._angular_limits], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._angular_limits], dtype=np.int32)
        axis = np.asarray([d.axis for d in self._angular_limits], dtype=np.float32)
        min_value = np.asarray(
            [d.min_value for d in self._angular_limits], dtype=np.float32
        )
        max_value = np.asarray(
            [d.max_value for d in self._angular_limits], dtype=np.float32
        )
        hertz = np.asarray(
            [d.hertz for d in self._angular_limits], dtype=np.float32
        )
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._angular_limits], dtype=np.float32
        )
        stiffness = np.asarray(
            [d.stiffness for d in self._angular_limits], dtype=np.float32
        )
        damping = np.asarray(
            [d.damping for d in self._angular_limits], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis1_d = wp.array(axis, dtype=wp.vec3f, device=device)
        axis2_d = wp.array(axis, dtype=wp.vec3f, device=device)
        min_value_d = wp.array(min_value, dtype=wp.float32, device=device)
        max_value_d = wp.array(max_value, dtype=wp.float32, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)
        stiffness_d = wp.array(stiffness, dtype=wp.float32, device=device)
        damping_d = wp.array(damping, dtype=wp.float32, device=device)

        wp.launch(
            angular_limit_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                axis1_d,
                axis2_d,
                min_value_d,
                max_value_d,
                hertz_d,
                damping_ratio_d,
                stiffness_d,
                damping_d,
            ],
            device=device,
        )

    def _init_linear_limits(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack linear-limit descriptors into ``constraints``.

        Mirrors :meth:`_init_linear_motors` but drives the limit
        fields instead of the motor fields.
        """
        n = len(self._linear_limits)

        body1 = np.asarray([d.body1 for d in self._linear_limits], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._linear_limits], dtype=np.int32)
        axis = np.asarray([d.axis for d in self._linear_limits], dtype=np.float32)
        anchor1 = np.asarray(
            [d.anchor1 for d in self._linear_limits], dtype=np.float32
        )
        anchor2 = np.asarray(
            [d.anchor2 for d in self._linear_limits], dtype=np.float32
        )
        min_value = np.asarray(
            [d.min_value for d in self._linear_limits], dtype=np.float32
        )
        max_value = np.asarray(
            [d.max_value for d in self._linear_limits], dtype=np.float32
        )
        hertz = np.asarray(
            [d.hertz for d in self._linear_limits], dtype=np.float32
        )
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._linear_limits], dtype=np.float32
        )
        stiffness = np.asarray(
            [d.stiffness for d in self._linear_limits], dtype=np.float32
        )
        damping = np.asarray(
            [d.damping for d in self._linear_limits], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis1_d = wp.array(axis, dtype=wp.vec3f, device=device)
        axis2_d = wp.array(axis, dtype=wp.vec3f, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)
        min_value_d = wp.array(min_value, dtype=wp.float32, device=device)
        max_value_d = wp.array(max_value, dtype=wp.float32, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)
        stiffness_d = wp.array(stiffness, dtype=wp.float32, device=device)
        damping_d = wp.array(damping, dtype=wp.float32, device=device)

        wp.launch(
            linear_limit_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                axis1_d,
                axis2_d,
                anchor1_d,
                anchor2_d,
                min_value_d,
                max_value_d,
                hertz_d,
                damping_ratio_d,
                stiffness_d,
                damping_d,
            ],
            device=device,
        )

    def _init_hinge_joints(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the fused hinge-joint descriptors into ``constraints``.

        One launch of :func:`hinge_joint_initialize_kernel` writes all
        three sub-blocks (BallSocket / HingeAngle / AngularMotor) plus
        the shared header into each fused column.
        """
        n = len(self._hinge_joint_descriptors)

        body1 = np.asarray(
            [d.body1 for d in self._hinge_joint_descriptors], dtype=np.int32
        )
        body2 = np.asarray(
            [d.body2 for d in self._hinge_joint_descriptors], dtype=np.int32
        )
        anchor = np.asarray(
            [d.hinge_center for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        axis = np.asarray(
            [d.hinge_axis for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        min_angle = np.asarray(
            [d.min_angle for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        max_angle = np.asarray(
            [d.max_angle for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        target_velocity = np.asarray(
            [d.target_velocity for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        max_force = np.asarray(
            [d.max_force for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        hertz_linear = np.asarray(
            [d.hertz_linear for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_linear = np.asarray(
            [d.damping_ratio_linear for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        hertz_lock = np.asarray(
            [d.hertz_lock for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_lock = np.asarray(
            [d.damping_ratio_lock for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        hertz_limit = np.asarray(
            [d.hertz_limit for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_limit = np.asarray(
            [d.damping_ratio_limit for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        hertz_motor = np.asarray(
            [d.hertz_motor for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_motor = np.asarray(
            [d.damping_ratio_motor for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        min_angle_d = wp.array(min_angle, dtype=wp.float32, device=device)
        max_angle_d = wp.array(max_angle, dtype=wp.float32, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_d = wp.array(max_force, dtype=wp.float32, device=device)
        hertz_linear_d = wp.array(hertz_linear, dtype=wp.float32, device=device)
        damping_ratio_linear_d = wp.array(
            damping_ratio_linear, dtype=wp.float32, device=device
        )
        hertz_lock_d = wp.array(hertz_lock, dtype=wp.float32, device=device)
        damping_ratio_lock_d = wp.array(
            damping_ratio_lock, dtype=wp.float32, device=device
        )
        hertz_limit_d = wp.array(hertz_limit, dtype=wp.float32, device=device)
        damping_ratio_limit_d = wp.array(
            damping_ratio_limit, dtype=wp.float32, device=device
        )
        hertz_motor_d = wp.array(hertz_motor, dtype=wp.float32, device=device)
        damping_ratio_motor_d = wp.array(
            damping_ratio_motor, dtype=wp.float32, device=device
        )

        wp.launch(
            hinge_joint_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                axis_d,
                min_angle_d,
                max_angle_d,
                target_velocity_d,
                max_force_d,
                hertz_linear_d,
                damping_ratio_linear_d,
                hertz_lock_d,
                damping_ratio_lock_d,
                hertz_limit_d,
                damping_ratio_limit_d,
                hertz_motor_d,
                damping_ratio_motor_d,
            ],
            device=device,
        )

    def _init_joints(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the unified revolute/prismatic joint descriptors.

        Same ingestion path for both modes: the init kernel branches on
        the per-constraint ``joint_mode`` field, auto-deriving the
        prismatic third anchor from ``|anchor2 - anchor1|`` when
        needed. One launch of
        :func:`actuated_double_ball_socket_initialize_kernel` writes
        every joint column; a non-actuated joint is just one with
        ``drive_mode = DRIVE_MODE_OFF`` and
        ``min_value == max_value == 0``.
        """
        descs = self._joint_descriptors
        n = len(descs)

        body1 = np.asarray([d.body1 for d in descs], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in descs], dtype=np.int32)
        anchor1 = np.asarray([d.anchor1 for d in descs], dtype=np.float32)
        anchor2 = np.asarray([d.anchor2 for d in descs], dtype=np.float32)
        hertz = np.asarray([d.hertz for d in descs], dtype=np.float32)
        damping_ratio = np.asarray([d.damping_ratio for d in descs], dtype=np.float32)
        joint_mode = np.asarray([int(d.mode) for d in descs], dtype=np.int32)
        drive_mode = np.asarray([int(d.drive_mode) for d in descs], dtype=np.int32)
        target = np.asarray([d.target for d in descs], dtype=np.float32)
        target_velocity = np.asarray([d.target_velocity for d in descs], dtype=np.float32)
        max_force_drive = np.asarray([d.max_force_drive for d in descs], dtype=np.float32)
        stiffness_drive = np.asarray(
            [d.stiffness_drive for d in descs], dtype=np.float32
        )
        damping_drive = np.asarray(
            [d.damping_drive for d in descs], dtype=np.float32
        )
        min_value = np.asarray([d.min_value for d in descs], dtype=np.float32)
        max_value = np.asarray([d.max_value for d in descs], dtype=np.float32)
        hertz_limit = np.asarray([d.hertz_limit for d in descs], dtype=np.float32)
        damping_ratio_limit = np.asarray(
            [d.damping_ratio_limit for d in descs], dtype=np.float32
        )
        stiffness_limit = np.asarray(
            [d.stiffness_limit for d in descs], dtype=np.float32
        )
        damping_limit = np.asarray(
            [d.damping_limit for d in descs], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)
        joint_mode_d = wp.array(joint_mode, dtype=wp.int32, device=device)
        drive_mode_d = wp.array(drive_mode, dtype=wp.int32, device=device)
        target_d = wp.array(target, dtype=wp.float32, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_drive_d = wp.array(max_force_drive, dtype=wp.float32, device=device)
        stiffness_drive_d = wp.array(stiffness_drive, dtype=wp.float32, device=device)
        damping_drive_d = wp.array(damping_drive, dtype=wp.float32, device=device)
        min_value_d = wp.array(min_value, dtype=wp.float32, device=device)
        max_value_d = wp.array(max_value, dtype=wp.float32, device=device)
        hertz_limit_d = wp.array(hertz_limit, dtype=wp.float32, device=device)
        damping_ratio_limit_d = wp.array(damping_ratio_limit, dtype=wp.float32, device=device)
        stiffness_limit_d = wp.array(stiffness_limit, dtype=wp.float32, device=device)
        damping_limit_d = wp.array(damping_limit, dtype=wp.float32, device=device)

        wp.launch(
            actuated_double_ball_socket_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor1_d,
                anchor2_d,
                hertz_d,
                damping_ratio_d,
                joint_mode_d,
                drive_mode_d,
                target_d,
                target_velocity_d,
                max_force_drive_d,
                stiffness_drive_d,
                damping_drive_d,
                min_value_d,
                max_value_d,
                hertz_limit_d,
                damping_ratio_limit_d,
                stiffness_limit_d,
                damping_limit_d,
            ],
            device=device,
        )

    def _init_prismatics(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the legacy standalone prismatic descriptors into ``constraints``.

        One launch of :func:`prismatic_initialize_kernel` writes every
        column: snapshots the world-space anchor into each body's
        local frame plus the slide axis into body 1's local frame, and
        caches the rest-pose relative orientation ``q0 = q2^* * q1``
        used by the quaternion-error rotational rows.
        """
        descs = self._prismatic_descriptors
        n = len(descs)

        body1 = np.asarray([d.body1 for d in descs], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in descs], dtype=np.int32)
        anchor = np.asarray([d.anchor for d in descs], dtype=np.float32)
        axis = np.asarray([d.axis for d in descs], dtype=np.float32)
        hertz_angular = np.asarray(
            [d.hertz_angular for d in descs], dtype=np.float32
        )
        damping_ratio_angular = np.asarray(
            [d.damping_ratio_angular for d in descs], dtype=np.float32
        )
        hertz_linear = np.asarray(
            [d.hertz_linear for d in descs], dtype=np.float32
        )
        damping_ratio_linear = np.asarray(
            [d.damping_ratio_linear for d in descs], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        hertz_angular_d = wp.array(hertz_angular, dtype=wp.float32, device=device)
        damping_ratio_angular_d = wp.array(
            damping_ratio_angular, dtype=wp.float32, device=device
        )
        hertz_linear_d = wp.array(hertz_linear, dtype=wp.float32, device=device)
        damping_ratio_linear_d = wp.array(
            damping_ratio_linear, dtype=wp.float32, device=device
        )

        wp.launch(
            prismatic_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                axis_d,
                hertz_angular_d,
                damping_ratio_angular_d,
                hertz_linear_d,
                damping_ratio_linear_d,
            ],
            device=device,
        )

    def _init_d6s(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the D6 (6-DoF) descriptors into ``constraints``.

        One launch of :func:`d6_initialize_kernel` writes each D6
        column. Per-axis :class:`D6AxisDrive`s on the descriptor are
        un-tupled into 6 ``vec3f`` arrays per knob (hertz_ang,
        damping_ang, hertz_lin, damping_lin, max_force_ang,
        max_force_lin, target_position_*, target_velocity_*) so the
        kernel sees a flat per-cid layout. ``max_force = math.inf``
        becomes a finite "huge" cap (1e30) to keep the per-axis clamp
        well-defined inside the kernel without needing an explicit
        "no clamp" branch.
        """
        n = len(self._d6_descriptors)
        descriptors = self._d6_descriptors

        body1 = np.asarray([d.body1 for d in descriptors], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in descriptors], dtype=np.int32)
        anchor = np.asarray([d.anchor for d in descriptors], dtype=np.float32)

        # Per-axis knob arrays, one row per cid, three components per
        # row (the three body-1-local axes).
        target_position_ang = np.zeros((n, 3), dtype=np.float32)
        target_position_lin = np.zeros((n, 3), dtype=np.float32)
        target_velocity_ang = np.zeros((n, 3), dtype=np.float32)
        target_velocity_lin = np.zeros((n, 3), dtype=np.float32)
        hertz_ang = np.zeros((n, 3), dtype=np.float32)
        damping_ang = np.zeros((n, 3), dtype=np.float32)
        hertz_lin = np.zeros((n, 3), dtype=np.float32)
        damping_lin = np.zeros((n, 3), dtype=np.float32)
        max_force_ang = np.zeros((n, 3), dtype=np.float32)
        max_force_lin = np.zeros((n, 3), dtype=np.float32)

        # Cap "infinite" max_force at a large finite number so the
        # iterate-path clamp stays a well-defined float operation.
        # 1e30 is far above any realistic per-substep impulse but well
        # under float32's ~3.4e38 ceiling so multiplications inside
        # the kernel can't overflow.
        _HUGE_FORCE = 1.0e30

        for i, d in enumerate(descriptors):
            for k in range(3):
                a = d.angular[k]
                lin = d.linear[k]

                target_position_ang[i, k] = a.target_position
                target_position_lin[i, k] = lin.target_position
                target_velocity_ang[i, k] = a.target_velocity
                target_velocity_lin[i, k] = lin.target_velocity
                hertz_ang[i, k] = a.hertz
                damping_ang[i, k] = a.damping_ratio
                hertz_lin[i, k] = lin.hertz
                damping_lin[i, k] = lin.damping_ratio
                max_force_ang[i, k] = (
                    _HUGE_FORCE if math.isinf(a.max_force) else a.max_force
                )
                max_force_lin[i, k] = (
                    _HUGE_FORCE if math.isinf(lin.max_force) else lin.max_force
                )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        target_position_ang_d = wp.array(target_position_ang, dtype=wp.vec3f, device=device)
        target_position_lin_d = wp.array(target_position_lin, dtype=wp.vec3f, device=device)
        target_velocity_ang_d = wp.array(target_velocity_ang, dtype=wp.vec3f, device=device)
        target_velocity_lin_d = wp.array(target_velocity_lin, dtype=wp.vec3f, device=device)
        hertz_ang_d = wp.array(hertz_ang, dtype=wp.vec3f, device=device)
        damping_ang_d = wp.array(damping_ang, dtype=wp.vec3f, device=device)
        hertz_lin_d = wp.array(hertz_lin, dtype=wp.vec3f, device=device)
        damping_lin_d = wp.array(damping_lin, dtype=wp.vec3f, device=device)
        max_force_ang_d = wp.array(max_force_ang, dtype=wp.vec3f, device=device)
        max_force_lin_d = wp.array(max_force_lin, dtype=wp.vec3f, device=device)

        wp.launch(
            d6_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                target_position_ang_d,
                target_position_lin_d,
                target_velocity_ang_d,
                target_velocity_lin_d,
                hertz_ang_d,
                damping_ang_d,
                hertz_lin_d,
                damping_lin_d,
                max_force_ang_d,
                max_force_lin_d,
            ],
            device=device,
        )
