# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-entity access-mode synchronization helpers.

Direct port of Jitter2's ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
(``MassSplitting/TinyRigidState.cs``). PhoenX stores rigid bodies and
particles in struct-of-arrays containers; this module exposes
container-agnostic ``@wp.func`` helpers that constraint kernels call to
*lazily* convert one entity (body or particle) between velocity-level
and position-level integration regimes inside a single substep.

## Mental model (matches ``TinyRigidState.cs`` line-for-line)

Each entity carries an ``access_mode`` int that tracks which dual is
"live":

* :data:`ACCESS_MODE_VELOCITY_LEVEL` -- ``velocity`` /
  ``angular_velocity`` are authoritative; reading ``position`` /
  ``orientation`` returns the substep-start snapshot
  (``position_prev_substep`` / ``orientation_prev_substep``).
* :data:`ACCESS_MODE_POSITION_LEVEL` -- ``position`` / ``orientation``
  are authoritative; reading ``velocity`` / ``angular_velocity``
  requires the finite-diff against the substep-start snapshot.
* :data:`ACCESS_MODE_STATIC` -- the entity is pinned; sync is a no-op.
* :data:`ACCESS_MODE_NONE` -- uninitialised; sync is a no-op (matches
  the C# default-constructed state).

The integer values match
:mod:`newton._src.solvers.phoenx.mass_splitting.state` so the same
tag flows through a future mass-splitting integration without
translation.

## API surface

* :func:`synchronize_pose_velocity` -- the full 6-DoF math (positions
  + orientations + velocities + angular velocities). Mirrors
  ``TinyRigidState.SynchronizeVelAndPosStateUpdates`` exactly.
  Returns the post-sync ``(position, orientation, velocity,
  angular_velocity, access_mode)`` 5-tuple by value; the caller
  scatters the relevant fields back into its SoA container.
* :func:`synchronize_position_velocity` -- the 3-DoF particle case
  (no orientation / angular velocity). Returns
  ``(position, velocity, access_mode)``.
* :func:`integrate_orientation` -- first-order quaternion update with
  renormalisation; the ``Velocity -> Position`` branch shares this
  with the rigid integrator.

The helpers are **value-based** rather than container-method-based so
the same primitives serve the body SoA, the particle SoA, and the
``TinyRigidState`` struct used by the (currently dormant) mass-
splitting subsystem. SoA call sites read out the dual fields, call
the helper, and scatter the result back -- the few extra local
variables Warp lifts into registers.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "ACCESS_MODE_NONE",
    "ACCESS_MODE_POSITION_LEVEL",
    "ACCESS_MODE_STATIC",
    "ACCESS_MODE_VELOCITY_LEVEL",
    "integrate_orientation",
    "synchronize_pose_velocity",
    "synchronize_position_velocity",
]


# ---------------------------------------------------------------------------
# Access-mode integer constants. Mirror ``ConstraintAccessMode`` in
# Jitter2 (``MassSplitting/TinyRigidState.cs:115``) and the parallel
# constants in :mod:`mass_splitting.state` byte-for-byte.
# ---------------------------------------------------------------------------

#: Uninitialised. ``synchronize_*`` no-ops on this mode. Matches the C#
#: ``ConstraintAccessMode.None`` value (0).
ACCESS_MODE_NONE: int = 0

#: Velocity-level integration: ``velocity`` / ``angular_velocity`` are
#: authoritative. Default after the substep-entry snapshot. Matches
#: ``ConstraintAccessMode.VelocityLevel`` (1).
ACCESS_MODE_VELOCITY_LEVEL: int = 1

#: Position-level integration (XPBD-style): ``position`` /
#: ``orientation`` are authoritative; switching back to velocity-level
#: recovers ``velocity`` from the position delta against the
#: substep-start snapshot. Matches ``ConstraintAccessMode.PositionLevel``
#: (2).
ACCESS_MODE_POSITION_LEVEL: int = 2

#: Pinned / static entity: no integration in either regime. Set on
#: bodies with ``inverse_mass == 0`` or ``motion_type != DYNAMIC`` and
#: on particles with ``inverse_mass == 0`` so the synchronize helpers
#: can short-circuit per-entity rather than at every call site.
ACCESS_MODE_STATIC: int = 3


_ACCESS_MODE_NONE = wp.constant(wp.int32(ACCESS_MODE_NONE))
_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))
_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_ACCESS_MODE_STATIC = wp.constant(wp.int32(ACCESS_MODE_STATIC))


# ---------------------------------------------------------------------------
# Quaternion integration helper (shared between rigid integrator and the
# Velocity -> Position branch).
# ---------------------------------------------------------------------------


@wp.func
def integrate_orientation(
    q0: wp.quatf,
    angular_velocity: wp.vec3f,
    dt: wp.float32,
) -> wp.quatf:
    """First-order quaternion integration with renormalisation.

    ``q(t + dt) = normalize(q(t) + 0.5 * w(t) * q(t) * dt)``. Same
    formulation as Jitter2's ``QuaternionIntegrationHelper.ApplyAngularVelocity``;
    the small-angle approximation is fine over a single substep.
    """
    omega_q = wp.quatf(angular_velocity[0], angular_velocity[1], angular_velocity[2], wp.float32(0.0))
    q_dot = wp.float32(0.5) * (omega_q * q0)
    q_new = wp.quatf(
        q0[0] + q_dot[0] * dt,
        q0[1] + q_dot[1] * dt,
        q0[2] + q_dot[2] * dt,
        q0[3] + q_dot[3] * dt,
    )
    return wp.normalize(q_new)


# ---------------------------------------------------------------------------
# Full 6-DoF synchronize -- the rigid-body case.
# ---------------------------------------------------------------------------


@wp.func
def synchronize_pose_velocity(
    position: wp.vec3f,
    orientation: wp.quatf,
    velocity: wp.vec3f,
    angular_velocity: wp.vec3f,
    position_prev_substep: wp.vec3f,
    orientation_prev_substep: wp.quatf,
    current_access_mode: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Switch a rigid body's access mode, integrating the dual fields.

    Direct port of Jitter2's ``SynchronizeVelAndPosStateUpdates``
    (``TinyRigidState.cs:59-89``). ``position`` / ``orientation`` are
    the *live* (per-iterate) values; ``position_prev_substep`` /
    ``orientation_prev_substep`` are the substep-start snapshot
    (Jitter2's ``body.Position`` / ``body.Orientation``). The two
    branches:

    * ``Position -> Velocity``: recover ``velocity`` /
      ``angular_velocity`` from the position-level delta
      (Macklin-style ``v = (p - p_prev) / dt``,
      ``omega = 2 * inv_dt * Im(q * conj(q_prev))``).
    * ``Velocity -> Position``: integrate the substep-start pose
      forward by ``dt = 1 / inv_dt`` using the current ``velocity`` /
      ``angular_velocity`` and overwrite ``position`` /
      ``orientation``.

    No-op when ``new_access_mode == current_access_mode`` or when
    ``current_access_mode`` is :data:`ACCESS_MODE_NONE` /
    :data:`ACCESS_MODE_STATIC`.

    Returns:
        ``(position, orientation, velocity, angular_velocity,
        access_mode)`` 5-tuple by value. The caller scatters whichever
        fields it owns back into its SoA container (the body branch
        scatters all four; the particle branch ignores
        ``orientation`` / ``angular_velocity``).
    """
    if current_access_mode == new_access_mode:
        return position, orientation, velocity, angular_velocity, current_access_mode
    if current_access_mode == _ACCESS_MODE_STATIC or current_access_mode == _ACCESS_MODE_NONE:
        return position, orientation, velocity, angular_velocity, new_access_mode

    if new_access_mode == _ACCESS_MODE_VELOCITY_LEVEL and current_access_mode == _ACCESS_MODE_POSITION_LEVEL:
        new_velocity = (position - position_prev_substep) * inv_dt
        delta_q = orientation * wp.quat_inverse(orientation_prev_substep)
        new_angular_velocity = wp.float32(2.0) * inv_dt * wp.vec3f(delta_q[0], delta_q[1], delta_q[2])
        if delta_q[3] < wp.float32(0.0):
            new_angular_velocity = -new_angular_velocity
        return position, orientation, new_velocity, new_angular_velocity, new_access_mode

    if new_access_mode == _ACCESS_MODE_POSITION_LEVEL and current_access_mode == _ACCESS_MODE_VELOCITY_LEVEL:
        dt = wp.float32(1.0) / inv_dt
        new_position = position_prev_substep + dt * velocity
        new_orientation = integrate_orientation(orientation_prev_substep, angular_velocity, dt)
        return new_position, new_orientation, velocity, angular_velocity, new_access_mode

    return position, orientation, velocity, angular_velocity, new_access_mode


# ---------------------------------------------------------------------------
# 3-DoF synchronize -- the particle case.
# ---------------------------------------------------------------------------


@wp.func
def synchronize_position_velocity(
    position: wp.vec3f,
    velocity: wp.vec3f,
    position_prev_substep: wp.vec3f,
    current_access_mode: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Particle (3-DoF) variant of :func:`synchronize_pose_velocity`.

    Same control flow without the orientation / angular-velocity
    arms. Returns ``(position, velocity, access_mode)``.
    """
    if current_access_mode == new_access_mode:
        return position, velocity, current_access_mode
    if current_access_mode == _ACCESS_MODE_STATIC or current_access_mode == _ACCESS_MODE_NONE:
        return position, velocity, new_access_mode

    if new_access_mode == _ACCESS_MODE_VELOCITY_LEVEL and current_access_mode == _ACCESS_MODE_POSITION_LEVEL:
        new_velocity = (position - position_prev_substep) * inv_dt
        return position, new_velocity, new_access_mode

    if new_access_mode == _ACCESS_MODE_POSITION_LEVEL and current_access_mode == _ACCESS_MODE_VELOCITY_LEVEL:
        dt = wp.float32(1.0) / inv_dt
        new_position = position_prev_substep + dt * velocity
        return new_position, velocity, new_access_mode

    return position, velocity, new_access_mode
