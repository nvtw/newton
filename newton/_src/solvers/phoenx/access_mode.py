# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-entity access-mode synchronization helpers.

Port of Jitter2's ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
(``MassSplitting/TinyRigidState.cs``). Container-agnostic ``@wp.func``
helpers that lazily convert one entity between
:data:`ACCESS_MODE_VELOCITY_LEVEL` (velocity is authoritative; flip to
position integrates the substep-start snapshot forward by ``velocity *
dt``) and :data:`ACCESS_MODE_POSITION_LEVEL` (position is authoritative;
flip to velocity recovers ``velocity`` from the position delta vs
``position_prev_substep`` -- Macklin XPBD).
:data:`ACCESS_MODE_STATIC` skips sync (pinned / kinematic).
:data:`ACCESS_MODE_NONE` silently promotes to the requested mode.

Pure value-based functions: read the dual fields from the container,
call the helper, scatter the result back.
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


#: Uninitialised. Sync silently promotes to the new mode.
ACCESS_MODE_NONE: int = 0
#: Velocity-level integration: ``velocity`` is authoritative.
ACCESS_MODE_VELOCITY_LEVEL: int = 1
#: Position-level integration: ``position`` is authoritative.
ACCESS_MODE_POSITION_LEVEL: int = 2
#: Pinned / static / kinematic: sync is a no-op.
ACCESS_MODE_STATIC: int = 3


_ACCESS_MODE_NONE = wp.constant(wp.int32(ACCESS_MODE_NONE))
_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))
_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_ACCESS_MODE_STATIC = wp.constant(wp.int32(ACCESS_MODE_STATIC))


@wp.func
def integrate_orientation(
    q0: wp.quatf,
    angular_velocity: wp.vec3f,
    dt: wp.float32,
) -> wp.quatf:
    """First-order quaternion integration with renormalisation.

    ``q(t + dt) = normalize(q(t) + 0.5 * omega * q(t) * dt)``. Same form
    as Jitter2 ``QuaternionIntegrationHelper.ApplyAngularVelocity``.
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

    Direct port of ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
    (Jitter2 ``MassSplitting/TinyRigidState.cs:59-89``).

    Returns ``(position, orientation, velocity, angular_velocity,
    access_mode)`` 5-tuple by value.
    """
    if current_access_mode == new_access_mode:
        return position, orientation, velocity, angular_velocity, current_access_mode
    if current_access_mode == _ACCESS_MODE_STATIC:
        return position, orientation, velocity, angular_velocity, _ACCESS_MODE_STATIC
    if current_access_mode == _ACCESS_MODE_NONE:
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


@wp.func
def synchronize_position_velocity(
    position: wp.vec3f,
    velocity: wp.vec3f,
    position_prev_substep: wp.vec3f,
    current_access_mode: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Particle (3-DoF) variant of :func:`synchronize_pose_velocity`."""
    if current_access_mode == new_access_mode:
        return position, velocity, current_access_mode
    if current_access_mode == _ACCESS_MODE_STATIC:
        return position, velocity, _ACCESS_MODE_STATIC
    if current_access_mode == _ACCESS_MODE_NONE:
        return position, velocity, new_access_mode

    if new_access_mode == _ACCESS_MODE_VELOCITY_LEVEL and current_access_mode == _ACCESS_MODE_POSITION_LEVEL:
        new_velocity = (position - position_prev_substep) * inv_dt
        return position, new_velocity, new_access_mode

    if new_access_mode == _ACCESS_MODE_POSITION_LEVEL and current_access_mode == _ACCESS_MODE_VELOCITY_LEVEL:
        dt = wp.float32(1.0) / inv_dt
        new_position = position_prev_substep + dt * velocity
        return new_position, velocity, new_access_mode

    return position, velocity, new_access_mode
