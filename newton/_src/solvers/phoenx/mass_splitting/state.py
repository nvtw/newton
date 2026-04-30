# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-(body, partition) velocity-state copy used by mass splitting.

Mirrors the C# ``TinyRigidState`` struct in
``PhoenX/CudaKernels/Common/BodyTypes.cs`` (lines 176-378). Each
copy carries enough of a body's instantaneous state for one
partition's constraint kernel to read / mutate / write back without
racing on the shared body store.

Layout matches C# field-for-field so future extensions
(``SetAccessMode`` regime switches between velocity-level and
position-level integration) port cleanly. The
:class:`AccessMode` enum is exposed as plain integer constants so
the same numeric value works on host and inside Warp kernels.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "ACCESS_MODE_NONE",
    "ACCESS_MODE_POSITION_LEVEL",
    "ACCESS_MODE_STATIC_BODY",
    "ACCESS_MODE_VELOCITY_LEVEL",
    "TinyRigidState",
    "tiny_rigid_state_from_body",
    "tiny_rigid_state_set_access_mode",
    "tiny_rigid_state_synchronize",
    "tiny_rigid_state_write_back",
]


# ---------------------------------------------------------------------------
# Access modes -- mirrors the C# ``ConstraintAccessMode`` enum verbatim.
# ---------------------------------------------------------------------------

#: No access mode set yet; the state has not been initialised by
#: :func:`tiny_rigid_state_from_body`. C# ``ConstraintAccessMode.None``.
ACCESS_MODE_NONE: int = 0

#: Velocity-level integration: the constraint solve operates on
#: ``Velocity`` / ``AngularVelocity`` and integrates ``Position`` /
#: ``Orientation`` lazily on access-mode switch. Default mode after
#: ``broadcast_rigid_to_copy_states``.
ACCESS_MODE_VELOCITY_LEVEL: int = 1

#: Position-level integration (XPBD-style): the constraint solve
#: operates on ``Position`` / ``Orientation`` directly; switching back
#: to velocity level recovers ``Velocity`` from the position delta
#: (Macklin-style :math:`\\dot p = (p - p_0) / \\Delta t`).
ACCESS_MODE_POSITION_LEVEL: int = 2

#: Body is static / pinned: no copy state exists for it; the
#: :func:`read_state` wrapper falls back to the body-store value
#: directly. ``GetRigidStateIndex`` returns -1 for these bodies.
ACCESS_MODE_STATIC_BODY: int = 3


_ACCESS_MODE_NONE = wp.constant(wp.int32(ACCESS_MODE_NONE))
_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))
_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_ACCESS_MODE_STATIC_BODY = wp.constant(wp.int32(ACCESS_MODE_STATIC_BODY))


# ---------------------------------------------------------------------------
# TinyRigidState -- per-(body, partition) state copy.
# ---------------------------------------------------------------------------


@wp.struct
class TinyRigidState:
    """Velocity-state copy for one ``(body, partition)`` pair.

    Lives in ``InteractionGraphData.tiny_states`` (a
    ``wp.array[TinyRigidState]``); the
    :func:`~newton._src.solvers.phoenx.mass_splitting.read_state.read_state`
    wrapper hands it to constraint kernels as a value type.

    Fields mirror the C# struct (BodyTypes.cs:176-244) byte-for-byte
    intent so a future binary interop with the C# reference solver
    stays plausible:

    * ``orientation`` -- body-frame -> world rotation [quat].
    * ``position``    -- center-of-mass position [m].
    * ``velocity``    -- COM linear velocity [m/s].
    * ``angular_velocity`` -- world-frame angular velocity [rad/s].
    * ``access_mode`` -- :data:`ACCESS_MODE_NONE` /
      :data:`ACCESS_MODE_VELOCITY_LEVEL` /
      :data:`ACCESS_MODE_POSITION_LEVEL` /
      :data:`ACCESS_MODE_STATIC_BODY`. Selects how
      :func:`tiny_rigid_state_synchronize` integrates between
      regimes.

    The Newton body container is SoA (separate ``position``,
    ``orientation``, ``velocity`` arrays); this struct is AoS to
    match C#'s ``Ptr<TinyRigidState>`` layout. Construction at
    substep boundaries packs the SoA fields into the AoS slot and
    write-back unpacks them.
    """

    orientation: wp.quatf
    position: wp.vec3f
    velocity: wp.vec3f
    angular_velocity: wp.vec3f
    access_mode: wp.int32


# ---------------------------------------------------------------------------
# State construction / write-back helpers.
# ---------------------------------------------------------------------------
#
# Free ``@wp.func`` helpers rather than struct methods because Warp
# 1.13 doesn't support member functions on ``@wp.struct`` types
# (everything has to be a free function). Same calling convention,
# semantics carried verbatim from the C# constructor / WriteBack /
# SetAccessMode methods.


@wp.func
def tiny_rigid_state_from_body(
    src_position: wp.vec3f,
    src_orientation: wp.quatf,
    src_velocity: wp.vec3f,
    src_angular_velocity: wp.vec3f,
    dt: wp.float32,
) -> TinyRigidState:
    """Construct a copy state from a body's current pose + velocity.

    Mirrors the C# ``TinyRigidState(ref TinyRigidState source, float dt)``
    constructor (BodyTypes.cs:233-244): velocities pass through
    unchanged, then position is integrated forward by ``dt`` and
    orientation by ``angular_velocity * dt`` so the per-partition
    copy starts the iteration from the *predicted* (post-integration)
    pose. ``access_mode`` is set to :data:`ACCESS_MODE_VELOCITY_LEVEL`
    -- the default regime after broadcast.

    Used by
    :func:`~newton._src.solvers.phoenx.mass_splitting.kernels.broadcast_rigid_to_copy_states`
    inside the per-body fan-out loop.
    """
    new_position = src_position + dt * src_velocity
    new_orientation = _integrate_orientation(src_orientation, src_angular_velocity, dt)
    state = TinyRigidState()
    state.orientation = new_orientation
    state.position = new_position
    state.velocity = src_velocity
    state.angular_velocity = src_angular_velocity
    state.access_mode = _ACCESS_MODE_VELOCITY_LEVEL
    return state


@wp.func
def tiny_rigid_state_synchronize(
    state: TinyRigidState,
    new_access_mode: wp.int32,
    body_position: wp.vec3f,
    body_orientation: wp.quatf,
    inv_dt: wp.float32,
) -> TinyRigidState:
    """Switch a state's access mode, integrating the dual fields.

    Mirrors the C# ``SynchronizeVelAndPosStateUpdates``
    (BodyTypes.cs:268-304):

    * ``Position -> Velocity``: recover ``velocity`` and
      ``angular_velocity`` from the position-level delta
      (Macklin-style ``v = (p - p_body) / dt``,
      ``omega = 2 * inv_dt * Im(q * conj(q_body))``).
    * ``Velocity -> Position``: integrate the body pose forward by
      ``dt = 1 / inv_dt`` using the current ``velocity`` /
      ``angular_velocity``.

    Returns the updated state by value. ``body_position`` /
    ``body_orientation`` are the parent body's pose at the start of
    the substep -- the reference frame the integration is relative
    to. No-op when ``new_access_mode == state.access_mode``.
    """
    if state.access_mode == new_access_mode:
        return state
    out = state
    if new_access_mode == _ACCESS_MODE_VELOCITY_LEVEL and state.access_mode == _ACCESS_MODE_POSITION_LEVEL:
        # Algorithm 2 in Macklin et al. "Detailed Rigid Body Simulation
        # with Extended Position Based Dynamics" (2020).
        out.velocity = (state.position - body_position) * inv_dt
        delta_q = state.orientation * wp.quat_inverse(body_orientation)
        out.angular_velocity = wp.float32(2.0) * inv_dt * wp.vec3f(delta_q[0], delta_q[1], delta_q[2])
        if delta_q[3] < wp.float32(0.0):
            out.angular_velocity = -out.angular_velocity
        out.access_mode = new_access_mode
        return out
    if new_access_mode == _ACCESS_MODE_POSITION_LEVEL and state.access_mode == _ACCESS_MODE_VELOCITY_LEVEL:
        dt = wp.float32(1.0) / inv_dt
        out.position = body_position + dt * state.velocity
        out.orientation = _integrate_orientation(body_orientation, state.angular_velocity, dt)
        out.access_mode = new_access_mode
        return out
    # Other transitions (NONE / STATIC_BODY) are no-ops on the
    # field set; just record the new mode.
    out.access_mode = new_access_mode
    return out


@wp.func
def tiny_rigid_state_set_access_mode(
    state: TinyRigidState,
    new_access_mode: wp.int32,
    body_position: wp.vec3f,
    body_orientation: wp.quatf,
    inv_dt: wp.float32,
) -> TinyRigidState:
    """Alias for :func:`tiny_rigid_state_synchronize`.

    The C# code keeps ``SetAccessMode`` as a thin wrapper around the
    sync helper (BodyTypes.cs:322-325); this Python port keeps both
    names so call sites can choose the more readable one.
    """
    return tiny_rigid_state_synchronize(
        state,
        new_access_mode,
        body_position,
        body_orientation,
        inv_dt,
    )


@wp.func
def tiny_rigid_state_write_back(
    state: TinyRigidState,
    body_position: wp.vec3f,
    body_orientation: wp.quatf,
    inv_dt: wp.float32,
):
    """Bring a copy state into velocity-level form for write-back.

    Mirrors the C# ``WriteBack`` (BodyTypes.cs:309-317): force the
    state to :data:`ACCESS_MODE_VELOCITY_LEVEL` (which the
    ``Position -> Velocity`` branch of
    :func:`tiny_rigid_state_synchronize` does) and return the
    resulting ``(velocity, angular_velocity)`` pair. The kernel
    caller scatters those back into the SoA body store.

    Returns ``(velocity, angular_velocity)`` so the caller can pick
    them off without indexing into the struct -- Warp's struct field
    access works at runtime but is harder to read at the call site.
    """
    synced = tiny_rigid_state_synchronize(
        state,
        _ACCESS_MODE_VELOCITY_LEVEL,
        body_position,
        body_orientation,
        inv_dt,
    )
    return synced.velocity, synced.angular_velocity


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


@wp.func
def _integrate_orientation(
    q0: wp.quatf,
    angular_velocity: wp.vec3f,
    dt: wp.float32,
) -> wp.quatf:
    """First-order quaternion integration with renormalisation.

    ``q(t + dt) = normalize(q(t) + 0.5 * w(t) * q(t) * dt)``. Same
    formulation as the C# ``QuaternionIntegrationHelper.ApplyAngularVelocity``;
    the small-angle approximation is fine here because the substep
    is short.
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
