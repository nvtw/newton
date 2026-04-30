# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pose snapshot + XPBD finite-difference sync kernels.

The two kernels here form the pre/post bookends of any
position-level constraint pass:

* :func:`snapshot_pose_kernel` -- runs once at the start of the
  position pass; captures the parent body store's
  ``(position, orientation)`` into the pre-pass buffers so the sync
  kernel can later measure the position delta.
* :func:`sync_position_to_velocity_kernel` -- runs once at the end
  of the position pass; computes
  ``v = (p - p_pre) / dt`` and the angular-velocity log-map from
  the quaternion delta, writing the recovered values into the
  body's velocity arrays. Mirrors the C# ``Position -> Velocity``
  branch of ``SynchronizeVelAndPosStateUpdates`` in
  ``BodyTypes.cs:278-290``.

Both kernels take plain SoA arrays (no :class:`BodyContainer`
struct, no PhoenX-internal types) so they're equally usable from
:class:`PhoenXWorld`'s substep hook *and* from a standalone cloth
solver running its own substep loop.

Static / kinematic bodies should be excluded from the pass via a
host-side mask -- this module trusts the caller. Including a
kinematic body in the pass would compute a finite-difference
velocity that overrides whatever the kinematic scripting wrote
during the position phase, which is almost certainly a bug.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "snapshot_pose_kernel",
    "sync_position_to_velocity_kernel",
]


# ---------------------------------------------------------------------------
# Pre-pass snapshot.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def snapshot_pose_kernel(
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    out_position_pre_pass: wp.array[wp.vec3f],
    out_orientation_pre_pass: wp.array[wp.quatf],
):
    """Copy ``(position, orientation)`` into the pre-pass buffers.

    One thread per body. Cheap (two loads, two stores). Run once at
    the start of every position-level pass; the recovered velocity
    in the sync kernel measures motion *relative to* this snapshot.

    The two output arrays are caller-allocated and may be aliased to
    a different stride / device than the input, but in practice
    :class:`PositionPass` allocates them at the same shape as the
    input.
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= body_position.shape[0]:
        return
    out_position_pre_pass[rigid_body_index] = body_position[rigid_body_index]
    out_orientation_pre_pass[rigid_body_index] = body_orientation[rigid_body_index]


# ---------------------------------------------------------------------------
# Post-pass XPBD finite-difference sync.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sync_position_to_velocity_kernel(
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    position_pre_pass: wp.array[wp.vec3f],
    orientation_pre_pass: wp.array[wp.quatf],
    body_velocity: wp.array[wp.vec3f],
    body_angular_velocity: wp.array[wp.vec3f],
    inv_dt: wp.float32,
):
    """Recover velocity from the pose delta over the position pass.

    For each body, given the pose at the start of the pass
    ``(p_pre, q_pre)`` and the pose now ``(p, q)``:

    .. code::

        velocity         = (p - p_pre) * inv_dt
        delta_q          = q * conj(q_pre)
        angular_velocity = 2 * inv_dt * delta_q.xyz   if delta_q.w >= 0
                         = -2 * inv_dt * delta_q.xyz  otherwise   (shortest arc)

    The sign flip on negative ``delta_q.w`` selects the shortest-arc
    interpretation of the rotation: a quaternion and its negation
    represent the same rotation, but the imaginary part flips sign
    between them, so without the check we'd lose half the precision
    of the angular-velocity recovery on rotations near 180 degrees.

    Mirrors the ``PositionLevel -> VelocityLevel`` branch of
    ``SynchronizeVelAndPosStateUpdates`` in
    ``BodyTypes.cs:278-290``. One thread per body. Cheap: one quat
    multiply, one cross-product-free angular-velocity extraction.

    The kernel **overwrites** ``body_velocity`` /
    ``body_angular_velocity`` -- callers that want to *add* the
    XPBD-recovered velocity to an existing field should sum
    host-side or in a separate kernel. Overwrite is the right
    default because XPBD's whole pitch is "the position projection
    *is* the new state; velocity follows from it". Pre-existing
    velocity is a side effect of the previous substep and gets
    superseded.
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= body_position.shape[0]:
        return
    p = body_position[rigid_body_index]
    q = body_orientation[rigid_body_index]
    p_pre = position_pre_pass[rigid_body_index]
    q_pre = orientation_pre_pass[rigid_body_index]
    body_velocity[rigid_body_index] = (p - p_pre) * inv_dt
    delta_q = q * wp.quat_inverse(q_pre)
    ang_vel = wp.float32(2.0) * inv_dt * wp.vec3f(delta_q[0], delta_q[1], delta_q[2])
    if delta_q[3] < wp.float32(0.0):
        ang_vel = -ang_vel
    body_angular_velocity[rigid_body_index] = ang_vel
