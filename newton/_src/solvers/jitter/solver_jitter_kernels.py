# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels and ``@wp.func`` helpers for the Jitter solver.

Kept separate from :mod:`solver_jitter` so the driver class file stays
readable; this module is pure GPU code (no Python control flow). Symbols
are re-exported from :mod:`solver_jitter` so callers don't need to know
about the split.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.jitter.constraints import BallSocketData
from newton._src.solvers.jitter.graph_coloring_common import (
    ElementInteractionData,
    element_interaction_data_make,
)

__all__ = [
    "_ball_socket_to_element_kernel",
    "_integrate_forces_kernel",
    "_integrate_velocities_kernel",
    "_rotation_quaternion",
    "_update_bodies_kernel",
    "pack_body_xforms_kernel",
]


@wp.kernel
def _ball_socket_to_element_kernel(
    constraints: wp.array[BallSocketData],
    num_constraints: wp.array[int],
    elements: wp.array[ElementInteractionData],
):
    """Project each :class:`BallSocketData` into the partitioner's
    :class:`ElementInteractionData` view: only the two body indices matter,
    the remaining slots are filled with ``-1``."""
    tid = wp.tid()
    if tid >= num_constraints[0]:
        return
    c = constraints[tid]
    elements[tid] = element_interaction_data_make(c.body1, c.body2, -1, -1, -1, -1, -1, -1)


@wp.func
def _rotation_quaternion(omega: wp.vec3f, dt: wp.float32) -> wp.quatf:
    """Build the rotation quaternion for ``omega * dt`` (axis-angle form).

    Mirrors Jitter2's ``MathHelper.RotationQuaternion``. The axis-angle
    form keeps unit norm by construction, which is significantly more
    stable across many sub-steps than the linearised
    ``q' = 0.5 * (omega, 0) * q`` derivative -- the linearised form
    grows the quaternion magnitude every step and relies on renormalising
    each frame to compensate.
    """
    omega_len = wp.length(omega)
    theta = omega_len * dt
    if theta < 1.0e-9:
        return wp.quatf(0.0, 0.0, 0.0, 1.0)
    half = theta * 0.5
    # axis * sin(half) = omega / |omega| * sin(half)
    s = wp.sin(half) / omega_len
    return wp.quatf(omega[0] * s, omega[1] * s, omega[2] * s, wp.cos(half))


@wp.kernel
def _update_bodies_kernel(
    bodies: BodyContainer,
    gravity: wp.vec3f,
    substep_dt: wp.float32,
):
    """Mirrors Jitter2's ``RigidBody.Update`` (called once per *step*).

    For each dynamic body:
      * Apply per-body damping to ``velocity`` / ``angular_velocity``.
      * Build ``delta_velocity`` and ``delta_angular_velocity`` from the
        accumulated ``force`` / ``torque`` plus optional gravity, scaled
        by ``substep_dt``. The substep loop's per-substep
        ``_integrate_forces`` then just adds these cached deltas (Jitter
        splits the work this way so the per-substep path is a single
        vector add).
      * Zero ``force`` / ``torque`` so the next step starts clean.
      * Refresh ``inverse_inertia_world = R * inverse_inertia * R^T`` from
        the current orientation so the constraint solver's effective-mass
        terms see the rotated inertia.

    Static bodies are skipped (their inertia / mass are already zero by
    construction).
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return

    v = bodies.velocity[i] * bodies.linear_damping[i]
    w = bodies.angular_velocity[i] * bodies.angular_damping[i]
    bodies.velocity[i] = v
    bodies.angular_velocity[i] = w

    inv_mass = bodies.inverse_mass[i]
    inv_inertia_world = bodies.inverse_inertia_world[i]
    f = bodies.force[i]
    t = bodies.torque[i]

    dv = f * (inv_mass * substep_dt)
    dw = (inv_inertia_world * t) * substep_dt
    if bodies.affected_by_gravity[i] != 0:
        dv = dv + gravity * substep_dt
    bodies.delta_velocity[i] = dv
    bodies.delta_angular_velocity[i] = dw

    bodies.force[i] = wp.vec3f(0.0, 0.0, 0.0)
    bodies.torque[i] = wp.vec3f(0.0, 0.0, 0.0)

    # InverseInertiaWorld = R * inverse_inertia * R^T, with R from the
    # current orientation. We use wp.quat_to_matrix so we don't depend on
    # any helper outside warp.
    r = wp.quat_to_matrix(bodies.orientation[i])
    bodies.inverse_inertia_world[i] = r * bodies.inverse_inertia[i] * wp.transpose(r)


@wp.kernel
def _integrate_forces_kernel(bodies: BodyContainer):
    """Mirrors Jitter2's per-substep ``IntegrateForces``: just add the
    cached deltas built once per step in :func:`_update_bodies_kernel`.

    Static bodies are skipped. Kinematic bodies are skipped as well --
    Jitter's ``IntegrateForces`` only advances dynamic bodies' velocities;
    kinematic velocities are user-scripted and only feed
    ``IntegrateVelocities``.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC:
        return
    bodies.velocity[i] = bodies.velocity[i] + bodies.delta_velocity[i]
    bodies.angular_velocity[i] = bodies.angular_velocity[i] + bodies.delta_angular_velocity[i]


@wp.kernel
def _integrate_velocities_kernel(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Mirrors Jitter2's IntegrateVelocities: advance position + orientation.

    Static bodies are skipped (kinematic bodies *do* advance, since their
    user-scripted velocities are meaningful; matches Jitter, which gates
    on ``MotionType != Static``).
    Orientation update uses the axis-angle rotation quaternion form (see
    :func:`_rotation_quaternion`) to stay numerically stable for large
    angular velocities.
    """
    i = wp.tid()
    if bodies.motion_type[i] == MOTION_STATIC:
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt

    q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
    bodies.orientation[i] = wp.normalize(q_rot * bodies.orientation[i])


@wp.kernel
def pack_body_xforms_kernel(
    bodies: BodyContainer,
    xforms: wp.array[wp.transform],
):
    """Pack ``(position, orientation)`` from a :class:`BodyContainer` into a
    flat :class:`wp.transform` array suitable for ``viewer.log_shapes``.
    Exposed at module scope so examples can render a Jitter ``World`` with
    the standard Newton viewer without writing their own kernel."""
    i = wp.tid()
    xforms[i] = wp.transform(bodies.position[i], bodies.orientation[i])
