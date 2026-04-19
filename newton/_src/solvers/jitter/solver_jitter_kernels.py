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
from newton._src.solvers.jitter.constraint_angular_motor import (
    angular_motor_iterate,
    angular_motor_prepare_for_iteration,
    angular_motor_world_wrench,
)
from newton._src.solvers.jitter.constraint_ball_socket import (
    ball_socket_iterate,
    ball_socket_prepare_for_iteration,
    ball_socket_world_wrench,
)
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_ANGULAR_MOTOR,
    CONSTRAINT_TYPE_BALL_SOCKET,
    CONSTRAINT_TYPE_HINGE_ANGLE,
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
    constraint_get_type,
)
from newton._src.solvers.jitter.constraint_hinge_angle import (
    hinge_angle_iterate,
    hinge_angle_prepare_for_iteration,
    hinge_angle_world_wrench,
)
from newton._src.solvers.jitter.graph_coloring_common import (
    ElementInteractionData,
    element_interaction_data_make,
)

__all__ = [
    "_constraint_gather_wrenches_kernel",
    "_constraint_iterate_kernel",
    "_constraint_prepare_for_iteration_kernel",
    "_constraints_to_elements_kernel",
    "_integrate_forces_kernel",
    "_integrate_velocities_kernel",
    "_rotation_quaternion",
    "_update_bodies_kernel",
    "pack_body_xforms_kernel",
]


# ---------------------------------------------------------------------------
# Unified constraint dispatch
# ---------------------------------------------------------------------------
#
# The solver no longer has per-type prepare/iterate kernels. Instead one
# pair of *type-agnostic* dispatcher kernels reads the constraint_type
# tag at the front of every column and routes to the correct ``wp.func``
# via an ``if/elif`` cascade. Each branch compiles to a tight inlined
# call (Warp inlines ``wp.func`` calls aggressively); the cascade adds
# one int compare per branch in the worst case which is far cheaper
# than the per-launch overhead of having one kernel per type.
#
# The build-time wiring is therefore:
#
#   * solver -> ``_constraint_prepare_for_iteration_kernel`` /
#               ``_constraint_iterate_kernel``
#   * dispatch kernel -> ``constraint_get_type(c, cid)`` -> per-type ``wp.func``
#
# Adding a new constraint type means: write its ``*_prepare_for_iteration``
# / ``*_iterate`` ``wp.func`` (same ``constraints, cid, bodies, idt``
# signature), add a ``CONSTRAINT_TYPE_FOO`` tag in
# :mod:`constraint_container`, stamp it in the type's init kernel, and
# add one ``elif`` branch to each dispatcher below. No change to
# :class:`World` or :class:`WorldBuilder` needed.


@wp.kernel
def _constraint_prepare_for_iteration_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    partition_element_ids: wp.array[wp.int32],
    partition_count: wp.array[wp.int32],
):
    """Dispatch the per-type ``prepare_for_iteration`` ``wp.func`` for
    each cid in the current graph-coloring partition.

    The launch is sized by the constraint *capacity*; threads with
    ``tid >= partition_count[0]`` early-out. The partitioner guarantees
    that no two cids in the same partition share a body, so the
    per-thread RMW of the body container is race-free regardless of
    which constraint type each thread happens to dispatch.
    """
    tid = wp.tid()
    if tid >= partition_count[0]:
        return
    cid = partition_element_ids[tid]

    t = constraint_get_type(constraints, cid)
    if t == CONSTRAINT_TYPE_BALL_SOCKET:
        ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
        hinge_angle_prepare_for_iteration(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
        angular_motor_prepare_for_iteration(constraints, cid, bodies, idt)


@wp.kernel
def _constraint_iterate_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    partition_element_ids: wp.array[wp.int32],
    partition_count: wp.array[wp.int32],
):
    """Dispatch the per-type ``iterate`` ``wp.func`` for each cid in the
    current graph-coloring partition. See
    :func:`_constraint_prepare_for_iteration_kernel` for the launch
    contract."""
    tid = wp.tid()
    if tid >= partition_count[0]:
        return
    cid = partition_element_ids[tid]

    t = constraint_get_type(constraints, cid)
    if t == CONSTRAINT_TYPE_BALL_SOCKET:
        ball_socket_iterate(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
        hinge_angle_iterate(constraints, cid, bodies, idt)
    elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
        angular_motor_iterate(constraints, cid, bodies, idt)


@wp.kernel
def _constraints_to_elements_kernel(
    constraints: ConstraintContainer,
    num_constraints: wp.array[wp.int32],
    elements: wp.array[ElementInteractionData],
):
    """Project every active constraint into the partitioner's
    :class:`ElementInteractionData` view, regardless of type.

    Only the two body indices are needed by the graph colourer; the
    remaining slots are filled with ``-1``. This is the type-agnostic
    successor to the previous per-type element-projection kernels:
    because the constraint header (constraint_type, body1, body2) lives
    at fixed dword offsets across every schema, a single kernel can
    pull body1/body2 with :func:`constraint_get_body1` /
    :func:`constraint_get_body2` without dispatching on type.

    Launched once per :meth:`World.step` so contact constraints (which
    arrive next) can rebuild their element rows from scratch each
    frame without coordinating with the joint constraints."""
    tid = wp.tid()
    if tid >= num_constraints[0]:
        return
    b1 = constraint_get_body1(constraints, tid)
    b2 = constraint_get_body2(constraints, tid)
    elements[tid] = element_interaction_data_make(b1, b2, -1, -1, -1, -1, -1, -1)


@wp.kernel
def _constraint_gather_wrenches_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    idt: wp.float32,
    out: wp.array[wp.spatial_vector],
):
    """Dispatch the per-type ``world_wrench`` ``wp.func`` for every
    constraint and write the result into ``out[cid]``.

    ``out`` carries the *world-frame* wrench applied by the constraint
    on its ``body2``: ``spatial_top = force [N]``,
    ``spatial_bottom = torque [N·m]``. ``idt`` is ``1 / substep_dt`` so
    the warm-started impulse is reported as the average force the
    constraint exerted during the most recent substep.

    No partitioning here -- each thread writes a unique slot, no body
    state is mutated, so a flat one-thread-per-cid launch is enough.
    """
    cid = wp.tid()
    if cid >= num_constraints:
        return

    t = constraint_get_type(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    if t == CONSTRAINT_TYPE_BALL_SOCKET:
        force, torque = ball_socket_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_HINGE_ANGLE:
        force, torque = hinge_angle_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_ANGULAR_MOTOR:
        force, torque = angular_motor_world_wrench(constraints, cid, bodies, idt)

    out[cid] = wp.spatial_vector(force, torque)


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
