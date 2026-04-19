# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of Jitter2's BallSocket constraint.

Direct translation of ``Jitter2.Dynamics.Constraints.BallSocket``
(``C:/git3/jitterphysics2/src/Jitter2/Dynamics/Constraints/BallSocket.cs``).

The C# code uses ``JHandle<RigidBodyData>`` to indirect into a partitioned
buffer; here a body handle is just an ``int32`` index into a
:class:`BodyContainer` (struct-of-arrays storage, see
:mod:`newton._src.solvers.jitter.body`).

State storage no longer uses one ``wp.array[BallSocketData]`` per type --
all constraint state of every type lives in a single shared
:class:`ConstraintContainer` (column-major-by-cid 2D float buffer). The
``@wp.struct BallSocketData`` below is *only* a schema: it never gets
instantiated at runtime; we derive its dword-offset table from it once
at module load and the runtime kernels read/write fields via the typed
``ball_socket_get_*`` / ``ball_socket_set_*`` accessors.

Concurrency: the constraint kernels assume the launcher has partitioned
the constraint set so that no two constraints in the same launch touch
the same body (this is what
:class:`newton._src.solvers.jitter.graph_coloring_incremental.IncrementalContactPartitioner`
produces). Within a partition each thread can therefore do a plain
read-modify-write of body ``body1`` / ``body2`` without atomics.

Mapping summary:

* ``JVector``                            -> ``wp.vec3f``
* ``JQuaternion``                        -> ``wp.quatf``
* ``JMatrix``                            -> ``wp.mat33f``
* ``JVector.Transform(v, M)``            -> ``M @ v`` (matrix-vector)
* ``JVector.Transform(v, q)``            -> ``wp.quat_rotate(q, v)``
* ``JVector.ConjugatedTransform(v, q)``  -> ``wp.quat_rotate_inv(q, v)``
* ``JMatrix.CreateCrossProduct(r)``      -> ``wp.skew(r)``
* ``JMatrix.MultiplyTransposed(A, B)``   -> ``A @ wp.transpose(B)``
* ``JMatrix.Identity``                   -> ``wp.identity(3, dtype=...)``
* ``JMatrix.Inverse(M, out M)``          -> ``wp.inverse(M)``
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_BALL_SOCKET,
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LINEAR,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    read_float,
    read_int,
    read_mat33,
    read_vec3,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_mat33,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords

__all__ = [
    "BS_DWORDS",
    "BallSocketData",
    "ball_socket_get_accumulated_impulse",
    "ball_socket_get_bias",
    "ball_socket_get_bias_rate",
    "ball_socket_get_body1",
    "ball_socket_get_body2",
    "ball_socket_get_damping_ratio",
    "ball_socket_get_effective_mass",
    "ball_socket_get_hertz",
    "ball_socket_get_impulse_coeff",
    "ball_socket_get_local_anchor1",
    "ball_socket_get_local_anchor2",
    "ball_socket_get_mass_coeff",
    "ball_socket_get_r1",
    "ball_socket_get_r2",
    "ball_socket_initialize_kernel",
    "ball_socket_iterate",
    "ball_socket_iterate_at",
    "ball_socket_prepare_for_iteration",
    "ball_socket_prepare_for_iteration_at",
    "ball_socket_set_accumulated_impulse",
    "ball_socket_set_bias",
    "ball_socket_set_bias_rate",
    "ball_socket_set_body1",
    "ball_socket_set_body2",
    "ball_socket_set_damping_ratio",
    "ball_socket_set_effective_mass",
    "ball_socket_set_hertz",
    "ball_socket_set_impulse_coeff",
    "ball_socket_set_local_anchor1",
    "ball_socket_set_local_anchor2",
    "ball_socket_set_mass_coeff",
    "ball_socket_set_r1",
    "ball_socket_set_r2",
    "ball_socket_world_wrench",
    "ball_socket_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class BallSocketData:
    """Per-constraint dword-layout schema for a ball-and-socket joint.

    *Schema only.* Runtime kernels never instantiate this struct; they
    read/write its fields out of a shared :class:`ConstraintContainer`
    using the dword offsets derived from this layout via
    :func:`dword_offset_of`. Field order therefore matters because it
    fixes the dword offsets, but value semantics live entirely in the
    typed accessors below.

    Fields are arranged in struct-natural order with no manual padding;
    everything is 32-bit so the struct is dword-aligned end-to-end (see
    :func:`num_dwords` which asserts this).

    The first field is the global ``constraint_type`` tag (mandatory
    contract for every constraint schema -- see
    :func:`assert_constraint_header`). It lets the dispatcher
    read the type at a fixed offset without per-type knowledge.
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    local_anchor1: wp.vec3f
    local_anchor2: wp.vec3f

    r1: wp.vec3f
    r2: wp.vec3f

    # User-facing soft-constraint knobs (Box2D v3 / Bepu / Nordby
    # formulation; see ``soft_constraint_coefficients`` in
    # :mod:`constraint_container`). ``hertz`` is the undamped natural
    # frequency [Hz]; ``damping_ratio`` is non-dimensional (1 = critical).
    # Persisted across substeps; the three derived coefficients below
    # are recomputed every prepare from the current substep dt.
    hertz: wp.float32
    damping_ratio: wp.float32

    # Cached per-substep coefficients. ``bias_rate`` [1/s] times the
    # positional separation gives the velocity-correction bias; the
    # other two scale the unsoftened effective-mass impulse and the
    # accumulated-impulse leak. Cleared/refilled in
    # ``ball_socket_prepare_for_iteration_at``.
    bias_rate: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32

    effective_mass: wp.mat33f
    accumulated_impulse: wp.vec3f
    bias: wp.vec3f


# Enforce the global constraint header contract (constraint_type / body1
# / body2 at dwords 0 / 1 / 2) at import time so a future field reorder
# fails loudly here instead of silently mis-tagging columns or
# scrambling body indices at runtime.
assert_constraint_header(BallSocketData)

# Dword offsets derived once from the schema. Each is a Python int;
# wrapped in wp.constant so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(BallSocketData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(BallSocketData, "body2"))
_OFF_LA1 = wp.constant(dword_offset_of(BallSocketData, "local_anchor1"))
_OFF_LA2 = wp.constant(dword_offset_of(BallSocketData, "local_anchor2"))
_OFF_R1 = wp.constant(dword_offset_of(BallSocketData, "r1"))
_OFF_R2 = wp.constant(dword_offset_of(BallSocketData, "r2"))
_OFF_HERTZ = wp.constant(dword_offset_of(BallSocketData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(BallSocketData, "damping_ratio"))
_OFF_BIAS_RATE = wp.constant(dword_offset_of(BallSocketData, "bias_rate"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(BallSocketData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(BallSocketData, "impulse_coeff"))
_OFF_EFFECTIVE_MASS = wp.constant(dword_offset_of(BallSocketData, "effective_mass"))
_OFF_ACCUMULATED_IMPULSE = wp.constant(dword_offset_of(BallSocketData, "accumulated_impulse"))
_OFF_BIAS = wp.constant(dword_offset_of(BallSocketData, "bias"))

#: Total dword count of one ball-socket constraint. Used by the host-side
#: container allocator to size ``ConstraintContainer.data``'s row count.
BS_DWORDS: int = num_dwords(BallSocketData)


# ---------------------------------------------------------------------------
# Typed accessors -- thin wrappers over column-major dword get/set
# ---------------------------------------------------------------------------


@wp.func
def ball_socket_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def ball_socket_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def ball_socket_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def ball_socket_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def ball_socket_get_local_anchor1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA1, cid)


@wp.func
def ball_socket_set_local_anchor1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA1, cid, v)


@wp.func
def ball_socket_get_local_anchor2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA2, cid)


@wp.func
def ball_socket_set_local_anchor2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA2, cid, v)


@wp.func
def ball_socket_get_r1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R1, cid)


@wp.func
def ball_socket_set_r1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R1, cid, v)


@wp.func
def ball_socket_get_r2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R2, cid)


@wp.func
def ball_socket_set_r2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R2, cid, v)


@wp.func
def ball_socket_get_hertz(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_HERTZ, cid)


@wp.func
def ball_socket_set_hertz(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ, cid, v)


@wp.func
def ball_socket_get_damping_ratio(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING_RATIO, cid)


@wp.func
def ball_socket_set_damping_ratio(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_RATIO, cid, v)


@wp.func
def ball_socket_get_bias_rate(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_RATE, cid)


@wp.func
def ball_socket_set_bias_rate(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_RATE, cid, v)


@wp.func
def ball_socket_get_mass_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MASS_COEFF, cid)


@wp.func
def ball_socket_set_mass_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MASS_COEFF, cid, v)


@wp.func
def ball_socket_get_impulse_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_IMPULSE_COEFF, cid)


@wp.func
def ball_socket_set_impulse_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_IMPULSE_COEFF, cid, v)


@wp.func
def ball_socket_get_effective_mass(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_EFFECTIVE_MASS, cid)


@wp.func
def ball_socket_set_effective_mass(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_EFFECTIVE_MASS, cid, v)


@wp.func
def ball_socket_get_accumulated_impulse(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACCUMULATED_IMPULSE, cid)


@wp.func
def ball_socket_set_accumulated_impulse(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACCUMULATED_IMPULSE, cid, v)


@wp.func
def ball_socket_get_bias(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS, cid)


@wp.func
def ball_socket_set_bias(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS, cid, v)


# ---------------------------------------------------------------------------
# Initialization (kernel; replaces the old host-side helper)
# ---------------------------------------------------------------------------


@wp.kernel
def ball_socket_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor: wp.array[wp.vec3f],
    hertz: wp.array[wp.float32],
    damping_ratio: wp.array[wp.float32],
):
    """Pack one batch of ball-socket descriptors into ``constraints``.

    Mirrors ``BallSocket.Initialize`` (BallSocket.cs:65-80), but the
    legacy ``bias_factor`` / ``softness`` knobs are replaced by
    ``hertz`` / ``damping_ratio`` (Box2D v3 / Bepu / Nordby formulation;
    see :func:`soft_constraint_coefficients` in
    :mod:`constraint_container`). The three derived per-substep
    coefficients are recomputed every prepare from the actual substep
    ``dt``, which is what makes the joint behaviour time-step-independent.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; this kernel only reads
            ``position`` / ``orientation`` of the two referenced bodies.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor: World-space anchor points [num_in_batch] [m].
        hertz: Per-constraint stiffness target [num_in_batch] [Hz].
        damping_ratio: Per-constraint damping ratio [num_in_batch]
            (1 = critically damped).
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a = anchor[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    orient1 = bodies.orientation[b1]
    orient2 = bodies.orientation[b2]

    la1 = wp.quat_rotate_inv(orient1, a - pos1)
    la2 = wp.quat_rotate_inv(orient2, a - pos2)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_BALL_SOCKET)
    ball_socket_set_body1(constraints, cid, b1)
    ball_socket_set_body2(constraints, cid, b2)
    ball_socket_set_local_anchor1(constraints, cid, la1)
    ball_socket_set_local_anchor2(constraints, cid, la2)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    ball_socket_set_r1(constraints, cid, zero3)
    ball_socket_set_r2(constraints, cid, zero3)
    ball_socket_set_bias(constraints, cid, zero3)
    ball_socket_set_accumulated_impulse(constraints, cid, zero3)

    ball_socket_set_hertz(constraints, cid, hertz[tid])
    ball_socket_set_damping_ratio(constraints, cid, damping_ratio[tid])
    ball_socket_set_bias_rate(constraints, cid, 0.0)
    ball_socket_set_mass_coeff(constraints, cid, 1.0)
    ball_socket_set_impulse_coeff(constraints, cid, 0.0)

    ball_socket_set_effective_mass(constraints, cid, wp.identity(3, dtype=wp.float32))


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Two access levels share the same math:
#
# * ``*_at`` variants take an explicit ``base_offset`` (dword offset of
#   the ball-socket sub-block within its column) and a
#   :class:`ConstraintBodies` carrier with the body indices. They are
#   what the *fused* :data:`CONSTRAINT_TYPE_HINGE_JOINT` constraint
#   calls when running its three sub-iterations sequentially on a
#   shared column -- the fused header carries body1/body2 once and the
#   sub-blocks contain only their per-body fields.
# * The plain ``ball_socket_prepare_for_iteration`` /
#   ``ball_socket_iterate`` / ``ball_socket_world_wrench`` are the
#   *direct* entry points the unified dispatcher cascade uses. They are
#   thin wrappers that read body1/body2 from this constraint's own
#   header at the standard offsets and forward to the ``*_at`` variant
#   with ``base_offset = 0``.


@wp.func
def ball_socket_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable ``PrepareForIterationBallSocket`` (BallSocket.cs:130).

    Reads/writes the ball-socket sub-block at column ``cid`` starting at
    dword ``base_offset``. Body indices come from ``body_pair`` (the
    fused-constraint header carries them once). When invoked as a
    standalone constraint, ``base_offset = 0`` and ``body_pair`` is
    populated from this same column's header.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    la1 = read_vec3(constraints, base_offset + _OFF_LA1, cid)
    la2 = read_vec3(constraints, base_offset + _OFF_LA2, cid)

    # JVector.Transform(LocalAnchor, body.Orientation, out R)
    r1 = wp.quat_rotate(orientation1, la1)
    r2 = wp.quat_rotate(orientation2, la2)
    write_vec3(constraints, base_offset + _OFF_R1, cid, r1)
    write_vec3(constraints, base_offset + _OFF_R2, cid, r2)

    # World-space anchor positions on each body.
    p1 = position1 + r1
    p2 = position2 + r2

    # cr_i = [r_i]_x   (skew-symmetric cross-product matrix)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

    # EffectiveMass = m1^-1 * I + cr1 * (InvI1 * cr1^T)
    #               + m2^-1 * I + cr2 * (InvI2 * cr2^T)
    # Box2D v3 / Bepu soft-constraint formulation: keep the effective
    # mass *unsoftened* here -- the softness scaling lives in
    # ``mass_coeff`` and ``impulse_coeff`` instead, so the rigid case
    # falls out naturally when ``hertz`` is large.
    eye3 = wp.identity(3, dtype=wp.float32)
    eff = inv_mass1 * eye3
    eff = eff + cr1 @ (inv_inertia1 @ wp.transpose(cr1))
    eff = eff + inv_mass2 * eye3
    eff = eff + cr2 @ (inv_inertia2 @ wp.transpose(cr2))

    write_mat33(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid, wp.inverse(eff))

    # Recompute soft-constraint coefficients from the current substep dt.
    # Using ``dt = 1 / idt`` keeps the solver dispatch signature unchanged
    # while letting the joint behaviour stay invariant under dt changes.
    dt = 1.0 / idt
    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        hertz, damping_ratio, dt
    )
    write_float(constraints, base_offset + _OFF_BIAS_RATE, cid, bias_rate)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    # Velocity-correction bias = bias_rate * positional separation.
    bias = (p2 - p1) * bias_rate
    write_vec3(constraints, base_offset + _OFF_BIAS, cid, bias)

    # Warm start: re-apply the previous solve's accumulated impulse.
    acc = read_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    velocity1 = bodies.velocity[b1] - inv_mass1 * acc
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1 @ acc)

    velocity2 = bodies.velocity[b2] + inv_mass2 * acc
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr2 @ acc)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def ball_socket_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable ``IterateBallSocket`` (BallSocket.cs:199).

    See :func:`ball_socket_prepare_for_iteration_at` for the
    ``base_offset`` / ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    velocity1 = bodies.velocity[b1]
    velocity2 = bodies.velocity[b2]
    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    r1 = read_vec3(constraints, base_offset + _OFF_R1, cid)
    r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

    acc = read_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    bias = read_vec3(constraints, base_offset + _OFF_BIAS, cid)
    eff = read_mat33(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    # jv = -v1 + cr1 * w1 + v2 - cr2 * w2
    jv = -velocity1 + cr1 @ angular_velocity1 + velocity2 - cr2 @ angular_velocity2

    # Box2D v3 / Bepu soft-constraint PGS update:
    #   lambda = -mass_coeff * EffectiveMass * (jv + bias)
    #            - impulse_coeff * accumulated_impulse
    # Setting (mass_coeff, impulse_coeff) = (1, 0) recovers the rigid
    # plain-PGS update that the legacy ``softness`` formulation
    # approached only asymptotically.
    lam = -mass_coeff * (eff @ (jv + bias)) - impulse_coeff * acc

    write_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, acc + lam)

    bodies.velocity[b1] = velocity1 - inv_mass1 * lam
    bodies.angular_velocity[b1] = angular_velocity1 - inv_inertia1 @ (cr1 @ lam)

    bodies.velocity[b2] = velocity2 + inv_mass2 * lam
    bodies.angular_velocity[b2] = angular_velocity2 + inv_inertia2 @ (cr2 @ lam)


@wp.func
def ball_socket_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable ball-socket wrench on body2; see
    :func:`ball_socket_world_wrench` for semantics."""
    acc = read_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)
    force = acc * idt
    torque = wp.cross(r2, force)
    return force, torque


@wp.func
def ball_socket_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``PrepareForIterationBallSocket`` (BallSocket.cs:130).

    Thin wrapper: reads ``body1`` / ``body2`` from this column's header
    and forwards to :func:`ball_socket_prepare_for_iteration_at` with
    ``base_offset = 0``.
    """
    b1 = ball_socket_get_body1(constraints, cid)
    b2 = ball_socket_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    ball_socket_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def ball_socket_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``IterateBallSocket`` (BallSocket.cs:199).

    Thin wrapper: see :func:`ball_socket_iterate_at`.
    """
    b1 = ball_socket_get_body1(constraints, cid)
    b2 = ball_socket_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    ball_socket_iterate_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def ball_socket_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this constraint exerts on body2.

    Force is the linear constraint impulse (``accumulated_impulse``)
    divided by the substep ``dt`` (``idt = 1 / substep_dt``); torque is
    that force's moment about body2's COM, using the cached lever arm
    ``r2`` from the most recent ``prepare_for_iteration``.
    """
    return ball_socket_world_wrench_at(constraints, cid, 0, idt)
