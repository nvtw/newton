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
    ConstraintContainer,
    assert_constraint_header,
    constraint_set_type,
    read_float,
    read_int,
    read_mat33,
    read_vec3,
    write_float,
    write_int,
    write_mat33,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords

__all__ = [
    "BS_DWORDS",
    "DEFAULT_LINEAR_BIAS",
    "DEFAULT_LINEAR_SOFTNESS",
    "BallSocketData",
    "ball_socket_get_accumulated_impulse",
    "ball_socket_get_bias",
    "ball_socket_get_bias_factor",
    "ball_socket_get_body1",
    "ball_socket_get_body2",
    "ball_socket_get_effective_mass",
    "ball_socket_get_local_anchor1",
    "ball_socket_get_local_anchor2",
    "ball_socket_get_r1",
    "ball_socket_get_r2",
    "ball_socket_get_softness",
    "ball_socket_initialize_kernel",
    "ball_socket_iterate",
    "ball_socket_prepare_for_iteration",
    "ball_socket_set_accumulated_impulse",
    "ball_socket_set_bias",
    "ball_socket_set_bias_factor",
    "ball_socket_set_body1",
    "ball_socket_set_body2",
    "ball_socket_set_effective_mass",
    "ball_socket_set_local_anchor1",
    "ball_socket_set_local_anchor2",
    "ball_socket_set_r1",
    "ball_socket_set_r2",
    "ball_socket_set_softness",
]


# ---------------------------------------------------------------------------
# Constants (mirrors Jitter2.Dynamics.Constraints.Constraint)
# ---------------------------------------------------------------------------

DEFAULT_LINEAR_SOFTNESS = wp.constant(wp.float32(0.00001))
DEFAULT_LINEAR_BIAS = wp.constant(wp.float32(0.2))


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
    u: wp.vec3f

    bias_factor: wp.float32
    softness: wp.float32

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
_OFF_U = wp.constant(dword_offset_of(BallSocketData, "u"))
_OFF_BIAS_FACTOR = wp.constant(dword_offset_of(BallSocketData, "bias_factor"))
_OFF_SOFTNESS = wp.constant(dword_offset_of(BallSocketData, "softness"))
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
def ball_socket_get_bias_factor(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_FACTOR, cid)


@wp.func
def ball_socket_set_bias_factor(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_FACTOR, cid, v)


@wp.func
def ball_socket_get_softness(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_SOFTNESS, cid)


@wp.func
def ball_socket_set_softness(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_SOFTNESS, cid, v)


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
):
    """Pack one batch of ball-socket descriptors into ``constraints``.

    Direct port of ``BallSocket.Initialize`` (BallSocket.cs:65-80) plus
    zero-initialisation of the derived/cached fields. Runs once per type
    inside :meth:`WorldBuilder.finalize`; the per-batch input arrays
    (``body1``, ``body2``, ``anchor``) live on the host in the builder
    and get uploaded just for this kernel.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; this kernel only reads
            ``position`` / ``orientation`` of the two referenced bodies.
        cid_offset: Global cid of the first constraint in this batch.
            Used so the same kernel can pack different ranges of the
            container if the builder ever batches by type.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor: World-space anchor points [num_in_batch] [m].
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

    ball_socket_set_bias_factor(constraints, cid, DEFAULT_LINEAR_BIAS)
    ball_socket_set_softness(constraints, cid, DEFAULT_LINEAR_SOFTNESS)

    ball_socket_set_effective_mass(constraints, cid, wp.identity(3, dtype=wp.float32))


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# These two ``wp.func``s are the only per-type entry points the solver
# needs at runtime. They share a fixed signature with the matching
# helpers in every other constraint module
# (``constraints, cid, bodies, idt``) so the unified dispatcher kernel
# in :mod:`solver_jitter_kernels` can call any of them through an
# ``if/elif`` cascade gated on the ``constraint_type`` tag.


@wp.func
def ball_socket_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``PrepareForIterationBallSocket`` (BallSocket.cs:130).

    Reads ``body1`` / ``body2`` from the constraint to look up the two
    bodies in the solver's :class:`BodyContainer`, recomputes the cached
    lever arms / effective mass / bias, and applies warm-start impulses
    to both bodies. Only the body fields actually consumed are loaded
    and only ``velocity`` / ``angular_velocity`` are written back.
    """
    b1 = ball_socket_get_body1(constraints, cid)
    b2 = ball_socket_get_body2(constraints, cid)

    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    la1 = ball_socket_get_local_anchor1(constraints, cid)
    la2 = ball_socket_get_local_anchor2(constraints, cid)

    # JVector.Transform(LocalAnchor, body.Orientation, out R)
    r1 = wp.quat_rotate(orientation1, la1)
    r2 = wp.quat_rotate(orientation2, la2)
    ball_socket_set_r1(constraints, cid, r1)
    ball_socket_set_r2(constraints, cid, r2)

    # World-space anchor positions on each body.
    p1 = position1 + r1
    p2 = position2 + r2

    # cr_i = [r_i]_x   (skew-symmetric cross-product matrix)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

    # EffectiveMass = m1^-1 * I + cr1 * (InvI1 * cr1^T)
    #               + m2^-1 * I + cr2 * (InvI2 * cr2^T)
    eye3 = wp.identity(3, dtype=wp.float32)
    eff = inv_mass1 * eye3
    eff = eff + cr1 @ (inv_inertia1 @ wp.transpose(cr1))
    eff = eff + inv_mass2 * eye3
    eff = eff + cr2 @ (inv_inertia2 @ wp.transpose(cr2))

    softness = ball_socket_get_softness(constraints, cid) * idt
    eff[0, 0] = eff[0, 0] + softness
    eff[1, 1] = eff[1, 1] + softness
    eff[2, 2] = eff[2, 2] + softness

    ball_socket_set_effective_mass(constraints, cid, wp.inverse(eff))

    bias_factor = ball_socket_get_bias_factor(constraints, cid)
    bias = (p2 - p1) * bias_factor * idt
    ball_socket_set_bias(constraints, cid, bias)

    # Warm start: re-apply the previous solve's accumulated impulse.
    acc = ball_socket_get_accumulated_impulse(constraints, cid)

    velocity1 = bodies.velocity[b1] - inv_mass1 * acc
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1 @ acc)

    velocity2 = bodies.velocity[b2] + inv_mass2 * acc
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr2 @ acc)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def ball_socket_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``IterateBallSocket`` (BallSocket.cs:199).

    One PGS-style iteration: build the constraint Jacobian-times-velocity,
    project to find the corrective impulse ``lambda``, accumulate it, and
    apply the matching velocity / angular-velocity deltas to the two bodies.
    Only velocity / angular-velocity / inverse mass / inverse world-inertia
    are touched -- positions and orientations are not needed here.
    """
    b1 = ball_socket_get_body1(constraints, cid)
    b2 = ball_socket_get_body2(constraints, cid)

    velocity1 = bodies.velocity[b1]
    velocity2 = bodies.velocity[b2]
    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    r1 = ball_socket_get_r1(constraints, cid)
    r2 = ball_socket_get_r2(constraints, cid)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

    acc = ball_socket_get_accumulated_impulse(constraints, cid)
    softness = ball_socket_get_softness(constraints, cid)
    bias = ball_socket_get_bias(constraints, cid)
    eff = ball_socket_get_effective_mass(constraints, cid)

    softness_vector = acc * softness * idt

    # jv = -v1 + cr1 * w1 + v2 - cr2 * w2
    jv = -velocity1 + cr1 @ angular_velocity1 + velocity2 - cr2 @ angular_velocity2

    # lambda = -EffectiveMass * (jv + Bias + softnessVector)
    lam = -(eff @ (jv + bias + softness_vector))

    ball_socket_set_accumulated_impulse(constraints, cid, acc + lam)

    bodies.velocity[b1] = velocity1 - inv_mass1 * lam
    bodies.angular_velocity[b1] = angular_velocity1 - inv_inertia1 @ (cr1 @ lam)

    bodies.velocity[b2] = velocity2 + inv_mass2 * lam
    bodies.angular_velocity[b2] = angular_velocity2 + inv_inertia2 @ (cr2 @ lam)
