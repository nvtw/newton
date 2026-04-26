# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Column-major dword-packed storage for constraint state.

State is one ``wp.array2d[wp.float32]`` of shape ``(num_dwords, num_constraints)``
-- column-major by cid. A partitioned PGS kernel reading any row hits
32 contiguous lanes in one coalesced load. Dword offsets per field are
derived at import time via :func:`dword_offset_of`.

Sharing one container across constraint types uses
``num_dwords = max(ADBS_DWORDS, CONTACT_DWORDS)``; a per-cid type tag at
dword 0 drives the dispatcher's ``if/elif``.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.helpers.data_packing import (
    dword_offset_of,
    reinterpret_float_as_int,
    reinterpret_int_as_float,
)

__all__ = [
    "CONSTRAINT_BODY1_OFFSET",
    "CONSTRAINT_BODY2_OFFSET",
    "CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET",
    "CONSTRAINT_TYPE_CONTACT",
    "CONSTRAINT_TYPE_INVALID",
    "CONSTRAINT_TYPE_OFFSET",
    "DEFAULT_DAMPING_RATIO",
    "DEFAULT_HERTZ_ANGULAR",
    "DEFAULT_HERTZ_CONTACT",
    "DEFAULT_HERTZ_LIMIT",
    "DEFAULT_HERTZ_LINEAR",
    "DEFAULT_HERTZ_MOTOR",
    "ConstraintBodies",
    "ConstraintContainer",
    "assert_constraint_header",
    "constraint_bodies_make",
    "constraint_container_zeros",
    "constraint_get_body1",
    "constraint_get_body2",
    "constraint_get_type",
    "constraint_set_body1",
    "constraint_set_body2",
    "constraint_set_type",
    "pd_coefficients",
    "read_float",
    "read_int",
    "read_mat33",
    "read_mat44",
    "read_quat",
    "read_vec3",
    "read_vec4",
    "soft_constraint_coefficients",
    "write_float",
    "write_int",
    "write_mat33",
    "write_mat44",
    "write_quat",
    "write_vec3",
    "write_vec4",
]


# ---------------------------------------------------------------------------
# Constraint header layout (type tag + body indices)
# ---------------------------------------------------------------------------
#
# Every per-type constraint schema MUST start with three int32 dwords:
#   dword 0: constraint_type,  dword 1: body1,  dword 2: body2.
# This lets the generic dispatcher / element-projection kernels read
# the type tag and body pair from any column without knowing the
# per-type schema. :func:`assert_constraint_header` verifies the
# layout at import time so reorderings fail loudly.
#
# Type tags: add monotonically, no holes, mirror the value in the
# per-type ``constraint_set_type`` call. Never reuse retired tag
# values -- bump the highest so stale persisted state shows up loudly.

#: Sentinel for unwritten / cleared columns.
CONSTRAINT_TYPE_INVALID = wp.constant(wp.int32(0))
#: Unified revolute / prismatic / ball-socket joint -- 5-DoF pure-point
#: positional lock solved as one PGS column via a rank-5 Schur complement,
#: plus an optional scalar actuator row that drives the free axis with a
#: soft position or velocity setpoint and clamps it to ``[min_value,
#: max_value]`` (one-sided spring-damper limits). See
#: :mod:`constraint_actuated_double_ball_socket`.
CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET = wp.constant(wp.int32(8))
#: Rigid-rigid contact constraint -- one column per ``(shape_a, shape_b)``
#: pair. Persistent warm-start lives in :class:`ContactContainer` keyed
#: by the contact index in the sorted Newton contact buffer.
CONSTRAINT_TYPE_CONTACT = wp.constant(wp.int32(9))

#: Dword offsets of the three header fields. By contract these are
#: 0 / 1 / 2 for every constraint schema (enforced by
#: :func:`assert_constraint_header`).
CONSTRAINT_TYPE_OFFSET = wp.constant(wp.int32(0))
CONSTRAINT_BODY1_OFFSET = wp.constant(wp.int32(1))
CONSTRAINT_BODY2_OFFSET = wp.constant(wp.int32(2))


def assert_constraint_header(struct_type: object) -> None:
    """Validate the three-field constraint header contract.

    Every per-type constraint schema must start with::

        @wp.struct
        class FooConstraintData:
            constraint_type: wp.int32  # dword 0
            body1: wp.int32  # dword 1
            body2: wp.int32  # dword 2
            # ... type-specific fields ...

    Each per-type module calls this at import time with its schema. If a
    future edit reorders or removes any of these three fields, the
    importing module fails immediately with a clear error rather than
    silently mis-routing constraint dispatch or scrambling body indices
    at runtime.
    """
    expected = [("constraint_type", 0), ("body1", 1), ("body2", 2)]
    for field, want in expected:
        try:
            got = dword_offset_of(struct_type, field)
        except AttributeError as exc:
            raise AttributeError(
                f"{struct_type!r} is missing the mandatory header field "
                f"{field!r} (every constraint schema must start with "
                "constraint_type / body1 / body2 at dwords 0 / 1 / 2)."
            ) from exc
        if got != want:
            raise ValueError(
                f"{struct_type!r}.{field} must be at dword {want} of the "
                f"schema (got {got}). The constraint header (constraint_type, "
                "body1, body2) must occupy the first three dwords of every "
                "constraint @wp.struct."
            )


@wp.struct
class ConstraintContainer:
    """Column-major dword-packed storage for all constraint types.

    ``data`` shape is ``(num_dwords, num_constraints)`` so that the
    constraint index ``cid`` is the inner (contiguous) dimension. A
    single field load like ``read_vec3(c, off, cid)`` issues 3
    ``data[off+i, cid]`` reads -- across a warp these become coalesced
    transactions because ``cid`` varies by lane and ``off`` is constant.
    """

    data: wp.array2d[wp.float32]


def constraint_container_zeros(
    num_constraints: int,
    num_dwords: int,
    device: wp.DeviceLike = None,
) -> ConstraintContainer:
    """Allocate a zero-initialised :class:`ConstraintContainer`.

    Args:
        num_constraints: Total number of constraints to store. Constraint
            indices ``[0, num_constraints)`` are globally unique across
            constraint types (ball-socket, motor, ...).
        num_dwords: Per-constraint dword count -- the maximum of the
            registered constraint types' ``num_dwords(StructType)``.
        device: Warp device.
    """
    c = ConstraintContainer()
    c.data = wp.zeros((num_dwords, num_constraints), dtype=wp.float32, device=device)
    return c


# ---------------------------------------------------------------------------
# Column-major dword accessors
# ---------------------------------------------------------------------------
#
# All read_* / write_* helpers take a fixed ``off`` (Python int wrapped in
# a wp.constant by the caller) and the per-thread ``cid``. They expand
# inside the kernel to plain indexed array accesses so the compiler sees
# them as ordinary loads/stores; nothing fancy.
#
# vec3 / quat / mat33 are addressed as 3 / 4 / 9 individual dwords with
# *no padding*. The compiler may or may not fuse the loads -- the win
# here comes from coalescing across the warp, not from wider per-thread
# transactions.


@wp.func
def read_float(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.float32:
    return c.data[off, cid]


@wp.func
def write_float(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.float32):
    c.data[off, cid] = v


@wp.func
def read_int(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.int32:
    """Bit-cast read of an int field stored in the float buffer."""
    return reinterpret_float_as_int(c.data[off, cid])


@wp.func
def write_int(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.int32):
    """Bit-cast write of an int field into the float buffer."""
    c.data[off, cid] = reinterpret_int_as_float(v)


@wp.func
def read_vec3(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.vec3f:
    return wp.vec3f(c.data[off + 0, cid], c.data[off + 1, cid], c.data[off + 2, cid])


@wp.func
def write_vec3(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.vec3f):
    c.data[off + 0, cid] = v[0]
    c.data[off + 1, cid] = v[1]
    c.data[off + 2, cid] = v[2]


@wp.func
def read_quat(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.quatf:
    return wp.quatf(
        c.data[off + 0, cid],
        c.data[off + 1, cid],
        c.data[off + 2, cid],
        c.data[off + 3, cid],
    )


@wp.func
def write_quat(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.quatf):
    c.data[off + 0, cid] = v[0]
    c.data[off + 1, cid] = v[1]
    c.data[off + 2, cid] = v[2]
    c.data[off + 3, cid] = v[3]


@wp.func
def read_mat33(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.mat33f:
    """Read a 3x3 matrix from 9 consecutive dwords, row-major.

    Field stored as ``[m00, m01, m02, m10, m11, m12, m20, m21, m22]``
    in struct order; we reconstruct the ``wp.mat33f`` with the same
    convention.
    """
    return wp.mat33f(
        c.data[off + 0, cid],
        c.data[off + 1, cid],
        c.data[off + 2, cid],
        c.data[off + 3, cid],
        c.data[off + 4, cid],
        c.data[off + 5, cid],
        c.data[off + 6, cid],
        c.data[off + 7, cid],
        c.data[off + 8, cid],
    )


@wp.func
def write_mat33(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.mat33f):
    c.data[off + 0, cid] = v[0, 0]
    c.data[off + 1, cid] = v[0, 1]
    c.data[off + 2, cid] = v[0, 2]
    c.data[off + 3, cid] = v[1, 0]
    c.data[off + 4, cid] = v[1, 1]
    c.data[off + 5, cid] = v[1, 2]
    c.data[off + 6, cid] = v[2, 0]
    c.data[off + 7, cid] = v[2, 1]
    c.data[off + 8, cid] = v[2, 2]


@wp.func
def read_vec4(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.vec4f:
    return wp.vec4f(
        c.data[off + 0, cid],
        c.data[off + 1, cid],
        c.data[off + 2, cid],
        c.data[off + 3, cid],
    )


@wp.func
def write_vec4(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.vec4f):
    c.data[off + 0, cid] = v[0]
    c.data[off + 1, cid] = v[1]
    c.data[off + 2, cid] = v[2]
    c.data[off + 3, cid] = v[3]


@wp.func
def read_mat44(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.mat44f:
    """Read a 4x4 matrix from 16 consecutive dwords, row-major.

    Field stored as ``[m00, m01, m02, m03, m10, ..., m33]`` in struct
    order; we reconstruct the ``wp.mat44f`` with the same convention.
    Used by the prismatic-mode Schur cache in
    :mod:`constraint_actuated_double_ball_socket`.
    """
    return wp.mat44f(
        c.data[off + 0, cid],
        c.data[off + 1, cid],
        c.data[off + 2, cid],
        c.data[off + 3, cid],
        c.data[off + 4, cid],
        c.data[off + 5, cid],
        c.data[off + 6, cid],
        c.data[off + 7, cid],
        c.data[off + 8, cid],
        c.data[off + 9, cid],
        c.data[off + 10, cid],
        c.data[off + 11, cid],
        c.data[off + 12, cid],
        c.data[off + 13, cid],
        c.data[off + 14, cid],
        c.data[off + 15, cid],
    )


@wp.func
def write_mat44(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.mat44f):
    c.data[off + 0, cid] = v[0, 0]
    c.data[off + 1, cid] = v[0, 1]
    c.data[off + 2, cid] = v[0, 2]
    c.data[off + 3, cid] = v[0, 3]
    c.data[off + 4, cid] = v[1, 0]
    c.data[off + 5, cid] = v[1, 1]
    c.data[off + 6, cid] = v[1, 2]
    c.data[off + 7, cid] = v[1, 3]
    c.data[off + 8, cid] = v[2, 0]
    c.data[off + 9, cid] = v[2, 1]
    c.data[off + 10, cid] = v[2, 2]
    c.data[off + 11, cid] = v[2, 3]
    c.data[off + 12, cid] = v[3, 0]
    c.data[off + 13, cid] = v[3, 1]
    c.data[off + 14, cid] = v[3, 2]
    c.data[off + 15, cid] = v[3, 3]


# ---------------------------------------------------------------------------
# Generic header accessors -- valid for any constraint type
# ---------------------------------------------------------------------------
#
# These intentionally do *not* import any per-type schema. The contract
# (asserted at module load by ``assert_constraint_header``) is that
# every schema's first three dwords are constraint_type / body1 / body2
# in that order, so the dispatcher and element-projection kernels can
# read them from any column without knowing which schema packed it.


@wp.func
def constraint_get_type(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    """Read the ``CONSTRAINT_TYPE_*`` tag of constraint ``cid``.

    Works on any column regardless of which per-type init kernel packed
    it. Used by the generic dispatch path to route each cid to the
    right prepare/iterate ``wp.func``.
    """
    return read_int(c, CONSTRAINT_TYPE_OFFSET, cid)


@wp.func
def constraint_set_type(c: ConstraintContainer, cid: wp.int32, t: wp.int32):
    """Write the ``CONSTRAINT_TYPE_*`` tag of constraint ``cid``.

    Called from each per-type initialise kernel to stamp the column
    with its type so :func:`constraint_get_type` can later route it.
    """
    write_int(c, CONSTRAINT_TYPE_OFFSET, cid, t)


@wp.func
def constraint_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    """Read the body 1 index of constraint ``cid``.

    Type-agnostic; works because every constraint schema is required to
    place ``body1`` at dword 1 (see :func:`assert_constraint_header`).
    Used by the unified element-projection kernel that builds the
    partitioner's element view from the shared container without having
    to dispatch on type.
    """
    return read_int(c, CONSTRAINT_BODY1_OFFSET, cid)


@wp.func
def constraint_set_body1(c: ConstraintContainer, cid: wp.int32, b: wp.int32):
    write_int(c, CONSTRAINT_BODY1_OFFSET, cid, b)


@wp.func
def constraint_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    """Read the body 2 index of constraint ``cid``. See
    :func:`constraint_get_body1` for the contract."""
    return read_int(c, CONSTRAINT_BODY2_OFFSET, cid)


@wp.func
def constraint_set_body2(c: ConstraintContainer, cid: wp.int32, b: wp.int32):
    write_int(c, CONSTRAINT_BODY2_OFFSET, cid, b)


# ---------------------------------------------------------------------------
# Body-pair carrier for composable constraint funcs
# ---------------------------------------------------------------------------
#
# When a *composite* constraint (e.g. the fused HingeJoint) wants to call
# its sub-component ``*_at`` funcs, those sub-funcs must not re-fetch
# body1 / body2 from the shared header (the composite's sub-blocks don't
# carry their own body indices -- there's only one header per column).
# Instead the composite reads the pair once and threads it through every
# sub-call as a tiny carrier struct, which the compiler treats as two
# scratch ints that stay in registers across the sequential sub-calls.


@wp.struct
class ConstraintBodies:
    """Body indices of the (body1, body2) pair a constraint operates on.

    Threaded through composable ``*_at`` constraint funcs so they can
    address ``BodyContainer`` without going back to the constraint
    column. Used by both the *direct* path (where the wrapper extracts
    the pair from the column header and forwards it) and the *composite*
    path (where the fused constraint reads the pair from its shared
    header once and forwards the same instance to all sub-funcs).
    """

    b1: wp.int32
    b2: wp.int32


@wp.func
def constraint_bodies_make(b1: wp.int32, b2: wp.int32) -> ConstraintBodies:
    """Construct a :class:`ConstraintBodies` carrier from two body indices."""
    bp = ConstraintBodies()
    bp.b1 = b1
    bp.b2 = b2
    return bp


# ---------------------------------------------------------------------------
# Soft-constraint coefficients (Box2D v3 / Bepu / Nordby formulation)
# ---------------------------------------------------------------------------
#
# User specifies an undamped natural frequency ``hertz`` (Hz) and a
# non-dimensional ``damping_ratio`` (1 = critically damped); per-substep
# PGS coefficients fall out from ``(hertz, damping_ratio, dt)``. Mass-
# and substep-independent; requested ``omega = 2*pi*hertz`` is clamped
# at the per-substep Nyquist rate so asking for more stiffness than the
# integrator can resolve yields the stiffest resolvable lock, never an
# aliased response. Defaults below sit above any realistic Nyquist so
# they always clamp, producing maximally-rigid locks; pass a smaller
# finite ``hertz`` for honest spring compliance.
#
# See https://box2d.org/posts/2024/02/solver2d/ ("Soft Constraints") for
# the derivation.

#: Critically damped by default -- no overshoot, no underdamped ringing.
DEFAULT_DAMPING_RATIO = wp.constant(wp.float32(1.0))
#: Linear joint stiffness; Nyquist-clamped -> maximally rigid.
#: Override with a smaller finite ``hertz`` for compliance, or ``0`` for
#: a rigid PGS row with no drift correction.
DEFAULT_HERTZ_LINEAR = wp.constant(wp.float32(1.0e9))
#: Angular lock stiffness; same contract as :data:`DEFAULT_HERTZ_LINEAR`.
DEFAULT_HERTZ_ANGULAR = wp.constant(wp.float32(1.0e9))
#: Hinge / cone limit stiffness; Nyquist-clamped so limits don't bounce.
DEFAULT_HERTZ_LIMIT = wp.constant(wp.float32(1.0e9))
#: Velocity-motor stiffness (only modulates the impulse coefficient
#: since velocity targets carry no positional bias).
DEFAULT_HERTZ_MOTOR = wp.constant(wp.float32(1.0e9))
#: Contact-pair stiffness; Nyquist-clamped. The big first-substep
#: impulse on initial deep penetration is bounded by the prepare
#: kernel's ``max_push_speed`` recovery-speed cap.
DEFAULT_HERTZ_CONTACT = wp.constant(wp.float32(1.0e9))


@wp.func
def soft_constraint_coefficients(
    hertz: wp.float32,
    damping_ratio: wp.float32,
    dt: wp.float32,
):
    """Map ``(hertz, damping_ratio, dt)`` to PGS soft-constraint coefficients.

    Box2D v3 / Bepu / Nordby formulation
    (https://box2d.org/posts/2024/02/solver2d/). ``omega = 2*pi*hertz``
    is clamped at the per-substep Nyquist ``pi/dt``, so requesting
    more stiffness than the current ``dt`` can resolve yields the
    stiffest resolvable lock rather than aliasing. Substep-independent:
    more substeps = higher Nyquist = more rigid.

    Returns ``(bias_rate [1/s], mass_coeff [-], impulse_coeff [-])``;
    the latter two lie in ``[0, 1]`` and feed straight into the PGS
    update.

    ``hertz <= 0`` yields a rigid constraint with no drift
    correction: ``(0, 1, 0)`` -- the un-softened plain-PGS row.
    """
    if hertz <= 0.0:
        return wp.float32(0.0), wp.float32(1.0), wp.float32(0.0)

    # Clamp omega at the per-substep Nyquist rate: the fastest spring
    # the integrator can resolve is one with half a cycle per step,
    # i.e. ``omega * dt == pi``. Requests beyond that alias, so we
    # saturate to the maximally-rigid response the current ``dt``
    # supports. See the block comment above for the full rationale.
    omega_request = 2.0 * 3.14159265358979 * hertz
    omega_nyquist = 3.14159265358979 / dt
    omega = wp.min(omega_request, omega_nyquist)
    a1 = 2.0 * damping_ratio + omega * dt
    a2 = dt * omega * a1
    a3 = 1.0 / (1.0 + a2)
    return omega / a1, a2 * a3, a3


@wp.func
def pd_coefficients(
    stiffness: wp.float32,
    damping: wp.float32,
    position_error: wp.float32,
    eff_mass_inv: wp.float32,
    dt: wp.float32,
):
    """Map ``(k, c, C, M_inv, dt)`` to Jitter2 PGS spring-damper triple.

    Implicit-Euler spring-damper from Jitter2 ``SpringConstraint.
    SetSpringParameters``. User specifies absolute gains ``k`` [N/m
    or N*m/rad] and ``c`` [N*s/m or N*m*s/rad]; unlike the Box2D-style
    :func:`soft_constraint_coefficients` these do *not* bake in the
    effective mass, so for mass-invariant behaviour the caller
    rescales.

    Returns ``(gamma, bias, eff_mass_softened)`` with ``idt = 1/dt``
    already folded in so the PGS iterate is one update:

    * ``gamma             = 1 / (c + dt*k) * idt``       [s^-1]
    * ``bias              = C * dt*k / (c + dt*k) * idt`` [m/s or rad/s]
    * ``eff_mass_softened = 1 / (M_inv + gamma)``         [kg or kg*m^2]

    yielding ``lambda = -M_eff_soft * (J v - bias_signed +
    gamma * lambda_acc)`` (caller sign-flips ``bias`` to match its
    Jacobian convention).

    Short-circuits to ``(0, 0, 0)`` -- a no-op PGS row -- when
    ``eff_mass_inv <= 0`` or both gains are zero. Callers additionally
    guard on a "drive off" flag to skip the solve entirely.

    Args:
        stiffness: Positional stiffness ``k >= 0``; ``0`` = pure damper.
        damping: Viscous damping ``c >= 0``; ``0`` = pure spring.
        position_error: ``C = actual - target`` [rad or m].
        eff_mass_inv: ``J M^{-1} J^T``; ``0`` short-circuits to no-op.
        dt: Substep [s], must be > 0.
    """
    if eff_mass_inv <= 0.0:
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    if stiffness <= 0.0 and damping <= 0.0:
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)

    # Jitter2 SetSpringParameters, but with k/c already in absolute
    # units (Jitter2's version rescales by the edge m_eff; we let the
    # caller decide that).
    softness = wp.float32(1.0) / (damping + dt * stiffness)
    bias_factor = dt * stiffness * softness
    idt = wp.float32(1.0) / dt

    # Fold the 1/dt scale into gamma and bias so the kernel iterate is
    # identical to the pure-velocity-motor path modulo these two
    # scalars.
    gamma = softness * idt
    bias = position_error * bias_factor * idt
    eff_mass_soft = wp.float32(1.0) / (eff_mass_inv + gamma)
    return gamma, bias, eff_mass_soft
