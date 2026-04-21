# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Column-major dword-packed storage shared by all constraint types.

All constraint state (ball-socket anchors, hinge axes, motor velocities,
contact normals, ...) lives in a *single* ``wp.array2d[wp.float32]``
shaped ``(num_dwords, num_constraints)`` -- column-major-by-cid. Each
column is one constraint's flat dword image; rows pick out individual
fields. With ``cid`` along the warp's lane axis, every per-field load
in a partitioned PGS kernel is a fully coalesced 128-byte transaction:
32 lanes hit the same row, consecutive columns, contiguous in memory.

The dword offsets for each field are derived once at module load via
:func:`newton._src.solvers.jitter.data_packing.dword_offset_of` from
the per-type ``@wp.struct`` schemas. The struct is *only* a schema --
runtime kernels never instantiate it; they read/write individual fields
via the typed accessors built on top of the helpers in this module.

Multiple constraint types share one container by allocating
``num_dwords = max(BS_DWORDS, AM_DWORDS, ...)`` rows; constraint
indices are global across types and a per-cid type tag (TBD when more
types land) tells the dispatcher which kernel to launch on which
partition. The wasted dwords are a flat tax for type-mixing
flexibility.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.data_packing import (
    dword_offset_of,
    reinterpret_float_as_int,
    reinterpret_int_as_float,
)

__all__ = [
    "CONSTRAINT_BODY1_OFFSET",
    "CONSTRAINT_BODY2_OFFSET",
    "CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET",
    "CONSTRAINT_TYPE_ANGULAR_LIMIT",
    "CONSTRAINT_TYPE_ANGULAR_MOTOR",
    "CONSTRAINT_TYPE_BALL_SOCKET",
    "CONSTRAINT_TYPE_CONTACT",
    "CONSTRAINT_TYPE_D6",
    "CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET",
    "CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC",
    "CONSTRAINT_TYPE_HINGE_ANGLE",
    "CONSTRAINT_TYPE_HINGE_JOINT",
    "CONSTRAINT_TYPE_INVALID",
    "CONSTRAINT_TYPE_LINEAR_LIMIT",
    "CONSTRAINT_TYPE_LINEAR_MOTOR",
    "CONSTRAINT_TYPE_OFFSET",
    "CONSTRAINT_TYPE_PRISMATIC",
    "DEFAULT_DAMPING_RATIO",
    "DEFAULT_HERTZ_ANGULAR",
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
    "read_float",
    "read_int",
    "read_mat33",
    "read_mat44",
    "read_quat",
    "read_vec3",
    "read_vec4",
    "pd_coefficients",
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
# Every per-type constraint schema *must* start with the same three
# dwords, in order:
#
#   dword 0: constraint_type  (wp.int32)
#   dword 1: body1            (wp.int32)
#   dword 2: body2            (wp.int32)
#
# Pinning this header lets the generic dispatcher and element-projection
# kernels read the type tag + body pair from *any* column without
# knowing which per-type schema packed it. The per-type ``*_get_body1``
# / ``*_get_body2`` accessors stay (they're symmetric with the rest of
# the per-type accessors) but they all collapse to the same load --
# they're effectively aliases of :func:`constraint_get_body1` /
# :func:`constraint_get_body2`.
#
# The contract is checked at import time by
# :func:`assert_constraint_header` -- if a future edit reorders fields,
# the importing module's ``import`` fails loudly instead of silently
# corrupting body indices at runtime.
#
# Type tags: add new tags here -- monotonically increasing, no holes
# -- and mirror the assignment in the per-type module's
# ``constraint_set_type`` call. Don't reuse retired tag values; bump
# the highest one instead so any stale persisted state shows up loudly.

#: Sentinel for unwritten / cleared columns.
CONSTRAINT_TYPE_INVALID = wp.constant(wp.int32(0))
CONSTRAINT_TYPE_BALL_SOCKET = wp.constant(wp.int32(1))
CONSTRAINT_TYPE_HINGE_ANGLE = wp.constant(wp.int32(2))
CONSTRAINT_TYPE_ANGULAR_MOTOR = wp.constant(wp.int32(3))
#: Fused single-column hinge joint (HingeAngle + BallSocket + AngularMotor).
#: A single PGS thread owns the (body1, body2) pair for the entire hinge,
#: which lets the three sub-iterations share one body-data load and lets
#: the partitioner colour one hinge per partition (instead of three),
#: dramatically improving convergence on heavily-loaded chains.
CONSTRAINT_TYPE_HINGE_JOINT = wp.constant(wp.int32(4))
#: Plain 5-DoF double-ball-socket hinge lock. A pair of BallSocket rows
#: sharing one column, anchored at two world-space points along the
#: hinge axis. Locks all relative linear DoFs; the single remaining
#: free DoF is rotation about the line through the two anchors. Pair
#: with :data:`CONSTRAINT_TYPE_ANGULAR_MOTOR` (in velocity or PD mode)
#: for a motorised revolute joint assembled from standalone pieces --
#: the test harness exercises this path to isolate the motor from the
#: integrated :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET`. See
#: :mod:`constraint_double_ball_socket` for the rank-5 Schur derivation.
CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET = wp.constant(wp.int32(5))
#: Legacy 5-DoF prismatic (slider) joint. Historically :mod:`constraint_prismatic`
#: used a mixed quaternion-error + tangent-translation formulation (3 rotational
#: rows + 2 tangent rows, 3x3 + 2x2 Schur). Kept around so users can compare
#: against the newer pure-point ``CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET``
#: prismatic mode side-by-side (see ``example_motorized_prismatic_chain``).
#: New code should prefer the unified joint.
CONSTRAINT_TYPE_PRISMATIC = wp.constant(wp.int32(6))
#: 6-DoF generalised joint (a.k.a. "D6") -- *all* 6 relative DoF (3 angular
#: + 3 linear) solved as one column via a 6x6 Schur complement (3x3 + 3x3
#: + 3x3 cross-block). Each axis carries an independent implicit-PD drive
#: (position-target + velocity-target + force/torque cap), expressed via
#: the Box2D / Bepu / Nordby ``soft_constraint_coefficients`` plumbing,
#: so a single descriptor covers rigid weld -> soft lock -> position PD
#: drive -> velocity drive -> free axis without changing matrix shape.
#: See :mod:`constraint_d6` for the derivation.
CONSTRAINT_TYPE_D6 = wp.constant(wp.int32(7))
#: Unified revolute / prismatic joint: a 5-DoF pure-point positional lock
#: (expressed as a double ball-socket in revolute mode or a 2+2+1
#: tangent-plane triad in prismatic mode, both solved as one PGS column
#: via a rank-5 Schur complement) plus an optional scalar actuator row
#: that drives the free axis with a soft *position* or *velocity*
#: setpoint and clamps it to a ``[min_value, max_value]`` interval (one-
#: sided spring-damper limits, Box2D / Bepu soft formulation). Limits /
#: drive are interpreted as angles [rad] in revolute mode and as
#: displacements [m] in prismatic mode. See
#: :mod:`constraint_actuated_double_ball_socket` for the math.
CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET = wp.constant(wp.int32(8))
#: Rigid-rigid contact constraint -- packs up to **6 contacts** belonging to
#: one ``(shape_a, shape_b)`` pair into a single PGS column. Newton's
#: CollisionPipeline sorts contacts by ``(shape_a, shape_b)`` (see
#: :func:`newton._src.geometry.contact_data.make_contact_sort_key`), so each
#: column just records a contiguous range ``[contact_first, contact_first +
#: contact_count)`` into the sorted ``Contacts`` buffer plus a per-slot
#: ``active_mask`` bit set (slots 0..5). Pairs with > 6 contacts are split
#: across ``N = ceil(count / 6)`` adjacent columns; pairs with < 6 contacts
#: leave the trailing slots with ``active_mask = 0`` so the PGS loop
#: early-outs on them without any branch divergence penalty (all lanes in
#: a warp see the same ``active_mask`` because each lane owns its own cid).
#: Persistent state (warm-start accumulated impulses) lives in a parallel
#: :class:`ContactContainer` keyed by the same cid; the column itself is
#: pure geometry + header, so contacts can be fully re-ingested each step
#: without touching the persistent lambdas. See
#: :mod:`constraint_contact` for the schema and per-iteration math.
CONSTRAINT_TYPE_CONTACT = wp.constant(wp.int32(9))
#: 1-DoF linear velocity / PD position drive -- the translational twin of
#: :data:`CONSTRAINT_TYPE_ANGULAR_MOTOR`. Acts between two bodies at an
#: anchor point along a world axis; velocity-target mode (Box2D-soft
#: ``hertz`` / ``damping_ratio``) or PD position-target mode (absolute
#: ``stiffness`` N/m / ``damping`` N*s/m) selected at prepare time by
#: whether ``stiffness`` or ``damping`` is positive. Pair with
#: :data:`CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC` for a motorised
#: prismatic joint assembled from standalone pieces; the test harness
#: exercises this path to isolate the motor from the integrated
#: :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` prismatic mode.
#: See :mod:`constraint_linear_motor` for the math.
CONSTRAINT_TYPE_LINEAR_MOTOR = wp.constant(wp.int32(10))
#: Plain 5-DoF prismatic lock built from three ball-socket anchors:
#: two on the slide axis (locking two translational DoFs perpendicular
#: to it) plus one auto-derived point offset perpendicular to the axis
#: (a single scalar row locking rotation about the axis). Same rank-5
#: Schur idiom as :data:`CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET` but with a
#: 4+1 block split instead of 3+2; the free DoF is translation along
#: the slide axis. Pair with :data:`CONSTRAINT_TYPE_LINEAR_MOTOR` (in
#: velocity or PD mode) for a motorised prismatic joint assembled from
#: standalone pieces; compare to the fused
#: :data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` prismatic path.
#: See :mod:`constraint_double_ball_socket_prismatic` for the math.
CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC = wp.constant(wp.int32(11))
#: 1-DoF angular limit -- one-sided or two-sided spring-damper clamp on
#: the relative twist angle between two bodies, along a fixed hinge
#: axis. Tracks the cumulative unbounded angle via
#: :func:`math_helpers.revolution_tracker_update`. Supports both
#: :func:`soft_constraint_coefficients` (``hertz`` / ``damping_ratio``)
#: and :func:`pd_coefficients` (``stiffness`` N*m/rad / ``damping``
#: N*m*s/rad) conventions selected at descriptor time. See
#: :mod:`constraint_angular_limit`.
CONSTRAINT_TYPE_ANGULAR_LIMIT = wp.constant(wp.int32(12))
#: 1-DoF linear limit -- translational twin of
#: :data:`CONSTRAINT_TYPE_ANGULAR_LIMIT`. Clamps the relative slide
#: between two bodies along a fixed axis to ``[min_value, max_value]``
#: with a spring-damper penalty. Same dual convention (hertz /
#: damping_ratio OR stiffness / damping). See :mod:`constraint_linear_limit`.
CONSTRAINT_TYPE_LINEAR_LIMIT = wp.constant(wp.int32(13))

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
        c.data[off + 0, cid],  c.data[off + 1, cid],  c.data[off + 2, cid],  c.data[off + 3, cid],
        c.data[off + 4, cid],  c.data[off + 5, cid],  c.data[off + 6, cid],  c.data[off + 7, cid],
        c.data[off + 8, cid],  c.data[off + 9, cid],  c.data[off + 10, cid], c.data[off + 11, cid],
        c.data[off + 12, cid], c.data[off + 13, cid], c.data[off + 14, cid], c.data[off + 15, cid],
    )


@wp.func
def write_mat44(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.mat44f):
    c.data[off + 0, cid]  = v[0, 0]
    c.data[off + 1, cid]  = v[0, 1]
    c.data[off + 2, cid]  = v[0, 2]
    c.data[off + 3, cid]  = v[0, 3]
    c.data[off + 4, cid]  = v[1, 0]
    c.data[off + 5, cid]  = v[1, 1]
    c.data[off + 6, cid]  = v[1, 2]
    c.data[off + 7, cid]  = v[1, 3]
    c.data[off + 8, cid]  = v[2, 0]
    c.data[off + 9, cid]  = v[2, 1]
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
# All bilateral joints in this solver share one formulation of "softness":
# the user specifies an undamped natural frequency ``hertz`` (Hz) and a
# non-dimensional ``damping_ratio`` (1 = critically damped) and the per-
# substep PGS coefficients are computed from those plus the current
# substep ``dt``. This makes the joint behaviour
#
#   * mass-independent  (the body inertia drops out of the spring math),
#   * time-step-independent (recomputed every prepare from the actual dt),
#
# which is the textbook fix for the "world-anchor joint goes unstable
# when you halve the substep" failure mode of plain Baumgarte / fixed-
# CFM solvers. See Box2D's solver2d post for the derivation:
# https://box2d.org/posts/2024/02/solver2d/  (the "Soft Constraints"
# section). The three returned coefficients plug straight into PGS:
#
#     incremental_lambda = -mass_coeff * effective_mass * (jv + bias_rate * separation)
#                          - impulse_coeff * accumulated_impulse
#
# Equivalently, the second term replaces the legacy ``softness * idt *
# accumulated_impulse`` "softness vector" -- it lets the accumulated
# impulse leak away over time so the joint asymptotes to the spring-
# damper response instead of the perfectly-rigid one.
#
# Recommended hertz: keep it at ~ (substep_rate / 4) or below
# (Nyquist + a safety factor). For a 60Hz frame rate at 8 substeps that's
# substep_rate = 480Hz, so hertz <= 120 stays comfortably stable. The
# defaults below sit at hertz = 60 / 30 / 30, well inside that envelope
# and rigid-feeling.

#: Critically damped by default -- no overshoot, no underdamped ringing.
DEFAULT_DAMPING_RATIO = wp.constant(wp.float32(1.0))
#: Linear (positional) joint stiffness target. Matches the Box2D v3
#: default (60 Hz) and, together with the solver's Box2D v3 TGS-soft
#: relax pass (``_constraint_relax_kernel``), closes positional drift
#: without bleeding drive / limit / spring impulse into the anchor.
#:
#: Historical note: this used to default to ``0`` (rigid PGS) because
#: a soft anchor with no relax pass would leak up to ~70% of a paired
#: drive impulse on bodies whose COM is offset from the anchor (the
#: lever-bob setup). The Box2D v3 "solve-with-bias then solve-without-
#: bias" substep loop implemented in :meth:`~newton._src.solvers.jitter.solver_jitter.World._relax_velocities`
#: eliminates that leak, so we can default to a finite, drift-closing
#: stiffness again. Users who want a truly rigid anchor (no drift
#: correction) can still pass ``hertz=0`` explicitly.
DEFAULT_HERTZ_LINEAR = wp.constant(wp.float32(60.0))
#: Angular (rotational lock) joint stiffness. Same reasoning as
#: :data:`DEFAULT_HERTZ_LINEAR`: 60 Hz closes angular drift, and the
#: relax pass prevents impulse leakage from motors / limits into the
#: rotational lock. Override per-joint with ``hertz=0`` for a fully
#: rigid lock or with a lower value for a compliant rotational joint.
DEFAULT_HERTZ_ANGULAR = wp.constant(wp.float32(60.0))
#: Hinge / cone limit stiffness. Same conservative value as the angular
#: hinge so a body hitting its limit doesn't bounce.
DEFAULT_HERTZ_LIMIT = wp.constant(wp.float32(30.0))
#: Velocity-target motors don't carry a positional bias, so hertz here
#: only modulates the impulse-coefficient (how aggressively the motor
#: clamps to the target velocity). Slightly higher than the angular lock
#: so motors feel responsive without ringing.
DEFAULT_HERTZ_MOTOR = wp.constant(wp.float32(60.0))


@wp.func
def soft_constraint_coefficients(
    hertz: wp.float32,
    damping_ratio: wp.float32,
    dt: wp.float32,
):
    """Map ``(hertz, damping_ratio, dt)`` to PGS soft-constraint coefficients.

    Direct port of the formulation in
    https://box2d.org/posts/2024/02/solver2d/ ("Soft Constraints",
    pseudo-code block) which Erin Catto credits to Ross Nordby of Bepu.
    The inputs are the user-facing knobs (independent of mass and
    time step); the outputs feed straight into the PGS update.

    Returns ``(bias_rate, mass_coeff, impulse_coeff)``:

      * ``bias_rate``    [1/s] -- multiplies a positional separation
        to produce a velocity-correction bias.
      * ``mass_coeff``   [-]   -- scales the unsoftened ``effective_mass``
        in the per-iteration impulse calculation, lies in ``[0, 1]``.
      * ``impulse_coeff``[-]   -- damps the accumulated impulse term
        ("softness leak"), also in ``[0, 1]``.

    As ``hertz -> infinity`` (or ``dt -> 0``) ``mass_coeff -> 1`` and
    ``impulse_coeff -> 0``, recovering the rigid-constraint PGS update.
    Lowering ``hertz`` makes the joint progressively softer.

    Setting ``hertz <= 0`` produces a *rigid* constraint:
    ``bias_rate = 0``, ``mass_coeff = 1``, ``impulse_coeff = 0`` --
    exactly the un-softened plain-PGS update.
    """
    if hertz <= 0.0:
        return wp.float32(0.0), wp.float32(1.0), wp.float32(0.0)

    omega = 2.0 * 3.14159265358979 * hertz
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

    Implicit-Euler spring-damper formulation transcribed verbatim from
    Jitter2's :class:`SpringConstraint.SetSpringParameters` (SoftBodies/
    SpringConstraint.cs). The user specifies absolute physical gains
    :math:`k` and :math:`c`:

      * ``stiffness`` ``k`` in [N/m] for linear rows or [N*m/rad] for
        angular rows -- the restoring force per unit position error.
        The applied force on the constraint row is ``-k * C`` with
        ``C`` the current position error.
      * ``damping`` ``c`` in [N*s/m] or [N*m*s/rad] -- the viscous
        force per unit relative velocity. The applied force is
        ``-c * v_rel``.

    unlike the Box2D / Bepu :func:`soft_constraint_coefficients` pair
    ``(hertz, damping_ratio)`` which bake the effective mass into the
    gains. For the same behaviour across bodies of different mass this
    PD form must be *rescaled by mass*, but it lets the user directly
    write "F = -k*x - c*v" without knowing ``m_eff`` up front.

    Jitter2's ``SetSpringParameters`` stores two scalars:

    .. code-block:: text

        Softness   = 1 / (c + dt * k)
        BiasFactor = dt * k * Softness

    and then every substep computes (``idt = 1/dt``)

    .. code-block:: text

        # prepare
        EffectiveMass = 1 / (M_inv + Softness * idt)
        Bias          = C * BiasFactor * idt
        # iterate
        softScalar    = acc * Softness * idt
        lambda        = -EffectiveMass * (jv + Bias + softScalar)

    To keep the kernel side uniform we return the three scalars
    ``(gamma, bias, eff_mass_softened)`` that absorb the ``idt``
    factor so the iterate is a single PGS update:

      * ``gamma``             = ``Softness * idt``  [s^-1]
      * ``bias``              = ``C * BiasFactor * idt``  [m/s or rad/s]
      * ``eff_mass_softened`` = ``1 / (M_inv + gamma)``  [kg or kg*m^2]

    The iterate then reduces to

    .. math::
        \\lambda = -M_{\\text{eff,soft}}\\,
            (J v - \\text{bias}_{\\text{signed}} + \\gamma \\lambda_{\\text{acc}})

    with ``bias_signed`` sign-flipped by the caller to match the
    constraint's Jacobian convention (see the per-joint prepare for
    the sign rule).

    At ``k = c = 0`` all three outputs are zero and the PGS row is a
    no-op -- the caller's iterate guards on the "drive off" flag to
    skip the solve entirely.

    Args:
        stiffness: Positional stiffness ``k``; expected non-negative.
            ``0`` turns off the spring part (pure damper).
        damping: Viscous damping ``c``; expected non-negative.
            ``0`` turns off the damper (pure spring).
        position_error: Current position error ``C = actual - target``
            (rad for revolute, m for prismatic). Folded into ``bias``
            per Jitter2 SpringConstraint.
        eff_mass_inv: Reciprocal of the raw (unsoftened) Jacobian-
            weighted effective mass of the constraint row (i.e.
            ``J M^{-1} J^T``). ``0`` (both bodies static or Jacobian
            null) short-circuits to all-zeros -- the PGS update is a
            no-op.
        dt: Substep time step [s]. Must be > 0.
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


