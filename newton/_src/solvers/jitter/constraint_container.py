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
    "CONSTRAINT_TYPE_ANGULAR_MOTOR",
    "CONSTRAINT_TYPE_BALL_SOCKET",
    "CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET",
    "CONSTRAINT_TYPE_HINGE_ANGLE",
    "CONSTRAINT_TYPE_HINGE_JOINT",
    "CONSTRAINT_TYPE_INVALID",
    "CONSTRAINT_TYPE_OFFSET",
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
    "read_quat",
    "read_vec3",
    "write_float",
    "write_int",
    "write_mat33",
    "write_quat",
    "write_vec3",
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
#: Fused two-anchor "double ball-socket" hinge: the entire 5-DoF joint
#: solved as one column via a Schur-complement (3x3 + 2x2) instead of
#: two redundant 3-row ball-sockets. See
#: :mod:`constraint_double_ball_socket` for the math.
CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET = wp.constant(wp.int32(5))

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
            constraint_type: wp.int32   # dword 0
            body1:           wp.int32   # dword 1
            body2:           wp.int32   # dword 2
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
        c.data[off + 0, cid], c.data[off + 1, cid], c.data[off + 2, cid],
        c.data[off + 3, cid], c.data[off + 4, cid], c.data[off + 5, cid],
        c.data[off + 6, cid], c.data[off + 7, cid], c.data[off + 8, cid],
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
