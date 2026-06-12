# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Column-major dword-packed storage for constraint state.

One ``wp.array2d[wp.float32]`` shaped ``(num_dwords, num_constraints)``; cid
is the inner dim so a row read coalesces across the warp. Dword offsets come
from :func:`dword_offset_of` at import time. Per-cid type tag at dword 0
drives dispatch.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.helpers.array_access import read2d_f32, write2d_f32
from newton._src.solvers.phoenx.helpers.data_packing import (
    dword_offset_of,
    reinterpret_float_as_int,
    reinterpret_int_as_float,
)

__all__ = [
    "CONSTRAINT_BODY1_OFFSET",
    "CONSTRAINT_BODY2_OFFSET",
    "CONSTRAINT_MULTIPLIER_DWORDS",
    "CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET",
    "CONSTRAINT_TYPE_CLOTH_BENDING",
    "CONSTRAINT_TYPE_CLOTH_TRIANGLE",
    "CONSTRAINT_TYPE_CONTACT",
    "CONSTRAINT_TYPE_INVALID",
    "CONSTRAINT_TYPE_OFFSET",
    "CONSTRAINT_TYPE_SOFT_HEXAHEDRON",
    "CONSTRAINT_TYPE_SOFT_TETRAHEDRON",
    "CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN",
    "DEFAULT_DAMPING_RATIO",
    "DEFAULT_HERTZ_ANGULAR",
    "DEFAULT_HERTZ_CONTACT",
    "DEFAULT_HERTZ_LIMIT",
    "DEFAULT_HERTZ_LINEAR",
    "DEFAULT_HERTZ_MOTOR",
    "ConstraintBodies",
    "ConstraintContainer",
    "assert_constraint_header",
    "constraint_accumulate_time_us",
    "constraint_bodies_make",
    "constraint_container_zeros",
    "constraint_get_body1",
    "constraint_get_body2",
    "constraint_get_type",
    "constraint_read_multiplier",
    "constraint_set_body1",
    "constraint_set_body2",
    "constraint_set_type",
    "constraint_write_multiplier",
    "pd_coefficients",
    "read_float",
    "read_int",
    "read_mat22",
    "read_mat33",
    "read_mat44",
    "read_quat",
    "read_vec3",
    "read_vec4",
    "soft_constraint_coefficients",
    "write_float",
    "write_int",
    "write_mat22",
    "write_mat33",
    "write_mat44",
    "write_quat",
    "write_vec3",
    "write_vec4",
]


# Constraint header: every schema starts with int32 (constraint_type, body1,
# body2) at dwords 0/1/2. Lets the generic dispatcher route any column.
# Type tags: monotonic, never reuse retired values.

CONSTRAINT_TYPE_INVALID = wp.constant(wp.int32(0))
#: Unified revolute/prismatic/ball-socket/fixed/cable joint. 5-DoF positional
#: lock + optional scalar actuator row.
CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET = wp.constant(wp.int32(8))
#: Rigid-rigid contact, one column per (shape_a, shape_b). Warm-start state
#: lives in :class:`ContactContainer`.
CONSTRAINT_TYPE_CONTACT = wp.constant(wp.int32(9))
#: XPBD cloth triangle (FemTriPBD area + shear rows). Endpoints body1/body2/body3
#: are particle indices. See :mod:`constraint_cloth_triangle`.
CONSTRAINT_TYPE_CLOTH_TRIANGLE = wp.constant(wp.int32(10))
#: XPBD soft-body tetrahedron (FemTetPBD corotational shear row). Endpoints
#: body1..body4 are particle indices. See :mod:`constraint_soft_tetrahedron`.
CONSTRAINT_TYPE_SOFT_TETRAHEDRON = wp.constant(wp.int32(11))
#: XPBD cloth bending hinge (Bergou / Wardetzky 2007 quadratic curvature
#: energy). Endpoints body1..body4 are particle indices. body1 / body2 are
#: opposite vertices of the two triangles; body3 / body4 are the shared
#: edge. See :mod:`constraint_cloth_bending`.
CONSTRAINT_TYPE_CLOTH_BENDING = wp.constant(wp.int32(12))
#: Block stable Neo-Hookean XPBD soft-body tetrahedron (Ton-That, Kry &
#: Andrews 2024 ``Parallel Block Neo-Hookean XPBD using Graph Clustering``).
#: Hydrostatic + deviatoric constraints solved as a coupled 2x2 Schur
#: system. Endpoints body1..body4 are particle indices. See
#: :mod:`constraint_soft_tet_neohookean`.
CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN = wp.constant(wp.int32(13))
#: 8-node trilinear hex soft-body constraint with selectable integrated
#: strain and reduced center volume rows. Endpoints body1..body8 are
#: particle indices. See :mod:`constraint_soft_hexahedron`.
CONSTRAINT_TYPE_SOFT_HEXAHEDRON = wp.constant(wp.int32(14))

CONSTRAINT_TYPE_OFFSET = wp.constant(wp.int32(0))
CONSTRAINT_BODY1_OFFSET = wp.constant(wp.int32(1))
CONSTRAINT_BODY2_OFFSET = wp.constant(wp.int32(2))


def assert_constraint_header(struct_type: object) -> None:
    """Validate the (constraint_type, body1, body2) header at dwords 0/1/2.
    Called at import time by each per-type module."""
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


#: Width of the per-cid slot cache. Matches
#: :data:`newton._src.solvers.phoenx.graph_coloring.graph_coloring_common.MAX_BODIES`
#: (the largest possible endpoint count -- soft-hex with 8 vertices).
#: Keeping it at the same upper bound means a single cache schema fits
#: every constraint type that uses mass-splitting slot routing.
SLOT_CACHE_MAX_BODIES: int = 8

#: Width of the per-cid mutable multiplier sidecar. This is intentionally
#: generic storage: constraint-family modules define their own row mapping.
CONSTRAINT_MULTIPLIER_DWORDS: int = 12


@wp.struct
class ConstraintContainer:
    """``data`` shape: (num_dwords, num_constraints), cid is inner dim for
    coalesced loads.

    ``slot_cache`` / ``count_cache`` are the per-cid slot / Tonge-count
    tables stamped once per frame by
    :func:`build_constraint_slot_cache`. The constraint iterates read
    them directly instead of calling :func:`get_state_index` 2-8 times
    per call, eliminating the dependent body-load -> slot-lookup chain
    on the hot path.
    """

    data: wp.array2d[wp.float32]
    #: Mutable accumulated multipliers, separated from read-mostly prepared data.
    multipliers: wp.array2d[wp.float32]
    #: Per-cid slot cache: ``slot_cache[cid, v]`` is the slot index for
    #: vertex ``v`` of constraint ``cid`` under the parallel-id that
    #: vertex's iterate will see (parallel_id = 0 for regular colours,
    #: ``csr_offset_in_overflow / ms_batch_size`` for overflow). ``-1``
    #: means no slot (direct body/particle storage).
    slot_cache: wp.array2d[wp.int32]
    #: Per-cid Tonge mass-divisor cache: ``count_cache[cid, v]`` is the
    #: number of copy-state slots vertex ``v`` occupies. Iterate kernels
    #: ignore this (slot is enough); prepare kernels use it to scale
    #: ``inv_mass`` by ``1 / N``.
    count_cache: wp.array2d[wp.int32]


def constraint_container_zeros(
    num_constraints: int,
    num_dwords: int,
    device: wp.DeviceLike = None,
) -> ConstraintContainer:
    """Allocate a zero-initialised :class:`ConstraintContainer`."""
    c = ConstraintContainer()
    c.data = wp.zeros((num_dwords, num_constraints), dtype=wp.float32, device=device)
    c.multipliers = wp.zeros((CONSTRAINT_MULTIPLIER_DWORDS, max(num_constraints, 1)), dtype=wp.float32, device=device)
    # Cache starts at the no-slot fallback so reads before the first
    # build hit the same direct-storage path as ``get_state_index``'s
    # mass-splitting-disabled fast path.
    c.slot_cache = wp.full(
        (max(num_constraints, 1), SLOT_CACHE_MAX_BODIES),
        value=-1,
        dtype=wp.int32,
        device=device,
    )
    c.count_cache = wp.full(
        (max(num_constraints, 1), SLOT_CACHE_MAX_BODIES),
        value=1,
        dtype=wp.int32,
        device=device,
    )
    return c


# Column-major dword accessors. vec3/quat/mat33 are 3/4/9 individual dwords
# (no padding); the warp-coalesce is the win, not per-thread fusion.


@wp.func
def read_float(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.float32:
    return read2d_f32(c.data, off, cid)


@wp.func
def write_float(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.float32):
    write2d_f32(c.data, off, cid, v)


@wp.func
def constraint_read_multiplier(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.float32:
    return read2d_f32(c.multipliers, off, cid)


@wp.func
def constraint_write_multiplier(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.float32):
    write2d_f32(c.multipliers, off, cid, v)


@wp.func
def read_int(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.int32:
    """Bit-cast read of an int field stored in the float buffer."""
    return reinterpret_float_as_int(read2d_f32(c.data, off, cid))


@wp.func
def write_int(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.int32):
    """Bit-cast write of an int field into the float buffer."""
    write2d_f32(c.data, off, cid, reinterpret_int_as_float(v))


@wp.func
def read_vec3(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(c.data, off + 0, cid),
        read2d_f32(c.data, off + 1, cid),
        read2d_f32(c.data, off + 2, cid),
    )


@wp.func
def write_vec3(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.vec3f):
    write2d_f32(c.data, off + 0, cid, v[0])
    write2d_f32(c.data, off + 1, cid, v[1])
    write2d_f32(c.data, off + 2, cid, v[2])


@wp.func
def read_quat(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.quatf:
    return wp.quatf(
        read2d_f32(c.data, off + 0, cid),
        read2d_f32(c.data, off + 1, cid),
        read2d_f32(c.data, off + 2, cid),
        read2d_f32(c.data, off + 3, cid),
    )


@wp.func
def write_quat(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.quatf):
    write2d_f32(c.data, off + 0, cid, v[0])
    write2d_f32(c.data, off + 1, cid, v[1])
    write2d_f32(c.data, off + 2, cid, v[2])
    write2d_f32(c.data, off + 3, cid, v[3])


@wp.func
def read_mat22(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.mat22f:
    """Row-major 2x2 from 4 consecutive dwords."""
    return wp.mat22f(
        read2d_f32(c.data, off + 0, cid),
        read2d_f32(c.data, off + 1, cid),
        read2d_f32(c.data, off + 2, cid),
        read2d_f32(c.data, off + 3, cid),
    )


@wp.func
def write_mat22(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.mat22f):
    write2d_f32(c.data, off + 0, cid, v[0, 0])
    write2d_f32(c.data, off + 1, cid, v[0, 1])
    write2d_f32(c.data, off + 2, cid, v[1, 0])
    write2d_f32(c.data, off + 3, cid, v[1, 1])


@wp.func
def read_mat33(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.mat33f:
    """Row-major 3x3 from 9 consecutive dwords."""
    return wp.mat33f(
        read2d_f32(c.data, off + 0, cid),
        read2d_f32(c.data, off + 1, cid),
        read2d_f32(c.data, off + 2, cid),
        read2d_f32(c.data, off + 3, cid),
        read2d_f32(c.data, off + 4, cid),
        read2d_f32(c.data, off + 5, cid),
        read2d_f32(c.data, off + 6, cid),
        read2d_f32(c.data, off + 7, cid),
        read2d_f32(c.data, off + 8, cid),
    )


@wp.func
def write_mat33(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.mat33f):
    write2d_f32(c.data, off + 0, cid, v[0, 0])
    write2d_f32(c.data, off + 1, cid, v[0, 1])
    write2d_f32(c.data, off + 2, cid, v[0, 2])
    write2d_f32(c.data, off + 3, cid, v[1, 0])
    write2d_f32(c.data, off + 4, cid, v[1, 1])
    write2d_f32(c.data, off + 5, cid, v[1, 2])
    write2d_f32(c.data, off + 6, cid, v[2, 0])
    write2d_f32(c.data, off + 7, cid, v[2, 1])
    write2d_f32(c.data, off + 8, cid, v[2, 2])


@wp.func
def read_vec4(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.vec4f:
    return wp.vec4f(
        read2d_f32(c.data, off + 0, cid),
        read2d_f32(c.data, off + 1, cid),
        read2d_f32(c.data, off + 2, cid),
        read2d_f32(c.data, off + 3, cid),
    )


@wp.func
def write_vec4(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.vec4f):
    write2d_f32(c.data, off + 0, cid, v[0])
    write2d_f32(c.data, off + 1, cid, v[1])
    write2d_f32(c.data, off + 2, cid, v[2])
    write2d_f32(c.data, off + 3, cid, v[3])


@wp.func
def read_mat44(c: ConstraintContainer, off: wp.int32, cid: wp.int32) -> wp.mat44f:
    """Row-major 4x4 from 16 consecutive dwords."""
    return wp.mat44f(
        read2d_f32(c.data, off + 0, cid),
        read2d_f32(c.data, off + 1, cid),
        read2d_f32(c.data, off + 2, cid),
        read2d_f32(c.data, off + 3, cid),
        read2d_f32(c.data, off + 4, cid),
        read2d_f32(c.data, off + 5, cid),
        read2d_f32(c.data, off + 6, cid),
        read2d_f32(c.data, off + 7, cid),
        read2d_f32(c.data, off + 8, cid),
        read2d_f32(c.data, off + 9, cid),
        read2d_f32(c.data, off + 10, cid),
        read2d_f32(c.data, off + 11, cid),
        read2d_f32(c.data, off + 12, cid),
        read2d_f32(c.data, off + 13, cid),
        read2d_f32(c.data, off + 14, cid),
        read2d_f32(c.data, off + 15, cid),
    )


@wp.func
def write_mat44(c: ConstraintContainer, off: wp.int32, cid: wp.int32, v: wp.mat44f):
    write2d_f32(c.data, off + 0, cid, v[0, 0])
    write2d_f32(c.data, off + 1, cid, v[0, 1])
    write2d_f32(c.data, off + 2, cid, v[0, 2])
    write2d_f32(c.data, off + 3, cid, v[0, 3])
    write2d_f32(c.data, off + 4, cid, v[1, 0])
    write2d_f32(c.data, off + 5, cid, v[1, 1])
    write2d_f32(c.data, off + 6, cid, v[1, 2])
    write2d_f32(c.data, off + 7, cid, v[1, 3])
    write2d_f32(c.data, off + 8, cid, v[2, 0])
    write2d_f32(c.data, off + 9, cid, v[2, 1])
    write2d_f32(c.data, off + 10, cid, v[2, 2])
    write2d_f32(c.data, off + 11, cid, v[2, 3])
    write2d_f32(c.data, off + 12, cid, v[3, 0])
    write2d_f32(c.data, off + 13, cid, v[3, 1])
    write2d_f32(c.data, off + 14, cid, v[3, 2])
    write2d_f32(c.data, off + 15, cid, v[3, 3])


# Type-agnostic header accessors (work on any column thanks to the dword
# 0/1/2 contract enforced by :func:`assert_constraint_header`).


@wp.func
def constraint_get_type(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    """Read the ``CONSTRAINT_TYPE_*`` tag of constraint ``cid``."""
    return read_int(c, CONSTRAINT_TYPE_OFFSET, cid)


@wp.func
def constraint_set_type(c: ConstraintContainer, cid: wp.int32, t: wp.int32):
    """Write the ``CONSTRAINT_TYPE_*`` tag of constraint ``cid``."""
    write_int(c, CONSTRAINT_TYPE_OFFSET, cid, t)


@wp.func
def constraint_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    """Read body1 of constraint ``cid``."""
    return read_int(c, CONSTRAINT_BODY1_OFFSET, cid)


@wp.func
def constraint_set_body1(c: ConstraintContainer, cid: wp.int32, b: wp.int32):
    write_int(c, CONSTRAINT_BODY1_OFFSET, cid, b)


@wp.func
def constraint_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    """Read body2 of constraint ``cid``."""
    return read_int(c, CONSTRAINT_BODY2_OFFSET, cid)


@wp.func
def constraint_set_body2(c: ConstraintContainer, cid: wp.int32, b: wp.int32):
    write_int(c, CONSTRAINT_BODY2_OFFSET, cid, b)


@wp.func
def constraint_accumulate_time_us(c: ConstraintContainer, time_us_off: wp.int32, cid: wp.int32, t_us: wp.float32):
    """Atomic-add ``t_us`` microseconds into the column's ``time_us`` slot.

    ``time_us_off`` is the per-schema dword offset (e.g. ``ADBS_TIME_US_OFFSET``).
    """
    wp.atomic_add(c.data, time_us_off, cid, t_us)


# Body-pair carrier for composable sub-constraint calls (header is per-column,
# so composites read body1/body2 once and thread through).


@wp.struct
class ConstraintBodies:
    """(body1, body2) carrier threaded through composable constraint funcs."""

    b1: wp.int32
    b2: wp.int32


@wp.func
def constraint_bodies_make(b1: wp.int32, b2: wp.int32) -> ConstraintBodies:
    """Build a :class:`ConstraintBodies` from two body indices."""
    bp = ConstraintBodies()
    bp.b1 = b1
    bp.b2 = b2
    return bp


# Soft-constraint coefficients (Box2D v3 / Bepu / Nordby; see
# https://box2d.org/posts/2024/02/solver2d/). User picks (hertz, damping_ratio).
# omega = 2*pi*hertz is Nyquist-clamped (pi/dt). Defaults are above any realistic
# Nyquist -> maximally rigid; pass a smaller hertz for compliance, or 0 for a
# rigid PGS row with no drift correction.

DEFAULT_DAMPING_RATIO = wp.constant(wp.float32(1.0))
DEFAULT_HERTZ_LINEAR = wp.constant(wp.float32(1.0e9))
DEFAULT_HERTZ_ANGULAR = wp.constant(wp.float32(1.0e9))
DEFAULT_HERTZ_LIMIT = wp.constant(wp.float32(1.0e9))

#: Cap on per-row ``nyquist_boost``. Boost N allows stiffness up to
#: N / (M_inv * dt^2); N=1 is the strict implicit-Euler bound.
_PD_NYQUIST_HEADROOM_MAX = wp.constant(wp.float32(10.0))
DEFAULT_HERTZ_MOTOR = wp.constant(wp.float32(1.0e9))
DEFAULT_HERTZ_CONTACT = wp.constant(wp.float32(1.0e9))


@wp.func
def soft_constraint_coefficients(
    hertz: wp.float32,
    damping_ratio: wp.float32,
    dt: wp.float32,
):
    """Box2D-v3 PGS soft-constraint coefficients. omega = 2*pi*hertz clamped to
    Nyquist (pi/dt). Returns (bias_rate, mass_coeff, impulse_coeff). hertz <= 0
    -> rigid (0, 1, 0)."""
    if hertz <= 0.0:
        return wp.float32(0.0), wp.float32(1.0), wp.float32(0.0)

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
    nyquist_boost: wp.float32,
):
    """Jitter2 implicit-Euler spring-damper triple (k absolute, not mass-baked).

    Returns ``(gamma, bias, eff_mass_softened)`` (idt folded in):
        gamma = 1 / (c + dt*k) * idt
        bias  = C * dt*k / (c + dt*k) * idt
        eff   = 1 / (M_inv + gamma)
    Yielding ``lambda = -eff * (Jv - bias_signed + gamma * lambda_acc)``.
    Short-circuits to (0,0,0) when M_inv<=0 or both gains zero.
    """
    if eff_mass_inv <= 0.0:
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    if stiffness <= 0.0 and damping <= 0.0:
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)

    # Nyquist clamp on k: beyond ~1/(M_inv * dt^2) PGS sees a rigid row.
    boost = wp.clamp(nyquist_boost, wp.float32(1.0), _PD_NYQUIST_HEADROOM_MAX)
    k_max = boost / (eff_mass_inv * dt * dt)
    k_clamped = wp.min(stiffness, k_max)

    softness = wp.float32(1.0) / (damping + dt * k_clamped)
    bias_factor = dt * k_clamped * softness
    idt = wp.float32(1.0) / dt

    gamma = softness * idt
    bias = position_error * bias_factor * idt
    eff_mass_soft = wp.float32(1.0) / (eff_mass_inv + gamma)
    return gamma, bias, eff_mass_soft


@wp.func
def pd_coefficients_split(
    stiffness: wp.float32,
    damping: wp.float32,
    position_error: wp.float32,
    eff_mass_inv: wp.float32,
    dt: wp.float32,
    nyquist_boost: wp.float32,
):
    """XPBD-style spring/damping split of :func:`pd_coefficients`. Returns
    ``(gamma_s, bias_s, eff_mass_s, damp_mass)``. Main solve uses spring triple
    + bias=True; relax uses damp_mass + bias=False. Decouples k and c so
    convergence is independent of damping; equivalent steady state up to O(dt^2)."""
    if eff_mass_inv <= 0.0:
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    if stiffness <= 0.0 and damping <= 0.0:
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)

    idt = wp.float32(1.0) / dt

    # Spring-only soft (damping=0); inlined to share the Nyquist clamp.
    gamma_s = wp.float32(0.0)
    bias_s = wp.float32(0.0)
    eff_mass_s = wp.float32(0.0)
    if stiffness > 0.0:
        boost = wp.clamp(nyquist_boost, wp.float32(1.0), _PD_NYQUIST_HEADROOM_MAX)
        k_max = boost / (eff_mass_inv * dt * dt)
        k_clamped = wp.min(stiffness, k_max)
        softness_s = wp.float32(1.0) / (dt * k_clamped)
        gamma_s = softness_s * idt
        # bias_factor = dt*k / (0 + dt*k) = 1 when damping = 0.
        bias_s = position_error * idt
        eff_mass_s = wp.float32(1.0) / (eff_mass_inv + gamma_s)

    # Damping-only: implicit Euler -> damp_mass = dt*c / (1 + dt*c*M_inv).
    damp_mass = wp.float32(0.0)
    if damping > 0.0:
        damp_mass = (dt * damping) / (wp.float32(1.0) + dt * damping * eff_mass_inv)

    return gamma_s, bias_s, eff_mass_s, damp_mass
