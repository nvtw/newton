# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level XPBD cloth bending: dihedral-angle hinge (PhysX style).

Direct port of PhysX's ``bendingEnergySolvePerTrianglePair``
(``physx/source/gpusimulationcontroller/src/CUDA/FEMClothUtil.cuh:543``).
Constraint is

    C(x) = clamp(atan2(sin θ, cos θ) - θ_rest, -π/2, π/2)

where ``θ`` is the dihedral angle between the two triangles sharing
edge ``(x2, x3)``, and ``(x0, x1)`` are the opposite vertices. The
``atan2`` form is robust at θ ≈ 0 (singular for the half-edge sine
formulation), and the ±π/2 clamp prevents the gradient blow-up at
θ ≈ ±π that the unclamped dihedral hinge suffers from.

Per-vertex gradients (PhysX derivation)::

    e23 = x3 - x2
    n0 = (x2 - x0) x (x3 - x0)         (unnormalised, scaled by 2*A_T1)
    n1 = (x3 - x1) x (x2 - x1)         (unnormalised, scaled by 2*A_T2)
    s0 = n0 / |n0|^2                   (n0 with inverse magnitude folded in)
    s1 = n1 / |n1|^2
    dC/dx0 = -|e23| * s0
    dC/dx1 = -|e23| * s1
    dC/dx2 = ((x3 - x0) · e23̂) * s0 + ((x3 - x1) · e23̂) * s1
    dC/dx3 = -(((x2 - x0) · e23̂) * s0 + ((x2 - x1) · e23̂) * s1)

(with ``e23̂`` the unit-length edge vector). Rest data:

* ``rest_angle`` (radians) read from ``model.edge_rest_angle``;
* XPBD compliance ``alpha = 1 / k_bend`` where ``k_bend`` is the
  per-edge stiffness from ``model.edge_bending_properties[t, 0]``.

This replaces the prior Bergou/Wardetzky quadratic curvature form
(``E = (3/(A1+A2)) | Σ K_i p_i |²``) to match the PhysX FEMCloth
behaviour the user already has intuition for. The Wardetzky form
was slightly more robust at θ ≈ ±π but couldn't represent non-flat
rest configurations (its rest state is implicit in ``Σ K_i p_rest_i``,
not a user-settable angle).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_POSITION_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_block import (
    PositionRows1,
    block_position_delta_1,
    block_solve_position_rows1,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CLOTH_BENDING,
    ConstraintContainer,
    assert_constraint_header,
    constraint_read_multiplier,
    constraint_write_multiplier,
    read_float,
    read_int,
    write_float,
    write_int,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.mass_splitting.access import (
    read_position_with_slot,
    set_access_mode_with_slot,
    write_position_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "CLOTH_BENDING_DWORDS",
    "ClothBendingData",
    "cloth_bending_init_rows_kernel",
    "cloth_bending_iterate_at",
    "cloth_bending_prepare_for_iteration_at",
    "cloth_bending_set_alpha",
    "cloth_bending_set_body1",
    "cloth_bending_set_body2",
    "cloth_bending_set_body3",
    "cloth_bending_set_body4",
    "cloth_bending_set_rest_angle",
    "cloth_bending_set_type",
]


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_PHOENX_CLOTH_BENDING_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))
_DEGENERATE_EPS = wp.constant(wp.float32(1.0e-12))
_FEMCLOTH_THRESHOLD = wp.constant(wp.float32(1.0e-7))
_HALF_PI = wp.constant(wp.float32(1.5707963267948966))
_TWO_PI = wp.constant(wp.float32(6.283185307179586))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class ClothBendingData:
    """Per-constraint dword layout. PhysX-style dihedral-angle hinge.

    Vertex order matches PhysX ``bendingEnergySolvePerTrianglePair``:

    * ``body1`` = ``x0`` = opposite vertex of triangle T1 (= Newton ``o0``)
    * ``body2`` = ``x1`` = opposite vertex of triangle T2 (= Newton ``o1``)
    * ``body3`` = ``x2`` = first shared-edge vertex (= Newton ``v1``)
    * ``body4`` = ``x3`` = second shared-edge vertex (= Newton ``v2``)
    """

    constraint_type: wp.int32
    body1: wp.int32  # x0: T1 opposite
    body2: wp.int32  # x1: T2 opposite
    body3: wp.int32  # x2: shared edge vertex 1
    body4: wp.int32  # x3: shared edge vertex 2

    rest_angle: wp.float32  # rest dihedral angle [rad]; 0 for flat cloth
    alpha: wp.float32  # XPBD compliance = 1 / bend stiffness
    lambda_sum: wp.float32  # scalar XPBD lambda accumulator (1D constraint)

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32
    inv_mass_d: wp.float32

    #: Opt-in per-column wall-clock accumulator (microseconds). See
    #: :func:`constraint_accumulate_time_us`.
    time_us: wp.float32


assert_constraint_header(ClothBendingData)


_OFF_BODY1 = wp.constant(dword_offset_of(ClothBendingData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ClothBendingData, "body2"))
_OFF_BODY3 = wp.constant(dword_offset_of(ClothBendingData, "body3"))
_OFF_BODY4 = wp.constant(dword_offset_of(ClothBendingData, "body4"))
_OFF_REST_ANGLE = wp.constant(dword_offset_of(ClothBendingData, "rest_angle"))
_OFF_ALPHA = wp.constant(dword_offset_of(ClothBendingData, "alpha"))
_OFF_LAMBDA_SUM = wp.constant(dword_offset_of(ClothBendingData, "lambda_sum"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_c"))
_OFF_INV_MASS_D = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_d"))
CLOTH_BENDING_TIME_US_OFFSET = wp.constant(dword_offset_of(ClothBendingData, "time_us"))

_MUL_LAMBDA_SUM = wp.constant(wp.int32(0))

CLOTH_BENDING_DWORDS: int = num_dwords(ClothBendingData)


@wp.func
def _read_lambda_sum(constraints: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return constraint_read_multiplier(constraints, _MUL_LAMBDA_SUM, cid)


@wp.func
def _write_lambda_sum(constraints: ConstraintContainer, cid: wp.int32, v: wp.float32):
    constraint_write_multiplier(constraints, _MUL_LAMBDA_SUM, cid, v)


@wp.func
def cloth_bending_set_type(c: ConstraintContainer, cid: wp.int32):
    write_int(c, wp.int32(0), cid, CONSTRAINT_TYPE_CLOTH_BENDING)


@wp.func
def cloth_bending_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def cloth_bending_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def cloth_bending_set_body3(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY3, cid, v)


@wp.func
def cloth_bending_set_body4(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY4, cid, v)


@wp.func
def cloth_bending_set_rest_angle(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_ANGLE, cid, v)


@wp.func
def cloth_bending_set_alpha(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA, cid, v)


# ---------------------------------------------------------------------------
# Prepare + iterate
# ---------------------------------------------------------------------------


@wp.func
def cloth_bending_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Substep-entry prepare. Flips access mode + caches scaled
    ``inv_mass`` per vertex; resets ``lambda_sum``. Rest angle and
    stiffness are precomputed at init."""
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies
    p_d = body_d - num_bodies

    # Per-cid slot / count cache stamped by
    # :func:`build_constraint_slot_cache` -- see
    # ``soft_tetrahedron_prepare_for_iteration_at``.
    slot_a = constraints.slot_cache[cid, 0]
    slot_b = constraints.slot_cache[cid, 1]
    slot_c = constraints.slot_cache[cid, 2]
    slot_d = constraints.slot_cache[cid, 3]
    inv_factor_a = constraints.count_cache[cid, 0]
    inv_factor_b = constraints.count_cache[cid, 1]
    inv_factor_c = constraints.count_cache[cid, 2]
    inv_factor_d = constraints.count_cache[cid, 3]

    set_access_mode_with_slot(
        bodies, particles, copy_state, body_a, slot_a, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_b, slot_b, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_c, slot_c, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_d, slot_d, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    write_float(constraints, _OFF_INV_MASS_A, cid, particles.inverse_mass[p_a] * wp.float32(inv_factor_a))
    write_float(constraints, _OFF_INV_MASS_B, cid, particles.inverse_mass[p_b] * wp.float32(inv_factor_b))
    write_float(constraints, _OFF_INV_MASS_C, cid, particles.inverse_mass[p_c] * wp.float32(inv_factor_c))
    write_float(constraints, _OFF_INV_MASS_D, cid, particles.inverse_mass[p_d] * wp.float32(inv_factor_d))

    _write_lambda_sum(constraints, cid, wp.float32(0.0))


@wp.func
def cloth_bending_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
):
    """One XPBD dihedral-angle sweep on a cloth bending edge.

    Per-vertex gradient formulas from PhysX
    ``bendingEnergySolvePerTrianglePair``; we keep the variable names
    (``x02, x03, x13, x12, x23``) close to the reference for
    line-by-line auditability.
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)

    slot_a = constraints.slot_cache[cid, 0]
    slot_b = constraints.slot_cache[cid, 1]
    slot_c = constraints.slot_cache[cid, 2]
    slot_d = constraints.slot_cache[cid, 3]

    set_access_mode_with_slot(
        bodies, particles, copy_state, body_a, slot_a, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_b, slot_b, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_c, slot_c, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_d, slot_d, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    rest_angle = read_float(constraints, _OFF_REST_ANGLE, cid)
    alpha = read_float(constraints, _OFF_ALPHA, cid)
    lambda_sum = _read_lambda_sum(constraints, cid)

    inv_mass_a = read_float(constraints, _OFF_INV_MASS_A, cid)
    inv_mass_b = read_float(constraints, _OFF_INV_MASS_B, cid)
    inv_mass_c = read_float(constraints, _OFF_INV_MASS_C, cid)
    inv_mass_d = read_float(constraints, _OFF_INV_MASS_D, cid)

    x0 = read_position_with_slot(bodies, particles, copy_state, body_a, slot_a, num_bodies)
    x1 = read_position_with_slot(bodies, particles, copy_state, body_b, slot_b, num_bodies)
    x2 = read_position_with_slot(bodies, particles, copy_state, body_c, slot_c, num_bodies)
    x3 = read_position_with_slot(bodies, particles, copy_state, body_d, slot_d, num_bodies)

    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    x23 = x3 - x2
    x23_len = wp.length(x23)
    if x23_len < _FEMCLOTH_THRESHOLD:
        return
    x23_len_inv = wp.float32(1.0) / x23_len
    x23_normalized = x23 * x23_len_inv

    scaled_n0 = wp.cross(x02, x03)
    scaled_n1 = wp.cross(x13, x12)
    n0_len = wp.length(scaled_n0)
    n1_len = wp.length(scaled_n1)
    if n0_len < _FEMCLOTH_THRESHOLD or n1_len < _FEMCLOTH_THRESHOLD:
        return
    n0_len_inv = wp.float32(1.0) / n0_len
    n1_len_inv = wp.float32(1.0) / n1_len
    n0 = scaled_n0 * n0_len_inv
    n1 = scaled_n1 * n1_len_inv

    cos_angle = wp.dot(n0, n1)
    sin_angle = wp.dot(wp.cross(n0, n1), x23_normalized)
    angle = wp.atan2(sin_angle, cos_angle)

    c = angle - rest_angle
    # Unwrap to the principal branch then clamp to ±π/2 so the
    # gradient stays well-conditioned even when the hinge has folded
    # past the singular configurations at θ = 0, ±π.
    if wp.abs(c + _TWO_PI) < wp.abs(c):
        c = c + _TWO_PI
    elif wp.abs(c - _TWO_PI) < wp.abs(c):
        c = c - _TWO_PI
    c = wp.clamp(c, -_HALF_PI, _HALF_PI)

    # Bridson-style "normal with inverse length folded in" -- saves a
    # division in the per-vertex gradient. ``temp0 = n0 / n0_len``,
    # ``temp1 = n1 / n1_len`` with ``n0`` / ``n1`` already unit.
    temp0 = n0 * n0_len_inv
    temp1 = n1 * n1_len_inv
    dCdx0 = -x23_len * temp0
    dCdx1 = -x23_len * temp1
    dCdx2 = wp.dot(x03, x23_normalized) * temp0 + wp.dot(x13, x23_normalized) * temp1
    dCdx3 = -(wp.dot(x02, x23_normalized) * temp0 + wp.dot(x12, x23_normalized) * temp1)

    bias = idt * idt * alpha
    grad_sq = (
        inv_mass_a * wp.dot(dCdx0, dCdx0)
        + inv_mass_b * wp.dot(dCdx1, dCdx1)
        + inv_mass_c * wp.dot(dCdx2, dCdx2)
        + inv_mass_d * wp.dot(dCdx3, dCdx3)
    )
    rows = PositionRows1()
    rows.A11 = grad_sq + bias
    rows.residual = c + bias * lambda_sum
    rows.lambda_old = lambda_sum
    rows.diag_floor = wp.float32(0.0)
    update = block_solve_position_rows1(rows, sor_boost)
    d_lam = update.delta
    lambda_sum = update.lambda_new

    x0 = x0 + block_position_delta_1(inv_mass_a, d_lam, dCdx0)
    x1 = x1 + block_position_delta_1(inv_mass_b, d_lam, dCdx1)
    x2 = x2 + block_position_delta_1(inv_mass_c, d_lam, dCdx2)
    x3 = x3 + block_position_delta_1(inv_mass_d, d_lam, dCdx3)

    write_position_unified(bodies, particles, copy_state, body_a, slot_a, num_bodies, x0)
    write_position_unified(bodies, particles, copy_state, body_b, slot_b, num_bodies, x1)
    write_position_unified(bodies, particles, copy_state, body_c, slot_c, num_bodies, x2)
    write_position_unified(bodies, particles, copy_state, body_d, slot_d, num_bodies, x3)

    _write_lambda_sum(constraints, cid, lambda_sum)


# ---------------------------------------------------------------------------
# Init kernel -- populate one bending row from Newton mesh data.
# ---------------------------------------------------------------------------


@wp.kernel
def cloth_bending_init_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    num_bodies: wp.int32,
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angle: wp.array[wp.float32],
    edge_bending_properties: wp.array2d[wp.float32],
    default_alpha_floor: wp.float32,
):
    """Stamp one cloth-bending row from Newton mesh API.

    Newton's ``edge_indices[t]`` is ``(o0, o1, v1, v2)``:

    * ``o0``, ``o1`` = the two opposite vertices (one per adjacent
      triangle). ``-1`` for boundary edges (only one adjacent triangle).
    * ``v1``, ``v2`` = the shared edge vertices.

    Mapping to PhysX's ``(x0, x1, x2, x3)``:

    * ``x0`` = ``o0`` (body1)
    * ``x1`` = ``o1`` (body2)
    * ``x2`` = ``v1`` (body3)
    * ``x3`` = ``v2`` (body4)

    Rest data:

    * ``edge_rest_angle[t]`` is the rest dihedral angle [rad].
    * ``edge_bending_properties[t, 0]`` is the bending stiffness in
      ``N·m/rad``. XPBD compliance ``alpha = 1 / k``.

    Boundary edges (``o0 < 0`` OR ``o1 < 0``) produce a no-op
    constraint by zeroing the stiffness; the iterate's gradient norm
    is unaffected (the geometry is still well-defined for any pair of
    triangles) but the denominator floor keeps it harmless.
    """
    t = wp.tid()
    cid = cid_offset + t

    o0 = edge_indices[t, 0]
    o1 = edge_indices[t, 1]
    v1_i = edge_indices[t, 2]
    v2_i = edge_indices[t, 3]

    cloth_bending_set_type(constraints, cid)

    boundary = (o0 < wp.int32(0)) or (o1 < wp.int32(0))
    if boundary:
        # Boundary edge: stamp a no-op row. Pick valid (non-negative)
        # vertex slots so the constraint container stays well-formed;
        # the alpha set to the floor ceiling makes the row contribute
        # ~zero impulse.
        safe_o0 = o0
        if safe_o0 < wp.int32(0):
            safe_o0 = v1_i
        safe_o1 = o1
        if safe_o1 < wp.int32(0):
            safe_o1 = v1_i
        cloth_bending_set_body1(constraints, cid, num_bodies + safe_o0)
        cloth_bending_set_body2(constraints, cid, num_bodies + safe_o1)
        cloth_bending_set_body3(constraints, cid, num_bodies + v1_i)
        cloth_bending_set_body4(constraints, cid, num_bodies + v2_i)
        cloth_bending_set_rest_angle(constraints, cid, wp.float32(0.0))
        cloth_bending_set_alpha(constraints, cid, wp.float32(1.0) / default_alpha_floor)
        _write_lambda_sum(constraints, cid, wp.float32(0.0))
        return

    cloth_bending_set_body1(constraints, cid, num_bodies + o0)
    cloth_bending_set_body2(constraints, cid, num_bodies + o1)
    cloth_bending_set_body3(constraints, cid, num_bodies + v1_i)
    cloth_bending_set_body4(constraints, cid, num_bodies + v2_i)

    cloth_bending_set_rest_angle(constraints, cid, edge_rest_angle[t])

    stiffness = edge_bending_properties[t, 0]
    if stiffness < default_alpha_floor:
        stiffness = default_alpha_floor
    cloth_bending_set_alpha(constraints, cid, wp.float32(1.0) / stiffness)

    _write_lambda_sum(constraints, cid, wp.float32(0.0))
