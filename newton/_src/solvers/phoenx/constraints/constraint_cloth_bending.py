# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level XPBD cloth bending: Bergou/Wardetzky quadratic
curvature.

Chosen over the dihedral-angle hinge (Gingold 2004 / Jitter2's
``FemTriBendingPBD``) because dihedral has singularities at
``theta=0`` and ``theta=pi`` where the ``atan2`` gradient blows up,
preventing one-step XPBD convergence at high stiffness.

Refs:

* Wardetzky/Bergou/Harmon/Zorin/Grinspun 2007 -- "Discrete Quadratic
  Curvature Energies" (canonical formulation).
* Bergou et al. 2008 -- "Discrete Elastic Rods".
* Bender/Mueller/Macklin 2017 survey, section 5.4 (XPBD integration).

Energy is linear in vertex positions::

    E_bend = (3 / (A_1 + A_2)) * | sum_i K_i * p_i |^2

``K_i`` = cotangent weights (4 per hinge), ``sum_i K_i = 0``. We
split into three scalar XPBD constraints, one per spatial dim::

    C_d(x) = (sum_i K_i * p_i)[d] - h_rest[d]   for d in {x, y, z}

Cotangent weights (Wardetzky 2007 Eq. 12; ``p_0/p_1`` opposite
vertices of the two triangles, ``p_2/p_3`` shared edge)::

    cot_a, cot_b: cotangents at p_2, p_3 in T1 = (p_2, p_3, p_0)
    cot_c, cot_d: cotangents at p_2, p_3 in T2 = (p_2, p_3, p_1)
    K_0 = cot_a + cot_b,  K_1 = cot_c + cot_d
    K_2 = -cot_a - cot_c, K_3 = -cot_b - cot_d

The area factor ``sqrt(3 / (A_1 + A_2))`` is baked into ``K_i`` so
the per-iter math is the bare linear constraint.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_POSITION_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CLOTH_BENDING,
    ConstraintContainer,
    assert_constraint_header,
    read_float,
    read_int,
    read_vec3,
    write_float,
    write_int,
    write_vec3,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.mass_splitting.access import (
    get_state_index,
    read_position_unified,
    set_access_mode_unified,
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
    "cloth_bending_set_k",
    "cloth_bending_set_rest_h",
    "cloth_bending_set_type",
]


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_PHOENX_CLOTH_BENDING_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))
_DEGENERATE_EPS = wp.constant(wp.float32(1.0e-12))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class ClothBendingData:
    """Per-constraint dword layout. Wardetzky 2007 quadratic bending.

    ``body1`` / ``body2`` = the two opposite vertices (one per
    adjacent triangle). ``body3`` / ``body4`` = the shared edge.
    The 4 cotangent weights ``k_*`` are precomputed at rest from
    triangle interior angles (area factor folded in). ``rest_h`` is
    the rest mean-curvature 3-vector ``sum_i K_i * p_rest_i``.
    ``lambda_sum_*`` accumulates per-dimension XPBD impulses.
    """

    constraint_type: wp.int32
    body1: wp.int32  # opposite vertex of triangle T1 (Wardetzky p_0)
    body2: wp.int32  # opposite vertex of triangle T2 (Wardetzky p_1)
    body3: wp.int32  # shared edge vertex (Wardetzky p_2)
    body4: wp.int32  # shared edge vertex (Wardetzky p_3)

    k_0: wp.float32
    k_1: wp.float32
    k_2: wp.float32
    k_3: wp.float32

    rest_h: wp.vec3f  # 3-vector h_rest = sum_i K_i * p_rest_i
    alpha: wp.float32  # XPBD compliance = 1 / bend_stiffness

    lambda_sum: wp.vec3f  # per-dimension lambda accumulators

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32
    inv_mass_d: wp.float32


assert_constraint_header(ClothBendingData)


_OFF_BODY1 = wp.constant(dword_offset_of(ClothBendingData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ClothBendingData, "body2"))
_OFF_BODY3 = wp.constant(dword_offset_of(ClothBendingData, "body3"))
_OFF_BODY4 = wp.constant(dword_offset_of(ClothBendingData, "body4"))
_OFF_K_0 = wp.constant(dword_offset_of(ClothBendingData, "k_0"))
_OFF_K_1 = wp.constant(dword_offset_of(ClothBendingData, "k_1"))
_OFF_K_2 = wp.constant(dword_offset_of(ClothBendingData, "k_2"))
_OFF_K_3 = wp.constant(dword_offset_of(ClothBendingData, "k_3"))
_OFF_REST_H = wp.constant(dword_offset_of(ClothBendingData, "rest_h"))
_OFF_ALPHA = wp.constant(dword_offset_of(ClothBendingData, "alpha"))
_OFF_LAMBDA_SUM = wp.constant(dword_offset_of(ClothBendingData, "lambda_sum"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_c"))
_OFF_INV_MASS_D = wp.constant(dword_offset_of(ClothBendingData, "inv_mass_d"))

CLOTH_BENDING_DWORDS: int = num_dwords(ClothBendingData)


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
def cloth_bending_set_k(
    c: ConstraintContainer, cid: wp.int32, k0: wp.float32, k1: wp.float32, k2: wp.float32, k3: wp.float32
):
    write_float(c, _OFF_K_0, cid, k0)
    write_float(c, _OFF_K_1, cid, k1)
    write_float(c, _OFF_K_2, cid, k2)
    write_float(c, _OFF_K_3, cid, k3)


@wp.func
def cloth_bending_set_rest_h(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_REST_H, cid, v)


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
    ``inv_mass`` per vertex; resets ``lambda_sum``. The cotangent
    weights ``K_i`` and ``rest_h`` are precomputed once at init -- the
    energy is linear so no per-substep recomputation is needed.
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies
    p_d = body_d - num_bodies

    set_access_mode_unified(
        bodies, particles, copy_state, body_a, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_unified(
        bodies, particles, copy_state, body_b, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_unified(
        bodies, particles, copy_state, body_c, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_unified(
        bodies, particles, copy_state, body_d, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    _sa, inv_factor_a = get_state_index(copy_state, body_a, parallel_id)
    _sb, inv_factor_b = get_state_index(copy_state, body_b, parallel_id)
    _sc, inv_factor_c = get_state_index(copy_state, body_c, parallel_id)
    _sd, inv_factor_d = get_state_index(copy_state, body_d, parallel_id)

    write_float(constraints, _OFF_INV_MASS_A, cid, particles.inverse_mass[p_a] * wp.float32(inv_factor_a))
    write_float(constraints, _OFF_INV_MASS_B, cid, particles.inverse_mass[p_b] * wp.float32(inv_factor_b))
    write_float(constraints, _OFF_INV_MASS_C, cid, particles.inverse_mass[p_c] * wp.float32(inv_factor_c))
    write_float(constraints, _OFF_INV_MASS_D, cid, particles.inverse_mass[p_d] * wp.float32(inv_factor_d))

    write_vec3(constraints, _OFF_LAMBDA_SUM, cid, wp.vec3f(0.0, 0.0, 0.0))


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
):
    """One XPBD sweep on a cloth bending edge.

    Three scalar XPBD constraints (one per spatial dimension), each
    linear in vertex positions. Gradient per vertex per dim is just
    ``K_i`` (constant). Denominator is identical across the 3 dims so
    we compute it once and reuse it. Single iteration converges
    exactly at high stiffness.
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)

    set_access_mode_unified(
        bodies, particles, copy_state, body_a, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_unified(
        bodies, particles, copy_state, body_b, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_unified(
        bodies, particles, copy_state, body_c, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_unified(
        bodies, particles, copy_state, body_d, parallel_id, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    k_0 = read_float(constraints, _OFF_K_0, cid)
    k_1 = read_float(constraints, _OFF_K_1, cid)
    k_2 = read_float(constraints, _OFF_K_2, cid)
    k_3 = read_float(constraints, _OFF_K_3, cid)
    rest_h = read_vec3(constraints, _OFF_REST_H, cid)
    alpha = read_float(constraints, _OFF_ALPHA, cid)
    lambda_sum = read_vec3(constraints, _OFF_LAMBDA_SUM, cid)

    inv_mass_a = read_float(constraints, _OFF_INV_MASS_A, cid)
    inv_mass_b = read_float(constraints, _OFF_INV_MASS_B, cid)
    inv_mass_c = read_float(constraints, _OFF_INV_MASS_C, cid)
    inv_mass_d = read_float(constraints, _OFF_INV_MASS_D, cid)

    xA, _ifa, slot_a = read_position_unified(bodies, particles, copy_state, body_a, parallel_id, num_bodies)
    xB, _ifb, slot_b = read_position_unified(bodies, particles, copy_state, body_b, parallel_id, num_bodies)
    xC, _ifc, slot_c = read_position_unified(bodies, particles, copy_state, body_c, parallel_id, num_bodies)
    xD, _ifd, slot_d = read_position_unified(bodies, particles, copy_state, body_d, parallel_id, num_bodies)

    # h(x) = K_0 * xA + K_1 * xB + K_2 * xC + K_3 * xD
    h = k_0 * xA + k_1 * xB + k_2 * xC + k_3 * xD
    c_vec = h - rest_h  # 3D residual

    # Shared denominator: independent of dimension because the gradient
    # is K_i along the d-th axis only. ||grad||^2 = K_i^2.
    bias = idt * idt * alpha
    grad_sq = inv_mass_a * k_0 * k_0 + inv_mass_b * k_1 * k_1 + inv_mass_c * k_2 * k_2 + inv_mass_d * k_3 * k_3
    denom = grad_sq + bias
    if denom <= wp.float32(0.0):
        return

    # Per-dim XPBD lambda update. The 3 dimensions are decoupled, so we
    # compute them in one go via vector math.
    d_lam = -(c_vec + bias * lambda_sum) / denom

    # Position updates: per vertex per dim, delta = K_i * m_inv_i * d_lam_d.
    xA = xA + (k_0 * inv_mass_a) * d_lam
    xB = xB + (k_1 * inv_mass_b) * d_lam
    xC = xC + (k_2 * inv_mass_c) * d_lam
    xD = xD + (k_3 * inv_mass_d) * d_lam

    write_position_unified(bodies, particles, copy_state, body_a, slot_a, num_bodies, xA)
    write_position_unified(bodies, particles, copy_state, body_b, slot_b, num_bodies, xB)
    write_position_unified(bodies, particles, copy_state, body_c, slot_c, num_bodies, xC)
    write_position_unified(bodies, particles, copy_state, body_d, slot_d, num_bodies, xD)

    write_vec3(constraints, _OFF_LAMBDA_SUM, cid, lambda_sum + d_lam)


# ---------------------------------------------------------------------------
# Cotangent helpers (host-callable @wp.func)
# ---------------------------------------------------------------------------


@wp.func
def _cot_at_vertex(v_self: wp.vec3f, v_a: wp.vec3f, v_b: wp.vec3f) -> wp.float32:
    """Cotangent of the angle at ``v_self`` between edges
    (v_self -> v_a) and (v_self -> v_b).

    ``cot(theta) = dot(u, v) / |cross(u, v)|`` where
    ``u = v_a - v_self``, ``v = v_b - v_self``. Numerically safe -- no
    division if the cross product is near zero (degenerate / collinear
    triangle).
    """
    u = v_a - v_self
    v = v_b - v_self
    cross_uv = wp.cross(u, v)
    cross_norm = wp.sqrt(wp.dot(cross_uv, cross_uv))
    if cross_norm < _DEGENERATE_EPS:
        return wp.float32(0.0)
    return wp.dot(u, v) / cross_norm


@wp.func
def _triangle_area(a: wp.vec3f, b: wp.vec3f, c: wp.vec3f) -> wp.float32:
    """Area of triangle (a, b, c) in 3-space. ``0.5 * |cross(b - a, c - a)|``."""
    cross_ab_ac = wp.cross(b - a, c - a)
    return wp.float32(0.5) * wp.sqrt(wp.dot(cross_ab_ac, cross_ab_ac))


# ---------------------------------------------------------------------------
# Init kernel -- populate one bending row from Newton mesh data.
# ---------------------------------------------------------------------------


@wp.kernel
def cloth_bending_init_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    num_bodies: wp.int32,
    edge_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3f],
    edge_bending_properties: wp.array2d[wp.float32],
    default_alpha_floor: wp.float32,
):
    """Stamp one cloth-bending row from Newton mesh API.

    Newton's ``edge_indices[t]`` is ``(o0, o1, v1, v2)``:

    * ``o0``, ``o1`` = the two opposite vertices (one per adjacent triangle).
      ``-1`` for boundary edges (only one adjacent triangle).
    * ``v1``, ``v2`` = the shared edge vertices.

    Wardetzky convention:

    * ``p_0`` (body1) = opposite vertex of triangle T1 = ``o0``
    * ``p_1`` (body2) = opposite vertex of triangle T2 = ``o1``
    * ``p_2`` (body3) = shared edge vertex 1 = ``v1``
    * ``p_3`` (body4) = shared edge vertex 2 = ``v2``

    Cotangent weights (Wardetzky 2007 Eq. 12, our notation):

    * ``cot_a`` = cot at ``v1`` in triangle T1 = (v1, v2, o0)
    * ``cot_b`` = cot at ``v2`` in triangle T1
    * ``cot_c`` = cot at ``v1`` in triangle T2 = (v1, v2, o1)
    * ``cot_d`` = cot at ``v2`` in triangle T2

    * ``K_0 =  cot_a + cot_b`` (multiplies o0)
    * ``K_1 =  cot_c + cot_d`` (multiplies o1)
    * ``K_2 = -cot_a - cot_c`` (multiplies v1)
    * ``K_3 = -cot_b - cot_d`` (multiplies v2)

    Sum ``K_0 + K_1 + K_2 + K_3 = 0`` so translations don't excite the
    constraint. Area factor ``sqrt(3 / (A_1 + A_2))`` is baked into
    ``K`` so the per-iteration math is just the bare linear constraint.

    Boundary edges (``o0 < 0`` OR ``o1 < 0``) produce a no-op
    constraint by zeroing all K_i and rest_h. The iterate's denom
    guard then drops the row cleanly.

    ``edge_bending_properties[t, 0]`` is the bending stiffness in
    ``N*m/rad`` (Newton's convention). XPBD compliance ``alpha =
    1 / k``.
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
        # the iterate's K=0 path will skip the row.
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
        cloth_bending_set_k(constraints, cid, wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        cloth_bending_set_rest_h(constraints, cid, wp.vec3f(0.0, 0.0, 0.0))
        cloth_bending_set_alpha(constraints, cid, wp.float32(1.0) / default_alpha_floor)
        write_vec3(constraints, _OFF_LAMBDA_SUM, cid, wp.vec3f(0.0, 0.0, 0.0))
        return

    cloth_bending_set_body1(constraints, cid, num_bodies + o0)
    cloth_bending_set_body2(constraints, cid, num_bodies + o1)
    cloth_bending_set_body3(constraints, cid, num_bodies + v1_i)
    cloth_bending_set_body4(constraints, cid, num_bodies + v2_i)

    x_o0 = particle_q[o0]
    x_o1 = particle_q[o1]
    x_v1 = particle_q[v1_i]
    x_v2 = particle_q[v2_i]

    # Cotangents at the shared-edge vertices in each triangle.
    # Triangle T1 = (v1, v2, o0); cot_a at v1, cot_b at v2.
    cot_a = _cot_at_vertex(x_v1, x_v2, x_o0)
    cot_b = _cot_at_vertex(x_v2, x_v1, x_o0)
    # Triangle T2 = (v1, v2, o1); cot_c at v1, cot_d at v2.
    cot_c = _cot_at_vertex(x_v1, x_v2, x_o1)
    cot_d = _cot_at_vertex(x_v2, x_v1, x_o1)

    # Area factor from Wardetzky 2007 -- multiply K by sqrt(3/(A1+A2))
    # so the bare scalar constraint matches the quadratic energy.
    a1 = _triangle_area(x_v1, x_v2, x_o0)
    a2 = _triangle_area(x_v1, x_v2, x_o1)
    a_sum = a1 + a2
    if a_sum < _DEGENERATE_EPS:
        # Degenerate (zero-area) hinge -- emit a no-op row.
        cloth_bending_set_k(constraints, cid, wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        cloth_bending_set_rest_h(constraints, cid, wp.vec3f(0.0, 0.0, 0.0))
        cloth_bending_set_alpha(constraints, cid, wp.float32(1.0) / default_alpha_floor)
        write_vec3(constraints, _OFF_LAMBDA_SUM, cid, wp.vec3f(0.0, 0.0, 0.0))
        return
    area_factor = wp.sqrt(wp.float32(3.0) / a_sum)

    k_0 = (cot_a + cot_b) * area_factor
    k_1 = (cot_c + cot_d) * area_factor
    k_2 = (-cot_a - cot_c) * area_factor
    k_3 = (-cot_b - cot_d) * area_factor

    cloth_bending_set_k(constraints, cid, k_0, k_1, k_2, k_3)

    # h_rest = sum_i K_i * x_rest_i.
    rest_h = k_0 * x_o0 + k_1 * x_o1 + k_2 * x_v1 + k_3 * x_v2
    cloth_bending_set_rest_h(constraints, cid, rest_h)

    stiffness = edge_bending_properties[t, 0]
    if stiffness < default_alpha_floor:
        stiffness = default_alpha_floor
    cloth_bending_set_alpha(constraints, cid, wp.float32(1.0) / stiffness)

    write_vec3(constraints, _OFF_LAMBDA_SUM, cid, wp.vec3f(0.0, 0.0, 0.0))
