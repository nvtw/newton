# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level XPBD cloth-triangle constraint, Jitter2 ``FemTriPBD`` port.

Direct port of
``experimentalsim/jitterphysics2/src/Jitter2/Dynamics/Constraints/FemTriPBD.cs``
with two simplifications relative to the opts-3 version:

* Operates directly on a :class:`ParticleContainer` rather than the
  unified body-or-particle store -- this constraint is cloth-only here
  (no rigid-attached cloth nodes, no contact interleave).
* No access-mode tag flipping: for cloth-only scenes the iterate is the
  only writer between predict and recover, so the position state is
  unambiguously authoritative throughout the substep.

The math (in-plane projector + 2x2 deformation gradient + iterative
2D rotation extraction + area row + shear row) is line-for-line
identical to ``FemTriPBD.cs``.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_POSITION_LEVEL,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_block import (
    block_position_delta_2d_2,
    block_solve_projected_xpbd_2,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CLOTH_TRIANGLE,
    ConstraintContainer,
    assert_constraint_header,
    read_float,
    read_int,
    read_mat22,
    write_float,
    write_int,
    write_mat22,
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
    "CLOTH_TRIANGLE_DWORDS",
    "ClothTriangleData",
    "cloth_lame_from_youngs_poisson_plane_strain",
    "cloth_lame_from_youngs_poisson_plane_stress",
    "cloth_triangle_iterate_at",
    "cloth_triangle_prepare_for_iteration_at",
    "cloth_triangle_set_alpha_lambda",
    "cloth_triangle_set_alpha_mu",
    "cloth_triangle_set_beta_lambda",
    "cloth_triangle_set_beta_mu",
    "cloth_triangle_set_body1",
    "cloth_triangle_set_body2",
    "cloth_triangle_set_body3",
    "cloth_triangle_set_inv_rest",
    "cloth_triangle_set_rest_area",
    "cloth_triangle_set_type",
]


# ---------------------------------------------------------------------------
# Lame parameter conversion (host-side helpers).
# ---------------------------------------------------------------------------


def cloth_lame_from_youngs_poisson_plane_strain(
    youngs_modulus: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """Plane-strain Lame parameters ``(lambda, mu)`` from ``(E, nu)``::

        lambda = E * nu / ((1 + nu) * (1 - 2 nu))
        mu = E / (2 * (1 + nu))

    Plane strain blows up at ``nu = 0.5``; for thin cloth use
    :func:`cloth_lame_from_youngs_poisson_plane_stress`.
    """
    if youngs_modulus <= 0.0:
        raise ValueError(f"youngs_modulus must be positive (got {youngs_modulus})")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError(f"poisson_ratio must be in (-1, 0.5) (got {poisson_ratio})")
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    return float(lam), float(mu)


def cloth_lame_from_youngs_poisson_plane_stress(
    youngs_modulus: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """Plane-stress Lame parameters ``(lambda, mu)`` from ``(E, nu)``::

        mu = E / (2 * (1 + nu))
        lambda = E * nu / (1 - nu**2)

    Stays finite for ``-1 < nu < 1``.
    """
    if youngs_modulus <= 0.0:
        raise ValueError(f"youngs_modulus must be positive (got {youngs_modulus})")
    if not -1.0 < poisson_ratio < 1.0:
        raise ValueError(f"poisson_ratio must be in (-1, 1) (got {poisson_ratio})")
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / (1.0 - poisson_ratio * poisson_ratio)
    return float(lam), float(mu)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class ClothTriangleData:
    """Per-constraint dword-layout schema for one cloth triangle.

    Mirrors ``FemTriPBD`` field-for-field with PhoenX's mandatory
    three-int32 header (``constraint_type`` / ``body1`` / ``body2`` at
    dwords 0, 1, 2).
    """

    constraint_type: wp.int32
    body1: wp.int32  # particle index of node A
    body2: wp.int32  # particle index of node B
    body3: wp.int32  # particle index of node C

    inv_rest: wp.mat22f
    rest_area: wp.float32

    alpha_lambda: wp.float32
    alpha_mu: wp.float32

    beta_lambda: wp.float32
    beta_mu: wp.float32

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32

    rotation: wp.float32
    lambda_sum_lambda: wp.float32
    lambda_sum_mu: wp.float32

    #: Opt-in per-column wall-clock accumulator (microseconds). See
    #: :func:`constraint_accumulate_time_us`.
    time_us: wp.float32


assert_constraint_header(ClothTriangleData)


_OFF_BODY1 = wp.constant(dword_offset_of(ClothTriangleData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ClothTriangleData, "body2"))
_OFF_BODY3 = wp.constant(dword_offset_of(ClothTriangleData, "body3"))
_OFF_INV_REST = wp.constant(dword_offset_of(ClothTriangleData, "inv_rest"))
_OFF_REST_AREA = wp.constant(dword_offset_of(ClothTriangleData, "rest_area"))
_OFF_ALPHA_LAMBDA = wp.constant(dword_offset_of(ClothTriangleData, "alpha_lambda"))
_OFF_ALPHA_MU = wp.constant(dword_offset_of(ClothTriangleData, "alpha_mu"))
_OFF_BETA_LAMBDA = wp.constant(dword_offset_of(ClothTriangleData, "beta_lambda"))
_OFF_BETA_MU = wp.constant(dword_offset_of(ClothTriangleData, "beta_mu"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(ClothTriangleData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(ClothTriangleData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(ClothTriangleData, "inv_mass_c"))
_OFF_ROTATION = wp.constant(dword_offset_of(ClothTriangleData, "rotation"))
_OFF_LAMBDA_SUM_LAMBDA = wp.constant(dword_offset_of(ClothTriangleData, "lambda_sum_lambda"))
_OFF_LAMBDA_SUM_MU = wp.constant(dword_offset_of(ClothTriangleData, "lambda_sum_mu"))
CLOTH_TRIANGLE_TIME_US_OFFSET = wp.constant(dword_offset_of(ClothTriangleData, "time_us"))

CLOTH_TRIANGLE_DWORDS: int = num_dwords(ClothTriangleData)


@wp.func
def cloth_triangle_set_type(c: ConstraintContainer, cid: wp.int32):
    write_int(c, wp.int32(0), cid, CONSTRAINT_TYPE_CLOTH_TRIANGLE)


@wp.func
def cloth_triangle_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def cloth_triangle_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def cloth_triangle_set_body3(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY3, cid, v)


@wp.func
def cloth_triangle_set_inv_rest(c: ConstraintContainer, cid: wp.int32, v: wp.mat22f):
    write_mat22(c, _OFF_INV_REST, cid, v)


@wp.func
def cloth_triangle_set_rest_area(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_AREA, cid, v)


@wp.func
def cloth_triangle_set_alpha_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_LAMBDA, cid, v)


@wp.func
def cloth_triangle_set_alpha_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_MU, cid, v)


@wp.func
def cloth_triangle_set_beta_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_LAMBDA, cid, v)


@wp.func
def cloth_triangle_set_beta_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_MU, cid, v)


# ---------------------------------------------------------------------------
# In-plane projector and rotation extraction
# ---------------------------------------------------------------------------


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_PROJECTOR_EPS = wp.constant(wp.float32(1.0e-12))
_EXTRACT_ROT_EPS = wp.constant(wp.float32(1.0e-6))
_EXTRACT_ROT_MAX_ITERS = wp.constant(wp.int32(15))
_DET_F_EPS = wp.constant(wp.float32(1.0e-12))
_CLOTH_BLOCK_DET_FLOOR = wp.constant(wp.float32(1.0e-30))


@wp.func
def _project_to_2d(x_axis: wp.vec3f, y_axis: wp.vec3f, dir: wp.vec3f) -> wp.vec2f:
    return wp.vec2f(wp.dot(x_axis, dir), wp.dot(y_axis, dir))


@wp.func
def _project_to_3d(x_axis: wp.vec3f, y_axis: wp.vec3f, dir2: wp.vec2f) -> wp.vec3f:
    return dir2[0] * x_axis + dir2[1] * y_axis


@wp.func
def _build_projector(xa: wp.vec3f, xb: wp.vec3f, xc: wp.vec3f):
    """In-plane (x_axis, y_axis) frame for triangle ABC."""
    ab = xb - xa
    ac = xc - xa
    normal = wp.cross(ab, ac)
    ab_len_sq = wp.dot(ab, ab)
    if ab_len_sq < _PROJECTOR_EPS:
        return wp.vec3f(1.0, 0.0, 0.0), wp.vec3f(0.0, 1.0, 0.0)
    x_axis = ab * (wp.float32(1.0) / wp.sqrt(ab_len_sq))
    y_unnorm = wp.cross(normal, ab)
    y_len_sq = wp.dot(y_unnorm, y_unnorm)
    if y_len_sq < _PROJECTOR_EPS:
        return x_axis, wp.vec3f(-x_axis[1], x_axis[0], wp.float32(0.0))
    y_axis = y_unnorm * (wp.float32(1.0) / wp.sqrt(y_len_sq))
    return x_axis, y_axis


@wp.func
def _extract_rotation_2d(
    f00: wp.float32, f01: wp.float32, f10: wp.float32, f11: wp.float32, angle: wp.float32
) -> wp.float32:
    """Closest 2D rotation angle for deformation gradient ``F``.

    Maximising ``trace(R(theta)^T F)`` gives the closed form
    ``theta = atan2(f10 - f01, f00 + f11)``. Keep the previous angle
    only for the fully degenerate case where the polar rotation is
    undefined.
    """
    y = f10 - f01
    x = f00 + f11
    if x * x + y * y <= _EXTRACT_ROT_EPS * _EXTRACT_ROT_EPS:
        return angle
    return wp.atan2(y, x)


# ---------------------------------------------------------------------------
# Prepare + iterate
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Substep-entry prepare: cache inverse masses; reset XPBD warm starts.

    Body fields are unified indices: ``i_p = body - num_bodies`` is the
    particle slot. The persisted ``rotation`` warm start is intentionally
    NOT reset -- the closest-rotation angle evolves continuously with
    the triangle's pose.

    Direct port of Jitter2's ``FemTriPBD.PrepareForIteration`` pattern:
    every vertex's access mode is set to POSITION_LEVEL via the
    slot-aware unified helper. Mass-splitting routes through the slot;
    rigid-only path falls through to ``particle_set_access_mode``.
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies

    # Per-cid slot / count cache stamped by
    # :func:`build_constraint_slot_cache`. Replaces three
    # ``get_state_index`` walks on the prepare hot path.
    slot_a = constraints.slot_cache[cid, 0]
    slot_b = constraints.slot_cache[cid, 1]
    slot_c = constraints.slot_cache[cid, 2]
    inv_factor_a = constraints.count_cache[cid, 0]
    inv_factor_b = constraints.count_cache[cid, 1]
    inv_factor_c = constraints.count_cache[cid, 2]

    # Flip access mode for each vertex (matches C# FemTriPBD prepare).
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_a, slot_a, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_b, slot_b, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_c, slot_c, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    write_float(constraints, _OFF_INV_MASS_A, cid, particles.inverse_mass[p_a] * wp.float32(inv_factor_a))
    write_float(constraints, _OFF_INV_MASS_B, cid, particles.inverse_mass[p_b] * wp.float32(inv_factor_b))
    write_float(constraints, _OFF_INV_MASS_C, cid, particles.inverse_mass[p_c] * wp.float32(inv_factor_c))

    write_float(constraints, _OFF_LAMBDA_SUM_LAMBDA, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, wp.float32(0.0))


@wp.func
def cloth_triangle_iterate_at(
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
    """One coupled area + corotational strain XPBD sweep on a cloth triangle."""
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies

    # Per-cid slot cache: one independent load per vertex vs. the
    # ``set_access_mode_unified`` + ``read_position_unified`` chains
    # each doing their own ``get_state_index`` walk.
    slot_a = constraints.slot_cache[cid, 0]
    slot_b = constraints.slot_cache[cid, 1]
    slot_c = constraints.slot_cache[cid, 2]

    # Flip each vertex's access mode (slot-aware). C# FemTriPBD calls
    # SetAccessMode on every iterate too; safe to repeat -- a no-op
    # when already POSITION_LEVEL or STATIC.
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_a, slot_a, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_b, slot_b, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_c, slot_c, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    inv_mass_a = read_float(constraints, _OFF_INV_MASS_A, cid)
    inv_mass_b = read_float(constraints, _OFF_INV_MASS_B, cid)
    inv_mass_c = read_float(constraints, _OFF_INV_MASS_C, cid)
    rest_area = read_float(constraints, _OFF_REST_AREA, cid)
    inv_rest = read_mat22(constraints, _OFF_INV_REST, cid)
    alpha_lambda = read_float(constraints, _OFF_ALPHA_LAMBDA, cid)
    alpha_mu = read_float(constraints, _OFF_ALPHA_MU, cid)
    beta_lambda = read_float(constraints, _OFF_BETA_LAMBDA, cid)
    beta_mu = read_float(constraints, _OFF_BETA_MU, cid)
    rotation = read_float(constraints, _OFF_ROTATION, cid)
    lambda_sum_lambda = read_float(constraints, _OFF_LAMBDA_SUM_LAMBDA, cid)
    lambda_sum_mu = read_float(constraints, _OFF_LAMBDA_SUM_MU, cid)

    # Slot-aware position reads. Without mass splitting the helpers
    # fall through to ``particles.position[p_*]``; with mass splitting,
    # they read from the slot matching ``parallel_id`` and the final
    # write goes back to that same slot. Per-iter dilution by ``1/N``
    # at average time is intentional (Tonge mass splitting: each iter
    # contributes ``1/N`` progress, accumulating across solver iters).
    # Matches Jitter2's ``FemTriPBD.Iterate`` single-slot pattern.
    x_a = read_position_with_slot(bodies, particles, copy_state, body_a, slot_a, num_bodies)
    x_b = read_position_with_slot(bodies, particles, copy_state, body_b, slot_b, num_bodies)
    x_c = read_position_with_slot(bodies, particles, copy_state, body_c, slot_c, num_bodies)
    # ``dx = x - position_prev_substep`` is the XPBD damping term anchor
    # (Macklin et al. 2020 "Detailed Rigid Body Simulation with XPBD"):
    # the ``gamma * grad . dx`` term in the lambda numerator is real
    # velocity-projected damping (does not damp at rest, unlike ether
    # damping). ``x`` is the *current* (gauss-seidel-mutable) slot
    # position; ``position_prev_substep`` is the substep-start snapshot
    # captured by ``cloth_predict_kernel`` and never modified during PGS.
    # Reading prev_substep here does NOT break Gauss-Seidel: the
    # iterate state ``x`` is the mutable one; prev_substep is constant
    # within a substep. Jitter2 ``FemTriPBD.Iterate`` omits this term
    # (bare XPBD); the damping is a Newton extension.
    dx_a = x_a - particles.position_prev_substep[p_a]
    dx_b = x_b - particles.position_prev_substep[p_b]
    dx_c = x_c - particles.position_prev_substep[p_c]

    dt = wp.float32(1.0) / idt
    idt_sq = idt * idt
    bias_lambda = idt_sq * alpha_lambda
    bias_mu = idt_sq * alpha_mu
    gamma_lambda = beta_lambda * dt
    gamma_mu = beta_mu * dt

    x_axis, y_axis = _build_projector(x_a, x_b, x_c)
    xab2 = _project_to_2d(x_axis, y_axis, x_b - x_a)
    xac2 = _project_to_2d(x_axis, y_axis, x_c - x_a)
    f00 = inv_rest[0, 0] * xab2[0] + inv_rest[1, 0] * xac2[0]
    f01 = inv_rest[0, 1] * xab2[0] + inv_rest[1, 1] * xac2[0]
    f10 = inv_rest[0, 0] * xab2[1] + inv_rest[1, 0] * xac2[1]
    f11 = inv_rest[0, 1] * xab2[1] + inv_rest[1, 1] * xac2[1]
    rotation = _extract_rotation_2d(f00, f01, f10, f11, rotation)

    q = inv_rest[0, 0] + inv_rest[1, 0]
    t = inv_rest[0, 1] + inv_rest[1, 1]

    g1_a_x = (-q * f11 + t * f10) * rest_area
    g1_a_y = (-t * f00 + q * f01) * rest_area
    g1_b_x = (inv_rest[0, 0] * f11 - f10 * inv_rest[0, 1]) * rest_area
    g1_b_y = (inv_rest[0, 1] * f00 - f01 * inv_rest[0, 0]) * rest_area
    g1_c_x = (inv_rest[1, 0] * f11 - f10 * inv_rest[1, 1]) * rest_area
    g1_c_y = (inv_rest[1, 1] * f00 - f01 * inv_rest[1, 0]) * rest_area
    c_lambda = (f00 * f11 - (wp.float32(1.0) + f10 * f01)) * rest_area

    cr = wp.cos(rotation)
    sr = wp.sin(rotation)
    df = f00 - cr
    dh = f01 - (-sr)
    dm = f10 - sr
    dq = f11 - cr
    du_sq = df * df + dm * dm + dh * dh + dq * dq

    g2_a_x = wp.float32(0.0)
    g2_a_y = wp.float32(0.0)
    g2_b_x = wp.float32(0.0)
    g2_b_y = wp.float32(0.0)
    g2_c_x = wp.float32(0.0)
    g2_c_y = wp.float32(0.0)
    c_mu = wp.float32(0.0)
    if du_sq > _DET_F_EPS:
        du = wp.sqrt(du_sq)
        shear_scale = rest_area / wp.max(_DET_F_EPS, wp.float32(2.0) * du)
        g2_a_x = wp.float32(-2.0) * (df * q + dh * t) * shear_scale
        g2_a_y = wp.float32(-2.0) * (dm * q + dq * t) * shear_scale
        g2_b_x = wp.float32(2.0) * (df * inv_rest[0, 0] + dh * inv_rest[0, 1]) * shear_scale
        g2_b_y = wp.float32(2.0) * (dm * inv_rest[0, 0] + dq * inv_rest[0, 1]) * shear_scale
        g2_c_x = wp.float32(2.0) * (df * inv_rest[1, 0] + dh * inv_rest[1, 1]) * shear_scale
        g2_c_y = wp.float32(2.0) * (dm * inv_rest[1, 0] + dq * inv_rest[1, 1]) * shear_scale
        c_mu = du * rest_area

    dx_a_2d = _project_to_2d(x_axis, y_axis, dx_a)
    dx_b_2d = _project_to_2d(x_axis, y_axis, dx_b)
    dx_c_2d = _project_to_2d(x_axis, y_axis, dx_c)

    grad_lambda_dot_dx = (
        g1_a_x * dx_a_2d[0]
        + g1_a_y * dx_a_2d[1]
        + g1_b_x * dx_b_2d[0]
        + g1_b_y * dx_b_2d[1]
        + g1_c_x * dx_c_2d[0]
        + g1_c_y * dx_c_2d[1]
    )
    grad_mu_dot_dx = (
        g2_a_x * dx_a_2d[0]
        + g2_a_y * dx_a_2d[1]
        + g2_b_x * dx_b_2d[0]
        + g2_b_y * dx_b_2d[1]
        + g2_c_x * dx_c_2d[0]
        + g2_c_y * dx_c_2d[1]
    )

    a_lambda_lambda = (
        inv_mass_a * (g1_a_x * g1_a_x + g1_a_y * g1_a_y)
        + inv_mass_b * (g1_b_x * g1_b_x + g1_b_y * g1_b_y)
        + inv_mass_c * (g1_c_x * g1_c_x + g1_c_y * g1_c_y)
    )
    a_mu_mu = (
        inv_mass_a * (g2_a_x * g2_a_x + g2_a_y * g2_a_y)
        + inv_mass_b * (g2_b_x * g2_b_x + g2_b_y * g2_b_y)
        + inv_mass_c * (g2_c_x * g2_c_x + g2_c_y * g2_c_y)
    )
    a_lambda_mu = (
        inv_mass_a * (g1_a_x * g2_a_x + g1_a_y * g2_a_y)
        + inv_mass_b * (g1_b_x * g2_b_x + g1_b_y * g2_b_y)
        + inv_mass_c * (g1_c_x * g2_c_x + g1_c_y * g2_c_y)
    )

    A11 = (wp.float32(1.0) + gamma_lambda) * a_lambda_lambda + bias_lambda
    A22 = (wp.float32(1.0) + gamma_mu) * a_mu_mu + bias_mu
    A12 = a_lambda_mu

    b_lambda = c_lambda + bias_lambda * lambda_sum_lambda + gamma_lambda * grad_lambda_dot_dx
    b_mu = c_mu + bias_mu * lambda_sum_mu + gamma_mu * grad_mu_dot_dx

    update = block_solve_projected_xpbd_2(
        A11,
        A12,
        A22,
        b_lambda,
        b_mu,
        lambda_sum_lambda,
        lambda_sum_mu,
        sor_boost,
        _CLOTH_BLOCK_DET_FLOOR,
    )
    lambda_sum_lambda = update.lambda_new[0]
    lambda_sum_mu = update.lambda_new[1]

    h1_a_x = (-q * f11 + t * f10) * rest_area
    h1_a_y = (-t * f00 + q * f01) * rest_area
    h1_b_x = (inv_rest[0, 0] * f11 - f10 * inv_rest[0, 1]) * rest_area
    h1_b_y = (inv_rest[0, 1] * f00 - f01 * inv_rest[0, 0]) * rest_area
    h1_c_x = (inv_rest[1, 0] * f11 - f10 * inv_rest[1, 1]) * rest_area
    h1_c_y = (inv_rest[1, 1] * f00 - f01 * inv_rest[1, 0]) * rest_area

    h2_a_x = wp.float32(0.0)
    h2_a_y = wp.float32(0.0)
    h2_b_x = wp.float32(0.0)
    h2_b_y = wp.float32(0.0)
    h2_c_x = wp.float32(0.0)
    h2_c_y = wp.float32(0.0)
    if du_sq > _DET_F_EPS:
        du = wp.sqrt(du_sq)
        shear_scale = rest_area / wp.max(_DET_F_EPS, wp.float32(2.0) * du)
        h2_a_x = wp.float32(-2.0) * (df * q + dh * t) * shear_scale
        h2_a_y = wp.float32(-2.0) * (dm * q + dq * t) * shear_scale
        h2_b_x = wp.float32(2.0) * (df * inv_rest[0, 0] + dh * inv_rest[0, 1]) * shear_scale
        h2_b_y = wp.float32(2.0) * (dm * inv_rest[0, 0] + dq * inv_rest[0, 1]) * shear_scale
        h2_c_x = wp.float32(2.0) * (df * inv_rest[1, 0] + dh * inv_rest[1, 1]) * shear_scale
        h2_c_y = wp.float32(2.0) * (dm * inv_rest[1, 0] + dq * inv_rest[1, 1]) * shear_scale

    delta_a = block_position_delta_2d_2(
        inv_mass_a,
        update.delta,
        wp.vec2f(h1_a_x, h1_a_y),
        wp.vec2f(h2_a_x, h2_a_y),
    )
    delta_b = block_position_delta_2d_2(
        inv_mass_b,
        update.delta,
        wp.vec2f(h1_b_x, h1_b_y),
        wp.vec2f(h2_b_x, h2_b_y),
    )
    delta_c = block_position_delta_2d_2(
        inv_mass_c,
        update.delta,
        wp.vec2f(h1_c_x, h1_c_y),
        wp.vec2f(h2_c_x, h2_c_y),
    )
    x_a = x_a + _project_to_3d(x_axis, y_axis, delta_a)
    x_b = x_b + _project_to_3d(x_axis, y_axis, delta_b)
    x_c = x_c + _project_to_3d(x_axis, y_axis, delta_c)
    write_position_unified(bodies, particles, copy_state, body_a, slot_a, num_bodies, x_a)
    write_position_unified(bodies, particles, copy_state, body_b, slot_b, num_bodies, x_b)
    write_position_unified(bodies, particles, copy_state, body_c, slot_c, num_bodies, x_c)

    write_float(constraints, _OFF_ROTATION, cid, rotation)
    write_float(constraints, _OFF_LAMBDA_SUM_LAMBDA, cid, lambda_sum_lambda)
    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, lambda_sum_mu)
