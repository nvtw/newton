# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-based cloth triangle constraint for :class:`PhoenXWorld`.

One constraint column per FEM triangle, two XPBD rows per column
(area / λ row + shear / μ row) following Aachen 2018 *Fast
Co-rotated FEM using Operator Splitting* energy density. Lives in
the same :class:`~newton._src.solvers.phoenx.constraints.constraint_container.ConstraintContainer`
ADBS joints live in -- the per-cid type tag at dword 0
(:data:`CONSTRAINT_TYPE_CLOTH_TRIANGLE`) routes the dispatcher.

This file is the *data* + *getters/setters* layer (Commit 1 of the
plan in :file:`PLAN_CLOTH_TRIANGLE.md`); the prepare / iterate
``@wp.func``s ship in Commit 2.

## Constraint endpoints

Three unified body-or-particle indices (see
:mod:`newton._src.solvers.phoenx.body_or_particle`). For now
particle-only -- cloth nodes that are points on a rigid body are
mathematically supported by the unified-index scheme but their
iterate-side gradient handling differs (rigid 6-DOF Jacobian at
local anchor vs. particle 3-DOF identity), so they're a separate
code path on the same constraint type and ship later.

The mandatory header convention requires fields ``body1`` and
``body2`` at dwords 1 and 2 (per
:func:`~newton._src.solvers.phoenx.constraints.constraint_container.assert_constraint_header`).
We honour that contract with ``body1`` / ``body2`` carrying the
first two cloth-triangle endpoints; the third endpoint is
``body3`` at dword 3.

## Row layout (~18 dwords; well inside ``ADBS_DWORDS``)

Header
    * ``constraint_type`` -- :data:`CONSTRAINT_TYPE_CLOTH_TRIANGLE`.
    * ``body1`` / ``body2`` / ``body3`` -- unified indices of the
      three cloth nodes (A, B, C).
Rest-shape (precomputed once at scene build)
    * ``inv_rest`` -- 2x2 inverse rest-pose matrix (Newton's
      ``Model.tri_poses[t]``). Maps rest tangent vectors to the
      tangent-frame edge basis.
    * ``rest_area`` -- triangle rest area [m²]
      (``Model.tri_areas[t]``).
XPBD compliance
    * ``alpha_lambda`` -- compliance for the area row [m·s²/N].
    * ``alpha_mu`` -- compliance for the shear row.
Per-substep cache (filled by prepare; read by iterate)
    * ``inv_mass_a/b/c`` -- looked up via
      :func:`~newton._src.solvers.phoenx.body_or_particle.get_inverse_mass`
      and parked in the row so iterate doesn't re-fetch.
    * ``bias_lambda`` / ``bias_mu`` -- ``alphã = alpha · invDt²`` precomputed
      per substep so iterate doesn't recompute the substep
      timestep on every iteration.
Cross-iteration accumulators
    * ``lambda_sum_lambda`` / ``lambda_sum_mu`` -- XPBD warm-start
      accumulators. Reset by prepare at substep entry; mutated by
      iterate; survive substep boundaries via the column row.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body_or_particle import (
    BodyOrParticleStore,
    get_inverse_mass,
    get_position,
    set_position,
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

__all__ = [
    "CLOTH_TRIANGLE_DWORDS",
    "ClothTriangleData",
    "cloth_lame_from_youngs_poisson_plane_stress",
    "cloth_triangle_get_alpha_lambda",
    "cloth_triangle_get_alpha_mu",
    "cloth_triangle_get_bias_lambda",
    "cloth_triangle_get_bias_mu",
    "cloth_triangle_get_body1",
    "cloth_triangle_get_body2",
    "cloth_triangle_get_body3",
    "cloth_triangle_get_inv_mass_a",
    "cloth_triangle_get_inv_mass_b",
    "cloth_triangle_get_inv_mass_c",
    "cloth_triangle_get_inv_rest",
    "cloth_triangle_get_lambda_sum_lambda",
    "cloth_triangle_get_lambda_sum_mu",
    "cloth_triangle_get_rest_area",
    "cloth_triangle_iterate_at",
    "cloth_triangle_prepare_for_iteration_at",
    "cloth_triangle_set_alpha_lambda",
    "cloth_triangle_set_alpha_mu",
    "cloth_triangle_set_bias_lambda",
    "cloth_triangle_set_bias_mu",
    "cloth_triangle_set_body1",
    "cloth_triangle_set_body2",
    "cloth_triangle_set_body3",
    "cloth_triangle_set_inv_mass_a",
    "cloth_triangle_set_inv_mass_b",
    "cloth_triangle_set_inv_mass_c",
    "cloth_triangle_set_inv_rest",
    "cloth_triangle_set_lambda_sum_lambda",
    "cloth_triangle_set_lambda_sum_mu",
    "cloth_triangle_set_rest_area",
    "cloth_triangle_set_type",
]


# ---------------------------------------------------------------------------
# Lamé parameter conversion (host-side helper).
# ---------------------------------------------------------------------------


def cloth_lame_from_youngs_poisson_plane_stress(
    youngs_modulus: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """Convert Young's modulus + Poisson ratio to plane-stress Lamé
    parameters ``(lam, mu)``.

    Cloth is modelled as a thin sheet -- bending out of plane is
    nearly free compared with in-plane stretch / shear, so the
    in-plane elasticity is plane stress, not 3D. The plane-stress
    Lamé conversion is::

        mu = E / (2 * (1 + nu))  # shear modulus
        lam = E * nu / (1 - nu**2)  # 1st Lamé (plane stress)

    Compare with the 3D (plane strain) conversion ``lam = E*nu /
    ((1+nu)*(1-2*nu))`` -- those denominators blow up at nu = 0.5
    (incompressible solid), but plane stress's ``1 - nu**2``
    stays finite for the entire physical range. This is the right
    choice for cloth, where the thickness direction is unconstrained.

    The PhoenX cloth iterate consumes ``mu`` as the shear-row
    stiffness (Newton's ``tri_materials[t, 0]`` /
    :attr:`~newton.ModelBuilder.default_tri_ke`) and ``lam`` as the
    area-preservation row stiffness (``tri_materials[t, 1]`` /
    :attr:`~newton.ModelBuilder.default_tri_ka`). XPBD compliance
    is then ``alpha = 1 / (k * area)`` per row.

    Args:
        youngs_modulus: Young's modulus ``E`` [Pa]. Cotton-weight
            cloth: ~1e7-1e8. Stiff garment / leather: 1e8-1e9.
        poisson_ratio: Poisson ratio ``nu`` [-]. Typical cloth:
            0.3-0.45. Must satisfy ``-1 < nu < 1`` (plane stress
            allows the full range, but values near ``+/-1`` produce
            very stiff area-preservation).

    Returns:
        ``(lam, mu)`` -- the Lamé first parameter (area-row
        stiffness) and shear modulus (shear-row stiffness), both
        in Pa. Pass to ``add_cloth_grid`` /
        ``add_cloth_mesh`` as ``tri_ka=lam, tri_ke=mu``.

    Raises:
        ValueError: if ``youngs_modulus <= 0`` or ``not -1 <
            poisson_ratio < 1``.
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

    Two XPBD rows (area + shear) on a 3-particle FEM triangle, with
    ``inv_rest`` / ``rest_area`` precomputed at scene build and
    ``bias_*`` / ``inv_mass_*`` re-derived per substep in prepare.
    The ``lambda_sum_*`` accumulators warm-start across substeps.

    Field order matches the dword layout in the
    :class:`ConstraintContainer` -- the
    :func:`~newton._src.solvers.phoenx.constraints.constraint_container.dword_offset_of`
    helper reads the order from this declaration.
    """

    # ---- Header (mandatory contract: dwords 0 / 1 / 2) ----------------
    constraint_type: wp.int32
    #: Unified body-or-particle index of cloth node A. For now must
    #: satisfy ``body1 >= num_bodies`` (particle-only); rigid-attached
    #: cloth nodes ship later.
    body1: wp.int32
    #: Unified body-or-particle index of cloth node B.
    body2: wp.int32

    # ---- Triangle endpoint C (the cloth-only third node) --------------
    #: Unified body-or-particle index of cloth node C. Sits next to
    #: ``body2`` so the three particle indices are contiguous in the
    #: row.
    body3: wp.int32

    # ---- Rest-shape (precomputed once at scene build) -----------------
    #: 2x2 inverse rest-pose matrix mapping rest tangent vectors to
    #: the tangent-frame edge basis. ``F = [xB-xA | xC-xA] · inv_rest``
    #: at iterate time. Stored row-major as 4 contiguous floats. Same
    #: convention Newton's :attr:`~newton.Model.tri_poses` (and Style3D /
    #: VBD) use.
    inv_rest: wp.mat22f

    #: Triangle rest area ``[m²]``. Scales the constraint rows so the
    #: stiffness has correct units. Comes from
    #: :attr:`~newton.Model.tri_areas`.
    rest_area: wp.float32

    # ---- XPBD compliance (precomputed once at scene build) ------------
    #: Compliance for the area row ``[m·s²/N]``. Pre-baked from Lamé λ
    #: and rest area at scene build so the iterate hot loop just reads
    #: a scalar.
    alpha_lambda: wp.float32
    #: Compliance for the shear row.
    alpha_mu: wp.float32

    # ---- Per-substep cache (filled by prepare; read by iterate) -------
    #: Inverse mass of node A. Mirrored from
    #: :func:`~newton._src.solvers.phoenx.body_or_particle.get_inverse_mass`
    #: into the row so iterate avoids re-touching the body /
    #: particle store per iteration.
    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32

    #: ``alphã_λ = alpha_lambda · invDt²``, precomputed by prepare.
    #: Substep-dependent so it can't move into the rest-shape block.
    bias_lambda: wp.float32
    #: ``alphã_μ = alpha_mu · invDt²``.
    bias_mu: wp.float32

    # ---- Cross-iteration accumulators (warm-start) --------------------
    #: XPBD ``λ_sum`` for the area row. Reset by prepare at substep
    #: entry, mutated by iterate during the PGS sweeps; the substep
    #: boundary lets the next substep warm-start from a near-converged
    #: state on stiff fabric.
    lambda_sum_lambda: wp.float32
    lambda_sum_mu: wp.float32


assert_constraint_header(ClothTriangleData)


# ---------------------------------------------------------------------------
# Dword offsets -- single source of truth keyed off the struct above
# ---------------------------------------------------------------------------
#
# All offsets are recovered from the schema rather than hand-coded so a
# field reordering can never silently scramble the row layout. Same
# pattern :file:`constraint_actuated_double_ball_socket.py` uses.

_OFF_BODY1 = wp.constant(dword_offset_of(ClothTriangleData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ClothTriangleData, "body2"))
_OFF_BODY3 = wp.constant(dword_offset_of(ClothTriangleData, "body3"))
_OFF_INV_REST = wp.constant(dword_offset_of(ClothTriangleData, "inv_rest"))
_OFF_REST_AREA = wp.constant(dword_offset_of(ClothTriangleData, "rest_area"))
_OFF_ALPHA_LAMBDA = wp.constant(dword_offset_of(ClothTriangleData, "alpha_lambda"))
_OFF_ALPHA_MU = wp.constant(dword_offset_of(ClothTriangleData, "alpha_mu"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(ClothTriangleData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(ClothTriangleData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(ClothTriangleData, "inv_mass_c"))
_OFF_BIAS_LAMBDA = wp.constant(dword_offset_of(ClothTriangleData, "bias_lambda"))
_OFF_BIAS_MU = wp.constant(dword_offset_of(ClothTriangleData, "bias_mu"))
_OFF_LAMBDA_SUM_LAMBDA = wp.constant(dword_offset_of(ClothTriangleData, "lambda_sum_lambda"))
_OFF_LAMBDA_SUM_MU = wp.constant(dword_offset_of(ClothTriangleData, "lambda_sum_mu"))


#: Total dword count of one cloth-triangle constraint column. Useful
#: for sizing assertions; the actual :class:`ConstraintContainer`
#: width is ``max(ADBS_DWORDS, CLOTH_TRIANGLE_DWORDS, ...)``.
CLOTH_TRIANGLE_DWORDS: int = num_dwords(ClothTriangleData)


# ---------------------------------------------------------------------------
# Type-tag setter (the only writer of dword 0)
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_set_type(c: ConstraintContainer, cid: wp.int32):
    """Stamp the constraint-type tag at dword 0.

    Called once per row when the cloth scene-build pipeline
    initialises the row; the iterate / dispatcher reads this tag
    via :func:`~newton._src.solvers.phoenx.constraints.constraint_container.read_int`
    at offset 0 to route to the cloth handler.
    """
    write_int(c, wp.int32(0), cid, CONSTRAINT_TYPE_CLOTH_TRIANGLE)


# ---------------------------------------------------------------------------
# Endpoint accessors -- body1 / body2 / body3 = particle indices A / B / C
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def cloth_triangle_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def cloth_triangle_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def cloth_triangle_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def cloth_triangle_get_body3(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY3, cid)


@wp.func
def cloth_triangle_set_body3(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY3, cid, v)


# ---------------------------------------------------------------------------
# Rest-shape accessors
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_get_inv_rest(c: ConstraintContainer, cid: wp.int32) -> wp.mat22f:
    return read_mat22(c, _OFF_INV_REST, cid)


@wp.func
def cloth_triangle_set_inv_rest(c: ConstraintContainer, cid: wp.int32, v: wp.mat22f):
    write_mat22(c, _OFF_INV_REST, cid, v)


@wp.func
def cloth_triangle_get_rest_area(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_REST_AREA, cid)


@wp.func
def cloth_triangle_set_rest_area(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_AREA, cid, v)


# ---------------------------------------------------------------------------
# Compliance accessors
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_get_alpha_lambda(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_ALPHA_LAMBDA, cid)


@wp.func
def cloth_triangle_set_alpha_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_LAMBDA, cid, v)


@wp.func
def cloth_triangle_get_alpha_mu(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_ALPHA_MU, cid)


@wp.func
def cloth_triangle_set_alpha_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_MU, cid, v)


# ---------------------------------------------------------------------------
# Per-substep cache accessors (inv_mass_*, bias_*)
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_get_inv_mass_a(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_INV_MASS_A, cid)


@wp.func
def cloth_triangle_set_inv_mass_a(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_INV_MASS_A, cid, v)


@wp.func
def cloth_triangle_get_inv_mass_b(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_INV_MASS_B, cid)


@wp.func
def cloth_triangle_set_inv_mass_b(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_INV_MASS_B, cid, v)


@wp.func
def cloth_triangle_get_inv_mass_c(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_INV_MASS_C, cid)


@wp.func
def cloth_triangle_set_inv_mass_c(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_INV_MASS_C, cid, v)


@wp.func
def cloth_triangle_get_bias_lambda(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_LAMBDA, cid)


@wp.func
def cloth_triangle_set_bias_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_LAMBDA, cid, v)


@wp.func
def cloth_triangle_get_bias_mu(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_MU, cid)


@wp.func
def cloth_triangle_set_bias_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_MU, cid, v)


# ---------------------------------------------------------------------------
# Cross-iteration warm-start accumulators
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_get_lambda_sum_lambda(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_LAMBDA_SUM_LAMBDA, cid)


@wp.func
def cloth_triangle_set_lambda_sum_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_LAMBDA_SUM_LAMBDA, cid, v)


@wp.func
def cloth_triangle_get_lambda_sum_mu(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_LAMBDA_SUM_MU, cid)


@wp.func
def cloth_triangle_set_lambda_sum_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_LAMBDA_SUM_MU, cid, v)


# ---------------------------------------------------------------------------
# Prepare + iterate (Aachen 2018 Fast Co-rotated FEM, XPBD form)
# ---------------------------------------------------------------------------
#
# Two XPBD rows per triangle, sequential row Gauss-Seidel:
#
#   Row lambda (area-preservation):
#     C_lam = current_area - rest_area = (sigma_0 * sigma_1 - 1) * rest_area
#
#   Row mu (deviatoric / shear, corotational):
#     C_mu  = ||F - R||_F * rest_area
#         = sqrt((sigma_0 - 1)^2 + (sigma_1 - 1)^2) * rest_area
#
# F is the 3x2 deformation gradient, R = polar(F) the closest 3x2
# matrix with orthonormal columns. sigma_0, sigma_1 are the singular
# values of F (recovered analytically from the 2x2 SPD F^T F).
#
# Per-row XPBD update (Macklin "Small Steps in Physics Simulation"):
#   alpha_tilde = alpha * idt^2
#   delta_lam   = -(C + alpha_tilde * lambda_sum)
#                 / (sum_node inv_mass_node * |grad_node|^2 + alpha_tilde)
#   x_node     += inv_mass_node * delta_lam * grad_node
#   lambda_sum += delta_lam


_C_LAMBDA_EPS = wp.constant(wp.float32(1.0e-12))
_C_MU_EPS = wp.constant(wp.float32(1.0e-6))
_DET_EPS = wp.constant(wp.float32(1.0e-12))


@wp.func
def cloth_triangle_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    store: BodyOrParticleStore,
    idt: wp.float32,
):
    """Prepare a cloth-triangle column for one substep's iterations.

    Caches per-substep quantities in the column row so the iterate
    hot loop just reads scalars:

    * ``inv_mass_a/b/c``  -- looked up from the body-or-particle
      store (constants for a substep -- particles' inverse mass
      doesn't change inside a step).
    * ``bias_lambda``     -- ``alpha_lambda * idt^2`` (XPBD compliance
      scaled by inverse-substep-squared per Macklin 2019).
    * ``bias_mu``         -- ``alpha_mu * idt^2``.

    Resets the warm-start accumulators ``lambda_sum_lambda`` /
    ``lambda_sum_mu`` to zero at substep entry. Cross-substep
    warm-starting works at a coarser level than a single solve --
    starting each substep at zero gives the iterate sweeps a clean
    XPBD trajectory; carrying lambda_sum across substeps would
    couple substep timesteps in ways that void the
    ``alpha_tilde = alpha * idt^2`` derivation.
    """
    # ``base_offset`` reserved for future compound-cid layouts; cloth
    # uses one block per cid so it's always 0.
    _ = base_offset

    body_a = cloth_triangle_get_body1(constraints, cid)
    body_b = cloth_triangle_get_body2(constraints, cid)
    body_c = cloth_triangle_get_body3(constraints, cid)

    inv_mass_a = get_inverse_mass(store, body_a)
    inv_mass_b = get_inverse_mass(store, body_b)
    inv_mass_c = get_inverse_mass(store, body_c)
    cloth_triangle_set_inv_mass_a(constraints, cid, inv_mass_a)
    cloth_triangle_set_inv_mass_b(constraints, cid, inv_mass_b)
    cloth_triangle_set_inv_mass_c(constraints, cid, inv_mass_c)

    alpha_lambda = cloth_triangle_get_alpha_lambda(constraints, cid)
    alpha_mu = cloth_triangle_get_alpha_mu(constraints, cid)
    idt_sq = idt * idt
    cloth_triangle_set_bias_lambda(constraints, cid, alpha_lambda * idt_sq)
    cloth_triangle_set_bias_mu(constraints, cid, alpha_mu * idt_sq)

    # Substep-fresh XPBD accumulators.
    cloth_triangle_set_lambda_sum_lambda(constraints, cid, wp.float32(0.0))
    cloth_triangle_set_lambda_sum_mu(constraints, cid, wp.float32(0.0))


@wp.func
def cloth_triangle_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    store: BodyOrParticleStore,
):
    """One XPBD sweep over a cloth-triangle column.

    Reads the three particle positions, runs the area row first
    (sequential row Gauss-Seidel: row mu sees row lambda's
    update), runs the shear row, writes back. Direct particle-
    position writes -- no mass-splitting indirection (deferred per
    PLAN_CLOTH_TRIANGLE.md).

    Numerical guards:

    * ``current_area < eps`` (degenerate triangle, or initial
      collapsed state): the lambda row early-outs because its
      gradient direction is ill-defined.
    * ``f_norm < eps`` (no shear deformation): the mu row early-
      outs because the gradient division by ``f_norm`` is
      ill-conditioned. The constraint is satisfied at this
      configuration so skipping is correct.
    * ``det_M <= eps`` (collapsed deformation gradient -- triangle
      mapped to a line): the polar decomposition diverges. Skip the
      mu row; the lambda row will push the triangle area back.
    """
    _ = base_offset

    body_a = cloth_triangle_get_body1(constraints, cid)
    body_b = cloth_triangle_get_body2(constraints, cid)
    body_c = cloth_triangle_get_body3(constraints, cid)

    inv_mass_a = cloth_triangle_get_inv_mass_a(constraints, cid)
    inv_mass_b = cloth_triangle_get_inv_mass_b(constraints, cid)
    inv_mass_c = cloth_triangle_get_inv_mass_c(constraints, cid)
    rest_area = cloth_triangle_get_rest_area(constraints, cid)
    inv_rest = cloth_triangle_get_inv_rest(constraints, cid)
    bias_lambda = cloth_triangle_get_bias_lambda(constraints, cid)
    bias_mu = cloth_triangle_get_bias_mu(constraints, cid)

    x_a = get_position(store, body_a)
    x_b = get_position(store, body_b)
    x_c = get_position(store, body_c)

    # ----- Row lambda: area preservation ----------------------------------
    # C_lam = 0.5 * |cross(xB-xA, xC-xA)| - rest_area.
    e1 = x_b - x_a
    e2 = x_c - x_a
    n = wp.cross(e1, e2)
    n_sq = wp.dot(n, n)
    if n_sq > _C_LAMBDA_EPS:
        n_norm = wp.sqrt(n_sq)
        n_hat = n / n_norm
        current_area = wp.float32(0.5) * n_norm
        c_lambda = current_area - rest_area
        # Translation-invariant gradients (sum to zero):
        #   grad_A = 0.5 * cross(xB - xC, n_hat)
        #   grad_B = 0.5 * cross(xC - xA, n_hat)
        #   grad_C = 0.5 * cross(xA - xB, n_hat)
        grad_a = wp.float32(0.5) * wp.cross(x_b - x_c, n_hat)
        grad_b = wp.float32(0.5) * wp.cross(x_c - x_a, n_hat)
        grad_c = wp.float32(0.5) * wp.cross(x_a - x_b, n_hat)
        denom_lam = (
            inv_mass_a * wp.dot(grad_a, grad_a)
            + inv_mass_b * wp.dot(grad_b, grad_b)
            + inv_mass_c * wp.dot(grad_c, grad_c)
            + bias_lambda
        )
        if denom_lam > wp.float32(0.0):
            lambda_sum_lambda = cloth_triangle_get_lambda_sum_lambda(constraints, cid)
            delta_lambda = -(c_lambda + bias_lambda * lambda_sum_lambda) / denom_lam
            x_a = x_a + (inv_mass_a * delta_lambda) * grad_a
            x_b = x_b + (inv_mass_b * delta_lambda) * grad_b
            x_c = x_c + (inv_mass_c * delta_lambda) * grad_c
            cloth_triangle_set_lambda_sum_lambda(constraints, cid, lambda_sum_lambda + delta_lambda)

    # ----- Row mu: shear / deviatoric (corotational) ----------------------
    # F is 3x2: F[:,j] = invRest[0,j]*e1 + invRest[1,j]*e2.
    e1 = x_b - x_a
    e2 = x_c - x_a
    f_col0 = inv_rest[0, 0] * e1 + inv_rest[1, 0] * e2
    f_col1 = inv_rest[0, 1] * e1 + inv_rest[1, 1] * e2
    # M = F^T F (2x2, SPD). Eigen-decomposition has a closed form:
    #   sigma_prod^2 = det(M),  sigma_sum^2 = trace(M) + 2*sigma_prod
    m00 = wp.dot(f_col0, f_col0)
    m01 = wp.dot(f_col0, f_col1)
    m11 = wp.dot(f_col1, f_col1)
    tr_m = m00 + m11
    det_m = m00 * m11 - m01 * m01
    if det_m > _DET_EPS:
        sigma_prod = wp.sqrt(det_m)
        sigma_sum_sq = tr_m + wp.float32(2.0) * sigma_prod
        if sigma_sum_sq > wp.float32(0.0):
            sigma_sum = wp.sqrt(sigma_sum_sq)
            # ||F-R||_F^2 = trace(M) - 2*(sigma_0+sigma_1) + 2.
            f_norm_sq = tr_m - wp.float32(2.0) * sigma_sum + wp.float32(2.0)
            if f_norm_sq > _C_MU_EPS:
                f_norm = wp.sqrt(f_norm_sq)
                c_mu = f_norm * rest_area
                # Recover R = F * S^-1 from the closed-form S = sqrt(M):
                #   S = (M + sigma_prod * I) / sigma_sum
                #   det(S) = sigma_prod  =>  S^-1 = adj(S) / det(S)
                # i.e. T = S^-1, with:
                #   inv_factor = 1 / (sigma_sum * sigma_prod)
                #   T00 = (m11 + sigma_prod) * inv_factor
                #   T01 = -m01 * inv_factor
                #   T10 = T01
                #   T11 = (m00 + sigma_prod) * inv_factor
                inv_factor = wp.float32(1.0) / (sigma_sum * sigma_prod)
                t00 = (m11 + sigma_prod) * inv_factor
                t01 = -m01 * inv_factor
                t11 = (m00 + sigma_prod) * inv_factor
                # R = F * T (3x2 = 3x2 * 2x2)
                r_col0 = f_col0 * t00 + f_col1 * t01
                r_col1 = f_col0 * t01 + f_col1 * t11
                # G = F - R, then H = G * invRest^T (3x2 = 3x2 * 2x2).
                g_col0 = f_col0 - r_col0
                g_col1 = f_col1 - r_col1
                h_col0 = inv_rest[0, 0] * g_col0 + inv_rest[0, 1] * g_col1
                h_col1 = inv_rest[1, 0] * g_col0 + inv_rest[1, 1] * g_col1
                # Gradients:
                #   grad_A = -(rest_area / f_norm) * (H[:,0] + H[:,1])
                #   grad_B =  (rest_area / f_norm) * H[:,0]
                #   grad_C =  (rest_area / f_norm) * H[:,1]
                inv_fnorm_area = rest_area / f_norm
                grad_a_mu = -inv_fnorm_area * (h_col0 + h_col1)
                grad_b_mu = inv_fnorm_area * h_col0
                grad_c_mu = inv_fnorm_area * h_col1
                denom_mu = (
                    inv_mass_a * wp.dot(grad_a_mu, grad_a_mu)
                    + inv_mass_b * wp.dot(grad_b_mu, grad_b_mu)
                    + inv_mass_c * wp.dot(grad_c_mu, grad_c_mu)
                    + bias_mu
                )
                if denom_mu > wp.float32(0.0):
                    lambda_sum_mu = cloth_triangle_get_lambda_sum_mu(constraints, cid)
                    delta_mu = -(c_mu + bias_mu * lambda_sum_mu) / denom_mu
                    x_a = x_a + (inv_mass_a * delta_mu) * grad_a_mu
                    x_b = x_b + (inv_mass_b * delta_mu) * grad_b_mu
                    x_c = x_c + (inv_mass_c * delta_mu) * grad_c_mu
                    cloth_triangle_set_lambda_sum_mu(constraints, cid, lambda_sum_mu + delta_mu)

    # Scatter the (possibly mutated) particle positions back into the
    # body-or-particle store. Direct write -- no mass splitting on
    # cloth yet.
    set_position(store, body_a, x_a)
    set_position(store, body_b, x_b)
    set_position(store, body_c, x_c)
