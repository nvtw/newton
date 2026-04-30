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
