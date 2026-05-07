# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level XPBD cloth-triangle constraint, Jitter2 ``FemTriPBD`` port.

Direct port of
``experimentalsim/jitterphysics2/src/Jitter2/Dynamics/Constraints/FemTriPBD.cs``
adapted to PhoenX's particle-or-body unified-index storage.  The two
deviations from Jitter2 are addressing-only:

* PhoenX's cloth nodes are *particles* (3-DoF point masses) rather than
  zero-inertia rigid bodies.  Particles are accessed through the
  :mod:`~newton._src.solvers.phoenx.body_or_particle` unified index so
  this constraint can address either.
* PhoenX uses a column-major dword-packed
  :class:`~newton._src.solvers.phoenx.constraints.constraint_container.ConstraintContainer`
  rather than C# struct layout; the field ordering and accessors below
  mirror the C# struct field-for-field but read/write through
  :func:`read_*` / :func:`write_*` helpers.

The math is otherwise identical to
:cite:`Macklin2016 XPBD <https://mmacklin.com/xpbd.pdf>` /
``FemTriPBD.cs`` line-for-line:

#. Substep entry: :func:`cloth_triangle_prepare_for_iteration_at`
   caches per-particle inverse masses and resets the warm-start
   accumulators ``lambda_sum_X`` / ``lambda_sum_Y``.
#. Each iterate sweep (one per colour, ``solver_iterations`` sweeps per
   substep) runs :func:`cloth_triangle_iterate_at`:

   * Build the in-plane projector :class:`FemTriProjector` from the
     three current particle positions; project edges ``B - A`` and
     ``C - A`` to 2D.
   * Compute the 2x2 deformation gradient ``F = invRest * [xAB | xAC]``.
   * Iterative angle extraction (warm-started across iterations via the
     persisted ``rotation`` scalar) recovers the closest 2x2 rotation
     ``R`` to ``F``.
   * **Area row.** ``C_lambda = (det F - 1) * rest_area``.  XPBD update
     in 2D, scattered back to 3D via the projector.
   * Rebuild projector + ``F`` + extract rotation again so the shear
     row sees the area-corrected positions.
   * **Shear row.** ``C_mu = ||F - R||_F * rest_area``.  XPBD update
     in 2D, scattered back to 3D.
   * Accumulate ``lambda_sum_X`` / ``lambda_sum_Y`` for the next sweep.

The relax wrapper :func:`cloth_triangle_relax_at` runs the same iterate
and then forces ``VELOCITY_LEVEL`` on the three vertices so the
substep's relax pass leaves vertices in a velocity-authoritative state
the contact relax can read consistently.

Compliance is set per-element from Newton's
``Model.tri_materials[t]``: ``alpha_X = 1 / lambda`` for the area row,
``alpha_Y = 1 / mu`` for the shear row -- exact match to
``FemTriPBD.cs`` line 60-61.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_POSITION_LEVEL,
    ACCESS_MODE_VELOCITY_LEVEL,
)
from newton._src.solvers.phoenx.body_or_particle import (
    BodyOrParticleStore,
    get_inverse_mass,
    get_position,
    get_position_prev_substep,
    set_access_mode,
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
    "cloth_lame_from_youngs_poisson_plane_strain",
    "cloth_lame_from_youngs_poisson_plane_stress",
    "cloth_triangle_get_alpha_lambda",
    "cloth_triangle_get_alpha_mu",
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
    "cloth_triangle_get_rotation",
    "cloth_triangle_iterate_at",
    "cloth_triangle_prepare_for_iteration_at",
    "cloth_triangle_relax_at",
    "cloth_triangle_set_alpha_lambda",
    "cloth_triangle_set_alpha_mu",
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
    "cloth_triangle_set_rotation",
    "cloth_triangle_set_type",
]


# ---------------------------------------------------------------------------
# Lame parameter conversion (host-side helpers).
# ---------------------------------------------------------------------------


def cloth_lame_from_youngs_poisson_plane_strain(
    youngs_modulus: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """Plane-strain Lame parameters ``(lambda, mu)`` from ``(E, nu)``.

    Direct port of ``ConstraintHelper.CalculateLameParameters``
    (experimentalsim ``ConstraintHelper.cs:8-15``)::

        lambda = E * nu / ((1 + nu) * (1 - 2 nu))
        mu = E / (2 * (1 + nu))

    Plane strain is what Jitter2's reference cloth uses.  For thin
    cloth, plane stress (:func:`cloth_lame_from_youngs_poisson_plane_stress`)
    is technically more correct, but the difference between the two is
    a constant ``(1 - nu) / (1 - nu**2)`` factor on lambda that cloth
    tuning can easily absorb.  We expose both so users can match the
    reference they're calibrating against.

    Args:
        youngs_modulus: Young's modulus ``E`` ``[Pa]``.
        poisson_ratio: Poisson ratio ``nu`` ``[-]``. Must satisfy
            ``-1 < nu < 0.5``; values approaching ``0.5`` make
            ``lambda`` blow up (incompressible limit).

    Returns:
        ``(lambda, mu)`` ``[Pa]``. Pass to
        ``add_cloth_grid`` as ``tri_ka=lambda, tri_ke=mu``.
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
    """Plane-stress Lame parameters ``(lambda, mu)`` from ``(E, nu)``.

    Thin-shell formulation::

        mu = E / (2 * (1 + nu))
        lambda = E * nu / (1 - nu**2)

    The plane-stress denominator ``1 - nu**2`` stays finite for the
    full ``-1 < nu < 1`` range, unlike plane strain which blows up at
    ``nu = 0.5``.

    Args:
        youngs_modulus: Young's modulus ``E`` ``[Pa]``.
        poisson_ratio: Poisson ratio ``nu`` ``[-]``. Must satisfy
            ``-1 < nu < 1``.

    Returns:
        ``(lambda, mu)`` ``[Pa]``.
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

    Field order is the dword layout the
    :class:`~newton._src.solvers.phoenx.constraints.constraint_container.ConstraintContainer`
    sees -- :func:`~newton._src.solvers.phoenx.helpers.data_packing.dword_offset_of`
    reads it from this declaration.

    Mirrors ``FemTriPBD`` (experimentalsim ``FemTriPBD.cs:11-31``)
    field for field, with PhoenX's mandatory three-field header
    (``constraint_type`` / ``body1`` / ``body2`` at dwords 0, 1, 2).
    """

    # ---- Header (mandatory contract: dwords 0 / 1 / 2) ----------------
    constraint_type: wp.int32
    #: Unified body-or-particle index of cloth node A. Particle-only
    #: for now (must satisfy ``body1 >= num_bodies``).
    body1: wp.int32
    #: Unified body-or-particle index of cloth node B.
    body2: wp.int32

    # ---- Triangle endpoint C ------------------------------------------
    #: Unified body-or-particle index of cloth node C.
    body3: wp.int32

    # ---- Rest-shape (precomputed once at scene build) -----------------
    #: 2x2 inverse rest-pose matrix in the triangle's in-plane 2D
    #: frame.  Built from
    #: ``inv([xAB_2D | xAC_2D])`` at scene build via the
    #: :func:`FemTriProjector` projection of the rest configuration.
    #: Mirrors ``FemTriPBD.invRest`` (``FemTriPBD.cs:20``).
    inv_rest: wp.mat22f

    #: Triangle rest area ``[m^2]``.  Mirrors ``FemTriPBD.restArea``
    #: (``FemTriPBD.cs:19``).
    rest_area: wp.float32

    # ---- XPBD compliance (precomputed once at scene build) ------------
    #: ``alpha_X = 1 / lambda`` ``[m^2 s^2 / kg]`` -- area row
    #: compliance.  Match to ``FemTriPBD.cs:60``.
    alpha_lambda: wp.float32
    #: ``alpha_Y = 1 / mu`` ``[m^2 s^2 / kg]`` -- shear row compliance.
    #: Match to ``FemTriPBD.cs:61``.
    alpha_mu: wp.float32

    # ---- XPBD Rayleigh damping coefficients ---------------------------
    #: Dimensionless damping factor for the area row.  Per Macklin XPBD
    #: 2016 eq. 26, damping enters the iterate as
    #: ``gamma_tilde = beta * dt`` and only acts along the row's
    #: gradient -- so a particle moving freely (zero gradient
    #: contribution) sees no damping, but high-frequency oscillation
    #: against the area constraint is dissipated.  Free fall is
    #: unaffected.  Range ``[0, 1]``; cloth typically uses ``0.05`` to
    #: ``0.3``.  Default is ``0.0`` (no damping); the populate kernel
    #: stamps a sensible cloth default.
    beta_lambda: wp.float32
    #: Dimensionless damping factor for the shear row.
    beta_mu: wp.float32

    # ---- Per-substep cache (filled by prepare; read by iterate) -------
    #: Inverse mass of node A.  Mirrored from the body-or-particle
    #: store at substep entry so iterate avoids re-touching the store
    #: per iteration.
    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32

    # ---- Cross-iteration warm-start ----------------------------------
    #: Persisted angle ``[rad]`` of the closest in-plane rotation to
    #: ``F`` -- iterative ``ExtractRotation`` warm-starts from this
    #: across iterations and substeps, dramatically reducing the
    #: per-call iteration count.  Mirrors ``FemTriPBD.rotation``
    #: (``FemTriPBD.cs:21``).
    rotation: wp.float32

    #: XPBD ``lambda_sum`` for the area row.  Reset by prepare; mutated
    #: by iterate; persists across iterations within a substep.
    #: Mirrors ``FemTriPBD.lambdaSum_X`` (``FemTriPBD.cs:26``).
    lambda_sum_lambda: wp.float32
    #: XPBD ``lambda_sum`` for the shear row.  Mirrors
    #: ``FemTriPBD.lambdaSum_Y`` (``FemTriPBD.cs:27``).
    lambda_sum_mu: wp.float32


assert_constraint_header(ClothTriangleData)


# ---------------------------------------------------------------------------
# Dword offsets -- single source of truth keyed off the struct above
# ---------------------------------------------------------------------------

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


#: Total dword count of one cloth-triangle constraint column.
CLOTH_TRIANGLE_DWORDS: int = num_dwords(ClothTriangleData)


# ---------------------------------------------------------------------------
# Type-tag setter
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_set_type(c: ConstraintContainer, cid: wp.int32):
    """Stamp the constraint-type tag at dword 0."""
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


@wp.func
def cloth_triangle_get_beta_lambda(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BETA_LAMBDA, cid)


@wp.func
def cloth_triangle_set_beta_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_LAMBDA, cid, v)


@wp.func
def cloth_triangle_get_beta_mu(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BETA_MU, cid)


@wp.func
def cloth_triangle_set_beta_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_MU, cid, v)


# ---------------------------------------------------------------------------
# Per-substep cache accessors
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


# ---------------------------------------------------------------------------
# Warm-start accessors
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_get_rotation(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_ROTATION, cid)


@wp.func
def cloth_triangle_set_rotation(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ROTATION, cid, v)


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
# In-plane projector and rotation extraction
# ---------------------------------------------------------------------------


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))
_PROJECTOR_EPS = wp.constant(wp.float32(1.0e-12))
_EXTRACT_ROT_EPS = wp.constant(wp.float32(1.0e-6))
_EXTRACT_ROT_MAX_ITERS = wp.constant(wp.int32(15))
_DET_F_EPS = wp.constant(wp.float32(1.0e-12))


@wp.func
def _project_to_2d(x_axis: wp.vec3f, y_axis: wp.vec3f, dir: wp.vec3f) -> wp.vec2f:
    """Project a 3D direction into the (x_axis, y_axis) 2D frame.

    Mirrors :class:`FemTriProjector.ProjectDirectionTo2D`
    (``FemTri.cs:25-28``)."""
    return wp.vec2f(wp.dot(x_axis, dir), wp.dot(y_axis, dir))


@wp.func
def _project_to_3d(x_axis: wp.vec3f, y_axis: wp.vec3f, dir2: wp.vec2f) -> wp.vec3f:
    """Lift a 2D direction back to 3D in the projector's frame.

    Mirrors :class:`FemTriProjector.ProjectDirectionTo3D`
    (``FemTri.cs:30-33``)."""
    return dir2[0] * x_axis + dir2[1] * y_axis


@wp.func
def _build_projector(xa: wp.vec3f, xb: wp.vec3f, xc: wp.vec3f):
    """Build the in-plane (x_axis, y_axis) frame for a triangle.

    ``x_axis`` runs along ``B - A``; ``y_axis`` is in-plane orthogonal
    to ``x_axis`` (in the ``A B C`` plane).  Mirrors
    :class:`FemTriProjector` (``FemTri.cs:11-23``).

    Returns a degenerate ``(x, y)`` if the triangle is collinear / has
    zero edge length; the iterate guards against that with a det-F
    floor before scattering.
    """
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
        # Collinear -- pick an arbitrary in-plane perp.  The constraint
        # gradient will be near-zero in this state anyway; we just need
        # a non-zero finite frame so the math doesn't divide by zero.
        return x_axis, wp.vec3f(-x_axis[1], x_axis[0], wp.float32(0.0))
    y_axis = y_unnorm * (wp.float32(1.0) / wp.sqrt(y_len_sq))
    return x_axis, y_axis


@wp.func
def _extract_rotation_2d(
    f00: wp.float32, f01: wp.float32, f10: wp.float32, f11: wp.float32, angle: wp.float32
) -> wp.float32:
    """Iterative closest-rotation extraction in 2D.

    Direct port of ``ConstraintHelper.ExtractRotation(JMatrix2, ...)``
    (``ConstraintHelper.cs:28-45``).  Iteratively refines ``angle`` so
    that ``R(angle)`` is the closest 2D rotation to ``F``.

    The 2D version of Mueller et al.'s polar decomposition by
    quaternion-axis iteration.  Each step computes the gradient of
    ``||F - R||^2`` along ``angle`` and damps it by the symmetric
    diagonal sum of ``R^T F``::

        crossSum = R[:,0] x F[:,0] + R[:,1] x F[:,1]
        dotSum   = R[:,0] . F[:,0] + R[:,1] . F[:,1]
        delta    = crossSum / (|dotSum| + eps)
        angle   += delta

    Where 2D cross is the scalar ``ax*by - ay*bx``.  Converges
    quadratically near the optimum; warm-starting via the persisted
    ``rotation`` field keeps the iteration count near 1 in steady
    state.
    """
    a = angle
    for _ in range(_EXTRACT_ROT_MAX_ITERS):
        c = wp.cos(a)
        s = wp.sin(a)
        # R columns: col0 = (c, s), col1 = (-s, c).
        cross_sum = (c * f10 - s * f00) + ((-s) * f11 - c * f01)
        dot_sum = (c * f00 + s * f10) + ((-s) * f01 + c * f11)
        delta = cross_sum / (wp.abs(dot_sum) + _EXTRACT_ROT_EPS)
        a = a + delta
        if delta < _EXTRACT_ROT_EPS and -delta < _EXTRACT_ROT_EPS:
            break
    return a


# ---------------------------------------------------------------------------
# Prepare + iterate
# ---------------------------------------------------------------------------


@wp.func
def cloth_triangle_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    store: BodyOrParticleStore,
    idt: wp.float32,
):
    """Substep-entry prepare: cache inverse masses and reset XPBD warm
    starts.

    The ``rotation`` warm start is *not* reset here -- the angle of
    the closest rotation to the current ``F`` is a function of the
    triangle's *pose*, which evolves continuously across substeps.
    Resetting it to zero would force the iterative extractor to
    re-converge from a cold start every substep, costing
    ``ExtractRotation`` its iteration budget on a problem the
    persistent state already solves for free.
    """
    _ = base_offset

    body_a = cloth_triangle_get_body1(constraints, cid)
    body_b = cloth_triangle_get_body2(constraints, cid)
    body_c = cloth_triangle_get_body3(constraints, cid)

    cloth_triangle_set_inv_mass_a(constraints, cid, get_inverse_mass(store, body_a))
    cloth_triangle_set_inv_mass_b(constraints, cid, get_inverse_mass(store, body_b))
    cloth_triangle_set_inv_mass_c(constraints, cid, get_inverse_mass(store, body_c))

    # Substep-fresh XPBD accumulators -- per Macklin XPBD (Macklin 2019).
    cloth_triangle_set_lambda_sum_lambda(constraints, cid, wp.float32(0.0))
    cloth_triangle_set_lambda_sum_mu(constraints, cid, wp.float32(0.0))


@wp.func
def cloth_triangle_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    store: BodyOrParticleStore,
    idt: wp.float32,
):
    """One XPBD sweep on a cloth triangle (area + shear rows).

    Direct port of ``FemTriPBD.Iterate`` (``FemTriPBD.cs:99-236``):

    #. Sync the three vertices to ``POSITION_LEVEL`` so the read of
       ``position`` reflects any prior contact-side velocity write.
    #. Build the in-plane projector from the *current* positions.
    #. Compute the 2x2 deformation gradient
       ``F = invRest * [xAB_2D | xAC_2D]``.
    #. Iterative ``ExtractRotation`` warm-started from
       ``constraints[cid].rotation``.
    #. **Area row (row lambda).**  ``C = (det F - 1) * rest_area``,
       gradients in 2D (six scalars), XPBD update, scatter back to 3D.
    #. **Shear row (row mu).**  Rebuild projector + ``F`` + extract
       rotation again so the shear row sees the area-corrected state;
       ``C = ||F - R||_F * rest_area``, gradients, XPBD update.
    #. Persist the updated ``rotation``, ``lambda_sum_lambda``,
       ``lambda_sum_mu`` for the next sweep.

    The 2D-projected math is more numerically robust than a 3x2
    deformation gradient in 3D: the projector strips out the
    out-of-plane component, so a flat triangle with a small
    out-of-plane perturbation gets a clean 2x2 ``F`` rather than a
    near-singular 3x2 one.
    """
    _ = base_offset

    body_a = cloth_triangle_get_body1(constraints, cid)
    body_b = cloth_triangle_get_body2(constraints, cid)
    body_c = cloth_triangle_get_body3(constraints, cid)

    set_access_mode(store, body_a, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode(store, body_b, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode(store, body_c, _ACCESS_MODE_POSITION_LEVEL, idt)

    inv_mass_a = cloth_triangle_get_inv_mass_a(constraints, cid)
    inv_mass_b = cloth_triangle_get_inv_mass_b(constraints, cid)
    inv_mass_c = cloth_triangle_get_inv_mass_c(constraints, cid)
    rest_area = cloth_triangle_get_rest_area(constraints, cid)
    inv_rest = cloth_triangle_get_inv_rest(constraints, cid)
    alpha_lambda = cloth_triangle_get_alpha_lambda(constraints, cid)
    alpha_mu = cloth_triangle_get_alpha_mu(constraints, cid)
    beta_lambda = cloth_triangle_get_beta_lambda(constraints, cid)
    beta_mu = cloth_triangle_get_beta_mu(constraints, cid)
    rotation = cloth_triangle_get_rotation(constraints, cid)
    lambda_sum_lambda = cloth_triangle_get_lambda_sum_lambda(constraints, cid)
    lambda_sum_mu = cloth_triangle_get_lambda_sum_mu(constraints, cid)

    x_a = get_position(store, body_a)
    x_b = get_position(store, body_b)
    x_c = get_position(store, body_c)
    # Per-vertex displacement within the substep -- the Rayleigh damping
    # term ``gradC . dx`` projects this onto the constraint gradient.
    # Free particles see ``gradC = 0``, so this damping does not affect
    # free fall (no "ether"); only motion that loads the constraint is
    # dissipated.  Mirrors the Macklin XPBD 2016 eq. 26 form.
    dx_a = x_a - get_position_prev_substep(store, body_a)
    dx_b = x_b - get_position_prev_substep(store, body_b)
    dx_c = x_c - get_position_prev_substep(store, body_c)

    dt = wp.float32(1.0) / idt
    idt_sq = idt * idt
    bias_lambda = idt_sq * alpha_lambda
    bias_mu = idt_sq * alpha_mu
    gamma_lambda = beta_lambda * dt
    gamma_mu = beta_mu * dt

    # ----- Pass 1: area row (row lambda) ------------------------------
    x_axis, y_axis = _build_projector(x_a, x_b, x_c)
    xab2 = _project_to_2d(x_axis, y_axis, x_b - x_a)
    xac2 = _project_to_2d(x_axis, y_axis, x_c - x_a)

    # F = invRest * [xAB_2D | xAC_2D] (matching FemTriPBD.cs:153-156).
    f00 = inv_rest[0, 0] * xab2[0] + inv_rest[1, 0] * xac2[0]
    f01 = inv_rest[0, 1] * xab2[0] + inv_rest[1, 1] * xac2[0]
    f10 = inv_rest[0, 0] * xab2[1] + inv_rest[1, 0] * xac2[1]
    f11 = inv_rest[0, 1] * xab2[1] + inv_rest[1, 1] * xac2[1]

    rotation = _extract_rotation_2d(f00, f01, f10, f11, rotation)

    # Constants for the gradient (FemTriPBD.cs:161-162).
    q = inv_rest[0, 0] + inv_rest[1, 0]  # column-0 sum, [m^-1]
    t = inv_rest[0, 1] + inv_rest[1, 1]  # column-1 sum, [m^-1]

    # Gradients of ``C_lambda = (det F - 1) * rest_area`` w.r.t. each
    # vertex's 2D position.  Direct port of FemTriPBD.cs:163-168.
    g1_a_x = (-q * f11 + t * f10) * rest_area
    g1_a_y = (-t * f00 + q * f01) * rest_area
    g1_b_x = (inv_rest[0, 0] * f11 - f10 * inv_rest[0, 1]) * rest_area
    g1_b_y = (inv_rest[0, 1] * f00 - f01 * inv_rest[0, 0]) * rest_area
    g1_c_x = (inv_rest[1, 0] * f11 - f10 * inv_rest[1, 1]) * rest_area
    g1_c_y = (inv_rest[1, 1] * f00 - f01 * inv_rest[1, 0]) * rest_area

    c_lambda = (f00 * f11 - (wp.float32(1.0) + f10 * f01)) * rest_area

    # Rayleigh damping numerator term: gamma * sum_i (grad_i . dx_i),
    # projected to 2D via the per-vertex 2D gradients above.  ``dx_i``
    # is the substep displacement; project it to the same 2D frame
    # before dotting with the 2D gradient.
    dx_a_2d_lam = _project_to_2d(x_axis, y_axis, dx_a)
    dx_b_2d_lam = _project_to_2d(x_axis, y_axis, dx_b)
    dx_c_2d_lam = _project_to_2d(x_axis, y_axis, dx_c)
    grad_dot_dx_lambda = (
        g1_a_x * dx_a_2d_lam[0] + g1_a_y * dx_a_2d_lam[1]
        + g1_b_x * dx_b_2d_lam[0] + g1_b_y * dx_b_2d_lam[1]
        + g1_c_x * dx_c_2d_lam[0] + g1_c_y * dx_c_2d_lam[1]
    )

    grad_dot_grad_inv_m_lambda = (
        inv_mass_a * (g1_a_x * g1_a_x + g1_a_y * g1_a_y)
        + inv_mass_b * (g1_b_x * g1_b_x + g1_b_y * g1_b_y)
        + inv_mass_c * (g1_c_x * g1_c_x + g1_c_y * g1_c_y)
    )
    denom_lambda = (wp.float32(1.0) + gamma_lambda) * grad_dot_grad_inv_m_lambda + bias_lambda

    if denom_lambda > wp.float32(0.0):
        # Macklin XPBD 2016 eq. 26 with damping.
        d_lam_lambda = -(c_lambda + bias_lambda * lambda_sum_lambda + gamma_lambda * grad_dot_dx_lambda) / denom_lambda
        delta_a = wp.vec2f(inv_mass_a * g1_a_x * d_lam_lambda, inv_mass_a * g1_a_y * d_lam_lambda)
        delta_b = wp.vec2f(inv_mass_b * g1_b_x * d_lam_lambda, inv_mass_b * g1_b_y * d_lam_lambda)
        delta_c = wp.vec2f(inv_mass_c * g1_c_x * d_lam_lambda, inv_mass_c * g1_c_y * d_lam_lambda)
        x_a = x_a + _project_to_3d(x_axis, y_axis, delta_a)
        x_b = x_b + _project_to_3d(x_axis, y_axis, delta_b)
        x_c = x_c + _project_to_3d(x_axis, y_axis, delta_c)
        lambda_sum_lambda = lambda_sum_lambda + d_lam_lambda

    # ----- Pass 2: shear row (row mu) ---------------------------------
    # Rebuild projector + F so the shear row sees the area-corrected
    # positions (FemTriPBD.cs:185-194).
    x_axis, y_axis = _build_projector(x_a, x_b, x_c)
    xab2 = _project_to_2d(x_axis, y_axis, x_b - x_a)
    xac2 = _project_to_2d(x_axis, y_axis, x_c - x_a)
    f00 = inv_rest[0, 0] * xab2[0] + inv_rest[1, 0] * xac2[0]
    f01 = inv_rest[0, 1] * xab2[0] + inv_rest[1, 1] * xac2[0]
    f10 = inv_rest[0, 0] * xab2[1] + inv_rest[1, 0] * xac2[1]
    f11 = inv_rest[0, 1] * xab2[1] + inv_rest[1, 1] * xac2[1]
    rotation = _extract_rotation_2d(f00, f01, f10, f11, rotation)
    cr = wp.cos(rotation)
    sr = wp.sin(rotation)
    # R columns: col0 = (cr, sr), col1 = (-sr, cr) -> M11=cr, M12=-sr, M21=sr, M22=cr.
    df = f00 - cr
    dh = f01 - (-sr)
    dm = f10 - sr
    dq = f11 - cr
    du_sq = df * df + dm * dm + dh * dh + dq * dq
    if du_sq > _DET_F_EPS:
        du = wp.sqrt(du_sq)
        dx = rest_area / wp.max(_DET_F_EPS, wp.float32(2.0) * du)
        # Gradients of ``C_mu = ||F - R||_F * rest_area`` (FemTriPBD.cs:203-208).
        g2_a_x = wp.float32(-2.0) * (df * q + dh * t) * dx
        g2_a_y = wp.float32(-2.0) * (dm * q + dq * t) * dx
        g2_b_x = wp.float32(2.0) * (df * inv_rest[0, 0] + dh * inv_rest[0, 1]) * dx
        g2_b_y = wp.float32(2.0) * (dm * inv_rest[0, 0] + dq * inv_rest[0, 1]) * dx
        g2_c_x = wp.float32(2.0) * (df * inv_rest[1, 0] + dh * inv_rest[1, 1]) * dx
        g2_c_y = wp.float32(2.0) * (dm * inv_rest[1, 0] + dq * inv_rest[1, 1]) * dx
        c_mu = du * rest_area
        # Rayleigh damping for the shear row, same form as the area row.
        dx_a_2d_mu = _project_to_2d(x_axis, y_axis, dx_a)
        dx_b_2d_mu = _project_to_2d(x_axis, y_axis, dx_b)
        dx_c_2d_mu = _project_to_2d(x_axis, y_axis, dx_c)
        grad_dot_dx_mu = (
            g2_a_x * dx_a_2d_mu[0] + g2_a_y * dx_a_2d_mu[1]
            + g2_b_x * dx_b_2d_mu[0] + g2_b_y * dx_b_2d_mu[1]
            + g2_c_x * dx_c_2d_mu[0] + g2_c_y * dx_c_2d_mu[1]
        )
        grad_dot_grad_inv_m_mu = (
            inv_mass_a * (g2_a_x * g2_a_x + g2_a_y * g2_a_y)
            + inv_mass_b * (g2_b_x * g2_b_x + g2_b_y * g2_b_y)
            + inv_mass_c * (g2_c_x * g2_c_x + g2_c_y * g2_c_y)
        )
        denom_mu = (wp.float32(1.0) + gamma_mu) * grad_dot_grad_inv_m_mu + bias_mu
        if denom_mu > wp.float32(0.0):
            d_lam_mu = -(c_mu + bias_mu * lambda_sum_mu + gamma_mu * grad_dot_dx_mu) / denom_mu
            delta_a = wp.vec2f(inv_mass_a * g2_a_x * d_lam_mu, inv_mass_a * g2_a_y * d_lam_mu)
            delta_b = wp.vec2f(inv_mass_b * g2_b_x * d_lam_mu, inv_mass_b * g2_b_y * d_lam_mu)
            delta_c = wp.vec2f(inv_mass_c * g2_c_x * d_lam_mu, inv_mass_c * g2_c_y * d_lam_mu)
            x_a = x_a + _project_to_3d(x_axis, y_axis, delta_a)
            x_b = x_b + _project_to_3d(x_axis, y_axis, delta_b)
            x_c = x_c + _project_to_3d(x_axis, y_axis, delta_c)
            lambda_sum_mu = lambda_sum_mu + d_lam_mu

    # Scatter the (possibly mutated) particle positions back.
    set_position(store, body_a, x_a)
    set_position(store, body_b, x_b)
    set_position(store, body_c, x_c)

    # Persist warm-start state.
    cloth_triangle_set_rotation(constraints, cid, rotation)
    cloth_triangle_set_lambda_sum_lambda(constraints, cid, lambda_sum_lambda)
    cloth_triangle_set_lambda_sum_mu(constraints, cid, lambda_sum_mu)


@wp.func
def cloth_triangle_relax_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    store: BodyOrParticleStore,
    idt: wp.float32,
):
    """Relax-phase wrapper: run iterate, then sync vertices to ``VELOCITY_LEVEL``.

    PhoenX's TGS-soft loop alternates a position-iterate phase and a
    velocity-relax phase.  Cloth XPBD is naturally position-level, so
    the relax wrapper runs the standard iterate (which writes
    positions) and then forces ``VELOCITY_LEVEL`` on the three
    vertices.  The ``POSITION_LEVEL -> VELOCITY_LEVEL`` sync inside
    :func:`set_access_mode` runs the finite-diff
    ``vel = (pos - pos_prev) / dt`` so the position correction is
    folded into the velocity field for the next velocity-level reader
    (e.g. a subsequent contact-relax sweep on the same particle).
    """
    body_a = cloth_triangle_get_body1(constraints, cid)
    body_b = cloth_triangle_get_body2(constraints, cid)
    body_c = cloth_triangle_get_body3(constraints, cid)

    cloth_triangle_iterate_at(constraints, cid, base_offset, store, idt)

    set_access_mode(store, body_a, _ACCESS_MODE_VELOCITY_LEVEL, idt)
    set_access_mode(store, body_b, _ACCESS_MODE_VELOCITY_LEVEL, idt)
    set_access_mode(store, body_c, _ACCESS_MODE_VELOCITY_LEVEL, idt)
