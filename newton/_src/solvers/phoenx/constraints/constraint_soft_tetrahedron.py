# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level XPBD soft-body tetrahedron (PhysX co-rotated ARAP).

Single per-tet XPBD row per PGS sweep::

    C = V_rest * ||F - R||_F
    F = (xB-xA, xC-xA, xD-xA) * inv_rest
    R = polar(F)               # warm-started across substeps

i.e. the rest-volume-weighted Frobenius norm of the deviation
between the deformation gradient and its closest rotation. Linear
in strain magnitude, so stays well-conditioned for stiff materials
(unlike a Neo-Hookean Voigt ``||F||_F^2 - 3`` which grows
quadratically).

Mirrors PhysX's ``ARAP_constraint`` (``softBodyGM.cu:722``) plus
their ``compute_dCdx`` chain-rule shortcut (``softBodyGM.cu:536``):
per-vertex gradients are obtained via
``dC/dx_s = dC/dF * inv_rest_row[s-1]`` (three mat33-vec3 products
plus a sum-rule), instead of the prior 12-coordinate hand-expansion.

The XPBD compliance ``alpha_mu = 1 / k_mu`` follows the original
PhoenX / Jitter2 convention, with the rest volume folded into the
constraint magnitude rather than into ``alpha``. (PhysX bakes V
into ``alpha`` instead; either choice is mathematically equivalent
once the units of ``k`` are interpreted accordingly.)

Rotation extraction schedule mirrors PhysX
(``softBodyGM.cu:288-298``)::

    prepare: ~30 Mueller polar-decomp iters (cold-start each substep)
    iterate:  ~4 iters (refine warm-started quaternion)

Drops a separate ``det(F) - 1`` volumetric row -- the ARAP constraint
already captures volumetric deviation through the Hookean
linearisation, matching the Jitter2 reference. For high Poisson
ratios (near 0.5) callers can later opt in to an explicit volume row.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_POSITION_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON,
    ConstraintContainer,
    assert_constraint_header,
    read_float,
    read_int,
    read_mat33,
    read_quat,
    write_float,
    write_int,
    write_mat33,
    write_quat,
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
    "SOFT_TET_DWORDS",
    "SoftTetrahedronData",
    "soft_tet_init_rows_kernel",
    "soft_tet_lame_from_youngs_poisson",
    "soft_tetrahedron_iterate_at",
    "soft_tetrahedron_prepare_for_iteration_at",
    "soft_tetrahedron_set_alpha_lambda",
    "soft_tetrahedron_set_alpha_mu",
    "soft_tetrahedron_set_beta_lambda",
    "soft_tetrahedron_set_beta_mu",
    "soft_tetrahedron_set_body1",
    "soft_tetrahedron_set_body2",
    "soft_tetrahedron_set_body3",
    "soft_tetrahedron_set_body4",
    "soft_tetrahedron_set_inv_rest",
    "soft_tetrahedron_set_rest_volume",
    "soft_tetrahedron_set_type",
]


_PHOENX_SOFT_TET_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))


# ---------------------------------------------------------------------------
# Lame parameter conversion (host-side helper).
# ---------------------------------------------------------------------------


def soft_tet_lame_from_youngs_poisson(
    youngs_modulus: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """3D Lame parameters ``(lambda, mu)`` from ``(E, nu)``::

        lambda = E * nu / ((1 + nu) * (1 - 2 nu))
        mu = E / (2 * (1 + nu))

    Blows up at ``nu = 0.5`` (incompressible limit). For practical soft
    bodies pick ``nu`` in ``[0.3, 0.45]``. Mirrors Jitter2's
    ``ConstraintHelper.CalculateLameParameters``.
    """
    if youngs_modulus <= 0.0:
        raise ValueError(f"youngs_modulus must be positive (got {youngs_modulus})")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError(f"poisson_ratio must be in (-1, 0.5) (got {poisson_ratio})")
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    return float(lam), float(mu)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class SoftTetrahedronData:
    """Per-constraint dword-layout schema for one soft-body tetrahedron.

    Mirrors ``FemTetPBD`` field-for-field with PhoenX's mandatory
    three-int32 header (``constraint_type`` / ``body1`` / ``body2`` at
    dwords 0, 1, 2).
    """

    constraint_type: wp.int32
    body1: wp.int32  # particle index of node A
    body2: wp.int32  # particle index of node B
    body3: wp.int32  # particle index of node C
    body4: wp.int32  # particle index of node D

    inv_rest: wp.mat33f
    rest_volume: wp.float32

    alpha_lambda: wp.float32  # 1 / Lame lambda (volume compliance, reserved)
    alpha_mu: wp.float32  # 1 / Lame mu (shear compliance)

    # Macklin XPBD damping coefficients. ``gamma = beta * dt`` enters
    # the lambda numerator as ``gamma * grad . (x - position_prev_substep)``
    # -- velocity-projected damping that does NOT damp at rest, unlike
    # ether damping. Mirrors the cloth-triangle iterate's same-shape
    # damping term (Macklin et al. 2020 "Detailed Rigid Body
    # Simulation with XPBD"). Jitter2 ``FemTetPBD`` omits this term;
    # Newton's variant collapses to bare XPBD when ``beta == 0``.
    beta_lambda: wp.float32  # PD damping on volume row (reserved -- volume row not yet implemented)
    beta_mu: wp.float32  # PD damping on shear row

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32
    inv_mass_d: wp.float32

    rotation: wp.quatf  # corotational warm-start (3D analogue of cloth's scalar angle)
    lambda_sum_lambda: wp.float32  # volume row accumulator (reserved)
    lambda_sum_mu: wp.float32  # shear row accumulator

    #: Opt-in per-column wall-clock accumulator (microseconds). See
    #: :func:`constraint_accumulate_time_us`.
    time_us: wp.float32


assert_constraint_header(SoftTetrahedronData)


_OFF_BODY1 = wp.constant(dword_offset_of(SoftTetrahedronData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(SoftTetrahedronData, "body2"))
_OFF_BODY3 = wp.constant(dword_offset_of(SoftTetrahedronData, "body3"))
_OFF_BODY4 = wp.constant(dword_offset_of(SoftTetrahedronData, "body4"))
_OFF_INV_REST = wp.constant(dword_offset_of(SoftTetrahedronData, "inv_rest"))
_OFF_REST_VOLUME = wp.constant(dword_offset_of(SoftTetrahedronData, "rest_volume"))
_OFF_ALPHA_LAMBDA = wp.constant(dword_offset_of(SoftTetrahedronData, "alpha_lambda"))
_OFF_ALPHA_MU = wp.constant(dword_offset_of(SoftTetrahedronData, "alpha_mu"))
_OFF_BETA_LAMBDA = wp.constant(dword_offset_of(SoftTetrahedronData, "beta_lambda"))
_OFF_BETA_MU = wp.constant(dword_offset_of(SoftTetrahedronData, "beta_mu"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(SoftTetrahedronData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(SoftTetrahedronData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(SoftTetrahedronData, "inv_mass_c"))
_OFF_INV_MASS_D = wp.constant(dword_offset_of(SoftTetrahedronData, "inv_mass_d"))
_OFF_ROTATION = wp.constant(dword_offset_of(SoftTetrahedronData, "rotation"))
_OFF_LAMBDA_SUM_LAMBDA = wp.constant(dword_offset_of(SoftTetrahedronData, "lambda_sum_lambda"))
_OFF_LAMBDA_SUM_MU = wp.constant(dword_offset_of(SoftTetrahedronData, "lambda_sum_mu"))
SOFT_TET_TIME_US_OFFSET = wp.constant(dword_offset_of(SoftTetrahedronData, "time_us"))

SOFT_TET_DWORDS: int = num_dwords(SoftTetrahedronData)


@wp.func
def soft_tetrahedron_set_type(c: ConstraintContainer, cid: wp.int32):
    write_int(c, wp.int32(0), cid, CONSTRAINT_TYPE_SOFT_TETRAHEDRON)


@wp.func
def soft_tetrahedron_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def soft_tetrahedron_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def soft_tetrahedron_set_body3(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY3, cid, v)


@wp.func
def soft_tetrahedron_set_body4(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY4, cid, v)


@wp.func
def soft_tetrahedron_set_inv_rest(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_INV_REST, cid, v)


@wp.func
def soft_tetrahedron_set_rest_volume(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_VOLUME, cid, v)


@wp.func
def soft_tetrahedron_set_alpha_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_LAMBDA, cid, v)


@wp.func
def soft_tetrahedron_set_alpha_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_MU, cid, v)


@wp.func
def soft_tetrahedron_set_beta_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_LAMBDA, cid, v)


@wp.func
def soft_tetrahedron_set_beta_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_MU, cid, v)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_DET_F_EPS = wp.constant(wp.float32(1.0e-8))
_ARAP_EPS = wp.constant(wp.float32(1.0e-8))
_EXTRACT_ROT_EPS = wp.constant(wp.float32(1.0e-6))
#: Cold-start polar-decomposition iteration count, used in
#: :func:`soft_tetrahedron_prepare_for_iteration_at`. Mirrors PhysX's
#: ``isFirstIteration ? 100`` branch (``softBodyGM.cu:290``); 30 is
#: sufficient on a quaternion warm-started from the previous substep
#: because the rotation barely changes between substeps in practice.
_PREPARE_ROT_ITERS = wp.constant(wp.int32(30))
#: Refine polar-decomposition iteration count, used per PGS sweep
#: inside :func:`soft_tetrahedron_iterate_at`. Mirrors PhysX's
#: ``isFirstIteration ? ... : 4`` branch (``softBodyGM.cu:290``);
#: 15 here matches the original PhoenX (Jitter2 port) so the
#: momentum-conservation test sees identical steady-state behaviour.
#: 4 also converges in practice once warm-started, but consumes a
#: little more contact-impulse budget during settling.
_ITERATE_ROT_ITERS = wp.constant(wp.int32(15))


# ---------------------------------------------------------------------------
# Mueller polar decomposition (quaternion-axis iteration). Direct port
# of ``jitterphysics2/.../ConstraintHelper.cs:ExtractRotation``, with
# the per-call iteration count parameterised so prepare can cold-start
# and iterate can cheaply refine.
# ---------------------------------------------------------------------------


@wp.func
def _extract_rotation(F: wp.mat33f, q_init: wp.quatf, max_iters: wp.int32) -> wp.quatf:
    """Closest-rotation quaternion from deformation gradient via
    Mueller polar decomposition, warm-started from ``q_init``."""
    q = q_init
    for _ in range(max_iters):
        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]
        d_ = qw * qx + qy * qz
        f_ = qx * qx
        g_ = qy * qy
        j_ = wp.float32(1.0) - wp.float32(2.0) * (f_ + g_)
        m_ = qz * qz
        p_ = wp.float32(1.0) - wp.float32(2.0) * (f_ + m_)
        s_ = qy * qz - qw * qx
        w_ = qx * qz - qw * qy
        bb = qw * qz + qx * qy
        bl = wp.float32(1.0) - wp.float32(2.0) * (g_ + m_)
        bn = qx * qy - qw * qz
        bs = qw * qy + qx * qz

        F11 = F[0, 0]
        F12 = F[0, 1]
        F13 = F[0, 2]
        F21 = F[1, 0]
        F22 = F[1, 1]
        F23 = F[1, 2]
        F31 = F[2, 0]
        F32 = F[2, 1]
        F33 = F[2, 2]

        denom = (
            wp.abs(
                j_ * F33
                + p_ * F22
                + bl * F11
                + wp.float32(2.0) * (bn * F12 + w_ * F31 + s_ * F23 + F13 * bs + F21 * bb + F32 * d_)
            )
            + _EXTRACT_ROT_EPS
        )
        cf = wp.float32(1.0) / denom

        omega_x = (
            -wp.float32(2.0) * F22 * d_ - j_ * F23 + p_ * F32 + wp.float32(2.0) * (s_ * F33 - w_ * F21 + F31 * bb)
        ) * cf
        omega_y = (
            -wp.float32(2.0) * F33 * bs - bl * F31 + j_ * F13 + wp.float32(2.0) * (w_ * F11 - bn * F32 + F12 * d_)
        ) * cf
        omega_z = (
            -wp.float32(2.0) * F11 * bb - p_ * F12 + bl * F21 + wp.float32(2.0) * (bn * F22 - s_ * F13 + F23 * bs)
        ) * cf

        w_mag = wp.sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z)
        if w_mag < _EXTRACT_ROT_EPS:
            break
        inv_w = wp.float32(1.0) / w_mag
        half = wp.float32(0.5) * w_mag
        sh = wp.sin(half)
        ch = wp.cos(half)
        tq = wp.quatf(omega_x * inv_w * sh, omega_y * inv_w * sh, omega_z * inv_w * sh, ch)
        q = wp.normalize(tq * q)
    return q


@wp.func
def _compute_F(
    x_a: wp.vec3f,
    x_b: wp.vec3f,
    x_c: wp.vec3f,
    x_d: wp.vec3f,
    inv_rest: wp.mat33f,
) -> wp.mat33f:
    """Deformation gradient ``F = D * inv_rest`` where ``D``'s columns
    are the three rest-from-A edges."""
    eAB = x_b - x_a
    eAC = x_c - x_a
    eAD = x_d - x_a
    F00 = inv_rest[0, 0] * eAB[0] + inv_rest[1, 0] * eAC[0] + inv_rest[2, 0] * eAD[0]
    F01 = inv_rest[0, 1] * eAB[0] + inv_rest[1, 1] * eAC[0] + inv_rest[2, 1] * eAD[0]
    F02 = inv_rest[0, 2] * eAB[0] + inv_rest[1, 2] * eAC[0] + inv_rest[2, 2] * eAD[0]
    F10 = inv_rest[0, 0] * eAB[1] + inv_rest[1, 0] * eAC[1] + inv_rest[2, 0] * eAD[1]
    F11 = inv_rest[0, 1] * eAB[1] + inv_rest[1, 1] * eAC[1] + inv_rest[2, 1] * eAD[1]
    F12 = inv_rest[0, 2] * eAB[1] + inv_rest[1, 2] * eAC[1] + inv_rest[2, 2] * eAD[1]
    F20 = inv_rest[0, 0] * eAB[2] + inv_rest[1, 0] * eAC[2] + inv_rest[2, 0] * eAD[2]
    F21 = inv_rest[0, 1] * eAB[2] + inv_rest[1, 1] * eAC[2] + inv_rest[2, 1] * eAD[2]
    F22 = inv_rest[0, 2] * eAB[2] + inv_rest[1, 2] * eAC[2] + inv_rest[2, 2] * eAD[2]
    return wp.mat33f(F00, F01, F02, F10, F11, F12, F20, F21, F22)


# ---------------------------------------------------------------------------
# Per-vertex gradient via the inv_rest chain-rule shortcut
# (PhysX ``compute_dCdx``, ``softBodyGM.cu:536``).
#
# For ``F = D * inv_rest`` with ``D = [xB-xA, xC-xA, xD-xA]``, the chain
# rule gives
#
#     dC/dx_s = (dC/dF) * inv_rest_row[s-1]      for s in {1, 2, 3}
#     dC/dx_0 = -(dC/dx_1 + dC/dx_2 + dC/dx_3)
#
# i.e. three mat33-vec3 products and a 4-vertex sum-rule. Vastly cheaper
# than the explicit 12-coordinate gradient expansion we used to do.
# ---------------------------------------------------------------------------


@wp.func
def _per_vertex_gradients(
    dCdF: wp.mat33f,
    inv_rest: wp.mat33f,
):
    """Chain-rule per-vertex gradients for a constraint expressed via
    ``F``. Returns ``(g_a, g_b, g_c, g_d)`` -- one 3-vector per tet
    vertex.
    """
    # Rows of inv_rest as column vectors (which is the action that
    # appears in the chain rule above).
    inv_row0 = wp.vec3f(inv_rest[0, 0], inv_rest[0, 1], inv_rest[0, 2])
    inv_row1 = wp.vec3f(inv_rest[1, 0], inv_rest[1, 1], inv_rest[1, 2])
    inv_row2 = wp.vec3f(inv_rest[2, 0], inv_rest[2, 1], inv_rest[2, 2])
    g_b = dCdF * inv_row0
    g_c = dCdF * inv_row1
    g_d = dCdF * inv_row2
    g_a = -(g_b + g_c + g_d)
    return g_a, g_b, g_c, g_d


# ---------------------------------------------------------------------------
# Prepare + iterate
# ---------------------------------------------------------------------------


@wp.func
def soft_tetrahedron_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Substep-entry prepare: flip each vertex's access mode to
    POSITION_LEVEL, cache inverse masses, reset XPBD warm starts, and
    cold-start the polar-decomposition rotation with ~30 iterations
    against the predicted positions (PhysX's
    ``isFirstIteration ? 100 : 4`` schedule, ``softBodyGM.cu:290`` --
    the cold path runs here once per substep so the cheap 4-iter
    refine in :func:`soft_tetrahedron_iterate_at` always has a
    quaternion close to the answer).

    Body fields are unified indices: ``i_p = body - num_bodies`` is the
    particle slot. The persisted ``rotation`` quaternion warm start is
    intentionally NOT reset between substeps -- the closest-rotation
    evolves continuously with the tet's pose so we re-use the previous
    substep's result and only refine.
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies
    p_d = body_d - num_bodies

    # Flip access mode (slot-aware).
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

    _slot_a, inv_factor_a = get_state_index(copy_state, body_a, parallel_id)
    _slot_b, inv_factor_b = get_state_index(copy_state, body_b, parallel_id)
    _slot_c, inv_factor_c = get_state_index(copy_state, body_c, parallel_id)
    _slot_d, inv_factor_d = get_state_index(copy_state, body_d, parallel_id)

    write_float(constraints, _OFF_INV_MASS_A, cid, particles.inverse_mass[p_a] * wp.float32(inv_factor_a))
    write_float(constraints, _OFF_INV_MASS_B, cid, particles.inverse_mass[p_b] * wp.float32(inv_factor_b))
    write_float(constraints, _OFF_INV_MASS_C, cid, particles.inverse_mass[p_c] * wp.float32(inv_factor_c))
    write_float(constraints, _OFF_INV_MASS_D, cid, particles.inverse_mass[p_d] * wp.float32(inv_factor_d))

    write_float(constraints, _OFF_LAMBDA_SUM_LAMBDA, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, wp.float32(0.0))

    # Cold-start polar decomposition against the substep-entry pose.
    x_a, _ifa, _sa = read_position_unified(bodies, particles, copy_state, body_a, parallel_id, num_bodies)
    x_b, _ifb, _sb = read_position_unified(bodies, particles, copy_state, body_b, parallel_id, num_bodies)
    x_c, _ifc, _sc = read_position_unified(bodies, particles, copy_state, body_c, parallel_id, num_bodies)
    x_d, _ifd, _sd = read_position_unified(bodies, particles, copy_state, body_d, parallel_id, num_bodies)
    inv_rest = read_mat33(constraints, _OFF_INV_REST, cid)
    F = _compute_F(x_a, x_b, x_c, x_d, inv_rest)
    rotation = read_quat(constraints, _OFF_ROTATION, cid)
    rotation = _extract_rotation(F, rotation, _PREPARE_ROT_ITERS)
    write_quat(constraints, _OFF_ROTATION, cid, rotation)


@wp.func
def soft_tetrahedron_iterate_at(
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
    """One PhysX-style ARAP PGS sweep on a soft-body tetrahedron.

    Refines the warm-started rotation with 4 Mueller polar-decomp
    iterations (the cold-start 30-iter pass happens once per substep
    in :func:`soft_tetrahedron_prepare_for_iteration_at`), then applies

        C = sqrt(||F - R||_F^2 + eps)
        dC/dF = (F - R) / C
        per-vertex grads via the inv_rest chain-rule shortcut

    with PhysX-style V-baked compliance ``alpha_mu = 1 / (k_mu * V)``.

    Body fields are unified indices: ``i_p = body - num_bodies`` is the
    particle slot. Reads / writes route through the slot-aware unified
    helpers so mass splitting routes position-level work into the slot
    when one exists; without it the helpers fall through to particle
    storage and behaviour matches the rigid-only path.
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies
    p_d = body_d - num_bodies

    # Slot-aware access-mode flip on each vertex.
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

    inv_mass_a = read_float(constraints, _OFF_INV_MASS_A, cid)
    inv_mass_b = read_float(constraints, _OFF_INV_MASS_B, cid)
    inv_mass_c = read_float(constraints, _OFF_INV_MASS_C, cid)
    inv_mass_d = read_float(constraints, _OFF_INV_MASS_D, cid)
    inv_rest = read_mat33(constraints, _OFF_INV_REST, cid)
    rest_volume = read_float(constraints, _OFF_REST_VOLUME, cid)
    alpha_mu = read_float(constraints, _OFF_ALPHA_MU, cid)
    beta_mu = read_float(constraints, _OFF_BETA_MU, cid)
    rotation = read_quat(constraints, _OFF_ROTATION, cid)
    lambda_sum_mu = read_float(constraints, _OFF_LAMBDA_SUM_MU, cid)

    x_a, _ifa, slot_a = read_position_unified(bodies, particles, copy_state, body_a, parallel_id, num_bodies)
    x_b, _ifb, slot_b = read_position_unified(bodies, particles, copy_state, body_b, parallel_id, num_bodies)
    x_c, _ifc, slot_c = read_position_unified(bodies, particles, copy_state, body_c, parallel_id, num_bodies)
    x_d, _ifd, slot_d = read_position_unified(bodies, particles, copy_state, body_d, parallel_id, num_bodies)

    # XPBD damping anchor (Macklin et al. 2020): velocity-projected
    # damping via ``gamma * grad . (x - position_prev_substep)``;
    # ``beta_mu == 0`` recovers bare XPBD.
    dx_a = x_a - particles.position_prev_substep[p_a]
    dx_b = x_b - particles.position_prev_substep[p_b]
    dx_c = x_c - particles.position_prev_substep[p_c]
    dx_d = x_d - particles.position_prev_substep[p_d]

    F = _compute_F(x_a, x_b, x_c, x_d, inv_rest)

    # Cheap refine of the warm-started rotation (PhysX
    # ``softBodyGM.cu:290``: ``isFirstIteration ? 100 : 4``).
    rotation = _extract_rotation(F, rotation, _ITERATE_ROT_ITERS)
    R = wp.quat_to_matrix(rotation)

    # ARAP residual: ``S = F - R``, ``c_norm = sqrt(||S||_F^2 + eps)``.
    # The constraint and gradient include the rest volume so the
    # XPBD compliance interpretation matches the prior PhoenX /
    # Jitter2 convention: ``C = V * ||F - R||_F`` with
    # ``alpha_mu = 1 / k_mu``. This keeps Young's-modulus
    # calibration of existing scenes (e.g. ``example_soft_body_drop``)
    # behaviourally consistent across the refactor.
    S = F - R
    s00 = S[0, 0]
    s01 = S[0, 1]
    s02 = S[0, 2]
    s10 = S[1, 0]
    s11 = S[1, 1]
    s12 = S[1, 2]
    s20 = S[2, 0]
    s21 = S[2, 1]
    s22 = S[2, 2]
    s_norm_sq = (
        s00 * s00 + s01 * s01 + s02 * s02
        + s10 * s10 + s11 * s11 + s12 * s12
        + s20 * s20 + s21 * s21 + s22 * s22
    )
    c_norm = wp.sqrt(s_norm_sq + _ARAP_EPS)

    if c_norm < _DET_F_EPS:
        # Pure rotation: nothing to apply. Persist the refined
        # quaternion and bail.
        write_quat(constraints, _OFF_ROTATION, cid, rotation)
        return

    # C = V * c_norm; dC/dF = V * S / c_norm.
    inv_c = wp.float32(1.0) / c_norm
    dCdF = (rest_volume * inv_c) * S
    c_arap = rest_volume * c_norm
    g_a, g_b, g_c, g_d = _per_vertex_gradients(dCdF, inv_rest)

    idt_sq = idt * idt
    dt = wp.float32(1.0) / idt
    bias_mu = idt_sq * alpha_mu
    gamma_mu = beta_mu * dt

    grad2_im = (
        inv_mass_a * wp.dot(g_a, g_a)
        + inv_mass_b * wp.dot(g_b, g_b)
        + inv_mass_c * wp.dot(g_c, g_c)
        + inv_mass_d * wp.dot(g_d, g_d)
    )
    grad_dot_dx = (
        wp.dot(g_a, dx_a)
        + wp.dot(g_b, dx_b)
        + wp.dot(g_c, dx_c)
        + wp.dot(g_d, dx_d)
    )
    denom = (wp.float32(1.0) + gamma_mu) * grad2_im + bias_mu

    if denom > wp.float32(0.0):
        d_lam = -(c_arap + bias_mu * lambda_sum_mu + gamma_mu * grad_dot_dx) / denom
        d_lam = d_lam * sor_boost
        x_a = x_a + (d_lam * inv_mass_a) * g_a
        x_b = x_b + (d_lam * inv_mass_b) * g_b
        x_c = x_c + (d_lam * inv_mass_c) * g_c
        x_d = x_d + (d_lam * inv_mass_d) * g_d
        lambda_sum_mu = lambda_sum_mu + d_lam

    write_position_unified(bodies, particles, copy_state, body_a, slot_a, num_bodies, x_a)
    write_position_unified(bodies, particles, copy_state, body_b, slot_b, num_bodies, x_b)
    write_position_unified(bodies, particles, copy_state, body_c, slot_c, num_bodies, x_c)
    write_position_unified(bodies, particles, copy_state, body_d, slot_d, num_bodies, x_d)

    write_quat(constraints, _OFF_ROTATION, cid, rotation)
    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, lambda_sum_mu)


@wp.kernel
def soft_tet_init_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    num_bodies: wp.int32,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3f],
    tet_poses: wp.array[wp.mat33f],
    tet_materials: wp.array2d[wp.float32],
    default_beta_lambda: wp.float32,
    default_beta_mu: wp.float32,
):
    """Stamp one soft-tetrahedron row from Newton mesh API.

    Body fields are unified indices: rigid bodies occupy ``[0, num_bodies)``
    and particles occupy ``[num_bodies, num_bodies + num_particles)``.

    ``tet_indices`` is Newton's ``[tet_count, 4]`` 2D layout (one row
    per tet, four particle indices per row). ``tet_poses[t]`` already
    stores the INVERTED rest pose ``inv(Dm)`` where ``Dm`` columns are
    ``(xB-xA, xC-xA, xD-xA)`` (see ``ModelBuilder.add_tetrahedron``).

    ``tet_materials[t, 0]`` is ``k_mu`` (shear modulus, Pa);
    ``tet_materials[t, 1]`` is ``k_lambda`` (volumetric Lame parameter,
    Pa); ``tet_materials[t, 2]`` is ``k_damp`` (currently unused).
    XPBD compliance ``alpha = 1 / k`` mirrors cloth-triangle init.
    """
    t = wp.tid()
    cid = cid_offset + t

    pa = tet_indices[t, 0]
    pb = tet_indices[t, 1]
    pc = tet_indices[t, 2]
    pd = tet_indices[t, 3]

    soft_tetrahedron_set_type(constraints, cid)
    soft_tetrahedron_set_body1(constraints, cid, num_bodies + pa)
    soft_tetrahedron_set_body2(constraints, cid, num_bodies + pb)
    soft_tetrahedron_set_body3(constraints, cid, num_bodies + pc)
    soft_tetrahedron_set_body4(constraints, cid, num_bodies + pd)

    xa = particle_q[pa]
    xb = particle_q[pb]
    xc = particle_q[pc]
    xd = particle_q[pd]

    # Rest volume = (1/6) * |det(Dm)| where Dm columns are edges from A.
    # Sign convention follows Newton's ``add_tetrahedron`` (positive for
    # properly-ordered tets); use absolute value for safety.
    e_ab = xb - xa
    e_ac = xc - xa
    e_ad = xd - xa
    det_dm = wp.dot(e_ab, wp.cross(e_ac, e_ad))
    rest_volume = wp.abs(det_dm) * (wp.float32(1.0) / wp.float32(6.0))
    soft_tetrahedron_set_rest_volume(constraints, cid, rest_volume)

    # ``tet_poses[t]`` is pre-inverted by Newton's builder; use directly.
    soft_tetrahedron_set_inv_rest(constraints, cid, tet_poses[t])

    k_mu = tet_materials[t, 0]
    if k_mu < _PHOENX_SOFT_TET_STIFFNESS_FLOOR:
        k_mu = _PHOENX_SOFT_TET_STIFFNESS_FLOOR
    k_lambda = tet_materials[t, 1]
    if k_lambda < _PHOENX_SOFT_TET_STIFFNESS_FLOOR:
        k_lambda = _PHOENX_SOFT_TET_STIFFNESS_FLOOR
    soft_tetrahedron_set_alpha_lambda(constraints, cid, wp.float32(1.0) / k_lambda)
    soft_tetrahedron_set_alpha_mu(constraints, cid, wp.float32(1.0) / k_mu)
    soft_tetrahedron_set_beta_lambda(constraints, cid, default_beta_lambda)
    soft_tetrahedron_set_beta_mu(constraints, cid, default_beta_mu)

    # Identity quaternion warm-start; the iterate's polar decomposition
    # will refine on the first call.
    write_quat(constraints, _OFF_ROTATION, cid, wp.quatf(0.0, 0.0, 0.0, 1.0))
    write_float(constraints, _OFF_LAMBDA_SUM_LAMBDA, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, wp.float32(0.0))
