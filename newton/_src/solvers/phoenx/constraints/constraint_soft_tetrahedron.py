# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level XPBD soft-body tetrahedron, Jitter2 ``FemTetPBD`` port.

Direct 3D analogue of :mod:`constraint_cloth_triangle`. Four-vertex
volumetric element with one corotational XPBD row:

* **Shear (mu) row** -- ``C = ||F - R||_F * rest_volume`` where ``F`` is
  the 3x3 deformation gradient and ``R`` is the corotational rotation
  extracted from ``F`` via Mueller polar decomposition (quaternion-axis
  iteration). ``F = (xB-xA, xC-xA, xD-xA) * inv_rest``.

The volume preservation row (``det(F)-1``) is intentionally NOT
applied -- the Jitter2 reference port (``FemTetPBD.cs:215-228``) computes
its gradients but leaves the lambda update commented out. ``||F-R||_F``
already captures pure-volumetric deviation (when ``F = c*R``, the row
fires), so a single corotational row is sufficient for the Hookean
energy. Adding the explicit ``det`` row is a follow-up.

The math is line-for-line identical to
``jitterphysics2/.../FemTetPBD.cs:151-298``. Only the shear row's
analytic gradients (``grad2_1..grad2_12``) are evaluated.
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

    # Macklin XPBD damping coefficients (gamma = beta * dt enters the
    # lambda numerator as ``gamma * grad . (x - position_prev_substep)``).
    # Reserved -- the soft-tet iterate currently runs bare XPBD without
    # the damping term (no ``position_prev_substep`` read). Cloth-tri
    # uses the equivalent formulation; mirror that pattern when
    # implementing soft-tet damping.
    beta_lambda: wp.float32  # PD damping on volume row (reserved)
    beta_mu: wp.float32  # PD damping on shear row (reserved)

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32
    inv_mass_d: wp.float32

    rotation: wp.quatf  # corotational warm-start (3D analogue of cloth's scalar angle)
    lambda_sum_lambda: wp.float32  # volume row accumulator (reserved)
    lambda_sum_mu: wp.float32  # shear row accumulator


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
# Rotation extraction (3D Mueller polar decomposition, quaternion-axis
# iteration). Direct port of
# ``jitterphysics2/.../ConstraintHelper.cs:ExtractRotation(JMatrix, ref JQuaternion)``.
# ---------------------------------------------------------------------------


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_EXTRACT_ROT_EPS = wp.constant(wp.float32(1.0e-6))
_EXTRACT_ROT_MAX_ITERS = wp.constant(wp.int32(15))
_DET_F_EPS = wp.constant(wp.float32(1.0e-8))


@wp.func
def _extract_rotation_3d(F: wp.mat33f, q_init: wp.quatf) -> wp.quatf:
    """3D closest-rotation extraction (Mueller polar decomposition by
    quaternion-axis iteration). Warm-starts from ``q_init``.

    Per-iteration: build the rotation matrix's columns implicitly from
    the quaternion components, compute ``omega = sum_i (R_col_i x
    F_col_i) / |sum_i R_col_i . F_col_i|``, rotate the quaternion by
    ``angle = |omega|`` about ``omega/|omega|``, and re-normalise.
    Convergence after 4-15 iterations is typical.

    Direct port of ``ConstraintHelper.ExtractRotation`` (3D variant).
    """
    q = q_init
    for _ in range(_EXTRACT_ROT_MAX_ITERS):
        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]
        # Pre-compute quaternion-derived matrix entries (mirrors Jitter2
        # variable names _d, _f, _g, ..., _bs).
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

        # F_ij is stored row-major: F[i, j] = M(i+1)(j+1) in Jitter2 notation.
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

        omega = wp.vec3f(omega_x, omega_y, omega_z)
        w_mag = wp.sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z)
        if w_mag < _EXTRACT_ROT_EPS:
            break
        axis = omega * (wp.float32(1.0) / w_mag)
        # Build the delta-rotation quaternion: q_d = (axis * sin(w/2), cos(w/2)).
        half = wp.float32(0.5) * w_mag
        sh = wp.sin(half)
        ch = wp.cos(half)
        tq = wp.quatf(axis[0] * sh, axis[1] * sh, axis[2] * sh, ch)
        q = tq * q
        q = wp.normalize(q)
    return q


@wp.func
def _quat_to_mat33(q: wp.quatf) -> wp.mat33f:
    """Build the 3x3 rotation matrix from a (xyz, w) quaternion."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return wp.mat33f(
        wp.float32(1.0) - wp.float32(2.0) * (yy + zz),
        wp.float32(2.0) * (xy - wz),
        wp.float32(2.0) * (xz + wy),
        wp.float32(2.0) * (xy + wz),
        wp.float32(1.0) - wp.float32(2.0) * (xx + zz),
        wp.float32(2.0) * (yz - wx),
        wp.float32(2.0) * (xz - wy),
        wp.float32(2.0) * (yz + wx),
        wp.float32(1.0) - wp.float32(2.0) * (xx + yy),
    )


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
    POSITION_LEVEL, cache inverse masses, reset XPBD warm starts.

    Body fields are unified indices: ``i_p = body - num_bodies`` is the
    particle slot. The persisted ``rotation`` quaternion warm start is
    intentionally NOT reset -- the closest-rotation evolves continuously
    with the tet's pose.

    Direct port of Jitter2's ``FemTetPBD.PrepareForIteration``
    (``FemTetPBD.cs:82-118``): every vertex's access mode is flipped
    via the slot-aware unified helper before reads / writes.
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies
    p_d = body_d - num_bodies

    # Flip access mode (slot-aware). Mirrors C# FemTetPBD prepare.
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
):
    """One XPBD sweep on a soft-body tetrahedron (corotational shear row).

    Direct port of ``FemTetPBD.Iterate`` (``FemTetPBD.cs:123-298``):
    only the shear row is applied (the volume row in the reference is
    commented out; the corotational ``||F-R||_F`` already captures
    volumetric deviation through Hookean linearisation).

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

    # Slot-aware access-mode flip on each vertex (C# FemTetPBD pattern).
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
    rest_volume = read_float(constraints, _OFF_REST_VOLUME, cid)
    inv_rest = read_mat33(constraints, _OFF_INV_REST, cid)
    alpha_mu = read_float(constraints, _OFF_ALPHA_MU, cid)
    rotation = read_quat(constraints, _OFF_ROTATION, cid)
    lambda_sum_mu = read_float(constraints, _OFF_LAMBDA_SUM_MU, cid)

    # Slot-aware position reads.
    x_a, _ifa, slot_a = read_position_unified(bodies, particles, copy_state, body_a, parallel_id, num_bodies)
    x_b, _ifb, slot_b = read_position_unified(bodies, particles, copy_state, body_b, parallel_id, num_bodies)
    x_c, _ifc, slot_c = read_position_unified(bodies, particles, copy_state, body_c, parallel_id, num_bodies)
    x_d, _ifd, slot_d = read_position_unified(bodies, particles, copy_state, body_d, parallel_id, num_bodies)

    # Deformation gradient F = (xB-xA, xC-xA, xD-xA) * inv_rest.
    # Layout matches Jitter2 (row-major, F.M_ij is F[i-1, j-1]).
    eAB = x_b - x_a
    eAC = x_c - x_a
    eAD = x_d - x_a
    # Column-by-column construction:
    #   F[i, j] = inv_rest[0, j] * eAB[i] + inv_rest[1, j] * eAC[i] + inv_rest[2, j] * eAD[i]
    # Jitter2's invRest is stored row-major; the formula is equivalent
    # because we treat invRest's rows / cols consistently below.
    a_x = eAB[0]
    c_x = eAC[0]
    f_x = eAD[0]
    F11 = inv_rest[0, 0] * a_x + inv_rest[1, 0] * c_x + inv_rest[2, 0] * f_x
    F12 = inv_rest[0, 1] * a_x + inv_rest[1, 1] * c_x + inv_rest[2, 1] * f_x
    F13 = inv_rest[0, 2] * a_x + inv_rest[1, 2] * c_x + inv_rest[2, 2] * f_x
    s_y = eAB[1]
    u_y = eAC[1]
    x_y = eAD[1]
    F21 = inv_rest[0, 0] * s_y + inv_rest[1, 0] * u_y + inv_rest[2, 0] * x_y
    F22 = inv_rest[0, 1] * s_y + inv_rest[1, 1] * u_y + inv_rest[2, 1] * x_y
    F23 = inv_rest[0, 2] * s_y + inv_rest[1, 2] * u_y + inv_rest[2, 2] * x_y
    bk_z = eAB[2]
    bm_z = eAC[2]
    bp_z = eAD[2]
    F31 = inv_rest[0, 0] * bk_z + inv_rest[1, 0] * bm_z + inv_rest[2, 0] * bp_z
    F32 = inv_rest[0, 1] * bk_z + inv_rest[1, 1] * bm_z + inv_rest[2, 1] * bp_z
    F33 = inv_rest[0, 2] * bk_z + inv_rest[1, 2] * bm_z + inv_rest[2, 2] * bp_z

    F = wp.mat33f(F11, F12, F13, F21, F22, F23, F31, F32, F33)
    rotation = _extract_rotation_3d(F, rotation)
    rot = _quat_to_mat33(rotation)

    # Sums of invRest entries, mirroring Jitter2's _cs, _cg, _cm.
    cs = inv_rest[0, 0] + inv_rest[1, 0] + inv_rest[2, 0]
    cg = inv_rest[0, 1] + inv_rest[1, 1] + inv_rest[2, 1]
    cm = inv_rest[0, 2] + inv_rest[1, 2] + inv_rest[2, 2]

    # Strain = F - R, expanded so each entry maps to a single Jitter2 var
    # (_kz, _ld, _li, _lp, _mb, _ml, _lu, _mf, _ms).
    kz = F11 - rot[0, 0]
    ld = F12 - rot[0, 1]
    li = F13 - rot[0, 2]
    lp = F21 - rot[1, 0]
    mb = F22 - rot[1, 1]
    ml = F23 - rot[1, 2]
    lu = F31 - rot[2, 0]
    mf = F32 - rot[2, 1]
    ms = F33 - rot[2, 2]

    mv = wp.sqrt(kz * kz + lp * lp + lu * lu + ld * ld + mb * mb + mf * mf + li * li + ml * ml + ms * ms)
    if mv < _DET_F_EPS:
        # Pure rotation: zero strain, nothing to apply. Persist the
        # rotation (already updated by the polar decomposition) and bail.
        write_quat(constraints, _OFF_ROTATION, cid, rotation)
        return

    my = (wp.float32(1.0) / (wp.float32(2.0) * mv)) * rest_volume

    # Analytic gradients per vertex per coordinate (grad2_1..grad2_12 in
    # Jitter2). Each is a partial derivative of ||F-R||_F w.r.t. one
    # particle coordinate, scaled by my.
    g1_x = -wp.float32(2.0) * (kz * cs + ld * cg + li * cm) * my
    g1_y = -wp.float32(2.0) * (lp * cs + mb * cg + ml * cm) * my
    g1_z = -wp.float32(2.0) * (lu * cs + mf * cg + ms * cm) * my
    g2_x = wp.float32(2.0) * (kz * inv_rest[0, 0] + ld * inv_rest[0, 1] + li * inv_rest[0, 2]) * my
    g2_y = wp.float32(2.0) * (lp * inv_rest[0, 0] + mb * inv_rest[0, 1] + ml * inv_rest[0, 2]) * my
    g2_z = wp.float32(2.0) * (lu * inv_rest[0, 0] + mf * inv_rest[0, 1] + ms * inv_rest[0, 2]) * my
    g3_x = wp.float32(2.0) * (kz * inv_rest[1, 0] + ld * inv_rest[1, 1] + li * inv_rest[1, 2]) * my
    g3_y = wp.float32(2.0) * (lp * inv_rest[1, 0] + mb * inv_rest[1, 1] + ml * inv_rest[1, 2]) * my
    g3_z = wp.float32(2.0) * (lu * inv_rest[1, 0] + mf * inv_rest[1, 1] + ms * inv_rest[1, 2]) * my
    g4_x = wp.float32(2.0) * (kz * inv_rest[2, 0] + ld * inv_rest[2, 1] + li * inv_rest[2, 2]) * my
    g4_y = wp.float32(2.0) * (lp * inv_rest[2, 0] + mb * inv_rest[2, 1] + ml * inv_rest[2, 2]) * my
    g4_z = wp.float32(2.0) * (lu * inv_rest[2, 0] + mf * inv_rest[2, 1] + ms * inv_rest[2, 2]) * my

    idt_sq = idt * idt
    bias_mu = idt_sq * alpha_mu

    denom = (
        inv_mass_a * (g1_x * g1_x + g1_y * g1_y + g1_z * g1_z)
        + inv_mass_b * (g2_x * g2_x + g2_y * g2_y + g2_z * g2_z)
        + inv_mass_c * (g3_x * g3_x + g3_y * g3_y + g3_z * g3_z)
        + inv_mass_d * (g4_x * g4_x + g4_y * g4_y + g4_z * g4_z)
        + bias_mu
    )

    if denom > wp.float32(0.0):
        c_mu = rest_volume * mv
        d_lam_mu = -(c_mu + bias_mu * lambda_sum_mu) / denom
        x_a = x_a + wp.vec3f(g1_x * d_lam_mu * inv_mass_a, g1_y * d_lam_mu * inv_mass_a, g1_z * d_lam_mu * inv_mass_a)
        x_b = x_b + wp.vec3f(g2_x * d_lam_mu * inv_mass_b, g2_y * d_lam_mu * inv_mass_b, g2_z * d_lam_mu * inv_mass_b)
        x_c = x_c + wp.vec3f(g3_x * d_lam_mu * inv_mass_c, g3_y * d_lam_mu * inv_mass_c, g3_z * d_lam_mu * inv_mass_c)
        x_d = x_d + wp.vec3f(g4_x * d_lam_mu * inv_mass_d, g4_y * d_lam_mu * inv_mass_d, g4_z * d_lam_mu * inv_mass_d)
        lambda_sum_mu = lambda_sum_mu + d_lam_mu

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
