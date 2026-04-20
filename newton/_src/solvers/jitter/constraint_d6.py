# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""6-DoF generalised ("D6") joint constraint.

Pure port of Jolt's ``SixDOFConstraint``
(``Jolt/Physics/Constraints/SixDOFConstraint.cpp``) and its
sub-parts. Where the previous implementation tried to solve the joint
as a single fused 6x6 Schur complement, this version follows Jolt
verbatim and dispatches each block of axes to the smallest possible
constraint primitive:

* **Translation block**

  - all 3 axes locked + no targets + no force caps -> a fused
    :class:`PointConstraintPart` (3-DoF ball-socket; the standard
    3x3 effective-mass solve).
  - any other configuration (free axes, motors, force caps) -> three
    independent :class:`AxisConstraintPart` 1-DoF rows, one per
    body-1-local linear axis.

* **Rotation block**

  - all 3 axes locked + no targets + no torque caps -> a fused
    :class:`RotationEulerConstraintPart` (3-DoF rotation lock with
    quaternion-error linearisation).
  - any other configuration -> three independent
    :class:`AngleConstraintPart` 1-DoF rows, one per body-1-local
    angular axis.

The rationale is the same one Jolt gives in the file header: trying
to fuse a partial-locked rotation block with a partial-locked
translation block into one 6x6 system poisons the per-axis impulse
clamp on free axes (the Schur back-substitution distributes a free
axis's "missing" impulse over the other axes, which then over-shoot
on the next iteration). Jolt sidesteps this entirely by solving each
1-DoF row independently.

Per-axis mode (``D6AxisDrive`` -> sub-part)
-------------------------------------------
Each of the 6 axes (3 angular + 3 linear) is independently
configurable as:

* **Rigid lock**     -- ``hertz <= 0`` and ``max_force = +inf``. The
  axis is welded.
* **Soft lock**      -- ``hertz > 0``, no targets. Spring + damper
  restoring the rest pose with the soft-constraint formulation in
  :class:`SpringPart` (Box2D / Bepu / Nordby implicit-Euler scheme).
* **Position drive** -- ``hertz > 0``, ``target_position != 0``,
  ``max_force`` finite. Implicit-PD drive that pulls the axis toward
  the target with the soft-spring stiffness, capped at
  ``max_force``.
* **Velocity drive** -- ``hertz > 0``, ``target_velocity != 0``,
  ``max_force`` finite. Implicit-PD velocity-tracking drive ("motor"
  mode); ``hertz`` modulates how aggressively the impulse converges
  to the velocity setpoint.
* **Free**           -- ``max_force = 0``. The per-axis row is
  *deactivated* (``effective_mass = 0``); no impulse is computed
  or applied for that axis. This is the canonical way to leave an
  axis unconstrained without changing the matrix shape.

Position + velocity targets compose, yielding an over-damped spring
with a steady-state velocity (e.g. "open the door at 1 rad/s while
pulling it towards 90°").

Constraint frame ("body-1 local")
---------------------------------
The 6 axes are interpreted in *body 1's local frame*: the joint
frame "rides" body 1 rigidly. The world-frame translation axes are
recomputed each prepare as ``e_k = q1 * e_k_local``; the world-frame
rotation axes for the per-axis :class:`AngleConstraintPart` rows are
the same axes (Jolt also uses body 1's frame for rotation axes when
no swing-twist limits are active). Position targets are folded into
the rest pose / rest anchor at :func:`d6_initialize_kernel` time so
the runtime kernel doesn't branch on "is there a position target".

Mapping summary (Jolt -> this file):
* ``PointConstraintPart``           -> :func:`_point_*` family
* ``RotationEulerConstraintPart``   -> :func:`_euler_*` family
* ``AxisConstraintPart``            -> :func:`_axis_*` family
* ``AngleConstraintPart``           -> :func:`_angle_*` family
* ``SpringPart``                    -> :func:`soft_constraint_coefficients`
                                       in :mod:`constraint_container`
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_D6,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    read_int,
    read_mat33,
    read_quat,
    read_vec3,
    write_int,
    write_mat33,
    write_quat,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords

__all__ = [
    "D6_DWORDS",
    "D6Data",
    "d6_initialize_kernel",
    "d6_iterate",
    "d6_iterate_at",
    "d6_prepare_for_iteration",
    "d6_prepare_for_iteration_at",
    "d6_world_wrench",
    "d6_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
#
# Layout strategy: a single column packs *both* the fused mode (Point /
# Euler) state *and* the per-axis mode (Axis x3 / Angle x3) state. The
# ``trans_mode`` / ``rot_mode`` flags chosen at init time decide which
# half of each block the iterate path actually uses; the unused half
# stays at zero (effective_mass = 0 = "deactivated", same as Jolt's
# ``Deactivate()``).
#
# Field naming follows Jolt's member naming (``mInvI1_R1X``,
# ``mTotalLambda``, etc.) with the appropriate snake_case translation.


@wp.struct
class D6Data:
    """Per-constraint dword-layout schema for a D6 (6-DoF) joint.

    *Schema only*: the struct is never instantiated at runtime; we
    derive its dword-offset table once at module load and the runtime
    kernels read/write fields via the column-major
    :class:`ConstraintContainer`.

    The first three fields are the global constraint header
    (``constraint_type``, ``body1``, ``body2``).
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # ---- Init-time constant state ------------------------------------
    # Anchor in each body's local frame, snapshotted at init from a
    # shared world-space anchor with the linear position targets folded
    # into ``local_anchor_b1`` along the constraint frame so the
    # runtime "rest pose" already includes the user-specified offset.
    local_anchor_b1: wp.vec3f
    local_anchor_b2: wp.vec3f

    # Constraint-to-body rotations. ``q_const_to_b1`` and
    # ``q_const_to_b2`` are the quaternions that take a vector from
    # the constraint frame to body 1 / body 2 local space. At init
    # time both are identity (the constraint frame coincides with
    # body 1's local frame); a future extension can let the user
    # specify an oriented constraint frame relative to either body
    # without touching the runtime path.
    q_const_to_b1: wp.quatf
    q_const_to_b2: wp.quatf

    # Inverse of the initial relative orientation (body 1 to body 2
    # in body 1 space), with the angular position targets folded in.
    # Used by the ``Euler`` (full-locked rotation) path:
    #   diff = q2 * q_inv_initial * q1^*
    # is the rotational error quaternion that the rotation-lock part
    # drives to identity. See Jolt's
    # ``RotationEulerConstraintPart::SolvePositionConstraint``.
    q_inv_initial_orientation: wp.quatf

    # Per-axis user knobs. ``hertz_*[k] <= 0`` -> rigid for axis ``k``;
    # ``max_force_*[k] = 0`` -> axis is *free* (per-axis row gets
    # deactivated at prepare time). ``+inf`` is mapped to ``1e30`` by
    # the host packer so the multiplication ``max_force * dt`` stays
    # a well-defined float; the iterate path reads it as "no cap".
    hertz_ang: wp.vec3f
    damping_ratio_ang: wp.vec3f
    hertz_lin: wp.vec3f
    damping_ratio_lin: wp.vec3f

    target_velocity_ang: wp.vec3f
    target_velocity_lin: wp.vec3f

    max_force_ang: wp.vec3f
    max_force_lin: wp.vec3f

    # ---- Dispatch flags (set at init from the per-axis knobs) -------
    # ``trans_mode == 0``  -> Point fused 3-DoF lock (all 3 lin
    #                         axes are rigid + no targets + no caps).
    # ``trans_mode == 1``  -> three independent Axis 1-DoF rows.
    #
    # ``rot_mode == 0``    -> Euler fused 3-DoF lock.
    # ``rot_mode == 1``    -> three independent Angle 1-DoF rows.
    #
    # Stored as wp.int32 so the schema layout is portable; only the
    # low bit ever matters.
    trans_mode: wp.int32
    rot_mode: wp.int32

    # ---- Per-substep cached state (recomputed every prepare) --------
    # World-space lever arms (anchor minus body COM) for each body.
    r1: wp.vec3f
    r2: wp.vec3f

    # World-space constraint axes. Rows of ``axes_world`` are the
    # three world-frame axes (``e_k_world = q1 * q_const_to_b1
    # * e_k_const``); rows are aligned with the per-axis fields below.
    axes_world: wp.mat33f

    # Per-substep impulse caps (``max_force * dt``) per axis. The
    # iterate path reads these directly to clamp Lagrange multipliers
    # in the per-axis Axis / Angle paths. Matches Jolt's pattern of
    # passing ``inDeltaTime * inMaxForceLimit`` as the per-iter clamp.
    max_lambda_ang: wp.vec3f
    max_lambda_lin: wp.vec3f

    # ---- Point part state (used iff trans_mode == 0) -----------------
    # ``mInvI1_R1X = InvI1 @ skew(r1)``, ``mInvI2_R2X = InvI2 @ skew(r2)``.
    # ``effective_mass = (m1^-1 I + skew(r1) InvI1 skew(r1)^T + ...) ^-1``
    # See Jolt ``PointConstraintPart::CalculateConstraintProperties``.
    point_inv_i1_r1x: wp.mat33f
    point_inv_i2_r2x: wp.mat33f
    point_effective_mass: wp.mat33f

    # Accumulated impulse for the point part (world-frame 3-vec).
    point_total_lambda: wp.vec3f

    # ---- Euler part state (used iff rot_mode == 0) -------------------
    # ``effective_mass = (InvI1 + InvI2)^-1`` (3x3, world frame).
    # Matches Jolt ``RotationEulerConstraintPart::CalculateConstraintProperties``.
    euler_effective_mass: wp.mat33f

    # Accumulated angular impulse for the euler part (world-frame
    # angular impulse 3-vec). Body-frame "torque" basis -- the Jacobian
    # for the euler part is just the identity.
    euler_total_lambda: wp.vec3f

    # ---- Axis parts state (used iff trans_mode == 1) -----------------
    # Three independent 1-DoF translation rows. The k-th row uses
    # ``axes_world[k]`` as its world-space axis; per-axis cached
    # quantities sit in three vec3f columns (componentwise across the
    # three rows).
    #
    # axis_r1_plus_u_x_axis[k] = (r1 + u) x axes_world[k]
    # axis_r2_x_axis[k]        = r2 x axes_world[k]
    # axis_inv_i1_r1pu_x[k]    = InvI1 @ axis_r1_plus_u_x_axis[k]
    # axis_inv_i2_r2_x[k]      = InvI2 @ axis_r2_x_axis[k]
    axis_r1_plus_u_x_axis_0: wp.vec3f
    axis_r1_plus_u_x_axis_1: wp.vec3f
    axis_r1_plus_u_x_axis_2: wp.vec3f
    axis_r2_x_axis_0: wp.vec3f
    axis_r2_x_axis_1: wp.vec3f
    axis_r2_x_axis_2: wp.vec3f
    axis_inv_i1_r1pu_x_0: wp.vec3f
    axis_inv_i1_r1pu_x_1: wp.vec3f
    axis_inv_i1_r1pu_x_2: wp.vec3f
    axis_inv_i2_r2_x_0: wp.vec3f
    axis_inv_i2_r2_x_1: wp.vec3f
    axis_inv_i2_r2_x_2: wp.vec3f

    # Per-axis effective mass (= 0 if axis deactivated -- "free" axis,
    # or zero summed inverse mass).
    axis_effective_mass: wp.vec3f

    # Per-axis spring state. ``spring_softness`` is gamma in Catto's
    # GDC2011 formulation; ``spring_bias`` is the constant bias term
    # ``b`` for the row's velocity constraint. For rigid axes both
    # are zero (plain PGS update). See :func:`_calc_spring_props`.
    axis_spring_softness: wp.vec3f
    axis_spring_bias: wp.vec3f

    # Per-axis accumulated impulse.
    axis_total_lambda: wp.vec3f

    # ---- Angle parts state (used iff rot_mode == 1) ------------------
    # Three independent 1-DoF rotation rows. Same layout as the axis
    # parts but no translation cross-term:
    #   angle_inv_i1_axis[k] = InvI1 @ axes_world[k]
    #   angle_inv_i2_axis[k] = InvI2 @ axes_world[k]
    angle_inv_i1_axis_0: wp.vec3f
    angle_inv_i1_axis_1: wp.vec3f
    angle_inv_i1_axis_2: wp.vec3f
    angle_inv_i2_axis_0: wp.vec3f
    angle_inv_i2_axis_1: wp.vec3f
    angle_inv_i2_axis_2: wp.vec3f
    angle_effective_mass: wp.vec3f
    angle_spring_softness: wp.vec3f
    angle_spring_bias: wp.vec3f
    angle_total_lambda: wp.vec3f


assert_constraint_header(D6Data)


# Dword offsets derived once from the schema. Each is wrapped in
# ``wp.constant`` so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(D6Data, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(D6Data, "body2"))
_OFF_LA_B1 = wp.constant(dword_offset_of(D6Data, "local_anchor_b1"))
_OFF_LA_B2 = wp.constant(dword_offset_of(D6Data, "local_anchor_b2"))
_OFF_Q_CONST_TO_B1 = wp.constant(dword_offset_of(D6Data, "q_const_to_b1"))
_OFF_Q_CONST_TO_B2 = wp.constant(dword_offset_of(D6Data, "q_const_to_b2"))
_OFF_Q_INV_INIT = wp.constant(dword_offset_of(D6Data, "q_inv_initial_orientation"))
_OFF_HERTZ_ANG = wp.constant(dword_offset_of(D6Data, "hertz_ang"))
_OFF_DAMPING_ANG = wp.constant(dword_offset_of(D6Data, "damping_ratio_ang"))
_OFF_HERTZ_LIN = wp.constant(dword_offset_of(D6Data, "hertz_lin"))
_OFF_DAMPING_LIN = wp.constant(dword_offset_of(D6Data, "damping_ratio_lin"))
_OFF_TARGET_VEL_ANG = wp.constant(dword_offset_of(D6Data, "target_velocity_ang"))
_OFF_TARGET_VEL_LIN = wp.constant(dword_offset_of(D6Data, "target_velocity_lin"))
_OFF_MAX_FORCE_ANG = wp.constant(dword_offset_of(D6Data, "max_force_ang"))
_OFF_MAX_FORCE_LIN = wp.constant(dword_offset_of(D6Data, "max_force_lin"))
_OFF_TRANS_MODE = wp.constant(dword_offset_of(D6Data, "trans_mode"))
_OFF_ROT_MODE = wp.constant(dword_offset_of(D6Data, "rot_mode"))
_OFF_R1 = wp.constant(dword_offset_of(D6Data, "r1"))
_OFF_R2 = wp.constant(dword_offset_of(D6Data, "r2"))
_OFF_AXES_WORLD = wp.constant(dword_offset_of(D6Data, "axes_world"))
_OFF_MAX_LAMBDA_ANG = wp.constant(dword_offset_of(D6Data, "max_lambda_ang"))
_OFF_MAX_LAMBDA_LIN = wp.constant(dword_offset_of(D6Data, "max_lambda_lin"))

_OFF_POINT_INV_I1_R1X = wp.constant(dword_offset_of(D6Data, "point_inv_i1_r1x"))
_OFF_POINT_INV_I2_R2X = wp.constant(dword_offset_of(D6Data, "point_inv_i2_r2x"))
_OFF_POINT_EFF_MASS = wp.constant(dword_offset_of(D6Data, "point_effective_mass"))
_OFF_POINT_TOTAL_LAMBDA = wp.constant(dword_offset_of(D6Data, "point_total_lambda"))

_OFF_EULER_EFF_MASS = wp.constant(dword_offset_of(D6Data, "euler_effective_mass"))
_OFF_EULER_TOTAL_LAMBDA = wp.constant(dword_offset_of(D6Data, "euler_total_lambda"))

_OFF_AXIS_R1PU_X_0 = wp.constant(dword_offset_of(D6Data, "axis_r1_plus_u_x_axis_0"))
_OFF_AXIS_R1PU_X_1 = wp.constant(dword_offset_of(D6Data, "axis_r1_plus_u_x_axis_1"))
_OFF_AXIS_R1PU_X_2 = wp.constant(dword_offset_of(D6Data, "axis_r1_plus_u_x_axis_2"))
_OFF_AXIS_R2_X_0 = wp.constant(dword_offset_of(D6Data, "axis_r2_x_axis_0"))
_OFF_AXIS_R2_X_1 = wp.constant(dword_offset_of(D6Data, "axis_r2_x_axis_1"))
_OFF_AXIS_R2_X_2 = wp.constant(dword_offset_of(D6Data, "axis_r2_x_axis_2"))
_OFF_AXIS_INV_I1_R1PU_X_0 = wp.constant(dword_offset_of(D6Data, "axis_inv_i1_r1pu_x_0"))
_OFF_AXIS_INV_I1_R1PU_X_1 = wp.constant(dword_offset_of(D6Data, "axis_inv_i1_r1pu_x_1"))
_OFF_AXIS_INV_I1_R1PU_X_2 = wp.constant(dword_offset_of(D6Data, "axis_inv_i1_r1pu_x_2"))
_OFF_AXIS_INV_I2_R2_X_0 = wp.constant(dword_offset_of(D6Data, "axis_inv_i2_r2_x_0"))
_OFF_AXIS_INV_I2_R2_X_1 = wp.constant(dword_offset_of(D6Data, "axis_inv_i2_r2_x_1"))
_OFF_AXIS_INV_I2_R2_X_2 = wp.constant(dword_offset_of(D6Data, "axis_inv_i2_r2_x_2"))
_OFF_AXIS_EFF_MASS = wp.constant(dword_offset_of(D6Data, "axis_effective_mass"))
_OFF_AXIS_SOFTNESS = wp.constant(dword_offset_of(D6Data, "axis_spring_softness"))
_OFF_AXIS_BIAS = wp.constant(dword_offset_of(D6Data, "axis_spring_bias"))
_OFF_AXIS_TOTAL_LAMBDA = wp.constant(dword_offset_of(D6Data, "axis_total_lambda"))

_OFF_ANGLE_INV_I1_AXIS_0 = wp.constant(dword_offset_of(D6Data, "angle_inv_i1_axis_0"))
_OFF_ANGLE_INV_I1_AXIS_1 = wp.constant(dword_offset_of(D6Data, "angle_inv_i1_axis_1"))
_OFF_ANGLE_INV_I1_AXIS_2 = wp.constant(dword_offset_of(D6Data, "angle_inv_i1_axis_2"))
_OFF_ANGLE_INV_I2_AXIS_0 = wp.constant(dword_offset_of(D6Data, "angle_inv_i2_axis_0"))
_OFF_ANGLE_INV_I2_AXIS_1 = wp.constant(dword_offset_of(D6Data, "angle_inv_i2_axis_1"))
_OFF_ANGLE_INV_I2_AXIS_2 = wp.constant(dword_offset_of(D6Data, "angle_inv_i2_axis_2"))
_OFF_ANGLE_EFF_MASS = wp.constant(dword_offset_of(D6Data, "angle_effective_mass"))
_OFF_ANGLE_SOFTNESS = wp.constant(dword_offset_of(D6Data, "angle_spring_softness"))
_OFF_ANGLE_BIAS = wp.constant(dword_offset_of(D6Data, "angle_spring_bias"))
_OFF_ANGLE_TOTAL_LAMBDA = wp.constant(dword_offset_of(D6Data, "angle_total_lambda"))

# Trans / rot mode encoding (kept in sync with init kernel).
_TRANS_MODE_POINT = wp.constant(0)
_TRANS_MODE_AXIS = wp.constant(1)
_ROT_MODE_EULER = wp.constant(0)
_ROT_MODE_ANGLE = wp.constant(1)

#: Total dword count of one D6 constraint. Used by the host-side
#: container allocator to size ``ConstraintContainer.data``'s row count.
D6_DWORDS: int = num_dwords(D6Data)


# ---------------------------------------------------------------------------
# Spring helpers (port of Jolt SpringPart)
# ---------------------------------------------------------------------------
#
# Jolt's SpringPart computes the per-row (softness, bias) pair from
# the spring settings. We use the equivalent Box2D / Bepu helper
# already in the codebase (:func:`soft_constraint_coefficients`),
# but expose it through a thin wrapper that returns the Jolt-style
# (softness, bias_with_spring) pair on top of a *separate* constant
# bias (used for velocity-driven motors where the bias is just
# ``-target_velocity``, not a position error).


@wp.func
def _calc_spring_props(
    inv_eff_mass: wp.float32,
    bias_constant: wp.float32,
    position_error: wp.float32,
    hertz: wp.float32,
    damping_ratio: wp.float32,
    dt: wp.float32,
) -> wp.vec3f:
    """Return ``(effective_mass, softness, bias)`` for a 1-DoF row.

    Mirrors Jolt's ``SpringPart::CalculateSpringPropertiesWithFrequencyAndDamping``
    but reuses our Box2D-formulated soft-constraint coefficients to
    keep one source of truth for the implicit-Euler spring.

    For a *rigid* row (``hertz <= 0``), the softness is 0 and the
    bias is just ``bias_constant``; the effective mass is
    ``1 / inv_eff_mass``. For a *soft* row, the softness is
    ``1 / (dt * (c + dt * k))`` from Catto's GDC11 formulation, the
    bias gets an extra ``dt * k * softness * position_error`` term,
    and the effective mass becomes ``1 / (inv_eff_mass + softness)``.

    Returned as a vec3 for kernel-friendly packing
    (warp doesn't have multi-return tuples in functions cleanly).
    """
    if inv_eff_mass <= 0.0:
        # Either both bodies are static along this axis, or the axis
        # has zero summed inverse mass for some other reason. Mark
        # the row deactivated.
        return wp.vec3f(0.0, 0.0, 0.0)

    if hertz <= 0.0:
        # Plain rigid row.
        return wp.vec3f(1.0 / inv_eff_mass, 0.0, bias_constant)

    # Soft row: derive (k, c) from (frequency, damping_ratio) the
    # Jolt way (k = m * omega^2, c = 2 * m * zeta * omega), then
    # plug into Catto's softness formula. Equivalent to our
    # ``soft_constraint_coefficients`` but with the bias_constant
    # term added in.
    # Compute a placeholder effective mass for the (k, c) recipe.
    eff_mass_unsoft = 1.0 / inv_eff_mass
    omega = 2.0 * 3.141592653589793 * hertz
    k = eff_mass_unsoft * omega * omega
    c = 2.0 * eff_mass_unsoft * damping_ratio * omega

    softness = 1.0 / (dt * (c + dt * k))
    bias = bias_constant + dt * k * softness * position_error
    eff_mass = 1.0 / (inv_eff_mass + softness)
    return wp.vec3f(eff_mass, softness, bias)


# ---------------------------------------------------------------------------
# PointConstraintPart helpers
# ---------------------------------------------------------------------------


@wp.func
def _point_calculate_properties(
    r1: wp.vec3f,
    r2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
):
    """Direct port of ``PointConstraintPart::CalculateConstraintProperties``.

    Returns ``(effective_mass, inv_i1_r1x, inv_i2_r2x)``. The latter
    two cache ``InvI * skew(r)`` for both bodies so that the velocity
    /impulse paths can apply them directly without recomputing the
    skew matrix.
    """
    eye3 = wp.identity(3, dtype=wp.float32)
    r1x = wp.skew(r1)
    r2x = wp.skew(r2)

    inv_i1_r1x = inv_inertia1 @ r1x
    inv_i2_r2x = inv_inertia2 @ r2x

    # K = sum_b ( m_b^-1 I + skew(r_b) Inv_I_b skew(r_b)^T )
    # Note: skew(r)^T = -skew(r), so skew(r) Inv_I skew(r)^T = -skew(r) Inv_I skew(r).
    inv_eff_mass = (inv_mass1 + inv_mass2) * eye3
    inv_eff_mass = inv_eff_mass + r1x @ (inv_inertia1 @ wp.transpose(r1x))
    inv_eff_mass = inv_eff_mass + r2x @ (inv_inertia2 @ wp.transpose(r2x))

    eff_mass = wp.inverse(inv_eff_mass)
    return eff_mass, inv_i1_r1x, inv_i2_r2x


@wp.func
def _point_apply_velocity_step(
    bodies: BodyContainer,
    b1: wp.int32,
    b2: wp.int32,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_i1_r1x: wp.mat33f,
    inv_i2_r2x: wp.mat33f,
    lam: wp.vec3f,
):
    """Apply ``lambda`` (world-frame linear impulse) to both bodies.

    Mirrors ``PointConstraintPart::ApplyVelocityStep``:
        v1' -= m1^-1 * lambda
        w1' -= InvI1 * skew(r1) * lambda
        v2' += m2^-1 * lambda
        w2' += InvI2 * skew(r2) * lambda
    """
    bodies.velocity[b1] = bodies.velocity[b1] - inv_mass1 * lam
    bodies.angular_velocity[b1] = bodies.angular_velocity[b1] - inv_i1_r1x @ lam
    bodies.velocity[b2] = bodies.velocity[b2] + inv_mass2 * lam
    bodies.angular_velocity[b2] = bodies.angular_velocity[b2] + inv_i2_r2x @ lam


# ---------------------------------------------------------------------------
# RotationEulerConstraintPart helpers
# ---------------------------------------------------------------------------


@wp.func
def _euler_calculate_properties(
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
) -> wp.mat33f:
    """Direct port of ``RotationEulerConstraintPart::CalculateConstraintProperties``.

    K = InvI1 + InvI2 (both world-frame); effective mass = K^-1.
    Returns the 3x3 effective mass.
    """
    inv_eff_mass = inv_inertia1 + inv_inertia2
    return wp.inverse(inv_eff_mass)


@wp.func
def _euler_apply_velocity_step(
    bodies: BodyContainer,
    b1: wp.int32,
    b2: wp.int32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    lam: wp.vec3f,
):
    """Apply ``lambda`` (world-frame angular impulse) to both bodies.

    Mirrors ``RotationEulerConstraintPart::ApplyVelocityStep``:
        w1' -= InvI1 * lambda
        w2' += InvI2 * lambda
    """
    bodies.angular_velocity[b1] = bodies.angular_velocity[b1] - inv_inertia1 @ lam
    bodies.angular_velocity[b2] = bodies.angular_velocity[b2] + inv_inertia2 @ lam


# ---------------------------------------------------------------------------
# AxisConstraintPart helpers (1-DoF translation row)
# ---------------------------------------------------------------------------


@wp.func
def _axis_calculate_inv_eff_mass(
    r1_plus_u: wp.vec3f,
    r2: wp.vec3f,
    axis: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
):
    """Port of ``AxisConstraintPart::CalculateInverseEffectiveMass``.

    Returns ``(inv_eff_mass, r1_plus_u_x_axis, r2_x_axis,
    inv_i1_r1pu_x_axis, inv_i2_r2_x_axis)``.

    K = m1^-1 + (r1+u x axis).(InvI1 (r1+u x axis))
        + m2^-1 + (r2 x axis).(InvI2 (r2 x axis))
    """
    r1pu_x = wp.cross(r1_plus_u, axis)
    r2_x = wp.cross(r2, axis)
    inv_i1_r1pu_x = inv_inertia1 @ r1pu_x
    inv_i2_r2_x = inv_inertia2 @ r2_x
    inv_eff_mass = (
        inv_mass1
        + inv_mass2
        + wp.dot(r1pu_x, inv_i1_r1pu_x)
        + wp.dot(r2_x, inv_i2_r2_x)
    )
    return inv_eff_mass, r1pu_x, r2_x, inv_i1_r1pu_x, inv_i2_r2_x


@wp.func
def _axis_apply_velocity_step(
    bodies: BodyContainer,
    b1: wp.int32,
    b2: wp.int32,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    axis: wp.vec3f,
    inv_i1_r1pu_x: wp.vec3f,
    inv_i2_r2_x: wp.vec3f,
    lam: wp.float32,
):
    """Apply scalar ``lambda`` to both bodies along ``axis``.

    Mirrors ``AxisConstraintPart::ApplyVelocityStep``:
        v1' -= lam * m1^-1 * axis
        w1' -= lam * InvI1 * (r1+u x axis)
        v2' += lam * m2^-1 * axis
        w2' += lam * InvI2 * (r2 x axis)
    """
    bodies.velocity[b1] = bodies.velocity[b1] - (lam * inv_mass1) * axis
    bodies.angular_velocity[b1] = bodies.angular_velocity[b1] - lam * inv_i1_r1pu_x
    bodies.velocity[b2] = bodies.velocity[b2] + (lam * inv_mass2) * axis
    bodies.angular_velocity[b2] = bodies.angular_velocity[b2] + lam * inv_i2_r2_x


# ---------------------------------------------------------------------------
# AngleConstraintPart helpers (1-DoF rotation row)
# ---------------------------------------------------------------------------


@wp.func
def _angle_calculate_inv_eff_mass(
    axis: wp.vec3f,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
):
    """Port of ``AngleConstraintPart::CalculateInverseEffectiveMass``.

    Returns ``(inv_eff_mass, inv_i1_axis, inv_i2_axis)``.

    K = axis.(InvI1 axis) + axis.(InvI2 axis)
    """
    inv_i1_axis = inv_inertia1 @ axis
    inv_i2_axis = inv_inertia2 @ axis
    inv_eff_mass = wp.dot(axis, inv_i1_axis) + wp.dot(axis, inv_i2_axis)
    return inv_eff_mass, inv_i1_axis, inv_i2_axis


@wp.func
def _angle_apply_velocity_step(
    bodies: BodyContainer,
    b1: wp.int32,
    b2: wp.int32,
    inv_i1_axis: wp.vec3f,
    inv_i2_axis: wp.vec3f,
    lam: wp.float32,
):
    """Apply scalar ``lambda`` to both bodies as an angular impulse.

    Mirrors ``AngleConstraintPart::ApplyVelocityStep``:
        w1' -= lam * InvI1 * axis
        w2' += lam * InvI2 * axis
    """
    bodies.angular_velocity[b1] = bodies.angular_velocity[b1] - lam * inv_i1_axis
    bodies.angular_velocity[b2] = bodies.angular_velocity[b2] + lam * inv_i2_axis


# ---------------------------------------------------------------------------
# Initialization (kernel)
# ---------------------------------------------------------------------------


@wp.func
def _is_axis_rigid(hertz: wp.float32, target_pos: wp.float32, max_force: wp.float32) -> bool:
    """An axis is "rigid" iff hertz<=0 and no position target and no force cap.

    These are the conditions Jolt's ``SixDOFConstraint`` uses to
    decide between the fused (PointConstraintPart /
    RotationEulerConstraintPart) and the per-axis dispatch
    (AxisConstraintPart / AngleConstraintPart) paths.

    "No force cap" means ``max_force >= 1e30`` -- the host packer
    encodes ``+inf`` as ``1e30`` to keep ``max_force * dt`` well-
    defined, and any axis with a smaller cap needs the per-axis path
    so the per-iter clamp can saturate.
    """
    return hertz <= 0.0 and target_pos == 0.0 and max_force >= 1.0e29


@wp.kernel
def d6_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor: wp.array[wp.vec3f],
    target_position_ang: wp.array[wp.vec3f],
    target_position_lin: wp.array[wp.vec3f],
    target_velocity_ang: wp.array[wp.vec3f],
    target_velocity_lin: wp.array[wp.vec3f],
    hertz_ang: wp.array[wp.vec3f],
    damping_ratio_ang: wp.array[wp.vec3f],
    hertz_lin: wp.array[wp.vec3f],
    damping_ratio_lin: wp.array[wp.vec3f],
    max_force_ang: wp.array[wp.vec3f],
    max_force_lin: wp.array[wp.vec3f],
):
    """Pack one batch of D6 descriptors into ``constraints``.

    Snapshots the user's world-space anchor + per-axis target poses
    into the rest configuration so the runtime kernel sees a single
    (rest pose, current pose) error rather than separate "lock" and
    "drive" terms. Also classifies each block (translation /
    rotation) into one of the two dispatch modes (fused vs per-axis)
    based on the per-axis knobs, and writes the resulting
    ``trans_mode`` / ``rot_mode`` flags into the column.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; reads ``position`` /
            ``orientation`` to snapshot the per-body local anchors.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor: World-space anchor point [num_in_batch] [m].
        target_position_ang: Per-axis angular position targets
            [num_in_batch] [rad], interpreted as XYZ Euler in body 1's
            constraint frame.
        target_position_lin: Per-axis linear position targets
            [num_in_batch] [m], interpreted in body 1's constraint
            frame.
        target_velocity_ang: Per-axis angular velocity targets
            [num_in_batch] [rad/s].
        target_velocity_lin: Per-axis linear velocity targets
            [num_in_batch] [m/s].
        hertz_ang: Per-axis angular soft-constraint stiffness
            [num_in_batch] [Hz]. ``hertz <= 0`` -> rigid axis.
        damping_ratio_ang: Per-axis angular damping ratios
            [num_in_batch] (1 = critical).
        hertz_lin: Per-axis linear stiffness [num_in_batch] [Hz].
        damping_ratio_lin: Per-axis linear damping ratios
            [num_in_batch].
        max_force_ang: Per-axis angular force cap [num_in_batch]
            [N*m]. ``0`` -> axis is free; ``+inf`` (passed as ``1e30``
            by the host packer) -> "rigid / no cap".
        max_force_lin: Per-axis linear force cap [num_in_batch] [N].
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a_world = anchor[tid]
    tp_ang = target_position_ang[tid]
    tp_lin = target_position_lin[tid]
    h_ang = hertz_ang[tid]
    h_lin = hertz_lin[tid]
    mf_ang = max_force_ang[tid]
    mf_lin = max_force_lin[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    # Body-1 local anchor with the linear position target folded in
    # (along body 1's constraint frame, which is body 1's local frame
    # at init time -- ``q_const_to_b1 = identity``).
    la_b1_rest = wp.quat_rotate_inv(q1, a_world - pos1)
    la_b1 = la_b1_rest + tp_lin
    la_b2 = wp.quat_rotate_inv(q2, a_world - pos2)

    # Build the angular target rotation in body 1's constraint frame
    # as an XYZ Euler composition (closed-form half-angle quaternion
    # product, branchless).
    half_x = tp_ang[0] * 0.5
    half_y = tp_ang[1] * 0.5
    half_z = tp_ang[2] * 0.5
    qx = wp.quatf(wp.sin(half_x), 0.0, 0.0, wp.cos(half_x))
    qy = wp.quatf(0.0, wp.sin(half_y), 0.0, wp.cos(half_y))
    qz = wp.quatf(0.0, 0.0, wp.sin(half_z), wp.cos(half_z))
    q_target = qx * qy * qz

    # Constraint-frame rotations: identity = constraint frame
    # coincides with body's local frame. A future extension can let
    # the user reorient the joint frame relative to either body
    # without touching the runtime path.
    q_id = wp.quatf(0.0, 0.0, 0.0, 1.0)

    # Inverse of initial relative orientation, with angular target
    # folded in. Defined so the runtime ``diff = q2 * q_inv_initial *
    # q1^*`` is identity exactly when body 2's orientation matches
    # the user-specified target offset relative to body 1.
    #
    # Derivation (matching Jolt's
    # ``RotationEulerConstraintPart::sGetInvInitialOrientation``):
    #   target relation:       q2_target = q1 * q_target
    #   inv_initial:            r0^-1 = q2_target^-1 * q1
    #                                = (q1 * q_target)^-1 * q1
    #                                = q_target^-1 * q1^-1 * q1
    #                                = q_target^-1
    # ... but only after we adopt Jolt's convention that q1 / q2 are
    # measured *now*; with general initial rotations we need
    # ``q_inv_init = q_target^-1 * q1^-1 * q2`` so the runtime error
    # ``q2 * q_inv_init * q1^-1 = q2 * q_target^-1 * q1^-1 * q2 *
    # q1^-1`` ... Let us just pick the same convention as before:
    #   we want a rest pose ``q0`` such that the rotation error
    #   ``q0 * q1^-1 * q2`` is identity at rest. Snapshot
    #   ``q0 = q2^-1 * (q1 * q_target)``. Then for the Euler part
    #   the equivalent ``q_inv_initial`` is ``q0^-1``.
    q0 = wp.quat_inverse(q2) * (q1 * q_target)
    q_inv_initial = wp.quat_inverse(q0)

    # ---- Classification: trans_mode and rot_mode --------------------
    # Translation: PointConstraint iff every axis is rigid (hertz<=0,
    # no position/velocity targets, and "no" force cap).
    tv_ang = target_velocity_ang[tid]
    tv_lin = target_velocity_lin[tid]
    trans_rigid = (
        _is_axis_rigid(h_lin[0], tp_lin[0], mf_lin[0])
        and _is_axis_rigid(h_lin[1], tp_lin[1], mf_lin[1])
        and _is_axis_rigid(h_lin[2], tp_lin[2], mf_lin[2])
        and tv_lin[0] == 0.0
        and tv_lin[1] == 0.0
        and tv_lin[2] == 0.0
    )
    rot_rigid = (
        _is_axis_rigid(h_ang[0], tp_ang[0], mf_ang[0])
        and _is_axis_rigid(h_ang[1], tp_ang[1], mf_ang[1])
        and _is_axis_rigid(h_ang[2], tp_ang[2], mf_ang[2])
        and tv_ang[0] == 0.0
        and tv_ang[1] == 0.0
        and tv_ang[2] == 0.0
    )

    if trans_rigid:
        trans_mode_v = wp.int32(0)  # _TRANS_MODE_POINT
    else:
        trans_mode_v = wp.int32(1)  # _TRANS_MODE_AXIS

    if rot_rigid:
        rot_mode_v = wp.int32(0)  # _ROT_MODE_EULER
    else:
        rot_mode_v = wp.int32(1)  # _ROT_MODE_ANGLE

    # ---- Write the column ------------------------------------------
    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_D6)
    write_int(constraints, _OFF_BODY1, cid, b1)
    write_int(constraints, _OFF_BODY2, cid, b2)
    write_vec3(constraints, _OFF_LA_B1, cid, la_b1)
    write_vec3(constraints, _OFF_LA_B2, cid, la_b2)
    write_quat(constraints, _OFF_Q_CONST_TO_B1, cid, q_id)
    write_quat(constraints, _OFF_Q_CONST_TO_B2, cid, q_id)
    write_quat(constraints, _OFF_Q_INV_INIT, cid, q_inv_initial)

    write_vec3(constraints, _OFF_HERTZ_ANG, cid, h_ang)
    write_vec3(constraints, _OFF_DAMPING_ANG, cid, damping_ratio_ang[tid])
    write_vec3(constraints, _OFF_HERTZ_LIN, cid, h_lin)
    write_vec3(constraints, _OFF_DAMPING_LIN, cid, damping_ratio_lin[tid])
    write_vec3(constraints, _OFF_TARGET_VEL_ANG, cid, tv_ang)
    write_vec3(constraints, _OFF_TARGET_VEL_LIN, cid, tv_lin)
    write_vec3(constraints, _OFF_MAX_FORCE_ANG, cid, mf_ang)
    write_vec3(constraints, _OFF_MAX_FORCE_LIN, cid, mf_lin)
    write_int(constraints, _OFF_TRANS_MODE, cid, trans_mode_v)
    write_int(constraints, _OFF_ROT_MODE, cid, rot_mode_v)

    # Zero out per-substep cached state and accumulated impulses.
    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    zero33 = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    write_vec3(constraints, _OFF_R1, cid, zero3)
    write_vec3(constraints, _OFF_R2, cid, zero3)
    write_mat33(constraints, _OFF_AXES_WORLD, cid, wp.identity(3, dtype=wp.float32))
    write_vec3(constraints, _OFF_MAX_LAMBDA_ANG, cid, zero3)
    write_vec3(constraints, _OFF_MAX_LAMBDA_LIN, cid, zero3)

    write_mat33(constraints, _OFF_POINT_INV_I1_R1X, cid, zero33)
    write_mat33(constraints, _OFF_POINT_INV_I2_R2X, cid, zero33)
    write_mat33(constraints, _OFF_POINT_EFF_MASS, cid, zero33)
    write_vec3(constraints, _OFF_POINT_TOTAL_LAMBDA, cid, zero3)

    write_mat33(constraints, _OFF_EULER_EFF_MASS, cid, zero33)
    write_vec3(constraints, _OFF_EULER_TOTAL_LAMBDA, cid, zero3)

    write_vec3(constraints, _OFF_AXIS_R1PU_X_0, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_R1PU_X_1, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_R1PU_X_2, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_R2_X_0, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_R2_X_1, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_R2_X_2, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_INV_I1_R1PU_X_0, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_INV_I1_R1PU_X_1, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_INV_I1_R1PU_X_2, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_INV_I2_R2_X_0, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_INV_I2_R2_X_1, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_INV_I2_R2_X_2, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_EFF_MASS, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_SOFTNESS, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_BIAS, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_TOTAL_LAMBDA, cid, zero3)

    write_vec3(constraints, _OFF_ANGLE_INV_I1_AXIS_0, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_INV_I1_AXIS_1, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_INV_I1_AXIS_2, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_INV_I2_AXIS_0, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_INV_I2_AXIS_1, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_INV_I2_AXIS_2, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_EFF_MASS, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_SOFTNESS, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_BIAS, cid, zero3)
    write_vec3(constraints, _OFF_ANGLE_TOTAL_LAMBDA, cid, zero3)


# ---------------------------------------------------------------------------
# Prepare-for-iteration
# ---------------------------------------------------------------------------


@wp.func
def d6_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass for the D6 joint.

    Recomputes world-space lever arms + constraint axes and dispatches
    to the appropriate sub-part(s) based on the ``trans_mode`` and
    ``rot_mode`` flags written at init. Then warm-starts the bodies
    with the cached accumulated impulses for whichever parts are
    active.

    See :func:`ball_socket_prepare_for_iteration_at` for the
    ``base_offset`` / ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    la_b1 = read_vec3(constraints, base_offset + _OFF_LA_B1, cid)
    la_b2 = read_vec3(constraints, base_offset + _OFF_LA_B2, cid)
    q_const_to_b1 = read_quat(constraints, base_offset + _OFF_Q_CONST_TO_B1, cid)

    trans_mode_v = read_int(constraints, base_offset + _OFF_TRANS_MODE, cid)
    rot_mode_v = read_int(constraints, base_offset + _OFF_ROT_MODE, cid)

    dt = 1.0 / idt

    # World-frame lever arms.
    r1 = wp.quat_rotate(q1, la_b1)
    r2 = wp.quat_rotate(q2, la_b2)
    write_vec3(constraints, base_offset + _OFF_R1, cid, r1)
    write_vec3(constraints, base_offset + _OFF_R2, cid, r2)

    # World-frame constraint axes (rows of ``axes_world``). At init
    # the constraint frame coincides with body 1's local frame, so the
    # k-th world-axis is just ``q1 * e_k``. With a non-trivial
    # ``q_const_to_b1`` it would be ``q1 * q_const_to_b1 * e_k``.
    q_world_const = q1 * q_const_to_b1
    e0_world = wp.quat_rotate(q_world_const, wp.vec3f(1.0, 0.0, 0.0))
    e1_world = wp.quat_rotate(q_world_const, wp.vec3f(0.0, 1.0, 0.0))
    e2_world = wp.quat_rotate(q_world_const, wp.vec3f(0.0, 0.0, 1.0))
    axes_world = wp.mat33f(
        e0_world[0], e0_world[1], e0_world[2],
        e1_world[0], e1_world[1], e1_world[2],
        e2_world[0], e2_world[1], e2_world[2],
    )
    write_mat33(constraints, base_offset + _OFF_AXES_WORLD, cid, axes_world)

    # Per-substep impulse caps: Jolt clamps Lagrange multipliers in
    # ``AxisConstraintPart::SolveVelocityConstraint`` against
    # ``[-FLT_MAX, FLT_MAX]`` for hard constraints and
    # ``[dt * minForce, dt * maxForce]`` for motors. We use a
    # symmetric cap ``+/- max_force * dt``; ``max_force = 0`` (free
    # axis) means the row will be deactivated, ``max_force = 1e30``
    # (rigid) gives an effectively-infinite cap.
    max_force_ang = read_vec3(constraints, base_offset + _OFF_MAX_FORCE_ANG, cid)
    max_force_lin = read_vec3(constraints, base_offset + _OFF_MAX_FORCE_LIN, cid)
    write_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_ANG, cid, max_force_ang * dt)
    write_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_LIN, cid, max_force_lin * dt)

    # ---------------------------------------------------------------
    # Translation block prepare
    # ---------------------------------------------------------------
    if trans_mode_v == _TRANS_MODE_POINT:
        # All 3 lin axes are rigid + no targets + no caps -> fused
        # PointConstraintPart. Same math as Jolt's
        # ``PointConstraintPart::CalculateConstraintProperties``.
        eff_mass, inv_i1_r1x, inv_i2_r2x = _point_calculate_properties(
            r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )
        write_mat33(constraints, base_offset + _OFF_POINT_INV_I1_R1X, cid, inv_i1_r1x)
        write_mat33(constraints, base_offset + _OFF_POINT_INV_I2_R2X, cid, inv_i2_r2x)
        write_mat33(constraints, base_offset + _OFF_POINT_EFF_MASS, cid, eff_mass)
        # Warm-start with cached lambda from previous step.
        total_lambda_p = read_vec3(
            constraints, base_offset + _OFF_POINT_TOTAL_LAMBDA, cid
        )
        _point_apply_velocity_step(
            bodies, b1, b2, inv_mass1, inv_mass2, inv_i1_r1x, inv_i2_r2x, total_lambda_p
        )
    else:
        # Per-axis Axis parts. Read in batch then process axis-by-axis
        # via a small helper to keep the kernel readable.
        u = (pos2 + r2) - (pos1 + r1)  # p2 - p1
        r1_plus_u = r1 + u  # = p2 - x1
        hertz_lin = read_vec3(constraints, base_offset + _OFF_HERTZ_LIN, cid)
        damp_lin = read_vec3(constraints, base_offset + _OFF_DAMPING_LIN, cid)
        target_velocity_lin_pre = read_vec3(
            constraints, base_offset + _OFF_TARGET_VEL_LIN, cid
        )
        max_force_lin_v = max_force_lin
        eff_v = wp.vec3f(0.0, 0.0, 0.0)
        soft_v = wp.vec3f(0.0, 0.0, 0.0)
        bias_v = wp.vec3f(0.0, 0.0, 0.0)

        # See angular block above for bias derivation; for linear
        # velocity motors the same convention applies. ``c_err`` is
        # zeroed for axes with a non-zero ``target_velocity`` so the
        # position spring doesn't fight the motor.
        tv_lin_v = target_velocity_lin_pre

        # ---- axis 0 ----
        if max_force_lin_v[0] > 0.0:
            ax = e0_world
            inv_eff, r1pu_x, r2_x, ii1, ii2 = _axis_calculate_inv_eff_mass(
                r1_plus_u, r2, ax, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
            )
            c_err = wp.dot(u, ax)
            if tv_lin_v[0] != 0.0:
                c_err = 0.0
                hertz_eff = 0.0
            else:
                hertz_eff = hertz_lin[0]
            props = _calc_spring_props(
                inv_eff, -tv_lin_v[0], c_err, hertz_eff, damp_lin[0], dt
            )
            eff_v[0] = props[0]
            soft_v[0] = props[1]
            bias_v[0] = props[2]
            write_vec3(constraints, base_offset + _OFF_AXIS_R1PU_X_0, cid, r1pu_x)
            write_vec3(constraints, base_offset + _OFF_AXIS_R2_X_0, cid, r2_x)
            write_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_0, cid, ii1)
            write_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_0, cid, ii2)

        # ---- axis 1 ----
        if max_force_lin_v[1] > 0.0:
            ax = e1_world
            inv_eff, r1pu_x, r2_x, ii1, ii2 = _axis_calculate_inv_eff_mass(
                r1_plus_u, r2, ax, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
            )
            c_err = wp.dot(u, ax)
            if tv_lin_v[1] != 0.0:
                c_err = 0.0
                hertz_eff = 0.0
            else:
                hertz_eff = hertz_lin[1]
            props = _calc_spring_props(
                inv_eff, -tv_lin_v[1], c_err, hertz_eff, damp_lin[1], dt
            )
            eff_v[1] = props[0]
            soft_v[1] = props[1]
            bias_v[1] = props[2]
            write_vec3(constraints, base_offset + _OFF_AXIS_R1PU_X_1, cid, r1pu_x)
            write_vec3(constraints, base_offset + _OFF_AXIS_R2_X_1, cid, r2_x)
            write_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_1, cid, ii1)
            write_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_1, cid, ii2)

        # ---- axis 2 ----
        if max_force_lin_v[2] > 0.0:
            ax = e2_world
            inv_eff, r1pu_x, r2_x, ii1, ii2 = _axis_calculate_inv_eff_mass(
                r1_plus_u, r2, ax, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
            )
            c_err = wp.dot(u, ax)
            if tv_lin_v[2] != 0.0:
                c_err = 0.0
                hertz_eff = 0.0
            else:
                hertz_eff = hertz_lin[2]
            props = _calc_spring_props(
                inv_eff, -tv_lin_v[2], c_err, hertz_eff, damp_lin[2], dt
            )
            eff_v[2] = props[0]
            soft_v[2] = props[1]
            bias_v[2] = props[2]
            write_vec3(constraints, base_offset + _OFF_AXIS_R1PU_X_2, cid, r1pu_x)
            write_vec3(constraints, base_offset + _OFF_AXIS_R2_X_2, cid, r2_x)
            write_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_2, cid, ii1)
            write_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_2, cid, ii2)

        write_vec3(constraints, base_offset + _OFF_AXIS_EFF_MASS, cid, eff_v)
        write_vec3(constraints, base_offset + _OFF_AXIS_SOFTNESS, cid, soft_v)
        write_vec3(constraints, base_offset + _OFF_AXIS_BIAS, cid, bias_v)

        # Warm-start each active row. As in the angular block above,
        # we reset ``total_lambda`` to zero per-substep on
        # velocity-motor axes so the per-substep impulse cap actually
        # limits the per-substep applied force.
        total_lambda_a_in = read_vec3(
            constraints, base_offset + _OFF_AXIS_TOTAL_LAMBDA, cid
        )
        if tv_lin_v[0] != 0.0:
            tla0 = wp.float32(0.0)
        else:
            tla0 = total_lambda_a_in[0]
        if tv_lin_v[1] != 0.0:
            tla1 = wp.float32(0.0)
        else:
            tla1 = total_lambda_a_in[1]
        if tv_lin_v[2] != 0.0:
            tla2 = wp.float32(0.0)
        else:
            tla2 = total_lambda_a_in[2]
        total_lambda_a = wp.vec3f(tla0, tla1, tla2)
        write_vec3(
            constraints, base_offset + _OFF_AXIS_TOTAL_LAMBDA, cid, total_lambda_a
        )
        if eff_v[0] > 0.0:
            ii1 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_0, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_0, cid)
            _axis_apply_velocity_step(
                bodies, b1, b2, inv_mass1, inv_mass2, e0_world, ii1, ii2,
                total_lambda_a[0],
            )
        if eff_v[1] > 0.0:
            ii1 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_1, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_1, cid)
            _axis_apply_velocity_step(
                bodies, b1, b2, inv_mass1, inv_mass2, e1_world, ii1, ii2,
                total_lambda_a[1],
            )
        if eff_v[2] > 0.0:
            ii1 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_2, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_2, cid)
            _axis_apply_velocity_step(
                bodies, b1, b2, inv_mass1, inv_mass2, e2_world, ii1, ii2,
                total_lambda_a[2],
            )

    # ---------------------------------------------------------------
    # Rotation block prepare
    # ---------------------------------------------------------------
    if rot_mode_v == _ROT_MODE_EULER:
        # All 3 ang axes are rigid -> fused RotationEulerConstraintPart.
        eff_mass = _euler_calculate_properties(inv_inertia1, inv_inertia2)
        write_mat33(constraints, base_offset + _OFF_EULER_EFF_MASS, cid, eff_mass)
        total_lambda_e = read_vec3(
            constraints, base_offset + _OFF_EULER_TOTAL_LAMBDA, cid
        )
        _euler_apply_velocity_step(
            bodies, b1, b2, inv_inertia1, inv_inertia2, total_lambda_e
        )
    else:
        # Per-axis Angle parts.
        hertz_ang = read_vec3(constraints, base_offset + _OFF_HERTZ_ANG, cid)
        damp_ang = read_vec3(constraints, base_offset + _OFF_DAMPING_ANG, cid)
        target_velocity_ang_pre = read_vec3(
            constraints, base_offset + _OFF_TARGET_VEL_ANG, cid
        )
        max_force_ang_v = max_force_ang
        eff_v = wp.vec3f(0.0, 0.0, 0.0)
        soft_v = wp.vec3f(0.0, 0.0, 0.0)
        bias_v = wp.vec3f(0.0, 0.0, 0.0)

        # Angular position error in body-1 constraint coords (only used
        # for axes that have a position drive). Same derivation as
        # Jolt's SetTargetOrientationCS / projected_diff path: the
        # error angles per axis are ``2 * axis_world . diff.xyz``
        # where ``diff`` is the relative-rotation quaternion.
        # For our current API we don't have asymmetric limits, so the
        # "constraint error" reduces to the rotation-from-rest along
        # each constraint axis, which we compute from ``q_inv_init``.
        q_inv_init = read_quat(constraints, base_offset + _OFF_Q_INV_INIT, cid)
        diff = q2 * q_inv_init * wp.quat_inverse(q1)
        if diff[3] < 0.0:
            sign = -1.0
        else:
            sign = 1.0
        diff_xyz = wp.vec3f(diff[0] * sign, diff[1] * sign, diff[2] * sign)

        # Per-axis: ``c_err`` -> spring "Baumgarte" pull toward the
        # rest pose / position target (the target is folded into
        # ``q_inv_init`` so any rotation drift gives a non-zero
        # ``diff_xyz``). The velocity bias is ``-target_velocity``
        # (Jolt convention) so that ``lambda = eff * ((w1-w2).ax -
        # (-target))`` drives ``(w2-w1).ax`` toward ``target``.
        #
        # Mirrors Jolt's split: an axis with a non-zero
        # ``target_velocity`` is treated as a *velocity motor* (no
        # position spring), so its ``c_err`` is forced to zero -- the
        # spring would otherwise fight the motor and drag the
        # steady-state velocity below the setpoint. An axis with
        # ``target_velocity == 0`` and ``hertz > 0`` keeps ``c_err``
        # active and behaves as a soft lock / position drive that
        # restores to the (target-folded) rest pose.
        tv_ang_v = target_velocity_ang_pre

        # ---- axis 0 ----
        if max_force_ang_v[0] > 0.0:
            ax = e0_world
            inv_eff, ii1, ii2 = _angle_calculate_inv_eff_mass(
                ax, inv_inertia1, inv_inertia2
            )
            c_err = 2.0 * wp.dot(ax, diff_xyz)
            # Velocity-motor rule (Jolt): rigid row + ``-target`` bias,
            # ignore ``hertz`` and the position spring entirely.
            if tv_ang_v[0] != 0.0:
                c_err = 0.0
                hertz_eff = 0.0
            else:
                hertz_eff = hertz_ang[0]
            props = _calc_spring_props(
                inv_eff, -tv_ang_v[0], c_err, hertz_eff, damp_ang[0], dt
            )
            eff_v[0] = props[0]
            soft_v[0] = props[1]
            bias_v[0] = props[2]
            write_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_0, cid, ii1)
            write_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_0, cid, ii2)

        # ---- axis 1 ----
        if max_force_ang_v[1] > 0.0:
            ax = e1_world
            inv_eff, ii1, ii2 = _angle_calculate_inv_eff_mass(
                ax, inv_inertia1, inv_inertia2
            )
            c_err = 2.0 * wp.dot(ax, diff_xyz)
            if tv_ang_v[1] != 0.0:
                c_err = 0.0
                hertz_eff = 0.0
            else:
                hertz_eff = hertz_ang[1]
            props = _calc_spring_props(
                inv_eff, -tv_ang_v[1], c_err, hertz_eff, damp_ang[1], dt
            )
            eff_v[1] = props[0]
            soft_v[1] = props[1]
            bias_v[1] = props[2]
            write_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_1, cid, ii1)
            write_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_1, cid, ii2)

        # ---- axis 2 ----
        if max_force_ang_v[2] > 0.0:
            ax = e2_world
            inv_eff, ii1, ii2 = _angle_calculate_inv_eff_mass(
                ax, inv_inertia1, inv_inertia2
            )
            c_err = 2.0 * wp.dot(ax, diff_xyz)
            if tv_ang_v[2] != 0.0:
                c_err = 0.0
                hertz_eff = 0.0
            else:
                hertz_eff = hertz_ang[2]
            props = _calc_spring_props(
                inv_eff, -tv_ang_v[2], c_err, hertz_eff, damp_ang[2], dt
            )
            eff_v[2] = props[0]
            soft_v[2] = props[1]
            bias_v[2] = props[2]
            write_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_2, cid, ii1)
            write_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_2, cid, ii2)

        write_vec3(constraints, base_offset + _OFF_ANGLE_EFF_MASS, cid, eff_v)
        write_vec3(constraints, base_offset + _OFF_ANGLE_SOFTNESS, cid, soft_v)
        write_vec3(constraints, base_offset + _OFF_ANGLE_BIAS, cid, bias_v)

        # Warm-start each active row. ``total_lambda`` is reset to
        # zero per-substep on velocity-motor axes so the per-substep
        # impulse cap (``max_force * dt``) actually limits the
        # per-substep applied torque -- otherwise the cumulative
        # warm-started impulse saturates against the cap and the
        # motor stalls. Matches Jolt's pattern of calling
        # ``Deactivate()`` on motor parts in ``ResetWarmStart()``.
        total_lambda_g_in = read_vec3(
            constraints, base_offset + _OFF_ANGLE_TOTAL_LAMBDA, cid
        )
        # Warp: build a fresh vec3 with the kept components and zero
        # out the motor-axis ones (indexed assignment to vec3 is not
        # supported in Warp kernels).
        if tv_ang_v[0] != 0.0:
            tlg0 = wp.float32(0.0)
        else:
            tlg0 = total_lambda_g_in[0]
        if tv_ang_v[1] != 0.0:
            tlg1 = wp.float32(0.0)
        else:
            tlg1 = total_lambda_g_in[1]
        if tv_ang_v[2] != 0.0:
            tlg2 = wp.float32(0.0)
        else:
            tlg2 = total_lambda_g_in[2]
        total_lambda_g = wp.vec3f(tlg0, tlg1, tlg2)
        write_vec3(
            constraints, base_offset + _OFF_ANGLE_TOTAL_LAMBDA, cid, total_lambda_g
        )
        if eff_v[0] > 0.0:
            ii1 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_0, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_0, cid)
            _angle_apply_velocity_step(bodies, b1, b2, ii1, ii2, total_lambda_g[0])
        if eff_v[1] > 0.0:
            ii1 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_1, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_1, cid)
            _angle_apply_velocity_step(bodies, b1, b2, ii1, ii2, total_lambda_g[1])
        if eff_v[2] > 0.0:
            ii1 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_2, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_2, cid)
            _angle_apply_velocity_step(bodies, b1, b2, ii1, ii2, total_lambda_g[2])


# ---------------------------------------------------------------------------
# Iterate (PGS step)
# ---------------------------------------------------------------------------


@wp.func
def d6_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable PGS iteration step for the D6 joint.

    Solves each active sub-part in the same order as Jolt's
    ``SixDOFConstraint::SolveVelocityConstraint``: rotation parts
    first, then translation parts. See
    :func:`ball_socket_iterate_at` for the ``base_offset`` /
    ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    trans_mode_v = read_int(constraints, base_offset + _OFF_TRANS_MODE, cid)
    rot_mode_v = read_int(constraints, base_offset + _OFF_ROT_MODE, cid)
    axes_world = read_mat33(constraints, base_offset + _OFF_AXES_WORLD, cid)
    e0_world = wp.vec3f(axes_world[0, 0], axes_world[0, 1], axes_world[0, 2])
    e1_world = wp.vec3f(axes_world[1, 0], axes_world[1, 1], axes_world[1, 2])
    e2_world = wp.vec3f(axes_world[2, 0], axes_world[2, 1], axes_world[2, 2])

    # ----- Rotation block --------------------------------------------
    if rot_mode_v == _ROT_MODE_EULER:
        # Fused 3-DoF angular lock. lambda = K * (w1 - w2)
        eff_mass = read_mat33(constraints, base_offset + _OFF_EULER_EFF_MASS, cid)
        total_lambda = read_vec3(
            constraints, base_offset + _OFF_EULER_TOTAL_LAMBDA, cid
        )
        w1_e = bodies.angular_velocity[b1]
        w2_e = bodies.angular_velocity[b2]
        lam_v = eff_mass @ (w1_e - w2_e)
        total_lambda = total_lambda + lam_v
        write_vec3(constraints, base_offset + _OFF_EULER_TOTAL_LAMBDA, cid, total_lambda)
        _euler_apply_velocity_step(bodies, b1, b2, inv_inertia1, inv_inertia2, lam_v)
    else:
        # Per-axis 1-DoF rotations.
        eff_v = read_vec3(constraints, base_offset + _OFF_ANGLE_EFF_MASS, cid)
        soft_v = read_vec3(constraints, base_offset + _OFF_ANGLE_SOFTNESS, cid)
        bias_v = read_vec3(constraints, base_offset + _OFF_ANGLE_BIAS, cid)
        max_lambda_ang = read_vec3(
            constraints, base_offset + _OFF_MAX_LAMBDA_ANG, cid
        )
        total_lambda = read_vec3(
            constraints, base_offset + _OFF_ANGLE_TOTAL_LAMBDA, cid
        )

        # ---- axis 0 ----
        if eff_v[0] > 0.0:
            ax = e0_world
            ii1 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_0, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_0, cid)
            w1 = bodies.angular_velocity[b1]
            w2 = bodies.angular_velocity[b2]
            # Constraint equation (Jolt convention): we want
            # ``(w2 - w1) . ax = target_vel_ang[k]`` at steady state.
            # Velocity row: ``J*v = ax . (w1 - w2)``; bias for a
            # velocity motor is ``-target`` (folded into ``bias_v``
            # at prepare time), so ``lambda = eff * (J*v - bias) =
            # eff * (ax.(w1-w2) + target)`` drives the row to zero.
            jv = wp.dot(ax, w1 - w2)
            # bias_with_softness = softness * total_lambda + bias_const
            bias_full = soft_v[0] * total_lambda[0] + bias_v[0]
            lam_s = eff_v[0] * (jv - bias_full)
            new_lambda = wp.clamp(
                total_lambda[0] + lam_s, -max_lambda_ang[0], max_lambda_ang[0]
            )
            lam_s = new_lambda - total_lambda[0]
            total_lambda[0] = new_lambda
            _angle_apply_velocity_step(bodies, b1, b2, ii1, ii2, lam_s)

        # ---- axis 1 ----
        if eff_v[1] > 0.0:
            ax = e1_world
            ii1 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_1, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_1, cid)
            w1 = bodies.angular_velocity[b1]
            w2 = bodies.angular_velocity[b2]
            jv = wp.dot(ax, w1 - w2)
            bias_full = soft_v[1] * total_lambda[1] + bias_v[1]
            lam_s = eff_v[1] * (jv - bias_full)
            new_lambda = wp.clamp(
                total_lambda[1] + lam_s, -max_lambda_ang[1], max_lambda_ang[1]
            )
            lam_s = new_lambda - total_lambda[1]
            total_lambda[1] = new_lambda
            _angle_apply_velocity_step(bodies, b1, b2, ii1, ii2, lam_s)

        # ---- axis 2 ----
        if eff_v[2] > 0.0:
            ax = e2_world
            ii1 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I1_AXIS_2, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_ANGLE_INV_I2_AXIS_2, cid)
            w1 = bodies.angular_velocity[b1]
            w2 = bodies.angular_velocity[b2]
            jv = wp.dot(ax, w1 - w2)
            bias_full = soft_v[2] * total_lambda[2] + bias_v[2]
            lam_s = eff_v[2] * (jv - bias_full)
            new_lambda = wp.clamp(
                total_lambda[2] + lam_s, -max_lambda_ang[2], max_lambda_ang[2]
            )
            lam_s = new_lambda - total_lambda[2]
            total_lambda[2] = new_lambda
            _angle_apply_velocity_step(bodies, b1, b2, ii1, ii2, lam_s)

        write_vec3(constraints, base_offset + _OFF_ANGLE_TOTAL_LAMBDA, cid, total_lambda)

    # ----- Translation block -----------------------------------------
    if trans_mode_v == _TRANS_MODE_POINT:
        # Fused 3-DoF ball-socket. lambda = K * (v1 - cross(r1, w1) -
        #                                        v2 + cross(r2, w2))
        eff_mass = read_mat33(constraints, base_offset + _OFF_POINT_EFF_MASS, cid)
        inv_i1_r1x = read_mat33(
            constraints, base_offset + _OFF_POINT_INV_I1_R1X, cid
        )
        inv_i2_r2x = read_mat33(
            constraints, base_offset + _OFF_POINT_INV_I2_R2X, cid
        )
        r1 = read_vec3(constraints, base_offset + _OFF_R1, cid)
        r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)
        v1 = bodies.velocity[b1]
        w1 = bodies.angular_velocity[b1]
        v2 = bodies.velocity[b2]
        w2 = bodies.angular_velocity[b2]
        jv_v = (v1 - wp.cross(r1, w1)) - (v2 - wp.cross(r2, w2))
        lam_v = eff_mass @ jv_v
        total_lambda_p = read_vec3(
            constraints, base_offset + _OFF_POINT_TOTAL_LAMBDA, cid
        )
        total_lambda_p = total_lambda_p + lam_v
        write_vec3(
            constraints, base_offset + _OFF_POINT_TOTAL_LAMBDA, cid, total_lambda_p
        )
        _point_apply_velocity_step(
            bodies, b1, b2, inv_mass1, inv_mass2, inv_i1_r1x, inv_i2_r2x, lam_v
        )
    else:
        # Per-axis 1-DoF translations.
        eff_v = read_vec3(constraints, base_offset + _OFF_AXIS_EFF_MASS, cid)
        soft_v = read_vec3(constraints, base_offset + _OFF_AXIS_SOFTNESS, cid)
        bias_v = read_vec3(constraints, base_offset + _OFF_AXIS_BIAS, cid)
        max_lambda_lin = read_vec3(
            constraints, base_offset + _OFF_MAX_LAMBDA_LIN, cid
        )
        total_lambda = read_vec3(
            constraints, base_offset + _OFF_AXIS_TOTAL_LAMBDA, cid
        )

        # ---- axis 0 ----
        if eff_v[0] > 0.0:
            ax = e0_world
            r1pu_x = read_vec3(constraints, base_offset + _OFF_AXIS_R1PU_X_0, cid)
            r2_x = read_vec3(constraints, base_offset + _OFF_AXIS_R2_X_0, cid)
            ii1 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_0, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_0, cid)
            v1 = bodies.velocity[b1]
            w1 = bodies.angular_velocity[b1]
            v2 = bodies.velocity[b2]
            w2 = bodies.angular_velocity[b2]
            # Linear velocity row, Jolt convention. Velocity-motor
            # target is folded into ``bias_v[k]`` (= ``-target``) at
            # prepare time, so this is just ``J*v``.
            jv = (
                wp.dot(ax, v1 - v2)
                + wp.dot(r1pu_x, w1)
                - wp.dot(r2_x, w2)
            )
            bias_full = soft_v[0] * total_lambda[0] + bias_v[0]
            lam_s = eff_v[0] * (jv - bias_full)
            new_lambda = wp.clamp(
                total_lambda[0] + lam_s, -max_lambda_lin[0], max_lambda_lin[0]
            )
            lam_s = new_lambda - total_lambda[0]
            total_lambda[0] = new_lambda
            _axis_apply_velocity_step(
                bodies, b1, b2, inv_mass1, inv_mass2, ax, ii1, ii2, lam_s
            )

        # ---- axis 1 ----
        if eff_v[1] > 0.0:
            ax = e1_world
            r1pu_x = read_vec3(constraints, base_offset + _OFF_AXIS_R1PU_X_1, cid)
            r2_x = read_vec3(constraints, base_offset + _OFF_AXIS_R2_X_1, cid)
            ii1 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_1, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_1, cid)
            v1 = bodies.velocity[b1]
            w1 = bodies.angular_velocity[b1]
            v2 = bodies.velocity[b2]
            w2 = bodies.angular_velocity[b2]
            jv = (
                wp.dot(ax, v1 - v2)
                + wp.dot(r1pu_x, w1)
                - wp.dot(r2_x, w2)
            )
            bias_full = soft_v[1] * total_lambda[1] + bias_v[1]
            lam_s = eff_v[1] * (jv - bias_full)
            new_lambda = wp.clamp(
                total_lambda[1] + lam_s, -max_lambda_lin[1], max_lambda_lin[1]
            )
            lam_s = new_lambda - total_lambda[1]
            total_lambda[1] = new_lambda
            _axis_apply_velocity_step(
                bodies, b1, b2, inv_mass1, inv_mass2, ax, ii1, ii2, lam_s
            )

        # ---- axis 2 ----
        if eff_v[2] > 0.0:
            ax = e2_world
            r1pu_x = read_vec3(constraints, base_offset + _OFF_AXIS_R1PU_X_2, cid)
            r2_x = read_vec3(constraints, base_offset + _OFF_AXIS_R2_X_2, cid)
            ii1 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I1_R1PU_X_2, cid)
            ii2 = read_vec3(constraints, base_offset + _OFF_AXIS_INV_I2_R2_X_2, cid)
            v1 = bodies.velocity[b1]
            w1 = bodies.angular_velocity[b1]
            v2 = bodies.velocity[b2]
            w2 = bodies.angular_velocity[b2]
            jv = (
                wp.dot(ax, v1 - v2)
                + wp.dot(r1pu_x, w1)
                - wp.dot(r2_x, w2)
            )
            bias_full = soft_v[2] * total_lambda[2] + bias_v[2]
            lam_s = eff_v[2] * (jv - bias_full)
            new_lambda = wp.clamp(
                total_lambda[2] + lam_s, -max_lambda_lin[2], max_lambda_lin[2]
            )
            lam_s = new_lambda - total_lambda[2]
            total_lambda[2] = new_lambda
            _axis_apply_velocity_step(
                bodies, b1, b2, inv_mass1, inv_mass2, ax, ii1, ii2, lam_s
            )

        write_vec3(constraints, base_offset + _OFF_AXIS_TOTAL_LAMBDA, cid, total_lambda)


# ---------------------------------------------------------------------------
# World-frame wrench
# ---------------------------------------------------------------------------


@wp.func
def d6_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable wrench on body 2; see :func:`d6_world_wrench`.

    Sums the contributions from whichever sub-parts are active. The
    linear constraint force is the per-axis Lagrange multipliers
    along their world axes (or the fused PointConstraint impulse) /
    dt; the torque is the rotational Lagrange multipliers along
    their world axes (or the fused EulerConstraint impulse) plus the
    moment of the linear force about body 2's COM.
    """
    trans_mode_v = read_int(constraints, base_offset + _OFF_TRANS_MODE, cid)
    rot_mode_v = read_int(constraints, base_offset + _OFF_ROT_MODE, cid)
    axes_world = read_mat33(constraints, base_offset + _OFF_AXES_WORLD, cid)
    e0_world = wp.vec3f(axes_world[0, 0], axes_world[0, 1], axes_world[0, 2])
    e1_world = wp.vec3f(axes_world[1, 0], axes_world[1, 1], axes_world[1, 2])
    e2_world = wp.vec3f(axes_world[2, 0], axes_world[2, 1], axes_world[2, 2])
    r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)

    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)

    if trans_mode_v == _TRANS_MODE_POINT:
        total_lambda_p = read_vec3(
            constraints, base_offset + _OFF_POINT_TOTAL_LAMBDA, cid
        )
        force = total_lambda_p * idt
    else:
        total_lambda_a = read_vec3(
            constraints, base_offset + _OFF_AXIS_TOTAL_LAMBDA, cid
        )
        force = (
            (total_lambda_a[0] * idt) * e0_world
            + (total_lambda_a[1] * idt) * e1_world
            + (total_lambda_a[2] * idt) * e2_world
        )

    if rot_mode_v == _ROT_MODE_EULER:
        total_lambda_e = read_vec3(
            constraints, base_offset + _OFF_EULER_TOTAL_LAMBDA, cid
        )
        torque = total_lambda_e * idt
    else:
        total_lambda_g = read_vec3(
            constraints, base_offset + _OFF_ANGLE_TOTAL_LAMBDA, cid
        )
        torque = (
            (total_lambda_g[0] * idt) * e0_world
            + (total_lambda_g[1] * idt) * e1_world
            + (total_lambda_g[2] * idt) * e2_world
        )

    # Add moment of the linear force about body 2's COM. Sign matches
    # ``hinge_angle_world_wrench_at`` / ``ball_socket_world_wrench_at``.
    torque = torque + wp.cross(r2, force)
    return force, torque


# ---------------------------------------------------------------------------
# Direct entry-point wrappers (cid -> body indices via header)
# ---------------------------------------------------------------------------


@wp.func
def d6_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct prepare entry: reads body indices from the column header
    and forwards to :func:`d6_prepare_for_iteration_at` with
    ``base_offset = 0``."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    d6_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def d6_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct iterate entry; see :func:`d6_iterate_at`."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    d6_iterate_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def d6_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this constraint exerts on body 2.

    Force is the linear constraint impulse divided by the substep
    ``dt`` (``idt = 1 / substep_dt``); torque is that force's moment
    about body 2's COM plus the angular impulse / dt. Per-axis caps
    show up here as saturated (``+/- max_force``) components in the
    reported wrench, decomposed in the world frame.
    """
    return d6_world_wrench_at(constraints, cid, 0, idt)
