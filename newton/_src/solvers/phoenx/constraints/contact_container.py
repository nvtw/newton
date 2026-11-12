# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-contact persistent + per-substep state, keyed by sorted-buffer index k.

* ``impulses`` -- mutable accumulated normal/tangent impulse rows.
* ``lambdas`` -- read-mostly persistent contact manifold data (normal, tangent,
  barycentrics).
* ``derived`` -- per-substep scratch (eff masses, biases); rebuilt every prepare.

All buffers are ``[dword, k]`` with k inner for coalesced loads.
"""

from __future__ import annotations

import math

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    contact_soft_coefficient_values,
    contact_spring_omega_dt,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.helpers.array_access import read2d_f32, write2d_f32

__all__ = [
    "CC_DERIVED_DWORDS_PER_CONTACT",
    "CC_DWORDS_PER_CONTACT",
    "CC_IMPULSE_DWORDS_PER_CONTACT",
    "CC_RIGID_DWORDS_PER_CONTACT",
    "ContactContainer",
    "cc_get_bias",
    "cc_get_bias_t1",
    "cc_get_bias_t2",
    "cc_get_eff_n",
    "cc_get_eff_t1",
    "cc_get_eff_t2",
    "cc_get_friction_anchor0",
    "cc_get_friction_anchor1",
    "cc_get_mobility_nt1",
    "cc_get_mobility_nt2",
    "cc_get_mobility_t1t2",
    "cc_get_normal",
    "cc_get_normal_lambda",
    "cc_get_normal_lambdas",
    "cc_get_pd_bias",
    "cc_get_pd_eff_soft",
    "cc_get_pd_gamma",
    "cc_get_prev_normal",
    "cc_get_prev_normal_lambda",
    "cc_get_prev_tangent1",
    "cc_get_prev_tangent1_lambda",
    "cc_get_prev_tangent2_lambda",
    "cc_get_r0",
    "cc_get_r1",
    "cc_get_side0_bary",
    "cc_get_side1_bary",
    "cc_get_start_gap",
    "cc_get_tangent1",
    "cc_get_tangent1_lambda",
    "cc_get_tangent2_lambda",
    "cc_set_bias",
    "cc_set_bias_t1",
    "cc_set_bias_t2",
    "cc_set_eff_n",
    "cc_set_eff_t1",
    "cc_set_eff_t2",
    "cc_set_friction_anchor0",
    "cc_set_friction_anchor1",
    "cc_set_mobility_nt1",
    "cc_set_mobility_nt2",
    "cc_set_mobility_t1t2",
    "cc_set_normal",
    "cc_set_normal_lambda",
    "cc_set_pd_bias",
    "cc_set_pd_eff_soft",
    "cc_set_pd_gamma",
    "cc_set_r0",
    "cc_set_r1",
    "cc_set_side0_bary",
    "cc_set_side1_bary",
    "cc_set_start_gap",
    "cc_set_tangent1",
    "cc_set_tangent1_lambda",
    "cc_set_tangent2_lambda",
    "cc_soft_coefficients",
    "contact_container_clear_reset_worlds",
    "contact_container_copy_current_to_prev",
    "contact_container_set_sweeps",
    "contact_container_zeros",
    "contact_solve_container_zeros",
]


#: Dwords of mutable persistent impulse per contact: normal + two tangents.
CC_IMPULSE_DWORDS_PER_CONTACT: int = 3

#: 12 = normal(3) + tangent1(3) + side0_bary(3) + side1_bary(3). The two
#: ``bary`` slots are populated by contact ingest when a side is a cloth triangle;
#: rigid sides use the same slots for body-local material friction anchors.
CC_DWORDS_PER_CONTACT: int = 12

#: Frame directions plus two body-local friction anchors. Collision witnesses
#: remain canonical geometry and are never changed by a friction-anchor reset.
CC_RIGID_DWORDS_PER_CONTACT: int = 12

#: 19 = eff_n + eff_t1 + eff_t2 + bias + bias_t1 + bias_t2 + pd_gamma + pd_bias +
#: pd_eff_soft + r0(3) + r1(3) + mobility(3) + start_gap. pd_* are non-zero only
#: for soft contacts (user K/D); pd_eff_soft > 0 switches the normal row to
#: absolute PD spring-damper. Rigid-contact prepare caches lever arms for the
#: velocity sweeps. Three slots cache the off-diagonal rigid contact-frame mobility terms reused
#: by every velocity sweep. The final slot stores the generation-time gap for
#: current contacts only; it is written by ingest and consumed by prepare.
#: Canonical eff_t1/eff_t2 contain inverse tangent mobility. Packed rigid solve
#: rows store tangent mobility directly; their prepare/projector factories
#: must both select tangent_mobility_layout.
CC_DERIVED_DWORDS_PER_CONTACT: int = 19


# Compile-time dword offsets.
_CC_OFF_NORMAL_LAMBDA = wp.constant(0)
_CC_OFF_TANGENT1_LAMBDA = wp.constant(1)
_CC_OFF_TANGENT2_LAMBDA = wp.constant(2)
_CC_OFF_NORMAL_X = wp.constant(0)
_CC_OFF_NORMAL_Y = wp.constant(1)
_CC_OFF_NORMAL_Z = wp.constant(2)
_CC_OFF_TANGENT1_X = wp.constant(3)
_CC_OFF_TANGENT1_Y = wp.constant(4)
_CC_OFF_TANGENT1_Z = wp.constant(5)
_CC_OFF_SIDE0_BARY_X = wp.constant(6)
_CC_OFF_SIDE0_BARY_Y = wp.constant(7)
_CC_OFF_SIDE0_BARY_Z = wp.constant(8)
_CC_OFF_SIDE1_BARY_X = wp.constant(9)
_CC_OFF_SIDE1_BARY_Y = wp.constant(10)
_CC_OFF_SIDE1_BARY_Z = wp.constant(11)

_CC_OFF_EFF_N = wp.constant(0)
_CC_OFF_EFF_T1 = wp.constant(1)
_CC_OFF_EFF_T2 = wp.constant(2)
_CC_OFF_BIAS = wp.constant(3)
_CC_OFF_BIAS_T1 = wp.constant(4)
_CC_OFF_BIAS_T2 = wp.constant(5)
# Soft-contact PD plumbing. pd_eff_soft == 0 = opt-out (Box2D path).
_CC_OFF_PD_GAMMA = wp.constant(6)
_CC_OFF_PD_BIAS = wp.constant(7)
_CC_OFF_PD_EFF_SOFT = wp.constant(8)
_CC_OFF_R0_X = wp.constant(9)
_CC_OFF_R0_Y = wp.constant(10)
_CC_OFF_R0_Z = wp.constant(11)
_CC_OFF_R1_X = wp.constant(12)
_CC_OFF_R1_Y = wp.constant(13)
_CC_OFF_R1_Z = wp.constant(14)
_CC_OFF_MOBILITY_NT1 = wp.constant(15)
_CC_OFF_MOBILITY_NT2 = wp.constant(16)
_CC_OFF_MOBILITY_T1T2 = wp.constant(17)
_CC_OFF_START_GAP = wp.constant(18)


@wp.struct
class ContactContainer:
    """Per-contact warm-start + derived state. All buffers are
    ``(dwords, rigid_contact_max)`` with k inner. k matches Newton's per-contact
    arrays."""

    impulses: wp.array2d[wp.float32]
    prev_impulses: wp.array2d[wp.float32]
    lambdas: wp.array2d[wp.float32]
    prev_lambdas: wp.array2d[wp.float32]
    derived: wp.array2d[wp.float32]
    #: Tangential target slip of conveyor-surface contacts along (t1, t2) [m/s]; empty when the
    #: container cannot carry it. Written by prepare and read by iterate only when the contacts
    #: have surface velocities, so other scenes never touch it.
    surface_target: wp.array[wp.vec2f]
    #: Signed normal gap [m] of each prepared row, advanced by the rigid motion
    #: of every substep that reuses the prepared rows; empty when every
    #: substep prepares its rows.
    gap: wp.array[wp.float32]
    #: Nonzero when the sweep schedule or a requested contact frequency caps
    #: the maximal contact spring below the Nyquist limit
    #: (:func:`contact_container_set_sweeps`). The capped
    #: spring's bias rate times the substep ``dt``, mass and impulse
    #: coefficients follow as launch constants; uncapped rows keep the
    #: Nyquist-limited :data:`DEFAULT_HERTZ_CONTACT` spring.
    soft_capped: wp.int32
    soft_bias_dt: wp.float32
    soft_mass_coeff: wp.float32
    soft_impulse_coeff: wp.float32
    #: Requested contact frequency [Hz], or zero when none was requested.
    hertz: wp.float32


@wp.func
def cc_soft_coefficients(cc: ContactContainer, idt: wp.float32):
    """Return ``(bias_rate, mass_coeff, impulse_coeff)`` of the contact rows at substep rate ``idt``."""
    if cc.soft_capped == 0:
        return soft_constraint_coefficients(DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt)
    return cc.soft_bias_dt * idt, cc.soft_mass_coeff, cc.soft_impulse_coeff


@wp.func
def cc_reduced_soft_coefficients(cc: ContactContainer, idt: wp.float32):
    """Return ``(bias_rate, mass_coeff, impulse_coeff)`` of contact rows solved in reduced coordinates.

    The rows keep the Nyquist-limited :data:`DEFAULT_HERTZ_CONTACT` spring,
    softened only to a requested contact frequency. The sweep-schedule cap of
    :func:`cc_soft_coefficients` bounds the slow modes of maximal-coordinate
    stacks, which the reduced solve does not have.
    """
    # Two calls keep the default spring's constants folded at compile time.
    if cc.hertz > 0.0:
        return soft_constraint_coefficients(cc.hertz, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt)
    return soft_constraint_coefficients(DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt)


@wp.func
def cc_lagged_soft_coefficients(cc: ContactContainer, idt: wp.float32, point_count: wp.int32):
    """Return ``(bias_rate, mass_coeff, impulse_coeff)`` of contact rows whose push-out acts a substep late.

    The contacts between fused trees stop an approach at once, but the push-out of their
    spring enters the next substep as a force. A spring of angular frequency ``omega`` and
    damping ratio ``zeta`` that acts ``dt`` late stays damped while ``omega dt < 2 zeta``;
    stiffer ones pump the rocking of stacked trees. The ``point_count`` points of one column
    press on the same bodies and their stiffnesses add, so the bound holds for the column:
    its rows take ``omega dt = zeta pi / (2 sqrt(point_count))``, or a lower requested contact
    frequency. Stacks of boxes and of hinged chains rest at that value and rock from 1.3
    times it.
    """
    hertz = DEFAULT_DAMPING_RATIO * idt * wp.float32(0.25) / wp.sqrt(wp.float32(wp.max(point_count, 1)))
    if cc.hertz > 0.0:
        hertz = wp.min(hertz, cc.hertz)
    return soft_constraint_coefficients(hertz, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt)


def contact_container_set_sweeps(
    cc: ContactContainer,
    solver_iterations: int,
    velocity_iterations: int,
    contact_hertz: float | None = None,
    dt: float = 0.0,
    step_dt: float = 0.0,
) -> None:
    """Set the contact spring for a sweep schedule and substep ``dt`` [s].

    The spring is the stiffest one the sweeps keep stable
    (:func:`contact_spring_omega_dt`), softened to ``contact_hertz`` [Hz]
    when that is given and lower. A fixed frequency gives each contact a
    stiffness that does not change with the substep count.

    Contact gaps are measured once per ``step_dt`` [s] and only extrapolated
    by its substeps. A spring that oscillates many times between two
    measurements corrects the extrapolation error of curved, moving contacts
    faster than it is measured and pumps energy into them (the analog-digital
    clock's gears hop at 900 Hz with 120 Hz steps, and rest at 480 Hz), so
    the spring stays within twice the step rate. Up to four substeps per step
    the sweep bound is the lower one.
    """
    omega_dt = contact_spring_omega_dt(solver_iterations, velocity_iterations)
    if step_dt > 0.0 and dt > 0.0:
        omega_dt = min(omega_dt, 4.0 * math.pi * float(dt) / float(step_dt))
    if contact_hertz is not None:
        omega_dt = min(omega_dt, 2.0 * math.pi * float(contact_hertz) * float(dt))
    cc.hertz = 0.0 if contact_hertz is None else float(contact_hertz)
    cc.soft_capped = int(omega_dt < math.pi)
    cc.soft_bias_dt, cc.soft_mass_coeff, cc.soft_impulse_coeff = contact_soft_coefficient_values(omega_dt)


@wp.kernel(enable_backward=False)
def _contact_container_copy_current_to_prev_kernel(cc: ContactContainer, valid_count: wp.array[wp.int32]):
    k = wp.tid()
    # Newton packs live contacts into [0, count); slots at or beyond the prior
    # solve's count hold no warm-start state, so skip them. The launch still
    # spans the full capacity (graph-stable dim); the early-out elides the copy
    # traffic for the typically large inactive tail.
    if k >= valid_count[0]:
        return
    for row in range(CC_IMPULSE_DWORDS_PER_CONTACT):
        cc.prev_impulses[row, k] = cc.impulses[row, k]
    for row in range(CC_RIGID_DWORDS_PER_CONTACT):
        cc.prev_lambdas[row, k] = cc.lambdas[row, k]


@wp.kernel(enable_backward=False)
def _contact_container_clear_reset_worlds_kernel(
    dones: wp.array[wp.float32],
    shape_world: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    cc: ContactContainer,
    cid_of_contact_cur: wp.array[wp.int32],
    cid_of_contact_prev: wp.array[wp.int32],
):
    k = wp.tid()

    world = wp.int32(-1)
    shape0 = rigid_contact_shape0[k]
    if shape0 >= wp.int32(0) and shape0 < shape_world.shape[0]:
        world0 = shape_world[shape0]
        if world0 >= wp.int32(0):
            world = world0

    shape1 = rigid_contact_shape1[k]
    if world < wp.int32(0) and shape1 >= wp.int32(0) and shape1 < shape_world.shape[0]:
        world1 = shape_world[shape1]
        if world1 >= wp.int32(0):
            world = world1

    if world < wp.int32(0) or world >= dones.shape[0] or dones[world] <= wp.float32(0.5):
        return

    for row in range(CC_IMPULSE_DWORDS_PER_CONTACT):
        cc.impulses[row, k] = wp.float32(0.0)
        cc.prev_impulses[row, k] = wp.float32(0.0)
    for row in range(CC_DWORDS_PER_CONTACT):
        cc.lambdas[row, k] = wp.float32(0.0)
    for row in range(CC_RIGID_DWORDS_PER_CONTACT):
        cc.prev_lambdas[row, k] = wp.float32(0.0)
    for row in range(CC_DERIVED_DWORDS_PER_CONTACT):
        cc.derived[row, k] = wp.float32(0.0)

    cid_of_contact_cur[k] = wp.int32(-1)
    cid_of_contact_prev[k] = wp.int32(-1)


# Mutable impulse accessors keyed by contact index k.


@wp.func
def cc_get_normal_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.impulses, _CC_OFF_NORMAL_LAMBDA, k)


@wp.func
def cc_get_normal_lambdas(cc: ContactContainer) -> wp.array[wp.float32]:
    """Return the normal-impulse row without copying its values."""
    return cc.impulses[_CC_OFF_NORMAL_LAMBDA]


@wp.func
def cc_set_normal_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.impulses, _CC_OFF_NORMAL_LAMBDA, k, v)


@wp.func
def cc_get_tangent1_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.impulses, _CC_OFF_TANGENT1_LAMBDA, k)


@wp.func
def cc_set_tangent1_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.impulses, _CC_OFF_TANGENT1_LAMBDA, k, v)


@wp.func
def cc_get_tangent2_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.impulses, _CC_OFF_TANGENT2_LAMBDA, k)


@wp.func
def cc_set_tangent2_lambda(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.impulses, _CC_OFF_TANGENT2_LAMBDA, k, v)


@wp.func
def cc_get_normal(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_NORMAL_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_NORMAL_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_NORMAL_Z, k),
    )


@wp.func
def cc_set_normal(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_NORMAL_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_NORMAL_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_NORMAL_Z, k, v[2])


@wp.func
def cc_get_tangent1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_TANGENT1_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Z, k),
    )


@wp.func
def cc_set_tangent1(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_TANGENT1_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_TANGENT1_Z, k, v[2])


# Per-side barycentric weights are populated only for deformable endpoints.
@wp.func
def cc_get_side0_bary(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Z, k),
    )


@wp.func
def cc_set_side0_bary(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE0_BARY_Z, k, v[2])


@wp.func
def cc_get_side1_bary(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_X, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Y, k),
        read2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Z, k),
    )


@wp.func
def cc_set_side1_bary(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_X, k, v[0])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Y, k, v[1])
    write2d_f32(cc.lambdas, _CC_OFF_SIDE1_BARY_Z, k, v[2])


@wp.func
def cc_get_start_gap(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_START_GAP, k)


@wp.func
def cc_set_start_gap(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_START_GAP, k, v)


@wp.func
def cc_get_friction_anchor0(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    """Read the rigid material anchor stored in the endpoint's body frame."""
    return cc_get_side0_bary(cc, k)


@wp.func
def cc_set_friction_anchor0(cc: ContactContainer, k: wp.int32, point: wp.vec3f):
    """Store a rigid friction reference independently of collision witnesses."""
    cc_set_side0_bary(cc, k, point)


@wp.func
def cc_get_friction_anchor1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    """Read the rigid material anchor stored in the endpoint's body frame."""
    return cc_get_side1_bary(cc, k)


@wp.func
def cc_set_friction_anchor1(cc: ContactContainer, k: wp.int32, point: wp.vec3f):
    """Store a rigid friction reference independently of collision witnesses."""
    cc_set_side1_bary(cc, k, point)


# ---- prev-step views ------------------------------------------------


@wp.func
def cc_get_prev_normal_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.prev_impulses, _CC_OFF_NORMAL_LAMBDA, k)


@wp.func
def cc_get_prev_tangent1_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.prev_impulses, _CC_OFF_TANGENT1_LAMBDA, k)


@wp.func
def cc_get_prev_tangent2_lambda(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.prev_impulses, _CC_OFF_TANGENT2_LAMBDA, k)


@wp.func
def cc_get_prev_normal(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.prev_lambdas, _CC_OFF_NORMAL_X, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_NORMAL_Y, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_NORMAL_Z, k),
    )


@wp.func
def cc_get_prev_tangent1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.prev_lambdas, _CC_OFF_TANGENT1_X, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_TANGENT1_Y, k),
        read2d_f32(cc.prev_lambdas, _CC_OFF_TANGENT1_Z, k),
    )


# ---------------------------------------------------------------------------
# Derived (per-substep scratch) accessors -- keyed by contact index k.
# ---------------------------------------------------------------------------


@wp.func
def cc_get_eff_n(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_EFF_N, k)


@wp.func
def cc_set_eff_n(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_EFF_N, k, v)


@wp.func
def cc_get_eff_t1(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_EFF_T1, k)


@wp.func
def cc_set_eff_t1(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_EFF_T1, k, v)


@wp.func
def cc_get_eff_t2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_EFF_T2, k)


@wp.func
def cc_set_eff_t2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_EFF_T2, k, v)


@wp.func
def cc_get_mobility_nt1(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_MOBILITY_NT1, k)


@wp.func
def cc_set_mobility_nt1(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_MOBILITY_NT1, k, v)


@wp.func
def cc_get_mobility_nt2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_MOBILITY_NT2, k)


@wp.func
def cc_set_mobility_nt2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_MOBILITY_NT2, k, v)


@wp.func
def cc_get_mobility_t1t2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_MOBILITY_T1T2, k)


@wp.func
def cc_set_mobility_t1t2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_MOBILITY_T1T2, k, v)


@wp.func
def cc_get_bias(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_BIAS, k)


@wp.func
def cc_set_bias(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_BIAS, k, v)


@wp.func
def cc_get_bias_t1(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_BIAS_T1, k)


@wp.func
def cc_set_bias_t1(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_BIAS_T1, k, v)


@wp.func
def cc_get_bias_t2(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_BIAS_T2, k)


@wp.func
def cc_set_bias_t2(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_BIAS_T2, k, v)


@wp.func
def cc_get_pd_gamma(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_PD_GAMMA, k)


@wp.func
def cc_set_pd_gamma(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_PD_GAMMA, k, v)


@wp.func
def cc_get_pd_bias(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_PD_BIAS, k)


@wp.func
def cc_set_pd_bias(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_PD_BIAS, k, v)


@wp.func
def cc_get_pd_eff_soft(cc: ContactContainer, k: wp.int32) -> wp.float32:
    return read2d_f32(cc.derived, _CC_OFF_PD_EFF_SOFT, k)


@wp.func
def cc_set_pd_eff_soft(cc: ContactContainer, k: wp.int32, v: wp.float32):
    write2d_f32(cc.derived, _CC_OFF_PD_EFF_SOFT, k, v)


@wp.func
def cc_get_r0(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.derived, _CC_OFF_R0_X, k),
        read2d_f32(cc.derived, _CC_OFF_R0_Y, k),
        read2d_f32(cc.derived, _CC_OFF_R0_Z, k),
    )


@wp.func
def cc_set_r0(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.derived, _CC_OFF_R0_X, k, v[0])
    write2d_f32(cc.derived, _CC_OFF_R0_Y, k, v[1])
    write2d_f32(cc.derived, _CC_OFF_R0_Z, k, v[2])


@wp.func
def cc_get_r1(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    return wp.vec3f(
        read2d_f32(cc.derived, _CC_OFF_R1_X, k),
        read2d_f32(cc.derived, _CC_OFF_R1_Y, k),
        read2d_f32(cc.derived, _CC_OFF_R1_Z, k),
    )


@wp.func
def cc_set_r1(cc: ContactContainer, k: wp.int32, v: wp.vec3f):
    write2d_f32(cc.derived, _CC_OFF_R1_X, k, v[0])
    write2d_f32(cc.derived, _CC_OFF_R1_Y, k, v[1])
    write2d_f32(cc.derived, _CC_OFF_R1_Z, k, v[2])


# ---------------------------------------------------------------------------
# Host-side factories.
# ---------------------------------------------------------------------------


def contact_container_zeros(
    rigid_contact_max: int,
    device: wp.DeviceLike = None,
    *,
    track_gap: bool = False,
    solver_iterations: int = 8,
    velocity_iterations: int = 1,
) -> ContactContainer:
    """Allocate a zero-initialised :class:`ContactContainer`. ``rigid_contact_max``
    must match Newton's Contacts buffer. ``track_gap`` allocates the row gaps
    needed when substeps reuse prepared rows. The sweep counts set the contact
    spring (:func:`contact_container_set_sweeps`)."""
    n = max(1, int(rigid_contact_max))
    cc = ContactContainer()
    cc.impulses = wp.zeros((CC_IMPULSE_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.prev_impulses = wp.zeros((CC_IMPULSE_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.lambdas = wp.zeros((CC_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.prev_lambdas = wp.zeros((CC_RIGID_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.derived = wp.zeros((CC_DERIVED_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.surface_target = wp.zeros(n, dtype=wp.vec2f, device=device)
    cc.gap = wp.zeros(n if track_gap else 0, dtype=wp.float32, device=device)
    contact_container_set_sweeps(cc, solver_iterations, velocity_iterations)
    return cc


def contact_solve_container_zeros(
    rigid_contact_max: int,
    device: wp.DeviceLike = None,
    *,
    rigid_only: bool = False,
    track_gap: bool = False,
    solver_iterations: int = 8,
    velocity_iterations: int = 1,
) -> ContactContainer:
    """Allocate only the rows consumed by rigid contact iterate/relax."""
    n = max(1, int(rigid_contact_max))
    cc = ContactContainer()
    cc.impulses = wp.zeros((CC_IMPULSE_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.prev_impulses = wp.zeros((1, 1), dtype=wp.float32, device=device)
    cc.prev_lambdas = wp.zeros((1, 1), dtype=wp.float32, device=device)
    # Prepare, iterate, and relax share this color-ordered state across
    # every substep, including manifold anchors needed by prepare.
    lambda_rows = CC_RIGID_DWORDS_PER_CONTACT if rigid_only else CC_DWORDS_PER_CONTACT
    cc.lambdas = wp.zeros((lambda_rows, n), dtype=wp.float32, device=device)
    cc.derived = wp.zeros((CC_DERIVED_DWORDS_PER_CONTACT, n), dtype=wp.float32, device=device)
    cc.surface_target = wp.empty(0, dtype=wp.vec2f, device=device)
    cc.gap = wp.zeros(n if track_gap else 0, dtype=wp.float32, device=device)
    contact_container_set_sweeps(cc, solver_iterations, velocity_iterations)
    return cc


def contact_container_copy_current_to_prev(
    cc: ContactContainer, valid_count: wp.array[wp.int32], device: wp.DeviceLike = None
) -> None:
    """Copy current persistent contact state into prev buffers.

    Args:
        valid_count: Single-element device array with the number of live
            contact slots (the prior solve's contact count). Slots beyond it
            are skipped, since Newton packs contacts into ``[0, count)``.
    """
    device_obj = wp.get_device(device)
    max_blocks = max(1, int(getattr(device_obj, "sm_count", 1))) * 4
    wp.launch(
        _contact_container_copy_current_to_prev_kernel,
        dim=int(cc.impulses.shape[1]),
        inputs=[cc, valid_count],
        device=device,
        max_blocks=max_blocks,
    )


def contact_container_clear_reset_worlds(
    cc: ContactContainer,
    cid_of_contact_cur: wp.array[wp.int32],
    cid_of_contact_prev: wp.array[wp.int32],
    contacts,
    shape_world: wp.array[wp.int32],
    dones: wp.array[wp.float32],
    device: wp.DeviceLike = None,
) -> None:
    """Clear contact warm-start state for worlds whose reset flag is set."""
    wp.launch(
        _contact_container_clear_reset_worlds_kernel,
        dim=int(cc.impulses.shape[1]),
        inputs=[
            dones,
            shape_world,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
        ],
        outputs=[cc, cid_of_contact_cur, cid_of_contact_prev],
        device=device,
    )
