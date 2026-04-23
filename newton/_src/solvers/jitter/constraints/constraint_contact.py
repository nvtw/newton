# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rigid-rigid contact constraint for the Jitter / PhoenX solvers.

One :data:`CONSTRAINT_TYPE_CONTACT` column represents one whole
``(shape_a, shape_b)`` shape-pair and stores just the range
``[contact_first, contact_first + contact_count)`` into the upstream
sorted :class:`newton._src.sim.contacts.Contacts` buffer. Every
per-contact quantity -- persistent warm-start state (lambdas, the
contact frame, body-local anchors) and per-substep derived quantities
(lever arms, effective masses, bias) -- lives in the parallel
:class:`newton._src.solvers.jitter.constraints.contact_container.ContactContainer`
indexed by the contact's sorted-buffer index ``k``.

This replaces the previous 6-slot-per-column layout. The old layout
capped each column at six contacts and split dense pairs across multiple
adjacent columns; because the graph colourer then had to assign those
adjacent columns to different colours (they share the same body pair),
intra-pair Gauss-Seidel was replaced with an inter-colour serial pass
that also inflated the number of PGS launches. The per-pair design
loops over every contact serially inside one kernel call, which is
literally Gauss-Seidel within the pair, and drops the colouring churn.

Friction: two-tangent pyramidal PGS. Each contact contributes three
scalar rows -- normal (non-penetration, ``lambda_n >= 0``) plus two
independent tangential rows (each clamped to the current
``[-mu * lambda_n, +mu * lambda_n]``). Matches what Box2D v3 and
solver2d do in their "simple friction" mode; the full Coulomb cone is
approximated by the circular clamp inside the iterate kernel.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_CONTACT,
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_get_type,
    read_float,
    read_int,
    soft_constraint_coefficients,
    write_float,
    write_int,
)
from newton._src.solvers.jitter.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_r1,
    cc_get_r2,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_bias,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_local_p0,
    cc_set_local_p1,
    cc_set_normal_lambda,
    cc_set_r1,
    cc_set_r2,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.jitter.helpers.data_packing import dword_offset_of, num_dwords

__all__ = [
    "CONTACT_DWORDS",
    "ContactConstraintData",
    "ContactViews",
    "contact_get_body1",
    "contact_get_body2",
    "contact_get_contact_count",
    "contact_get_contact_first",
    "contact_get_friction",
    "contact_get_friction_dynamic",
    "contact_iterate",
    "contact_iterate_at",
    "contact_pair_wrench_kernel",
    "contact_per_contact_error_kernel",
    "contact_per_contact_wrench_kernel",
    "contact_per_k_error_at",
    "contact_per_k_wrench_at",
    "contact_position_iterate",
    "contact_position_iterate_at",
    "contact_prepare_for_iteration",
    "contact_prepare_for_iteration_at",
    "contact_set_body1",
    "contact_set_body2",
    "contact_set_contact_count",
    "contact_set_contact_first",
    "contact_set_friction",
    "contact_set_friction_dynamic",
    "contact_views_make",
    "contact_world_error",
    "contact_world_error_at",
    "contact_world_wrench",
    "contact_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------


@wp.struct
class ContactConstraintData:
    """Column schema for one :data:`CONSTRAINT_TYPE_CONTACT` constraint.

    A single column now covers one whole ``(shape_a, shape_b)`` pair;
    the ``[contact_first, contact_first + contact_count)`` range spans
    every contact the narrow phase emitted for that pair in the current
    step's sorted :class:`Contacts` buffer. All per-contact state lives
    in :class:`ContactContainer` and is indexed by the contact's
    sorted-buffer index ``k``; iterate / prepare / position-iterate
    loop from ``contact_first`` to ``contact_first + contact_count``
    internally.
    """

    #: Tag -- :data:`CONSTRAINT_TYPE_CONTACT`. Must sit at dword 0 to
    #: satisfy the shared constraint-header contract.
    constraint_type: wp.int32
    #: Body index of body 1 (``model.shape_body[shape_a]``).
    body1: wp.int32
    #: Body index of body 2 (``model.shape_body[shape_b]``).
    body2: wp.int32

    #: Static Coulomb friction coefficient for the pair -- the "stick"
    #: threshold the per-iteration raw tangent impulse must stay under
    #: to keep the contact in the static-friction regime. Resolved at
    #: ingest time from the material table; falls back to
    #: ``default_friction`` when materials aren't configured.
    friction: wp.float32
    #: Kinetic (dynamic) Coulomb friction coefficient -- the clamp once
    #: the static threshold has been crossed. Typically ``<=`` static
    #: so slipping bodies decelerate less than they resisted starting
    #: the slide.
    friction_dynamic: wp.float32

    #: Start index into the sorted :class:`Contacts` buffer for the
    #: first contact of this column.
    contact_first: wp.int32
    #: Number of contacts belonging to this column; the loop body in
    #: each kernel runs from ``k = contact_first`` through
    #: ``k < contact_first + contact_count``.
    contact_count: wp.int32


assert_constraint_header(ContactConstraintData)


_OFF_BODY1 = wp.constant(dword_offset_of(ContactConstraintData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ContactConstraintData, "body2"))
_OFF_FRICTION = wp.constant(dword_offset_of(ContactConstraintData, "friction"))
_OFF_FRICTION_DYNAMIC = wp.constant(
    dword_offset_of(ContactConstraintData, "friction_dynamic")
)
_OFF_CONTACT_FIRST = wp.constant(
    dword_offset_of(ContactConstraintData, "contact_first")
)
_OFF_CONTACT_COUNT = wp.constant(
    dword_offset_of(ContactConstraintData, "contact_count")
)


#: Total dword count of one contact constraint column. Used by
#: :class:`ConstraintContainer` to size the shared storage's row
#: dimension. The per-pair layout drops every per-slot field so this
#: is now just the fixed header footprint.
CONTACT_DWORDS: int = num_dwords(ContactConstraintData)


# ---------------------------------------------------------------------------
# ContactViews -- bundle Newton Contacts arrays into a single wp.struct
# ---------------------------------------------------------------------------


@wp.struct
class ContactViews:
    """Kernel-visible view onto the fields of a Newton :class:`Contacts`
    buffer.

    Populated on the host once per :meth:`World.step` from the
    user-supplied :class:`Contacts` + the Newton model's ``shape_body``
    mapping. Only the subset of arrays actually consumed by the contact
    ingest / prepare / iterate / gather kernels is carried through.
    """

    #: Single-element active-count slice.
    rigid_contact_count: wp.array[wp.int32]

    #: Contact point on shape 0, expressed in shape 0's body frame.
    rigid_contact_point0: wp.array[wp.vec3f]
    #: Contact point on shape 1, expressed in shape 1's body frame.
    rigid_contact_point1: wp.array[wp.vec3f]
    #: World-space contact normal pointing from shape 0 -> shape 1
    #: (A-to-B convention). Unit length.
    rigid_contact_normal: wp.array[wp.vec3f]

    #: Shape indices for the pair.
    rigid_contact_shape0: wp.array[wp.int32]
    rigid_contact_shape1: wp.array[wp.int32]

    #: For current-frame contact ``k``, the index of the matched contact
    #: in the *previous* frame's sorted buffer, or
    #: :data:`MATCH_NOT_FOUND` / :data:`MATCH_BROKEN` for unmatched.
    rigid_contact_match_index: wp.array[wp.int32]

    #: Surface thickness for each shape (radius + skin margin); folded
    #: into the penetration depth so the constraint targets
    #: ``point0 - point1`` being at least ``margin0 + margin1`` apart.
    rigid_contact_margin0: wp.array[wp.float32]
    rigid_contact_margin1: wp.array[wp.float32]

    #: Newton model's per-shape body index (``model.shape_body``).
    shape_body: wp.array[wp.int32]


def contact_views_make(
    rigid_contact_count: wp.array,
    rigid_contact_point0: wp.array,
    rigid_contact_point1: wp.array,
    rigid_contact_normal: wp.array,
    rigid_contact_shape0: wp.array,
    rigid_contact_shape1: wp.array,
    rigid_contact_match_index: wp.array,
    rigid_contact_margin0: wp.array,
    rigid_contact_margin1: wp.array,
    shape_body: wp.array,
) -> ContactViews:
    """Build a :class:`ContactViews` from already-allocated Warp arrays.
    Pure struct pack on the host; no device allocations."""
    v = ContactViews()
    v.rigid_contact_count = rigid_contact_count
    v.rigid_contact_point0 = rigid_contact_point0
    v.rigid_contact_point1 = rigid_contact_point1
    v.rigid_contact_normal = rigid_contact_normal
    v.rigid_contact_shape0 = rigid_contact_shape0
    v.rigid_contact_shape1 = rigid_contact_shape1
    v.rigid_contact_match_index = rigid_contact_match_index
    v.rigid_contact_margin0 = rigid_contact_margin0
    v.rigid_contact_margin1 = rigid_contact_margin1
    v.shape_body = shape_body
    return v


# ---------------------------------------------------------------------------
# Typed accessors -- header fields
# ---------------------------------------------------------------------------


@wp.func
def contact_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def contact_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def contact_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def contact_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def contact_get_friction(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_FRICTION, cid)


@wp.func
def contact_set_friction(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_FRICTION, cid, v)


@wp.func
def contact_get_friction_dynamic(
    c: ConstraintContainer, cid: wp.int32
) -> wp.float32:
    return read_float(c, _OFF_FRICTION_DYNAMIC, cid)


@wp.func
def contact_set_friction_dynamic(
    c: ConstraintContainer, cid: wp.int32, v: wp.float32
):
    write_float(c, _OFF_FRICTION_DYNAMIC, cid, v)


@wp.func
def contact_get_contact_first(
    c: ConstraintContainer, cid: wp.int32
) -> wp.int32:
    return read_int(c, _OFF_CONTACT_FIRST, cid)


@wp.func
def contact_set_contact_first(
    c: ConstraintContainer, cid: wp.int32, v: wp.int32
):
    write_int(c, _OFF_CONTACT_FIRST, cid, v)


@wp.func
def contact_get_contact_count(
    c: ConstraintContainer, cid: wp.int32
) -> wp.int32:
    return read_int(c, _OFF_CONTACT_COUNT, cid)


@wp.func
def contact_set_contact_count(
    c: ConstraintContainer, cid: wp.int32, v: wp.int32
):
    write_int(c, _OFF_CONTACT_COUNT, cid, v)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


@wp.func
def _effective_mass_scalar(
    axis: wp.vec3f,
    r1: wp.vec3f,
    r2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
) -> wp.float32:
    """Scalar ``1 / J M^-1 J^T`` for an axis-aligned contact row."""
    rc1 = wp.cross(r1, axis)
    rc2 = wp.cross(r2, axis)
    w = (
        inv_mass1
        + inv_mass2
        + wp.dot(rc1, inv_inertia1 @ rc1)
        + wp.dot(rc2, inv_inertia2 @ rc2)
    )
    if w > 1.0e-12:
        return 1.0 / w
    return 0.0


# ---------------------------------------------------------------------------
# Prepare / iterate / position_iterate
# ---------------------------------------------------------------------------
#
# Every kernel below processes one whole shape pair per thread: it reads
# the column's ``(contact_first, contact_count)`` range and loops over
# the per-contact state in :class:`ContactContainer`. Body velocity
# reads / writes are batched -- the per-pair loop accumulates an impulse
# into thread-local velocity registers and scatters once at the end,
# so friction / normal cross-talk between contacts of the same pair
# stays Gauss-Seidel without extra global memory traffic.


@wp.func
def contact_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Prepare one contact column for the upcoming PGS iterations.

    For every contact ``k`` in the column's range:

      1. Re-project the stored body-frame anchors into world space
         using the current body transforms (PhoenX's
         ``UpdatePosition`` / TGS sub-step refresh).
      2. Compute lever arms ``r1 = p1 - com1`` / ``r2 = p2 - com2``.
      3. Compute scalar effective masses for the normal + two tangent
         rows and cache them in :class:`ContactContainer`'s ``derived``
         buffer.
      4. Compute the Baumgarte-style positional bias +
         static-friction tangent bias.
      5. Apply the warm-start impulse to the bodies' velocities so
         the PGS loop starts from a converged guess.

    The warm-start impulse scatter is batched: the loop accumulates
    the per-contact world-space impulse in local registers and
    applies it to the body velocities / angular velocities in one
    write after all contacts have been processed.

    ``base_offset`` is unused here (all contact state lives in
    :class:`ContactContainer`, not in the constraint column) but kept
    in the signature so the dispatcher can call this ``wp.func``
    uniformly with the fused joint types.
    """
    # Silence unused-parameter warnings; every contact column writes at
    # header offset 0 of its cid, so there's no sub-block offset to
    # thread through.

    b1 = body_pair.b1
    b2 = body_pair.b2

    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    # Newton's narrow phase expresses contact anchors in each body's
    # *origin* frame, but :attr:`BodyContainer.position` is the body's
    # *COM* in world space. Pull the per-body origin-to-COM offset once
    # so every ``local_p`` -> world projection below subtracts it and
    # the resulting lever arm ``(p - COM)`` lines up with what the
    # narrow phase actually sees. Zero for symmetric primitives; ~cm
    # for asymmetric meshes like the bunny.
    body_com1 = bodies.body_com[b1]
    body_com2 = bodies.body_com[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    # Box2D v3 / solver2d soft-constraint coefficients for the normal
    # row; same plumbing the joints use. ``hertz = DEFAULT_HERTZ_CONTACT``
    # saturates at the substep Nyquist rate (stiffest resolvable lock)
    # and ``damping_ratio = 1`` gives critical damping so the zero-
    # crossing oscillation from perfectly-rigid constraints is damped
    # out without needing a ``penetration_slop`` dead zone.
    dt_substep = wp.float32(1.0) / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    # Friction-row position bias. Sub-millimetre ``friction_slop`` clamps
    # raw drift before it drives the tangent bias; ``friction_bias_factor``
    # is dimensionless. Both are scale-invariant in length; the Coulomb
    # cone (|lam_t| <= mu * lam_n) still caps the final tangent impulse.
    friction_bias_factor = wp.float32(0.08)
    friction_slop = wp.float32(0.001)

    # Push / approach speed caps for the normal bias. Only numbers here
    # that aren't dimensionless -- scale up in scenes with scene_scale >> 1.
    max_push_speed = wp.float32(2.0)
    max_approach_speed = wp.float32(10.0)

    # Sticky-break thresholds.
    slip_threshold = wp.float32(0.002)

    # Accumulated warm-start impulse we'll apply to the bodies after
    # processing every contact; batched so the body-velocity scatter
    # happens once, not ``contact_count`` times.
    total_lin_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)
    total_ang_imp_on_b1 = wp.vec3f(0.0, 0.0, 0.0)
    total_ang_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)

    mu_s_col = contact_get_friction(constraints, cid)

    for i in range(contact_count):
        k = contact_first + i

        # Persistent contact frame + anchors. Matched contacts carry
        # these across frames; fresh contacts were PhoenX-``Initialize``-d
        # at gather time.
        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        local_p0 = cc_get_local_p0(cc, k)
        local_p1 = cc_get_local_p1(cc, k)

        # Body-local anchors -> world, including the per-shape surface
        # margin shift so the lever arm lands on the actual surface
        # contact point (not the body-frame anchor). Matters for
        # sphere / SDF primitives where the anchor is the body centre.
        # ``local_p`` is in the body-*origin* frame, so we subtract
        # ``body_com`` to express the lever arm from the COM (which
        # is what :attr:`bodies.position` tracks); otherwise asymmetric
        # meshes see the contact anchor shifted by ``body_com`` in
        # world space and the mesh visually sinks into the contact
        # surface.
        margin0 = contacts.rigid_contact_margin0[k]
        margin1 = contacts.rigid_contact_margin1[k]
        p1_world = (
            position1
            + wp.quat_rotate(orientation1, local_p0 - body_com1)
            + margin0 * n
        )
        p2_world = (
            position2
            + wp.quat_rotate(orientation2, local_p1 - body_com2)
            - margin1 * n
        )

        r1 = p1_world - position1
        r2 = p2_world - position2

        eff_n = _effective_mass_scalar(
            n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )
        eff_t1 = _effective_mass_scalar(
            t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )
        eff_t2 = _effective_mass_scalar(
            t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )

        effective_gap = wp.dot(p2_world - p1_world, n)

        # Load-scaled correction: contacts under heavier normal load
        # get stronger position-level correction. Uses the previous
        # substep's warm-started lam_n as a proxy for normal load,
        # capped at 4x to stop warm-start noise from producing runaway
        # corrections.
        lam_n_ws = cc_get_normal_lambda(cc, k)
        lam_n_ref = wp.float32(1.0) / wp.max(eff_n * idt, wp.float32(1.0e-6))
        load_boost = wp.min(
            wp.float32(1.0) + lam_n_ws / lam_n_ref, wp.float32(4.0)
        )

        # Speculative vs penetrating bias (Box2D v3 / solver2d model).
        # For a *separated* contact (``gap > 0``) we want the row to only
        # fire when the relative closing velocity would actually push
        # the bodies past each other within this substep -- i.e. when
        # ``-jv_n > gap / dt``. The target closing velocity is therefore
        # ``gap / dt = gap * idt``, so bodies closing slower than that
        # see no impulse (the clamp below fires) and bodies closing
        # faster get decelerated to exactly the rate that reaches zero
        # gap at the end of the substep. Using the softened ``bias_rate
        # ~ 0.6 / dt`` here instead (as the old code did) set the target
        # at ``gap * bias_rate``, which is slower than the "just-close-
        # in-one-step" rate and creates a soft spring that stops falling
        # bodies mid-air at the speculative gap distance -- the "honey"
        # artefact when a dropping mesh hits the speculative detection
        # shell and decelerates like it's in fluid.
        #
        # For a *penetrating* contact (``gap < 0``) we want the soft
        # Baumgarte push-apart the old formula produces, so we keep the
        # soft bias rate there. ``bias_val`` is always non-negative for
        # separated contacts and always non-positive for penetrating
        # contacts after the clamp; iterate() reads the sign to pick
        # rigid vs soft PGS coefficients.
        if effective_gap > wp.float32(0.0):
            bias_val = effective_gap * idt
        else:
            bias_val = effective_gap * bias_rate
        bias_val = wp.clamp(bias_val, -max_push_speed, max_approach_speed)

        # Sticky-friction drift + break. Mirrors solver2d:
        # break the anchor on drift / normal-rotation / Coulomb
        # saturation; reset to the fresh narrow-phase anchors and
        # (for Coulomb-saturation breaks) zero the tangent lambdas so
        # kinetic friction re-accumulates along the current slip axis.
        p_diff = p2_world - p1_world
        drift_t1_raw = wp.dot(p_diff, t1_dir)
        drift_t2_raw = wp.dot(p_diff, t2_dir)
        drift_sq = drift_t1_raw * drift_t1_raw + drift_t2_raw * drift_t2_raw

        fresh_n = contacts.rigid_contact_normal[k]
        normal_aligned = wp.dot(n, fresh_n)
        lam_t1_prev = cc_get_tangent1_lambda(cc, k)
        lam_t2_prev = cc_get_tangent2_lambda(cc, k)
        lam_n_prev = cc_get_normal_lambda(cc, k)
        fric_limit_prev = mu_s_col * lam_n_prev
        cone_margin = wp.float32(0.98)
        lam_t_mag_sq = lam_t1_prev * lam_t1_prev + lam_t2_prev * lam_t2_prev
        coulomb_saturated = (
            lam_n_prev > wp.float32(0.0)
            and lam_t_mag_sq
            >= (cone_margin * fric_limit_prev) * (cone_margin * fric_limit_prev)
        )
        if (
            drift_sq > slip_threshold * slip_threshold
            or normal_aligned < wp.float32(0.95)
            or coulomb_saturated
        ):
            fresh_lp0 = contacts.rigid_contact_point0[k]
            fresh_lp1 = contacts.rigid_contact_point1[k]
            cc_set_local_p0(cc, k, fresh_lp0)
            cc_set_local_p1(cc, k, fresh_lp1)
            if coulomb_saturated:
                cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
                cc_set_tangent2_lambda(cc, k, wp.float32(0.0))
            p1_world = (
                position1
                + wp.quat_rotate(orientation1, fresh_lp0 - body_com1)
                + margin0 * n
            )
            p2_world = (
                position2
                + wp.quat_rotate(orientation2, fresh_lp1 - body_com2)
                - margin1 * n
            )
            r1 = p1_world - position1
            r2 = p2_world - position2
            eff_n = _effective_mass_scalar(
                n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
            )
            eff_t1 = _effective_mass_scalar(
                t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
            )
            eff_t2 = _effective_mass_scalar(
                t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
            )
            p_diff = p2_world - p1_world
            drift_t1_raw = wp.dot(p_diff, t1_dir)
            drift_t2_raw = wp.dot(p_diff, t2_dir)

        drift_t1 = wp.clamp(drift_t1_raw, -friction_slop, friction_slop)
        drift_t2 = wp.clamp(drift_t2_raw, -friction_slop, friction_slop)
        bias_t1_val = friction_bias_factor * drift_t1 * idt * load_boost
        bias_t2_val = friction_bias_factor * drift_t2 * idt * load_boost

        cc_set_r1(cc, k, r1)
        cc_set_r2(cc, k, r2)
        cc_set_eff_n(cc, k, eff_n)
        cc_set_eff_t1(cc, k, eff_t1)
        cc_set_eff_t2(cc, k, eff_t2)
        cc_set_bias(cc, k, bias_val)
        cc_set_bias_t1(cc, k, bias_t1_val)
        cc_set_bias_t2(cc, k, bias_t2_val)

        # Warm-start impulse (current-buffer lambdas were seeded by the
        # gather kernel before prepare). Accumulate in world space and
        # scatter once after the full column has been processed.
        lam_n = cc_get_normal_lambda(cc, k)
        lam_t1 = cc_get_tangent1_lambda(cc, k)
        lam_t2 = cc_get_tangent2_lambda(cc, k)
        imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
        total_lin_imp_on_b2 += imp
        total_ang_imp_on_b1 += wp.cross(r1, imp)
        total_ang_imp_on_b2 += wp.cross(r2, imp)

    # Contact impulse acts from body 1 onto body 2 along +normal.
    bodies.velocity[b1] = bodies.velocity[b1] - inv_mass1 * total_lin_imp_on_b2
    bodies.velocity[b2] = bodies.velocity[b2] + inv_mass2 * total_lin_imp_on_b2
    bodies.angular_velocity[b1] = (
        bodies.angular_velocity[b1] - inv_inertia1 @ total_ang_imp_on_b1
    )
    bodies.angular_velocity[b2] = (
        bodies.angular_velocity[b2] + inv_inertia2 @ total_ang_imp_on_b2
    )


@wp.func
def contact_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
):
    """One PGS sweep over every contact of one shape pair.

    Sequential Gauss-Seidel within the pair: contacts are processed in
    order ``k = contact_first .. contact_first + contact_count`` and
    each contact's lambda update modifies thread-local velocity
    registers that the next contact's ``vel_rel`` reads. Body-velocity
    writes happen once at the end; graph coloring guarantees the
    plain store is race-free across threads.

    Per contact, three scalar PGS rows are solved: normal first (clamp
    ``lambda_n >= 0``), then the two tangent rows under the
    two-regime circular Coulomb cone.

    ``use_bias`` toggles the Box2D v3 TGS-soft behaviour: ``True``
    during the main solve (positional drift + speculative separation
    push velocity toward closing the gap), ``False`` during the relax
    pass so the normal row enforces ``Jv = 0`` without re-injecting
    the positional-correction velocity.
    """

    b1 = body_pair.b1
    b2 = body_pair.b2

    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    mu_s = contact_get_friction(constraints, cid)
    mu_k = contact_get_friction_dynamic(constraints, cid)

    dt_substep = wp.float32(1.0) / idt
    _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    v1 = bodies.velocity[b1]
    v2 = bodies.velocity[b2]
    w1 = bodies.angular_velocity[b1]
    w2 = bodies.angular_velocity[b2]

    for i in range(contact_count):
        k = contact_first + i

        r1 = cc_get_r1(cc, k)
        r2 = cc_get_r2(cc, k)
        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        eff_n = cc_get_eff_n(cc, k)
        eff_t1 = cc_get_eff_t1(cc, k)
        eff_t2 = cc_get_eff_t2(cc, k)
        bias_val = cc_get_bias(cc, k)
        bias_t1_val = cc_get_bias_t1(cc, k)
        bias_t2_val = cc_get_bias_t2(cc, k)
        # ``bias_val > 0`` marks a speculative contact (see
        # :func:`contact_prepare_for_iteration_at`); in that regime
        # Box2D's ``s > 0`` branch runs unconditionally of ``useBias``,
        # so the relax pass (``use_bias=False``) MUST keep the
        # ``gap * inv_dt`` bias, otherwise the row degenerates into a
        # pure ``-eff_n * jv_n`` push on every closing body -- which
        # applies a *large* decelerating impulse to anything falling
        # toward the surface while still separated. That's the honey
        # artefact: a speculative-contact shell that decelerates the
        # body even though the main solve correctly kept it rigid.
        # For penetrating contacts (``bias_val <= 0``) the relax pass
        # DOES zero the Baumgarte bias so the row settles on ``Jv = 0``
        # instead of re-injecting positional drift velocity.
        is_speculative = bias_val > wp.float32(0.0)
        if not use_bias:
            if not is_speculative:
                bias_val = wp.float32(0.0)
            bias_t1_val = wp.float32(0.0)
            bias_t2_val = wp.float32(0.0)

        # Relative velocity at the contact point (body 2 from body 1).
        # One vel_rel per contact; we project onto (n, t1, t2). The
        # three row axes are orthonormal so this stays algorithmically
        # equivalent to a full per-row GS re-projection while halving
        # the cross-product work.
        vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)

        jv_n = wp.dot(vel_rel, n)
        jv_t1 = wp.dot(vel_rel, t1_dir)
        jv_t2 = wp.dot(vel_rel, t2_dir)

        # Normal row: Box2D v3 soft-constraint solve + clamp. Three
        # regimes, matching ``b2SolveOverflowContacts`` and
        # ``b2SolveContactsTask`` in Box2D v3:
        #
        # 1. Speculative (``is_speculative``, i.e. ``effective_gap > 0``
        #    at prepare time): rigid PGS (``mass_coeff = 1``,
        #    ``impulse_coeff = 0``) + bias ``gap * inv_dt`` -- runs
        #    unconditionally of ``use_bias`` so the relax pass keeps
        #    capping closing at ``gap / dt`` instead of degenerating
        #    into a pure ``-eff_n * jv_n`` brake.
        # 2. Penetrating + main solve (``!is_speculative`` and
        #    ``use_bias``): soft PGS with the Box2D rollover
        #    (``mass_coeff``, ``impulse_coeff`` from
        #    :func:`soft_constraint_coefficients`).
        # 3. Penetrating + relax (``!is_speculative`` and not
        #    ``use_bias``): rigid PGS with zero bias -- pure
        #    ``Jv = 0`` enforcement without the positional-bias
        #    velocity that the main solve just injected.
        if is_speculative:
            mass_coeff_n = wp.float32(1.0)
            impulse_coeff_n = wp.float32(0.0)
        elif use_bias:
            mass_coeff_n = mass_coeff
            impulse_coeff_n = impulse_coeff
        else:
            mass_coeff_n = wp.float32(1.0)
            impulse_coeff_n = wp.float32(0.0)
        d_lam_n_us = -eff_n * (jv_n + bias_val)
        lam_n_old = cc_get_normal_lambda(cc, k)
        d_lam_n = mass_coeff_n * d_lam_n_us - impulse_coeff_n * lam_n_old
        lam_n_new = wp.max(lam_n_old + d_lam_n, wp.float32(0.0))
        d_lam_n = lam_n_new - lam_n_old

        fric_limit_static = mu_s * lam_n_new
        fric_limit_kinetic = mu_k * lam_n_new

        d_lam_t1 = -eff_t1 * (jv_t1 + bias_t1_val)
        d_lam_t2 = -eff_t2 * (jv_t2 + bias_t2_val)
        lam_t1_old = cc_get_tangent1_lambda(cc, k)
        lam_t2_old = cc_get_tangent2_lambda(cc, k)
        lam_t1_raw = lam_t1_old + d_lam_t1
        lam_t2_raw = lam_t2_old + d_lam_t2

        # Two-regime circular Coulomb cone.
        lam_t_sq = lam_t1_raw * lam_t1_raw + lam_t2_raw * lam_t2_raw
        static_limit_sq = fric_limit_static * fric_limit_static
        if lam_t_sq > static_limit_sq and lam_t_sq > wp.float32(1.0e-30):
            inv_mag = fric_limit_kinetic / wp.sqrt(lam_t_sq)
            lam_t1_new = lam_t1_raw * inv_mag
            lam_t2_new = lam_t2_raw * inv_mag
        else:
            lam_t1_new = lam_t1_raw
            lam_t2_new = lam_t2_raw
        d_lam_t1 = lam_t1_new - lam_t1_old
        d_lam_t2 = lam_t2_new - lam_t2_old

        cc_set_normal_lambda(cc, k, lam_n_new)
        cc_set_tangent1_lambda(cc, k, lam_t1_new)
        cc_set_tangent2_lambda(cc, k, lam_t2_new)

        imp = d_lam_n * n + d_lam_t1 * t1_dir + d_lam_t2 * t2_dir
        v1 -= inv_mass1 * imp
        v2 += inv_mass2 * imp
        w1 -= inv_inertia1 @ wp.cross(r1, imp)
        w2 += inv_inertia2 @ wp.cross(r2, imp)

    bodies.velocity[b1] = v1
    bodies.velocity[b2] = v2
    bodies.angular_velocity[b1] = w1
    bodies.angular_velocity[b2] = w2


@wp.func
def contact_position_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    cc: ContactContainer,
):
    """XPBD-style position iteration for contact tangent drift.

    One sweep of position-level PGS over every contact of the column's
    shape pair. Modifies body positions / orientations (not velocities)
    so static-friction drift converges to zero within the iteration
    budget, bypassing the Nyquist-rate ceiling that velocity-level
    Baumgarte hits.

    Sliding contacts (drift > slip_threshold) are gated out and handled
    by the velocity friction path; resting-velocity contacts are the
    only ones this kernel touches. Per-body deltas accumulate across
    contacts of the column and apply once at the end.
    """

    b1 = body_pair.b1
    b2 = body_pair.b2

    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    if inv_mass1 + inv_mass2 < wp.float32(1.0e-12):
        return

    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    body_com1 = bodies.body_com[b1]
    body_com2 = bodies.body_com[b2]
    v1 = bodies.velocity[b1]
    v2 = bodies.velocity[b2]
    w1 = bodies.angular_velocity[b1]
    w2 = bodies.angular_velocity[b2]

    slip_threshold = wp.float32(0.002)
    rest_vel_thresh = wp.float32(0.005)

    d_pos1 = wp.vec3f(0.0, 0.0, 0.0)
    d_pos2 = wp.vec3f(0.0, 0.0, 0.0)
    d_omega1 = wp.vec3f(0.0, 0.0, 0.0)
    d_omega2 = wp.vec3f(0.0, 0.0, 0.0)

    for i in range(contact_count):
        k = contact_first + i

        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        local_p0 = cc_get_local_p0(cc, k)
        local_p1 = cc_get_local_p1(cc, k)

        # ``local_p`` is in body-origin frame; ``position`` is the
        # COM. Subtract ``body_com`` before rotating so the lever arm
        # is expressed from the COM (see
        # :func:`contact_prepare_for_iteration_at` for the convention
        # rationale).
        p1_world = position1 + wp.quat_rotate(orientation1, local_p0 - body_com1)
        p2_world = position2 + wp.quat_rotate(orientation2, local_p1 - body_com2)
        r1 = p1_world - position1
        r2 = p2_world - position2

        p_diff = p2_world - p1_world
        drift_t1 = wp.dot(p_diff, t1_dir)
        drift_t2 = wp.dot(p_diff, t2_dir)

        drift_sq = drift_t1 * drift_t1 + drift_t2 * drift_t2
        if drift_sq > slip_threshold * slip_threshold:
            continue

        vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)
        vel_t1 = wp.dot(vel_rel, t1_dir)
        vel_t2 = wp.dot(vel_rel, t2_dir)
        if vel_t1 * vel_t1 + vel_t2 * vel_t2 > rest_vel_thresh * rest_vel_thresh:
            continue

        eff_t1 = _effective_mass_scalar(
            t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )
        eff_t2 = _effective_mass_scalar(
            t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )

        d_lam_t1 = -drift_t1 * eff_t1
        d_lam_t2 = -drift_t2 * eff_t2
        pos_imp = d_lam_t1 * t1_dir + d_lam_t2 * t2_dir

        d_pos1 = d_pos1 - inv_mass1 * pos_imp
        d_pos2 = d_pos2 + inv_mass2 * pos_imp
        d_omega1 = d_omega1 - inv_inertia1 @ wp.cross(r1, pos_imp)
        d_omega2 = d_omega2 + inv_inertia2 @ wp.cross(r2, pos_imp)

    bodies.position[b1] = position1 + d_pos1
    bodies.position[b2] = position2 + d_pos2

    dq1 = wp.quat(d_omega1[0], d_omega1[1], d_omega1[2], wp.float32(0.0))
    new_q1 = orientation1 + wp.float32(0.5) * (dq1 * orientation1)
    bodies.orientation[b1] = wp.normalize(new_q1)

    dq2 = wp.quat(d_omega2[0], d_omega2[1], d_omega2[2], wp.float32(0.0))
    new_q2 = orientation2 + wp.float32(0.5) * (dq2 * orientation2)
    bodies.orientation[b2] = wp.normalize(new_q2)


# ---------------------------------------------------------------------------
# Direct entry points (base_offset = 0)
# ---------------------------------------------------------------------------


@wp.func
def contact_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at(
        constraints, cid, 0, bodies, body_pair, idt, cc, contacts
    )


@wp.func
def contact_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at(
        constraints, cid, 0, bodies, body_pair, idt, cc, contacts, use_bias
    )


@wp.func
def contact_position_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    cc: ContactContainer,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_position_iterate_at(constraints, cid, 0, bodies, body_pair, cc)


# ---------------------------------------------------------------------------
# Per-column / per-contact wrench + error reporting
# ---------------------------------------------------------------------------


@wp.func
def contact_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Column-summed world-frame wrench this contact pair applies to body 2.

    Sums the per-contact impulse ``lambda_n * n + lambda_t1 * t1 +
    lambda_t2 * t2`` across the column's ``contact_count`` contacts
    and divides by ``dt`` for a force. Torque is computed against
    body 2's COM using the cached lever arm ``r2``.
    """
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(contact_count):
        k = contact_first + i
        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        r2 = cc_get_r2(cc, k)
        lam_n = cc_get_normal_lambda(cc, k)
        lam_t1 = cc_get_tangent1_lambda(cc, k)
        lam_t2 = cc_get_tangent2_lambda(cc, k)
        imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
        f = imp * idt
        force += f
        torque += wp.cross(r2, f)
    return force, torque


@wp.func
def contact_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    return contact_world_wrench_at(constraints, cid, 0, idt, cc, contacts)


@wp.func
def contact_per_k_wrench_at(
    k: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
):
    """World-frame wrench on body 2 for one contact ``k``.

    ``force`` is ``imp / dt`` where ``imp = lam_n n + lam_t1 t1 +
    lam_t2 t2`` is the world-space impulse the contact applied during
    the most recent substep; ``torque`` is the moment about body 2's
    COM using the cached lever arm ``r2``.
    """
    n = cc_get_normal(cc, k)
    t1_dir = cc_get_tangent1(cc, k)
    t2_dir = wp.cross(n, t1_dir)
    r2 = cc_get_r2(cc, k)
    lam_n = cc_get_normal_lambda(cc, k)
    lam_t1 = cc_get_tangent1_lambda(cc, k)
    lam_t2 = cc_get_tangent2_lambda(cc, k)
    imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
    force = imp * idt
    torque = wp.cross(r2, force)
    return force, torque


@wp.kernel(enable_backward=False)
def contact_per_contact_wrench_kernel(
    constraints: ConstraintContainer,
    cc: ContactContainer,
    contacts: ContactViews,
    cid_base: wp.int32,
    cid_capacity: wp.int32,
    idt: wp.float32,
    # out
    out: wp.array[wp.spatial_vector],
):
    """Scatter per-contact wrenches into a per-rigid-contact output array.

    Launch dim must be at least ``cid_capacity - cid_base`` (one
    thread per potential contact column). Each thread walks the
    ``[contact_first, contact_first + contact_count)`` range and writes
    ``out[k] = spatial_vector(force, torque)`` for every contact ``k``
    it covers. Contacts outside any active column retain the caller's
    zero-fill.
    """
    tid = wp.tid()
    cid = cid_base + tid
    if cid >= cid_capacity:
        return
    if constraint_get_type(constraints, cid) != CONSTRAINT_TYPE_CONTACT:
        return
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    for i in range(contact_count):
        k = contact_first + i
        f, tau = contact_per_k_wrench_at(k, idt, cc)
        out[k] = wp.spatial_vector(f, tau)


@wp.kernel(enable_backward=False)
def contact_pair_wrench_kernel(
    constraints: ConstraintContainer,
    cc: ContactContainer,
    contacts: ContactViews,
    cid_base: wp.int32,
    cid_capacity: wp.int32,
    idt: wp.float32,
    # out
    wrenches: wp.array[wp.spatial_vector],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    contact_count_out: wp.array[wp.int32],
):
    """Per-column summary: total wrench + ``(body1, body2)`` + contact count.

    One output slot ``i = cid - cid_base`` per contact column. The
    per-pair design means one column now covers the whole shape pair,
    so the per-column output is already the per-pair wrench -- no
    post-pass summation across adjacent columns needed.
    """
    tid = wp.tid()
    cid = cid_base + tid
    if cid >= cid_capacity:
        return
    if constraint_get_type(constraints, cid) != CONSTRAINT_TYPE_CONTACT:
        wrenches[tid] = wp.spatial_vector(
            wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0)
        )
        body1[tid] = -1
        body2[tid] = -1
        contact_count_out[tid] = 0
        return
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    count = contact_get_contact_count(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(count):
        k = contact_first + i
        f, tau = contact_per_k_wrench_at(k, idt, cc)
        force += f
        torque += tau
    wrenches[tid] = wp.spatial_vector(force, torque)
    body1[tid] = b1
    body2[tid] = b2
    contact_count_out[tid] = count


# ---------------------------------------------------------------------------
# Per-contact constraint error
# ---------------------------------------------------------------------------


@wp.func
def contact_per_k_error_at(
    k: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    cc: ContactContainer,
    contacts: ContactViews,
) -> wp.vec3f:
    """Position-level residual for one contact ``k``.

    Returns ``(gap, drift_t1, drift_t2)`` -- the raw residuals the
    prepare kernel folds into the normal + tangent biases, surfaced
    here without the ``friction_slop`` clamp so the caller sees the
    exact drift.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    body_com1 = bodies.body_com[b1]
    body_com2 = bodies.body_com[b2]

    n = cc_get_normal(cc, k)
    t1_dir = cc_get_tangent1(cc, k)
    t2_dir = wp.cross(n, t1_dir)
    local_p0 = cc_get_local_p0(cc, k)
    local_p1 = cc_get_local_p1(cc, k)

    # ``local_p`` in body-origin frame, ``position`` is COM -- subtract
    # ``body_com`` so the residual gap matches the real world-space
    # separation (see :func:`contact_prepare_for_iteration_at`).
    p1_world = position1 + wp.quat_rotate(orientation1, local_p0 - body_com1)
    p2_world = position2 + wp.quat_rotate(orientation2, local_p1 - body_com2)
    p_diff = p2_world - p1_world

    margin_sum = contacts.rigid_contact_margin0[k] + contacts.rigid_contact_margin1[k]
    gap = wp.dot(p_diff, n) - margin_sum
    drift_t1 = wp.dot(p_diff, t1_dir)
    drift_t2 = wp.dot(p_diff, t2_dir)
    return wp.vec3f(gap, drift_t1, drift_t2)


@wp.kernel(enable_backward=False)
def contact_per_contact_error_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cc: ContactContainer,
    contacts: ContactViews,
    cid_base: wp.int32,
    cid_capacity: wp.int32,
    # out
    out: wp.array[wp.vec3f],
):
    """Scatter per-contact residuals into a per-rigid-contact output array.

    Companion to :func:`contact_per_contact_wrench_kernel`: same launch
    contract, writes ``out[k] = (gap, drift_t1, drift_t2)`` for every
    contact ``k`` in every active contact column.
    """
    tid = wp.tid()
    cid = cid_base + tid
    if cid >= cid_capacity:
        return
    if constraint_get_type(constraints, cid) != CONSTRAINT_TYPE_CONTACT:
        return
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    for i in range(contact_count):
        k = contact_first + i
        out[k] = contact_per_k_error_at(k, bodies, body_pair, cc, contacts)


@wp.func
def contact_world_error_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
) -> wp.spatial_vector:
    """Per-column constraint residual placeholder for contact cids.

    Contact columns report zero through the per-constraint dispatcher
    because meaningful contact residuals are per-contact (populated by
    :func:`contact_per_contact_error_kernel`). Returning zero here keeps
    the generic error gather branch uniform with the other constraint
    types without contaminating the output with an arbitrary aggregate.
    """
    return wp.spatial_vector(
        wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0)
    )


@wp.func
def contact_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
) -> wp.spatial_vector:
    return contact_world_error_at(constraints, cid, 0)
