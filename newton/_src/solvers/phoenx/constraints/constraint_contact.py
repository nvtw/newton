# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rigid-rigid contact constraint for :class:`PhoenXWorld`.

One :data:`CONSTRAINT_TYPE_CONTACT` column covers one whole
``(shape_a, shape_b)`` pair, storing only the contiguous index range
``[contact_first, contact_first + contact_count)`` into the sorted
:class:`newton._src.sim.contacts.Contacts` buffer. Per-contact
persistent state (lambdas, contact frame, body-local anchors) and
per-substep derived quantities (lever arms, effective masses, bias)
live in :class:`ContactContainer` keyed by the contact's
sorted-buffer index ``k``. The PGS loop walks each pair's contact
range serially inside one kernel call, which is Gauss-Seidel within
the pair by construction.

Friction: two-tangent pyramidal PGS. Each contact contributes a
normal row (``lambda_n >= 0``) plus two tangential rows clamped to
``[-mu * lambda_n, +mu * lambda_n]``. Same "simple friction" pattern
as Box2D v3; the full Coulomb cone is approximated by the circular
clamp inside the iterate kernel.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_access_mode import ConstraintAccessMode
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    assert_constraint_header,
    constraint_bodies_make,
    pd_coefficients,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    endpoint_apply_impulse,
    endpoint_load,
    endpoint_warmstart_apply_impulse,
)
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.constraints.contact_container import (
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
    cc_get_pd_bias,
    cc_get_pd_eff_soft,
    cc_get_pd_gamma,
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
    cc_set_pd_bias,
    cc_set_pd_eff_soft,
    cc_set_pd_gamma,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.helpers.data_packing import (
    dword_offset_of,
    num_dwords,
    reinterpret_float_as_int,
    reinterpret_int_as_float,
)
from newton._src.solvers.phoenx.helpers.math_helpers import (
    effective_mass_scalar,
)
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_BOOST_CONTACT_NORMAL,
)

__all__ = [
    "ACCESS_MODE",
    "CONTACT_DWORDS",
    "ContactColumnContainer",
    "ContactConstraintData",
    "ContactViews",
    "contact_column_container_zeros",
    "contact_get_body1",
    "contact_get_body2",
    "contact_get_contact_count",
    "contact_get_contact_first",
    "contact_get_friction",
    "contact_get_friction_dynamic",
    "contact_iterate",
    "contact_iterate_at",
    "contact_iterate_at_multi",
    "contact_iterate_multi",
    "contact_pair_wrench_kernel",
    "contact_per_contact_error_kernel",
    "contact_per_contact_wrench_kernel",
    "contact_per_k_error_at",
    "contact_per_k_wrench_at",
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


#: Velocity-level: ``contact_iterate_at`` applies impulses to
#: ``bodies.velocity`` / ``angular_velocity``. See
#: :file:`CONSTRAINT_ACCESS_MODE.md`.
ACCESS_MODE = ConstraintAccessMode.VELOCITY_LEVEL


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
    #: Body index of body 1. RIGID endpoint: ``model.shape_body[shape_a]``.
    #: TRIANGLE endpoint: ``0`` (anchor); the real triangle index lives
    #: in :attr:`endpoint_idx1`.
    body1: wp.int32
    #: Body index of body 2. Same convention as :attr:`body1`.
    body2: wp.int32

    #: Static Coulomb friction coefficient for the pair.
    friction: wp.float32
    #: Kinetic (dynamic) Coulomb friction coefficient.
    friction_dynamic: wp.float32

    #: Start / count into the sorted :class:`Contacts` buffer.
    contact_first: wp.int32
    contact_count: wp.int32

    #: Endpoint kind tag for endpoint 1: :data:`ENDPOINT_KIND_RIGID` or
    #: :data:`ENDPOINT_KIND_TRIANGLE`. See :mod:`contact_endpoint`.
    endpoint_kind1: wp.int32
    #: Endpoint index for endpoint 1. Body index for RIGID; triangle
    #: index ``t in [0, T)`` for TRIANGLE.
    endpoint_idx1: wp.int32
    #: Endpoint kind / idx for endpoint 2. Same convention as
    #: endpoint 1.
    endpoint_kind2: wp.int32
    endpoint_idx2: wp.int32


assert_constraint_header(ContactConstraintData)


_OFF_BODY1 = wp.constant(dword_offset_of(ContactConstraintData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ContactConstraintData, "body2"))
_OFF_FRICTION = wp.constant(dword_offset_of(ContactConstraintData, "friction"))
_OFF_FRICTION_DYNAMIC = wp.constant(dword_offset_of(ContactConstraintData, "friction_dynamic"))
_OFF_CONTACT_FIRST = wp.constant(dword_offset_of(ContactConstraintData, "contact_first"))
_OFF_CONTACT_COUNT = wp.constant(dword_offset_of(ContactConstraintData, "contact_count"))
_OFF_ENDPOINT_KIND1 = wp.constant(dword_offset_of(ContactConstraintData, "endpoint_kind1"))
_OFF_ENDPOINT_IDX1 = wp.constant(dword_offset_of(ContactConstraintData, "endpoint_idx1"))
_OFF_ENDPOINT_KIND2 = wp.constant(dword_offset_of(ContactConstraintData, "endpoint_kind2"))
_OFF_ENDPOINT_IDX2 = wp.constant(dword_offset_of(ContactConstraintData, "endpoint_idx2"))


#: Total dword count of one contact constraint column. The per-pair
#: layout drops every per-slot field so the column carries only the
#: fixed header (type tag, body / endpoint indices, friction, contact
#: range). 11 dwords vs. the 154-dword ADBS joint header.
CONTACT_DWORDS: int = num_dwords(ContactConstraintData)


# ---------------------------------------------------------------------------
# ContactColumnContainer -- dedicated storage for the contact column header
# ---------------------------------------------------------------------------
#
# Narrow sibling of :class:`ConstraintContainer` for contact cids
# only. A single wide container would have wasted ~1.4 GB at h1_flat
# 4096 on padding (contacts use 7 of 154 dwords). Contact cids are
# local here: ``local_cid = global_cid - num_joints``, with the
# subtraction done once at the dispatch branch.
#
# Layout mirrors :class:`ConstraintContainer`: one
# ``wp.array2d[wp.float32]`` of shape ``(CONTACT_DWORDS, num_columns)``
# with cid on the inner axis, and one ``wp.func`` per field.


@wp.struct
class ContactColumnContainer:
    """Per-shape-pair column header storage for contacts.

    Shape ``(CONTACT_DWORDS=7, num_columns)``. Each column occupies
    7 dwords: type tag (unused but kept for header-contract parity),
    body1, body2, friction, friction_dynamic, contact_first,
    contact_count. Per-contact solver state (lambdas, lever arms,
    effective masses, bias) lives in :class:`ContactContainer` keyed
    by the individual contact's sorted-buffer index ``k``.
    """

    data: wp.array2d[wp.float32]


def contact_column_container_zeros(
    num_columns: int,
    device: wp.DeviceLike = None,
) -> ContactColumnContainer:
    """Allocate a zero-initialised :class:`ContactColumnContainer`.

    Args:
        num_columns: Capacity (= ``max_contact_columns``). Contact
            cid ``global_cid`` in ``[num_joints, num_joints + num_columns)``
            maps to ``local_cid = global_cid - num_joints`` here.
        device: Warp device.
    """
    c = ContactColumnContainer()
    c.data = wp.zeros((int(CONTACT_DWORDS), int(num_columns)), dtype=wp.float32, device=device)
    return c


# Column-local read / write helpers, analogous to the
# ``read_int / read_float / write_int / write_float`` helpers in
# :mod:`constraint_container` but keyed by ``local_cid`` into
# :class:`ContactColumnContainer`. Kept as module-private so kernels
# never accidentally hit the wide joint container for contact data.
@wp.func
def _col_read_int(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32) -> wp.int32:
    return reinterpret_float_as_int(c.data[off, local_cid])


@wp.func
def _col_write_int(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32, v: wp.int32):
    c.data[off, local_cid] = reinterpret_int_as_float(v)


@wp.func
def _col_read_float(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32) -> wp.float32:
    return c.data[off, local_cid]


@wp.func
def _col_write_float(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32, v: wp.float32):
    c.data[off, local_cid] = v


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

    #: Per-contact absolute stiffness [N/m] written by narrow phases
    #: that support soft contacts (e.g. hydroelastic). ``0`` at a
    #: contact slot means "use the legacy Box2D hertz-based normal
    #: row". Allocated by :class:`~newton.Contacts` only when
    #: ``per_contact_shape_properties=True``; when the user-supplied
    #: ``Contacts`` buffer didn't allocate it, the solver installs a
    #: zero-length sentinel and every contact falls through to the
    #: legacy path.
    rigid_contact_stiffness: wp.array[wp.float32]
    #: Per-contact absolute damping [N*s/m]. Same semantics as
    #: :attr:`rigid_contact_stiffness`: non-zero switches the normal
    #: row to the absolute PD spring-damper path.
    rigid_contact_damping: wp.array[wp.float32]
    #: Per-contact friction coefficient override [dimensionless]. When
    #: the allocated Contacts array has a positive value at slot ``k``,
    #: the solver uses it instead of the material-table entry (lets
    #: hydroelastic and other per-contact narrow phases publish
    #: pressure-dependent friction). ``0`` or the sentinel buffer
    #: means "fall back to the material table".
    rigid_contact_friction: wp.array[wp.float32]


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
    rigid_contact_stiffness: wp.array,
    rigid_contact_damping: wp.array,
    rigid_contact_friction: wp.array,
) -> ContactViews:
    """Build a :class:`ContactViews` from already-allocated Warp arrays.
    Pure struct pack on the host; no device allocations.

    The three soft-contact arrays (stiffness / damping / friction) are
    optional on Newton's :class:`~newton.Contacts`; the caller must
    substitute a zero-length sentinel array (same dtype) when they're
    unset, so every :class:`ContactViews` field stays addressable.
    """
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
    v.rigid_contact_stiffness = rigid_contact_stiffness
    v.rigid_contact_damping = rigid_contact_damping
    v.rigid_contact_friction = rigid_contact_friction
    return v


# ---------------------------------------------------------------------------
# Typed accessors -- header fields
# ---------------------------------------------------------------------------


@wp.func
def contact_get_body1(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_BODY1, local_cid)


@wp.func
def contact_set_body1(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY1, local_cid, v)


@wp.func
def contact_get_body2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_BODY2, local_cid)


@wp.func
def contact_set_body2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY2, local_cid, v)


@wp.func
def contact_get_friction(c: ContactColumnContainer, local_cid: wp.int32) -> wp.float32:
    return _col_read_float(c, _OFF_FRICTION, local_cid)


@wp.func
def contact_set_friction(c: ContactColumnContainer, local_cid: wp.int32, v: wp.float32):
    _col_write_float(c, _OFF_FRICTION, local_cid, v)


@wp.func
def contact_get_friction_dynamic(c: ContactColumnContainer, local_cid: wp.int32) -> wp.float32:
    return _col_read_float(c, _OFF_FRICTION_DYNAMIC, local_cid)


@wp.func
def contact_set_friction_dynamic(c: ContactColumnContainer, local_cid: wp.int32, v: wp.float32):
    _col_write_float(c, _OFF_FRICTION_DYNAMIC, local_cid, v)


@wp.func
def contact_get_contact_first(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_CONTACT_FIRST, local_cid)


@wp.func
def contact_set_contact_first(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_CONTACT_FIRST, local_cid, v)


@wp.func
def contact_get_contact_count(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_CONTACT_COUNT, local_cid)


@wp.func
def contact_set_contact_count(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_CONTACT_COUNT, local_cid, v)


@wp.func
def contact_get_endpoint_kind1(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_ENDPOINT_KIND1, local_cid)


@wp.func
def contact_set_endpoint_kind1(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_ENDPOINT_KIND1, local_cid, v)


@wp.func
def contact_get_endpoint_idx1(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_ENDPOINT_IDX1, local_cid)


@wp.func
def contact_set_endpoint_idx1(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_ENDPOINT_IDX1, local_cid, v)


@wp.func
def contact_get_endpoint_kind2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_ENDPOINT_KIND2, local_cid)


@wp.func
def contact_set_endpoint_kind2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_ENDPOINT_KIND2, local_cid, v)


@wp.func
def contact_get_endpoint_idx2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_ENDPOINT_IDX2, local_cid)


@wp.func
def contact_set_endpoint_idx2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_ENDPOINT_IDX2, local_cid, v)


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
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    """Prepare one contact column for the upcoming PGS iterations.

    Endpoint-dispatched: each endpoint may be rigid or triangle.
    :func:`endpoint_load` reconstructs the world contact point + per-
    endpoint velocity / lever-arm / mass; the lambda math below is
    endpoint-agnostic. Warm-start impulse scatter happens per-contact
    via :func:`endpoint_warmstart_apply_impulse` (rigid endpoints lose
    the batched scatter optimisation, but the math is identical).
    """
    # ``base_offset`` / ``body_pair`` retained for signature compat
    # with the joint dispatcher.

    kind1 = contact_get_endpoint_kind1(constraints, cid)
    kind2 = contact_get_endpoint_kind2(constraints, cid)
    idx1 = contact_get_endpoint_idx1(constraints, cid)
    idx2 = contact_get_endpoint_idx2(constraints, cid)

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
    bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(
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

    mu_s_col = contact_get_friction(constraints, cid)
    two_pi = wp.float32(6.283185307179586)

    for i in range(contact_count):
        k = contact_first + i

        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        local_p0 = cc_get_local_p0(cc, k)
        local_p1 = cc_get_local_p1(cc, k)

        margin0 = contacts.rigid_contact_margin0[k]
        margin1 = contacts.rigid_contact_margin1[k]
        # Per-endpoint world contact point + velocity / mass. RIGID
        # consumes a body-origin-frame anchor; TRIANGLE consumes
        # barycentric weights ``(w_a, w_b, w_c)`` packed as a vec3.
        # ``margin_sign``: +1 for endpoint 1 (push along +n), -1 for
        # endpoint 2 (push along -n) -- matches the original rigid
        # form ``+ margin0 * n`` / ``- margin1 * n``.
        ep1 = endpoint_load(kind1, idx1, local_p0, margin0, wp.float32(1.0), n, bodies, particles, tri_indices)
        ep2 = endpoint_load(kind2, idx2, local_p1, margin1, wp.float32(-1.0), n, bodies, particles, tri_indices)

        eff_n = effective_mass_scalar(
            n, ep1.r, ep2.r, ep1.inv_mass, ep2.inv_mass, ep1.inv_inertia, ep2.inv_inertia
        )
        eff_t1 = effective_mass_scalar(
            t1_dir, ep1.r, ep2.r, ep1.inv_mass, ep2.inv_mass, ep1.inv_inertia, ep2.inv_inertia
        )
        eff_t2 = effective_mass_scalar(
            t2_dir, ep1.r, ep2.r, ep1.inv_mass, ep2.inv_mass, ep1.inv_inertia, ep2.inv_inertia
        )

        effective_gap = wp.dot(ep2.p_world - ep1.p_world, n)

        # Load-scaled correction: contacts under heavier normal load
        # get stronger position-level correction. Uses the previous
        # substep's warm-started lam_n as a proxy for normal load,
        # capped at 4x to stop warm-start noise from producing runaway
        # corrections.
        lam_n_ws = cc_get_normal_lambda(cc, k)
        lam_n_ref = wp.float32(1.0) / wp.max(eff_n * idt, wp.float32(1.0e-6))
        load_boost = wp.min(wp.float32(1.0) + lam_n_ws / lam_n_ref, wp.float32(4.0))

        # Speculative vs penetrating bias (Box2D v3 / solver2d).
        # Separated (``gap > 0``): target closing velocity ``gap*idt``
        # so the row only fires when ``-jv_n > gap/dt``. Using the
        # soft ``bias_rate ~ 0.6/dt`` here (as old code did) creates a
        # "honey" artefact where dropping bodies decelerate at the
        # speculative shell as if in fluid.
        # Penetrating (``gap < 0``): keep the soft Baumgarte push-
        # apart. ``bias_val`` ends up >= 0 for separated and <= 0 for
        # penetrating contacts; iterate() reads the sign to pick
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
        p_diff = ep2.p_world - ep1.p_world
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
        coulomb_saturated = lam_n_prev > wp.float32(0.0) and lam_t_mag_sq >= (cone_margin * fric_limit_prev) * (
            cone_margin * fric_limit_prev
        )
        if drift_sq > slip_threshold * slip_threshold or normal_aligned < wp.float32(0.95) or coulomb_saturated:
            fresh_lp0 = contacts.rigid_contact_point0[k]
            fresh_lp1 = contacts.rigid_contact_point1[k]
            cc_set_local_p0(cc, k, fresh_lp0)
            cc_set_local_p1(cc, k, fresh_lp1)
            if coulomb_saturated:
                cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
                cc_set_tangent2_lambda(cc, k, wp.float32(0.0))
            ep1 = endpoint_load(kind1, idx1, fresh_lp0, margin0, wp.float32(1.0), n, bodies, particles, tri_indices)
            ep2 = endpoint_load(kind2, idx2, fresh_lp1, margin1, wp.float32(-1.0), n, bodies, particles, tri_indices)
            eff_n = effective_mass_scalar(
                n, ep1.r, ep2.r, ep1.inv_mass, ep2.inv_mass, ep1.inv_inertia, ep2.inv_inertia
            )
            eff_t1 = effective_mass_scalar(
                t1_dir, ep1.r, ep2.r, ep1.inv_mass, ep2.inv_mass, ep1.inv_inertia, ep2.inv_inertia
            )
            eff_t2 = effective_mass_scalar(
                t2_dir, ep1.r, ep2.r, ep1.inv_mass, ep2.inv_mass, ep1.inv_inertia, ep2.inv_inertia
            )
            p_diff = ep2.p_world - ep1.p_world
            drift_t1_raw = wp.dot(p_diff, t1_dir)
            drift_t2_raw = wp.dot(p_diff, t2_dir)

        drift_t1 = wp.clamp(drift_t1_raw, -friction_slop, friction_slop)
        drift_t2 = wp.clamp(drift_t2_raw, -friction_slop, friction_slop)
        bias_t1_val = friction_bias_factor * drift_t1 * idt * load_boost
        bias_t2_val = friction_bias_factor * drift_t2 * idt * load_boost

        # r1 / r2 used to be cached here; we recompute them inline in
        # iterate from local_p0 / local_p1 (already in lambdas) plus
        # the body pose -- saves 6 dwords/contact in CC.derived.
        cc_set_eff_n(cc, k, eff_n)
        cc_set_eff_t1(cc, k, eff_t1)
        cc_set_eff_t2(cc, k, eff_t2)
        cc_set_bias(cc, k, bias_val)
        cc_set_bias_t1(cc, k, bias_t1_val)
        cc_set_bias_t2(cc, k, bias_t2_val)

        # ---- Optional soft-contact PD plumbing -----------------------
        # The per-contact ``rigid_contact_stiffness / damping`` arrays
        # carry both narrow-phase absolute K/D (hydroelastic etc.,
        # positive values) and Material-derived softness encoded as
        # ``(-hertz, -damping_ratio)``. Sign of stiffness gates the
        # decode:
        #   * ``> 0``: absolute K/D, used as-is (hydroelastic).
        #   * ``< 0``: Material-encoded; decode via
        #     ``k = omega^2 * m_eff`` and
        #     ``c = 2 * zeta * omega * m_eff`` so pd_coefficients sees
        #     the same absolute units.
        #   * ``== 0``: legacy Box2D-rigid normal row; pd_eff_soft = 0.
        # Sentinel arrays (length 0) short-circuit to the legacy path,
        # which keeps pre-soft-contact scenes bit-for-bit unchanged.
        k_n = wp.float32(0.0)
        c_n = wp.float32(0.0)
        stiffness_arr_len = contacts.rigid_contact_stiffness.shape[0]
        damping_arr_len = contacts.rigid_contact_damping.shape[0]
        if stiffness_arr_len > k:
            k_n = contacts.rigid_contact_stiffness[k]
        if damping_arr_len > k:
            c_n = contacts.rigid_contact_damping[k]
        # Material decode: negative stiffness encodes hertz; negative
        # damping encodes damping_ratio. Convert to absolute K/C using
        # ``eff_n`` (forward effective mass).
        if k_n < wp.float32(0.0) and eff_n > wp.float32(0.0):
            hertz = -k_n
            damping_ratio = wp.float32(0.0)
            if c_n < wp.float32(0.0):
                damping_ratio = -c_n
            omega = two_pi * hertz
            k_n = omega * omega * eff_n
            c_n = wp.float32(2.0) * damping_ratio * omega * eff_n
        if (k_n > wp.float32(0.0) or c_n > wp.float32(0.0)) and eff_n > wp.float32(0.0):
            # ``pd_coefficients`` wants inverse effective mass;
            # ``eff_n`` is the forward mass. Sign: ``jv_n = (v2-v1).n``
            # (closing when >0), ``effective_gap > 0`` separated /
            # ``< 0`` penetrating, so spring depth is
            # ``-effective_gap``. Iterate enters ``lam = -eff_soft *
            # (jv_n - bias + ...)``; positive depth -> positive bias
            # -> positive normal impulse (pushes b2 off b1). Matches
            # Box2D Baumgarte sign and the ``lam_n >= 0`` unilateral
            # clamp.
            eff_inv_n = wp.float32(1.0) / eff_n
            pd_gamma_n, pd_bias_n, pd_eff_soft_n = pd_coefficients(
                k_n, c_n, -effective_gap, eff_inv_n, dt_substep, PHOENX_BOOST_CONTACT_NORMAL
            )
            cc_set_pd_gamma(cc, k, pd_gamma_n)
            cc_set_pd_bias(cc, k, pd_bias_n)
            cc_set_pd_eff_soft(cc, k, pd_eff_soft_n)
        else:
            cc_set_pd_eff_soft(cc, k, wp.float32(0.0))

        # Warm-start impulse: scatter per contact via the endpoint
        # helper. Loses the rigid pair's batched scatter, but keeps
        # the math identical and works uniformly for triangles. Graph
        # coloring guarantees no two cids in one colour share an
        # endpoint, so the plain stores are race-free.
        lam_n = cc_get_normal_lambda(cc, k)
        lam_t1 = cc_get_tangent1_lambda(cc, k)
        lam_t2 = cc_get_tangent2_lambda(cc, k)
        imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
        endpoint_warmstart_apply_impulse(
            kind1, idx1, local_p0, imp, wp.cross(ep1.r, imp), ep1.inv_mass, ep1.inv_inertia,
            wp.float32(-1.0), bodies, particles, tri_indices,
        )
        endpoint_warmstart_apply_impulse(
            kind2, idx2, local_p1, imp, wp.cross(ep2.r, imp), ep2.inv_mass, ep2.inv_inertia,
            wp.float32(+1.0), bodies, particles, tri_indices,
        )


@wp.func
def contact_iterate_at(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    """One PGS sweep over every contact of one shape pair.

    Endpoint-dispatched. Per contact: :func:`endpoint_load` builds the
    per-endpoint state (world contact point, velocity, lever arm,
    inverse mass / inertia) from the column's ``(kind, idx)`` tags,
    the shared lambda math runs unchanged, and
    :func:`endpoint_apply_impulse` scatters the contact impulse back
    to the endpoint stores. The original rigid-pair register-resident
    velocity optimisation is gone (loads happen per-contact); graph
    coloring keeps stores race-free.
    """

    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    kind1 = contact_get_endpoint_kind1(constraints, cid)
    kind2 = contact_get_endpoint_kind2(constraints, cid)
    idx1 = contact_get_endpoint_idx1(constraints, cid)
    idx2 = contact_get_endpoint_idx2(constraints, cid)

    mu_s = contact_get_friction(constraints, cid)
    mu_k = contact_get_friction_dynamic(constraints, cid)

    dt_substep = wp.float32(1.0) / idt
    _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    for i in range(contact_count):
        k = contact_first + i

        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        local_p0 = cc_get_local_p0(cc, k)
        local_p1 = cc_get_local_p1(cc, k)
        margin0 = contacts.rigid_contact_margin0[k]
        margin1 = contacts.rigid_contact_margin1[k]
        ep1 = endpoint_load(kind1, idx1, local_p0, margin0, wp.float32(1.0), n, bodies, particles, tri_indices)
        ep2 = endpoint_load(kind2, idx2, local_p1, margin1, wp.float32(-1.0), n, bodies, particles, tri_indices)

        eff_n = cc_get_eff_n(cc, k)
        eff_t1 = cc_get_eff_t1(cc, k)
        eff_t2 = cc_get_eff_t2(cc, k)
        bias_val = cc_get_bias(cc, k)
        bias_t1_val = cc_get_bias_t1(cc, k)
        bias_t2_val = cc_get_bias_t2(cc, k)
        is_speculative = bias_val > wp.float32(0.0)
        if not use_bias:
            if not is_speculative:
                bias_val = wp.float32(0.0)
            bias_t1_val = wp.float32(0.0)
            bias_t2_val = wp.float32(0.0)

        # Relative velocity at the contact point (endpoint 2 from
        # endpoint 1). Triangle endpoints carry zero angular velocity
        # / zero ``r``, so the rigid-only ``v + w x r`` form collapses
        # to the per-vertex barycentric reduction stored in ``ep.v``.
        vel_rel = ep2.v + wp.cross(ep2.w, ep2.r) - ep1.v - wp.cross(ep1.w, ep1.r)

        jv_n = wp.dot(vel_rel, n)
        jv_t1 = wp.dot(vel_rel, t1_dir)
        jv_t2 = wp.dot(vel_rel, t2_dir)

        # Normal row: Box2D v3 soft-constraint solve + clamp. Three
        # regimes (mirrors ``b2SolveOverflowContacts`` /
        # ``b2SolveContactsTask``):
        #  1. Speculative (``gap > 0`` at prepare): rigid PGS
        #     (``mass_coeff=1``, ``impulse_coeff=0``) + ``gap*inv_dt``
        #     bias, runs unconditionally so relax still caps closing
        #     at ``gap/dt`` instead of braking on ``-eff_n*jv_n``.
        #  2. Penetrating main solve (``!speculative && use_bias``):
        #     soft PGS via :func:`soft_constraint_coefficients`.
        #  3. Penetrating relax (``!speculative && !use_bias``):
        #     rigid PGS, zero bias -- pure ``Jv=0`` without the
        #     positional bias the main solve just injected.
        # Soft-contact PD normal row (PhysX-style absolute K/D),
        # active iff prepare wrote non-zero ``pd_eff_soft``. Bias is
        # the spring term ``k*depth`` and STAYS ON during relax --
        # zeroing it would let ``-eff_soft*gamma*acc`` cancel the
        # accumulated impulse, same as ADBS drive/limit/cable rows.
        pd_eff_soft_n = cc_get_pd_eff_soft(cc, k)
        lam_n_old = cc_get_normal_lambda(cc, k)
        if pd_eff_soft_n > wp.float32(0.0):
            pd_gamma_n = cc_get_pd_gamma(cc, k)
            pd_bias_n = cc_get_pd_bias(cc, k)
            # ``lam = -eff_soft * (jv - bias + gamma * acc)``; unit-
            # matched to the absolute-PD rows in the ADBS joint
            # drive / limit.
            d_lam_n_us = -pd_eff_soft_n * (jv_n - pd_bias_n + pd_gamma_n * lam_n_old)
            lam_n_new = wp.max(lam_n_old + d_lam_n_us, wp.float32(0.0))
            d_lam_n = lam_n_new - lam_n_old
        else:
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
        endpoint_apply_impulse(
            kind1, idx1, local_p0, imp, ep1.inv_mass, ep1.inv_inertia, ep1.r,
            wp.float32(-1.0), bodies, particles, tri_indices,
        )
        endpoint_apply_impulse(
            kind2, idx2, local_p1, imp, ep2.inv_mass, ep2.inv_inertia, ep2.r,
            wp.float32(+1.0), bodies, particles, tri_indices,
        )


# ---------------------------------------------------------------------------
# Direct entry points (base_offset = 0)
# ---------------------------------------------------------------------------


@wp.func
def contact_prepare_for_iteration(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at(
        constraints, cid, 0, bodies, body_pair, idt, cc, contacts, particles, tri_indices
    )


@wp.func
def contact_iterate(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at(
        constraints, cid, 0, bodies, body_pair, idt, cc, contacts, use_bias, particles, tri_indices
    )


@wp.func
def contact_iterate_at_multi(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    """``num_sweeps`` PGS sweeps on one contact column.

    Endpoint-dispatched (see :func:`contact_iterate_at`). Cids in the
    same graph colour don't share endpoints, so running ``num_sweeps``
    sweeps before moving to the next cid is equivalent to round-robin
    within the colour.
    """
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    kind1 = contact_get_endpoint_kind1(constraints, cid)
    kind2 = contact_get_endpoint_kind2(constraints, cid)
    idx1 = contact_get_endpoint_idx1(constraints, cid)
    idx2 = contact_get_endpoint_idx2(constraints, cid)

    mu_s = contact_get_friction(constraints, cid)
    mu_k = contact_get_friction_dynamic(constraints, cid)

    dt_substep = wp.float32(1.0) / idt
    _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    it = wp.int32(0)
    while it < num_sweeps:
        for i in range(contact_count):
            k = contact_first + i

            n = cc_get_normal(cc, k)
            t1_dir = cc_get_tangent1(cc, k)
            t2_dir = wp.cross(n, t1_dir)
            local_p0 = cc_get_local_p0(cc, k)
            local_p1 = cc_get_local_p1(cc, k)
            margin0 = contacts.rigid_contact_margin0[k]
            margin1 = contacts.rigid_contact_margin1[k]
            ep1 = endpoint_load(kind1, idx1, local_p0, margin0, wp.float32(1.0), n, bodies, particles, tri_indices)
            ep2 = endpoint_load(kind2, idx2, local_p1, margin1, wp.float32(-1.0), n, bodies, particles, tri_indices)

            eff_n = cc_get_eff_n(cc, k)
            eff_t1 = cc_get_eff_t1(cc, k)
            eff_t2 = cc_get_eff_t2(cc, k)
            bias_val = cc_get_bias(cc, k)
            bias_t1_val = cc_get_bias_t1(cc, k)
            bias_t2_val = cc_get_bias_t2(cc, k)
            is_speculative = bias_val > wp.float32(0.0)
            if not use_bias:
                if not is_speculative:
                    bias_val = wp.float32(0.0)
                bias_t1_val = wp.float32(0.0)
                bias_t2_val = wp.float32(0.0)

            vel_rel = ep2.v + wp.cross(ep2.w, ep2.r) - ep1.v - wp.cross(ep1.w, ep1.r)
            jv_n = wp.dot(vel_rel, n)
            jv_t1 = wp.dot(vel_rel, t1_dir)
            jv_t2 = wp.dot(vel_rel, t2_dir)

            pd_eff_soft_n = cc_get_pd_eff_soft(cc, k)
            lam_n_old = cc_get_normal_lambda(cc, k)
            if pd_eff_soft_n > wp.float32(0.0):
                pd_gamma_n = cc_get_pd_gamma(cc, k)
                pd_bias_n = cc_get_pd_bias(cc, k)
                d_lam_n_us = -pd_eff_soft_n * (jv_n - pd_bias_n + pd_gamma_n * lam_n_old)
                lam_n_new = wp.max(lam_n_old + d_lam_n_us, wp.float32(0.0))
                d_lam_n = lam_n_new - lam_n_old
            else:
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
            endpoint_apply_impulse(
                kind1, idx1, local_p0, imp, ep1.inv_mass, ep1.inv_inertia, ep1.r,
                wp.float32(-1.0), bodies, particles, tri_indices,
            )
            endpoint_apply_impulse(
                kind2, idx2, local_p1, imp, ep2.inv_mass, ep2.inv_inertia, ep2.r,
                wp.float32(+1.0), bodies, particles, tri_indices,
            )
        it += 1


@wp.func
def contact_iterate_multi(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at_multi(
        constraints, cid, 0, bodies, body_pair, idt, cc, contacts, use_bias, num_sweeps, particles, tri_indices
    )


# ---------------------------------------------------------------------------
# Per-column / per-contact wrench + error reporting
# ---------------------------------------------------------------------------


@wp.func
def _contact_recompute_r2(
    bodies: BodyContainer,
    contacts: ContactViews,
    b2: wp.int32,
    k: wp.int32,
    n: wp.vec3f,
    cc: ContactContainer,
) -> wp.vec3f:
    """Recompute the cached-style ``r2`` lever arm for contact ``k``.

    Same expression as the prepare's ``r2 = quat_rotate(orient2,
    local_p1 - body_com2) - margin1 * n``. Used by the wrench
    helpers in place of the dropped ``cc_get_r2`` cache.
    """
    local_p1 = cc_get_local_p1(cc, k)
    body_com2 = bodies.body_com[b2]
    orientation2 = bodies.orientation[b2]
    margin1 = contacts.rigid_contact_margin1[k]
    return wp.quat_rotate(orientation2, local_p1 - body_com2) - margin1 * n


@wp.func
def contact_world_wrench_at(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Column-summed world-frame wrench this contact pair applies to body 2.

    Sums the per-contact impulse ``lambda_n * n + lambda_t1 * t1 +
    lambda_t2 * t2`` across the column's ``contact_count`` contacts
    and divides by ``dt`` for a force. Torque is computed against
    body 2's COM using the recomputed lever arm ``r2`` (see
    :func:`_contact_recompute_r2`).
    """
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(contact_count):
        k = contact_first + i
        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        r2 = _contact_recompute_r2(bodies, contacts, b2, k, n, cc)
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
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    return contact_world_wrench_at(constraints, cid, 0, bodies, idt, cc, contacts)


@wp.func
def contact_per_k_wrench_at(
    k: wp.int32,
    b2: wp.int32,
    bodies: BodyContainer,
    contacts: ContactViews,
    idt: wp.float32,
    cc: ContactContainer,
):
    """World-frame wrench on body 2 for one contact ``k``.

    ``force`` is ``imp / dt`` where ``imp = lam_n n + lam_t1 t1 +
    lam_t2 t2`` is the world-space impulse the contact applied during
    the most recent substep; ``torque`` is the moment about body 2's
    COM using the recomputed lever arm ``r2``.
    """
    n = cc_get_normal(cc, k)
    t1_dir = cc_get_tangent1(cc, k)
    t2_dir = wp.cross(n, t1_dir)
    r2 = _contact_recompute_r2(bodies, contacts, b2, k, n, cc)
    lam_n = cc_get_normal_lambda(cc, k)
    lam_t1 = cc_get_tangent1_lambda(cc, k)
    lam_t2 = cc_get_tangent2_lambda(cc, k)
    imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
    force = imp * idt
    torque = wp.cross(r2, force)
    return force, torque


@wp.kernel(enable_backward=False)
def contact_per_contact_wrench_kernel(
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    cc: ContactContainer,
    contacts: ContactViews,
    num_active_columns: wp.int32,
    idt: wp.float32,
    # out
    out: wp.array[wp.spatial_vector],
):
    """Scatter per-contact wrenches into a per-rigid-contact output array.

    Launch dim = ``max_contact_columns`` (one thread per contact
    column slot). Each thread walks the
    ``[contact_first, contact_first + contact_count)`` range and writes
    ``out[k] = spatial_vector(force, torque)`` for every contact ``k``
    it covers. Slots outside ``[0, num_active_columns)`` early-out.
    """
    tid = wp.tid()
    if tid >= num_active_columns:
        return
    contact_first = contact_get_contact_first(contact_cols, tid)
    contact_count = contact_get_contact_count(contact_cols, tid)
    b2 = contact_get_body2(contact_cols, tid)
    for i in range(contact_count):
        k = contact_first + i
        f, tau = contact_per_k_wrench_at(k, b2, bodies, contacts, idt, cc)
        out[k] = wp.spatial_vector(f, tau)


@wp.kernel(enable_backward=False)
def contact_pair_wrench_kernel(
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    cc: ContactContainer,
    contacts: ContactViews,
    num_active_columns: wp.int32,
    idt: wp.float32,
    # out
    wrenches: wp.array[wp.spatial_vector],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    contact_count_out: wp.array[wp.int32],
):
    """Per-column summary: total wrench + ``(body1, body2)`` + contact count.

    One output slot per contact column. The per-pair design means one
    column now covers the whole shape pair, so the per-column output
    is already the per-pair wrench -- no post-pass summation across
    adjacent columns needed.
    """
    tid = wp.tid()
    if tid >= num_active_columns:
        wrenches[tid] = wp.spatial_vector(wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0))
        body1[tid] = -1
        body2[tid] = -1
        contact_count_out[tid] = 0
        return
    b1 = contact_get_body1(contact_cols, tid)
    b2 = contact_get_body2(contact_cols, tid)
    contact_first = contact_get_contact_first(contact_cols, tid)
    count = contact_get_contact_count(contact_cols, tid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(count):
        k = contact_first + i
        f, tau = contact_per_k_wrench_at(k, b2, bodies, contacts, idt, cc)
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
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    cc: ContactContainer,
    contacts: ContactViews,
    num_active_columns: wp.int32,
    # out
    out: wp.array[wp.vec3f],
):
    """Scatter per-contact residuals into a per-rigid-contact output array.

    Companion to :func:`contact_per_contact_wrench_kernel`: same launch
    contract, writes ``out[k] = (gap, drift_t1, drift_t2)`` for every
    contact ``k`` in every active contact column.
    """
    tid = wp.tid()
    if tid >= num_active_columns:
        return
    b1 = contact_get_body1(contact_cols, tid)
    b2 = contact_get_body2(contact_cols, tid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_first = contact_get_contact_first(contact_cols, tid)
    contact_count = contact_get_contact_count(contact_cols, tid)
    for i in range(contact_count):
        k = contact_first + i
        out[k] = contact_per_k_error_at(k, bodies, body_pair, cc, contacts)


@wp.func
def contact_world_error_at(
    constraints: ContactColumnContainer,
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
    return wp.spatial_vector(wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0))


@wp.func
def contact_world_error(
    constraints: ContactColumnContainer,
    cid: wp.int32,
) -> wp.spatial_vector:
    return contact_world_error_at(constraints, cid, 0)
