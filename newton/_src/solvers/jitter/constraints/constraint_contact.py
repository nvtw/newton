# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rigid-rigid contact constraint for the Jitter solver.

One :data:`CONSTRAINT_TYPE_CONTACT` column packs up to six contact
points belonging to a single ``(shape_a, shape_b)`` pair. Geometry
(body-frame anchor points, world-frame normals) is *not* duplicated
into the column -- the column stores just a ``[contact_first,
contact_count]`` range into the upstream sorted
:class:`newton._src.sim.contacts.Contacts` buffer plus the
per-iteration *derived* quantities (lever arms ``r1``/``r2``,
tangents, effective masses, velocity bias). This keeps the ingest
kernel cheap (a range write per column) and makes Jitter2-style TGS
sub-step refresh free: every ``contact_prepare_for_iteration`` simply
re-reads ``rigid_contact_point0[k]`` / ``rigid_contact_point1[k]``
and re-projects them with the *current* body orientations.

Persistent state (accumulated normal + tangent impulses used for
warm-starting) lives in a parallel
:class:`newton._src.solvers.jitter.constraints.contact_container.ContactContainer`
keyed by the same cid (see that module's docstring for the split
rationale).

Friction: two-tangent pyramidal PGS. Each active slot contributes three
scalar rows -- normal (non-penetration, ``lambda_n >= 0``) plus two
independent tangential rows (each clamped to ``[-mu * lambda_n,
+mu * lambda_n]``). This is an approximation to the true Coulomb cone
but integrates trivially with the existing PGS infrastructure and
matches what Box2D v3 and solver2d do in their "simple friction" mode.
The schema keeps enough room for a future friction-anchor upgrade
(same storage footprint, different formulation) without a schema
break.
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
    read_vec3,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_vec3,
)
from newton._src.solvers.jitter.constraints.contact_container import (
    MAX_SLOTS,
    ContactContainer,
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_local_p0,
    cc_set_local_p1,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.jitter.helpers.data_packing import dword_offset_of, num_dwords

__all__ = [
    "CONTACT_DWORDS",
    "CONTACT_MAX_SLOTS",
    "ContactConstraintData",
    "ContactViews",
    "contact_get_active_mask",
    "contact_get_body1",
    "contact_get_body2",
    "contact_get_contact_first",
    "contact_get_friction",
    "contact_get_friction_dynamic",
    "contact_iterate",
    "contact_iterate_at",
    "contact_pair_wrench_kernel",
    "contact_per_contact_error_kernel",
    "contact_per_contact_wrench_kernel",
    "contact_per_slot_error_at",
    "contact_per_slot_wrench_at",
    "contact_position_iterate",
    "contact_position_iterate_at",
    "contact_prepare_for_iteration",
    "contact_prepare_for_iteration_at",
    "contact_set_active_mask",
    "contact_set_body1",
    "contact_set_body2",
    "contact_set_contact_first",
    "contact_set_friction",
    "contact_set_friction_dynamic",
    "contact_views_make",
    "contact_world_error",
    "contact_world_error_at",
    "contact_world_wrench",
    "contact_world_wrench_at",
]


#: Number of contact slots packed into one column. Mirrored from
#: :data:`contact_container.MAX_SLOTS` as a module constant so the
#: schema stays self-contained at import time.
CONTACT_MAX_SLOTS: int = int(MAX_SLOTS)


# ---------------------------------------------------------------------------
# Per-slot sub-struct
# ---------------------------------------------------------------------------
#
# The 6-slot layout is modelled with an explicit ``ContactSlotData`` inner
# struct tiled 6x inside the outer schema. This keeps the dword offsets
# for slot ``k`` computable as ``_OFF_SLOT0 + k * _SLOT_DWORDS`` (one
# multiply + add in the kernel) instead of 6 copies of every field.
# Runtime kernels only see the flat ``wp.array2d[wp.float32]`` backing
# store -- the struct hierarchy is a host-side offset table.


@wp.struct
class _ContactSlotData:
    """Per-slot derived quantities recomputed every ``prepare_for_iteration``.

    Body-local anchor points and world normals are *not* stored here --
    they're read straight out of the upstream ``Contacts`` buffer at
    each prepare (see :func:`contact_prepare_for_iteration_at`). This
    makes TGS sub-step refresh free and keeps the column slim.

    Tangent directions are *also* not stored; they're rebuilt from
    the (unit, substep-invariant) contact normal inside the iterate
    kernel via :func:`_build_tangents`. The rebuild is deterministic
    and bit-identical to the value `prepare` would have cached, so
    the only cost is ~20 FLOPS per slot per iteration -- a win
    against the 6 dwords / slot of storage it saves.

    Warm-start impulses (``*_lambda``) live in the parallel
    :class:`ContactContainer`.
    """

    #: Lever arm of the contact anchor on body 1 relative to body 1's
    #: COM, in *world* space. Recomputed each prepare from
    #: ``rigid_contact_point0`` + the current body 1 pose.
    r1: wp.vec3f
    #: Lever arm of the contact anchor on body 2 relative to body 2's
    #: COM, in *world* space.
    r2: wp.vec3f

    #: Scalar effective mass for the normal row (``1 / J M^-1 J^T``).
    #: Unsoftened; softness is applied at the iterate site via
    #: ``mass_coeff`` / ``impulse_coeff`` if needed (contacts are
    #: rigid in v1 -- coefficients baked to the rigid value).
    effective_mass_n: wp.float32
    #: Scalar effective mass for tangent1 / tangent2 (identical in
    #: the canonical friction formulation but stored separately so a
    #: future anisotropic friction upgrade is one-line).
    effective_mass_t1: wp.float32
    effective_mass_t2: wp.float32

    #: Velocity-correction bias [m/s] for the normal row --
    #: ``bias_rate * max(penetration, 0)``. Zero for non-penetrating
    #: contacts so contacts that are strictly separating don't
    #: generate spurious corrective impulses.
    bias: wp.float32
    #: Position-level tangent-row biases [m/s]. Kinetic slide is
    #: zeroed by an anchor-reset at prepare; the remaining in-slop
    #: drift goes in here so static friction pulls the body back to
    #: the anchor proportional to its displacement.
    bias_t1: wp.float32
    bias_t2: wp.float32


_SLOT_DWORDS: int = num_dwords(_ContactSlotData)


# ---------------------------------------------------------------------------
# Outer schema
# ---------------------------------------------------------------------------


@wp.struct
class ContactConstraintData:
    """Column schema for one :data:`CONSTRAINT_TYPE_CONTACT` constraint.

    Six contacts are tiled flat inside one column (no nested
    ``wp.array``: Warp ``@wp.struct`` fields can't be arrays). Access
    slot ``k`` via :func:`_slot_offset`. Manifolds larger than six
    contacts are split across multiple adjacent columns at ingest time;
    see :func:`_contact_pack_columns_kernel`.

    The ``body1`` / ``body2`` header fields are the two *rigid bodies*
    attached to the pair's ``(shape_a, shape_b)``, resolved via
    ``model.shape_body[]`` during ingest. All six slots within one
    column therefore share the same body pair by construction.
    """

    #: Tag -- :data:`CONSTRAINT_TYPE_CONTACT`. Must be at dword 0.
    constraint_type: wp.int32
    #: Body index of body 1 (``model.shape_body[shape_a]``).
    body1: wp.int32
    #: Body index of body 2.
    body2: wp.int32

    #: Static Coulomb friction coefficient for the pair. This is the
    #: "stick" threshold -- if the per-iteration raw tangent impulse
    #: magnitude stays inside ``mu_static * lam_n`` the contact is in
    #: the static-friction regime and no slip occurs. Resolved at
    #: ingest time from the material table (see
    #: :func:`materials.resolve_frictions_in_kernel`); falls back to
    #: ``default_friction`` when no materials are installed.
    friction: wp.float32
    #: Kinetic (dynamic) Coulomb friction coefficient. Used as the
    #: tangent-row clamp once the static threshold has been crossed
    #: and the contact is slipping. Typically ``<=`` static friction
    #: so a sliding body decelerates less than it would have
    #: resisted starting the slide. When materials aren't configured
    #: with separate values this ends up equal to ``friction`` and
    #: the two-regime clamp collapses to the single-coefficient case.
    friction_dynamic: wp.float32

    #: Bitmask of active slots (bit ``k`` set iff slot ``k`` holds a
    #: real contact). Used by the per-iteration loop to skip inactive
    #: slots without a per-slot sentinel load. Always equals
    #: ``(1 << contact_count) - 1`` at ingest time but kept as an
    #: explicit mask so a future selective-deactivation pass
    #: (e.g. culling persistently zero-lambda slots) is a one-line
    #: mask clear.
    active_mask: wp.int32

    #: Start index into the sorted ``Contacts`` buffer for the first
    #: active slot of this column. Slot ``k`` maps to contact
    #: ``contact_first + k`` as long as ``(active_mask >> k) & 1``.
    contact_first: wp.int32

    # Six slots tiled inline. Field order matters -- the kernels
    # compute slot ``k``'s base dword offset as
    # ``_OFF_SLOT0 + k * _SLOT_DWORDS``, which only works if the six
    # sub-structs are packed consecutively with no other fields
    # interleaved between them.
    slot0: _ContactSlotData
    slot1: _ContactSlotData
    slot2: _ContactSlotData
    slot3: _ContactSlotData
    slot4: _ContactSlotData
    slot5: _ContactSlotData


# Enforce the three-field constraint header contract at import time so
# a future field reorder fails loudly here.
assert_constraint_header(ContactConstraintData)


# Dword offsets of the header / per-column fields. Each is a Python int
# wrapped in wp.constant so kernels can use them as compile-time
# literals (same pattern as the other constraint modules).
_OFF_BODY1 = wp.constant(dword_offset_of(ContactConstraintData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ContactConstraintData, "body2"))
_OFF_FRICTION = wp.constant(dword_offset_of(ContactConstraintData, "friction"))
_OFF_FRICTION_DYNAMIC = wp.constant(
    dword_offset_of(ContactConstraintData, "friction_dynamic")
)
_OFF_ACTIVE_MASK = wp.constant(dword_offset_of(ContactConstraintData, "active_mask"))
_OFF_CONTACT_FIRST = wp.constant(dword_offset_of(ContactConstraintData, "contact_first"))

# Offset of slot 0's first field. Slot ``k``'s offset is
# ``_OFF_SLOT0 + k * _SLOT_DWORDS``; offsets of the fields *inside* a
# slot are constant across slots and stored in the ``_OFF_SLOT_*``
# block below.
_OFF_SLOT0 = wp.constant(dword_offset_of(ContactConstraintData, "slot0"))

_OFF_SLOT_R1 = wp.constant(dword_offset_of(_ContactSlotData, "r1"))
_OFF_SLOT_R2 = wp.constant(dword_offset_of(_ContactSlotData, "r2"))
_OFF_SLOT_EFF_MASS_N = wp.constant(dword_offset_of(_ContactSlotData, "effective_mass_n"))
_OFF_SLOT_EFF_MASS_T1 = wp.constant(dword_offset_of(_ContactSlotData, "effective_mass_t1"))
_OFF_SLOT_EFF_MASS_T2 = wp.constant(dword_offset_of(_ContactSlotData, "effective_mass_t2"))
_OFF_SLOT_BIAS = wp.constant(dword_offset_of(_ContactSlotData, "bias"))
_OFF_SLOT_BIAS_T1 = wp.constant(dword_offset_of(_ContactSlotData, "bias_t1"))
_OFF_SLOT_BIAS_T2 = wp.constant(dword_offset_of(_ContactSlotData, "bias_t2"))

_SLOT_DWORDS_CONST = wp.constant(wp.int32(_SLOT_DWORDS))

#: Total dword count of one contact constraint column. Used by
#: :class:`ConstraintContainer` to size the shared storage's row
#: dimension; every contact column eats this many rows regardless of
#: how many slots are active.
CONTACT_DWORDS: int = num_dwords(ContactConstraintData)


# ---------------------------------------------------------------------------
# ContactViews -- bundle Newton Contacts arrays into a single wp.struct
# ---------------------------------------------------------------------------
#
# Newton's user-facing :class:`newton._src.sim.contacts.Contacts` class is
# a plain Python object (not a ``@wp.struct``), so it can't be passed
# directly into a kernel. We mirror its subset of arrays the solver
# needs into a thin Warp struct, built on the host once per step. This
# lets the unified dispatcher keep its tidy 5-arg signature -- the
# contact branch just adds one extra struct argument.


@wp.struct
class ContactViews:
    """Kernel-visible view onto the fields of a Newton ``Contacts`` buffer.

    Populated on the host once per :meth:`World.step` from the
    user-supplied :class:`Contacts` + the Newton model's ``shape_body``
    mapping. Only the subset of arrays actually read by the contact
    prepare / iterate / gather kernels is carried through -- the
    soft-contact arrays and gradient-only arrays are omitted.
    """

    #: Single-element active-count slice from :class:`Contacts`
    #: (``Contacts.rigid_contact_count[0:1]``); used by the ingest
    #: kernel to gate per-contact threads at the active/inactive
    #: boundary. Read-only.
    rigid_contact_count: wp.array[wp.int32]

    #: Contact point on shape 0, expressed in shape 0's body frame
    #: (i.e. relative to the body's origin, not its COM). Combined
    #: with the body transform at ``prepare_for_iteration`` time to
    #: produce the world-frame anchor. Shape (rigid_contact_max,).
    rigid_contact_point0: wp.array[wp.vec3f]
    #: Contact point on shape 1 in shape 1's body frame. Shape
    #: (rigid_contact_max,).
    rigid_contact_point1: wp.array[wp.vec3f]
    #: World-space contact normal pointing from shape 0 -> shape 1
    #: (A-to-B convention). Unit length. Stays fixed across substeps
    #: (only the anchor world positions move). Shape (rigid_contact_max,).
    rigid_contact_normal: wp.array[wp.vec3f]

    #: Shape indices for the pair. Shape (rigid_contact_max,).
    rigid_contact_shape0: wp.array[wp.int32]
    rigid_contact_shape1: wp.array[wp.int32]

    #: Frame-to-frame match: for contact ``k`` in the current-frame
    #: sorted buffer, ``rigid_contact_match_index[k]`` is the index
    #: (into the *previous* frame's sorted buffer) of the contact
    #: this one was matched to, or :data:`MATCH_NOT_FOUND` (``-1``)
    #: / :data:`MATCH_BROKEN` (``-2``) for unmatched contacts. Used
    #: exclusively by the warm-start gather kernel.
    rigid_contact_match_index: wp.array[wp.int32]

    #: Surface thickness for each shape (``effective_radius + margin``);
    #: folded into the penetration depth so the constraint targets
    #: ``contact_point0 - contact_point1`` being at least
    #: ``margin0 + margin1`` apart in world space. Shape
    #: (rigid_contact_max,).
    rigid_contact_margin0: wp.array[wp.float32]
    rigid_contact_margin1: wp.array[wp.float32]

    #: Newton model's per-shape body index (``model.shape_body``).
    #: Resolves a shape id to a rigid-body id. Same length as
    #: ``model.shape_count``.
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

    No device allocations; this is a pure struct pack on the host.
    Call once per :meth:`World.step` after the upstream collision
    pipeline has finished writing the :class:`Contacts` buffer.
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
    """Static Coulomb friction coefficient of this contact column."""
    return read_float(c, _OFF_FRICTION, cid)


@wp.func
def contact_get_friction_dynamic(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    """Kinetic (dynamic) Coulomb friction coefficient of this contact
    column -- the clamp once the static threshold has been exceeded.
    """
    return read_float(c, _OFF_FRICTION_DYNAMIC, cid)


@wp.func
def contact_set_friction_dynamic(
    c: ConstraintContainer, cid: wp.int32, v: wp.float32
):
    """Set the kinetic (dynamic) friction coefficient. Used by the
    contact ingest pipeline; kernels should prefer
    :func:`contact_get_friction_dynamic`.
    """
    write_float(c, _OFF_FRICTION_DYNAMIC, cid, v)


@wp.func
def contact_set_friction(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_FRICTION, cid, v)


@wp.func
def contact_get_active_mask(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_ACTIVE_MASK, cid)


@wp.func
def contact_set_active_mask(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_ACTIVE_MASK, cid, v)


@wp.func
def contact_get_contact_first(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_CONTACT_FIRST, cid)


@wp.func
def contact_set_contact_first(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_CONTACT_FIRST, cid, v)


# ---------------------------------------------------------------------------
# Per-slot accessors -- add ``slot * _SLOT_DWORDS`` onto the base
# ---------------------------------------------------------------------------
#
# Every per-slot field lives at ``base_offset + _OFF_SLOT0 +
# slot * _SLOT_DWORDS + _OFF_SLOT_<field>``. ``base_offset`` is the
# sub-block offset of the contact constraint inside its column (0 for
# a stand-alone contact constraint, but kept for symmetry with the
# other constraint modules' fused variants).


@wp.func
def _slot_base(base_offset: wp.int32, slot: wp.int32) -> wp.int32:
    return base_offset + _OFF_SLOT0 + slot * _SLOT_DWORDS_CONST


@wp.func
def _slot_get_r1(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.vec3f:
    return read_vec3(c, base + _OFF_SLOT_R1, cid)


@wp.func
def _slot_set_r1(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.vec3f):
    write_vec3(c, base + _OFF_SLOT_R1, cid, v)


@wp.func
def _slot_get_r2(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.vec3f:
    return read_vec3(c, base + _OFF_SLOT_R2, cid)


@wp.func
def _slot_set_r2(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.vec3f):
    write_vec3(c, base + _OFF_SLOT_R2, cid, v)


@wp.func
def _slot_get_eff_n(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.float32:
    return read_float(c, base + _OFF_SLOT_EFF_MASS_N, cid)


@wp.func
def _slot_set_eff_n(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.float32):
    write_float(c, base + _OFF_SLOT_EFF_MASS_N, cid, v)


@wp.func
def _slot_get_eff_t1(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.float32:
    return read_float(c, base + _OFF_SLOT_EFF_MASS_T1, cid)


@wp.func
def _slot_set_eff_t1(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.float32):
    write_float(c, base + _OFF_SLOT_EFF_MASS_T1, cid, v)


@wp.func
def _slot_get_eff_t2(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.float32:
    return read_float(c, base + _OFF_SLOT_EFF_MASS_T2, cid)


@wp.func
def _slot_set_eff_t2(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.float32):
    write_float(c, base + _OFF_SLOT_EFF_MASS_T2, cid, v)


@wp.func
def _slot_get_bias(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.float32:
    return read_float(c, base + _OFF_SLOT_BIAS, cid)


@wp.func
def _slot_set_bias(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.float32):
    write_float(c, base + _OFF_SLOT_BIAS, cid, v)


@wp.func
def _slot_get_bias_t1(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.float32:
    return read_float(c, base + _OFF_SLOT_BIAS_T1, cid)


@wp.func
def _slot_set_bias_t1(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.float32):
    write_float(c, base + _OFF_SLOT_BIAS_T1, cid, v)


@wp.func
def _slot_get_bias_t2(c: ConstraintContainer, cid: wp.int32, base: wp.int32) -> wp.float32:
    return read_float(c, base + _OFF_SLOT_BIAS_T2, cid)


@wp.func
def _slot_set_bias_t2(c: ConstraintContainer, cid: wp.int32, base: wp.int32, v: wp.float32):
    write_float(c, base + _OFF_SLOT_BIAS_T2, cid, v)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


@wp.func
def _build_tangents(n: wp.vec3f):
    """Build an orthonormal tangent basis from a unit normal.

    Uses the branch-free construction from Duff et al. 2017,
    "Building an Orthonormal Basis, Revisited": the basis varies
    smoothly with ``n`` everywhere except at the single anti-pole
    ``n.z = -1``. This matters for warm-starting friction impulses.

    The earlier "most-orthogonal axis" pick flips the tangent basis
    by 90 degrees when two abs-components of ``n`` straddle ``<=``
    (e.g. a vertical contact normal going from ``(0, 0, 1)`` to
    ``(1e-9, 0, 1)`` swaps ``t1`` from ``+Y`` to ``-X``). Because
    ``lam_t1`` / ``lam_t2`` are scalars stored *in this basis*, a
    basis flip between substeps or frames turns a warm-started
    friction impulse into an impulse pointing along the sliding
    direction -- the exact symptom of a pyramid quietly drifting
    sideways even when the net tangential velocity is zero.
    """
    # Work in a frame where +Z is the principal axis so the
    # singularity sits at n.z = -1 (anti-pole). We take the sign of
    # n.z as the branch selector; the construction is otherwise
    # branch-free and produces a continuous basis in a 4pi steradian
    # region minus that anti-pole.
    sign = wp.float32(1.0)
    if n[2] < wp.float32(0.0):
        sign = wp.float32(-1.0)
    a = wp.float32(-1.0) / (sign + n[2])
    b = n[0] * n[1] * a
    t1 = wp.vec3f(
        wp.float32(1.0) + sign * n[0] * n[0] * a,
        sign * b,
        -sign * n[0],
    )
    t2 = wp.vec3f(
        b,
        sign + n[1] * n[1] * a,
        -n[1],
    )
    return t1, t2


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
    """Scalar 1/JM^-1J^T for an axis-aligned contact row.

    For a single row with Jacobian rows ``J_lin = -axis, axis`` and
    ``J_ang = -(r1 x axis), (r2 x axis)``, the effective mass inverse is
    ``m^-1 + m2^-1 + axis^T (I1 (r1 x axis) x r1 + I2 (r2 x axis) x r2) axis``
    which collapses in vector form to the expression below.
    """
    rc1 = wp.cross(r1, axis)
    rc2 = wp.cross(r2, axis)
    w = inv_mass1 + inv_mass2 + wp.dot(rc1, inv_inertia1 @ rc1) + wp.dot(rc2, inv_inertia2 @ rc2)
    if w > 1.0e-12:
        return 1.0 / w
    return 0.0


# ---------------------------------------------------------------------------
# Prepare / iterate / wrench
# ---------------------------------------------------------------------------


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
    """Prepare one 6-slot contact column for the upcoming PGS iterations.

    Every active slot:

      1. Re-projects the body-frame anchors
         (``contacts.rigid_contact_point0/1[k]``) into *world* space
         using the current body transforms. This is the TGS sub-step
         refresh -- direct port of Jitter2's
         ``Contact.UpdatePosition`` (``p1 = b1.pos + b1.rot * Position1``).
         The stored normal (``rigid_contact_normal[k]``) is what
         separation is measured against; it stays fixed across substeps.
      2. Computes lever arms ``r1 = p1 - com1``, ``r2 = p2 - com2``
         (in Jitter's body model, ``body.position`` *is* the COM, so
         the subtraction reduces to ``p1 - bodies.position[b1]``).
      3. Builds a stable orthonormal tangent basis from the normal and
         caches the three scalar effective masses.
      4. Sets ``bias = bias_rate * max(penetration, 0)`` as a
         Baumgarte-style positional restoration (rigid contacts, so
         ``bias_rate`` is the hard-contact constant
         ``CONTACT_BIAS_RATE`` scaled by how deep the penetration is).
      5. Applies the previously-gathered warm-start impulse
         (``cc_get_normal_lambda(cc, slot, cid)`` etc.) to both
         bodies' velocities so the PGS loop starts from a converged
         guess.

    Inactive slots (``active_mask`` bit clear) early-out with all
    per-slot fields left untouched -- the iterate kernel skips them
    by the same mask.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    active_mask = contact_get_active_mask(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)

    # Baumgarte-style positional bias rate. Hardcoded for v1 -- maps
    # to "recover full penetration over ~2 substeps". Box2D v3 uses
    # the same kind of knob (``contact_hertz`` in its demos).
    # Box2D v3 / solver2d soft-constraint coefficients for the normal
    # row: the same formulation the joints already use in this
    # codebase (see ``soft_constraint_coefficients`` in
    # ``constraint_container.py``). Setting ``hertz`` to
    # ``DEFAULT_HERTZ_LINEAR = 1e9`` makes the helper saturate at the
    # per-substep Nyquist rate (stiffest resolvable lock for the
    # current ``dt``); ``damping_ratio = 1`` is critical damping. No
    # length-scale slop is needed because the softness itself
    # (``mass_coeff < 1``, ``impulse_coeff > 0``) damps out the
    # zero-crossing oscillation the old Baumgarte dead-zone was
    # hiding. Scale-invariant: every number here is either
    # dimensionless or a frequency (1/s).
    dt_substep = wp.float32(1.0) / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    # Friction-row position bias. Drives tangential drift between the
    # stored (sticky) body-local anchors back toward zero with a
    # gentle Baumgarte pull. Rationale:
    #
    #   * Newton's sticky contact matching freezes ``local_p0/p1`` at
    #     contact-birth time (see ``_replay_matched_kernel`` in
    #     ``newton._src.geometry.contact_match``), so the stored
    #     anchor is the body-local point where the contact was first
    #     established. When the body drifts tangentially, the
    #     re-projected anchor moves with it and ``p2 - p1`` picks up
    #     the drift.
    #
    #   * Pure velocity-level friction (bias=0) cleans up every
    #     substep's tangential velocity but never undoes position
    #     drift that accumulated during integration. Tall stacks
    #     demonstrate this: each layer's normal impulse imbalance
    #     induces a micro-rotation, which translates the COM
    #     slightly, which velocity-level friction can't counter.
    #
    # Scale discipline: both constants are scale-invariant.
    # ``friction_bias_factor`` is dimensionless (same family as the
    # normal-row ``bias_factor``; see that block below for the full
    # scale-discipline docstring). ``friction_slop`` clamps the raw
    # drift before it drives the tangent bias -- sub-millimetre so
    # it sits below the supported smallest-object size (~a few mm
    # per the solver contract) and acts as a pure numerical-noise
    # guard rather than a length-scale threshold. The Coulomb cone
    # (``|lam_t| <= mu * lam_n``) still limits the final applied
    # tangent impulse, but the clamp also keeps the tangent bias
    # itself from spiking on any stray large drift.
    friction_bias_factor = wp.float32(0.08)
    # Friction drift clamp. Same scale-discipline caveat as
    # ``penetration_slop``: a length in metres, empirically tuned for
    # mm-scale features. 1 mm sits below the supported smallest-object
    # size while being large enough that a spinning SDF contact (like
    # the nut-bolt thread) can accumulate one frame of tangent motion
    # without the tangent bias spiking into the friction cone.
    friction_slop = wp.float32(0.001)

    # Accumulated warm-start impulse we'll apply to the bodies after
    # processing every active slot; batched so the body-velocity
    # scatter happens once, not 6x.
    total_lin_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)
    total_ang_imp_on_b1 = wp.vec3f(0.0, 0.0, 0.0)
    total_ang_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)

    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue

        base = _slot_base(base_offset, slot)
        k = contact_first + slot

        # Pull the persistent contact frame from the ``ContactContainer``
        # instead of Newton's per-frame narrow-phase output. Matched
        # contacts carry their ``(normal, tangent1, local_p0, local_p1)``
        # forward from the previous frame verbatim (see
        # :func:`_contact_warmstart_gather_kernel`); fresh contacts were
        # PhoenX-``Initialize``-d at gather time. Either way this slot's
        # frame is fixed for the duration of the contact's lifetime,
        # which is what makes the scalar ``lam_t1`` / ``lam_t2``
        # warm-starts physically meaningful.
        n = cc_get_normal(cc, slot, cid)
        t1_dir = cc_get_tangent1(cc, slot, cid)
        t2_dir = wp.cross(n, t1_dir)
        local_p0 = cc_get_local_p0(cc, slot, cid)
        local_p1 = cc_get_local_p1(cc, slot, cid)

        # Body-local -> world. Re-projecting the persistent local
        # anchors with the *current* orientations is PhoenX's
        # ``UpdatePosition``: the contact is rigidly attached to its
        # body-frame anchor, so the world-frame lever arm tracks any
        # rotation the body has undergone since the contact was born.
        #
        # Newton's narrow-phase convention: ``rigid_contact_point0/1``
        # is the body-local anchor *without* the surface offset. Each
        # shape carries an additional ``margin`` (radius + skin) so the
        # actual surface contact lies at
        # ``point0 + margin0 * normal_in_body_frame_0`` for shape 0,
        # and ``point1 - margin1 * normal_in_body_frame_1`` for shape 1
        # (normal points 0 -> 1, so shape 1's surface lies on the
        # *negative* normal side of its anchor). For convex primitives
        # like a sphere the anchor is the body centre and the margin
        # equals the radius -- without this shift the lever arm
        # ``r2 = p2 - position2`` collapses to zero and friction
        # contributes no torque, which is the bug that kept e.g. the
        # nut-on-bolt test from spinning. Mirror what XPBD does in its
        # friction path (``bx_b += transform_vector(X_wb, offset_b)``).
        p1_world = (
            position1
            + wp.quat_rotate(orientation1, local_p0)
            + contacts.rigid_contact_margin0[k] * n
        )
        p2_world = (
            position2
            + wp.quat_rotate(orientation2, local_p1)
            - contacts.rigid_contact_margin1[k] * n
        )

        r1 = p1_world - position1
        r2 = p2_world - position2

        eff_n = _effective_mass_scalar(n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
        eff_t1 = _effective_mass_scalar(
            t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )
        eff_t2 = _effective_mass_scalar(
            t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )

        # Signed separation along the *stored* normal. Now that
        # ``p1_world`` / ``p2_world`` already include the per-shape
        # margin shift to the actual surface, the gap *is* the
        # effective gap -- no extra margin subtraction needed.
        effective_gap = wp.dot(p2_world - p1_world, n)

        # Load-scaled contact correction: contacts under heavier
        # normal load get stronger position-level correction on both
        # the normal and tangent rows. Physically motivated -- at the
        # bottom of a tall stack each contact supports many cubes of
        # weight, and needs a correspondingly tighter position lock
        # than a lightly-loaded contact elsewhere in the scene. We
        # read the warm-started ``lam_n`` (previous substep's
        # converged impulse) as a proxy for the pair's normal load;
        # ``lam_n_ref = 1 / (eff_n * idt)`` is a "typical" contact
        # impulse so a reference-load contact gets boost 1.0 and
        # heavier contacts get proportionally more. The 4x cap
        # prevents warm-start noise from producing runaway
        # corrections.
        #
        # Applied uniformly to both normal and tangent biases so
        # static friction and penetration correction scale together
        # at the bottom of stacks. For light contacts (nut-on-bolt
        # SDF manifold, single objects on a plane) ``lam_n_ws`` is
        # small, ``load_boost ~ 1.0``, and the formulation reduces
        # to the un-boosted baseline -- which is what those cases
        # need to avoid over-correction.
        lam_n_ws = cc_get_normal_lambda(cc, slot, cid)
        lam_n_ref = wp.float32(1.0) / wp.max(eff_n * idt, wp.float32(1.0e-6))
        load_boost = wp.min(
            wp.float32(1.0) + lam_n_ws / lam_n_ref, wp.float32(4.0)
        )

        # Soft-constraint bias: ``bias = effective_gap * bias_rate``.
        # ``bias_rate`` has units 1/s (derived from hertz+damping) so
        # the formulation is scale-invariant in length -- resting
        # contacts converge to ``effective_gap ~ 0`` -> bias ~ 0 with
        # no length-scale slop.
        #
        # Recovery-speed cap: same idea as Box2D v3's
        # ``maxPushSpeed`` (solver2d post). Prevents a pathologically
        # deep initial penetration (dropped-through-ground tests, SDF
        # corner cases) from injecting tens of m/s of bias in a
        # single substep. The cap IS a velocity in m/s and therefore
        # the one remaining scale-sensitive knob; scale up with the
        # scene if you're simulating at ``scene_scale >> 1``.
        # ``max_approach_speed`` plays the symmetric role for the
        # speculative-closing branch.
        max_push_speed = wp.float32(2.0)
        max_approach_speed = wp.float32(10.0)
        bias_val = effective_gap * bias_rate
        bias_val = wp.clamp(bias_val, -max_push_speed, max_approach_speed)

        # ---- Tangential drift for static friction ----
        # Drift of body 2's contact anchor away from body 1's anchor,
        # projected onto the stored tangent frame. Solver2d-style
        # "sticky friction": if the drift has grown past the slip
        # threshold the bodies have physically slid past the static
        # regime, so reset the anchor to the current midpoint (breaks
        # the persistent friction anchor, equivalent to solver2d's
        # ``frictionPersisted = false``). Fresh anchor -> zero drift
        # this step, and the tangent lambdas are cleared so the
        # tangent row starts from scratch in the kinetic regime.
        p_diff = p2_world - p1_world
        drift_t1_raw = wp.dot(p_diff, t1_dir)
        drift_t2_raw = wp.dot(p_diff, t2_dir)
        # Sticky-break slip threshold. Same scale-discipline caveat as
        # ``penetration_slop``: a length in metres, tuned so that one
        # frame of typical tangent motion on a mm-scale feature does
        # *not* break the sticky anchor (the nut-bolt threads walk
        # ~sub-mm per frame at normal threading speed, so a 2 mm
        # threshold keeps the warm-started tangent lambda alive
        # across frames). Tightening this breaks the nut-bolt;
        # tuning downward makes sense only for finer contact features.
        slip_threshold = wp.float32(0.002)
        drift_sq = drift_t1_raw * drift_t1_raw + drift_t2_raw * drift_t2_raw

        # Break sticky on any of THREE conditions (solver2d pattern):
        #   * tangent drift past ``slip_threshold`` (bodies slid past
        #     the numerical-noise floor)
        #   * stored contact normal has rotated > ~18 degrees from
        #     the fresh narrow-phase normal (body rotation has aged
        #     the stored tangent basis)
        #   * previous-step tangent impulse saturated the Coulomb
        #     cap (|lam_t| >= 0.98 * mu * lam_n). This is the
        #     solver2d trigger for scenes where Coulomb saturates
        #     well below the drift threshold -- e.g. low-mu contacts
        #     (mu=0.01 on nut/bolt threads) where the tangent
        #     impulse reaches the cone with sub-0.1 mm drift and the
        #     nut would otherwise stay static-friction-locked to the
        #     bolt instead of threading down.
        # All three triggers fall through to the same reset: re-
        # anchor to the fresh narrow-phase contact point. Tangent
        # lambdas are preserved so kinetic friction keeps its
        # warm-start.
        fresh_n = contacts.rigid_contact_normal[k]
        normal_aligned = wp.dot(n, fresh_n)
        lam_t1_prev = cc_get_tangent1_lambda(cc, slot, cid)
        lam_t2_prev = cc_get_tangent2_lambda(cc, slot, cid)
        lam_n_prev = cc_get_normal_lambda(cc, slot, cid)
        fric_limit_prev = contact_get_friction(constraints, cid) * lam_n_prev
        cone_margin = wp.float32(0.98)
        lam_t_mag_sq = lam_t1_prev * lam_t1_prev + lam_t2_prev * lam_t2_prev
        coulomb_saturated = (
            lam_n_prev > wp.float32(0.0)
            and lam_t_mag_sq >= (cone_margin * fric_limit_prev) * (cone_margin * fric_limit_prev)
        )
        if (
            drift_sq > slip_threshold * slip_threshold
            or normal_aligned < wp.float32(0.95)
            or coulomb_saturated
        ):
            # Solver2d-style "break sticky": the stored anchor has
            # drifted past the static regime so reset it to the
            # current narrow-phase contact point. Using the fresh
            # narrow-phase anchors (not a midpoint of drifted points)
            # puts the new anchor exactly where the bodies actually
            # touch this frame, so drift is truly zero going into
            # iterate. Tangent lambdas are preserved: their world-
            # space impulse lam_t * t_dir is physically meaningful
            # regardless of anchor position and kinetic friction needs
            # the warm-start to reach Coulomb saturation within
            # iteration budget.
            fresh_lp0 = contacts.rigid_contact_point0[k]
            fresh_lp1 = contacts.rigid_contact_point1[k]
            cc_set_local_p0(cc, slot, cid, fresh_lp0)
            cc_set_local_p1(cc, slot, cid, fresh_lp1)
            # Coulomb-saturation reset is a true kinetic break: the
            # previous step's lam_t was at the cone boundary and its
            # direction is the PAST slip direction, which on a
            # rotating contact (nut-on-bolt threads, rolling ball)
            # is likely the OPPOSITE of the current slip direction.
            # Zero the tangent lambdas so fresh kinetic friction
            # re-accumulates along the current slip axis. The
            # drift / normal-rotation triggers still preserve lambda
            # (those are gentler breaks where the old direction is
            # still approximately right).
            if coulomb_saturated:
                cc_set_tangent1_lambda(cc, slot, cid, wp.float32(0.0))
                cc_set_tangent2_lambda(cc, slot, cid, wp.float32(0.0))
            # Mirror the surface-shift applied above so the reset
            # path lands on the actual surface contact point, not the
            # body-frame anchor (matters for sphere/SDF primitives
            # where the anchor is the body centre).
            p1_world = (
                position1
                + wp.quat_rotate(orientation1, fresh_lp0)
                + contacts.rigid_contact_margin0[k] * n
            )
            p2_world = (
                position2
                + wp.quat_rotate(orientation2, fresh_lp1)
                - contacts.rigid_contact_margin1[k] * n
            )
            r1 = p1_world - position1
            r2 = p2_world - position2
            eff_n = _effective_mass_scalar(n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
            eff_t1 = _effective_mass_scalar(t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
            eff_t2 = _effective_mass_scalar(t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
            p_diff = p2_world - p1_world
            drift_t1_raw = wp.dot(p_diff, t1_dir)
            drift_t2_raw = wp.dot(p_diff, t2_dir)
        drift_t1 = wp.clamp(drift_t1_raw, -friction_slop, friction_slop)
        drift_t2 = wp.clamp(drift_t2_raw, -friction_slop, friction_slop)
        bias_t1_val = friction_bias_factor * drift_t1 * idt * load_boost
        bias_t2_val = friction_bias_factor * drift_t2 * idt * load_boost

        _slot_set_r1(constraints, cid, base, r1)
        _slot_set_r2(constraints, cid, base, r2)
        _slot_set_eff_n(constraints, cid, base, eff_n)
        _slot_set_eff_t1(constraints, cid, base, eff_t1)
        _slot_set_eff_t2(constraints, cid, base, eff_t2)
        _slot_set_bias(constraints, cid, base, bias_val)
        _slot_set_bias_t1(constraints, cid, base, bias_t1_val)
        _slot_set_bias_t2(constraints, cid, base, bias_t2_val)

        # Warm-start impulse from the ContactContainer's *current*
        # buffers -- the gather kernel has already seeded them from
        # the match_index before this prepare runs. Build the total
        # contact impulse in world space and accumulate for a single
        # per-body scatter at the end.
        lam_n = cc_get_normal_lambda(cc, slot, cid)
        lam_t1 = cc_get_tangent1_lambda(cc, slot, cid)
        lam_t2 = cc_get_tangent2_lambda(cc, slot, cid)

        imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir

        total_lin_imp_on_b2 += imp
        total_ang_imp_on_b1 += wp.cross(r1, imp)
        total_ang_imp_on_b2 += wp.cross(r2, imp)

    # Scatter the combined warm-start impulse. Sign convention: the
    # contact impulse acts from body 1 onto body 2 along +normal, so
    # body 2 picks up ``+imp`` while body 1 picks up ``-imp``.
    if active_mask != 0:
        bodies.velocity[b1] = bodies.velocity[b1] - inv_mass1 * total_lin_imp_on_b2
        bodies.velocity[b2] = bodies.velocity[b2] + inv_mass2 * total_lin_imp_on_b2
        bodies.angular_velocity[b1] = bodies.angular_velocity[b1] - inv_inertia1 @ total_ang_imp_on_b1
        bodies.angular_velocity[b2] = bodies.angular_velocity[b2] + inv_inertia2 @ total_ang_imp_on_b2


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
    """One PGS sweep over every active slot of one contact column.

    Sequential Gauss-Seidel within the column (slots are processed in
    order 0..5 and each updates the body velocities read by the next).
    Across columns the outer graph-colouring partitioner guarantees no
    two cids in the same partition touch the same body, so the
    per-slot RMWs on ``bodies.{velocity, angular_velocity}`` need no
    atomics.

    Per slot, three scalar PGS rows are solved in this order: normal
    first (clamp accumulated ``lambda_n`` to ``>= 0``), then the two
    tangential rows (each clamp to ``[-mu * lambda_n, +mu * lambda_n]``
    using the *current* ``lambda_n``). Solving normal first stops the
    tangent clamp from under- or over-estimating the friction budget
    when the same slot is revisited over successive iterations.

    ``use_bias`` follows the Box2D v3 TGS-soft pattern: ``True`` during
    the main solve (positional drift + speculative separation push
    velocity toward closing the gap), ``False`` during the relax pass
    so the normal row enforces ``Jv = 0`` without re-injecting the
    positional-correction velocity. Injecting bias every relax pass
    is the main reason resting contacts drift sideways (the bias adds
    a tangent-normal mix that friction then has to chase).
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    active_mask = contact_get_active_mask(constraints, cid)
    if active_mask == 0:
        return

    contact_first = contact_get_contact_first(constraints, cid)
    # Two-regime Coulomb friction:
    #   * ``mu_s`` gates the slide transition -- if the per-iteration
    #     raw tangent impulse stays inside ``mu_s * lam_n`` the pair
    #     is "stuck" and the raw value is accepted verbatim.
    #   * ``mu_k`` is the clamp applied once the slide starts -- when
    #     the raw impulse breaches ``mu_s * lam_n`` we clamp
    #     magnitude-wise to ``mu_k * lam_n``. With ``mu_k <= mu_s``
    #     (PhysX / Material validation enforces this) a body that
    #     overcomes static friction settles into kinetic friction.
    #     When the two coefficients are equal (the common default),
    #     the two-regime clamp collapses to the single-coefficient
    #     circular cone.
    mu_s = contact_get_friction(constraints, cid)
    mu_k = contact_get_friction_dynamic(constraints, cid)

    # Soft-constraint coefficients for the normal row (see the same
    # block in ``contact_prepare_for_iteration_at``): recomputed from
    # ``idt`` here rather than stored per-slot so the contact
    # container schema stays unchanged. ``bias_rate`` was already
    # folded into ``bias_val`` during prepare; iterate only needs
    # ``mass_coeff`` and ``impulse_coeff``.
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

    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue

        base = _slot_base(base_offset, slot)

        # Geometry cached by ``prepare_for_iteration_at`` for this
        # slot. Lever arms ``r1`` / ``r2`` come from the column (they
        # depend on the substep's body orientation); the persistent
        # contact frame ``(n, t1, t2)`` comes from the ``ContactContainer``
        # (matched contacts carry it across frames, fresh contacts were
        # PhoenX-``Initialize``-d at gather time). ``t2`` is
        # ``cross(n, t1)`` -- stored ``t1`` is already unit, and
        # ``cross`` preserves unit length given orthogonality.
        r1 = _slot_get_r1(constraints, cid, base)
        r2 = _slot_get_r2(constraints, cid, base)
        n = cc_get_normal(cc, slot, cid)
        t1_dir = cc_get_tangent1(cc, slot, cid)
        t2_dir = wp.cross(n, t1_dir)
        eff_n = _slot_get_eff_n(constraints, cid, base)
        eff_t1 = _slot_get_eff_t1(constraints, cid, base)
        eff_t2 = _slot_get_eff_t2(constraints, cid, base)
        bias_val = _slot_get_bias(constraints, cid, base)
        bias_t1_val = _slot_get_bias_t1(constraints, cid, base)
        bias_t2_val = _slot_get_bias_t2(constraints, cid, base)
        if not use_bias:
            bias_val = wp.float32(0.0)
            bias_t1_val = wp.float32(0.0)
            bias_t2_val = wp.float32(0.0)

        # Relative velocity at the contact point (of body 2 seen from
        # body 1). ``v_contact = v2 + w2 x r2 - v1 - w1 x r1``. We
        # compute it ONCE per slot and project onto the three rows:
        # PhoenX's within-slot Jacobi. This is algorithmically
        # equivalent to within-slot Gauss-Seidel because the three
        # row axes (n, t1, t2) are orthonormal, so applying the
        # normal impulse changes ``vel_rel`` only in the normal
        # direction (linear) plus a cross-coupled angular component
        # that would show up in the tangent rows only if the
        # contact geometry had a non-degenerate angular coupling.
        # For practical contact geometries the coupling is small
        # enough that one-vel_rel-per-slot converges indistinguishably
        # from the full GS variant while halving the cross products
        # and cutting the angular-update FLOPS from 3x to 1x.
        # Empirically verified against tall-stack stability tests:
        # a GS variant (apply normal impulse, re-read velocities,
        # then solve tangents) regresses the 20-cube-tower side-slip
        # metric by 2.4x even though it slightly improves N=5 stacks.
        # The extra iteration coupling destabilises the friction
        # clamp on the bottom layers where the normal impulse is
        # huge and re-projection noise compounds.
        vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)

        jv_n = wp.dot(vel_rel, n)
        jv_t1 = wp.dot(vel_rel, t1_dir)
        jv_t2 = wp.dot(vel_rel, t2_dir)

        # ---- Normal row: Box2D v3 soft-constraint solve + clamp ----
        # ``d_lam_n_us`` is the unsoftened PGS delta; the softened
        # delta mixes it with a fraction of the accumulated lambda
        # so the effective constraint is a damped spring rather than
        # an infinitely stiff one (which is what required the old
        # ``penetration_slop`` dead zone to suppress zero-crossing
        # oscillation). Coefficients come from the same
        # ``soft_constraint_coefficients`` helper the joints use, fed
        # with ``DEFAULT_HERTZ_LINEAR`` + ``DEFAULT_DAMPING_RATIO``
        # so the response is critically damped at the Nyquist rate of
        # the current substep.
        d_lam_n_us = -eff_n * (jv_n + bias_val)
        lam_n_old = cc_get_normal_lambda(cc, slot, cid)
        d_lam_n = mass_coeff * d_lam_n_us - impulse_coeff * lam_n_old
        lam_n_new = wp.max(lam_n_old + d_lam_n, wp.float32(0.0))
        d_lam_n = lam_n_new - lam_n_old

        # Friction budgets use the *new* normal accum so the tangent
        # clamp matches the post-normal solution (PhoenX's
        # ``ApplyMax(...); maxTangentImpulse = friction *
        # accumulatedNormalImpulse;`` pattern). ``fric_limit_static``
        # is the "stick" budget; ``fric_limit_kinetic`` is what we
        # clamp to once the static threshold is breached.
        fric_limit_static = mu_s * lam_n_new
        fric_limit_kinetic = mu_k * lam_n_new

        # ---- Tangent rows: solve from pre-normal vel_rel ----
        # We trust the Coulomb clamp to do the static/dynamic regime
        # switch implicitly: when the body is stationary the position
        # bias dominates ``jv_t + bias`` and drives the anchor back;
        # once the body slides fast enough that ``|jv_t + bias|``
        # would exceed the cone, the clamp naturally caps the impulse
        # and the residual (kinetic friction) is the cone's surface.
        #
        # An explicit mode switch (only apply bias when not clamped,
        # velocity-only when clamped) was tried and regressed tall
        # stacks severely -- once the bottom cubes saturate their
        # cone, dropping the bias entirely loses the position-level
        # pull direction and lateral drift accumulates unbounded.
        # Reducing the bias on kinetic-regime iterations helps some
        # stacks but the tuning sweet spot is harder to find than
        # the always-on bias with a small gain (0.03), so we keep
        # the simpler formulation.
        #
        # Friction cone: we use a **circular** cone
        # ``sqrt(lam_t1^2 + lam_t2^2) <= mu * lam_n`` (matching XPBD,
        # MuJoCo, Bullet 3, PhysX). The earlier boxed variant
        # clamped each axis independently, which allows a diagonal
        # friction impulse of up to ``mu * sqrt(2) * lam_n`` -- a
        # 41 % over-shoot. For non-rotating stacks the difference is
        # cosmetic (the tangent basis tends to align with the slip
        # axis and the box/cone only differ on the corner), but for
        # rotating contacts (e.g. a nut on a helical SDF) the
        # extra corner budget lets the two tangent rows fight each
        # other and cancels out rotation the normal row should
        # drive. Circular-cone clamping keeps friction isotropic in
        # the tangent plane, which is the property the nut-bolt
        # threading relies on.
        d_lam_t1 = -eff_t1 * (jv_t1 + bias_t1_val)
        d_lam_t2 = -eff_t2 * (jv_t2 + bias_t2_val)
        lam_t1_old = cc_get_tangent1_lambda(cc, slot, cid)
        lam_t2_old = cc_get_tangent2_lambda(cc, slot, cid)
        lam_t1_raw = lam_t1_old + d_lam_t1
        lam_t2_raw = lam_t2_old + d_lam_t2
        # Two-regime circular cone. The raw 2D impulse
        # ``(lam_t1_raw, lam_t2_raw)`` is the tangent PGS row wants
        # to apply this iteration.
        #   1. If its magnitude sits inside the static budget
        #      (``mu_s * lam_n``) the contact is stuck -- accept the
        #      raw value as-is.
        #   2. Otherwise the contact is slipping. Rescale the raw
        #      impulse to the kinetic budget (``mu_k * lam_n``),
        #      preserving its direction so the friction impulse stays
        #      aligned with slip. This matches XPBD's
        #      ``lambda_fr = max(lambda_fr, -lambda_n * mu)`` along
        #      the single slip direction, promoted to 2D.
        # When ``mu_k == mu_s`` (the default for scenes that haven't
        # configured separate static + kinetic coefficients on the
        # material table) the two-regime clamp collapses to the
        # single-coefficient circular cone.
        lam_t_sq = lam_t1_raw * lam_t1_raw + lam_t2_raw * lam_t2_raw
        static_limit_sq = fric_limit_static * fric_limit_static
        if lam_t_sq > static_limit_sq and lam_t_sq > wp.float32(1.0e-30):
            # Kinetic regime.
            inv_mag = fric_limit_kinetic / wp.sqrt(lam_t_sq)
            lam_t1_new = lam_t1_raw * inv_mag
            lam_t2_new = lam_t2_raw * inv_mag
        else:
            # Static regime: raw value sits inside the stick budget.
            lam_t1_new = lam_t1_raw
            lam_t2_new = lam_t2_raw
        d_lam_t1 = lam_t1_new - lam_t1_old
        d_lam_t2 = lam_t2_new - lam_t2_old

        cc_set_normal_lambda(cc, slot, cid, lam_n_new)
        cc_set_tangent1_lambda(cc, slot, cid, lam_t1_new)
        cc_set_tangent2_lambda(cc, slot, cid, lam_t2_new)

        # ---- Combined world-space impulse: apply once ----
        # Building the full world impulse up front means the angular
        # update only needs one ``r x imp`` cross and one
        # ``invI @ ...`` matvec per body, versus three of each when
        # the rows were applied one at a time. For a 4-slot contact
        # column that's 16 crosses per cid per iter saved.
        imp = d_lam_n * n + d_lam_t1 * t1_dir + d_lam_t2 * t2_dir
        v1 -= inv_mass1 * imp
        v2 += inv_mass2 * imp
        w1 -= inv_inertia1 @ wp.cross(r1, imp)
        w2 += inv_inertia2 @ wp.cross(r2, imp)

    # Single body-velocity scatter after all slots -- mirrors what
    # ``ball_socket_iterate_at`` does. Graph coloring guarantees the
    # plain store is race-free.
    bodies.velocity[b1] = v1
    bodies.velocity[b2] = v2
    bodies.angular_velocity[b1] = w1
    bodies.angular_velocity[b2] = w2


@wp.func
def contact_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    """World-frame wrench this contact column applies to body 2.

    Sums the per-slot impulse ``lambda_n * n + lambda_t1 * t1 +
    lambda_t2 * t2`` across all active slots, divides by ``dt`` for
    a force, and computes the torque about body 2's COM using the
    cached lever arm ``r2``. Signs match the sign convention of
    ``*_iterate_at`` (contact impulse acts *from* body 1 *onto* body
    2), so the reported force / torque is what body 2 "felt" from
    the pair during the most recent substep.
    """
    active_mask = contact_get_active_mask(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue
        base = _slot_base(base_offset, slot)
        n = cc_get_normal(cc, slot, cid)
        t1_dir = cc_get_tangent1(cc, slot, cid)
        t2_dir = wp.cross(n, t1_dir)
        r2 = _slot_get_r2(constraints, cid, base)
        lam_n = cc_get_normal_lambda(cc, slot, cid)
        lam_t1 = cc_get_tangent1_lambda(cc, slot, cid)
        lam_t2 = cc_get_tangent2_lambda(cc, slot, cid)
        imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
        f = imp * idt
        force += f
        torque += wp.cross(r2, f)
    return force, torque


# ---------------------------------------------------------------------------
# Direct entry points (base_offset = 0)
# ---------------------------------------------------------------------------
#
# Thin wrappers mirroring the ``ball_socket_*`` direct entry points --
# read the body pair from this column's header and forward with
# ``base_offset = 0``. The unified dispatcher in ``solver_jitter_kernels``
# calls these.


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
    contact_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt, cc, contacts)


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
    contact_iterate_at(constraints, cid, 0, bodies, body_pair, idt, cc, contacts, use_bias)


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

    One sweep of a position-level PGS on the tangent rows of every
    active slot in one column. Directly modifies body positions and
    orientations (not velocities) so static-friction drift converges
    to zero within the iteration budget, bypassing the Nyquist-rate
    ceiling that velocity-level Baumgarte hits (~0.61/step).

    Algorithm per slot (Muller / Macklin XPBD, alpha=0 rigid):
        C_t_k = dot(p2_world - p1_world, t_k_dir)
        d_lam_k = -C_t_k * eff_t_k  (position impulse, kg*m)
        pos_imp = d_lam_1 * t1 + d_lam_2 * t2
        dx_1 = -inv_mass1 * pos_imp
        dx_2 = +inv_mass2 * pos_imp
        domega_1 = -inv_inertia1 * (r1 x pos_imp)
        domega_2 = +inv_inertia2 * (r2 x pos_imp)

    Per-body deltas accumulate across slots of the column and apply
    once at the end -- prevents the same body from being teleported
    multiple times by a 6-slot manifold.

    Gated on drift below the slip threshold: kinetic-regime pairs
    (drift > slip_threshold) are filtered out here and handled by the
    anchor-reset + velocity bias path. No Coulomb clamp in this pass
    because the static gate ensures we never would have saturated the
    cone anyway.

    Graph coloring guarantees body writes are race-free across
    threads in one kernel launch; sequential colors serialise so
    iteration k+1 reads iteration k's positions and converges the
    cascade.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    active_mask = contact_get_active_mask(constraints, cid)
    if active_mask == 0:
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
    v1 = bodies.velocity[b1]
    v2 = bodies.velocity[b2]
    w1 = bodies.angular_velocity[b1]
    w2 = bodies.angular_velocity[b2]

    slip_threshold = wp.float32(0.002)
    # Rest gate: 5 mm/s tangential contact velocity. A cube sliding at
    # 5 cm/s has a per-substep drift of 0.2 mm (well below slip_threshold)
    # but is kinetic, not static -- position correction on such a
    # pair would inject an opposing position-space impulse that
    # fights the kinetic friction velocity solve and breaks analytic
    # stop distance. Gating on velocity identifies the regime
    # properly.
    rest_vel_thresh = wp.float32(0.005)

    d_pos1 = wp.vec3f(0.0, 0.0, 0.0)
    d_pos2 = wp.vec3f(0.0, 0.0, 0.0)
    d_omega1 = wp.vec3f(0.0, 0.0, 0.0)
    d_omega2 = wp.vec3f(0.0, 0.0, 0.0)

    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue

        n = cc_get_normal(cc, slot, cid)
        t1_dir = cc_get_tangent1(cc, slot, cid)
        t2_dir = wp.cross(n, t1_dir)
        local_p0 = cc_get_local_p0(cc, slot, cid)
        local_p1 = cc_get_local_p1(cc, slot, cid)

        p1_world = position1 + wp.quat_rotate(orientation1, local_p0)
        p2_world = position2 + wp.quat_rotate(orientation2, local_p1)
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


@wp.func
def contact_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    return contact_world_wrench_at(constraints, cid, 0, idt, cc, contacts)


# ---------------------------------------------------------------------------
# Per-contact / per-column force reporting
# ---------------------------------------------------------------------------
#
# Two granularity levels on top of ``contact_world_wrench_at``:
#
# 1. ``contact_per_contact_wrench_kernel`` -- one entry per individual
#    contact ``k`` in the upstream :class:`Contacts` buffer. Index ``k``
#    is the same index used by ``rigid_contact_normal[k]`` etc., so the
#    output aligns 1:1 with Newton's contact arrays and can be zipped
#    with ``rigid_contact_point0`` / ``_point1`` / ``_shape0`` / etc.
#    Slots that are inactive (masked out of the column) leave
#    ``out[k]`` untouched -- caller zeros the buffer first.
#
# 2. ``contact_pair_wrench_kernel`` -- one entry per contact column,
#    summed across its up-to-six active slots. This is exactly
#    ``contact_world_wrench_at`` broadcast as a dense launch, packaged
#    alongside ``(body1, body2)`` so callers know which pair each
#    wrench belongs to. When a shape-pair spans multiple columns
#    (count > 6), each column reports its own partial sum and the
#    caller sums adjacent ``body1 == body2`` entries.
#
# Both kernels launch over the contact cid range
# ``[cid_base, cid_base + max_contact_columns)`` at fixed dim and
# early-out on empty / non-contact cids. This makes them safe inside
# ``wp.ScopedCapture``.


@wp.func
def contact_per_slot_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    slot: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    """World-frame wrench for one slot of a contact column.

    ``force`` is the normal + tangential impulse the contact applied
    to *body 2* during the most recent substep divided by ``dt``;
    ``torque`` is the moment of that force about body 2's COM using
    the cached lever arm ``r2``. Caller must have already verified
    the slot is active.
    """
    base = _slot_base(base_offset, slot)
    n = cc_get_normal(cc, slot, cid)
    t1_dir = cc_get_tangent1(cc, slot, cid)
    t2_dir = wp.cross(n, t1_dir)
    r2 = _slot_get_r2(constraints, cid, base)
    lam_n = cc_get_normal_lambda(cc, slot, cid)
    lam_t1 = cc_get_tangent1_lambda(cc, slot, cid)
    lam_t2 = cc_get_tangent2_lambda(cc, slot, cid)
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
    """Scatter per-slot wrenches into a per-rigid-contact output array.

    Launch dim must be at least ``cid_capacity - cid_base`` (one
    thread per potential contact column). Each thread iterates the
    6 slots of its column; active slots write
    ``out[rigid_contact_index] = spatial_vector(force, torque)``.
    Inactive slots leave ``out`` untouched, so the caller is expected
    to zero the buffer before the launch.
    """
    tid = wp.tid()
    cid = cid_base + tid
    if cid >= cid_capacity:
        return
    if constraint_get_type(constraints, cid) != CONSTRAINT_TYPE_CONTACT:
        return
    active_mask = contact_get_active_mask(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue
        f, tau = contact_per_slot_wrench_at(constraints, cid, 0, slot, idt, cc, contacts)
        out[contact_first + slot] = wp.spatial_vector(f, tau)


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
    contact_count: wp.array[wp.int32],
):
    """Per-column summary: total wrench + ``(body1, body2)`` + contact count.

    One output slot ``i = cid - cid_base`` per contact column. For
    inactive columns (type != CONTACT) every output is zeroed so
    callers can safely reduce without masking.

    Launch dim must equal ``cid_capacity - cid_base``.
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
        contact_count[tid] = 0
        return
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    active_mask = contact_get_active_mask(constraints, cid)
    count = wp.int32(0)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue
        f, tau = contact_per_slot_wrench_at(constraints, cid, 0, slot, idt, cc, contacts)
        force += f
        torque += tau
        count += 1
    wrenches[tid] = wp.spatial_vector(force, torque)
    body1[tid] = b1
    body2[tid] = b2
    contact_count[tid] = count


# ---------------------------------------------------------------------------
# Per-slot contact constraint error
# ---------------------------------------------------------------------------
#
# Mirrors the prepare kernel's ``effective_gap`` / tangent-drift
# computation but without the slop clamp so callers get the raw
# position-level residuals. Consumed by ``World.gather_contact_errors``.


@wp.func
def contact_per_slot_error_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    slot: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    cc: ContactContainer,
    contacts: ContactViews,
) -> wp.vec3f:
    """Position-level residual for one active contact slot.

    Computes the same quantities the prepare kernel folds into
    ``bias`` / ``bias_t1`` / ``bias_t2``, but without the
    ``friction_slop`` clamp so the raw residuals are surfaced:

      * ``gap = dot(p2_world - p1_world, n) - (margin0 + margin1)``
        -- signed separation along the stored normal, positive when
        strictly separating, negative when penetrating.
      * ``drift_t1 = dot(p2_world - p1_world, t1)`` -- unprojected
        tangential drift along the stored ``t1`` basis vector.
      * ``drift_t2 = dot(p2_world - p1_world, t2)`` where
        ``t2 = cross(n, t1)``.

    Anchor points are rebuilt from the persisted body-local anchors
    in the :class:`ContactContainer` and the current body pose --
    same refresh the prepare kernel performs each substep.

    Caller must have already verified the slot is active.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]

    contact_first = contact_get_contact_first(constraints, cid)
    k = contact_first + slot

    n = cc_get_normal(cc, slot, cid)
    t1_dir = cc_get_tangent1(cc, slot, cid)
    t2_dir = wp.cross(n, t1_dir)
    local_p0 = cc_get_local_p0(cc, slot, cid)
    local_p1 = cc_get_local_p1(cc, slot, cid)

    p1_world = position1 + wp.quat_rotate(orientation1, local_p0)
    p2_world = position2 + wp.quat_rotate(orientation2, local_p1)
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
    """Scatter per-slot residuals into a per-rigid-contact output array.

    Companion to :func:`contact_per_contact_wrench_kernel`: same launch
    contract, same index mapping. Each thread iterates the 6 slots of
    its contact column; active slots write
    ``out[contact_first + slot] = (gap, drift_t1, drift_t2)``
    computed by :func:`contact_per_slot_error_at`. Inactive slots
    leave ``out`` untouched, so the caller is expected to zero the
    buffer before the launch.
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
    active_mask = contact_get_active_mask(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue
        out[contact_first + slot] = contact_per_slot_error_at(
            constraints, cid, 0, slot, bodies, body_pair, cc, contacts
        )


@wp.func
def contact_world_error_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
) -> wp.spatial_vector:
    """Per-column constraint residual placeholder for contact cids.

    Contact columns report zero through the per-constraint dispatcher
    because meaningful contact residuals are per-*slot* (up to 6 per
    column) and live in a separate output array populated by
    :func:`contact_per_contact_error_kernel` /
    :meth:`World.gather_contact_errors`. Returning zero here keeps the
    :func:`_constraint_gather_errors_kernel` branch uniform with the
    other constraint types without contaminating the output with an
    arbitrary aggregate (sum? max? penetration-only?) that callers
    would have to un-pack anyway.
    """
    return wp.spatial_vector(
        wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0)
    )


@wp.func
def contact_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
) -> wp.spatial_vector:
    """Direct wrapper around :func:`contact_world_error_at`."""
    return contact_world_error_at(constraints, cid, 0)
