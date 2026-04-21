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
:class:`newton._src.solvers.jitter.contact_container.ContactContainer`
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
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_CONTACT,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_get_type,
    read_float,
    read_int,
    read_vec3,
    write_float,
    write_int,
    write_vec3,
)
from newton._src.solvers.jitter.contact_container import (
    MAX_SLOTS,
    ContactContainer,
    cc_get_normal_lambda,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords

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
    "contact_iterate",
    "contact_iterate_at",
    "contact_pair_wrench_kernel",
    "contact_per_contact_wrench_kernel",
    "contact_per_slot_wrench_at",
    "contact_prepare_for_iteration",
    "contact_prepare_for_iteration_at",
    "contact_set_active_mask",
    "contact_set_body1",
    "contact_set_body2",
    "contact_set_contact_first",
    "contact_set_friction",
    "contact_views_make",
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

    #: Coulomb friction coefficient for the pair. Pulled from the
    #: upstream material system at ingest time (today: fixed default
    #: because the Jitter solver doesn't yet wire through
    #: ``shape_material_mu``).
    friction: wp.float32

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
    return read_float(c, _OFF_FRICTION, cid)


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
    #
    # The bias term is signed so the same scalar handles both:
    #   * Penetration (``gap < 0``): ``bias`` becomes positive, which
    #     subtracts from ``-eff_n * (jv_n + bias)`` producing a
    #     positive normal impulse that pushes the pair apart.
    #   * Speculative separation (``gap > 0``): ``bias`` becomes
    #     negative. The PGS row solves for a normal impulse whose
    #     relative normal velocity equals ``-gap * idt`` -- i.e. just
    #     enough approach to close the current gap in one substep.
    #     Combined with the ``lam_n >= 0`` clamp this means "no push
    #     as long as the pair isn't approaching faster than closing
    #     the current gap", which is exactly Box2D's speculative
    #     contact handling.
    bias_factor = wp.float32(0.2)
    penetration_slop = wp.float32(0.005)

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

        # Body-local -> world for both anchors. Newton's
        # ``rigid_contact_point0/1`` are *body-frame* relative to the
        # body origin; Jitter's ``bodies.position`` == COM, so the
        # transform + subtract gives the COM-relative lever arm.
        local_p0 = contacts.rigid_contact_point0[k]
        local_p1 = contacts.rigid_contact_point1[k]
        p1_world = position1 + wp.quat_rotate(orientation1, local_p0)
        p2_world = position2 + wp.quat_rotate(orientation2, local_p1)

        r1 = p1_world - position1
        r2 = p2_world - position2

        n = contacts.rigid_contact_normal[k]

        t1_dir, t2_dir = _build_tangents(n)

        eff_n = _effective_mass_scalar(n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
        eff_t1 = _effective_mass_scalar(
            t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )
        eff_t2 = _effective_mass_scalar(
            t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
        )

        # Signed separation along the normal. The convention we
        # inherit from Newton: ``rigid_contact_normal`` points from
        # shape0 towards shape1, so ``gap = dot(p2 - p1, n)`` is
        # positive when the pair is separated and negative under
        # penetration. Surface thicknesses (margins) shrink the
        # effective gap so the solver rests at zero *overlap of the
        # rounded surfaces*, not at zero centre-to-centre separation.
        gap = wp.dot(p2_world - p1_world, n)
        margin_sum = contacts.rigid_contact_margin0[k] + contacts.rigid_contact_margin1[k]
        effective_gap = gap - margin_sum

        # Unified bias: one scalar that encodes both speculative
        # separation and Baumgarte-style penetration pushout. With a
        # slop dead-zone so idle resting contacts don't drift the
        # lambdas.
        #   * effective_gap > slop  -> bias > 0, allow approach up
        #                              to effective_gap/dt (speculative)
        #   * -slop < eg <= slop    -> bias = 0, resting contact
        #   * effective_gap < -slop -> bias < 0, Baumgarte pushout
        bias_val = wp.float32(0.0)
        if effective_gap > penetration_slop:
            bias_val = bias_factor * effective_gap * idt
        elif effective_gap < -penetration_slop:
            bias_val = bias_factor * (effective_gap + penetration_slop) * idt

        _slot_set_r1(constraints, cid, base, r1)
        _slot_set_r2(constraints, cid, base, r2)
        _slot_set_eff_n(constraints, cid, base, eff_n)
        _slot_set_eff_t1(constraints, cid, base, eff_t1)
        _slot_set_eff_t2(constraints, cid, base, eff_t2)
        _slot_set_bias(constraints, cid, base, bias_val)

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
    mu = contact_get_friction(constraints, cid)

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
        k = contact_first + slot

        # Geometry cached by ``prepare_for_iteration_at`` for this
        # slot. Normal is re-read from the upstream buffer (not cached
        # in the column) -- it's constant across substeps so the
        # per-iteration read is still cheap and keeps the column to a
        # manageable size. Tangents are rebuilt from the normal each
        # iteration via :func:`_build_tangents` -- ~20 FLOPS, saves
        # 6 dwords of storage per slot (and the per-slot write back
        # in prepare). Deterministic on ``n``, so the rebuilt basis
        # is bit-identical to what prepare would have cached.
        r1 = _slot_get_r1(constraints, cid, base)
        r2 = _slot_get_r2(constraints, cid, base)
        n = contacts.rigid_contact_normal[k]
        t1_dir, t2_dir = _build_tangents(n)
        eff_n = _slot_get_eff_n(constraints, cid, base)
        eff_t1 = _slot_get_eff_t1(constraints, cid, base)
        eff_t2 = _slot_get_eff_t2(constraints, cid, base)
        bias_val = _slot_get_bias(constraints, cid, base)
        if not use_bias:
            bias_val = wp.float32(0.0)

        # Relative velocity at the contact point (of body 2 seen from
        # body 1). ``v_contact = v2 + w2 x r2 - v1 - w1 x r1``.
        vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)

        # ---- Normal row ----
        jv_n = wp.dot(vel_rel, n)
        d_lam_n = -eff_n * (jv_n + bias_val)
        lam_n_old = cc_get_normal_lambda(cc, slot, cid)
        lam_n_new = wp.max(lam_n_old + d_lam_n, wp.float32(0.0))
        d_lam_n = lam_n_new - lam_n_old
        cc_set_normal_lambda(cc, slot, cid, lam_n_new)

        # Apply the corrective impulse before touching the tangent
        # rows so the tangent JV sees the post-normal velocities
        # (standard sequential-GS recipe for frictional contacts).
        imp_n = d_lam_n * n
        v1 -= inv_mass1 * imp_n
        v2 += inv_mass2 * imp_n
        w1 -= inv_inertia1 @ wp.cross(r1, imp_n)
        w2 += inv_inertia2 @ wp.cross(r2, imp_n)

        # Friction budget = mu * current normal impulse. When the
        # normal row is inactive (no penetration, lam_n = 0) the
        # tangent clamp pins friction to zero, which is the desired
        # "contact separating -> no friction" behaviour.
        fric_limit = mu * lam_n_new

        vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)

        # ---- Tangent 1 row ----
        jv_t1 = wp.dot(vel_rel, t1_dir)
        d_lam_t1 = -eff_t1 * jv_t1
        lam_t1_old = cc_get_tangent1_lambda(cc, slot, cid)
        lam_t1_new = wp.clamp(lam_t1_old + d_lam_t1, -fric_limit, fric_limit)
        d_lam_t1 = lam_t1_new - lam_t1_old
        cc_set_tangent1_lambda(cc, slot, cid, lam_t1_new)

        imp_t1 = d_lam_t1 * t1_dir
        v1 -= inv_mass1 * imp_t1
        v2 += inv_mass2 * imp_t1
        w1 -= inv_inertia1 @ wp.cross(r1, imp_t1)
        w2 += inv_inertia2 @ wp.cross(r2, imp_t1)

        vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)

        # ---- Tangent 2 row ----
        jv_t2 = wp.dot(vel_rel, t2_dir)
        d_lam_t2 = -eff_t2 * jv_t2
        lam_t2_old = cc_get_tangent2_lambda(cc, slot, cid)
        lam_t2_new = wp.clamp(lam_t2_old + d_lam_t2, -fric_limit, fric_limit)
        d_lam_t2 = lam_t2_new - lam_t2_old
        cc_set_tangent2_lambda(cc, slot, cid, lam_t2_new)

        imp_t2 = d_lam_t2 * t2_dir
        v1 -= inv_mass1 * imp_t2
        v2 += inv_mass2 * imp_t2
        w1 -= inv_inertia1 @ wp.cross(r1, imp_t2)
        w2 += inv_inertia2 @ wp.cross(r2, imp_t2)

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
    contact_first = contact_get_contact_first(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    for slot in range(CONTACT_MAX_SLOTS):
        if (active_mask & (wp.int32(1) << slot)) == 0:
            continue
        base = _slot_base(base_offset, slot)
        k = contact_first + slot
        n = contacts.rigid_contact_normal[k]
        t1_dir, t2_dir = _build_tangents(n)
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
    contact_first = contact_get_contact_first(constraints, cid)
    k = contact_first + slot
    n = contacts.rigid_contact_normal[k]
    t1_dir, t2_dir = _build_tangents(n)
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
