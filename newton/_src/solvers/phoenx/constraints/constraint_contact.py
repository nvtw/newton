# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unified contact constraint for :class:`PhoenXWorld`.

One column covers one whole ``(shape_a, shape_b)`` pair, storing only
the contiguous index range ``[contact_first, contact_first +
contact_count)`` into the sorted
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

## Three contact kinds via one schema

The same column header carries rigid-rigid, triangle-rigid (rigid vs
cloth-triangle) and triangle-triangle (cloth vs cloth) contacts.
Endpoint storage is *side-symmetric*: side A spans
``body0 / body1 / body2`` + ``weights_a``; side B spans
``body3 / body4 / body5`` + ``weights_b``. The
:data:`CONSTRAINT_TYPE_CONTACT_*` subtype tag at dword 0 selects the
fetch / apply path the iterate uses on each side:

* ``CONSTRAINT_TYPE_CONTACT_RIGID_RIGID``: each side uses one body
  index (``body0`` / ``body3``); the other slots are ``-1`` and the
  weights are ``(1, 0, 0)`` for compactness. This is the legacy
  rigid-rigid case; iterate / prepare on this subtype dead-code-
  eliminate the triangle path and stay bit-identical to the
  pre-unification rigid kernel.
* ``CONSTRAINT_TYPE_CONTACT_TRIANGLE_RIGID``: side A holds three
  cloth particle indices (``body0..body2``) + barycentric weights
  ``weights_a``; side B holds one rigid body (``body3``).
* ``CONSTRAINT_TYPE_CONTACT_TRIANGLE_TRIANGLE``: both sides hold
  three particle indices + their weights.

The unified body-or-particle indexing scheme (see
:mod:`newton._src.solvers.phoenx.body_or_particle`) lets every slot
address either a rigid body or a particle through one integer; the
fetch helpers in
:mod:`~newton._src.solvers.phoenx.constraints.contact_sides`
internalise the lookup.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.body_or_particle import (
    BodyOrParticleStore,
    get_velocity,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    assert_constraint_header,
    constraint_bodies_make,
    pd_coefficients,
    soft_constraint_coefficients,
)
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
from newton._src.solvers.phoenx.constraints.contact_sides import (
    apply_rigid_side,
    apply_rigid_side_batched,
    apply_triangle_side,
    fetch_rigid_pose,
    fetch_triangle_pose,
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
    "CONTACT_DWORDS",
    "CONTACT_KIND_RIGID",
    "CONTACT_KIND_TRIANGLE",
    "CONTACT_SUBTYPE_RIGID_RIGID",
    "CONTACT_SUBTYPE_RIGID_TRIANGLE",
    "CONTACT_SUBTYPE_TRIANGLE_TRIANGLE",
    "ContactColumnContainer",
    "ContactConstraintData",
    "ContactViews",
    "contact_column_container_zeros",
    "contact_get_body0",
    "contact_get_body1",
    "contact_get_body2",
    "contact_get_body3",
    "contact_get_body4",
    "contact_get_body5",
    "contact_get_body_a2",
    "contact_get_body_a3",
    "contact_get_body_b2",
    "contact_get_body_b3",
    "contact_get_contact_count",
    "contact_get_contact_first",
    "contact_get_friction",
    "contact_get_friction_dynamic",
    "contact_get_subtype",
    "contact_get_weights_a",
    "contact_get_weights_b",
    "contact_iterate_at_RR",
    "contact_iterate_at_RT",
    "contact_iterate_at_TT",
    "contact_iterate_at_multi_RR",
    "contact_iterate_at_multi_RT",
    "contact_iterate_at_multi_TT",
    "contact_pair_wrench_kernel",
    "contact_per_contact_error_kernel",
    "contact_per_contact_wrench_kernel",
    "contact_per_k_error_at",
    "contact_per_k_wrench_at",
    "contact_prepare_at_RR",
    "contact_prepare_at_RT",
    "contact_prepare_at_TT",
    "contact_set_body0",
    "contact_set_body1",
    "contact_set_body2",
    "contact_set_body3",
    "contact_set_body4",
    "contact_set_body5",
    "contact_set_body_a2",
    "contact_set_body_a3",
    "contact_set_body_b2",
    "contact_set_body_b3",
    "contact_set_contact_count",
    "contact_set_contact_first",
    "contact_set_friction",
    "contact_set_friction_dynamic",
    "contact_set_rigid_rigid_endpoints",
    "contact_set_rigid_triangle_endpoints",
    "contact_set_subtype",
    "contact_set_triangle_triangle_endpoints",
    "contact_set_weights_a",
    "contact_set_weights_b",
    "contact_views_make",
    "contact_world_error",
    "contact_world_error_at",
    "contact_world_wrench",
    "contact_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Subtype tags -- per-cid selector for the iterate's fetch / apply path
# ---------------------------------------------------------------------------
#
# The shared :data:`CONSTRAINT_TYPE_CONTACT` tag lives at dword 0 of every
# contact column (header contract). The per-cid *subtype* below lives at
# its own dedicated slot inside the contact-column header (the joint side
# of the cid space already uses dword 0 for *its own* constraint type, so
# cycling on dword 0 from the dispatcher would mean reading a different
# field for joints vs contacts). The subtype tag drives the kind of side
# A / side B endpoint the iterate fetches:
#
#   * RIGID_RIGID:        sideA = 1 rigid, sideB = 1 rigid   (pre-cloth path)
#   * TRIANGLE_RIGID:     sideA = 3 particles (cloth tri),   sideB = 1 rigid
#   * TRIANGLE_TRIANGLE:  sideA = 3 particles, sideB = 3 particles
#
# Values are tag constants for the subtype dword; they share the int32
# space but are *not* :data:`CONSTRAINT_TYPE_*` values. Ingest currently
# only writes :data:`CONTACT_SUBTYPE_RIGID_RIGID` -- triangle subtypes are
# wired by the cloth-narrow-phase work tracked separately.

#: Side kind for the iterate's compile-time fetch / apply selection.
#: Plain Python ints (not ``wp.constant``) because they're consumed by
#: ``wp.static(...)`` factory branches at compile time, not by kernels
#: at runtime.
CONTACT_KIND_RIGID: int = 0
CONTACT_KIND_TRIANGLE: int = 1

#: Subtype tag -- legacy rigid-rigid contact, side A and side B both
#: address one rigid body each. Slot 0 is the existing ``body1`` /
#: ``body2`` pair from the pre-unification schema; the extra slots
#: stay ``-1`` and the weights are ``(1, 0, 0)`` so the same fetch
#: helper works for rigid endpoints uniformly.
CONTACT_SUBTYPE_RIGID_RIGID = wp.constant(wp.int32(0))
#: Subtype tag -- side A is one rigid body, side B is a cloth
#: triangle (three particle indices + barycentric weights).  Side
#: ordering matches phoenx's natural shape index convention: rigid
#: shapes occupy ``[0, S)`` and cloth triangles occupy ``[S, S+T)``,
#: so the canonical ``shape_a < shape_b`` pair from the narrow phase
#: always has the rigid on side A.
CONTACT_SUBTYPE_RIGID_TRIANGLE = wp.constant(wp.int32(1))
#: Subtype tag -- both sides are cloth triangles.
CONTACT_SUBTYPE_TRIANGLE_TRIANGLE = wp.constant(wp.int32(2))


@wp.struct
class ContactConstraintData:
    """Column schema for one contact constraint -- unified across
    rigid-rigid, triangle-rigid, and triangle-triangle contacts.

    A single column covers one whole ``(shape_a, shape_b)`` pair; the
    ``[contact_first, contact_first + contact_count)`` range spans
    every contact the narrow phase emitted for that pair in the
    current step's sorted :class:`Contacts` buffer. All per-contact
    state lives in :class:`ContactContainer` and is indexed by the
    contact's sorted-buffer index ``k``; iterate / prepare loop from
    ``contact_first`` to ``contact_first + contact_count`` internally.

    Endpoints are *side-symmetric*: side A occupies ``body1 / body_a2 /
    body_a3`` (alias names ``body0 / body1 / body2``) plus
    ``weights_a``; side B occupies ``body2 / body_b2 / body_b3`` (alias
    names ``body3 / body4 / body5``) plus ``weights_b``. The two
    "header" slots ``body1`` / ``body2`` (dwords 1 / 2) are kept by
    name to satisfy
    :func:`~newton._src.solvers.phoenx.constraints.constraint_container.assert_constraint_header`;
    they double as the side-A / side-B representative slots.

    The per-cid :data:`subtype` tag (``CONTACT_SUBTYPE_*``) selects
    which side-fetch / side-apply path the iterate compiles to. The
    iterate factory closes over the (side_a_kind, side_b_kind) pair at
    Python build time so the wrong branches dead-code-eliminate.
    """

    #: Top-level constraint-type tag -- :data:`CONSTRAINT_TYPE_CONTACT`
    #: for every contact subtype. Must sit at dword 0 to satisfy the
    #: shared constraint-header contract.
    constraint_type: wp.int32
    #: Side-A representative endpoint -- a rigid body slot for
    #: :data:`CONTACT_SUBTYPE_RIGID_RIGID`, the first cloth particle
    #: for the triangle subtypes. Unified body-or-particle index.
    body1: wp.int32
    #: Side-B representative endpoint. Unified body-or-particle index.
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

    #: Per-cid contact subtype tag (:data:`CONTACT_SUBTYPE_*`). Drives
    #: the iterate's fetch / apply selection at the dispatcher level.
    subtype: wp.int32

    #: Side-A second / third endpoints. Used only for triangle side A
    #: (``CONTACT_SUBTYPE_TRIANGLE_*``); ``-1`` otherwise. Together with
    #: ``body1`` they form the three particle indices of the cloth
    #: triangle on side A, in the same order the cloth-triangle
    #: constraint stores them.
    body_a2: wp.int32
    body_a3: wp.int32

    #: Side-B second / third endpoints. Used only for triangle side B
    #: (``CONTACT_SUBTYPE_TRIANGLE_TRIANGLE``); ``-1`` otherwise. Same
    #: convention as ``body_a2`` / ``body_a3``.
    body_b2: wp.int32
    body_b3: wp.int32

    #: Side-A barycentric weights ``(w0, w1, w2)`` for the contact
    #: point on the triangle. Sums to ``1`` for triangle sides; for
    #: rigid sides this is ``(1, 0, 0)`` so the per-side fetch helper
    #: can run the weighted sum uniformly. Stored as a ``vec3f``
    #: (3 dwords).
    weights_a: wp.vec3f
    #: Side-B barycentric weights. Same convention as ``weights_a``.
    weights_b: wp.vec3f


assert_constraint_header(ContactConstraintData)


# ``body1`` / ``body2`` at dwords 1 / 2 are also the side-A / side-B
# representative slots; expose them under the "body0" / "body3" alias
# names so the per-side fetch helpers can talk in side-uniform indices
# (``body{0,1,2}`` for side A, ``body{3,4,5}`` for side B).
_OFF_BODY1 = wp.constant(dword_offset_of(ContactConstraintData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ContactConstraintData, "body2"))
_OFF_BODY_A2 = wp.constant(dword_offset_of(ContactConstraintData, "body_a2"))
_OFF_BODY_A3 = wp.constant(dword_offset_of(ContactConstraintData, "body_a3"))
_OFF_BODY_B2 = wp.constant(dword_offset_of(ContactConstraintData, "body_b2"))
_OFF_BODY_B3 = wp.constant(dword_offset_of(ContactConstraintData, "body_b3"))
_OFF_FRICTION = wp.constant(dword_offset_of(ContactConstraintData, "friction"))
_OFF_FRICTION_DYNAMIC = wp.constant(dword_offset_of(ContactConstraintData, "friction_dynamic"))
_OFF_CONTACT_FIRST = wp.constant(dword_offset_of(ContactConstraintData, "contact_first"))
_OFF_CONTACT_COUNT = wp.constant(dword_offset_of(ContactConstraintData, "contact_count"))
_OFF_SUBTYPE = wp.constant(dword_offset_of(ContactConstraintData, "subtype"))
_OFF_WEIGHTS_A = wp.constant(dword_offset_of(ContactConstraintData, "weights_a"))
_OFF_WEIGHTS_B = wp.constant(dword_offset_of(ContactConstraintData, "weights_b"))


#: Total dword count of one contact constraint column. The per-pair
#: layout drops every per-slot field so this is just the unified-
#: header footprint -- now ~16 dwords (6 endpoint slots, 2 ``vec3f``
#: weights, friction pair, range pair, two type tags). Compared with
#: the 154-dword ADBS joint header it's still tiny; contact columns
#: live in their own narrow-width :class:`ContactColumnContainer`
#: (below).
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
#
# Endpoint naming convention
# --------------------------
# The schema's mandatory header (dwords 1 / 2) carries ``body1`` and
# ``body2`` -- the legacy rigid-rigid pair, kept under those names so
# :func:`assert_constraint_header` is satisfied. Under the unified
# schema they double as the *side A representative slot* and *side B
# representative slot*; the per-side fetch helpers in
# :mod:`~newton._src.solvers.phoenx.constraints.contact_sides`
# additionally read four extra slots:
#
#     side A: body1, body_a2, body_a3   (cloth triangle 3-particle)
#     side B: body2, body_b2, body_b3   (cloth triangle 3-particle)
#
# Aliases ``body0`` / ``body3`` expose the header pair under the
# side-uniform name some callers prefer; ``body4`` / ``body5`` expose
# the side-B extras under the same naming.


@wp.func
def contact_get_body1(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Side-A representative slot (= header dword 1).

    For :data:`CONTACT_SUBTYPE_RIGID_RIGID` this is the rigid body
    index of side A; for triangle subtypes it's the first cloth-
    particle index of side A's triangle.
    """
    return _col_read_int(c, _OFF_BODY1, local_cid)


@wp.func
def contact_set_body1(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY1, local_cid, v)


@wp.func
def contact_get_body2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Side-B representative slot (= header dword 2)."""
    return _col_read_int(c, _OFF_BODY2, local_cid)


@wp.func
def contact_set_body2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY2, local_cid, v)


# Side-uniform aliases. ``body0`` == schema's ``body1``, etc.
@wp.func
def contact_get_body0(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Alias of :func:`contact_get_body1` (side-A representative)."""
    return _col_read_int(c, _OFF_BODY1, local_cid)


@wp.func
def contact_set_body0(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY1, local_cid, v)


@wp.func
def contact_get_body3(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Alias of :func:`contact_get_body2` (side-B representative)."""
    return _col_read_int(c, _OFF_BODY2, local_cid)


@wp.func
def contact_set_body3(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY2, local_cid, v)


# Triangle-only extra slots. Rigid sides leave these at ``-1``.
@wp.func
def contact_get_body_a2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Side-A second triangle particle. ``-1`` for rigid side A."""
    return _col_read_int(c, _OFF_BODY_A2, local_cid)


@wp.func
def contact_set_body_a2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY_A2, local_cid, v)


@wp.func
def contact_get_body_a3(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Side-A third triangle particle."""
    return _col_read_int(c, _OFF_BODY_A3, local_cid)


@wp.func
def contact_set_body_a3(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY_A3, local_cid, v)


@wp.func
def contact_get_body_b2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Side-B second triangle particle."""
    return _col_read_int(c, _OFF_BODY_B2, local_cid)


@wp.func
def contact_set_body_b2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY_B2, local_cid, v)


@wp.func
def contact_get_body_b3(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Side-B third triangle particle."""
    return _col_read_int(c, _OFF_BODY_B3, local_cid)


@wp.func
def contact_set_body_b3(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY_B3, local_cid, v)


# Side-uniform aliases for the triangle slots.
@wp.func
def contact_get_body4(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Alias of :func:`contact_get_body_b2`."""
    return _col_read_int(c, _OFF_BODY_B2, local_cid)


@wp.func
def contact_set_body4(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY_B2, local_cid, v)


@wp.func
def contact_get_body5(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Alias of :func:`contact_get_body_b3`."""
    return _col_read_int(c, _OFF_BODY_B3, local_cid)


@wp.func
def contact_set_body5(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_BODY_B3, local_cid, v)


@wp.func
def contact_get_subtype(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Per-cid contact subtype tag (:data:`CONTACT_SUBTYPE_*`)."""
    return _col_read_int(c, _OFF_SUBTYPE, local_cid)


@wp.func
def contact_set_subtype(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_SUBTYPE, local_cid, v)


@wp.func
def contact_get_weights_a(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3f:
    """Side-A barycentric weights ``(w0, w1, w2)``. Rigid side A reads
    back ``(1, 0, 0)``."""
    return wp.vec3f(
        _col_read_float(c, _OFF_WEIGHTS_A + 0, local_cid),
        _col_read_float(c, _OFF_WEIGHTS_A + 1, local_cid),
        _col_read_float(c, _OFF_WEIGHTS_A + 2, local_cid),
    )


@wp.func
def contact_set_weights_a(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3f):
    _col_write_float(c, _OFF_WEIGHTS_A + 0, local_cid, v[0])
    _col_write_float(c, _OFF_WEIGHTS_A + 1, local_cid, v[1])
    _col_write_float(c, _OFF_WEIGHTS_A + 2, local_cid, v[2])


@wp.func
def contact_get_weights_b(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3f:
    """Side-B barycentric weights. Rigid side B reads back ``(1, 0, 0)``."""
    return wp.vec3f(
        _col_read_float(c, _OFF_WEIGHTS_B + 0, local_cid),
        _col_read_float(c, _OFF_WEIGHTS_B + 1, local_cid),
        _col_read_float(c, _OFF_WEIGHTS_B + 2, local_cid),
    )


@wp.func
def contact_set_weights_b(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3f):
    _col_write_float(c, _OFF_WEIGHTS_B + 0, local_cid, v[0])
    _col_write_float(c, _OFF_WEIGHTS_B + 1, local_cid, v[1])
    _col_write_float(c, _OFF_WEIGHTS_B + 2, local_cid, v[2])


@wp.func
def contact_set_rigid_rigid_endpoints(
    c: ContactColumnContainer,
    local_cid: wp.int32,
    b1: wp.int32,
    b2: wp.int32,
):
    """Stamp the two rigid body indices + neutral defaults for every
    other endpoint slot.

    Sets ``body1 = b1`` (side A representative), ``body2 = b2`` (side B
    representative), the four extra triangle slots to ``-1``, both
    weight vectors to ``(1, 0, 0)``, and the subtype tag to
    :data:`CONTACT_SUBTYPE_RIGID_RIGID`. Used by
    :mod:`~newton._src.solvers.phoenx.constraints.contact_ingest` for
    every rigid-rigid contact column so the unified iterate's
    triangle-fetch reads a deterministic ``(1, 0, 0)`` weight even on
    the rigid-only fast path.
    """
    contact_set_body1(c, local_cid, b1)
    contact_set_body2(c, local_cid, b2)
    contact_set_body_a2(c, local_cid, wp.int32(-1))
    contact_set_body_a3(c, local_cid, wp.int32(-1))
    contact_set_body_b2(c, local_cid, wp.int32(-1))
    contact_set_body_b3(c, local_cid, wp.int32(-1))
    contact_set_weights_a(c, local_cid, wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0)))
    contact_set_weights_b(c, local_cid, wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0)))
    contact_set_subtype(c, local_cid, CONTACT_SUBTYPE_RIGID_RIGID)


@wp.func
def contact_set_rigid_triangle_endpoints(
    c: ContactColumnContainer,
    local_cid: wp.int32,
    b1: wp.int32,
    tri_a: wp.int32,
    tri_b: wp.int32,
    tri_c: wp.int32,
    weights_b: wp.vec3f,
):
    """Stamp side A rigid + side B cloth-triangle endpoints.

    Side A: ``body1 = b1`` (rigid body index), ``body_a2 = body_a3 = -1``,
    ``weights_a = (1, 0, 0)``. Side B: ``body2 = tri_a``, ``body_b2 = tri_b``,
    ``body_b3 = tri_c`` (the three particle indices in the unified
    body-or-particle space), ``weights_b = (w_a, w_b, w_c)`` (barycentric
    coordinates of the contact point on the triangle).  Subtype tag is
    :data:`CONTACT_SUBTYPE_RIGID_TRIANGLE`.
    """
    contact_set_body1(c, local_cid, b1)
    contact_set_body2(c, local_cid, tri_a)
    contact_set_body_a2(c, local_cid, wp.int32(-1))
    contact_set_body_a3(c, local_cid, wp.int32(-1))
    contact_set_body_b2(c, local_cid, tri_b)
    contact_set_body_b3(c, local_cid, tri_c)
    contact_set_weights_a(c, local_cid, wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0)))
    contact_set_weights_b(c, local_cid, weights_b)
    contact_set_subtype(c, local_cid, CONTACT_SUBTYPE_RIGID_TRIANGLE)


@wp.func
def contact_set_triangle_triangle_endpoints(
    c: ContactColumnContainer,
    local_cid: wp.int32,
    tri_a_a: wp.int32,
    tri_a_b: wp.int32,
    tri_a_c: wp.int32,
    weights_a: wp.vec3f,
    tri_b_a: wp.int32,
    tri_b_b: wp.int32,
    tri_b_c: wp.int32,
    weights_b: wp.vec3f,
):
    """Stamp side A + side B both as cloth triangles.

    Side A: ``body1 = tri_a_a``, ``body_a2 = tri_a_b``, ``body_a3 = tri_a_c``;
    ``weights_a`` = barycentric on triangle A.  Side B same with the
    second triangle's particles + weights.  Subtype tag is
    :data:`CONTACT_SUBTYPE_TRIANGLE_TRIANGLE`.
    """
    contact_set_body1(c, local_cid, tri_a_a)
    contact_set_body_a2(c, local_cid, tri_a_b)
    contact_set_body_a3(c, local_cid, tri_a_c)
    contact_set_body2(c, local_cid, tri_b_a)
    contact_set_body_b2(c, local_cid, tri_b_b)
    contact_set_body_b3(c, local_cid, tri_b_c)
    contact_set_weights_a(c, local_cid, weights_a)
    contact_set_weights_b(c, local_cid, weights_b)
    contact_set_subtype(c, local_cid, CONTACT_SUBTYPE_TRIANGLE_TRIANGLE)


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


# ---------------------------------------------------------------------------
# Prepare / iterate / multi-sweep (factory-generated)
# ---------------------------------------------------------------------------
#
# The unified ``wp.func`` definitions live further down behind
# :func:`_make_contact_prepare_at` / :func:`_make_contact_iterate_at` /
# :func:`_make_contact_iterate_at_multi`. Each factory specialises on
# ``(side_a_kind, side_b_kind)`` and emits one of (RIGID, RIGID) /
# (TRIANGLE, RIGID) / (TRIANGLE, TRIANGLE); the rigid-rigid case is
# what every dispatcher (single-world persistent / fused, multi-world
# fast-tail) calls when ``cloth_support=False``, with ``wp.static``
# dead-code-eliminating the triangle branches so the rigid-only build
# stays as fast as before the unification.


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
    # separation (see :func:`_make_contact_prepare_at`).
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


# ---------------------------------------------------------------------------
# Unified contact iterate / prepare / multi-iterate factory
# ---------------------------------------------------------------------------
#
# Every dispatcher (single-world persistent / fused, multi-world fast-
# tail) routes contact prepare / iterate through one of the three
# factory-built specialisations below:
#
#   * contact_iterate_at_RR / contact_prepare_at_RR / contact_iterate_at_multi_RR
#   * contact_iterate_at_RT / contact_prepare_at_RT / contact_iterate_at_multi_RT
#   * contact_iterate_at_TT / contact_prepare_at_TT / contact_iterate_at_multi_TT
#
# Rigid-only scenes (``cloth_support=False``) compile down to the
# ``_RR`` variant only -- the dispatcher's per-cid subtype check is
# erased by ``wp.static`` so the rigid-rigid hot path stays as fast as
# before the unification.
#
# All three are produced by the same Python factory, parameterised on
# (side_a_kind, side_b_kind) where each kind is one of
# :data:`CONTACT_KIND_RIGID` / :data:`CONTACT_KIND_TRIANGLE`. The
# per-row PGS math (Box2D v3 normal soft / speculative branches,
# two-tangent Coulomb cone, soft PD, sticky-break, warm-start
# scatter) is *identical* across the three; only the side-fetch /
# side-apply at the start / end of each contact iteration changes,
# and ``wp.static(side_a_kind)`` / ``wp.static(side_b_kind)``
# dead-code-eliminates the unused branches at compile time so each
# specialisation compiles to clean kind-specific kernel IR.
#
# The factories take a :class:`BodyOrParticleStore` rather than a
# :class:`BodyContainer` so a triangle side can address its three
# cloth-particle nodes through the unified body-or-particle index
# scheme. Rigid sides go through the same store: the
# :func:`fetch_rigid_*` / :func:`apply_rigid_side*` helpers in
# :mod:`~newton._src.solvers.phoenx.constraints.contact_sides` read /
# write through ``store.bodies`` directly so the rigid endpoints
# match the legacy hot path's body-velocity register caching
# behaviour.


@wp.func
def _read_side_indices_a(
    constraints: ContactColumnContainer,
    cid: wp.int32,
):
    """Read side-A's three slot indices + barycentric weights."""
    b1 = contact_get_body1(constraints, cid)
    b_a2 = contact_get_body_a2(constraints, cid)
    b_a3 = contact_get_body_a3(constraints, cid)
    w_a = contact_get_weights_a(constraints, cid)
    return b1, b_a2, b_a3, w_a


@wp.func
def _read_side_indices_b(
    constraints: ContactColumnContainer,
    cid: wp.int32,
):
    """Read side-B's three slot indices + barycentric weights."""
    b2 = contact_get_body2(constraints, cid)
    b_b2 = contact_get_body_b2(constraints, cid)
    b_b3 = contact_get_body_b3(constraints, cid)
    w_b = contact_get_weights_b(constraints, cid)
    return b2, b_b2, b_b3, w_b


def _make_contact_prepare_at(*, side_a_kind: int, side_b_kind: int):
    """Factory for the unified per-side contact prepare ``wp.func``.

    Generates one of three specialisations based on
    ``(side_a_kind, side_b_kind)``:

    * ``(RIGID, RIGID)`` -- rigid-rigid contact prepare. All math is
      routed through :class:`SideKinematics` / :class:`SidePose` so
      the same source serves all three cases; ``wp.static`` makes the
      triangle branches dead code in this specialisation.
    * ``(TRIANGLE, RIGID)`` -- side A is a cloth triangle; per-contact
      anchor lookup uses the barycentric weighted sum of node
      positions instead of a body-frame anchor projected through
      ``orientation``.
    * ``(TRIANGLE, TRIANGLE)`` -- both sides are cloth triangles.

    The per-row math (effective masses, Baumgarte / speculative bias,
    sticky-break threshold check, warm-start scatter) is the same on
    every variant; ``wp.static(side_*_kind)`` switches the side-fetch
    / side-apply only.
    """

    @wp.func
    def prepare_at(
        constraints: ContactColumnContainer,
        cid: wp.int32,
        base_offset: wp.int32,
        store: BodyOrParticleStore,
        idt: wp.float32,
        cc: ContactContainer,
        contacts: ContactViews,
    ):
        _ = base_offset

        contact_first = contact_get_contact_first(constraints, cid)
        contact_count = contact_get_contact_count(constraints, cid)
        if contact_count == 0:
            return

        # ---- Per-side hoists -----------------------------------------
        # Read the kind-specific endpoint slots + weights once per
        # column. Triangle sides cache the per-node inverse masses
        # via :func:`fetch_triangle_pose`; rigid sides read body
        # pose + inertia. ``wp.static`` makes the wrong branch dead
        # code in each specialisation.
        b1_a, b2_a, b3_a, weights_a = _read_side_indices_a(constraints, cid)
        b1_b, b2_b, b3_b, weights_b = _read_side_indices_b(constraints, cid)

        if wp.static(side_a_kind == CONTACT_KIND_RIGID):
            pose_a = fetch_rigid_pose(store.bodies, b1_a)
        else:
            pose_a = fetch_triangle_pose(store, b1_a, b2_a, b3_a, weights_a)

        if wp.static(side_b_kind == CONTACT_KIND_RIGID):
            pose_b = fetch_rigid_pose(store.bodies, b1_b)
        else:
            pose_b = fetch_triangle_pose(store, b1_b, b2_b, b3_b, weights_b)

        # ---- Soft-constraint coefficients (shared with legacy path) --
        dt_substep = wp.float32(1.0) / idt
        bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(
            DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
        )
        friction_bias_factor = wp.float32(0.08)
        friction_slop = wp.float32(0.001)
        max_push_speed = wp.float32(2.0)
        max_approach_speed = wp.float32(10.0)
        slip_threshold = wp.float32(0.002)

        mu_s_col = contact_get_friction(constraints, cid)
        two_pi = wp.float32(6.283185307179586)

        # Warm-start impulse accumulators -- only used for rigid sides
        # since their body-velocity scatter benefits from the
        # register-batched write at end of column. Triangle sides
        # apply per-contact inside the loop (per-node scatter can't be
        # coalesced across contacts because the three node indices
        # vary per contact).
        total_lin_imp_b = wp.vec3f(0.0, 0.0, 0.0)
        total_ang_imp_a = wp.vec3f(0.0, 0.0, 0.0)
        total_ang_imp_b = wp.vec3f(0.0, 0.0, 0.0)

        for i in range(contact_count):
            k = contact_first + i

            n = cc_get_normal(cc, k)
            t1_dir = cc_get_tangent1(cc, k)
            t2_dir = wp.cross(n, t1_dir)

            margin0 = contacts.rigid_contact_margin0[k]
            margin1 = contacts.rigid_contact_margin1[k]
            local_p0 = cc_get_local_p0(cc, k)
            local_p1 = cc_get_local_p1(cc, k)

            # Side-A contact-point world position. Rigid: body-frame
            # anchor projected through orientation, plus margin shift.
            # Triangle: barycentric weighted-sum world point (the
            # "anchor" lives in barycentric space, which is invariant
            # under the cloth's per-substep position update -- the
            # weights describe a fixed point on the triangle, not on
            # any individual node).
            if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                p_a_world = (
                    pose_a.position + wp.quat_rotate(pose_a.orientation, local_p0 - pose_a.body_com) + margin0 * n
                )
            else:
                p_a_world = pose_a.position + margin0 * n

            if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                p_b_world = (
                    pose_b.position + wp.quat_rotate(pose_b.orientation, local_p1 - pose_b.body_com) - margin1 * n
                )
            else:
                p_b_world = pose_b.position - margin1 * n

            # Lever arms -- zero for triangle sides (the
            # ``effective_mass_scalar`` helper collapses correctly
            # when r = 0 + inv_inertia = 0, and the prepare's
            # sticky-break drift uses ``p_diff = p_b - p_a`` directly,
            # not the lever arms).
            if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                r_a = p_a_world - pose_a.position
            else:
                r_a = wp.vec3f(0.0, 0.0, 0.0)
            if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                r_b = p_b_world - pose_b.position
            else:
                r_b = wp.vec3f(0.0, 0.0, 0.0)

            eff_n = effective_mass_scalar(
                n, r_a, r_b, pose_a.inv_mass, pose_b.inv_mass, pose_a.inv_inertia, pose_b.inv_inertia
            )
            eff_t1 = effective_mass_scalar(
                t1_dir, r_a, r_b, pose_a.inv_mass, pose_b.inv_mass, pose_a.inv_inertia, pose_b.inv_inertia
            )
            eff_t2 = effective_mass_scalar(
                t2_dir, r_a, r_b, pose_a.inv_mass, pose_b.inv_mass, pose_a.inv_inertia, pose_b.inv_inertia
            )

            effective_gap = wp.dot(p_b_world - p_a_world, n)

            # Load-scaled correction -- shared with legacy.
            lam_n_ws = cc_get_normal_lambda(cc, k)
            lam_n_ref = wp.float32(1.0) / wp.max(eff_n * idt, wp.float32(1.0e-6))
            load_boost = wp.min(wp.float32(1.0) + lam_n_ws / lam_n_ref, wp.float32(4.0))

            if effective_gap > wp.float32(0.0):
                bias_val = effective_gap * idt
            else:
                bias_val = effective_gap * bias_rate
            bias_val = wp.clamp(bias_val, -max_push_speed, max_approach_speed)

            # Sticky-friction drift / break test -- shared with legacy
            # via the unified ``p_diff`` computation. For triangle
            # sides ``local_p`` carries the (frozen) barycentric-
            # weight world anchor at the moment the contact was last
            # re-anchored; the comparison with the current weighted
            # world point yields the drift in the same units as the
            # rigid path.
            p_diff = p_b_world - p_a_world
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
                # Re-project the (now-fresh) anchors. Rigid sides
                # re-anchor in body frame; triangle sides keep their
                # weighted-sum (the per-cid weights stay the same;
                # the contact point is by construction the same
                # barycentric point on the triangle).
                if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                    p_a_world = (
                        pose_a.position + wp.quat_rotate(pose_a.orientation, fresh_lp0 - pose_a.body_com) + margin0 * n
                    )
                    r_a = p_a_world - pose_a.position
                if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                    p_b_world = (
                        pose_b.position + wp.quat_rotate(pose_b.orientation, fresh_lp1 - pose_b.body_com) - margin1 * n
                    )
                    r_b = p_b_world - pose_b.position
                eff_n = effective_mass_scalar(
                    n, r_a, r_b, pose_a.inv_mass, pose_b.inv_mass, pose_a.inv_inertia, pose_b.inv_inertia
                )
                eff_t1 = effective_mass_scalar(
                    t1_dir, r_a, r_b, pose_a.inv_mass, pose_b.inv_mass, pose_a.inv_inertia, pose_b.inv_inertia
                )
                eff_t2 = effective_mass_scalar(
                    t2_dir, r_a, r_b, pose_a.inv_mass, pose_b.inv_mass, pose_a.inv_inertia, pose_b.inv_inertia
                )
                p_diff = p_b_world - p_a_world
                drift_t1_raw = wp.dot(p_diff, t1_dir)
                drift_t2_raw = wp.dot(p_diff, t2_dir)

            drift_t1 = wp.clamp(drift_t1_raw, -friction_slop, friction_slop)
            drift_t2 = wp.clamp(drift_t2_raw, -friction_slop, friction_slop)
            bias_t1_val = friction_bias_factor * drift_t1 * idt * load_boost
            bias_t2_val = friction_bias_factor * drift_t2 * idt * load_boost

            cc_set_eff_n(cc, k, eff_n)
            cc_set_eff_t1(cc, k, eff_t1)
            cc_set_eff_t2(cc, k, eff_t2)
            cc_set_bias(cc, k, bias_val)
            cc_set_bias_t1(cc, k, bias_t1_val)
            cc_set_bias_t2(cc, k, bias_t2_val)

            # ---- Soft-contact PD plumbing (shared) --------------------
            k_n = wp.float32(0.0)
            c_n = wp.float32(0.0)
            stiffness_arr_len = contacts.rigid_contact_stiffness.shape[0]
            damping_arr_len = contacts.rigid_contact_damping.shape[0]
            if stiffness_arr_len > k:
                k_n = contacts.rigid_contact_stiffness[k]
            if damping_arr_len > k:
                c_n = contacts.rigid_contact_damping[k]
            if k_n < wp.float32(0.0) and eff_n > wp.float32(0.0):
                hertz = -k_n
                damping_ratio = wp.float32(0.0)
                if c_n < wp.float32(0.0):
                    damping_ratio = -c_n
                omega = two_pi * hertz
                k_n = omega * omega * eff_n
                c_n = wp.float32(2.0) * damping_ratio * omega * eff_n
            if (k_n > wp.float32(0.0) or c_n > wp.float32(0.0)) and eff_n > wp.float32(0.0):
                eff_inv_n = wp.float32(1.0) / eff_n
                pd_gamma_n, pd_bias_n, pd_eff_soft_n = pd_coefficients(
                    k_n, c_n, -effective_gap, eff_inv_n, dt_substep, PHOENX_BOOST_CONTACT_NORMAL
                )
                cc_set_pd_gamma(cc, k, pd_gamma_n)
                cc_set_pd_bias(cc, k, pd_bias_n)
                cc_set_pd_eff_soft(cc, k, pd_eff_soft_n)
            else:
                cc_set_pd_eff_soft(cc, k, wp.float32(0.0))

            # ---- Warm-start impulse scatter --------------------------
            lam_n = cc_get_normal_lambda(cc, k)
            lam_t1 = cc_get_tangent1_lambda(cc, k)
            lam_t2 = cc_get_tangent2_lambda(cc, k)
            imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir

            if wp.static(side_a_kind == CONTACT_KIND_RIGID and side_b_kind == CONTACT_KIND_RIGID):
                # Hot-path register-batched scatter (rigid-rigid).
                # ``-imp`` on side A, ``+imp`` on side B; angular
                # contributions accumulated against each body's
                # ``r``. Single warm-start scatter at end of column
                # mirrors the rigid-rigid prepare write-back.
                total_lin_imp_b += imp
                total_ang_imp_a += wp.cross(r_a, imp)
                total_ang_imp_b += wp.cross(r_b, imp)
            else:
                # Per-contact apply for triangle-bearing variants.
                if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                    apply_rigid_side(store.bodies, b1_a, imp, r_a, wp.float32(-1.0))
                else:
                    apply_triangle_side(store, b1_a, b2_a, b3_a, weights_a, imp, wp.float32(-1.0))
                if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                    apply_rigid_side(store.bodies, b1_b, imp, r_b, wp.float32(1.0))
                else:
                    apply_triangle_side(store, b1_b, b2_b, b3_b, weights_b, imp, wp.float32(1.0))

        # End-of-column register-batched scatter for the rigid-rigid
        # specialisation. The triangle-bearing variants already
        # applied per-contact inside the loop.
        if wp.static(side_a_kind == CONTACT_KIND_RIGID and side_b_kind == CONTACT_KIND_RIGID):
            store.bodies.velocity[b1_a] = store.bodies.velocity[b1_a] - pose_a.inv_mass * total_lin_imp_b
            store.bodies.velocity[b1_b] = store.bodies.velocity[b1_b] + pose_b.inv_mass * total_lin_imp_b
            store.bodies.angular_velocity[b1_a] = (
                store.bodies.angular_velocity[b1_a] - pose_a.inv_inertia @ total_ang_imp_a
            )
            store.bodies.angular_velocity[b1_b] = (
                store.bodies.angular_velocity[b1_b] + pose_b.inv_inertia @ total_ang_imp_b
            )

    return prepare_at


def _make_contact_iterate_at(*, side_a_kind: int, side_b_kind: int):
    """Factory for the unified per-side contact iterate ``wp.func``.

    Per-row PGS math (Box2D v3 normal soft / speculative branches,
    two-tangent Coulomb cone, soft PD plumbing, warm-start scatter)
    is identical across the (RR / TR / TT) specialisations; only the
    side-fetch / side-apply at the start / end of each contact
    iteration differs and is gated on ``wp.static(side_*_kind)`` so
    the wrong branch dead-code-eliminates per specialisation.
    """

    @wp.func
    def iterate_at(
        constraints: ContactColumnContainer,
        cid: wp.int32,
        base_offset: wp.int32,
        store: BodyOrParticleStore,
        idt: wp.float32,
        cc: ContactContainer,
        contacts: ContactViews,
        use_bias: wp.bool,
    ):
        _ = base_offset

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

        b1_a, b2_a, b3_a, weights_a = _read_side_indices_a(constraints, cid)
        b1_b, b2_b, b3_b, weights_b = _read_side_indices_b(constraints, cid)

        # Hoist body-side pose / inertia for rigid sides; triangle
        # sides re-fetch per contact because the per-node ``v_world``
        # is already a weighted sum over three nodes whose indices
        # don't vary across the column.
        if wp.static(side_a_kind == CONTACT_KIND_RIGID):
            orientation_a = store.bodies.orientation[b1_a]
            body_com_a = store.bodies.body_com[b1_a]
            inv_mass_a_rigid = store.bodies.inverse_mass[b1_a]
            inv_inertia_a_rigid = store.bodies.inverse_inertia_world[b1_a]
            v_a = store.bodies.velocity[b1_a]
            w_a = store.bodies.angular_velocity[b1_a]
        else:
            orientation_a = wp.quatf(0.0, 0.0, 0.0, 1.0)
            body_com_a = wp.vec3f(0.0, 0.0, 0.0)
            inv_mass_a_rigid = wp.float32(0.0)
            inv_inertia_a_rigid = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            v_a = wp.vec3f(0.0, 0.0, 0.0)
            w_a = wp.vec3f(0.0, 0.0, 0.0)

        if wp.static(side_b_kind == CONTACT_KIND_RIGID):
            orientation_b = store.bodies.orientation[b1_b]
            body_com_b = store.bodies.body_com[b1_b]
            inv_mass_b_rigid = store.bodies.inverse_mass[b1_b]
            inv_inertia_b_rigid = store.bodies.inverse_inertia_world[b1_b]
            v_b = store.bodies.velocity[b1_b]
            w_b = store.bodies.angular_velocity[b1_b]
        else:
            orientation_b = wp.quatf(0.0, 0.0, 0.0, 1.0)
            body_com_b = wp.vec3f(0.0, 0.0, 0.0)
            inv_mass_b_rigid = wp.float32(0.0)
            inv_inertia_b_rigid = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            v_b = wp.vec3f(0.0, 0.0, 0.0)
            w_b = wp.vec3f(0.0, 0.0, 0.0)

        for i in range(contact_count):
            k = contact_first + i

            n = cc_get_normal(cc, k)
            t1_dir = cc_get_tangent1(cc, k)
            t2_dir = wp.cross(n, t1_dir)

            local_p0 = cc_get_local_p0(cc, k)
            local_p1 = cc_get_local_p1(cc, k)
            margin0 = contacts.rigid_contact_margin0[k]
            margin1 = contacts.rigid_contact_margin1[k]

            # Side-A lever arm + contact-point velocity.
            if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                r_a = wp.quat_rotate(orientation_a, local_p0 - body_com_a) + margin0 * n
                v_a_world = v_a + wp.cross(w_a, r_a)
            else:
                # Triangle side: re-fetch the three particles' velocity
                # via the access-mode store, weighted-sum to the
                # contact-point velocity. ``r = 0`` for triangle.
                v_node_a0 = get_velocity(store, b1_a)
                v_node_a1 = get_velocity(store, b2_a)
                v_node_a2 = get_velocity(store, b3_a)
                v_a_world = weights_a[0] * v_node_a0 + weights_a[1] * v_node_a1 + weights_a[2] * v_node_a2
                r_a = wp.vec3f(0.0, 0.0, 0.0)

            if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                r_b = wp.quat_rotate(orientation_b, local_p1 - body_com_b) - margin1 * n
                v_b_world = v_b + wp.cross(w_b, r_b)
            else:
                v_node_b0 = get_velocity(store, b1_b)
                v_node_b1 = get_velocity(store, b2_b)
                v_node_b2 = get_velocity(store, b3_b)
                v_b_world = weights_b[0] * v_node_b0 + weights_b[1] * v_node_b1 + weights_b[2] * v_node_b2
                r_b = wp.vec3f(0.0, 0.0, 0.0)

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

            vel_rel = v_b_world - v_a_world

            jv_n = wp.dot(vel_rel, n)
            jv_t1 = wp.dot(vel_rel, t1_dir)
            jv_t2 = wp.dot(vel_rel, t2_dir)

            # ---- Normal row (shared) ---------------------------------
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

            # ---- Per-side apply --------------------------------------
            # Rigid sides update register-cached ``(v, w)`` so the
            # next contact in the column sees the impulse without
            # going through global memory; we scatter once at end of
            # column. Triangle sides scatter per contact because the
            # node indices vary per contact.
            if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                v_a, w_a = apply_rigid_side_batched(
                    v_a, w_a, inv_mass_a_rigid, inv_inertia_a_rigid, r_a, imp, wp.float32(-1.0)
                )
            else:
                apply_triangle_side(store, b1_a, b2_a, b3_a, weights_a, imp, wp.float32(-1.0))

            if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                v_b, w_b = apply_rigid_side_batched(
                    v_b, w_b, inv_mass_b_rigid, inv_inertia_b_rigid, r_b, imp, wp.float32(1.0)
                )
            else:
                apply_triangle_side(store, b1_b, b2_b, b3_b, weights_b, imp, wp.float32(1.0))

        if wp.static(side_a_kind == CONTACT_KIND_RIGID):
            store.bodies.velocity[b1_a] = v_a
            store.bodies.angular_velocity[b1_a] = w_a
        if wp.static(side_b_kind == CONTACT_KIND_RIGID):
            store.bodies.velocity[b1_b] = v_b
            store.bodies.angular_velocity[b1_b] = w_b

    return iterate_at


def _make_contact_iterate_at_multi(*, side_a_kind: int, side_b_kind: int):
    """Multi-sweep variant of :func:`_make_contact_iterate_at`.

    Same contract as :func:`_make_contact_iterate_at`: run
    ``num_sweeps`` PGS sweeps on the column with body-velocity /
    inertia / pose registers held across sweeps, then scatter once
    at the end. Triangle sides re-fetch per-node velocities each
    sweep because the column doesn't own those slots exclusively
    (graph coloring guarantees no other column touches the same
    triangle in the same colour, but sweeps within one column see
    the same nodes update each iteration). The cross-sweep
    register caching is a rigid-side optimisation that triangle
    sides skip; the math stays identical otherwise.
    """

    @wp.func
    def iterate_at_multi(
        constraints: ContactColumnContainer,
        cid: wp.int32,
        base_offset: wp.int32,
        store: BodyOrParticleStore,
        idt: wp.float32,
        cc: ContactContainer,
        contacts: ContactViews,
        use_bias: wp.bool,
        num_sweeps: wp.int32,
    ):
        _ = base_offset

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

        b1_a, b2_a, b3_a, weights_a = _read_side_indices_a(constraints, cid)
        b1_b, b2_b, b3_b, weights_b = _read_side_indices_b(constraints, cid)

        if wp.static(side_a_kind == CONTACT_KIND_RIGID):
            orientation_a = store.bodies.orientation[b1_a]
            body_com_a = store.bodies.body_com[b1_a]
            inv_mass_a_rigid = store.bodies.inverse_mass[b1_a]
            inv_inertia_a_rigid = store.bodies.inverse_inertia_world[b1_a]
            v_a = store.bodies.velocity[b1_a]
            w_a = store.bodies.angular_velocity[b1_a]
        else:
            orientation_a = wp.quatf(0.0, 0.0, 0.0, 1.0)
            body_com_a = wp.vec3f(0.0, 0.0, 0.0)
            inv_mass_a_rigid = wp.float32(0.0)
            inv_inertia_a_rigid = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            v_a = wp.vec3f(0.0, 0.0, 0.0)
            w_a = wp.vec3f(0.0, 0.0, 0.0)

        if wp.static(side_b_kind == CONTACT_KIND_RIGID):
            orientation_b = store.bodies.orientation[b1_b]
            body_com_b = store.bodies.body_com[b1_b]
            inv_mass_b_rigid = store.bodies.inverse_mass[b1_b]
            inv_inertia_b_rigid = store.bodies.inverse_inertia_world[b1_b]
            v_b = store.bodies.velocity[b1_b]
            w_b = store.bodies.angular_velocity[b1_b]
        else:
            orientation_b = wp.quatf(0.0, 0.0, 0.0, 1.0)
            body_com_b = wp.vec3f(0.0, 0.0, 0.0)
            inv_mass_b_rigid = wp.float32(0.0)
            inv_inertia_b_rigid = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            v_b = wp.vec3f(0.0, 0.0, 0.0)
            w_b = wp.vec3f(0.0, 0.0, 0.0)

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

                if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                    r_a = wp.quat_rotate(orientation_a, local_p0 - body_com_a) + margin0 * n
                    v_a_world = v_a + wp.cross(w_a, r_a)
                else:
                    v_node_a0 = get_velocity(store, b1_a)
                    v_node_a1 = get_velocity(store, b2_a)
                    v_node_a2 = get_velocity(store, b3_a)
                    v_a_world = weights_a[0] * v_node_a0 + weights_a[1] * v_node_a1 + weights_a[2] * v_node_a2
                    r_a = wp.vec3f(0.0, 0.0, 0.0)

                if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                    r_b = wp.quat_rotate(orientation_b, local_p1 - body_com_b) - margin1 * n
                    v_b_world = v_b + wp.cross(w_b, r_b)
                else:
                    v_node_b0 = get_velocity(store, b1_b)
                    v_node_b1 = get_velocity(store, b2_b)
                    v_node_b2 = get_velocity(store, b3_b)
                    v_b_world = weights_b[0] * v_node_b0 + weights_b[1] * v_node_b1 + weights_b[2] * v_node_b2
                    r_b = wp.vec3f(0.0, 0.0, 0.0)

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

                vel_rel = v_b_world - v_a_world

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

                if wp.static(side_a_kind == CONTACT_KIND_RIGID):
                    v_a, w_a = apply_rigid_side_batched(
                        v_a, w_a, inv_mass_a_rigid, inv_inertia_a_rigid, r_a, imp, wp.float32(-1.0)
                    )
                else:
                    apply_triangle_side(store, b1_a, b2_a, b3_a, weights_a, imp, wp.float32(-1.0))

                if wp.static(side_b_kind == CONTACT_KIND_RIGID):
                    v_b, w_b = apply_rigid_side_batched(
                        v_b, w_b, inv_mass_b_rigid, inv_inertia_b_rigid, r_b, imp, wp.float32(1.0)
                    )
                else:
                    apply_triangle_side(store, b1_b, b2_b, b3_b, weights_b, imp, wp.float32(1.0))
            it += 1

        if wp.static(side_a_kind == CONTACT_KIND_RIGID):
            store.bodies.velocity[b1_a] = v_a
            store.bodies.angular_velocity[b1_a] = w_a
        if wp.static(side_b_kind == CONTACT_KIND_RIGID):
            store.bodies.velocity[b1_b] = v_b
            store.bodies.angular_velocity[b1_b] = w_b

    return iterate_at_multi


# Three specialisations -- one per (side_a_kind, side_b_kind) pair.
# Names follow the schema's subtype convention:
#   _RR -> CONTACT_SUBTYPE_RIGID_RIGID
#   _RT -> CONTACT_SUBTYPE_RIGID_TRIANGLE  (side A rigid, side B triangle)
#   _TT -> CONTACT_SUBTYPE_TRIANGLE_TRIANGLE
#
# The mixed case is RT (not TR) because phoenx's natural shape index
# ordering puts rigid shapes in ``[0, S)`` and cloth triangles in
# ``[S, S+T)``, so the canonical ``shape_a < shape_b`` pair from the
# narrow phase always has the rigid on side A.
contact_prepare_at_RR = _make_contact_prepare_at(side_a_kind=CONTACT_KIND_RIGID, side_b_kind=CONTACT_KIND_RIGID)
contact_prepare_at_RT = _make_contact_prepare_at(side_a_kind=CONTACT_KIND_RIGID, side_b_kind=CONTACT_KIND_TRIANGLE)
contact_prepare_at_TT = _make_contact_prepare_at(side_a_kind=CONTACT_KIND_TRIANGLE, side_b_kind=CONTACT_KIND_TRIANGLE)

contact_iterate_at_RR = _make_contact_iterate_at(side_a_kind=CONTACT_KIND_RIGID, side_b_kind=CONTACT_KIND_RIGID)
contact_iterate_at_RT = _make_contact_iterate_at(side_a_kind=CONTACT_KIND_RIGID, side_b_kind=CONTACT_KIND_TRIANGLE)
contact_iterate_at_TT = _make_contact_iterate_at(side_a_kind=CONTACT_KIND_TRIANGLE, side_b_kind=CONTACT_KIND_TRIANGLE)

contact_iterate_at_multi_RR = _make_contact_iterate_at_multi(
    side_a_kind=CONTACT_KIND_RIGID, side_b_kind=CONTACT_KIND_RIGID
)
contact_iterate_at_multi_RT = _make_contact_iterate_at_multi(
    side_a_kind=CONTACT_KIND_RIGID, side_b_kind=CONTACT_KIND_TRIANGLE
)
contact_iterate_at_multi_TT = _make_contact_iterate_at_multi(
    side_a_kind=CONTACT_KIND_TRIANGLE, side_b_kind=CONTACT_KIND_TRIANGLE
)
