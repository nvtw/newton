# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rigid-rigid contact constraint.

One CONTACT column per (shape_a, shape_b) pair, storing only the index range
[contact_first, contact_first + contact_count) into Newton's sorted Contacts.
Per-contact state lives in :class:`ContactContainer`, keyed by sorted-buffer k.
PGS walks each pair's range serially (Gauss-Seidel within the pair).

Friction: two-tangent pyramidal PGS. Normal lambda_n >= 0; tangents clamped to
[-mu*lambda_n, +mu*lambda_n] with circular clamp (Coulomb cone approximation).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import (
    BodyContainer,
    body_load_inv_inertia_sym6,
    body_load_orientation,
    body_load_vw,
    body_store_vw,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    assert_constraint_header,
    constraint_bodies_make,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    CC_DERIVED_DWORDS_PER_CONTACT,
    CC_DWORDS_PER_CONTACT,
    CC_IMPULSE_DWORDS_PER_CONTACT,
    CC_LOCAL_ANCHOR_DWORDS,
    CC_LOCAL_ANCHOR_FIRST_ROW,
    CC_RIGID_DWORDS_PER_CONTACT,
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
    cc_get_start_gap,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_start_gap,
)
from newton._src.solvers.phoenx.constraints.contact_patch_friction import (
    ContactPatchFriction,
    contact_patch_friction_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_projection import (
    contact_frame_velocity_update,
    contact_frame_velocity_update_no_soft_pd,
)
from newton._src.solvers.phoenx.helpers.array_access import read2d_f32, write2d_f32
from newton._src.solvers.phoenx.helpers.data_packing import (
    dword_offset_of,
    num_dwords,
    reinterpret_float_as_int,
    reinterpret_int_as_float,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "CONTACT_DWORDS",
    "CONTACT_TIME_US_OFFSET",
    "ContactColumnContainer",
    "ContactConstraintData",
    "ContactViews",
    "contact_accumulate_time_us",
    "contact_build_colored_row_offsets",
    "contact_column_container_zeros",
    "contact_gather_colored_rows",
    "contact_get_body1",
    "contact_get_body2",
    "contact_get_contact_count",
    "contact_get_contact_first",
    "contact_get_count1",
    "contact_get_count2",
    "contact_get_friction",
    "contact_get_friction_dynamic",
    "contact_get_original_contact_first",
    "contact_get_side0_counts_extra",
    "contact_get_side0_kind",
    "contact_get_side0_nodes_extra",
    "contact_get_side0_slots_extra",
    "contact_get_side1_counts_extra",
    "contact_get_side1_kind",
    "contact_get_side1_nodes_extra",
    "contact_get_side1_slots_extra",
    "contact_get_slot1",
    "contact_get_slot2",
    "contact_iterate_at_multi",
    "contact_iterate_at_multi_no_soft_pd",
    "contact_iterate_multi",
    "contact_iterate_multi_no_soft_pd",
    "contact_pack_colored_headers",
    "contact_pair_wrench_kernel",
    "contact_per_contact_error_kernel",
    "contact_per_contact_wrench_kernel",
    "contact_per_k_error_at",
    "contact_per_k_wrench_at",
    "contact_scatter_colored_rows",
    "contact_set_body1",
    "contact_set_body2",
    "contact_set_contact_count",
    "contact_set_contact_first",
    "contact_set_count1",
    "contact_set_count2",
    "contact_set_friction",
    "contact_set_friction_dynamic",
    "contact_set_side0_counts_extra",
    "contact_set_side0_kind",
    "contact_set_side0_nodes_extra",
    "contact_set_side0_slots_extra",
    "contact_set_side1_counts_extra",
    "contact_set_side1_kind",
    "contact_set_side1_nodes_extra",
    "contact_set_side1_slots_extra",
    "contact_set_slot1",
    "contact_set_slot2",
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
    """Column schema for one ``(shape_a, shape_b)`` contact pair.

    The ``[contact_first, contact_first + contact_count)`` range covers
    every per-pair contact in the current step's sorted :class:`Contacts`
    buffer; per-contact state lives in :class:`ContactContainer` keyed
    by ``k``.
    """

    #: Constraint type tag (must be at dword 0 for header contract).
    constraint_type: wp.int32
    #: ``model.shape_body[shape_a]``.
    body1: wp.int32
    #: ``model.shape_body[shape_b]``.
    body2: wp.int32

    #: Static Coulomb friction coefficient (stick threshold).
    friction: wp.float32
    #: Kinetic friction coefficient (slip clamp). Typically <= static.
    friction_dynamic: wp.float32

    #: Start index into the sorted Contacts buffer.
    contact_first: wp.int32
    #: Number of contacts in this column.
    contact_count: wp.int32

    #: Mass-splitting slot cache for the first endpoint node on each side.
    slot1: wp.int32
    slot2: wp.int32
    #: Per-endpoint slot count used as Tonge's inverse-mass factor.
    #: ``1`` means no split / no slot.
    count1: wp.int32
    count2: wp.int32

    #: Endpoint kind per side: ``0`` = rigid body, ``1`` = cloth triangle,
    #: ``2`` = soft tetrahedron. ``body1`` / ``body2`` store the first
    #: unified body-or-particle index; ``*_nodes_extra`` holds the 2nd /
    #: 3rd / 4th element particles (``(-1, -1, -1)`` for rigid; the cloth
    #: path stores ``(extra0, extra1, -1)`` with the 4th slot as a sentinel;
    #: soft-tet fills all three).
    side0_kind: wp.int32
    side1_kind: wp.int32
    side0_nodes_extra: wp.vec3i
    side1_nodes_extra: wp.vec3i

    #: Slot cache for endpoint nodes 1..3 on each side. Rigid endpoints
    #: leave these at ``-1``.
    side0_slots_extra: wp.vec3i
    side1_slots_extra: wp.vec3i
    side0_counts_extra: wp.vec3i
    side1_counts_extra: wp.vec3i

    #: Opt-in per-column wall-clock accumulator (microseconds). Written
    #: atomically by the head/tail dispatch when
    #: :attr:`PhoenXWorld.enable_column_timers` is set; cleared at the
    #: start of every :meth:`PhoenXWorld.step`.
    time_us: wp.float32


assert_constraint_header(ContactConstraintData)


_OFF_BODY1 = wp.constant(dword_offset_of(ContactConstraintData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ContactConstraintData, "body2"))
_OFF_FRICTION = wp.constant(dword_offset_of(ContactConstraintData, "friction"))
_OFF_FRICTION_DYNAMIC = wp.constant(dword_offset_of(ContactConstraintData, "friction_dynamic"))
_OFF_CONTACT_FIRST = wp.constant(dword_offset_of(ContactConstraintData, "contact_first"))
_OFF_CONTACT_COUNT = wp.constant(dword_offset_of(ContactConstraintData, "contact_count"))
_OFF_SIDE0_KIND = wp.constant(dword_offset_of(ContactConstraintData, "side0_kind"))
_OFF_SIDE1_KIND = wp.constant(dword_offset_of(ContactConstraintData, "side1_kind"))
_OFF_SIDE0_NODES_EXTRA = wp.constant(dword_offset_of(ContactConstraintData, "side0_nodes_extra"))
_OFF_SIDE1_NODES_EXTRA = wp.constant(dword_offset_of(ContactConstraintData, "side1_nodes_extra"))
_OFF_SLOT1 = wp.constant(dword_offset_of(ContactConstraintData, "slot1"))
_OFF_SLOT2 = wp.constant(dword_offset_of(ContactConstraintData, "slot2"))
_OFF_SIDE0_SLOTS_EXTRA = wp.constant(dword_offset_of(ContactConstraintData, "side0_slots_extra"))
_OFF_SIDE1_SLOTS_EXTRA = wp.constant(dword_offset_of(ContactConstraintData, "side1_slots_extra"))
_OFF_COUNT1 = wp.constant(dword_offset_of(ContactConstraintData, "count1"))
_OFF_COUNT2 = wp.constant(dword_offset_of(ContactConstraintData, "count2"))
_OFF_SIDE0_COUNTS_EXTRA = wp.constant(dword_offset_of(ContactConstraintData, "side0_counts_extra"))
_OFF_SIDE1_COUNTS_EXTRA = wp.constant(dword_offset_of(ContactConstraintData, "side1_counts_extra"))
CONTACT_TIME_US_OFFSET = wp.constant(dword_offset_of(ContactConstraintData, "time_us"))


#: Dwords per contact column. The per-pair layout drops every per-slot
#: field, leaving just the header (~22x smaller than the joint container).
CONTACT_DWORDS: int = num_dwords(ContactConstraintData)
# Rigid mass-splitting reads the prefix through ``count2``. Endpoint
# metadata occupies holes in that prefix but is compile-time dead in the
# rigid solve; retaining the original offsets avoids a second getter family.
_OFF_ORIGINAL_CONTACT_FIRST: int = int(_OFF_COUNT2) + 1
RIGID_CONTACT_SOLVE_DWORDS: int = _OFF_ORIGINAL_CONTACT_FIRST + 1


# ---------------------------------------------------------------------------
# ContactColumnContainer -- narrow sibling of ConstraintContainer for contact
# cids. ``local_cid = global_cid - num_joints``.
# ---------------------------------------------------------------------------


@wp.struct
class ContactColumnContainer:
    """Per-shape-pair column header storage for contacts.

    Shape ``(CONTACT_DWORDS, num_columns)``. Per-contact solver state
    (lambdas, eff masses, bias) lives in :class:`ContactContainer`
    keyed by ``k``.
    """

    data: wp.array2d[wp.float32]
    patch: ContactPatchFriction
    articulation_owner: wp.array[wp.int32]


def contact_column_container_zeros(
    num_columns: int,
    device: wp.DeviceLike = None,
    *,
    enable_patch_friction: bool = False,
    data_dwords: int = CONTACT_DWORDS,
) -> ContactColumnContainer:
    """Allocate a zero-initialised :class:`ContactColumnContainer`."""
    c = ContactColumnContainer()
    c.data = wp.zeros((int(data_dwords), int(num_columns)), dtype=wp.float32, device=device)
    c.articulation_owner = wp.full(int(num_columns), -1, dtype=wp.int32, device=device)
    patch_capacity = num_columns if enable_patch_friction else 1
    c.patch = contact_patch_friction_zeros(patch_capacity, device=device)
    return c


# Column-local int/float read+write helpers, keyed by ``local_cid``.
@wp.func
def _col_read_int(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32) -> wp.int32:
    return reinterpret_float_as_int(read2d_f32(c.data, off, local_cid))


@wp.func
def _col_write_int(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32, v: wp.int32):
    write2d_f32(c.data, off, local_cid, reinterpret_int_as_float(v))


@wp.func
def _col_read_float(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32) -> wp.float32:
    return read2d_f32(c.data, off, local_cid)


@wp.func
def contact_get_original_contact_first(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    """Return the canonical first-contact index stored in a packed header."""
    return _col_read_int(c, wp.int32(_OFF_ORIGINAL_CONTACT_FIRST), local_cid)


@wp.kernel(enable_backward=False)
def _contact_pack_colored_headers_kernel(
    source: ContactColumnContainer,
    destination: ContactColumnContainer,
    element_ids_by_color: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    contact_offset: wp.int32,
):
    slot = wp.tid()
    if slot >= num_active_constraints[0]:
        return
    cid = element_ids_by_color[slot]
    if cid < contact_offset:
        return
    local_cid = cid - contact_offset
    destination.data[_OFF_BODY1, slot] = source.data[_OFF_BODY1, local_cid]
    destination.data[_OFF_BODY2, slot] = source.data[_OFF_BODY2, local_cid]
    destination.data[_OFF_FRICTION, slot] = source.data[_OFF_FRICTION, local_cid]
    destination.data[_OFF_FRICTION_DYNAMIC, slot] = source.data[_OFF_FRICTION_DYNAMIC, local_cid]
    destination.data[_OFF_CONTACT_FIRST, slot] = source.data[_OFF_CONTACT_FIRST, local_cid]
    destination.data[_OFF_CONTACT_COUNT, slot] = source.data[_OFF_CONTACT_COUNT, local_cid]
    destination.data[_OFF_SLOT1, slot] = source.data[_OFF_SLOT1, local_cid]
    destination.data[_OFF_SLOT2, slot] = source.data[_OFF_SLOT2, local_cid]
    destination.data[_OFF_COUNT1, slot] = source.data[_OFF_COUNT1, local_cid]
    destination.data[_OFF_COUNT2, slot] = source.data[_OFF_COUNT2, local_cid]
    destination.data[_OFF_ORIGINAL_CONTACT_FIRST, slot] = source.data[_OFF_CONTACT_FIRST, local_cid]


@wp.kernel(enable_backward=False)
def _contact_colored_row_counts_kernel(
    packed_headers: ContactColumnContainer,
    element_ids_by_color: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    contact_offset: wp.int32,
    counts: wp.array[wp.int32],
):
    slot = wp.tid()
    count = wp.int32(0)
    if slot < num_active_constraints[0] and element_ids_by_color[slot] >= contact_offset:
        count = contact_get_contact_count(packed_headers, slot)
    counts[slot] = count


@wp.kernel(enable_backward=False)
def _contact_build_colored_row_map_kernel(
    packed_headers: ContactColumnContainer,
    element_ids_by_color: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    contact_offset: wp.int32,
    offsets: wp.array[wp.int32],
    source_indices: wp.array[wp.int32],
    slots: wp.array[wp.int32],
):
    slot = wp.tid()
    if slot >= num_active_constraints[0] or element_ids_by_color[slot] < contact_offset:
        return
    source_first = reinterpret_float_as_int(packed_headers.data[_OFF_ORIGINAL_CONTACT_FIRST, slot])
    count = contact_get_contact_count(packed_headers, slot)
    destination_first = offsets[slot]
    for i in range(count):
        destination_k = destination_first + i
        source_indices[destination_k] = source_first + i
        slots[destination_k] = slot


@wp.kernel(enable_backward=False)
def _contact_colored_row_total_kernel(
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    total: wp.array[wp.int32],
):
    active = num_active_constraints[0]
    if active > wp.int32(0):
        last = active - wp.int32(1)
        total[0] = offsets[last] + counts[last]
    else:
        total[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _contact_gather_colored_rows_kernel(
    source: ContactContainer,
    destination: ContactContainer,
    packed_headers: ContactColumnContainer,
    offsets: wp.array[wp.int32],
    source_indices: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    total: wp.array[wp.int32],
):
    destination_k = wp.tid()
    if destination_k >= total[0]:
        return
    source_k = source_indices[destination_k]
    slot = slots[destination_k]
    if destination_k == offsets[slot]:
        packed_headers.data[_OFF_CONTACT_FIRST, slot] = reinterpret_int_as_float(destination_k)
    for row in range(CC_IMPULSE_DWORDS_PER_CONTACT):
        destination.impulses[row, destination_k] = source.impulses[row, source_k]
    for row in range(CC_DWORDS_PER_CONTACT):
        destination.lambdas[row, destination_k] = source.lambdas[row, source_k]
    for row in range(CC_DERIVED_DWORDS_PER_CONTACT):
        destination.derived[row, destination_k] = source.derived[row, source_k]


@wp.kernel(enable_backward=False)
def _contact_gather_colored_rigid_rows_kernel(
    source: ContactContainer,
    destination: ContactContainer,
    packed_headers: ContactColumnContainer,
    offsets: wp.array[wp.int32],
    source_indices: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    total: wp.array[wp.int32],
):
    """Gather only persistent rigid rows; prepare overwrites all other derived state."""
    destination_k = wp.tid()
    if destination_k >= total[0]:
        return
    source_k = source_indices[destination_k]
    slot = slots[destination_k]
    if destination_k == offsets[slot]:
        packed_headers.data[_OFF_CONTACT_FIRST, slot] = reinterpret_int_as_float(destination_k)
    for row in range(CC_IMPULSE_DWORDS_PER_CONTACT):
        destination.impulses[row, destination_k] = source.impulses[row, source_k]
    for row in range(CC_RIGID_DWORDS_PER_CONTACT):
        destination.lambdas[row, destination_k] = source.lambdas[row, source_k]
    cc_set_start_gap(destination, destination_k, cc_get_start_gap(source, source_k))


@wp.kernel(enable_backward=False)
def _contact_scatter_colored_rows_kernel(
    source: ContactContainer,
    destination: ContactContainer,
    packed_headers: ContactColumnContainer,
    offsets: wp.array[wp.int32],
    source_indices: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    total: wp.array[wp.int32],
):
    source_k = wp.tid()
    if source_k >= total[0]:
        return
    destination_k = source_indices[source_k]
    slot = slots[source_k]
    if source_k == offsets[slot]:
        original_first = reinterpret_float_as_int(packed_headers.data[_OFF_ORIGINAL_CONTACT_FIRST, slot])
        packed_headers.data[_OFF_CONTACT_FIRST, slot] = reinterpret_int_as_float(original_first)
    for row in range(CC_IMPULSE_DWORDS_PER_CONTACT):
        destination.impulses[row, destination_k] = source.impulses[row, source_k]
    for i in range(CC_LOCAL_ANCHOR_DWORDS):
        row = wp.int32(CC_LOCAL_ANCHOR_FIRST_ROW + i)
        destination.lambdas[row, destination_k] = source.lambdas[row, source_k]


def contact_pack_colored_headers(
    source: ContactColumnContainer,
    destination: ContactColumnContainer,
    element_ids_by_color: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    contact_offset: int,
    capacity: int,
    device: wp.DeviceLike = None,
) -> None:
    """Gather contact headers into the partitioner's color-slot order."""
    wp.launch(
        _contact_pack_colored_headers_kernel,
        dim=max(1, int(capacity)),
        inputs=[source, destination, element_ids_by_color, num_active_constraints, wp.int32(contact_offset)],
        device=device,
    )


def contact_build_colored_row_offsets(
    packed_headers: ContactColumnContainer,
    element_ids_by_color: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    contact_offset: int,
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    source_indices: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    total: wp.array[wp.int32],
    capacity: int,
    device: wp.DeviceLike = None,
) -> None:
    """Build general exclusive row offsets for color-ordered contact columns."""
    wp.launch(
        _contact_colored_row_counts_kernel,
        dim=max(1, int(capacity)),
        inputs=[
            packed_headers,
            element_ids_by_color,
            num_active_constraints,
            wp.int32(contact_offset),
            counts,
        ],
        device=device,
    )
    wp.utils.array_scan(counts, offsets, inclusive=False)
    wp.launch(
        _contact_build_colored_row_map_kernel,
        dim=max(1, int(capacity)),
        inputs=[
            packed_headers,
            element_ids_by_color,
            num_active_constraints,
            wp.int32(contact_offset),
            offsets,
            source_indices,
            slots,
        ],
        device=device,
    )
    wp.launch(
        _contact_colored_row_total_kernel,
        dim=1,
        inputs=[counts, offsets, num_active_constraints, total],
        device=device,
    )


def contact_gather_colored_rows(
    source: ContactContainer,
    destination: ContactContainer,
    packed_headers: ContactColumnContainer,
    offsets: wp.array[wp.int32],
    source_indices: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    total: wp.array[wp.int32],
    row_capacity: int,
    device: wp.DeviceLike = None,
    *,
    rigid_only: bool = False,
) -> None:
    """Gather iterate rows and switch packed headers to color-order indices."""
    wp.launch(
        _contact_gather_colored_rigid_rows_kernel if rigid_only else _contact_gather_colored_rows_kernel,
        dim=max(1, int(row_capacity)),
        inputs=[
            source,
            destination,
            packed_headers,
            offsets,
            source_indices,
            slots,
            total,
        ],
        device=device,
    )


def contact_scatter_colored_rows(
    source: ContactContainer,
    destination: ContactContainer,
    packed_headers: ContactColumnContainer,
    offsets: wp.array[wp.int32],
    source_indices: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    total: wp.array[wp.int32],
    row_capacity: int,
    device: wp.DeviceLike = None,
) -> None:
    """Scatter accumulated impulses and restore canonical contact indices."""
    wp.launch(
        _contact_scatter_colored_rows_kernel,
        dim=max(1, int(row_capacity)),
        inputs=[
            source,
            destination,
            packed_headers,
            offsets,
            source_indices,
            slots,
            total,
        ],
        device=device,
    )


@wp.func
def _col_write_float(c: ContactColumnContainer, off: wp.int32, local_cid: wp.int32, v: wp.float32):
    write2d_f32(c.data, off, local_cid, v)


@wp.func
def contact_accumulate_time_us(c: ContactColumnContainer, local_cid: wp.int32, t_us: wp.float32):
    """Atomic-add ``t_us`` microseconds into the column's ``time_us`` slot."""
    wp.atomic_add(c.data, CONTACT_TIME_US_OFFSET, local_cid, t_us)


# ---------------------------------------------------------------------------
# ContactViews -- bundle Newton Contacts arrays into a single wp.struct
# ---------------------------------------------------------------------------


@wp.struct
class ContactViews:
    """Kernel-visible view onto a Newton :class:`Contacts` buffer.

    Carries only the arrays consumed by the contact ingest / prepare /
    iterate / gather kernels. Soft-contact arrays (stiffness, damping,
    friction) may be zero-length sentinels for scenes without them.
    """

    rigid_contact_count: wp.array[wp.int32]

    #: Contact points in each shape's body frame.
    rigid_contact_point0: wp.array[wp.vec3f]
    rigid_contact_point1: wp.array[wp.vec3f]
    #: World-space normal, shape 0 -> shape 1, unit length.
    rigid_contact_normal: wp.array[wp.vec3f]

    rigid_contact_shape0: wp.array[wp.int32]
    rigid_contact_shape1: wp.array[wp.int32]

    #: Index of previous-frame matched contact, or MATCH_NOT_FOUND/MATCH_BROKEN.
    rigid_contact_match_index: wp.array[wp.int32]

    #: Surface thicknesses [m] folded into penetration depth.
    rigid_contact_margin0: wp.array[wp.float32]
    rigid_contact_margin1: wp.array[wp.float32]

    #: ``model.shape_body``.
    shape_body: wp.array[wp.int32]
    #: ``model.shape_type`` (or unified collision-pipeline shape types).
    shape_type: wp.array[wp.int32]

    #: Soft-contact stiffness [N/m] / damping [N·s/m]. ``0`` falls
    #: back to the legacy Box2D hertz-based normal row.
    rigid_contact_stiffness: wp.array[wp.float32]
    rigid_contact_damping: wp.array[wp.float32]
    #: Per-contact friction override. ``0`` falls back to materials.
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
    shape_type: wp.array,
    rigid_contact_stiffness: wp.array,
    rigid_contact_damping: wp.array,
    rigid_contact_friction: wp.array,
) -> ContactViews:
    """Build a :class:`ContactViews` from allocated Warp arrays.

    Soft-contact arrays are optional on Newton's Contacts; the caller
    must substitute a zero-length sentinel when unset.
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
    v.shape_type = shape_type
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


# Per-side endpoint metadata for cloth-aware contact iterate.


@wp.func
def contact_get_side0_kind(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_SIDE0_KIND, local_cid)


@wp.func
def contact_set_side0_kind(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_SIDE0_KIND, local_cid, v)


@wp.func
def contact_get_side1_kind(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_SIDE1_KIND, local_cid)


@wp.func
def contact_set_side1_kind(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_SIDE1_KIND, local_cid, v)


@wp.func
def contact_get_side0_nodes_extra(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3i:
    return wp.vec3i(
        _col_read_int(c, _OFF_SIDE0_NODES_EXTRA + wp.int32(0), local_cid),
        _col_read_int(c, _OFF_SIDE0_NODES_EXTRA + wp.int32(1), local_cid),
        _col_read_int(c, _OFF_SIDE0_NODES_EXTRA + wp.int32(2), local_cid),
    )


@wp.func
def contact_set_side0_nodes_extra(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3i):
    _col_write_int(c, _OFF_SIDE0_NODES_EXTRA + wp.int32(0), local_cid, v[0])
    _col_write_int(c, _OFF_SIDE0_NODES_EXTRA + wp.int32(1), local_cid, v[1])
    _col_write_int(c, _OFF_SIDE0_NODES_EXTRA + wp.int32(2), local_cid, v[2])


@wp.func
def contact_get_side1_nodes_extra(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3i:
    return wp.vec3i(
        _col_read_int(c, _OFF_SIDE1_NODES_EXTRA + wp.int32(0), local_cid),
        _col_read_int(c, _OFF_SIDE1_NODES_EXTRA + wp.int32(1), local_cid),
        _col_read_int(c, _OFF_SIDE1_NODES_EXTRA + wp.int32(2), local_cid),
    )


@wp.func
def contact_set_side1_nodes_extra(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3i):
    _col_write_int(c, _OFF_SIDE1_NODES_EXTRA + wp.int32(0), local_cid, v[0])
    _col_write_int(c, _OFF_SIDE1_NODES_EXTRA + wp.int32(1), local_cid, v[1])
    _col_write_int(c, _OFF_SIDE1_NODES_EXTRA + wp.int32(2), local_cid, v[2])


@wp.func
def contact_get_side0_slots_extra(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3i:
    return wp.vec3i(
        _col_read_int(c, _OFF_SIDE0_SLOTS_EXTRA + wp.int32(0), local_cid),
        _col_read_int(c, _OFF_SIDE0_SLOTS_EXTRA + wp.int32(1), local_cid),
        _col_read_int(c, _OFF_SIDE0_SLOTS_EXTRA + wp.int32(2), local_cid),
    )


@wp.func
def contact_set_side0_slots_extra(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3i):
    _col_write_int(c, _OFF_SIDE0_SLOTS_EXTRA + wp.int32(0), local_cid, v[0])
    _col_write_int(c, _OFF_SIDE0_SLOTS_EXTRA + wp.int32(1), local_cid, v[1])
    _col_write_int(c, _OFF_SIDE0_SLOTS_EXTRA + wp.int32(2), local_cid, v[2])


@wp.func
def contact_get_side1_slots_extra(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3i:
    return wp.vec3i(
        _col_read_int(c, _OFF_SIDE1_SLOTS_EXTRA + wp.int32(0), local_cid),
        _col_read_int(c, _OFF_SIDE1_SLOTS_EXTRA + wp.int32(1), local_cid),
        _col_read_int(c, _OFF_SIDE1_SLOTS_EXTRA + wp.int32(2), local_cid),
    )


@wp.func
def contact_set_side1_slots_extra(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3i):
    _col_write_int(c, _OFF_SIDE1_SLOTS_EXTRA + wp.int32(0), local_cid, v[0])
    _col_write_int(c, _OFF_SIDE1_SLOTS_EXTRA + wp.int32(1), local_cid, v[1])
    _col_write_int(c, _OFF_SIDE1_SLOTS_EXTRA + wp.int32(2), local_cid, v[2])


@wp.func
def contact_get_slot1(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_SLOT1, local_cid)


@wp.func
def contact_set_slot1(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_SLOT1, local_cid, v)


@wp.func
def contact_get_slot2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_SLOT2, local_cid)


@wp.func
def contact_set_slot2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_SLOT2, local_cid, v)


@wp.func
def contact_get_count1(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_COUNT1, local_cid)


@wp.func
def contact_set_count1(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_COUNT1, local_cid, v)


@wp.func
def contact_get_count2(c: ContactColumnContainer, local_cid: wp.int32) -> wp.int32:
    return _col_read_int(c, _OFF_COUNT2, local_cid)


@wp.func
def contact_set_count2(c: ContactColumnContainer, local_cid: wp.int32, v: wp.int32):
    _col_write_int(c, _OFF_COUNT2, local_cid, v)


@wp.func
def contact_get_side0_counts_extra(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3i:
    return wp.vec3i(
        _col_read_int(c, _OFF_SIDE0_COUNTS_EXTRA + wp.int32(0), local_cid),
        _col_read_int(c, _OFF_SIDE0_COUNTS_EXTRA + wp.int32(1), local_cid),
        _col_read_int(c, _OFF_SIDE0_COUNTS_EXTRA + wp.int32(2), local_cid),
    )


@wp.func
def contact_set_side0_counts_extra(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3i):
    _col_write_int(c, _OFF_SIDE0_COUNTS_EXTRA + wp.int32(0), local_cid, v[0])
    _col_write_int(c, _OFF_SIDE0_COUNTS_EXTRA + wp.int32(1), local_cid, v[1])
    _col_write_int(c, _OFF_SIDE0_COUNTS_EXTRA + wp.int32(2), local_cid, v[2])


@wp.func
def contact_get_side1_counts_extra(c: ContactColumnContainer, local_cid: wp.int32) -> wp.vec3i:
    return wp.vec3i(
        _col_read_int(c, _OFF_SIDE1_COUNTS_EXTRA + wp.int32(0), local_cid),
        _col_read_int(c, _OFF_SIDE1_COUNTS_EXTRA + wp.int32(1), local_cid),
        _col_read_int(c, _OFF_SIDE1_COUNTS_EXTRA + wp.int32(2), local_cid),
    )


@wp.func
def contact_set_side1_counts_extra(c: ContactColumnContainer, local_cid: wp.int32, v: wp.vec3i):
    _col_write_int(c, _OFF_SIDE1_COUNTS_EXTRA + wp.int32(0), local_cid, v[0])
    _col_write_int(c, _OFF_SIDE1_COUNTS_EXTRA + wp.int32(1), local_cid, v[1])
    _col_write_int(c, _OFF_SIDE1_COUNTS_EXTRA + wp.int32(2), local_cid, v[2])


# ---------------------------------------------------------------------------
# Prepare / iterate (rigid single-sweep) live in
# :mod:`constraint_contact_cloth`. One factory there generates both rigid
# and cloth-aware variants via ``wp.static(cloth_support)``. This file
# keeps only the multi-sweep ``*_at_multi`` rigid-only path used by the
# fast-tail kernels (different register-caching pattern).
# ---------------------------------------------------------------------------


def _make_contact_iterate_at_multi(has_soft_contact_pd: bool):
    @wp.func
    def impl(
        constraints: ContactColumnContainer,
        cid: wp.int32,
        base_offset: wp.int32,
        bodies: BodyContainer,
        particles: ParticleContainer,
        num_bodies: wp.int32,
        body_pair: ConstraintBodies,
        idt: wp.float32,
        cc: ContactContainer,
        contacts: ContactViews,
        use_bias: wp.bool,
        num_sweeps: wp.int32,
        copy_state: CopyStateContainer,
        parallel_id: wp.int32,
        sor_boost: wp.float32,
    ):
        """Run one contact column for ``num_sweeps`` local sweeps.

        Callers must preserve global color order when multiple colors are
        active; local multi-sweeps are only equivalent for single-color worlds.
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

        # ``contact_iterate_at_multi`` is only invoked from the multi-world
        # fast-tail kernel, where mass splitting is rejected at construction
        # time. Use direct SoA reads (no slot lookup) so NVRTC doesn't compile
        # the unreachable ``read_*_unified`` slow path into the kernel binary.
        v1, w1 = body_load_vw(bodies, b1)
        v2, w2 = body_load_vw(bodies, b2)
        inv_mass1 = bodies.inverse_mass[b1]
        inv_mass2 = bodies.inverse_mass[b2]
        inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b1))
        inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b2))

        # Body pose for per-contact lever-arm recompute.
        orientation1 = body_load_orientation(bodies, b1)
        orientation2 = body_load_orientation(bodies, b2)
        body_com1 = bodies.body_com[b1]
        body_com2 = bodies.body_com[b2]

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
                r1 = wp.quat_rotate(orientation1, local_p0 - body_com1) + margin0 * n
                r2 = wp.quat_rotate(orientation2, local_p1 - body_com2) - margin1 * n
                eff_n = cc_get_eff_n(cc, k)
                eff_t1 = cc_get_eff_t1(cc, k)
                eff_t2 = cc_get_eff_t2(cc, k)
                bias_val = cc_get_bias(cc, k)
                speculative_bias = bias_val
                bias_t1_val = cc_get_bias_t1(cc, k)
                bias_t2_val = cc_get_bias_t2(cc, k)
                is_speculative = speculative_bias > wp.float32(0.0)
                if is_speculative and not use_bias:
                    continue
                if not use_bias:
                    bias_val = wp.float32(0.0)
                    bias_t1_val = wp.float32(0.0)
                    bias_t2_val = wp.float32(0.0)

                pd_eff_soft_n = wp.float32(0.0)
                pd_gamma_n = wp.float32(0.0)
                pd_bias_n = wp.float32(0.0)
                if wp.static(has_soft_contact_pd):
                    pd_eff_soft_n = cc_get_pd_eff_soft(cc, k)
                    if pd_eff_soft_n > wp.float32(0.0):
                        pd_gamma_n = cc_get_pd_gamma(cc, k)
                        pd_bias_n = cc_get_pd_bias(cc, k)
                if use_bias:
                    mass_coeff_n = mass_coeff
                    impulse_coeff_n = impulse_coeff
                    mu_s_eff = mu_s
                    mu_k_eff = mu_k
                    if is_speculative:
                        mu_s_eff = wp.float32(0.0)
                        mu_k_eff = wp.float32(0.0)
                else:
                    mass_coeff_n = wp.float32(1.0)
                    impulse_coeff_n = wp.float32(0.0)
                    mu_s_eff = mu_s
                    mu_k_eff = mu_k
                if wp.static(has_soft_contact_pd):
                    update = contact_frame_velocity_update(
                        cc,
                        k,
                        n,
                        t1_dir,
                        t2_dir,
                        r1,
                        r2,
                        v1,
                        w1,
                        v2,
                        w2,
                        inv_mass1,
                        inv_mass2,
                        inv_inertia1,
                        inv_inertia2,
                        eff_n,
                        eff_t1,
                        eff_t2,
                        bias_val,
                        bias_t1_val,
                        bias_t2_val,
                        mu_s_eff,
                        mu_k_eff,
                        mass_coeff_n,
                        impulse_coeff_n,
                        sor_boost,
                        pd_eff_soft_n,
                        pd_gamma_n,
                        pd_bias_n,
                    )
                else:
                    update = contact_frame_velocity_update_no_soft_pd(
                        cc,
                        k,
                        n,
                        t1_dir,
                        t2_dir,
                        r1,
                        r2,
                        v1,
                        w1,
                        v2,
                        w2,
                        inv_mass1,
                        inv_mass2,
                        inv_inertia1,
                        inv_inertia2,
                        eff_n,
                        eff_t1,
                        eff_t2,
                        bias_val,
                        bias_t1_val,
                        bias_t2_val,
                        mu_s_eff,
                        mu_k_eff,
                        mass_coeff_n,
                        impulse_coeff_n,
                        sor_boost,
                        wp.float32(0.0),
                        wp.float32(0.0),
                        wp.float32(0.0),
                    )
                v1 = update.v_a
                w1 = update.w_a
                v2 = update.v_b
                w2 = update.w_b
            it += 1

        # Direct SoA writes -- fast-tail-only function (no mass splitting).
        body_store_vw(bodies, b1, v1, w1)
        body_store_vw(bodies, b2, v2, w2)

    return impl


contact_iterate_at_multi = _make_contact_iterate_at_multi(has_soft_contact_pd=True)
contact_iterate_at_multi_no_soft_pd = _make_contact_iterate_at_multi(has_soft_contact_pd=False)


@wp.func
def contact_iterate_multi(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    if constraints.articulation_owner[cid] >= wp.int32(0):
        return
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    # Sleep-transition safety (drop frozen-vs-frozen contacts) is now
    # the caller's responsibility, gated at the kernel level by
    # ``wp.static(has_sleeping)``. Rigid scenes without sleeping skip
    # the 4 scattered motion_type/island_root loads entirely.
    # Access-mode flip is the caller's responsibility now -- moved out of the
    # constraint hot path into the dispatcher under ``wp.static(cloth_support)``.
    # The fast-tail kernel that calls this entry is rigid-only by design,
    # so the flip is provably a no-op there.
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at_multi(
        constraints,
        cid,
        wp.int32(0),
        bodies,
        particles,
        num_bodies,
        body_pair,
        idt,
        cc,
        contacts,
        use_bias,
        num_sweeps,
        copy_state,
        parallel_id,
        sor_boost,
    )


@wp.func
def contact_iterate_multi_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    if constraints.articulation_owner[cid] >= wp.int32(0):
        return
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at_multi_no_soft_pd(
        constraints,
        cid,
        wp.int32(0),
        bodies,
        particles,
        num_bodies,
        body_pair,
        idt,
        cc,
        contacts,
        use_bias,
        num_sweeps,
        copy_state,
        parallel_id,
        sor_boost,
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
    """``r2 = quat_rotate(orient2, local_p1 - body_com2) - margin1 * n``."""
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
    """Column-summed world-frame wrench applied to body 2.

    Sums per-contact ``lam_n * n + lam_t1 * t1 + lam_t2 * t2``, divides
    by ``dt`` for force; torque is about body 2's COM via ``r2``.
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
    """World-frame wrench on body 2 for one contact ``k``."""
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
    """Scatter per-contact wrenches into ``out[k]``. Launch dim =
    ``max_contact_columns``; each thread walks its column's contact range."""
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
    """Per-column summary: total wrench + ``(body1, body2)`` + contact count."""
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
    """Position-level residual ``(gap, drift_t1, drift_t2)`` for contact ``k``.

    Same expression prepare uses for the biases, but without the
    ``friction_slop`` clamp.
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

    # local_p is in body-origin frame; subtract body_com so r is from COM.
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
    """Scatter per-contact residuals into ``out[k] = (gap, drift_t1, drift_t2)``."""
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
    """Zero placeholder; contact residuals are per-contact (see
    :func:`contact_per_contact_error_kernel`)."""
    return wp.spatial_vector(wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0))


@wp.func
def contact_world_error(
    constraints: ContactColumnContainer,
    cid: wp.int32,
) -> wp.spatial_vector:
    return contact_world_error_at(constraints, cid, 0)
