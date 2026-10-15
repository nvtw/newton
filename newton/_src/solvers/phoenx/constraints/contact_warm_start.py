# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply rigid contact warm-start impulses per body instead of per color.

Contact preparation only needs colors because the warm start writes body
velocities. Listing each column once per endpoint, sorted by body, lets one
launch gather every impulse a body receives in a fixed order, so geometry can
be prepared with one thread per row and no color sweep.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import (
    BodyContainer,
    body_load_inv_inertia_sym6,
    body_load_vw,
    body_store_vw,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_r0,
    cc_get_r1,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import warp_aggregated_increment
from newton._src.solvers.phoenx.helpers.math_helpers import apply_body_spatial_impulse
from newton._src.solvers.phoenx.helpers.scan_and_sort import active_prefix_sort_levels, radix_sort_active_prefix
from newton._src.solvers.phoenx.mass_splitting.kernels import subgroup_accumulate8_vec3

#: Sorts after every body id; its low bits are all ones for any key width.
_NO_BODY = wp.constant(wp.int32(2147483647))


@wp.kernel(enable_backward=False)
def _contact_body_entry_keys_kernel(
    columns: ContactColumnContainer,
    element_ids_by_color: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    contact_offset: wp.int32,
    articulation_owner: wp.array[wp.int32],
    keys: wp.array[wp.int32],
    values: wp.array[wp.int32],
    entry_count: wp.array[wp.int32],
):
    """Emit ``(body, 2 * slot + side)`` for both endpoints of each live column."""
    slot = wp.tid()
    active = num_active_constraints[0]
    if slot == wp.int32(0):
        entry_count[0] = wp.int32(2) * active
    key0 = _NO_BODY
    key1 = _NO_BODY
    if slot < active:
        eid = element_ids_by_color[slot]
        # Articulation-owned columns apply their load through the owner.
        if eid >= contact_offset and articulation_owner[eid - contact_offset] < wp.int32(0):
            key0 = contact_get_body1(columns, slot)
            key1 = contact_get_body2(columns, slot)
    keys[2 * slot] = key0
    values[2 * slot] = 2 * slot
    keys[2 * slot + 1] = key1
    values[2 * slot + 1] = 2 * slot + 1


@wp.kernel(enable_backward=False)
def _contact_body_entry_counts_kernel(
    keys: wp.array[wp.int32],
    entry_count: wp.array[wp.int32],
    counts: wp.array[wp.int32],
):
    entry = wp.tid()
    if entry >= entry_count[0]:
        return
    body = keys[entry]
    if body != _NO_BODY:
        warp_aggregated_increment(counts, body)


@wp.func
def store_row_warm_start(
    cc: ContactContainer,
    k: wp.int32,
    linear: wp.array[wp.vec3f],
    angular0: wp.array[wp.vec3f],
    angular1: wp.array[wp.vec3f],
):
    """Store prepared row ``k``'s cached impulse and its torques on both bodies.

    The row-parallel prepare writes these contiguously, so the per-body
    gather reads two vectors per row instead of the scattered contact frame.
    """
    n = cc_get_normal(cc, k)
    t1 = cc_get_tangent1(cc, k)
    imp = (
        cc_get_normal_lambda(cc, k) * n
        + cc_get_tangent1_lambda(cc, k) * t1
        + cc_get_tangent2_lambda(cc, k) * wp.cross(n, t1)
    )
    linear[k] = imp
    angular0[k] = wp.cross(cc_get_r0(cc, k), imp)
    angular1[k] = wp.cross(cc_get_r1(cc, k), imp)


@wp.kernel(enable_backward=False)
def _gathered_contact_warm_start_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    body_entry_start: wp.array[wp.int32],
    entries: wp.array[wp.int32],
    row_linear: wp.array[wp.vec3f],
    row_angular0: wp.array[wp.vec3f],
    row_angular1: wp.array[wp.vec3f],
):
    """Apply every cached contact impulse acting on one body.

    Eight lanes share a body: lane ``j`` sums entries ``j, j + 8, ...`` and
    the lanes are then reduced in lane order. Entries are sorted by body and
    column slot, so the summation order is fixed. Signs follow the pair
    update: the column impulse acts from body 1 onto body 2.
    """
    tid = wp.tid()
    body = tid >> wp.int32(3)
    lane = tid & wp.int32(7)
    inverse_mass = bodies.inverse_mass[body]
    if inverse_mass <= wp.float32(0.0):
        return
    linear = wp.vec3f(0.0)
    angular = wp.vec3f(0.0)
    for entry in range(body_entry_start[body] + lane, body_entry_start[body + 1], 8):
        code = entries[entry]
        slot = code >> wp.int32(1)
        first = contact_get_contact_first(columns, slot)
        end = first + contact_get_contact_count(columns, slot)
        if (code & wp.int32(1)) == wp.int32(0):
            for k in range(first, end):
                linear -= row_linear[k]
                angular -= row_angular0[k]
        else:
            for k in range(first, end):
                linear += row_linear[k]
                angular += row_angular1[k]
    mask = wp.uint32(255) << wp.uint32(tid & wp.int32(24))
    linear = subgroup_accumulate8_vec3(wp.vec3f(0.0), linear, mask)
    angular = subgroup_accumulate8_vec3(wp.vec3f(0.0), angular, mask)
    if lane != wp.int32(0):
        return
    v, w = body_load_vw(bodies, body)
    v, w = apply_body_spatial_impulse(
        v, w, inverse_mass, mat33_from_sym6(body_load_inv_inertia_sym6(bodies, body)), linear, angular
    )
    body_store_vw(bodies, body, v, w)


class ContactBodyEntries:
    """Per-frame list of live contact columns sorted by endpoint body."""

    def __init__(self, constraint_capacity: int, row_capacity: int, num_nodes: int, device: wp.DeviceLike) -> None:
        entry_capacity = 2 * max(1, int(constraint_capacity))
        self.num_nodes = int(num_nodes)
        self.node_bits = max(1, self.num_nodes.bit_length())
        self.keys = wp.zeros(2 * entry_capacity, dtype=wp.int32, device=device)
        self.values = wp.zeros(2 * entry_capacity, dtype=wp.int32, device=device)
        self.entry_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.prefix_flags = wp.zeros(active_prefix_sort_levels(entry_capacity), dtype=wp.int32, device=device)
        self.counts = wp.zeros(self.num_nodes + 1, dtype=wp.int32, device=device)
        self.starts = wp.zeros(self.num_nodes + 1, dtype=wp.int32, device=device)
        # Per-row warm-start impulse and torques, written by the row prepare.
        row_capacity = max(1, int(row_capacity))
        self.row_linear = wp.zeros(row_capacity, dtype=wp.vec3f, device=device)
        self.row_angular0 = wp.zeros(row_capacity, dtype=wp.vec3f, device=device)
        self.row_angular1 = wp.zeros(row_capacity, dtype=wp.vec3f, device=device)
        self.device = device

    def build(
        self,
        columns: ContactColumnContainer,
        element_ids_by_color: wp.array[wp.int32],
        num_active_constraints: wp.array[wp.int32],
        contact_offset: int,
        articulation_owner: wp.array[wp.int32],
    ) -> None:
        """Sort live column endpoints by body once per step."""
        wp.launch(
            _contact_body_entry_keys_kernel,
            dim=self.keys.shape[0] // 4,
            inputs=[
                columns,
                element_ids_by_color,
                num_active_constraints,
                wp.int32(contact_offset),
                articulation_owner,
                self.keys,
                self.values,
                self.entry_count,
            ],
            device=self.device,
        )
        radix_sort_active_prefix(self.keys, self.values, self.entry_count, self.prefix_flags, end_bit=self.node_bits)
        self.counts.zero_()
        wp.launch(
            _contact_body_entry_counts_kernel,
            dim=self.keys.shape[0] // 2,
            inputs=[self.keys, self.entry_count, self.counts],
            device=self.device,
        )
        wp.utils.array_scan(self.counts, self.starts, inclusive=False)

    def apply_warm_start(self, columns: ContactColumnContainer, bodies: BodyContainer) -> None:
        """Add the stored row impulses of every live column to body velocities."""
        wp.launch(
            _gathered_contact_warm_start_kernel,
            dim=8 * self.num_nodes,
            inputs=[columns, bodies, self.starts, self.values, self.row_linear, self.row_angular0, self.row_angular1],
            device=self.device,
        )


__all__ = ["ContactBodyEntries", "store_row_warm_start"]
