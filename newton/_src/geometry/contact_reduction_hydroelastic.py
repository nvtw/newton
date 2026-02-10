# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hydroelastic contact reduction using hashtable-based tracking.

This module provides hydroelastic-specific contact reduction functionality,
building on the core ``GlobalContactReducer`` from ``contact_reduction_global.py``.

**Sign Convention:**

Uses standard SDF sign convention: negative depth = penetrating, positive = separated.
This matches the global contact reducer and other contact systems.

**Hydroelastic Contact Features:**

- Aggregate stiffness calculation: ``c_stiffness = k_eff * |agg_force| / total_depth``
- Normal matching: rotates reduced normals to align with aggregate force direction
- Anchor contact: synthetic contact at center of pressure for moment balance
- Moment matching: friction scaling to match aggregate moment from unreduced contacts

**Usage:**

Use ``HydroelasticContactReduction`` for the high-level API, or call the individual
kernels for more control over the pipeline.

See Also:
    :class:`GlobalContactReducer` in ``contact_reduction_global.py`` for the
    core contact reduction system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from newton._src.geometry.hashtable import hashtable_find_or_insert

from .contact_data import ContactData
from .contact_reduction import (
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_voxel_index,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .contact_reduction_global import (
    BETA_THRESHOLD,
    VALUES_PER_KEY,
    GlobalContactReducer,
    GlobalContactReducerData,
    _threadfence,
    export_contact_to_buffer,
    is_contact_already_exported,
    make_contact_key,
    make_contact_value,
    reduction_update_slot,
    unpack_contact,
    unpack_contact_id,
)

# =============================================================================
# Constants for hydroelastic export
# =============================================================================

EPS_LARGE = 1e-8
EPS_SMALL = 1e-20


# =============================================================================
# Hydroelastic contact buffer function
# =============================================================================


@wp.func
def export_hydroelastic_contact_to_buffer(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    area: float,
    k_eff: float,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Store a hydroelastic contact in the buffer with area and stiffness.

    Extends :func:`export_contact_to_buffer` by storing additional hydroelastic
    data (area and effective stiffness).

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating, standard convention)
        area: Contact surface area
        k_eff: Effective stiffness coefficient k_a*k_b/(k_a+k_b)
        reducer_data: GlobalContactReducerData with all arrays

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    # Use base function to store common contact data
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, reducer_data)

    if contact_id >= 0:
        # Store hydroelastic-specific data
        reducer_data.contact_area[contact_id] = area
        reducer_data.contact_k_eff[contact_id] = k_eff

    return contact_id


# =============================================================================
# Inline (on-the-fly) hydroelastic registration
# =============================================================================


@wp.func
def _write_slot(
    slot_id: int,
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    area: float,
    k_eff: float,
    reducer_data: GlobalContactReducerData,
):
    """Write contact data to contact storage at the given slot (inline mode only)."""
    if reducer_data.position_depth.shape[0] == 0:
        return
    if slot_id >= reducer_data.position_depth.shape[0]:
        return
    reducer_data.position_depth[slot_id] = wp.vec4(position[0], position[1], position[2], depth)
    reducer_data.normal[slot_id] = normal
    reducer_data.shape_pairs[slot_id] = wp.vec2i(shape_a, shape_b)
    reducer_data.contact_area[slot_id] = area
    reducer_data.contact_k_eff[slot_id] = k_eff


@wp.func
def _reduction_update_slot_and_maybe_write_slot(
    entry_idx: int,
    slot_id: int,
    value: wp.uint64,
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    area: float,
    k_eff: float,
    reducer_data: GlobalContactReducerData,
):
    """Atomically update a slot value and write contact data under a per-slot spinlock.

    Uses a hybrid fast-path / locked-path strategy:

    1. **Fast-path filter** -- read the current value without locking.  If we
       cannot possibly win (current >= ours), bail out immediately.  The vast
       majority of threads exit here with zero lock contention.
    2. **Spinlock acquire** -- ``atomic_cas`` on the per-slot lock.
    3. **Guarded compare-and-write** -- under the lock, re-read the value.  If
       we still win, write both the new value *and* the contact data
       atomically from the observer's perspective.
    4. **Threadfence + release** -- ``__threadfence()`` ensures all stores are
       globally visible before the lock is released via ``atomic_exch``.

    This eliminates the TOCTOU race of the previous lock-free approach where
    the value and the contact data could be written by different threads.

    ``value`` must pack ``(score, slot_storage_id)`` with
    ``slot_storage_id = slot_id * ht_capacity + entry_idx``.
    """
    ht_capacity = reducer_data.ht_capacity
    value_idx = slot_id * ht_capacity + entry_idx

    # Fast path: skip if we cannot win (no lock needed)
    if reducer_data.ht_values[value_idx] >= value:
        return

    # Acquire per-slot spinlock
    while wp.atomic_cas(reducer_data.slot_locks, value_idx, 0, 1) != 0:
        pass

    # Under lock: compare and conditionally write value + contact data
    if reducer_data.ht_values[value_idx] < value:
        reducer_data.ht_values[value_idx] = value
        _write_slot(value_idx, shape_a, shape_b, position, normal, depth, area, k_eff, reducer_data)

    # Device-scope fence: ensure all stores are visible before release
    _threadfence()

    # Release lock
    wp.atomic_exch(reducer_data.slot_locks, value_idx, 0)


@wp.func
def register_hydroelastic_contact_inline(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    area: float,
    k_eff: float,
    reducer_data: GlobalContactReducerData,
    shape_local_aabb_lower: wp.array(dtype=wp.vec3),
    shape_local_aabb_upper: wp.array(dtype=wp.vec3),
    shape_voxel_resolution: wp.array(dtype=wp.vec3i),
):
    """Register one hydroelastic contact in the hashtable on the fly.

    Use when reducer is in inline mode (slot-backed storage in contact arrays). Performs
    hashtable find_or_insert, atomic max per slot, conditional slot write, and aggregate
    accumulation for penetrating contacts.
    """
    if reducer_data.position_depth.shape[0] == 0:
        return

    aabb_lower = shape_local_aabb_lower[shape_b]
    aabb_upper = shape_local_aabb_upper[shape_b]
    ht_capacity = reducer_data.ht_capacity

    # === Part 1: Normal-binned reduction ===
    bin_id = get_slot(normal)
    pos_2d = project_point_to_plane(bin_id, position)
    key = make_contact_key(shape_a, shape_b, bin_id)
    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if entry_idx >= 0:
        use_beta = depth < wp.static(BETA_THRESHOLD) * wp.length(aabb_upper - aabb_lower)
        for dir_i in range(NUM_SPATIAL_DIRECTIONS):
            if use_beta:
                dir_2d = get_spatial_direction_2d(dir_i)
                score = wp.dot(pos_2d, dir_2d)
                slot_storage_id = dir_i * ht_capacity + entry_idx
                value = make_contact_value(score, slot_storage_id)
                _reduction_update_slot_and_maybe_write_slot(
                    entry_idx,
                    dir_i,
                    value,
                    shape_a,
                    shape_b,
                    position,
                    normal,
                    depth,
                    area,
                    k_eff,
                    reducer_data,
                )
        max_depth_slot_id = NUM_SPATIAL_DIRECTIONS
        slot_storage_id = max_depth_slot_id * ht_capacity + entry_idx
        max_depth_value = make_contact_value(-depth, slot_storage_id)
        _reduction_update_slot_and_maybe_write_slot(
            entry_idx,
            max_depth_slot_id,
            max_depth_value,
            shape_a,
            shape_b,
            position,
            normal,
            depth,
            area,
            k_eff,
            reducer_data,
        )
        if depth < 0.0:
            force_weight = area * (-depth)
            wp.atomic_add(reducer_data.agg_force, entry_idx, force_weight * normal)
            wp.atomic_add(reducer_data.weighted_pos_sum, entry_idx, force_weight * position)
            wp.atomic_add(reducer_data.weight_sum, entry_idx, force_weight)

    # === Part 2: Voxel-based reduction ===
    voxel_res = shape_voxel_resolution[shape_b]
    voxel_idx = compute_voxel_index(position, aabb_lower, aabb_upper, voxel_res)
    voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))
    voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    voxel_group = voxel_idx // voxels_per_group
    voxel_local_slot = voxel_idx % voxels_per_group
    voxel_bin_id = NUM_NORMAL_BINS + voxel_group
    voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)
    voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if voxel_entry_idx >= 0:
        slot_storage_id = voxel_local_slot * ht_capacity + voxel_entry_idx
        voxel_value = make_contact_value(-depth, slot_storage_id)
        _reduction_update_slot_and_maybe_write_slot(
            voxel_entry_idx,
            voxel_local_slot,
            voxel_value,
            shape_a,
            shape_b,
            position,
            normal,
            depth,
            area,
            k_eff,
            reducer_data,
        )


# =============================================================================
# Hydroelastic reduction kernels
# =============================================================================


@wp.kernel(enable_backward=False)
def reduce_hydroelastic_contacts_kernel(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array(dtype=wp.transform),
    shape_local_aabb_lower: wp.array(dtype=wp.vec3),
    shape_local_aabb_upper: wp.array(dtype=wp.vec3),
    shape_voxel_resolution: wp.array(dtype=wp.vec3i),
    total_num_threads: int,
):
    """Register hydroelastic contacts in the hashtable for reduction.

    Uses the fixed BETA_THRESHOLD for spatial competition.
    Uses pre-computed shape local AABBs and voxel resolution for voxel indices
    (matching the global contact reducer pattern).

    Also accumulates aggregate force per (shape_pair, normal_bin) for stiffness calculation:
    agg_force = sum(area * |depth| * normal) for all penetrating contacts.

    Uses standard sign convention: negative depth = penetrating (same as global reducer).
    - Spatial competition: includes contacts with depth < BETA_THRESHOLD
    - Max-depth/voxel slots: uses -depth so atomic_max selects most penetrating (most negative)
    """
    tid = wp.tid()

    # Get total number of contacts written
    num_contacts = reducer_data.contact_count[0]

    # Early exit if no contacts
    if num_contacts == 0:
        return

    # Cap at capacity
    num_contacts = wp.min(num_contacts, reducer_data.capacity)

    # Grid stride loop over contacts
    for i in range(tid, num_contacts, total_num_threads):
        # Read contact data from buffer
        pd = reducer_data.position_depth[i]
        normal = reducer_data.normal[i]
        pair = reducer_data.shape_pairs[i]
        area = reducer_data.contact_area[i]

        position = wp.vec3(pd[0], pd[1], pd[2])
        depth = pd[3]
        shape_a = pair[0]  # First shape
        shape_b = pair[1]  # Second shape

        aabb_lower = shape_local_aabb_lower[shape_b]
        aabb_upper = shape_local_aabb_upper[shape_b]

        ht_capacity = reducer_data.ht_capacity

        # === Part 1: Normal-binned reduction (spatial extremes + max-depth per bin) ===
        bin_id = get_slot(normal)
        pos_2d = project_point_to_plane(bin_id, position)

        key = make_contact_key(shape_a, shape_b, bin_id)

        entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
        if entry_idx >= 0:
            # Standard convention: depth < 0 means penetrating
            # Include all penetrating contacts and near-surface contacts in spatial competition
            use_beta = depth < wp.static(BETA_THRESHOLD) * wp.length(aabb_upper - aabb_lower)
            for dir_i in range(NUM_SPATIAL_DIRECTIONS):
                if use_beta:
                    dir_2d = get_spatial_direction_2d(dir_i)
                    score = wp.dot(pos_2d, dir_2d)
                    value = make_contact_value(score, i)
                    slot_id = dir_i
                    reduction_update_slot(entry_idx, slot_id, value, reducer_data.ht_values, ht_capacity)

            # Max-depth slot (last slot = 6)
            # Use -depth so atomic_max selects most penetrating (most negative depth)
            max_depth_slot_id = NUM_SPATIAL_DIRECTIONS
            max_depth_value = make_contact_value(-depth, i)
            reduction_update_slot(entry_idx, max_depth_slot_id, max_depth_value, reducer_data.ht_values, ht_capacity)

            # Accumulate aggregates for penetrating contacts (depth < 0)
            # These are used for stiffness calculation and anchor contact
            # Use |depth| (negate since depth is negative) for weighting
            if depth < 0.0:
                force_weight = area * (-depth)  # -depth to get positive weight
                # agg_force = sum(area * |depth| * normal) - for stiffness calculation
                wp.atomic_add(reducer_data.agg_force, entry_idx, force_weight * normal)
                # weighted_pos_sum = sum(area * |depth| * position) - for anchor position
                wp.atomic_add(reducer_data.weighted_pos_sum, entry_idx, force_weight * position)
                # weight_sum = sum(area * |depth|) - for normalizing anchor position
                wp.atomic_add(reducer_data.weight_sum, entry_idx, force_weight)

        # === Part 2: Voxel-based reduction using shape_b's (SDF) local space ===
        # Hydroelastic contacts are in SDF local space (shape_b's frame)
        # Use shape_b's local AABB for voxel computation (contact surface lives here)
        voxel_res = shape_voxel_resolution[shape_b]
        voxel_idx = compute_voxel_index(position, aabb_lower, aabb_upper, voxel_res)
        voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

        # Group voxels by 7 to maximize slot utilization (matches values_per_key)
        voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)  # = 7
        voxel_group = voxel_idx // voxels_per_group
        voxel_local_slot = voxel_idx % voxels_per_group

        voxel_bin_id = NUM_NORMAL_BINS + voxel_group
        voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

        voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
        if voxel_entry_idx >= 0:
            # Use -depth so atomic_max selects most penetrating (most negative depth)
            voxel_value = make_contact_value(-depth, i)
            reduction_update_slot(voxel_entry_idx, voxel_local_slot, voxel_value, reducer_data.ht_values, ht_capacity)


# =============================================================================
# Hydroelastic export kernel factory
# =============================================================================


def create_export_hydroelastic_reduced_contacts_kernel(
    writer_func: Any,
    margin_contact_area: float,
    normal_matching: bool = True,
    anchor_contact: bool = False,
):
    """Create a kernel that exports reduced hydroelastic contacts using a custom writer function.

    Similar to create_export_reduced_contacts_kernel but computes contact stiffness
    using the aggregate stiffness formula matching the original hydroelastic binning system:
        c_stiffness = k_eff * |agg_force| / total_depth

    where:
    - agg_force = sum(area * depth * normal) for ALL contacts in the entry (accumulated in reduce kernel)
    - total_depth = sum(depth) for SELECTED contacts (computed during export)

    This ensures the total contact force matches the aggregate force from all original contacts.

    Args:
        writer_func: A warp function with signature (ContactData, writer_data, int) -> None
        margin_contact_area: Contact area to use for non-penetrating contacts at the margin
        normal_matching: If True, rotate contact normals so their weighted sum aligns with aggregate force
        anchor_contact: If True, add an anchor contact at the center of pressure for each entry
    Returns:
        A warp kernel that can be launched to export reduced hydroelastic contacts.
    """
    # Define vector types for tracking exported contact data
    exported_ids_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)
    exported_depths_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)

    @wp.kernel(enable_backward=False)
    def export_hydroelastic_reduced_contacts_kernel(
        # Hashtable arrays
        ht_keys: wp.array(dtype=wp.uint64),
        ht_values: wp.array(dtype=wp.uint64),
        ht_active_slots: wp.array(dtype=wp.int32),
        # Aggregate data per entry (from reduce kernel)
        agg_force: wp.array(dtype=wp.vec3),
        weighted_pos_sum: wp.array(dtype=wp.vec3),
        weight_sum: wp.array(dtype=wp.float32),
        # Contact storage:
        # - legacy mode: sequential contacts
        # - inline mode: slot-major contacts (slot_id = slot * ht_capacity + entry_idx)
        position_depth: wp.array(dtype=wp.vec4),
        normal: wp.array(dtype=wp.vec3),
        shape_pairs: wp.array(dtype=wp.vec2i),
        contact_area: wp.array(dtype=wp.float32),
        contact_k_eff: wp.array(dtype=wp.float32),
        # Shape data for margin
        shape_contact_margin: wp.array(dtype=float),
        shape_transform: wp.array(dtype=wp.transform),
        # Writer data (custom struct)
        writer_data: Any,
        # Grid stride parameters
        total_num_threads: int,
    ):
        """Export reduced hydroelastic contacts to the writer with aggregate stiffness.

        In inline mode, value low bits are slot_id into slot-major contact storage.
        In legacy mode, value low bits are contact_id into sequential contact storage.
        """
        tid = wp.tid()
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]

        if num_active == 0:
            return

        for i in range(tid, num_active, total_num_threads):
            entry_idx = ht_active_slots[i]

            exported_ids = exported_ids_vec()
            exported_depths = exported_depths_vec()
            num_exported = int(0)
            total_depth = float(0.0)
            max_pen_depth = float(0.0)
            k_eff_first = float(0.0)
            shape_a_first = int(0)
            shape_b_first = int(0)
            selected_normal_sum = wp.vec3(0.0, 0.0, 0.0)

            for slot in range(wp.static(VALUES_PER_KEY)):
                value = ht_values[slot * ht_capacity + entry_idx]
                if value == wp.uint64(0):
                    continue

                slot_or_contact_id = unpack_contact_id(value)
                if is_contact_already_exported(slot_or_contact_id, exported_ids, num_exported):
                    continue

                pd = position_depth[slot_or_contact_id]
                contact_normal = normal[slot_or_contact_id]
                depth = pd[3]

                exported_ids[num_exported] = slot_or_contact_id
                exported_depths[num_exported] = depth
                num_exported = num_exported + 1

                if depth < 0.0:
                    pen_magnitude = -depth
                    total_depth = total_depth + pen_magnitude
                    max_pen_depth = wp.max(max_pen_depth, pen_magnitude)
                    if wp.static(normal_matching):
                        selected_normal_sum = selected_normal_sum + pen_magnitude * contact_normal

                if k_eff_first == 0.0:
                    k_eff_first = contact_k_eff[slot_or_contact_id]
                    pair = shape_pairs[slot_or_contact_id]
                    shape_a_first = pair[0]
                    shape_b_first = pair[1]

            # Skip entries with no contacts
            if num_exported == 0:
                continue

            # === Compute stiffness and optional features based on entry type ===
            # Normal bin entries (bin_id 0-19): have aggregate force, use aggregate stiffness
            # Voxel bin entries (bin_id 20+): no aggregate force, use per-contact stiffness
            agg_force_vec = agg_force[entry_idx]
            agg_force_mag = wp.length(agg_force_vec)
            use_aggregate_stiffness = agg_force_mag > wp.static(EPS_LARGE)

            # Compute anchor position (center of pressure) for normal bin entries
            anchor_pos = wp.vec3(0.0, 0.0, 0.0)
            add_anchor = int(0)
            entry_weight_sum = weight_sum[entry_idx]
            if wp.static(anchor_contact) and use_aggregate_stiffness and max_pen_depth > 1e-6:
                if entry_weight_sum > wp.static(EPS_SMALL):
                    anchor_pos = weighted_pos_sum[entry_idx] / entry_weight_sum
                    add_anchor = 1

            # Compute total_depth including anchor contribution
            anchor_depth = max_pen_depth  # Anchor uses max penetration depth (positive magnitude)
            total_depth_with_anchor = total_depth + wp.float32(add_anchor) * anchor_depth

            # Compute shared stiffness for normal bin entries
            # c_stiffness = k_eff * |agg_force| / total_depth (matches original hydroelastic system)
            shared_stiffness = float(0.0)
            if use_aggregate_stiffness:
                if total_depth_with_anchor > 0.0:
                    shared_stiffness = k_eff_first * agg_force_mag / (total_depth_with_anchor + wp.static(EPS_LARGE))
                else:
                    # Fallback for non-penetrating contacts
                    shared_stiffness = wp.static(margin_contact_area) * k_eff_first

            # Compute normal matching rotation quaternion
            rotation_q = wp.quat_identity()
            if wp.static(normal_matching) and use_aggregate_stiffness:
                selected_mag = wp.length(selected_normal_sum)
                if selected_mag > wp.static(EPS_LARGE) and agg_force_mag > wp.static(EPS_LARGE):
                    selected_dir = selected_normal_sum / selected_mag
                    agg_dir = agg_force_vec / agg_force_mag

                    cross = wp.cross(selected_dir, agg_dir)
                    cross_mag = wp.length(cross)
                    dot_val = wp.dot(selected_dir, agg_dir)

                    if cross_mag > wp.static(EPS_LARGE):
                        # Normal case: compute rotation around cross product axis
                        axis = cross / cross_mag
                        angle = wp.acos(wp.clamp(dot_val, -1.0, 1.0))
                        rotation_q = wp.quat_from_axis_angle(axis, angle)
                    elif dot_val < 0.0:
                        # Vectors are anti-parallel: rotate 180 degrees around a perpendicular axis
                        perp = wp.vec3(1.0, 0.0, 0.0)
                        if wp.abs(wp.dot(selected_dir, perp)) > 0.9:
                            perp = wp.vec3(0.0, 1.0, 0.0)
                        axis = wp.normalize(wp.cross(selected_dir, perp))
                        rotation_q = wp.quat_from_axis_angle(axis, 3.14159265359)

            # Get transform and margin (same for all contacts in the entry)
            transform_b = shape_transform[shape_b_first]
            margin_a = shape_contact_margin[shape_a_first]
            margin_b = shape_contact_margin[shape_b_first]
            margin = margin_a + margin_b

            # Friction scaling defaults to 1.0 for all reduced contacts.
            unique_friction = wp.float32(1.0)
            anchor_friction = wp.float32(1.0)

            # === Second pass: export contacts ===
            for idx in range(num_exported):
                slot_or_contact_id = exported_ids[idx]
                depth = exported_depths[idx]

                position, contact_normal, _ = unpack_contact(slot_or_contact_id, position_depth, normal)
                pair = shape_pairs[slot_or_contact_id]
                shape_a = pair[0]
                shape_b = pair[1]

                final_normal = contact_normal
                if wp.static(normal_matching) and use_aggregate_stiffness and depth < 0.0:
                    final_normal = wp.normalize(wp.quat_rotate(rotation_q, contact_normal))

                if use_aggregate_stiffness:
                    c_stiffness = shared_stiffness
                else:
                    area = contact_area[slot_or_contact_id]
                    k_eff = contact_k_eff[slot_or_contact_id]
                    if depth < 0.0:
                        c_stiffness = area * k_eff
                    else:
                        c_stiffness = wp.static(margin_contact_area) * k_eff

                # Transform contact to world space
                normal_world = wp.transform_vector(transform_b, final_normal)
                pos_world = wp.transform_point(transform_b, position)

                # Create ContactData struct
                # contact_distance = 2 * depth (depth is already negative for penetrating)
                # This gives negative contact_distance for penetrating contacts
                contact_data = ContactData()
                contact_data.contact_point_center = pos_world
                contact_data.contact_normal_a_to_b = normal_world
                contact_data.contact_distance = 2.0 * depth  # depth is negative = penetrating
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.thickness_a = 0.0
                contact_data.thickness_b = 0.0
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.margin = margin
                contact_data.contact_stiffness = c_stiffness
                # Apply friction scaling for penetrating contacts.
                contact_data.contact_friction_scale = unique_friction * wp.float32(depth < 0.0)

                # Call the writer function
                writer_func(contact_data, writer_data, -1)

            # === Export anchor contact if enabled ===
            if add_anchor == 1:
                # Anchor normal is aligned with aggregate force direction
                anchor_normal = wp.normalize(agg_force_vec)
                anchor_normal_world = wp.transform_vector(transform_b, anchor_normal)
                anchor_pos_world = wp.transform_point(transform_b, anchor_pos)

                # Create ContactData for anchor
                # anchor_depth is positive magnitude, so negate for standard convention
                contact_data = ContactData()
                contact_data.contact_point_center = anchor_pos_world
                contact_data.contact_normal_a_to_b = anchor_normal_world
                contact_data.contact_distance = -2.0 * anchor_depth  # anchor_depth is positive magnitude
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.thickness_a = 0.0
                contact_data.thickness_b = 0.0
                contact_data.shape_a = shape_a_first
                contact_data.shape_b = shape_b_first
                contact_data.margin = margin
                contact_data.contact_stiffness = shared_stiffness
                # Apply friction scaling for anchor.
                contact_data.contact_friction_scale = anchor_friction

                # Call the writer function for anchor
                writer_func(contact_data, writer_data, -1)

    return export_hydroelastic_reduced_contacts_kernel


# =============================================================================
# Hydroelastic Contact Reduction API
# =============================================================================


@dataclass
class HydroelasticReductionConfig:
    """Configuration for hydroelastic contact reduction.

    Attributes:
        normal_matching: If True, rotate reduced contact normals so their weighted
            sum aligns with the aggregate force direction.
        anchor_contact: If True, add an anchor contact at the center of pressure
            for each normal bin. The anchor contact helps preserve moment balance.
        margin_contact_area: Contact area used for non-penetrating contacts at the margin.
    """

    normal_matching: bool = True
    anchor_contact: bool = False
    margin_contact_area: float = 1e-2


class HydroelasticContactReduction:
    """High-level API for hydroelastic contact reduction.

    This class encapsulates the hydroelastic contact reduction pipeline, providing
    a clean interface that hides the low-level kernel launch details. It manages:

    1. A ``GlobalContactReducer`` for contact storage and hashtable tracking
    2. The reduction kernels for hashtable registration
    3. The export kernel for writing reduced contacts

    **Usage Pattern:**

    The typical usage in a contact generation pipeline is:

    1. Call ``clear()`` at the start of each frame
    2. Write contacts to the buffer using ``export_hydroelastic_contact_to_buffer``
       in your contact generation kernel (use ``get_data_struct()`` to get the data)
    3. Call ``reduce()`` to register contacts in the hashtable
    4. Call ``export()`` to write reduced contacts using the writer function

    Example:

        .. code-block:: python

            # Initialize once
            config = HydroelasticReductionConfig(normal_matching=True)
            reduction = HydroelasticContactReduction(
                capacity=100000,
                device="cuda:0",
                writer_func=my_writer_func,
                config=config,
            )

            # Each frame
            reduction.clear()

            # Launch your contact generation kernel that uses:
            # export_hydroelastic_contact_to_buffer(..., reduction.get_data_struct())

            reduction.reduce(shape_transform, shape_sdf_data, grid_size)
            reduction.export(shape_contact_margin, shape_transform, writer_data, grid_size)

    Attributes:
        reducer: The underlying ``GlobalContactReducer`` instance.
        config: The ``HydroelasticReductionConfig`` for this instance.
        contact_count: Array containing the number of contacts in the buffer.

    See Also:
        :func:`export_hydroelastic_contact_to_buffer`: Warp function for writing
            contacts to the buffer from custom kernels.
        :class:`GlobalContactReducerData`: Struct for passing reducer data to kernels.
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        writer_func: Any = None,
        config: HydroelasticReductionConfig | None = None,
        num_shape_pairs: int | None = None,
        use_inline_reduction: bool = False,
    ):
        """Initialize the hydroelastic contact reduction system.

        Args:
            capacity: Maximum number of contacts to store in the buffer (legacy mode only).
            device: Warp device (e.g., "cuda:0", "cpu"). If None, uses default device.
            writer_func: Warp function for writing decoded contacts. Must have signature
                ``(ContactData, writer_data, int) -> None``.
            config: Configuration options. If None, uses default ``HydroelasticReductionConfig``.
            num_shape_pairs: Number of hydroelastic shape pairs (for inline-mode hashtable sizing).
                Used when use_inline_reduction is True.
            use_inline_reduction: If True, use slot-backed inline storage
                (no large sequential contact buffer).
                Caller should set True only when using reduced contacts.
        """
        if config is None:
            config = HydroelasticReductionConfig()
        self.config = config
        self.device = device

        if use_inline_reduction:
            ht_size = max((num_shape_pairs or 0) * 35, 1024)  # 35 bins per pair (20 normal + 15 voxel)
            self.reducer = GlobalContactReducer(
                capacity=0,
                device=device,
                store_hydroelastic_data=True,
                use_inline_reduction=True,
                hashtable_size=ht_size,
            )
        else:
            self.reducer = GlobalContactReducer(
                capacity=capacity,
                device=device,
                store_hydroelastic_data=True,
                use_inline_reduction=False,
            )

        # Create the export kernel with the configured options
        self._export_kernel = create_export_hydroelastic_reduced_contacts_kernel(
            writer_func=writer_func,
            margin_contact_area=config.margin_contact_area,
            normal_matching=config.normal_matching,
            anchor_contact=config.anchor_contact,
        )

    @property
    def contact_count(self) -> wp.array:
        """Array containing the current number of contacts in the buffer."""
        return self.reducer.contact_count

    @property
    def capacity(self) -> int:
        """Maximum number of contacts that can be stored."""
        return self.reducer.capacity

    def get_data_struct(self) -> GlobalContactReducerData:
        """Get the data struct for passing to Warp kernels.

        Returns:
            A ``GlobalContactReducerData`` struct containing all arrays needed
            for contact storage and reduction.
        """
        return self.reducer.get_data_struct()

    def clear(self):
        """Clear all contacts and reset for a new frame.

        This efficiently clears only the active hashtable entries and resets
        the contact counter. Call this at the start of each simulation step.
        """
        self.reducer.clear_active()

    def reduce(
        self,
        shape_transform: wp.array,
        shape_local_aabb_lower: wp.array,
        shape_local_aabb_upper: wp.array,
        shape_voxel_resolution: wp.array,
        grid_size: int,
    ):
        """Register buffered contacts in the hashtable for reduction.

        This launches the reduction kernel that processes all contacts in the
        buffer and registers them in the hashtable based on spatial extremes,
        max-depth per normal bin, and voxel-based slots.

        Also accumulates aggregate force per (shape_pair, normal_bin) for
        stiffness calculation.

        Args:
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            shape_local_aabb_lower: Per-shape local AABB lower bounds (dtype: wp.vec3).
            shape_local_aabb_upper: Per-shape local AABB upper bounds (dtype: wp.vec3).
            shape_voxel_resolution: Per-shape voxel grid resolution (dtype: wp.vec3i).
            grid_size: Number of threads for the kernel launch.
        """
        reducer_data = self.reducer.get_data_struct()
        wp.launch(
            kernel=reduce_hydroelastic_contacts_kernel,
            dim=[grid_size],
            inputs=[
                reducer_data,
                shape_transform,
                shape_local_aabb_lower,
                shape_local_aabb_upper,
                shape_voxel_resolution,
                grid_size,
            ],
            device=self.device,
        )

    def export(
        self,
        shape_contact_margin: wp.array,
        shape_transform: wp.array,
        writer_data: Any,
        grid_size: int,
    ):
        """Export reduced contacts using the writer function.

        This exports the winning contacts from the hashtable, computing
        aggregate stiffness and applying optional normal matching.

        Args:
            shape_contact_margin: Per-shape contact margin (dtype: float).
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            writer_data: Data struct for the writer function.
            grid_size: Number of threads for the kernel launch.
        """
        wp.launch(
            kernel=self._export_kernel,
            dim=[grid_size],
            inputs=[
                self.reducer.hashtable.keys,
                self.reducer.ht_values,
                self.reducer.hashtable.active_slots,
                self.reducer.agg_force,
                self.reducer.weighted_pos_sum,
                self.reducer.weight_sum,
                self.reducer.position_depth,
                self.reducer.normal,
                self.reducer.shape_pairs,
                self.reducer.contact_area,
                self.reducer.contact_k_eff,
                shape_contact_margin,
                shape_transform,
                writer_data,
                grid_size,
            ],
            device=self.device,
        )

    def reduce_and_export(
        self,
        shape_transform: wp.array,
        shape_local_aabb_lower: wp.array,
        shape_local_aabb_upper: wp.array,
        shape_voxel_resolution: wp.array,
        shape_contact_margin: wp.array,
        writer_data: Any,
        grid_size: int,
    ):
        """Convenience method to reduce and export in one call.

        Combines ``reduce()`` and ``export()`` into a single method call.

        Args:
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            shape_local_aabb_lower: Per-shape local AABB lower bounds (dtype: wp.vec3).
            shape_local_aabb_upper: Per-shape local AABB upper bounds (dtype: wp.vec3).
            shape_voxel_resolution: Per-shape voxel grid resolution (dtype: wp.vec3i).
            shape_contact_margin: Per-shape contact margin (dtype: float).
            writer_data: Data struct for the writer function.
            grid_size: Number of threads for the kernel launch.
        """
        # In inline mode reduction was done in the generation kernel; skip separate reduce pass.
        if not self.reducer.use_inline_reduction:
            self.reduce(
                shape_transform,
                shape_local_aabb_lower,
                shape_local_aabb_upper,
                shape_voxel_resolution,
                grid_size,
            )
        self.export(shape_contact_margin, shape_transform, writer_data, grid_size)
