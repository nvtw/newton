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

"""Global GPU contact reduction using hashtable-based tracking.

This module provides a global contact reduction system that uses a hashtable
to track the best contacts across shape pairs, normal bins, and scan directions.
Unlike the shared-memory based approach in contact_reduction.py, this works
across the entire GPU without block-level synchronization constraints.

Key Design:
- Contacts are stored in a global buffer (struct of arrays, packed into vec4)
- A hashtable tracks the best contact per (shape_pair, normal_bin, scan_direction)
- Each contact is registered 6 times (once per scan direction)
- Atomic max selects the best contact based on spatial projection score
"""

from __future__ import annotations

from typing import Any

import warp as wp

from newton._src.geometry.hashtable_reduction import ReductionHashTable, hashtable_insert_slot

from .contact_reduction import NUM_SPATIAL_DIRECTIONS, float_flip, get_scan_dir, get_slot

# Bit layout for hashtable key (64 bits total):
# Key is (shape_a, shape_b, bin_id) - NO slot_id (slots are handled via values_per_key)
# - Bits 0-28:   shape_a (29 bits, up to ~537M shapes)
# - Bits 29-57:  shape_b (29 bits, up to ~537M shapes)
# - Bits 58-62:  icosahedron_bin (5 bits, 0-19)
# - Bit 63:      unused (could be used for flags)
# Total: 63 bits used

SHAPE_ID_BITS = wp.constant(wp.uint64(29))
SHAPE_ID_MASK = wp.constant(wp.uint64((1 << 29) - 1))
BIN_BITS = wp.constant(wp.uint64(5))
BIN_MASK = wp.constant(wp.uint64((1 << 5) - 1))


@wp.func
def make_contact_key(shape_a: int, shape_b: int, bin_id: int) -> wp.uint64:
    """Create a hashtable key from shape pair and normal bin.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        bin_id: Icosahedron bin index (0-19)

    Returns:
        64-bit key for hashtable lookup
    """
    key = wp.uint64(shape_a) & SHAPE_ID_MASK
    key = key | ((wp.uint64(shape_b) & SHAPE_ID_MASK) << SHAPE_ID_BITS)
    key = key | ((wp.uint64(bin_id) & BIN_MASK) << wp.uint64(58))
    return key


@wp.func
def make_contact_value(score: float, contact_id: int) -> wp.uint64:
    """Pack score and contact_id into hashtable value for atomic max.

    High 32 bits: float_flip(score) - makes floats comparable as unsigned ints
    Low 32 bits: contact_id - identifies which contact in the buffer

    Args:
        score: Spatial projection score (higher is better)
        contact_id: Index into the contact buffer

    Returns:
        64-bit value for hashtable (atomic max will select highest score)
    """
    return (wp.uint64(float_flip(score)) << wp.uint64(32)) | wp.uint64(contact_id)


@wp.func_native("""
return static_cast<int32_t>(packed & 0xFFFFFFFFull);
""")
def unpack_contact_id(packed: wp.uint64) -> int:
    """Extract contact_id from packed value."""
    ...


@wp.func_native("""
return reinterpret_cast<float&>(i);
""")
def int_as_float(i: wp.int32) -> float:
    """Reinterpret int32 bits as float32."""
    ...


@wp.func_native("""
return reinterpret_cast<int&>(f);
""")
def float_as_int(f: float) -> wp.int32:
    """Reinterpret float32 bits as int32."""
    ...


@wp.struct
class GlobalContactReducerData:
    """Struct for passing GlobalContactReducer arrays to kernels.

    This struct bundles all the arrays needed for global contact reduction
    so they can be passed as a single argument to warp kernels/functions.
    """

    # Contact buffer arrays
    position_depth: wp.array(dtype=wp.vec4)
    normal_feature: wp.array(dtype=wp.vec4)
    shape_pairs: wp.array(dtype=wp.vec2i)
    contact_count: wp.array(dtype=wp.int32)
    capacity: int

    # Hashtable arrays
    ht_keys: wp.array(dtype=wp.uint64)
    ht_values: wp.array(dtype=wp.uint64)
    ht_active_slots: wp.array(dtype=wp.int32)
    ht_values_per_key: int


class GlobalContactReducer:
    """Global contact reduction using hashtable-based tracking.

    This class manages:
    1. A global contact buffer storing contact data (struct of arrays)
    2. A hashtable tracking the best contact per (shape_pair, bin, slot)

    The hashtable key is (shape_a, shape_b, bin_id). Each key has multiple
    values (one per slot = direction × beta + deepest). This allows one thread
    to process all slots for a bin and deduplicate locally.

    Contact data is packed into vec4 for efficient memory access:
    - position_depth: vec4(position.x, position.y, position.z, depth)
    - normal_feature: vec4(normal.x, normal.y, normal.z, float_bits(feature))

    Attributes:
        capacity: Maximum number of contacts that can be stored
        values_per_key: Number of value slots per hashtable entry (13 for 2 betas)
        position_depth: vec4 array storing position.xyz and depth
        normal_feature: vec4 array storing normal.xyz and feature
        shape_pairs: vec2i array storing (shape_a, shape_b) per contact
        contact_count: Atomic counter for allocated contacts
        hashtable: ReductionHashTable for tracking best contacts
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        num_betas: int = 2,
    ):
        """Initialize the global contact reducer.

        Args:
            capacity: Maximum number of contacts to store
            device: Warp device (e.g., "cuda:0", "cpu")
            num_betas: Number of depth thresholds for contact reduction.
                       Total slots per bin = 6 directions * num_betas + 1 deepest.
                       Default 2 gives 13 slots per bin.
        """
        self.capacity = capacity
        self.device = device
        self.num_betas = num_betas

        # Values per key: 6 directions x num_betas + 1 deepest
        self.values_per_key = NUM_SPATIAL_DIRECTIONS * num_betas + 1

        # Contact buffer (struct of arrays with vec4 packing)
        self.position_depth = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.normal_feature = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.shape_pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)

        # Atomic counter for contact allocation
        self.contact_count = wp.zeros(1, dtype=wp.int32, device=device)

        # Hashtable: sized for worst case
        # Keys are (shape_pair, bin), so max keys = num_contacts x 20 bins
        # Use 2x for load factor
        hashtable_size = capacity * 20 * 2
        self.hashtable = ReductionHashTable(
            hashtable_size, values_per_key=self.values_per_key, device=device
        )

    def clear(self):
        """Clear all contacts and reset the reducer (full clear)."""
        self.contact_count.zero_()
        self.hashtable.clear()

    def clear_active(self):
        """Clear only the active entries (efficient for sparse usage)."""
        self.contact_count.zero_()
        self.hashtable.clear_active()

    def get_contact_count(self) -> int:
        """Get the current number of stored contacts."""
        return int(self.contact_count.numpy()[0])

    def get_active_slot_count(self) -> int:
        """Get the number of active hashtable slots."""
        # active_slots[capacity] stores the count
        return int(self.hashtable.active_slots.numpy()[self.hashtable.capacity])

    def get_winning_contacts(self) -> list[int]:
        """Extract the winning contact IDs from the hashtable.

        Returns:
            List of unique contact IDs that won at least one hashtable slot
        """
        values = self.hashtable.values.numpy()
        capacity = self.hashtable.capacity
        values_per_key = self.hashtable.values_per_key

        contact_ids = set()

        # Iterate over active slots
        active_slots_np = self.hashtable.active_slots.numpy()
        count = active_slots_np[capacity]

        for i in range(count):
            entry_idx = active_slots_np[i]
            base = entry_idx * values_per_key
            for slot in range(values_per_key):
                val = values[base + slot]
                if val != 0:
                    contact_id = val & 0xFFFFFFFF
                    contact_ids.add(int(contact_id))

        return sorted(contact_ids)

    def get_data_struct(self) -> GlobalContactReducerData:
        """Get a GlobalContactReducerData struct for passing to kernels.

        Returns:
            A GlobalContactReducerData struct containing all arrays.
        """
        data = GlobalContactReducerData()
        data.position_depth = self.position_depth
        data.normal_feature = self.normal_feature
        data.shape_pairs = self.shape_pairs
        data.contact_count = self.contact_count
        data.capacity = self.capacity
        data.ht_keys = self.hashtable.keys
        data.ht_values = self.hashtable.values
        data.ht_active_slots = self.hashtable.active_slots
        data.ht_values_per_key = self.values_per_key
        return data


@wp.func
def export_and_reduce_contact(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    feature: int,
    reducer_data: GlobalContactReducerData,
    beta0: float,
    beta1: float,
) -> int:
    """Store a contact and register it in the hashtable for reduction.

    This function:
    1. Allocates a slot in the contact buffer
    2. Stores the contact data (packed into vec4)
    3. Computes the icosahedron bin from the normal
    4. Registers the contact for each (direction, beta) combination where depth < beta
    5. Also registers for the max-depth slot (deepest contact per bin)

    The hashtable key is (shape_a, shape_b, bin_id). Each key has values_per_key
    value slots. The slot layout is:
    - Slots 0..5: direction 0 beta 0, direction 0 beta 1, ..., direction 2 beta 1
    - ...continues for all 6 directions × 2 betas...
    - Slot 12: max depth slot (deepest contact per bin)

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating)
        feature: Feature identifier for deduplication
        reducer_data: GlobalContactReducerData with all arrays
        beta0: First depth threshold (typically large, e.g., 1000000.0)
        beta1: Second depth threshold (typically small, e.g., 0.0001)

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    # Allocate contact slot
    contact_id = wp.atomic_add(reducer_data.contact_count, 0, 1)
    if contact_id >= reducer_data.capacity:
        return -1

    # Store contact data (packed into vec4)
    reducer_data.position_depth[contact_id] = wp.vec4(position[0], position[1], position[2], depth)
    reducer_data.normal_feature[contact_id] = wp.vec4(
        normal[0], normal[1], normal[2], int_as_float(wp.int32(feature))
    )
    reducer_data.shape_pairs[contact_id] = wp.vec2i(shape_a, shape_b)

    # Get icosahedron bin from normal
    bin_id = get_slot(normal)

    # Key is (shape_a, shape_b, bin_id) - NO slot in key
    key = make_contact_key(shape_a, shape_b, bin_id)

    # Register in hashtable for all 6 scan directions × 2 betas
    for dir_i in range(NUM_SPATIAL_DIRECTIONS):
        scan_dir = get_scan_dir(bin_id, dir_i)
        score = wp.dot(scan_dir, position)
        value = make_contact_value(score, contact_id)

        # Beta 0 slot
        if depth < beta0:
            slot_id = dir_i * 2
            hashtable_insert_slot(
                key, slot_id, value,
                reducer_data.ht_keys, reducer_data.ht_values,
                reducer_data.ht_active_slots, reducer_data.ht_values_per_key
            )

        # Beta 1 slot
        if depth < beta1:
            slot_id = dir_i * 2 + 1
            hashtable_insert_slot(
                key, slot_id, value,
                reducer_data.ht_keys, reducer_data.ht_values,
                reducer_data.ht_active_slots, reducer_data.ht_values_per_key
            )

    # Also register for max-depth slot (last slot)
    # Use -depth as score so atomic_max selects the deepest (most negative depth)
    max_depth_slot_id = NUM_SPATIAL_DIRECTIONS * 2  # = 12
    max_depth_value = make_contact_value(-depth, contact_id)
    hashtable_insert_slot(
        key, max_depth_slot_id, max_depth_value,
        reducer_data.ht_keys, reducer_data.ht_values,
        reducer_data.ht_active_slots, reducer_data.ht_values_per_key
    )

    return contact_id


@wp.func
def unpack_contact(
    contact_id: int,
    position_depth: wp.array(dtype=wp.vec4),
    normal_feature: wp.array(dtype=wp.vec4),
):
    """Unpack contact data from the buffer.

    Args:
        contact_id: Index into the contact buffer
        position_depth: Contact buffer for position.xyz + depth
        normal_feature: Contact buffer for normal.xyz + feature

    Returns:
        Tuple of (position, normal, depth, feature)
    """
    pd = position_depth[contact_id]
    nf = normal_feature[contact_id]

    position = wp.vec3(pd[0], pd[1], pd[2])
    depth = pd[3]
    normal = wp.vec3(nf[0], nf[1], nf[2])
    feature = float_as_int(nf[3])

    return position, normal, depth, feature


@wp.func
def write_contact_to_reducer(
    contact_data: Any,  # ContactData struct
    reducer_data: GlobalContactReducerData,
    beta0: float,
    beta1: float,
):
    """Writer function that stores contacts in GlobalContactReducer for reduction.

    This follows the same signature as write_contact_simple in narrow_phase.py,
    so it can be used with create_compute_gjk_mpr_contacts and other contact
    generation functions.

    Args:
        contact_data: ContactData struct from contact computation
        reducer_data: GlobalContactReducerData struct with all reducer arrays
        beta0: First depth threshold (typically large, e.g., 1000000.0)
        beta1: Second depth threshold (typically small, e.g., 0.0001)
    """
    # Extract contact info from ContactData
    position = contact_data.contact_point_center
    normal = contact_data.contact_normal_a_to_b
    depth = contact_data.contact_distance
    shape_a = contact_data.shape_a
    shape_b = contact_data.shape_b
    feature = int(contact_data.feature)

    # Store contact and register for reduction
    export_and_reduce_contact(
        shape_a=shape_a,
        shape_b=shape_b,
        position=position,
        normal=normal,
        depth=depth,
        feature=feature,
        reducer_data=reducer_data,
        beta0=beta0,
        beta1=beta1,
    )


def create_export_reduced_contacts_kernel(writer_func: Any, values_per_key: int = 13):
    """Create a kernel that exports reduced contacts using a custom writer function.

    The kernel processes one hashtable ENTRY per thread (not one value slot).
    Each entry has values_per_key value slots. The thread reads all slots,
    collects unique contact IDs, and exports each unique contact once.

    This naturally deduplicates: one thread handles one (shape_pair, bin) entry
    and can locally track which contact IDs it has already exported.

    Args:
        writer_func: A warp function with signature (ContactData, writer_data) -> None
                     This follows the same pattern as narrow_phase.py's write_contact_simple.
        values_per_key: Number of value slots per hashtable entry (default 13 for 2 betas)

    Returns:
        A warp kernel that can be launched to export reduced contacts.
    """
    # Import here to avoid circular imports
    from newton._src.geometry.contact_data import ContactData

    @wp.kernel(enable_backward=False)
    def export_reduced_contacts_kernel(
        # Hashtable arrays
        ht_keys: wp.array(dtype=wp.uint64),
        ht_values: wp.array(dtype=wp.uint64),
        ht_active_slots: wp.array(dtype=wp.int32),
        # Contact buffer arrays
        position_depth: wp.array(dtype=wp.vec4),
        normal_feature: wp.array(dtype=wp.vec4),
        shape_pairs: wp.array(dtype=wp.vec2i),
        # Shape data for extracting thickness
        shape_data: wp.array(dtype=wp.vec4),
        # Parameters
        margin: float,
        values_per_key_param: int,
        # Writer data (custom struct)
        writer_data: Any,
        # Grid stride parameters
        total_num_threads: int,
    ):
        """Export reduced contacts to the writer.

        Uses grid stride loop to iterate over active hashtable ENTRIES.
        For each entry, reads all value slots, collects unique contact IDs,
        and exports each unique contact once.
        """
        tid = wp.tid()

        # Get number of active entries (stored at index = ht_capacity)
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]

        # Grid stride loop over active entries
        for i in range(tid, num_active, total_num_threads):
            # Get the hashtable entry index
            entry_idx = ht_active_slots[i]

            # Track exported contact IDs for this entry (up to 13 slots)
            # Use a simple array to track seen IDs - most entries have few unique contacts
            # Declare as dynamic variables using int() for Warp loop mutation
            exported_ids_0 = int(-1)
            exported_ids_1 = int(-1)
            exported_ids_2 = int(-1)
            exported_ids_3 = int(-1)
            exported_ids_4 = int(-1)
            exported_ids_5 = int(-1)
            exported_ids_6 = int(-1)
            exported_ids_7 = int(-1)
            exported_ids_8 = int(-1)
            exported_ids_9 = int(-1)
            exported_ids_10 = int(-1)
            exported_ids_11 = int(-1)
            exported_ids_12 = int(-1)
            num_exported = int(0)

            # Read all value slots for this entry
            value_base = entry_idx * values_per_key_param
            for slot in range(values_per_key_param):
                value = ht_values[value_base + slot]

                # Skip empty slots (value = 0)
                if value == wp.uint64(0):
                    continue

                # Extract contact ID from low 32 bits
                contact_id = unpack_contact_id(value)

                # Check if we've already exported this contact ID
                already_exported = False
                if contact_id == exported_ids_0:
                    already_exported = True
                elif contact_id == exported_ids_1:
                    already_exported = True
                elif contact_id == exported_ids_2:
                    already_exported = True
                elif contact_id == exported_ids_3:
                    already_exported = True
                elif contact_id == exported_ids_4:
                    already_exported = True
                elif contact_id == exported_ids_5:
                    already_exported = True
                elif contact_id == exported_ids_6:
                    already_exported = True
                elif contact_id == exported_ids_7:
                    already_exported = True
                elif contact_id == exported_ids_8:
                    already_exported = True
                elif contact_id == exported_ids_9:
                    already_exported = True
                elif contact_id == exported_ids_10:
                    already_exported = True
                elif contact_id == exported_ids_11:
                    already_exported = True
                elif contact_id == exported_ids_12:
                    already_exported = True

                if already_exported:
                    continue

                # Record this contact ID as exported
                if num_exported == 0:
                    exported_ids_0 = contact_id
                elif num_exported == 1:
                    exported_ids_1 = contact_id
                elif num_exported == 2:
                    exported_ids_2 = contact_id
                elif num_exported == 3:
                    exported_ids_3 = contact_id
                elif num_exported == 4:
                    exported_ids_4 = contact_id
                elif num_exported == 5:
                    exported_ids_5 = contact_id
                elif num_exported == 6:
                    exported_ids_6 = contact_id
                elif num_exported == 7:
                    exported_ids_7 = contact_id
                elif num_exported == 8:
                    exported_ids_8 = contact_id
                elif num_exported == 9:
                    exported_ids_9 = contact_id
                elif num_exported == 10:
                    exported_ids_10 = contact_id
                elif num_exported == 11:
                    exported_ids_11 = contact_id
                elif num_exported == 12:
                    exported_ids_12 = contact_id
                num_exported = num_exported + 1

                # Unpack contact data
                position, normal, depth, feature = unpack_contact(
                    contact_id, position_depth, normal_feature
                )

                # Get shape pair
                pair = shape_pairs[contact_id]
                shape_a = pair[0]
                shape_b = pair[1]

                # Extract thickness from shape_data (stored in w component)
                thickness_a = shape_data[shape_a][3]
                thickness_b = shape_data[shape_b][3]

                # Create ContactData struct
                contact_data = ContactData()
                contact_data.contact_point_center = position
                contact_data.contact_normal_a_to_b = normal
                contact_data.contact_distance = depth
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.thickness_a = thickness_a
                contact_data.thickness_b = thickness_b
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.margin = margin
                contact_data.feature = wp.uint32(feature)
                contact_data.feature_pair_key = wp.uint64(0)

                # Call the writer function
                writer_func(contact_data, writer_data)

    return export_reduced_contacts_kernel


def create_mesh_triangle_contacts_to_reducer_kernel(beta0: float, beta1: float):
    """Create a kernel that processes mesh-triangle contacts and stores them in GlobalContactReducer.

    This kernel processes triangle pairs (mesh-shape, convex-shape, triangle_index) and
    computes contacts using GJK/MPR, storing results in the GlobalContactReducer for
    subsequent reduction and export.

    Args:
        beta0: First depth threshold for contact reduction
        beta1: Second depth threshold for contact reduction

    Returns:
        A warp kernel for processing mesh-triangle contacts with global reduction.
    """
    # Import here to avoid circular imports
    from newton._src.geometry.collision_core import (
        build_pair_key3,
        create_compute_gjk_mpr_contacts,
        get_triangle_shape_from_mesh,
    )
    from newton._src.geometry.narrow_phase import extract_shape_data

    # Create a writer function that captures beta0 and beta1
    @wp.func
    def write_to_reducer_with_betas(
        contact_data: Any,  # ContactData struct
        reducer_data: GlobalContactReducerData,
    ):
        write_contact_to_reducer(contact_data, reducer_data, beta0, beta1)

    @wp.kernel(enable_backward=False)
    def mesh_triangle_contacts_to_reducer_kernel(
        shape_types: wp.array(dtype=int),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        triangle_pairs: wp.array(dtype=wp.vec3i),
        triangle_pairs_count: wp.array(dtype=int),
        reducer_data: GlobalContactReducerData,
        total_num_threads: int,
    ):
        """Process triangle pairs and store contacts in GlobalContactReducer.

        Uses grid stride loop over triangle pairs.
        """
        tid = wp.tid()

        num_triangle_pairs = triangle_pairs_count[0]

        for i in range(tid, num_triangle_pairs, total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break

            triple = triangle_pairs[i]
            shape_a = triple[0]  # Mesh shape
            shape_b = triple[1]  # Convex shape
            tri_idx = triple[2]

            # Get mesh data for shape A
            mesh_id_a = shape_source[shape_a]
            if mesh_id_a == wp.uint64(0):
                continue

            scale_data_a = shape_data[shape_a]
            mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])

            # Get mesh world transform
            X_mesh_ws_a = shape_transform[shape_a]

            # Extract triangle shape data from mesh
            shape_data_a, v0_world = get_triangle_shape_from_mesh(
                mesh_id_a, mesh_scale_a, X_mesh_ws_a, tri_idx
            )

            # Extract shape B data
            pos_b, quat_b, shape_data_b, _scale_b, thickness_b = extract_shape_data(
                shape_b,
                shape_transform,
                shape_types,
                shape_data,
                shape_source,
            )

            # Set pos_a to be vertex A (origin of triangle in local frame)
            pos_a = v0_world
            quat_a = wp.quat_identity()  # Triangle has no orientation

            # Extract thickness for shape A
            thickness_a = shape_data[shape_a][3]

            # Use per-shape contact margin
            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = wp.max(margin_a, margin_b)

            # Build pair key including triangle index
            pair_key = build_pair_key3(
                wp.uint32(shape_a), wp.uint32(shape_b), wp.uint32(tri_idx)
            )

            # Compute and write contacts using GJK/MPR
            wp.static(create_compute_gjk_mpr_contacts(write_to_reducer_with_betas))(
                shape_data_a,
                shape_data_b,
                quat_a,
                quat_b,
                pos_a,
                pos_b,
                margin,
                shape_a,
                shape_b,
                thickness_a,
                thickness_b,
                reducer_data,
                pair_key,
            )

    return mesh_triangle_contacts_to_reducer_kernel

