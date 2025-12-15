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

from newton._src.core.hashtable import HashTable, hashtable_insert_with_index

from .contact_reduction import NUM_SPATIAL_DIRECTIONS, float_flip, get_scan_dir, get_slot

# Bit layout for hashtable key (64 bits total):
# - Bits 0-27:   shape_a (28 bits, up to ~268M shapes)
# - Bits 28-55:  shape_b (28 bits, up to ~268M shapes)
# - Bits 56-60:  icosahedron_bin (5 bits, 0-19)
# - Bits 61-63:  scan_direction (3 bits, 0-5)
# Total: 64 bits fully used

SHAPE_ID_BITS = wp.constant(wp.uint64(28))
SHAPE_ID_MASK = wp.constant(wp.uint64((1 << 28) - 1))
BIN_BITS = wp.constant(wp.uint64(5))
BIN_MASK = wp.constant(wp.uint64((1 << 5) - 1))
DIR_BITS = wp.constant(wp.uint64(3))
DIR_MASK = wp.constant(wp.uint64((1 << 3) - 1))


@wp.func
def make_contact_key(shape_a: int, shape_b: int, bin_id: int, scan_dir: int) -> wp.uint64:
    """Create a hashtable key from shape pair, normal bin, and scan direction.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        bin_id: Icosahedron bin index (0-19)
        scan_dir: Scan direction index (0-5)

    Returns:
        64-bit key for hashtable lookup
    """
    key = wp.uint64(shape_a) & SHAPE_ID_MASK
    key = key | ((wp.uint64(shape_b) & SHAPE_ID_MASK) << SHAPE_ID_BITS)
    key = key | ((wp.uint64(bin_id) & BIN_MASK) << wp.uint64(56))
    key = key | ((wp.uint64(scan_dir) & DIR_MASK) << wp.uint64(61))
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


class GlobalContactReducer:
    """Global contact reduction using hashtable-based tracking.

    This class manages:
    1. A global contact buffer storing contact data (struct of arrays)
    2. A hashtable tracking the best contact per (shape_pair, bin, direction)

    Contact data is packed into vec4 for efficient memory access:
    - position_depth: vec4(position.x, position.y, position.z, depth)
    - normal_feature: vec4(normal.x, normal.y, normal.z, float_bits(feature))

    Attributes:
        capacity: Maximum number of contacts that can be stored
        hashtable_capacity: Capacity of the hashtable (auto-sized)
        position_depth: vec4 array storing position.xyz and depth
        normal_feature: vec4 array storing normal.xyz and feature
        shape_pairs: vec2i array storing (shape_a, shape_b) per contact
        contact_count: Atomic counter for allocated contacts
        hashtable: HashTable for tracking best contacts
    """

    def __init__(self, capacity: int, device: str | None = None):
        """Initialize the global contact reducer.

        Args:
            capacity: Maximum number of contacts to store
            device: Warp device (e.g., "cuda:0", "cpu")
        """
        self.capacity = capacity
        self.device = device

        # Contact buffer (struct of arrays with vec4 packing)
        self.position_depth = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.normal_feature = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.shape_pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)

        # Atomic counter for contact allocation
        self.contact_count = wp.zeros(1, dtype=wp.int32, device=device)

        # Hashtable: sized for worst case of capacity contacts * 6 directions * some slack
        # Each contact registers up to 6 entries (one per scan direction)
        hashtable_size = capacity * NUM_SPATIAL_DIRECTIONS * 2  # 2x for load factor
        self.hashtable = HashTable(hashtable_size, device=device)

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
        return self.hashtable.get_active_count()

    def get_winning_contacts(self) -> list[int]:
        """Extract the winning contact IDs from the hashtable.

        Returns:
            List of unique contact IDs that won at least one hashtable slot
        """
        entries = self.hashtable.get_entries()
        contact_ids = set()
        for _key, value in entries:
            contact_id = value & 0xFFFFFFFF
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
        return data


@wp.func
def export_and_reduce_contact(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    feature: int,
    # Reducer arrays
    position_depth: wp.array(dtype=wp.vec4),
    normal_feature: wp.array(dtype=wp.vec4),
    shape_pairs: wp.array(dtype=wp.vec2i),
    contact_count: wp.array(dtype=wp.int32),
    ht_keys: wp.array(dtype=wp.uint64),
    ht_values: wp.array(dtype=wp.uint64),
    ht_active_slots: wp.array(dtype=wp.int32),
    capacity: int,
) -> int:
    """Store a contact and register it in the hashtable for reduction.

    This function:
    1. Allocates a slot in the contact buffer
    2. Stores the contact data (packed into vec4)
    3. Computes the icosahedron bin from the normal
    4. Registers the contact 6 times in the hashtable (once per scan direction)

    Each hashtable entry tracks the contact with the highest spatial projection
    score for that (shape_pair, bin, direction) combination.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating)
        feature: Feature identifier for deduplication
        position_depth: Contact buffer for position.xyz + depth
        normal_feature: Contact buffer for normal.xyz + feature
        shape_pairs: Contact buffer for shape pairs
        contact_count: Atomic counter for allocation
        ht_keys: Hashtable keys array
        ht_values: Hashtable values array
        ht_active_slots: Hashtable active slots array (size = ht_capacity + 1)
        capacity: Maximum contact capacity

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    # Allocate contact slot
    contact_id = wp.atomic_add(contact_count, 0, 1)
    if contact_id >= capacity:
        return -1

    # Store contact data (packed into vec4)
    position_depth[contact_id] = wp.vec4(position[0], position[1], position[2], depth)
    normal_feature[contact_id] = wp.vec4(
        normal[0], normal[1], normal[2], int_as_float(wp.int32(feature))
    )
    shape_pairs[contact_id] = wp.vec2i(shape_a, shape_b)

    # Get icosahedron bin from normal
    bin_id = get_slot(normal)

    # Register in hashtable for all 6 scan directions
    for dir_i in range(NUM_SPATIAL_DIRECTIONS):
        scan_dir = get_scan_dir(bin_id, dir_i)
        score = wp.dot(scan_dir, position)

        key = make_contact_key(shape_a, shape_b, bin_id, dir_i)
        value = make_contact_value(score, contact_id)
        hashtable_insert_with_index(key, value, ht_keys, ht_values, ht_active_slots)

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
):
    """Writer function that stores contacts in GlobalContactReducer for reduction.

    This follows the same signature as write_contact_simple in narrow_phase.py,
    so it can be used with create_compute_gjk_mpr_contacts and other contact
    generation functions.

    Args:
        contact_data: ContactData struct from contact computation
        reducer_data: GlobalContactReducerData struct with all reducer arrays
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
        position_depth=reducer_data.position_depth,
        normal_feature=reducer_data.normal_feature,
        shape_pairs=reducer_data.shape_pairs,
        contact_count=reducer_data.contact_count,
        ht_keys=reducer_data.ht_keys,
        ht_values=reducer_data.ht_values,
        ht_active_slots=reducer_data.ht_active_slots,
        capacity=reducer_data.capacity,
    )


def create_export_reduced_contacts_kernel(writer_func: Any):
    """Create a kernel that exports reduced contacts using a custom writer function.

    The kernel iterates over the active hashtable slots using grid stride loops,
    extracts the winning contact ID from each slot, and calls the writer function.

    Args:
        writer_func: A warp function with signature (ContactData, writer_data) -> None
                     This follows the same pattern as narrow_phase.py's write_contact_simple.

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
        # Writer data (custom struct)
        writer_data: Any,
        # Grid stride parameters
        total_num_threads: int,
    ):
        """Export reduced contacts to the writer.

        Uses grid stride loop to iterate over active hashtable slots.
        For each slot, extracts the winning contact and calls the writer function.
        """
        tid = wp.tid()

        # Get number of active slots (stored at index = ht_capacity)
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]

        # Grid stride loop over active slots
        for i in range(tid, num_active, total_num_threads):
            # Get the hashtable slot index
            slot_idx = ht_active_slots[i]

            # Get the value from this slot (contains score in high bits, contact_id in low bits)
            value = ht_values[slot_idx]

            # Extract contact ID from low 32 bits
            contact_id = unpack_contact_id(value)

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
            # radius_eff is 0 for mesh-triangle contacts (triangles have no radius)
            # For sphere/capsule contacts, this would need to be stored in the buffer
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


def create_mesh_triangle_contacts_to_reducer_kernel():
    """Create a kernel that processes mesh-triangle contacts and stores them in GlobalContactReducer.

    This kernel processes triangle pairs (mesh-shape, convex-shape, triangle_index) and
    computes contacts using GJK/MPR, storing results in the GlobalContactReducer for
    subsequent reduction and export.

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

    # Create the contact computation function with our reducer writer
    compute_contacts = create_compute_gjk_mpr_contacts(write_contact_to_reducer)

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

            # Compute contacts and store in reducer
            compute_contacts(
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

