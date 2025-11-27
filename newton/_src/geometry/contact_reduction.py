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

import warp as wp


# http://stereopsis.com/radix.html
@wp.func_native("""
uint32_t i = reinterpret_cast<uint32_t&>(f);
uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000;
return i ^ mask;
""")
def float_flip(f: float) -> wp.uint32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def synchronize(): ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
atomicMax(reinterpret_cast<unsigned long long*>(ptr) + idx, static_cast<unsigned long long>(val));
#endif
""")
def atomic_max_uint64(ptr: wp.uint64, idx: int, val: wp.uint64): ...


@wp.func
def pack_value_thread_id(value: float, thread_id: int) -> wp.uint64:
    """Pack float value and thread_id into uint64 for atomic argmax.

    High 32 bits: float_flip(value) - makes floats comparable as unsigned ints
    Low 32 bits: thread_id - for deterministic tie-breaking

    atomicMax on this packed value will select:
    1. The thread with the highest value
    2. For equal values, the thread with the highest thread_id (deterministic)
    """
    return (wp.uint64(float_flip(value)) << wp.uint64(32)) | wp.uint64(thread_id)


@wp.func
def unpack_thread_id(packed: wp.uint64) -> int:
    """Extract thread_id from packed value."""
    return wp.int32(packed & wp.uint64(0xFFFFFFFF))


@wp.struct
class ContactStruct:
    position: wp.vec3
    normal: wp.vec3
    depth: wp.float32
    feature: wp.int32
    projection: wp.float32


_mat20x3 = wp.types.matrix(shape=(20, 3), dtype=wp.float32)

icosahedronFaceNormals = _mat20x3(
    0.49112338,
    0.79465455,
    0.35682216,
    -0.18759243,
    0.7946545,
    0.57735026,
    -0.6070619,
    0.7946545,
    0.0,
    -0.18759237,
    0.7946545,
    -0.57735026,
    0.4911234,
    0.79465455,
    -0.3568221,
    0.18759249,
    -0.7946544,
    0.57735026,
    -0.49112338,
    -0.7946545,
    0.35682213,
    -0.49112338,
    -0.79465455,
    -0.35682213,
    0.18759243,
    -0.7946544,
    -0.57735026,
    0.607062,
    -0.7946544,
    0.0,
    0.9822469,
    -0.18759257,
    0.0,
    0.7946544,
    0.18759239,
    -0.5773503,
    0.30353096,
    -0.18759252,
    0.93417233,
    0.7946544,
    0.18759243,
    0.5773503,
    -0.7946545,
    -0.18759249,
    0.5773503,
    -0.30353105,
    0.18759243,
    0.9341724,
    -0.7946544,
    -0.1875924,
    -0.5773503,
    -0.9822469,
    0.18759254,
    0.0,
    0.30353096,
    -0.1875925,
    -0.93417233,
    -0.30353084,
    0.18759246,
    -0.9341724,
)

_mat60x3 = wp.types.matrix(shape=(60, 3), dtype=wp.float32)

icosahedronTriangles = _mat60x3(
    0.0,
    1.0,
    0.0,
    0.27639318,
    0.4472136,
    0.85065085,
    0.8944273,
    0.44721365,
    0.0,
    0.0,
    1.0,
    0.0,
    -0.7236069,
    0.44721365,
    0.5257311,
    0.27639318,
    0.4472136,
    0.85065085,
    0.0,
    1.0,
    0.0,
    -0.72360677,
    0.4472136,
    -0.5257312,
    -0.7236069,
    0.44721365,
    0.5257311,
    0.0,
    1.0,
    0.0,
    0.27639332,
    0.4472136,
    -0.8506508,
    -0.72360677,
    0.4472136,
    -0.5257312,
    0.0,
    1.0,
    0.0,
    0.8944273,
    0.44721365,
    0.0,
    0.27639332,
    0.4472136,
    -0.8506508,
    0.0,
    -1.0,
    0.0,
    0.7236068,
    -0.4472136,
    0.5257311,
    -0.27639323,
    -0.4472136,
    0.8506508,
    0.0,
    -1.0,
    0.0,
    -0.27639323,
    -0.4472136,
    0.8506508,
    -0.8944273,
    -0.44721365,
    0.0,
    0.0,
    -1.0,
    0.0,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    0.0,
    -1.0,
    0.0,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.0,
    -1.0,
    0.0,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.7236068,
    -0.4472136,
    0.5257311,
    0.8944273,
    0.44721365,
    0.0,
    0.7236068,
    -0.4472136,
    0.5257311,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.8944273,
    0.44721365,
    0.0,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.27639332,
    0.4472136,
    -0.8506508,
    0.27639318,
    0.4472136,
    0.85065085,
    -0.27639323,
    -0.4472136,
    0.8506508,
    0.7236068,
    -0.4472136,
    0.5257311,
    0.27639318,
    0.4472136,
    0.85065085,
    0.7236068,
    -0.4472136,
    0.5257311,
    0.8944273,
    0.44721365,
    0.0,
    -0.7236069,
    0.44721365,
    0.5257311,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.27639323,
    -0.4472136,
    0.8506508,
    -0.7236069,
    0.44721365,
    0.5257311,
    -0.27639323,
    -0.4472136,
    0.8506508,
    0.27639318,
    0.4472136,
    0.85065085,
    -0.72360677,
    0.4472136,
    -0.5257312,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.72360677,
    0.4472136,
    -0.5257312,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.7236069,
    0.44721365,
    0.5257311,
    0.27639332,
    0.4472136,
    -0.8506508,
    0.72360677,
    -0.4472136,
    -0.52573115,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    0.27639332,
    0.4472136,
    -0.8506508,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    -0.72360677,
    0.4472136,
    -0.5257312,
)


@wp.func
def get_scan_dir(icosahedron_face_id: int, i: int) -> wp.vec3:
    """Get scan direction for contact reduction.

    Args:
        icosahedron_face_id: ID of the icosahedron face
        i: Index in the range 0...5

    Returns:
        Edge of the triangle (not normalized).
        Indices 3, 4, 5 return the edges with negated direction.
    """
    result = icosahedronTriangles[icosahedron_face_id + (i + 1) % 3] - icosahedronTriangles[icosahedron_face_id + i % 3]
    if i >= 3:
        result = -result
    return result


@wp.func
def get_slot(normal: wp.vec3) -> int:
    """Returns the index of the icosahedron face that best matches the normal.

    Optimized by checking only relevant faces based on vertical component.

    Args:
        normal: Normal vector to match

    Returns:
        Index of the best matching icosahedron face
    """
    # Check Y-axis dot product to determine region
    up_dot = normal[1]  # Y component

    start_idx = 0
    end_idx = 0

    # Determine which faces to check based on vertical orientation
    if up_dot > 0.44721365:
        start_idx = 0
        end_idx = 5
    elif up_dot < -0.44721365:
        start_idx = 5
        end_idx = 10
    else:
        start_idx = 10
        end_idx = 20

    best_slot = start_idx
    max_dot = wp.dot(normal, icosahedronFaceNormals[start_idx])

    for i in range(start_idx + 1, end_idx):
        d = wp.dot(normal, icosahedronFaceNormals[i])
        if d > max_dot:
            max_dot = d
            best_slot = i

    return best_slot


NUM_REDUCTION_SLOTS = 140  # 20 icosahedron faces x 7 directions


@wp.func
def store_reduced_contact(
    thread_id: int,
    active: bool,
    c: ContactStruct,  # Contact data
    buffer: wp.array(dtype=ContactStruct),  # Contact buffer
    active_ids: wp.array(dtype=int),
    buffer_capacity: int,
    empty_marker: float,
):
    """Store contact in linear array using direct atomic reduction.

    OPTIMIZED VERSION: Uses 140 atomic slots (20 slots x 7 directions) with
    minimal synchronization. Only 3 sync barriers instead of 22+.

    Args:
        thread_id: Thread index within the block
        active: Whether this thread has an active contact
        c: Contact data for this thread
        buffer: Output buffer for reduced contacts
        active_ids: Array to store active slot IDs
        empty_marker: Marker value for empty slots
    """
    # Get shared memory for atomic winner tracking (140 uint64 slots)
    winner_slots = wp.array(
        ptr=get_shared_memory_pointer_140_uint64(),
        shape=(NUM_REDUCTION_SLOTS,),
        dtype=wp.uint64,
    )

    # Initialize shared memory - all threads participate for speed
    for i in range(thread_id, NUM_REDUCTION_SLOTS, wp.block_dim()):
        winner_slots[i] = wp.uint64(0)

    synchronize()  # SYNC 1: Ensure initialization complete

    # Compute slot and base key for this contact
    slot = 0
    if active:
        slot = get_slot(c.normal)

    # Compute all 7 dot products and do atomic updates in ONE pass
    # No sync needed between atomic operations!
    if active:
        base_key = slot * 7

        # Direction 0-5: spatial extremes
        for i in range(6):
            scan_direction = get_scan_dir(slot, i)
            dot = wp.dot(scan_direction, c.position)
            packed = pack_value_thread_id(dot, thread_id)
            atomic_max_uint64(winner_slots.ptr, base_key + i, packed)

        # Direction 6: depth (minimize depth = maximize -depth)
        dot6 = -c.depth
        packed6 = pack_value_thread_id(dot6, thread_id)
        atomic_max_uint64(winner_slots.ptr, base_key + 6, packed6)

    synchronize()  # SYNC 2: Ensure all atomics complete

    # Check if this thread won any slots and write results
    if active:
        base_key = slot * 7

        for i in range(7):
            key = base_key + i
            winner_packed = winner_slots[key]
            winner_id = unpack_thread_id(winner_packed)

            if winner_id == thread_id:
                # This thread won this slot
                p = buffer[key].projection
                if p == empty_marker:
                    id = wp.atomic_add(active_ids, buffer_capacity, 1)
                    if id < buffer_capacity:
                        active_ids[id] = key

                # Compute the dot product again for storage
                if i == 6:
                    dot = -c.depth
                else:
                    scan_direction = get_scan_dir(slot, i)
                    dot = wp.dot(scan_direction, c.position)

                if dot > p:
                    c.projection = dot
                    buffer[key] = c

    synchronize()  # SYNC 3: Ensure writes complete before next batch


@wp.func
def filter_unique_contacts(
    thread_id: int,
    buffer: wp.array(dtype=ContactStruct),
    active_ids: wp.array(dtype=int),
    buffer_capacity: int,
    empty_marker: float,
):
    """Filter out duplicate contacts after reduction, keeping only unique ones.

    A contact is considered a duplicate if it has the same feature (triangle index).
    For duplicates, we deterministically keep the one with the LOWEST slot key.

    OPTIMIZATION: Each of the 20 normal bins has exactly 7 directions (keys).
    Thread t (where t < 20) handles normal bin t, checking all 7 keys at once.
    This is O(20 * 7) = O(140) work total, parallelized across 20 threads.

    This function modifies active_ids in-place:
    - Rebuilds active_ids with only unique contacts
    - Updates the count at active_ids[buffer_capacity]

    Must be called after store_reduced_contact and a synchronize().

    Args:
        thread_id: Thread index within the block
        buffer: Contact buffer (read-only in this function)
        active_ids: Array of active slot IDs, with count at index buffer_capacity
        buffer_capacity: Capacity of the buffer (140)
        empty_marker: Value used to mark empty slots (same as passed to store_reduced_contact)
    """
    # Use shared memory for keep_flags[0..139]: 1 if key should be kept, 0 if duplicate
    keep_flags = wp.array(
        ptr=get_shared_memory_pointer_140_keep_flags(),
        shape=(NUM_REDUCTION_SLOTS,),
        dtype=wp.int32,
    )

    # Initialize: mark all as not-kept (0), we'll set to 1 for unique contacts
    for i in range(thread_id, NUM_REDUCTION_SLOTS, wp.block_dim()):
        keep_flags[i] = 0

    synchronize()

    # Each thread 0-19 handles one normal bin (20 bins total)
    # Thread t checks keys [t*7, t*7+6] and deduplicates by feature
    if thread_id < 20:
        bin_id = thread_id
        base_key = bin_id * 7

        # For each direction in this bin, check if there's valid data
        for dir_i in range(7):
            key_i = base_key + dir_i
            proj_i = buffer[key_i].projection

            # Check if this key has valid contact data (not empty)
            if proj_i > empty_marker:
                feature_i = buffer[key_i].feature

                # Check if an earlier direction in this bin has same feature
                is_duplicate = int(0)
                for dir_j in range(dir_i):
                    key_j = base_key + dir_j
                    proj_j = buffer[key_j].projection

                    if proj_j > empty_marker:
                        feature_j = buffer[key_j].feature
                        if feature_i == feature_j:
                            is_duplicate = 1

                # Keep if not a duplicate (first occurrence of this feature in bin)
                if is_duplicate == 0:
                    keep_flags[key_i] = 1

    synchronize()

    # Rebuild active_ids with only unique contacts
    # Thread 0 does serial compaction for correctness and determinism
    if thread_id == 0:
        write_idx = int(0)
        for key in range(NUM_REDUCTION_SLOTS):
            if keep_flags[key] == 1:
                active_ids[write_idx] = key
                write_idx = write_idx + 1
        active_ids[buffer_capacity] = write_idx

    synchronize()


def create_shared_memory_pointer_func_4_byte_aligned(
    array_size: int,
):
    """Create a shared memory pointer function for a specific array size.

    Args:
        array_size: Number of int elements in the shared memory array.

    Returns:
        A Warp function that returns a pointer to shared memory
    """

    snippet = f"""
    constexpr int array_size = {array_size};
    __shared__ int s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


def create_shared_memory_pointer_func_8_byte_aligned(
    array_size: int,
):
    """Create a shared memory pointer function for a specific array size.

    Args:
        array_size: Number of int elements in the shared memory array.

    Returns:
        A Warp function that returns a pointer to shared memory
    """

    snippet = f"""
    constexpr int array_size = {array_size};
    __shared__ uint64_t s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


# Create the specific functions used in the codebase
get_shared_memory_pointer_141_ints = create_shared_memory_pointer_func_4_byte_aligned(141)
get_shared_memory_pointer_140_contacts = create_shared_memory_pointer_func_4_byte_aligned(140 * 9)
# Shared memory for 140 reduction slots (20 faces x 7 directions) - 140 uint64
get_shared_memory_pointer_140_uint64 = create_shared_memory_pointer_func_8_byte_aligned(NUM_REDUCTION_SLOTS)
# Shared memory for keep flags used by filter_unique_contacts (140 ints)
get_shared_memory_pointer_140_keep_flags = create_shared_memory_pointer_func_4_byte_aligned(NUM_REDUCTION_SLOTS)


def create_shared_memory_pointer_block_dim_func(
    add: int,
):
    """Create a shared memory pointer function for a block-dimension-dependent array size.

    Args:
        add: Number of additional int elements beyond WP_TILE_BLOCK_DIM.

    Returns:
        A Warp function that returns a pointer to shared memory
    """

    snippet = f"""
    constexpr int array_size = WP_TILE_BLOCK_DIM +{add};
    __shared__ int s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


get_shared_memory_pointer_block_dim_plus_2_ints = create_shared_memory_pointer_block_dim_func(2)
