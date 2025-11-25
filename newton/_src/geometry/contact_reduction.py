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


# http://stereopsis.com/radix.html
@wp.func_native("""
uint32_t mask = ((i >> 31) - 1) | 0x80000000;
uint32_t res = i ^ mask;
return reinterpret_cast<float&>(res);
""")
def ifloat_flip(i: wp.uint32) -> float: ...


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
def pack_slot_value(slot_id: int, value: float) -> wp.uint64:
    """Pack slot ID and float value into a single 64-bit unsigned integer."""
    return (wp.uint64(slot_id) << wp.uint64(32)) | wp.uint64(float_flip(value))


@wp.func
def get_slot_from_packed(packed: wp.uint64) -> int:
    """Extract slot ID from packed value."""
    return wp.int32(packed >> wp.uint64(32))


@wp.func
def get_value(packed: wp.uint64) -> float:
    """Extract float value from packed value."""
    return ifloat_flip(wp.uint32(packed))


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


def create_segmented_argmax_func_21_segments(tile_size: int):
    """Create a segmented argmax function using atomic operations.

    This is O(n) instead of O(n log² n) compared to the sorting-based approach.
    Uses atomic_max on packed (value, thread_id) for efficient winner selection.

    Args:
        tile_size: Size of the tile (for API compatibility, not used internally)

    Returns:
        A Warp function that performs segmented argmax using atomics
    """

    NUM_SEGMENTS = 21  # 0 for inactive, 1-20 for the 20 icosahedron face slots
    @wp.func
    def segmented_argmax(
        thread_id: int,
        segment_id: int,
        value: float,
    ) -> int:
        """Find the thread with maximum value within each segment using atomics.

        All threads in the block must call this function together.

        Args:
            thread_id: Thread index within the block
            segment_id: Segment ID for this thread (0 to NUM_SEGMENTS-1)
            value: Value for this thread

        Returns:
            Thread ID of the winner for this thread's segment
        """
        # Get shared memory for atomic operations
        # Use wp.static() to make NUM_SEGMENTS a compile-time constant
        shared_mem = wp.array(
            ptr=get_shared_memory_pointer_21_uint64(), shape=(wp.static(NUM_SEGMENTS),), dtype=wp.uint64
        )

        # Initialize shared memory to 0 (represents -infinity, any real value beats it)
        # Only threads < NUM_SEGMENTS participate in initialization
        if thread_id < wp.static(NUM_SEGMENTS):
            shared_mem[thread_id] = wp.uint64(0)

        synchronize()

        # Pack value and thread_id: higher value wins, then higher thread_id for ties
        packed = pack_value_thread_id(value, thread_id)

        # Atomically update the segment's slot - highest packed value wins
        # Use native CUDA atomicMax for uint64 support
        atomic_max_uint64(shared_mem.ptr, segment_id, packed)

        synchronize()

        # Read back the winner for this thread's segment
        winner_packed = shared_mem[segment_id]
        winner_id = unpack_thread_id(winner_packed)

        return winner_id

    return segmented_argmax


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


def create_contact_reduction_func(tile_size: int):
    @wp.func
    def StoreContactLinearArray(
        thread_id: int,
        active: bool,
        c: ContactStruct,  # Contact data
        buffer: wp.array(dtype=ContactStruct),  # Contact buffer
        active_ids: wp.array(dtype=int),
        buffer_capacity: int,
        empty_marker: float,
    ):
        """Store contact in linear array using segmented argmax for reduction.

        Args:
            thread_id: Thread index within the block
            active: Whether this thread has an active contact
            c: Contact data for this thread
            buffer: Output buffer for reduced contacts
            active_ids: Array to store active slot IDs
            empty_marker: Marker value for empty slots
            segmented_argmax: Segmented argmax function
            get_slot: Function to get slot ID from contact normal
            get_scan_dir: Function to get scan direction
        """
        slot = 0
        warp_slot = 0
        if active:
            slot = get_slot(c.normal)
            warp_slot = slot + 1

        inactive_dot_value = empty_marker  # Ensure inactive threads never win

        for i in range(7):
            synchronize()
            # for i in range(1):  # For debugging
            scan_direction = get_scan_dir(slot, i)
            dot = inactive_dot_value
            if active:
                if i == 6:
                    dot = -c.depth  # We want to maximize the minimum depth (therefore the minus sign)
                else:
                    dot = wp.dot(scan_direction, c.position)

            winner = wp.static(create_segmented_argmax_func_21_segments(tile_size))(thread_id, warp_slot, dot)
            if not active:
                winner = -1

            if active and winner == thread_id:
                key = slot * 7 + i

                p = buffer[key].projection
                if p == empty_marker:
                    id = wp.atomic_add(
                        active_ids, buffer_capacity, 1
                    )  # active_ids has one additional element to store the number of active elements
                    if id < buffer_capacity:
                        active_ids[id] = key

                if dot > p:
                    # Store contact data in buffer
                    c.projection = dot
                    buffer[key] = c

        synchronize()

    return StoreContactLinearArray


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
# Shared memory for 21 segments (0=inactive, 1-20=slots) - 21 uint64 = 42 int32
get_shared_memory_pointer_21_uint64 = create_shared_memory_pointer_func_8_byte_aligned(21)


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
