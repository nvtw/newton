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


def create_segmented_argmax_func(tile_size: int):
    """Create a segmented argmax function for a specific tile size.

    This uses Warp's tile operations for efficient cooperative argmax computation
    across segments within a thread block.

    Args:
        tile_size: Size of the tile (MUST match block size, e.g., 256)

    Returns:
        A Warp function that performs segmented argmax using tile operations
    """

    @wp.func
    def segmented_argmax(
        thread_id: int,
        segment_id: int,
        value: float,
    ) -> int:
        """Find the thread with maximum value within each segment cooperatively.

        All threads in the block must call this function together.

        Args:
            thread_id: Thread index within the block (0 to tile_size-1)
            segment_id: Segment ID for this thread
            value: Value for this thread

        Returns:
            Thread ID of the winner for this thread's segment
        """
        # Allocate temporary tile memory
        sort_buffer = wp.tile_zeros(shape=(tile_size,), dtype=wp.uint64)

        # Each thread writes its packed value to tile
        # Trick: reverse sort order of floats (negative) for easier max scan later
        packed = pack_slot_value(segment_id, -value)
        sort_buffer[thread_id] = packed

        # Create payload with thread indices
        sort_payload = wp.tile_arrange(tile_size, dtype=wp.int32)

        # Perform bitonic sort on the tiles
        wp.tile_sort(sort_buffer, sort_payload)

        # Detect segment boundaries (thread_id is now sorted position)
        slot_boundary = True
        if thread_id > 0:
            current_slot = get_slot_from_packed(sort_buffer[thread_id])
            prev_slot = get_slot_from_packed(sort_buffer[thread_id - 1])
            if current_slot == prev_slot:
                slot_boundary = False

        # Build scan input tile
        scan_input = wp.tile_zeros(shape=(tile_size,), dtype=wp.int32)
        write = 0
        if slot_boundary:
            write = thread_id
        scan_input[thread_id] = (
            write  # Keep this helper variable since writing into a tile emits a __syncthreads() under the hood
        )

        # Perform max scan to find the maximum index per segment
        max_scan_result = wp.tile_max_scan_inclusive(scan_input)

        # Extract the winner ID for this thread's position
        max_scan = wp.int32(max_scan_result[thread_id])
        winner_id = wp.int32(sort_payload[max_scan])

        # Reorder: prepare mapping from original thread_id to winner_id
        sort_buffer[sort_payload[thread_id]] = wp.uint64(winner_id)

        # Apply reordering to get final result for this thread
        result = wp.int32(sort_buffer[thread_id])

        synchronize()

        return result

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

        for i in range(6):
            # for i in range(1):  # For debugging
            scan_direction = get_scan_dir(slot, i)
            dot = inactive_dot_value
            if active:
                dot = wp.dot(scan_direction, c.position)

            winner = wp.static(create_segmented_argmax_func(tile_size))(thread_id, warp_slot, dot)
            if not active:
                winner = -1

            if active and winner == thread_id:
                key = slot * 6 + i

                p = buffer[key].projection
                if p == empty_marker:
                    id = wp.atomic_add(
                        active_ids, buffer_capacity, 1
                    )  # active_ids has one additional element to store the number of active elements
                    active_ids[id] = key

                if dot > p:
                    # Store contact data in buffer
                    c.projection = dot
                    buffer[key] = c

        synchronize()

    return StoreContactLinearArray


def create_shared_memory_pointer_func(
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


# Create the specific functions used in the codebase
get_shared_memory_pointer_121_ints = create_shared_memory_pointer_func(121)
get_shared_memory_pointer_120_contacts = create_shared_memory_pointer_func(120 * 9)


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
