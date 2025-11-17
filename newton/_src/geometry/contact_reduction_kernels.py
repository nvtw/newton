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

import numpy as np
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
def get_slot(packed: wp.uint64) -> int:
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
        sort_payload = wp.tile_arange(tile_size, dtype=wp.int32)

        # Perform bitonic sort on the tiles
        wp.tile_sort(sort_buffer, sort_payload)

        # Detect segment boundaries (thread_id is now sorted position)
        slot_boundary = True
        if thread_id > 0:
            current_slot = get_slot(sort_buffer[thread_id])
            prev_slot = get_slot(sort_buffer[thread_id - 1])
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


snippet_121 = """
        constexpr int array_size = 121;
        __shared__ int s[array_size];
        auto ptr = &s[0];
        return (uint64_t)ptr;
        """


@wp.func_native(snippet_121)
def get_shared_memory_pointer_121_ints() -> wp.uint64: ...


snippet_120_contact = """
        constexpr int array_size = 120;
        __shared__ float s[9*array_size]; //9 = sizeof(ContactStruct) / sizeof(float)
        auto ptr = &s[0];
        return (uint64_t)ptr;
        """


@wp.func_native(snippet_120_contact)
def get_shared_memory_pointer_120_contacts() -> wp.uint64: ...


@wp.func
def generate_arbitrary_contact_data(t: int) -> ContactStruct:
    c = ContactStruct()

    ft = float(t)

    # For testing, generate some arbitrary contact data
    c.position = wp.vec3(wp.sin(ft * 0.1) * ft * 0.01, 0.0, wp.cos(ft * 0.1) * ft * 0.01)
    c.normal = wp.vec3(0.0, 1.0, 0.0)
    c.depth = 0.1
    c.feature = 0
    c.projection = 0.0

    return c


@wp.kernel(enable_backward=False)
def contact_reduction_test_kernel(out_contacts: wp.array(dtype=ContactStruct), out_count: wp.array(dtype=int)):
    block_id, t = wp.tid()
    empty_marker = -1000000000.0

    active_contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_121_ints(), shape=(121,), dtype=wp.int32)
    contacts_shared_mem = wp.array(ptr=get_shared_memory_pointer_120_contacts(), shape=(120,), dtype=ContactStruct)

    for i in range(t, 120, wp.block_dim()):
        contacts_shared_mem[i].projection = empty_marker

    if t == 0:
        active_contacts_shared_mem[120] = 0

    synchronize()

    has_contact = t % 2 == 0
    c = generate_arbitrary_contact_data(t)

    wp.static(create_contact_reduction_func(128))(
        t,
        has_contact,
        c,
        contacts_shared_mem,
        active_contacts_shared_mem,
        120,
        empty_marker,
    )

    # Now write the reduced contacts to the output array
    num_contacts_to_keep = active_contacts_shared_mem[120]

    # Thread 0 writes the count
    if t == 0:
        out_count[0] = num_contacts_to_keep

    # All threads cooperatively write the contacts
    for i in range(t, num_contacts_to_keep, wp.block_dim()):
        contact_id = active_contacts_shared_mem[i]
        out_contacts[i] = contacts_shared_mem[contact_id]


def contact_reduction_test_host(num_threads=128):
    """Sequential reference implementation of contact reduction.

    Args:
        num_threads: Number of threads/contacts to process (default 128)

    Returns:
        tuple: (reduced_contacts_dict, active_keys) where:
            - reduced_contacts_dict: dict mapping (slot, direction) -> contact
            - active_keys: list of active (slot, direction) keys
    """
    # Generate the same contacts as the GPU kernel
    contacts = []
    for t in range(num_threads):
        has_contact = t % 2 == 0
        if has_contact:
            # Call the same function used in the GPU kernel
            c = generate_arbitrary_contact_data(t)

            contact = {
                "position": np.array([c.position[0], c.position[1], c.position[2]]),
                "normal": np.array([c.normal[0], c.normal[1], c.normal[2]]),
                "depth": c.depth,
                "feature": c.feature,
                "thread_id": t,
            }
            contacts.append(contact)

    # Dictionary to store the best contact for each (slot, direction) pair
    # Key: (slot, direction_idx), Value: (contact, projection_value)
    best_contacts = {}

    # Process each contact
    for contact in contacts:
        # Get the slot for this contact's normal - call @wp.func directly
        slot = get_slot(wp.vec3(contact["normal"]))

        # Try all 6 scan directions for this slot
        for direction_idx in range(6):
            # Call @wp.func directly
            scan_direction = get_scan_dir(slot, direction_idx)
            scan_direction_np = np.array([scan_direction[0], scan_direction[1], scan_direction[2]])

            # Compute dot product (projection)
            dot = np.dot(scan_direction_np, contact["position"])

            key = (slot, direction_idx)

            # Keep the contact with maximum projection for this (slot, direction)
            if key not in best_contacts or dot > best_contacts[key][1]:
                best_contacts[key] = (contact, dot)

    # Build output in the same format as GPU (sorted by key for consistency)
    active_keys = sorted(best_contacts.keys())
    reduced_contacts = {}

    for key in active_keys:
        contact, projection = best_contacts[key]
        # Store contact with projection value
        reduced_contacts[key] = {
            "position": contact["position"],
            "normal": contact["normal"],
            "depth": contact["depth"],
            "feature": contact["feature"],
            "projection": projection,
            "thread_id": contact["thread_id"],
        }

    return reduced_contacts, active_keys


def validate_contact_reduction(gpu_contacts, gpu_count):
    """Validate GPU contact reduction results against CPU reference.

    Verifies that the set of contacts is identical (order doesn't matter).

    Args:
        gpu_contacts: Warp array of contacts from GPU
        gpu_count: Warp array with count of contacts from GPU

    Returns:
        bool: True if validation passes, False otherwise
    """
    # Get CPU reference solution
    cpu_contacts, cpu_keys = contact_reduction_test_host(num_threads=128)

    # Read GPU results
    gpu_count_val = gpu_count.numpy()[0]
    gpu_contacts_np = gpu_contacts.numpy()

    print(f"\nValidation:")
    print(f"  CPU reference: {len(cpu_keys)} contacts")
    print(f"  GPU result: {gpu_count_val} contacts")

    # Check if counts match
    if gpu_count_val != len(cpu_keys):
        print(f"  FAILED: Contact count mismatch!")
        return False

    # Build sets of contacts for comparison (order-independent)
    # Use (position, normal, depth, projection) as the contact signature
    tolerance = 1e-4  # Relaxed tolerance for floating-point comparisons

    def contact_signature(contact):
        """Create a hashable signature for a contact (rounded for tolerance)."""
        pos = tuple(np.round(contact["position"] / tolerance).astype(int))
        normal = tuple(np.round(contact["normal"] / tolerance).astype(int))
        depth = int(np.round(contact["depth"] / tolerance))
        proj = int(np.round(contact["projection"] / tolerance))
        return (pos, normal, depth, proj)

    # Build CPU contact set
    cpu_contact_set = set()
    cpu_contact_list = []
    for cpu_key in cpu_keys:
        cpu_contact = cpu_contacts[cpu_key]
        sig = contact_signature(cpu_contact)
        cpu_contact_set.add(sig)
        cpu_contact_list.append((sig, cpu_contact, cpu_key))

    # Build GPU contact set
    gpu_contact_set = set()
    gpu_contact_list = []
    for i in range(gpu_count_val):
        gpu_contact = gpu_contacts_np[i]
        sig = contact_signature(gpu_contact)
        gpu_contact_set.add(sig)
        gpu_contact_list.append((sig, gpu_contact, i))

    # Check if sets are identical
    missing_in_gpu = cpu_contact_set - gpu_contact_set
    extra_in_gpu = gpu_contact_set - cpu_contact_set

    success = True

    if missing_in_gpu:
        print(f"  FAILED: {len(missing_in_gpu)} contacts in CPU reference but not in GPU:")
        for sig in list(missing_in_gpu)[:5]:  # Show first 5
            # Find the original contact
            for cpu_sig, cpu_contact, cpu_key in cpu_contact_list:
                if cpu_sig == sig:
                    print(f"     Key {cpu_key}: pos={cpu_contact['position']}, proj={cpu_contact['projection']:.6f}")
                    break
        if len(missing_in_gpu) > 5:
            print(f"     ... and {len(missing_in_gpu) - 5} more")
        success = False

    if extra_in_gpu:
        print(f"  FAILED: {len(extra_in_gpu)} contacts in GPU but not in CPU reference:")
        for sig in list(extra_in_gpu)[:5]:  # Show first 5
            # Find the original contact
            for gpu_sig, gpu_contact, gpu_idx in gpu_contact_list:
                if gpu_sig == sig:
                    print(f"     Index {gpu_idx}: pos={gpu_contact['position']}, proj={gpu_contact['projection']:.6f}")
                    break
        if len(extra_in_gpu) > 5:
            print(f"     ... and {len(extra_in_gpu) - 5} more")
        success = False

    if success:
        print(f"  PASSED: Contact sets are identical (order-independent)!")

    return success


def test_contact_reduction():
    """Launch the contact reduction test kernel with a single thread block of 256 threads."""
    wp.init()

    # Allocate output arrays
    max_contacts = 120  # Maximum possible contacts (20 faces * 6 directions)
    out_contacts = wp.zeros(max_contacts, dtype=ContactStruct)
    out_count = wp.zeros(1, dtype=int)

    # Launch with 1 block of 128 threads using tiled launch
    wp.launch_tiled(
        kernel=contact_reduction_test_kernel,
        dim=1,  # Total number of blocks
        inputs=[out_contacts, out_count],
        block_dim=128,  # Threads per block
    )

    # Read back results
    wp.synchronize()
    count = out_count.numpy()[0]
    contacts = out_contacts.numpy()

    print(f"Contact reduction test kernel completed successfully!")
    print(f"Number of contacts kept: {count}")

    # Print first few contacts for debugging
    for i in range(min(count, 10)):
        c = contacts[i]
        print(
            f"  Contact {i}: pos=({c['position'][0]:.4f}, {c['position'][1]:.4f}, {c['position'][2]:.4f}), "
            f"normal=({c['normal'][0]:.4f}, {c['normal'][1]:.4f}, {c['normal'][2]:.4f}), "
            f"depth={c['depth']:.4f}, projection={c['projection']:.4f}"
        )

    # Run validation
    validation_passed = validate_contact_reduction(out_contacts, out_count)

    return validation_passed


@wp.kernel(enable_backward=False)
def test_argmax_per_slot_kernel(
    values: wp.array(dtype=float),
    slots: wp.array(dtype=int),
    result: wp.array(dtype=int),
):
    """Test kernel for segmented argmax."""
    block_id, thread_id = wp.tid()

    slot = slots[thread_id]
    value = values[thread_id]

    winner = wp.static(create_segmented_argmax_func(128))(thread_id, slot, value)

    result[thread_id] = winner


def test_argmax():
    """Test the segmented argmax function."""
    import numpy as np

    wp.init()

    # Allocate arrays
    result = wp.zeros(128, dtype=int)
    values = wp.zeros(128, dtype=float)
    slots = wp.zeros(128, dtype=int)

    # Create test data: multiple slots with different values
    np.random.seed(42)
    values_np = np.random.rand(128).astype(np.float32) * 100.0
    slots_np = np.arange(128, dtype=np.int32) % 8  # 8 different slots, each with 16 threads

    values.assign(values_np)
    slots.assign(slots_np)

    # Launch with 1 block of 128 threads using tiled launch
    wp.launch_tiled(
        kernel=test_argmax_per_slot_kernel,
        dim=1,  # Total number of blocks
        inputs=[values, slots, result],
        block_dim=128,  # Threads per block
    )

    # Read back results
    wp.synchronize()
    result_np = result.numpy()

    # Compare with CPU results
    slot_to_argmax = {}
    slot_to_max_value = {}

    # Compute expected results on CPU
    for i in range(128):
        slot = slots_np[i]
        value = values_np[i]

        if slot not in slot_to_max_value or value > slot_to_max_value[slot]:
            slot_to_max_value[slot] = value
            slot_to_argmax[slot] = i

    # Verify GPU results
    success = True
    for i in range(128):
        slot = slots_np[i]
        expected_argmax = slot_to_argmax[slot]
        gpu_result = result_np[i]

        if gpu_result != expected_argmax:
            print(f"Thread {i} in slot {slot}: Expected argmax={expected_argmax}, got {gpu_result}")
            success = False

    if success:
        print("TestArgMax: All tests passed!")
    else:
        print("TestArgMax: FAILED!")

    return success


if __name__ == "__main__":
    # Run argmax test first
    print("=" * 60)
    print("Running Segmented ArgMax Test")
    print("=" * 60)
    test_argmax()

    print("\n" + "=" * 60)
    print("Running Contact Reduction Test")
    print("=" * 60)
    test_contact_reduction()
