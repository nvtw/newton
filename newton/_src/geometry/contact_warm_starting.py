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

"""
Contact warm starting for physics solvers.

Stores contact impulses from previous timestep and applies them to matching
contacts in the current timestep, improving solver convergence and stability.
"""

import warp as wp


@wp.func
def binary_search_pair(
    pairs: wp.array(dtype=wp.vec2i, ndim=1),
    target: wp.vec2i,
    num_pairs: int
) -> int:
    """Binary search for a body pair in a sorted array of pairs."""
    lower = 0
    upper = num_pairs

    while lower < upper:
        mid = (lower + upper) >> 1
        mid_pair = pairs[mid]

        # Compare pairs lexicographically (first by x, then by y)
        if mid_pair[0] < target[0] or (mid_pair[0] == target[0] and mid_pair[1] < target[1]):
            lower = mid + 1
        elif mid_pair[0] > target[0] or (mid_pair[0] == target[0] and mid_pair[1] > target[1]):
            upper = mid
        else:
            # Found exact match
            return mid

    # Not found
    return -1


@wp.struct
class StoredWrench3D:
    """
    Stored wrench (force + torque) from a contact point.

    Attributes:
        total_force: Total 3D impulse vector (normal + tangent combined)
        torque: Resulting torque relative to center of mass
    """

    total_force: wp.vec3  # Total 3D impulse vector (normal + tangent combined)
    torque: wp.vec3  # Resulting torque relative to CoM


########################################################
# Accumulate torque points
########################################################


@wp.kernel
def accumulate_torque_point(
    contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
    contact_points: wp.array(dtype=wp.vec3, ndim=1),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    compaction_map: wp.array(dtype=wp.int32, ndim=1),
    torque_point_accumulator: wp.array(dtype=wp.vec4, ndim=1),
):
    t = wp.tid()
    if t >= contact_count[0]:
        return

    impulse = contact_impulses[t]
    point = contact_points[t]

    weight = wp.length(impulse)

    compact_id = compaction_map[t]
    wp.atomic_add(torque_point_accumulator, compact_id, wp.vec4(weight * point[0], weight * point[1], weight * point[2], weight))


@wp.kernel
def accumulate_wrench(
    contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
    contact_points: wp.array(dtype=wp.vec3, ndim=1),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    compaction_map: wp.array(dtype=wp.int32, ndim=1),
    torque_point_accumulator: wp.array(dtype=wp.vec4, ndim=1),
    # Outputs
    wrenches: wp.array(dtype=StoredWrench3D, ndim=1),
):
    t = wp.tid()
    if t >= contact_count[0]:
        return

    impulse = contact_impulses[t]
    point = contact_points[t]
    compact_id = compaction_map[t]

    torque_point_4 = torque_point_accumulator[compact_id]
    weight = torque_point_4.w
    torque_point = wp.vec3(torque_point_4.x, torque_point_4.y, torque_point_4.z)
    if weight > 0.0:
        torque_point = torque_point / weight

    lever_arm = point - torque_point

    wp.atomic_add(wrenches[compact_id].total_force, impulse)
    wp.atomic_add(wrenches[compact_id].torque, wp.cross(lever_arm, impulse))


@wp.kernel
def store_local_torque_anchors(
    torque_point_accumulator: wp.array(dtype=wp.vec4, ndim=1),
    duplicate_free_body_pairs: wp.array(dtype=wp.vec2i, ndim=1), # Has same length as torque_point_accumulator
    all_body_transforms: wp.array(dtype=wp.transform, ndim=1),
    num_torque_points: wp.array(dtype=wp.int32, ndim=1),
    # Outputs
    torque_point_local_frame_a: wp.array(dtype=wp.vec3, ndim=1),
    torque_point_local_frame_b: wp.array(dtype=wp.vec3, ndim=1),
):
    t = wp.tid()
    if t >= num_torque_points[0]:
        return

    torque_point_4 = torque_point_accumulator[t]
    weight = torque_point_4.w
    torque_point = wp.vec3(torque_point_4.x, torque_point_4.y, torque_point_4.z)
    if weight > 0.0:
        torque_point = torque_point / weight

    pair = duplicate_free_body_pairs[t]
    body_a = pair[0]
    body_b = pair[1]
    transform_a = all_body_transforms[body_a]
    transform_b = all_body_transforms[body_b]
    torque_point_local_frame_a[t] = wp.transform_inverse(transform_a) * torque_point
    torque_point_local_frame_b[t] = wp.transform_inverse(transform_b) * torque_point


########################################################
# Apply warm starting
########################################################

@wp.kernel
def prepare_sort_body_pairs(
    body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    sorted_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    sorted_to_unsorted_map: wp.array(dtype=wp.int32, ndim=1),
    num_contacts: wp.array(dtype=wp.int32, ndim=1),
):
    """Copy body pairs for sorting and initialize index map."""
    tid = wp.tid()
    if tid < num_contacts[0]:
        sorted_body_pairs[tid] = body_pairs[tid]
        sorted_to_unsorted_map[tid] = tid


@wp.kernel
def mark_first_occurrences(
    sorted_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_contacts: wp.array(dtype=wp.int32, ndim=1),
    # Output
    is_first: wp.array(dtype=wp.int32, ndim=1),
):
    """Mark first occurrence of each unique body pair with 1, duplicates with 0."""
    tid = wp.tid()
    if tid >= num_contacts[0]:
        is_first[tid] = 0
        return

    current_pair = sorted_body_pairs[tid]

    # First element is always a first occurrence
    if tid == 0:
        is_first[tid] = 1
        return

    # Check if same as previous
    prev_pair = sorted_body_pairs[tid - 1]
    if prev_pair[0] == current_pair[0] and prev_pair[1] == current_pair[1]:
        is_first[tid] = 0  # Duplicate
    else:
        is_first[tid] = 1  # First occurrence


@wp.kernel
def create_compaction_map_from_scan(
    sorted_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    sorted_to_unsorted_map: wp.array(dtype=wp.int32, ndim=1),
    is_first: wp.array(dtype=wp.int32, ndim=1),
    prefix_sum: wp.array(dtype=wp.int32, ndim=1),
    num_contacts: wp.array(dtype=wp.int32, ndim=1),
    # Outputs
    unsorted_to_compact_map: wp.array(dtype=wp.int32, ndim=1),
    duplicate_free_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_unique_pairs: wp.array(dtype=wp.int32, ndim=1),
):
    """Use prefix sum to create compaction map and extract unique pairs."""
    tid = wp.tid()
    if tid >= num_contacts[0]:
        return

    unsorted_idx = sorted_to_unsorted_map[tid]

    # Compact ID is prefix_sum[tid] (exclusive scan gives us the index directly)
    compact_id = prefix_sum[tid]
    unsorted_to_compact_map[unsorted_idx] = compact_id

    # If this is a first occurrence, write the unique pair
    if is_first[tid] == 1:
        current_pair = sorted_body_pairs[tid]
        duplicate_free_body_pairs[compact_id] = current_pair

    # Last thread writes total count
    if tid == num_contacts[0] - 1:
        num_unique_pairs[0] = prefix_sum[tid] + is_first[tid]


@wp.kernel
def clear_accumulators(
    num_unique_pairs: wp.array(dtype=wp.int32, ndim=1),
    F_accumulator: wp.array(dtype=float, ndim=1),
    T_accumulator: wp.array(dtype=float, ndim=1),
    accumulator_count: wp.array(dtype=wp.int32, ndim=1),
):
    """Clear accumulator arrays for the number of unique pairs."""
    tid = wp.tid()
    if tid >= num_unique_pairs[0]:
        return

    base_offset = tid * 9
    for i in range(9):
        F_accumulator[base_offset + i] = 0.0
        T_accumulator[base_offset + i] = 0.0

    accumulator_count[tid] = 0


@wp.kernel
def accumulate_F_and_T(
    duplicate_free_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    torque_point_local_frame_a: wp.array(dtype=wp.vec3, ndim=1),
    torque_point_local_frame_b: wp.array(dtype=wp.vec3, ndim=1),
    num_torque_points: wp.array(dtype=wp.int32, ndim=1),
    new_contact_points: wp.array(dtype=wp.vec3, ndim=1),
    new_contact_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_new_contact_points: wp.array(dtype=wp.int32, ndim=1),
    all_body_transforms: wp.array(dtype=wp.transform, ndim=1),
    # Outputs (flat arrays: 9 floats per matrix)
    F_accumulator: wp.array(dtype=float, ndim=1),
    T_accumulator: wp.array(dtype=float, ndim=1),
    accumulator_count_per_element: wp.array(dtype=wp.int32, ndim=1),
):
    i = wp.tid()
    if i >= num_new_contact_points[0]:
        return

    pair = new_contact_pairs[i]

    body_a = pair[0]
    body_b = pair[1]
    transform_a = all_body_transforms[body_a]
    transform_b = all_body_transforms[body_b]

    compact_id = binary_search_pair(duplicate_free_body_pairs, pair, num_torque_points[0])

    if compact_id < 0:
        return

    reference_point_a = transform_a * torque_point_local_frame_a[compact_id]
    reference_point_b = transform_b * torque_point_local_frame_b[compact_id]
    reference_point = (reference_point_a + reference_point_b) * 0.5

    ri = reference_point

    # Base offset for this compact_id's 3x3 matrix (9 elements)
    base_offset = compact_id * 9

    # [r]×^T = -[r]× has rows: [0, rz, -ry], [-rz, 0, rx], [ry, -rx, 0]
    # Store in row-major order
    wp.atomic_add(F_accumulator, base_offset + 0, 0.0)       # F[0,0]
    wp.atomic_add(F_accumulator, base_offset + 1, ri[2])     # F[0,1]
    wp.atomic_add(F_accumulator, base_offset + 2, -ri[1])    # F[0,2]
    wp.atomic_add(F_accumulator, base_offset + 3, -ri[2])    # F[1,0]
    wp.atomic_add(F_accumulator, base_offset + 4, 0.0)       # F[1,1]
    wp.atomic_add(F_accumulator, base_offset + 5, ri[0])     # F[1,2]
    wp.atomic_add(F_accumulator, base_offset + 6, ri[1])     # F[2,0]
    wp.atomic_add(F_accumulator, base_offset + 7, -ri[0])    # F[2,1]
    wp.atomic_add(F_accumulator, base_offset + 8, 0.0)       # F[2,2]

    # [r]× * [r]×^T = |r|² I - r ⊗ r
    # Store in row-major order
    rSq = wp.dot(ri, ri)
    wp.atomic_add(T_accumulator, base_offset + 0, rSq - ri[0] * ri[0])  # T[0,0]
    wp.atomic_add(T_accumulator, base_offset + 1, -ri[0] * ri[1])       # T[0,1]
    wp.atomic_add(T_accumulator, base_offset + 2, -ri[0] * ri[2])       # T[0,2]
    wp.atomic_add(T_accumulator, base_offset + 3, -ri[0] * ri[1])       # T[1,0]
    wp.atomic_add(T_accumulator, base_offset + 4, rSq - ri[1] * ri[1])  # T[1,1]
    wp.atomic_add(T_accumulator, base_offset + 5, -ri[1] * ri[2])       # T[1,2]
    wp.atomic_add(T_accumulator, base_offset + 6, -ri[0] * ri[2])       # T[2,0]
    wp.atomic_add(T_accumulator, base_offset + 7, -ri[1] * ri[2])       # T[2,1]
    wp.atomic_add(T_accumulator, base_offset + 8, rSq - ri[2] * ri[2])  # T[2,2]

    wp.atomic_add(accumulator_count_per_element, compact_id, 1)


@wp.kernel
def apply_warm_starting(
    duplicate_free_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    torque_point_local_frame_a: wp.array(dtype=wp.vec3, ndim=1),
    torque_point_local_frame_b: wp.array(dtype=wp.vec3, ndim=1),
    num_torque_points: wp.array(dtype=wp.int32, ndim=1),
    new_contact_points: wp.array(dtype=wp.vec3, ndim=1),
    new_contact_normals: wp.array(dtype=wp.vec3, ndim=1),
    new_contact_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_new_contact_points: wp.array(dtype=wp.int32, ndim=1),
    all_body_transforms: wp.array(dtype=wp.transform, ndim=1),
    F_accumulator: wp.array(dtype=float, ndim=1),
    T_accumulator: wp.array(dtype=float, ndim=1),
    accumulator_count_per_element: wp.array(dtype=wp.int32, ndim=1),
    wrenches: wp.array(dtype=StoredWrench3D, ndim=1),
    # Outputs
    contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
):
    i = wp.tid()
    if i >= num_new_contact_points[0]:
        return

    pair = new_contact_pairs[i]
    contact_point = new_contact_points[i]

    compact_id = binary_search_pair(duplicate_free_body_pairs, pair, num_torque_points[0])

    if compact_id < 0:
        return

    # Get the number of contacts for this body pair
    n = accumulator_count_per_element[compact_id]
    if n == 0:
        return

    invN = 1.0 / float(n)

    # Get stored wrench (totalForce and totalTorque)
    wrench = wrenches[compact_id]
    totalForce = wrench.total_force
    totalTorque = wrench.torque

    # Load F_T and T_T matrices from flat arrays
    base_offset = compact_id * 9

    # F_T matrix (row-major)
    F_T = wp.mat33(
        F_accumulator[base_offset + 0], F_accumulator[base_offset + 1], F_accumulator[base_offset + 2],
        F_accumulator[base_offset + 3], F_accumulator[base_offset + 4], F_accumulator[base_offset + 5],
        F_accumulator[base_offset + 6], F_accumulator[base_offset + 7], F_accumulator[base_offset + 8]
    )

    # T_T matrix (row-major, symmetric)
    T_T = wp.mat33(
        T_accumulator[base_offset + 0], T_accumulator[base_offset + 1], T_accumulator[base_offset + 2],
        T_accumulator[base_offset + 3], T_accumulator[base_offset + 4], T_accumulator[base_offset + 5],
        T_accumulator[base_offset + 6], T_accumulator[base_offset + 7], T_accumulator[base_offset + 8]
    )

    # Compute Schur complement: S = T_T - (1/n) * F_T^T * F_T
    # F_T^T * F_T is symmetric
    S = wp.mat33()
    S[0, 0] = T_T[0, 0] - invN * (F_T[0, 0]*F_T[0, 0] + F_T[1, 0]*F_T[1, 0] + F_T[2, 0]*F_T[2, 0])
    S[1, 1] = T_T[1, 1] - invN * (F_T[0, 1]*F_T[0, 1] + F_T[1, 1]*F_T[1, 1] + F_T[2, 1]*F_T[2, 1])
    S[2, 2] = T_T[2, 2] - invN * (F_T[0, 2]*F_T[0, 2] + F_T[1, 2]*F_T[1, 2] + F_T[2, 2]*F_T[2, 2])
    S[0, 1] = T_T[0, 1] - invN * (F_T[0, 0]*F_T[0, 1] + F_T[1, 0]*F_T[1, 1] + F_T[2, 0]*F_T[2, 1])
    S[0, 2] = T_T[0, 2] - invN * (F_T[0, 0]*F_T[0, 2] + F_T[1, 0]*F_T[1, 2] + F_T[2, 0]*F_T[2, 2])
    S[1, 2] = T_T[1, 2] - invN * (F_T[0, 1]*F_T[0, 2] + F_T[1, 1]*F_T[1, 2] + F_T[2, 1]*F_T[2, 2])
    S[1, 0] = S[0, 1]  # Symmetric
    S[2, 0] = S[0, 2]
    S[2, 1] = S[1, 2]

    # Tikhonov regularization: S' = S + ε*I
    TIKHONOV_DAMPING = 1e-5
    S[0, 0] += TIKHONOV_DAMPING
    S[1, 1] += TIKHONOV_DAMPING
    S[2, 2] += TIKHONOV_DAMPING

    # Compute modified RHS: rhs = totalTorque - (1/n) * F_T^T * totalForce
    F_T_T_times_force = wp.transpose(F_T) * totalForce
    rhs = totalTorque - F_T_T_times_force * invN

    # Solve 3×3 system: S * lambda_t = rhs
    # Check if matrix is invertible
    det = wp.determinant(S)
    if wp.abs(det) < 1e-12:
        # Fallback: uniform force distribution
        contact_impulses[i] = totalForce * invN
        return

    # Solve using matrix inverse
    S_inv = wp.inverse(S)
    lambda_t = S_inv * rhs

    # Back-substitute to get lambda_f: lambda_f = (1/n) * (totalForce - F_T * lambda_t)
    F_T_times_lambda_t = F_T * lambda_t
    lambda_f = (totalForce - F_T_times_lambda_t) * invN

    # Compute reference point for this contact
    body_a = pair[0]
    body_b = pair[1]
    transform_a = all_body_transforms[body_a]
    transform_b = all_body_transforms[body_b]
    reference_point_a = transform_a * torque_point_local_frame_a[compact_id]
    reference_point_b = transform_b * torque_point_local_frame_b[compact_id]
    reference_point = (reference_point_a + reference_point_b) * 0.5

    # Compute r[i] = contact_point - reference_point
    r_i = contact_point - reference_point

    # Reconstruct impulse: impulse[i] = lambda_f - r[i] × lambda_t
    impulse = lambda_f - wp.cross(r_i, lambda_t)

    # Ensure contact can only push, not pull
    # Decompose into normal and tangential components
    contact_normal = new_contact_normals[i]
    impulse_normal_magnitude = wp.dot(impulse, contact_normal)

    # Clamp normal component to be non-negative (only push)
    if impulse_normal_magnitude < 0.0:
        # Remove pulling component by projecting onto tangent plane
        impulse = impulse - impulse_normal_magnitude * contact_normal

    contact_impulses[i] = impulse




class ContactWarmStarting:
    """
    Manages contact warm starting for physics solvers.

    Stores contact impulses from previous timestep and applies them to matching
    contacts to improve solver convergence.

    Args:
        max_num_contacts: Maximum number of contacts to track
        device: Device to allocate buffers on (None for default)
    """

    def __init__(self, max_num_contacts: int, device=None):
        self.max_num_contacts = max_num_contacts
        self.device = device

        self.torque_point_accumulator = wp.zeros(max_num_contacts, dtype=wp.vec4, device=device)

        self.sorted_body_pairs = wp.zeros(max_num_contacts, dtype=wp.vec2i, device=device)
        self.sorted_to_unsorted_map = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)
        self.unsorted_to_compact_map = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)

        # Buffers for scan-based compaction
        self.is_first_occurrence = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)
        self.prefix_sum = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)

        self.duplicate_free_sorted_body_pairs = wp.zeros(max_num_contacts, dtype=wp.vec2i, device=device)
        self.duplicate_free_sorted_body_pairs_count = wp.zeros(1, dtype=wp.int32, device=device)
        # Wrenches and torque points are stored once per dupplicate free (=unique) body pair
        self.wrenches = wp.zeros(max_num_contacts, dtype=StoredWrench3D, device=device)
        self.torque_point_local_frame_a = wp.zeros(max_num_contacts, dtype=wp.vec3, device=device)
        self.torque_point_local_frame_b = wp.zeros(max_num_contacts, dtype=wp.vec3, device=device)

        # Flat arrays: 9 floats per 3x3 matrix (row-major order)
        self.F_accumulator = wp.zeros(max_num_contacts * 9, dtype=float, device=device)
        self.T_accumulator = wp.zeros(max_num_contacts * 9, dtype=float, device=device)
        self.accumulator_count_per_element = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)

    def capture_wrenches(
        self,
        contact_points: wp.array(dtype=wp.vec3, ndim=1),
        contact_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
        contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
        num_contacts: wp.array(dtype=wp.int32, ndim=1),
        all_body_transforms: wp.array(dtype=wp.transform, ndim=1),
        device=None,
    ):
        """
        Capture wrenches from solved contacts for warm starting next timestep.

        Args:
            contact_points: Contact points in world space
            contact_body_pairs: Body pair indices for each contact
            contact_impulses: Solved impulses from this timestep
            num_contacts: Number of active contacts
            all_body_transforms: Current body transforms
            device: Device to launch on (None for default)
        """
        if device is None:
            device = self.device

        # Step 1: Prepare and sort body pairs
        wp.launch(
            kernel=prepare_sort_body_pairs,
            dim=self.max_num_contacts,
            inputs=[
                contact_body_pairs,
                self.sorted_body_pairs,
                self.sorted_to_unsorted_map,
                num_contacts,
            ],
            device=device,
        )

        # Cast to int64 for radix_sort_pairs
        body_pairs_as_int64 = wp.array(
            ptr=self.sorted_body_pairs.ptr,
            dtype=wp.int64,
            shape=(self.max_num_contacts,),
            device=device,
            copy=False,
        )

        wp.utils.radix_sort_pairs(body_pairs_as_int64, self.sorted_to_unsorted_map, self.max_num_contacts)

        # Step 2: Mark first occurrences of unique pairs
        wp.launch(
            kernel=mark_first_occurrences,
            dim=self.max_num_contacts,
            inputs=[
                self.sorted_body_pairs,
                num_contacts,
                self.is_first_occurrence,
            ],
            device=device,
        )

        # Step 3: Compute prefix sum (exclusive scan) to get compact indices
        wp.utils.array_scan(self.is_first_occurrence, self.prefix_sum, inclusive=False)

        # Step 4: Create compaction map and extract unique pairs using scan results
        self.duplicate_free_sorted_body_pairs_count.zero_()
        wp.launch(
            kernel=create_compaction_map_from_scan,
            dim=self.max_num_contacts,
            inputs=[
                self.sorted_body_pairs,
                self.sorted_to_unsorted_map,
                self.is_first_occurrence,
                self.prefix_sum,
                num_contacts,
                self.unsorted_to_compact_map,
                self.duplicate_free_sorted_body_pairs,
                self.duplicate_free_sorted_body_pairs_count,
            ],
            device=device,
        )

        # Step 5: Accumulate torque points (weighted by impulse magnitude)
        self.torque_point_accumulator.zero_()
        wp.launch(
            kernel=accumulate_torque_point,
            dim=self.max_num_contacts,
            inputs=[
                contact_impulses,
                contact_points,
                num_contacts,
                self.unsorted_to_compact_map,
                self.torque_point_accumulator,
            ],
            device=device,
        )

        # Step 6: Accumulate wrenches (force and torque)
        self.wrenches.zero_()
        wp.launch(
            kernel=accumulate_wrench,
            dim=self.max_num_contacts,
            inputs=[
                contact_impulses,
                contact_points,
                num_contacts,
                self.unsorted_to_compact_map,
                self.torque_point_accumulator,
                self.wrenches,
            ],
            device=device,
        )

        # Step 7: Store torque points in local frames
        wp.launch(
            kernel=store_local_torque_anchors,
            dim=self.max_num_contacts,
            inputs=[
                self.torque_point_accumulator,
                self.duplicate_free_sorted_body_pairs,
                all_body_transforms,
                self.duplicate_free_sorted_body_pairs_count,
                self.torque_point_local_frame_a,
                self.torque_point_local_frame_b,
            ],
            device=device,
        )

    def apply_warm_starting(
        self,
        new_contact_points: wp.array(dtype=wp.vec3, ndim=1),
        new_contact_normals: wp.array(dtype=wp.vec3, ndim=1),
        new_contact_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
        num_new_contact_points: wp.array(dtype=wp.int32, ndim=1),
        all_body_transforms: wp.array(dtype=wp.transform, ndim=1),
        # Outputs
        contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
        device=None,
    ):
        """
        Apply warm starting to new contacts using stored wrench data.

        Args:
            new_contact_points: Contact points in world space
            new_contact_normals: Contact normals (pointing from A to B)
            new_contact_body_pairs: Body pair indices for each contact
            num_new_contact_points: Number of active contacts
            all_body_transforms: Current body transforms
            contact_impulses: Output array for reconstructed impulses
            device: Device to launch on
        """
        if device is None:
            device = self.device

        # Clear accumulators
        self.F_accumulator.zero_()
        self.T_accumulator.zero_()
        self.accumulator_count_per_element.zero_()

        # Step 1: Accumulate F_T and T_T matrices
        wp.launch(
            kernel=accumulate_F_and_T,
            dim=self.max_num_contacts,
            inputs=[
                self.duplicate_free_sorted_body_pairs,
                self.torque_point_local_frame_a,
                self.torque_point_local_frame_b,
                self.duplicate_free_sorted_body_pairs_count,
                new_contact_points,
                new_contact_body_pairs,
                num_new_contact_points,
                all_body_transforms,
                self.F_accumulator,
                self.T_accumulator,
                self.accumulator_count_per_element,
            ],
            device=device,
        )

        # Step 2: Apply warm starting to reconstruct impulses
        wp.launch(
            kernel=apply_warm_starting,
            dim=self.max_num_contacts,
            inputs=[
                self.duplicate_free_sorted_body_pairs,
                self.torque_point_local_frame_a,
                self.torque_point_local_frame_b,
                self.duplicate_free_sorted_body_pairs_count,
                new_contact_points,
                new_contact_normals,
                new_contact_body_pairs,
                num_new_contact_points,
                all_body_transforms,
                self.F_accumulator,
                self.T_accumulator,
                self.accumulator_count_per_element,
                self.wrenches,
                contact_impulses,
            ],
            device=device,
        )
