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

LEAST SQUARES WRENCH DISTRIBUTION:

When multiple contacts exist between a body pair, we store their combined wrench
(total force F_total and torque T_total). To reconstruct individual contact
impulses, we solve: minimize Σ||f_i||² subject to Σf_i = F_total and Σ(r_i x f_i) = T_total.

Rather than solving a 3Nx3N system for N contacts, we exploit the problem structure.
The optimal solution has the form f_i = λ_f - r_i x λ_t, where λ_f ∈ ℝ³ is a force
scaling and λ_t ∈ ℝ³ is a torque multiplier (Lagrange multipliers).

We precompute two 3x3 matrices by accumulating over all contacts:
  F_T = Σ[r_i]x        (sum of skew-symmetric matrices)
  T_T = Σ[r_i]x[r_i]xᵀ (sum of cross-product squared matrices)

Then solve the 3x3 Schur complement system S·λ_t = rhs, where:
  S = T_T - (1/N)·F_Tᵀ·F_T
  rhs = T_total - (1/N)·F_Tᵀ·F_total

Finally, λ_f = (F_total - F_T·λ_t)/N and each impulse is f_i = λ_f - r_i x λ_t.

This reduces arbitrary N contacts to a constant-size 3x3 linear system.
"""

import warp as wp


@wp.func
def binary_search_pair(pairs: wp.array(dtype=wp.vec2i, ndim=1), target: wp.vec2i, num_pairs: int) -> int:
    """Binary search for a body pair in a sorted array of pairs."""
    lower = int(0)
    upper = int(num_pairs)

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

    total_force_in_body_a_frame: wp.vec3  # Total 3D impulse vector (normal + tangent combined)
    total_force_in_body_b_frame: wp.vec3  # Total 3D impulse vector (normal + tangent combined)
    torque_a_in_body_a_frame: wp.vec3  # Resulting torque relative to CoM of body A
    torque_b_in_body_b_frame: wp.vec3  # Resulting torque relative to CoM of body B


########################################################
# Accumulate torque points
########################################################


@wp.kernel
def accumulate_wrench(
    contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
    contact_points: wp.array(dtype=wp.vec3, ndim=1),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    compaction_map: wp.array(dtype=wp.int32, ndim=1),
    duplicate_free_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    all_body_COM_transforms: wp.array(dtype=wp.transform, ndim=1),
    # Outputs (flat array: 12 floats per wrench)
    wrenches: wp.array(dtype=float, ndim=1),
):
    t = wp.tid()
    if t >= contact_count[0]:
        return

    impulse = contact_impulses[t]
    point = contact_points[t]
    compact_id = compaction_map[t]

    # Get body pair
    pair = duplicate_free_body_pairs[compact_id]
    body_a = pair[0]
    body_b = pair[1]

    # Get transforms
    transform_a = all_body_COM_transforms[body_a]
    transform_b = all_body_COM_transforms[body_b]

    # Extract COM positions
    com_a = wp.transform_get_translation(transform_a)
    com_b = wp.transform_get_translation(transform_b)

    # Compute torques in world space (relative to each COM)
    torque_a_world = wp.cross(point - com_a, impulse)
    torque_b_world = wp.cross(point - com_b, impulse)

    # Transform forces and torques to local body frames
    impulse_a_local = wp.transform_vector(wp.transform_inverse(transform_a), impulse)
    impulse_b_local = wp.transform_vector(wp.transform_inverse(transform_b), impulse)
    torque_a_local = wp.transform_vector(wp.transform_inverse(transform_a), torque_a_world)
    torque_b_local = wp.transform_vector(wp.transform_inverse(transform_b), torque_b_world)

    # Accumulate in local frames (component-wise since atomic_add doesn't support vec3)
    # Flat array layout: 12 floats per wrench
    # [0-2]: total_force_in_body_a_frame
    # [3-5]: total_force_in_body_b_frame
    # [6-8]: torque_a_in_body_a_frame
    # [9-11]: torque_b_in_body_b_frame
    base_offset = compact_id * 12

    # For total_force_in_body_a_frame
    wp.atomic_add(wrenches, base_offset + 0, impulse_a_local[0])
    wp.atomic_add(wrenches, base_offset + 1, impulse_a_local[1])
    wp.atomic_add(wrenches, base_offset + 2, impulse_a_local[2])
    # For total_force_in_body_b_frame
    wp.atomic_add(wrenches, base_offset + 3, impulse_b_local[0])
    wp.atomic_add(wrenches, base_offset + 4, impulse_b_local[1])
    wp.atomic_add(wrenches, base_offset + 5, impulse_b_local[2])
    # For torque_a_in_body_a_frame
    wp.atomic_add(wrenches, base_offset + 6, torque_a_local[0])
    wp.atomic_add(wrenches, base_offset + 7, torque_a_local[1])
    wp.atomic_add(wrenches, base_offset + 8, torque_a_local[2])
    # For torque_b_in_body_b_frame
    wp.atomic_add(wrenches, base_offset + 9, torque_b_local[0])
    wp.atomic_add(wrenches, base_offset + 10, torque_b_local[1])
    wp.atomic_add(wrenches, base_offset + 11, torque_b_local[2])


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
    """Copy body pairs for sorting and initialize index map. Mask unused entries with max int."""
    tid = wp.tid()
    if tid < num_contacts[0]:
        sorted_body_pairs[tid] = body_pairs[tid]
        sorted_to_unsorted_map[tid] = tid
    else:
        # Mask unused entries so they sort to the end
        # Use max int32 value so these pairs sort after all valid pairs
        sorted_body_pairs[tid] = wp.vec2i(2147483647, 2147483647)
        # sorted_to_unsorted_map[tid] = tid # Not needed - save bandwith by not writing to it


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

    # Compact ID is prefix_sum[tid] - 1 (inclusive scan gives us the count, subtract 1 for 0-based index)
    compact_id = prefix_sum[tid] - 1
    unsorted_to_compact_map[unsorted_idx] = compact_id

    # If this is a first occurrence, write the unique pair
    if is_first[tid] == 1:
        current_pair = sorted_body_pairs[tid]
        duplicate_free_body_pairs[compact_id] = current_pair

    # Last thread writes total count (inclusive scan gives us the total directly)
    if tid == num_contacts[0] - 1:
        num_unique_pairs[0] = prefix_sum[tid]


@wp.kernel
def clear_accumulators(
    num_unique_pairs: wp.array(dtype=wp.int32, ndim=1),
    F_accumulator_a: wp.array(dtype=float, ndim=1),
    T_accumulator_a: wp.array(dtype=float, ndim=1),
    F_accumulator_b: wp.array(dtype=float, ndim=1),
    T_accumulator_b: wp.array(dtype=float, ndim=1),
    accumulator_count: wp.array(dtype=wp.int32, ndim=1),
):
    """Clear accumulator arrays for the number of unique pairs."""
    tid = wp.tid()
    if tid >= num_unique_pairs[0]:
        return

    base_offset = tid * 9
    for i in range(9):
        F_accumulator_a[base_offset + i] = 0.0
        T_accumulator_a[base_offset + i] = 0.0
        F_accumulator_b[base_offset + i] = 0.0
        T_accumulator_b[base_offset + i] = 0.0

    accumulator_count[tid] = 0


@wp.func
def solve_wrench_distribution(
    totalForce: wp.vec3,
    totalTorque: wp.vec3,
    F_T: wp.mat33,
    T_T: wp.mat33,
    r_i: wp.vec3,
    invN: float,
) -> wp.vec3:
    """
    Solve the least-squares problem to distribute a total wrench to a single contact point.

    Args:
        totalForce: Total force in world frame
        totalTorque: Total torque in world frame
        F_T: Accumulated cross-product matrix for all contacts
        T_T: Accumulated cross-product squared matrix for all contacts
        r_i: Lever arm from COM to contact point
        invN: 1/n where n is the number of contacts in this body pair

    Returns:
        Reconstructed impulse at the contact point
    """
    # Compute Schur complement: S = T_T - (1/n) * F_T^T * F_T
    S = wp.mat33()
    S[0, 0] = T_T[0, 0] - invN * (F_T[0, 0] * F_T[0, 0] + F_T[1, 0] * F_T[1, 0] + F_T[2, 0] * F_T[2, 0])
    S[1, 1] = T_T[1, 1] - invN * (F_T[0, 1] * F_T[0, 1] + F_T[1, 1] * F_T[1, 1] + F_T[2, 1] * F_T[2, 1])
    S[2, 2] = T_T[2, 2] - invN * (F_T[0, 2] * F_T[0, 2] + F_T[1, 2] * F_T[1, 2] + F_T[2, 2] * F_T[2, 2])
    S[0, 1] = T_T[0, 1] - invN * (F_T[0, 0] * F_T[0, 1] + F_T[1, 0] * F_T[1, 1] + F_T[2, 0] * F_T[2, 1])
    S[0, 2] = T_T[0, 2] - invN * (F_T[0, 0] * F_T[0, 2] + F_T[1, 0] * F_T[1, 2] + F_T[2, 0] * F_T[2, 2])
    S[1, 2] = T_T[1, 2] - invN * (F_T[0, 1] * F_T[0, 2] + F_T[1, 1] * F_T[1, 2] + F_T[2, 1] * F_T[2, 2])
    S[1, 0] = S[0, 1]  # Symmetric
    S[2, 0] = S[0, 2]
    S[2, 1] = S[1, 2]

    # Tikhonov regularization: S' = S + ε*I
    TIKHONOV_DAMPING = 1e-5
    S[0, 0] += TIKHONOV_DAMPING
    S[1, 1] += TIKHONOV_DAMPING
    S[2, 2] += TIKHONOV_DAMPING

    # Compute modified RHS: rhs = totalTorque - (1/n) * F_T^T * totalForce
    rhs = totalTorque - wp.transpose(F_T) * totalForce * invN

    # Solve 3x3 system or use fallback
    impulse = wp.vec3(0.0, 0.0, 0.0)
    det = wp.determinant(S)
    if wp.abs(det) < 1e-12:
        # Fallback: uniform force distribution
        impulse = totalForce * invN
    else:
        # Solve using matrix inverse
        S_inv = wp.inverse(S)
        lambda_t = S_inv * rhs
        lambda_f = (totalForce - F_T * lambda_t) * invN
        # Reconstruct impulse: impulse[i] = lambda_f - r[i] x lambda_t
        impulse = lambda_f - wp.cross(r_i, lambda_t)

    return impulse


@wp.kernel
def accumulate_F_and_T(
    duplicate_free_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_unique_pairs: wp.array(dtype=wp.int32, ndim=1),
    new_contact_points: wp.array(dtype=wp.vec3, ndim=1),
    new_contact_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_new_contact_points: wp.array(dtype=wp.int32, ndim=1),
    all_body_COM_transforms: wp.array(dtype=wp.transform, ndim=1),
    # Outputs (flat arrays: 9 floats per matrix, separate for A and B)
    F_accumulator_a: wp.array(dtype=float, ndim=1),
    T_accumulator_a: wp.array(dtype=float, ndim=1),
    F_accumulator_b: wp.array(dtype=float, ndim=1),
    T_accumulator_b: wp.array(dtype=float, ndim=1),
    accumulator_count_per_element: wp.array(dtype=wp.int32, ndim=1),
):
    i = wp.tid()
    if i >= num_new_contact_points[0]:
        return

    pair = new_contact_pairs[i]
    contact_point = new_contact_points[i]

    body_a = pair[0]
    body_b = pair[1]
    transform_a = all_body_COM_transforms[body_a]
    transform_b = all_body_COM_transforms[body_b]

    compact_id = binary_search_pair(duplicate_free_body_pairs, pair, num_unique_pairs[0])

    if compact_id < 0:
        return

    # Get COM positions (reference points)
    com_a = wp.transform_get_translation(transform_a)
    com_b = wp.transform_get_translation(transform_b)

    # Compute r relative to each COM
    r_a = contact_point - com_a
    r_b = contact_point - com_b

    # Base offset for this compact_id's 3x3 matrix (9 elements)
    base_offset = compact_id * 9

    # Accumulate for body A
    # [r]x^T = -[r]x has rows: [0, rz, -ry], [-rz, 0, rx], [ry, -rx, 0]
    wp.atomic_add(F_accumulator_a, base_offset + 0, 0.0)  # F[0,0]
    wp.atomic_add(F_accumulator_a, base_offset + 1, r_a[2])  # F[0,1]
    wp.atomic_add(F_accumulator_a, base_offset + 2, -r_a[1])  # F[0,2]
    wp.atomic_add(F_accumulator_a, base_offset + 3, -r_a[2])  # F[1,0]
    wp.atomic_add(F_accumulator_a, base_offset + 4, 0.0)  # F[1,1]
    wp.atomic_add(F_accumulator_a, base_offset + 5, r_a[0])  # F[1,2]
    wp.atomic_add(F_accumulator_a, base_offset + 6, r_a[1])  # F[2,0]
    wp.atomic_add(F_accumulator_a, base_offset + 7, -r_a[0])  # F[2,1]
    wp.atomic_add(F_accumulator_a, base_offset + 8, 0.0)  # F[2,2]

    # [r]x * [r]x^T = |r|² I - r ⊗ r
    rSq_a = wp.dot(r_a, r_a)
    wp.atomic_add(T_accumulator_a, base_offset + 0, rSq_a - r_a[0] * r_a[0])  # T[0,0]
    wp.atomic_add(T_accumulator_a, base_offset + 1, -r_a[0] * r_a[1])  # T[0,1]
    wp.atomic_add(T_accumulator_a, base_offset + 2, -r_a[0] * r_a[2])  # T[0,2]
    wp.atomic_add(T_accumulator_a, base_offset + 3, -r_a[0] * r_a[1])  # T[1,0]
    wp.atomic_add(T_accumulator_a, base_offset + 4, rSq_a - r_a[1] * r_a[1])  # T[1,1]
    wp.atomic_add(T_accumulator_a, base_offset + 5, -r_a[1] * r_a[2])  # T[1,2]
    wp.atomic_add(T_accumulator_a, base_offset + 6, -r_a[0] * r_a[2])  # T[2,0]
    wp.atomic_add(T_accumulator_a, base_offset + 7, -r_a[1] * r_a[2])  # T[2,1]
    wp.atomic_add(T_accumulator_a, base_offset + 8, rSq_a - r_a[2] * r_a[2])  # T[2,2]

    # Accumulate for body B
    wp.atomic_add(F_accumulator_b, base_offset + 0, 0.0)  # F[0,0]
    wp.atomic_add(F_accumulator_b, base_offset + 1, r_b[2])  # F[0,1]
    wp.atomic_add(F_accumulator_b, base_offset + 2, -r_b[1])  # F[0,2]
    wp.atomic_add(F_accumulator_b, base_offset + 3, -r_b[2])  # F[1,0]
    wp.atomic_add(F_accumulator_b, base_offset + 4, 0.0)  # F[1,1]
    wp.atomic_add(F_accumulator_b, base_offset + 5, r_b[0])  # F[1,2]
    wp.atomic_add(F_accumulator_b, base_offset + 6, r_b[1])  # F[2,0]
    wp.atomic_add(F_accumulator_b, base_offset + 7, -r_b[0])  # F[2,1]
    wp.atomic_add(F_accumulator_b, base_offset + 8, 0.0)  # F[2,2]

    rSq_b = wp.dot(r_b, r_b)
    wp.atomic_add(T_accumulator_b, base_offset + 0, rSq_b - r_b[0] * r_b[0])  # T[0,0]
    wp.atomic_add(T_accumulator_b, base_offset + 1, -r_b[0] * r_b[1])  # T[0,1]
    wp.atomic_add(T_accumulator_b, base_offset + 2, -r_b[0] * r_b[2])  # T[0,2]
    wp.atomic_add(T_accumulator_b, base_offset + 3, -r_b[0] * r_b[1])  # T[1,0]
    wp.atomic_add(T_accumulator_b, base_offset + 4, rSq_b - r_b[1] * r_b[1])  # T[1,1]
    wp.atomic_add(T_accumulator_b, base_offset + 5, -r_b[1] * r_b[2])  # T[1,2]
    wp.atomic_add(T_accumulator_b, base_offset + 6, -r_b[0] * r_b[2])  # T[2,0]
    wp.atomic_add(T_accumulator_b, base_offset + 7, -r_b[1] * r_b[2])  # T[2,1]
    wp.atomic_add(T_accumulator_b, base_offset + 8, rSq_b - r_b[2] * r_b[2])  # T[2,2]

    wp.atomic_add(accumulator_count_per_element, compact_id, 1)


@wp.kernel
def apply_warm_starting(
    duplicate_free_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_unique_pairs: wp.array(dtype=wp.int32, ndim=1),
    new_contact_points: wp.array(dtype=wp.vec3, ndim=1),
    new_contact_normals: wp.array(dtype=wp.vec3, ndim=1),
    new_contact_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_new_contact_points: wp.array(dtype=wp.int32, ndim=1),
    all_body_COM_transforms: wp.array(dtype=wp.transform, ndim=1),
    F_accumulator_a: wp.array(dtype=float, ndim=1),
    T_accumulator_a: wp.array(dtype=float, ndim=1),
    F_accumulator_b: wp.array(dtype=float, ndim=1),
    T_accumulator_b: wp.array(dtype=float, ndim=1),
    accumulator_count_per_element: wp.array(dtype=wp.int32, ndim=1),
    wrenches: wp.array(dtype=float, ndim=1),  # Flat array: 12 floats per wrench
    # Outputs
    contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
):
    i = wp.tid()
    if i >= num_new_contact_points[0]:
        return

    pair = new_contact_pairs[i]
    contact_point = new_contact_points[i]

    compact_id = binary_search_pair(duplicate_free_body_pairs, pair, num_unique_pairs[0])

    if compact_id < 0:
        return

    # Get the number of contacts for this body pair
    n = accumulator_count_per_element[compact_id]
    if n == 0:
        return

    invN = 1.0 / float(n)

    # Load stored wrench from flat array (12 floats per wrench)
    wrench_offset = compact_id * 12
    total_force_in_body_a_frame = wp.vec3(
        wrenches[wrench_offset + 0],
        wrenches[wrench_offset + 1],
        wrenches[wrench_offset + 2],
    )
    total_force_in_body_b_frame = wp.vec3(
        wrenches[wrench_offset + 3],
        wrenches[wrench_offset + 4],
        wrenches[wrench_offset + 5],
    )
    torque_a_in_body_a_frame = wp.vec3(
        wrenches[wrench_offset + 6],
        wrenches[wrench_offset + 7],
        wrenches[wrench_offset + 8],
    )
    torque_b_in_body_b_frame = wp.vec3(
        wrenches[wrench_offset + 9],
        wrenches[wrench_offset + 10],
        wrenches[wrench_offset + 11],
    )

    # Get body transforms
    body_a = pair[0]
    body_b = pair[1]
    transform_a = all_body_COM_transforms[body_a]
    transform_b = all_body_COM_transforms[body_b]

    # Get COM positions
    com_a = wp.transform_get_translation(transform_a)
    com_b = wp.transform_get_translation(transform_b)

    # Compute r[i] for both bodies
    r_i_a = contact_point - com_a
    r_i_b = contact_point - com_b

    # Load F_T and T_T matrices from flat arrays
    base_offset = compact_id * 9

    # Solve from body A's perspective
    totalForce_a = wp.transform_vector(transform_a, total_force_in_body_a_frame)
    totalTorque_a = wp.transform_vector(transform_a, torque_a_in_body_a_frame)
    F_T_a = wp.mat33(
        F_accumulator_a[base_offset + 0],
        F_accumulator_a[base_offset + 1],
        F_accumulator_a[base_offset + 2],
        F_accumulator_a[base_offset + 3],
        F_accumulator_a[base_offset + 4],
        F_accumulator_a[base_offset + 5],
        F_accumulator_a[base_offset + 6],
        F_accumulator_a[base_offset + 7],
        F_accumulator_a[base_offset + 8],
    )
    T_T_a = wp.mat33(
        T_accumulator_a[base_offset + 0],
        T_accumulator_a[base_offset + 1],
        T_accumulator_a[base_offset + 2],
        T_accumulator_a[base_offset + 3],
        T_accumulator_a[base_offset + 4],
        T_accumulator_a[base_offset + 5],
        T_accumulator_a[base_offset + 6],
        T_accumulator_a[base_offset + 7],
        T_accumulator_a[base_offset + 8],
    )
    impulse_a = solve_wrench_distribution(totalForce_a, totalTorque_a, F_T_a, T_T_a, r_i_a, invN)

    # Solve from body B's perspective
    totalForce_b = wp.transform_vector(transform_b, total_force_in_body_b_frame)
    totalTorque_b = wp.transform_vector(transform_b, torque_b_in_body_b_frame)
    F_T_b = wp.mat33(
        F_accumulator_b[base_offset + 0],
        F_accumulator_b[base_offset + 1],
        F_accumulator_b[base_offset + 2],
        F_accumulator_b[base_offset + 3],
        F_accumulator_b[base_offset + 4],
        F_accumulator_b[base_offset + 5],
        F_accumulator_b[base_offset + 6],
        F_accumulator_b[base_offset + 7],
        F_accumulator_b[base_offset + 8],
    )
    T_T_b = wp.mat33(
        T_accumulator_b[base_offset + 0],
        T_accumulator_b[base_offset + 1],
        T_accumulator_b[base_offset + 2],
        T_accumulator_b[base_offset + 3],
        T_accumulator_b[base_offset + 4],
        T_accumulator_b[base_offset + 5],
        T_accumulator_b[base_offset + 6],
        T_accumulator_b[base_offset + 7],
        T_accumulator_b[base_offset + 8],
    )
    impulse_b = solve_wrench_distribution(totalForce_b, totalTorque_b, F_T_b, T_T_b, r_i_b, invN)

    # Average the two solutions
    impulse = (impulse_a + impulse_b) * 0.5

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

        # Radix sort requires arrays to be at least 2x count for temporary storage
        self.sorted_body_pairs = wp.zeros(max_num_contacts * 2, dtype=wp.vec2i, device=device)
        self.sorted_to_unsorted_map = wp.zeros(max_num_contacts * 2, dtype=wp.int32, device=device)
        self.unsorted_to_compact_map = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)

        # Buffers for scan-based compaction
        self.is_first_occurrence = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)
        self.prefix_sum = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)

        self.duplicate_free_sorted_body_pairs = wp.zeros(max_num_contacts, dtype=wp.vec2i, device=device)
        self.duplicate_free_sorted_body_pairs_count = wp.zeros(1, dtype=wp.int32, device=device)
        # Wrenches are stored once per duplicate free (=unique) body pair
        # Store as flat float array: 12 floats per wrench (4 vec3s)
        self.wrenches = wp.zeros(max_num_contacts * 12, dtype=float, device=device)

        # Flat arrays: 9 floats per 3x3 matrix (row-major order), separate for body A and B
        self.F_accumulator_a = wp.zeros(max_num_contacts * 9, dtype=float, device=device)
        self.T_accumulator_a = wp.zeros(max_num_contacts * 9, dtype=float, device=device)
        self.F_accumulator_b = wp.zeros(max_num_contacts * 9, dtype=float, device=device)
        self.T_accumulator_b = wp.zeros(max_num_contacts * 9, dtype=float, device=device)
        self.accumulator_count_per_element = wp.zeros(max_num_contacts, dtype=wp.int32, device=device)

    def capture_wrenches(
        self,
        contact_points: wp.array(dtype=wp.vec3, ndim=1),
        contact_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
        contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
        num_contacts: wp.array(dtype=wp.int32, ndim=1),
        all_body_COM_transforms: wp.array(dtype=wp.transform, ndim=1),
        device=None,
    ):
        """
        Capture wrenches from solved contacts for warm starting next timestep.

        Process:
        1. Groups contacts by body pair (sorts and deduplicates)
        2. For each contact: transforms impulse to both body frames (A and B)
        3. Computes torques: τ = (contact_point - COM) x impulse
        4. Accumulates total wrench per body pair: (ΣF, Στ) in local frames

        The stored wrenches are frame-invariant (stored in body frames), allowing
        accurate reconstruction even when bodies rotate between timesteps.

        Args:
            contact_points: Contact points in world space
            contact_body_pairs: Body pair indices for each contact (body A, body B)
            contact_impulses: Solved impulses from this timestep
            num_contacts: Number of active contacts
            all_body_COM_transforms: Current body transforms (positions + orientations)
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
        # Arrays are 2x size for radix sort temporary storage
        body_pairs_as_int64 = wp.array(
            ptr=self.sorted_body_pairs.ptr,
            dtype=wp.int64,
            shape=(self.max_num_contacts * 2,),
            device=device,
            copy=False,
        )

        # Always sort over max_num_contacts (unused entries are masked with max int in prepare kernel)
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

        # Step 3: Compute prefix sum (inclusive scan) to get compact indices
        # Scan over all max_num_contacts entries. Entries beyond num_contacts are already masked with 0
        # in mark_first_occurrences, so they won't affect the scan result.
        wp.utils.array_scan(self.is_first_occurrence, self.prefix_sum, inclusive=True)

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

        # Step 5: Accumulate wrenches (force and torque in local frames)
        self.wrenches.zero_()
        wp.launch(
            kernel=accumulate_wrench,
            dim=self.max_num_contacts,
            inputs=[
                contact_impulses,
                contact_points,
                num_contacts,
                self.unsorted_to_compact_map,
                self.duplicate_free_sorted_body_pairs,
                all_body_COM_transforms,
                self.wrenches,
            ],
            device=device,
        )

    def apply_warm_starting(
        self,
        new_contact_points: wp.array(dtype=wp.vec3, ndim=1),
        new_contact_normals: wp.array(dtype=wp.vec3, ndim=1),
        new_contact_body_pairs: wp.array(dtype=wp.vec2i, ndim=1),
        num_new_contact_points: wp.array(dtype=wp.int32, ndim=1),
        all_body_COM_transforms: wp.array(dtype=wp.transform, ndim=1),
        # Outputs
        contact_impulses: wp.array(dtype=wp.vec3, ndim=1),
        device=None,
    ):
        """
        Apply warm starting to new contacts using stored wrench data.

        Process:
        1. Looks up stored wrench (F_total, τ_total) for each body pair
        2. Builds geometry matrices from new contact points:
           - F_T = Σ[r_i]x (sum of skew-symmetric matrices)
           - T_T = Σ[r_i]x[r_i]xᵀ (sum of cross-product squared)
        3. Solves constrained least squares: find forces {f_i} that minimize
           Σ||f_i||² (smallest forces = minimum energy) subject to matching
           the stored wrench: Σf_i = F_total and Σ(r_i x f_i) = τ_total.
           Solution via Lagrange multipliers reduces to a 3x3 Schur complement
           system for the torque multiplier λ_t, then back-solves for λ_f.
        4. Reconstructs impulse at each contact: f_i = λ_f - r_i x λ_t
        5. Solves from both body perspectives and averages for robustness

        Handles variable contact counts gracefully: contact number can differ
        between capture and reconstruction with minimal force error.

        Args:
            new_contact_points: Contact points in world space
            new_contact_normals: Contact normals (pointing from A to B)
            new_contact_body_pairs: Body pair indices for each contact
            num_new_contact_points: Number of active contacts
            all_body_COM_transforms: Current body transforms (positions + orientations)
            contact_impulses: Output array for reconstructed impulses
            device: Device to launch on (None for default)
        """
        if device is None:
            device = self.device

        # Clear accumulators
        self.F_accumulator_a.zero_()
        self.T_accumulator_a.zero_()
        self.F_accumulator_b.zero_()
        self.T_accumulator_b.zero_()
        self.accumulator_count_per_element.zero_()

        # Step 1: Accumulate F_T and T_T matrices for both bodies
        wp.launch(
            kernel=accumulate_F_and_T,
            dim=self.max_num_contacts,
            inputs=[
                self.duplicate_free_sorted_body_pairs,
                self.duplicate_free_sorted_body_pairs_count,
                new_contact_points,
                new_contact_body_pairs,
                num_new_contact_points,
                all_body_COM_transforms,
                self.F_accumulator_a,
                self.T_accumulator_a,
                self.F_accumulator_b,
                self.T_accumulator_b,
                self.accumulator_count_per_element,
            ],
            device=device,
        )

        # Step 2: Apply warm starting to reconstruct impulses (solve twice and average)
        wp.launch(
            kernel=apply_warm_starting,
            dim=self.max_num_contacts,
            inputs=[
                self.duplicate_free_sorted_body_pairs,
                self.duplicate_free_sorted_body_pairs_count,
                new_contact_points,
                new_contact_normals,
                new_contact_body_pairs,
                num_new_contact_points,
                all_body_COM_transforms,
                self.F_accumulator_a,
                self.T_accumulator_a,
                self.F_accumulator_b,
                self.T_accumulator_b,
                self.accumulator_count_per_element,
                self.wrenches,
                contact_impulses,
            ],
            device=device,
        )
