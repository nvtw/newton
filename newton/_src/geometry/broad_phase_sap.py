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

from typing import Any

import warp as wp

from .broad_phase_common import (
    binary_search,
    check_aabb_overlap,
    test_environment_and_group_pair,
    write_pair,
)

wp.set_module_options({"enable_backward": False})

snippet = """
//unsigned int i = __float_as_uint(f);
unsigned int i = *(unsigned int*)&f;
unsigned int mask = (unsigned int)(-(int)(i >> 31)) | 0x80000000;
unsigned int result = i ^ mask;
return (int64_t)result;
"""


@wp.func_native(snippet)
def _float_to_sortable_uint(f: float) -> wp.int64: ...


@wp.func
def _build_sort_key(collision_group: int, value: float) -> wp.int64:
    if collision_group <= 0:
        return wp.int64(2147483647) * wp.int64(2147483647)
    # strange formula to work around bugs when having very large values in python
    # The multiplication is equivalent to a shift by 32 bits
    upper = wp.int64(collision_group) * (wp.int64(2147483647) + wp.int64(2147483647) + wp.int64(2))
    lower = _float_to_sortable_uint(value)
    sum = upper + lower
    return sum


@wp.kernel
def _flag_group_id_kernel(
    geom_collision_groups: wp.array(dtype=int, ndim=1),
    marker: wp.array(dtype=int, ndim=1),
    negative_group_counter: wp.array(dtype=int, ndim=1),
    negative_group_indices: wp.array(dtype=int, ndim=1),
):
    id = wp.tid()

    group = geom_collision_groups[id]
    if group > 0 and marker[group] == 0:
        marker[group] = 1
    if group < 0:
        index = wp.atomic_add(negative_group_counter, 0, 1)
        negative_group_indices[index] = id


@wp.kernel
def _write_unique_group_id_kernel(
    marker: wp.array(dtype=int, ndim=1),
    unique_group_ids: wp.array(dtype=int, ndim=1),
    unique_group_id_counter: wp.array(dtype=int, ndim=1),
):
    id = wp.tid()
    if marker[id] == 1:
        index = wp.atomic_add(unique_group_id_counter, 0, 1)
        unique_group_ids[index] = id


@wp.func
def _sap_project_aabb(
    elementid: int,
    direction: wp.vec3,  # Must be normalized
    geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
) -> wp.vec2:
    lower = geom_bounding_box_lower[elementid]
    upper = geom_bounding_box_upper[elementid]
    cutoff = geom_cutoff[elementid]

    half_size = 0.5 * (upper - lower)
    half_size = wp.vec3(half_size[0] + cutoff, half_size[1] + cutoff, half_size[2] + cutoff)
    radius = wp.dot(direction, half_size)
    center = wp.dot(direction, 0.5 * (lower + upper))
    return wp.vec2(center - radius, center + radius)

@wp.func
def _sap_project_sphere(
    elementid: int,
    direction: wp.vec3,  # Must be normalized
    geom_center: wp.array(dtype=wp.vec3, ndim=1),
    geom_radius: wp.array(dtype=float, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (add to radius)
) -> wp.vec2:
    center = geom_center[elementid]
    radius = geom_radius[elementid]
    cutoff = geom_cutoff[elementid]

    # Project center onto direction
    center_proj = wp.dot(direction, center)
    # Expanded radius includes cutoff
    expanded_radius = radius + cutoff

    return wp.vec2(center_proj - expanded_radius, center_proj + expanded_radius)


def _make_sap_project_func(project_func):
    """Factory to create SAP projection functions for different bounding volume types."""

    @wp.func
    def _sap_project_func_template(
        id: int,
        direction: wp.vec3,
        geom_data1: wp.array(dtype=wp.vec3, ndim=1),
        geom_data2: Any,
        geom_cutoff: wp.array(dtype=float, ndim=1),
        geom_collision_groups: wp.array(dtype=int, ndim=1),
        unique_group_id_counter: wp.array(dtype=int, ndim=1),
        unique_group_ids: wp.array(dtype=int, ndim=1),
        negative_group_counter: wp.array(dtype=int, ndim=1),
        negative_group_indices: wp.array(dtype=int, ndim=1),
        num_boxes: int,
        sap_projection_lower_out: wp.array(dtype=wp.int64, ndim=1),
        sap_projection_upper_out: wp.array(dtype=wp.int64, ndim=1),
        sap_sort_index_out: wp.array(dtype=int, ndim=1),
    ):
        num_negative_groups = negative_group_counter[0]
        num_work_packages = num_boxes + num_negative_groups * unique_group_id_counter[0]

        if id < num_work_packages:
            box_id = 0
            group_id = 0
            if id < num_boxes:
                box_id = id
                group_id = geom_collision_groups[box_id]
            else:
                id2 = id - num_boxes
                box_id = negative_group_indices[id2 % num_negative_groups]
                group_id = unique_group_ids[id2 // num_negative_groups]

            if group_id <= 0:
                group_id = -1

            range = project_func(box_id, direction, geom_data1, geom_data2, geom_cutoff)

            sap_projection_lower_out[id] = _build_sort_key(group_id, range[0])
            sap_projection_upper_out[id] = _build_sort_key(group_id, range[1])

            if group_id > 0:
                sap_sort_index_out[id] = id
            else:
                sap_sort_index_out[id] = -1
        else:
            sap_projection_lower_out[id] = wp.int64(2147483647) * wp.int64(2147483647)
            sap_projection_upper_out[id] = wp.int64(2147483647) * wp.int64(2147483647)
            sap_sort_index_out[id] = -1

    return _sap_project_func_template


@wp.kernel
def _sap_project_kernel(
    direction: wp.vec3,
    geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),
    geom_collision_groups: wp.array(dtype=int, ndim=1),
    unique_group_id_counter: wp.array(dtype=int, ndim=1),
    unique_group_ids: wp.array(dtype=int, ndim=1),
    negative_group_counter: wp.array(dtype=int, ndim=1),
    negative_group_indices: wp.array(dtype=int, ndim=1),
    num_boxes: int,
    sap_projection_lower_out: wp.array(dtype=wp.int64, ndim=1),
    sap_projection_upper_out: wp.array(dtype=wp.int64, ndim=1),
    sap_sort_index_out: wp.array(dtype=int, ndim=1),
):
    id = wp.tid()
    wp.static(_make_sap_project_func(_sap_project_aabb))(
        id,
        direction,
        geom_bounding_box_lower,
        geom_bounding_box_upper,
        geom_cutoff,
        geom_collision_groups,
        unique_group_id_counter,
        unique_group_ids,
        negative_group_counter,
        negative_group_indices,
        num_boxes,
        sap_projection_lower_out,
        sap_projection_upper_out,
        sap_sort_index_out,
    )


@wp.kernel
def _sap_project_bounding_spheres_kernel(
    direction: wp.vec3,
    geom_center: wp.array(dtype=wp.vec3, ndim=1),
    geom_radius: wp.array(dtype=float, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),
    geom_collision_groups: wp.array(dtype=int, ndim=1),
    unique_group_id_counter: wp.array(dtype=int, ndim=1),
    unique_group_ids: wp.array(dtype=int, ndim=1),
    negative_group_counter: wp.array(dtype=int, ndim=1),
    negative_group_indices: wp.array(dtype=int, ndim=1),
    num_boxes: int,
    sap_projection_lower_out: wp.array(dtype=wp.int64, ndim=1),
    sap_projection_upper_out: wp.array(dtype=wp.int64, ndim=1),
    sap_sort_index_out: wp.array(dtype=int, ndim=1),
):
    id = wp.tid()
    wp.static(_make_sap_project_func(_sap_project_sphere))(
        id,
        direction,
        geom_center,
        geom_radius,
        geom_cutoff,
        geom_collision_groups,
        unique_group_id_counter,
        unique_group_ids,
        negative_group_counter,
        negative_group_indices,
        num_boxes,
        sap_projection_lower_out,
        sap_projection_upper_out,
        sap_sort_index_out,
    )


@wp.func
def _sap_range_func(
    elementid: int,
    num_elements: int,
    sap_projection_lower_in: wp.array(dtype=wp.int64, ndim=1),
    sap_projection_upper_in: wp.array(dtype=wp.int64, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),
):
    # current bounding geom
    idx = sap_sort_index_in[elementid]

    upper = sap_projection_upper_in[idx]

    limit = binary_search(sap_projection_lower_in, upper, elementid + 1, num_elements)
    limit = wp.min(num_elements, limit)

    # range of geoms for the sweep and prune process
    return limit - elementid - 1


@wp.kernel
def _sap_range_kernel(
    num_boxes: int,
    negative_group_counter: wp.array(dtype=int, ndim=1),
    unique_group_id_counter: wp.array(dtype=int, ndim=1),
    sap_projection_lower_in: wp.array(dtype=wp.int64, ndim=1),
    sap_projection_upper_in: wp.array(dtype=wp.int64, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),
    sap_range_out: wp.array(dtype=int, ndim=1),
):
    num_negative_groups = negative_group_counter[0]
    num_work_packages = num_boxes + num_negative_groups * unique_group_id_counter[0]

    elementid = wp.tid()
    if elementid >= num_work_packages:
        sap_range_out[elementid] = 0
        return
    count = _sap_range_func(
        elementid, num_work_packages, sap_projection_lower_in, sap_projection_upper_in, sap_sort_index_in
    )
    sap_range_out[elementid] = count


@wp.func
def _check_sphere_overlap(
    center1: wp.vec3,
    radius1: float,
    cutoff1: float,
    center2: wp.vec3,
    radius2: float,
    cutoff2: float,
) -> bool:
    # Check if two spheres overlap considering their cutoffs
    expanded_radius1 = radius1 + cutoff1
    expanded_radius2 = radius2 + cutoff2
    sum_radii = expanded_radius1 + expanded_radius2

    dist_sq = wp.length_sq(center1 - center2)
    return dist_sq <= sum_radii * sum_radii


def _make_process_single_sap_pair_func(overlap_check_func):
    """Factory to create SAP pair processing functions for different bounding volume types."""

    @wp.func
    def _process_single_sap_pair_template(
        pair: wp.vec2i,
        geom_data1: wp.array(dtype=wp.vec3, ndim=1),
        geom_data2: Any,
        geom_cutoff: wp.array(dtype=float, ndim=1),
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        max_candidate_pair: int,
    ):
        geom1 = pair[0]
        geom2 = pair[1]

        if overlap_check_func(geom_data1, geom_data2, geom_cutoff, geom1, geom2):
            write_pair(
                pair,
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            )

    return _process_single_sap_pair_template


@wp.func
def _check_aabb_overlap_indexed(
    geom_lower: wp.array(dtype=wp.vec3, ndim=1),
    geom_upper: wp.array(dtype=wp.vec3, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),
    geom1: int,
    geom2: int,
) -> bool:
    return check_aabb_overlap(
        geom_lower[geom1],
        geom_upper[geom1],
        geom_cutoff[geom1],
        geom_lower[geom2],
        geom_upper[geom2],
        geom_cutoff[geom2],
    )


@wp.func
def _check_sphere_overlap_indexed(
    geom_center: wp.array(dtype=wp.vec3, ndim=1),
    geom_radius: wp.array(dtype=float, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),
    geom1: int,
    geom2: int,
) -> bool:
    return _check_sphere_overlap(
        geom_center[geom1],
        geom_radius[geom1],
        geom_cutoff[geom1],
        geom_center[geom2],
        geom_radius[geom2],
        geom_cutoff[geom2],
    )


def _make_sap_broadphase_func(process_pair_func):
    """Factory to create SAP broadphase functions for different bounding volume types."""

    @wp.func
    def _sap_broadphase_func_template(
        geomid: int,
        geom_data1: wp.array(dtype=wp.vec3, ndim=1),
        geom_data2: Any,
        num_boxes: int,
        upper_count: int,
        negative_group_indices: wp.array(dtype=int, ndim=1),
        num_negative_group_counter: wp.array(dtype=int, ndim=1),
        unique_group_id_counter: wp.array(dtype=int, ndim=1),
        collision_group: wp.array(dtype=int, ndim=1),
        shape_group: wp.array(dtype=int, ndim=1),
        sap_sort_index_in: wp.array(dtype=int, ndim=1),
        sap_cumulative_sum_in: wp.array(dtype=int, ndim=1),
        geom_cutoff: wp.array(dtype=float, ndim=1),
        nsweep_in: int,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        max_candidate_pair: int,
    ):
        nworkpackages = sap_cumulative_sum_in[upper_count - 1]

        num_negative_group_elements = num_negative_group_counter[0]
        upper_bound = num_boxes + num_negative_group_elements * unique_group_id_counter[0]

        while geomid < nworkpackages:
            # binary search to find current and next geom pair indices
            i = binary_search(sap_cumulative_sum_in, geomid, 0, upper_bound)
            j = i + geomid + 1

            if i > 0:
                j -= sap_cumulative_sum_in[i - 1]

            # get geom indices and swap if necessary
            geom1 = sap_sort_index_in[i]
            geom2 = sap_sort_index_in[j]

            if geom1 >= num_boxes:
                geom1 = negative_group_indices[(geom1 - num_boxes) % num_negative_group_elements]

            if geom2 >= num_boxes:
                geom2 = negative_group_indices[(geom2 - num_boxes) % num_negative_group_elements]

            # Get collision and environment groups
            col_group1 = collision_group[geom1]
            col_group2 = collision_group[geom2]
            env_group1 = shape_group[geom1]
            env_group2 = shape_group[geom2]

            # Check both environment and collision groups
            if not test_environment_and_group_pair(env_group1, env_group2, col_group1, col_group2):
                geomid += nsweep_in
                continue

            if geom1 > geom2:
                tmp = geom1
                geom1 = geom2
                geom2 = tmp

            process_pair_func(
                wp.vec2i(geom1, geom2),
                geom_data1,
                geom_data2,
                geom_cutoff,
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            )

            geomid += nsweep_in

    return _sap_broadphase_func_template


@wp.kernel
def _sap_broadphase_kernel(
    geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    num_boxes: int,
    upper_count: int,
    negative_group_indices: wp.array(dtype=int, ndim=1),
    num_negative_group_counter: wp.array(dtype=int, ndim=1),
    unique_group_id_counter: wp.array(dtype=int, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_group: wp.array(dtype=int, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),
    sap_cumulative_sum_in: wp.array(dtype=int, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),
    nsweep_in: int,
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    geomid = wp.tid()
    wp.static(_make_sap_broadphase_func(_make_process_single_sap_pair_func(_check_aabb_overlap_indexed)))(
        geomid,
        geom_bounding_box_lower,
        geom_bounding_box_upper,
        num_boxes,
        upper_count,
        negative_group_indices,
        num_negative_group_counter,
        unique_group_id_counter,
        collision_group,
        shape_group,
        sap_sort_index_in,
        sap_cumulative_sum_in,
        geom_cutoff,
        nsweep_in,
        candidate_pair,
        num_candidate_pair,
        max_candidate_pair,
    )


@wp.kernel
def _sap_broadphase_spheres_kernel(
    geom_center: wp.array(dtype=wp.vec3, ndim=1),
    geom_radius: wp.array(dtype=float, ndim=1),
    num_boxes: int,
    upper_count: int,
    negative_group_indices: wp.array(dtype=int, ndim=1),
    num_negative_group_counter: wp.array(dtype=int, ndim=1),
    unique_group_id_counter: wp.array(dtype=int, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_group: wp.array(dtype=int, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),
    sap_cumulative_sum_in: wp.array(dtype=int, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),
    nsweep_in: int,
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    geomid = wp.tid()
    wp.static(_make_sap_broadphase_func(_make_process_single_sap_pair_func(_check_sphere_overlap_indexed)))(
        geomid,
        geom_center,
        geom_radius,
        num_boxes,
        upper_count,
        negative_group_indices,
        num_negative_group_counter,
        unique_group_id_counter,
        collision_group,
        shape_group,
        sap_sort_index_in,
        sap_cumulative_sum_in,
        geom_cutoff,
        nsweep_in,
        candidate_pair,
        num_candidate_pair,
        max_candidate_pair,
    )


class BroadPhaseSAP:
    """Sweep and Prune (SAP) broad phase collision detection.

    This class implements the sweep and prune algorithm for broad phase collision detection.
    It efficiently finds potentially colliding pairs of objects by sorting their bounding box
    projections along a fixed axis and checking for overlaps.
    """

    def __init__(
        self,
        max_broad_phase_elements: int,
        max_num_distinct_positive_groups: int,
        max_num_negative_group_members: int,
        sweep_thread_count_multiplier: int = 5,
    ):
        """Initialize arrays for sweep and prune broad phase collision detection.

        Args:
            max_broad_phase_elements: Maximum number of elements to process
            max_num_distinct_positive_groups: Maximum number of unique positive collision groups
            max_num_negative_group_members: Maximum number of elements with negative groups
            sweep_thread_count_multiplier: Multiplier for number of threads used in sweep phase
        """
        self.max_broad_phase_elements = max_broad_phase_elements
        self.max_num_distinct_positive_groups = max_num_distinct_positive_groups
        self.max_num_negative_group_members = max_num_negative_group_members
        self.sweep_thread_count_multiplier = sweep_thread_count_multiplier

        upper_bound = max_broad_phase_elements + max_num_negative_group_members * max_num_distinct_positive_groups

        # Temp memory
        self.negative_group_indices = wp.zeros(n=max_broad_phase_elements, dtype=int)
        self.negative_group_counter = wp.zeros(n=1, dtype=int)
        self.unique_group_ids = wp.zeros(n=max_broad_phase_elements, dtype=int)
        self.unique_group_id_counter = wp.zeros(n=1, dtype=int)

        # Factor 2 in some arrays is required for radix sort
        self.sap_projection_lower = wp.zeros(n=2 * upper_bound, dtype=wp.int64)
        self.sap_projection_upper = wp.zeros(n=upper_bound, dtype=wp.int64)
        self.sap_sort_index = wp.zeros(n=2 * upper_bound, dtype=int)
        self.sap_range = wp.zeros(n=upper_bound, dtype=int)
        self.sap_cumulative_sum = wp.zeros(n=upper_bound, dtype=int)

    def _launch_internal(
        self,
        geom_data1: wp.array(dtype=wp.vec3, ndim=1),
        geom_data2: Any,
        geom_cutoffs: wp.array(dtype=float, ndim=1),
        geom_collision_group: wp.array(dtype=int, ndim=1),
        geom_shape_group: wp.array(dtype=int, ndim=1),
        geom_count: int,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        project_kernel,
        broadphase_kernel,
    ):
        """Internal method that implements the SAP algorithm for any bounding volume type.

        Args:
            geom_data1: First geometry data array (AABB lower OR sphere center)
            geom_data2: Second geometry data array (AABB upper OR sphere radius)
            geom_cutoffs: Cutoff distances
            geom_collision_group: Collision group IDs
            geom_shape_group: Environment group IDs
            geom_count: Number of geometries
            candidate_pair: Output array for pairs
            num_candidate_pair: Output count
            project_kernel: Kernel for projecting bounding volumes
            broadphase_kernel: Kernel for broadphase sweep
        """
        # TODO: Choose an optimal direction
        # random fixed direction
        direction = wp.vec3(0.5935, 0.7790, 0.1235)
        direction = wp.normalize(direction)

        max_candidate_pair = candidate_pair.shape[0]
        num_threads = geom_count + self.max_num_negative_group_members * self.max_num_distinct_positive_groups

        # Temporarily use sap_cumulative_sum since it's not used until later in the method
        self.sap_cumulative_sum.zero_()
        num_candidate_pair.zero_()

        wp.launch(
            kernel=_flag_group_id_kernel,
            dim=geom_count,
            inputs=[
                geom_collision_group,
                self.sap_cumulative_sum,
                self.negative_group_counter,
                self.negative_group_indices,
            ],
        )

        wp.launch(
            kernel=_write_unique_group_id_kernel,
            dim=geom_count,
            inputs=[self.sap_cumulative_sum],
            outputs=[
                self.unique_group_ids,
                self.unique_group_id_counter,
            ],
        )

        wp.launch(
            kernel=project_kernel,
            dim=num_threads,
            inputs=[
                direction,
                geom_data1,
                geom_data2,
                geom_cutoffs,
                geom_collision_group,
                self.unique_group_id_counter,
                self.unique_group_ids,
                self.negative_group_counter,
                self.negative_group_indices,
                geom_count,
            ],
            outputs=[
                self.sap_projection_lower,
                self.sap_projection_upper,
                self.sap_sort_index,
            ],
        )

        wp.utils.radix_sort_pairs(
            self.sap_projection_lower,
            self.sap_sort_index,
            num_threads,
        )

        wp.launch(
            kernel=_sap_range_kernel,
            dim=num_threads,
            inputs=[
                geom_count,
                self.negative_group_counter,
                self.unique_group_id_counter,
                self.sap_projection_lower,
                self.sap_projection_upper,
                self.sap_sort_index,
                self.sap_range,
            ],
        )

        wp.utils.array_scan(self.sap_range.reshape(-1), self.sap_cumulative_sum, True)

        # estimate number of overlap checks
        nsweep_in = self.sweep_thread_count_multiplier * num_threads

        wp.launch(
            kernel=broadphase_kernel,
            dim=nsweep_in,
            inputs=[
                geom_data1,
                geom_data2,
                geom_count,
                num_threads,
                self.negative_group_indices,
                self.negative_group_counter,
                self.unique_group_id_counter,
                geom_collision_group,
                geom_shape_group,
                self.sap_sort_index,
                self.sap_cumulative_sum,
                geom_cutoffs,
                nsweep_in,
            ],
            outputs=[
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            ],
        )

    def launch_bounding_spheres(
        self,
        geom_center: wp.array(dtype=wp.vec3, ndim=1),  # Centers of geometry bounding spheres
        geom_radius: wp.array(dtype=float, ndim=1),  # Radii of geometry bounding spheres
        geom_cutoffs: wp.array(dtype=float, ndim=1),  # Cutoff distance per geometry sphere
        geom_collision_group: wp.array(dtype=int, ndim=1),  # Collision group ID per sphere
        geom_shape_group: wp.array(dtype=int, ndim=1),  # Environment group ID per sphere
        geom_count: int,  # Number of active bounding spheres
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping geometry pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
    ):
        """Launch the sweep and prune broad phase collision detection for bounding spheres.

        This method performs collision detection between geometries using a sweep and prune algorithm
        along a fixed axis. It projects the bounding spheres onto the sweep axis, sorts them, and checks
        for overlaps between nearby spheres. The method also handles collision filtering based on both
        environment groups and collision groups.

        Args:
            geom_center: Array of center positions for each geometry's bounding sphere
            geom_radius: Array of radii for each geometry's bounding sphere
            geom_cutoffs: Array of cutoff distances for each geometry
            geom_collision_group: Array of collision group IDs for each geometry. Positive values indicate
                groups that only collide with themselves (and with negative groups). Negative values indicate
                groups that collide with everything except their negative counterpart. Zero indicates no collisions.
            geom_shape_group: Array of environment group IDs for each geometry. Group -1 indicates global entities
                that collide with all environments. Groups 0, 1, 2, ... indicate environment-specific entities.
            geom_count: Number of active bounding spheres to check
            candidate_pair: Output array to store overlapping geometry pairs
            num_candidate_pair: Output array to store number of overlapping pairs found

        The method will populate candidate_pair with the indices of geometry pairs whose bounding spheres overlap
        when expanded by their cutoff distances, whose collision groups allow interaction, and whose environment
        groups are compatible (same environment or at least one is global). The number of pairs found will be
        written to num_candidate_pair[0].
        """
        self._launch_internal(
            geom_center,
            geom_radius,
            geom_cutoffs,
            geom_collision_group,
            geom_shape_group,
            geom_count,
            candidate_pair,
            num_candidate_pair,
            _sap_project_bounding_spheres_kernel,
            _sap_broadphase_spheres_kernel,
        )


    def launch(
        self,
        geom_lower: wp.array(dtype=wp.vec3, ndim=1),  # Lower bounds of geometry bounding boxes
        geom_upper: wp.array(dtype=wp.vec3, ndim=1),  # Upper bounds of geometry bounding boxes
        geom_cutoffs: wp.array(dtype=float, ndim=1),  # Cutoff distance per geometry box
        geom_collision_group: wp.array(dtype=int, ndim=1),  # Collision group ID per box
        geom_shape_group: wp.array(dtype=int, ndim=1),  # Environment group ID per box
        geom_count: int,  # Number of active bounding boxes
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping geometry pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
    ):
        """Launch the sweep and prune broad phase collision detection.

        This method performs collision detection between geometries using a sweep and prune algorithm along a fixed axis.
        It projects the bounding boxes onto the sweep axis, sorts them, and checks for overlaps between nearby boxes.
        The method also handles collision filtering based on both environment groups and collision groups.

        Args:
            geom_lower: Array of lower bounds for each geometry's AABB
            geom_upper: Array of upper bounds for each geometry's AABB
            geom_cutoffs: Array of cutoff distances for each geometry
            geom_collision_groups: Array of collision group IDs for each geometry. Positive values indicate
                groups that only collide with themselves (and with negative groups). Negative values indicate
                groups that collide with everything except their negative counterpart. Zero indicates no collisions.
            geom_shape_groups: Array of environment group IDs for each geometry. Group -1 indicates global entities
                that collide with all environments. Groups 0, 1, 2, ... indicate environment-specific entities.
                Can be None if environment groups are not used.
            geom_count: Number of active bounding boxes to check
            candidate_pair: Output array to store overlapping geometry pairs
            num_candidate_pair: Output array to store number of overlapping pairs found

        The method will populate candidate_pair with the indices of geometry pairs whose AABBs overlap
        when expanded by their cutoff distances, whose collision groups allow interaction, and whose environment groups
        are compatible (same environment or at least one is global). The number of pairs found will be written to
        num_candidate_pair[0].
        """
        self._launch_internal(
            geom_lower,
            geom_upper,
            geom_cutoffs,
            geom_collision_group,
            geom_shape_group,
            geom_count,
            candidate_pair,
            num_candidate_pair,
            _sap_project_kernel,
            _sap_broadphase_kernel,
        )
