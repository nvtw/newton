# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NxN (all-pairs) broad phase collision detection.

Provides O(N^2) broad phase using AABB overlap tests. Simple and effective
for small scenes (<100 shapes). For larger scenes, use SAP broad phase.

See Also:
    :class:`BroadPhaseSAP` in ``broad_phase_sap.py`` for O(N log N) performance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .broad_phase_common import (
    EmptyFilterData,
    check_aabb_overlap,
    is_pair_excluded,
    keep_all_filter,
    precompute_world_map,
    test_world_and_group_pair,
    write_pair,
)


def create_nxn_broadphase_precomputed_pairs_kernel(filter_func: Any, filter_data_type: Any):
    """Build an explicit-pairs NxN broad-phase kernel bound to ``filter_func``.

    The returned kernel runs ``filter_func(pair, filter_data)`` after the
    AABB overlap test and skips pairs for which it returns ``0``.  Using
    :data:`broad_phase_common.keep_all_filter` /
    :class:`broad_phase_common.EmptyFilterData` produces a kernel that is
    bit-equivalent to the pre-callback path (Warp folds away the constant
    return).
    """

    _module = f"nxn_broadphase_explicit_{filter_func.__name__}_{filter_data_type.__name__}"

    @wp.kernel(enable_backward=False, module=_module)
    def kernel(
        shape_bounding_box_lower: wp.array[wp.vec3],
        shape_bounding_box_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float],
        nxn_shape_pair: wp.array[wp.vec2i],
        filter_data: Any,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: int,
    ):
        elementid = wp.tid()

        pair = nxn_shape_pair[elementid]
        shape1 = pair[0]
        shape2 = pair[1]

        gap1 = 0.0
        gap2 = 0.0
        if shape_gap.shape[0] > 0:
            gap1 = shape_gap[shape1]
            gap2 = shape_gap[shape2]

        if check_aabb_overlap(
            shape_bounding_box_lower[shape1],
            shape_bounding_box_upper[shape1],
            gap1,
            shape_bounding_box_lower[shape2],
            shape_bounding_box_upper[shape2],
            gap2,
        ):
            if filter_func(pair, filter_data) == wp.int32(0):
                return
            write_pair(
                pair,
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            )

    return kernel


_nxn_broadphase_precomputed_pairs = create_nxn_broadphase_precomputed_pairs_kernel(keep_all_filter, EmptyFilterData)


@wp.func
def _get_lower_triangular_indices(index: int, matrix_size: int) -> tuple[int, int]:
    total = (matrix_size * (matrix_size - 1)) >> 1
    if index >= total:
        # In Warp, we can't throw, so return an invalid pair
        return -1, -1

    low = int(0)
    high = matrix_size - 1
    while low < high:
        mid = (low + high) >> 1
        count = (mid * (2 * matrix_size - mid - 1)) >> 1
        if count <= index:
            low = mid + 1
        else:
            high = mid
    r = low - 1
    f = (r * (2 * matrix_size - r - 1)) >> 1
    c = (index - f) + r + 1
    return r, c


@wp.func
def _find_world_and_local_id(
    tid: int,
    world_cumsum_lower_tri: wp.array[int],
):
    """Binary search to find world ID and local ID from thread ID.

    Args:
        tid: Global thread ID
        world_cumsum_lower_tri: Cumulative sum of lower triangular elements per world

    Returns:
        tuple: (world_id, local_id) - World ID and local index within that world
    """
    world_count = world_cumsum_lower_tri.shape[0]

    # Find world_id using binary search
    # Declare as dynamic variables for loop mutation
    low = int(0)
    high = int(world_count - 1)
    world_id = int(0)

    while low <= high:
        mid = (low + high) >> 1
        if tid < world_cumsum_lower_tri[mid]:
            high = mid - 1
            world_id = mid
        else:
            low = mid + 1

    # Calculate local index within this world
    local_id = tid
    if world_id > 0:
        local_id = tid - world_cumsum_lower_tri[world_id - 1]

    return world_id, local_id


def create_nxn_broadphase_kernel(filter_func: Any, filter_data_type: Any):
    """Build the NxN sweep broad-phase kernel bound to ``filter_func``.

    See :func:`create_nxn_broadphase_precomputed_pairs_kernel` for the
    semantics of the filter callback.
    """

    _module = f"nxn_broadphase_{filter_func.__name__}_{filter_data_type.__name__}"

    @wp.kernel(enable_backward=False, module=_module)
    def kernel(
        shape_bounding_box_lower: wp.array[wp.vec3],
        shape_bounding_box_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float],
        collision_group: wp.array[int],
        shape_world: wp.array[int],
        world_cumsum_lower_tri: wp.array[int],
        world_slice_ends: wp.array[int],
        world_index_map: wp.array[int],
        num_regular_worlds: int,
        filter_pairs: wp.array[wp.vec2i],
        num_filter_pairs: int,
        filter_data: Any,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: int,
    ):
        tid = wp.tid()

        world_id, local_id = _find_world_and_local_id(tid, world_cumsum_lower_tri)

        world_slice_start = 0
        if world_id > 0:
            world_slice_start = world_slice_ends[world_id - 1]
        world_slice_end = world_slice_ends[world_id]

        num_shapes_in_world = world_slice_end - world_slice_start

        local_shape1, local_shape2 = _get_lower_triangular_indices(local_id, num_shapes_in_world)

        shape1_tmp = world_index_map[world_slice_start + local_shape1]
        shape2_tmp = world_index_map[world_slice_start + local_shape2]

        shape1 = wp.min(shape1_tmp, shape2_tmp)
        shape2 = wp.max(shape1_tmp, shape2_tmp)

        world1 = shape_world[shape1]
        world2 = shape_world[shape2]
        collision_group1 = collision_group[shape1]
        collision_group2 = collision_group[shape2]

        is_dedicated_minus_one_segment = world_id >= num_regular_worlds
        if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:
            return

        if not test_world_and_group_pair(world1, world2, collision_group1, collision_group2):
            return

        gap1 = 0.0
        gap2 = 0.0
        if shape_gap.shape[0] > 0:
            gap1 = shape_gap[shape1]
            gap2 = shape_gap[shape2]

        if check_aabb_overlap(
            shape_bounding_box_lower[shape1],
            shape_bounding_box_upper[shape1],
            gap1,
            shape_bounding_box_lower[shape2],
            shape_bounding_box_upper[shape2],
            gap2,
        ):
            pair = wp.vec2i(shape1, shape2)
            if num_filter_pairs > 0 and is_pair_excluded(pair, filter_pairs, num_filter_pairs):
                return
            if filter_func(pair, filter_data) == wp.int32(0):
                return
            write_pair(
                pair,
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            )

    return kernel


_nxn_broadphase_kernel = create_nxn_broadphase_kernel(keep_all_filter, EmptyFilterData)


class BroadPhaseAllPairs:
    """A broad phase collision detection class that performs N x N collision checks between all geometry pairs.

    This class performs collision detection between all possible pairs of geometries by checking for
    axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
    checking each pair twice.

    The collision checks take into account per-geometry cutoff distances and collision groups. Two geometries
    will only be considered as a candidate pair if:
    1. Their AABBs overlap when expanded by their cutoff distances
    2. Their collision groups allow interaction (determined by test_group_pair())

    The class outputs an array of candidate collision pairs that need more detailed narrow phase collision
    checking.
    """

    def __init__(
        self,
        shape_world: wp.array[wp.int32] | np.ndarray,
        shape_flags: wp.array[wp.int32] | np.ndarray | None = None,
        device: Devicelike | None = None,
        filter_func: Any | None = None,
        filter_data_type: Any | None = None,
    ) -> None:
        """Initialize the broad phase with world ID information.

        Args:
            shape_world: Array of world IDs (numpy or warp array).
                Positive/zero values represent distinct worlds, negative values represent
                shared entities that belong to all worlds.
            shape_flags: Optional array of shape flags (numpy or warp array). If provided,
                only shapes with the COLLIDE_SHAPES flag will be included in collision checks.
                This efficiently filters out visual-only shapes.
            device: Device to store the precomputed arrays on. If None, uses CPU for numpy
                arrays or the device of the input warp array.
            filter_func: Optional ``@wp.func`` invoked as
                ``filter_func(pair: wp.vec2i, data: filter_data_type) -> wp.int32``
                after the AABB overlap test. Returning ``0`` drops the pair.
                When ``None``, the default no-op kernel is reused.
            filter_data_type: ``wp.struct`` type for the ``filter_data`` argument
                forwarded to :meth:`launch`.  Required when ``filter_func`` is set.
        """
        if (filter_func is None) != (filter_data_type is None):
            raise ValueError("filter_func and filter_data_type must be provided together")

        if filter_func is None:
            self._kernel = _nxn_broadphase_kernel
            self._has_custom_filter = False
        else:
            self._kernel = create_nxn_broadphase_kernel(filter_func, filter_data_type)
            self._has_custom_filter = True
        self._empty_filter_data = EmptyFilterData()

        # Convert to numpy if it's a warp array
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
            if device is None:
                device = "cpu"

        # Convert shape_flags to numpy if provided
        shape_flags_np = None
        if shape_flags is not None:
            if isinstance(shape_flags, wp.array):
                shape_flags_np = shape_flags.numpy()
            else:
                shape_flags_np = shape_flags

        # Precompute the world map (filters out non-colliding shapes if flags provided)
        index_map_np, slice_ends_np = precompute_world_map(shape_world_np, shape_flags_np)

        # Calculate number of regular worlds (excluding dedicated -1 segment at end)
        # Must be derived from filtered slices since precompute_world_map applies flags
        # slice_ends_np has length (num_filtered_worlds + 1), where +1 is the dedicated -1 segment
        num_regular_worlds = max(0, len(slice_ends_np) - 1)

        # Calculate cumulative sum of lower triangular elements per world
        # For each world, compute n*(n-1)/2 where n is the number of geometries in that world
        world_count = len(slice_ends_np)
        world_cumsum_lower_tri_np = np.zeros(world_count, dtype=np.int32)

        start_idx = 0
        cumsum = 0
        for world_idx in range(world_count):
            end_idx = slice_ends_np[world_idx]
            # Number of geometries in this world (including shared geometries)
            num_geoms_in_world = end_idx - start_idx
            # Number of lower triangular elements for this world
            num_lower_tri = (num_geoms_in_world * (num_geoms_in_world - 1)) // 2
            cumsum += num_lower_tri
            world_cumsum_lower_tri_np[world_idx] = cumsum
            start_idx = end_idx

        # Store as warp arrays
        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)
        self.world_cumsum_lower_tri = wp.array(world_cumsum_lower_tri_np, dtype=wp.int32, device=device)

        # Store total number of kernel threads needed (last element of cumsum)
        self.num_kernel_threads = int(world_cumsum_lower_tri_np[-1]) if world_count > 0 else 0

        # Store number of regular worlds (for distinguishing dedicated -1 segment)
        self.num_regular_worlds = int(num_regular_worlds)

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],  # Lower bounds of shape bounding boxes
        shape_upper: wp.array[wp.vec3],  # Upper bounds of shape bounding boxes
        shape_gap: wp.array[float] | None,  # Optional per-shape effective gaps
        shape_collision_group: wp.array[int],  # Collision group ID per box
        shape_world: wp.array[int],  # World index per box
        shape_count: int,  # Number of active bounding boxes
        # Outputs
        candidate_pair: wp.array[wp.vec2i],  # Array to store overlapping shape pairs
        candidate_pair_count: wp.array[int],
        device: Devicelike | None = None,  # Device to launch on
        filter_pairs: wp.array[wp.vec2i] | None = None,  # Sorted excluded pairs
        num_filter_pairs: int | None = None,
        skip_count_zero: bool = False,  # Skip candidate_pair_count.zero_() if already zeroed by the caller
        filter_data: Any | None = None,  # Instance of filter_data_type for the user-supplied filter callback
    ) -> None:
        """Launch the N x N broad phase collision detection.

        This method performs collision detection between all possible pairs of geometries by checking for
        axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
        checking each pair twice.

        Args:
            shape_lower: Array of lower bounds for each shape's AABB
            shape_upper: Array of upper bounds for each shape's AABB
            shape_gap: Optional array of per-shape effective gaps. If None or empty array,
                assumes AABBs are pre-expanded (gaps = 0). If provided, gaps are added during overlap checks.
            shape_collision_group: Array of collision group IDs for each shape. Positive values indicate
                groups that only collide with themselves (and with negative groups). Negative values indicate
                groups that collide with everything except their negative counterpart. Zero indicates no collisions.
            shape_world: Array of world indices for each shape. Index -1 indicates global entities
                that collide with all worlds. Indices 0, 1, 2, ... indicate world-specific entities.
            shape_count: Number of active bounding boxes to check
            candidate_pair: Output array to store overlapping shape pairs
            candidate_pair_count: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.
            skip_count_zero: If True, skip the internal ``candidate_pair_count.zero_()``.
                The caller guarantees ``candidate_pair_count[0] == 0`` on entry (e.g. when
                the counter was zeroed by a preceding fused kernel).  Defaults to False so
                the launch remains self-contained.

        The method will populate candidate_pair with the indices of shape pairs (i,j) where i < j whose AABBs overlap
        (with optional margin expansion), whose collision groups allow interaction, and whose world indices are
        compatible (same world or at least one is global). Pairs in filter_pairs (if provided) are excluded.
        The number of pairs found will be written to candidate_pair_count[0].
        """
        max_candidate_pair = candidate_pair.shape[0]

        if not skip_count_zero:
            candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device

        # If no gaps provided, pass empty array (kernel will use 0.0 gaps)
        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)

        # Exclusion filter: empty array and 0 when not provided or empty
        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        if self._has_custom_filter:
            if filter_data is None:
                raise ValueError(
                    "BroadPhaseAllPairs was constructed with filter_func=...; launch() requires filter_data."
                )
            kernel_filter_data = filter_data
        else:
            kernel_filter_data = self._empty_filter_data

        # Launch with the precomputed number of kernel threads
        wp.launch(
            self._kernel,
            dim=self.num_kernel_threads,
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                shape_collision_group,
                shape_world,
                self.world_cumsum_lower_tri,
                self.world_slice_ends,
                self.world_index_map,
                self.num_regular_worlds,
                filter_pairs_arr,
                n_filter,
                kernel_filter_data,
            ],
            outputs=[candidate_pair, candidate_pair_count, max_candidate_pair],
            device=device,
            record_tape=False,
        )


class BroadPhaseExplicit:
    """A broad phase collision detection class that only checks explicitly provided geometry pairs.

    This class performs collision detection only between geometry pairs that are explicitly specified,
    rather than checking all possible pairs. This can be more efficient when the set of potential
    collision pairs is known ahead of time.

    The class checks for axis-aligned bounding box (AABB) overlaps between the specified geometry pairs,
    taking into account per-geometry cutoff distances.
    """

    def __init__(
        self,
        filter_func: Any | None = None,
        filter_data_type: Any | None = None,
    ) -> None:
        """Initialize the explicit-pairs broad phase.

        Args:
            filter_func: Optional ``@wp.func`` invoked as
                ``filter_func(pair, data) -> wp.int32`` after the AABB overlap test.
                Returning ``0`` drops the pair.
            filter_data_type: ``wp.struct`` type for the ``filter_data`` argument
                forwarded to :meth:`launch`. Required when ``filter_func`` is set.
        """
        if (filter_func is None) != (filter_data_type is None):
            raise ValueError("filter_func and filter_data_type must be provided together")
        if filter_func is None:
            self._kernel = _nxn_broadphase_precomputed_pairs
            self._has_custom_filter = False
        else:
            self._kernel = create_nxn_broadphase_precomputed_pairs_kernel(filter_func, filter_data_type)
            self._has_custom_filter = True
        self._empty_filter_data = EmptyFilterData()

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],  # Lower bounds of shape bounding boxes
        shape_upper: wp.array[wp.vec3],  # Upper bounds of shape bounding boxes
        shape_gap: wp.array[float] | None,  # Optional per-shape effective gaps
        shape_pairs: wp.array[wp.vec2i],  # Precomputed pairs to check
        shape_pair_count: int,
        # Outputs
        candidate_pair: wp.array[wp.vec2i],  # Array to store overlapping shape pairs
        candidate_pair_count: wp.array[int],
        device: Devicelike | None = None,  # Device to launch on
        skip_count_zero: bool = False,  # Skip candidate_pair_count.zero_() if already zeroed
        filter_data: Any | None = None,  # Instance of filter_data_type for the user-supplied filter callback
    ) -> None:
        """Launch the explicit pairs broad phase collision detection.

        This method checks for AABB overlaps only between explicitly specified shape pairs,
        rather than checking all possible pairs. It populates the candidate_pair array with
        indices of shape pairs whose AABBs overlap.

        Args:
            shape_lower: Array of lower bounds for shape bounding boxes
            shape_upper: Array of upper bounds for shape bounding boxes
            shape_gap: Optional array of per-shape effective gaps. If None or empty array,
                assumes AABBs are pre-expanded (gaps = 0). If provided, gaps are added during overlap checks.
            shape_pairs: Array of precomputed shape pairs to check
            shape_pair_count: Number of shape pairs to check
            candidate_pair: Output array to store overlapping shape pairs
            candidate_pair_count: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.
            skip_count_zero: If True, skip the internal ``candidate_pair_count.zero_()``.
                The caller guarantees ``candidate_pair_count[0] == 0`` on entry (e.g. when
                the counter was zeroed by a preceding fused kernel).  Defaults to False so
                the launch remains self-contained.

        The method will populate candidate_pair with the indices of shape pairs whose AABBs overlap
        (with optional margin expansion), but only checking the explicitly provided pairs.
        """

        max_candidate_pair = candidate_pair.shape[0]

        if not skip_count_zero:
            candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device

        # If no gaps provided, pass empty array (kernel will use 0.0 gaps)
        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)

        if self._has_custom_filter:
            if filter_data is None:
                raise ValueError(
                    "BroadPhaseExplicit was constructed with filter_func=...; launch() requires filter_data."
                )
            kernel_filter_data = filter_data
        else:
            kernel_filter_data = self._empty_filter_data

        wp.launch(
            kernel=self._kernel,
            dim=shape_pair_count,
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                shape_pairs,
                kernel_filter_data,
            ],
            outputs=[
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            ],
            device=device,
            record_tape=False,
        )
