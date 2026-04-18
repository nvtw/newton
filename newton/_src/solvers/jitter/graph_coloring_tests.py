# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the graph-coloring / maximal-independent-set partitioner
translated from PhoenX's ``ContactPartitions`` / ``PartitioningKernels``.

The core correctness property verified here is the same one the C#
``ValidatePartitions`` method checks: within any single partition, no two
elements may reference the same vertex (body). Each partition therefore forms
an independent set in the element-interaction graph, which is exactly what a
graph colouring guarantees.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.graph_coloring import (
    MAX_BODIES,
    ContactPartitionerIncremental,
    ElementInteractionData,
    maximal_independent_set_partitioning,
)


@wp.kernel
def _tag_partition_ids_kernel(
    partition_data: wp.array[int],
    partition_start: wp.array[int],
    partition_end: wp.array[int],
    color: wp.array[int],
    interaction_id_to_partition: wp.array[int],
):
    """Stamp the partition index onto every element emitted in the last step.

    Launched with dim=capacity (<= max_num_interactions). Threads whose id
    lies outside ``[partition_start[0], partition_end[0])`` early-return.
    ``color[0]`` is read *after* the step's color-increment, so the partition
    that was just emitted has index ``color[0] - 1``.
    """
    tid = wp.tid()
    start = partition_start[0]
    end = partition_end[0]
    if tid < start or tid >= end:
        return
    element_id = partition_data[tid]
    interaction_id_to_partition[element_id] = color[0] - 1


def _build_elements_array(elements_bodies: list[list[int]], capacity: int) -> wp.array:
    """Build a ``wp.array[ElementInteractionData]`` of the given capacity.

    Inactive slots past ``len(elements_bodies)`` are filled with ``-1`` bodies
    and should never be visited by kernels (they early-return on
    ``tid >= num_elements[0]``).
    """
    max_bodies = int(MAX_BODIES)
    # Warp expects a structured numpy dtype matching the @wp.struct layout:
    # a single field ``bodies`` of type int32[8].
    struct_dtype = np.dtype({"names": ["bodies"], "formats": [(np.int32, max_bodies)], "offsets": [0], "itemsize": 32})
    data = np.zeros(capacity, dtype=struct_dtype)
    data["bodies"][:] = -1
    for i, bodies in enumerate(elements_bodies):
        assert len(bodies) <= max_bodies
        data["bodies"][i, : len(bodies)] = bodies
    return wp.from_numpy(data, dtype=ElementInteractionData)


def _run_partitioner(
    elements_bodies: list[list[int]],
    max_num_nodes: int,
    max_num_partitions: int,
    max_num_interactions: int | None = None,
    random_values: np.ndarray | None = None,
) -> dict:
    """Allocate all buffers, run the partitioner, and return the results as
    numpy arrays for host-side validation."""
    if max_num_interactions is None:
        max_num_interactions = len(elements_bodies)

    num_elements = len(elements_bodies)
    assert num_elements <= max_num_interactions

    # Elements + count.
    elements = _build_elements_array(elements_bodies, max_num_interactions)
    num_elements_arr = wp.array([num_elements], dtype=wp.int32)

    # ContactPartitionsGpu buffers.
    partition_ends = wp.zeros(max_num_partitions + 1, dtype=wp.int32)
    num_partitions_arr = wp.zeros(1, dtype=wp.int32)
    has_additional_partition = wp.zeros(1, dtype=wp.int32)
    max_used_color = wp.zeros(1, dtype=wp.int32)
    partition_data_concat = wp.zeros(2 * max_num_interactions, dtype=wp.int32)
    interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32)
    removed_marker_array = wp.zeros(max_num_interactions, dtype=wp.int32)

    # Random values per element -- used as Luby MIS priorities. These must be
    # pairwise distinct for the algorithm to avoid ties (which would allow two
    # neighbouring elements to both claim the same partition). We use a seeded
    # permutation of [1, max_num_interactions] to guarantee uniqueness.
    if random_values is None:
        rng = np.random.default_rng(12345)
        random_values = rng.permutation(max_num_interactions).astype(np.int32) + 1
    else:
        random_values = np.asarray(random_values, dtype=np.int32)
        assert random_values.shape == (max_num_interactions,)
    random_values_arr = wp.from_numpy(random_values, dtype=wp.int32)

    # Adjacency buffers. Upper bound on adjacency entries: every element
    # contributes at most MAX_BODIES vertex references.
    adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32)
    vertex_to_adjacent_elements = wp.zeros(max_num_interactions * int(MAX_BODIES), dtype=wp.int32)

    max_num_contacts = max_num_interactions
    section_marker_single_el_arr = wp.array([max_num_interactions], dtype=wp.int32)

    # Sort scratch (key-value pairs, 2*N each).
    partition_data_concat_sort_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32)

    maximal_independent_set_partitioning(
        elements=elements,
        num_elements=num_elements_arr,
        max_num_nodes=max_num_nodes,
        section_marker_single_el_arr=section_marker_single_el_arr,
        partition_ends=partition_ends,
        num_partitions=num_partitions_arr,
        has_additional_partition=has_additional_partition,
        max_used_color=max_used_color,
        max_num_partitions=max_num_partitions,
        partition_data_concat=partition_data_concat,
        interaction_id_to_partition=interaction_id_to_partition,
        removed_marker_array=removed_marker_array,
        random_values=random_values_arr,
        adjacency_section_end_indices=adjacency_section_end_indices,
        vertex_to_adjacent_elements=vertex_to_adjacent_elements,
        max_num_contacts=max_num_contacts,
        partition_data_concat_sort_values=partition_data_concat_sort_values,
    )

    return {
        "num_elements": num_elements,
        "num_partitions": int(num_partitions_arr.numpy()[0]),
        "has_additional_partition": int(has_additional_partition.numpy()[0]),
        "max_used_color": int(max_used_color.numpy()[0]),
        "partition_ends": partition_ends.numpy(),
        "partition_data_concat": partition_data_concat.numpy()[: max_num_interactions],
        "interaction_id_to_partition": interaction_id_to_partition.numpy(),
        "max_num_partitions": max_num_partitions,
    }


def _get_partition_slice(result: dict, partition_index: int) -> np.ndarray:
    """Python port of ``ContactPartitionsGpu.GetPartition`` — returns the
    sorted slice of ``partition_data_concat`` that contains the element ids
    belonging to partition ``partition_index``."""
    num_partitions = result["num_partitions"]
    if partition_index > num_partitions:
        return np.empty(0, dtype=np.int32)
    if partition_index == num_partitions and result["has_additional_partition"] == 0:
        return np.empty(0, dtype=np.int32)

    ends = result["partition_ends"]
    start = 0 if partition_index == 0 else int(ends[partition_index - 1])
    end = int(ends[partition_index])
    return result["partition_data_concat"][start:end]


def _validate_partitions(elements_bodies: list[list[int]], result: dict, max_num_nodes: int) -> None:
    """Python port of the C# ``ValidatePartitions``. Raises ``AssertionError``
    on any invariant violation.

    Matches the C# contract: only the real (non-overflow) partitions must form
    independent sets. The overflow partition at index ``num_partitions`` is
    intentionally exempt -- it holds elements that could not be placed within
    the ``max_num_partitions`` budget and may have internal conflicts.
    """
    num_partitions = result["num_partitions"]
    has_additional = result["has_additional_partition"]

    seen_elements: set[int] = set()
    total_assigned = 0

    for p in range(num_partitions):
        partition = _get_partition_slice(result, p)
        total_assigned += len(partition)

        node_was_accessed = np.zeros(max_num_nodes, dtype=bool)
        accessed_nodes: set[int] = set()

        for element_id in partition:
            assert element_id not in seen_elements, f"Element {element_id} appears in two partitions"
            seen_elements.add(int(element_id))

            assert 0 <= element_id < len(elements_bodies), f"Element id {element_id} out of range"
            bodies = elements_bodies[element_id]
            for body_id in bodies:
                if body_id < 0:
                    continue
                assert body_id < max_num_nodes, f"Body id {body_id} exceeds max_num_nodes={max_num_nodes}"
                assert body_id not in accessed_nodes, (
                    f"Partition {p}: body {body_id} used by two elements (element {element_id})"
                )
                accessed_nodes.add(int(body_id))
                assert not node_was_accessed[body_id]
                node_was_accessed[body_id] = True

    if has_additional:
        overflow = _get_partition_slice(result, num_partitions)
        for element_id in overflow:
            assert element_id not in seen_elements, f"Element {element_id} appears in two partitions"
            seen_elements.add(int(element_id))
            total_assigned += 1

    assert total_assigned == len(elements_bodies), (
        f"Expected every element to be assigned exactly once; got {total_assigned} assignments for "
        f"{len(elements_bodies)} elements"
    )


class TestGraphColoring(unittest.TestCase):
    def test_disjoint_elements_one_partition(self):
        # Four elements, each pair uses its own vertices. All should fit in
        # partition 0.
        elements_bodies = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]
        result = _run_partitioner(elements_bodies, max_num_nodes=8, max_num_partitions=4)
        _validate_partitions(elements_bodies, result, max_num_nodes=8)
        self.assertGreaterEqual(result["num_partitions"], 1)

        # With no conflicts, we expect the first partition to hold all elements
        # and no "additional" overflow.
        self.assertEqual(result["has_additional_partition"], 0)
        first = _get_partition_slice(result, 0)
        self.assertEqual(len(first), len(elements_bodies))

    def test_fully_connected_each_element_own_partition(self):
        # Every element touches body 0 => a clique. All elements conflict
        # pairwise; each needs its own partition.
        elements_bodies = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ]
        result = _run_partitioner(elements_bodies, max_num_nodes=5, max_num_partitions=8)
        _validate_partitions(elements_bodies, result, max_num_nodes=5)

        # Each partition contains exactly one element.
        non_empty = 0
        for p in range(result["num_partitions"] + result["has_additional_partition"]):
            sl = _get_partition_slice(result, p)
            if len(sl) > 0:
                self.assertEqual(len(sl), 1)
                non_empty += 1
        self.assertEqual(non_empty, len(elements_bodies))

    def test_path_graph(self):
        # A path: 0-1, 1-2, 2-3, 3-4. Classic 2-colourable structure.
        elements_bodies = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ]
        result = _run_partitioner(elements_bodies, max_num_nodes=5, max_num_partitions=8)
        _validate_partitions(elements_bodies, result, max_num_nodes=5)

    def test_multibody_elements(self):
        # Elements using 3 and 4 bodies each (articulated constraints).
        elements_bodies = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8, 9],
            [0, 5, 10],  # conflicts with elements 0 and 1
        ]
        result = _run_partitioner(elements_bodies, max_num_nodes=11, max_num_partitions=8)
        _validate_partitions(elements_bodies, result, max_num_nodes=11)

    def test_overflow_partition_used(self):
        # Force overflow: 6 mutually-conflicting elements with only 2 partitions.
        # All elements share body 0, so they pairwise conflict. With
        # max_num_partitions=2 only 2 elements fit into regular partitions; the
        # rest should land in the "additional" overflow partition.
        elements_bodies = [[0, i + 1] for i in range(6)]
        result = _run_partitioner(elements_bodies, max_num_nodes=7, max_num_partitions=2)
        _validate_partitions(elements_bodies, result, max_num_nodes=7)

        self.assertEqual(result["has_additional_partition"], 1)
        # Overflow partition is at index num_partitions.
        overflow = _get_partition_slice(result, result["num_partitions"])
        self.assertGreater(len(overflow), 0)

    def test_capacity_larger_than_active(self):
        # Allocate buffers larger than the active element count to make sure
        # the early-return guards in every kernel work.
        elements_bodies = [
            [0, 1],
            [2, 3],
        ]
        result = _run_partitioner(
            elements_bodies,
            max_num_nodes=4,
            max_num_partitions=4,
            max_num_interactions=16,
        )
        _validate_partitions(elements_bodies, result, max_num_nodes=4)


# ---------------------------------------------------------------------------
# Incremental partitioner tests (exercised via ``wp.capture_while``).
# ---------------------------------------------------------------------------


def _run_incremental_capture_while(
    elements_bodies: list[list[int]],
    max_num_nodes: int,
    device,
    use_graph_capture: bool,
) -> dict:
    """Drive ``ContactPartitionerIncremental`` with ``wp.capture_while``.

    When ``use_graph_capture`` is ``True`` the entire reset + while loop is
    recorded inside a ``wp.ScopedCapture`` and replayed via
    ``wp.capture_launch``. When ``False`` the same code path runs eagerly --
    ``capture_while`` then polls the device condition host-side.
    """
    with wp.ScopedDevice(device):
        N = len(elements_bodies)
        max_num_interactions = N

        elements = _build_elements_array(elements_bodies, max_num_interactions)
        num_elements_arr = wp.array([N], dtype=wp.int32)

        partitioner = ContactPartitionerIncremental(
            max_num_interactions=max_num_interactions,
            max_num_nodes=max_num_nodes,
            device=device,
            seed=12345,
        )

        # Per-element partition assignment, filled in by `_tag_partition_ids_kernel`
        # after each step. Initialised to -1 so unassigned elements are easy to spot.
        interaction_id_to_partition = wp.full(max_num_interactions, -1, dtype=wp.int32, device=device)

        def body():
            partitioner.launch(elements, num_elements_arr)
            wp.launch(
                _tag_partition_ids_kernel,
                dim=max_num_interactions,
                inputs=[
                    partitioner.partition_data,
                    partitioner.partition_start,
                    partitioner.partition_end,
                    partitioner.color,
                    interaction_id_to_partition,
                ],
            )

        if use_graph_capture:
            wp.load_module(device=device)
            with wp.ScopedCapture(force_module_load=False) as capture:
                partitioner.reset(elements, num_elements_arr)
                wp.capture_while(partitioner.not_done, body)
            wp.capture_launch(capture.graph)
        else:
            partitioner.reset(elements, num_elements_arr)
            wp.capture_while(partitioner.not_done, body)

        return {
            "num_elements": N,
            "num_partitions": int(partitioner.color.numpy()[0]),
            "interaction_id_to_partition": interaction_id_to_partition.numpy(),
        }


def _validate_incremental_result(
    elements_bodies: list[list[int]], result: dict, max_num_nodes: int
) -> None:
    """Check that each partition forms an independent set and that every
    element was assigned to exactly one partition (no overflow in incremental
    mode)."""
    assignment = result["interaction_id_to_partition"]
    num_partitions = result["num_partitions"]

    assert np.all(assignment[: result["num_elements"]] >= 0), (
        f"some elements were not assigned: {assignment[: result['num_elements']].tolist()}"
    )
    assert int(assignment[: result["num_elements"]].max()) < num_partitions

    for p in range(num_partitions):
        members = np.where(assignment == p)[0].tolist()
        accessed_bodies: set[int] = set()
        for element_id in members:
            for body_id in elements_bodies[element_id]:
                if body_id < 0:
                    continue
                assert body_id < max_num_nodes
                assert body_id not in accessed_bodies, (
                    f"Partition {p}: body {body_id} used by two elements (element {element_id}); "
                    f"members={members}"
                )
                accessed_bodies.add(body_id)


_CUDA_DEVICE_AVAILABLE = any(d.is_cuda for d in wp.get_devices())


# Graphs exercised by both the capture and non-capture paths. Each tuple
# is (label, elements_bodies, max_num_nodes).
_INCREMENTAL_CASES = (
    ("disjoint", [[0, 1], [2, 3], [4, 5], [6, 7]], 8),
    ("clique", [[0, 1], [0, 2], [0, 3], [0, 4]], 5),
    ("path", [[0, 1], [1, 2], [2, 3], [3, 4]], 5),
    ("multibody", [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], [0, 5, 10]], 11),
)


class TestGraphColoringIncremental(unittest.TestCase):
    @unittest.skipUnless(_CUDA_DEVICE_AVAILABLE, "capture_while needs a CUDA device")
    def test_incremental_capture_while_nocapture(self):
        device = wp.get_cuda_device()
        for label, elements_bodies, max_num_nodes in _INCREMENTAL_CASES:
            with self.subTest(graph=label):
                result = _run_incremental_capture_while(
                    elements_bodies, max_num_nodes, device, use_graph_capture=False
                )
                _validate_incremental_result(elements_bodies, result, max_num_nodes)

    @unittest.skipUnless(
        _CUDA_DEVICE_AVAILABLE and wp.is_conditional_graph_supported(),
        "conditional graph nodes not supported on this device",
    )
    def test_incremental_capture_while_with_graph(self):
        device = wp.get_cuda_device()
        for label, elements_bodies, max_num_nodes in _INCREMENTAL_CASES:
            with self.subTest(graph=label):
                result = _run_incremental_capture_while(
                    elements_bodies, max_num_nodes, device, use_graph_capture=True
                )
                _validate_incremental_result(elements_bodies, result, max_num_nodes)

    @unittest.skipUnless(_CUDA_DEVICE_AVAILABLE, "stress test needs a CUDA device")
    def test_incremental_stress_high_degree(self):
        """50k elements, ~25 element-references per body, validated vectorised.

        Each body appears in 20-30 elements on average, producing a high-degree
        conflict graph whose chromatic number is of the same order. This
        exercises the incremental kernel across many partitions and makes sure
        the shared 2*N partition_data_concat buffer, adjacency indices, and
        random-priority permutation all scale beyond the toy test cases.
        """
        device = wp.get_cuda_device()

        num_elements = 50_000
        avg_refs_per_body = 25
        bodies_per_element = 2  # pairwise interactions
        num_bodies = (num_elements * bodies_per_element) // avg_refs_per_body

        rng = np.random.default_rng(20260418)

        # Build body pairs with potential self-loops rejected. Drawing uniformly
        # and resampling duplicates keeps the distribution close to "each body
        # referenced ~25 times" without a per-element Python loop.
        bodies_a = rng.integers(0, num_bodies, size=num_elements, dtype=np.int32)
        bodies_b = rng.integers(0, num_bodies, size=num_elements, dtype=np.int32)
        # Resolve self-loops by bumping one index; a single pass is enough for
        # this data size since collisions are rare (~0.04%).
        mask = bodies_a == bodies_b
        bodies_b[mask] = (bodies_b[mask] + 1) % num_bodies

        # Sanity-check the actual degree distribution -- we want most bodies in
        # the 20-30 range as requested.
        counts = np.bincount(np.concatenate([bodies_a, bodies_b]), minlength=num_bodies)
        mean_refs = counts.mean()
        self.assertGreater(mean_refs, 20.0)
        self.assertLess(mean_refs, 32.0)

        # Assemble the structured element buffer in one vectorised step
        # (per-element Python loops would dominate the runtime at this size).
        max_bodies = int(MAX_BODIES)
        struct_dtype = np.dtype(
            {"names": ["bodies"], "formats": [(np.int32, max_bodies)], "offsets": [0], "itemsize": 32}
        )
        host_elements = np.zeros(num_elements, dtype=struct_dtype)
        host_elements["bodies"][:] = -1
        host_elements["bodies"][:, 0] = bodies_a
        host_elements["bodies"][:, 1] = bodies_b
        elements = wp.from_numpy(host_elements, dtype=ElementInteractionData, device=device)
        num_elements_arr = wp.array([num_elements], dtype=wp.int32, device=device)

        partitioner = ContactPartitionerIncremental(
            max_num_interactions=num_elements,
            max_num_nodes=num_bodies,
            device=device,
            seed=20260418,
        )
        interaction_id_to_partition = wp.full(num_elements, -1, dtype=wp.int32, device=device)

        def body():
            partitioner.launch(elements, num_elements_arr)
            wp.launch(
                _tag_partition_ids_kernel,
                dim=num_elements,
                inputs=[
                    partitioner.partition_data,
                    partitioner.partition_start,
                    partitioner.partition_end,
                    partitioner.color,
                    interaction_id_to_partition,
                ],
            )

        partitioner.reset(elements, num_elements_arr)
        wp.capture_while(partitioner.not_done, body)

        assignment = interaction_id_to_partition.numpy()
        num_partitions = int(partitioner.color.numpy()[0])

        # Every element must be assigned, and the assignment must stay within
        # the advertised partition count.
        unassigned = int(np.count_nonzero(assignment < 0))
        self.assertEqual(unassigned, 0, f"{unassigned} elements were not assigned to any partition")
        self.assertLess(int(assignment.max()), num_partitions)

        # The chromatic number of this graph lower-bounds by max body degree.
        # We don't require tightness but we do require "many" partitions --
        # otherwise the test wouldn't be stressing the incremental loop.
        self.assertGreaterEqual(num_partitions, 20)

        # Vectorised independent-set check. For every body, gather the partition
        # ids of every element that references it and assert they are distinct.
        # Concatenate (body_id, partition_id) pairs for bodies_a and bodies_b,
        # sort by (body_id, partition_id), and check that no two consecutive
        # pairs share a (body_id, partition_id). That would mean the same body
        # appears twice inside one partition, violating independence.
        body_ids = np.concatenate([bodies_a, bodies_b])
        partition_ids = np.concatenate([assignment, assignment])
        order = np.lexsort((partition_ids, body_ids))
        body_sorted = body_ids[order]
        part_sorted = partition_ids[order]
        same_pair = (body_sorted[1:] == body_sorted[:-1]) & (part_sorted[1:] == part_sorted[:-1])
        if np.any(same_pair):
            first_violation = int(np.argmax(same_pair))
            self.fail(
                f"Partition {int(part_sorted[first_violation])} has two elements sharing "
                f"body {int(body_sorted[first_violation])}"
            )


if __name__ == "__main__":
    wp.init()
    unittest.main()
