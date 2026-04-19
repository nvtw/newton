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
    ContactPartitioner,
    ElementInteractionData,
    maximal_independent_set_partitioning,
)
from newton._src.solvers.jitter.graph_coloring_incremental import IncrementalContactPartitioner


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
    # Sort key buffer (int64 packed marker|color|tid) + its int32 element-id view.
    partition_data_concat = wp.zeros(2 * max_num_interactions, dtype=wp.int64)
    partition_data_elements = wp.zeros(max_num_interactions, dtype=wp.int32)
    interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32)

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
    # 1-element device array feeding the per-color loop inside the partitioner.
    color_arr = wp.zeros(1, dtype=wp.int32)

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
        partition_data_elements=partition_data_elements,
        interaction_id_to_partition=interaction_id_to_partition,
        random_values=random_values_arr,
        adjacency_section_end_indices=adjacency_section_end_indices,
        vertex_to_adjacent_elements=vertex_to_adjacent_elements,
        max_num_contacts=max_num_contacts,
        partition_data_concat_sort_values=partition_data_concat_sort_values,
        color_arr=color_arr,
    )

    return {
        "num_elements": num_elements,
        "num_partitions": int(num_partitions_arr.numpy()[0]),
        "has_additional_partition": int(has_additional_partition.numpy()[0]),
        "max_used_color": int(max_used_color.numpy()[0]),
        "partition_ends": partition_ends.numpy(),
        "partition_data_concat": partition_data_elements.numpy()[:max_num_interactions],
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


def _generate_stress_workload(
    num_elements: int,
    num_bodies: int,
    high_deg_nodes: int,
    high_deg_refs: tuple[int, int],
    bodies_per_elem: tuple[int, int],
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic element->bodies table with a few high-degree "hub"
    vertices that force many pairwise conflicts."""
    rng = np.random.default_rng(seed)
    max_bodies = int(MAX_BODIES)

    hub_ids = rng.choice(num_bodies, size=high_deg_nodes, replace=False)
    hub_refs = rng.integers(high_deg_refs[0], high_deg_refs[1] + 1, size=high_deg_nodes)

    per_elem_count = rng.integers(bodies_per_elem[0], bodies_per_elem[1] + 1, size=num_elements)
    per_elem_count = np.minimum(per_elem_count, max_bodies)

    bodies = np.full((num_elements, max_bodies), -1, dtype=np.int32)
    for i in range(num_elements):
        k = per_elem_count[i]
        bodies[i, :k] = rng.choice(num_bodies, size=k, replace=False)

    for hub, ref in zip(hub_ids, hub_refs, strict=True):
        elems = rng.choice(num_elements, size=int(min(ref, num_elements)), replace=False)
        for e in elems:
            if hub in bodies[e]:
                continue
            free = np.where(bodies[e] == -1)[0]
            if len(free) > 0:
                bodies[e, free[0]] = hub
            else:
                bodies[e, -1] = hub

    # Deduplicate per-element bodies (safety net).
    for i in range(num_elements):
        uniq: list[int] = []
        for b in bodies[i]:
            if b >= 0 and int(b) not in uniq:
                uniq.append(int(b))
        row = np.full(max_bodies, -1, dtype=np.int32)
        row[: len(uniq)] = uniq
        bodies[i] = row

    return bodies


def _make_stress_elements_array(bodies_np: np.ndarray, device) -> wp.array:
    num_elements = bodies_np.shape[0]
    struct_dtype = np.dtype({"names": ["bodies"], "formats": [(np.int32, int(MAX_BODIES))], "offsets": [0], "itemsize": 32})
    arr = np.zeros(num_elements, dtype=struct_dtype)
    arr["bodies"] = bodies_np
    return wp.from_numpy(arr, dtype=ElementInteractionData, device=device)


def _validate_stress_partitions(
    bodies_np: np.ndarray,
    partition_ends: np.ndarray,
    partition_data_concat: np.ndarray,
    num_partitions: int,
    has_additional: int,
    max_num_partitions: int,
) -> tuple[bool, str]:
    """Verify that every non-overflow partition is an independent set and that
    every element is assigned exactly once (including the overflow bucket)."""
    num_elements = bodies_np.shape[0]
    seen = np.zeros(num_elements, dtype=bool)

    start = 0
    for p in range(num_partitions):
        end = int(partition_ends[p])
        used_bodies: set[int] = set()
        for k in range(start, end):
            eid = int(partition_data_concat[k])
            if eid < 0 or eid >= num_elements:
                return False, f"partition {p}: bad element id {eid}"
            if seen[eid]:
                return False, f"partition {p}: element {eid} already assigned"
            seen[eid] = True
            for b in bodies_np[eid]:
                if b < 0:
                    continue
                if int(b) in used_bodies:
                    return False, f"partition {p}: body {int(b)} shared (elem {eid})"
                used_bodies.add(int(b))
        start = end

    if has_additional:
        end = int(partition_ends[max_num_partitions])
        for k in range(start, end):
            eid = int(partition_data_concat[k])
            if 0 <= eid < num_elements:
                seen[eid] = True

    if not seen.all():
        missing = np.where(~seen)[0][:5]
        return False, f"elements not covered: {missing}"
    return True, "ok"


class TestGraphColoringStress(unittest.TestCase):
    """Large-input stress tests for ``ContactPartitioner``.

    Each case sweeps several ``max_num_partitions`` budgets and verifies that:
      * every non-overflow partition is an independent set,
      * every element is assigned exactly once, and
      * two back-to-back runs with the same seed produce byte-identical output
        on every buffer (determinism).

    A correct colouring can always fit the elements within (Delta + 1) colours
    where Delta is the max degree of the conflict graph, so the sweep covers
    budgets both below and well above that bound.
    """

    def _run_once(
        self,
        bodies_np: np.ndarray,
        num_bodies: int,
        budget: int,
        device,
        seed: int = 0,
    ) -> dict:
        num_elements = bodies_np.shape[0]
        num_elements_arr = wp.array([num_elements], dtype=wp.int32, device=device)
        elements_arr = _make_stress_elements_array(bodies_np, device)
        partitioner = ContactPartitioner(
            max_num_interactions=num_elements,
            max_num_nodes=num_bodies,
            max_num_partitions=budget,
            device=device,
            seed=seed,
        )
        partitioner.launch(elements_arr, num_elements_arr)
        return {
            "num_partitions": partitioner.num_partitions.numpy().copy(),
            "has_additional_partition": partitioner.has_additional_partition.numpy().copy(),
            "partition_ends": partitioner.partition_ends.numpy().copy(),
            "partition_data_concat": partitioner.partition_data_concat.numpy().copy(),
            "interaction_id_to_partition": partitioner.interaction_id_to_partition.numpy().copy(),
        }

    def _run_case(
        self,
        num_elements: int,
        num_bodies: int,
        high_deg_nodes: int,
        high_deg_refs: tuple[int, int],
        bodies_per_elem: tuple[int, int],
        budgets: list[int],
        expect_empty_overflow: bool = True,
    ) -> None:
        device = wp.get_preferred_device()
        bodies_np = _generate_stress_workload(
            num_elements, num_bodies, high_deg_nodes, high_deg_refs, bodies_per_elem, seed=0
        )

        first_empty_overflow: int | None = None
        for budget in budgets:
            r1 = self._run_once(bodies_np, num_bodies, budget, device, seed=0)
            r2 = self._run_once(bodies_np, num_bodies, budget, device, seed=0)

            for key in (
                "num_partitions",
                "has_additional_partition",
                "partition_ends",
                "partition_data_concat",
                "interaction_id_to_partition",
            ):
                self.assertTrue(
                    np.array_equal(r1[key], r2[key]),
                    msg=f"non-deterministic output for budget={budget}, field={key}",
                )

            ok, msg = _validate_stress_partitions(
                bodies_np,
                r1["partition_ends"],
                r1["partition_data_concat"][:num_elements],
                int(r1["num_partitions"][0]),
                int(r1["has_additional_partition"][0]),
                budget,
            )
            self.assertTrue(ok, msg=f"budget={budget}: {msg}")

            if int(r1["has_additional_partition"][0]) == 0 and first_empty_overflow is None:
                first_empty_overflow = budget

        if expect_empty_overflow:
            self.assertIsNotNone(
                first_empty_overflow,
                msg="no budget in the sweep produced an empty overflow partition",
            )

    def test_stress_small_hub_graph(self):
        # 500 elements, 200 bodies, 5 high-degree hubs. Conflict-graph max
        # degree is ~117 for this seed, so budgets straddle that bound and go
        # well past it (the upper bound stresses out-of-bounds edges in the
        # prepare kernel for the ``adjacency_section_end_indices`` buffer).
        self._run_case(
            num_elements=500,
            num_bodies=200,
            high_deg_nodes=5,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            budgets=[29, 58, 116, 117, 118, 119, 234, 468],
        )

    def test_stress_medium_hub_graph(self):
        self._run_case(
            num_elements=5_000,
            num_bodies=2_000,
            high_deg_nodes=10,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            budgets=[22, 44, 88, 89, 90, 91, 178, 356],
        )

    def test_stress_large_hub_graph(self):
        self._run_case(
            num_elements=50_000,
            num_bodies=20_000,
            high_deg_nodes=20,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            budgets=[21, 42, 84, 85, 86, 87, 170, 340],
        )


def _run_incremental_to_completion(
    bodies_np: np.ndarray,
    num_bodies: int,
    device,
    seed: int = 0,
) -> dict:
    """Run :class:`IncrementalContactPartitioner` until every element is
    colored. Returns the accumulated per-partition element lists alongside
    the final ``interaction_id_to_partition``."""
    num_elements = bodies_np.shape[0]
    elements_arr = _make_stress_elements_array(bodies_np, device)
    num_elements_arr = wp.array([num_elements], dtype=wp.int32, device=device)

    p = IncrementalContactPartitioner(
        max_num_interactions=num_elements,
        max_num_nodes=num_bodies,
        device=device,
        seed=seed,
    )
    p.reset(elements_arr, num_elements_arr)

    partitions: list[np.ndarray] = []
    # Hard upper bound on the number of colours; prevents infinite loops if
    # the implementation regresses. JP needs <= Delta+1 <= MAX_BODIES*num_elements.
    max_launches = num_elements + 1
    launches = 0
    while True:
        if int(p.num_remaining.numpy()[0]) == 0:
            break
        p.launch()
        count = int(p.partition_count.numpy()[0])
        ids = p.partition_element_ids.numpy()[:count].copy()
        partitions.append(ids)
        launches += 1
        if launches > max_launches:
            raise AssertionError(f"incremental partitioner did not terminate after {launches} launches")

    return {
        "partitions": partitions,
        "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
        "num_remaining": int(p.num_remaining.numpy()[0]),
        "current_color": int(p.current_color.numpy()[0]),
        "launches": launches,
    }


class TestIncrementalGraphColoring(unittest.TestCase):
    """Tests for the one-partition-per-launch incremental partitioner."""

    def _workload(self, **kwargs) -> np.ndarray:
        return _generate_stress_workload(seed=0, **kwargs)

    def test_small_disjoint_graph(self):
        # Four non-conflicting elements all fit into partition 0.
        bodies_np = np.array(
            [
                [0, 1, -1, -1, -1, -1, -1, -1],
                [2, 3, -1, -1, -1, -1, -1, -1],
                [4, 5, -1, -1, -1, -1, -1, -1],
                [6, 7, -1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        device = wp.get_preferred_device()
        result = _run_incremental_to_completion(bodies_np, num_bodies=8, device=device)
        self.assertEqual(result["num_remaining"], 0)
        self.assertEqual(result["launches"], 1)
        self.assertEqual(len(result["partitions"]), 1)
        self.assertEqual(sorted(result["partitions"][0].tolist()), [0, 1, 2, 3])

    def test_small_clique_graph(self):
        # Clique: every element touches body 0. Each needs its own colour.
        bodies_np = np.full((4, 8), -1, dtype=np.int32)
        for i in range(4):
            bodies_np[i, 0] = 0
            bodies_np[i, 1] = i + 1
        device = wp.get_preferred_device()
        result = _run_incremental_to_completion(bodies_np, num_bodies=5, device=device)
        self.assertEqual(result["num_remaining"], 0)
        self.assertEqual(result["launches"], 4)
        # Every partition holds exactly one element.
        for part in result["partitions"]:
            self.assertEqual(len(part), 1)

    def _validate_incremental_partitions(
        self,
        bodies_np: np.ndarray,
        partitions: list[np.ndarray],
        interaction_id_to_partition: np.ndarray,
    ) -> None:
        """Every partition must be an independent set, every element assigned
        exactly once, and ``interaction_id_to_partition`` consistent with the
        per-call lists."""
        num_elements = bodies_np.shape[0]
        seen = np.zeros(num_elements, dtype=bool)
        for color, part in enumerate(partitions):
            used_bodies: set[int] = set()
            for raw_eid in part:
                eid = int(raw_eid)
                self.assertFalse(seen[eid], msg=f"element {eid} assigned twice")
                seen[eid] = True
                self.assertEqual(
                    int(interaction_id_to_partition[eid]),
                    color,
                    msg=f"element {eid}: interaction_id_to_partition mismatch",
                )
                for b in bodies_np[eid]:
                    if b < 0:
                        continue
                    self.assertNotIn(
                        int(b),
                        used_bodies,
                        msg=f"partition {color}: body {int(b)} shared (elem {eid})",
                    )
                    used_bodies.add(int(b))
        self.assertTrue(seen.all(), msg="not every element got a partition")

    def test_stress_per_call_independent_set(self):
        bodies_np = _generate_stress_workload(
            num_elements=5_000,
            num_bodies=2_000,
            high_deg_nodes=10,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()
        result = _run_incremental_to_completion(bodies_np, num_bodies=2_000, device=device)
        self._validate_incremental_partitions(
            bodies_np, result["partitions"], result["interaction_id_to_partition"]
        )

    def test_stress_determinism(self):
        bodies_np = _generate_stress_workload(
            num_elements=5_000,
            num_bodies=2_000,
            high_deg_nodes=10,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()
        r1 = _run_incremental_to_completion(bodies_np, num_bodies=2_000, device=device, seed=0)
        r2 = _run_incremental_to_completion(bodies_np, num_bodies=2_000, device=device, seed=0)

        self.assertEqual(r1["launches"], r2["launches"])
        self.assertTrue(
            np.array_equal(r1["interaction_id_to_partition"], r2["interaction_id_to_partition"]),
            msg="interaction_id_to_partition differed between runs",
        )
        for color, (p1, p2) in enumerate(zip(r1["partitions"], r2["partitions"], strict=True)):
            self.assertTrue(
                np.array_equal(p1, p2),
                msg=f"partition {color} element-id list differs between runs",
            )

    def _eager_incremental_reference(
        self,
        bodies_np: np.ndarray,
        num_bodies: int,
        device,
        seed: int = 0,
    ) -> dict:
        """Run the incremental partitioner in eager mode and return the final
        per-element colour assignment and the number of launches. This serves
        as the ground truth when validating graph-captured runs."""
        return _run_incremental_to_completion(bodies_np, num_bodies, device, seed=seed)

    def _run_capture_while_once(
        self,
        bodies_np: np.ndarray,
        num_bodies: int,
        device,
        seed: int = 0,
    ) -> dict:
        """Build a graph that runs ``wp.capture_while`` on the partitioner's
        ``num_remaining`` counter, launch it, and return the resulting state."""
        num_elements = bodies_np.shape[0]
        elements_arr = _make_stress_elements_array(bodies_np, device)
        num_elements_arr = wp.array([num_elements], dtype=wp.int32, device=device)

        p = IncrementalContactPartitioner(
            max_num_interactions=num_elements,
            max_num_nodes=num_bodies,
            device=device,
            seed=seed,
            use_tile_scan=True,
        )
        p.reset(elements_arr, num_elements_arr)

        # Capture a graph whose body repeats ``launch()`` as long as there are
        # still-uncolored elements. ``num_remaining`` is the device-side
        # condition array; ``incremental_advance_kernel`` decrements it every
        # iteration, so the loop terminates when every element is colored.
        # ``use_tile_scan=True`` routes the per-call scan through a
        # single-block tile-scan kernel, which -- unlike ``wp.utils.array_scan``
        # -- performs no implicit device allocations and is therefore a valid
        # graph-capture body.
        with wp.ScopedCapture() as capture:
            wp.capture_while(
                p.num_remaining,
                p.launch,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        return {
            "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
            "num_remaining": int(p.num_remaining.numpy()[0]),
            "current_color": int(p.current_color.numpy()[0]),
        }

    @unittest.skipUnless(
        wp.get_device().is_cuda and wp.is_conditional_graph_supported(),
        "Conditional graph nodes not supported",
    )
    def test_stress_capture_while(self):
        # Medium stress workload: run the partitioner inside a captured graph
        # driven by ``wp.capture_while`` on the device-side ``num_remaining``
        # counter, and verify the colouring matches the eager reference.
        bodies_np = _generate_stress_workload(
            num_elements=5_000,
            num_bodies=2_000,
            high_deg_nodes=10,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()

        eager = self._eager_incremental_reference(bodies_np, 2_000, device, seed=0)
        captured = self._run_capture_while_once(bodies_np, 2_000, device, seed=0)

        self.assertEqual(captured["num_remaining"], 0)
        self.assertEqual(captured["current_color"], eager["launches"])
        self.assertTrue(
            np.array_equal(
                captured["interaction_id_to_partition"],
                eager["interaction_id_to_partition"],
            ),
            msg="graph-captured run disagrees with eager run on element -> partition assignment",
        )

    @unittest.skipUnless(
        wp.get_device().is_cuda and wp.is_conditional_graph_supported(),
        "Conditional graph nodes not supported",
    )
    def test_capture_while_determinism(self):
        # Run the captured graph twice with identical inputs and the same seed;
        # the outputs must match byte-for-byte.
        bodies_np = _generate_stress_workload(
            num_elements=5_000,
            num_bodies=2_000,
            high_deg_nodes=10,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()

        r1 = self._run_capture_while_once(bodies_np, 2_000, device, seed=0)
        r2 = self._run_capture_while_once(bodies_np, 2_000, device, seed=0)

        self.assertEqual(r1["num_remaining"], 0)
        self.assertEqual(r2["num_remaining"], 0)
        self.assertEqual(r1["current_color"], r2["current_color"])
        self.assertTrue(
            np.array_equal(r1["interaction_id_to_partition"], r2["interaction_id_to_partition"]),
            msg="graph-captured runs produced different results for identical inputs",
        )

    @unittest.skipUnless(
        wp.get_device().is_cuda and wp.is_conditional_graph_supported(),
        "Conditional graph nodes not supported",
    )
    def test_capture_while_relaunch(self):
        # Capture once, launch twice (after resetting element state between
        # launches). The graph must be re-usable and produce the same output
        # each time -- the key operational requirement for physics engines
        # that rebuild the partitioner each frame.
        bodies_np = _generate_stress_workload(
            num_elements=2_000,
            num_bodies=800,
            high_deg_nodes=5,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()

        num_elements = bodies_np.shape[0]
        elements_arr = _make_stress_elements_array(bodies_np, device)
        num_elements_arr = wp.array([num_elements], dtype=wp.int32, device=device)

        p = IncrementalContactPartitioner(
            max_num_interactions=num_elements,
            max_num_nodes=800,
            device=device,
            seed=0,
            use_tile_scan=True,
        )
        p.reset(elements_arr, num_elements_arr)

        with wp.ScopedCapture() as capture:
            wp.capture_while(p.num_remaining, p.launch)

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        first_assignment = p.interaction_id_to_partition.numpy().copy()
        first_color = int(p.current_color.numpy()[0])

        # Reset to the same state and re-run the captured graph. Reset writes
        # to the same pre-allocated buffers the graph references, so the graph
        # is still valid.
        p.reset(elements_arr, num_elements_arr)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        second_assignment = p.interaction_id_to_partition.numpy().copy()
        second_color = int(p.current_color.numpy()[0])

        self.assertEqual(first_color, second_color)
        self.assertTrue(
            np.array_equal(first_assignment, second_assignment),
            msg="re-launch of the captured graph produced different output",
        )

    def test_matches_batch_interaction_id_to_partition(self):
        # For a budget large enough to avoid overflow, the batch partitioner
        # assigns the same colour to each element as the incremental version
        # (both are deterministic and use the same JP priorities).
        bodies_np = _generate_stress_workload(
            num_elements=5_000,
            num_bodies=2_000,
            high_deg_nodes=10,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()

        # Incremental.
        inc = _run_incremental_to_completion(bodies_np, num_bodies=2_000, device=device, seed=0)

        # Batch with a budget high enough to avoid overflow.
        num_elements = bodies_np.shape[0]
        num_elements_arr = wp.array([num_elements], dtype=wp.int32, device=device)
        elements_arr = _make_stress_elements_array(bodies_np, device)
        batch = ContactPartitioner(
            max_num_interactions=num_elements,
            max_num_nodes=2_000,
            max_num_partitions=128,
            device=device,
            seed=0,
        )
        batch.launch(elements_arr, num_elements_arr)
        batch_id_to_partition = batch.interaction_id_to_partition.numpy()

        self.assertEqual(int(batch.has_additional_partition.numpy()[0]), 0)
        # Both paths use the same JP priorities + the same tie-breaking, so the
        # per-element colour must agree.
        self.assertTrue(
            np.array_equal(inc["interaction_id_to_partition"], batch_id_to_partition),
            msg="incremental and batch disagree on element -> partition assignment",
        )
        # Number of colours used should also match.
        self.assertEqual(inc["launches"], int(batch.num_partitions.numpy()[0]))


if __name__ == "__main__":
    wp.init()
    unittest.main()
