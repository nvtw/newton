# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Overflow-bucket coloring tests for :class:`IncrementalContactPartitioner`.

Soft-cap behaviour (Step 2 of the mass-splitting integration):

* ``max_colored_partitions=K`` instructs the partitioner to produce
  colours ``0..K-1`` under normal MIS rules. Any remaining elements
  that couldn't be coloured in that budget are dumped into colour
  ``K`` — the overflow bucket — without MIS-independence guarantees.
* Colours ``0..K-1`` must still be valid independent sets (the
  per-colour PGS sweep relies on this).
* Colour ``K`` is allowed to share bodies between elements (mass
  splitting resolves those conflicts via copy states).
* Total assignment must still be a partition: every element appears
  in exactly one colour.

Tests cover both the greedy and JP coloring paths.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    GREEDY_MAX_COLORS,
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    MAX_COLORS,
    IncrementalContactPartitioner,
)


def _make_elements_array(bodies_per_elem: list[list[int]], device) -> wp.array:
    """Pack a Python list of per-element body lists into the Warp struct array."""
    n = len(bodies_per_elem)
    max_bodies = int(MAX_BODIES)
    struct_dtype = np.dtype(
        {"names": ["bodies"], "formats": [(np.int32, max_bodies)], "offsets": [0], "itemsize": 4 * max_bodies}
    )
    arr = np.zeros(n, dtype=struct_dtype)
    arr["bodies"][:] = -1
    for i, blist in enumerate(bodies_per_elem):
        assert len(blist) <= max_bodies
        arr["bodies"][i, : len(blist)] = blist
    return wp.from_numpy(arr, dtype=ElementInteractionData, device=device)


def _hub_clique_elements(num_elements: int) -> list[list[int]]:
    """One hub body shared across every element + a unique partner per element.
    Creates a clique on the hub: the only valid coloring assigns each element
    its own colour, so MIS needs ``num_elements`` colours."""
    return [[0, 1 + i] for i in range(num_elements)]


def _validate_colored_independence(
    bodies_per_elem: list[list[int]],
    element_ids_by_color: np.ndarray,
    color_starts: np.ndarray,
    max_colored_partitions: int,
    num_colors: int,
) -> None:
    """For each colour < K, assert the colour is an independent set in the
    body-sharing graph. The overflow colour K is allowed to share bodies."""
    for c in range(min(num_colors, max_colored_partitions)):
        start = int(color_starts[c])
        end = int(color_starts[c + 1])
        used: set[int] = set()
        for slot in range(start, end):
            eid = int(element_ids_by_color[slot])
            for b in bodies_per_elem[eid]:
                if b < 0:
                    continue
                assert b not in used, f"colour {c}: body {b} appears in two different elements (eids around {eid})"
                used.add(b)


def _validate_full_partition(
    n_elements: int,
    element_ids_by_color: np.ndarray,
    color_starts: np.ndarray,
    num_colors: int,
    interaction_id_to_partition: np.ndarray,
) -> None:
    """Every element must appear in exactly one colour and
    interaction_id_to_partition must agree with the CSR layout."""
    seen = np.zeros(n_elements, dtype=bool)
    for c in range(num_colors):
        start = int(color_starts[c])
        end = int(color_starts[c + 1])
        for slot in range(start, end):
            eid = int(element_ids_by_color[slot])
            assert not seen[eid], f"element {eid} appears in two colours"
            seen[eid] = True
            assert int(interaction_id_to_partition[eid]) == c, (
                f"element {eid}: interaction_id_to_partition mismatch (csr={c}, "
                f"recorded={int(interaction_id_to_partition[eid])})"
            )
    assert seen.all(), f"{(~seen).sum()} elements not assigned"


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX tests are CUDA-only.",
)
class TestColoringOverflowBucket(unittest.TestCase):
    def _run_greedy(
        self,
        bodies_per_elem: list[list[int]],
        max_colored_partitions: int,
        num_bodies: int,
    ):
        device = wp.get_preferred_device()
        n = len(bodies_per_elem)
        elements_arr = _make_elements_array(bodies_per_elem, device)
        num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

        p = IncrementalContactPartitioner(
            max_num_interactions=n,
            max_num_nodes=num_bodies,
            device=device,
            seed=0,
            use_tile_scan=True,
            max_colored_partitions=max_colored_partitions,
        )
        p.reset(elements_arr, num_elements_arr)

        # Warm-up then capture for the load-bearing graph-capture assertion.
        p.build_csr_greedy_with_jp_fallback()

        with wp.ScopedCapture(device=device) as capture:
            p.build_csr_greedy_with_jp_fallback()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        return {
            "num_colors": int(p.num_colors.numpy()[0]),
            "color_starts": p.color_starts.numpy().copy(),
            "element_ids_by_color": p.element_ids_by_color.numpy().copy(),
            "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
        }

    def _run_jp(
        self,
        bodies_per_elem: list[list[int]],
        max_colored_partitions: int,
        num_bodies: int,
    ):
        device = wp.get_preferred_device()
        n = len(bodies_per_elem)
        elements_arr = _make_elements_array(bodies_per_elem, device)
        num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

        p = IncrementalContactPartitioner(
            max_num_interactions=n,
            max_num_nodes=num_bodies,
            device=device,
            seed=0,
            use_tile_scan=True,
            max_colored_partitions=max_colored_partitions,
        )
        p.reset(elements_arr, num_elements_arr)

        # Warm-up + capture.
        p.build_csr()
        with wp.ScopedCapture(device=device) as capture:
            p.build_csr()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        return {
            "num_colors": int(p.num_colors.numpy()[0]),
            "color_starts": p.color_starts.numpy().copy(),
            "element_ids_by_color": p.element_ids_by_color.numpy().copy(),
            "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
        }

    def test_greedy_no_overflow_when_under_cap(self):
        # K=4 colours, three pairwise-disjoint elements — fits in colour 0
        # alone, no overflow.
        bodies = [[0, 1], [2, 3], [4, 5]]
        result = self._run_greedy(bodies, max_colored_partitions=4, num_bodies=6)
        # All three elements colour 0; bucket K=4 empty.
        self.assertLessEqual(result["num_colors"], 4)
        _validate_full_partition(
            n_elements=3,
            element_ids_by_color=result["element_ids_by_color"],
            color_starts=result["color_starts"],
            num_colors=result["num_colors"],
            interaction_id_to_partition=result["interaction_id_to_partition"],
        )
        # Overflow bucket (slot K=4) should be empty.
        overflow_start = int(result["color_starts"][4])
        overflow_end = int(result["color_starts"][5]) if result["num_colors"] >= 5 else overflow_start
        self.assertEqual(overflow_end - overflow_start, 0)

    def test_greedy_hub_clique_overflows_to_bucket(self):
        # 12 elements share body 0 (= hub). MIS needs 12 colours.
        # With K=4 cap: colours 0..3 each get one element; remaining 8
        # land in overflow colour 4.
        n = 12
        K = 4
        bodies = _hub_clique_elements(n)
        result = self._run_greedy(bodies, max_colored_partitions=K, num_bodies=n + 1)

        # Coloring produced K + 1 colours (0..K-1 normal + K overflow).
        self.assertEqual(result["num_colors"], K + 1)

        # Colours 0..K-1 are independent sets.
        _validate_colored_independence(
            bodies_per_elem=bodies,
            element_ids_by_color=result["element_ids_by_color"],
            color_starts=result["color_starts"],
            max_colored_partitions=K,
            num_colors=result["num_colors"],
        )
        _validate_full_partition(
            n_elements=n,
            element_ids_by_color=result["element_ids_by_color"],
            color_starts=result["color_starts"],
            num_colors=result["num_colors"],
            interaction_id_to_partition=result["interaction_id_to_partition"],
        )

        # Overflow bucket (colour K) contains the remainder. The hub
        # clique forces 1 element per colour into [0..K-1], so K-1
        # elements are in normal colours and the rest in overflow.
        overflow_size = int(result["color_starts"][K + 1]) - int(result["color_starts"][K])
        self.assertEqual(overflow_size, n - K)

    def test_jp_hub_clique_overflows_to_bucket(self):
        # Same scenario via the JP path.
        n = 12
        K = 4
        bodies = _hub_clique_elements(n)
        result = self._run_jp(bodies, max_colored_partitions=K, num_bodies=n + 1)

        self.assertEqual(result["num_colors"], K + 1)
        _validate_colored_independence(
            bodies_per_elem=bodies,
            element_ids_by_color=result["element_ids_by_color"],
            color_starts=result["color_starts"],
            max_colored_partitions=K,
            num_colors=result["num_colors"],
        )
        _validate_full_partition(
            n_elements=n,
            element_ids_by_color=result["element_ids_by_color"],
            color_starts=result["color_starts"],
            num_colors=result["num_colors"],
            interaction_id_to_partition=result["interaction_id_to_partition"],
        )
        overflow_size = int(result["color_starts"][K + 1]) - int(result["color_starts"][K])
        self.assertEqual(overflow_size, n - K)

    def test_none_default_preserves_legacy_behaviour(self):
        # No cap → same as the existing graph_coloring tests: 12-clique
        # forces 12 distinct colours, no overflow bucket.
        n = 12
        bodies = _hub_clique_elements(n)
        device = wp.get_preferred_device()
        elements_arr = _make_elements_array(bodies, device)
        num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

        p = IncrementalContactPartitioner(
            max_num_interactions=n,
            max_num_nodes=n + 1,
            device=device,
            seed=0,
            use_tile_scan=True,
            # max_colored_partitions left at default None → legacy
            # error-on-overflow behaviour.
        )
        p.reset(elements_arr, num_elements_arr)
        p.build_csr_greedy_with_jp_fallback()
        self.assertEqual(int(p.num_colors.numpy()[0]), n)
        # No overflow flag must have been raised.
        self.assertEqual(int(p._overflow_flag.numpy()[0]), 0)

    def test_validation_rejects_negative_cap(self):
        with self.assertRaises(ValueError):
            IncrementalContactPartitioner(
                max_num_interactions=8,
                max_num_nodes=8,
                max_colored_partitions=-1,
            )

    def test_validation_rejects_cap_at_or_above_greedy_max(self):
        with self.assertRaises(ValueError):
            IncrementalContactPartitioner(
                max_num_interactions=8,
                max_num_nodes=8,
                max_colored_partitions=int(GREEDY_MAX_COLORS),
            )

    def test_validation_rejects_cap_at_or_above_max_colors(self):
        with self.assertRaises(ValueError):
            IncrementalContactPartitioner(
                max_num_interactions=8,
                max_num_nodes=8,
                max_colored_partitions=MAX_COLORS,
            )

    def test_overflow_bucket_distinct_per_element(self):
        # Sanity: the overflow bucket's elements list should be a
        # permutation of the un-colourable elements; no duplicates.
        n = 20
        K = 3
        bodies = _hub_clique_elements(n)
        result = self._run_greedy(bodies, max_colored_partitions=K, num_bodies=n + 1)
        ov_start = int(result["color_starts"][K])
        ov_end = int(result["color_starts"][K + 1])
        ov_ids = result["element_ids_by_color"][ov_start:ov_end]
        self.assertEqual(len(ov_ids), len(set(ov_ids.tolist())))


if __name__ == "__main__":
    wp.init()
    unittest.main()
