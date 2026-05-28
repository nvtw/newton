# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the union-find island builder ported from PhoenX.

Structure mirrors :mod:`test_graph_coloring` -- small hand-built
scenes for the easy invariants + a synthetic stress workload that
exercises a high-degree hub vertex. A dedicated
:class:`TestIslandBuilderDeterminism` runs the builder twice on the
same inputs and byte-compares every output buffer: the underlying
union-find atomics race by design, so determinism falls out of the
post-processing trick (min-index-per-set + sort by min-index) rather
than the atomic ordering itself.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

if not wp.get_preferred_device().is_cuda:
    raise unittest.SkipTest("PhoenX tests require CUDA")

from newton._src.solvers.phoenx.islands.island_builder import (
    MAX_BODIES_PER_INTERACTION,
    UnionFindIslandBuilder,
)


def _build_interaction_bodies_array(elements_bodies: list[list[int]], capacity: int) -> wp.array:
    """Produce a ``wp.array2d[wp.int32]`` of shape ``(capacity, 8)``.

    Row ``i`` holds the bodies referenced by interaction ``i``, with
    unused slots set to ``-1``. Matches the layout
    :class:`UnionFindIslandBuilder` expects.
    """
    data = np.full((capacity, MAX_BODIES_PER_INTERACTION), -1, dtype=np.int32)
    for i, bodies in enumerate(elements_bodies):
        assert len(bodies) <= MAX_BODIES_PER_INTERACTION
        data[i, : len(bodies)] = bodies
    return wp.from_numpy(data, dtype=wp.int32)


def _run_builder(
    elements_bodies: list[list[int]],
    num_bodies: int,
    capacity: int | None = None,
) -> tuple[UnionFindIslandBuilder, dict[str, np.ndarray]]:
    """Allocate a builder, run it, return the builder + host snapshots.

    The snapshot dict contains ``numpy`` copies of every output array
    the tests care about so later comparisons can run without
    re-reading the device.
    """
    if capacity is None:
        capacity = num_bodies
    assert num_bodies <= capacity

    # Interactions buffer is sized to len(elements_bodies) -- independent
    # of body capacity because a scene often has many more constraints
    # than bodies. Round up to at least 1 so the wp.array2d never has a
    # zero-sized leading extent.
    interactions_cap = max(1, len(elements_bodies))
    interaction_bodies = _build_interaction_bodies_array(elements_bodies, interactions_cap)
    num_elements_arr = wp.array([len(elements_bodies)], dtype=wp.int32)
    num_bodies_arr = wp.array([num_bodies], dtype=wp.int32)

    builder = UnionFindIslandBuilder(num_bodies_capacity=capacity)
    builder.build_islands(interaction_bodies, num_elements_arr, num_bodies_arr)

    n = builder.num_islands()
    snapshot = {
        "num_islands": n,
        "set_nr": builder.set_nr.numpy()[:num_bodies].copy(),
        "set_sizes": builder.set_sizes.numpy()[:num_bodies].copy(),
        "set_sizes_compact": builder.set_sizes_compact.numpy()[: max(n, 1)].copy(),
    }
    return builder, snapshot


def _islands_as_sets(set_nr: np.ndarray, num_bodies: int) -> list[frozenset[int]]:
    """Group bodies by their ``set_nr`` id and return each island as a
    frozenset. Order is by island id (which is already deterministic
    after the sort step)."""
    by_island: dict[int, set[int]] = {}
    for body in range(num_bodies):
        nr = int(set_nr[body])
        by_island.setdefault(nr, set()).add(body)
    return [frozenset(by_island[k]) for k in sorted(by_island.keys())]


class TestIslandBuilder(unittest.TestCase):
    def test_four_disjoint_pairs(self):
        # Four pairs sharing no bodies -> four islands of size 2.
        elements = [[0, 1], [2, 3], [4, 5], [6, 7]]
        _builder, snap = _run_builder(elements, num_bodies=8)

        self.assertEqual(snap["num_islands"], 4)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=8)
        self.assertCountEqual(
            islands,
            [
                frozenset({0, 1}),
                frozenset({2, 3}),
                frozenset({4, 5}),
                frozenset({6, 7}),
            ],
        )
        # Every island has size 2 -> ``set_sizes_compact[i] - [i-1] == 2``.
        ends = snap["set_sizes_compact"]
        lengths = [int(ends[0])] + [int(ends[i] - ends[i - 1]) for i in range(1, snap["num_islands"])]
        self.assertEqual(lengths, [2, 2, 2, 2])

    def test_chain_one_island(self):
        # A linear chain 0-1-2-3-4 merges into one island.
        elements = [[0, 1], [1, 2], [2, 3], [3, 4]]
        _builder, snap = _run_builder(elements, num_bodies=5)

        self.assertEqual(snap["num_islands"], 1)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=5)
        self.assertEqual(islands, [frozenset({0, 1, 2, 3, 4})])
        self.assertEqual(int(snap["set_sizes_compact"][0]), 5)

    def test_star_one_island(self):
        # A star: every edge touches body 0. Classic hub that stresses
        # the atomic union-find's retry path.
        elements = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
        _builder, snap = _run_builder(elements, num_bodies=6)

        self.assertEqual(snap["num_islands"], 1)
        self.assertEqual(int(snap["set_sizes_compact"][0]), 6)

    def test_multibody_interactions(self):
        # 3- and 4-body interactions (articulated constraints).
        elements = [
            [0, 1, 2],  # 0-1-2 chain
            [3, 4, 5, 6],  # 3-4-5-6 chain
            [1, 4],  # bridges the two -> one island of {0..6}
        ]
        _builder, snap = _run_builder(elements, num_bodies=7)

        self.assertEqual(snap["num_islands"], 1)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=7)
        self.assertEqual(islands, [frozenset({0, 1, 2, 3, 4, 5, 6})])

    def test_singletons_pass_through(self):
        # Two pairs + one lone body (no interaction references it) ->
        # the lone body forms its own singleton island.
        elements = [[0, 1], [2, 3]]
        _builder, snap = _run_builder(elements, num_bodies=5)

        self.assertEqual(snap["num_islands"], 3)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=5)
        self.assertCountEqual(islands, [frozenset({0, 1}), frozenset({2, 3}), frozenset({4})])

    def test_capacity_larger_than_active(self):
        # Buffers larger than the active body count; inactive-tail
        # guards (``tid >= num_bodies[0]``) must skip those threads
        # cleanly.
        elements = [[0, 1], [1, 2]]
        _builder, snap = _run_builder(elements, num_bodies=3, capacity=16)
        self.assertEqual(snap["num_islands"], 1)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=3)
        self.assertEqual(islands, [frozenset({0, 1, 2})])

    def test_island_ids_sorted_by_min_body(self):
        # Three disjoint islands whose smallest bodies are 2, 0, 5 in
        # interaction order. The builder's post-sort must re-order
        # them by min-body-id -> island 0 is {0, 1}, island 1 is
        # {2, 3, 4}, island 2 is {5, 6}.
        elements = [
            [2, 3],
            [3, 4],
            [0, 1],
            [5, 6],
        ]
        builder, snap = _run_builder(elements, num_bodies=7)

        self.assertEqual(snap["num_islands"], 3)
        # Per-island lowest body ids via the convenience helper.
        lowest = [builder.get_island_lowest(i) for i in range(3)]
        self.assertEqual(lowest, [0, 2, 5])

        # get_island() returns body ids sorted ascending.
        self.assertEqual(builder.get_island(0), [0, 1])
        self.assertEqual(builder.get_island(1), [2, 3, 4])
        self.assertEqual(builder.get_island(2), [5, 6])


class TestIslandBuilderDeterminism(unittest.TestCase):
    """A single large stress workload run multiple times must produce
    identical outputs despite the underlying atomic union-find being
    non-deterministic.

    The check compares ``set_nr`` / ``set_sizes`` / ``set_sizes_compact``
    byte-for-byte across runs; any mismatch indicates the
    post-processing min-index-sort path has regressed.
    """

    def _stress_workload(self, seed: int, num_bodies: int = 256, num_edges: int = 1024) -> list[list[int]]:
        """Random edge-set + a few high-degree hub vertices to maximise
        concurrent union contention."""
        rng = np.random.default_rng(seed)
        elements: list[list[int]] = []
        # Random pair edges.
        for _ in range(num_edges):
            a, b = rng.choice(num_bodies, size=2, replace=False)
            elements.append([int(a), int(b)])
        # A handful of hubs with many incident edges.
        hubs = rng.choice(num_bodies, size=4, replace=False)
        for h in hubs:
            partners = rng.choice(num_bodies, size=32, replace=False)
            for p in partners:
                if int(p) == int(h):
                    continue
                elements.append([int(h), int(p)])
        return elements

    def test_determinism_small(self):
        elements = [[0, 1], [1, 2], [3, 4], [4, 5], [0, 5]]
        _, snap_a = _run_builder(elements, num_bodies=6)
        _, snap_b = _run_builder(elements, num_bodies=6)
        np.testing.assert_array_equal(snap_a["set_nr"], snap_b["set_nr"])
        np.testing.assert_array_equal(snap_a["set_sizes"], snap_b["set_sizes"])
        np.testing.assert_array_equal(snap_a["set_sizes_compact"], snap_b["set_sizes_compact"])

    def test_determinism_stress(self):
        elements = self._stress_workload(seed=12345)
        _, snap_a = _run_builder(elements, num_bodies=256)
        _, snap_b = _run_builder(elements, num_bodies=256)
        self.assertEqual(snap_a["num_islands"], snap_b["num_islands"])
        np.testing.assert_array_equal(snap_a["set_nr"], snap_b["set_nr"])
        np.testing.assert_array_equal(snap_a["set_sizes"], snap_b["set_sizes"])
        np.testing.assert_array_equal(snap_a["set_sizes_compact"], snap_b["set_sizes_compact"])

    def test_determinism_many_runs(self):
        # Run three times; all three snapshots must match. A racing
        # union-find without the min-index sort would occasionally
        # produce different ``set_nr`` labels across runs here.
        elements = self._stress_workload(seed=7, num_bodies=128, num_edges=512)
        snaps = []
        for _ in range(3):
            _, snap = _run_builder(elements, num_bodies=128)
            snaps.append(snap)
        for s in snaps[1:]:
            np.testing.assert_array_equal(snaps[0]["set_nr"], s["set_nr"])
            np.testing.assert_array_equal(snaps[0]["set_sizes"], s["set_sizes"])
            np.testing.assert_array_equal(snaps[0]["set_sizes_compact"], s["set_sizes_compact"])

    def _clustered_mixed_arity_workload(
        self,
        seed: int,
        cluster_sizes: list[int],
        pair_edges_per_cluster: int,
        triplet_edges_per_cluster: int,
        quad_edges_per_cluster: int,
        six_edges_per_cluster: int,
        hubs_per_cluster: int,
        hub_degree: int,
    ) -> tuple[list[list[int]], int]:
        """Synthetic workload with multiple disjoint clusters. Each
        cluster owns a contiguous body-id range and emits its own
        mixed-arity interactions + a few high-degree hubs that union
        within the cluster only. Returns ``(elements, num_bodies)``.

        Multiple clusters guarantee multiple islands so the
        deterministic post-sort path (min-index sort + label rewrite)
        is exercised under load -- a single-cluster workload collapses
        into one island after the first few unions and never touches
        the multi-island branch.
        """
        rng = np.random.default_rng(seed)
        elements: list[list[int]] = []
        offset = 0
        cluster_ranges: list[tuple[int, int]] = []
        for size in cluster_sizes:
            cluster_ranges.append((offset, offset + size))
            offset += size
        num_bodies = offset

        def _distinct_in(lo: int, hi: int, n: int) -> list[int]:
            return [int(x) for x in rng.choice(hi - lo, size=n, replace=False) + lo]

        for lo, hi in cluster_ranges:
            for _ in range(pair_edges_per_cluster):
                elements.append(_distinct_in(lo, hi, 2))
            for _ in range(triplet_edges_per_cluster):
                elements.append(_distinct_in(lo, hi, 3))
            for _ in range(quad_edges_per_cluster):
                elements.append(_distinct_in(lo, hi, 4))
            for _ in range(six_edges_per_cluster):
                elements.append(_distinct_in(lo, hi, MAX_BODIES_PER_INTERACTION))
            # High-degree hubs: contention path that stresses the
            # atomic union retries on the cluster's representative.
            for h in _distinct_in(lo, hi, hubs_per_cluster):
                for p in _distinct_in(lo, hi, hub_degree):
                    if int(p) == int(h):
                        continue
                    elements.append([int(h), int(p)])

        rng.shuffle(elements)
        return elements, num_bodies

    def test_determinism_large_2k_plus(self):
        """2000+ interactions across multiple disjoint clusters; the
        builder must produce byte-identical outputs across repeated
        runs and the multi-island post-sort path must be exercised.

        Six clusters of 64 bodies (384 total) with ~470 mixed-arity
        interactions per cluster + small hubs put the total well over
        2000 elements while guaranteeing six distinct islands. Both
        the union-find races (within-cluster contention) and the
        deterministic post-sort path (min-index sort, label rewrite
        across six islands) run under load.
        """
        elements, num_bodies = self._clustered_mixed_arity_workload(
            seed=2026,
            cluster_sizes=[64, 64, 64, 64, 64, 64],
            pair_edges_per_cluster=200,
            triplet_edges_per_cluster=100,
            quad_edges_per_cluster=80,
            six_edges_per_cluster=40,
            hubs_per_cluster=2,
            hub_degree=24,
        )
        self.assertGreaterEqual(
            len(elements),
            2000,
            msg=f"workload below 2k elements ({len(elements)}); raise the band sizes",
        )

        # Run six times; every snapshot must byte-match snapshot 0.
        # Six is enough to flush the kernel module cache once and run
        # several rounds with hot module state -- catches subtle race
        # paths the first run might mask.
        snaps = []
        for _ in range(6):
            _, snap = _run_builder(elements, num_bodies=num_bodies)
            snaps.append(snap)

        ref = snaps[0]
        # Sanity: every cluster collapsed into exactly one island,
        # producing six islands total. Anything else means the
        # within-cluster unions didn't converge (which determinism
        # asserts below would still catch, but this is a louder failure).
        self.assertEqual(
            ref["num_islands"],
            6,
            msg=f"expected 6 islands (one per cluster), got {ref['num_islands']}",
        )

        for idx, s in enumerate(snaps[1:], start=1):
            self.assertEqual(
                s["num_islands"],
                ref["num_islands"],
                msg=f"run {idx}: num_islands {s['num_islands']} != ref {ref['num_islands']}",
            )
            np.testing.assert_array_equal(
                ref["set_nr"],
                s["set_nr"],
                err_msg=f"run {idx}: set_nr drifted from reference",
            )
            np.testing.assert_array_equal(
                ref["set_sizes"],
                s["set_sizes"],
                err_msg=f"run {idx}: set_sizes drifted from reference",
            )
            np.testing.assert_array_equal(
                ref["set_sizes_compact"],
                s["set_sizes_compact"],
                err_msg=f"run {idx}: set_sizes_compact drifted from reference",
            )

        # Cross-check the island id <-> min-body invariant on the
        # large workload: island k's smallest body id must be strictly
        # less than island (k+1)'s. This is the property the post-sort
        # step exists to enforce.
        bodies_by_island: dict[int, int] = {}
        for body in range(num_bodies):
            nr = int(ref["set_nr"][body])
            if nr < 0:
                continue
            prev = bodies_by_island.get(nr, body)
            bodies_by_island[nr] = min(prev, body)
        min_ids = [bodies_by_island[k] for k in sorted(bodies_by_island.keys())]
        self.assertEqual(
            min_ids,
            sorted(min_ids),
            msg="island ids are not ordered by min-body-id; post-sort regressed",
        )


if __name__ == "__main__":
    unittest.main()
