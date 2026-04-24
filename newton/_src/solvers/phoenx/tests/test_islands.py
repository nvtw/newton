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

from newton._src.solvers.phoenx.islands.island_builder import (
    MAX_BODIES_PER_INTERACTION,
    UnionFindIslandBuilder,
)


def _build_interaction_bodies_array(
    elements_bodies: list[list[int]], capacity: int
) -> wp.array:
    """Produce a ``wp.array2d[wp.int32]`` of shape ``(capacity, 8)``.

    Row ``i`` holds the bodies referenced by interaction ``i``, with
    unused slots set to ``-1``. Matches the layout
    :class:`UnionFindIslandBuilder` expects.
    """
    data = np.full(
        (capacity, MAX_BODIES_PER_INTERACTION), -1, dtype=np.int32
    )
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
    interaction_bodies = _build_interaction_bodies_array(
        elements_bodies, interactions_cap
    )
    num_elements_arr = wp.array([len(elements_bodies)], dtype=wp.int32)
    num_bodies_arr = wp.array([num_bodies], dtype=wp.int32)

    builder = UnionFindIslandBuilder(num_bodies_capacity=capacity)
    builder.build_islands(interaction_bodies, num_elements_arr, num_bodies_arr)

    n = builder.num_islands()
    snapshot = {
        "num_islands": n,
        "set_nr": builder.set_nr.numpy()[:num_bodies].copy(),
        "set_sizes": builder.set_sizes.numpy()[:num_bodies].copy(),
        "set_sizes_compact": builder.set_sizes_compact.numpy()[:max(n, 1)].copy(),
    }
    return builder, snapshot


def _islands_as_sets(
    set_nr: np.ndarray, num_bodies: int
) -> list[frozenset[int]]:
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
        builder, snap = _run_builder(elements, num_bodies=8)

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
        lengths = [int(ends[0])] + [
            int(ends[i] - ends[i - 1]) for i in range(1, snap["num_islands"])
        ]
        self.assertEqual(lengths, [2, 2, 2, 2])

    def test_chain_one_island(self):
        # A linear chain 0-1-2-3-4 merges into one island.
        elements = [[0, 1], [1, 2], [2, 3], [3, 4]]
        builder, snap = _run_builder(elements, num_bodies=5)

        self.assertEqual(snap["num_islands"], 1)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=5)
        self.assertEqual(islands, [frozenset({0, 1, 2, 3, 4})])
        self.assertEqual(int(snap["set_sizes_compact"][0]), 5)

    def test_star_one_island(self):
        # A star: every edge touches body 0. Classic hub that stresses
        # the atomic union-find's retry path.
        elements = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
        builder, snap = _run_builder(elements, num_bodies=6)

        self.assertEqual(snap["num_islands"], 1)
        self.assertEqual(int(snap["set_sizes_compact"][0]), 6)

    def test_multibody_interactions(self):
        # 3- and 4-body interactions (articulated constraints).
        elements = [
            [0, 1, 2],  # 0-1-2 chain
            [3, 4, 5, 6],  # 3-4-5-6 chain
            [1, 4],  # bridges the two -> one island of {0..6}
        ]
        builder, snap = _run_builder(elements, num_bodies=7)

        self.assertEqual(snap["num_islands"], 1)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=7)
        self.assertEqual(islands, [frozenset({0, 1, 2, 3, 4, 5, 6})])

    def test_singletons_pass_through(self):
        # Two pairs + one lone body (no interaction references it) ->
        # the lone body forms its own singleton island.
        elements = [[0, 1], [2, 3]]
        builder, snap = _run_builder(elements, num_bodies=5)

        self.assertEqual(snap["num_islands"], 3)
        islands = _islands_as_sets(snap["set_nr"], num_bodies=5)
        self.assertCountEqual(
            islands, [frozenset({0, 1}), frozenset({2, 3}), frozenset({4})]
        )

    def test_capacity_larger_than_active(self):
        # Buffers larger than the active body count; inactive-tail
        # guards (``tid >= num_bodies[0]``) must skip those threads
        # cleanly.
        elements = [[0, 1], [1, 2]]
        builder, snap = _run_builder(elements, num_bodies=3, capacity=16)
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

    def _stress_workload(
        self, seed: int, num_bodies: int = 256, num_edges: int = 1024
    ) -> list[list[int]]:
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
        np.testing.assert_array_equal(
            snap_a["set_sizes_compact"], snap_b["set_sizes_compact"]
        )

    def test_determinism_stress(self):
        elements = self._stress_workload(seed=12345)
        _, snap_a = _run_builder(elements, num_bodies=256)
        _, snap_b = _run_builder(elements, num_bodies=256)
        self.assertEqual(snap_a["num_islands"], snap_b["num_islands"])
        np.testing.assert_array_equal(snap_a["set_nr"], snap_b["set_nr"])
        np.testing.assert_array_equal(snap_a["set_sizes"], snap_b["set_sizes"])
        np.testing.assert_array_equal(
            snap_a["set_sizes_compact"], snap_b["set_sizes_compact"]
        )

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
            np.testing.assert_array_equal(
                snaps[0]["set_sizes"], s["set_sizes"]
            )
            np.testing.assert_array_equal(
                snaps[0]["set_sizes_compact"], s["set_sizes_compact"]
            )


if __name__ == "__main__":
    unittest.main()
