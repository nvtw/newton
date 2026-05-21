# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``SupernodalElements``.

The supernodal emitter consumes :class:`ConstraintClusterBuilder`
output and produces one ``ElementInteractionData`` per cluster whose
``bodies`` slot is the union of its members' bodies.

Invocation: per repo memory, never run this file via plain
``python -m unittest`` -- use

    uv run --extra dev -m newton.tests -k test_supernodal_elements
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.clustering.cluster_builder import (
    MAX_BODIES_PER_CLUSTER,
    ConstraintClusterBuilder,
)
from newton._src.solvers.phoenx.clustering.supernodal_elements import SupernodalElements
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)

# --- Helpers (subset of test_cluster_builder's, kept local to avoid
# cross-test imports) ---------------------------------------------------------


def _make_elements(bodies_np: np.ndarray, device) -> wp.array:
    n = bodies_np.shape[0]
    max_bodies = int(MAX_BODIES)
    struct_dtype = np.dtype(
        {
            "names": ["bodies"],
            "formats": [(np.int32, max_bodies)],
            "offsets": [0],
            "itemsize": 4 * max_bodies,
        }
    )
    arr = np.zeros(n, dtype=struct_dtype)
    arr["bodies"] = bodies_np
    return wp.from_numpy(arr, dtype=ElementInteractionData, device=device)


def _pad_bodies(rows: list[list[int]]) -> np.ndarray:
    max_bodies = int(MAX_BODIES)
    n = len(rows)
    out = np.full((max(n, 1), max_bodies), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        assert len(row) <= max_bodies, f"row {i} has {len(row)} > MAX_BODIES={max_bodies} bodies"
        out[i, : len(row)] = row
    return out


def _cluster_and_supernodal(
    bodies_rows: list[list[int]],
    num_bodies: int,
    device,
    *,
    capacity: int | None = None,
    seed: int = 0,
    priorities: np.ndarray | None = None,
) -> dict:
    """End-to-end: cluster builder -> supernodal elements. Returns host
    snapshots of both sets of outputs."""
    n = len(bodies_rows)
    if capacity is None:
        capacity = max(n, 1)
    bodies_np = _pad_bodies(bodies_rows)
    if bodies_np.shape[0] < capacity:
        pad = np.full((capacity - bodies_np.shape[0], int(MAX_BODIES)), -1, dtype=np.int32)
        bodies_np = np.vstack([bodies_np, pad])

    elements = _make_elements(bodies_np, device)
    num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

    cb = ConstraintClusterBuilder(
        max_num_interactions=capacity,
        max_num_nodes=max(num_bodies, 1),
        device=device,
        seed=seed,
    )
    pp = None
    if priorities is not None:
        assert priorities.shape == (capacity,)
        pp = wp.from_numpy(priorities.astype(np.int32), dtype=wp.int32, device=device)
    cb.build_clusters(elements, num_elements_arr, packed_priorities=pp)

    se = SupernodalElements(max_num_clusters=capacity, device=device)
    se.build(cb.cluster_members, cb.num_clusters, elements)
    wp.synchronize_device(device)

    return {
        "cluster_builder": cb,
        "supernodal": se,
        "elements": elements,
        "num_active": n,
        "bodies_np": bodies_np[:n],
        "num_clusters": int(cb.num_clusters.numpy()[0]),
        "cluster_members": cb.cluster_members.numpy().copy(),
        "supernodal_elements_bodies": _supernodal_bodies_np(se),
        "supernodal_member_counts": se.member_counts.numpy().copy(),
    }


def _supernodal_bodies_np(se: SupernodalElements) -> np.ndarray:
    """Extract supernodal_elements.bodies as a numpy ``(N, MAX_BODIES)``
    int32 array. Avoids relying on the warp struct's host dtype layout."""
    raw = se.elements.numpy()
    # raw dtype is a record with field 'bodies' of shape (MAX_BODIES,).
    return np.ascontiguousarray(raw["bodies"], dtype=np.int32)


def _body_set(row: np.ndarray) -> set[int]:
    return {int(b) for b in row if int(b) >= 0}


# --- Tests -------------------------------------------------------------------


class TestSupernodalElementsSmallCases(unittest.TestCase):
    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_empty_active_range(self) -> None:
        """num_active = 0 -> no clusters -> every supernodal slot is
        empty (all -1)."""
        device = wp.get_preferred_device()
        result = _cluster_and_supernodal([], num_bodies=1, device=device, capacity=4)
        self.assertEqual(result["num_clusters"], 0)
        sb = result["supernodal_elements_bodies"]
        self.assertTrue(
            (sb == -1).all(),
            msg=f"empty-input supernodal slots should be all -1, got {sb}",
        )
        mc = result["supernodal_member_counts"]
        self.assertTrue((mc == 0).all())

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_singleton_supernodal_equals_constraint(self) -> None:
        """1 element -> 1 cluster of size 1 -> supernodal element's body
        set equals the original constraint's body set."""
        device = wp.get_preferred_device()
        bodies = [[7, 13, 21]]
        result = _cluster_and_supernodal(bodies, num_bodies=22, device=device)
        self.assertEqual(result["num_clusters"], 1)
        self.assertEqual(int(result["supernodal_member_counts"][0]), 1)
        # The supernodal element's body set must equal the original.
        self.assertEqual(
            _body_set(result["supernodal_elements_bodies"][0]),
            {7, 13, 21},
        )

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_perfect_star_union(self) -> None:
        """Hub-with-3-leaves cluster: supernodal body union should be the
        full union of all 4 members' bodies."""
        device = wp.get_preferred_device()
        # Same layout as test_cluster_builder.test_perfect_star_with_priority_hint:
        # hub touches 100,101,102; each leaf touches one of those plus a
        # unique private body.
        bodies = [
            [100, 101, 102],  # hub
            [100, 200],
            [101, 201],
            [102, 202],
        ]
        n = len(bodies)
        priorities = np.array([10, 1, 2, 3], dtype=np.int32)  # hub priority-max
        result = _cluster_and_supernodal(
            bodies,
            num_bodies=300,
            device=device,
            capacity=n,
            priorities=priorities,
        )
        self.assertEqual(result["num_clusters"], 1)
        self.assertEqual(int(result["supernodal_member_counts"][0]), 4)
        expected = {100, 101, 102, 200, 201, 202}
        actual = _body_set(result["supernodal_elements_bodies"][0])
        self.assertEqual(actual, expected)
        self.assertLessEqual(
            len(actual),
            int(MAX_BODIES_PER_CLUSTER),
            msg="supernodal body union exceeded MAX_BODIES_PER_CLUSTER",
        )

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_disconnected_singletons(self) -> None:
        """Two disconnected constraints -> two singleton clusters ->
        each supernodal element matches its lone member."""
        device = wp.get_preferred_device()
        bodies = [[0, 1], [10, 11]]
        result = _cluster_and_supernodal(bodies, num_bodies=12, device=device)
        self.assertEqual(result["num_clusters"], 2)
        # cluster_members are emitted with ascending member ids; cluster
        # 0 holds {0}, cluster 1 holds {1}.
        member_sets = []
        for c in range(2):
            row = result["cluster_members"][c]
            members = sorted([int(v) for v in row if int(v) >= 0])
            member_sets.append(members)
        member_sets.sort()
        self.assertEqual(member_sets, [[0], [1]])
        # Supernodal body union equals the lone-member body set, per cluster.
        for c in range(2):
            row = result["cluster_members"][c]
            lone_member = next(int(v) for v in row if int(v) >= 0)
            expected_bodies = _body_set(result["bodies_np"][lone_member])
            actual_bodies = _body_set(result["supernodal_elements_bodies"][c])
            self.assertEqual(actual_bodies, expected_bodies)

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_inactive_tail_is_empty(self) -> None:
        """Slots past num_clusters[0] must be the empty element (all -1)."""
        device = wp.get_preferred_device()
        bodies = [[7, 13], [20, 21]]
        result = _cluster_and_supernodal(
            bodies,
            num_bodies=22,
            device=device,
            capacity=8,  # spare capacity > num clusters
        )
        nc = result["num_clusters"]
        sb = result["supernodal_elements_bodies"]
        # Tail must be all -1.
        self.assertTrue(
            (sb[nc:] == -1).all(),
            msg=f"supernodal tail past num_clusters={nc} should be -1, got {sb[nc:]}",
        )
        mc = result["supernodal_member_counts"]
        self.assertTrue((mc[nc:] == 0).all())


class TestSupernodalElementsInvariants(unittest.TestCase):
    """Invariants that must hold for any valid clustering."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_synthetic_invariants(self) -> None:
        device = wp.get_preferred_device()
        rng = np.random.default_rng(31)
        n_elements = 300
        n_bodies = 100
        bodies_rows = []
        for _ in range(n_elements):
            k = int(rng.integers(2, 4))
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            bodies_rows.append(row)

        result = _cluster_and_supernodal(bodies_rows, n_bodies, device, seed=0)

        nc = result["num_clusters"]
        members = result["cluster_members"]
        sb = result["supernodal_elements_bodies"]
        mc = result["supernodal_member_counts"]
        bodies_np = result["bodies_np"]

        for c in range(nc):
            # Recompute expected union from raw members.
            expected: set[int] = set()
            count = 0
            for v in members[c]:
                vi = int(v)
                if vi < 0:
                    continue
                count += 1
                for b in bodies_np[vi]:
                    bi = int(b)
                    if bi >= 0:
                        expected.add(bi)
            actual = _body_set(sb[c])
            self.assertEqual(
                actual,
                expected,
                msg=(
                    f"cluster {c}: supernodal body union mismatch. "
                    f"expected={sorted(expected)} actual={sorted(actual)}, "
                    f"members={[int(v) for v in members[c] if int(v) >= 0]}"
                ),
            )
            self.assertEqual(
                int(mc[c]),
                count,
                msg=f"cluster {c}: member_counts={int(mc[c])} but actual members={count}",
            )
            self.assertLessEqual(
                len(actual),
                int(MAX_BODIES_PER_CLUSTER),
                msg=f"cluster {c} supernodal body union {len(actual)} > 8",
            )
            # Each supernodal body must come from at least one member.
            for b in actual:
                covered = False
                for v in members[c]:
                    vi = int(v)
                    if vi < 0:
                        continue
                    if b in {int(x) for x in bodies_np[vi] if int(x) >= 0}:
                        covered = True
                        break
                self.assertTrue(
                    covered,
                    msg=f"cluster {c}: supernodal body {b} not present in any member",
                )


class TestSupernodalElementsDeterminism(unittest.TestCase):
    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_repeated_runs_match(self) -> None:
        device = wp.get_preferred_device()
        rng = np.random.default_rng(11)
        n_elements = 200
        n_bodies = 80
        bodies_rows = []
        for _ in range(n_elements):
            k = int(rng.integers(2, 4))
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            bodies_rows.append(row)

        r1 = _cluster_and_supernodal(bodies_rows, n_bodies, device, seed=0)
        r2 = _cluster_and_supernodal(bodies_rows, n_bodies, device, seed=0)

        self.assertEqual(r1["num_clusters"], r2["num_clusters"])
        self.assertTrue(
            np.array_equal(r1["supernodal_elements_bodies"], r2["supernodal_elements_bodies"]),
            msg="supernodal element bodies differ between runs",
        )
        self.assertTrue(
            np.array_equal(r1["supernodal_member_counts"], r2["supernodal_member_counts"]),
            msg="supernodal member counts differ between runs",
        )


class TestSupernodalElementsFeedsPartitioner(unittest.TestCase):
    """Smoke test: feed the supernodal elements through the existing
    ContactPartitioner and verify the result is a valid colouring (each
    colour is an independent set in the supernodal adjacency graph)."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_partitioner_colors_supernodal_elements(self) -> None:
        device = wp.get_preferred_device()
        rng = np.random.default_rng(7)
        n_elements = 150
        n_bodies = 60
        bodies_rows = []
        for _ in range(n_elements):
            k = int(rng.integers(2, 4))
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            bodies_rows.append(row)

        result = _cluster_and_supernodal(bodies_rows, n_bodies, device, seed=0)

        nc = result["num_clusters"]
        se = result["supernodal"]
        sb = result["supernodal_elements_bodies"]

        # Lazy import to avoid circular concerns; this is the existing
        # batch-mode partitioner used as a reference.
        from newton._src.solvers.phoenx.graph_coloring.graph_coloring import (  # noqa: PLC0415
            ContactPartitioner,
        )

        p = ContactPartitioner(
            max_num_interactions=se.max_num_clusters,
            max_num_nodes=n_bodies,
            max_num_partitions=64,
            device=device,
            seed=0,
        )
        p.launch(se.elements, se.num_clusters)
        wp.synchronize_device(device)

        # Per-cluster colour assignment.
        eid_to_color = p.interaction_id_to_partition.numpy()

        # Invariant: within each non-overflow colour, no two clusters
        # share a body in their supernodal body union.
        for c in range(nc):
            color = int(eid_to_color[c])
            self.assertGreaterEqual(color, 0, msg=f"cluster {c} not assigned a colour")

        # Group clusters by colour and check the IS property.
        color_to_clusters: dict[int, list[int]] = {}
        for c in range(nc):
            color_to_clusters.setdefault(int(eid_to_color[c]), []).append(c)

        for color, members in color_to_clusters.items():
            used_bodies: set[int] = set()
            for c in members:
                bs = _body_set(sb[c])
                overlap = used_bodies & bs
                self.assertFalse(
                    overlap,
                    msg=(
                        f"colour {color}: cluster {c} shares body(s) {overlap} "
                        f"with an earlier cluster in the same colour"
                    ),
                )
                used_bodies.update(bs)


if __name__ == "__main__":
    wp.init()
    unittest.main()
