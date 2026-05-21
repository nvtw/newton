# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ConstraintClusterBuilder``.

The builder is graph-capture-only in production usage; these tests run
in eager mode for fast feedback (same pattern as
:mod:`test_graph_coloring_warm_start`). A dedicated ``TestGraphCapture``
case captures and replays through ``wp.ScopedCapture`` to exercise the
capture path explicitly.

Invocation note (per repo memory): never invoke this file via plain
``python -m unittest`` -- run

    uv run --extra dev -m newton.tests -k test_cluster_builder

otherwise the graph-capture machinery hangs.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.adjacency import ElementVertexAdjacency
from newton._src.solvers.phoenx.clustering.cluster_builder import (
    MAX_BODIES_PER_CLUSTER,
    MAX_CLUSTER_SIZE,
    ConstraintClusterBuilder,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)

# --- Test helpers ------------------------------------------------------------


def _make_elements(bodies_np: np.ndarray, device) -> wp.array:
    """Convert ``(N, MAX_BODIES)`` int32 numpy into ``ElementInteractionData`` wp array."""
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
    """Pad ragged body lists to ``(N, MAX_BODIES)`` with ``-1``."""
    max_bodies = int(MAX_BODIES)
    n = len(rows)
    out = np.full((max(n, 1), max_bodies), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        assert len(row) <= max_bodies, f"row {i} has {len(row)} > MAX_BODIES={max_bodies} bodies"
        out[i, : len(row)] = row
    return out


def _build_once(
    bodies_rows: list[list[int]],
    num_bodies: int,
    device,
    *,
    capacity: int | None = None,
    seed: int = 0,
    priorities: np.ndarray | None = None,
) -> dict:
    """Allocate a builder, run it, return host snapshots of all outputs."""
    n = len(bodies_rows)
    if capacity is None:
        capacity = max(n, 1)
    assert n <= capacity
    bodies_np = _pad_bodies(bodies_rows)
    # Pad to capacity rows so the wp.array is sized to capacity (matches
    # max_num_interactions). Unused tail rows stay all -1.
    if bodies_np.shape[0] < capacity:
        pad = np.full((capacity - bodies_np.shape[0], int(MAX_BODIES)), -1, dtype=np.int32)
        bodies_np = np.vstack([bodies_np, pad])

    elements = _make_elements(bodies_np, device)
    num_elements_arr = wp.array([n], dtype=wp.int32, device=device)
    builder = ConstraintClusterBuilder(
        max_num_interactions=capacity,
        max_num_nodes=max(num_bodies, 1),
        device=device,
        seed=seed,
    )

    pp_override = None
    if priorities is not None:
        assert priorities.shape == (capacity,)
        pp_override = wp.from_numpy(priorities.astype(np.int32), dtype=wp.int32, device=device)

    builder.build_clusters(elements, num_elements_arr, packed_priorities=pp_override)
    wp.synchronize_device(device)

    return {
        "builder": builder,
        "bodies_np": bodies_np[:n],
        "num_active": n,
        "num_clusters": int(builder.num_clusters.numpy()[0]),
        "cluster_members": builder.cluster_members.numpy().copy(),
        "element_to_cluster": builder.element_to_cluster.numpy().copy(),
    }


def _clusters_as_sets(result: dict) -> list[frozenset[int]]:
    """Convert ``cluster_members`` (vec4i) to a list of frozensets,
    sorted by min-id within each cluster for canonical comparison."""
    n_c = result["num_clusters"]
    members = result["cluster_members"]
    out = []
    for c in range(n_c):
        # vec4i layout via numpy: each row is a length-4 record.
        row = members[c]
        ids = sorted([int(v) for v in row if int(v) >= 0])
        out.append(frozenset(ids))
    out.sort(key=lambda s: min(s) if s else -1)
    return out


def _assert_invariants(test: unittest.TestCase, result: dict) -> None:
    """Universal invariants every clustering result must satisfy."""
    n = result["num_active"]
    n_c = result["num_clusters"]
    members = result["cluster_members"]
    e2c = result["element_to_cluster"]
    bodies_np = result["bodies_np"]

    # 1) Every active element is in exactly one cluster.
    seen = np.zeros(n, dtype=bool)
    for c in range(n_c):
        for v in members[c]:
            vi = int(v)
            if vi < 0:
                continue
            test.assertGreaterEqual(vi, 0)
            test.assertLess(vi, n, msg=f"cluster {c} references element {vi} past num_active={n}")
            test.assertFalse(seen[vi], msg=f"element {vi} appears in more than one cluster")
            seen[vi] = True
            test.assertEqual(
                int(e2c[vi]),
                c,
                msg=f"element {vi}: element_to_cluster says {int(e2c[vi])} but cluster {c} owns it",
            )
    test.assertTrue(
        seen.all(),
        msg=f"{int((~seen).sum())} active elements left unclustered (active={n}, clusters={n_c})",
    )

    # 2) cluster size <= MAX_CLUSTER_SIZE.
    for c in range(n_c):
        size = int(np.sum(members[c] >= 0))
        test.assertLessEqual(
            size,
            int(MAX_CLUSTER_SIZE),
            msg=f"cluster {c} has size {size} > MAX_CLUSTER_SIZE={int(MAX_CLUSTER_SIZE)}",
        )
        test.assertGreater(size, 0, msg=f"cluster {c} has zero members")

    # 3) vec4i members are sorted ascending (deterministic emission).
    for c in range(n_c):
        ids = [int(v) for v in members[c] if int(v) >= 0]
        test.assertEqual(ids, sorted(ids), msg=f"cluster {c} members not ascending: {ids}")

    # 4) Body union of each cluster <= MAX_BODIES_PER_CLUSTER.
    for c in range(n_c):
        body_union: set[int] = set()
        for v in members[c]:
            vi = int(v)
            if vi < 0:
                continue
            for b in bodies_np[vi]:
                bi = int(b)
                if bi >= 0:
                    body_union.add(bi)
        test.assertLessEqual(
            len(body_union),
            int(MAX_BODIES_PER_CLUSTER),
            msg=(
                f"cluster {c} touches {len(body_union)} bodies (cap={int(MAX_BODIES_PER_CLUSTER)}): "
                f"members={ids if c < n_c else '?'}, bodies={sorted(body_union)}"
            ),
        )

    # 5) Inactive tail of cluster_members is all (-1,-1,-1,-1).
    for c in range(n_c, len(members)):
        row = members[c]
        test.assertTrue(
            all(int(v) == -1 for v in row),
            msg=f"inactive cluster slot {c} not all-(-1): {[int(v) for v in row]}",
        )


# --- Test cases --------------------------------------------------------------


class TestClusterBuilderSmallCases(unittest.TestCase):
    """Hand-built graphs that pin down specific cluster shapes."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_empty(self) -> None:
        # capacity must be > 0 (builder asserts), but num_active = 0.
        device = wp.get_preferred_device()
        result = _build_once([], num_bodies=1, device=device, capacity=4)
        self.assertEqual(result["num_clusters"], 0)
        # Inactive tail invariant.
        for row in result["cluster_members"]:
            self.assertTrue(all(int(v) == -1 for v in row))

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_singleton(self) -> None:
        device = wp.get_preferred_device()
        result = _build_once([[10, 11]], num_bodies=12, device=device)
        self.assertEqual(result["num_clusters"], 1)
        sets = _clusters_as_sets(result)
        self.assertEqual(sets, [frozenset({0})])
        _assert_invariants(self, result)

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_disconnected_pair(self) -> None:
        device = wp.get_preferred_device()
        # Two elements that share no bodies -> two singleton clusters.
        result = _build_once(
            [[0, 1], [2, 3]],
            num_bodies=4,
            device=device,
        )
        self.assertEqual(result["num_clusters"], 2)
        sets = _clusters_as_sets(result)
        self.assertEqual(sets, [frozenset({0}), frozenset({1})])
        _assert_invariants(self, result)

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_perfect_star_with_priority_hint(self) -> None:
        """Central hub element shares one body with each of 3 leaves; with
        the hub priority-max it must claim all 3 into a single K=4
        cluster (body union = 1 hub-shared body + 6 distinct leaf bodies
        = 7 unique, within the 8 cap)."""
        device = wp.get_preferred_device()
        # elem 0: hub. bodies (100, 101, 102) -- shared one each with leaves
        # elem 1..3: leaves. Each touches the hub via one body plus one private body.
        bodies = [
            [100, 101, 102],  # hub
            [100, 200],  # leaf 1
            [101, 201],  # leaf 2
            [102, 202],  # leaf 3
        ]
        # Priorities: hub highest so it becomes the only seed.
        n = len(bodies)
        capacity = n
        priorities = np.zeros(capacity, dtype=np.int32)
        priorities[0] = 10  # hub gets max
        priorities[1] = 1
        priorities[2] = 2
        priorities[3] = 3
        result = _build_once(
            bodies,
            num_bodies=300,
            device=device,
            capacity=capacity,
            priorities=priorities,
        )
        self.assertEqual(result["num_clusters"], 1)
        sets = _clusters_as_sets(result)
        self.assertEqual(sets, [frozenset({0, 1, 2, 3})])
        _assert_invariants(self, result)

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_linear_chain_eight(self) -> None:
        """Chain of 8 constraints where elem i shares body i with elem
        i-1 and body i+1 with elem i+1. Verifies the Ks_target cascade
        absorbs odd-length tails."""
        device = wp.get_preferred_device()
        bodies = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
        ]
        result = _build_once(bodies, num_bodies=9, device=device)
        _assert_invariants(self, result)
        # Whole chain has 9 bodies but each cluster has at most 4
        # constraints touching at most 5 bodies (chain of 4 -> bodies
        # i..i+4 = 5 bodies) so the 8-body cap is non-binding here.

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_body_cap_blocks_overgrown_cluster(self) -> None:
        """4 disjoint pairs all sharing one common hub body would be a
        clique in the constraint graph, but admitting 4 of them would
        touch 9 distinct bodies (1 shared hub + 4*2 unique tails =
        9) -- the 8-body cap must force at least one split."""
        device = wp.get_preferred_device()
        # 4 elements; each is (hub_body=0, unique_tail), so they're
        # pairwise adjacent through body 0.
        bodies = [
            [0, 10, 11],  # touches hub + 2 unique
            [0, 12, 13],
            [0, 14, 15],
            [0, 16, 17],
        ]
        # Body union of all four = {0, 10..17} = 9 bodies. Must split.
        # Two members: {0, 10, 11, 12, 13} = 5 bodies, fits.
        # Three members: {0, 10..15} = 7 bodies, fits.
        # Four members: 9 bodies, exceeds cap.
        n = len(bodies)
        priorities = np.array([4, 3, 2, 1], dtype=np.int32)  # elem 0 highest

        result = _build_once(
            bodies,
            num_bodies=20,
            device=device,
            capacity=n,
            priorities=priorities,
        )
        _assert_invariants(self, result)
        # Critical assertion: no cluster touches more than 8 bodies.
        # (Already checked by _assert_invariants, but spell it out.)
        for c in range(result["num_clusters"]):
            body_union = set()
            for v in result["cluster_members"][c]:
                if int(v) < 0:
                    continue
                for b in result["bodies_np"][int(v)]:
                    if int(b) >= 0:
                        body_union.add(int(b))
            self.assertLessEqual(
                len(body_union),
                int(MAX_BODIES_PER_CLUSTER),
                msg=f"cluster {c} touches {len(body_union)} bodies",
            )


class TestClusterBuilderDeterminism(unittest.TestCase):
    """Two independent runs on the same input must produce byte-identical
    outputs. Catches stray atomic-induced non-determinism."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_repeated_runs_match(self) -> None:
        device = wp.get_preferred_device()
        # Mid-sized synthetic graph -- enough body-sharing to exercise
        # the MIS + atomic_min race resolution, small enough to be fast.
        rng = np.random.default_rng(7)
        n_elements = 200
        n_bodies = 80
        bodies_rows = []
        for _ in range(n_elements):
            k = int(rng.integers(2, 4))  # 2..3 bodies per constraint
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            bodies_rows.append(row)

        r1 = _build_once(bodies_rows, n_bodies, device, seed=0)
        r2 = _build_once(bodies_rows, n_bodies, device, seed=0)

        self.assertEqual(r1["num_clusters"], r2["num_clusters"])
        self.assertTrue(
            np.array_equal(r1["cluster_members"], r2["cluster_members"]),
            msg="cluster_members differ between runs",
        )
        self.assertTrue(
            np.array_equal(r1["element_to_cluster"], r2["element_to_cluster"]),
            msg="element_to_cluster differs between runs",
        )
        _assert_invariants(self, r1)


class TestClusterBuilderSyntheticStress(unittest.TestCase):
    """Larger workload to exercise the cascade through K=4 -> K=1."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_synthetic_dense(self) -> None:
        device = wp.get_preferred_device()
        rng = np.random.default_rng(13)
        n_elements = 500
        n_bodies = 150
        bodies_rows = []
        for _ in range(n_elements):
            k = int(rng.integers(2, 5))  # 2..4 bodies per constraint
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            bodies_rows.append(row)

        result = _build_once(bodies_rows, n_bodies, device, seed=42)
        _assert_invariants(self, result)
        # Sanity: clustering should reduce the count by at least 1/4
        # versus singleton (a typical greedy K=4 lands well below
        # n_elements / 2). Wide bound; just a smoke check.
        self.assertLess(
            result["num_clusters"],
            n_elements,
            msg="clustering produced as many clusters as elements -- nothing got grouped",
        )


class TestClusterBuilderExternalAdjacency(unittest.TestCase):
    """When the caller builds adjacency externally and passes it in, the
    result must match the internal-build path."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_external_adjacency_path_matches(self) -> None:
        device = wp.get_preferred_device()
        rng = np.random.default_rng(99)
        n_elements = 120
        n_bodies = 50
        bodies_rows = []
        for _ in range(n_elements):
            k = int(rng.integers(2, 4))
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            bodies_rows.append(row)

        bodies_np = _pad_bodies(bodies_rows)
        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([n_elements], dtype=wp.int32, device=device)

        # Internal-build path.
        b1 = ConstraintClusterBuilder(
            max_num_interactions=n_elements,
            max_num_nodes=n_bodies,
            device=device,
            seed=0,
        )
        b1.build_clusters(elements, num_elements_arr)
        wp.synchronize_device(device)
        members_internal = b1.cluster_members.numpy().copy()
        e2c_internal = b1.element_to_cluster.numpy().copy()
        num_internal = int(b1.num_clusters.numpy()[0])

        # External-build path: same priorities (same seed), but
        # adjacency built outside the cluster builder.
        adj = ElementVertexAdjacency(
            max_num_interactions=n_elements,
            max_num_nodes=n_bodies,
            device=device,
        )
        adj.build(elements, num_elements_arr)
        b2 = ConstraintClusterBuilder(
            max_num_interactions=n_elements,
            max_num_nodes=n_bodies,
            device=device,
            seed=0,
        )
        b2.build_clusters(elements, num_elements_arr, adjacency=adj)
        wp.synchronize_device(device)
        members_external = b2.cluster_members.numpy().copy()
        e2c_external = b2.element_to_cluster.numpy().copy()
        num_external = int(b2.num_clusters.numpy()[0])

        self.assertEqual(num_internal, num_external)
        self.assertTrue(np.array_equal(members_internal, members_external))
        self.assertTrue(np.array_equal(e2c_internal, e2c_external))


class TestClusterBuilderGraphCapture(unittest.TestCase):
    """Explicit ``wp.ScopedCapture`` exercise: record the build into a
    CUDA graph, replay it, and verify the outputs match an eager run."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_capture_and_replay(self) -> None:
        device = wp.get_preferred_device()
        rng = np.random.default_rng(2026)
        n_elements = 80
        n_bodies = 40
        bodies_rows = []
        for _ in range(n_elements):
            k = int(rng.integers(2, 4))
            row = list(rng.choice(n_bodies, size=k, replace=False).astype(int))
            bodies_rows.append(row)

        bodies_np = _pad_bodies(bodies_rows)
        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([n_elements], dtype=wp.int32, device=device)

        builder = ConstraintClusterBuilder(
            max_num_interactions=n_elements,
            max_num_nodes=n_bodies,
            device=device,
            seed=0,
        )

        # Warm-up (eager) so any first-launch JIT / module load happens
        # before the capture.
        builder.build_clusters(elements, num_elements_arr)
        wp.synchronize_device(device)
        eager_members = builder.cluster_members.numpy().copy()
        eager_e2c = builder.element_to_cluster.numpy().copy()
        eager_num = int(builder.num_clusters.numpy()[0])

        # Capture + replay.
        with wp.ScopedCapture(device=device) as capture:
            builder.build_clusters(elements, num_elements_arr)
        graph = capture.graph
        wp.capture_launch(graph)
        wp.synchronize_device(device)

        captured_members = builder.cluster_members.numpy().copy()
        captured_e2c = builder.element_to_cluster.numpy().copy()
        captured_num = int(builder.num_clusters.numpy()[0])

        self.assertEqual(eager_num, captured_num)
        self.assertTrue(np.array_equal(eager_members, captured_members))
        self.assertTrue(np.array_equal(eager_e2c, captured_e2c))

        result = {
            "num_active": n_elements,
            "num_clusters": captured_num,
            "cluster_members": captured_members,
            "element_to_cluster": captured_e2c,
            "bodies_np": bodies_np,
        }
        _assert_invariants(self, result)


if __name__ == "__main__":
    wp.init()
    unittest.main()
