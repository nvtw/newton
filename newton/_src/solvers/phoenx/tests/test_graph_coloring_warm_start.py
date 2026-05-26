# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cross-frame warm-start cache used by
:class:`IncrementalContactPartitioner`.

Three properties are exercised here:

1. **Cold-start equivalence**: with the cache empty (first build, or
   warm-start disabled) the colouring is identical to the version with
   warm-start disabled. Seeding only kicks in after the first
   ``build_csr_greedy`` populates the cache.
2. **Per-colour independent-set property after warm-start**: the
   second build (cache populated) must still produce a valid
   colouring -- no two elements within a non-overflow colour may
   share a body.
3. **Determinism**: two back-to-back warm-started builds with
   identical inputs produce identical outputs.

CUDA-only; the validation pass is graph-capture safe and gets used
from inside the production solver's captured per-step graph, but
these tests exercise the eager path for fast unit feedback.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    IncrementalContactPartitioner,
)


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


def _generate_stress_workload(
    num_elements: int,
    num_bodies: int,
    high_deg_nodes: int,
    high_deg_refs: tuple[int, int],
    bodies_per_elem: tuple[int, int],
    seed: int = 0,
) -> np.ndarray:
    """Synthetic element->bodies table with a few high-degree hubs."""
    rng = np.random.default_rng(seed)
    max_bodies = int(MAX_BODIES)

    hub_ids = rng.choice(num_bodies, size=high_deg_nodes, replace=False)
    hub_refs = rng.integers(high_deg_refs[0], high_deg_refs[1] + 1, size=high_deg_nodes)

    per_elem = rng.integers(bodies_per_elem[0], bodies_per_elem[1] + 1, size=num_elements)
    per_elem = np.minimum(per_elem, max_bodies)

    bodies = np.full((num_elements, max_bodies), -1, dtype=np.int32)
    for i in range(num_elements):
        k = per_elem[i]
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

    for i in range(num_elements):
        uniq: list[int] = []
        for b in bodies[i]:
            if b >= 0 and int(b) not in uniq:
                uniq.append(int(b))
        row = np.full(max_bodies, -1, dtype=np.int32)
        row[: len(uniq)] = uniq
        bodies[i] = row

    return bodies


def _make_partitioner(
    n: int,
    num_bodies: int,
    device,
    *,
    enable_warm_start: bool,
    seed: int = 0,
    max_colored_partitions: int | None = None,
) -> IncrementalContactPartitioner:
    return IncrementalContactPartitioner(
        max_num_interactions=n,
        max_num_nodes=num_bodies,
        device=device,
        seed=seed,
        use_tile_scan=True,
        max_colored_partitions=max_colored_partitions,
        enable_warm_start=enable_warm_start,
    )


def _build_once(
    bodies_np: np.ndarray,
    num_bodies: int,
    device,
    *,
    enable_warm_start: bool,
    seed: int = 0,
    max_colored_partitions: int | None = None,
) -> dict:
    """Allocate a partitioner, run ``reset`` + ``build_csr_greedy`` once,
    and return the host-side CSR + colour map."""
    n = bodies_np.shape[0]
    elements = _make_elements(bodies_np, device)
    num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

    p = _make_partitioner(
        n,
        num_bodies,
        device,
        enable_warm_start=enable_warm_start,
        seed=seed,
        max_colored_partitions=max_colored_partitions,
    )
    p.reset(elements, num_elements_arr)
    p.build_csr_greedy()
    wp.synchronize_device(device)

    return {
        "num_colors": int(p.num_colors.numpy()[0]),
        "color_starts": p.color_starts.numpy().copy(),
        "element_ids_by_color": p.element_ids_by_color.numpy().copy(),
        "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
    }


def _validate_independent_set(
    test: unittest.TestCase,
    bodies_np: np.ndarray,
    result: dict,
    *,
    overflow_color: int | None = None,
) -> None:
    """Every non-overflow colour must be an independent set, and every
    element must be assigned exactly once."""
    n = bodies_np.shape[0]
    seen = np.zeros(n, dtype=bool)
    num_colors = result["num_colors"]
    starts = result["color_starts"]
    ids = result["element_ids_by_color"]
    eid_to_color = result["interaction_id_to_partition"]

    for color in range(num_colors):
        slice_ = ids[int(starts[color]) : int(starts[color + 1])]
        is_overflow = overflow_color is not None and color == overflow_color
        used: set[int] = set()
        for raw_eid in slice_:
            eid = int(raw_eid)
            test.assertFalse(seen[eid], msg=f"elem {eid} double-assigned (colour {color})")
            seen[eid] = True
            test.assertEqual(
                int(eid_to_color[eid]),
                color,
                msg=f"elem {eid}: eid_to_color mismatch (colour {color})",
            )
            if is_overflow:
                continue
            for b in bodies_np[eid]:
                bi = int(b)
                if bi < 0:
                    continue
                test.assertNotIn(
                    bi,
                    used,
                    msg=(f"colour {color}: body {bi} shared between two elements (offender elem={eid})"),
                )
                used.add(bi)
    test.assertTrue(seen.all(), msg=f"not every element was assigned a colour ({(~seen).sum()} missing)")


class TestWarmStartColdEquivalence(unittest.TestCase):
    """First build with warm-start *enabled* must match cold-start: the
    cache is empty so the seed kernel writes nothing and every cid runs
    through the normal MIS path."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_first_build_matches_disabled(self) -> None:
        bodies_np = _generate_stress_workload(
            num_elements=1_500,
            num_bodies=600,
            high_deg_nodes=6,
            high_deg_refs=(15, 25),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()

        cold = _build_once(bodies_np, 600, device, enable_warm_start=False, seed=0)
        warm = _build_once(bodies_np, 600, device, enable_warm_start=True, seed=0)

        self.assertEqual(cold["num_colors"], warm["num_colors"])
        self.assertTrue(
            np.array_equal(cold["color_starts"], warm["color_starts"]),
            msg="color_starts differ between cold and empty-cache warm-start builds",
        )
        self.assertTrue(
            np.array_equal(
                cold["interaction_id_to_partition"],
                warm["interaction_id_to_partition"],
            ),
            msg="per-element colour assignment differs on the first warm-start build",
        )
        self.assertTrue(
            np.array_equal(
                cold["element_ids_by_color"],
                warm["element_ids_by_color"],
            ),
            msg="CSR element ordering differs on the first warm-start build",
        )


class TestWarmStartIndependentSet(unittest.TestCase):
    """After the cache is populated, a second build must still produce
    a valid colouring (per-colour independent sets, every element
    assigned exactly once)."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_second_build_valid_independent_set(self) -> None:
        bodies_np = _generate_stress_workload(
            num_elements=1_500,
            num_bodies=600,
            high_deg_nodes=6,
            high_deg_refs=(15, 25),
            bodies_per_elem=(2, 3),
            seed=1,
        )
        device = wp.get_preferred_device()
        n = bodies_np.shape[0]

        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

        p = _make_partitioner(n, 600, device, enable_warm_start=True, seed=0)
        # First build: populates the cache.
        p.reset(elements, num_elements_arr)
        p.build_csr_greedy()
        # The cache is now populated.
        wp.synchronize_device(device)
        cache_size = int(p._warm_start_cache.num_entries.numpy()[0])
        self.assertGreater(cache_size, 0, msg="warm-start cache should be populated after first build")
        self.assertLessEqual(
            cache_size,
            n,
            msg=f"cache holds {cache_size} entries > num_elements={n}",
        )

        # Second build with the same adjacency: validation pass must
        # accept (most) cached colours and the final CSR must remain a
        # valid colouring.
        p.reset(elements, num_elements_arr)
        p.build_csr_greedy()
        wp.synchronize_device(device)

        result = {
            "num_colors": int(p.num_colors.numpy()[0]),
            "color_starts": p.color_starts.numpy().copy(),
            "element_ids_by_color": p.element_ids_by_color.numpy().copy(),
            "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
        }
        _validate_independent_set(self, bodies_np, result)


class TestWarmStartDeterminism(unittest.TestCase):
    """Two back-to-back warm-started builds with identical inputs must
    produce byte-identical outputs (no atomic-induced non-determinism in
    the dedup or seed kernels)."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_repeated_warm_builds_match(self) -> None:
        bodies_np = _generate_stress_workload(
            num_elements=1_500,
            num_bodies=600,
            high_deg_nodes=6,
            high_deg_refs=(15, 25),
            bodies_per_elem=(2, 3),
            seed=2,
        )
        device = wp.get_preferred_device()
        n = bodies_np.shape[0]

        def run_two_builds() -> dict:
            elements = _make_elements(bodies_np, device)
            num_elements_arr = wp.array([n], dtype=wp.int32, device=device)
            p = _make_partitioner(n, 600, device, enable_warm_start=True, seed=0)
            p.reset(elements, num_elements_arr)
            p.build_csr_greedy()  # populates cache
            p.reset(elements, num_elements_arr)
            p.build_csr_greedy()  # warm-started
            wp.synchronize_device(device)
            return {
                "num_colors": int(p.num_colors.numpy()[0]),
                "color_starts": p.color_starts.numpy().copy(),
                "element_ids_by_color": p.element_ids_by_color.numpy().copy(),
                "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
                "cache_keys": p._warm_start_cache.keys.numpy().copy(),
                "cache_colors": p._warm_start_cache.colors.numpy().copy(),
                "cache_num_entries": int(p._warm_start_cache.num_entries.numpy()[0]),
            }

        r1 = run_two_builds()
        r2 = run_two_builds()

        self.assertEqual(r1["num_colors"], r2["num_colors"])
        self.assertEqual(r1["cache_num_entries"], r2["cache_num_entries"])
        self.assertTrue(
            np.array_equal(r1["color_starts"], r2["color_starts"]),
            msg="color_starts differ between repeated warm-started runs",
        )
        self.assertTrue(
            np.array_equal(
                r1["interaction_id_to_partition"],
                r2["interaction_id_to_partition"],
            ),
            msg="per-element colour assignment differs between repeated warm-started runs",
        )
        self.assertTrue(
            np.array_equal(
                r1["element_ids_by_color"],
                r2["element_ids_by_color"],
            ),
            msg="CSR element ordering differs between repeated warm-started runs",
        )
        # Cache content must also match (sorted-keys invariant + dedup
        # tie-break are both deterministic).
        n_entries = r1["cache_num_entries"]
        self.assertTrue(
            np.array_equal(r1["cache_keys"][:n_entries], r2["cache_keys"][:n_entries]),
            msg="warm-start cache keys differ between repeated runs",
        )
        self.assertTrue(
            np.array_equal(r1["cache_colors"][:n_entries], r2["cache_colors"][:n_entries]),
            msg="warm-start cache colours differ between repeated runs",
        )


class TestWarmStartCacheStructure(unittest.TestCase):
    """The persisted cache must be (a) sorted ascending by key, (b)
    duplicate-free, and (c) consistent with the build's
    ``interaction_id_to_partition`` for every unique body-pair signature."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_cache_sorted_and_unique(self) -> None:
        bodies_np = _generate_stress_workload(
            num_elements=800,
            num_bodies=400,
            high_deg_nodes=4,
            high_deg_refs=(10, 20),
            bodies_per_elem=(2, 3),
            seed=3,
        )
        device = wp.get_preferred_device()
        n = bodies_np.shape[0]

        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

        p = _make_partitioner(n, 400, device, enable_warm_start=True, seed=0)
        p.reset(elements, num_elements_arr)
        p.build_csr_greedy()
        wp.synchronize_device(device)

        cache_size = int(p._warm_start_cache.num_entries.numpy()[0])
        self.assertGreater(cache_size, 0)
        keys = p._warm_start_cache.keys.numpy()[:cache_size]
        colors = p._warm_start_cache.colors.numpy()[:cache_size]

        # Strictly ascending (dedup is supposed to eliminate ties).
        self.assertTrue(
            np.all(keys[1:] > keys[:-1]),
            msg=(
                "warm-start cache keys not strictly ascending -- duplicates leaked through dedup. "
                f"First violation near index {int(np.where(keys[1:] <= keys[:-1])[0][0]) if (keys[1:] <= keys[:-1]).any() else -1}"
            ),
        )

        # Each cached colour must be a valid (color+1) encoding.
        self.assertTrue(
            (colors > 0).all(),
            msg="cache contains zero / uncoloured entries -- emit kernel should have skipped them",
        )

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_cache_consistent_with_assignment(self) -> None:
        """For every unique body-pair signature in the build, the cache
        must store the (one) colour that the build actually assigned to
        a constraint with that signature."""
        bodies_np = _generate_stress_workload(
            num_elements=800,
            num_bodies=400,
            high_deg_nodes=4,
            high_deg_refs=(10, 20),
            bodies_per_elem=(2, 3),
            seed=4,
        )
        device = wp.get_preferred_device()
        n = bodies_np.shape[0]

        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

        p = _make_partitioner(n, 400, device, enable_warm_start=True, seed=0)
        p.reset(elements, num_elements_arr)
        p.build_csr_greedy()
        wp.synchronize_device(device)

        cache_size = int(p._warm_start_cache.num_entries.numpy()[0])
        keys = p._warm_start_cache.keys.numpy()[:cache_size]
        colors = p._warm_start_cache.colors.numpy()[:cache_size]
        eid_to_color = p.interaction_id_to_partition.numpy()

        # Reconstruct the host-side body-pair key (matches
        # ``warm_start_key_func`` -- two smallest non-negative bodies,
        # sentinel = INT32_MAX).
        BODY_INF = 0x7FFFFFFF

        def host_key(bodies_row: np.ndarray) -> int:
            b_min = BODY_INF
            b_2nd = BODY_INF
            for raw in bodies_row:
                b = int(raw)
                if b < 0:
                    break
                if b < b_min:
                    b_2nd = b_min
                    b_min = b
                elif b < b_2nd:
                    b_2nd = b
            return (b_2nd << 32) | (b_min & 0xFFFFFFFF)

        host_keys = np.array([host_key(bodies_np[i]) for i in range(n)], dtype=np.int64)

        # Map each unique key -> set of colours actually assigned to
        # constraints with that key.
        from collections import defaultdict

        per_key_colors: dict[int, set[int]] = defaultdict(set)
        for cid in range(n):
            color = int(eid_to_color[cid])
            if color < 0:
                continue
            per_key_colors[int(host_keys[cid])].add(color)

        # Every cache entry must reference a colour that was actually
        # used by at least one constraint with that key.
        for k, c_plus_one in zip(keys, colors, strict=True):
            assigned_set = per_key_colors[int(k)]
            self.assertGreater(
                len(assigned_set),
                0,
                msg=f"cache key {int(k):#x} not present in any assigned constraint",
            )
            self.assertIn(
                int(c_plus_one) - 1,
                assigned_set,
                msg=(
                    f"cache colour {int(c_plus_one) - 1} for key {int(k):#x} "
                    f"does not match any assigned colour {sorted(assigned_set)}"
                ),
            )


class TestWarmStartMassSplittingOverflow(unittest.TestCase):
    """With ``max_colored_partitions`` set, the overflow colour is exempt
    from validation (its same-colour neighbours are by design). Run a
    second warm-started build and verify the colouring is still valid
    on every non-overflow colour."""

    @unittest.skipUnless(wp.get_device().is_cuda, "CUDA required")
    def test_overflow_bucket_exempt_from_invalidation(self) -> None:
        # Hub-heavy workload so a tight colour budget forces overflow.
        bodies_np = _generate_stress_workload(
            num_elements=1_000,
            num_bodies=300,
            high_deg_nodes=8,
            high_deg_refs=(25, 40),
            bodies_per_elem=(2, 3),
            seed=5,
        )
        device = wp.get_preferred_device()
        n = bodies_np.shape[0]
        K = 12  # mass-splitting partition cap

        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

        p = _make_partitioner(
            n,
            300,
            device,
            enable_warm_start=True,
            seed=0,
            max_colored_partitions=K,
        )
        p.reset(elements, num_elements_arr)
        p.build_csr_greedy()  # cold
        wp.synchronize_device(device)

        # Expect overflow bucket (colour K) to receive at least one
        # element under this workload; if not, the test below is still
        # valid but doesn't exercise the exemption.
        num_colors_first = int(p.num_colors.numpy()[0])
        starts_first = p.color_starts.numpy()
        overflow_count_first = int(starts_first[K + 1] - starts_first[K]) if num_colors_first > K else 0

        p.reset(elements, num_elements_arr)
        p.build_csr_greedy()  # warm
        wp.synchronize_device(device)

        result = {
            "num_colors": int(p.num_colors.numpy()[0]),
            "color_starts": p.color_starts.numpy().copy(),
            "element_ids_by_color": p.element_ids_by_color.numpy().copy(),
            "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
        }
        _validate_independent_set(self, bodies_np, result, overflow_color=K)

        # Sanity: the budget really did push elements into overflow
        # at least on the first build. Reports a useful diagnostic when
        # the test is run on a workload that fits.
        if overflow_count_first == 0:
            self.skipTest("workload fit within K colours; overflow exemption path not exercised")


if __name__ == "__main__":
    wp.init()
    unittest.main()
