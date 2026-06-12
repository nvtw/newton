# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic speculative coloring path and the
warm-start cache stir (periodic invalidate + rotating skip colour).

Three properties exercised here, in addition to the standard
"every colour is an independent set" check:

1. **Validity** -- both speculative and JP-MIS produce no within-colour
   body sharing on a representative dense synthetic graph.
2. **Determinism** -- two back-to-back runs with identical inputs
   return identical colourings (required for reproducible physics).
3. **Cache stir correctness** -- warm-start with ``rotate_skip`` /
   ``invalidate_period`` still produces valid colourings.

CUDA-only and every run is exercised under ``wp.ScopedCapture`` +
``wp.capture_launch`` -- that's the production code path and the
only one we care about. A short uncaptured warm-up build precedes
the captured build to populate state.
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


def _stress_graph(seed: int = 0) -> tuple[np.ndarray, int]:
    """Modest dense graph: small enough that all builds finish in
    well under a second on warm caches; dense enough that
    speculative coloring's "commits at multiple colours per round"
    advantage actually shows up over JP-MIS."""
    rng = np.random.default_rng(seed)
    num_elements = 600
    num_bodies = 250
    max_bodies = int(MAX_BODIES)
    hub_ids = rng.choice(num_bodies, size=4, replace=False)
    bodies = np.full((num_elements, max_bodies), -1, dtype=np.int32)
    for i in range(num_elements):
        k = int(rng.integers(2, 4))
        picks = list(rng.choice(num_bodies, size=k, replace=False))
        if i % 2 == 0:
            h = int(hub_ids[i % len(hub_ids)])
            if h not in picks:
                picks[0] = h
        bodies[i, : len(picks)] = picks
    return bodies, num_bodies


def _make_partitioner(
    n: int,
    num_bodies: int,
    device,
    *,
    enable_warm_start: bool = False,
    speculative: bool = False,
    capture_while: bool = True,
    rotate_skip: bool = False,
    invalidate_period: int = 0,
    seed: int = 0,
) -> IncrementalContactPartitioner:
    p = IncrementalContactPartitioner(
        max_num_interactions=n,
        max_num_nodes=num_bodies,
        device=device,
        seed=seed,
        use_tile_scan=True,
        enable_warm_start=enable_warm_start,
    )
    p.set_speculative_coloring(speculative)
    p.set_capture_while_greedy(capture_while)
    p.set_warm_start_rotate_skip(rotate_skip)
    p.set_warm_start_invalidate_period(invalidate_period)
    return p


def _captured_build(
    p: IncrementalContactPartitioner,
    elements: wp.array,
    num_elements_arr: wp.array,
    device,
    num_replays: int = 3,
) -> dict:
    """Run a single uncaptured warm-up build to compile kernels, then
    capture one build into a CUDA graph and replay it ``num_replays``
    times. Returns the host-side colouring snapshot after the last
    replay."""
    p.reset(elements, num_elements_arr)
    p.build_csr_greedy_with_jp_fallback()
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as cap:
        p.reset(elements, num_elements_arr)
        p.build_csr_greedy_with_jp_fallback()
    for _ in range(num_replays):
        wp.capture_launch(cap.graph)
    return {
        "num_colors": int(p.num_colors.numpy()[0]),
        "color_starts": p.color_starts.numpy().copy(),
        "element_ids_by_color": p.element_ids_by_color.numpy().copy(),
        "interaction_id_to_partition": p.interaction_id_to_partition.numpy().copy(),
    }


def _assert_valid_coloring(test: unittest.TestCase, bodies_np: np.ndarray, result: dict) -> None:
    """Every colour must be an independent set; every element assigned exactly once."""
    n = bodies_np.shape[0]
    seen = np.zeros(n, dtype=bool)
    num_colors = result["num_colors"]
    starts = result["color_starts"]
    ids = result["element_ids_by_color"]
    eid_to_color = result["interaction_id_to_partition"]
    for color in range(num_colors):
        slice_ = ids[int(starts[color]) : int(starts[color + 1])]
        used: set[int] = set()
        for raw_eid in slice_:
            eid = int(raw_eid)
            test.assertFalse(seen[eid], f"elem {eid} double-assigned (colour {color})")
            seen[eid] = True
            test.assertEqual(int(eid_to_color[eid]), color, f"elem {eid}: eid_to_color mismatch")
            for b in bodies_np[eid]:
                bi = int(b)
                if bi < 0:
                    continue
                test.assertNotIn(bi, used, f"colour {color}: body {bi} shared (elem={eid})")
                used.add(bi)
    test.assertTrue(seen.all(), f"{(~seen).sum()} elements unassigned")


class TestSpeculativeColoring(unittest.TestCase):
    @unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA required")
    def test_speculative_valid_under_capture(self) -> None:
        """Captured speculative build must produce a valid colouring
        (no within-colour body sharing) and survive multiple replays."""
        device = wp.get_preferred_device()
        bodies_np, num_bodies = _stress_graph(seed=42)
        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([bodies_np.shape[0]], dtype=wp.int32, device=device)
        p = _make_partitioner(
            bodies_np.shape[0],
            num_bodies,
            device,
            speculative=True,
            capture_while=True,
        )
        result = _captured_build(p, elements, num_elements_arr, device)
        _assert_valid_coloring(self, bodies_np, result)

    @unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA required")
    def test_speculative_deterministic_under_capture(self) -> None:
        """Two captured builds with identical inputs produce identical
        colourings -- required for reproducible physics."""
        device = wp.get_preferred_device()
        bodies_np, num_bodies = _stress_graph(seed=7)
        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([bodies_np.shape[0]], dtype=wp.int32, device=device)

        def run() -> np.ndarray:
            p = _make_partitioner(
                bodies_np.shape[0],
                num_bodies,
                device,
                speculative=True,
                capture_while=True,
                seed=123,
            )
            r = _captured_build(p, elements, num_elements_arr, device, num_replays=1)
            return r["interaction_id_to_partition"]

        a = run()
        b = run()
        np.testing.assert_array_equal(a, b, err_msg="speculative coloring not deterministic")

    @unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA required")
    def test_speculative_warm_start_under_capture(self) -> None:
        """Captured speculative build with warm-start must produce a
        valid colouring across multiple replays (the cache fills up
        after the first replay; subsequent replays exercise the
        seeded path)."""
        device = wp.get_preferred_device()
        bodies_np, num_bodies = _stress_graph(seed=11)
        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([bodies_np.shape[0]], dtype=wp.int32, device=device)
        p = _make_partitioner(
            bodies_np.shape[0],
            num_bodies,
            device,
            speculative=True,
            capture_while=True,
            enable_warm_start=True,
            seed=7,
        )
        result = _captured_build(p, elements, num_elements_arr, device, num_replays=5)
        _assert_valid_coloring(self, bodies_np, result)


class TestWarmStartStir(unittest.TestCase):
    @unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA required")
    def test_rotate_skip_valid_under_capture(self) -> None:
        """Rotate-skip with warm-start must produce a valid colouring
        across multiple captured replays. Each replay picks a
        different cached colour to re-MIS via the round-robin step
        counter."""
        device = wp.get_preferred_device()
        bodies_np, num_bodies = _stress_graph(seed=3)
        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([bodies_np.shape[0]], dtype=wp.int32, device=device)
        p = _make_partitioner(
            bodies_np.shape[0],
            num_bodies,
            device,
            enable_warm_start=True,
            capture_while=True,
            rotate_skip=True,
        )
        result = _captured_build(p, elements, num_elements_arr, device, num_replays=8)
        _assert_valid_coloring(self, bodies_np, result)

    @unittest.skipUnless(wp.get_preferred_device().is_cuda, "CUDA required")
    def test_invalidate_period_valid_under_capture(self) -> None:
        """Periodic invalidation (full cache rebuild every N steps)
        must produce a valid colouring on the rebuild step and on the
        cached steps between rebuilds. ``num_replays`` straddles two
        full-invalidate boundaries."""
        device = wp.get_preferred_device()
        bodies_np, num_bodies = _stress_graph(seed=5)
        elements = _make_elements(bodies_np, device)
        num_elements_arr = wp.array([bodies_np.shape[0]], dtype=wp.int32, device=device)
        p = _make_partitioner(
            bodies_np.shape[0],
            num_bodies,
            device,
            enable_warm_start=True,
            capture_while=True,
            invalidate_period=3,
        )
        result = _captured_build(p, elements, num_elements_arr, device, num_replays=7)
        _assert_valid_coloring(self, bodies_np, result)


if __name__ == "__main__":
    unittest.main()
