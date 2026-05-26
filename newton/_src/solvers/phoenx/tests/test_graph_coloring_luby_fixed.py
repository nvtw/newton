# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`FixedIterationLubyPartitioner`.

The core correctness property mirrors :mod:`test_graph_coloring`: within
any single non-overflow colour, no two elements may share a node. The
final colour (``= max_luby_colors``) is the overflow bucket and is
exempt from this rule — mass splitting resolves its conflicts via
per-(body, partition) copy states.

CUDA-only, graph-captured where the production solver captures.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.luby_fixed import (
    FixedIterationLubyPartitioner,
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
    """Synthetic element->bodies table with a few high-degree hubs that
    force many pairwise conflicts."""
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

    # Deduplicate per-element bodies.
    for i in range(num_elements):
        uniq: list[int] = []
        for b in bodies[i]:
            if b >= 0 and int(b) not in uniq:
                uniq.append(int(b))
        row = np.full(max_bodies, -1, dtype=np.int32)
        row[: len(uniq)] = uniq
        bodies[i] = row

    return bodies


def _run_luby(
    bodies_np: np.ndarray,
    num_bodies: int,
    device,
    seed: int = 0,
    max_colored_partitions: int | None = None,
    max_luby_colors: int | None = None,
) -> dict:
    """Allocate, run :meth:`build_csr`, return host-side CSR + colour map."""
    n = bodies_np.shape[0]
    elements = _make_elements(bodies_np, device)
    num_elements_arr = wp.array([n], dtype=wp.int32, device=device)

    kwargs = dict(
        max_num_interactions=n,
        max_num_nodes=num_bodies,
        device=device,
        seed=seed,
    )
    if max_colored_partitions is not None:
        kwargs["max_colored_partitions"] = max_colored_partitions
    if max_luby_colors is not None:
        kwargs["max_luby_colors"] = max_luby_colors

    p = FixedIterationLubyPartitioner(**kwargs)
    p.reset(elements, num_elements_arr)
    p.build_csr()
    wp.synchronize_device(device)

    num_colors = int(p.num_colors.numpy()[0])
    starts = p.color_starts.numpy().copy()
    ids = p.element_ids_by_color.numpy().copy()
    eid_to_color = p.interaction_id_to_partition.numpy().copy()
    return {
        "num_colors": num_colors,
        "color_starts": starts,
        "element_ids_by_color": ids,
        "interaction_id_to_partition": eid_to_color,
        "max_luby_colors": p.max_luby_colors,
    }


class TestFixedIterationLubyPartitioner(unittest.TestCase):
    """Per-colour independent-set property + assignment consistency."""

    def _validate_within_color_disjoint(
        self,
        bodies_np: np.ndarray,
        result: dict,
        overflow_color: int,
    ) -> None:
        """Assert that within every non-overflow colour, all referenced
        nodes are distinct across elements."""
        num_colors = result["num_colors"]
        starts = result["color_starts"]
        ids = result["element_ids_by_color"]
        eid_to_color = result["interaction_id_to_partition"]

        n = bodies_np.shape[0]
        seen = np.zeros(n, dtype=bool)

        for color in range(num_colors):
            slice_ = ids[starts[color] : starts[color + 1]]
            if color == overflow_color:
                # Overflow exempt -- still record assignment for the
                # "every element assigned exactly once" check.
                for eid in slice_:
                    eid = int(eid)
                    self.assertFalse(seen[eid], msg=f"elem {eid} double-assigned")
                    seen[eid] = True
                    self.assertEqual(
                        int(eid_to_color[eid]),
                        color,
                        msg=f"elem {eid}: eid_to_color mismatch (overflow)",
                    )
                continue
            used: set[int] = set()
            for raw_eid in slice_:
                eid = int(raw_eid)
                self.assertFalse(seen[eid], msg=f"elem {eid} double-assigned")
                seen[eid] = True
                self.assertEqual(
                    int(eid_to_color[eid]),
                    color,
                    msg=f"elem {eid}: eid_to_color mismatch (colour {color})",
                )
                for b in bodies_np[eid]:
                    b = int(b)
                    if b < 0:
                        continue
                    self.assertNotIn(
                        b,
                        used,
                        msg=(f"colour {color}: node {b} shared between two elements (offender elem={eid})"),
                    )
                    used.add(b)
        self.assertTrue(seen.all(), msg="not every element was assigned a colour")

    # --- Tiny scripted graphs -----------------------------------------

    def test_disjoint_elements_single_colour(self):
        """4 elements, no shared nodes -> all land in colour 0."""
        bodies = np.full((4, int(MAX_BODIES)), -1, dtype=np.int32)
        bodies[0, :2] = [0, 1]
        bodies[1, :2] = [2, 3]
        bodies[2, :2] = [4, 5]
        bodies[3, :2] = [6, 7]
        device = wp.get_preferred_device()
        result = _run_luby(bodies, num_bodies=8, device=device, max_luby_colors=4)
        self.assertEqual(result["num_colors"], 1)
        first_slice = result["element_ids_by_color"][: result["color_starts"][1]]
        self.assertEqual(sorted(int(e) for e in first_slice), [0, 1, 2, 3])
        self._validate_within_color_disjoint(bodies, result, overflow_color=4)

    def test_clique_one_per_colour(self):
        """Star: every element touches hub body 0 -> at most 1 per
        colour; with budget 4, all 4 fit without overflow."""
        bodies = np.full((4, int(MAX_BODIES)), -1, dtype=np.int32)
        for i in range(4):
            bodies[i, 0] = 0
            bodies[i, 1] = i + 1
        device = wp.get_preferred_device()
        result = _run_luby(bodies, num_bodies=5, device=device, max_luby_colors=4)
        # Hub-induced clique requires >=4 colours; with a 4-colour budget
        # the partitioner should leave the overflow empty.
        for color in range(result["num_colors"]):
            if color == result["max_luby_colors"]:
                continue  # overflow
            count = result["color_starts"][color + 1] - result["color_starts"][color]
            self.assertLessEqual(count, 1, msg=f"colour {color}: {count} > 1 in clique")
        self._validate_within_color_disjoint(bodies, result, overflow_color=4)

    def test_overflow_under_tight_budget(self):
        """4-clique with a 2-colour budget MUST land 2 elements in the
        overflow bucket (no per-colour disjointness check on overflow)."""
        bodies = np.full((4, int(MAX_BODIES)), -1, dtype=np.int32)
        for i in range(4):
            bodies[i, 0] = 0
            bodies[i, 1] = i + 1
        device = wp.get_preferred_device()
        result = _run_luby(bodies, num_bodies=5, device=device, max_luby_colors=2)
        overflow = result["max_luby_colors"]
        # 4-clique cannot fit in 2 colours; >=2 must spill to overflow.
        overflow_count = int(result["color_starts"][overflow + 1] - result["color_starts"][overflow])
        self.assertGreaterEqual(
            overflow_count,
            2,
            msg=f"4-clique with budget=2 should overflow >=2 elements; got {overflow_count}",
        )
        # Non-overflow colours still must be independent sets.
        self._validate_within_color_disjoint(bodies, result, overflow_color=overflow)

    # --- Stress workloads --------------------------------------------

    def test_stress_small_hub_graph(self):
        bodies = _generate_stress_workload(
            num_elements=500,
            num_bodies=200,
            high_deg_nodes=5,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()
        # Generous budget so overflow stays small or empty.
        result = _run_luby(bodies, num_bodies=200, device=device, max_luby_colors=63)
        self._validate_within_color_disjoint(bodies, result, overflow_color=63)

    def test_stress_medium_hub_graph(self):
        bodies = _generate_stress_workload(
            num_elements=5_000,
            num_bodies=2_000,
            high_deg_nodes=10,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()
        result = _run_luby(bodies, num_bodies=2_000, device=device, max_luby_colors=63)
        self._validate_within_color_disjoint(bodies, result, overflow_color=63)

    def test_stress_tight_budget_overflow_only_in_overflow_bucket(self):
        """Hub-heavy workload with budget BELOW the chromatic number.
        Every conflict that doesn't fit MUST be funnelled to the overflow
        colour -- never silently into a coloured slot."""
        bodies = _generate_stress_workload(
            num_elements=2_000,
            num_bodies=500,
            high_deg_nodes=8,
            high_deg_refs=(40, 60),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()
        budget = 16  # likely well below required chromatic number
        result = _run_luby(bodies, num_bodies=500, device=device, max_luby_colors=budget)
        self._validate_within_color_disjoint(bodies, result, overflow_color=budget)

    # --- Determinism --------------------------------------------------

    def test_determinism_same_seed(self):
        bodies = _generate_stress_workload(
            num_elements=2_000,
            num_bodies=800,
            high_deg_nodes=6,
            high_deg_refs=(20, 30),
            bodies_per_elem=(2, 3),
            seed=0,
        )
        device = wp.get_preferred_device()
        r1 = _run_luby(bodies, num_bodies=800, device=device, seed=0, max_luby_colors=63)
        r2 = _run_luby(bodies, num_bodies=800, device=device, seed=0, max_luby_colors=63)
        self.assertEqual(r1["num_colors"], r2["num_colors"])
        self.assertTrue(
            np.array_equal(
                r1["interaction_id_to_partition"],
                r2["interaction_id_to_partition"],
            ),
            msg="non-deterministic per-element colour map",
        )


if __name__ == "__main__":
    unittest.main()
