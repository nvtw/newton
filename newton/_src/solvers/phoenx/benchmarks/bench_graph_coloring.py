# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone benchmark for the PhoenX graph-coloring partitioner.

Replays a snapshot of one step's constraint graph (dumped by
``example_kapla_tower.py`` when ``PHOENX_DUMP_COLORING_GRAPH=<frame>``
is set) and runs one or more colouring algorithms against it.

Each algorithm produces:

    colors        Number of colours used.
    max_size      Largest color size (= longest sequential PGS step).
    ratio         colors / max_body_degree (1.0 == optimal lower bound).
    time_ms       Median time-to-color over ``--repeats`` runs.

The benchmark is fully isolated from the simulator: no physics,
no contacts, just the colouring kernels operating on the dumped
adjacency. Useful for:

    1. Iterating fast on coloring algorithm changes.
    2. Regression-testing color count + correctness across snapshots.
    3. Comparing candidate algorithms head-to-head.

Usage:

    python -m newton._src.solvers.phoenx.benchmarks.bench_graph_coloring \\
        --snapshots /tmp/coloring_snapshots/kapla_frame2.npz \\
                    /tmp/coloring_snapshots/kapla_frame10.npz \\
        --algorithms baseline_jp degree_jp \\
        --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    IncrementalContactPartitioner,
)

# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------


class Snapshot:
    """Replayable constraint-graph snapshot.

    All fields are host-side numpy arrays. :meth:`to_warp` materialises
    the GPU-resident copies needed by the partitioner.
    """

    __slots__ = ("bodies", "cost_values", "frame_index", "num_bodies", "path", "random_values")

    def __init__(self, path: Path) -> None:
        self.path = path
        data = np.load(path)
        self.bodies = data["bodies"].astype(np.int32, copy=False)
        self.cost_values = data["cost_values"].astype(np.int32, copy=False)
        self.random_values = data["random_values"].astype(np.int32, copy=False)
        self.num_bodies = int(data["num_bodies"])
        self.frame_index = int(data["frame_index"])
        if self.bodies.shape[1] != int(MAX_BODIES):
            raise ValueError(f"snapshot bodies shape {self.bodies.shape} doesn't match MAX_BODIES={int(MAX_BODIES)}")

    @property
    def num_elements(self) -> int:
        return int(self.bodies.shape[0])

    def max_body_degree(self) -> int:
        """Max number of elements incident to any single body.

        Equals the lower bound on chromatic number for the conflict
        graph — constraints sharing a body form a clique.
        """
        counts = np.zeros(self.num_bodies, dtype=np.int64)
        for j in range(int(MAX_BODIES)):
            col = self.bodies[:, j]
            mask = col >= 0
            if mask.any():
                np.add.at(counts, col[mask], 1)
        return int(counts.max(initial=0))

    def to_warp(self, device: wp.DeviceLike) -> tuple[wp.array, wp.array, np.ndarray, np.ndarray]:
        """Build the GPU-side arrays the partitioner consumes.

        Returns ``(elements_arr, num_elements_arr, cost_host, jitter_host)``.
        Costs/jitter come back on the host so callers can swap them
        before injection.
        """
        n = self.num_elements
        struct_dtype = np.dtype(
            {"names": ["bodies"], "formats": [(np.int32, int(MAX_BODIES))], "offsets": [0], "itemsize": 32}
        )
        # IncrementalContactPartitioner sizes itself from
        # max_num_interactions; pad the struct array to that capacity.
        capacity = n
        data = np.zeros(capacity, dtype=struct_dtype)
        data["bodies"][:] = -1
        data["bodies"][:n] = self.bodies
        elements = wp.from_numpy(data, dtype=ElementInteractionData, device=device)
        num_elements = wp.array([n], dtype=wp.int32, device=device)
        return elements, num_elements, self.cost_values.copy(), self.random_values.copy()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_coloring(snapshot: Snapshot, color_per_element: np.ndarray) -> None:
    """Raise if any element has the same color as a neighbour.

    Two elements neighbour iff they share a body. We walk the adjacency
    on the host (slow but unambiguous) so the check is independent of
    the colourer being tested.
    """
    n = snapshot.num_elements
    bodies = snapshot.bodies
    body_to_elements: dict[int, list[int]] = {}
    for i in range(n):
        for j in range(int(MAX_BODIES)):
            v = int(bodies[i, j])
            if v < 0:
                break
            body_to_elements.setdefault(v, []).append(i)
    for v, elems in body_to_elements.items():
        seen: dict[int, int] = {}
        for e in elems:
            c = int(color_per_element[e])
            if c < 0:
                raise AssertionError(f"element {e} has unassigned color {c}")
            if c in seen:
                other = seen[c]
                raise AssertionError(f"elements {other} and {e} both colored {c} but share body {v}")
            seen[c] = e


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


def _run_partitioner_with_costs(
    snapshot: Snapshot,
    cost_host: np.ndarray,
    jitter_host: np.ndarray,
    device: wp.DeviceLike,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Common driver: build a partitioner, push the supplied costs/jitter,
    run ``build_csr``, return ``(num_colors, color_per_element, color_starts)``.
    """
    n = snapshot.num_elements
    elements, num_elements, _, _ = snapshot.to_warp(device)

    # The partitioner needs adjacency built from the elements first;
    # ``reset()`` does that and seeds its own random_values from a
    # default RNG. We then overwrite both _random_values and
    # _cost_values with the snapshot's payloads (or any candidate
    # priorities) before running build_csr.
    p = IncrementalContactPartitioner(
        max_num_interactions=n,
        max_num_nodes=snapshot.num_bodies,
        device=device,
        use_tile_scan=True,
    )
    p.reset(elements, num_elements)

    # Inject costs + jitter. The partitioner allocates its own
    # _random_values internally; we replace them with the snapshot's
    # so multiple algorithms running on the same snapshot see the
    # same tiebreaker noise (only the cost word matters for
    # comparing algorithms).
    p._cost_values.assign(cost_host)
    p._random_values.assign(jitter_host)

    p.build_csr()
    wp.synchronize_device(device)
    num_colors = int(p.num_colors.numpy()[0])
    color_per_element = p.interaction_id_to_partition.numpy()[:n].astype(np.int32, copy=True)
    color_starts = p.color_starts.numpy()[: num_colors + 1].astype(np.int32, copy=True)
    return num_colors, color_per_element, color_starts


def algo_baseline_jp(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Current production: cost = contact_count (already in snapshot)."""
    return _run_partitioner_with_costs(
        snapshot,
        cost_host=snapshot.cost_values.copy(),
        jitter_host=snapshot.random_values.copy(),
        device=device,
    )


def algo_gpu_greedy_jp(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Phase 1 production candidate: GPU JP-MIS + greedy colour selection.

    Same partitioner (:class:`IncrementalContactPartitioner`) but
    drives :meth:`build_csr_greedy` instead of :meth:`build_csr`.
    Should match the ``host_jp_greedy`` reference (28 colours on
    Kapla) at JP-like throughput.
    """
    n = snapshot.num_elements
    elements, num_elements, _, _ = snapshot.to_warp(device)

    p = IncrementalContactPartitioner(
        max_num_interactions=n,
        max_num_nodes=snapshot.num_bodies,
        device=device,
        use_tile_scan=True,
    )
    p.reset(elements, num_elements)
    p._cost_values.assign(np.zeros(n, dtype=np.int32))
    p._random_values.assign(snapshot.random_values.copy())
    p.build_csr_greedy()
    wp.synchronize_device(device)
    num_colors = int(p.num_colors.numpy()[0])
    color_per_element = p.interaction_id_to_partition.numpy()[:n].astype(np.int32, copy=True)
    color_starts = p.color_starts.numpy()[: num_colors + 1].astype(np.int32, copy=True)
    return num_colors, color_per_element, color_starts


def algo_zero_cost_jp(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Pure-jitter JP. Sanity check: contact-count costs should at
    most match this; if zero-cost is meaningfully different, the
    contact-count cost is doing something."""
    return _run_partitioner_with_costs(
        snapshot,
        cost_host=np.zeros_like(snapshot.cost_values),
        jitter_host=snapshot.random_values.copy(),
        device=device,
    )


def algo_degree_jp(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Phase 1: cost = sum of body degrees in the conflict graph.

    Highest-degree elements are colored first — equivalent to the
    Welsh-Powell ordering imposed through the JP cost slot. Pure
    host-side computation here so the bench can iterate without
    modifying the partitioner; the production version will compute
    this on-device from ``adjacency_section_end_indices``.
    """
    n = snapshot.num_elements
    bodies = snapshot.bodies
    body_deg = np.zeros(snapshot.num_bodies, dtype=np.int64)
    for j in range(int(MAX_BODIES)):
        col = bodies[:, j]
        mask = col >= 0
        if mask.any():
            np.add.at(body_deg, col[mask], 1)
    cost = np.zeros(n, dtype=np.int64)
    for j in range(int(MAX_BODIES)):
        col = bodies[:, j]
        mask = col >= 0
        cost[mask] += body_deg[col[mask]]
    # Clamp to int32; degrees are small (max ~28 on Kapla, sum ≤ 224).
    cost_i32 = np.clip(cost, 0, np.iinfo(np.int32).max - 1).astype(np.int32)
    return _run_partitioner_with_costs(
        snapshot,
        cost_host=cost_i32,
        jitter_host=snapshot.random_values.copy(),
        device=device,
    )


def algo_degree_max_jp(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Variant of degree_jp using max(deg(b1), deg(b2)) instead of sum.

    Two elements that share a high-degree body but pair with low-
    degree bodies still need separation, so the max is a tighter
    proxy for "how constrained is this element by its hardest
    body". Cheaper to compute too.
    """
    n = snapshot.num_elements
    bodies = snapshot.bodies
    body_deg = np.zeros(snapshot.num_bodies, dtype=np.int64)
    for j in range(int(MAX_BODIES)):
        col = bodies[:, j]
        mask = col >= 0
        if mask.any():
            np.add.at(body_deg, col[mask], 1)
    cost = np.zeros(n, dtype=np.int64)
    for j in range(int(MAX_BODIES)):
        col = bodies[:, j]
        mask = col >= 0
        cost[mask] = np.maximum(cost[mask], body_deg[col[mask]])
    cost_i32 = np.clip(cost, 0, np.iinfo(np.int32).max - 1).astype(np.int32)
    return _run_partitioner_with_costs(
        snapshot,
        cost_host=cost_i32,
        jitter_host=snapshot.random_values.copy(),
        device=device,
    )


def _body_degrees(snapshot: Snapshot) -> np.ndarray:
    bodies = snapshot.bodies
    body_deg = np.zeros(snapshot.num_bodies, dtype=np.int64)
    for j in range(int(MAX_BODIES)):
        col = bodies[:, j]
        mask = col >= 0
        if mask.any():
            np.add.at(body_deg, col[mask], 1)
    return body_deg


def algo_low_degree_first_jp(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Inverse of degree_jp: low-degree elements colored first.

    Tests whether the priority direction matters. JP wants
    *parallelism per round*, not seriality, so high-degree-first
    pessimises by serialising the cliques. Low-degree-first
    spreads the easy-to-color tail early.
    """
    n = snapshot.num_elements
    bodies = snapshot.bodies
    body_deg = _body_degrees(snapshot)
    cost = np.zeros(n, dtype=np.int64)
    for j in range(int(MAX_BODIES)):
        col = bodies[:, j]
        mask = col >= 0
        cost[mask] += body_deg[col[mask]]
    # Invert: max - deg, so low-degree elements get the highest cost.
    max_cost = cost.max(initial=0)
    cost = max_cost - cost
    cost_i32 = np.clip(cost, 0, np.iinfo(np.int32).max - 1).astype(np.int32)
    return _run_partitioner_with_costs(
        snapshot,
        cost_host=cost_i32,
        jitter_host=snapshot.random_values.copy(),
        device=device,
    )


def algo_multiseed_jp(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Run zero-cost JP with several jitter seeds, pick the result
    with the fewest colors.

    Cheap, fully GPU-bound, deterministic per seed. JP variance is
    small (a few %) but free wins are free.
    """
    best_nc = 10**9
    best_cpe = None
    best_cs = None
    rng_master = np.random.default_rng(20260427)
    for _ in range(8):
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        jitter = rng.permutation(snapshot.num_elements).astype(np.int32) + 1
        nc, cpe, cs = _run_partitioner_with_costs(
            snapshot,
            cost_host=np.zeros(snapshot.num_elements, dtype=np.int32),
            jitter_host=jitter,
            device=device,
        )
        if nc < best_nc:
            best_nc, best_cpe, best_cs = nc, cpe, cs
    assert best_cpe is not None and best_cs is not None
    return best_nc, best_cpe, best_cs


# ---------------------------------------------------------------------------
# Host-side reference algorithms
#
# These run entirely on the CPU and exist purely to upper-bound what
# any GPU port would achieve. If the host algorithm doesn't beat
# baseline by a meaningful margin, there is no point porting it.
# ---------------------------------------------------------------------------


def _build_adjacency_lists(snapshot: Snapshot) -> list[list[int]]:
    """Element -> sorted list of neighbour element ids.

    Two elements are neighbours iff they share at least one body.
    Built in two passes (body -> elements, then element -> union)
    to keep it O(N * avg_degree) and skip the quadratic walk.
    """
    n = snapshot.num_elements
    bodies = snapshot.bodies
    body_to_elems: list[list[int]] = [[] for _ in range(snapshot.num_bodies)]
    for i in range(n):
        for j in range(int(MAX_BODIES)):
            v = int(bodies[i, j])
            if v < 0:
                break
            body_to_elems[v].append(i)
    adj: list[list[int]] = [[] for _ in range(n)]
    for elems in body_to_elems:
        # Each pair of elements sharing this body is a neighbour pair.
        # We build set-of-neighbours per element to dedupe across
        # multiple shared bodies.
        for a in elems:
            for b in elems:
                if a != b:
                    adj[a].append(b)
    # Dedupe per element.
    return [sorted(set(neigh)) for neigh in adj]


def algo_host_sequential_greedy(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Sequential greedy: walk elements in id order, assign smallest free color.

    Reference upper bound for what GPU greedy can achieve. NOT a JP
    method — runs entirely on the host. Slow but tight: this is what
    a parallel greedy with perfect conflict resolution converges to.
    """
    n = snapshot.num_elements
    adj = _build_adjacency_lists(snapshot)
    color = np.full(n, -1, dtype=np.int32)
    for v in range(n):
        forbidden = {int(color[u]) for u in adj[v] if color[u] >= 0}
        c = 0
        while c in forbidden:
            c += 1
        color[v] = c
    num_colors = int(color.max() + 1)
    # Build CSR-style color_starts for compatibility with the report.
    starts = np.zeros(num_colors + 1, dtype=np.int32)
    sizes = np.bincount(color, minlength=num_colors)
    starts[1:] = np.cumsum(sizes)
    return num_colors, color, starts


def algo_host_dsatur(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """DSATUR: at each step, pick uncolored vertex with highest
    saturation (= number of distinct colors in colored neighborhood),
    breaking ties by degree.

    Sequential. Optimal-ish on bounded-degree graphs.
    """
    n = snapshot.num_elements
    adj = _build_adjacency_lists(snapshot)
    deg = np.array([len(neigh) for neigh in adj], dtype=np.int32)
    color = np.full(n, -1, dtype=np.int32)
    saturation = np.zeros(n, dtype=np.int32)
    neighbor_colors: list[set[int]] = [set() for _ in range(n)]
    # Seed: pick highest-degree vertex.
    first = int(np.argmax(deg))
    color[first] = 0
    for u in adj[first]:
        if 0 not in neighbor_colors[u]:
            neighbor_colors[u].add(0)
            saturation[u] += 1
    remaining = set(range(n))
    remaining.discard(first)
    while remaining:
        # Pick uncolored vertex with max (saturation, degree).
        best_v = -1
        best_key = (-1, -1)
        for v in remaining:
            key = (int(saturation[v]), int(deg[v]))
            if key > best_key:
                best_key = key
                best_v = v
        v = best_v
        forbidden = neighbor_colors[v]
        c = 0
        while c in forbidden:
            c += 1
        color[v] = c
        for u in adj[v]:
            if c not in neighbor_colors[u]:
                neighbor_colors[u].add(c)
                saturation[u] += 1
        remaining.discard(v)
    num_colors = int(color.max() + 1)
    starts = np.zeros(num_colors + 1, dtype=np.int32)
    sizes = np.bincount(color, minlength=num_colors)
    starts[1:] = np.cumsum(sizes)
    return num_colors, color, starts


def algo_host_parallel_greedy(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Gebremedhin-Manne style parallel greedy with conflict resolution.

    Round structure:

      1. For every uncolored vertex in parallel: assign the smallest
         color not used by any *currently colored* neighbour.
      2. Detect conflicts (two neighbours with same color); the
         lower-priority one un-colors itself.
      3. Repeat until no conflicts.

    Converges in O(log N) rounds with high probability and uses the
    same color count as sequential greedy in practice. Host
    simulation here proves the upper bound; GPU port lives in
    Phase 1 of the plan if this beats baseline.
    """
    n = snapshot.num_elements
    adj = _build_adjacency_lists(snapshot)
    color = np.full(n, -1, dtype=np.int32)
    # Use the snapshot's jitter as priority for tie-breaking.
    priority = snapshot.random_values.astype(np.int64, copy=False)
    iteration = 0
    while True:
        iteration += 1
        uncolored = np.flatnonzero(color < 0)
        if uncolored.size == 0:
            break
        # Phase 1: tentative coloring (all uncolored vertices in parallel
        # would do this; here we walk in id order but only inspect
        # already-colored neighbours, which is what the parallel kernel
        # does -- each thread sees the prior-round color array).
        snapshot_color = color.copy()
        for v in uncolored:
            forbidden = {int(snapshot_color[u]) for u in adj[v] if snapshot_color[u] >= 0}
            c = 0
            while c in forbidden:
                c += 1
            color[v] = c
        # Phase 2: detect conflicts. For each edge with both endpoints
        # the same color, the lower-priority endpoint un-colors.
        conflicts = 0
        for v in uncolored:
            cv = int(color[v])
            if cv < 0:
                continue
            for u in adj[v]:
                if int(color[u]) == cv and u != v:
                    if int(priority[v]) < int(priority[u]):
                        color[v] = -1
                        conflicts += 1
                        break
        # Useful telemetry on big graphs.
        # print(f"  parallel_greedy iter={iteration} uncolored_in={uncolored.size} conflicts={conflicts}")
        if conflicts == 0:
            break
    num_colors = int(color.max() + 1)
    starts = np.zeros(num_colors + 1, dtype=np.int32)
    sizes = np.bincount(color, minlength=num_colors)
    starts[1:] = np.cumsum(sizes)
    return num_colors, color, starts


def algo_host_jp_greedy(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Hybrid: JP-MIS picks many vertices per round; each picked vertex
    greedily chooses the smallest color not used by its colored neighbours.

    Combines:
      * JP's parallelism (many per round, no conflicts within a round
        because picks are an independent set).
      * Greedy color selection (re-uses existing colors aggressively
        instead of one-color-per-round like vanilla JP).

    No conflict-resolution phase needed: the MIS guarantees no two
    picked vertices in the same round share an edge, so independent
    color choices are safe.

    Round count is the same as JP (~50-100 rounds) but each round
    can produce VERY different colors — the algorithm converges to
    near-greedy color counts at near-JP throughput.

    Reference: this is the "iterated MIS + greedy" coloring used in
    Bozdağ et al. and several KOKKOS-Kernels variants.
    """
    n = snapshot.num_elements
    adj = _build_adjacency_lists(snapshot)
    color = np.full(n, -1, dtype=np.int32)
    priority = snapshot.random_values.astype(np.int64, copy=False)
    iteration = 0
    while True:
        iteration += 1
        uncolored = np.flatnonzero(color < 0)
        if uncolored.size == 0:
            break
        # MIS pick: vertex v is in the MIS iff it has the highest
        # priority among all *uncolored* neighbours. Walks like
        # the existing JP coloring kernel.
        in_mis = np.zeros(n, dtype=bool)
        for v in uncolored:
            pv = priority[v]
            is_local_max = True
            for u in adj[v]:
                if color[u] < 0 and priority[u] > pv:
                    is_local_max = False
                    break
            if is_local_max:
                in_mis[v] = True
        # Greedy color: each MIS vertex picks the smallest color not
        # used by its colored neighbours.
        for v in np.flatnonzero(in_mis):
            forbidden = {int(color[u]) for u in adj[v] if color[u] >= 0}
            c = 0
            while c in forbidden:
                c += 1
            color[v] = c
    num_colors = int(color.max() + 1)
    starts = np.zeros(num_colors + 1, dtype=np.int32)
    sizes = np.bincount(color, minlength=num_colors)
    starts[1:] = np.cumsum(sizes)
    return num_colors, color, starts


def algo_host_jp_greedy_verbose(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Verbose variant of algo_host_jp_greedy for sizing the GPU port."""
    n = snapshot.num_elements
    adj = _build_adjacency_lists(snapshot)
    color = np.full(n, -1, dtype=np.int32)
    priority = snapshot.random_values.astype(np.int64, copy=False)
    iteration = 0
    while True:
        iteration += 1
        uncolored = np.flatnonzero(color < 0)
        if uncolored.size == 0:
            break
        in_mis = np.zeros(n, dtype=bool)
        for v in uncolored:
            pv = priority[v]
            is_local_max = True
            for u in adj[v]:
                if color[u] < 0 and priority[u] > pv:
                    is_local_max = False
                    break
            if is_local_max:
                in_mis[v] = True
        n_mis = int(in_mis.sum())
        for v in np.flatnonzero(in_mis):
            forbidden = {int(color[u]) for u in adj[v] if color[u] >= 0}
            c = 0
            while c in forbidden:
                c += 1
            color[v] = c
        nc_so_far = int(color[color >= 0].max(initial=-1)) + 1
        if iteration <= 5 or iteration % 5 == 0 or uncolored.size < 200:
            print(
                f"    [round {iteration:>3d}] uncolored_in={uncolored.size:>6d}  mis_picked={n_mis:>6d}  colors_so_far={nc_so_far}"
            )
    num_colors = int(color.max() + 1)
    starts = np.zeros(num_colors + 1, dtype=np.int32)
    sizes = np.bincount(color, minlength=num_colors)
    starts[1:] = np.cumsum(sizes)
    return num_colors, color, starts


def algo_host_parallel_greedy_verbose(snapshot: Snapshot, device: wp.DeviceLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Same as algo_host_parallel_greedy but prints round-by-round stats.

    Used to size the GPU kernel's outer loop and to verify the
    expected ~10-round convergence on Kapla.
    """
    n = snapshot.num_elements
    adj = _build_adjacency_lists(snapshot)
    color = np.full(n, -1, dtype=np.int32)
    priority = snapshot.random_values.astype(np.int64, copy=False)
    iteration = 0
    while True:
        iteration += 1
        uncolored = np.flatnonzero(color < 0)
        if uncolored.size == 0:
            break
        snapshot_color = color.copy()
        for v in uncolored:
            forbidden = {int(snapshot_color[u]) for u in adj[v] if snapshot_color[u] >= 0}
            c = 0
            while c in forbidden:
                c += 1
            color[v] = c
        conflicts = 0
        for v in uncolored:
            cv = int(color[v])
            if cv < 0:
                continue
            for u in adj[v]:
                if int(color[u]) == cv and u != v:
                    if int(priority[v]) < int(priority[u]):
                        color[v] = -1
                        conflicts += 1
                        break
        nc_so_far = int(color[color >= 0].max(initial=-1)) + 1
        print(
            f"    [round {iteration:>3d}] uncolored_in={uncolored.size:>6d}  conflicts={conflicts:>6d}  colors_so_far={nc_so_far}"
        )
        if conflicts == 0:
            break
    num_colors = int(color.max() + 1)
    starts = np.zeros(num_colors + 1, dtype=np.int32)
    sizes = np.bincount(color, minlength=num_colors)
    starts[1:] = np.cumsum(sizes)
    return num_colors, color, starts


ALGORITHMS = {
    "baseline_jp": algo_baseline_jp,
    "gpu_greedy_jp": algo_gpu_greedy_jp,
    "zero_cost_jp": algo_zero_cost_jp,
    "degree_jp": algo_degree_jp,
    "degree_max_jp": algo_degree_max_jp,
    "low_degree_first_jp": algo_low_degree_first_jp,
    "multiseed_jp": algo_multiseed_jp,
    "host_sequential_greedy": algo_host_sequential_greedy,
    # DSATUR is O(N^2) at this size — only enable on small snapshots.
    # "host_dsatur": algo_host_dsatur,
    "host_parallel_greedy": algo_host_parallel_greedy,
    "host_parallel_greedy_verbose": algo_host_parallel_greedy_verbose,
    "host_jp_greedy": algo_host_jp_greedy,
    "host_jp_greedy_verbose": algo_host_jp_greedy_verbose,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_one(
    snapshot: Snapshot,
    algo_name: str,
    device: wp.DeviceLike,
    repeats: int,
    validate: bool,
) -> dict:
    """Run one algorithm against one snapshot ``repeats`` times.

    Validates the first run, times all runs (including the validated
    one — the validation cost lives on the host, never inside the
    timed region).
    """
    fn = ALGORITHMS[algo_name]
    times_ms: list[float] = []
    num_colors = 0
    color_starts = None
    for r in range(repeats):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        nc, cpe, cs = fn(snapshot, device)
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
        if r == 0:
            num_colors = nc
            color_starts = cs
            if validate:
                validate_coloring(snapshot, cpe)
    color_sizes = np.diff(color_starts)
    return {
        "algo": algo_name,
        "snapshot": snapshot.path.name,
        "num_elements": snapshot.num_elements,
        "max_body_degree": snapshot.max_body_degree(),
        "num_colors": num_colors,
        "max_color_size": int(color_sizes.max(initial=0)),
        "ratio": num_colors / max(1, snapshot.max_body_degree()),
        "time_ms_median": statistics.median(times_ms),
        "time_ms_min": min(times_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", nargs="+", type=Path, required=True, help="Paths to .npz coloring snapshots")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
        help="Algorithms to benchmark",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per (snapshot, algorithm) pair")
    parser.add_argument("--device", default="cuda:0", help="Warp device")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the host-side coloring validity check (saves ~1s per snapshot on big graphs)",
    )
    args = parser.parse_args()

    device = wp.get_device(args.device)
    print(f"device={device.alias}")
    print()

    rows: list[dict] = []
    for snap_path in args.snapshots:
        snap = Snapshot(snap_path)
        print(
            f"=== snapshot: {snap_path.name}  n={snap.num_elements}  bodies={snap.num_bodies}  "
            f"max_body_deg={snap.max_body_degree()} ==="
        )
        for algo_name in args.algorithms:
            row = run_one(snap, algo_name, device, args.repeats, validate=not args.no_validate)
            rows.append(row)
            print(
                f"  {algo_name:<14s}  colors={row['num_colors']:>4d}  "
                f"max_size={row['max_color_size']:>5d}  "
                f"ratio={row['ratio']:>4.2f}x  "
                f"t_med={row['time_ms_median']:>6.2f}ms  t_min={row['time_ms_min']:>6.2f}ms"
            )
        print()

    # Pivot: per-algorithm summary across snapshots.
    print("=== summary (median across snapshots) ===")
    by_algo: dict[str, list[dict]] = {}
    for r in rows:
        by_algo.setdefault(r["algo"], []).append(r)
    for algo_name, group in by_algo.items():
        med_colors = statistics.median(r["num_colors"] for r in group)
        med_ratio = statistics.median(r["ratio"] for r in group)
        med_time = statistics.median(r["time_ms_median"] for r in group)
        print(
            f"  {algo_name:<14s}  med_colors={med_colors:>5.1f}  "
            f"med_ratio={med_ratio:>4.2f}x  med_time={med_time:>6.2f}ms"
        )


if __name__ == "__main__":
    main()
