# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare exact GPU broad-phase pairs on Kapla and mixed-size scenes.

Run with ``uv run python scripts/benchmark_broadphase_mixed.py``. The reported
times are CUDA graph replay costs for broad phase only, excluding rendering.
"""

import argparse
import functools
import time

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_grid import BroadPhaseGrid
from newton._src.geometry.broad_phase_implicit_tree import BroadPhaseImplicitTree
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP
from newton._src.solvers.phoenx.examples.kapla_tower_data import BRICK_FULL_EXTENTS, ORIENTATIONS, POSITIONS


def kapla_bounds():
    scale = 0.1
    p = POSITIONS.astype(np.float32) * scale
    q = ORIENTATIONS.astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    r = np.empty((len(q), 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    half = np.einsum("nij,j->ni", np.abs(r), 0.5 * scale * np.asarray(BRICK_FULL_EXTENTS)) + 0.01
    return p - half, p + half


def timed_graph(launch, device, repeats=8):
    with wp.ScopedCapture(device=device) as capture:
        launch()
    for _ in range(2):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(repeats):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--cases",
    nargs="+",
    choices=("mixed100", "mixed1000", "tower2x2"),
    default=("mixed100", "mixed1000", "tower2x2"),
)
args = parser.parse_args()
device = wp.get_device("cuda:0")
base_lower, base_upper = kapla_bounds()
for case in args.cases:
    if case.startswith("mixed"):
        medium_count = int(case[5:])
        rng = np.random.default_rng(42)
        center = rng.uniform((-3.5, -2.5, 0.0), (3.5, 2.5, 7.0), (medium_count, 3)).astype(np.float32)
        half = rng.uniform(0.4, 0.8, (medium_count, 3)).astype(np.float32)
        lower_np = np.concatenate((base_lower, center - half), axis=0).astype(np.float32)
        upper_np = np.concatenate((base_upper, center + half), axis=0).astype(np.float32)
    else:
        shifts = np.array([(-4.5, -4.5, 0), (-4.5, 4.5, 0), (4.5, -4.5, 0), (4.5, 4.5, 0)], dtype=np.float32)
        lower_np = np.concatenate([base_lower + s for s in shifts])
        upper_np = np.concatenate([base_upper + s for s in shifts])
    n = len(lower_np)
    lower = wp.array(lower_np, dtype=wp.vec3, device=device)
    upper = wp.array(upper_np, dtype=wp.vec3, device=device)
    world = wp.zeros(n, dtype=wp.int32, device=device)
    group = wp.ones(n, dtype=wp.int32, device=device)
    pair_capacity = 2_000_000
    tree = BroadPhaseImplicitTree(world, queue_capacity_factor=64, device=device)
    grid = BroadPhaseGrid(world, cell_width=0.3, device=device)
    sap = BroadPhaseSAP(world, direction_search=False, device=device)
    outputs = {}
    for name, bp in [("tree", tree), ("grid", grid), ("sap", sap)]:
        pairs = wp.empty(pair_capacity, dtype=wp.vec2i, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        launch = functools.partial(bp.launch, lower, upper, None, group, world, n, pairs, count, device=device)
        launch()
        npairs = int(count.numpy()[0])
        if npairs >= pair_capacity:
            raise RuntimeError(f"{case} {name} pair overflow {npairs}")
        pair_set = {tuple(x) for x in pairs.numpy()[:npairs]}
        if len(pair_set) != npairs:
            raise RuntimeError(f"{case} {name} duplicate pairs {npairs} vs {len(pair_set)}")
        ms = timed_graph(launch, device)
        outputs[name] = (npairs, pair_set, ms)
        print(f"{case} {name}: pairs={npairs}, ms={ms:.3f}", flush=True)
    tset = outputs["tree"][1]
    for name in ("grid", "sap"):
        if tset != outputs[name][1]:
            raise RuntimeError(
                f"{case} tree != {name}: extra={len(tset - outputs[name][1])}, missing={len(outputs[name][1] - tset)}"
            )
    print(f"{case} exact equality, tree overflow={int(tree.overflow.numpy()[0])}", flush=True)
