# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""No-regression check for BroadPhaseImplicitTree (speed + exact pair sets).

Run before and after a refactor and compare the printed digests and times::

    uv run python local_studies/implicit_tree_bvh/bench_broadphase.py
"""

import hashlib
import time

import numpy as np
import warp as wp
from gpu_idle import wait_gpu_idle

from newton._src.geometry.broad_phase_implicit_tree import BroadPhaseImplicitTree


def scene(case, rng):
    if case == "mixed2k":
        c = rng.uniform((-3.5, -2.5, 0.0), (3.5, 2.5, 7.0), (2000, 3))
        h = np.concatenate([rng.uniform(0.02, 0.1, (1800, 3)), rng.uniform(0.4, 0.8, (200, 3))])
    elif case == "boxes32k":
        c = rng.uniform(-8.0, 8.0, (32768, 3))
        h = rng.uniform(0.05, 0.3, (32768, 3))
    elif case == "boxes200k":
        c = rng.uniform(-40.0, 40.0, (200000, 3))
        h = rng.uniform(0.05, 0.4, (200000, 3))
    return (c - h).astype(np.float32), (c + h).astype(np.float32)


def best_graph_ms(launch, device, repeats=200, trials=7):
    with wp.ScopedCapture(device=device) as capture:
        launch()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    best = float("inf")
    for _ in range(trials):
        wait_gpu_idle()
        start = time.perf_counter()
        for _ in range(repeats):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        best = min(best, (time.perf_counter() - start) * 1000.0 / repeats)
    return best


def main():
    device = wp.get_device("cuda:0")
    for case in ("mixed2k", "boxes32k", "boxes200k"):
        rng = np.random.default_rng(7)
        lo, hi = scene(case, rng)
        n = len(lo)
        lower = wp.array(lo, dtype=wp.vec3, device=device)
        upper = wp.array(hi, dtype=wp.vec3, device=device)
        world = wp.zeros(n, dtype=wp.int32, device=device)
        group = wp.ones(n, dtype=wp.int32, device=device)
        gap = wp.array(rng.uniform(0.0, 0.02, n).astype(np.float32), dtype=wp.float32, device=device)
        tree = BroadPhaseImplicitTree(world, device=device)
        pairs = wp.empty(4_000_000, dtype=wp.vec2i, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)

        def launch(
            tree=tree, lower=lower, upper=upper, gap=gap, group=group, world=world, n=n, pairs=pairs, count=count
        ):
            tree.launch(lower, upper, gap, group, world, n, pairs, count, device=device)

        launch()
        npairs = int(count.numpy()[0])
        p = pairs.numpy()[:npairs]
        p = np.sort(np.sort(p, axis=1).view(np.int64).ravel())
        digest = hashlib.sha1(p.tobytes()).hexdigest()[:12]
        ms = best_graph_ms(launch, device)
        print(f"{case:10s} n={n:7d} pairs={npairs:8d} digest={digest} best_ms={ms:.4f}", flush=True)


if __name__ == "__main__":
    main()
