# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone benchmark: clustered vs direct graph colouring.

Replays a snapshot of one step's constraint graph (dumped by
``example_soft_body_drop.py`` or ``example_kapla_tower.py`` when
``PHOENX_DUMP_COLORING_GRAPH=<frame>`` is set) and compares:

    baseline  Direct colouring on the per-constraint element graph.
    clustered ConstraintClusterBuilder -> SupernodalElements -> coloring
              on the supernodal graph (one element per cluster).

For each path the bench reports the colour count plus a per-call
median time over ``--repeats`` runs. The clustered timing is split
into build / supernodal / colour so the dominant cost is visible.

Usage:

    python -m newton._src.solvers.phoenx.benchmarks.bench_clustering \\
        --snapshots /tmp/soft_body_drop_graph.npz \\
        --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_graph_coloring import Snapshot
from newton._src.solvers.phoenx.clustering.cluster_builder import (
    MAX_CLUSTER_SIZE,
    ConstraintClusterBuilder,
)
from newton._src.solvers.phoenx.clustering.supernodal_elements import SupernodalElements
from newton._src.solvers.phoenx.graph_coloring.graph_coloring import ContactPartitioner


def _median_ms(samples: list[float]) -> float:
    """Median in milliseconds. Empty -> NaN sentinel (-1)."""
    return statistics.median(samples) * 1000.0 if samples else -1.0


def _time_block(fn, repeats: int, device) -> tuple[float, float]:
    """Run ``fn`` ``repeats`` times. Returns ``(median_ms, total_ms)``."""
    samples: list[float] = []
    for _ in range(repeats):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        samples.append(time.perf_counter() - t0)
    return _median_ms(samples), sum(samples) * 1000.0


def _baseline_colors(snapshot: Snapshot, device, repeats: int) -> dict:
    """Direct colouring on the per-constraint graph."""
    n = snapshot.num_elements
    elements, num_elements, _, _ = snapshot.to_warp(device)
    # Use ContactPartitioner (batch JP-MIS) for the apples-to-apples
    # comparison: same algorithm used for the supernodal path.
    partitioner = ContactPartitioner(
        max_num_interactions=n,
        max_num_nodes=snapshot.num_bodies,
        max_num_partitions=128,
        device=device,
    )
    # Warm-up.
    partitioner.launch(elements, num_elements)
    wp.synchronize_device(device)
    median_ms, _ = _time_block(
        lambda: partitioner.launch(elements, num_elements),
        repeats,
        device,
    )
    num_colors = int(partitioner.num_partitions.numpy()[0])
    has_overflow = bool(int(partitioner.has_additional_partition.numpy()[0]))
    return {
        "num_elements": n,
        "num_colors": num_colors,
        "has_overflow": has_overflow,
        "color_time_ms": median_ms,
    }


def _clustered_colors(snapshot: Snapshot, device, repeats: int) -> dict:
    """ConstraintClusterBuilder -> SupernodalElements -> ContactPartitioner."""
    n = snapshot.num_elements
    elements, num_elements, _, _ = snapshot.to_warp(device)

    cb = ConstraintClusterBuilder(
        max_num_interactions=n,
        max_num_nodes=snapshot.num_bodies,
        device=device,
    )
    se = SupernodalElements(max_num_clusters=n, device=device)
    cp = ContactPartitioner(
        max_num_interactions=n,
        max_num_nodes=snapshot.num_bodies,
        max_num_partitions=128,
        device=device,
    )

    # Warm-up to JIT-compile + populate outputs we need post-bench.
    cb.build_clusters(elements, num_elements)
    se.build(cb.cluster_members, cb.num_clusters, elements)
    cp.launch(se.elements, se.num_clusters)
    wp.synchronize_device(device)
    num_clusters = int(cb.num_clusters.numpy()[0])
    num_colors = int(cp.num_partitions.numpy()[0])
    has_overflow = bool(int(cp.has_additional_partition.numpy()[0]))
    # Cluster-size histogram (host-side, off the timing path).
    member_counts = se.member_counts.numpy()[:num_clusters].astype(np.int32, copy=False)
    size_histogram = {int(s): int((member_counts == s).sum()) for s in range(1, int(MAX_CLUSTER_SIZE) + 1)}

    # Per-stage timings.
    build_ms, _ = _time_block(
        lambda: cb.build_clusters(elements, num_elements),
        repeats,
        device,
    )
    super_ms, _ = _time_block(
        lambda: se.build(cb.cluster_members, cb.num_clusters, elements),
        repeats,
        device,
    )
    color_ms, _ = _time_block(
        lambda: cp.launch(se.elements, se.num_clusters),
        repeats,
        device,
    )

    return {
        "num_clusters": num_clusters,
        "num_colors": num_colors,
        "has_overflow": has_overflow,
        "size_histogram": size_histogram,
        "mean_cluster_size": float(member_counts.mean()) if num_clusters else 0.0,
        "max_cluster_size": int(member_counts.max(initial=0)),
        "cluster_build_ms": build_ms,
        "supernodal_build_ms": super_ms,
        "color_time_ms": color_ms,
    }


def _report(snapshot_path: Path, baseline: dict, clustered: dict) -> None:
    """One-screen summary per snapshot."""
    n = baseline["num_elements"]
    print(f"\n=== {snapshot_path.name} ===")
    print(f"  constraints   : {n}")
    print()
    print(
        f"  baseline      : {baseline['num_colors']:>5d} colours" + (" (+overflow)" if baseline["has_overflow"] else "")
    )
    print(f"    colour time : {baseline['color_time_ms']:>8.3f} ms")
    print()
    nc = clustered["num_clusters"]
    print(
        f"  clustered     : {clustered['num_colors']:>5d} colours"
        + (" (+overflow)" if clustered["has_overflow"] else "")
    )
    print(
        f"  clusters      : {nc} (mean size {clustered['mean_cluster_size']:.2f}, max {clustered['max_cluster_size']})"
    )
    hist = clustered["size_histogram"]
    hist_str = " ".join(f"K={s}:{c}" for s, c in hist.items() if c > 0)
    print(f"  cluster sizes : {hist_str}")
    print(f"    cluster bld : {clustered['cluster_build_ms']:>8.3f} ms")
    print(f"    supernodal  : {clustered['supernodal_build_ms']:>8.3f} ms")
    print(f"    colour time : {clustered['color_time_ms']:>8.3f} ms")
    total_cluster_ms = clustered["cluster_build_ms"] + clustered["supernodal_build_ms"] + clustered["color_time_ms"]
    print(f"    total       : {total_cluster_ms:>8.3f} ms")
    print()
    if clustered["num_colors"] > 0:
        ratio_c = baseline["num_colors"] / clustered["num_colors"]
        print(f"  colour ratio  : {ratio_c:.2f}x (baseline / clustered)")
    if total_cluster_ms > 0:
        ratio_t = baseline["color_time_ms"] / total_cluster_ms
        print(f"  time ratio    : {ratio_t:.2f}x (baseline colour / clustered pipeline)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots",
        nargs="+",
        required=True,
        type=Path,
        help="Path(s) to .npz snapshot dump(s) from PHOENX_DUMP_COLORING_GRAPH.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed runs per kernel launch path (default 5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Warp device (default: preferred CUDA).",
    )
    args = parser.parse_args()
    device = wp.get_device(args.device) if args.device else wp.get_preferred_device()
    if not device.is_cuda:
        raise SystemExit(f"clustering bench requires CUDA (got {device})")

    for snap_path in args.snapshots:
        snapshot = Snapshot(snap_path)
        baseline = _baseline_colors(snapshot, device, args.repeats)
        clustered = _clustered_colors(snapshot, device, args.repeats)
        _report(snap_path, baseline, clustered)


if __name__ == "__main__":
    main()
