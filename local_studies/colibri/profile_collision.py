# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure full Colibri collision detection, reduction, and contact matching."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.kamino.example_kamino_colibri import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_collision_profile.json"))
    args = parser.parse_args()
    model = build_scene(contact_gap=0.001, source_contact_offsets=True, mesh_cylinders=True).finalize(
        skip_validation_joints=True
    )
    state = model.state()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192, contact_matching="sticky")
    contacts = pipeline.contacts()
    for _ in range(3):
        pipeline.collide(state, contacts)
    wp.synchronize_device(model.device)
    with wp.ScopedCapture(device=model.device) as capture:
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)
    for _ in range(10):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(model.device)
    times = []
    for _ in range(args.frames):
        start = time.perf_counter()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(model.device)
        times.append(time.perf_counter() - start)
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count >= contacts.rigid_contact_max:
        raise AssertionError("Collision contact capacity exhausted")
    with wp.ScopedTimer("Collision kernels", cuda_filter=wp.TIMING_KERNEL, print=False) as timer:
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)
    grouped = defaultdict(lambda: [0, 0.0])
    for result in timer.timing_results:
        grouped[result.name][0] += 1
        grouped[result.name][1] += result.elapsed
    rows = [
        {"kernel": key, "count": value[0], "total_ms": value[1]}
        for key, value in sorted(grouped.items(), key=lambda item: item[1][1], reverse=True)
    ]
    report = {
        "body_count": model.body_count,
        "shape_count": model.shape_count,
        "contacts": count,
        "frames": args.frames,
        "collision_updates_per_frame": 2,
        "collision_graph_mean_ms": 1000.0 * float(np.mean(times)),
        "collision_graph_p95_ms": 1000.0 * float(np.percentile(times, 95)),
        "kernel_ms_sum_instrumented": sum(row["total_ms"] for row in rows),
        "scope": "Fixed authored full-scene poses; detection, reduction, sort and sticky matching; no dynamics or rendering",
        "config": "SI, gravity1, source SDF resolutions,32-sided source-phase cylinders, authored offsets/fallback1mm",
        "kernels": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "kernels": rows[:10]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
