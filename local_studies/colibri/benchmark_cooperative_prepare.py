"""Isolated frozen preparation timing; no scene simulation or physics changes."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.check_cooperative_bilateral_prepare import compare, snapshot_fixture
from local_studies.colibri.cooperative_bilateral_prepare import launch
from newton._src.solvers.phoenx.constraints.bilateral_joint import prepare_bilateral_joint_blocks


def main():
    wp.init()
    device = "cuda:0"
    constraints, bodies, copies, count = snapshot_fixture(
        "/tmp/colibri_current_native_velocity_iterations_snapshot.npz"
    )
    for block in (32, 64):
        compare(constraints, bodies, copies, count, block)
    graphs = {}
    for label, block in (("scalar", 0), ("cooperative32", 32), ("cooperative64", 64)):
        with wp.ScopedCapture(device=device) as captured:
            for _ in range(128):
                if block:
                    launch(constraints, bodies, copies, count, device, block)
                else:
                    wp.launch(prepare_bilateral_joint_blocks, count, [constraints, bodies, copies], device=device)
        graphs[label] = captured.graph
        for _ in range(3):
            wp.capture_launch(captured.graph)
    wp.synchronize_device(device)
    stream = wp.get_stream(device)
    samples = {label: [] for label in graphs}
    for repeat in range(12):
        labels = list(graphs)
        if repeat % 2:
            labels.reverse()
        for label in labels:
            start = wp.Event(device, enable_timing=True)
            end = wp.Event(device, enable_timing=True)
            stream.record_event(start)
            wp.capture_launch(graphs[label])
            stream.record_event(end)
            samples[label].append(wp.get_event_elapsed_time(start, end) / 128)
    report = {label: {"samples_ms": values, "median_ms": float(np.median(values))} for label, values in samples.items()}
    report["scope"] = (
        "Frozen38joint preparation only;128calls per graph;12 forward/reverse interleaved samples, same captured inputs"
    )
    Path("/tmp/cooperative_prepare_frozen_timing.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
