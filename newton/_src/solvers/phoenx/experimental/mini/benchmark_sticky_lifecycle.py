# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark sticky history copies against pre-gather replay."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

_ROWS = 5
_BYTES_PER_RECORD = _ROWS * 3 * 4


@wp.kernel(enable_backward=False, grid_stride=False)
def _gather(
    scratch: wp.array2d[wp.vec3],
    permutation: wp.array[wp.int32],
    live: wp.array2d[wp.vec3],
):
    i = wp.tid()
    p = permutation[i]
    for row in range(_ROWS):
        live[row, i] = scratch[row, p]


@wp.kernel(enable_backward=False, grid_stride=False)
def _replay(
    history: wp.array2d[wp.vec3],
    match_index: wp.array[wp.int32],
    live: wp.array2d[wp.vec3],
):
    i = wp.tid()
    previous = match_index[i]
    if previous >= wp.int32(0):
        for row in range(_ROWS):
            live[row, i] = history[row, previous]


@wp.kernel(enable_backward=False, grid_stride=False)
def _save(live: wp.array2d[wp.vec3], history: wp.array2d[wp.vec3]):
    i = wp.tid()
    for row in range(_ROWS):
        history[row, i] = live[row, i]


@wp.kernel(enable_backward=False, grid_stride=False)
def _pre_gather_replay(
    live: wp.array2d[wp.vec3],
    match_index: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    scratch: wp.array2d[wp.vec3],
):
    i = wp.tid()
    previous = match_index[i]
    p = permutation[i]
    for row in range(_ROWS):
        scratch[row, p] = live[row, previous]


@wp.kernel(enable_backward=False, grid_stride=False)
def _build_overlay(
    live: wp.array2d[wp.vec3],
    scratch: wp.array2d[wp.vec3],
    match_index: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    overlay: wp.array2d[wp.vec3],
):
    i = wp.tid()
    previous = match_index[i]
    if previous >= wp.int32(0):
        for row in range(_ROWS):
            overlay[row, i] = live[row, previous]
    else:
        p = permutation[i]
        for row in range(_ROWS):
            overlay[row, i] = scratch[row, p]


@wp.kernel(enable_backward=False, grid_stride=False)
def _gather_overlay(overlay: wp.array2d[wp.vec3], live: wp.array2d[wp.vec3]):
    i = wp.tid()
    for row in range(_ROWS):
        live[row, i] = overlay[row, i]


def _measure(graph: wp.Graph, replays: int, device: wp.Device) -> float:
    for _ in range(30):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(replays):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    return (time.perf_counter() - start) * 1.0e6 / replays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=int, default=1_048_576)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fresh-percent", type=float, default=0.0)
    args = parser.parse_args()

    wp.init()
    device = wp.get_device("cuda:0")
    count = args.contacts
    rng = np.random.default_rng(args.seed)
    permutation = wp.array(rng.permutation(count).astype(np.int32), device=device)
    match_np = np.arange(count, dtype=np.int32).reshape(-1, 4)[:, ::-1].reshape(-1)
    fresh_count = round(count * args.fresh_percent / 100.0)
    match_np[rng.choice(count, fresh_count, replace=False)] = -1
    match_index = wp.array(match_np, device=device)
    scratch = wp.empty((_ROWS, count), dtype=wp.vec3, device=device)
    live = wp.empty((_ROWS, count), dtype=wp.vec3, device=device)
    history = wp.empty((_ROWS, count), dtype=wp.vec3, device=device)
    overlay = wp.empty((_ROWS, count), dtype=wp.vec3, device=device)

    with wp.ScopedCapture(device=device) as baseline_capture:
        wp.launch(_gather, count, [scratch, permutation], [live], device=device)
        wp.launch(_replay, count, [history, match_index], [live], device=device)
        wp.launch(_save, count, [live], [history], device=device)

    with wp.ScopedCapture(device=device) as pre_gather_capture:
        wp.launch(_pre_gather_replay, count, [live, match_index, permutation], [scratch], device=device)
        wp.launch(_gather, count, [scratch, permutation], [live], device=device)

    with wp.ScopedCapture(device=device) as overlay_capture:
        wp.launch(_build_overlay, count, [live, scratch, match_index, permutation], [overlay], device=device)
        wp.launch(_gather_overlay, count, [overlay], [live], device=device)

    baseline_us = _measure(baseline_capture.graph, args.replays, device)
    pre_gather_us = _measure(pre_gather_capture.graph, args.replays, device)
    overlay_us = _measure(overlay_capture.graph, args.replays, device)
    baseline_gbps = count * 6 * _BYTES_PER_RECORD / baseline_us / 1.0e3
    pre_gather_gbps = count * 4 * _BYTES_PER_RECORD / pre_gather_us / 1.0e3
    overlay_gbps = count * 4 * _BYTES_PER_RECORD / overlay_us / 1.0e3
    print(
        json.dumps(
            {
                "contacts": count,
                "fresh_percent": args.fresh_percent,
                "baseline_us": baseline_us,
                "pre_gather_us": pre_gather_us,
                "speedup": baseline_us / pre_gather_us,
                "overlay_us": overlay_us,
                "overlay_speedup": baseline_us / overlay_us,
                "baseline_logical_gbps": baseline_gbps,
                "pre_gather_logical_gbps": pre_gather_gbps,
                "overlay_logical_gbps": overlay_gbps,
                "baseline_sequential_percent": 100.0 * baseline_gbps / 1489.14,
                "pre_gather_sequential_percent": 100.0 * pre_gather_gbps / 1489.14,
                "overlay_sequential_percent": 100.0 * overlay_gbps / 1489.14,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
