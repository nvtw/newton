# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated benchmark for deterministic contact-run compaction."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False, grid_stride=False)
def _mark_all_runs(shape0: wp.array[wp.int32], shape1: wp.array[wp.int32], boundary: wp.array[wp.int32]):
    i = wp.tid()
    is_start = i == wp.int32(0)
    if i > wp.int32(0):
        is_start = shape0[i] != shape0[i - wp.int32(1)] or shape1[i] != shape1[i - wp.int32(1)]
    boundary[i] = wp.int32(is_start)


@wp.kernel(enable_backward=False, grid_stride=False)
def _scatter_run_starts(
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    boundary: wp.array[wp.int32],
    run_id: wp.array[wp.int32],
    run_first: wp.array[wp.int32],
    run_shape0: wp.array[wp.int32],
    run_shape1: wp.array[wp.int32],
):
    i = wp.tid()
    if boundary[i] != wp.int32(0):
        run = run_id[i] - wp.int32(1)
        run_first[run] = i
        run_shape0[run] = shape0[i]
        run_shape1[run] = shape1[i]


@wp.kernel(enable_backward=False, grid_stride=False)
def _count_and_keep_runs(
    contact_count: wp.int32,
    run_id: wp.array[wp.int32],
    run_first: wp.array[wp.int32],
    run_shape0: wp.array[wp.int32],
    run_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    run_count: wp.array[wp.int32],
    keep: wp.array[wp.int32],
):
    run = wp.tid()
    count = run_id[contact_count - wp.int32(1)]
    if run >= count:
        keep[run] = wp.int32(0)
        return
    first = run_first[run]
    end = contact_count
    if run + wp.int32(1) < count:
        end = run_first[run + wp.int32(1)]
    run_count[run] = end - first
    keep[run] = wp.int32(shape_body[run_shape0[run]] != shape_body[run_shape1[run]])


@wp.kernel(enable_backward=False, grid_stride=False)
def _scatter_kept_runs(
    contact_count: wp.int32,
    keep: wp.array[wp.int32],
    kept_offset: wp.array[wp.int32],
    run_first: wp.array[wp.int32],
    run_count: wp.array[wp.int32],
    run_shape0: wp.array[wp.int32],
    run_shape1: wp.array[wp.int32],
    column_first: wp.array[wp.int32],
    column_count: wp.array[wp.int32],
    column_shapes: wp.array[wp.vec2i],
    contact_column: wp.array[wp.int32],
):
    run = wp.tid()
    if run >= contact_count or keep[run] == wp.int32(0):
        return
    column = kept_offset[run]
    first = run_first[run]
    count = run_count[run]
    column_first[column] = first
    column_count[column] = count
    column_shapes[column] = wp.vec2i(run_shape0[run], run_shape1[run])
    for j in range(count):
        contact_column[first + j] = column


@wp.kernel(enable_backward=False, grid_stride=False)
def _mark_kept_runs(
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    boundary: wp.array[wp.int32],
):
    i = wp.tid()
    is_start = i == wp.int32(0)
    if i > wp.int32(0):
        is_start = shape0[i] != shape0[i - wp.int32(1)] or shape1[i] != shape1[i - wp.int32(1)]
    keep = is_start and shape_body[shape0[i]] != shape_body[shape1[i]]
    boundary[i] = wp.int32(keep)


@wp.kernel(enable_backward=False, grid_stride=False)
def _materialize_kept_runs(
    contact_count: wp.int32,
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    boundary: wp.array[wp.int32],
    column_id: wp.array[wp.int32],
    column_first: wp.array[wp.int32],
    column_count: wp.array[wp.int32],
    column_shapes: wp.array[wp.vec2i],
    contact_column: wp.array[wp.int32],
):
    first = wp.tid()
    if boundary[first] == wp.int32(0):
        return
    column = column_id[first] - wp.int32(1)
    sa = shape0[first]
    sb = shape1[first]
    end = first + wp.int32(1)
    while end < contact_count and shape0[end] == sa and shape1[end] == sb:
        end += wp.int32(1)
    count = end - first
    column_first[column] = first
    column_count[column] = count
    column_shapes[column] = wp.vec2i(sa, sb)
    for j in range(count):
        contact_column[first + j] = column


def _measure(graph: wp.Graph, replays: int, device: wp.Device) -> float:
    for _ in range(20):
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
    parser.add_argument("--points-per-pair", type=int, default=4)
    parser.add_argument("--replays", type=int, default=1000)
    args = parser.parse_args()
    wp.init()
    device = wp.get_device("cuda:0")
    count = args.contacts - args.contacts % args.points_per_pair
    pair = np.arange(count, dtype=np.int32) // args.points_per_pair
    shape0_np = pair * 2
    shape1_np = shape0_np + 1
    shape_body_np = np.arange(int(shape1_np[-1]) + 1, dtype=np.int32)
    shape0 = wp.array(shape0_np, device=device)
    shape1 = wp.array(shape1_np, device=device)
    shape_body = wp.array(shape_body_np, device=device)
    boundary = wp.empty(count, dtype=wp.int32, device=device)
    scan0 = wp.empty(count, dtype=wp.int32, device=device)
    scan1 = wp.empty(count, dtype=wp.int32, device=device)
    run_first = wp.empty(count, dtype=wp.int32, device=device)
    run_count = wp.empty(count, dtype=wp.int32, device=device)
    run_shape0 = wp.empty(count, dtype=wp.int32, device=device)
    run_shape1 = wp.empty(count, dtype=wp.int32, device=device)
    keep = wp.empty(count, dtype=wp.int32, device=device)
    column_first = wp.empty(count, dtype=wp.int32, device=device)
    column_count = wp.empty(count, dtype=wp.int32, device=device)
    column_shapes = wp.empty(count, dtype=wp.vec2i, device=device)
    contact_column = wp.empty(count, dtype=wp.int32, device=device)

    with wp.ScopedCapture(device=device) as baseline_capture:
        wp.launch(_mark_all_runs, count, [shape0, shape1], [boundary], device=device)
        wp.utils.array_scan(boundary, scan0, inclusive=True)
        wp.launch(
            _scatter_run_starts,
            count,
            [shape0, shape1, boundary, scan0],
            [run_first, run_shape0, run_shape1],
            device=device,
        )
        wp.launch(
            _count_and_keep_runs,
            count,
            [count, scan0, run_first, run_shape0, run_shape1, shape_body],
            [run_count, keep],
            device=device,
        )
        wp.utils.array_scan(keep, scan1, inclusive=False)
        wp.launch(
            _scatter_kept_runs,
            count,
            [count, keep, scan1, run_first, run_count, run_shape0, run_shape1],
            [column_first, column_count, column_shapes, contact_column],
            device=device,
        )

    with wp.ScopedCapture(device=device) as compact_capture:
        wp.launch(_mark_kept_runs, count, [shape0, shape1, shape_body], [boundary], device=device)
        wp.utils.array_scan(boundary, scan0, inclusive=True)
        wp.launch(
            _materialize_kept_runs,
            count,
            [count, shape0, shape1, boundary, scan0],
            [column_first, column_count, column_shapes, contact_column],
            device=device,
        )

    baseline_us = _measure(baseline_capture.graph, args.replays, device)
    compact_us = _measure(compact_capture.graph, args.replays, device)
    print(
        json.dumps(
            {
                "contacts": count,
                "points_per_pair": args.points_per_pair,
                "baseline_us": baseline_us,
                "compact_us": compact_us,
                "speedup": baseline_us / compact_us,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
