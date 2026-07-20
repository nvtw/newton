# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated scalar-SoA versus vec4-SoA constraint-stream benchmark.

The dense probe models the immutable payload consumed by a prepared revolute
iterate: twelve adjacent four-dword groups are all used.  The sparse probe
loads one lane from each group and bounds the downside for optional fields.
Both layouts contain the same logical values and use coalesced constraint IDs.
"""

from __future__ import annotations

import argparse
import json
import time

import warp as wp

from newton._src.solvers.phoenx.helpers.array_access import read2d_f32

_GROUPS = 12
_DWORDS = 4 * _GROUPS
_SEQUENTIAL_ROOF_GBS = 1489.14


@wp.kernel(enable_backward=False)
def _dense_scalar_kernel(data: wp.array2d[wp.float32], output: wp.array[wp.float32]):
    cid = wp.tid()
    total = wp.float32(0.0)
    for group in range(_GROUPS):
        off = wp.int32(4 * group)
        total += read2d_f32(data, off + wp.int32(0), cid)
        total += read2d_f32(data, off + wp.int32(1), cid)
        total += read2d_f32(data, off + wp.int32(2), cid)
        total += read2d_f32(data, off + wp.int32(3), cid)
    output[cid] = total


@wp.kernel(enable_backward=False)
def _dense_vec4_kernel(data: wp.array2d[wp.vec4f], output: wp.array[wp.float32]):
    cid = wp.tid()
    total = wp.float32(0.0)
    for group in range(_GROUPS):
        value = data[group, cid]
        total += value[0] + value[1] + value[2] + value[3]
    output[cid] = total


@wp.kernel(enable_backward=False)
def _sparse_scalar_kernel(data: wp.array2d[wp.float32], output: wp.array[wp.float32]):
    cid = wp.tid()
    total = wp.float32(0.0)
    for group in range(_GROUPS):
        total += read2d_f32(data, wp.int32(4 * group), cid)
    output[cid] = total


@wp.kernel(enable_backward=False)
def _sparse_vec4_kernel(data: wp.array2d[wp.vec4f], output: wp.array[wp.float32]):
    cid = wp.tid()
    total = wp.float32(0.0)
    for group in range(_GROUPS):
        total += data[group, cid][0]
    output[cid] = total


def _measure(kernel, data, output, count: int, warmup: int, replays: int) -> float:
    wp.launch(kernel, dim=count, inputs=[data], outputs=[output])
    with wp.ScopedCapture() as capture:
        wp.launch(kernel, dim=count, inputs=[data], outputs=[output])
    for _ in range(warmup):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(data.device)
    start = time.perf_counter()
    for _ in range(replays):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(data.device)
    return (time.perf_counter() - start) / replays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = wp.get_device("cuda:0")
    scalar = wp.full((_DWORDS, args.count), 1.0, dtype=wp.float32, device=device)
    packed = wp.full((_GROUPS, args.count), wp.vec4f(1.0), dtype=wp.vec4f, device=device)
    output = wp.empty(args.count, dtype=wp.float32, device=device)

    dense_scalar = _measure(_dense_scalar_kernel, scalar, output, args.count, args.warmup, args.replays)
    dense_vec4 = _measure(_dense_vec4_kernel, packed, output, args.count, args.warmup, args.replays)
    sparse_scalar = _measure(_sparse_scalar_kernel, scalar, output, args.count, args.warmup, args.replays)
    sparse_vec4 = _measure(_sparse_vec4_kernel, packed, output, args.count, args.warmup, args.replays)

    dense_bytes = args.count * (_DWORDS * 4 + 4)
    sparse_scalar_bytes = args.count * (_GROUPS * 4 + 4)
    sparse_vec4_bytes = args.count * (_GROUPS * 16 + 4)
    result = {
        "count": args.count,
        "working_set_mib": args.count * _DWORDS * 4 / (1024 * 1024),
        "dense_scalar_us": dense_scalar * 1.0e6,
        "dense_vec4_us": dense_vec4 * 1.0e6,
        "dense_speedup": dense_scalar / dense_vec4,
        "dense_scalar_gbs": dense_bytes / dense_scalar / 1.0e9,
        "dense_vec4_gbs": dense_bytes / dense_vec4 / 1.0e9,
        "dense_scalar_roof_pct": 100.0 * dense_bytes / dense_scalar / 1.0e9 / _SEQUENTIAL_ROOF_GBS,
        "dense_vec4_roof_pct": 100.0 * dense_bytes / dense_vec4 / 1.0e9 / _SEQUENTIAL_ROOF_GBS,
        "sparse_scalar_us": sparse_scalar * 1.0e6,
        "sparse_vec4_us": sparse_vec4 * 1.0e6,
        "sparse_speedup": sparse_scalar / sparse_vec4,
        "sparse_scalar_physical_gbs": sparse_scalar_bytes / sparse_scalar / 1.0e9,
        "sparse_vec4_physical_gbs": sparse_vec4_bytes / sparse_vec4 / 1.0e9,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
