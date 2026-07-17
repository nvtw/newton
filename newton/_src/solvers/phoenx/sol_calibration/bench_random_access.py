# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure indexed random-read device-memory throughput."""

from __future__ import annotations

import argparse
import functools
import gc

import warp as wp

from .common import BenchmarkResult, print_report, require_cuda, throughput_result, time_cuda_graph


@wp.kernel
def _init_indices(indices: wp.array[wp.int32], mask: int):
    tid = wp.tid()
    # An odd multiplier permutes a power-of-two range while separating adjacent lanes.
    indices[tid] = (tid * 8191) & mask


@wp.kernel
def _gather_float(src: wp.array[wp.float32], indices: wp.array[wp.int32], dst: wp.array[wp.float32]):
    tid = wp.tid()
    dst[tid] = src[indices[tid]]


@wp.kernel
def _gather_vec2(src: wp.array[wp.vec2f], indices: wp.array[wp.int32], dst: wp.array[wp.vec2f]):
    tid = wp.tid()
    dst[tid] = src[indices[tid]]


@wp.kernel
def _gather_vec4(src: wp.array[wp.vec4f], indices: wp.array[wp.int32], dst: wp.array[wp.vec4f]):
    tid = wp.tid()
    dst[tid] = src[indices[tid]]


_LAYOUTS = (
    ("float", wp.float32, 4, _gather_float),
    ("vec2", wp.vec2f, 8, _gather_vec2),
    ("vec4", wp.vec4f, 16, _gather_vec4),
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def run(
    *,
    device_name: str = "cuda:0",
    array_mib: int = 256,
    warmup: int = 5,
    repetitions: int = 20,
    trials: int = 5,
) -> list[BenchmarkResult]:
    """Run indexed gather benchmarks for all supported layouts."""
    if not _is_power_of_two(array_mib):
        raise ValueError("array_mib must be a power of two so the index mapping is a permutation")
    device = require_cuda(device_name)
    array_bytes = array_mib * 1024 * 1024
    results: list[BenchmarkResult] = []
    for name, dtype, element_bytes, kernel in _LAYOUTS:
        count = array_bytes // element_bytes
        src = wp.zeros(count, dtype=dtype, device=device)
        dst = wp.empty(count, dtype=dtype, device=device)
        indices = wp.empty(count, dtype=wp.int32, device=device)
        wp.launch(_init_indices, dim=count, inputs=[indices, count - 1], device=device)
        wp.synchronize_device(device)

        launch = functools.partial(wp.launch, kernel, dim=count, inputs=[src, indices], outputs=[dst], device=device)

        best_ms, median_ms = time_cuda_graph(
            launch, device=device, warmup=warmup, repetitions=repetitions, trials=trials
        )
        # Traffic includes one index read, one random value read, and one coalesced value write.
        traffic_bytes = count * (4 + 2 * element_bytes)
        results.append(
            throughput_result(
                category="Random-access memory throughput",
                workload="indexed gather",
                layout=name,
                problem_size=f"{array_mib} MiB data",
                work_per_launch=traffic_bytes / 1.0e9,
                unit="GB/s",
                best_ms=best_ms,
                median_ms=median_ms,
            )
        )
        del src, dst, indices
        gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--array-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    print_report(
        run(
            device_name=args.device,
            array_mib=args.array_mib,
            warmup=args.warmup,
            repetitions=args.repetitions,
            trials=args.trials,
        )
    )


if __name__ == "__main__":
    main()
