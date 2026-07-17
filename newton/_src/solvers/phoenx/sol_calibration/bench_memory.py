# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure coalesced device-memory copy bandwidth."""

from __future__ import annotations

import argparse
import functools
import gc

import warp as wp

from .common import BenchmarkResult, print_report, require_cuda, throughput_result, time_cuda_graph


@wp.kernel
def _copy_float(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def _copy_vec2(src: wp.array[wp.vec2f], dst: wp.array[wp.vec2f]):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def _copy_vec4(src: wp.array[wp.vec4f], dst: wp.array[wp.vec4f]):
    tid = wp.tid()
    dst[tid] = src[tid]


_LAYOUTS = (
    ("float", wp.float32, 4, _copy_float),
    ("vec2", wp.vec2f, 8, _copy_vec2),
    ("vec4", wp.vec4f, 16, _copy_vec4),
)


def run(
    *,
    device_name: str = "cuda:0",
    array_mib: int = 256,
    warmup: int = 5,
    repetitions: int = 20,
    trials: int = 5,
) -> list[BenchmarkResult]:
    """Run coalesced copy benchmarks for all supported layouts."""
    device = require_cuda(device_name)
    array_bytes = array_mib * 1024 * 1024
    results: list[BenchmarkResult] = []
    for name, dtype, element_bytes, kernel in _LAYOUTS:
        count = array_bytes // element_bytes
        src = wp.zeros(count, dtype=dtype, device=device)
        dst = wp.empty(count, dtype=dtype, device=device)

        launch = functools.partial(wp.launch, kernel, dim=count, inputs=[src], outputs=[dst], device=device)

        best_ms, median_ms = time_cuda_graph(
            launch, device=device, warmup=warmup, repetitions=repetitions, trials=trials
        )
        results.append(
            throughput_result(
                category="Sequential memory throughput",
                workload="copy",
                layout=name,
                problem_size=f"{array_mib} MiB x 2",
                work_per_launch=2.0 * count * element_bytes / 1.0e9,
                unit="GB/s",
                best_ms=best_ms,
                median_ms=median_ms,
            )
        )
        del src, dst
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
