# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure representative floating-point compute ceilings."""

from __future__ import annotations

import argparse
import functools
import gc

import warp as wp

from .common import BenchmarkResult, get_hardware_limits, print_report, require_cuda, throughput_result, time_cuda_graph


@wp.kernel
def _fma_kernel(iterations: int, output: wp.array[wp.float32]):
    tid = wp.tid()
    x = wp.float32(tid & 255) * 1.0e-5
    a0 = x + 0.101
    a1 = x + 0.202
    a2 = x + 0.303
    a3 = x + 0.404
    a4 = x + 0.505
    a5 = x + 0.606
    a6 = x + 0.707
    a7 = x + 0.808
    for _ in range(iterations):
        a0 = a0 * 0.99991 + 0.00011
        a1 = a1 * 0.99989 + 0.00013
        a2 = a2 * 0.99987 + 0.00017
        a3 = a3 * 0.99983 + 0.00019
        a4 = a4 * 0.99979 + 0.00023
        a5 = a5 * 0.99977 + 0.00029
        a6 = a6 * 0.99971 + 0.00031
        a7 = a7 * 0.99967 + 0.00037
    output[tid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7


@wp.kernel
def _vector_kernel(iterations: int, output: wp.array[wp.float32]):
    tid = wp.tid()
    x = wp.float32(tid & 255) * 1.0e-5
    a = wp.vec3f(x + 0.2, x + 0.3, x + 0.4)
    b = wp.vec3f(x + 0.7, x + 0.8, x + 0.9)
    total = wp.float32(0.0)
    for _ in range(iterations):
        c = wp.cross(a, b)
        total = total + wp.dot(a, b) + wp.dot(c, c)
        a = a * 0.9999 + c * 0.0001
        b = b * 0.9998 - c * 0.0001
    output[tid] = total + a[0] + b[0]


@wp.kernel
def _rsqrt_kernel(iterations: int, output: wp.array[wp.float32]):
    tid = wp.tid()
    x0 = wp.float32(tid & 255) * 1.0e-4 + 1.1
    x1 = x0 + 0.2
    x2 = x0 + 0.4
    x3 = x0 + 0.6
    for _ in range(iterations):
        x0 = 1.0 / wp.sqrt(x0 + 0.9)
        x1 = 1.0 / wp.sqrt(x1 + 0.8)
        x2 = 1.0 / wp.sqrt(x2 + 0.7)
        x3 = 1.0 / wp.sqrt(x3 + 0.6)
    output[tid] = x0 + x1 + x2 + x3


def run(
    *,
    device_name: str = "cuda:0",
    elements: int = 1_048_576,
    iterations: int = 256,
    warmup: int = 5,
    repetitions: int = 20,
    trials: int = 5,
    theoretical_gflops: float | None = None,
) -> list[BenchmarkResult]:
    """Run compute-heavy kernels representative of physics workloads."""
    device = require_cuda(device_name)
    if theoretical_gflops is None:
        theoretical_gflops = get_hardware_limits(device).fp32_gflops
    output = wp.empty(elements, dtype=wp.float32, device=device)
    cases = (
        ("independent FMA", _fma_kernel, elements * iterations * 16.0 / 1.0e9, "GFLOP/s"),
        ("vec3 dot/cross", _vector_kernel, elements * iterations * 39.0 / 1.0e9, "GFLOP/s"),
        ("sqrt + reciprocal", _rsqrt_kernel, elements * iterations * 8.0 / 1.0e9, "Gop/s"),
    )
    results: list[BenchmarkResult] = []
    for workload, kernel, work, unit in cases:
        launch = functools.partial(
            wp.launch, kernel, dim=elements, inputs=[iterations], outputs=[output], device=device
        )

        best_ms, median_ms = time_cuda_graph(
            launch, device=device, warmup=warmup, repetitions=repetitions, trials=trials
        )
        results.append(
            throughput_result(
                category="Floating-point throughput",
                workload=workload,
                layout="float32",
                problem_size=f"{elements:,} x {iterations}",
                work_per_launch=work,
                unit=unit,
                best_ms=best_ms,
                median_ms=median_ms,
                theoretical=theoretical_gflops if unit == "GFLOP/s" else None,
            )
        )
    del output
    gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    print_report(
        run(
            device_name=args.device,
            elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
            repetitions=args.repetitions,
            trials=args.trials,
        )
    )


if __name__ == "__main__":
    main()
