# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared measurement and reporting utilities for SOL calibration."""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Callable, Sequence

import warp as wp


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """One hardware calibration result."""

    category: str
    workload: str
    layout: str
    problem_size: str
    peak: float
    median: float
    unit: str
    median_ms: float
    theoretical: float | None


@dataclasses.dataclass(frozen=True)
class HardwareLimits:
    """Published hardware throughput ceilings."""

    memory_gbps: float | None
    fp32_gflops: float | None
    bf16_tensor_gflops: float | None


_KNOWN_HARDWARE_LIMITS = {
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": HardwareLimits(
        memory_gbps=1792.0,
        fp32_gflops=125_000.0,
        bf16_tensor_gflops=503_800.0,
    ),
}


def get_hardware_limits(device: wp.context.Device) -> HardwareLimits:
    """Return published limits for ``device``, if known."""
    return _KNOWN_HARDWARE_LIMITS.get(device.name, HardwareLimits(None, None, None))


def require_cuda(device_name: str) -> wp.context.Device:
    """Resolve ``device_name`` and require a CUDA timing target."""
    device = wp.get_device(device_name)
    if not device.is_cuda:
        raise ValueError(f"SOL calibration requires a CUDA device, got {device}")
    return device


def time_cuda_graph(
    launch: Callable[[], None],
    *,
    device: wp.context.Device,
    warmup: int,
    repetitions: int,
    trials: int,
) -> tuple[float, float]:
    """Return the best and median milliseconds per graph replay."""
    if warmup < 1 or repetitions < 1 or trials < 1:
        raise ValueError("warmup, repetitions, and trials must all be positive")

    for _ in range(warmup):
        launch()
    wp.synchronize_device(device)

    with wp.ScopedCapture(device=device) as capture:
        launch()
    graph = capture.graph
    wp.synchronize_device(device)

    samples_ms: list[float] = []
    for _ in range(trials):
        begin = wp.Event(device=device, enable_timing=True)
        end = wp.Event(device=device, enable_timing=True)
        wp.record_event(begin)
        for _ in range(repetitions):
            wp.capture_launch(graph)
        wp.record_event(end)
        wp.synchronize_device(device)
        samples_ms.append(wp.get_event_elapsed_time(begin, end) / repetitions)

    return min(samples_ms), statistics.median(samples_ms)


def throughput_result(
    *,
    category: str,
    workload: str,
    layout: str,
    problem_size: str,
    work_per_launch: float,
    unit: str,
    best_ms: float,
    median_ms: float,
    theoretical: float | None = None,
) -> BenchmarkResult:
    """Build a result whose work rate is expressed per second."""
    scale = 1.0e-3
    return BenchmarkResult(
        category=category,
        workload=workload,
        layout=layout,
        problem_size=problem_size,
        peak=work_per_launch / (best_ms * scale),
        median=work_per_launch / (median_ms * scale),
        unit=unit,
        median_ms=median_ms,
        theoretical=theoretical,
    )


def print_report(results: Sequence[BenchmarkResult]) -> None:
    """Print calibration results as compact category tables."""
    for category in dict.fromkeys(result.category for result in results):
        rows = [result for result in results if result.category == category]
        headers = (
            "Workload",
            "Layout",
            "Problem",
            "Peak",
            "Median",
            "Theoretical",
            "% theoretical",
            "Unit",
            "Median ms",
        )
        rendered = [
            (
                row.workload,
                row.layout,
                row.problem_size,
                f"{row.peak:,.2f}",
                f"{row.median:,.2f}",
                f"{row.theoretical:,.2f}" if row.theoretical is not None else "n/a",
                f"{100.0 * row.peak / row.theoretical:.1f}%" if row.theoretical is not None else "n/a",
                row.unit,
                f"{row.median_ms:,.3f}",
            )
            for row in rows
        ]
        widths = [max(len(headers[i]), *(len(row[i]) for row in rendered)) for i in range(len(headers))]
        print(f"\n{category}")
        print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
        print("  ".join("-" * width for width in widths))
        for row in rendered:
            cells = [row[i].rjust(widths[i]) if i >= 3 and i != 7 else row[i].ljust(widths[i]) for i in range(len(row))]
            print("  ".join(cells))
