# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX RL-style bfloat16 tiled matrix throughput."""

from __future__ import annotations

import argparse
import functools
import gc

import warp as wp

from .common import BenchmarkResult, get_hardware_limits, print_report, require_cuda, throughput_result, time_cuda_graph

_TILE_M = 128
_TILE_K = 16
_TILE_N = 32
_TILE_BLOCK_DIM = 256

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _matmul_bfloat16(
    a: wp.array2d[wp.bfloat16],
    b: wp.array2d[wp.bfloat16],
    inner_dim: int,
    c: wp.array2d[wp.float32],
):
    row_tile, col_tile = wp.tid()
    total = wp.tile_zeros(shape=(_TILE_M, _TILE_N), dtype=wp.float32)
    for inner_tile in range(inner_dim // _TILE_K):
        a_tile = wp.tile_load(
            a,
            shape=(_TILE_M, _TILE_K),
            offset=(row_tile * _TILE_M, inner_tile * _TILE_K),
        )
        b_tile = wp.tile_load(
            b,
            shape=(_TILE_K, _TILE_N),
            offset=(inner_tile * _TILE_K, col_tile * _TILE_N),
        )
        wp.tile_matmul(a_tile, b_tile, total)
    wp.tile_store(c, total, offset=(row_tile * _TILE_M, col_tile * _TILE_N))


@wp.kernel
def _repeated_mma_bfloat16(
    a: wp.array2d[wp.bfloat16],
    b: wp.array2d[wp.bfloat16],
    iterations: int,
    c: wp.array2d[wp.float32],
):
    tile = wp.tid()
    a_tile = wp.tile_load(a, shape=(_TILE_M, _TILE_K), offset=(tile * _TILE_M, 0))
    b_tile = wp.tile_load(b, shape=(_TILE_K, _TILE_N), offset=(tile * _TILE_K, 0))
    total = wp.tile_zeros(shape=(_TILE_M, _TILE_N), dtype=wp.float32)
    for _ in range(iterations):
        wp.tile_matmul(a_tile, b_tile, total)
    wp.tile_store(c, total, offset=(tile * _TILE_M, 0))


def run(
    *,
    device_name: str = "cuda:0",
    matrix_size: int = 8192,
    mma_iterations: int = 4096,
    warmup: int = 5,
    repetitions: int = 20,
    trials: int = 5,
    theoretical_gflops: float | None = None,
) -> list[BenchmarkResult]:
    """Run a square BF16 GEMM and a compute-bound repeated tile MMA."""
    if matrix_size < _TILE_M or matrix_size % _TILE_M != 0:
        raise ValueError(f"matrix_size must be a multiple of {_TILE_M}")
    if mma_iterations < 1:
        raise ValueError("mma_iterations must be positive")
    device = require_cuda(device_name)
    if theoretical_gflops is None:
        theoretical_gflops = get_hardware_limits(device).bf16_tensor_gflops

    a = wp.zeros((matrix_size, matrix_size), dtype=wp.bfloat16, device=device)
    b = wp.zeros((matrix_size, matrix_size), dtype=wp.bfloat16, device=device)
    c = wp.empty((matrix_size, matrix_size), dtype=wp.float32, device=device)
    launch = functools.partial(
        wp.launch_tiled,
        _matmul_bfloat16,
        dim=(matrix_size // _TILE_M, matrix_size // _TILE_N),
        inputs=[a, b, matrix_size],
        outputs=[c],
        block_dim=_TILE_BLOCK_DIM,
        device=device,
    )
    best_ms, median_ms = time_cuda_graph(launch, device=device, warmup=warmup, repetitions=repetitions, trials=trials)
    results = [
        throughput_result(
            category="Tensor Core throughput",
            workload="RL-style GEMM",
            layout="bfloat16 -> float32",
            problem_size=f"{matrix_size:,} cubed",
            work_per_launch=2.0 * matrix_size**3 / 1.0e9,
            unit="GFLOP/s",
            best_ms=best_ms,
            median_ms=median_ms,
            theoretical=theoretical_gflops,
        )
    ]
    del a, b, c
    gc.collect()

    tile_count = max(1, int(device.sm_count) * 4)
    a = wp.zeros((tile_count * _TILE_M, _TILE_K), dtype=wp.bfloat16, device=device)
    b = wp.zeros((tile_count * _TILE_K, _TILE_N), dtype=wp.bfloat16, device=device)
    c = wp.empty((tile_count * _TILE_M, _TILE_N), dtype=wp.float32, device=device)
    launch = functools.partial(
        wp.launch_tiled,
        _repeated_mma_bfloat16,
        dim=tile_count,
        inputs=[a, b, mma_iterations],
        outputs=[c],
        block_dim=_TILE_BLOCK_DIM,
        device=device,
    )
    best_ms, median_ms = time_cuda_graph(launch, device=device, warmup=warmup, repetitions=repetitions, trials=trials)
    results.append(
        throughput_result(
            category="Tensor Core throughput",
            workload="repeated tile MMA",
            layout="bfloat16 -> float32",
            problem_size=f"{tile_count:,} x {mma_iterations:,}",
            work_per_launch=2.0 * tile_count * mma_iterations * _TILE_M * _TILE_K * _TILE_N / 1.0e9,
            unit="GFLOP/s",
            best_ms=best_ms,
            median_ms=median_ms,
            theoretical=theoretical_gflops,
        )
    )
    del a, b, c
    gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--mma-iterations", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    print_report(
        run(
            device_name=args.device,
            matrix_size=args.matrix_size,
            mma_iterations=args.mma_iterations,
            warmup=args.warmup,
            repetitions=args.repetitions,
            trials=args.trials,
        )
    )


if __name__ == "__main__":
    main()
