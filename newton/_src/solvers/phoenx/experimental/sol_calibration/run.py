# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the complete PhoenX speed-of-light calibration suite."""

from __future__ import annotations

import argparse

import warp as wp

from . import bench_compute, bench_memory, bench_random_access, bench_tensor
from .common import get_hardware_limits, print_report, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to calibrate")
    parser.add_argument("--array-mib", type=int, default=256, help="MiB in each data array; use a power of two")
    parser.add_argument("--compute-elements", type=int, default=1_048_576)
    parser.add_argument("--compute-iterations", type=int, default=256)
    parser.add_argument("--tensor-matrix-size", type=int, default=8192)
    parser.add_argument("--tensor-mma-iterations", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--memory-peak-gbps", type=float)
    parser.add_argument("--fp32-peak-gflops", type=float)
    parser.add_argument("--bf16-tensor-peak-gflops", type=float)
    parser.add_argument("--quick", action="store_true", help="Use small inputs for a fast smoke run")
    args = parser.parse_args()

    wp.init()
    device = require_cuda(args.device)
    if args.quick:
        args.array_mib = 8
        args.compute_elements = 65_536
        args.compute_iterations = 32
        args.tensor_matrix_size = 2048
        args.tensor_mma_iterations = 256
        args.warmup = 2
        args.repetitions = 3
        args.trials = 2

    total_gib = float(device.total_memory) / 1024**3
    limits = get_hardware_limits(device)
    memory_peak = args.memory_peak_gbps or limits.memory_gbps
    fp32_peak = args.fp32_peak_gflops or limits.fp32_gflops
    bf16_tensor_peak = args.bf16_tensor_peak_gflops or limits.bf16_tensor_gflops
    if args.quick:
        # Quick inputs are cache-resident and too short for stable ceiling estimates.
        memory_peak = 0.0
        fp32_peak = 0.0
        bf16_tensor_peak = 0.0
    print("PhoenX speed-of-light calibration")
    print(f"Device: {device.name} ({device})")
    print(f"CUDA architecture: {device.arch} | Device memory: {total_gib:.1f} GiB")
    print(
        f"Timing: {args.warmup} warmup, {args.repetitions} replays/trial, {args.trials} trials; peak is the best trial"
    )
    if args.quick:
        print("Mode: quick smoke test; cache-resident throughput is not a hardware roofline calibration")

    common = {
        "device_name": args.device,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "trials": args.trials,
    }
    results = []
    results.extend(bench_memory.run(array_mib=args.array_mib, theoretical_gbps=memory_peak, **common))
    results.extend(bench_random_access.run(array_mib=args.array_mib, theoretical_gbps=memory_peak, **common))
    results.extend(
        bench_compute.run(
            elements=args.compute_elements,
            iterations=args.compute_iterations,
            theoretical_gflops=fp32_peak,
            **common,
        )
    )
    results.extend(
        bench_tensor.run(
            matrix_size=args.tensor_matrix_size,
            mma_iterations=args.tensor_mma_iterations,
            theoretical_gflops=bf16_tensor_peak,
            **common,
        )
    )
    print_report(results)
    print("\nNotes: memory GB/s counts logical kernel traffic; random access includes index reads and output writes.")
    print("FMA uses two FLOPs per multiply-add. Reciprocal sqrt is reported as operations/s, not FLOPs/s.")


if __name__ == "__main__":
    main()
