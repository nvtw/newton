# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the complete PhoenX speed-of-light calibration suite."""

from __future__ import annotations

import argparse

import warp as wp

from . import bench_compute, bench_memory, bench_random_access
from .common import print_report, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to calibrate")
    parser.add_argument("--array-mib", type=int, default=256, help="MiB in each data array; use a power of two")
    parser.add_argument("--compute-elements", type=int, default=1_048_576)
    parser.add_argument("--compute-iterations", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use small inputs for a fast smoke run")
    args = parser.parse_args()

    wp.init()
    device = require_cuda(args.device)
    if args.quick:
        args.array_mib = 8
        args.compute_elements = 65_536
        args.compute_iterations = 32
        args.warmup = 2
        args.repetitions = 3
        args.trials = 2

    total_gib = float(device.total_memory) / 1024**3
    print("PhoenX speed-of-light calibration")
    print(f"Device: {device.name} ({device})")
    print(f"CUDA architecture: {device.arch} | Device memory: {total_gib:.1f} GiB")
    print(
        f"Timing: {args.warmup} warmup, {args.repetitions} replays/trial, {args.trials} trials; peak is the best trial"
    )

    common = {
        "device_name": args.device,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "trials": args.trials,
    }
    results = []
    results.extend(bench_memory.run(array_mib=args.array_mib, **common))
    results.extend(bench_random_access.run(array_mib=args.array_mib, **common))
    results.extend(
        bench_compute.run(
            elements=args.compute_elements,
            iterations=args.compute_iterations,
            **common,
        )
    )
    print_report(results)
    print("\nNotes: memory GB/s counts logical kernel traffic; random access includes index reads and output writes.")
    print("FMA uses two FLOPs per multiply-add. Reciprocal sqrt is reported as operations/s, not FLOPs/s.")


if __name__ == "__main__":
    main()
