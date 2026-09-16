# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Summarize an isolated Nsight capture without interpreting instrumented time as FPS."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    """Report kernel costs and overlap from a known captured frame interval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT k.start,k.end,s.value,k.registersPerThread,k.localMemoryPerThread "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON k.demangledName=s.id "
            "WHERE k.deviceId=? ORDER BY k.start",
            (args.device,),
        ).fetchall()
    if not rows:
        raise RuntimeError("Capture has no kernel events for the requested device")
    groups = defaultdict(list)
    merged_ns = 0
    left, right = rows[0][:2]
    for start, end, name, registers, local_bytes in rows:
        if end < start:
            raise ValueError("Kernel end precedes start")
        groups[name].append((end - start, registers, local_bytes))
        if start > right:
            merged_ns += right - left
            left, right = start, end
        else:
            right = max(right, end)
    merged_ns += right - left
    costs = []
    for name, launches in groups.items():
        duration = np.array([item[0] for item in launches], dtype=np.float64)
        costs.append(
            {
                "kernel": name,
                "calls": len(launches),
                "calls_per_frame": len(launches) / args.frames,
                "ms_per_frame": float(duration.sum() / (1e6 * args.frames)),
                "mean_us": float(duration.mean() / 1e3),
                "p95_us": float(np.percentile(duration, 95) / 1e3),
                "max_us": float(duration.max() / 1e3),
                "registers_per_thread": sorted({item[1] for item in launches}),
                "local_bytes_per_thread": sorted({item[2] for item in launches}),
            }
        )
    costs.sort(key=lambda item: item["ms_per_frame"], reverse=True)
    report = {
        "database": str(args.database.resolve()),
        "captured_frames": args.frames,
        "device": args.device,
        "scope": "Instrumented kernel attribution; neither wall-clock physics time nor FPS",
        "kernel_calls": len(rows),
        "kernel_sum_ms_per_frame": sum(item["ms_per_frame"] for item in costs),
        "kernel_union_ms_per_frame": merged_ns / (1e6 * args.frames),
        "first_to_last_kernel_ms_per_frame": (max(row[1] for row in rows) - rows[0][0]) / (1e6 * args.frames),
        "kernels": costs,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "kernels"}, indent=2))
    for item in costs[:12]:
        print(f"{item['ms_per_frame']:.6f} ms/frame {item['kernel']}")


if __name__ == "__main__":
    main()
