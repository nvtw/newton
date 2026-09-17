# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure captured Colibri physics separately from rendering and validation.

Use --profile with nsys --capture-range=cudaProfilerApi --capture-range-end=stop.
The warmup advances the scene; --validate checks warmup and measured frames.
"""

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.joint_accuracy import measure as measure_joints
from newton.examples.kamino.example_kamino_colibri import Example as AssemblyChecks
from newton.examples.phoenx.example_phoenx_colibri import Example
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--motor-off",
        action="store_true",
        help="Measure the unpowered mechanism; exclude powered shaking from creep tests.",
    )
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup must be nonnegative")
    if args.profile and args.validate:
        parser.error("Run profiling and validation separately to keep audit kernels out of the trace")
    example = Example(ViewerNull(), Example.create_parser().parse_args(["--motor-off"] if args.motor_off else []))
    times, poses, velocities = [], [], []
    peak_depth = 0.0
    support_failure = None
    joint_peaks = {}
    cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so") if args.profile else None
    for frame in range(args.warmup + args.frames):
        if frame == args.warmup:
            wp.synchronize_device(example.model.device)
            if cuda is not None:
                cuda.cudaProfilerStart()
        start = time.perf_counter()
        example.step()
        wp.synchronize_device(example.model.device)
        elapsed = (time.perf_counter() - start) * 1000
        if frame >= args.warmup:
            times.append(elapsed)
        if args.validate:
            AssemblyChecks.test_post_step(example)
            example._test_drive_tracking()
            depth, labels, _ = example._measure_contact_penetration()
            peak_depth = max(peak_depth, depth)
            assert depth < 0.001, (frame, depth, labels)
            try:
                example._test_support_stationarity()
            except AssertionError as exc:
                if support_failure is None:
                    support_failure = {"frame": frame, "error": str(exc)}
            body_q = example.state_0.body_q.numpy()
            joint_errors = measure_joints(body_q, example.model.body_label)
            for row in joint_errors["joints"]:
                for metric in ("anchor_error_m", "axis_error_rad"):
                    if metric in row and (metric not in joint_peaks or row[metric] > joint_peaks[metric]["value"]):
                        joint_peaks[metric] = {"value": row[metric], "joint": row["joint"], "frame": frame}
            poses.append(body_q)
            velocities.append(example.state_0.body_qd.numpy())
        if frame % 600 == 0:
            print(f"Frame {frame}: {elapsed:.3f} ms", flush=True)
    if cuda is not None:
        cuda.cudaProfilerStop()
    report = {
        "frames": args.frames,
        "warmup": args.warmup,
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "peak_depth_m": peak_depth if args.validate else None,
        "support_failure": support_failure,
        "motor_enabled": example.motor_enabled,
        "joint_peaks": joint_peaks if args.validate else None,
        "collision_hz": 120,
        "substeps_per_refresh": 24,
        "rendering": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        times_ms=times,
        labels=example.model.body_label,
        poses=poses if args.validate else example.state_0.body_q.numpy(),
        velocities=velocities if args.validate else example.state_0.body_qd.numpy(),
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
