# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure lower Colibri substep counts against unchanged scene quality gates.

Collision updates remain at 120 Hz. Each count runs 60 warmup frames plus
--frames measured frames, or stops on the first quality failure. Error peaks
include warmup; times exclude it. Failed runs report partial trajectories,
not full-duration error bounds. This is separate from fixed-work performance
optimization and does not change example defaults.
"""

import argparse
import gc
import json
import time

import numpy as np
import warp as wp

from local_studies.colibri.joint_accuracy import measure
from newton.examples.kamino.example_kamino_colibri import Example as Checks
from newton.examples.phoenx.example_phoenx_colibri import Example
from newton.viewer import ViewerNull


def main():
    """Run each requested count and save timing, error peaks, and failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substeps", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--motor-off", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.worlds < 1 or min(args.substeps) < 1:
        parser.error("Frame, world and substep counts must be positive")
    reports = []
    for substeps in args.substeps:
        options = ["--num-worlds", str(args.worlds), "--substeps", str(substeps)]
        if args.motor_off:
            options.append("--motor-off")
        example = Example(ViewerNull(), Example.create_parser().parse_args(options))
        times = []
        peaks = {"penetration_m": 0.0, "anchor_m": 0.0, "axis_rad": 0.0}
        failure = None
        for frame in range(args.frames + 60):
            start = time.perf_counter()
            example.step()
            wp.synchronize_device(example.model.device)
            elapsed = (time.perf_counter() - start) * 1000
            if frame >= 60:
                times.append(elapsed)
            q = example.state_0.body_q.numpy()
            if not np.isfinite(q).all():
                failure = "Non-finite poses"
                break
            n = example.model.body_count // args.worlds
            for world in range(args.worlds):
                joints = measure(q[world * n : (world + 1) * n], example.model.body_label[world * n : (world + 1) * n])
                for row in joints["joints"]:
                    peaks["anchor_m"] = max(peaks["anchor_m"], row.get("anchor_error_m", 0))
                    peaks["axis_rad"] = max(peaks["axis_rad"], row.get("axis_error_rad", 0))
            depth, labels, _ = example._measure_contact_penetration()
            peaks["penetration_m"] = max(peaks["penetration_m"], depth)
            try:
                Checks.test_post_step(example)
                example._test_drive_tracking()
                example._test_support_stationarity()
                assert depth < 0.001, f"Penetration {depth:.6f} m: {labels}"
            except AssertionError as error:
                failure = str(error)
                break
            if frame and frame % 600 == 0:
                print("progress", substeps, frame, flush=True)
        report = {
            "substeps_per_refresh": substeps,
            "collision_hz": 120,
            "worlds": args.worlds,
            "motor_off": args.motor_off,
            "completed_frames": frame + 1,
            "requested_frames": args.frames + 60,
            "mean_ms": float(np.mean(times)) if times else None,
            "peaks": peaks,
            "failure": failure,
        }
        reports.append(report)
        with open(args.output, "w") as f:
            json.dump(reports, f, indent=2)
        print(json.dumps(report), flush=True)
        del example
        gc.collect()


if __name__ == "__main__":
    main()
