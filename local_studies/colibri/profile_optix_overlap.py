# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare serialized and overlapping Colibri physics/OptiX on identical snapshots."""

import argparse
import ctypes
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from cuda.bindings import runtime

from newton.examples.phoenx.example_phoenx_colibri import Example
from newton.viewer import ViewerOptix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serial", "overlap", "priority"), required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    viewer = ViewerOptix(headless=True, width=1920, height=1080, vsync=False)
    example = Example(viewer, Example.create_parser().parse_args(["--num-worlds", str(args.num_worlds)]))
    render_stream = wp.get_stream(example.model.device)
    create_stream = runtime.cudaStreamCreateWithPriority
    cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so") if args.profile else None
    times = []
    for frame in range(args.warmup + args.frames):
        if frame == args.warmup and cuda is not None:
            cuda.cudaProfilerStart()
        start = time.perf_counter()
        if args.mode == "serial":
            example.prepare_render_state()
            example.step()
        elif args.mode == "overlap":
            with patch.object(
                runtime, "cudaStreamCreateWithPriority", side_effect=lambda flags, priority: create_stream(flags, 0)
            ):
                viewer.launch_simulation_step(example.step, prepare_render_state=example.prepare_render_state)
        else:
            viewer.launch_simulation_step(example.step, prepare_render_state=example.prepare_render_state)
        example.render()
        viewer.synchronize_simulation_step()
        wp.synchronize_stream(render_stream)
        if frame >= args.warmup:
            times.append((time.perf_counter() - start) * 1000)
        if frame % 30 == 0:
            print(f"Frame {frame}", flush=True)
    if cuda is not None:
        cuda.cudaProfilerStop()
    render_priority = runtime.cudaStreamGetPriority(render_stream.cuda_stream)[1]
    simulation_priority = (
        runtime.cudaStreamGetPriority(viewer._deferred_simulation_raw_stream)[1]
        if args.mode != "serial"
        else render_priority
    )
    report = {
        "render_priority": render_priority,
        "simulation_priority": simulation_priority,
        "mode": args.mode,
        "num_worlds": args.num_worlds,
        "frames": args.frames,
        "warmup": args.warmup,
        "width": 1920,
        "height": 1080,
        "dlss_enabled": viewer._api.dlss_enabled,
        "dlss_quality": viewer.dlss_quality,
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "fps": 1000 / float(np.mean(times)),
        "scope": "Headless OptiX plus physics; both streams complete each measured frame",
    }
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        times_ms=times,
        poses=example.state_0.body_q.numpy(),
        velocities=example.state_0.body_qd.numpy(),
        image=viewer._api.get_frame_uint8(),
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    viewer.close()


if __name__ == "__main__":
    main()
