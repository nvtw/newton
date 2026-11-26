# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure rendered factory frames as cartons accumulate on the pallets.

Run `uv run scripts/benchmark_phoenx_bottling_plant.py` with the local plant
assets installed. The default run uses headless OptiX and advances 330 seconds
of simulation. Use `--window START STOP` to select frame intervals (the run
then stops after the last window unless `--num-frames` is given) and
`--timing-output FILE` to save the timings and final controller report, and
`--profile` under `nsys profile --capture-range=cudaProfilerApi` to trace
only the first window.
Each window records contact and production counts at both ends so runs can
be compared at the same factory workload.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

import newton.examples
from newton.examples.phoenx.example_phoenx_bottling_plant import Example


def _workload(example):
    workload = {"rigid_contacts": int(example.contacts.rigid_contact_count.numpy()[0])}
    controller = example.controller
    if controller is not None:
        workload.update(
            placed_bottles=controller.placed.numpy().tolist(),
            sealed_cases=int(controller.sealed.numpy().sum()),
            palletized_cases={name: len(plc.placed) for name, plc in controller.transport.palletizers.items()},
        )
    return workload


def main():
    parser = Example.create_parser()
    parser.set_defaults(viewer="optix", headless=True, num_frames=None)
    parser.add_argument("--window", type=int, nargs=2, action="append", metavar=("START", "STOP"))
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Limit an `nsys --capture-range=cudaProfilerApi` run to the first window.",
    )
    # Simulate only up to the last requested window; the full run is 330 s of factory time.
    requested, _ = parser.parse_known_args()
    if requested.num_frames is None:
        parser.set_defaults(num_frames=max(stop for _, stop in requested.window) if requested.window else 19_800)
    viewer, args = newton.examples.init(parser)
    windows = args.window or [(600, 1800), (6600, 7800), (13_200, 14_400)]
    if any(start < 0 or start >= stop or stop > args.num_frames for start, stop in windows):
        parser.error("each timing window must fit within --num-frames and contain at least one frame")

    example = Example(viewer, args)
    running = viewer.is_running
    boundaries = []
    endpoints = {frame for window in windows for frame in window}
    timings = []
    workloads = {}
    observation_time = 0.0

    def measure_boundary():
        nonlocal observation_time
        frame = len(boundaries)
        if frame in endpoints:
            viewer.synchronize_simulation_step()
            wp.synchronize_device(example.model.device)
            if args.profile and frame in windows[0]:
                from cuda.bindings import runtime  # noqa: PLC0415

                toggle = runtime.cudaProfilerStart if frame == windows[0][0] else runtime.cudaProfilerStop
                (error,) = toggle()
                if error != runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"{toggle.__name__} failed: {error}")
        now = perf_counter()
        boundaries.append(now - observation_time)
        if frame in endpoints:
            workloads[frame] = _workload(example)
        for start, stop in windows:
            if frame == stop:
                elapsed = np.diff(boundaries[start : stop + 1]) * 1000
                median, p95, p99 = np.percentile(elapsed, (50, 95, 99))
                result = {
                    "start": start,
                    "stop": stop,
                    "fps": 1000 / elapsed.mean(),
                    "median_ms": median,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "start_workload": workloads[start],
                    "stop_workload": workloads[stop],
                }
                timings.append(result)
                print(json.dumps(result), flush=True)
        if frame in endpoints:
            # Shared or overlapping windows can contain another window's readback and reporting.
            observation_time += perf_counter() - now
        return frame < args.num_frames and running()

    # Keep the normal viewer loop, including its render snapshot and CUDA stream ordering.
    viewer.is_running = measure_boundary
    newton.examples.run(example, args)
    if len(boundaries) - 1 != args.num_frames:
        raise RuntimeError(f"Viewer stopped after {len(boundaries) - 1} of {args.num_frames} frames")
    expected_time = args.num_frames * example.frame_dt
    if not np.isclose(example.sim_time, expected_time, rtol=1e-9, atol=1e-9):
        raise RuntimeError(f"Factory advanced {example.sim_time}s while rendering {expected_time}s of frames")
    example.test_final()
    if args.timing_output:
        result = {
            "device": example.model.device.name,
            "viewer": args.viewer,
            "frames": args.num_frames,
            "sim_time": example.sim_time,
            "timings": timings,
            "production": example.controller.report() if example.controller is not None else None,
        }
        Path(args.timing_output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
