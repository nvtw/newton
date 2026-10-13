# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the full ragdoll scene with rendering and persist replay evidence."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.phoenx.example_phoenx_ragdoll_pile import Example
from newton.viewer import ViewerOptix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--measure-after", type=int, default=300)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.measure_after < args.frames:
        parser.error("--measure-after must be nonnegative and smaller than --frames")
    utilization = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True
        ).splitlines()[0]
    )
    if utilization >= 10:
        raise SystemExit(f"GPU idle gate failed: {utilization}%")
    print(f"GPU idle gate: {utilization}%", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    viewer = ViewerOptix(width=1280, height=720, headless=True, enable_imgui=False, max_instances=192001)
    try:
        example = Example(
            viewer, SimpleNamespace(grid=(40, 40, 20), seed=42, substeps=8, iterations=4, joint_solver="exact")
        )
        assert example.solver.world.velocity_iterations == 1
        assert not example.solver.world._sleeping_enabled
        for frame in range(args.frames):
            if frame == args.measure_after:
                viewer.synchronize_simulation_step()
                wp.synchronize()
                if args.profile:
                    from cuda.bindings import runtime  # noqa: PLC0415

                    (error,) = runtime.cudaProfilerStart()
                    if error != runtime.cudaError_t.cudaSuccess:
                        raise RuntimeError(error)
                started = time.perf_counter()
            viewer.synchronize_simulation_step()
            viewer.launch_simulation_step(example.step, prepare_render_state=example.prepare_render_state)
            example.render()
            if (frame + 1) % 60 == 0:
                print(f"Rendered {frame + 1}/{args.frames}", flush=True)
        viewer.synchronize_simulation_step()
        wp.synchronize()
        elapsed = time.perf_counter() - started
        if args.profile:
            runtime.cudaProfilerStop()
        poses = example.state.body_q.numpy()
        velocities = example.state.body_qd.numpy()
        quality_error = None
        try:
            example.test_final()
        except AssertionError as error:
            quality_error = str(error)
        result = {
            "frames": args.frames,
            "measured_frames": args.frames - args.measure_after,
            "rendered_fps": (args.frames - args.measure_after) / elapsed,
            "ms_per_frame": elapsed * 1000 / (args.frames - args.measure_after),
            "render_resolution": [1280, 720],
            "substeps": 8,
            "solver_iterations": 4,
            "velocity_iterations": 1,
            "state_hash": hashlib.sha256(poses.tobytes() + velocities.tobytes()).hexdigest(),
            "final_quality_passed": quality_error is None,
            "quality_error": quality_error,
            "contact_points": int(example.contacts.rigid_contact_count.numpy()[0]),
            "gpu_start_utilization": utilization,
        }
        np.savez(args.output.with_suffix(".npz"), poses=poses, velocities=velocities)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if quality_error is not None:
            raise AssertionError(quality_error)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
