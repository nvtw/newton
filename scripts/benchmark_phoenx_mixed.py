# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the unchanged Kapla/Hoberman scene with FHD OptiX rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples import example_kapla_hoberman as scene
from newton._src.solvers.phoenx.examples.example_kapla_hoberman import Example
from newton.viewer import ViewerOptix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--measure-after", type=int, default=300)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--record-checkpoints", action="store_true")
    parser.add_argument("--quality-every", type=int, default=0)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--blocks-per-sm", type=int)
    parser.add_argument("--singleton-lanes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--joint-priority", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.measure_after < args.frames:
        parser.error("--measure-after must be smaller than --frames")
    utilization = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True
        ).splitlines()[0]
    )
    if utilization >= 10:
        raise SystemExit(f"GPU idle gate failed: {utilization}%")
    print(f"GPU idle gate: {utilization}%. Rendering at 1920x1080.", flush=True)
    if args.blocks_per_sm is not None or args.singleton_lanes is not None or args.joint_priority is not None:
        original_solver = scene.newton.solvers.SolverPhoenX

        def configured_solver(model, **kwargs):
            if args.blocks_per_sm is not None:
                kwargs["max_thread_blocks"] = args.blocks_per_sm * model.device.sm_count
            if args.singleton_lanes is not None:
                kwargs["colored_singleton_contact_lanes"] = args.singleton_lanes
            if args.joint_priority is not None:
                kwargs["joint_coloring_priority"] = args.joint_priority
            return original_solver(model, **kwargs)

        scene.newton.solvers.SolverPhoenX = configured_solver
    viewer = ViewerOptix(width=1920, height=1080, headless=True, enable_imgui=False, max_instances=65536)
    checkpoints = []
    quality_samples = []
    example = None
    try:
        example = Example(viewer, SimpleNamespace())
        assert example.solver.world.velocity_iterations == 1
        assert not example.solver.world._sleeping_enabled
        print(
            f"Settings: ticks/frame=2 substeps/tick={example.substeps} sweeps={example.solver_iterations} "
            "velocity_iterations=1 sleeping=off; springs and geometry unchanged.",
            flush=True,
        )
        started = None
        profile_active = False
        for frame in range(args.frames):
            if frame == args.measure_after:
                viewer.synchronize_simulation_step()
                wp.synchronize()
                if args.profile:
                    from cuda.bindings import runtime  # noqa: PLC0415

                    (error,) = runtime.cudaProfilerStart()
                    if error != runtime.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"cudaProfilerStart failed: {error}")
                    profile_active = True
                started = time.perf_counter()
            viewer.synchronize_simulation_step()
            viewer.launch_simulation_step(example.step, prepare_render_state=example.prepare_render_state)
            example.render()
            if args.record_checkpoints and (frame + 1) % 50 == 0:
                viewer.synchronize_simulation_step()
                poses = example.state.body_q.numpy()
                velocities = example.state.body_qd.numpy()
                checkpoints.append(
                    {"frame": frame + 1, "hash": hashlib.sha256(poses.tobytes() + velocities.tobytes()).hexdigest()}
                )
            if (frame + 1) % 60 == 0:
                print(f"Rendered {frame + 1}/{args.frames} frames", flush=True)
            if args.quality_every and (frame + 1) % args.quality_every == 0:
                viewer.synchronize_simulation_step()
                quality_samples.append({"frame": frame + 1, **example.measure_quality()})
        viewer.synchronize_simulation_step()
        wp.synchronize()
        elapsed = time.perf_counter() - started
        if profile_active:
            runtime.cudaProfilerStop()
        quality_error = None
        try:
            example.test_final()
        except AssertionError as error:
            quality_error = str(error)
        quality = example.measure_quality()
        for sample in quality_samples:
            if (
                sample["maximum_joint_attachment_error_m"] >= 0.02
                or sample["maximum_hinge_axis_error_deg"] >= 10.0
                or sample["tower_bricks_fallen_50cm"] != 0
            ):
                quality_error = f"Quality gate failed at frame {sample['frame']}: {sample}"
        poses = example.state.body_q.numpy()
        velocities = example.state.body_qd.numpy()
        state_hash = hashlib.sha256(poses.tobytes() + velocities.tobytes()).hexdigest()
        measured_frames = args.frames - args.measure_after
        result = {
            "frames": args.frames,
            "measured_frames": measured_frames,
            "rendered_fps": measured_frames / elapsed,
            "ms_per_frame": 1000.0 * elapsed / measured_frames,
            "state_hash": state_hash,
            "checkpoints": checkpoints,
            "checkpoint_monitoring": args.record_checkpoints,
            "quality_monitoring": bool(args.quality_every),
            "quality": quality,
            "quality_samples": quality_samples,
            "render_resolution": [1920, 1080],
            "gpu_start_utilization": utilization,
            "contact_points": int(example.contacts.rigid_contact_count.numpy()[0]),
            "quality_passed": quality_error is None,
            "quality_error": quality_error,
            "grid_threads": example.solver.world._singleworld_total_threads,
            "singleton_lanes": example.solver.world._colored_singleton_contact_lanes,
            "joint_coloring_priority": example.solver.world.joint_coloring_priority,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output.with_suffix(".npz"), poses=poses, velocities=velocities)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.snapshot:
            from PIL import Image

            Image.fromarray(viewer.get_frame().numpy()).save(args.output.with_suffix(".png"))
        print(json.dumps(result), flush=True)
        if quality_error is not None:
            raise AssertionError(quality_error)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
