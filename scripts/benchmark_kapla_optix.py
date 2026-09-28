# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the PhoenX Kapla scene with real OptiX rendering.

Run with ``uv run python scripts/benchmark_kapla_optix.py --tower-grid 2x2
--num-frames 400``. The grid remains one PhoenX world.
"""

from __future__ import annotations

import functools
import importlib
import time

import newton
import newton.examples as examples


def _parse_grid(value: str) -> tuple[int, int]:
    try:
        x, y = (int(component) for component in value.lower().split("x"))
    except (ValueError, TypeError) as error:
        raise ValueError("tower grid must use NxM, for example 2x2") from error
    if x < 1 or y < 1:
        raise ValueError("tower grid dimensions must be positive")
    return x, y


def main() -> None:
    parser = examples.create_parser()
    parser.add_argument("--tower-grid", type=_parse_grid, default=(1, 1))
    parser.add_argument("--grid-broadphase", action="store_true", help="Use the uniform-grid rigid broad phase")
    parser.add_argument(
        "--implicit-tree-broadphase", action="store_true", help="Use the experimental implicit-BVH rigid broad phase"
    )
    parser.add_argument("--ticks-per-frame", type=int, default=None, help="Physics ticks per rendered frame")
    parser.add_argument("--sim-substeps", type=int, default=None, help="Solver substeps per physics tick")
    parser.add_argument("--solver-iterations", type=int, default=None, help="Biased contact sweeps per substep")
    parser.add_argument("--velocity-iterations", type=int, default=None, help="Velocity relaxation passes per substep")
    parser.add_argument("--trace-gpu-events", action="store_true", help="Sample OptiX and physics GPU durations")
    parser.add_argument(
        "--trace-on-slowdown", action="store_true", help="Start GPU event sampling after a frame exceeds 50 ms"
    )
    parser.add_argument(
        "--no-overlap", action="store_true", help="Run physics and rendering sequentially for comparison"
    )
    parser.set_defaults(viewer="optix", headless=True, quiet=True)
    parsed = parser.parse_args()
    if parsed.grid_broadphase and parsed.implicit_tree_broadphase:
        parser.error("select only one broad phase")
    module = importlib.import_module("newton._src.solvers.phoenx.examples.example_kapla_tower")
    module.TOWER_GRID_DIMS = parsed.tower_grid
    module.USE_GRID_BROAD_PHASE = parsed.grid_broadphase
    module.USE_IMPLICIT_TREE_BROAD_PHASE = parsed.implicit_tree_broadphase
    if parsed.ticks_per_frame is not None and parsed.ticks_per_frame < 1:
        parser.error("--ticks-per-frame must be positive")
    if parsed.sim_substeps is not None and parsed.sim_substeps < 1:
        parser.error("--sim-substeps must be positive")
    if parsed.solver_iterations is not None and parsed.solver_iterations < 1:
        parser.error("--solver-iterations must be positive")
    if parsed.velocity_iterations is not None and parsed.velocity_iterations < 1:
        parser.error("--velocity-iterations must be positive")

    original_example = module.Example

    class BenchmarkExample(original_example):
        def _build_scene(self):
            if parsed.ticks_per_frame is not None:
                self.steps_per_frame = parsed.ticks_per_frame
                self.fps = self.render_fps * self.steps_per_frame
                self.frame_dt = 1.0 / self.fps
            if parsed.sim_substeps is not None:
                self.sim_substeps = parsed.sim_substeps
            if parsed.solver_iterations is not None:
                self.solver_iterations = parsed.solver_iterations
            if parsed.velocity_iterations is not None:
                self.velocity_iterations = parsed.velocity_iterations
            super()._build_scene()

    # ViewerOptix's ordinary 16,384-instance default fits one tower. The
    # four-tower benchmark needs a larger allocation before scene logging.
    required_instances = parsed.tower_grid[0] * parsed.tower_grid[1] * module.NUM_BRICKS + 32
    if required_instances > 16384:
        newton.viewer.ViewerOptix = functools.partial(
            newton.viewer.ViewerOptix,
            max_instances=required_instances,
        )

    viewer, args = examples.init(parser)
    example = BenchmarkExample(viewer, args)
    if args.no_overlap:
        example.overlap_simulation_render = False
    gpu_samples = []
    trace_active = [not args.trace_on_slowdown]
    trigger_frame = [None]
    if args.trace_gpu_events or args.trace_on_slowdown:
        import warp as wp  # noqa: PLC0415
        from cuda.bindings import runtime as cuda_runtime  # noqa: PLC0415

        def timed(original, label, get_stream):
            count = 0

            def call():
                nonlocal count
                count += 1
                if not trace_active[0] or count % (1 if args.trace_on_slowdown else 20):
                    return original()
                error, start = cuda_runtime.cudaEventCreateWithFlags(cuda_runtime.cudaEventDefault)
                if int(error) != 0:
                    raise RuntimeError(f"CUDA event creation failed: {error}")
                error, end = cuda_runtime.cudaEventCreateWithFlags(cuda_runtime.cudaEventDefault)
                if int(error) != 0:
                    raise RuntimeError(f"CUDA event creation failed: {error}")
                stream = int(get_stream().cuda_stream)
                cuda_runtime.cudaEventRecord(start, stream)
                result = original()
                cuda_runtime.cudaEventRecord(end, stream)
                gpu_samples.append((label, count, start, end))
                return result

            return call

        example.step = timed(example.step, "physics", lambda: wp.get_stream(viewer.device))
        renderer = viewer._api.viewer
        renderer._launch_optix = timed(renderer._launch_optix, "optix", lambda: renderer._render_stream)
    frame_times: list[float] = []
    original_throttle = examples._throttle_render_fps

    def record_frame(start: float, cap: float | None) -> None:
        frame_times.append(time.perf_counter())
        frame = len(frame_times)
        if (
            args.trace_on_slowdown
            and trigger_frame[0] is None
            and frame > 40
            and frame_times[-1] - frame_times[-2] > 0.05
        ):
            if not trace_active[0]:
                print(f"gpu_trace_trigger_frame={frame}", flush=True)
                trigger_frame[0] = frame
                viewer.num_frames = frame + 40 if viewer.num_frames is None else min(viewer.num_frames, frame + 40)
            trace_active[0] = True
        if trigger_frame[0] is not None and frame >= trigger_frame[0] + 40:
            trace_active[0] = False
        if frame > 20 and frame % 20 == 0:
            fps = 20.0 / (frame_times[-1] - frame_times[-21])
            intervals = [frame_times[i] - frame_times[i - 1] for i in range(frame - 20, frame)]
            worst = max(range(len(intervals)), key=intervals.__getitem__)
            print(
                f"frame={frame} fps20={fps:.2f} worst_frame={frame - 19 + worst} "
                f"worst_ms={1000.0 * intervals[worst]:.2f}",
                flush=True,
            )
        if frame == 80:
            print(f"steady_fps={50.0 / (frame_times[-1] - frame_times[29]):.2f}", flush=True)
        if frame == 400:
            print(f"late_fps={50.0 / (frame_times[-1] - frame_times[349]):.2f}", flush=True)
        if frame == args.num_frames and frame > 41:
            print(f"sustained_fps={float(frame - 41) / (frame_times[-1] - frame_times[40]):.2f}", flush=True)
        original_throttle(start, cap)

    examples._throttle_render_fps = record_frame
    try:
        examples.run(example, args)
    finally:
        examples._throttle_render_fps = original_throttle
        if args.trace_gpu_events or args.trace_on_slowdown:
            for label, frame, start, end in gpu_samples:
                cuda_runtime.cudaEventSynchronize(end)
                error, duration_ms = cuda_runtime.cudaEventElapsedTime(start, end)
                if int(error) == 0:
                    print(f"gpu_sample={label} frame={frame} ms={duration_ms:.3f}", flush=True)
                cuda_runtime.cudaEventDestroy(start)
                cuda_runtime.cudaEventDestroy(end)


if __name__ == "__main__":
    main()
