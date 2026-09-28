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
    parser.set_defaults(viewer="optix", headless=True, quiet=True)
    parsed = parser.parse_args()
    module = importlib.import_module("newton._src.solvers.phoenx.examples.example_kapla_tower")
    module.TOWER_GRID_DIMS = parsed.tower_grid

    # ViewerOptix's ordinary 16,384-instance default fits one tower. The
    # four-tower benchmark needs a larger allocation before scene logging.
    required_instances = parsed.tower_grid[0] * parsed.tower_grid[1] * module.NUM_BRICKS + 32
    if required_instances > 16384:
        newton.viewer.ViewerOptix = functools.partial(
            newton.viewer.ViewerOptix,
            max_instances=required_instances,
        )

    viewer, args = examples.init(parser)
    example = module.Example(viewer, args)
    frame_times: list[float] = []
    original_throttle = examples._throttle_render_fps

    def record_frame(start: float, cap: float | None) -> None:
        frame_times.append(time.perf_counter())
        frame = len(frame_times)
        if frame > 20 and frame % 20 == 0:
            fps = 20.0 / (frame_times[-1] - frame_times[-21])
            print(f"frame={frame} fps20={fps:.2f}", flush=True)
        if frame == 80:
            print(f"steady_fps={50.0 / (frame_times[-1] - frame_times[29]):.2f}", flush=True)
        if frame == 400:
            print(f"late_fps={50.0 / (frame_times[-1] - frame_times[349]):.2f}", flush=True)
        original_throttle(start, cap)

    examples._throttle_render_fps = record_frame
    try:
        examples.run(example, args)
    finally:
        examples._throttle_render_fps = original_throttle


if __name__ == "__main__":
    main()
