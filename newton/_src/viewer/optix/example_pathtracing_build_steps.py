# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step-by-step path tracing bring-up entrypoint.

This sample lives in ``warp/examples/core/pyoptix`` as requested and builds up in
small steps starting from the working MiniRenderer framework:

Step 1:
    Use only general-purpose pyoptix helpers (`mini_renderer.py`) to validate
    OptiX context/pipeline/SBT/launch on a tiny scene.

Step 2:
    Run the translated emissive gallery path tracing sample from
    ``pyoptix/pathtracing`` (DLSS disabled).

Step 3:
    Open a realtime live viewer window rendering the pathtracing sample.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import warp as wp
from newton._src.viewer.optix.example_pathtracing_live_viewer import _run_gui as _run_step3_live_gui
from newton._src.viewer.optix.mini_renderer import MiniRenderer, pack_rgba8
from newton._src.viewer.optix.pathtracing.emissive_gallery_sample import create_emissive_gallery_viewer


def _run_step1_mini_renderer(width: int, height: int, frames: int) -> int:
    """Step 1: minimal helper-framework rendering smoke test."""
    print("[Step 1] MiniRenderer helper-framework smoke test")
    renderer = MiniRenderer(width=width, height=height)

    cube = renderer.register_cube(0.8)
    plane = renderer.register_plane(30.0, 30.0)

    color_index = 0
    for x in (-2, 0, 2):
        for y in (0, 2):
            for z in (-2, 0, 2):
                color = pack_rgba8(60 + color_index * 20, 120 + color_index * 10, 220 - color_index * 8, 255)
                renderer.add_render_instance(cube, [x, y, z, 0.0, 0.0, 0.0, 1.0], color_rgba8=color)
                color_index += 1

    renderer.add_render_instance(
        plane, [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], color_rgba8=pack_rgba8(220, 220, 220, 255)
    )
    renderer.set_camera_pose([8.0, 5.0, 9.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0])
    renderer.rebuild_instance_tree()

    for _ in range(max(1, int(frames))):
        renderer.render_frame()

    img = renderer.color.numpy().reshape(height, width)
    checksum = int(np.bitwise_xor.reduce(img.reshape(-1).astype(np.uint32)))
    print(f"[Step 1] OK checksum={checksum}")
    return checksum


def _run_step2_pathtracing(width: int, height: int, frames: int) -> tuple[tuple[int, ...], float, float]:
    """Step 2: pathtracing emissive gallery test run."""
    print("[Step 2] Pathtracing emissive gallery")
    viewer = create_emissive_gallery_viewer(width=width, height=height)
    if not viewer.build():
        raise RuntimeError("PathTracingViewer build failed")

    frame_count = max(1, int(frames))
    t0 = time.perf_counter()
    for i in range(frame_count):
        viewer.render()
        if (i + 1) % 4 == 0:
            print(f"[Step 2] Frame {i + 1}/{frames}")
    wp.synchronize_device("cuda")
    dt = max(time.perf_counter() - t0, 1.0e-9)
    fps = frame_count / dt
    ms_per_frame = 1000.0 / fps

    output = viewer.get_output()
    output_shape = tuple(output.shape)
    output_min = float(output.min())
    output_max = float(output.max())
    print(
        f"[Step 2] OK shape={output_shape} range=[{output_min:.4f}, {output_max:.4f}] "
        f"fps={fps:.2f} ms/frame={ms_per_frame:.2f}"
    )
    return output_shape, output_min, output_max


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--step", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames-step1", type=int, default=4)
    parser.add_argument("--frames-step2", type=int, default=16)
    parser.add_argument("--frames-step3", type=int, default=0, help="0 means run until window close")
    args = parser.parse_args()

    wp.init()
    print("=" * 72)
    print("PyOptiX Pathtracing Build Steps")
    print("=" * 72)
    print(f"File: {Path(__file__).name}")

    if args.step in ("1", "all"):
        _run_step1_mini_renderer(args.width, args.height, args.frames_step1)

    if args.step in ("2", "all"):
        _run_step2_pathtracing(args.width, args.height, args.frames_step2)

    if args.step in ("3", "all"):
        print("[Step 3] Pathtracing live viewer")
        _run_step3_live_gui(args.width, args.height, args.fps, args.frames_step3)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
