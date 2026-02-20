# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Step-by-step OptiX bring-up script with optional external scene loading."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

try:
    from newton._src.viewer.optix.example_a_beautiful_game_live_viewer import _run_gui as _run_step3_live_gui
    from newton._src.viewer.optix.mini_renderer import MiniRenderer, pack_rgba8
    from newton._src.viewer.optix.pathtracing import PathTracingBridge, create_a_beautiful_game_viewer
except ImportError:
    # Support direct script execution from this folder.
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    stale = [name for name in sys.modules if name == "newton" or name.startswith("newton.")]
    for name in stale:
        del sys.modules[name]
    from newton._src.viewer.optix.example_a_beautiful_game_live_viewer import _run_gui as _run_step3_live_gui
    from newton._src.viewer.optix.mini_renderer import MiniRenderer, pack_rgba8
    from newton._src.viewer.optix.pathtracing import PathTracingBridge, create_a_beautiful_game_viewer


def _run_step1_mini_renderer(width: int, height: int, frames: int) -> int:
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


def _run_step2_pathtracing(
    width: int,
    height: int,
    frames: int,
    scene_gltf: str | None = None,
    scene_obj: str | None = None,
    use_procedural_sky: bool = True,
) -> tuple[tuple[int, ...], float, float]:
    print("[Step 2] OptiX pathtracing")

    if scene_gltf or scene_obj:
        bridge = PathTracingBridge(width=width, height=height, enable_dlss_rr=True)
        bridge.initialize()
        loaded = False
        if scene_gltf:
            loaded = bridge.load_scene_from_gltf(scene_gltf)
        if (not loaded) and scene_obj:
            loaded = bridge.load_scene_from_obj(scene_obj)
        if not loaded:
            raise RuntimeError("External scene failed to load from provided glTF/OBJ paths")
        if use_procedural_sky:
            bridge.set_use_procedural_sky(True)
            bridge.set_sky_parameters(
                sun_direction=(0.0, 1.0, 0.5),
                multiplier=1.0,
                haze=0.0,
                red_blue_shift=0.0,
                saturation=1.0,
                horizon_height=0.0,
                ground_color=(0.4, 0.4, 0.4),
                horizon_blur=1.0,
                night_color=(0.0, 0.0, 0.0),
                sun_disk_intensity=1.0,
                sun_disk_scale=1.0,
                sun_glow_intensity=1.0,
                y_is_up=1,
            )
        bridge.set_camera_angles(position=(0.0, 0.0, 6.0), yaw=180.0, pitch=0.0, fov=45.0)
        frame_count = max(1, int(frames))
        t0 = time.perf_counter()
        for i in range(frame_count):
            bridge.render_frame()
            if (i + 1) % 4 == 0:
                print(f"[Step 2] Frame {i + 1}/{frame_count}")
        wp.synchronize_device("cuda")
        dt = max(time.perf_counter() - t0, 1.0e-9)
        fps = frame_count / dt
        ms_per_frame = 1000.0 / fps
        output = bridge.get_frame()
        output_shape = tuple(output.shape)
        output_min = float(output.min())
        output_max = float(output.max())
        print(
            f"[Step 2] OK (external scene) shape={output_shape} range=[{output_min:.4f}, {output_max:.4f}] "
            f"fps={fps:.2f} ms/frame={ms_per_frame:.2f}"
        )
        return output_shape, output_min, output_max

    viewer = create_a_beautiful_game_viewer(width=width, height=height)
    if use_procedural_sky:
        # Force procedural-sky path for C# parity testing.
        viewer._env_map = None
        viewer.sky_rgb_unit_conversion = (1.0 / 80000.0, 1.0 / 80000.0, 1.0 / 80000.0)
        viewer.sky_multiplier = 1.0
        viewer.sky_haze = 0.0
        viewer.sky_redblueshift = 0.0
        viewer.sky_saturation = 1.0
        viewer.sky_horizon_height = 0.0
        viewer.sky_ground_color = (0.4, 0.4, 0.4)
        viewer.sky_horizon_blur = 1.0
        viewer.sky_night_color = (0.0, 0.0, 0.0)
        viewer.sky_sun_disk_intensity = 1.0
        viewer.sky_sun_direction = (0.0, 1.0, 0.5)
        viewer.sky_sun_disk_scale = 1.0
        viewer.sky_sun_glow_intensity = 1.0
        viewer.sky_y_is_up = 1
    if not viewer.build():
        raise RuntimeError("A Beautiful Game viewer build failed")
    frame_count = max(1, int(frames))
    t0 = time.perf_counter()
    for i in range(frame_count):
        viewer.render()
        if (i + 1) % 4 == 0:
            print(f"[Step 2] Frame {i + 1}/{frame_count}")
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
    parser.add_argument("--step", choices=["1", "2", "3", "all"], default="3")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames-step1", type=int, default=4)
    parser.add_argument("--frames-step2", type=int, default=16)
    parser.add_argument("--frames-step3", type=int, default=0, help="0 means run until window close")
    parser.add_argument(
        "--use-procedural-sky",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use procedural sky instead of HDR environment map (default: enabled).",
    )
    parser.add_argument("--scene-gltf", type=str, default=None, help="Optional external glTF/GLB scene path for step 2.")
    parser.add_argument("--scene-obj", type=str, default=None, help="Optional external OBJ scene path for step 2 fallback.")
    args = parser.parse_args()

    wp.init()
    print("=" * 72)
    print("OptiX Pathtracing Build Steps")
    print("=" * 72)
    if args.step == "3":
        print("[Fast start] Running live viewer directly (skip steps 1/2).")

    if args.step in ("1", "all"):
        _run_step1_mini_renderer(args.width, args.height, args.frames_step1)
    if args.step in ("2", "all"):
        _run_step2_pathtracing(
            args.width,
            args.height,
            args.frames_step2,
            scene_gltf=args.scene_gltf,
            scene_obj=args.scene_obj,
            use_procedural_sky=bool(args.use_procedural_sky),
        )
    if args.step in ("3", "all"):
        print("[Step 3] A Beautiful Game live viewer")
        _run_step3_live_gui(
            args.width,
            args.height,
            args.fps,
            args.frames_step3,
            use_procedural_sky=bool(args.use_procedural_sky),
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
