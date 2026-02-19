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

"""Realtime viewer for the pathtracing emissive gallery sample."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import warp as wp

try:
    from newton._src.viewer.optix.pathtracing.emissive_gallery_sample import create_emissive_gallery_viewer
except ImportError:
    # Script mode fallback.
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    stale = [name for name in sys.modules if name == "newton" or name.startswith("newton.")]
    for name in stale:
        del sys.modules[name]
    from newton._src.viewer.optix.pathtracing.emissive_gallery_sample import create_emissive_gallery_viewer


@wp.kernel
def _pack_display_rgba8(
    src: wp.array2d(dtype=wp.vec4),
    dst: wp.array(dtype=wp.uint32),
    width: int,
    height: int,
):
    x, y = wp.tid()
    if x >= width or y >= height:
        return

    c = src[y, x]
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    a = wp.uint32(255)
    dst[y * width + x] = (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


def _run_headless(width: int, height: int, frames: int) -> None:
    viewer = create_emissive_gallery_viewer(width=width, height=height)
    if not viewer.build():
        raise RuntimeError("Failed to build pathtracing viewer")
    frame_count = max(1, int(frames))
    t0 = time.perf_counter()
    for _ in range(frame_count):
        viewer.render()
    wp.synchronize_device("cuda")
    dt = max(time.perf_counter() - t0, 1.0e-9)
    fps = frame_count / dt
    ms_per_frame = 1000.0 / fps
    out = viewer.get_output()
    print(
        f"Headless OK shape={out.shape} range=[{float(out.min()):.4f}, {float(out.max()):.4f}] "
        f"fps={fps:.2f} ms/frame={ms_per_frame:.2f}"
    )


def _run_gui(width: int, height: int, fps: int, frames: int) -> None:
    import pyglet  # noqa: PLC0415
    from pyglet import gl  # noqa: PLC0415
    from pyglet.window import key, mouse  # noqa: PLC0415

    viewer = create_emissive_gallery_viewer(width=width, height=height)
    if not viewer.build():
        raise RuntimeError("Failed to build pathtracing viewer")

    window = pyglet.window.Window(width=width, height=height, caption="Pathtracing Emissive Gallery", vsync=False)
    texture = pyglet.image.Texture.create(width=width, height=height, rectangle=False)
    texture.min_filter = gl.GL_NEAREST
    texture.mag_filter = gl.GL_NEAREST
    sprite = pyglet.sprite.Sprite(texture, x=0, y=0)

    pbo = gl.GLuint()
    gl.glGenBuffers(1, pbo)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
    gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_DYNAMIC_DRAW)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

    cuda_gl = wp.RegisteredGLBuffer(
        int(pbo.value),
        device="cuda",
        flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
        fallback_to_copy=False,
    )
    display_u32 = wp.empty(width * height, dtype=wp.uint32, device="cuda")
    keys = key.KeyStateHandler()
    window.push_handlers(keys)
    pressed_keys: set[int] = set()

    state = {"frames": 0, "fps_last_t": time.perf_counter(), "fps_last_frames": 0}

    @window.event
    def on_draw():
        window.clear()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        gl.glBindTexture(texture.target, texture.id)
        gl.glTexSubImage2D(
            texture.target,
            0,
            0,
            0,
            width,
            height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(texture.target, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        sprite.draw()

    @window.event
    def on_close():
        gl.glDeleteBuffers(1, pbo)
        window.close()

    @window.event
    def on_deactivate():
        return None

    @window.event
    def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
        has_delta = abs(float(dx)) > 0.001 or abs(float(dy)) > 0.001
        if not has_delta:
            return
        if buttons & mouse.LEFT:
            # Match C# exactly: Rotate(dx * 0.005f, -dy * 0.005f)
            viewer.camera.rotate(yaw=0.005 * float(dx), pitch=0.005 * float(dy))
        elif buttons & mouse.MIDDLE:
            viewer.camera.orbit(yaw=0.005 * float(dx), pitch=0.005 * float(dy))
        elif buttons & mouse.RIGHT:
            viewer.camera.pan(-0.01 * float(dx), 0.01 * float(dy))

    @window.event
    def on_mouse_scroll(_x, _y, _sx, sy):
        viewer.camera.zoom(delta=0.5 * float(sy))

    @window.event
    def on_key_press(symbol, _modifiers):
        pressed_keys.add(symbol)
        if symbol == key._1:
            viewer.output_mode = viewer.OUTPUT_FINAL
            print("[Live] output_mode=FINAL")
        elif symbol == key._2:
            viewer.output_mode = viewer.OUTPUT_DEPTH
            print("[Live] output_mode=DEPTH (DLSS input style)")
        elif symbol == key._3:
            viewer.output_mode = viewer.OUTPUT_NORMAL
            print("[Live] output_mode=NORMAL")
        elif symbol == key._4:
            viewer.output_mode = viewer.OUTPUT_ROUGHNESS
            print("[Live] output_mode=ROUGHNESS")
        elif symbol == key._5:
            viewer.output_mode = viewer.OUTPUT_DIFFUSE
            print("[Live] output_mode=DIFFUSE (material base color)")
        elif symbol == key._6:
            viewer.output_mode = viewer.OUTPUT_SPECULAR
            print("[Live] output_mode=SPECULAR (material metallic)")
        elif symbol == key._7:
            viewer.output_mode = viewer.OUTPUT_MOTION
            print("[Live] output_mode=MOTION (camera proxy)")
        elif symbol == key._8:
            viewer.output_mode = viewer.OUTPUT_SPEC_HITDIST
            print("[Live] output_mode=SPEC_HITDIST (first specular segment distance)")

    @window.event
    def on_key_release(symbol, _modifiers):
        if symbol in pressed_keys:
            pressed_keys.remove(symbol)

    def _is_down(symbol: int) -> bool:
        # Use both event-driven and KeyStateHandler checks for robustness across platforms.
        return symbol in pressed_keys or bool(keys[symbol])

    def update(_dt):
        move = 5.0 * float(_dt)
        if _is_down(key.W):
            viewer.camera.move_forward(move)
        if _is_down(key.S):
            viewer.camera.move_forward(-move)
        if _is_down(key.A):
            viewer.camera.move_right(-move)
        if _is_down(key.D):
            viewer.camera.move_right(move)
        if _is_down(key.Q):
            viewer.camera.move_up(-move)
        if _is_down(key.E):
            viewer.camera.move_up(move)

        viewer.render()
        wp.launch(
            _pack_display_rgba8,
            dim=(width, height),
            inputs=[viewer._tonemapper.output, display_u32, width, height],
            device="cuda",
        )
        mapped = cuda_gl.map(dtype=wp.uint32, shape=(width * height,))
        wp.copy(mapped, display_u32)
        cuda_gl.unmap()

        state["frames"] += 1
        now = time.perf_counter()
        elapsed = now - state["fps_last_t"]
        if elapsed >= 0.5:
            frames_delta = state["frames"] - state["fps_last_frames"]
            fps_val = frames_delta / elapsed if elapsed > 0.0 else 0.0
            window.set_caption(f"Pathtracing Emissive Gallery - {fps_val:.1f} FPS")
            state["fps_last_t"] = now
            state["fps_last_frames"] = state["frames"]
        if frames > 0 and state["frames"] >= frames:
            pyglet.app.exit()

    pyglet.clock.schedule_interval(update, 1.0 / float(max(1, fps)))
    pyglet.app.run()


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames", type=int, default=0, help="Auto-exit after N frames (0=run until close)")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    wp.init()
    if args.headless:
        _run_headless(args.width, args.height, args.frames if args.frames > 0 else 2)
    else:
        print(
            "Controls: LMB rotate, MMB orbit, RMB pan, wheel zoom, WASDQE move, 1=final, 2=depth, 3=normal, 4=roughness, 5=diffuse, 6=specular, 7=motion"
        )
        _run_gui(args.width, args.height, args.fps, args.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
