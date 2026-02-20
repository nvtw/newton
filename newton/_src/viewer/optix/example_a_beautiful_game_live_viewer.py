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

"""Realtime viewer for the A Beautiful Game path tracing sample."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import warp as wp

try:
    from newton._src.viewer.optix.pathtracing.a_beautiful_game_sample import create_a_beautiful_game_viewer
except ImportError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    stale = [name for name in sys.modules if name == "newton" or name.startswith("newton.")]
    for name in stale:
        del sys.modules[name]
    from newton._src.viewer.optix.pathtracing.a_beautiful_game_sample import create_a_beautiful_game_viewer


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
    viewer = create_a_beautiful_game_viewer(width=width, height=height)
    if not viewer.build():
        raise RuntimeError("Failed to build A Beautiful Game viewer")
    frame_count = max(1, int(frames))
    t0 = time.perf_counter()
    for _ in range(frame_count):
        viewer.render()
    wp.synchronize_device("cuda")
    dt = max(time.perf_counter() - t0, 1.0e-9)
    fps = frame_count / dt
    out = viewer.get_output()
    print(
        f"Headless OK shape={out.shape} range=[{float(out.min()):.4f}, {float(out.max()):.4f}] "
        f"fps={fps:.2f} ms/frame={1000.0 / fps:.2f}"
    )


def _run_gui(width: int, height: int, fps: int, frames: int, use_procedural_sky: bool = True) -> None:
    import pyglet
    from pyglet import gl
    from pyglet.window import key, mouse

    viewer = create_a_beautiful_game_viewer(width=width, height=height)
    if use_procedural_sky:
        # Force procedural sky path for parity testing against C# bridge behavior.
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
        raise RuntimeError("Failed to build A Beautiful Game viewer")
    if use_procedural_sky and viewer._env_map is not None:
        raise RuntimeError("Procedural sky requested, but an environment map is still active.")

    window = pyglet.window.Window(width=width, height=height, caption="A Beautiful Game Pathtracing", vsync=False)
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
        if abs(float(dx)) <= 0.001 and abs(float(dy)) <= 0.001:
            return
        if buttons & mouse.LEFT:
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
        elif symbol == key._2:
            viewer.output_mode = viewer.OUTPUT_DEPTH
        elif symbol == key._3:
            viewer.output_mode = viewer.OUTPUT_NORMAL
        elif symbol == key._4:
            viewer.output_mode = viewer.OUTPUT_ROUGHNESS
        elif symbol == key._5:
            viewer.output_mode = viewer.OUTPUT_DIFFUSE
        elif symbol == key._6:
            viewer.output_mode = viewer.OUTPUT_SPECULAR
        elif symbol == key._7:
            viewer.output_mode = viewer.OUTPUT_MOTION
        elif symbol == key._8:
            viewer.output_mode = viewer.OUTPUT_SPEC_HITDIST

    @window.event
    def on_key_release(symbol, _modifiers):
        if symbol in pressed_keys:
            pressed_keys.remove(symbol)

    def _is_down(symbol: int) -> bool:
        return symbol in pressed_keys or bool(keys[symbol])

    def update(dt: float):
        move = 5.0 * float(dt)
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
            window.set_caption(f"A Beautiful Game Pathtracing - {fps_val:.1f} FPS")
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
        print("Controls: LMB rotate, MMB orbit, RMB pan, wheel zoom, WASDQE move, 1..7 output modes")
        _run_gui(args.width, args.height, args.fps, args.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
