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

"""OptiX-backed Newton viewer implementation."""

from __future__ import annotations

import ctypes
import time
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core.types import override
from .optix.pathtracing import PathTracingBridge
from .viewer import ViewerBase

if TYPE_CHECKING:
    import newton


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


class ViewerOptix(ViewerBase):
    """Newton viewer backend that renders through the OptiX path tracer."""

    def __init__(self, width: int = 1280, height: int = 720, headless: bool = False, enable_dlss_rr: bool = True):
        super().__init__()
        self._width = int(width)
        self._height = int(height)
        self._headless = bool(headless)
        self._bridge = PathTracingBridge(width=self._width, height=self._height, enable_dlss_rr=bool(enable_dlss_rr))
        # Use a visible procedural-sky baseline for generic Newton scenes.
        # PathTracingViewer defaults are tuned for HDR workflows and can appear too dark.
        self._bridge.set_use_procedural_sky(True)
        self._bridge.set_sky_parameters(
            sun_direction=(0.2, 0.9, 0.25),
            multiplier=1.0,
            haze=0.05,
            red_blue_shift=0.0,
            saturation=1.0,
            horizon_height=0.0,
            ground_color=(0.20, 0.20, 0.30),
            horizon_blur=1.0,
            night_color=(0.50, 0.70, 1.00),
            sun_disk_intensity=0.0,
            sun_disk_scale=1.0,
            sun_glow_intensity=0.0,
            y_is_up=1,
        )
        self._bridge.viewer.sky_rgb_unit_conversion = (1.0, 1.0, 1.0)
        self._running = True
        self._paused = False
        self._scene_dirty = False
        self._instance_transforms_dirty = False

        # Name/path registries to preserve stable IDs across frames.
        self._mesh_name_to_id: dict[str, int] = {}
        self._instance_ids_by_name: dict[str, list[int]] = {}
        self._instance_mesh_by_name: dict[str, str] = {}

        # Live presentation state (created lazily on first frame).
        self._window = None
        self._pyglet = None
        self._gl = None
        self._key = None
        self._keys = None
        self._pressed_keys: set[int] = set()
        self._texture = None
        self._sprite = None
        self._pbo = None
        self._cuda_gl = None
        self._display_u32 = None
        self._fps_last_t = time.perf_counter()
        self._fps_last_frames = 0
        self._presented_frames = 0

    def _ensure_window(self):
        if self._headless or self._window is not None:
            return
        try:
            import pyglet
            from pyglet import gl
            from pyglet.window import key
        except Exception:
            print("[ViewerOptix] pyglet is not available, running headless.")
            self._headless = True
            return

        self._pyglet = pyglet
        self._gl = gl
        self._key = key
        self._window = pyglet.window.Window(
            width=self._width,
            height=self._height,
            caption="Newton Viewer (OptiX)",
            vsync=False,
        )
        self._texture = pyglet.image.Texture.create(width=self._width, height=self._height, rectangle=False)
        self._texture.min_filter = gl.GL_NEAREST
        self._texture.mag_filter = gl.GL_NEAREST
        self._sprite = pyglet.sprite.Sprite(self._texture, x=0, y=0)

        pbo = gl.GLuint()
        gl.glGenBuffers(1, pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, self._width * self._height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self._pbo = pbo
        self._cuda_gl = wp.RegisteredGLBuffer(
            int(pbo.value),
            device="cuda",
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=False,
        )
        self._display_u32 = wp.empty(self._width * self._height, dtype=wp.uint32, device="cuda")

        keys = key.KeyStateHandler()
        self._window.push_handlers(keys)
        self._keys = keys

        @self._window.event
        def on_close():
            self._running = False
            self._window.close()

        @self._window.event
        def on_key_press(symbol, _modifiers):
            self._pressed_keys.add(symbol)
            if symbol == key.SPACE:
                self._paused = not self._paused

        @self._window.event
        def on_key_release(symbol, _modifiers):
            self._pressed_keys.discard(symbol)

    def _is_key_down(self, symbol: int) -> bool:
        return symbol in self._pressed_keys or (self._keys is not None and bool(self._keys[symbol]))

    def _update_camera_from_input(self):
        if self._window is None or self._key is None:
            return
        move = 0.06
        cam = self._bridge.viewer.camera
        key = self._key
        if self._is_key_down(key.W):
            cam.move_forward(move)
        if self._is_key_down(key.S):
            cam.move_forward(-move)
        if self._is_key_down(key.A):
            cam.move_right(-move)
        if self._is_key_down(key.D):
            cam.move_right(move)
        if self._is_key_down(key.Q):
            cam.move_up(-move)
        if self._is_key_down(key.E):
            cam.move_up(move)

    def _present_live_frame(self):
        if self._headless:
            return
        self._ensure_window()
        if self._window is None or self._gl is None:
            return

        self._window.switch_to()
        self._window.dispatch_events()
        if not self._running:
            return
        self._update_camera_from_input()

        wp.launch(
            _pack_display_rgba8,
            dim=(self._width, self._height),
            inputs=[self._bridge.viewer._tonemapper.output, self._display_u32, self._width, self._height],
            device="cuda",
        )
        mapped = self._cuda_gl.map(dtype=wp.uint32, shape=(self._width * self._height,))
        wp.copy(mapped, self._display_u32)
        self._cuda_gl.unmap()

        gl = self._gl
        self._window.clear()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self._pbo)
        gl.glBindTexture(self._texture.target, self._texture.id)
        gl.glTexSubImage2D(
            self._texture.target,
            0,
            0,
            0,
            self._width,
            self._height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(self._texture.target, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self._sprite.draw()
        self._window.flip()

        self._presented_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_t
        if elapsed >= 0.5:
            fps = (self._presented_frames - self._fps_last_frames) / elapsed if elapsed > 0.0 else 0.0
            self._window.set_caption(f"Newton Viewer (OptiX) - {fps:.1f} FPS")
            self._fps_last_t = now
            self._fps_last_frames = self._presented_frames

    @override
    def set_model(self, model: newton.Model, max_worlds: int | None = None):
        super().set_model(model, max_worlds=max_worlds)
        self._scene_dirty = True
        self._instance_transforms_dirty = False

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        self._bridge.set_camera_angles(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            yaw=float(yaw),
            pitch=float(pitch),
            fov=45.0,
        )

    @override
    def begin_frame(self, time):
        super().begin_frame(time)
        self._bridge.begin_frame(float(time))

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture: np.ndarray | str | None = None,
        hidden=False,
        backface_culling=True,
    ):
        del texture, backface_culling
        if name in self._mesh_name_to_id:
            return

        points_np = np.asarray(points.numpy(), dtype=np.float32).reshape(-1, 3)
        indices_np = np.asarray(indices.numpy(), dtype=np.int64).reshape(-1, 3).astype(np.uint32)
        normals_np = None if normals is None else np.asarray(normals.numpy(), dtype=np.float32).reshape(-1, 3)
        uvs_np = None if uvs is None else np.asarray(uvs.numpy(), dtype=np.float32).reshape(-1, 2)
        mesh_id = self._bridge.create_mesh(points_np, indices_np, normals=normals_np, uvs=uvs_np, material_id=0)
        self._mesh_name_to_id[str(name)] = int(mesh_id)
        if not hidden:
            self._scene_dirty = True

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        del colors, materials
        if hidden or xforms is None:
            return

        mesh_name = str(mesh)
        mesh_id = self._mesh_name_to_id.get(mesh_name)
        if mesh_id is None:
            return

        xforms_np = np.asarray(xforms.numpy(), dtype=np.float32).reshape(-1, 7)
        scales_np = None if scales is None else np.asarray(scales.numpy(), dtype=np.float32).reshape(-1, 3)
        desired_count = int(xforms_np.shape[0])
        key = str(name)

        current_ids = self._instance_ids_by_name.get(key)
        if current_ids is None or len(current_ids) != desired_count or self._instance_mesh_by_name.get(key) != mesh_name:
            current_ids = []
            for i in range(desired_count):
                p = xforms_np[i, 0:3]
                q = xforms_np[i, 3:7]
                s = float(scales_np[i, 0]) if scales_np is not None and i < len(scales_np) else 1.0
                instance_id = self._bridge.create_instance_with_transform(mesh_id, p, q, s)
                current_ids.append(int(instance_id))
            self._instance_ids_by_name[key] = current_ids
            self._instance_mesh_by_name[key] = mesh_name
            self._scene_dirty = True
            self._instance_transforms_dirty = False
            return

        for i, instance_id in enumerate(current_ids):
            p = xforms_np[i, 0:3]
            q = xforms_np[i, 3:7]
            s = float(scales_np[i, 0]) if scales_np is not None and i < len(scales_np) else 1.0
            self._bridge.set_instance_transform(instance_id, p, q, s)
        self._instance_transforms_dirty = True

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        del name, starts, ends, colors, width, hidden

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        del name, points, radii, colors, hidden

    @override
    def log_array(self, name, array):
        del name, array

    @override
    def log_scalar(self, name, value):
        del name, value

    @override
    def apply_forces(self, state):
        del state

    @override
    def end_frame(self):
        if self._scene_dirty:
            self._bridge.build_scene()
            self._scene_dirty = False
            self._instance_transforms_dirty = False
        elif self._instance_transforms_dirty:
            self._bridge.rebuild_tlas()
            self._instance_transforms_dirty = False
        self._bridge.render_frame()
        self._present_live_frame()

    def get_frame(self) -> wp.array:
        image = self._bridge.get_frame_uint8()
        rgb = image[..., :3].copy()
        return wp.array(rgb, dtype=wp.uint8, device="cuda")

    @override
    def is_running(self) -> bool:
        return self._running and self._bridge.is_running()

    @override
    def is_paused(self) -> bool:
        return self._paused

    @override
    def close(self):
        self._running = False
        if self._gl is not None and self._pbo is not None:
            self._gl.glDeleteBuffers(1, self._pbo)
            self._pbo = None
        if self._window is not None:
            self._window.close()
            self._window = None
        self._bridge.close()


__all__ = ["ViewerOptix"]
