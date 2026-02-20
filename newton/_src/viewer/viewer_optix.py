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

"""OptiX-backed Newton viewer implementation, mirroring hybrid viewer behavior."""

from __future__ import annotations

import ctypes
import math
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


def _quat_to_mat3(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _build_local_transform(position: np.ndarray, rotation_xyzw: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    px, py, pz = [float(v) for v in position]
    qx, qy, qz, qw = [float(v) for v in rotation_xyzw]
    if isinstance(scale, np.ndarray):
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    else:
        sx = sy = sz = float(scale)
    m = np.eye(4, dtype=np.float32)
    r = _quat_to_mat3(qx, qy, qz, qw)
    r[:, 0] *= sx
    r[:, 1] *= sy
    r[:, 2] *= sz
    m[:3, :3] = r
    m[:3, 3] = np.array([px, py, pz], dtype=np.float32)
    return m


class ViewerOptix(ViewerBase):
    """Newton OptiX viewer following hybrid-viewer camera/scene semantics."""

    def __init__(self, width: int = 1280, height: int = 720, headless: bool = False, enable_dlss_rr: bool = True):
        super().__init__()
        self._width = int(width)
        self._height = int(height)
        self._headless = bool(headless)
        self._bridge = PathTracingBridge(width=self._width, height=self._height, enable_dlss_rr=bool(enable_dlss_rr))

        self._running = True
        self._paused = False
        self._scene_dirty = False
        self._instance_transforms_dirty = False

        self._up_axis = 1  # 0=X, 1=Y, 2=Z
        self._global_transform = np.eye(4, dtype=np.float32)
        self._user_camera_control = False
        self._camera_pos = np.array([0.0, 2.0, 8.0], dtype=np.float32)
        self._camera_yaw = 180.0
        self._camera_pitch = 0.0
        self._camera_fov = 45.0
        self._cam_speed = 4.0
        self._last_time = time.perf_counter()

        self._mesh_ids: dict[int, int] = {}
        self._mesh_name_to_hash: dict[str, int] = {}
        self._mesh_geometry: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = {}
        self._mesh_color_variants: dict[tuple[int, tuple[float, float, float]], int] = {}
        self._instance_ids: dict[str, list[int]] = {}
        self._instance_mesh_hash: dict[str, int] = {}
        self._color_to_material: dict[tuple[float, float, float], int] = {}
        self._default_material_id = -1

        self._window = None
        self._pyglet = None
        self._gl = None
        self._key = None
        self._mouse = None
        self._keys = None
        self._keys_down: set[int] = set()
        self._pbo = None
        self._texture = None
        self._sprite = None
        self._cuda_gl = None
        self._display_u32 = None
        self._frame_count = 0
        self._fps_last_t = time.perf_counter()
        self._fps_last_frame = 0

        self._configure_default_sky()

    @staticmethod
    def _srgb_to_linear(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    def _get_or_create_material(self, color: tuple[float, float, float]) -> int:
        color_key = (round(color[0], 1), round(color[1], 1), round(color[2], 1))
        if color_key in self._color_to_material:
            return self._color_to_material[color_key]
        linear = (
            self._srgb_to_linear(color_key[0]),
            self._srgb_to_linear(color_key[1]),
            self._srgb_to_linear(color_key[2]),
        )
        mat_id = self._bridge.create_diffuse_material(linear)
        self._color_to_material[color_key] = mat_id
        return mat_id

    def _configure_default_sky(self):
        self._bridge.set_use_procedural_sky(True)
        self._bridge.set_sky_parameters(
            sun_direction=(-0.3, 0.7, 0.5),
            multiplier=1.5,
            haze=0.03,
            red_blue_shift=0.0,
            saturation=1.0,
            horizon_height=0.0,
            ground_color=(0.7, 0.7, 0.75),
            horizon_blur=0.3,
            night_color=(0.0, 0.0, 0.0),
            sun_disk_intensity=1.0,
            sun_disk_scale=1.0,
            sun_glow_intensity=0.8,
            y_is_up=1,
        )
        self._bridge.viewer._env_map = None

    def _setup_global_transform(self, up_axis: int):
        if up_axis == 2:
            angle = np.pi / 2.0
            c, s = np.cos(angle), np.sin(angle)
            self._global_transform = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, c, -s, 0.0],
                    [0.0, s, c, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        elif up_axis == 0:
            angle = -np.pi / 2.0
            c, s = np.cos(angle), np.sin(angle)
            self._global_transform = np.array(
                [
                    [c, -s, 0.0, 0.0],
                    [s, c, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        else:
            self._global_transform = np.eye(4, dtype=np.float32)

    def _apply_global_transform(self, local: np.ndarray) -> np.ndarray:
        # This renderer path uses column-vector math for instance transforms:
        # p_world = M * p_local. To convert Newton world (up_axis-dependent)
        # into viewer Y-up world, apply global first in world space:
        # p_render = G * (L * p_local) => M_final = G @ L.
        return self._global_transform @ np.asarray(local, dtype=np.float32)

    def _apply_camera_to_bridge(self):
        self._bridge.set_camera_angles(
            position=(float(self._camera_pos[0]), float(self._camera_pos[1]), float(self._camera_pos[2])),
            yaw=float(self._camera_yaw),
            pitch=float(self._camera_pitch),
            fov=float(self._camera_fov),
        )

    def _ensure_window(self):
        if self._headless or self._window is not None:
            return
        try:
            import pyglet
            from pyglet import gl
            from pyglet.window import key, mouse
        except Exception:
            self._headless = True
            print("[ViewerOptix] pyglet unavailable, running headless.")
            return

        self._pyglet = pyglet
        self._gl = gl
        self._key = key
        self._mouse = mouse
        self._window = pyglet.window.Window(
            width=self._width,
            height=self._height,
            caption="Newton Viewer (OptiX)",
            vsync=False,
            resizable=True,
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
            self._keys_down.add(symbol)
            if symbol == key.SPACE:
                self._paused = not self._paused
            elif symbol == key.ESCAPE:
                self._running = False
                self._window.close()

        @self._window.event
        def on_key_release(symbol, _modifiers):
            self._keys_down.discard(symbol)

        @self._window.event
        def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
            if buttons & mouse.LEFT:
                self._camera_yaw -= float(dx) * 0.1
                self._camera_pitch += float(dy) * 0.1
                self._camera_pitch = max(-89.0, min(89.0, self._camera_pitch))
                self._user_camera_control = True

        @self._window.event
        def on_mouse_scroll(_x, _y, _sx, sy):
            self._camera_fov = max(10.0, min(120.0, self._camera_fov - float(sy) * 2.0))
            self._user_camera_control = True

    def _is_down(self, symbol: int) -> bool:
        return symbol in self._keys_down or (self._keys is not None and bool(self._keys[symbol]))

    def _update_camera_from_input(self, dt: float):
        if self._key is None:
            return
        yaw_rad = math.radians(self._camera_yaw)
        pitch_rad = math.radians(self._camera_pitch)
        forward = np.array(
            [
                math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                math.cos(yaw_rad) * math.cos(pitch_rad),
            ],
            dtype=np.float32,
        )
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        rn = float(np.linalg.norm(right))
        if rn > 1.0e-6:
            right /= rn

        direction = np.zeros(3, dtype=np.float32)
        has_input = False
        key = self._key
        if self._is_down(key.W) or self._is_down(key.UP):
            direction += forward
            has_input = True
        if self._is_down(key.S) or self._is_down(key.DOWN):
            direction -= forward
            has_input = True
        if self._is_down(key.A) or self._is_down(key.LEFT):
            direction -= right
            has_input = True
        if self._is_down(key.D) or self._is_down(key.RIGHT):
            direction += right
            has_input = True
        if self._is_down(key.Q):
            direction -= world_up
            has_input = True
        if self._is_down(key.E):
            direction += world_up
            has_input = True

        if has_input:
            dn = float(np.linalg.norm(direction))
            if dn > 1.0e-6:
                self._camera_pos += (direction / dn) * self._cam_speed * float(dt)
            self._user_camera_control = True

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

        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera_from_input(dt)

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

        self._frame_count += 1
        elapsed = now - self._fps_last_t
        if elapsed >= 1.0:
            frames_delta = self._frame_count - self._fps_last_frame
            fps = frames_delta / elapsed if elapsed > 0.0 else 0.0
            self._window.set_caption(f"Newton Viewer (OptiX) - {fps:.1f} FPS")
            self._fps_last_t = now
            self._fps_last_frame = self._frame_count

    @override
    def set_model(self, model: newton.Model, max_worlds: int | None = None):
        super().set_model(model, max_worlds=max_worlds)
        self._up_axis = int(getattr(model, "up_axis", 1))
        self._setup_global_transform(self._up_axis)
        self._configure_default_sky()

        # Match hybrid initial camera convention.
        self._camera_pos = np.array([0.0, 2.0, 8.0], dtype=np.float32)
        self._camera_yaw = 180.0
        self._camera_pitch = 0.0
        self._camera_fov = 45.0
        self._user_camera_control = False

        if self._default_material_id < 0:
            self._default_material_id = self._get_or_create_material((0.8, 0.8, 0.8))

        self._scene_dirty = True
        self._instance_transforms_dirty = False

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        if self._user_camera_control:
            return
        self._camera_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        self._camera_pitch = float(pitch)
        self._camera_yaw = float(yaw)

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
        mesh_name = str(name)
        if mesh_name in self._mesh_name_to_hash:
            return

        points_np = np.asarray(points.numpy(), dtype=np.float32).reshape(-1, 3)
        indices_np = np.asarray(indices.numpy(), dtype=np.uint32).reshape(-1, 3)
        normals_np = None if normals is None else np.asarray(normals.numpy(), dtype=np.float32).reshape(-1, 3)
        uvs_np = None if uvs is None else np.asarray(uvs.numpy(), dtype=np.float32).reshape(-1, 2)

        mesh_hash = hash((mesh_name, len(points_np), len(indices_np)))
        self._mesh_name_to_hash[mesh_name] = mesh_hash
        self._mesh_geometry[mesh_hash] = (points_np, indices_np, normals_np, uvs_np)
        mesh_id = self._bridge.create_mesh(points_np, indices_np, normals=normals_np, uvs=uvs_np, material_id=self._default_material_id)
        self._mesh_ids[mesh_hash] = int(mesh_id)
        self._scene_dirty = True

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        del materials
        if hidden or xforms is None:
            return
        mesh_hash = self._mesh_name_to_hash.get(str(mesh))
        if mesh_hash is None or mesh_hash not in self._mesh_ids:
            return

        xforms_np = np.asarray(xforms.numpy(), dtype=np.float32).reshape(-1, 7)
        scales_np = None if scales is None else np.asarray(scales.numpy(), dtype=np.float32).reshape(-1, 3)
        colors_np = None if colors is None else np.asarray(colors.numpy(), dtype=np.float32).reshape(-1, 3)
        num_instances = int(xforms_np.shape[0])
        if num_instances == 0:
            return

        color_groups: dict[tuple[float, float, float], list[int]] = {}
        for i in range(num_instances):
            if colors_np is not None and i < len(colors_np):
                c = colors_np[i]
                ck = (round(float(c[0]), 1), round(float(c[1]), 1), round(float(c[2]), 1))
            else:
                ck = (0.8, 0.8, 0.8)
            color_groups.setdefault(ck, []).append(i)

        instance_key = f"{name}_{num_instances}"
        if instance_key not in self._instance_ids or self._instance_mesh_hash.get(instance_key) != mesh_hash:
            instance_ids: list[int] = [0] * num_instances
            for color_key, indices_list in color_groups.items():
                variant_key = (mesh_hash, color_key)
                if variant_key not in self._mesh_color_variants:
                    mat_id = self._get_or_create_material(color_key)
                    pts, inds, nrms, tex = self._mesh_geometry[mesh_hash]
                    self._mesh_color_variants[variant_key] = int(
                        self._bridge.create_mesh(pts, inds, normals=nrms, uvs=tex, material_id=mat_id)
                    )
                variant_mesh_id = self._mesh_color_variants[variant_key]
                for i in indices_list:
                    p = xforms_np[i, 0:3]
                    q = xforms_np[i, 3:7]
                    s = scales_np[i, 0:3] if scales_np is not None and i < len(scales_np) else 1.0
                    local_m = _build_local_transform(p, q, s if isinstance(s, np.ndarray) else float(s))
                    final_m = self._apply_global_transform(local_m)
                    inst_id = int(self._bridge.create_instance(int(variant_mesh_id)))
                    self._bridge.set_instance_transform_matrix(inst_id, final_m)
                    instance_ids[i] = inst_id
            self._instance_ids[instance_key] = instance_ids
            self._instance_mesh_hash[instance_key] = mesh_hash
            self._scene_dirty = True
            self._instance_transforms_dirty = False
            return

        instance_ids = self._instance_ids[instance_key]
        count = min(num_instances, len(instance_ids))
        for i in range(count):
            p = xforms_np[i, 0:3]
            q = xforms_np[i, 3:7]
            s = scales_np[i, 0:3] if scales_np is not None and i < len(scales_np) else 1.0
            local_m = _build_local_transform(p, q, s if isinstance(s, np.ndarray) else float(s))
            final_m = self._apply_global_transform(local_m)
            self._bridge.set_instance_transform_matrix(int(instance_ids[i]), final_m)
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

        self._apply_camera_to_bridge()
        self._bridge.render_frame()
        self._present_live_frame()

    def get_frame(self) -> wp.array:
        image = self._bridge.get_frame_uint8()
        rgb = image[..., :3].copy()
        return wp.array(rgb, dtype=wp.uint8, device="cuda")

    @override
    def is_running(self) -> bool:
        win_running = self._window is None or not bool(getattr(self._window, "has_exit", False))
        return self._running and win_running and self._bridge.is_running()

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
