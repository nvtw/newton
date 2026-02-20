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

"""OptiX path-tracing viewer for Newton physics simulations.

Drop-in replacement for :class:`ViewerGL` that renders scenes with hardware-
accelerated ray tracing via NVIDIA OptiX. The viewer uses the same
:class:`ViewerBase` interface so existing simulation loops work unchanged::

    # swap ViewerGL for ViewerOptix - everything else stays the same
    viewer = ViewerOptix(width=1920, height=1080)
    viewer.set_model(model)
    while viewer.is_running():
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()
    viewer.close()
"""

from __future__ import annotations

import ctypes
import math
import time
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import newton as nt

from ..core.types import override
from .viewer import ViewerBase

if TYPE_CHECKING:
    from .optix.pathtracing.pathtracer_api import PathTracerAPI
    from .optix.pathtracing.scene import Scene as OptixScene


@wp.kernel
def _pack_display_rgba8(
    src: wp.array2d(dtype=wp.vec4),
    dst: wp.array(dtype=wp.uint32),
    width: int,
    height: int,
):
    """Pack HDR vec4 tonemapped output into RGBA8 uint32 for GL texture upload."""
    x, y = wp.tid()
    if x >= width or y >= height:
        return
    c = src[y, x]
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    a = wp.uint32(255)
    dst[y * width + x] = (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


@wp.kernel
def _extract_rgb_from_tonemapped(
    src: wp.array2d(dtype=wp.vec4),
    dst: wp.array(dtype=wp.uint8, ndim=3),
    width: int,
    height: int,
):
    """Extract RGB uint8 from tonemapped vec4 output into (H, W, 3) array."""
    x, y = wp.tid()
    if x >= width or y >= height:
        return
    c = src[y, x]
    dst[y, x, 0] = wp.uint8(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    dst[y, x, 1] = wp.uint8(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    dst[y, x, 2] = wp.uint8(wp.clamp(c[2] * 255.0, 0.0, 255.0))


@wp.func
def _rotate_zup_to_yup(v: wp.vec3) -> wp.vec3:
    # Rotate -90 deg around X: (x, y, z) -> (x, z, -y)
    return wp.vec3(v[0], v[2], -v[1])


@wp.func
def _rotate_xup_to_yup(v: wp.vec3) -> wp.vec3:
    # Rotate -90 deg around Z: (x, y, z) -> (y, -x, z)
    return wp.vec3(v[1], -v[0], v[2])


@wp.kernel
def _pack_instance_mats_from_xforms(
    xforms: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    up_mode: int,  # 0: Y-up, 1: Z-up -> Y-up, 2: X-up -> Y-up
    out_mats: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    t = xforms[tid]
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)
    r = wp.quat_to_matrix(q)
    s = scales[tid]

    c0 = wp.vec3(r[0, 0] * s[0], r[1, 0] * s[0], r[2, 0] * s[0])
    c1 = wp.vec3(r[0, 1] * s[1], r[1, 1] * s[1], r[2, 1] * s[1])
    c2 = wp.vec3(r[0, 2] * s[2], r[1, 2] * s[2], r[2, 2] * s[2])

    if up_mode == 1:
        p = _rotate_zup_to_yup(p)
        c0 = _rotate_zup_to_yup(c0)
        c1 = _rotate_zup_to_yup(c1)
        c2 = _rotate_zup_to_yup(c2)
    elif up_mode == 2:
        p = _rotate_xup_to_yup(p)
        c0 = _rotate_xup_to_yup(c0)
        c1 = _rotate_xup_to_yup(c1)
        c2 = _rotate_xup_to_yup(c2)

    # Match _build_instance_matrix() convention:
    # - columns c0/c1/c2 are scaled rotation basis vectors
    # - translation is in the last column
    out_mats[tid] = wp.mat44(
        c0[0], c1[0], c2[0], p[0],
        c0[1], c1[1], c2[1], p[1],
        c0[2], c1[2], c2[2], p[2],
        0.0, 0.0, 0.0, 1.0,
    )


class ViewerOptix(ViewerBase):
    """OptiX path-tracing interactive viewer for Newton physics models.

    Provides the same public API as :class:`ViewerGL` so it can be used as a
    drop-in replacement in simulation loops.  Rendering is performed by the
    OptiX path tracer (via :class:`PathTracerAPI`) and displayed in a
    Pyglet window with CUDA-GL interop.

    Key features:
        - Hardware-accelerated path tracing with PBR materials.
        - Procedural sky / HDR environment lighting.
        - DLSS Ray Reconstruction denoising (when available).
        - Automatic coordinate system conversion (Z-up / X-up -> Y-up).
        - sRGB-to-linear color conversion for correct PBR shading.
        - WASD/QE + mouse camera controls.
        - 1-8 debug output modes (final, depth, normal, roughness, ...).
        - Headless mode for offscreen rendering.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        headless: bool = False,
        enable_dlss_rr: bool = True,
    ):
        """Initialize the OptiX viewer.

        Args:
            width: Render / window width in pixels.
            height: Render / window height in pixels.
            vsync: Enable vertical sync (Pyglet window only).
            headless: If ``True``, skip window creation (offscreen rendering).
            enable_dlss_rr: Enable DLSS Ray Reconstruction denoising.
        """
        super().__init__()

        self._width = int(width)
        self._height = int(height)
        self._vsync = bool(vsync)
        self._headless = bool(headless)
        self._enable_dlss_rr = bool(enable_dlss_rr)

        self._api: PathTracerAPI | None = None
        self._scene: OptixScene | None = None
        self._built = False
        self._running = True
        self._paused = False

        # Mapping from geometry mesh name -> OptiX mesh index
        self._mesh_name_to_optix_id: dict[str, int] = {}
        # Mapping from instance name -> list of OptiX instance indices
        self._instance_name_to_optix_ids: dict[str, list[int]] = {}
        # Mapping from instance name -> whether it is currently hidden
        self._instance_hidden: dict[str, bool] = {}
        # Material cache: (r, g, b, roughness, metallic) -> material id
        self._material_cache: dict[tuple[float, ...], int] = {}
        # Per-instance material assignments: optix_instance_id -> material_id
        self._instance_material_map: dict[int, int] = {}
        # Number of meshes at last scene build (to detect new geometry)
        self._meshes_at_last_build = 0
        # Whether per-instance material IDs need uploading after scene build
        self._materials_dirty = False

        # Global transform for coordinate system conversion (identity = Y-up native)
        self._global_transform: np.ndarray | None = None

        # Pyglet window resources (created lazily in _ensure_window)
        self._window = None
        self._gl_texture = None
        self._sprite = None
        self._pbo = None
        self._cuda_gl = None

        # Camera state (pitch/yaw in degrees)
        # Convention matches hybrid_viewer / C#: yaw=0 looks in +Z
        self._cam_pos = np.array([0.0, 2.0, 8.0], dtype=np.float32)
        self._cam_pitch = 0.0
        self._cam_yaw = 180.0  # Look toward -Z (toward origin)
        self._cam_fov = 45.0
        self._cam_speed = 4.0  # m/s
        self._cam_user_set = False

        # Previous camera state for dirty detection (None = never synced)
        self._prev_cam_pos: np.ndarray | None = None
        self._prev_cam_pitch: float | None = None
        self._prev_cam_yaw: float | None = None
        self._prev_cam_fov: float | None = None

        # Input state
        self._keys_down: set[int] = set()
        self._mouse_buttons: int = 0
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Timing
        self._last_time = time.perf_counter()
        self._fps_last_t = time.perf_counter()
        self._fps_last_frames = 0
        self._frame_count = 0
        self._current_fps = 0.0

        # Track whether scene needs TLAS rebuild
        self._tlas_dirty = False
        # Reusable transform staging buffers for device-path instance updates
        self._instance_mat_capacity: dict[str, int] = {}
        self._instance_mats_device: dict[str, wp.array] = {}
        self._instance_mats_host: dict[str, wp.array] = {}
        # Cached default-scale buffer to avoid per-call allocation
        self._default_scales_dev: wp.array | None = None
        self._default_scales_count = 0

    # ------------------------------------------------------------------
    # Path tracer lifecycle
    # ------------------------------------------------------------------

    def _ensure_api(self):
        """Lazily create the PathTracerAPI and initialize OptiX."""
        if self._api is not None:
            return

        from .optix.pathtracing.pathtracer_api import PathTracerAPI  # noqa: PLC0415

        self._api = PathTracerAPI(
            width=self._width,
            height=self._height,
            enable_dlss_rr=self._enable_dlss_rr,
        )
        ok = self._api.initialize()
        if not ok:
            raise RuntimeError("Failed to initialize OptiX path tracer. Check GPU drivers and OptiX SDK installation.")
        self._scene = self._api.scene

        # Configure procedural sky matching PythonBridge.cs PhysicalSkyParameters.Default
        viewer = self._api.viewer
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

    @staticmethod
    def _create_texture_2d_compat(pyglet_module, gl_module, width: int, height: int):
        """Create a 2D pyglet texture across pyglet API versions.

        Newer pyglet versions (2.1.x) expose ``target`` and do not accept the
        legacy ``rectangle`` argument, while older versions use ``rectangle``.
        """
        try:
            return pyglet_module.image.Texture.create(width=width, height=height, rectangle=False)
        except TypeError:
            return pyglet_module.image.Texture.create(
                width=width,
                height=height,
                target=gl_module.GL_TEXTURE_2D,
            )

    def _ensure_window(self):
        """Lazily create the Pyglet display window and GL resources."""
        if self._headless or self._window is not None:
            return

        import pyglet
        from pyglet import gl

        self._window = pyglet.window.Window(
            width=self._width,
            height=self._height,
            caption="Newton Viewer (OptiX)",
            vsync=self._vsync,
            resizable=True,
        )
        self._window.set_minimum_size(128, 128)

        self._gl_texture = self._create_texture_2d_compat(pyglet, gl, self._width, self._height)
        self._gl_texture.min_filter = gl.GL_NEAREST
        self._gl_texture.mag_filter = gl.GL_NEAREST
        self._sprite = pyglet.sprite.Sprite(self._gl_texture, x=0, y=0)

        pbo = gl.GLuint()
        gl.glGenBuffers(1, pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        gl.glBufferData(
            gl.GL_PIXEL_UNPACK_BUFFER,
            self._width * self._height * 4,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self._pbo = pbo

        self._cuda_gl = wp.RegisteredGLBuffer(
            int(pbo.value),
            device="cuda",
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=False,
        )

        self._window.push_handlers(self)

    def _recreate_gl_resources(self):
        """Recreate GL texture, PBO, and CUDA interop for the current resolution."""
        if self._window is None:
            return

        import pyglet
        from pyglet import gl

        w, h = self._width, self._height

        # Release CUDA-GL interop before deleting the GL buffer
        self._cuda_gl = None
        if self._pbo is not None:
            gl.glDeleteBuffers(1, self._pbo)
            self._pbo = None

        # Recreate texture + sprite
        self._gl_texture = self._create_texture_2d_compat(pyglet, gl, w, h)
        self._gl_texture.min_filter = gl.GL_NEAREST
        self._gl_texture.mag_filter = gl.GL_NEAREST
        self._sprite = pyglet.sprite.Sprite(self._gl_texture, x=0, y=0)

        # Recreate PBO
        pbo = gl.GLuint()
        gl.glGenBuffers(1, pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, w * h * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self._pbo = pbo

        self._cuda_gl = wp.RegisteredGLBuffer(
            int(pbo.value),
            device="cuda",
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=False,
        )

        # Update GL viewport
        gl.glViewport(0, 0, w, h)

    # ------------------------------------------------------------------
    # ViewerBase abstract method implementations
    # ------------------------------------------------------------------

    @override
    def set_model(self, model: nt.Model | None, max_worlds: int | None = None):
        super().set_model(model, max_worlds=max_worlds)

        if model is not None:
            self._setup_global_transform(model.up_axis)

            if not self._cam_user_set:
                self._cam_pos = np.array([0.0, 2.0, 8.0], dtype=np.float32)
                self._cam_yaw = 180.0
                self._cam_pitch = 0.0
            self._prev_cam_pos = self._cam_pos.copy()
            self._prev_cam_pitch = self._cam_pitch
            self._prev_cam_yaw = self._cam_yaw
            self._prev_cam_fov = self._cam_fov

    def _setup_global_transform(self, up_axis: int):
        """Set up global transform based on model's up_axis.

        Matches hybrid_viewer.py: converts physics coordinates to Y-up
        rendering space by applying a rotation to all instance transforms.

        Args:
            up_axis: 0 for X-up, 1 for Y-up, 2 for Z-up.
        """
        if up_axis == 2:  # Z-up -> Y-up: rotate -90 degrees around X
            angle = -np.pi / 2
            c, s = float(np.cos(angle)), float(np.sin(angle))
            self._global_transform = np.array([
                [1, 0, 0, 0],
                [0, c, -s, 0],
                [0, s, c, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        elif up_axis == 0:  # X-up -> Y-up: rotate -90 degrees around Z
            angle = -np.pi / 2
            c, s = float(np.cos(angle)), float(np.sin(angle))
            self._global_transform = np.array([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        else:  # Y-up (native) - no transform needed
            self._global_transform = None

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        self._cam_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        self._cam_pitch = float(pitch)
        self._cam_yaw = float(yaw)
        self._cam_user_set = True

    @override
    def begin_frame(self, time_val):
        super().begin_frame(time_val)

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture=None,
        hidden=False,
        backface_culling=True,
    ):
        self._ensure_api()

        if name in self._mesh_name_to_optix_id:
            return

        pts_np = points.numpy().astype(np.float32).reshape(-1, 3)
        idx_np = indices.numpy().astype(np.uint32).reshape(-1, 3)
        nrm_np = normals.numpy().astype(np.float32).reshape(-1, 3) if normals is not None else None
        uv_np = uvs.numpy().astype(np.float32).reshape(-1, 2) if uvs is not None else None

        if self._scene is not None and self._scene.materials.count == 0:
            self._scene.materials.add_diffuse((0.8, 0.8, 0.8))

        mesh_id = self._api.create_mesh(
            positions=pts_np,
            indices=idx_np,
            normals=nrm_np,
            uvs=uv_np,
            material_id=0,
        )
        self._mesh_name_to_optix_id[name] = mesh_id

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self._ensure_api()

        mesh_id = self._mesh_name_to_optix_id.get(mesh)
        if mesh_id is None:
            return

        transform_count = len(xforms) if xforms is not None else 0
        if transform_count == 0:
            self._instance_hidden[name] = True
            return

        self._instance_hidden[name] = hidden

        xforms_np = None
        scales_np = None
        use_device_path = False

        if (
            isinstance(xforms, wp.array)
            and xforms.device is not None
            and xforms.device.is_cuda
            and (
                scales is None
                or (isinstance(scales, wp.array) and scales.device is not None and scales.device.is_cuda)
            )
        ):
            use_device_path = True
            xforms_np, scales_np = self._pack_instance_mats_device_to_host(name, xforms, scales, transform_count)
        else:
            xforms_np = xforms.numpy() if isinstance(xforms, wp.array) else np.asarray(xforms)
            scales_np = scales.numpy() if isinstance(scales, wp.array) else np.asarray(scales) if scales is not None else None

        existing_ids = self._instance_name_to_optix_ids.get(name)

        if existing_ids is not None and len(existing_ids) == transform_count:
            for i, inst_id in enumerate(existing_ids):
                if use_device_path:
                    mat = xforms_np[i]
                else:
                    mat = self._build_instance_matrix(xforms_np[i], scales_np[i] if scales_np is not None else None)
                self._scene.set_instance_transform(inst_id, mat)
            self._tlas_dirty = True
        else:
            instance_ids = []
            for i in range(transform_count):
                if use_device_path:
                    mat = xforms_np[i]
                else:
                    mat = self._build_instance_matrix(xforms_np[i], scales_np[i] if scales_np is not None else None)
                inst_id = self._scene.add_instance(mesh_id, transform=mat)
                instance_ids.append(inst_id)
            self._instance_name_to_optix_ids[name] = instance_ids
            self._tlas_dirty = True

        # Assign per-instance materials from colors (sRGB -> linear conversion)
        if colors is not None and self._scene is not None:
            colors_np = colors.numpy() if isinstance(colors, wp.array) else np.asarray(colors)
            materials_np = materials.numpy() if isinstance(materials, wp.array) and materials is not None else None

            inst_ids = self._instance_name_to_optix_ids.get(name, [])
            for i in range(min(len(inst_ids), len(colors_np))):
                r_srgb = round(float(colors_np[i][0]), 2)
                g_srgb = round(float(colors_np[i][1]), 2)
                b_srgb = round(float(colors_np[i][2]), 2)
                roughness = round(float(materials_np[i][0]), 3) if materials_np is not None and i < len(materials_np) else 0.5
                metallic = round(float(materials_np[i][1]), 3) if materials_np is not None and i < len(materials_np) else 0.0

                cache_key = (r_srgb, g_srgb, b_srgb, roughness, metallic)
                mat_id = self._material_cache.get(cache_key)
                if mat_id is None:
                    r_lin = self._srgb_to_linear(r_srgb)
                    g_lin = self._srgb_to_linear(g_srgb)
                    b_lin = self._srgb_to_linear(b_srgb)
                    mat_id = self._scene.materials.add_pbr(
                        base_color=(r_lin, g_lin, b_lin),
                        roughness=roughness,
                        metallic=metallic,
                    )
                    self._material_cache[cache_key] = mat_id

                self._instance_material_map[inst_ids[i]] = mat_id

            self._materials_dirty = True

    def _pack_instance_mats_device_to_host(self, name: str, xforms: wp.array, scales, count: int):
        """Pack instance matrices on GPU, stage to pinned host, return numpy views."""
        capacity = self._instance_mat_capacity.get(name, 0)
        if capacity < count:
            new_capacity = max(count, 1, capacity * 2)
            self._instance_mats_device[name] = wp.empty(new_capacity, dtype=wp.mat44, device="cuda")
            self._instance_mats_host[name] = wp.empty(new_capacity, dtype=wp.mat44, device="cpu", pinned=True)
            self._instance_mat_capacity[name] = new_capacity

        mats_dev = self._instance_mats_device[name]
        mats_host = self._instance_mats_host[name]
        mats_dev_view = mats_dev[:count]
        mats_host_view = mats_host[:count]

        if scales is None:
            if self._default_scales_count < count:
                self._default_scales_dev = wp.empty(max(count, self._default_scales_count * 2, 64), dtype=wp.vec3, device="cuda")
                self._default_scales_dev.fill_(wp.vec3(1.0, 1.0, 1.0))
                self._default_scales_count = self._default_scales_dev.shape[0]
            scales_dev = self._default_scales_dev
        else:
            scales_dev = scales

        up_mode = 0
        if self._global_transform is not None:
            # Mapping based on set_model()/up-axis conversion behavior.
            if self.model is not None and self.model.up_axis == 2:
                up_mode = 1
            elif self.model is not None and self.model.up_axis == 0:
                up_mode = 2

        wp.launch(
            _pack_instance_mats_from_xforms,
            dim=count,
            inputs=[xforms, scales_dev, up_mode],
            outputs=[mats_dev_view],
            device="cuda",
            record_tape=False,
        )
        wp.copy(mats_host_view, mats_dev_view)
        wp.synchronize_device("cuda")

        mats_np = mats_host_view.numpy()
        return mats_np, None

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        pass

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        pass

    @override
    def log_array(self, name, array):
        pass

    @override
    def log_scalar(self, name, value):
        pass

    @override
    def end_frame(self):
        self._ensure_api()
        self._ensure_window()

        self._sync_camera()

        if self._scene is not None and len(self._scene._meshes) > 0:
            mesh_count = len(self._scene._meshes)
            if not self._built:
                self._build_scene_if_needed()
            elif mesh_count > self._meshes_at_last_build:
                self._api.build_scene()
                self._meshes_at_last_build = mesh_count
                self._tlas_dirty = False
                self._upload_instance_materials()
            elif self._tlas_dirty:
                self._scene.rebuild_tlas()
                self._tlas_dirty = False

        if self._materials_dirty and self._built:
            self._upload_instance_materials()
            self._materials_dirty = False

        if self._built:
            self._api.viewer.render()

        if not self._headless and self._window is not None:
            if self._built:
                self._blit_to_window()
            self._process_window_events()

        self._update_timing()

    @override
    def is_running(self) -> bool:
        if self._headless:
            return self._running
        if self._window is not None:
            return not self._window.has_exit
        return self._running

    @override
    def is_paused(self) -> bool:
        return self._paused

    @override
    def is_key_down(self, key) -> bool:
        try:
            import pyglet
        except ImportError:
            return False

        if isinstance(key, str):
            key = key.lower()
            if len(key) == 1 and key.isalpha():
                key_code = getattr(pyglet.window.key, key.upper(), None)
            elif len(key) == 1 and key.isdigit():
                key_code = getattr(pyglet.window.key, f"_{key}", None)
            else:
                special = {
                    "space": pyglet.window.key.SPACE,
                    "escape": pyglet.window.key.ESCAPE,
                    "enter": pyglet.window.key.ENTER,
                    "tab": pyglet.window.key.TAB,
                    "shift": pyglet.window.key.LSHIFT,
                    "ctrl": pyglet.window.key.LCTRL,
                    "alt": pyglet.window.key.LALT,
                    "up": pyglet.window.key.UP,
                    "down": pyglet.window.key.DOWN,
                    "left": pyglet.window.key.LEFT,
                    "right": pyglet.window.key.RIGHT,
                }
                key_code = special.get(key)
            if key_code is None:
                return False
        else:
            key_code = key

        return key_code in self._keys_down

    @override
    def close(self):
        self._running = False
        if self._window is not None:
            from pyglet import gl

            if self._pbo is not None:
                gl.glDeleteBuffers(1, self._pbo)
                self._pbo = None
            self._cuda_gl = None
            self._window.close()
            self._window = None
        if self._api is not None:
            self._api.close()

    def get_frame(self, target_image: wp.array | None = None, render_ui: bool = False) -> wp.array:
        """Retrieve the last rendered frame as a GPU array.

        Args:
            target_image: Optional pre-allocated array with shape ``(height, width, 3)``
                and dtype ``wp.uint8``.
            render_ui: Ignored (OptiX viewer has no ImGui UI).

        Returns:
            GPU array of shape ``(height, width, 3)`` with dtype ``wp.uint8``.
        """
        self._ensure_api()

        h, w = self._height, self._width
        tonemapped = self._api.viewer._tonemapper.output

        if target_image is None:
            target_image = wp.empty(shape=(h, w, 3), dtype=wp.uint8, device="cuda")

        if target_image.shape != (h, w, 3):
            raise ValueError(f"target_image shape must be ({h}, {w}, 3), got {target_image.shape}")

        wp.launch(
            _extract_rgb_from_tonemapped,
            dim=(w, h),
            inputs=[tonemapped, target_image, w, h],
            device="cuda",
        )
        return target_image

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _srgb_to_linear(c: float) -> float:
        """Convert a single sRGB channel value to linear space.

        Matches hybrid_viewer.py's conversion for correct PBR shading.
        """
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    @staticmethod
    def _quat_scale_to_mat4(xform_np, scale_np=None) -> np.ndarray:
        """Convert a Newton transform (pos + quat) and optional scale to a 4x4 matrix.

        Matches pathtracer_api.py ``_build_transform``: rotation is stored row-major
        and scale is applied per-column (axis-aligned in world space).
        """
        px, py, pz = float(xform_np[0]), float(xform_np[1]), float(xform_np[2])
        qx, qy, qz, qw = float(xform_np[3]), float(xform_np[4]), float(xform_np[5]), float(xform_np[6])

        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz

        m = np.eye(4, dtype=np.float32)
        m[0, 0] = 1.0 - 2.0 * (yy + zz)
        m[0, 1] = 2.0 * (xy - wz)
        m[0, 2] = 2.0 * (xz + wy)
        m[1, 0] = 2.0 * (xy + wz)
        m[1, 1] = 1.0 - 2.0 * (xx + zz)
        m[1, 2] = 2.0 * (yz - wx)
        m[2, 0] = 2.0 * (xz - wy)
        m[2, 1] = 2.0 * (yz + wx)
        m[2, 2] = 1.0 - 2.0 * (xx + yy)
        m[0, 3] = px
        m[1, 3] = py
        m[2, 3] = pz

        if scale_np is not None:
            sx, sy, sz = float(scale_np[0]), float(scale_np[1]), float(scale_np[2])
            m[:3, 0] *= sx
            m[:3, 1] *= sy
            m[:3, 2] *= sz

        return m

    def _build_instance_matrix(self, xform_np, scale_np=None) -> np.ndarray:
        """Build a 4x4 instance matrix with the global up-axis transform applied.

        OptiX uses column-vector convention (``p' = M * p``), so the global
        transform (coordinate system conversion) must pre-multiply the local
        object transform: ``final = global @ local``.
        """
        local = self._quat_scale_to_mat4(xform_np, scale_np)
        if self._global_transform is not None:
            return self._global_transform @ local
        return local

    def _camera_changed(self) -> bool:
        """Return True if the camera moved since the last sync (or never synced)."""
        if self._prev_cam_pos is None:
            return True
        if not np.allclose(self._cam_pos, self._prev_cam_pos, atol=1e-6):
            return True
        if abs(self._cam_pitch - self._prev_cam_pitch) > 1e-4:
            return True
        if abs(self._cam_yaw - self._prev_cam_yaw) > 1e-4:
            return True
        if abs(self._cam_fov - self._prev_cam_fov) > 1e-4:
            return True
        return False

    def _sync_camera(self):
        """Push the viewer camera state into the OptiX camera only when it changes.

        Uses the C# camera convention (matching hybrid_viewer / PythonBridge):
        yaw=0 looks in +Z, X = sin(yaw)*cos(pitch), Z = cos(yaw)*cos(pitch).
        """
        if self._api is None:
            return

        if not self._camera_changed():
            return

        yaw_rad = math.radians(self._cam_yaw)
        pitch_rad = math.radians(self._cam_pitch)

        cos_p = math.cos(pitch_rad)
        fwd = np.array(
            [
                math.sin(yaw_rad) * cos_p,
                math.sin(pitch_rad),
                math.cos(yaw_rad) * cos_p,
            ],
            dtype=np.float32,
        )

        target = self._cam_pos + fwd
        cam = self._api.viewer.camera
        cam.position = self._cam_pos.copy()
        cam.target = target
        cam.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cam.fov = self._cam_fov
        cam.set_aspect_ratio(self._width, self._height)

        self._prev_cam_pos = self._cam_pos.copy()
        self._prev_cam_pitch = self._cam_pitch
        self._prev_cam_yaw = self._cam_yaw
        self._prev_cam_fov = self._cam_fov

    def _build_scene_if_needed(self):
        """Build the OptiX scene (BLAS + TLAS) if not yet built."""
        if self._built:
            return
        if self._api is None:
            return
        if self._scene is None or len(self._scene._meshes) == 0:
            return

        self._api.build_scene()
        self._meshes_at_last_build = len(self._scene._meshes)
        self._built = True
        self._tlas_dirty = False

        self._upload_instance_materials()
        self._materials_dirty = False

    def _upload_instance_materials(self):
        """Overwrite the per-instance material ID buffer on the GPU.

        Also rebuilds the compact material table so the shader can look up
        any materials that were added after the initial scene build.
        """
        if not self._instance_material_map or self._scene is None:
            return
        if self._scene._instance_material_ids is None:
            return

        num_instances = len(self._scene._instances)
        mat_ids = self._scene._instance_material_ids.numpy()
        for inst_id, mat_id in self._instance_material_map.items():
            if 0 <= inst_id < num_instances:
                mat_ids[inst_id] = np.uint32(mat_id)

        self._scene._instance_material_ids = wp.array(mat_ids, dtype=wp.uint32, device="cuda")

        self._rebuild_compact_materials()

    def _rebuild_compact_materials(self):
        """Rebuild the compact material table on the GPU from current materials."""
        if self._scene is None:
            return
        mat_count = self._scene.materials.count
        if mat_count == 0:
            return

        compact_dt = np.dtype(
            [
                ("baseColor", np.float32, (3,)),
                ("emissive", np.float32, (3,)),
                ("roughness", np.float32),
                ("metallic", np.float32),
                ("transmission", np.float32),
                ("ior", np.float32),
                ("specularColor", np.float32, (3,)),
                ("specular", np.float32),
                ("clearcoat", np.float32),
                ("clearcoatRoughness", np.float32),
                ("clearcoatNormalTexIndex", np.int32),
                ("clearcoatNormalTexCoord", np.int32),
                ("opacity", np.float32),
                ("baseColorTexIndex", np.int32),
                ("baseColorTexCoord", np.int32),
                ("metallicRoughnessTexIndex", np.int32),
                ("metallicRoughnessTexCoord", np.int32),
                ("normalTexIndex", np.int32),
                ("normalTexCoord", np.int32),
                ("emissiveTexIndex", np.int32),
                ("emissiveTexCoord", np.int32),
                ("normalScale", np.float32),
                ("baseColorUvTransform", np.float32, (6,)),
                ("metallicRoughnessUvTransform", np.float32, (6,)),
                ("normalUvTransform", np.float32, (6,)),
                ("emissiveUvTransform", np.float32, (6,)),
                ("clearcoatNormalUvTransform", np.float32, (6,)),
            ],
            align=True,
        )
        compact = np.zeros(mat_count, dtype=compact_dt)
        for i, mat in enumerate(self._scene.materials._materials):
            compact[i]["baseColor"] = mat["pbrBaseColorFactor"][:3]
            compact[i]["emissive"] = mat["emissiveFactor"]
            compact[i]["roughness"] = mat["pbrRoughnessFactor"]
            compact[i]["metallic"] = mat["pbrMetallicFactor"]
            compact[i]["transmission"] = mat["transmissionFactor"]
            compact[i]["ior"] = mat["ior"]
            compact[i]["specularColor"] = mat["specularColorFactor"]
            compact[i]["specular"] = mat["specularFactor"]
            compact[i]["clearcoat"] = mat["clearcoatFactor"]
            compact[i]["clearcoatRoughness"] = mat["clearcoatRoughness"]
            compact[i]["clearcoatNormalTexIndex"] = mat["clearcoatNormalTexture"]["index"]
            compact[i]["clearcoatNormalTexCoord"] = mat["clearcoatNormalTexture"]["texCoord"]
            compact[i]["opacity"] = mat["pbrBaseColorFactor"][3]
            compact[i]["baseColorTexIndex"] = mat["pbrBaseColorTexture"]["index"]
            compact[i]["baseColorTexCoord"] = mat["pbrBaseColorTexture"]["texCoord"]
            compact[i]["metallicRoughnessTexIndex"] = mat["pbrMetallicRoughnessTexture"]["index"]
            compact[i]["metallicRoughnessTexCoord"] = mat["pbrMetallicRoughnessTexture"]["texCoord"]
            compact[i]["normalTexIndex"] = mat["normalTexture"]["index"]
            compact[i]["normalTexCoord"] = mat["normalTexture"]["texCoord"]
            compact[i]["emissiveTexIndex"] = mat["emissiveTexture"]["index"]
            compact[i]["emissiveTexCoord"] = mat["emissiveTexture"]["texCoord"]
            compact[i]["normalScale"] = mat["normalTextureScale"]
            for field_name, mat_key in (
                ("baseColorUvTransform", "pbrBaseColorTexture"),
                ("metallicRoughnessUvTransform", "pbrMetallicRoughnessTexture"),
                ("normalUvTransform", "normalTexture"),
                ("emissiveUvTransform", "emissiveTexture"),
                ("clearcoatNormalUvTransform", "clearcoatNormalTexture"),
            ):
                compact[i][field_name] = (
                    mat[mat_key]["uvTransform00"],
                    mat[mat_key]["uvTransform01"],
                    mat[mat_key]["uvTransform02"],
                    mat[mat_key]["uvTransform10"],
                    mat[mat_key]["uvTransform11"],
                    mat[mat_key]["uvTransform12"],
                )
        compact_bytes = compact.view(np.uint8).reshape(-1)
        self._scene._compact_materials = wp.array(compact_bytes, dtype=wp.uint8, device="cuda")

    def _blit_to_window(self):
        """Copy the tonemapped output to the Pyglet GL texture via PBO."""
        from pyglet import gl

        viewer = self._api.viewer
        tonemapped = viewer._tonemapper.output
        w, h = self._width, self._height

        mapped = self._cuda_gl.map(dtype=wp.uint32, shape=(w * h,))
        wp.launch(
            _pack_display_rgba8,
            dim=(w, h),
            inputs=[tonemapped, mapped, w, h],
            device="cuda",
        )
        self._cuda_gl.unmap()

        self._window.switch_to()
        self._window.clear()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self._pbo)
        gl.glBindTexture(self._gl_texture.target, self._gl_texture.id)
        gl.glTexSubImage2D(
            self._gl_texture.target,
            0, 0, 0,
            w, h,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(self._gl_texture.target, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self._sprite.draw()
        self._window.flip()

    def _process_window_events(self):
        """Poll Pyglet events and update camera from keyboard input."""
        self._window.dispatch_events()

        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now

        self._update_camera_movement(dt)

    def _update_camera_movement(self, dt: float):
        """Move camera directly at constant speed (no inertia).

        Matches hybrid_viewer._update_camera_from_input: immediate response,
        no velocity damping.
        """
        try:
            from pyglet.window import key
        except ImportError:
            return

        yaw_rad = math.radians(self._cam_yaw)
        pitch_rad = math.radians(self._cam_pitch)

        cos_p = math.cos(pitch_rad)
        forward = np.array(
            [math.sin(yaw_rad) * cos_p, math.sin(pitch_rad), math.cos(yaw_rad) * cos_p],
            dtype=np.float32,
        )
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn > 1e-6:
            right /= rn

        direction = np.zeros(3, dtype=np.float32)
        has_input = False
        if key.W in self._keys_down or key.UP in self._keys_down:
            direction += forward
            has_input = True
        if key.S in self._keys_down or key.DOWN in self._keys_down:
            direction -= forward
            has_input = True
        if key.A in self._keys_down or key.LEFT in self._keys_down:
            direction -= right
            has_input = True
        if key.D in self._keys_down or key.RIGHT in self._keys_down:
            direction += right
            has_input = True
        if key.Q in self._keys_down:
            direction -= world_up
            has_input = True
        if key.E in self._keys_down:
            direction += world_up
            has_input = True

        if has_input:
            dn = float(np.linalg.norm(direction))
            if dn > 1e-6:
                self._cam_pos += (direction / dn) * self._cam_speed * dt

    def _update_timing(self):
        """Track FPS and update window title."""
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_t
        if elapsed >= 0.5:
            delta = self._frame_count - self._fps_last_frames
            self._current_fps = delta / elapsed if elapsed > 0.0 else 0.0
            self._fps_last_t = now
            self._fps_last_frames = self._frame_count
            if self._window is not None:
                self._window.set_caption(f"Newton Viewer (OptiX) - {self._current_fps:.1f} FPS")

    # ------------------------------------------------------------------
    # Pyglet event handlers (pushed via push_handlers)
    # ------------------------------------------------------------------

    def on_key_press(self, symbol, modifiers):
        self._keys_down.add(symbol)
        try:
            from pyglet.window import key
        except ImportError:
            return

        if symbol == key.SPACE:
            self._paused = not self._paused
        elif symbol == key.ESCAPE:
            if self._window is not None:
                self._window.close()
        elif self._api is not None:
            viewer = self._api.viewer
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

    def on_key_release(self, symbol, modifiers):
        self._keys_down.discard(symbol)

    def on_mouse_press(self, x, y, button, modifiers):
        self._mouse_buttons |= button
        self._last_mouse_x = float(x)
        self._last_mouse_y = float(y)

    def on_mouse_release(self, x, y, button, modifiers):
        self._mouse_buttons &= ~button

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        try:
            from pyglet.window import mouse
        except ImportError:
            return

        if buttons & mouse.LEFT:
            sensitivity = 0.1
            self._cam_yaw -= float(dx) * sensitivity
            self._cam_pitch += float(dy) * sensitivity
            self._cam_pitch = max(-89.0, min(89.0, self._cam_pitch))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self._cam_fov -= float(scroll_y) * 2.0
        self._cam_fov = max(15.0, min(90.0, self._cam_fov))

    def on_close(self):
        self._running = False

    def on_resize(self, width, height):
        if width == 0 or height == 0:
            return
        self._width = width
        self._height = height

        if self._api is not None:
            self._api.viewer.resize(width, height)
            self._api.viewer.sample_index = 0
            self._api.viewer.frame_index = 0

        self._recreate_gl_resources()

        # Force camera re-sync so aspect ratio is updated
        self._prev_cam_pos = None

    def on_deactivate(self):
        pass
