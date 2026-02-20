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

"""PythonBridge-style OptiX API for standalone path tracing usage."""

from __future__ import annotations

from collections.abc import Iterable
import importlib.util
import os
import sys

import numpy as np

from .pathtracing_viewer import PathTracingViewer, _get_optix_include_dir
from .scene import Mesh


def _quat_to_mat3(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
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


def _build_transform(position: Iterable[float], rotation_xyzw: Iterable[float], scale: float | Iterable[float]) -> np.ndarray:
    """Build a 4x4 row-major transform matrix from position/quaternion/scale."""
    px, py, pz = [float(v) for v in position]
    qx, qy, qz, qw = [float(v) for v in rotation_xyzw]
    if isinstance(scale, Iterable) and not isinstance(scale, (str, bytes)):
        sx, sy, sz = [float(v) for v in scale]
    else:
        s = float(scale)
        sx = sy = sz = s
    m = np.eye(4, dtype=np.float32)
    r = _quat_to_mat3(qx, qy, qz, qw)
    # Column scale for non-uniform axis scaling.
    r[:, 0] *= sx
    r[:, 1] *= sy
    r[:, 2] *= sz
    m[:3, :3] = r
    m[:3, 3] = np.array([px, py, pz], dtype=np.float32)
    return m


class PathTracerAPI:
    """High-level API for driving the OptiX path tracer directly from Python."""

    def __init__(self, width: int = 1280, height: int = 720, enable_dlss_rr: bool = True):
        self.width = int(width)
        self.height = int(height)
        self._viewer = PathTracingViewer(
            width=self.width,
            height=self.height,
            scene_setup=lambda _scene: None,
            enable_dlss_rr=bool(enable_dlss_rr),
            accumulate_samples=False,
        )
        self._built = False
        self._running = True
        self._time = 0.0
        self._init_error: str | None = None

    def _build_init_error_message(self) -> str:
        """Build actionable diagnostics for OptiX initialization failures."""
        optix_available = importlib.util.find_spec("optix") is not None
        optix_sdk_env = os.environ.get("OPTIX_SDK_INCLUDE_DIR")
        optix_sdk_auto = _get_optix_include_dir()
        return (
            "PathTracerAPI initialization failed.\n"
            f"- python executable: {sys.executable}\n"
            f"- optix module importable: {optix_available}\n"
            f"- OPTIX_SDK_INCLUDE_DIR: {optix_sdk_env!r}\n"
            f"- auto-detected OptiX include dir: {optix_sdk_auto!r}\n"
            "Ensure `optix` is installed in THIS interpreter and OptiX SDK include path is valid."
        )

    @property
    def viewer(self) -> PathTracingViewer:
        return self._viewer

    @property
    def scene(self):
        return self._viewer._scene  # internal, initialized by build()

    def _require_scene(self):
        """Ensure an initialized scene is available or raise a clear error."""
        if self._viewer._scene is None:
            ok = self.initialize()
            if (not ok) or self._viewer._scene is None:
                raise RuntimeError(self._init_error or self._build_init_error_message())
        return self._viewer._scene

    def initialize(self) -> bool:
        if self._built:
            return True
        self._built = bool(self._viewer.build())
        if not self._built:
            self._init_error = self._build_init_error_message()
        return self._built

    def is_running(self) -> bool:
        return self._running

    def close(self):
        self._running = False

    def begin_frame(self, time_sec: float):
        self._time = float(time_sec)

    def end_frame(self):
        # Rendering is explicit in render_frame(); kept for API parity.
        return None

    def render_frame(self):
        self.initialize()
        self._viewer.render()

    def get_frame(self) -> np.ndarray:
        self.initialize()
        return self._viewer.get_output()

    def get_frame_uint8(self) -> np.ndarray:
        image = np.clip(self.get_frame(), 0.0, 1.0)
        return (image * 255.0).astype(np.uint8)

    def build_scene(self):
        scene = self._require_scene()
        scene.build(self._viewer._optix)
        self._viewer._create_sbt()
        self._viewer.sample_index = 0
        self._viewer.frame_index = 0

    def rebuild_tlas(self):
        scene = self._require_scene()
        scene.rebuild_tlas()
        self._viewer.sample_index = 0
        self._viewer.frame_index = 0

    def clear_scene(self):
        self._require_scene().clear()

    def load_scene_from_gltf(self, gltf_path: str) -> bool:
        ok = bool(self._require_scene().load_from_gltf(gltf_path))
        if ok:
            self.build_scene()
        return ok

    def load_scene_from_obj(self, obj_path: str) -> bool:
        ok = bool(self._require_scene().load_from_obj(obj_path))
        if ok:
            self.build_scene()
        return ok

    def create_mesh(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        material_id: int = 0,
    ) -> int:
        scene = self._require_scene()
        if scene.materials.count == 0:
            scene.materials.add_diffuse((0.8, 0.8, 0.8))
        mat_id = int(material_id)
        if mat_id < 0 or mat_id >= scene.materials.count:
            mat_id = 0
        mesh = Mesh(
            vertices=np.asarray(positions, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.uint32),
            normals=None if normals is None else np.asarray(normals, dtype=np.float32),
            texcoords=None if uvs is None else np.asarray(uvs, dtype=np.float32),
            material_id=mat_id,
        )
        return int(scene.add_mesh(mesh))

    def create_instance(self, mesh_id: int) -> int:
        return int(self._require_scene().add_instance(int(mesh_id)))

    def create_instance_with_transform(
        self,
        mesh_id: int,
        position: Iterable[float],
        rotation_xyzw: Iterable[float],
        scale: float | Iterable[float] = 1.0,
    ) -> int:
        transform = _build_transform(position, rotation_xyzw, scale)
        return int(self._require_scene().add_instance(int(mesh_id), transform=transform))

    def set_instance_transform(
        self,
        instance_id: int,
        position: Iterable[float],
        rotation_xyzw: Iterable[float],
        scale: float | Iterable[float] = 1.0,
    ):
        transform = _build_transform(position, rotation_xyzw, scale)
        self._require_scene().set_instance_transform(int(instance_id), transform)

    def set_instance_transform_matrix(self, instance_id: int, matrix: np.ndarray):
        m = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
        self._require_scene().set_instance_transform(int(instance_id), m)

    def set_instance_transforms_batch(self, instance_ids: Iterable[int], transforms_flat: np.ndarray):
        self.initialize()
        ids = list(instance_ids)
        arr = np.asarray(transforms_flat, dtype=np.float32).reshape(-1, 8)
        for i, instance_id in enumerate(ids):
            if i >= len(arr):
                break
            row = arr[i]
            self.set_instance_transform(
                int(instance_id),
                position=row[0:3],
                rotation_xyzw=row[3:7],
                scale=float(row[7]),
            )

    def create_diffuse_material(self, color: Iterable[float]) -> int:
        return int(self._require_scene().materials.add_diffuse(tuple(float(v) for v in color)))

    def create_metallic_material(self, color: Iterable[float], roughness: float = 0.1) -> int:
        return int(self._require_scene().materials.add_metal(tuple(float(v) for v in color), float(roughness)))

    def create_emissive_material(self, color: Iterable[float], intensity: float = 1.0) -> int:
        return int(self._require_scene().materials.add_emissive(tuple(float(v) for v in color), float(intensity)))

    def create_pbr_material(self, color: Iterable[float], roughness: float, metallic: float) -> int:
        scene = self._require_scene()
        return int(
            scene.materials.add_pbr(
                base_color=tuple(float(v) for v in color),
                roughness=float(roughness),
                metallic=float(metallic),
            )
        )

    def add_box(self, min_pt: Iterable[float], max_pt: Iterable[float], material_id: int) -> int:
        return int(
            self._require_scene().add_box(tuple(float(v) for v in min_pt), tuple(float(v) for v in max_pt), int(material_id))
        )

    def add_sphere(self, center: Iterable[float], radius: float, segments: int, material_id: int) -> int:
        return int(
            self._require_scene().add_sphere(
                tuple(float(v) for v in center),
                float(radius),
                int(segments),
                int(material_id),
            )
        )

    def set_camera_look_at(
        self,
        position: Iterable[float],
        target: Iterable[float],
        up: Iterable[float] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
    ):
        self.initialize()
        self._viewer.camera.position = np.asarray(list(position), dtype=np.float32)
        self._viewer.camera.target = np.asarray(list(target), dtype=np.float32)
        self._viewer.camera.up = np.asarray(list(up), dtype=np.float32)
        self._viewer.camera.fov = float(fov)
        self._viewer.sample_index = 0

    def set_camera_angles(
        self,
        position: Iterable[float],
        yaw: float,
        pitch: float,
        fov: float = 45.0,
    ):
        self.initialize()
        yaw_rad = np.deg2rad(float(yaw))
        pitch_rad = np.deg2rad(float(pitch))
        direction = np.array(
            [
                np.sin(yaw_rad) * np.cos(pitch_rad),
                np.sin(pitch_rad),
                np.cos(yaw_rad) * np.cos(pitch_rad),
            ],
            dtype=np.float32,
        )
        pos = np.asarray(list(position), dtype=np.float32)
        self.set_camera_look_at(pos, pos + direction, (0.0, 1.0, 0.0), float(fov))

    def set_debug_buffer_mode(self, mode: int):
        self.initialize()
        self._viewer.output_mode = int(mode)

    def set_use_procedural_sky(self, enabled: bool):
        self.initialize()
        if enabled:
            self._viewer._env_map = None

    def set_sun_direction(self, x: float, y: float, z: float, intensity: float = 1.0):
        self.initialize()
        direction = np.array([x, y, z], dtype=np.float32)
        nrm = np.linalg.norm(direction)
        if nrm > 0.0:
            direction = direction / nrm
        self._viewer.sky_sun_direction = (float(direction[0]), float(direction[1]), float(direction[2]))
        self._viewer.sky_multiplier = float(intensity)

    def set_sky_parameters(
        self,
        sun_direction: Iterable[float],
        multiplier: float = 1.0,
        haze: float = 0.0,
        red_blue_shift: float = 0.0,
        saturation: float = 1.0,
        horizon_height: float = 0.0,
        ground_color: Iterable[float] = (0.4, 0.4, 0.4),
        horizon_blur: float = 1.0,
        night_color: Iterable[float] = (0.0, 0.0, 0.0),
        sun_disk_intensity: float = 1.0,
        sun_disk_scale: float = 1.0,
        sun_glow_intensity: float = 1.0,
        y_is_up: int = 1,
    ):
        self.initialize()
        self._viewer.sky_sun_direction = tuple(float(v) for v in sun_direction)
        self._viewer.sky_multiplier = float(multiplier)
        self._viewer.sky_haze = float(haze)
        self._viewer.sky_redblueshift = float(red_blue_shift)
        self._viewer.sky_saturation = float(saturation)
        self._viewer.sky_horizon_height = float(horizon_height)
        self._viewer.sky_ground_color = tuple(float(v) for v in ground_color)
        self._viewer.sky_horizon_blur = float(horizon_blur)
        self._viewer.sky_night_color = tuple(float(v) for v in night_color)
        self._viewer.sky_sun_disk_intensity = float(sun_disk_intensity)
        self._viewer.sky_sun_disk_scale = float(sun_disk_scale)
        self._viewer.sky_sun_glow_intensity = float(sun_glow_intensity)
        self._viewer.sky_y_is_up = int(y_is_up)

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        self.initialize()
        self._viewer.set_environment_hdr(hdr_path, scaling=float(scaling))

    def set_environment_color(self, color: Iterable[float]):
        self.initialize()
        self._viewer.set_environment_color(tuple(float(v) for v in color))
