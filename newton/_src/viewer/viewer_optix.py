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

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core.types import override
from .optix.pathtracing import PathTracingBridge
from .viewer import ViewerBase

if TYPE_CHECKING:
    import newton


class ViewerOptix(ViewerBase):
    """Newton viewer backend that renders through the OptiX path tracer."""

    def __init__(self, width: int = 1280, height: int = 720, headless: bool = False, enable_dlss_rr: bool = True):
        super().__init__()
        self._width = int(width)
        self._height = int(height)
        self._headless = bool(headless)
        self._bridge = PathTracingBridge(width=self._width, height=self._height, enable_dlss_rr=bool(enable_dlss_rr))
        self._running = True
        self._paused = False
        self._scene_dirty = False

        # Name/path registries to preserve stable IDs across frames.
        self._mesh_name_to_id: dict[str, int] = {}
        self._instance_ids_by_name: dict[str, list[int]] = {}
        self._instance_mesh_by_name: dict[str, str] = {}

    @override
    def set_model(self, model: newton.Model, max_worlds: int | None = None):
        super().set_model(model, max_worlds=max_worlds)
        self._scene_dirty = True

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
            return

        for i, instance_id in enumerate(current_ids):
            p = xforms_np[i, 0:3]
            q = xforms_np[i, 3:7]
            s = float(scales_np[i, 0]) if scales_np is not None and i < len(scales_np) else 1.0
            self._bridge.set_instance_transform(instance_id, p, q, s)
        self._scene_dirty = True

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
        self._bridge.render_frame()

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
        self._bridge.close()


__all__ = ["ViewerOptix"]
