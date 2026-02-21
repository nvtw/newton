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

"""OptiX pathtracing scene example based on the former build-steps scene."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples


def _build_scene_instances() -> tuple[wp.array, wp.array, wp.array]:
    cube_xforms = []
    cube_colors = []
    cube_materials = []

    color_index = 0
    for x in (-2.0, 0.0, 2.0):
        for y in (0.0, 2.0):
            for z in (-2.0, 0.0, 2.0):
                cube_xforms.append(wp.transform((x, y, z), wp.quat_identity()))
                cube_colors.append(
                    wp.vec3(
                        (60 + color_index * 20) / 255.0,
                        (120 + color_index * 10) / 255.0,
                        (220 - color_index * 8) / 255.0,
                    )
                )
                # roughness, metallic, checker, texture_enable
                cube_materials.append(wp.vec4(0.35, 0.05, 0.0, 0.0))
                color_index += 1

    return (
        wp.array(cube_xforms, dtype=wp.transform),
        wp.array(cube_colors, dtype=wp.vec3),
        wp.array(cube_materials, dtype=wp.vec4),
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame = 0
        self.loaded_external_scene = False
        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 2.0

        self._try_load_external_scene()

        self.cube_xforms, self.cube_colors, self.cube_materials = _build_scene_instances()
        self.plane_xforms = wp.array([wp.transform_identity()], dtype=wp.transform)
        self.plane_colors = wp.array([wp.vec3(220.0 / 255.0, 220.0 / 255.0, 220.0 / 255.0)], dtype=wp.vec3)
        self.plane_materials = wp.array([wp.vec4(0.7, 0.0, 0.0, 0.0)], dtype=wp.vec4)

    def _resolve_a_beautiful_game_gltf(self) -> Path | None:
        if self.args.scene_gltf:
            p = Path(self.args.scene_gltf).expanduser().resolve()
            if p.exists():
                return p
            raise FileNotFoundError(f"--scene-gltf does not exist: {p}")

        local_candidates = [
            Path(r"C:\git\downloaded_resources\ABeautifulGame\glTF\ABeautifulGame.gltf"),
            Path(r"C:\git\downloaded_resources\ABeautifulGame\glTF-Binary\ABeautifulGame.glb"),
            Path(r"C:\git\single-file-vulkan-pathtracing\assets\gltf\ABeautifulGame\ABeautifulGame.gltf"),
        ]
        for p in local_candidates:
            if p.exists():
                return p

        if self.args.no_auto_download:
            return None

        try:
            downloaded_gltf = _download_a_beautiful_game_gltf()
            if downloaded_gltf.exists():
                return downloaded_gltf
        except Exception as e:  # pragma: no cover - network-dependent fallback path
            print(f"[optix] auto-download failed: {e}")

        return None

    def _resolve_racerx_glbs(self) -> tuple[Path, Path]:
        b3_candidates = [
            Path(r"C:\Documents\Meshes\RacerX\glb\B3_physics.glb"),
            Path.home() / "Documents" / "Meshes" / "RacerX" / "glb" / "B3_physics.glb",
        ]
        a3_candidates = [
            Path(r"C:\Documents\Meshes\RacerX\glb3\A3_physics.glb"),
            Path.home() / "Documents" / "Meshes" / "RacerX" / "glb3" / "A3_physics.glb",
        ]

        b3 = next((p for p in b3_candidates if p.exists()), None)
        a3 = next((p for p in a3_candidates if p.exists()), None)
        if b3 is None or a3 is None:
            raise RuntimeError(
                "Could not locate RacerX GLBs. Expected B3_physics.glb and A3_physics.glb under "
                "C:\\Documents\\Meshes\\RacerX (or your user Documents)."
            )
        return b3, a3

    def _make_root_transform(self, x: float, y: float, z: float, yaw_rad: float) -> np.ndarray:
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return m

    def _set_camera_from_scene_bounds(self):
        api = getattr(self.viewer, "_api", None)
        if api is None:
            return
        scene = getattr(api, "scene", None)
        meshes = getattr(scene, "_meshes", None) if scene is not None else None
        if not meshes:
            return

        mins = []
        maxs = []
        for mesh in meshes:
            verts = getattr(mesh, "vertices", None)
            if verts is None or len(verts) == 0:
                continue
            mins.append(np.min(verts, axis=0))
            maxs.append(np.max(verts, axis=0))
        if not mins:
            return

        bmin = np.min(np.vstack(mins), axis=0)
        bmax = np.max(np.vstack(maxs), axis=0)
        center = 0.5 * (bmin + bmax)
        extent = np.maximum(bmax - bmin, 1.0e-3)
        radius = 0.5 * float(np.linalg.norm(extent))

        cam_pos = wp.vec3(float(center[0]), float(center[1] + max(2.0, radius * 0.25)), float(center[2] + max(6.0, radius * 1.8)))
        self.viewer.set_camera(cam_pos, pitch=-12.0, yaw=180.0)

    def _set_viewer_camera_look_at(self, position: tuple[float, float, float], target: tuple[float, float, float]):
        pos = np.asarray(position, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        d = tgt - pos
        dn = float(np.linalg.norm(d))
        if dn < 1.0e-6:
            return
        d /= dn
        yaw = math.degrees(math.atan2(float(d[0]), float(d[2])))
        pitch = math.degrees(math.asin(float(np.clip(d[1], -1.0, 1.0))))
        self.viewer.set_camera(wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), pitch=pitch, yaw=yaw)

    def _load_racerx_circle(self, api):
        b3_path, a3_path = self._resolve_racerx_glbs()
        scene_paths = (b3_path, a3_path)
        count = max(2, int(self.args.circle_count))
        radius = float(self.args.circle_radius)

        api.clear_scene()

        scene = api.scene
        template_mesh_ids: list[list[int]] = []
        template_instance_ids: list[list[int]] = []
        template_min_y: list[float] = []
        template_xz_extent: list[float] = []

        # Load each unique asset once, then reuse its meshes via instancing.
        for scene_path in scene_paths:
            mesh_start = scene.mesh_count
            inst_start = scene.instance_count
            ok = api.load_scene_from_gltf(
                str(scene_path),
                root_transform=None,
                clear_existing=False,
                build_scene=False,
            )
            if not ok:
                raise RuntimeError(f"Failed to load RacerX scene: {scene_path}")
            mesh_end = scene.mesh_count
            inst_end = scene.instance_count
            template_mesh_ids.append(list(range(mesh_start, mesh_end)))
            template_instance_ids.append(list(range(inst_start, inst_end)))
            mesh_ids = template_mesh_ids[-1]
            if not mesh_ids:
                raise RuntimeError(f"Loaded RacerX scene had no meshes: {scene_path}")

            min_y = float("inf")
            max_x_abs = 0.0
            max_z_abs = 0.0
            for mesh_id in mesh_ids:
                verts = scene._meshes[mesh_id].vertices
                min_y = min(min_y, float(np.min(verts[:, 1])))
                max_x_abs = max(max_x_abs, float(np.max(np.abs(verts[:, 0]))))
                max_z_abs = max(max_z_abs, float(np.max(np.abs(verts[:, 2]))))
            template_min_y.append(min_y)
            template_xz_extent.append(max(max_x_abs, max_z_abs))

        per_asset_placed = [0, 0]
        for i in range(count):
            asset_index = i % 2
            angle = (2.0 * math.pi * i) / count
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            # Make each car face outward from the circle center.
            yaw = angle + math.pi
            root_transform = self._make_root_transform(x, 0.0, z, yaw)

            if per_asset_placed[asset_index] == 0:
                # Reuse the initial template instances for the first placement.
                for inst_id in template_instance_ids[asset_index]:
                    scene.set_instance_transform(inst_id, root_transform)
            else:
                # Additional placements instantiate existing meshes only.
                for mesh_id in template_mesh_ids[asset_index]:
                    scene.add_instance(mesh_id, transform=root_transform)
            per_asset_placed[asset_index] += 1

        # Place ground based on loaded mesh bounds so it sits just below vehicles.
        global_min_y = min(template_min_y)
        max_car_extent_xz = max(template_xz_extent)
        # Align ground top exactly with the lowest vehicle point so tires touch.
        ground_top_y = global_min_y
        ground_thickness = max(0.15, 0.05 * max(1.0, max_car_extent_xz))
        ground_extent = max(24.0, radius + max_car_extent_xz + 3.0)
        ground_mat = api.create_diffuse_material((0.42, 0.44, 0.46))
        api.add_box(
            (-ground_extent, ground_top_y - ground_thickness, -ground_extent),
            (ground_extent, ground_top_y, ground_extent),
            ground_mat,
        )

        api.build_scene()
        look_y = global_min_y + max(0.6, 0.25 * max_car_extent_xz)
        cam_height = look_y + max(4.5, 0.35 * radius)
        cam_dist = max(radius * 1.2, 10.0)
        self._set_viewer_camera_look_at(
            position=(cam_dist, cam_height, cam_dist),
            target=(0.0, look_y, 0.0),
        )
        self.loaded_external_scene = True
        print(f"[optix] loaded RacerX circle scene: count={count}, radius={radius:.2f}")

    def _try_load_external_scene(self):
        if self.args.viewer != "optix":
            return
        if not hasattr(self.viewer, "_ensure_api"):
            return

        self.viewer._ensure_api()
        api = getattr(self.viewer, "_api", None)
        if api is None:
            raise RuntimeError("ViewerOptix PathTracerAPI was not created.")

        if self.args.scene_layout == "racerx-circle":
            self._load_racerx_circle(api)
            return

        gltf_path = self._resolve_a_beautiful_game_gltf()
        if gltf_path is None:
            raise RuntimeError(
                "Could not locate ABeautifulGame (.gltf or .glb). "
                "Provide --scene-gltf, place the scene in C:\\git\\downloaded_resources, "
                "or enable network for auto-download."
            )

        ok = api.load_scene_from_gltf(str(gltf_path))
        if not ok:
            raise RuntimeError(f"Failed to load scene (.gltf/.glb): {gltf_path}")

        if self.args.scene_gltf:
            # Use scene bounds when user overrides asset path.
            self._set_camera_from_scene_bounds()
        else:
            self.viewer.set_camera(wp.vec3(-0.803, 0.340, 0.327), pitch=-21.8, yaw=115.2)
        self.loaded_external_scene = True
        print(f"[optix] loaded ABeautifulGame scene: {gltf_path}")

    def step(self):
        self.frame += 1

    def render(self):
        self.viewer.begin_frame(self.frame / 60.0)

        if not self.loaded_external_scene:
            self.viewer.log_shapes(
                "/optix_pathtracing/cubes",
                newton.GeoType.BOX,
                (0.4, 0.4, 0.4),
                self.cube_xforms,
                self.cube_colors,
                self.cube_materials,
            )
            self.viewer.log_shapes(
                "/optix_pathtracing/ground",
                newton.GeoType.PLANE,
                (30.0, 30.0),
                self.plane_xforms,
                self.plane_colors,
                self.plane_materials,
            )

        self.viewer.end_frame()

    def test_final(self):
        if self.frame <= 0:
            raise ValueError("Example did not advance any frames.")


def _download_a_beautiful_game_gltf() -> Path:
    git_exe = shutil.which("git")
    if git_exe is None:
        raise RuntimeError("git executable not found on PATH.")

    cache_root = Path.home() / ".cache" / "newton" / "external-scenes"
    repo_dir = cache_root / "gltf-sample-assets"
    target_rel = Path("Models/ABeautifulGame/glTF")
    target_dir = repo_dir / target_rel
    target_gltf = target_dir / "ABeautifulGame.gltf"
    cache_root.mkdir(parents=True, exist_ok=True)

    if target_gltf.exists():
        return target_gltf

    if not repo_dir.exists():
        subprocess.run(
            [git_exe, "clone", "--depth", "1", "https://github.com/KhronosGroup/glTF-Sample-Assets.git", str(repo_dir)],
            check=True,
        )
    else:
        subprocess.run([git_exe, "-C", str(repo_dir), "pull", "--ff-only"], check=True)

    if not target_gltf.exists():
        raise RuntimeError(f"ABeautifulGame.gltf not found in cloned repo under {target_rel}.")
    return target_gltf


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="optix")
    parser.add_argument(
        "--scene-gltf",
        type=str,
        default=None,
        help="Path to ABeautifulGame.gltf or ABeautifulGame.glb (optional override).",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable auto-download from KhronosGroup/glTF-Sample-Assets.",
    )
    parser.add_argument(
        "--scene-layout",
        type=str,
        default="beautiful-game",
        choices=("beautiful-game", "racerx-circle"),
        help="Choose external scene layout to load.",
    )
    parser.add_argument(
        "--circle-count",
        type=int,
        default=12,
        help="Number of vehicles to place on the RacerX circle layout.",
    )
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=18.0,
        help="Circle radius [m] for the RacerX circle layout.",
    )
    viewer, args = newton.examples.init(parser=parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
