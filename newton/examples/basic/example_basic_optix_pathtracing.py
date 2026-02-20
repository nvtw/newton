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

import shutil
import subprocess
from pathlib import Path

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

        self._try_load_a_beautiful_game()

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

    def _try_load_a_beautiful_game(self):
        if self.args.viewer != "optix":
            return
        if not hasattr(self.viewer, "_ensure_bridge"):
            return

        gltf_path = self._resolve_a_beautiful_game_gltf()
        if gltf_path is None:
            raise RuntimeError(
                "Could not locate ABeautifulGame.gltf. "
                "Provide --scene-gltf, place the scene in C:\\git\\downloaded_resources, "
                "or enable network for auto-download."
            )

        self.viewer._ensure_bridge()
        bridge = getattr(self.viewer, "_bridge", None)
        if bridge is None:
            raise RuntimeError("ViewerOptix bridge was not created.")

        ok = bridge.load_scene_from_gltf(str(gltf_path))
        if not ok:
            raise RuntimeError(f"Failed to load glTF scene: {gltf_path}")

        bridge.set_camera_angles(position=(0.0, 0.0, 6.0), yaw=180.0, pitch=0.0, fov=45.0)
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
    parser.add_argument("--scene-gltf", type=str, default=None, help="Path to ABeautifulGame.gltf (optional override).")
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable auto-download from KhronosGroup/glTF-Sample-Assets.",
    )
    viewer, args = newton.examples.init(parser=parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
