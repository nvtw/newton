# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Emissive Gallery sample translated from MinimalDlssRR C# sample.

This is intended as the Python/OptiX path tracer entry point for testing.
DLSS is intentionally excluded for now, while scene content and camera framing
track the C# sample as closely as possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    # Package mode: python -m newton._src.viewer.optix.pathtracing.emissive_gallery_sample
    from .camera import Camera
    from .pathtracing_viewer import PathTracingViewer
    from .scene import Scene
except ImportError:
    # Script mode: python emissive_gallery_sample.py
    # Add repository root so absolute package imports resolve.
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    stale = [name for name in sys.modules if name == "newton" or name.startswith("newton.")]
    for name in stale:
        del sys.modules[name]
    from newton._src.viewer.optix.pathtracing.camera import Camera
    from newton._src.viewer.optix.pathtracing.pathtracing_viewer import PathTracingViewer
    from newton._src.viewer.optix.pathtracing.scene import Scene


def _create_gallery_room(scene: Scene) -> None:
    wall_mat = scene.materials.add_diffuse((0.12, 0.12, 0.14), 0.9)
    floor_mat = scene.materials.add_diffuse((0.05, 0.05, 0.06), 0.85)
    ceiling_mat = scene.materials.add_diffuse((0.08, 0.08, 0.10), 0.9)

    width = 12.0
    height = 5.0
    depth = 18.0

    scene.add_box((-width / 2, -0.1, -depth / 2), (width / 2, 0.0, depth / 2), floor_mat)
    scene.add_box((-width / 2, height, -depth / 2), (width / 2, height + 0.1, depth / 2), ceiling_mat)
    scene.add_box((-width / 2, 0.0, -depth / 2 - 0.1), (width / 2, height, -depth / 2), wall_mat)
    scene.add_box((-width / 2 - 0.1, 0.0, -depth / 2), (-width / 2, height, depth / 2), wall_mat)
    scene.add_box((width / 2, 0.0, -depth / 2), (width / 2 + 0.1, height, depth / 2), wall_mat)


def _create_emissive_lights(scene: Scene) -> None:
    red = scene.materials.add_emissive((1.0, 0.0, 0.0), 10.0)
    orange = scene.materials.add_emissive((1.0, 0.5, 0.0), 10.0)
    yellow = scene.materials.add_emissive((1.0, 1.0, 0.0), 10.0)
    green = scene.materials.add_emissive((0.0, 1.0, 0.0), 10.0)
    cyan = scene.materials.add_emissive((0.0, 1.0, 1.0), 10.0)
    blue = scene.materials.add_emissive((0.0, 0.0, 1.0), 10.0)
    violet = scene.materials.add_emissive((0.58, 0.0, 0.83), 10.0)

    scene.add_box((-5.92, 1.2, -6.5), (-5.85, 3.8, -5.0), red)
    scene.add_box((-5.92, 1.2, -4.5), (-5.85, 3.8, -3.0), orange)
    scene.add_box((-5.92, 1.2, -2.5), (-5.85, 3.8, -1.0), yellow)
    scene.add_box((-5.92, 1.2, -0.5), (-5.85, 3.8, 1.0), green)
    scene.add_box((-5.92, 1.2, 1.5), (-5.85, 3.8, 3.0), cyan)
    scene.add_box((-5.92, 1.2, 3.5), (-5.85, 3.8, 5.0), blue)
    scene.add_box((-5.92, 1.2, 5.5), (-5.85, 3.8, 7.0), violet)

    magenta = scene.materials.add_emissive((1.0, 0.0, 1.0), 10.0)
    indigo = scene.materials.add_emissive((0.29, 0.0, 0.51), 12.0)
    teal = scene.materials.add_emissive((0.0, 0.8, 0.8), 10.0)
    lime = scene.materials.add_emissive((0.5, 1.0, 0.0), 10.0)
    amber = scene.materials.add_emissive((1.0, 0.75, 0.0), 10.0)
    scarlet = scene.materials.add_emissive((1.0, 0.14, 0.0), 10.0)

    scene.add_box((5.85, 1.2, -6.5), (5.92, 3.8, -5.0), magenta)
    scene.add_box((5.85, 1.2, -4.5), (5.92, 3.8, -3.0), indigo)
    scene.add_box((5.85, 1.2, -2.5), (5.92, 3.8, -1.0), blue)
    scene.add_box((5.85, 1.2, -0.5), (5.92, 3.8, 1.0), teal)
    scene.add_box((5.85, 1.2, 1.5), (5.92, 3.8, 3.0), lime)
    scene.add_box((5.85, 1.2, 3.5), (5.92, 3.8, 5.0), amber)
    scene.add_box((5.85, 1.2, 5.5), (5.92, 3.8, 7.0), scarlet)

    hot_pink = scene.materials.add_emissive((1.0, 0.0, 0.5), 15.0)
    electric_blue = scene.materials.add_emissive((0.0, 0.5, 1.0), 15.0)
    neon_green = scene.materials.add_emissive((0.0, 1.0, 0.25), 15.0)
    scene.add_sphere((0.0, 3.8, -5.5), 0.5, 32, hot_pink)
    scene.add_sphere((0.0, 3.8, 0.0), 0.5, 32, electric_blue)
    scene.add_sphere((0.0, 3.8, 5.5), 0.5, 32, neon_green)

    scene.add_sphere((-3.0, 2.2, -4.0), 0.3, 24, orange)
    scene.add_sphere((3.0, 2.2, -4.0), 0.3, 24, cyan)
    scene.add_sphere((-3.0, 2.2, 4.0), 0.3, 24, violet)
    scene.add_sphere((3.0, 2.2, 4.0), 0.3, 24, yellow)

    floor_red = scene.materials.add_emissive((1.0, 0.0, 0.1), 6.0)
    floor_blue = scene.materials.add_emissive((0.1, 0.0, 1.0), 6.0)
    scene.add_box((-5.7, 0.02, -7.5), (-5.5, 0.08, 7.5), floor_red)
    scene.add_box((5.5, 0.02, -7.5), (5.7, 0.08, 7.5), floor_blue)

    back_z = -7.9
    arch_red = scene.materials.add_emissive((1.0, 0.0, 0.0), 12.0)
    arch_orange = scene.materials.add_emissive((1.0, 0.4, 0.0), 12.0)
    arch_yellow = scene.materials.add_emissive((1.0, 0.9, 0.0), 12.0)
    arch_green = scene.materials.add_emissive((0.2, 1.0, 0.0), 12.0)
    arch_cyan = scene.materials.add_emissive((0.0, 1.0, 0.8), 12.0)
    arch_blue = scene.materials.add_emissive((0.0, 0.3, 1.0), 12.0)
    arch_violet = scene.materials.add_emissive((0.6, 0.0, 1.0), 12.0)

    scene.add_box((-4.0, 0.5, back_z), (-3.2, 4.0, back_z + 0.08), arch_red)
    scene.add_box((-2.8, 0.5, back_z), (-2.0, 4.3, back_z + 0.08), arch_orange)
    scene.add_box((-1.6, 0.5, back_z), (-0.8, 4.5, back_z + 0.08), arch_yellow)
    scene.add_box((-0.4, 0.5, back_z), (0.4, 4.6, back_z + 0.08), arch_green)
    scene.add_box((0.8, 0.5, back_z), (1.6, 4.5, back_z + 0.08), arch_cyan)
    scene.add_box((2.0, 0.5, back_z), (2.8, 4.3, back_z + 0.08), arch_blue)
    scene.add_box((3.2, 0.5, back_z), (4.0, 4.0, back_z + 0.08), arch_violet)


def _create_display_objects(scene: Scene) -> None:
    pedestal = scene.materials.add_pbr(base_color=(0.02, 0.02, 0.02), roughness=0.05, metallic=0.3)
    for z in (-5.0, -2.0, 1.0, 4.0):
        scene.add_box((-0.5, 0.0, z - 0.5), (0.5, 0.9, z + 0.5), pedestal)

    chrome = scene.materials.add_metal((0.98, 0.98, 0.98), 0.01)
    scene.add_sphere((0.0, 1.6, -5.0), 0.65, 48, chrome)
    scene.add_sphere((0.0, 1.6, -2.0), 0.65, 48, chrome)
    scene.add_sphere((0.0, 1.6, 1.0), 0.65, 48, chrome)
    scene.add_sphere((0.0, 1.6, 4.0), 0.65, 48, chrome)

    polished_chrome = scene.materials.add_metal((0.95, 0.95, 0.95), 0.03)
    for x, z in [(-4.0, -5.0), (4.0, -5.0), (-4.0, 0.0), (4.0, 0.0), (-4.0, 5.0), (4.0, 5.0)]:
        scene.add_sphere((x, 0.5, z), 0.5, 32, polished_chrome)

    pure_white_diffuse = scene.materials.add_diffuse((1.0, 1.0, 1.0), 0.95)
    for x, z in [(-2.5, -3.5), (2.5, -3.5), (-2.5, 2.5), (2.5, 2.5), (-2.5, 6.0), (2.5, 6.0)]:
        scene.add_sphere((x, 0.35, z), 0.35, 32, pure_white_diffuse)

    white_matte = scene.materials.add_diffuse((0.95, 0.95, 0.95), 0.9)
    scene.add_box((-4.2, 0.0, -2.0), (-3.4, 0.8, -1.2), white_matte)
    scene.add_box((3.4, 0.0, -2.0), (4.2, 0.8, -1.2), white_matte)
    scene.add_box((-4.2, 0.0, 2.5), (-3.4, 0.8, 3.3), white_matte)
    scene.add_box((3.4, 0.0, 2.5), (4.2, 0.8, 3.3), white_matte)

    mirror_floor = scene.materials.add_pbr(base_color=(0.01, 0.01, 0.01), roughness=0.02, metallic=0.4)
    scene.add_box((-5.0, 0.002, -7.0), (5.0, 0.015, 7.0), mirror_floor)

    glass = scene.materials.add_glass(ior=1.5, tint=(1.0, 1.0, 1.0), transmission=0.95)
    scene.add_sphere((0.0, 0.7, -0.5), 0.7, 48, glass)


def build_emissive_gallery(scene: Scene) -> None:
    """Build the translated Emissive Gallery scene into ``scene``."""
    scene.clear()
    _create_gallery_room(scene)
    _create_emissive_lights(scene)
    _create_display_objects(scene)


def create_emissive_gallery_viewer(width: int = 1280, height: int = 720) -> PathTracingViewer:
    """Create a viewer configured like the C# EmissiveGallerySample."""
    camera = Camera(
        position=(0.0, 2.8, 12.0),
        target=(0.0, 1.2, -1.0),
        up=(0.0, 1.0, 0.0),
        fov=55.0,
        aspect_ratio=width / height,
    )
    viewer = PathTracingViewer(
        width=width,
        height=height,
        scene_setup=build_emissive_gallery,
        camera=camera,
        accumulate_samples=False,
        # Realtime-friendly defaults; quality can be increased explicitly when needed.
        samples_per_frame=1,
        max_bounces=4,
        direct_light_samples=1,
        use_halton_jitter=True,
    )
    # Match C# EmissiveGallery dark procedural sky setup.
    viewer.sky_rgb_unit_conversion = (1.0 / 80000.0, 1.0 / 80000.0, 1.0 / 80000.0)
    viewer.sky_multiplier = 0.001
    viewer.sky_haze = 0.0
    viewer.sky_redblueshift = 0.0
    viewer.sky_saturation = 0.0
    viewer.sky_horizon_height = 0.0
    viewer.sky_ground_color = (0.005, 0.005, 0.005)
    viewer.sky_horizon_blur = 1.0
    viewer.sky_night_color = (0.002, 0.003, 0.005)
    viewer.sky_sun_disk_intensity = 0.0
    viewer.sky_sun_direction = (0.0, -1.0, 0.0)
    viewer.sky_sun_disk_scale = 0.0
    viewer.sky_sun_glow_intensity = 0.0
    viewer.sky_y_is_up = 1
    return viewer


def main() -> int:
    print("=" * 60)
    print("Emissive Gallery (Python/OptiX, DLSS disabled)")
    print("=" * 60)

    viewer = create_emissive_gallery_viewer(width=1280, height=720)
    if not viewer.build():
        print("Failed to build emissive gallery viewer")
        return 1

    print("Rendering 16 warm-up frames...")
    for i in range(16):
        viewer.render()
        if (i + 1) % 4 == 0:
            print(f"  Frame {i + 1}/16")

    output = viewer.get_output()
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{float(np.min(output)):.4f}, {float(np.max(output)):.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
