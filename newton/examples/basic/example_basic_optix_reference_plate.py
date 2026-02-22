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

"""OptiX reference plate scene matching a soft gray studio look."""

from __future__ import annotations

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame = 0

        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 3.0

        if args.viewer != "optix":
            raise RuntimeError("This example is intended for the OptiX viewer (`--viewer optix`).")

        if not hasattr(self.viewer, "_ensure_api"):
            raise RuntimeError("Viewer does not expose OptiX API hooks.")

        self.viewer._ensure_api()
        api = getattr(self.viewer, "_api", None)
        if api is None:
            raise RuntimeError("ViewerOptix PathTracerAPI was not created.")

        # Procedural sky with a warm sun for directional shadows.
        # NOTE: do NOT call set_environment_color after this -- it replaces the
        # procedural sky with a flat color map, killing the sun and shadows.
        api.set_use_procedural_sky(True)
        api.set_sky_parameters(
            sun_direction=(-0.3, 0.7, 0.5),
            multiplier=0.6,
            haze=0.25,
            red_blue_shift=0.06,
            saturation=0.7,
            horizon_height=0.0,
            ground_color=(0.72, 0.72, 0.74),
            horizon_blur=0.8,
            night_color=(0.0, 0.0, 0.0),
            sun_disk_intensity=0.35,
            sun_disk_scale=1.2,
            sun_glow_intensity=0.5,
            y_is_up=1,
        )

        api.clear_scene()

        bottom_mat = api.create_pbr_material(
            color=(0.36, 0.37, 0.40),
            roughness=0.82,
            metallic=0.02,
        )
        top_mat = api.create_pbr_material(
            color=(0.12, 0.11, 0.11),
            roughness=0.18,
            metallic=0.62,
        )
        ground_mat = api.create_pbr_material(
            color=(0.86, 0.86, 0.86),
            roughness=0.9,
            metallic=0.0,
        )

        # Clean rainbow palette (no pink/rosa): red, orange, yellow, green, teal, blue.
        palette = [
            (0.85, 0.12, 0.08),
            (0.92, 0.52, 0.08),
            (0.92, 0.82, 0.15),
            (0.22, 0.72, 0.22),
            (0.12, 0.68, 0.68),
            (0.14, 0.38, 0.88),
        ]
        pyramid_mats = [api.create_pbr_material(color=c, roughness=0.45, metallic=0.05) for c in palette]

        # Ground plane - much larger than the presenter plate.
        api.add_box(min_pt=(-20.0, -0.09, -20.0), max_pt=(20.0, -0.07, 20.0), material_id=ground_mat)

        # Bottom layer (square presenter plate).
        plate_half = 1.30
        api.add_box(
            min_pt=(-plate_half, -0.07, -plate_half),
            max_pt=(plate_half, -0.02, plate_half),
            material_id=bottom_mat,
        )

        # Top layer: same square footprint, thinner, darker, metallic, top at y=0.
        api.add_box(
            min_pt=(-plate_half, -0.02, -plate_half),
            max_pt=(plate_half, 0.00, plate_half),
            material_id=top_mat,
        )

        # Colorful 3-2-1 box pyramid for quick visual checks.
        cube_half = 0.14
        cube_spacing = 2.05 * cube_half
        base_y = 0.0
        color_idx = 0
        for level, count in enumerate((3, 2, 1)):
            y0 = base_y + level * (2.0 * cube_half)
            y1 = y0 + 2.0 * cube_half
            x_start = -0.5 * (count - 1) * cube_spacing
            for i in range(count):
                x = x_start + i * cube_spacing
                z = 0.0
                mat = pyramid_mats[color_idx % len(pyramid_mats)]
                color_idx += 1
                api.add_box(
                    min_pt=(x - cube_half, y0, z - cube_half),
                    max_pt=(x + cube_half, y1, z + cube_half),
                    material_id=mat,
                )

        api.build_scene()
        api.set_camera_look_at(
            position=(0.2, 0.55, 2.95),
            target=(0.2, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=43.0,
        )

    def step(self):
        self.frame += 1

    def render(self):
        self.viewer.begin_frame(self.frame / 60.0)
        self.viewer.end_frame()

    def test_final(self):
        if self.frame <= 0:
            raise ValueError("Example did not advance any frames.")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="optix")
    viewer, args = newton.examples.init(parser=parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
