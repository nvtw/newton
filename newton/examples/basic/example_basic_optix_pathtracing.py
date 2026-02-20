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

        self.cube_xforms, self.cube_colors, self.cube_materials = _build_scene_instances()
        self.plane_xforms = wp.array([wp.transform_identity()], dtype=wp.transform)
        self.plane_colors = wp.array([wp.vec3(220.0 / 255.0, 220.0 / 255.0, 220.0 / 255.0)], dtype=wp.vec3)
        self.plane_materials = wp.array([wp.vec4(0.7, 0.0, 0.0, 0.0)], dtype=wp.vec4)

    def step(self):
        self.frame += 1

    def render(self):
        self.viewer.begin_frame(self.frame / 60.0)

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


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="optix")
    viewer, args = newton.examples.init(parser=parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
