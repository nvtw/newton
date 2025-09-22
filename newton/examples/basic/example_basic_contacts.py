# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.collision_convex import create_solve_convex_contact
from newton._src.geometry.support_function import (
    GenericShapeData,
    GeoType,  # re-exported through newton too
    SupportMapDataProvider,
    support_map as support_map_func,
)


class Example:
    def __init__(self, viewer):
        # setup timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.time = 0.0

        self.viewer = viewer

        # Colors
        self.col_box = wp.array([wp.vec3(0.2, 0.6, 0.9)], dtype=wp.vec3)
        self.col_capsule = wp.array([wp.vec3(0.9, 0.6, 0.2)], dtype=wp.vec3)
        self.col_points = wp.array([wp.vec3(0.2, 1.0, 0.2), wp.vec3(1.0, 0.2, 0.2)], dtype=wp.vec3)

        # Geometry params (local space)
        # Box half extents
        self.box_scale = wp.vec3(0.4, 0.4, 0.4)
        # Capsule: radius in x, half-height in y (axis +Z)
        self.capsule_scale = wp.vec3(0.25, 0.6, 0.0)

        # Shapes
        self.geom_box = GenericShapeData()
        self.geom_box.shape_type = int(GeoType.BOX)
        self.geom_box.scale = self.box_scale

        self.geom_capsule = GenericShapeData()
        self.geom_capsule.shape_type = int(GeoType.CAPSULE)
        self.geom_capsule.scale = self.capsule_scale

        # Contact buffers
        self.contact_valid = wp.array([False], dtype=wp.bool_)
        self.point_a = wp.array([wp.vec3()], dtype=wp.vec3)
        self.point_b = wp.array([wp.vec3()], dtype=wp.vec3)
        self.normal = wp.array([wp.vec3()], dtype=wp.vec3)
        self.penetration = wp.array([0.0], dtype=float)
        self.feature_a = wp.array([0], dtype=int)
        self.feature_b = wp.array([0], dtype=int)

        # Transform arrays for rendering
        self.x_box = wp.array([wp.transform_identity()], dtype=wp.transform)
        self.x_capsule = wp.array([wp.transform_identity()], dtype=wp.transform)

        # Colors per-instance
        self.box_colors = self.col_box
        self.capsule_colors = self.col_capsule

        # Initialize solver factory for convex contact (gjk then mpr)
        self.solve_convex = create_solve_convex_contact(support_map_func, lambda g, dp: wp.vec3(0.0, 0.0, 0.0))

        # Simple data provider placeholder
        self.data_provider = SupportMapDataProvider()

        # Offsets to keep shapes very close but not overlapping initially
        self.base_pos_box = wp.vec3(-0.6, 0.0, 0.0)
        self.base_pos_capsule = wp.vec3(0.6, 0.0, 0.0)

        # Render once to set up viewer
        self.render()

    def _anim_transforms(self, t: float):
        # Rotate around Z at different speeds, keep near each other
        angle_box = 0.7 * t
        angle_capsule = -0.9 * t

        q_box = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_box)
        q_capsule = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_capsule)

        # Slight breathing along X to vary distance but keep it near-touching
        dx = 0.05 * np.sin(1.5 * t)
        p_box = self.base_pos_box + wp.vec3(dx, 0.0, 0.0)
        p_capsule = self.base_pos_capsule - wp.vec3(dx, 0.0, 0.0)

        self.x_box[0] = wp.transform(p_box, q_box)
        self.x_capsule[0] = wp.transform(p_capsule, q_capsule)

    @wp.kernel
    def _compute_contact_kernel(
        geom_a: GenericShapeData,
        geom_b: GenericShapeData,
        xform_a: wp.transform,
        xform_b: wp.transform,
        sum_of_contact_offsets: float,
        data_provider: SupportMapDataProvider,
        valid_out: wp.array(dtype=wp.bool_),
        point_a_out: wp.array(dtype=wp.vec3),
        point_b_out: wp.array(dtype=wp.vec3),
        normal_out: wp.array(dtype=wp.vec3),
        penetration_out: wp.array(dtype=float),
        feature_a_out: wp.array(dtype=int),
        feature_b_out: wp.array(dtype=int),
    ):
        collide = create_solve_convex_contact(support_map_func, lambda g, dp: wp.vec3(0.0, 0.0, 0.0))

        result, pa, pb, n, pen, fa, fb = collide(
            geom_a,
            geom_b,
            wp.transform_get_rotation(xform_a),
            wp.transform_get_rotation(xform_b),
            wp.transform_get_translation(xform_a),
            wp.transform_get_translation(xform_b),
            sum_of_contact_offsets,
            data_provider,
        )

        valid_out[0] = result
        point_a_out[0] = pa
        point_b_out[0] = pb
        normal_out[0] = n
        penetration_out[0] = pen
        feature_a_out[0] = fa
        feature_b_out[0] = fb

    def step(self):
        # advance time and update transforms
        self.time += self.frame_dt
        self._anim_transforms(self.time)

        # Run kernel to compute contact
        wp.launch(
            kernel=self._compute_contact_kernel,
            dim=1,
            inputs=[
                self.geom_box,
                self.geom_capsule,
                self.x_box[0],
                self.x_capsule[0],
                0.0,  # no contact offset
                self.data_provider,
                self.contact_valid,
                self.point_a,
                self.point_b,
                self.normal,
                self.penetration,
                self.feature_a,
                self.feature_b,
            ],
            device=wp.get_device(),
        )

        # Render
        self.render()

    def render(self):
        self.viewer.begin_frame(self.time)

        # Draw shapes
        self.viewer.log_shapes(
            "/box",
            newton.GeoType.BOX,
            (self.box_scale[0], self.box_scale[1], self.box_scale[2]),
            self.x_box,
            self.box_colors,
            None,
        )
        self.viewer.log_shapes(
            "/capsule",
            newton.GeoType.CAPSULE,
            (self.capsule_scale[0], self.capsule_scale[1]),
            self.x_capsule,
            self.capsule_colors,
            None,
        )

        # Draw contact points and connecting line when valid
        if self.contact_valid.numpy()[0]:
            pa = self.point_a.numpy()[0]
            pb = self.point_b.numpy()[0]
            pts = wp.array([pa, pb], dtype=wp.vec3)
            radii = wp.array([0.04, 0.04], dtype=float)
            self.viewer.log_points("/contact_pts", pts, radii, self.col_points)
            self.viewer.log_lines("/contact_line", wp.array([pa], dtype=wp.vec3), wp.array([pb], dtype=wp.vec3), (1.0, 0.0, 0.0))
        else:
            # Clear by logging empty arrays
            self.viewer.log_points("/contact_pts", wp.array([], dtype=wp.vec3), wp.array([], dtype=float), wp.array([], dtype=wp.vec3))
            self.viewer.log_lines("/contact_line", wp.array([], dtype=wp.vec3), wp.array([], dtype=wp.vec3), (1.0, 0.0, 0.0))

        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)