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
    center_func as center_map,
)


class ShapePair:
    """Represents a pair of shapes for contact testing"""

    def __init__(
        self,
        shape_a: GenericShapeData,
        shape_b: GenericShapeData,
        name_a: str,
        name_b: str,
        color_a: wp.vec3,
        color_b: wp.vec3,
    ):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.name_a = name_a
        self.name_b = name_b
        self.color_a = color_a
        self.color_b = color_b
        
        # Contact results
        self.contact_valid = False
        self.point_a = wp.vec3(0.0, 0.0, 0.0)
        self.point_b = wp.vec3(0.0, 0.0, 0.0)
        self.normal = wp.vec3(0.0, 0.0, 0.0)
        self.penetration = 0.0
        self.feature_a = 0
        self.feature_b = 0

        # Current transforms
        self.transform_a = wp.transform_identity()
        self.transform_b = wp.transform_identity()


class Example:
    def __init__(self, viewer):
        # setup timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.time = 0.0

        self.viewer = viewer

        # Initialize solver factory for convex contact (gjk then mpr)
        self.solve_convex = create_solve_convex_contact(support_map_func, center_map)
        self.data_provider = SupportMapDataProvider()

        # Create all shape types and their combinations (only viewer-supported types)
        self.shape_types = [GeoType.BOX, GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE]
        self.shape_names = ["Box", "Sphere", "Capsule", "Cylinder", "Cone"]

        # Create shape pairs matrix
        self.shape_pairs: list[list[ShapePair]] = []
        self._create_shape_matrix()

        # Grid layout parameters
        self.grid_spacing = 4.0
        self.pair_spacing = 1.2  # Distance between shapes in each pair

        # Contact buffers for batch processing
        num_pairs = len(self.shape_types) * len(self.shape_types)
        self.contact_valid = wp.array([False] * num_pairs, dtype=wp.bool)
        self.point_a = wp.array([wp.vec3()] * num_pairs, dtype=wp.vec3)
        self.point_b = wp.array([wp.vec3()] * num_pairs, dtype=wp.vec3)
        self.normal = wp.array([wp.vec3()] * num_pairs, dtype=wp.vec3)
        self.penetration = wp.array([0.0] * num_pairs, dtype=float)
        self.feature_a = wp.array([0] * num_pairs, dtype=int)
        self.feature_b = wp.array([0] * num_pairs, dtype=int)

        # Render once to set up viewer
        self.render()

    def _create_shape(self, shape_type: GeoType) -> GenericShapeData:
        """Create a shape with consistent sizing"""
        shape = GenericShapeData()
        shape.shape_type = int(shape_type)

        # Consistent sizing for all shapes to have similar volumes
        if shape_type == GeoType.BOX:
            shape.scale = wp.vec3(0.3, 0.3, 0.3)  # half extents
        elif shape_type == GeoType.SPHERE:
            shape.scale = wp.vec3(0.35, 0.0, 0.0)  # radius in x
        elif shape_type == GeoType.CAPSULE:
            shape.scale = wp.vec3(0.2, 0.4, 0.0)  # radius in x, half-height in y
        elif shape_type == GeoType.CYLINDER:
            shape.scale = wp.vec3(0.25, 0.4, 0.0)  # radius in x, half-height in y
        elif shape_type == GeoType.CONE:
            shape.scale = wp.vec3(0.3, 0.4, 0.0)  # radius in x, half-height in y
        else:
            shape.scale = wp.vec3(0.3, 0.3, 0.3)  # default

        return shape

    def _get_shape_color(self, shape_type: GeoType, is_first: bool) -> wp.vec3:
        """Get distinct colors for each shape type"""
        colors = {
            GeoType.BOX: (wp.vec3(0.2, 0.6, 0.9), wp.vec3(0.1, 0.4, 0.7)),
            GeoType.SPHERE: (wp.vec3(0.9, 0.2, 0.2), wp.vec3(0.7, 0.1, 0.1)),
            GeoType.CAPSULE: (wp.vec3(0.2, 0.9, 0.2), wp.vec3(0.1, 0.7, 0.1)),
            GeoType.CYLINDER: (wp.vec3(0.9, 0.2, 0.9), wp.vec3(0.7, 0.1, 0.7)),
            GeoType.CONE: (wp.vec3(0.2, 0.9, 0.9), wp.vec3(0.1, 0.7, 0.7)),
        }
        return colors.get(shape_type, (wp.vec3(0.5, 0.5, 0.5), wp.vec3(0.3, 0.3, 0.3)))[0 if is_first else 1]

    def _create_shape_matrix(self):
        """Create a matrix of all shape pair combinations"""
        for i, type_a in enumerate(self.shape_types):
            row = []
            for j, type_b in enumerate(self.shape_types):
                shape_a = self._create_shape(type_a)
                shape_b = self._create_shape(type_b)
                color_a = self._get_shape_color(type_a, True)
                color_b = self._get_shape_color(type_b, False)

                pair = ShapePair(shape_a, shape_b, self.shape_names[i], self.shape_names[j], color_a, color_b)
                row.append(pair)
            self.shape_pairs.append(row)

    def _get_grid_position(self, row: int, col: int) -> tuple[wp.vec3, wp.vec3]:
        """Get world positions for a shape pair in the grid"""
        # Center the grid around origin
        offset_x = (len(self.shape_types) - 1) * self.grid_spacing * 0.5
        offset_y = (len(self.shape_types) - 1) * self.grid_spacing * 0.5

        center_x = col * self.grid_spacing - offset_x
        center_y = row * self.grid_spacing - offset_y

        # Position shapes slightly apart
        pos_a = wp.vec3(center_x - self.pair_spacing * 0.5, center_y, 0.0)
        pos_b = wp.vec3(center_x + self.pair_spacing * 0.5, center_y, 0.0)

        return pos_a, pos_b

    def _animate_pair(self, pair: ShapePair, row: int, col: int, t: float):
        """Animate a specific shape pair with unique motion"""
        pos_a, pos_b = self._get_grid_position(row, col)

        # Different animation patterns for variety
        pattern = (row + col) % 4

        if pattern == 0:
            # Rotation around Z axis
            angle_a = 0.8 * t + row * 0.5
            angle_b = -0.6 * t + col * 0.3
            q_a = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_a)
            q_b = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_b)

        elif pattern == 1:
            # Rotation around X axis
            angle_a = 0.7 * t + row * 0.4
            angle_b = -0.9 * t + col * 0.2
            q_a = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle_a)
            q_b = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle_b)

        elif pattern == 2:
            # Rotation around Y axis
            angle_a = 0.6 * t + row * 0.3
            angle_b = -0.8 * t + col * 0.4
            q_a = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle_a)
            q_b = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle_b)

        else:
            # Mixed rotation
            angle_a = 0.5 * t + row * 0.2
            angle_b = -0.7 * t + col * 0.5
            axis_a = wp.normalize(wp.vec3(1.0, 1.0, 0.0))
            axis_b = wp.normalize(wp.vec3(0.0, 1.0, 1.0))
            q_a = wp.quat_from_axis_angle(axis_a, angle_a)
            q_b = wp.quat_from_axis_angle(axis_b, angle_b)

        # Add breathing motion to vary distance
        breath = 0.15 * np.sin(1.2 * t + row + col)
        pos_a_final = pos_a + wp.vec3(-breath, 0.0, 0.0)
        pos_b_final = pos_b + wp.vec3(breath, 0.0, 0.0)

        pair.transform_a = wp.transform(pos_a_final, q_a)
        pair.transform_b = wp.transform(pos_b_final, q_b)

    def _anim_transforms(self, t: float):
        """Update transforms for all shape pairs"""
        for i, row in enumerate(self.shape_pairs):
            for j, pair in enumerate(row):
                self._animate_pair(pair, i, j, t)

    @wp.kernel
    def _compute_contact_matrix_kernel(
        geom_a_array: wp.array(dtype=GenericShapeData),
        geom_b_array: wp.array(dtype=GenericShapeData),
        xform_a_array: wp.array(dtype=wp.transform),
        xform_b_array: wp.array(dtype=wp.transform),
        sum_of_contact_offsets: float,
        data_provider: SupportMapDataProvider,
        valid_out: wp.array(dtype=wp.bool),
        point_a_out: wp.array(dtype=wp.vec3),
        point_b_out: wp.array(dtype=wp.vec3),
        normal_out: wp.array(dtype=wp.vec3),
        penetration_out: wp.array(dtype=float),
        feature_a_out: wp.array(dtype=int),
        feature_b_out: wp.array(dtype=int),
    ):
        tid = wp.tid()

        geom_a = geom_a_array[tid]
        geom_b = geom_b_array[tid]
        xform_a = xform_a_array[tid]
        xform_b = xform_b_array[tid]

        result, pa, pb, n, pen, fa, fb = wp.static(create_solve_convex_contact(support_map_func, center_map))(
            geom_a,
            geom_b,
            wp.transform_get_rotation(xform_a),
            wp.transform_get_rotation(xform_b),
            wp.transform_get_translation(xform_a),
            wp.transform_get_translation(xform_b),
            sum_of_contact_offsets,
            data_provider,
        )

        valid_out[tid] = result
        point_a_out[tid] = pa
        point_b_out[tid] = pb
        normal_out[tid] = n
        penetration_out[tid] = pen
        feature_a_out[tid] = fa
        feature_b_out[tid] = fb

    def step(self):
        # advance time and update transforms
        self.time += self.frame_dt
        self._anim_transforms(self.time)

        # Prepare arrays for batch contact computation
        num_pairs = len(self.shape_types) * len(self.shape_types)
        geom_a_list = []
        geom_b_list = []
        xform_a_list = []
        xform_b_list = []

        # Flatten the matrix into arrays for the kernel
        for row in self.shape_pairs:
            for pair in row:
                geom_a_list.append(pair.shape_a)
                geom_b_list.append(pair.shape_b)
                xform_a_list.append(pair.transform_a)
                xform_b_list.append(pair.transform_b)

        # Convert to warp arrays
        geom_a_array = wp.array(geom_a_list, dtype=GenericShapeData)
        geom_b_array = wp.array(geom_b_list, dtype=GenericShapeData)
        xform_a_array = wp.array(xform_a_list, dtype=wp.transform)
        xform_b_array = wp.array(xform_b_list, dtype=wp.transform)

        # Run kernel to compute all contacts in one batch
        wp.launch(
            kernel=self._compute_contact_matrix_kernel,
            dim=num_pairs,
            inputs=[
                geom_a_array,
                geom_b_array,
                xform_a_array,
                xform_b_array,
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

        # Update shape pairs with contact results
        contact_valid_np = self.contact_valid.numpy()
        point_a_np = self.point_a.numpy()
        point_b_np = self.point_b.numpy()
        normal_np = self.normal.numpy()
        penetration_np = self.penetration.numpy()
        feature_a_np = self.feature_a.numpy()
        feature_b_np = self.feature_b.numpy()

        idx = 0
        for row in self.shape_pairs:
            for pair in row:
                pair.contact_valid = contact_valid_np[idx]
                pair.point_a = point_a_np[idx]
                pair.point_b = point_b_np[idx]
                pair.normal = normal_np[idx]
                pair.penetration = penetration_np[idx]
                pair.feature_a = feature_a_np[idx]
                pair.feature_b = feature_b_np[idx]
                idx += 1

        # Render
        self.render()

    def render(self):
        self.viewer.begin_frame(self.time)

        # Render all shape pairs in the matrix
        for i, row in enumerate(self.shape_pairs):
            for j, pair in enumerate(row):
                # Create unique names for each shape pair
                name_a = f"/shapes/row{i}_col{j}_a_{pair.name_a}"
                name_b = f"/shapes/row{i}_col{j}_b_{pair.name_b}"

                # Get shape parameters for rendering
                params_a = self._get_shape_render_params(pair.shape_a)
                params_b = self._get_shape_render_params(pair.shape_b)

                # Render shape A
                self.viewer.log_shapes(
                    name_a,
                    newton.GeoType(pair.shape_a.shape_type),
                    params_a,
                    wp.array([pair.transform_a], dtype=wp.transform),
                    wp.array([pair.color_a], dtype=wp.vec3),
                    None,
                )

                # Render shape B
                self.viewer.log_shapes(
                    name_b,
                    newton.GeoType(pair.shape_b.shape_type),
                    params_b,
                    wp.array([pair.transform_b], dtype=wp.transform),
                    wp.array([pair.color_b], dtype=wp.vec3),
                    None,
                )

                # Render contact points and lines
                self._render_contact_visualization(pair, i, j)

        self.viewer.end_frame()

    def _get_shape_render_params(self, shape: GenericShapeData) -> tuple:
        """Get rendering parameters for a shape"""
        shape_type = GeoType(shape.shape_type)

        if shape_type == GeoType.BOX:
            return (shape.scale[0], shape.scale[1], shape.scale[2])
        elif shape_type == GeoType.SPHERE:
            return (shape.scale[0],)
        elif shape_type == GeoType.CAPSULE:
            return (shape.scale[0], shape.scale[1])
        elif shape_type == GeoType.CYLINDER:
            return (shape.scale[0], shape.scale[1])
        elif shape_type == GeoType.CONE:
            return (shape.scale[0], shape.scale[1])
        else:
            return (shape.scale[0], shape.scale[1], shape.scale[2])

    def _render_contact_visualization(self, pair: ShapePair, row: int, col: int):
        """Render contact points and lines for a shape pair"""
        pa = pair.point_a
        pb = pair.point_b

        # Create contact points
        pts = wp.array([pa, pb], dtype=wp.vec3)
        radii = wp.array([0.03, 0.03], dtype=float)

        # Use different colors for actual contacts vs closest points
        if pair.contact_valid:
            # Red for actual contacts
            colors = wp.array([wp.vec3(1.0, 0.2, 0.2), wp.vec3(1.0, 0.2, 0.2)], dtype=wp.vec3)
            line_color = (1.0, 0.2, 0.2)
        else:
            # Green for closest points when not touching
            colors = wp.array([wp.vec3(0.2, 1.0, 0.2), wp.vec3(0.2, 1.0, 0.2)], dtype=wp.vec3)
            line_color = (0.2, 1.0, 0.2)

        # Render contact points
        contact_name = f"/contacts/row{row}_col{col}_pts"
        self.viewer.log_points(contact_name, pts, radii, colors)

        # Render connecting line
        line_name = f"/contacts/row{row}_col{col}_line"
        self.viewer.log_lines(line_name, wp.array([pa], dtype=wp.vec3), wp.array([pb], dtype=wp.vec3), line_color)


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)
