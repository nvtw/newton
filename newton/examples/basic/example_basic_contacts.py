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
from newton._src.geometry.collision_convex import create_solve_convex_multi_contact, create_solve_convex_contact
from newton._src.geometry.gjk_stateless import create_solve_gjk
from newton._src.geometry.support_function import (
    GenericShapeData,
    GeoType,  # re-exported through newton too
    SupportMapDataProvider,
    support_map as support_map_func,
    center_func as center_map,
)


# Centralized color table for all shapes
SHAPE_COLORS = {
    GeoType.BOX: {
        "primary": wp.vec3(0.8, 0.2, 0.0),  # Red-orange
        "secondary": wp.vec3(0.6, 0.15, 0.0),  # Dark red-orange
        "name": "Red-orange",
    },
    GeoType.SPHERE: {
        "primary": wp.vec3(1.0, 0.5, 0.0),  # Orange
        "secondary": wp.vec3(0.7, 0.35, 0.0),  # Dark orange
        "name": "Orange",
    },
    GeoType.CAPSULE: {
        "primary": wp.vec3(1.0, 1.0, 0.0),  # Yellow
        "secondary": wp.vec3(0.7, 0.7, 0.0),  # Dark yellow
        "name": "Yellow",
    },
    GeoType.CYLINDER: {
        "primary": wp.vec3(0.0, 0.7, 0.0),  # Green
        "secondary": wp.vec3(0.0, 0.5, 0.0),  # Dark green
        "name": "Green",
    },
    GeoType.CONE: {
        "primary": wp.vec3(0.0, 0.0, 1.0),  # Blue
        "secondary": wp.vec3(0.0, 0.0, 0.7),  # Dark blue
        "name": "Blue",
    },
}

# Contact visualization colors
CONTACT_COLORS = {
    "active_contact": wp.vec3(1.0, 0.2, 0.2),  # Red for actual contacts
    "closest_point": wp.vec3(0.2, 1.0, 0.2),  # Green for closest points
    "line_default": (0.5, 0.8, 0.5),  # Default line color
}


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

        # Initialize solver factories
        self.solve_convex_multi = create_solve_convex_multi_contact(support_map_func, center_map)
        self.solve_gjk = create_solve_gjk(support_map_func, center_map)
        self.data_provider = SupportMapDataProvider()

        # Create all shape types and their combinations (only viewer-supported types)
        self.shape_types = [GeoType.BOX, GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE]
        self.shape_names = [SHAPE_COLORS[shape_type]["name"] for shape_type in self.shape_types]

        # Create shape pairs matrix
        self.shape_pairs: list[list[ShapePair]] = []
        self._create_shape_matrix()

        # Grid layout parameters
        self.grid_spacing = 4.0
        self.base_pair_spacing = 0.8  # Base distance between shapes in each pair (reduced for more contacts)

        # Contact buffers for batch processing
        self.num_pairs = len(self.shape_types) * len(self.shape_types)
        self.contact_valid = wp.array([False] * self.num_pairs, dtype=wp.bool)
        self.point_a = wp.array([wp.vec3()] * self.num_pairs, dtype=wp.vec3)
        self.point_b = wp.array([wp.vec3()] * self.num_pairs, dtype=wp.vec3)
        self.normal = wp.array([wp.vec3()] * self.num_pairs, dtype=wp.vec3)
        self.penetration = wp.array([0.0] * self.num_pairs, dtype=float)
        self.feature_a = wp.array([0] * self.num_pairs, dtype=int)
        self.feature_b = wp.array([0] * self.num_pairs, dtype=int)

        # Pre-allocate arrays for geometry and transforms (reused each frame)
        # Initialize geometry arrays once since geometry doesn't change
        geom_a_list = []
        geom_b_list = []
        for row in self.shape_pairs:
            for pair in row:
                geom_a_list.append(pair.shape_a)
                geom_b_list.append(pair.shape_b)

        self.geom_a_array = wp.array(geom_a_list, dtype=GenericShapeData)
        self.geom_b_array = wp.array(geom_b_list, dtype=GenericShapeData)
        self.xform_a_array = wp.array([wp.transform_identity()] * self.num_pairs, dtype=wp.transform)
        self.xform_b_array = wp.array([wp.transform_identity()] * self.num_pairs, dtype=wp.transform)

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
        """Get distinct colors for each shape type from centralized color table"""
        color_info = SHAPE_COLORS.get(
            shape_type, {"primary": wp.vec3(0.5, 0.5, 0.5), "secondary": wp.vec3(0.3, 0.3, 0.3)}
        )
        return color_info["primary"] if is_first else color_info["secondary"]

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

        # Position shapes slightly apart (base positions)
        pos_a = wp.vec3(center_x - self.base_pair_spacing * 0.5, center_y, 0.0)
        pos_b = wp.vec3(center_x + self.base_pair_spacing * 0.5, center_y, 0.0)

        return pos_a, pos_b

    def _animate_pair(self, pair: ShapePair, row: int, col: int, t: float):
        """Animate a specific shape pair with unique motion and slowly changing rotation axes"""
        pos_a, pos_b = self._get_grid_position(row, col)

        # Different animation patterns for variety
        pattern = (row + col) % 4

        # Slowly evolving rotation axes - change over ~20 second cycles
        axis_evolution_speed = 0.1
        axis_phase_a = axis_evolution_speed * t + row * 1.5
        axis_phase_b = axis_evolution_speed * t + col * 2.0

        if pattern == 0:
            # Evolving between Z and XY plane rotations
            base_axis_a = wp.vec3(0.0, 0.0, 1.0)
            evolve_axis_a = wp.vec3(np.sin(axis_phase_a), np.cos(axis_phase_a), 0.5)
            axis_a = wp.normalize(base_axis_a + 0.3 * evolve_axis_a)

            base_axis_b = wp.vec3(0.0, 0.0, 1.0)
            evolve_axis_b = wp.vec3(np.cos(axis_phase_b), np.sin(axis_phase_b), 0.3)
            axis_b = wp.normalize(base_axis_b + 0.4 * evolve_axis_b)

            angle_a = 0.8 * t + row * 0.5
            angle_b = -0.6 * t + col * 0.3

        elif pattern == 1:
            # Evolving between X and YZ plane rotations
            base_axis_a = wp.vec3(1.0, 0.0, 0.0)
            evolve_axis_a = wp.vec3(0.2, np.sin(axis_phase_a), np.cos(axis_phase_a))
            axis_a = wp.normalize(base_axis_a + 0.5 * evolve_axis_a)

            base_axis_b = wp.vec3(1.0, 0.0, 0.0)
            evolve_axis_b = wp.vec3(0.3, np.cos(axis_phase_b), np.sin(axis_phase_b))
            axis_b = wp.normalize(base_axis_b + 0.4 * evolve_axis_b)

            angle_a = 0.7 * t + row * 0.4
            angle_b = -0.9 * t + col * 0.2

        elif pattern == 2:
            # Evolving between Y and XZ plane rotations
            base_axis_a = wp.vec3(0.0, 1.0, 0.0)
            evolve_axis_a = wp.vec3(np.sin(axis_phase_a), 0.3, np.cos(axis_phase_a))
            axis_a = wp.normalize(base_axis_a + 0.6 * evolve_axis_a)

            base_axis_b = wp.vec3(0.0, 1.0, 0.0)
            evolve_axis_b = wp.vec3(np.cos(axis_phase_b), 0.2, np.sin(axis_phase_b))
            axis_b = wp.normalize(base_axis_b + 0.5 * evolve_axis_b)

            angle_a = 0.6 * t + row * 0.3
            angle_b = -0.8 * t + col * 0.4

        else:
            # Complex evolving axes with all three components changing
            axis_a = wp.normalize(
                wp.vec3(
                    1.0 + 0.5 * np.sin(axis_phase_a),
                    1.0 + 0.4 * np.cos(axis_phase_a * 1.3),
                    0.3 * np.sin(axis_phase_a * 0.7),
                )
            )
            axis_b = wp.normalize(
                wp.vec3(
                    0.2 * np.cos(axis_phase_b * 1.1),
                    1.0 + 0.6 * np.sin(axis_phase_b),
                    1.0 + 0.3 * np.cos(axis_phase_b * 0.9),
                )
            )

            angle_a = 0.5 * t + row * 0.2
            angle_b = -0.7 * t + col * 0.5

        # Create rotations with the evolving axes
        q_a = wp.quat_from_axis_angle(axis_a, angle_a)
        q_b = wp.quat_from_axis_angle(axis_b, angle_b)

        # Enhanced motion for more intersections
        # 1. Aggressive breathing motion that brings shapes together
        breath_freq = 0.8 + 0.1 * (row + col)  # Slower frequency for more sustained contact
        breath_amplitude = 0.25  # Increased amplitude
        breath = breath_amplitude * np.sin(breath_freq * t + row + col)

        # 2. Periodic approach cycles - shapes get very close every few seconds
        approach_cycle_freq = 0.3 + 0.05 * (row + col)  # ~3-4 second cycles
        approach_phase = approach_cycle_freq * t + (row * 2.1 + col * 1.7)
        # Use a sharper function to create periods of close approach
        approach_factor = 0.4 * (1.0 + np.cos(approach_phase)) * np.exp(-0.5 * (np.sin(approach_phase * 0.5)) ** 2)

        # 3. Orbital motion - shapes orbit around their center point
        orbital_freq = 0.5 + 0.1 * (row - col)
        orbital_phase_a = orbital_freq * t + row * 0.8
        orbital_phase_b = -orbital_freq * t + col * 1.2
        orbital_radius = 0.15 * (1.0 - approach_factor)  # Smaller orbits during approach

        orbital_offset_a = wp.vec3(
            orbital_radius * np.cos(orbital_phase_a),
            orbital_radius * np.sin(orbital_phase_a) * 0.7,  # Elliptical orbit
            0.0,
        )
        orbital_offset_b = wp.vec3(
            orbital_radius * np.cos(orbital_phase_b), orbital_radius * np.sin(orbital_phase_b) * 0.7, 0.0
        )

        # 4. Vertical oscillation for 3D contact scenarios
        vertical_freq = 1.1 + 0.2 * (row + col)
        vertical_motion_a = 0.08 * np.sin(vertical_freq * t + row * 0.7 + col * 1.1)
        vertical_motion_b = 0.08 * np.sin(vertical_freq * t + row * 1.3 + col * 0.9 + np.pi * 0.3)

        # Combine all motions with distance constraints
        motion_a = wp.vec3(-breath - approach_factor + orbital_offset_a[0], orbital_offset_a[1], vertical_motion_a)
        motion_b = wp.vec3(breath + approach_factor + orbital_offset_b[0], orbital_offset_b[1], vertical_motion_b)

        # Apply distance constraints to keep shapes reasonably close
        max_separation = 1.2  # Maximum allowed distance between shape centers
        min_separation = 0.05  # Minimum allowed distance to prevent excessive overlap

        pos_a_temp = pos_a + motion_a
        pos_b_temp = pos_b + motion_b

        # Calculate current separation
        separation_vec = pos_b_temp - pos_a_temp
        separation_dist = wp.length(separation_vec)

        # Constrain separation distance
        if separation_dist > max_separation:
            # Pull shapes closer together
            correction = (separation_dist - max_separation) * 0.5
            correction_vec = wp.normalize(separation_vec) * correction
            pos_a_temp = pos_a_temp + correction_vec
            pos_b_temp = pos_b_temp - correction_vec
        elif separation_dist < min_separation and separation_dist > 1e-6:
            # Push shapes slightly apart to prevent excessive overlap
            correction = (min_separation - separation_dist) * 0.5
            correction_vec = wp.normalize(separation_vec) * correction
            pos_a_temp = pos_a_temp - correction_vec
            pos_b_temp = pos_b_temp + correction_vec

        pos_a_final = pos_a_temp
        pos_b_final = pos_b_temp

        pair.transform_a = wp.transform(pos_a_final, q_a)
        pair.transform_b = wp.transform(pos_b_final, q_b)

    def _anim_transforms(self, t: float):
        """Update transforms for all shape pairs"""
        for i, row in enumerate(self.shape_pairs):
            for j, pair in enumerate(row):
                self._animate_pair(pair, i, j, t)

    @wp.kernel(enable_backward=False)
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

        # Build multi-contact manifold and take the first contact
        count, _penetrations, pts_a, pts_b, features = wp.static(
            create_solve_convex_multi_contact(support_map_func, center_map)
        )(
            geom_a,
            geom_b,
            wp.transform_get_rotation(xform_a),
            wp.transform_get_rotation(xform_b),
            wp.transform_get_translation(xform_a),
            wp.transform_get_translation(xform_b),
            sum_of_contact_offsets,
            data_provider,
        )

        # Use GJK to get collision boolean, normal, and penetration (keeps visualization unchanged)
        result, _pa_gjk, _pb_gjk, n, pen, _fa_gjk, _fb_gjk = wp.static(
            create_solve_convex_contact(support_map_func, center_map)
        )(
            geom_a,
            geom_b,
            wp.transform_get_rotation(xform_a),
            wp.transform_get_rotation(xform_b),
            wp.transform_get_translation(xform_a),
            wp.transform_get_translation(xform_b),
            sum_of_contact_offsets,
            data_provider,
        )

        # Write outputs: first manifold point when available
        # valid_out[tid] = count > 0 and result
        # pa = pts_a[0]
        # pb = pts_b[0]
        # fa = int(features[0])
        # fb = int(features[0])

        valid_out[tid] = result
        pa = _pa_gjk
        pb = _pb_gjk
        fa = _fa_gjk
        fb = _fb_gjk

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

        # Update only transform arrays (geometry is static)
        xform_a_list = []
        xform_b_list = []

        for row in self.shape_pairs:
            for pair in row:
                xform_a_list.append(pair.transform_a)
                xform_b_list.append(pair.transform_b)

        # Update transform arrays only (geometry arrays are static)
        self.xform_a_array.assign(wp.array(xform_a_list, dtype=wp.transform))
        self.xform_b_array.assign(wp.array(xform_b_list, dtype=wp.transform))

        # Run kernel to compute all contacts in one batch
        wp.launch(
            kernel=self._compute_contact_matrix_kernel,
            dim=self.num_pairs,
            inputs=[
                self.geom_a_array,
                self.geom_b_array,
                self.xform_a_array,
                self.xform_b_array,
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
            block_dim=128,
        )

        # Update shape pairs with contact results (only read from GPU once)
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

        # Batch render shapes by type for better performance
        self._render_shapes_batched()

        # Render all contact visualizations
        self._render_all_contacts()

        self.viewer.end_frame()

    def _render_shapes_batched(self):
        """Render shapes batched by type for better performance"""
        # Group shapes by type
        shape_batches = {}

        for i, row in enumerate(self.shape_pairs):
            for j, pair in enumerate(row):
                # Process shape A
                type_a = pair.shape_a.shape_type
                if type_a not in shape_batches:
                    shape_batches[type_a] = {"transforms": [], "colors": [], "params": None, "geo_type": None}

                shape_batches[type_a]["transforms"].append(pair.transform_a)
                shape_batches[type_a]["colors"].append(pair.color_a)
                if shape_batches[type_a]["params"] is None:
                    shape_batches[type_a]["params"] = self._get_shape_render_params(pair.shape_a)
                    shape_batches[type_a]["geo_type"] = newton.GeoType(pair.shape_a.shape_type)

                # Process shape B
                type_b = pair.shape_b.shape_type
                if type_b not in shape_batches:
                    shape_batches[type_b] = {"transforms": [], "colors": [], "params": None, "geo_type": None}

                shape_batches[type_b]["transforms"].append(pair.transform_b)
                shape_batches[type_b]["colors"].append(pair.color_b)
                if shape_batches[type_b]["params"] is None:
                    shape_batches[type_b]["params"] = self._get_shape_render_params(pair.shape_b)
                    shape_batches[type_b]["geo_type"] = newton.GeoType(pair.shape_b.shape_type)

        # Render each batch
        for shape_type, batch in shape_batches.items():
            if batch["transforms"]:
                name = f"/shapes/{GeoType(shape_type).name.lower()}_batch"
                self.viewer.log_shapes(
                    name,
                    batch["geo_type"],
                    batch["params"],
                    wp.array(batch["transforms"], dtype=wp.transform),
                    wp.array(batch["colors"], dtype=wp.vec3),
                    None,
                )

    def _render_all_contacts(self):
        """Render all contact points and lines efficiently"""
        contact_points = []
        contact_radii = []
        contact_colors = []
        line_starts = []
        line_ends = []

        for i, row in enumerate(self.shape_pairs):
            for j, pair in enumerate(row):
                pa = pair.point_a
                pb = pair.point_b

                # Add contact points
                contact_points.extend([pa, pb])
                contact_radii.extend([0.03, 0.03])

                # Use different colors for actual contacts vs closest points
                if pair.contact_valid:
                    # Red for actual contacts
                    contact_colors.extend([CONTACT_COLORS["active_contact"], CONTACT_COLORS["active_contact"]])
                else:
                    # Green for closest points when not touching
                    contact_colors.extend([CONTACT_COLORS["closest_point"], CONTACT_COLORS["closest_point"]])

                # Add line
                line_starts.append(pa)
                line_ends.append(pb)

        # Render all contact points in one call
        if contact_points:
            self.viewer.log_points(
                "/contacts/all_points",
                wp.array(contact_points, dtype=wp.vec3),
                wp.array(contact_radii, dtype=float),
                wp.array(contact_colors, dtype=wp.vec3),
            )

        # Render all lines in one call
        if line_starts and line_ends:
            self.viewer.log_lines(
                "/contacts/all_lines",
                wp.array(line_starts, dtype=wp.vec3),
                wp.array(line_ends, dtype=wp.vec3),
                CONTACT_COLORS["line_default"],
            )

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


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)
