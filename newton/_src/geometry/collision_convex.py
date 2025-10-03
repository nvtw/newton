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

"""
High-level collision detection functions for convex shapes.

This module provides the main entry points for collision detection between convex shapes,
combining GJK, MPR, and multi-contact manifold generation into easy-to-use functions.

Two main collision modes are provided:
1. Single contact: Returns one contact point with penetration depth and normal
2. Multi-contact: Returns up to 4 contact points for stable physics simulation

The implementation uses a hybrid approach:
- GJK for fast separation tests (when shapes don't overlap)
- MPR for accurate penetration depth and contact points (when shapes overlap)
- Perturbed support mapping + polygon clipping for multi-contact manifolds

All functions are created via factory pattern to bind a specific support mapping function,
allowing the same collision pipeline to work with any convex shape type.
"""

from typing import Any

import warp as wp

from .mpr import create_solve_mpr
from .multicontact import create_build_manifold
from .simplex_solver import create_solve_closest_distance


_mat43f = wp.types.matrix((4, 3), wp.float32)


def create_solve_convex_multi_contact(support_func: Any):
    """
    Factory function to create a multi-contact collision solver for convex shapes.

    This function creates a collision detector that generates up to 4 contact points
    for stable physics simulation. It combines GJK, MPR, and manifold generation:
    1. MPR for initial collision detection and penetration (fast for overlapping shapes)
    2. GJK as fallback for separated shapes
    3. Multi-contact manifold generation for stable contact resolution

    Args:
        support_func: Support mapping function for shapes that takes
                     (geometry, direction, data_provider) and returns (point, feature_id)

    Returns:
        solve_convex_multi_contact function that computes up to 4 contact points.
    """

    @wp.func
    def solve_convex_multi_contact(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
        contact_threshold: float = 0.0,
        skip_multi_contact: bool = False,
    ) -> tuple[
        int,
        wp.vec3,
        wp.vec4,
        wp.types.matrix((4, 3), wp.float32),
        wp.vec4i,
    ]:
        """
        Compute up to 4 contact points between two convex shapes.

        This function generates a multi-contact manifold for stable contact resolution:
        1. Runs MPR first (fast for overlapping shapes, which is the common case)
        2. Falls back to GJK if MPR detects no collision
        3. Generates multi-contact manifold via perturbed support mapping + polygon clipping

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_a: Orientation quaternion of shape A
            orientation_b: Orientation quaternion of shape B
            position_a: World position of shape A
            position_b: World position of shape B
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider
            contact_threshold: Penetration threshold; skip manifold if penetration > threshold (default: 0.0)
            skip_multi_contact: If True, return only single contact point (default: False)

        Returns:
            Tuple of:
                count (int): Number of valid contact points (0-4)
                normal (wp.vec3): Contact normal from A to B (same for all contacts)
                penetrations (wp.vec4): Penetration depths for each contact (negative when overlapping)
                points (matrix(4,3)): Contact points in world space (midpoint between shapes)
                features (wp.vec4i): Feature IDs for contact tracking
        """
        # Try MPR first (optimized for overlapping shapes, which is the common case)
        collision, penetration, point, normal, feature_a_id, feature_b_id = wp.static(create_solve_mpr(support_func))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            sum_of_contact_offsets,
            data_provider,
        )

        if not collision:
            # MPR reported no collision, fall back to GJK for separated shapes
            collision, penetration, point, normal, feature_a_id, feature_b_id = wp.static(
                create_solve_closest_distance(support_func)
            )(
                geom_a,
                geom_b,
                orientation_a,
                orientation_b,
                position_a,
                position_b,
                sum_of_contact_offsets,
                data_provider,
            )

        # Skip multi-contact manifold generation if requested or penetration exceeds threshold
        if skip_multi_contact or penetration > contact_threshold:
            count = 1
            penetrations = wp.vec4(penetration, 0.0, 0.0, 0.0)
            points = _mat43f()
            points[0] = point
            features = wp.vec4i(0)
            return count, normal, penetrations, points, features

        # Generate multi-contact manifold using perturbed support mapping and polygon clipping
        count, penetrations, points, features = wp.static(create_build_manifold(support_func))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            point - normal * (penetration * 0.5),  # Anchor point on shape A
            point + normal * (penetration * 0.5),  # Anchor point on shape B
            normal,
            feature_a_id,
            feature_b_id,
            data_provider,
        )

        return count, normal, penetrations, points, features

    return solve_convex_multi_contact
