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
Stateless GJK (Gilbert-Johnson-Keerthi) collision detection algorithm.

This module implements a stateless version of the GJK algorithm for collision detection
between convex shapes using Warp kernels.
"""

from typing import Any

import warp as wp

from .mpr import Vert, vert_v
from .support_function import GenericShapeData, SupportMapDataProvider


@wp.struct
class SupportPoint:
    """
    A support point in the Minkowski difference with feature tracking.

    This struct stores a 3D point along with feature information from both shapes
    used for contact manifold generation and feature tracking.
    """

    point: wp.vec3
    feature_a_id: int
    feature_b_id: int


# Constants
MAX_SIMPLEX_POINTS = 4


@wp.struct
class GJKSimplex:
    """
    A simplex for the GJK algorithm.

    This struct stores up to 4 support points and provides methods for
    simplex operations during GJK collision detection.
    """

    point0: SupportPoint
    point1: SupportPoint
    point2: SupportPoint
    point3: SupportPoint
    count: int


@wp.func
def gjk_simplex_clear(simplex: GJKSimplex) -> GJKSimplex:
    """Clear the simplex."""
    result = simplex
    result.count = 0
    return result


@wp.func
def gjk_simplex_add_support_point(simplex: GJKSimplex, support_point: SupportPoint) -> GJKSimplex:
    """
    Add a support point to the simplex.

    Insert at beginning (like the reference implementation) and shift existing points.
    """
    result = simplex

    # Limit count to avoid overflow
    if result.count >= MAX_SIMPLEX_POINTS:
        result.count = MAX_SIMPLEX_POINTS - 1

    # Shift existing points
    result.point3 = result.point2
    result.point2 = result.point1
    result.point1 = result.point0
    result.point0 = support_point
    result.count += 1

    return result


@wp.func
def gjk_simplex_get_point(simplex: GJKSimplex, index: int) -> SupportPoint:
    """Get a point from the simplex by index."""
    if index == 0:
        return simplex.point0
    elif index == 1:
        return simplex.point1
    elif index == 2:
        return simplex.point2
    elif index == 3:
        return simplex.point3
    else:
        # Return default SupportPoint
        result = SupportPoint()
        result.point = wp.vec3(0.0, 0.0, 0.0)
        result.feature_a_id = -1
        result.feature_b_id = -1
        return result


@wp.func
def gjk_simplex_set_points_2(simplex: GJKSimplex, a: SupportPoint, b: SupportPoint) -> GJKSimplex:
    """Set simplex to contain two points."""
    result = simplex
    result.count = 2
    result.point0 = a
    result.point1 = b
    return result


@wp.func
def gjk_simplex_set_points_3(simplex: GJKSimplex, a: SupportPoint, b: SupportPoint, c: SupportPoint) -> GJKSimplex:
    """Set simplex to contain three points."""
    result = simplex
    result.count = 3
    result.point0 = a
    result.point1 = b
    result.point2 = c
    return result


@wp.func
def gjk_simplex_contains_origin_line(simplex: GJKSimplex, direction: wp.vec3) -> tuple[bool, GJKSimplex, wp.vec3]:
    """Check if line simplex contains origin and update direction."""
    a = simplex.point0
    b = simplex.point1

    ab = b.point - a.point
    ao = -a.point

    new_direction = direction
    new_simplex = simplex

    if wp.dot(ab, ao) > 0.0:
        # Triple product: (ab x ao) x ab
        ab_cross_ao = wp.cross(ab, ao)
        new_direction = wp.cross(ab_cross_ao, ab)
    else:
        new_simplex.count = 1
        new_simplex.point0 = a
        new_direction = ao

    return False, new_simplex, new_direction


@wp.func
def gjk_simplex_contains_origin_triangle(simplex: GJKSimplex, direction: wp.vec3) -> tuple[bool, GJKSimplex, wp.vec3]:
    """Check if triangle simplex contains origin and update direction."""
    a = simplex.point0
    b = simplex.point1
    c = simplex.point2

    ab = b.point - a.point
    ac = c.point - a.point
    ao = -a.point

    abc = wp.cross(ab, ac)
    new_direction = direction
    new_simplex = simplex

    abc_cross_ac = wp.cross(abc, ac)
    if wp.dot(abc_cross_ac, ao) > 0.0:
        if wp.dot(ac, ao) > 0.0:
            new_simplex = gjk_simplex_set_points_2(simplex, a, c)
            # Triple product: (ac x ao) x ac
            ac_cross_ao = wp.cross(ac, ao)
            new_direction = wp.cross(ac_cross_ao, ac)
        else:
            new_simplex = gjk_simplex_set_points_2(simplex, a, b)
            contains_origin, new_simplex, new_direction = gjk_simplex_contains_origin_line(new_simplex, direction)
            return contains_origin, new_simplex, new_direction
    else:
        ab_cross_abc = wp.cross(ab, abc)
        if wp.dot(ab_cross_abc, ao) > 0.0:
            new_simplex = gjk_simplex_set_points_2(simplex, a, b)
            contains_origin, new_simplex, new_direction = gjk_simplex_contains_origin_line(new_simplex, direction)
            return contains_origin, new_simplex, new_direction
        else:
            if wp.dot(abc, ao) > 0.0:
                new_direction = abc
            else:
                new_simplex = gjk_simplex_set_points_3(simplex, a, c, b)
                new_direction = -abc

    return False, new_simplex, new_direction


@wp.func
def gjk_simplex_contains_origin_tetrahedron(
    simplex: GJKSimplex, direction: wp.vec3
) -> tuple[bool, GJKSimplex, wp.vec3]:
    """Check if tetrahedron simplex contains origin and update direction."""
    a = simplex.point0
    b = simplex.point1
    c = simplex.point2
    d = simplex.point3

    ab = b.point - a.point
    ac = c.point - a.point
    ad = d.point - a.point
    ao = -a.point

    abc = wp.cross(ab, ac)
    acd = wp.cross(ac, ad)
    adb = wp.cross(ad, ab)

    new_direction = direction
    new_simplex = simplex

    if wp.dot(abc, ao) > 0.0:
        new_simplex = gjk_simplex_set_points_3(simplex, a, b, c)
        contains_origin, new_simplex, new_direction = gjk_simplex_contains_origin_triangle(new_simplex, direction)
        return contains_origin, new_simplex, new_direction

    if wp.dot(acd, ao) > 0.0:
        new_simplex = gjk_simplex_set_points_3(simplex, a, c, d)
        contains_origin, new_simplex, new_direction = gjk_simplex_contains_origin_triangle(new_simplex, direction)
        return contains_origin, new_simplex, new_direction

    if wp.dot(adb, ao) > 0.0:
        new_simplex = gjk_simplex_set_points_3(simplex, a, d, b)
        contains_origin, new_simplex, new_direction = gjk_simplex_contains_origin_triangle(new_simplex, direction)
        return contains_origin, new_simplex, new_direction

    return True, new_simplex, new_direction


@wp.func
def gjk_simplex_contains_origin(simplex: GJKSimplex, direction: wp.vec3) -> tuple[bool, GJKSimplex, wp.vec3]:
    """Check if simplex contains origin and update direction."""
    if simplex.count == 4:
        return gjk_simplex_contains_origin_tetrahedron(simplex, direction)
    elif simplex.count == 3:
        return gjk_simplex_contains_origin_triangle(simplex, direction)
    elif simplex.count == 2:
        return gjk_simplex_contains_origin_line(simplex, direction)

    return False, simplex, direction


# Constants for GJK algorithm
MAX_GJK_ITERATIONS = 32
GJK_EPSILON = 1e-6


def create_solve_gjk(support_func: Any, center_func: Any):
    """
    Factory function to create GJK solver with specific support and center functions.

    This follows the factory pattern used in multicontact.py for build_manifold.

    Args:
        support_func: Support mapping function for shapes
        center_func: Geometric center function for shapes

    Returns:
        GJK solver function
    """

    # Support mapping functions (replacing MinkowskiDiff struct methods)
    @wp.func
    def support_map_b(
        geom_b: GenericShapeData,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: SupportMapDataProvider,
    ) -> tuple[wp.vec3, int]:
        """
        Support mapping for shape B with transformation.

        Args:
            geom_b: Shape B geometry data
            direction: Support direction in world space
            orientation_b: Orientation of shape B
            position_b: Position of shape B
            data_provider: Support mapping data provider

        Returns:
            Tuple of (support point in world space, feature ID)
        """
        # Transform direction to local space of shape B
        tmp = wp.quat_rotate_inv(orientation_b, direction)

        # Get support point in local space
        result, feature_id = support_func(geom_b, tmp, data_provider)

        # Transform result to world space
        result = wp.quat_rotate(orientation_b, result)
        result = result + position_b

        return result, feature_id

    @wp.func
    def minkowski_support(
        geom_a: GenericShapeData,
        geom_b: GenericShapeData,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: SupportMapDataProvider,
    ) -> tuple[Vert, int, int]:
        """
        Compute support point on Minkowski difference A - B.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            direction: Support direction
            orientation_b: Orientation of shape B
            position_b: Position of shape B
            extend: Contact offset extension
            data_provider: Support mapping data provider

        Returns:
            Tuple of (Vert containing support points, feature ID A, feature ID B)
        """
        v = Vert()

        # Support point on A in positive direction
        tmp_a, feature_a_id = support_func(geom_a, direction, data_provider)
        v.A = tmp_a

        # Support point on B in negative direction
        tmp_direction = -direction
        v.B, feature_b_id = support_map_b(geom_b, tmp_direction, orientation_b, position_b, data_provider)

        # Apply contact offset extension
        d = wp.normalize(direction) * extend * 0.5
        v.A = v.A + d
        v.B = v.B - d

        return v, feature_a_id, feature_b_id

    @wp.func
    def geometric_center(
        geom_a: GenericShapeData,
        geom_b: GenericShapeData,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: SupportMapDataProvider,
    ) -> Vert:
        """
        Compute geometric center of Minkowski difference A - B.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_b: Orientation of shape B
            position_b: Position of shape B
            data_provider: Support mapping data provider

        Returns:
            Vert containing center points
        """
        center = Vert()
        center.A = center_func(geom_a, data_provider)
        center.B = center_func(geom_b, data_provider)

        # Transform center B to world space
        center.B = wp.quat_rotate(orientation_b, center.B)
        center.B = position_b + center.B

        return center

    @wp.func
    def calculate_support_point(
        geom_a: GenericShapeData,
        geom_b: GenericShapeData,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: SupportMapDataProvider,
    ) -> SupportPoint:
        """
        Calculate a support point for the GJK algorithm.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            direction: Support direction
            orientation_b: Orientation of shape B
            position_b: Position of shape B
            extend: Contact offset extension
            data_provider: Support mapping data provider

        Returns:
            SupportPoint containing the Minkowski difference point and feature IDs
        """
        support_vert, feature_a_id, feature_b_id = minkowski_support(
            geom_a, geom_b, direction, orientation_b, position_b, extend, data_provider
        )

        result = SupportPoint()
        result.point = vert_v(support_vert)  # A - B
        result.feature_a_id = feature_a_id
        result.feature_b_id = feature_b_id

        return result

    @wp.func
    def solve_gjk_internal(
        geom_a: GenericShapeData,
        geom_b: GenericShapeData,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: SupportMapDataProvider,
    ) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float, int, int]:
        """
        Internal GJK solver function.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_b: Relative orientation of shape B to shape A
            position_b: Relative position of shape B to shape A
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider

        Returns:
            Tuple of (intersection found, point A, point B, normal, penetration, feature A ID, feature B ID)
        """
        # Initialize output values
        point_a = wp.vec3(0.0, 0.0, 0.0)
        point_b = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 0.0)
        penetration = 0.0
        feature_a_id = -1
        feature_b_id = -1

        # GJK initialization
        simplex = GJKSimplex()
        simplex = gjk_simplex_clear(simplex)

        # Initial direction - from center of shape B to center of shape A
        center_a = geometric_center(geom_a, geom_b, orientation_b, position_b, data_provider)
        direction = -vert_v(center_a)

        # Ensure direction is not zero
        if wp.length_sq(direction) < GJK_EPSILON * GJK_EPSILON:
            direction = wp.vec3(1.0, 0.0, 0.0)

        # First support point
        support = calculate_support_point(
            geom_a, geom_b, direction, orientation_b, position_b, sum_of_contact_offsets, data_provider
        )
        simplex = gjk_simplex_add_support_point(simplex, support)

        direction = -support.point

        iterations = 0

        while iterations < MAX_GJK_ITERATIONS:
            support = calculate_support_point(
                geom_a, geom_b, direction, orientation_b, position_b, sum_of_contact_offsets, data_provider
            )

            # Check if we're making progress towards the origin
            if wp.dot(support.point, direction) <= 0.0:
                return False, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

            simplex = gjk_simplex_add_support_point(simplex, support)

            contains_origin, simplex, direction = gjk_simplex_contains_origin(simplex, direction)

            if contains_origin:
                # Intersection detected - for GJK only, we just return true
                # In a full implementation, EPA would be called here for penetration info

                # Set the last support point's feature IDs as output
                feature_a_id = support.feature_a_id
                feature_b_id = support.feature_b_id

                # For GJK-only implementation, we can't provide exact contact points
                # Set approximate values based on the last support point
                last_vert, feature_a_id, feature_b_id = minkowski_support(
                    geom_a, geom_b, direction, orientation_b, position_b, sum_of_contact_offsets, data_provider
                )
                point_a = last_vert.A
                point_b = last_vert.B
                normal = wp.normalize(direction)
                penetration = 0.001  # Small positive value indicating intersection

                return True, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

            iterations += 1

        return False, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

    @wp.func
    def solve_gjk(
        geom_a: GenericShapeData,
        geom_b: GenericShapeData,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: SupportMapDataProvider,
    ) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float, int, int]:
        """
        Main GJK solver function.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_a: Orientation of shape A
            orientation_b: Orientation of shape B
            position_a: Position of shape A
            position_b: Position of shape B
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider

        Returns:
            Tuple of (intersection found, point A, point B, normal, penetration, feature A ID, feature B ID)
        """
        # Transform to shape A's local coordinate system
        relative_orientation_b = wp.quat_multiply(wp.quat_inverse(orientation_a), orientation_b)
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        result, point_a, point_b, normal, penetration, feature_a_id, feature_b_id = solve_gjk_internal(
            geom_a, geom_b, relative_orientation_b, relative_position_b, sum_of_contact_offsets, data_provider
        )

        if result:
            # Transform results back to world space
            point_a = wp.quat_rotate(orientation_a, point_a) + position_a
            point_b = wp.quat_rotate(orientation_a, point_b) + position_a
            normal = wp.quat_rotate(orientation_a, normal)

        return result, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

    return solve_gjk
