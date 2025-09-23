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

    # -----------------
    # Helper constants
    # -----------------
    FLOAT_MAX = 1.0e30
    MINVAL = 1.0e-15
    mat43 = wp.types.matrix(shape=(4, 3), dtype=float)

    # --------------
    # Helper math
    # --------------
    @wp.func
    def _linear_combine(n: int, coefs: wp.vec4, mat: mat43) -> wp.vec3:
        v = wp.vec3(0.0)
        if n == 1:
            v = coefs[0] * mat[0]
        elif n == 2:
            v = coefs[0] * mat[0] + coefs[1] * mat[1]
        elif n == 3:
            v = coefs[0] * mat[0] + coefs[1] * mat[1] + coefs[2] * mat[2]
        else:
            v = coefs[0] * mat[0] + coefs[1] * mat[1] + coefs[2] * mat[2] + coefs[3] * mat[3]
        return v

    @wp.func
    def _almost_equal(v1: wp.vec3, v2: wp.vec3) -> bool:
        return wp.abs(v1[0] - v2[0]) < MINVAL and wp.abs(v1[1] - v2[1]) < MINVAL and wp.abs(v1[2] - v2[2]) < MINVAL

    @wp.func
    def _det3(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> float:
        return wp.dot(v1, wp.cross(v2, v3))

    @wp.func
    def _same_sign(a: float, b: float) -> int:
        if a > 0 and b > 0:
            return 1
        if a < 0 and b < 0:
            return -1
        return 0

    @wp.func
    def _project_origin_line(v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
        diff = v2 - v1
        scl = -(wp.dot(v2, diff) / wp.dot(diff, diff))
        return v2 + scl * diff

    @wp.func
    def _project_origin_plane(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> tuple[wp.vec3, int]:
        z = wp.vec3(0.0)
        diff21 = v2 - v1
        diff31 = v3 - v1
        diff32 = v3 - v2

        n = wp.cross(diff32, diff21)
        nv = wp.dot(n, v2)
        nn = wp.dot(n, n)
        if nn == 0:
            return z, 1
        if nv != 0 and nn > MINVAL:
            v = (nv / nn) * n
            return v, 0

        n = wp.cross(diff21, diff31)
        nv = wp.dot(n, v1)
        nn = wp.dot(n, n)
        if nn == 0:
            return z, 1
        if nv != 0 and nn > MINVAL:
            v = (nv / nn) * n
            return v, 0

        n = wp.cross(diff31, diff32)
        nv = wp.dot(n, v3)
        nn = wp.dot(n, n)
        v = (nv / nn) * n
        return v, 0

    @wp.func
    def _S1D(s1: wp.vec3, s2: wp.vec3) -> wp.vec2:
        p_o = _project_origin_line(s1, s2)
        mu_max = 0.0
        index = 0
        for i in range(3):
            mu = s1[i] - s2[i]
            if wp.abs(mu) >= wp.abs(mu_max):
                mu_max = mu
                index = i
        C1 = p_o[index] - s2[index]
        C2 = s1[index] - p_o[index]
        if _same_sign(mu_max, C1) and _same_sign(mu_max, C2):
            return wp.vec2(C1 / mu_max, C2 / mu_max)
        return wp.vec2(0.0, 1.0)

    @wp.func
    def _S2D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3) -> wp.vec3:
        p_o, ret = _project_origin_plane(s1, s2, s3)
        if ret:
            v = _S1D(s1, s2)
            return wp.vec3(v[0], v[1], 0.0)

        M_14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1]
        M_24 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0]
        M_34 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0]

        M_max = 0.0
        s1_2D = wp.vec2(0.0)
        s2_2D = wp.vec2(0.0)
        s3_2D = wp.vec2(0.0)
        p_o_2D = wp.vec2(0.0)

        mu1 = wp.abs(M_14)
        mu2 = wp.abs(M_24)
        mu3 = wp.abs(M_34)

        if mu1 >= mu2 and mu1 >= mu3:
            M_max = M_14
            s1_2D[0] = s1[1]
            s1_2D[1] = s1[2]
            s2_2D[0] = s2[1]
            s2_2D[1] = s2[2]
            s3_2D[0] = s3[1]
            s3_2D[1] = s3[2]
            p_o_2D[0] = p_o[1]
            p_o_2D[1] = p_o[2]
        elif mu2 >= mu3:
            M_max = M_24
            s1_2D[0] = s1[0]
            s1_2D[1] = s1[2]
            s2_2D[0] = s2[0]
            s2_2D[1] = s2[2]
            s3_2D[0] = s3[0]
            s3_2D[1] = s3[2]
            p_o_2D[0] = p_o[0]
            p_o_2D[1] = p_o[2]
        else:
            M_max = M_34
            s1_2D[0] = s1[0]
            s1_2D[1] = s1[1]
            s2_2D[0] = s2[0]
            s2_2D[1] = s2[1]
            s3_2D[0] = s3[0]
            s3_2D[1] = s3[1]
            p_o_2D[0] = p_o[0]
            p_o_2D[1] = p_o[1]

        C31 = (
            p_o_2D[0] * s2_2D[1]
            + p_o_2D[1] * s3_2D[0]
            + s2_2D[0] * s3_2D[1]
            - p_o_2D[0] * s3_2D[1]
            - p_o_2D[1] * s2_2D[0]
            - s3_2D[0] * s2_2D[1]
        )
        C32 = (
            p_o_2D[0] * s3_2D[1]
            + p_o_2D[1] * s1_2D[0]
            + s3_2D[0] * s1_2D[1]
            - p_o_2D[0] * s1_2D[1]
            - p_o_2D[1] * s3_2D[0]
            - s1_2D[0] * s3_2D[1]
        )
        C33 = (
            p_o_2D[0] * s1_2D[1]
            + p_o_2D[1] * s2_2D[0]
            + s1_2D[0] * s2_2D[1]
            - p_o_2D[0] * s2_2D[1]
            - p_o_2D[1] * s1_2D[0]
            - s2_2D[0] * s1_2D[1]
        )

        comp1 = _same_sign(M_max, C31)
        comp2 = _same_sign(M_max, C32)
        comp3 = _same_sign(M_max, C33)

        if comp1 and comp2 and comp3:
            return wp.vec3(C31 / M_max, C32 / M_max, C33 / M_max)

        dmin = FLOAT_MAX
        coordinates = wp.vec3(0.0, 0.0, 0.0)

        if not comp1:
            subcoord = _S1D(s2, s3)
            x = subcoord[0] * s2 + subcoord[1] * s3
            d = wp.dot(x, x)
            coordinates[0] = 0.0
            coordinates[1] = subcoord[0]
            coordinates[2] = subcoord[1]
            dmin = d

        if not comp2:
            subcoord = _S1D(s1, s3)
            x = subcoord[0] * s1 + subcoord[1] * s3
            d = wp.dot(x, x)
            if d < dmin:
                coordinates[0] = subcoord[0]
                coordinates[1] = 0.0
                coordinates[2] = subcoord[1]
                dmin = d

        if not comp3:
            subcoord = _S1D(s1, s2)
            x = subcoord[0] * s1 + subcoord[1] * s2
            d = wp.dot(x, x)
            if d < dmin:
                coordinates[0] = subcoord[0]
                coordinates[1] = subcoord[1]
                coordinates[2] = 0.0
        return coordinates

    @wp.func
    def _S3D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3, s4: wp.vec3) -> wp.vec4:
        C41 = -_det3(s2, s3, s4)
        C42 = _det3(s1, s3, s4)
        C43 = -_det3(s1, s2, s4)
        C44 = _det3(s1, s2, s3)

        m_det = C41 + C42 + C43 + C44

        comp1 = _same_sign(m_det, C41)
        comp2 = _same_sign(m_det, C42)
        comp3 = _same_sign(m_det, C43)
        comp4 = _same_sign(m_det, C44)

        if comp1 and comp2 and comp3 and comp4:
            return wp.vec4(C41 / m_det, C42 / m_det, C43 / m_det, C44 / m_det)

        coordinates = wp.vec4(0.0, 0.0, 0.0, 0.0)
        dmin = FLOAT_MAX

        if not comp1:
            subcoord = _S2D(s2, s3, s4)
            x = subcoord[0] * s2 + subcoord[1] * s3 + subcoord[2] * s4
            d = wp.dot(x, x)
            coordinates[0] = 0.0
            coordinates[1] = subcoord[0]
            coordinates[2] = subcoord[1]
            coordinates[3] = subcoord[2]
            dmin = d

        if not comp2:
            subcoord = _S2D(s1, s3, s4)
            x = subcoord[0] * s1 + subcoord[1] * s3 + subcoord[2] * s4
            d = wp.dot(x, x)
            if d < dmin:
                coordinates[0] = subcoord[0]
                coordinates[1] = 0.0
                coordinates[2] = subcoord[1]
                coordinates[3] = subcoord[2]
                dmin = d

        if not comp3:
            subcoord = _S2D(s1, s2, s4)
            x = subcoord[0] * s1 + subcoord[1] * s2 + subcoord[2] * s4
            d = wp.dot(x, x)
            if d < dmin:
                coordinates[0] = subcoord[0]
                coordinates[1] = subcoord[1]
                coordinates[2] = 0.0
                coordinates[3] = subcoord[2]
                dmin = d

        if not comp4:
            subcoord = _S2D(s1, s2, s3)
            x = subcoord[0] * s1 + subcoord[1] * s2 + subcoord[2] * s3
            d = wp.dot(x, x)
            if d < dmin:
                coordinates[0] = subcoord[0]
                coordinates[1] = subcoord[1]
                coordinates[2] = subcoord[2]
                coordinates[3] = 0.0
        return coordinates

    @wp.func
    def _subdistance(n: int, simplex: mat43) -> wp.vec4:
        if n == 4:
            return _S3D(simplex[0], simplex[1], simplex[2], simplex[3])
        if n == 3:
            coordinates3 = _S2D(simplex[0], simplex[1], simplex[2])
            return wp.vec4(coordinates3[0], coordinates3[1], coordinates3[2], 0.0)
        if n == 2:
            coordinates2 = _S1D(simplex[0], simplex[1])
            return wp.vec4(coordinates2[0], coordinates2[1], 0.0, 0.0)
        return wp.vec4(1.0, 0.0, 0.0, 0.0)

    # -------------------------
    # Support mapping utilities
    # -------------------------
    @wp.func
    def support_map_b(
        geom_b: Any,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: Any,
    ) -> tuple[wp.vec3, int]:
        tmp = wp.quat_rotate_inv(orientation_b, direction)
        result, feature_id = support_func(geom_b, tmp, data_provider)
        result = wp.quat_rotate(orientation_b, result)
        result = result + position_b
        return result, feature_id

    @wp.func
    def minkowski_support(
        geom_a: Any,
        geom_b: Any,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
    ) -> tuple[Vert, int, int]:
        v = Vert()
        tmp_result_a = support_func(geom_a, direction, data_provider)
        v.A = tmp_result_a[0]
        feature_a_id = tmp_result_a[1]

        tmp_direction = -direction
        tmp_result_b = support_map_b(geom_b, tmp_direction, orientation_b, position_b, data_provider)
        v.B = tmp_result_b[0]
        feature_b_id = tmp_result_b[1]

        d = wp.normalize(direction) * extend * 0.5
        v.A = v.A + d
        v.B = v.B - d

        return v, feature_a_id, feature_b_id

    @wp.func
    def geometric_center(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: Any,
    ) -> Vert:
        center = Vert()
        center.A = center_func(geom_a, data_provider)
        center.B = center_func(geom_b, data_provider)
        center.B = wp.quat_rotate(orientation_b, center.B)
        center.B = position_b + center.B
        return center

    # --------------
    # GJK iteration
    # --------------
    @wp.func
    def solve_gjk(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
    ) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float, int, int]:
        MAX_ITER = 30
        TOLERANCE = 1.0e-6

        # Transform shape B into A's local space
        rel_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        rel_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        # Initial guess from geometric centers in A-space
        v0 = geometric_center(geom_a, geom_b, rel_orientation_b, rel_position_b, data_provider)
        x_k = v0.A - v0.B

        simplex = mat43()
        simplex1 = mat43()
        simplex2 = mat43()
        simplex_index1 = wp.vec4i()
        simplex_index2 = wp.vec4i()
        coordinates = wp.vec4()
        n = int(0)

        epsilon = 0.5 * TOLERANCE * TOLERANCE

        feature_a_id = int(0)
        feature_b_id = int(0)

        for _ in range(MAX_ITER):
            xnorm = wp.dot(x_k, x_k)
            used_fallback = bool(False)
            dir_neg = wp.vec3(1.0, 0.0, 0.0)
            if xnorm >= 1.0e-12:
                dir_neg = x_k / wp.sqrt(xnorm)
            else:
                used_fallback = True

            # Support on Minkowski difference in A-space
            v, feature_a_id, feature_b_id = minkowski_support(
                geom_a, geom_b, -dir_neg, rel_orientation_b, rel_position_b, sum_of_contact_offsets, data_provider
            )

            simplex1[n] = v.A
            simplex2[n] = v.B
            simplex[n] = vert_v(v)
            simplex_index1[n] = feature_a_id
            simplex_index2[n] = feature_b_id

            # Frank-Wolfe duality gap stopping (skip on fallback to avoid premature exit)
            if not used_fallback:
                if wp.dot(x_k, x_k - simplex[n]) < epsilon:
                    break

            # Barycentric on current simplex
            coordinates = _subdistance(n + 1, simplex)

            # Compress simplex
            n = int(0)
            for i in range(4):
                if coordinates[i] == 0.0:
                    continue
                simplex[n] = simplex[i]
                simplex1[n] = simplex1[i]
                simplex2[n] = simplex2[i]
                simplex_index1[n] = simplex_index1[i]
                simplex_index2[n] = simplex_index2[i]
                coordinates[n] = coordinates[i]
                n += int(1)

            if n < 1:
                break

            x_next = _linear_combine(n, coordinates, simplex)
            if _almost_equal(x_next, x_k):
                break
            x_k = x_next

            if n == 4:
                break

        # Compute witness points in A-space
        point_a_local = _linear_combine(n, coordinates, simplex1)
        point_b_local = _linear_combine(n, coordinates, simplex2)
        diff_local = _linear_combine(n, coordinates, simplex)
        dist = wp.norm_l2(diff_local)

        # Choose feature ids from dominant barycentric weight
        max_w = float(-1.0)
        feature_a_pick = int(0)
        feature_b_pick = int(0)
        for i in range(n):
            if coordinates[i] > max_w:
                max_w = coordinates[i]
                feature_a_pick = simplex_index1[i]
                feature_b_pick = simplex_index2[i]

        # Transform results back to world space
        point_a = wp.quat_rotate(orientation_a, point_a_local) + position_a
        point_b = wp.quat_rotate(orientation_a, point_b_local) + position_a

        # Determine collision and compute normal/penetration
        collision = bool(n == 4)
        normal = wp.vec3(0.0, 0.0, 0.0)
        penetration = float(0.0)

        if collision:
            # Objects are overlapping - return negative penetration depth
            # For overlapping objects, the distance represents penetration depth
            penetration = -dist
            if dist > 0.0:
                # Normal points from B to A (separation direction)
                normal_local = diff_local / dist
                normal = wp.quat_rotate(orientation_a, normal_local)
        else:
            # Objects are separated - return positive distance
            penetration = dist
            if dist > 0.0:
                normal_local = diff_local / dist
                normal = wp.quat_rotate(orientation_a, normal_local)

        return collision, point_a, point_b, normal, penetration, feature_a_pick, feature_b_pick

    return solve_gjk
