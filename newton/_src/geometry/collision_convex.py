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
from typing import Any

import warp as wp

from .simplex_solver import create_solve_closest_distance
from .mpr import create_solve_mpr
from .multicontact import create_build_manifold


def create_solve_convex_contact(support_func: Any):
    @wp.func
    def solve_convex_contact(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
    ) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float, int, int]:
        # First run closest distance solver to test overlap quickly while keeping only the tuple live
        collision, point_a, point_b, normal, penetration, feature_a_id, feature_b_id = wp.static(
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
        if not collision:
            return (
                collision,
                point_a,
                point_b,
                normal,
                penetration,
                feature_a_id,
                feature_b_id,
            )

        # Use MPR to get accurate contact points and penetration info and return its result directly
        collision, point_a, point_b, normal, penetration, feature_a_id, feature_b_id = wp.static(
            create_solve_mpr(support_func)
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
        return (
            collision,
            point_a,
            point_b,
            normal,
            penetration,
            feature_a_id,
            feature_b_id,
        )

    return solve_convex_contact


_mat43f = wp.types.matrix((4, 3), wp.float32)


def create_solve_convex_multi_contact(support_func: Any):
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
        # Broad check with closest distance solver; refine with MPR on overlap for better anchors/normal
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

            # if not collision:
            #     wp.printf("GJK point_a: (%f,%f,%f), point_b: (%f,%f,%f), normal: (%f,%f,%f), dist: %f\n", point_a[0], point_a[1], point_a[2], point_b[0], point_b[1], point_b[2], normal[0], normal[1], normal[2], penetration)

            # wp.printf("MPR result: collision=%d, penetration=%f, point_a=(%f,%f,%f), point_b=(%f,%f,%f), normal=(%f,%f,%f), feature_ids=(%d,%d)\n",
            #          int(collision), penetration,
            #          point_a[0], point_a[1], point_a[2],
            #          point_b[0], point_b[1], point_b[2],
            #          normal[0], normal[1], normal[2],
            #          feature_a_id, feature_b_id)

        # if collision:
        #     wp.printf("MPR point_a: (%f,%f,%f), point_b: (%f,%f,%f), normal: (%f,%f,%f), dist: %f\n", point_a[0], point_a[1], point_a[2], point_b[0], point_b[1], point_b[2], normal[0], normal[1], normal[2], penetration)

        #    wp.printf("MPR point_a: (%f,%f,%f), point_b: (%f,%f,%f), normal: (%f,%f,%f), dist: %f\n", point_a[0], point_a[1], point_a[2], point_b[0], point_b[1], point_b[2], normal[0], normal[1], normal[2], penetration)

        if skip_multi_contact or penetration > contact_threshold:
            count = 1
            penetrations = wp.vec4(penetration, 0.0, 0.0, 0.0)
            points = _mat43f()
            points[0] = point
            features = wp.vec4i(0)
            return count, normal, penetrations, points, features

        count, penetrations, points, features = wp.static(create_build_manifold(support_func))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            point - normal * (penetration * 0.5),
            point + normal * (penetration * 0.5),
            normal,
            feature_a_id,
            feature_b_id,
            data_provider,
        )

        # wp.printf("Manifold result: count=%d, normal=(%f,%f,%f), points_a=(%f,%f,%f),(%f,%f,%f),(%f,%f,%f),(%f,%f,%f), points_b=(%f,%f,%f),(%f,%f,%f),(%f,%f,%f),(%f,%f,%f), penetrations=(%f,%f,%f,%f)\n",
        #          count,
        #          normal[0], normal[1], normal[2],
        #          points_a[0][0], points_a[0][1], points_a[0][2],
        #          points_a[1][0], points_a[1][1], points_a[1][2],
        #          points_a[2][0], points_a[2][1], points_a[2][2],
        #          points_a[3][0], points_a[3][1], points_a[3][2],
        #          points_b[0][0], points_b[0][1], points_b[0][2],
        #          points_b[1][0], points_b[1][1], points_b[1][2],
        #          points_b[2][0], points_b[2][1], points_b[2][2],
        #          points_b[3][0], points_b[3][1], points_b[3][2],
        #          penetrations[0], penetrations[1], penetrations[2], penetrations[3])

        # if count == 0:
        #     print("create_build_manifold removed all contacts")

        # penetrations = wp.vec4(penetration)

        return count, normal, penetrations, points, features

    return solve_convex_multi_contact
