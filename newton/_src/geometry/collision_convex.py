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

from .gjk_stateless import create_solve_gjk
from .mpr import create_solve_mpr


def create_solve_convex_contact(support_func: Any, center_func: Any):
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
        # First run GJK to test overlap quickly
        (
            gjk_collision,
            gjk_point_a,
            gjk_point_b,
            gjk_normal,
            gjk_penetration,
            gjk_feature_a,
            gjk_feature_b,
        ) = wp.static(create_solve_gjk(support_func, center_func))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            sum_of_contact_offsets,
            data_provider,
        )

        if gjk_collision:
            # Use MPR to get accurate contact points and penetration info
            (
                mpr_collision,
                mpr_point_a,
                mpr_point_b,
                mpr_normal,
                mpr_penetration,
                mpr_feature_a,
                mpr_feature_b,
            ) = wp.static(create_solve_mpr(support_func, center_func))(
                geom_a,
                geom_b,
                orientation_a,
                orientation_b,
                position_a,
                position_b,
                sum_of_contact_offsets,
                data_provider,
            )

            if mpr_collision:
                return (
                    True,
                    mpr_point_a,
                    mpr_point_b,
                    mpr_normal,
                    mpr_penetration,
                    mpr_feature_a,
                    mpr_feature_b,
                )
            else:
                # Fallback to GJK result if MPR fails for any reason
                return (
                    True,
                    gjk_point_a,
                    gjk_point_b,
                    gjk_normal,
                    gjk_penetration,
                    gjk_feature_a,
                    gjk_feature_b,
                )

        # No overlap, return the GJK result (likely gjk_collision == False)
        return (
            gjk_collision,
            gjk_point_a,
            gjk_point_b,
            gjk_normal,
            gjk_penetration,
            gjk_feature_a,
            gjk_feature_b,
        )

    return solve_convex_contact
