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
        # First run GJK to test overlap quickly while keeping only the tuple live
        gjk_res = wp.static(create_solve_gjk(support_func, center_func))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            sum_of_contact_offsets,
            data_provider,
        )
        gjk_collision = gjk_res[0]
        if not gjk_collision:
            return (
                gjk_res[0],
                gjk_res[1],
                gjk_res[2],
                gjk_res[3],
                gjk_res[4],
                gjk_res[5],
                gjk_res[6],
            )

        # Use MPR to get accurate contact points and penetration info and return its result directly
        return wp.static(create_solve_mpr(support_func, center_func))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            sum_of_contact_offsets,
            data_provider,
        )

    return solve_convex_contact
