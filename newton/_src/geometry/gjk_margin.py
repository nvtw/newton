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

"""MPR with inflation for unified signed-distance computation.

Inflates both shapes by ``margin/2`` via the existing MPR ``extend`` parameter,
then corrects the signed distance. This handles both overlapping and separated
shapes in a single MPR pass, eliminating the need for a separate GJK fallback.
"""

from typing import Any

import warp as wp

from .mpr import create_solve_mpr


def create_solve_mpr_margin(support_func: Any):
    """Factory: build an MPR-with-inflation signed-distance solver.

    Inflates both shapes by ``margin/2`` each, runs standard MPR, then
    corrects the signed distance by adding ``margin``.

    For shapes separated by less than ``margin``, MPR reports them as
    overlapping (due to inflation) and returns an accurate signed distance
    and contact point.  For shapes separated by more than ``margin``,
    MPR returns ``collision=False``.

    Args:
        support_func: Per-shape support mapping.

    Returns:
        ``solve_mpr_margin`` warp function.
    """

    solve_mpr = create_solve_mpr(support_func)

    @wp.func
    def solve_mpr_margin(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        margin: float,
        data_provider: Any,
    ) -> tuple[bool, float, wp.vec3, wp.vec3]:
        """Compute signed distance between two convex shapes using inflated MPR.

        Returns:
            ``(converged, signed_distance, contact_point_center, normal_a_to_b)``.
            ``converged`` is False when shapes are separated by more than ``margin``.
        """
        collision, signed_distance, point, normal = solve_mpr(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            margin,  # inflate both shapes by margin/2 each
            data_provider,
        )

        # Undo the inflation: the inflated signed distance is too negative by margin
        signed_distance = signed_distance + margin

        return collision, signed_distance, point, normal

    return solve_mpr_margin
