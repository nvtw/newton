# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional per-axis state for maximal-coordinate D6 inequalities."""

import warp as wp

D6_AXIS_COUNT = 6
D6_ROW_AXIS = 0
D6_ROW_DISTANCE = 1


@wp.struct
class D6JointData:
    """Compact common state for limited, frictional, or speed-capped D6 axes.

    Authored axes stay separate from bilateral factor rows because a drive and
    its unilateral bounds may coexist on the same degree of freedom.
    """

    enabled: wp.int32
    row_count: wp.array[wp.int32]
    row_axis: wp.array2d[wp.int32]
    row_kind: wp.array2d[wp.int32]
    linear_count: wp.array[wp.int32]
    angular_count: wp.array[wp.int32]
    axis: wp.array2d[wp.vec3f]
    joint_x_p: wp.array[wp.transform]
    joint_x_c: wp.array[wp.transform]
    lower: wp.array2d[wp.float32]
    upper: wp.array2d[wp.float32]
    velocity_limit: wp.array2d[wp.float32]
    friction: wp.array2d[wp.float32]
    unwrap_angle: wp.array2d[wp.int32]
    condense_translation: wp.array2d[wp.int32]
    revolution_counter: wp.array2d[wp.int32]
    previous_angle: wp.array2d[wp.float32]
    wrench0: wp.array2d[wp.spatial_vector]
    wrench1: wp.array2d[wp.spatial_vector]
    coordinate: wp.array2d[wp.float32]
    effective_mass_inverse: wp.array2d[wp.float32]
    lower_impulse: wp.array2d[wp.float32]
    upper_impulse: wp.array2d[wp.float32]
    friction_impulse: wp.array2d[wp.float32]
