# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional prepared bilateral joint blocks for the colored rigid solver."""

import warp as wp

Vec6d = wp.types.vector(length=6, dtype=wp.float64)
Mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


@wp.struct
class BilateralJointData:
    enabled: wp.int32
    row_count: wp.array[wp.int32]
    row_indices: wp.array2d[wp.int32]
    structural_index: wp.array[wp.int32]
    row_local: wp.array[wp.int32]
    row_dynamic: wp.array[wp.bool]
    wrench0: wp.array2d[wp.spatial_vector]
    wrench1: wp.array2d[wp.spatial_vector]
    bias: wp.array2d[wp.float32]
    reference: wp.array[wp.float32]
    dynamic_mass: wp.array[wp.float32]
    accumulated: wp.array[wp.float32]
    response0: wp.array2d[wp.spatial_vector]
    response1: wp.array2d[wp.spatial_vector]
    lower: wp.array[Mat66d]
    diagonal: wp.array[Vec6d]
    valid: wp.array[wp.int32]
