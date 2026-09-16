# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Plain fixed-six loop bounds for compiler-unrolled block preparation."""

import warp as wp

from newton._src.solvers.phoenx.constraints.bilateral_joint import (
    BodyContainer,
    ConstraintContainer,
    CopyStateContainer,
    Mat66d,
    Vec6d,
    _dot_double,
    _response,
    constraint_get_body1,
    constraint_get_body2,
)


@wp.kernel(enable_backward=False)
def fixed_prepare(constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer):
    cid = wp.tid()
    data = constraints.bilateral
    count = data.row_count[cid]
    data.valid[cid] = wp.int32(0)
    if count == 0:
        return
    structural = data.structural_index[cid]
    body0 = constraint_get_body1(constraints, cid)
    body1 = constraint_get_body2(constraints, cid)
    factor0 = wp.float32(1.0)
    factor1 = wp.float32(1.0)
    if copy_state.highest_index_in_use[0] > 0:
        factor0 = wp.float32(wp.max(copy_state.count_per_node[body0], wp.int32(1)))
        factor1 = wp.float32(wp.max(copy_state.count_per_node[body1], wp.int32(1)))
    for i in range(6):
        if i < count:
            row = data.row_indices[cid, i]
            local = data.row_local[row]
            data.response0[cid, i] = factor0 * _response(bodies, body0, data.wrench0[structural, local])
            data.response1[cid, i] = factor1 * _response(bodies, body1, data.wrench1[structural, local])
    matrix = Mat66d()
    for i in range(6):
        if i < count:
            row = data.row_indices[cid, i]
            local = data.row_local[row]
            j0 = data.wrench0[structural, local]
            j1 = data.wrench1[structural, local]
            for j in range(6):
                if j < i + 1:
                    value = _dot_double(j0, data.response0[cid, j]) + _dot_double(j1, data.response1[cid, j])
                    if i == j and data.row_dynamic[row]:
                        value += wp.float64(1.0) / wp.float64(data.dynamic_mass[row])
                    matrix[i, j] = value
                    matrix[j, i] = value
    lower = Mat66d()
    diagonal = Vec6d()
    valid = wp.bool(True)
    for i in range(6):
        if i < count:
            pivot = matrix[i, i]
            for j in range(6):
                if j < i:
                    pivot -= lower[i, j] * lower[i, j] * diagonal[j]
            diagonal[i] = pivot
            lower[i, i] = wp.float64(1.0)
            if pivot <= wp.float64(0.0):
                valid = False
            if valid:
                for j in range(6):
                    if j >= i + 1 and j < count:
                        value = matrix[j, i]
                        for k in range(6):
                            if k < i:
                                value -= lower[j, k] * lower[i, k] * diagonal[k]
                        lower[j, i] = value / pivot
    data.lower[cid] = lower
    data.diagonal[cid] = diagonal
    if valid:
        data.valid[cid] = wp.int32(1)
