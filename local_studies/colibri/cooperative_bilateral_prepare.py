"""Local eight-lane joint preparation; scalar arithmetic order retained."""

import warp as wp

from newton._src.solvers.phoenx.constraints.bilateral_joint import (
    BodyContainer,
    ConstraintContainer,
    CopyStateContainer,
    Mat66d,
    Vec6d,
    _dot_double,
    _response,
    _shuffle_bilateral_rhs,
    constraint_get_body1,
    constraint_get_body2,
)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    unsigned int mask = 0xffu << (threadIdx.x & 24);
    return __shfl_sync(mask, value, source_lane, 8);
#else
    return value;
#endif
""")
def shuffle_float(value: wp.float32, source_lane: wp.int32) -> wp.float32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    double squared = __dmul_rn(lower, lower);
    return __fma_rn(-squared, diagonal, pivot);
#else
    return fma(-(lower * lower), diagonal, pivot);
#endif
""")
def subtract_pivot(pivot: wp.float64, lower: wp.float64, diagonal: wp.float64) -> wp.float64: ...


@wp.kernel(enable_backward=False)
def cooperative_prepare(constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer):
    tid = wp.tid()
    cid = tid // 8
    lane = tid % 8
    data = constraints.bilateral
    count = data.row_count[cid]
    if lane == 0:
        data.valid[cid] = 0
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
    response0 = wp.spatial_vector()
    response1 = wp.spatial_vector()
    wrench0 = wp.spatial_vector()
    wrench1 = wp.spatial_vector()
    row = wp.int32(0)
    if lane < count:
        row = data.row_indices[cid, lane]
        local = data.row_local[row]
        wrench0 = data.wrench0[structural, local]
        wrench1 = data.wrench1[structural, local]
        response0 = factor0 * _response(bodies, body0, wrench0)
        response1 = factor1 * _response(bodies, body1, wrench1)
        data.response0[cid, lane] = response0
        data.response1[cid, lane] = response1
    matrix_row = Vec6d()
    for j in range(6):
        other0 = wp.spatial_vector()
        other1 = wp.spatial_vector()
        for component in range(6):
            other0[component] = shuffle_float(response0[component], j)
            other1[component] = shuffle_float(response1[component], j)
        if lane < count and j <= lane:
            value = _dot_double(wrench0, other0) + _dot_double(wrench1, other1)
            if j == lane and data.row_dynamic[row]:
                value += wp.float64(1.0) / wp.float64(data.dynamic_mass[row])
            matrix_row[j] = value
    lower_row = Vec6d()
    diagonal = Vec6d()
    valid = wp.bool(True)
    for i in range(6):
        pivot_local = wp.float64(0.0)
        if i < count and lane == i:
            pivot_local = matrix_row[i]
            for j in range(6):
                if j < i:
                    pivot_local = subtract_pivot(pivot_local, lower_row[j], diagonal[j])
        pivot = _shuffle_bilateral_rhs(pivot_local, i)
        if i < count:
            diagonal[i] = pivot
            if lane == i:
                lower_row[i] = wp.float64(1.0)
            if pivot <= wp.float64(0.0):
                valid = False
            if valid:
                value = matrix_row[i]
                for k in range(6):
                    pivot_row_lower = _shuffle_bilateral_rhs(lower_row[k], i)
                    if k < i and lane > i and lane < count:
                        value -= lower_row[k] * pivot_row_lower * diagonal[k]
                if lane > i and lane < count:
                    lower_row[i] = value / pivot
    # The factor array holds whole structs: only the leader writes them.
    lower = Mat66d()
    for i in range(6):
        for j in range(6):
            entry = _shuffle_bilateral_rhs(lower_row[j], i)
            if lane == 0:
                lower[i, j] = entry
    if lane == 0:
        data.lower[cid] = lower
        data.diagonal[cid] = diagonal
        if valid:
            data.valid[cid] = 1


def launch(constraints, bodies, copies, count, device, block_dim=32):
    """Launch complete eight-lane groups; no cross-subgroup barriers."""
    if block_dim not in (32, 64):
        raise ValueError("Use32 or64 threads per block")
    wp.launch(
        cooperative_prepare, dim=count * 8, inputs=[constraints, bodies, copies], device=device, block_dim=block_dim
    )
