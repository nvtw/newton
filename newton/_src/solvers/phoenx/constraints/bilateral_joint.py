# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mass-metric joint blocks evaluated in the same colors as rigid contacts."""

import functools

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.bilateral_joint_data import BilateralJointData, Mat66d, Vec6d
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import _ms_load_body_pair, _ms_store_body_pair
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@wp.func
def _dot_double(a: wp.spatial_vector, b: wp.spatial_vector) -> wp.float64:
    value = wp.float64(0.0)
    for component in range(6):
        value += wp.float64(a[component]) * wp.float64(b[component])
    return value


@wp.func
def _response(bodies: BodyContainer, body: wp.int32, wrench: wp.spatial_vector) -> wp.spatial_vector:
    linear = bodies.inverse_mass[body] * wp.spatial_top(wrench)
    angular = mat33_from_sym6(bodies.inverse_inertia_world[body]) * wp.spatial_bottom(wrench)
    return wp.spatial_vector(linear, angular)


@wp.kernel(enable_backward=False)
def prepare_bilateral_joint_blocks(
    constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer
):
    # A joint has at most six rows. Constant loop bounds let Warp unroll
    # matrix indexing while the guards preserve the original active-row order.
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


@wp.func
def _backward_bilateral_impulses(
    data: BilateralJointData,
    cid: wp.int32,
    count: wp.int32,
    structural: wp.int32,
    solution: Vec6d,
):
    lower = data.lower[cid]
    # Constant bounds let Warp unroll matrix indexing without changing
    # descending solve rows or ascending subtractions within each row.
    for reverse in range(6):
        i = 5 - reverse
        if i < count:
            value = solution[i]
            for j in range(6):
                if j > i and j < count:
                    value -= lower[j, i] * solution[j]
            solution[i] = value

    impulse0 = wp.spatial_vector()
    impulse1 = wp.spatial_vector()
    for i in range(count):
        row = data.row_indices[cid, i]
        local = data.row_local[row]
        impulse = wp.float32(solution[i])
        data.accumulated[row] += impulse
        impulse0 += impulse * data.wrench0[structural, local]
        impulse1 += impulse * data.wrench1[structural, local]
    return impulse0, impulse1


@wp.func
def _solve_bilateral_impulses(
    data: BilateralJointData,
    cid: wp.int32,
    count: wp.int32,
    structural: wp.int32,
    rhs: Vec6d,
):
    lower = data.lower[cid]
    diagonal = data.diagonal[cid]
    solution = Vec6d()
    for i in range(count):
        value = rhs[i]
        for j in range(i):
            value -= lower[i, j] * solution[j]
        solution[i] = value
    for i in range(count):
        solution[i] /= diagonal[i]
    return _backward_bilateral_impulses(data, cid, count, structural, solution)


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    // Each constraint owns eight consecutive lanes in a CUDA warp.
    unsigned int mask = 0xffu << (threadIdx.x & 24);
    return __shfl_sync(mask, value, source_lane, 8);
#else
    return value;
#endif
"""
)
def _shuffle_bilateral_rhs(value: wp.float64, source_lane: wp.int32) -> wp.float64: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    unsigned int mask = 0xffu << (threadIdx.x & 24);
    return __shfl_sync(mask, value, source_lane, 8);
#else
    return value;
#endif
""")
def _shuffle_bilateral_response(value: wp.float32, source_lane: wp.int32) -> wp.float32: ...


# The scalar compiler fuses the pivot subtraction after a rounded square.
# Explicit FP64 operations prevent row-product reuse from changing that
# contraction in the cooperative kernel. FP32 body responses stay unchanged.
@wp.func_native("""
#if defined(__CUDA_ARCH__)
    double squared = __dmul_rn(lower, lower);
    return __fma_rn(-squared, diagonal, pivot);
#else
    return fma(-(lower * lower), diagonal, pivot);
#endif
""")
def _subtract_bilateral_pivot(pivot: wp.float64, lower: wp.float64, diagonal: wp.float64) -> wp.float64: ...


@wp.kernel(enable_backward=False)
def _prepare_bilateral_joint_blocks_cooperative(
    constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer
):
    # Eight consecutive CUDA lanes own a joint; every lane must participate
    # in the subgroup shuffles, including the two unused row lanes.
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
            other0[component] = _shuffle_bilateral_response(response0[component], j)
            other1[component] = _shuffle_bilateral_response(response1[component], j)
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
                    pivot_local = _subtract_bilateral_pivot(pivot_local, lower_row[j], diagonal[j])
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


@functools.cache
def get_iterate_bilateral_joint_block(cooperative: bool = False, *, temporal_springs: bool = False):
    """Build scalar or eight-lane RHS and forward solve with shared back solve.

    The cooperative variant requires eight participating CUDA lanes per joint.
    All lanes take the same enabled/count/valid returns, and gather before
    nonleaders exit. Each row retains its original FP64 component and
    subtraction order; backward substitution and impulse scatter stay scalar.

    Temporal springs solve a fresh force increment each substep. Their existing
    velocity already includes earlier increments, so their residual omits the
    accumulated-impulse compliance term. The factor and paired wrench scatter
    are unchanged. This mode requires unbounded drives with zero armature and
    one spring solve per temporal substep. Velocity relaxation solves only
    structural rows; applying springs there would advance them beyond the step.
    """

    @wp.func
    def iterate(
        constraints: ConstraintContainer,
        cid: wp.int32,
        bodies: BodyContainer,
        particles: ParticleContainer,
        copy_state: CopyStateContainer,
        num_bodies: wp.int32,
        parallel_id: wp.int32,
        use_bias: wp.bool,
        tile_lane: wp.int32,
    ):
        data = constraints.bilateral
        if data.enabled == 0:
            return
        count = data.row_count[cid]
        if wp.static(temporal_springs):
            if not use_bias:
                # Topology orders structural rows before springs. The leading
                # LDL factor solves the structural block without spring forces,
                # as PhysX conclude1DStep requires for velocity iterations.
                for i in range(count):
                    if data.row_dynamic[data.row_indices[cid, i]]:
                        count = i
                        break
        if count == 0 or data.valid[cid] == 0:
            return
        structural = data.structural_index[cid]
        body0 = constraint_get_body1(constraints, cid)
        body1 = constraint_get_body2(constraints, cid)
        v0, v1, w0, w1, inv_m0, inv_m1, inv_i0, inv_i1, slot0, slot1 = _ms_load_body_pair(
            bodies, particles, copy_state, body0, body1, parallel_id, num_bodies
        )
        twist0 = wp.spatial_vector(v0, w0)
        twist1 = wp.spatial_vector(v1, w1)
        rhs = Vec6d()
        if wp.static(cooperative):
            rhs_value = wp.float64(0.0)
            if tile_lane < count:
                i = tile_lane
                row = data.row_indices[cid, i]
                local = data.row_local[row]
                residual = _dot_double(data.wrench0[structural, local], twist0)
                residual += _dot_double(data.wrench1[structural, local], twist1)
                if data.row_dynamic[row]:
                    if wp.static(not temporal_springs):
                        residual += wp.float64(data.accumulated[row]) / wp.float64(data.dynamic_mass[row])
                    residual -= wp.float64(data.reference[row])
                elif use_bias:
                    residual += wp.float64(data.bias[structural, local])
                rhs_value = -residual

            # Each row subtracts solved earlier rows in the same ascending
            # order as the scalar forward solve. Scaling must follow all rows.
            lower = data.lower[cid]
            for forward_row in range(count):
                solved = _shuffle_bilateral_rhs(rhs_value, forward_row)
                if tile_lane > forward_row and tile_lane < count:
                    rhs_value -= lower[tile_lane, forward_row] * solved
            if tile_lane < count:
                diagonal = data.diagonal[cid]
                rhs_value /= diagonal[tile_lane]
            rhs = Vec6d()
            for row_lane in range(count):
                rhs[row_lane] = _shuffle_bilateral_rhs(rhs_value, row_lane)
            if tile_lane != 0:
                return
        else:
            rhs = Vec6d()
            for i in range(count):
                row = data.row_indices[cid, i]
                local = data.row_local[row]
                residual = _dot_double(data.wrench0[structural, local], twist0)
                residual += _dot_double(data.wrench1[structural, local], twist1)
                if data.row_dynamic[row]:
                    if wp.static(not temporal_springs):
                        residual += wp.float64(data.accumulated[row]) / wp.float64(data.dynamic_mass[row])
                    residual -= wp.float64(data.reference[row])
                elif use_bias:
                    residual += wp.float64(data.bias[structural, local])
                rhs[i] = -residual

        if wp.static(cooperative):
            impulse0, impulse1 = _backward_bilateral_impulses(data, cid, count, structural, rhs)
        else:
            impulse0, impulse1 = _solve_bilateral_impulses(data, cid, count, structural, rhs)
        v0 += inv_m0 * wp.spatial_top(impulse0)
        v1 += inv_m1 * wp.spatial_top(impulse1)
        w0 += inv_i0 * wp.spatial_bottom(impulse0)
        w1 += inv_i1 * wp.spatial_bottom(impulse1)
        _ms_store_body_pair(bodies, particles, copy_state, body0, body1, slot0, slot1, num_bodies, v0, w0, v1, w1)

    return iterate


_iterate_bilateral_scalar = get_iterate_bilateral_joint_block(False)


@wp.func
def iterate_bilateral_joint_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    use_bias: wp.bool,
):
    _iterate_bilateral_scalar(
        constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, use_bias, wp.int32(0)
    )
