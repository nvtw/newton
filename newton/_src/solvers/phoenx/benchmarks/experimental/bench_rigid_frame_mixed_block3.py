# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-shaped benchmark for row-wise mixed dense frame unification.

``RigidFrameBlock3`` unifies contact, point-lock, and angular 3-row blocks when
all rows in the block share the same residual/apply shape. This benchmark tests
the next step: one dense frame descriptor with per-row modes, so mixed blocks
such as planar constraints can also use the same local PGS operation.

The split baseline mirrors typed production helpers:

* contact rows use scalar cone projection at a point;
* point-lock rows use dense ``VelocityBlock3`` at an anchor;
* angular rows use dense ``VelocityBlock3`` as a pure couple;
* planar rows use dense ``VelocityBlock3`` with one linear row and two angular
  rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.experimental.bench_rigid_rows3_sidecar import (
    _alloc_mat,
    _alloc_vec,
    _bench,
    _max_err,
)
from newton._src.solvers.phoenx.benchmarks.experimental.bench_rigid_rows3_world_loop import _build_schedule
from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    VELOCITY_BLOCK_PROJECT_IDENTITY,
    VELOCITY_ROWS3_PROJECT_BOUNDS,
    VELOCITY_ROWS3_PROJECT_CONTACT_CONE,
    RigidFrameBlock3Mixed,
    RigidFrameRows3State,
    VelocityBlock3,
    VelocityBlockProjection,
    VelocityRows3Op,
    block_solve_rigid_frame_block3_mixed,
    block_solve_velocity_block3_projected,
    block_solve_velocity_rows3_op,
)

_FAMILY_CONTACT = wp.constant(wp.int32(0))
_FAMILY_POINT = wp.constant(wp.int32(1))
_FAMILY_ANGULAR = wp.constant(wp.int32(2))
_FAMILY_PLANAR = wp.constant(wp.int32(3))


@dataclass(frozen=True)
class ScenePreset:
    rows_per_color: tuple[int, ...]
    contact_ratio: float
    point_ratio: float
    angular_ratio: float
    planar_ratio: float


_SCENE_PRESETS: dict[str, ScenePreset] = {
    "h1": ScenePreset((18, 14, 9, 5), 0.12, 0.24, 0.22, 0.20),
    "g1": ScenePreset((28, 21, 13, 6), 0.00, 0.26, 0.24, 0.24),
    "dr_legs": ScenePreset((24, 21, 15, 8, 3), 0.08, 0.24, 0.22, 0.24),
    "tower": ScenePreset((96, 91, 86, 80, 74, 68, 62, 56, 50, 44, 38, 32), 1.0, 0.0, 0.0, 0.0),
}


@wp.func
def _identity_block_projection() -> VelocityBlockProjection:
    projection = VelocityBlockProjection()
    projection.mode = VELOCITY_BLOCK_PROJECT_IDENTITY
    return projection


@wp.func
def _diag33(v: wp.vec3f) -> wp.mat33f:
    return wp.mat33f(
        v[0],
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        v[1],
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        v[2],
    )


@wp.kernel(enable_backward=False)
def _init_mixed_frame_kernel(
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    linear_mode: wp.array[wp.vec3f],
    cross_mode: wp.array[wp.vec3f],
    angular_mode: wp.array[wp.vec3f],
    k_inv_diag: wp.array[wp.vec3f],
    k_inv_block: wp.array[wp.mat33f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    period: wp.int32,
    contact_slots: wp.int32,
    point_slots: wp.int32,
    angular_slots: wp.int32,
):
    tid = wp.tid()
    lane = tid % period
    point_limit = contact_slots + point_slots
    angular_limit = point_limit + angular_slots
    phase = wp.float32((tid * wp.int32(37)) & wp.int32(255)) * wp.float32(0.00390625)

    v_a[tid] = wp.vec3f(phase, wp.float32(0.2) - phase, wp.float32(0.1) + wp.float32(0.5) * phase)
    w_a[tid] = wp.vec3f(wp.float32(0.3) * phase, wp.float32(0.4) - phase, wp.float32(0.2) * phase)
    v_b[tid] = wp.vec3f(wp.float32(0.15) - phase, wp.float32(0.1) + phase, wp.float32(0.25) * phase)
    w_b[tid] = wp.vec3f(wp.float32(0.2) + wp.float32(0.25) * phase, -wp.float32(0.1) * phase, phase)
    inv_m_a[tid] = wp.float32(0.8) + wp.float32(0.1) * phase
    inv_m_b[tid] = wp.float32(0.7) + wp.float32(0.2) * phase
    inv_i_a[tid] = wp.mat33f(
        wp.float32(0.60),
        wp.float32(0.02),
        wp.float32(0.00),
        wp.float32(0.02),
        wp.float32(0.70),
        wp.float32(0.01),
        wp.float32(0.00),
        wp.float32(0.01),
        wp.float32(0.80),
    )
    inv_i_b[tid] = wp.mat33f(
        wp.float32(0.90),
        -wp.float32(0.01),
        wp.float32(0.02),
        -wp.float32(0.01),
        wp.float32(0.65),
        wp.float32(0.00),
        wp.float32(0.02),
        wp.float32(0.00),
        wp.float32(0.75),
    )

    n = wp.normalize(wp.vec3f(wp.float32(1.0), wp.float32(0.2), wp.float32(0.1)))
    t1 = wp.normalize(wp.vec3f(-wp.float32(0.2), wp.float32(1.0), wp.float32(0.0)))
    t2 = wp.normalize(wp.cross(n, t1))
    rr0 = wp.vec3f(wp.float32(0.10), wp.float32(0.03) + wp.float32(0.02) * phase, -wp.float32(0.05))
    rr1 = wp.vec3f(-wp.float32(0.08), wp.float32(0.04), wp.float32(0.06) + wp.float32(0.01) * phase)
    r0[tid] = rr0
    r1[tid] = rr1
    bias[tid] = wp.vec3f(wp.float32(0.02) - wp.float32(0.03) * phase, wp.float32(0.01), -wp.float32(0.015) * phase)
    lambda_old[tid] = wp.vec3f(wp.float32(0.05) + wp.float32(0.02) * phase, wp.float32(0.01), wp.float32(0.015) * phase)

    if lane < contact_slots:
        family[tid] = _FAMILY_CONTACT
        axis0[tid] = n
        axis1[tid] = t1
        axis2[tid] = t2
        k_diag = wp.vec3f(
            wp.float32(0.35) + wp.float32(0.1) * phase,
            wp.float32(0.45) + wp.float32(0.1) * phase,
            wp.float32(0.55) + wp.float32(0.1) * phase,
        )
        k_inv_diag[tid] = k_diag
        k_inv_block[tid] = _diag33(k_diag)
        linear_mode[tid] = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
        cross_mode[tid] = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
        angular_mode[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        mass_coeff[tid] = wp.vec3f(wp.float32(0.82), wp.float32(1.0), wp.float32(1.0))
        impulse_coeff[tid] = wp.vec3f(wp.float32(0.11), wp.float32(0.0), wp.float32(0.0))
        lambda_min[tid] = wp.vec3f(wp.float32(0.0), -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        lambda_max[tid] = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_CONTACT_CONE
        friction_static[tid] = wp.float32(0.8)
        friction_kinetic[tid] = wp.float32(0.6)
    elif lane < point_limit:
        family[tid] = _FAMILY_POINT
        axis0[tid] = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
        axis1[tid] = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(0.0))
        axis2[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
        k_inv_diag[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        k_inv_block[tid] = wp.mat33f(
            wp.float32(0.56) + wp.float32(0.03) * phase,
            wp.float32(0.04),
            -wp.float32(0.02),
            wp.float32(0.04),
            wp.float32(0.63) + wp.float32(0.02) * phase,
            wp.float32(0.03),
            -wp.float32(0.02),
            wp.float32(0.03),
            wp.float32(0.72) + wp.float32(0.01) * phase,
        )
        linear_mode[tid] = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
        cross_mode[tid] = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
        angular_mode[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        mass_coeff[tid] = wp.vec3f(wp.float32(0.86), wp.float32(0.86), wp.float32(0.86))
        impulse_coeff[tid] = wp.vec3f(wp.float32(0.08), wp.float32(0.08), wp.float32(0.08))
        lambda_min[tid] = wp.vec3f(-BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        lambda_max[tid] = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_BOUNDS
        friction_static[tid] = wp.float32(0.0)
        friction_kinetic[tid] = wp.float32(0.0)
    elif lane < angular_limit:
        family[tid] = _FAMILY_ANGULAR
        axis0[tid] = wp.normalize(wp.vec3f(wp.float32(1.0), wp.float32(0.1), -wp.float32(0.1)))
        axis1[tid] = wp.normalize(wp.vec3f(-wp.float32(0.1), wp.float32(1.0), wp.float32(0.2)))
        axis2[tid] = wp.normalize(wp.cross(axis0[tid], axis1[tid]))
        r0[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        r1[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        k_inv_diag[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        k_inv_block[tid] = wp.mat33f(
            wp.float32(0.44) + wp.float32(0.02) * phase,
            -wp.float32(0.03),
            wp.float32(0.01),
            -wp.float32(0.03),
            wp.float32(0.58) + wp.float32(0.02) * phase,
            wp.float32(0.02),
            wp.float32(0.01),
            wp.float32(0.02),
            wp.float32(0.67) + wp.float32(0.01) * phase,
        )
        linear_mode[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        cross_mode[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        angular_mode[tid] = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
        mass_coeff[tid] = wp.vec3f(wp.float32(0.86), wp.float32(0.86), wp.float32(0.86))
        impulse_coeff[tid] = wp.vec3f(wp.float32(0.08), wp.float32(0.08), wp.float32(0.08))
        lambda_min[tid] = wp.vec3f(-BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        lambda_max[tid] = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_BOUNDS
        friction_static[tid] = wp.float32(0.0)
        friction_kinetic[tid] = wp.float32(0.0)
    else:
        family[tid] = _FAMILY_PLANAR
        axis0[tid] = n
        axis1[tid] = t1
        axis2[tid] = t2
        r0[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        r1[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        k_inv_diag[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        k_inv_block[tid] = wp.mat33f(
            wp.float32(0.51) + wp.float32(0.02) * phase,
            wp.float32(0.02),
            -wp.float32(0.01),
            wp.float32(0.02),
            wp.float32(0.47) + wp.float32(0.03) * phase,
            wp.float32(0.03),
            -wp.float32(0.01),
            wp.float32(0.03),
            wp.float32(0.62) + wp.float32(0.01) * phase,
        )
        linear_mode[tid] = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
        cross_mode[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        angular_mode[tid] = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(1.0))
        mass_coeff[tid] = wp.vec3f(wp.float32(0.86), wp.float32(0.86), wp.float32(0.86))
        impulse_coeff[tid] = wp.vec3f(wp.float32(0.08), wp.float32(0.08), wp.float32(0.08))
        lambda_min[tid] = wp.vec3f(-BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        lambda_max[tid] = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_BOUNDS
        friction_static[tid] = wp.float32(0.0)
        friction_kinetic[tid] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _solve_split_kernel(
    row_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    k_inv_diag: wp.array[wp.vec3f],
    k_inv_block: wp.array[wp.mat33f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                row = row_ids[cursor]
                va = v_a[row]
                wa = w_a[row]
                vb = v_b[row]
                wb = w_b[row]
                a0 = axis0[row]
                a1 = axis1[row]
                a2 = axis2[row]
                rr0 = r0[row]
                rr1 = r1[row]
                fam = family[row]

                if fam == _FAMILY_CONTACT:
                    rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                    residual = wp.vec3f(wp.dot(a0, rel), wp.dot(a1, rel), wp.dot(a2, rel)) + bias[row]
                    op = VelocityRows3Op()
                    op.k_inv = k_inv_diag[row]
                    op.residual = residual
                    op.lambda_old = lambda_old[row]
                    op.mass_coeff = mass_coeff[row]
                    op.impulse_coeff = impulse_coeff[row]
                    op.lambda_min = lambda_min[row]
                    op.lambda_max = lambda_max[row]
                    op.projection_mode = projection_mode[row]
                    op.friction_static = friction_static[row]
                    op.friction_kinetic = friction_kinetic[row]
                    contact_update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
                    impulse = contact_update.delta[0] * a0 + contact_update.delta[1] * a1 + contact_update.delta[2] * a2
                    out_va[row] = va - inv_m_a[row] * impulse
                    out_vb[row] = vb + inv_m_b[row] * impulse
                    out_wa[row] = wa - inv_i_a[row] @ wp.cross(rr0, impulse)
                    out_wb[row] = wb + inv_i_b[row] @ wp.cross(rr1, impulse)
                    out_lambda[row] = contact_update.lambda_new
                elif fam == _FAMILY_POINT:
                    rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                    residual = wp.vec3f(wp.dot(a0, rel), wp.dot(a1, rel), wp.dot(a2, rel)) + bias[row]
                    block = VelocityBlock3()
                    block.k_inv = k_inv_block[row]
                    block.residual = residual
                    block.lambda_old = lambda_old[row]
                    block.mass_coeff = mass_coeff[row][0]
                    block.impulse_coeff = impulse_coeff[row][0]
                    point_update = block_solve_velocity_block3_projected(
                        block, _identity_block_projection(), wp.float32(1.0)
                    )
                    impulse = point_update.delta[0] * a0 + point_update.delta[1] * a1 + point_update.delta[2] * a2
                    out_va[row] = va - inv_m_a[row] * impulse
                    out_vb[row] = vb + inv_m_b[row] * impulse
                    out_wa[row] = wa - inv_i_a[row] @ wp.cross(rr0, impulse)
                    out_wb[row] = wb + inv_i_b[row] @ wp.cross(rr1, impulse)
                    out_lambda[row] = point_update.lambda_new
                elif fam == _FAMILY_ANGULAR:
                    rel_w = wb - wa
                    residual = wp.vec3f(wp.dot(a0, rel_w), wp.dot(a1, rel_w), wp.dot(a2, rel_w)) + bias[row]
                    block = VelocityBlock3()
                    block.k_inv = k_inv_block[row]
                    block.residual = residual
                    block.lambda_old = lambda_old[row]
                    block.mass_coeff = mass_coeff[row][0]
                    block.impulse_coeff = impulse_coeff[row][0]
                    angular_update = block_solve_velocity_block3_projected(
                        block, _identity_block_projection(), wp.float32(1.0)
                    )
                    angular_impulse = (
                        angular_update.delta[0] * a0 + angular_update.delta[1] * a1 + angular_update.delta[2] * a2
                    )
                    out_va[row] = va
                    out_vb[row] = vb
                    out_wa[row] = wa - inv_i_a[row] @ angular_impulse
                    out_wb[row] = wb + inv_i_b[row] @ angular_impulse
                    out_lambda[row] = angular_update.lambda_new
                else:
                    v_rel = vb - va
                    w_rel = wb - wa
                    residual = wp.vec3f(wp.dot(a0, v_rel), wp.dot(a1, w_rel), wp.dot(a2, w_rel)) + bias[row]
                    block = VelocityBlock3()
                    block.k_inv = k_inv_block[row]
                    block.residual = residual
                    block.lambda_old = lambda_old[row]
                    block.mass_coeff = mass_coeff[row][0]
                    block.impulse_coeff = impulse_coeff[row][0]
                    planar_update = block_solve_velocity_block3_projected(
                        block, _identity_block_projection(), wp.float32(1.0)
                    )
                    linear_impulse = planar_update.delta[0] * a0
                    angular_impulse = planar_update.delta[1] * a1 + planar_update.delta[2] * a2
                    out_va[row] = va - inv_m_a[row] * linear_impulse
                    out_vb[row] = vb + inv_m_b[row] * linear_impulse
                    out_wa[row] = wa - inv_i_a[row] @ angular_impulse
                    out_wb[row] = wb + inv_i_b[row] @ angular_impulse
                    out_lambda[row] = planar_update.lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


@wp.kernel(enable_backward=False)
def _solve_mixed_kernel(
    row_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    linear_mode: wp.array[wp.vec3f],
    cross_mode: wp.array[wp.vec3f],
    angular_mode: wp.array[wp.vec3f],
    k_inv_block: wp.array[wp.mat33f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                row = row_ids[cursor]
                rows = RigidFrameBlock3Mixed()
                rows.axis0 = axis0[row]
                rows.axis1 = axis1[row]
                rows.axis2 = axis2[row]
                rows.r0 = r0[row]
                rows.r1 = r1[row]
                rows.linear_mode = linear_mode[row]
                rows.cross_mode = cross_mode[row]
                rows.angular_mode = angular_mode[row]
                rows.k_inv = k_inv_block[row]
                rows.bias = bias[row]
                rows.lambda_old = lambda_old[row]
                rows.mass_coeff = mass_coeff[row]
                rows.impulse_coeff = impulse_coeff[row]
                rows.lambda_min = lambda_min[row]
                rows.lambda_max = lambda_max[row]
                rows.projection_mode = projection_mode[row]
                rows.friction_static = friction_static[row]
                rows.friction_kinetic = friction_kinetic[row]

                state = RigidFrameRows3State()
                state.v_a = v_a[row]
                state.w_a = w_a[row]
                state.v_b = v_b[row]
                state.w_b = w_b[row]
                state.inv_m_a = inv_m_a[row]
                state.inv_m_b = inv_m_b[row]
                state.inv_i_a = inv_i_a[row]
                state.inv_i_b = inv_i_b[row]

                update = block_solve_rigid_frame_block3_mixed(rows, state, wp.float32(1.0))
                out_va[row] = update.v_a
                out_wa[row] = update.w_a
                out_vb[row] = update.v_b
                out_wb[row] = update.w_b
                out_lambda[row] = update.lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


@wp.kernel(enable_backward=False)
def _solve_mixed_tagged_kernel(
    row_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    k_inv_block: wp.array[wp.mat33f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                row = row_ids[cursor]
                fam = family[row]
                linear = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
                cross = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
                angular = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
                if fam == _FAMILY_CONTACT:
                    linear = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                    cross = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                elif fam == _FAMILY_POINT:
                    linear = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                    cross = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                elif fam == _FAMILY_ANGULAR:
                    angular = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                else:
                    linear = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
                    angular = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(1.0))

                rows = RigidFrameBlock3Mixed()
                rows.axis0 = axis0[row]
                rows.axis1 = axis1[row]
                rows.axis2 = axis2[row]
                rows.r0 = r0[row]
                rows.r1 = r1[row]
                rows.linear_mode = linear
                rows.cross_mode = cross
                rows.angular_mode = angular
                rows.k_inv = k_inv_block[row]
                rows.bias = bias[row]
                rows.lambda_old = lambda_old[row]
                rows.mass_coeff = mass_coeff[row]
                rows.impulse_coeff = impulse_coeff[row]
                rows.lambda_min = lambda_min[row]
                rows.lambda_max = lambda_max[row]
                rows.projection_mode = projection_mode[row]
                rows.friction_static = friction_static[row]
                rows.friction_kinetic = friction_kinetic[row]

                state = RigidFrameRows3State()
                state.v_a = v_a[row]
                state.w_a = w_a[row]
                state.v_b = v_b[row]
                state.w_b = w_b[row]
                state.inv_m_a = inv_m_a[row]
                state.inv_m_b = inv_m_b[row]
                state.inv_i_a = inv_i_a[row]
                state.inv_i_b = inv_i_b[row]

                update = block_solve_rigid_frame_block3_mixed(rows, state, wp.float32(1.0))
                out_va[row] = update.v_a
                out_wa[row] = update.w_a
                out_vb[row] = update.v_b
                out_wb[row] = update.w_b
                out_lambda[row] = update.lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


@wp.kernel(enable_backward=False)
def _solve_hybrid_contact_mixed_kernel(
    row_ids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.mat33f],
    inv_i_b: wp.array[wp.mat33f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    k_inv_diag: wp.array[wp.vec3f],
    k_inv_block: wp.array[wp.mat33f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_va: wp.array[wp.vec3f],
    out_wa: wp.array[wp.vec3f],
    out_vb: wp.array[wp.vec3f],
    out_wb: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    num_worlds: wp.int32,
    iterations: wp.int32,
    threads_per_world: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    epoch = wp.int32(0)
    while epoch < iterations:
        color = color_begin
        while color < color_end:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + local_tid
            while cursor < end:
                row = row_ids[cursor]
                fam = family[row]
                va = v_a[row]
                wa = w_a[row]
                vb = v_b[row]
                wb = w_b[row]
                a0 = axis0[row]
                a1 = axis1[row]
                a2 = axis2[row]
                rr0 = r0[row]
                rr1 = r1[row]

                if fam == _FAMILY_CONTACT:
                    rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
                    residual = wp.vec3f(wp.dot(a0, rel), wp.dot(a1, rel), wp.dot(a2, rel)) + bias[row]
                    op = VelocityRows3Op()
                    op.k_inv = k_inv_diag[row]
                    op.residual = residual
                    op.lambda_old = lambda_old[row]
                    op.mass_coeff = mass_coeff[row]
                    op.impulse_coeff = impulse_coeff[row]
                    op.lambda_min = lambda_min[row]
                    op.lambda_max = lambda_max[row]
                    op.projection_mode = projection_mode[row]
                    op.friction_static = friction_static[row]
                    op.friction_kinetic = friction_kinetic[row]
                    contact_update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
                    impulse = contact_update.delta[0] * a0 + contact_update.delta[1] * a1 + contact_update.delta[2] * a2
                    out_va[row] = va - inv_m_a[row] * impulse
                    out_vb[row] = vb + inv_m_b[row] * impulse
                    out_wa[row] = wa - inv_i_a[row] @ wp.cross(rr0, impulse)
                    out_wb[row] = wb + inv_i_b[row] @ wp.cross(rr1, impulse)
                    out_lambda[row] = contact_update.lambda_new
                else:
                    linear = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
                    cross = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
                    angular = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
                    if fam == _FAMILY_POINT:
                        linear = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                        cross = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                    elif fam == _FAMILY_ANGULAR:
                        angular = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
                    else:
                        linear = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
                        angular = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(1.0))

                    rows = RigidFrameBlock3Mixed()
                    rows.axis0 = a0
                    rows.axis1 = a1
                    rows.axis2 = a2
                    rows.r0 = rr0
                    rows.r1 = rr1
                    rows.linear_mode = linear
                    rows.cross_mode = cross
                    rows.angular_mode = angular
                    rows.k_inv = k_inv_block[row]
                    rows.bias = bias[row]
                    rows.lambda_old = lambda_old[row]
                    rows.mass_coeff = mass_coeff[row]
                    rows.impulse_coeff = impulse_coeff[row]
                    rows.lambda_min = lambda_min[row]
                    rows.lambda_max = lambda_max[row]
                    rows.projection_mode = projection_mode[row]
                    rows.friction_static = friction_static[row]
                    rows.friction_kinetic = friction_kinetic[row]

                    state = RigidFrameRows3State()
                    state.v_a = va
                    state.w_a = wa
                    state.v_b = vb
                    state.w_b = wb
                    state.inv_m_a = inv_m_a[row]
                    state.inv_m_b = inv_m_b[row]
                    state.inv_i_a = inv_i_a[row]
                    state.inv_i_b = inv_i_b[row]

                    mixed_update = block_solve_rigid_frame_block3_mixed(rows, state, wp.float32(1.0))
                    out_va[row] = mixed_update.v_a
                    out_wa[row] = mixed_update.w_a
                    out_vb[row] = mixed_update.v_b
                    out_wb[row] = mixed_update.w_b
                    out_lambda[row] = mixed_update.lambda_new
                cursor = cursor + threads_per_world
            color = color + wp.int32(1)
        epoch = epoch + wp.int32(1)


def _parse_scenes(value: str) -> tuple[str, ...]:
    scenes = tuple(raw.strip() for raw in value.split(",") if raw.strip())
    for scene in scenes:
        if scene not in _SCENE_PRESETS:
            raise ValueError(f"unknown scene {scene!r}; choices={tuple(_SCENE_PRESETS)}")
    return scenes


def _slot_counts(preset: ScenePreset, period: int) -> tuple[int, int, int, int]:
    ratios = np.asarray(
        [preset.contact_ratio, preset.point_ratio, preset.angular_ratio, preset.planar_ratio], dtype=np.float64
    )
    total = float(ratios.sum())
    if total <= 0.0:
        return 0, 0, 0, period
    counts = np.rint(ratios / total * float(period)).astype(np.int32)
    while int(counts.sum()) > period:
        counts[int(np.argmax(counts))] -= 1
    while int(counts.sum()) < period:
        counts[int(np.argmax(ratios))] += 1
    return int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])


def _run_scene(args: argparse.Namespace, scene: str, device: wp.context.Devicelike) -> None:
    preset = _SCENE_PRESETS[scene]
    schedule = _build_schedule(preset, int(args.worlds))
    rows = int(schedule.rows)
    contact_slots, point_slots, angular_slots, planar_slots = _slot_counts(preset, int(args.period))

    row_ids = wp.array(schedule.row_ids, dtype=wp.int32, device=device)
    color_starts = wp.array(schedule.color_starts, dtype=wp.int32, device=device)
    world_color_starts = wp.array(schedule.world_color_starts, dtype=wp.int32, device=device)

    family = wp.empty(rows, dtype=wp.int32, device=device)
    v_a = _alloc_vec(rows, device)
    w_a = _alloc_vec(rows, device)
    v_b = _alloc_vec(rows, device)
    w_b = _alloc_vec(rows, device)
    inv_m_a = wp.empty(rows, dtype=wp.float32, device=device)
    inv_m_b = wp.empty(rows, dtype=wp.float32, device=device)
    inv_i_a = _alloc_mat(rows, device)
    inv_i_b = _alloc_mat(rows, device)
    axis0 = _alloc_vec(rows, device)
    axis1 = _alloc_vec(rows, device)
    axis2 = _alloc_vec(rows, device)
    r0 = _alloc_vec(rows, device)
    r1 = _alloc_vec(rows, device)
    linear_mode = _alloc_vec(rows, device)
    cross_mode = _alloc_vec(rows, device)
    angular_mode = _alloc_vec(rows, device)
    k_inv_diag = _alloc_vec(rows, device)
    k_inv_block = _alloc_mat(rows, device)
    bias = _alloc_vec(rows, device)
    lambda_old = _alloc_vec(rows, device)
    mass_coeff = _alloc_vec(rows, device)
    impulse_coeff = _alloc_vec(rows, device)
    lambda_min = _alloc_vec(rows, device)
    lambda_max = _alloc_vec(rows, device)
    projection_mode = wp.empty(rows, dtype=wp.int32, device=device)
    friction_static = wp.empty(rows, dtype=wp.float32, device=device)
    friction_kinetic = wp.empty(rows, dtype=wp.float32, device=device)

    split_va = _alloc_vec(rows, device)
    split_wa = _alloc_vec(rows, device)
    split_vb = _alloc_vec(rows, device)
    split_wb = _alloc_vec(rows, device)
    split_lambda = _alloc_vec(rows, device)
    mixed_va = _alloc_vec(rows, device)
    mixed_wa = _alloc_vec(rows, device)
    mixed_vb = _alloc_vec(rows, device)
    mixed_wb = _alloc_vec(rows, device)
    mixed_lambda = _alloc_vec(rows, device)
    tagged_va = _alloc_vec(rows, device)
    tagged_wa = _alloc_vec(rows, device)
    tagged_vb = _alloc_vec(rows, device)
    tagged_wb = _alloc_vec(rows, device)
    tagged_lambda = _alloc_vec(rows, device)
    hybrid_va = _alloc_vec(rows, device)
    hybrid_wa = _alloc_vec(rows, device)
    hybrid_vb = _alloc_vec(rows, device)
    hybrid_wb = _alloc_vec(rows, device)
    hybrid_lambda = _alloc_vec(rows, device)

    wp.launch(
        _init_mixed_frame_kernel,
        dim=rows,
        inputs=[
            family,
            v_a,
            w_a,
            v_b,
            w_b,
            inv_m_a,
            inv_m_b,
            inv_i_a,
            inv_i_b,
            axis0,
            axis1,
            axis2,
            r0,
            r1,
            linear_mode,
            cross_mode,
            angular_mode,
            k_inv_diag,
            k_inv_block,
            bias,
            lambda_old,
            mass_coeff,
            impulse_coeff,
            lambda_min,
            lambda_max,
            projection_mode,
            friction_static,
            friction_kinetic,
            int(args.period),
            contact_slots,
            point_slots,
            angular_slots,
        ],
        device=device,
    )

    split_inputs = [
        row_ids,
        color_starts,
        world_color_starts,
        family,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        k_inv_diag,
        k_inv_block,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        split_va,
        split_wa,
        split_vb,
        split_wb,
        split_lambda,
        int(args.worlds),
        int(args.iterations),
        int(args.threads_per_world),
    ]
    mixed_inputs = [
        row_ids,
        color_starts,
        world_color_starts,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        linear_mode,
        cross_mode,
        angular_mode,
        k_inv_block,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        mixed_va,
        mixed_wa,
        mixed_vb,
        mixed_wb,
        mixed_lambda,
        int(args.worlds),
        int(args.iterations),
        int(args.threads_per_world),
    ]
    tagged_inputs = [
        row_ids,
        color_starts,
        world_color_starts,
        family,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        k_inv_block,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        tagged_va,
        tagged_wa,
        tagged_vb,
        tagged_wb,
        tagged_lambda,
        int(args.worlds),
        int(args.iterations),
        int(args.threads_per_world),
    ]
    hybrid_inputs = [
        row_ids,
        color_starts,
        world_color_starts,
        family,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        axis0,
        axis1,
        axis2,
        r0,
        r1,
        k_inv_diag,
        k_inv_block,
        bias,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        hybrid_va,
        hybrid_wa,
        hybrid_vb,
        hybrid_wb,
        hybrid_lambda,
        int(args.worlds),
        int(args.iterations),
        int(args.threads_per_world),
    ]

    def split_run() -> None:
        wp.launch(
            _solve_split_kernel,
            dim=max(1, int(args.worlds) * int(args.threads_per_world)),
            inputs=split_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def mixed_run() -> None:
        wp.launch(
            _solve_mixed_kernel,
            dim=max(1, int(args.worlds) * int(args.threads_per_world)),
            inputs=mixed_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def tagged_run() -> None:
        wp.launch(
            _solve_mixed_tagged_kernel,
            dim=max(1, int(args.worlds) * int(args.threads_per_world)),
            inputs=tagged_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def hybrid_run() -> None:
        wp.launch(
            _solve_hybrid_contact_mixed_kernel,
            dim=max(1, int(args.worlds) * int(args.threads_per_world)),
            inputs=hybrid_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    split_run()
    mixed_run()
    tagged_run()
    hybrid_run()
    mixed_err = _max_err(
        (mixed_va, split_va),
        (mixed_wa, split_wa),
        (mixed_vb, split_vb),
        (mixed_wb, split_wb),
        (mixed_lambda, split_lambda),
    )
    tagged_err = _max_err(
        (tagged_va, split_va),
        (tagged_wa, split_wa),
        (tagged_vb, split_vb),
        (tagged_wb, split_wb),
        (tagged_lambda, split_lambda),
    )
    hybrid_err = _max_err(
        (hybrid_va, split_va),
        (hybrid_wa, split_wa),
        (hybrid_vb, split_vb),
        (hybrid_wb, split_wb),
        (hybrid_lambda, split_lambda),
    )
    split_ms, _ = _bench(split_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    mixed_ms, _ = _bench(mixed_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    tagged_ms, _ = _bench(tagged_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    hybrid_ms, _ = _bench(hybrid_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
    mixed_speedup = split_ms / mixed_ms if mixed_ms > 0.0 else float("nan")
    tagged_speedup = split_ms / tagged_ms if tagged_ms > 0.0 else float("nan")
    hybrid_speedup = split_ms / hybrid_ms if hybrid_ms > 0.0 else float("nan")
    print(
        f"{scene:8s} worlds={int(args.worlds):5d} rows={rows:7d} colors={schedule.colors:5d} "
        f"slots=(c{contact_slots},p{point_slots},a{angular_slots},m{planar_slots}) "
        f"split={split_ms:8.4f}ms mixed={mixed_ms:8.4f}ms tagged={tagged_ms:8.4f}ms "
        f"hybrid={hybrid_ms:8.4f}ms speedups=({mixed_speedup:6.3f}x,{tagged_speedup:6.3f}x,{hybrid_speedup:6.3f}x) "
        f"errs=({mixed_err:.6g},{tagged_err:.6g},{hybrid_err:.6g})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", default="h1,g1,dr_legs,tower")
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--threads-per-world", type=int, default=32)
    parser.add_argument("--period", type=int, default=32)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    print(
        f"device={device} worlds={args.worlds} iterations={args.iterations} "
        f"tpw={args.threads_per_world} n_runs={args.n_runs} trials={args.trials}"
    )
    for scene in _parse_scenes(args.scenes):
        _run_scene(args, scene, device)


if __name__ == "__main__":
    main()
