# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental rigid ``VelocityRows3Op`` sidecar benchmark.

This prototypes the larger PhoenX unification target: one prepared rigid row
block whose iterate step performs the same residual fetch, projection, and
``M^-1 J^T`` apply whether the source row came from a contact or a joint.

The benchmark is intentionally isolated from graph coloring: each operation owns
its own synthetic body pair, so timings measure the unified fetch/apply shape
rather than write conflicts. It compares:

* ``split``: a mixed typed kernel with a family branch for contact vs angular
  joint rows.
* ``dense``: a generic packed-Jacobian kernel with no contact/joint fetch or
  apply branch. Projection still uses the current ``VelocityRows3Op`` path.
* ``frame``: a compact branchless descriptor using three axes, two offsets,
  and mode coefficients for contact-style and angular-direct rows.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    VELOCITY_ROWS3_PROJECT_BOUNDS,
    VELOCITY_ROWS3_PROJECT_CONTACT_CONE,
    VelocityRows3Op,
    block_solve_velocity_rows3_op,
)

_FAMILY_CONTACT = wp.constant(wp.int32(1))
_FAMILY_ANGULAR = wp.constant(wp.int32(2))


@wp.func
def _mul_diag(diag: wp.vec3f, x: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(diag[0] * x[0], diag[1] * x[1], diag[2] * x[2])


@wp.func
def _rows3_dot(
    row0: wp.vec3f,
    row1: wp.vec3f,
    row2: wp.vec3f,
    x: wp.vec3f,
) -> wp.vec3f:
    return wp.vec3f(wp.dot(row0, x), wp.dot(row1, x), wp.dot(row2, x))


@wp.func
def _rows3_t_mul(
    row0: wp.vec3f,
    row1: wp.vec3f,
    row2: wp.vec3f,
    d: wp.vec3f,
) -> wp.vec3f:
    return row0 * d[0] + row1 * d[1] + row2 * d[2]


@wp.func
def _make_projection_op(
    k_inv: wp.vec3f,
    residual: wp.vec3f,
    lambda_old: wp.vec3f,
    lambda_min: wp.vec3f,
    lambda_max: wp.vec3f,
    projection_mode: wp.int32,
    friction_static: wp.float32,
    friction_kinetic: wp.float32,
) -> VelocityRows3Op:
    op = VelocityRows3Op()
    op.k_inv = k_inv
    op.residual = residual
    op.lambda_old = lambda_old
    op.mass_coeff = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
    op.impulse_coeff = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    op.lambda_min = lambda_min
    op.lambda_max = lambda_max
    op.projection_mode = projection_mode
    op.friction_static = friction_static
    op.friction_kinetic = friction_kinetic
    return op


@wp.kernel
def _init_rows_kernel(
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.vec3f],
    inv_i_b: wp.array[wp.vec3f],
    normal: wp.array[wp.vec3f],
    tangent1: wp.array[wp.vec3f],
    tangent2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    jla0: wp.array[wp.vec3f],
    jla1: wp.array[wp.vec3f],
    jla2: wp.array[wp.vec3f],
    jaa0: wp.array[wp.vec3f],
    jaa1: wp.array[wp.vec3f],
    jaa2: wp.array[wp.vec3f],
    jlb0: wp.array[wp.vec3f],
    jlb1: wp.array[wp.vec3f],
    jlb2: wp.array[wp.vec3f],
    jab0: wp.array[wp.vec3f],
    jab1: wp.array[wp.vec3f],
    jab2: wp.array[wp.vec3f],
    k_inv: wp.array[wp.vec3f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    frame_mode: wp.array[wp.vec3f],
    period: wp.int32,
    contacts_per_period: wp.int32,
):
    tid = wp.tid()
    lane = tid % period
    is_contact = lane < contacts_per_period
    phase = wp.float32((tid * wp.int32(29)) & wp.int32(255)) * wp.float32(0.00390625)

    va = wp.vec3f(phase, wp.float32(0.2) - phase, wp.float32(0.1) + wp.float32(0.5) * phase)
    wa = wp.vec3f(wp.float32(0.3) * phase, wp.float32(0.4) - phase, wp.float32(0.2) * phase)
    vb = wp.vec3f(wp.float32(0.15) - phase, wp.float32(0.1) + phase, wp.float32(0.25) * phase)
    wb = wp.vec3f(wp.float32(0.2) + wp.float32(0.25) * phase, -wp.float32(0.1) * phase, phase)
    v_a[tid] = va
    w_a[tid] = wa
    v_b[tid] = vb
    w_b[tid] = wb
    inv_m_a[tid] = wp.float32(0.8) + wp.float32(0.1) * phase
    inv_m_b[tid] = wp.float32(0.7) + wp.float32(0.2) * phase
    inv_i_a[tid] = wp.vec3f(wp.float32(0.6), wp.float32(0.7), wp.float32(0.8))
    inv_i_b[tid] = wp.vec3f(wp.float32(0.9), wp.float32(0.65), wp.float32(0.75))

    n = wp.normalize(wp.vec3f(wp.float32(1.0), wp.float32(0.2), wp.float32(0.1)))
    t1 = wp.normalize(wp.vec3f(-wp.float32(0.2), wp.float32(1.0), wp.float32(0.0)))
    t2 = wp.normalize(wp.cross(n, t1))
    rr0 = wp.vec3f(wp.float32(0.10), wp.float32(0.03) + wp.float32(0.02) * phase, -wp.float32(0.05))
    rr1 = wp.vec3f(-wp.float32(0.08), wp.float32(0.04), wp.float32(0.06) + wp.float32(0.01) * phase)
    normal[tid] = n
    tangent1[tid] = t1
    tangent2[tid] = t2
    r0[tid] = rr0
    r1[tid] = rr1

    a0 = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
    a1 = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(0.0))
    a2 = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
    axis0[tid] = a0
    axis1[tid] = a1
    axis2[tid] = a2

    k_inv[tid] = wp.vec3f(
        wp.float32(0.35) + wp.float32(0.1) * phase,
        wp.float32(0.45) + wp.float32(0.1) * phase,
        wp.float32(0.55) + wp.float32(0.1) * phase,
    )
    bias[tid] = wp.vec3f(
        wp.float32(0.02) - wp.float32(0.03) * phase,
        wp.float32(0.01) + wp.float32(0.02) * phase,
        -wp.float32(0.015) * phase,
    )
    lambda_old[tid] = wp.vec3f(
        wp.float32(0.05) + wp.float32(0.02) * phase,
        wp.float32(0.01) - wp.float32(0.02) * phase,
        wp.float32(0.015) * phase,
    )

    if is_contact:
        family[tid] = _FAMILY_CONTACT
        axis0[tid] = n
        axis1[tid] = t1
        axis2[tid] = t2
        frame_mode[tid] = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(0.0))
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_CONTACT_CONE
        lambda_min[tid] = wp.vec3f(wp.float32(0.0), -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        lambda_max[tid] = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        friction_static[tid] = wp.float32(0.8)
        friction_kinetic[tid] = wp.float32(0.6)

        jla0[tid] = -n
        jla1[tid] = -t1
        jla2[tid] = -t2
        jaa0[tid] = -wp.cross(rr0, n)
        jaa1[tid] = -wp.cross(rr0, t1)
        jaa2[tid] = -wp.cross(rr0, t2)
        jlb0[tid] = n
        jlb1[tid] = t1
        jlb2[tid] = t2
        jab0[tid] = wp.cross(rr1, n)
        jab1[tid] = wp.cross(rr1, t1)
        jab2[tid] = wp.cross(rr1, t2)
    else:
        family[tid] = _FAMILY_ANGULAR
        frame_mode[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_BOUNDS
        lambda_min[tid] = wp.vec3f(-wp.float32(0.6), wp.float32(0.0), -wp.float32(0.25))
        lambda_max[tid] = wp.vec3f(wp.float32(0.6), BLOCK_LAMBDA_INF, wp.float32(0.25))
        friction_static[tid] = wp.float32(0.0)
        friction_kinetic[tid] = wp.float32(0.0)

        zero = wp.vec3f(wp.float32(0.0))
        jla0[tid] = zero
        jla1[tid] = zero
        jla2[tid] = zero
        jaa0[tid] = -a0
        jaa1[tid] = -a1
        jaa2[tid] = -a2
        jlb0[tid] = zero
        jlb1[tid] = zero
        jlb2[tid] = zero
        jab0[tid] = a0
        jab1[tid] = a1
        jab2[tid] = a2


@wp.kernel
def _solve_sidecar_kernel(
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.vec3f],
    inv_i_b: wp.array[wp.vec3f],
    jla0: wp.array[wp.vec3f],
    jla1: wp.array[wp.vec3f],
    jla2: wp.array[wp.vec3f],
    jaa0: wp.array[wp.vec3f],
    jaa1: wp.array[wp.vec3f],
    jaa2: wp.array[wp.vec3f],
    jlb0: wp.array[wp.vec3f],
    jlb1: wp.array[wp.vec3f],
    jlb2: wp.array[wp.vec3f],
    jab0: wp.array[wp.vec3f],
    jab1: wp.array[wp.vec3f],
    jab2: wp.array[wp.vec3f],
    k_inv: wp.array[wp.vec3f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
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
):
    tid = wp.tid()
    va = v_a[tid]
    wa = w_a[tid]
    vb = v_b[tid]
    wb = w_b[tid]

    residual = (
        _rows3_dot(jla0[tid], jla1[tid], jla2[tid], va)
        + _rows3_dot(jaa0[tid], jaa1[tid], jaa2[tid], wa)
        + _rows3_dot(jlb0[tid], jlb1[tid], jlb2[tid], vb)
        + _rows3_dot(jab0[tid], jab1[tid], jab2[tid], wb)
        + bias[tid]
    )
    op = _make_projection_op(
        k_inv[tid],
        residual,
        lambda_old[tid],
        lambda_min[tid],
        lambda_max[tid],
        projection_mode[tid],
        friction_static[tid],
        friction_kinetic[tid],
    )
    update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
    d = update.delta

    out_va[tid] = va + inv_m_a[tid] * _rows3_t_mul(jla0[tid], jla1[tid], jla2[tid], d)
    out_wa[tid] = wa + _mul_diag(inv_i_a[tid], _rows3_t_mul(jaa0[tid], jaa1[tid], jaa2[tid], d))
    out_vb[tid] = vb + inv_m_b[tid] * _rows3_t_mul(jlb0[tid], jlb1[tid], jlb2[tid], d)
    out_wb[tid] = wb + _mul_diag(inv_i_b[tid], _rows3_t_mul(jab0[tid], jab1[tid], jab2[tid], d))
    out_lambda[tid] = update.lambda_new


@wp.kernel
def _solve_frame_kernel(
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.vec3f],
    inv_i_b: wp.array[wp.vec3f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    frame_mode: wp.array[wp.vec3f],
    k_inv: wp.array[wp.vec3f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
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
):
    tid = wp.tid()
    va = v_a[tid]
    wa = w_a[tid]
    vb = v_b[tid]
    wb = w_b[tid]
    a0 = axis0[tid]
    a1 = axis1[tid]
    a2 = axis2[tid]
    rr0 = r0[tid]
    rr1 = r1[tid]
    mode = frame_mode[tid]
    linear_scale = mode[0]
    cross_scale = mode[1]
    angular_scale = mode[2]

    rel = linear_scale * (vb - va) + cross_scale * (wp.cross(wb, rr1) - wp.cross(wa, rr0)) + angular_scale * (wb - wa)
    residual = _rows3_dot(a0, a1, a2, rel) + bias[tid]

    op = _make_projection_op(
        k_inv[tid],
        residual,
        lambda_old[tid],
        lambda_min[tid],
        lambda_max[tid],
        projection_mode[tid],
        friction_static[tid],
        friction_kinetic[tid],
    )
    update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
    d = update.delta
    impulse = d[0] * a0 + d[1] * a1 + d[2] * a2

    out_va[tid] = va - linear_scale * inv_m_a[tid] * impulse
    out_vb[tid] = vb + linear_scale * inv_m_b[tid] * impulse
    out_wa[tid] = wa + _mul_diag(inv_i_a[tid], -cross_scale * wp.cross(rr0, impulse) - angular_scale * impulse)
    out_wb[tid] = wb + _mul_diag(inv_i_b[tid], cross_scale * wp.cross(rr1, impulse) + angular_scale * impulse)
    out_lambda[tid] = update.lambda_new


@wp.kernel
def _solve_split_kernel(
    family: wp.array[wp.int32],
    v_a: wp.array[wp.vec3f],
    w_a: wp.array[wp.vec3f],
    v_b: wp.array[wp.vec3f],
    w_b: wp.array[wp.vec3f],
    inv_m_a: wp.array[wp.float32],
    inv_m_b: wp.array[wp.float32],
    inv_i_a: wp.array[wp.vec3f],
    inv_i_b: wp.array[wp.vec3f],
    normal: wp.array[wp.vec3f],
    tangent1: wp.array[wp.vec3f],
    tangent2: wp.array[wp.vec3f],
    r0: wp.array[wp.vec3f],
    r1: wp.array[wp.vec3f],
    axis0: wp.array[wp.vec3f],
    axis1: wp.array[wp.vec3f],
    axis2: wp.array[wp.vec3f],
    k_inv: wp.array[wp.vec3f],
    bias: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
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
):
    tid = wp.tid()
    va = v_a[tid]
    wa = w_a[tid]
    vb = v_b[tid]
    wb = w_b[tid]

    if family[tid] == _FAMILY_CONTACT:
        n = normal[tid]
        t1 = tangent1[tid]
        t2 = tangent2[tid]
        rr0 = r0[tid]
        rr1 = r1[tid]
        rel = vb + wp.cross(wb, rr1) - va - wp.cross(wa, rr0)
        residual = wp.vec3f(wp.dot(n, rel), wp.dot(t1, rel), wp.dot(t2, rel)) + bias[tid]
        op = _make_projection_op(
            k_inv[tid],
            residual,
            lambda_old[tid],
            lambda_min[tid],
            lambda_max[tid],
            projection_mode[tid],
            friction_static[tid],
            friction_kinetic[tid],
        )
        update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
        impulse = update.delta[0] * n + update.delta[1] * t1 + update.delta[2] * t2
        out_va[tid] = va - inv_m_a[tid] * impulse
        out_wa[tid] = wa - _mul_diag(inv_i_a[tid], wp.cross(rr0, impulse))
        out_vb[tid] = vb + inv_m_b[tid] * impulse
        out_wb[tid] = wb + _mul_diag(inv_i_b[tid], wp.cross(rr1, impulse))
        out_lambda[tid] = update.lambda_new
    else:
        a0 = axis0[tid]
        a1 = axis1[tid]
        a2 = axis2[tid]
        rel_w = wb - wa
        residual = wp.vec3f(wp.dot(a0, rel_w), wp.dot(a1, rel_w), wp.dot(a2, rel_w)) + bias[tid]
        op = _make_projection_op(
            k_inv[tid],
            residual,
            lambda_old[tid],
            lambda_min[tid],
            lambda_max[tid],
            projection_mode[tid],
            friction_static[tid],
            friction_kinetic[tid],
        )
        update = block_solve_velocity_rows3_op(op, wp.float32(1.0))
        angular_impulse = update.delta[0] * a0 + update.delta[1] * a1 + update.delta[2] * a2
        out_va[tid] = va
        out_wa[tid] = wa - _mul_diag(inv_i_a[tid], angular_impulse)
        out_vb[tid] = vb
        out_wb[tid] = wb + _mul_diag(inv_i_b[tid], angular_impulse)
        out_lambda[tid] = update.lambda_new


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(raw.strip()) for raw in value.split(",") if raw.strip())


def _bench(fn, *, n_runs: int, warmup: int, trials: int, device: wp.context.Devicelike) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    wp.synchronize_device()
    with wp.ScopedCapture(device=device) as capture:
        fn()
    graph = capture.graph
    wp.synchronize_device()

    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            wp.capture_launch(graph)
        wp.synchronize_device()
        times.append((time.perf_counter() - t0) * 1000.0 / float(n_runs))
    arr = np.asarray(times)
    return float(arr.min()), float(np.median(arr))


def _alloc_vec(rows: int, device: wp.context.Devicelike):
    return wp.empty(rows, dtype=wp.vec3f, device=device)


def _max_err(*pairs) -> float:
    err = 0.0
    for left, right in pairs:
        err = max(err, float(np.max(np.abs(left.numpy() - right.numpy()))))
    return err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--ratios", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--period", type=int, default=32)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    rows = int(args.rows)

    family = wp.empty(rows, dtype=wp.int32, device=device)
    v_a = _alloc_vec(rows, device)
    w_a = _alloc_vec(rows, device)
    v_b = _alloc_vec(rows, device)
    w_b = _alloc_vec(rows, device)
    inv_m_a = wp.empty(rows, dtype=wp.float32, device=device)
    inv_m_b = wp.empty(rows, dtype=wp.float32, device=device)
    inv_i_a = _alloc_vec(rows, device)
    inv_i_b = _alloc_vec(rows, device)
    normal = _alloc_vec(rows, device)
    tangent1 = _alloc_vec(rows, device)
    tangent2 = _alloc_vec(rows, device)
    r0 = _alloc_vec(rows, device)
    r1 = _alloc_vec(rows, device)
    axis0 = _alloc_vec(rows, device)
    axis1 = _alloc_vec(rows, device)
    axis2 = _alloc_vec(rows, device)
    jla0 = _alloc_vec(rows, device)
    jla1 = _alloc_vec(rows, device)
    jla2 = _alloc_vec(rows, device)
    jaa0 = _alloc_vec(rows, device)
    jaa1 = _alloc_vec(rows, device)
    jaa2 = _alloc_vec(rows, device)
    jlb0 = _alloc_vec(rows, device)
    jlb1 = _alloc_vec(rows, device)
    jlb2 = _alloc_vec(rows, device)
    jab0 = _alloc_vec(rows, device)
    jab1 = _alloc_vec(rows, device)
    jab2 = _alloc_vec(rows, device)
    k_inv = _alloc_vec(rows, device)
    bias = _alloc_vec(rows, device)
    lambda_old = _alloc_vec(rows, device)
    lambda_min = _alloc_vec(rows, device)
    lambda_max = _alloc_vec(rows, device)
    projection_mode = wp.empty(rows, dtype=wp.int32, device=device)
    friction_static = wp.empty(rows, dtype=wp.float32, device=device)
    friction_kinetic = wp.empty(rows, dtype=wp.float32, device=device)
    frame_mode = _alloc_vec(rows, device)

    side_va = _alloc_vec(rows, device)
    side_wa = _alloc_vec(rows, device)
    side_vb = _alloc_vec(rows, device)
    side_wb = _alloc_vec(rows, device)
    side_lambda = _alloc_vec(rows, device)
    split_va = _alloc_vec(rows, device)
    split_wa = _alloc_vec(rows, device)
    split_vb = _alloc_vec(rows, device)
    split_wb = _alloc_vec(rows, device)
    split_lambda = _alloc_vec(rows, device)
    frame_va = _alloc_vec(rows, device)
    frame_wa = _alloc_vec(rows, device)
    frame_vb = _alloc_vec(rows, device)
    frame_wb = _alloc_vec(rows, device)
    frame_lambda = _alloc_vec(rows, device)

    init_inputs = [
        family,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        normal,
        tangent1,
        tangent2,
        r0,
        r1,
        axis0,
        axis1,
        axis2,
        jla0,
        jla1,
        jla2,
        jaa0,
        jaa1,
        jaa2,
        jlb0,
        jlb1,
        jlb2,
        jab0,
        jab1,
        jab2,
        k_inv,
        bias,
        lambda_old,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        frame_mode,
    ]
    sidecar_inputs = [
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        jla0,
        jla1,
        jla2,
        jaa0,
        jaa1,
        jaa2,
        jlb0,
        jlb1,
        jlb2,
        jab0,
        jab1,
        jab2,
        k_inv,
        bias,
        lambda_old,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        side_va,
        side_wa,
        side_vb,
        side_wb,
        side_lambda,
    ]
    frame_inputs = [
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
        frame_mode,
        k_inv,
        bias,
        lambda_old,
        lambda_min,
        lambda_max,
        projection_mode,
        friction_static,
        friction_kinetic,
        frame_va,
        frame_wa,
        frame_vb,
        frame_wb,
        frame_lambda,
    ]
    split_inputs = [
        family,
        v_a,
        w_a,
        v_b,
        w_b,
        inv_m_a,
        inv_m_b,
        inv_i_a,
        inv_i_b,
        normal,
        tangent1,
        tangent2,
        r0,
        r1,
        axis0,
        axis1,
        axis2,
        k_inv,
        bias,
        lambda_old,
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
    ]

    def sidecar_run():
        wp.launch(
            _solve_sidecar_kernel,
            dim=rows,
            inputs=sidecar_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def frame_run():
        wp.launch(
            _solve_frame_kernel,
            dim=rows,
            inputs=frame_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    def split_run():
        wp.launch(
            _solve_split_kernel,
            dim=rows,
            inputs=split_inputs,
            device=device,
            block_dim=int(args.block_dim),
        )

    print("contact_ratio,split_ms,dense_ms,frame_ms,dense_speedup,frame_speedup,dense_err,frame_err")
    for ratio in _parse_csv_floats(args.ratios):
        contacts_per_period = int(round(max(0.0, min(1.0, ratio)) * float(args.period)))
        wp.launch(
            _init_rows_kernel,
            dim=rows,
            inputs=[*init_inputs, int(args.period), contacts_per_period],
            device=device,
        )
        split_run()
        sidecar_run()
        frame_run()
        dense_err = _max_err(
            (side_va, split_va),
            (side_wa, split_wa),
            (side_vb, split_vb),
            (side_wb, split_wb),
            (side_lambda, split_lambda),
        )
        frame_err = _max_err(
            (frame_va, split_va),
            (frame_wa, split_wa),
            (frame_vb, split_vb),
            (frame_wb, split_wb),
            (frame_lambda, split_lambda),
        )
        split_ms, _ = _bench(split_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
        side_ms, _ = _bench(sidecar_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
        frame_ms, _ = _bench(frame_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
        dense_speedup = split_ms / side_ms if side_ms > 0.0 else float("nan")
        frame_speedup = split_ms / frame_ms if frame_ms > 0.0 else float("nan")
        print(
            f"{ratio:.2f},{split_ms:.6f},{side_ms:.6f},{frame_ms:.6f},"
            f"{dense_speedup:.3f},{frame_speedup:.3f},{dense_err:.6g},{frame_err:.6g}"
        )


if __name__ == "__main__":
    main()
