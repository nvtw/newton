# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental benchmark for unified three-row velocity projection.

This isolates one narrow question in the PhoenX unification work: if contacts
and bounded joint/motor rows share the same ``VelocityRows3Op`` shape, is it
profitable to remove the projection-mode branch inside the op?

The benchmark intentionally keeps the data synthetic but varies the contact /
bounded-row mix. It runs under graph capture, matching the production
constraint-solve requirement.
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
    block_solve_velocity_rows3_op_uniform_projection,
)


@wp.kernel
def _init_ops_kernel(
    k_inv: wp.array[wp.vec3f],
    residual: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    period: wp.int32,
    contacts_per_period: wp.int32,
):
    tid = wp.tid()
    lane = tid % period
    is_contact = lane < contacts_per_period

    phase = wp.float32((tid * wp.int32(17)) & wp.int32(255)) * wp.float32(0.00390625)
    k_inv[tid] = wp.vec3f(
        wp.float32(0.25) + phase,
        wp.float32(0.35) + wp.float32(0.5) * phase,
        wp.float32(0.45) + wp.float32(0.25) * phase,
    )
    residual[tid] = wp.vec3f(
        phase - wp.float32(0.45),
        wp.float32(0.8) * phase - wp.float32(0.25),
        wp.float32(0.6) * phase - wp.float32(0.15),
    )
    lambda_old[tid] = wp.vec3f(
        wp.float32(0.05) + wp.float32(0.02) * phase,
        wp.float32(0.02) - wp.float32(0.04) * phase,
        wp.float32(0.03) * phase - wp.float32(0.01),
    )
    mass_coeff[tid] = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
    impulse_coeff[tid] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))

    if is_contact:
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_CONTACT_CONE
        lambda_min[tid] = wp.vec3f(wp.float32(0.0), -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        lambda_max[tid] = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        friction_static[tid] = wp.float32(0.8)
        friction_kinetic[tid] = wp.float32(0.6)
    else:
        projection_mode[tid] = VELOCITY_ROWS3_PROJECT_BOUNDS
        lambda_min[tid] = wp.vec3f(-wp.float32(0.7), wp.float32(0.0), -wp.float32(0.2))
        lambda_max[tid] = wp.vec3f(wp.float32(0.7), BLOCK_LAMBDA_INF, wp.float32(0.2))
        friction_static[tid] = wp.float32(0.0)
        friction_kinetic[tid] = wp.float32(0.0)


@wp.kernel
def _solve_branchy_kernel(
    k_inv: wp.array[wp.vec3f],
    residual: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_delta: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    sor_boost: wp.float32,
):
    tid = wp.tid()
    op = VelocityRows3Op()
    op.k_inv = k_inv[tid]
    op.residual = residual[tid]
    op.lambda_old = lambda_old[tid]
    op.mass_coeff = mass_coeff[tid]
    op.impulse_coeff = impulse_coeff[tid]
    op.lambda_min = lambda_min[tid]
    op.lambda_max = lambda_max[tid]
    op.projection_mode = projection_mode[tid]
    op.friction_static = friction_static[tid]
    op.friction_kinetic = friction_kinetic[tid]

    update = block_solve_velocity_rows3_op(op, sor_boost)
    out_delta[tid] = update.delta
    out_lambda[tid] = update.lambda_new


@wp.kernel
def _solve_uniform_kernel(
    k_inv: wp.array[wp.vec3f],
    residual: wp.array[wp.vec3f],
    lambda_old: wp.array[wp.vec3f],
    mass_coeff: wp.array[wp.vec3f],
    impulse_coeff: wp.array[wp.vec3f],
    lambda_min: wp.array[wp.vec3f],
    lambda_max: wp.array[wp.vec3f],
    projection_mode: wp.array[wp.int32],
    friction_static: wp.array[wp.float32],
    friction_kinetic: wp.array[wp.float32],
    out_delta: wp.array[wp.vec3f],
    out_lambda: wp.array[wp.vec3f],
    sor_boost: wp.float32,
):
    tid = wp.tid()
    op = VelocityRows3Op()
    op.k_inv = k_inv[tid]
    op.residual = residual[tid]
    op.lambda_old = lambda_old[tid]
    op.mass_coeff = mass_coeff[tid]
    op.impulse_coeff = impulse_coeff[tid]
    op.lambda_min = lambda_min[tid]
    op.lambda_max = lambda_max[tid]
    op.projection_mode = projection_mode[tid]
    op.friction_static = friction_static[tid]
    op.friction_kinetic = friction_kinetic[tid]

    update = block_solve_velocity_rows3_op_uniform_projection(op, sor_boost)
    out_delta[tid] = update.delta
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


def _launch_init(arrays: tuple[object, ...], *, rows: int, ratio: float, period: int, device: wp.context.Devicelike):
    contacts_per_period = int(round(max(0.0, min(1.0, ratio)) * float(period)))
    wp.launch(
        _init_ops_kernel,
        dim=rows,
        inputs=[*arrays[:10], period, contacts_per_period],
        device=device,
    )


def _make_solve(kernel, arrays: tuple[object, ...], out_delta, out_lambda, *, rows: int, block_dim: int, device):
    def run():
        wp.launch(
            kernel,
            dim=rows,
            inputs=[*arrays[:10], out_delta, out_lambda, wp.float32(1.0)],
            device=device,
            block_dim=block_dim,
        )

    return run


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

    arrays = (
        wp.empty(rows, dtype=wp.vec3f, device=device),
        wp.empty(rows, dtype=wp.vec3f, device=device),
        wp.empty(rows, dtype=wp.vec3f, device=device),
        wp.empty(rows, dtype=wp.vec3f, device=device),
        wp.empty(rows, dtype=wp.vec3f, device=device),
        wp.empty(rows, dtype=wp.vec3f, device=device),
        wp.empty(rows, dtype=wp.vec3f, device=device),
        wp.empty(rows, dtype=wp.int32, device=device),
        wp.empty(rows, dtype=wp.float32, device=device),
        wp.empty(rows, dtype=wp.float32, device=device),
    )
    branch_delta = wp.empty(rows, dtype=wp.vec3f, device=device)
    branch_lambda = wp.empty(rows, dtype=wp.vec3f, device=device)
    uniform_delta = wp.empty(rows, dtype=wp.vec3f, device=device)
    uniform_lambda = wp.empty(rows, dtype=wp.vec3f, device=device)

    print("ratio,branchy_ms,uniform_ms,speedup,max_abs_lambda_err")
    for ratio in _parse_csv_floats(args.ratios):
        _launch_init(arrays, rows=rows, ratio=ratio, period=int(args.period), device=device)

        branch_run = _make_solve(
            _solve_branchy_kernel,
            arrays,
            branch_delta,
            branch_lambda,
            rows=rows,
            block_dim=int(args.block_dim),
            device=device,
        )
        uniform_run = _make_solve(
            _solve_uniform_kernel,
            arrays,
            uniform_delta,
            uniform_lambda,
            rows=rows,
            block_dim=int(args.block_dim),
            device=device,
        )

        branch_run()
        uniform_run()
        max_err = float(np.max(np.abs(branch_lambda.numpy() - uniform_lambda.numpy())))

        branch_ms, _ = _bench(branch_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
        uniform_ms, _ = _bench(uniform_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=device)
        speedup = branch_ms / uniform_ms if uniform_ms > 0.0 else float("nan")
        print(f"{ratio:.2f},{branch_ms:.6f},{uniform_ms:.6f},{speedup:.3f},{max_err:.6g}")


if __name__ == "__main__":
    main()
