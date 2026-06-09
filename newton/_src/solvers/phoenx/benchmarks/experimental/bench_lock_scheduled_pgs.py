# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Synthetic lock-scheduled PGS benchmark.

This is a local-solver experiment, not a production PhoenX dispatch path. It
compares two ways to run the same scalar projected Gauss-Seidel update:

* ``colored``: precompute conflict-free colors, launch one kernel per color;
* ``locked-pass``: launch a fixed number of one-try lock passes per solver
  iteration. Successful rows solve immediately; unsuccessful rows retry next
  pass;
* ``locked-spin``: launch one persistent-ish kernel per solver iteration, let
  workers scan rows until every row completes.

The locked path is the closest candidate for a local, GS-like solver that does
not require graph coloring. It keeps immediate body-state updates and local
projection; the cost is runtime atomics, possible spin, and nondeterministic
row order.

Current takeaway on RTX PRO 6000 Blackwell: ordinary multi-kernel lock
schedulers are not competitive with colored PGS. On a 2048-world mixed graph
with 131k rows and four iterations, colored PGS measured about 0.32 ms,
fixed-pass locking about 0.71 ms when complete, queued retries about 7.7 ms,
and spin locking about 16 ms. This keeps the evidence local and reproducible
while pointing future work toward cooperative megakernel/shared-memory queues
rather than global-memory lock waves.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.experimental.bench_lock_scheduled_pgs
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time

import numpy as np
import warp as wp

LAMBDA_INF = 1.0e20


@dataclasses.dataclass(frozen=True)
class SyntheticProblem:
    """Device buffers for a scalar local PGS problem."""

    body0: wp.array[wp.int32]
    body1: wp.array[wp.int32]
    w0: wp.array[wp.float32]
    w1: wp.array[wp.float32]
    inv_mass: wp.array[wp.float32]
    rhs: wp.array[wp.float32]
    lambda_min: wp.array[wp.float32]
    lambda_max: wp.array[wp.float32]
    initial_v: wp.array[wp.float32]
    body_v: wp.array[wp.float32]
    lambdas: wp.array[wp.float32]
    row_done: wp.array[wp.int32]
    body_locks: wp.array[wp.int32]
    queue_a: wp.array[wp.int32]
    queue_b: wp.array[wp.int32]
    queue_count_a: wp.array[wp.int32]
    queue_count_b: wp.array[wp.int32]
    total_done: wp.array[wp.int32]
    failed: wp.array[wp.int32]
    total_rows: int
    total_bodies: int


@dataclasses.dataclass(frozen=True)
class HostGraph:
    """Host-side graph metadata used to build device buffers."""

    body0: np.ndarray
    body1: np.ndarray
    color_rows: np.ndarray
    color_starts: np.ndarray
    num_colors: int
    total_bodies: int


@wp.func
def _solve_row(
    row: wp.int32,
    body0: wp.array[wp.int32],
    body1: wp.array[wp.int32],
    w0: wp.array[wp.float32],
    w1: wp.array[wp.float32],
    inv_mass: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    lambda_min: wp.array[wp.float32],
    lambda_max: wp.array[wp.float32],
    body_v: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    sor: wp.float32,
):
    b0 = body0[row]
    b1 = body1[row]
    jw0 = w0[row]
    jw1 = w1[row]
    residual = jw0 * body_v[b0] + jw1 * body_v[b1] + rhs[row]
    diag = inv_mass[b0] * jw0 * jw0 + inv_mass[b1] * jw1 * jw1 + wp.float32(1.0e-6)
    old_lambda = lambdas[row]
    candidate = old_lambda - sor * residual / diag
    new_lambda = wp.clamp(candidate, lambda_min[row], lambda_max[row])
    delta = new_lambda - old_lambda
    lambdas[row] = new_lambda
    body_v[b0] = body_v[b0] + inv_mass[b0] * jw0 * delta
    body_v[b1] = body_v[b1] + inv_mass[b1] * jw1 * delta


@wp.kernel(enable_backward=False)
def _reset_state_kernel(
    initial_v: wp.array[wp.float32],
    body_v: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    row_done: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    total_rows: wp.int32,
    total_bodies: wp.int32,
):
    tid = wp.tid()
    if tid < total_bodies:
        body_v[tid] = initial_v[tid]
        body_locks[tid] = wp.int32(0)
    if tid < total_rows:
        lambdas[tid] = wp.float32(0.0)
        row_done[tid] = wp.int32(0)
    if tid == wp.int32(0):
        total_done[0] = wp.int32(0)
        failed[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _reset_iteration_kernel(
    row_done: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    total_rows: wp.int32,
    total_bodies: wp.int32,
):
    tid = wp.tid()
    if tid < total_bodies:
        body_locks[tid] = wp.int32(0)
    if tid < total_rows:
        row_done[tid] = wp.int32(0)
    if tid == wp.int32(0):
        total_done[0] = wp.int32(0)
        failed[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _clear_locks_kernel(body_locks: wp.array[wp.int32], total_bodies: wp.int32):
    tid = wp.tid()
    if tid < total_bodies:
        body_locks[tid] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _colored_pgs_kernel(
    color_rows: wp.array[wp.int32],
    body0: wp.array[wp.int32],
    body1: wp.array[wp.int32],
    w0: wp.array[wp.float32],
    w1: wp.array[wp.float32],
    inv_mass: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    lambda_min: wp.array[wp.float32],
    lambda_max: wp.array[wp.float32],
    body_v: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    color_start: wp.int32,
    sor: wp.float32,
):
    lane = wp.tid()
    row = color_rows[color_start + lane]
    _solve_row(row, body0, body1, w0, w1, inv_mass, rhs, lambda_min, lambda_max, body_v, lambdas, sor)


@wp.kernel(enable_backward=False)
def _locked_pgs_kernel(
    body0: wp.array[wp.int32],
    body1: wp.array[wp.int32],
    w0: wp.array[wp.float32],
    w1: wp.array[wp.float32],
    inv_mass: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    lambda_min: wp.array[wp.float32],
    lambda_max: wp.array[wp.float32],
    body_v: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    row_done: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    total_rows: wp.int32,
    scheduler_threads: wp.int32,
    max_attempts: wp.int32,
    sor: wp.float32,
):
    tid = wp.tid()
    cursor = tid
    attempts = wp.int32(0)
    done = wp.atomic_add(total_done, 0, wp.int32(0))
    while done < total_rows and attempts < max_attempts:
        row = cursor % total_rows
        if wp.atomic_add(row_done, row, wp.int32(0)) == wp.int32(0):
            b0 = body0[row]
            b1 = body1[row]
            first = wp.min(b0, b1)
            second = wp.max(b0, b1)
            old_first = wp.atomic_cas(body_locks, first, wp.int32(0), wp.int32(1))
            if old_first == wp.int32(0):
                old_second = wp.atomic_cas(body_locks, second, wp.int32(0), wp.int32(1))
                if old_second == wp.int32(0):
                    if wp.atomic_add(row_done, row, wp.int32(0)) == wp.int32(0):
                        _solve_row(
                            row,
                            body0,
                            body1,
                            w0,
                            w1,
                            inv_mass,
                            rhs,
                            lambda_min,
                            lambda_max,
                            body_v,
                            lambdas,
                            sor,
                        )
                        row_done[row] = wp.int32(1)
                        wp.atomic_add(total_done, 0, wp.int32(1))
                    wp.atomic_exch(body_locks, second, wp.int32(0))
                wp.atomic_exch(body_locks, first, wp.int32(0))
        cursor = cursor + scheduler_threads
        attempts = attempts + wp.int32(1)
        done = wp.atomic_add(total_done, 0, wp.int32(0))

    if done < total_rows:
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _locked_pass_pgs_kernel(
    body0: wp.array[wp.int32],
    body1: wp.array[wp.int32],
    w0: wp.array[wp.float32],
    w1: wp.array[wp.float32],
    inv_mass: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    lambda_min: wp.array[wp.float32],
    lambda_max: wp.array[wp.float32],
    body_v: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    row_done: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    pass_index: wp.int32,
    total_rows: wp.int32,
    sor: wp.float32,
):
    slot = wp.tid()
    if slot >= total_rows:
        return
    multiplier = wp.int32(1103515245) + pass_index * wp.int32(24690)
    row = (slot * multiplier + pass_index * wp.int32(7919)) % total_rows
    if wp.atomic_add(row_done, row, wp.int32(0)) != wp.int32(0):
        return

    b0 = body0[row]
    b1 = body1[row]
    first = wp.min(b0, b1)
    second = wp.max(b0, b1)
    old_first = wp.atomic_cas(body_locks, first, wp.int32(0), wp.int32(1))
    if old_first == wp.int32(0):
        old_second = wp.atomic_cas(body_locks, second, wp.int32(0), wp.int32(1))
        if old_second == wp.int32(0):
            if wp.atomic_add(row_done, row, wp.int32(0)) == wp.int32(0):
                _solve_row(row, body0, body1, w0, w1, inv_mass, rhs, lambda_min, lambda_max, body_v, lambdas, sor)
                row_done[row] = wp.int32(1)
                wp.atomic_add(total_done, 0, wp.int32(1))
            wp.atomic_exch(body_locks, second, wp.int32(0))
        wp.atomic_exch(body_locks, first, wp.int32(0))


@wp.kernel(enable_backward=False)
def _mark_incomplete_kernel(total_done: wp.array[wp.int32], failed: wp.array[wp.int32], total_rows: wp.int32):
    if wp.tid() == wp.int32(0):
        if total_done[0] < total_rows:
            failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _init_queue_kernel(queue: wp.array[wp.int32], queue_count: wp.array[wp.int32], total_rows: wp.int32):
    tid = wp.tid()
    if tid < total_rows:
        queue[tid] = tid
    if tid == wp.int32(0):
        queue_count[0] = total_rows


@wp.kernel(enable_backward=False)
def _reset_queue_wave_kernel(
    body_locks: wp.array[wp.int32],
    queue_count_out: wp.array[wp.int32],
    total_bodies: wp.int32,
):
    tid = wp.tid()
    if tid < total_bodies:
        body_locks[tid] = wp.int32(0)
    if tid == wp.int32(0):
        queue_count_out[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _queued_lock_pgs_kernel(
    queue_in: wp.array[wp.int32],
    queue_count_in: wp.array[wp.int32],
    queue_out: wp.array[wp.int32],
    queue_count_out: wp.array[wp.int32],
    body0: wp.array[wp.int32],
    body1: wp.array[wp.int32],
    w0: wp.array[wp.float32],
    w1: wp.array[wp.float32],
    inv_mass: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    lambda_min: wp.array[wp.float32],
    lambda_max: wp.array[wp.float32],
    body_v: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    row_done: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    sor: wp.float32,
):
    slot = wp.tid()
    active = wp.atomic_add(queue_count_in, 0, wp.int32(0))
    if slot >= active:
        return

    row = queue_in[slot]
    if wp.atomic_add(row_done, row, wp.int32(0)) != wp.int32(0):
        return

    b0 = body0[row]
    b1 = body1[row]
    first = wp.min(b0, b1)
    second = wp.max(b0, b1)
    old_first = wp.atomic_cas(body_locks, first, wp.int32(0), wp.int32(1))
    if old_first != wp.int32(0):
        pos = wp.atomic_add(queue_count_out, 0, wp.int32(1))
        queue_out[pos] = row
        return

    old_second = wp.atomic_cas(body_locks, second, wp.int32(0), wp.int32(1))
    if old_second != wp.int32(0):
        wp.atomic_exch(body_locks, first, wp.int32(0))
        pos = wp.atomic_add(queue_count_out, 0, wp.int32(1))
        queue_out[pos] = row
        return

    claimed = wp.atomic_cas(row_done, row, wp.int32(0), wp.int32(1))
    if claimed == wp.int32(0):
        _solve_row(row, body0, body1, w0, w1, inv_mass, rhs, lambda_min, lambda_max, body_v, lambdas, sor)
        wp.atomic_add(total_done, 0, wp.int32(1))
    wp.atomic_exch(body_locks, second, wp.int32(0))
    wp.atomic_exch(body_locks, first, wp.int32(0))


@wp.kernel(enable_backward=False)
def _checksum_kernel(
    body_v: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    checksum: wp.array[wp.float32],
    total_rows: wp.int32,
    total_bodies: wp.int32,
):
    tid = wp.tid()
    if tid < total_bodies:
        wp.atomic_add(checksum, 0, wp.abs(body_v[tid]))
    if tid < total_rows:
        wp.atomic_add(checksum, 0, wp.float32(0.01) * wp.abs(lambdas[tid]))


def _build_host_graph(
    *,
    worlds: int,
    bodies_per_world: int,
    rows_per_world: int,
    pattern: str,
    seed: int,
) -> HostGraph:
    rng = np.random.default_rng(seed)
    total_rows = worlds * rows_per_world
    total_bodies = worlds * bodies_per_world
    body0 = np.empty(total_rows, dtype=np.int32)
    body1 = np.empty(total_rows, dtype=np.int32)

    for world in range(worlds):
        row_start = world * rows_per_world
        body_start = world * bodies_per_world
        for r in range(rows_per_world):
            row = row_start + r
            if pattern == "chain":
                local0 = r % bodies_per_world
                local1 = (local0 + 1) % bodies_per_world
            elif pattern == "hub":
                local0 = 0
                local1 = 1 + (r % (bodies_per_world - 1))
            elif pattern == "mixed":
                if r % 4 == 0:
                    local0 = 0
                    local1 = 1 + int(rng.integers(0, bodies_per_world - 1))
                else:
                    local0 = int(rng.integers(0, bodies_per_world))
                    local1 = int(rng.integers(0, bodies_per_world - 1))
                    if local1 >= local0:
                        local1 += 1
            else:
                local0 = int(rng.integers(0, bodies_per_world))
                local1 = int(rng.integers(0, bodies_per_world - 1))
                if local1 >= local0:
                    local1 += 1
            body0[row] = body_start + local0
            body1[row] = body_start + local1

    body_colors: list[set[int]] = [set() for _ in range(total_bodies)]
    color_lists: list[list[int]] = []
    for row in range(total_rows):
        c = 0
        b0 = int(body0[row])
        b1 = int(body1[row])
        while c in body_colors[b0] or c in body_colors[b1]:
            c += 1
        while len(color_lists) <= c:
            color_lists.append([])
        color_lists[c].append(row)
        body_colors[b0].add(c)
        body_colors[b1].add(c)

    color_starts = np.zeros(len(color_lists) + 1, dtype=np.int32)
    for c, rows in enumerate(color_lists):
        color_starts[c + 1] = color_starts[c] + len(rows)
    color_rows = np.empty(total_rows, dtype=np.int32)
    for c, rows in enumerate(color_lists):
        s = int(color_starts[c])
        color_rows[s : s + len(rows)] = np.asarray(rows, dtype=np.int32)

    return HostGraph(
        body0=body0,
        body1=body1,
        color_rows=color_rows,
        color_starts=color_starts,
        num_colors=len(color_lists),
        total_bodies=total_bodies,
    )


def _build_problem(host: HostGraph, *, seed: int, device: wp.context.Devicelike) -> SyntheticProblem:
    rng = np.random.default_rng(seed)
    total_rows = int(host.body0.shape[0])
    total_bodies = host.total_bodies
    w0 = rng.uniform(0.5, 1.5, size=total_rows).astype(np.float32)
    w1 = -rng.uniform(0.5, 1.5, size=total_rows).astype(np.float32)
    inv_mass = rng.uniform(0.5, 2.0, size=total_bodies).astype(np.float32)
    rhs = rng.normal(0.0, 0.05, size=total_rows).astype(np.float32)
    initial_v = rng.normal(0.0, 0.2, size=total_bodies).astype(np.float32)

    kind = rng.integers(0, 3, size=total_rows, dtype=np.int32)
    lambda_min = np.full(total_rows, -LAMBDA_INF, dtype=np.float32)
    lambda_max = np.full(total_rows, LAMBDA_INF, dtype=np.float32)
    unilateral = kind == 1
    bounded = kind == 2
    lambda_min[unilateral] = 0.0
    lambda_max[bounded] = rng.uniform(0.2, 2.0, size=int(np.count_nonzero(bounded))).astype(np.float32)
    lambda_min[bounded] = -lambda_max[bounded]

    return SyntheticProblem(
        body0=wp.array(host.body0, dtype=wp.int32, device=device),
        body1=wp.array(host.body1, dtype=wp.int32, device=device),
        w0=wp.array(w0, dtype=wp.float32, device=device),
        w1=wp.array(w1, dtype=wp.float32, device=device),
        inv_mass=wp.array(inv_mass, dtype=wp.float32, device=device),
        rhs=wp.array(rhs, dtype=wp.float32, device=device),
        lambda_min=wp.array(lambda_min, dtype=wp.float32, device=device),
        lambda_max=wp.array(lambda_max, dtype=wp.float32, device=device),
        initial_v=wp.array(initial_v, dtype=wp.float32, device=device),
        body_v=wp.zeros(total_bodies, dtype=wp.float32, device=device),
        lambdas=wp.zeros(total_rows, dtype=wp.float32, device=device),
        row_done=wp.zeros(total_rows, dtype=wp.int32, device=device),
        body_locks=wp.zeros(total_bodies, dtype=wp.int32, device=device),
        queue_a=wp.zeros(total_rows, dtype=wp.int32, device=device),
        queue_b=wp.zeros(total_rows, dtype=wp.int32, device=device),
        queue_count_a=wp.zeros(1, dtype=wp.int32, device=device),
        queue_count_b=wp.zeros(1, dtype=wp.int32, device=device),
        total_done=wp.zeros(1, dtype=wp.int32, device=device),
        failed=wp.zeros(1, dtype=wp.int32, device=device),
        total_rows=total_rows,
        total_bodies=total_bodies,
    )


def _launch_reset(problem: SyntheticProblem) -> None:
    wp.launch(
        _reset_state_kernel,
        dim=max(problem.total_rows, problem.total_bodies),
        inputs=[
            problem.initial_v,
            problem.body_v,
            problem.lambdas,
            problem.row_done,
            problem.body_locks,
            problem.total_done,
            problem.failed,
            wp.int32(problem.total_rows),
            wp.int32(problem.total_bodies),
        ],
    )


def _launch_colored(
    problem: SyntheticProblem,
    color_rows: wp.array[wp.int32],
    color_starts: np.ndarray,
    *,
    iterations: int,
    sor: float,
) -> None:
    _launch_reset(problem)
    for _ in range(iterations):
        for color in range(len(color_starts) - 1):
            start = int(color_starts[color])
            end = int(color_starts[color + 1])
            count = end - start
            if count == 0:
                continue
            wp.launch(
                _colored_pgs_kernel,
                dim=count,
                inputs=[
                    color_rows,
                    problem.body0,
                    problem.body1,
                    problem.w0,
                    problem.w1,
                    problem.inv_mass,
                    problem.rhs,
                    problem.lambda_min,
                    problem.lambda_max,
                    problem.body_v,
                    problem.lambdas,
                    wp.int32(start),
                    wp.float32(sor),
                ],
            )


def _launch_locked(
    problem: SyntheticProblem,
    *,
    iterations: int,
    scheduler_threads: int,
    max_attempts: int,
    sor: float,
) -> None:
    _launch_reset(problem)
    for _ in range(iterations):
        wp.launch(
            _reset_iteration_kernel,
            dim=max(problem.total_rows, problem.total_bodies),
            inputs=[
                problem.row_done,
                problem.body_locks,
                problem.total_done,
                problem.failed,
                wp.int32(problem.total_rows),
                wp.int32(problem.total_bodies),
            ],
        )
        wp.launch(
            _locked_pgs_kernel,
            dim=scheduler_threads,
            inputs=[
                problem.body0,
                problem.body1,
                problem.w0,
                problem.w1,
                problem.inv_mass,
                problem.rhs,
                problem.lambda_min,
                problem.lambda_max,
                problem.body_v,
                problem.lambdas,
                problem.row_done,
                problem.body_locks,
                problem.total_done,
                problem.failed,
                wp.int32(problem.total_rows),
                wp.int32(scheduler_threads),
                wp.int32(max_attempts),
                wp.float32(sor),
            ],
        )


def _launch_locked_pass(
    problem: SyntheticProblem,
    *,
    iterations: int,
    passes: int,
    sor: float,
) -> None:
    _launch_reset(problem)
    for _ in range(iterations):
        wp.launch(
            _reset_iteration_kernel,
            dim=max(problem.total_rows, problem.total_bodies),
            inputs=[
                problem.row_done,
                problem.body_locks,
                problem.total_done,
                problem.failed,
                wp.int32(problem.total_rows),
                wp.int32(problem.total_bodies),
            ],
        )
        for pass_index in range(passes):
            wp.launch(
                _clear_locks_kernel,
                dim=problem.total_bodies,
                inputs=[problem.body_locks, wp.int32(problem.total_bodies)],
            )
            wp.launch(
                _locked_pass_pgs_kernel,
                dim=problem.total_rows,
                inputs=[
                    problem.body0,
                    problem.body1,
                    problem.w0,
                    problem.w1,
                    problem.inv_mass,
                    problem.rhs,
                    problem.lambda_min,
                    problem.lambda_max,
                    problem.body_v,
                    problem.lambdas,
                    problem.row_done,
                    problem.body_locks,
                    problem.total_done,
                    wp.int32(pass_index),
                    wp.int32(problem.total_rows),
                    wp.float32(sor),
                ],
            )
        wp.launch(
            _mark_incomplete_kernel,
            dim=1,
            inputs=[problem.total_done, problem.failed, wp.int32(problem.total_rows)],
        )


def _launch_locked_queue(
    problem: SyntheticProblem,
    *,
    iterations: int,
    passes: int,
    sor: float,
) -> None:
    _launch_reset(problem)
    for _ in range(iterations):
        wp.launch(
            _reset_iteration_kernel,
            dim=max(problem.total_rows, problem.total_bodies),
            inputs=[
                problem.row_done,
                problem.body_locks,
                problem.total_done,
                problem.failed,
                wp.int32(problem.total_rows),
                wp.int32(problem.total_bodies),
            ],
        )
        wp.launch(
            _init_queue_kernel,
            dim=problem.total_rows,
            inputs=[problem.queue_a, problem.queue_count_a, wp.int32(problem.total_rows)],
        )
        queue_in = problem.queue_a
        queue_out = problem.queue_b
        count_in = problem.queue_count_a
        count_out = problem.queue_count_b
        for _ in range(passes):
            wp.launch(
                _reset_queue_wave_kernel,
                dim=problem.total_bodies,
                inputs=[problem.body_locks, count_out, wp.int32(problem.total_bodies)],
            )
            wp.launch(
                _queued_lock_pgs_kernel,
                dim=problem.total_rows,
                inputs=[
                    queue_in,
                    count_in,
                    queue_out,
                    count_out,
                    problem.body0,
                    problem.body1,
                    problem.w0,
                    problem.w1,
                    problem.inv_mass,
                    problem.rhs,
                    problem.lambda_min,
                    problem.lambda_max,
                    problem.body_v,
                    problem.lambdas,
                    problem.row_done,
                    problem.body_locks,
                    problem.total_done,
                    wp.float32(sor),
                ],
            )
            queue_in, queue_out = queue_out, queue_in
            count_in, count_out = count_out, count_in
        wp.launch(
            _mark_incomplete_kernel,
            dim=1,
            inputs=[problem.total_done, problem.failed, wp.int32(problem.total_rows)],
        )


def _time_captured(
    launch_fn,
    *,
    n_runs: int,
    warmup: int,
    trials: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        launch_fn()
    wp.synchronize_device()

    graph = None
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            launch_fn()
        graph = capture.graph
        wp.synchronize_device()

    timings_ms: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            if graph is None:
                launch_fn()
            else:
                wp.capture_launch(graph)
        wp.synchronize_device()
        timings_ms.append((time.perf_counter() - t0) * 1000.0 / float(n_runs))
    return min(timings_ms), statistics.median(timings_ms)


def _checksum(problem: SyntheticProblem) -> float:
    checksum = wp.zeros(1, dtype=wp.float32, device=problem.body_v.device)
    wp.launch(
        _checksum_kernel,
        dim=max(problem.total_rows, problem.total_bodies),
        inputs=[
            problem.body_v,
            problem.lambdas,
            checksum,
            wp.int32(problem.total_rows),
            wp.int32(problem.total_bodies),
        ],
    )
    return float(checksum.numpy()[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--bodies-per-world", type=int, default=32)
    parser.add_argument("--rows-per-world", type=int, default=64)
    parser.add_argument("--pattern", choices=("random", "mixed", "chain", "hub"), default="mixed")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--lock-mode", choices=("queue", "pass", "spin"), default="queue")
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--scheduler-threads", type=int, default=65536)
    parser.add_argument("--max-attempts", type=int, default=2048)
    parser.add_argument("--sor", type=float, default=1.0)
    parser.add_argument("--n-runs", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    wp.init()
    device = wp.get_device(args.device) if args.device is not None else wp.get_device()
    host = _build_host_graph(
        worlds=args.worlds,
        bodies_per_world=args.bodies_per_world,
        rows_per_world=args.rows_per_world,
        pattern=args.pattern,
        seed=args.seed,
    )
    color_rows = wp.array(host.color_rows, dtype=wp.int32, device=device)

    colored = _build_problem(host, seed=args.seed + 1, device=device)
    locked = _build_problem(host, seed=args.seed + 1, device=device)

    def colored_launch() -> None:
        _launch_colored(
            colored,
            color_rows,
            host.color_starts,
            iterations=args.iterations,
            sor=args.sor,
        )

    if args.lock_mode == "queue":

        def locked_launch() -> None:
            _launch_locked_queue(
                locked,
                iterations=args.iterations,
                passes=args.passes,
                sor=args.sor,
            )

    elif args.lock_mode == "pass":

        def locked_launch() -> None:
            _launch_locked_pass(
                locked,
                iterations=args.iterations,
                passes=args.passes,
                sor=args.sor,
            )

    else:

        def locked_launch() -> None:
            _launch_locked(
                locked,
                iterations=args.iterations,
                scheduler_threads=args.scheduler_threads,
                max_attempts=args.max_attempts,
                sor=args.sor,
            )

    colored_min, colored_median = _time_captured(
        colored_launch,
        n_runs=args.n_runs,
        warmup=args.warmup,
        trials=args.trials,
    )
    locked_min, locked_median = _time_captured(
        locked_launch,
        n_runs=args.n_runs,
        warmup=args.warmup,
        trials=args.trials,
    )
    locked_failed = int(locked.failed.numpy()[0])
    locked_done = int(locked.total_done.numpy()[0])

    colored_checksum = _checksum(colored)
    locked_checksum = _checksum(locked)
    speedup = colored_min / locked_min if locked_min > 0.0 else 0.0
    row_steps = args.worlds * args.rows_per_world * args.iterations

    print(
        "lock_scheduled_pgs "
        f"device={device.alias} pattern={args.pattern} worlds={args.worlds} "
        f"bodies/world={args.bodies_per_world} rows/world={args.rows_per_world} "
        f"rows={args.worlds * args.rows_per_world} colors={host.num_colors} "
        f"iterations={args.iterations} lock_mode={args.lock_mode} passes={args.passes} "
        f"scheduler_threads={args.scheduler_threads}"
    )
    print(
        f"colored_min_ms={colored_min:.4f} colored_median_ms={colored_median:.4f} "
        f"colored_row_steps/s={row_steps / (colored_min * 1.0e-3):.3e}"
    )
    print(
        f"locked_min_ms={locked_min:.4f} locked_median_ms={locked_median:.4f} "
        f"locked_row_steps/s={row_steps / (locked_min * 1.0e-3):.3e} "
        f"speedup_vs_colored={speedup:.3f} failed={locked_failed} done={locked_done}/{locked.total_rows}"
    )
    print(f"checksum_colored={colored_checksum:.6e} checksum_locked={locked_checksum:.6e}")


if __name__ == "__main__":
    main()
