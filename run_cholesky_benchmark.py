# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone microbenchmark for Kamino's direct LLT (Cholesky) linear solver.

Benchmarks ``LLTBlockedSolver`` (Warp tile-based LAPACK-style blocked Cholesky)
on a single dense SPD system of size 100x100 and 1000x1000.

Each case times (separately) on a CUDA device, using CUDA graph capture:
    * factorize:      ``solver.compute(A)``  -> L
    * solve (b, x):   ``solver.solve(b, x)`` -> forward + backward substitution
    * solve_inplace:  ``solver.solve_inplace(x)`` (x starts as rhs)

The solve timings are the primary metric -- that's the back-substitution path
that dominates per-step cost when the factor is reused across many solves.

Run:
    python run_cholesky_benchmark.py
    python run_cholesky_benchmark.py --device cpu  # no graph capture on cpu
    python run_cholesky_benchmark.py --iters 500 --warmup 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import float32
from newton._src.solvers.kamino._src.linalg.core import (
    DenseLinearOperatorData,
    DenseSquareMultiLinearInfo,
)
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver
from newton._src.solvers.kamino.tests.utils.rand import RandomProblemLLT


def build_problem(n: int, device: wp.DeviceLike, seed: int = 42):
    """Build a single SPD system of size `n` and wrap it in a DenseLinearOperator."""
    problem = RandomProblemLLT(
        dims=n,
        seed=seed,
        np_dtype=np.float32,
        wp_dtype=float32,
        device=device,
    )
    opinfo = DenseSquareMultiLinearInfo()
    opinfo.finalize(dimensions=problem.dims, dtype=problem.wp_dtype, device=device)
    operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
    return problem, operator


def time_replay(
    launch_fn,
    iters: int,
    warmup: int,
    device: wp.Device,
    use_graph: bool,
) -> tuple[float, float]:
    """Time ``launch_fn`` (which performs one or more Warp launches).

    With ``use_graph=True`` the callable is captured once into a CUDA graph and
    replayed with ``wp.capture_launch``. One sync per iteration is used so the
    per-iteration number includes the full kernel runtime.

    Returns (mean_ms, stdev_ms) per replay.
    """
    if use_graph:
        # Warm up before capture so any JIT/loads are out of the way.
        for _ in range(warmup):
            launch_fn()
        wp.synchronize_device(device)

        with wp.ScopedDevice(device):
            wp.capture_begin()
            try:
                launch_fn()
            finally:
                graph = wp.capture_end()

        def run_once():
            wp.capture_launch(graph)
    else:
        def run_once():
            launch_fn()

    # Warm the replay path itself.
    for _ in range(max(1, warmup // 2)):
        run_once()
    wp.synchronize_device(device)

    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        run_once()
        wp.synchronize_device(device)
        samples[i] = (time.perf_counter() - t0) * 1e3  # ms
    return float(samples.mean()), float(samples.std())


def max_abs_err(x_wp: wp.array, x_ref: np.ndarray, n: int) -> float:
    return float(np.max(np.abs(x_wp.numpy()[:n] - x_ref)))


def residual_metrics(
    x_wp: wp.array,
    A_np: np.ndarray,
    b_np: np.ndarray,
    x_ref_np: np.ndarray,
    n: int,
) -> tuple[float, float, float]:
    """Compute (relative residual, relative solution error, max |delta|).

    * relative residual = ||A x - b|| / max(||b||, eps)
    * relative solution error = ||x - x_ref|| / max(||x_ref||, eps)
    * max |delta| = max |x - x_ref|
    """
    x_np = x_wp.numpy()[:n].astype(np.float64)
    A = A_np.astype(np.float64)
    b = b_np.astype(np.float64)
    xr = x_ref_np.astype(np.float64)

    r = A @ x_np - b
    eps = np.finfo(np.float64).eps
    rel_res = float(np.linalg.norm(r) / max(np.linalg.norm(b), eps))
    rel_err = float(np.linalg.norm(x_np - xr) / max(np.linalg.norm(xr), eps))
    max_abs = float(np.max(np.abs(x_np - xr)))
    return rel_res, rel_err, max_abs


def run_case(
    solver_cls,
    solver_label: str,
    n: int,
    device: wp.Device,
    iters: int,
    warmup: int,
    seed: int,
    use_graph: bool,
    block_size: int | None = None,
) -> dict:
    problem, operator = build_problem(n, device=device, seed=seed)

    solver_kwargs: dict = {"operator": operator, "device": device}
    if block_size is not None:
        solver_kwargs["block_size"] = block_size
    solver = solver_cls(**solver_kwargs)
    x_wp = wp.zeros_like(problem.b_wp, device=device)
    x_inplace_wp = wp.clone(problem.b_wp, device=device)

    # Pre-timing correctness check (eager, no graph).
    solver.compute(A=problem.A_wp)
    solver.solve(b=problem.b_wp, x=x_wp)
    wp.synchronize_device(device)
    rel_res_pre, rel_err_pre, max_abs_pre = residual_metrics(
        x_wp, problem.A_np[0], problem.b_np[0], problem.x_np[0], n
    )

    wp.copy(x_inplace_wp, problem.b_wp)
    solver.solve_inplace(x=x_inplace_wp)
    wp.synchronize_device(device)
    rel_res_ip_pre, rel_err_ip_pre, max_abs_ip_pre = residual_metrics(
        x_inplace_wp, problem.A_np[0], problem.b_np[0], problem.x_np[0], n
    )

    # --- Factorize timing ---------------------------------------------------
    factor_mean, factor_std = time_replay(
        launch_fn=lambda: solver.compute(A=problem.A_wp),
        iters=iters,
        warmup=warmup,
        device=device,
        use_graph=use_graph,
    )

    # --- Solve timing (primary metric) --------------------------------------
    # Factor must already exist; compute() above already set it. A graph replay
    # reuses the cached L produced before capture.
    solve_mean, solve_std = time_replay(
        launch_fn=lambda: solver.solve(b=problem.b_wp, x=x_wp),
        iters=iters,
        warmup=warmup,
        device=device,
        use_graph=use_graph,
    )

    # --- Solve-inplace timing ----------------------------------------------
    # solve_inplace overwrites x; to make successive replays equivalent we copy
    # b -> x_inplace_wp before each timed call. We include that copy in the
    # measurement so that graph capture actually captures a useful unit of work.
    def solve_inplace_launch():
        wp.copy(x_inplace_wp, problem.b_wp)
        solver.solve_inplace(x=x_inplace_wp)

    solve_ip_mean, solve_ip_std = time_replay(
        launch_fn=solve_inplace_launch,
        iters=iters,
        warmup=warmup,
        device=device,
        use_graph=use_graph,
    )

    # Post-timing correctness re-check: the factor may have been re-computed
    # many times via graph replay. Run one fresh compute+solve (no graph) and
    # verify the result still matches the reference.
    solver.compute(A=problem.A_wp)
    solver.solve(b=problem.b_wp, x=x_wp)
    wp.synchronize_device(device)
    rel_res_post, rel_err_post, max_abs_post = residual_metrics(
        x_wp, problem.A_np[0], problem.b_np[0], problem.x_np[0], n
    )

    return {
        "solver": solver_label,
        "block_size": block_size,
        "n": n,
        "factor_ms_mean": factor_mean,
        "factor_ms_std": factor_std,
        "solve_ms_mean": solve_mean,
        "solve_ms_std": solve_std,
        "solve_ip_ms_mean": solve_ip_mean,
        "solve_ip_ms_std": solve_ip_std,
        "rel_res_pre": rel_res_pre,
        "rel_err_pre": rel_err_pre,
        "max_abs_pre": max_abs_pre,
        "rel_res_ip_pre": rel_res_ip_pre,
        "rel_err_ip_pre": rel_err_ip_pre,
        "max_abs_ip_pre": max_abs_ip_pre,
        "rel_res_post": rel_res_post,
        "rel_err_post": rel_err_post,
        "max_abs_post": max_abs_post,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Warp device, e.g. 'cuda:0' or 'cpu' (default: Warp default).")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations per case (default: 200).")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations per case (default: 10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SPD matrix generation (default: 42).")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="Matrix sizes to benchmark (default: 100 1000).",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable CUDA graph capture (useful for CPU or when profiling kernel code paths).",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[16],
        help=(
            "Tile/block sizes to benchmark for LLTBlockedSolver (default: 16, matching "
            "socu's float32 choice for sizes that aren't a multiple of 32). Each listed "
            "size compiles its own set of kernels -- compile time grows roughly "
            "quadratically in block_size."
        ),
    )
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    use_graph = (not args.no_graph) and device.is_cuda
    print(f"[info] Warp device: {device}")
    print(f"[info] iters={args.iters}, warmup={args.warmup}, seed={args.seed}, sizes={args.sizes}, graph={use_graph}")
    print()

    cases: list[tuple] = [
        (LLTBlockedSolver, f"LLTBlocked(bs={bs})", bs) for bs in args.block_sizes
    ]

    results: list[dict] = []
    for n in args.sizes:
        for solver_cls, label, bs in cases:
            print(f"[run]  {label:<18s}  n={n}  ...", flush=True)
            results.append(
                run_case(
                    solver_cls=solver_cls,
                    solver_label=label,
                    n=n,
                    device=device,
                    iters=args.iters,
                    warmup=args.warmup,
                    seed=args.seed,
                    use_graph=use_graph,
                    block_size=bs,
                )
            )

    # Timing summary
    header = (
        f"{'solver':<18s}  {'n':>6s}  "
        f"{'factor (ms)':>20s}  {'solve (ms)':>20s}  {'solve_ip (ms)':>22s}"
    )
    print()
    print("=== Timings ===")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['solver']:<18s}  {r['n']:>6d}  "
            f"{r['factor_ms_mean']:>10.4f} \u00b1 {r['factor_ms_std']:>6.4f}  "
            f"{r['solve_ms_mean']:>10.4f} \u00b1 {r['solve_ms_std']:>6.4f}  "
            f"{r['solve_ip_ms_mean']:>12.4f} \u00b1 {r['solve_ip_ms_std']:>6.4f}  "
        )

    # Correctness summary
    print()
    print("=== Correctness (|| A x - b || / || b ||   |   || x - x_ref || / || x_ref ||   |   max |x - x_ref|) ===")
    corr_header = (
        f"{'solver':<18s}  {'n':>6s}  "
        f"{'solve: rel_res':>16s}  {'rel_err':>12s}  {'max_abs':>12s}  "
        f"{'ip: rel_res':>14s}  {'rel_err':>12s}  {'max_abs':>12s}  "
        f"{'post: rel_res':>16s}  {'rel_err':>12s}  {'max_abs':>12s}"
    )
    print(corr_header)
    print("-" * len(corr_header))
    for r in results:
        print(
            f"{r['solver']:<18s}  {r['n']:>6d}  "
            f"{r['rel_res_pre']:>16.2e}  {r['rel_err_pre']:>12.2e}  {r['max_abs_pre']:>12.2e}  "
            f"{r['rel_res_ip_pre']:>14.2e}  {r['rel_err_ip_pre']:>12.2e}  {r['max_abs_ip_pre']:>12.2e}  "
            f"{r['rel_res_post']:>16.2e}  {r['rel_err_post']:>12.2e}  {r['max_abs_post']:>12.2e}"
        )

    # Hard-fail gate on clearly wrong results.
    rel_res_budget = 1e-3  # float32 + n up to 1000 => a few x 1e-5 typical
    rel_err_budget = 1e-2
    failures = []
    for r in results:
        for label, rel_res, rel_err in [
            ("solve",        r["rel_res_pre"],    r["rel_err_pre"]),
            ("solve_inplace", r["rel_res_ip_pre"], r["rel_err_ip_pre"]),
            ("solve_post",   r["rel_res_post"],   r["rel_err_post"]),
        ]:
            if not (rel_res < rel_res_budget and rel_err < rel_err_budget):
                failures.append(
                    f"{r['solver']}(n={r['n']}) {label}: rel_res={rel_res:.2e} (budget {rel_res_budget:.0e}), "
                    f"rel_err={rel_err:.2e} (budget {rel_err_budget:.0e})"
                )

    print()
    if failures:
        print("!!! CORRECTNESS FAILURES !!!")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[ok] All solves passed correctness check "
          f"(rel_res < {rel_res_budget:.0e}, rel_err < {rel_err_budget:.0e}).")


if __name__ == "__main__":
    main()
