# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark for the blocked LLT (Cholesky) factorize / solve kernels.

Measures GPU-side wall-clock time per launch using ``wp.ScopedTimer(synchronize=True)``
over a range of matrix sizes. Each size is benchmarked independently (no batching) so the
numbers reflect the per-problem kernel runtime.

Run:

    uv run newton/_src/solvers/kamino/_src/linalg/factorize/benchmark_llt_blocked.py

Optional flags:

    --sizes 32 64 128 256 512 1024    # override size list
    --num-iters 50                    # iterations per timed loop
    --block-size 16                   # tile block size used by the kernels
    --block-dim-factor 64             # threads per tile-block for factorize
    --block-dim-solve 64              # threads per tile-block for solve
    --device cuda:0                   # warp device override
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked import (
    llt_blocked_factorize,
    llt_blocked_solve,
    llt_blocked_solve_inplace,
    make_llt_blocked_factorize_kernel,
    make_llt_blocked_solve_inplace_kernel,
    make_llt_blocked_solve_kernel,
)


###
# Problem construction
###


def _make_spd_problem(n: int, block_size: int, seed: int, device: wp.DeviceLike):
    """Create a single SPD problem on the device and return the warp arrays.

    Layout: the flat backing allocations are oversized to ``padded_n*padded_n`` floats
    (matrix) and ``padded_n`` floats (rhs), but only the first ``n*n`` / ``n`` elements
    hold meaningful data at row-major stride ``n``. The kernel creates a view of shape
    ``(n, n)`` at the base pointer (stride ``n``); its padding fill-in overwrites the
    out-of-bounds loads (which read from the zeroed tail of the allocation) with
    identity values. This matches how the real solver is exercised by
    ``DenseSquareMultiLinearInfo`` with ``maxdim == dim``.
    """
    padded_n = ((n + block_size - 1) // block_size) * block_size

    rng = np.random.default_rng(seed)

    # Active SPD block
    A_active = rng.standard_normal((n, n)).astype(np.float32)
    A_active = A_active @ A_active.T + np.float32(n) * np.eye(n, dtype=np.float32)
    b_active = rng.standard_normal((n,)).astype(np.float32)

    # Oversized flat backing arrays. The kernel's ``wp.array(ptr, shape=(n, n))`` view
    # sees only the first ``n*n`` floats at stride ``n``; the remaining zeros are read
    # by OOB tile loads when ``n`` is not a multiple of ``block_size``, and are then
    # overwritten with identity by the kernel's padding fill-in.
    mat_size = padded_n * padded_n
    vec_size = padded_n

    A_flat = np.zeros(mat_size, dtype=np.float32)
    A_flat[: n * n] = A_active.reshape(-1)

    b_flat = np.zeros(vec_size, dtype=np.float32)
    b_flat[:n] = b_active

    dim_wp = wp.array([n], dtype=wp.int32, device=device)
    mio_wp = wp.array([0], dtype=wp.int32, device=device)
    vio_wp = wp.array([0], dtype=wp.int32, device=device)
    A_wp = wp.array(A_flat, dtype=wp.float32, device=device)
    b_wp = wp.array(b_flat, dtype=wp.float32, device=device)
    L_wp = wp.zeros_like(A_wp)
    y_wp = wp.zeros_like(b_wp)
    x_wp = wp.zeros_like(b_wp)

    return {
        "n": n,
        "padded_n": padded_n,
        "A_np": A_active,
        "b_np": b_active,
        "dim": dim_wp,
        "mio": mio_wp,
        "vio": vio_wp,
        "A": A_wp,
        "b": b_wp,
        "L": L_wp,
        "y": y_wp,
        "x": x_wp,
    }


###
# Verification (runs once, pre-timing)
###


def _verify(prob, factor_kernel, solve_kernel, solve_inplace_kernel, block_dim_factor, block_dim_solve, device):
    """Run factorize + solve once and return (cholesky_err, residual_norm, solve_diff)."""
    n = prob["n"]
    padded_n = prob["padded_n"]

    llt_blocked_factorize(
        kernel=factor_kernel,
        num_blocks=1,
        dim=prob["dim"],
        mio=prob["mio"],
        A=prob["A"],
        L=prob["L"],
        block_dim=block_dim_factor,
        device=device,
    )

    llt_blocked_solve(
        kernel=solve_kernel,
        num_blocks=1,
        dim=prob["dim"],
        mio=prob["mio"],
        vio=prob["vio"],
        L=prob["L"],
        b=prob["b"],
        y=prob["y"],
        x=prob["x"],
        block_dim=block_dim_solve,
        device=device,
    )

    # In-place solve (writes result to x_inplace)
    x_inplace_wp = wp.zeros_like(prob["b"])
    wp.copy(x_inplace_wp, prob["b"])
    llt_blocked_solve_inplace(
        kernel=solve_inplace_kernel,
        num_blocks=1,
        dim=prob["dim"],
        mio=prob["mio"],
        vio=prob["vio"],
        L=prob["L"],
        y=prob["y"],
        x=x_inplace_wp,
        block_dim=block_dim_solve,
        device=device,
    )
    wp.synchronize_device()

    # Kernel sees L as an (n, n) view at the base pointer; read the first n*n floats.
    L_np = prob["L"].numpy()[: n * n].reshape(n, n)
    x_np = prob["x"].numpy()[:n]
    x_inplace_np = x_inplace_wp.numpy()[:n]

    # Numerical reference
    L_ref = np.linalg.cholesky(prob["A_np"])
    x_ref = np.linalg.solve(prob["A_np"], prob["b_np"])

    chol_err = float(np.linalg.norm(L_np @ L_np.T - prob["A_np"]))
    res_norm = float(np.linalg.norm(prob["A_np"] @ x_np - prob["b_np"]))
    diff_ref = float(np.linalg.norm(x_np - x_ref))
    inplace_diff = float(np.linalg.norm(x_np - x_inplace_np))

    return {
        "chol_err": chol_err,
        "res_norm": res_norm,
        "diff_ref": diff_ref,
        "inplace_diff": inplace_diff,
        "L_ref_err": float(np.linalg.norm(L_np - L_ref)),
    }


###
# Timing
###


def _time_us(name, fn, num_iters, device, warmup_iters=50):
    # Warmup: clocks, caches, and scheduler need a few iterations to stabilize,
    # especially for small kernels where the first run can be 5-10x slower.
    for _ in range(warmup_iters):
        fn()
    wp.synchronize_device()

    # Take the best of K trials to reduce the impact of transient stalls
    # (e.g. GPU clocks briefly downthrottling). Each trial runs ``num_iters``
    # launches inside a single ScopedTimer so the per-launch averaging is
    # identical to before; we just pick the trial with the lowest mean.
    best = float("inf")
    for _ in range(3):
        with wp.ScopedTimer(name, print=False, synchronize=True) as t:
            for _ in range(num_iters):
                fn()
        best = min(best, t.elapsed * 1000.0 / num_iters)
    return best


def benchmark(sizes, block_size, block_dim_factor, block_dim_solve, num_iters, seed, device, verbose=True):
    factor_kernel = make_llt_blocked_factorize_kernel(block_size)
    solve_kernel = make_llt_blocked_solve_kernel(block_size)
    solve_inplace_kernel = make_llt_blocked_solve_inplace_kernel(block_size)

    if verbose:
        print(
            f"\nBenchmark settings: block_size={block_size}  "
            f"block_dim_factor={block_dim_factor}  block_dim_solve={block_dim_solve}  "
            f"num_iters={num_iters}  device={device}"
        )
        hdr = f"{'n':>6}  {'factor [us]':>12}  {'solve [us]':>12}  {'solve_ip [us]':>14}  {'chol_err':>10}  {'res_norm':>10}"
        print(hdr)
        print("-" * len(hdr))

    results = []
    for n in sizes:
        prob = _make_spd_problem(n, block_size, seed, device)

        # Warmup + correctness check
        verify = _verify(
            prob, factor_kernel, solve_kernel, solve_inplace_kernel, block_dim_factor, block_dim_solve, device
        )

        # Factor
        def _factor():
            llt_blocked_factorize(
                kernel=factor_kernel,
                num_blocks=1,
                dim=prob["dim"],
                mio=prob["mio"],
                A=prob["A"],
                L=prob["L"],
                block_dim=block_dim_factor,
                device=device,
            )

        # Solve (out of place)
        def _solve():
            llt_blocked_solve(
                kernel=solve_kernel,
                num_blocks=1,
                dim=prob["dim"],
                mio=prob["mio"],
                vio=prob["vio"],
                L=prob["L"],
                b=prob["b"],
                y=prob["y"],
                x=prob["x"],
                block_dim=block_dim_solve,
                device=device,
            )

        # Solve in-place (x holds the rhs then is overwritten with the solution)
        x_ip = wp.zeros_like(prob["b"])

        def _solve_inplace():
            wp.copy(x_ip, prob["b"])
            llt_blocked_solve_inplace(
                kernel=solve_inplace_kernel,
                num_blocks=1,
                dim=prob["dim"],
                mio=prob["mio"],
                vio=prob["vio"],
                L=prob["L"],
                y=prob["y"],
                x=x_ip,
                block_dim=block_dim_solve,
                device=device,
            )

        factor_us = _time_us("factor", _factor, num_iters, device)
        solve_us = _time_us("solve", _solve, num_iters, device)
        solve_ip_us = _time_us("solve_ip", _solve_inplace, num_iters, device)

        row = {
            "n": n,
            "factor_us": factor_us,
            "solve_us": solve_us,
            "solve_ip_us": solve_ip_us,
            **verify,
        }
        results.append(row)

        if verbose:
            print(
                f"{n:>6}  {factor_us:>12.1f}  {solve_us:>12.1f}  {solve_ip_us:>14.1f}  "
                f"{verify['chol_err']:>10.2e}  {verify['res_norm']:>10.2e}"
            )

    return results


###
# Entry point
###


def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32, 70, 128, 192, 257, 320, 401, 1000],
        help="Matrix sizes to benchmark.",
    )
    parser.add_argument("--block-size", type=int, default=16, help="Tile block size.")
    parser.add_argument(
        "--block-dim-factor",
        type=int,
        default=64,
        help="Threads per tile-block for factorize.",
    )
    parser.add_argument(
        "--block-dim-solve",
        type=int,
        default=64,
        help="Threads per tile-block for solve.",
    )
    parser.add_argument("--num-iters", type=int, default=50, help="Iterations per timed loop.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for problem generation.")
    parser.add_argument("--device", type=str, default=None, help="Warp device override (e.g. cuda:0).")

    args = parser.parse_args(argv)

    device = wp.get_device(args.device)
    with wp.ScopedDevice(args.device):
        benchmark(
            sizes=args.sizes,
            block_size=args.block_size,
            block_dim_factor=args.block_dim_factor,
            block_dim_solve=args.block_dim_solve,
            num_iters=args.num_iters,
            seed=args.seed,
            device=device,
        )


if __name__ == "__main__":
    sys.exit(main())
