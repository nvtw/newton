# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Head-to-head benchmark: LLTBlockedSolver vs LLTBlockedNDSolver.

Measures the cost of a single factorize followed by 50 solves, captured in a
CUDA graph and replayed. This mirrors how the two solvers are actually used
inside the PADMM step: the factorization is done once, then many solves share
the same factor. Both solvers are wrapped in *the same* graph capture pattern
so the comparison isolates the kernel-level differences.

Workload
--------

For each requested ``density`` the benchmark builds a multi-block batched
problem: ``num_blocks`` SPD matrices of the same size ``n``, each with a
per-matrix random sparsity pattern of roughly the requested density, packed
into the flat-batched layout expected by ``llt_blocked`` (``dim`` / ``mio`` /
``vio``). This matches the real multi-world PADMM Delassus workload.

Metrics (per solver, per density)
---------------------------------

- ``replay_ms`` : best-of-N replay time of (1 factorize + 50 solves)
- ``speedup``   : ``replay_ms(ref) / replay_ms(nd)`` ( > 1 means ND wins )
- ``max_residual`` : ``max_i ||A_i x_i - b_i||_inf`` on block 0, as a sanity
  check that both solvers produce correct answers.
- ``tile_fill``  : (ND only) fraction of non-zero tiles in the ND tile mask,
  ``nnz(tile_pattern) / num_tiles``. Lower means the ND + symbolic fill-in
  decided more tiles can be skipped. Strictly upper-bounded by the dense
  solver's effective "1.0" (it reads every tile).

The expected result: for very-low-density inputs ND should beat LLTB because
the tile mask lets it skip most of the work in both factorize and solve.
For dense inputs LLTB wins because the ND machinery adds overhead without
any skippable tiles.

Usage
-----

    python -m newton._src.solvers.kamino.benchmarks.bench_llt_blocked_vs_nd
    python -m newton._src.solvers.kamino.benchmarks.bench_llt_blocked_vs_nd --n 256 --blocks 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import float32
from newton._src.solvers.kamino._src.linalg.core import (
    DenseLinearOperatorData,
    DenseSquareMultiLinearInfo,
)
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_nd_solver import (
    LLTBlockedNDSolver,
)
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver


# ---------------------------------------------------------------------------
# Synthetic problem generation
# ---------------------------------------------------------------------------


def _random_sparse_spd(
    n: int,
    density: float,
    seed: int,
    np_dtype=np.float32,
) -> np.ndarray:
    """SPD matrix with a controlled off-diagonal nonzero pattern density.

    Strategy: draw a symmetric Bernoulli mask with the requested off-diagonal
    density, fill the allowed entries with small Gaussian noise, then make
    the result strictly diagonally dominant by setting
    ``A[i,i] = sum_{j != i} |A[i,j]| + 1``. This gives a pattern whose
    *actual* density matches the requested density (to Bernoulli noise),
    which is what we need to study how the ND solver behaves as the input
    gets sparser.
    """
    rng = np.random.default_rng(seed)
    density = float(np.clip(density, 0.0, 1.0))

    # 1. Symmetric off-diagonal Bernoulli mask.
    iu = np.triu_indices(n, k=1)
    mask_upper = rng.random(len(iu[0])) < density
    mask = np.zeros((n, n), dtype=bool)
    mask[iu] = mask_upper
    mask = mask | mask.T

    # 2. Fill allowed off-diagonal entries with small Gaussian noise.
    vals_upper = (rng.standard_normal(len(iu[0])) * np_dtype(0.1)).astype(np_dtype)
    vals_upper = vals_upper * mask_upper  # zero out non-pattern entries
    M = np.zeros((n, n), dtype=np_dtype)
    M[iu] = vals_upper
    M = M + M.T

    # 3. Diagonally dominate to guarantee SPD.
    D = (np.abs(M).sum(axis=1) + np_dtype(1.0)).astype(np_dtype)
    return M + np.diag(D)


def _banded_sparse_spd(
    n: int,
    bandwidth_frac: float,
    seed: int,
    scramble: bool = True,
    np_dtype=np.float32,
) -> np.ndarray:
    """SPD matrix whose sparsity pattern has genuine *bandwidth structure*.

    Builds a banded matrix of bandwidth ``int(bandwidth_frac * n)``, then
    (if ``scramble=True``) applies a random symmetric permutation to hide the
    bandwidth. RCM-style reordering should be able to recover the structure.
    """
    rng = np.random.default_rng(seed)
    bw = max(1, int(round(bandwidth_frac * n)))
    M = np.zeros((n, n), dtype=np_dtype)
    for i in range(n):
        for j in range(max(0, i - bw), min(n, i + bw + 1)):
            if i == j:
                continue
            if i < j:
                v = np_dtype(0.1 * rng.standard_normal())
                M[i, j] = v
                M[j, i] = v
    D = (np.abs(M).sum(axis=1) + np_dtype(1.0)).astype(np_dtype)
    M = M + np.diag(D)
    if scramble:
        perm = rng.permutation(n)
        M = M[perm][:, perm]
    return M


def _pattern_density(A: np.ndarray, tol: float = 0.0) -> float:
    """Off-diagonal nonzero density of a matrix."""
    n = A.shape[0]
    mask = np.abs(A) > tol
    off = int(mask.sum()) - int(np.diag(mask).sum())
    total = n * (n - 1)
    return float(off) / float(max(1, total))


def _pack_flat(A_blocks: list[np.ndarray], b_blocks: list[np.ndarray], device: wp.Device):
    """Pack per-block matrices/vectors into the flat-batched layout."""
    dims = [A.shape[0] for A in A_blocks]
    mat_sizes = [d * d for d in dims]
    vec_sizes = dims[:]
    mio: list[int] = [0]
    for s in mat_sizes[:-1]:
        mio.append(mio[-1] + s)
    vio: list[int] = [0]
    for s in vec_sizes[:-1]:
        vio.append(vio[-1] + s)

    A_flat = np.zeros(sum(mat_sizes), dtype=np.float32)
    b_flat = np.zeros(sum(vec_sizes), dtype=np.float32)
    for i, (A, b) in enumerate(zip(A_blocks, b_blocks, strict=False)):
        A_flat[mio[i] : mio[i] + mat_sizes[i]] = A.astype(np.float32).reshape(-1)
        b_flat[vio[i] : vio[i] + vec_sizes[i]] = b.astype(np.float32)

    A_wp = wp.array(A_flat, dtype=wp.float32, device=device)
    b_wp = wp.array(b_flat, dtype=wp.float32, device=device)
    return A_wp, b_wp, dims, mio, vio


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _min_replay_ms(graph: wp.Graph, n_warmup: int, n_timed: int, device: wp.Device) -> float:
    """Minimum replay time (ms) over ``n_timed`` samples, after ``n_warmup`` warmups."""
    for _ in range(n_warmup):
        wp.capture_launch(graph)
    wp.synchronize_device(device)

    best_ms = float("inf")
    for _ in range(n_timed):
        t0 = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        best_ms = min(best_ms, dt_ms)
    return best_ms


def _build_solver_graph(
    run_once: Callable[[], None],
    device: wp.Device,
) -> wp.Graph:
    """Capture (1 factorize + 50 solves) into a single CUDA graph.

    ``run_once`` is a closure that calls ``solver.compute(A)`` followed by
    50 ``solver.solve(b, x)`` launches. We capture the whole sequence and
    return the graph handle.
    """
    with wp.ScopedDevice(device):
        wp.capture_begin()
        try:
            run_once()
        finally:
            graph = wp.capture_end()
    return graph


# ---------------------------------------------------------------------------
# Benchmark case
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    density_target: float
    density_actual: float
    n: int
    num_blocks: int
    ref_replay_ms: float
    nd_replay_ms: float
    rcm_replay_ms: float
    speedup: float        # LLTB / ND-spectral
    speedup_rcm: float    # LLTB / ND-RCM
    ref_max_residual: float
    nd_max_residual: float
    rcm_max_residual: float
    nd_tile_fill: float
    rcm_tile_fill: float


def _block_residual_max(A: np.ndarray, x_flat: np.ndarray, b_flat: np.ndarray,
                        dims: list[int], vio: list[int], bi: int) -> float:
    """``||A_bi x_bi - b_bi||_inf`` normalized by ``||b_bi||_inf``."""
    s = vio[bi]
    e = s + dims[bi]
    xi = x_flat[s:e]
    bi_ = b_flat[s:e]
    res = A @ xi - bi_
    return float(np.max(np.abs(res)) / max(1e-8, float(np.max(np.abs(bi_)))))


def run_case(
    n: int,
    num_blocks: int,
    density: float,
    n_solves: int,
    n_warmup: int,
    n_timed: int,
    block_size: int,
    device: wp.Device,
    seed: int,
    nd_power_iters: int | str | None = None,
    matrix_kind: str = "random",
) -> CaseResult:
    # 1) Build synthetic per-block sparse SPDs.
    if matrix_kind == "random":
        A_blocks = [
            _random_sparse_spd(n, density=density, seed=seed + bi)
            for bi in range(num_blocks)
        ]
    elif matrix_kind == "banded":
        # Use the requested density to pick a bandwidth: for a banded matrix
        # with bw, off-diag density = 2 * bw / (n - 1) approximately. So
        # bw = ceil(density * (n - 1) / 2). This lets the same density sweep
        # compare apples-to-apples with the 'random' kind.
        A_blocks = [
            _banded_sparse_spd(
                n,
                bandwidth_frac=max(1.0 / n, density / 2.0),
                seed=seed + bi,
                scramble=True,
            )
            for bi in range(num_blocks)
        ]
    else:
        raise ValueError(f"matrix_kind must be 'random' or 'banded', got {matrix_kind!r}")
    rng = np.random.default_rng(seed + 10_000)
    b_blocks = [rng.standard_normal(n).astype(np.float32) for _ in range(num_blocks)]

    # Record the *actual* density after construction (may differ slightly from target).
    density_actual = float(np.mean([_pattern_density(A) for A in A_blocks]))

    # 2) Pack into flat-batched layout.
    A_wp, b_wp, dims, _mio, vio = _pack_flat(A_blocks, b_blocks, device)

    opinfo = DenseSquareMultiLinearInfo()
    opinfo.finalize(dimensions=dims, dtype=float32, device=device)
    operator = DenseLinearOperatorData(info=opinfo, mat=A_wp)

    # 3) Build all three solvers on independent operators (each mutates its mat).
    #    Use the same sparsity pattern (from A_blocks above) for fair comparison.
    A_wp_ref, _, _, _, _ = _pack_flat(A_blocks, b_blocks, device)
    A_wp_nd, _, _, _, _ = _pack_flat(A_blocks, b_blocks, device)
    A_wp_rcm, _, _, _, _ = _pack_flat(A_blocks, b_blocks, device)

    opinfo_ref = DenseSquareMultiLinearInfo()
    opinfo_ref.finalize(dimensions=dims, dtype=float32, device=device)
    op_ref = DenseLinearOperatorData(info=opinfo_ref, mat=A_wp_ref)

    opinfo_nd = DenseSquareMultiLinearInfo()
    opinfo_nd.finalize(dimensions=dims, dtype=float32, device=device)
    op_nd = DenseLinearOperatorData(info=opinfo_nd, mat=A_wp_nd)

    opinfo_rcm = DenseSquareMultiLinearInfo()
    opinfo_rcm.finalize(dimensions=dims, dtype=float32, device=device)
    op_rcm = DenseLinearOperatorData(info=opinfo_rcm, mat=A_wp_rcm)

    ref = LLTBlockedSolver(
        operator=op_ref, block_size=block_size,
        dtype=float32, device=device,
    )
    nd_kwargs: dict = {}
    if nd_power_iters is not None:
        nd_kwargs["nd_power_iters"] = nd_power_iters
    nd = LLTBlockedNDSolver(
        operator=op_nd, block_size=block_size,
        dtype=float32, device=device,
        reorder_algorithm="spectral",
        **nd_kwargs,
    )
    rcm = LLTBlockedNDSolver(
        operator=op_rcm, block_size=block_size,
        dtype=float32, device=device,
        reorder_algorithm="rcm_batch",
    )

    # Output buffers.
    x_ref = wp.zeros_like(b_wp, device=device)
    x_nd = wp.zeros_like(b_wp, device=device)
    x_rcm = wp.zeros_like(b_wp, device=device)

    # 4) Warm the kernel cache with a non-captured run (JIT compile happens here).
    ref.compute(A_wp_ref)
    ref.solve(b_wp, x_ref)
    nd.compute(A_wp_nd)
    nd.solve(b_wp, x_nd)
    rcm.compute(A_wp_rcm)
    rcm.solve(b_wp, x_rcm)
    wp.synchronize_device(device)

    # 5) Correctness sanity on block 0 before timing.
    x_ref_np = x_ref.numpy()
    x_nd_np = x_nd.numpy()
    x_rcm_np = x_rcm.numpy()
    b_np = b_wp.numpy()
    ref_res = _block_residual_max(A_blocks[0], x_ref_np, b_np, dims, vio, 0)
    nd_res = _block_residual_max(A_blocks[0], x_nd_np, b_np, dims, vio, 0)
    rcm_res = _block_residual_max(A_blocks[0], x_rcm_np, b_np, dims, vio, 0)

    # 6) Capture (1 factorize + n_solves solves) as a single CUDA graph per solver.
    def run_ref():
        ref.compute(A_wp_ref)
        for _ in range(n_solves):
            ref.solve(b_wp, x_ref)

    def run_nd():
        nd.compute(A_wp_nd)
        for _ in range(n_solves):
            nd.solve(b_wp, x_nd)

    def run_rcm():
        rcm.compute(A_wp_rcm)
        for _ in range(n_solves):
            rcm.solve(b_wp, x_rcm)

    graph_ref = _build_solver_graph(run_ref, device)
    graph_nd = _build_solver_graph(run_nd, device)
    graph_rcm = _build_solver_graph(run_rcm, device)

    ref_ms = _min_replay_ms(graph_ref, n_warmup=n_warmup, n_timed=n_timed, device=device)
    nd_ms = _min_replay_ms(graph_nd, n_warmup=n_warmup, n_timed=n_timed, device=device)
    rcm_ms = _min_replay_ms(graph_rcm, n_warmup=n_warmup, n_timed=n_timed, device=device)

    # 7) Report per-backend tile-pattern fill ratio (skippability measure).
    tp_nd = nd.tile_pattern.numpy()
    tp_rcm = rcm.tile_pattern.numpy()
    nd_tile_fill = float(np.count_nonzero(tp_nd)) / float(max(1, tp_nd.size))
    rcm_tile_fill = float(np.count_nonzero(tp_rcm)) / float(max(1, tp_rcm.size))

    return CaseResult(
        density_target=density,
        density_actual=density_actual,
        n=n,
        num_blocks=num_blocks,
        ref_replay_ms=ref_ms,
        nd_replay_ms=nd_ms,
        rcm_replay_ms=rcm_ms,
        speedup=ref_ms / max(1e-9, nd_ms),
        speedup_rcm=ref_ms / max(1e-9, rcm_ms),
        ref_max_residual=ref_res,
        nd_max_residual=nd_res,
        rcm_max_residual=rcm_res,
        nd_tile_fill=nd_tile_fill,
        rcm_tile_fill=rcm_tile_fill,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


DEFAULT_DENSITIES = (0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256, help="per-block matrix size")
    parser.add_argument("--blocks", type=int, default=8, help="number of blocks")
    parser.add_argument("--solves", type=int, default=50, help="solves per captured graph")
    parser.add_argument("--block-size", type=int, default=16, help="LLT block/tile size")
    parser.add_argument("--warmup", type=int, default=3, help="captured-graph warmup replays")
    parser.add_argument("--timed", type=int, default=11, help="captured-graph timed replays (best taken)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--densities",
        type=str,
        default=",".join(f"{d:g}" for d in DEFAULT_DENSITIES),
        help="comma-separated list of target off-diagonal densities",
    )
    parser.add_argument("--device", type=str, default=None, help="warp device (e.g. cuda:0)")
    parser.add_argument("--json-out", type=str, default=None, help="optional JSON result path")
    parser.add_argument(
        "--nd-power-iters",
        type=str,
        default=None,
        help="override ND power_iters (int or 'auto'); None = solver default",
    )
    parser.add_argument(
        "--matrix-kind",
        type=str,
        choices=("random", "banded"),
        default="random",
        help="'random' = Bernoulli off-diagonal pattern; 'banded' = banded "
        "pattern scrambled by a random permutation (RCM should excel)",
    )
    args = parser.parse_args()

    nd_power_iters: int | str | None
    if args.nd_power_iters is None:
        nd_power_iters = None
    elif args.nd_power_iters == "auto":
        nd_power_iters = "auto"
    else:
        nd_power_iters = int(args.nd_power_iters)

    wp.init()
    device = wp.get_device(args.device)
    print(f"Device: {device}")
    print(
        f"Workload: {args.blocks} block(s) of n={args.n}, "
        f"{args.solves} solves per captured graph, block_size={args.block_size}, "
        f"matrix_kind={args.matrix_kind}"
    )
    print(f"Timing: best of {args.timed} replay(s) after {args.warmup} warmup replay(s)")
    print()

    densities = [float(s) for s in args.densities.split(",") if s.strip()]

    results: list[CaseResult] = []
    for density in densities:
        t0 = time.perf_counter()
        res = run_case(
            n=args.n,
            num_blocks=args.blocks,
            density=density,
            n_solves=args.solves,
            n_warmup=args.warmup,
            n_timed=args.timed,
            block_size=args.block_size,
            device=device,
            seed=args.seed,
            nd_power_iters=nd_power_iters,
            matrix_kind=args.matrix_kind,
        )
        dt = time.perf_counter() - t0
        results.append(res)
        print(
            f"  density={res.density_target:6.4f} "
            f"(actual={res.density_actual:6.4f}): "
            f"LLTB={res.ref_replay_ms:8.3f}ms  "
            f"NDs={res.nd_replay_ms:8.3f}ms (sp={res.speedup:4.2f}x, fill={res.nd_tile_fill:.3f})  "
            f"RCM={res.rcm_replay_ms:8.3f}ms (sp={res.speedup_rcm:4.2f}x, fill={res.rcm_tile_fill:.3f})  "
            f"[case {dt:4.1f}s]"
        )

    # Summary table
    print()
    print("Summary (best-of-N replay_ms; sp = LLTB / backend)")
    print("-" * 120)
    print(
        f"{'target%':>8} {'actual%':>8} {'LLTB ms':>10} "
        f"{'NDspec ms':>10} {'sp(ND)':>7} {'fill(ND)':>9} "
        f"{'RCM ms':>10} {'sp(RCM)':>8} {'fill(RCM)':>10} "
        f"{'winner':>10}"
    )
    print("-" * 120)
    for r in results:
        best = min(r.ref_replay_ms, r.nd_replay_ms, r.rcm_replay_ms)
        if best == r.ref_replay_ms:
            winner = "LLTB"
        elif best == r.rcm_replay_ms:
            winner = "RCM"
        else:
            winner = "ND-spec"
        print(
            f"{r.density_target * 100:>7.2f}% "
            f"{r.density_actual * 100:>7.2f}% "
            f"{r.ref_replay_ms:>10.3f} "
            f"{r.nd_replay_ms:>10.3f} {r.speedup:>6.2f}x {r.nd_tile_fill:>9.4f} "
            f"{r.rcm_replay_ms:>10.3f} {r.speedup_rcm:>7.2f}x {r.rcm_tile_fill:>10.4f} "
            f"{winner:>10}"
        )
    print()

    # Optional JSON dump.
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(
                {
                    "device": str(device),
                    "n": args.n,
                    "num_blocks": args.blocks,
                    "n_solves": args.solves,
                    "block_size": args.block_size,
                    "seed": args.seed,
                    "results": [asdict(r) for r in results],
                },
                fh,
                indent=2,
            )
        print(f"JSON results written to {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
