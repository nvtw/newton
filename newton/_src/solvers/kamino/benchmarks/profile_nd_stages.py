# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: break out LLTBlockedNDSolver wall time into stages.

Captures each sub-step of the ND factorize/solve as its own CUDA graph
and times best-of-N replay. This tells us which stage is responsible
for the ~8x gap vs LLTBlockedSolver that the head-to-head benchmark
reported, so the optimization work can target the right thing.

Stages (per captured graph):

- stage_nd_perblock  : all per-block ND launches (compute P)
- stage_build_invP   : build inv_P from P
- stage_permute_A    : A -> A_hat (symmetric permutation)
- stage_symbolic     : zero_ + build_tile_pattern + symbolic_fill_in
- stage_nd_factorize : tile-pattern-aware numeric factorize of A_hat
- stage_solve_once   : permute b + nd_solve + un-permute  (single solve)
- stage_solve_x50    : 50 full solves

The first five stages together constitute ``compute(A)``; together they
should sum to roughly the time of a captured-graph ``compute(A)`` alone.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import float32
from newton._src.solvers.kamino._src.linalg.core import (
    DenseLinearOperatorData,
    DenseSquareMultiLinearInfo,
)
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_nd import (
    llt_blocked_nd_build_inv_P,
    llt_blocked_nd_build_tile_pattern,
    llt_blocked_nd_factorize,
    llt_blocked_nd_permute_matrix,
    llt_blocked_nd_permute_vector,
    llt_blocked_nd_solve,
    llt_blocked_nd_symbolic_fill_in,
)
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_nd_solver import (
    LLTBlockedNDSolver,
)
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver


def _random_sparse_spd(n: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    density = float(np.clip(density, 0.0, 1.0))
    iu = np.triu_indices(n, k=1)
    mask_upper = rng.random(len(iu[0])) < density
    vals_upper = (rng.standard_normal(len(iu[0])) * 0.1).astype(np.float32) * mask_upper
    M = np.zeros((n, n), dtype=np.float32)
    M[iu] = vals_upper
    M = M + M.T
    D = (np.abs(M).sum(axis=1) + 1.0).astype(np.float32)
    return M + np.diag(D)


def _pack_flat(A_blocks, b_blocks, device):
    dims = [A.shape[0] for A in A_blocks]
    mat_sizes = [d * d for d in dims]
    vec_sizes = dims[:]
    mio = [0]
    for s in mat_sizes[:-1]:
        mio.append(mio[-1] + s)
    vio = [0]
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


def _min_ms(graph, device, warmup=3, timed=11):
    for _ in range(warmup):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    best = float("inf")
    for _ in range(timed):
        t0 = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _capture(fn, device) -> wp.Graph:
    with wp.ScopedDevice(device):
        wp.capture_begin()
        try:
            fn()
        finally:
            g = wp.capture_end()
    return g


def main() -> None:
    wp.init()
    device = wp.get_device(None)
    print(f"Device: {device}")

    # Same workload as the head-to-head benchmark: 8 blocks x n=256, 0.5% density
    # (the sparsest case, where ND should theoretically win the most).
    n = 256
    num_blocks = 8
    density = 0.005
    block_size = 16
    seed = 42
    print(f"Workload: {num_blocks} blocks of n={n}, density={density:g}, block_size={block_size}")
    print()

    A_blocks = [_random_sparse_spd(n, density, seed + bi) for bi in range(num_blocks)]
    rng = np.random.default_rng(seed + 10_000)
    b_blocks = [rng.standard_normal(n).astype(np.float32) for _ in range(num_blocks)]
    A_wp, b_wp, dims, _mio, _vio = _pack_flat(A_blocks, b_blocks, device)

    opinfo = DenseSquareMultiLinearInfo()
    opinfo.finalize(dimensions=dims, dtype=float32, device=device)
    operator = DenseLinearOperatorData(info=opinfo, mat=A_wp)

    ref = LLTBlockedSolver(
        operator=operator, block_size=block_size, dtype=float32, device=device,
    )
    nd = LLTBlockedNDSolver(
        operator=operator, block_size=block_size, dtype=float32, device=device,
    )

    x_ref = wp.zeros_like(b_wp, device=device)
    x_nd = wp.zeros_like(b_wp, device=device)

    # Warm JITs.
    ref.compute(A_wp)
    ref.solve(b_wp, x_ref)
    nd.compute(A_wp)
    nd.solve(b_wp, x_nd)
    wp.synchronize_device(device)

    # --- Reference timings (whole captured graphs, for context) ---
    g_ref_all = _capture(lambda: (ref.compute(A_wp), *[ref.solve(b_wp, x_ref) for _ in range(50)]), device)
    g_nd_all = _capture(lambda: (nd.compute(A_wp), *[nd.solve(b_wp, x_nd) for _ in range(50)]), device)
    g_ref_compute = _capture(lambda: ref.compute(A_wp), device)
    g_ref_solve_x50 = _capture(lambda: [ref.solve(b_wp, x_ref) for _ in range(50)], device)
    g_nd_compute = _capture(lambda: nd.compute(A_wp), device)
    g_nd_solve_x50 = _capture(lambda: [nd.solve(b_wp, x_nd) for _ in range(50)], device)

    print(f"Totals (1 factorize + 50 solves, captured):")
    print(f"  LLTB total   : {_min_ms(g_ref_all, device):8.3f} ms  (compute={_min_ms(g_ref_compute, device):6.3f}  solvex50={_min_ms(g_ref_solve_x50, device):6.3f})")
    print(f"  ND   total   : {_min_ms(g_nd_all, device):8.3f} ms  (compute={_min_ms(g_nd_compute, device):6.3f}  solvex50={_min_ms(g_nd_solve_x50, device):6.3f})")
    print()

    # --- ND stage-level timings ---
    info = nd._operator.info  # we intentionally peek at internals for profiling

    def stage_nd_perblock():
        nd._ensure_nd_launches_bound(A_wp)
        for cb in nd._nd_callbacks:
            cb()

    def stage_build_invP():
        llt_blocked_nd_build_inv_P(
            kernel=nd._build_inv_P_kernel,
            dim=info.dim, vio=info.vio,
            P=nd._P, inv_P=nd._inv_P,
            num_blocks=info.num_blocks, max_dim=nd._max_dim, device=device,
        )

    def stage_permute_A():
        llt_blocked_nd_permute_matrix(
            kernel=nd._permute_matrix_kernel,
            dim=info.dim, mio=info.mio, vio=info.vio,
            P=nd._P, A=A_wp, A_hat=nd._A_hat,
            num_blocks=info.num_blocks, max_dim=nd._max_dim, device=device,
        )

    def stage_symbolic():
        nd._tile_pattern.zero_()
        llt_blocked_nd_build_tile_pattern(
            kernel=nd._build_tile_pattern_kernel,
            dim=info.dim, mio=info.mio, tpo=nd._tpo,
            tol=nd._nd_tol, A_hat=nd._A_hat, tile_pattern=nd._tile_pattern,
            num_blocks=info.num_blocks, max_dim=nd._max_dim, device=device,
        )
        llt_blocked_nd_symbolic_fill_in(
            kernel=nd._symbolic_fill_in_kernel,
            dim=info.dim, tpo=nd._tpo, block_size=nd._block_size,
            tile_pattern=nd._tile_pattern,
            num_blocks=info.num_blocks, device=device,
        )

    def stage_nd_factorize():
        llt_blocked_nd_factorize(
            kernel=nd._factorize_kernel,
            dim=info.dim, mio=info.mio, tpo=nd._tpo,
            A=nd._A_hat, L=nd._L, tile_pattern=nd._tile_pattern,
            num_blocks=info.num_blocks, block_dim=nd._factortize_block_dim, device=device,
        )

    def stage_solve_once():
        llt_blocked_nd_permute_vector(
            kernel=nd._permute_vector_kernel,
            dim=info.dim, vio=info.vio,
            P=nd._P, src=b_wp, dst=nd._b_hat,
            num_blocks=info.num_blocks, max_dim=nd._max_dim, device=device,
        )
        llt_blocked_nd_solve(
            kernel=nd._solve_kernel,
            dim=info.dim, mio=info.mio, vio=info.vio, tpo=nd._tpo,
            L=nd._L, tile_pattern=nd._tile_pattern,
            b=nd._b_hat, y=nd._y, x=nd._x_hat,
            num_blocks=info.num_blocks, block_dim=nd._solve_block_dim, device=device,
        )
        llt_blocked_nd_permute_vector(
            kernel=nd._permute_vector_kernel,
            dim=info.dim, vio=info.vio,
            P=nd._inv_P, src=nd._x_hat, dst=x_nd,
            num_blocks=info.num_blocks, max_dim=nd._max_dim, device=device,
        )

    def stage_solve_x50():
        for _ in range(50):
            stage_solve_once()

    stages = [
        ("stage_nd_perblock    ", stage_nd_perblock),
        ("stage_build_invP     ", stage_build_invP),
        ("stage_permute_A      ", stage_permute_A),
        ("stage_symbolic       ", stage_symbolic),
        ("stage_nd_factorize   ", stage_nd_factorize),
        ("stage_solve_once     ", stage_solve_once),
        ("stage_solve_x50      ", stage_solve_x50),
    ]

    print("ND per-stage captured-graph replay time:")
    print(f"  {'stage':22} {'min_ms':>10}")
    for name, fn in stages:
        g = _capture(fn, device)
        ms = _min_ms(g, device)
        print(f"  {name} {ms:>10.3f}")


if __name__ == "__main__":
    main()
