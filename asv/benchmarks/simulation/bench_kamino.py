# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton.examples
from newton.examples.kamino.example_kamino_robot_dr_legs import Example as ExampleDrLegs
from newton.viewer import ViewerNull

CHOL_N = 92
CHOL_TILE = 16
CHOL_N_BLOCKS = (CHOL_N + CHOL_TILE - 1) // CHOL_TILE
CHOL_N_PADDED = CHOL_N_BLOCKS * CHOL_TILE
CHOL_DEFAULT_ENVS = 4096
SOLVE_TILE = 32
SOLVE_DEFAULT_ENVS = 32768

WP_CHOL_TILE = wp.constant(CHOL_TILE)
WP_CHOL_N_BLOCKS = wp.constant(CHOL_N_BLOCKS)
WP_SOLVE_TILE = wp.constant(SOLVE_TILE)


def _has_warp_builtin(name: str) -> bool:
    try:
        from warp._src.context import builtin_functions  # noqa: PLC0415
    except Exception:
        return False
    return name in builtin_functions


def _make_spd_matrices(n_envs: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    r = rng.standard_normal((n_envs, CHOL_N, CHOL_N)).astype(np.float32)
    a = np.einsum("bij,bik->bjk", r, r) + np.eye(CHOL_N, dtype=np.float32)

    a_padded = np.zeros((n_envs, CHOL_N_PADDED, CHOL_N_PADDED), dtype=np.float32)
    a_padded[:, :CHOL_N, :CHOL_N] = a
    for i in range(CHOL_N, CHOL_N_PADDED):
        a_padded[:, i, i] = 1.0
    return a_padded


@wp.kernel(enable_backward=False)
def _cholesky_warp_tile(
    a: wp.array3d[wp.float32],
    l: wp.array3d[wp.float32],
):
    env = wp.tid()

    for kb in range(WP_CHOL_N_BLOCKS):
        k0 = kb * WP_CHOL_TILE

        a_kk = wp.tile_load(a[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(k0, k0), bounds_check=False)

        for jb in range(kb):
            j0 = jb * WP_CHOL_TILE
            l_kj = wp.tile_load(l[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(k0, j0), bounds_check=False)
            l_kj_t = wp.tile_transpose(l_kj)
            wp.tile_matmul(l_kj, l_kj_t, a_kk, alpha=-1.0)

        l_kk = wp.tile_cholesky(a_kk)
        wp.tile_store(l[env], l_kk, offset=(k0, k0), bounds_check=False)

        for ib in range(kb + 1, WP_CHOL_N_BLOCKS):
            i0 = ib * WP_CHOL_TILE
            a_ik = wp.tile_load(a[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(i0, k0), bounds_check=False)

            for jb in range(kb):
                j0 = jb * WP_CHOL_TILE
                l_ij = wp.tile_load(l[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(i0, j0), bounds_check=False)
                l_kj = wp.tile_load(l[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(k0, j0), bounds_check=False)
                l_kj_t = wp.tile_transpose(l_kj)
                wp.tile_matmul(l_ij, l_kj_t, a_ik, alpha=-1.0)

            a_ik_t = wp.tile_transpose(a_ik)
            wp.tile_lower_solve_inplace(l_kk, a_ik_t)
            l_ik = wp.tile_transpose(a_ik_t)
            wp.tile_store(l[env], l_ik, offset=(i0, k0), bounds_check=False)


@wp.kernel(enable_backward=False)
def _cholesky_warp_tile_matmul_transpose_update(
    a: wp.array3d[wp.float32],
    l: wp.array3d[wp.float32],
):
    env = wp.tid()

    for kb in range(WP_CHOL_N_BLOCKS):
        k0 = kb * WP_CHOL_TILE

        a_kk = wp.tile_load(a[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(k0, k0), bounds_check=False)

        for jb in range(kb):
            j0 = jb * WP_CHOL_TILE
            l_kj = wp.tile_load(l[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(k0, j0), bounds_check=False)
            wp.tile_matmul_transpose_update(a_kk, l_kj, l_kj, alpha=-1.0)

        l_kk = wp.tile_cholesky(a_kk)
        wp.tile_store(l[env], l_kk, offset=(k0, k0), bounds_check=False)

        for ib in range(kb + 1, WP_CHOL_N_BLOCKS):
            i0 = ib * WP_CHOL_TILE
            a_ik = wp.tile_load(a[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(i0, k0), bounds_check=False)

            for jb in range(kb):
                j0 = jb * WP_CHOL_TILE
                l_ij = wp.tile_load(l[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(i0, j0), bounds_check=False)
                l_kj = wp.tile_load(l[env], shape=(WP_CHOL_TILE, WP_CHOL_TILE), offset=(k0, j0), bounds_check=False)
                wp.tile_matmul_transpose_update(a_ik, l_ij, l_kj, alpha=-1.0)

            a_ik_t = wp.tile_transpose(a_ik)
            wp.tile_lower_solve_inplace(l_kk, a_ik_t)
            l_ik = wp.tile_transpose(a_ik_t)
            wp.tile_store(l[env], l_ik, offset=(i0, k0), bounds_check=False)


_CHOL_KERNELS = {
    "tile_matmul": _cholesky_warp_tile,
    "tile_matmul_transpose_update": _cholesky_warp_tile_matmul_transpose_update,
}


@wp.kernel(enable_backward=False)
def _back_solve_update_tile_matmul(
    l: wp.array3d[wp.float32],
    x: wp.array3d[wp.float32],
    rhs: wp.array3d[wp.float32],
    out: wp.array3d[wp.float32],
):
    env = wp.tid()
    l_tile = wp.tile_load(l[env], shape=(WP_SOLVE_TILE, WP_SOLVE_TILE), bounds_check=False)
    x_tile = wp.tile_load(x[env], shape=(WP_SOLVE_TILE, 1), bounds_check=False)
    rhs_tile = wp.tile_load(rhs[env], shape=(WP_SOLVE_TILE, 1), bounds_check=False)
    l_t_tile = wp.tile_transpose(l_tile)
    wp.tile_matmul(l_t_tile, x_tile, rhs_tile, alpha=-1.0)
    wp.tile_store(out[env], rhs_tile, bounds_check=False)


@wp.kernel(enable_backward=False)
def _back_solve_update_left_transpose(
    l: wp.array3d[wp.float32],
    x: wp.array3d[wp.float32],
    rhs: wp.array3d[wp.float32],
    out: wp.array3d[wp.float32],
):
    env = wp.tid()
    l_tile = wp.tile_load(l[env], shape=(WP_SOLVE_TILE, WP_SOLVE_TILE), bounds_check=False)
    x_tile = wp.tile_load(x[env], shape=(WP_SOLVE_TILE, 1), bounds_check=False)
    rhs_tile = wp.tile_load(rhs[env], shape=(WP_SOLVE_TILE, 1), bounds_check=False)
    wp.tile_matmul_left_transpose_update(rhs_tile, l_tile, x_tile, alpha=-1.0)
    wp.tile_store(out[env], rhs_tile, bounds_check=False)


_BACK_SOLVE_UPDATE_KERNELS = {
    "tile_matmul": _back_solve_update_tile_matmul,
    "tile_matmul_left_transpose_update": _back_solve_update_left_transpose,
}


def _cuda_device():
    if wp.get_cuda_device_count() == 0:
        raise SkipNotImplemented
    return wp.get_device("cuda:0")


def _allocate_cholesky_problem(n_envs: int, device: wp.Device):
    a_np = _make_spd_matrices(n_envs)
    a = wp.array(a_np, dtype=wp.float32, device=device)
    l = wp.zeros((n_envs, CHOL_N_PADDED, CHOL_N_PADDED), dtype=wp.float32, device=device)
    return a_np, a, l


def _allocate_back_solve_update_problem(n_envs: int, device: wp.Device):
    rng = np.random.default_rng(43)
    l_np = rng.standard_normal((n_envs, SOLVE_TILE, SOLVE_TILE)).astype(np.float32)
    x_np = rng.standard_normal((n_envs, SOLVE_TILE, 1)).astype(np.float32)
    rhs_np = rng.standard_normal((n_envs, SOLVE_TILE, 1)).astype(np.float32)
    l = wp.array(l_np, dtype=wp.float32, device=device)
    x = wp.array(x_np, dtype=wp.float32, device=device)
    rhs = wp.array(rhs_np, dtype=wp.float32, device=device)
    out = wp.zeros_like(rhs)
    return l, x, rhs, out


def _benchmark_back_solve_update_kernel(
    kernel, l, x, rhs, out, block_dim: int, warmup: int, iters: int, device: wp.Device
) -> float:
    for _ in range(warmup):
        wp.launch_tiled(kernel, dim=l.shape[0], inputs=[l, x, rhs, out], block_dim=block_dim, device=device)
    wp.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(iters):
        wp.launch_tiled(kernel, dim=l.shape[0], inputs=[l, x, rhs, out], block_dim=block_dim, device=device)
    wp.synchronize_device(device)
    return (time.perf_counter() - start) * 1.0e6 / iters


def validate_back_solve_tile_update(block_dim: int = 256, n_envs: int = 64) -> float:
    if not _has_warp_builtin("tile_matmul_left_transpose_update"):
        raise SkipNotImplemented

    device = _cuda_device()
    l, x, rhs, baseline = _allocate_back_solve_update_problem(n_envs, device)
    update = wp.zeros_like(baseline)

    wp.launch_tiled(
        _back_solve_update_tile_matmul, dim=n_envs, inputs=[l, x, rhs, baseline], block_dim=block_dim, device=device
    )
    wp.launch_tiled(
        _back_solve_update_left_transpose,
        dim=n_envs,
        inputs=[l, x, rhs, update],
        block_dim=block_dim,
        device=device,
    )
    wp.synchronize_device(device)

    max_diff = float(np.max(np.abs(update.numpy() - baseline.numpy())))
    if not np.isfinite(max_diff):
        raise AssertionError("Back-solve tile update produced non-finite output")
    return max_diff


def _benchmark_cholesky_kernel(kernel, a, l, block_dim: int, warmup: int, iters: int, device: wp.Device) -> float:
    for _ in range(warmup):
        wp.launch_tiled(kernel, dim=a.shape[0], inputs=[a, l], block_dim=block_dim, device=device)
    wp.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(iters):
        wp.launch_tiled(kernel, dim=a.shape[0], inputs=[a, l], block_dim=block_dim, device=device)
    wp.synchronize_device(device)
    return (time.perf_counter() - start) * 1.0e6 / iters


def validate_cholesky_tile_update(block_dim: int = 64, n_envs: int = 8) -> tuple[float, float]:
    if not _has_warp_builtin("tile_matmul_transpose_update"):
        raise SkipNotImplemented

    device = _cuda_device()
    a_np, a, l_baseline = _allocate_cholesky_problem(n_envs, device)
    l_update = wp.zeros_like(l_baseline)

    wp.launch_tiled(_cholesky_warp_tile, dim=n_envs, inputs=[a, l_baseline], block_dim=block_dim, device=device)
    wp.launch_tiled(
        _cholesky_warp_tile_matmul_transpose_update,
        dim=n_envs,
        inputs=[a, l_update],
        block_dim=block_dim,
        device=device,
    )
    wp.synchronize_device(device)

    baseline_np = np.tril(l_baseline.numpy()[:, :CHOL_N, :CHOL_N])
    update_np = np.tril(l_update.numpy()[:, :CHOL_N, :CHOL_N])
    max_diff = float(np.max(np.abs(update_np - baseline_np)))

    max_reconstruction_error = 0.0
    for i in range(n_envs):
        recon = update_np[i] @ update_np[i].T
        err = float(np.max(np.abs(recon - a_np[i, :CHOL_N, :CHOL_N])))
        max_reconstruction_error = max(max_reconstruction_error, err)

    if not np.isfinite(max_diff) or not np.isfinite(max_reconstruction_error):
        raise AssertionError("Cholesky tile update produced non-finite output")

    return max_diff, max_reconstruction_error


class FastKaminoCholeskyTileUpdate:
    param_names = ["kernel", "block_dim"]
    params = [["tile_matmul", "tile_matmul_transpose_update"], [32, 64, 128, 256]]
    repeat = 2
    number = 1
    warmup = 20
    iters = 100
    n_envs = CHOL_DEFAULT_ENVS

    def setup(self, kernel, block_dim):
        if kernel == "tile_matmul_transpose_update" and not _has_warp_builtin("tile_matmul_transpose_update"):
            raise SkipNotImplemented

        self.device = _cuda_device()
        _, self.a, self.l = _allocate_cholesky_problem(self.n_envs, self.device)
        self.kernel = _CHOL_KERNELS[kernel]

    def track_factorize_us(self, kernel, block_dim):
        return _benchmark_cholesky_kernel(
            self.kernel,
            self.a,
            self.l,
            block_dim,
            self.warmup,
            self.iters,
            self.device,
        )

    track_factorize_us.unit = "us"


class FastKaminoCholeskyTileUpdateSpeedup:
    param_names = ["block_dim"]
    params = [[32, 64, 128, 256]]
    repeat = 2
    number = 1
    warmup = 20
    iters = 100
    n_envs = CHOL_DEFAULT_ENVS

    def setup(self, block_dim):
        if not _has_warp_builtin("tile_matmul_transpose_update"):
            raise SkipNotImplemented

        self.device = _cuda_device()
        _, self.a, self.l = _allocate_cholesky_problem(self.n_envs, self.device)

    def track_speedup(self, block_dim):
        baseline_us = _benchmark_cholesky_kernel(
            _cholesky_warp_tile,
            self.a,
            self.l,
            block_dim,
            self.warmup,
            self.iters,
            self.device,
        )
        update_us = _benchmark_cholesky_kernel(
            _cholesky_warp_tile_matmul_transpose_update,
            self.a,
            self.l,
            block_dim,
            self.warmup,
            self.iters,
            self.device,
        )
        return baseline_us / update_us

    track_speedup.unit = "x"


class FastKaminoBackSolveTileUpdate:
    param_names = ["kernel", "block_dim"]
    params = [["tile_matmul", "tile_matmul_left_transpose_update"], [128, 256]]
    repeat = 2
    number = 1
    warmup = 20
    iters = 200
    n_envs = SOLVE_DEFAULT_ENVS

    def setup(self, kernel, block_dim):
        if kernel == "tile_matmul_left_transpose_update" and not _has_warp_builtin("tile_matmul_left_transpose_update"):
            raise SkipNotImplemented

        self.device = _cuda_device()
        self.l, self.x, self.rhs, self.out = _allocate_back_solve_update_problem(self.n_envs, self.device)
        self.kernel = _BACK_SOLVE_UPDATE_KERNELS[kernel]

    def track_update_us(self, kernel, block_dim):
        return _benchmark_back_solve_update_kernel(
            self.kernel,
            self.l,
            self.x,
            self.rhs,
            self.out,
            block_dim,
            self.warmup,
            self.iters,
            self.device,
        )

    track_update_us.unit = "us"


class FastKaminoBackSolveTileUpdateSpeedup:
    param_names = ["block_dim"]
    params = [[128, 256]]
    repeat = 2
    number = 1
    warmup = 20
    iters = 200
    n_envs = SOLVE_DEFAULT_ENVS

    def setup(self, block_dim):
        if not _has_warp_builtin("tile_matmul_left_transpose_update"):
            raise SkipNotImplemented

        self.device = _cuda_device()
        self.l, self.x, self.rhs, self.out = _allocate_back_solve_update_problem(self.n_envs, self.device)

    def track_speedup(self, block_dim):
        baseline_us = _benchmark_back_solve_update_kernel(
            _back_solve_update_tile_matmul,
            self.l,
            self.x,
            self.rhs,
            self.out,
            block_dim,
            self.warmup,
            self.iters,
            self.device,
        )
        update_us = _benchmark_back_solve_update_kernel(
            _back_solve_update_left_transpose,
            self.l,
            self.x,
            self.rhs,
            self.out,
            block_dim,
            self.warmup,
            self.iters,
            self.device,
        )
        return baseline_us / update_us

    track_speedup.unit = "x"


class FastKaminoDrLegsDirect:
    param_names = ["linear_solver_type", "solve_block_dim"]
    params = [["LLTBRCM", "LLTB"], [128, 256]]
    repeat = 2
    number = 1
    num_frames = 20
    world_count = 64

    def setup(self, linear_solver_type, solve_block_dim):
        _cuda_device()

        args = newton.examples.default_args(ExampleDrLegs.create_parser())
        args.world_count = self.world_count
        args.use_kamino_contacts = True
        args.linear_solver_type = linear_solver_type
        args.linear_solver_kwargs = {"solve_block_dim": solve_block_dim}
        self.example = ExampleDrLegs(ViewerNull(num_frames=self.num_frames), args)
        wp.synchronize_device()

    def track_frame_ms(self, linear_solver_type, solve_block_dim):
        start = time.perf_counter()
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()
        return (time.perf_counter() - start) * 1000.0 / self.num_frames

    track_frame_ms.unit = "ms/frame"


class FastKaminoDrLegsDirectGraphConditionals:
    param_names = ["use_graph_conditionals"]
    params = [[True, False]]
    repeat = 2
    number = 1
    num_frames = 20
    world_count = 64

    def setup(self, use_graph_conditionals):
        _cuda_device()

        args = newton.examples.default_args(ExampleDrLegs.create_parser())
        args.world_count = self.world_count
        args.use_kamino_contacts = True
        args.linear_solver_type = "LLTBRCM"
        args.linear_solver_kwargs = {"solve_block_dim": 256}
        args.use_graph_conditionals = use_graph_conditionals
        self.example = ExampleDrLegs(ViewerNull(num_frames=self.num_frames), args)
        wp.synchronize_device()

    def track_frame_ms(self, use_graph_conditionals):
        start = time.perf_counter()
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()
        return (time.perf_counter() - start) * 1000.0 / self.num_frames

    track_frame_ms.unit = "ms/frame"


if __name__ == "__main__":
    from newton.utils import run_benchmark

    benchmark_list = {
        "FastKaminoCholeskyTileUpdate": FastKaminoCholeskyTileUpdate,
        "FastKaminoCholeskyTileUpdateSpeedup": FastKaminoCholeskyTileUpdateSpeedup,
        "FastKaminoBackSolveTileUpdate": FastKaminoBackSolveTileUpdate,
        "FastKaminoBackSolveTileUpdateSpeedup": FastKaminoBackSolveTileUpdateSpeedup,
        "FastKaminoDrLegsDirect": FastKaminoDrLegsDirect,
        "FastKaminoDrLegsDirectGraphConditionals": FastKaminoDrLegsDirectGraphConditionals,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    parser.add_argument("--number", default=1, type=int, help="Number of measurement samples per benchmark.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip tile-kernel correctness checks.")
    args = parser.parse_known_args()[0]

    benchmarks = benchmark_list.keys() if args.bench is None else args.bench

    if not args.skip_validation and any("Cholesky" in key for key in benchmarks):
        max_diff, max_recon = validate_cholesky_tile_update()
        print(f"Cholesky tile update validation: max |L_update - L_baseline| = {max_diff:.3e}")
        print(f"Cholesky tile update validation: max reconstruction error = {max_recon:.3e}")

    if not args.skip_validation and any("BackSolve" in key for key in benchmarks):
        max_diff = validate_back_solve_tile_update()
        print(f"Back-solve tile update validation: max |update - baseline| = {max_diff:.3e}")

    for key in benchmarks:
        run_benchmark(benchmark_list[key], number=args.number)
