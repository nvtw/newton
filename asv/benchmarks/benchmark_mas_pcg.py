# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evidence benchmarks for the experimental batched sparse MAS-PCG solver."""

import time

import numpy as np
import warp as wp

from newton._src.solvers.mas_pcg import BatchedBSRMatrix, BatchedMASPCG


def _morton_code(x, y, z):
    code = np.zeros_like(x, dtype=np.uint64)
    for bit in range(10):
        code |= ((x >> bit) & 1).astype(np.uint64) << (3 * bit)
        code |= ((y >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
        code |= ((z >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
    return code


def make_stiff_grid(side, diagonal_shift=0.02):
    """Create a Morton-ordered anisotropic 3D block Laplacian."""
    node_count = side**3
    linear = np.arange(node_count, dtype=np.int32)
    x = linear % side
    y = (linear // side) % side
    z = linear // (side * side)
    permutation = np.argsort(_morton_code(x, y, z), kind="stable")
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(node_count, dtype=np.int32)

    stiffness = np.diag(np.asarray([1.0, 10.0, 100.0], dtype=np.float32))
    shift = np.eye(3, dtype=np.float32) * np.float32(diagonal_shift)
    rows = [0]
    cols = []
    values = []
    directions = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    for ordered_row in range(node_count):
        original = int(permutation[ordered_row])
        px = original % side
        py = (original // side) % side
        pz = original // (side * side)
        neighbors = []
        for dx, dy, dz in directions:
            nx, ny, nz = px + dx, py + dy, pz + dz
            if 0 <= nx < side and 0 <= ny < side and 0 <= nz < side:
                neighbor = nx + side * (ny + side * nz)
                neighbors.append(int(inverse[neighbor]))
        for col in sorted(neighbors):
            cols.append(col)
            values.append(-stiffness)
        cols.append(ordered_row)
        values.append(len(neighbors) * stiffness + shift)
        rows.append(len(cols))
    return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32), np.asarray(values, dtype=np.float32)


class MASPCGBenchmark:
    """Measure solve, numeric refit, iteration count, and linear storage scaling."""

    params = (
        [(8, 1), (8, 16), (16, 1), (16, 8), (32, 1)],
        ["fp32", "fp32_async", "fp32_tile", "bf16_tile", "bf16_vector"],
    )
    param_names = ("side_worlds", "mas_apply_mode")
    timeout = 600

    def setup(self, side_worlds, mas_apply_mode):
        if not wp.is_cuda_available():
            raise NotImplementedError("CUDA is required")
        side, world_count = side_worlds
        rows, cols, values = make_stiff_grid(side)
        matrix = BatchedBSRMatrix.from_host(
            [rows] * world_count,
            [cols] * world_count,
            [values] * world_count,
            device="cuda:0",
        )
        rng = np.random.default_rng(17)
        rhs = wp.array(
            rng.normal(size=matrix.total_scalar_count).astype(np.float32),
            device=matrix.device,
        )
        self.x = wp.zeros_like(rhs)
        self.rhs = rhs
        self.scratch = wp.zeros_like(rhs)
        self.solver = BatchedMASPCG(
            matrix,
            rtol=1.0e-4,
            atol=1.0e-6,
            max_iterations=200,
            use_cuda_graph=True,
            loop_granularity=2,
            mas_apply_mode=mas_apply_mode,
        )
        self.solve_graph = self.solver.capture(rhs, self.x, refit=False)
        self.frame_graph = self.solver.capture(rhs, self.x, refit=True)
        wp.capture_launch(self.solve_graph)
        wp.synchronize_device()

    def time_solve_graph(self, _side_worlds, _mas_apply_mode):
        wp.capture_launch(self.solve_graph)
        wp.synchronize_device()

    def time_refit_and_solve_graph(self, _side_worlds, _mas_apply_mode):
        wp.capture_launch(self.frame_graph)
        wp.synchronize_device()

    def time_spmv(self, _side_worlds, _mas_apply_mode):
        self.solver.matrix.gemv(self.rhs, self.scratch, self.solver.world_active)
        wp.synchronize_device()

    def time_preconditioner_apply(self, _side_worlds, _mas_apply_mode):
        self.solver.preconditioner.apply(self.rhs, self.scratch, self.solver.world_active)
        wp.synchronize_device()

    def time_dot(self, _side_worlds, _mas_apply_mode):
        self.solver._solver.compute_dot(
            self.rhs,
            self.rhs,
            self.solver.matrix.active_dims,
            self.solver.world_active,
        )
        wp.synchronize_device()

    def track_max_iterations(self, _side_worlds, _mas_apply_mode):
        wp.capture_launch(self.solve_graph)
        return int(self.solver.iterations.numpy().max())

    def track_storage_bytes_per_node(self, _side_worlds, _mas_apply_mode):
        return self.solver.storage_bytes / self.solver.matrix.total_row_count


class MASPCGConditioningBenchmark:
    """Measure convergence as the shifted Laplacian approaches singularity."""

    params = (
        [2.0e-2, 2.0e-3, 2.0e-4, 1.0e-4],
        ["fp32_async", "bf16_vector"],
        [1, 2, 4],
    )
    param_names = ("diagonal_shift", "mas_apply_mode", "refinement_passes")
    timeout = 600

    def setup(self, diagonal_shift, mas_apply_mode, refinement_passes):
        rows, cols, values = make_stiff_grid(8, diagonal_shift)
        matrix = BatchedBSRMatrix.from_host([rows], [cols], [values], device="cuda:0")
        rng = np.random.default_rng(31)
        self.rhs = wp.array(rng.normal(size=matrix.total_scalar_count).astype(np.float32), device=matrix.device)
        self.x = wp.zeros_like(self.rhs)
        self.scratch = wp.zeros_like(self.rhs)
        self.solver = BatchedMASPCG(
            matrix,
            rtol=1.0e-4,
            atol=1.0e-6,
            max_iterations=500,
            use_cuda_graph=True,
            loop_granularity=2,
            mas_apply_mode=mas_apply_mode,
            refinement_passes=refinement_passes,
        )
        self.graph = self.solver.capture(self.rhs, self.x, refit=False)
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def time_solve_graph(self, _diagonal_shift, _mas_apply_mode, _refinement_passes):
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def track_iterations(self, _diagonal_shift, _mas_apply_mode, _refinement_passes):
        wp.capture_launch(self.graph)
        return int(self.solver.iterations.numpy()[0])

    def track_true_relative_residual(self, _diagonal_shift, _mas_apply_mode, _refinement_passes):
        wp.capture_launch(self.graph)
        self.solver.matrix.gemv(self.x, self.scratch, self.solver.world_active)
        wp.synchronize_device()
        residual = self.scratch.numpy() - self.rhs.numpy()
        return float(np.linalg.norm(residual) / np.linalg.norm(self.rhs.numpy()))


def run_conditioning():
    """Print the residual-replacement sweep without the size benchmarks."""
    print("conditioning mode passes shift condition_upper_bound solve_ms iterations true_relative_residual")
    for mode in MASPCGConditioningBenchmark.params[1]:
        for passes in MASPCGConditioningBenchmark.params[2]:
            for shift in MASPCGConditioningBenchmark.params[0]:
                benchmark = MASPCGConditioningBenchmark()
                benchmark.setup(shift, mode, passes)
                solve_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    benchmark.time_solve_graph(shift, mode, passes)
                    solve_times.append(time.perf_counter() - start)
                print(
                    "conditioning",
                    mode,
                    passes,
                    f"{shift:.1e}",
                    f"{(1200.0 + shift) / shift:.1e}",
                    f"{1.0e3 * np.median(solve_times):.3f}",
                    benchmark.track_iterations(shift, mode, passes),
                    f"{benchmark.track_true_relative_residual(shift, mode, passes):.2e}",
                )


def run_sizes(modes=None):
    """Print the varied-size sweep, optionally for selected apply modes."""
    wp.config.log_level = wp.LOG_WARNING
    print("precision side worlds nodes/world total_nodes mas_ms solve_ms frame_ms iterations bytes/node")
    modes = MASPCGBenchmark.params[1] if modes is None else modes
    for precision in modes:
        for case in MASPCGBenchmark.params[0]:
            benchmark = MASPCGBenchmark()
            benchmark.setup(case, precision)
            solve_times = []
            frame_times = []
            mas_times = []
            for _ in range(20):
                start = time.perf_counter()
                benchmark.time_preconditioner_apply(case, precision)
                mas_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                benchmark.time_solve_graph(case, precision)
                solve_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                benchmark.time_refit_and_solve_graph(case, precision)
                frame_times.append(time.perf_counter() - start)
            side, worlds = case
            print(
                precision,
                side,
                worlds,
                side**3,
                worlds * side**3,
                f"{1.0e3 * np.median(mas_times):.3f}",
                f"{1.0e3 * np.median(solve_times):.3f}",
                f"{1.0e3 * np.median(frame_times):.3f}",
                benchmark.track_max_iterations(case, precision),
                f"{benchmark.track_storage_bytes_per_node(case, precision):.1f}",
            )


def main():
    """Run the representative matrix locally and print median graph timings."""
    run_sizes()
    run_conditioning()


if __name__ == "__main__":
    main()
