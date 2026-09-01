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


def make_contact_contributions(side, reserved_layers, active_layers, world_count):
    """Create fixed-capacity structural and changing-contact BSR contributions."""
    if not 0 <= active_layers <= reserved_layers:
        raise ValueError("active contact layers must fit in reserved row capacity")
    rows, cols, values = make_stiff_grid(side)
    node_count = side**3
    block_rows = np.repeat(np.arange(node_count, dtype=np.int32), np.diff(rows))
    rng = np.random.default_rng(91)
    contact_rows = []
    contact_cols = []
    contact_values = []
    stiffness = np.eye(3, dtype=np.float32) * 30.0
    for _ in range(active_layers):
        permutation = rng.permutation(node_count).astype(np.int32)
        first = permutation[0::2]
        second = permutation[1::2]
        for row, col in zip(first, second, strict=True):
            contact_rows.extend((row, row, col, col))
            contact_cols.extend((row, col, col, row))
            contact_values.extend((stiffness, -stiffness, stiffness, -stiffness))

    local_rows = np.concatenate((block_rows, np.asarray(contact_rows, dtype=np.int32)))
    local_cols = np.concatenate((cols, np.asarray(contact_cols, dtype=np.int32)))
    local_values = np.concatenate((values, np.asarray(contact_values, dtype=np.float32).reshape((-1, 3, 3))))
    contribution_rows = []
    contribution_cols = []
    contribution_values = []
    for world in range(world_count):
        offset = world * node_count
        contribution_rows.append(local_rows + offset)
        contribution_cols.append(local_cols + offset)
        contribution_values.append(local_values)
    capacities = np.diff(rows) + 2 * reserved_layers
    return (
        rows,
        cols,
        values,
        capacities,
        np.concatenate(contribution_rows),
        np.concatenate(contribution_cols),
        np.concatenate(contribution_values),
    )


class MASPCGBenchmark:
    """Measure solve, numeric refit, iteration count, and linear storage scaling."""

    params = ([(8, 1), (8, 16), (16, 1), (16, 8), (32, 1)],)
    param_names = ("side_worlds",)
    timeout = 600

    def setup(self, side_worlds):
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
        )
        self.solve_graph = self.solver.capture(rhs, self.x, refit=False)
        self.frame_graph = self.solver.capture(rhs, self.x, refit=True)
        wp.capture_launch(self.solve_graph)
        wp.synchronize_device()

    def time_solve_graph(self, _side_worlds):
        wp.capture_launch(self.solve_graph)
        wp.synchronize_device()

    def time_refit_and_solve_graph(self, _side_worlds):
        wp.capture_launch(self.frame_graph)
        wp.synchronize_device()

    def time_spmv(self, _side_worlds):
        self.solver.matrix.gemv(self.rhs, self.scratch, self.solver.world_active)
        wp.synchronize_device()

    def time_preconditioner_apply(self, _side_worlds):
        self.solver.preconditioner.apply(self.rhs, self.scratch, self.solver.world_active)
        wp.synchronize_device()

    def track_max_iterations(self, _side_worlds):
        wp.capture_launch(self.solve_graph)
        return int(self.solver.iterations.numpy().max())

    def track_storage_bytes_per_node(self, _side_worlds):
        return self.solver.storage_bytes / self.solver.matrix.total_row_count


class MASPCGConditioningBenchmark:
    """Measure convergence as the shifted Laplacian approaches singularity."""

    params = ([2.0e-2, 2.0e-3, 2.0e-4, 1.0e-4], [1, 2, 4])
    param_names = ("diagonal_shift", "refinement_passes")
    timeout = 600

    def setup(self, diagonal_shift, refinement_passes):
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
            refinement_passes=refinement_passes,
        )
        self.graph = self.solver.capture(self.rhs, self.x, refit=False)
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def time_solve_graph(self, _diagonal_shift, _refinement_passes):
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def track_iterations(self, _diagonal_shift, _refinement_passes):
        wp.capture_launch(self.graph)
        return int(self.solver.iterations.numpy()[0])

    def track_true_relative_residual(self, _diagonal_shift, _refinement_passes):
        wp.capture_launch(self.graph)
        self.solver.matrix.gemv(self.x, self.scratch, self.solver.world_active)
        wp.synchronize_device()
        residual = self.scratch.numpy() - self.rhs.numpy()
        return float(np.linalg.norm(residual) / np.linalg.norm(self.rhs.numpy()))


class MASPCGContactBenchmark:
    """Measure captured sparse assembly, MAS refit, and solve with contact edges."""

    params = ([(8, 4, 0, 1), (8, 4, 1, 1), (8, 4, 4, 1), (8, 4, 4, 16), (16, 2, 2, 1)],)
    param_names = ("side_reserved_active_worlds",)
    timeout = 600

    def setup(self, side_reserved_active_worlds):
        if not wp.is_cuda_available():
            raise NotImplementedError("CUDA is required")
        side, reserved_layers, active_layers, world_count = side_reserved_active_worlds
        data = make_contact_contributions(side, reserved_layers, active_layers, world_count)
        rows, cols, values, capacities, contribution_rows, contribution_cols, contribution_values = data
        matrix = BatchedBSRMatrix.from_host(
            [rows] * world_count,
            [cols] * world_count,
            [values] * world_count,
            row_capacities=[capacities] * world_count,
            device="cuda:0",
        )
        self.contribution_rows = wp.array(contribution_rows, device=matrix.device)
        self.contribution_cols = wp.array(contribution_cols, device=matrix.device)
        self.contribution_values = wp.array(contribution_values, dtype=wp.mat33f, device=matrix.device)
        self.contribution_count = wp.array([contribution_rows.size], dtype=wp.int32, device=matrix.device)
        rng = np.random.default_rng(29)
        self.rhs = wp.array(
            rng.normal(size=matrix.total_scalar_count).astype(np.float32),
            device=matrix.device,
        )
        self.x = wp.zeros_like(self.rhs)
        self.solver = BatchedMASPCG(
            matrix,
            rtol=1.0e-4,
            atol=1.0e-6,
            max_iterations=300,
            use_cuda_graph=True,
            loop_granularity=2,
        )
        with wp.ScopedCapture(matrix.device) as capture:
            self.x.zero_()
            matrix.begin_assembly()
            matrix.insert_blocks(
                self.contribution_rows,
                self.contribution_cols,
                self.contribution_values,
                self.contribution_count,
            )
            self.solver.solve(self.rhs, self.x, refit=True)
        self.frame_graph = capture.graph
        wp.capture_launch(self.frame_graph)
        wp.synchronize_device()

    def time_assemble_refit_solve_graph(self, _side_reserved_active_worlds):
        wp.capture_launch(self.frame_graph)
        wp.synchronize_device()

    def track_iterations(self, _side_reserved_active_worlds):
        wp.capture_launch(self.frame_graph)
        return int(self.solver.iterations.numpy().max())

    def track_overflow(self, _side_reserved_active_worlds):
        wp.capture_launch(self.frame_graph)
        return int(self.solver.matrix.overflow.numpy()[0])


def run_conditioning():
    """Print the residual-replacement sweep without the size benchmarks."""
    print("conditioning passes shift condition_upper_bound solve_ms iterations true_relative_residual")
    for passes in MASPCGConditioningBenchmark.params[1]:
        for shift in MASPCGConditioningBenchmark.params[0]:
            benchmark = MASPCGConditioningBenchmark()
            benchmark.setup(shift, passes)
            solve_times = []
            for _ in range(10):
                start = time.perf_counter()
                benchmark.time_solve_graph(shift, passes)
                solve_times.append(time.perf_counter() - start)
            print(
                "conditioning",
                passes,
                f"{shift:.1e}",
                f"{(1200.0 + shift) / shift:.1e}",
                f"{1.0e3 * np.median(solve_times):.3f}",
                benchmark.track_iterations(shift, passes),
                f"{benchmark.track_true_relative_residual(shift, passes):.2e}",
            )


def run_sizes():
    """Print the varied-size sweep."""
    wp.config.log_level = wp.LOG_WARNING
    print("side worlds nodes/world total_nodes mas_ms solve_ms frame_ms iterations bytes/node")
    for case in MASPCGBenchmark.params[0]:
        benchmark = MASPCGBenchmark()
        benchmark.setup(case)
        solve_times = []
        frame_times = []
        mas_times = []
        for _ in range(20):
            start = time.perf_counter()
            benchmark.time_preconditioner_apply(case)
            mas_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            benchmark.time_solve_graph(case)
            solve_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            benchmark.time_refit_and_solve_graph(case)
            frame_times.append(time.perf_counter() - start)
        side, worlds = case
        print(
            side,
            worlds,
            side**3,
            worlds * side**3,
            f"{1.0e3 * np.median(mas_times):.3f}",
            f"{1.0e3 * np.median(solve_times):.3f}",
            f"{1.0e3 * np.median(frame_times):.3f}",
            benchmark.track_max_iterations(case),
            f"{benchmark.track_storage_bytes_per_node(case):.1f}",
        )


def run_contacts():
    """Print the dynamic contact-assembly sweep."""
    print("contact side reserved active worlds nodes/world total_nodes frame_ms iterations overflow")
    for case in MASPCGContactBenchmark.params[0]:
        benchmark = MASPCGContactBenchmark()
        benchmark.setup(case)
        times = []
        for _ in range(20):
            start = time.perf_counter()
            benchmark.time_assemble_refit_solve_graph(case)
            times.append(time.perf_counter() - start)
        side, reserved, active, worlds = case
        print(
            "contact",
            side,
            reserved,
            active,
            worlds,
            side**3,
            worlds * side**3,
            f"{1.0e3 * np.median(times):.3f}",
            benchmark.track_iterations(case),
            benchmark.track_overflow(case),
        )


def main():
    """Run the representative matrix locally and print median graph timings."""
    run_sizes()
    run_contacts()
    run_conditioning()


if __name__ == "__main__":
    main()
