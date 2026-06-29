# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse scalar-interpolation Galerkin corrections for articulation graphs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.coarse_aggregate import _solve_aggregate_kernel
from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem

_BLOCK_SIZE = 6
_BLOCK_DIM = 256


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.kernel
def _assemble_interpolation_kernel(
    fine_diag: wp.array3d[wp.float32],
    fine_off: wp.array3d[wp.float32],
    fine_rhs: wp.array2d[wp.float32],
    interpolation_ptr: wp.array[wp.int32],
    interpolation_coarse: wp.array[wp.int32],
    interpolation_weight: wp.array[wp.float32],
    diag_source: wp.array[wp.int32],
    diag_target: wp.array[wp.int32],
    diag_is_diagonal: wp.array[wp.int32],
    diag_weight: wp.array[wp.float32],
    off_source: wp.array[wp.int32],
    off_target: wp.array[wp.int32],
    off_is_diagonal: wp.array[wp.int32],
    off_transpose: wp.array[wp.int32],
    off_weight: wp.array[wp.float32],
    fine_blocks: wp.int32,
    diag_contributions: wp.int32,
    off_contributions: wp.int32,
    coarse_blocks: wp.int32,
    coarse_edges: wp.int32,
    rows: wp.int32,
    coarse_diag: wp.array3d[wp.float32],
    coarse_off: wp.array3d[wp.float32],
    coarse_rhs: wp.array2d[wp.float32],
    coarse_solution: wp.array2d[wp.float32],
):
    lane = wp.tid()
    coarse = lane
    while coarse < coarse_blocks:
        for r in range(_BLOCK_SIZE):
            coarse_rhs[coarse, r] = wp.float32(0.0)
            coarse_solution[coarse, r] = wp.float32(0.0)
            for c in range(_BLOCK_SIZE):
                coarse_diag[coarse, r, c] = wp.float32(0.0)
        coarse += wp.int32(_BLOCK_DIM)
    edge = lane
    while edge < coarse_edges:
        for r in range(_BLOCK_SIZE):
            for c in range(_BLOCK_SIZE):
                coarse_off[edge, r, c] = wp.float32(0.0)
        edge += wp.int32(_BLOCK_DIM)
    _sync_threads()

    fine = lane
    while fine < fine_blocks:
        for slot in range(interpolation_ptr[fine], interpolation_ptr[fine + wp.int32(1)]):
            coarse = interpolation_coarse[slot]
            weight = interpolation_weight[slot]
            for r in range(_BLOCK_SIZE):
                if wp.int32(r) < rows:
                    wp.atomic_add(coarse_rhs, coarse, r, weight * fine_rhs[fine, r])
        fine += wp.int32(_BLOCK_DIM)

    contribution = lane
    while contribution < diag_contributions:
        source = diag_source[contribution]
        target = diag_target[contribution]
        weight = diag_weight[contribution]
        is_diagonal = diag_is_diagonal[contribution]
        for r in range(_BLOCK_SIZE):
            if wp.int32(r) < rows:
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < rows:
                        value = weight * fine_diag[source, r, c]
                        if is_diagonal != wp.int32(0):
                            wp.atomic_add(coarse_diag, target, r, c, value)
                        else:
                            wp.atomic_add(coarse_off, target, r, c, value)
        contribution += wp.int32(_BLOCK_DIM)

    contribution = lane
    while contribution < off_contributions:
        source = off_source[contribution]
        target = off_target[contribution]
        weight = off_weight[contribution]
        is_diagonal = off_is_diagonal[contribution]
        transpose = off_transpose[contribution]
        for r in range(_BLOCK_SIZE):
            if wp.int32(r) < rows:
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < rows:
                        value = weight * fine_off[source, r, c]
                        if is_diagonal != wp.int32(0):
                            wp.atomic_add(coarse_diag, target, r, c, value)
                            wp.atomic_add(coarse_diag, target, c, r, value)
                        elif transpose != wp.int32(0):
                            wp.atomic_add(coarse_off, target, c, r, value)
                        else:
                            wp.atomic_add(coarse_off, target, r, c, value)
        contribution += wp.int32(_BLOCK_DIM)


@wp.kernel
def _prolongate_interpolation_kernel(
    coarse_solution: wp.array2d[wp.float32],
    interpolation_ptr: wp.array[wp.int32],
    interpolation_coarse: wp.array[wp.int32],
    interpolation_weight: wp.array[wp.float32],
    fine_blocks: wp.int32,
    rows: wp.int32,
    active_block_offsets: wp.array[wp.int32],
    pivot_order: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    fine_solution: wp.array[wp.float32],
):
    fine_row = wp.tid()
    fine = fine_row / wp.int32(_BLOCK_SIZE)
    row = fine_row - fine * wp.int32(_BLOCK_SIZE)
    if fine < fine_blocks and row < block_sizes[fine]:
        value = wp.float32(0.0)
        if row < rows:
            for slot in range(interpolation_ptr[fine], interpolation_ptr[fine + wp.int32(1)]):
                value += interpolation_weight[slot] * coarse_solution[interpolation_coarse[slot], row]
        active_block = pivot_order[fine]
        fine_solution[active_block_offsets[active_block] + row] = value


class CoarseInterpolationSolver:
    """Reusable sparse scalar-interpolation Galerkin correction."""

    def __init__(
        self,
        interpolation: Sequence[Sequence[tuple[int, float]]],
        fine_off_row: np.ndarray,
        fine_off_col: np.ndarray,
        rows: int,
        color_sweeps: int,
        device,
    ):
        self.fine_blocks = len(interpolation)
        self.rows = int(rows)
        self.color_sweeps = int(color_sweeps)
        off_row = np.asarray(fine_off_row, dtype=np.int32)
        off_col = np.asarray(fine_off_col, dtype=np.int32)

        interpolation_ptr = np.zeros(self.fine_blocks + 1, dtype=np.int32)
        interpolation_coarse: list[int] = []
        interpolation_weight: list[float] = []
        normalized: list[tuple[tuple[int, float], ...]] = []
        for fine, entries_value in enumerate(interpolation):
            entries = tuple((int(coarse), float(weight)) for coarse, weight in entries_value)
            if not entries:
                raise ValueError("every fine block requires at least one interpolation entry")
            if abs(sum(weight for _, weight in entries) - 1.0) > 1.0e-6:
                raise ValueError("interpolation weights must sum to one")
            normalized.append(entries)
            interpolation_coarse.extend(coarse for coarse, _ in entries)
            interpolation_weight.extend(weight for _, weight in entries)
            interpolation_ptr[fine + 1] = len(interpolation_coarse)
        self.coarse_blocks = max(interpolation_coarse) + 1
        if self.coarse_blocks > _BLOCK_DIM:
            raise ValueError("interpolation correction currently supports at most 256 coarse blocks")

        diag_raw: list[tuple[int, tuple[int, int], float]] = []
        off_raw: list[tuple[int, tuple[int, int], float, int]] = []
        coarse_pairs: set[tuple[int, int]] = set()
        for fine, entries in enumerate(normalized):
            for first, (coarse_a, weight_a) in enumerate(entries):
                for coarse_b, weight_b in entries[first:]:
                    pair = (max(coarse_a, coarse_b), min(coarse_a, coarse_b))
                    diag_raw.append((fine, pair, weight_a * weight_b))
                    if pair[0] != pair[1]:
                        coarse_pairs.add(pair)
        for edge, (row, col) in enumerate(zip(off_row, off_col, strict=True)):
            for coarse_row, weight_row in normalized[int(row)]:
                for coarse_col, weight_col in normalized[int(col)]:
                    pair = (max(coarse_row, coarse_col), min(coarse_row, coarse_col))
                    transpose = int(coarse_row < coarse_col)
                    off_raw.append((edge, pair, weight_row * weight_col, transpose))
                    if pair[0] != pair[1]:
                        coarse_pairs.add(pair)

        self.coarse_pairs = tuple(sorted(coarse_pairs))
        pair_to_edge = {pair: edge for edge, pair in enumerate(self.coarse_pairs)}
        diag_source = np.asarray([source for source, _, _ in diag_raw], dtype=np.int32)
        diag_target = np.asarray(
            [pair[0] if pair[0] == pair[1] else pair_to_edge[pair] for _, pair, _ in diag_raw], dtype=np.int32
        )
        diag_is_diagonal = np.asarray([pair[0] == pair[1] for _, pair, _ in diag_raw], dtype=np.int32)
        diag_weight = np.asarray([weight for _, _, weight in diag_raw], dtype=np.float32)
        off_source = np.asarray([source for source, _, _, _ in off_raw], dtype=np.int32)
        off_target = np.asarray(
            [pair[0] if pair[0] == pair[1] else pair_to_edge[pair] for _, pair, _, _ in off_raw], dtype=np.int32
        )
        off_is_diagonal = np.asarray([pair[0] == pair[1] for _, pair, _, _ in off_raw], dtype=np.int32)
        off_weight = np.asarray([weight for _, _, weight, _ in off_raw], dtype=np.float32)
        off_transpose = np.asarray([transpose for _, _, _, transpose in off_raw], dtype=np.int32)

        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.coarse_blocks)]
        edge_row = np.empty(len(self.coarse_pairs), dtype=np.int32)
        for edge, (row, col) in enumerate(self.coarse_pairs):
            edge_row[edge] = row
            adjacency[row].append((col, edge))
            adjacency[col].append((row, edge))
        adjacency_ptr = np.zeros(self.coarse_blocks + 1, dtype=np.int32)
        adjacency_neighbor: list[int] = []
        adjacency_edge: list[int] = []
        for node, entries in enumerate(adjacency):
            entries.sort()
            adjacency_ptr[node] = len(adjacency_neighbor)
            for neighbor, edge in entries:
                adjacency_neighbor.append(neighbor)
                adjacency_edge.append(edge)
        adjacency_ptr[self.coarse_blocks] = len(adjacency_neighbor)
        colors = np.full(self.coarse_blocks, -1, dtype=np.int32)
        for node in range(self.coarse_blocks):
            forbidden = {int(colors[neighbor]) for neighbor, _ in adjacency[node] if colors[neighbor] >= 0}
            color = 0
            while color in forbidden:
                color += 1
            colors[node] = color
        self.color_count = int(colors.max()) + 1

        def device_array(values, dtype):
            return wp.array(values, dtype=dtype, device=device)

        sentinel_i = np.zeros(1, dtype=np.int32)
        sentinel_f = np.zeros(1, dtype=np.float32)
        self.interpolation_ptr = device_array(interpolation_ptr, wp.int32)
        self.interpolation_coarse = device_array(np.asarray(interpolation_coarse, dtype=np.int32), wp.int32)
        self.interpolation_weight = device_array(np.asarray(interpolation_weight, dtype=np.float32), wp.float32)
        self.diag_source = device_array(diag_source if diag_source.size else sentinel_i, wp.int32)
        self.diag_target = device_array(diag_target if diag_target.size else sentinel_i, wp.int32)
        self.diag_is_diagonal = device_array(diag_is_diagonal if diag_is_diagonal.size else sentinel_i, wp.int32)
        self.diag_weight = device_array(diag_weight if diag_weight.size else sentinel_f, wp.float32)
        self.off_source = device_array(off_source if off_source.size else sentinel_i, wp.int32)
        self.off_target = device_array(off_target if off_target.size else sentinel_i, wp.int32)
        self.off_is_diagonal = device_array(off_is_diagonal if off_is_diagonal.size else sentinel_i, wp.int32)
        self.off_transpose = device_array(off_transpose if off_transpose.size else sentinel_i, wp.int32)
        self.off_weight = device_array(off_weight if off_weight.size else sentinel_f, wp.float32)
        self.edge_row = device_array(edge_row if edge_row.size else sentinel_i, wp.int32)
        self.adjacency_ptr = device_array(adjacency_ptr, wp.int32)
        self.adjacency_neighbor = device_array(np.asarray(adjacency_neighbor or [0], dtype=np.int32), wp.int32)
        self.adjacency_edge = device_array(np.asarray(adjacency_edge or [0], dtype=np.int32), wp.int32)
        self.colors = device_array(colors, wp.int32)
        self.diag_contributions = int(diag_source.size)
        self.off_contributions = int(off_source.size)

        shape3 = (self.coarse_blocks, _BLOCK_SIZE, _BLOCK_SIZE)
        edge_shape = (max(len(self.coarse_pairs), 1), _BLOCK_SIZE, _BLOCK_SIZE)
        self.diag = wp.zeros(shape3, dtype=wp.float32, device=device)
        self.off = wp.zeros(edge_shape, dtype=wp.float32, device=device)
        self.factor = wp.zeros(shape3, dtype=wp.float32, device=device)
        self.rhs = wp.zeros((self.coarse_blocks, _BLOCK_SIZE), dtype=wp.float32, device=device)
        self.work = wp.zeros_like(self.rhs)
        self.solution = wp.zeros_like(self.rhs)

    def solve(self, system: ArticulationDeviceSystem, *, device=None) -> None:
        system.gather_block_rhs(device=device)
        wp.launch(
            _assemble_interpolation_kernel,
            dim=_BLOCK_DIM,
            block_dim=_BLOCK_DIM,
            inputs=[
                system.block_diag,
                system.block_off,
                system.block_rhs,
                self.interpolation_ptr,
                self.interpolation_coarse,
                self.interpolation_weight,
                self.diag_source,
                self.diag_target,
                self.diag_is_diagonal,
                self.diag_weight,
                self.off_source,
                self.off_target,
                self.off_is_diagonal,
                self.off_transpose,
                self.off_weight,
                wp.int32(self.fine_blocks),
                wp.int32(self.diag_contributions),
                wp.int32(self.off_contributions),
                wp.int32(self.coarse_blocks),
                wp.int32(len(self.coarse_pairs)),
                wp.int32(self.rows),
            ],
            outputs=[self.diag, self.off, self.rhs, self.solution],
            device=device,
        )
        wp.launch(
            _solve_aggregate_kernel,
            dim=_BLOCK_DIM,
            block_dim=_BLOCK_DIM,
            inputs=[
                self.diag,
                self.off,
                self.rhs,
                self.edge_row,
                self.adjacency_ptr,
                self.adjacency_neighbor,
                self.adjacency_edge,
                self.colors,
                wp.int32(self.coarse_blocks),
                wp.int32(self.rows),
                wp.int32(self.color_count),
                wp.int32(self.color_sweeps),
            ],
            outputs=[self.factor, self.work, self.solution],
            device=device,
        )
        wp.launch(
            _prolongate_interpolation_kernel,
            dim=self.fine_blocks * _BLOCK_SIZE,
            inputs=[
                self.solution,
                self.interpolation_ptr,
                self.interpolation_coarse,
                self.interpolation_weight,
                wp.int32(self.fine_blocks),
                wp.int32(self.rows),
                system.active_block_offsets,
                system.block_pivot_order,
                system.block_sizes,
            ],
            outputs=[system.solution],
            device=device,
        )
