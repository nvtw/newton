# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental one-hot aggregate Galerkin correction for sparse joint graphs."""

from __future__ import annotations

import numpy as np
import warp as wp

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
def _assemble_aggregate_kernel(
    fine_diag: wp.array3d[wp.float32],
    fine_off: wp.array3d[wp.float32],
    fine_rhs: wp.array2d[wp.float32],
    fine_to_coarse: wp.array[wp.int32],
    off_target: wp.array[wp.int32],
    off_is_diag: wp.array[wp.int32],
    off_transpose: wp.array[wp.int32],
    fine_blocks: wp.int32,
    fine_edges: wp.int32,
    coarse_blocks: wp.int32,
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
    while edge < fine_edges:
        if off_is_diag[edge] == wp.int32(0):
            target = off_target[edge]
            for r in range(_BLOCK_SIZE):
                for c in range(_BLOCK_SIZE):
                    coarse_off[target, r, c] = wp.float32(0.0)
        edge += wp.int32(_BLOCK_DIM)
    _sync_threads()

    fine = lane
    while fine < fine_blocks:
        target = fine_to_coarse[fine]
        for r in range(_BLOCK_SIZE):
            if wp.int32(r) < rows:
                wp.atomic_add(coarse_rhs, target, r, fine_rhs[fine, r])
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < rows:
                        wp.atomic_add(coarse_diag, target, r, c, fine_diag[fine, r, c])
        fine += wp.int32(_BLOCK_DIM)

    edge = lane
    while edge < fine_edges:
        target = off_target[edge]
        is_diag = off_is_diag[edge]
        transpose = off_transpose[edge]
        for r in range(_BLOCK_SIZE):
            if wp.int32(r) < rows:
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < rows:
                        value = fine_off[edge, r, c]
                        if is_diag != wp.int32(0):
                            wp.atomic_add(coarse_diag, target, r, c, value)
                            wp.atomic_add(coarse_diag, target, c, r, value)
                        elif transpose != wp.int32(0):
                            wp.atomic_add(coarse_off, target, c, r, value)
                        else:
                            wp.atomic_add(coarse_off, target, r, c, value)
        edge += wp.int32(_BLOCK_DIM)


@wp.kernel
def _solve_aggregate_kernel(
    coarse_diag: wp.array3d[wp.float32],
    coarse_off: wp.array3d[wp.float32],
    coarse_rhs: wp.array2d[wp.float32],
    edge_row: wp.array[wp.int32],
    adjacency_ptr: wp.array[wp.int32],
    adjacency_neighbor: wp.array[wp.int32],
    adjacency_edge: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    coarse_blocks: wp.int32,
    rows: wp.int32,
    color_count: wp.int32,
    color_sweeps: wp.int32,
    factor_diag: wp.array3d[wp.float32],
    work: wp.array2d[wp.float32],
    solution: wp.array2d[wp.float32],
):
    node = wp.tid()
    if node < coarse_blocks:
        for i in range(_BLOCK_SIZE):
            for j in range(_BLOCK_SIZE):
                value = wp.float32(0.0)
                if wp.int32(i) < rows and wp.int32(j) < rows:
                    value = coarse_diag[node, i, j]
                elif i == j:
                    value = wp.float32(1.0)
                factor_diag[node, i, j] = value
        for k in range(_BLOCK_SIZE):
            if wp.int32(k) < rows:
                diag = wp.sqrt(wp.max(factor_diag[node, k, k], wp.float32(1.0e-20)))
                factor_diag[node, k, k] = diag
                for i in range(_BLOCK_SIZE):
                    if wp.int32(i) > wp.int32(k) and wp.int32(i) < rows:
                        factor_diag[node, i, k] /= diag
                for j in range(_BLOCK_SIZE):
                    if wp.int32(j) > wp.int32(k) and wp.int32(j) < rows:
                        l_jk = factor_diag[node, j, k]
                        for i in range(_BLOCK_SIZE):
                            if wp.int32(i) >= wp.int32(j) and wp.int32(i) < rows:
                                factor_diag[node, i, j] -= factor_diag[node, i, k] * l_jk
    _sync_threads()

    sweep = wp.int32(0)
    while sweep < color_sweeps:
        color = sweep % color_count
        if node < coarse_blocks and colors[node] == color:
            for r in range(_BLOCK_SIZE):
                if wp.int32(r) < rows:
                    value = coarse_rhs[node, r]
                    for adjacency in range(adjacency_ptr[node], adjacency_ptr[node + wp.int32(1)]):
                        neighbor = adjacency_neighbor[adjacency]
                        edge = adjacency_edge[adjacency]
                        for c in range(_BLOCK_SIZE):
                            if wp.int32(c) < rows:
                                if edge_row[edge] == node:
                                    value -= coarse_off[edge, r, c] * solution[neighbor, c]
                                else:
                                    value -= coarse_off[edge, c, r] * solution[neighbor, c]
                    for c in range(_BLOCK_SIZE):
                        if wp.int32(c) < wp.int32(r):
                            value -= factor_diag[node, r, c] * work[node, c]
                    work[node, r] = value / factor_diag[node, r, r]
            for reverse in range(_BLOCK_SIZE):
                r = _BLOCK_SIZE - 1 - reverse
                if wp.int32(r) < rows:
                    value = work[node, r]
                    for c in range(_BLOCK_SIZE):
                        if wp.int32(c) > wp.int32(r) and wp.int32(c) < rows:
                            value -= factor_diag[node, c, r] * solution[node, c]
                    solution[node, r] = value / factor_diag[node, r, r]
        _sync_threads()
        sweep += wp.int32(1)


@wp.kernel
def _prolongate_aggregate_kernel(
    coarse_solution: wp.array2d[wp.float32],
    fine_to_coarse: wp.array[wp.int32],
    fine_blocks: wp.int32,
    rows: wp.int32,
    fine_solution: wp.array[wp.float32],
):
    fine_row = wp.tid()
    fine = fine_row / rows
    row = fine_row - fine * rows
    if fine < fine_blocks:
        fine_solution[fine_row] = coarse_solution[fine_to_coarse[fine], row]


def parent_aggregate_mapping(
    body1: np.ndarray,
    body2: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Return factor-2 one-hot parent aggregates for a rooted body tree."""
    joint_for_child = {int(child): joint for joint, child in enumerate(body2)}
    children: list[list[int]] = [[] for _ in range(body1.size)]
    for joint, parent_body in enumerate(body1):
        parent_joint = joint_for_child.get(int(parent_body))
        if parent_joint is not None:
            children[parent_joint].append(joint)
    coarse_mask = depth % 2 == 1
    coarse_mask |= np.asarray([not child_joints for child_joints in children])
    coarse_joints = np.nonzero(coarse_mask)[0]
    coarse_index = {int(joint): index for index, joint in enumerate(coarse_joints)}
    mapping = np.empty(body1.size, dtype=np.int32)
    for joint in range(body1.size):
        representative = joint
        if not coarse_mask[joint]:
            representative = joint_for_child[int(body1[joint])]
        mapping[joint] = coarse_index[representative]
    return mapping


class CoarseAggregateSolver:
    """Reusable buffers and topology for one sparse aggregate correction."""

    def __init__(
        self,
        fine_to_coarse: np.ndarray,
        fine_off_row: np.ndarray,
        fine_off_col: np.ndarray,
        rows: int,
        color_sweeps: int,
        device,
    ):
        mapping = np.asarray(fine_to_coarse, dtype=np.int32)
        off_row = np.asarray(fine_off_row, dtype=np.int32)
        off_col = np.asarray(fine_off_col, dtype=np.int32)
        self.fine_blocks = int(mapping.size)
        self.fine_edges = int(off_row.size)
        self.coarse_blocks = int(mapping.max()) + 1
        self.rows = int(rows)
        self.color_sweeps = int(color_sweeps)

        coarse_pairs = sorted(
            {
                (max(int(mapping[row]), int(mapping[col])), min(int(mapping[row]), int(mapping[col])))
                for row, col in zip(off_row, off_col, strict=True)
                if mapping[row] != mapping[col]
            }
        )
        pair_to_edge = {pair: edge for edge, pair in enumerate(coarse_pairs)}
        self.coarse_pairs = tuple(coarse_pairs)
        off_target = np.empty(self.fine_edges, dtype=np.int32)
        off_is_diag = np.empty(self.fine_edges, dtype=np.int32)
        off_transpose = np.zeros(self.fine_edges, dtype=np.int32)
        for edge, (row, col) in enumerate(zip(off_row, off_col, strict=True)):
            coarse_row = int(mapping[row])
            coarse_col = int(mapping[col])
            if coarse_row == coarse_col:
                off_target[edge] = coarse_row
                off_is_diag[edge] = 1
            else:
                pair = (max(coarse_row, coarse_col), min(coarse_row, coarse_col))
                off_target[edge] = pair_to_edge[pair]
                off_is_diag[edge] = 0
                off_transpose[edge] = int(coarse_row < coarse_col)

        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.coarse_blocks)]
        edge_row = np.empty(len(coarse_pairs), dtype=np.int32)
        for edge, (row, col) in enumerate(coarse_pairs):
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

        self.fine_to_coarse = device_array(mapping, wp.int32)
        self.off_target = device_array(off_target, wp.int32)
        self.off_is_diag = device_array(off_is_diag, wp.int32)
        self.off_transpose = device_array(off_transpose, wp.int32)
        self.edge_row = device_array(edge_row if edge_row.size else np.zeros(1, np.int32), wp.int32)
        self.adjacency_ptr = device_array(adjacency_ptr, wp.int32)
        self.adjacency_neighbor = device_array(np.asarray(adjacency_neighbor or [0], dtype=np.int32), wp.int32)
        self.adjacency_edge = device_array(np.asarray(adjacency_edge or [0], dtype=np.int32), wp.int32)
        self.colors = device_array(colors, wp.int32)

        shape3 = (self.coarse_blocks, _BLOCK_SIZE, _BLOCK_SIZE)
        edge_shape = (max(len(coarse_pairs), 1), _BLOCK_SIZE, _BLOCK_SIZE)
        self.diag = wp.zeros(shape3, dtype=wp.float32, device=device)
        self.off = wp.zeros(edge_shape, dtype=wp.float32, device=device)
        self.factor = wp.zeros(shape3, dtype=wp.float32, device=device)
        self.rhs = wp.zeros((self.coarse_blocks, _BLOCK_SIZE), dtype=wp.float32, device=device)
        self.work = wp.zeros_like(self.rhs)
        self.solution = wp.zeros_like(self.rhs)

    def solve(self, system: ArticulationDeviceSystem, *, device=None) -> None:
        system.gather_block_rhs(device=device)
        wp.launch(
            _assemble_aggregate_kernel,
            dim=_BLOCK_DIM,
            block_dim=_BLOCK_DIM,
            inputs=[
                system.block_diag,
                system.block_off,
                system.block_rhs,
                self.fine_to_coarse,
                self.off_target,
                self.off_is_diag,
                self.off_transpose,
                wp.int32(self.fine_blocks),
                wp.int32(self.fine_edges),
                wp.int32(self.coarse_blocks),
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
            _prolongate_aggregate_kernel,
            dim=self.fine_blocks * self.rows,
            inputs=[self.solution, self.fine_to_coarse, wp.int32(self.fine_blocks), wp.int32(self.rows)],
            outputs=[system.solution],
            device=device,
        )
