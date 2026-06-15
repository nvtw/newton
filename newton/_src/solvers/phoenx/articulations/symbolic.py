# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Symbolic block factorization for PhoenX articulation DVI systems."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

_BLOCK_SIZE = 6


@dataclass
class ConstraintGraph:
    """Constraint graph whose nodes are equality blocks."""

    num_nodes: int
    adjacency: dict[int, set[int]] = field(default_factory=dict)

    def add_edge(self, i: int, j: int) -> None:
        """Add an undirected graph edge."""
        if i == j:
            return
        self.adjacency.setdefault(int(i), set()).add(int(j))
        self.adjacency.setdefault(int(j), set()).add(int(i))

    def neighbors(self, node: int) -> set[int]:
        """Return neighbors of ``node``."""
        return self.adjacency.get(int(node), set())

    def copy(self) -> ConstraintGraph:
        """Deep-copy adjacency sets."""
        graph = ConstraintGraph(self.num_nodes)
        for node, neighbors in self.adjacency.items():
            graph.adjacency[node] = neighbors.copy()
        return graph


@dataclass(frozen=True)
class BlockSparseSymbolic:
    """Symbolic data for a block-sparse direct articulation solve.

    Pivot indices are compact active-block indices. ``pivot_order`` maps a
    pivot index to the original joint index, which lets runtime kernels gather
    joint rows without carrying inactive blocks.
    """

    num_blocks: int
    block_size: int
    block_sizes: np.ndarray
    block_offsets: np.ndarray
    total_rows: int
    pivot_order: np.ndarray
    inv_pivot_order: np.ndarray
    l_col_ptr: np.ndarray
    l_row_idx: np.ndarray
    l_row_ptr: np.ndarray
    l_col_idx: np.ndarray
    l_csr_to_csc: np.ndarray
    n_off_col_ptr: np.ndarray
    n_off_row_idx: np.ndarray
    n_off_col_idx: np.ndarray
    n_off_to_l: np.ndarray
    parent: np.ndarray
    pred_diag_ptr: np.ndarray
    pred_diag_slot: np.ndarray
    pred_off_ptr: np.ndarray
    pred_off_slot_ik: np.ndarray
    pred_off_slot_jk: np.ndarray
    level_ptr: np.ndarray
    level_pivots: np.ndarray
    fill_edges: frozenset[tuple[int, int]]

    @property
    def nnz_l(self) -> int:
        """Number of strict lower-triangular L blocks."""
        return int(self.l_col_ptr[-1])

    @property
    def nnz_n(self) -> int:
        """Number of strict lower-triangular original matrix blocks."""
        return int(self.n_off_col_ptr[-1])

    @property
    def num_levels(self) -> int:
        """Number of elimination tree levels."""
        return max(0, int(self.level_ptr.size) - 1)


def build_constraint_graph(body1: np.ndarray, body2: np.ndarray) -> ConstraintGraph:
    """Build the constraint graph from active-block body pairs."""
    b1 = np.asarray(body1, dtype=np.int32)
    b2 = np.asarray(body2, dtype=np.int32)
    if b1.shape != b2.shape:
        raise ValueError(f"body1 and body2 must have identical shape, got {b1.shape} and {b2.shape}")

    graph = ConstraintGraph(num_nodes=int(b1.size))
    body_to_blocks: dict[int, list[int]] = defaultdict(list)
    for block in range(int(b1.size)):
        if b1[block] >= 0:
            body_to_blocks[int(b1[block])].append(block)
        if b2[block] >= 0:
            body_to_blocks[int(b2[block])].append(block)

    for blocks in body_to_blocks.values():
        for i, block_i in enumerate(blocks):
            for block_j in blocks[i + 1 :]:
                graph.add_edge(block_i, block_j)
    return graph


def _meca_ordering(graph: ConstraintGraph) -> tuple[list[int], list[set[tuple[int, int]]]]:
    """Minimum-edge-creation elimination ordering."""
    remaining = set(range(graph.num_nodes))
    order: list[int] = []
    fill_per_pivot: list[set[tuple[int, int]]] = []
    work_graph = graph.copy()

    while remaining:
        best_node = -1
        best_fill_edges: set[tuple[int, int]] = set()
        best_fill_count = graph.num_nodes * graph.num_nodes + 1

        for node in sorted(remaining):
            neighbors = sorted(work_graph.neighbors(node) & remaining)
            fill_edges: set[tuple[int, int]] = set()
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1 :]:
                    if n2 not in work_graph.neighbors(n1):
                        fill_edges.add((min(n1, n2), max(n1, n2)))
            if len(fill_edges) < best_fill_count:
                best_node = node
                best_fill_edges = fill_edges
                best_fill_count = len(fill_edges)

        order.append(best_node)
        fill_per_pivot.append(best_fill_edges)
        for n1, n2 in best_fill_edges:
            work_graph.add_edge(n1, n2)
        remaining.remove(best_node)

    return order, fill_per_pivot


def _natural_ordering(graph: ConstraintGraph) -> tuple[list[int], list[set[tuple[int, int]]]]:
    order = list(range(graph.num_nodes))
    fill_per_pivot: list[set[tuple[int, int]]] = [set() for _ in order]
    work_graph = graph.copy()
    eliminated: set[int] = set()
    for pivot_index, pivot in enumerate(order):
        neighbors = sorted(work_graph.neighbors(pivot) - eliminated)
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1 :]:
                if n2 not in work_graph.neighbors(n1):
                    edge = (min(n1, n2), max(n1, n2))
                    fill_per_pivot[pivot_index].add(edge)
                    work_graph.add_edge(*edge)
        eliminated.add(pivot)
    return order, fill_per_pivot


def compute_block_sparse_symbolic(
    body1: np.ndarray,
    body2: np.ndarray,
    block_row_counts: np.ndarray,
    *,
    use_meca: bool = True,
) -> BlockSparseSymbolic:
    """Compute symbolic factorization for active articulation blocks.

    Args:
        body1: Parent body index per active block, ``-1`` for static.
        body2: Child body index per active block, ``-1`` for static.
        block_row_counts: Live equality row count per active block.
        use_meca: Use MECA ordering instead of natural block order.

    Returns:
        Symbolic factorization data in pivot order.
    """
    b1 = np.asarray(body1, dtype=np.int32)
    b2 = np.asarray(body2, dtype=np.int32)
    row_counts = np.asarray(block_row_counts, dtype=np.int32)
    if b1.shape != b2.shape or b1.shape != row_counts.shape:
        raise ValueError(
            "body1, body2, and block_row_counts must have identical shape, "
            f"got {b1.shape}, {b2.shape}, {row_counts.shape}"
        )

    active = np.nonzero(row_counts > 0)[0].astype(np.int32)
    if active.size == 0:
        return _empty_symbolic()

    b1_active = b1[active]
    b2_active = b2[active]
    row_counts_active = row_counts[active]
    graph = build_constraint_graph(b1_active, b2_active)
    order, fill_per_pivot = _meca_ordering(graph) if use_meca else _natural_ordering(graph)
    pivot_local = np.asarray(order, dtype=np.int32)
    pivot_order = active[pivot_local].astype(np.int32)

    num_blocks = int(active.size)
    inv_local = np.full(num_blocks, -1, dtype=np.int32)
    for pivot, local_block in enumerate(pivot_local):
        inv_local[int(local_block)] = pivot

    inv_pivot_order = np.full(int(row_counts.size), -1, dtype=np.int32)
    for pivot, raw_block in enumerate(pivot_order):
        inv_pivot_order[int(raw_block)] = pivot

    block_sizes = row_counts_active[pivot_local].astype(np.int32)
    block_offsets = np.zeros(num_blocks + 1, dtype=np.int32)
    np.cumsum(block_sizes, out=block_offsets[1:])
    total_rows = int(block_offsets[-1])

    n_off_pairs: set[tuple[int, int]] = set()
    body_to_pivots: dict[int, list[int]] = defaultdict(list)
    for pivot, local_block in enumerate(pivot_local):
        for body in (int(b1_active[local_block]), int(b2_active[local_block])):
            if body >= 0:
                body_to_pivots[body].append(pivot)
    for pivots in body_to_pivots.values():
        pivots_sorted = sorted(pivots)
        for i, p0 in enumerate(pivots_sorted):
            for p1 in pivots_sorted[i + 1 :]:
                n_off_pairs.add((max(p0, p1), min(p0, p1)))

    n_off_col_ptr, n_off_row_idx, n_off_col_idx = _build_csc(num_blocks, n_off_pairs)

    l_pattern = set(n_off_pairs)
    all_fill_edges: set[tuple[int, int]] = set()
    for local_fill_edges in fill_per_pivot:
        for e0, e1 in local_fill_edges:
            p0 = int(inv_local[e0])
            p1 = int(inv_local[e1])
            if p0 >= 0 and p1 >= 0:
                edge = (max(p0, p1), min(p0, p1))
                l_pattern.add(edge)
                all_fill_edges.add(edge)

    parent = np.full(num_blocks, -1, dtype=np.int32)
    for col in range(num_blocks):
        rows = sorted(row for row, c in l_pattern if c == col and row > col)
        if rows:
            parent[col] = rows[0]
        for i, row_i in enumerate(rows):
            for row_j in rows[i + 1 :]:
                l_pattern.add((row_j, row_i))

    l_col_ptr, l_row_idx, _l_col_idx_unused = _build_csc(num_blocks, l_pattern)
    l_row_ptr, l_col_idx, l_csr_to_csc = _build_csr_from_csc(num_blocks, l_col_ptr, l_row_idx)

    l_lookup: dict[tuple[int, int], int] = {}
    for col in range(num_blocks):
        for slot in range(int(l_col_ptr[col]), int(l_col_ptr[col + 1])):
            l_lookup[(int(l_row_idx[slot]), col)] = slot

    n_off_to_l = np.zeros(max(int(n_off_col_ptr[-1]), 1), dtype=np.int32)
    for col in range(num_blocks):
        for slot in range(int(n_off_col_ptr[col]), int(n_off_col_ptr[col + 1])):
            row = int(n_off_row_idx[slot])
            n_off_to_l[slot] = l_lookup[(row, col)]

    pred_diag_ptr, pred_diag_slot = _build_pred_diag(num_blocks, l_row_ptr, l_csr_to_csc)
    pred_off_ptr, pred_off_slot_ik, pred_off_slot_jk = _build_pred_off(
        num_blocks, l_col_ptr, l_row_idx, l_row_ptr, l_col_idx, l_csr_to_csc
    )
    level_ptr, level_pivots = _build_levels(parent)

    return BlockSparseSymbolic(
        num_blocks=num_blocks,
        block_size=_BLOCK_SIZE,
        block_sizes=block_sizes,
        block_offsets=block_offsets,
        total_rows=total_rows,
        pivot_order=pivot_order,
        inv_pivot_order=inv_pivot_order,
        l_col_ptr=l_col_ptr,
        l_row_idx=l_row_idx,
        l_row_ptr=l_row_ptr,
        l_col_idx=l_col_idx,
        l_csr_to_csc=l_csr_to_csc,
        n_off_col_ptr=n_off_col_ptr,
        n_off_row_idx=n_off_row_idx,
        n_off_col_idx=n_off_col_idx,
        n_off_to_l=n_off_to_l,
        parent=parent,
        pred_diag_ptr=pred_diag_ptr,
        pred_diag_slot=pred_diag_slot,
        pred_off_ptr=pred_off_ptr,
        pred_off_slot_ik=pred_off_slot_ik,
        pred_off_slot_jk=pred_off_slot_jk,
        level_ptr=level_ptr,
        level_pivots=level_pivots,
        fill_edges=frozenset(all_fill_edges),
    )


def _build_csc(num_blocks: int, pairs: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols: dict[int, list[int]] = defaultdict(list)
    for row, col in pairs:
        if row > col:
            cols[int(col)].append(int(row))
    for rows in cols.values():
        rows.sort()

    nnz = sum(len(rows) for rows in cols.values())
    col_ptr = np.zeros(num_blocks + 1, dtype=np.int32)
    row_idx = np.zeros(max(nnz, 1), dtype=np.int32)
    col_idx = np.zeros(max(nnz, 1), dtype=np.int32)
    cursor = 0
    for col in range(num_blocks):
        col_ptr[col] = cursor
        for row in cols.get(col, []):
            row_idx[cursor] = row
            col_idx[cursor] = col
            cursor += 1
    col_ptr[num_blocks] = cursor
    return col_ptr, row_idx, col_idx


def _build_csr_from_csc(
    num_blocks: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for col in range(num_blocks):
        for slot in range(int(col_ptr[col]), int(col_ptr[col + 1])):
            rows[int(row_idx[slot])].append((col, slot))
    for row_entries in rows.values():
        row_entries.sort()

    nnz = int(col_ptr[-1])
    row_ptr = np.zeros(num_blocks + 1, dtype=np.int32)
    col_idx = np.zeros(max(nnz, 1), dtype=np.int32)
    csr_to_csc = np.zeros(max(nnz, 1), dtype=np.int32)
    cursor = 0
    for row in range(num_blocks):
        row_ptr[row] = cursor
        for col, csc_slot in rows.get(row, []):
            col_idx[cursor] = col
            csr_to_csc[cursor] = csc_slot
            cursor += 1
    row_ptr[num_blocks] = cursor
    return row_ptr, col_idx, csr_to_csc


def _build_pred_diag(
    num_blocks: int,
    row_ptr: np.ndarray,
    csr_to_csc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(num_blocks + 1, dtype=np.int32)
    slots: list[int] = []
    for row in range(num_blocks):
        ptr[row] = len(slots)
        for ridx in range(int(row_ptr[row]), int(row_ptr[row + 1])):
            slots.append(int(csr_to_csc[ridx]))
    ptr[num_blocks] = len(slots)
    return ptr, np.asarray(slots if slots else [0], dtype=np.int32)


def _build_pred_off(
    num_blocks: int,
    col_ptr: np.ndarray,
    row_idx: np.ndarray,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    csr_to_csc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nnz_l = int(col_ptr[-1])
    pred_ptr = np.zeros(nnz_l + 1, dtype=np.int32)
    pred_ik: list[int] = []
    pred_jk: list[int] = []

    row_slots: dict[int, dict[int, int]] = {}
    for row in range(num_blocks):
        row_slots[row] = {
            int(col_idx[ridx]): int(csr_to_csc[ridx]) for ridx in range(int(row_ptr[row]), int(row_ptr[row + 1]))
        }

    for col in range(num_blocks):
        col_predecessors = row_slots[col]
        for slot in range(int(col_ptr[col]), int(col_ptr[col + 1])):
            row = int(row_idx[slot])
            pred_ptr[slot] = len(pred_ik)
            for pred_col, slot_jk in col_predecessors.items():
                slot_ik = row_slots[row].get(pred_col, -1)
                if slot_ik >= 0:
                    pred_ik.append(slot_ik)
                    pred_jk.append(slot_jk)
    pred_ptr[nnz_l] = len(pred_ik)

    return (
        pred_ptr,
        np.asarray(pred_ik if pred_ik else [0], dtype=np.int32),
        np.asarray(pred_jk if pred_jk else [0], dtype=np.int32),
    )


def _build_levels(parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_blocks = int(parent.size)
    if num_blocks == 0:
        return np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32)

    children = [[] for _ in range(num_blocks)]
    roots: list[int] = []
    for block, p in enumerate(parent):
        if p < 0:
            roots.append(block)
        else:
            children[int(p)].append(block)

    depth = np.zeros(num_blocks, dtype=np.int32)
    stack = roots[:]
    while stack:
        node = stack.pop()
        for child in children[node]:
            depth[child] = depth[node] + 1
            stack.append(child)

    num_levels = int(depth.max()) + 1
    counts = np.zeros(num_levels + 1, dtype=np.int32)
    for d in depth:
        counts[int(d) + 1] += 1
    level_ptr = np.zeros(num_levels + 1, dtype=np.int32)
    np.cumsum(counts[1:], out=level_ptr[1:])
    cursors = level_ptr.copy()
    level_pivots = np.zeros(num_blocks, dtype=np.int32)
    for block, d in enumerate(depth):
        slot = int(cursors[int(d)])
        level_pivots[slot] = block
        cursors[int(d)] += 1
    return level_ptr, level_pivots


def _empty_symbolic() -> BlockSparseSymbolic:
    zeros1 = np.zeros(1, dtype=np.int32)
    return BlockSparseSymbolic(
        num_blocks=0,
        block_size=_BLOCK_SIZE,
        block_sizes=np.zeros(0, dtype=np.int32),
        block_offsets=zeros1.copy(),
        total_rows=0,
        pivot_order=np.zeros(0, dtype=np.int32),
        inv_pivot_order=np.zeros(0, dtype=np.int32),
        l_col_ptr=zeros1.copy(),
        l_row_idx=zeros1.copy(),
        l_row_ptr=zeros1.copy(),
        l_col_idx=zeros1.copy(),
        l_csr_to_csc=zeros1.copy(),
        n_off_col_ptr=zeros1.copy(),
        n_off_row_idx=zeros1.copy(),
        n_off_col_idx=zeros1.copy(),
        n_off_to_l=zeros1.copy(),
        parent=np.zeros(0, dtype=np.int32),
        pred_diag_ptr=zeros1.copy(),
        pred_diag_slot=zeros1.copy(),
        pred_off_ptr=zeros1.copy(),
        pred_off_slot_ik=zeros1.copy(),
        pred_off_slot_jk=zeros1.copy(),
        level_ptr=zeros1.copy(),
        level_pivots=np.zeros(0, dtype=np.int32),
        fill_edges=frozenset(),
    )
