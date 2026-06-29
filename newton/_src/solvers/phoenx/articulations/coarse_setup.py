# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Topology setup for PhoenX bilateral articulation coarse corrections."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .coarse_aggregate import CoarseAggregateSolver, parent_aggregate_mapping
from .coarse_path import CoarsePathSolver
from .symbolic import BlockSparseSymbolic, build_constraint_graph, compute_block_sparse_symbolic
from .topology import ArticulationTopology


@dataclass(frozen=True)
class ArticulationCoarseSetup:
    """Resolved topology, symbolic matrix layout, and runtime coarse solver."""

    mode: str
    symbolic: BlockSparseSymbolic
    solver: CoarsePathSolver | CoarseAggregateSolver


def normalize_articulation_coarse_mode(value: str | None) -> str | None:
    """Normalize the optional articulation coarse-correction mode."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized not in {"auto", "path", "tree", "graph"}:
        raise ValueError("articulation_coarse_mode must be None, auto, path, tree, or graph")
    return normalized


def _shared_rows(topology: ArticulationTopology) -> int:
    counts = topology.active_row_counts
    if counts.size == 0:
        raise ValueError("articulation coarse correction requires at least one active joint block")
    # PhoenX articulation rows begin with the three anchor-translation rows.
    # Restricting to the common prefix gives mixed joint trees a physically
    # meaningful coarse space while fine PGS retains every remaining row.
    return int(np.min(counts))


def _path_component_count(topology: ArticulationTopology) -> tuple[int, int]:
    graph = build_constraint_graph(topology.active_body1, topology.active_body2)
    remaining = set(range(graph.num_nodes))
    lengths: list[int] = []
    while remaining:
        pending = [min(remaining)]
        component: set[int] = set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(graph.neighbors(node) - component)
        remaining -= component
        if any(len(graph.neighbors(node) & component) > 2 for node in component):
            raise ValueError("articulation constraint graph is not a path forest")
        edge_count = sum(len(graph.neighbors(node) & component) for node in component) // 2
        if edge_count != len(component) - 1:
            raise ValueError("articulation constraint graph contains a cycle")
        ordered = sorted(component)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise ValueError("path blocks must be contiguous in active-joint order")
        if any(next_node not in graph.neighbors(node) for node, next_node in pairwise(ordered)):
            raise ValueError("path blocks must follow active-joint order")
        lengths.append(len(ordered))
    if len(set(lengths)) != 1:
        raise ValueError("packed path correction currently requires equal-length path islands")
    if lengths[0] > 128:
        raise ValueError("packed path correction currently supports at most 128 joints per path")
    return len(lengths), lengths[0]


def _rooted_joint_depths(body1: np.ndarray, body2: np.ndarray) -> np.ndarray:
    child_to_joint: dict[int, int] = {}
    for joint, child_value in enumerate(body2):
        child = int(child_value)
        if child < 0:
            raise ValueError("articulation tree joints must have a dynamic child body")
        if child in child_to_joint:
            raise ValueError("articulation tree requires each dynamic body to have one parent joint")
        child_to_joint[child] = joint
    depth = np.zeros(body1.size, dtype=np.int32)
    visiting = np.zeros(body1.size, dtype=bool)

    def resolve(joint: int) -> int:
        if depth[joint] > 0:
            return int(depth[joint])
        if visiting[joint]:
            raise ValueError("articulation coarse correction does not support cyclic joint graphs")
        visiting[joint] = True
        parent_joint = child_to_joint.get(int(body1[joint]))
        value = 1 if parent_joint is None else resolve(parent_joint) + 1
        visiting[joint] = False
        depth[joint] = value
        return value

    for joint in range(body1.size):
        resolve(joint)
    return depth


def _graph_aggregate_mapping(topology: ArticulationTopology) -> np.ndarray:
    """Build deterministic one-hot aggregates for a general constraint graph."""
    graph = build_constraint_graph(topology.active_body1, topology.active_body2)
    mapping = np.full(graph.num_nodes, -1, dtype=np.int32)
    coarse = 0
    for node in range(graph.num_nodes):
        if mapping[node] >= 0:
            continue
        mapping[node] = coarse
        for neighbor in sorted(graph.neighbors(node)):
            if mapping[neighbor] < 0:
                mapping[neighbor] = coarse
                break
        coarse += 1
    return mapping


def build_articulation_coarse_setup(
    topology: ArticulationTopology,
    *,
    mode: str,
    color_sweeps: int,
    device,
) -> ArticulationCoarseSetup:
    """Build a path, rooted-tree, or general-graph coarse correction."""
    resolved = normalize_articulation_coarse_mode(mode)
    if resolved is None:
        raise ValueError("coarse setup requires an enabled mode")
    rows = _shared_rows(topology)
    symbolic = compute_block_sparse_symbolic(
        topology.active_body1,
        topology.active_body2,
        topology.active_row_counts,
        use_meca=False,
    )

    if resolved in {"auto", "path"}:
        try:
            path_count, _ = _path_component_count(topology)
        except ValueError:
            if resolved == "path":
                raise
        else:
            solver = CoarsePathSolver(
                topology.active_joint_count,
                rows,
                color_sweeps,
                device,
                path_count=path_count,
            )
            return ArticulationCoarseSetup("path", symbolic, solver)

    body1 = topology.active_body1
    body2 = topology.active_body2
    mapping: np.ndarray | None = None
    aggregate_mode = resolved
    if resolved in {"auto", "tree"}:
        try:
            depth = _rooted_joint_depths(body1, body2)
        except ValueError:
            if resolved == "tree":
                raise
        else:
            mapping = parent_aggregate_mapping(body1, body2, depth)
            aggregate_mode = "tree"
    if mapping is None:
        mapping = _graph_aggregate_mapping(topology)
        aggregate_mode = "graph"
    solver = CoarseAggregateSolver(
        mapping,
        symbolic.n_off_row_idx[: symbolic.nnz_n],
        symbolic.n_off_col_idx[: symbolic.nnz_n],
        rows,
        color_sweeps,
        device,
    )
    if solver.coarse_blocks > 256:
        raise ValueError("tree coarse correction currently supports at most 256 aggregate blocks")
    return ArticulationCoarseSetup(aggregate_mode, symbolic, solver)
