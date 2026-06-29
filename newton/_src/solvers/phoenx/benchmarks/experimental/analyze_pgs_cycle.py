# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analyze adjacent-aggregate coarse correction on a cyclic joint graph."""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.symbolic import build_constraint_graph
from newton._src.solvers.phoenx.benchmarks.experimental.analyze_pgs_branched_tree import _two_level_colored
from newton._src.solvers.phoenx.benchmarks.experimental.analyze_pgs_motor_chain import BlockSGS, _iterate
from newton._src.solvers.phoenx.constraints.constraint_joint import JOINT_MODE_BALL_SOCKET
from newton._src.solvers.phoenx.tests.test_articulation_dvi import _make_adbs_world


def _adjacent_mapping(body1: np.ndarray, body2: np.ndarray) -> np.ndarray:
    graph = build_constraint_graph(body1, body2)
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


def _neighbor_interpolation(body1: np.ndarray, body2: np.ndarray, rows: int) -> np.ndarray:
    graph = build_constraint_graph(body1, body2)
    coarse_nodes: list[int] = []
    covered: set[int] = set()
    for node in range(graph.num_nodes):
        if node not in covered:
            coarse_nodes.append(node)
            covered.add(node)
            covered.update(graph.neighbors(node))
    coarse_index = {node: index for index, node in enumerate(coarse_nodes)}
    prolongation = np.zeros((graph.num_nodes * rows, len(coarse_nodes) * rows))
    for fine in range(graph.num_nodes):
        if fine in coarse_index:
            weights = ((coarse_index[fine], 1.0),)
        else:
            neighbors = [coarse_index[node] for node in sorted(graph.neighbors(fine)) if node in coarse_index]
            weight = 1.0 / len(neighbors)
            weights = tuple((coarse, weight) for coarse in neighbors)
        for row in range(rows):
            for coarse, weight in weights:
                prolongation[fine * rows + row, coarse * rows + row] = weight
    return prolongation


def run(args: argparse.Namespace) -> dict[str, object]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("cycle convergence analysis requires CUDA")
    nodes = int(args.nodes)
    dynamic_bodies = np.arange(1, nodes + 1, dtype=np.int32)
    body1 = dynamic_bodies
    body2 = np.roll(dynamic_bodies, -1)
    angle = 2.0 * np.pi * np.arange(nodes) / nodes
    positions = np.zeros((nodes + 1, 3), dtype=np.float32)
    positions[1:, 0] = np.cos(angle)
    positions[1:, 1] = np.sin(angle)
    world = _make_adbs_world(
        device,
        body1,
        body2,
        np.full(nodes, int(JOINT_MODE_BALL_SOCKET), dtype=np.int32),
        positions_np=positions,
        world_kwargs={"cache_articulation_topology": True, "gravity": (0.0, 0.0, 0.0)},
    )
    system = world.articulation_device_system
    system.populate_from_adbs_constraints(world.constraints, world.bodies, dt=args.dt, device=device)
    system.assemble_dense_matrix(world.bodies.inverse_mass, world.bodies.inverse_inertia_world, device=device)
    size = system.total_rows
    matrix = system.matrix.numpy()[:size, :size].astype(np.float64)
    matrix += float(args.regularization) * np.eye(size)
    rng = np.random.default_rng(args.seed)
    rhs = rng.normal(size=size)
    offsets = system.active_block_offsets.numpy().astype(np.int32)
    rows = int(offsets[1] - offsets[0])
    sgs = BlockSGS(matrix, rhs, offsets)

    mapping = _adjacent_mapping(body1, body2)
    coarse_blocks = int(mapping.max()) + 1
    prolongation = np.zeros((size, coarse_blocks * rows))
    for fine, coarse in enumerate(mapping):
        for row in range(rows):
            prolongation[fine * rows + row, coarse * rows + row] = 1.0

    results = [_iterate("block_sgs", sgs, matrix, rhs, args.iterations)]
    color_count = 0
    for color_sweeps in (8, 16, 32):
        update, _, color_count = _two_level_colored(sgs, prolongation, rows, color_sweeps)
        results.append(
            _iterate(
                f"adjacent_aggregate_{color_sweeps}",
                update,
                matrix,
                rhs,
                args.iterations,
            )
        )
    interpolated = _neighbor_interpolation(body1, body2, rows)
    interpolated_colors = 0
    for color_sweeps in (8, 16, 32):
        update, _, interpolated_colors = _two_level_colored(sgs, interpolated, rows, color_sweeps)
        results.append(
            _iterate(
                f"neighbor_interpolation_{color_sweeps}",
                update,
                matrix,
                rhs,
                args.iterations,
            )
        )
    return {
        "nodes": nodes,
        "fine_condition": float(np.linalg.cond(matrix)),
        "coarse_blocks": coarse_blocks,
        "coarse_colors": color_count,
        "interpolated_coarse_blocks": interpolated.shape[1] // rows,
        "interpolated_coarse_colors": interpolated_colors,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--dt", type=float, default=1.0 / 6000.0)
    parser.add_argument("--regularization", type=float, default=1.0e-3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
