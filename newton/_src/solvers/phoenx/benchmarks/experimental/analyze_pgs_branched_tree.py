# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analyze alternating-depth coarse spaces on a branched articulation."""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem
from newton._src.solvers.phoenx.articulations.symbolic import compute_block_sparse_symbolic
from newton._src.solvers.phoenx.benchmarks.experimental.analyze_pgs_motor_chain import (
    BlockSGS,
    _anderson_fixed_point,
    _iterate,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import JOINT_MODE_BALL_SOCKET
from newton._src.solvers.phoenx.tests.test_articulation_dvi import _make_adbs_world


def _tree_edges(trunk: int, arms: int, arm_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body1: list[int] = []
    body2: list[int] = []
    depth: list[int] = []
    parent = 0
    next_body = 1
    for level in range(1, trunk + 1):
        body1.append(parent)
        body2.append(next_body)
        depth.append(level)
        parent = next_body
        next_body += 1
    branch_body = parent
    for _ in range(arms):
        parent = branch_body
        for local in range(1, arm_length + 1):
            body1.append(parent)
            body2.append(next_body)
            depth.append(trunk + local)
            parent = next_body
            next_body += 1
    return np.asarray(body1, dtype=np.int32), np.asarray(body2, dtype=np.int32), np.asarray(depth, dtype=np.int32)


def _build_system(device, trunk: int, arms: int, arm_length: int, dt: float):
    body1, body2, depth = _tree_edges(trunk, arms, arm_length)
    positions = np.zeros((int(body2.max()) + 1, 3), dtype=np.float32)
    world = _make_adbs_world(
        device,
        body1,
        body2,
        np.full(body1.size, int(JOINT_MODE_BALL_SOCKET), dtype=np.int32),
        positions_np=positions,
        world_kwargs={"cache_articulation_topology": True, "gravity": (0.0, 0.0, 0.0)},
    )
    velocity = np.zeros_like(positions)
    velocity[1:, 2] = -9.81 * dt
    world.bodies.velocity.assign(velocity)
    topology = world.articulation_topology
    symbolic = compute_block_sparse_symbolic(
        topology.active_body1,
        topology.active_body2,
        topology.active_row_counts,
        use_meca=False,
    )
    system = ArticulationDeviceSystem.from_topology(topology, device, symbolic)
    system.populate_from_adbs_constraints(world.constraints, world.bodies, dt=dt, device=device)
    system.compute_residual(world.bodies, dt=dt, recovery_speed=0.0, device=device)
    system.assemble_dense_matrix(world.bodies.inverse_mass, world.bodies.inverse_inertia_world, device=device)
    rows = system.total_rows
    return (
        system.matrix.numpy()[:rows, :rows].astype(np.float64),
        system.rhs.numpy()[:rows].astype(np.float64),
        system.active_block_offsets.numpy().astype(np.int32),
        body1,
        body2,
        depth,
    )


def _tree_prolongation(
    body1: np.ndarray,
    body2: np.ndarray,
    depth: np.ndarray,
    rows: int,
    mode: str,
):
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
    prolongation = np.zeros((body1.size * rows, coarse_joints.size * rows))
    for joint in range(body1.size):
        if coarse_mask[joint]:
            weights = ((coarse_index[joint], 1.0),)
        else:
            parent_joint = joint_for_child.get(int(body1[joint]))
            coarse_parent = parent_joint if parent_joint is not None and coarse_mask[parent_joint] else None
            coarse_children = [child for child in children[joint] if coarse_mask[child]]
            if mode == "parent" and coarse_parent is not None:
                weights = ((coarse_index[coarse_parent], 1.0),)
            elif mode == "balanced" and coarse_parent is not None and coarse_children:
                child_weight = 0.5 / len(coarse_children)
                weights = (
                    (coarse_index[coarse_parent], 0.5),
                    *((coarse_index[child], child_weight) for child in coarse_children),
                )
            else:
                neighbors = ([coarse_parent] if coarse_parent is not None else []) + coarse_children
                weight = 1.0 / len(neighbors)
                weights = tuple((coarse_index[neighbor], weight) for neighbor in neighbors)
        for row in range(rows):
            for coarse, weight in weights:
                prolongation[joint * rows + row, coarse * rows + row] = weight
    return prolongation, coarse_joints


def _parent_aggregate_prolongation(
    body1: np.ndarray,
    body2: np.ndarray,
    depth: np.ndarray,
    rows: int,
    stride: int,
):
    joint_for_child = {int(child): joint for joint, child in enumerate(body2)}
    children: list[list[int]] = [[] for _ in range(body1.size)]
    for joint, parent_body in enumerate(body1):
        parent_joint = joint_for_child.get(int(parent_body))
        if parent_joint is not None:
            children[parent_joint].append(joint)
    coarse_mask = (depth - 1) % stride == 0
    coarse_mask |= np.asarray([not child_joints for child_joints in children])
    coarse_joints = np.nonzero(coarse_mask)[0]
    coarse_index = {int(joint): index for index, joint in enumerate(coarse_joints)}
    prolongation = np.zeros((body1.size * rows, coarse_joints.size * rows))
    for joint in range(body1.size):
        representative = joint
        while not coarse_mask[representative]:
            representative = joint_for_child[int(body1[representative])]
        for row in range(rows):
            prolongation[joint * rows + row, coarse_index[representative] * rows + row] = 1.0
    return prolongation, coarse_joints


def _two_level_exact(sgs: BlockSGS, prolongation: np.ndarray):
    coarse_matrix = prolongation.T @ sgs.matrix @ prolongation

    def update(x: np.ndarray) -> np.ndarray:
        x = sgs.forward(x)
        residual = sgs.rhs - sgs.matrix @ x
        x += prolongation @ np.linalg.solve(coarse_matrix, prolongation.T @ residual)
        return sgs.backward(x)

    return update, coarse_matrix


def _greedy_block_colors(matrix: np.ndarray, rows: int) -> np.ndarray:
    blocks = matrix.shape[0] // rows
    colors = np.full(blocks, -1, dtype=np.int32)
    threshold = 1.0e-12 * np.linalg.norm(matrix)
    for block in range(blocks):
        forbidden = {
            int(colors[other])
            for other in range(block)
            if colors[other] >= 0
            and np.linalg.norm(matrix[block * rows : (block + 1) * rows, other * rows : (other + 1) * rows]) > threshold
        }
        color = 0
        while color in forbidden:
            color += 1
        colors[block] = color
    return colors


def _two_level_colored(
    sgs: BlockSGS,
    prolongation: np.ndarray,
    rows: int,
    color_sweeps: int,
):
    coarse_matrix = prolongation.T @ sgs.matrix @ prolongation
    colors = _greedy_block_colors(coarse_matrix, rows)
    color_count = int(colors.max()) + 1

    def update(x: np.ndarray) -> np.ndarray:
        x = sgs.forward(x)
        coarse_rhs = prolongation.T @ (sgs.rhs - sgs.matrix @ x)
        coarse_error = np.zeros_like(coarse_rhs)
        for sweep in range(color_sweeps):
            color = sweep % color_count
            for block in np.nonzero(colors == color)[0]:
                row = slice(block * rows, (block + 1) * rows)
                residual = coarse_rhs[row] - coarse_matrix[row] @ coarse_error
                coarse_error[row] += np.linalg.solve(coarse_matrix[row, row], residual)
        return sgs.backward(x + prolongation @ coarse_error)

    return update, coarse_matrix, color_count


def _two_level_colored_anderson(
    sgs: BlockSGS,
    prolongation: np.ndarray,
    rows: int,
    anderson_iterations: int,
):
    coarse_matrix = prolongation.T @ sgs.matrix @ prolongation
    colors = _greedy_block_colors(coarse_matrix, rows)
    color_count = int(colors.max()) + 1

    def update(x: np.ndarray) -> np.ndarray:
        x = sgs.forward(x)
        coarse_rhs = prolongation.T @ (sgs.rhs - sgs.matrix @ x)

        def coarse_cycle(error: np.ndarray) -> np.ndarray:
            error = error.copy()
            for color in (*range(color_count), *range(color_count - 1, -1, -1)):
                for block in np.nonzero(colors == color)[0]:
                    row = slice(block * rows, (block + 1) * rows)
                    residual = coarse_rhs[row] - coarse_matrix[row] @ error
                    error[row] += np.linalg.solve(coarse_matrix[row, row], residual)
            return error

        coarse_error = _anderson_fixed_point(
            coarse_cycle,
            coarse_matrix,
            coarse_rhs,
            anderson_iterations,
            depth=2,
        )
        return sgs.backward(x + prolongation @ coarse_error)

    return update


def run(args: argparse.Namespace) -> dict[str, object]:
    matrix, rhs, offsets, body1, body2, depth = _build_system(
        wp.get_device(args.device), args.trunk, args.arms, args.arm_length, args.dt
    )
    rows = int(offsets[1] - offsets[0])
    sgs = BlockSGS(matrix, rhs, offsets)
    methods = [_iterate("block_sgs", sgs, matrix, rhs, args.iterations)]
    coarse_matrix = None
    coarse_joints = None
    coarse_color_count = 0
    for mode in ("uniform", "balanced", "parent"):
        prolongation, mode_coarse_joints = _tree_prolongation(body1, body2, depth, rows, mode)
        two_level, mode_coarse_matrix = _two_level_exact(sgs, prolongation)
        methods.append(_iterate(f"two_level_tree_{mode}", two_level, matrix, rhs, args.iterations))
        if mode == "parent":
            coarse_matrix = mode_coarse_matrix
            coarse_joints = mode_coarse_joints
            for color_sweeps in (8, 16, 32):
                colored, _, coarse_color_count = _two_level_colored(sgs, prolongation, rows, color_sweeps)
                methods.append(
                    _iterate(
                        f"two_level_tree_parent_colored_{color_sweeps}",
                        colored,
                        matrix,
                        rhs,
                        args.iterations,
                    )
                )
            for anderson_iterations in (2, 4, 8):
                accelerated = _two_level_colored_anderson(sgs, prolongation, rows, anderson_iterations)
                methods.append(
                    _iterate(
                        f"two_level_tree_parent_anderson_{anderson_iterations}",
                        accelerated,
                        matrix,
                        rhs,
                        args.iterations,
                    )
                )
    for stride in (4, 8):
        prolongation, _ = _parent_aggregate_prolongation(body1, body2, depth, rows, stride)
        (
            exact,
            _,
        ) = _two_level_exact(sgs, prolongation)
        methods.append(_iterate(f"two_level_tree_parent_stride_{stride}", exact, matrix, rhs, args.iterations))
        for color_sweeps in (16, 32):
            colored, _, _color_count = _two_level_colored(sgs, prolongation, rows, color_sweeps)
            methods.append(
                _iterate(
                    f"two_level_tree_parent_stride_{stride}_colored_{color_sweeps}",
                    colored,
                    matrix,
                    rhs,
                    args.iterations,
                )
            )
    assert coarse_matrix is not None and coarse_joints is not None
    block_norm = np.zeros((coarse_joints.size, coarse_joints.size))
    for i in range(coarse_joints.size):
        for j in range(coarse_joints.size):
            block_norm[i, j] = np.linalg.norm(coarse_matrix[i * rows : (i + 1) * rows, j * rows : (j + 1) * rows])
    threshold = 1.0e-12 * block_norm.max()
    block_nnz = int(np.count_nonzero(block_norm > threshold))
    return {
        "fine_blocks": int(body1.size),
        "coarse_blocks": int(coarse_joints.size),
        "coarse_block_nnz": block_nnz,
        "coarse_color_count": coarse_color_count,
        "coarse_block_density": block_nnz / float(coarse_joints.size**2),
        "fine_condition": float(np.linalg.cond(matrix)),
        "coarse_condition": float(np.linalg.cond(coarse_matrix)),
        "results": methods,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trunk", type=int, default=32)
    parser.add_argument("--arms", type=int, default=3)
    parser.add_argument("--arm-length", type=int, default=24)
    parser.add_argument("--dt", type=float, default=1.0 / 6000.0)
    parser.add_argument("--iterations", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
