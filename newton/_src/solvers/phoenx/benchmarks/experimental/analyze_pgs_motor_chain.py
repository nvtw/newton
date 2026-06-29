# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analyze fixed-point accelerators on the motor chain Delassus matrix."""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples import example_motorized_hinge_chain as scene
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _build_system(device: wp.context.Device, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bodies = body_container_zeros(scene.NUM_BODIES, device=device)
    scene._populate_chain_bodies(bodies, device)
    constraints = PhoenXWorld.make_constraint_container(num_joints=scene.NUM_HINGES, device=device)
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        substeps=1,
        solver_iterations=1,
        velocity_iterations=0,
        gravity=(0.0, 0.0, -9.81),
        rigid_contact_max=0,
        num_joints=scene.NUM_HINGES,
        cache_articulation_topology=True,
        device=device,
    )
    world.initialize_actuated_double_ball_socket_joints(**scene._build_joint_arrays(device))
    velocity = world.bodies.velocity.numpy()
    velocity[1:, 2] = -9.81 * dt
    world.bodies.velocity.assign(velocity)

    system = world.articulation_device_system
    symbolic = world.articulation_system
    if system is None or symbolic is None:
        raise RuntimeError("failed to build articulation system")
    system.populate_from_adbs_constraints(constraints, bodies, dt=dt, device=device)
    system.compute_residual(bodies, dt=dt, alpha=0.0, recovery_speed=-1.0, device=device)
    system.assemble_dense_matrix(bodies.inverse_mass, bodies.inverse_inertia_world, device=device)
    n = system.total_rows
    matrix = system.matrix.numpy()[:n, :n].astype(np.float64)
    rhs = system.rhs.numpy()[:n].astype(np.float64)
    offsets = system.active_block_offsets.numpy().astype(np.int32)
    return matrix, rhs, offsets


def _block_triangular(matrix: np.ndarray, offsets: np.ndarray, *, lower: bool) -> np.ndarray:
    result = np.zeros_like(matrix)
    blocks = len(offsets) - 1
    for row in range(blocks):
        r = slice(offsets[row], offsets[row + 1])
        columns = range(row + 1) if lower else range(row, blocks)
        for column in columns:
            c = slice(offsets[column], offsets[column + 1])
            result[r, c] = matrix[r, c]
    return result


class BlockSGS:
    def __init__(self, matrix: np.ndarray, rhs: np.ndarray, offsets: np.ndarray):
        self.matrix = matrix
        self.rhs = rhs
        self.lower = _block_triangular(matrix, offsets, lower=True)
        self.upper = _block_triangular(matrix, offsets, lower=False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + np.linalg.solve(self.lower, self.rhs - self.matrix @ x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x + np.linalg.solve(self.upper, self.rhs - self.matrix @ x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.backward(self.forward(x))

    def iteration_matrix(self) -> np.ndarray:
        identity = np.eye(self.matrix.shape[0])
        forward = identity - np.linalg.solve(self.lower, self.matrix)
        backward = identity - np.linalg.solve(self.upper, self.matrix)
        return backward @ forward


def _relative_residual(matrix: np.ndarray, rhs: np.ndarray, x: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(rhs - matrix @ x) / denominator)


def _iterate(name: str, update, matrix: np.ndarray, rhs: np.ndarray, iterations: int) -> dict[str, object]:
    x = np.zeros_like(rhs)
    residuals = [_relative_residual(matrix, rhs, x)]
    for _ in range(iterations):
        x = update(x)
        residuals.append(_relative_residual(matrix, rhs, x))
    return {"method": name, "residuals": residuals, "final": residuals[-1]}


def _chebyshev(sgs: BlockSGS, rho: float, iterations: int) -> dict[str, object]:
    previous = np.zeros_like(sgs.rhs)
    current = sgs(previous)
    residuals = [
        _relative_residual(sgs.matrix, sgs.rhs, previous),
        _relative_residual(sgs.matrix, sgs.rhs, current),
    ]
    omega = 1.0
    for iteration in range(1, iterations):
        raw = sgs(current)
        if iteration == 1:
            omega = 2.0 / (2.0 - rho * rho)
        else:
            omega = 4.0 / (4.0 - rho * rho * omega)
        accelerated = omega * (raw - previous) + previous
        previous, current = current, accelerated
        residuals.append(_relative_residual(sgs.matrix, sgs.rhs, current))
    return {"method": "chebyshev_sgs", "rho": rho, "residuals": residuals, "final": residuals[-1]}


def _anderson(sgs: BlockSGS, depth: int, iterations: int) -> dict[str, object]:
    x = np.zeros_like(sgs.rhs)
    xs: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    residuals = [_relative_residual(sgs.matrix, sgs.rhs, x)]
    restarts = 0
    for _ in range(iterations):
        g = sgs(x)
        xs.append(x.copy())
        gs.append(g.copy())
        xs = xs[-(depth + 1) :]
        gs = gs[-(depth + 1) :]
        if len(gs) == 1:
            candidate = g
        else:
            fixed_residuals = np.column_stack([gi - xi for gi, xi in zip(gs, xs, strict=True)])
            gram = fixed_residuals.T @ fixed_residuals
            ones = np.ones(len(gs))
            kkt = np.block([[gram + 1.0e-12 * np.eye(len(gs)), ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            alpha = np.linalg.solve(kkt, np.append(np.zeros(len(gs)), 1.0))[:-1]
            candidate = np.column_stack(gs) @ alpha
        raw_residual = _relative_residual(sgs.matrix, sgs.rhs, g)
        candidate_residual = _relative_residual(sgs.matrix, sgs.rhs, candidate)
        if not np.isfinite(candidate_residual) or candidate_residual > 1.2 * raw_residual:
            candidate = g
            xs = [x.copy()]
            gs = [g.copy()]
            restarts += 1
        x = candidate
        residuals.append(_relative_residual(sgs.matrix, sgs.rhs, x))
    return {
        "method": f"anderson_{depth}",
        "restarts": restarts,
        "residuals": residuals,
        "final": residuals[-1],
    }


def _pcg(
    sgs: BlockSGS,
    preconditioner: str,
    iterations: int,
    apply_override=None,
) -> dict[str, object]:
    matrix = sgs.matrix
    rhs = sgs.rhs
    diagonal = sgs.lower + sgs.upper - matrix

    def apply_preconditioner(residual: np.ndarray) -> np.ndarray:
        if apply_override is not None:
            return apply_override(residual)
        if preconditioner == "none":
            return residual.copy()
        if preconditioner == "block_jacobi":
            return np.linalg.solve(diagonal, residual)
        forward = np.linalg.solve(sgs.lower, residual)
        return np.linalg.solve(sgs.upper, diagonal @ forward)

    x = np.zeros_like(rhs)
    residual = rhs.copy()
    z = apply_preconditioner(residual)
    direction = z.copy()
    rz = float(residual @ z)
    residuals = [_relative_residual(matrix, rhs, x)]
    for _ in range(iterations):
        matrix_direction = matrix @ direction
        denominator = float(direction @ matrix_direction)
        if denominator <= np.finfo(np.float64).eps:
            break
        alpha = rz / denominator
        x = x + alpha * direction
        residual = residual - alpha * matrix_direction
        residuals.append(_relative_residual(matrix, rhs, x))
        z = apply_preconditioner(residual)
        rz_next = float(residual @ z)
        if rz <= np.finfo(np.float64).eps:
            break
        direction = z + (rz_next / rz) * direction
        rz = rz_next
    return {
        "method": f"pcg_{preconditioner}",
        "residuals": residuals,
        "final": residuals[-1],
    }


def _coarse_sgs(sgs: BlockSGS, offsets: np.ndarray, aggregate: int):
    rows_per_joint = int(offsets[1] - offsets[0])
    joints = len(offsets) - 1
    coarse_nodes = np.arange(0, joints, aggregate, dtype=np.int32)
    if coarse_nodes[-1] != joints - 1:
        coarse_nodes = np.append(coarse_nodes, joints - 1)
    coarse_joints = len(coarse_nodes)
    prolongation = np.zeros((sgs.matrix.shape[0], coarse_joints * rows_per_joint))
    for joint in range(joints):
        upper = int(np.searchsorted(coarse_nodes, joint, side="left"))
        if upper == 0:
            weights = ((0, 1.0),)
        elif upper == coarse_joints:
            weights = ((coarse_joints - 1, 1.0),)
        elif coarse_nodes[upper] == joint:
            weights = ((upper, 1.0),)
        else:
            lower = upper - 1
            span = float(coarse_nodes[upper] - coarse_nodes[lower])
            upper_weight = float(joint - coarse_nodes[lower]) / span
            weights = ((lower, 1.0 - upper_weight), (upper, upper_weight))
        for row in range(rows_per_joint):
            for coarse, weight in weights:
                prolongation[offsets[joint] + row, coarse * rows_per_joint + row] = weight
    coarse_matrix = prolongation.T @ sgs.matrix @ prolongation

    def update(x: np.ndarray) -> np.ndarray:
        x = sgs.forward(x)
        residual = sgs.rhs - sgs.matrix @ x
        x = x + prolongation @ np.linalg.solve(coarse_matrix, prolongation.T @ residual)
        return sgs.backward(x)

    return update


def _linear_prolongation(joints: int, rows_per_joint: int) -> tuple[np.ndarray, int]:
    coarse_nodes = np.arange(0, joints, 2, dtype=np.int32)
    if coarse_nodes[-1] != joints - 1:
        coarse_nodes = np.append(coarse_nodes, joints - 1)
    coarse_joints = len(coarse_nodes)
    prolongation = np.zeros((joints * rows_per_joint, coarse_joints * rows_per_joint))
    for joint in range(joints):
        upper = int(np.searchsorted(coarse_nodes, joint, side="left"))
        if upper == 0:
            weights = ((0, 1.0),)
        elif coarse_nodes[upper] == joint:
            weights = ((upper, 1.0),)
        else:
            lower = upper - 1
            upper_weight = float(joint - coarse_nodes[lower]) / float(coarse_nodes[upper] - coarse_nodes[lower])
            weights = ((lower, 1.0 - upper_weight), (upper, upper_weight))
        for row in range(rows_per_joint):
            for coarse, weight in weights:
                prolongation[joint * rows_per_joint + row, coarse * rows_per_joint + row] = weight
    return prolongation, coarse_joints


def _anderson_fixed_point(update, matrix: np.ndarray, rhs: np.ndarray, iterations: int, depth: int) -> np.ndarray:
    x = np.zeros_like(rhs)
    xs: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    for _ in range(iterations):
        g = update(x)
        xs.append(x.copy())
        gs.append(g.copy())
        xs = xs[-(depth + 1) :]
        gs = gs[-(depth + 1) :]
        candidate = g
        if len(gs) > 1:
            fixed_residuals = np.column_stack([gi - xi for gi, xi in zip(gs, xs, strict=True)])
            gram = fixed_residuals.T @ fixed_residuals
            ones = np.ones(len(gs))
            kkt = np.block([[gram + 1.0e-12 * np.eye(len(gs)), ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            alpha = np.linalg.solve(kkt, np.append(np.zeros(len(gs)), 1.0))[:-1]
            accelerated = np.column_stack(gs) @ alpha
            raw_norm = np.linalg.norm(rhs - matrix @ g)
            accelerated_norm = np.linalg.norm(rhs - matrix @ accelerated)
            if np.isfinite(accelerated_norm) and accelerated_norm <= 1.2 * raw_norm:
                candidate = accelerated
            else:
                xs = [x.copy()]
                gs = [g.copy()]
        x = candidate
    return x


def _block_tridiagonal_solve(matrix: np.ndarray, rhs: np.ndarray, block_size: int) -> np.ndarray:
    blocks = matrix.shape[0] // block_size
    diagonal = [
        matrix[i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size].copy()
        for i in range(blocks)
    ]
    upper = [
        matrix[i * block_size : (i + 1) * block_size, (i + 1) * block_size : (i + 2) * block_size]
        for i in range(blocks - 1)
    ]
    lower = [
        matrix[(i + 1) * block_size : (i + 2) * block_size, i * block_size : (i + 1) * block_size]
        for i in range(blocks - 1)
    ]
    reduced_rhs = [rhs[i * block_size : (i + 1) * block_size].copy() for i in range(blocks)]
    for i in range(1, blocks):
        factor = np.linalg.solve(diagonal[i - 1].T, lower[i - 1].T).T
        diagonal[i] -= factor @ upper[i - 1]
        reduced_rhs[i] -= factor @ reduced_rhs[i - 1]
    result = np.zeros_like(rhs)
    result[-block_size:] = np.linalg.solve(diagonal[-1], reduced_rhs[-1])
    for i in range(blocks - 2, -1, -1):
        row = slice(i * block_size, (i + 1) * block_size)
        next_row = slice((i + 1) * block_size, (i + 2) * block_size)
        result[row] = np.linalg.solve(diagonal[i], reduced_rhs[i] - upper[i] @ result[next_row])
    return result


def _block_pcr_solve(matrix: np.ndarray, rhs: np.ndarray, block_size: int) -> np.ndarray:
    blocks = matrix.shape[0] // block_size
    lower = np.zeros((blocks, block_size, block_size), dtype=matrix.dtype)
    diagonal = np.zeros_like(lower)
    upper = np.zeros_like(lower)
    reduced_rhs = rhs.reshape(blocks, block_size).copy()
    for i in range(blocks):
        row = slice(i * block_size, (i + 1) * block_size)
        diagonal[i] = matrix[row, row]
        if i > 0:
            lower[i] = matrix[row, slice((i - 1) * block_size, i * block_size)]
        if i + 1 < blocks:
            upper[i] = matrix[row, slice((i + 1) * block_size, (i + 2) * block_size)]

    stride = 1
    while stride < blocks:
        next_lower = np.zeros_like(lower)
        next_diagonal = diagonal.copy()
        next_upper = np.zeros_like(upper)
        next_rhs = reduced_rhs.copy()
        for i in range(blocks):
            if i >= stride:
                factor = -np.linalg.solve(diagonal[i - stride].T, lower[i].T).T
                next_lower[i] = factor @ lower[i - stride]
                next_diagonal[i] += factor @ upper[i - stride]
                next_rhs[i] += factor @ reduced_rhs[i - stride]
            if i + stride < blocks:
                factor = -np.linalg.solve(diagonal[i + stride].T, upper[i].T).T
                next_upper[i] = factor @ upper[i + stride]
                next_diagonal[i] += factor @ lower[i + stride]
                next_rhs[i] += factor @ reduced_rhs[i + stride]
        lower, diagonal, upper, reduced_rhs = next_lower, next_diagonal, next_upper, next_rhs
        stride *= 2

    result = np.empty_like(reduced_rhs)
    for i in range(blocks):
        result[i] = np.linalg.solve(diagonal[i], reduced_rhs[i])
    return result.reshape(-1)


def _two_level_iterative_sgs(
    sgs: BlockSGS,
    offsets: np.ndarray,
    *,
    coarse_iterations: int,
    coarse_method: str,
):
    rows_per_joint = int(offsets[1] - offsets[0])
    joints = len(offsets) - 1
    prolongation, coarse_joints = _linear_prolongation(joints, rows_per_joint)
    coarse_matrix = prolongation.T @ sgs.matrix @ prolongation
    coarse_offsets = np.arange(coarse_joints + 1, dtype=np.int32) * rows_per_joint

    def update(x: np.ndarray) -> np.ndarray:
        x = sgs.forward(x)
        coarse_rhs = prolongation.T @ (sgs.rhs - sgs.matrix @ x)
        coarse_sgs = BlockSGS(coarse_matrix, coarse_rhs, coarse_offsets)
        if coarse_method == "block_thomas":
            coarse_error = _block_tridiagonal_solve(coarse_matrix, coarse_rhs, rows_per_joint)
        elif coarse_method == "block_pcr":
            coarse_error = _block_pcr_solve(coarse_matrix, coarse_rhs, rows_per_joint)
        elif coarse_method == "sgs":
            coarse_error = np.zeros_like(coarse_rhs)
            for _ in range(coarse_iterations):
                coarse_error = coarse_sgs(coarse_error)
        elif coarse_method == "anderson2":
            coarse_error = _anderson_fixed_point(
                coarse_sgs,
                coarse_matrix,
                coarse_rhs,
                coarse_iterations,
                depth=2,
            )
        else:
            raise ValueError(f"unknown coarse method: {coarse_method}")
        return sgs.backward(x + prolongation @ coarse_error)

    return update


def _multilevel_sgs(
    matrix: np.ndarray,
    rhs: np.ndarray,
    offsets: np.ndarray,
    *,
    coarse_cycles: int = 1,
    smooth_sweeps: int = 1,
    as_preconditioner: bool = False,
):
    rows_per_joint = int(offsets[1] - offsets[0])
    matrices = [matrix]
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    prolongations: list[np.ndarray] = []
    joints = len(offsets) - 1
    while joints > 3:
        level_offsets = np.arange(joints + 1, dtype=np.int32) * rows_per_joint
        lowers.append(_block_triangular(matrices[-1], level_offsets, lower=True))
        uppers.append(_block_triangular(matrices[-1], level_offsets, lower=False))
        prolongation, joints = _linear_prolongation(joints, rows_per_joint)
        prolongations.append(prolongation)
        matrices.append(prolongation.T @ matrices[-1] @ prolongation)

    def cycle(level: int, x: np.ndarray, level_rhs: np.ndarray) -> np.ndarray:
        level_matrix = matrices[level]
        if level == len(prolongations):
            return np.linalg.solve(level_matrix, level_rhs)
        for _ in range(smooth_sweeps):
            x = x + np.linalg.solve(lowers[level], level_rhs - level_matrix @ x)
        residual = level_rhs - level_matrix @ x
        prolongation = prolongations[level]
        coarse_rhs = prolongation.T @ residual
        coarse_error = np.zeros(prolongation.shape[1])
        for _ in range(coarse_cycles):
            coarse_error = cycle(level + 1, coarse_error, coarse_rhs)
        x = x + prolongation @ coarse_error
        for _ in range(smooth_sweeps):
            x = x + np.linalg.solve(uppers[level], level_rhs - level_matrix @ x)
        return x

    if as_preconditioner:
        return lambda residual: cycle(0, np.zeros_like(residual), residual)
    return lambda x: cycle(0, x, rhs)


def run(args: argparse.Namespace) -> dict[str, object]:
    matrix, rhs, offsets = _build_system(wp.get_device(args.device), args.dt)
    sgs = BlockSGS(matrix, rhs, offsets)
    iteration_matrix = sgs.iteration_matrix()
    rho = float(np.max(np.abs(np.linalg.eigvals(iteration_matrix))))
    results = [
        _iterate("block_sgs", sgs, matrix, rhs, args.iterations),
        _chebyshev(sgs, min(rho, 0.999999), args.iterations),
        _anderson(sgs, 2, args.iterations),
        _anderson(sgs, 4, args.iterations),
        _pcg(sgs, "none", args.iterations),
        _pcg(sgs, "block_jacobi", args.iterations),
        _pcg(sgs, "sgs", args.iterations),
        _pcg(
            sgs,
            "multilevel_smooth2",
            args.iterations,
            _multilevel_sgs(matrix, rhs, offsets, smooth_sweeps=2, as_preconditioner=True),
        ),
        _iterate("multilevel_v", _multilevel_sgs(matrix, rhs, offsets), matrix, rhs, args.iterations),
        _iterate(
            "multilevel_w2",
            _multilevel_sgs(matrix, rhs, offsets, coarse_cycles=2),
            matrix,
            rhs,
            args.iterations,
        ),
        _iterate(
            "multilevel_smooth2",
            _multilevel_sgs(matrix, rhs, offsets, smooth_sweeps=2),
            matrix,
            rhs,
            args.iterations,
        ),
    ]
    for aggregate in (2, 4, 8, 16):
        results.append(
            _iterate(
                f"two_level_{aggregate}",
                _coarse_sgs(sgs, offsets, aggregate),
                matrix,
                rhs,
                args.iterations,
            )
        )
    for coarse_iterations in (1, 2, 4, 8):
        results.append(
            _iterate(
                f"two_level_iterative_sgs_{coarse_iterations}",
                _two_level_iterative_sgs(
                    sgs,
                    offsets,
                    coarse_iterations=coarse_iterations,
                    coarse_method="sgs",
                ),
                matrix,
                rhs,
                args.iterations,
            )
        )
    results.append(
        _iterate(
            "two_level_block_thomas",
            _two_level_iterative_sgs(
                sgs,
                offsets,
                coarse_iterations=1,
                coarse_method="block_thomas",
            ),
            matrix,
            rhs,
            args.iterations,
        )
    )
    results.append(
        _iterate(
            "two_level_block_pcr",
            _two_level_iterative_sgs(
                sgs,
                offsets,
                coarse_iterations=1,
                coarse_method="block_pcr",
            ),
            matrix,
            rhs,
            args.iterations,
        )
    )
    for coarse_iterations in (2, 4, 8):
        results.append(
            _iterate(
                f"two_level_iterative_anderson2_{coarse_iterations}",
                _two_level_iterative_sgs(
                    sgs,
                    offsets,
                    coarse_iterations=coarse_iterations,
                    coarse_method="anderson2",
                ),
                matrix,
                rhs,
                args.iterations,
            )
        )
    return {
        "rows": matrix.shape[0],
        "joint_blocks": len(offsets) - 1,
        "sgs_spectral_radius": rho,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=1.0 / 6000.0)
    parser.add_argument("--iterations", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
