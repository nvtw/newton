# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare DVI block-coupling convergence on problems with exact solutions.

The benchmark constructs strictly convex complementarity problems

    v = D lambda + b,
    v_b = 0,
    0 <= lambda_u perpendicular to v_u >= 0,

with a known exact KKT solution. ``D`` is assembled from a positive-definite
bilateral block, a positive-definite reduced Schur operator, and a controlled
bilateral/unilateral coupling. This guarantees that the full operator is
positive definite and that solution error is unambiguous.

Both methods receive the same number of projected scalar sweeps. Reported KKT
residuals are evaluated against the original full system, never against the
reduced operator used by the Schur method.

Run from the repository root::

    uv run scripts/evaluate_dvi_block_convergence.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ComplementarityProblem:
    """Strictly convex block complementarity problem with a known solution."""

    matrix: np.ndarray
    bias: np.ndarray
    bilateral_count: int
    exact_impulse: np.ndarray
    exact_velocity: np.ndarray


def _spd_matrix(rng: np.random.Generator, size: int, condition: float) -> np.ndarray:
    """Construct a symmetric positive-definite matrix with known condition."""
    orthogonal, _ = np.linalg.qr(rng.normal(size=(size, size)))
    eigenvalues = np.geomspace(1.0, condition, size)
    return (orthogonal * eigenvalues) @ orthogonal.T


def make_problem(
    *,
    seed: int,
    bilateral_count: int,
    unilateral_count: int,
    coupling: float,
    condition: float,
) -> ComplementarityProblem:
    """Construct a block problem whose exact KKT point is prescribed."""
    rng = np.random.default_rng(seed)
    d_bb = _spd_matrix(rng, bilateral_count, condition)
    schur = _spd_matrix(rng, unilateral_count, condition)
    response = coupling * rng.normal(size=(bilateral_count, unilateral_count)) / np.sqrt(unilateral_count)
    d_bu = d_bb @ response
    d_uu = schur + response.T @ d_bb @ response
    matrix = np.block([[d_bb, d_bu], [d_bu.T, d_uu]])

    exact_bilateral = rng.normal(size=bilateral_count)
    exact_unilateral = np.zeros(unilateral_count)
    active = np.arange(unilateral_count) % 2 == 0
    exact_unilateral[active] = rng.uniform(0.1, 1.0, np.count_nonzero(active))
    exact_velocity = np.zeros(bilateral_count + unilateral_count)
    exact_velocity[bilateral_count:][~active] = rng.uniform(0.1, 1.0, np.count_nonzero(~active))
    exact_impulse = np.concatenate((exact_bilateral, exact_unilateral))
    bias = exact_velocity - matrix @ exact_impulse
    return ComplementarityProblem(matrix, bias, bilateral_count, exact_impulse, exact_velocity)


def _projected_sweeps(
    matrix: np.ndarray,
    bias: np.ndarray,
    impulse: np.ndarray,
    sweeps: int,
    omega: float,
) -> None:
    """Apply in-place projected Gauss-Seidel sweeps to nonnegative rows."""
    for _ in range(sweeps):
        for row in range(impulse.size):
            velocity = float(matrix[row] @ impulse + bias[row])
            impulse[row] = max(0.0, impulse[row] - omega * velocity / matrix[row, row])


def solve_schur(problem: ComplementarityProblem, sweeps: int, omega: float) -> np.ndarray:
    """Eliminate bilateral rows and project the reduced system."""
    nb = problem.bilateral_count
    d_bb = problem.matrix[:nb, :nb]
    d_bu = problem.matrix[:nb, nb:]
    d_ub = problem.matrix[nb:, :nb]
    d_uu = problem.matrix[nb:, nb:]
    b_b = problem.bias[:nb]
    b_u = problem.bias[nb:]

    response = np.linalg.solve(d_bb, d_bu)
    schur = d_uu - d_ub @ response
    reduced_bias = b_u - d_ub @ np.linalg.solve(d_bb, b_b)
    impulse_u = np.zeros(d_uu.shape[0])
    _projected_sweeps(schur, reduced_bias, impulse_u, sweeps, omega)
    impulse_b = -np.linalg.solve(d_bb, b_b + d_bu @ impulse_u)
    return np.concatenate((impulse_b, impulse_u))


def solve_alternating(
    problem: ComplementarityProblem,
    outer_iterations: int,
    sweeps_per_iteration: int,
    omega: float,
) -> np.ndarray:
    """Alternate projected unilateral sweeps and factored bilateral solves."""
    nb = problem.bilateral_count
    d_bb = problem.matrix[:nb, :nb]
    d_bu = problem.matrix[:nb, nb:]
    d_uu = problem.matrix[nb:, nb:]
    b_b = problem.bias[:nb]
    b_u = problem.bias[nb:]
    impulse_u = np.zeros(d_uu.shape[0])

    for _ in range(outer_iterations):
        impulse_b = -np.linalg.solve(d_bb, b_b + d_bu @ impulse_u)
        unilateral_bias = b_u + problem.matrix[nb:, :nb] @ impulse_b
        _projected_sweeps(d_uu, unilateral_bias, impulse_u, sweeps_per_iteration, omega)
    impulse_b = -np.linalg.solve(d_bb, b_b + d_bu @ impulse_u)
    return np.concatenate((impulse_b, impulse_u))


def evaluate(problem: ComplementarityProblem, impulse: np.ndarray) -> dict[str, float]:
    """Evaluate solution error and full-system KKT residuals."""
    nb = problem.bilateral_count
    velocity = problem.matrix @ impulse + problem.bias
    impulse_u = impulse[nb:]
    velocity_u = velocity[nb:]
    natural_map = impulse_u - np.maximum(0.0, impulse_u - velocity_u)
    scale = max(float(np.linalg.norm(problem.exact_impulse)), np.finfo(np.float64).eps)
    velocity_scale = max(float(np.linalg.norm(problem.exact_velocity)), np.finfo(np.float64).eps)
    return {
        "relative_impulse_l2_error": float(np.linalg.norm(impulse - problem.exact_impulse) / scale),
        "relative_velocity_l2_error": float(np.linalg.norm(velocity - problem.exact_velocity) / velocity_scale),
        "natural_map_inf": float(np.linalg.norm(natural_map, ord=np.inf)),
        "bilateral_velocity_inf": float(np.linalg.norm(velocity[:nb], ord=np.inf)),
        "primal_feasibility_inf": float(np.max(np.maximum(-impulse_u, 0.0))),
        "dual_feasibility_inf": float(np.max(np.maximum(-velocity_u, 0.0))),
        "complementarity_inf": float(np.linalg.norm(impulse_u * velocity_u, ord=np.inf)),
    }


def run(args: argparse.Namespace) -> dict:
    """Generate deterministic convergence curves for all requested couplings."""
    if args.bilateral_count <= 0 or args.unilateral_count <= 0:
        raise ValueError("Block dimensions must be positive")
    if args.condition < 1.0:
        raise ValueError("--condition must be at least one")
    if not 0.0 < args.omega < 2.0:
        raise ValueError("--omega must lie in (0, 2)")
    if args.sweeps_per_iteration <= 0 or any(value <= 0 for value in args.outer_iterations):
        raise ValueError("Iteration and sweep counts must be positive")

    results = {
        "problem": {
            "seed": args.seed,
            "bilateral_count": args.bilateral_count,
            "unilateral_count": args.unilateral_count,
            "condition": args.condition,
            "omega": args.omega,
            "sweeps_per_iteration": args.sweeps_per_iteration,
        },
        "cases": {},
    }
    for coupling in args.coupling:
        problem = make_problem(
            seed=args.seed,
            bilateral_count=args.bilateral_count,
            unilateral_count=args.unilateral_count,
            coupling=coupling,
            condition=args.condition,
        )
        exact_metrics = evaluate(problem, problem.exact_impulse)
        if max(exact_metrics.values()) > 1.0e-10:
            raise RuntimeError(f"Constructed exact solution failed its KKT check: {exact_metrics}")

        curves = {"schur": {}, "alternating": {}}
        for outer_iterations in args.outer_iterations:
            total_sweeps = outer_iterations * args.sweeps_per_iteration
            schur_impulse = solve_schur(problem, total_sweeps, args.omega)
            alternating_impulse = solve_alternating(
                problem,
                outer_iterations,
                args.sweeps_per_iteration,
                args.omega,
            )
            curves["schur"][str(total_sweeps)] = evaluate(problem, schur_impulse)
            curves["alternating"][str(total_sweeps)] = evaluate(problem, alternating_impulse)

        results["cases"][str(coupling)] = {
            "full_matrix_condition": float(np.linalg.cond(problem.matrix)),
            "exact_kkt": exact_metrics,
            "curves_by_total_projected_sweeps": curves,
        }
    return results


def plot_results(results: dict, output: Path) -> None:
    """Plot full-system natural-map convergence for every coupling."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(results["cases"]), figsize=(10.5, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (coupling, case) in zip(axes, results["cases"].items(), strict=True):
        curves = case["curves_by_total_projected_sweeps"]
        for method, style in (("schur", "o-"), ("alternating", "s--")):
            sweeps = np.asarray([int(value) for value in curves[method]])
            residual = np.asarray([entry["natural_map_inf"] for entry in curves[method].values()])
            axis.semilogy(sweeps, residual, style, label=method.capitalize(), linewidth=1.8, markersize=4)
        axis.set_title(rf"Coupling $\alpha={coupling}$")
        axis.set_xlabel("Projected sweeps")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"Full-system natural-map residual $\|R(\lambda)\|_\infty$")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse convergence benchmark arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4125)
    parser.add_argument("--bilateral-count", type=int, default=24)
    parser.add_argument("--unilateral-count", type=int, default=32)
    parser.add_argument("--condition", type=float, default=100.0)
    parser.add_argument("--coupling", nargs="+", type=float, default=[0.25, 1.0, 2.0])
    parser.add_argument("--outer-iterations", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--sweeps-per-iteration", type=int, default=2)
    parser.add_argument("--omega", type=float, default=1.2)
    parser.add_argument("--output", type=Path, help="Write JSON results to this path")
    parser.add_argument("--plot", type=Path, help="Write a convergence plot to this path")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    benchmark_results = run(arguments)
    serialized = json.dumps(benchmark_results, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    if arguments.plot:
        plot_results(benchmark_results, arguments.plot)
    print(serialized)
