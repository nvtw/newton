"""Convex impulse feasibility in certified zero-response gauges, fixed velocity."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import response_factor
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def main():
    from scipy.optimize import minimize

    d = np.load("/tmp/high_mass_consistent960_362_5.npz")
    geometry = np.load("/tmp/high_mass_gauge_support_geometry.npz")
    J = geometry["J_exact_rounded"]
    W = d["inverse_mass"]
    initial = d["solution"]
    A = J @ W @ J.T
    rhs = J @ d["u"] - A @ d["initial"]
    F = response_factor(J, W)
    N = geometry["N"]
    # Basis comes from exact-rational dynamic J.T.nullspace(), rank 11.
    # No numerical singular threshold selects a physical mode.
    singular = np.linalg.svd(F, compute_uv=False)
    gauge_defect = float(np.max(np.abs(F.T @ N)))
    gamma = F.shape[0] * np.finfo(float).eps / (1 - F.shape[0] * np.finfo(float).eps)
    gauge_bound = 8 * gamma * float(np.linalg.norm(F.T, ord=np.inf) * np.linalg.norm(N, ord=np.inf))
    assert gauge_defect < gauge_bound
    gradient = A @ initial + rhs
    triples = initial.reshape(-1, 3)
    g = gradient.reshape(-1, 3)
    assert min(g[:, 0]) >= -1e-8, "Gauge corrections cannot fix closing velocity"
    # Values below the existing original-law residual tolerance remain
    # explicitly reported and are rechecked by that unchanged full map.
    sliding = np.flatnonzero(np.linalg.norm(g[:, 1:], axis=1) > 1e-8)
    separating = np.flatnonzero(g[:, 0] > 1e-8)
    E = []
    for k in sliding:
        direction = g[k, 1:] / np.linalg.norm(g[k, 1:])
        for t in range(2):
            row = np.zeros(len(initial))
            row[3 * k] = 0.5 * direction[t]
            row[3 * k + 1 + t] = 1
            E.append(row)
    for k in separating:
        E.extend(np.eye(len(initial))[3 * k : 3 * k + 3])
    E = np.array(E).reshape(-1, len(initial))

    def inequalities(z):
        value = (initial + N @ z).reshape(-1, 3)
        return np.r_[value[:, 0], 0.5 * value[:, 0] - np.linalg.norm(value[:, 1:], axis=1)]

    def inequality_jac(z):
        value = (initial + N @ z).reshape(-1, 3)
        rows = np.zeros((2 * len(value), len(initial)))
        for k in range(len(value)):
            rows[k, 3 * k] = 1
            rows[len(value) + k, 3 * k] = 0.5
            rows[len(value) + k, 3 * k + 1 : 3 * k + 3] = -value[k, 1:] / max(np.linalg.norm(value[k, 1:]), 1e-300)
        return rows @ N

    constraints = [{"type": "ineq", "fun": inequalities, "jac": inequality_jac}]
    if len(E):
        constraints.append({"type": "eq", "fun": lambda z: E @ (initial + N @ z), "jac": lambda z: E @ N})
    solved = minimize(
        lambda z: 0.5 * z @ z,
        np.zeros(N.shape[1]),
        jac=lambda z: z,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 100, "ftol": 1e-14},
    )
    value = initial + N @ solved.x
    response_delta = F.T @ (value - initial)
    physical_delta = W @ J.T @ (value - initial)
    evaluate = natural_map_evaluator(A, rhs, np.zeros(len(triples)), 0.5)[0]
    error = float(np.max(np.abs(evaluate(value)[0])))
    disk = float(np.min(inequalities(solved.x)))
    work_before = float(initial @ rhs + 0.5 * initial @ A @ initial)
    work_after = float(value @ rhs + 0.5 * value @ A @ value)
    accepted = (
        error < 1e-8
        and disk >= -1e-8
        and np.max(np.abs(physical_delta)) < 1e-10
        and abs(work_after - work_before) < 1e-10
    )
    result = {
        "accepted": bool(accepted),
        "optimizer_success": bool(solved.success),
        "message": solved.message,
        "iterations": solved.nit,
        "function_evaluations": solved.nfev,
        "gauge_dimension": N.shape[1],
        "rounded_factor_singular_fp64": [float(v) for v in singular],
        "gauge_defect": gauge_defect,
        "gauge_bound": gauge_bound,
        "sliding_points": sliding.tolist(),
        "separating_points": separating.tolist(),
        "original_normal_velocities": g[:, 0].tolist(),
        "original_slip_speeds": np.linalg.norm(g[:, 1:], axis=1).tolist(),
        "before_residual": float(np.max(np.abs(evaluate(initial)[0]))),
        "after_residual": error,
        "min_normal_or_disk_margin_Ns": disk,
        "physical_velocity_change": float(np.max(np.abs(physical_delta))),
        "response_change": float(np.max(np.abs(response_delta))),
        "work_change_J": work_after - work_before,
    }
    Path("/tmp/high_mass_gauge_feasibility_exact.json").write_text(json.dumps(result, indent=2))
    np.savez(
        "/tmp/high_mass_gauge_feasibility_exact.npz",
        solution=value,
        initial=initial,
        N=N,
        F=F,
        gauge_coordinates=solved.x,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
