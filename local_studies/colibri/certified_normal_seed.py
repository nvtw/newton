"""Objective-roundoff ties for numerical normal seeds; no live impulse update."""

import itertools
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import response_factor, solve
from local_studies.colibri.reference_high_mass_connected_normals import active_set, tangent_step


def tied_active_set(matrix, rhs, factor):
    n = len(rhs)
    candidates = []
    eps = np.finfo(float).eps
    gamma = (3 * n + 3) * eps / (1 - (3 * n + 3) * eps)
    for bits in itertools.product((False, True), repeat=n):
        active = np.flatnonzero(bits)
        value = np.zeros(n)
        if len(active):
            try:
                value[active] = np.linalg.solve(matrix[np.ix_(active, active)], -rhs[active])
            except np.linalg.LinAlgError:
                continue
        # Preserve the existing seed feasibility gate verbatim.
        if np.min(value) < -1e-9 or np.min(matrix @ value + rhs) < -1e-8:
            continue
        objective = float(0.5 * value @ matrix @ value + rhs @ value)
        error = float(gamma * (0.5 * np.abs(value) @ np.abs(matrix) @ np.abs(value) + np.abs(rhs) @ np.abs(value)))
        candidates.append((objective, error, value, bits))
    if not candidates:
        raise RuntimeError("No feasible active set found")
    best = min(candidates, key=lambda c: c[0])
    ties = []
    excluded = 0
    max_response = 0.0
    for candidate in candidates:
        if abs(candidate[0] - best[0]) > candidate[1] + best[1]:
            continue
        delta = candidate[2] - best[2]
        response = float(np.max(np.abs(factor.T @ delta)))
        # Bound matrix products/subtraction against the magnitude of the two
        # original impulse responses, without a physical rank cutoff.
        bound = float(8 * gamma * np.max(np.abs(factor.T) @ (np.abs(candidate[2]) + np.abs(best[2]))))
        if response > bound:
            excluded += 1
            continue
        max_response = max(max_response, response)
        ties.append(candidate)
    selected = min(ties, key=lambda c: (float(c[2] @ c[2]), c[3]))
    return selected[2], {
        "feasible": len(candidates),
        "ties": len(ties),
        "nonneutral_objective_ties": excluded,
        "objective_error_bound": max(c[1] for c in ties),
        "max_tied_response": max_response,
        "selected_norm": float(np.linalg.norm(selected[2])),
        "strict_norm": float(np.linalg.norm(best[2])),
    }


def main(cases=("362_5", "362_5_exact", "263_2", "300_2", "244_0", "245_1"), prefix="/tmp/high_mass_tied_seed"):
    reports = {}
    for case in cases:
        source = case.replace("_exact", "")
        old = np.load("/tmp/high_mass_consistent960_" + source + ".npz")
        d = {k: old[k].copy() for k in ("A", "J", "inverse_mass", "rhs", "initial", "normal_regularization")}
        if case.endswith("_exact"):
            d["J"] = np.load("/tmp/high_mass_gauge_support_geometry.npz")["J_exact_rounded"]
            d["A"] = d["J"] @ d["inverse_mass"] @ d["J"].T
            d["rhs"] = d["J"] @ old["u"] - d["A"] @ d["initial"]
        A = d["A"]
        rhs = d["rhs"]
        n = np.arange(0, len(rhs), 3)
        t = np.setdiff1d(np.arange(len(rhs)), n)
        F = response_factor(d["J"], d["inverse_mass"])
        seed = d["initial"].copy()
        strict = seed.copy()
        logs = []
        for _ in range(8):
            seed[n], log = tied_active_set(A[np.ix_(n, n)], rhs[n] + A[np.ix_(n, t)] @ seed[t], F[n])
            logs.append(log)
            strict[n] = active_set(A[np.ix_(n, n)], rhs[n] + A[np.ix_(n, t)] @ strict[t])
            for k in range(len(n)):
                ids = np.array([3 * k + 1, 3 * k + 2])
                K = A[np.ix_(ids, ids)]
                for x in (seed, strict):
                    x[ids] = tangent_step(K, A[ids] @ x + rhs[ids], x[ids], 0.5 * x[3 * k])
        value, attempts = solve(d, seed=seed)
        report = {
            "accepted": any(a["accepted"] for a in attempts),
            "residual": attempts[-1]["residual"]
            if any(a["accepted"] for a in attempts)
            else min(a["residual"] for a in attempts),
            "seed_change": float(np.max(np.abs(seed - strict))),
            "seed_response_change": float(np.max(np.abs(F.T @ (seed - strict)))),
            "tie_sweeps": logs,
            "attempts": attempts,
            "linear_solves": sum("linear_defect" in h for a in attempts for h in a["history"]),
            "residual_evaluations": sum(a["history"][-1]["total_residual_evaluations"] for a in attempts),
        }
        raw = np.load("/tmp/high_mass_window120_live_before_" + source + ".npz")
        J, W, u = d["J"], d["inverse_mass"], old["u"]
        delta = J.T @ (value - d["initial"])
        v = u + W @ delta
        M = np.zeros_like(W)
        for b in (1, 2):
            M[6 * b : 6 * b + 6, 6 * b : 6 * b + 6] = np.linalg.inv(W[6 * b : 6 * b + 6, 6 * b : 6 * b + 6])
        reaction = delta.reshape(-1, 6).copy()
        reaction[1:] = (M @ (v - u)).reshape(-1, 6)[1:]
        linear = reaction[:, :3].sum(0)
        angular = (reaction[:, 3:] + np.cross(raw["positions"], reaction[:, :3])).sum(0)
        energy = 0.5 * (v @ M @ v - u @ M @ u)
        work = delta @ ((u + v) * 0.5)
        report["physical"] = {
            "linear_balance_max": float(np.max(np.abs(linear))),
            "angular_balance_max": float(np.max(np.abs(angular))),
            "energy_change_J": float(energy),
            "work_error_J": float(energy - work),
        }
        report["seed_normal_masks"] = 8 * 2 ** len(n)
        report["seed_tangent_updates"] = 8 * len(n)
        report["control_seed_masks"] = 8 * 2 ** len(n)
        report["factorizations"] = sum("rank" in h for a in attempts for h in a["history"])
        if report["accepted"]:
            assert max(np.max(np.abs(linear)), np.max(np.abs(angular)), abs(energy - work)) < 1e-10
        reports[case] = report
        np.savez(prefix + "_" + case + ".npz", **d, seed=seed, strict_seed=strict, solution=value)
        print(case, {k: v for k, v in report.items() if k not in ("tie_sweeps", "attempts")}, flush=True)
        Path(prefix + "_replay.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
