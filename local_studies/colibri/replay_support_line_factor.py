"""Bounded response solve with an exactly certified support-line free twist."""

import hashlib
import json
from pathlib import Path

import numpy as np

from local_studies.colibri import bounded_island_coulomb as bounded
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.reference_high_mass_connected_normals import active_set, tangent_step


def main(certified=True):
    path = Path("/tmp/high_mass_window120_live_362_5.npz")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = np.load(path)
    old = np.load("/tmp/high_mass_consistent960_362_5.npz")
    geometry = np.load("/tmp/high_mass_gauge_support_geometry.npz")
    certificate = json.loads(Path("/tmp/high_mass_gauge_support_geometry.json").read_text())
    assert certificate["exact_rational_dynamic_rank"] == 11
    raw = np.load("/tmp/high_mass_window120_live_before_362_5.npz")
    J = geometry["J_exact_rounded"]
    L = geometry["L"]
    W = original["inverse_mass"]
    initial = original["initial"]
    u = old["u"]
    free = np.linalg.solve(L[6:18], geometry["twist"][6:18])
    free /= np.linalg.norm(free)
    Q = np.linalg.qr(free[:, None], mode="complete")[0][:, 1:]
    F = J @ L @ Q if certified else J @ L
    A = J @ W @ J.T
    rhs = J @ u - A @ initial
    d = {
        "A": A,
        "J": J,
        "inverse_mass": W,
        "rhs": rhs,
        "initial": initial,
        "normal_regularization": original["normal_regularization"],
    }
    assert np.all(d["normal_regularization"] == 0)
    seed = initial.copy()
    n = np.arange(0, len(initial), 3)
    t = np.setdiff1d(np.arange(len(initial)), n)
    for _ in range(8):
        seed[n] = active_set(A[np.ix_(n, n)], rhs[n] + A[np.ix_(n, t)] @ seed[t])
        for k in range(len(n)):
            ids = np.array([3 * k + 1, 3 * k + 2])
            seed[ids] = tangent_step(A[np.ix_(ids, ids)], A[ids] @ seed + rhs[ids], seed[ids], 0.5 * seed[3 * k])
    native = bounded.response_factor
    bounded.response_factor = lambda supplied_J, supplied_W: F
    try:
        value, attempts = bounded.solve(d, seed=seed)
    finally:
        bounded.response_factor = native
    delta = J.T @ (value - initial)
    v = u + W @ delta
    M = np.zeros_like(W)
    for b in (1, 2):
        M[6 * b : 6 * b + 6, 6 * b : 6 * b + 6] = np.linalg.inv(W[6 * b : 6 * b + 6, 6 * b : 6 * b + 6])
    reaction = delta.reshape(-1, 6).copy()
    reaction[1:] = (M @ (v - u)).reshape(-1, 6)[1:]
    linear = reaction[:, :3].sum(0)
    angular = (reaction[:, 3:] + np.cross(raw["positions"], reaction[:, :3])).sum(0)
    energy = float(0.5 * (v @ M @ v - u @ M @ u))
    work = float(delta @ ((u + v) * 0.5))
    residual = float(np.max(np.abs(natural_map_evaluator(A, rhs, np.zeros(len(n)), 0.5)[0](value)[0])))
    accepted = any(a["accepted"] for a in attempts)
    free_change = float(free @ np.linalg.solve(L[6:18], (v - u)[6:18]))
    if accepted:
        assert (
            residual < 1e-8
            and max(np.max(np.abs(linear)), np.max(np.abs(angular)), abs(energy - work), abs(free_change)) < 1e-10
        )
    report = {
        "accepted": accepted,
        "residual": residual,
        "best_attempt_residual": min(a["residual"] for a in attempts),
        "attempts": attempts,
        "factor_columns": F.shape[1],
        "factor_covariance_error": float(np.max(np.abs(F @ F.T - A))),
        "free_response_defect": float(np.max(np.abs(J @ L @ free))),
        "free_velocity_component_change": free_change,
        "physical": {
            "linear_balance": linear.tolist(),
            "angular_balance": angular.tolist(),
            "energy_change_J": energy,
            "work_error_J": energy - work,
        },
        "seed_normal_blocks": 8,
        "seed_normal_masks": 8 * 2 ** len(n),
        "seed_tangent_updates": 8 * len(n),
        "linear_solves": sum("linear_defect" in h for a in attempts for h in a["history"]),
        "factorizations": sum("rank" in h for a in attempts for h in a["history"]),
        "residual_evaluations": sum(a["history"][-1]["total_residual_evaluations"] for a in attempts),
        "original_input_sha256": digest,
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    Path(
        "/tmp/high_mass_support_line_solve.json" if certified else "/tmp/high_mass_support_line_control.json"
    ).write_text(json.dumps(report, indent=2))
    np.savez(
        "/tmp/high_mass_support_line_solve.npz" if certified else "/tmp/high_mass_support_line_control.npz",
        **d,
        F=F,
        Q=Q,
        free=free,
        u=u,
        v=v,
        seed=seed,
        solution=value,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
