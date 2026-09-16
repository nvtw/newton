"""Explicit common-world-pivot rounding reform, frozen physical input retained."""

import hashlib
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import solve
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.reference_high_mass_connected_normals import active_set, tangent_step


def main(result_prefix="/tmp/high_mass_consistent", cases=("244_0", "245_1"), allow_revisits=False):
    reports = {}
    for case in cases:
        path = Path("/tmp/high_mass_window120_live_" + case + ".npz")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        original = np.load(path)
        raw = np.load("/tmp/high_mass_window120_live_before_" + case + ".npz")
        points = original["selected_points"]
        h = raw["headers"].view(np.int32)
        J = np.zeros_like(original["J"])
        shifts = []
        for index, p in enumerate(points):
            col = next(c for c in range(int(raw["column_count"][0])) if h[5, c] <= p < h[5, c] + h[6, c])
            a, b = map(int, h[1:3, col])
            assert raw["headers"][3, col] == raw["headers"][4, col] == 0.5
            pa, pb = raw["positions"][[a, b]].astype(float)
            wa = pa + raw["derived"][9:12, p].astype(float)
            wb = pb + raw["derived"][12:15, p].astype(float)
            pivot = (wa + wb) * 0.5
            shifts.append(np.linalg.norm(pivot - wa))
            axes = original["J"][3 * index : 3 * index + 3, 6 * b : 6 * b + 3]
            J[3 * index : 3 * index + 3, 6 * a : 6 * a + 6] = -np.concatenate(
                (axes, np.cross(pivot - pa, axes)), axis=1
            )
            J[3 * index : 3 * index + 3, 6 * b : 6 * b + 6] = np.concatenate((axes, np.cross(pivot - pb, axes)), axis=1)
        W = original["inverse_mass"]
        initial = original["initial"]
        u = np.concatenate((raw["velocity"], raw["angular_velocity"]), axis=1).astype(float).ravel()
        A = J @ W @ J.T
        rhs = J @ u - A @ initial
        d = {
            "A": A,
            "J": J,
            "inverse_mass": W,
            "initial": initial,
            "rhs": rhs,
            "normal_regularization": np.zeros(len(points)),
        }
        seed = initial.copy()
        n = np.arange(0, len(initial), 3)
        t = np.setdiff1d(np.arange(len(initial)), n)
        for _ in range(8):
            seed[n] = active_set(A[np.ix_(n, n)], rhs[n] + A[np.ix_(n, t)] @ seed[t])
            for k in range(len(points)):
                ids = np.array([3 * k + 1, 3 * k + 2])
                K = A[np.ix_(ids, ids)]
                seed[ids] = tangent_step(K, A[ids] @ seed + rhs[ids], seed[ids], 0.5 * seed[3 * k])
        value, attempts = solve(d, seed=seed, allow_revisits=allow_revisits)
        delta = J.T @ (value - initial)
        v = u + W @ delta
        M = np.zeros_like(W)
        moving = raw["inverse_mass"] > 0
        for b in np.flatnonzero(moving):
            M[6 * b : 6 * b + 6, 6 * b : 6 * b + 6] = np.linalg.inv(W[6 * b : 6 * b + 6, 6 * b : 6 * b + 6])
        reaction = delta.reshape(-1, 6).copy()
        reaction[moving] = (M @ (v - u)).reshape(-1, 6)[moving]
        linear = reaction[:, :3].sum(0)
        angular = (reaction[:, 3:] + np.cross(raw["positions"], reaction[:, :3])).sum(0)
        energy = float(0.5 * (v @ M @ v - u @ M @ u))
        work = float(delta @ ((u + v) * 0.5))
        residual = float(np.max(np.abs(natural_map_evaluator(A, rhs, np.zeros(len(points)), 0.5)[0](value)[0])))
        physical = {
            "energy_change_J": energy,
            "work_error_J": energy - work,
            "linear_balance": linear.tolist(),
            "angular_balance": angular.tolist(),
        }
        success = any(a["accepted"] for a in attempts)
        if success:
            assert residual < 1e-8 and abs(energy - work) < 1e-10
            assert np.max(np.abs(linear)) < 1e-10 and np.max(np.abs(angular)) < 1e-10
        reports[case] = {
            "accepted": success,
            "original_input_sha256": digest,
            "max_pivot_shift_m": float(max(shifts)),
            "J_change": float(np.max(np.abs(J - original["J"]))),
            "rhs_change": float(np.max(np.abs(rhs - original["rhs"]))),
            "residual": residual,
            "physical": physical,
            "attempts": attempts,
            "seed_normal_blocks": 8,
            "seed_normal_masks": 8 * 2 ** len(points),
            "seed_tangent_updates": 8 * len(points),
            "linear_solves": sum("linear_defect" in h for a in attempts for h in a["history"]),
            "factorizations": sum("rank" in h for a in attempts for h in a["history"]),
            "residual_evaluations": sum(a["history"][-1]["total_residual_evaluations"] for a in attempts),
        }
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        np.savez(result_prefix + "_" + case + ".npz", **d, seed=seed, solution=value, u=u, v=v)
        print(case, {k: v for k, v in reports[case].items() if k != "attempts"}, flush=True)
    Path(result_prefix + "_replay.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
