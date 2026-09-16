"""Eight original-law frozen sweeps as numerical seeds, never live impulses."""

import hashlib
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import solve
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.reference_high_mass_connected_normals import active_set, tangent_step


def main():
    report = {}
    for frame in ("244_0", "245_1"):
        path = Path("/tmp/high_mass_window120_live_" + frame + ".npz")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        d = np.load(path)
        A, rhs = d["A"], d["rhs"]
        initial = d["initial"]
        normal = np.arange(0, len(initial), 3)
        tangent = np.setdiff1d(np.arange(len(initial)), normal)
        evaluate = natural_map_evaluator(A, rhs, np.zeros(len(normal)), 0.5)[0]
        for method in ("scalar8", "normal8"):
            seed = initial.copy()
            bisections = 0
            for _ in range(8):
                if method == "normal8":
                    seed[normal] = active_set(
                        A[np.ix_(normal, normal)], rhs[normal] + A[np.ix_(normal, tangent)] @ seed[tangent]
                    )
                for k in range(len(normal)):
                    n = 3 * k
                    if method == "scalar8":
                        seed[n] = max(0, seed[n] - (A[n] @ seed + rhs[n]) / A[n, n])
                    ids = np.array([n + 1, n + 2])
                    K = A[np.ix_(ids, ids)]
                    gradient = A[ids] @ seed + rhs[ids]
                    radius = 0.5 * seed[n]
                    if radius > 0 and np.linalg.norm(np.linalg.solve(K, K @ seed[ids] - gradient)) > radius:
                        bisections += 60
                    seed[ids] = tangent_step(K, gradient, seed[ids], radius)
            value, attempts = solve(d, seed=seed)
            key = frame + "_" + method
            report[key] = {
                "accepted": any(a["accepted"] for a in attempts),
                "input_residual": float(np.max(np.abs(evaluate(initial)[0]))),
                "seed_residual": float(np.max(np.abs(evaluate(seed)[0]))),
                "best_attempt_residual": min(a["residual"] for a in attempts),
                "scalar_normal_updates": 8 * len(normal) if method == "scalar8" else 0,
                "connected_normal_solves": 8 if method == "normal8" else 0,
                "normal_masks": 8 * 2 ** len(normal) if method == "normal8" else 0,
                "metric_tangent_updates": 8 * len(normal),
                "tangent_bisections": bisections,
                "newton_solves": sum("linear_defect" in h for a in attempts for h in a["history"]),
                "factorizations": sum("rank" in h for a in attempts for h in a["history"]),
                "residual_evaluations": sum(a["history"][-1]["total_residual_evaluations"] for a in attempts),
                "input_sha256": digest,
                "attempts": attempts,
            }
            np.savez("/tmp/high_mass_bounded_seed_" + key + ".npz", seed=seed, solution=value)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            print(key, {k: v for k, v in report[key].items() if k != "attempts"}, flush=True)
    Path("/tmp/high_mass_bounded_seeds.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
