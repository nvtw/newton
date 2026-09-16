"""Original cheap natural solve, bounded numerical continuation only on failure."""

import hashlib
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt, response_factor, solve


def main(cases=("263_2", "300_2", "245_1", "244_0", "362_5"), prefix="/tmp/high_mass_adaptive_proximal"):
    reports = {}
    for case in cases:
        path = Path("/tmp/high_mass_consistent960_" + case + ".npz")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = np.load(path)
        d = {k: saved[k] for k in ("A", "J", "inverse_mass", "rhs", "initial", "normal_regularization")}
        F = response_factor(d["J"], d["inverse_mass"])
        strict = saved["seed"].copy()
        value, first = solve(d, seed=strict, max_face_visits=0)
        stages = []
        final = []
        if not any(a["accepted"] for a in first):
            for epsilon in (0.1, 0.01):
                value, history = attempt(
                    F,
                    d["rhs"],
                    value,
                    np.full(len(value) // 3, 0.5),
                    steps=8,
                    numerical_epsilon=epsilon,
                    numerical_reference=strict,
                )
                stages.append({"epsilon": epsilon, "history": history})
            value, final = solve(d, seed=value, natural_steps=0, face_steps=6)
        attempts = first + final
        accepted = any(a["accepted"] for a in attempts)
        histories = [a["history"] for a in attempts] + [a["history"] for a in stages]
        raw = np.load("/tmp/high_mass_window120_live_before_" + case + ".npz")
        J, W, u = d["J"], d["inverse_mass"], saved["u"]
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
        report = {
            "accepted": accepted,
            "used_proximal": bool(stages),
            "first": first,
            "stages": stages,
            "final": final,
            "original_residual": min(a["residual"] for a in attempts),
            "linear_solves": sum("linear_defect" in h for hist in histories for h in hist),
            "factorizations": sum("rank" in h for hist in histories for h in hist),
            "residual_evaluations": sum(hist[-1]["total_residual_evaluations"] for hist in histories),
            "physical": {
                "linear_balance_max": float(np.max(np.abs(linear))),
                "angular_balance_max": float(np.max(np.abs(angular))),
                "energy_change_J": float(energy),
                "work_error_J": float(energy - work),
            },
            "maximum_linear_budget": 76,
            "seed_normal_masks": 8 * 2 ** (len(value) // 3),
            "seed_tangent_updates": 8 * (len(value) // 3),
            "input_sha256": digest,
        }
        assert report["linear_solves"] <= 76
        if accepted:
            assert max(np.max(np.abs(linear)), np.max(np.abs(angular)), abs(energy - work)) < 1e-10
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        reports[case] = report
        np.savez(prefix + "_" + case + ".npz", **d, solution=value, u=u, v=v)
        print(case, {k: v for k, v in report.items() if k not in ("first", "stages", "final")}, flush=True)
        Path(prefix + "_replay.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
