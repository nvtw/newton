"""One topology-derived interface candidate, independently physically checked."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt, response_factor, solve
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def main():
    reports = {}
    for case in ("300_2", "244_0", "245_1", "263_2", "362_5"):
        d = np.load("/tmp/high_mass_consistent960_" + case + ".npz")
        raw = np.load("/tmp/high_mass_window120_live_before_" + case + ".npz")
        original = np.load("/tmp/high_mass_window120_live_" + case + ".npz")
        F = response_factor(d["J"], d["inverse_mass"])
        mu = np.full(len(d["rhs"]) // 3, 0.5)
        strict = d["seed"]
        value, first = solve(d, seed=strict, max_face_visits=0)
        stages = []
        for eps in (0.1, 0.01):
            value, hist = attempt(F, d["rhs"], value, mu, steps=8, numerical_epsilon=eps, numerical_reference=strict)
            stages.append(hist)
        gradient = (d["A"] @ value + d["rhs"]).reshape(-1, 3)
        h = raw["headers"].view(np.int32)
        groups = {}
        for k, p in enumerate(original["selected_points"]):
            c = next(c for c in range(int(raw["column_count"][0])) if h[5, c] <= p < h[5, c] + h[6, c])
            pair = tuple(sorted(map(int, h[1:3, c])))
            groups.setdefault(pair, []).append(k)
        scores = {pair: float(np.sqrt(np.mean(np.sum(gradient[ids, 1:] ** 2, axis=1)))) for pair, ids in groups.items()}
        selected = min(scores, key=lambda pair: (-scores[pair], pair))
        modes = np.where(value[::3] > 0, 1, 0)
        for k in groups[selected]:
            if modes[k]:
                modes[k] = 2
        seed = value.copy()
        value, history = attempt(F, d["rhs"], seed, mu, modes=modes, steps=6)
        r = natural_map_evaluator(d["A"], d["rhs"], np.zeros(len(mu)), mu)[0](value)[0]
        g = (d["A"] @ value + d["rhs"]).reshape(-1, 3)
        triples = value.reshape(-1, 3)
        normal = float(np.max(np.abs(np.minimum(value[::3] * np.diag(d["A"])[::3], g[:, 0]))))
        cone = float(np.max(np.linalg.norm(triples[:, 1:], axis=1) - 0.5 * triples[:, 0]))
        contactwork = float(value @ d["rhs"] + 0.5 * value @ d["A"] @ value)
        accepted = (
            max(abs(r)) < 1e-8 and normal < 1e-8 and cone < 1e-8 and min(value[::3]) >= -1e-8 and contactwork <= 1e-8
        )
        J, W, u = d["J"], d["inverse_mass"], d["u"]
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
            "accepted": bool(accepted),
            "modes": modes.tolist(),
            "groups": {str(k): v for k, v in groups.items()},
            "interface_RMS_slip": {str(k): v for k, v in scores.items()},
            "selected_pair": list(selected),
            "original_residual": float(max(abs(r))),
            "normal": normal,
            "cone": cone,
            "history": history,
            "physical": {
                "linear_balance_max": float(max(abs(linear))),
                "angular_balance_max": float(max(abs(angular))),
                "energy_change_J": float(energy),
                "work_error_J": float(energy - work),
            },
            "total_linear_solves": sum(
                "linear_defect" in h for hist in [first[0]["history"], *stages, history] for h in hist
            ),
        }
        if accepted:
            assert max(*abs(linear), *abs(angular), abs(energy - work)) < 1e-10
        reports[case] = report
        np.savez("/tmp/high_mass_interface_move_" + case + ".npz", solution=value, seed=seed, modes=modes)
        print(case, report, flush=True)
        Path("/tmp/high_mass_interface_move.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
