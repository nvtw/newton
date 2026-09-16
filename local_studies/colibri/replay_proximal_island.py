"""Fixed-budget numerical proximal continuation; only zero-stage roots accepted."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt, response_factor, solve


def main():
    reports = {}
    for case in ("263_2", "300_2", "245_1"):
        saved = np.load("/tmp/high_mass_tied_seed_" + case + ".npz")
        d = {k: saved[k] for k in ("A", "J", "inverse_mass", "rhs", "initial", "normal_regularization")}
        F = response_factor(d["J"], d["inverse_mass"])
        seed = saved["seed"].copy()
        reference = seed.copy()
        stages = []
        jvp_errors = []

        def audit(x, evaluate, errors=jvp_errors):
            rng = np.random.default_rng(3625)
            x = x + 1e-3 * rng.normal(size=len(x))
            r, jac = evaluate(x)
            for _ in range(8):
                direction = rng.normal(size=len(x))
                direction /= np.linalg.norm(direction)
                h = 1e-6
                fd = (evaluate(x + h * direction)[0] - evaluate(x - h * direction)[0]) / (2 * h)
                error = float(np.max(np.abs(fd - jac @ direction)))
                errors.append(error)
                assert error < 1e-6

        for epsilon in (0.1, 0.01):
            seed, history = attempt(
                F,
                d["rhs"],
                seed,
                np.full(len(seed) // 3, 0.5),
                steps=8,
                numerical_epsilon=epsilon,
                numerical_reference=reference,
                audit_evaluator=audit,
            )
            stages.append({"epsilon": epsilon, "history": history})
        value, attempts = solve(d, seed=seed, face_steps=6)
        old = np.load("/tmp/high_mass_consistent960_" + case + ".npz")
        raw = np.load("/tmp/high_mass_window120_live_before_" + case + ".npz")
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
        accepted = any(a["accepted"] for a in attempts)
        histories = [s["history"] for s in stages] + [a["history"] for a in attempts]
        report = {
            "accepted": accepted,
            "stages": stages,
            "zero_stage": attempts,
            "original_residual": min(a["residual"] for a in attempts),
            "linear_solves": sum("linear_defect" in h for hist in histories for h in hist),
            "factorizations": sum("rank" in h for hist in histories for h in hist),
            "residual_evaluations_including_derivative_audit": sum(
                hist[-1]["total_residual_evaluations"] for hist in histories
            ),
            "derivative_audit_evaluations": 34,
            "max_JVP_error": max(jvp_errors),
            "physical": {
                "linear_balance_max": float(np.max(np.abs(linear))),
                "angular_balance_max": float(np.max(np.abs(angular))),
                "energy_change_J": float(energy),
                "work_error_J": float(energy - work),
            },
            "maximum_linear_budget": 76,
        }
        if accepted:
            assert max(np.max(np.abs(linear)), np.max(np.abs(angular)), abs(energy - work)) < 1e-10
        reports[case] = report
        np.savez("/tmp/high_mass_proximal_" + case + ".npz", **d, solution=value, seed=seed, u=u, v=v)
        print(case, {k: v for k, v in report.items() if k not in ("stages", "zero_stage")}, flush=True)
        Path("/tmp/high_mass_proximal_replay.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
