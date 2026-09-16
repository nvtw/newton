"""CPU-only face proposals accepted against every original Coulomb equation."""

# ruff: noqa: TID253 -- standalone numerical experiment.
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.coupled_support_online import assemble_snapshot
from local_studies.colibri.physical_face_support import audit, factor, modes, seed_sweeps, solve_face


def normal_proposal(a, seed):
    """Solve only original normal complementarity, keeping current friction fixed."""
    normal = np.arange(0, len(seed), 3)
    matrix = a["A"][np.ix_(normal, normal)] + np.diag(a["gamma"])
    fixed = seed.copy()
    fixed[normal] = 0
    rhs = (a["A"] @ fixed + a["rhs"])[normal]
    scale = 1 / np.sqrt(np.diag(matrix))
    hessian = matrix * scale[:, None] * scale[None, :]
    linear = rhs * scale
    result = minimize(
        lambda x: 0.5 * x @ hessian @ x + linear @ x,
        np.maximum(seed[normal], 0) / scale,
        jac=lambda x: hessian @ x + linear,
        method="SLSQP",
        bounds=[(0, None)] * len(normal),
        options={"maxiter": 100, "ftol": 1e-16},
    )
    trial = fixed.copy()
    trial[normal] = result.x * scale
    for p in range(len(normal)):
        tangent = trial[3 * p + 1 : 3 * p + 3]
        radius = a["mu"][p] * trial[3 * p]
        norm = np.linalg.norm(tangent)
        if norm > radius:
            tangent *= radius / norm
    gradient = matrix @ trial[normal] + rhs
    natural = np.diag(matrix) * (trial[normal] - np.maximum(0, trial[normal] - gradient / np.diag(matrix)))
    return trial, {
        "iterations": int(result.nit),
        "status": str(result.message),
        "normal_error": float(np.max(abs(natural))),
    }


def solve(a):
    f, _ = factor(a)
    value = seed_sweeps(a, a["old"], 2)
    evaluate, *_ = natural_map_evaluator(a["A"], a["rhs"], a["gamma"], a["mu"])
    history = []
    for outer in range(6):
        residual = evaluate(value)[0]
        merit = float(residual @ residual)
        if np.max(abs(residual)) < 1e-8:
            break
        proposals = []
        direct_active, direct_stick = modes(a, value)
        proposals.append(("current_face", value, direct_active, direct_stick, {}))
        normal, report = normal_proposal(a, value)
        active, stick = modes(a, normal)
        proposals.append(("normal_reclassification", normal, active, stick, report))
        best = value
        best_merit = merit
        for label, seed, active, stick, extra in proposals:
            try:
                trial, face = solve_face(a, f, seed, active, stick)
                attempts = [face]
                pivots = []
                # Numerical face proposals only: never apply negative impulses.
                # Every omitted normal remains in the final all-row KKT gate.
                for _ in range(3):
                    point = int(np.argmin(trial[::3]))
                    if trial[3 * point] >= -1e-12:
                        break
                    pivots.append({"point": point, "negative_trial_normal": float(trial[3 * point])})
                    active = active.copy()
                    stick = stick.copy()
                    active[point] = False
                    stick[point] = False
                    trial, face = solve_face(a, f, seed, active, stick)
                    attempts.append(face)
                face = dict(face)
                face["attempts_before_final"] = attempts[:-1]
                face["negative_normal_pivots"] = pivots
            except (ValueError, np.linalg.LinAlgError) as exc:
                history.append({"outer": outer, "label": label, "rejected": str(exc)})
                continue
            # A trial face solution may be physically infeasible. Its direction
            # is only a numerical proposal; each accepted impulse stays in cones.
            direction = trial - value
            accepted = False
            for backtrack in range(20):
                alpha = 2.0 ** (-backtrack)
                candidate = value + alpha * direction
                for p in range(len(a["mu"])):
                    row = 3 * p
                    candidate[row] = max(0.0, candidate[row])
                    tangent = candidate[row + 1 : row + 3]
                    radius = a["mu"][p] * candidate[row]
                    norm = np.linalg.norm(tangent)
                    if norm > radius:
                        tangent *= radius / norm
                rr = evaluate(candidate)[0]
                mm = float(rr @ rr)
                if mm <= merit * (1 - 1e-4 * alpha) and mm < best_merit:
                    best, best_merit = candidate, mm
                    accepted = True
                    break
            history.append(
                {
                    "outer": outer,
                    "label": label,
                    "active": np.flatnonzero(active).tolist(),
                    "stick": np.flatnonzero(stick).tolist(),
                    "normal": extra,
                    "face": face,
                    "accepted": accepted,
                    "alpha": alpha,
                    "before_merit": merit,
                    "after_merit": mm,
                }
            )
        if best_merit >= merit:
            break
        value = best
    return value, {
        "history": history,
        "audit": audit(a, value),
        "bounds": {
            "seed_sweeps": 2,
            "outer": 6,
            "normal_qp_iterations": 100,
            "face_newton_steps": 12,
            "face_backtracks": 20,
            "negative_normal_pivots": 3,
            "merit_backtracks": 20,
        },
        "scope": "Numerical face proposals only; every final original contact/joint equation checked; no reference face",
    }


def main():
    z = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    reports = []
    for phase in ("biased", "relax"):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_solved.")}
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        value, report = solve(a)
        report["phase"] = phase
        reports.append(report)
        np.savez(f"/tmp/colibri_physical_face_active_{phase}.npz", lam=value)
    Path("/tmp/colibri_physical_face_active.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
