"""Fixed auxiliary-friction seeds; only original-friction solution is physical."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator, solve_coulomb
from local_studies.colibri.polish_frame_crank_newton import polish


def friction_continuation(A, rhs, initial, original_mu):
    """Generate seeds at four fixed coefficients; only the last is physical."""
    seed = initial.copy()
    reports = []
    for fraction in (0.0, 0.25, 0.5, 1.0):
        mu = fraction * original_mu
        solution, attempts = solve_coulomb(A, rhs, seed, np.zeros(len(mu)), mu, residual_scale=1e4)
        evaluate, _, _, _ = natural_map_evaluator(A, rhs, np.zeros(len(mu)), mu)
        error = float(np.max(np.abs(evaluate(solution)[0])))
        if error >= 1e-8:
            solution, history, error = polish(A, rhs, solution, mu)
            attempts.append({"method": "bounded_newton_armijo", "history": history})
        cone = float(np.max(np.linalg.norm(solution.reshape(-1, 3)[:, 1:], axis=1) - mu * solution[::3]))
        reports.append(
            {
                "friction_fraction": fraction,
                "residual": error,
                "cone": cone,
                "accepted_auxiliary": error < 1e-8 and cone < 1e-8,
                "attempts": attempts,
            }
        )
        seed = solution
    return seed, reports, error


def main(resume=False):
    d = np.load("/tmp/colibri_physical_gs_combined_frame_crank_3.npz")
    A = d["A"]
    rhs = d["rhs"]
    original_mu = d["friction"]
    seed = d["solution"].copy()
    reports = []
    accepted = False
    fractions = (0.0, 0.25, 0.5, 1.0)
    if resume:
        reports = json.loads(Path("/tmp/colibri_frame_crank_friction_continuation.json").read_text())["stages"]
        seed = np.load("/tmp/colibri_frame_crank_friction_continuation.npz")["solution"]
        fractions = fractions[len(reports) :]
    for fraction in fractions:
        mu = fraction * original_mu
        solution, attempts = solve_coulomb(A, rhs, seed, np.zeros(len(mu)), mu, residual_scale=1e4)
        evaluate, _, _, _ = natural_map_evaluator(A, rhs, np.zeros(len(mu)), mu)
        error = float(np.max(np.abs(evaluate(solution)[0])))
        if error >= 1e-8:
            solution, history, error = polish(A, rhs, solution, mu)
            attempts.append({"method": "bounded_newton_armijo", "history": history})
        cone = float(np.max(np.linalg.norm(solution.reshape(-1, 3)[:, 1:], axis=1) - mu * solution[::3]))
        passed = error < 1e-8 and cone < 1e-8
        reports.append(
            {
                "friction_fraction": fraction,
                "residual": error,
                "cone": cone,
                "accepted_auxiliary": passed,
                "attempts": attempts,
            }
        )
        np.savez_compressed(f"/tmp/colibri_frame_crank_mu_stage_{fraction}.npz", solution=solution, friction=mu)
        print("STAGE", fraction, error, cone, passed, flush=True)
        seed = solution
        accepted = fraction == 1.0 and passed
    report = {"accepted_original_friction": accepted, "auxiliary_only": True, "stages": reports}
    Path("/tmp/colibri_frame_crank_friction_continuation.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed("/tmp/colibri_frame_crank_friction_continuation.npz", solution=seed)


if __name__ == "__main__":
    main()
