"""Bounded residual-gated condensed reference, with inferred-face polishing."""

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.physical_face_active_set import solve as solve_active
from local_studies.colibri.physical_face_support import factor, modes, solve_face
from local_studies.colibri.two_body_condensed_combined import solve as solve_pgs


def solve(a, sweeps=32):
    """Accept only the original all-point law; never discard a failed row."""
    evaluate, *_ = natural_map_evaluator(a["A"], a["rhs"], a["gamma"], a["mu"])
    current = dict(a)
    stages = []
    total = 0
    if len(a["mu"]):
        try:
            trial, info = solve_active(a)
            trial_error = float(np.max(abs(evaluate(trial)[0])))
            stages.append({"kind": "two_seed_active_set", "contact_sweeps": 2, "error": trial_error, "details": info})
            if info["audit"]["accepted"] and np.isfinite(trial_error) and trial_error < 1e-8:
                seeded = dict(a)
                seeded["old"] = trial
                lam, v, joint, response = solve_pgs(seeded, 0)
                response["stages"] = stages
                return lam, v, joint, response
        except (ValueError, AssertionError, np.linalg.LinAlgError) as exc:
            stages.append({"kind": "two_seed_active_set_rejected", "reason": str(exc)})
    for budget in (32, 64, 128, 256):
        lam, v, joint, response = solve_pgs(current, budget - total)
        total = budget
        error = float(np.max(abs(evaluate(lam)[0]))) if len(lam) else 0.0
        stages.append({"contact_sweeps": budget, "error": error, "kind": "native_metric_pgs"})
        if error < 1e-8:
            response["stages"] = stages
            return lam, v, joint, response
        try:
            f, _ = factor(a)
            active, stick = modes(a, lam)
            trial, info = solve_face(a, f, lam, active, stick)
            trial_error = float(np.max(abs(evaluate(trial)[0])))
            stages.append({"kind": "inferred_face", "error": trial_error, "details": info})
            if np.isfinite(trial_error) and trial_error < 1e-8:
                v = response["baseline"] + response["response"] @ trial
                y = response["joint_solved"]
                joint = y[:, 0] - y[:, 1:] @ trial
                response["stages"] = stages
                return trial, v, joint, response
        except (ValueError, AssertionError, np.linalg.LinAlgError) as exc:
            stages.append({"kind": "inferred_face_rejected", "reason": str(exc)})
        current["old"] = lam.copy()
    response["stages"] = stages
    return lam, v, joint, response
