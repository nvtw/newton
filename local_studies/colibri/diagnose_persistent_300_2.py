"""Frozen failure diagnosis with an independent saved reference root control."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt, response_factor
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.replay_adaptive_proximal import main as replay


def main():
    d = np.load("/tmp/high_mass_consistent960_300_2.npz")
    old = np.load("/tmp/high_mass_window120_live_300_2.npz")
    F = response_factor(d["J"], d["inverse_mass"])
    A = d["A"]
    rhs = d["rhs"]
    mu = np.full(len(rhs) // 3, 0.5)
    records = []
    snapshots = {}

    def trace(frame, event, arg):
        if frame.f_code is attempt.__code__ and event == "return":
            loc = frame.f_locals
            value = arg[0]
            x = loc["x"]
            r, jac = loc["evaluate"](x)
            gradient = A @ value + rhs
            modes = loc["modes"]
            nv = loc["nv"]
            m = loc["m"]
            rec = {
                "epsilon": loc["numerical_epsilon"],
                "modes": None if modes is None else modes.tolist(),
                "impulse": value.reshape(-1, 3).tolist(),
                "physical_velocity": gradient.reshape(-1, 3).tolist(),
                "sliding_multipliers": x[nv + m :].tolist(),
                "equation_residual": float(np.max(np.abs(r))),
                "physical_natural_residual": float(
                    np.max(np.abs(natural_map_evaluator(A, rhs, np.zeros(len(mu)), mu)[0](value)[0]))
                ),
            }
            records.append(rec)
            if "rejected_response_null" in arg[1][-1]:
                for name in ("F", "x", "jac", "colscale", "scaled", "s", "Vh", "keep", "null"):
                    if name in loc:
                        snapshots[str(len(records) - 1) + "_" + name] = loc[name]
        return trace

    sys.settrace(trace)
    try:
        replay(cases=("300_2",), prefix="/tmp/high_mass_300_2_diagnostic")
    finally:
        sys.settrace(None)
    reference = old["coupled_solution"]
    # Reference is used only to establish root existence and physical modes,
    # never as a seed in the bounded production-candidate replay above.
    value, history = attempt(F, rhs, reference, mu, modes=np.array([2, 2, 2, 1, 1, 1, 1]), steps=8)
    evaluate = natural_map_evaluator(A, rhs, np.zeros(len(mu)), mu)[0]
    error = float(np.max(np.abs(evaluate(value)[0])))
    g = (A @ value + rhs).reshape(-1, 3)
    x = value.reshape(-1, 3)
    W = d["inverse_mass"]
    u = d["u"]
    delta = d["J"].T @ (value - d["initial"])
    v = u + W @ delta
    M = np.zeros_like(W)
    for b in (1, 2):
        M[6 * b : 6 * b + 6, 6 * b : 6 * b + 6] = np.linalg.inv(W[6 * b : 6 * b + 6, 6 * b : 6 * b + 6])
    raw = np.load("/tmp/high_mass_window120_live_before_300_2.npz")
    reaction = delta.reshape(-1, 6).copy()
    reaction[1:] = (M @ (v - u)).reshape(-1, 6)[1:]
    linear = reaction[:, :3].sum(0)
    angular = (reaction[:, 3:] + np.cross(raw["positions"], reaction[:, :3])).sum(0)
    energy = 0.5 * (v @ M @ v - u @ M @ u)
    work = delta @ ((u + v) * 0.5)
    report = {
        "sample": "300_2",
        "selected_points": old["selected_points"].tolist(),
        "attempts": records,
        "reference_control": {
            "residual": error,
            "history": history,
            "impulses": x.tolist(),
            "velocities": g.tolist(),
            "disk_margin": (0.5 * x[:, 0] - np.linalg.norm(x[:, 1:], axis=1)).tolist(),
            "linear_balance": linear.tolist(),
            "angular_balance": angular.tolist(),
            "energy_change_J": float(energy),
            "work_error_J": float(energy - work),
        },
    }
    Path("/tmp/high_mass_300_2_diagnosis.json").write_text(json.dumps(report, indent=2))
    np.savez("/tmp/high_mass_300_2_rejected_matrices.npz", **snapshots)
    np.savez("/tmp/high_mass_300_2_reference_root.npz", J=d["J"], A=A, rhs=rhs, solution=value, u=u, v=v)
    print(json.dumps(report["reference_control"], indent=2))


if __name__ == "__main__":
    main()
