"""Bounded exact natural-map Newton with deterministic backtracking."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def polish(A, rhs, initial, friction, max_steps=40):
    evaluate, scale, operator, mu = natural_map_evaluator(A, rhs, np.zeros(len(rhs) // 3), friction)
    value = initial.copy()
    history = []
    for step in range(max_steps):
        residual, jac = evaluate(value)
        error = float(np.max(np.abs(residual)))
        merit = float(residual @ residual) * 0.5
        cone = float(np.max(np.linalg.norm(value.reshape(-1, 3)[:, 1:], axis=1) - mu * value[::3]))
        if error < 1e-8 and cone < 1e-8:
            history.append({"step": step, "residual": error, "cone": cone, "accepted": True})
            break
        scaled = jac / scale[None, :]
        direction, _, rank, singular = np.linalg.lstsq(scaled, -residual, rcond=None)
        direction /= scale
        defect = float(np.max(np.abs(jac @ direction + residual)))
        derivative = float(residual @ (jac @ direction))
        fraction = 1.0
        accepted = False
        for _backtrack in range(30):
            candidate = value + fraction * direction
            next_residual, _ = evaluate(candidate)
            if 0.5 * float(next_residual @ next_residual) <= merit + 1e-4 * fraction * derivative:
                accepted = True
                break
            fraction *= 0.5
        history.append(
            {
                "step": step,
                "residual": error,
                "cone": cone,
                "rank": int(rank),
                "smallest_retained": float(singular[rank - 1]),
                "linear_defect": defect,
                "direction_norm": float(np.linalg.norm(direction)),
                "fraction": fraction,
                "backtracks": _backtrack,
                "line_search_accepted": accepted,
            }
        )
        if not accepted:
            break
        value = candidate
    residual, _ = evaluate(value)
    return value, history, float(np.max(np.abs(residual)))


def main():
    d = np.load("/tmp/colibri_physical_gs_combined_frame_crank_2.npz")
    solution, history, error = polish(d["A"], d["rhs"], d["solution"], d["friction"])
    report = {
        "initial_residual": float(
            np.max(
                np.abs(
                    natural_map_evaluator(d["A"], d["rhs"], np.zeros(len(d["friction"])), d["friction"])[0](
                        d["solution"]
                    )[0]
                )
            )
        ),
        "final_residual": error,
        "accepted": error < 1e-8,
        "history": history,
    }
    Path("/tmp/colibri_frame_crank_newton_polish.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed("/tmp/colibri_frame_crank_newton_polish.npz", solution=solution)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
