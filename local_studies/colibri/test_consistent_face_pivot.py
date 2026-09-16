"""A simultaneous face change justified by converged-face KKT violations."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt, response_factor
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def main():
    d = np.load("/tmp/high_mass_consistent_245_1.npz")
    record = json.loads(Path("/tmp/high_mass_consistent_modes.json").read_text())[4]
    assert record["face_equation_residual"] < 1e-8
    modes = np.array(record["modes"])
    before = modes.copy()
    seed = np.array(record["impulse"]).ravel()
    for k in range(len(modes)):
        if modes[k] == 1 and record["disk_violation"][k] > 1e-8:
            modes[k] = 2
    F = response_factor(d["J"], d["inverse_mass"])
    value, history = attempt(F, d["rhs"], seed, np.full(len(modes), 0.5), modes=modes, steps=8)
    r, _ = natural_map_evaluator(d["A"], d["rhs"], np.zeros(len(modes)), 0.5)[0](value)
    cone = float(np.max(np.linalg.norm(value.reshape(-1, 3)[:, 1:], axis=1) - 0.5 * value[::3]))
    work = float(value @ d["rhs"] + 0.5 * value @ d["A"] @ value)
    result = {
        "before_modes": before.tolist(),
        "after_modes": modes.tolist(),
        "residual": float(np.max(np.abs(r))),
        "cone": cone,
        "normal_min": float(min(value[::3])),
        "contact_work_J": work,
        "accepted": bool(max(abs(r)) < 1e-8 and cone < 1e-8 and min(value[::3]) >= -1e-8 and work <= 1e-8),
        "history": history,
    }
    Path("/tmp/high_mass_consistent_face_pivot.json").write_text(json.dumps(result, indent=2))
    np.savez("/tmp/high_mass_consistent_face_pivot.npz", solution=value)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
