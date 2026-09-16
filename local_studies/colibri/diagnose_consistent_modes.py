"""Inspect full face feasibility after each unchanged bounded face solve."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt, solve


def main():
    d = np.load("/tmp/high_mass_consistent_245_1.npz")
    A, rhs = d["A"], d["rhs"]
    records = []

    def trace(frame, event, arg):
        if frame.f_code is attempt.__code__ and event == "return":
            loc = frame.f_locals
            value = arg[0]
            modes = loc["modes"]
            gradient = A @ value + rhs
            triples = value.reshape(-1, 3)
            disk = np.linalg.norm(triples[:, 1:], axis=1) - 0.5 * triples[:, 0]
            records.append(
                {
                    "modes": None if modes is None else modes.tolist(),
                    "impulse": triples.tolist(),
                    "velocity": gradient.reshape(-1, 3).tolist(),
                    "disk_violation": disk.tolist(),
                    "sliding_multiplier": loc["x"][loc["nv"] + loc["m"] :].tolist(),
                    "face_equation_residual": float(np.max(np.abs(loc["evaluate"](loc["x"])[0]))),
                }
            )
        return trace

    sys.settrace(trace)
    try:
        solve(d, seed=d["seed"])
    finally:
        sys.settrace(None)
    Path("/tmp/high_mass_consistent_modes.json").write_text(json.dumps(records, indent=2))
    for i, r in enumerate(records):
        modes = r["modes"]
        if modes is None:
            continue
        bad = []
        for k, mode in enumerate(modes):
            lam = r["impulse"][k]
            g = r["velocity"][k]
            if mode == 0 and g[0] < -1e-8:
                bad.append((k, "inactive_closing", g[0]))
            if mode != 0 and lam[0] < -1e-8:
                bad.append((k, "negative_normal", lam[0]))
            if mode == 1 and r["disk_violation"][k] > 1e-8:
                bad.append((k, "stick_outside_disk", r["disk_violation"][k]))
            if mode == 2:
                index = [j for j, m in enumerate(modes) if m == 2].index(k)
                if r["sliding_multiplier"][index] < -1e-8:
                    bad.append((k, "negative_slip_multiplier", r["sliding_multiplier"][index]))
        print(i, modes, "eq", r["face_equation_residual"], "bad", bad)


if __name__ == "__main__":
    main()
