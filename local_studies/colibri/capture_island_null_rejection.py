"""Read-only Python trace of the first response-null rejection."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt, response_factor


def main():
    d = np.load("/tmp/high_mass_window120_live_245_1.npz")
    seed = np.load("/tmp/high_mass_bounded_seed_245_1_normal8.npz")["seed"]
    F = response_factor(d["J"], d["inverse_mass"])
    saved = {}

    def trace(frame, event, arg):
        if frame.f_code is attempt.__code__ and event == "line" and not saved:
            local = frame.f_locals
            if local.get("response_null", 0) > 1e-10:
                for key in (
                    "F",
                    "rhs",
                    "initial",
                    "mu",
                    "x",
                    "r",
                    "jac",
                    "colscale",
                    "scaled",
                    "U",
                    "s",
                    "Vh",
                    "keep",
                    "null",
                    "scale",
                ):
                    saved[key] = np.asarray(local[key]).copy()
                saved["nv"] = np.array([local["nv"]])
        return trace

    sys.settrace(trace)
    try:
        _, history = attempt(F, d["rhs"], seed, np.full(len(seed) // 3, 0.5))
    finally:
        sys.settrace(None)
    assert saved
    nv = int(saved["nv"][0])
    saved["local_derivative"] = saved["jac"][nv:, nv:].copy()
    saved["response_derivative"] = saved["jac"][nv:, :nv].copy()
    saved["physical_A"] = d["A"].copy()
    np.savez("/tmp/high_mass_seed_null_rejection.npz", **saved)
    Path("/tmp/high_mass_seed_null_rejection.json").write_text(json.dumps(history, indent=2))
    print({key: value.shape for key, value in saved.items()})


if __name__ == "__main__":
    main()
