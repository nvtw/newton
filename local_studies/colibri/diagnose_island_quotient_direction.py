"""High-precision diagnostic retaining the independently certified tiny mode."""

import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

from local_studies.colibri.bounded_island_coulomb import attempt


def main():
    d = np.load("/tmp/high_mass_seed_null_rejection.npz")
    captured = {}

    def trace(frame, event, arg):
        if frame.f_code is attempt.__code__ and event == "line" and not captured:
            loc = frame.f_locals
            if loc.get("response_null", 0) > 1e-10:
                captured.update(evaluate=loc["evaluate"], x=loc["x"].copy())
        return trace

    sys.settrace(trace)
    try:
        attempt(d["F"], d["rhs"], d["initial"], d["mu"])
    finally:
        sys.settrace(None)
    assert captured
    mp.mp.dps = 70
    # This exact matrix is independently certified to have four pure-dual
    # null directions. Keep all32remaining directions, including sigma8e-19.
    matrix = mp.matrix(d["scaled"].tolist())
    residual = mp.matrix(d["r"].tolist())
    U, s, Vh = mp.svd(matrix)
    projected = U.T * (-residual)
    delta_scaled = Vh[:32, :].T * mp.matrix([projected[k] / s[k] for k in range(32)])
    delta = mp.matrix([delta_scaled[k] / mp.mpf(float(d["colscale"][k])) for k in range(36)])
    dx = np.array([float(v) for v in delta])
    r = d["r"]
    jac = d["jac"]
    merit = 0.5 * float(r @ r)
    slope = float(r @ (jac @ dx))
    trials = []
    for k in range(24):
        fraction = 2.0 ** (-k)
        value = captured["x"] + fraction * dx
        rr, _ = captured["evaluate"](value)
        candidate = 0.5 * float(rr @ rr)
        ok = candidate <= merit + 1e-4 * fraction * slope
        trials.append({"backtrack": k, "merit": candidate, "accepted": ok})
        if ok:
            break
    result = {
        "precision_digits": 70,
        "removed_gauges": 4,
        "smallest_retained": float(s[31]),
        "largest_removed": float(s[32]),
        "delta_norm": float(np.linalg.norm(dx)),
        "response_delta_norm": float(np.linalg.norm(dx[:12])),
        "mp_scaled_linear_residual": float(mp.norm(matrix * delta_scaled + residual, p=mp.inf)),
        "fp64_linear_residual": float(np.max(np.abs(jac @ dx + r))),
        "slope": slope,
        "trials": trials,
    }
    Path("/tmp/high_mass_quotient_direction.json").write_text(json.dumps(result, indent=2))
    np.savez("/tmp/high_mass_quotient_direction.npz", delta=dx, singular=np.array([float(v) for v in s]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
