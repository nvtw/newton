"""Bounded fully coupled non-associated Coulomb feasibility oracle."""

import json
import time
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def main():
    from scipy.linalg import block_diag
    from scipy.optimize import least_squares

    a = np.load("/tmp/colibri_joint_tree_sweeps64.npz")
    p = np.load("/tmp/colibri_joint_reformed_input.npz")
    C = a["C"]
    G = a["G"]
    old = a["old"]
    base = a["base"]
    seed = a["lam"].ravel().copy()
    F = C.reshape(-1, G.shape[0]) @ G
    H = F @ F.T
    rhs = C.reshape(-1, G.shape[0]) @ base - H @ old.ravel()
    evaluate, _, _, _ = natural_map_evaluator(H, rhs, np.zeros(len(C)), friction=a["mu"])
    lower = np.full(len(seed), -np.inf)
    lower[::3] = 0
    seed[::3] = np.maximum(seed[::3], 1e-15)
    start = time.perf_counter()
    result = least_squares(
        lambda x: 1e4 * evaluate(x)[0],
        seed,
        jac=lambda x: 1e4 * evaluate(x)[1],
        bounds=(lower, np.inf),
        x_scale="jac",
        max_nfev=100,
        ftol=1e-13,
        gtol=1e-13,
        xtol=1e-13,
    )
    value = result.x.reshape(-1, 3)
    coordinates = F.T @ (result.x - old.ravel())
    dv = G @ coordinates
    velocity = base + dv
    gradient = C @ velocity
    W = block_diag(*(p["W"][body] for body in a["active"]))
    physicalH = C @ W @ C.transpose(0, 2, 1)
    normal = []
    tangent = []
    for k in range(len(C)):
        ss = float(np.max(np.diag(physicalH[k])))
        g = gradient[k]
        normal.append(max(-g[0], -value[k, 0] * ss, abs(min(value[k, 0] * ss, g[0])), 0))
        trial = value[k, 1:] - g[1:] / ss
        radius = a["mu"][k] * max(value[k, 0], 0)
        projection = trial * min(1, radius / max(np.linalg.norm(trial), 1e-300))
        tangent.append(np.linalg.norm(value[k, 1:] - projection) * ss)
    out = {
        "seed": "64 metric point sweeps,4480pointupdates,81984diskbisections",
        "max_nfev": 100,
        "nfev": result.nfev,
        "njev": result.njev,
        "solver_status": result.message,
        "seconds": time.perf_counter() - start,
        "natural_map_residual": float(abs(evaluate(result.x)[0]).max()),
        "max_normal_residual": float(max(normal)),
        "max_tangent_residual": float(max(tangent)),
        "max_cone_violation": float(np.max(np.linalg.norm(value[:, 1:], axis=1) - a["mu"] * value[:, 0])),
        "max_impulse": float(abs(value).max()),
    }
    np.savez(
        "/tmp/colibri_joint_coulomb210.npz", lam=value, coordinates=coordinates, dv=dv, velocity=velocity, H=H, rhs=rhs
    )
    Path("/tmp/colibri_joint_coulomb210.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
