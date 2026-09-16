"""Bounded discrete physical-mode reference, no live changes."""

import itertools
import json
import time

import numpy as np
from scipy.optimize import root  # noqa: TID253 - offline reference only

r = np.load("/tmp/high_mass_relax_live_245_1.npz")
A = r["A"][:21, :21]
q = r["rhs"][:21]
n = 7
seed = np.load("/tmp/high_mass_relax_245_1_duplicate_aggregate.npz")["coupled_solution"][:21].copy()
seed[12:15] *= 2
gseed = (A @ seed + q).reshape(-1, 3)
start = time.perf_counter()
attempts = 0
best = 1e9
accepted = None
for size in range(n, 0, -1):
    for active in itertools.combinations(range(n), size):
        for sliding_count in range(1, size + 1):
            for sliding in itertools.combinations(active, sliding_count):
                sticking = [k for k in active if k not in sliding]
                nv = len(active) + 2 * len(sticking)
                eq = np.array([3 * k for k in active] + [3 * k + j for k in sticking for j in (1, 2)])

                def solve(theta, nv=nv, active=active, sliding=sliding, sticking=sticking, eq=eq):
                    T = np.zeros((3 * n, nv))
                    for col, k in enumerate(active):
                        T[3 * k, col] = 1
                        if k in sliding:
                            j = sliding.index(k)
                            T[3 * k + 1, col] = -0.5 * np.cos(theta[j])
                            T[3 * k + 2, col] = -0.5 * np.sin(theta[j])
                    for j, k in enumerate(sticking):
                        T[3 * k + 1, len(active) + 2 * j] = 1
                        T[3 * k + 2, len(active) + 2 * j + 1] = 1
                    coeff = np.linalg.lstsq(A[eq] @ T, -q[eq], rcond=1e-11)[0]
                    v = T @ coeff
                    g = A @ v + q
                    f = np.array(
                        [
                            -np.sin(theta[j]) * g[3 * k + 1] + np.cos(theta[j]) * g[3 * k + 2]
                            for j, k in enumerate(sliding)
                        ]
                    )
                    return f, v, g

                theta = np.array([np.arctan2(gseed[k, 2], gseed[k, 1]) for k in sliding])
                sol = root(lambda t: solve(t)[0], theta, method="hybr", options={"xtol": 1e-10, "maxfev": 100})
                f, v, g = solve(sol.x)
                attempts += 1
                normals = v[::3]
                gn = g[::3]
                cone = np.linalg.norm(v.reshape(-1, 3)[:, 1:], axis=1) - 0.5 * normals
                error = max(
                    float(max(abs(np.minimum(np.diag(A)[::3] * normals, gn)))), float(max(cone)), float(max(abs(g[eq])))
                )
                # eq includes normal and sticking velocities; sliding residual is perpendicular slip plus nonnegative slip direction.
                for j, k in enumerate(sliding):
                    error = max(error, abs(f[j]), -np.cos(sol.x[j]) * g[3 * k + 1] - np.sin(sol.x[j]) * g[3 * k + 2])
                best = min(best, error)
                if error < 1e-8:
                    accepted = {
                        "active": active,
                        "sliding": sliding,
                        "error": error,
                        "attempts": attempts,
                        "seconds": time.perf_counter() - start,
                    }
                    print("ACCEPT", accepted, flush=True)
                    full = np.zeros(24)
                    full[:21] = v
                    full[12:15] *= 0.5
                    full[21:24] = full[12:15]
                    np.savez("/tmp/high_mass_relax_245_1_mode_solution.npz", coupled_solution=full)
                    break
            if accepted:
                break
        if accepted:
            break
    if accepted:
        break
    print("Finished active size", size, "attempts", attempts, "best", best, flush=True)
print("FINAL", accepted, "attempts", attempts, "best", best, "seconds", time.perf_counter() - start)
open("/tmp/high_mass_relax_245_1_mode_search.json", "w").write(
    json.dumps(
        {"accepted": accepted, "attempts": attempts, "best": best, "seconds": time.perf_counter() - start}, indent=2
    )
)
