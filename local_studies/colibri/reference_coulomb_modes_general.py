"""Bounded discrete physical-mode reference, no live changes."""

import argparse
import itertools
import json
import time

import numpy as np
from scipy.optimize import root  # noqa: TID253 - offline reference only

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--prior-mode")
parser.add_argument("--direction-offset", type=float, default=0.0)
args = parser.parse_args()
r = np.load(args.snapshot)
A = r["A"]
q = r["rhs"]
n = len(q) // 3
seed = r["coupled_solution"].copy()
gseed = (A @ seed + q).reshape(-1, 3)
prior = json.load(open(args.prior_mode))["accepted"] if args.prior_mode else None
start = time.perf_counter()
attempts = 0
best = 1e9
accepted = None
sizes = list(range(n, 0, -1))
if prior and len(prior["active"]) in sizes:
    sizes.remove(len(prior["active"]))
    sizes.insert(0, len(prior["active"]))
for size in sizes:
    active_sets = list(itertools.combinations(range(n), size))
    if prior and tuple(prior["active"]) in active_sets:
        active_sets.remove(tuple(prior["active"]))
        active_sets.insert(0, tuple(prior["active"]))
    for active in active_sets:
        sliding_sizes = list(range(1, size + 1))
        if prior and len(prior["sliding"]) in sliding_sizes:
            sliding_sizes.remove(len(prior["sliding"]))
            sliding_sizes.insert(0, len(prior["sliding"]))
        for sliding_count in sliding_sizes:
            sliding_sets = list(itertools.combinations(active, sliding_count))
            if prior and tuple(prior["sliding"]) in sliding_sets:
                sliding_sets.remove(tuple(prior["sliding"]))
                sliding_sets.insert(0, tuple(prior["sliding"]))
            for sliding in sliding_sets:
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

                theta = np.array([np.arctan2(gseed[k, 2], gseed[k, 1]) for k in sliding]) + args.direction_offset
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
                    np.savez(args.output + ".npz", coupled_solution=v)
                    break
            if accepted:
                break
        if accepted:
            break
    if accepted:
        break
    print("Finished active size", size, "attempts", attempts, "best", best, flush=True)
print("FINAL", accepted, "attempts", attempts, "best", best, "seconds", time.perf_counter() - start)
open(args.output + ".json", "w").write(
    json.dumps(
        {"accepted": accepted, "attempts": attempts, "best": best, "seconds": time.perf_counter() - start}, indent=2
    )
)
