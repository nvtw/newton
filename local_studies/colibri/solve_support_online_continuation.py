"""Bounded online friction continuation; seed uses captured impulses, not a root."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, least_squares
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--direct", action="store_true")
parser.add_argument("--gpu-response", action="store_true")
args = parser.parse_args()
output = "/tmp/colibri_support_online_direct" if args.direct else "/tmp/colibri_support_online_continuation"
if args.gpu_response:
    output += "_gpu_response"
d = np.load("/tmp/colibri_two_body_full_biased.npz")
A, q, gamma, mu = d["A"], d["rhs"], d["regularization"], d["mu"]
if args.gpu_response:
    gpu = np.load("/tmp/single_block_support_biased.npz")
    W, B, C = d["W"], d["B"], d["C"]
    free = d["velocity"] - W @ (C.T @ d["initial"] + B.T @ d["old_joint"])
    K = B @ W @ B.T + np.diag(d["diagonal"])
    baseline = free + W @ B.T @ np.linalg.solve(K, d["targets"] - B @ free)
    q = q + C @ (gpu["baseline"] - baseline)
    A = C @ gpu["G"]
n = len(mu)
N = A[::3, ::3] + np.diag(gamma)
qn = q[::3]
scale = 1 / np.sqrt(np.diag(N))
H = N * scale[:, None] * scale[None, :]
f = qn * scale
result = minimize(
    lambda x: 0.5 * x @ H @ x + f @ x,
    np.maximum(d["initial"][::3], 0) / scale,
    jac=lambda x: H @ x + f,
    method="SLSQP",
    bounds=[(0, None)] * n,
    options=dict(maxiter=200, ftol=1e-15),
)
normal = result.x * scale
lam = np.zeros(3 * n)
lam[::3] = normal
stages = []
for alpha in (0.0, 1.0) if args.direct else (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    evaluate, _, _, _ = natural_map_evaluator(A, q, gamma, mu * alpha)
    if alpha == 0:
        err = float(np.max(np.abs(evaluate(lam)[0])))
        stages.append(dict(alpha=0.0, error=err, normal_iterations=int(result.nit), status=str(result.message)))
    # Contact activity is selected from current equations, all other rows remain
    # explicit zero and are checked after each solve. A contact can re-enter.
    for outer in range(6):
        gradient = (A @ lam + q).reshape(-1, 3)
        gradient[:, 0] += gamma * lam[::3]
        active = (lam[::3] > 1e-12) | (gradient[:, 0] < -1e-9)
        points = np.flatnonzero(active)
        ids = (3 * points[:, None] + np.arange(3)).ravel()
        base = np.zeros_like(lam)

        def expanded(x):
            z = base.copy()
            z[ids] = x
            return z

        def residual(x):
            return evaluate(expanded(x))[0][ids] * 1e4

        def jacobian(x):
            return evaluate(expanded(x))[1][np.ix_(ids, ids)] * 1e4

        lower = np.full(len(ids), -np.inf)
        lower[::3] = 0
        seed = lam[ids].copy()
        seed[::3] = np.maximum(seed[::3], 1e-14)
        solved = least_squares(
            residual,
            seed,
            jac=jacobian,
            bounds=(lower, np.inf),
            x_scale="jac",
            max_nfev=80,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
        )
        lam = expanded(solved.x)
        error = float(np.max(np.abs(evaluate(lam)[0])))
        stages.append(
            dict(
                alpha=alpha,
                outer=outer,
                active_points=points.tolist(),
                evaluations=solved.nfev,
                error=error,
                status=str(solved.message),
            )
        )
        if error < 1e-8:
            break
        # Opening rows leave the active set; inactive violating normals re-enter.
        # If activity and residual stall, do not spend another identical solve.
        gradient = (A @ lam + q).reshape(-1, 3)
        gradient[:, 0] += gamma * lam[::3]
        new = (lam[::3] > 1e-12) | (gradient[:, 0] < -1e-9)
        if np.array_equal(new, active):
            break
    if error >= 1e-8:
        break
evaluate, *_ = natural_map_evaluator(A, q, gamma, mu)
report = dict(
    stages=stages,
    final_physical_error=float(np.max(np.abs(evaluate(lam)[0]))),
    final_alpha=alpha,
    accepted=bool(alpha == 1 and error < 1e-8),
    scope="Online normal convex seed and friction homotopy; all-point original natural-map acceptance; no oracle face.",
)
np.savez(output + ".npz", solution=lam, **{k: d[k] for k in d.files if k != "solution"})
Path(output + ".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
