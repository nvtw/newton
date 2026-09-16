"""Bounded CPU small-island Newton and neighboring-face experiment.

All trial states are private. Original non-associated residuals decide success.
Response coordinates keep the mass-weighted Jacobian explicit in linear solves.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def response_factor(J, W):
    """Factor only positive physical body blocks, keeping every dynamic DOF."""
    cols = []
    for start in range(0, len(W), 6):
        block = W[start : start + 6, start : start + 6]
        if np.all(block == 0):
            continue
        L = np.linalg.cholesky(block)
        cols.append(J[:, start : start + 6] @ L)
    return np.concatenate(cols, axis=1)


def attempt(
    F, rhs, initial, mu, modes=None, steps=12, numerical_epsilon=0.0, numerical_reference=None, audit_evaluator=None
):
    """Use response-coordinate augmented equations and bounded Armijo steps."""
    m, nv = F.shape
    count = m // 3
    scale = np.repeat(np.sum(F * F, axis=1).reshape(-1, 3).max(axis=1), 3)
    proximal = numerical_epsilon * scale
    reference = initial if numerical_reference is None else numerical_reference
    sliding = [] if modes is None else list(np.flatnonzero(modes == 2))
    x = np.r_[F.T @ initial, initial, np.zeros(len(sliding))]
    for j, k in enumerate(sliding):
        t = initial[3 * k + 1 : 3 * k + 3]
        g = F[3 * k + 1 : 3 * k + 3] @ x[:nv] + rhs[3 * k + 1 : 3 * k + 3]
        g = g + proximal[3 * k + 1 : 3 * k + 3] * (initial[3 * k + 1 : 3 * k + 3] - reference[3 * k + 1 : 3 * k + 3])
        x[nv + m + j] = max(0, -float(t @ g) / max(float(t @ t), 1e-300))

    residual_evaluations = 0

    def evaluate(x):
        nonlocal residual_evaluations
        residual_evaluations += 1
        y, value = x[:nv], x[nv : nv + m]
        gradient = F @ y + rhs + proximal * (value - reference)
        residual = np.zeros(len(x))
        jac = np.zeros((len(x), len(x)))
        residual[:nv] = y - F.T @ value
        jac[:nv, :nv] = np.eye(nv)
        jac[:nv, nv : nv + m] = -F.T
        for k in range(count):
            n = 3 * k
            t = slice(n + 1, n + 3)
            row = nv + n
            if modes is None:
                trial = value[n : n + 3] - gradient[n : n + 3] / scale[n]
                if trial[0] > 0:
                    residual[row] = gradient[n]
                    jac[row, :nv] = F[n]
                    jac[row, nv + n] = proximal[n]
                else:
                    residual[row] = scale[n] * value[n]
                    jac[row, nv + n] = scale[n]
                radius = mu[k] * max(0, value[n])
                length = np.linalg.norm(trial[1:])
                if length <= radius:
                    residual[row + 1 : row + 3] = gradient[t]
                    jac[row + 1 : row + 3, :nv] = F[t]
                    jac[row + 1 : row + 3, nv + n + 1 : nv + n + 3] = np.diag(proximal[t])
                else:
                    direction = trial[1:] / max(length, 1e-300)
                    P = radius / max(length, 1e-300) * (np.eye(2) - np.outer(direction, direction))
                    residual[row + 1 : row + 3] = scale[n] * (value[t] - radius * direction)
                    jac[row + 1 : row + 3, :nv] = P @ F[t]
                    jac[row + 1 : row + 3, nv + n + 1 : nv + n + 3] = scale[n] * (np.eye(2) - P) + P @ np.diag(
                        proximal[t]
                    )
                    if value[n] > 0:
                        jac[row + 1 : row + 3, nv + n] = -scale[n] * mu[k] * direction
            elif modes[k] == 0:
                residual[row : row + 3] = scale[n] * value[n : n + 3]
                jac[row : row + 3, nv + n : nv + n + 3] = scale[n] * np.eye(3)
            else:
                residual[row : row + 3] = gradient[n : n + 3]
                jac[row : row + 3, :nv] = F[n : n + 3]
                jac[row : row + 3, nv + n : nv + n + 3] = np.diag(proximal[n : n + 3])
                if modes[k] == 2:
                    extra = nv + m + sliding.index(k)
                    alpha = x[extra]
                    residual[row + 1 : row + 3] += alpha * value[t]
                    jac[row + 1 : row + 3, nv + n + 1 : nv + n + 3] += alpha * np.eye(2)
                    jac[row + 1 : row + 3, extra] = value[t]
                    length = np.linalg.norm(value[t])
                    residual[extra] = scale[n] * (length - mu[k] * value[n])
                    jac[extra, nv + n] = -scale[n] * mu[k]
                    jac[extra, nv + n + 1 : nv + n + 3] = scale[n] * value[t] / max(length, 1e-300)
        return residual, jac

    if audit_evaluator is not None:
        audit_evaluator(x.copy(), evaluate)
    history = []
    for _ in range(steps):
        r, jac = evaluate(x)
        if np.max(np.abs(r)) < 1e-9:
            break
        colscale = np.maximum(np.linalg.norm(jac, axis=0), np.finfo(float).tiny)
        scaled = jac / colscale
        U, s, Vh = np.linalg.svd(scaled, full_matrices=False)
        cutoff = np.finfo(float).eps * max(scaled.shape) * s[0]
        keep = s > cutoff
        # A discarded coordinate must be response-neutral as well as a
        # machine-null equation direction. Otherwise this trial is unresolved.
        null = Vh[~keep].T / colscale[:, None]
        response_null = float(np.max(np.abs(null[:nv]), initial=0))
        if response_null > 1e-10:
            history.append({"rank": int(keep.sum()), "rejected_response_null": response_null})
            break
        dx = (Vh[keep].T @ ((U[:, keep].T @ (-r)) / s[keep])) / colscale
        derivative = float(r @ (jac @ dx))
        accepted = False
        for backtrack in range(24):
            fraction = 2.0 ** (-backtrack)
            trial = x + fraction * dx
            rr, _ = evaluate(trial)
            if float(rr @ rr) <= float(r @ r) + 2e-4 * fraction * derivative:
                accepted = True
                break
        history.append(
            {
                "rank": int(keep.sum()),
                "backtracks": backtrack,
                "linear_defect": float(np.max(np.abs(jac @ dx + r))),
                "direction_norm": float(np.linalg.norm(dx)),
                "smallest_retained": float(s[keep][-1]),
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        x = trial
    if not history:
        history.append({"already_converged": True})
    final_residual, _ = evaluate(x)
    history[-1]["face_equation_residual"] = float(np.max(np.abs(final_residual)))
    history[-1]["sliding_multipliers"] = x[nv + m :].tolist()
    history[-1]["total_residual_evaluations"] = residual_evaluations
    return x[nv : nv + m], history


def solve(d, seed=None, allow_revisits=False, face_steps=8, natural_steps=12, max_face_visits=8):
    A, J, W, rhs, initial = (d[k].astype(float) for k in ("A", "J", "inverse_mass", "rhs", "initial"))
    assert np.all(d["normal_regularization"] == 0)
    assert len(initial) <= 27
    F = response_factor(J, W)
    np.testing.assert_allclose(F @ F.T, A, atol=1e-12, rtol=1e-13)
    mu = np.full(len(initial) // 3, 0.5)
    evaluate, scale, _, _ = natural_map_evaluator(A, rhs, np.zeros(len(mu)), mu)

    def audit(value):
        r, _ = evaluate(value)
        normal = float(np.max(np.abs(np.minimum(value[::3] * np.diag(A)[::3], (A @ value + rhs)[::3]))))
        cone = float(np.max(np.linalg.norm(value.reshape(-1, 3)[:, 1:], axis=1) - mu * value[::3]))
        work = float(value @ rhs + 0.5 * value @ A @ value)
        return {
            "residual": float(np.max(np.abs(r))),
            "normal": normal,
            "cone": cone,
            "work_J": work,
            "accepted": bool(
                np.max(np.abs(r)) < 1e-8 and normal < 1e-8 and cone < 1e-8 and min(value[::3]) >= -1e-8 and work <= 1e-8
            ),
        }

    best, history = attempt(F, rhs, initial if seed is None else seed, mu, steps=natural_steps)
    reports = [dict(kind="natural", history=history, **audit(best))]
    if reports[-1]["accepted"]:
        return best, reports
    gradient = A @ best + rhs
    trial = (best - gradient / scale).reshape(-1, 3)
    modes = np.where(
        trial[:, 0] <= 0, 0, np.where(np.linalg.norm(trial[:, 1:], axis=1) <= mu * np.maximum(best[::3], 0), 1, 2)
    )
    scores = np.max(np.abs(evaluate(best)[0].reshape(-1, 3)), axis=1)
    queue = [modes.copy()]
    for k in np.argsort(-scores, kind="stable"):
        for mode in (0, 1, 2):
            if mode != modes[k]:
                neighbor = modes.copy()
                neighbor[k] = mode
                queue.append(neighbor)
    queue = [(mode, best, "neighbor") for mode in queue]
    visited = {}
    visits = 0
    skipped_revisits = 0
    revisit_decisions = []
    while queue and visits < max_face_visits:
        mode, face_seed, origin = queue.pop(0)
        key = tuple(mode)
        seed_error = float(np.max(np.abs(evaluate(face_seed)[0])))
        epsilon = np.finfo(float).eps
        gamma = len(rhs) * epsilon / (1.0 - len(rhs) * epsilon)
        noise = 32.0 * gamma * float(np.max(np.abs(A) @ np.abs(face_seed) + np.abs(rhs) + scale * np.abs(face_seed)))
        prior_error = visited.get(key)
        if key in visited:
            permit = bool(allow_revisits and seed_error < visited[key] - noise)
            revisit_decisions.append(
                {
                    "modes": [int(v) for v in key],
                    "prior_best": visited[key],
                    "candidate_seed": seed_error,
                    "noise": noise,
                    "permitted": permit,
                }
            )
            if not permit:
                skipped_revisits += 1
                continue
        visits += 1
        value, history = attempt(F, rhs, face_seed, mu, mode, steps=face_steps)
        check = audit(value)
        visited[key] = min(visited.get(key, np.inf), seed_error, check["residual"])
        reports.append(
            dict(
                kind="face",
                modes=mode.tolist(),
                origin=origin,
                skipped_revisits=skipped_revisits,
                revisit=prior_error is not None,
                seed_residual=seed_error,
                seed_noise_bound=noise,
                revisit_decisions=revisit_decisions.copy(),
                history=history,
                **check,
            )
        )
        if check["accepted"]:
            return value, reports
        if history[-1]["face_equation_residual"] < 1e-8:
            gradient = (A @ value + rhs).reshape(-1, 3)
            triples = value.reshape(-1, 3)
            new_mode = mode.copy()
            multipliers = iter(history[-1]["sliding_multipliers"])
            for k, current in enumerate(mode):
                alpha = next(multipliers) if current == 2 else 0.0
                if current != 0 and triples[k, 0] < -1e-8:
                    new_mode[k] = 0
                elif current == 0 and gradient[k, 0] < -1e-8:
                    new_mode[k] = 1
                elif current == 1 and np.linalg.norm(triples[k, 1:]) > mu[k] * triples[k, 0] + 1e-8:
                    new_mode[k] = 2
                elif current == 2 and alpha < -1e-8:
                    new_mode[k] = 1
            if not np.array_equal(new_mode, mode):
                queue.insert(0, (new_mode, value, "kkt_pivot"))
    return best, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.snapshot)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    d = np.load(path)
    start = time.perf_counter()
    value, reports = solve(d)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    result = {
        "accepted": any(r["accepted"] for r in reports),
        "seconds": time.perf_counter() - start,
        "linear_solves": sum("linear_defect" in h for r in reports for h in r["history"]),
        "factorizations": sum("rank" in h for r in reports for h in r["history"]),
        "residual_evaluations": sum(r["history"][-1]["total_residual_evaluations"] for r in reports),
        "backtrack_limit": 24,
        "input_sha256": digest,
        "attempts": reports,
    }
    Path(args.output + ".json").write_text(json.dumps(result, indent=2))
    np.savez(args.output + ".npz", solution=value)
    print(json.dumps({k: v for k, v in result.items() if k != "attempts"}))


if __name__ == "__main__":
    main()
