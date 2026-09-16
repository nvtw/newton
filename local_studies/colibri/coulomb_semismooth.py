"""Offline non-associated Coulomb reference with explicit projection Jacobian."""

import numpy as np


def natural_map_evaluator(matrix, rhs, normal_regularization, friction=0.5):
    """Return the original physical residual and exact piecewise Jacobian."""
    count = len(rhs) // 3
    friction = np.broadcast_to(np.asarray(friction, dtype=np.float64), (count,))
    operator = matrix.copy()
    normals = np.arange(0, len(rhs), 3)
    operator[normals, normals] += normal_regularization
    scale = np.repeat(np.diag(operator).reshape(-1, 3).max(axis=1), 3)

    def evaluate(value):
        gradient = operator @ value + rhs
        trial = value - gradient / scale
        residual = np.zeros_like(value)
        jacobian = np.zeros_like(operator)
        for k in range(count):
            normal = 3 * k
            tangent = slice(normal + 1, normal + 3)
            if trial[normal] > 0:
                residual[normal] = gradient[normal]
                jacobian[normal] = operator[normal]
            else:
                residual[normal] = scale[normal] * value[normal]
                jacobian[normal, normal] = scale[normal]
            radius = friction[k] * max(value[normal], 0.0)
            length = np.linalg.norm(trial[tangent])
            if length <= radius:
                residual[tangent] = gradient[tangent]
                jacobian[tangent] = operator[tangent]
            else:
                direction = trial[tangent] / max(length, 1e-30)
                projection = radius / max(length, 1e-30) * (np.eye(2) - np.outer(direction, direction))
                trial_jacobian = -operator[tangent] / scale[tangent, None]
                trial_jacobian[:, tangent] += np.eye(2)
                jacobian[tangent] = -scale[tangent, None] * (projection @ trial_jacobian)
                jacobian[normal + 1, normal + 1] += scale[normal + 1]
                jacobian[normal + 2, normal + 2] += scale[normal + 2]
                if value[normal] > 0:
                    jacobian[tangent, normal] -= scale[tangent] * friction[k] * direction
                residual[tangent] = scale[tangent] * (value[tangent] - radius * direction)
        return residual, jacobian

    return evaluate, scale, operator, friction


def solve_coulomb(matrix, rhs, initial, normal_regularization, friction=0.5, residual_scale=1.0):
    """Try bounded and independently measurable roots; never alter live defaults."""
    from scipy.optimize import least_squares, root

    normals = np.arange(0, len(rhs), 3)
    evaluate, _scale, _operator, friction = natural_map_evaluator(matrix, rhs, normal_regularization, friction)

    lower = np.full(len(rhs), -np.inf)
    lower[normals] = 0.0
    reports = []
    best = initial.copy()
    best_error = float(np.max(np.abs(evaluate(best)[0])))
    for name, source in (("warm", initial), ("zero", np.zeros_like(initial))):
        seed = source.astype(np.float64).copy()
        seed[normals] = np.maximum(seed[normals], 1e-12)
        result = least_squares(
            lambda value: residual_scale * evaluate(value)[0],
            seed,
            jac=lambda value: residual_scale * evaluate(value)[1],
            bounds=(lower, np.inf),
            x_scale="jac",
            max_nfev=500,
            ftol=1e-13,
            gtol=1e-13,
            xtol=1e-13,
        )
        error = float(np.max(np.abs(evaluate(result.x)[0])))
        triples = result.x.reshape(-1, 3)
        cone = float(np.max(np.linalg.norm(triples[:, 1:], axis=1) - friction * triples[:, 0]))
        reports.append({"seed": name, "evaluations": int(result.nfev), "residual": error, "cone_violation": cone})
        if error < best_error:
            best, best_error = result.x, error
        if error < 1e-8 and cone < 1e-8:
            break
    if best_error >= 1e-8:
        for name, source in (("zero", np.zeros_like(initial)), ("best", best)):
            result = root(
                lambda value: evaluate(value)[0],
                source,
                jac=lambda value: evaluate(value)[1],
                method="hybr",
                options={"xtol": 1e-11, "maxfev": 300},
            )
            error = float(np.max(np.abs(evaluate(result.x)[0])))
            triples = result.x.reshape(-1, 3)
            cone = float(np.max(np.linalg.norm(triples[:, 1:], axis=1) - friction * triples[:, 0]))
            reports.append(
                {
                    "seed": name,
                    "method": "unconstrained_hybrid",
                    "evaluations": int(result.nfev),
                    "residual": error,
                    "cone_violation": cone,
                }
            )
            if error < best_error:
                best, best_error = result.x, error
            if error < 1e-8 and cone < 1e-8:
                break
    return best, reports
