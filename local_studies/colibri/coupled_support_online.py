"""Reusable bounded physical normal seed and all-point Coulomb outer."""

import numpy as np
from scipy.optimize import minimize, least_squares, root
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator



def polish_active_newton(evaluate, value, ids):
    """Cross a stiff friction mode transition without regularizing resolved modes."""
    value = value.copy()
    history = []
    for iteration in range(8):
        residual, jacobian = evaluate(value)
        error = float(np.max(np.abs(residual)))
        if error < 1e-10:
            break
        matrix = jacobian[np.ix_(ids, ids)]
        try:
            delta = np.linalg.solve(matrix, -residual[ids])
        except np.linalg.LinAlgError:
            break
        linear_error = float(np.max(np.abs(matrix @ delta + residual[ids])))
        if not np.isfinite(delta).all() or linear_error > 1e-10:
            break
        alpha = 1.0
        closing = delta[::3] < 0
        if np.any(closing):
            alpha = min(alpha, float(np.min(-.99 * value[ids][::3][closing] / delta[::3][closing])))
        merit = residual @ residual
        accepted = False
        for backtrack in range(20):
            trial = value.copy()
            trial[ids] += alpha * delta
            trial_residual = evaluate(trial)[0]
            if trial_residual @ trial_residual < merit * (1 - 1e-4 * alpha):
                value = trial
                accepted = True
                break
            alpha *= .5
        history.append(dict(iteration=iteration, residual=error, alpha=alpha,
                            linear_error=linear_error, accepted=accepted))
        if not accepted:
            break
    return value, history



def cross_contact_boundary(evaluate, value, ids, physical_jacobian, operator, rhs, scale, friction):
    """Explore a friction boundary along a weak response direction, retaining it."""
    trial = value - (operator @ value + rhs) / scale
    sticking = np.flatnonzero(
        (value[::3] > 1e-12)
        & (np.linalg.norm(trial.reshape(-1,3)[:,1:],axis=1) <= friction * value[::3])
    )
    if len(sticking) < 2:
        return value, []
    feasible = value.copy()
    for point in range(len(friction)):
        tangent = feasible[3*point+1:3*point+3]
        radius = friction[point] * max(feasible[3*point],0.)
        length = np.linalg.norm(tangent)
        if length > radius:
            tangent *= radius / length
    tangents = (3*sticking[:,None]+np.array([1,2])).ravel()
    # A search direction only: every full physical response mode is retained.
    _, singular, vt = np.linalg.svd(physical_jacobian.T[:,tangents])
    direction = np.zeros_like(value)
    direction[tangents] = vt[-1]
    best = value.copy()
    best_error = float(np.max(np.abs(evaluate(best)[0])))
    records = []
    for sign in (-1.,1.):
        z = direction * sign
        limits = []
        for point in sticking:
            t = feasible[3*point+1:3*point+3]
            dz = z[3*point+1:3*point+3]
            radius = friction[point] * feasible[3*point]
            roots = np.roots([dz@dz,2*t@dz,t@t-radius*radius])
            positive = [float(r.real) for r in roots if abs(r.imag)<1e-15 and r.real>1e-15]
            if positive:
                limits.append(min(positive))
        if not limits:
            continue
        candidate = feasible + min(limits) * z
        candidate, polish = polish_active_newton(evaluate,candidate,ids)
        unit = max(float(np.max(np.abs(candidate[ids]))),1e-12)
        def expand(x):
            out = np.zeros_like(value)
            out[ids] = unit*x
            return out
        solved = root(lambda x:evaluate(expand(x))[0][ids],candidate[ids]/unit,
            jac=lambda x:evaluate(expand(x))[1][np.ix_(ids,ids)]*unit,
            method="hybr",options=dict(maxfev=120,xtol=1e-11))
        candidate = expand(solved.x)
        error = float(np.max(np.abs(evaluate(candidate)[0])))
        records.append(dict(sign=sign,search_singular_values=singular.tolist(),
                            evaluations=int(solved.nfev),error=error,polish=polish))
        if error < best_error and np.min(candidate[::3]) >= -1e-12:
            best,best_error = candidate,error
        if best_error < 1e-8:
            break
    return best,records


def solve_online(A, q, gamma, mu, initial, physical_jacobian=None):
    n = len(mu)
    if n == 0:
        return np.zeros(0), dict(accepted=True, final_physical_error=0.0, stages=[], scope="Joint-only phase; no eligible contact rows")
    N = A[::3, ::3] + np.diag(gamma)
    qn = q[::3]
    scale = 1 / np.sqrt(np.diag(N))
    H = N * scale[:, None] * scale[None, :]
    f = qn * scale
    result = minimize(
        lambda x: 0.5 * x @ H @ x + f @ x,
        np.maximum(initial[::3], 0) / scale,
        jac=lambda x: H @ x + f,
        method="SLSQP",
        bounds=[(0, None)] * n,
        options=dict(maxiter=200, ftol=1e-15),
    )
    normal = result.x * scale
    lam = np.zeros(3 * n)
    lam[::3] = normal
    stages = []
    for alpha in (0.0, 1.0):
        evaluate, natural_scale, physical_operator, friction = natural_map_evaluator(A, q, gamma, mu * alpha)
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
            polish_history = []
            if error >= 1e-8:
                lam, polish_history = polish_active_newton(evaluate, lam, ids)
                error = float(np.max(np.abs(evaluate(lam)[0])))
            boundary_history = []
            if error >= 1e-8 and physical_jacobian is not None:
                lam,boundary_history = cross_contact_boundary(
                    evaluate,lam,ids,physical_jacobian,physical_operator,q,natural_scale,friction)
                error = float(np.max(np.abs(evaluate(lam)[0])))
            stages.append(
                dict(
                    alpha=alpha,
                    outer=outer,
                    active_points=points.tolist(),
                    evaluations=solved.nfev,
                    newton_polish=polish_history,
                    boundary_search=boundary_history,
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
    return lam, report


def assemble_snapshot(d, phase, dt, num_joints):
    """Rebuild all current two-body response rows and actual phase targets."""
    h, jd = d["headers"].view(np.int32), d["joint_data"].view(np.int32)
    support = 1
    joints = [j for j in range(int(num_joints)) if support in jd[1:3, j]]
    bodies = sorted({int(b) for j in joints for b in jd[1:3, j] if d["inverse_mass"][b] > 0})
    index = {b: i for i, b in enumerate(bodies)}
    size = len(bodies) * 6
    W = np.zeros((size, size))
    velocity = np.zeros(size)
    for b, i in index.items():
        xx, yy, zz, xy, xz, yz = d["inverse_inertia"][b].astype(float)
        W[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = np.eye(3) * d["inverse_mass"][b]
        W[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        start, end = int(d["copy_section_end"][b - 1]), int(d["copy_section_end"][b])
        velocity[6 * i : 6 * i + 6] = np.r_[
            d["copy_velocity"][start:end].astype(float).mean(0),
            d["copy_angular_velocity"][start:end].astype(float).mean(0),
        ]
    B, diagonal, targets, old_joint = [], [], [], []
    for j in joints:
        st = int(d["joint_structural_index"][j])
        for row in d["joint_row_indices"][j, : int(d["joint_row_count"][j])]:
            local = int(d["joint_row_local"][row])
            jac = np.zeros(size)
            for body, key in zip(jd[1:3, j], ("joint_wrench0", "joint_wrench1"), strict=True):
                if body in index:
                    i = index[body]
                    jac[6 * i : 6 * i + 6] += d[key][st, local]
            dynamic = bool(d["joint_row_dynamic"][row])
            B.append(jac)
            diagonal.append(1 / float(d["joint_dynamic_mass"][row]) if dynamic else 0)
            targets.append(
                float(d["joint_reference"][row])
                if dynamic
                else (-float(d["joint_bias"][st, local]) if phase == "biased" else 0.0)
            )
            old_joint.append(float(d["joint_accumulated"][row]))
    B, diagonal, targets, old_joint = map(np.asarray, (B, diagonal, targets, old_joint))
    C, points, mu, old, bias, regularization = [], [], [], [], [], []
    ground_wrenches = []
    mc, ic = 0.9417003989219666, 0.05829954519867897
    for col in range(int(d["column_count"][0])):
        a, b = h[1:3, col]
        if 0 not in (a, b) or not (a in index or b in index):
            continue
        for p in range(h[5, col], h[5, col] + h[6, col]):
            if phase == "relax" and d["derived"][3, p] > 0:
                continue  # Native relax holds these exact captured impulses fixed.
            assert d["derived"][8, p] <= 0, "Explicit contact PD needs its own captured equation"
            assert d["headers"][3, col] == d["headers"][4, col], "Distinct static/dynamic cones require explicit branch"
            n, t = d["lambdas"][:3, p].astype(float), d["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            jac = np.zeros((3, size))
            ground = np.zeros((3, 6))
            for body, sign, start in ((a, -1, 9), (b, 1, 12)):
                wrench = sign * np.c_[axes, np.cross(d["derived"][start : start + 3, p].astype(float), axes)]
                if body in index:
                    i = index[body]
                    jac[:, 6 * i : 6 * i + 6] += wrench
                else:
                    ground += wrench
            C.extend(jac)
            ground_wrenches.append(ground)
            points.append(int(p))
            mu.append(float(d["headers"][3, col]))
            old.extend(d["impulses"][:, p].astype(float))
            bias.append(float(d["derived"][3, p]))
            regularization.append(
                0.0 if phase == "relax" or d["derived"][3, p] > 0 else ic / (mc * float(d["derived"][0, p]))
            )
            if d["derived"][3, p] > float(dt) ** -1 * 0.002:
                mu[-1] = 0.0
    C, old, mu, bias, regularization = map(np.asarray, (C, old, mu, bias, regularization))
    C = C.reshape(-1, size)
    # Remove TOTAL owned impulse exactly once; retain captured unowned reactions.
    free = velocity - W @ (C.T @ old + B.T @ old_joint)
    K = B @ W @ B.T + np.diag(diagonal)
    vbar = free + W @ B.T @ np.linalg.solve(K, targets - B @ free)
    P = W - W @ B.T @ np.linalg.solve(K, B @ W)
    A = C @ P @ C.T
    rhs = C @ vbar
    if phase == "biased":
        rhs += d["derived"][3:6, points].T.astype(float).ravel()
    return dict(
        W=W,
        B=B,
        C=C,
        free=free,
        K=K,
        vbar=vbar,
        P=P,
        A=A,
        rhs=rhs,
        old=old,
        old_joint=old_joint,
        diagonal=diagonal,
        targets=targets,
        mu=mu,
        gamma=regularization,
        points=points,
        bodies=bodies,
        joints=joints,
        velocity=velocity,
        ground_wrenches=ground_wrenches,
    )
