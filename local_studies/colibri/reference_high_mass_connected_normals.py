"""Frozen connected normal active-set reference; never changes live simulation."""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import solve_coulomb
from newton._src.solvers.phoenx.constraints import constraint_contact as schema


def residual(matrix, rhs, impulse):
    gradient = matrix @ impulse + rhs
    return float(np.max(np.where(impulse > 1e-9, np.abs(gradient), np.maximum(-gradient, 0.0))))


def active_set(matrix, rhs):
    n = len(rhs)
    best = None
    for bits in itertools.product((False, True), repeat=n):
        active = np.flatnonzero(bits)
        value = np.zeros(n)
        if len(active):
            try:
                value[active] = np.linalg.solve(matrix[np.ix_(active, active)], -rhs[active])
            except np.linalg.LinAlgError:
                continue
        if np.min(value) < -1e-9 or np.min(matrix @ value + rhs) < -1e-8:
            continue
        objective = 0.5 * value @ matrix @ value + rhs @ value
        if best is None or objective < best[0]:
            best = (objective, value)
    if best is None:
        raise RuntimeError("No feasible active set found")
    return best[1]


def tangent_step(matrix, gradient, old, radius):
    target = matrix @ old - gradient
    value = np.linalg.solve(matrix, target)
    if np.linalg.norm(value) <= radius:
        return value
    if radius <= 0:
        return np.zeros(2)
    lower, upper = 0.0, np.linalg.norm(target) / radius
    for _ in range(60):
        alpha = 0.5 * (lower + upper)
        value = np.linalg.solve(matrix + alpha * np.eye(2), target)
        if np.linalg.norm(value) > radius:
            lower = alpha
        else:
            upper = alpha
    return np.linalg.solve(matrix + upper * np.eye(2), target)


def main():
    from scipy.optimize import least_squares, minimize, root

    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", choices=("original", "normal-only", "none"), default="original")
    parser.add_argument("--output", default="/tmp/high_mass_connected_reference")
    parser.add_argument("--normal-scale", type=float, default=1.0)
    parser.add_argument("--seed")
    parser.add_argument("--snapshot", default="/tmp/high_mass_connected_normal.npz")
    parser.add_argument("--relax", action="store_true")
    args = parser.parse_args()
    d = np.load(args.snapshot)
    positions = d["positions"].astype(np.float64)
    inv_m = d["inverse_mass"].astype(np.float64)
    inv_i = d["inverse_inertia"].astype(np.float64)
    bodies = len(inv_m)
    columns = d["headers"].view(np.int32)
    count = d["impulses"].shape[1]
    J = np.zeros((3 * count, 6 * bodies))
    mobility = np.zeros((6 * bodies, 6 * bodies))
    u = np.concatenate((d["velocity"], d["angular_velocity"]), axis=1).astype(np.float64).ravel()
    for b in range(bodies):
        mobility[6 * b : 6 * b + 3, 6 * b : 6 * b + 3] = inv_m[b] * np.eye(3)
        mobility[6 * b + 3 : 6 * b + 6, 6 * b + 3 : 6 * b + 6] = inv_i[b]
    points = []
    for cid in range(int(d["column_count"][0])):
        b0, b1 = columns[int(schema._OFF_BODY1), cid], columns[int(schema._OFF_BODY2), cid]
        first, n = columns[int(schema._OFF_CONTACT_FIRST), cid], columns[int(schema._OFF_CONTACT_COUNT), cid]
        for k in range(first, first + n):
            normal, tangent = d["lambdas"][:3, k], d["lambdas"][3:6, k]
            r0, r1 = d["derived"][9:12, k], d["derived"][12:15, k]
            points.append(np.linalg.norm(positions[b0] + r0 - positions[b1] - r1))
            for axis, direction in enumerate((normal, tangent, np.cross(normal, tangent))):
                row = 3 * k + axis
                J[row, 6 * b0 : 6 * b0 + 3] = -direction
                J[row, 6 * b0 + 3 : 6 * b0 + 6] = -np.cross(r0, direction)
                J[row, 6 * b1 : 6 * b1 + 3] = direction
                J[row, 6 * b1 + 3 : 6 * b1 + 6] = np.cross(r1, direction)
    selected_points = np.flatnonzero(d["derived"][3] <= 0) if args.relax else np.arange(count)
    selected_rows = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in selected_points])
    J = J[selected_rows]
    count = len(selected_points)
    cached_eff_n = d["derived"][0, selected_points]
    A = J @ mobility @ J.T
    initial = d["impulses"][:, selected_points].T.astype(np.float64).ravel()
    bias = np.stack((d["derived"][3], d["derived"][4], d["derived"][5]), axis=1).ravel()[selected_rows]
    if args.relax:
        bias[:] = 0.0
    if args.recovery == "normal-only":
        bias.reshape(-1, 3)[:, 1:] = 0.0
    elif args.recovery == "none":
        bias[:] = 0.0
    bias.reshape(-1, 3)[:, 0] *= args.normal_scale
    rhs = J @ u + bias - A @ initial
    normals = np.arange(0, 3 * count, 3)
    tangents = np.setdiff1d(np.arange(3 * count), normals)
    regularization = np.where(d["derived"][3] <= 0, (0.05829954519867897 / 0.9417003989219666) / d["derived"][0], 0.0)
    regularization = regularization[selected_points]
    if args.relax:
        regularization[:] = 0.0
    B = A[np.ix_(normals, normals)] + np.diag(regularization)
    q = rhs[normals] + A[np.ix_(normals, tangents)] @ initial[tangents]
    scalar = initial[normals].copy()
    for _ in range(8):
        for i in range(count):
            scalar[i] = max(0, scalar[i] - (B[i] @ scalar + q[i]) / B[i, i])
    exact = active_set(B, q)
    report = {
        "recovery": args.recovery,
        "relax": args.relax,
        "selected_points": selected_points.tolist(),
        "normal_scale": args.normal_scale,
        "normal_rows": count,
        "dynamic_body_dof": int(6 * np.count_nonzero(inv_m)),
        "common_point_error_m": float(max(points)),
        "cached_normal_mobility_relative_error": float(np.max(np.abs(np.diag(A)[normals] * cached_eff_n - 1))),
        "initial_normal_residual_m_s": residual(B, q, initial[normals]),
        "scalar_8_more_sweeps_residual_m_s": residual(B, q, scalar),
        "connected_normal_residual_m_s": residual(B, q, exact),
        "normal_matrix_condition": float(np.linalg.cond(B)),
    }

    def conservation(delta):
        impulse = (J.T @ delta).reshape(bodies, 6)
        change = (mobility @ J.T @ delta).reshape(bodies, 6)
        dynamic = inv_m > 0
        # Static endpoints carry the external reaction; internal contributions
        # cancel in both world linear and angular momentum.
        linear = (change[dynamic, :3] / inv_m[dynamic, None]).sum(axis=0) + impulse[~dynamic, :3].sum(axis=0)
        angular = (
            np.cross(positions[dynamic], change[dynamic, :3] / inv_m[dynamic, None])
            + np.einsum("bij,bj->bi", np.linalg.inv(inv_i[dynamic]), change[dynamic, 3:])
        ).sum(axis=0)
        angular += (np.cross(positions[~dynamic], impulse[~dynamic, :3]) + impulse[~dynamic, 3:]).sum(axis=0)
        return {
            "linear_reaction_balance": float(np.linalg.norm(linear)),
            "angular_reaction_balance": float(np.linalg.norm(angular)),
        }

    candidate = initial.copy()
    candidate[normals] = exact
    report["normal_block_conservation"] = conservation(candidate - initial)

    def coupled_residual(value):
        ng = A[np.ix_(normals, np.arange(3 * count))] @ value + rhs[normals] + regularization * value[normals]
        nr = float(np.max(np.where(value[normals] > 1e-9, np.abs(ng), np.maximum(-ng, 0))))
        tr = 0.0
        for k in range(count):
            ids = np.array([3 * k + 1, 3 * k + 2])
            K = A[np.ix_(ids, ids)]
            gradient = A[ids] @ value + rhs[ids]
            expected = tangent_step(K, gradient, value[ids], 0.5 * value[3 * k])
            tr = max(tr, float(np.linalg.norm(K @ (expected - value[ids]))))
        return nr, tr

    histories = {}
    for block in (False, True):
        value = initial.copy()
        history = []
        for sweep in range(64):
            if block:
                value[normals] = active_set(B, rhs[normals] + A[np.ix_(normals, tangents)] @ value[tangents])
            for k in range(count):
                if not block:
                    i = 3 * k
                    gradient = A[i] @ value + rhs[i] + regularization[k] * value[i]
                    value[i] = max(0, value[i] - gradient / B[k, k])
                ids = np.array([3 * k + 1, 3 * k + 2])
                K = A[np.ix_(ids, ids)]
                value[ids] = tangent_step(K, A[ids] @ value + rhs[ids], value[ids], 0.5 * value[3 * k])
            if sweep + 1 in (1, 2, 4, 8, 16, 32, 64):
                nr, tr = coupled_residual(value)
                history.append({"sweeps": sweep + 1, "normal_residual_m_s": nr, "tangent_fixedpoint_residual_m_s": tr})
        histories["block_normal" if block else "scalar"] = history
        report["full_friction_" + ("block" if block else "scalar") + "_conservation"] = conservation(value - initial)
    report["full_friction"] = histories

    def natural_map(value):
        # Non-associated Coulomb law: normal closure has NO derivative of
        # tangent cone radius. Friction uses the existing normal load.
        gradient = A @ value + rhs
        result = np.zeros_like(value)
        ng = gradient[normals] + regularization * value[normals]
        result[normals] = np.diag(B) * (value[normals] - np.maximum(0.0, value[normals] - ng / np.diag(B)))
        for k in range(count):
            ids = np.array([3 * k + 1, 3 * k + 2])
            K = A[np.ix_(ids, ids)]
            target = tangent_step(K, gradient[ids], value[ids], 0.5 * max(0.0, value[3 * k]))
            result[ids] = K @ (value[ids] - target)
        return result

    start = time.perf_counter()
    solver_initial = np.load(args.seed)["coupled_solution"] if args.seed else initial
    solver_initial = solver_initial.copy()
    solver_initial[normals] = np.maximum(solver_initial[normals], 0.0)
    first = root(natural_map, solver_initial, method="hybr", options={"xtol": 1.0e-10, "maxfev": 1000})
    first_report = {
        "success": bool(first.success),
        "nfev": int(first.nfev),
        "residual": float(np.max(np.abs(natural_map(first.x)))),
    }
    lower = np.full_like(initial, -np.inf)
    lower[normals] = 0.0
    solved = least_squares(
        natural_map,
        solver_initial,
        bounds=(lower, np.inf),
        method="trf",
        max_nfev=300,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        x_scale="jac",
    )
    solution = solved.x.copy()
    intermediate_gradient = A @ solution + rhs
    normal_gradient = intermediate_gradient[normals] + regularization * solution[normals]
    # Select the normal face from the natural-map branch, rather than
    # classifying arbitrarily small positive least-squares impulses as active.
    active_contacts = np.flatnonzero(np.diag(B) * solution[normals] - normal_gradient > 0.0)
    active_variables = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in active_contacts])

    def reduced_map(values):
        full = np.zeros_like(initial)
        full[active_variables] = values
        return natural_map(full)[active_variables]

    refined = root(reduced_map, solution[active_variables], method="hybr", options={"xtol": 1.0e-10, "maxfev": 1000})
    solution = np.zeros_like(initial)
    solution[active_variables] = refined.x
    # Once contact/stick/slide faces are identified, solve their exact KKT
    # equations with an analytic Jacobian. The normal rows remain separate
    # from friction multipliers (non-associated Coulomb, not a cone QP).
    sliding = [
        k for k in active_contacts if np.linalg.norm(solution[3 * k + 1 : 3 * k + 3]) >= 0.5 * solution[3 * k] - 1e-6
    ]
    active_lookup = {int(v): i for i, v in enumerate(active_variables)}

    def kkt_system(unknown):
        full = np.zeros_like(initial)
        full[active_variables] = unknown[: len(active_variables)]
        g = A @ full + rhs
        g[normals] += regularization * full[normals]
        equation = g[active_variables].copy()
        jacobian = np.zeros((len(unknown), len(unknown)))
        jacobian[: len(active_variables), : len(active_variables)] = A[np.ix_(active_variables, active_variables)]
        for k in active_contacts:
            loc = active_lookup[3 * k]
            jacobian[loc, loc] += regularization[k]
        extra = []
        for index, k in enumerate(sliding):
            ids = np.array([3 * k + 1, 3 * k + 2])
            local_ids = np.array([active_lookup[int(v)] for v in ids])
            multiplier = unknown[len(active_variables) + index]
            tangent = full[ids]
            equation[local_ids] += multiplier * tangent
            jacobian[local_ids, local_ids] += multiplier
            jacobian[local_ids, len(active_variables) + index] = tangent
            length = np.linalg.norm(tangent)
            extra.append(length - 0.5 * full[3 * k])
            jacobian[len(active_variables) + index, local_ids] = tangent / max(length, 1e-30)
            jacobian[len(active_variables) + index, active_lookup[3 * k]] = -0.5
        return np.r_[equation, extra], jacobian

    seed_solution = solution.copy()
    full_compliant = A.copy()
    full_compliant[normals, normals] += regularization
    sticking_matrix = full_compliant[np.ix_(active_variables, active_variables)]
    eigenvalues, eigenvectors = np.linalg.eigh(sticking_matrix)
    null_vectors = eigenvectors[:, eigenvalues < 1e-10 * np.max(eigenvalues)]
    null_bias = float(np.linalg.norm(null_vectors.T @ (A @ seed_solution + rhs)[active_variables]))
    # Redundant sticking contacts admit a family of impulse distributions.
    # Search that nullspace for Coulomb feasibility; this is not an associated
    # cone velocity solve because every sticking velocity is constrained to zero.
    particular = np.linalg.lstsq(sticking_matrix, -rhs[active_variables], rcond=1e-10)[0]

    def sticking_candidate(z):
        full = np.zeros_like(initial)
        full[active_variables] = particular + null_vectors @ z
        return full

    def sticking_inequalities(z):
        full = sticking_candidate(z)
        triples = full.reshape(-1, 3)[active_contacts]
        inactive = np.setdiff1d(np.arange(count), active_contacts)
        gradient = A @ full + rhs
        return np.r_[0.5 * triples[:, 0] - np.linalg.norm(triples[:, 1:], axis=1), gradient[normals[inactive]]]

    sticking = (
        minimize(
            lambda z: 0.5 * z @ z,
            np.zeros(null_vectors.shape[1]),
            jac=lambda z: z,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": sticking_inequalities}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if null_vectors.shape[1]
        else None
    )
    sticking_value = sticking_candidate(sticking.x if sticking is not None else np.zeros(0))
    sticking_ok = bool(
        np.max(np.abs(natural_map(sticking_value))) < 1e-8
        and np.min(sticking_inequalities(sticking.x if sticking is not None else np.zeros(0))) >= -1e-8
    )
    attempts = 0
    accepted = False
    # Enumerate friction faces only on this bounded eight-contact reference.
    # Every candidate is independently checked against all original rows.
    for size in range(1, len(active_contacts) + 1):
        for subset in itertools.combinations(active_contacts, size):
            sliding = list(subset)
            kkt_start = np.r_[seed_solution[active_variables], np.full(len(sliding), 1e-4)]
            kkt = root(
                lambda x: kkt_system(x)[0],
                kkt_start,
                jac=lambda x: kkt_system(x)[1],
                method="hybr",
                options={"xtol": 1e-10, "maxfev": 300},
            )
            attempts += 1
            candidate = np.zeros_like(initial)
            candidate[active_variables] = kkt.x[: len(active_variables)]
            if not np.isfinite(candidate).all():
                continue
            kkt_error = float(np.max(np.abs(kkt_system(kkt.x)[0])))
            cone_error = float(
                np.max(np.linalg.norm(candidate.reshape(-1, 3)[:, 1:], axis=1) - 0.5 * candidate[normals])
            )
            normal_gradient = (A @ candidate + rhs)[normals] + regularization * candidate[normals]
            complementarity = float(np.max(np.abs(np.minimum(np.diag(B) * candidate[normals], normal_gradient))))
            if (
                kkt_error < 1e-8
                and cone_error < 1e-8
                and complementarity < 1e-8
                and np.min(kkt.x[len(active_variables) :]) >= -1e-8
            ):
                solution = candidate
                accepted = True
                break
        if accepted:
            break
    if not accepted:
        solution = sticking_value if sticking_ok else seed_solution
    semismooth_reports = []
    if np.max(np.abs(natural_map(solution))) >= 1e-8:
        candidate, semismooth_reports = solve_coulomb(A, rhs, solution, regularization)
        if np.max(np.abs(natural_map(candidate))) < np.max(np.abs(natural_map(solution))):
            solution = candidate
    # Preserve an independently supplied physical-mode seed if later local
    # root refinements land on a worse active face. Acceptance below remains
    # the original non-associated Coulomb test.
    if args.seed and np.max(np.abs(natural_map(solver_initial))) < np.max(np.abs(natural_map(solution))):
        solution = solver_initial.copy()
    gradient = A @ solution + rhs
    ng = gradient[normals] + regularization * solution[normals]
    friction_kkt = []
    for k in range(count):
        tangent = solution[3 * k + 1 : 3 * k + 3]
        slip = gradient[3 * k + 1 : 3 * k + 3]
        radius = 0.5 * max(0.0, solution[3 * k])
        if radius <= 1.0e-9:
            error = np.linalg.norm(tangent)
        elif np.linalg.norm(tangent) < radius - 1.0e-8:
            error = np.linalg.norm(slip)
        else:
            multiplier = max(0.0, -float(tangent @ slip) / float(tangent @ tangent))
            error = np.linalg.norm(slip + multiplier * tangent)
        friction_kkt.append(float(error))
    updated = u + mobility @ J.T @ (solution - initial)
    physical_mass = np.linalg.pinv(mobility)
    free_velocity = u - mobility @ J.T @ initial
    fd_jacobian = np.zeros((len(kkt.x), len(kkt.x)))
    for column in range(len(kkt.x)):
        h = 1e-6 * max(1.0, abs(kkt.x[column]))
        perturb = np.zeros_like(kkt.x)
        perturb[column] = h
        fd_jacobian[:, column] = (kkt_system(kkt.x + perturb)[0] - kkt_system(kkt.x - perturb)[0]) / (2 * h)
    exact_jacobian = kkt_system(kkt.x)[1]
    report["coupled_natural_map"] = {
        "unglobalized_hybrid": first_report,
        "semismooth_restarts": semismooth_reports,
        "optimizer_terminated": bool(solved.success),
        "physical_kkt_accepted": bool(
            np.max(np.abs(natural_map(solution))) < 1e-8
            and max(friction_kkt) < 1e-8
            and np.min(solution[normals]) >= -1e-8
            and np.max(np.linalg.norm(solution.reshape(-1, 3)[:, 1:], axis=1) - 0.5 * solution[normals]) < 1e-8
        ),
        "message": str(solved.message),
        "function_evaluations": int(solved.nfev),
        "active_contacts": active_contacts.tolist(),
        "active_face_refinement_success": bool(refined.success),
        "analytic_kkt_success": bool(accepted),
        "all_sticking_nullspace_feasible": sticking_ok,
        "friction_face_attempts": attempts,
        "sticking_nullity": int(null_vectors.shape[1]),
        "sticking_null_bias_norm_m_s": null_bias,
        "analytic_kkt_evaluations": int(kkt.nfev),
        "last_analytic_trial_sliding_contacts": list(map(int, sliding)),
        "last_analytic_trial_minimum_sliding_multiplier": float(np.min(kkt.x[len(active_variables) :], initial=np.inf)),
        "active_face_refinement_evaluations": int(refined.nfev),
        "cpu_reference_seconds": time.perf_counter() - start,
        "max_fixedpoint_velocity_residual_m_s": float(np.max(np.abs(natural_map(solution)))),
        "normal_complementarity_residual_m_s": float(
            np.max(np.where(solution[normals] > 1e-9, np.abs(ng), np.maximum(-ng, 0.0)))
        ),
        "minimum_normal_impulse_Ns": float(np.min(solution[normals])),
        "normal_natural_complementarity_m_s": float(np.max(np.abs(np.minimum(np.diag(B) * solution[normals], ng)))),
        "normal_complementarity_product_J": float(np.max(np.abs(solution[normals] * ng))),
        "max_friction_disk_violation_Ns": float(
            np.max(np.linalg.norm(solution.reshape(-1, 3)[:, 1:], axis=1) - 0.5 * solution[normals])
        ),
        "max_friction_kkt_residual_m_s": max(friction_kkt),
        "max_impulse_Ns": float(np.max(np.abs(solution))),
        "analytic_trial_jacobian_relative_error": float(
            np.linalg.norm(fd_jacobian - exact_jacobian) / max(1.0, np.linalg.norm(exact_jacobian))
        ),
        "kinetic_energy_free_J": float(0.5 * free_velocity @ physical_mass @ free_velocity),
        "physical_contact_work_J": float(solution @ (J @ free_velocity) + 0.5 * solution @ A @ solution),
        "kinetic_energy_before_J": float(0.5 * u @ physical_mass @ u),
        "kinetic_energy_after_J": float(0.5 * updated @ physical_mass @ updated),
        "conservation": conservation(solution - initial),
    }
    np.savez(
        args.output + ".npz",
        J=J,
        inverse_mass=mobility,
        A=A,
        B=B,
        rhs=rhs,
        initial=initial,
        normal_solution=exact,
        normal_rhs=q,
        coupled_solution=solution,
        normal_regularization=regularization,
        selected_points=selected_points,
        updated_velocity=updated.reshape(bodies, 6),
        physical_free_velocity=free_velocity.reshape(bodies, 6),
    )
    Path(args.output + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
