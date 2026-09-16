"""Frozen physical block-GS with explicit support ownership, no live edits."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from local_studies.colibri import coupled_support_reference as support
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def run(coupled=True, cycles=4, component="support"):
    from scipy.optimize import brentq

    d = dict(np.load("/tmp/colibri_support_relax330.npz"))
    sref = np.load("/tmp/colibri_coupled_support_reference.npz")
    count = len(d["inverse_mass"])
    W = np.zeros((count, 6, 6))
    W[:, :3, :3] = d["inverse_mass"][:, None, None] * np.eye(3)
    W[:, 3:, 3:] = inertia_sym6_unpack_np(d["inverse_inertia"])
    moving = d["inverse_mass"] > 0
    M = np.zeros_like(W)
    M[moving] = np.linalg.inv(W[moving])
    v = np.concatenate((d["velocity"], d["angular_velocity"]), axis=1).astype(float)
    initial_v = v.copy()
    impulse = d["impulses"].astype(float).copy()
    accumulated = d["joint_accumulated"].astype(float).copy()
    jd = d["joint_data"].view(np.int32)
    h = d["headers"].view(np.int32)
    nj = int(d["num_joints"][0])
    nc = int(d["column_count"][0])
    assert not np.any(d["joint_bounded_drive"])
    assert not np.any(d["derived"][8, : max(h[5, :nc] + h[6, :nc])])
    owned = set(sref["points"].tolist())
    component_bodies = np.array([1, 2])
    owned_joints = {0}
    component_solve = support.main
    if component == "tail":
        from local_studies.colibri.tail_component_reference import main as tail_solve

        topology = np.load("/tmp/colibri_tail_component_operator.npz")
        owned = set(topology["points"].tolist())
        component_bodies = topology["bodies"]
        owned_joints = set(topology["joints"].tolist())
        component_solve = tail_solve
    component_specs = [(component, component_bodies, owned_joints, owned, component_solve)]
    if component == "combined":
        from functools import partial

        from local_studies.colibri.tail_component_reference import main as generic_solve

        component_specs = []
        for name in ("frame_crank", "tail"):
            path = f"/tmp/colibri_{name}_component_operator.npz"
            top = np.load(path)
            component_specs.append(
                (
                    name,
                    top["bodies"],
                    set(top["joints"].tolist()),
                    set(top["points"].tolist()),
                    partial(
                        generic_solve,
                        operator=path,
                        optimizer_scale=1e4 if name == "frame_crank" else 1.0,
                        newton_fallback=name == "frame_crank",
                        friction_seed_ladder=name == "frame_crank",
                    ),
                )
            )
        assert not (component_specs[0][2] & component_specs[1][2])
        assert not (component_specs[0][3] & component_specs[1][3])
        owned_joints = set.union(*(entry[2] for entry in component_specs))
        owned = set.union(*(entry[3] for entry in component_specs))
    joints = []
    for j in range(nj):
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j]
        idx = d["joint_row_indices"][j, : d["joint_row_count"][j]].astype(int)
        loc = d["joint_row_local"][idx]
        structural = int(d["joint_structural_index"][j])
        ja = d["joint_wrench0"][structural, loc].astype(float)
        jb = d["joint_wrench1"][structural, loc].astype(float)
        dyn = d["joint_row_dynamic"][idx].astype(bool)
        compliance = np.zeros(len(idx))
        compliance[dyn] = 1 / d["joint_dynamic_mass"][idx[dyn]]
        K = ja @ W[a] @ ja.T + jb @ W[b] @ jb.T + np.diag(compliance)
        joints.append((int(a), int(b), idx, ja, jb, dyn, compliance, K))
    contacts = []
    for col in range(nc):
        a, b = h[1:3, col]
        mus, muk = d["headers"][[3, 4], col]
        assert mus == muk
        rows = []
        for p in range(h[5, col], h[5, col] + h[6, col]):
            n = d["lambdas"][:3, p].astype(float)
            t = d["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            ja = -np.concatenate((axes, np.cross(d["derived"][9:12, p], axes)), axis=1)
            jb = np.concatenate((axes, np.cross(d["derived"][12:15, p], axes)), axis=1)
            K = ja @ W[a] @ ja.T + jb @ W[b] @ jb.T
            rows.append((int(p), ja, jb, K, bool(d["derived"][3, p] > 0)))
        contacts.append((int(a), int(b), float(muk), rows))
    # Deterministic physical GS order: original joint IDs, then column/point IDs.
    # This is a different scheduling experiment from colored copied sweeps.
    applied = np.zeros_like(v)
    work = {"contact": 0.0, "hard_joint": 0.0, "drive": 0.0}

    def apply(a, b, ja, jb, delta, kind):
        fa = ja.T @ delta
        fb = jb.T @ delta
        before_a = v[a].copy()
        before_b = v[b].copy()
        v[a] += W[a] @ fa
        v[b] += W[b] @ fb
        applied[a] += fa
        applied[b] += fb
        # Dynamic kinetic work only; moving prescribed endpoints are external.
        if moving[a]:
            work[kind] += float(fa @ ((before_a + v[a]) * 0.5))
        if moving[b]:
            work[kind] += float(fb @ ((before_b + v[b]) * 0.5))

    def residuals():
        groups = {"joint_linear": [], "joint_angular": [], "drive": [], "normal": [], "tangent": []}
        details = {key: [] for key in groups}
        for joint, (a, b, idx, ja, jb, dyn, compliance, _K) in enumerate(joints):
            residual = ja @ v[a] + jb @ v[b] + compliance * accumulated[idx]
            residual[dyn] -= d["joint_reference"][idx[dyn]]
            for k, value in enumerate(residual):
                kind = (
                    "drive"
                    if dyn[k]
                    else (
                        "joint_linear" if np.linalg.norm(ja[k, :3]) + np.linalg.norm(jb[k, :3]) > 0 else "joint_angular"
                    )
                )
                groups[kind].append(float(value))
                details[kind].append({"joint": joint, "row": int(idx[k]), "bodies": [a, b], "residual": float(value)})
        for column, (a, b, mu, rows) in enumerate(contacts):
            for p, ja, jb, K, fixed in rows:
                if fixed:
                    continue
                g = ja @ v[a] + jb @ v[b]
                lam = impulse[:, p]
                scale = max(np.diag(K))
                groups["normal"].append(float(max(-g[0], -lam[0] * scale, abs(min(lam[0] * scale, g[0])), 0)))
                trial = lam[1:] - g[1:] / scale
                projection = trial * min(1.0, mu * max(lam[0], 0) / max(np.linalg.norm(trial), 1e-300))
                groups["tangent"].append(float(np.linalg.norm((lam[1:] - projection) * scale)))
                for kind in ("normal", "tangent"):
                    details[kind].append(
                        {
                            "column": column,
                            "point": p,
                            "bodies": [a, b],
                            "residual": groups[kind][-1],
                            "normal_impulse": float(lam[0]),
                        }
                    )
        result = {
            k: {"max": float(np.max(np.abs(x))) if x else 0.0, "l2": float(np.linalg.norm(x)), "count": len(x)}
            for k, x in groups.items()
        }
        for kind in groups:
            result[kind]["worst"] = sorted(details[kind], key=lambda row: abs(row["residual"]), reverse=True)[:5]
        energy = 0.5 * np.einsum("ni,nij,nj->", v, M, v) - 0.5 * np.einsum("ni,nij,nj->", initial_v, M, initial_v)
        delta_p = np.einsum("nij,nj->ni", M, v - initial_v)
        total = np.zeros(6)
        for body in range(count):
            term = delta_p[body] + (applied[body] if not moving[body] else 0)
            total[:3] += term[:3]
            total[3:] += term[3:] + np.cross(d["position"][body], term[:3])
        result.update(
            energy_change_J=float(energy),
            work_J=work.copy(),
            work_balance_error_J=float(energy - sum(work.values())),
            momentum_minus_external=total.tolist(),
        )
        return result

    history = [dict(cycle=0, **residuals())]
    evaluations = []
    solved_joint_blocks = 0
    solved_contact_points = 0
    failure = None
    for cycle in range(1, cycles + 1):
        if coupled:
            for block_name, component_bodies, block_joints, block_points, component_solve in component_specs:
                modified = d.copy()
                modified["velocity"] = v[:, :3].copy()
                modified["angular_velocity"] = v[:, 3:].copy()
                modified["impulses"] = impulse.copy()
                modified["joint_accumulated"] = accumulated.copy()
                path = "/tmp/colibri_physical_gs_input.npz"
                prefix = f"/tmp/colibri_physical_gs_{component}_{block_name}_{cycle}"
                np.savez_compressed(path, **modified)
                with contextlib.redirect_stdout(io.StringIO()):
                    component_solve(snapshot=path, output=prefix)
                report = json.loads(Path(prefix + ".json").read_text())
                evaluations.append(report["solver_reports"])
                if not report["accepted"]:
                    failure = {
                        "cycle": cycle,
                        "component": block_name,
                        "artifact": prefix,
                        "normal_residual": report["normal_residual_m_s"],
                        "tangent_residual": report["tangent_residual_m_s"],
                    }
                    break
                r = np.load(prefix + ".npz")
                old = v[component_bodies].copy()
                v[component_bodies] = r["v"].reshape(-1, 6)
                selected = (3 * r["active"][:, None] + np.arange(3)).ravel()
                forces = {
                    "contact": r["J"][selected].T @ (r["solution"] - r["initial"]),
                    "hard_joint": r["Jb"].T @ r["beta"],
                    "drive": r["Js"].T @ r["delta_soft"],
                }
                for kind, f in forces.items():
                    applied[component_bodies] += f.reshape(-1, 6)
                    work[kind] += float(f @ ((old.ravel() + r["v"]) * 0.5))
                # Static support reaction belongs to the same common-point impulse.
                for a, b, _mu, rows in contacts:
                    for p, ja, jb, _K, _fixed in rows:
                        if p not in block_points:
                            continue
                        index = int(np.flatnonzero(r["points"] == p)[0])
                        delta = r["full_impulses"].reshape(-1, 3)[index] - impulse[:, p]
                        if not moving[a]:
                            applied[a] += ja.T @ delta
                        if not moving[b]:
                            applied[b] += jb.T @ delta
                impulse[:, r["points"]] = r["full_impulses"].reshape(-1, 3).T
                hard_offset = 0
                soft_offset = 0
                for joint in sorted(block_joints):
                    a, b, idx, ja, jb, dyn, compliance, K = joints[joint]
                    nh = int(np.count_nonzero(~dyn))
                    ns = int(np.count_nonzero(dyn))
                    accumulated[idx[~dyn]] += r["beta"][hard_offset : hard_offset + nh]
                    accumulated[idx[dyn]] += r["delta_soft"][soft_offset : soft_offset + ns]
                    hard_offset += nh
                    soft_offset += ns
        if failure is not None:
            break
        for j, (a, b, idx, ja, jb, dyn, compliance, K) in enumerate(joints):
            if coupled and j in owned_joints:
                continue
            residual = ja @ v[a] + jb @ v[b] + compliance * accumulated[idx]
            residual[dyn] -= d["joint_reference"][idx[dyn]]
            delta = np.linalg.solve(K, -residual)
            # Joint impulse is applied simultaneously; split work with its shared midpoint.
            before_a = v[a].copy()
            before_b = v[b].copy()
            fa = ja.T @ delta
            fb = jb.T @ delta
            v[a] += W[a] @ fa
            v[b] += W[b] @ fb
            applied[a] += fa
            applied[b] += fb
            for mask, kind in ((~dyn, "hard_joint"), (dyn, "drive")):
                if moving[a]:
                    work[kind] += float((ja[mask].T @ delta[mask]) @ ((before_a + v[a]) * 0.5))
                if moving[b]:
                    work[kind] += float((jb[mask].T @ delta[mask]) @ ((before_b + v[b]) * 0.5))
            accumulated[idx] += delta
            solved_joint_blocks += 1
        for a, b, mu, rows in contacts:
            for p, ja, jb, K, fixed in rows:
                if fixed or (coupled and p in owned):
                    continue
                old = impulse[:, p].copy()
                g = ja @ v[a] + jb @ v[b]
                new_normal = max(0.0, old[0] - g[0] / K[0, 0])
                delta = np.array([new_normal - old[0], 0.0, 0.0])
                apply(a, b, ja, jb, delta, "contact")
                g = ja @ v[a] + jb @ v[b]
                T = K[1:, 1:]
                trial = old[1:] - np.linalg.solve(T, g[1:])
                radius = mu * new_normal
                if np.linalg.norm(trial) <= radius:
                    new_t = trial
                elif radius == 0:
                    new_t = np.zeros(2)
                else:
                    q = T @ trial
                    e, Q = np.linalg.eigh(T)
                    z = Q.T @ q

                    def length(alpha, z=z, e=e, radius=radius):
                        return np.linalg.norm(z / (e + alpha)) - radius

                    upper = max(np.linalg.norm(q) / radius, *e)
                    while length(upper) > 0:
                        upper *= 2
                    alpha = brentq(length, 0, upper, xtol=1e-14)
                    new_t = Q @ (z / (e + alpha))
                delta = np.r_[0.0, new_t - old[1:]]
                apply(a, b, ja, jb, delta, "contact")
                impulse[:, p] = np.r_[new_normal, new_t]
                solved_contact_points += 1
        entry = dict(
            cycle=cycle, joint_blocks=solved_joint_blocks, contact_point_blocks=solved_contact_points, **residuals()
        )
        history.append(entry)
        print(
            "CYCLE",
            coupled,
            cycle,
            {k: entry[k]["max"] for k in ("normal", "joint_linear", "joint_angular", "drive")},
            flush=True,
        )
    final_v = v.copy()
    final_impulse = impulse.copy()
    final_accumulated = accumulated.copy()
    v[:] = np.concatenate((d["after_velocity"], d["after_angular_velocity"]), axis=1)
    impulse[:] = d["after_impulses"]
    accumulated[:] = d["after_joint_accumulated"]
    native_report = residuals()
    native_report = {key: native_report[key] for key in ("normal", "tangent", "joint_linear", "joint_angular", "drive")}
    v[:] = final_v
    impulse[:] = final_impulse
    accumulated[:] = final_accumulated
    report = {
        "native_post_relax_all_rows": native_report,
        "body_mass_kg": [float(1 / x) if x > 0 else None for x in d["inverse_mass"]],
        "failure": failure,
        "component": component,
        "coupled": coupled,
        "cycles_requested": cycles,
        "pose_frozen": True,
        "order": "support then remaining joint IDs then contact columns/points",
        "history": history,
        "support_solver_evaluations": evaluations,
    }
    prefix = "/tmp/colibri_physical_block_gs_" + (
        ("coupled" if component == "support" else component) if coupled else "plain"
    )
    Path(prefix + ".json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(prefix + ".npz", v=v, impulses=impulse, accumulated=accumulated)


if __name__ == "__main__":
    run(False)
    run(True)
