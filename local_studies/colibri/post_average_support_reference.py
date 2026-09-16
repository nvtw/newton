"""Frozen replacement of native owned increments; audit all incident rows."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri import coupled_support_reference as coupled
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def main():
    d = dict(np.load("/tmp/colibri_support_relax330.npz"))
    r = np.load("/tmp/colibri_coupled_support_reference.npz")
    n = len(d["inverse_mass"])
    u = np.concatenate((d["velocity"], d["angular_velocity"]), axis=1).astype(float)
    native = np.concatenate((d["after_velocity"], d["after_angular_velocity"]), axis=1).astype(float)
    inertia = inertia_sym6_unpack_np(d["inverse_inertia"]).astype(float)

    def response(f):
        result = f.copy()
        result[:, :3] *= d["inverse_mass"][:, None]
        result[:, 3:] = np.einsum("nij,nj->ni", inertia, f[:, 3:])
        return result

    jd = d["joint_data"].view(np.int32)
    headers = d["headers"].view(np.int32)
    owned_points = set(r["points"].tolist())
    total = np.zeros((n, 6))
    owned = np.zeros_like(total)
    owned_contact = np.zeros_like(total)
    owned_hard = np.zeros_like(total)
    owned_drive = np.zeros_like(total)
    joints = []
    contacts = []
    for joint in range(int(d["num_joints"][0])):
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], joint]
        structural = int(d["joint_structural_index"][joint])
        for k in range(int(d["joint_row_count"][joint])):
            row = int(d["joint_row_indices"][joint, k])
            local = int(d["joint_row_local"][row])
            ja = d["joint_wrench0"][structural, local].astype(float)
            jb = d["joint_wrench1"][structural, local].astype(float)
            delta = float(d["after_joint_accumulated"][row]) - float(d["joint_accumulated"][row])
            for body, jac in ((a, ja), (b, jb)):
                total[body] += jac * delta
                if joint == 0:
                    owned[body] += jac * delta
                    target = owned_drive if d["joint_row_dynamic"][row] else owned_hard
                    target[body] += jac * delta
            if joint != 0 and (a in (1, 2) or b in (1, 2)):
                joints.append((joint, row, int(a), int(b), ja, jb))
    for column in range(int(d["column_count"][0])):
        a, b = headers[1:3, column]
        for p in range(headers[5, column], headers[5, column] + headers[6, column]):
            normal = d["lambdas"][:3, p].astype(float)
            tangent = d["lambdas"][3:6, p].astype(float)
            axes = np.array([normal, tangent, np.cross(normal, tangent)])
            ja = -np.concatenate((axes, np.cross(d["derived"][9:12, p], axes)), axis=1)
            jb = np.concatenate((axes, np.cross(d["derived"][12:15, p], axes)), axis=1)
            delta = d["after_impulses"][:, p].astype(float) - d["impulses"][:, p].astype(float)
            for body, jac in ((a, ja), (b, jb)):
                total[body] += jac.T @ delta
                if p in owned_points:
                    owned[body] += jac.T @ delta
                    owned_contact[body] += jac.T @ delta
            if p not in owned_points and (a in (1, 2) or b in (1, 2)):
                contacts.append((column, p, int(a), int(b), ja, jb))
    reconstructed = u + response(total)
    movable = d["inverse_mass"] > 0
    reconstruction_error = float(np.max(np.abs(reconstructed[movable] - native[movable])))
    # Exact captured unowned impulse contributions are held fixed. Their values
    # still depend on the original row order, so this is not a skipped-row replay.
    without_owned = native - response(owned)
    np.testing.assert_allclose(without_owned + response(owned), native, atol=1e-15, rtol=0)
    modified = d.copy()
    modified["velocity"] = without_owned[:, :3]
    modified["angular_velocity"] = without_owned[:, 3:]
    np.savez_compressed("/tmp/colibri_post_average_support_input.npz", **modified)
    coupled.main(
        snapshot="/tmp/colibri_post_average_support_input.npz", output="/tmp/colibri_post_average_support_reference"
    )
    solved = np.load("/tmp/colibri_post_average_support_reference.npz")
    candidate = without_owned.copy()
    candidate[[1, 2]] = solved["v"].reshape(2, 6)
    selected = (3 * solved["active"][:, None] + np.arange(3)).ravel()
    contact_net = solved["J"][selected].T @ (solved["solution"] - solved["initial"]) - owned_contact[[1, 2]].ravel()
    hard_net = solved["Jb"].T @ solved["beta"] - owned_hard[[1, 2]].ravel()
    drive_net = solved["Js"].T @ solved["delta_soft"] - owned_drive[[1, 2]].ravel()
    net = contact_net + hard_net + drive_net
    before_component = native[[1, 2]].ravel()
    after_component = candidate[[1, 2]].ravel()
    np.testing.assert_allclose(after_component - before_component, solved["W"] @ net, atol=1e-14, rtol=0)
    M = np.linalg.inv(solved["W"])
    midpoint = (before_component + after_component) * 0.5
    delta_energy = 0.5 * (after_component @ M @ after_component - before_component @ M @ before_component)
    delta_momentum = M @ (after_component - before_component) - contact_net
    balance = np.zeros(6)
    for i, body in enumerate((1, 2)):
        value = delta_momentum[6 * i : 6 * i + 6]
        balance[:3] += value[:3]
        balance[3:] += value[3:] + np.cross(d["position"][body], value[:3])
    replacement_budget = {
        "energy_change_J": float(delta_energy),
        "contact_work_J": float(contact_net @ midpoint),
        "hard_joint_work_J": float(hard_net @ midpoint),
        "drive_work_J": float(drive_net @ midpoint),
        "work_error_J": float(delta_energy - net @ midpoint),
        "momentum_minus_static_reaction": balance.tolist(),
    }
    joint_rows = []
    for joint, row, a, b, ja, jb in joints:
        dynamic = bool(d["joint_row_dynamic"][row])
        offset = (
            float(d["after_joint_accumulated"][row]) / float(d["joint_dynamic_mass"][row])
            - float(d["joint_reference"][row])
            if dynamic
            else 0.0
        )
        before = float(ja @ native[a] + jb @ native[b] + offset)
        after = float(ja @ candidate[a] + jb @ candidate[b] + offset)
        joint_rows.append(
            {
                "joint": joint,
                "row": row,
                "dynamic": dynamic,
                "bounded": bool(d["joint_bounded_drive"][row]),
                "before": before,
                "after": after,
                "linear_norm": float(np.linalg.norm(ja[:3])),
                "angular_norm": float(np.linalg.norm(ja[3:])),
            }
        )
    contact_rows = []
    for col, p, a, b, ja, jb in contacts:
        impulse = d["after_impulses"][:, p].astype(float)
        gamma = float(d["derived"][6, p]) if d["derived"][8, p] > 0 else 0.0
        bias = -float(d["derived"][7, p]) if d["derived"][8, p] > 0 else 0.0
        before = ja @ native[a] + jb @ native[b]
        after = ja @ candidate[a] + jb @ candidate[b]
        before[0] += gamma * impulse[0] + bias
        after[0] += gamma * impulse[0] + bias
        contact_rows.append(
            {
                "column": col,
                "point": p,
                "fixed_speculative": bool(d["derived"][3, p] > 0),
                "impulse": impulse.tolist(),
                "before": before.tolist(),
                "after": after.tolist(),
                "mu_static": float(d["headers"][3, col]),
                "mu_dynamic": float(d["headers"][4, col]),
            }
        )
    for entry, (_, _, a, b, ja, jb) in zip(contact_rows, contacts, strict=True):
        mobility = np.zeros((3, 3))
        for body, jac in ((a, ja), (b, jb)):
            mobility += d["inverse_mass"][body] * jac[:, :3] @ jac[:, :3].T
            mobility += jac[:, 3:] @ inertia[body] @ jac[:, 3:].T
        scale = float(np.max(np.diag(mobility)))
        impulse = np.array(entry["impulse"])
        for stage in ("before", "after"):
            gradient = np.array(entry[stage])
            entry[stage + "_normal_kkt"] = float(
                max(-gradient[0], -impulse[0] * scale, abs(min(impulse[0] * scale, gradient[0])), 0.0)
            )
            trial = impulse[1:] - gradient[1:] / scale
            radius = entry["mu_dynamic"] * max(impulse[0], 0)
            projected = trial * min(1.0, radius / max(np.linalg.norm(trial), 1e-300))
            entry[stage + "_tangent_map"] = float(np.linalg.norm((impulse[1:] - projected) * scale))
        entry["equal_friction_coefficients"] = entry["mu_static"] == entry["mu_dynamic"]
    report = {
        "stage": "Native post-average; remove captured owned increments once and replace, holding native unowned impulses fixed",
        "not_a_skipped_row_replay": True,
        "replacement_vs_native_budget": replacement_budget,
        "native_reconstruction_error": reconstruction_error,
        "owned_points": len(owned_points),
        "owned_joints": [0],
        "external_joint_rows": joint_rows,
        "external_contact_points": contact_rows,
        "external_joint_max_before": max(abs(x["before"]) for x in joint_rows),
        "external_joint_max_after": max(abs(x["after"]) for x in joint_rows),
    }
    free_contacts = [x for x in contact_rows if not x["fixed_speculative"]]
    report["external_free_contact_count"] = len(free_contacts)
    report["external_fixed_speculative_count"] = len(contact_rows) - len(free_contacts)
    report["contact_friction_map_scope_complete"] = all(x["equal_friction_coefficients"] for x in free_contacts)
    report["bounded_joint_rows_present"] = any(x["bounded"] for x in joint_rows)
    for stage in ("before", "after"):
        for field in ("normal_kkt", "tangent_map"):
            report["external_" + stage + "_" + field] = max((x[stage + "_" + field] for x in free_contacts), default=0)
        for kind in ("linear", "angular"):
            rows = [x for x in joint_rows if not x["dynamic"] and ((x["linear_norm"] > 0) == (kind == "linear"))]
            report["external_" + stage + "_hard_" + kind] = max((abs(x[stage]) for x in rows), default=0)
    report["native_reconstruction_per_body"] = np.max(np.abs(reconstructed - native), axis=1).tolist()
    Path("/tmp/colibri_post_average_support_audit.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        "/tmp/colibri_post_average_support_audit.npz",
        native=native,
        without_owned=without_owned,
        candidate=candidate,
        owned_impulse=owned,
        total_impulse=total,
        reconstructed=reconstructed,
    )
    print(
        "AUDIT",
        reconstruction_error,
        len(joint_rows),
        len(contact_rows),
        report["external_joint_max_before"],
        report["external_joint_max_after"],
        flush=True,
    )


if __name__ == "__main__":
    main()
