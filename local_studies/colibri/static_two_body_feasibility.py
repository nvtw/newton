"""Rigid static support feasibility under captured authored drive and gravity."""

import json
from pathlib import Path

import numpy as np


def main():
    """Certify a resting circular-cone solution using an inscribed-ray LP witness."""
    from scipy.optimize import linprog

    source = "/tmp/colibri_two_body_full_biased.npz"
    x = np.load(source)
    report = json.load(open("/tmp/colibri_two_body_full_biased.json"))
    a = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    d = {k.split(".", 1)[1]: a[k] for k in a.files if k.startswith("biased_solved.")}
    points = np.asarray(report["points"])
    ids = np.flatnonzero(d["derived"][3, points] <= 0)
    # Separated points cannot furnish static support. Every overlapping witness remains eligible.
    C = x["C"].reshape(-1, 3, 12)[ids].reshape(-1, 12)
    mu = x["mu"][ids]
    count = len(ids)
    rays = np.zeros((3 * count, 64 * count))
    for i in range(count):
        for j in range(64):
            angle = 2 * np.pi * j / 64
            rays[3 * i : 3 * i + 3, 64 * i + j] = [1, mu[i] * np.cos(angle), mu[i] * np.sin(angle)]
    hard = x["diagonal"] == 0
    B = x["B"]
    W = x["W"]
    D = x["diagonal"]
    drive = x["targets"][~hard] / D[~hard]
    gravity = np.zeros(12)
    dt = float(a["dt"][0])
    for i in range(2):
        gravity[6 * i + 2] = -dt / W[6 * i, 6 * i]  # Authored gravity is -1m/s².
    external = gravity + B[~hard].T @ drive
    operator = np.c_[C.T @ rays, B[hard].T] * 1e-4
    scale = np.maximum(np.linalg.norm(operator, axis=1), np.abs(external))
    result = linprog(
        np.r_[np.ones(64 * count), np.zeros(sum(hard))],
        A_eq=operator / scale[:, None],
        b_eq=-external / scale,
        bounds=[(0, None)] * (64 * count) + [(None, None)] * sum(hard),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    out = {
        "scope": "Static rigid-contact force feasibility at captured pose, zero velocities; original authored finite-gain drive and gravity; NOT numerical recovery/compliance equilibrium or live dynamics",
        "eligible_points": points[ids].tolist(),
        "separated_points_excluded": len(points) - count,
        "gravity_m_s2": -1.0,
        "dt": dt,
        "drive_impulse_Nms": drive.tolist(),
        "lp_success": bool(result.success),
        "message": result.message,
        "infeasibility_limit": "Inscribed polygon failure alone would not disprove circular-cone feasibility",
    }
    if result.success:
        contact = (rays @ result.x[: 64 * count] * 1e-4).reshape(-1, 3)
        joint = result.x[64 * count :] * 1e-4
        balance = C.T @ contact.ravel() + B[hard].T @ joint + external
        cone = np.linalg.norm(contact[:, 1:], axis=1) - mu * contact[:, 0]
        assert np.max(np.abs(balance)) < 1e-12
        assert np.max(cone) < 1e-12 and np.min(contact[:, 0]) >= -1e-14
        normal_by_body = {}
        for b in (1, 2):
            own = np.linalg.norm(C.reshape(-1, 3, 12)[:, :, 6 * (b - 1) : 6 * b], axis=(1, 2)) > 0
            normal_by_body[str(b)] = float(contact[own, 0].sum())
        out.update(
            force_balance_max_Ns=float(np.max(np.abs(balance))),
            cone_violation=float(np.max(cone)),
            normal_impulse_by_body=normal_by_body,
            contacts=contact.tolist(),
            hard_joint_impulses=joint.tolist(),
            static_kinetic_work_J=0.0,
        )
    outer_rays = rays.copy()
    outer_rays.reshape(count, 3, -1)[:, 1:, :] /= np.cos(np.pi / 64)
    outer_operator = np.c_[C.T @ outer_rays, B[hard].T] * 1e-4
    outer = linprog(
        np.r_[np.ones(64 * count), np.zeros(sum(hard))],
        A_eq=outer_operator / scale[:, None],
        b_eq=-external / scale,
        bounds=[(0, None)] * (64 * count) + [(None, None)] * sum(hard),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    out["outer_polygon_success"] = bool(outer.success)
    out["outer_polygon_status"] = outer.message
    out["outer_polygon_scope"] = (
        "Circumscribed64gon contains circular cones; infeasibility is a numerical LP certificate for the circular static system, subject to stated solver tolerances"
    )
    drive_operator = np.c_[outer_operator, B[~hard].T * 1e-4]
    drive_scale = np.maximum(np.linalg.norm(drive_operator, axis=1), abs(gravity))
    extrema = []
    for sign in (1.0, -1.0):
        objective = np.zeros(drive_operator.shape[1])
        objective[-1] = sign
        fit = linprog(
            objective,
            A_eq=drive_operator / drive_scale[:, None],
            b_eq=-gravity / drive_scale,
            bounds=[(0, None)] * (64 * count) + [(None, None)] * (sum(hard) + 1),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        extrema.append(float(fit.x[-1] * 1e-4) if fit.success else fit.message)
    out["outer_feasible_drive_impulse_minmax_Nms"] = extrema
    Path("/tmp/colibri_two_body_static_feasibility.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
