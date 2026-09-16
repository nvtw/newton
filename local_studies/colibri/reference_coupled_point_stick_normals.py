"""Offline all-contact normal+stick certificate for an actual frozen final-relax state.

Changes only a hypothetical incremental impulse at fixed captured pose. Speculative
rows remain exactly held, current hard joints and finite drive are retained. An
inscribed 16-ray cone LP is a sufficient feasibility proposal, never a replacement
contact law: acceptance rechecks the original circular cones and every physical row.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.physx_patch_reference_ledger import rotation


def main():
    from scipy.optimize import linprog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=Path("/tmp/colibri_native_direct_owned_fixed3600.native_state.npz")
    )
    parser.add_argument("--properties", type=Path, default=Path("/tmp/colibri_physx_patch_hybrid3600.patch.npz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_coupled_normal_stick_reference.json"))
    args = parser.parse_args()
    x, properties = np.load(args.snapshot), np.load(args.properties)
    mass = properties["body_mass"].astype(float)
    inertia = properties["body_inertia"].astype(float)
    m = np.zeros((12, 12))
    for k, b in enumerate((1, 2)):
        r = rotation(x["body_orientation"][b].astype(float))
        m[6 * k : 6 * k + 3, 6 * k : 6 * k + 3] = mass[b - 1] * np.eye(3)
        m[6 * k + 3 : 6 * k + 6, 6 * k + 3 : 6 * k + 6] = r @ inertia[b - 1] @ r.T
    w = np.linalg.inv(m)
    j = np.concatenate([x["direct_row_wrench0"][0], x["direct_row_wrench1"][0]], axis=1).astype(float)
    dynamic = x["direct_row_dynamic"].astype(bool)
    compliance = np.zeros(6)
    compliance[dynamic] = 1 / x["direct_dynamic_mass"][dynamic].astype(float)
    a = j @ w @ j.T + np.diag(compliance)
    chol = np.linalg.cholesky(a)

    def solve(b):
        return np.linalg.solve(chol.T, np.linalg.solve(chol, b))

    wc = w - w @ j.T @ solve(j @ w)
    v = np.c_[x["body_velocity"][1:3], x["body_angular_velocity"][1:3]].astype(float).ravel()
    count = int(x["contact_valid_count"][0])
    data = x["contact_lambdas"].astype(float)
    d = x["contact_derived"].astype(float)
    headers = x["contact_columns_data"]
    h = headers.view(np.int32)
    c = np.zeros((count, 3, 12))
    mus = np.zeros(count)
    pairs = np.zeros((count, 2), int)
    for col in range(h.shape[1]):
        first, n = h[5:7, col]
        if first < 0 or n <= 0 or first + n > count:
            continue
        b0, b1 = h[1:3, col]
        for k in range(first, first + n):
            pairs[k] = [b0, b1]
            mus[k] = headers[3, col]
            normal = data[:3, k]
            t = data[3:6, k]
            for row, axis in enumerate((normal, t, np.cross(normal, t))):
                for body, lever, sign in ((b0, d[9:12, k], -1), (b1, d[12:15, k], 1)):
                    if body in (1, 2):
                        c[k, row, 6 * (body - 1) : 6 * body] = sign * np.r_[axis, np.cross(lever, axis)]
    lam = x["contact_impulses"][:, :count].T.astype(float)
    ids = np.flatnonzero((pairs[:, 0] == 1) & (pairs[:, 1] == 0))
    assert len(ids) > 0 and np.max(abs(data[:3, ids].T - data[:3, ids[0]])) == 0
    normal = data[:3, ids[0]]
    t0 = data[3:6, ids[0]]
    t1 = np.cross(normal, t0)
    points = x["body_position"][1] + d[9:12, ids].T
    origin = points.mean(axis=0)
    np.c_[(points - origin) @ t0, (points - origin) @ t1, np.zeros(len(ids))]
    mus[ids] * np.maximum(lam[ids, 0], 0)

    eligible = np.flatnonzero(d[3, :count] <= 0)
    assert np.all(pairs[eligible] == [1, 0]), "Bounded two-body final-relax certificate"
    ce = c[eligible].reshape(-1, 12)
    old_force = ce.T @ lam[eligible].ravel()
    h = wc[:6, :6]
    np.linalg.cholesky(h)
    # Base zero spatial velocity guarantees normal and tangent equations on every
    # support point, including actual noncoplanar lever geometry.
    requested = old_force[:6] - np.linalg.solve(h, v[:6])
    amap = c[eligible, :, :6].reshape(-1, 6).T
    weights = np.repeat(np.maximum(lam[eligible, 0], 0), 3)
    gram = (amap * weights) @ amap.T
    weighted = weights * (amap.T @ np.linalg.solve(gram, requested))
    weighted = weighted.reshape(-1, 3)
    print(
        "WEIGHTED",
        {
            "mobility_condition": float(np.linalg.cond(h)),
            "gram_condition": float(np.linalg.cond(gram)),
            "minnormal": float(weighted[:, 0].min()),
            "maxcone": float(np.max(np.linalg.norm(weighted[:, 1:], axis=1) - mus[eligible] * weighted[:, 0])),
            "wrencherror": float(np.max(abs(amap @ weighted.ravel() - requested))),
        },
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        mobility=h,
        point_map=amap,
        velocity=v,
        operator=w,
        conditioned=wc,
        joints=j,
        compliance=compliance,
        contact_rows=c,
        impulses=lam,
        eligible=eligible,
        coefficients=mus,
        derived=d[:, :count],
        mass=m,
        drive_impulses=x["direct_accumulated_impulse"],
        drive_reference=x["direct_velocity_reference"],
        dynamic=dynamic,
    )
    rays = []
    coefficients = []
    for k in eligible:
        for angle in np.arange(16) * 2 * np.pi / 16:
            local = np.array([1.0, mus[k] * np.cos(angle), mus[k] * np.sin(angle)])
            rays.append(c[k, :, :6].T @ local)
            coefficients.append(local)
    ray_matrix = np.array(rays).T
    coefficients = np.array(coefficients).reshape(len(eligible), 16, 3)
    impulse_scale = 1e-4
    length_scale = max(float(np.max(np.linalg.norm(d[9:12, eligible].T, axis=1))), 1e-3)
    row_scale = np.r_[np.ones(3), np.full(3, 1 / length_scale)]
    result = linprog(
        np.ones(len(rays)),
        A_eq=row_scale[:, None] * ray_matrix,
        b_eq=row_scale * requested / impulse_scale,
        bounds=(0, None),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    report = {
        "scope": __doc__,
        "snapshot": str(args.snapshot),
        "eligible_points": eligible.tolist(),
        "held_speculative_points": np.flatnonzero(d[3, :count] > 0).tolist(),
        "proposal_solver_status": result.message,
        "accepted": False,
        "ray_count_per_point": 16,
    }
    if result.success:
        values = result.x * impulse_scale
        # A successful LP is verified below in original SI equations, not solver-scaled residuals.
        proposed = np.einsum("nr,nrk->nk", values.reshape(len(eligible), 16), coefficients)
        new_lam = lam.copy()
        new_lam[eligible] = proposed
        delta = new_lam - lam
        force = c.reshape(-1, 12).T @ delta.ravel()
        reaction = -solve(j @ w @ force)
        dv = w @ (force + j.T @ reaction)
        after = v + dv
        cv_before = np.einsum("nki,i->nk", c, v)
        cv_after = np.einsum("nki,i->nk", c, after)
        positive = new_lam[:, 0] > 0
        normal = np.where(positive, abs(cv_after[:, 0]), np.maximum(-cv_after[:, 0], 0))
        cone = np.linalg.norm(new_lam[:, 1:], axis=1) - mus * new_lam[:, 0]
        tangent = float(np.max(abs(cv_after[eligible, 1:])))
        norm = float(np.max(normal[eligible]))
        hard = float(np.max(abs((j @ dv)[~dynamic])))
        drive_res = (
            j @ after + compliance * (x["direct_accumulated_impulse"] + reaction) - x["direct_velocity_reference"]
        )
        drive = float(np.max(abs(drive_res[dynamic])))
        midpoint = 0.5 * (v + after)
        momentum = m @ dv
        ground = np.zeros(6)
        for k in eligible:
            axis = np.stack([data[:3, k], data[3:6, k], np.cross(data[:3, k], data[3:6, k])])
            f = delta[k] @ axis
            point = x["body_position"][1] + d[9:12, k]
            ground += np.r_[f, np.cross(point, f)]
        dp = momentum[:3] + momentum[6:9]
        dl = (
            momentum[3:6]
            + momentum[9:12]
            + np.cross(x["body_position"][1], momentum[:3])
            + np.cross(x["body_position"][2], momentum[6:9])
        )
        spec_margin = cv_after[d[3, :count] > 0, 0] + d[3, :count][d[3, :count] > 0]
        report["held_speculative_min_vn_plus_bias_m_s"] = float(spec_margin.min())
        report["held_policy"] = (
            "Velocity pass skips these rows: unchanged impulses plus predictive vn+bias admissibility, NOT complementary re-solve of held lambda"
        )
        absolute_hard = float(np.max(abs((j @ after)[~dynamic])))
        report["absolute_hard_joint_before"] = float(np.max(abs((j @ v)[~dynamic])))
        report["absolute_hard_joint_after"] = absolute_hard
        report["contact_and_homogeneous_joint_certificate"] = bool(
            norm <= 1e-8
            and tangent <= 1e-8
            and hard <= 1e-8
            and drive <= 1e-8
            and np.min(new_lam[:, 0]) >= 0
            and max(cone) <= 1e-12
            and np.min(spec_margin) >= -1e-8
        )
        report.update(
            accepted=bool(
                absolute_hard <= 1e-8
                and np.min(spec_margin) >= -1e-8
                and norm <= 1e-8
                and tangent <= 1e-8
                and hard <= 1e-8
                and drive <= 1e-8
                and np.min(new_lam[:, 0]) >= 0
                and max(cone) <= 1e-12
            ),
            normal_residual_m_s=norm,
            tangent_stick_residual_m_s=tangent,
            drive_residual_rad_s=drive,
            hard_joint_delta_residual_mixed=hard,
            max_cone_excess_Ns=float(max(cone)),
            normal_load_sum_before_Ns=float(lam[eligible, 0].sum()),
            normal_load_sum_after_Ns=float(proposed[:, 0].sum()),
            before_base_velocity=v[:6].tolist(),
            after_base_velocity=after[:6].tolist(),
            speculative_velocity_change_m_s=float(np.max(abs((cv_after - cv_before)[d[3, :count] > 0, 0]))),
            held_speculative_impulses_byte_equal=bool(np.array_equal(new_lam[d[3, :count] > 0], lam[d[3, :count] > 0])),
            linear_ground_ledger_error_Ns=float(np.max(abs(dp + ground[:3]))),
            angular_ground_ledger_error_Nms=float(np.max(abs(dl + ground[3:]))),
            work_closure_J=float(
                0.5 * after @ m @ after - 0.5 * v @ m @ v - force @ midpoint - reaction @ (j @ midpoint)
            ),
            proposed_impulses_Ns=new_lam.tolist(),
            contact_velocity_after=cv_after.tolist(),
        )
        # Biased all-STICK is separately screened with its unchanged captured targets.
        # Remove only the exactly absent normal-translation column; all five tangential
        # physical columns (including weak out-of-plane lever modes) are retained.
        trows = c[eligible, 1:, :6].reshape(-1, 6)
        basis = np.zeros((6, 5))
        basis[:3, 0] = t0
        basis[:3, 1] = t1
        basis[3:, 2:] = np.eye(3)
        tb = trows @ basis
        target = -d[4:6, eligible].T.ravel()
        fit, _, rank, singular = np.linalg.lstsq(tb, target, rcond=0)
        report["biased_all_stick_target_screen"] = {
            "rank": int(rank),
            "singular_values": singular.tolist(),
            "best_max_target_residual_m_s": float(np.max(abs(tb @ fit - target))),
            "note": "Unchanged tangent targets alone are incompatible; no biased soft-normal solution claimed or normal compliance altered.",
        }
    args.output.write_text(json.dumps(report, indent=2))
    print(
        {
            key: value
            for key, value in report.items()
            if key
            not in ("proposed_impulses_Ns", "contact_velocity_after", "eligible_points", "held_speculative_points")
        }
    )


if __name__ == "__main__":
    main()
