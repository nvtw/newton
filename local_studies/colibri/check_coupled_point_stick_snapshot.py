"""Frozen two-body total point-friction proposal with full normal/joint acceptance.

Offline reconstructed physical response; no state writes or native-GPU equivalence
claim. The three-coordinate proposal is audited through ALL original contact rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.physx_patch_reference_ledger import rotation
from local_studies.colibri.point_friction_wrench_reference import distribute_stick


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=Path("/tmp/colibri_native_direct_owned_fixed3600.native_state.npz")
    )
    parser.add_argument("--properties", type=Path, default=Path("/tmp/colibri_physx_patch_hybrid3600.patch.npz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_coupled_point_stick_frozen.json"))
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
    planar = np.c_[(points - origin) @ t0, (points - origin) @ t1, np.zeros(len(ids))]
    capacities = mus[ids] * np.maximum(lam[ids, 0], 0)
    # G maps desired planar wrench to the two-body generalized contact impulse.
    g = np.zeros((12, 3))
    for q, axis in enumerate((t0, t1)):
        g[:6, q] = -np.r_[axis, np.cross(origin - x["body_position"][1], axis)]
    g[3:6, 2] = -normal
    old = c[ids, 1:, :].reshape(-1, 12).T @ lam[ids, 1:].ravel()
    free = v - wc @ old
    mobility = g.T @ wc @ g
    factor = np.linalg.cholesky(mobility)
    requested = -np.linalg.solve(factor.T, np.linalg.solve(factor, g.T @ free))
    force2, proposal = distribute_stick(planar, capacities, requested, tolerance=1e-12)
    assert force2 is not None
    world = force2[:, 0, None] * t0 + force2[:, 1, None] * t1
    proposed = np.stack(
        [np.sum(world * data[3:6, ids].T, axis=1), np.sum(world * np.cross(data[:3, ids].T, data[3:6, ids].T), axis=1)],
        axis=1,
    )
    delta = proposed - lam[ids, 1:]
    force = c[ids, 1:, :].reshape(-1, 12).T @ delta.ravel()
    reaction = -solve(j @ w @ force)
    dv = w @ (force + j.T @ reaction)
    after = v + dv
    before_cv = np.einsum("nki,i->nk", c, v)
    after_cv = np.einsum("nki,i->nk", c, after)
    overlap = d[3, :count] <= 0
    loaded = lam[:, 0] > 0

    def normal_residual(cv):
        return np.where(loaded, abs(cv[:, 0]), np.maximum(-cv[:, 0], 0))

    drive_before = j @ v + compliance * x["direct_accumulated_impulse"] - x["direct_velocity_reference"]
    drive_after = j @ after + compliance * (x["direct_accumulated_impulse"] + reaction) - x["direct_velocity_reference"]
    positive = capacities > 0
    max_tangent = float(np.max(abs(after_cv[ids[positive], 1:])))
    max_normal = float(np.max(normal_residual(after_cv)[overlap]))
    cone = float(np.max(np.linalg.norm(proposed, axis=1) - capacities))
    hard = float(np.max(abs((j @ dv)[~dynamic])))
    drive = float(np.max(abs(drive_after[dynamic])))
    tol = 1e-8
    accepted = bool(
        proposal["accepted"]
        and max_tangent <= tol
        and max_normal <= tol
        and cone <= 1e-12
        and hard <= tol
        and drive <= tol
    )
    midpoint = 0.5 * (v + after)
    report = {
        "scope": __doc__,
        "snapshot": str(args.snapshot),
        "properties": str(args.properties),
        "accepted": accepted,
        "proposal": proposal,
        "loaded_stick_residual_m_s": max_tangent,
        "normal_overlap_before_m_s": float(np.max(normal_residual(before_cv)[overlap])),
        "normal_overlap_after_m_s": max_normal,
        "speculative_normal_velocity_change_m_s": float(np.max(abs((after_cv - before_cv)[~overlap, 0]))),
        "drive_before_rad_s": float(np.max(abs(drive_before[dynamic]))),
        "drive_after_rad_s": drive,
        "hard_joint_delta_residual_mixed": hard,
        "max_cone_excess_Ns": cone,
        "base_before_velocity": v[:6].tolist(),
        "base_after_velocity": after[:6].tolist(),
        "full_work_closure_J": float(
            0.5 * after @ m @ after - 0.5 * v @ m @ v - force @ midpoint - reaction @ (j @ midpoint)
        ),
        "point_plane_height_range_m": float(np.ptp((points - origin) @ normal)),
        "per_point_before": before_cv.tolist(),
        "per_point_after": after_cv.tolist(),
        "normal_residual_after": normal_residual(after_cv).tolist(),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(
        {
            k: val
            for k, val in report.items()
            if k not in ("per_point_before", "per_point_after", "normal_residual_after")
        }
    )


if __name__ == "__main__":
    main()
