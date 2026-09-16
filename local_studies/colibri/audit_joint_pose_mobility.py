"""Independent row-gauge and contact mobility checks at captured poses."""

import json
from pathlib import Path

import numpy as np
import sympy as sp


def main():
    import scipy.linalg as la

    outputs = []
    for frame in (1, 30, 120):
        p = np.load(f"/tmp/colibri_joint_pose{frame}_reformed.npz")
        s = np.load(f"/tmp/colibri_joint_pose{frame}.npz")
        ref = np.load(f"/tmp/colibri_joint_pose{frame}_rank.npz")
        active = ref["active"]
        J = p["J"][:, active, :].reshape(188, -1)
        R = p["compliance"]
        drives = np.flatnonzero(R > 0)
        F = np.concatenate((J.T, np.eye(188)[R > 0]), axis=0)
        reverse = list(reversed(range(188)))
        exact = sp.polys.matrices.DomainMatrix.from_Matrix(
            sp.Matrix([[sp.Rational(float(x)) for x in row] for row in F[:, reverse]])
        )
        keep = [reverse[k] for k in exact.rref()[1]]
        assert len(keep) == 184
        L = la.block_diag(*(la.cholesky(p["W"][b], lower=True) for b in active))
        C = np.column_stack((J @ L, np.diag(np.sqrt(R))[:, drives]))
        scale = 1 / np.linalg.norm(C[keep], axis=1)
        Q, T = la.qr((C[keep] * scale[:, None]).T, mode="full")
        n = len(keep)
        G = L @ Q[: len(L), n:]
        a = la.solve_triangular(T[:n, :n].T, -p["rhs"][keep] * scale, lower=True)
        z = Q[:, :n] @ a
        dv = L @ z[: len(L)]
        lam = np.zeros(188)
        lam[keep] = scale * la.solve_triangular(T[:n, :n], a)
        h = s["headers"].view(np.int32)
        contact = []
        for col in range(int(s["column_count"][0])):
            x, y = h[1:3, col]
            for point in range(h[5, col], h[5, col] + h[6, col]):
                if s["derived"][3, point] > 0:
                    continue
                normal = s["lambdas"][:3, point].astype(float)
                tangent = s["lambdas"][3:6, point].astype(float)
                axes = np.array([normal, tangent, np.cross(normal, tangent)])
                c = np.zeros((3, 38, 6))
                c[:, x] = -np.concatenate((axes, np.cross(s["derived"][9:12, point].astype(float), axes)), axis=1)
                c[:, y] = np.concatenate((axes, np.cross(s["derived"][12:15, point].astype(float), axes)), axis=1)
                contact.extend(c[:, active].reshape(3, -1))
        contact = np.array(contact)
        cg = contact @ G
        cgr = contact @ ref["G"]
        H = cg @ cg.T
        Hr = cgr @ cgr.T
        response = ref["G"] @ cgr.T
        drive = (ref["slack"] @ cgr.T) / np.sqrt(R[drives])[:, None]
        r = J @ response
        r[drives] += R[drives, None] * drive
        M = la.block_diag(*(np.linalg.inv(p["W"][b]) for b in active))
        work = np.sum(contact.T * response, axis=0)
        kin = np.sum(response * (M @ response), axis=0)
        penalty = np.sum(R[drives, None] * drive**2, axis=0)
        out = {
            "frame": frame,
            "eligible_points": len(contact) // 3,
            "alternate_selected_rank": len(keep),
            "original_row_residual_alternate": float(abs(J @ dv + R * lam + p["rhs"]).max()),
            "max_baseline_dv_difference_between_gauges": float(abs(dv - ref["dv"].ravel()).max()),
            "contact_mobility_relative_gauge_difference": float(np.linalg.norm(H - Hr) / np.linalg.norm(Hr)),
            "max_contact_mobility_gauge_difference": float(abs(H - Hr).max()),
            "max_joint_contact_response_residual_per_unit_impulse": float(abs(r).max()),
            "max_relative_work_error": float(np.max(abs(work - kin - penalty) / np.maximum(abs(work), 1))),
            "symmetry_error": float(abs(Hr - Hr.T).max()),
        }
        if frame == 1:
            mp_basis = np.load("/tmp/colibri_joint_pose1_mp_mobility.npz")
            mp_response = np.load("/tmp/colibri_joint_pose1_mp.npz")
            cgmp = contact @ mp_basis["G"]
            Hmp = cgmp @ cgmp.T
            out["fixed_gauge_FP64_vs_MP_contact_relative_error"] = float(np.linalg.norm(Hr - Hmp) / np.linalg.norm(Hmp))
            out["fixed_gauge_FP64_vs_MP_max_contact_error"] = float(abs(Hr - Hmp).max())
            out["fixed_gauge_FP64_vs_MP_max_dv_error"] = float(abs(ref["dv"] - mp_response["dv"]).max())
            out["reverse_gauge_FP64_vs_MP_contact_relative_error"] = float(
                np.linalg.norm(H - Hmp) / np.linalg.norm(Hmp)
            )
        outputs.append(out)
    Path("/tmp/colibri_joint_pose_gauge_mobility.json").write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
