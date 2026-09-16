"""CPU native metric contact PGS with exact six-row joint-condensed mobility."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.check_d6_frozen_rows import contact_sweep
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.coupled_support_online import assemble_snapshot


def run(a, d, phase):
    """Preserve original joint compliance while applying every contact response."""
    w, b, c, k = a["W"], a["B"], a["C"], a["K"]
    p = a["P"]
    oldjoint = a["old_joint"].copy()
    lam = a["old"].copy()
    old = lam.copy()
    start = a["velocity"].copy()
    joint = np.linalg.solve(k, a["targets"] - b @ a["free"] - b @ w @ c.T @ lam)
    v = a["free"] + w @ (b.T @ joint + c.T @ lam)
    h = c @ p @ c.T
    np.testing.assert_allclose(h, h.T, atol=1e-9, rtol=1e-12)
    bias = a["rhs"] - c @ a["vbar"]
    arrays = [wp.array(x, dtype=wp.float64, device="cpu") for x in (c, p @ c.T, h, bias, a["gamma"], a["mu"])]
    evaluate, *_ = natural_map_evaluator(h, a["rhs"], a["gamma"], a["mu"])
    records = []
    mass = np.linalg.inv(w)
    for sweep in range(1, 33):
        la = wp.array(lam, dtype=wp.float64, device="cpu")
        va = wp.array(v, dtype=wp.float64, device="cpu")
        wp.launch(contact_sweep, dim=1, inputs=[*arrays, la, va, 0, len(a["mu"]), len(v)], device="cpu")
        lam, v = la.numpy(), va.numpy()
        joint = np.linalg.solve(k, a["targets"] - b @ a["free"] - b @ w @ c.T @ lam)
        impulse = b.T @ (joint - oldjoint) + c.T @ (lam - old)
        response = float(np.max(np.abs(v - start - w @ impulse)))
        residual = b @ v + a["diagonal"] * joint - a["targets"]
        ground = np.einsum("kij,ki->j", np.asarray(a["ground_wrenches"]), (lam - old).reshape(-1, 3))
        momentum = np.zeros(6)
        for i, body in enumerate(a["bodies"]):
            pi = impulse[6 * i : 6 * i + 6]
            momentum += np.r_[pi[:3], pi[3:] + np.cross(d["position"][body], pi[:3])]
        momentum += np.r_[ground[:3], ground[3:] + np.cross(d["position"][0], ground[:3])]
        energy = 0.5 * (v @ mass @ v - start @ mass @ start)
        work = impulse @ ((v + start) * 0.5)
        error = float(np.max(np.abs(evaluate(lam)[0])))
        assert response < 1e-9
        assert np.max(np.abs(residual)) < 1e-9
        assert np.max(np.abs(momentum)) < 1e-8
        assert abs(energy - work) < 1e-10
        records.append(
            {
                "sweep": sweep,
                "contact_kkt": error,
                "joint_residual": residual.tolist(),
                "momentum_error": momentum.tolist(),
                "work_error": float(energy - work),
                "response_error": response,
                "base_velocity": v[:6].tolist(),
                "contact_point_visits": sweep * len(a["mu"]),
            }
        )
    return {
        "phase": phase,
        "points": len(a["mu"]),
        "records": records,
        "normal_compliance": "Original captured gamma preserved, not recomputed from condensed diagonal",
    }


def main():
    """Run both native phase operators without reusing future contact modes."""
    wp.init()
    z = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    report = {
        "scope": "Frozen six-row exact joint elimination, all-point contact PGS, SOR1, production metric helper on CPU",
        "cases": [],
    }
    for phase in ("biased", "relax"):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_solved.")}
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        report["cases"].append(run(a, d, phase))
    path = Path("/tmp/colibri_two_body_condensed_pgs.json")
    path.write_text(json.dumps(report, indent=2))
    for case in report["cases"]:
        print(
            case["phase"],
            [
                (r["sweep"], r["contact_kkt"], max(map(abs, r["joint_residual"])))
                for r in case["records"]
                if r["sweep"] in (1, 4, 16, 32)
            ],
        )


if __name__ == "__main__":
    main()
