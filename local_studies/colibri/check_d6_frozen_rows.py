"""Frozen physical two-body D6-style row ordering versus bilateral block control.

CPU-only diagnostic. Joint updates use the exact native block equations in
FP64; contact sweeps invoke the production metric-friction helper on Warp CPU.
This is not a live solver or bit-equivalence claim for native FP32 scatter.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.coupled_support_online import assemble_snapshot
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric


@wp.kernel(enable_backward=False)
def contact_sweep(
    c: wp.array2d[wp.float64],
    wct: wp.array2d[wp.float64],
    h: wp.array2d[wp.float64],
    bias: wp.array[wp.float64],
    gamma: wp.array[wp.float64],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    velocity: wp.array[wp.float64],
    first: int,
    count: int,
    size: int,
):
    """Apply original-order normal and production metric tangent updates."""
    for p in range(first, count):
        row = 3 * p
        vn = bias[row]
        for j in range(size):
            vn += c[row, j] * velocity[j]
        oldn = lam[row]
        newn = wp.max(wp.float64(0.0), oldn - (vn + gamma[p] * oldn) / (h[row, row] + gamma[p]))
        dn = newn - oldn
        lam[row] = newn
        for j in range(size):
            velocity[j] += wct[j, row] * dn
        vt1 = bias[row + 1]
        vt2 = bias[row + 2]
        for j in range(size):
            vt1 += c[row + 1, j] * velocity[j]
            vt2 += c[row + 2, j] * velocity[j]
        old1 = lam[row + 1]
        old2 = lam[row + 2]
        radius = wp.float32(mu[p] * newn)
        tangent = contact_project_friction_metric(
            wp.float32(h[row + 1, row + 1]),
            wp.float32(h[row + 1, row + 2]),
            wp.float32(h[row + 2, row + 2]),
            wp.float32(vt1),
            wp.float32(vt2),
            wp.float32(old1),
            wp.float32(old2),
            radius,
            radius,
        )
        lam[row + 1] = wp.float64(tangent[0])
        lam[row + 2] = wp.float64(tangent[1])
        for j in range(size):
            velocity[j] += wct[j, row + 1] * (lam[row + 1] - old1) + wct[j, row + 2] * (lam[row + 2] - old2)


def transform(b, w, r, kind):
    """Transform hard rows only; keep the compliant drive in original units."""
    drive = list(np.flatnonzero(r > 0))
    hard = list(np.flatnonzero(r == 0))
    angular = [i for i in hard if np.max(np.abs(b[i].reshape(-1, 6)[:, :3])) == 0]
    linear = [i for i in hard if i not in angular]
    order = drive + angular + linear
    t = np.eye(len(r))[order]
    groups = []
    if kind == "d6_groups":
        groups = [list(range(len(drive), len(drive) + len(angular))), list(range(len(drive) + len(angular), len(r)))]
    elif kind == "hard_orthogonal":
        groups = [list(range(len(drive), len(r)))]
    for group in groups:
        for k, i in enumerate(group):
            for j in group[:k]:
                bi, bj = t[i] @ b, t[j] @ b
                denominator = bj @ w @ bj
                if denominator <= 0:
                    raise RuntimeError("Unresolved hard row; no mode deletion allowed")
                t[i] -= (bi @ w @ bj / denominator) * t[j]
    rr = t @ np.diag(r) @ t.T
    np.testing.assert_allclose(rr, np.diag(np.diag(rr)), atol=1e-15, rtol=0)
    np.testing.assert_allclose(rr[: len(drive), : len(drive)], np.diag(r[drive]), atol=0, rtol=0)
    assert abs(np.linalg.det(t)) > 0.99
    return t, order


def run(a, d, kind, phase):
    """Compare equal joint/contact sweeps and audit physical impulse accounting."""
    w, b, c = a["W"], a["B"], a["C"]
    r, target = a["diagonal"], a["targets"]
    v = a["velocity"].copy()
    joint = a["old_joint"].copy()
    lam = a["old"].copy()
    h = c @ w @ c.T
    bias = a["rhs"] - c @ a["vbar"]
    t, order = transform(b, w, r, kind)
    bt = t @ b
    rt = t @ np.diag(r) @ t.T
    tt = t @ target
    # Independent original-equation gate: transformed compliance and impulse
    # backmap must reproduce the simultaneous original block solution.
    original_root = np.linalg.solve(b @ w @ b.T + np.diag(r), target - b @ a["free"])
    transformed_root = np.linalg.solve(bt @ w @ bt.T + rt, tt - bt @ a["free"])
    np.testing.assert_allclose(t.T @ transformed_root, original_root, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(w @ bt.T @ transformed_root, w @ b.T @ original_root, atol=1e-12, rtol=1e-12)
    eta = np.linalg.solve(t.T, joint)
    arrays = [wp.array(x, dtype=wp.float64, device="cpu") for x in (c, w @ c.T, h, bias, a["gamma"], a["mu"])]
    records = []
    mass = np.linalg.inv(w)
    start = v.copy()
    oldjoint = joint.copy()
    oldlam = lam.copy()

    def residual(vv, jj):
        return b @ vv + r * jj - target

    # Preserve the actual captured contact-column boundaries, including partial
    # chunks and skipped positive-bias points in the relaxation phase.
    headers = d["headers"].view(np.int32)
    chunks = []
    covered = []
    for col in range(int(d["column_count"][0])):
        begin, count = map(int, headers[5:7, col])
        indices = [i for i, p in enumerate(a["points"]) if begin <= p < begin + count]
        if indices:
            assert indices == list(range(indices[0], indices[-1] + 1))
            chunks.append((indices[0], indices[-1] + 1))
            covered.extend(indices)
    assert covered == list(range(len(a["mu"])))
    joint_applications = 0

    def joint_update():
        nonlocal v, joint, eta, joint_applications
        joint_applications += 1
        if kind in ("block", "joint_last", "symmetric", "native_chunk_joint"):
            delta = np.linalg.solve(b @ w @ b.T + np.diag(r), -residual(v, joint))
            joint += delta
            v += w @ b.T @ delta
        else:
            for i in range(len(r)):
                delta = -(bt[i] @ v + rt[i] @ eta - tt[i]) / (bt[i] @ w @ bt[i] + rt[i, i])
                eta[i] += delta
                v += w @ bt[i] * delta
            joint = t.T @ eta

    def contacts(first, last):
        nonlocal lam, v
        la = wp.array(lam, dtype=wp.float64, device="cpu")
        va = wp.array(v, dtype=wp.float64, device="cpu")
        wp.launch(contact_sweep, dim=1, inputs=[*arrays, la, va, first, last, len(v)], device="cpu")
        lam, v = la.numpy(), va.numpy()

    for sweep in range(1, 17):
        if kind not in ("joint_last", "native_chunk_joint"):
            joint_update()
        before = residual(v, joint)
        if kind == "native_chunk_joint":
            for first, last in chunks:
                contacts(first, last)
                joint_update()
        else:
            contacts(0, len(a["mu"]))
            if kind in ("joint_last", "symmetric"):
                joint_update()
        after = residual(v, joint)
        impulse = b.T @ (joint - oldjoint) + c.T @ (lam - oldlam)
        response = float(np.max(np.abs(v - start - w @ impulse)))
        ground = np.einsum("kij,ki->j", np.asarray(a["ground_wrenches"]), (lam - oldlam).reshape(-1, 3))
        momentum = np.zeros(6)
        for i, body in enumerate(a["bodies"]):
            p = impulse[6 * i : 6 * i + 6]
            momentum += np.r_[p[:3], p[3:] + np.cross(d["position"][body], p[:3])]
        momentum += np.r_[ground[:3], ground[3:] + np.cross(d["position"][0], ground[:3])]
        work = impulse @ ((v + start) * 0.5)
        energy = 0.5 * (v @ mass @ v - start @ mass @ start)
        # Freeze the current joint impulse when evaluating the physical contact law.
        q = c @ (a["free"] + w @ b.T @ joint) + bias
        evaluate, *_ = natural_map_evaluator(h, q, a["gamma"], a["mu"])
        error = float(np.max(np.abs(evaluate(lam)[0])))
        assert response < 1e-10
        assert np.max(np.abs(momentum)) < 1e-8
        assert abs(work - energy) < 1e-10
        if True:
            records.append(
                {
                    "sweep": sweep,
                    "joint_before_contact": float(abs(before).max()),
                    "joint_after_contact": float(abs(after).max()),
                    "checkpoint": "After complete schedule; joint-last variants include final joint solve",
                    "hard_joint_max": float(abs(after[r == 0]).max()),
                    "hard_joint_l2": float(np.linalg.norm(after[r == 0])),
                    "joint_l2": float(np.linalg.norm(after)),
                    "original_joint_residual": after.tolist(),
                    "joint_applications": joint_applications,
                    "joint_row_visits": joint_applications * len(r),
                    "contact_point_visits": sweep * len(a["mu"]),
                    "contact_scalar_row_visits": sweep * 3 * len(a["mu"]),
                    "drive_before": before[r > 0].tolist(),
                    "drive_after": after[r > 0].tolist(),
                    "contact_kkt": error,
                    "momentum_error": momentum.tolist(),
                    "response_error": response,
                    "work_error": float(work - energy),
                    "base_velocity": v[:6].tolist(),
                }
            )
    return {
        "kind": kind,
        "phase": phase,
        "order": list(map(int, order)),
        "transform": t.tolist(),
        "contact_chunk_sizes": [last - first for first, last in chunks],
        "records": records,
    }


def main():
    """Run the same actual captured two-body geometry under seven local schedules."""
    wp.init()
    source = Path("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    z = np.load(source)
    result = {
        "source": str(source),
        "scope": "Frozen physical unsplit equations, total-normal Coulomb; Warp CPU native metric friction; joint FP64 equation control, not native full-dispatch replay",
        "cases": [],
    }
    for phase in ("biased", "relax"):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_solved.")}
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        for kind in (
            "block",
            "ordered",
            "d6_groups",
            "hard_orthogonal",
            "joint_last",
            "symmetric",
            "native_chunk_joint",
        ):
            result["cases"].append(run(a, d, kind, phase))
    path = Path("/tmp/colibri_d6_frozen_schedules.json")
    path.write_text(json.dumps(result, indent=2))
    print(path)
    for case in result["cases"]:
        last = case["records"][-1]
        print(case["phase"], case["kind"], last["drive_after"], last["contact_kkt"])


if __name__ == "__main__":
    main()
