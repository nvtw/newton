"""Audit actual retained contact rows and final phases from the native tail ring."""

import argparse
import json
from pathlib import Path

import numpy as np


def rotate(q, v):
    """Apply saved XYZW quaternion using the solver's vector convention."""
    return v + 2 * np.cross(q[..., :3], np.cross(q[..., :3], v) + q[..., 3:] * v)


def main():
    """Require exact replay before interpreting physical averaged row velocities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/tmp/colibri_velocity4_native_tail")
    prefix = parser.parse_args().prefix
    assert all(json.loads(Path(prefix + ".prefix.json").read_text()).values())
    with np.load(prefix + ".trace.npz", allow_pickle=True) as archive:
        d = {k: archive[k] for k in archive.files}
    filters = {tuple(pair) for pair in d["shape_filter_pairs"]}
    records = []
    pairs = ((36, 37), (36, 64), (33, 34), (64, 103), (37, 65))
    com = np.vstack((np.zeros((1, 3)), d["body_com"]))
    for slot in np.argsort(d["step_ids"]):
        step = int(d["step_ids"][slot])
        last = 4 * slot + 3
        count = int(d["rigid_contact_count"][last, 0])
        assert count > 0, (step, "missing ring entry")
        shapes = np.column_stack((d["rigid_contact_shape0"][last, :count], d["rigid_contact_shape1"][last, :count]))
        owner = {}
        h = d["headers"][last].view(np.int32)
        for column in range(int(d["column_count"][last, 0])):
            for k in range(h[5, column], h[5, column] + h[6, column]):
                assert k not in owner
                owner[k] = (int(h[1, column]), int(h[2, column]))
        for pair in pairs:
            points = np.flatnonzero(np.all(shapes == pair, axis=1))
            if not len(points):
                continue
            a, b = d["shape_body"][list(pair)] + 1
            assert all(owner[k] == (a, b) for k in points)
            point0 = d["rigid_contact_point0"][last, points]
            point1 = d["rigid_contact_point1"][last, points]
            margins = d["rigid_contact_margin0"][last, points] + d["rigid_contact_margin1"][last, points]
            gen = d["generation_q"][4 * slot]
            gen0 = gen[a - 1, :3] + rotate(gen[a - 1, 3:], point0)
            gen1 = gen[b - 1, :3] + rotate(gen[b - 1, 3:], point1)
            gn = d["rigid_contact_normal"][last, points]
            generation_gap = np.sum((gen1 - gen0) * gn, axis=1) - margins
            phases = []
            for phase, name in enumerate(
                ("before_final_biased", "after_final_biased", "before_final_relax", "after_final_relax")
            ):
                row = 4 * slot + phase
                n = d["contact_lambdas"][row, :3][:, points].T.astype(float)
                t = d["contact_lambdas"][row, 3:6][:, points].T.astype(float)
                r0 = d["contact_derived"][row, 9:12][:, points].T.astype(float)
                r1 = d["contact_derived"][row, 12:15][:, points].T.astype(float)
                v = d["body_velocity"][row].astype(float)
                w = d["body_angular_velocity"][row].astype(float)
                rel = v[b] + np.cross(w[b], r1) - v[a] - np.cross(w[a], r0)
                vn = np.sum(rel * n, axis=1)
                vt = np.column_stack((np.sum(rel * t, axis=1), np.sum(rel * np.cross(n, t), axis=1)))
                pos, q = d["body_position"][row], d["body_orientation"][row]
                x0 = pos[a] + rotate(q[a], point0 - com[a])
                x1 = pos[b] + rotate(q[b], point1 - com[b])
                gap = np.sum((x1 - x0) * n, axis=1) - margins
                impulse = d["contact_impulses"][row][:, points].T
                bias = d["contact_derived"][row, 3, points]
                phases.append(
                    {
                        "phase": name,
                        "minimum_transported_gap_m": float(gap.min()),
                        "minimum_vn_m_s": float(vn.min()),
                        "max_slip_m_s": float(np.linalg.norm(vt, axis=1).max()),
                        "normal_impulse_sum_Ns": float(impulse[:, 0].sum()),
                        "max_normal_impulse_Ns": float(impulse[:, 0].max()),
                        "nonzero_normal_points": int(np.count_nonzero(impulse[:, 0])),
                        "max_tangent_impulse_Ns": float(np.linalg.norm(impulse[:, 1:], axis=1).max()),
                        "bias_min_max": [float(bias.min()), float(bias.max())],
                        "copy_counts": [int(d["copy_count"][row, a]), int(d["copy_count"][row, b])],
                    }
                )
            records.append(
                {
                    "step": step,
                    "generation_time": step / 120,
                    "post_time": (step + 1) / 120,
                    "pair": list(pair),
                    "labels": d["shape_labels"][list(pair)].tolist(),
                    "filtered": tuple(pair) in filters or tuple(reversed(pair)) in filters,
                    "count": len(points),
                    "point_ids": points.tolist(),
                    "generation_gap_min_m": float(generation_gap.min()),
                    "phases": phases,
                }
            )
    Path(prefix + ".analysis.json").write_text(json.dumps(records, indent=2))
    for r in records:
        if r["pair"] in ([36, 37], [36, 64], [64, 103]):
            print(
                r["step"],
                r["pair"],
                r["count"],
                "gengap",
                r["generation_gap_min_m"],
                [
                    (p["minimum_transported_gap_m"], p["minimum_vn_m_s"], p["normal_impulse_sum_Ns"])
                    for p in r["phases"]
                ],
            )


if __name__ == "__main__":
    main()
