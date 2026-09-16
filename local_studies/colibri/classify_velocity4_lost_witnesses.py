"""Classify lost midpoint candidates using only generation geometry and twist."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.analyze_velocity4_native_tail import rotate
from newton._src.geometry.contact_reduction import FACE_NORMALS


def bins(normals):
    """Mirror the canonical icosahedral candidate ranges using stored FP32 faces."""
    faces = np.asarray(FACE_NORMALS).reshape(20, 3)
    out = []
    for n in normals:
        lo, hi = (0, 5) if n[1] > 0.65 else (15, 20) if n[1] < -0.65 else (0, 15) if n[1] >= 0 else (5, 20)
        out.append(lo + int(np.argmax(faces[lo:hi] @ n)))
    return np.array(out)


def main():
    """Separate retrospective lost-feature selection from causal generation metrics."""
    raw = np.load("/tmp/colibri_velocity4_midpoint_9291_geometric_raw.npz")
    kept = np.load("/tmp/colibri_velocity4_midpoint_9291_geometric_reduced.npz")
    trace = np.load("/tmp/colibri_velocity4_native_tail.trace.npz")
    slot = int(np.flatnonzero(trace["step_ids"] == 9291)[0])
    q = trace["generation_q"][4 * slot].astype(float)
    qd = trace["generation_qd"][4 * slot].astype(float)
    a, b = trace["shape_body"][[36, 37]]
    x0 = q[a, :3] + rotate(q[a, 3:], raw["point0"])
    x1 = q[b, :3] + rotate(q[b, 3:], raw["point1"])
    c0 = q[a, :3] + rotate(q[a, 3:], trace["body_com"][a])
    c1 = q[b, :3] + rotate(q[b, 3:], trace["body_com"][b])
    v0 = qd[a, :3] + np.cross(qd[a, 3:], x0 - c0)
    v1 = qd[b, :3] + np.cross(qd[b, 3:], x1 - c1)
    vn = np.sum((v1 - v0) * raw["normals"], axis=1)
    predicted = raw["pre_gap"] + vn / 120
    lost = raw["post_gap"] < -0.0005
    rawbins = bins(raw["normals"])
    keepbins = bins(kept["normals"])
    normaldot = raw["normals"] @ kept["normals"].T
    nearest_angle = np.arccos(np.clip(normaldot.max(axis=1), -1, 1))
    rows = []
    for binid in np.unique(rawbins[lost]):
        mask = lost & (rawbins == binid)
        kmask = keepbins == binid
        rows.append(
            {
                "bin": int(binid),
                "lost": int(mask.sum()),
                "kept": int(kmask.sum()),
                "lost_gap_range": [float(raw["pre_gap"][mask].min()), float(raw["pre_gap"][mask].max())],
                "kept_gap_range": [float(kept["pre_gap"][kmask].min()), float(kept["pre_gap"][kmask].max())]
                if kmask.any()
                else None,
                "lost_vn_range": [float(vn[mask].min()), float(vn[mask].max())],
                "lost_predicted_gap_range": [float(predicted[mask].min()), float(predicted[mask].max())],
                "lost_initially_closing": int(np.count_nonzero(vn[mask] < 0)),
                "lost_predicted_crossing": int(np.count_nonzero(predicted[mask] <= 0)),
                "nearest_retained_normal_angle_range_rad": [
                    float(nearest_angle[mask].min()),
                    float(nearest_angle[mask].max()),
                ],
            }
        )
    deepest = int(np.argmin(raw["post_gap"]))
    report = {
        "raw_count": len(vn),
        "kept_count": len(kept["normals"]),
        "lost_post_below_500um": int(lost.sum()),
        "scope": "Post gap identifies retrospective candidates only; vn/prediction use actual9291 generation pose and twist",
        "bins": rows,
        "deepest": {
            "raw_index": deepest,
            "normal": raw["normals"][deepest].tolist(),
            "bin": int(rawbins[deepest]),
            "pre_gap": float(raw["pre_gap"][deepest]),
            "generation_vn": float(vn[deepest]),
            "predicted_gap_120Hz": float(predicted[deepest]),
            "transported_actual_post_gap": float(raw["post_gap"][deepest]),
            "point0_body_local": raw["point0"][deepest].tolist(),
            "point1_body_local": raw["point1"][deepest].tolist(),
        },
    }
    Path("/tmp/colibri_velocity4_lost_witnesses.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
