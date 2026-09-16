"""Transport failed-pose witnesses backward; this does not resample the SDF."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.analyze_velocity4_native_tail import rotate


def main():
    prefix = "/tmp/colibri_spatial_priority_velocity4_trace18000"
    trace = np.load(prefix + ".trace.npz")
    query = np.load(prefix + ".query_29839_geometric_raw_post.npz")
    slot = int(np.flatnonzero(trace["step_ids"] == 29839)[0])
    q = trace["generation_q"][4 * slot].astype(float)
    qd = trace["generation_qd"][4 * slot].astype(float)
    a, b = trace["shape_body"][[46, 48]]
    x0 = q[a, :3] + rotate(q[a, 3:], query["point0"])
    x1 = q[b, :3] + rotate(q[b, 3:], query["point1"])
    c0 = q[a, :3] + rotate(q[a, 3:], trace["body_com"][a])
    c1 = q[b, :3] + rotate(q[b, 3:], trace["body_com"][b])
    v0 = qd[a, :3] + np.cross(qd[a, 3:], x0 - c0)
    v1 = qd[b, :3] + np.cross(qd[b, 3:], x1 - c1)
    n = query["normals"]
    pre = np.sum((x1 - x0) * n, axis=1) - query["margins"].sum(axis=1)
    vn = np.sum((v1 - v0) * n, axis=1)
    post = query["post_gap"]
    mask = post < -0.0005
    deepest = int(np.argmin(post))
    result = {
        "scope": "Backward transported post-witness projection, NOT fresh pre-pose signed-distance resampling.",
        "count": int(mask.sum()),
        "base_pair_gap": float(trace["shape_gap"][[46, 48]].sum()),
        "initial_projected_gap_range": [float(pre[mask].min()), float(pre[mask].max())],
        "initial_vn_range": [float(vn[mask].min()), float(vn[mask].max())],
        "predicted_gap_range": [float((pre + vn / 120)[mask].min()), float((pre + vn / 120)[mask].max())],
        "deepest": {
            "id": deepest,
            "post_gap": float(post[deepest]),
            "pre_projected_gap": float(pre[deepest]),
            "initial_vn": float(vn[deepest]),
            "predicted_gap": float(pre[deepest] + vn[deepest] / 120),
            "normal": n[deepest].tolist(),
            "point0": query["point0"][deepest].tolist(),
            "point1": query["point1"][deepest].tolist(),
        },
    }
    Path(prefix + ".post_witness_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
