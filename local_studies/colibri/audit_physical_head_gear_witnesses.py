"""Compare saved live witnesses with the failed pose without regenerating geometry."""

import json
from pathlib import Path

import numpy as np

prefix = "/tmp/colibri_physical_head12_batch1024"
reference = np.load(prefix + ".npz")
data = np.load(prefix + "_contacts.npz")
gate = {
    key: reference[key].shape == data[key].shape
    and reference[key].dtype == data[key].dtype
    and reference[key].tobytes() == data[key].tobytes()
    for key in reference.files
}
assert all(gate.values()), gate
count = int(data["contact_count"][0])
a, b = data["contact_shape0"][:count], data["contact_shape1"][:count]
ids = np.flatnonzero(((a == 48) & (b == 50)) | ((a == 50) & (b == 48)))
q = data["q"].astype(float)
world = []
for side in (0, 1):
    points = data[f"contact_point{side}"][ids].astype(float).copy()
    for j, k in enumerate(ids):
        body = data["shape_body"][data[f"contact_shape{side}"][k]]
        if body >= 0:
            pose = q[body]
            v = pose[3:6]
            points[j] += 2 * np.cross(v, np.cross(v, points[j]) + pose[6] * points[j]) + pose[:3]
    world.append(points)
gaps = np.sum((world[1] - world[0]) * data["contact_normal"][ids], axis=1)
gaps -= data["contact_margin0"][ids] + data["contact_margin1"][ids]
report = {
    "trajectory_byte_gate": gate,
    "shape_pair": [48, 50],
    "shape_labels": data["shape_labels"][[48, 50]].tolist(),
    "live_contact_count": len(ids),
    "contact_ids": ids.tolist(),
    "transported_final_gaps_m": gaps.tolist(),
    "minimum_transported_final_gap_m": float(gaps.min()),
    "fresh_failure_depth_m_reported": 0.001624,
    "scope": "Stored live witness projection at failed pose, not a fresh SDF query or generation-time distance.",
    "conclusion": "Retained witness directions do not describe the deepest fresh feature. Raw generation versus reduction remains unresolved.",
}
Path(prefix + "_witness_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ("transported_final_gaps_m", "contact_ids")}, indent=2))
