"""CPU spatial-bank alternatives; post pose is used only for coverage audit."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.analyze_velocity4_native_tail import rotate
from local_studies.colibri.classify_velocity4_lost_witnesses import bins
from newton._src.geometry.contact_reduction import FACE_NORMALS


def main():
    """Compare geometry-only bank winners, preserving explicit count limitations."""
    raw = np.load("/tmp/colibri_velocity4_midpoint_9291_geometric_raw.npz")
    kept = np.load("/tmp/colibri_velocity4_midpoint_9291_geometric_reduced.npz")
    trace = np.load("/tmp/colibri_velocity4_native_tail.trace.npz")
    slot = int(np.flatnonzero(trace["step_ids"] == 9291)[0])
    q = trace["generation_q"][4 * slot]
    a, b = trace["shape_body"][[36, 37]]
    x0 = q[a, :3] + rotate(q[a, 3:], raw["point0"])
    x1 = q[b, :3] + rotate(q[b, 3:], raw["point1"])
    center = (x0 + x1) * 0.5
    normalbins = bins(raw["normals"])
    faces = np.asarray(FACE_NORMALS).reshape(20, 3).astype(float)
    depth = raw["pre_gap"]
    threshold = 0.0002124052552971989
    directions = np.column_stack((np.cos(np.arange(6) * 2 * np.pi / 6), np.sin(np.arange(6) * 2 * np.pi / 6)))
    selected = {
        name: set() for name in ("current_spatial", "A_outer_spatial", "B_all_spatial_depth", "C_overlap_priority")
    }
    for binid in np.unique(normalbins):
        indices = np.flatnonzero(normalbins == binid)
        face = faces[binid]
        ref = np.array([0.0, 1.0, 0.0]) if abs(face[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = ref - np.dot(ref, face) * face
        u /= np.linalg.norm(u)
        v = np.cross(face, u)
        scores = center @ np.column_stack((u, v)) @ directions.T
        for name, winners in selected.items():
            candidates = indices
            if name == "current_spatial" and np.any(depth[indices] < threshold):
                candidates = indices[depth[indices] < threshold]
            if name == "A_outer_spatial":
                candidates = indices[depth[indices] >= threshold]
            if name == "C_overlap_priority" and np.any(depth[indices] <= 0):
                candidates = indices[depth[indices] <= 0]
            if not len(candidates):
                continue
            for direction in range(6):
                winners.add(int(candidates[np.argmax(scores[candidates, direction])]))
            if name == "B_all_spatial_depth":
                winners.add(int(indices[np.argmin(depth[indices])]))
    out = {}
    for name, subset in selected.items():
        ids = np.array(sorted(subset), dtype=int)
        post = raw["post_gap"][ids]
        out[name] = {
            "selected_count": len(ids),
            "post_min_gap": float(post.min()),
            "future_below_500um": int(np.count_nonzero(post < -0.0005)),
            "selected_ids": ids.tolist(),
            "selected_future_danger_ids": ids[post < -0.0005].tolist(),
            "selected_gap_minmax": [float(depth[ids].min()), float(depth[ids].max())],
            "union_with_current42_count_upper_bound": len(kept["pre_gap"]) + len(ids),
        }
    report = {
        "scope": "FP64 CPU spatial geometry emulation; same-bin score ties use first raw index. No post-pose selection. Counts exclude independent voxel bank; union counts are conservative upper bounds, not exact canonical export counts.",
        "candidates": out,
    }
    Path("/tmp/colibri_velocity4_spatial_alternatives.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
