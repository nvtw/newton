"""Quantify strictly interior prior tangent impulses reset during preparation."""

import json
from pathlib import Path

import numpy as np

x = np.load("/tmp/colibri_base_frame_totalnormal_frame330.npz")
keys = ("impulses", "lambdas", "headers", "column_count", "derived")
a = {k: x[k] for k in x.files if k.startswith(("warm_before.", "warm_solved.")) and k.split(".", 1)[1] in keys}
rows = []
for i in range(1, 60):
    if i == 30:
        continue
    old = a["warm_before.impulses"][i]
    new = a["warm_solved.impulses"][i]
    h = a["warm_before.headers"][i].view(np.int32)
    for col in range(int(a["warm_before.column_count"][i, 0])):
        mu = float(a["warm_before.headers"][i, 3, col])
        for p in range(h[5, col], h[5, col] + h[6, col]):
            limit = mu * old[0, p]
            ratio = float(np.linalg.norm(old[1:, p]) / limit) if limit > 0 else 0
            if 0.98 <= ratio < 1:
                changed = not np.array_equal(a["warm_before.lambdas"][i, 6:12, p], a["warm_solved.lambdas"][i, 6:12, p])
                rows.append(
                    {
                        "slot": i,
                        "point": p,
                        "body": h[1:3, col].tolist(),
                        "ratio": ratio,
                        "anchor_changed": changed,
                        "tangent_cleared": bool(np.all(new[1:, p] == 0)),
                        "current_bias": float(a["warm_solved.derived"][i, 3, p]),
                    }
                )
out = {
    "scope": "Actual preparation/warm sweep before to after within generation. Near-cap prior state does not prove current trial would stick.",
    "near_cap_count": len(rows),
    "cleared_changed": sum(r["tangent_cleared"] and r["anchor_changed"] for r in rows),
    "overlap_cleared_changed": sum(
        r["tangent_cleared"] and r["anchor_changed"] and r["current_bias"] <= 0 for r in rows
    ),
    "rows": rows,
}
Path("/tmp/colibri_anchor_break_ring.json").write_text(json.dumps(out, indent=2))
print({k: v for k, v in out.items() if k != "rows"})
