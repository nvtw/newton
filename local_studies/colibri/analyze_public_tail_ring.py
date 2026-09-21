"""Summarize cached witness transport and impulses in the public failure ring."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.joint_accuracy import measure, rotate

archive = np.load("/tmp/colibri_public_group4_tail.trace.npz")
d = {key: archive[key] for key in archive.files}
labels = list(d["body_labels"])
shape_body = d["shape_body"]
records = []
for slot in np.argsort(d["step_ids"]):
    step = int(d["step_ids"][slot])
    if step < 0:
        continue
    count = int(d["counts"][slot])
    pre, post = d["pre_q"][slot], d["post_q"][slot]
    rows = []
    for k in range(count):
        s0, s1 = d["shapes"][slot, k]
        names = [str(d["shape_labels"][s0]), str(d["shape_labels"][s1])]
        if not any("TailRack" in x or "TailPinion" in x for x in names):
            continue
        gap = []
        for q in (pre, post):
            points = []
            for s, p in ((s0, d["point0"][slot, k]), (s1, d["point1"][slot, k])):
                b = shape_body[s]
                points.append(p if b < 0 else q[b, :3] + rotate(q[b, 3:], p))
            gap.append(float(np.dot(points[1] - points[0], d["normals"][slot, k]) - d["margins"][slot, k].sum()))
        rows.append(
            dict(index=k, shapes=names, pre_gap=gap[0], post_gap=gap[1], impulse=d["impulses"][slot, k].tolist())
        )
    pinion = labels.index("TailPinion")
    joint = next(r for r in measure(post, labels)["joints"] if r["joint"] == "TailMount/TailPinion")
    records.append(
        dict(
            step=step,
            time=(step + 1) / 120,
            count=count,
            joint=joint,
            pinion_omega=float(np.linalg.norm(d["post_qd"][slot, pinion, 3:])),
            contacts=rows,
        )
    )
Path("/tmp/colibri_public_group4_tail.ring.json").write_text(json.dumps(records, indent=2))
for r in records[-20:]:
    rows = r["contacts"]
    worst = min(rows, key=lambda x: x["post_gap"])
    strongest = max(rows, key=lambda x: abs(x["impulse"][0]))
    print(
        r["step"],
        r["joint"]["axis_error_rad"],
        r["pinion_omega"],
        worst["shapes"],
        worst["pre_gap"],
        worst["post_gap"],
        strongest["shapes"],
        strongest["impulse"][0],
    )
