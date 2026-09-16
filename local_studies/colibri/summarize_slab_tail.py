# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU transported-contact and velocity audit of the captured slab trajectory."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri.analyze_late_contacts import gaps, transform

path = Path(sys.argv[1])
source = np.load(path)
data = {name: source[name] for name in source.files}
records = []
for slot in np.argsort(data["step_ids"]):
    step = int(data["step_ids"][slot])
    if step < 680:
        continue
    count = int(data["counts"][slot])
    shapes = data["shapes"][slot, :count]
    labels = data["shape_labels"]
    keep = np.array(
        [str(labels[a]).startswith("TailRack/") or str(labels[b]).startswith("TailRack/") for a, b in shapes]
    )
    indices = np.flatnonzero(keep)
    pairs = {}
    for index in indices:
        pair = tuple(int(s) for s in shapes[index])
        pairs.setdefault(pair, []).append(index)
    for pair, raw in pairs.items():
        ids = np.array(raw)
        normals = data["normals"][slot, ids]
        bodies = data["shape_body"][np.array(pair)]
        record = {"step": step, "shapes": [str(labels[s]) for s in pair], "points": len(ids)}
        for phase in ("pre", "post"):
            q = data[phase + "_q"][slot]
            qd = data[phase + "_qd"][slot]
            separation = gaps(
                q,
                data["shape_body"],
                shapes[ids],
                data["point0"][slot, ids],
                data["point1"][slot, ids],
                normals,
                data["margins"][slot, ids],
            )
            world0 = transform(q[bodies[0]], data["point0"][slot, ids])
            world1 = transform(q[bodies[1]], data["point1"][slot, ids])
            midpoint = (world0 + world1) * 0.5
            velocities = []
            for body in bodies:
                com = transform(q[body], data["body_com"][body])
                velocities.append(qd[body, :3] + np.cross(qd[body, 3:], midpoint - com))
            vn = np.sum((velocities[1] - velocities[0]) * normals, axis=1)
            worst = int(np.argmin(separation))
            record[phase] = {
                "min_gap_m": float(separation[worst]),
                "worst_gap_vn": float(vn[worst]),
                "min_vn": float(vn.min()),
                "worst_gap_point": int(ids[worst]),
                "worst_gap_lambda": data["impulses"][slot, ids[worst]].tolist(),
            }
        record["normal_impulse_sum"] = float(data["impulses"][slot, ids, 0].sum())
        record["normal_nonzero"] = int(np.count_nonzero(data["impulses"][slot, ids, 0]))
        record["body_copies"] = data["body_copy_counts"][slot, bodies + 1].tolist()
        record["pair_slabs"] = sorted(set(data["point_slab"][slot, ids].tolist()))
        records.append(record)
output = path.with_suffix(".transport.json")
output.write_text(json.dumps(records, indent=2))
for row in records:
    if row["step"] >= 730:
        print(row)
