# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare cached and independently regenerated TailRack/Pinion contacts."""

import argparse
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.validate_phoenx import ContactAudit
from newton.examples.kamino.example_kamino_colibri import build_scene


def transform(q, points):
    xyz = q[:3]
    vector = q[3:6]
    scalar = q[6]
    return points + 2.0 * np.cross(vector, np.cross(vector, points) + scalar * points) + xyz


def gaps(q, shape_body, shapes, point0, point1, normals, margins):
    world0 = point0.copy()
    world1 = point1.copy()
    for i, (s0, s1) in enumerate(shapes):
        b0, b1 = shape_body[s0], shape_body[s1]
        if b0 >= 0:
            world0[i] = transform(q[b0], point0[i])
        if b1 >= 0:
            world1[i] = transform(q[b1], point1[i])
    return np.sum((world1 - world0) * normals, axis=1) - np.sum(margins, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--offset-map", type=Path, default=Path("/tmp/colibri_physx_exact_contact_offsets_m.json"))
    parser.add_argument("--all-rack", action="store_true")
    parser.add_argument("--start-step", type=int, default=0)
    args = parser.parse_args()
    data = np.load(args.trace)
    builder = build_scene(
        body_count=36, fix_base=False, contact_gap=0.001, source_contact_offsets=True, mesh_cylinders=True
    )
    offsets = json.loads(args.offset_map.read_text())
    for i, label in enumerate(builder.shape_label):
        if label in offsets:
            builder.shape_gap[i] = offsets[label]
    model = builder.finalize(skip_validation_joints=True)
    state = model.state()
    audit = ContactAudit(model)
    shape_body = data["shape_body"]
    labels = data["shape_labels"]
    rack = {i for i, label in enumerate(labels) if str(label).startswith("TailRack/")}
    pinion = {i for i, label in enumerate(labels) if str(label).startswith("TailPinion/")}

    def selected(shapes):
        if args.all_rack:
            return np.array([int(a) in rack or int(b) in rack for a, b in shapes], dtype=bool)
        return np.array(
            [(int(a) in rack and int(b) in pinion) or (int(b) in rack and int(a) in pinion) for a, b in shapes],
            dtype=bool,
        )

    records = []
    for slot in np.argsort(data["step_ids"]):
        step = int(data["step_ids"][slot])
        if step < args.start_step:
            continue
        count = int(data["counts"][slot])
        shapes = data["shapes"][slot, :count]
        keep = selected(shapes)
        cached = {}
        for phase in ("pre", "post"):
            q = data[phase + "_q"][slot]
            separation = gaps(
                q,
                shape_body,
                shapes[keep],
                data["point0"][slot, :count][keep],
                data["point1"][slot, :count][keep],
                data["normals"][slot, :count][keep],
                data["margins"][slot, :count][keep],
            )
            state.body_q.assign(q)
            metric = audit.check(state)
            c = audit.contacts
            fresh_count = int(c.rigid_contact_count.numpy()[0])
            fresh_shapes = np.column_stack(
                (c.rigid_contact_shape0.numpy()[:fresh_count], c.rigid_contact_shape1.numpy()[:fresh_count])
            )
            fresh_keep = selected(fresh_shapes)
            fresh_separation = audit.separation.numpy()[:fresh_count][fresh_keep]
            cached[phase] = {
                "cached_gaps_m": separation.tolist(),
                "fresh_gaps_m": fresh_separation.tolist(),
                "fresh_shapes": fresh_shapes[fresh_keep].tolist(),
                "fresh_point0": c.rigid_contact_point0.numpy()[:fresh_count][fresh_keep].tolist(),
                "fresh_point1": c.rigid_contact_point1.numpy()[:fresh_count][fresh_keep].tolist(),
                "fresh_normals": c.rigid_contact_normal.numpy()[:fresh_count][fresh_keep].tolist(),
                "global_metric": metric,
            }
        records.append(
            {
                "outer_step": step,
                "frame_after": (step + 1) / 2.0,
                "cached_shapes": shapes[keep].tolist(),
                "cached_normals": data["normals"][slot, :count][keep].tolist(),
                "cached_impulses": data["impulses"][slot, :count][keep].tolist(),
                **cached,
            }
        )
    output = args.trace.with_suffix(".rack.analysis.json" if args.all_rack else ".analysis.json")
    output.write_text(json.dumps({"records": records, "color_audits": json.loads(str(data["color_audits"]))}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
