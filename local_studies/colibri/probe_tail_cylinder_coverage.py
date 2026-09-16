# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare raw and reduced Cylinder33 contact features at the saved missed approach."""

import json

import numpy as np
import warp as wp

import newton
from local_studies.colibri.analyze_late_contacts import gaps
from newton.examples.kamino.example_kamino_colibri import build_scene


def main():
    trace = np.load("/tmp/colibri_tail_trace_30x1_exact.trace.npz")
    offsets = json.load(open("/tmp/colibri_physx_exact_contact_offsets_m.json"))
    builder = build_scene(
        body_count=36, fix_base=False, contact_gap=0.001, source_contact_offsets=True, mesh_cylinders=True
    )
    for i, label in enumerate(builder.shape_label):
        if label in offsets:
            builder.shape_gap[i] = offsets[label]
    model = builder.finalize(skip_validation_joints=True)
    state = model.state()
    slot = int(np.flatnonzero(trace["step_ids"] == 592)[0])
    q = trace["pre_q"][slot]
    state.body_q.assign(q)
    state.body_qd.assign(trace["pre_qd"][slot])
    pair = [model.shape_label.index("TailRack/Tail_Rack"), model.shape_label.index("TailMount/Cylinder_33")]
    shape_body = model.shape_body.numpy()
    result = {}
    for name, reduction, cap in (("reduced", True, None), ("raw", False, None), ("predictive", True, 0.005)):
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array([pair], dtype=wp.vec2i, device=model.device),
            reduce_contacts=reduction,
            rigid_contact_max=32768,
            contact_matching="disabled",
            speculative_contact_gap_max=cap,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=1.0 / 120.0)
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert count < 32768, count
        shapes = np.column_stack(
            (contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
        )
        p0 = contacts.rigid_contact_point0.numpy()[:count]
        p1 = contacts.rigid_contact_point1.numpy()[:count]
        normal = contacts.rigid_contact_normal.numpy()[:count]
        margins = np.column_stack(
            (contacts.rigid_contact_margin0.numpy()[:count], contacts.rigid_contact_margin1.numpy()[:count])
        )
        separation = gaps(q, shape_body, shapes, p0, p1, normal, margins)
        rack_points = np.where((shapes[:, 0] == pair[0])[:, None], p0, p1)
        opposite = rack_points[:, 2] < 0
        np.savez(
            "/tmp/colibri_cylinder33_" + name + "_592.npz",
            shapes=shapes,
            point0=p0,
            point1=p1,
            normal=normal,
            separation=separation,
        )
        result[name] = {
            "count": count,
            "min_gap_m": float(np.min(separation)),
            "opposite_edge_count": int(np.count_nonzero(opposite)),
            "opposite_min_gap_m": float(np.min(separation[opposite])) if np.any(opposite) else None,
        }
    print(json.dumps(result, indent=2))
    with open("/tmp/colibri_cylinder33_coverage_592.json", "w") as out:
        json.dump(result, out, indent=2)


if __name__ == "__main__":
    main()
