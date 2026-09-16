# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay frozen tail pair inputs with raw/reduced predictive generation."""

import json

import numpy as np
import warp as wp

import newton
from local_studies.colibri.analyze_late_contacts import gaps
from newton.examples.kamino.example_kamino_colibri import build_scene

archive = np.load("/tmp/colibri_public_group4_tail.trace.npz")
trace = {name: archive[name] for name in ("step_ids", "pre_q", "pre_qd", "post_q")}
builder = build_scene(
    body_count=36, fix_base=False, contact_gap=0.001, source_contact_offsets=True, mesh_cylinders=True
)
offsets = json.load(open("/tmp/colibri_physx_exact_contact_offsets_m.json"))
for i, label in enumerate(builder.shape_label):
    if label in offsets:
        builder.shape_gap[i] = offsets[label]
model = builder.finalize(skip_validation_joints=True)
state = model.state()
pair = [model.shape_label.index("TailRack/Tail_Rack"), model.shape_label.index("TailPinion/Tail_Pinion_Thick")]
shape_body = model.shape_body.numpy()
records = []
for name, reduction, cap in (
    ("canonical_velocity", True, 0.005),
    ("canonical_geometry", True, 0.005),
    ("canonical_geometry_raw", False, 0.005),
):
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        shape_pairs_filtered=wp.array([pair], dtype=wp.vec2i, device=model.device),
        reduce_contacts=reduction,
        rigid_contact_max=32768,
        contact_matching="disabled",
        speculative_contact_gap_max=cap,
        speculative_contact_velocity_filter=name == "canonical_velocity",
        deterministic=True,
    )
    contacts = pipeline.contacts()
    for step in (8147,):
        slot = int(np.flatnonzero(trace["step_ids"] == step)[0])
        pre, post = trace["pre_q"][slot], trace["post_q"][slot]
        state.body_q.assign(pre)
        state.body_qd.assign(trace["pre_qd"][slot])
        previous = None
        for repeat in range(2):
            pipeline.collide(state, contacts, dt=1 / 120)
            if name == "reduced_predictive" and step == 746 and repeat == 0:
                reducer = pipeline.narrow_phase.global_contact_reducer
                np.savez(
                    "/tmp/colibri_slab_tail_envelope_746_reducer.npz",
                    keys=reducer.hashtable.keys.numpy(),
                    values=reducer.ht_values.numpy(),
                    active=reducer.hashtable.active_slots.numpy(),
                    **{
                        field: getattr(reducer, field).numpy()
                        for field in (
                            "position_depth",
                            "normal",
                            "shape_pairs",
                            "contact_fingerprints",
                            "exported_flags",
                            "contact_count",
                        )
                    },
                    shape_linear_velocity=pipeline._shape_linear_velocity.numpy(),
                    shape_angular_velocity=pipeline._shape_angular_velocity.numpy(),
                )
            n = int(contacts.rigid_contact_count.numpy()[0])
            assert n < 32768
            shapes = np.column_stack(
                (contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n])
            )
            p0, p1 = contacts.rigid_contact_point0.numpy()[:n], contacts.rigid_contact_point1.numpy()[:n]
            normals = contacts.rigid_contact_normal.numpy()[:n]
            margins = np.column_stack(
                (contacts.rigid_contact_margin0.numpy()[:n], contacts.rigid_contact_margin1.numpy()[:n])
            )
            before = gaps(pre, shape_body, shapes, p0, p1, normals, margins)
            after = gaps(post, shape_body, shapes, p0, p1, normals, margins)
            signature = np.column_stack((p0, p1, normals))
            # Raw atomic insertion order is immaterial to coverage; sort each
            # complete witness before comparing identical frozen queries.
            signature = signature[np.lexsort(signature.T[::-1])]
            identical = previous is None or np.array_equal(signature, previous)
            previous = signature
            records.append(
                {
                    "step": step,
                    "variant": name,
                    "repeat": repeat,
                    "count": n,
                    "pre_min_gap_m": float(before.min()) if n else None,
                    "transported_post_min_gap_m": float(after.min()) if n else None,
                    "transported_penetrating_points": int(np.count_nonzero(after < -0.0001)),
                    "same_input_witness_set_identical": identical,
                }
            )
            if repeat == 0:
                np.savez(
                    f"/tmp/colibri_public8147_admission_{step}_{name}.npz",
                    point0=p0,
                    point1=p1,
                    normals=normals,
                    shapes=shapes,
                    pre_gap=before,
                    transported_post_gap=after,
                )
print(json.dumps(records, indent=2))
with open("/tmp/colibri_public8147_admission_coverage.json", "w") as stream:
    json.dump(records, stream, indent=2)

velocity = next(row for row in records if row["variant"] == "canonical_velocity")
geometric = next(row for row in records if row["variant"] == "canonical_geometry")
assert velocity["transported_penetrating_points"] == 0
assert geometric["transported_penetrating_points"] > 0
assert all(row["same_input_witness_set_identical"] for row in records)
