"""Compare exact native midpoint generation against fresh reduced/raw coverage."""

import json
from pathlib import Path

import numpy as np

import newton
from local_studies.colibri.analyze_late_contacts import gaps
from newton.examples.kamino.example_kamino_colibri import build_scene
from newton.examples.phoenx.example_phoenx_colibri import CONTACT_OFFSETS


def main():
    """Re-query identical9291 generation input, preserving source material gaps."""
    trace = np.load("/tmp/colibri_velocity4_native_tail.trace.npz")
    builder = build_scene(
        body_count=36,
        fix_base=False,
        contact_gap=0.001,
        source_contact_offsets=True,
        mesh_cylinders=True,
        sdf_resolution=0,
    )
    for i, label in enumerate(builder.shape_label):
        if label in CONTACT_OFFSETS:
            builder.shape_gap[i] = CONTACT_OFFSETS[label]
    model = builder.finalize(skip_validation_joints=True)
    state = model.state()
    body = model.shape_body.numpy()
    records = []
    for variant, reduced, cap in (
        ("geometric_reduced", True, 0.005),
        ("geometric_raw", False, 0.005),
        ("ordinary_reduced", True, None),
    ):
        pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=65536,
            contact_matching="disabled",
            reduce_contacts=reduced,
            speculative_contact_gap_max=cap,
            speculative_contact_velocity_filter=cap is None,
        )
        contacts = pipeline.contacts()
        for step in (9290, 9291):
            slot = int(np.flatnonzero(trace["step_ids"] == step)[0])
            pre = trace["generation_q"][4 * slot]
            post = trace["post_q"][4 * slot + 3]
            state.body_q.assign(pre)
            state.body_qd.assign(trace["generation_qd"][4 * slot])
            pipeline.collide(state, contacts, dt=1 / 120)
            total = int(contacts.rigid_contact_count.numpy()[0])
            assert total < 65536
            shapes = np.column_stack(
                (contacts.rigid_contact_shape0.numpy()[:total], contacts.rigid_contact_shape1.numpy()[:total])
            )
            selected = np.flatnonzero(np.all(shapes == [36, 37], axis=1))
            shapes = shapes[selected]
            p0 = contacts.rigid_contact_point0.numpy()[selected]
            p1 = contacts.rigid_contact_point1.numpy()[selected]
            normals = contacts.rigid_contact_normal.numpy()[selected]
            margins = np.column_stack(
                (contacts.rigid_contact_margin0.numpy()[selected], contacts.rigid_contact_margin1.numpy()[selected])
            )
            before = gaps(pre, body, shapes, p0, p1, normals, margins)
            after = gaps(post, body, shapes, p0, p1, normals, margins)
            record = {
                "step": step,
                "variant": variant,
                "total": total,
                "pair_count": len(selected),
                "pre_min_gap": float(before.min()),
                "post_transported_min_gap": float(after.min()),
                "post_below_500um": int(np.count_nonzero(after < -0.0005)),
            }
            records.append(record)
            np.savez(
                f"/tmp/colibri_velocity4_priority_fixed_midpoint_{step}_{variant}.npz",
                shapes=shapes,
                point0=p0,
                point1=p1,
                normals=normals,
                margins=margins,
                pre_gap=before,
                post_gap=after,
            )
            print("MIDPOINT_COVERAGE", json.dumps(record), flush=True)
    Path("/tmp/colibri_velocity4_priority_fixed_midpoint_coverage.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
