"""Compare exact native midpoint generation against fresh reduced/raw coverage."""

import argparse
import json
from pathlib import Path

import numpy as np

import newton
from local_studies.colibri.analyze_late_contacts import gaps
from newton.examples.kamino.example_kamino_colibri import build_scene
from newton.examples.phoenx.example_phoenx_colibri import CONTACT_OFFSETS


def main():
    """Re-query identical9291 generation input, preserving source material gaps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--pair", default="46:48")
    parser.add_argument("--steps", default="29838,29839")
    parser.add_argument("--include-post", action="store_true")
    args = parser.parse_args()
    pair = list(map(int, args.pair.split(":")))
    steps = list(map(int, args.steps.split(",")))
    trace = np.load(args.prefix + ".trace.npz")
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
        queries = [(step, False) for step in steps]
        if args.include_post:
            queries += [(steps[-1], True)]
        for step, at_post in queries:
            slot = int(np.flatnonzero(trace["step_ids"] == step)[0])
            pre = trace["generation_q"][4 * slot]
            post = trace["post_q"][4 * slot + 3]
            if at_post:
                pre = post
            state.body_q.assign(pre)
            state.body_qd.assign(trace["post_qd"][4 * slot + 3] if at_post else trace["generation_qd"][4 * slot])
            query_variant = variant + ("_post" if at_post else "")
            pipeline.collide(state, contacts, dt=1 / 120)
            total = int(contacts.rigid_contact_count.numpy()[0])
            assert total < 65536
            shapes = np.column_stack(
                (contacts.rigid_contact_shape0.numpy()[:total], contacts.rigid_contact_shape1.numpy()[:total])
            )
            selected = np.flatnonzero(np.all(shapes == pair, axis=1))
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
                "variant": query_variant,
                "total": total,
                "pair_count": len(selected),
                "pre_min_gap": float(before.min()) if len(before) else None,
                "post_transported_min_gap": float(after.min()) if len(after) else None,
                "post_below_500um": int(np.count_nonzero(after < -0.0005)),
            }
            records.append(record)
            np.savez(
                f"{args.prefix}.query_{step}_{query_variant}.npz",
                shapes=shapes,
                point0=p0,
                point1=p1,
                normals=normals,
                margins=margins,
                pre_gap=before,
                post_gap=after,
            )
            print("MIDPOINT_COVERAGE", json.dumps(record), flush=True)
    Path(args.prefix + ".query.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
