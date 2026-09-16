"""Query fresh collision witnesses around saved velocity4 tail failure."""

import json
from pathlib import Path

import numpy as np

import newton
from local_studies.colibri.analyze_late_contacts import gaps
from newton.examples.kamino.example_kamino_colibri import build_scene
from newton.examples.phoenx.example_phoenx_colibri import CONTACT_OFFSETS


def main():
    """Preserve public geometry/filtering and distinguish fresh from transported gaps."""
    archive = np.load("/tmp/colibri_native_velocity4_clean18000.npz")
    q = np.concatenate((archive["q_history"][-4:], archive["q"][None]))
    qd = np.concatenate((archive["qd_history"][-4:], archive["qd"][None]))
    times = np.r_[archive["history_times"][-4:], 77.45]
    builder = build_scene(
        body_count=36,
        fix_base=False,
        contact_gap=0.001,
        source_contact_offsets=True,
        mesh_cylinders=True,
        sdf_resolution=0,
    )
    for index, label in enumerate(builder.shape_label):
        if label in CONTACT_OFFSETS:
            builder.shape_gap[index] = CONTACT_OFFSETS[label]
    model = builder.finalize(skip_validation_joints=True)
    state = model.state()
    shape_body = model.shape_body.numpy()
    records = []
    for variant in ("fresh_audit", "geometric", "velocity_filtered"):
        kwargs = {}
        if variant != "fresh_audit":
            kwargs.update(
                speculative_contact_gap_max=0.005, speculative_contact_velocity_filter=variant == "velocity_filtered"
            )
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32768, contact_matching="disabled", **kwargs)
        contacts = pipeline.contacts()
        for index, pose in enumerate(q):
            state.body_q.assign(pose)
            state.body_qd.assign(qd[index])
            pipeline.collide(state, contacts, dt=1 / 120)
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
            gap = gaps(pose, shape_body, shapes, p0, p1, normals, margins)
            after = gaps(q[min(index + 1, len(q) - 1)], shape_body, shapes, p0, p1, normals, margins)
            pairrows = []
            for pair in np.unique(shapes, axis=0):
                names = [model.shape_label[int(x)] for x in pair]
                if not any("Tail" in name for name in names):
                    continue
                mask = np.all(shapes == pair, axis=1)
                pairrows.append(
                    {
                        "shapes": names,
                        "shape_ids": pair.tolist(),
                        "count": int(mask.sum()),
                        "fresh_min_gap_m": float(gap[mask].min()),
                        "transported_next60Hz_min_gap_m": float(after[mask].min()),
                    }
                )
            record = {
                "time": float(times[index]),
                "variant": variant,
                "contact_count": n,
                "tail_pairs": sorted(pairrows, key=lambda x: x["fresh_min_gap_m"]),
            }
            records.append(record)
            np.savez(
                f"/tmp/colibri_velocity4_geometry_{variant}_{index}.npz",
                shapes=shapes,
                point0=p0,
                point1=p1,
                normals=normals,
                margins=margins,
                gap=gap,
                transported_next_gap=after,
            )
            print("TAIL_GEOMETRY", json.dumps(record), flush=True)
    Path("/tmp/colibri_velocity4_tail_geometry.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
