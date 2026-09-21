import json
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.phoenx.example_phoenx_colibri import _contact_separation


def audit(example, args):
    data = np.load("/tmp/colibri_public_group4_grid64_18000.npz")
    pipe = newton.CollisionPipeline(example.model, rigid_contact_max=16384, contact_matching="disabled")
    contacts = pipe.contacts()
    out = wp.zeros(16384, dtype=float, device=example.model.device)
    records = []
    poses = list(data["q_history"][-8:]) + [data["q"]]
    times = list(data["history_times"][-8:]) + [67.9166666666641]
    for q, t in zip(poses, times):
        example.state_0.body_q.assign(q)
        pipe.collide(example.state_0, contacts)
        n = int(contacts.rigid_contact_count.numpy()[0])
        wp.launch(
            _contact_separation,
            n,
            inputs=[
                example.state_0.body_q,
                example.model.shape_body,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
            ],
            outputs=[out],
            device=example.model.device,
        )
        gaps = out.numpy()[:n]
        a = contacts.rigid_contact_shape0.numpy()[:n]
        b = contacts.rigid_contact_shape1.numpy()[:n]
        p0 = contacts.rigid_contact_point0.numpy()[:n]
        p1 = contacts.rigid_contact_point1.numpy()[:n]
        norm = contacts.rigid_contact_normal.numpy()[:n]
        order = np.argsort(gaps)
        rows = []
        for k in order:
            labels = [example.model.shape_label[a[k]], example.model.shape_label[b[k]]]
            if len(rows) < 15 or any("Tail" in x or "Cylinder_33" in x for x in labels):
                rows.append(
                    dict(
                        shape_ids=[int(a[k]), int(b[k])],
                        labels=labels,
                        gap=float(gaps[k]),
                        point0=p0[k].tolist(),
                        point1=p1[k].tolist(),
                        normal=norm[k].tolist(),
                    )
                )
        records.append(dict(time=float(t), count=n, worst_gap=float(gaps.min()), contacts=rows))
        print(t, n, float(gaps.min()), rows[0]["labels"], flush=True)
    Path("/tmp/colibri_public_group4_failed_pose_contacts.json").write_text(json.dumps(records, indent=2))


newton.examples.run = audit
sys.argv = ["phoenx_colibri", "--viewer", "null", "--num-frames", "1"]
runpy.run_module("newton.examples.phoenx.example_phoenx_colibri", run_name="__main__")
