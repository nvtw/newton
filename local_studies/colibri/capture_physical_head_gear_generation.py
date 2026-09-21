"""Capture graph-executed collision poses and replay raw/reduced gear generation."""

import json
import runpy
from pathlib import Path

import numpy as np
import warp as wp

import newton
from local_studies.colibri.analyze_late_contacts import gaps
from local_studies.colibri.physical_head_overflow import install

install()
original = newton.solvers.SolverPhoenX.step
slots = []
owner = []


def step(self, state_in, *args, **kwargs):
    q = wp.empty_like(state_in.body_q)
    qd = wp.empty_like(state_in.body_qd)
    wp.copy(q, state_in.body_q)
    wp.copy(qd, state_in.body_qd)
    slots.append((q, qd))
    owner[:] = [self]
    return original(self, state_in, *args, **kwargs)


newton.solvers.SolverPhoenX.step = step
try:
    runpy.run_module("local_studies.colibri.total_normal_friction", run_name="__main__")
finally:
    data = np.load("/tmp/colibri_physical_head12_batch1024_generation.npz")
    ref = np.load("/tmp/colibri_physical_head12_batch1024.npz")
    gate = {
        k: data[k].dtype == ref[k].dtype and data[k].shape == ref[k].shape and data[k].tobytes() == ref[k].tobytes()
        for k in ref.files
    }
    assert all(gate.values()), gate
    assert len(slots) == 2, len(slots)
    pre = slots[-1][0].numpy()
    preqd = slots[-1][1].numpy()
    post = data["q"]
    np.savez(
        "/tmp/physical_head_gear_generation_pose.npz",
        pre_q=pre,
        pre_qd=preqd,
        first_half_q=slots[0][0].numpy(),
        post_q=post,
    )
    model = owner[0].model
    state = model.state()
    shape_body = model.shape_body.numpy()
    physical_gap = model.shape_gap.numpy().copy()
    records = []
    for reduction in (False, True):
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array([[48, 50]], dtype=wp.vec2i, device=model.device),
            reduce_contacts=reduction,
            rigid_contact_max=65536,
            contact_matching="disabled",
            speculative_contact_velocity_filter=False,
            speculative_contact_gap_max=0.005,
        )
        contacts = pipeline.contacts()
        for pose_name, q, qd in (("generation", pre, preqd), ("failed", post, data["qd"])):
            state.body_q.assign(q)
            state.body_qd.assign(qd)
            pipeline.collide(state, contacts, dt=1 / 120)
            n = int(contacts.rigid_contact_count.numpy()[0])
            assert n < 65536
            shapes = np.column_stack(
                [contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]]
            )
            p0 = contacts.rigid_contact_point0.numpy()[:n]
            p1 = contacts.rigid_contact_point1.numpy()[:n]
            normals = contacts.rigid_contact_normal.numpy()[:n]
            margins = np.column_stack(
                [contacts.rigid_contact_margin0.numpy()[:n], contacts.rigid_contact_margin1.numpy()[:n]]
            )
            local_gap = gaps(q, shape_body, shapes, p0, p1, normals, margins)
            post_gap = gaps(post, shape_body, shapes, p0, p1, normals, margins)
            label = pose_name + ("_reduced" if reduction else "_raw")
            np.savez(
                "/tmp/physical_head_gear_" + label + ".npz",
                shapes=shapes,
                point0=p0,
                point1=p1,
                normals=normals,
                margins=margins,
                query_gap=local_gap,
                transported_post_gap=post_gap,
            )
            records.append(
                dict(
                    variant=label,
                    count=n,
                    query_min_gap=float(local_gap.min()) if n else None,
                    post_min_gap=float(post_gap.min()) if n else None,
                    post_deeper_than_1mm=int(np.count_nonzero(post_gap < -0.001)),
                )
            )
    assert physical_gap.tobytes() == model.shape_gap.numpy().tobytes()
    report = dict(
        gate=gate,
        generation_time_s=0.14166666666666666,
        failed_time_s=0.15,
        records=records,
        physical_gaps_unchanged=True,
        scope="Exact captured last 120 Hz generation pose; queries preserve model and physical gaps. Transported gap is not a new SDF query.",
    )
    Path("/tmp/physical_head_gear_generation_audit.json").write_text(json.dumps(report, indent=2))
    print("GEAR_GENERATION_AUDIT", json.dumps(report), flush=True)
