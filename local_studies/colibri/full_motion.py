"""Check full Colibri motion and save checkpoints for visual inspection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import warp as wp
import newton
from PIL import Image
from newton.examples.kamino.example_kamino_colibri import Example, JOINTS
from newton.viewer import ViewerGL

wp.init()
viewer = ViewerGL(width=640, height=640, headless=True)
e = Example(viewer, argparse.Namespace(body_count=36))
poses = []
velocities = []
counts = []
anchor_errors = []
axis_errors = []
body_ids = {name: i for i,name in enumerate(e.model.body_label)}
parents = e.model.joint_parent.numpy()
children = e.model.joint_child.numpy()
frames_a = e.model.joint_X_p.numpy()
frames_b = e.model.joint_X_c.numpy()
types = e.model.joint_type.numpy()
start = time.perf_counter()
for frame in range(601):
    if frame:
        e.step()
        e.test_post_step()
    poses.append(e.state_0.body_q.numpy())
    velocities.append(e.state_0.body_qd.numpy())
    q = poses[-1]
    errors = []
    for j, (a, b) in enumerate(zip(parents, children, strict=True)):
        if types[j] == newton.JointType.FREE:
            continue
        pa = wp.transform_identity() if a < 0 else wp.transform(q[a,:3],q[a,3:])
        pb = wp.transform(q[b,:3],q[b,3:])
        x = wp.transform_point(pa,wp.vec3(frames_a[j,:3]))
        y = wp.transform_point(pb,wp.vec3(frames_b[j,:3]))
        errors.append(float(np.linalg.norm(np.asarray(x)-np.asarray(y))))
    anchor_errors.append(max(errors,default=0.0))
    axes = []
    for a,b,kind,axis,fa,fb,drive in JOINTS:
        if kind != "revolute":
            continue
        v = wp.vec3(np.eye(3)["XYZ".index(axis)])
        qa = wp.mul(wp.quat(q[body_ids[a],3:]),wp.quat(fa[3:]))
        qb = wp.mul(wp.quat(q[body_ids[b],3:]),wp.quat(fb[3:]))
        axes.append(float(np.linalg.norm(np.asarray(wp.quat_rotate(qa,v))-np.asarray(wp.quat_rotate(qb,v)))))
    axis_errors.append(max(axes,default=0.0))
    assert axis_errors[-1] < 0.03, f"Revolute axes misaligned at frame {frame}: {axis_errors[-1]}"
    counts.append(int(e.solver._contacts_kamino.model_active_contacts.numpy()[0]))
    assert counts[-1] < e.contact_capacity
    if frame in (0, 60, 180, 300, 600):
        e.render()
        Image.fromarray(viewer.get_frame().numpy()).save(f"/tmp/colibri_motion_{frame}.png")
        i = body_ids["Crank"]
        j = body_ids["Frame"]
        axis = np.asarray(wp.quat_rotate(wp.quat(q[j,3:]),wp.vec3(0,0,1)))
        crank_speed = float(np.dot(velocities[-1][i,3:]-velocities[-1][j,3:],axis))
        rack_distance = float(np.linalg.norm(q[body_ids["TailRack"],:3]-q[body_ids["TailMount"],:3]))
        np.savez("/tmp/colibri_motion.npz", poses=poses, velocities=velocities, contacts=counts,
                 anchor_errors=anchor_errors, axis_errors=axis_errors, labels=e.model.body_label)
        print("CHECKPOINT", frame, "contacts", counts[-1], "max_anchor_error", max(anchor_errors),
              "crank_deg_s", np.degrees(crank_speed), "rack_distance", rack_distance, flush=True)
np.savez("/tmp/colibri_motion.npz", poses=poses, velocities=velocities, contacts=counts, anchor_errors=anchor_errors, axis_errors=axis_errors, labels=e.model.body_label)
print("PASS full 10-second simulation", time.perf_counter()-start, flush=True)
viewer.close()
