# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-number performance metric for Box3D solver optimization.

Outputs the total step time (ms) for:
  - 1000 worlds × 10-box stacks (contact-heavy)
  - 1000 worlds × 20-link revolute chains (joint-heavy)
Sum of both = the optimization metric. Lower is better.

Usage: uv run python3 newton/tests/bench_box3d_metric.py
Output: single float (total ms)
"""

import time
import warp as wp
import newton
from newton._src.solvers.box3d.config import Box3DConfig

wp.init()
device = "cuda:0"


def build_box_stack():
    b = newton.ModelBuilder()
    b.add_ground_plane()
    h = 0.5
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    for i in range(10):
        shift = -0.01 if i % 2 == 0 else 0.01
        body = b.add_body(xform=wp.transform(wp.vec3(shift, 0.0, h + 2.0 * h * i)))
        b.add_shape_box(body=body, hx=h, hy=h, hz=h, cfg=cfg)
    return b


def build_chain():
    b = newton.ModelBuilder()
    anchor = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 20.0)), is_kinematic=True)
    b.add_shape_sphere(body=anchor, radius=0.05)
    cfg = newton.ModelBuilder.ShapeConfig(density=10.0, mu=0.5)
    prev = anchor
    for i in range(20):
        link = b.add_body(xform=wp.transform(wp.vec3(0.5 + 1.0 * i, 0.0, 20.0)))
        b.add_shape_capsule(body=link, radius=0.1, half_height=0.4, cfg=cfg)
        p_xf = wp.transform_identity() if prev == anchor else wp.transform(wp.vec3(0.5, 0.0, 0.0))
        b.add_joint_revolute(parent=prev, child=link,
                             parent_xform=p_xf,
                             child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)),
                             axis=newton.Axis.Y)
        prev = link
    return b


def measure(scene_fn, num_worlds, num_steps=50, warmup=10):
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    sub = scene_fn()
    for _ in range(num_worlds):
        scene.add_world(sub)
    model = scene.finalize(device=device)
    s_in = model.state()
    s_out = model.state()
    cfg = Box3DConfig(num_substeps=1, num_velocity_iters=2, num_relaxation_iters=2,
                      contact_hertz=30.0, joint_hertz=60.0,
                      linear_damping=0.1, angular_damping=0.1,
                      enable_graph=True,
                      max_bodies_per_world=128,
                      max_joints_per_world=128,
                      max_contacts_per_world=1024,
                      max_colors=32)
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=False)
    contacts = pipeline.contacts()
    dt = 1.0 / 60.0
    for _ in range(warmup):
        contacts.clear()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, dt)
        s_in, s_out = s_out, s_in
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        contacts.clear()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, dt)
        s_in, s_out = s_out, s_in
    wp.synchronize_device(device)
    return (time.perf_counter() - t0) / num_steps * 1000.0


contact_ms = measure(build_box_stack, 1000)
joint_ms = measure(build_chain, 1000)
total = contact_ms + joint_ms

# Output single number — the optimization metric
print(f"{total:.3f}")
