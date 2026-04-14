# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Performance metric: 2000 worlds × 20-row pyramid, 2 substeps, 6+3 iters."""

import time
import warp as wp
import newton
from newton._src.solvers.box3d.config import Box3DConfig

wp.init()
device = "cuda:0"


def build_pyramid():
    b = newton.ModelBuilder()
    b.add_ground_plane()
    h = 0.5
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    for i in range(20):
        z = (2.0 * i + 1.0) * h
        for j in range(i, 20):
            x = (i + 1.0) * h + 2.0 * (j - i) * h - h * 20
            body = b.add_body(xform=wp.transform(wp.vec3(x, 0.0, z)))
            b.add_shape_box(body=body, hx=h, hy=h, hz=h, cfg=cfg)
    return b


scene = newton.ModelBuilder()
scene.add_ground_plane()
sub = build_pyramid()
for _ in range(2000):
    scene.add_world(sub)
model = scene.finalize(device=device)
s_in = model.state()
s_out = model.state()
cfg = Box3DConfig(
    num_substeps=2, num_velocity_iters=6, num_relaxation_iters=3,
    contact_hertz=30.0, linear_damping=0.1, angular_damping=0.1,
    enable_graph=True,
    max_bodies_per_world=256, max_contacts_per_world=4096, max_colors=32, block_dim=96,
)
solver = newton.solvers.SolverBox3D(model, config=cfg)
pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=False)
contacts = pipeline.contacts()
dt = 1.0 / 60.0
for _ in range(20):
    contacts.clear(); pipeline.collide(s_in, contacts)
    solver.step(s_in, s_out, None, contacts, dt)
    s_in, s_out = s_out, s_in
wp.synchronize_device(device)
N = 50
t0 = time.perf_counter()
for _ in range(N):
    contacts.clear(); pipeline.collide(s_in, contacts)
    solver.step(s_in, s_out, None, contacts, dt)
    s_in, s_out = s_out, s_in
wp.synchronize_device(device)
print(f"{(time.perf_counter() - t0) / N * 1000:.3f}")
