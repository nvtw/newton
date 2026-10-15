"""Trace the nut-on-bolt scene: nut pose, contact normals and separations per frame."""

import sys

import numpy as np

import newton
from newton.geometry import SDFCachedSampling

sys.path.insert(0, "local_studies/sdf_dcd")
import scenes
from compare import sdf_method

method = sys.argv[1]
frames = int(sys.argv[2])
cfg = SDFCachedSampling.Config()
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    cur = getattr(cfg, k)
    setattr(cfg, k, v if isinstance(cur, str) else type(cur)(v))
with sdf_method(method, cfg):
    scene = scenes.make_nut_bolt()
    model = scene.model
    pipe = newton.CollisionPipeline(model, contact_matching="sticky")
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipe, **scene.solver_options)
c = pipe.contacts()
s0, s1 = model.state(), model.state()
ctrl = model.control()
for f in range(frames):
    pipe.collide(s0, c)
    n = int(c.rigid_contact_count.numpy()[0])
    nr = c.rigid_contact_normal.numpy()[:n]
    q = s0.body_q.numpy()[0]
    # tilt of the nut axis from vertical
    x, y, z, w = q[3:]
    axis_z = 1 - 2 * (x * x + y * y)
    tilt = np.degrees(np.arccos(np.clip(axis_z, -1, 1)))
    nz = nr[:, 2] if n else np.zeros(0)
    print(
        f"f{f:3d} z={q[2] * 1000:7.2f}mm tilt={tilt:5.2f}deg n={n:3d} "
        f"nz>0.5:{int((nz > 0.5).sum()):3d} nz<-0.5:{int((nz < -0.5).sum()):3d} radial:{int((abs(nz) <= 0.5).sum()):3d} "
        f"wz={s0.body_qd.numpy()[0][5]:6.2f}"
    )
    solver.step(s0, s1, ctrl, c, 1 / 60)
    s0, s1 = s1, s0
