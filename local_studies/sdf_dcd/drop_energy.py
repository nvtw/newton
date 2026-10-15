import os
import sys

import numpy as np

import newton
from newton.geometry import SDFCachedSampling

sys.path.insert(0, "local_studies/sdf_dcd")
import scenes
from compare import sdf_method

method = sys.argv[1]
scene_name = os.environ.get("SCENE", "make_bunny_drops")
only = None if sys.argv[2] == "all" else int(sys.argv[2])
frames = int(sys.argv[3])
extra = dict(a.split("=") for a in sys.argv[4:])
cfg = SDFCachedSampling.Config()
for k, v in extra.items():
    cur = getattr(cfg, k)
    setattr(cfg, k, v if isinstance(cur, str) else type(cur)(v))
with sdf_method(method, cfg):
    scene = getattr(scenes, scene_name)() if only is None else scenes.make_bunny_drops(only=only)
    model = scene.model
    pipe = newton.CollisionPipeline(model, contact_matching="sticky")
    import os

    if os.environ.get("SOLVER") == "xpbd":
        solver = newton.solvers.SolverXPBD(model, iterations=10)
    else:
        solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipe, **scene.solver_options)
c = pipe.contacts()
s0, s1 = model.state(), model.state()
ctrl = model.control()
ke = scenes._kinetic_energy(model)
trace = []
import os  # noqa: E402

sub = 8 if os.environ.get("SOLVER") == "xpbd" else 1
for f in range(frames):
    for _ in range(sub):
        s0.clear_forces()
        pipe.collide(s0, c)
        solver.step(s0, s1, ctrl, c, 1 / 60 / sub)
        s0, s1 = s1, s0
    if f % 30 == 29:
        trace.append(ke(s0)["kinetic_energy_j"])
w = np.linalg.norm(s0.body_qd.numpy()[:, 3:], axis=1)
print(
    method,
    only,
    extra,
    "KE:",
    " ".join(f"{x:.0e}" for x in trace),
    "| final max w",
    np.round(w.max(), 2),
    "moving(w>0.05)",
    int((w > 0.05).sum()),
)
