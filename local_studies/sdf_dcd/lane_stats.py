import sys

import newton
from newton.geometry import SDFCachedSampling

sys.path.insert(0, "local_studies/sdf_dcd")
from compare import load_scene, sdf_method

name = sys.argv[1]
extra = dict(a.split("=") for a in sys.argv[2:])
cfg = SDFCachedSampling.Config()
for k, v in extra.items():
    cur = getattr(cfg, k)
    setattr(cfg, k, v if isinstance(cur, str) else type(cur)(v))
with sdf_method("cached_sampling", cfg):
    scene = load_scene(name)
    model = scene.model
    pipe = (
        scene.pipeline_factory()
        if scene.pipeline_factory
        else newton.CollisionPipeline(model, contact_matching="sticky")
    )
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipe, **scene.solver_options)
cs = pipe.narrow_phase.sdf_cached_sampling
c = pipe.contacts()
s0, s1 = model.state(), model.state()
ctrl = model.control()
for _f in range(240):
    cs.iteration_count.zero_()
    pipe.collide(s0, c, dt=1 / 120)
    solver.step(s0, s1, ctrl, c, 1 / 60)
    s0, s1 = s1, s0
pairs = int(cs.counts.numpy()[0])
fl = cs.cand_flags.numpy()[: pairs * cs.lanes]
lanes = int((fl & 1).sum())
print(
    "pairs",
    pairs,
    "valid lanes",
    lanes,
    "iters total",
    int(cs.iteration_count.numpy()[0]),
    "per lane",
    cs.iteration_count.numpy()[0] / max(lanes, 1),
)
act = (fl & 2) != 0
fro = (fl & 8) != 0
print("cap", cs.max_pairs, "lanes", cs.lanes)
print("active lanes", int(act.sum()), "frozen", int(fro.sum()))
print("cached lanes", int(((fl & 4) != 0).sum()), "seeds per sdf", cs.seed_range.numpy()[:3])
