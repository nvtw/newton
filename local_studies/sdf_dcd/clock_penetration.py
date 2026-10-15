"""Which shape pairs penetrate in the analog clock under cached sampling?"""

import sys

import numpy as np

import newton

sys.path.insert(0, "local_studies/sdf_dcd")
from compare import Oracle, load_scene, sdf_method

frames = int(sys.argv[1]) if len(sys.argv) > 1 else 300
with sdf_method("cached_sampling"):
    scene = load_scene("make_analog_clock_scene")
    model = scene.model
    pipe = scene.pipeline_factory()
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipe, **scene.solver_options)
c = pipe.contacts()
ctrl = model.control()
st = model.state()
if scene.initialize_state:
    scene.initialize_state(st)
oracle = Oracle(model)
dt = scene.frame_dt / scene.collision_updates
labels = list(model.shape_label)
types = model.shape_type.numpy()
for f in range(frames):
    for u in range(scene.collision_updates):
        st.clear_forces()
        if scene.before_update:
            scene.before_update(st, ctrl, f * scene.collision_updates + u, dt)
        pipe.collide(st, c, dt=dt)
        solver.step(st, st, ctrl, c, dt)
oracle.penetration(st)
oc = oracle.contacts
n = int(oc.rigid_contact_count.numpy()[0])
sep = oracle.sep.numpy()[:n]
s0 = oc.rigid_contact_shape0.numpy()[:n]
s1 = oc.rigid_contact_shape1.numpy()[:n]
worst = {}
for i in np.argsort(sep)[:200]:
    k = (int(s0[i]), int(s1[i]))
    worst[k] = min(worst.get(k, 0.0), float(sep[i]))
cs = pipe.narrow_phase.sdf_cached_sampling
print("cached-sampling pairs", int(cs.counts.numpy()[0]), "edge-rest", int(cs.counts.numpy()[1]))
for k, v in sorted(worst.items(), key=lambda kv: kv[1])[:8]:
    print(f"pen {v * 1000:7.2f} mm  {labels[k[0]]} (type {types[k[0]]})  <->  {labels[k[1]]} (type {types[k[1]]})")
