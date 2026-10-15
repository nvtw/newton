"""Run a scene for warmup, then 100 collide calls for nsys."""

import sys

import warp as wp

import newton

sys.path.insert(0, "local_studies/sdf_dcd")
from compare import load_scene, sdf_method

name, method = sys.argv[1], sys.argv[2]
with sdf_method(method):
    scene = load_scene(name)
    model = scene.model
    pipe = (
        scene.pipeline_factory()
        if scene.pipeline_factory
        else newton.CollisionPipeline(model, contact_matching="sticky")
    )
    solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipe, **scene.solver_options)
c = pipe.contacts()
s0, s1 = model.state(), model.state()
ctrl = model.control()
for _f in range(240):
    pipe.collide(s0, c, dt=1 / 120)
    solver.step(s0, s1, ctrl, c, 1 / 60)
    s0, s1 = s1, s0
wp.synchronize()
with wp.ScopedCapture() as cap:
    pipe.collide(s0, c, dt=1 / 120)
for _ in range(10):
    wp.capture_launch(cap.graph)
wp.synchronize()
wp.config.verify_cuda = False
for _ in range(100):
    pipe.collide(s0, c, dt=1 / 120)
wp.synchronize()
