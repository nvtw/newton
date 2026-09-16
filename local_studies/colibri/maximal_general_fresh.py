import newton
import argparse
import time
import numpy as np
from pathlib import Path
import warp as wp
from newton.viewer import ViewerNull
from local_studies.colibri.phoenx_scene import Example
import newton._src.solvers.phoenx.solver as solver_module
solver_module.find_full_coordinate_revolute_trees=lambda *args: []
p=argparse.ArgumentParser()
p.add_argument("--body-count",type=int,default=36)
p.add_argument("--mode",default="maximal")
p.add_argument("--substeps",type=int,default=8)
p.add_argument("--iterations",type=int,default=8)
p.add_argument("--outer-substeps",type=int,default=1)
p.add_argument("--frames",type=int,default=180)
p.add_argument("--layout",default="single_world")
p.add_argument("--stages",action="store_true")
p.add_argument("--start",type=int,default=1)
p.add_argument("--frictionless",action="store_true")
p.add_argument("--project-initial",action="store_true")
p.add_argument("--matching",choices=["disabled","latest","sticky"],default="sticky")
p.add_argument("--free-base",action="store_true")
p.add_argument("--fix-base",action="store_true")
p.add_argument("--contact-gap",type=float,default=0.0001)
p.add_argument("--source-contact-offsets",action="store_true")
p.add_argument("--source-damping",action="store_true")
p.add_argument("--mesh-cylinders",action="store_true")
a=p.parse_args()
if a.fix_base and a.free_base:
    p.error("Choose only one base mode")
import gc
for n in (range(a.start,a.body_count+1) if a.stages else [a.body_count]):
    a.body_count=n
    e=Example(ViewerNull(),a)
    assert e.solver._maximal_contact_response is None
    assert e.solver._direct_contact_response is not None
    fresh_pipeline=newton.CollisionPipeline(e.model,contact_matching="disabled",rigid_contact_max=8192)
    fresh=fresh_pipeline.contacts()
    shape_body=e.model.shape_body.numpy()
    worst_depth=0.0
    t=time.perf_counter()
    for frame in range(a.frames):
        e.step()
        if frame % 10 == 0:
            print("PROGRESS", n, frame, time.perf_counter()-t, flush=True)
        fresh_pipeline.collide(e.state_0,fresh)
        count=int(fresh.rigid_contact_count.numpy()[0])
        q=e.state_0.body_q.numpy()
        sides=[]
        for side in (0,1):
            shapes=getattr(fresh,f"rigid_contact_shape{side}").numpy()[:count]
            points=getattr(fresh,f"rigid_contact_point{side}").numpy()[:count]
            sides.append(np.array([np.array(wp.transform_point(wp.transform(q[shape_body[s],:3],q[shape_body[s],3:]),wp.vec3(pt))) if shape_body[s]>=0 else pt for s,pt in zip(shapes,points)]))
        gaps=np.einsum("ij,ij->i",sides[1]-sides[0],fresh.rigid_contact_normal.numpy()[:count])
        worst_depth=max(worst_depth,float(-gaps.min()))
        if frame%10==0 or gaps.min()<-.001:
            k=int(gaps.argmin())
            print("FRESH_DEPTH",frame,float(-gaps[k]),"maximum",worst_depth,"shapes",*[e.model.shape_label[getattr(fresh,f"rigid_contact_shape{side}").numpy()[k]] for side in (0,1)],flush=True)
        if gaps.min()<-.001: break
        try: e.test_post_step()
        except AssertionError:
            q = e.state_0.body_q.numpy()
            qd = e.state_0.body_qd.numpy()
            speeds = np.linalg.norm(qd, axis=1)
            print("FAILED BODY",a.body_count,"FRAME",frame,"SECONDS",time.perf_counter()-t,flush=True)
            print("FASTEST",[(e.model.body_label[i],float(speeds[i])) for i in np.argsort(speeds)[-5:]],flush=True)
            target = Path(f"/tmp/colibri_phoenx_failure_{a.mode}_{a.body_count}_{a.substeps}_{a.outer_substeps}_{a.iterations}.npz")
            np.savez(target,q=q,qd=qd,labels=e.model.body_label,contact_derived=e.solver.world._contact_container.derived.numpy(),contact_impulses=e.solver.world._contact_container.impulses.numpy(),args=str(vars(a)),frame=frame)
            print("SAVED",target,flush=True)
            raise
        if (frame+1) % 60 == 0:
            print("PROGRESS", n, frame+1, "seconds", time.perf_counter()-t, flush=True)
    wp.synchronize()
    elapsed=time.perf_counter()-t
    print("FAIL_DEPTH" if worst_depth>.001 else "PASS",vars(a),"seconds",elapsed,"fps_with_checks",a.frames/elapsed,flush=True)
    del e
    gc.collect()
