import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop import Example
from newton._src.solvers.phoenx.articulations.direct_contact_gs import warm_start_direct_contact_runs_kernel
wp.config.quiet=True
e=Example(ViewerNull(),width=4,height=4)
e.graph=None
w=e.solver.world
cube=e.cube_body+1
old=wp.launch_tiled
counter=0
frame=0
def launch(kernel,*args,**kwargs):
 global counter
 if kernel is warm_start_direct_contact_runs_kernel:
  before=w.bodies.velocity.numpy()[cube].copy()
  old(kernel,*args,**kwargs)
  after=w.bodies.velocity.numpy()[cube].copy()
  if np.linalg.norm(after-before)>1e-6 and counter<15:
   print(frame,"warm",before,after,"owner",e.solver._direct_contact_response.data.body_mechanism.numpy()[cube],flush=True)
   counter+=1
 else:
  old(kernel,*args,**kwargs)
wp.launch_tiled=launch
for frame in range(90):
 e.step()
 if frame%10==0: print("frame",frame,"z",e.state.body_q.numpy()[e.cube_body,2],flush=True)
