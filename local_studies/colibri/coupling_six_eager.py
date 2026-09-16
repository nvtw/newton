from newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop import Example
from newton.viewer import ViewerNull
import time
e=Example(ViewerNull(),width=6,height=6)
e.graph=None
t=time.perf_counter()
for frame in range(120):
    e.step()
    if frame%10==0:
        print(frame,e.state.body_q.numpy()[e.cube_body,2],time.perf_counter()-t,flush=True)
e.test_final()
print("HEIGHT",e.state.body_q.numpy()[e.cube_body,2],flush=True)
