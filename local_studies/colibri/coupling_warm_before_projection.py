from newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop import Example
from newton.viewer import ViewerNull
import time
e=Example(ViewerNull(),width=6,height=6)
e.graph=None
import warp as wp
from newton._src.solvers.phoenx.articulations.direct_contact_gs import warm_start_direct_contact_runs_kernel
original_launch=wp.launch_tiled
def filtered_launch(*args,**kwargs):
    kernel=kwargs.get("kernel",args[0] if args else None)
    if getattr(kernel,"key","")=="warm_start_direct_contact_runs_kernel": return
    return original_launch(*args,**kwargs)
wp.launch_tiled=filtered_launch
w=e.solver.world
d=e.solver._direct_equality_system
original_solve=d.solve
original_prepare=d.prepare_and_factor
warmed=[False]
def prepare(*args,**kwargs):
    warmed[0]=False
    return original_prepare(*args,**kwargs)
d.prepare_and_factor=prepare
assert w._combine_direct_prepare_projection
def early_solve(*,use_bias):
    if not warmed[0] and w._contact_input_active_this_step:
        warmed[0]=True
        response=w._direct_contact_response
        schedule=w._direct_contact_schedule
        response.compute(w._contact_container)
        original_launch(warm_start_direct_contact_runs_kernel,dim=response.active_mechanism.size,block_dim=64,inputs=[response.active_mechanism,response.data,w.bodies,w._contact_cols,w._contact_container,schedule.columns,schedule.section_end],device=w.device)
    return original_solve(use_bias=use_bias)
d.solve=early_solve

t=time.perf_counter()
for frame in range(120):
    e.step()
    if frame%10==0:
        print(frame,e.state.body_q.numpy()[e.cube_body,2],time.perf_counter()-t,flush=True)
e.test_final()
print("HEIGHT",e.state.body_q.numpy()[e.cube_body,2],flush=True)
