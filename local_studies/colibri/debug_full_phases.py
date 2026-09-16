import argparse
import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from local_studies.colibri.phoenx_scene import Example
wp.config.quiet=True
e=Example(ViewerNull(),argparse.Namespace(body_count=36,source_contact_offsets=True,contact_gap=.001,mesh_cylinders=True,outer_substeps=2,mode="maximal",layout="single_world",substeps=10,iterations=8))
e.graph=None
e.frame_dt=1/60
w=e.solver.world
d=e.solver._direct_equality_system
rj=np.asarray(d.topology.row_joint); rl=d.row_local.numpy(); js=d.joint_to_structural.numpy()
pa=e.model.joint_parent.numpy()+1; ch=e.model.joint_child.numpy()+1
mask=d.row_dynamic.numpy()==0

def report(label):
    v=np.concatenate((w.bodies.velocity.numpy(),w.bodies.angular_velocity.numpy()),axis=1)
    j0=d.row_wrench0.numpy()[js[rj],rl]; j1=d.row_wrench1.numpy()[js[rj],rl]
    residual=np.sum(j0*v[pa[rj]],axis=1)+np.sum(j1*v[ch[rj]],axis=1)
    bias=d.row_bias.numpy()[js[rj],rl]
    print(label,"speed",np.max(abs(v)),"Jv",np.max(abs(residual[mask])),"Jv+b",np.max(abs((residual+bias)[mask])),flush=True)
old_contact=w._solve_direct_contacts
old_equal=d.solve
def contacts(**kwargs):
    report("before-contact"+str(kwargs))
    old_contact(**kwargs)
    report("after-contact")
def equal(**kwargs):
    report("before-equal"+str(kwargs))
    old_equal(**kwargs)
    report("after-equal")
w._solve_direct_contacts=contacts
d.solve=equal
for frame in range(5):
    print("FRAME",frame,flush=True)
    e.step()
