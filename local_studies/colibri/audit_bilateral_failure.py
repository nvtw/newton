"""Inspect physical joint/contact residuals at a saved Colibri failure pose."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
import warp as wp
from local_studies.colibri.phoenx_scene import Example
from local_studies.colibri.validate_phoenx import ContactAudit
from newton.viewer import ViewerNull

p=argparse.ArgumentParser()
p.add_argument("snapshot")
a=p.parse_args()
saved=np.load(a.snapshot)
cfg=json.loads(str(saved["config"]))
cfg.update(no_graph=True,mode="maximal",layout="single_world")
e=Example(ViewerNull(),SimpleNamespace(**cfg))
assert list(saved["labels"])==list(e.model.body_label)
e.state_0.body_q.assign(saved["q"])
e.state_0.body_qd.assign(saved["qd"])
s=e.solver
w=s.world
s._import_body_state(e.state_0)
w._refresh_world_inertia()
d=s._direct_equality_system
d.refresh_geometry(wp.float32(120*cfg["substeps"]))
q=saved["q"]
vel=w.bodies.velocity.numpy()
omega=w.bodies.angular_velocity.numpy()
pos=w.bodies.position.numpy()
twists=np.concatenate((vel,omega),axis=1).astype(np.float64)
rjoint=d.row_joint.numpy()
local=d.row_local.numpy()
struct=d.joint_to_structural.numpy()[rjoint]
parent=e.model.joint_parent.numpy()[rjoint]+1
child=e.model.joint_child.numpy()[rjoint]+1
j0=d.row_wrench0.numpy()[struct,local].astype(np.float64)
j1=d.row_wrench1.numpy()[struct,local].astype(np.float64)
jv=np.einsum("ij,ij->i",j0,twists[parent])+np.einsum("ij,ij->i",j1,twists[child])
bias=d.row_bias.numpy()[struct,local]
error=d.row_error.numpy()[struct,local]
dynamic=d.row_dynamic.numpy()
order=np.argsort(np.abs(jv)*(~dynamic))[-12:][::-1]
print("JOINT_ROWS",json.dumps([dict(joint=e.model.joint_label[rjoint[i]],local=int(local[i]),velocity=float(jv[i]),error=float(error[i]),bias=float(bias[i])) for i in order],indent=2),flush=True)
audit=ContactAudit(e.model)
print("FRESH",audit.check(e.state_0),flush=True)
c=audit.contacts
count=int(c.rigid_contact_count.numpy()[0])
normal=c.rigid_contact_normal.numpy()[:count]
shape0=c.rigid_contact_shape0.numpy()[:count]
shape1=c.rigid_contact_shape1.numpy()[:count]
shape_body=e.model.shape_body.numpy()
b0=shape_body[shape0]+1
b1=shape_body[shape1]+1
points=[]
for side in (0,1):
    shape=shape0 if side==0 else shape1
    points_local=getattr(c,f"rigid_contact_point{side}").numpy()[:count]
    pts=[]
    for sh,point in zip(shape,points_local):
        body=shape_body[sh]
        pts.append(np.array(wp.transform_point(wp.transform(q[body,:3],q[body,3:]),wp.vec3(point))) if body>=0 else point)
    points.append(np.asarray(pts))
midpoint=.5*(points[0]+points[1])
v0=vel[b0]+np.cross(omega[b0],midpoint-pos[b0])
v1=vel[b1]+np.cross(omega[b1],midpoint-pos[b1])
vn=np.einsum("ij,ij->i",v1-v0,normal)
gaps=audit.separation.numpy()[:count]
def row(i):
    return dict(shape0=e.model.shape_label[shape0[i]],shape1=e.model.shape_label[shape1[i]],gap=float(gaps[i]),velocity=float(vn[i]),normal=normal[i].tolist())
print("DEEPEST",json.dumps([row(i) for i in np.argsort(gaps)[:12]],indent=2),flush=True)
near=np.where(gaps<.0001)[0]
print("CLOSING",json.dumps([row(i) for i in near[np.argsort(vn[near])[:12]]],indent=2),flush=True)
np.savez(a.snapshot.replace(".npz","_physical_residual.npz"),joint_velocity=jv,joint_error=error,joint_bias=bias,joint_index=rjoint,local=local,contact_velocity=vn,contact_gap=gaps,contact_shape0=shape0,contact_shape1=shape1,normal=normal)
