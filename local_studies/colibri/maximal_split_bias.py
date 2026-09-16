import argparse
import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from local_studies.colibri.phoenx_scene import Example
wp.config.quiet=True
e=Example(ViewerNull(),argparse.Namespace(body_count=15,mode="maximal",layout="single_world",substeps=1,iterations=8))
e.graph=None
e.frame_dt=1/1200
w=e.solver.world
d=e.solver._direct_equality_system
rj=np.asarray(d.topology.row_joint); rl=d.row_local.numpy(); js=d.joint_to_structural.numpy()
pa=e.model.joint_parent.numpy()+1; ch=e.model.joint_child.numpy()+1
mask=d.row_dynamic.numpy()==0

def report(label):
    v=np.concatenate((w.bodies.velocity.numpy(),w.bodies.angular_velocity.numpy()),axis=1)
    n=int(e.contacts.rigid_contact_count.numpy()[0]); cc=w._contact_container; impulse=cc.impulses.numpy()[:,:n]; mobi=w._maximal_contact_schedule.mobility.numpy()[:,:n]
    if label.startswith("after-contact"):
        order=np.argsort(np.linalg.norm(impulse,axis=0))[-3:]
        print(label,"speed",np.max(abs(v)),"impulses",[(int(k),impulse[:,k].tolist(),mobi[:,k].tolist(),float(cc.derived.numpy()[15,k])) for k in order],flush=True)
old_contact=w._solve_maximal_articulated_contacts
old_equal=d.solve
old_integrate=w._integrate_positions
def contacts(**kwargs):
    report("before-contact"+str(kwargs))
    if kwargs["use_bias"]:
        from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
        nb=len(w.bodies.inverse_mass.numpy()); j=np.zeros((sum(mask),6*nb))
        rows=np.flatnonzero(mask)
        j0=d.row_wrench0.numpy()[js[rj],rl]; j1=d.row_wrench1.numpy()[js[rj],rl]
        for out,row in enumerate(rows):
            j[out,6*pa[rj[row]]:6*pa[rj[row]]+6]+=j0[row]
            j[out,6*ch[rj[row]]:6*ch[rj[row]]+6]+=j1[row]
        invm=w.bodies.inverse_mass.numpy(); invi=inertia_sym6_unpack_np(w.bodies.inverse_inertia_world.numpy())
        m=np.zeros((nb*6,nb*6))
        for b in range(1,nb):
            if invm[b]>0:
                m[6*b:6*b+3,6*b:6*b+3]=np.eye(3)*invm[b]
                m[6*b+3:6*b+6,6*b+3:6*b+6]=invi[b]
        bias=d.row_bias.numpy()[js[rj],rl][mask]
        correction=(m@j.T@np.linalg.lstsq(j@m@j.T,-bias,rcond=1e-12)[0]).reshape(nb,6)
        w.bodies.velocity.assign(w.bodies.velocity.numpy()-correction[:,:3])
        w.bodies.angular_velocity.assign(w.bodies.angular_velocity.numpy()-correction[:,3:])
    old_contact(**kwargs)
    if kwargs["use_bias"]:
        w.bodies.velocity.assign(w.bodies.velocity.numpy()+correction[:,:3])
        w.bodies.angular_velocity.assign(w.bodies.angular_velocity.numpy()+correction[:,3:])
    report("after-contact")
def equal(**kwargs):
    old_equal(**kwargs)
    if kwargs["use_bias"]: report("after-biased-equal")
w._solve_maximal_articulated_contacts=contacts
d.solve=equal
def integrate():
    report("before-integrate")
    old_integrate()
    report("after-integrate")
w._integrate_positions=integrate
for frame in range(20):
    print("FRAME",frame,flush=True)
    e.step()
