import argparse
import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from local_studies.colibri.phoenx_scene import Example
wp.config.quiet=True
e=Example(ViewerNull(),argparse.Namespace(body_count=15,mode="maximal",layout="single_world",substeps=1,iterations=8))
e.graph=None
e.frame_dt=1/1200
w=e.solver.world
d=e.solver._direct_equality_system
response=e.solver._direct_contact_response
r=response.data
rj=np.asarray(d.topology.row_joint); rl=d.row_local.numpy(); js=d.joint_to_structural.numpy()
pa=e.model.joint_parent.numpy()+1; ch=e.model.joint_child.numpy()+1
old=w._solve_direct_contacts
calls=0
def hook(**kwargs):
    global calls
    calls+=1
    if calls%8==1:
        cc=w._contact_container
        response.compute(cc)
        n=int(e.contacts.rigid_contact_count.numpy()[0])
        nb=len(w.bodies.inverse_mass.numpy())
        j=np.zeros((len(rj),6*nb))
        j0=d.row_wrench0.numpy()[js[rj],rl]; j1=d.row_wrench1.numpy()[js[rj],rl]
        for row in range(len(rj)):
            j[row,6*pa[rj[row]]:6*pa[rj[row]]+6]+=j0[row]
            j[row,6*ch[rj[row]]:6*ch[rj[row]]+6]+=j1[row]
        invm=w.bodies.inverse_mass.numpy(); invi=inertia_sym6_unpack_np(w.bodies.inverse_inertia_world.numpy())
        m=np.zeros((nb*6,nb*6))
        for b in range(1,nb):
            if invm[b]>0:
                m[6*b:6*b+3,6*b:6*b+3]=np.eye(3)*invm[b]
                m[6*b+3:6*b+6,6*b+3:6*b+6]=invi[b]
        c=np.zeros((n,6*nb)); normals=cc.lambdas.numpy()[:3,:n].T; derived=cc.derived.numpy()
        b0=r.contact_body0.numpy()[:n]; b1=r.contact_body1.numpy()[:n]
        for k in range(n):
            for b,sgn,start in [(b0[k],-1,9),(b1[k],1,12)]:
                direction=sgn*normals[k]
                c[k,6*b:6*b+3]+=direction
                c[k,6*b+3:6*b+6]+=np.cross(derived[start:start+3,k],direction)
        a=j@m@j.T
        scale=d.row_scale.numpy().astype(float)
        mat=d.matrix.numpy(); ar=np.zeros_like(a)
        rr=d.matrix_row.numpy(); cl=d.matrix_column.numpy(); st=d.matrix_storage.numpy()
        ar[rr,cl]=mat[st]; ar[cl,rr]=mat[st]
        # Diagonal indices are not necessarily in the off-diagonal list.
        np.fill_diagonal(ar,1.0)
        ar/=scale[:,None]*scale[None,:]
        un=np.einsum("ij,ij->i",c@m,c)
        rhs=j@m@c.T
        ref=un-np.sum(rhs*np.linalg.solve(ar,rhs),axis=0)
        exact=un-np.sum(rhs*np.linalg.pinv(a,rcond=1e-11)@rhs,axis=0) if False else un-np.sum(rhs*(np.linalg.pinv(a,rcond=1e-11)@rhs),axis=0)
        gpu=un-r.gram.numpy()[0,:n]
        eff=r.mobility.numpy()[0,:n]
        print("CALL",calls,"contacts",n,"rank",np.linalg.matrix_rank(a),"rows",len(rj),"matrixdiff",np.max(abs(ar-a)),flush=True)
        inds=np.argsort(eff)[-12:]
        for k in inds:
            print(k,b0[k],b1[k],"eff",eff[k],"un",un[k],"gpu",gpu[k],"regref",ref[k],"exactref",exact[k],"gap",derived[15,k],flush=True)
        np.savez("/tmp/colibri_schur_ref_%d.npz"%calls,j=j,m=m,c=c,a=a,ar=ar,un=un,ref=ref,exact=exact,gpu=gpu,eff=eff,b0=b0,b1=b1,gap=derived[15,:n])
    old(**kwargs)
w._solve_direct_contacts=hook
for frame in range(30):
    print("FRAME",frame,flush=True)
    e.step()
