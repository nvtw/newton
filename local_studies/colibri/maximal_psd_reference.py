import argparse
import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from newton._src.solvers.phoenx.constraints.constraint_contact import _OFF_BODY1, _OFF_BODY2, _OFF_CONTACT_FIRST, _OFF_CONTACT_COUNT
from local_studies.colibri.phoenx_scene import Example
wp.config.quiet=True
e=Example(ViewerNull(),argparse.Namespace(body_count=15,mode="maximal",layout="single_world",substeps=1,iterations=8))
e.graph=None; e.frame_dt=1/1200
w=e.solver.world; response=e.solver._maximal_contact_response; r=response.data; t=response.projector.data
old=w._solve_maximal_articulated_contacts
calls=0
def hook(**kwargs):
    global calls
    old(**kwargs);calls+=1
    if not kwargs["use_bias"]: return
    cc=w._contact_container; n=int(e.contacts.rigid_contact_count.numpy()[0]); shapes=e.model.shape_body.numpy()+1
    b0=shapes[e.contacts.rigid_contact_shape0.numpy()[:n]]; b1=shapes[e.contacts.rigid_contact_shape1.numpy()[:n]]
    cols=w._contact_cols.data.numpy().view(np.int32)
    for column in range(cols.shape[1]):
        first=cols[int(_OFF_CONTACT_FIRST),column]; count=cols[int(_OFF_CONTACT_COUNT),column]
        if first>=0 and count>0 and first+count<=n:
            b0[first:first+count]=cols[int(_OFF_BODY1),column];b1[first:first+count]=cols[int(_OFF_BODY2),column]
    art=r.body_articulation.numpy(); lanes=r.body_lane.numpy(); parent=t.parent.numpy(); depth=t.depth.numpy(); motion=t.motion.numpy().astype(float); invd=t.inverse_d.numpy().astype(float)
    mapping=r.conditional_map.numpy().astype(float); mobility=r.mobility.numpy().astype(float); deriv=cc.derived.numpy(); normals=cc.lambdas.numpy()[:3,:n].T
    olddiag=[]; psddiag=[]
    for k in range(n):
        a=art[b0[k]]
        if a<0 or art[b1[k]]!=a: olddiag.append(0);psddiag.append(0);continue
        no0=lanes[b0[k]]; no1=lanes[b1[k]]; f0=np.r_[-normals[k],np.cross(deriv[9:12,k],-normals[k])].astype(float); f1=np.r_[normals[k],np.cross(deriv[12:15,k],normals[k])].astype(float)
        original=f0@mobility[a,no0]@f0+f1@mobility[a,no1]@f1
        value=0
        while no0!=no1:
            if depth[a,no0]>=depth[a,no1]:
                value+=invd[a,no0]*(motion[a,no0]@f0)**2;f0=mapping[a,no0].T@f0;no0=parent[a,no0]
            else:
                value+=invd[a,no1]*(motion[a,no1]@f1)**2;f1=mapping[a,no1].T@f1;no1=parent[a,no1]
        olddiag.append(original+2*f0@mobility[a,no0]@f1)
        f=f0+f1; value+=f@mobility[a,no0]@f;psddiag.append(value)
    olddiag=np.array(olddiag);psddiag=np.array(psddiag); eff=w._maximal_contact_schedule.mobility.numpy()[0,:n]
    order=np.argsort(eff)[-10:]
    print("CALL",calls,"speed",np.max(abs(w.bodies.angular_velocity.numpy())),"count",n,"mostnegative",np.min(olddiag) if n else 0,flush=True)
    for k in order: print(k,b0[k],b1[k],"eff",eff[k],"sum4",olddiag[k],"PSD",psddiag[k],"gap",deriv[15,k],flush=True)
    np.savez("/tmp/colibri_maximal_psd_%d.npz"%calls,b0=b0,b1=b1,old=olddiag,psd=psddiag,eff=eff,normal=normals,derived=deriv[:,:n],motion=motion,mapping=mapping,mobility=mobility,invd=invd,parent=parent,depth=depth,art=art,lanes=lanes)
w._solve_maximal_articulated_contacts=hook
for frame in range(40):e.step()
