import argparse
import numpy as np
import warp as wp
import trimesh
from newton.viewer import ViewerNull
from newton.examples.kamino.example_kamino_colibri import build_scene
from local_studies.colibri.phoenx_scene import Example
wp.config.quiet=True
e=Example(ViewerNull(),argparse.Namespace(body_count=15,mode="maximal",layout="single_world",substeps=1,iterations=8))
e.graph=None; e.frame_dt=1/1200
builder=build_scene(body_count=15)
shape_body=e.model.shape_body.numpy(); shape_q=e.model.shape_transform.numpy()
def point(q,p): return np.array(wp.transform_point(wp.transform(wp.vec3(q[:3]),wp.quat(q[3:])),wp.vec3(p)))
def inverse_point(q,p): return np.array(wp.transform_point(wp.transform_inverse(wp.transform(wp.vec3(q[:3]),wp.quat(q[3:]))),wp.vec3(p)))
meshes={}
for k,src in enumerate(builder.shape_source):
    if hasattr(src,"vertices"):
        meshes[k]=trimesh.Trimesh(vertices=np.array(src.vertices),faces=np.array(src.indices).reshape(-1,3),process=False)
for frame in range(4):
    e.collision_pipeline.collide(e.state_0,e.contacts)
    count=int(e.contacts.rigid_contact_count.numpy()[0]); s0=e.contacts.rigid_contact_shape0.numpy()[:count]; s1=e.contacts.rigid_contact_shape1.numpy()[:count]
    p0=e.contacts.rigid_contact_point0.numpy()[:count]; p1=e.contacts.rigid_contact_point1.numpy()[:count]; n=e.contacts.rigid_contact_normal.numpy()[:count]; q=e.state_0.body_q.numpy()
    report=[]
    for k in range(count):
        b0=shape_body[s0[k]]; b1=shape_body[s1[k]]
        if set((b0,b1))!={14,1}: continue
        a=point(q[b0],p0[k]); b=point(q[b1],p1[k]); gap=np.dot(b-a,n[k])
        report.append((gap,k,a,b))
    report.sort(key=lambda v:v[0])
    print("FRAME",frame,"count",count,"speed",np.max(abs(e.state_0.body_qd.numpy())),"hypoframe",len(report),flush=True)
    for gap,k,a,b in report[:3]:
        print("CONTACT",k,"gap",gap,"normal",n[k],"shapes",s0[k],s1[k],e.model.shape_label[s0[k]],e.model.shape_label[s1[k]],"world",a,b,flush=True)
        for sid,pt in ((s0[k],b),(s1[k],a)):
            if sid not in meshes:
                local=inverse_point(shape_q[sid],inverse_point(q[shape_body[sid]],pt))
                scale=e.model.shape_scale.numpy()[sid]
                radial=np.linalg.norm(local[:2])-scale[0]
                axial=abs(local[2])-scale[1]
                sdf=min(max(radial,axial),0)+np.linalg.norm(np.maximum([radial,axial],0))
                print("CYL",sid,"scale",scale,"point",local,"radial",radial,"axial",axial,"sdf",sdf,flush=True)
                continue
            bidx=shape_body[sid]; local=inverse_point(shape_q[sid],inverse_point(q[bidx],pt))
            closest,dist,tri=trimesh.proximity.closest_point_naive(meshes[sid],local[None,:])
            norm=meshes[sid].face_normals[tri[0]]
            print("TRI",sid,"dist",dist[0],"signed_normal",np.dot(local-closest[0],norm),"localquery",local,"closest",closest[0],"normal",norm,flush=True)
    np.savez("/tmp/colibri_geometry_frame%d.npz"%frame,q=q,s0=s0,s1=s1,p0=p0,p1=p1,n=n)
    e.step()
