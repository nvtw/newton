import numpy as np
import warp as wp
import newton
b=newton.ModelBuilder(gravity=wp.vec3(0.0))
body=b.add_link()
b.add_shape_sphere(body,xform=wp.transform(wp.vec3(0.02,0,0.00999),wp.quat_identity()),radius=0.01,cfg=newton.ModelBuilder.ShapeConfig(density=1000,mu=0.0))
j=b.add_joint_revolute(-1,body,axis=wp.normalize(wp.vec3(0,0.001,1)))
b.add_articulation([j])
b.add_ground_plane()
m=b.finalize(device="cuda:0")
p=newton.CollisionPipeline(m,rigid_contact_max=32,contact_matching="sticky")
s=newton.solvers.SolverPhoenX(m,collision_pipeline=p,articulation_mode="reduced",step_layout="single_world",sor_boost=1.0,substeps=1,solver_iterations=8)
st=m.state(); c=p.contacts(); p.collide(st,c); s.step(st,st,m.control(),c,1/1200)
print("POSE",st.body_q.numpy(),"JOINT",st.joint_q.numpy()); print("VELOCITY",st.body_qd.numpy(),"CONTACTS",c.rigid_contact_count.numpy(),"BIAS",s.world._contact_container.derived.numpy()[3,:4],flush=True)
assert np.max(np.abs(st.body_qd.numpy()))<1, "Nearly blocked penetration correction accelerates hinge"
assert abs(float(st.joint_q.numpy()[0])) < 0.001, "Nearly blocked penetration correction rotates the hinge excessively"
