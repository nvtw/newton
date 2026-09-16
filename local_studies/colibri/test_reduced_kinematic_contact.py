import numpy as np
import warp as wp
import newton
b=newton.ModelBuilder(gravity=wp.vec3(0.0)); kin=b.add_link(xform=wp.transform(wp.vec3(0,0,1),wp.quat_identity()),is_kinematic=True); dyn=b.add_link(xform=wp.transform(wp.vec3(0,0,1.019),wp.quat_identity()))
for body in [kin,dyn]: b.add_shape_sphere(body,radius=.01,cfg=newton.ModelBuilder.ShapeConfig(density=1000,mu=0)); b.add_articulation([b.add_joint_free(body)])
m=b.finalize(device="cuda:0"); p=newton.CollisionPipeline(m,contact_matching="sticky",rigid_contact_max=32); s=newton.solvers.SolverPhoenX(m,collision_pipeline=p,articulation_mode="reduced",step_layout="single_world",sor_boost=1,substeps=1); st=m.state(); c=p.contacts(); p.collide(st,c); s.step(st,st,m.control(),c,1/1200); print("POSE",st.body_q.numpy(),"VEL",st.body_qd.numpy(),flush=True); np.testing.assert_array_equal(st.body_qd.numpy()[kin],np.zeros(6)); np.testing.assert_allclose(st.body_q.numpy()[kin,:3],[0,0,1],atol=1e-6)
