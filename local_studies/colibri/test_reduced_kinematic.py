import numpy as np
import warp as wp
import newton
b=newton.ModelBuilder(); q=wp.transform(wp.vec3(0,0,1),wp.quat_identity()); body=b.add_link(xform=q,is_kinematic=True); b.add_shape_sphere(body,radius=.01); j=b.add_joint_free(body); b.add_articulation([j]); m=b.finalize(device="cuda:0"); p=newton.CollisionPipeline(m,contact_matching="sticky",rigid_contact_max=32); s=newton.solvers.SolverPhoenX(m,collision_pipeline=p,articulation_mode="reduced",step_layout="single_world",sor_boost=1,substeps=1); st=m.state(); c=p.contacts()
for i in range(2): p.collide(st,c); s.step(st,st,m.control(),c,1/60); print(i,st.body_q.numpy(),st.body_qd.numpy(),flush=True)
np.testing.assert_allclose(st.body_q.numpy()[0,:3],[0,0,1],atol=1e-6)
