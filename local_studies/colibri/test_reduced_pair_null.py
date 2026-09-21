import numpy as np
import warp as wp

import newton

b = newton.ModelBuilder(gravity=wp.vec3(0.0))
rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, 0.7, 0.5)), 0.73)
q = wp.transform(wp.vec3(0.031, 0.017, 0.009), rot)
local = wp.transform(wp.vec3(0, 0, 0.019), wp.quat_identity())
b0 = b.add_link(xform=q)
b1 = b.add_link(xform=q * local)
for body in [b0, b1]:
    b.add_shape_box(body, hx=0.01, hy=0.02, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000, mu=0))
j0 = b.add_joint_free(b0)
j1 = b.add_joint_revolute(b0, b1, axis=newton.Axis.Z, parent_xform=local, collision_filter_parent=False)
b.add_articulation([j0, j1])
m = b.finalize(device="cuda:0")
p = newton.CollisionPipeline(m, contact_matching="sticky", rigid_contact_max=32)
s = newton.solvers.SolverPhoenX(
    m, collision_pipeline=p, articulation_mode="reduced", step_layout="single_world", sor_boost=1, substeps=1
)
st = m.state()
c = p.contacts()
p.collide(st, c)
s.step(st, st, m.control(), c, 1 / 1200)
n = int(c.rigid_contact_count.numpy()[0])
d = s.world._contact_container.derived.numpy()
print(
    "NORMAL_MASS",
    d[0, :n],
    "NORMAL_IMPULSE",
    s.world._contact_container.impulses.numpy()[0, :n],
    "VEL",
    st.body_qd.numpy(),
    flush=True,
)
assert n > 0
np.testing.assert_array_equal(d[0, :n], np.zeros(n))
