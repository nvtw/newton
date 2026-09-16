
import numpy as np
import warp as wp
import newton
from newton.examples.kamino.example_kamino_colibri import build_scene, JOINTS
model=build_scene(mesh_cylinders=True).finalize(skip_validation_joints=True)
pipeline=newton.CollisionPipeline(model,rigid_contact_max=8192,contact_matching="sticky")
contacts=pipeline.contacts()
state=model.state()
for label in ("initial","failure"):
    if label=="failure":
        snapshot=np.load("/tmp/colibri_phoenx_failure_reduced_36_10_2_8.npz")
        state.body_q.assign(snapshot["q"])
        state.body_qd.assign(snapshot["qd"])
    pipeline.collide(state,contacts)
    count=int(contacts.rigid_contact_count.numpy()[0])
    shapes0=contacts.rigid_contact_shape0.numpy()[:count]
    shapes1=contacts.rigid_contact_shape1.numpy()[:count]
    sb=model.shape_body.numpy()
    joint_pairs={frozenset((a,b)) for a,b,*_ in JOINTS}
    bad=[]
    rack=[]
    for index,(s0,s1) in enumerate(zip(shapes0,shapes1)):
        b0,b1=int(sb[s0]),int(sb[s1])
        n0=model.body_label[b0] if b0>=0 else "ground"
        n1=model.body_label[b1] if b1>=0 else "ground"
        if frozenset((n0,n1)) in joint_pairs:
            bad.append((index,n0,n1))
        if "TailRack" in (n0,n1):
            rack.append((n0,n1))
    from collections import Counter
    print(label,"contacts",count,"forbidden",bad,"rack",Counter(rack),flush=True)
    assert not bad
