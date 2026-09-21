import argparse

import numpy as np
import warp as wp

import newton
from local_studies.colibri.phoenx_scene import Example
from newton.examples.kamino.example_kamino_colibri import JOINTS
from newton.viewer import ViewerNull


@wp.kernel
def contact_gap(
    q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    s0: wp.array[wp.int32],
    s1: wp.array[wp.int32],
    p0: wp.array[wp.vec3],
    p1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    m0: wp.array[wp.float32],
    m1: wp.array[wp.float32],
    gap: wp.array[wp.float32],
):
    i = wp.tid()
    b0 = shape_body[s0[i]]
    b1 = shape_body[s1[i]]
    x0 = p0[i]
    x1 = p1[i]
    if b0 >= 0:
        x0 = wp.transform_point(q[b0], x0)
    if b1 >= 0:
        x1 = wp.transform_point(q[b1], x1)
    gap[i] = wp.dot(x1 - x0, normal[i]) - m0[i] - m1[i]


p = argparse.ArgumentParser()
p.add_argument("--body-count", type=int, default=15)
p.add_argument("--frames", type=int, default=180)
p.add_argument("--forest", action="store_true")
a = p.parse_args()
e = Example(
    ViewerNull(),
    argparse.Namespace(
        body_count=a.body_count,
        mode="reduced",
        layout="single_world",
        substeps=10,
        outer_substeps=2,
        iterations=8,
        source_contact_offsets=True,
        contact_gap=0.001,
        mesh_cylinders=True,
        no_graph=a.forest,
    ),
)
if a.forest:
    from newton._src.solvers.phoenx.articulations.reduced_forest import ReducedForestContactSystem

    e.solver._reduced_articulation.forest_contact_system = ReducedForestContactSystem(
        e.solver._reduced_articulation, e.contact_capacity
    )
    with wp.ScopedCapture(device=e.model.device) as capture:
        e.simulate()
    e.graph = capture.graph
audit = newton.CollisionPipeline(e.model, rigid_contact_max=8192, contact_matching="disabled")
fresh = audit.contacts()
gap = wp.zeros(8192, dtype=wp.float32, device=e.model.device)
pairs = {frozenset((j[0], j[1])) for j in JOINTS}
shape_body = e.model.shape_body.numpy()
for frame in range(a.frames + 1):
    if frame:
        e.step()
    audit.collide(e.state_0, fresh)
    n = int(fresh.rigid_contact_count.numpy()[0])
    wp.launch(
        contact_gap,
        dim=n,
        inputs=[
            e.state_0.body_q,
            e.model.shape_body,
            fresh.rigid_contact_shape0,
            fresh.rigid_contact_shape1,
            fresh.rigid_contact_point0,
            fresh.rigid_contact_point1,
            fresh.rigid_contact_normal,
            fresh.rigid_contact_margin0,
            fresh.rigid_contact_margin1,
        ],
        outputs=[gap],
        device=e.model.device,
    )
    gaps = gap.numpy()[:n]
    s0 = fresh.rigid_contact_shape0.numpy()[:n]
    s1 = fresh.rigid_contact_shape1.numpy()[:n]
    for x, y in zip(s0, s1):
        b0, b1 = int(shape_body[x]), int(shape_body[y])
        n0 = e.model.body_label[b0] if b0 >= 0 else "ground"
        n1 = e.model.body_label[b1] if b1 >= 0 else "ground"
        assert frozenset((n0, n1)) not in pairs, (n0, n1)
    i = int(np.argmin(gaps))
    if frame % 10 == 0 or gaps[i] < -0.003:
        print(
            "FRAME",
            frame,
            "N",
            n,
            "MIN_GAP",
            float(gaps[i]),
            "PAIR",
            e.model.shape_label[s0[i]],
            e.model.shape_label[s1[i]],
            "SPEED",
            np.max(np.linalg.norm(e.state_0.body_qd.numpy(), axis=1)),
            flush=True,
        )
    try:
        e.test_post_step()
    except AssertionError:
        print("FAILED", frame, flush=True)
        raise
