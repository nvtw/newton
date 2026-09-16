"""CPU proof that dormant matching preserves fresh collision geometry."""

import numpy as np
import warp as wp

import newton
from local_studies.colibri.history_only_matching import install


def run(candidate):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = 0.002
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09999)))
    builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device="cpu")
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
    if candidate:
        install(pipeline)
    contacts = pipeline.contacts()
    results = []
    for x, z in ((0.0, 0.09999), (0.0001, 0.100005), (0.0001, 0.1006)):
        q = state.body_q.numpy()
        q[0, 0], q[0, 2] = x, z
        state.body_q.assign(q)
        contacts.clear()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert count == 1
        fields = {}
        for name in ("point0", "point1", "normal", "offset0", "offset1", "margin0", "margin1", "shape0", "shape1"):
            fields[name] = getattr(contacts, "rigid_contact_" + name).numpy()[:count].copy()
        results.append((int(contacts.rigid_contact_match_index.numpy()[0]), fields))
    return results


def main():
    baseline, candidate = run(False), run(True)
    assert baseline[1][0] < 0 and candidate[1][0] == 0
    assert candidate[2][0] < 0
    for b, c in zip(baseline, candidate, strict=True):
        for name in b[1]:
            np.testing.assert_array_equal(b[1][name], c[1][name], err_msg=name)
    print("PASS: 5um positive-gap identity retained; >500um rejected; all9 geometry fields byte-equal at all3 queries")


if __name__ == "__main__":
    main()
