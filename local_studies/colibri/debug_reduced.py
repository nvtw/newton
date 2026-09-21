import argparse

import numpy as np

from local_studies.colibri.phoenx_scene import Example
from newton.viewer import ViewerNull

e = Example(
    ViewerNull(),
    argparse.Namespace(body_count=15, fix_base=True, mode="reduced", layout="single_world", substeps=1, iterations=8),
)
e.graph = None
e.frame_dt = 1.0 / 1200
for frame in range(100):
    e.step()
    cc = e.solver.world._contact_container
    n = int(e.contacts.rigid_contact_count.numpy()[0])
    d = cc.derived.numpy()
    impulses = cc.impulses.numpy()[0]
    q = e.state_0.body_q.numpy()
    v = e.state_0.body_qd.numpy()
    print(
        frame,
        "contacts",
        n,
        "speed",
        np.nanmax(abs(v)),
        "effmass",
        np.nanmax(d[:3, :n]),
        "gap",
        np.nanmin(d[15, :n]),
        "impulse",
        np.nanmax(impulses),
        flush=True,
    )
    if not np.isfinite(q).all() or not np.isfinite(v).all() or np.nanmax(abs(v)) > 100:
        print("SPEEDS", list(zip(e.model.body_label, np.linalg.norm(v, axis=1))), flush=True)
        for k in np.argsort(impulses)[-10:]:
            print("CONTACT", int(k), impulses[k], d[:, k], flush=True)
        break

r = e.solver._reduced_articulation.contact_block_system
for name in [
    "packed_jacobian",
    "packed_response",
    "point_contact",
    "point_count",
    "row_body",
    "row_body_pair",
    "row_wrench",
    "row_wrench_pair",
]:
    a = getattr(r, name).numpy()
    print(name, a.shape, flush=True)
    np.save("/tmp/reduced_" + name + ".npy", a)
np.save("/tmp/reduced_derived.npy", d)
