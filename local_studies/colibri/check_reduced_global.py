import argparse

import warp as wp

from local_studies.colibri.phoenx_scene import Example
from newton.viewer import ViewerNull

p = argparse.ArgumentParser()
p.add_argument("--frames", type=int, default=180)
p.add_argument("--body-count", type=int, default=15)
p.add_argument("--fix-base", action="store_true")
a = p.parse_args()
a.mode = "reduced"
a.layout = "single_world"
a.substeps = 10
a.outer_substeps = 2
a.iterations = 8
a.source_contact_offsets = True
e = Example(ViewerNull(), a)
e.graph = None
cbs = e.solver._reduced_articulation.contact_block_system
cbs.biased_page_launcher = None
cbs.relax_page_launcher = None
original = cbs.solve


def global_sweep(*args, **kwargs):
    iterations = args[6]
    args = (*args[:6], 1)
    original(*args, **kwargs)
    kwargs["prepare"] = False
    for _ in range(iterations - 1):
        original(*args, **kwargs)


cbs.solve = global_sweep
with wp.ScopedCapture() as capture:
    e.simulate()
e.graph = capture.graph
for i in range(a.frames):
    e.step()
    try:
        e.test_post_step()
    except AssertionError:
        import numpy as np

        q = e.state_0.body_q.numpy()
        v = e.state_0.body_qd.numpy()
        print(
            "FAILED",
            i,
            list(
                zip(
                    e.model.body_label, np.linalg.norm(q[:, :3] - e.initial_q[:, :3], axis=1), np.linalg.norm(v, axis=1)
                )
            ),
            flush=True,
        )
        raise
print("PASS GLOBAL", vars(a), flush=True)
