import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.phoenx_scene import Example
from newton.viewer import ViewerNull

p = argparse.ArgumentParser()
p.add_argument("--body-count", type=int, default=36)
p.add_argument("--mode", default="maximal")
p.add_argument("--substeps", type=int, default=8)
p.add_argument("--iterations", type=int, default=8)
p.add_argument("--outer-substeps", type=int, default=1)
p.add_argument("--frames", type=int, default=180)
p.add_argument("--layout", default="single_world")
p.add_argument("--stages", action="store_true")
p.add_argument("--start", type=int, default=1)
p.add_argument("--frictionless", action="store_true")
p.add_argument("--project-initial", action="store_true")
p.add_argument("--matching", choices=["disabled", "latest", "sticky"], default="sticky")
p.add_argument("--free-base", action="store_true")
p.add_argument("--fix-base", action="store_true")
p.add_argument("--contact-gap", type=float, default=0.0001)
p.add_argument("--source-contact-offsets", action="store_true")
p.add_argument("--source-damping", action="store_true")
p.add_argument("--mesh-cylinders", action="store_true")
a = p.parse_args()
if a.fix_base and a.free_base:
    p.error("Choose only one base mode")
import gc

for n in range(a.start, a.body_count + 1) if a.stages else [a.body_count]:
    a.body_count = n
    e = Example(ViewerNull(), a)
    t = time.perf_counter()
    for frame in range(a.frames):
        e.step()
        if frame % 10 == 0:
            print("PROGRESS", n, frame, time.perf_counter() - t, flush=True)
        try:
            e.test_post_step()
        except AssertionError:
            q = e.state_0.body_q.numpy()
            qd = e.state_0.body_qd.numpy()
            speeds = np.linalg.norm(qd, axis=1)
            print("FAILED BODY", a.body_count, "FRAME", frame, "SECONDS", time.perf_counter() - t, flush=True)
            print("FASTEST", [(e.model.body_label[i], float(speeds[i])) for i in np.argsort(speeds)[-5:]], flush=True)
            target = Path(
                f"/tmp/colibri_phoenx_failure_{a.mode}_{a.body_count}_{a.substeps}_{a.outer_substeps}_{a.iterations}.npz"
            )
            np.savez(
                target,
                q=q,
                qd=qd,
                labels=e.model.body_label,
                contact_derived=e.solver.world._contact_container.derived.numpy(),
                contact_impulses=e.solver.world._contact_container.impulses.numpy(),
                args=str(vars(a)),
                frame=frame,
            )
            print("SAVED", target, flush=True)
            raise
        if (frame + 1) % 60 == 0:
            print("PROGRESS", n, frame + 1, "seconds", time.perf_counter() - t, flush=True)
    wp.synchronize()
    elapsed = time.perf_counter() - t
    print("PASS", vars(a), "seconds", elapsed, "fps_with_checks", a.frames / elapsed, flush=True)
    del e
    gc.collect()
