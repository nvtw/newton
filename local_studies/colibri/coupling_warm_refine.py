import time

from newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop import Example
from newton.viewer import ViewerNull

e = Example(ViewerNull(), width=6, height=6)
e.graph = None
import warp as wp

original_launch = wp.launch_tiled


def filtered_launch(*args, **kwargs):
    result = original_launch(*args, **kwargs)
    kernel = kwargs.get("kernel", args[0] if args else None)
    if getattr(kernel, "key", "") == "warm_start_direct_contact_runs_kernel":
        d = e.solver._direct_equality_system
        d.compute_bias_velocity()
        d.apply_bias_velocity(-1.0)
        d.solve(use_bias=False)
        d.apply_bias_velocity(1.0)
    return result


wp.launch_tiled = filtered_launch
t = time.perf_counter()
for frame in range(120):
    e.step()
    if frame % 10 == 0:
        print(frame, e.state.body_q.numpy()[e.cube_body, 2], time.perf_counter() - t, flush=True)
e.test_final()
print("HEIGHT", e.state.body_q.numpy()[e.cube_body, 2], flush=True)
