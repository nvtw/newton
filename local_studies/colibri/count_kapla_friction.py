"""Count actual friction-root iterations during a Kapla solver frame."""
import json
import numpy as np
import warp as wp
from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as cloth
from local_studies.colibri import counted_contact_projection as counted
from newton._src.solvers.phoenx.examples import example_kapla_tower as ek
from newton._src.solvers.phoenx.benchmarks.bench_phoenx_kapla import _run_one

cloth.contact_project_coupled_velocity_update = counted.contact_project_coupled_velocity_update
cloth.contact_project_coupled_velocity_update_no_soft_pd = counted.contact_project_coupled_velocity_update_no_soft_pd
original = ek.Example
instances = []
def build(*args, **kwargs):
    example = original(*args, **kwargs)
    cc = example.world._contact_container
    values = cc.derived.numpy()
    cc.derived = wp.array(np.pad(values, ((0, 1), (0, 0))), dtype=wp.float32, device=example.device)
    instances.append(example)
    return example
ek.Example = build
_run_one(mass_splitting=False, substeps=4, solver_iterations=10, max_colored_partitions=8,
    prepare_refresh_stride=1, warmup_frames=0, measured_frames=1, grid_dims=(1, 1),
    blocks_per_sm=8, colored_contact_layout=False, partitioner_algorithm="greedy")
hist = instances[0].world._contact_container.derived.numpy()[-1, :41].astype(np.int64)
print("HISTOGRAM", json.dumps(hist.tolist()))
print("SLIDING_MEAN", float(hist @ np.arange(41) / max(1, hist[1:].sum())))
