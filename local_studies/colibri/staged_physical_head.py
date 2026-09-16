"""Authored staged mechanism with physical head and one sequential overflow batch."""

import json
import runpy
import sys
from pathlib import Path

import newton.solvers
from local_studies.colibri import physical_head_overflow

output = Path(sys.argv[sys.argv.index("--output") + 1])
original = newton.solvers.SolverPhoenX
worlds = []


class ConfiguredSolver(original):
    """Keep the solve budget and physical inputs, changing only ownership/schedule."""

    def __init__(self, *args, **kwargs):
        kwargs.update(mass_splitting_color_group_size=0, max_colored_partitions=12, mass_splitting_batch_size=1024)
        super().__init__(*args, **kwargs)
        worlds.append(self.world)


physical_head_overflow.install()
newton.solvers.SolverPhoenX = ConfiguredSolver
try:
    runpy.run_module("local_studies.colibri.staged_base_mechanism", run_name="__main__")
finally:
    newton.solvers.SolverPhoenX = original
    records = []
    for world in worlds:
        counts = world._copy_state.count_per_node.numpy()
        starts = world._partitioner.color_starts.numpy()
        records.append(
            {
                "counts": counts.tolist(),
                "row_partitions": world._partitioner.interaction_id_to_partition.numpy()[
                    : int(world._num_active_constraints.numpy()[0])
                ].tolist(),
                "color_starts": starts[:14].tolist(),
                "overflow_rows": int(starts[13] - starts[12]),
                "phase": "final graph",
                "max_colored_partitions": 12,
                "overflow_batch_size": 1024,
            }
        )
        assert int(starts[13] - starts[12]) <= 1024
        assert max(counts) <= 1
    output.with_suffix(".ownership.json").write_text(json.dumps(records, indent=2))
