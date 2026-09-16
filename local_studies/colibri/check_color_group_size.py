# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Vary only the public constructor color-group option in the Colibri audit."""

import argparse
import json
import runpy
import sys
from pathlib import Path

import newton.solvers

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--color-group-size", type=int, required=True)
parser.add_argument("--ownership-output", type=Path)
parser.add_argument("--max-colored-partitions", type=int)
parser.add_argument("--mass-splitting-batch-size", type=int)
options, remaining = parser.parse_known_args()
worlds = []
original = newton.solvers.SolverPhoenX


class ConfiguredSolver(original):
    def __init__(self, *args, **kwargs):
        kwargs["mass_splitting_color_group_size"] = options.color_group_size
        if options.max_colored_partitions is not None:
            kwargs["max_colored_partitions"] = options.max_colored_partitions
        if options.mass_splitting_batch_size is not None:
            kwargs["mass_splitting_batch_size"] = options.mass_splitting_batch_size
        super().__init__(*args, **kwargs)
        worlds.append(self.world)


newton.solvers.SolverPhoenX = ConfiguredSolver
sys.argv = [sys.argv[0], *remaining]
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    newton.solvers.SolverPhoenX = original
    if options.ownership_output is not None:
        records = []
        for world in worlds:
            partitioner = world._partitioner
            num_colors = int(partitioner.num_colors.numpy()[0])
            starts = partitioner.color_starts.numpy()
            cap = int(world.max_colored_partitions)
            count = world._copy_state.count_per_node.numpy()
            group_data = world._color_group_data
            records.append(
                {
                    "scope": "Final captured graph ownership; no simulation kernel changes",
                    "color_group_size": world.mass_splitting_color_group_size,
                    "max_colored_partitions": cap,
                    "mass_splitting_batch_size": world.mass_splitting_batch_size,
                    "partitioner_algorithm": world.partitioner_algorithm,
                    "count_per_node": count.tolist(),
                    "num_colors": (
                        group_data["num_colors"].numpy().tolist() if group_data is not None else [num_colors]
                    ),
                    "ordinary_num_colors": num_colors,
                    "ordinary_color_starts": starts[: num_colors + 1].tolist(),
                    "ordinary_rows_per_color": (starts[1 : num_colors + 1] - starts[:num_colors]).tolist(),
                    "ordinary_overflow_row_count": int(starts[cap + 1] - starts[cap]) if cap < num_colors else 0,
                    "ordinary_active_row_count": int(starts[num_colors]),
                    "ordinary_element_ids_by_color": partitioner.element_ids_by_color.numpy()[
                        : int(starts[num_colors])
                    ].tolist(),
                    "ordinary_csr_is_solve_schedule": group_data is None,
                    "num_active_constraints": int(world._num_active_constraints.numpy()[0]),
                    "body_copy_counts": count[: world.num_bodies].tolist(),
                }
            )
        options.ownership_output.write_text(json.dumps(records, indent=2))
