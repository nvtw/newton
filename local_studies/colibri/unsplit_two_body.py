"""Unsplit two-body native control; audit complete graph ownership after each frame.

Compose outside this wrapper with friction_break_state and total_normal_friction.
No kernel, constraint order, physical parameter or source file is modified.
"""

import json
import runpy
import sys
from pathlib import Path

import numpy as np

import newton.examples
import newton.solvers

original_solver = newton.solvers.SolverPhoenX
original_run = newton.examples.run
output = Path(sys.argv[sys.argv.index("--output") + 1])
records = []
checks = 0
worlds = []


class UnsplitSolver(original_solver):
    """Disable splitting without restricting the ordinary native color graph."""

    def __init__(self, *args, **kwargs):
        kwargs["mass_splitting"] = False
        kwargs["mass_splitting_color_group_size"] = 0
        kwargs["mass_splitting_unrolled"] = False
        # No soft cap is used when mass splitting is disabled. Preserve normal
        # greedy iteration budget; the native non-MS path has a JP fallback.
        super().__init__(*args, **kwargs)
        w = self.world
        assert not w.mass_splitting_enabled
        assert w.max_colored_partitions is None
        assert w._partitioner.max_colored_partitions is None
        assert w._color_group_data is None
        assert w.num_joints == 1
        worlds.append(w)


def audit(world):
    """Reject missing/duplicate rows, conflicting colors, or any copy ownership."""
    global checks
    partitioner = world._partitioner
    nc = int(partitioner.num_colors.numpy()[0])
    starts = partitioner.color_starts.numpy()[: nc + 1]
    active = int(world._num_active_constraints.numpy()[0])
    ids = partitioner.element_ids_by_color.numpy()[: int(starts[-1])]
    np.testing.assert_array_equal(np.sort(ids), np.arange(active))
    assert np.all(np.diff(starts) >= 0)
    assert starts[0] == 0
    nodes = world._elements.numpy()["bodies"][:active]
    for first, last in zip(starts[:-1], starts[1:], strict=True):
        used = set()
        for row in ids[first:last]:
            dynamic = set(int(n) for n in nodes[row] if n >= 0)
            assert not used.intersection(dynamic), "Conflicting ordinary color"
            used.update(dynamic)
    copies = int(world._copy_state.highest_index_in_use.numpy()[0])
    assert copies == 0
    assert np.all(world._copy_state.count_per_node.numpy() == 0)
    columns = int(world._ingest_scratch.num_contact_columns.numpy()[0])
    assert active == world._contact_offset + columns
    headers = world._contact_cols.data.numpy().view(np.int32)
    points = []
    for first, count in headers[5:7, :columns].T:
        points.extend(range(int(first), int(first + count)))
    assert len(set(points)) == len(points)
    assert int(world.constraints.bilateral.row_count.numpy()[0]) == 6
    record = {
        "num_colors": nc,
        "active_constraints": active,
        "contact_columns": columns,
        "contact_points": len(points),
        "copy_slots": copies,
        "rows_per_color": np.diff(starts).tolist(),
        "all_active_ids_present_once": True,
        "colors_conflict_free": True,
        "joint_rows": 6,
        "ordinary_soft_cap": None,
    }
    checks += 1
    if not records or record != records[-1]:
        records.append(record)


def audited_run(example, args):
    """Observe completed native graphs while keeping existing physics checks."""
    assert args.body_count == 2
    original_step = example.step

    def step():
        original_step()
        audit(example.solver.world)

    example.step = step
    return original_run(example, args)


newton.solvers.SolverPhoenX = UnsplitSolver
newton.examples.run = audited_run
try:
    runpy.run_module("local_studies.colibri.staged_base_mechanism", run_name="__main__")
finally:
    newton.solvers.SolverPhoenX = original_solver
    newton.examples.run = original_run
    output.with_suffix(".unsplit.json").write_text(
        json.dumps(
            {
                "scope": "Native unsplit dispatcher; ordinary uncapped coloring; post-frame graph ownership audit",
                "checks": checks,
                "records": records,
                "world_count": len(worlds),
                "no_kernel_changes": True,
                "not_isolated_timing": True,
            },
            indent=2,
        )
    )
