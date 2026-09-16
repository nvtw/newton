# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Capture final copy velocities immediately before mass-splitting averaging."""

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.solver import SolverPhoenX

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--copy-snapshot", type=Path, required=True)
options, remaining = parser.parse_known_args()
if "--output" in remaining:
    report_path = Path(remaining[remaining.index("--output") + 1])
    if options.copy_snapshot.resolve() == report_path.with_suffix(".npz").resolve():
        raise ValueError("Copy snapshot must differ from the public state snapshot")
sys.argv = [sys.argv[0], *remaining]
original_init = SolverPhoenX.__init__
captured = {}


def traced_init(self, model, *args, **kwargs):
    """Insert read-only copies before the example captures its simulation graph."""
    original_init(self, model, *args, **kwargs)
    world = self.world
    state = world._copy_state
    buffers = {
        "velocity": wp.empty_like(state.velocity),
        "angular_velocity": wp.empty_like(state.angular_velocity),
    }
    original_average = world._mass_splitting_average_and_broadcast

    def traced_average(inv_dt):
        wp.copy(buffers["velocity"], state.velocity)
        wp.copy(buffers["angular_velocity"], state.angular_velocity)
        original_average(inv_dt)

    world._mass_splitting_average_and_broadcast = traced_average
    captured.update(buffers=buffers, state=state)


SolverPhoenX.__init__ = traced_init
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    if captured:
        state = captured["state"]
        np.savez_compressed(
            options.copy_snapshot,
            **{name: array.numpy() for name, array in captured["buffers"].items()},
            section_end=state.section_end.numpy(),
            partition_list=state.partition_list.numpy(),
            count_per_node=state.count_per_node.numpy(),
            highest_index_in_use=state.highest_index_in_use.numpy(),
        )
