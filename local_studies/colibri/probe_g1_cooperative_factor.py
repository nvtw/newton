# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay actual G1 factor inputs through depth and cooperative kernels."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations import reduced
from newton.examples.robot.example_robot_policy import Example
from newton.viewer import ViewerNull

reduced._WARP_FACTOR_MIN_ARTICULATIONS = 1
wp.init()
example = Example(ViewerNull(), Example.create_parser().parse_args(["--robot", "g1_29dof", "--solver", "phoenx"]))
backend = example.solver._reduced_articulation
system = backend.system
split = system._use_factor_dof_split
fields = (
    "joint_factor_diagonal",
    "joint_implicit_force",
    "joint_qd_internal",
    "body_i_s",
    "joint_s",
    "reduced_inertia",
    "joint_u_matrix",
    "joint_d_inv",
)
results = []
for step in range(3):
    if step:
        example.step()
    outputs = []
    for cooperative in (False, True):
        system.use_warp_factor = cooperative
        system._use_factor_dof_split = cooperative and split
        system.factor(backend.state, backend.control, example.solver.world.substep_dt)
        outputs.append({field: getattr(system, field).numpy().copy() for field in fields})
    differences = {}
    for field in fields:
        a, b = outputs[0][field], outputs[1][field]
        differences[field] = {
            "exact": a.tobytes() == b.tobytes(),
            "max_abs": float(np.max(np.abs(a - b))),
            "finite": bool(np.isfinite(b).all()),
        }
    results.append({"step": step, "fields": differences})
Path("/tmp/g1_cooperative_factor_frozen.json").write_text(
    json.dumps({"split_roots": split, "articulations": system.model.articulation_count, "cases": results}, indent=2)
)
print(json.dumps(results))
