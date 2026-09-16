# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Actual FP32 scalar-row callback: closed two-rotor implicit drive gate."""

from .colored_d6_scalar import install

install()

import json  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402


def main():
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    links = []
    for inertia in (0.02, 0.01):
        links.append(builder.add_link(mass=1.0, inertia=wp.mat33(inertia, 0, 0, 0, inertia, 0, 0, 0, inertia)))
    joint = builder.add_joint_revolute(
        parent=links[0],
        child=links[1],
        axis=(0, 0, 1),
        target_ke=0.572957795,
        target_kd=0.572957795,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        effort_limit=0.0,
        armature=0.0,
        damping=0.0,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device="cuda:0")
    solver = newton.solvers.SolverPhoenX(
        model,
        joint_solver="block_pgs",
        articulation_mode="maximal",
        step_layout="single_world",
        substeps=1,
        solver_iterations=1,
        velocity_iterations=0,
        sor_boost=1.0,
        mass_splitting=False,
    )
    state = model.state()
    control = model.control()
    control.joint_target_q.assign(np.asarray([0.3], np.float32))
    dt = 1.0 / 3600
    state.clear_forces()
    solver.step(state, state, control, None, dt)
    print(
        "ROWS",
        solver.world.constraints.bilateral.row_count.numpy(),
        "DYNAMIC",
        solver.world.constraints.bilateral.row_dynamic.numpy(),
        "REFERENCE",
        solver.world.constraints.bilateral.reference.numpy(),
        flush=True,
    )
    qd = state.body_qd.numpy()
    actual = float(qd[1, 5] - qd[0, 5])
    inverse = 1.0 / 0.02 + 1.0 / 0.01
    expected = dt * 0.572957795 * inverse * 0.3 / (1 + dt * 0.572957795 * inverse + dt * dt * 0.572957795 * inverse)
    momentum = 0.02 * float(qd[0, 5]) + 0.01 * float(qd[1, 5])
    result = {"actual": actual, "expected": expected, "error": abs(actual - expected), "angular_momentum": momentum}
    print(json.dumps(result, indent=2))
    assert abs(actual - expected) < 2e-7
    assert abs(momentum) < 1e-9


if __name__ == "__main__":
    main()
