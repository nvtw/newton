# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure heavy-light-heavy joint convergence against an analytical projection."""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from local_studies.colibri.bilateral_pgs import install_fused


def run(fused, iterations):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    masses = np.array([1000.0, 1.0, 1000.0])
    bodies = []
    for mass in masses:
        bodies.append(
            builder.add_link(
                mass=float(mass), inertia=wp.mat33(float(mass), 0.0, 0.0, 0.0, float(mass), 0.0, 0.0, 0.0, float(mass))
            )
        )
    builder.add_articulation(
        [
            builder.add_joint_free(bodies[0]),
            builder.add_joint_fixed(bodies[0], bodies[1]),
            builder.add_joint_fixed(bodies[1], bodies[2]),
        ]
    )
    model = builder.finalize(device="cuda:0")
    solver = newton.solvers.SolverPhoenX(
        model,
        articulation_mode="maximal",
        step_layout="single_world",
        substeps=1,
        solver_iterations=iterations,
        sor_boost=1.0,
        prepare_refresh_stride=1,
    )
    if fused:
        install_fused(solver)
    state = model.state()
    qd = state.body_qd.numpy()
    qd[:] = 0.0
    qd[bodies[-1], 0] = 1.0
    state.body_qd.assign(qd)
    state.clear_forces()
    solver.step(state, state, model.control(), None, 0.001)
    after = state.body_qd.numpy()[bodies, 0].astype(np.float64)
    expected = masses[-1] / masses.sum()
    return {
        "route": "fused_bilateral" if fused else "direct",
        "iterations": iterations,
        "mass_kg": masses.tolist(),
        "velocity_x_m_s": after.tolist(),
        "expected_common_velocity_m_s": float(expected),
        "max_velocity_error_m_s": float(np.max(np.abs(after - expected))),
        "linear_momentum_error_kg_m_s": float(np.dot(masses, after) - masses[-1]),
        "kinetic_energy_before_j": float(0.5 * masses[-1]),
        "kinetic_energy_after_j": float(0.5 * np.dot(masses, after * after)),
    }


if __name__ == "__main__":
    rows = [run(False, 1)] + [run(True, count) for count in (1, 8, 64)]
    output = Path("/tmp/colibri_bilateral_heavy_light_heavy.json")
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)
