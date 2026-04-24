# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""H1 humanoid on a flat ground plane -- benchmark scenario.

Headless, parameter-swappable clone of
``newton.examples.robot_h1.Example``.
"""

from __future__ import annotations

import warp as wp

import newton
import newton.utils
from newton import JointTargetMode
from newton._src.solvers.phoenx.benchmarks.runner import (
    SceneHandle,
    _gpu_used_bytes,
)


def build(
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    velocity_iterations: int = 1,
) -> SceneHandle:
    """Build an H1-on-ground scene with ``num_worlds`` replicated
    humanoids driven by ``solver_name`` ('phoenx' or 'mujoco')."""
    device = wp.get_device()
    mem_before = _gpu_used_bytes()

    h1 = newton.ModelBuilder()
    if solver_name == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(h1)
    h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    h1.default_shape_cfg.ke = 2.0e3
    h1.default_shape_cfg.kd = 1.0e2
    h1.default_shape_cfg.kf = 1.0e3
    h1.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("unitree_h1")
    asset_file = str(asset_path / "usd_structured" / "h1.usda")
    h1.add_usd(
        asset_file,
        ignore_paths=["/GroundPlane"],
        enable_self_collisions=False,
    )
    h1.approximate_meshes("bounding_box")
    for i in range(len(h1.joint_target_ke)):
        h1.joint_target_ke[i] = 150
        h1.joint_target_kd[i] = 5
        h1.joint_target_mode[i] = int(JointTargetMode.POSITION)

    builder = newton.ModelBuilder()
    builder.replicate(h1, num_worlds)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()

    model = builder.finalize()

    fps = 50
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / substeps

    if solver_name == "phoenx":
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_iterations=velocity_iterations,
        )
    elif solver_name == "mujoco":
        solver = newton.solvers.SolverMuJoCo(
            model,
            iterations=solver_iterations,
            ls_iterations=50,
            njmax=100,
            nconmax=210,
        )
    else:
        raise ValueError(f"unknown solver '{solver_name}'")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = model.contacts()

    box = {"state_0": state_0, "state_1": state_1}

    def simulate_one_frame() -> None:
        model.collide(box["state_0"], contacts)
        for _ in range(substeps):
            box["state_0"].clear_forces()
            solver.step(box["state_0"], box["state_1"], control, contacts, sim_dt)
            box["state_0"], box["state_1"] = box["state_1"], box["state_0"]

    wp.synchronize_device()
    setup_bytes = max(0, _gpu_used_bytes() - mem_before)

    return SceneHandle(
        name="h1_flat",
        solver_name=solver_name,
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
