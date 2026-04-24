# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G1 humanoid on a flat ground plane -- benchmark scenario.

Headless, parameter-swappable clone of
``newton.examples.robot_g1.Example`` stripped of the viewer / test
hooks. The body-building code is identical to the example so the
two workloads stay comparable; only the solver / substep / iteration
parameters are exposed for sweeps.
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
    """Build a G1-on-ground scene with ``num_worlds`` replicated
    humanoids driven by ``solver_name`` ('phoenx' or 'mujoco')."""
    device = wp.get_device()
    mem_before = _gpu_used_bytes()

    g1 = newton.ModelBuilder()
    if solver_name == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
    g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    g1.default_shape_cfg.ke = 1.0e3
    g1.default_shape_cfg.kd = 2.0e2
    g1.default_shape_cfg.kf = 1.0e3
    g1.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("unitree_g1")
    g1.add_usd(
        str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0.2)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        skip_mesh_approximation=True,
    )
    for i in range(6, g1.joint_dof_count):
        g1.joint_target_ke[i] = 500.0
        g1.joint_target_kd[i] = 10.0
        g1.joint_target_mode[i] = int(JointTargetMode.POSITION)
    g1.approximate_meshes("bounding_box")

    builder = newton.ModelBuilder()
    builder.replicate(g1, num_worlds)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()

    model = builder.finalize()

    fps = 60
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
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=150,
            cone="elliptic",
            impratio=100,
            iterations=solver_iterations,
            ls_iterations=50,
        )
    else:
        raise ValueError(f"unknown solver '{solver_name}'")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = model.contacts()

    # Lambda-closure over mutable state box so ``simulate_one_frame``
    # can swap ``state_0`` / ``state_1`` in place across substeps.
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
        name="g1_flat",
        solver_name=solver_name,
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
