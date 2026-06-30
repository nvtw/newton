# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark reduced PhoenX on realistic articulated robots.

Examples::

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_reduced_articulation_robots \
        --robots anymal h1 g1 --robot-counts 1 64 512 --layouts multi_world

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_reduced_articulation_robots \
        --robots anymal g1 --robot-counts 64 --layouts single_world multi_world --contacts
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import warp as wp

import newton
import newton.utils
from newton._src.solvers.phoenx.benchmarks.runner import SceneHandle, run_one


def _robot_builder(name: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3,
        limit_kd=1.0e1,
        friction=1.0e-5,
    )
    builder.default_shape_cfg.ke = 2.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    if name == "anymal":
        asset = newton.utils.download_asset("anybotics_anymal_d")
        builder.add_usd(
            str(asset / "usd" / "anymal_d.usda"),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        builder.joint_q[:3] = [0.0, 0.0, 0.68]
    elif name == "h1":
        asset = newton.utils.download_asset("unitree_h1")
        builder.add_usd(
            str(asset / "usd_structured" / "h1.usda"),
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
        )
        builder.approximate_meshes("bounding_box")
    elif name == "g1":
        asset = newton.utils.download_asset("unitree_g1")
        builder.add_usd(
            str(asset / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.2)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        builder.approximate_meshes("bounding_box")
    else:
        raise ValueError(f"unknown robot {name!r}")

    for dof in range(builder.joint_dof_count):
        builder.joint_target_ke[dof] = 150.0
        builder.joint_target_kd[dof] = 5.0
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
    return builder


def _build_model(
    robot: str,
    robot_count: int,
    layout: str,
    contacts: bool,
    device: wp.context.Device,
) -> newton.Model:
    template = _robot_builder(robot)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    if layout == "multi_world":
        builder.replicate(template, world_count=robot_count)
    elif layout == "single_world":
        width = max(1, int(math.ceil(math.sqrt(robot_count))))
        for robot_index in range(robot_count):
            x = 3.0 * float(robot_index % width)
            y = 3.0 * float(robot_index // width)
            shape_start = builder.shape_count
            builder.add_builder(template, xform=wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity()))
            for shape in range(shape_start, builder.shape_count):
                builder.shape_collision_group[shape] = robot_index + 1
    else:
        raise ValueError(f"unknown layout {layout!r}")
    if contacts:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_handle(
    robot: str,
    robot_count: int,
    layout: str,
    solver_name: str,
    contacts_enabled: bool,
    substeps: int,
    solver_iterations: int,
    velocity_iterations: int,
    device: wp.context.Device,
) -> tuple[SceneHandle, list[newton.State]]:
    model = _build_model(robot, robot_count, layout, contacts_enabled, device)
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    if solver_name == "reduced":
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_iterations=velocity_iterations,
        )
        outer_steps = 1
    elif solver_name == "classic":
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_iterations=velocity_iterations,
        )
        outer_steps = 1
    elif solver_name == "featherstone":
        solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
        outer_steps = substeps
    else:
        raise ValueError(f"unknown solver {solver_name!r}")

    contacts = model.contacts() if contacts_enabled else None
    state_box = [state0, state1]
    frame_dt = 1.0 / 200.0

    def simulate_one_frame() -> None:
        if contacts is not None:
            model.collide(state_box[0], contacts)
        call_dt = frame_dt if outer_steps == 1 else frame_dt / outer_steps
        for _ in range(outer_steps):
            state_box[0].clear_forces()
            solver.step(state_box[0], state_box[1], control, contacts, call_dt)
            state_box[0], state_box[1] = state_box[1], state_box[0]

    return (
        SceneHandle(
            name=f"{robot}_{layout}_{'contacts' if contacts_enabled else 'free'}",
            solver_name=solver_name,
            num_worlds=robot_count,
            substeps=substeps,
            solver_iterations=solver_iterations,
            simulate_one_frame=simulate_one_frame,
        ),
        state_box,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", nargs="+", choices=("anymal", "h1", "g1"), default=("anymal", "h1", "g1"))
    parser.add_argument("--robot-counts", nargs="+", type=int, default=(1, 64, 512))
    parser.add_argument("--layouts", nargs="+", choices=("single_world", "multi_world"), default=("multi_world",))
    parser.add_argument(
        "--solvers", nargs="+", choices=("reduced", "classic", "featherstone"), default=("reduced", "featherstone")
    )
    parser.add_argument("--contacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--measure-frames", type=int, default=64)
    args = parser.parse_args()

    device = wp.get_preferred_device()
    if not device.is_cuda:
        raise RuntimeError("real-robot reduced-articulation benchmarks require CUDA graph capture")

    for robot in args.robots:
        for layout in args.layouts:
            for robot_count in args.robot_counts:
                for solver_name in args.solvers:
                    handle, state_box = _build_handle(
                        robot,
                        robot_count,
                        layout,
                        solver_name,
                        args.contacts,
                        args.substeps,
                        args.solver_iterations,
                        args.velocity_iterations,
                        device,
                    )
                    result = run_one(
                        handle,
                        warmup_frames=args.warmup_frames,
                        measure_frames=args.measure_frames,
                    )
                    result["robot_count"] = robot_count
                    result["layout"] = layout
                    joint_q = state_box[0].joint_q.numpy()
                    joint_qd = state_box[0].joint_qd.numpy()
                    if not np.isfinite(joint_q).all() or not np.isfinite(joint_qd).all():
                        raise RuntimeError(f"{handle.name}/{solver_name} produced non-finite articulation state")
                    result["max_abs_joint_q"] = float(np.abs(joint_q).max(initial=0.0))
                    result["max_abs_joint_qd"] = float(np.abs(joint_qd).max(initial=0.0))
                    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
