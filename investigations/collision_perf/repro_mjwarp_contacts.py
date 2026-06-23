# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from typing import Any

import numpy as np
import warp as wp

import newton


def _nvidia_smi_used_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    first_line = result.stdout.strip().splitlines()[0].strip()
    try:
        return int(first_line)
    except ValueError:
        return None


def _mib(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / (1024.0 * 1024.0)


def _memory_snapshot(device: wp.context.Device) -> dict[str, float | int | None]:
    wp.synchronize_device(device)
    return {
        "mempool_current_mib": _mib(wp.get_mempool_used_mem_current(device)),
        "mempool_high_mib": _mib(wp.get_mempool_used_mem_high(device)),
        "device_free_mib": _mib(device.free_memory),
        "nvidia_smi_used_mib": _nvidia_smi_used_mib(),
    }


def _box_mesh() -> newton.Mesh:
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices, faces.flatten())


def _add_box_shape(
    builder: newton.ModelBuilder,
    *,
    body: int,
    xform: wp.transform,
    half_extents: tuple[float, float, float],
    cfg: newton.ModelBuilder.ShapeConfig,
    shape_kind: str,
    mesh: newton.Mesh | None,
    label: str,
) -> int:
    if shape_kind == "mesh":
        if mesh is None:
            raise ValueError("mesh shape kind requires a mesh")
        return builder.add_shape_mesh(
            body,
            xform=xform,
            mesh=mesh,
            scale=(2.0 * half_extents[0], 2.0 * half_extents[1], 2.0 * half_extents[2]),
            cfg=cfg,
            label=label,
        )

    return builder.add_shape_box(
        body,
        xform=xform,
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=cfg,
        label=label,
    )


def _make_one_world_builder(args: argparse.Namespace) -> newton.ModelBuilder:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = args.contact_margin
    builder.default_shape_cfg.gap = args.contact_gap

    shape_cfg = builder.default_shape_cfg.copy()
    mesh = _box_mesh() if args.shape_kind == "mesh" else None

    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        mass=0.2,
        label="object",
    )
    _add_box_shape(
        builder,
        body=body,
        xform=wp.transform_identity(),
        half_extents=(0.03, 0.045, 0.045),
        cfg=shape_cfg,
        shape_kind=args.shape_kind,
        mesh=mesh,
        label="object_box",
    )

    # Static per-world "fingertip" colliders. They are all attached to the
    # world body, so Newton filters static-static pairs and keeps only
    # object-static candidate pairs.
    for i in range(args.static_colliders_per_world):
        angle = 2.0 * math.pi * i / max(args.static_colliders_per_world, 1)
        radius = 0.035 + 0.004 * ((i % 3) - 1)
        z = -0.012 + 0.008 * (i % 4)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        _add_box_shape(
            builder,
            body=-1,
            xform=wp.transform((x, y, z), wp.quat_identity()),
            half_extents=(0.018, 0.018, 0.035),
            cfg=shape_cfg,
            shape_kind=args.shape_kind,
            mesh=mesh,
            label=f"static_contact_{i}",
        )

    return builder


def _make_allegro_builder(args: argparse.Namespace) -> newton.ModelBuilder:
    newton.use_coord_layout_targets = True

    allegro_hand = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(allegro_hand)
    allegro_hand.default_shape_cfg.ke = 1.0e3
    allegro_hand.default_shape_cfg.kd = 1.0e2
    allegro_hand.default_shape_cfg.margin = args.contact_margin
    allegro_hand.default_shape_cfg.gap = args.contact_gap

    asset_path = newton.utils.download_asset("wonik_allegro")
    asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
    allegro_hand.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)),
        enable_self_collisions=False,
        ignore_paths=[".*Dummy", ".*CollisionPlane"],
        hide_collision_shapes=True,
    )

    for i in range(allegro_hand.joint_dof_count - 6):
        allegro_hand.joint_target_ke[i] = 150.0
        allegro_hand.joint_target_kd[i] = 5.0
        allegro_hand.joint_q[i] = 0.3
        allegro_hand.joint_target_q[i] = 0.3
        if allegro_hand.joint_label[i][-2:] == "_0":
            allegro_hand.joint_q[i] = 0.6
            allegro_hand.joint_target_q[i] = 0.6
        allegro_hand.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
        if allegro_hand.joint_type[i] == newton.JointType.REVOLUTE:
            allegro_hand.joint_armature[i] = 1.0e-2

    q = np.array(allegro_hand.joint_q)
    q[-7:-4] += np.array([0.0, 0.0, 0.05])
    q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
    allegro_hand.joint_q = q.tolist()

    builder = newton.ModelBuilder()
    builder.replicate(allegro_hand, args.world_count)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.margin = args.contact_margin
    builder.default_shape_cfg.gap = args.contact_gap
    builder.add_ground_plane()
    return builder


def _make_model(args: argparse.Namespace, device: wp.context.Device) -> newton.Model:
    if args.scene == "allegro":
        builder = _make_allegro_builder(args)
    else:
        one_world = _make_one_world_builder(args)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(one_world, args.world_count)
    return builder.finalize(device=device)


def _shape_pairs_override(model: newton.Model, mode: str) -> int | None:
    if mode == "fixed":
        return None
    if mode == "global":
        return model.shape_count * (model.shape_count - 1) // 2
    raise ValueError(f"Unknown shape pair capacity mode: {mode}")


def _make_contacts(
    args: argparse.Namespace,
    model: newton.Model,
    solver: newton.solvers.SolverMuJoCo,
) -> tuple[newton.Contacts, newton.CollisionPipeline | None]:
    if args.contact_mode == "mujoco":
        return newton.Contacts(solver.get_max_contact_count(), 0, device=model.device), None

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase=args.broad_phase,
        rigid_contact_max=args.contact_budget_per_world * args.world_count,
        max_triangle_pairs=args.max_triangle_pairs,
        shape_pairs_max=_shape_pairs_override(model, args.shape_pair_capacity),
    )
    return model.contacts(collision_pipeline=pipeline), pipeline


def _simulate_full(
    args: argparse.Namespace,
    model: newton.Model,
    solver: newton.solvers.SolverMuJoCo,
    state_0: newton.State,
    state_1: newton.State,
    control: newton.Control,
    contacts: newton.Contacts,
    steps: int,
) -> tuple[newton.State, newton.State]:
    for _ in range(steps):
        if args.contact_mode == "newton":
            model.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, args.dt)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def _time_gpu(device: wp.context.Device, fn) -> float:
    start = time.perf_counter()
    fn()
    wp.synchronize_device(device)
    return time.perf_counter() - start


def _active_contacts(
    args: argparse.Namespace,
    model: newton.Model,
    solver: newton.solvers.SolverMuJoCo,
    state: newton.State,
    contacts: newton.Contacts,
) -> int:
    if args.contact_mode == "newton":
        model.collide(state, contacts)
    else:
        solver.update_contacts(contacts, state)
    return int(contacts.rigid_contact_count.numpy()[0])


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    stages: list[dict[str, Any]] = []

    def add_stage(name: str, **extra: Any) -> None:
        stages.append({"stage": name, **_memory_snapshot(device), **extra})

    add_stage("start")

    model = _make_model(args, device)
    add_stage(
        "model_finalized",
        body_count=model.body_count,
        joint_count=model.joint_count,
        shape_count=model.shape_count,
        shape_contact_pair_count=model.shape_contact_pair_count,
        world_count=model.world_count,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    add_stage("state_control_allocated")

    use_mujoco_contacts = args.contact_mode == "mujoco"
    solver = newton.solvers.SolverMuJoCo(
        model,
        solver="newton",
        integrator="implicitfast",
        njmax=args.njmax,
        nconmax=args.nconmax,
        impratio=10.0,
        cone="elliptic",
        iterations=args.iterations,
        ls_iterations=args.ls_iterations,
        ccd_iterations=args.ccd_iterations,
        update_data_interval=args.update_data_interval,
        use_mujoco_contacts=use_mujoco_contacts,
    )
    add_stage(
        "solver_initialized",
        solver_contact_mode=args.contact_mode,
        solver_max_contact_count=solver.get_max_contact_count(),
        mjw_nconmax=getattr(solver.mjw_data, "nconmax", None),
        mjw_naconmax=getattr(solver.mjw_data, "naconmax", None),
        mjw_njmax=getattr(solver.mjw_data, "njmax", None),
    )

    contacts, pipeline = _make_contacts(args, model, solver)
    pipeline_stats = {}
    if pipeline is not None:
        pipeline_stats = {
            "pipeline_broad_phase": pipeline.broad_phase_mode,
            "pipeline_shape_pairs_max": pipeline.shape_pairs_max,
            "pipeline_rigid_contact_max": pipeline.rigid_contact_max,
            "pipeline_soft_contact_max": pipeline.soft_contact_max,
        }
    add_stage(
        "contacts_allocated",
        contacts_rigid_contact_max=contacts.rigid_contact_max,
        **pipeline_stats,
    )

    state_0, state_1 = _simulate_full(
        args,
        model,
        solver,
        state_0,
        state_1,
        control,
        contacts,
        args.warmup_steps,
    )
    add_stage("warmup_complete")

    full_seconds = _time_gpu(
        device,
        lambda: _simulate_full(args, model, solver, state_0, state_1, control, contacts, args.steps),
    )
    add_stage("timed_full_complete")

    collision_seconds = None
    if args.contact_mode == "newton":
        collision_seconds = _time_gpu(device, lambda: [model.collide(state_0, contacts) for _ in range(args.steps)])
        add_stage("timed_collision_only_complete")

    active_contacts = _active_contacts(args, model, solver, state_0, contacts)
    add_stage("active_contacts_sampled", active_contacts=active_contacts)

    shapes_per_world = model.shape_count // model.world_count if model.world_count else model.shape_count
    global_pair_capacity = model.shape_count * (model.shape_count - 1) // 2
    return {
        "args": vars(args),
        "device": str(device),
        "summary": {
            "world_count": model.world_count,
            "shapes_per_world": shapes_per_world,
            "shape_count": model.shape_count,
            "shape_contact_pair_count": model.shape_contact_pair_count,
            "global_shape_pair_capacity": global_pair_capacity,
            "solver_max_contact_count": solver.get_max_contact_count(),
            "contacts_rigid_contact_max": contacts.rigid_contact_max,
            "active_contacts": active_contacts,
            "full_seconds": full_seconds,
            "full_ms_per_step": 1000.0 * full_seconds / args.steps,
            "collision_seconds": collision_seconds,
            "collision_ms_per_step": None
            if collision_seconds is None
            else 1000.0 * collision_seconds / args.steps,
        },
        "stages": stages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=("synthetic", "allegro"), default="synthetic")
    parser.add_argument("--world-count", type=int, default=256)
    parser.add_argument("--static-colliders-per-world", type=int, default=16)
    parser.add_argument("--shape-kind", choices=("primitive", "mesh"), default="primitive")
    parser.add_argument("--contact-mode", choices=("mujoco", "newton"), default="newton")
    parser.add_argument("--broad-phase", choices=("explicit", "nxn", "sap"), default="explicit")
    parser.add_argument("--shape-pair-capacity", choices=("fixed", "global"), default="fixed")
    parser.add_argument("--contact-budget-per-world", type=int, default=200)
    parser.add_argument("--max-triangle-pairs", type=int, default=1_000_000)
    parser.add_argument("--nconmax", type=int, default=200)
    parser.add_argument("--njmax", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--ls-iterations", type=int, default=15)
    parser.add_argument("--ccd-iterations", type=int, default=35)
    parser.add_argument("--update-data-interval", type=int, default=2)
    parser.add_argument("--contact-margin", type=float, default=0.0)
    parser.add_argument("--contact-gap", type=float, default=0.002)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
