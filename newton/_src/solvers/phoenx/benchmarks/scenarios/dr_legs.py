# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Disney Research DR Legs on a flat ground plane.

Headless benchmark clone of ``newton.examples.robot.example_robot_dr_legs_phoenx``.
The scene exercises a mixed articulated workload: driven hinges, passive loop
closures, and foot contacts.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.utils
from newton._src.solvers.phoenx.benchmarks.runner import SceneHandle, _gpu_used_bytes
from newton.examples.robot.example_robot_dr_legs_phoenx import (
    _ANIMATION_CHANNEL_SIGN,
    _ANIMATION_JOINT_PATHS,
    _FLIPPED_JOINTS,
    _LOOP_CLOSER_JOINTS,
    _advance_sim_time,
    _flip_joint,
    _scatter_animation_targets,
)


def _init_animation(
    model: newton.Model, num_worlds: int, asset_path
) -> tuple[int, float, wp.array2d[float], wp.array2d[int]]:
    anim_file = str(asset_path / "dr_legs/animation" / "dr_legs_animation_100fps.npy")
    anim = np.load(anim_file).astype(np.float32)
    if anim.shape[1] != len(_ANIMATION_JOINT_PATHS):
        raise RuntimeError(f"animation has {anim.shape[1]} channels, expected {len(_ANIMATION_JOINT_PATHS)}")

    joint_label = list(model.joint_label)
    joint_qd_start = model.joint_qd_start.numpy()
    try:
        channel_dofs = np.array(
            [joint_qd_start[joint_label.index(path)] for path in _ANIMATION_JOINT_PATHS],
            dtype=np.int32,
        )
    except ValueError as e:
        raise RuntimeError(f"animation joint not found in model.joint_label: {e}") from e

    n_dof_per_world = model.joint_dof_count // num_worlds
    world_offsets = np.arange(num_worlds, dtype=np.int32) * n_dof_per_world
    animation_indices = channel_dofs[None, :] + world_offsets[:, None]
    animation_data = wp.array(anim * _ANIMATION_CHANNEL_SIGN[None, :], dtype=float)
    return anim.shape[0], 100.0, animation_data, wp.array(animation_indices, dtype=int)


def build(
    num_worlds: int,
    solver_name: str,
    substeps: int,
    solver_iterations: int,
    velocity_iterations: int = 1,
    *,
    fps: int = 60,
    animation: bool = True,
    animation_speed: float = 1.0,
    armature: float = 0.001,
    step_layout: str = "multi_world",
    prepare_refresh_stride: int | str = "auto",
) -> SceneHandle:
    """Build a DR-Legs-on-ground PhoenX scene."""
    if solver_name != "phoenx":
        raise ValueError("dr_legs currently supports solver_name='phoenx' only")

    from pxr import Sdf, Usd, UsdPhysics

    mem_before = _gpu_used_bytes()
    frame_dt = 1.0 / float(fps)

    dr_legs = newton.ModelBuilder(up_axis=newton.Axis.Z)
    dr_legs.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3,
        limit_kd=1.0e1,
        friction=1e-5,
        armature=armature,
    )
    dr_legs.default_shape_cfg.ke = 2.0e3
    dr_legs.default_shape_cfg.kd = 1.0e2
    dr_legs.default_shape_cfg.kf = 1.0e3
    dr_legs.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("disneyresearch")
    asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")
    stage = Usd.Stage.Open(asset_file)
    if stage is None:
        raise RuntimeError(f"Failed to open dr_legs USD stage: {asset_file}")
    # Usd.Stage.Open may return a cached mutable stage. Reload before applying
    # the benchmark's joint edits so repeated builds do not flip joints twice.
    stage.Reload()

    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/DR_Legs/RigidBodies/pelvis"))
    for joint_path in _FLIPPED_JOINTS:
        _flip_joint(stage, joint_path)
    for joint_path in _LOOP_CLOSER_JOINTS:
        stage.GetPrimAtPath(joint_path).CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(
            True
        )

    dr_legs.add_usd(
        stage,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.265)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(dr_legs, num_worlds)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()

    model = builder.finalize()
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=velocity_iterations,
        step_layout=step_layout,
        prepare_refresh_stride=prepare_refresh_stride,
    )

    state_0 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = model.contacts()

    sim_time_wp = wp.array([0.0], dtype=float)
    animation_n_frames = 0
    animation_fps = 100.0
    animation_data_wp = wp.zeros((1, len(_ANIMATION_JOINT_PATHS)), dtype=float)
    animation_indices_wp = wp.zeros((max(1, num_worlds), len(_ANIMATION_JOINT_PATHS)), dtype=int)
    if animation:
        animation_n_frames, animation_fps, animation_data_wp, animation_indices_wp = _init_animation(
            model, num_worlds, asset_path
        )

    def simulate_one_frame() -> None:
        if animation:
            wp.launch(
                _scatter_animation_targets,
                dim=(num_worlds, len(_ANIMATION_JOINT_PATHS)),
                inputs=[
                    sim_time_wp,
                    wp.float32(animation_speed),
                    wp.float32(animation_fps),
                    wp.int32(animation_n_frames),
                    animation_data_wp,
                    animation_indices_wp,
                    control.joint_target_q,
                ],
            )
        model.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_0, control, contacts, frame_dt)
        if animation:
            wp.launch(_advance_sim_time, dim=1, inputs=[sim_time_wp, wp.float32(frame_dt)])

    wp.synchronize_device()
    setup_bytes = max(0, _gpu_used_bytes() - mem_before)

    return SceneHandle(
        name="dr_legs",
        solver_name=solver_name,
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        simulate_one_frame=simulate_one_frame,
        setup_bytes=setup_bytes,
    )
