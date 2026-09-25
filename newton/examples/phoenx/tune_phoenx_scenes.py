# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Factories matching the authored PhoenX mechanism examples for tuning."""

import math

import warp as wp

import newton
import newton.utils
from newton import JointTargetMode
from newton.examples.kamino.example_kamino_colibri import build_scene as build_colibri
from newton.examples.phoenx.example_phoenx_bike_transmission import (
    DEFAULT_CADENCE_RPM,
    DEFAULT_STARTUP_RAMP_TIME,
    SPECULATIVE_CONTACT_GAP_MAX,
    _approach_joint_velocity,
    _simulation_schedule,
)
from newton.examples.phoenx.example_phoenx_bike_transmission import (
    build_scene as build_bike,
)
from newton.examples.phoenx.example_phoenx_caterpillar import build_scene as build_caterpillar
from newton.examples.phoenx.example_phoenx_colibri import CONTACT_OFFSETS
from newton.examples.phoenx.tune_phoenx import TuningScene


def make_bike_scene():
    """BikeTransmission at the example's default 60 rpm and 120 Hz contacts."""
    model = build_bike().finalize(skip_validation_joints=True)
    updates, substeps = _simulation_schedule(DEFAULT_CADENCE_RPM, 0, 0)
    labels = list(model.joint_label)
    front_joint = labels.index("/World/Xform/FrontGears/RevoluteJoint")
    front_dof = int(model.joint_qd_start.numpy()[front_joint])
    drive_target = -DEFAULT_CADENCE_RPM * 2.0 * math.pi / 60.0

    def before_update(_state, control, update_index, _dt):
        if update_index % updates == 0:
            wp.launch(
                _approach_joint_velocity,
                dim=1,
                inputs=[
                    control.joint_target_qd,
                    front_dof,
                    drive_target,
                    abs(drive_target) / (60.0 * DEFAULT_STARTUP_RAMP_TIME),
                ],
                device=model.device,
            )

    def pipeline():
        return newton.CollisionPipeline(
            model,
            contact_matching="sticky",
            rigid_contact_max=32768,
            speculative_contact_gap_max=SPECULATIVE_CONTACT_GAP_MAX,
            speculative_contact_velocity_filter=True,
            contact_reduction_voxel_depth=False,
        )

    return TuningScene(
        model,
        frame_dt=1.0 / 60.0,
        collision_updates=updates,
        solver_options={
            "joint_mode": "maximal_direct",
            "step_layout": "single_world",
            "substeps": substeps,
            "solver_iterations": 4,
            "velocity_iterations": 1,
            "direct_joint_projection_passes": 2,
            "parallel_contact_prepare": False,
            "enable_body_pair_grouping": True,
            "contact_chunk_size": 64,
            "mass_splitting": True,
            "mass_splitting_color_group_size": 2,
            "mass_splitting_batch_size": 2,
            "max_colored_partitions": 8,
        },
        pipeline_factory=pipeline,
        before_update=before_update,
        in_place=True,
    )


def make_colibri_scene(num_worlds=4):
    """Driven four-world Colibri, matching the example's temporal solve."""
    builder = build_colibri(
        contact_gap=0.001,
        source_contact_offsets=True,
        mesh_cylinders=True,
        sdf_resolution=0,
        counterweight_density_scale=1.0,
        attach_flower_to_base=True,
        enable_frame_drive=False,
        enable_crank_drive=True,
    )
    for index, label in enumerate(builder.shape_label):
        if label in CONTACT_OFFSETS:
            builder.shape_gap[index] = CONTACT_OFFSETS[label]
    if num_worlds > 1:
        replicated = newton.ModelBuilder()
        replicated.replicate(builder, num_worlds)
        builder = replicated
    model = builder.finalize(skip_validation_joints=True)
    newton.eval_ik(model, model, model.joint_q, model.joint_qd)

    def pipeline():
        return newton.CollisionPipeline(
            model,
            contact_matching="latest",
            rigid_contact_max=8192 * num_worlds,
            speculative_contact_gap_max=0.005,
            speculative_contact_velocity_filter=False,
        )

    return TuningScene(
        model,
        frame_dt=1.0 / 60.0,
        collision_updates=2,
        solver_options={
            "joint_mode": "maximal_pgs",
            "solver_scheme": "tgs",
            "step_layout": "single_world",
            "parallel_contact_prepare": True,
            "contact_chunk_size": 0,
            "mass_splitting": True,
            "mass_splitting_color_group_size": 2,
            "mass_splitting_batch_size": 2,
            "max_colored_partitions": 8,
            "substeps": 24,
            "solver_iterations": 1,
            "velocity_relaxation": "final_substep",
            "prepare_refresh_stride": 1,
            "sor_boost": 1.0,
        },
        pipeline_factory=pipeline,
        in_place=True,
    )


def make_caterpillar_scene():
    """Single Caterpillar with authored hydraulic drives and 120 Hz contacts."""
    model = build_caterpillar().finalize(skip_validation_joints=True)

    def pipeline():
        return newton.CollisionPipeline(
            model,
            contact_matching="latest",
            rigid_contact_max=65536,
            speculative_contact_gap_max=0.01,
            speculative_contact_velocity_filter=False,
        )

    return TuningScene(
        model,
        frame_dt=1.0 / 60.0,
        collision_updates=2,
        solver_options={
            "joint_mode": "maximal_direct",
            "step_layout": "single_world",
            "substeps": 5,
            "solver_iterations": 2,
            "velocity_iterations": 1,
            "parallel_contact_prepare": model.device.is_cuda,
            "contact_chunk_size": 64,
            "mass_splitting": True,
            "mass_splitting_color_group_size": 2,
            "mass_splitting_batch_size": 2,
            "max_colored_partitions": 8,
        },
        pipeline_factory=pipeline,
        in_place=True,
    )


def make_g1_scene(num_worlds=16):
    """Replicate the flat-ground G1 benchmark's 16-world PhoenX workload."""
    g1 = newton.ModelBuilder()
    g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
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

    def initialize_state(state):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    return TuningScene(
        model,
        frame_dt=1.0 / 60.0,
        collision_updates=5,
        solver_options={
            "joint_mode": "maximal_direct",
            "step_layout": "multi_world",
            "substeps": 1,
            "solver_iterations": 2,
            "velocity_iterations": 1,
        },
        initialize_state=initialize_state,
        pipeline_factory=lambda: newton.CollisionPipeline(model, contact_matching="sticky"),
    )
