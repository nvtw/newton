# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph tests for PhoenX reduced-coordinate articulations."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.reduced import (
    ReducedArticulationSystem,
    _flush_deferred_articulation,
)
from newton._src.solvers.phoenx.articulations.reduced_contact import (
    _apply_deferred_impulse,
    _deferred_link_delta_twist,
)
from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _CACHED_PAGE_COUNT,
    _POINTS_PER_PAGE,
    _RESPONSE_TILE,
    _RESPONSE_TILES_PER_ARTICULATION,
    _build_generalized_contact_rows_kernel,
    _build_packed_generalized_contact_rows_kernel,
    _transpose_generalized_contact_response_kernel,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.contact_endpoint import _articulation_pair_wrench_response


def _make_mixed_tree_builder():
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=2.0)
    slider = builder.add_link(mass=1.5)
    ball = builder.add_link(mass=1.0)
    builder.add_shape_box(root, hx=0.25, hy=0.15, hz=0.1)
    builder.add_shape_box(slider, hx=0.2, hy=0.12, hz=0.1)
    builder.add_shape_box(ball, hx=0.15, hy=0.1, hz=0.08)
    root_joint = builder.add_joint_revolute(
        parent=-1,
        child=root,
        axis=newton.Axis.Z,
        child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()),
    )
    slider_joint = builder.add_joint_prismatic(
        parent=root,
        child=slider,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    ball_joint = builder.add_joint_ball(
        parent=slider,
        child=ball,
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root_joint, slider_joint, ball_joint])
    return builder


def _make_mixed_tree(device):
    return _make_mixed_tree_builder().finalize(device=device)


def _make_branched_tree(device):
    builder = newton.ModelBuilder(gravity=-9.2, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=3.0)
    left = builder.add_link(mass=1.5)
    left_tip = builder.add_link(mass=0.8)
    right = builder.add_link(mass=1.2)
    right_tip = builder.add_link(mass=0.6)
    for body, half_extent in (
        (root, 0.24),
        (left, 0.18),
        (left_tip, 0.12),
        (right, 0.16),
        (right_tip, 0.1),
    ):
        builder.add_shape_box(body, hx=half_extent, hy=0.08, hz=0.06)

    root_joint = builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)
    left_joint = builder.add_joint_revolute(
        parent=root,
        child=left,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.3, 0.15, 0.0), wp.quat_identity()),
    )
    left_tip_joint = builder.add_joint_ball(
        parent=left,
        child=left_tip,
        parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
    )
    right_joint = builder.add_joint_prismatic(
        parent=root,
        child=right,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(-0.25, -0.15, 0.0), wp.quat_identity()),
    )
    right_tip_joint = builder.add_joint_revolute(
        parent=right,
        child=right_tip,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root_joint, left_joint, left_tip_joint, right_joint, right_tip_joint])
    return builder.finalize(device=device)


def _make_wide_tree(device, child_count):
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0, collision_group=0)
    builder = newton.ModelBuilder(gravity=-9.2, up_axis=newton.Axis.Z)
    root = builder.add_link()
    builder.add_shape_box(root, hx=0.12, hy=0.1, hz=0.08, cfg=cfg)
    joints = [builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)]
    for child_index in range(child_count):
        child = builder.add_link()
        builder.add_shape_box(child, hx=0.08, hy=0.04, hz=0.04, cfg=cfg)
        angle = 2.0 * np.pi * child_index / child_count
        joints.append(
            builder.add_joint_revolute(
                parent=root,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(
                    wp.vec3(0.2 * np.cos(angle), 0.2 * np.sin(angle), 0.0),
                    wp.quat_identity(),
                ),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _make_four_bar_builder():
    crank_length = 0.5
    coupler_length = 1.0
    angle = np.pi / 3.0
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0)
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    crank = builder.add_link()
    coupler = builder.add_link()
    rocker = builder.add_link()
    builder.add_shape_box(crank, hx=0.5 * crank_length, hy=0.03, hz=0.03, cfg=cfg)
    builder.add_shape_box(coupler, hx=0.5 * coupler_length, hy=0.03, hz=0.03, cfg=cfg)
    builder.add_shape_box(rocker, hx=0.5 * crank_length, hy=0.03, hz=0.03, cfg=cfg)
    joint0 = builder.add_joint_revolute(
        parent=-1,
        child=crank,
        axis=newton.Axis.Z,
        child_xform=wp.transform(wp.vec3(-0.5 * crank_length, 0.0, 0.0), wp.quat_identity()),
    )
    joint1 = builder.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.5 * crank_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * coupler_length, 0.0, 0.0), wp.quat_identity()),
    )
    joint2 = builder.add_joint_revolute(
        parent=coupler,
        child=rocker,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.5 * coupler_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5 * crank_length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint0, joint1, joint2])
    loop_joint = builder.add_joint_revolute(
        parent=-1,
        child=rocker,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(coupler_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * crank_length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.joint_articulation[loop_joint] = -1
    builder.joint_q[joint0] = angle
    builder.joint_q[joint1] = -angle
    builder.joint_q[joint2] = angle
    return builder, loop_joint


def _make_four_bar_loop(device):
    builder, loop_joint = _make_four_bar_builder()
    return builder.finalize(device=device), loop_joint


def _make_contact_four_bar_loop(device):
    builder, loop_joint = _make_four_bar_builder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(collision_group=-1, mu=0.6, restitution=0.0))
    return builder.finalize(device=device), loop_joint


def _make_four_bar_fleet(device, world_count):
    blueprint, _loop_joint = _make_four_bar_builder()
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    builder.replicate(blueprint, world_count=world_count)
    model = builder.finalize(device=device)
    loop_joints = np.flatnonzero(model.joint_articulation.numpy() < 0)
    return model, loop_joints


def _make_four_bar_clusters(device, *, worlds, articulations_per_world, loop_copies=1):
    blueprint, loop_joint = _make_four_bar_builder()
    for _copy in range(1, loop_copies):
        duplicate = blueprint.add_joint_revolute(
            parent=blueprint.joint_parent[loop_joint],
            child=blueprint.joint_child[loop_joint],
            axis=newton.Axis.Z,
            parent_xform=blueprint.joint_X_p[loop_joint],
            child_xform=blueprint.joint_X_c[loop_joint],
        )
        blueprint.joint_articulation[duplicate] = -1
    cluster = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    for articulation in range(articulations_per_world):
        cluster.add_builder(
            blueprint,
            xform=wp.transform(wp.vec3(0.0, 2.0 * articulation, 0.0), wp.quat_identity()),
        )
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    if worlds == 1:
        builder.add_builder(cluster)
    else:
        builder.replicate(cluster, world_count=worlds)
    model = builder.finalize(device=device)
    loop_joints = np.flatnonzero(model.joint_articulation.numpy() < 0)
    return model, loop_joints


def _make_floating_tree_builder():
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=2.0)
    child = builder.add_link(mass=1.0)
    builder.add_shape_box(root, hx=0.25, hy=0.15, hz=0.1)
    builder.add_shape_box(child, hx=0.2, hy=0.1, hz=0.08)
    free_joint = builder.add_joint_free(parent=-1, child=root)
    hinge_joint = builder.add_joint_revolute(
        parent=root,
        child=child,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, hinge_joint])
    return builder


def _make_floating_tree(device):
    return _make_floating_tree_builder().finalize(device=device)


def _make_translated_floating_pair(device, translation=4096.0):
    blueprint = _make_floating_tree_builder()
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    builder.add_builder(blueprint)
    builder.add_builder(
        blueprint,
        xform=wp.transform(wp.vec3(translation, -0.5 * translation, 0.25 * translation), wp.quat_identity()),
    )
    return builder.finalize(device=device)


def _make_descendant_free_distance(device, joint_type, *, parent_kinematic=True):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    base = builder.add_link(is_kinematic=parent_kinematic, mass=1.0)
    child = builder.add_link(mass=1.0)
    builder.add_shape_sphere(base, radius=0.1)
    builder.add_shape_sphere(child, radius=0.1)
    builder.body_com[child] = wp.vec3(0.25, 0.11, -0.17)
    root_rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, -0.2, 1.0)), 0.55)
    if parent_kinematic:
        root = builder.add_joint_fixed(
            parent=-1,
            child=base,
            parent_xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), root_rotation),
        )
    else:
        root = builder.add_joint_revolute(
            parent=-1,
            child=base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), root_rotation),
        )
    parent_xform = wp.transform(
        wp.vec3(0.7, -0.2, 0.4),
        wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, 1.0, -0.3)), 0.7),
    )
    child_xform = wp.transform(
        wp.vec3(0.15, -0.05, 0.2),
        wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -0.2, 0.4)), -0.9),
    )
    if joint_type == newton.JointType.FREE:
        descendant = builder.add_joint_free(
            parent=base,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    else:
        descendant = builder.add_joint_distance(
            parent=base,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            min_distance=-1.0,
            max_distance=-1.0,
        )
    builder.add_articulation([root, descendant])
    return builder.finalize(device=device), base, child, root, descendant


def _make_floating_triangle_loop(device):
    length = 1.0
    angle = 2.0 * np.pi / 3.0
    cfg = newton.ModelBuilder.ShapeConfig(density=200.0)
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    body0 = builder.add_link()
    body1 = builder.add_link()
    body2 = builder.add_link()
    for body in (body0, body1, body2):
        builder.add_shape_box(body, hx=0.5 * length, hy=0.03, hz=0.03, cfg=cfg)
    joint0 = builder.add_joint_free(parent=-1, child=body0)
    joint1 = builder.add_joint_revolute(
        parent=body0,
        child=body1,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    joint2 = builder.add_joint_revolute(
        parent=body1,
        child=body2,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint0, joint1, joint2])
    loop_joint = builder.add_joint_ball(
        parent=body0,
        child=body2,
        parent_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.joint_articulation[loop_joint] = -1
    builder.joint_q[:3] = [0.5 * length, 0.0, 0.0]
    builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
    builder.joint_q[7] = angle
    builder.joint_q[8] = angle
    return builder.finalize(device=device), loop_joint


def _make_free_body_loop_builder(joint_type):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    body = builder.add_link()
    builder.add_shape_box(body, hx=0.2, hy=0.15, hz=0.1)
    tree_joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([tree_joint])
    if joint_type == newton.JointType.BALL:
        loop_joint = builder.add_joint_ball(parent=-1, child=body)
    elif joint_type == newton.JointType.PRISMATIC:
        loop_joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.X)
    else:
        loop_joint = builder.add_joint_fixed(parent=-1, child=body)
    builder.joint_articulation[loop_joint] = -1
    return builder


def _make_free_body_loop(device, joint_type):
    return _make_free_body_loop_builder(joint_type).finalize(device=device)


def _make_free_body_loop_cluster(device, joint_type, articulation_count):
    blueprint = _make_free_body_loop_builder(joint_type)
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    for articulation in range(articulation_count):
        builder.add_builder(
            blueprint,
            xform=wp.transform(wp.vec3(0.0, float(articulation), 0.0), wp.quat_identity()),
        )
    return builder.finalize(device=device)


def _make_contact_momentum_model(device, *, with_cloth=False, cloth_contact=False):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=1.0)
    child = builder.add_link(mass=0.5)
    collider = builder.add_link(mass=1.0)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    builder.add_shape_sphere(root, radius=0.2, cfg=shape_cfg)
    builder.add_shape_sphere(collider, radius=0.2, cfg=shape_cfg)
    free_joint = builder.add_joint_free(parent=-1, child=root)
    hinge_joint = builder.add_joint_revolute(
        parent=root,
        child=child,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, hinge_joint])
    if with_cloth:
        cloth_position = wp.vec3(-0.3, -0.1, 0.19) if cloth_contact else wp.vec3(10.0, 0.0, 0.0)
        builder.add_cloth_grid(
            pos=cloth_position,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.05,
            fix_left=True,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            particle_radius=0.01,
        )
    return builder.finalize(device=device), collider


def _make_dense_articulation_contact_chain(device, articulation_count=48):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    for index in range(articulation_count):
        body = builder.add_link(mass=1.0)
        builder.add_shape_sphere(body, radius=0.21, cfg=shape_cfg)
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.joint_q[-7:] = [0.40 * index, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return builder.finalize(device=device)


def _make_high_degree_articulation_star(device, satellite_count=72):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.0, restitution=0.0)
    center = builder.add_link(mass=2.0)
    builder.add_shape_sphere(center, radius=1.0, cfg=shape_cfg)
    center_joint = builder.add_joint_free(parent=-1, child=center)
    builder.add_articulation([center_joint])
    builder.joint_q[-7:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    for index in range(satellite_count):
        angle = 2.0 * np.pi * float(index) / float(satellite_count)
        position = 1.039 * np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        body = builder.add_link(mass=0.1)
        builder.add_shape_sphere(body, radius=0.04, cfg=shape_cfg)
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.joint_q[-7:] = [*position, 0.0, 0.0, 0.0, 1.0]
    return builder.finalize(device=device)


def _make_large_contact_articulation(device, ball_joint_count=20):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    root = builder.add_link(mass=2.0)
    builder.add_shape_sphere(root, radius=0.2, cfg=shape_cfg)
    joints = [builder.add_joint_free(parent=-1, child=root)]
    parent = root
    for _ in range(ball_joint_count):
        child = builder.add_link(mass=0.2)
        builder.add_shape_sphere(child, radius=0.04, cfg=shape_cfg)
        joints.append(
            builder.add_joint_ball(
                parent=parent,
                child=child,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.joint_q[:3] = [0.0, 0.0, 0.19]
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(collision_group=-1))
    return builder.finalize(device=device)


def _make_contact_overflow_model(device, contact_count=_POINTS_PER_PAGE + 8):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    body = builder.add_link(mass=2.0)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, restitution=0.0)
    width = 6
    for index in range(contact_count):
        x = 0.12 * float(index % width)
        y = 0.12 * float(index // width)
        builder.add_shape_sphere(
            body,
            radius=0.055,
            xform=wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity()),
            cfg=shape_cfg,
        )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.joint_q[-7:] = [-0.3, -0.18, 0.05, 0.0, 0.0, 0.0, 1.0]
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(collision_group=-1))
    return builder.finalize(device=device)


def _make_grounded_articulation_cluster(device, *, worlds=3, articulations_per_world=2):
    cluster = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    for articulation in range(articulations_per_world):
        body = cluster.add_link(mass=1.0)
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.0,
            restitution=0.0,
            collision_group=articulation + 1,
        )
        cluster.add_shape_sphere(body, radius=0.2, cfg=shape_cfg)
        joint = cluster.add_joint_free(parent=-1, child=body)
        cluster.add_articulation([joint])
        cluster.joint_q[-7:] = [float(articulation), 0.0, 0.19, 0.0, 0.0, 0.0, 1.0]
    cluster.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(collision_group=-1))
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    builder.replicate(cluster, world_count=worlds)
    return builder.finalize(device=device)


def _make_mixed_reduced_maximal_ground_model(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    reduced_body = builder.add_link(mass=1.0)
    maximal_body = builder.add_link(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.19), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(reduced_body, radius=0.2, cfg=shape_cfg)
    builder.add_shape_sphere(maximal_body, radius=0.2, cfg=shape_cfg)
    free_joint = builder.add_joint_free(parent=-1, child=reduced_body)
    builder.add_articulation([free_joint])
    builder.joint_q[-7:] = [-1.0, 0.0, 0.19, 0.0, 0.0, 0.0, 1.0]
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(collision_group=-1))
    return builder.finalize(device=device)


def _make_self_contact_model(device):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    root = builder.add_link(mass=2.0)
    middle = builder.add_link(mass=1.0)
    tip = builder.add_link(mass=1.5)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    builder.add_shape_sphere(root, radius=0.2, cfg=shape_cfg)
    builder.add_shape_sphere(tip, radius=0.2, cfg=shape_cfg)
    free_joint = builder.add_joint_free(parent=-1, child=root)
    middle_joint = builder.add_joint_revolute(
        parent=root,
        child=middle,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
    )
    tip_joint = builder.add_joint_revolute(
        parent=middle,
        child=tip,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, middle_joint, tip_joint])
    return builder.finalize(device=device)


@wp.kernel
def _apply_internal_wrench_pair(
    bodies: BodyContainer,
    body_slot0: wp.int32,
    body_slot1: wp.int32,
    torque: wp.vec3,
):
    _articulation_pair_wrench_response(
        bodies,
        body_slot0,
        wp.spatial_vector(wp.vec3(0.0), torque),
        body_slot1,
        wp.spatial_vector(wp.vec3(0.0), -torque),
        wp.bool(True),
    )


@wp.kernel
def _compare_cached_impulse_response(
    bodies: BodyContainer,
    body_slot: wp.int32,
    wrench: wp.spatial_vector,
    result: wp.array[wp.float32],
    response_result: wp.array[wp.spatial_vector],
    result_index: wp.int32,
):
    exact = _articulation_pair_wrench_response(
        bodies,
        body_slot,
        wrench,
        wp.int32(-1),
        wp.spatial_vector(),
        wp.bool(False),
    )
    body = body_slot - wp.int32(1)
    response = wp.spatial_vector()
    for column in range(6):
        response += bodies.reduced.impulse_response[body, column] * wrench[column]
    result[result_index] = exact
    result[result_index + wp.int32(1)] = wp.dot(wrench, response)
    response_result[result_index] = bodies.reduced.body_acceleration[body]
    response_result[result_index + wp.int32(1)] = response


@wp.kernel
def _compare_deferred_contact_response(
    bodies: BodyContainer,
    body_slot: wp.int32,
    point: wp.vec3,
    impulse_on_static: wp.vec3,
    exact_twist: wp.array[wp.spatial_vector],
    deferred_twist: wp.array[wp.spatial_vector],
    exact_qd: wp.array[wp.float32],
):
    link = body_slot - wp.int32(1)
    _articulation_pair_wrench_response(
        bodies,
        body_slot,
        wp.spatial_vector(-impulse_on_static, wp.cross(point, -impulse_on_static)),
        wp.int32(-1),
        wp.spatial_vector(),
        wp.bool(False),
    )
    exact_twist[0] = bodies.reduced.body_acceleration[link]
    articulation = bodies.reduced.body_articulation[body_slot]
    start = bodies.reduced.articulation_start[articulation]
    end = bodies.reduced.articulation_end[articulation]
    dof_start = bodies.reduced.joint_qd_start[start]
    dof_end = bodies.reduced.joint_qd_start[end]
    for dof in range(dof_start, dof_end):
        exact_qd[dof] = bodies.reduced.generalized_response[dof]

    _apply_deferred_impulse(
        bodies,
        body_slot,
        point,
        wp.int32(0),
        point,
        impulse_on_static,
    )
    deferred_twist[0] = _deferred_link_delta_twist(bodies, body_slot)
    _flush_deferred_articulation(bodies, articulation)


@wp.kernel
def _compare_generalized_contact_row(
    bodies: BodyContainer,
    body_slot: wp.int32,
    wrench: wp.spatial_vector,
    jacobian: wp.array3d[wp.float32],
    probe: wp.array[wp.float32],
    exact_response: wp.array[wp.float32],
    virtual_work: wp.array[wp.float32],
):
    _articulation_pair_wrench_response(
        bodies,
        body_slot,
        wrench,
        wp.int32(-1),
        wp.spatial_vector(),
        wp.bool(False),
    )
    articulation = bodies.reduced.body_articulation[body_slot]
    start = bodies.reduced.articulation_start[articulation]
    end = bodies.reduced.articulation_end[articulation]
    dof_start = bodies.reduced.joint_qd_start[start]
    dof_end = bodies.reduced.joint_qd_start[end]
    for dof in range(dof_start, dof_end):
        exact_response[dof - dof_start] = bodies.reduced.generalized_response[dof]

    body = body_slot - wp.int32(1)
    twist = wp.spatial_vector()
    path_start = bodies.reduced.body_path_start[body]
    path_end = bodies.reduced.body_path_start[body + wp.int32(1)]
    for path_index in range(path_start, path_end):
        joint = bodies.reduced.body_path_joint[path_index]
        for dof in range(
            bodies.reduced.joint_qd_start[joint],
            bodies.reduced.joint_qd_start[joint + wp.int32(1)],
        ):
            twist += bodies.reduced.joint_s[dof] * probe[dof - dof_start]
    jacobian_work = wp.float32(0.0)
    for local_dof in range(dof_end - dof_start):
        jacobian_work += jacobian[articulation, 0, local_dof] * probe[local_dof]
    virtual_work[0] = wp.dot(wrench, twist)
    virtual_work[1] = jacobian_work


@wp.kernel
def _compute_body_momentum(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    momentum: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    transform = body_q[body]
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(transform))
    position_com = wp.transform_point(transform, body_com[body])
    twist = body_qd[body]
    linear = body_mass[body] * wp.spatial_top(twist)
    inertia_world = rotation * body_inertia[body] * wp.transpose(rotation)
    angular = inertia_world * wp.spatial_bottom(twist) + wp.cross(position_com, linear)
    momentum[body] = wp.spatial_vector(linear, angular)


@wp.kernel
def _compute_body_kinetic_energy(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    energy: wp.array[wp.float32],
):
    body = wp.tid()
    rotation = wp.transform_get_rotation(body_q[body])
    twist = body_qd[body]
    velocity = wp.spatial_top(twist)
    omega_body = wp.quat_rotate_inv(rotation, wp.spatial_bottom(twist))
    energy[body] = wp.float32(0.5) * (
        body_mass[body] * wp.dot(velocity, velocity) + wp.dot(omega_body, body_inertia[body] * omega_body)
    )


def _body_momenta(model, state):
    momentum = wp.empty(model.body_count, dtype=wp.spatial_vector, device=model.device)
    wp.launch(
        _compute_body_momentum,
        dim=model.body_count,
        inputs=[state.body_q, state.body_qd, model.body_com, model.body_mass, model.body_inertia],
        outputs=[momentum],
        device=model.device,
    )
    return momentum.numpy().astype(np.float64)


def _total_momentum(model, state):
    return np.sum(_body_momenta(model, state), axis=0, dtype=np.float64)


def _kinetic_energy(model, state):
    energy = wp.empty(model.body_count, dtype=wp.float32, device=model.device)
    wp.launch(
        _compute_body_kinetic_energy,
        dim=model.body_count,
        inputs=[state.body_q, state.body_qd, model.body_mass, model.body_inertia],
        outputs=[energy],
        device=model.device,
    )
    return float(np.sum(energy.numpy(), dtype=np.float64))


def _loop_closure_error(model, state, loop_joint):
    body = int(model.joint_child.numpy()[loop_joint])
    body_transform = wp.transform(*state.body_q.numpy()[body])
    child_anchor_local = model.joint_X_c.numpy()[loop_joint, :3]
    child_anchor = np.asarray(wp.transform_point(body_transform, wp.vec3(*child_anchor_local)))
    parent = int(model.joint_parent.numpy()[loop_joint])
    parent_anchor_local = model.joint_X_p.numpy()[loop_joint, :3]
    if parent < 0:
        parent_anchor = parent_anchor_local
    else:
        parent_transform = wp.transform(*state.body_q.numpy()[parent])
        parent_anchor = np.asarray(wp.transform_point(parent_transform, wp.vec3(*parent_anchor_local)))
    return float(np.linalg.norm(child_anchor - parent_anchor))


class TestReducedArticulation(unittest.TestCase):
    def test_aba_matches_common_mass_matrix_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_mixed_tree(device)
        state = model.state()
        q = state.joint_q.numpy()
        q[0] = 0.35
        q[1] = 0.12
        q[2:6] = np.asarray(wp.quat_rpy(0.2, -0.15, 0.3), dtype=np.float32)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        system = ReducedArticulationSystem(model)
        system.use_warp_factor = True
        tau_np = np.array([0.7, -0.4, 0.2, -0.3, 0.5], dtype=np.float32)
        tau = wp.array(tau_np, dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            system.factor(state)
            result = system.solve_generalized(tau)
        wp.capture_launch(capture.graph)

        h = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
        h += np.diag(model.joint_armature.numpy())
        expected = np.linalg.solve(h.astype(np.float64), tau_np.astype(np.float64))
        np.testing.assert_allclose(result.numpy(), expected, rtol=2.0e-4, atol=2.0e-5)

    def test_far_translated_floating_trees_are_invariant_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_translated_floating_pair(device)
        state0 = model.state()
        state1 = model.state()
        q = state0.joint_q.numpy()
        qd = np.zeros(model.joint_dof_count, dtype=np.float32)
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        reference_qd = np.array([0.35, -0.2, 0.15, 0.4, -0.3, 0.25, 0.8], dtype=np.float32)
        for articulation in range(2):
            start = int(articulation_start[articulation])
            end = int(articulation_end[articulation])
            q[int(joint_q_start[start]) + 3 : int(joint_q_start[start]) + 7] = np.asarray(
                wp.quat_rpy(0.2, -0.15, 0.3), dtype=np.float32
            )
            q[int(joint_q_start[start + 1])] = 0.35
            qd[int(joint_qd_start[start]) : int(joint_qd_start[end])] = reference_qd
        for state in (state0, state1):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        dt = 1.0 / 1000.0
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state0, state1, None, None, dt)
            solver.step(state1, state0, None, None, dt)
        for _ in range(64):
            wp.capture_launch(capture.graph)

        result_q = state0.joint_q.numpy()
        result_qd = state0.joint_qd.numpy()
        start0 = int(articulation_start[0])
        end0 = int(articulation_end[0])
        start1 = int(articulation_start[1])
        end1 = int(articulation_end[1])
        np.testing.assert_allclose(
            result_qd[int(joint_qd_start[start1]) : int(joint_qd_start[end1])],
            result_qd[int(joint_qd_start[start0]) : int(joint_qd_start[end0])],
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            result_q[int(joint_q_start[start1]) + 3 : int(joint_q_start[start1]) + 7],
            result_q[int(joint_q_start[start0]) + 3 : int(joint_q_start[start0]) + 7],
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            result_q[int(joint_q_start[start1 + 1])],
            result_q[int(joint_q_start[start0 + 1])],
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(state0.body_qd.numpy()[2:], state0.body_qd.numpy()[:2], rtol=3.0e-5, atol=3.0e-6)

    def test_forward_dynamics_body_forces_are_translation_invariant_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_translated_floating_pair(device)
        state = model.state()
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        reference_qd = np.array([0.25, -0.15, 0.1, 0.2, -0.3, 0.35, -0.45], dtype=np.float32)
        for articulation in range(2):
            start = int(articulation_start[articulation])
            end = int(articulation_end[articulation])
            q[int(q_start[start]) + 3 : int(q_start[start]) + 7] = np.asarray(
                wp.quat_rpy(0.2, -0.15, 0.3), dtype=np.float32
            )
            q[int(q_start[start + 1])] = 0.35
            qd[int(qd_start[start]) : int(qd_start[end])] = reference_qd
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        body_force_pair = np.array(
            [
                [0.4, -0.2, 0.3, -0.1, 0.25, 0.15],
                [-0.3, 0.35, -0.2, 0.2, -0.15, 0.1],
            ],
            dtype=np.float32,
        )
        state.body_f.assign(np.tile(body_force_pair, (2, 1)))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        reference_force = np.array([0.2, -0.1, 0.15, -0.2, 0.1, 0.25, -0.3], dtype=np.float32)
        control.joint_f.assign(np.tile(reference_force, 2))
        system = ReducedArticulationSystem(model)

        with wp.ScopedCapture(device=device) as capture:
            acceleration = system.forward_dynamics(state, control)
        wp.capture_launch(capture.graph)

        result = acceleration.numpy()
        start0 = int(articulation_start[0])
        end0 = int(articulation_end[0])
        start1 = int(articulation_start[1])
        end1 = int(articulation_end[1])
        np.testing.assert_allclose(
            result[int(qd_start[start1]) : int(qd_start[end1])],
            result[int(qd_start[start0]) : int(qd_start[end0])],
            rtol=2.0e-5,
            atol=2.0e-5,
        )

    def test_live_body_mass_update_refreshes_reduced_inertia_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        body = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        dt = 1.0e-3

        def run_captured_step():
            state = model.state()
            output = model.state()
            body_force = np.zeros((model.body_count, 6), dtype=np.float32)
            body_force[body, 0] = 1.0
            state.body_f.assign(body_force)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            with wp.ScopedCapture(device=device) as capture:
                solver.step(state, output, None, None, dt)
            wp.capture_launch(capture.graph)
            return output.joint_qd.numpy().copy()

        velocity_before = run_captured_step()
        mass = model.body_mass.numpy()
        inverse_mass = model.body_inv_mass.numpy()
        mass[body] *= 2.0
        inverse_mass[body] *= 0.5
        model.body_mass.assign(mass)
        model.body_inv_mass.assign(inverse_mass)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        velocity_after = run_captured_step()

        self.assertGreater(abs(float(velocity_before[0])), 1.0e-6)
        np.testing.assert_allclose(velocity_after[0], 0.5 * velocity_before[0], rtol=2.0e-4, atol=1.0e-7)

    def test_descendant_free_distance_preserves_com_velocity_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        for joint_type in (newton.JointType.FREE, newton.JointType.DISTANCE):
            with self.subTest(joint_type=joint_type):
                model, _base, child, _root, joint = _make_descendant_free_distance(device, joint_type)
                state0 = model.state()
                state1 = model.state()
                q = state0.joint_q.numpy()
                qd = state0.joint_qd.numpy()
                q_start = int(model.joint_q_start.numpy()[joint])
                qd_start = int(model.joint_qd_start.numpy()[joint])
                q[q_start : q_start + 3] = np.array([0.4, -0.25, 0.3], dtype=np.float32)
                rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.5, -0.2)), 0.35)
                q[q_start + 3 : q_start + 7] = np.asarray(rotation, dtype=np.float32)
                expected_qd = np.array([0.0, 0.0, 0.0, 0.3, -0.4, 0.5], dtype=np.float32)
                qd[qd_start : qd_start + 6] = expected_qd
                for state in (state0, state1):
                    state.joint_q.assign(q)
                    state.joint_qd.assign(qd)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                initial_transform = wp.transform(*state0.body_q.numpy()[child])
                initial_com = np.asarray(
                    wp.transform_point(initial_transform, wp.vec3(*model.body_com.numpy()[child])),
                    dtype=np.float64,
                )
                solver = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="reduced",
                    substeps=1,
                    solver_iterations=1,
                    velocity_iterations=0,
                )
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state0, state1, None, None, 0.01)
                    solver.step(state1, state0, None, None, 0.01)
                for _ in range(5):
                    wp.capture_launch(capture.graph)

                final_transform = wp.transform(*state0.body_q.numpy()[child])
                final_com = np.asarray(
                    wp.transform_point(final_transform, wp.vec3(*model.body_com.numpy()[child])),
                    dtype=np.float64,
                )
                self.assertLess(float(np.linalg.norm(final_com - initial_com)), 2.0e-4)
                np.testing.assert_allclose(state0.body_qd.numpy()[child, :3], 0.0, atol=2.0e-4)
                np.testing.assert_allclose(
                    state0.joint_qd.numpy()[qd_start : qd_start + 6],
                    expected_qd,
                    rtol=1.0e-6,
                    atol=2.0e-4,
                )

    def test_descendant_free_distance_decouples_parent_torque_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        for joint_type in (newton.JointType.FREE, newton.JointType.DISTANCE):
            with self.subTest(joint_type=joint_type):
                model, base, child, root, joint = _make_descendant_free_distance(
                    device,
                    joint_type,
                    parent_kinematic=False,
                )
                state0 = model.state()
                state1 = model.state()
                control = model.control()
                q = state0.joint_q.numpy()
                q_start = int(model.joint_q_start.numpy()[joint])
                q[q_start : q_start + 3] = np.array([0.4, -0.25, 0.3], dtype=np.float32)
                rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.5, -0.2)), 0.35)
                q[q_start + 3 : q_start + 7] = np.asarray(rotation, dtype=np.float32)
                joint_f = control.joint_f.numpy()
                joint_f[int(model.joint_qd_start.numpy()[root])] = 7.5
                control.joint_f.assign(joint_f)
                for state in (state0, state1):
                    state.joint_q.assign(q)
                    state.joint_qd.zero_()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                initial_child_pose = state0.body_q.numpy()[child].copy()

                solver = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="reduced",
                    substeps=1,
                    solver_iterations=1,
                    velocity_iterations=0,
                )
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state0, state1, control, None, 0.001)
                    solver.step(state1, state0, control, None, 0.001)
                for _ in range(10):
                    wp.capture_launch(capture.graph)

                self.assertGreater(float(np.linalg.norm(state0.body_qd.numpy()[base, 3:])), 1.0e-3)
                np.testing.assert_allclose(state0.body_qd.numpy()[child], 0.0, atol=5.0e-4)
                np.testing.assert_allclose(state0.body_q.numpy()[child, :3], initial_child_pose[:3], atol=2.0e-5)
                self.assertGreater(
                    float(
                        np.linalg.norm(
                            state0.joint_qd.numpy()[
                                int(model.joint_qd_start.numpy()[joint]) : int(model.joint_qd_start.numpy()[joint + 1])
                            ]
                        )
                    ),
                    1.0e-3,
                )

    def test_long_horizon_floating_tree_conserves_momentum_and_bounds_energy(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state0 = model.state()
        state1 = model.state()
        q = state0.joint_q.numpy()
        q[:3] = np.array([0.2, -0.1, 0.4], dtype=np.float32)
        q[3:7] = np.asarray(wp.quat_rpy(0.2, -0.1, 0.3), dtype=np.float32)
        q[7] = 0.45
        qd = np.array([0.4, -0.2, 0.15, 0.3, -0.25, 0.2, 1.1], dtype=np.float32)
        for state in (state0, state1):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        initial_momentum = _total_momentum(model, state0)
        initial_energy = _kinetic_energy(model, state0)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        dt = 1.0 / 1000.0
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state0, state1, None, None, dt)
            solver.step(state1, state0, None, None, dt)
        for _ in range(256):
            wp.capture_launch(capture.graph)

        final_momentum = _total_momentum(model, state0)
        final_energy = _kinetic_energy(model, state0)
        momentum_error = final_momentum - initial_momentum
        linear_relative_error = np.linalg.norm(momentum_error[:3]) / np.linalg.norm(initial_momentum[:3])
        angular_relative_error = np.linalg.norm(momentum_error[3:]) / np.linalg.norm(initial_momentum[3:])
        self.assertLess(linear_relative_error, 5.0e-4)
        self.assertLess(angular_relative_error, 5.0e-4)
        self.assertLess(abs(final_energy - initial_energy) / initial_energy, 2.0e-3)
        self.assertTrue(np.isfinite(state0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state0.joint_qd.numpy()).all())

        fk_state = model.state()
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, fk_state)
        np.testing.assert_allclose(state0.body_q.numpy(), fk_state.body_q.numpy(), atol=2.0e-5)
        np.testing.assert_allclose(state0.body_qd.numpy(), fk_state.body_qd.numpy(), atol=2.0e-5)

    def test_hybrid_maximal_floating_tree_conserves_momentum_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("hybrid articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state0 = model.state()
        state1 = model.state()
        q = state0.joint_q.numpy()
        q[:3] = np.array([0.2, -0.1, 0.4], dtype=np.float32)
        q[3:7] = np.asarray(wp.quat_rpy(0.2, -0.1, 0.3), dtype=np.float32)
        q[7] = 0.45
        qd = np.array([0.4, -0.2, 0.15, 0.3, -0.25, 0.2, 1.1], dtype=np.float32)
        for state in (state0, state1):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        momentum_before = _total_momentum(model, state0)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="hybrid",
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
        )
        np.testing.assert_array_equal(
            solver.world._joint_pgs_enabled.numpy(),
            np.ones(solver._adbs.num_joint_columns, dtype=np.int32),
        )
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state0, state1, None, None, 1.0 / 1000.0)
            solver.step(state1, state0, None, None, 1.0 / 1000.0)
        for _ in range(128):
            wp.capture_launch(capture.graph)

        momentum_after = _total_momentum(model, state0)
        momentum_error = momentum_after - momentum_before
        linear_relative_error = np.linalg.norm(momentum_error[:3]) / np.linalg.norm(momentum_before[:3])
        angular_relative_error = np.linalg.norm(momentum_error[3:]) / np.linalg.norm(momentum_before[3:])
        self.assertLess(linear_relative_error, 2.0e-4)
        self.assertLess(angular_relative_error, 2.0e-4)
        self.assertTrue(np.isfinite(state0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state0.joint_qd.numpy()).all())

    def test_floating_internal_loop_conserves_momentum_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model, loop_joint = _make_floating_triangle_loop(device)
        state = model.state()
        output = model.state()
        qd = state.joint_qd.numpy()
        qd[:6] = np.array([0.2, -0.1, 0.05, 0.1, -0.2, 0.3], dtype=np.float32)
        starts = model.joint_qd_start.numpy()
        qd[int(starts[1])] = 0.8
        qd[int(starts[2])] = -0.45
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        momentum_before = _total_momentum(model, state)
        qd_before = state.joint_qd.numpy().copy()

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
        )
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state, output, None, None, 1.0 / 2000.0)
        wp.capture_launch(capture.graph)

        momentum_after = _total_momentum(model, output)
        np.testing.assert_allclose(momentum_after, momentum_before, rtol=0.0, atol=1.0e-5)
        self.assertLess(_loop_closure_error(model, output, loop_joint), 1.0e-4)
        self.assertGreater(float(np.linalg.norm(output.joint_qd.numpy() - qd_before)), 1.0e-3)

    def test_ball_prismatic_and_fixed_loop_rows_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        initial = np.array([0.5, -0.4, 0.3, 0.2, -0.1, 0.35], dtype=np.float32)
        joint_types = (newton.JointType.BALL, newton.JointType.PRISMATIC, newton.JointType.FIXED)
        for joint_type in joint_types:
            with self.subTest(joint_type=joint_type):
                model = _make_free_body_loop(device, joint_type)
                state = model.state()
                output = model.state()
                qd = state.joint_qd.numpy()
                qd[:6] = initial
                state.joint_qd.assign(qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                solver = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="reduced",
                    substeps=1,
                    solver_iterations=2,
                    velocity_iterations=0,
                )
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state, output, None, None, 1.0 / 1000.0)
                wp.capture_launch(capture.graph)
                velocity = output.body_qd.numpy()[0]
                if joint_type == newton.JointType.BALL:
                    self.assertLess(float(np.linalg.norm(velocity[:3])), 4.5e-2)
                    np.testing.assert_allclose(velocity[3:], initial[3:], rtol=0.0, atol=5.0e-5)
                elif joint_type == newton.JointType.PRISMATIC:
                    self.assertAlmostEqual(float(velocity[0]), float(initial[0]), delta=5.0e-5)
                    self.assertLess(float(np.linalg.norm(velocity[1:])), 4.5e-2)
                else:
                    self.assertLess(float(np.linalg.norm(velocity)), 5.0e-2)

    def test_tiled_loop_response_handles_six_dof_roots(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        articulation_count = 64
        model = _make_free_body_loop_cluster(device, newton.JointType.BALL, articulation_count)
        state = model.state()
        output = model.state()
        initial = np.array([0.5, -0.4, 0.3, 0.2, -0.1, 0.35], dtype=np.float32)
        qd = state.joint_qd.numpy()
        qd_start = model.joint_qd_start.numpy()
        articulation_start = model.articulation_start.numpy()
        for joint in articulation_start[:articulation_count]:
            start = int(qd_start[joint])
            qd[start : start + 6] = initial
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=2,
            velocity_iterations=0,
        )
        self.assertEqual(solver._reduced_articulation.loop_system.color_count, 1)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state, output, None, None, 1.0 / 1000.0)
        wp.capture_launch(capture.graph)

        velocity = output.body_qd.numpy()
        self.assertLess(float(np.max(np.linalg.norm(velocity[:, :3], axis=1))), 4.5e-2)
        expected_angular = np.broadcast_to(initial[3:], velocity[:, 3:].shape)
        np.testing.assert_allclose(velocity[:, 3:], expected_angular, rtol=0.0, atol=5.0e-4)

    def test_revolute_four_bar_loop_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model, loop_joint = _make_four_bar_loop(device)
        state0 = model.state()
        state1 = model.state()
        qd = state0.joint_qd.numpy()
        starts = model.joint_qd_start.numpy()
        qd[int(starts[0])] = 1.0
        qd[int(starts[1])] = -1.0
        qd[int(starts[2])] = 1.0
        state0.joint_qd.assign(qd)
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
        state1.joint_q.assign(state0.joint_q)
        state1.joint_qd.assign(state0.joint_qd)
        state1.body_q.assign(state0.body_q)
        state1.body_qd.assign(state0.body_qd)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
        )
        loop_cid = int(solver._adbs.joint_idx_to_cid.numpy()[loop_joint])
        self.assertGreaterEqual(loop_cid, 0)
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[loop_cid]), 0)
        self.assertEqual(solver._reduced_articulation.loop_system.count, 1)

        control = model.control()
        dt = 1.0 / 1000.0
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state0, state1, control, None, dt)
            solver.step(state1, state0, control, None, dt)
        for _ in range(100):
            wp.capture_launch(capture.graph)

        closure_error = _loop_closure_error(model, state0, loop_joint)
        self.assertLess(closure_error, 5.0e-4)
        self.assertTrue(np.all(np.isfinite(state0.joint_q.numpy())))
        self.assertGreater(abs(float(state0.joint_q.numpy()[0] - model.joint_q.numpy()[0])), 0.05)

    def test_contacting_four_bar_loop_remains_closed_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model, loop_joint = _make_contact_four_bar_loop(device)
        state0 = model.state()
        state1 = model.state()
        qd = state0.joint_qd.numpy()
        starts = model.joint_qd_start.numpy()
        qd[int(starts[0])] = 0.5
        qd[int(starts[1])] = -0.5
        qd[int(starts[2])] = 0.5
        state0.joint_qd.assign(qd)
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
        state1.joint_q.assign(state0.joint_q)
        state1.joint_qd.assign(state0.joint_qd)
        state1.body_q.assign(state0.body_q)
        state1.body_qd.assign(state0.body_qd)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=1,
        )
        contacts = model.contacts()
        control = model.control()
        dt = 1.0 / 2000.0
        with wp.ScopedCapture(device=device) as capture:
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt)
            model.collide(state1, contacts)
            solver.step(state1, state0, control, contacts, dt)
        for _ in range(64):
            wp.capture_launch(capture.graph)

        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(_loop_closure_error(model, state0, loop_joint), 1.0e-3)
        self.assertTrue(np.isfinite(state0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state0.joint_qd.numpy()).all())
        self.assertTrue(np.isfinite(state0.body_qd.numpy()).all())

        fk_state = model.state()
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, fk_state)
        np.testing.assert_allclose(state0.body_qd.numpy(), fk_state.body_qd.numpy(), atol=5.0e-5)

    def test_multi_world_revolute_loops_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        world_count = 8
        model, loop_joints = _make_four_bar_fleet(device, world_count)
        state0 = model.state()
        state1 = model.state()
        qd = state0.joint_qd.numpy()
        qd_start = model.joint_qd_start.numpy()
        articulation_start = model.articulation_start.numpy()
        for world in range(world_count):
            joint = int(articulation_start[world])
            speed = 0.5 + 0.1 * world
            qd[int(qd_start[joint])] = speed
            qd[int(qd_start[joint + 1])] = -speed
            qd[int(qd_start[joint + 2])] = speed
        state0.joint_qd.assign(qd)
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
        state1.joint_q.assign(state0.joint_q)
        state1.joint_qd.assign(state0.joint_qd)
        state1.body_q.assign(state0.body_q)
        state1.body_qd.assign(state0.body_qd)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
        )
        self.assertEqual(solver._reduced_articulation.loop_system.count, world_count)
        joint_to_cid = solver._adbs.joint_idx_to_cid.numpy()
        pgs_enabled = solver.world._joint_pgs_enabled.numpy()
        for loop_joint in loop_joints:
            self.assertEqual(int(pgs_enabled[int(joint_to_cid[loop_joint])]), 0)

        control = model.control()
        dt = 1.0 / 1000.0
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state0, state1, control, None, dt)
            solver.step(state1, state0, control, None, dt)
        for _ in range(50):
            wp.capture_launch(capture.graph)

        errors = np.array([_loop_closure_error(model, state0, int(joint)) for joint in loop_joints])
        self.assertLess(float(np.max(errors)), 5.0e-4)
        q = state0.joint_q.numpy()
        angles = np.array([q[int(model.joint_q_start.numpy()[start])] for start in articulation_start[:world_count]])
        self.assertGreater(float(np.ptp(angles)), 0.02)

    def test_many_articulations_per_world_use_parallel_loop_schedule(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        layouts = ((1, 64, 1, 1), (8, 8, 1, 1), (1, 32, 2, 2))
        for worlds, articulations_per_world, loop_copies, expected_colors in layouts:
            with self.subTest(
                worlds=worlds,
                articulations_per_world=articulations_per_world,
                loop_copies=loop_copies,
            ):
                model, loop_joints = _make_four_bar_clusters(
                    device,
                    worlds=worlds,
                    articulations_per_world=articulations_per_world,
                    loop_copies=loop_copies,
                )
                state0 = model.state()
                state1 = model.state()
                qd = state0.joint_qd.numpy()
                qd_start = model.joint_qd_start.numpy()
                articulation_start = model.articulation_start.numpy()
                for articulation, joint in enumerate(articulation_start[: model.articulation_count]):
                    speed = 0.5 + 0.01 * (articulation % 11)
                    qd[int(qd_start[joint])] = speed
                    qd[int(qd_start[joint + 1])] = -speed
                    qd[int(qd_start[joint + 2])] = speed
                state0.joint_qd.assign(qd)
                newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
                state1.joint_q.assign(state0.joint_q)
                state1.joint_qd.assign(state0.joint_qd)
                state1.body_q.assign(state0.body_q)
                state1.body_qd.assign(state0.body_qd)
                initial_q = state0.joint_q.numpy().copy()

                solver = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="reduced",
                    substeps=1,
                    solver_iterations=4,
                    velocity_iterations=1,
                )
                loop_system = solver._reduced_articulation.loop_system
                self.assertEqual(loop_system.count, 64)
                self.assertEqual(loop_system.color_count, expected_colors)

                control = model.control()
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state0, state1, control, None, 1.0 / 1000.0)
                    solver.step(state1, state0, control, None, 1.0 / 1000.0)
                for _ in range(30):
                    wp.capture_launch(capture.graph)

                errors = np.array([_loop_closure_error(model, state0, int(joint)) for joint in loop_joints])
                self.assertLess(float(np.max(errors)), 5.0e-4)
                self.assertTrue(np.all(np.isfinite(state0.joint_q.numpy())))
                self.assertGreater(float(np.linalg.norm(state0.joint_q.numpy() - initial_q)), 0.05)

    def test_cached_impulse_response_matches_aba_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state = model.state()
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        bridge = solver._reduced_articulation
        result = wp.zeros(4, dtype=wp.float32, device=device)
        response_result = wp.zeros(4, dtype=wp.spatial_vector, device=device)
        wrench0 = wp.spatial_vector(0.7, -0.2, 0.4, -0.3, 0.6, 0.1)
        wrench1 = wp.spatial_vector(-0.1, 0.8, -0.5, 0.2, -0.4, 0.9)
        with wp.ScopedCapture(device=device) as capture:
            bridge.import_step(state, model.control())
            bridge.begin_substep(0.0, split_dynamics=False)
            wp.launch(
                _compare_cached_impulse_response,
                dim=1,
                inputs=[solver.bodies, wp.int32(1), wrench0, result, response_result, wp.int32(0)],
                device=device,
            )
            wp.launch(
                _compare_cached_impulse_response,
                dim=1,
                inputs=[solver.bodies, wp.int32(2), wrench1, result, response_result, wp.int32(2)],
                device=device,
            )
        wp.capture_launch(capture.graph)

        values = result.numpy()
        responses = response_result.numpy()
        np.testing.assert_allclose(responses[[1, 3]], responses[[0, 2]], rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(values[[1, 3]], values[[0, 2]], rtol=2.0e-5, atol=2.0e-6)

    def test_generalized_contact_rows_match_aba_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state = model.state()
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        bridge = solver._reduced_articulation
        block = bridge.contact_block_system
        dof_count = int(model.joint_dof_count)
        wrench = wp.spatial_vector(0.7, -0.2, 0.4, -0.3, 0.6, 0.1)
        row_body = np.ones(block.row_body.shape, dtype=np.int32)
        rng = np.random.default_rng(20260702)
        row_wrench = rng.uniform(-0.8, 0.8, (*block.row_wrench.shape, 6)).astype(np.float32)
        row_wrench[0, 0] = np.asarray(wrench, dtype=np.float32)
        block.enabled.assign(np.ones(block.enabled.shape, dtype=np.int32))
        block.point_count.assign(np.full(block.point_count.shape, _POINTS_PER_PAGE, dtype=np.int32))
        block.row_body.assign(row_body)
        block.row_wrench.assign(row_wrench)
        point_contact = np.arange(block.point_contact.size, dtype=np.int32).reshape(block.point_contact.shape)
        block.point_contact.assign(point_contact)
        probe_np = np.linspace(-0.35, 0.45, dof_count, dtype=np.float32)
        probe = wp.array(probe_np, device=device)
        exact_response = wp.zeros(dof_count, dtype=wp.float32, device=device)
        virtual_work = wp.zeros(2, dtype=wp.float32, device=device)
        reference_jacobian = wp.zeros(
            (1, block.row_wrench.shape[1], block.contact_dof_width),
            dtype=wp.float32,
            device=device,
        )
        reference_response = wp.zeros_like(reference_jacobian)
        assert block.packed_jacobian is not None
        assert block.packed_response is not None

        with wp.ScopedCapture(device=device) as capture:
            bridge.import_step(state, model.control())
            bridge.begin_substep(0.0, split_dynamics=False)
            wp.launch(
                _build_generalized_contact_rows_kernel,
                dim=(1, block.row_wrench.shape[1]),
                inputs=[
                    solver.bodies,
                    block.enabled,
                    block.point_count,
                    block.row_body,
                    block.row_wrench,
                ],
                outputs=[
                    reference_jacobian,
                    reference_response,
                    block.aba_joint_work,
                    block.aba_body_response,
                ],
                device=device,
            )
            wp.launch(
                _build_packed_generalized_contact_rows_kernel,
                dim=(1, block.row_wrench.shape[1]),
                inputs=[
                    solver.bodies,
                    block.enabled,
                    block.point_count,
                    block.row_body,
                    block.row_wrench,
                    block.point_contact,
                    solver.world._contact_container,
                    block.max_page_count,
                    block.page_index,
                    wp.bool(True),
                ],
                outputs=[
                    block.packed_jacobian,
                    block.packed_response,
                    block.aba_joint_work,
                    block.aba_body_response,
                ],
                device=device,
            )
            wp.launch_tiled(
                _transpose_generalized_contact_response_kernel,
                dim=[_RESPONSE_TILES_PER_ARTICULATION],
                block_dim=_RESPONSE_TILE,
                inputs=[
                    solver.bodies,
                    block.enabled,
                    block.point_count,
                    block.page_index,
                    block.max_page_count,
                    wp.bool(True),
                    block.aba_joint_work,
                ],
                outputs=[block.packed_response],
                device=device,
            )
            wp.launch(
                _compare_generalized_contact_row,
                dim=1,
                inputs=[
                    solver.bodies,
                    wp.int32(2),
                    wrench,
                    reference_jacobian,
                    probe,
                    exact_response,
                    virtual_work,
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(
            reference_response.numpy()[0, 0, :dof_count],
            exact_response.numpy(),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_array_equal(
            block.packed_jacobian.numpy()[0, :dof_count],
            reference_jacobian.numpy()[0, 0, :dof_count],
        )
        np.testing.assert_array_equal(
            block.packed_response.numpy()[: block.row_wrench.shape[1], :dof_count],
            reference_response.numpy()[0, :, :dof_count],
        )
        np.testing.assert_allclose(virtual_work.numpy()[1], virtual_work.numpy()[0], rtol=2.0e-5, atol=2.0e-6)
        jacobian_row = block.packed_jacobian.numpy()[0, :dof_count]
        response_row = block.packed_response.numpy()[0, :dof_count]
        expected_effective_mass = 1.0 / float(np.dot(jacobian_row, response_row))
        actual_effective_mass = float(solver.world._contact_container.derived.numpy()[0, 0])
        self.assertAlmostEqual(actual_effective_mass, expected_effective_mass, delta=2.0e-5)

    def test_deferred_contact_response_matches_aba_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state = model.state()
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        bridge = solver._reduced_articulation
        exact_twist = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        deferred_twist = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        exact_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        initial_qd = bridge.system.joint_qd_internal.numpy().copy()
        point = wp.vec3(0.45, -0.2, 0.35)
        impulse = wp.vec3(-0.7, 0.4, -0.3)
        with wp.ScopedCapture(device=device) as capture:
            bridge.import_step(state, model.control())
            bridge.begin_substep(0.0, split_dynamics=False)
            wp.launch(
                _compare_deferred_contact_response,
                dim=1,
                inputs=[
                    solver.bodies,
                    wp.int32(2),
                    point,
                    impulse,
                    exact_twist,
                    deferred_twist,
                    exact_qd,
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(deferred_twist.numpy(), exact_twist.numpy(), rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(
            bridge.system.joint_qd_internal.numpy() - initial_qd,
            exact_qd.numpy(),
            rtol=2.0e-5,
            atol=2.0e-6,
        )

    def test_spatial_wrench_pair_conserves_momentum_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state = model.state()
        control = model.control()
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        bridge = solver._reduced_articulation
        self.assertIsNotNone(bridge)
        momentum_before = _total_momentum(model, state)

        with wp.ScopedCapture(device=device) as capture:
            bridge.import_step(state, control)
            bridge.begin_substep(0.0, split_dynamics=False)
            wp.launch(
                _apply_internal_wrench_pair,
                dim=1,
                inputs=[solver.bodies, wp.int32(1), wp.int32(2), wp.vec3(0.3, -0.2, 0.4)],
                device=device,
            )
            bridge.end_substep(0.0, split_dynamics=False)
        wp.capture_launch(capture.graph)

        momentum_after = _total_momentum(model, bridge.state)
        np.testing.assert_allclose(momentum_after, momentum_before, rtol=0.0, atol=1.0e-5)
        self.assertGreater(float(np.linalg.norm(bridge.state.joint_qd.numpy())), 1.0e-4)

    def test_warp_advance_matches_serial_on_branched_tree_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_branched_tree(device)
        state_serial = model.state()
        state_warp = model.state()
        output_serial = model.state()
        output_warp = model.state()
        qd = np.linspace(-0.45, 0.55, int(model.joint_dof_count), dtype=np.float32)
        body_f = np.linspace(-0.7, 0.9, int(model.body_count) * 6, dtype=np.float32).reshape(-1, 6)
        for state in (state_serial, state_warp):
            state.joint_qd.assign(qd)
            state.body_f.assign(body_f)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        control = model.control()
        control.joint_f.assign(np.linspace(0.6, -0.4, int(model.joint_dof_count), dtype=np.float32))
        serial = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        warp = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
        )
        serial._reduced_articulation.system.use_warp_advance = False
        serial._reduced_articulation.system.use_warp_kinematics = False
        serial._reduced_articulation.system.use_warp_publish = False
        warp._reduced_articulation.system.use_warp_kinematics = True
        warp._reduced_articulation.system.use_warp_publish = True
        dt = 1.0 / 240.0

        with wp.ScopedCapture(device=device) as serial_capture:
            serial.step(state_serial, output_serial, control, None, dt)
            serial.step(output_serial, state_serial, control, None, dt)
        with wp.ScopedCapture(device=device) as warp_capture:
            warp.step(state_warp, output_warp, control, None, dt)
            warp.step(output_warp, state_warp, control, None, dt)
        for _ in range(16):
            wp.capture_launch(serial_capture.graph)
            wp.capture_launch(warp_capture.graph)

        np.testing.assert_allclose(state_warp.joint_qd.numpy(), state_serial.joint_qd.numpy(), rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(state_warp.joint_q.numpy(), state_serial.joint_q.numpy(), rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(state_warp.body_qd.numpy(), state_serial.body_qd.numpy(), rtol=2.0e-6, atol=2.0e-6)

    def test_subwarp_kinematics_advance_and_publish_match_serial_for_all_topology_widths(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        for child_count, expected_tile_width in ((7, 8), (9, 16), (17, 32)):
            with self.subTest(tile_width=expected_tile_width):
                model = _make_wide_tree(device, child_count)
                state_serial = model.state()
                state_warp = model.state()
                output_serial = model.state()
                output_warp = model.state()
                qd = np.linspace(-0.35, 0.45, int(model.joint_dof_count), dtype=np.float32)
                body_f = np.linspace(-0.5, 0.7, int(model.body_count) * 6, dtype=np.float32).reshape(-1, 6)
                for state in (state_serial, state_warp):
                    state.joint_qd.assign(qd)
                    state.body_f.assign(body_f)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                control = model.control()
                control.joint_f.assign(np.linspace(0.4, -0.3, int(model.joint_dof_count), dtype=np.float32))
                serial = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="reduced",
                    substeps=1,
                    solver_iterations=1,
                    velocity_iterations=0,
                )
                warp = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="reduced",
                    substeps=1,
                    solver_iterations=1,
                    velocity_iterations=0,
                )
                serial._reduced_articulation.system.use_warp_advance = False
                serial._reduced_articulation.system.use_warp_kinematics = False
                serial._reduced_articulation.system.use_warp_publish = False
                self.assertEqual(warp._reduced_articulation.system.advance_tile_width, expected_tile_width)
                self.assertTrue(warp._reduced_articulation.system.use_warp_publish)
                dt = 1.0 / 240.0

                with wp.ScopedCapture(device=device) as serial_capture:
                    serial.step(state_serial, output_serial, control, None, dt)
                    serial.step(output_serial, state_serial, control, None, dt)
                with wp.ScopedCapture(device=device) as warp_capture:
                    warp.step(state_warp, output_warp, control, None, dt)
                    warp.step(output_warp, state_warp, control, None, dt)
                for _ in range(4):
                    wp.capture_launch(serial_capture.graph)
                    wp.capture_launch(warp_capture.graph)

                np.testing.assert_allclose(
                    state_warp.joint_qd.numpy(),
                    state_serial.joint_qd.numpy(),
                    rtol=2.0e-6,
                    atol=2.0e-6,
                )
                np.testing.assert_allclose(
                    state_warp.joint_q.numpy(),
                    state_serial.joint_q.numpy(),
                    rtol=2.0e-6,
                    atol=2.0e-6,
                )
                np.testing.assert_allclose(
                    state_warp.body_qd.numpy(),
                    state_serial.body_qd.numpy(),
                    rtol=2.0e-6,
                    atol=2.0e-6,
                )

    def test_common_solver_api_matches_featherstone_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_mixed_tree(device)
        state_phoenx = model.state()
        state_featherstone = model.state()
        output_phoenx = model.state()
        output_featherstone = model.state()
        q = state_phoenx.joint_q.numpy()
        q[0] = 0.35
        q[1] = 0.12
        q[2:6] = np.asarray(wp.quat_rpy(0.2, -0.15, 0.3), dtype=np.float32)
        qd = np.array([0.3, -0.2, 0.1, -0.15, 0.25], dtype=np.float32)
        state_phoenx.joint_q.assign(q)
        state_phoenx.joint_qd.assign(qd)
        state_featherstone.joint_q.assign(q)
        state_featherstone.joint_qd.assign(qd)
        newton.eval_fk(model, state_phoenx.joint_q, state_phoenx.joint_qd, state_phoenx)
        newton.eval_fk(model, state_featherstone.joint_q, state_featherstone.joint_qd, state_featherstone)

        control = model.control()
        control.joint_f.assign(np.array([0.7, -0.4, 0.2, -0.3, 0.5], dtype=np.float32))
        phoenx = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
        )
        featherstone = newton.solvers.SolverFeatherstone(model)
        dt = 1.0 / 240.0

        with wp.ScopedCapture(device=device) as capture_phoenx:
            phoenx.step(state_phoenx, output_phoenx, control, None, dt)
        wp.capture_launch(capture_phoenx.graph)
        with wp.ScopedCapture(device=device) as capture_featherstone:
            featherstone.step(state_featherstone, output_featherstone, control, None, dt)
        wp.capture_launch(capture_featherstone.graph)

        np.testing.assert_allclose(
            output_phoenx.joint_qd.numpy(), output_featherstone.joint_qd.numpy(), rtol=3.0e-4, atol=3.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.joint_q.numpy(), output_featherstone.joint_q.numpy(), rtol=3.0e-4, atol=3.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.body_q.numpy(), output_featherstone.body_q.numpy(), rtol=3.0e-4, atol=3.0e-5
        )

    def test_floating_common_solver_api_matches_featherstone_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_floating_tree(device)
        state_phoenx = model.state()
        state_featherstone = model.state()
        output_phoenx = model.state()
        output_featherstone = model.state()
        q = state_phoenx.joint_q.numpy()
        q[:3] = np.array([0.2, -0.1, 0.4], dtype=np.float32)
        q[3:7] = np.asarray(wp.quat_rpy(0.2, -0.1, 0.3), dtype=np.float32)
        q[7] = 0.25
        qd = np.array([0.4, -0.2, 0.15, 0.3, -0.1, 0.2, -0.35], dtype=np.float32)
        body_f = np.array(
            [
                [0.3, -0.2, 0.1, -0.15, 0.25, 0.05],
                [-0.1, 0.15, -0.05, 0.2, -0.1, 0.3],
            ],
            dtype=np.float32,
        )
        for state in (state_phoenx, state_featherstone):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            state.body_f.assign(body_f)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        control = model.control()
        control.joint_f.assign(np.array([0.2, -0.1, 0.15, -0.3, 0.25, 0.1, 0.6], dtype=np.float32))
        phoenx = newton.solvers.SolverPhoenX(
            model, articulation_mode="reduced", substeps=1, solver_iterations=1, velocity_iterations=1
        )
        featherstone = newton.solvers.SolverFeatherstone(model)
        dt = 1.0 / 240.0

        with wp.ScopedCapture(device=device) as capture_phoenx:
            phoenx.step(state_phoenx, output_phoenx, control, None, dt)
        wp.capture_launch(capture_phoenx.graph)
        with wp.ScopedCapture(device=device) as capture_featherstone:
            featherstone.step(state_featherstone, output_featherstone, control, None, dt)
        wp.capture_launch(capture_featherstone.graph)

        # PhoenX projects the floating-base rigid-motion coordinates after
        # configuration drift to preserve world momentum. Featherstone uses
        # an unprojected semi-implicit update, so generalized velocities are
        # expected to differ slightly while poses remain much closer.
        np.testing.assert_allclose(
            output_phoenx.joint_qd.numpy(), output_featherstone.joint_qd.numpy(), rtol=2.5e-3, atol=8.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.joint_q.numpy(), output_featherstone.joint_q.numpy(), rtol=6.0e-4, atol=6.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.body_q.numpy(), output_featherstone.body_q.numpy(), rtol=6.0e-4, atol=6.0e-5
        )

    def test_multi_world_common_api_matches_featherstone_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        world_count = 8
        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        builder.replicate(_make_mixed_tree_builder(), world_count=world_count)
        model = builder.finalize(device=device)
        state_phoenx = model.state()
        state_featherstone = model.state()
        output_phoenx = model.state()
        output_featherstone = model.state()

        q = state_phoenx.joint_q.numpy().reshape(world_count, -1)
        qd = state_phoenx.joint_qd.numpy().reshape(world_count, -1)
        for world in range(world_count):
            scale = float(world + 1) / float(world_count)
            q[world, 0] = 0.3 * scale
            q[world, 1] = -0.1 * scale
            q[world, 2:6] = np.asarray(wp.quat_rpy(0.1 * scale, -0.2 * scale, 0.15 * scale))
            qd[world] = np.array([0.2, -0.1, 0.05, -0.15, 0.25], dtype=np.float32) * scale
        for state in (state_phoenx, state_featherstone):
            state.joint_q.assign(q.reshape(-1))
            state.joint_qd.assign(qd.reshape(-1))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        control = model.control()
        joint_f = np.empty((world_count, 5), dtype=np.float32)
        for world in range(world_count):
            scale = float(world + 1) / float(world_count)
            joint_f[world] = np.array([0.6, -0.3, 0.2, -0.4, 0.1], dtype=np.float32) * scale
        control.joint_f.assign(joint_f.reshape(-1))
        phoenx = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
        )
        featherstone = newton.solvers.SolverFeatherstone(model)
        dt = 1.0 / 240.0

        with wp.ScopedCapture(device=device) as capture_phoenx:
            phoenx.step(state_phoenx, output_phoenx, control, None, dt)
        wp.capture_launch(capture_phoenx.graph)
        with wp.ScopedCapture(device=device) as capture_featherstone:
            featherstone.step(state_featherstone, output_featherstone, control, None, dt)
        wp.capture_launch(capture_featherstone.graph)

        np.testing.assert_allclose(
            output_phoenx.joint_qd.numpy(), output_featherstone.joint_qd.numpy(), rtol=4.0e-4, atol=4.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.joint_q.numpy(), output_featherstone.joint_q.numpy(), rtol=4.0e-4, atol=4.0e-5
        )
        np.testing.assert_allclose(
            output_phoenx.body_q.numpy(), output_featherstone.body_q.numpy(), rtol=4.0e-4, atol=4.0e-5
        )

    def test_self_contact_conserves_spatial_momentum_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_self_contact_model(device)
        state = model.state()
        output = model.state()
        qd = state.joint_qd.numpy()
        qd[0] = 0.4
        qd[1] = -0.15
        qd[5] = 0.2
        qd[6] = -0.3
        qd[7] = 0.25
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=0,
        )
        contacts = model.contacts()
        momentum_before = _total_momentum(model, state)
        dt = 1.0 / 2000.0

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, dt)
        wp.capture_launch(capture.graph)

        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertFalse(solver._reduced_articulation.contact_block_system._skip_fallback_coloring)
        momentum_after = _total_momentum(model, output)
        np.testing.assert_allclose(momentum_after, momentum_before, rtol=0.0, atol=1.0e-5)

        fk_state = model.state()
        newton.eval_fk(model, output.joint_q, output.joint_qd, fk_state)
        np.testing.assert_allclose(output.body_qd.numpy(), fk_state.body_qd.numpy(), atol=2.0e-5)

    def test_reduced_contacts_honor_prepare_refresh_stride_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_grounded_articulation_cluster(device, worlds=1, articulations_per_world=1)
        state = model.state()
        output = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=3,
            solver_iterations=2,
            velocity_iterations=1,
            prepare_refresh_stride=2,
        )
        contacts = model.contacts()
        block = solver._reduced_articulation.contact_block_system
        original_solve = block.solve
        prepare_flags = []

        def record_solve(*args, **kwargs):
            prepare_flags.append(bool(kwargs["prepare"]))
            return original_solve(*args, **kwargs)

        block.solve = record_solve
        try:
            with wp.ScopedCapture(device=device) as capture:
                model.collide(state, contacts)
                solver.step(state, output, None, contacts, 1.0 / 240.0)
        finally:
            block.solve = original_solve
        wp.capture_launch(capture.graph)

        self.assertEqual(prepare_flags, [True, False, False, False, True, False])
        self.assertTrue(np.isfinite(output.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(output.joint_qd.numpy()).all())

    def test_multiple_grounded_articulations_per_world_use_grouped_schedule_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        worlds = 3
        articulations_per_world = 2
        model = _make_grounded_articulation_cluster(
            device,
            worlds=worlds,
            articulations_per_world=articulations_per_world,
        )
        state = model.state()
        output = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=2,
            solver_iterations=2,
            velocity_iterations=1,
        )
        contacts = model.contacts()

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 1000.0)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        expected = worlds * articulations_per_world
        self.assertEqual(model.world_count, worlds)
        self.assertEqual(model.articulation_count, expected)
        self.assertGreaterEqual(int(contacts.rigid_contact_count.numpy()[0]), expected)
        self.assertTrue(block._schedule_already_grouped)
        self.assertTrue(block._skip_fallback_coloring)
        self.assertEqual(int(np.count_nonzero(block.enabled.numpy())), expected)
        self.assertTrue(np.isfinite(output.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(output.joint_qd.numpy()).all())

    def test_maximal_rigid_contact_retains_fallback_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_mixed_reduced_maximal_ground_model(device)
        state = model.state()
        output = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=4,
            velocity_iterations=1,
        )
        contacts = model.contacts()

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 2000.0)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        self.assertGreaterEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
        self.assertFalse(block._skip_fallback_coloring)
        self.assertFalse(block._schedule_already_grouped)
        self.assertGreater(int(block.fallback_count.numpy()[0]), 0)
        self.assertTrue(np.isfinite(output.body_q.numpy()).all())
        self.assertTrue(np.isfinite(output.body_qd.numpy()).all())

    def test_contact_block_skips_deferred_fallback_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_contact_overflow_model(device, contact_count=8)
        state = model.state()
        output = model.state()
        qd = state.joint_qd.numpy()
        qd[2] = -0.2
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=1,
        )
        contacts = model.contacts()

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 2000.0)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        self.assertEqual(block.contact_dof_width, 48)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(int(block.enabled.numpy()[0]), 1)
        self.assertEqual(int(block.deferred_active.numpy()[0]), 0)
        self.assertTrue(block._skip_fallback_coloring)
        self.assertTrue(block._schedule_already_grouped)
        self.assertEqual(int(block.max_page_count.numpy()[0]), 1)
        self.assertEqual(int(block.page_index.numpy()[0]), 0)
        self.assertTrue(np.isfinite(output.joint_qd.numpy()).all())
        self.assertGreater(float(output.joint_qd.numpy()[2]), -0.2)

    def test_large_contact_articulation_uses_exact_fallback_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_large_contact_articulation(device)
        self.assertGreater(model.joint_dof_count, 64)
        state0 = model.state()
        state1 = model.state()
        qd = state0.joint_qd.numpy()
        qd[2] = -0.2
        state0.joint_qd.assign(qd)
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
        state1.joint_q.assign(state0.joint_q)
        state1.joint_qd.assign(state0.joint_qd)
        state1.body_q.assign(state0.body_q)
        state1.body_qd.assign(state0.body_qd)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=1,
        )
        contacts = model.contacts()
        dt = 1.0 / 2000.0
        with wp.ScopedCapture(device=device) as capture:
            model.collide(state0, contacts)
            solver.step(state0, state1, None, contacts, dt)
            model.collide(state1, contacts)
            solver.step(state1, state0, None, contacts, dt)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        self.assertEqual(block.contact_dof_width, 64)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(int(block.enabled.numpy()[0]), 0)
        self.assertEqual(int(block.deferred_active.numpy()[0]), 1)
        self.assertTrue(np.isfinite(state0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state0.joint_qd.numpy()).all())
        self.assertGreater(float(state0.joint_qd.numpy()[2]), -0.2)

        fk_state = model.state()
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, fk_state)
        np.testing.assert_allclose(state0.body_qd.numpy(), fk_state.body_qd.numpy(), atol=5.0e-5)

    def test_contact_block_pages_arbitrary_contact_count_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_contact_overflow_model(device, contact_count=2 * _POINTS_PER_PAGE + 8)
        state = model.state()
        output = model.state()
        qd = state.joint_qd.numpy()
        qd[2] = -0.2
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=1,
        )
        contacts = model.contacts()

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 2000.0)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        column_count = int(solver.world._ingest_scratch.num_contact_columns.numpy()[0])
        self.assertGreaterEqual(column_count, 1)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 2 * _POINTS_PER_PAGE)
        self.assertEqual(int(block.enabled.numpy()[0]), 1)
        self.assertEqual(int(block.deferred_active.numpy()[0]), 0)
        self.assertGreater(int(block.max_page_count.numpy()[0]), _CACHED_PAGE_COUNT)
        self.assertEqual(int(block.overflow_page_active.numpy()[0]), 1)
        self.assertEqual(int(block.transpose_active.numpy()[0]), 1)
        self.assertEqual(int(block.fallback_count.numpy()[0]), 0)
        self.assertTrue(np.isfinite(output.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(output.joint_qd.numpy()).all())
        self.assertGreater(float(output.joint_qd.numpy()[2]), -0.2)

        fk_state = model.state()
        newton.eval_fk(model, output.joint_q, output.joint_qd, fk_state)
        np.testing.assert_allclose(output.body_qd.numpy(), fk_state.body_qd.numpy(), atol=2.0e-4)

        # The same graph must adapt its conditional page loop to changing
        # device-side contact counts without resizing or recapture.
        contact_q = state.joint_q.numpy()
        separated_q = contact_q.copy()
        separated_q[2] = 2.0
        state.joint_q.assign(separated_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.capture_launch(capture.graph)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(int(block.max_page_count.numpy()[0]), 0)
        self.assertEqual(int(block.multi_page_active.numpy()[0]), 0)
        self.assertEqual(int(block.overflow_page_active.numpy()[0]), 0)
        self.assertEqual(int(block.transpose_active.numpy()[0]), 0)

        state.joint_q.assign(contact_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.capture_launch(capture.graph)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 2 * _POINTS_PER_PAGE)
        self.assertGreater(int(block.max_page_count.numpy()[0]), _CACHED_PAGE_COUNT)
        self.assertEqual(int(block.overflow_page_active.numpy()[0]), 1)
        self.assertEqual(int(block.multi_page_active.numpy()[0]), 1)
        self.assertEqual(int(block.transpose_active.numpy()[0]), 1)

    def test_fused_contact_apply_matches_separate_path_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_contact_overflow_model(device)
        state_separate = model.state()
        state_fused = model.state()
        output_separate = model.state()
        output_fused = model.state()
        qd = state_separate.joint_qd.numpy()
        qd[2] = -0.2
        for state in (state_separate, state_fused):
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        separate = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=1,
        )
        fused = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=1,
        )
        separate._reduced_articulation.contact_block_system.fuse_apply = False
        contacts_separate = model.contacts()
        contacts_fused = model.contacts()
        dt = 1.0 / 2000.0

        with wp.ScopedCapture(device=device) as separate_capture:
            model.collide(state_separate, contacts_separate)
            separate.step(state_separate, output_separate, None, contacts_separate, dt)
        with wp.ScopedCapture(device=device) as fused_capture:
            model.collide(state_fused, contacts_fused)
            fused.step(state_fused, output_fused, None, contacts_fused, dt)
        wp.capture_launch(separate_capture.graph)
        wp.capture_launch(fused_capture.graph)

        self.assertGreater(int(contacts_fused.rigid_contact_count.numpy()[0]), _POINTS_PER_PAGE)
        fused_block = fused._reduced_articulation.contact_block_system
        self.assertEqual(int(fused_block.max_page_count.numpy()[0]), _CACHED_PAGE_COUNT)
        self.assertEqual(int(fused_block.overflow_page_active.numpy()[0]), 0)
        np.testing.assert_allclose(
            output_fused.joint_qd.numpy(),
            output_separate.joint_qd.numpy(),
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            output_fused.body_qd.numpy(),
            output_separate.body_qd.numpy(),
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            output_fused.joint_q.numpy(),
            output_separate.joint_q.numpy(),
            rtol=2.0e-6,
            atol=2.0e-6,
        )

    def test_velocity_only_relax_publish_matches_full_publish_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model = _make_grounded_articulation_cluster(device, worlds=3, articulations_per_world=2)
        state_reference = model.state()
        state_optimized = model.state()
        output_reference = model.state()
        output_optimized = model.state()
        newton.eval_fk(model, state_reference.joint_q, state_reference.joint_qd, state_reference)
        state_optimized.joint_q.assign(state_reference.joint_q)
        state_optimized.joint_qd.assign(state_reference.joint_qd)
        state_optimized.body_q.assign(state_reference.body_q)
        state_optimized.body_qd.assign(state_reference.body_qd)
        reference = newton.solvers.SolverPhoenX(
            model, articulation_mode="reduced", substeps=2, solver_iterations=2, velocity_iterations=1
        )
        optimized = newton.solvers.SolverPhoenX(
            model, articulation_mode="reduced", substeps=2, solver_iterations=2, velocity_iterations=1
        )
        reference_reduced = reference._reduced_articulation
        reference_reduced.finish_relax = lambda: reference_reduced._publish_state(0.0)
        contacts_reference = model.contacts()
        contacts_optimized = model.contacts()

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state_reference, contacts_reference)
            reference.step(state_reference, output_reference, None, contacts_reference, 1.0 / 240.0)
            model.collide(state_optimized, contacts_optimized)
            optimized.step(state_optimized, output_optimized, None, contacts_optimized, 1.0 / 240.0)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(
            output_optimized.joint_q.numpy(), output_reference.joint_q.numpy(), rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            output_optimized.joint_qd.numpy(), output_reference.joint_qd.numpy(), rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(output_optimized.body_q.numpy(), output_reference.body_q.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            output_optimized.body_qd.numpy(), output_reference.body_qd.numpy(), rtol=0.0, atol=0.0
        )

    def test_dense_articulation_contacts_are_deterministic_and_conserve_momentum(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        articulation_count = 48
        model = _make_dense_articulation_contact_chain(device, articulation_count)
        state = model.state()
        output = model.state()
        qd = state.joint_qd.numpy().reshape(articulation_count, 6)
        qd[:, 0] = np.where(np.arange(articulation_count) % 2 == 0, 0.15, -0.15)
        state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=6,
            velocity_iterations=0,
        )
        solver._reduced_articulation.contact_block_system.fallback_worker_count = 1
        contacts = model.contacts()
        momentum_before = _total_momentum(model, state)
        dt = 1.0 / 2000.0

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, dt)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        fallback_count = int(block.fallback_count.numpy()[0])
        self.assertGreaterEqual(fallback_count, articulation_count - 1)
        first_columns = block.fallback_column.numpy()[:fallback_count].copy()
        first_colors = block.fallback_partitioner.interaction_id_to_partition.numpy()[:fallback_count].copy()
        momentum_after = _total_momentum(model, output)
        np.testing.assert_allclose(momentum_after, momentum_before, rtol=0.0, atol=1.0e-4)
        self.assertTrue(np.isfinite(output.joint_qd.numpy()).all())

        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(block.fallback_column.numpy()[:fallback_count], first_columns)
        np.testing.assert_array_equal(
            block.fallback_partitioner.interaction_id_to_partition.numpy()[:fallback_count], first_colors
        )

    def test_actual_cloth_contact_keeps_classic_and_reduced_ownership_disjoint(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model, collider = _make_contact_momentum_model(device, with_cloth=True, cloth_contact=True)
        state = model.state()
        output = model.state()
        q = state.joint_q.numpy()
        q[0] = -0.19
        q[6] = 1.0
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        body_q = state.body_q.numpy()
        body_q[collider, :3] = np.array([0.19, 0.0, 0.0], dtype=np.float32)
        body_q[collider, 3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        state.body_q.assign(body_q)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=4,
            velocity_iterations=0,
        )
        contacts = model.contacts()
        with wp.ScopedCapture(device=device) as capture:
            solver.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 2000.0)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        total = int(solver.world._ingest_scratch.num_contact_columns.numpy()[0])
        classic = int(block.classic_column_count.numpy()[0])
        reduced = int(block.reduced_column_count.numpy()[0])
        self.assertGreater(classic, 0)
        self.assertGreater(reduced, 0)
        self.assertEqual(classic + reduced, total)
        self.assertEqual(int(solver.world._num_active_constraints.numpy()[0]), solver.world._contact_offset + classic)
        self.assertTrue(np.isfinite(output.body_qd.numpy()).all())
        self.assertTrue(np.isfinite(output.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(output.particle_qd.numpy()).all())

    def test_mixed_cloth_does_not_double_solve_reduced_contacts(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        outputs = []
        solvers = []
        for with_cloth in (False, True):
            model, collider = _make_contact_momentum_model(device, with_cloth=with_cloth)
            state = model.state()
            output = model.state()
            q = state.joint_q.numpy()
            q[0] = -0.19
            q[6] = 1.0
            qd = np.zeros(model.joint_dof_count, dtype=np.float32)
            qd[0] = 0.5
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            body_q = state.body_q.numpy()
            body_q[collider, :3] = np.array([0.19, 0.0, 0.0], dtype=np.float32)
            body_q[collider, 3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            state.body_q.assign(body_q)
            body_qd = state.body_qd.numpy()
            body_qd[collider, 0] = -0.5
            state.body_qd.assign(body_qd)

            solver = newton.solvers.SolverPhoenX(
                model,
                articulation_mode="reduced",
                substeps=1,
                solver_iterations=8,
                velocity_iterations=0,
            )
            contacts = model.contacts()
            with wp.ScopedCapture(device=device) as capture:
                model.collide(state, contacts)
                solver.step(state, output, None, contacts, 1.0 / 2000.0)
            wp.capture_launch(capture.graph)
            outputs.append((output.joint_qd.numpy()[:7].copy(), output.body_qd.numpy()[:3].copy()))
            solvers.append(solver)

        self.assertFalse(solvers[0].world._regular_pgs_active_this_step)
        self.assertTrue(solvers[1].world._regular_pgs_active_this_step)
        np.testing.assert_allclose(outputs[1][0], outputs[0][0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(outputs[1][1], outputs[0][1], rtol=0.0, atol=2.0e-6)

    def test_high_degree_fallback_overflow_is_finite_and_conserves_momentum(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        satellite_count = 72
        model = _make_high_degree_articulation_star(device, satellite_count)
        state = model.state()
        output = model.state()
        qd = state.joint_qd.numpy().reshape(satellite_count + 1, 6)
        for index in range(satellite_count):
            angle = 2.0 * np.pi * float(index) / float(satellite_count)
            qd[index + 1, :2] = -0.005 * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=3,
            velocity_iterations=0,
        )
        contacts = model.contacts()
        momentum_before = _total_momentum(model, state)

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, 1.0 / 2000.0)
        wp.capture_launch(capture.graph)

        block = solver._reduced_articulation.contact_block_system
        self.assertGreaterEqual(int(block.fallback_count.numpy()[0]), satellite_count)
        self.assertGreater(int(block.fallback_partitioner.num_colors.numpy()[0]), 63)
        self.assertTrue(np.isfinite(output.joint_qd.numpy()).all())
        body_momenta = _body_momenta(model, output)
        momentum_after = np.sum(body_momenta, axis=0, dtype=np.float64)
        residual = momentum_after - momentum_before
        exchanged_momentum = np.linalg.norm(body_momenta[:, :3], axis=1).sum()
        exchanged_momentum += np.linalg.norm(body_momenta[:, 3:], axis=1).sum()
        self.assertLess(np.linalg.norm(residual), 1.0e-3 * exchanged_momentum + 1.0e-6)

    def test_contact_conserves_spatial_momentum_under_graph_capture(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("reduced articulation tests require CUDA graph capture")

        model, collider = _make_contact_momentum_model(device)
        state = model.state()
        output = model.state()
        q = state.joint_q.numpy()
        q[0] = -0.19
        q[6] = 1.0
        qd = np.zeros(model.joint_dof_count, dtype=np.float32)
        qd[0] = 0.5
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        body_q = state.body_q.numpy()
        body_q[collider, :3] = np.array([0.19, 0.0, 0.0], dtype=np.float32)
        body_q[collider, 3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        state.body_q.assign(body_q)
        body_qd = state.body_qd.numpy()
        body_qd[collider, 0] = -0.5
        state.body_qd.assign(body_qd)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode="reduced",
            substeps=1,
            solver_iterations=8,
            velocity_iterations=0,
        )
        contacts = model.contacts()
        momentum_before = _total_momentum(model, state)
        dt = 1.0 / 2000.0

        with wp.ScopedCapture(device=device) as capture:
            model.collide(state, contacts)
            solver.step(state, output, None, contacts, dt)
        wp.capture_launch(capture.graph)

        momentum_after = _total_momentum(model, output)
        np.testing.assert_allclose(momentum_after[:3], momentum_before[:3], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(momentum_after[3:], momentum_before[3:], rtol=0.0, atol=1.0e-5)

        fk_state = model.state()
        newton.eval_fk(model, output.joint_q, output.joint_qd, fk_state)
        np.testing.assert_allclose(output.body_qd.numpy()[:2], fk_state.body_qd.numpy()[:2], atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
