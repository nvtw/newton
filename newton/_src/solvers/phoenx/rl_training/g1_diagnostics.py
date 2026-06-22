# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_normal_lambda,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
)

G1_FOOT_CONTACT_METRIC_COUNT = 0
G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE = 1
G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE = 2
G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM = 3
G1_FOOT_CONTACT_METRIC_SPECULATIVE_COUNT = 4
G1_FOOT_CONTACT_METRIC_TANGENT_BIAS = 5
G1_FOOT_CONTACT_METRIC_HIGH_TANGENT_RATIO_COUNT = 6
G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT = 7
G1_FOOT_CONTACT_METRIC_ACTIVE_TANGENT_COUNT = 8
G1_FOOT_CONTACT_METRIC_COUNT_TOTAL = 9


@wp.func
def _g1_add_foot_contact_metric(
    shape_id: wp.int32,
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    body_stride: wp.int32,
    left_foot_body: wp.int32,
    right_foot_body: wp.int32,
    normal_impulse: wp.float32,
    tangent_impulse: wp.float32,
    tangent_normal_ratio: wp.float32,
    speculative_count: wp.float32,
    tangent_bias: wp.float32,
    high_tangent_ratio_count: wp.float32,
    active_normal_count: wp.float32,
    active_tangent_count: wp.float32,
    foot_contact_metrics: wp.array3d[wp.float32],
):
    if shape_id < wp.int32(0):
        return
    body = shape_body[shape_id]
    if body < wp.int32(0):
        return

    world = shape_world[shape_id]
    if world < wp.int32(0):
        if body_stride <= wp.int32(0):
            return
        world = body / body_stride
    if world < wp.int32(0) or world >= foot_contact_metrics.shape[0]:
        return

    local_body = body
    if body_stride > wp.int32(0):
        local_body = body - world * body_stride

    foot = wp.int32(-1)
    if local_body == left_foot_body:
        foot = wp.int32(0)
    elif local_body == right_foot_body:
        foot = wp.int32(1)
    if foot < wp.int32(0):
        return

    wp.atomic_add(foot_contact_metrics, world, foot, wp.int32(G1_FOOT_CONTACT_METRIC_COUNT), wp.float32(1.0))
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE),
        normal_impulse,
    )
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE),
        tangent_impulse,
    )
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM),
        tangent_normal_ratio,
    )
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_SPECULATIVE_COUNT),
        speculative_count,
    )
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_TANGENT_BIAS),
        tangent_bias,
    )
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_HIGH_TANGENT_RATIO_COUNT),
        high_tangent_ratio_count,
    )
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT),
        active_normal_count,
    )
    wp.atomic_add(
        foot_contact_metrics,
        world,
        foot,
        wp.int32(G1_FOOT_CONTACT_METRIC_ACTIVE_TANGENT_COUNT),
        active_tangent_count,
    )


@wp.kernel(enable_backward=False)
def g1_scan_foot_contact_metrics_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    contact_container: ContactContainer,
    body_stride: wp.int32,
    left_foot_body: wp.int32,
    right_foot_body: wp.int32,
    high_tangent_ratio_threshold: wp.float32,
    foot_contact_metrics: wp.array3d[wp.float32],
):
    tid = wp.tid()
    count = rigid_contact_count[0]
    if count > rigid_contact_shape0.shape[0]:
        count = rigid_contact_shape0.shape[0]
    if tid >= count:
        return

    normal_impulse = wp.max(cc_get_normal_lambda(contact_container, tid), wp.float32(0.0))
    tangent1 = cc_get_tangent1_lambda(contact_container, tid)
    tangent2 = cc_get_tangent2_lambda(contact_container, tid)
    tangent_impulse = wp.sqrt(tangent1 * tangent1 + tangent2 * tangent2)
    tangent_normal_ratio = wp.float32(0.0)
    if normal_impulse > wp.float32(1.0e-8):
        tangent_normal_ratio = tangent_impulse / normal_impulse

    bias = cc_get_bias(contact_container, tid)
    bias_t1 = cc_get_bias_t1(contact_container, tid)
    bias_t2 = cc_get_bias_t2(contact_container, tid)
    tangent_bias = wp.sqrt(bias_t1 * bias_t1 + bias_t2 * bias_t2)

    speculative_count = wp.float32(0.0)
    if bias > wp.float32(1.0e-8):
        speculative_count = wp.float32(1.0)

    active_normal_count = wp.float32(0.0)
    if normal_impulse > wp.float32(1.0e-8):
        active_normal_count = wp.float32(1.0)

    active_tangent_count = wp.float32(0.0)
    if tangent_impulse > wp.float32(1.0e-8):
        active_tangent_count = wp.float32(1.0)

    high_tangent_ratio_count = wp.float32(0.0)
    if tangent_normal_ratio >= high_tangent_ratio_threshold and active_normal_count > wp.float32(0.0):
        high_tangent_ratio_count = wp.float32(1.0)

    _g1_add_foot_contact_metric(
        rigid_contact_shape0[tid],
        shape_body,
        shape_world,
        body_stride,
        left_foot_body,
        right_foot_body,
        normal_impulse,
        tangent_impulse,
        tangent_normal_ratio,
        speculative_count,
        tangent_bias,
        high_tangent_ratio_count,
        active_normal_count,
        active_tangent_count,
        foot_contact_metrics,
    )
    _g1_add_foot_contact_metric(
        rigid_contact_shape1[tid],
        shape_body,
        shape_world,
        body_stride,
        left_foot_body,
        right_foot_body,
        normal_impulse,
        tangent_impulse,
        tangent_normal_ratio,
        speculative_count,
        tangent_bias,
        high_tangent_ratio_count,
        active_normal_count,
        active_tangent_count,
        foot_contact_metrics,
    )


def scan_g1_foot_contact_metrics(
    env, foot_contact_metrics: wp.array3d[wp.float32], high_tangent_ratio_threshold: float = 0.5
) -> None:
    """Reduce current G1 foot contact count, impulse totals, and tangent/load ratios."""

    foot_contact_metrics.zero_()
    if not getattr(env, "_can_scan_foot_contacts", False) or int(env.contacts.rigid_contact_max) <= 0:
        return
    contact_container = getattr(env.solver.world, "_contact_container", None)
    if contact_container is None:
        return

    wp.launch(
        g1_scan_foot_contact_metrics_kernel,
        dim=int(env.contacts.rigid_contact_max),
        inputs=[
            env.contacts.rigid_contact_count,
            env.contacts.rigid_contact_shape0,
            env.contacts.rigid_contact_shape1,
            env.model.shape_body,
            env.model.shape_world,
            contact_container,
            env.body_stride,
            env._left_foot_body_local,
            env._right_foot_body_local,
            float(high_tangent_ratio_threshold),
        ],
        outputs=[foot_contact_metrics],
        device=env.device,
    )
