# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Where a step's rigid contacts lie while the bodies move within the step.

The collision pass finds each contact once per step, at the step's starting pose: a witness
point on each body and a normal. Substeps and relaxation sweeps then need the contact at the
current pose. Carrying each witness along as a point fixed in its body is wrong for curved
surfaces: a wheel that turns by ``theta`` lifts its witness by ``r (1 - cos theta)`` off the
ground and swings it behind the axle, so the contact reads as separating and its normal
impulse gains a lever about the axle. The rows here instead move the witnesses to first order
in each body's motion since the step started, ``dx + dtheta x r``. A body turning about its
own axis then moves its contact tangentially only, and pure rolling reads as no slip.

Both bodies take their impulse at one common point, so a contact conserves angular momentum.
That point follows the bodies' translation weighted by inverse mass: a body resting on a
static one keeps its lever arm, and a pair of equal bodies shares the motion.
"""

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer


@wp.kernel(enable_backward=False)
def snapshot_step_pose_kernel(bodies: BodyContainer):
    """Record every body's pose at the start of the step, where its contacts were found."""
    b = wp.tid()
    bodies.position_step[b] = bodies.position[b]
    bodies.orientation_step[b] = bodies.orientation[b]


@wp.func
def body_step_rotation(bodies: BodyContainer, b: wp.int32) -> wp.vec3f:
    """Rotation vector [rad] that turns body ``b`` from its step-start orientation to its current one."""
    # q_now * conj(q_step) for unit quaternions, written out so that equal orientations give
    # exactly no rotation.
    now = bodies.orientation[b]
    start = bodies.orientation_step[b]
    now_axis = wp.vec3f(now[0], now[1], now[2])
    start_axis = wp.vec3f(start[0], start[1], start[2])
    axis = start[3] * now_axis - now[3] * start_axis - wp.cross(now_axis, start_axis)
    cosine = now[3] * start[3] + wp.dot(now_axis, start_axis)
    if cosine < wp.float32(0.0):
        axis = -axis
    sine = wp.length(axis)
    if sine < wp.float32(1.0e-12):
        return wp.vec3f(0.0)
    return axis * (wp.float32(2.0) * wp.atan2(sine, wp.abs(cosine)) / sine)


@wp.func
def contact_point_weight(bodies: BodyContainer, b0: wp.int32, b1: wp.int32) -> wp.float32:
    """Share of body ``b0``'s translation in the motion of a contact's common impulse point."""
    inverse0 = bodies.inverse_mass[b0]
    inverse1 = bodies.inverse_mass[b1]
    if inverse0 + inverse1 <= wp.float32(0.0):
        return wp.float32(0.5)
    return inverse0 / (inverse0 + inverse1)


@wp.func
def rigid_contact_kinematics(
    bodies: BodyContainer,
    b0: wp.int32,
    b1: wp.int32,
    local0: wp.vec3f,
    local1: wp.vec3f,
    margin0: wp.float32,
    margin1: wp.float32,
    normal: wp.vec3f,
):
    """Return a contact's common impulse point and its separation vector at the current pose.

    ``local0`` and ``local1`` are the body-frame witnesses the collision pass found at the
    step's starting pose. The separation is witness 1 minus witness 0, each moved to first
    order since then; its normal component is the gap.
    """
    # Witness levers from each step-start COM, and the step-start midpoint's levers.
    lever0 = wp.quat_rotate(bodies.orientation_step[b0], local0 - bodies.body_com[b0]) + margin0 * normal
    lever1 = wp.quat_rotate(bodies.orientation_step[b1], local1 - bodies.body_com[b1]) - margin1 * normal
    start0 = bodies.position_step[b0]
    start1 = bodies.position_step[b1]
    half_span = wp.float32(0.5) * ((start1 + lever1) - (start0 + lever0))
    midpoint_lever0 = lever0 + half_span
    midpoint_lever1 = lever1 - half_span
    # Sums start from the current COMs, so no step-start position cancels in rounding.
    position0 = bodies.position[b0]
    position1 = bodies.position[b1]
    separation = (
        (position1 - position0)
        + (lever1 - lever0)
        + wp.cross(body_step_rotation(bodies, b1), midpoint_lever1)
        - wp.cross(body_step_rotation(bodies, b0), midpoint_lever0)
    )
    # The common point keeps the levers it had at the step start, weighted by inverse mass.
    weight0 = contact_point_weight(bodies, b0, b1)
    impulse_point = weight0 * (position0 + midpoint_lever0) + (wp.float32(1.0) - weight0) * (
        position1 + midpoint_lever1
    )
    return impulse_point, separation


@wp.func
def body_anchor_world(bodies: BodyContainer, b: wp.int32, anchor: wp.vec3f) -> wp.vec3f:
    """World position of the body-frame friction anchor ``anchor``, moved to first order since the step started."""
    lever = wp.quat_rotate(bodies.orientation_step[b], anchor - bodies.body_com[b])
    return bodies.position[b] + (lever + wp.cross(body_step_rotation(bodies, b), lever))


@wp.func
def body_anchor_local(bodies: BodyContainer, b: wp.int32, point: wp.vec3f) -> wp.vec3f:
    """Body-frame anchor whose :func:`body_anchor_world` is ``point`` (to first order)."""
    offset = point - bodies.position[b]
    lever = offset - wp.cross(body_step_rotation(bodies, b), offset)
    return bodies.body_com[b] + wp.quat_rotate_inv(bodies.orientation_step[b], lever)
