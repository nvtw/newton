# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX adapter for penetration-free rigid-rigid motion certification.

PhoenX advances maximal-coordinate rigid poses once per substep, in
:func:`_integrate_velocities_kernel`. The adapter certifies that update
against the half-space constraints and motion budgets produced by
:class:`~newton.CollisionPipeline` and commits only the admissible fraction of
each body's screw motion. Truncated bodies keep the TGS invariant
``x1 = x0 + v dt`` by scaling their velocities with the same fraction.

Before integration, a velocity limiter removes the part of each dynamic
body's velocity that would carry a constrained feature past its limit, and
pushes features closer than :data:`LIMITER_RESERVE` of the tolerance to their
limit back out with a pseudo-velocity that is removed after integration.
Unlike truncation, it keeps tangential and rotational motion, so resting piles
slide and topple instead of jamming. Truncation stays the rigorous
certificate for whatever the limiter's linearization misses.

Kinematic bodies follow prescribed targets and cannot be truncated. Their
substep motion is checked after interpolation and violations are recorded in
the status mask.
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.penetration_free import (
    STATUS_BUDGET_EXHAUSTED,
    STATUS_FATAL_MASK,
    STATUS_KINEMATIC_VIOLATION,
    STATUS_REFERENCE_MISMATCH,
    Constraints,
    PenetrationFreeData,
    feature_support_point,
    launch_limit_motion,
    launch_place_limits,
    motion_budget_fraction,
    pair_count,
    screw_motion_delta,
    screw_motion_interpolate,
    screw_motion_travel,
    side_normal,
)
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    BodyContainer,
    mat33_from_sym6,
    sym6_from_mat33,
)
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia

__all__ = ["PhoenXPenetrationFree"]

# Relative pose tolerance for matching an imported state against the last committed pose.
_REFERENCE_POSITION_TOLERANCE = 1.0e-5
_REFERENCE_ROTATION_TOLERANCE = 1.0e-5

LIMITER_RESERVE = 0.1
"""Fraction of the tolerance the velocity limiter keeps between a feature and its limit."""

LIMITER_ITERATIONS = 4
"""Jacobi passes of the velocity limiter per substep."""


@wp.kernel(enable_backward=False)
def _reset_fraction_kernel(fraction: wp.array[wp.float32]):
    fraction[wp.tid()] = 1.0


@wp.kernel(enable_backward=False)
def _check_reference_kernel(
    body_q: wp.array[wp.transform],
    tracked_q: wp.array[wp.transform],
    motion_type: wp.array[wp.int32],
    # output
    status: wp.array[wp.int32],
):
    """Flag states whose non-kinematic poses differ from the last certified commit."""
    b = wp.tid()
    if motion_type[b + 1] == MOTION_KINEMATIC:
        return
    a = body_q[b]
    t = tracked_q[b]
    pa = wp.transform_get_translation(a)
    pt = wp.transform_get_translation(t)
    scale = 1.0 + wp.length(pt)
    if wp.length(pa - pt) > _REFERENCE_POSITION_TOLERANCE * scale:
        wp.atomic_or(status, 0, STATUS_REFERENCE_MISMATCH)
        return
    qa = wp.transform_get_rotation(a)
    qt = wp.transform_get_rotation(t)
    if wp.abs(wp.dot(qa, qt)) < 1.0 - _REFERENCE_ROTATION_TOLERANCE:
        wp.atomic_or(status, 0, STATUS_REFERENCE_MISMATCH)


@wp.func
def _limited_body(bodies: BodyContainer, slot: int) -> bool:
    return bodies.motion_type[slot] == MOTION_DYNAMIC and bodies.island_root[slot] < 0


@wp.kernel(enable_backward=False)
def _limit_velocity_kernel(
    constraints: Constraints,
    bodies: BodyContainer,
    dt: float,
    reserve: float,
    # outputs
    acc_v: wp.array[wp.vec3],
    acc_w: wp.array[wp.vec3],
    acc_pseudo_v: wp.array[wp.vec3],
    acc_pseudo_w: wp.array[wp.vec3],
    acc_count: wp.array[wp.int32],
):
    """Compute the normal impulses that keep a constraint's support point within its limit after one substep.

    The real impulse removes approach beyond the room left above
    ``limit - reserve``; the pseudo impulse moves a feature inside the
    reserve back out to ``limit - reserve``.
    """
    c, side = wp.tid()
    if c >= pair_count(constraints):
        return
    slot = constraints.body[c][side] + 1
    if slot <= 0 or not _limited_body(bodies, slot):
        return
    q = bodies.orientation[slot]
    x = bodies.position[slot]
    w = bodies.angular_velocity[slot]
    n = side_normal(constraints, c, side)
    f = constraints.feature[c, side]
    inv_inertia = mat33_from_sym6(bodies.inverse_inertia_world[slot])
    inv_mass = bodies.inverse_mass[slot]
    # Limit the current support point and the one that becomes extremal at the
    # orientation predicted for the end of the substep; a resting face has many.
    q_end = wp.normalize(wp.quat_from_axis_angle(wp.normalize(w + wp.vec3(1.0e-12)), wp.length(w) * dt) * q)
    for k in range(2):
        q_support = q
        if k == 1:
            q_support = q_end
        r = wp.quat_rotate(q, feature_support_point(f, wp.quat_rotate_inv(q_support, n)))
        room = constraints.limit[c][side] - reserve - wp.dot(n, x + r)
        rn = wp.cross(r, n)
        mass = inv_mass + wp.dot(rn, inv_inertia * rn)
        if mass > 0.0:
            u = wp.dot(n, bodies.velocity[slot]) + wp.dot(rn, w)
            real = wp.max(u - wp.max(room, 0.0) / dt, 0.0) / mass
            pseudo = float(0.0)
            if room < 0.0:
                pseudo = wp.max(wp.min(u - real * mass, 0.0) - room / dt, 0.0) / mass
            if real > 0.0 or pseudo > 0.0:
                wp.atomic_add(acc_v, slot, -inv_mass * real * n)
                wp.atomic_add(acc_w, slot, -real * (inv_inertia * rn))
                wp.atomic_add(acc_pseudo_v, slot, -inv_mass * pseudo * n)
                wp.atomic_add(acc_pseudo_w, slot, -pseudo * (inv_inertia * rn))
                wp.atomic_add(acc_count, slot, 1)


@wp.kernel(enable_backward=False)
def _apply_velocity_limit_kernel(
    bodies: BodyContainer,
    acc_v: wp.array[wp.vec3],
    acc_w: wp.array[wp.vec3],
    acc_pseudo_v: wp.array[wp.vec3],
    acc_pseudo_w: wp.array[wp.vec3],
    acc_count: wp.array[wp.int32],
    # outputs
    pseudo_v: wp.array[wp.vec3],
    pseudo_w: wp.array[wp.vec3],
):
    """Apply the averaged limiter impulses and remember the pseudo part for removal after integration."""
    i = wp.tid()
    count = acc_count[i]
    if count == 0:
        return
    scale = 1.0 / float(count)
    dpv = acc_pseudo_v[i] * scale
    dpw = acc_pseudo_w[i] * scale
    bodies.velocity[i] = bodies.velocity[i] + acc_v[i] * scale + dpv
    bodies.angular_velocity[i] = bodies.angular_velocity[i] + acc_w[i] * scale + dpw
    pseudo_v[i] = pseudo_v[i] + dpv
    pseudo_w[i] = pseudo_w[i] + dpw
    acc_v[i] = wp.vec3(0.0)
    acc_w[i] = wp.vec3(0.0)
    acc_pseudo_v[i] = wp.vec3(0.0)
    acc_pseudo_w[i] = wp.vec3(0.0)
    acc_count[i] = 0


@wp.kernel(enable_backward=False)
def _commit_dynamic_kernel(
    bodies: BodyContainer,
    fraction: wp.array[wp.float32],
    status: wp.array[wp.int32],
    body_budget: wp.array[wp.float32],
    body_radius: wp.array[wp.float32],
    body_used: wp.array[wp.float32],
    pseudo_v: wp.array[wp.vec3],
    pseudo_w: wp.array[wp.vec3],
):
    """Commit the admissible fraction of each dynamic body's substep motion and drop limiter pseudo-velocity."""
    i = wp.tid()
    if i == 0 or not _limited_body(bodies, i):
        return
    bodies.velocity[i] = bodies.velocity[i] - pseudo_v[i]
    bodies.angular_velocity[i] = bodies.angular_velocity[i] - pseudo_w[i]
    pseudo_v[i] = wp.vec3(0.0)
    pseudo_w[i] = wp.vec3(0.0)
    b = i - 1
    x0 = bodies.position_prev_substep[i]
    q0 = bodies.orientation_prev_substep[i]
    dx, axis, angle = screw_motion_delta(x0, q0, bodies.position[i], bodies.orientation[i])
    travel = screw_motion_travel(dx, angle, body_radius[b])
    budget_fraction = motion_budget_fraction(body_used[b], body_budget[b], travel)
    alpha = wp.min(fraction[i], budget_fraction)
    if (status[0] & STATUS_FATAL_MASK) != 0:
        alpha = 0.0
    if alpha < 1.0:
        if budget_fraction < 1.0:
            wp.atomic_or(status, 0, STATUS_BUDGET_EXHAUSTED)
        x, q = screw_motion_interpolate(x0, q0, dx, axis, angle, alpha)
        bodies.position[i] = x
        bodies.orientation[i] = q
        # Only a constraint removes approach velocity. A budget cut keeps the
        # velocity so the next detection grants a budget matching the motion.
        velocity_scale = wp.min(fraction[i], 1.0)
        bodies.velocity[i] = bodies.velocity[i] * velocity_scale
        bodies.angular_velocity[i] = bodies.angular_velocity[i] * velocity_scale
        bodies.inverse_inertia_world[i] = sym6_from_mat33(
            rotate_inertia(wp.quat_to_matrix(q), bodies.inverse_inertia[i])
        )
    body_used[b] = body_used[b] + alpha * travel


@wp.kernel(enable_backward=False)
def _verify_kinematic_kernel(
    bodies: BodyContainer,
    fraction: wp.array[wp.float32],
    status: wp.array[wp.int32],
    body_budget: wp.array[wp.float32],
    body_radius: wp.array[wp.float32],
    body_used: wp.array[wp.float32],
):
    """Record kinematic substep motion and flag motion that violates a constraint or budget."""
    i = wp.tid()
    if i == 0 or bodies.motion_type[i] != MOTION_KINEMATIC:
        return
    b = i - 1
    dx, _axis, angle = screw_motion_delta(
        bodies.position_prev_substep[i],
        bodies.orientation_prev_substep[i],
        bodies.position[i],
        bodies.orientation[i],
    )
    used = body_used[b] + screw_motion_travel(dx, angle, body_radius[b])
    body_used[b] = used
    if fraction[i] < 1.0 or used > body_budget[b]:
        wp.atomic_or(status, 0, STATUS_KINEMATIC_VIOLATION)


class PhoenXPenetrationFree:
    """Per-world scratch and launches for penetration-free substep certification."""

    def __init__(self, num_bodies: int, device):
        self.device = device
        self.num_bodies = int(num_bodies)
        n = max(self.num_bodies, 1)
        self.fraction = wp.ones(n, dtype=wp.float32, device=device)
        self.pseudo_v = wp.zeros(n, dtype=wp.vec3, device=device)
        self.pseudo_w = wp.zeros(n, dtype=wp.vec3, device=device)
        self._accumulators = [
            wp.zeros(n, dtype=wp.vec3, device=device),
            wp.zeros(n, dtype=wp.vec3, device=device),
            wp.zeros(n, dtype=wp.vec3, device=device),
            wp.zeros(n, dtype=wp.vec3, device=device),
            wp.zeros(n, dtype=wp.int32, device=device),
        ]

    def check_reference(self, data: PenetrationFreeData, body_q: wp.array, bodies: BodyContainer) -> None:
        """Flag a state that does not continue from the last certified commit."""
        if data.body_count == 0:
            return
        wp.launch(
            _check_reference_kernel,
            dim=data.body_count,
            inputs=[body_q, data.body_tracked_q, bodies.motion_type],
            outputs=[data.status],
            device=self.device,
        )

    def track(self, data: PenetrationFreeData, body_q: wp.array) -> None:
        """Remember the exported poses as the start of the next certified step."""
        if data.body_count == 0:
            return
        wp.copy(data.body_tracked_q, body_q, count=data.body_count)

    def _limit(self, data: PenetrationFreeData, bodies: BodyContainer) -> None:
        wp.launch(_reset_fraction_kernel, dim=self.num_bodies, outputs=[self.fraction], device=self.device)
        launch_limit_motion(
            data,
            bodies.position_prev_substep,
            bodies.orientation_prev_substep,
            bodies.position,
            bodies.orientation,
            1,
            self.fraction,
        )

    def limit_velocities(self, data: PenetrationFreeData, bodies: BodyContainer, dt: float) -> None:
        """Re-place the pair limits, then remove velocity that would carry dynamic features past them."""
        if self.num_bodies <= 1 or data.body_count == 0:
            return
        launch_place_limits(data, bodies.position, bodies.orientation, bodies.velocity, bodies.angular_velocity, 1)
        reserve = LIMITER_RESERVE * float(data.config.tolerance)
        for _ in range(LIMITER_ITERATIONS):
            wp.launch(
                _limit_velocity_kernel,
                dim=(data.constraints.body.shape[0], 2),
                inputs=[data.constraints, bodies, float(dt), reserve],
                outputs=self._accumulators,
                device=self.device,
            )
            wp.launch(
                _apply_velocity_limit_kernel,
                dim=self.num_bodies,
                inputs=[bodies, *self._accumulators],
                outputs=[self.pseudo_v, self.pseudo_w],
                device=self.device,
            )

    def constrain_dynamic(self, data: PenetrationFreeData, bodies: BodyContainer) -> None:
        """Truncate the integrated substep motion of dynamic bodies."""
        if self.num_bodies <= 1 or data.body_count == 0:
            return
        self._limit(data, bodies)
        wp.launch(
            _commit_dynamic_kernel,
            dim=self.num_bodies,
            inputs=[
                bodies,
                self.fraction,
                data.status,
                data.body_budget,
                data.body_radius,
                data.body_used,
                self.pseudo_v,
                self.pseudo_w,
            ],
            device=self.device,
        )

    def verify_kinematic(self, data: PenetrationFreeData, bodies: BodyContainer) -> None:
        """Check the interpolated substep motion of kinematic bodies."""
        if self.num_bodies <= 1 or data.body_count == 0:
            return
        self._limit(data, bodies)
        wp.launch(
            _verify_kinematic_kernel,
            dim=self.num_bodies,
            inputs=[bodies, self.fraction, data.status, data.body_budget, data.body_radius, data.body_used],
            device=self.device,
        )
