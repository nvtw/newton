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

LIMITER_ITERATIONS = 8
"""Passes of the velocity limiter per substep; each applies the most needed correction of every body."""


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


@wp.func
def _limiter_row(constraints: Constraints, bodies: BodyContainer, c: int, side: int, k: int, dt: float, reserve: float):
    """Return ``(real, pseudo, n, rn)`` normal impulses for one support point of one pair side.

    Support point ``k = 0`` is extremal at the current orientation and ``k = 1``
    at the orientation predicted for the end of the substep; a resting face
    has many extremal points. The real impulse removes approach beyond the
    room left above ``limit - reserve``; the pseudo impulse moves a feature
    inside the reserve back out to ``limit - reserve``.
    """
    slot = constraints.body[c][side] + 1
    q = bodies.orientation[slot]
    w = bodies.angular_velocity[slot]
    n = side_normal(constraints, c, side)
    q_support = q
    if k == 1:
        q_support = wp.normalize(wp.quat_from_axis_angle(wp.normalize(w + wp.vec3(1.0e-12)), wp.length(w) * dt) * q)
    r = wp.quat_rotate(q, feature_support_point(constraints.feature[c, side], wp.quat_rotate_inv(q_support, n)))
    room = constraints.limit[c][side] - reserve - wp.dot(n, bodies.position[slot] + r)
    rn = wp.cross(r, n)
    mass = bodies.inverse_mass[slot] + wp.dot(rn, mat33_from_sym6(bodies.inverse_inertia_world[slot]) * rn)
    real = float(0.0)
    pseudo = float(0.0)
    if mass > 0.0:
        u = wp.dot(n, bodies.velocity[slot]) + wp.dot(rn, w)
        real = wp.max(u - wp.max(room, 0.0) / dt, 0.0) / mass
        if room < 0.0:
            pseudo = wp.max(wp.min(u - real * mass, 0.0) - room / dt, 0.0) / mass
    return real, pseudo, n, rn, mass


@wp.func
def _limiter_key(constraints: Constraints, bodies: BodyContainer, c: int, side: int, k: int, dt: float, reserve: float):
    """Return a per-body unique key ordering rows by the velocity correction they need, or 0 if none."""
    real, pseudo, _n, _rn, mass = _limiter_row(constraints, bodies, c, side, k, dt, reserve)
    correction = (real + pseudo) * mass
    if correction <= 0.0:
        return wp.uint64(0)
    level = wp.uint64(wp.min(correction * 1.0e7, 2.0e9)) + wp.uint64(1)
    return (level << wp.uint64(32)) | wp.uint64((c * 2 + side) * 2 + k)


@wp.func
def _limiter_target(constraints: Constraints, bodies: BodyContainer, c: int, side: int) -> int:
    if c >= pair_count(constraints):
        return 0
    slot = constraints.body[c][side] + 1
    if slot <= 0 or not _limited_body(bodies, slot):
        return 0
    return slot


@wp.kernel(enable_backward=False)
def _select_limiter_row_kernel(
    constraints: Constraints,
    bodies: BodyContainer,
    dt: float,
    reserve: float,
    # output
    best: wp.array[wp.uint64],
):
    """Find each body's row needing the largest velocity correction."""
    c, side, k = wp.tid()
    slot = _limiter_target(constraints, bodies, c, side)
    if slot == 0:
        return
    key = _limiter_key(constraints, bodies, c, side, k, dt, reserve)
    if key != wp.uint64(0):
        wp.atomic_max(best, slot, key)


@wp.kernel(enable_backward=False)
def _apply_limiter_row_kernel(
    constraints: Constraints,
    bodies: BodyContainer,
    dt: float,
    reserve: float,
    best: wp.array[wp.uint64],
    # outputs
    pseudo_v: wp.array[wp.vec3],
    pseudo_w: wp.array[wp.vec3],
):
    """Apply each body's selected row and clear the selection for the next pass."""
    slot = wp.tid()
    key = best[slot]
    if key == wp.uint64(0):
        return
    best[slot] = wp.uint64(0)
    row = int(key & wp.uint64(0xFFFFFFFF))
    c = row // 4
    side = (row // 2) % 2
    k = row % 2
    real, pseudo, n, rn, _mass = _limiter_row(constraints, bodies, c, side, k, dt, reserve)
    inv_inertia = mat33_from_sym6(bodies.inverse_inertia_world[slot])
    inv_mass = bodies.inverse_mass[slot]
    dpv = -inv_mass * pseudo * n
    dpw = -pseudo * (inv_inertia * rn)
    bodies.velocity[slot] = bodies.velocity[slot] - inv_mass * real * n + dpv
    bodies.angular_velocity[slot] = bodies.angular_velocity[slot] - real * (inv_inertia * rn) + dpw
    pseudo_v[slot] = pseudo_v[slot] + dpv
    pseudo_w[slot] = pseudo_w[slot] + dpw


@wp.kernel(enable_backward=False)
def _commit_dynamic_kernel(
    bodies: BodyContainer,
    fraction: wp.array[wp.float32],
    status: wp.array[wp.int32],
    body_budget: wp.array[wp.float32],
    body_radius: wp.array[wp.float32],
    body_used: wp.array[wp.float32],
    body_blocked: wp.array[wp.int32],
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
    if (status[0] & STATUS_FATAL_MASK) != 0 or body_blocked[b] != 0:
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
        self._best_row = wp.zeros(n, dtype=wp.uint64, device=device)

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
        rows = (data.constraints.body.shape[0], 2, 2)
        inputs = [data.constraints, bodies, float(dt), reserve]
        self._best_row.zero_()
        for _ in range(LIMITER_ITERATIONS):
            wp.launch(
                _select_limiter_row_kernel,
                dim=rows,
                inputs=inputs,
                outputs=[self._best_row],
                device=self.device,
                max_blocks=data.max_blocks,
            )
            wp.launch(
                _apply_limiter_row_kernel,
                dim=self.num_bodies,
                inputs=[*inputs, self._best_row],
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
                data.constraints.body_blocked,
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
