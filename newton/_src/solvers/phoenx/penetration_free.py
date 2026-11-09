# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX adapter for penetration-free rigid-rigid motion certification.

PhoenX advances maximal-coordinate rigid poses once per substep, in
:func:`_integrate_velocities_kernel`. The adapter certifies that update
against the half-space constraints and motion budgets produced by
:class:`~newton.CollisionPipeline` and commits only the admissible fraction of
each body's screw motion. Truncated bodies keep the TGS invariant
``x1 = x0 + v dt`` by scaling their velocities with the same fraction.

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
    PenetrationFreeData,
    launch_limit_motion,
    motion_budget_fraction,
    screw_motion_delta,
    screw_motion_interpolate,
    screw_motion_travel,
)
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    BodyContainer,
    sym6_from_mat33,
)
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia

__all__ = ["PhoenXPenetrationFree"]

# Relative pose tolerance for matching an imported state against the last committed pose.
_REFERENCE_POSITION_TOLERANCE = 1.0e-5
_REFERENCE_ROTATION_TOLERANCE = 1.0e-5


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


@wp.kernel(enable_backward=False)
def _commit_dynamic_kernel(
    bodies: BodyContainer,
    fraction: wp.array[wp.float32],
    status: wp.array[wp.int32],
    body_budget: wp.array[wp.float32],
    body_radius: wp.array[wp.float32],
    body_used: wp.array[wp.float32],
):
    """Commit the admissible fraction of each dynamic body's substep motion."""
    i = wp.tid()
    if i == 0 or bodies.motion_type[i] != MOTION_DYNAMIC or bodies.island_root[i] >= 0:
        return
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
        self.fraction = wp.ones(max(self.num_bodies, 1), dtype=wp.float32, device=device)

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

    def constrain_dynamic(self, data: PenetrationFreeData, bodies: BodyContainer) -> None:
        """Truncate the integrated substep motion of dynamic bodies."""
        if self.num_bodies <= 1 or data.body_count == 0:
            return
        self._limit(data, bodies)
        wp.launch(
            _commit_dynamic_kernel,
            dim=self.num_bodies,
            inputs=[bodies, self.fraction, data.status, data.body_budget, data.body_radius, data.body_used],
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
