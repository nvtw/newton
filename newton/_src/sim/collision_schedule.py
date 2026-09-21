# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-selected collision scheduling over a fixed simulation substep count."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence

import numpy as np
import warp as wp

from .model import Model
from .state import State

__all__ = ["CollisionSubstepScheduler"]


@wp.kernel(enable_backward=False)
def _select_collision_interval(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_collision_radius: wp.array[wp.float32],
    substep_dt: float,
    travel_budget: float,
    collision_intervals: wp.array[wp.int32],
    interval_conditions: wp.array[wp.int32],
    interval_overflow: wp.array[wp.int32],
):
    """Select the longest safe interval using a global two-shape speed bound."""
    shape_id = wp.tid()
    body_id = shape_body[shape_id]
    if body_id < 0:
        return

    X_wb = body_q[body_id]
    X_ws = wp.transform_multiply(X_wb, shape_transform[shape_id])
    shape_origin_world = wp.transform_get_translation(X_ws)
    com_world = wp.transform_point(X_wb, body_com[body_id])
    twist = body_qd[body_id]
    angular_velocity = wp.spatial_bottom(twist)
    shape_origin_velocity = wp.spatial_top(twist) + wp.cross(angular_velocity, shape_origin_world - com_world)

    furthest = wp.max(wp.abs(shape_collision_aabb_lower[shape_id]), wp.abs(shape_collision_aabb_upper[shape_id]))
    angular_radius = wp.max(wp.length(furthest), shape_collision_radius[shape_id])
    point_speed_bound = wp.length(shape_origin_velocity) + wp.length(angular_velocity) * angular_radius
    relative_travel_per_substep = 2.0 * point_speed_bound * substep_dt

    interval_count = collision_intervals.shape[0]
    selected_interval_index = int(0)
    for interval_index in range(interval_count):
        if float(collision_intervals[interval_index]) * relative_travel_per_substep <= travel_budget:
            selected_interval_index = interval_index
    for interval_index in range(selected_interval_index, interval_count):
        wp.atomic_max(interval_conditions, interval_index, 1)
    if relative_travel_per_substep > travel_budget:
        wp.atomic_max(interval_overflow, 0, 1)


class CollisionSubstepScheduler:
    """Schedule collision refreshes over a fixed number of simulation substeps.

    The scheduler always invokes ``substep_callback`` exactly ``substeps``
    times per frame. It invokes ``collision_callback`` before substep zero and
    periodically thereafter, choosing the largest safe interval from the
    divisors of ``substeps``. The selection stays on the device and uses
    :func:`warp.capture_if`, so it is compatible with CUDA graph capture.

    :meth:`step` works both directly and during CUDA graph capture. When an
    outer capture is active, the complete fixed schedule is recorded inline.
    CUDA graph conditional nodes require CUDA 12.4 or newer.

    The two states are used as ping-pong buffers. ``substeps`` must be even so
    every frame begins and ends in ``states[0]``. Callbacks must be capture-safe
    and must preallocate all storage before the first call.

    .. experimental::

        This scheduling API may change without notice.

    Args:
        model: Simulation model whose rigid-shape velocities determine the schedule.
        states: Two simulation states used as input/output ping-pong buffers.
        collision_callback: Called as ``callback(state, collision_dt)`` before
            each scheduled collision refresh. ``collision_dt`` is the time
            until the next refresh [s].
        substep_callback: Called as ``callback(state_in, state_out, dt)`` exactly
            ``substeps`` times per frame.
        frame_dt: Fixed frame duration [s].
        substeps: Fixed positive even number of solver substeps per frame.
        speculative_contact_gap_max: Pair-level speculative extension cap [m].
        pair_gap_lower_bound: Conservative lower bound on every possible
            colliding pair's authored gap sum [m]. Use ``0.0`` when unknown.
    """

    def __init__(
        self,
        model: Model,
        states: Sequence[State],
        *,
        collision_callback: Callable[[State, float], None],
        substep_callback: Callable[[State, State, float], None],
        frame_dt: float,
        substeps: int,
        speculative_contact_gap_max: float,
        pair_gap_lower_bound: float = 0.0,
    ):
        if len(states) != 2:
            raise ValueError(f"states must contain exactly two entries, got {len(states)}")
        if not callable(collision_callback) or not callable(substep_callback):
            raise TypeError("collision_callback and substep_callback must be callable")
        if not np.isfinite(frame_dt) or frame_dt <= 0.0:
            raise ValueError(f"frame_dt must be a positive finite number, got {frame_dt!r}")
        if isinstance(substeps, bool):
            raise TypeError("substeps must be an integer")
        try:
            substeps = operator.index(substeps)
        except TypeError as error:
            raise TypeError("substeps must be an integer") from error
        if substeps <= 0 or substeps % 2 != 0:
            raise ValueError(f"substeps must be a positive even integer, got {substeps}")
        if not np.isfinite(speculative_contact_gap_max) or speculative_contact_gap_max < 0.0:
            raise ValueError(
                f"speculative_contact_gap_max must be a non-negative finite number, got {speculative_contact_gap_max!r}"
            )
        if not np.isfinite(pair_gap_lower_bound) or pair_gap_lower_bound < 0.0:
            raise ValueError(f"pair_gap_lower_bound must be a non-negative finite number, got {pair_gap_lower_bound!r}")

        travel_budget = max(float(pair_gap_lower_bound), float(speculative_contact_gap_max))
        if travel_budget <= 0.0:
            raise ValueError("adaptive collision scheduling requires a positive speculative extension or gap bound")
        state_tuple = tuple(states)
        for index, state in enumerate(state_tuple):
            if state.body_q is None or state.body_qd is None:
                raise ValueError(f"states[{index}] must contain body_q and body_qd")
            if state.body_q.device != model.device or state.body_qd.device != model.device:
                raise ValueError(f"states[{index}] must be allocated on the model device")

        self.model = model
        self.states = state_tuple
        self.collision_callback = collision_callback
        self.substep_callback = substep_callback
        self.frame_dt = float(frame_dt)
        self.substeps = substeps
        self.substep_dt = self.frame_dt / self.substeps
        self.travel_budget = travel_budget
        self.collision_intervals = tuple(value for value in range(1, substeps + 1) if substeps % value == 0)

        device = model.device
        self._interval_values = wp.array(self.collision_intervals, dtype=wp.int32, device=device)
        self._interval_conditions = wp.zeros(len(self.collision_intervals), dtype=wp.int32, device=device)
        self._interval_condition_views = [
            self._interval_conditions[index : index + 1] for index in range(len(self.collision_intervals))
        ]
        self.interval_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        """Device scalar set to one when collision every substep is still insufficient."""

    def _dispatch_collision(self, substep_index: int, interval_index: int = 0) -> None:
        interval = self.collision_intervals[interval_index]

        def run_collision():
            if substep_index % interval == 0:
                self.collision_callback(self.states[substep_index % 2], interval * self.substep_dt)

        if interval_index == len(self.collision_intervals) - 1:
            run_collision()
            return
        wp.capture_if(
            self._interval_condition_views[interval_index],
            on_true=run_collision,
            on_false=lambda: self._dispatch_collision(substep_index, interval_index + 1),
        )

    def step(self) -> None:
        """Execute or record one complete fixed-substep frame."""
        self._interval_conditions.zero_()
        self.interval_overflow.zero_()
        model = self.model
        state = self.states[0]
        wp.launch(
            _select_collision_interval,
            dim=model.shape_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.shape_body,
                model.shape_transform,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                model.shape_collision_radius,
                self.substep_dt,
                self.travel_budget,
                self._interval_values,
                self._interval_conditions,
                self.interval_overflow,
            ],
            device=model.device,
        )
        for substep_index in range(self.substeps):
            self._dispatch_collision(substep_index)
            state_in = substep_index % 2
            self.substep_callback(self.states[state_in], self.states[1 - state_in], self.substep_dt)
