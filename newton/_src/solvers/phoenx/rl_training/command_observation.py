# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph-replay-safe velocity-command sampling for PhoenX RL tasks."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def sample_done_velocity_commands_kernel(
    seed_counter: wp.array[wp.int32],
    dones: wp.array[wp.float32],
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    standing_probability: wp.float32,
    command: wp.array2d[wp.float32],
):
    world = wp.tid()
    if dones[world] <= wp.float32(0.5):
        return
    rng = wp.rand_init(seed_counter[0], world * wp.int32(747796405) + wp.int32(915488749))
    command[world, 0] = x_min + (x_max - x_min) * wp.randf(rng)
    command[world, 1] = y_min + (y_max - y_min) * wp.randf(rng)
    command[world, 2] = yaw_min + (yaw_max - yaw_min) * wp.randf(rng)
    if wp.randf(rng) < wp.min(wp.max(standing_probability, wp.float32(0.0)), wp.float32(1.0)):
        command[world, 0] = wp.float32(0.0)
        command[world, 1] = wp.float32(0.0)
        command[world, 2] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def advance_command_seed_kernel(seed_counter: wp.array[wp.int32]):
    if wp.tid() == wp.int32(0):
        seed_counter[0] = seed_counter[0] + wp.int32(1)
