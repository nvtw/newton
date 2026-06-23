# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

"""Composable Warp reward/cost primitives for PhoenX RL environments.

The functions are intentionally small and sign-neutral. They return positive
rewards, positive penalties, or simple indicators; environment recipes decide
whether a term is added or subtracted by choosing the scale sign.
"""


@wp.func
def clamp01(x: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, wp.float32(0.0)), wp.float32(1.0))


@wp.func
def square(x: wp.float32) -> wp.float32:
    return x * x


@wp.func
def vec2_length_sq(x: wp.float32, y: wp.float32) -> wp.float32:
    return x * x + y * y


@wp.func
def gaussian_reward(error: wp.float32, sigma: wp.float32) -> wp.float32:
    sigma_sq = wp.max(sigma * sigma, wp.float32(1.0e-6))
    return wp.exp(-(error * error) / sigma_sq)


@wp.func
def gaussian_reward_sq(error_sq: wp.float32, sigma: wp.float32) -> wp.float32:
    sigma_sq = wp.max(sigma * sigma, wp.float32(1.0e-6))
    return wp.exp(-error_sq / sigma_sq)


@wp.func
def tracking_reward_2d(
    value_x: wp.float32,
    value_y: wp.float32,
    target_x: wp.float32,
    target_y: wp.float32,
    sigma: wp.float32,
) -> wp.float32:
    return gaussian_reward_sq(vec2_length_sq(value_x - target_x, value_y - target_y), sigma)


@wp.func
def height_tracking_reward(base_z: wp.float32, target_z: wp.float32, sigma: wp.float32) -> wp.float32:
    return gaussian_reward(base_z - target_z, sigma)


@wp.func
def command_progress_2d(vx: wp.float32, vy: wp.float32, cmd_x: wp.float32, cmd_y: wp.float32) -> wp.float32:
    return vx * cmd_x + vy * cmd_y


@wp.func
def command_progress_quality_2d(progress: wp.float32, cmd_x: wp.float32, cmd_y: wp.float32) -> wp.float32:
    speed_sq = vec2_length_sq(cmd_x, cmd_y)
    quality = wp.float32(1.0)
    if speed_sq > wp.float32(1.0e-6):
        quality = clamp01(progress / speed_sq)
    return quality


@wp.func
def projected_gravity_upright_reward(gravity_b: wp.vec3) -> wp.float32:
    return clamp01(-gravity_b[2])


@wp.func
def projected_gravity_flat_penalty(gravity_b: wp.vec3) -> wp.float32:
    return vec2_length_sq(gravity_b[0], gravity_b[1])


@wp.func
def radius_success_2d(delta_x: wp.float32, delta_y: wp.float32, radius: wp.float32) -> wp.float32:
    success = wp.float32(0.0)
    if vec2_length_sq(delta_x, delta_y) < radius * radius:
        success = wp.float32(1.0)
    return success


@wp.func
def fall_indicator(
    base_z: wp.float32, min_base_z: wp.float32, upright: wp.float32, min_upright: wp.float32
) -> wp.float32:
    fall = wp.float32(0.0)
    if base_z < min_base_z or upright < min_upright:
        fall = wp.float32(1.0)
    return fall


@wp.func
def progress_delta(previous_distance: wp.float32, current_distance: wp.float32) -> wp.float32:
    return previous_distance - current_distance


@wp.func
def action_rate_penalty(action: wp.float32, previous_action: wp.float32) -> wp.float32:
    return square(action - previous_action)


@wp.func
def joint_position_penalty(q: wp.float32, target_q: wp.float32) -> wp.float32:
    return square(q - target_q)


@wp.func
def pd_torque(target_q: wp.float32, q: wp.float32, qd: wp.float32, ke: wp.float32, kd: wp.float32) -> wp.float32:
    return ke * (target_q - q) - kd * qd


@wp.func
def abs_mechanical_power(torque: wp.float32, qd: wp.float32) -> wp.float32:
    return wp.abs(torque * qd)
