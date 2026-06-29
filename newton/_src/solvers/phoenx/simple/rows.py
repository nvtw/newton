# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified scalar equation rows for the simple PhoenX Jacobi solver."""

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, mat33_from_sym6

__all__ = ["ScalarRowContainer", "scalar_row_container_zeros"]


@wp.struct
class ScalarRowContainer:
    """Structure-of-arrays storage for independent scalar equations.

    Every active row represents ``J v + bias + softness * lambda = 0``.
    Bounds apply to the accumulated multiplier. ``split_anchor`` marks one
    representative row per contact point or joint column for copy-free Tonge
    mass splitting. The solver deliberately stores no dense blocks, Schur complements, adjacency, or colour metadata.
    """

    active: wp.array[wp.int32]
    split_anchor: wp.array[wp.int32]
    body_a: wp.array[wp.int32]
    body_b: wp.array[wp.int32]
    jacobian_linear_a: wp.array[wp.vec3f]
    jacobian_angular_a: wp.array[wp.vec3f]
    jacobian_linear_b: wp.array[wp.vec3f]
    jacobian_angular_b: wp.array[wp.vec3f]
    bias: wp.array[wp.float32]
    softness: wp.array[wp.float32]
    lower: wp.array[wp.float32]
    upper: wp.array[wp.float32]
    bound_row: wp.array[wp.int32]
    bound_scale: wp.array[wp.float32]
    multiplier: wp.array[wp.float32]


def scalar_row_container_zeros(count: int, device: wp.DeviceLike = None) -> ScalarRowContainer:
    """Allocate ``count`` unified scalar rows."""
    rows = ScalarRowContainer()
    rows.active = wp.zeros(count, dtype=wp.int32, device=device)
    rows.split_anchor = wp.zeros(count, dtype=wp.int32, device=device)
    rows.body_a = wp.zeros(count, dtype=wp.int32, device=device)
    rows.body_b = wp.zeros(count, dtype=wp.int32, device=device)
    rows.jacobian_linear_a = wp.zeros(count, dtype=wp.vec3f, device=device)
    rows.jacobian_angular_a = wp.zeros(count, dtype=wp.vec3f, device=device)
    rows.jacobian_linear_b = wp.zeros(count, dtype=wp.vec3f, device=device)
    rows.jacobian_angular_b = wp.zeros(count, dtype=wp.vec3f, device=device)
    rows.bias = wp.zeros(count, dtype=wp.float32, device=device)
    rows.softness = wp.zeros(count, dtype=wp.float32, device=device)
    rows.lower = wp.full(count, -1.0e30, dtype=wp.float32, device=device)
    rows.upper = wp.full(count, 1.0e30, dtype=wp.float32, device=device)
    rows.bound_row = wp.full(count, -1, dtype=wp.int32, device=device)
    rows.bound_scale = wp.zeros(count, dtype=wp.float32, device=device)
    rows.multiplier = wp.zeros(count, dtype=wp.float32, device=device)
    return rows


@wp.kernel(enable_backward=False)
def clear_row_activity_kernel(rows: ScalarRowContainer):
    rows.active[wp.tid()] = wp.int32(0)


@wp.kernel(enable_backward=False)
def snapshot_body_velocities_kernel(
    bodies: BodyContainer,
    velocity_snapshot: wp.array[wp.vec3f],
    angular_velocity_snapshot: wp.array[wp.vec3f],
    delta_velocity: wp.array[wp.vec3f],
    delta_angular_velocity: wp.array[wp.vec3f],
):
    body = wp.tid()
    velocity_snapshot[body] = bodies.velocity[body]
    angular_velocity_snapshot[body] = bodies.angular_velocity[body]
    delta_velocity[body] = wp.vec3f(0.0)
    delta_angular_velocity[body] = wp.vec3f(0.0)


@wp.kernel(enable_backward=False)
def clear_body_split_counts_kernel(body_split_count: wp.array[wp.int32]):
    """Clear per-body Tonge partition counts before row assembly."""
    body_split_count[wp.tid()] = wp.int32(0)


@wp.kernel(enable_backward=False)
def count_body_split_incidence_kernel(
    rows: ScalarRowContainer,
    body_split_count: wp.array[wp.int32],
):
    """Count one split partition per active contact point or joint."""
    row = wp.tid()
    if rows.active[row] == wp.int32(0) or rows.split_anchor[row] == wp.int32(0):
        return
    body_a = rows.body_a[row]
    body_b = rows.body_b[row]
    wp.atomic_add(body_split_count, body_a, wp.int32(1))
    if body_b != body_a:
        wp.atomic_add(body_split_count, body_b, wp.int32(1))


@wp.kernel(enable_backward=False)
def snapshot_row_multipliers_kernel(
    rows: ScalarRowContainer,
    multiplier_snapshot: wp.array[wp.float32],
):
    row = wp.tid()
    multiplier_snapshot[row] = rows.multiplier[row]


@wp.kernel(enable_backward=False)
def solve_scalar_rows_jacobi_kernel(
    rows: ScalarRowContainer,
    bodies: BodyContainer,
    velocity_snapshot: wp.array[wp.vec3f],
    angular_velocity_snapshot: wp.array[wp.vec3f],
    multiplier_snapshot: wp.array[wp.float32],
    body_split_count: wp.array[wp.int32],
    relaxation: wp.float32,
    delta_velocity: wp.array[wp.vec3f],
    delta_angular_velocity: wp.array[wp.vec3f],
):
    row = wp.tid()
    if rows.active[row] == wp.int32(0):
        return

    a = rows.body_a[row]
    b = rows.body_b[row]
    jva = rows.jacobian_linear_a[row]
    jwa = rows.jacobian_angular_a[row]
    jvb = rows.jacobian_linear_b[row]
    jwb = rows.jacobian_angular_b[row]

    # Each partition uses split mass M/N (inverse mass N/M). The
    # post-atomic average below divides its provisional velocity by N.
    split_a = wp.float32(wp.max(body_split_count[a], wp.int32(1)))
    split_b = wp.float32(wp.max(body_split_count[b], wp.int32(1)))
    inv_mass_a = bodies.inverse_mass[a] * split_a
    inv_mass_b = bodies.inverse_mass[b] * split_b
    inv_inertia_a = mat33_from_sym6(bodies.inverse_inertia_world[a]) * split_a
    inv_inertia_b = mat33_from_sym6(bodies.inverse_inertia_world[b]) * split_b
    angular_response_a = inv_inertia_a @ jwa
    angular_response_b = inv_inertia_b @ jwb
    diagonal = (
        inv_mass_a * wp.dot(jva, jva)
        + wp.dot(jwa, angular_response_a)
        + inv_mass_b * wp.dot(jvb, jvb)
        + wp.dot(jwb, angular_response_b)
        + rows.softness[row]
    )
    if diagonal <= wp.float32(1.0e-12):
        return

    jv = (
        wp.dot(jva, velocity_snapshot[a])
        + wp.dot(jwa, angular_velocity_snapshot[a])
        + wp.dot(jvb, velocity_snapshot[b])
        + wp.dot(jwb, angular_velocity_snapshot[b])
    )
    old_multiplier = multiplier_snapshot[row]
    bound = rows.bound_scale[row] * wp.max(multiplier_snapshot[rows.bound_row[row]], wp.float32(0.0))
    lower = rows.lower[row] - bound
    upper = rows.upper[row] + bound
    delta_multiplier = -(jv + rows.bias[row] + rows.softness[row] * old_multiplier) / diagonal
    new_multiplier = wp.clamp(
        old_multiplier + relaxation * delta_multiplier,
        lower,
        upper,
    )
    applied_multiplier = new_multiplier - old_multiplier
    rows.multiplier[row] = new_multiplier

    # The atomics target a separate delta buffer. Thus all rows observe the
    # same snapshot and the sweep remains Jacobi regardless of launch order.
    wp.atomic_add(delta_velocity, a, inv_mass_a * jva * applied_multiplier)
    wp.atomic_add(delta_angular_velocity, a, angular_response_a * applied_multiplier)
    wp.atomic_add(delta_velocity, b, inv_mass_b * jvb * applied_multiplier)
    wp.atomic_add(delta_angular_velocity, b, angular_response_b * applied_multiplier)


@wp.kernel(enable_backward=False)
def apply_body_velocity_deltas_kernel(
    bodies: BodyContainer,
    body_split_count: wp.array[wp.int32],
    delta_velocity: wp.array[wp.vec3f],
    delta_angular_velocity: wp.array[wp.vec3f],
):
    # Averaging the implicit partition copies recovers one physical body
    # state and preserves each row's equal-and-opposite impulse.
    body = wp.tid()
    inv_split = wp.float32(1.0) / wp.float32(wp.max(body_split_count[body], wp.int32(1)))
    bodies.velocity[body] = bodies.velocity[body] + inv_split * delta_velocity[body]
    bodies.angular_velocity[body] = bodies.angular_velocity[body] + inv_split * delta_angular_velocity[body]
