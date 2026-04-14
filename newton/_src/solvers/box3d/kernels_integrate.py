# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Velocity and position integration kernels for the Box3D solver.

These are standard ``@wp.kernel`` launches (not tiled) operating on the
2-D ``[world, body]`` arrays.  They integrate velocities (gravity +
damping) and positions (semi-implicit Euler + quaternion integration)
each sub-step.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def integrate_velocities_2d(
    body_vel: wp.array2d(dtype=wp.vec3),
    body_ang_vel: wp.array2d(dtype=wp.vec3),
    body_inv_mass: wp.array2d(dtype=float),
    bodies_per_world: wp.array[wp.int32],
    gravity: wp.vec3,
    linear_damping: float,
    angular_damping: float,
    sub_dt: float,
):
    """Apply gravity and damping to velocities.

    Launched with ``dim = (num_worlds, max_bodies_per_world)``.
    """
    wid, tid = wp.tid()
    num_bodies = bodies_per_world[wid]
    if tid >= num_bodies:
        return

    im = body_inv_mass[wid, tid]
    if im <= 0.0:
        return  # static/kinematic

    v = body_vel[wid, tid]
    w = body_ang_vel[wid, tid]

    # Following Box2D: v_new = gravity_delta + damping * v_old
    # Gravity delta is NOT damped; only existing velocity is damped.
    ld = 1.0 / (1.0 + sub_dt * linear_damping)
    ad = 1.0 / (1.0 + sub_dt * angular_damping)
    v = gravity * sub_dt + v * ld
    w = w * ad

    body_vel[wid, tid] = v
    body_ang_vel[wid, tid] = w


@wp.func
def _quat_integrate(q: wp.quat, w: wp.vec3, dt: float) -> wp.quat:
    """Integrate a quaternion forward in time by angular velocity.

    Uses first-order quaternion integration:
    ``q' = normalize(q + 0.5 * dt * quat(w, 0) * q)``.
    """
    hw = w * (dt * 0.5)
    dq = wp.quat(
        hw[0] * q[3] + hw[1] * q[2] - hw[2] * q[1],
        -hw[0] * q[2] + hw[1] * q[3] + hw[2] * q[0],
        hw[0] * q[1] - hw[1] * q[0] + hw[2] * q[3],
        -hw[0] * q[0] - hw[1] * q[1] - hw[2] * q[2],
    )
    return wp.normalize(
        wp.quat(q[0] + dq[0], q[1] + dq[1], q[2] + dq[2], q[3] + dq[3])
    )


@wp.kernel
def update_world_inertia_2d(
    body_ori: wp.array2d(dtype=wp.quat),
    body_inv_mass: wp.array2d(dtype=float),
    body_inv_inertia_body: wp.array2d(dtype=wp.mat33),
    body_inv_inertia_world: wp.array2d(dtype=wp.mat33),
    bodies_per_world: wp.array[wp.int32],
):
    """Recompute world-frame inverse inertia from current orientation.

    ``I_world^{-1} = R @ I_body^{-1} @ R^T``

    Launched with ``dim = (num_worlds, max_bodies_per_world)``.
    """
    wid, tid = wp.tid()
    num_bodies = bodies_per_world[wid]
    if tid >= num_bodies:
        return
    im = body_inv_mass[wid, tid]
    if im <= 0.0:
        return
    ori = body_ori[wid, tid]
    R = wp.quat_to_matrix(ori)
    I_body_inv = body_inv_inertia_body[wid, tid]
    body_inv_inertia_world[wid, tid] = R * I_body_inv * wp.transpose(R)


@wp.kernel
def integrate_positions_2d(
    body_pos: wp.array2d(dtype=wp.vec3),
    body_ori: wp.array2d(dtype=wp.quat),
    body_vel: wp.array2d(dtype=wp.vec3),
    body_ang_vel: wp.array2d(dtype=wp.vec3),
    body_inv_mass: wp.array2d(dtype=float),
    body_delta_pos: wp.array2d(dtype=wp.vec3),
    bodies_per_world: wp.array[wp.int32],
    sub_dt: float,
):
    """Integrate positions and accumulate delta-position for substep separation updates.

    Launched with ``dim = (num_worlds, max_bodies_per_world)``.
    """
    wid, tid = wp.tid()
    num_bodies = bodies_per_world[wid]
    if tid >= num_bodies:
        return

    im = body_inv_mass[wid, tid]
    if im <= 0.0:
        return  # static/kinematic

    v = body_vel[wid, tid]
    w = body_ang_vel[wid, tid]

    body_pos[wid, tid] = body_pos[wid, tid] + v * sub_dt
    body_delta_pos[wid, tid] = body_delta_pos[wid, tid] + v * sub_dt
    body_ori[wid, tid] = _quat_integrate(body_ori[wid, tid], w, sub_dt)
