# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PhoenX shared solver kernels: integration, world-inertia update, graph-coloring helpers."""

from __future__ import annotations

import warp as wp

from .schemas import BODY_FLAG_STATIC

# ---------------------------------------------------------------------------
# Constants (matching C# PhoenX)
# ---------------------------------------------------------------------------

MAX_VELOCITY = wp.constant(100.0)


# ---------------------------------------------------------------------------
# Build elements for GraphColoring
# ---------------------------------------------------------------------------


@wp.kernel
def build_elements_kernel(
    body0: wp.array(dtype=wp.int32),
    body1: wp.array(dtype=wp.int32),
    elements: wp.array2d(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
):
    """Build the ``(N, 8)`` element array needed by :class:`GraphColoring`."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    elements[tid, 0] = body0[tid]
    elements[tid, 1] = body1[tid]
    for j in range(2, 8):
        elements[tid, j] = -1


@wp.kernel
def add_int32_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    """``out[0] = a[0] + b[0]`` on device."""
    out[0] = a[0] + b[0]


# ---------------------------------------------------------------------------
# Semi-implicit Euler integration
# ---------------------------------------------------------------------------


@wp.func
def _quat_integrate(q: wp.quat, w: wp.vec3, dt: float) -> wp.quat:
    """Integrate orientation by angular velocity (C# ``QuaternionIntegrationHelper``)."""
    angle = wp.length(w)
    if angle < 0.001:
        # Taylor expansion: sin(θ/2)/θ ≈ dt/2 - dt^3*θ^2/48
        scale = 0.5 * dt - dt * dt * dt * 0.020833333333 * angle * angle
    else:
        scale = wp.sin(0.5 * angle * dt) / angle
    ax = w * scale
    dq = wp.quat(ax[0], ax[1], ax[2], wp.cos(angle * dt * 0.5))
    return wp.normalize(dq * q)


@wp.kernel
def integrate_velocities_kernel(
    velocity: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=wp.float32),
    flags: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
    count: wp.array(dtype=wp.int32),
):
    """Apply gravity to linear velocities of dynamic bodies."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    if (flags[tid] & BODY_FLAG_STATIC) != 0:
        return
    if inverse_mass[tid] == 0.0:
        return
    velocity[tid] = velocity[tid] + gravity * dt


@wp.kernel
def apply_external_forces_kernel(
    velocity: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=wp.float32),
    inverse_inertia_world: wp.array(dtype=wp.mat33),
    flags: wp.array(dtype=wp.int32),
    ext_force: wp.array(dtype=wp.vec3),
    ext_torque: wp.array(dtype=wp.vec3),
    dt: float,
    count: wp.array(dtype=wp.int32),
):
    """Apply per-body external forces/torques (C# ``ApplyExternalForcesKernel``)."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    if (flags[tid] & BODY_FLAG_STATIC) != 0:
        return
    f = ext_force[tid]
    tau = ext_torque[tid]
    if wp.dot(f, f) == 0.0 and wp.dot(tau, tau) == 0.0:
        return
    inv_m = inverse_mass[tid]
    velocity[tid] = velocity[tid] + f * inv_m * dt
    angular_velocity[tid] = angular_velocity[tid] + inverse_inertia_world[tid] * tau * dt


@wp.kernel
def integrate_positions_kernel(
    position: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
    dt: float,
    count: wp.array(dtype=wp.int32),
):
    """Integrate positions and orientations using semi-implicit Euler.

    Velocity magnitude is clamped to :data:`MAX_VELOCITY` (100 m/s)
    matching the C# PhoenX ``LimitMagnitude`` safety guard.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    if (flags[tid] & BODY_FLAG_STATIC) != 0:
        return

    v = velocity[tid]
    w = angular_velocity[tid]

    # Clamp linear velocity magnitude (C# LimitMagnitude)
    speed = wp.length(v)
    if speed > MAX_VELOCITY:
        v = v * (MAX_VELOCITY / speed)
        velocity[tid] = v

    position[tid] = position[tid] + v * dt
    orientation[tid] = _quat_integrate(orientation[tid], w, dt)


# ---------------------------------------------------------------------------
# World-frame inverse inertia update
# ---------------------------------------------------------------------------


@wp.kernel
def update_world_inertia_kernel(
    orientation: wp.array(dtype=wp.quat),
    inv_inertia_local: wp.array(dtype=wp.mat33),
    inv_inertia_world: wp.array(dtype=wp.mat33),
    velocity: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    linear_damping: wp.array(dtype=wp.float32),
    angular_damping: wp.array(dtype=wp.float32),
    flags: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
):
    """Recompute world-frame inverse inertia and apply per-frame damping.

    Matches C# PhoenX ``UpdateInertiaKernel`` which applies velocity
    damping once per frame (not per substep).
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    q = orientation[tid]
    r = wp.quat_to_matrix(q)
    inv_inertia_world[tid] = r * inv_inertia_local[tid] * wp.transpose(r)

    if (flags[tid] & BODY_FLAG_STATIC) != 0:
        return
    velocity[tid] = velocity[tid] * linear_damping[tid]
    angular_velocity[tid] = angular_velocity[tid] * angular_damping[tid]
