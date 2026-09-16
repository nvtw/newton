# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
# Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
# Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause
"""FP32 semantic reference for unlocked, awake GPU TGS rigid-body motion.

Derived from NVIDIA PhysX ed6e5ca2474c9c80ad4f4826591b88476779c6ef,
solverMultiBlockTGS.cu:421-666 and integration.cuh:380-435.
NumPy trigonometry is not claimed bit-identical to CUDA __sincosf/FMA.
No production import, contact-law substitution, sleeping, or lock emulation.
"""

from dataclasses import dataclass

import numpy as np

F32 = np.float32


def require32(*arrays):
    """Reject implicit double-precision candidate inputs."""
    for array in arrays:
        assert np.asarray(array).dtype == np.float32


def quat_multiply(a, b):
    """Multiply xyzw quaternions in FP32, left world increment first."""
    require32(a, b)
    out = np.empty(4, dtype=np.float32)
    out[:3] = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    out[3] = a[3] * b[3] - np.dot(a[:3], b[:3])
    return out


def normalize(q):
    """Match the normalized quaternion state semantics without FP64 promotion."""
    require32(q)
    return q / np.sqrt(np.dot(q, q))


@dataclass
class Motion:
    """Separate contact predictors from the physical pose delta."""

    linear_delta: np.ndarray
    angular_delta: np.ndarray
    position_delta: np.ndarray
    rotation_delta: np.ndarray


def start_motion():
    """Reset at the beginning of one solver interval, never at each position pass."""
    return Motion(
        np.zeros(3, np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32), np.array([0, 0, 0, 1], np.float32)
    )


def advance_motion(
    motion, linear_velocity, angular_momocity, sqrt_inverse_inertia, step_dt, *, velocity_iteration=False
):
    """Advance after the averaged position-pass velocity; do nothing on velocity passes.

    ``angular_momocity`` is the actual GPU angular solver state u, not world omega.
    The supplied symmetric S is frozen from outer-step preparation; omega=S*u.
    Caller owns force integration, solve ordering, and copy averaging.
    """
    require32(linear_velocity, angular_momocity, sqrt_inverse_inertia, step_dt)
    if velocity_iteration:
        return
    omega = sqrt_inverse_inertia @ angular_momocity
    delta_p = linear_velocity * step_dt
    motion.linear_delta += delta_p
    motion.angular_delta += angular_momocity * step_dt
    w2 = np.dot(omega, omega)
    if w2 != F32(0):
        w = np.sqrt(w2)
        half_angle = w * F32(0.5) * step_dt
        s, c = np.sin(half_angle), np.cos(half_angle)
        s = s / w
        quaternion_velocity = np.empty(4, np.float32)
        quaternion_velocity[:3] = omega * s
        quaternion_velocity[3] = F32(0)
        result = quat_multiply(quaternion_velocity, motion.rotation_delta)
        result += motion.rotation_delta * c
        motion.rotation_delta = normalize(result)
    motion.position_delta += delta_p
    require32(motion.linear_delta, motion.angular_delta, motion.position_delta, motion.rotation_delta)


def writeback_pose(position_initial, orientation_initial, motion):
    """Awake integration writes the physical pose once after all solver passes."""
    require32(position_initial, orientation_initial)
    return position_initial + motion.position_delta, normalize(
        quat_multiply(motion.rotation_delta, orientation_initial)
    )


def contact_error(
    initial_error, direction, angular_row0, angular_row1, motion0, motion1, target_velocity, elapsed_time
):
    """GPU fixed-row predictor: angular rows must be S*(r cross direction).

    Angular deltas remain in momocity coordinates. This predictor is linearized;
    do not replace it by an exact finite-quaternion witness displacement.
    """
    require32(initial_error, direction, angular_row0, angular_row1, target_velocity, elapsed_time)
    relative_linear = motion0.linear_delta - motion1.linear_delta
    result = initial_error - target_velocity * elapsed_time
    result = result + np.dot(angular_row0, motion0.angular_delta)
    result = result - np.dot(angular_row1, motion1.angular_delta)
    result = result + np.dot(direction, relative_linear)
    assert result.dtype == np.float32
    return result
